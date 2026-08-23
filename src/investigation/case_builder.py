from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json

import joblib
import numpy as np
import pandas as pd

from src.risk.risk_analysis import RiskEngine


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_fraud_model.joblib"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "investigation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TARGET = "isFraud"
ID_COLUMN = "TransactionID"


# ============================================================
# INVESTIGATION CASE
# ============================================================

@dataclass
class InvestigationCase:
    transaction_id: int
    actual_label: int
    fraud_probability: float
    risk_score: float
    risk_level: str
    decision: str
    model_version: str
    threshold_version: str
    top_risk_factors: list[dict[str, Any]]
    top_risk_reducing_factors: list[dict[str, Any]]


# ============================================================
# FEATURE NAME CLEANING
# ============================================================

def get_readable_feature_name(
    feature_name: str,
) -> str:
    """
    Convert sklearn-generated feature names
    into easier-to-read feature names.

    Example:
        numerical__TransactionAmt
        ->
        TransactionAmt
    """

    if "__" in feature_name:
        return feature_name.split(
            "__",
            1,
        )[1]

    return feature_name


# ============================================================
# SHAP EVIDENCE
# ============================================================

def build_shap_evidence(
    pipeline,
    X_transaction: pd.DataFrame,
    top_n: int = 5,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    Generate transaction-level SHAP evidence.

    Returns:
        risk_increasing_factors
        risk_reducing_factors
    """

    print(
        "\nCalculating SHAP evidence..."
    )

    # --------------------------------------------------------
    # Extract preprocessing and XGBoost model
    # --------------------------------------------------------

    preprocessor = (
        pipeline.named_steps[
            "preprocessor"
        ]
    )

    xgb_model = (
        pipeline.named_steps[
            "model"
        ]
    )

    # --------------------------------------------------------
    # Transform transaction
    # --------------------------------------------------------

    transformed = (
        preprocessor.transform(
            X_transaction
        )
    )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------------
    # Import SHAP
    # --------------------------------------------------------

    import shap

    # --------------------------------------------------------
    # SHAP TreeExplainer
    # --------------------------------------------------------

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = (
        explainer.shap_values(
            transformed
        )
    )

    # --------------------------------------------------------
    # Extract SHAP row
    # --------------------------------------------------------

    shap_row = np.asarray(
        shap_values[0]
    )

    transformed_row = transformed[0]

    # --------------------------------------------------------
    # Convert sparse row if required
    # --------------------------------------------------------

    if hasattr(
        transformed_row,
        "toarray",
    ):

        transformed_row = (
            transformed_row
            .toarray()
            .ravel()
        )

    else:

        transformed_row = np.asarray(
            transformed_row
        ).ravel()

    # --------------------------------------------------------
    # Build evidence list
    # --------------------------------------------------------

    evidence = []

    for index, feature_name in enumerate(
        feature_names
    ):

        shap_value = float(
            shap_row[index]
        )

        feature_value = transformed_row[
            index
        ]

        # Ignore effectively zero contributions.
        if abs(shap_value) < 1e-9:
            continue

        evidence.append(
            {
                "feature": (
                    get_readable_feature_name(
                        feature_name
                    )
                ),
                "model_feature": feature_name,
                "feature_value": float(
                    feature_value
                ),
                "shap_value": round(
                    shap_value,
                    6,
                ),
            }
        )

    # --------------------------------------------------------
    # Positive SHAP = risk increasing
    # --------------------------------------------------------

    risk_increasing = sorted(
        [
            item
            for item in evidence
            if item["shap_value"] > 0
        ],
        key=lambda item: item[
            "shap_value"
        ],
        reverse=True,
    )[:top_n]

    # --------------------------------------------------------
    # Negative SHAP = risk reducing
    # --------------------------------------------------------

    risk_reducing = sorted(
        [
            item
            for item in evidence
            if item["shap_value"] < 0
        ],
        key=lambda item: item[
            "shap_value"
        ],
    )[:top_n]

    return (
        risk_increasing,
        risk_reducing,
    )


# ============================================================
# BUILD ONE INVESTIGATION CASE
# ============================================================

def build_investigation_case(
    pipeline,
    transaction: pd.Series,
    risk_engine: RiskEngine,
    model_version: str = "xgboost-v1",
) -> InvestigationCase:
    """
    Build a complete investigation case
    from one transaction.
    """

    # --------------------------------------------------------
    # Transaction information
    # --------------------------------------------------------

    transaction_id = int(
        transaction[
            ID_COLUMN
        ]
    )

    actual_label = int(
        transaction[
            TARGET
        ]
    )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    X_transaction = (
        transaction
        .drop(
            labels=[
                TARGET,
                ID_COLUMN,
            ]
        )
        .to_frame()
        .T
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    print(
        "\nGenerating fraud probability..."
    )

    fraud_probability = float(
        pipeline.predict_proba(
            X_transaction
        )[0, 1]
    )

    # --------------------------------------------------------
    # Risk engine
    # --------------------------------------------------------

    risk_decision = (
        risk_engine.evaluate(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
        )
    )

    # --------------------------------------------------------
    # SHAP evidence
    # --------------------------------------------------------

    (
        risk_factors,
        risk_reducing_factors,
    ) = build_shap_evidence(
        pipeline=pipeline,
        X_transaction=X_transaction,
        top_n=5,
    )

    # --------------------------------------------------------
    # Build case
    # --------------------------------------------------------

    case = InvestigationCase(
        transaction_id=transaction_id,
        actual_label=actual_label,
        fraud_probability=(
            risk_decision.fraud_probability
        ),
        risk_score=(
            risk_decision.risk_score
        ),
        risk_level=(
            risk_decision.risk_level
        ),
        decision=(
            risk_decision.decision
        ),
        model_version=model_version,
        threshold_version=(
            risk_decision.threshold_version
        ),
        top_risk_factors=risk_factors,
        top_risk_reducing_factors=(
            risk_reducing_factors
        ),
    )

    return case


# ============================================================
# DISPLAY CASE
# ============================================================

def display_case(
    case: InvestigationCase,
) -> None:
    """
    Display a human-readable investigation case.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "TRANSACTION RESULT"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTransaction ID: "
        f"{case.transaction_id}"
    )

    print(
        f"Actual Label: "
        f"{case.actual_label}"
    )

    print(
        f"Fraud Probability: "
        f"{case.fraud_probability:.6f}"
    )

    print(
        f"Risk Score: "
        f"{case.risk_score}"
    )

    print(
        f"Risk Level: "
        f"{case.risk_level}"
    )

    print(
        f"Decision: "
        f"{case.decision}"
    )

    print(
        f"Model Version: "
        f"{case.model_version}"
    )

    print(
        f"Threshold Version: "
        f"{case.threshold_version}"
    )

    # --------------------------------------------------------
    # Risk-increasing factors
    # --------------------------------------------------------

    print(
        "\nTop Risk-Increasing Factors:"
    )

    if case.top_risk_factors:

        for factor in (
            case.top_risk_factors
        ):

            print(
                f"  - "
                f"{factor['feature']}: "
                f"SHAP="
                f"{factor['shap_value']:.6f}"
            )

    else:

        print(
            "  None available."
        )

    # --------------------------------------------------------
    # Risk-reducing factors
    # --------------------------------------------------------

    print(
        "\nTop Risk-Reducing Factors:"
    )

    if case.top_risk_reducing_factors:

        for factor in (
            case.top_risk_reducing_factors
        ):

            print(
                f"  - "
                f"{factor['feature']}: "
                f"SHAP="
                f"{factor['shap_value']:.6f}"
            )

    else:

        print(
            "  None available."
        )


# ============================================================
# SAVE CASE
# ============================================================

def save_case(
    case: InvestigationCase,
) -> Path:
    """
    Save investigation case as JSON.
    """

    output_path = (
        OUTPUT_DIR
        / f"case_{case.transaction_id}.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            asdict(case),
            file,
            indent=4,
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "RazorGuard AI - Investigation Case Builder"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load XGBoost model
    # --------------------------------------------------------

    print(
        "\nLoading XGBoost model..."
    )

    pipeline = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    print(
        "Loading validation data..."
    )

    df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: "
        f"{len(df):,}"
    )

    # --------------------------------------------------------
    # Demo transactions
    # --------------------------------------------------------
    #
    # 3444522 = legitimate transaction
    # 3427568 = fraudulent transaction
    #
    # These were selected from the held-out
    # validation dataset.
    # --------------------------------------------------------

    demo_transactions = [
        3444522,
        3427568,
    ]

    # --------------------------------------------------------
    # Risk engine
    # --------------------------------------------------------

    risk_engine = RiskEngine(
        medium_threshold=0.30,
        high_threshold=0.80,
        threshold_version="v1",
    )

    # --------------------------------------------------------
    # Process each demo transaction
    # --------------------------------------------------------

    for transaction_id in (
        demo_transactions
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"Processing transaction "
            f"{transaction_id}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Find transaction
        # ----------------------------------------------------

        matching_rows = df[
            df[ID_COLUMN]
            == transaction_id
        ]

        if matching_rows.empty:

            print(
                f"\nERROR: Transaction "
                f"{transaction_id} "
                f"was not found."
            )

            continue

        transaction = (
            matching_rows.iloc[0]
        )

        # ----------------------------------------------------
        # Build investigation case
        # ----------------------------------------------------

        print(
            "\nBuilding investigation case..."
        )

        case = build_investigation_case(
            pipeline=pipeline,
            transaction=transaction,
            risk_engine=risk_engine,
            model_version="xgboost-v1",
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        display_case(
            case
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = save_case(
            case
        )

        print(
            "\nInvestigation case saved to:"
        )

        print(
            output_path
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL DEMO INVESTIGATION CASES COMPLETE"
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()