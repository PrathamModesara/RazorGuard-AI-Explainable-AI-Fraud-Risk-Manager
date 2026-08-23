from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


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
    / "shap"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "isFraud"
ID_COLUMN = "TransactionID"


def get_feature_name(
    feature_name: str,
) -> str:
    """
    Convert sklearn-generated feature names
    into easier-to-read names.
    """

    if "__" in feature_name:
        return feature_name.split(
            "__",
            1,
        )[1]

    return feature_name


def main() -> None:

    print("=" * 70)
    print(
        "RazorGuard AI - Transaction Explainability"
    )
    print("=" * 70)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading XGBoost model...")

    pipeline = joblib.load(
        MODEL_PATH
    )

    # --------------------------------------------------
    # Load validation data
    # --------------------------------------------------

    print(
        "\nLoading validation data..."
    )

    df = pd.read_csv(
        VALIDATION_PATH
    )

    # --------------------------------------------------
    # Select one transaction
    # --------------------------------------------------

    transaction = df.iloc[
        [0]
    ].copy()

    transaction_id = int(
        transaction[
            ID_COLUMN
        ].iloc[0]
    )

    actual_label = int(
        transaction[
            TARGET
        ].iloc[0]
    )

    X_transaction = (
        transaction.drop(
            columns=[
                TARGET,
                ID_COLUMN,
            ]
        )
    )

    print(
        f"\nTransaction ID: "
        f"{transaction_id}"
    )

    print(
        f"Actual label: "
        f"{actual_label}"
    )

    # --------------------------------------------------
    # Fraud probability
    # --------------------------------------------------

    fraud_probability = float(
        pipeline.predict_proba(
            X_transaction
        )[0, 1]
    )

    print(
        f"Fraud probability: "
        f"{fraud_probability:.6f}"
    )

    # --------------------------------------------------
    # Transform transaction
    # --------------------------------------------------

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

    X_transformed = (
        preprocessor.transform(
            X_transaction
        )
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    # --------------------------------------------------
    # SHAP
    # --------------------------------------------------

    print(
        "\nCalculating SHAP explanation..."
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    shap_values = (
        explainer.shap_values(
            X_transformed
        )
    )

    # --------------------------------------------------
    # Extract transaction SHAP values
    # --------------------------------------------------

    shap_row = np.asarray(
        shap_values[0]
    )

    transformed_row = (
        X_transformed[0]
    )

    # Convert sparse matrix row
    # into a dense array when required.
    if hasattr(
        transformed_row,
        "toarray",
    ):
        transformed_row = (
            transformed_row.toarray()
            .ravel()
        )

    else:
        transformed_row = np.asarray(
            transformed_row
        ).ravel()

    explanation = pd.DataFrame(
        {
            "feature": feature_names,
            "value": transformed_row,
            "shap_value": shap_row,
        }
    )

    explanation[
        "abs_shap_value"
    ] = explanation[
        "shap_value"
    ].abs()

    explanation[
        "readable_feature"
    ] = explanation[
        "feature"
    ].apply(
        get_feature_name
    )

    # --------------------------------------------------
    # Top positive risk factors
    # --------------------------------------------------

    positive_factors = (
        explanation[
            explanation[
                "shap_value"
            ] > 0
        ]
        .sort_values(
            "shap_value",
            ascending=False,
        )
        .head(10)
    )

    # --------------------------------------------------
    # Top negative factors
    # --------------------------------------------------

    negative_factors = (
        explanation[
            explanation[
                "shap_value"
            ] < 0
        ]
        .sort_values(
            "shap_value",
            ascending=True,
        )
        .head(10)
    )

    # --------------------------------------------------
    # Print explanation
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP RISK-INCREASING FEATURES"
    )

    print(
        "=" * 70
    )

    for _, row in (
        positive_factors.iterrows()
    ):

        print(
            f"{row['readable_feature']}: "
            f"SHAP={row['shap_value']:.6f}"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP RISK-DECREASING FEATURES"
    )

    print(
        "=" * 70
    )

    for _, row in (
        negative_factors.iterrows()
    ):

        print(
            f"{row['readable_feature']}: "
            f"SHAP={row['shap_value']:.6f}"
        )

    # --------------------------------------------------
    # Save complete explanation
    # --------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"transaction_{transaction_id}_explanation.csv"
    )

    explanation.to_csv(
        output_path,
        index=False,
    )

    # --------------------------------------------------
    # Save summary
    # --------------------------------------------------

    summary = pd.DataFrame(
        {
            "transaction_id": [
                transaction_id
            ],
            "actual_label": [
                actual_label
            ],
            "fraud_probability": [
                fraud_probability
            ],
        }
    )

    summary_path = (
        OUTPUT_DIR
        / f"transaction_{transaction_id}_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "TRANSACTION EXPLANATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nExplanation saved to:"
    )

    print(
        output_path
    )

    print(
        f"\nSummary saved to:"
    )

    print(
        summary_path
    )


if __name__ == "__main__":
    main()