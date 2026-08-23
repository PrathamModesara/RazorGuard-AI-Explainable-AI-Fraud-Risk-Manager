from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cost"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "isFraud"
ID_COLUMN = "TransactionID"


# ------------------------------------------------------
# BUSINESS COST ASSUMPTIONS
# ------------------------------------------------------
#
# These are configurable scenario assumptions.
# They are NOT Razorpay's actual internal costs.
#
# FP = legitimate transaction incorrectly flagged
# FN = fraudulent transaction incorrectly approved
#

COST_FALSE_POSITIVE = 5.0
COST_FALSE_NEGATIVE = 100.0


def calculate_cost(
    false_positives: int,
    false_negatives: int,
) -> float:
    """
    Calculate total business cost.
    """

    return (
        false_positives
        * COST_FALSE_POSITIVE
        +
        false_negatives
        * COST_FALSE_NEGATIVE
    )


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate expected business cost across thresholds.
    """

    thresholds = [
        round(
            threshold,
            2,
        )
        for threshold in np.arange(
            0.05,
            1.00,
            0.05,
        )
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        tn, fp, fn, tp = (
            confusion_matrix(
                y_true,
                predictions,
                labels=[0, 1],
            ).ravel()
        )

        total_cost = calculate_cost(
            false_positives=fp,
            false_negatives=fn,
        )

        average_cost = (
            total_cost
            / len(y_true)
        )

        results.append(
            {
                "threshold": threshold,
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
                "total_cost": total_cost,
                "average_cost_per_transaction": (
                    average_cost
                ),
            }
        )

    return pd.DataFrame(
        results
    )


def main() -> None:

    print("=" * 70)

    print(
        "RazorGuard AI - False Positive Cost Analysis"
    )

    print("=" * 70)

    # --------------------------------------------------
    # Display assumptions
    # --------------------------------------------------

    print(
        "\nBUSINESS COST ASSUMPTIONS"
    )

    print(
        "-" * 50
    )

    print(
        f"False Positive Cost: "
        f"{COST_FALSE_POSITIVE}"
    )

    print(
        f"False Negative Cost: "
        f"{COST_FALSE_NEGATIVE}"
    )

    print(
        "\nThese are scenario assumptions for "
        "decision analysis."
    )

    print(
        "They are NOT Razorpay internal costs."
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print(
        "\nLoading XGBoost model..."
    )

    model = joblib.load(
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

    X = df.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    )

    y = df[TARGET]

    print(
        f"Validation rows: "
        f"{len(X):,}"
    )

    # --------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------

    print(
        "\nGenerating fraud probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X
        )[:, 1]
    )

    # --------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------

    print(
        "\nEvaluating thresholds..."
    )

    results = evaluate_thresholds(
        y,
        probabilities,
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD COST RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "threshold",
        "false_positives",
        "false_negatives",
        "total_cost",
        "average_cost_per_transaction",
    ]

    print(
        results[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # --------------------------------------------------
    # Best threshold
    # --------------------------------------------------

    best_row = results.loc[
        results[
            "total_cost"
        ].idxmin()
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "LOWEST-COST THRESHOLD"
    )

    print(
        "=" * 70
    )

    print(
        f"Threshold: "
        f"{best_row['threshold']:.2f}"
    )

    print(
        f"False Positives: "
        f"{int(best_row['false_positives']):,}"
    )

    print(
        f"False Negatives: "
        f"{int(best_row['false_negatives']):,}"
    )

    print(
        f"Total Cost: "
        f"{best_row['total_cost']:,.2f}"
    )

    print(
        f"Average Cost / Transaction: "
        f"{best_row['average_cost_per_transaction']:.6f}"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = (
        REPORT_DIR
        / "threshold_cost_results.csv"
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nResults saved to:"
    )

    print(
        output_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "COST ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()