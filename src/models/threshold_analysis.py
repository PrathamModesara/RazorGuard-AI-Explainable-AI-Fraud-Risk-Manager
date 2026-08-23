from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "xgboost_fraud_model.joblib"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "threshold"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "isFraud"
ID_COLUMN = "TransactionID"


def load_validation_data():
    """Load validation dataset."""

    print(
        "\nLoading validation dataset..."
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

    return X, y


def calculate_threshold_metrics(
    y_true,
    probabilities,
):
    """Calculate fraud metrics across thresholds."""

    thresholds = [
        round(
            threshold,
            2,
        )
        for threshold in [
            x / 100
            for x in range(
                5,
                96,
                5,
            )
        ]
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

        precision = (
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        )

        false_positive_rate = (
            fp / (fp + tn)
        )

        false_negative_rate = (
            fn / (fn + tp)
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positive_rate": (
                    false_positive_rate
                ),
                "false_negative_rate": (
                    false_negative_rate
                ),
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            }
        )

    return pd.DataFrame(
        results
    )


def plot_precision_recall(
    results,
):
    """Plot precision and recall against threshold."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results["threshold"],
        results["precision"],
        marker="o",
        label="Precision",
    )

    plt.plot(
        results["threshold"],
        results["recall"],
        marker="o",
        label="Recall",
    )

    plt.plot(
        results["threshold"],
        results["f1"],
        marker="o",
        label="F1",
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Score"
    )

    plt.title(
        "Precision, Recall and F1 by Threshold"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "threshold_metrics.png",
        dpi=150,
    )

    plt.close()


def plot_error_rates(
    results,
):
    """Plot false positive and false negative rates."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        results["threshold"],
        results["false_positive_rate"],
        marker="o",
        label="False Positive Rate",
    )

    plt.plot(
        results["threshold"],
        results["false_negative_rate"],
        marker="o",
        label="False Negative Rate",
    )

    plt.xlabel(
        "Decision Threshold"
    )

    plt.ylabel(
        "Rate"
    )

    plt.title(
        "False Positive and False Negative Rates"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "error_rates.png",
        dpi=150,
    )

    plt.close()


def main():
    print("=" * 70)

    print(
        "RazorGuard AI - Threshold Analysis"
    )

    print("=" * 70)

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

    X_validation, y_validation = (
        load_validation_data()
    )

    print(
        f"Validation rows: "
        f"{len(X_validation):,}"
    )

    # --------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------

    print(
        "\nGenerating fraud probabilities..."
    )

    probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    pr_auc = (
        average_precision_score(
            y_validation,
            probabilities,
        )
    )

    print(
        f"PR-AUC: "
        f"{pr_auc:.4f}"
    )

    # --------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------

    print(
        "\nCalculating threshold metrics..."
    )

    results = (
        calculate_threshold_metrics(
            y_validation,
            probabilities,
        )
    )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "threshold",
        "precision",
        "recall",
        "f1",
        "false_positive_rate",
        "false_negative_rate",
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
    # Best F1
    # --------------------------------------------------

    best_f1 = (
        results.loc[
            results["f1"].idxmax()
        ]
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BEST F1 THRESHOLD"
    )

    print(
        "=" * 70
    )

    print(
        f"Threshold: "
        f"{best_f1['threshold']:.2f}"
    )

    print(
        f"Precision: "
        f"{best_f1['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{best_f1['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{best_f1['f1']:.4f}"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    results_path = (
        REPORT_DIR
        / "threshold_results.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------
    # Create charts
    # --------------------------------------------------

    plot_precision_recall(
        results
    )

    plot_error_rates(
        results
    )

    print(
        "\nThreshold results saved to:"
    )

    print(
        results_path
    )

    print(
        "\nCharts saved to:"
    )

    print(
        REPORT_DIR
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "THRESHOLD ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()