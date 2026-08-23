from pathlib import Path

import joblib
import matplotlib.pyplot as plt
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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "shap"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "isFraud"
ID_COLUMN = "TransactionID"


def main() -> None:

    print("=" * 70)
    print("RazorGuard AI - SHAP Explainability")
    print("=" * 70)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    print("\nLoading XGBoost pipeline...")

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

    X = df.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    )

    print(
        f"Validation rows: "
        f"{len(X):,}"
    )

    # --------------------------------------------------
    # Select sample
    # --------------------------------------------------

    sample_size = min(
        1000,
        len(X),
    )

    X_sample = X.iloc[
        :sample_size
    ].copy()

    print(
        f"SHAP sample size: "
        f"{len(X_sample):,}"
    )

    # --------------------------------------------------
    # Extract preprocessing and model
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

    # --------------------------------------------------
    # Transform features
    # --------------------------------------------------

    print(
        "\nTransforming features..."
    )

    X_transformed = (
        preprocessor.transform(
            X_sample
        )
    )

    print(
        f"Transformed shape: "
        f"{X_transformed.shape}"
    )

    # --------------------------------------------------
    # Feature names
    # --------------------------------------------------

    print(
        "\nExtracting feature names..."
    )

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    print(
        f"Number of feature names: "
        f"{len(feature_names):,}"
    )

    # --------------------------------------------------
    # SHAP explainer
    # --------------------------------------------------

    print(
        "\nCreating SHAP TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    # --------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X_transformed
    )

    print(
        "SHAP calculation complete."
    )

    # --------------------------------------------------
    # Global feature importance
    # --------------------------------------------------

    print(
        "\nGenerating global feature importance..."
    )

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    global_path = (
        REPORT_DIR
        / "global_feature_importance.png"
    )

    plt.savefig(
        global_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Bar-style importance
    # --------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_transformed,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=20,
    )

    plt.tight_layout()

    bar_path = (
        REPORT_DIR
        / "feature_importance_bar.png"
    )

    plt.savefig(
        bar_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------
    # Save raw SHAP values
    # --------------------------------------------------

    shap_dataframe = pd.DataFrame(
        shap_values,
        columns=feature_names,
    )

    shap_path = (
        REPORT_DIR
        / "shap_values.csv"
    )

    shap_dataframe.to_csv(
        shap_path,
        index=False,
    )

    # --------------------------------------------------
    # Print top features
    # --------------------------------------------------

    mean_absolute_shap = (
        shap_dataframe
        .abs()
        .mean()
        .sort_values(
            ascending=False
        )
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP SHAP FEATURES"
    )

    print(
        "=" * 70
    )

    for feature, importance in (
        mean_absolute_shap
        .head(20)
        .items()
    ):

        print(
            f"{feature}: "
            f"{importance:.6f}"
        )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SHAP EXPLAINABILITY COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nReports saved to:"
    )

    print(
        REPORT_DIR
    )

    print(
        "\nGlobal plot:"
    )

    print(
        global_path
    )

    print(
        "\nBar plot:"
    )

    print(
        bar_path
    )

    print(
        "\nSHAP values:"
    )

    print(
        shap_path
    )


if __name__ == "__main__":
    main()