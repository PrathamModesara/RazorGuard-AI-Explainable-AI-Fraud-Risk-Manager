from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "train.csv"
)

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "validation.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TARGET = "isFraud"

ID_COLUMN = "TransactionID"


NUMERICAL_FEATURES = [
    "TransactionAmt",
    "TransactionDT",
    "addr1",
    "dist1",
    "amount_log",
    "transaction_hour",
    "transaction_day",
    "is_early_morning",
    "is_night",
    "p_email_missing",
    "r_email_missing",
    "card4_missing",
    "card6_missing",
    "addr1_missing",
    "dist1_missing",
    "device_type_missing",
    "device_info_missing",
]


CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "amount_bucket",
]


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline."""

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model() -> Pipeline:
    """Build complete preprocessing + model pipeline."""

    preprocessor = build_preprocessor()

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Evaluate model using fraud-focused metrics."""

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    average_precision = (
        average_precision_score(
            y,
            probabilities,
        )
    )

    roc_auc = (
        roc_auc_score(
            y,
            probabilities,
        )
    )

    precision = (
        precision_score(
            y,
            predictions,
            zero_division=0,
        )
    )

    recall = (
        recall_score(
            y,
            predictions,
            zero_division=0,
        )
    )

    f1 = (
        f1_score(
            y,
            predictions,
            zero_division=0,
        )
    )

    print("\n" + "=" * 70)
    print("BASELINE MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nAverage Precision / PR-AUC: "
        f"{average_precision:.4f}"
    )

    print(
        f"ROC-AUC: "
        f"{roc_auc:.4f}"
    )

    print(
        f"Precision: "
        f"{precision:.4f}"
    )

    print(
        f"Recall: "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score: "
        f"{f1:.4f}"
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y,
            predictions,
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )


def main() -> None:

    print("=" * 70)
    print("RazorGuard AI - Baseline Fraud Model")
    print("=" * 70)

    print("\nLoading training data...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    print(
        f"Training rows: "
        f"{len(train_df):,}"
    )

    print("\nLoading validation data...")

    validation_df = pd.read_csv(
        VALIDATION_PATH
    )

    print(
        f"Validation rows: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------
    # Separate features and target
    # --------------------------------------------------

    X_train = train_df.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    )

    y_train = train_df[
        TARGET
    ]

    X_validation = validation_df.drop(
        columns=[
            TARGET,
            ID_COLUMN,
        ]
    )

    y_validation = validation_df[
        TARGET
    ]

    print(
        f"\nTraining features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Validation features: "
        f"{X_validation.shape[1]}"
    )

    # --------------------------------------------------
    # Build model
    # --------------------------------------------------

    print(
        "\nBuilding preprocessing + "
        "Logistic Regression pipeline..."
    )

    pipeline = build_model()

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print(
        "\nTraining baseline model..."
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    print(
        "\nEvaluating on validation data..."
    )

    evaluate_model(
        pipeline,
        X_validation,
        y_validation,
    )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------

    model_path = (
        MODEL_DIR
        / "logistic_regression_baseline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    print(
        f"\nModel saved to:"
    )

    print(
        model_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "BASELINE TRAINING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()