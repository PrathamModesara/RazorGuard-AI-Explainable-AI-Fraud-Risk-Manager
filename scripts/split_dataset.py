from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "razorguard_features.csv"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

SPLIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


RANDOM_STATE = 42


def print_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:
    """Print dataset size and fraud distribution."""

    total = len(df)

    fraud_count = (
        df["isFraud"]
        .sum()
    )

    fraud_rate = (
        fraud_count
        / total
        * 100
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * 50
    )

    print(
        f"Rows: {total:,}"
    )

    print(
        f"Fraudulent: {fraud_count:,}"
    )

    print(
        f"Legitimate: "
        f"{total - fraud_count:,}"
    )

    print(
        f"Fraud rate: "
        f"{fraud_rate:.4f}%"
    )


def main() -> None:

    print("=" * 70)
    print(
        "RazorGuard AI - Dataset Splitting"
    )
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: "
            f"{FEATURE_PATH}"
        )

    print(
        "\nLoading processed feature dataset..."
    )

    df = pd.read_csv(
        FEATURE_PATH
    )

    print(
        f"Total rows: {len(df):,}"
    )

    # --------------------------------------------------
    # First split:
    #
    # 70% training
    # 30% temporary
    # --------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["isFraud"],
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------
    # Second split:
    #
    # 15% validation
    # 15% test
    #
    # temp = 30%
    #
    # Half of temp = 15% overall
    # --------------------------------------------------

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["isFraud"],
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------
    # Print distributions
    # --------------------------------------------------

    print_distribution(
        "FULL DATASET",
        df,
    )

    print_distribution(
        "TRAINING DATASET",
        train_df,
    )

    print_distribution(
        "VALIDATION DATASET",
        validation_df,
    )

    print_distribution(
        "TEST DATASET",
        test_df,
    )

    # --------------------------------------------------
    # Save datasets
    # --------------------------------------------------

    train_path = (
        SPLIT_DIR
        / "train.csv"
    )

    validation_path = (
        SPLIT_DIR
        / "validation.csv"
    )

    test_path = (
        SPLIT_DIR
        / "test.csv"
    )

    print(
        "\nSaving datasets..."
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    print(
        f"\nTrain: {train_path}"
    )

    print(
        f"Validation: {validation_path}"
    )

    print(
        f"Test: {test_path}"
    )

    # --------------------------------------------------
    # Verify row counts
    # --------------------------------------------------

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    print(
        "\nSplit verification:"
    )

    print(
        f"Original rows: "
        f"{len(df):,}"
    )

    print(
        f"Split rows: "
        f"{total_split_rows:,}"
    )

    if total_split_rows == len(df):
        print(
            "✓ Row count verification passed."
        )

    else:
        raise RuntimeError(
            "Split row count does not "
            "match original dataset."
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "DATASET SPLITTING COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()