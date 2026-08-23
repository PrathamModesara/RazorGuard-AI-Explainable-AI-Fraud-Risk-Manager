from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "razorguard_features.csv"
)


def main() -> None:
    print("=" * 70)
    print("RazorGuard AI - Feature Validation")
    print("=" * 70)

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {FEATURE_PATH}"
        )

    print("\nLoading feature dataset...")

    df = pd.read_csv(FEATURE_PATH)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------
    # 1. Column information
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COLUMN INFORMATION")
    print("=" * 70)

    print(df.dtypes.to_string())

    # --------------------------------------------------
    # 2. Duplicate Transaction IDs
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATE CHECK")
    print("=" * 70)

    duplicate_ids = (
        df["TransactionID"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate TransactionID values: "
        f"{duplicate_ids:,}"
    )

    # --------------------------------------------------
    # 3. Target validation
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET VALIDATION")
    print("=" * 70)

    print(
        df["isFraud"]
        .value_counts(dropna=False)
        .to_string()
    )

    missing_target = (
        df["isFraud"]
        .isna()
        .sum()
    )

    print(
        f"\nMissing target values: "
        f"{missing_target:,}"
    )

    invalid_targets = (
        ~df["isFraud"].isin([0, 1])
    ).sum()

    print(
        f"Invalid target values: "
        f"{invalid_targets:,}"
    )

    # --------------------------------------------------
    # 4. Missing values
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = (
        df.isna()
        .sum()
        .sort_values(
            ascending=False
        )
    )

    missing = missing[
        missing > 0
    ]

    if missing.empty:
        print(
            "No missing values found."
        )

    else:
        missing_percentage = (
            missing
            / len(df)
            * 100
        )

        missing_table = pd.DataFrame(
            {
                "missing_count": missing,
                "missing_percentage": (
                    missing_percentage
                    .round(2)
                ),
            }
        )

        print(
            missing_table.to_string()
        )

    # --------------------------------------------------
    # 5. Numerical statistics
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("NUMERICAL FEATURE STATISTICS")
    print("=" * 70)

    numerical_columns = (
        df.select_dtypes(
            include=["number"]
        ).columns
    )

    print(
        df[numerical_columns]
        .describe()
        .T
        .to_string()
    )

    # --------------------------------------------------
    # 6. Infinite values
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("INFINITE VALUE CHECK")
    print("=" * 70)

    numerical_df = df[
        numerical_columns
    ]

    infinity_count = np.isinf(
        numerical_df.to_numpy()
    ).sum()

    print(
        f"Infinite numerical values: "
        f"{infinity_count:,}"
    )

    # --------------------------------------------------
    # 7. Categorical cardinality
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CATEGORICAL CARDINALITY")
    print("=" * 70)

    categorical_columns = (
        df.select_dtypes(
            include=["object", "string"]
        ).columns
    )

    for column in categorical_columns:

        unique_count = (
            df[column]
            .nunique(
                dropna=False
            )
        )

        print(
            f"{column}: "
            f"{unique_count:,} unique values"
        )

    # --------------------------------------------------
    # 8. Constant columns
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CONSTANT FEATURE CHECK")
    print("=" * 70)

    constant_columns = [
        column
        for column in df.columns
        if df[column].nunique(
            dropna=False
        ) <= 1
    ]

    if constant_columns:

        print(
            "Constant columns:"
        )

        for column in constant_columns:
            print(
                f"  - {column}"
            )

    else:

        print(
            "No constant columns found."
        )

    # --------------------------------------------------
    # 9. Duplicate rows
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("DUPLICATE ROW CHECK")
    print("=" * 70)

    duplicate_rows = (
        df.duplicated()
        .sum()
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    # --------------------------------------------------
    # 10. Target distribution
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("TARGET DISTRIBUTION")
    print("=" * 70)

    target_distribution = (
        df["isFraud"]
        .value_counts()
        .sort_index()
    )

    total = len(df)

    for target, count in (
        target_distribution.items()
    ):

        percentage = (
            count
            / total
            * 100
        )

        label = (
            "Legitimate"
            if target == 0
            else "Fraudulent"
        )

        print(
            f"{label}: "
            f"{count:,} "
            f"({percentage:.4f}%)"
        )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE VALIDATION COMPLETE")
    print("=" * 70)

    print("\nValidation summary:")

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Features: {len(df.columns):,}"
    )

    print(
        f"Duplicate TransactionIDs: "
        f"{duplicate_ids:,}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Missing target values: "
        f"{missing_target:,}"
    )

    print(
        f"Invalid target values: "
        f"{invalid_targets:,}"
    )

    print(
        f"Infinite numerical values: "
        f"{infinity_count:,}"
    )

    print("\nAll basic validation checks completed.")


if __name__ == "__main__":
    main()