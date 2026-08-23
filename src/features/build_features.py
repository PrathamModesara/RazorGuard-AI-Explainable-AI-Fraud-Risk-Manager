from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train_transaction.csv"
)

IDENTITY_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "train_identity.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRANSACTION_COLUMNS = [
    "TransactionID",
    "isFraud",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "addr1",
    "dist1",
]


IDENTITY_COLUMNS = [
    "TransactionID",
    "DeviceType",
    "DeviceInfo",
]


def load_transaction_data() -> pd.DataFrame:
    """Load selected transaction columns."""

    print("Loading transaction data...")

    df = pd.read_csv(
        TRANSACTION_PATH,
        usecols=TRANSACTION_COLUMNS,
    )

    print(
        f"Transaction rows loaded: {len(df):,}"
    )

    return df


def load_identity_data() -> pd.DataFrame:
    """Load selected identity columns."""

    print("Loading identity data...")

    df = pd.read_csv(
        IDENTITY_PATH,
        usecols=IDENTITY_COLUMNS,
    )

    print(
        f"Identity rows loaded: {len(df):,}"
    )

    return df


def create_transaction_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create transaction-level derived features."""

    data = df.copy()

    # Log transformation reduces the effect
    # of extreme transaction amounts.
    data["amount_log"] = np.log1p(
        data["TransactionAmt"]
    )

    # Transaction amount buckets.
    data["amount_bucket"] = pd.cut(
        data["TransactionAmt"],
        bins=[
            -np.inf,
            10,
            25,
            50,
            100,
            250,
            500,
            1000,
            5000,
            np.inf,
        ],
        labels=[
            "very_low",
            "low",
            "lower_medium",
            "medium",
            "upper_medium",
            "high",
            "very_high",
            "extreme",
            "ultra_extreme",
        ],
    )

    return data


def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create time-related features from TransactionDT."""

    data = df.copy()

    # TransactionDT represents elapsed time.
    # We use the repeating daily cycle for
    # initial time-of-day features.
    seconds_in_day = 24 * 60 * 60

    seconds_of_day = (
        data["TransactionDT"]
        % seconds_in_day
    )

    data["transaction_hour"] = (
        seconds_of_day // 3600
    ).astype(int)

    data["transaction_day"] = (
        data["TransactionDT"]
        // seconds_in_day
    ).astype(int)

    data["is_early_morning"] = (
        data["transaction_hour"]
        .between(4, 8)
        .astype(int)
    )

    data["is_night"] = (
        (
            data["transaction_hour"] >= 22
        )
        | (
            data["transaction_hour"] <= 4
        )
    ).astype(int)

    return data


def create_missingness_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create explicit missing-value indicators."""

    data = df.copy()

    data["p_email_missing"] = (
        data["P_emaildomain"]
        .isna()
        .astype(int)
    )

    data["r_email_missing"] = (
        data["R_emaildomain"]
        .isna()
        .astype(int)
    )

    data["card4_missing"] = (
        data["card4"]
        .isna()
        .astype(int)
    )

    data["card6_missing"] = (
        data["card6"]
        .isna()
        .astype(int)
    )

    data["addr1_missing"] = (
        data["addr1"]
        .isna()
        .astype(int)
    )

    data["dist1_missing"] = (
        data["dist1"]
        .isna()
        .astype(int)
    )

    return data


def clean_categorical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert missing categorical values to an explicit category."""

    data = df.copy()

    categorical_columns = [
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "amount_bucket",
    ]

    for column in categorical_columns:
        data[column] = (
            data[column]
            .astype("object")
            .where(
                data[column].notna(),
                "Missing",
            )
        )

    return data


def merge_identity_data(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge selected identity information."""

    print("Merging transaction and identity data...")

    merged = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    print(
        f"Merged rows: {len(merged):,}"
    )

    return merged


def create_identity_missingness_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create missing indicators for identity features."""

    data = df.copy()

    data["device_type_missing"] = (
        data["DeviceType"]
        .isna()
        .astype(int)
    )

    data["device_info_missing"] = (
        data["DeviceInfo"]
        .isna()
        .astype(int)
    )

    data["DeviceType"] = (
        data["DeviceType"]
        .astype("object")
        .where(
            data["DeviceType"].notna(),
            "Missing",
        )
    )

    data["DeviceInfo"] = (
        data["DeviceInfo"]
        .astype("object")
        .where(
            data["DeviceInfo"].notna(),
            "Missing",
        )
    )

    return data


def select_final_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Select the initial RazorGuard feature set."""

    columns = [
        "TransactionID",
        "isFraud",

        # Raw transaction features
        "TransactionAmt",
        "TransactionDT",
        "ProductCD",

        # Payment features
        "card4",
        "card6",

        # Email features
        "P_emaildomain",
        "R_emaildomain",

        # Address features
        "addr1",
        "dist1",

        # Identity features
        "DeviceType",
        "DeviceInfo",

        # Amount features
        "amount_log",
        "amount_bucket",

        # Time features
        "transaction_hour",
        "transaction_day",
        "is_early_morning",
        "is_night",

        # Missingness features
        "p_email_missing",
        "r_email_missing",
        "card4_missing",
        "card6_missing",
        "addr1_missing",
        "dist1_missing",
        "device_type_missing",
        "device_info_missing",
    ]

    return df[columns].copy()


def main() -> None:
    """Run the RazorGuard feature engineering pipeline."""

    print("=" * 70)
    print("RazorGuard AI - Feature Engineering")
    print("=" * 70)

    transaction_df = load_transaction_data()

    identity_df = load_identity_data()

    transaction_df = create_transaction_features(
        transaction_df
    )

    transaction_df = create_time_features(
        transaction_df
    )

    transaction_df = create_missingness_features(
        transaction_df
    )

    transaction_df = merge_identity_data(
        transaction_df,
        identity_df,
    )

    transaction_df = (
        create_identity_missingness_features(
            transaction_df
        )
    )

    transaction_df = clean_categorical_features(
        transaction_df
    )

    final_df = select_final_features(
        transaction_df
    )

    print("\nFinal feature dataset:")
    print(
        f"Rows: {len(final_df):,}"
    )

    print(
        f"Columns: {len(final_df.columns)}"
    )

    print("\nFeature columns:")

    for column in final_df.columns:
        print(f"  - {column}")

    output_path = (
        OUTPUT_DIR
        / "razorguard_features.csv"
    )

    print(
        f"\nSaving processed dataset to:"
    )

    print(output_path)

    final_df.to_csv(
        output_path,
        index=False,
    )

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()