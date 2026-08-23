from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_PATH = (
    PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_selected_columns() -> pd.DataFrame:
    """Load only the columns required for the initial EDA."""

    columns = [
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

    chunks = []

    for chunk in pd.read_csv(
        TRANSACTION_PATH,
        usecols=columns,
        chunksize=100_000,
    ):
        chunks.append(chunk)

    return pd.concat(chunks, ignore_index=True)


def fraud_distribution(df: pd.DataFrame) -> None:
    """Create fraud vs legitimate transaction chart."""

    counts = df["isFraud"].value_counts().sort_index()

    labels = [
        "Legitimate",
        "Fraudulent",
    ]

    values = [
        counts.get(0, 0),
        counts.get(1, 0),
    ]

    plt.figure(figsize=(8, 5))

    plt.bar(
        labels,
        values,
    )

    plt.title("Transaction Fraud Distribution")
    plt.xlabel("Transaction Type")
    plt.ylabel("Number of Transactions")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fraud_distribution.png",
        dpi=150,
    )

    plt.close()


def transaction_amount_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze transaction amounts."""

    print("\nTransaction Amount Statistics")
    print("=" * 50)

    print(
        df["TransactionAmt"].describe()
    )

    legitimate = df.loc[
        df["isFraud"] == 0,
        "TransactionAmt",
    ]

    fraud = df.loc[
        df["isFraud"] == 1,
        "TransactionAmt",
    ]

    print(
        "\nMedian legitimate transaction:"
    )

    print(
        f"₹{legitimate.median():.2f}"
    )

    print(
        "\nMedian fraudulent transaction:"
    )

    print(
        f"₹{fraud.median():.2f}"
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        legitimate.clip(upper=1000),
        bins=50,
        alpha=0.6,
        label="Legitimate",
    )

    plt.hist(
        fraud.clip(upper=1000),
        bins=50,
        alpha=0.6,
        label="Fraudulent",
    )

    plt.title(
        "Transaction Amount Distribution"
    )

    plt.xlabel(
        "Transaction Amount"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "transaction_amount_distribution.png",
        dpi=150,
    )

    plt.close()


def fraud_rate_by_category(
    df: pd.DataFrame,
    column: str,
    filename: str,
) -> None:
    """Calculate and plot fraud rate for a categorical column."""

    data = df.copy()

    # Convert missing categorical values
    # into an explicit category.
    data[column] = data[column].astype(
        "object"
    ).where(
        data[column].notna(),
        "Missing",
    )

    summary = (
        data.groupby(column)["isFraud"]
        .agg(
            [
                "count",
                "sum",
            ]
        )
        .rename(
            columns={
                "count": "transactions",
                "sum": "fraud_transactions",
            }
        )
    )

    summary["fraud_rate"] = (
        summary["fraud_transactions"]
        / summary["transactions"]
        * 100
    )

    summary = summary.sort_values(
        "fraud_rate",
        ascending=False,
    )

    print(
        f"\nFraud Rate by {column}"
    )

    print(
        "=" * 50
    )

    print(
        summary.head(20).to_string()
    )

    plot_data = (
        summary.head(15)
        .copy()
    )

    categories = (
        plot_data.index.astype(str)
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        categories,
        plot_data["fraud_rate"].values,
    )

    plt.title(
        f"Fraud Rate by {column}"
    )

    plt.xlabel(
        column
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
    )

    plt.close()


def fraud_rate_by_amount_bucket(
    df: pd.DataFrame,
) -> None:
    """Analyze fraud rate across transaction amount ranges."""

    data = df.copy()

    data["amount_bucket"] = pd.cut(
        data["TransactionAmt"],
        bins=[
            0,
            10,
            25,
            50,
            100,
            250,
            500,
            1000,
            5000,
            float("inf"),
        ],
    )

    summary = (
        data.groupby(
            "amount_bucket",
            observed=False,
        )["isFraud"]
        .agg(
            [
                "count",
                "mean",
            ]
        )
    )

    summary["fraud_rate"] = (
        summary["mean"] * 100
    )

    print(
        "\nFraud Rate by Transaction Amount"
    )

    print(
        "=" * 50
    )

    print(
        summary.to_string()
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        summary.index.astype(str),
        summary["fraud_rate"],
    )

    plt.title(
        "Fraud Rate by Transaction Amount"
    )

    plt.xlabel(
        "Transaction Amount Bucket"
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fraud_rate_by_amount.png",
        dpi=150,
    )

    plt.close()


def missingness_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze missing values in selected columns."""

    missing = (
        df.isnull()
        .mean()
        .mul(100)
        .sort_values(
            ascending=False
        )
    )

    print(
        "\nMissing Value Percentage"
    )

    print(
        "=" * 50
    )

    print(
        missing.to_string()
    )

    top_missing = (
        missing.head(10)
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top_missing.index,
        top_missing.values,
    )

    plt.title(
        "Top Missing Features"
    )

    plt.xlabel(
        "Missing Values (%)"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "missing_values.png",
        dpi=150,
    )

    plt.close()


def time_analysis(
    df: pd.DataFrame,
) -> None:
    """Analyze transaction behavior over TransactionDT."""

    data = (
        df.sort_values(
            "TransactionDT"
        )
        .copy()
    )

    # TransactionDT represents elapsed time.
    # For initial EDA we use the repeating
    # 24-hour cycle as an approximate hour.
    data["time_hour"] = (
        (
            data["TransactionDT"]
            % 86400
        )
        // 3600
    )

    hourly = (
        data.groupby(
            "time_hour"
        )["isFraud"]
        .mean()
        .mul(100)
    )

    print(
        "\nFraud Rate by Transaction Hour"
    )

    print(
        "=" * 50
    )

    print(
        hourly.to_string()
    )

    plt.figure(figsize=(10, 5))

    plt.plot(
        hourly.index,
        hourly.values,
        marker="o",
    )

    plt.title(
        "Fraud Rate by Transaction Hour"
    )

    plt.xlabel(
        "Hour"
    )

    plt.ylabel(
        "Fraud Rate (%)"
    )

    plt.xticks(
        range(0, 24)
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fraud_rate_by_hour.png",
        dpi=150,
    )

    plt.close()


def main() -> None:
    """Run the complete initial fraud EDA."""

    print(
        "=" * 70
    )

    print(
        "RazorGuard AI - Fraud EDA"
    )

    print(
        "=" * 70
    )

    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: "
            f"{TRANSACTION_PATH}"
        )

    print(
        "\nLoading selected transaction columns..."
    )

    df = load_selected_columns()

    print(
        f"\nRows loaded: {len(df):,}"
    )

    print(
        f"Columns loaded: {len(df.columns)}"
    )

    print(
        "\nFraud counts:"
    )

    print(
        df["isFraud"].value_counts()
    )

    fraud_rate = (
        df["isFraud"].mean()
        * 100
    )

    print(
        f"\nOverall fraud rate: "
        f"{fraud_rate:.4f}%"
    )

    # --------------------------------------------------
    # Fraud distribution
    # --------------------------------------------------

    fraud_distribution(df)

    # --------------------------------------------------
    # Transaction amount analysis
    # --------------------------------------------------

    transaction_amount_analysis(
        df
    )

    # --------------------------------------------------
    # Categorical fraud analysis
    # --------------------------------------------------

    fraud_rate_by_category(
        df,
        "ProductCD",
        "fraud_rate_by_product.png",
    )

    fraud_rate_by_category(
        df,
        "card4",
        "fraud_rate_by_card4.png",
    )

    fraud_rate_by_category(
        df,
        "card6",
        "fraud_rate_by_card6.png",
    )

    fraud_rate_by_category(
        df,
        "P_emaildomain",
        "fraud_rate_by_p_emaildomain.png",
    )

    # --------------------------------------------------
    # Amount bucket analysis
    # --------------------------------------------------

    fraud_rate_by_amount_bucket(
        df
    )

    # --------------------------------------------------
    # Missing value analysis
    # --------------------------------------------------

    missingness_analysis(
        df
    )

    # --------------------------------------------------
    # Time analysis
    # --------------------------------------------------

    time_analysis(
        df
    )

    # --------------------------------------------------
    # Completion
    # --------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EDA COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nCharts saved to:"
    )

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":
    main()