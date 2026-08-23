from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRANSACTION_PATH = PROJECT_ROOT / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = PROJECT_ROOT / "data" / "raw" / "train_identity.csv"


def inspect_csv(path: Path, name: str) -> None:
    """Inspect a CSV using a small sample first."""

    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    # Read only a small sample.
    sample = pd.read_csv(path, nrows=5000)

    print(f"\nFile: {path}")
    print(f"Sample rows: {len(sample):,}")
    print(f"Number of columns: {len(sample.columns):,}")

    print("\nColumn names:")
    for column in sample.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(sample.dtypes.to_string())

    print("\nMissing values in sample:")
    missing = sample.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  No missing values in sample.")
    else:
        print(missing.to_string())

    print("\nSample memory usage:")
    memory_mb = sample.memory_usage(deep=True).sum() / (1024 ** 2)
    print(f"  {memory_mb:.2f} MB")


def inspect_transaction_target() -> None:
    """Inspect the fraud target using chunks."""

    print("\n" + "=" * 70)
    print("FRAUD TARGET ANALYSIS")
    print("=" * 70)

    fraud_counts = {}

    total_rows = 0

    for chunk in pd.read_csv(
        TRANSACTION_PATH,
        usecols=["TransactionID", "isFraud"],
        chunksize=100_000,
    ):
        total_rows += len(chunk)

        counts = chunk["isFraud"].value_counts()

        for value, count in counts.items():
            fraud_counts[value] = fraud_counts.get(value, 0) + int(count)

    print(f"\nTotal transactions: {total_rows:,}")

    legitimate = fraud_counts.get(0, 0)
    fraud = fraud_counts.get(1, 0)

    print(f"Legitimate transactions: {legitimate:,}")
    print(f"Fraudulent transactions: {fraud:,}")

    if total_rows:
        fraud_rate = fraud / total_rows * 100
        print(f"Fraud rate: {fraud_rate:.4f}%")

    print("\nTarget distribution:")
    for label, count in sorted(fraud_counts.items()):
        percentage = count / total_rows * 100
        print(f"  isFraud={label}: {count:,} ({percentage:.4f}%)")


def main() -> None:
    print("RazorGuard AI - Dataset Inspection")

    if not TRANSACTION_PATH.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {TRANSACTION_PATH}"
        )

    if not IDENTITY_PATH.exists():
        raise FileNotFoundError(
            f"Identity dataset not found: {IDENTITY_PATH}"
        )

    inspect_csv(
        TRANSACTION_PATH,
        "Transaction",
    )

    inspect_csv(
        IDENTITY_PATH,
        "Identity",
    )

    inspect_transaction_target()

    print("\n" + "=" * 70)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()