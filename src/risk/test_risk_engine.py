from src.risk.risk_analysis import RiskEngine


def main():
    engine = RiskEngine(
        medium_threshold=0.30,
        high_threshold=0.80,
        threshold_version="v1",
    )

    test_transactions = [
        (100001, 0.12),
        (100002, 0.45),
        (100003, 0.81),
        (100004, 0.97),
    ]

    print("=" * 70)
    print("RazorGuard AI - Risk Engine Test")
    print("=" * 70)

    for transaction_id, probability in test_transactions:

        result = engine.evaluate(
            transaction_id=transaction_id,
            fraud_probability=probability,
        )

        print(
            f"\nTransaction: "
            f"{result.transaction_id}"
        )

        print(
            f"Fraud probability: "
            f"{result.fraud_probability}"
        )

        print(
            f"Risk score: "
            f"{result.risk_score}"
        )

        print(
            f"Risk level: "
            f"{result.risk_level}"
        )

        print(
            f"Decision: "
            f"{result.decision}"
        )

        print(
            f"Threshold version: "
            f"{result.threshold_version}"
        )


if __name__ == "__main__":
    main()