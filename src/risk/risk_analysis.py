from dataclasses import dataclass


@dataclass
class RiskDecision:
    transaction_id: int
    fraud_probability: float
    risk_score: float
    risk_level: str
    decision: str
    threshold_version: str


class RiskEngine:
    """
    Convert model fraud probability into a
    business-oriented RazorGuard risk decision.
    """

    def __init__(
        self,
        medium_threshold: float = 0.30,
        high_threshold: float = 0.80,
        threshold_version: str = "v1",
    ):
        if not 0 <= medium_threshold <= 1:
            raise ValueError(
                "medium_threshold must be between 0 and 1."
            )

        if not 0 <= high_threshold <= 1:
            raise ValueError(
                "high_threshold must be between 0 and 1."
            )

        if medium_threshold >= high_threshold:
            raise ValueError(
                "medium_threshold must be lower than "
                "high_threshold."
            )

        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.threshold_version = threshold_version

    def calculate_risk_score(
        self,
        fraud_probability: float,
    ) -> float:
        """
        Convert fraud probability to a 0-100 risk score.
        """

        if not 0 <= fraud_probability <= 1:
            raise ValueError(
                "fraud_probability must be between 0 and 1."
            )

        return round(
            fraud_probability * 100,
            2,
        )

    def determine_risk_level(
        self,
        fraud_probability: float,
    ) -> str:
        """
        Assign LOW, MEDIUM, or HIGH risk.
        """

        if fraud_probability >= self.high_threshold:
            return "HIGH"

        if fraud_probability >= self.medium_threshold:
            return "MEDIUM"

        return "LOW"

    def determine_decision(
        self,
        fraud_probability: float,
    ) -> str:
        """
        Map risk level to an initial business action.
        """

        if fraud_probability >= self.high_threshold:
            return "BLOCK_OR_REVIEW"

        if fraud_probability >= self.medium_threshold:
            return "STEP_UP_VERIFICATION"

        return "APPROVE"

    def evaluate(
        self,
        transaction_id: int,
        fraud_probability: float,
    ) -> RiskDecision:
        """
        Produce a complete risk decision.
        """

        risk_score = self.calculate_risk_score(
            fraud_probability
        )

        risk_level = self.determine_risk_level(
            fraud_probability
        )

        decision = self.determine_decision(
            fraud_probability
        )

        return RiskDecision(
            transaction_id=transaction_id,
            fraud_probability=round(
                fraud_probability,
                6,
            ),
            risk_score=risk_score,
            risk_level=risk_level,
            decision=decision,
            threshold_version=self.threshold_version,
        )