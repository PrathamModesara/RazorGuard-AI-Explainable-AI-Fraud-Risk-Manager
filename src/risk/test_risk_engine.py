import pytest

from src.risk.risk_analysis import RiskEngine


@pytest.fixture
def engine():
    return RiskEngine(
        medium_threshold=0.30,
        high_threshold=0.80,
        threshold_version="v1",
    )


def test_low_risk_approve(engine):
    result = engine.evaluate(
        transaction_id=100001,
        fraud_probability=0.12,
    )

    assert result.risk_score == 12.0
    assert result.risk_level == "LOW"
    assert result.decision == "APPROVE"
    assert result.threshold_version == "v1"


def test_medium_risk_step_up_verification(engine):
    result = engine.evaluate(
        transaction_id=100002,
        fraud_probability=0.45,
    )

    assert result.risk_score == 45.0
    assert result.risk_level == "MEDIUM"
    assert result.decision == "STEP_UP_VERIFICATION"


def test_high_risk_block_or_review(engine):
    result = engine.evaluate(
        transaction_id=100003,
        fraud_probability=0.81,
    )

    assert result.risk_score == 81.0
    assert result.risk_level == "HIGH"
    assert result.decision == "BLOCK_OR_REVIEW"


def test_threshold_boundaries(engine):
    medium = engine.evaluate(
        transaction_id=100004,
        fraud_probability=0.30,
    )

    high = engine.evaluate(
        transaction_id=100005,
        fraud_probability=0.80,
    )

    assert medium.risk_level == "MEDIUM"
    assert medium.decision == "STEP_UP_VERIFICATION"

    assert high.risk_level == "HIGH"
    assert high.decision == "BLOCK_OR_REVIEW"


@pytest.mark.parametrize(
    "probability",
    [-0.01, 1.01],
)
def test_invalid_probability_rejected(engine, probability):
    with pytest.raises(ValueError):
        engine.evaluate(
            transaction_id=100006,
            fraud_probability=probability,
        )


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError):
        RiskEngine(
            medium_threshold=0.80,
            high_threshold=0.30,
        )

    with pytest.raises(ValueError):
        RiskEngine(
            medium_threshold=-0.10,
            high_threshold=0.80,
        )

    with pytest.raises(ValueError):
        RiskEngine(
            medium_threshold=0.30,
            high_threshold=1.10,
        )
