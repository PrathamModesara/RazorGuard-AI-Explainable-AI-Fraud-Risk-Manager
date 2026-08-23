from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INVESTIGATION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "investigation"
)


def load_case(
    transaction_id: int,
) -> dict:
    """Load a structured investigation case."""

    case_path = (
        INVESTIGATION_DIR
        / f"case_{transaction_id}.json"
    )

    if not case_path.exists():
        raise FileNotFoundError(
            f"Investigation case not found: "
            f"{case_path}"
        )

    with open(
        case_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def format_factors(
    factors: list[dict],
) -> str:
    """Format SHAP factors for investigation."""

    if not factors:
        return "None available."

    lines = []

    for factor in factors:
        lines.append(
            f"- {factor['feature']}: "
            f"SHAP contribution "
            f"{factor['shap_value']:.6f}"
        )

    return "\n".join(lines)


def build_investigation_prompt(
    case: dict,
) -> str:
    """
    Build a grounded investigation prompt.

    The investigator is instructed to use only
    evidence present in the structured case.
    """

    risk_factors = format_factors(
        case[
            "top_risk_factors"
        ]
    )

    reducing_factors = format_factors(
        case[
            "top_risk_reducing_factors"
        ]
    )

    return f"""
You are RazorGuard AI, a financial transaction
risk investigation assistant.

Your job is to interpret the evidence produced
by the RazorGuard machine-learning system.

IMPORTANT RULES:

1. Use ONLY the evidence provided below.
2. Do NOT invent customer history.
3. Do NOT invent previous transactions.
4. Do NOT invent device locations.
5. Do NOT invent financial losses.
6. Do NOT claim that a feature caused fraud.
7. Describe SHAP values as model contributions.
8. Clearly distinguish model evidence from
   business decisions.
9. If information is unavailable, say so.
10. Do not change the risk decision produced
    by the Risk Engine.

INVESTIGATION CASE
------------------

Transaction ID:
{case["transaction_id"]}

Actual Label:
{case["actual_label"]}

Fraud Probability:
{case["fraud_probability"]}

Risk Score:
{case["risk_score"]}

Risk Level:
{case["risk_level"]}

Decision:
{case["decision"]}

Model Version:
{case["model_version"]}

Threshold Version:
{case["threshold_version"]}

TOP RISK-INCREASING FACTORS
---------------------------

{risk_factors}

TOP RISK-REDUCING FACTORS
-------------------------

{reducing_factors}

Prepare an investigation report with exactly
these sections:

1. Risk Assessment
2. Key Evidence
3. Risk-Reducing Evidence
4. Decision
5. Recommended Action
6. Limitations

The report should be concise, professional,
and suitable for a fraud analyst reviewing
a payment transaction.
"""



def generate_grounded_report(
    case: dict,
) -> str:
    """
    Generate a deterministic evidence-grounded
    investigation report.

    This is the initial local agent implementation.
    A production LLM can later consume the same
    structured prompt.
    """

    probability = (
        case["fraud_probability"]
    )

    risk_level = (
        case["risk_level"]
    )

    decision = (
        case["decision"]
    )

    risk_factors = case[
        "top_risk_factors"
    ]

    reducing_factors = case[
        "top_risk_reducing_factors"
    ]

    # --------------------------------------------------
    # Risk assessment
    # --------------------------------------------------

    risk_assessment = (
        f"The model assigned a fraud probability "
        f"of {probability:.2%}, resulting in a "
        f"{risk_level} risk classification."
    )

    # --------------------------------------------------
    # Key evidence
    # --------------------------------------------------

    if risk_factors:

        evidence_lines = []

        for factor in risk_factors:
            evidence_lines.append(
                f"- {factor['feature']} "
                f"contributed positively to the "
                f"model's fraud prediction "
                f"(SHAP {factor['shap_value']:.6f})."
            )

        key_evidence = "\n".join(
            evidence_lines
        )

    else:

        key_evidence = (
            "No positive SHAP contributors "
            "were available."
        )

    # --------------------------------------------------
    # Risk reducing evidence
    # --------------------------------------------------

    if reducing_factors:

        reducing_lines = []

        for factor in reducing_factors:
            reducing_lines.append(
                f"- {factor['feature']} "
                f"contributed negatively to the "
                f"model's fraud prediction "
                f"(SHAP {factor['shap_value']:.6f})."
            )

        risk_reducing = "\n".join(
            reducing_lines
        )

    else:

        risk_reducing = (
            "No negative SHAP contributors "
            "were available."
        )

    # --------------------------------------------------
    # Recommended action
    # --------------------------------------------------

    if decision == "APPROVE":

        action = (
            "Approve the transaction under the "
            "current RazorGuard policy."
        )

    elif decision == "STEP_UP_VERIFICATION":

        action = (
            "Request additional verification "
            "before completing the transaction."
        )

    else:

        action = (
            "Block the transaction or send it "
            "for manual fraud review."
        )

    # --------------------------------------------------
    # Limitations
    # --------------------------------------------------

    limitations = (
        "This assessment is based on the features "
        "available to the model and the SHAP "
        "contributions generated for this case. "
        "The model evidence does not establish "
        "that any individual feature caused fraud. "
        "No additional customer history or external "
        "fraud intelligence was provided."
    )

    # --------------------------------------------------
    # Final report
    # --------------------------------------------------

    report = f"""
============================================================
RAZORGUARD AI - FRAUD INVESTIGATION REPORT
============================================================

1. Risk Assessment
------------------

{risk_assessment}


2. Key Evidence
---------------

{key_evidence}


3. Risk-Reducing Evidence
-------------------------

{risk_reducing}


4. Decision
-----------

{decision}


5. Recommended Action
---------------------

{action}


6. Limitations
--------------

{limitations}

============================================================
Model Version: {case["model_version"]}
Threshold Version: {case["threshold_version"]}
Transaction ID: {case["transaction_id"]}
============================================================
"""

    return report


def main() -> None:

    print(
        "=" * 70
    )

    print(
        "RazorGuard AI - Fraud Investigation Agent"
    )

    print(
        "=" * 70
    )

    transaction_id = 3444522

    print(
        f"\nLoading investigation case "
        f"for transaction {transaction_id}..."
    )

    case = load_case(
        transaction_id
    )

    print(
        "Investigation case loaded."
    )

    print(
        "\nBuilding grounded investigation prompt..."
    )

    prompt = build_investigation_prompt(
        case
    )

    prompt_path = (
        INVESTIGATION_DIR
        / f"case_{transaction_id}_prompt.txt"
    )

    with open(
        prompt_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            prompt.strip()
        )

    print(
        f"Prompt saved to:\n{prompt_path}"
    )

    print(
        "\nGenerating grounded investigation report..."
    )

    report = generate_grounded_report(
        case
    )

    report_path = (
        INVESTIGATION_DIR
        / f"case_{transaction_id}_report.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report.strip()
        )

    print(
        "\n" + report
    )

    print(
        "\nInvestigation report saved to:"
    )

    print(
        report_path
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "AI INVESTIGATION AGENT COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()