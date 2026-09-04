import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INVESTIGATION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "investigation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "ai_investigation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# LLM CONFIGURATION
# ============================================================

# Ollama local model.
# Can be overridden through .env if required.
MODEL_NAME = os.getenv(
    "RAZORGUARD_LLM_MODEL",
    "llama3.2:3b",
)

# Ollama exposes an OpenAI-compatible API locally.
OLLAMA_BASE_URL = os.getenv(
    "RAZORGUARD_LLM_BASE_URL",
    "http://localhost:11434/v1",
)

MAX_OUTPUT_TOKENS = 700


# ============================================================
# CASE LOADING
# ============================================================

def load_investigation_case(
    transaction_id: int,
) -> dict:
    """
    Load a previously generated investigation case.
    """

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


# ============================================================
# PROMPT CONSTRUCTION
# ============================================================

def build_system_prompt() -> str:
    """
    System instructions for the fraud investigation LLM.

    The LLM is an explanation layer only. The ML model and
    Risk Engine remain the source of truth.
    """

    return """
You are RazorGuard AI, an AI-assisted financial fraud
investigation analyst.

Your job is ONLY to summarize and explain the structured
evidence produced by RazorGuard AI.

The XGBoost fraud model and Risk Engine are the source of
truth. You are NOT allowed to make or change fraud decisions.

STRICT GROUNDING RULES:

1. Use ONLY information explicitly provided in the
   investigation case.

2. Never invent customer history, previous transactions,
   geography, device history, merchant history, financial
   loss, chargebacks, external intelligence, or user behavior.

3. Never infer facts that are not explicitly provided.

4. Do NOT interpret a raw feature value as a real-world fact
   unless the case explicitly provides that interpretation.

5. SHAP values describe the model's contribution from a
   feature. They do NOT prove fraud and do NOT prove causation.

6. Positive SHAP values indicate that the feature increased
   the model's fraud prediction relative to the model baseline.

7. Negative SHAP values indicate that the feature decreased
   the model's fraud prediction relative to the model baseline.

8. Do NOT change, recalculate, reinterpret, or contradict the
   supplied fraud probability.

9. Do NOT change, recalculate, reinterpret, or contradict the
   supplied risk score.

10. Do NOT change, recalculate, reinterpret, or contradict the
    supplied risk level.

11. Do NOT change, recalculate, reinterpret, or contradict the
    supplied Risk Engine decision.

12. Treat the Risk Engine decision as authoritative.

13. If evidence is unavailable, say:
    "Not available in the investigation case."

14. Do not use words such as "confirmed fraud",
    "fraudulent customer", or "criminal" unless the case
    explicitly contains such evidence.

15. Do not claim that a transaction is fraudulent solely because
    its fraud probability is high. Say that the model assigns
    a high fraud probability.

16. Do not describe an email domain, device, card, location,
    amount, or other feature as good, bad, valid, invalid,
    suspicious, trusted, or fraudulent unless that conclusion
    is explicitly supported by the supplied evidence.

17. Keep the report concise and professional.

18. The Recommended Action must match the supplied Risk Engine
    decision. Do not create a different action.

Use exactly these six sections:

1. Executive Assessment
2. Model Evidence
3. Risk-Reducing Evidence
4. Decision
5. Recommended Action
6. Investigation Limitations

In the Executive Assessment, state:
- the supplied fraud probability
- the supplied risk level
- the supplied risk score
- the supplied decision
- a neutral explanation based only on the evidence

In Model Evidence:
- list the most important risk-increasing SHAP features
- preserve their supplied names and values
- do not invent interpretations

In Risk-Reducing Evidence:
- list the supplied risk-reducing features and SHAP values
- explain only the direction of their model contribution

In Decision:
- reproduce the exact supplied Risk Engine decision
- reproduce the exact supplied fraud probability and risk score

In Recommended Action:
- reproduce the action implied by the supplied decision
- do not invent additional operational actions

In Investigation Limitations:
- state that SHAP values are model contributions, not proof
- state that the investigation is limited to the supplied case evidence
- state that the model and Risk Engine remain the source of truth
"""

def build_case_prompt(
    case: dict,
) -> str:
    """
    Convert the structured case into a tightly
    controlled LLM input.
    """

    risk_factors = json.dumps(
        case[
            "top_risk_factors"
        ],
        indent=2,
    )

    reducing_factors = json.dumps(
        case[
            "top_risk_reducing_factors"
        ],
        indent=2,
    )

    return f"""
Analyze the following RazorGuard investigation case.

TRANSACTION
-----------
Transaction ID:
{case["transaction_id"]}

ACTUAL LABEL
------------
{case["actual_label"]}

MODEL OUTPUT
------------
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

TOP RISK-INCREASING MODEL CONTRIBUTIONS
---------------------------------------

{risk_factors}

TOP RISK-REDUCING MODEL CONTRIBUTIONS
-------------------------------------

{reducing_factors}

Now produce the investigation report according
to the system instructions.

Important:
- Preserve the probability exactly.
- Preserve the risk level exactly.
- Preserve the decision exactly.
- Do not invent missing information.
- Treat SHAP values as model contributions.
"""


# ============================================================
# LLM CALL
# ============================================================

def generate_llm_report(
    case: dict,
) -> str:
    """
    Send the structured investigation case to the
    local Ollama LLM.

    Ollama provides an OpenAI-compatible API,
    therefore the existing OpenAI Python SDK can
    be reused without an OpenAI API key.
    """

    client = OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )

    system_prompt = (
        build_system_prompt()
    )

    user_prompt = (
        build_case_prompt(
            case
        )
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    if not response.choices:
        raise RuntimeError(
            "Ollama returned no response choices."
        )

    report = (
        response
        .choices[0]
        .message
        .content
    )

    if not report:
        raise RuntimeError(
            "Ollama returned an empty investigation report."
        )

    return report


# ============================================================
# SAVE REPORT
# ============================================================

def save_report(
    transaction_id: int,
    report: str,
) -> Path:
    """
    Save the generated AI investigation report.
    """

    output_path = (
        OUTPUT_DIR
        / f"case_{transaction_id}_llm_report.txt"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            report.strip()
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=" * 70
    )

    print(
        "RazorGuard AI - Local LLM Investigation Agent"
    )

    print(
        "=" * 70
    )

    print(
        f"LLM Provider: Ollama"
    )

    print(
        f"LLM Model: {MODEL_NAME}"
    )

    print(
        f"LLM Endpoint: {OLLAMA_BASE_URL}"
    )

    # --------------------------------------------------------
    # Demo cases
    # --------------------------------------------------------

    transaction_ids = [
        3444522,
        3427568,
    ]

    for transaction_id in transaction_ids:

        print(
            "\n" + "=" * 70
        )

        print(
            f"Investigating transaction "
            f"{transaction_id}"
        )

        print(
            "=" * 70
        )

        # ----------------------------------------------------
        # Load case
        # ----------------------------------------------------

        print(
            "\nLoading investigation case..."
        )

        case = load_investigation_case(
            transaction_id
        )

        print(
            "Investigation case loaded."
        )

        print(
            f"Fraud probability: "
            f"{case['fraud_probability']:.6f}"
        )

        print(
            f"Risk level: "
            f"{case['risk_level']}"
        )

        print(
            f"Decision: "
            f"{case['decision']}"
        )

        # ----------------------------------------------------
        # Generate report
        # ----------------------------------------------------

        print(
            "\nCalling local Ollama LLM investigator..."
        )

        try:

            report = generate_llm_report(
                case
            )

        except Exception as error:

            print(
                "\nERROR: Local LLM investigation failed."
            )

            print(
                f"Reason: {error}"
            )

            print(
                "\nMake sure Ollama is running and the"
                " llama3.2:3b model is installed."
            )

            raise

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "AI INVESTIGATION REPORT"
        )

        print(
            "=" * 70
        )

        print(
            "\n"
            + report
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_path = save_report(
            transaction_id,
            report,
        )

        print(
            "\nReport saved to:"
        )

        print(
            output_path
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "LLM investigation completed successfully."
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
