# RazorGuard AI — Explainable AI Fraud Risk Manager

RazorGuard AI is an explainable, cost-aware fraud risk management system designed for payment transaction screening.

It combines an XGBoost fraud detection model, risk-based decisioning, SHAP explainability, cost-sensitive threshold analysis, and a grounded local AI investigation agent to help risk analysts understand and act on suspicious transactions.

> **Built for the Razorpay AI Risk Manager track.**

---

## 1. Problem Statement

Payment fraud creates two major challenges:

1. Fraudulent transactions can cause financial losses.
2. Aggressive fraud blocking can reject legitimate customers and create unnecessary false positives.

A practical fraud-risk system therefore needs more than a binary classifier. It should:

- Detect suspicious transactions
- Estimate fraud probability
- Convert probability into operational risk levels
- Balance false-positive and false-negative costs
- Explain why a transaction received its risk score
- Provide an auditable investigation summary

RazorGuard AI is designed around this workflow.

---

## 2. Solution

RazorGuard AI processes a transaction through the following pipeline:

```text
Transaction Data
       |
       v
Feature Engineering
       |
       v
XGBoost Fraud Model
       |
       v
Fraud Probability
       |
       v
Cost-Aware Risk Engine
       |
       +--------------------+
       |                    |
       v                    v
Risk Decision         SHAP Explanation
       |                    |
       +---------+----------+
                 |
                 v
       Investigation Case
                 |
                 v
     Grounded AI Investigator
          (Local Ollama)
                 |
                 v
       Analyst Investigation Report
```

The ML model and Risk Engine remain the source of truth.

The AI investigation layer explains the supplied evidence and does not override the fraud probability, risk score, risk level, or Risk Engine decision.

---

## 3. Key Features

### Fraud Detection

- XGBoost-based fraud detection
- Logistic Regression baseline
- Stratified train/validation/test split
- Precision, recall, F1, PR-AUC and ROC-AUC evaluation
- Class imbalance handling using `scale_pos_weight`

### Risk Decisioning

| Risk Level | Probability | Decision |
|---|---:|---|
| LOW | `< 0.30` | `APPROVE` |
| MEDIUM | `0.30 – < 0.80` | `STEP_UP_VERIFICATION` |
| HIGH | `>= 0.80` | `BLOCK_OR_REVIEW` |

Threshold configuration is versioned as `v1`.

### Cost-Aware Decisioning

Illustrative project assumptions:

- False Positive Cost = 5
- False Negative Cost = 100

Lowest-cost threshold evaluated under this scenario:

- Threshold = 0.55
- False Positives = 11,604
- False Negatives = 1,028
- Total Cost = 160,820
- Average Cost / Transaction = 1.8155

These are project assumptions and are **not Razorpay internal costs**.

### Explainable AI

SHAP is used to explain individual model predictions through risk-increasing and risk-reducing contributions.

SHAP values are treated as model contributions, not proof of fraud or causation.

### AI Investigation Agent

The project includes a grounded local LLM investigation layer using:

```text
Ollama
+
Llama 3.2 3B
```

The LLM converts structured investigation evidence into a concise analyst-oriented report.

It does not override the ML model or Risk Engine.

---

# 4. Model Results

## Logistic Regression Baseline

| Metric | Result |
|---|---:|
| PR-AUC | 0.2393 |
| ROC-AUC | 0.7869 |
| Precision | 0.0975 |
| Recall | 0.6513 |
| F1 | 0.1696 |

## XGBoost

| Metric | Result |
|---|---:|
| PR-AUC | 0.3247 |
| ROC-AUC | 0.8550 |
| Precision | 0.1295 |
| Recall | 0.7184 |
| F1 | 0.2194 |

The XGBoost model improves PR-AUC and ROC-AUC over the Logistic Regression baseline.

Because fraud is an imbalanced classification problem, PR-AUC is particularly useful for evaluating performance on the fraud class.

---

# 5. Threshold Analysis

The best F1 threshold identified during validation was:

```text
Threshold = 0.80
Precision = 0.3410
Recall    = 0.3787
F1        = 0.3589
```

A separate cost analysis was performed because the F1-optimal threshold does not necessarily minimize operational cost.

Under:

```text
FP Cost = 5
FN Cost = 100
```

the lowest-cost threshold evaluated was:

```text
0.55
```

This demonstrates the difference between predictive optimization and operational risk optimization.

---

# 6. Explainable AI with SHAP

For an investigated transaction, RazorGuard identifies:

```text
Risk-Increasing Contributions
            +
Risk-Reducing Contributions
```

A positive SHAP value indicates that the feature increased the model's fraud prediction relative to the model baseline.

A negative SHAP value indicates that the feature decreased the model's fraud prediction relative to the model baseline.

SHAP values do not establish causation or independently prove fraudulent intent.

---

# 7. AI Investigation Agent

The investigation architecture is:

```text
Transaction
     |
     v
Feature Engineering
     |
     v
XGBoost
     |
     v
Fraud Probability
     |
     v
Risk Engine
     |
     v
Risk Decision
     |
     v
SHAP Evidence
     |
     v
Investigation Case
     |
     v
Local LLM Investigator
     |
     v
Analyst Report
```

The LLM is an explanation and investigation layer.

It is explicitly restricted from:

- Inventing customer history
- Inventing previous transactions
- Inventing geography
- Inventing device history
- Inventing merchant history
- Inventing financial losses
- Inventing chargebacks
- Inventing external fraud intelligence
- Changing fraud probability
- Changing risk score
- Changing risk level
- Changing the Risk Engine decision

If information is unavailable, the investigator is instructed to state that it is unavailable.

---

# 8. Example — Low Risk Transaction

```text
Transaction ID: 3444522

Fraud Probability: 0.124914
Risk Score:        12.49
Risk Level:        LOW
Decision:          APPROVE
```

Example risk-increasing SHAP contributions:

```text
TransactionAmt                    +0.233389
amount_log                        +0.049939
transaction_day                   +0.045346
card4_mastercard                  +0.028645
P_emaildomain_anonymous.com       +0.022673
```

Example risk-reducing SHAP contributions:

```text
P_emaildomain_att.net             -0.981183
r_email_missing                   -0.168585
dist1                             -0.164006
card6_credit                      -0.156136
device_type_missing               -0.154297
```

Risk Engine output:

```text
LOW
APPROVE
```

---

# 9. Example — High Risk Transaction

```text
Transaction ID: 3427568

Fraud Probability: 0.991299
Risk Score:        99.13
Risk Level:        HIGH
Decision:          BLOCK_OR_REVIEW
```

Top risk-increasing SHAP contributions:

```text
DeviceInfo_SM-A300H Build/LRX22G    +2.121027
ProductCD_C                         +0.827413
transaction_hour                    +0.385517
card6_credit                        +0.351995
is_early_morning                    +0.338679
```

Top risk-reducing SHAP contributions:

```text
TransactionAmt                      -0.124155
P_emaildomain_hotmail.com            -0.090223
R_emaildomain_gmail.com              -0.054541
amount_log                            -0.042481
addr1                                 -0.017826
```

Risk Engine output:

```text
Risk Level: HIGH
Decision: BLOCK_OR_REVIEW
```

The AI investigator converts this evidence into an analyst-readable investigation report without changing the underlying decision.

---

# 10. Dataset

RazorGuard AI was developed using the IEEE-CIS Fraud Detection dataset.

Processed modeling dataset:

```text
Rows:        590,540
Fraud Cases: 20,663
Fraud Rate:  3.4990%
Features:    27
```

Raw and processed datasets are intentionally excluded from the Git repository.

---

# 11. Feature Engineering

The feature pipeline includes:

### Transaction Features

- Transaction amount
- Log-transformed transaction amount
- Transaction amount bucket

### Temporal Features

- Transaction hour
- Transaction day
- Early-morning indicator
- Night indicator

### Card Features

- Card type
- Card category

### Email Features

- Purchaser email domain
- Recipient email domain
- Email missingness indicators

### Device Features

- Device type
- Device information
- Device missingness

### Address / Distance Features

- Address information
- Distance information
- Missingness indicators

---

# 12. Data Validation

The feature dataset is validated before model training.

Validation checks include:

- Duplicate TransactionIDs
- Missing target values
- Invalid target values
- Infinite numerical values
- Constant features
- Duplicate rows
- Fraud-rate consistency

---

# 13. Dataset Split

The dataset uses a stratified train/validation/test split.

| Split | Rows | Fraud Cases |
|---|---:|---:|
| Train | 413,378 | 14,464 |
| Validation | 88,581 | 3,100 |
| Test | 88,581 | 3,099 |
| Total | 590,540 | 20,663 |

The test set is kept separate for final evaluation.

---

# 14. Technology Stack

### Machine Learning

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
Joblib
```

### Explainability

```text
SHAP
```

### AI Investigation

```text
Ollama
Llama 3.2 3B
OpenAI-compatible local API
```

### Configuration

```text
YAML
python-dotenv
```

---

# 15. Project Structure

```text
RazorGuard-AI/
│
├── api/
│   └── __init__.py
│
├── configs/
│   ├── config.yaml
│   ├── feature_config.yaml
│   └── model_config.yaml
│
├── scripts/
│   ├── eda_fraud.py
│   ├── inspect_dataset.py
│   ├── split_dataset.py
│   └── validate_features.py
│
├── src/
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── investigator.py
│   │   └── llm_investigator.py
│   │
│   ├── agents/
│   │   └── __init__.py
│   │
│   ├── data/
│   │   └── __init__.py
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   └── transaction_explainer.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   │
│   ├── investigation/
│   │   ├── __init__.py
│   │   └── case_builder.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train_baseline.py
│   │   ├── train_xgboost.py
│   │   ├── threshold_analysis.py
│   │   └── cost_analysis.py
│   │
│   ├── monitoring/
│   │   └── __init__.py
│   │
│   ├── risk/
│   │   ├── risk_analysis.py
│   │   └── test_risk_engine.py
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   └── config.py
│   │
│   └── main.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

# 16. Local Setup

## Clone

```bash
git clone https://github.com/PrathamModesara/RazorGuard-AI-Explainable-AI-Fraud-Risk-Manager.git
cd RazorGuard-AI-Explainable-AI-Fraud-Risk-Manager
```

## Create Virtual Environment

```bash
python -m venv .venv
```

### Linux / WSL

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scriptsctivate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 17. Environment Configuration

Copy the example environment file:

```bash
cp .env.example .env
```

The `.env` file is intentionally ignored by Git.

Do not commit API keys or other secrets.

---

# 18. Ollama Setup

Install Ollama locally.

Download the investigation model:

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

Expected model:

```text
llama3.2:3b
```

Ollama runs locally at:

```text
http://localhost:11434
```

The RazorGuard investigator uses:

```text
http://localhost:11434/v1
```

No external LLM API key is required for the local investigation agent.

---

# 19. Run the AI Investigation Agent

Activate the environment:

```bash
source .venv/bin/activate
```

Run:

```bash
python src/agent/llm_investigator.py
```

Reports are saved under:

```text
reports/ai_investigation/
```

Generated reports are excluded from Git.

---

# 20. Model Training Workflow

### Dataset Inspection

```bash
python scripts/inspect_dataset.py
```

### Exploratory Data Analysis

```bash
python scripts/eda_fraud.py
```

### Feature Engineering

```bash
python src/features/build_features.py
```

### Feature Validation

```bash
python scripts/validate_features.py
```

### Dataset Splitting

```bash
python scripts/split_dataset.py
```

### Train Baseline

```bash
python src/models/train_baseline.py
```

### Train XGBoost

```bash
python src/models/train_xgboost.py
```

### Threshold Analysis

```bash
python src/models/threshold_analysis.py
```

### Cost Analysis

```bash
python src/models/cost_analysis.py
```

---

# 21. Auditability

For an investigation, RazorGuard can preserve structured evidence including:

- Transaction ID
- Model probability
- Risk score
- Risk level
- Risk Engine decision
- Model version
- Threshold version
- Top risk-increasing factors
- Top risk-reducing factors
- AI-generated investigation report

This provides a foundation for an auditable fraud investigation workflow.

---

# 22. Why RazorGuard AI Fits AI Risk Manager

RazorGuard AI is designed around a payment fraud defense workflow:

```text
Detect
  ↓
Score
  ↓
Assess Risk
  ↓
Explain
  ↓
Investigate
  ↓
Act
```

The system combines:

- Fraud detection
- Risk-based decisioning
- Cost-aware optimization
- Explainable AI
- Grounded AI investigation
- Auditability

Instead of simply returning:

```text
Fraud = Yes / No
```

RazorGuard attempts to answer:

```text
How risky is this transaction?
Why did the model assign this risk?
What should the Risk Engine do?
What evidence can an analyst review?
```

---

# 23. Buildathon Alignment

RazorGuard AI targets the:

```text
AI Risk Manager
```

track.

The system focuses on identifying suspicious payment transactions and routing them toward appropriate operational actions.

The project emphasizes:

- Measurable precision and recall
- Fraud detection performance
- Cost-aware false-positive / false-negative analysis
- Explainability
- Controlled AI assistance
- Auditable investigation evidence

The architecture is intentionally defense-oriented.

---

# 24. Security and Responsible AI

RazorGuard follows several principles relevant to financial-risk applications.

### Model and Risk Engine First

The ML model and deterministic Risk Engine remain the source of truth.

### Grounded Generation

The LLM is restricted to the evidence supplied by the investigation case.

### No Fabricated Intelligence

The investigator cannot invent customer history, geography, device history, losses, or external intelligence.

### No Autonomous Override

The LLM cannot override:

```text
Fraud Probability
Risk Score
Risk Level
Risk Engine Decision
```

### Explainability

SHAP is used to provide model-level evidence while explicitly avoiding causal claims.

### Human Oversight

The AI investigator is designed as decision support for a risk analyst rather than an autonomous financial decision-maker.

---

# 25. Limitations

RazorGuard AI is a project-level fraud-risk prototype and should not be interpreted as a production payment-fraud system.

Important limitations include:

- The IEEE-CIS dataset is not Razorpay production data.
- The cost values used in the analysis are illustrative assumptions.
- Model performance depends on the dataset and feature distribution.
- The system does not have real-time external fraud intelligence.
- The local LLM is used for investigation support and evidence summarization.
- Production deployment would require additional monitoring, calibration, security, privacy, governance, and compliance controls.
- Real-world fraud systems would require continuous model evaluation and retraining.

---

# 26. Future Improvements

Potential future extensions include:

- Real-time transaction scoring API
- Model probability calibration
- Real-time fraud monitoring
- Data and model drift detection
- Automated retraining workflows
- Analyst feedback loops
- Human-in-the-loop review queues
- Richer investigation case management
- Production-grade authentication and authorization
- Transaction-level audit trails
- Online feature stores
- Continuous model evaluation
- Production deployment

---

# 27. Current Project Status

The following components have been implemented:

- [x] Fraud dataset inspection
- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Feature validation
- [x] Stratified train/validation/test split
- [x] Logistic Regression baseline
- [x] XGBoost fraud model
- [x] Class imbalance handling
- [x] Threshold analysis
- [x] Cost-sensitive analysis
- [x] Risk Engine
- [x] SHAP global explainability
- [x] Transaction-level explainability
- [x] Investigation case builder
- [x] Grounded rule-based investigator
- [x] Local LLM investigation agent
- [x] Ollama integration
- [x] Llama 3.2 3B integration
- [x] Investigation report generation

---

# 28. End-to-End Flow

A typical transaction passes through:

```text
1. Transaction received
          ↓
2. Features generated
          ↓
3. XGBoost predicts fraud probability
          ↓
4. Risk Engine converts probability into risk level
          ↓
5. Risk Engine produces operational decision
          ↓
6. SHAP explains model contribution
          ↓
7. Investigation case is created
          ↓
8. Local LLM summarizes evidence
          ↓
9. Analyst reviews the investigation report
```

Example:

```text
Transaction ID
3427568

        ↓

Fraud Probability
0.991299

        ↓

Risk Score
99.13

        ↓

Risk Level
HIGH

        ↓

Decision
BLOCK_OR_REVIEW

        ↓

SHAP Explanation

        ↓

AI Investigation Report
```

---

# 29. Key Takeaway

RazorGuard AI demonstrates that a practical fraud-risk system should combine:

```text
Machine Learning
       +
Cost-Aware Decisioning
       +
Explainability
       +
Grounded Generative AI
       +
Auditability
```

The goal is not simply to build a highly accurate classifier.

The goal is to build a system that helps a risk analyst:

```text
Detect → Understand → Decide → Investigate
```

while keeping the final risk decision controlled and explainable.

---

# 30. Author

## Pratham Modesara

Graduate Student  
Alliance University  
Bangalore, India

GitHub:

https://github.com/PrathamModesara

---
