# Credit Card Fraud Detection

A production-grade, end-to-end machine learning pipeline designed to detect fraudulent credit card transactions in real time. Built with robust data cleaning, domain-driven feature engineering, class imbalance mitigation via SMOTE, gradient-boosted decision trees (LightGBM), optimal decision threshold calibration, and SHAP explainability.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Dataset Overview](#dataset-overview)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Model Architecture and Comparison](#model-architecture-and-comparison)
- [Evaluation and Performance Metrics](#evaluation-and-performance-metrics)
- [Model Interpretability (SHAP)](#model-interpretability-shap)
- [Real-Time Inference Pipeline](#real-time-inference-pipeline)
- [Installation and Getting Started](#installation-and-getting-started)
- [Artifacts and Documentation](#artifacts-and-documentation)
- [License and Citation](#license-and-citation)
- [Author](#author)

---

## Overview

Credit card fraud presents a significant challenge for financial institutions, characterized by extreme class imbalance and fast-evolving fraud patterns. In financial ecosystems, the cost of a False Negative (undetected fraud resulting in direct capital loss and chargeback liabilities) drastically outweighs the operational cost of a False Positive (customer friction from verification prompts).

This repository implements a complete, reproducible machine learning solution developed to address:
- **Severe Class Imbalance**: Managing a 2.06% fraud prevalence (~47.7 : 1 ratio of legitimate to fraudulent transactions) across 100,000 records.
- **Zero Data Leakage**: Enforcing strict separation of train and test sets prior to resampling and transformation.
- **Explainability**: Providing granular, game-theoretic feature attribution using SHAP TreeExplainer for compliance and auditability.
- **Inference Readiness**: Exporting a lightweight, serialized model bundle capable of low-latency, real-time transaction scoring.

---

## Key Features

- **Disciplined Pipeline**: Fully decoupled data ingestion, transformation, SMOTE oversampling, and model training routines.
- **Domain-Specific Feature Engineering**: Behavioral velocity indicators, spending deviation ratios, geodetic impossible-travel flags, and risk composite proxies.
- **Multi-Model Benchmarking**: Baseline Logistic Regression (regularized with SAGA solver) compared against LightGBM Gradient Boosted Decision Trees.
- **Threshold Optimization**: Comprehensive sweep across probability thresholds (0.05 to 0.95) to maximize operational F1-score rather than defaulting to an arbitrary 0.5 threshold.
- **Explainable AI (XAI)**: Global and local interpretability via SHAP (SHapley Additive exPlanations) to satisfy regulatory transparency standards.
- **Production Artifact Packaging**: Clean serialization containing trained model weights, feature order contracts, scaler parameters, and metadata registry.

---

## Repository Structure

```text
Credit Card Fraud Detection/
|-- Dataset/
|   `-- credit_card_fraud_detection_2026.csv   # Raw transaction dataset (100,000 rows)
|-- Model/
|   |-- credit_card_fraud_detection.ipynb      # Main pipeline notebook (14 sections)
|   `-- model_artifacts/                       # Serialized models, registry, and plots
|       |-- lgb_fraud_detector.pkl             # Trained LightGBM champion model
|       |-- feature_scaler.pkl                 # StandardScaler fitted on training data
|       |-- feature_names.json                 # Ordered schema contract for inference
|       |-- model_registry.json                # Model metadata, parameters, and metrics
|       |-- MODEL_KNOWLEDGE.md                 # Complete engineering documentation
|       |-- 01_class_distribution.png          # Target class distribution plot
|       |-- 02_fraud_type_distribution.png     # Fraud sub-type breakdown plot
|       |-- 03_feature_distributions.png       # Numerical feature distribution plots
|       |-- 04_fraud_rate_by_category.png      # Categorical risk breakdown
|       |-- 05_correlation_heatmap.png         # Pearson correlation matrix
|       |-- 06_smote_balance.png               # Pre/post-SMOTE balance comparison
|       |-- 07_model_comparison.png            # Model comparison benchmark chart
|       |-- 08_shap_bar.png                    # Global feature importance ranking
|       |-- 09_shap_beeswarm.png               # SHAP feature impact distribution
|       |-- 10_threshold_analysis.png          # Precision-Recall vs. threshold curve
|       |-- eval_logistic_regression.png       # Baseline ROC, PR, and Confusion Matrix
|       `-- eval_lightgbm.png                  # Champion ROC, PR, and Confusion Matrix
|-- requirements.txt                           # Pinned dependencies for reproduction
|-- LICENSE                                    # MIT License with attribution clause
`-- README.md                                  # Repository documentation
```

---

## Dataset Overview

The dataset (`credit_card_fraud_detection_2026.csv`) contains 100,000 synthetic transaction records reflecting modern banking payment rails.

| Attribute | Specification |
|---|---|
| Total Observations | 100,000 transactions |
| Target Variable | `is_fraud` (0 = Legitimate, 1 = Fraud) |
| Fraud Class Proportion | 2.06% (2,055 fraudulent cases) |
| Legitimate Class Proportion | 97.94% (97,945 legitimate cases) |
| Imbalance Ratio | ~47.7 : 1 |
| Feature Modalities | Numerical, Categorical, Datetime, Geospatial, Historical velocity |

### Feature Groups
1. **Transaction Attributes**: `transaction_amount`, `transaction_hour`, `transaction_day_of_week`, `is_weekend`, `card_present`, `transaction_channel`, `device_type`, `browser`, `used_chip`, `used_pin`.
2. **Merchant Attributes**: `merchant_category`, `merchant_country`, `international_transaction`, `high_risk_country`.
3. **Customer and Account Attributes**: `customer_age`, `customer_gender`, `customer_state`, `account_age_days`, `credit_limit`, `credit_score`, `num_cards_held`, `email_domain`.
4. **Behavioral Velocity Metrics**: `transactions_last_1h`, `transactions_last_24h`, `transactions_last_7d`, `amount_last_1h`, `amount_last_24h`, `days_since_last_transaction`, `distance_from_home_km`, `distance_from_last_tx_km`, `avg_transaction_amount_30d`, `std_transaction_amount_30d`, `amount_vs_avg_ratio`.
5. **Historical Risk Metrics**: `num_declined_last_30d`, `num_chargebacks_ever`.

---

## Data Cleaning and Preprocessing

1. **Target Leakage Prevention**:
   - `fraud_type` and `fraud_amount_lost` were excluded because they are downstream consequences recorded only after a transaction is classified as fraud.
   - Non-predictive identifiers (`transaction_id`, `customer_id`, `card_id`, `merchant_id`) were dropped to prevent spurious correlation and memorization.
   - `transaction_datetime` was parsed for temporal signals (`transaction_hour`, `transaction_day_of_week`) and subsequently dropped.

2. **Structural Missingness Treatment**:
   - The `browser` feature exhibits ~41.7% missingness. Analysis reveals this is structural: physical transactions conducted at POS terminals or ATMs do not generate a browser signature.
   - Instead of dropping rows or applying mean/mode imputation, missing entries are designated with a dedicated categorical token `'N/A'`. This preserves an informative behavioral indicator of offline channel usage.

3. **Encoding**:
   - Categorical columns are transformed using `LabelEncoder`. Encoders are fitted strictly on the training partition and transformed across validation/test partitions to prevent data snooping.

4. **Data Splitting**:
   - An 80/20 train/test split is applied with stratification on the target variable (`stratify=y`, `random_state=42`), preserving the exact 2.06% fraud distribution across both subsets.

---

## Feature Engineering

Six domain-informed features were constructed to capture complex transaction anomalies and behavioral shifts:

| Feature Name | Calculation / Logic | Fraud Detection Objective |
|---|---|---|
| `velocity_ratio_1h_24h` | `transactions_last_1h / (transactions_last_24h + 1)` | Detects high-frequency transaction bursts indicative of automated card testing. |
| `amount_spike_ratio` | `transaction_amount / (avg_transaction_amount_30d + 1)` | Identifies sudden capital extraction exceeding historical baseline habits. |
| `impossible_travel_flag` | `1 if (distance_from_last_tx_km > 500 and days_since_last_tx < 1) else 0` | Flags geographically impossible consecutive transactions across short intervals. |
| `risk_score_proxy` | Weighted sum of `high_risk_country`, `international_transaction`, `num_chargebacks_ever`, `num_declined_last_30d` | Aggregates multi-factor account and location risk into a composite metric. |
| `credit_utilization_ratio` | `transaction_amount / (credit_limit + 1)` | Normalizes single-transaction impact relative to account credit headroom. |
| `security_strength` | `used_chip + used_pin + card_present` | Measures authentication depth (0 to 3 scale) to flag weak, unauthenticated channels. |

---

## Handling Class Imbalance

With fewer than 2.1% positive cases, models trained on raw distributions exhibit strong bias toward the majority class.

- **Method**: Synthetic Minority Over-sampling Technique (SMOTE) via `imblearn.over_sampling.SMOTE` (`k_neighbors=5`, `random_state=42`).
- **Resampling Scope**: SMOTE is applied **strictly to the training subset**. The test partition remains completely untouched to reflect authentic real-world transaction distributions.
- **Result**: Resampled the training set from 78,356 negative / 1,644 positive samples to a balanced 78,356 / 78,356 distribution (156,712 total training samples).
- **Ensemble Compensation**: In addition to SMOTE, LightGBM incorporates internal leaf weight adjustments (`scale_pos_weight`) to penalize minority misclassifications during gradient boosting.

---

## Model Architecture and Comparison

Two distinct classification paradigms were evaluated:

### 1. Baseline Model: Regularized Logistic Regression
- **Scaler**: `StandardScaler` fitted on the SMOTE-resampled training data.
- **Parameters**: `C=0.1`, `penalty='l2'`, `solver='saga'`, `max_iter=1000`, `random_state=42`.
- **Purpose**: Establishes a transparent linear baseline to benchmark non-linear gradient-boosted gains.

### 2. Champion Model: LightGBM Classifier
- **Parameters**:
  - `objective`: `'binary'`
  - `metric`: `['binary_logloss', 'auc']`
  - `boosting_type`: `'gbdt'`
  - `num_leaves`: `63`
  - `max_depth`: `7`
  - `learning_rate`: `0.05`
  - `n_estimators`: `600` (equipped with 50-round early stopping on validation PR-AUC)
  - `min_child_samples`: `30`
  - `subsample`: `0.8`
  - `colsample_bytree`: `0.8`
  - `reg_alpha`: `0.1` (L1 regularization)
  - `reg_lambda`: `0.1` (L2 regularization)
  - `random_state`: `42`

---

## Evaluation and Performance Metrics

Because accuracy is misleading in highly skewed domains (a naive model predicting 0 for all transactions achieves 97.94% accuracy), models are evaluated using:
- **Precision-Recall Area Under Curve (PR-AUC)**: Primary metric, focusing exclusively on positive class discovery.
- **ROC-AUC**: Discriminative separation capacity across all operating points.
- **Recall (Sensitivity)**: Minimizing fraudulent transactions that bypass the system.
- **Precision**: Minimizing unnecessary account freezes for legitimate cardholders.
- **F1-Score**: Harmonic balance between precision and recall.

### Summary Results (Test Set: 20,000 Transactions)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1-Score | Optimal Threshold |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.7014 | 0.0431 | 0.6180 | 0.0401 | 0.0753 | 0.50 |
| **LightGBM (Champion)** | **0.7912** | **0.0762** | **0.4453** | **0.0754** | **0.1290** | **0.35 - 0.54** |

### Threshold Tuning
Default classification thresholds (0.50) are rarely optimal under extreme class skew. By evaluating performance across increments of 0.01 between 0.05 and 0.95, the decision boundary is calibrated to optimize the business tradeoff between customer intervention and fraud containment.

---

## Model Interpretability (SHAP)

To prevent black-box vulnerability and ensure compliance with financial fair-lending and auditing regulations, the LightGBM model is analyzed using `shap.TreeExplainer`:

- **Global Feature Importance**: Identifies the primary features driving model decisions across the overall population. Top contributors include transaction amount spikes relative to 30-day moving averages, impossible travel indicators, velocity within the preceding hour, and composite risk proxies.
- **Local Attribution**: Each scored transaction can be accompanied by an individual SHAP contribution breakdown, detailing exactly which features raised or lowered the fraud probability for dispute resolution teams.

Artifacts generated:
- `Model/model_artifacts/08_shap_bar.png`
- `Model/model_artifacts/09_shap_beeswarm.png`

---

## Real-Time Inference Pipeline

The deployed architecture is fully self-contained. The following script demonstrates loading artifacts and scoring incoming transactions:

```python
import joblib
import json
import pandas as pd
import numpy as np

# 1. Load serialized assets
model = joblib.load('Model/model_artifacts/lgb_fraud_detector.pkl')

with open('Model/model_artifacts/feature_names.json', 'r') as f:
    feature_names = json.load(f)

with open('Model/model_artifacts/model_registry.json', 'r') as f:
    registry = json.load(f)

decision_threshold = registry.get('optimal_threshold', 0.54)

# 2. Prepare incoming transaction (example schema)
sample_transaction = {
    # Must contain all features in feature_names.json with appropriate preprocessing
    # and engineered fields computed prior to invocation.
}
input_df = pd.DataFrame([sample_transaction])[feature_names]

# 3. Generate inference
probability = float(model.predict_proba(input_df)[:, 1][0])
is_fraudulent = int(probability >= decision_threshold)

print(f"Fraud Probability: {probability:.4f}")
print(f"Classification: {'FRAUD' if is_fraudulent == 1 else 'LEGITIMATE'}")
```

---

## Installation and Getting Started

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/NumiKun/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Set Up a Virtual Environment
Using `venv` on Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

Using PowerShell on Windows:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline Notebook
Launch JupyterLab or Notebook interface:
```bash
jupyter lab
```
Navigate to `Model/credit_card_fraud_detection.ipynb` and execute cells sequentially.

---

## Artifacts and Documentation

For complete mathematical specifications, hyperparameter definitions, and deployment considerations, refer to:
- Technical Documentation: [MODEL_KNOWLEDGE.md](Model/model_artifacts/MODEL_KNOWLEDGE.md)
- Configuration Registry: [model_registry.json](Model/model_artifacts/model_registry.json)
- Feature Contract: [feature_names.json](Model/model_artifacts/feature_names.json)

---

## License and Citation

This project is released under the **MIT License**. Under the terms of this license and project guidelines, any party using, modifying, adapting, or distributing this codebase, models, or pipeline is required to provide appropriate attribution to the original author.

### Citation Format

```text
NumiKun (Rizki Nugroho), "Credit Card Fraud Detection Pipeline",
GitHub repository: https://github.com/NumiKun/Credit-Card-Fraud-Detection
```

Refer to the [LICENSE](LICENSE) file for complete legal terms.

---

## Author

- **NumiKun (Rizki Nugroho)**
- GitHub: [NumiKun](https://github.com/NumiKun)
- Email: nugrohorizki20@gmail.com
- Repository: [Credit-Card-Fraud-Detection](https://github.com/NumiKun/Credit-Card-Fraud-Detection)
