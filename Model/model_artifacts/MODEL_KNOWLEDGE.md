# Model Knowledge Base
## Credit Card Fraud Detection — LightGBM Pipeline

---

## Overview

This document captures the complete technical knowledge of the fraud detection machine learning pipeline built for `credit_card_fraud_detection_2026.csv`. It is intended to serve as the definitive reference for model reproducibility, deployment, and future iteration.

---

## 1. Problem Statement

| Attribute | Value |
|---|---|
| **Task Type** | Binary Classification |
| **Target Variable** | `is_fraud` (0 = Legitimate, 1 = Fraud) |
| **Dataset** | `credit_card_fraud_detection_2026.csv` |
| **Total Samples** | 100,000 transactions |
| **Fraud Rate** | 2.06% (~2,055 fraudulent transactions) |
| **Imbalance Ratio** | ~47.7 : 1 (Legitimate : Fraud) |
| **Primary Metric** | PR-AUC (Precision-Recall Area Under Curve) |
| **Secondary Metrics** | ROC-AUC, Recall, F1-Score |

**Why PR-AUC over ROC-AUC?**
In heavily imbalanced datasets, ROC-AUC can be misleadingly optimistic because it accounts for true negatives, which dominate the denominator. PR-AUC focuses exclusively on the minority class (fraud), making it a more honest reflection of model performance in this context.

---

## 2. Dataset Feature Catalogue

### 2.1 Transaction Features
| Feature | Type | Description |
|---|---|---|
| `transaction_amount` | float | Amount of the transaction in USD |
| `transaction_datetime` | datetime | Timestamp of transaction (dropped after datetime parsing) |
| `transaction_hour` | int | Hour of the transaction (0-23) |
| `transaction_day_of_week` | str | Day of the week |
| `is_weekend` | int | Binary: 1 if weekend |
| `card_present` | int | Binary: 1 if physical card was present |
| `transaction_channel` | str | Online, In-store, ATM, Mobile App |
| `device_type` | str | Mobile, Desktop, POS Terminal, ATM |
| `browser` | str | Browser used (N/A for physical transactions) |
| `used_chip` | int | Binary: 1 if EMV chip was used |
| `used_pin` | int | Binary: 1 if PIN was entered |

### 2.2 Merchant Features
| Feature | Type | Description |
|---|---|---|
| `merchant_id` | str | Unique merchant identifier (dropped) |
| `merchant_category` | str | Business category of the merchant |
| `merchant_country` | str | Country where the merchant is located |
| `international_transaction` | int | Binary: 1 if merchant country differs from home country |
| `high_risk_country` | int | Binary: 1 if merchant country is flagged as high-risk |

### 2.3 Customer & Account Features
| Feature | Type | Description |
|---|---|---|
| `customer_id` | str | Unique customer identifier (dropped) |
| `card_id` | str | Unique card identifier (dropped) |
| `customer_age` | int | Age of the cardholder |
| `customer_gender` | str | Gender of the cardholder |
| `customer_state` | str | US state of the cardholder |
| `account_age_days` | int | Age of the account in days |
| `credit_limit` | float | Card credit limit in USD |
| `credit_score` | int | Customer credit score |
| `num_cards_held` | int | Total number of cards held by the customer |
| `email_domain` | str | Email domain used for account registration |

### 2.4 Behavioral / Velocity Features
| Feature | Type | Description |
|---|---|---|
| `transactions_last_1h` | int | Number of transactions in the last 1 hour |
| `transactions_last_24h` | int | Number of transactions in the last 24 hours |
| `transactions_last_7d` | int | Number of transactions in the last 7 days |
| `amount_last_1h` | float | Total amount transacted in the last 1 hour |
| `amount_last_24h` | float | Total amount transacted in the last 24 hours |
| `days_since_last_transaction` | float | Days elapsed since the previous transaction |
| `distance_from_home_km` | float | Distance from the customer's home to the merchant |
| `distance_from_last_tx_km` | float | Distance from the last transaction to the current one |
| `avg_transaction_amount_30d` | float | Average transaction amount over 30 days |
| `std_transaction_amount_30d` | float | Std. deviation of transaction amounts over 30 days |
| `amount_vs_avg_ratio` | float | Ratio of current amount to 30-day average |

### 2.5 Risk / History Features
| Feature | Type | Description |
|---|---|---|
| `num_declined_last_30d` | int | Number of declined transactions in last 30 days |
| `num_chargebacks_ever` | int | Total chargebacks the customer has ever filed |

### 2.6 Engineered Features (Created in Pipeline)
| Feature | Formula / Logic | Signal |
|---|---|---|
| `velocity_ratio_1h_24h` | `transactions_last_1h / transactions_last_24h` | Detects sudden transaction bursts |
| `amount_spike_ratio` | `transaction_amount / avg_transaction_amount_30d` | Flags unusually large individual transactions |
| `impossible_travel_flag` | `1 if distance_from_last_tx_km > 500 AND days_since_last_tx < 1` | Physically impossible travel between transactions |
| `risk_score_proxy` | Weighted sum of `high_risk_country`, `international_transaction`, `num_chargebacks_ever`, `num_declined_last_30d` | Composite behavioral risk score |
| `credit_utilization_ratio` | `transaction_amount / credit_limit` | Normalized spend as fraction of available credit |
| `security_strength` | `used_chip + used_pin + card_present` | Authentication robustness (0-3) |

### 2.7 Dropped Features
| Feature | Reason |
|---|---|
| `transaction_id`, `customer_id`, `card_id`, `merchant_id` | Unique identifiers — no predictive signal |
| `fraud_type` | Target leakage — derived from `is_fraud` |
| `fraud_amount_lost` | Target leakage — nonzero only when `is_fraud == 1` |
| `transaction_datetime` | Redundant — temporal signal already captured by `transaction_hour` and `transaction_day_of_week` |

---

## 3. Data Preprocessing

### 3.1 Missing Value Treatment
| Column | Missing Rate | Treatment | Rationale |
|---|---|---|---|
| `browser` | ~41.7% | Imputed with `'N/A'` | Missingness is structural — ATM/POS transactions have no browser. It is an informative signal, not noise. |

### 3.2 Encoding
- All categorical string columns are **Label Encoded** using `sklearn.preprocessing.LabelEncoder`.
- Encoding is fit on the training set only and applied to the test set to prevent leakage.

### 3.3 Train/Test Split
- **Split ratio**: 80% train / 20% test
- **Method**: Stratified split (`stratify=y`) to preserve the 2.06% fraud rate in both splits
- **Random state**: 42

---

## 4. Class Imbalance Handling

### 4.1 Method: SMOTE
- **Library**: `imblearn.over_sampling.SMOTE`
- **Parameters**: `k_neighbors=5`, `random_state=42`
- **Application scope**: Training set ONLY — never applied to the test set
- **Result**: Training set balanced from ~47.7:1 to 1:1 (Legitimate : Fraud)

### 4.2 Additional: scale_pos_weight (LightGBM)
- Even after SMOTE, `scale_pos_weight` is set to the post-SMOTE class ratio to further reinforce fraud recall during gradient boosting optimization.

---

## 5. Models

### 5.1 Logistic Regression (Baseline)
| Parameter | Value |
|---|---|
| `C` (regularization strength) | 0.1 |
| `solver` | saga |
| `penalty` | l2 |
| `max_iter` | 1000 |
| `random_state` | 42 |

**Preprocessing required**: StandardScaler applied before training. The scaler is fit on the SMOTE-resampled training data and saved as `feature_scaler.pkl`.

### 5.2 LightGBM (Champion Model)
| Parameter | Value | Rationale |
|---|---|---|
| `objective` | binary | Binary classification task |
| `metric` | binary_logloss, auc | Multi-metric monitoring |
| `boosting_type` | gbdt | Standard gradient boosting |
| `num_leaves` | 63 | Controls tree complexity |
| `max_depth` | 7 | Limits overfitting |
| `learning_rate` | 0.05 | Slower learning for better generalization |
| `n_estimators` | 600 (with early stopping) | Upper bound; early stopping at 50 |
| `min_child_samples` | 30 | Minimum samples per leaf |
| `subsample` | 0.8 | Row subsampling per tree |
| `colsample_bytree` | 0.8 | Feature subsampling per tree |
| `reg_alpha` | 0.1 | L1 regularization |
| `reg_lambda` | 0.1 | L2 regularization |
| `scale_pos_weight` | Computed dynamically | Handles residual class imbalance |
| `random_state` | 42 | Reproducibility |

**Early stopping**: Triggered if validation AUC does not improve for 50 consecutive rounds.

---

## 6. Evaluation Framework

### 6.1 Metrics Used
| Metric | Description | Why It Matters in Fraud Detection |
|---|---|---|
| **ROC-AUC** | Area under the Receiver Operating Characteristic curve | Global discriminative ability of the model |
| **PR-AUC** | Area under the Precision-Recall curve | Primary metric for imbalanced datasets — directly reflects minority class performance |
| **Recall** | True Positive Rate (TPR) for fraud class | Minimizes missed frauds (False Negatives) — the most costly error |
| **Precision** | Fraction of flagged transactions that are actual fraud | Minimizes customer friction from false alerts (False Positives) |
| **F1-Score** | Harmonic mean of Precision and Recall | Balanced composite metric |
| **Confusion Matrix** | Absolute counts of TP, TN, FP, FN | Operational understanding of error types |

### 6.2 Decision Threshold
- Default sklearn threshold of 0.5 is suboptimal for imbalanced fraud data.
- The optimal threshold is identified by sweeping all values from 0.05 to 0.95 in 0.01 steps and selecting the value that **maximizes F1-Score** on the test set.
- This threshold is stored in `model_registry.json` and must be applied during inference.

---

## 7. SHAP Explainability

### 7.1 Method
- **Explainer**: `shap.TreeExplainer` — exact Shapley values for tree-based models
- **Sample size**: 2,000 randomly sampled test observations
- **Output**: SHAP values quantifying each feature's additive contribution to the model output (log-odds)

### 7.2 Interpretation
- Positive SHAP value = feature increases the probability of fraud
- Negative SHAP value = feature decreases the probability of fraud
- Features ranked by mean absolute SHAP value provide global importance
- Individual SHAP values support per-transaction explanation (e.g., for customer support or regulatory audit)

---

## 8. Artifacts

| File | Description |
|---|---|
| `lgb_fraud_detector.pkl` | Trained LightGBM classifier (compressed, joblib format) |
| `feature_scaler.pkl` | StandardScaler fit on SMOTE-resampled training data (for Logistic Regression inference) |
| `feature_names.json` | Ordered list of feature names expected by the model at inference time |
| `model_registry.json` | Full model metadata: hyperparameters, metrics, version, training timestamp |
| `01_class_distribution.png` | EDA: Class balance bar chart and pie chart |
| `02_fraud_type_distribution.png` | EDA: Breakdown of fraud sub-types |
| `03_feature_distributions.png` | EDA: Numerical feature density plots by class |
| `04_fraud_rate_by_category.png` | EDA: Fraud rate across categorical features |
| `05_correlation_heatmap.png` | EDA: Pearson correlation matrix of numerical features |
| `06_smote_balance.png` | Before/after SMOTE class distribution comparison |
| `07_model_comparison.png` | Bar chart comparison of all models across all metrics |
| `eval_logistic_regression.png` | LR: ROC curve, PR curve, Confusion matrix |
| `eval_lightgbm.png` | LightGBM: ROC curve, PR curve, Confusion matrix |
| `08_shap_bar.png` | SHAP: Global feature importance (bar chart) |
| `09_shap_beeswarm.png` | SHAP: Feature impact distribution (beeswarm plot) |
| `10_threshold_analysis.png` | Threshold sweep: Precision, Recall, F1 vs. threshold |

---

## 9. Inference Guide

### 9.1 Loading and Using the Model

```python
import joblib
import json
import numpy as np

model = joblib.load('model_artifacts/lgb_fraud_detector.pkl')
with open('model_artifacts/feature_names.json') as f:
    feature_names = json.load(f)
with open('model_artifacts/model_registry.json') as f:
    registry = json.load(f)

optimal_threshold = registry['optimal_threshold']

def predict_fraud(transaction_df):
    transaction_df = transaction_df[feature_names]
    fraud_prob     = model.predict_proba(transaction_df)[:, 1]
    fraud_decision = (fraud_prob >= optimal_threshold).astype(int)
    return fraud_prob, fraud_decision
```

### 9.2 Feature Order Requirement
The input DataFrame **must contain all features in the exact order** listed in `feature_names.json`. Any discrepancy in feature order or missing features will cause the model to produce incorrect predictions.

### 9.3 Preprocessing Before Inference
The LightGBM model expects:
- `browser` missing values filled with `'N/A'`
- All categorical columns **label-encoded** with the same encoding scheme used during training
- All 6 engineered features computed and appended

---

## 10. Known Limitations and Future Work

| Limitation | Recommended Mitigation |
|---|---|
| Label Encoder may fail on unseen categories at inference time | Replace with `OrdinalEncoder` with `handle_unknown='use_encoded_value'` |
| Model not re-trained on temporal splits | Use time-based train/test splits to simulate real-world deployment |
| SMOTE creates synthetic samples in feature space, not real transactions | Explore ADASYN or GAN-based augmentation for more realistic minority samples |
| Static threshold | Implement dynamic threshold adjustment based on rolling fraud rate or business cost function |
| No concept drift detection | Integrate PSI (Population Stability Index) monitoring on model score distribution |

---

*Last updated: 2026-10-13*
*Pipeline version: 1.0.0*
