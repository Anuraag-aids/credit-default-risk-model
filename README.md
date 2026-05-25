# Credit Default Risk Assessment — Vitto DS Internship Assignment

> End-to-end machine learning pipeline for predicting credit card default using the **UCI Credit Card Default Dataset** (30,000 clients · Taiwan · 2005). Built to simulate how a data-driven FinTech builds credit-risk signals from real borrower behaviour.

-----

## Table of Contents

1. [Project Overview](#project-overview)
1. [Dataset](#dataset)
1. [Project Structure](#project-structure)
1. [Setup & Reproduction](#setup--reproduction)
1. [Methodology](#methodology)
1. [Feature Engineering](#feature-engineering)
1. [Models & Evaluation](#models--evaluation)
1. [Key Findings](#key-findings)
1. [Business Recommendations](#business-recommendations)
1. [Data Quality Decisions](#data-quality-decisions)
1. [Deliverables Checklist](#deliverables-checklist)

-----

## Project Overview

This project performs an end-to-end credit risk analysis — covering exploratory data analysis, feature engineering, classification modelling, and business communication — as if presenting findings to a credit risk team at a digital lender.

**Core questions answered:**

- What behavioural signals most strongly predict default?
- Can we build a reliable classifier despite ~22% class imbalance?
- What concrete actions should a credit team take?

-----

## Dataset

|Attribute        |Detail                                                                                                   |
|-----------------|---------------------------------------------------------------------------------------------------------|
|**Source**       |[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)|
|**Records**      |30,000 credit card clients                                                                               |
|**Features**     |24 input variables + 1 binary target                                                                     |
|**Period**       |April – September 2005                                                                                   |
|**Geography**    |Taiwan                                                                                                   |
|**Target**       |`default.payment.next.month` (1 = default, 0 = no default)                                               |
|**Class balance**|~22% default / ~78% no-default                                                                           |

### Key Variable Groups

|Group           |Variables                            |Description                                                |
|----------------|-------------------------------------|-----------------------------------------------------------|
|Demographics    |`SEX`, `EDUCATION`, `MARRIAGE`, `AGE`|Client profile                                             |
|Credit profile  |`LIMIT_BAL`                          |Credit limit in NT dollars                                 |
|Repayment status|`PAY_0` – `PAY_6`                    |Monthly delay status (−1 = paid duly, 1–9 = months delayed)|
|Bill amounts    |`BILL_AMT1` – `BILL_AMT6`            |Statement balance per month                                |
|Payment amounts |`PAY_AMT1` – `PAY_AMT6`              |Actual payment made per month                              |


> **Note:** `PAY_x` contains undocumented values: `−2` (no consumption) alongside `−1` (paid duly). `EDUCATION` contains undocumented values 0, 5, 6. Handling of these is documented in [Data Quality Decisions](#data-quality-decisions).

-----

## Project Structure

```text
credit-default-risk-model/
│
├── data/
│   └── UCI_Credit_Card.csv          # Raw dataset (not tracked in git — see setup)
│
├── notebooks/
│   └── credit_default_analysis.ipynb  # Main analysis notebook (fully reproducible)
│
├── plots/                            # All generated visualisations (PNG)
│   ├── class_balance.png
│   ├── distributions.png
│   ├── repayment_delay_by_default.png
│   ├── correlation_heatmap.png
│   ├── default_rate_by_demographics.png
│   ├── roc_curve.png
│   └── confusion_matrix.png
│
├── src/
│   └── features.py                  # Feature engineering functions
│
├── README.md
├── requirements.txt
└── writeup_report.pdf               # 1–2 page summary of approach and findings
```

-----

## Setup & Reproduction

**1. Clone the repository**

```bash
git clone https://github.com/Anuraag-aids/credit-default-risk-model.git
cd credit-default-risk-model
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from the [UCI repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) and place the CSV at:

```
data/UCI_Credit_Card.csv
```

**4. Run the notebook**

```bash
jupyter notebook notebooks/credit_default_analysis.ipynb
```

> The notebook is designed to run end-to-end without errors on a clean Python environment. All cells execute top-to-bottom with no hidden state.

### Requirements

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
jupyter
```

-----

## Methodology

### 1. Exploratory Data Analysis

- Shape, dtypes, null counts, and class balance check
- Distribution plots for `LIMIT_BAL`, `AGE`, and `PAY_0`
- Default rate comparison across `SEX`, `EDUCATION`, `MARRIAGE`, and `AGE` bands
- Repayment delay visualisation across `PAY_0`–`PAY_6` by default outcome
- Correlation heatmap with top 5 features associated with default identified

### 2. Feature Engineering

Three behavioural risk indicators engineered from raw repayment history — see [Feature Engineering](#feature-engineering).

### 3. Encoding

- `EDUCATION`: values 0, 5, 6 merged into `other` category (undocumented in original schema)
- `MARRIAGE`: value 0 merged into `other`
- All categoricals label-encoded for model compatibility

### 4. Class Imbalance

Default rate is ~22% — addressed using `class_weight='balanced'` in Logistic Regression and XGBoost. Chosen over SMOTE to avoid synthetic data leakage across cross-validation folds.

### 5. Validation Strategy

Stratified 5-fold cross-validation used throughout to preserve class ratio in each fold.

-----

## Feature Engineering

|Feature             |Formula                                           |Interpretation                                            |
|--------------------|--------------------------------------------------|----------------------------------------------------------|
|`AVG_UTIL_RATE`     |`mean(BILL_AMTx / LIMIT_BAL)` across 6 months     |Sustained credit utilisation pressure                     |
|`AVG_PAY_RATIO`     |`mean(PAY_AMTx / BILL_AMTx)` where `BILL_AMTx > 0`|Repayment consistency across billing cycles               |
|`TOTAL_DELAY_MONTHS`|Count of `PAY_x > 0` values                       |Cumulative delinquency — total months of delayed repayment|

All feature choices are justified in markdown cells within the notebook.

-----

## Models & Evaluation

### Logistic Regression (Baseline)

Interpretable linear baseline. Used to establish statistical significance of key features and set a performance floor.

### XGBoost Classifier (Primary Model)

Gradient boosted trees — captures non-linear interactions in repayment behaviour that a linear model misses.

### Metrics

|Metric                 |Why it matters here                                 |
|-----------------------|----------------------------------------------------|
|**Precision**          |Avoids falsely flagging good borrowers              |
|**Recall**             |Catches as many true defaulters as possible         |
|**F1 Score**           |Balances precision and recall for imbalanced classes|
|**ROC-AUC**            |Overall discrimination capability                   |
|**CV AUC (mean ± std)**|Robustness across folds — guards against overfitting|

Outputs include: ROC curve, confusion matrix, and top 5 feature importance scores from the final model.

-----

## Key Findings

- **Repayment delay (`PAY_0`, `PAY_2`, `PAY_3`) are the strongest default predictors** — far more predictive than demographic variables.
- Customers with **repeated delinquency across multiple months** show substantially elevated default probability.
- **High credit utilisation** (`AVG_UTIL_RATE`) correlates strongly with financial stress and default risk.
- **Low repayment ratios** (`AVG_PAY_RATIO`) — paying only minimum amounts — signal high latent risk even before delays appear.
- Tree-based models outperform logistic regression on this structured financial dataset.
- Demographic variables (`SEX`, `MARRIAGE`) show limited predictive lift once behavioural features are included.

-----

## Business Recommendations

**1. Flag repeat delinquents for proactive intervention**
Customers with 2+ months of repayment delays (`TOTAL_DELAY_MONTHS ≥ 2`) should be escalated to a risk monitoring queue. Early outreach — before a third missed payment — significantly reduces loss exposure.

**2. Implement dynamic credit limits tied to utilisation and repayment consistency**
Customers consistently above 80% utilisation with declining `AVG_PAY_RATIO` should have credit limit increases paused and trigger a soft review. This is a leading indicator — acting here is cheaper than recovery post-default.

-----

## Data Quality Decisions

|Issue                        |Decision                    |Rationale                                                                |
|-----------------------------|----------------------------|-------------------------------------------------------------------------|
|`EDUCATION` values 0, 5, 6   |Merged into `other` category|Undocumented in original schema; too few observations to model separately|
|`MARRIAGE` value 0           |Merged into `other`         |Same as above                                                            |
|`PAY_x = −2` (no consumption)|Treated as `−1` (no delay)  |Semantically equivalent — no outstanding balance, no delinquency         |
|Negative `BILL_AMT` values   |Flagged and retained as-is  |Represent overpayments (credit balance) — economically valid, not errors |
|No null values found         |No imputation needed        |Dataset is complete                                                      |

-----

## Deliverables Checklist

- [x] GitHub repository with clean commit history
- [x] Jupyter notebook — fully reproducible, end-to-end
- [x] README (this document)
- [x] Write-up PDF (`writeup_report.pdf`)
- [x] Minimum 5 publication-quality plots
- [x] ROC curve and confusion matrix
- [x] 200–300 word non-technical summary (in notebook)
- [x] Two concrete business actions stated

-----

## Author

**Anuraag Juneja**  
Bachelor’s in Artificial Intelligence & Data Science

-----

*Submitted as part of the Vitto Data Science Internship Technical Assessment.*