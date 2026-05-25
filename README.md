# Credit Default Risk Assessment for Digital Lending

> A production-grade machine learning pipeline to analyze, model, and predict customer credit default behavior using the **UCI Credit Card Default Dataset**.

-----

## Project Objective

Identify high-risk borrower behavior patterns and build predictive models that help digital lenders proactively manage credit risk exposure.

**The analysis focuses on:**

- Behavioral repayment trends
- Credit utilization patterns
- Delinquency history
- Predictive default risk modeling

-----

## Dataset

|Attribute     |Detail                                |
|--------------|--------------------------------------|
|**Source**    |UCI Machine Learning Repository       |
|**Records**   |30,000 customer records               |
|**Features**  |25 financial and demographic variables|
|**Population**|Taiwan credit card clients (2005)     |

**Dataset Link:** <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients>

-----

## Project Structure

```text
credit-default-risk-model/
│
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter analysis notebooks
├── plots/                 # Generated visualizations
├── src/                   # Source code and pipeline scripts
├── README.md
├── requirements.txt
└── writeup_report.pdf
```

-----

## Technologies Used

|Category           |Tools                |
|-------------------|---------------------|
|**Language**       |Python               |
|**Data Processing**|Pandas, NumPy        |
|**Modeling**       |Scikit-learn, XGBoost|
|**Visualization**  |Matplotlib, Seaborn  |
|**Version Control**|Git & GitHub         |

-----

## Exploratory Data Analysis

Key analysis performed:

- Class imbalance inspection
- Distribution analysis of financial variables
- Repayment delay visualization
- Correlation heatmap analysis
- Default rate comparison across demographic groups

-----

## Feature Engineering

Three behavioral risk indicators were engineered from raw repayment history:

### `AVG_UTIL_RATE`

Average credit utilization across 6 months — captures sustained financial stress.

### `AVG_PAY_RATIO`

Repayment consistency ratio across billing cycles — measures reliability of repayment behavior.

### `TOTAL_DELAY_MONTHS`

Cumulative repayment delinquency signal — aggregates the severity of missed payments over time.

-----

## Machine Learning Models

### Logistic Regression

Used as an interpretable baseline to establish statistical significance of key features.

### XGBoost Classifier *(Primary Model)*

Captures non-linear repayment behavior patterns with significantly higher predictive power.

**Evaluation Metrics:**

|Metric     |Description                                          |
|-----------|-----------------------------------------------------|
|Precision  |Proportion of flagged defaults that are true defaults|
|Recall     |Proportion of actual defaults correctly identified   |
|F1 Score   |Harmonic mean of precision and recall                |
|ROC-AUC    |Overall model discrimination capability              |
|CV Strategy|Stratified 5-Fold Cross Validation                   |

-----

## Key Findings

- **Repayment delay behavior** is the strongest single predictor of default risk.
- Customers with **repeated delinquency patterns** exhibit substantially higher default probability.
- **High utilization ratios** correlate strongly with elevated financial stress.
- **Tree-based models** consistently outperform linear baselines on structured financial data.

-----

## Business Recommendations

1. **Proactive Risk Flagging** — Flag customers with repeated repayment delays for early intervention before default occurs.
1. **Dynamic Credit Policies** — Implement adaptive credit limit policies driven by real-time utilization and repayment consistency signals.

-----

## Setup Instructions

**Clone the repository:**

```bash
git clone https://github.com/Anuraag-aids/credit-default-risk-model.git
cd credit-default-risk-model
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Launch the notebook:**

```bash
jupyter notebook
```

-----

## Author

**Anuraag Juneja**  
Bachelor’s in Artificial Intelligence & Data Science

-----

*Built to demonstrate end-to-end ML pipeline development for real-world credit risk applications.*