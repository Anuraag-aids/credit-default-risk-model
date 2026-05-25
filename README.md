# Credit Default Risk Assessment for Digital Lending

A production-grade machine learning pipeline developed to analyze, model, and predict customer credit default behavior using the UCI Credit Card Default dataset.

---

## Project Objective

The goal of this project is to identify high-risk borrower behavior patterns and build predictive models that help digital lenders proactively manage credit risk exposure.

The analysis focuses on:
- Behavioral repayment trends
- Credit utilization patterns
- Delinquency history
- Predictive default risk modeling

---

## Dataset

Source:
UCI Machine Learning Repository — Default of Credit Card Clients Dataset

- 30,000 customer records
- 25 financial and demographic variables
- Taiwan credit card clients (2005)

Dataset Link:
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

---

## Project Structure

```text
credit-default-risk-model/
│
├── data/
├── notebooks/
├── plots/
├── src/
├── README.md
├── requirements.txt
└── writeup_report.pdf
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Git & GitHub

---

## Exploratory Data Analysis

Key analysis performed:
- Class imbalance inspection
- Distribution analysis of financial variables
- Repayment delay visualization
- Correlation heatmap analysis
- Default rate comparison across demographic groups

---

## Feature Engineering

Engineered behavioral risk indicators include:

### AVG_UTIL_RATE
Average credit utilization across 6 months.

### AVG_PAY_RATIO
Repayment consistency ratio across billing cycles.

### TOTAL_DELAY_MONTHS
Cumulative repayment delinquency signal.

---

## Machine Learning Models

### Logistic Regression
Used as an interpretable baseline model.

### XGBoost Classifier
Implemented as the primary predictive model for capturing non-linear repayment behavior patterns.

Evaluation Metrics:
- Precision
- Recall
- F1 Score
- ROC-AUC
- Stratified 5-Fold Cross Validation

---

## Key Findings

- Repayment delay behavior is the strongest predictor of default risk.
- Customers with repeated delinquency patterns exhibit substantially higher default probability.
- High utilization ratios correlate strongly with elevated financial stress.
- Tree-based models outperform linear baselines on structured financial data.

---

## Business Recommendations

1. Flag customers with repeated repayment delays for proactive risk monitoring.

2. Implement dynamic credit limit policies based on utilization and repayment consistency.

---

## Setup Instructions

Clone repository:

```bash
git clone https://github.com/Anuraag-aids/credit-default-risk-model.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch notebook:

```bash
jupyter notebook
```

---

## Author

Anuraag Juneja
Bachelor's in Artificial Intelligence & Data Science