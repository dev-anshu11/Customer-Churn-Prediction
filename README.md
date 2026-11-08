# Customer Churn Prediction

Machine learning project that predicts which telecom customers are likely to leave, and shows which customer traits go with higher churn.

## Problem
Losing customers is expensive. Predicting churn early lets a retention team contact at-risk customers with offers or support.

## Dataset
Telco Customer Churn (IBM sample dataset): 7,043 customers, 21 columns. Included as `telco.csv`.

## Approach
1. Data cleaning (fixed TotalCharges data type, handled 11 missing values, dropped ID column)
2. Exploratory data analysis (churn by contract, internet service, payment method, tenure, charges)
3. Feature engineering (average monthly charges, number of add-on services) and one-hot encoding
4. Trained Logistic Regression and Random Forest with class weighting for the imbalanced target
5. Evaluated with accuracy, precision, recall, F1-score, ROC-AUC and 5-fold cross-validation

## Results (test set)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.740 | 0.506 | 0.786 | 0.616 | 0.842 |
| Random Forest | 0.761 | 0.534 | 0.778 | 0.633 | 0.844 |

I held out 20% of the data (1,409 of 7,043 customers) as a test set, stratified by churn, so the metrics come from customers the models didn't train on.

An always-"no churn" baseline would score about 73.5% accuracy but find none of the churners. Accuracy is only 0.761 for the Random Forest, so I judge the models on recall and ROC-AUC.

**Which model I'd give a team:** Logistic Regression. It finds slightly more of the customers who leave (recall 0.786 against 0.778), and its ROC-AUC is within 0.002 of the Random Forest's. The cost is a lower precision (0.506), so just over half of the customers it flags won't actually leave. I chose it assuming a missed churner costs more than a wasted retention call. I haven't tested that assumption.

**How I chose:** I compared both models on the same test set and then picked Logistic Regression, so its test scores may be slightly optimistic. The gap between the models is only 0.002 ROC-AUC, so the effect is small.

## Key insights
- Overall churn rate is 26.5%.
- Month-to-month contracts churn at 42.7%, against 2.8% for two-year contracts.
- Churned customers have a median tenure of 10 months, against 38 for retained customers.
- Fiber optic users (41.9%) and electronic check payers (45.3%) churn the most.
- Customers with tech support or online security churn far less.

## How to run
```
pip install -r requirements.txt
jupyter notebook
```

## Tech stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
