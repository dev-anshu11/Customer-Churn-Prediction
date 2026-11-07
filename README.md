# Customer Churn Prediction

A machine learning project that predicts which telecom customers are likely to leave and explores customer characteristics associated with churn.

## 1. Problem Statement

Customer churn can be costly for telecom companies. A churn prediction model can help a retention team identify customers who may need attention.

The model is a screening tool, not proof that a customer will leave.

## 2. Dataset

- **Dataset:** IBM Telco Customer Churn sample
- **Total records:** 7,043 customers
- **Total columns:** 21
- **Dataset file:** `telco.csv`

## 3. Approach

The project follows these steps:

- Data cleaning and preprocessing.
- Converted `TotalCharges` into numeric format.
- Handled 11 missing values.
- Removed the customer ID column.
- Performed exploratory data analysis (EDA).
- Created additional features and encoded categorical variables.
- Trained Logistic Regression and Random Forest models.
- Used class weighting to address the imbalanced target variable.
- Evaluated the models using 5-fold cross-validation and a separate test set.

A stratified 80/20 train-test split was used. The held-out test set contains 1,409 customers.

For Logistic Regression, scaling was placed inside a Scikit-learn Pipeline during cross-validation to prevent preprocessing leakage between folds.

## 4. Cross-Validation Results

The following are mean scores across 5 folds on the training data.

| Model | Mean ROC-AUC | Mean Recall |
|---|---:|---:|
| Logistic Regression | 0.845 | 0.801 |
| Random Forest | 0.847 | 0.765 |

**Observation:**

- Random Forest achieved a slightly higher mean ROC-AUC.
- Logistic Regression achieved a higher mean recall.
- The ROC-AUC difference between the models is small.

## 5. Held-Out Test Results

The following results were calculated using the separate test set of 1,409 customers.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.740 | 0.506 | 0.786 | 0.616 | 0.842 |
| Random Forest | 0.761 | 0.534 | 0.778 | 0.633 | 0.844 |

The overall churn rate in the dataset is approximately 26.5%.

An always-"no churn" baseline would achieve approximately 73.5% accuracy but would identify none of the customers who actually churned. Therefore, accuracy alone is not sufficient for evaluating this problem.

## 6. Model Comparison and Decision

On the held-out test set:

- Logistic Regression identified 294 of the 374 actual churners.
- Random Forest identified 291 of the 374 actual churners.
- Logistic Regression identified 3 additional churners.
- However, Logistic Regression also generated 33 additional false-positive flags (287 compared with 254).

This means approximately 11 additional false-positive flags for each additional churner identified in this test-set comparison.

I would currently select Logistic Regression for further review because it achieved higher mean recall during cross-validation and its coefficients are easier to interpret.

However, the difference between the models is small. Whether this trade-off is worthwhile depends on the cost of missing a churner compared with the cost of contacting a customer who would not have churned. I have not tested those business costs.

## 7. Key Exploratory Findings

- Overall customer churn rate: 26.5%.
- Month-to-month contracts had a churn rate of 42.7%, compared with 2.8% for two-year contracts.
- Churned customers had a median tenure of 10 months, compared with 38 months for retained customers.
- Fiber optic users had a churn rate of 41.9%.
- Electronic-check payers had a churn rate of 45.3%.
- Customers with tech support or online security had lower churn rates in this dataset.

These findings show associations in the dataset and do not establish that any particular factor causes customer churn.

## 8. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

## 9. How to Run the Project

Clone the repository:

```bash
git clone https://github.com/dev-anshu11/Customer-Churn-Prediction.git
```

Move into the project directory:

```bash
cd Customer-Churn-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Customer_Churn_Prediction.ipynb` and run the notebook cells.

## 10. Conclusion

This project compares Logistic Regression and Random Forest for telecom customer churn prediction.

The cross-validation results show that Logistic Regression achieved higher mean recall, while Random Forest achieved slightly
