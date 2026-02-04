# Credit Risk Analysis using Machine Learning

This project focuses on predicting the credit risk level of clients by combining
application data with historical credit behavior. The goal is to classify clients
into **Trusted**, **Risky**, or **Bad** categories to support better credit
decision-making.

# Description

Credit risk assessment is a critical task for financial institutions such as banks
and lending platforms. Before approving a loan or credit card, institutions must
evaluate the likelihood that a client will default or exhibit risky repayment
behavior. Incorrect risk assessment can lead to financial losses or missed
business opportunities.

In real-world scenarios, credit risk data is often distributed across multiple
sources. Application data provides static information such as income, age,
employment status, and demographic details, while credit history data records a
client’s repayment behavior over time.

This project addresses that challenge by transforming monthly credit history
records into meaningful client-level risk labels and integrating them with
application data. The problem is formulated as a **multi-class classification**
task, where each client is categorized as **Trusted**, **Risky**, or **Bad** based
on the severity, frequency, and recency of past repayment behavior.

## Problem Statement

Given:
- Client application data (demographics and financial attributes)
- Historical monthly credit repayment records

Predict the **credit risk category** of a client as:
- **Trusted**
- **Risky**
- **Bad**

The primary objective is to correctly identify high-risk clients while minimizing
false negatives.


## Datasets

- **Application Record**
  - Contains demographic and financial information such as income, age,
    employment status, education, and family details.
- **Credit Record**
  - Contains monthly repayment status information for each client.

Each client may have multiple credit history records, making aggregation a key
challenge.

## Methodology

1. **Credit History Aggregation**
   - Monthly credit records were aggregated at the client level.
   - Risk labels were created using severity, frequency, and recency of
     delinquency.

2. **Label Creation**
   - A new target variable `TYPE_OF_CLIENT` was created with three classes:
     Trusted, Risky, and Bad.

3. **Data Preprocessing**
   - Application data was merged with the generated risk labels.
   - Categorical features were converted using one-hot encoding.
   - Numerical features (income, age, employment length) were normalized for
     scale-sensitive models.
4. **Modeling**
   - **Logistic Regression** was used as a baseline model.
   - **Random Forest Classifier** was used as the final model to capture
     non-linear relationships and handle class imbalance.

5. **Evaluation**
   - Models were evaluated using:
     - Confusion Matrix
     - Precision, Recall, and F1-score
     - Emphasis was placed on **recall for high-risk (Bad) clients** rather than
     overall accuracy.

## Models Used

- Logistic Regression (Baseline)
- Random Forest Classifier (Final Model)

Logistic Regression was trained on scaled numerical features, while Random Forest
was trained on unscaled features, following best practices for each model type.


## Results

- Logistic Regression provided a baseline understanding of the problem but showed
  limitations due to class imbalance and linear assumptions.
- Random Forest achieved high recall for Risky and Bad clients, making it suitable
  for credit risk screening tasks where identifying high-risk clients is critical.

CREDIT_RISK_ANALYSIS/

│
├── data/
├── src/ 
│ ├── data_loading.py
│ ├── label_creation.py
│ ├── preprocessing.py
│ ├── train.py
│ ├── evaluate.py
│ └── main.py
│
├── README.md
└── requirements.txt


## How to Run

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Key Concepts Demonstrated

Time-series aggregation

Credit risk modeling

Class imbalance handling

Feature encoding and normalization

Model comparison and evaluation

Modular and clean ML project structure

## Limitations and Future Work

Risk labels are rule-based and not sourced from ground-truth default records.

Future work could include:

Hyperparameter tuning

Cost-sensitive learning

Gradient boosting models

Probability-based risk scoring

## Author

Ishya Bommireddy