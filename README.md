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

