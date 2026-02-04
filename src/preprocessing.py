
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(application_df, labels_df):
    df = application_df.merge(labels_df, on='ID', how='inner')

   
    df['AGE'] = -df['DAYS_BIRTH'] / 365
    df['YEARS_EMPLOYED'] = -df['DAYS_EMPLOYED'] / 365
    df['YEARS_EMPLOYED'].replace(365243 / 365, np.nan, inplace=True)

    df.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED'], inplace=True)

    y = df['TYPE_OF_CLIENT']
    X = df.drop(columns=['ID', 'TYPE_OF_CLIENT'])

    
    X = pd.get_dummies(X, drop_first=True)

    num_cols = ['AMT_INCOME_TOTAL', 'AGE', 'YEARS_EMPLOYED']

   
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])

   
    X_unscaled = X.copy()

    return X_scaled, X_unscaled, y