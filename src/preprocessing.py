import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(data_path='../data/credit_risk_dataset.csv', scale=True):
    """
    Load original data, preprocess, and optionally apply scaling.
    scale=True is recommended for Logistic Regression and SVM.
    """
    print("Loading and preprocessing data...")

    # Load original dataset
    df = pd.read_csv(data_path)

    # Step 1: Handle missing values
    df['person_emp_length'] = df['person_emp_length'].fillna(df['person_emp_length'].median())
    df['loan_int_rate'] = df['loan_int_rate'].fillna(df['loan_int_rate'].median())

    # Step 2: One-Hot Encoding for categorical columns
    categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features and target
    X = df_encoded.drop('loan_status', axis=1)
    y = df_encoded['loan_status']

    # Step 3: Feature Scaling
    if scale:
        scaler = StandardScaler()
        # Scale only numerical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        print("Feature scaling applied (StandardScaler)")

    print("Preprocessing completed!")
    print("Final shape:", df_encoded.shape)

    return X, y, df_encoded


# Optional: Function to save preprocessed data
def save_preprocessed_data(df_encoded, output_path='../data/credit_risk_preprocessed.csv'):
    df_encoded.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")