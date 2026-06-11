import pandas as pd


class DataValidationError(Exception):
    pass


def validate_customer_data(df):
    expected_columns = [
        "CustomerID",
        "Age",
        "Gender",
        "SubscriptionMonths",
        "MonthlyCharges",
        "Churn",
    ]

    for col in expected_columns:
        if col not in df.columns:
            raise DataValidationError(f"Missing column: {col}")

    if df["CustomerID"].duplicated().any():
        raise DataValidationError("Duplicate CustomerID found")

    if (df["Age"] < 0).any():
        raise DataValidationError("Invalid Age found")

    if df.isnull().sum().sum() > 0:
        raise DataValidationError("Missing values found")

    return True