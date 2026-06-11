import pandas as pd
from src.data_validation import validate_customer_data

df = pd.read_csv("data/sample_customer_data.csv")

result = validate_customer_data(df)

print("Validation Passed:", result)