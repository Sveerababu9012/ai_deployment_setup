import pandas as pd


def clean_dataframe(df: pd.DataFrame, columns_to_fill_mean: list) -> pd.DataFrame:
    df_cleaned = df.copy()

    for col in columns_to_fill_mean:
        if col in df_cleaned.columns:
            mean_value = df_cleaned[col].mean()

            # FIX
            df_cleaned[col] = df_cleaned[col].fillna(mean_value)

    return df_cleaned


def normalize_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    df_normalized = df.copy()

    if column_name in df_normalized.columns:
        min_val = df_normalized[column_name].min()
        max_val = df_normalized[column_name].max()

        if max_val - min_val != 0:
            df_normalized[column_name] = (
                df_normalized[column_name] - min_val
            ) / (max_val - min_val)
        else:
            df_normalized[column_name] = 0.0

    return df_normalized