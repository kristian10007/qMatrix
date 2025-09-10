import pandas as pd

def q_function(df: pd.DataFrame, col_A_name: str, col_B_name: str) -> float:
    """
    Calculates the Q-score for two columns, measuring the logical dependency
    of col_B on col_A. A score of 0 indicates perfect functional dependency (A -> B).
    """
    if col_A_name not in df.columns or col_B_name not in df.columns:
        raise ValueError("One or both column names are not in the DataFrame.")

    unique_A = df[col_A_name].dropna().unique()
    unique_B = df[col_B_name].dropna().unique()

    num_unique_A = len(unique_A)
    num_unique_B = len(unique_B)

    if num_unique_A == 0 or num_unique_B <= 1:
        return 0.0

    num_observed_pairs = df[[col_A_name, col_B_name]].dropna().drop_duplicates().shape[0]

    numerator = num_observed_pairs - num_unique_A
    denominator = num_unique_A * (num_unique_B - 1)

    if denominator == 0:
        return 0.0

    return numerator / denominator


