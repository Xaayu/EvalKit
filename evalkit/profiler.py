import pandas as pd


def profile_data(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    profile = {
        "rows": X.shape[0],
        "columns": X.shape[1],
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(X.duplicated().sum()),
    }

    return profile