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

def feature_analysis(X):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    features = {}

    for column in X.columns:
        series = X[column]

        features[str(column)] = {
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "missing_percentage": float(
                series.isna().mean() * 100
            ),
            "unique_values": int(series.nunique(dropna=True)),
            "constant": bool(series.nunique(dropna=True) <= 1),
        }

    return features