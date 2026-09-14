import numpy as np
import pandas as pd


def feature_statistics(data):
    frame = pd.DataFrame(data)
    statistics = {}

    for column in frame.columns:
        values = frame[column]
        result = {
            "dtype": str(values.dtype),
            "missing": int(values.isna().sum()),
            "unique": int(values.nunique(dropna=True)),
        }

        if pd.api.types.is_numeric_dtype(values):
            result.update({
                "mean": float(values.mean()),
                "std": float(values.std()) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            })
        else:
            result["top_values"] = {
                str(key): int(value)
                for key, value in values.value_counts().head(5).items()
            }

        statistics[str(column)] = result

    return statistics


def prediction_distribution(predictions):
    values = np.asarray(predictions)
    result = {
        "count": int(values.size),
        "unique": int(np.unique(values).size),
    }

    if np.issubdtype(values.dtype, np.number):
        result.update({
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    else:
        result["counts"] = {
            str(key): int(value)
            for key, value in zip(*np.unique(values, return_counts=True))
        }

    return result


def feature_importance(model, feature_names=None):
    values = getattr(model, "feature_importances_", None)
    if values is None:
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            return {}
        values = np.mean(np.abs(np.asarray(coefficients)), axis=0)

    names = feature_names or [str(index) for index in range(len(values))]
    return {
        str(name): float(value)
        for name, value in sorted(
            zip(names, values),
            key=lambda item: item[1],
            reverse=True,
        )
    }