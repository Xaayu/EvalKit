from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

CUSTOM_METRICS = {}

def register_metric(name, function):
    if not callable(function):
        raise TypeError("metric must be callable")
    CUSTOM_METRICS[name] = function


def unregister_metric(name):
    CUSTOM_METRICS.pop(name, None)


def custom_metrics(y_true, y_pred):
    return {
        name: float(function(y_true, y_pred))
        for name, function in CUSTOM_METRICS.items()
    }


def classification_metrics(y_true, y_pred):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }
    results.update(custom_metrics(y_true, y_pred))
    return results


def regression_metrics(y_true, y_pred):
    results = {
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": mean_squared_error(
            y_true,
            y_pred,
        ) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }
    results.update(custom_metrics(y_true, y_pred))
    return results