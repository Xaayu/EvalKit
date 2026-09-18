import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
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

def classification_probability_metrics(y_true, y_proba):
    y_proba = np.asarray(y_proba)

    # Binary classification
    if y_proba.ndim == 1:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)

    # Binary with probabilities for both classes
    elif y_proba.shape[1] == 2:
        roc_auc = roc_auc_score(y_true, y_proba[:, 1])
        pr_auc = average_precision_score(y_true, y_proba[:, 1])

    # Multiclass
    else:
        roc_auc = roc_auc_score(
            y_true,
            y_proba,
            multi_class="ovr",
            average="weighted",
        )

        pr_auc = average_precision_score(
            y_true,
            y_proba,
            average="weighted",
        )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "log_loss": float(log_loss(y_true, y_proba)),
    }

def top_k_accuracy_metrics(y_true, y_proba, k_values=(1, 3, 5)):
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if y_proba.ndim != 2:
        raise ValueError(
            "y_proba must be a 2D array of class probabilities."
        )

    n_classes = y_proba.shape[1]

    results = {}

    for k in k_values:
        if k > n_classes:
            continue

        top_k_predictions = np.argsort(
            y_proba,
            axis=1
        )[:, -k:]

        correct = np.array([
            true_label in predictions
            for true_label, predictions in zip(
                y_true,
                top_k_predictions
            )
        ])

        results[f"top_{k}_accuracy"] = float(
            np.mean(correct)
        )

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