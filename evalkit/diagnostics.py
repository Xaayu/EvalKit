import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


def validate_evaluation_inputs(y_true, y_pred, y_proba=None, model=None):
    warnings = []
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.size == 0 or y_pred.size == 0:
        warnings.append("Empty dataset detected; evaluation cannot proceed.")
        return {"valid": False, "warnings": warnings}

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    if len(y_true) < 2:
        warnings.append("Small dataset detected; metrics may be unstable.")

    if len(np.unique(y_true)) < 2:
        warnings.append("Single-class target detected; classification metrics may be undefined or misleading.")

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.size == 0:
            warnings.append("Probability scores are empty.")
        elif not np.all(np.isfinite(y_proba)):
            warnings.append("Probability scores contain NaN or infinite values.")
        elif y_proba.ndim not in (1, 2):
            warnings.append("Probability scores must be a 1D or 2D array.")
        elif np.any(y_proba < 0) or np.any(y_proba > 1):
            warnings.append("Probability scores must be between 0 and 1.")
        elif y_proba.ndim == 2 and y_proba.shape[0] == len(y_true) and y_proba.shape[1] > 1:
            row_sums = y_proba.sum(axis=1)
            if not np.allclose(row_sums, 1.0, atol=1e-6):
                warnings.append("Probability rows should sum to 1 for multiclass predictions.")

    if model is not None and not hasattr(model, "predict_proba") and y_proba is not None:
        warnings.append("Model does not implement predict_proba; probability metrics are unavailable.")

    return {"valid": not any("Empty dataset" in warning or "Single-class target" in warning for warning in warnings), "warnings": warnings}


def _safe_pearson_correlation(feature_values, target_values):
    feature_values = pd.to_numeric(feature_values, errors="coerce")
    target_values = pd.to_numeric(target_values, errors="coerce")

    if feature_values.empty or target_values.empty:
        return np.nan

    feature_std = feature_values.std(ddof=0)
    target_std = target_values.std(ddof=0)

    if np.isclose(feature_std, 0.0) or np.isclose(target_std, 0.0):
        return np.nan

    return feature_values.corr(target_values)


def detect_leakage(X, y, id_keywords=None, correlation_threshold=0.9, constant_ratio=0.05):
    warnings = []
    feature_risks = []
    id_keywords = id_keywords or ["id", "uuid", "user", "customer", "record", "index", "email", "account"]

    if X is None or y is None:
        return {"warnings": ["Missing dataset or target for leakage detection."], "feature_risks": [], "leakage_score": 0.0}

    if isinstance(X, (list, tuple, np.ndarray)):
        X = pd.DataFrame(X)

    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        frame = pd.DataFrame(X)

    if frame.empty or len(y) == 0:
        return {"warnings": ["Empty dataset detected during leakage analysis."], "feature_risks": [], "leakage_score": 0.0}

    y_numeric = pd.Series(np.asarray(y)).astype(float)
    valid_rows = frame.shape[0] == len(y_numeric)
    if not valid_rows:
        raise ValueError("X and y must have the same number of rows.")

    if np.isclose(y_numeric.std(ddof=0), 0.0):
        warnings.append("Target is constant; correlation-based leakage checks are undefined.")

    for column_name in frame.columns:
        values = frame[column_name]
        lower_name = str(column_name).lower()
        unique_ratio = values.nunique(dropna=True) / max(len(values), 1)
        if any(keyword in lower_name for keyword in id_keywords) and unique_ratio > 0.9:
            warning = (
                f"Identifier-like feature '{column_name}' has near-unique values, which may indicate leakage."
            )
            warnings.append(warning)
            feature_risks.append({"feature": column_name, "type": "identifier", "risk": "high"})
            continue

        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().sum() > 1:
            if np.isclose(numeric_values.nunique(dropna=True), 1) or unique_ratio <= constant_ratio:
                warning = (
                    f"Constant feature '{column_name}' has very low variation and may be a leakage or data-quality signal."
                )
                warnings.append(warning)
                feature_risks.append({"feature": column_name, "type": "constant", "risk": "medium"})
                continue

            if np.isclose(y_numeric.std(ddof=0), 0.0):
                continue

            correlation = _safe_pearson_correlation(numeric_values, y_numeric)
            if pd.notna(correlation) and abs(correlation) >= correlation_threshold:
                warning = (
                    f"Feature '{column_name}' has strong target correlation ({correlation:.4f}), which may indicate leakage."
                )
                warnings.append(warning)
                feature_risks.append({"feature": column_name, "type": "correlation", "risk": "high"})
                continue

        if any(keyword in lower_name for keyword in ["target", "label", "outcome", "class", "label_", "y_"]):
            warning = (
                f"Feature '{column_name}' resembles a target-like field and may contain leakage."
            )
            warnings.append(warning)
            feature_risks.append({"feature": column_name, "type": "target_like", "risk": "high"})

    leakage_score = min(1.0, len(warnings) / max(len(frame.columns), 1)) if warnings else 0.0
    return {"warnings": warnings, "feature_risks": feature_risks, "leakage_score": float(leakage_score)}


def classification_diagnostics(y_true, y_pred):

    matrix = confusion_matrix(y_true, y_pred)

    classes, counts = np.unique(
        y_true,
        return_counts=True
    )

    class_distribution = {
        str(cls): int(count)
        for cls, count in zip(classes, counts)
    }

    return {
        "confusion_matrix": matrix.tolist(),
        "class_distribution": class_distribution,
        "number_of_classes": len(classes),
    }
    
def per_class_metrics(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "precision": {
            str(i): float(value)
            for i, value in enumerate(precision)
        },
        "recall": {
            str(i): float(value)
            for i, value in enumerate(recall)
        },
        "f1": {
            str(i): float(value)
            for i, value in enumerate(f1)
        },
        "support": {
            str(i): int(value)
            for i, value in enumerate(support)
        },
    }
    
def class_imbalance(y_true):
    classes, counts = np.unique(y_true, return_counts=True)

    total = len(y_true)

    distribution = {
        str(cls): {
            "count": int(count),
            "percentage": float(count / total * 100),
        }
        for cls, count in zip(classes, counts)
    }

    majority_count = counts.max()
    minority_count = counts.min()

    imbalance_ratio = float(
        majority_count / minority_count
    )

    return {
        "distribution": distribution,
        "imbalance_ratio": imbalance_ratio,
        "minority_class": str(
            classes[np.argmin(counts)]
        ),
        "majority_class": str(
            classes[np.argmax(counts)]
        ),
    }
    
def misclassified_samples(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    misclassified = []

    for index, (actual, predicted) in enumerate(
        zip(y_true, y_pred)
    ):
        if actual != predicted:
            misclassified.append({
                "index": int(index),
                "actual": str(actual),
                "predicted": str(predicted),
            })

    return misclassified

def regression_diagnostics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    residuals = y_true - y_pred
    absolute_errors = np.abs(residuals)

    return {
        "mean_residual": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "max_absolute_error": float(np.max(absolute_errors)),
    }