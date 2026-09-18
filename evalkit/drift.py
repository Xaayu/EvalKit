import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI)
    between two numeric distributions.
    """

    expected = np.asarray(expected)
    actual = np.asarray(actual)

    breakpoints = np.percentile(
        expected,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints
    )

    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)

    # Avoid division by zero
    expected_pct = np.where(
        expected_pct == 0,
        0.0001,
        expected_pct
    )

    actual_pct = np.where(
        actual_pct == 0,
        0.0001,
        actual_pct
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )
    
    

    return float(psi)


def _numeric_series_with_missing(series):
    """Convert a series to numeric and treat invalid values as missing."""
    converted = pd.to_numeric(series, errors="coerce")
    return converted


def _numeric_drift_severity(psi_value, threshold):
    """Map PSI magnitude to a human-readable severity label."""
    if psi_value < threshold:
        return "none"
    if psi_value >= threshold * 3:
        return "severe"
    if psi_value >= threshold * 1.5:
        return "high"
    return "moderate"


def detect_numeric_drift(
    reference,
    current,
    threshold=0.2
):
    """
    Detect drift between reference and current
    numeric datasets.
    """

    reference = pd.DataFrame(reference)
    current = pd.DataFrame(current)

    results = {}

    common_columns = reference.columns.intersection(
        current.columns
    )

    for column in common_columns:
        ref_series = _numeric_series_with_missing(reference[column])
        cur_series = _numeric_series_with_missing(current[column])

        if not (
            pd.api.types.is_numeric_dtype(ref_series)
            and
            pd.api.types.is_numeric_dtype(cur_series)
        ):
            continue

        reference_missing = int(ref_series.isna().sum())
        current_missing = int(cur_series.isna().sum())

        reference_values = ref_series.dropna()
        current_values = cur_series.dropna()

        if len(reference_values) == 0 or len(current_values) == 0:
            psi = 0.0
            status = "stable"
            severity = "none"
            ks_statistic = 0.0
            ks_pvalue = 1.0
            reference_mean = None
            current_mean = None
            reference_std = 0.0
            current_std = 0.0
        else:
            psi = calculate_psi(
                reference_values,
                current_values
            )
            ks_statistic, ks_pvalue = ks_2samp(
                reference_values,
                current_values
            )
            status = "drifted" if psi >= threshold else "stable"
            severity = _numeric_drift_severity(psi, threshold)
            reference_mean = float(reference_values.mean())
            current_mean = float(current_values.mean())
            reference_std = float(reference_values.std(ddof=0)) if len(reference_values) > 1 else 0.0
            current_std = float(current_values.std(ddof=0)) if len(current_values) > 1 else 0.0

        results[column] = {
            "psi": float(psi),
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
            "threshold": threshold,
            "status": status,
            "severity": severity,
            "reference_mean": reference_mean,
            "current_mean": current_mean,
            "reference_std": float(reference_std),
            "current_std": float(current_std),
            "reference_missing": reference_missing,
            "current_missing": current_missing,
            "reference_count": int(len(reference_values)),
            "current_count": int(len(current_values)),
        }

    return results

def detect_categorical_drift(
    reference,
    current,
    threshold=0.2
):
    """
    Detect drift between reference and current
    categorical datasets.
    """

    reference = pd.DataFrame(reference)
    current = pd.DataFrame(current)

    results = {}

    common_columns = reference.columns.intersection(
        current.columns
    )

    for column in common_columns:

        if (
            pd.api.types.is_numeric_dtype(reference[column])
            or
            pd.api.types.is_numeric_dtype(current[column])
        ):
            continue

        reference_counts = (
            reference[column]
            .value_counts(normalize=True)
        )

        current_counts = (
            current[column]
            .value_counts(normalize=True)
        )

        categories = set(reference_counts.index).union(
            current_counts.index
        )

        psi = 0.0

        for category in categories:

            expected = reference_counts.get(
                category,
                0
            )

            actual = current_counts.get(
                category,
                0
            )

            expected = max(expected, 0.0001)
            actual = max(actual, 0.0001)

            psi += (
                (actual - expected)
                * np.log(actual / expected)
            )

        reference_probability = np.array([
            max(reference_counts.get(category, 0), 0.0001)
            for category in categories
        ])
        current_probability = np.array([
            max(current_counts.get(category, 0), 0.0001)
            for category in categories
        ])
        reference_probability /= reference_probability.sum()
        current_probability /= current_probability.sum()
        midpoint = (reference_probability + current_probability) / 2
        js_divergence = 0.5 * np.sum(
            reference_probability * np.log(reference_probability / midpoint)
        ) + 0.5 * np.sum(
            current_probability * np.log(current_probability / midpoint)
        )

        if psi >= threshold:
            status = "drifted"
        else:
            status = "stable"

        results[column] = {
            "psi": float(psi),
            "js_divergence": float(js_divergence),
            "threshold": threshold,
            "status": status
        }

    return results


def detect_drift(
    reference,
    current,
    threshold=0.2
):
    """
    Automatically detect numeric and categorical
    drift between reference and current datasets.
    """

    numeric_results = detect_numeric_drift(
        reference,
        current,
        threshold=threshold
    )

    categorical_results = detect_categorical_drift(
        reference,
        current,
        threshold=threshold
    )

    drifted_features = sorted([
        name for name, result in numeric_results.items() if result.get("status") == "drifted"
    ] + [
        name for name, result in categorical_results.items() if result.get("status") == "drifted"
    ])
    stable_features = sorted([
        name for name, result in numeric_results.items() if result.get("status") == "stable"
    ] + [
        name for name, result in categorical_results.items() if result.get("status") == "stable"
    ])

    return {
        "numeric": numeric_results,
        "categorical": categorical_results,
        "summary": {
            "total_features": len(drifted_features) + len(stable_features),
            "drifted_features": drifted_features,
            "stable_features": stable_features,
        },
    }
    
