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

        if not (
            pd.api.types.is_numeric_dtype(
                reference[column]
            )
            and
            pd.api.types.is_numeric_dtype(
                current[column]
            )
        ):
            continue

        reference_values = reference[column].dropna()
        current_values = current[column].dropna()

        if len(reference_values) == 0 or len(current_values) == 0:
            continue

        psi = calculate_psi(
            reference_values,
            current_values
        )
        ks_statistic, ks_pvalue = ks_2samp(
            reference_values,
            current_values
        )

        if psi >= threshold:
            status = "drifted"
        else:
            status = "stable"

        results[column] = {
            "psi": psi,
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
            "threshold": threshold,
            "status": status
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

    return {
        "numeric": numeric_results,
        "categorical": categorical_results
    }