METRIC_DIRECTION = {
    "accuracy": "higher",
    "precision": "higher",
    "recall": "higher",
    "f1": "higher",

    "mae": "lower",
    "mse": "lower",
    "rmse": "lower",

    "r2": "higher",

    "fit_time": "lower",
    "prediction_time": "lower",
}


def _coerce_threshold(raw_threshold):
    if raw_threshold is None:
        return 0.0, "max_change"

    if isinstance(raw_threshold, dict):
        if "max_change" in raw_threshold:
            return float(raw_threshold["max_change"]), "max_change"
        if "min_change" in raw_threshold:
            return float(raw_threshold["min_change"]), "min_change"
        if "threshold" in raw_threshold:
            return float(raw_threshold["threshold"]), "threshold"
        return 0.0, "max_change"

    return float(raw_threshold), "max_change"


class ReportComparator:

    def __init__(self, previous, current):
        self.previous = previous
        self.current = current

    def compare(self, thresholds=None):
        if thresholds is None:
            thresholds = {}

        results = {}

        previous_version = self.previous.get("model_version")
        current_version = self.current.get("model_version")

        version_changed = (
            previous_version != current_version
        )

        previous_metrics = self.previous["metrics"]
        current_metrics = self.current["metrics"]

        for metric, current_value in current_metrics.items():

            if metric not in previous_metrics:
                continue

            previous_value = previous_metrics[metric]
            change = current_value - previous_value

            direction = METRIC_DIRECTION.get(metric, "unknown")
            raw_threshold = thresholds.get(metric, 0)
            threshold, threshold_key = _coerce_threshold(raw_threshold)

            if direction == "higher":
                if change < -threshold:
                    status = "degraded"
                    reason = (
                        f"{metric} decreased by {abs(change):.4f}, exceeding the "
                        f"allowed drop of {threshold:.4f}."
                    )
                elif change > threshold:
                    status = "improved"
                    reason = (
                        f"{metric} increased by {change:.4f}, surpassing the "
                        f"threshold of {threshold:.4f}."
                    )
                else:
                    status = "unchanged"
                    reason = (
                        f"{metric} change of {change:.4f} is within the threshold "
                        f"of {threshold:.4f}."
                    )

            elif direction == "lower":
                if change > threshold:
                    status = "degraded"
                    reason = (
                        f"{metric} increased by {change:.4f}, exceeding the allowed "
                        f"increase of {threshold:.4f}."
                    )
                elif change < -threshold:
                    status = "improved"
                    reason = (
                        f"{metric} decreased by {abs(change):.4f}, improving beyond the "
                        f"threshold of {threshold:.4f}."
                    )
                else:
                    status = "unchanged"
                    reason = (
                        f"{metric} change of {change:.4f} is within the threshold of "
                        f"{threshold:.4f}."
                    )

            else:
                status = "unknown"
                reason = f"{metric} direction is unknown; no automatic regression check applied."

            results[metric] = {
                "previous": previous_value,
                "current": current_value,
                "change": change,
                "threshold": raw_threshold,
                "threshold_value": threshold,
                "threshold_key": threshold_key,
                "direction": direction,
                "status": status,
                "reason": reason,
            }

        results["_model_version"] = {
            "previous": previous_version,
            "current": current_version,
            "changed": version_changed,
        }

        return results

