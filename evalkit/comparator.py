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

            direction = METRIC_DIRECTION.get(metric)
            threshold = thresholds.get(metric, 0)

            if direction == "higher":

                if change < -threshold:
                    status = "degraded"
                elif change > threshold:
                    status = "improved"
                else:
                    status = "unchanged"

            elif direction == "lower":

                if change > threshold:
                    status = "degraded"
                elif change < -threshold:
                    status = "improved"
                else:
                    status = "unchanged"

            else:
                status = "unknown"

            results[metric] = {
                "previous": previous_value,
                "current": current_value,
                "change": change,
                "threshold": threshold,
                "status": status,
            }
            
        results["_model_version"] = {
            "previous": previous_version,
            "current": current_version,
            "changed": version_changed,
        }

        return results
    
