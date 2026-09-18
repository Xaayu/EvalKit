from dataclasses import dataclass
from datetime import datetime, timezone
import json


@dataclass
class EvaluationReport:
    model_name: str
    metrics: dict
    diagnostics: dict
    data_profile: dict
    performance: dict
    previous_report: dict = None
    drift: dict = None
    model_version: str = None
    timestamp: str = None
    run_id: str = None
    model_metadata: dict = None
    dataset_metadata: dict = None
    feature_monitoring: dict = None
    prediction_distribution: dict = None
    explainability: dict = None
    rule_results: dict = None
    leakage_warnings: list = None
    validation_warnings: list = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "run_id": self.run_id,
            "metrics": self.metrics,
            "diagnostics": self.diagnostics,
            "data_profile": self.data_profile,
            "performance": self.performance,
            "drift": self.drift,
            "model_metadata": self.model_metadata or {},
            "dataset_metadata": self.dataset_metadata or {},
            "feature_monitoring": self.feature_monitoring or {},
            "prediction_distribution": self.prediction_distribution or {},
            "explainability": self.explainability or {},
            "rule_results": self.rule_results or {},
            "leakage_warnings": self.leakage_warnings or [],
            "validation_warnings": self.validation_warnings or [],
        }

    def to_json(self, indent=4):
        return json.dumps(
            self.to_dict(),
            indent=indent
        )

    def summary(self):
        drift_block = self.drift or {}
        numeric_drift = drift_block.get("numeric", {})
        categorical_drift = drift_block.get("categorical", {})
        drift_summary = drift_block.get("summary", {})

        drifted = list(drift_summary.get("drifted_features", []))
        stable = list(drift_summary.get("stable_features", []))

        if not drifted and not stable:
            for result in numeric_drift.values():
                if result.get("status") == "drifted":
                    drifted.append(result.get("column"))
            for result in categorical_drift.values():
                if result.get("status") == "drifted":
                    drifted.append(result.get("column"))

        rule_results = self.rule_results or {}
        passed_rules = sum(1 for result in rule_results.values() if result.get("status") == "passed")
        failed_rules = sum(1 for result in rule_results.values() if result.get("status") == "failed")

        leakage_warnings = self.leakage_warnings or []
        validation_warnings = self.validation_warnings or []

        regression_summary = {"status": "no_previous_report", "results": {}}
        if self.previous_report is not None:
            from .comparator import ReportComparator

            comparator = ReportComparator(self.previous_report, self.to_dict())
            results = comparator.compare()
            has_regression = any(
                result.get("status") == "degraded"
                for metric, result in results.items()
                if metric != "_model_version"
            )
            regression_summary = {
                "status": "regression_detected" if has_regression else "no_regression",
                "results": results,
                "degraded_metrics": [
                    metric for metric, result in results.items()
                    if metric != "_model_version" and result.get("status") == "degraded"
                ],
            }

        return {
            "model": self.model_name,
            "model_version": self.model_version,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
            "drift_summary": {
                "total_features": len(drifted) + len(stable),
                "drifted_features": drifted,
                "stable_features": stable,
                "numeric_count": len(numeric_drift),
                "categorical_count": len(categorical_drift),
            },
            "rule_summary": {
                "total": len(rule_results),
                "passed": passed_rules,
                "failed": failed_rules,
                "results": rule_results,
            },
            "leakage_summary": {
                "warning_count": len(leakage_warnings),
                "warnings": leakage_warnings,
            },
            "validation_summary": {
                "warning_count": len(validation_warnings),
                "warnings": validation_warnings,
            },
            "regression_summary": regression_summary,
        }

    def render_summary(self):
        summary = self.summary()
        lines = [
            "========== EvalKit Summary ==========",
            f"Model: {summary['model']}",
            f"Model version: {summary['model_version']}",
            f"Run ID: {summary['run_id']}",
            "",
            "Metrics:",
        ]

        for name, value in summary["metrics"].items():
            lines.append(f"- {name}: {value}")

        lines.extend([
            "",
            "Drift:",
            f"- drifted_features: {summary['drift_summary']['drifted_features']}",
            f"- stable_features: {summary['drift_summary']['stable_features']}",
            "",
            "Rules:",
            f"- passed: {summary['rule_summary']['passed']}",
            f"- failed: {summary['rule_summary']['failed']}",
            "",
            "Leakage:",
            f"- warning_count: {summary['leakage_summary']['warning_count']}",
            "",
            "Validation:",
            f"- warning_count: {summary['validation_summary']['warning_count']}",
            "",
            "Regression:",
            f"- status: {summary['regression_summary']['status']}",
            "====================================",
        ])
        return "\n".join(lines)

    def check_regression(self, thresholds=None):
        if self.previous_report is None:
            result = {
                "status": "no_previous_report",
                "results": {}
            }

            print("\n========== Regression Check ==========")
            print("Status: No previous report available.")
            print("======================================\n")

            return result

        previous_version = self.previous_report.get("model_version")
        version_changed = previous_version != self.model_version
        print(f"Version: {self.model_version}")

        from .comparator import ReportComparator

        comparator = ReportComparator(
            self.previous_report,
            self.to_dict()
        )

        results = comparator.compare(
            thresholds=thresholds
        )

        has_regression = any(
            result.get("status") == "degraded"
            for metric, result in results.items()
            if metric != "_model_version"
        )

        status = (
            "regression_detected"
            if has_regression
            else "no_regression"
        )

        print("\n========== Regression Check ==========")

        print(
            f"Model version: {previous_version} -> "
            f"{self.model_version} "
            f"({'changed' if version_changed else 'unchanged'})"
        )

        if has_regression:
            print("Status: REGRESSION DETECTED")
        else:
            print("Status: NO REGRESSION")

        print()

        for metric, result in results.items():

            if metric == "_model_version":
                continue

            previous = result["previous"]
            current = result["current"]
            change = result["change"]
            threshold = result["threshold"]
            metric_status = result["status"]

            print(f"{metric.upper()}")

            print(f"  {previous:.4f} -> {current:.4f}")
            print(f"  Change: {change:+.4f}")
            print(f"  Threshold: {threshold}")

            if metric_status == "degraded":
                print("  Status: DEGRADED")
            elif metric_status == "improved":
                print("  Status: IMPROVED")
            else:
                print("  Status: UNCHANGED")

            print()

        print("======================================\n")

        return {
            "status": status,
            "results": results,
            "model_version": {
                "previous": previous_version,
                "current": self.model_version,
                "changed": version_changed,
            },
        }
    
    
    def save(self, filepath):
        with open(filepath, "w") as file:
            file.write(self.to_json())

    def show(self):

        print("\n========== EvalKit ==========")

        print(f"Model: {self.model_name}")
        print(f"Model version: {self.model_version}")
        print(f"Run ID: {self.run_id}")

        if self.previous_report is not None:
            previous_version = self.previous_report.get("model_version")
            print(
                f"Version change: {previous_version} -> "
                f"{self.model_version} "
                f"({'changed' if previous_version != self.model_version else 'unchanged'})"
            )

        print("\nMetrics:")

        for name, value in self.metrics.items():
            print(
                f"{name.capitalize():<12} "
                f"{value:.4f}"
            )

        if self.model_metadata:
            print("\nModel Metadata:")
            for name, value in self.model_metadata.items():
                print(f"{name}: {value}")

        if self.dataset_metadata:
            print("\nDataset Metadata:")
            for name, value in self.dataset_metadata.items():
                print(f"{name}: {value}")

        if self.prediction_distribution:
            print("\nPrediction Distribution:")
            for name, value in self.prediction_distribution.items():
                print(f"{name}: {value}")

        if self.explainability:
            print("\nFeature Importance:")
            for name, value in self.explainability.get("feature_importance", {}).items():
                print(f"{name}: {value:.6f}")

        if self.diagnostics:

            print("\nDiagnostics:")

            for name, value in self.diagnostics.items():
                print(f"{name}: {value}")

        if self.leakage_warnings:
            print("\nLeakage Warnings:")
            for warning in self.leakage_warnings:
                print(f"- {warning}")

        if self.validation_warnings:
            print("\nValidation Warnings:")
            for warning in self.validation_warnings:
                print(f"- {warning}")

        if self.drift:
            print("\nData Drift:")

            numeric_drift = self.drift.get(
                "numeric",
                {}
            )

            categorical_drift = self.drift.get(
                "categorical",
                {}
            )

            if numeric_drift:
                print("Numeric:")

                for column, result in numeric_drift.items():
                    advanced_metrics = ""
                    if "ks_statistic" in result:
                        advanced_metrics = (
                            f" KS={result['ks_statistic']:.4f}"
                            f" p={result['ks_pvalue']:.4f}"
                        )
                    print(
                        f"  {column}: "
                        f"PSI={result['psi']:.4f} "
                        f"{advanced_metrics} "
                        f"Threshold={result['threshold']} "
                        f"Status={result['status']}"
                    )

            if categorical_drift:
                print("Categorical:")

                for column, result in categorical_drift.items():
                    advanced_metrics = ""
                    if "js_divergence" in result:
                        advanced_metrics = (
                            f" JS={result['js_divergence']:.4f}"
                        )
                    print(
                        f"  {column}: "
                        f"PSI={result['psi']:.4f} "
                        f"{advanced_metrics} "
                        f"Threshold={result['threshold']} "
                        f"Status={result['status']}"
                    )