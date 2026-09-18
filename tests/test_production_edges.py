import json

import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from evalkit import track
from evalkit.cli import build_parser, load_report, parse_thresholds
from evalkit.config import DEFAULT_CONFIG, load_config, merge_thresholds
from evalkit.drift import detect_drift, detect_numeric_drift
from evalkit.history import EvaluationHistory
from evalkit.report import EvaluationReport
from evalkit.rules import evaluate_rules, rules_passed


def test_failed_monitoring_hook_does_not_break_report(caplog):
    tracker = track(DecisionTreeClassifier(random_state=1))
    tracker.add_monitoring_hook(lambda report: (_ for _ in ()).throw(RuntimeError("sink down")))
    tracker.fit([[0], [1], [2], [3]], [0, 0, 1, 1])
    tracker.predict([[0], [3]])

    with caplog.at_level("ERROR"):
        report = tracker.report([0, 1])

    assert report.run_id.startswith("run-")
    assert "Monitoring hook failed" in caplog.text


def test_cli_loads_saved_report(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({
        "model": "Demo",
        "metrics": {"accuracy": 1.0},
    }))

    report = load_report(path)
    assert report.model_name == "Demo"
    assert report.timestamp is not None


def test_invalid_hook_is_rejected():
    with pytest.raises(TypeError):
        track(DecisionTreeClassifier()).add_monitoring_hook("not callable")


def test_history_tracks_runs_metric_trends_and_model_versions(tmp_path):
    history = EvaluationHistory(str(tmp_path / "history.json"))

    assert history.all() == []
    assert history.latest() is None
    assert history.previous() is None

    class Run:
        def __init__(self, payload):
            self.payload = payload

        def to_dict(self):
            return self.payload

    first = Run({
        "model": "DemoModel",
        "model_version": "v1",
        "run_id": "run-1",
        "timestamp": "2024-01-01T00:00:00Z",
        "metrics": {"accuracy": 0.81},
        "dataset_metadata": {"rows": 100},
    })
    second = Run({
        "model": "DemoModel",
        "model_version": "v2",
        "run_id": "run-2",
        "timestamp": "2024-01-02T00:00:00Z",
        "metrics": {"accuracy": 0.89},
        "dataset_metadata": {"rows": 120},
    })

    history.add(first)
    history.add(second)

    assert history.latest()["run_id"] == "run-2"
    assert history.latest_run()["run_id"] == "run-2"
    assert history.recent(1)[0]["run_id"] == "run-2"
    assert history.metric_history("accuracy") == [0.81, 0.89]
    assert history.model_versions() == ["v1", "v2"]
    assert history.by_model_version("v2")[0]["run_id"] == "run-2"
    assert history.metadata_history("dataset_metadata", "rows") == [100, 120]


def test_regression_comparator_uses_metric_specific_thresholds_and_reasons():
    previous = {
        "model_version": "v1",
        "metrics": {"accuracy": 0.90, "rmse": 0.30},
    }
    current = {
        "model_version": "v2",
        "metrics": {"accuracy": 0.84, "rmse": 0.55},
    }

    thresholds = {
        "accuracy": 0.02,
        "rmse": {"max_change": 0.1},
    }

    comparison = track(DecisionTreeClassifier(random_state=1))
    results = comparison.__class__ if False else None

    from evalkit.comparator import ReportComparator

    results = ReportComparator(previous, current).compare(thresholds=thresholds)

    assert results["accuracy"]["status"] == "degraded"
    assert "accuracy" in results["accuracy"]["reason"].lower()
    assert results["rmse"]["status"] == "degraded"
    assert "rmse" in results["rmse"]["reason"].lower()
    assert results["accuracy"]["direction"] == "higher"
    assert results["rmse"]["direction"] == "lower"
    assert results["accuracy"]["threshold"] == 0.02
    assert results["rmse"]["threshold"] == {"max_change": 0.1}


def test_rule_evaluation_returns_structured_results_and_severity():
    metrics = {"accuracy": 0.81, "latency": 0.4}
    rules = {
        "accuracy": {"min": 0.85, "severity": "error", "description": "minimum acceptable accuracy"},
        "latency": {"max": 0.3, "severity": "warning", "description": "latency threshold"},
        "custom": {"min": 0.5, "severity": "warning", "description": "custom minimum"},
    }

    results = evaluate_rules(metrics, rules)

    assert results["accuracy"]["status"] == "failed"
    assert results["accuracy"]["severity"] == "error"
    assert results["accuracy"]["rule"] == "accuracy"
    assert "minimum acceptable accuracy" in results["accuracy"]["description"].lower()
    assert results["latency"]["status"] == "failed"
    assert results["latency"]["severity"] == "warning"
    assert results["custom"]["status"] == "failed"
    assert results["custom"]["actual"] is None
    assert not rules_passed(results)


def test_cli_thresholds_allow_whitespace_and_helpful_parser_output():
    parser = build_parser()
    args = parser.parse_args([
        "compare",
        "--previous",
        "previous.json",
        "--current",
        "current.json",
        "--threshold",
        " accuracy = 0.80 ",
    ])

    thresholds = parse_thresholds(args.threshold)
    assert thresholds["accuracy"] == 0.8
    assert "compare" in parser.format_help().lower()


def test_report_summary_includes_structured_sections_and_human_text():
    report = EvaluationReport(
        model_name="DemoModel",
        metrics={"accuracy": 0.91, "rmse": 0.4},
        diagnostics={"validation_warnings": ["num rows small"]},
        data_profile={"rows": 120},
        performance={"fit_time": 0.2},
        drift={
            "numeric": {"feature_a": {"status": "drifted", "psi": 0.31}, "feature_b": {"status": "stable", "psi": 0.05}},
            "summary": {"drifted_features": ["feature_a"], "stable_features": ["feature_b"]},
        },
        rule_results={
            "accuracy": {"status": "passed", "severity": "warning"},
        },
        leakage_warnings=["feature leakage risk"],
        validation_warnings=["small dataset"],
        previous_report={"model_version": "v1", "metrics": {"accuracy": 0.88, "rmse": 0.5}},
        model_version="v2",
        run_id="run-123",
    )

    summary = report.summary()
    assert summary["model"] == "DemoModel"
    assert summary["drift_summary"]["drifted_features"] == ["feature_a"]
    assert summary["rule_summary"]["passed"] == 1
    assert summary["leakage_summary"]["warning_count"] == 1
    assert summary["validation_summary"]["warning_count"] == 1
    assert summary["regression_summary"]["status"] in {"no_regression", "regression_detected"}

    rendered = report.render_summary()
    assert "DemoModel" in rendered
    assert "Drift" in rendered or "drift" in rendered.lower()
    assert "Rules" in rendered or "rules" in rendered.lower()


def test_config_loader_handles_missing_valid_partial_and_invalid_yaml(tmp_path):
    assert load_config(str(tmp_path / "missing.yaml")) == DEFAULT_CONFIG

    config_path = tmp_path / "valid.yaml"
    config_path.write_text(
        "metrics:\n  - accuracy\n  - recall\nthresholds:\n  accuracy: 0.80\nrules:\n  accuracy:\n    min: 0.75\n"
    )
    config = load_config(str(config_path))
    assert config["metrics"] == ["accuracy", "recall"]
    assert config["thresholds"]["accuracy"] == 0.8
    assert config["rules"]["accuracy"]["min"] == 0.75

    partial_path = tmp_path / "partial.yaml"
    partial_path.write_text("thresholds:\n  recall: 0.90\n")
    partial = load_config(str(partial_path))
    assert partial["thresholds"]["recall"] == 0.9

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("thresholds:\n  accuracy: nope\n")
    with pytest.raises(ValueError, match="numeric"):
        load_config(str(invalid_path))

    merged = merge_thresholds({"thresholds": {"accuracy": 0.8}}, {"accuracy": 0.9})
    assert merged["accuracy"] == 0.9


def test_numeric_drift_reports_feature_level_summary_and_severity():
    reference = pd.DataFrame({
        "feature_a": [10, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        "feature_b": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    })
    current = pd.DataFrame({
        "feature_a": [30, 32, 34, 36, 38, 40, 42, 44, 46, 48],
        "feature_b": [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    })

    results = detect_numeric_drift(reference, current, threshold=0.2)
    assert "feature_a" in results
    assert "feature_b" in results

    feature_a = results["feature_a"]
    assert feature_a["status"] == "drifted"
    assert feature_a["severity"] in {"moderate", "high", "severe"}
    assert feature_a["reference_mean"] < feature_a["current_mean"]
    assert feature_a["reference_missing"] == 0
    assert feature_a["current_missing"] == 0

    feature_b = results["feature_b"]
    assert feature_b["status"] == "stable"
    assert feature_b["severity"] == "none"

    drift = detect_drift(reference, current, threshold=0.2)
    assert drift["summary"]["drifted_features"] == ["feature_a"]
    assert drift["summary"]["stable_features"] == ["feature_b"]
