from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from evalkit import track
from evalkit.ci import check_report_with_rules, ci_output
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor

def test_tracker_report_contains_monitoring_data():
    X, y = load_iris(return_X_y=True)
    received = []
    tracker = track(DecisionTreeClassifier(random_state=1))
    tracker.add_monitoring_hook(lambda report: received.append(report.run_id))
    tracker.fit(X[:100], y[:100])
    tracker.predict(X[100:])
    report = tracker.report(y[100:])

    assert report.run_id.startswith("run-")
    assert report.prediction_distribution["count"] == 50
    assert report.feature_monitoring["reference"]
    assert report.explainability["feature_importance"]
    assert received == [report.run_id]


def test_ci_output_formats():
    X, y = load_iris(return_X_y=True)
    tracker = track(DecisionTreeClassifier(random_state=1))
    tracker.fit(X[:100], y[:100])
    tracker.predict(X[100:])
    report = tracker.evaluate(y[100:])
    report.previous_report = None

    passed = check_report_with_rules(
        report,
        rules={"accuracy": {"min": 0.0}},
    )

    assert passed
    assert '"passed": true' in ci_output(report, passed, "json")
    assert "::notice" in ci_output(report, passed, "github")
    
def test_tracker_report_contains_per_class_metrics():
    X, y = load_iris(return_X_y=True)

    tracker = track(DecisionTreeClassifier(random_state=1))

    tracker.fit(X[:100], y[:100])
    tracker.predict(X[100:])

    report = tracker.evaluate(y[100:])

    assert "per_class_metrics" in report.diagnostics

    per_class = report.diagnostics["per_class_metrics"]

    assert "precision" in per_class
    assert "recall" in per_class
    assert "f1" in per_class
    assert "support" in per_class
    
def test_tracker_report_contains_class_imbalance():
    X, y = load_iris(return_X_y=True)

    tracker = track(DecisionTreeClassifier(random_state=1))

    tracker.fit(X[:100], y[:100])
    tracker.predict(X[100:])

    report = tracker.evaluate(y[100:])

    assert "class_imbalance" in report.diagnostics

    imbalance = report.diagnostics["class_imbalance"]

    assert "distribution" in imbalance
    assert "imbalance_ratio" in imbalance
    assert "minority_class" in imbalance
    assert "majority_class" in imbalance
    
def test_misclassified_samples():
    from evalkit.diagnostics import misclassified_samples

    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 2, 2, 0])

    results = misclassified_samples(y_true, y_pred)

    assert len(results) == 2

    assert results[0] == {
        "index": 1,
        "actual": "1",
        "predicted": "2",
    }

    assert results[1] == {
        "index": 3,
        "actual": "1",
        "predicted": "0",
    }
    
def test_tracker_report_contains_misclassified_samples():
    X, y = load_iris(return_X_y=True)

    tracker = track(DecisionTreeClassifier(random_state=1))

    tracker.fit(X[:100], y[:100])
    tracker.predict(X[100:])

    report = tracker.evaluate(y[100:])

    assert "misclassified_samples" in report.diagnostics

    misclassified = report.diagnostics["misclassified_samples"]

    assert isinstance(misclassified, list)

    for sample in misclassified:
        assert "index" in sample
        assert "actual" in sample
        assert "predicted" in sample
        
def test_regression_diagnostics():
    from evalkit.diagnostics import regression_diagnostics

    y_true = np.array([10, 20, 30])
    y_pred = np.array([11, 18, 33])

    results = regression_diagnostics(y_true, y_pred)

    assert "mean_residual" in results
    assert "residual_std" in results
    assert "max_absolute_error" in results

    assert results["mean_residual"] == -2 / 3
    assert results["max_absolute_error"] == 3
    
def test_tracker_report_contains_regression_diagnostics():
    X, y = load_diabetes(return_X_y=True)

    tracker = track(
        DecisionTreeRegressor(random_state=1)
    )

    tracker.fit(X[:400], y[:400])
    tracker.predict(X[400:])

    report = tracker.evaluate(y[400:])

    assert "mean_residual" in report.diagnostics
    assert "residual_std" in report.diagnostics
    assert "max_absolute_error" in report.diagnostics