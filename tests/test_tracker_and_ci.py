from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

from evalkit import track
from evalkit.ci import check_report_with_rules, ci_output


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
