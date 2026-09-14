import json

import pytest
from sklearn.tree import DecisionTreeClassifier

from evalkit import track
from evalkit.cli import load_report


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
