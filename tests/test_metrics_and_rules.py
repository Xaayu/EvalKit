from evalkit.metrics import register_metric, unregister_metric
from evalkit.rules import evaluate_rules, rules_passed


def test_custom_metric_registry():
    register_metric("exact_match", lambda y_true, y_pred: sum(
        left == right for left, right in zip(y_true, y_pred)
    ) / len(y_true))

    from evalkit.metrics import classification_metrics
    results = classification_metrics([1, 0], [1, 1])

    assert results["exact_match"] == 0.5
    unregister_metric("exact_match")


def test_custom_rules():
    results = evaluate_rules(
        {"accuracy": 0.9, "latency": 0.4},
        {"accuracy": {"min": 0.8}, "latency": {"max": 0.5}},
    )

    assert rules_passed(results)
    assert results["accuracy"]["passed"]


def test_failed_rule():
    results = evaluate_rules({"accuracy": 0.7}, {"accuracy": {"min": 0.8}})
    assert not rules_passed(results)
