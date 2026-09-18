from evalkit.metrics import register_metric, unregister_metric
from evalkit.rules import evaluate_rules, rules_passed
import numpy as np
from evalkit.metrics import classification_probability_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from evalkit.evaluator import Evaluator
from evalkit.diagnostics import per_class_metrics, detect_leakage, validate_evaluation_inputs
import pandas as pd

from evalkit.metrics import (
    classification_probability_metrics,
    top_k_accuracy_metrics,
)

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


def test_classification_probability_metrics_binary():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.8, 0.9])

    results = classification_probability_metrics(
        y_true,
        y_proba,
    )

    assert "roc_auc" in results
    assert "pr_auc" in results
    assert "log_loss" in results

    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1
    assert results["log_loss"] >= 0
    
    
def test_classification_probability_metrics_multiclass():
    y_true = np.array([0, 1, 2, 0, 1, 2])

    y_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
    ])

    results = classification_probability_metrics(
        y_true,
        y_proba,
    )

    assert "roc_auc" in results
    assert "pr_auc" in results
    assert "log_loss" in results

    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1
    assert results["log_loss"] >= 0 
    
def test_evaluator_includes_probability_metrics():
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        y_test,
        y_pred,
        X_test,
    )

    assert "roc_auc" in results
    assert "pr_auc" in results
    assert "log_loss" in results

    assert 0 <= results["roc_auc"] <= 1
    assert 0 <= results["pr_auc"] <= 1
    assert results["log_loss"] >= 0
    
def test_evaluator_skips_probability_metrics_without_predict_proba():
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = LinearSVC()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        y_test,
        y_pred,
        X_test,
    )

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1" in results

    assert "roc_auc" not in results
    assert "pr_auc" not in results
    assert "log_loss" not in results
    
def test_top_k_accuracy_metrics():
    y_true = np.array([0, 1, 2, 1])

    y_proba = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.2, 0.3, 0.5],
        [0.6, 0.3, 0.1],
    ])

    results = top_k_accuracy_metrics(
        y_true,
        y_proba,
        k_values=(1, 2, 3),
    )

    assert "top_1_accuracy" in results
    assert "top_2_accuracy" in results
    assert "top_3_accuracy" in results

    assert results["top_1_accuracy"] == 0.75
    assert results["top_2_accuracy"] == 1.0
    assert results["top_3_accuracy"] == 1.0
    
def test_evaluator_includes_top_k_metrics():
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    evaluator = Evaluator(model)

    results = evaluator.evaluate(
        y_test,
        y_pred,
        X_test,
    )

    assert "top_1_accuracy" in results
    assert "top_3_accuracy" in results

    assert 0 <= results["top_1_accuracy"] <= 1
    assert 0 <= results["top_3_accuracy"] <= 1
    
def test_per_class_metrics():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    results = per_class_metrics(
        y_true,
        y_pred,
    )

    assert "precision" in results
    assert "recall" in results
    assert "f1" in results
    assert "support" in results

    assert set(results["precision"].keys()) == {"0", "1", "2"}
    assert set(results["recall"].keys()) == {"0", "1", "2"}
    assert set(results["f1"].keys()) == {"0", "1", "2"}

    assert results["support"]["0"] == 2
    assert results["support"]["1"] == 2
    assert results["support"]["2"] == 2
    
def test_class_imbalance():
    from evalkit.diagnostics import class_imbalance

    y_true = np.array([
        0, 0, 0, 0, 0,
        1, 1, 1,
        2, 2
    ])

    results = class_imbalance(y_true)

    assert "distribution" in results
    assert "imbalance_ratio" in results
    assert "minority_class" in results
    assert "majority_class" in results

    assert results["distribution"]["0"]["count"] == 5
    assert results["distribution"]["1"]["count"] == 3
    assert results["distribution"]["2"]["count"] == 2

    assert results["minority_class"] == "2"
    assert results["majority_class"] == "0"
    assert results["imbalance_ratio"] == 2.5
    
def test_feature_analysis():
    from evalkit.profiler import feature_analysis

    X = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "score": [10, 10, 10, 10],
        "missing": [1, None, 3, None],
    })

    results = feature_analysis(X)

    assert "age" in results
    assert "score" in results
    assert "missing" in results

    assert results["age"]["unique_values"] == 4
    assert results["age"]["missing"] == 0
    assert results["age"]["constant"] is False

    assert results["score"]["unique_values"] == 1
    assert results["score"]["constant"] is True

    assert results["missing"]["missing"] == 2
    assert results["missing"]["missing_percentage"] == 50.0


def test_detect_leakage_constant_feature_warning():
    X = pd.DataFrame({
        "constant_feature": [1, 1, 1, 1, 1, 1],
        "varying_feature": [0, 1, 2, 3, 4, 5],
    })
    y = np.array([0, 1, 0, 1, 0, 1])

    result = detect_leakage(X, y)
    warnings = "\n".join(result["warnings"]).lower()

    assert "constant feature" in warnings


def test_detect_leakage_constant_target_is_handled_safely():
    X = pd.DataFrame({
        "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [10, 11, 12, 13, 14, 15],
    })
    y = np.array([1, 1, 1, 1, 1, 1])

    result = detect_leakage(X, y)
    warnings = "\n".join(result["warnings"]).lower()

    assert "constant" in warnings
    assert "correlation-based leakage checks are undefined" in warnings


def test_detect_leakage_varying_feature_has_no_spurious_warning():
    X = pd.DataFrame({
        "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [9.0, 9.0, 7.0, 6.0, 8.0, 5.0],
    })
    y = np.array([0, 1, 0, 1, 0, 1])

    result = detect_leakage(X, y)
    warnings = "\n".join(result["warnings"]).lower()

    assert "strong target correlation" not in warnings
    assert "constant" not in warnings


def test_detect_leakage_high_correlation_feature_warns():
    X = pd.DataFrame({
        "leaky_feature": [0, 1, 0, 1, 0, 1],
        "other_feature": [1, 2, 3, 4, 5, 6],
    })
    y = np.array([0, 1, 0, 1, 0, 1])

    result = detect_leakage(X, y, correlation_threshold=0.9)
    warnings = "\n".join(result["warnings"]).lower()

    assert "strong target correlation" in warnings


def test_validate_evaluation_inputs_handles_empty_and_shape_mismatch():
    empty_result = validate_evaluation_inputs(np.array([]), np.array([]))
    assert empty_result["valid"] is False
    assert any("empty" in warning.lower() for warning in empty_result["warnings"])

    try:
        validate_evaluation_inputs(np.array([0, 1]), np.array([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for mismatched shapes")