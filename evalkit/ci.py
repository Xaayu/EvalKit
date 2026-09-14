import json

from .rules import evaluate_rules, rules_passed


def check_report(report, thresholds=None):
    result = report.check_regression(
        thresholds=thresholds
    )

    if result["status"] == "regression_detected":
        return False

    return True


def check_report_with_rules(report, thresholds=None, rules=None):
    regression_passed = check_report(report, thresholds)
    rule_results = evaluate_rules(report.metrics, rules)
    report.rule_results = rule_results
    return regression_passed and rules_passed(rule_results)


def ci_output(report, passed, fmt="text"):
    payload = {
        "run_id": report.run_id,
        "model": report.model_name,
        "passed": passed,
        "metrics": report.metrics,
        "rules": report.rule_results or {},
    }
    if fmt == "json":
        return json.dumps(payload, indent=4)
    if fmt == "github":
        level = "notice" if passed else "error"
        return f"::{level} title=EvalKit::CI {'PASSED' if passed else 'FAILED'}"
    return f"CI Status: {'PASSED' if passed else 'FAILED'}"