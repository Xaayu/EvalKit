def evaluate_rules(metrics, rules):
    results = {}

    for name, rule in (rules or {}).items():
        actual = metrics.get(name)
        rule_config = rule if isinstance(rule, dict) else {"min": rule}
        severity = rule_config.get("severity", "error")
        description = rule_config.get("description", "") or ""

        if actual is None:
            results[name] = {
                "rule": name,
                "status": "failed",
                "severity": severity,
                "passed": False,
                "actual": None,
                "threshold": None,
                "description": description or "metric not found",
                "reason": "metric not found",
            }
            continue

        if "min" in rule_config:
            threshold = rule_config["min"]
            passed = actual >= threshold
            reason = f"must be >= {threshold}"
            comparator = ">="
        elif "max" in rule_config:
            threshold = rule_config["max"]
            passed = actual <= threshold
            reason = f"must be <= {threshold}"
            comparator = "<="
        else:
            raise ValueError(f"unsupported rule for {name}")

        results[name] = {
            "rule": name,
            "status": "passed" if passed else "failed",
            "severity": severity,
            "passed": passed,
            "actual": actual,
            "threshold": threshold,
            "description": description or reason,
            "reason": reason,
            "comparator": comparator,
        }

    return results


def rules_passed(results):
    return all(result.get("passed", False) for result in results.values())