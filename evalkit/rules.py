def evaluate_rules(metrics, rules):
    results = {}

    for name, rule in (rules or {}).items():
        actual = metrics.get(name)
        if actual is None:
            results[name] = {
                "passed": False,
                "actual": None,
                "reason": "metric not found",
            }
            continue

        if isinstance(rule, dict):
            if "min" in rule:
                passed = actual >= rule["min"]
                reason = f"must be >= {rule['min']}"
            elif "max" in rule:
                passed = actual <= rule["max"]
                reason = f"must be <= {rule['max']}"
            else:
                raise ValueError(f"unsupported rule for {name}")
        else:
            passed = actual >= rule
            reason = f"must be >= {rule}"

        results[name] = {
            "passed": passed,
            "actual": actual,
            "reason": reason,
        }

    return results


def rules_passed(results):
    return all(result["passed"] for result in results.values())