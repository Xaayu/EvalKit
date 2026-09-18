from pathlib import Path


DEFAULT_CONFIG = {
    "history_file": "evalkit_history.json",
    "thresholds": {},
    "report_format": "text",
    "metrics": [],
    "rules": {},
    "drift": {},
}


def _validate_thresholds(thresholds):
    if thresholds is None:
        return
    if not isinstance(thresholds, dict):
        raise ValueError("Configuration thresholds must be a mapping of metric names to numeric values")

    for metric, value in thresholds.items():
        if isinstance(value, dict):
            for nested_name, nested_value in value.items():
                try:
                    float(nested_value)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Threshold for {metric}.{nested_name} must be numeric"
                    ) from error
            continue

        try:
            float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Threshold for {metric} must be numeric") from error


def load_config(filepath="evalkit.yaml"):
    config = dict(DEFAULT_CONFIG)
    path = Path(filepath)

    if not path.exists():
        return config

    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML configuration requires PyYAML. Install it with 'pip install pyyaml'."
        ) from error

    with path.open("r") as file:
        loaded = yaml.safe_load(file) or {}

    if not isinstance(loaded, dict):
        raise ValueError("Configuration must contain a YAML mapping")

    config.update(loaded)

    if "thresholds" in config:
        _validate_thresholds(config["thresholds"])

    if "rules" in config and config["rules"] is not None and not isinstance(config["rules"], dict):
        raise ValueError("Configuration rules must be a mapping")

    if "metrics" in config and config["metrics"] is not None and not isinstance(config["metrics"], list):
        raise ValueError("Configuration metrics must be a list")

    if "drift" in config and config["drift"] is not None and not isinstance(config["drift"], dict):
        raise ValueError("Configuration drift settings must be a mapping")

    return config


def merge_thresholds(config, cli_thresholds):
    thresholds = dict(config.get("thresholds", {}) or {})
    thresholds.update(cli_thresholds)
    return thresholds