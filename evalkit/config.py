from pathlib import Path


DEFAULT_CONFIG = {
    "history_file": "evalkit_history.json",
    "thresholds": {},
    "report_format": "text",
}


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
    return config


def merge_thresholds(config, cli_thresholds):
    thresholds = dict(config.get("thresholds", {}) or {})
    thresholds.update(cli_thresholds)
    return thresholds