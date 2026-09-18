import json
from pathlib import Path


class EvaluationHistory:

    def __init__(self, filepath="evalkit_history.json"):
        self.filepath = Path(filepath)
        self.runs = []

        self._load()

    def _load(self):

        if self.filepath.exists():

            with open(self.filepath, "r") as file:
                self.runs = json.load(file)

    def add(self, report):

        self.runs.append(report.to_dict())

        self._save()

    def _save(self):

        with open(self.filepath, "w") as file:
            json.dump(
                self.runs,
                file,
                indent=4
            )

    def latest(self):

        if not self.runs:
            return None

        return self.runs[-1]

    def latest_run(self):
        return self.latest()

    def previous(self):

        if len(self.runs) < 2:
            return None

        return self.runs[-2]

    def recent(self, limit=5):
        if limit is None or limit <= 0:
            return []
        return list(self.runs[-limit:])

    def all(self):

        return self.runs

    def metric_history(self, metric_name):
        history = []

        for run in self.runs:
            metrics = run.get("metrics", {}) if isinstance(run, dict) else {}
            if metric_name in metrics:
                history.append(metrics[metric_name])

        return history

    def model_versions(self):
        versions = []

        for run in self.runs:
            model_version = run.get("model_version") if isinstance(run, dict) else None
            if model_version is not None and model_version not in versions:
                versions.append(model_version)

        return versions

    def by_model_version(self, version):
        if version is None:
            return []

        return [
            run for run in self.runs
            if isinstance(run, dict) and run.get("model_version") == version
        ]

    def metadata_history(self, metadata_name, field=None):
        history = []

        for run in self.runs:
            if not isinstance(run, dict):
                continue

            metadata = run.get(metadata_name, {})
            if not isinstance(metadata, dict):
                if field is None:
                    history.append(metadata)
                continue

            if field is None:
                history.append(metadata)
            elif field in metadata:
                history.append(metadata[field])

        return history
    
    
    