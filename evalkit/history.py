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

    def previous(self):

        if len(self.runs) < 2:
            return None

        return self.runs[-2]

    def all(self):

        return self.runs
    
    
    