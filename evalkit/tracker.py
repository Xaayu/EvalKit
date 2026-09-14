import time
import uuid
import logging
from .drift import detect_drift
from sklearn.base import is_classifier
from .versioning import generate_model_version
from .evaluator import Evaluator
from .profiler import profile_data
from .diagnostics import classification_diagnostics
from .report import EvaluationReport
from .history import EvaluationHistory
from .monitoring import (
    feature_importance,
    feature_statistics,
    prediction_distribution,
)

logger = logging.getLogger(__name__)


class ModelTracker:

    def __init__(self, model, metrics=None, history_file=None):
        self.model = model
        self.metrics = metrics or {}
        self.history_file = history_file

        self.fit_time = None
        self.predict_times = []

        self.last_predictions = None

        self.train_profile = None
        self.test_profile = None
        
        self.train_data = None
        self.drift_result = None
        
        self.model_version = generate_model_version(model)
        self.run_id = f"run-{uuid.uuid4().hex[:12]}"
        self.production_hooks = []
        self.feature_monitoring = {}
        
    def fit(self, X, y):
        self.train_profile = profile_data(X)
        self.train_data = X

        start = time.perf_counter()
        self.model.fit(X, y)
        self.fit_time = time.perf_counter() - start

        return self

    def predict(self, X):
        self.test_profile = profile_data(X)
        self.feature_monitoring = {
            "reference": feature_statistics(self.train_data),
            "current": feature_statistics(X),
        }

        self.drift_result = detect_drift(
            self.train_data,
            X
        )

        start = time.perf_counter()
        predictions = self.model.predict(X)
        elapsed = time.perf_counter() - start

        self.predict_times.append(elapsed)
        self.last_predictions = predictions

        return predictions

    def evaluate(self, y_true):

        evaluator = Evaluator(self.model, metrics=self.metrics)
        

        metrics = evaluator.evaluate(
            y_true,
            self.last_predictions
        )

        diagnostics = {}

        if is_classifier(self.model):

            diagnostics = classification_diagnostics(
                y_true,
                self.last_predictions
            )

        performance = {
            "fit_time": self.fit_time,
            "prediction_time": self.predict_times[-1]
        }

        data_profile = {
            "training_rows":
                self.train_profile["rows"],

            "training_columns":
                self.train_profile["columns"],

            "training_missing":
                self.train_profile["missing_values"],

            "test_rows":
                self.test_profile["rows"],

            "test_missing":
                self.test_profile["missing_values"],
        }

        feature_names = getattr(self.train_data, "columns", None)
        explainability = {
            "feature_importance": feature_importance(
                self.model,
                list(feature_names) if feature_names is not None else None,
            )
        }
        
        previous_report = None
        if self.history_file:
            previous_report = EvaluationHistory(
                self.history_file
            ).latest()

        return EvaluationReport(
        model_name=self.model.__class__.__name__,
        metrics=metrics,
        diagnostics=diagnostics,
        data_profile=data_profile,
        performance=performance,
        previous_report=previous_report,
        drift=self.drift_result,
        model_version=self.model_version,
        run_id=self.run_id,
        model_metadata={
            "class": self.model.__class__.__name__,
            "parameters": self.model.get_params(),
        },
        dataset_metadata={
            "training_rows": self.train_profile["rows"],
            "training_columns": self.train_profile["columns"],
            "test_rows": self.test_profile["rows"],
            "test_columns": self.test_profile["columns"],
        },
        feature_monitoring=self.feature_monitoring,
        prediction_distribution=prediction_distribution(
            self.last_predictions
        ),
        explainability=explainability,
)

    def report(self, y_true):

        evaluation = self.evaluate(y_true)

        evaluation.show()

        for hook in self.production_hooks:
            try:
                hook(evaluation)
            except Exception:
                logger.exception("Monitoring hook failed for run %s", self.run_id)

        return evaluation

    def add_monitoring_hook(self, hook):
        if not callable(hook):
            raise TypeError("monitoring hook must be callable")
        self.production_hooks.append(hook)
        return self


def track(model, metrics=None, history_file=None):
    if isinstance(model, ModelTracker):
        return model

    return ModelTracker(
        model,
        metrics=metrics,
        history_file=history_file,
    )