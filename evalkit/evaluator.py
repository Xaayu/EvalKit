from sklearn.base import is_classifier, is_regressor

from .metrics import (
    classification_metrics,
    regression_metrics,
)


class Evaluator:

    def __init__(self, model, metrics=None):
        self.model = model
        self.metrics = metrics or {}

    def evaluate(self, y_true, y_pred):
        if is_classifier(self.model):
            results = classification_metrics(
                y_true,
                y_pred
            )
        elif is_regressor(self.model):
            results = regression_metrics(
                y_true,
                y_pred
            )
        else:
            raise NotImplementedError(
                "This model type is not supported yet."
            )

        results.update({
            name: float(function(y_true, y_pred))
            for name, function in self.metrics.items()
        })
        return results