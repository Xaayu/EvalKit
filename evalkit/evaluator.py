from sklearn.base import is_classifier, is_regressor

from .metrics import (
    classification_metrics,
    classification_probability_metrics,
    regression_metrics,
    top_k_accuracy_metrics,
)


class Evaluator:
    def __init__(self, model, metrics=None):
        self.model = model
        self.metrics = metrics or {}

    def evaluate(self, y_true, y_pred, X=None):
        if is_classifier(self.model):
            results = classification_metrics(
                y_true,
                y_pred
            )

            if X is not None and hasattr(
                self.model,
                "predict_proba"
            ):
                y_proba = self.model.predict_proba(X)

                results.update(
                    classification_probability_metrics(
                        y_true,
                        y_proba
                    )
                )
                
                results.update(
                     top_k_accuracy_metrics(
                        y_true,
                        y_proba
        )
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