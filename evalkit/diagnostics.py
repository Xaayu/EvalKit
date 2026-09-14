import numpy as np
from sklearn.metrics import confusion_matrix


def classification_diagnostics(y_true, y_pred):

    matrix = confusion_matrix(y_true, y_pred)

    classes, counts = np.unique(
        y_true,
        return_counts=True
    )

    class_distribution = {
        str(cls): int(count)
        for cls, count in zip(classes, counts)
    }

    return {
        "confusion_matrix": matrix.tolist(),
        "class_distribution": class_distribution,
        "number_of_classes": len(classes),
    }