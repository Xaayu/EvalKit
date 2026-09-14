import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from evalkit.history import EvaluationHistory
from evalkit.comparator import ReportComparator

from evalkit import track

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

from sklearn.dummy import DummyRegressor

model = track(
    DummyRegressor(strategy="mean")
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

report = model.evaluate(y_test)

history = EvaluationHistory("evalkit_regression_history.json")
history.add(report)

previous = history.previous()
current = history.latest()
report.previous_report = previous

if previous and current:
    comparator = ReportComparator(previous, current)
    comparison = comparator.compare(
        thresholds={
            "mae": 2.0,
            "mse": 100.0,
            "rmse": 3.0,
            "r2": 0.01,
        }
    )

    print("\n========== Comparison ==========")

    for metric, result in comparison.items():

        if metric == "_model_version":
            continue

        print(
            f"{metric}: "
            f"{result['previous']:.4f} -> {result['current']:.4f} "
            f"({result['status']})"
        )

regression = report.check_regression(
    thresholds={
        "mae": 2.0,
        "mse": 100.0,
        "rmse": 3.0,
        "r2": 0.01,
    }
)

print("\nRegression Check:")
print(regression)

report.show()