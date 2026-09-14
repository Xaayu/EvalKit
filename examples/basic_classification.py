import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from evalkit.history import EvaluationHistory
from evalkit.comparator import ReportComparator
from sklearn.tree import DecisionTreeClassifier
from evalkit import track
from sklearn.dummy import DummyClassifier
from evalkit.ci import check_report




# Load data
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Normal sklearn model
model=RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Add EvalKit
model = track(model)


# Everything else stays almost identical
model.fit(X_train, y_train)

predictions = model.predict(X_test)

report = model.evaluate(y_test)

report.save("current.json")
report.save("previous.json")


history = EvaluationHistory()

history.add(report)

previous = history.previous()
current = history.latest()

if previous and current:

    comparator = ReportComparator(
        previous,
        current
    )

    comparison = comparator.compare(
    thresholds={
        "accuracy": 0.02,
        "f1": 0.02
    }
)

    print("\n========== Comparison ==========")

    for metric, result in comparison.items():

        if metric == "_model_version":
            continue

        print(
            f"{metric}: "
            f"{result['previous']:.4f} → "
            f"{result['current']:.4f} "
            f"({result['status']})"
        )
        

passed = check_report(
    report,
    thresholds={
        "accuracy": 0.02,
        "precision": 0.02,
        "recall": 0.02,
        "f1": 0.02,
    }
)

print("CI Status:", "PASSED" if passed else "FAILED")