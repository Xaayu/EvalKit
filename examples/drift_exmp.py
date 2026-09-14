import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit.drift import detect_numeric_drift
from evalkit.drift import detect_drift

reference = pd.DataFrame({
    "age": [
        20, 22, 24, 25, 27,
        29, 30, 31, 33, 35
    ],
    "income": [
        30000, 32000, 35000, 36000, 38000,
        40000, 42000, 45000, 47000, 50000
    ]
})


current = pd.DataFrame({
    "age": [
        40, 42, 44, 45, 47,
        49, 50, 51, 53, 55
    ],
    "income": [
        60000, 62000, 65000, 66000, 68000,
        70000, 72000, 75000, 77000, 80000
    ]
})

reference = pd.DataFrame({
    "age": [20, 22, 24, 25, 27, 29, 30, 31, 33, 35],
    "income": [30000, 32000, 35000, 36000, 38000,
               40000, 42000, 45000, 47000, 50000],
    "city": [
        "Delhi", "Delhi", "Delhi", "Mumbai", "Mumbai",
        "Mumbai", "Pune", "Pune", "Pune", "Pune"
    ]
})


current = pd.DataFrame({
    "age": [40, 42, 44, 45, 47, 49, 50, 51, 53, 55],
    "income": [60000, 62000, 65000, 66000, 68000,
               70000, 72000, 75000, 77000, 80000],
    "city": [
        "Bangalore", "Bangalore", "Bangalore",
        "Bangalore", "Bangalore", "Mumbai",
        "Mumbai", "Mumbai", "Mumbai", "Mumbai"
    ]
})


from evalkit.drift import (
    detect_numeric_drift,
    detect_categorical_drift
)

numeric_results = detect_numeric_drift(
    reference,
    current
)

categorical_results = detect_categorical_drift(
    reference,
    current
)

print("\n========== Numeric Drift ==========")

for column, result in numeric_results.items():
    print(
        f"{column}: "
        f"PSI={result['psi']:.4f} "
        f"Status={result['status']}"
    )

print("\n======= Categorical Drift =========")

for column, result in categorical_results.items():
    print(
        f"{column}: "
        f"PSI={result['psi']:.4f} "
        f"Status={result['status']}"
    )

print("====================================\n")




results = detect_drift(
    reference,
    current
)

print("\n========== Data Drift ==========")

for column, result in results["numeric"].items():
    print(
        f"{column}: "
        f"PSI={result['psi']:.4f} "
        f"Status={result['status']}"
    )

for column, result in results["categorical"].items():
    print(
        f"{column}: "
        f"PSI={result['psi']:.4f} "
        f"Status={result['status']}"
    )

print("================================\n")