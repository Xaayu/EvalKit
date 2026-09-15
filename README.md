# EvalKit

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.1.1-informational)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A lightweight, production-focused evaluation and monitoring toolkit for scikit-learn models.**

EvalKit helps you evaluate models, track performance, detect data drift, compare runs, enforce custom evaluation rules, and integrate model checks directly into CI/CD.

```text
Train → Predict → Evaluate → Monitor → Compare → Validate → CI/CD
```

## Why EvalKit?

Model evaluation often ends with a few metrics printed in a notebook.

EvalKit turns that into a repeatable engineering workflow:

* 📊 **Model evaluation** — classification and regression metrics
* 🔍 **Diagnostics** — confusion matrices, class distributions, and dataset profiling
* 📈 **Performance tracking** — training and prediction latency
* 🧪 **Run history** — save and compare evaluation results
* 🔄 **Data drift detection** — numeric and categorical feature drift
* 🏷️ **Model versioning** — deterministic model configuration versions
* 🚨 **Regression detection** — detect performance degradation against previous runs
* 🧩 **Custom metrics & rules** — extend evaluation for your own requirements
* 🔌 **Monitoring hooks** — connect evaluation results to external workflows
* 🤖 **CI/CD integration** — automatically fail builds when evaluation rules fail
* 🖥️ **CLI** — run evaluations and comparisons outside Python
* 📦 **JSON reports** — save evaluation results for automation and analysis

---

## Installation

### From PyPI

```powershell
python -m pip install evalkit-ml
```

### From source

```powershell
python -m pip install -e .
```

Once installed, the `evalkit` CLI is available:

```powershell
evalkit --help
```

---

## Quick Start

```python
from evalkit import track

tracker = track(model)

tracker.fit(X_train, y_train)

tracker.predict(X_test)

report = tracker.evaluate(y_test)

report.show()
```

Save the evaluation:

```python
report.save("current.json")
```

A report contains information such as:

```text
Model
Version
Metrics
Diagnostics
Dataset Profile
Performance
Drift
Timestamp
```

---

## Model Evaluation

EvalKit automatically detects whether the model is a classifier or regressor.

### Classification

Supported metrics include:

* Accuracy
* Precision
* Recall
* F1

### Regression

Supported metrics include:

* MAE
* MSE
* RMSE
* R²

Example:

```python
tracker = track(model)

tracker.fit(X_train, y_train)
tracker.predict(X_test)

report = tracker.evaluate(y_test)
report.show()
```

---

## Data & Feature Monitoring

EvalKit profiles the datasets used during evaluation and can detect changes between training and evaluation data.

### Numeric features

Uses **Population Stability Index (PSI)** to identify distribution changes.

### Categorical features

Compares categorical distributions and reports drift using PSI.

Example report:

```text
Data Drift:

Numeric:
  age:    PSI=0.0842  Status=stable
  income: PSI=0.7341  Status=drifted

Categorical:
  city:   PSI=0.4217  Status=drifted
```

---

## Model Versioning

EvalKit automatically generates a deterministic version from the model configuration.

For example:

```text
RandomForestClassifier
→ v-caaca18c
```

Changing the model configuration produces a different version.

This makes it possible to identify which model configuration produced an evaluation result.

---

## Run History & Comparison

History is explicit rather than globally shared.

```python
tracker = track(
    model,
    history_file="history.json"
)
```

This allows separate models or experiments to maintain independent histories.

Compare two reports:

```powershell
evalkit compare `
  --current current.json `
  --previous previous.json
```

Example:

```text
ACCURACY
  0.9412 -> 0.9125
  Change: -0.0287
  Status: DEGRADED
```

---

## Regression Checks

EvalKit can detect whether a model's performance has degraded compared with a previous evaluation.

```powershell
evalkit check `
  --report current.json `
  --previous previous.json `
  --threshold accuracy=0.02 `
  --threshold f1=0.02
```

Metric direction is handled automatically.

For example:

```text
Accuracy ↓  → degradation
RMSE ↑      → degradation
R² ↑        → improvement
```

---

## CI/CD

EvalKit can act as a quality gate in CI pipelines.

```powershell
evalkit check `
  --report current.json `
  --previous previous.json
```

Exit codes:

```text
0 → Passed
1 → Regression / rule failure
```

This makes EvalKit suitable for automated CI/CD workflows.

### GitHub Actions

A typical workflow can simply run:

```yaml
- name: Run EvalKit checks
  run: evalkit check --report current.json --previous previous.json
```

A failed evaluation causes the CI job to fail automatically.

---

## Configuration

Create an `evalkit.yaml` file:

```yaml
thresholds:
  accuracy: 0.02
  precision: 0.02
  recall: 0.02
  f1: 0.02

rules:
  accuracy:
    min: 0.8
```

This keeps evaluation policy separate from application code.

---

## Custom Metrics

Register reusable metrics:

```python
from evalkit import register_metric

register_metric(
    "custom_score",
    my_metric_function
)
```

Or provide metrics directly:

```python
tracker = track(
    model,
    metrics={
        "custom_score": my_metric_function
    }
)
```

This allows EvalKit to support domain-specific evaluation without modifying the core library.

---

## Custom Evaluation Rules

Define rules for model quality requirements.

For example:

```yaml
rules:
  accuracy:
    min: 0.8
```

A rule failure can cause the CI check to return exit code `1`.

This lets you enforce requirements such as:

```text
Accuracy must be ≥ 0.80
F1 must be ≥ 0.75
RMSE must be ≤ 50
```

---

## Monitoring Hooks

Evaluation results can be passed to monitoring hooks.

Hooks are isolated from model evaluation, meaning a monitoring hook failure does **not** silently break the model evaluation itself.

This makes integrations safer for production workflows.

---

## CLI

Available commands:

```powershell
evalkit check --report current.json --previous previous.json

evalkit compare --current current.json --previous previous.json

evalkit history --file evalkit_history.json
```

For automation, output formats are available:

```powershell
evalkit check ... --output json
```

or:

```powershell
evalkit check ... --output github
```

---

## Architecture

```text
                    ┌─────────────────┐
                    │   scikit-learn  │
                    │      Model      │
                    └────────┬────────┘
                             │
                          track()
                             │
                    ┌────────▼────────┐
                    │  ModelTracker   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
      Evaluation         Monitoring          Versioning
          │                  │                  │
          ▼                  ▼                  ▼
       Metrics             Drift             Model ID
      Diagnostics        Features
      Profiling          Predictions
          │                  │
          └──────────┬───────┘
                     ▼
              EvaluationReport
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       History    Compare      Rules
          │          │          │
          └──────────┼──────────┘
                     ▼
                 CI / CLI
```

---

## Design Goals

EvalKit is designed around a few principles:

**Simple**

Wrap an existing scikit-learn model without changing your ML workflow.

**Explicit**

History and monitoring configuration should be controlled by the user rather than hidden global state.

**Extensible**

Custom metrics, rules, and monitoring hooks can be added without modifying the core evaluator.

**CI-friendly**

Evaluation results should be machine-readable and capable of failing a build when quality requirements are not met.

**Production-focused**

Evaluation should include more than a single accuracy number: performance, data quality, drift, model version, and reproducibility all matter.

---

## Development

Clone the repository and install it in editable mode:

```powershell
python -m pip install -e .
```

Run the test suite:

```powershell
pytest
```

Run the examples:

```powershell
python examples/basic_classification.py

python examples/regression.py

python examples/drift.py

python examples/versioning.py
```

---

## Project Status

EvalKit is currently under active development.

The core evaluation, monitoring, comparison, versioning, CLI, and CI/CD workflows are implemented and covered by automated validation.

Current validation includes:

```text
5 tests passed
Classification example     ✓
Regression example         ✓
CLI check/compare/history  ✓
CI exit codes 0/1          ✓
Custom metrics             ✓
Custom rules               ✓
Monitoring hooks           ✓
Edge cases                 ✓
```

---

## License

EvalKit is released under the [MIT License](LICENSE).

---

**EvalKit — evaluate models like production systems, not just notebooks.**
