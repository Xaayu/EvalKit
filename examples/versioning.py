from sklearn.ensemble import RandomForestClassifier
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evalkit.versioning import generate_model_version


model = RandomForestClassifier(
    n_estimators=120,
    random_state=42
)

version = generate_model_version(model)

print("Model:", model.__class__.__name__)
print("Version:", version)