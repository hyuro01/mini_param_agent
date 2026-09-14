"""A minimal MLExperimentTool-compatible scikit-learn training script.

Run it directly for one baseline experiment, or let ``run_ml_experiment`` set
the ML_EXPERIMENT_* environment variables for each Optuna trial.
"""

import json
import os
from pathlib import Path

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main() -> None:
    params = json.loads(os.environ.get("ML_EXPERIMENT_PARAMS", "{}"))
    seed = int(os.environ.get("ML_EXPERIMENT_SEED", "42"))

    features, labels = make_classification(
        n_samples=1_200,
        n_features=20,
        n_informative=8,
        n_redundant=4,
        class_sep=0.85,
        flip_y=0.04,
        random_state=seed,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        features, labels, test_size=0.25, random_state=seed, stratify=labels
    )

    # Values are supplied by MLExperimentTool. Defaults make this executable
    # by itself and define a sensible baseline configuration.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(params.get("C", 1.0)),
            solver="liblinear",
            max_iter=int(params.get("max_iter", 200)),
            random_state=seed,
        ),
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_validation)
    predictions = model.predict(x_validation)
    metrics = {
        "val_accuracy": accuracy_score(y_validation, predictions),
        "val_log_loss": log_loss(y_validation, probabilities),
    }

    metrics_path = Path(os.environ.get("ML_EXPERIMENT_METRICS_PATH", "metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"ML_METRICS: {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
