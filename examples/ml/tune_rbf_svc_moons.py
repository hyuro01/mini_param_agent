"""A visibly tunable non-linear scikit-learn experiment.

Unlike the logistic-regression example, an RBF SVC on noisy moon-shaped data is
sensitive to both C and gamma. It is useful for confirming that an Optuna search
produces a measurable validation improvement.
"""

import json
import os
from pathlib import Path

from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ML_EXPERIMENT_PARAMS_START
ML_EXPERIMENT_PARAMS = {'C': 8.471801418819982, 'gamma': 0.6251373574521749}
# ML_EXPERIMENT_PARAMS_END


def main() -> None:
    params = (
        json.loads(os.environ["ML_EXPERIMENT_PARAMS"])
        if "ML_EXPERIMENT_PARAMS" in os.environ
        else ML_EXPERIMENT_PARAMS
    )
    seed = int(os.environ.get("ML_EXPERIMENT_SEED", "42"))
    features, labels = make_moons(n_samples=1_400, noise=0.30, random_state=seed)
    x_train, x_validation, y_train, y_validation = train_test_split(
        features, labels, test_size=0.25, random_state=seed, stratify=labels
    )

    model = make_pipeline(
        StandardScaler(),
        SVC(
            C=float(params.get("C", 1.0)),
            gamma=float(params.get("gamma", 1.0)),
            probability=True,
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
