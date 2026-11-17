
import mlflow
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TRACKING_DIR = BASE_DIR / "mlflow" / "tracking"

TRACKING_DIR.mkdir(parents=True, exist_ok=True)

# Use SQLite database instead of the old file-based backend
DB_PATH = TRACKING_DIR / "mlflow.db"

tracking_uri = f"sqlite:///{DB_PATH.as_posix()}"

mlflow.set_tracking_uri(tracking_uri)

# Create or select experiment
mlflow.set_experiment("Retail_Demand_Forecasting")

# Existing model validation results
experiments = [
    {
        "name": "Random_Forest",
        "file": "random_forest_model.pkl",
        "params": {
            "n_estimators": 50,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "random_state": 42,
            "n_jobs": -1
        },
        "metrics": {
            "MAE": 109.2781,
            "RMSE": 374.7767,
            "R2": 0.9098720
        }
    },
    {
        "name": "HistGradientBoosting",
        "file": "hist_gradient_boosting_model.pkl",
        "params": {
            "max_iter": 150,
            "learning_rate": 0.1,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
            "random_state": 42
        },
        "metrics": {
            "MAE": 134.140447,
            "RMSE": 333.925357,
            "R2": 0.9284494
        }
    }
]

# Track both models
for experiment in experiments:

    model_path = MODEL_DIR / experiment["file"]

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        continue

    with mlflow.start_run(run_name=experiment["name"]):

        mlflow.log_params(experiment["params"])

        mlflow.log_metrics(experiment["metrics"])

        mlflow.set_tag("model_type", experiment["name"])
        mlflow.set_tag(
            "project",
            "Retail Demand Forecasting"
        )

        mlflow.log_artifact(
            str(model_path),
            artifact_path="model_bundle"
        )

        print(f"Successfully logged: {experiment['name']}")

print("\nMLflow experiment tracking completed!")
print(f"Database: {DB_PATH}")
