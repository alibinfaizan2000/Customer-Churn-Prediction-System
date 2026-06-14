"""
mlflow_tracking.py — MLflow experiment tracking integration.

WHY MLFLOW:
ML development without experiment tracking is like coding without version control.
MLflow lets you:
  - Record every training run with metrics, parameters, and artifacts
  - Compare runs visually in the MLflow UI
  - Reproduce any experiment exactly
  - Register models with versioning

HOW TO VIEW: Run `mlflow ui` in the project root, then open http://localhost:5000

INTERVIEW TIP: MLflow is the most common open-source experiment tracker.
Cloud equivalents include: Weights & Biases, Comet ML, Neptune.ai
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.sklearn
    import mlflow.xgboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Experiment tracking disabled.")


def setup_mlflow(tracking_uri: str, experiment_name: str) -> bool:
    """Initialize MLflow with tracking URI and experiment name."""
    if not MLFLOW_AVAILABLE:
        return False
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        logger.info(f"MLflow experiment: {experiment_name}")
        return True
    except Exception as e:
        logger.error(f"MLflow setup failed: {e}")
        return False


def log_training_run(
    model_name: str,
    model: Any,
    params: dict,
    metrics: dict,
    feature_names: list[str],
    run_name: str | None = None
) -> str | None:
    """
    Log a complete training run to MLflow.

    Logs:
    - Parameters (hyperparameters)
    - Metrics (accuracy, F1, ROC-AUC, etc.)
    - Model artifact (the trained model file)
    - Feature names

    Returns: MLflow run ID for reference
    """
    if not MLFLOW_AVAILABLE:
        logger.info(f"[MOCK MLflow] Would log run: {model_name}")
        logger.info(f"[MOCK MLflow] Metrics: {metrics}")
        return None

    try:
        with mlflow.start_run(run_name=run_name or model_name) as run:
            # Log hyperparameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics({
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float))
            })

            # Log feature count
            mlflow.log_param("n_features", len(feature_names))
            mlflow.log_param("model_type", model_name)

            # Log model artifact
            import xgboost as xgb
            if isinstance(model, xgb.XGBClassifier):
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")

            # Log feature names as a text artifact
            import tempfile, os
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("\n".join(feature_names))
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, "feature_names.txt")
            os.unlink(tmp_path)

            run_id = run.info.run_id
            logger.info(f"MLflow run logged: {run_id}")
            return run_id

    except Exception as e:
        logger.error(f"MLflow logging failed: {e}")
        return None


def log_all_models(all_metrics: dict, trained_models: dict, feature_names: list[str]) -> None:
    """Log all trained models to MLflow for comparison."""
    for model_name, metrics in all_metrics.items():
        if model_name in trained_models:
            model = trained_models[model_name]

            # Get model params safely
            try:
                params = model.get_params() if hasattr(model, "get_params") else {}
                # Filter to only simple types
                params = {k: str(v) for k, v in params.items() if v is not None}
            except Exception:
                params = {"model": model_name}

            log_training_run(
                model_name=model_name,
                model=model,
                params=params,
                metrics=metrics,
                feature_names=feature_names,
                run_name=model_name
            )
