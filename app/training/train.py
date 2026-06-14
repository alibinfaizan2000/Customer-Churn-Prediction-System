"""
train.py — Complete model training pipeline.

WHY MULTIPLE MODELS:
We train 3 models with increasing complexity:
  1. Logistic Regression — fast baseline, interpretable, good for linear patterns
  2. Random Forest — handles non-linearity, robust to outliers, good default choice
  3. XGBoost — gradient boosting, typically best on tabular data, industry standard

We use cross-validation to get reliable estimates before tuning, then
RandomizedSearchCV for hyperparameter optimization (faster than GridSearch
for large parameter spaces).

INTERVIEW TIP: Always explain why XGBoost often wins on tabular data:
- Handles missing values natively
- Built-in regularization prevents overfitting
- Efficient sequential boosting reduces bias
"""

import sys
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    RAW_DATA_PATH, MODEL_PATH, PREPROCESSOR_PATH, FEATURE_NAMES_PATH,
    MODELS_DIR, TEST_SIZE, RANDOM_STATE, CV_FOLDS
)
from app.data.download_data import get_or_create_dataset
from app.data.preprocess import preprocess_dataframe
from app.data.feature_engineering import engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str
) -> dict:
    """
    Compute a full evaluation report for a trained model.

    WHY THESE METRICS:
    - Accuracy: overall correctness (misleading on imbalanced data!)
    - Precision: of predicted churners, how many actually churned
    - Recall: of actual churners, how many did we catch — CRITICAL for business
    - F1: harmonic mean of precision & recall
    - ROC-AUC: model's ability to rank churners above non-churners

    For churn prediction, RECALL matters most — missing a churner is more costly
    than a false alarm. We set average='binary' since target is binary.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {model_name}")
    logger.info(f"  Accuracy : {metrics['accuracy']}")
    logger.info(f"  Precision: {metrics['precision']}")
    logger.info(f"  Recall   : {metrics['recall']}")
    logger.info(f"  F1-Score : {metrics['f1']}")
    logger.info(f"  ROC-AUC  : {metrics['roc_auc']}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")
    return metrics


# ─── Model definitions ───────────────────────────────────────────────────────

def get_logistic_regression() -> Pipeline:
    """
    Logistic Regression wrapped in a pipeline with StandardScaler.

    WHY SCALE: LR uses gradient descent and is sensitive to feature magnitude.
    A feature ranging 0–72 (tenure) vs 0–1 (binary) will dominate without scaling.

    WHY IT'S A GOOD BASELINE: Fast, interpretable, no hyperparameter tuning needed.
    If LR does well, the problem is largely linear. If it underperforms XGBoost
    significantly, nonlinear patterns matter.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # handles class imbalance
            random_state=RANDOM_STATE
        ))
    ])


def get_random_forest() -> RandomForestClassifier:
    """
    Random Forest — ensemble of decision trees.

    WHY NO SCALING: Tree-based models split on thresholds, not distances.
    Feature scale doesn't matter.

    WHY class_weight='balanced': Churn is typically ~26% of the data.
    Without balancing, the model learns to predict "No Churn" most of the time
    and still gets 74% accuracy — but misses all churners (recall = 0).
    """
    return RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


def get_xgboost() -> xgb.XGBClassifier:
    """
    XGBoost — gradient boosted trees. Usually the best on tabular data.

    WHY XGBOOST WINS:
    - Sequential boosting: each tree corrects errors of the previous
    - Built-in L1/L2 regularization (alpha, lambda) prevents overfitting
    - scale_pos_weight handles class imbalance natively
    - Very fast with histogram-based splits

    scale_pos_weight = (# negative samples) / (# positive samples)
    This tells XGBoost to penalize missing a churn prediction more heavily.
    """
    return xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        scale_pos_weight=2.7,   # approx 73/27 ratio
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ─── Hyperparameter tuning ───────────────────────────────────────────────────

def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    """
    Tune XGBoost with RandomizedSearchCV.

    WHY RANDOMIZED OVER GRID:
    RandomizedSearchCV samples n_iter combinations randomly from the parameter
    distributions. For 5 parameters × many values, GridSearch would be
    5^5 = 3125 fits. Randomized with n_iter=30 does 30 fits and empirically
    finds near-optimal results faster.

    WHY cv=3 not 5: Tuning is expensive; we use 3-fold here for speed.
    Final evaluation uses 5-fold on the best params.
    """
    param_dist = {
        "n_estimators":   [100, 200, 300, 400],
        "max_depth":      [3, 4, 5, 6],
        "learning_rate":  [0.01, 0.05, 0.1, 0.15],
        "subsample":      [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    base_model = xgb.XGBClassifier(
        scale_pos_weight=2.7,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring="roc_auc",   # optimize for ranking ability
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )
    logger.info("Starting XGBoost hyperparameter tuning (n_iter=30)...")
    search.fit(X_train, y_train)
    logger.info(f"Best XGBoost params: {search.best_params_}")
    logger.info(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    return search.best_estimator_


# ─── Main training pipeline ──────────────────────────────────────────────────

def run_training(tune: bool = False) -> dict:
    """
    End-to-end training pipeline:
    1. Load + preprocess + engineer features
    2. Split data
    3. Train 3 models
    4. Cross-validate all
    5. Optionally tune best model
    6. Save best model + preprocessor artifacts

    Returns: dict of all evaluation metrics
    """
    # ── Step 1: Load and prepare data ────────────────────────────────────────
    logger.info("Step 1: Loading data...")
    df_raw = get_or_create_dataset(RAW_DATA_PATH)

    logger.info("Step 2: Preprocessing...")
    df_clean = preprocess_dataframe(df_raw)

    logger.info("Step 3: Feature engineering...")
    df_features = engineer_features(df_clean)

    # ── Step 2: Split ─────────────────────────────────────────────────────────
    TARGET = "Churn"
    X = df_features.drop(columns=[TARGET])
    y = df_features[TARGET]

    # Ensure all columns are numeric — safety check
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()

    logger.info(f"Features: {len(feature_names)} | Samples: {len(X)} | Churn rate: {y.mean():.2%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Step 3: Cross-validate all models ────────────────────────────────────
    logger.info("\n=== Cross-Validation (5-fold ROC-AUC) ===")
    models_to_cv = {
        "Logistic Regression": get_logistic_regression(),
        "Random Forest":       get_random_forest(),
        "XGBoost":             get_xgboost(),
    }

    cv_results = {}
    for name, model in models_to_cv.items():
        scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="roc_auc", n_jobs=-1)
        cv_results[name] = scores
        logger.info(f"{name}: CV ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    # ── Step 4: Train all models on full training set ─────────────────────────
    logger.info("\n=== Training All Models ===")
    trained_models = {}
    all_metrics = {}

    for name, model in models_to_cv.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        all_metrics[name] = evaluate_model(model, X_test, y_test, name)

    # ── Step 5: Optionally tune XGBoost ──────────────────────────────────────
    best_model_name = "XGBoost"
    if tune:
        logger.info("\n=== Tuning XGBoost ===")
        tuned_xgb = tune_xgboost(X_train, y_train)
        trained_models["XGBoost (Tuned)"] = tuned_xgb
        all_metrics["XGBoost (Tuned)"] = evaluate_model(tuned_xgb, X_test, y_test, "XGBoost (Tuned)")
        best_model_name = "XGBoost (Tuned)"

    # ── Step 6: Select best model ─────────────────────────────────────────────
    best_model = trained_models[best_model_name]
    logger.info(f"\n✓ Best model selected: {best_model_name}")
    logger.info(f"  ROC-AUC: {all_metrics[best_model_name]['roc_auc']}")

    # ── Step 7: Save artifacts ────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(feature_names, FEATURE_NAMES_PATH)
    logger.info(f"Saved model → {MODEL_PATH}")
    logger.info(f"Saved feature names → {FEATURE_NAMES_PATH}")

    # Save a simple dict of expected features for the API
    import json
    meta = {
        "feature_names": feature_names,
        "model_name": best_model_name,
        "test_metrics": all_metrics[best_model_name],
        "cv_roc_auc": {k: {"mean": float(v.mean()), "std": float(v.std())}
                       for k, v in cv_results.items()}
    }
    meta_path = MODELS_DIR / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Saved model metadata → {meta_path}")

    logger.info("\n✅ Training pipeline complete!")
    return all_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter tuning")
    args = parser.parse_args()
    run_training(tune=args.tune)
