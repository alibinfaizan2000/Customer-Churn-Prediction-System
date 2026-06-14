"""
predict.py — Model loading and inference logic.

WHY SEPARATE FROM train.py:
Training and inference have very different concerns:
  - Training: needs all data, is slow, runs rarely
  - Inference: needs a loaded model, must be fast, runs constantly

This module handles the inference path only.
It's imported by both the FastAPI backend and the Streamlit UI.
"""

import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    MODEL_PATH, FEATURE_NAMES_PATH, MODELS_DIR,
    RISK_LOW_MAX, RISK_MED_MAX
)
from app.data.preprocess import preprocess_single_input
from app.data.feature_engineering import engineer_features
from app.utils.explainability import (
    get_shap_explainer, get_shap_values_for_instance,
    get_top_features, format_explanation_for_api
)

logger = logging.getLogger(__name__)

# ─── Singleton model cache ────────────────────────────────────────────────────
# WHY SINGLETON: Loading the model from disk on every request would be very slow.
# We load once on startup and keep in memory. This is standard practice.
_model = None
_feature_names: list[str] = []
_explainer = None
_meta: dict = {}


def load_artifacts() -> None:
    """Load model and artifacts into module-level cache."""
    global _model, _feature_names, _explainer, _meta

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Please run: python app/training/train.py"
        )

    logger.info(f"Loading model from {MODEL_PATH}...")
    _model = joblib.load(MODEL_PATH)

    logger.info(f"Loading feature names from {FEATURE_NAMES_PATH}...")
    _feature_names = joblib.load(FEATURE_NAMES_PATH)

    # Load metadata
    meta_path = MODELS_DIR / "model_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            _meta = json.load(f)

    # Initialize SHAP explainer
    logger.info("Initializing SHAP explainer...")
    _explainer = get_shap_explainer(_model)

    logger.info("✓ All artifacts loaded successfully")


def get_model():
    """Return cached model, loading if necessary."""
    global _model
    if _model is None:
        load_artifacts()
    return _model


def get_feature_names() -> list[str]:
    """Return cached feature names."""
    global _feature_names
    if not _feature_names:
        load_artifacts()
    return _feature_names


def get_meta() -> dict:
    """Return model metadata."""
    return _meta


def _assign_risk_level(prob: float) -> str:
    """
    Convert a churn probability to a business-friendly risk tier.

    WHY THREE TIERS:
    Binary yes/no is too coarse for customer success teams.
    Three tiers allow targeted interventions:
    - Low: standard engagement
    - Medium: proactive outreach
    - High: retention offer / escalation
    """
    if prob <= RISK_LOW_MAX:
        return "Low"
    elif prob <= RISK_MED_MAX:
        return "Medium"
    else:
        return "High"


def _prepare_features(raw_input: dict) -> pd.DataFrame:
    """
    Run the same preprocessing + feature engineering as training.

    WHY THIS MATTERS: Any discrepancy between training-time and
    inference-time transformations causes "training-serving skew"
    — the model sees different data than it was trained on.
    Using the same functions prevents this.
    """
    # Preprocess single row
    df = preprocess_single_input(raw_input)

    # Feature engineering (don't need to drop Churn — it won't exist in raw_input)
    df = engineer_features(df)

    # Align columns to training feature set
    feature_names = get_feature_names()
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0  # missing features default to 0

    df = df[feature_names]
    return df


def predict_single(customer: dict) -> dict:
    """
    Make a single churn prediction with SHAP explanation.

    Input: raw customer dict (as received from the API)
    Output: {churn_probability, risk_level, explanation, ...}
    """
    model = get_model()
    feature_names = get_feature_names()

    # Prepare features
    X = _prepare_features(customer)

    # Predict
    churn_prob = float(model.predict_proba(X)[0, 1])
    risk_level = _assign_risk_level(churn_prob)

    # SHAP explanation
    explanation = []
    try:
        if _explainer is not None:
            shap_vals = get_shap_values_for_instance(_explainer, X, feature_names)
            top_features = get_top_features(shap_vals, top_n=5)
            explanation = format_explanation_for_api(top_features)
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")

    return {
        "churn_probability": round(churn_prob, 4),
        "risk_level":        risk_level,
        "prediction":        1 if churn_prob >= 0.5 else 0,
        "explanation":       explanation,
    }


def predict_batch(customers: list[dict]) -> list[dict]:
    """
    Make predictions for multiple customers at once.
    More efficient than calling predict_single in a loop
    because we batch the model.predict_proba call.

    WHY: predict_proba on a batch is faster than N individual calls
    due to vectorization in numpy/XGBoost.
    """
    model = get_model()
    feature_names = get_feature_names()

    results = []
    # Prepare all rows
    dfs = [_prepare_features(c) for c in customers]
    X_batch = pd.concat(dfs, ignore_index=True)
    X_batch = X_batch[feature_names]

    probs = model.predict_proba(X_batch)[:, 1]

    for i, (customer, prob) in enumerate(zip(customers, probs)):
        churn_prob = float(prob)
        risk_level = _assign_risk_level(churn_prob)

        # SHAP for each row (can be slow for large batches)
        explanation = []
        try:
            if _explainer is not None:
                row_df = X_batch.iloc[[i]]
                shap_vals = get_shap_values_for_instance(_explainer, row_df, feature_names)
                top_features = get_top_features(shap_vals, top_n=3)  # fewer for batch
                explanation = format_explanation_for_api(top_features)
        except Exception:
            pass

        results.append({
            "index":             i,
            "churn_probability": round(churn_prob, 4),
            "risk_level":        risk_level,
            "prediction":        1 if churn_prob >= 0.5 else 0,
            "explanation":       explanation,
        })

    return results
