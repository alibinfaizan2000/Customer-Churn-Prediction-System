"""
explainability.py — SHAP-based model explanation module.

WHY SHAP (SHapley Additive exPlanations):
SHAP is grounded in game theory (Shapley values from cooperative game theory).
It assigns each feature a "contribution" to a prediction in a mathematically
fair way — features that cooperate to make a prediction each get credit.

INTERVIEW TALKING POINTS:
1. SHAP is model-agnostic — works with any model
2. Additive: feature contributions sum to the prediction offset from baseline
3. Consistent: if a feature matters more, its SHAP value is always larger
4. SHAP > LIME for global explanations (LIME only explains locally)
5. For tree models (XGBoost, RF), SHAP is exact and fast via TreeExplainer

BUSINESS VALUE:
- Regulators often require model explanations (GDPR "right to explanation")
- Support agents can show customers WHY they're flagged for churn
- Data scientists can debug models and find data leakage
"""

import logging
import numpy as np
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Falling back to feature importance explanations.")


def get_shap_explainer(model: Any, X_background: pd.DataFrame | None = None):
    """
    Create the right SHAP explainer for the model type.

    WHY DIFFERENT EXPLAINERS:
    - TreeExplainer: optimized for tree-based models (XGBoost, RF, LightGBM)
      Uses exact computation via tree structure — very fast
    - LinearExplainer: for linear models (LogReg, Ridge)
    - KernelExplainer: model-agnostic fallback — slow but works on anything

    We detect the model type automatically to pick the right one.
    """
    if not SHAP_AVAILABLE:
        return None

    model_type = type(model).__name__

    # Handle sklearn Pipeline — unwrap to get the actual model
    actual_model = model
    if hasattr(model, "named_steps"):
        # It's a Pipeline — get the last step
        actual_model = list(model.named_steps.values())[-1]
        model_type = type(actual_model).__name__

    logger.info(f"Creating SHAP explainer for model type: {model_type}")

    try:
        if model_type in ("XGBClassifier", "RandomForestClassifier",
                          "GradientBoostingClassifier", "LGBMClassifier",
                          "DecisionTreeClassifier"):
            # Tree models → exact TreeExplainer
            explainer = shap.TreeExplainer(actual_model)
            logger.info("Using TreeExplainer (fast, exact)")

        elif model_type in ("LogisticRegression", "LinearSVC", "Ridge"):
            # Linear models → LinearExplainer
            if X_background is None:
                logger.warning("LinearExplainer needs background data. Using zeros.")
                X_background = pd.DataFrame(
                    np.zeros((1, X_background.shape[1] if X_background is not None else 1))
                )
            explainer = shap.LinearExplainer(actual_model, X_background)
            logger.info("Using LinearExplainer")

        else:
            # Generic fallback — slow, requires background data
            logger.warning(f"Unknown model type {model_type}, using KernelExplainer (slow)")
            if X_background is None:
                raise ValueError("KernelExplainer requires background data")
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_background, 100)  # sample 100 for speed
            )

        return explainer

    except Exception as e:
        logger.error(f"Failed to create SHAP explainer: {e}")
        return None


def get_shap_values_for_instance(
    explainer,
    X_instance: pd.DataFrame,
    feature_names: list[str]
) -> dict:
    """
    Compute SHAP values for a single prediction and return structured output.

    Returns a dict mapping feature names to their SHAP values.
    Positive SHAP → feature pushes towards churn
    Negative SHAP → feature pushes away from churn

    WHY TOP_N=5: Showing all 30+ features is overwhelming. The top 5 SHAP
    features capture >80% of the prediction explanation in practice.
    """
    if explainer is None or not SHAP_AVAILABLE:
        return get_fallback_explanation(X_instance, feature_names)

    try:
        # Ensure correct input format
        if hasattr(explainer, "shap_values"):
            shap_vals = explainer.shap_values(X_instance)
        else:
            shap_vals = explainer(X_instance).values

        # Handle different output shapes
        # Binary classification: shap_values can be [class0, class1] or just class1
        if isinstance(shap_vals, list):
            # Take class 1 (churn) values
            vals = shap_vals[1][0] if len(shap_vals) > 1 else shap_vals[0][0]
        elif len(shap_vals.shape) == 3:
            vals = shap_vals[0, :, 1]  # (n_samples, n_features, n_classes)
        elif len(shap_vals.shape) == 2:
            vals = shap_vals[0]        # (n_samples, n_features)
        else:
            vals = shap_vals

        # Build explanation dict
        explanation = {
            feature_names[i]: float(vals[i])
            for i in range(min(len(vals), len(feature_names)))
        }
        return explanation

    except Exception as e:
        logger.error(f"SHAP computation failed: {e}. Using fallback.")
        return get_fallback_explanation(X_instance, feature_names)


def get_top_features(
    shap_explanation: dict,
    top_n: int = 5
) -> list[dict]:
    """
    Extract the top N most impactful features from SHAP values.

    Returns a list of dicts sorted by absolute impact (most impactful first):
    [
      {"feature": "Contract_Two year", "shap_value": -0.45, "direction": "decreases churn"},
      {"feature": "tenure", "shap_value": -0.31, "direction": "decreases churn"},
      ...
    ]
    """
    sorted_features = sorted(
        shap_explanation.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    result = []
    for feature, shap_val in sorted_features:
        result.append({
            "feature":    feature,
            "shap_value": round(shap_val, 4),
            "impact":     round(abs(shap_val), 4),
            "direction":  "increases churn" if shap_val > 0 else "decreases churn"
        })
    return result


def get_fallback_explanation(
    X_instance: pd.DataFrame,
    feature_names: list[str]
) -> dict:
    """
    Fallback when SHAP is unavailable: use simple feature values normalized.
    Not as rigorous as SHAP but gives the API something to return.
    """
    vals = X_instance.values.flatten()
    explanation = {}
    for i, name in enumerate(feature_names):
        if i < len(vals):
            explanation[name] = float(vals[i]) * 0.01  # crude proxy
    return explanation


def format_explanation_for_api(top_features: list[dict]) -> list[dict]:
    """
    Format SHAP explanation into a clean, human-readable API response.
    Makes the explanation understandable without ML knowledge.
    """
    FEATURE_DESCRIPTIONS = {
        "Contract_Two year":         "Long-term contract",
        "Contract_One year":         "One-year contract",
        "tenure":                    "Customer tenure (months)",
        "MonthlyCharges":            "Monthly charge amount",
        "TotalCharges":              "Total amount paid",
        "charge_ratio":              "Monthly vs total charge ratio",
        "service_count":             "Number of subscribed services",
        "contract_risk":             "Contract risk level",
        "charge_increase":           "Recent charge increase",
        "avg_monthly_spend":         "Historical avg monthly spend",
        "paperless_no_support":      "Uses paperless billing without tech support",
        "InternetService_Fiber optic": "Fiber optic internet user",
        "PaymentMethod_Electronic check": "Pays by electronic check",
    }

    formatted = []
    for item in top_features:
        feature = item["feature"]
        friendly_name = FEATURE_DESCRIPTIONS.get(feature, feature.replace("_", " ").title())
        formatted.append({
            "feature":      feature,
            "display_name": friendly_name,
            "shap_value":   item["shap_value"],
            "direction":    item["direction"],
            "impact":       item["impact"]
        })
    return formatted
