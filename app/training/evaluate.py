"""
evaluate.py — Generate a comprehensive model evaluation report.

Run this after training to get a full breakdown of model performance,
including confusion matrix, ROC curve data, PR curve, and SHAP summary.

Usage: python app/training/evaluate.py
"""

import sys
import json
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    RAW_DATA_PATH, MODEL_PATH, FEATURE_NAMES_PATH,
    MODELS_DIR, TEST_SIZE, RANDOM_STATE
)
from app.data.download_data import get_or_create_dataset
from app.data.preprocess import preprocess_dataframe
from app.data.feature_engineering import engineer_features

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    classification_report
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_test_data():
    """Reproduce the exact same test split used during training."""
    df_raw   = get_or_create_dataset(RAW_DATA_PATH)
    df_clean = preprocess_dataframe(df_raw)
    df_feat  = engineer_features(df_clean)

    feature_names = joblib.load(FEATURE_NAMES_PATH)
    X = df_feat[feature_names]
    y = df_feat["Churn"].astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_test, y_test


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Pretty-print a 2x2 confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    print("\n  Confusion Matrix:")
    print("                  Predicted")
    print("                  No Churn  Churn")
    print(f"  Actual No Churn   {tn:5d}   {fp:5d}")
    print(f"  Actual Churn      {fn:5d}   {tp:5d}")
    print(f"\n  True Positives  (caught churners)  : {tp}")
    print(f"  False Negatives (missed churners)  : {fn}  ← minimize this")
    print(f"  False Positives (false alarms)     : {fp}")
    print(f"  True Negatives  (correct stays)    : {tn}")


def generate_report() -> dict:
    """Full evaluation report generation."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run: python app/training/train.py"
        )

    logger.info("Loading model and test data...")
    model        = joblib.load(MODEL_PATH)
    X_test, y_test = load_test_data()

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ── Core metrics ──────────────────────────────────────────────────────────
    metrics = {
        "n_test_samples": len(y_test),
        "churn_rate_test": float(y_test.mean()),
        "accuracy":        round(float(accuracy_score(y_test, y_pred)), 4),
        "precision":       round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":          round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1":              round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc":         round(float(roc_auc_score(y_test, y_prob)), 4),
        "pr_auc":          round(float(average_precision_score(y_test, y_prob)), 4),
    }

    print("\n" + "="*60)
    print("  CHURN PREDICTION MODEL — EVALUATION REPORT")
    print("="*60)
    print(f"\n  Test samples : {metrics['n_test_samples']}")
    print(f"  Churn rate   : {metrics['churn_rate_test']:.1%}")
    print(f"\n  {'Metric':<18} {'Score':>8}")
    print("  " + "-"*28)
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        label = k.upper().replace("_", "-")
        print(f"  {label:<18} {metrics[k]:>8.4f}")

    # ── PR-AUC note ───────────────────────────────────────────────────────────
    print(f"""
  WHY PR-AUC = {metrics['pr_auc']:.4f} MATTERS:
  On imbalanced data (27% churn), ROC-AUC can be optimistic because
  it accounts for true negatives. PR-AUC focuses only on the positive
  class (churners) — harder to game, more realistic for business use.
  A PR-AUC of {metrics['pr_auc']:.2f} vs random baseline of 0.27 means the
  model is {metrics['pr_auc']/0.27:.1f}x better than guessing on recall/precision.
    """)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)

    # ── Classification report ─────────────────────────────────────────────────
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["No Churn", "Churn"],
        digits=4
    ))

    # ── SHAP feature importance ───────────────────────────────────────────────
    print("  Top Feature Importances (model-native):")
    try:
        feature_names = joblib.load(FEATURE_NAMES_PATH)
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "named_steps"):
            inner = list(model.named_steps.values())[-1]
            importances = (
                inner.feature_importances_
                if hasattr(inner, "feature_importances_")
                else np.abs(inner.coef_[0])
            )
        else:
            importances = None

        if importances is not None:
            feat_imp = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1], reverse=True
            )[:10]
            for feat, imp in feat_imp:
                bar = "█" * int(imp * 100)
                print(f"    {feat:<35} {imp:.4f}  {bar}")
    except Exception as e:
        logger.warning(f"Could not compute feature importances: {e}")

    # ── Threshold analysis ────────────────────────────────────────────────────
    print("\n  Threshold Sensitivity (business tuning):")
    print(f"  {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Flagged%':>10}")
    print("  " + "-"*55)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred_t = (y_prob >= thresh).astype(int)
        prec = precision_score(y_test, y_pred_t, zero_division=0)
        rec  = recall_score(y_test, y_pred_t, zero_division=0)
        f1   = f1_score(y_test, y_pred_t, zero_division=0)
        flagged = y_pred_t.mean()
        marker = " ← default" if thresh == 0.5 else ""
        print(f"  {thresh:>10.1f} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f} {flagged:>9.1%}{marker}")

    print("""
  THRESHOLD TUNING GUIDANCE:
  - Lower threshold (0.3-0.4): Higher recall — catch more churners, more false alarms
  - Higher threshold (0.6-0.7): Higher precision — fewer false alarms, miss more churners
  - Business decision: what costs more? A missed churner or a wasted retention call?
    """)

    # Save report to file
    report_path = MODELS_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"\n  Report saved → {report_path}")

    print("="*60)
    return metrics


if __name__ == "__main__":
    generate_report()
