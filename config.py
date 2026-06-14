"""
config.py — Central configuration for the entire project.

WHY: Having a single config file avoids hardcoded values scattered across files.
This is a standard production practice — change one value here, it propagates everywhere.
In real projects this often comes from environment variables (.env files) or
cloud secret managers. We keep it simple here but structured correctly.
"""

import os
from pathlib import Path

# ─── Project root ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# ─── Data paths ──────────────────────────────────────────────────────────────
DATA_DIR        = BASE_DIR / "app" / "data"
RAW_DATA_PATH   = DATA_DIR / "telco_churn.csv"
PROCESSED_DATA  = DATA_DIR / "processed_churn.csv"

# ─── Model paths ─────────────────────────────────────────────────────────────
MODELS_DIR      = BASE_DIR / "app" / "models"
MODEL_PATH      = MODELS_DIR / "best_model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

# ─── MLflow tracking ─────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT   = "churn_prediction"

# ─── Training settings ───────────────────────────────────────────────────────
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
CV_FOLDS        = 5

# ─── API settings ────────────────────────────────────────────────────────────
API_HOST        = os.getenv("API_HOST", "0.0.0.0")
API_PORT        = int(os.getenv("API_PORT", 8000))

# ─── UI settings ─────────────────────────────────────────────────────────────
STREAMLIT_PORT  = int(os.getenv("STREAMLIT_PORT", 8501))
API_BASE_URL    = os.getenv("API_BASE_URL", "http://localhost:8000")

# ─── Monitoring ──────────────────────────────────────────────────────────────
PREDICTIONS_LOG = BASE_DIR / "predictions_log.jsonl"
DRIFT_THRESHOLD = 0.1   # alert if mean churn prob shifts by more than 10%

# ─── Risk thresholds ─────────────────────────────────────────────────────────
# These define what probability counts as Low / Medium / High churn risk.
# In a real product these would be tuned with the business team.
RISK_LOW_MAX    = 0.35
RISK_MED_MAX    = 0.65
# > 0.65 → High risk

# ─── Feature engineering constants ───────────────────────────────────────────
TENURE_BINS   = [0, 12, 24, 60, float("inf")]
TENURE_LABELS = ["new", "developing", "established", "loyal"]

# ─── Authentication ───────────────────────────────────────────────────────────
# Keys are read from environment variables — never hardcode in source.
# Generate fresh keys with: python app/api/auth.py
# Defaults below are for local development only — CHANGE IN PRODUCTION.
ADMIN_API_KEY    = os.getenv("ADMIN_API_KEY",    "churn-admin-key-change-in-production")
SERVICE_API_KEY  = os.getenv("SERVICE_API_KEY",  "churn-service-key-change-in-production")
READONLY_API_KEY = os.getenv("READONLY_API_KEY", "churn-readonly-key-change-in-production")

# The key the Streamlit UI uses to call the API (service role is sufficient)
UI_API_KEY       = os.getenv("UI_API_KEY", SERVICE_API_KEY)
