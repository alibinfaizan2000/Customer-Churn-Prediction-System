"""
preprocess.py — Data cleaning and preprocessing pipeline.

WHY THIS MODULE EXISTS:
Raw data is never ML-ready. This module handles:
  1. Loading the Telco Churn dataset
  2. Fixing data type issues (TotalCharges has hidden spaces)
  3. Handling missing values
  4. Encoding categorical variables
  5. Returning a clean DataFrame ready for feature engineering

INTERVIEW TIP: Always explain that preprocessing is separate from feature engineering.
Preprocessing = making data valid. Feature engineering = making data useful.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


# ─── Column groups ────────────────────────────────────────────────────────────
# Separating column types makes the code readable and maintainable.
BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
]

CATEGORICAL_COLS = ["InternetService", "Contract", "PaymentMethod", "gender"]

TARGET_COL = "Churn"
DROP_COLS  = ["customerID"]  # Not predictive — just an identifier


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """Load the raw Telco CSV file."""
    logger.info(f"Loading raw data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: TotalCharges is stored as a string with spaces for new customers
    (tenure=0). We must convert it to float and fill NaN with 0.
    This is a classic real-world data quality issue.
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_count = df["TotalCharges"].isna().sum()
    if missing_count > 0:
        logger.warning(f"Filling {missing_count} NaN TotalCharges with 0")
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Columns with Yes/No or Male/Female can be mapped to 1/0.
    This is simpler than one-hot encoding for binary variables
    and reduces dimensionality.
    """
    df = df.copy()
    # Standard Yes/No mapping
    yes_no_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}

    # Columns like MultipleLines can also have "No phone service" — treat as No
    no_service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in no_service_cols:
        if col in df.columns:
            df[col] = df[col].replace("No phone service", "No")
            df[col] = df[col].replace("No internet service", "No")
            df[col] = df[col].map(yes_no_map)

    simple_binary = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    for col in simple_binary:
        if col in df.columns:
            df[col] = df[col].map(yes_no_map)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(gender_map)

    return df


def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Multi-class categoricals need one-hot encoding so the model
    doesn't assume ordinal relationships between categories.
    e.g., "DSL" < "Fiber optic" is meaningless — they're unordered.

    We use drop_first=True to avoid the dummy variable trap
    (perfect multicollinearity in linear models).
    """
    df = df.copy()
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns and c != "gender"]
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    logger.info(f"One-hot encoded columns: {cat_cols}")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Churn column from Yes/No to 1/0."""
    df = df.copy()
    # Use numpy where for robust handling of ArrowStringArray (pandas 2.x)
    col = df[TARGET_COL]
    import numpy as np
    df[TARGET_COL] = np.where(
        np.array(col) == "Yes", 1, 0
    ).astype(int)
    return df


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that add no predictive value."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Applies all cleaning steps in order.

    Returns a clean DataFrame ready for feature engineering.
    """
    logger.info("Starting preprocessing pipeline...")
    df = fix_total_charges(df)
    df = encode_target(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = drop_unnecessary_columns(df)
    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


def preprocess_single_input(input_dict: dict) -> pd.DataFrame:
    """
    Preprocess a single customer's data from the API request.
    Creates a one-row DataFrame and runs the same pipeline.

    WHY: Using the same pipeline for both training and inference
    prevents training-serving skew — a major source of bugs in production.
    """
    df = pd.DataFrame([input_dict])
    df = fix_total_charges(df)
    df = encode_binary_columns(df)
    df = encode_categorical_columns(df)
    df = drop_unnecessary_columns(df)
    return df
