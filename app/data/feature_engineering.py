"""
feature_engineering.py — Creating meaningful features from raw data.

WHY FEATURE ENGINEERING MATTERS:
Raw columns like "tenure" or "MonthlyCharges" are useful, but derived features
often carry more signal. This module creates features that capture patterns
the model wouldn't easily discover on its own.

INTERVIEW TIP: Always explain the business logic behind each feature.
A good feature = a hypothesis about why it predicts churn.
"""

import logging
import pandas as pd
import numpy as np
from config import TENURE_BINS, TENURE_LABELS

logger = logging.getLogger(__name__)


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Tenure (months) as a raw number is hard for tree models to split
    efficiently for business-meaningful segments. Bucketing makes the
    "new customer" effect explicit:
      - new (0–12 months): High churn risk — still evaluating the service
      - developing (12–24 months): Moderate risk
      - established (24–60 months): Lower risk — invested in service
      - loyal (60+ months): Lowest risk — committed long-term users

    This is a form of domain knowledge injection.
    """
    df = df.copy()
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=False
    )
    # One-hot encode the bucket (ordinal encoding would imply false ordering)
    bucket_dummies = pd.get_dummies(df["tenure_bucket"], prefix="tenure_bucket")
    df = pd.concat([df, bucket_dummies], axis=1)
    df = df.drop(columns=["tenure_bucket"])
    logger.debug("Added tenure bucket features")
    return df


def add_charge_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Monthly-to-total charge ratio reveals billing pattern anomalies.

    If MonthlyCharges is high relative to TotalCharges, the customer is new
    (short tenure) and paying a lot — classic churn risk profile.
    Customers who've paid a lot in total tend to be more invested.

    Formula: MonthlyCharges / (TotalCharges + 1)
    The +1 avoids division by zero for brand-new customers.
    """
    df = df.copy()
    df["charge_ratio"] = df["MonthlyCharges"] / (df["TotalCharges"] + 1)
    logger.debug("Added charge_ratio feature")
    return df


def add_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Customers who subscribe to many services are more embedded in the
    ecosystem — harder to churn. A customer with 6 services loses more
    by leaving than one with 1 service.

    This feature captures "stickiness" — a key churn predictor.
    """
    df = df.copy()
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    available = [c for c in service_cols if c in df.columns]
    df["service_count"] = df[available].sum(axis=1)
    logger.debug(f"Added service_count using {len(available)} service columns")
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: TotalCharges / tenure gives us average monthly spend, which is
    different from the current MonthlyCharges if a customer changed plans.
    A spike in current vs historical spend signals dissatisfaction.
    """
    df = df.copy()
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    logger.debug("Added avg_monthly_spend feature")
    return df


def add_charge_increase(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: If current monthly charge is significantly higher than average
    historical spend, the customer likely got a price increase — a major
    churn trigger. This feature captures that delta.
    """
    df = df.copy()
    df["charge_increase"] = df["MonthlyCharges"] - df["avg_monthly_spend"]
    logger.debug("Added charge_increase feature")
    return df


def add_contract_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Contract type is the single strongest predictor of churn.
    Month-to-month customers can leave anytime; two-year customers are locked in.
    We create an explicit risk score:
      - Month-to-month → 2 (high risk)
      - One year       → 1 (medium risk)
      - Two year       → 0 (low risk)

    This captures the ordinal relationship that one-hot encoding would lose.
    """
    df = df.copy()
    # Look for the one-hot encoded contract columns created during preprocessing
    if "Contract_One year" in df.columns and "Contract_Two year" in df.columns:
        # Month-to-month is the dropped (reference) category
        df["contract_risk"] = (
            2
            - df["Contract_One year"].astype(int)
            - 2 * df["Contract_Two year"].astype(int)
        )
        logger.debug("Added contract_risk from one-hot columns")
    elif "Contract" in df.columns:
        contract_map = {"Month-to-month": 2, "One year": 1, "Two year": 0}
        df["contract_risk"] = df["Contract"].map(contract_map).fillna(1)
        logger.debug("Added contract_risk from raw Contract column")
    return df


def add_paperless_tech_combo(df: pd.DataFrame) -> pd.DataFrame:
    """
    WHY: Customers who use paperless billing + no tech support are often
    self-service customers who price-compare online — higher churn risk.
    This interaction feature captures a specific risk persona.
    """
    df = df.copy()
    if "PaperlessBilling" in df.columns and "TechSupport" in df.columns:
        df["paperless_no_support"] = (
            (df["PaperlessBilling"] == 1) & (df["TechSupport"] == 0)
        ).astype(int)
        logger.debug("Added paperless_no_support interaction feature")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline.
    Call this AFTER preprocess_dataframe().

    Returns a DataFrame with all engineered features added.
    """
    logger.info("Starting feature engineering...")
    df = add_avg_monthly_spend(df)     # must come before charge_increase
    df = add_charge_increase(df)
    df = add_charge_ratio(df)
    df = add_service_count(df)
    df = add_tenure_bucket(df)
    df = add_contract_risk(df)
    df = add_paperless_tech_combo(df)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
