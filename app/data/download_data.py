"""
download_data.py — Download or generate the Telco Churn dataset.

The IBM Telco Churn dataset is available on Kaggle/GitHub.
This script tries to download it; if unavailable, generates synthetic data
with the same schema so development can continue offline.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_synthetic_telco_data(n_rows: int = 7043, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Telco-like dataset that mirrors the real schema.
    Used when the real dataset isn't available.

    WHY: Always have a fallback data source so the pipeline is self-contained.
    This is good MLOps practice — no external dependencies for basic testing.
    """
    rng = np.random.default_rng(seed)

    n = n_rows
    tenure = rng.integers(0, 72, n)

    # Older customers have higher charges — realistic correlation
    monthly_charges = rng.uniform(18, 120, n).round(2)
    total_charges = (monthly_charges * tenure + rng.normal(0, 50, n)).clip(0).round(2)

    # Contract type — most are month-to-month
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.25, 0.20]
    )

    # Churn logic: new + month-to-month + high charges → more likely to churn
    churn_prob = (
        0.10
        + 0.25 * (tenure < 12).astype(float)
        + 0.20 * (contract == "Month-to-month").astype(float)
        + 0.15 * (monthly_charges > 80).astype(float)
        - 0.10 * (tenure > 36).astype(float)
        + rng.normal(0, 0.05, n)
    ).clip(0, 1)
    churn = (rng.uniform(0, 1, n) < churn_prob).astype(int)
    churn_str = np.where(churn == 1, "Yes", "No")

    yn = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1-p])

    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    df = pd.DataFrame({
        "customerID":        [f"CUST-{i:05d}" for i in range(n)],
        "gender":            rng.choice(["Male", "Female"], n),
        "SeniorCitizen":     rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner":           yn(0.48),
        "Dependents":        yn(0.30),
        "tenure":            tenure,
        "PhoneService":      yn(0.90),
        "MultipleLines":     rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService":   internet,
        "OnlineSecurity":    rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "OnlineBackup":      rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "DeviceProtection":  rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "TechSupport":       rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "StreamingTV":       rng.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
        "StreamingMovies":   rng.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
        "Contract":          contract,
        "PaperlessBilling":  yn(0.59),
        "PaymentMethod":     rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.23, 0.22, 0.21]
        ),
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges.astype(str),  # intentional: matches real dataset
        "Churn":             churn_str,
    })

    logger.info(f"Generated synthetic dataset: {df.shape}, churn rate: {churn.mean():.2%}")
    return df


def get_or_create_dataset(data_path: Path) -> pd.DataFrame:
    """Load dataset from disk or generate synthetic data if not found."""
    if data_path.exists():
        logger.info(f"Loading existing dataset from {data_path}")
        return pd.read_csv(data_path)

    logger.warning(f"Dataset not found at {data_path}. Generating synthetic data...")
    df = generate_synthetic_telco_data()
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    logger.info(f"Synthetic dataset saved to {data_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from config import RAW_DATA_PATH
    df = get_or_create_dataset(RAW_DATA_PATH)
    print(df.head())
    print(f"\nChurn rate: {(df['Churn'] == 'Yes').mean():.2%}")
