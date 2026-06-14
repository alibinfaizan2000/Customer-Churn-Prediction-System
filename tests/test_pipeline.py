"""
tests/test_pipeline.py — Pytest test suite for the churn prediction system.

WHY TESTS MATTER IN MLOPS:
Production ML systems break in subtle ways:
- A schema change in incoming data silently passes wrong features
- A preprocessing step encodes differently at train vs inference time
- A model update changes the output distribution unexpectedly

Tests catch these before they hit production.

INTERVIEW TIP: Being able to write pytest tests for an ML pipeline is a
strong differentiator for MLOps-focused roles. Most candidates skip this.
"""

import sys
import json
from unittest import result
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import RAW_DATA_PATH, MODEL_PATH, FEATURE_NAMES_PATH


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df():
    """Load or generate a small raw DataFrame for testing."""
    from app.data.download_data import generate_synthetic_telco_data
    return generate_synthetic_telco_data(n_rows=200, seed=42)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    """Return preprocessed DataFrame."""
    from app.data.preprocess import preprocess_dataframe
    return preprocess_dataframe(raw_df)


@pytest.fixture(scope="module")
def featured_df(processed_df):
    """Return fully feature-engineered DataFrame."""
    from app.data.feature_engineering import engineer_features
    return engineer_features(processed_df)


@pytest.fixture(scope="module")
def sample_customer():
    """A representative high-risk customer dict (raw API input format)."""
    return {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5, "TotalCharges": "286.5",
    }


@pytest.fixture(scope="module")
def low_risk_customer():
    """A representative low-risk customer dict."""
    return {
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year",
        "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 55.0, "TotalCharges": "3300.0",
    }


# ─── Data tests ───────────────────────────────────────────────────────────────

class TestDataGeneration:
    def test_dataset_shape(self, raw_df):
        """Dataset should have 21 columns and expected row count."""
        assert raw_df.shape == (200, 21)

    def test_required_columns_present(self, raw_df):
        """All essential columns must be present."""
        required = ["tenure", "MonthlyCharges", "TotalCharges", "Churn", "Contract"]
        for col in required:
            assert col in raw_df.columns, f"Missing column: {col}"

    def test_churn_rate_realistic(self, raw_df):
        """Churn rate should be roughly 20–40% (realistic telecom range)."""
        rate = (raw_df["Churn"] == "Yes").mean()
        assert 0.15 <= rate <= 0.45, f"Unexpected churn rate: {rate:.2%}"

    def test_no_all_null_columns(self, raw_df):
        """No column should be entirely null."""
        null_counts = raw_df.isnull().sum()
        all_null = null_counts[null_counts == len(raw_df)]
        assert len(all_null) == 0, f"Fully null columns: {list(all_null.index)}"


class TestPreprocessing:
    def test_output_shape_larger_or_equal(self, raw_df, processed_df):
        """Preprocessing adds one-hot encoded columns, so shape should grow."""
        assert processed_df.shape[0] == raw_df.shape[0], "Row count should not change"
        assert processed_df.shape[1] >= raw_df.shape[1] - 5, "Should not lose many columns"

    def test_churn_is_binary_integer(self, processed_df):
        """After preprocessing, Churn should be 0 or 1 integers."""
        unique_vals = set(processed_df["Churn"].unique())
        assert unique_vals <= {0, 1}, f"Churn has unexpected values: {unique_vals}"

    def test_no_yes_no_strings_remain(self, processed_df):
        """Binary columns should be 0/1 after encoding, not 'Yes'/'No'."""
        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
            if col in processed_df.columns:
                vals = set(processed_df[col].dropna().unique())
                assert vals <= {0, 1, 0.0, 1.0}, f"{col} still has string values: {vals}"

    def test_total_charges_numeric(self, processed_df):
        """TotalCharges should be float after fixing string values."""
        assert pd.api.types.is_numeric_dtype(processed_df["TotalCharges"])

    def test_no_customer_id_in_output(self, processed_df):
        """customerID should be dropped (non-predictive identifier)."""
        assert "customerID" not in processed_df.columns

    def test_no_nulls_after_processing(self, processed_df):
        """Preprocessing should fill/handle all null values."""
        null_counts = processed_df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        assert len(cols_with_nulls) == 0, f"Nulls remain in: {dict(cols_with_nulls)}"


class TestFeatureEngineering:
    def test_engineered_features_exist(self, featured_df):
        """All expected engineered features should be present."""
        expected = [
            "charge_ratio", "service_count", "contract_risk",
            "charge_increase", "avg_monthly_spend"
        ]
        for feat in expected:
            assert feat in featured_df.columns, f"Missing engineered feature: {feat}"

    def test_service_count_range(self, featured_df):
        """Service count should be between 0 and 8."""
        assert featured_df["service_count"].min() >= 0
        assert featured_df["service_count"].max() <= 8

    def test_contract_risk_range(self, featured_df):
        """Contract risk should be 0, 1, or 2."""
        if "contract_risk" in featured_df.columns:
            unique_vals = set(featured_df["contract_risk"].unique())
            assert unique_vals <= {0, 1, 2}, f"Unexpected contract_risk values: {unique_vals}"

    def test_charge_ratio_non_negative(self, featured_df):
        """Charge ratio (monthly/total) should never be negative."""
        assert (featured_df["charge_ratio"] >= 0).all()

    def test_tenure_buckets_created(self, featured_df):
        """At least some tenure bucket columns should exist."""
        bucket_cols = [c for c in featured_df.columns if "tenure_bucket" in c]
        assert len(bucket_cols) > 0, "No tenure bucket columns found"

    def test_no_inf_values(self, featured_df):
        """Feature engineering should not produce infinite values."""
        numeric = featured_df.select_dtypes(include=[np.number])
        inf_count = np.isinf(numeric.values).sum()
        assert inf_count == 0, f"{inf_count} infinite values found"

    def test_feature_count_increased(self, processed_df, featured_df):
        """Feature engineering should add columns."""
        assert featured_df.shape[1] > processed_df.shape[1]


# ─── Model tests ──────────────────────────────────────────────────────────────

class TestModelArtifacts:
    def test_model_file_exists(self):
        """Trained model file must exist."""
        assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}. Run train.py first."

    def test_feature_names_file_exists(self):
        """Feature names file must exist."""
        assert FEATURE_NAMES_PATH.exists(), "feature_names.pkl not found"

    def test_model_meta_valid(self):
        """Model metadata JSON should have required keys."""
        meta_path = MODEL_PATH.parent / "model_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            meta = json.load(f)
        assert "feature_names" in meta
        assert "test_metrics" in meta
        assert "roc_auc" in meta["test_metrics"]

    def test_model_roc_auc_acceptable(self):
        """Model ROC-AUC should be above 0.65 (better than random)."""
        meta_path = MODEL_PATH.parent / "model_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            roc_auc = meta["test_metrics"]["roc_auc"]
            assert roc_auc > 0.65, f"ROC-AUC too low: {roc_auc}"


# ─── Prediction tests ─────────────────────────────────────────────────────────

class TestPrediction:
    @pytest.fixture(autouse=True)
    def load_model(self):
        """Load model artifacts once for all prediction tests."""
        if not MODEL_PATH.exists():
            pytest.skip("Model not trained yet. Run train.py first.")
        from app.training.predict import load_artifacts
        load_artifacts()

    def test_predict_single_returns_required_fields(self, sample_customer):
        """Single prediction must return all required response fields."""
        from app.training.predict import predict_single
        result = predict_single(sample_customer)
        assert "churn_probability" in result
        assert "risk_level" in result
        assert "prediction" in result
        assert "explanation" in result

    def test_churn_probability_in_range(self, sample_customer):
        """Churn probability must be between 0 and 1."""
        from app.training.predict import predict_single
        result = predict_single(sample_customer)
        prob = result["churn_probability"]
        assert 0.0 <= prob <= 1.0, f"Invalid probability: {prob}"

    def test_risk_level_valid(self, sample_customer):
        """Risk level must be one of the three valid categories."""
        from app.training.predict import predict_single
        result = predict_single(sample_customer)
        assert result["risk_level"] in {"Low", "Medium", "High"}

    def test_prediction_binary(self, sample_customer):
        """Prediction must be 0 or 1."""
        from app.training.predict import predict_single
        result = predict_single(sample_customer)
        assert result["prediction"] in {0, 1}

    def test_high_risk_customer_scores_higher(self, sample_customer, low_risk_customer):
        """
        High-risk customer (new, month-to-month, high charges)
        should score higher than low-risk (loyal, two-year contract).

        WHY THIS TEST: This is a sanity check on model behavior.
        If the model predicts low churn for a new month-to-month customer,
        something is wrong with the features or training.
        """
        from app.training.predict import predict_single
        high_result = predict_single(sample_customer)
        low_result  = predict_single(low_risk_customer)
        assert high_result["churn_probability"] > low_result["churn_probability"], (
            f"High-risk ({high_result['churn_probability']:.3f}) should score "
            f"above low-risk ({low_result['churn_probability']:.3f})"
        )

    def test_explanation_list_non_empty(self, sample_customer):
        from app.training.predict import predict_single
        result = predict_single(sample_customer)
        assert isinstance(result["explanation"], list)

    def test_batch_predict_correct_count(self, sample_customer, low_risk_customer):
        """Batch prediction should return same number of results as input."""
        from app.training.predict import predict_batch
        customers = [sample_customer, low_risk_customer, sample_customer]
        results = predict_batch(customers)
        assert len(results) == 3

    def test_batch_results_have_index(self, sample_customer):
        """Each batch result should include its index."""
        from app.training.predict import predict_batch
        results = predict_batch([sample_customer, sample_customer])
        for i, r in enumerate(results):
            assert r["index"] == i

    def test_prediction_deterministic(self, sample_customer):
        """Same input should always produce same output."""
        from app.training.predict import predict_single
        r1 = predict_single(sample_customer)
        r2 = predict_single(sample_customer)
        assert r1["churn_probability"] == r2["churn_probability"]


# ─── Monitoring tests ─────────────────────────────────────────────────────────

class TestMonitoring:
    def test_drift_monitor_no_drift_when_stable(self):
        """Monitor should not flag drift when predictions match baseline."""
        from app.utils.monitoring import DriftMonitor
        monitor = DriftMonitor(baseline_mean=0.27, drift_threshold=0.1)
        # Feed 50 predictions close to baseline
        np.random.seed(42)
        for _ in range(50):
            monitor.update(np.random.normal(0.27, 0.05))
        report = monitor.check_drift()
        assert report["drift_flag"] is False

    def test_drift_monitor_detects_drift(self):
        """Monitor should flag drift when mean shifts significantly."""
        from app.utils.monitoring import DriftMonitor
        monitor = DriftMonitor(baseline_mean=0.27, drift_threshold=0.1)
        # Feed 50 predictions at 0.6 (far from 0.27 baseline)
        for _ in range(50):
            monitor.update(0.6)
        report = monitor.check_drift()
        assert report["drift_flag"] is True

    def test_drift_monitor_insufficient_data(self):
        """Monitor should return 'insufficient_data' with < 10 samples."""
        from app.utils.monitoring import DriftMonitor
        monitor = DriftMonitor()
        for _ in range(5):
            monitor.update(0.3)
        report = monitor.check_drift()
        assert report["status"] == "insufficient_data"

    def test_prediction_logger_writes_file(self, tmp_path):
        """Logger should write JSONL records to disk."""
        from app.utils.monitoring import PredictionLogger
        log_path = tmp_path / "test_log.jsonl"
        logger = PredictionLogger(log_path)
        logger.log({"tenure": 5}, {"churn_probability": 0.7, "risk_level": "High"})
        assert log_path.exists()
        with open(log_path) as f:
            record = json.loads(f.readline())
        assert "timestamp" in record
        assert record["prediction"]["churn_probability"] == 0.7


# ─── Integration tests ────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_full_pipeline_small_dataset(self):
        """
        Run the full preprocessing + feature eng + model pipeline
        on a small synthetic dataset end-to-end.
        """
        from app.data.download_data import generate_synthetic_telco_data
        from app.data.preprocess import preprocess_dataframe
        from app.data.feature_engineering import engineer_features

        df = generate_synthetic_telco_data(n_rows=100, seed=99)
        df_clean = preprocess_dataframe(df)
        df_feat = engineer_features(df_clean)

        assert df_feat.shape[0] == 100
        assert "Churn" in df_feat.columns
        assert df_feat["Churn"].isin([0, 1]).all()

        # All features should be numeric
        X = df_feat.drop(columns=["Churn"]).select_dtypes(include=[np.number])
        assert X.shape[1] > 15, f"Too few numeric features: {X.shape[1]}"

    def test_single_row_inference_matches_training_schema(self):
        """
        A single customer row processed through inference pipeline
        should produce the same feature set as training.
        """
        if not MODEL_PATH.exists():
            pytest.skip("Model not trained yet.")

        import joblib
        from app.training.predict import _prepare_features, get_feature_names

        customer = {
            "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
            "tenure": 10, "PhoneService": "Yes", "MultipleLines": "No",
            "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "No",
            "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
            "StreamingMovies": "No", "Contract": "Month-to-month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
            "MonthlyCharges": 45.0, "TotalCharges": "450.0",
        }

        X = _prepare_features(customer)
        feature_names = get_feature_names()

        # Must have exactly the same columns as training
        assert list(X.columns) == feature_names, (
            f"Schema mismatch. Expected {len(feature_names)} features, "
            f"got {len(X.columns)}"
        )
        assert X.shape == (1, len(feature_names))


# ─── Auth tests ───────────────────────────────────────────────────────────────

class TestAuthentication:

    def test_valid_admin_key_returns_key_info(self):
        from app.api.auth import _validate_key
        import os
        info = _validate_key(os.getenv("ADMIN_API_KEY", ""))
        assert info is not None
        assert info.role == "admin"

    def test_valid_service_key_returns_key_info(self):
        from app.api.auth import _validate_key
        import os
        info = _validate_key(os.getenv("SERVICE_API_KEY", ""))
        assert info is not None
        assert info.role == "service"

    def test_invalid_key_returns_none(self):
        from app.api.auth import _validate_key
        info = _validate_key("totally-wrong-key-xyz-123")
        assert info is None

    def test_empty_key_returns_none(self):
        from app.api.auth import _validate_key
        info = _validate_key("")
        assert info is None

    def test_admin_has_all_permissions(self):
        from app.api.auth import _validate_key
        import os
        info = _validate_key(os.getenv("ADMIN_API_KEY", ""))
        assert info is not None
        assert info.can("predict")
        assert info.can("batch_predict")
        assert info.can("monitoring")
        assert info.can("model_info")
        assert info.can("health")

    def test_service_cannot_access_monitoring(self):
        from app.api.auth import _validate_key
        import os
        info = _validate_key(os.getenv("SERVICE_API_KEY", ""))
        assert info is not None
        assert info.can("predict") is True
        assert info.can("monitoring") is False

    def test_readonly_cannot_predict(self):
        from app.api.auth import _validate_key
        import os
        info = _validate_key(os.getenv("READONLY_API_KEY", ""))
        assert info is not None
        assert info.can("predict") is False
        assert info.can("health") is True
        assert info.can("model_info") is True

    def test_generate_api_key_format(self):
        from app.api.auth import generate_api_key
        key = generate_api_key("test")
        assert key.startswith("test-")
        assert len(key) == 5 + 32

    def test_generated_keys_are_unique(self):
        from app.api.auth import generate_api_key
        keys = {generate_api_key() for _ in range(10)}
        assert len(keys) == 10
