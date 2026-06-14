"""
main.py — FastAPI backend for the Churn Prediction System.

WHY FASTAPI:
- Automatic OpenAPI docs at /docs (huge time saver)
- Pydantic validation: wrong input types return clear errors automatically
- Async support: non-blocking I/O for production-scale traffic
- Type hints make the code self-documenting
- 3x faster than Flask for JSON serialization

AUTHENTICATION STRATEGY:
- /health          → public   (load balancers + Docker healthchecks need this)
- /predict         → service+ (admin, service roles)
- /batch_predict   → service+ (admin, service roles)
- /model/info      → readonly+ (all authenticated roles)
- /monitoring/stats→ admin only

WHY THIS SPLIT: Not all consumers need full access. A Grafana dashboard only
needs /monitoring/stats. An internal scoring job only needs /batch_predict.
Least-privilege principle — each key gets exactly what it needs.
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.models import APIKey
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import PREDICTIONS_LOG, API_HOST, API_PORT
from app.training.predict import (
    load_artifacts, predict_single, predict_batch, get_meta
)
from app.utils.monitoring import PredictionLogger, DriftMonitor
from app.api.auth import (
    get_api_key, require_predict_permission, require_admin_permission,
    APIKeyInfo
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─── Global state ─────────────────────────────────────────────────────────────
prediction_logger = PredictionLogger(PREDICTIONS_LOG)
drift_monitor     = DriftMonitor()


# ─── Lifespan (startup/shutdown) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    WHY LIFESPAN: Loading ML models at startup (not per-request) is critical.
    Loading a model takes 1–5 seconds. Per-request loading would make the API
    useless. Lifespan context manager handles startup/shutdown cleanly.
    """
    logger.info("🚀 Starting Churn Prediction API...")
    load_artifacts()
    logger.info("✓ Model loaded. API ready.")
    yield
    logger.info("👋 Shutting down API.")


# ─── App initialization ───────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-style ML API for predicting customer churn. "
        "Returns churn probability, risk level, and SHAP-based explanations.\n\n"
        "**Authentication:** Include header `X-API-Key: <your-key>` on all protected endpoints.\n\n"
        "**Roles:** `admin` → full access | `service` → predictions only | `readonly` → health + info"
    ),
    version="1.1.0",
    lifespan=lifespan,
)

# CORS middleware — allows the Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    gender:           str   = Field("Male", description="Male or Female")
    SeniorCitizen:    int   = Field(0, ge=0, le=1)
    Partner:          str   = Field("No")
    Dependents:       str   = Field("No")
    tenure:           int   = Field(12, ge=0, le=72, description="Months with company")
    PhoneService:     str   = Field("Yes")
    MultipleLines:    str   = Field("No")
    InternetService:  str   = Field("DSL", description="DSL, Fiber optic, or No")
    OnlineSecurity:   str   = Field("No")
    OnlineBackup:     str   = Field("No")
    DeviceProtection: str   = Field("No")
    TechSupport:      str   = Field("No")
    StreamingTV:      str   = Field("No")
    StreamingMovies:  str   = Field("No")
    Contract:         str   = Field("Month-to-month", description="Month-to-month, One year, Two year")
    PaperlessBilling: str   = Field("Yes")
    PaymentMethod:    str   = Field("Electronic check")
    MonthlyCharges:   float = Field(65.0, ge=0, le=200)
    TotalCharges:     float = Field(780.0, ge=0)

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v):
        valid = ["Month-to-month", "One year", "Two year"]
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    @field_validator("InternetService")
    @classmethod
    def validate_internet(cls, v):
        valid = ["DSL", "Fiber optic", "No"]
        if v not in valid:
            raise ValueError(f"InternetService must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "gender": "Male", "SeniorCitizen": 0,
                "Partner": "No", "Dependents": "No",
                "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
                "InternetService": "Fiber optic", "OnlineSecurity": "No",
                "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.5, "TotalCharges": 286.5
            }
        }
    }


class BatchInput(BaseModel):
    customers: list[CustomerInput] = Field(..., min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    churn_probability: float
    risk_level:        str
    prediction:        int
    explanation:       list[dict]


class BatchResponse(BaseModel):
    total:   int
    results: list[dict]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """
    Public endpoint — no auth required.

    WHY PUBLIC: Docker HEALTHCHECK, AWS ALB health probes, and uptime monitors
    all need to reach this without credentials. Gating it breaks infrastructure.
    """
    meta = get_meta()
    drift_report = drift_monitor.check_drift()
    return {
        "status":        "healthy",
        "model":         meta.get("model_name", "Unknown"),
        "test_roc_auc":  meta.get("test_metrics", {}).get("roc_auc", "N/A"),
        "drift_monitor": drift_report,
        "stats":         drift_monitor.get_stats(),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(
    customer: CustomerInput,
    key_info: APIKeyInfo = Depends(require_predict_permission),   # ← auth injected here
):
    """
    Predict churn probability for a single customer.

    **Requires:** `service` or `admin` API key.

    Returns churn probability, risk level (Low/Medium/High),
    binary prediction, and top SHAP feature explanations.
    """
    try:
        customer_dict = customer.model_dump()
        customer_dict["TotalCharges"] = str(customer_dict["TotalCharges"])

        result = predict_single(customer_dict)

        prediction_logger.log(customer_dict, result)
        drift_monitor.update(result["churn_probability"])

        logger.info(
            f"[{key_info.role}] Prediction: prob={result['churn_probability']:.3f} "
            f"risk={result['risk_level']}"
        )
        return PredictionResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchResponse, tags=["Prediction"])
async def batch_predict_churn(
    batch: BatchInput,
    key_info: APIKeyInfo = Depends(require_predict_permission),   # ← auth injected here
):
    """
    Predict churn for multiple customers in one request.

    **Requires:** `service` or `admin` API key.

    More efficient than N individual /predict calls — use for nightly scoring jobs.
    """
    try:
        customer_dicts = []
        for c in batch.customers:
            d = c.model_dump()
            d["TotalCharges"] = str(d["TotalCharges"])
            customer_dicts.append(d)

        results = predict_batch(customer_dicts)

        for r in results:
            drift_monitor.update(r["churn_probability"])

        prediction_logger.log(
            {"batch_size": len(customer_dicts)},
            {"batch_results_count": len(results)}
        )

        logger.info(f"[{key_info.role}] Batch prediction: {len(results)} customers")
        return BatchResponse(total=len(results), results=results)

    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info(
    key_info: APIKeyInfo = Depends(get_api_key),   # any valid key
):
    """
    Return model metadata: training metrics, feature count, CV results.

    **Requires:** any valid API key (readonly, service, or admin).
    """
    meta = get_meta()
    logger.info(f"[{key_info.role}] Accessed model info")
    return meta


@app.get("/monitoring/stats", tags=["Monitoring"])
async def monitoring_stats(
    key_info: APIKeyInfo = Depends(require_admin_permission),   # admin only
):
    """
    Return prediction distribution stats and drift detection report.

    **Requires:** `admin` API key only.

    WHY ADMIN ONLY: Monitoring data reveals prediction volume and distribution,
    which could expose business intelligence to unauthorized parties.
    """
    logger.info(f"[{key_info.role}] Accessed monitoring stats")
    return {
        "prediction_stats": drift_monitor.get_stats(),
        "drift_check":      drift_monitor.check_drift(),
    }


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=API_HOST, port=API_PORT, reload=False)
