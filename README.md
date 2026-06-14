# 📉 Customer Churn Prediction System
### Production-Style ML Pipeline | Interview-Ready Portfolio Project

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![SHAP](https://img.shields.io/badge/SHAP-0.42-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)

---

## 🎯 Project Overview

A complete, production-inspired ML system that predicts whether a telecom customer will churn (cancel their service). This project demonstrates an end-to-end ML pipeline covering:

- **Data Preprocessing** → cleaning, encoding, missing value handling
- **Feature Engineering** → domain-driven features with business rationale
- **Model Training** → 3 models with cross-validation + hyperparameter tuning
- **Explainability** → SHAP values for every prediction
- **Deployment** → FastAPI REST API + Streamlit UI
- **MLOps** → MLflow tracking, prediction logging, drift detection
- **Containerization** → Docker + Docker Compose

---

## 🗂️ Project Structure

```
churn_prediction/
├── config.py                    # Central config (paths, hyperparams, thresholds)
├── requirements.txt
│
├── app/
│   ├── data/
│   │   ├── download_data.py     # Dataset loader / synthetic data generator
│   │   ├── preprocess.py        # Cleaning, encoding, missing value handling
│   │   └── feature_engineering.py  # Domain-driven feature creation
│   │
│   ├── training/
│   │   ├── train.py             # Full training pipeline (run this first)
│   │   └── predict.py           # Inference module (used by API + UI)
│   │
│   ├── api/
│   │   └── main.py              # FastAPI backend
│   │
│   ├── ui/
│   │   └── streamlit_app.py     # Streamlit frontend
│   │
│   ├── utils/
│   │   ├── explainability.py    # SHAP integration
│   │   ├── monitoring.py        # Drift detection + prediction logging
│   │   └── mlflow_tracking.py   # MLflow experiment tracking
│   │
│   └── models/                  # Saved artifacts (gitignored)
│       ├── best_model.pkl
│       ├── feature_names.pkl
│       └── model_meta.json
│
├── notebooks/
│   └── 01_EDA_and_modeling.ipynb
│
└── docker/
    ├── Dockerfile.api
    ├── Dockerfile.ui
    └── docker-compose.yml
```

---

## 🚀 Quick Start

### Option A: Local (recommended for development)

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd churn_prediction
pip install -r requirements.txt

# 2. Train the model (generates data + trains + saves artifacts)
python app/training/train.py

# With hyperparameter tuning (slower, better results):
python app/training/train.py --tune

# 3. Start the FastAPI backend
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# 4. In a new terminal, start the Streamlit UI
streamlit run app/ui/streamlit_app.py
```

Then visit:
- **Streamlit UI**: http://localhost:8501
- **API Docs** (Swagger): http://localhost:8000/docs
- **MLflow UI**: `mlflow ui` → http://localhost:5000

---

### Option B: Docker (one-command deployment)

```bash
cd docker
docker-compose up --build
```

This will:
1. Build both images
2. Train the model in a `train` container
3. Start the API container (waits for training to complete)
4. Start the UI container (waits for API to be healthy)

---

## 🔌 API Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 3,
    "Contract": "Month-to-month",
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5,
    "InternetService": "Fiber optic",
    "TechSupport": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7823,
  "risk_level": "High",
  "prediction": 1,
  "explanation": [
    {
      "feature": "Contract_Two year",
      "display_name": "Long-term contract",
      "shap_value": -0.452,
      "direction": "decreases churn",
      "impact": 0.452
    },
    ...
  ]
}
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{"customers": [<customer1>, <customer2>]}'
```

### System Health
```bash
curl http://localhost:8000/health
```

---

## 🧠 Model Architecture

### Why These Three Models?

| Model | Why | Weakness |
|-------|-----|----------|
| **Logistic Regression** | Fast baseline, interpretable coefficients, good for linear patterns | Fails on non-linear interactions |
| **Random Forest** | Handles non-linearity, robust, no scaling needed | Slower predictions, less explainable |
| **XGBoost** ⭐ | Best on tabular data, built-in regularization, handles imbalance | More hyperparameters to tune |

**XGBoost wins because:**
- Sequential boosting: each tree corrects errors of the previous
- Built-in L1/L2 regularization prevents overfitting
- `scale_pos_weight` handles the ~73/27 class imbalance
- Empirically the top performer on tabular classification

### Evaluation Strategy

```
Data → Train/Test Split (80/20, stratified)
     → 5-Fold Cross Validation (on training set)
     → Best model selected by ROC-AUC
     → Evaluated on held-out test set
```

**Why ROC-AUC over Accuracy?**
With ~27% churn rate, a model that predicts "never churn" gets 73% accuracy but 0% recall.
ROC-AUC measures ranking ability across all thresholds — it's imbalance-immune.

---

## 🔬 Feature Engineering

| Feature | Business Logic |
|---------|----------------|
| `tenure_bucket` | New customers churn 4x more — explicit encoding of this effect |
| `charge_ratio` | High monthly vs total ratio = new + expensive = high risk |
| `service_count` | More services = more embedded = less likely to leave |
| `contract_risk` | Month-to-month (2) > One year (1) > Two year (0) |
| `charge_increase` | Current > historical spend = price shock = churn trigger |
| `paperless_no_support` | Self-service + no help = price-shopping customer profile |

---

## 📊 SHAP Explainability

**Why SHAP?**
- Grounded in game theory (Shapley values)
- Mathematically fair attribution of credit to each feature
- Additive: feature contributions sum to the prediction
- Required for GDPR "right to explanation" in EU deployments

**For each prediction, SHAP answers:**
- *Which features drove this churn prediction?*
- *Is "Month-to-month contract" pushing churn up or down?*
- *By how much?*

---

## 📈 MLOps Components

| Component | Implementation | Production Equivalent |
|-----------|---------------|----------------------|
| Experiment tracking | MLflow | Weights & Biases, Comet |
| Model serving | FastAPI + Uvicorn | BentoML, Seldon, TorchServe |
| Prediction logging | JSONL file | BigQuery, Snowflake |
| Drift detection | Mean/std shift | Evidently AI, WhyLabs |
| Containerization | Docker Compose | Kubernetes + Helm |

---

## 🎤 Interview Talking Points

1. **"Why XGBoost over Random Forest?"** → Gradient boosting is sequential correction; RF is parallel. XGB's regularization and scale_pos_weight for imbalance gives edge on tabular data.

2. **"How do you handle class imbalance?"** → `class_weight='balanced'` in LR/RF; `scale_pos_weight` in XGBoost; evaluate with F1 and ROC-AUC not accuracy.

3. **"Why SHAP over LIME?"** → SHAP is globally consistent; LIME approximates locally and can be unstable. SHAP is also exact for tree models via TreeExplainer.

4. **"What is training-serving skew?"** → Model trained on transformed data; inference gets raw data. Fixed by using the same preprocessing functions for both paths.

5. **"Why FastAPI over Flask?"** → Async I/O, automatic Pydantic validation, automatic OpenAPI docs, 3x faster JSON serialization.

6. **"What would you add to production?"** → Feature store (Feast), proper model registry (MLflow Registry), CI/CD (GitHub Actions), monitoring alerts (PagerDuty), A/B testing.

---

## 📦 Tech Stack

```
ML:          scikit-learn, XGBoost, SHAP
API:         FastAPI, Uvicorn, Pydantic
UI:          Streamlit, Plotly
MLOps:       MLflow, joblib
Infra:       Docker, Docker Compose
Data:        Pandas, NumPy
```
