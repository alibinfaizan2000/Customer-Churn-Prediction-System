"""
streamlit_app.py — Professional Streamlit UI for the Churn Prediction System.

WHY STREAMLIT:
- Pure Python — no HTML/CSS/JS needed
- Ideal for ML demos and internal tools
- Rapid prototyping: this UI took ~100 lines
- Built-in widgets: sliders, dropdowns, buttons
- Easy deployment: just `streamlit run`

AUTH INTEGRATION:
The UI reads UI_API_KEY from config and passes it as the X-API-Key header
on every API call. Users of the UI never see or handle the key directly.
This is the standard pattern for internal tools — the UI acts as a
trusted service client, not an anonymous browser.
"""

import sys
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import API_BASE_URL, RISK_LOW_MAX, RISK_MED_MAX, UI_API_KEY

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnGuard — Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 700; color: #1f77b4; margin-bottom: 0; }
    .subtitle { color: #666; font-size: 1rem; margin-bottom: 2rem; }
    .risk-high   { color: #dc3545; font-weight: bold; font-size: 1.5rem; }
    .risk-medium { color: #fd7e14; font-weight: bold; font-size: 1.5rem; }
    .risk-low    { color: #28a745; font-weight: bold; font-size: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─── Auth headers ─────────────────────────────────────────────────────────────
# WHY: All API calls from the UI include this header automatically.
# The key comes from config (which reads from environment variable UI_API_KEY).
# Users of the UI never see or manage API keys.
def _auth_headers() -> dict:
    return {"X-API-Key": UI_API_KEY}


# ─── Helper functions ─────────────────────────────────────────────────────────

def check_api_health() -> tuple[bool, str]:
    """Ping the FastAPI backend. Returns (is_healthy, status_message)."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return r.status_code == 200, "Connected"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect"
    except Exception as e:
        return False, str(e)


def call_predict(payload: dict) -> dict | None:
    """Call /predict with auth header."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/predict",
            json=payload,
            headers=_auth_headers(),
            timeout=10
        )
        if r.status_code == 200:
            return r.json()
        elif r.status_code == 401:
            st.error("❌ API key missing. Set UI_API_KEY in your .env file.")
        elif r.status_code == 403:
            st.error("❌ API key invalid or insufficient permissions.")
        else:
            st.error(f"API Error {r.status_code}: {r.text}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to FastAPI backend. Is it running?")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def get_model_info() -> dict:
    """Fetch model metadata (requires any valid key)."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/model/info",
            headers=_auth_headers(),
            timeout=3
        )
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def get_monitoring_stats() -> dict:
    """Fetch monitoring stats (requires admin key)."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/monitoring/stats",
            headers=_auth_headers(),
            timeout=3
        )
        if r.status_code == 403:
            return {"error": "Admin key required for monitoring stats"}
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def render_probability_gauge(prob: float) -> go.Figure:
    color = "#dc3545" if prob > RISK_MED_MAX else ("#fd7e14" if prob > RISK_LOW_MAX else "#28a745")
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob * 100,
        title={"text": "Churn Probability (%)", "font": {"size": 16}},
        delta={"reference": 50, "increasing": {"color": "#dc3545"}},
        gauge={
            "axis":  {"range": [0, 100], "tickwidth": 1},
            "bar":   {"color": color},
            "steps": [
                {"range": [0, 35],  "color": "#d4edda"},
                {"range": [35, 65], "color": "#fff3cd"},
                {"range": [65, 100],"color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50},
        },
        number={"suffix": "%", "font": {"size": 28}}
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0))
    return fig


def render_shap_chart(explanation: list[dict]) -> go.Figure:
    if not explanation:
        return None
    features  = [e["display_name"] for e in explanation]
    shap_vals = [e["shap_value"] for e in explanation]
    colors    = ["#dc3545" if v > 0 else "#28a745" for v in shap_vals]
    fig = go.Figure(go.Bar(
        x=shap_vals, y=features, orientation="h",
        marker_color=colors,
        hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.3f}<extra></extra>"
    ))
    fig.update_layout(
        title="Feature Contributions (SHAP values)",
        xaxis_title="SHAP Value (→ increases churn | ← decreases churn)",
        height=300, margin=dict(l=10, r=10, t=40, b=40),
        xaxis=dict(zeroline=True, zerolinecolor="black", zerolinewidth=1.5),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    return fig


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("## 📉 ChurnGuard")
        st.markdown("*Customer Churn Prediction System*")
        st.divider()

        healthy, msg = check_api_health()
        if healthy:
            st.success(f"✅ API Connected")
        else:
            st.error(f"❌ API Offline: {msg}")
            st.info("Start the backend:\n```\npython app/api/main.py\n```")

        st.divider()

        '''page = st.radio(
            "Navigation",
            ["🔮 Single Prediction", "📊 Model Performance", "📈 Monitoring"],
            label_visibility="collapsed"
        )'''

        page = st.radio(
            "Navigation",
            ["🔮 Single Prediction", "📊 Model Performance"],
            label_visibility="collapsed"
        )

        st.divider()
        st.caption("Built with FastAPI + XGBoost + SHAP + Streamlit")
        return page


# ─── Pages ────────────────────────────────────────────────────────────────────

def page_predict():
    st.markdown('<p class="main-title">🔮 Predict Customer Churn</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Fill in customer details to get a churn prediction with AI explanation</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        st.subheader("👤 Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)

        with col2:
            st.markdown("**Phone & Internet**")
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**Additional Services**")
            device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        st.subheader("💳 Billing")
        col4, col5, col6 = st.columns(3)

        with col4:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
        with col5:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
        with col6:
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0,
                                            float(monthly_charges * tenure), step=1.0)

        submitted = st.form_submit_button("🔮 Predict Churn", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "gender": gender, "SeniorCitizen": senior, "Partner": partner,
            "Dependents": dependents, "tenure": tenure, "PhoneService": phone_service,
            "MultipleLines": multiple_lines, "InternetService": internet,
            "OnlineSecurity": online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_prot, "TechSupport": tech_support,
            "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        }

        with st.spinner("Running prediction..."):
            result = call_predict(payload)

        if result:
            st.divider()
            st.subheader("📊 Prediction Results")

            col_gauge, col_risk, col_rec = st.columns([2, 1, 2])

            with col_gauge:
                st.plotly_chart(render_probability_gauge(result["churn_probability"]), use_container_width=True)

            with col_risk:
                risk = result["risk_level"]
                st.markdown("**Risk Level**")
                st.markdown(f'<p class="risk-{risk.lower()}">{risk}</p>', unsafe_allow_html=True)
                st.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
                verdict = "⚠️ Likely to Churn" if result["prediction"] == 1 else "✅ Likely to Stay"
                st.markdown(f"**Verdict:** {verdict}")

            with col_rec:
                st.markdown("**💡 Recommended Actions**")
                prob = result["churn_probability"]
                if prob > RISK_MED_MAX:
                    st.error("🚨 Immediate retention action needed")
                    st.markdown("- Offer retention discount\n- Personal account manager call\n- Upgrade to long-term contract")
                elif prob > RISK_LOW_MAX:
                    st.warning("📞 Proactive outreach recommended")
                    st.markdown("- Send satisfaction survey\n- Offer loyalty rewards\n- Review service quality")
                else:
                    st.success("💚 No immediate action needed")
                    st.markdown("- Continue standard engagement\n- Upsell additional services")

            if result.get("explanation"):
                st.subheader("🧠 AI Explanation (SHAP)")
                shap_fig = render_shap_chart(result["explanation"])
                if shap_fig:
                    st.plotly_chart(shap_fig, use_container_width=True)
                st.markdown("**Top Contributing Factors:**")
                for item in result["explanation"]:
                    icon = "🔴" if item["shap_value"] > 0 else "🟢"
                    direction = "↑ increases" if item["shap_value"] > 0 else "↓ decreases"
                    st.markdown(
                        f'{icon} **{item["display_name"]}** — '
                        f'{direction} churn risk (SHAP: {item["shap_value"]:+.3f})'
                    )


def page_model_performance():
    st.markdown('<p class="main-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Training metrics and model comparison</p>', unsafe_allow_html=True)

    meta = get_model_info()
    if not meta:
        st.warning("Could not fetch model info. Is the API running with a valid key?")
        return

    test_metrics = meta.get("test_metrics", {})
    if test_metrics:
        st.subheader("🏆 Best Model: " + meta.get("model_name", "Unknown"))
        col1, col2, col3, col4, col5 = st.columns(5)
        for col, (name, key) in zip(
            [col1, col2, col3, col4, col5],
            [("Accuracy","accuracy"),("Precision","precision"),
             ("Recall","recall"),("F1-Score","f1"),("ROC-AUC","roc_auc")]
        ):
            with col:
                st.metric(name, f"{test_metrics.get(key, 0):.3f}")

        cv_data = meta.get("cv_roc_auc", {})
        if cv_data:
            st.subheader("📈 Cross-Validation Comparison (5-Fold ROC-AUC)")
            cv_df = pd.DataFrame([
                {"Model": k, "CV ROC-AUC (Mean)": v["mean"], "Std": v["std"]}
                for k, v in cv_data.items()
            ])
            fig = px.bar(cv_df, x="Model", y="CV ROC-AUC (Mean)", error_y="Std",
                         color="CV ROC-AUC (Mean)", color_continuous_scale="Blues",
                         title="Model Comparison — Cross-Validated ROC-AUC")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("📚 Understanding the Metrics"):
        st.markdown("""
| Metric | What It Measures | Why It Matters for Churn |
|--------|-----------------|--------------------------|
| **Accuracy** | % of correct predictions | Misleading on imbalanced data |
| **Precision** | Of predicted churners, how many actually churned | Low = wasted retention calls |
| **Recall** | Of actual churners, how many did we catch | **Most critical** — missing churners is costly |
| **F1-Score** | Harmonic mean of precision & recall | Best single metric for imbalanced classes |
| **ROC-AUC** | Ranking ability across all thresholds | Gold standard, threshold-independent |
        """)


def page_monitoring():
    st.markdown('<p class="main-title">📈 Monitoring Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Track model predictions and detect drift</p>', unsafe_allow_html=True)

    stats = get_monitoring_stats()

    if "error" in stats:
        st.warning(f"⚠️ {stats['error']} — set ADMIN_API_KEY as UI_API_KEY to access this page.")
        return

    if not stats:
        st.info("No monitoring data yet. Make some predictions first!")
        return

    pred_stats = stats.get("prediction_stats", {})
    drift = stats.get("drift_check", {})

    if drift.get("drift_flag"):
        st.error(f"🚨 Drift Detected! Mean shift: {drift.get('mean_shift', 0):.3f}")
    else:
        st.success("✅ No drift detected — model predictions are stable")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Predictions Tracked", pred_stats.get("count", 0))
    with col2: st.metric("Mean Churn Prob", f"{pred_stats.get('mean', 0):.3f}")
    with col3: st.metric("Baseline Mean", f"{drift.get('baseline_mean', 0.27):.3f}")
    with col4: st.metric("Mean Shift", f"{drift.get('mean_shift', 0):.3f}")

    with st.expander("📚 What is Drift?"):
        st.markdown("""
**Data drift** = input feature distributions change over time.
**Concept drift** = the relationship between features and churn changes.

Monitoring lets you know when to retrain the model.
        """)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    page = render_sidebar()
    if "Prediction" in page:
        page_predict()
    elif "Performance" in page:
        page_model_performance()
    elif "Monitoring" in page:
        page_monitoring()


if __name__ == "__main__":
    main()
