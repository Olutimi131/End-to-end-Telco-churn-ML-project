"""Interactive Streamlit Frontend for Telco Customer Churn Intelligence.

Provides single-customer prediction with prescriptive retention actions,
batch scoring for uploaded CSV datasets, and interactive model performance metrics.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

# ==============================================================================
# CONFIGURATION & STYLING
# ==============================================================================

st.set_page_config(
    page_title="Telco Churn Intelligence Hub",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .badge-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
    }
    .recommendation-card {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 14px 18px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .driver-card {
        background-color: #FFF7ED;
        border-left: 4px solid #EA580C;
        padding: 12px 16px;
        border-radius: 6px;
        margin-top: 6px;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# SIDEBAR CONFIGURATION
# ==============================================================================

st.sidebar.image("https://img.icons8.com/clouds/200/telecommunication.png", width=120)
st.sidebar.title("Configuration")

api_url = st.sidebar.text_input("FastAPI Backend URL", value="http://localhost:8000")
use_direct_model_fallback = st.sidebar.checkbox("Direct Model Fallback (if API offline)", value=True)

# Preset customer profiles for quick evaluation
preset_choice = st.sidebar.selectbox(
    "Load Customer Preset",
    ["Custom Input", "High-Risk Attrition Candidate", "Loyal Long-term Customer", "Standard New Subscriber"],
)

PRESETS = {
    "High-Risk Attrition Candidate": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 94.85,
        "TotalCharges": 189.70,
    },
    "Loyal Long-term Customer": {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 64.50,
        "TotalCharges": 3870.00,
    },
    "Standard New Subscriber": {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 45.20,
        "TotalCharges": 542.40,
    },
}

selected_preset = PRESETS.get(preset_choice, PRESETS["High-Risk Attrition Candidate"])

# Check API health
api_online = False
try:
    health_resp = requests.get(f"{api_url}/health", timeout=1.5)
    if health_resp.status_code == 200:
        api_online = True
        st.sidebar.success("Backend: Connected (HTTP 200)")
    else:
        st.sidebar.warning(f"Backend Degraded ({health_resp.status_code})")
except Exception:
    st.sidebar.error("Backend: Offline (FastAPI not reachable)")

st.sidebar.markdown("---")
st.sidebar.caption("Telco Churn AI Production Scaffold | v1.0.0")

# ==============================================================================
# HEADER
# ==============================================================================

st.markdown('<div class="main-title">Telco Customer Churn Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Real-time churn risk forecasting, predictive driver discovery, and prescriptive retention actions.</div>',
    unsafe_allow_html=True,
)

tab_single, tab_batch, tab_analytics = st.tabs([
    "👤 Single Customer Prediction",
    "📁 Batch Scoring (CSV)",
    "📈 Model Analytics & Explanations",
])

# ==============================================================================
# TAB 1: SINGLE CUSTOMER PREDICTION
# ==============================================================================

with tab_single:
    st.subheader("Customer Profile & Contract Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender = st.selectbox("Gender", ["Female", "Male"], index=0 if selected_preset["gender"] == "Female" else 1)
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", index=selected_preset["SeniorCitizen"])
        partner = st.selectbox("Partner", ["No", "Yes"], index=1 if selected_preset["Partner"] == "Yes" else 0)
        dependents = st.selectbox("Dependents", ["No", "Yes"], index=1 if selected_preset["Dependents"] == "Yes" else 0)
        tenure = st.slider("Tenure (Months with Company)", min_value=0, max_value=72, value=int(selected_preset["tenure"]))

    with col2:
        st.markdown("**Services Subscribed**")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if selected_preset["PhoneService"] == "Yes" else 1)
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=["No", "Yes", "No phone service"].index(selected_preset["MultipleLines"]))
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(selected_preset["InternetService"]))
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(selected_preset["OnlineSecurity"]))
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(selected_preset["OnlineBackup"]))
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(selected_preset["DeviceProtection"]))
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(selected_preset["TechSupport"]))
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(selected_preset["StreamingTV"]))
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(selected_preset["StreamingMovies"]))

    with col3:
        st.markdown("**Contract & Financials**")
        contract = st.selectbox("Contract Term", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(selected_preset["Contract"]))
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if selected_preset["PaperlessBilling"] == "Yes" else 1)
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(selected_preset["PaymentMethod"]),
        )
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=10.0, max_value=250.0, value=float(selected_preset["MonthlyCharges"]), step=1.0)
        default_total = float(selected_preset["TotalCharges"]) if selected_preset["TotalCharges"] != " " else round(tenure * monthly_charges, 2)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=15000.0, value=float(default_total), step=10.0)

    customer_payload = {
        "customerID": "PRED-001",
        "gender": gender,
        "SeniorCitizen": int(senior),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": int(tenure),
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    if st.button("🚀 Predict Churn Probability & Strategy", type="primary", use_container_width=True):
        prediction_result = None

        if api_online:
            try:
                res = requests.post(f"{api_url}/predict", json=customer_payload, timeout=3.0)
                if res.status_code == 200:
                    prediction_result = res.json()
                else:
                    st.error(f"API Error ({res.status_code}): {res.text}")
            except Exception as e:
                st.warning(f"Failed to communicate with API: {e}")

        # Fallback to local model service if API is offline
        if prediction_result is None and use_direct_model_fallback:
            try:
                from app.main import model_service
                prediction_result = model_service.predict_single(customer_payload).model_dump()
                st.info("ℹ️ Scored using local model fallback.")
            except Exception as e:
                st.error(f"Local model scoring failed: {e}")

        if prediction_result:
            prob = prediction_result["churn_probability"]
            risk = prediction_result["risk_level"]
            will_churn = prediction_result["churn_prediction"]
            threshold = prediction_result["decision_threshold"]

            st.markdown("---")
            st.subheader("Evaluation Results & Prescriptive Retention Actions")

            res_col1, res_col2, res_col3 = st.columns([1, 1, 1.5])

            with res_col1:
                st.metric("Churn Probability", f"{prob * 100:.1f}%")
                st.progress(float(prob))

            with res_col2:
                badge_class = f"badge-{risk.lower()}"
                st.markdown(f"**Risk Level:**<br><span class='{badge_class}'>{risk.upper()} RISK</span>", unsafe_allow_html=True)
                st.caption(f"Decision Threshold: {threshold}")

            with res_col3:
                st.markdown("**Executive Action Summary:**")
                st.info(prediction_result["recommended_action"])

            rec_col1, rec_col2 = st.columns(2)

            with rec_col1:
                st.markdown("#### ⚠️ Key Risk Drivers")
                for driver in prediction_result.get("risk_drivers", []):
                    st.markdown(f"<div class='driver-card'>• {driver}</div>", unsafe_allow_html=True)

            with rec_col2:
                st.markdown("#### 💡 Prescriptive Retention Recommendations")
                for rec in prediction_result.get("retention_recommendations", []):
                    st.markdown(f"<div class='recommendation-card'>✅ {rec}</div>", unsafe_allow_html=True)

# ==============================================================================
# TAB 2: BATCH SCORING
# ==============================================================================

with tab_batch:
    st.subheader("Batch Scoring via CSV Dataset")
    st.markdown("Upload a customer CSV matching the Telco schema to compute bulk churn risk and export scores.")

    sample_test_path = Path("data/processed/test.csv")
    if sample_test_path.exists():
        with open(sample_test_path, "rb") as f:
            st.download_button(
                label="📥 Download Sample CSV Template (test.csv)",
                data=f,
                file_name="telco_churn_batch_template.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} rows. Dataset preview:")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("⚡ Score Uploaded Dataset", type="primary"):
            scored_data = None

            if api_online:
                try:
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    resp = requests.post(f"{api_url}/predict/csv", files=files, timeout=30.0)
                    if resp.status_code == 200:
                        scored_data = resp.json()
                    else:
                        st.error(f"API scoring failed: {resp.text}")
                except Exception as e:
                    st.warning(f"API error: {e}")

            if scored_data is None and use_direct_model_fallback:
                try:
                    from app.main import model_service
                    records = batch_df.to_dict(orient="records")
                    batch_res = model_service.predict_batch(records)
                    scored_data = batch_res.model_dump()
                    st.info("ℹ️ Scored using local model fallback.")
                except Exception as e:
                    st.error(f"Fallback batch scoring failed: {e}")

            if scored_data:
                st.markdown("---")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Customers Scored", scored_data["total_customers"])
                m2.metric("High-Risk Count", scored_data["high_risk_count"])
                m3.metric("Medium-Risk Count", scored_data["medium_risk_count"])
                m4.metric("Predicted Churn Rate", f"{scored_data['churn_rate_predicted_pct']}%")

                # Format scored table
                preds = scored_data["predictions"]
                pred_df = pd.DataFrame(preds)

                # Combine with original if same length
                if len(pred_df) == len(batch_df):
                    display_df = pd.concat([batch_df.reset_index(drop=True), pred_df[["churn_probability", "churn_prediction", "risk_level", "recommended_action"]]], axis=1)
                else:
                    display_df = pred_df

                st.markdown("#### Scored Records Preview")
                st.dataframe(display_df, use_container_width=True)

                csv_buffer = io.StringIO()
                display_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="💾 Download Complete Scored Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="telco_churn_predictions.csv",
                    mime="text/csv",
                )

# ==============================================================================
# TAB 3: MODEL ANALYTICS & EXPLANATIONS
# ==============================================================================

with tab_analytics:
    st.subheader("Model Evaluation & Global Interpretability")

    metrics_file = Path("models/metrics.json")
    importance_file = Path("models/feature_importance.json")

    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Selected Model", metrics.get("model_name", "xgboost").upper())
        k2.metric("Test ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}")
        k3.metric("Test Accuracy", f"{metrics.get('accuracy', 0.0):.4f}")
        k4.metric("Test F1 Score", f"{metrics.get('f1', 0.0):.4f}")
        k5.metric("Decision Threshold", f"{metrics.get('optimal_threshold', 0.50):.4f}")

        col_cm, col_chart = st.columns([1, 1.8])

        with col_cm:
            st.markdown("#### Confusion Matrix")
            cm = np.array(metrics.get("confusion_matrix", [[0, 0], [0, 0]]))
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.8)
            fig_cm.colorbar(cax)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_cm.text(x=j, y=i, s=str(cm[i, j]), va="center", ha="center", size="large", weight="bold")
            ax_cm.set_xlabel("Predicted Label")
            ax_cm.set_ylabel("True Label")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Retain (0)", "Churn (1)"])
            ax_cm.set_yticklabels(["Retain (0)", "Churn (1)"])
            st.pyplot(fig_cm)

        with col_chart:
            st.markdown("#### Top Feature Importances")
            if importance_file.exists():
                with open(importance_file, "r", encoding="utf-8") as f:
                    feat_imp = json.load(f)

                if isinstance(feat_imp, list):
                    top_features = feat_imp[:12]
                    names = [item["feature"] for item in top_features][::-1]
                    scores = [item["importance"] for item in top_features][::-1]
                else:
                    top_features = list(feat_imp.items())[:12]
                    names = [item[0] for item in top_features][::-1]
                    scores = [item[1] for item in top_features][::-1]

                fig_imp, ax_imp = plt.subplots(figsize=(7, 4.2))
                ax_imp.barh(names, scores, color="#2563EB")
                ax_imp.set_xlabel("Relative Feature Importance")
                ax_imp.set_title("Top 12 Churn Predictive Factors")
                plt.tight_layout()
                st.pyplot(fig_imp)
            else:
                st.info("Feature importance artifact not found.")

    else:
        st.warning("Model metrics not found. Run 'python src/train.py' to generate performance statistics.")
