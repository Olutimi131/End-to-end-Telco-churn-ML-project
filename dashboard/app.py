"""Interactive Streamlit Web Dashboard for Telco Customer Churn Intelligence."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.telco_churn.config import load_config
from src.telco_churn.models.predict import TelcoChurnPredictor
from src.telco_churn.monitoring.drift import evaluate_dataset_drift
from src.telco_churn.utils.io import load_json

# Page configuration
st.set_page_config(
    page_title="Telco Churn Intelligence Hub",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    .risk-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

cfg = load_config()

# Cache model predictor
@st.cache_resource
def get_predictor():
    return TelcoChurnPredictor(config=cfg)

# Cache metrics and artifacts
@st.cache_data
def get_artifacts():
    metrics = load_json(cfg.paths.metrics_file) if cfg.paths.metrics_file.exists() else {}
    feat_imp = load_json(cfg.paths.feature_importance_file) if cfg.paths.feature_importance_file.exists() else []
    baseline = load_json(cfg.paths.drift_baseline_file) if cfg.paths.drift_baseline_file.exists() else {}
    return metrics, feat_imp, baseline

try:
    predictor = get_predictor()
    metrics_data, feature_importance_data, drift_baseline_data = get_artifacts()
    model_ready = True
except Exception as e:
    model_ready = False
    st.error(f"Could not load trained model artifacts: {e}. Please run `python scripts/run_pipeline.py --all` first.")

# Sidebar
st.sidebar.image("https://img.icons8.com/fluency/96/telemarketer.png", width=64)
st.sidebar.title("Telco Intelligence")
st.sidebar.markdown("**Production ML Churn System**")

if model_ready:
    opt_th = metrics_data.get("optimal_threshold_metrics", {}).get("optimal_threshold", 0.47)
    auc_score = metrics_data.get("metrics", {}).get("roc_auc", 0.8447)
    best_model_name = metrics_data.get("model_name", "xgboost").upper()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏆 Active Model")
    st.sidebar.markdown(f"**Model:** `{best_model_name}`")
    st.sidebar.markdown(f"**Test ROC-AUC:** `{auc_score:.4f}`")
    st.sidebar.markdown(f"**Optimal Threshold:** `{opt_th:.2f}`")

    st.sidebar.markdown("---")
    threshold_slider = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=float(opt_th),
        step=0.01,
        help="Probability cutoff above which a customer is flagged for churn intervention.",
    )
else:
    threshold_slider = 0.5

# Navigation tabs
tab_pred, tab_batch, tab_analytics, tab_simulator, tab_drift = st.tabs(
    [
        "🔮 Single Customer Predictor",
        "📂 Batch Scoring & Export",
        "📊 Model Analytics & Explainability",
        "💰 Retention ROI Simulator",
        "📡 Data Drift Monitor",
    ]
)

# ==============================================================================
# TAB 1: Single Customer Predictor
# ==============================================================================
with tab_pred:
    st.markdown('<div class="main-header">Customer Churn Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluate individual subscriber churn probability and generate retention actions.</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes (Senior)" if x == 1 else "No")
        partner = st.selectbox("Partner", ["No", "Yes"])
        dependents = st.selectbox("Dependents", ["No", "Yes"])

    with col2:
        st.subheader("📋 Account & Contract")
        tenure = st.slider("Tenure (Months with company)", min_value=0, max_value=72, value=4)
        contract = st.selectbox("Contract Term", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

    with col3:
        st.subheader("💳 Financials")
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=15.0, max_value=150.0, value=75.50, step=0.5)
        # Default total charges calculation based on tenure
        def_total = max(round(monthly_charges * max(tenure, 1), 2), monthly_charges)
        total_charges = st.number_input("Total Cumulative Charges ($)", min_value=0.0, max_value=10000.0, value=float(def_total), step=10.0)

    st.subheader("🌐 Services Subscribed")
    sc1, sc2, sc3, sc4 = st.columns(4)

    with sc1:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])

    with sc2:
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

    with sc3:
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

    with sc4:
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    predict_btn = st.button("🚀 Score Customer Churn Risk", type="primary", use_container_width=True)

    if predict_btn and model_ready:
        customer_payload = {
            "customerID": "DEMO-001",
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multi_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }

        res = predictor.predict_single(customer_payload, threshold=threshold_slider)
        prob = res["churn_probability"]
        is_churn = res["churn_prediction"]
        risk_level = res["risk_level"]

        st.markdown("---")
        st.markdown("### 🎯 Prediction Results")

        rcol1, rcol2, rcol3 = st.columns([1, 1, 2])

        with rcol1:
            st.metric(
                label="Churn Probability",
                value=f"{prob * 100:.1f}%",
                delta=f"Threshold: {threshold_slider:.2f}",
                delta_color="inverse" if prob >= threshold_slider else "normal",
            )

        with rcol2:
            if risk_level == "High":
                st.markdown('<p style="font-size:1.1rem; margin-top:10px;">Risk Level: <span class="risk-high">HIGH RISK</span></p>', unsafe_allow_html=True)
            elif risk_level == "Medium":
                st.markdown('<p style="font-size:1.1rem; margin-top:10px;">Risk Level: <span class="risk-medium">MEDIUM RISK</span></p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="font-size:1.1rem; margin-top:10px;">Risk Level: <span class="risk-low">LOW RISK</span></p>', unsafe_allow_html=True)

            if is_churn:
                st.error("⚠️ Prediction: Likely to Churn")
            else:
                st.success("✅ Prediction: Likely to Stay")

        with rcol3:
            st.info(f"**Recommended Strategy:** {res['recommended_action']}")

        # Drivers & Retention
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.markdown("#### 🔍 Primary Risk Drivers")
            for driver in res["risk_drivers"]:
                st.markdown(f"- 🔴 {driver}")

        with dcol2:
            st.markdown("#### 🎁 Prescriptive Retention Actions")
            for action in res["retention_recommendations"]:
                st.markdown(f"- 💡 **{action}**")

# ==============================================================================
# TAB 2: Batch Scoring
# ==============================================================================
with tab_batch:
    st.markdown('<div class="main-header">Batch Customer Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload subscriber cohorts to batch score churn probabilities and export targeted retention lists.</div>', unsafe_allow_html=True)

    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        uploaded_file = st.file_uploader("Upload Customer CSV File", type=["csv"])
    with bcol2:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample_btn = st.button("📁 Load Test Set Sample (200 records)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif load_sample_btn:
        if cfg.paths.test_data_file.exists():
            test_full = pd.read_csv(cfg.paths.test_data_file)
            batch_df = test_full.head(200).copy()
            st.success("Loaded 200 records from test split.")

    if batch_df is not None and model_ready:
        with st.spinner("Scoring batch customers..."):
            scored = predictor.predict_batch(batch_df, threshold=threshold_slider)

        total = len(scored)
        churn_count = (scored["churn_prediction"] == "Yes").sum()
        churn_rate = (churn_count / total) * 100
        high_risk_count = (scored["risk_tier"] == "High").sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Customers", total)
        m2.metric("Flagged Churners", f"{churn_count} ({churn_rate:.1f}%)")
        m3.metric("High Risk Band", f"{high_risk_count} ({high_risk_count / total * 100:.1f}%)")
        m4.metric("Applied Threshold", f"{threshold_slider:.2f}")

        # Risk distribution chart
        st.markdown("#### 📊 Risk Tiers Distribution")
        risk_counts = scored["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0)
        
        fig, ax = plt.subplots(figsize=(8, 2.8))
        colors = ["#10B981", "#F59E0B", "#EF4444"]
        sns.barplot(x=risk_counts.index, y=risk_counts.values, palette=colors, ax=ax)
        ax.set_ylabel("Customer Count")
        ax.set_title("Customer Risk Tier Breakdown")
        st.pyplot(fig)
        plt.close(fig)

        # Filter option
        filter_tier = st.multiselect("Filter by Risk Tier:", ["High", "Medium", "Low"], default=["High", "Medium"])
        display_df = scored[scored["risk_tier"].isin(filter_tier)]

        st.dataframe(display_df, use_container_width=True)

        # Export CSV
        csv_data = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Retention List (CSV)",
            data=csv_data,
            file_name="telco_retention_targets.csv",
            mime="text/csv",
            type="primary",
        )

# ==============================================================================
# TAB 3: Model Analytics & Explainability
# ==============================================================================
with tab_analytics:
    st.markdown('<div class="main-header">Model Performance & Feature Explainability</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation benchmarks, confusion matrix, and global feature importance.</div>', unsafe_allow_html=True)

    if model_ready and metrics_data:
        metrics = metrics_data.get("metrics", {})

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        c2.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.4f}")
        c3.metric("F1-Score", f"{metrics.get('f1', 0):.4f}")
        c4.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        c5.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        c6.metric("Recall", f"{metrics.get('recall', 0):.4f}")

        st.markdown("---")
        gcol1, gcol2 = st.columns(2)

        with gcol1:
            st.markdown("#### 🎯 Confusion Matrix (Test Set)")
            cm_data = metrics_data.get("confusion_matrix", {}).get("matrix", [[0, 0], [0, 0]])
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                cm_data,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Retained (Pred)", "Churned (Pred)"],
                yticklabels=["Retained (True)", "Churned (True)"],
                ax=ax,
            )
            ax.set_title("Test Confusion Matrix (Threshold = 0.5)")
            st.pyplot(fig)
            plt.close(fig)

        with gcol2:
            st.markdown("#### 📈 ROC & PR Curves")
            roc_pts = pd.DataFrame(metrics_data.get("roc_curve", []))
            if not roc_pts.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(roc_pts["fpr"], roc_pts["tpr"], color="#2563EB", lw=2, label=f"ROC (AUC = {metrics.get('roc_auc', 0):.3f})")
                ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("Receiver Operating Characteristic")
                ax.legend(loc="lower right")
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")
        st.markdown("#### 🌟 Top Global Churn Drivers (Feature Importance)")
        if feature_importance_data:
            top_feats = pd.DataFrame(feature_importance_data).head(15)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                data=top_feats,
                x="relative_pct",
                y="feature",
                palette="viridis",
                ax=ax,
            )
            ax.set_xlabel("Relative Importance (%)")
            ax.set_ylabel("Feature")
            ax.set_title("Top 15 Predictive Features for Telco Churn")
            st.pyplot(fig)
            plt.close(fig)

# ==============================================================================
# TAB 4: Retention ROI Simulator
# ==============================================================================
with tab_simulator:
    st.markdown('<div class="main-header">Retention Campaign ROI Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulate business financial returns of targeted machine learning retention campaigns.</div>', unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1, 2])

    with sim_col1:
        st.subheader("⚙️ Campaign Parameters")
        target_pool = st.number_input("Subscriber Base Size", min_value=500, max_value=100000, value=10000, step=500)
        base_churn = st.slider("Expected Baseline Churn Rate (%)", min_value=5.0, max_value=50.0, value=26.5, step=0.5) / 100.0
        clv_value = st.number_input("Customer Lifetime Value / Loss ($)", min_value=50.0, max_value=2000.0, value=400.0, step=25.0)
        campaign_cost = st.number_input("Cost per Retention Offer ($)", min_value=5.0, max_value=300.0, value=50.0, step=5.0)
        offer_success = st.slider("Offer Acceptance / Retention Success Rate (%)", min_value=10.0, max_value=90.0, value=40.0, step=5.0) / 100.0

    with sim_col2:
        st.subheader("📊 Financial Impact Comparison")

        actual_churners = int(target_pool * base_churn)
        non_churners = target_pool - actual_churners

        # Strategy 1: Passive (No action)
        lost_passive = actual_churners * clv_value

        # Strategy 2: Blanket Campaign (Offer to everyone)
        blanket_cost = target_pool * campaign_cost
        blanket_saved = int(actual_churners * offer_success)
        blanket_value_saved = blanket_saved * clv_value
        blanket_net_profit = blanket_value_saved - blanket_cost

        # Strategy 3: ML Targeted Campaign (Using model precision & recall at optimal threshold)
        opt_prec = metrics_data.get("optimal_threshold_metrics", {}).get("precision_at_optimal", 0.58)
        opt_rec = metrics_data.get("optimal_threshold_metrics", {}).get("recall_at_optimal", 0.72)

        ml_targeted_churners = int(actual_churners * opt_rec)
        ml_false_positives = int(ml_targeted_churners * (1 - opt_prec) / max(opt_prec, 0.01))
        ml_total_targeted = ml_targeted_churners + ml_false_positives

        ml_cost = ml_total_targeted * campaign_cost
        ml_saved = int(ml_targeted_churners * offer_success)
        ml_value_saved = ml_saved * clv_value
        ml_net_profit = ml_value_saved - ml_cost

        b1, b2, b3 = st.columns(3)
        b1.metric("Passive Loss (No ML)", f"-${lost_passive:,.0f}")
        b2.metric("Blanket Net Profit", f"${blanket_net_profit:,.0f}")
        b3.metric("ML Targeted Net Profit", f"${ml_net_profit:,.0f}", delta=f"+${ml_net_profit - blanket_net_profit:,.0f} vs Blanket")

        # Visual Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(8, 3.5))
        strategies = ["Blanket Campaign", "ML Targeted Strategy"]
        profits = [blanket_net_profit, ml_net_profit]
        bar_colors = ["#F59E0B" if blanket_net_profit > 0 else "#EF4444", "#10B981"]

        bars = ax.bar(strategies, profits, color=bar_colors, width=0.45)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
        ax.set_ylabel("Net Financial Gain ($)")
        ax.set_title("Net Campaign Profit: Blanket Outreach vs ML Precision Targeting")
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (max(profits) * 0.03 if height >= 0 else -max(profits) * 0.08),
                f"${height:,.0f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontweight="bold",
            )
        st.pyplot(fig)
        plt.close(fig)

        st.success(
            f"💡 **Business Takeaway:** Precision ML targeting saves **${ml_cost:,.0f}** in campaign waste "
            f"while retaining **{ml_saved:,} customers**, generating a net profit of **${ml_net_profit:,.0f}**!"
        )

# ==============================================================================
# TAB 5: Data Drift Monitor
# ==============================================================================
with tab_drift:
    st.markdown('<div class="main-header">Data & Model Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Detect Population Stability Index (PSI) shifts and covariate feature drift.</div>', unsafe_allow_html=True)

    if model_ready and drift_baseline_data:
        st.markdown(f"**Reference Training Baseline:** `{drift_baseline_data.get('sample_count', 0)}` records.")

        drift_source = st.radio("Select Evaluation Batch:", ["Current Test Split", "Simulated Drift Batch (Fiber Optic Shift)"])

        if drift_source == "Current Test Split":
            test_data = pd.read_csv(cfg.paths.test_data_file)
            eval_df = test_data
        else:
            # Simulate a real-world drift scenario: shifted tenure and fiber-optic adoption
            test_data = pd.read_csv(cfg.paths.test_data_file).copy()
            test_data["MonthlyCharges"] = test_data["MonthlyCharges"] * 1.35
            test_data["tenure"] = np.clip(test_data["tenure"] - 15, 0, 72)
            eval_df = test_data

        drift_report = evaluate_dataset_drift(eval_df, drift_baseline_data, cfg)

        dstatus = drift_report["overall_status"]
        if dstatus == "HEALTHY":
            st.success("🟢 System Status: HEALTHY — No statistically significant drift detected.")
        elif dstatus == "WARNING_MODERATE_DRIFT":
            st.warning(f"🟡 System Status: WARNING — Moderate shift detected in {drift_report['flagged_features_count']} feature(s).")
        else:
            st.error(f"🔴 System Status: DRIFT DETECTED — Retraining recommended. Flagged: {drift_report['flagged_features']}")

        st.markdown("#### 🔬 Feature-by-Feature Drift Summary")
        drift_rows = []
        for feat, dinfo in drift_report["feature_drifts"].items():
            if dinfo["type"] == "numeric":
                drift_rows.append(
                    {
                        "Feature": feat,
                        "Type": "Numeric",
                        "PSI": dinfo["psi"],
                        "KS p-value": dinfo["p_value"],
                        "Status": dinfo["status"],
                    }
                )
            else:
                drift_rows.append(
                    {
                        "Feature": feat,
                        "Type": "Categorical",
                        "Max Shift": dinfo["max_category_diff"],
                        "KS p-value": "N/A",
                        "Status": dinfo["status"],
                    }
                )

        st.table(pd.DataFrame(drift_rows))
