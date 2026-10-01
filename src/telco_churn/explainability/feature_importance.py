"""Feature importance extraction and customer-level risk explanations."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def extract_pipeline_feature_names(preprocessor: Pipeline) -> list[str]:
    """Retrieve feature names out of the fitted preprocessing pipeline."""
    try:
        col_transformer = preprocessor.named_steps.get("column_preprocessor")
        if col_transformer is not None and hasattr(col_transformer, "get_feature_names_out"):
            raw_names = col_transformer.get_feature_names_out()
            # Clean up prefixes like 'num__' and 'cat__' for cleaner display
            clean_names = [
                name.replace("num__", "").replace("cat__", "") for name in raw_names
            ]
            return list(clean_names)
    except Exception as e:
        logger.warning(f"Could not extract feature names from preprocessor: {e}")
    return []


def extract_global_feature_importance(pipeline: Pipeline, top_n: int = 25) -> list[dict[str, Any]]:
    """Extract global feature importance or coefficient weights from fitted pipeline."""
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if preprocessor is None or classifier is None:
        return []

    feature_names = extract_pipeline_feature_names(preprocessor)
    if not feature_names:
        return []

    importances = None
    metric_type = "importance"

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
        metric_type = "feature_importance"
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_[0])
        metric_type = "abs_coefficient"

    if importances is None or len(importances) != len(feature_names):
        return []

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    total_imp = importance_df["importance"].sum()
    if total_imp > 0:
        importance_df["relative_pct"] = (importance_df["importance"] / total_imp) * 100
    else:
        importance_df["relative_pct"] = 0.0

    top_features = importance_df.head(top_n).to_dict(orient="records")
    for item in top_features:
        item["importance"] = round(float(item["importance"]), 5)
        item["relative_pct"] = round(float(item["relative_pct"]), 2)
        item["metric_type"] = metric_type

    return top_features


def explain_customer_risk_factors(
    customer: pd.Series | dict[str, Any],
    churn_probability: float,
    top_n: int = 4,
) -> dict[str, Any]:
    """Generate human-readable risk drivers and retention recommendations for a customer.

    Analyzes key predictive attributes and returns prescriptive retention recommendations.
    """
    row = dict(customer)
    risk_drivers = []
    retention_recommendations = []

    # 1. Contract risk
    contract = str(row.get("Contract", ""))
    if contract == "Month-to-month":
        risk_drivers.append("Month-to-month contract (low commitment & high churn elasticity)")
        retention_recommendations.append("Offer 15% discount on an annual contract upgrade")
    elif contract == "One year":
        retention_recommendations.append("Check satisfaction survey 60 days before contract renewal")

    # 2. Tenure risk
    try:
        tenure = float(row.get("tenure", 0))
        if tenure <= 6:
            risk_drivers.append(f"Early customer lifecycle (tenure {int(tenure)} months - critical onboarding period)")
            retention_recommendations.append("Assign onboarding specialist and trigger proactive check-in call")
        elif tenure <= 12:
            risk_drivers.append(f"First-year customer (tenure {int(tenure)} months)")
    except Exception:
        pass

    # 3. Monthly charges / Fiber optic
    try:
        monthly = float(row.get("MonthlyCharges", 0))
        if monthly >= 80.0:
            risk_drivers.append(f"High monthly billing (${monthly:.2f}/mo increases price sensitivity)")
            retention_recommendations.append("Review bundle package to optimize plan cost or offer loyalty credit")
    except Exception:
        pass

    # 4. Payment method
    payment = str(row.get("PaymentMethod", ""))
    if payment == "Electronic check":
        risk_drivers.append("Payment method: Electronic check (correlated with higher payment friction & churn)")
        retention_recommendations.append("Incentivize switch to Auto-Pay with a one-time $10 account credit")

    # 5. Technical support & security services
    tech_support = str(row.get("TechSupport", ""))
    online_security = str(row.get("OnlineSecurity", ""))
    if tech_support == "No" and row.get("InternetService") != "No":
        risk_drivers.append("No active Tech Support subscription")
        retention_recommendations.append("Offer 3 months free Tech Support VIP bundle")
    if online_security == "No" and row.get("InternetService") != "No":
        risk_drivers.append("No Online Security add-on installed")

    # 6. Senior citizen
    senior = row.get("SeniorCitizen", 0)
    if senior in (1, "1", "Yes"):
        retention_recommendations.append("Provide senior customer dedicated phone support priority")

    if not risk_drivers:
        risk_drivers.append("Stable tenure and balanced usage profile")
    if not retention_recommendations:
        retention_recommendations.append("Maintain standard service quality and regular satisfaction check-ins")

    # Risk level banding
    if churn_probability < 0.30:
        risk_level = "Low"
        recommended_action = "Nurture & cross-sell optional services"
    elif churn_probability < 0.60:
        risk_level = "Medium"
        recommended_action = "Targeted engagement & value reinforcement campaign"
    else:
        risk_level = "High"
        recommended_action = "Immediate retention outreach & contract incentive"

    return {
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "risk_drivers": risk_drivers[:top_n],
        "retention_recommendations": retention_recommendations[:top_n],
    }
