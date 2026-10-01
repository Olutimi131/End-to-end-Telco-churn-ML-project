"""Unit tests for model prediction, probability bounds, risk banding, and metrics."""

import pandas as pd
import pytest

from src.telco_churn.config import load_config
from src.telco_churn.data.dataset import generate_synthetic_data
from src.telco_churn.models.predict import TelcoChurnPredictor


@pytest.fixture(scope="module")
def predictor():
    cfg = load_config()
    return TelcoChurnPredictor(config=cfg)


def test_predict_single_bounds_and_structure(predictor):
    """Test single customer inference returns expected schema and valid probabilities."""
    sample_customer = {
        "customerID": "TEST-001",
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
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.35,
        "TotalCharges": 139.05,
    }

    result = predictor.predict_single(sample_customer)

    assert 0.0 <= result["churn_probability"] <= 1.0
    assert isinstance(result["churn_prediction"], bool)
    assert result["risk_level"] in ["Low", "Medium", "High"]
    assert len(result["risk_drivers"]) > 0
    assert len(result["retention_recommendations"]) > 0


def test_predict_batch_returns_expected_columns(predictor):
    """Test batch inference appends probability, prediction, and risk tier."""
    df = generate_synthetic_data(n_samples=25, seed=99)
    scored = predictor.predict_batch(df)

    assert len(scored) == 25
    assert "churn_probability" in scored.columns
    assert "churn_prediction" in scored.columns
    assert "risk_tier" in scored.columns
    assert scored["churn_probability"].between(0.0, 1.0).all()
    assert set(scored["risk_tier"].unique()).issubset({"Low", "Medium", "High"})


def test_custom_threshold_effect(predictor):
    """Test custom threshold alters the binary prediction cutoff."""
    sample = {
        "customerID": "TEST-002",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 40,
        "PhoneService": "Yes",
        "MultipleLines": "No",
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
        "MonthlyCharges": 45.0,
        "TotalCharges": 1800.0,
    }

    res_strict = predictor.predict_single(sample, threshold=0.99)
    assert res_strict["churn_prediction"] is False

    res_lenient = predictor.predict_single(sample, threshold=0.01)
    assert res_lenient["churn_prediction"] is True
