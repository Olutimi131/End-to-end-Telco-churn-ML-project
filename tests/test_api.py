"""Integration tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)


def test_api_root():
    """Verify root endpoint responds with metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["docs"] == "/docs"


def test_api_health():
    """Verify health endpoint reports healthy and model loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert data["model_loaded"] is True
    assert data["model_name"] is not None


def test_api_predict_single_valid():
    """Verify predict endpoint handles valid customer payload."""
    payload = {
        "customerID": "API-TEST-01",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 79.85,
        "TotalCharges": 239.55,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["customer_id"] == "API-TEST-01"
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "risk_drivers" in data
    assert "retention_recommendations" in data


def test_api_predict_validation_error():
    """Verify FastAPI schema validation rejects invalid inputs (e.g. invalid SeniorCitizen value)."""
    invalid_payload = {
        "customerID": "API-BAD-01",
        "gender": "Alien",  # Invalid enum
        "SeniorCitizen": 5,  # Out of range 0..1
        "Partner": "Maybe",  # Invalid enum
    }

    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_api_predict_batch():
    """Verify batch prediction endpoint scores multiple records."""
    sample = {
        "customerID": "CUST-B1",
        "gender": "Male",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.0,
        "TotalCharges": 660.0,
    }

    batch_payload = {"customers": [sample, sample]}
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2


def test_api_metrics_endpoint():
    """Verify /metrics returns evaluation metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]


def test_api_feature_importance_endpoint():
    """Verify /feature-importance returns ranked features."""
    response = client.get("/feature-importance?top_n=5")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) <= 5
    if len(data) > 0:
        assert "feature" in data[0]
        assert "relative_pct" in data[0]
