"""Unit and integration tests for scaffolded Telco Customer Churn components."""

from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, model_service
from src.data_ingestion import DataIngestionPipeline
from src.train import ChurnTrainingPipeline


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_customer():
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
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
        "MonthlyCharges": 85.50,
        "TotalCharges": 256.50,
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "xgboost" in data["model_name"].lower() or "forest" in data["model_name"].lower()


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "roc_auc" in data
    assert data["roc_auc"] > 0.70
    assert "accuracy" in data
    assert "optimal_threshold" in data


def test_features_endpoint(client):
    response = client.get("/features")
    assert response.status_code == 200
    features = response.json()
    assert isinstance(features, (dict, list))
    assert len(features) > 0


def test_single_prediction(client, sample_customer):
    response = client.post("/predict", json=sample_customer)
    assert response.status_code == 200
    res = response.json()
    assert "churn_probability" in res
    assert 0.0 <= res["churn_probability"] <= 1.0
    assert res["risk_level"] in ["Low", "Medium", "High"]
    assert len(res["risk_drivers"]) > 0
    assert len(res["retention_recommendations"]) > 0


def test_batch_prediction(client, sample_customer):
    batch_payload = {
        "customers": [sample_customer, {**sample_customer, "customerID": "TEST-002", "Contract": "Two year", "tenure": 60}],
        "threshold": 0.5,
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    res = response.json()
    assert res["total_customers"] == 2
    assert len(res["predictions"]) == 2
    assert res["predictions"][0]["customer_id"] == "TEST-001"
    assert res["predictions"][1]["customer_id"] == "TEST-002"


def test_data_ingestion_pipeline(tmp_path):
    pipeline = DataIngestionPipeline(
        raw_data_path="data/raw/telco_churn.csv",
        processed_dir=tmp_path,
        test_size=0.2,
        random_state=42,
    )
    train_df, test_df = pipeline.run()
    assert len(train_df) > 0
    assert len(test_df) > 0
    assert (tmp_path / "train.csv").exists()
    assert (tmp_path / "test.csv").exists()
