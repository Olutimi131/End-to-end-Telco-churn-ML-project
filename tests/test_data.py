"""Unit tests for data generation, cleaning, and preprocessing."""

import numpy as np
import pandas as pd
import pytest

from src.telco_churn.config import load_config
from src.telco_churn.data.dataset import generate_synthetic_data, split_data
from src.telco_churn.data.preprocessing import build_preprocessor, clean_telco_data


def test_generate_synthetic_data_schema():
    """Verify generated synthetic dataset matches expected IBM schema and has no NaN in key columns."""
    df = generate_synthetic_data(n_samples=200, seed=123)
    assert len(df) == 200
    expected_cols = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    for col in expected_cols:
        assert col in df.columns
    assert set(df["Churn"].unique()).issubset({"Yes", "No"})


def test_clean_telco_data_handles_blanks_and_types():
    """Verify cleaning handles blanks in TotalCharges and maps target Churn."""
    raw = pd.DataFrame(
        {
            "customerID": ["123", "456"],
            "tenure": [0, 10],
            "MonthlyCharges": [25.0, 50.0],
            "TotalCharges": [" ", "500.0"],
            "Churn": ["No", "Yes"],
        }
    )
    cleaned = clean_telco_data(raw, is_training=True)

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["TotalCharges"].iloc[1] == 500.0
    assert cleaned["Churn"].iloc[0] == 0
    assert cleaned["Churn"].iloc[1] == 1


def test_split_data_stratification():
    """Verify train/test split maintains target class stratification."""
    cfg = load_config()
    df = generate_synthetic_data(n_samples=500, seed=42)
    train_df, test_df = split_data(df, cfg, save=False)

    assert len(train_df) == 400
    assert len(test_df) == 100

    train_churn_pct = (train_df["Churn"] == "Yes").mean()
    test_churn_pct = (test_df["Churn"] == "Yes").mean()
    assert abs(train_churn_pct - test_churn_pct) < 0.05


def test_preprocessing_pipeline_fit_transform():
    """Verify preprocessing pipeline fits and transforms cleanly into numerical array."""
    cfg = load_config()
    df = generate_synthetic_data(n_samples=150, seed=42)
    cleaned = clean_telco_data(df, is_training=False)

    pipeline = build_preprocessor(cfg)
    transformed = pipeline.fit_transform(cleaned)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 150
    assert transformed.shape[1] > 20
    assert not np.isnan(transformed).any()
