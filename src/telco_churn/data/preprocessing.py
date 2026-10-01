"""Data cleaning, custom feature engineering transformers, and preprocessing pipelines."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import AppConfig, load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def clean_telco_data(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """Perform initial cleaning on Telco Customer Churn DataFrame.

    - Strips whitespace.
    - Converts TotalCharges to numeric, replacing whitespace with 0.0 for zero tenure.
    - Maps target 'Churn' to binary 0/1 if present.
    """
    df = df.copy()

    # Drop customerID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # TotalCharges cleaning
    if "TotalCharges" in df.columns:
        if df["TotalCharges"].dtype == object:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip(),
                errors="coerce",
            )
        # If tenure == 0 and TotalCharges is NaN, fill with 0.0
        if "tenure" in df.columns:
            df.loc[(df["tenure"] == 0) & (df["TotalCharges"].isna()), "TotalCharges"] = 0.0
        # If any remaining NaN, fill with median or monthlyCharges * tenure
        median_total = df["TotalCharges"].median() if not df["TotalCharges"].isna().all() else 0.0
        df["TotalCharges"] = df["TotalCharges"].fillna(median_total)

    # Clean Churn target column
    if "Churn" in df.columns and is_training:
        if df["Churn"].dtype == object:
            df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)

    return df


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Domain-specific feature engineering transformer for Telco Churn."""

    def __init__(self, add_tenure_groups: bool = True, add_service_aggregations: bool = True):
        self.add_tenure_groups = add_tenure_groups
        self.add_service_aggregations = add_service_aggregations

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TelcoFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # Handle TotalCharges numeric conversion if needed
        if "TotalCharges" in X_out.columns and X_out["TotalCharges"].dtype == object:
            X_out["TotalCharges"] = pd.to_numeric(
                X_out["TotalCharges"].astype(str).str.strip(),
                errors="coerce",
            )
            if "tenure" in X_out.columns:
                X_out.loc[(X_out["tenure"] == 0) & (X_out["TotalCharges"].isna()), "TotalCharges"] = 0.0
            X_out["TotalCharges"] = X_out["TotalCharges"].fillna(0.0)

        # Derived Financial & Usage Ratios
        if "MonthlyCharges" in X_out.columns and "TotalCharges" in X_out.columns and "tenure" in X_out.columns:
            # Expected cumulative charges vs actual cumulative charges
            expected_charges = X_out["MonthlyCharges"] * np.maximum(X_out["tenure"], 1)
            X_out["charges_difference"] = X_out["TotalCharges"] - expected_charges
            X_out["monthly_to_total_ratio"] = X_out["MonthlyCharges"] / (X_out["TotalCharges"] + 1.0)

        # Tenure Grouping
        if self.add_tenure_groups and "tenure" in X_out.columns:
            bins = [-1, 12, 24, 48, 72]
            labels = ["0-12m", "12-24m", "24-48m", "48-72m"]
            X_out["tenure_group"] = pd.cut(X_out["tenure"], bins=bins, labels=labels).astype(str)

        # Family flag
        if "Partner" in X_out.columns and "Dependents" in X_out.columns:
            X_out["has_family"] = (
                ((X_out["Partner"] == "Yes") | (X_out["Dependents"] == "Yes")).astype(int).astype(str)
            )

        # Payment automatic flag
        if "PaymentMethod" in X_out.columns:
            X_out["is_automatic_payment"] = (
                X_out["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int).astype(str)
            )

        # Service aggregations
        if self.add_service_aggregations:
            security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
            streaming_cols = ["StreamingTV", "StreamingMovies"]

            sec_present = [col for col in security_cols if col in X_out.columns]
            if sec_present:
                X_out["security_services_count"] = (X_out[sec_present] == "Yes").sum(axis=1)

            stream_present = [col for col in streaming_cols if col in X_out.columns]
            if stream_present:
                X_out["streaming_services_count"] = (X_out[stream_present] == "Yes").sum(axis=1)

        return X_out


def build_preprocessor(config: AppConfig | None = None) -> Pipeline:
    """Build the complete scikit-learn preprocessing ColumnTransformer pipeline."""
    cfg = config or load_config()

    # Base feature lists
    numeric_features = list(cfg.data.numeric_features) + [
        "charges_difference",
        "monthly_to_total_ratio",
        "security_services_count",
        "streaming_services_count",
    ]
    
    categorical_features = list(cfg.data.categorical_features) + [
        "tenure_group",
        "has_family",
        "is_automatic_payment",
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    column_preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    full_preprocessor = Pipeline(
        steps=[
            ("feature_engineer", TelcoFeatureEngineer()),
            ("column_preprocessor", column_preprocessor),
        ]
    )

    return full_preprocessor
