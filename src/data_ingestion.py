"""Data Ingestion Pipeline for Telco Customer Churn.

This module provides production-grade data ingestion, validation, cleaning,
and train/test splitting without cloud dependencies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("data_ingestion")

# Standard schema definition for Telco Customer Churn
EXPECTED_COLUMNS = [
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

NUMERICAL_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]
TARGET_COLUMN = "Churn"


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic Telco churn dataset when raw data is not found."""
    logger.info(f"Generating {n_samples} synthetic Telco customer records...")
    rng = np.random.default_rng(seed)

    customer_ids = [
        f"{rng.integers(1000, 9999)}-{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}"
        for _ in range(n_samples)
    ]
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], p=[0.84, 0.16], size=n_samples)
    partner = rng.choice(["Yes", "No"], p=[0.48, 0.52], size=n_samples)
    dependents = rng.choice(["Yes", "No"], p=[0.30, 0.70], size=n_samples)
    tenure = rng.integers(0, 73, size=n_samples)
    phone_service = rng.choice(["Yes", "No"], p=[0.90, 0.10], size=n_samples)

    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.45, 0.55]))

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22], size=n_samples)

    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    addons_data = {addon: [] for addon in addons}

    for net in internet_service:
        for addon in addons:
            if net == "No":
                addons_data[addon].append("No internet service")
            else:
                addons_data[addon].append(rng.choice(["Yes", "No"], p=[0.38, 0.62]))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.21, 0.24], size=n_samples)
    paperless = rng.choice(["Yes", "No"], p=[0.59, 0.41], size=n_samples)
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        p=[0.34, 0.23, 0.22, 0.21],
        size=n_samples,
    )

    monthly_charges = []
    for net in internet_service:
        if net == "No":
            monthly_charges.append(round(float(rng.uniform(18.5, 25.5)), 2))
        elif net == "DSL":
            monthly_charges.append(round(float(rng.uniform(30.0, 75.0)), 2))
        else:
            monthly_charges.append(round(float(rng.uniform(70.0, 118.5)), 2))

    total_charges = []
    for t, m in zip(tenure, monthly_charges):
        if t == 0:
            total_charges.append(" ")
        else:
            jitter = rng.normal(0, 10.0)
            total = max(round(t * m + jitter, 2), m)
            total_charges.append(str(round(total, 2)))

    # Real-world churn probability formula
    prob = (
        0.35 * (contract == "Month-to-month")
        + 0.25 * (payment_method == "Electronic check")
        + 0.20 * (internet_service == "Fiber optic")
        + 0.15 * (senior_citizen == 1)
        - 0.35 * (tenure > 36)
        - 0.20 * (contract == "Two year")
        + rng.normal(0, 0.15, size=n_samples)
    )
    prob = 1 / (1 + np.exp(-prob))
    churn = ["Yes" if p > 0.52 else "No" for p in prob]

    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": addons_data["OnlineSecurity"],
        "OnlineBackup": addons_data["OnlineBackup"],
        "DeviceProtection": addons_data["DeviceProtection"],
        "TechSupport": addons_data["TechSupport"],
        "StreamingTV": addons_data["StreamingTV"],
        "StreamingMovies": addons_data["StreamingMovies"],
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })
    return df


class DataIngestionPipeline:
    """Manages raw data loading, schema validation, type parsing, and dataset splitting."""

    def __init__(
        self,
        raw_data_path: str | Path = "data/raw/telco_churn.csv",
        processed_dir: str | Path = "data/processed",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.raw_data_path = Path(raw_data_path)
        self.processed_dir = Path(processed_dir)
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self) -> pd.DataFrame:
        """Load data from disk or generate synthetic dataset if not found."""
        if self.raw_data_path.exists():
            logger.info(f"Loading raw dataset from {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)
        else:
            logger.warning(f"Raw data file not found at {self.raw_data_path}. Generating synthetic dataset.")
            self.raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df = generate_synthetic_data(n_samples=2500, seed=self.random_state)
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Saved generated raw dataset to {self.raw_data_path}")

        logger.info(f"Raw dataset loaded with shape: {df.shape}")
        return df

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate required columns in incoming dataframe."""
        missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        logger.info("Schema validation successful.")
        return True

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data: handle blank TotalCharges, cast types, map binary Churn."""
        df_clean = df.copy()

        # Handle TotalCharges blanks / whitespace
        if "TotalCharges" in df_clean.columns:
            if df_clean["TotalCharges"].dtype == object:
                df_clean["TotalCharges"] = pd.to_numeric(
                    df_clean["TotalCharges"].astype(str).str.strip(),
                    errors="coerce",
                )
            # Impute NaN TotalCharges where tenure == 0 with 0.0
            if "tenure" in df_clean.columns:
                df_clean.loc[(df_clean["tenure"] == 0) & (df_clean["TotalCharges"].isna()), "TotalCharges"] = 0.0

            # Fallback for remaining NaNs
            median_val = df_clean["TotalCharges"].median()
            df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(median_val if not np.isnan(median_val) else 0.0)

        # Standardize Churn target to 0 and 1
        if "Churn" in df_clean.columns:
            if df_clean["Churn"].dtype == object:
                df_clean["Churn"] = df_clean["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)

        logger.info(f"Data cleaning complete. Non-null records: {len(df_clean)}")
        return df_clean

    def split_and_save(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform stratified split into train and test sets and persist to processed directory."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        stratify = df["Churn"] if "Churn" in df.columns else None
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

        train_path = self.processed_dir / "train.csv"
        test_path = self.processed_dir / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info(f"Saved train set ({len(train_df)} rows) -> {train_path}")
        logger.info(f"Saved test set ({len(test_df)} rows) -> {test_path}")

        return train_df, test_df

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Execute complete ingestion pipeline."""
        df = self.load_data()
        self.validate_schema(df)
        df_clean = self.clean_data(df)
        train_df, test_df = self.split_and_save(df_clean)
        return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Ingest and preprocess Telco Customer Churn data.")
    parser.add_argument("--raw-path", type=str, default="data/raw/telco_churn.csv", help="Path to raw CSV dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory for split outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test set (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed")
    args = parser.parse_args()

    pipeline = DataIngestionPipeline(
        raw_data_path=args.raw_path,
        processed_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.seed,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
