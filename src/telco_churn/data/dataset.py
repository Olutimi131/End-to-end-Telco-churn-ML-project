"""Data loading, acquisition, synthetic generation, and train/test splitting."""

import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import AppConfig, load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_synthetic_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic Telco Churn dataset matching IBM schema.

    Useful for offline development, integration tests, and simulations.
    """
    rng = np.random.default_rng(seed)

    customer_ids = [f"{rng.integers(1000, 9999)}-{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}" for _ in range(n_samples)]
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], p=[0.84, 0.16], size=n_samples)
    partner = rng.choice(["Yes", "No"], p=[0.48, 0.52], size=n_samples)
    dependents = rng.choice(["Yes", "No"], p=[0.30, 0.70], size=n_samples)
    
    # Tenure in months (0 to 72)
    tenure = rng.integers(0, 73, size=n_samples)

    phone_service = rng.choice(["Yes", "No"], p=[0.90, 0.10], size=n_samples)
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(rng.choice(["Yes", "No"], p=[0.45, 0.55]))

    internet_service = rng.choice(["DSL", "Fiber optic", "No"], p=[0.34, 0.44, 0.22], size=n_samples)
    
    internet_addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    addons_data: dict[str, list[str]] = {addon: [] for addon in internet_addons}
    
    for net in internet_service:
        for addon in internet_addons:
            if net == "No":
                addons_data[addon].append("No internet service")
            else:
                addons_data[addon].append(rng.choice(["Yes", "No"], p=[0.38, 0.62]))

    contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.21, 0.24], size=n_samples)
    paperless = rng.choice(["Yes", "No"], p=[0.59, 0.41], size=n_samples)
    payment_method = rng.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        p=[0.34, 0.23, 0.22, 0.21],
        size=n_samples,
    )

    monthly_charges = []
    for net in internet_service:
        if net == "No":
            monthly_charges.append(round(rng.uniform(18.5, 25.5), 2))
        elif net == "DSL":
            monthly_charges.append(round(rng.uniform(30.0, 75.0), 2))
        else:  # Fiber optic
            monthly_charges.append(round(rng.uniform(70.0, 118.5), 2))

    total_charges = []
    for t, m in zip(tenure, monthly_charges):
        if t == 0:
            total_charges.append(" ")  # Canonical IBM dataset representation of 0 tenure
        else:
            jitter = rng.normal(0, 10.0)
            total = max(round(t * m + jitter, 2), m)
            total_charges.append(str(round(total, 2)))

    # Compute realistic churn probability
    churn_prob = (
        0.35 * (contract == "Month-to-month")
        + 0.25 * (payment_method == "Electronic check")
        + 0.20 * (internet_service == "Fiber optic")
        + 0.15 * (senior_citizen == 1)
        - 0.35 * (tenure > 36)
        - 0.20 * (contract == "Two year")
        + rng.normal(0, 0.15, size=n_samples)
    )
    churn_prob = 1 / (1 + np.exp(-churn_prob))
    churn = ["Yes" if p > 0.52 else "No" for p in churn_prob]

    data = {
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        **addons_data,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    }

    return pd.DataFrame(data)


def download_raw_data(url: str, destination: Path) -> Path:
    """Download the raw dataset from a remote URL."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Telco Customer Churn dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, str(destination))
        logger.info(f"Dataset successfully downloaded to {destination}")
    except Exception as e:
        logger.warning(f"Could not download dataset from {url}: {e}. Generating synthetic fallback...")
        df = generate_synthetic_data(n_samples=7043)
        df.to_csv(destination, index=False)
        logger.info(f"Generated synthetic dataset saved to {destination}")
    return destination


def load_raw_data(config: AppConfig | None = None) -> pd.DataFrame:
    """Load the raw Telco dataset from disk or download/generate if missing."""
    cfg = config or load_config()
    file_path = cfg.paths.raw_data_file

    if not file_path.exists():
        download_raw_data(cfg.data.source_url, file_path)

    logger.info(f"Loading raw data from {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    config: AppConfig | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the dataset into stratified train and test partitions.

    Args:
        df: Raw or cleaned DataFrame containing target column.
        config: Application configuration.
        save: Whether to save train and test sets to disk.

    Returns:
        tuple of (train_df, test_df)
    """
    cfg = config or load_config()
    target_col = cfg.data.target_column
    test_size = cfg.data.test_size
    seed = cfg.project.seed

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[target_col],
    )

    logger.info(f"Split data into train={train_df.shape} and test={test_df.shape}")

    if save:
        cfg.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(cfg.paths.train_data_file, index=False)
        test_df.to_csv(cfg.paths.test_data_file, index=False)
        logger.info(f"Saved train data to {cfg.paths.train_data_file}")
        logger.info(f"Saved test data to {cfg.paths.test_data_file}")

    return train_df, test_df
