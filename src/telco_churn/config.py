"""Configuration manager and schemas for Telco Churn."""

from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "telco-customer-churn"
    version: str = "1.0.0"
    description: str = "Telco Churn End-to-End ML System"
    seed: int = 42


class PathsConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    raw_data_file: Path = Path("data/raw/telco_churn.csv")
    processed_data_dir: Path = Path("data/processed")
    train_data_file: Path = Path("data/processed/train.csv")
    test_data_file: Path = Path("data/processed/test.csv")
    model_dir: Path = Path("models")
    model_pipeline_file: Path = Path("models/churn_pipeline.joblib")
    metrics_file: Path = Path("models/metrics.json")
    feature_importance_file: Path = Path("models/feature_importance.json")
    drift_baseline_file: Path = Path("models/drift_baseline.json")
    logs_dir: Path = Path("logs")


class DataConfig(BaseModel):
    source_url: str = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    test_size: float = 0.2
    target_column: str = "Churn"
    id_column: str = "customerID"
    numeric_features: list[str] = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
    ]
    categorical_features: list[str] = [
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


class BusinessEconomicsConfig(BaseModel):
    default_threshold: float = 0.5
    customer_lifetime_value: float = 400.0
    retention_cost: float = 50.0
    retention_success_rate: float = 0.40


class ModelCandidateConfig(BaseModel):
    enabled: bool = True
    params: dict[str, list[Any]] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    primary_metric: str = "roc_auc"
    cv_folds: int = 5
    candidates: dict[str, ModelCandidateConfig] = Field(default_factory=dict)


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    business_economics: BusinessEconomicsConfig = Field(default_factory=BusinessEconomicsConfig)


def find_project_root() -> Path:
    """Find the root directory of the project."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "config" / "config.yaml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Optional path to config YAML. If None, default config/config.yaml is used.

    Returns:
        AppConfig instance with resolved paths.
    """
    root = find_project_root()

    if config_path is None:
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            config_path = root / config_path

    if not config_path.exists():
        # Fallback to defaults
        cfg = AppConfig()
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}
        cfg = AppConfig(**raw_data)

    # Resolve paths relative to project root
    paths = cfg.paths
    for attr in [
        "raw_data_dir",
        "raw_data_file",
        "processed_data_dir",
        "train_data_file",
        "test_data_file",
        "model_dir",
        "model_pipeline_file",
        "metrics_file",
        "feature_importance_file",
        "drift_baseline_file",
        "logs_dir",
    ]:
        val: Path = getattr(paths, attr)
        if not val.is_absolute():
            setattr(paths, attr, (root / val).resolve())

    return cfg
