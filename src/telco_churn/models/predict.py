"""Inference engine for single customer and batch Telco churn scoring."""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import AppConfig, load_config
from ..explainability.feature_importance import explain_customer_risk_factors
from ..utils.io import load_joblib, load_json
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class TelcoChurnPredictor:
    """Production inference service for Telco Churn predictions."""

    def __init__(
        self,
        model_pipeline: Pipeline | None = None,
        config: AppConfig | None = None,
    ):
        self.config = config or load_config()
        self.pipeline = model_pipeline or self._load_pipeline()
        self.metrics = self._load_metrics()
        self.optimal_threshold = self.metrics.get(
            "optimal_threshold_metrics", {}
        ).get("optimal_threshold", self.config.business_economics.default_threshold)

    def _load_pipeline(self) -> Pipeline:
        pipeline_path = self.config.paths.model_pipeline_file
        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Model pipeline not found at {pipeline_path}. Please run training pipeline first."
            )
        logger.info(f"Loading model pipeline from {pipeline_path}")
        return load_joblib(pipeline_path)

    def _load_metrics(self) -> dict[str, Any]:
        metrics_path = self.config.paths.metrics_file
        if metrics_path.exists():
            try:
                return load_json(metrics_path)
            except Exception:
                pass
        return {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict churn probability (class 1) for input features."""
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(
        self,
        X: pd.DataFrame,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Predict binary churn outcome based on specified or optimal threshold."""
        th = threshold if threshold is not None else self.optimal_threshold
        probabilities = self.predict_proba(X)
        return (probabilities >= th).astype(int)

    def predict_single(
        self,
        customer_data: dict[str, Any],
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """Generate prediction, probability, risk level, and retention advice for a single customer."""
        th = threshold if threshold is not None else self.optimal_threshold
        df = pd.DataFrame([customer_data])
        
        prob = float(self.predict_proba(df)[0])
        will_churn = bool(prob >= th)

        explanation = explain_customer_risk_factors(
            customer=customer_data,
            churn_probability=prob,
        )

        return {
            "churn_probability": round(prob, 4),
            "churn_prediction": will_churn,
            "decision_threshold": round(th, 2),
            "risk_level": explanation["risk_level"],
            "recommended_action": explanation["recommended_action"],
            "risk_drivers": explanation["risk_drivers"],
            "retention_recommendations": explanation["retention_recommendations"],
        }

    def predict_batch(
        self,
        df: pd.DataFrame,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        """Score a batch dataframe and append prediction columns."""
        th = threshold if threshold is not None else self.optimal_threshold
        result_df = df.copy()

        # Isolate scoring features
        features = df.copy()
        if "customerID" in features.columns:
            features = features.drop(columns=["customerID"])
        if "Churn" in features.columns:
            features = features.drop(columns=["Churn"])

        probs = self.predict_proba(features)
        preds = (probs >= th).astype(int)

        result_df["churn_probability"] = np.round(probs, 4)
        result_df["churn_prediction"] = ["Yes" if p == 1 else "No" for p in preds]

        # Risk tiering
        conditions = [
            result_df["churn_probability"] < 0.30,
            (result_df["churn_probability"] >= 0.30) & (result_df["churn_probability"] < 0.60),
            result_df["churn_probability"] >= 0.60,
        ]
        choices = ["Low", "Medium", "High"]
        result_df["risk_tier"] = np.select(conditions, choices, default="Medium")

        return result_df
