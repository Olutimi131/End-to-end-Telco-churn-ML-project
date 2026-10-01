"""FastAPI application for Telco Customer Churn Prediction and Explainability."""

from pathlib import Path
from typing import Any
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.telco_churn.config import load_config
from src.telco_churn.models.predict import TelcoChurnPredictor
from src.telco_churn.monitoring.drift import evaluate_dataset_drift
from src.telco_churn.utils.io import load_json
from src.telco_churn.utils.logger import setup_logger

from .schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerInput,
    HealthResponse,
    PredictionResponse,
)

logger = setup_logger("telco_api")
cfg = load_config()

app = FastAPI(
    title="Telco Customer Churn Intelligence API",
    description="Production-grade ML API for predicting churn probability, identifying risk factors, and recommending retention actions.",
    version=cfg.project.version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web dashboards and cross-origin clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
_predictor: TelcoChurnPredictor | None = None


def get_predictor() -> TelcoChurnPredictor:
    """Lazy-load the predictor singleton."""
    global _predictor
    if _predictor is None:
        try:
            _predictor = TelcoChurnPredictor(config=cfg)
        except Exception as e:
            logger.error(f"Failed to load model pipeline: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not available: {str(e)}",
            )
    return _predictor


@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Root endpoint with basic service info and documentation links."""
    return {
        "service": "Telco Customer Churn Prediction API",
        "version": cfg.project.version,
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health_check() -> HealthResponse:
    """System health check and model status endpoint."""
    pipeline_exists = cfg.paths.model_pipeline_file.exists()
    model_name = None
    roc_auc = None
    optimal_th = None

    if cfg.paths.metrics_file.exists():
        try:
            metrics_data = load_json(cfg.paths.metrics_file)
            model_name = metrics_data.get("model_name")
            roc_auc = metrics_data.get("metrics", {}).get("roc_auc")
            optimal_th = metrics_data.get("optimal_threshold_metrics", {}).get("optimal_threshold")
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if pipeline_exists else "degraded",
        service="telco-churn-api",
        version=cfg.project.version,
        model_loaded=pipeline_exists,
        model_name=model_name,
        roc_auc=roc_auc,
        optimal_threshold=optimal_th,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single_customer(
    customer: CustomerInput,
    threshold: float | None = None,
) -> PredictionResponse:
    """Predict churn probability and generate retention recommendations for a single customer."""
    predictor = get_predictor()
    customer_dict = customer.model_dump()
    customer_id = customer_dict.get("customerID")

    result = predictor.predict_single(customer_dict, threshold=threshold)
    result["customer_id"] = customer_id

    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch_customers(
    batch: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Score a batch of customer records and return risk classifications."""
    predictor = get_predictor()
    records = [cust.model_dump() for cust in batch.customers]
    df = pd.DataFrame(records)

    scored_df = predictor.predict_batch(df, threshold=batch.threshold)

    predictions = []
    for idx, row in scored_df.iterrows():
        customer_record = records[idx]
        single_result = predictor.predict_single(customer_record, threshold=batch.threshold)
        single_result["customer_id"] = customer_record.get("customerID")
        predictions.append(PredictionResponse(**single_result))

    total = len(predictions)
    high_count = sum(1 for p in predictions if p.risk_level == "High")
    med_count = sum(1 for p in predictions if p.risk_level == "Medium")
    low_count = sum(1 for p in predictions if p.risk_level == "Low")
    churn_count = sum(1 for p in predictions if p.churn_prediction)
    churn_rate = round((churn_count / total) * 100, 2) if total > 0 else 0.0

    return BatchPredictionResponse(
        total_customers=total,
        high_risk_count=high_count,
        medium_risk_count=med_count,
        low_risk_count=low_count,
        churn_rate_predicted_pct=churn_rate,
        predictions=predictions,
    )


@app.get("/metrics", tags=["Analytics"])
def get_metrics() -> dict[str, Any]:
    """Retrieve test set evaluation metrics and optimal threshold profit curve."""
    if not cfg.paths.metrics_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not found. Run training pipeline first.",
        )
    return load_json(cfg.paths.metrics_file)


@app.get("/feature-importance", tags=["Analytics"])
def get_feature_importance(top_n: int = 20) -> list[dict[str, Any]]:
    """Retrieve top global feature importances from the trained model."""
    if not cfg.paths.feature_importance_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance data not found. Run training pipeline first.",
        )
    all_importances = load_json(cfg.paths.feature_importance_file)
    return all_importances[:top_n]


@app.post("/drift", tags=["Monitoring"])
def check_drift(batch: list[CustomerInput]) -> dict[str, Any]:
    """Evaluate feature drift between the submitted customer batch and training baseline."""
    if not cfg.paths.drift_baseline_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Baseline distribution data not found. Run training pipeline first.",
        )

    baseline = load_json(cfg.paths.drift_baseline_file)
    records = [c.model_dump() for c in batch]
    df = pd.DataFrame(records)

    drift_report = evaluate_dataset_drift(df, baseline, cfg)
    return drift_report
