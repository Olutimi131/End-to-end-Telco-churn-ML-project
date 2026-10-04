"""Production FastAPI Backend for Telco Customer Churn Prediction.

Provides RESTful endpoints for single and batch predictions, health checks,
model metrics, and retention recommendation generation.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("telco_churn_api")

MODEL_PATH = Path("models/churn_pipeline.joblib")
METRICS_PATH = Path("models/metrics.json")
IMPORTANCE_PATH = Path("models/feature_importance.json")

# ==============================================================================
# PYDANTIC SCHEMAS
# ==============================================================================

class CustomerInput(BaseModel):
    """Input payload for a single customer churn prediction."""

    customerID: Optional[str] = Field(default="CUST-001", description="Customer ID")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Senior citizen indicator (1: Yes, 0: No)")
    Partner: Literal["Yes", "No"] = Field(..., description="Has partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Has phone service")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple phone lines")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet service type")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="Online security")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Online backup")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Device protection")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Tech support")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming TV")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming movies")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges amount ($)")
    TotalCharges: float | str = Field(..., description="Total cumulative charges ($)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 6,
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
                "TotalCharges": 479.10,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction output details for a customer."""

    customer_id: Optional[str] = Field(default=None, description="Customer identifier")
    churn_prediction: bool = Field(..., description="True if customer is predicted to churn")
    churn_probability: float = Field(..., description="Predicted churn probability between 0.0 and 1.0")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Customer risk category")
    decision_threshold: float = Field(..., description="Threshold used for churn classification")
    recommended_action: str = Field(..., description="Primary business action recommendation")
    risk_drivers: List[str] = Field(..., description="Key factors contributing to churn risk")
    retention_recommendations: List[str] = Field(..., description="Specific retention strategies")


class BatchPredictionRequest(BaseModel):
    """Payload for multi-customer batch scoring."""

    customers: List[CustomerInput] = Field(..., description="List of customer records")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Custom decision threshold")


class BatchPredictionResponse(BaseModel):
    """Summary and details of batch prediction."""

    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    churn_rate_predicted_pct: float
    decision_threshold: float
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check payload."""

    status: str
    service: str
    version: str
    model_loaded: bool
    model_name: Optional[str] = None
    optimal_threshold: Optional[float] = None


# ==============================================================================
# MODEL SERVICE & LOGIC
# ==============================================================================

class ChurnModelService:
    """Singleton service for loading model pipeline and running inferences."""

    def __init__(self):
        self.model = None
        self.metrics = {}
        self.feature_importance = {}
        self.optimal_threshold = 0.50
        self.model_name = "Unknown"
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            try:
                logger.info(f"Loading pipeline from {MODEL_PATH}")
                self.model = joblib.load(MODEL_PATH)
            except Exception as e:
                logger.error(f"Error loading model from {MODEL_PATH}: {e}")
                self.model = None

        if METRICS_PATH.exists():
            try:
                with open(METRICS_PATH, "r", encoding="utf-8") as f:
                    self.metrics = json.load(f)
                    self.optimal_threshold = self.metrics.get("optimal_threshold", 0.50)
                    self.model_name = self.metrics.get("model_name", "xgboost")
            except Exception as e:
                logger.warning(f"Error reading metrics from {METRICS_PATH}: {e}")

        if IMPORTANCE_PATH.exists():
            try:
                with open(IMPORTANCE_PATH, "r", encoding="utf-8") as f:
                    self.feature_importance = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading feature importance: {e}")

    def is_ready(self) -> bool:
        return self.model is not None

    def explain_and_recommend(
        self,
        customer: Dict[str, Any],
        prob: float,
        threshold: float,
    ) -> Tuple[Literal["Low", "Medium", "High"], str, List[str], List[str]]:
        """Compute risk level, explain risk factors, and recommend tailored retention actions."""
        # Risk band
        if prob >= 0.60:
            risk = "High"
            action = "Urgent: Immediate Customer Success Outreach & Incentive Offer"
        elif prob >= 0.35:
            risk = "Medium"
            action = "Caution: Proactive Engagement & Service Review"
        else:
            risk = "Low"
            action = "Healthy: Maintain Relationship & Regular Service Quality"

        drivers: List[str] = []
        recommendations: List[str] = []

        # Analyze risk factors
        if customer.get("Contract") == "Month-to-month":
            drivers.append("Month-to-month contract (flexible cancellation)")
            recommendations.append("Offer a 15% discount for upgrading to a 1-year or 2-year contract commitment.")

        if customer.get("PaymentMethod") == "Electronic check":
            drivers.append("Electronic check payment (historically highest churn frequency)")
            recommendations.append("Incentivize automated payment methods (Bank Transfer or Credit Card) with a $5 recurring credit.")

        if customer.get("InternetService") == "Fiber optic":
            if customer.get("TechSupport") == "No":
                drivers.append("Fiber optic subscription without TechSupport service")
                recommendations.append("Bundle complimentary 3-month premium TechSupport and DeviceProtection.")
            if customer.get("OnlineSecurity") == "No":
                drivers.append("Fiber optic subscription lacking OnlineSecurity")
                recommendations.append("Activate free trial of cybersecurity & malware protection package.")

        tenure = customer.get("tenure", 0)
        if tenure <= 6:
            drivers.append(f"Early customer lifecycle risk (tenure: {tenure} months)")
            recommendations.append("Schedule proactive onboarding check-in call with dedicated account specialist.")
        elif tenure > 48:
            recommendations.append("Enroll in VIP Loyalty Program with milestone tenure perks.")

        monthly = customer.get("MonthlyCharges", 0.0)
        if monthly > 80.0:
            drivers.append(f"High monthly billing rate (${monthly:.2f}/mo)")
            recommendations.append("Perform plan audit to eliminate unused add-ons or align with tailored budget bundle.")

        if not drivers:
            drivers.append("Stable tenure and standard contract configuration")

        if not recommendations:
            recommendations.append("Continue regular standard engagement and monitor usage patterns.")

        return risk, action, drivers, recommendations

    def predict_single(
        self,
        customer_data: Dict[str, Any],
        custom_threshold: Optional[float] = None,
    ) -> PredictionResponse:
        """Run single prediction on dictionary of customer values."""
        if not self.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model pipeline is not loaded. Train the model via 'python src/train.py' first.",
            )

        threshold = custom_threshold if custom_threshold is not None else self.optimal_threshold
        df = pd.DataFrame([customer_data])

        # Clean TotalCharges if needed
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip(),
                errors="coerce",
            ).fillna(0.0)

        cust_id = customer_data.get("customerID")
        features_df = df.drop(columns=["customerID", "Churn"], errors="ignore")

        prob = float(self.model.predict_proba(features_df)[0, 1])
        prediction = bool(prob >= threshold)
        risk, action, drivers, recs = self.explain_and_recommend(customer_data, prob, threshold)

        return PredictionResponse(
            customer_id=cust_id,
            churn_prediction=prediction,
            churn_probability=round(prob, 4),
            risk_level=risk,
            decision_threshold=round(threshold, 4),
            recommended_action=action,
            risk_drivers=drivers,
            retention_recommendations=recs,
        )

    def predict_batch(
        self,
        customers: List[Dict[str, Any]],
        custom_threshold: Optional[float] = None,
    ) -> BatchPredictionResponse:
        """Score multiple customers efficiently."""
        if not self.is_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model pipeline is not loaded.",
            )

        threshold = custom_threshold if custom_threshold is not None else self.optimal_threshold
        df = pd.DataFrame(customers)

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(
                df["TotalCharges"].astype(str).str.strip(),
                errors="coerce",
            ).fillna(0.0)

        features_df = df.drop(columns=["customerID", "Churn"], errors="ignore")
        probs = self.model.predict_proba(features_df)[:, 1]

        predictions: List[PredictionResponse] = []
        high_risk = 0
        med_risk = 0
        low_risk = 0

        for cust, prob in zip(customers, probs):
            prob_float = float(prob)
            prediction_bool = bool(prob_float >= threshold)
            risk, action, drivers, recs = self.explain_and_recommend(cust, prob_float, threshold)

            if risk == "High":
                high_risk += 1
            elif risk == "Medium":
                med_risk += 1
            else:
                low_risk += 1

            predictions.append(
                PredictionResponse(
                    customer_id=cust.get("customerID"),
                    churn_prediction=prediction_bool,
                    churn_probability=round(prob_float, 4),
                    risk_level=risk,
                    decision_threshold=round(threshold, 4),
                    recommended_action=action,
                    risk_drivers=drivers,
                    retention_recommendations=recs,
                )
            )

        total = len(predictions)
        churn_count = sum(1 for p in predictions if p.churn_prediction)
        churn_rate = (churn_count / total * 100.0) if total > 0 else 0.0

        return BatchPredictionResponse(
            total_customers=total,
            high_risk_count=high_risk,
            medium_risk_count=med_risk,
            low_risk_count=low_risk,
            churn_rate_predicted_pct=round(churn_rate, 2),
            decision_threshold=round(threshold, 4),
            predictions=predictions,
        )


# Initialize singleton model service
model_service = ChurnModelService()

# ==============================================================================
# FASTAPI APPLICATION SETUP
# ==============================================================================

app = FastAPI(
    title="Telco Customer Churn Intelligence API",
    description="Production-ready REST API for churn risk prediction, retention recommendations, and batch scoring.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["General"])
def root():
    """Root metadata endpoint."""
    return {
        "service": "Telco Customer Churn Intelligence API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "ready": model_service.is_ready(),
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Health check verifying model pipeline readiness."""
    return HealthResponse(
        status="healthy" if model_service.is_ready() else "degraded",
        service="telco-churn-api",
        version="1.0.0",
        model_loaded=model_service.is_ready(),
        model_name=model_service.model_name,
        optimal_threshold=model_service.optimal_threshold,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(customer: CustomerInput, threshold: Optional[float] = Query(None, ge=0.0, le=1.0)):
    """Predict churn probability and generate retention recommendations for a single customer."""
    return model_service.predict_single(customer.model_dump(), custom_threshold=threshold)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(payload: BatchPredictionRequest):
    """Score multiple customer records in batch mode."""
    cust_dicts = [c.model_dump() for c in payload.customers]
    return model_service.predict_batch(cust_dicts, custom_threshold=payload.threshold)


@app.post("/predict/csv", tags=["Inference"])
async def predict_from_csv(file: UploadFile = File(...), threshold: Optional[float] = Query(None)):
    """Upload CSV file of customers and return scored results with risk classifications."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV.",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to parse CSV file: {e}",
        )

    records = df.to_dict(orient="records")
    batch_res = model_service.predict_batch(records, custom_threshold=threshold)
    return batch_res


@app.get("/metrics", tags=["Model Analytics"])
def get_metrics():
    """Retrieve test set evaluation metrics of the serialized production model."""
    if not model_service.metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not found. Train the model first.",
        )
    return model_service.metrics


@app.get("/features", tags=["Model Analytics"])
def get_feature_importance():
    """Retrieve global feature importances extracted during training."""
    if not model_service.feature_importance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Feature importance artifact not found.",
        )
    return model_service.feature_importance


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
