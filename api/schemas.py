"""Pydantic schemas for Telco Churn REST API requests and responses."""

from typing import Any, Literal
from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    """Schema for a single customer churn prediction request."""

    customerID: str | None = Field(default="CUST-001", description="Optional unique customer identifier")
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether customer is a senior citizen (1: Yes, 0: No)")
    Partner: Literal["Yes", "No"] = Field(..., description="Whether customer has a partner")
    Dependents: Literal["Yes", "No"] = Field(..., description="Whether customer has dependents")
    tenure: int = Field(..., ge=0, le=120, description="Months customer has stayed with company")
    PhoneService: Literal["Yes", "No"] = Field(..., description="Whether customer has phone service")
    MultipleLines: Literal["No phone service", "No", "Yes"] = Field(..., description="Multiple lines status")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(..., description="Internet service provider")
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(..., description="Online security add-on")
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(..., description="Online backup add-on")
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(..., description="Device protection add-on")
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(..., description="Tech support add-on")
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming TV add-on")
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(..., description="Streaming movies add-on")
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(..., description="Contract term")
    PaperlessBilling: Literal["Yes", "No"] = Field(..., description="Paperless billing status")
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Payment method")
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly charges amount")
    TotalCharges: float | str = Field(..., description="Total cumulative charges amount")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Schema for prediction output of a customer."""

    customer_id: str | None = Field(default=None, description="Customer ID")
    churn_probability: float = Field(..., description="Predicted probability of churn (0.0 to 1.0)")
    churn_prediction: bool = Field(..., description="Boolean churn outcome based on threshold")
    decision_threshold: float = Field(..., description="Threshold applied for binary classification")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk category band")
    recommended_action: str = Field(..., description="High-level retention action recommendation")
    risk_drivers: list[str] = Field(..., description="Top contributing factors to churn risk")
    retention_recommendations: list[str] = Field(..., description="Actionable retention steps")


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction requests."""

    customers: list[CustomerInput] = Field(..., description="List of customer records to score")
    threshold: float | None = Field(default=None, description="Optional custom classification threshold")


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction responses."""

    total_customers: int = Field(..., description="Total number of evaluated customers")
    high_risk_count: int = Field(..., description="Number of customers flagged as high risk")
    medium_risk_count: int = Field(..., description="Number of customers flagged as medium risk")
    low_risk_count: int = Field(..., description="Number of customers flagged as low risk")
    churn_rate_predicted_pct: float = Field(..., description="Percentage predicted to churn")
    predictions: list[PredictionResponse] = Field(..., description="Individual customer predictions")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    service: str
    version: str
    model_loaded: bool
    model_name: str | None = None
    roc_auc: float | None = None
    optimal_threshold: float | None = None
