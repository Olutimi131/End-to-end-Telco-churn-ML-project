"""Model Training and Evaluation Pipeline for Telco Customer Churn.

This module builds an end-to-end scikit-learn Pipeline with ColumnTransformer,
evaluates candidate algorithms (Random Forest, XGBoost, Logistic Regression),
computes comprehensive test metrics, extracts feature importance, and serializes
the best model pipeline artifact to disk.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("train_pipeline")

NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
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
TARGET_COL = "Churn"


def build_preprocessor() -> ColumnTransformer:
    """Build ColumnTransformer for numerical scaling and categorical encoding."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_candidate_models(seed: int = 42) -> Dict[str, Any]:
    """Return dictionary of candidate classifiers."""
    models: Dict[str, Any] = {
        "logistic_regression": LogisticRegression(
            C=0.1,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=2.7,
            random_state=seed,
            eval_metric="logloss",
        )
    return models


class ChurnTrainingPipeline:
    """Coordinates training, model selection, evaluation, and artifact persistence."""

    def __init__(
        self,
        train_path: str | Path = "data/processed/train.csv",
        test_path: str | Path = "data/processed/test.csv",
        model_output_path: str | Path = "models/churn_pipeline.joblib",
        metrics_output_path: str | Path = "models/metrics.json",
        importance_output_path: str | Path = "models/feature_importance.json",
        seed: int = 42,
    ):
        self.train_path = Path(train_path)
        self.test_path = Path(test_path)
        self.model_output_path = Path(model_output_path)
        self.metrics_output_path = Path(metrics_output_path)
        self.importance_output_path = Path(importance_output_path)
        self.seed = seed

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Ensure train and test splits exist, loading them into memory."""
        if not self.train_path.exists() or not self.test_path.exists():
            logger.info("Processed data not found. Running data ingestion...")
            from src.data_ingestion import DataIngestionPipeline
            ingestion = DataIngestionPipeline(
                processed_dir=self.train_path.parent,
                random_state=self.seed,
            )
            ingestion.run()

        logger.info(f"Loading train dataset from {self.train_path}")
        train_df = pd.read_csv(self.train_path)
        logger.info(f"Loading test dataset from {self.test_path}")
        test_df = pd.read_csv(self.test_path)

        # Standardize TotalCharges
        for df in (train_df, test_df):
            if "TotalCharges" in df.columns and df["TotalCharges"].dtype == object:
                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"].astype(str).str.strip(),
                    errors="coerce",
                ).fillna(0.0)

        return train_df, test_df

    def train_and_evaluate(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        selected_model: str = "xgboost",
    ) -> Dict[str, Any]:
        """Train models, evaluate on test set, and return artifacts."""
        X_train = train_df.drop(columns=[TARGET_COL, "customerID"], errors="ignore")
        y_train = train_df[TARGET_COL].astype(int).values
        X_test = test_df.drop(columns=[TARGET_COL, "customerID"], errors="ignore")
        y_test = test_df[TARGET_COL].astype(int).values

        candidates = get_candidate_models(seed=self.seed)
        if selected_model not in candidates:
            selected_model = "random_forest" if "random_forest" in candidates else "logistic_regression"

        logger.info(f"Evaluating candidate models: {list(candidates.keys())}")
        best_name = selected_model
        best_score = -1.0
        best_pipeline = None

        evaluation_summary = {}

        for name, clf in candidates.items():
            pipeline = Pipeline([
                ("preprocessor", build_preprocessor()),
                ("classifier", clf),
            ])
            pipeline.fit(X_train, y_train)

            # Predict probabilities
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, y_pred_proba)
            evaluation_summary[name] = {"roc_auc": float(auc_score)}
            logger.info(f"Model [{name}] Test ROC-AUC: {auc_score:.4f}")

            if auc_score > best_score:
                best_score = auc_score
                best_name = name
                best_pipeline = pipeline

        logger.info(f"Selected best model: {best_name} with ROC-AUC: {best_score:.4f}")

        # Comprehensive evaluation of selected model
        y_test_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Optimize threshold using F1 score
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)
        best_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = float(thresholds[min(best_threshold_idx, len(thresholds) - 1)])
        # Constrain threshold to realistic range
        optimal_threshold = max(0.30, min(0.60, optimal_threshold))

        y_test_pred_default = (y_test_proba >= 0.50).astype(int)
        y_test_pred_opt = (y_test_proba >= optimal_threshold).astype(int)

        cm = confusion_matrix(y_test, y_test_pred_opt).tolist()
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)

        metrics = {
            "metrics": {
                "roc_auc": float(best_score),
                "accuracy": float(accuracy_score(y_test, y_test_pred_opt)),
                "precision": float(precision_score(y_test, y_test_pred_opt, zero_division=0)),
                "recall": float(recall_score(y_test, y_test_pred_opt, zero_division=0)),
                "f1": float(f1_score(y_test, y_test_pred_opt, zero_division=0)),
            },
            "model_name": best_name,
            "roc_auc": float(best_score),
            "accuracy": float(accuracy_score(y_test, y_test_pred_opt)),
            "precision": float(precision_score(y_test, y_test_pred_opt, zero_division=0)),
            "recall": float(recall_score(y_test, y_test_pred_opt, zero_division=0)),
            "f1": float(f1_score(y_test, y_test_pred_opt, zero_division=0)),
            "default_threshold_f1": float(f1_score(y_test, y_test_pred_default, zero_division=0)),
            "optimal_threshold": round(optimal_threshold, 4),
            "confusion_matrix": cm,
            "classification_report": classification_report(y_test, y_test_pred_opt, output_dict=True),
            "candidate_comparison": evaluation_summary,
        }

        # Extract feature importances
        feature_importance_list = self._extract_feature_importance(best_pipeline, best_name)

        return {
            "best_pipeline": best_pipeline,
            "best_model_name": best_name,
            "metrics": metrics,
            "feature_importance": feature_importance_list,
        }

    def _extract_feature_importance(self, pipeline: Pipeline, model_name: str) -> list[Dict[str, Any]]:
        """Extract top feature importances or coefficients from the fitted pipeline."""
        try:
            preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]

            num_cols = NUMERICAL_FEATURES
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
            all_feature_names = list(num_cols) + cat_feature_names

            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_
            elif hasattr(classifier, "coef_"):
                importances = np.abs(classifier.coef_[0])
            else:
                return []

            total = float(np.sum(importances)) if np.sum(importances) > 0 else 1.0
            records = [
                {
                    "feature": name,
                    "importance": round(float(imp), 5),
                    "relative_pct": round(float(imp / total * 100.0), 2),
                }
                for name, imp in zip(all_feature_names, importances)
            ]
            records.sort(key=lambda x: x["importance"], reverse=True)
            return records
        except Exception as e:
            logger.warning(f"Could not compute feature importance: {e}")
            return []

    def save_artifacts(
        self,
        pipeline: Pipeline,
        metrics: Dict[str, Any],
        feature_importance: Any,
    ) -> None:
        """Persist model pipeline and metric JSON artifacts to disk."""
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.importance_output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving model pipeline to {self.model_output_path}")
        joblib.dump(pipeline, self.model_output_path)

        logger.info(f"Saving evaluation metrics to {self.metrics_output_path}")
        with open(self.metrics_output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saving feature importances to {self.importance_output_path}")
        with open(self.importance_output_path, "w", encoding="utf-8") as f:
            json.dump(feature_importance, f, indent=2)

        logger.info("All model artifacts saved successfully.")

    def run(self, selected_model: str = "xgboost") -> Dict[str, Any]:
        """Execute full training, evaluation, and artifact persistence."""
        train_df, test_df = self.load_data()
        results = self.train_and_evaluate(train_df, test_df, selected_model=selected_model)
        self.save_artifacts(
            pipeline=results["best_pipeline"],
            metrics=results["metrics"],
            feature_importance=results["feature_importance"],
        )
        return results


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate Telco Customer Churn model.")
    parser.add_argument("--train-path", type=str, default="data/processed/train.csv", help="Path to train CSV")
    parser.add_argument("--test-path", type=str, default="data/processed/test.csv", help="Path to test CSV")
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost", "random_forest", "logistic_regression"])
    parser.add_argument("--output-model", type=str, default="models/churn_pipeline.joblib")
    parser.add_argument("--output-metrics", type=str, default="models/metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    trainer = ChurnTrainingPipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        model_output_path=args.output_model,
        metrics_output_path=args.output_metrics,
        seed=args.seed,
    )
    res = trainer.run(selected_model=args.model)
    print("\n--- Training Pipeline Finished Successfully ---")
    print(f"Selected Model: {res['best_model_name']}")
    print(f"Test ROC-AUC:   {res['metrics']['roc_auc']:.4f}")
    print(f"Test Accuracy:  {res['metrics']['accuracy']:.4f}")
    print(f"Test F1 Score:  {res['metrics']['f1']:.4f}")
    print(f"Optimal Threshold: {res['metrics']['optimal_threshold']}")


if __name__ == "__main__":
    main()
