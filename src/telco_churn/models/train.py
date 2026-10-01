"""Model training, model comparison, cross-validation, and pipeline serialization."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ..config import AppConfig, load_config
from ..data.dataset import load_raw_data, split_data
from ..data.preprocessing import build_preprocessor, clean_telco_data
from ..explainability.feature_importance import extract_global_feature_importance
from ..monitoring.drift import compute_baseline_statistics
from ..utils.io import save_joblib, save_json
from ..utils.logger import setup_logger
from .evaluate import evaluate_model

logger = setup_logger(__name__)


def build_candidate_models(seed: int = 42) -> dict[str, Any]:
    """Instantiate candidate classifiers."""
    return {
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
        "xgboost": XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            scale_pos_weight=2.7,  # Adjusts for ~26.5% churn rate in Telco dataset
            random_state=seed,
            eval_metric="logloss",
        ),
    }


def train_and_select_best_model(config: AppConfig | None = None) -> dict[str, Any]:
    """Train candidate models, select best performer by cross-validation, and persist pipeline."""
    cfg = config or load_config()
    seed = cfg.project.seed

    # 1. Load data
    train_file = cfg.paths.train_data_file
    test_file = cfg.paths.test_data_file

    if not train_file.exists() or not test_file.exists():
        logger.info("Processed train/test files not found. Ingesting and splitting raw data...")
        df_raw = load_raw_data(cfg)
        train_df, test_df = split_data(df_raw, cfg, save=True)
    else:
        logger.info(f"Loading cached splits from {train_file} and {test_file}")
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)

    # 2. Clean data
    train_cleaned = clean_telco_data(train_df, is_training=True)
    test_cleaned = clean_telco_data(test_df, is_training=True)

    target_col = cfg.data.target_column
    X_train = train_cleaned.drop(columns=[target_col])
    y_train = train_cleaned[target_col].values
    X_test = test_cleaned.drop(columns=[target_col])
    y_test = test_cleaned[target_col].values

    logger.info(f"Training features shape: {X_train.shape}, positive class ratio: {y_train.mean():.3f}")

    # 3. Model comparison via cross-validation
    candidates = build_candidate_models(seed=seed)
    cv = StratifiedKFold(n_splits=cfg.models.cv_folds, shuffle=True, random_state=seed)

    best_model_name = None
    best_cv_score = -float("inf")
    cv_comparison = {}

    preprocessor = build_preprocessor(cfg)
    X_train_transformed = preprocessor.fit_transform(X_train)

    for name, clf in candidates.items():
        logger.info(f"Cross-validating {name}...")
        scores = cross_val_score(
            clf,
            X_train_transformed,
            y_train,
            cv=cv,
            scoring=cfg.models.primary_metric,
            n_jobs=-1,
        )
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))
        cv_comparison[name] = {
            "mean_roc_auc": round(mean_score, 4),
            "std_roc_auc": round(std_score, 4),
            "all_scores": [round(float(s), 4) for s in scores],
        }
        logger.info(f"[{name}] {cfg.models.primary_metric.upper()}: {mean_score:.4f} (+/- {std_score:.4f})")

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_model_name = name

    logger.info(f"Best performing model selected: '{best_model_name}' (CV Score: {best_cv_score:.4f})")

    # 4. Fit final complete pipeline on full training set
    best_estimator = candidates[best_model_name]
    final_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(cfg)),
            ("classifier", best_estimator),
        ]
    )

    logger.info("Fitting final end-to-end pipeline on full training dataset...")
    final_pipeline.fit(X_train, y_train)

    # 5. Evaluate on held-out test set
    logger.info("Evaluating final pipeline on test set...")
    eval_report = evaluate_model(final_pipeline, X_test, y_test, cfg)
    eval_report["model_name"] = best_model_name
    eval_report["cv_comparison"] = cv_comparison

    # 6. Extract global feature importances
    feature_importances = extract_global_feature_importance(final_pipeline, top_n=30)

    # 7. Compute baseline dataset statistics for drift monitoring
    drift_baseline = compute_baseline_statistics(train_df, cfg)

    # 8. Persist artifacts
    save_joblib(final_pipeline, cfg.paths.model_pipeline_file)
    save_json(eval_report, cfg.paths.metrics_file)
    save_json(feature_importances, cfg.paths.feature_importance_file)
    save_json(drift_baseline, cfg.paths.drift_baseline_file)

    logger.info(f"Artifacts successfully serialized to {cfg.paths.model_dir}")
    return {
        "best_model": best_model_name,
        "cv_score": best_cv_score,
        "evaluation": eval_report,
        "feature_importances": feature_importances,
    }
