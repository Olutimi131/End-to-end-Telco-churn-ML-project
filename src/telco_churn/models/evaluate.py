"""Model evaluation metrics, ROC/PR curves, and business threshold optimization."""

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..config import AppConfig, load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_confusion_matrix_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Compute true negatives, false positives, false negatives, true positives."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def optimize_decision_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    clv: float = 400.0,
    retention_cost: float = 50.0,
    success_rate: float = 0.40,
) -> dict[str, Any]:
    """Find the optimal classification threshold maximizing expected business profit.

    Business logic:
    - True Positive (TP): We intervene. Cost = retention_cost. If customer accepts (success_rate),
      we retain CLV value. Net gain = success_rate * CLV - retention_cost.
    - False Positive (FP): We needlessly intervene on a customer who wouldn't churn. Cost = retention_cost.
    - False Negative (FN): We failed to intervene on a churner. Lost opportunity = 0 (or lost CLV).
    - True Negative (TN): Correctly ignored non-churner. Net impact = 0.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    results = []

    best_profit = -float("inf")
    best_threshold = 0.5

    for th in thresholds:
        preds = (y_proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        net_profit = tp * (clv * success_rate - retention_cost) - fp * retention_cost
        f1 = f1_score(y_true, preds, zero_division=0)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)

        results.append(
            {
                "threshold": round(float(th), 2),
                "net_profit": round(float(net_profit), 2),
                "f1": round(float(f1), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            }
        )

        if net_profit > best_profit:
            best_profit = net_profit
            best_threshold = round(float(th), 2)

    return {
        "optimal_threshold": best_threshold,
        "max_net_profit": round(best_profit, 2),
        "threshold_curve": results,
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Compute comprehensive evaluation metrics on a test set."""
    cfg = config or load_config()
    y_true = np.array(y_test)

    # Predict probabilities and standard classes
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_default = (y_proba >= cfg.business_economics.default_threshold).astype(int)

    # Core classification metrics
    accuracy = float(accuracy_score(y_true, y_pred_default))
    precision = float(precision_score(y_true, y_pred_default, zero_division=0))
    recall = float(recall_score(y_true, y_pred_default, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_default, zero_division=0))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    pr_auc = float(average_precision_score(y_true, y_proba))

    cm_breakdown = compute_confusion_matrix_breakdown(y_true, y_pred_default)

    # ROC curve points (subsampled for compact JSON payload)
    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    step = max(1, len(fpr) // 50)
    roc_curve_data = [
        {"fpr": round(float(f), 4), "tpr": round(float(t), 4), "threshold": round(float(th), 4)}
        for f, t, th in zip(fpr[::step], tpr[::step], roc_thresh[::step])
    ]

    # Precision-Recall curve
    precisions, recalls, pr_thresh = precision_recall_curve(y_true, y_proba)
    pr_step = max(1, len(precisions) // 50)
    pr_curve_data = [
        {"precision": round(float(p), 4), "recall": round(float(r), 4)}
        for p, r in zip(precisions[::pr_step], recalls[::pr_step])
    ]

    # Business threshold optimization
    threshold_opt = optimize_decision_threshold(
        y_true=y_true,
        y_proba=y_proba,
        clv=cfg.business_economics.customer_lifetime_value,
        retention_cost=cfg.business_economics.retention_cost,
        success_rate=cfg.business_economics.retention_success_rate,
    )

    opt_th = threshold_opt["optimal_threshold"]
    y_pred_opt = (y_proba >= opt_th).astype(int)
    cm_opt = compute_confusion_matrix_breakdown(y_true, y_pred_opt)

    evaluation_report = {
        "metrics": {
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "default_threshold": cfg.business_economics.default_threshold,
        "confusion_matrix": cm_breakdown,
        "optimal_threshold_metrics": {
            "optimal_threshold": opt_th,
            "max_net_profit": threshold_opt["max_net_profit"],
            "f1_at_optimal": round(float(f1_score(y_true, y_pred_opt, zero_division=0)), 4),
            "precision_at_optimal": round(float(precision_score(y_true, y_pred_opt, zero_division=0)), 4),
            "recall_at_optimal": round(float(recall_score(y_true, y_pred_opt, zero_division=0)), 4),
            "confusion_matrix": cm_opt,
        },
        "roc_curve": roc_curve_data,
        "pr_curve": pr_curve_data,
        "threshold_profit_curve": threshold_opt["threshold_curve"],
        "classification_report": classification_report(y_true, y_pred_default, output_dict=True),
    }

    logger.info(
        f"Evaluation: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
        f"F1={f1:.4f}, Optimal Threshold={opt_th} (Profit=${threshold_opt['max_net_profit']})"
    )

    return evaluation_report
