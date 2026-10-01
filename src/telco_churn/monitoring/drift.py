"""Data drift detection and Population Stability Index (PSI) monitoring."""

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

from ..config import AppConfig, load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_psi(
    expected: np.ndarray | list[float],
    actual: np.ndarray | list[float],
    num_buckets: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between two numeric distributions.

    PSI < 0.1: No significant change / stable.
    0.1 <= PSI < 0.2: Moderate shift.
    PSI >= 0.2: Significant drift detected.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Filter out NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bins = np.percentile(expected, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = (expected_counts / len(expected)) + epsilon
    actual_pct = (actual_counts / len(actual)) + epsilon

    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))


def compute_baseline_statistics(df: pd.DataFrame, config: AppConfig | None = None) -> dict[str, Any]:
    """Compute and return reference baseline statistics for feature drift tracking."""
    cfg = config or load_config()
    baseline = {"numeric": {}, "categorical": {}, "sample_count": len(df)}

    # Numeric features
    for col in cfg.data.numeric_features:
        if col in df.columns:
            series = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce").dropna()
            baseline["numeric"][col] = {
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "p25": round(float(series.quantile(0.25)), 4),
                "p50": round(float(series.quantile(0.50)), 4),
                "p75": round(float(series.quantile(0.75)), 4),
                "values_sample": series.sample(min(500, len(series)), random_state=42).tolist(),
            }

    # Categorical features
    for col in cfg.data.categorical_features:
        if col in df.columns:
            val_counts = df[col].astype(str).value_counts(normalize=True).to_dict()
            baseline["categorical"][col] = {
                k: round(float(v), 4) for k, v in val_counts.items()
            }

    return baseline


def evaluate_dataset_drift(
    current_df: pd.DataFrame,
    baseline_stats: dict[str, Any],
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """Compare a new dataset batch against reference baseline statistics."""
    cfg = config or load_config()
    feature_drifts = {}
    flagged_features = []

    # 1. Numeric features using PSI and Kolmogorov-Smirnov test
    numeric_baseline = baseline_stats.get("numeric", {})
    for col, base_data in numeric_baseline.items():
        if col in current_df.columns:
            cur_series = pd.to_numeric(
                current_df[col].astype(str).str.strip(), errors="coerce"
            ).dropna()
            if len(cur_series) > 10:
                base_sample = base_data.get("values_sample", [])
                psi = calculate_psi(base_sample, cur_series.values)

                # Kolmogorov-Smirnov test
                ks_stat, p_val = stats.ks_2samp(base_sample, cur_series.values)

                status = "STABLE"
                if psi >= 0.2 or p_val < 0.01:
                    status = "DRIFT_DETECTED"
                    flagged_features.append(col)
                elif psi >= 0.1:
                    status = "MODERATE_SHIFT"

                feature_drifts[col] = {
                    "type": "numeric",
                    "psi": psi,
                    "ks_statistic": round(float(ks_stat), 4),
                    "p_value": round(float(p_val), 4),
                    "baseline_mean": base_data["mean"],
                    "current_mean": round(float(cur_series.mean()), 4),
                    "status": status,
                }

    # 2. Categorical features
    cat_baseline = baseline_stats.get("categorical", {})
    for col, base_dist in cat_baseline.items():
        if col in current_df.columns:
            cur_dist = (
                current_df[col].astype(str).value_counts(normalize=True).to_dict()
            )
            # Maximum absolute category frequency difference
            max_diff = 0.0
            all_keys = set(base_dist.keys()) | set(cur_dist.keys())
            for k in all_keys:
                diff = abs(base_dist.get(k, 0.0) - cur_dist.get(k, 0.0))
                if diff > max_diff:
                    max_diff = diff

            status = "STABLE"
            if max_diff >= 0.15:
                status = "DRIFT_DETECTED"
                flagged_features.append(col)
            elif max_diff >= 0.08:
                status = "MODERATE_SHIFT"

            feature_drifts[col] = {
                "type": "categorical",
                "max_category_diff": round(float(max_diff), 4),
                "baseline_distribution": base_dist,
                "current_distribution": {k: round(float(v), 4) for k, v in cur_dist.items()},
                "status": status,
            }

    overall_status = "HEALTHY"
    if len(flagged_features) >= 3:
        overall_status = "ACTION_REQUIRED_DRIFT_DETECTED"
    elif len(flagged_features) > 0:
        overall_status = "WARNING_MODERATE_DRIFT"

    return {
        "overall_status": overall_status,
        "flagged_features_count": len(flagged_features),
        "flagged_features": flagged_features,
        "feature_drifts": feature_drifts,
        "evaluated_records": len(current_df),
    }
