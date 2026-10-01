"""Unit tests for Population Stability Index (PSI) and data drift monitoring."""

import numpy as np
import pandas as pd
import pytest

from src.telco_churn.config import load_config
from src.telco_churn.data.dataset import generate_synthetic_data
from src.telco_churn.monitoring.drift import (
    calculate_psi,
    compute_baseline_statistics,
    evaluate_dataset_drift,
)


def test_calculate_psi_identical_distributions():
    """Identical distributions should have PSI close to 0 (< 0.05)."""
    rng = np.random.default_rng(42)
    dist = rng.normal(50, 10, size=1000)
    psi = calculate_psi(dist, dist)
    assert psi < 0.05


def test_calculate_psi_shifted_distribution():
    """Significantly shifted distribution should produce high PSI (> 0.2)."""
    rng = np.random.default_rng(42)
    dist1 = rng.normal(30, 5, size=1000)
    dist2 = rng.normal(80, 5, size=1000)
    psi = calculate_psi(dist1, dist2)
    assert psi >= 0.2


def test_baseline_and_drift_evaluation():
    """Verify baseline extraction and evaluate_dataset_drift returns structured report."""
    cfg = load_config()
    baseline_df = generate_synthetic_data(n_samples=300, seed=1)
    baseline_stats = compute_baseline_statistics(baseline_df, cfg)

    assert "numeric" in baseline_stats
    assert "categorical" in baseline_stats
    assert baseline_stats["sample_count"] == 300

    # Test with similar data
    test_df = generate_synthetic_data(n_samples=200, seed=2)
    report = evaluate_dataset_drift(test_df, baseline_stats, cfg)

    assert "overall_status" in report
    assert "flagged_features" in report
    assert "feature_drifts" in report
    assert report["evaluated_records"] == 200
