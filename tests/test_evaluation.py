"""
Unit tests for src.evaluation.compute_metrics.

The metric function is deceptively important: every headline number on the
Streamlit dashboard and every API /metrics response ultimately comes out
of it, so we pin down the exact behaviour for a few controlled inputs.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import boxcox

from src.evaluation import compute_metrics


def test_perfect_prediction_gives_zero_error_and_r2_one():
    raw = np.linspace(10.0, 80.0, 80)
    transformed, lam = boxcox(raw)

    m = compute_metrics(transformed, transformed.copy(), lambda_bc=lam, label="perfect")

    assert m["MAE"] == 0.0
    assert m["RMSE"] == 0.0
    assert m["SMAPE"] == 0.0
    assert m["R2"] == 1.0


def test_metric_keys_are_complete():
    raw = np.linspace(10.0, 80.0, 60)
    transformed, lam = boxcox(raw)
    preds = transformed + np.random.default_rng(0).normal(0, 0.05, size=60)

    m = compute_metrics(transformed, preds, lambda_bc=lam)

    expected = {"R2", "MAE", "RMSE", "SMAPE", "exc_precision", "exc_recall", "exc_f1"}
    assert expected.issubset(m.keys())


def test_exceedance_metrics_flag_threshold_crossings():
    """If ground truth has exceedances and predictions mirror them, F1 → 1.0."""
    raw_true = np.array([20.0, 30.0, 55.0, 70.0, 15.0, 60.0])
    raw_pred = raw_true.copy()

    stacked = np.concatenate([raw_true, raw_pred])
    _, lam = boxcox(stacked)
    from src.utils import safe_inv_boxcox  # noqa: F401

    from scipy.special import boxcox as _fwd
    y_true_bc = _fwd(raw_true, lam)
    y_pred_bc = _fwd(raw_pred, lam)

    m = compute_metrics(y_true_bc, y_pred_bc, lambda_bc=lam, eu_limit=50.0)

    assert m["exc_precision"] == 1.0
    assert m["exc_recall"] == 1.0
    assert m["exc_f1"] == 1.0


def test_nonfinite_inputs_are_ignored():
    raw = np.linspace(10.0, 80.0, 30)
    transformed, lam = boxcox(raw)

    preds = transformed.copy()
    preds[0] = np.nan
    preds[-1] = np.inf

    m = compute_metrics(transformed, preds, lambda_bc=lam)
    assert np.isfinite(m["MAE"])
    assert np.isfinite(m["RMSE"])
