"""
Unit tests for src.utils.

Covers the two helpers that are critical for correctness:
- ``date_split`` — the ONE thing standing between the pipeline and classic
  look-ahead leakage in a time-series setup.
- ``safe_inv_boxcox`` — the inverse transform every model prediction
  ultimately passes through before it reaches the UI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import boxcox

from src.utils import date_split, safe_inv_boxcox


def test_date_split_is_strictly_chronological():
    idx = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)

    train, val, test = date_split(df, train_end="2022-12-31", val_end="2023-06-30")

    assert train.index.max() <= pd.Timestamp("2022-12-31")
    assert val.index.min() > pd.Timestamp("2022-12-31")
    assert val.index.max() <= pd.Timestamp("2023-06-30")
    assert test.index.min() > pd.Timestamp("2023-06-30")


def test_date_split_partitions_are_disjoint_and_complete():
    idx = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    df = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)

    train, val, test = date_split(df, "2022-12-31", "2023-06-30")

    joined = pd.concat([train, val, test])
    assert len(joined) == len(df)
    assert not joined.index.has_duplicates


def test_safe_inv_boxcox_round_trip_is_close():
    raw = np.array([5.0, 12.0, 23.4, 45.1, 78.0, 110.0])
    transformed, lam = boxcox(raw)

    recovered = safe_inv_boxcox(transformed, lam)

    np.testing.assert_allclose(recovered, raw, rtol=1e-6)


def test_safe_inv_boxcox_clips_physical_range():
    """Extreme Box-Cox inputs must not explode — concentrations are non-negative and bounded."""
    lam = 0.3
    x = np.array([-1e3, -10.0, 0.0, 5.0, 1e3])

    y = safe_inv_boxcox(x, lam, clip_min=0.0, clip_max=500.0)

    assert np.all(y >= 0.0)
    assert np.all(y <= 500.0)
    assert np.all(np.isfinite(y))


def test_safe_inv_boxcox_respects_custom_bounds():
    lam = 0.0
    x = np.array([-5.0, 0.0, 3.0, 10.0])
    y = safe_inv_boxcox(x, lam, clip_min=1.0, clip_max=50.0)
    assert y.min() >= 1.0
    assert y.max() <= 50.0
