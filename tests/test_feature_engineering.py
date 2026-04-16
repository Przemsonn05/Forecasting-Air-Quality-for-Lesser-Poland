"""
Unit tests for src.feature_engineering.

Focus areas:
- Box-Cox lambda is fitted on training data only
- Lag features are *exactly* shift(k) of the transformed column (no same-day leakage)
- Rolling features are built on a shift(1) base (no same-day leakage)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    apply_boxcox_transform,
)


def test_boxcox_lambda_only_uses_training_data(synthetic_raw_pm10):
    """If we shuffle values *after* train_end the lambda must not change."""
    df = synthetic_raw_pm10.copy()
    train_end = "2021-12-31"

    _, lambda_train = apply_boxcox_transform(df, raw_col="MpKrakWadow", train_end=train_end)

    df_mutated = df.copy()
    post_mask = df_mutated.index > train_end
    df_mutated.loc[post_mask, "MpKrakWadow"] *= 3.7

    _, lambda_mutated = apply_boxcox_transform(
        df_mutated, raw_col="MpKrakWadow", train_end=train_end,
    )

    assert lambda_train == pytest.approx(lambda_mutated, rel=1e-9), (
        "Lambda changed after mutating post-train values → Box-Cox peeked at val/test"
    )


def test_boxcox_transformed_column_keeps_nans_for_missing_inputs():
    idx = pd.date_range("2022-01-01", periods=10, freq="D")
    values = [10.0, 12.0, np.nan, 15.0, 14.0, 13.0, np.nan, 20.0, 22.0, 18.0]
    df = pd.DataFrame({"pm": values}, index=idx)

    out, _ = apply_boxcox_transform(df, raw_col="pm", train_end="2022-01-10")

    assert out["PM10_transformed"].isna().sum() == 2
    assert out["PM10_transformed"].notna().sum() == 8


def test_lag_feature_equals_shift_of_transformed_column():
    idx = pd.date_range("2022-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {"PM10_transformed": np.linspace(1.0, 30.0, 30)},
        index=idx,
    )

    out = add_lag_features(df, lags=[1, 2, 7])

    pd.testing.assert_series_equal(
        out["lag_1d"],
        df["PM10_transformed"].shift(1).rename("lag_1d"),
    )
    pd.testing.assert_series_equal(
        out["lag_7d"],
        df["PM10_transformed"].shift(7).rename("lag_7d"),
    )


def test_rolling_features_are_built_on_shift1_base():
    """Same-day value must never leak into rolling mean/std."""
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    df = pd.DataFrame({"pm_raw": np.arange(1.0, 41.0)}, index=idx)

    out = add_rolling_features(df, raw_col="pm_raw", windows=[3, 7, 14])

    expected = (
        df["pm_raw"].shift(1).rolling(3, min_periods=1).mean().rename("rolling_mean_3d")
    )
    pd.testing.assert_series_equal(out["rolling_mean_3d"], expected)

    assert not out["rolling_mean_3d"].equals(df["pm_raw"].rolling(3).mean()), (
        "Rolling feature must not equal a same-day rolling — that would leak today's value"
    )


def test_rolling_features_add_momentum_difference():
    idx = pd.date_range("2022-01-01", periods=40, freq="D")
    df = pd.DataFrame({"pm_raw": np.arange(1.0, 41.0)}, index=idx)

    out = add_rolling_features(df, raw_col="pm_raw", windows=[7, 14])

    computed = out["rolling_diff_7d"].dropna()
    reference = (out["rolling_mean_7d"] - out["rolling_mean_14d"]).dropna()
    pd.testing.assert_series_equal(computed, reference, check_names=False)
