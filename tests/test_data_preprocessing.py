"""
Unit tests for src.data_preprocessing.

The most important invariant here is the long-gap policy: short gaps
(≤ ``limit`` days) must be interpolated linearly, long gaps must stay NaN
*and* be flagged in a companion ``{station}_long_gap`` column so that
downstream feature engineering can drop (rather than fabricate) those rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_preprocessing import impute_gaps


def test_short_gap_is_interpolated(synthetic_raw_pm10):
    df = impute_gaps(synthetic_raw_pm10, stations=["MpKrakWadow"], limit=3)

    short_gap_slice = df["MpKrakWadow"].iloc[10:12]
    assert short_gap_slice.notna().all(), "Short gaps (≤ limit) must be filled"
    assert short_gap_slice.between(0, 200).all()


def test_long_gap_is_preserved_as_nan(synthetic_raw_pm10):
    """Long gaps must NOT be silently filled — they should remain NaN."""
    df = impute_gaps(synthetic_raw_pm10, stations=["MpKrakWadow"], limit=3)

    long_gap_slice = df["MpKrakWadow"].iloc[100:112]
    assert long_gap_slice.isna().all(), (
        "Long gaps were filled by ffill/bfill — downstream models will train "
        "on fabricated values. The impute_gaps fix must preserve them as NaN."
    )


def test_long_gap_flag_column_is_set(synthetic_raw_pm10):
    df = impute_gaps(synthetic_raw_pm10, stations=["MpKrakWadow"], limit=3)

    flag = df["MpKrakWadow_long_gap"]
    assert flag.dtype.kind in ("i", "u")
    assert flag.iloc[100:112].eq(1).all(), "Long-gap rows should be flagged 1"

    other_mask = np.ones(len(df), dtype=bool)
    other_mask[100:112] = False
    assert flag[other_mask].eq(0).all(), "Non-long-gap rows should be flagged 0"


def test_short_gap_does_not_trigger_long_gap_flag(synthetic_raw_pm10):
    df = impute_gaps(synthetic_raw_pm10, stations=["MpKrakWadow"], limit=3)

    assert df["MpKrakWadow_long_gap"].iloc[10:12].eq(0).all()


def test_flag_count_matches_long_gap_length(synthetic_raw_pm10):
    df = impute_gaps(synthetic_raw_pm10, stations=["MpKrakWadow"], limit=3)
    assert int(df["MpKrakWadow_long_gap"].sum()) == 12


def test_other_stations_are_unaffected_when_only_one_passed(synthetic_raw_pm10):
    """Only the stations in the ``stations`` arg should get a long-gap flag."""
    df = impute_gaps(synthetic_raw_pm10, stations=["MpKrakWadow"], limit=3)
    assert "MpKrakBujaka_long_gap" not in df.columns
    assert "MpKrakBulwar_long_gap" not in df.columns


def test_imputation_limit_is_respected():
    idx = pd.date_range("2022-01-01", periods=20, freq="D")
    values = np.linspace(10, 30, 20)
    values[5:9] = np.nan
    df = pd.DataFrame({"s": values}, index=idx)

    out = impute_gaps(df, stations=["s"], limit=3)

    assert out["s_long_gap"].iloc[5:9].eq(1).all(), (
        "A 4-day gap with limit=3 must be flagged as a long gap"
    )
    assert out["s"].iloc[5:9].isna().all(), (
        "A 4-day gap with limit=3 must stay NaN"
    )
