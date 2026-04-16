"""
Data preprocessing: gap imputation, long-gap flagging, and weather join.

Short gaps (≤ ``limit`` days) are interpolated linearly on the time axis.
Long gaps (> ``limit`` days) are left as NaN so that downstream models do
not train on artificially propagated values — a companion
``{station}_long_gap`` binary flag marks those rows.  Only truly isolated
boundary NaNs (i.e. single-day gaps at the very start or end of the series
that interpolation cannot reach) are closed with ffill/bfill; long-gap rows
are explicitly restored to NaN afterwards.
"""

import numpy as np
import pandas as pd

from src.utils import get_logger

logger = get_logger(__name__)

def impute_gaps(
    df: pd.DataFrame,
    stations: list[str],
    limit: int = 3,
) -> pd.DataFrame:
    """Impute short gaps and flag long gaps for each station column.

    Per-station logic:
    1. Identify runs of consecutive NaN values.
    2. Gaps of ``limit`` days or fewer → time-based linear interpolation.
    3. Gaps longer than ``limit`` → binary ``{col}_long_gap`` flag (1 = long
       gap); **values remain NaN** so the model does not learn from fabricated
       data.
    4. Isolated boundary NaNs (interpolation cannot reach the first or last
       values) are closed with a short ``ffill`` / ``bfill`` pass, but long-gap
       rows are explicitly restored to NaN afterwards.

    Parameters
    ----------
    df:
        Date-indexed DataFrame with one column per station (daily frequency).
    stations:
        List of station column names to process.
    limit:
        Maximum gap length (in days) eligible for interpolation.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values and ``*_long_gap`` indicator columns
        appended (copy).
    """
    df = df.copy()

    for col in stations:
        is_null = df[col].isnull()
        gap_id = is_null.ne(is_null.shift()).cumsum()
        gap_sizes = is_null.groupby(gap_id).transform("sum")

        long_gap_mask = (is_null) & (gap_sizes > limit)
        df[f"{col}_long_gap"] = long_gap_mask.astype(int)

        df[col] = df[col].interpolate(method="time", limit=limit)

        df[col] = df[col].ffill().bfill()
        df.loc[long_gap_mask, col] = np.nan

        n_long = int(df[f"{col}_long_gap"].sum())
        n_nan_after = int(df[col].isna().sum())
        logger.info(
            "Column '%s': %d long-gap days flagged (>%d d); "
            "%d NaNs preserved in target after imputation",
            col, n_long, limit, n_nan_after,
        )

    return df

def merge_weather(
    pm10_df: pd.DataFrame,
    weather_df: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join PM10 station data with weather data on the date index.

    Parameters
    ----------
    pm10_df:
        Date-indexed PM10 DataFrame (output of :func:`impute_gaps`).
    weather_df:
        Date-indexed weather DataFrame (output of
        :func:`src.data_loading.fetch_weather`).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame (same row count as ``pm10_df``).
    """
    merged  = pm10_df.join(weather_df, how="left")
    missing = merged[weather_df.columns].isna().sum().sum()
    if missing:
        logger.warning(
            "Weather join left %d missing values — check date alignment",
            missing,
        )
    logger.info("Merged DataFrame shape: %s", merged.shape)
    return merged