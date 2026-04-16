"""
Shared pytest fixtures.

Builds a small synthetic daily time series with the same schema as the real
preprocessed PM10 frame, so the unit tests don't need the raw GIOŚ Excel
files or live network access to Open-Meteo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET = "MpKrakWadow"
AUX_STATIONS = ["MpKrakBujaka", "MpKrakBulwar"]
RNG = np.random.default_rng(42)


@pytest.fixture(scope="session")
def daily_index() -> pd.DatetimeIndex:
    """Two full years of daily timestamps."""
    return pd.date_range("2021-01-01", "2022-12-31", freq="D", name="Date")


@pytest.fixture
def synthetic_raw_pm10(daily_index) -> pd.DataFrame:
    """Seasonal PM10 series with a deliberate long gap and a short gap.

    The series is strictly positive, which is the precondition the Box-Cox
    transform requires.
    """
    n = len(daily_index)
    seasonal = 40 + 25 * np.sin(2 * np.pi * np.arange(n) / 365.0)
    noise = RNG.normal(0, 4, size=n)
    series = np.clip(seasonal + noise, 5, None)

    df = pd.DataFrame(
        {
            TARGET: series,
            AUX_STATIONS[0]: series + RNG.normal(0, 3, size=n),
            AUX_STATIONS[1]: series + RNG.normal(0, 3, size=n),
        },
        index=daily_index,
    )
    df = df.asfreq("D")

    df.iloc[10:12, 0] = np.nan
    df.iloc[100:112, 0] = np.nan

    return df


@pytest.fixture
def synthetic_weather(daily_index) -> pd.DataFrame:
    """Minimal weather frame — the columns consumed by feature engineering."""
    n = len(daily_index)
    temp_avg = 10 + 12 * np.sin(2 * np.pi * (np.arange(n) - 120) / 365.0)
    return pd.DataFrame(
        {
            "temp_avg": temp_avg,
            "temp_min": temp_avg - 4,
            "temp_max": temp_avg + 4,
            "rain_sum": RNG.gamma(1.0, 1.2, size=n),
            "wind_max": RNG.gamma(2.0, 2.0, size=n),
            "wind_mean": RNG.gamma(1.5, 1.2, size=n),
            "pressure_avg": 1013 + RNG.normal(0, 6, size=n),
            "humidity_avg": 60 + RNG.normal(0, 10, size=n),
            "snowfall_sum": np.zeros(n),
            "wind_dir_sin": RNG.uniform(-1, 1, size=n),
            "wind_dir_cos": RNG.uniform(-1, 1, size=n),
        },
        index=daily_index,
    )


@pytest.fixture
def synthetic_merged(synthetic_raw_pm10, synthetic_weather) -> pd.DataFrame:
    """PM10 + weather merged on the daily index — mirrors merge_weather output."""
    return synthetic_raw_pm10.join(synthetic_weather, how="left")
