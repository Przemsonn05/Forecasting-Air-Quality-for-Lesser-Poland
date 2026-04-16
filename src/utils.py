"""
Shared utilities: logging setup, plotting style, Box-Cox helpers,
and small data helpers used across multiple pipeline stages.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import inv_boxcox

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger.

    Parameters
    ----------
    name:
        Usually ``__name__`` of the calling module.
    level:
        Logging level (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

def set_plot_style() -> None:
    """Apply a consistent Seaborn / Matplotlib style project-wide."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        "figure.dpi": 120,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save a matplotlib figure and close it.

    Parameters
    ----------
    fig:
        The Figure object to save.
    path:
        Destination file path (PNG recommended).
    dpi:
        Output resolution.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def safe_inv_boxcox(
    values: np.ndarray,
    lambda_bc: float,
    clip_min: float = 0.0,
    clip_max: float = 500.0,
) -> np.ndarray:
    """Invert a Box-Cox transform with full NaN / inf protection.

    ``scipy.special.inv_boxcox`` can return NaN or ±inf when the input falls
    outside the analytic domain (e.g. ``(1 + lam·y)`` going negative for a
    fractional lambda).  This helper pre-clips the input to a physically
    plausible Box-Cox range and then post-sanitises the output so that a
    caller downstream of a forecasting model never receives a silent NaN or
    an exploding prediction.

    The default bounds match Kraków PM10 reality: concentrations are
    non-negative and almost always ≤ 500 µg/m³ (the all-time GIOŚ record).

    Parameters
    ----------
    values:
        Transformed (Box-Cox scale) values to invert.
    lambda_bc:
        Lambda returned by ``scipy.stats.boxcox``.
    clip_min, clip_max:
        Final output bounds (default 0 and 500 µg/m³).

    Returns
    -------
    np.ndarray
        Original-scale PM10 values in µg/m³, always finite and within
        ``[clip_min, clip_max]``.
    """
    y = np.clip(np.asarray(values, dtype=float), -5.0, 50.0)
    out = np.asarray(inv_boxcox(y, lambda_bc), dtype=float)
    out = np.nan_to_num(out, nan=clip_min, posinf=clip_max, neginf=clip_min)
    return np.clip(out, clip_min, clip_max)


def inverse_boxcox_transform(values: np.ndarray, lambda_bc: float) -> np.ndarray:
    """Alias for :func:`safe_inv_boxcox` with default clip bounds."""
    return safe_inv_boxcox(values, lambda_bc)

def date_split(
    df: pd.DataFrame,
    train_end: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-indexed DataFrame by fixed calendar dates.

    Mirrors the notebook's exact boundary approach:
    - train : index <= train_end
    - val   : train_end < index <= val_end
    - test  : index > val_end

    Parameters
    ----------
    df:
        DataFrame with a ``DatetimeIndex`` sorted ascending.
    train_end:
        Last date (inclusive) of the training set, e.g. ``"2022-12-31"``.
    val_end:
        Last date (inclusive) of the validation set, e.g. ``"2023-12-31"``.

    Returns
    -------
    train, val, test : tuple of DataFrames
    """
    train = df[df.index <= train_end]
    val = df[(df.index > train_end) & (df.index <= val_end)]
    test = df[df.index > val_end]

    logger = get_logger(__name__)
    logger.info(
        "Date split — train: %d rows (%s … %s)  "
        "val: %d rows (%s … %s)  "
        "test: %d rows (%s … %s)",
        len(train), train.index.min().date(), train.index.max().date(),
        len(val),   val.index.min().date(),   val.index.max().date(),
        len(test),  test.index.min().date(),  test.index.max().date(),
    )
    return train, val, test