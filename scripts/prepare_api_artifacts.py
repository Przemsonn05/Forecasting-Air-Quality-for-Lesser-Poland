"""
End-to-end training + serialisation of every artefact the FastAPI backend
needs.  Running this single script from a fresh clone is sufficient to
produce a working API deployment.

Typical usage
-------------
    python scripts/prepare_api_artifacts.py                 # train all models
    python scripts/prepare_api_artifacts.py --optuna        # + Optuna tuning for LGBM
    python scripts/prepare_api_artifacts.py --models lgbm   # train one model

Outputs (models/)
-----------------
    lgbm_model.joblib        LightGBM regressor (features = src.config.LGBM_FEATURES)
    arima_model.pkl          ARIMAResults refitted on train+val
    sarimax_model.pkl        SARIMAXResults refitted on train+val
    lambda_bc.pkl            Box-Cox lambda (for inverse transform at inference)
    recent_history.pkl       Last 60 days of engineered features (API lag retrieval)
    scaler.pkl               StandardScaler for SARIMAX_EXOG fitted on train only
    kmeans_model.pkl         KMeans(3) for regime classification
    metrics.pkl              Unified metric dict:
                             {model: {mae,rmse,smape,r2,exc_precision,exc_recall,exc_f1}}
    validation_results.pkl   Per-model val-set predictions for the Streamlit app

Walk-forward validation metrics and production-serving models are
computed separately:
- metrics come from a strict train-only-then-forecast walk-forward
  (identical to what ``python main.py`` reports);
- the models pickled to disk are refit on train+val so the API can serve
  one-step-ahead forecasts starting immediately after the validation
  period.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    DATA_DIR, YEARS, STATIONS, TARGET, TRAIN_END, VAL_END,
    WEATHER_API_URL, WEATHER_PARAMS, WEATHER_COL_RENAME,
    AUX_STATIONS, HEATING_MONTHS, LAG_DAYS, ROLLING_WINDOWS,
    SARIMAX_EXOG, LGBM_FEATURES, LGBM_PARAMS,
    LGBM_EARLY_STOPPING_ROUNDS, LGBM_ES_FRACTION,
    GAP_INTERP_LIMIT, EU_PM10_DAILY_LIMIT, REFIT_EVERY,
)
from src.data_loading import load_pm10_raw, parse_pm10_stations, fetch_weather
from src.data_preprocessing import impute_gaps, merge_weather
from src.evaluation import compute_metrics
from src.feature_engineering import build_features
from src.models import (
    train_predict_arima,
    train_predict_lgbm,
    train_predict_sarimax,
)
from src.utils import date_split, safe_inv_boxcox

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
BC_COL = "PM10_transformed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)


def _normalise_metric_keys(m: dict) -> dict:
    """Map ``compute_metrics`` uppercase keys to the lowercase schema the API + UI use."""
    return {
        "mae":           float(m.get("MAE")) if m.get("MAE") is not None else None,
        "rmse":          float(m.get("RMSE")) if m.get("RMSE") is not None else None,
        "smape":         float(m.get("SMAPE")) if m.get("SMAPE") is not None else None,
        "r2":            float(m.get("R2")) if m.get("R2") is not None else None,
        "exc_precision": _float_or_none(m.get("exc_precision")),
        "exc_recall":    _float_or_none(m.get("exc_recall")),
        "exc_f1":        _float_or_none(m.get("exc_f1")),
    }


def _float_or_none(v) -> float | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return float(v)


def _save_pkl(obj, path: Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    log.info("Saved %s", path.name)


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


def _fit_final_arima(endog_full: pd.Series, order: tuple) -> None:
    """Refit ARIMA on train+val so the API can forecast immediately after val."""
    from statsmodels.tsa.arima.model import ARIMA
    final = ARIMA(endog_full, order=order).fit()
    _save_pkl(final, MODELS_DIR / "arima_model.pkl")


def _fit_final_sarimax(
    endog_full: pd.Series,
    exog_full_scaled: pd.DataFrame,
    order: tuple,
    seasonal_order: tuple,
) -> None:
    """Refit SARIMAX on train+val (already-scaled exog) for production serving."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    final = SARIMAX(
        endog_full,
        exog=exog_full_scaled,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    _save_pkl(final, MODELS_DIR / "sarimax_model.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--models", nargs="+",
        choices=["lgbm", "arima", "sarimax"],
        default=["lgbm", "arima", "sarimax"],
        help="Which models to train (default: all three)",
    )
    p.add_argument(
        "--optuna", action="store_true",
        help="Run Optuna hyper-parameter search for LightGBM (slow)",
    )
    p.add_argument(
        "--skip-data", action="store_true",
        help="Reuse an already-cached merged frame if present (dev convenience)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=== Stage 1: load + preprocess ===")
    raw = load_pm10_raw(DATA_DIR, YEARS)
    pm10 = parse_pm10_stations(raw, STATIONS)
    weather = fetch_weather(WEATHER_API_URL, WEATHER_PARAMS, WEATHER_COL_RENAME)
    pm10 = impute_gaps(pm10, STATIONS, limit=GAP_INTERP_LIMIT)
    merged = merge_weather(pm10, weather)

    log.info("=== Stage 2: feature engineering ===")
    df_feat, lambda_bc = build_features(
        merged, TARGET, TRAIN_END, AUX_STATIONS,
        HEATING_MONTHS, LAG_DAYS, ROLLING_WINDOWS,
    )
    log.info("Box-Cox lambda = %.6f", lambda_bc)
    _save_pkl(lambda_bc, MODELS_DIR / "lambda_bc.pkl")

    train, val, _ = date_split(df_feat, TRAIN_END, VAL_END)

    log.info("=== Stage 3: auxiliary artefacts ===")

    recent = df_feat.tail(60).copy()
    keep_cols = [BC_COL, TARGET] + [
        c for c in df_feat.columns
        if any(c.startswith(p) for p in
               ["temp_", "wind_", "rain_", "pressure_", "humidity_", "snow", "MpKrak"])
    ]
    recent = recent[[c for c in keep_cols if c in recent.columns]]
    _save_pkl(recent, MODELS_DIR / "recent_history.pkl")

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # NB: fit on the TRAIN split only to prevent exog leakage.
    exog_train = train[SARIMAX_EXOG].ffill().bfill()
    scaler.fit(exog_train)
    _save_pkl(scaler, MODELS_DIR / "scaler.pkl")

    from sklearn.cluster import KMeans
    km_cols = [c for c in ["temp_avg", "wind_max", "is_heating_season", "lag_1d"]
               if c in df_feat.columns]
    X_km = df_feat[km_cols].dropna()
    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_km)
    _save_pkl(km, MODELS_DIR / "kmeans_model.pkl")

    log.info("=== Stage 4: model training + walk-forward evaluation ===")

    actual_val_pm10 = safe_inv_boxcox(val[BC_COL].values, lambda_bc)
    metrics: dict[str, dict] = {}
    val_results: dict = {
        "dates":  [str(d.date()) for d in val.index],
        "actual": actual_val_pm10.tolist(),
    }

    # ---------- LightGBM ----------
    if "lgbm" in args.models:
        log.info("Training LightGBM (optuna=%s) …", args.optuna)
        preds_bc, lgbm = train_predict_lgbm(
            train, val,
            target_col=BC_COL,
            feature_cols=LGBM_FEATURES,
            params=LGBM_PARAMS,
            early_stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS,
            es_fraction=LGBM_ES_FRACTION,
            use_optuna=args.optuna,
        )
        joblib.dump(lgbm, MODELS_DIR / "lgbm_model.joblib")
        log.info("Saved lgbm_model.joblib")

        metrics["LightGBM"] = _normalise_metric_keys(compute_metrics(
            val[BC_COL].values, preds_bc, lambda_bc, "LightGBM",
            eu_limit=EU_PM10_DAILY_LIMIT,
        ))
        val_results["LightGBM"] = {
            "predicted": safe_inv_boxcox(preds_bc, lambda_bc).tolist(),
        }

    # ---------- ARIMA ----------
    if "arima" in args.models:
        log.info("Training ARIMA (walk-forward refit every %d steps) …", REFIT_EVERY)
        preds_a, _, _ = train_predict_arima(
            train, val, target_col=BC_COL, refit_every=REFIT_EVERY,
        )
        metrics["ARIMA"] = _normalise_metric_keys(compute_metrics(
            val[BC_COL].values, preds_a, lambda_bc, "ARIMA",
            eu_limit=EU_PM10_DAILY_LIMIT,
        ))
        val_results["ARIMA"] = {
            "predicted": safe_inv_boxcox(preds_a, lambda_bc).tolist(),
        }

        import pmdarima as pm
        endog_full = pd.concat([train[BC_COL], val[BC_COL]]).dropna()
        log.info("Selecting final ARIMA order on train+val (auto_arima) …")
        auto = pm.auto_arima(
            endog_full, seasonal=False, stepwise=True,
            suppress_warnings=True, error_action="ignore",
        )
        _fit_final_arima(endog_full, auto.order)

    # ---------- SARIMAX ----------
    if "sarimax" in args.models:
        log.info("Training SARIMAX (walk-forward refit every %d steps) …", REFIT_EVERY)
        preds_s = train_predict_sarimax(
            train, val, target_col=BC_COL, exog_cols=SARIMAX_EXOG,
            refit_every=REFIT_EVERY,
        )
        metrics["SARIMAX"] = _normalise_metric_keys(compute_metrics(
            val[BC_COL].values, preds_s, lambda_bc, "SARIMAX",
            eu_limit=EU_PM10_DAILY_LIMIT,
        ))
        val_results["SARIMAX"] = {
            "predicted": safe_inv_boxcox(preds_s, lambda_bc).tolist(),
        }

        import pmdarima as pm
        endog_full = pd.concat([train[BC_COL], val[BC_COL]]).dropna()
        # Apply the train-fitted scaler to train+val exog so the pickled model
        # and the live API both operate in the same scaled space.
        exog_tv = pd.concat([train[SARIMAX_EXOG], val[SARIMAX_EXOG]])
        exog_tv = exog_tv.loc[endog_full.index].ffill().bfill()
        exog_tv_scaled = pd.DataFrame(
            scaler.transform(exog_tv),
            index=exog_tv.index,
            columns=SARIMAX_EXOG,
        )
        log.info("Selecting final SARIMAX order on train+val (auto_arima m=7) …")
        auto_s = pm.auto_arima(
            endog_full, X=exog_tv_scaled, seasonal=True, m=7,
            stepwise=True, suppress_warnings=True, error_action="ignore",
        )
        _fit_final_sarimax(endog_full, exog_tv_scaled, auto_s.order, auto_s.seasonal_order)

    # ---------- Naïve persistence baseline (on µg/m³ scale) ----------
    naive = np.roll(actual_val_pm10, 1)
    naive[0] = actual_val_pm10[0]
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    y_exc = (actual_val_pm10 >= EU_PM10_DAILY_LIMIT).astype(int)
    p_exc = (naive >= EU_PM10_DAILY_LIMIT).astype(int)
    from sklearn.metrics import precision_score, recall_score, f1_score
    metrics["Naïve"] = {
        "mae":   round(float(mean_absolute_error(actual_val_pm10, naive)), 4),
        "rmse":  round(float(np.sqrt(mean_squared_error(actual_val_pm10, naive))), 4),
        "smape": round(_smape(actual_val_pm10, naive), 4),
        "r2":    None,
        "exc_precision": round(float(precision_score(y_exc, p_exc, zero_division=0)), 4),
        "exc_recall":    round(float(recall_score(y_exc, p_exc, zero_division=0)), 4),
        "exc_f1":        round(float(f1_score(y_exc, p_exc, zero_division=0)), 4),
    }

    _save_pkl(metrics, MODELS_DIR / "metrics.pkl")
    _save_pkl(val_results, MODELS_DIR / "validation_results.pkl")

    log.info("=== Done.  Models available: %s ===", sorted(metrics.keys()))


if __name__ == "__main__":
    main()
