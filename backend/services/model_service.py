"""
ModelService — singleton that owns all model artefacts and prediction logic.

Artefacts loaded from models/:
    lgbm_model.joblib      – LightGBMRegressor (68 features)
    arima_model.pkl        – ARIMAResultsWrapper (history embedded)
    sarimax_model.pkl      – SARIMAXResultsWrapper (history embedded)
    lambda_bc.pkl          – float  (Box-Cox lambda, required for back-transform)
    recent_history.pkl     – DataFrame  last ~60 days of features + PM10_transformed
    scaler.pkl             – StandardScaler for SARIMAX exog (optional)
    kmeans_model.pkl       – KMeans(3) for regime classification (optional)
    metrics.pkl            – dict {model_name: {mae, rmse, smape, r2}}

Run scripts/prepare_api_artifacts.py once to generate the supplementary artefacts.
"""

from __future__ import annotations

import logging
import math
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import holidays
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS = Path(__file__).resolve().parent.parent.parent / "models"

# Target station the models were trained on
_TARGET_STATION = "MpKrakWadow"

# Derived at import time from the dynamically detected station list in config
from config.config import STATIONS_META as _STATIONS_META
AVAILABLE_STATIONS: list[str] = list(_STATIONS_META.keys())

_LGBM_PATH       = _MODELS / "lgbm_model.joblib"
_ARIMA_PATH      = _MODELS / "arima_model.pkl"
_SARIMAX_PATH    = _MODELS / "sarimax_model.pkl"
_LAMBDA_PATH     = _MODELS / "lambda_bc.pkl"
_HISTORY_PATH    = _MODELS / "recent_history.pkl"
_SCALER_PATH     = _MODELS / "scaler.pkl"
_KMEANS_PATH     = _MODELS / "kmeans_model.pkl"
_METRICS_PATH    = _MODELS / "metrics.pkl"
_VALIDATION_PATH = _MODELS / "validation_results.pkl"


def _load_pkl(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _safe_inv_boxcox(values: np.ndarray, lam: float) -> np.ndarray:
    """Thin re-export of :func:`src.utils.safe_inv_boxcox` to keep the old import path."""
    from src.utils import safe_inv_boxcox
    return safe_inv_boxcox(values, lam)


def _pm10_level(pm10: float) -> str:
    if pm10 < 25:   return "Good"
    if pm10 < 50:   return "Moderate"
    if pm10 < 100:  return "High"
    return "Very High"


def _trend_label(forecasts: list[float]) -> str:
    if len(forecasts) < 2:
        return "Stable"
    delta = forecasts[-1] - forecasts[0]
    if delta > 5:   return "Rising"
    if delta < -5:  return "Falling"
    return "Stable"



def _cyclical(value: float, period: float) -> tuple[float, float]:
    rad = 2 * math.pi * value / period
    return math.sin(rad), math.cos(rad)


def _compute_lgbm_features(
    w: dict,         
    dt: pd.Timestamp,
    history: Optional[pd.DataFrame],
    lambda_bc: float,
) -> pd.DataFrame:
    """
    Build the full 68-column feature vector that the saved LightGBM model expects.

    'history' is a DataFrame with at least 'PM10_transformed' and weather columns
    for the last ~30 days prior to 'dt', sorted ascending.  If not available,
    sensible defaults are used.
    """
    month   = dt.month
    dow     = dt.dayofweek  
    doy     = dt.dayofyear
    year    = dt.year
    is_wknd = int(dow >= 5)
    is_heat = int(month in [1, 2, 3, 10, 11, 12])

    pl_hols = holidays.Poland(years=year)
    is_hol  = int(dt.date() in pl_hols)

    ms, mc = _cyclical(month, 12)
    ds, dc = _cyclical(dow, 7)
    ys, yc = _cyclical(doy, 365)

    temp_avg     = w["temp_avg"]
    wind_max     = w["wind_max"]
    wind_mean    = w["wind_mean"]
    pressure_avg = w["pressure_avg"]
    humidity_avg = w["humidity_avg"]
    rain_sum     = w["rain_sum"]
    snowfall_sum = w["snowfall_sum"]
    temp_min     = w.get("temp_min") or (temp_avg - 5)
    temp_max     = w.get("temp_max") or (temp_avg + 5)

    wind_dir_sin = 0.0
    wind_dir_cos = 0.0

    is_frost     = int(temp_avg <= 0)
    is_calm      = int(wind_mean <= 2)
    wind_inverse = 1.0 / (wind_max + 0.1)
    hdd          = max(0.0, 15.0 - temp_avg)
    temp_ampl    = temp_max - temp_min
    inversion    = int(temp_ampl < 4 and temp_avg < 5 and is_calm)

    if history is not None and len(history) >= 2:
        pm10_t = history["PM10_transformed"].dropna()

        def _lag(n: int) -> float:
            return float(pm10_t.iloc[-n]) if len(pm10_t) >= n else float(pm10_t.iloc[0])

        def _roll_mean(w_: int) -> float:
            return float(pm10_t.tail(w_).mean()) if len(pm10_t) >= 1 else 2.0

        def _roll_std(w_: int) -> float:
            return float(pm10_t.tail(w_).std()) if len(pm10_t) >= 2 else 0.5

        lag1 = _lag(1);  lag2 = _lag(2); lag7 = _lag(7); lag14 = _lag(14)
        rm3  = _roll_mean(3);  rs3  = _roll_std(3)
        rm7  = _roll_mean(7);  rs7  = _roll_std(7)
        rm14 = _roll_mean(14); rs14 = _roll_std(14)
        rm30 = _roll_mean(30); rs30 = _roll_std(30)
        rd7  = rm7 - rm14

        wind_7d = float(history["wind_max"].tail(7).mean()) if "wind_max" in history.columns else wind_max
        hdd_7d  = float((15 - history["temp_avg"].clip(upper=15)).tail(7).sum()) if "temp_avg" in history.columns else hdd * 3
        rain_3d = float(history["rain_sum"].tail(3).sum()) if "rain_sum" in history.columns else rain_sum
        dry_spell = int((history["rain_sum"].tail(14) == 0).sum()) if "rain_sum" in history.columns else 0
        pressure_trend = float(history["pressure_avg"].diff(3).iloc[-1]) if "pressure_avg" in history.columns else 0.0
        high_press = int(pressure_avg > float(history["pressure_avg"].tail(30).mean())) if "pressure_avg" in history.columns else 0

        aux_cols = [c for c in history.columns if c.startswith("MpKrak") and "lag" not in c]
        if aux_cols:
            aux_lag1 = history[aux_cols].iloc[-1]
            aux_mean = float(aux_lag1.mean())
            aux_std  = float(aux_lag1.std())
            aux_max  = float(aux_lag1.max())
            aux_sp   = float(aux_lag1.max() - aux_lag1.min())
            bujaka_l1 = float(history["MpKrakBujaka"].iloc[-1]) if "MpKrakBujaka" in history.columns else aux_mean
            bulwar_l1 = float(history["MpKrakBulwar"].iloc[-1]) if "MpKrakBulwar" in history.columns else aux_mean
            swoszo_l1 = float(history["MpKrakSwoszo"].iloc[-1]) if "MpKrakSwoszo" in history.columns else aux_mean
        else:
            bujaka_l1 = bulwar_l1 = swoszo_l1 = lag1
            aux_mean = aux_std = aux_max = aux_sp = lag1

        def _wlag(col: str, n: int) -> float:
            if col in history.columns and len(history) >= n:
                return float(history[col].iloc[-n])
            return w.get(col, 0.0)

        t_l2 = _wlag("temp_avg", 2);     t_l3 = _wlag("temp_avg", 3)
        p_l2 = _wlag("pressure_avg", 2); p_l3 = _wlag("pressure_avg", 3)
        wm_l2= _wlag("wind_mean", 2);    wm_l3= _wlag("wind_mean", 3)
        h_l2 = _wlag("humidity_avg", 2); h_l3 = _wlag("humidity_avg", 3)

    else:
        lag1 = lag2 = lag7 = lag14 = 2.0
        rm3 = rm7 = rm14 = rm30 = 2.0
        rs3 = rs7 = rs14 = rs30 = 0.5
        rd7 = 0.0
        wind_7d = wind_max; hdd_7d = hdd * 3
        rain_3d = rain_sum; dry_spell = 0
        pressure_trend = 0.0; high_press = 0
        bujaka_l1 = bulwar_l1 = swoszo_l1 = 2.0
        aux_mean = aux_std = aux_max = aux_sp = 2.0
        t_l2 = t_l3 = temp_avg
        p_l2 = p_l3 = pressure_avg
        wm_l2 = wm_l3 = wind_mean
        h_l2 = h_l3 = humidity_avg

    is_frost_calm = is_frost * is_calm
    is_heat_calm  = is_heat  * is_calm
    hdd_calm_     = hdd      * is_calm
    cold_dry_calm = int(temp_avg < 0) * int(rain_sum == 0) * is_calm
    regime_cluster = _heuristic_regime(temp_avg, wind_max, is_heat)

    row = {
        "temp_avg": temp_avg, "temp_min": temp_min, "temp_max": temp_max,
        "rain_sum": rain_sum, "wind_max": wind_max, "wind_mean": wind_mean,
        "pressure_avg": pressure_avg, "humidity_avg": humidity_avg, "snowfall_sum": snowfall_sum,
        "wind_dir_sin": wind_dir_sin, "wind_dir_cos": wind_dir_cos,
        "month": month, "week": dow, "year": year,
        "month_sin": ms, "month_cos": mc,
        "dow_sin": ds, "dow_cos": dc,
        "doy_sin": ys, "doy_cos": yc,
        "is_weekend": is_wknd, "is_heating_season": is_heat,
        "lag_1d": lag1, "lag_2d": lag2, "lag_7d": lag7, "lag_14d": lag14,
        "rolling_mean_3d": rm3,  "rolling_std_3d": rs3,
        "rolling_mean_7d": rm7,  "rolling_std_7d": rs7,
        "rolling_mean_14d": rm14,"rolling_std_14d": rs14,
        "rolling_mean_30d": rm30,"rolling_std_30d": rs30,
        "rolling_diff_7d": rd7,
        "is_holiday": is_hol,
        "is_frost": is_frost, "is_calm_wind": is_calm,
        "wind_inverse": wind_inverse, "wind_7d_mean": wind_7d,
        "pressure_trend_3d": pressure_trend, "high_pressure_flag": high_press,
        "rain_3d_sum": rain_3d, "dry_spell_days": dry_spell,
        "temp_avg_lag2d": t_l2, "temp_avg_lag3d": t_l3,
        "pressure_avg_lag2d": p_l2, "pressure_avg_lag3d": p_l3,
        "wind_mean_lag2d": wm_l2, "wind_mean_lag3d": wm_l3,
        "humidity_avg_lag2d": h_l2, "humidity_avg_lag3d": h_l3,
        "MpKrakBujaka_lag1": bujaka_l1, "MpKrakBulwar_lag1": bulwar_l1,
        "MpKrakSwoszo_lag1": swoszo_l1,
        "aux_mean_lag1": aux_mean, "aux_std_lag1": aux_std,
        "aux_max_lag1": aux_max, "aux_spread_lag1": aux_sp,
        "is_frost_calm": is_frost_calm, "is_heating_season_calm": is_heat_calm,
        "temp_amplitude": temp_ampl, "inversion_proxy": inversion,
        "regime_cluster": regime_cluster,
        "heating_degree_days": hdd, "hdd_7d": hdd_7d,
        "hdd_calm": hdd_calm_, "cold_dry_calm": cold_dry_calm,
    }
    return pd.DataFrame([row])


def _get_station_history(
    history: Optional[pd.DataFrame],
    station_id: str,
    lambda_bc: float,
) -> Optional[pd.DataFrame]:
    """
    Return a copy of *history* with PM10_transformed derived from the
    requested station's raw PM10 column.

    For the training target (MpKrakWadow) the stored PM10_transformed is used
    as-is.  For auxiliary stations the same Box-Cox lambda is applied to the
    station's raw PM10 column (an acceptable approximation — the models are
    the same for all stations).

    ARIMA and SARIMAX were trained solely on MpKrakWadow; their predictions
    are not station-specific regardless of this function.
    """
    if history is None:
        return None
    if station_id == _TARGET_STATION:
        return history
    if station_id not in history.columns:
        logger.warning(
            "Station %s not found in history columns; falling back to %s history.",
            station_id, _TARGET_STATION,
        )
        return history

    h = history.copy()
    raw = h[station_id].clip(lower=0.01).fillna(0.01)
    # Apply Box-Cox forward transform with the stored lambda
    if lambda_bc == 0:
        h["PM10_transformed"] = np.log(raw)
    else:
        h["PM10_transformed"] = (raw ** lambda_bc - 1) / lambda_bc
    return h


def _heuristic_regime(temp_avg: float, wind_max: float, is_heat: int) -> int:
    """Classify into 0=Clean, 1=Moderate, 2=Polluted via simple rules.
    Used as fallback when KMeans model is unavailable."""
    if temp_avg > 10 and wind_max > 5:
        return 0
    if is_heat and temp_avg < 2 and wind_max < 3:
        return 2
    return 1


def _extend_history(
    history: pd.DataFrame,
    weather: dict,
    raw_bc: float,
    pm10: float,
    dt: pd.Timestamp,
) -> pd.DataFrame:
    """Append a synthetic history row for the next day's LightGBM forecast.

    Inherits the previous row's schema so all columns expected by
    :func:`_compute_lgbm_features` are present; overwrites the few columns
    that change day-to-day (target PM10, its Box-Cox form, and weather if
    supplied).  Auxiliary stations are held at their last observed value
    (persistence) — a sensible default when no new sensor reading is yet
    available.
    """
    last = history.iloc[-1].copy()

    last["PM10_transformed"] = raw_bc
    if _TARGET_STATION in history.columns:
        last[_TARGET_STATION] = pm10

    for key, value in weather.items():
        if key in history.columns and value is not None:
            last[key] = value

    if isinstance(history.index, pd.DatetimeIndex):
        new_idx = history.index[-1] + pd.Timedelta(days=1)
    else:
        new_idx = dt

    new_row = pd.DataFrame([last.values], columns=history.columns, index=[new_idx])
    return pd.concat([history, new_row])

class ModelService:
    """Singleton that loads and exposes all model artefacts."""

    _instance: Optional[ModelService] = None

    def __new__(cls) -> ModelService:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return

        logger.info("Loading model artefacts …")

        self.lgbm    = joblib.load(_LGBM_PATH) if _LGBM_PATH.exists() else None
        self.arima   = _load_pkl(_ARIMA_PATH)
        self.sarimax = _load_pkl(_SARIMAX_PATH)
        loaded_lam = _load_pkl(_LAMBDA_PATH)
        self.lambda_bc: float = loaded_lam if loaded_lam is not None else -0.2503  # fallback
        self.history: Optional[pd.DataFrame] = _load_pkl(_HISTORY_PATH)
        self.scaler  = _load_pkl(_SCALER_PATH)
        self.kmeans  = _load_pkl(_KMEANS_PATH)

        raw_metrics = _load_pkl(_METRICS_PATH)
        self.metrics: dict = raw_metrics if isinstance(raw_metrics, dict) else {}

        raw_val = _load_pkl(_VALIDATION_PATH)
        self.validation_results: Optional[dict] = raw_val if isinstance(raw_val, dict) else None

        if not self.metrics:
            logger.warning(
                "metrics.pkl not found or empty — /metrics endpoint will return 503. "
                "Run `python scripts/prepare_api_artifacts.py` to generate metrics."
            )

        avail = [n for n, m in [
            ("LightGBM", self.lgbm), ("ARIMA", self.arima), ("SARIMAX", self.sarimax)
        ] if m is not None]
        logger.info("Available models: %s | lambda_bc=%.4f", avail, self.lambda_bc)

        self._loaded = True

    def predict(
        self,
        model_name: str,
        weather: dict,
        forecast_date: date,
        horizon: int = 1,
        station_id: str = _TARGET_STATION,
    ) -> tuple[list[float], Optional[list[float]], Optional[list[float]]]:
        """
        Returns (point_forecasts, lower_bounds, upper_bounds).
        All values in µg/m³ (PM10 physical scale).

        LightGBM uses station-specific lag history when station_id differs from
        the training target.  ARIMA and SARIMAX were trained on MpKrakWadow
        only — their outputs are station-agnostic.
        """
        if model_name == "LightGBM":
            history = self.get_station_history(station_id)
            return self._predict_lgbm(weather, forecast_date, horizon, history)
        elif model_name == "SARIMAX":
            return self._predict_sarimax(weather, forecast_date, horizon)
        elif model_name == "ARIMA":
            return self._predict_arima(horizon)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def get_station_history(self, station_id: str) -> Optional[pd.DataFrame]:
        """Return history with PM10_transformed for the requested station."""
        return _get_station_history(self.history, station_id, self.lambda_bc)

    def classify_regime(self, weather: dict, lag1: Optional[float] = None) -> int:
        """Returns cluster id 0/1/2."""
        if self.kmeans is not None:
            try:
                X = pd.DataFrame([{
                    "temp_avg": weather["temp_avg"],
                    "wind_max": weather["wind_max"],
                    "is_heating_season": int(pd.Timestamp.now().month in [1,2,3,10,11,12]),
                    "lag_1d": lag1 if lag1 is not None else 2.0,
                }])
                return int(self.kmeans.predict(X)[0])
            except Exception:
                pass
        return _heuristic_regime(
            weather["temp_avg"], weather["wind_max"],
            int(pd.Timestamp.now().month in [1, 2, 3, 10, 11, 12]),
        )

    def _predict_lgbm(
        self,
        weather: dict,
        forecast_date: date,
        horizon: int,
        history: Optional[pd.DataFrame] = None,
    ) -> tuple[list[float], None, None]:
        """Multi-day LightGBM forecast with proper history extension.

        After each one-step prediction the ``history`` DataFrame is extended
        by one day: the just-predicted Box-Cox value becomes the new lag-1,
        the back-transformed µg/m³ value is written into the target station's
        raw column, auxiliary stations are held at their last observed value
        (persistence), and the caller's weather dict is treated as a
        persistence assumption for the whole horizon.  This preserves the
        raw-PM10 + weather schema that :func:`_compute_lgbm_features`
        expects, which the previous implementation did not.
        """
        if self.lgbm is None:
            raise RuntimeError("LightGBM model not loaded")

        history = history.copy() if history is not None else None
        preds: list[float] = []

        for day_offset in range(horizon):
            dt = pd.Timestamp(forecast_date) + pd.Timedelta(days=day_offset)

            X = _compute_lgbm_features(weather, dt, history, self.lambda_bc)
            expected = self.lgbm.feature_name_
            for col in expected:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[expected].fillna(0.0)

            raw = float(self.lgbm.predict(X)[0])
            pm10 = float(_safe_inv_boxcox(np.array([raw]), self.lambda_bc)[0])
            preds.append(pm10)

            if day_offset + 1 < horizon and history is not None and len(history) > 0:
                history = _extend_history(history, weather, raw, pm10, dt)

        return preds, None, None

    def _predict_sarimax(
        self, weather: dict, forecast_date: date, horizon: int
    ) -> tuple[list[float], list[float], list[float]]:
        if self.sarimax is None:
            raise RuntimeError("SARIMAX model not loaded")

        preds, lowers, uppers = [], [], []
        for day_offset in range(horizon):
            dt    = pd.Timestamp(forecast_date) + pd.Timedelta(days=day_offset)
            month = dt.month
            is_heat = int(month in [1, 2, 3, 10, 11, 12])
            is_calm = int(weather["wind_mean"] <= 2)
            hdd     = max(0.0, 15.0 - weather["temp_avg"])
            rain_3d = weather["rain_sum"]
            inv_    = int(weather["temp_avg"] < 5 and is_calm)

            exog_row = np.array([[
                weather["temp_avg"], weather["wind_max"], is_heat,
                is_calm, hdd * is_calm, rain_3d, inv_,
            ]])

            if self.scaler is not None:
                exog_row = self.scaler.transform(exog_row)

            try:
                fc  = self.sarimax.get_forecast(steps=1, exog=exog_row)
                raw = float(fc.predicted_mean.iloc[0])
                ci  = fc.conf_int(alpha=0.10)
                ci_arr = np.asarray(ci)
                raw_lower = float(ci_arr[0, 0])
                raw_upper = float(ci_arr[0, 1])
                pm10 = float(_safe_inv_boxcox(np.array([raw]), self.lambda_bc)[0])
                lo   = float(_safe_inv_boxcox(np.array([raw_lower]), self.lambda_bc)[0])
                hi   = float(_safe_inv_boxcox(np.array([raw_upper]), self.lambda_bc)[0])
                preds.append(pm10)
                lowers.append(lo)
                uppers.append(hi)
            except Exception as exc:
                logger.warning("SARIMAX forecast failed: %s", exc)
                preds.append(30.0); lowers.append(20.0); uppers.append(40.0)

        return preds, lowers, uppers

    def _predict_arima(
        self, horizon: int
    ) -> tuple[list[float], list[float], list[float]]:
        if self.arima is None:
            raise RuntimeError("ARIMA model not loaded")

        try:
            fc  = self.arima.get_forecast(steps=horizon)
            pm  = fc.predicted_mean
            raw_preds  = np.asarray(pm).flatten()
            ci         = fc.conf_int(alpha=0.10)
            ci_arr     = np.asarray(ci)
            lowers_raw = ci_arr[:, 0].flatten()
            uppers_raw = ci_arr[:, 1].flatten()
        except Exception as exc:
            logger.warning("ARIMA get_forecast failed: %s — using last value", exc)
            last_bc = float(self.arima.model.endog[-1])
            raw_preds  = np.full(horizon, last_bc)
            lowers_raw = raw_preds * 0.85
            uppers_raw = raw_preds * 1.15

        preds  = _safe_inv_boxcox(raw_preds,  self.lambda_bc).tolist()
        lowers = _safe_inv_boxcox(lowers_raw, self.lambda_bc).tolist()
        uppers = _safe_inv_boxcox(uppers_raw, self.lambda_bc).tolist()
        return preds, lowers, uppers

_service: Optional[ModelService] = None

def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
        _service.load()
    return _service