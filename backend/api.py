"""
FastAPI backend for the PM10 Air Quality Forecasting system.

Endpoints
---------
POST /predict   – multi-day PM10 forecast
GET  /metrics   – pre-computed model metrics
POST /explain   – SHAP feature attributions (LightGBM)
POST /interpret – NLG natural-language explanation
GET  /health    – liveness check

Run:
    uvicorn backend.api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.schemas import (
    ExplainRequest, ExplainResponse, FeatureContribution,
    InterpretRequest, InterpretResponse,
    MetricsResponse, ModelMetrics,
    PredictRequest, PredictResponse, DayForecast,
)
from backend.services.model_service import (
    ModelService, get_model_service, AVAILABLE_STATIONS,
    _compute_lgbm_features, _pm10_level, _trend_label,
)
from backend.services.explainability_service import ExplainabilityService
from backend.services.interpretability_service import InterpretabilityService
from config.config import REGIME_LABELS, REGIME_COLORS, STATIONS_META

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models …")
    svc = get_model_service()        
    app.state.explain_svc = ExplainabilityService(svc.lgbm)
    app.state.interpret_svc = InterpretabilityService()
    logger.info("Models loaded.  Ready.")
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="PM10 Air Quality Forecasting API",
    description="Predicts daily PM10 levels for Kraków monitoring stations.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _model_svc() -> ModelService:
    return get_model_service()

def _explain_svc() -> ExplainabilityService:
    return app.state.explain_svc

def _interpret_svc() -> InterpretabilityService:
    return app.state.interpret_svc

def _weather_dict(w) -> dict:
    return {
        "temp_avg": w.temp_avg,
        "wind_max": w.wind_max,
        "wind_mean": w.wind_mean,
        "humidity_avg": w.humidity_avg,
        "pressure_avg": w.pressure_avg,
        "rain_sum": w.rain_sum,
        "snowfall_sum": w.snowfall_sum,
        "temp_min": w.temp_min,
        "temp_max": w.temp_max,
    }

@app.get("/stations")
async def stations() -> dict[str, Any]:
    """Return all available monitoring stations with metadata."""
    return {
        "stations": [
            {
                "id": sid,
                "name": STATIONS_META.get(sid, {}).get("name", sid),
                "lat":  STATIONS_META.get(sid, {}).get("lat"),
                "lon":  STATIONS_META.get(sid, {}).get("lon"),
            }
            for sid in AVAILABLE_STATIONS
            if sid in STATIONS_META
        ]
    }

@app.get("/validation")
async def validation(
    svc: ModelService = Depends(_model_svc),
) -> dict[str, Any]:
    """Return pre-computed validation-set predictions for all models (2023)."""
    if svc.validation_results is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "Validation results not found. "
                "Run: python scripts/prepare_api_artifacts.py"
            ),
        )
    return svc.validation_results

@app.get("/health")
async def health() -> dict[str, Any]:
    svc = get_model_service()
    return {
        "status": "ok",
        "models": {
            "LightGBM": svc.lgbm   is not None,
            "ARIMA": svc.arima  is not None,
            "SARIMAX": svc.sarimax is not None,
        },
        "lambda_bc": svc.lambda_bc,
        "history_rows": len(svc.history) if svc.history is not None else 0,
    }

@app.post("/predict", response_model=PredictResponse)
async def predict(
    req: PredictRequest,
    svc: ModelService = Depends(_model_svc),
) -> PredictResponse:
    weather = _weather_dict(req.weather)

    try:
        points, lowers, uppers = svc.predict(
            req.model_name, weather, req.forecast_date, req.horizon, req.station_id
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction error")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    forecasts = []
    for i, pm10 in enumerate(points):
        dt = req.forecast_date + timedelta(days=i)
        forecasts.append(DayForecast(
            date=str(dt),
            pm10=round(pm10, 2),
            pm10_lower=round(lowers[i], 2) if lowers else None,
            pm10_upper=round(uppers[i], 2) if uppers else None,
        ))

    trend = _trend_label(points)
    cluster = svc.classify_regime(weather, lag1=None)
    regime = REGIME_LABELS.get(cluster, "Moderate")
    level = _pm10_level(points[0])

    return PredictResponse(
        model_name=req.model_name,
        forecasts=forecasts,
        trend=trend,
        regime=regime,
        pm10_level=level,
        regime_color=REGIME_COLORS.get(regime, "#888888"),
    )

@app.get("/metrics", response_model=MetricsResponse)
async def metrics(
    svc: ModelService = Depends(_model_svc),
) -> MetricsResponse:
    if not svc.metrics:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Metrics are not available. "
                "Run `python scripts/prepare_api_artifacts.py` to generate "
                "metrics.pkl before querying /metrics."
            ),
        )

    result: dict[str, ModelMetrics] = {}
    for name, m in svc.metrics.items():
        result[name] = ModelMetrics(
            mae=m["mae"],
            rmse=m["rmse"],
            smape=m["smape"],
            r2=m.get("r2"),
            exc_precision=m.get("exc_precision"),
            exc_recall=m.get("exc_recall"),
            exc_f1=m.get("exc_f1"),
        )

    best = min(result, key=lambda k: result[k].mae)
    return MetricsResponse(metrics=result, best_model=best)

@app.post("/explain", response_model=ExplainResponse)
async def explain(
    req: ExplainRequest,
    svc: ModelService = Depends(_model_svc),
    ex_svc: ExplainabilityService = Depends(_explain_svc),
) -> ExplainResponse:
    if req.model_name != "LightGBM":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="SHAP explainability is only supported for LightGBM.",
        )
    if svc.lgbm is None:
        raise HTTPException(status_code=503, detail="LightGBM model not loaded.")

    weather = _weather_dict(req.weather)
    dt = import_pd().Timestamp(req.forecast_date)
    station_history = svc.get_station_history(req.station_id)
    X  = _compute_lgbm_features(weather, dt, station_history, svc.lambda_bc)

    for col in svc.lgbm.feature_name_:
        if col not in X.columns:
            X[col] = 0.0
    X = X[svc.lgbm.feature_name_].fillna(0.0)

    contribs, base = ex_svc.explain(X)
    return ExplainResponse(
        model_name=req.model_name,
        contributions=[FeatureContribution(**c) for c in contribs],
        base_value=base,
    )

@app.post("/interpret", response_model=InterpretResponse)
async def interpret(
    req: InterpretRequest,
    svc: ModelService = Depends(_model_svc),
    ex_svc: ExplainabilityService = Depends(_explain_svc),
    int_svc: InterpretabilityService = Depends(_interpret_svc),
) -> InterpretResponse:
    weather = _weather_dict(req.weather)
    level = _pm10_level(req.pm10_forecast)

    contributions: list[dict] = []
    if svc.lgbm is not None:
        try:
            dt = import_pd().Timestamp(req.forecast_date)
            station_history = svc.get_station_history(req.station_id)
            X  = _compute_lgbm_features(weather, dt, station_history, svc.lambda_bc)
            for col in svc.lgbm.feature_name_:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[svc.lgbm.feature_name_].fillna(0.0)
            contributions, _ = ex_svc.explain(X)
        except Exception:
            pass

    result = int_svc.interpret(
        pm10=req.pm10_forecast,
        pm10_level=level,
        regime=req.regime,
        feature_contributions=contributions,
        forecast_date=req.forecast_date,
        weather=weather,
    )

    return InterpretResponse(**result)

def import_pd():
    import pandas as pd
    return pd