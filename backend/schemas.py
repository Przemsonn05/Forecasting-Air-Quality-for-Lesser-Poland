"""
Pydantic request/response schemas for the FastAPI backend.
All numeric fields include sensible validation bounds for Kraków's climate.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from config.config import STATIONS_META

class WeatherInput(BaseModel):
    """Direct meteorological observations provided by the caller."""
    temp_avg:    float = Field(..., ge=-30, le=45, description="Mean temperature [°C]")
    wind_max:    float = Field(..., ge=0, le=60, description="Maximum wind speed [m/s]")
    wind_mean:   float = Field(..., ge=0, le=40, description="Mean wind speed [m/s]")
    humidity_avg:float = Field(..., ge=0, le=100, description="Mean relative humidity [%]")
    pressure_avg:float = Field(..., ge=950, le=1050,description="Mean surface pressure [hPa]")
    rain_sum:    float = Field(0.0, ge=0, le=200, description="Total precipitation [mm]")
    snowfall_sum:float = Field(0.0, ge=0, le=100, description="Total snowfall [cm]")
    temp_min:    Optional[float] = Field(None, ge=-40, le=45)
    temp_max:    Optional[float] = Field(None, ge=-30, le=50)

_ALLOWED_STATIONS: frozenset[str] = frozenset(STATIONS_META.keys())

class PredictRequest(BaseModel):
    model_name:   str = Field(..., description="LightGBM | SARIMAX | ARIMA")
    forecast_date:date = Field(..., description="Date to forecast (YYYY-MM-DD)")
    weather:      WeatherInput
    horizon: int = Field(1, ge=1, le=3, description="Days ahead to forecast")
    station_id: str = Field("MpKrakWadow", description="Station identifier")

    @field_validator("model_name")
    @classmethod
    def validate_model(cls, v: str) -> str:
        allowed = {"LightGBM", "SARIMAX", "ARIMA"}
        if v not in allowed:
            raise ValueError(f"model_name must be one of {allowed}")
        return v

    @field_validator("station_id")
    @classmethod
    def validate_station(cls, v: str) -> str:
        if v not in _ALLOWED_STATIONS:
            raise ValueError(f"station_id must be one of {_ALLOWED_STATIONS}")
        return v

class DayForecast(BaseModel):
    date: str
    pm10: float
    pm10_lower:Optional[float] = None
    pm10_upper:Optional[float] = None


class PredictResponse(BaseModel):
    model_name: str
    forecasts: list[DayForecast]
    trend: str  
    regime: str  
    pm10_level: str    
    regime_color:str

class ModelMetrics(BaseModel):
    mae: float
    rmse: float
    smape: float
    r2: Optional[float] = None
    exc_precision: Optional[float] = None
    exc_recall: Optional[float] = None
    exc_f1: Optional[float] = None


class MetricsResponse(BaseModel):
    metrics: dict[str, ModelMetrics]
    best_model: str

class ExplainRequest(BaseModel):
    model_name: str
    forecast_date:date
    weather: WeatherInput
    station_id: str = "MpKrakWadow"

class FeatureContribution(BaseModel):
    feature: str
    value: float  
    contribution: float  

class ExplainResponse(BaseModel):
    model_name: str
    contributions:list[FeatureContribution]  
    base_value: Optional[float] = None     

class InterpretRequest(BaseModel):
    model_name: str
    forecast_date:date
    weather: WeatherInput
    pm10_forecast:float
    regime: str
    station_id: str = "MpKrakWadow"

class InterpretResponse(BaseModel):
    summary: str  
    risk_level: str
    key_drivers: list[str]
    recommendation:str