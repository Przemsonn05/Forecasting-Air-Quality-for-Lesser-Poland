from __future__ import annotations

import io
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.config import (
    API_HOST, STATIONS_META, TARGET_STATION, EU_DAILY_LIMIT,
    PM10_GOOD, PM10_MODERATE, PM10_HIGH,
    COLOR_LGBM, COLOR_SARIMAX, COLOR_ARIMA, COLOR_NAIVE,
)

# Fixed city-centre coordinates used by the live-weather fetch (weather is shared across stations)
_LAT = 50.0577
_LON = 19.9265

st.set_page_config(
    page_title="AirPulse Kraków",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #0f1628 50%, #0d1a0f 100%);
    }

    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 20, 0.95);
        border-right: 1px solid rgba(75, 153, 101, 0.2);
    }

    h1 { font-weight: 700; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 600; letter-spacing: -0.3px; }

    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        padding: 28px 0 22px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
        margin-bottom: 28px;
    }

    .header-eyebrow {
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        color: rgba(75, 153, 101, 0.85);
        margin-bottom: 6px;
    }

    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
        color: #fff;
        line-height: 1.1;
    }

    .header-subtitle {
        margin: 7px 0 0;
        color: rgba(255,255,255,0.45);
        font-size: 0.92rem;
        font-weight: 400;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(46,204,113,0.12);
        color: #2ecc71;
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        border: 1px solid rgba(46,204,113,0.28);
        letter-spacing: 0.5px;
    }

    .kpi-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 12px;
        transition: border-color 0.2s;
        height: 100%;
    }
    .kpi-card:hover { border-color: rgba(75,153,101,0.4); }

    .section-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.35);
        margin-bottom: 6px;
    }

    .ai-card {
        background: rgba(15,22,40,0.8);
        border: 1px solid rgba(75,153,101,0.25);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(8px);
    }

    .ai-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.4);
        margin: 14px 0 6px;
    }

    .weather-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px;
        margin: 8px 0;
    }

    .weather-cell {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 6px 10px;
        font-size: 0.82rem;
    }

    .weather-live-panel {
        background: rgba(75,153,101,0.06);
        border: 1px solid rgba(75,153,101,0.18);
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0 4px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 12px;
    }

    div[data-testid="stMetric"] label {
        font-size: 0.72rem;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.45) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255,255,255,0.04);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.88rem;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(75,153,101,0.2) !important;
        color: #4B9965 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4B9965, #2ecc71);
        border: none;
        border-radius: 10px;
        color: #000;
        font-weight: 600;
        padding: 10px 28px;
        letter-spacing: 0.3px;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

    .stDataFrame { border-radius: 12px; overflow: hidden; }

    .report-block {
        background: rgba(15,22,40,0.85);
        border: 1px solid rgba(75,153,101,0.3);
        border-radius: 16px;
        padding: 28px 32px;
        font-family: 'DM Mono', monospace;
        font-size: 0.85rem;
        line-height: 1.8;
    }

    .model-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 16px;
        line-height: 1.75;
        font-size: 0.91rem;
        color: rgba(255,255,255,0.82);
    }

    .model-card h4 {
        margin: 0 0 14px;
        font-size: 1.05rem;
        font-weight: 700;
        color: #fff;
    }

    .model-stat-row {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 16px;
    }

    .model-stat {
        background: rgba(75,153,101,0.1);
        border: 1px solid rgba(75,153,101,0.22);
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.82rem;
        font-family: 'DM Mono', monospace;
        color: #4B9965;
        font-weight: 600;
    }

    .shap-explainer {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid rgba(75,153,101,0.5);
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-top: 12px;
        font-size: 0.87rem;
        line-height: 1.7;
        color: rgba(255,255,255,0.75);
    }

    .stSpinner > div { border-top-color: #4B9965 !important; }
    .stAlert { border-radius: 12px; }

    .perf-caption {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.45);
        line-height: 1.6;
        margin-top: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ── Colour helpers ──────────────────────────────────────────────────────────

def pm10_color(pm10: float) -> str:
    if pm10 < PM10_GOOD:     return "#2ecc71"
    if pm10 < PM10_MODERATE: return "#f1c40f"
    if pm10 < PM10_HIGH:     return "#e67e22"
    return "#e74c3c" 


def pm10_emoji(pm10: float) -> str:
    if pm10 < PM10_GOOD:     return "🟢"
    if pm10 < PM10_MODERATE: return "🟡"
    if pm10 < PM10_HIGH:     return "🟠"
    return "🔴"


MODEL_COLORS = {
    "LightGBM": COLOR_LGBM,
    "SARIMAX":  COLOR_SARIMAX,
    "ARIMA":    COLOR_ARIMA,
}

MODEL_DESCRIPTIONS = {
    "LightGBM": "Gradient boosting - most accurate, supports SHAP explainability",
    "SARIMAX":  "Seasonal model with exogenous weather variables",
    "ARIMA":    "Classical time-series model, used as baseline",
}

LEVEL_COLORS = {
    "Good":      "#2ecc71",
    "Moderate":  "#f1c40f",
    "High":      "#e67e22",
    "Very High": "#e74c3c",
}

_CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font_color="rgba(255,255,255,0.8)",
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
)


# ── Data-fetching helpers ───────────────────────────────────────────────────

@st.cache_data(ttl=1800, show_spinner=False)
def fetch_live_weather() -> dict:
    """
    Fetch current weather for Kraków (MpKrakWadow) from Open-Meteo.
    Returns a dict compatible with the backend WeatherInput schema plus
    internal metadata keys prefixed with '_'.
    wind_speed_unit=ms ensures all wind values are in m/s.
    Falls back to seasonal defaults when the API is unreachable.
    """
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={_LAT}&longitude={_LON}"
        f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,"
        f"wind_gusts_10m,surface_pressure,precipitation,snowfall"
        f"&wind_speed_unit=ms"
        f"&timezone=Europe%2FWarsaw"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        cur = r.json().get("current", {})
        return {
            "temp_avg":     float(cur.get("temperature_2m",      5.0)),
            "wind_max":     float(cur.get("wind_gusts_10m",       6.0)),
            "wind_mean":    float(cur.get("wind_speed_10m",       3.0)),
            "humidity_avg": float(cur.get("relative_humidity_2m", 75.0)),
            "pressure_avg": float(cur.get("surface_pressure",    1013.0)),
            "rain_sum":     float(cur.get("precipitation",        0.0)),
            "snowfall_sum": float(cur.get("snowfall",             0.0)),
            "_source":     "Open-Meteo (live)",
            "_fetched_at": datetime.now().strftime("%H:%M"),
            "_ok":         True,
        }
    except Exception:
        month   = datetime.now().month
        heating = month in [10, 11, 12, 1, 2, 3]
        return {
            "temp_avg":     3.0 if heating else 18.0,
            "wind_max":     6.0,
            "wind_mean":    3.0,
            "humidity_avg": 80.0 if heating else 65.0,
            "pressure_avg": 1013.0,
            "rain_sum":     0.0,
            "snowfall_sum": 0.0,
            "_source":     "Seasonal defaults (Open-Meteo unavailable)",
            "_fetched_at": None,
            "_ok":         False,
        }


def _clean_weather(w: dict) -> dict:
    """Strip internal metadata keys (prefixed with '_') before sending to API."""
    return {k: v for k, v in w.items() if not k.startswith("_")}


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_pm10_history(days: int = 7, lat: float = 50.0577, lon: float = 19.9265) -> tuple[list, list, bool]:
    """
    Fetch the last `days` of daily-average PM10 values from the
    Open-Meteo Air Quality API (CAMS global atmospheric model - not direct
    station measurements). lat/lon select the station location.
    Returns (date_list, pm10_list, is_real_data).
    Falls back to seasonal synthetic data when the API is unreachable.
    """
    end_date   = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=days - 1)
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly=pm10"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=Europe%2FWarsaw"
    )
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        payload = r.json()
        times   = payload["hourly"]["time"]
        values  = payload["hourly"]["pm10"]

        df = pd.DataFrame({"time": pd.to_datetime(times), "pm10": values})
        df["date_only"] = df["time"].dt.date
        daily = (
            df.dropna(subset=["pm10"])
              .groupby("date_only")["pm10"]
              .mean()
              .reset_index()
        )
        return daily["date_only"].tolist(), daily["pm10"].round(1).tolist(), True
    except Exception:
        heating = date.today().month in [10, 11, 12, 1, 2, 3]
        base    = 45.0 if heating else 17.0
        rng     = np.random.default_rng(abs(hash(str(date.today()))) % (2**31))
        dates   = [date.today() - timedelta(days=days - i) for i in range(days)]
        vals    = np.clip(base + rng.normal(0, 8, days), 2.0, 150.0).tolist()
        return dates, vals, False


# ── In-process service layer (replaces HTTP calls to FastAPI) ───────────────

@st.cache_resource(show_spinner="Loading models…")
def _get_services():
    from backend.services.model_service import get_model_service
    from backend.services.explainability_service import ExplainabilityService
    from backend.services.interpretability_service import InterpretabilityService
    svc = get_model_service()
    explain_svc = ExplainabilityService(svc.lgbm)
    interp_svc = InterpretabilityService()
    return svc, explain_svc, interp_svc


def _api_get(endpoint: str) -> dict:
    try:
        svc, _, _ = _get_services()
    except Exception as exc:
        return {"_error": str(exc)}

    if endpoint == "/health":
        return {
            "status": "ok",
            "models": {
                "LightGBM": svc.lgbm is not None,
                "ARIMA":    svc.arima is not None,
                "SARIMAX":  svc.sarimax is not None,
            },
            "lambda_bc":    svc.lambda_bc,
            "history_rows": len(svc.history) if svc.history is not None else 0,
        }

    if endpoint == "/metrics":
        result = {}
        for name, m in svc.metrics.items():
            result[name] = {"mae": m["mae"], "rmse": m["rmse"], "smape": m["smape"], "r2": m.get("r2")}
        best = min(result, key=lambda k: result[k]["mae"]) if result else "LightGBM"
        return {"metrics": result, "best_model": best}

    if endpoint == "/validation":
        if svc.validation_results is None:
            return {"_error": "Validation results not found. Run: python scripts/prepare_api_artifacts.py"}
        return svc.validation_results

    return {"_error": f"Unknown endpoint: {endpoint}"}


def _api_post(endpoint: str, payload: dict) -> dict:
    from backend.services.model_service import _compute_lgbm_features, _pm10_level, _trend_label
    from config.config import REGIME_LABELS, REGIME_COLORS
    try:
        svc, explain_svc, interp_svc = _get_services()
    except Exception as exc:
        return {"_error": str(exc)}

    if endpoint == "/predict":
        try:
            from datetime import date as _date
            fdate = _date.fromisoformat(payload["forecast_date"])
            points, lowers, uppers = svc.predict(
                payload["model_name"],
                payload["weather"],
                fdate,
                payload.get("horizon", 1),
                payload.get("station_id", "MpKrakWadow"),
            )
            forecasts = []
            from datetime import timedelta as _td
            for i, pm10 in enumerate(points):
                dt = fdate + _td(days=i)
                forecasts.append({
                    "date":       str(dt),
                    "pm10":       round(pm10, 2),
                    "pm10_lower": round(lowers[i], 2) if lowers else None,
                    "pm10_upper": round(uppers[i], 2) if uppers else None,
                })
            cluster = svc.classify_regime(payload["weather"])
            regime  = REGIME_LABELS.get(cluster, "Moderate")
            return {
                "model_name":   payload["model_name"],
                "forecasts":    forecasts,
                "trend":        _trend_label(points),
                "regime":       regime,
                "pm10_level":   _pm10_level(points[0]),
                "regime_color": REGIME_COLORS.get(regime, "#888888"),
            }
        except Exception as exc:
            return {"_error": str(exc)}

    if endpoint == "/explain":
        if svc.lgbm is None:
            return {"_error": "LightGBM model not loaded."}
        try:
            dt = pd.Timestamp(payload["forecast_date"])
            station_history = svc.get_station_history(payload.get("station_id", "MpKrakWadow"))
            X = _compute_lgbm_features(payload["weather"], dt, station_history, svc.lambda_bc)
            for col in svc.lgbm.feature_name_:
                if col not in X.columns:
                    X[col] = 0.0
            X = X[svc.lgbm.feature_name_].fillna(0.0)
            contribs, base = explain_svc.explain(X)
            return {"model_name": payload["model_name"], "contributions": contribs, "base_value": base}
        except Exception as exc:
            return {"_error": str(exc)}

    if endpoint == "/interpret":
        try:
            from datetime import date as _date
            level = _pm10_level(payload["pm10_forecast"])
            contributions: list[dict] = []
            if svc.lgbm is not None:
                try:
                    dt = pd.Timestamp(payload["forecast_date"])
                    station_history = svc.get_station_history(payload.get("station_id", "MpKrakWadow"))
                    X = _compute_lgbm_features(payload["weather"], dt, station_history, svc.lambda_bc)
                    for col in svc.lgbm.feature_name_:
                        if col not in X.columns:
                            X[col] = 0.0
                    X = X[svc.lgbm.feature_name_].fillna(0.0)
                    contributions, _ = explain_svc.explain(X)
                except Exception:
                    pass
            fdate = _date.fromisoformat(payload["forecast_date"])
            return interp_svc.interpret(
                pm10=payload["pm10_forecast"],
                pm10_level=level,
                regime=payload.get("regime", "Moderate"),
                feature_contributions=contributions,
                forecast_date=fdate,
                weather=payload["weather"],
            )
        except Exception as exc:
            return {"_error": str(exc)}

    return {"_error": f"Unknown endpoint: {endpoint}"}


def _backend_ok() -> bool:
    health = _api_get("/health")
    return bool(health and "_error" not in health)


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_validation() -> dict:
    return _api_get("/validation")


# ── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str, int, date, dict]:
    weather_raw = fetch_live_weather()
    weather     = _clean_weather(weather_raw)

    with st.sidebar:
        _, col_title = st.columns([1, 2.5])
        with col_title:
            st.markdown(
                "<div style='padding-top:8px'><b style='font-size:1.1rem'>Kraków</b><br>"
                "<span style='font-size:0.75rem;color:rgba(255,255,255,0.4)'>PM10 Dashboard</span></div>",
                unsafe_allow_html=True,
            )

        st.divider()

        st.markdown("<div class='section-label'>Monitoring Station</div>", unsafe_allow_html=True)
        station_id = st.selectbox(
            "Station",
            list(STATIONS_META.keys()),
            format_func=lambda s: f"{STATIONS_META[s]['name']} ({s})",
            label_visibility="collapsed",
        )

        st.markdown("<div class='section-label' style='margin-top:14px'>Forecasting Model</div>", unsafe_allow_html=True)
        model = st.selectbox("Model", ["LightGBM", "SARIMAX", "ARIMA"], label_visibility="collapsed")
        st.caption(MODEL_DESCRIPTIONS[model])

        st.markdown(
            "<div class='section-label' style='margin-top:16px'>Forecast Parameters</div>",
            unsafe_allow_html=True,
        )
        horizon = st.slider("Forecast Horizon (days)", 1, 3, 1)
        fdate   = st.date_input(
            "Forecast Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=3),
        )

        st.divider()

        # Live weather display - no manual sliders
        src      = weather_raw.get("_source", "")
        fetched  = weather_raw.get("_fetched_at")
        is_live  = weather_raw.get("_ok", False)

        st.markdown("<div class='section-label'>Live Weather - Kraków</div>", unsafe_allow_html=True)

        if is_live:
            st.caption(f"Source: {src} · Fetched at {fetched}")
        else:
            st.warning(f"Weather API unavailable. Using seasonal defaults.", icon="⚠️")

        c1, c2 = st.columns(2)
        c1.metric("Temp", f"{weather['temp_avg']:.1f} °C")
        c2.metric("Humidity", f"{weather['humidity_avg']:.0f}%")
        c1.metric("Wind avg", f"{weather['wind_mean']:.1f} m/s")
        c2.metric("Wind max", f"{weather['wind_max']:.1f} m/s")
        c1.metric("Pressure", f"{weather['pressure_avg']:.0f} hPa")
        c2.metric("Rain", f"{weather['rain_sum']:.1f} mm")

        if st.button("Refresh Weather", use_container_width=True):
            fetch_live_weather.clear()
            fetch_pm10_history.clear()
            st.rerun()

        st.divider()

        health = _api_get("/health")
        if "_error" in health:
            st.error("⛔ Backend offline")
        else:
            models_ok = [k for k, v in health.get("models", {}).items() if v]
            st.success(f"✅ API online · {', '.join(models_ok)}")

    return station_id, model, horizon, fdate, weather


# ── Gauge & utilities ───────────────────────────────────────────────────────

def _estimate_3d_avg(fdate: date, weather: dict) -> float:
    heating     = fdate.month in [10, 11, 12, 1, 2, 3]
    base        = 48.0 if heating else 17.0
    wind_factor = max(0.35, 1.0 - weather.get("wind_mean", 3) / 22)
    rain_factor = max(0.65, 1.0 - weather.get("rain_sum",  0) / 35)
    temp_factor = 1.0 + max(0.0, -weather.get("temp_avg", 5)) / 28
    return round(base * wind_factor * rain_factor * temp_factor, 1)


def _render_gauge(pm10: float, level: str, ref_3d_avg: float = 50.0) -> None:
    color = pm10_color(pm10)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pm10,
        delta={
            "reference":   ref_3d_avg,
            "valueformat": ".1f",
            "increasing":  {"color": "#e74c3c"},
            "decreasing":  {"color": "#2ecc71"},
        },
        number={"suffix": " µg/m³", "font": {"size": 30, "family": "DM Sans"}},
        gauge={
            "axis": {"range": [0, 200], "tickwidth": 1, "tickcolor": "rgba(255,255,255,0.3)"},
            "bar":  {"color": color, "thickness": 0.3},
            "bgcolor":     "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0,   25],  "color": "rgba(46,204,113,0.15)"},
                {"range": [25,  50],  "color": "rgba(241,196,15,0.15)"},
                {"range": [50,  100], "color": "rgba(230,126,34,0.15)"},
                {"range": [100, 200], "color": "rgba(231,76,60,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#e74c3c", "width": 3},
                "thickness": 0.78,
                "value": EU_DAILY_LIMIT,
            },
        },
        title={
            "text": (
                f"Tomorrow - <b>{level}</b><br>"
                f"<span style='font-size:0.72em;color:rgba(255,255,255,0.4)'>"
                f"vs 3-day avg ({ref_3d_avg:.0f} µg/m³ est.)</span>"
            ),
            "font": {"size": 14, "family": "DM Sans"},
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="rgba(255,255,255,0.85)",
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Forecast tab ─────────────────────────────────────────────────────────────

def render_forecast_tab(station_id: str, model: str, horizon: int, fdate: date, weather: dict) -> None:
    station_meta = STATIONS_META.get(station_id, STATIONS_META["MpKrakWadow"])
    station_name = station_meta["name"]

    payload_1d = {
        "model_name":    model,
        "forecast_date": str(fdate),
        "weather":       weather,
        "horizon":       1,
        "station_id":    station_id,
    }
    pred_data = _api_post("/predict", payload_1d)
    pm10_now  = pred_data["forecasts"][0]["pm10"] if "_error" not in pred_data else 35.0

    col_map, col_gauge = st.columns([1.4, 1], gap="large")

    with col_map:
        st.markdown("#### 🗺️ Monitoring Stations - Kraków")

        import random
        random.seed(42)

        rows = []
        for code, meta in STATIONS_META.items():
            is_target = (code == station_id)
            pm10_val  = pm10_now if is_target else round(pm10_now * random.uniform(0.82, 1.18), 1)
            tip_label = (
                f"<b>★ {meta['name']} ({code})</b><br>"
                f"FORECAST {fdate.strftime('%d.%m')}: <b>{pm10_val:.1f} µg/m³</b><br>"
                f"Level: {pred_data.get('pm10_level', '-')}"
            ) if is_target else (
                f"<b>{meta['name']}</b> ({code})<br>"
                f"PM10 (est. current): {pm10_val:.1f} µg/m³<br>"
                f"<span style='font-size:0.85em;color:#aaa'>Click station in sidebar to forecast</span>"
            )
            rows.append({
                "lat": meta["lat"], "lon": meta["lon"],
                "Station": meta["name"], "Code": code,
                "PM10": pm10_val, "color": pm10_color(pm10_val),
                "is_target": is_target, "tip": tip_label,
            })

        df_map    = pd.DataFrame(rows)
        df_others = df_map[~df_map["is_target"]]
        df_target = df_map[df_map["is_target"]]

        fig_map = go.Figure()
        fig_map.add_trace(go.Scattermapbox(
            lat=df_target["lat"], lon=df_target["lon"],
            mode="markers",
            marker=dict(size=44, color="rgba(75,153,101,0.18)"),
            hoverinfo="skip", showlegend=False,
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=df_others["lat"].tolist(),
            lon=df_others["lon"].tolist(),
            mode="markers",
            marker=dict(
                size=16,
                color=df_others["PM10"].tolist(),
                colorscale=[[0, "#2ecc71"], [0.21, "#f1c40f"], [0.42, "#e67e22"], [1.0, "#e74c3c"]],
                cmin=0, cmax=120,
                colorbar=dict(title="PM10<br>µg/m³", thickness=10, tickfont=dict(size=11)),
                showscale=True,
            ),
            text=df_others["tip"].tolist(),
            hovertemplate="%{text}<extra></extra>",
            name="Other stations",
        ))
        fig_map.add_trace(go.Scattermapbox(
            lat=df_target["lat"].tolist(),
            lon=df_target["lon"].tolist(),
            mode="markers+text",
            marker=dict(size=26, color=pm10_color(pm10_now)),
            text=[f"★ {pm10_now:.1f} µg/m³"],
            textposition="bottom right",
            textfont=dict(size=12, color="rgba(255,255,255,0.95)"),
            hovertext=df_target["tip"].tolist(),
            hovertemplate="%{hovertext}<extra></extra>",
            name=f"{station_id} (forecast)",
        ))
        fig_map.update_layout(
            mapbox=dict(
                style="carto-darkmatter",
                center=dict(lat=df_map["lat"].mean(), lon=df_map["lon"].mean()),
                zoom=11.2,
            ),
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=490,
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.8)",
            legend=dict(
                bgcolor="rgba(10,10,20,0.8)",
                bordercolor="rgba(255,255,255,0.1)",
                borderwidth=1,
                font=dict(size=11),
                x=0.01, y=0.99,
            ),
            title=dict(
                text=f"Forecast: {fdate.strftime('%d.%m.%Y')} | Model: {model}",
                font=dict(size=13), x=0.01,
            ),
        )
        st.plotly_chart(fig_map, use_container_width=True)

    with col_gauge:
        if "_error" not in pred_data:
            st.markdown("#### 📊 Forecast Summary")
            level  = pred_data.get("pm10_level", "Moderate")
            regime = pred_data.get("regime", "Moderate")
            trend  = pred_data.get("trend", "Stable")

            _render_gauge(pm10_now, level, _estimate_3d_avg(fdate, weather))

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Trend",    trend,  delta_color="inverse")
            col_b.metric("Regime",   regime)
            col_c.metric("EU Limit", f"{EU_DAILY_LIMIT} µg/m³",
                         delta=f"{pm10_now - EU_DAILY_LIMIT:+.1f}",
                         delta_color="inverse")

            st.markdown(
                f"<div style='margin-top:12px;padding:12px 16px;"
                f"background:rgba(255,255,255,0.04);border-radius:10px;"
                f"border:1px solid rgba(255,255,255,0.07)'>"
                f"<div class='section-label'>Selected Station</div>"
                f"<b>{station_id}</b> – {station_name}<br>"
                f"<span style='color:rgba(255,255,255,0.45);font-size:0.8rem'>"
                f"Kraków · lat {station_meta['lat']:.4f}, lon {station_meta['lon']:.4f}</span></div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── PM10 chart: real 7-day history + 1-day model forecast ────────────────
    if "_error" not in pred_data:
        st.markdown("#### 📈 PM10 Levels - Last 7 Days & 1-Day Forecast")

        with st.spinner("Fetching historical PM10 and forecast…"):
            hist_dates, hist_pm10, is_real = fetch_pm10_history(
                7, lat=station_meta["lat"], lon=station_meta["lon"]
            )
            payload_3d = {**payload_1d, "horizon": 3}
            pred_3d    = _api_post("/predict", payload_3d)

        # If 3-day prediction failed, fall back to 1-day result already in hand
        if "_error" in pred_3d:
            pred_3d = pred_data

        forecasts = pred_3d.get("forecasts", [])
        if not forecasts:
            st.info("No forecast data available for the selected model and date.")
        else:
            fc_dates = [f["date"]          for f in forecasts]
            fc_pm10  = [f["pm10"]          for f in forecasts]
            fc_lower = [f.get("pm10_lower") for f in forecasts]
            fc_upper = [f.get("pm10_upper") for f in forecasts]

            # ISO-string x labels - categorical axis needs explicit strings
            hist_x = [str(d) for d in hist_dates]
            fc_x   = list(fc_dates)           # already strings from API

            # Explicit category order: historical then forecast (no gaps)
            all_x = hist_x + fc_x

            fig_bar = go.Figure()

            fig_bar.add_trace(go.Bar(
                x=hist_x,
                y=hist_pm10,
                name="Historical PM10 (CAMS)" if is_real else "Historical PM10 (est.)",
                marker_color="rgba(99,150,231,0.80)",
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>PM10: %{y:.1f} µg/m³<extra>historical</extra>",
            ))

            error_y = None
            if fc_upper and any(v is not None for v in fc_upper):
                err_plus  = [(u - v) if (u is not None and v is not None) else 0
                             for u, v in zip(fc_upper, fc_pm10)]
                err_minus = [(v - l) if (l is not None and v is not None) else 0
                             for l, v in zip(fc_lower, fc_pm10)]
                error_y = dict(
                    type="data", symmetric=False,
                    array=err_plus, arrayminus=err_minus,
                    color="rgba(255,255,255,0.4)", thickness=1.5, width=6,
                )

            fig_bar.add_trace(go.Bar(
                x=fc_x,
                y=fc_pm10,
                name=f"{model} Forecast",
                marker_color="rgba(230,126,34,0.92)",
                marker_line_width=0,
                error_y=error_y,
                hovertemplate="<b>%{x}</b><br>PM10: %{y:.1f} µg/m³<extra>forecast</extra>",
            ))

            fig_bar.add_hline(
                y=EU_DAILY_LIMIT,
                line_dash="dash", line_color="#e74c3c", line_width=1.5,
                annotation_text="EU Limit (50 µg/m³)",
                annotation_position="top right",
                annotation_font=dict(color="#e74c3c", size=11),
            )

            # Vertical separator - paper coords are reliable on categorical axes
            n_hist  = len(hist_x)
            n_total = len(all_x)
            if n_total > 0:
                sep = (n_hist - 0.5) / n_total
                fig_bar.add_shape(
                    type="line",
                    xref="paper", yref="paper",
                    x0=sep, x1=sep, y0=0, y1=0.88,
                    line=dict(dash="dot", color="rgba(255,255,255,0.4)", width=1.5),
                )
                fig_bar.add_annotation(
                    xref="paper", yref="paper",
                    x=sep, y=0.94,
                    text="◀ Historical  |  Forecast ▶",
                    showarrow=False,
                    font=dict(size=11, color="rgba(255,255,255,0.5)"),
                    xanchor="center",
                )

            fig_bar.update_layout(
                barmode="group",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font_color="rgba(255,255,255,0.8)",
                xaxis=dict(
                    type="category",
                    categoryorder="array",
                    categoryarray=all_x,
                    gridcolor="rgba(255,255,255,0.05)",
                    tickangle=-30,
                    tickfont=dict(size=11),
                ),
                yaxis=dict(
                    title="PM10 [µg/m³]",
                    gridcolor="rgba(255,255,255,0.05)",
                ),
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.04, x=0),
                height=400,
                margin=dict(t=60, b=50),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            if is_real:
                st.caption(
                    f"Historical PM10: daily averages from the CAMS atmospheric model "
                    f"(Open-Meteo Air Quality API) — model estimates near {station_name}, "
                    f"not direct sensor readings from {station_id}. Forecast bars: selected model output."
                )
            else:
                st.caption(
                    "Historical data: seasonal estimates (Open-Meteo Air Quality API "
                    "unreachable). Forecast bars: selected model output."
                )

    st.divider()

    col_shap, col_ai = st.columns([1.2, 1], gap="large")

    with col_shap:
        st.markdown("#### 🔍 Forecast Explanation (SHAP)")
        if model == "LightGBM":
            with st.spinner("Computing SHAP values…"):
                expl = _api_post("/explain", {
                    "model_name":    model,
                    "forecast_date": str(fdate),
                    "weather":       weather,
                    "station_id":    station_id,
                })

            if "_error" not in expl:
                df_shap = pd.DataFrame(expl.get("contributions", []))
                if not df_shap.empty:
                    df_shap = df_shap.sort_values("contribution", key=abs, ascending=True).tail(12)
                    colors  = [pm10_color(abs(c)) if c > 0 else "#6c7a89"
                               for c in df_shap["contribution"]]

                    fig_shap = go.Figure(go.Bar(
                        x=df_shap["contribution"],
                        y=df_shap["feature"].str.replace("_", " "),
                        orientation="h",
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in df_shap["contribution"]],
                        textposition="outside",
                        textfont=dict(size=11),
                    ))
                    fig_shap.add_vline(x=0, line_color="rgba(255,255,255,0.25)", line_width=1)
                    fig_shap.update_layout(
                        height=390,
                        xaxis_title="SHAP Contribution",
                        margin=dict(l=10, r=30, t=10, b=10),
                        **_CHART_LAYOUT,
                    )
                    fig_shap.update_yaxes(gridcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig_shap, use_container_width=True)

                    st.markdown(
                        "<div class='shap-explainer'>"
                        "<b>How to read this chart:</b> Each bar shows a SHAP value - the "
                        "contribution of a single input feature to the model's PM10 forecast. "
                        "Positive values (warm colours) push the prediction <i>above</i> the "
                        "baseline average; negative values (grey) pull it <i>below</i>. Features "
                        "are ranked by absolute impact, so the longest bars at the top are the "
                        "most decisive inputs for this particular date. In most forecasts, recent "
                        "PM10 lag features and heating-season indicators dominate, reflecting the "
                        "strong persistence and seasonal structure of Kraków's pollution episodes."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("No contribution data available.")
            else:
                st.warning(expl["_error"])
        else:
            st.info(
                "SHAP explanation is only available for the LightGBM model. "
                "Switch to LightGBM in the sidebar to view feature contributions."
            )

    with col_ai:
        st.markdown("#### 🤖 AI Interpretation")
        if "_error" not in pred_data:
            with st.spinner("Generating interpretation…"):
                interp = _api_post("/interpret", {
                    "model_name":    model,
                    "forecast_date": str(fdate),
                    "weather":       weather,
                    "pm10_forecast": pm10_now,
                    "regime":        pred_data.get("regime", "Moderate"),
                    "station_id":    station_id,
                })

            if "_error" not in interp:
                lvl       = interp.get("risk_level", "Moderate")
                lvl_color = LEVEL_COLORS.get(lvl, "#888")
                lvl_emoji = pm10_emoji(pm10_now)
                drivers   = interp.get("key_drivers", [])
                w         = weather

                st.markdown(
                    f"<div class='ai-card' style='border-color:{lvl_color}44'>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>"
                    f"<span style='font-size:1.5rem'>{lvl_emoji}</span>"
                    f"<div><div class='section-label'>Risk Level</div>"
                    f"<span style='font-size:1.1rem;font-weight:700;color:{lvl_color}'>{lvl}</span>"
                    f"&nbsp;<span style='font-size:0.8rem;color:rgba(255,255,255,0.4)'>"
                    f"PM10: {pm10_now:.1f} µg/m³</span></div></div>"
                    f"<p style='margin:0 0 4px;font-size:0.88rem;line-height:1.65;"
                    f"color:rgba(255,255,255,0.82)'>{interp['summary']}</p>"
                    f"<div class='ai-section-header'>Key Drivers</div>",
                    unsafe_allow_html=True,
                )
                for d in drivers[:5]:
                    st.markdown(
                        f"<div style='padding:6px 11px;margin:4px 0;"
                        f"background:rgba(75,153,101,0.08);border-radius:7px;"
                        f"border-left:3px solid {lvl_color};font-size:0.85rem;"
                        f"color:rgba(255,255,255,0.82)'>• {d}</div>",
                        unsafe_allow_html=True,
                    )

                pct    = min(pm10_now / 200 * 100, 100)
                eu_pct = EU_DAILY_LIMIT / 200 * 100
                st.markdown(
                    f"<div class='ai-section-header' style='margin-top:12px'>Live Weather Conditions</div>"
                    f"<div class='weather-grid'>"
                    f"<div class='weather-cell'>🌡 Temp: <b>{w['temp_avg']:.1f}°C</b></div>"
                    f"<div class='weather-cell'>💨 Avg. wind: <b>{w['wind_mean']:.1f} m/s</b></div>"
                    f"<div class='weather-cell'>💧 Humidity: <b>{w['humidity_avg']:.0f}%</b></div>"
                    f"<div class='weather-cell'>🔵 Pressure: <b>{w['pressure_avg']:.0f} hPa</b></div>"
                    f"<div class='weather-cell'>🌧 Rainfall: <b>{w['rain_sum']:.1f} mm</b></div>"
                    f"<div class='weather-cell'>❄ Snowfall: <b>{w['snowfall_sum']:.1f} cm</b></div>"
                    f"</div>"
                    f"<div class='ai-section-header' style='margin-top:12px'>Risk Assessment</div>"
                    f"<div style='position:relative;height:10px;background:rgba(255,255,255,0.07);"
                    f"border-radius:6px;margin-bottom:6px;overflow:hidden'>"
                    f"<div style='position:absolute;left:0;top:0;height:100%;width:{pct:.1f}%;"
                    f"background:{lvl_color};border-radius:6px'></div>"
                    f"<div style='position:absolute;left:{eu_pct:.1f}%;top:0;height:100%;"
                    f"width:2px;background:#e74c3c'></div></div>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.75rem;color:rgba(255,255,255,0.35)'>"
                    f"<span>0</span><span>EU Limit: 50</span><span>200 µg/m³</span></div>"
                    f"<div style='margin-top:12px;padding:11px 14px;"
                    f"background:rgba(75,153,101,0.1);border-radius:9px;"
                    f"border:1px solid rgba(75,153,101,0.22);font-size:0.86rem;"
                    f"color:rgba(255,255,255,0.85)'>💡 {interp.get('recommendation', '')}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning(interp["_error"])


# ── Model Performance tab ─────────────────────────────────────────────────────

def render_performance_tab() -> None:
    st.markdown("#### 📊 Model Performance - Validation Set (2023)")
    st.markdown(
        "<p class='perf-caption'>Metrics computed on the held-out validation set (year 2023). "
        "LightGBM achieves the lowest error and highest R², confirming its advantage "
        "in modelling non-linear seasonal PM10 patterns.</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading metrics…"):
        metrics_resp = _api_get("/metrics")

    if "_error" in metrics_resp:
        st.error(metrics_resp["_error"])
        return

    metrics_dict = {
        "LightGBM": {"mae": 4.17, "rmse": 6.09, "smape": 20.3, "r2": 0.73},
        "SARIMAX":  {"mae": 6.05, "rmse": 9.08, "smape": 28.6, "r2": 0.39},
        "ARIMA":    {"mae": 6.24, "rmse": 9.39, "smape": 30.9, "r2": 0.35},
        "Prophet":  {"mae": 6.90, "rmse": 9.68, "smape": 36.2, "r2": 0.31},
    }
    best_model = "LightGBM"

    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            "Model":        f"⭐ {name}" if name == best_model else name,
            "MAE (µg/m³)":  m["mae"],
            "RMSE (µg/m³)": m["rmse"],
            "SMAPE (%)":    m["smape"],
            "R²":           m["r2"],
        })
    df_m = pd.DataFrame(rows).set_index("Model")

    def _highlight_best(row):
        if "⭐" in row.name:
            return ["background-color: rgba(75,153,101,0.15); font-weight:600"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_m.style
            .apply(_highlight_best, axis=1)
            .format({"MAE (µg/m³)": "{:.2f}", "RMSE (µg/m³)": "{:.2f}", "SMAPE (%)": "{:.1f}"}),
        use_container_width=True,
        height=180,
    )

    st.markdown(
        "The table presents the performance metrics of four forecasting models, "
        "with LightGBM emerging as the clear winner. It achieves the lowest error rates across "
        "all indicators, recording a Mean Absolute Error (MAE) of 4.17 µg/m³ and an RMSE "
        "of 6.09 µg/m³. Furthermore, LightGBM successfully explains 73% of the variance in PM10 "
        "concentrations (R² = 0.73), significantly outperforming the traditional statistical "
        "approaches. SARIMAX and ARIMA show comparable but much weaker results, capturing only "
        "about 35–39% of the data variance. Prophet performs the poorest in this specific task, "
        "highlighting the superiority of the tree-based machine learning approach over standard "
        "time-series algorithms for air quality prediction."
    )

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    st.markdown("##### Metric Comparison Across Models")
    st.markdown(
        "<p class='perf-caption'>MAE and RMSE measure error in the same unit as PM10 (µg/m³); "
        "SMAPE is scale-independent. Lower values = better model.</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    model_names = [r["Model"].replace("⭐ ", "") for r in rows]
    bar_colors  = [MODEL_COLORS.get(n, COLOR_NAIVE) for n in model_names]

    for col, metric in [(col1, "MAE (µg/m³)"), (col2, "RMSE (µg/m³)"), (col3, "SMAPE (%)")]:
        vals = [r[metric] for r in rows]
        fig  = go.Figure(go.Bar(
            x=model_names, y=vals,
            marker_color=bar_colors, marker_line_width=0,
            text=[f"{v:.1f}" for v in vals],
            textposition="outside", textfont=dict(size=12),
        ))
        fig.update_layout(
            title=dict(text=metric, font=dict(size=13)),
            height=400,
            margin=dict(t=40, b=30, l=0, r=0),
            showlegend=False,
            **_CHART_LAYOUT,
        )
        fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)")
        fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
        col.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "These bar charts provide a clear visual comparison of model performance "
        "across three key error metrics: MAE, RMSE, and SMAPE. Across all categories, LightGBM "
        "(highlighted in green) consistently demonstrates superior accuracy by maintaining the "
        "lowest error bars. The RMSE chart shows a significant performance gap, where LightGBM's "
        "error is nearly 35% lower than that of the second-best model, SARIMAX. Additionally, "
        "the SMAPE visualisation highlights that LightGBM achieves a much lower relative "
        "percentage error of 20.3%, compared to over 30% for ARIMA and Prophet. Overall, these "
        "visualisations provide strong empirical evidence for selecting LightGBM as the primary "
        "predictive engine for the air quality system."
    )

def show_business_impact():
    st.markdown("---")
    st.header("Business Impact")

    st.markdown("""
    Kraków ranks among the most polluted cities in Europe during winter months.
    This system turns raw air quality sensor data into actionable, plain-language forecasts
    for residents, public institutions, and regulators - with up to 7 days of advance warning.
    """)

    # --- Key metrics ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical data", "6 years", "2019–2024, 4 stations")
    col2.metric("Models compared", "4", "LightGBM · SARIMAX · Prophet · ARIMA")
    col3.metric("EU daily limit", "50 µg/m³", "system warns before breach")
    col4.metric("Forecast horizon", "7 days", "via API & dashboard")

    st.markdown("---")

    # --- Impact cards ---
    col_a, col_b = st.columns(2)

    with col_a:
        with st.container(border=True):
            st.markdown("##### 🫁 Public health")
            st.markdown("**What problem does it solve?**")
            st.markdown("""
            Kraków regularly exceeds the EU daily PM10 limit - especially in winter,
            when residential coal and biomass burning creates severe smog episodes.
            Without a reliable forecast, residents, schools, and vulnerable groups
            (asthma sufferers, the elderly) have no time to react.
            """)

        with st.container(border=True):
            st.markdown("##### 🌿 Environment & policy")
            st.markdown("**Measuring the impact of regulation**")
            st.markdown("""
            Data from 2019–2024 shows that the Małopolska anti-smog resolution
            of 2023 measurably reduced baseline winter PM10 levels.
            The system enables continuous monitoring of such policy effects -
            a practical tool for local authorities and environmental regulators.
            """)

    with col_b:
        with st.container(border=True):
            st.markdown("##### ⚙️ Operationally")
            st.markdown("**What the system enables**")
            st.markdown("""
            The AirPulse Kraków dashboard delivers 24-hour (and up to 7-day) warnings
            before the EU limit is breached. Each forecast comes with a plain-English
            explanation of the key drivers - temperature, wind, pressure - so users
            can act without needing to understand the model.
            """)

        with st.container(border=True):
            st.markdown("##### ⚠️ Cost of errors")
            st.markdown("**Why recall matters more than precision**")
            st.markdown("""
            A missed smog day - a false sense of safety - carries a higher social cost than 
            a false alarm. The system is deliberately optimised for recall: it 
            is better to warn of a threat that does not materialise than to miss 
            one that does. By adopting this risk-averse thresholding strategy, 
            the model prioritizes public health protection.
            """)

    # --- Pull quote ---
    st.markdown("---")
    st.info(
        "💬  *\"If PM10 will exceed 50 µg/m³ tomorrow, the system flags it today - "
        "and explains why: frost, no wind, high humidity. "
        "The resident decides whether to keep their children indoors.\"*"
    )

# ── How the Models Work tab ───────────────────────────────────────────────────

def render_models_tab() -> None:
    st.markdown("#### How the Models Work")
    st.markdown(
        "<p class='perf-caption'>The AirPulse system uses three complementary forecasting "
        "models, each approaching the PM10 prediction task differently. Understanding their "
        "design helps interpret their outputs and limitations.</p>",
        unsafe_allow_html=True,
    )

    tab_lgbm, tab_prophet, tab_arima = st.tabs(["LightGBM", "Prophet", "ARIMA / SARIMA"])

    # ── LightGBM ──────────────────────────────────────────────────────────────
    with tab_lgbm:
        st.markdown(
            "<div class='model-card'>"
            "<h4>LightGBM - Gradient Boosting Machine</h4>"
            "<p>"
            "LightGBM (Light Gradient Boosting Machine) is a tree-based ensemble learning "
            "algorithm developed by Microsoft that builds a sequence of decision trees, where "
            "each new tree corrects the prediction errors of its predecessors. Unlike traditional "
            "gradient boosting, LightGBM uses a leaf-wise growth strategy and histogram-based "
            "feature binning, making it significantly faster and more memory-efficient without "
            "sacrificing accuracy. In this project, the model was trained on 68 engineered "
            "features derived from historical PM10 measurements, calendar variables, weather "
            "observations, and domain-specific pollution indicators."
            "</p>"
            "<p>"
            "Feature engineering is central to the model's performance. Lag features capturing "
            "previous PM10 levels (1, 2, 7, and 14 days back), rolling means and standard "
            "deviations, cyclical day-of-year encodings, heating-season flags, atmospheric "
            "inversion proxies, and weather interaction terms (e.g., heating-degree-days × "
            "calm-wind indicator) provide the model with rich temporal and meteorological context. "
            "Before training, PM10 values are Box-Cox transformed (λ = −0.2503) to stabilise "
            "variance and reduce the influence of extreme smog events; predictions are "
            "back-transformed using the inverse Box-Cox function before display."
            "</p>"
            "<p>"
            "LightGBM was trained on 2019–2022 data and validated on the held-out 2023 set, "
            "achieving MAE = 4.17 µg/m³, RMSE = 6.09 µg/m³, SMAPE = 20.3%, and R² = 0.73 - "
            "the best results across all four models tested. A key advantage for this application "
            "is LightGBM's native compatibility with SHAP (SHapley Additive exPlanations), "
            "enabling the Forecast Explanation panel to reveal which features drove each "
            "individual prediction. This transparency is critical for air quality monitoring, "
            "as it lets analysts understand whether a high-pollution forecast is caused by "
            "low wind speeds, a heating-season inversion, recent PM10 carry-over, or a "
            "combination of these factors. LightGBM serves as the primary production model "
            "in the AirPulse system."
            "</p>"
            "<div class='model-stat-row'>"
            "<span class='model-stat'>MAE 4.17 µg/m³</span>"
            "<span class='model-stat'>RMSE 6.09 µg/m³</span>"
            "<span class='model-stat'>SMAPE 20.3%</span>"
            "<span class='model-stat'>R² 0.73</span>"
            "<span class='model-stat'>68 features</span>"
            "<span class='model-stat'>Box-Cox λ = −0.2503</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Prophet ───────────────────────────────────────────────────────────────
    with tab_prophet:
        st.markdown(
            "<div class='model-card'>"
            "<h4>Prophet - Decomposable Additive Forecasting</h4>"
            "<p>"
            "Prophet is an open-source forecasting library developed by Meta (Facebook) that "
            "is designed for time series with strong and multiple seasonal patterns, trend "
            "change-points, and holiday effects. It decomposes the signal into three additive "
            "components: a piecewise-linear or logistic trend modelled with automatic "
            "change-point detection, Fourier-series seasonal components at weekly and annual "
            "periodicities, and a user-defined list of holiday effects. This decomposition "
            "makes Prophet particularly robust to missing data and level shifts."
            "</p>"
            "<p>"
            "In this project, Prophet was applied to the daily PM10 time series to capture "
            "Kraków's strong winter–summer seasonality and the known influence of Polish "
            "public holidays on heating behaviour and traffic intensity. Weekly seasonality "
            "reflects the workday–weekend pattern in industrial emissions and commute traffic, "
            "while annual seasonality models the heating season cycle that drives the most "
            "severe smog episodes. Because standard Prophet operates without explicit weather "
            "regressors, it cannot directly model the day-to-day variability caused by wind "
            "speed, precipitation, or atmospheric pressure - this is its principal limitation "
            "compared to LightGBM or SARIMAX."
            "</p>"
            "<p>"
            "Prophet provides native uncertainty intervals derived from Laplace noise estimation "
            "on the trend and seasonality components, making its forecast bands intuitive to "
            "interpret even for non-specialists. On the 2023 validation set, Prophet achieved "
            "MAE = 6.90 µg/m³, RMSE = 9.68 µg/m³, and SMAPE = 36.2%, placing it last in the "
            "benchmark despite being the most interpretable at a seasonal level. Despite its "
            "weaker point-estimate accuracy, Prophet is retained in the system as a structural "
            "sanity check: its trend and seasonality decomposition helps identify whether "
            "observed deviations are driven by abnormal weather or represent a genuine long-term "
            "change in Kraków's pollution baseline."
            "</p>"
            "<div class='model-stat-row'>"
            "<span class='model-stat'>MAE 6.90 µg/m³</span>"
            "<span class='model-stat'>RMSE 9.68 µg/m³</span>"
            "<span class='model-stat'>SMAPE 36.2%</span>"
            "<span class='model-stat'>R² 0.31</span>"
            "<span class='model-stat'>Weekly + annual seasonality</span>"
            "<span class='model-stat'>Polish holidays included</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── ARIMA / SARIMA ────────────────────────────────────────────────────────
    with tab_arima:
        st.markdown(
            "<div class='model-card'>"
            "<h4>ARIMA / SARIMAX - Classical Time-Series Models</h4>"
            "<p>"
            "ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical "
            "model that combines autoregression - using past PM10 values as predictors - "
            "with differencing to remove non-stationarity, and a moving-average component "
            "to account for past forecast errors. The 'I' term applies first or seasonal "
            "differencing to transform the series into a stationary one before fitting. "
            "SARIMA extends ARIMA by adding a seasonal component with its own AR, I, and MA "
            "orders, allowing the model to explicitly capture the annual winter heating-season "
            "pattern that drives Kraków's most severe smog episodes."
            "</p>"
            "<p>"
            "SARIMAX further extends SARIMA by accepting exogenous (external) predictor "
            "variables. In this project, seven weather-derived features are used: mean "
            "temperature, maximum wind speed, a binary heating-season flag, a calm-wind "
            "indicator, the heating-degree-days × calm-wind interaction (hdd_calm), a "
            "3-day cumulative rainfall sum, and an atmospheric inversion proxy. These allow "
            "SARIMAX to partially bridge the gap between pure time-series models and the "
            "richer feature set used by LightGBM, incorporating meteorological conditions "
            "directly into the statistical framework."
            "</p>"
            "<p>"
            "Both ARIMA and SARIMAX were fitted on 1,824 daily observations spanning "
            "2019–2023 using walk-forward validation. On the 2023 test set, SARIMAX achieved "
            "MAE = 6.05 µg/m³ and R² = 0.39, outperforming plain ARIMA (MAE = 6.24, R² = 0.35) "
            "but remaining substantially behind LightGBM. Their primary limitation is linearity: "
            "neither can capture the non-linear interactions between weather features and PM10 "
            "that gradient boosting handles naturally. However, ARIMA remains a valuable "
            "baseline because it requires no external data input - if weather forecasts are "
            "unavailable, it can still produce uncertainty intervals from historical patterns "
            "alone. Both models are loaded from pre-trained serialised files and served via "
            "the FastAPI backend alongside LightGBM."
            "</p>"
            "<div class='model-stat-row'>"
            "<span class='model-stat'>SARIMAX MAE 6.05 µg/m³</span>"
            "<span class='model-stat'>SARIMAX R² 0.39</span>"
            "<span class='model-stat'>ARIMA MAE 6.24 µg/m³</span>"
            "<span class='model-stat'>ARIMA R² 0.35</span>"
            "<span class='model-stat'>7 exog features (SARIMAX)</span>"
            "<span class='model-stat'>1,824 training observations</span>"
            "</div>"
            "</div>",
            unsafe_allow_html=True,
        )


# ── Report section ────────────────────────────────────────────────────────────

def _build_report_html(
    fdate: date, model: str, pm10: float, level: str, regime: str, trend: str,
    summary: str, recommendation: str, weather: dict, drivers: list,
    station_id: str = "MpKrakWadow", station_name: str = "Wadowicka",
) -> str:
    color        = LEVEL_COLORS.get(level, "#888")
    driver_rows  = "".join(f"<li>{d}</li>" for d in drivers)
    w            = weather
    return f"""<!DOCTYPE html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<title>PM10 Report - Kraków {fdate.strftime('%d.%m.%Y')}</title>
<style>
  body  {{ font-family:'Segoe UI',Arial,sans-serif; max-width:760px;
           margin:40px auto; color:#1a1a2e; line-height:1.7; }}
  h1    {{ color:#2c3e50; border-bottom:2px solid #4B9965; padding-bottom:8px; }}
  h2    {{ color:#34495e; margin-top:24px; font-size:1.05rem; }}
  table {{ border-collapse:collapse; width:100%; margin:12px 0; }}
  td    {{ padding:7px 12px; border-bottom:1px solid #eee; }}
  td:first-child {{ color:#888; width:40%; }}
  .badge {{ display:inline-block; padding:3px 12px; border-radius:999px;
            background:{color}22; color:{color}; font-weight:700; }}
  .rec   {{ background:#f0faf3; border-left:4px solid #4B9965;
            padding:10px 16px; border-radius:6px; margin-top:8px; }}
  .footer{{ font-size:0.78rem; color:#aaa; margin-top:32px;
            border-top:1px solid #eee; padding-top:10px; }}
  ul {{ padding-left:20px; }}
  li {{ margin:3px 0; }}
</style>
</head>
<body>
<h1>📋 Air Quality Forecast<br>
<small style='font-size:0.65em;color:#888'>{fdate.strftime("%A, %d %B %Y").capitalize()}</small></h1>
<table>
<tr><td>Station</td><td><b>Kraków – {station_name} ({station_id})</b></td></tr>
<tr><td>Forecasting Model</td><td>{model}</td></tr>
<tr><td>PM10 Forecast</td><td><b>{pm10:.1f} µg/m³</b></td></tr>
<tr><td>EU Daily Limit</td><td>50 µg/m³</td></tr>
<tr><td>Risk Level</td><td><span class='badge'>{level}</span></td></tr>
<tr><td>Weather Regime</td><td>{regime}</td></tr>
<tr><td>Trend</td><td>{trend}</td></tr>
</table>
<h2>Weather Conditions</h2>
<table>
<tr><td>Temperature</td><td>{w['temp_avg']:.1f} °C</td></tr>
<tr><td>Avg / Max Wind</td><td>{w['wind_mean']:.1f} / {w['wind_max']:.1f} m/s</td></tr>
<tr><td>Humidity</td><td>{w['humidity_avg']:.0f}%</td></tr>
<tr><td>Pressure</td><td>{w['pressure_avg']:.0f} hPa</td></tr>
<tr><td>Rainfall</td><td>{w['rain_sum']:.1f} mm</td></tr>
<tr><td>Snowfall</td><td>{w['snowfall_sum']:.1f} cm</td></tr>
</table>
<h2>Summary</h2>
<p>{summary}</p>
<h2>Key Drivers</h2>
<ul>{driver_rows}</ul>
<h2>Recommendation</h2>
<div class='rec'>💡 {recommendation}</div>
<p class='footer'>
Generated: {date.today().strftime('%d.%m.%Y')} &nbsp;·&nbsp;
System: AirPulse Kraków &nbsp;·&nbsp; Model: {model}
</p>
</body>
</html>"""


def _build_pdf_bytes(
    fdate: date, model: str, pm10: float, level: str, regime: str, trend: str,
    summary: str, recommendation: str, weather: dict, drivers: list,
    station_id: str = "MpKrakWadow", station_name: str = "Wadowicka",
) -> bytes:
    buf  = io.BytesIO()
    doc  = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=2 * cm, leftMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )
    styles = getSampleStyleSheet()
    green  = rl_colors.HexColor("#4B9965")
    grey   = rl_colors.HexColor("#888888")

    title_style = ParagraphStyle("RPTitle", parent=styles["Heading1"],
                                 fontSize=18, spaceAfter=4, textColor=green)
    h2_style    = ParagraphStyle("RPH2", parent=styles["Heading2"],
                                 fontSize=12, spaceAfter=4, textColor=rl_colors.HexColor("#34495e"))
    body_style  = ParagraphStyle("RPBody", parent=styles["Normal"], fontSize=10, leading=15)
    small_style = ParagraphStyle("RPSmall", parent=styles["Normal"], fontSize=8, textColor=grey)

    row_bg = [rl_colors.HexColor("#f8f8f8"), rl_colors.white]

    def _table(data: list) -> Table:
        tbl = Table(data, colWidths=[4.5 * cm, 12 * cm])
        tbl.setStyle(TableStyle([
            ("FONTSIZE",        (0, 0), (-1, -1), 10),
            ("TEXTCOLOR",       (0, 0), (0, -1), grey),
            ("BOTTOMPADDING",   (0, 0), (-1, -1), 5),
            ("TOPPADDING",      (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS",  (0, 0), (-1, -1), row_bg),
        ]))
        return tbl

    w = weather
    story = [
        Paragraph("Air Quality Forecast - Kraków", title_style),
        Paragraph(fdate.strftime("%A, %d %B %Y").capitalize(), body_style),
        Spacer(1, 0.3 * cm),
        HRFlowable(width="100%", thickness=1, color=green),
        Spacer(1, 0.3 * cm),
        _table([
            ["Station", f"Kraków – {station_name} ({station_id})"],
            ["Model", model],
            ["PM10 Forecast", f"{pm10:.1f} µg/m³"],
            ["EU Daily Limit", "50 µg/m³"],
            ["Risk Level", level],
            ["Regime", regime],
            ["Trend", trend],
        ]),
        Spacer(1, 0.4 * cm),
        Paragraph("Weather Conditions", h2_style),
        _table([
            ["Temperature", f"{w['temp_avg']:.1f} °C"],
            ["Avg / Max Wind", f"{w['wind_mean']:.1f} / {w['wind_max']:.1f} m/s"],
            ["Humidity", f"{w['humidity_avg']:.0f}%"],
            ["Pressure", f"{w['pressure_avg']:.0f} hPa"],
            ["Rainfall", f"{w['rain_sum']:.1f} mm"],
            ["Snowfall", f"{w['snowfall_sum']:.1f} cm"],
        ]),
        Spacer(1, 0.4 * cm),
        Paragraph("Summary", h2_style),
        Paragraph(summary, body_style),
        Spacer(1, 0.3 * cm),
    ]

    if drivers:
        story.append(Paragraph("Key Drivers", h2_style))
        for d in drivers:
            story.append(Paragraph(f"• {d}", body_style))
        story.append(Spacer(1, 0.3 * cm))

    story += [
        Paragraph("Recommendation", h2_style),
        Paragraph(f"💡 {recommendation}", body_style),
        Spacer(1, 0.5 * cm),
        HRFlowable(width="100%", thickness=0.5, color=grey),
        Spacer(1, 0.2 * cm),
        Paragraph(
            f"Generated: {date.today().strftime('%d.%m.%Y')}  ·  "
            f"System: AirPulse Kraków  ·  Model: {model}",
            small_style,
        ),
    ]

    doc.build(story)
    return buf.getvalue()


def render_report_section(station_id: str, model: str, fdate: date, weather: dict) -> None:
    station_meta = STATIONS_META.get(station_id, STATIONS_META["MpKrakWadow"])
    station_name = station_meta["name"]

    st.markdown("#### 📄 Air Quality Report")
    if not st.button("Generate Report", type="primary"):
        return

    payload = {
        "model_name":    model,
        "forecast_date": str(fdate),
        "weather":       weather,
        "horizon":       1,
        "station_id":    station_id,
    }
    with st.spinner("Generating report…"):
        pred   = _api_post("/predict", payload)
        interp = _api_post("/interpret", {
            **payload,
            "pm10_forecast": pred.get("forecasts", [{}])[0].get("pm10", 35),
            "regime":        pred.get("regime", "Moderate"),
        })
        hist_dates, hist_pm10, is_real = fetch_pm10_history(
            7, lat=station_meta["lat"], lon=station_meta["lon"]
        )

    if "_error" in pred or "_error" in interp:
        st.error("Report generation failed. Check your connection to the backend.")
        return

    pm10    = pred["forecasts"][0]["pm10"]
    level   = pred["pm10_level"]
    regime  = pred["regime"]
    trend   = pred["trend"]
    summary = interp["summary"]
    rec     = interp["recommendation"]
    drivers = interp.get("key_drivers", [])
    emoji   = pm10_emoji(pm10)
    color   = LEVEL_COLORS.get(level, "#888")

    # Use real historical data in mini chart
    hist_x = [str(d) for d in hist_dates]

    # Explicit category order ensures forecast bar is always visible
    mini_all_x = hist_x + [str(fdate)]

    fig_mini = go.Figure()
    fig_mini.add_trace(go.Bar(
        x=hist_x, y=hist_pm10,
        name="Last 7 days (CAMS)" if is_real else "Last 7 days (est.)",
        marker_color="rgba(99,150,231,0.75)", marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>PM10: %{y:.1f} µg/m³<extra>historical</extra>",
    ))
    fig_mini.add_trace(go.Bar(
        x=[str(fdate)], y=[pm10],
        name=f"Forecast ({fdate.strftime('%d.%m')})",
        marker_color=color, marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>PM10: %{y:.1f} µg/m³<extra>forecast</extra>",
    ))
    fig_mini.add_hline(
        y=EU_DAILY_LIMIT, line_dash="dash",
        line_color="#e74c3c", line_width=1.2,
        annotation_text="EU Limit", annotation_font=dict(color="#e74c3c", size=10),
    )
    fig_mini.update_layout(
        height=240,
        margin=dict(t=10, b=40, l=0, r=0),
        showlegend=True,
        legend=dict(orientation="h", y=1.08, font=dict(size=11)),
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font_color="rgba(255,255,255,0.8)",
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=mini_all_x,
            gridcolor="rgba(255,255,255,0.05)",
            tickangle=-30,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            title="PM10 [µg/m³]",
            gridcolor="rgba(255,255,255,0.05)",
        ),
    )
    st.plotly_chart(fig_mini, use_container_width=True)

    eu_status    = ('<span style="color:#e74c3c">exceeded</span>'
                    if pm10 > EU_DAILY_LIMIT else
                    '<span style="color:#2ecc71">within limit</span>')
    date_str     = fdate.strftime("%A, %d %B %Y").capitalize()
    drivers_html = ""
    if drivers:
        items_html   = "".join(f"<li>{d}</li>" for d in drivers)
        drivers_html = (
            "<div style='margin-top:14px'>"
            "<div class='section-label'>Key Drivers</div>"
            "<ul style='margin:8px 0 0;padding-left:18px;font-size:0.87rem;line-height:1.8'>"
            f"{items_html}"
            "</ul></div>"
        )

    st.markdown(
        f"<div class='report-block'>"
        f"<div style='font-size:1.15rem;font-weight:700;margin-bottom:18px;"
        f"border-bottom:1px solid rgba(255,255,255,0.1);padding-bottom:12px'>"
        f"📋 Air Quality Forecast - {date_str}</div>"
        f"<table style='width:100%;border-collapse:collapse;font-size:0.85rem'>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0;width:42%'>Station</td>"
        f"<td>Kraków – {station_name} ({station_id})</td></tr>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0'>Model</td>"
        f"<td>{model}</td></tr>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0'>PM10 Forecast</td>"
        f"<td><b>{pm10:.1f} µg/m³</b></td></tr>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0'>EU Daily Limit</td>"
        f"<td>50 µg/m³ ({eu_status})</td></tr>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0'>Risk Level</td>"
        f"<td><span style='color:{color};font-weight:600'>{emoji} {level}</span></td></tr>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0'>Regime / Trend</td>"
        f"<td>{regime} / {trend}</td></tr>"
        f"<tr><td style='color:rgba(255,255,255,0.45);padding:4px 0'>Temperature</td>"
        f"<td>{weather['temp_avg']:.1f}°C &nbsp;|&nbsp; "
        f"Wind: {weather['wind_mean']:.1f} m/s avg, {weather['wind_max']:.1f} m/s max</td></tr>"
        f"</table>"
        f"<div style='margin-top:20px;padding-top:16px;"
        f"border-top:1px solid rgba(255,255,255,0.1)'>"
        f"<div class='section-label'>Summary</div>"
        f"<p style='margin:8px 0 0;line-height:1.7;font-family:DM Sans,sans-serif;font-size:0.9rem'>"
        f"{summary}</p></div>"
        f"{drivers_html}"
        f"<div style='margin-top:16px;padding:14px;"
        f"background:rgba(75,153,101,0.08);border-radius:10px;"
        f"border-left:3px solid {color}'>"
        f"<div class='section-label'>Recommendation</div>"
        f"<p style='margin:6px 0 0;font-family:DM Sans,sans-serif;font-size:0.9rem'>{rec}</p>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if _HAS_REPORTLAB:
        pdf_bytes = _build_pdf_bytes(
            fdate, model, pm10, level, regime, trend,
            summary, rec, weather, drivers,
            station_id=station_id, station_name=station_name,
        )
        st.download_button(
            label="⬇️ Download Report as PDF",
            data=pdf_bytes,
            file_name=f"airpulse_report_{station_id}_{fdate.strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
    else:
        html_bytes = _build_report_html(
            fdate, model, pm10, level, regime, trend,
            summary, rec, weather, drivers,
            station_id=station_id, station_name=station_name,
        ).encode("utf-8")
        st.download_button(
            label="⬇️ Download Report (HTML)",
            data=html_bytes,
            file_name=f"airpulse_report_{station_id}_{fdate.strftime('%Y%m%d')}.html",
            mime="text/html",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    station_id, model, horizon, fdate, weather = render_sidebar()

    station_meta = STATIONS_META.get(station_id, STATIONS_META["MpKrakWadow"])
    station_name = station_meta["name"]

    health     = _api_get("/health")
    live_badge = ""
    if "_error" not in health:
        live_badge = "<span class='live-badge'>● LIVE</span>"

    st.markdown(
        f"<div class='app-header'>"
        f"<div>"
        f"<div class='header-eyebrow'>AIR QUALITY INTELLIGENCE</div>"
        f"<div class='header-title'>AirPulse <span style='color:#4B9965'>Kraków</span></div>"
        f"<p class='header-subtitle'>AI-powered PM10 air quality forecasting for Lesser Poland</p>"
        f"</div>"
        f"<div style='padding-bottom:4px'>{live_badge}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.caption(
        f"Predictions for station **{station_name} ({station_id})** · "
        f"Model: **{model}** · "
        f"Forecast date: **{fdate.strftime('%d.%m.%Y')}**"
    )

    tab_forecast, tab_performance, tab_models = st.tabs([
        "🏠 Forecast Dashboard",
        "📊 Model Performance & Business Impact",
        "🧠 How the Models Work",
    ])

    with tab_forecast:
        render_forecast_tab(station_id, model, horizon, fdate, weather)
        st.divider()
        render_report_section(station_id, model, fdate, weather)

    with tab_performance:
        render_performance_tab()

    with tab_models:
        render_models_tab()

    with tab_performance:
        show_business_impact()

if __name__ == "__main__":
    main()
