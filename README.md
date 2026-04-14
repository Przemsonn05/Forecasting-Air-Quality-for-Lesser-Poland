# AirPulse Kraków - Smart Air Quality Forecasting System

**End-to-end machine learning system for daily PM10 concentration forecasting across seven monitoring stations in Kraków.**  
Covers data engineering, exploratory analysis, multi-model forecasting, SHAP explainability, a REST API, and an interactive web dashboard - all containerised with Docker.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Architecture](#2-project-architecture)
3. [Data Loading](#3-data-loading)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Exploratory Data Analysis](#5-exploratory-data-analysis)
6. [Feature Engineering](#6-feature-engineering)
7. [Modeling](#7-modeling)
8. [Evaluation](#8-evaluation)
9. [Advanced Analysis](#9-advanced-analysis)
10. [Streamlit Application](#10-streamlit-application)
11. [FastAPI Backend](#11-fastapi-backend)
12. [Docker Setup](#12-docker-setup)
13. [Business Insights](#13-business-insights)
14. [Conclusion](#14-conclusion)
15. [How to Run](#15-how-to-run)

---

## 🚀 Live Demo

| Platform | Link | Description |
| :--- | :--- | :--- |
| **Streamlit App** | [![Streamlit](https://img.shields.io/badge/Launch-App-FF4B4B)](https://airpulse-krakow-smart-air-quality-forecasting-system-hnsfun9ue.streamlit.app) | Interactive dashboard and models explanations |

---

## 1. Project Overview

Kraków ranks among the most polluted cities in Europe during winter months. PM10 - particulate matter with a diameter of 10 micrometres or less - poses serious health risks, particularly for people with respiratory and cardiovascular conditions. Daily 24-hour average concentrations frequently exceed the EU regulatory limit of **50 µg/m³**, triggering public-health advisories and transport restrictions.

This project builds a **production-ready forecasting system** that predicts next-day (and multi-day) PM10 concentrations across **all active Kraków GIOŚ monitoring stations**, with `MpKrakWadow` (Wadowicka) as the primary training target.

**System summary:**

| Layer | Technology |
|---|---|
| ML pipeline | Jupyter Notebook + modular `src/` package |
| Models | LightGBM, SARIMAX, ARIMA |
| Explainability | SHAP (SHapley Additive exPlanations) |
| REST API | FastAPI + Uvicorn |
| Dashboard | Streamlit ("AirPulse Kraków") |
| Containerisation | Docker + Docker Compose |

---

## 2. Project Architecture

```
.
├── notebooks/
│   └── air_quality_forecast.ipynb   # Main analysis and modelling notebook
├── src/                             # Modularised pipeline (extracted from notebook)
│   ├── data_loading.py              # PM10 Excel ingestion + Open-Meteo weather fetch
│   ├── data_preprocessing.py        # Gap imputation, weather merge
│   ├── feature_engineering.py       # Full feature pipeline (Box-Cox, lags, rolling, etc.)
│   ├── models.py                    # ARIMA, SARIMAX, LightGBM training
│   ├── evaluation.py                # Metrics, plots, exceedance analysis
│   ├── eda.py                       # EDA plots
│   ├── config.py                    # Training constants (stations, dates, hyperparams)
│   └── utils.py                     # Logger, Box-Cox inverse, plot helpers
├── backend/
│   ├── api.py                       # FastAPI app (predict / explain / interpret / metrics / stations)
│   ├── schemas.py                   # Pydantic v2 request/response models
│   └── services/
│       ├── model_service.py         # Model loading, prediction, regime classification
│       ├── explainability_service.py# SHAP computation
│       └── interpretability_service.py # Rule-based NLG forecast explanations
├── frontend/
│   └── app.py                       # Streamlit dashboard ("AirPulse Kraków")
├── config/
│   └── config.py                    # Shared runtime config (API host, stations, thresholds, colours)
├── scripts/
│   └── prepare_api_artifacts.py     # One-off: train models and serialise → models/
├── models/                          # Serialised artefacts (*.pkl, *.joblib)
├── data/                            # Raw yearly PM10 Excel files (2019–2024)
├── images/                          # All generated plots
├── docker-compose.yml
├── requirements.txt
└── main.py                          # CLI entry-point for the full pipeline
```

### Component interaction

```
┌───────────────────────┐    HTTP     ┌───────────────────────┐
│   Streamlit Frontend  │ ──────────► │   FastAPI Backend     │
│   (port 8501)         │ ◄────────── │   (port 8000)         │
└───────────────────────┘             └──────────┬────────────┘
                                                 │ loads at startup
                                      ┌──────────▼────────────┐
                                      │  models/*.pkl/.joblib │
                                      │  (LightGBM, ARIMA,    │
                                      │   SARIMAX, KMeans,    │
                                      │   scaler, λ_bc,       │
                                      │   recent_history)     │
                                      └───────────────────────┘
```

The **Streamlit frontend** collects a monitoring station selection, model choice, and forecast date from the user, forwards them to the **FastAPI backend**, and renders the returned forecasts, confidence intervals, SHAP contributions, and AI-generated narrative. All trained models are serialised as `.pkl` / `.joblib` artefacts and loaded into memory once at API startup. The `src/` modules contain the same logic as the notebook and are used by both `scripts/prepare_api_artifacts.py` and the backend services.

---

## 3. Data Loading

### Sources

| Source | Format | Coverage |
|---|---|---|
| GIOŚ (Polish Chief Inspectorate for Environmental Protection) | Excel (`.xlsx`) | 2019–2024, daily 24-hour PM10 averages |
| Open-Meteo Archive API | JSON via HTTP | Daily weather for Kraków (2019–2024) |

### PM10 data

The GIOŚ export format places the header row (`Kod stacji`) at a variable row position depending on the year. The loader scans each file dynamically to find this row before re-reading with the correct `header` argument:

```python
load_pm10_raw(data_dir, years=range(2019, 2025))
```

Station codes are **detected automatically** from the Excel files at runtime - the system reads only the first 5 rows of each workbook (fast), then scans for column names starting with `MpKrak*`. This means newly added GIOŚ stations are picked up without code changes.

All seven currently known Kraków monitoring stations:

| Code | Location |
|---|---|
| `MpKrakWadow` | Wadowicka (primary training target) |
| `MpKrakAlKras` | Al. Krasińskiego |
| `MpKrakBujaka` | Bujaka |
| `MpKrakBulwar` | Bulwarowa |
| `MpKrakOsPias` | Os. Piastów |
| `MpKrakSwoszo` | Swoszowice |
| `MpKrakZloRog` | Złoty Róg |

Decimal separators in the GIOŚ export use commas (Polish locale); these are normalised to dots during parsing. A strict daily `DatetimeIndex` is enforced via `asfreq('D')`, inserting `NaN` for any missing dates.

### Weather data

Daily meteorological variables are fetched from the **Open-Meteo** archive endpoint for the city-centre coordinates (shared across all stations):

- Temperature (avg, min, max)
- Precipitation (rain sum, snowfall sum)
- Wind (mean speed, max gust, dominant direction)
- Relative humidity (avg)
- Surface pressure (avg)

Dominant wind direction is encoded as **sin/cos cyclical features** immediately after download to preserve the circular topology (i.e., NW and N are close; N and S are far).

---

## 4. Data Preprocessing

### Steps and rationale

| Step | What it does | Why it's needed |
|---|---|---|
| Header detection | Locates `Kod stacji` row dynamically | GIOŚ format varies by year |
| Datetime conversion | `pd.to_datetime` with `errors='coerce'` | Non-date rows (metadata) become `NaT` and are dropped |
| `asfreq('D')` | Enforces strict daily frequency | Gaps would silently misalign lag features |
| Short-gap interpolation | Time-based linear interpolation for gaps ≤ 3 days | Sensor outages of 1–3 days are common and interpolation is reliable |
| Long-gap flagging | Binary `{station}_long_gap` column for gaps > 3 days | Avoids artificially constructing data where readings are genuinely absent |
| Boundary fill | `ffill()` / `bfill()` | Closes isolated NaNs at the start or end of the series after interpolation |
| Weather merge | Left join PM10 DataFrame on date index | Attaches meteorological context to each observation |

Short gaps are interpolated on a **time axis** (not row index) to respect the irregular distribution of readings. Long gaps are preserved as `NaN` with a companion indicator feature so the model can learn that the context is unreliable.

---

## 5. Exploratory Data Analysis

### Time series overview

![Time series](images/eda_time_series.png)

The raw PM10 series for `MpKrakWadow` (2019–2024) reveals a strong annual seasonality with pronounced **winter spikes** driven by residential coal and biomass combustion. Clean summer periods with concentrations below 20 µg/m³ contrast sharply with heating-season episodes exceeding 200 µg/m³. The COVID-19 lockdowns in early 2020 produced a brief but notable reduction in baseline levels, visible as a slightly cleaner January–March window compared to surrounding years. The 50 µg/m³ EU daily limit is overlaid as a reference line, making it immediately apparent how frequently the threshold is crossed during the heating season. Year-over-year differences are clearly visible: 2022 stands out as a relatively clean winter, while 2023–2024 saw a modest uptick that may reflect changing weather patterns or partial reversal of emission reductions. The series exhibits marked heteroscedasticity — residual volatility is substantially higher in cold months — which directly motivates the Box-Cox variance-stabilising transformation applied later in the pipeline.

---

### Monthly distribution

![Monthly boxplots](images/eda_monthly_boxplots.png)

Boxplots by calendar month confirm that median PM10 is approximately **3–5× higher in winter (December–February) than in summer (June–August)**. The interquartile range also widens considerably in winter, reflecting greater day-to-day variability driven by weather conditions such as wind speed, temperature inversions, and precipitation. The EU daily limit of 50 µg/m³ is routinely breached from October through March, with December and January showing the highest median values and the widest spread. Summer months (June–August) display a narrow, low distribution with medians below 20 µg/m³ and few outliers, confirming that forecasting difficulty is concentrated in the heating season. Spring and autumn (March–April, September–October) act as transitional periods, with moderate interquartile ranges that reflect the uncertain onset and retreat of residential heating. The extreme outliers visible in January and December correspond to severe smog episodes where PM10 exceeds 150–200 µg/m³ — events that pose acute health hazards and typically trigger public transport-free days and outdoor-activity advisories in Kraków.

---

### Year × Month heatmap

![Heatmap month year](images/eda_heatmap_month_year.png)

The heatmap of mean monthly PM10 by year reveals that **2020 and 2022 had notably cleaner winters**, while **2023 and early 2024 saw elevated concentrations**. Year-over-year variation is substantial, reflecting both meteorological differences and gradual policy changes (e.g., the Małopolska anti-smog resolution restricting solid-fuel heating). The colour scale transitions from deep blue (clean) to dark red (heavily polluted), making seasonal and inter-annual patterns immediately legible without requiring numerical labels. A consistent band of elevated values runs across the winter columns (November–February) for every year, confirming the dominance of the heating season as the primary pollution driver. The 2020 anomaly — a relatively cooler colour in January–February — aligns with a milder-than-average meteorological winter combined with early pandemic movement restrictions. Summer months form a uniformly cool horizontal band at the bottom of each column, reinforcing the finding that solar radiation, longer days, and higher wind speeds create effective natural pollutant dispersion. The heatmap serves as a compact diagnostic tool for tracking the long-term impact of anti-smog regulation: a gradual cooling of winter colours after 2022 is tentatively visible, suggesting the ordinance is having a measurable effect on baseline PM10 levels.

---

### STL decomposition

![STL decomposition](images/eda_stl_decomposition_analysis.png)

STL (Seasonal-Trend decomposition using LOESS) separates the signal into trend, seasonal, and residual components, providing a cleaner view of each structural element than classical additive decomposition. The seasonal component confirms a dominant annual cycle with a peak-to-trough amplitude of roughly 40 µg/m³, consistent with the shift from heating to non-heating periods. The trend component reveals a subtle downward drift after 2022, tentatively linked to the progressive enforcement of the Małopolska anti-smog ordinance banning high-emission solid-fuel boilers. The residuals exhibit clear heteroscedasticity — variance is noticeably higher in winter — which directly motivates the **Box-Cox transformation** applied in the feature engineering pipeline to stabilise variance before modelling. Unlike classical decompositions, STL is robust to outliers because LOESS fitting uses locally weighted regression, preventing extreme smog-episode days from distorting the estimated seasonal shape. The LOESS bandwidth parameters were tuned to capture the annual cycle without over-smoothing multi-week heating episodes. Understanding the relative magnitude of trend, seasonal, and residual components guides model design: the strong seasonal component is addressed via calendar and lag features, while the residuals justify the inclusion of daily weather covariates to explain unexplained day-to-day variation.

---

## 6. Feature Engineering

The full feature pipeline is implemented in `src/feature_engineering.py` and runs in a fixed order to prevent data leakage.

### Feature groups

**Calendar features**
- `month`, `year`, `season`, `is_weekend`, `is_holiday` (Polish public holidays)
- Cyclical encoding: `month_sin/cos`, `doy_sin/cos`, `dow_sin/cos` - removes artificial discontinuities at year/week boundaries

**Box-Cox transformation**
- Lambda (λ = −0.243) is estimated exclusively on training data, then applied to the full series
- Stabilises variance and normalises the heavy right tail of PM10, which improves both tree-based and statistical model performance

**Lag features** (computed on the Box-Cox-transformed target)
- `lag_1d`, `lag_2d`, `lag_7d`, `lag_14d`
- Yesterday's PM10 is the single strongest predictor; weekly lags capture the seasonal autocorrelation structure

**Rolling statistics** (computed on raw PM10 with `shift(1)` to prevent leakage)
- `rolling_mean_{3,7,14,30}d`, `rolling_std_{7,14}d`
- `rolling_diff_7d` (7-day minus 14-day mean): captures whether pollution is accelerating or easing

**Weather-derived features**

| Feature | Description |
|---|---|
| `is_frost` | Temperature ≤ 0 °C - proxy for increased heating demand |
| `is_calm_wind` | Wind mean ≤ 2 m/s - weak dispersion of pollutants |
| `wind_inverse` | 1 / (wind_max + 0.1) - non-linear dispersion proxy |
| `heating_degree_days` | max(0, 15 − temp_avg) - physical heating demand |
| `hdd_7d` | 7-day rolling HDD sum - accumulated thermal demand |
| `rain_yesterday`, `rain_3d_sum`, `rain_7d_sum` | Washout effects of recent precipitation |
| `dry_spell_days` | Days without rain in last 14 - particle accumulation |
| `inversion_proxy` | frost × calm × low temperature amplitude - detects inversions |

**Multi-station spatial features**
- Per-station `lag_1d` for all auxiliary stations (Swoszowice, Bujaka, Bulwarowa, and others detected in data)
- `aux_mean_lag1`, `aux_max_lag1`, `aux_spread_lag1`
- Inter-station Pearson correlations exceed 0.90; spatial aggregates provide a compact regional signal

**Interaction terms**
- `is_frost_calm`: frost × calm wind - double stagnation, highest smog risk
- `is_heating_season_calm`: heating season × calm - sustained elevated risk
- `hdd_calm`: heating demand × no wind - physically motivated
- `cold_dry_calm`: below-zero × no rain × calm - conditions for severe episodes

### SHAP analysis

![SHAP summary](images/lgbm_shap_summary.png)

SHAP values from the LightGBM model confirm that **lag features and rolling means dominate predictions**, followed by heating-season indicators and inversion proxies. Weather interaction terms (e.g., `hdd_calm`, `inversion_proxy`) consistently rank in the top 15 features, validating the domain-driven feature design. The beeswarm plot renders each training observation as a dot coloured by the feature's raw value (red = high, blue = low), making the direction and magnitude of each feature's effect immediately legible without requiring separate partial dependence plots. For `lag_1d`, high values (red dots) shift the SHAP contribution strongly positive — confirming strong positive autocorrelation — while low values pull forecasts below the baseline, consistent with the physical persistence of PM10 in still air. Rolling mean features (`rolling_mean_7d`, `rolling_mean_14d`) show a similar directional pattern, indicating that sustained elevated pollution over the prior weeks is a reliable forward signal for continued high concentrations. Calendar features such as `heating_season` and `month_sin/cos` appear lower in the ranking but provide stable, unconditional signal that anchors predictions when recent lag data alone is ambiguous (e.g., at the start of the season). The spread of SHAP values for `inversion_proxy` is narrower but consistently positive for high feature values, capturing the specific meteorological conditions — frost, calm wind, low temperature amplitude — that trap combustion emissions close to the surface.

---

## 7. Modeling

Three complementary modelling approaches are used to cover different aspects of the forecasting problem:

### LightGBM (primary model)

A gradient-boosted decision tree model operating on the full 68-feature engineered feature set.

- **Why:** Handles non-linear interactions, missing values, and heteroscedastic targets natively; fastest to train; supports SHAP
- **Training:** Early stopping on a chronological 15% holdout of training data (`LGBM_ES_FRACTION = 0.15`); 3 000 estimators max, learning rate 0.02
- **Target:** Box-Cox-transformed PM10 (λ = −0.243); predictions are back-transformed at evaluation time
- **Split:** Strict time-series split - no shuffling; train ≤ 2022-12-31, val = 2023

### SARIMAX (statistical baseline with weather)

Seasonal ARIMA with exogenous regressors, order selected via `pmdarima.auto_arima` with weekly seasonality (`m=7`).

- **Why:** Interpretable; captures linear AR/MA dynamics explicitly; exogenous weather variables (`temp_avg`, `wind_max`, `is_heating_season`, `inversion_proxy`, `hdd_calm`, `rain_3d_sum`, `is_calm_wind`) are included as regressors
- **Training:** Walk-forward validation with full refit every 7 steps; exogenous features are standardised on the training set only
- **Confidence intervals:** Available from the SARIMAX state-space covariance

### ARIMA (pure time-series baseline)

A non-seasonal ARIMA fitted on the Box-Cox series, order selected by ADF stationarity test + `auto_arima`.

- **Why:** Serves as a reference point for how much is gained by adding weather covariates and richer features
- **Walk-forward:** Same refit schedule as SARIMAX, with 90% prediction intervals

### Naïve persistence baseline

`PM10(t+1) = PM10(t)` - predicts tomorrow equals today. All models are benchmarked against this baseline; meaningful improvement over persistence is the minimum bar for a useful forecast.

---

## 8. Evaluation

All metrics are computed on **back-transformed µg/m³ values** to be directly interpretable.

### Regression metrics

| Metric | Formula | Purpose |
|---|---|---|
| **R²** | 1 − SS_res / SS_tot | Overall variance explained; 1 is perfect |
| **MAE** | mean(\|y − ŷ\|) | Average absolute error in µg/m³; robust to outliers |
| **RMSE** | √mean((y − ŷ)²) | Penalises large errors more heavily; sensitive to smog peaks |
| **SMAPE** | Symmetric MAPE variant | Bounded, handles low-concentration days more fairly |

### Validation set results (2023)

| Model | MAE (µg/m³) | RMSE (µg/m³) | MAPE (%) | R² |
|---|---|---|---|---|
| **LightGBM** | **4.21** | **6.09** | **20.32** | **0.73** |
| SARIMAX | 6.05 | 9.08 | 28.61 | 0.39 |
| ARIMA | 6.24 | 9.39 | 30.85 | 0.35 |
| Prophet |	6.90 |	9.68 |	36.17% |	0.31 |

LightGBM achieves the lowest MAE and RMSE on the validation set, outperforming SARIMAX, ARIMA, and Prophet models. It provides the best metrics and an interpretable tool for forecasting PM10 levels, which are usually hard to predict, especially in heavily polluted, smog-prone Kraków.

### Exceedance classification metrics

Since **health impact depends on whether the 50 µg/m³ EU limit is breached**, a binary classification view is computed alongside regression metrics:

| Metric | Purpose |
|---|---|
| **Precision** | Of all predicted exceedances, how many were real? |
| **Recall** | Of all real exceedances, how many were caught? |
| **F1** | Harmonic mean - balances false alarms and missed events |

For public-health use cases, **recall is prioritised**: missing a real smog day is more costly than a false alarm.

### Model comparison

![Model metrics comparison](images/model_comparison_neutral.png)

The model comparison bar chart presents the four core regression metrics — MAE, RMSE, SMAPE, and R² — for all evaluated models on the 2023 validation set, displayed side by side for direct comparison. LightGBM achieves the best result across every metric (MAE 4.21 µg/m³, RMSE 6.09 µg/m³, R² 0.73), outperforming the naïve persistence baseline by approximately 35% on MAE. SARIMAX, which incorporates exogenous weather variables, scores significantly better than the pure ARIMA model, demonstrating the value of meteorological covariates even within a linear statistical framework. The ARIMA baseline — trained on the Box-Cox series without external regressors — still outperforms naïve persistence on RMSE, confirming that even simple autocorrelation structure carries meaningful predictive information. Colour coding distinguishes each model across all metric panels, making it straightforward to assess relative strengths and weaknesses at a glance. The chart provides a regression-level sanity check: all three trained models surpass the persistence baseline, confirming that the engineered features and modelling choices add genuine forecasting value rather than merely fitting historical noise.

---

![Forecast comparison](images/model_comparison_forecast.png)

The forecast comparison chart overlays actual PM10 observations (black line) with predictions from all three models across the 2023 validation period, enabling a visual inspection of tracking quality that summary statistics alone cannot convey. LightGBM follows the daily fluctuations most closely, particularly during moderate pollution episodes in autumn and spring, while maintaining competitive accuracy during clean summer periods. SARIMAX shows broader residuals on extreme winter peaks but generally captures the seasonally elevated base level with reasonable fidelity, and its confidence intervals provide useful uncertainty bounds for operational use. ARIMA tends to underestimate sharp pollution spikes — especially multi-day smog episodes — because it lacks the non-linear interactions between weather variables and emissions that LightGBM is able to exploit through its gradient-boosted tree structure. The 50 µg/m³ EU threshold is shown as a horizontal reference line, making it straightforward to assess visually which models correctly anticipate limit exceedances and which miss them. Time-series overlay complements the numerical metrics table: while R² and MAE summarise average performance, this chart exposes systematic biases such as lag in peak detection or persistent underestimation of smog tails that aggregate statistics can obscure.

---

## 9. Advanced Analysis

### Stratified analysis

![Stratified metrics](images/stratified_metrics.png)

Metrics are broken down by **pollution regime** (Clean / Moderate / Polluted) and by **season** to reveal where each model struggles and where its assumptions hold well. Models perform well in moderate conditions but consistently underestimate the highest peaks, a bias typical of mean-regression learners when the target distribution is heavily right-tailed. LightGBM shows the best recall during Polluted episodes, benefiting from interaction features such as `hdd_calm` and `inversion_proxy` that encode the meteorological conditions driving severe smog events. SARIMAX is more reliable in clean-air periods, where the linear AR/MA structure is sufficient to capture the low-volatility summer regime. The Clean regime (PM10 < 25 µg/m³) yields the lowest absolute errors for all models, because summer concentrations cluster tightly around 15 µg/m³ with minimal day-to-day variance. In the Polluted regime (PM10 > 50 µg/m³), the gap between LightGBM and the statistical models widens substantially, confirming that non-linear feature interactions are essential for capturing extreme episodes. Seasonal stratification further reveals that winter performance is approximately 40% weaker than summer performance across all models, motivating the future work direction of training a separate, winter-focused model with additional extreme-event features.

### Validation vs. test performance

The validation period (2023) and the held-out test period (2024) show a **notable gap in performance**, with test metrics being weaker. Likely causes:

- **Distribution shift:** The 2024 season had an unusual weather pattern not well-represented in training data
- **Heating policy changes:** The Małopolska anti-smog regulation tightened in 2023–2024, shifting the base pollution level downward relative to prior years
- **Overfitting to validation period:** Hyperparameter tuning was informed by validation-set performance

### Exceedance classification

![Exceedance classification](images/exceedance_classification.png)

The confusion matrices compare how well each model identifies days that breach the 50 µg/m³ EU threshold versus days that remain below it, framing the forecasting problem as a binary early-warning task. The goal is high recall — catching real exceedance events — even at the cost of some false positives, because a missed smog day has greater public-health consequences than an unnecessary advisory. LightGBM achieves the highest recall among all models, correctly flagging the majority of dangerous days and leaving the fewest exceedances undetected. SARIMAX's recall is competitive but comes with a slightly higher false-positive rate, meaning it more frequently issues warnings on days that ultimately stay below the limit. ARIMA struggles most with exceedance detection, consistent with its weaker overall performance on high-pollution days, and produces the highest false-negative rate across the validation set. The decision threshold for all models is tuned below the default 0.5 to shift the trade-off towards sensitivity, reflecting the asymmetric cost structure of the application. Together, the four confusion matrices provide a clear view of each model's alert reliability and support an informed choice of model for operational deployment.

### Smog episode detection

Beyond single-day exceedances, **consecutive smog days** are grouped into episodes. The episode-level analysis provides:

- Number and duration of multi-day smog events per year
- Spatial coherence - whether multiple stations are simultaneously elevated
- A timeline of historical episodes (see `images/exceedance_timeline.png`)

This forms the basis for a proactive alert system.

---

## 10. Streamlit Application

The **AirPulse Kraków** dashboard (`frontend/app.py`) is a dark-themed interactive web application built with Streamlit. It connects to the FastAPI backend and requires no manual weather input - live conditions are fetched automatically.

### Sidebar controls

- **Monitoring Station** - dropdown listing all GIOŚ stations detected in the data (up to 7 Kraków stations); the map and forecast update accordingly
- **Forecasting Model** - `LightGBM`, `SARIMAX`, or `ARIMA` with a brief description of each
- **Forecast Horizon** - 1 to 3 days ahead (slider)
- **Forecast Date** - today + 1 day by default; up to today + 3 days
- **Live Weather** - current conditions fetched automatically from Open-Meteo (temperature, humidity, wind, pressure, rain); a "Refresh Weather" button clears the cache
- **API Status** - live indicator showing which models are loaded in the backend

### Forecast tab

- **Interactive map** (Plotly Scattermapbox, dark tile layer) showing all monitoring stations; the selected station is highlighted with a star marker and PM10 forecast label, remaining stations show colour-coded estimated PM10 values
- **Forecast gauge** - animated Plotly Indicator gauge with colour zones (green / amber / red) and a delta against a 3-day rolling estimate
- **Multi-day bar chart** - point forecast ± confidence bounds per day, with the EU 50 µg/m³ limit shown as a reference line
- **Historical PM10 chart** - last 7 days of Open-Meteo CAMS air-quality model estimates for the selected station location
- **SHAP waterfall chart** (LightGBM only) - top feature contributions to the current forecast with signed attribution values
- **AI narrative** - rule-based natural-language summary of dominant weather drivers, health implications, and recommended actions scaled to the forecasted severity level

![app1](images/st_app1.png)

---

![app2](images/st_app2.png)

### Report export

One-click download of the current forecast as a structured PDF generated with `reportlab`:
- Station name, forecast date, model used
- PM10 point forecast and confidence interval
- Key weather drivers
- Colour-coded pollution level and health recommendation
- File named `airpulse_report_{station_id}_{date}.pdf`

### Model Performance tab

- Metrics table (MAE, RMSE, SMAPE, R²) fetched live from `/metrics`; best model highlighted in green
- Bar chart comparing all models across all metrics
- Descriptive analysis of LightGBM's advantage over statistical baselines

![app4](images/st_app4.png)

### How the Models Work tab

Plain-language explanations of LightGBM, ARIMA / SARIMAX, and their trade-offs - aimed at non-technical users.

![app5](images/st_app5.png)

### Business Impact section

- Key headline metrics (6 years of data, 4 models compared, 50 µg/m³ EU limit, 3-day forecast horizon)
- Impact cards: public health, environmental policy, operational use, cost of missed alarms

![app7](images/st_app7.png)

---

## 11. FastAPI Backend

The API (`backend/api.py`) is the central serving layer for all model inference.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/stations` | List of all available monitoring stations with coordinates |
| `POST` | `/predict` | Multi-day PM10 forecast for a selected model and station |
| `GET` | `/metrics` | Pre-computed validation metrics for all models |
| `GET` | `/validation` | Full 2023 validation-set predictions (actual vs. forecast) |
| `POST` | `/explain` | SHAP feature attributions (LightGBM only) |
| `POST` | `/interpret` | Rule-based NLG narrative for a forecast |
| `GET` | `/health` | Liveness check - reports loaded models, λ_bc, and history size |

### Design notes

- **Lifespan startup:** Models are loaded once into a `ModelService` singleton at startup; no cold-load latency on subsequent requests
- **Station-aware predictions:** LightGBM uses the last 60 days of per-station lag history from `recent_history.pkl`; ARIMA and SARIMAX are trained on the primary station but serve all station requests
- **Box-Cox back-transform:** All predictions are returned in original µg/m³ units regardless of the internal transform
- **Regime classification:** A `KMeans(3)` model clusters current weather conditions into Clean / Moderate / Polluted regimes for contextual display
- **CORS:** Open (`allow_origins=["*"]`) for local development; restrict in production
- **Confidence intervals:** Available for ARIMA and SARIMAX (state-space covariance); approximated for LightGBM

**Run locally:**
```bash
uvicorn backend.api:app --reload --port 8000
```

**Interactive docs:** `http://localhost:8000/docs`

---

## 12. Docker Setup

The system is fully containerised with **two services** sharing a private bridge network.

```yaml
# docker-compose.yml (simplified)
services:
  backend:
    build: { dockerfile: backend/Dockerfile }
    ports: ["8000:8000"]
    volumes: ["./models:/app/models:ro"]
    healthcheck: { test: ["CMD", "python", "/app/backend/healthcheck.py"] }

  frontend:
    build: { dockerfile: frontend/Dockerfile }
    ports: ["8501:8501"]
    environment: { API_HOST: "http://backend:8000" }
    depends_on: { backend: { condition: service_healthy } }
```

Key design decisions:

- The `models/` directory is mounted **read-only** into the backend container; no model files are baked into the image
- The frontend **waits for the backend health check to pass** before starting, preventing connection errors at launch
- `restart: unless-stopped` ensures automatic recovery from transient failures
- Environment variables (`BACKEND_PORT`, `FRONTEND_PORT`, `API_HOST`) can be overridden via `.env`

**Pre-requisite:** Generate model artefacts on the host before the first Docker run:
```bash
python scripts/prepare_api_artifacts.py
```

---

## 13. Business Insights

### A. Technical insights

- **Lag features dominate:** `lag_1d` is consistently the most important feature, confirming strong autocorrelation in daily PM10
- **Weather interactions matter:** `hdd_calm` (heating degree days × calm wind) and `inversion_proxy` significantly improve exceedance recall - raw weather variables alone are insufficient
- **LightGBM outperforms classical models** on all regression metrics with ~35% lower MAE than the naïve baseline
- **Box-Cox transformation is essential:** Without it, all models produce larger errors on high-concentration days due to heavy-tailed residuals
- **Multi-station spatial features add value:** Auxiliary station lag-1 readings improve both validation MAE and exceedance recall, confirming regional pollution coherence

### B. Non-technical insights

- **Winter is the critical period:** The vast majority of EU limit exceedances occur between October and March; summer forecasts are straightforward
- **Calm, cold, dry nights are the danger signal:** The combination of below-freezing temperatures, no wind, and no recent rain creates stagnant air conditions that trap coal-combustion emissions
- **A few days of lag drive the forecast:** If PM10 was high yesterday, it is almost certain to be elevated today - real-time sensor data is the most valuable input
- **Residents can act on 1-day forecasts:** Kraków operates a public alert system; a reliable 24-hour forecast allows vulnerable residents, schools, and cyclists to plan accordingly
- **Policy impact is measurable:** The Małopolska anti-smog regulation appears to have lowered baseline winter concentrations after 2022, visible as a downward trend in the year × month heatmap

---

## 14. Conclusion

### Summary

This project delivers a complete, production-oriented PM10 forecasting pipeline - from raw GIOŚ Excel files to a live REST API and interactive multi-station dashboard. LightGBM achieves the best overall performance (MAE 4.21 µg/m³, R² 0.73 on the 2023 validation set), powered by a rich set of lag, rolling, weather, and interaction features engineered from domain knowledge. SARIMAX and ARIMA add interpretability and serve as useful comparison points.

### Strengths

- Strict temporal data splits at every stage - no leakage
- Domain-aware feature engineering guided by atmospheric physics
- SHAP-based interpretability for every LightGBM prediction
- Dynamic multi-station support: new GIOŚ stations are detected automatically without code changes
- Full observability: structured logging throughout the `src/` pipeline
- Containerised, reproducible deployment

### Limitations

- **Single-target training:** The LightGBM model is trained on `MpKrakWadow`; non-target stations reuse the same model weights with station-specific lag history as a reasonable approximation
- **No real-time sensor feed:** Weather inputs come from the Open-Meteo API; PM10 history uses the CAMS atmospheric model rather than live sensor readings
- **Distribution shift in 2024:** The model was validated on 2023 data; test performance on 2024 indicates the need for periodic retraining as emission patterns evolve

### Future improvements

- **Automated retraining pipeline:** Scheduled weekly refit using Cron or Airflow on the most recent rolling window, preventing model drift
- **Hybrid ensemble:** Weighted combination of LightGBM and SARIMAX to leverage complementary strengths (non-linear vs. linear seasonal patterns)
- **Robust testing & CI/CD:** Pytest suite covering unit tests for preprocessing and integration tests for all API endpoints

---

## 15. How to Run

### Option A - Docker (recommended)

**Requirements:** Docker Desktop, model artefacts in `models/`

```bash
# 1. Generate model artefacts (run once, on host)
python scripts/prepare_api_artifacts.py

# 2. Start both services
docker-compose up --build

# 3. Open the dashboard
#    http://localhost:8501

# 4. Explore the API
#    http://localhost:8000/docs
```

To stop:
```bash
docker-compose down
```

---

### Option B - Local (development)

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Generate model artefacts**

```bash
python scripts/prepare_api_artifacts.py
```

**3. Start the backend**

```bash
uvicorn backend.api:app --reload --port 8000
```

**4. Start the frontend** (in a separate terminal)

```bash
streamlit run frontend/app.py
```

**5. (Optional) Run the full pipeline notebook**

```bash
jupyter lab notebooks/air_quality_forecast.ipynb
```

---

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `http://localhost:8000` | Backend URL seen by the frontend |
| `BACKEND_PORT` | `8000` | Backend listen port |
| `FRONTEND_PORT` | `8501` | Frontend listen port |

Copy `.env` and adjust as needed before running.

---

## Tech Stack

`Python 3.11+` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` · `statsmodels` · `pmdarima` · `SHAP` · `FastAPI` · `Pydantic v2` · `Streamlit` · `Plotly` · `Docker` · `scipy` · `holidays` · `reportlab` · `requests`

---

<div align="center">

**⭐ If you found this project helpful, please star the repository!**

![GitHub stars](https://img.shields.io/github/stars/Przemsonn05/AirPulse-Krakow-Smart-Air-Quality-Forecasting-System)
