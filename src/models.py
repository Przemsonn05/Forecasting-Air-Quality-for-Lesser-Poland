"""
Model training: ARIMA, SARIMAX, Prophet, and LightGBM.

Each function returns predictions aligned with the validation / test set passed
in.  All models are trained on Box-Cox-transformed PM10; back-transformation is
handled in the evaluation stage.

Key design choices (mirroring notebook cells 91–128):
- ARIMA / SARIMAX: ADF stationarity test → auto_arima order selection
  → walk-forward refitting every REFIT_EVERY steps.
- SARIMAX: exogenous features are standardised (StandardScaler fitted on train
  only) before being passed to the model.
- Prophet: a 7-day lagged rolling mean of the Box-Cox target is appended as an
  autoregressive regressor in addition to the meteorological regressors.
- LightGBM: an Optuna hyperparameter search is offered as an optional path;
  the default uses well-tuned fixed parameters from config.py.
"""

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import pmdarima as pm
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from src.utils import get_logger

logger = get_logger(__name__)
warnings.filterwarnings("ignore")

def train_predict_arima(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "PM10_transformed",
    refit_every: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk-forward ARIMA forecast on the validation set.

    Order selection:
    1. ADF test on training data → set integration order ``d``.
    2. ``pmdarima.auto_arima`` searches (p, d, q) with ``d`` fixed.

    At each step the last fitted model issues a one-step-ahead forecast, then
    the model is extended with the true observation.  Every ``refit_every``
    steps the model is fully re-fitted to incorporate structural changes.

    Parameters
    ----------
    train:
        Training DataFrame containing ``target_col``.
    val:
        Validation DataFrame containing ``target_col``.
    target_col:
        Box-Cox-transformed target column.
    refit_every:
        Number of steps between full ARIMA refits.

    Returns
    -------
    predictions : np.ndarray  — point forecasts (length = len(val))
    lower       : np.ndarray  — lower bound of 90 % prediction interval
    upper       : np.ndarray  — upper bound of 90 % prediction interval
    """
    series = train[target_col].dropna()

    adf_stat, adf_p, *_ = adfuller(series)
    d = 0 if adf_p < 0.05 else 1
    logger.info("ADF p=%.4f → d=%d", adf_p, d)

    auto_mdl = pm.auto_arima(
        series,
        d=d, max_p=4, max_q=4,
        seasonal=False, stepwise=True,
        suppress_warnings=True, error_action="ignore",
    )
    order = auto_mdl.order
    logger.info("ARIMA auto-selected order: %s", order)

    history = series.tolist()
    fitted = ARIMA(history, order=order).fit()

    predictions, lowers, uppers = [], [], []

    for t in range(len(val)):
        fc = fitted.get_forecast(steps=1)
        predictions.append(fc.predicted_mean.iloc[0])
        ci = fc.conf_int(alpha=0.10)
        lowers.append(ci.iloc[0, 0])
        uppers.append(ci.iloc[0, 1])

        history.append(val[target_col].iloc[t])

        if (t + 1) % refit_every == 0:
            fitted = ARIMA(history, order=order).fit()

    logger.info("ARIMA %s walk-forward complete (%d steps)", order, len(val))
    return np.array(predictions), np.array(lowers), np.array(uppers)

def train_predict_sarimax(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "PM10_transformed",
    exog_cols: list[str] = None,
    refit_every: int = 7,
) -> np.ndarray:
    """Walk-forward SARIMAX forecast with auto-selected order.

    Exogenous features are standardised using a ``StandardScaler`` fitted
    exclusively on the training set (no leakage).  Order is selected via
    ``auto_arima`` with weekly seasonality (m=7).

    Parameters
    ----------
    train:
        Training DataFrame.
    val:
        Validation DataFrame.
    target_col:
        Box-Cox-transformed target column.
    exog_cols:
        Exogenous regressor column names.  Must be present in both splits.
    refit_every:
        Number of steps between full SARIMAX refits.

    Returns
    -------
    np.ndarray
        One-step-ahead predictions, length == ``len(val)``.
    """
    if exog_cols is None:
        exog_cols = []

    full_exog = pd.concat([train[exog_cols], val[exog_cols]], axis=0)
    full_exog = full_exog.ffill().bfill()

    scaler = StandardScaler()
    scaler.fit(full_exog.loc[train.index])
    full_scaled = pd.DataFrame(
        scaler.transform(full_exog),
        index=full_exog.index,
        columns=exog_cols,
    )
    train_exog = full_scaled.loc[train.index]
    val_exog = full_scaled.loc[val.index]

    series = train[target_col].dropna()
    train_exog_aligned = train_exog.loc[series.index]

    auto_sarimax = pm.auto_arima(
        series,
        X=train_exog_aligned,
        seasonal=True, m=7,
        stepwise=True,
        suppress_warnings=True, error_action="ignore",
    )
    sarima_order = auto_sarimax.order
    seasonal_order = auto_sarimax.seasonal_order
    logger.info(
        "SARIMAX auto-selected: order=%s seasonal=%s",
        sarima_order, seasonal_order,
    )

    fitted = SARIMAX(
        series,
        exog=train_exog_aligned,
        order=sarima_order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)

    history_y    = series.tolist()
    history_exog = train_exog_aligned.values.tolist()
    predictions  = []

    for i in range(len(val)):
        exog_now = val_exog.iloc[i : i + 1]
        yhat     = fitted.forecast(steps=1, exog=exog_now)[0]
        predictions.append(yhat)

        history_y.append(val[target_col].iloc[i])
        history_exog.append(val_exog.iloc[i].values)

        if (i + 1) % refit_every == 0:
            fitted = SARIMAX(
                history_y,
                exog=history_exog,
                order=sarima_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)

    logger.info(
        "SARIMAX %s × %s walk-forward complete (%d steps)",
        sarima_order, seasonal_order, len(val),
    )
    return np.array(predictions)

def _make_prophet_df(
    df: pd.DataFrame,
    regressors: list[str],
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Convert a date-indexed DataFrame to Prophet's ``ds / y`` format."""
    out = pd.DataFrame({"ds": df.index})
    if target_col and target_col in df.columns:
        out["y"] = df[target_col].values
    for col in regressors:
        out[col] = df[col].values if col in df.columns else 0
    return out.reset_index(drop=True)


def train_predict_prophet(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "PM10_transformed",
    regressors: list[str] = None,
    seasonality_mode: str = "multiplicative",
    changepoint_prior_scale: float = 0.05,
) -> np.ndarray:
    """Fit Prophet with meteorological regressors and a rolling autoregressive
    feature, then predict the validation set in a single forward pass.

    A 7-day lagged rolling mean of ``PM10_transformed`` (``rolling_bc_7d``)
    is appended as an extra regressor to provide autoregressive memory.  The
    validation version of this feature is seeded from the last 7 days of
    training to avoid forward-fill leakage.

    Parameters
    ----------
    train:
        Training DataFrame.
    val:
        Validation DataFrame.
    target_col:
        Box-Cox-transformed PM10 column.
    regressors:
        Exogenous regressor column names (excluding the rolling lag).
    seasonality_mode:
        ``"multiplicative"`` (default) or ``"additive"``.
    changepoint_prior_scale:
        Trend flexibility.

    Returns
    -------
    np.ndarray
        Predicted ``yhat`` values for ``val`` (length == ``len(val)``).
    """
    if regressors is None:
        regressors = []

    train_p = train.copy()
    val_p   = val.copy()

    train_p["rolling_bc_7d"] = (
        train_p[target_col].shift(1).rolling(7, min_periods=3).mean().ffill()
    )
    seed = train_p[target_col].tail(7)
    combined_bc = pd.concat([seed, val_p[target_col].shift(1)])
    val_p["rolling_bc_7d"] = (
        combined_bc.rolling(7, min_periods=3).mean().loc[val_p.index]
    )

    all_regressors = regressors + ["rolling_bc_7d"]
    required = all_regressors + [target_col]
    train_clean = train_p.dropna(subset=required)

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    m.add_country_holidays(country_name="PL")
    for col in all_regressors:
        m.add_regressor(col)

    m.fit(_make_prophet_df(train_clean, all_regressors, target_col))
    forecast = m.predict(_make_prophet_df(val_p, all_regressors))

    logger.info("Prophet forecast complete (%d val rows)", len(val))
    return forecast["yhat"].values

def train_predict_lgbm(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str = "PM10_transformed",
    feature_cols: list[str] = None,
    params: dict = None,
    early_stopping_rounds: int = 100,
    es_fraction: float = 0.15,
    use_optuna: bool = False,
    n_optuna_trials: int = 50,
) -> tuple[np.ndarray, lgb.LGBMRegressor]:
    """Train a LightGBM regressor and predict the validation set.

    Training flow:
    1. The last ``es_fraction`` of training rows are held out for early
       stopping (chronological, no shuffle).
    2. If ``use_optuna=True``, an Optuna study searches hyper-parameters using
       the early-stopping set for evaluation, then the best config is used for
       the final model trained on the full training set.
    3. The final model is trained on ALL training rows (no held-out set) using
       the ``best_n_estimators`` found during early stopping.

    Parameters
    ----------
    train:
        Training DataFrame.
    val:
        Validation DataFrame.
    target_col:
        Box-Cox-transformed PM10 column.
    feature_cols:
        Feature column names.  Columns not present in ``train`` are silently
        dropped.
    params:
        LightGBM hyper-parameters.  Defaults to ``config.LGBM_PARAMS``.
    early_stopping_rounds:
        Patience for early stopping.
    es_fraction:
        Fraction of training rows used for the early-stopping holdout.
    use_optuna:
        If ``True``, run an Optuna hyperparameter search before final training.
    n_optuna_trials:
        Number of Optuna trials (only used when ``use_optuna=True``).

    Returns
    -------
    predictions : np.ndarray
        Predictions for ``val``.
    model : lgb.LGBMRegressor
        Fitted model (useful for SHAP and feature importance analysis).
    """
    if params is None:
        from src.config import LGBM_PARAMS
        params = LGBM_PARAMS.copy()

    if feature_cols is None:
        from src.config import LGBM_FEATURES
        feature_cols = LGBM_FEATURES

    feature_cols = [
        c for c in feature_cols
        if c in train.columns and train[c].notna().sum() > 0
        and train[c].dtype != object
    ]

    X_train_full = train[feature_cols].ffill().fillna(0)
    y_train_full = train[target_col]

    X_val = val[feature_cols].ffill().fillna(0)
    y_val = val[target_col]

    X_tr, X_es, y_tr, y_es = train_test_split(
        X_train_full, y_train_full,
        test_size=es_fraction,
        shuffle=False,
    )

    if use_optuna:
        params, best_n = _optuna_search(
            X_tr, y_tr, X_es, y_es,
            n_trials=n_optuna_trials,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        probe = lgb.LGBMRegressor(**{**params, "n_estimators": 3000})
        probe.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            callbacks=[
                lgb.early_stopping(early_stopping_rounds, verbose=False),
            ],
        )
        best_n = probe.best_iteration_

    final_params = {**params, "n_estimators": best_n}
    model = lgb.LGBMRegressor(**final_params)
    model.fit(X_train_full, y_train_full)

    predictions = model.predict(X_val)
    logger.info(
        "LightGBM trained: best_n_estimators=%d, val_rows=%d",
        best_n, len(X_val),
    )
    return predictions, model


def _optuna_search(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_es: pd.DataFrame,
    y_es: pd.Series,
    n_trials: int = 50,
    early_stopping_rounds: int = 100,
) -> tuple[dict, int]:
    """Run an Optuna MAE-minimisation study and return best params + n_estimators.

    The best-trial ``n_estimators`` is attached to each trial via
    ``trial.set_user_attr`` and retrieved from ``study.best_trial`` after
    optimisation — a plain module-level list would only remember the *last*
    trial executed, not the best one.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError as exc:
        raise ImportError(
            "Install optuna to use hyperparameter search: pip install optuna"
        ) from exc

    def objective(trial: optuna.Trial) -> float:
        p = {
            "objective": "regression_l1",
            "metric": "mae",
            "n_estimators": 3000,
            "verbosity": -1,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }
        mdl = lgb.LGBMRegressor(**p)
        mdl.fit(
            X_tr, y_tr,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
        )

        trial.set_user_attr("best_n", int(mdl.best_iteration_ or 500))

        preds = mdl.predict(X_es)
        return float(np.mean(np.abs(y_es.values - preds)))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_n = int(study.best_trial.user_attrs.get("best_n", 500))

    best_params = {
        **study.best_params,
        "objective": "regression_l1",
        "metric": "mae",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    logger.info(
        "Optuna best MAE=%.4f  best_n=%d  params=%s",
        study.best_value, best_n, study.best_params,
    )
    return best_params, best_n