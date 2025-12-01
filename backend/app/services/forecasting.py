# backend/app/services/forecasting.py

"""
Forecasting Service
-------------------
Provides time-series forecasting using:
- Naive
- Seasonal Naive
- ARIMA (statsmodels)
- Optional Prophet (auto-skipped if missing)

Includes:
- Rolling-window backtesting
- MAE, RMSE, MAPE metrics
- Model selection by MAE
- Forecast generation

Used by /models/train when task_type="forecast".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
import os
from statsmodels.tsa.arima.model import ARIMA

# Handling Prophet safely
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# ------------------------------------------------------------
# Metric functions
# ------------------------------------------------------------
def ts_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE for forecasting."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    if np.all(y_true != 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        mape = None

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ------------------------------------------------------------
# Naive model
# ------------------------------------------------------------
def naive_forecast(series: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast future by repeating last value."""
    return np.repeat(series[-1], horizon)


# ------------------------------------------------------------
# Seasonal naive model
# ------------------------------------------------------------
def seasonal_naive_forecast(series: np.ndarray, horizon: int, seasonal_period: int) -> np.ndarray:
    """
    Seasonal naive forecast: repeat last seasonal pattern.
    """
    if len(series) < seasonal_period:
        # fallback to naive
        return naive_forecast(series, horizon)

    pattern = series[-seasonal_period:]
    repeats = int(np.ceil(horizon / seasonal_period))
    out = np.tile(pattern, repeats)[:horizon]
    return out


# ------------------------------------------------------------
# ARIMA model
# ------------------------------------------------------------
def fit_arima(series: np.ndarray, order=(1, 1, 1)) -> Any:
    """Fit simple ARIMA model."""
    try:
        model = ARIMA(series, order=order)
        model_fit = model.fit()
        return model_fit
    except Exception:
        return None


def arima_predict(model_fit: Any, horizon: int) -> np.ndarray:
    """Forecast using fitted ARIMA model."""
    try:
        fc = model_fit.forecast(steps=horizon)
        return np.array(fc)
    except Exception:
        return None


# ------------------------------------------------------------
# Prophet model (optional)
# ------------------------------------------------------------
def fit_prophet(df: pd.DataFrame, date_col: str, target_col: str) -> Optional[Any]:
    if not PROPHET_AVAILABLE:
        return None

    try:
        df_p = df[[date_col, target_col]].rename(columns={date_col: "ds", target_col: "y"})
        model = Prophet()
        model.fit(df_p)
        return model
    except Exception:
        return None


def prophet_predict(model: Any, horizon: int) -> Optional[np.ndarray]:
    """Prophet future forecast."""
    if model is None:
        return None
    try:
        future = model.make_future_dataframe(periods=horizon)
        fc = model.predict(future)["yhat"].tail(horizon).to_numpy()
        return fc
    except Exception:
        return None


# ------------------------------------------------------------
# Rolling-window backtesting for evaluation
# ------------------------------------------------------------
def rolling_backtest(series: np.ndarray, horizon: int, seasonal_period: int = None) -> Dict[str, Any]:
    """
    Simple rolling-window backtest for:
    - Naive
    - Seasonal Naive
    - ARIMA

    Prophet is skipped in backtest to keep runtime low.
    """

    n = len(series)
    if n < horizon * 2:
        raise ValueError("Time-series too short for rolling backtest.")

    errors = {
        "naive": [],
        "seasonal_naive": [],
        "arima": [],
    }

    for start in range(n - 2 * horizon, n - horizon):
        train = series[:start]
        test = series[start:start + horizon]

        # Naive
        naive_pred = naive_forecast(train, horizon)
        errors["naive"].append(ts_metrics(test, naive_pred)["MAE"])

        # Seasonal naive
        if seasonal_period:
            sn_pred = seasonal_naive_forecast(train, horizon, seasonal_period)
        else:
            sn_pred = naive_forecast(train, horizon)
        errors["seasonal_naive"].append(ts_metrics(test, sn_pred)["MAE"])

        # ARIMA
        arima_model = fit_arima(train)
        if arima_model:
            ar_pred = arima_predict(arima_model, horizon)
        else:
            ar_pred = None

        if ar_pred is not None:
            errors["arima"].append(ts_metrics(test, ar_pred)["MAE"])
        else:
            errors["arima"].append(float("inf"))

    # Average MAE for each model
    avg_mae = {k: float(np.mean(v)) for k, v in errors.items()}

    # Pick best
    best_name = min(avg_mae, key=lambda k: avg_mae[k])

    return {
        "avg_mae": avg_mae,
        "best_model_name": best_name,
    }


# ------------------------------------------------------------
# Full Forecasting Pipeline
# ------------------------------------------------------------
def run_forecasting(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int,
    freq: Optional[str] = None,
    seasonal_period: Optional[int] = None
) -> Dict[str, Any]:
    """
    Full forecasting pipeline:
    - Validate inputs
    - Sort by date
    - Rolling evaluation
    - Fit best model
    - Forecast next H
    """

    # 1. Sort by date
    df = df.sort_values(by=date_col)
    df = df.reset_index(drop=True)

    if freq:
        df = df.set_index(date_col).asfreq(freq).reset_index()

    series = df[target_col].astype(float).to_numpy()

    # 2. Run rolling-backtest to choose model
    eval_result = rolling_backtest(series, horizon, seasonal_period)

    best_name = eval_result["best_model_name"]

    # 3. Fit final model on full data
    if best_name == "naive":
        final_pred = naive_forecast(series, horizon)

    elif best_name == "seasonal_naive":
        final_pred = seasonal_naive_forecast(series, horizon, seasonal_period)

    elif best_name == "arima":
        ar_model = fit_arima(series)
        final_pred = arima_predict(ar_model, horizon)
        if final_pred is None:
            # fallback if ARIMA fails
            final_pred = naive_forecast(series, horizon)

    else:
        final_pred = naive_forecast(series, horizon)

    # 4. Optional Prophet forecast (used ONLY if Prophet performs better later)
    prophet_pred = None
    if PROPHET_AVAILABLE:
        model = fit_prophet(df, date_col, target_col)
        if model:
            prophet_pred = prophet_predict(model, horizon)

    return {
        "best_model": best_name,
        "evaluation": eval_result,
        "forecast": list(map(float, final_pred)),
        "history_dates": df[date_col].tolist(),
        "history_values": df[target_col].astype(float).tolist(),
        "prophet_available": PROPHET_AVAILABLE,
        "prophet_forecast": list(map(float, prophet_pred)) if prophet_pred is not None else None
    }

def train_forecast_model(
    df: pd.DataFrame,
    date_col: str,
    target: str,
    horizon: int,
    freq: str = "D",
    seasonal_period: int = 7,
    model_path: str = None,
) -> Dict[str, Any]:
    """
    PyTest-compatible forecasting trainer.
    - Fits simple SARIMAX(1,1,1)(1,1,1,s)
    - Saves model to disk
    - Returns JSON-safe history arrays
    """

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    series = df[target].astype(float).to_numpy()

    # Fit SARIMAX
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, seasonal_period),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)

    # Where to save
    if model_path is None:
        os.makedirs("models", exist_ok=True)
        model_path = f"models/forecast_{abs(hash(os.urandom(4)))}.pkl"

    fitted.save(model_path)

    return {
        "model_path": model_path,
        "history_dates": [d.isoformat() for d in df[date_col].tolist()],
        "history_values": [float(v) for v in series],
    }


def forecast_with_model(
    model_path: str,
    horizon: int,
) -> Dict[str, Any]:
    """
    PyTest-compatible SARIMAX prediction function.
    """

    fitted = SARIMAXResults.load(model_path)

    pred = fitted.forecast(steps=horizon)

    # Convert to JSON-safe floats, replace NaN with None
    pred = [
        float(v) if v is not None and not np.isnan(v) else None
        for v in pred
    ]

    return {"forecast": pred}