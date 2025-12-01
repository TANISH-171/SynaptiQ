# backend/app/routers/models.py

from __future__ import annotations

import math
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException

# ✅ Use relative imports so we share the SAME registry as tests
from ..schemas.models import (
    ModelTrainRequest,
    ModelTrainResponse,
    ModelPredictRequest,
    TabularPredictResponse,
    ForecastPredictResponse,
)
from ..services import registry as reg_service
from ..services.model_store import save_artifact, load_artifact
from ..services.automl import train_tabular
from ..services.forecasting import (
    run_forecasting,
    train_forecast_model,
    forecast_with_model,
)

router = APIRouter(tags=["models"])


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _json_safe(obj: Any) -> Any:
    """
    Recursively replace non-finite floats (NaN / inf / -inf) with None
    so that FastAPI's JSON encoder never crashes.
    """
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _sanitize_leaderboard(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure leaderboard entries are JSON-safe and don't include estimators.
    train_tabular() already strips estimators; we just make sure floats are safe.
    """
    cleaned: List[Dict[str, Any]] = []
    for e in entries:
        cleaned.append(_json_safe(e))
    return cleaned


# -------------------------------------------------------------------
# TRAINING ENDPOINT
# -------------------------------------------------------------------
@router.post("/train", response_model=ModelTrainResponse)
def train_model(req: ModelTrainRequest) -> ModelTrainResponse:
    """
    Train models for:
    - Tabular Regression / Classification (AutoML)
    - Time-series Forecasting
    """

    # ---------------------------------------------------------
    # TABULAR (Regression / Classification)
    # ---------------------------------------------------------
    if req.task_type in ("regression", "classification"):
        cfg = req.tabular
        if cfg is None:
            raise HTTPException(400, "Tabular config is required for this task_type")

        # ✅ Use the same registry helpers as tests
        ds = reg_service.get_dataset(cfg.dataset_id)
        if not ds:
            raise HTTPException(404, "Dataset not found")

        df = pd.read_csv(ds["path"])

        # Phase-3 helper that wraps run_automl and makes output JSON-safe
        result = train_tabular(
            df=df,
            target=cfg.target,
            features=cfg.features,
            test_size=cfg.test_size,
            search_iter=cfg.search_iter,
            task_type=req.task_type,
        )

        model_id = f"model_{uuid.uuid4().hex}"

        # Persist estimator artifact
        path = save_artifact(model_id, result["estimator"])

        # Build metadata (no estimators inside, only JSON-safe stuff)
        best = result["best_model"]
        metadata: Dict[str, Any] = {
            "id": model_id,
            "dataset_id": cfg.dataset_id,
            "task_type": result["task_type"],
            "target": cfg.target,
            "features": result["features"],
            "params": best.get("params", {}),
            "metrics": best.get("metrics", {}),
            "path": path,
            "created_at": datetime.utcnow().isoformat(),
        }

        # ✅ Save via the same helper that tests use
        reg_service.upsert_model(model_id, metadata)

        leaderboard = _sanitize_leaderboard(result["leaderboard"])

        return ModelTrainResponse(
            ok=True,
            model_id=model_id,
            best=metadata,
            leaderboard=leaderboard,
        )

    # ---------------------------------------------------------
    # FORECASTING
    # ---------------------------------------------------------
    if req.task_type == "forecast":
        cfg = req.forecast
        if cfg is None:
            raise HTTPException(400, "Forecast config is required for this task_type")

        ds = reg_service.get_dataset(cfg.dataset_id)
        if not ds:
            raise HTTPException(404, "Dataset not found")

        df = pd.read_csv(ds["path"])

        # 1) Use run_forecasting for evaluation (naive / seasonal_naive / arima)
        eval_result = run_forecasting(
            df=df,
            date_col=cfg.date_col,
            target_col=cfg.target,
            horizon=cfg.horizon,
            freq=cfg.freq,
            seasonal_period=cfg.seasonal_period,
        )

        avg_mae = eval_result.get("evaluation", {}).get("avg_mae", {})
        best_name = eval_result.get("best_model")

        def _safe(val):
            try:
                v = float(val)
                return v if math.isfinite(v) else None
            except Exception:
                return None

        # Build a 3-entry leaderboard as required by tests
        leaderboard: List[Dict[str, Any]] = []
        for model_name in ["naive", "seasonal_naive", "arima"]:
            mae = _safe(avg_mae.get(model_name))
            leaderboard.append(
                {
                    "model": model_name,
                    "metrics": {"MAE": mae},
                    "primary_metric": mae,
                }
            )

        # 2) Train a real SARIMAX model for prediction & persist it
        trained = train_forecast_model(
            df=df,
            date_col=cfg.date_col,
            target=cfg.target,
            horizon=cfg.horizon,
            freq=cfg.freq,
            seasonal_period=cfg.seasonal_period,
        )
        model_path = trained["model_path"]
        history_dates = trained["history_dates"]
        history_values = trained["history_values"]

        model_id = f"model_{uuid.uuid4().hex}"

        params = {
            "date_col": cfg.date_col,
            "target": cfg.target,
            "horizon": cfg.horizon,
            "freq": cfg.freq,
            "seasonal_period": cfg.seasonal_period,
        }

        best_mae = _safe(avg_mae.get(best_name)) if best_name in avg_mae else None

        # Metadata stored in registry (JSON-safe)
        metadata: Dict[str, Any] = {
            "id": model_id,
            "dataset_id": cfg.dataset_id,
            "task_type": "forecast",
            "target": cfg.target,
            "features": None,
            "params": params,
            "metrics": {
                "model": best_name,
                "MAE": best_mae,
            },
            "path": model_path,
            "created_at": datetime.utcnow().isoformat(),
            "history_dates": history_dates,
            "history_values": history_values,
        }

        reg_service.upsert_model(model_id, _json_safe(metadata))

        return ModelTrainResponse(
            ok=True,
            model_id=model_id,
            best=metadata,
            leaderboard=_sanitize_leaderboard(leaderboard),
        )

    # ---------------------------------------------------------
    # Invalid task_type
    # ---------------------------------------------------------
    raise HTTPException(400, "Invalid task_type")


# -------------------------------------------------------------------
# LIST MODELS  (HTTP)
# -------------------------------------------------------------------
@router.get("/list")
def list_models_route():
    """
    HTTP endpoint for listing models.

    Tests only assert that:
      - status_code == 200
      - response JSON has key "models"

    So we return a plain dict instead of a Pydantic response model.
    """
    models = reg_service.list_models()
    return {"ok": True, "models": _json_safe(models)}


# -------------------------------------------------------------------
# PREDICT ENDPOINT
# -------------------------------------------------------------------
@router.post("/{model_id}/predict")
def predict(model_id: str, req: ModelPredictRequest):
    """
    - For tabular models: body = {"records": [ {...}, {...} ]}
    - For forecast models: body = {"horizon": 5}
    Matches the Phase-3 test expectations.
    """

    model_meta = reg_service.get_model(model_id)
    if not model_meta:
        raise HTTPException(404, "Model not found")

    task_type = model_meta.get("task_type")

    # ---------------------------------------------------------
    # TABULAR PREDICTION
    # ---------------------------------------------------------
    if task_type in ("regression", "classification"):
        if not req.records:
            raise HTTPException(400, "records are required for tabular prediction")

        path = model_meta.get("path")
        if not path:
            raise HTTPException(500, "Model artifact path missing in registry")

        model = load_artifact(path)

        df = pd.DataFrame(req.records)
        preds = model.predict(df)

        preds_list = [float(p) for p in preds]

        return TabularPredictResponse(ok=True, predictions=preds_list)

    # ---------------------------------------------------------
    # FORECASTING PREDICTION
    # ---------------------------------------------------------
    if task_type == "forecast":
        horizon = req.horizon or model_meta.get("params", {}).get("horizon")
        if not horizon:
            raise HTTPException(400, "horizon is required for forecast prediction")

        path = model_meta.get("path")
        if not path:
            raise HTTPException(500, "Forecast model artifact path missing in registry")

        result = forecast_with_model(model_path=path, horizon=horizon)
        forecast_vals = [
            float(v) if v is not None and math.isfinite(float(v)) else None
            for v in result["forecast"]
        ]

        return ForecastPredictResponse(
            ok=True,
            forecast=forecast_vals,
            history_dates=model_meta.get("history_dates", []),
            history_values=model_meta.get("history_values", []),
        )

    raise HTTPException(400, "Unsupported model type")
