from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# ============================================================
# TRAIN REQUEST
# ============================================================
class TabularTrainConfig(BaseModel):
    dataset_id: str
    target: str
    features: List[str]
    test_size: float = 0.2
    search_iter: int = 5


class ForecastTrainConfig(BaseModel):
    dataset_id: str
    date_col: str
    target: str
    horizon: int
    freq: str = "D"
    seasonal_period: int = 7


class ModelTrainRequest(BaseModel):
    task_type: str                      # "classification" | "regression" | "forecast"
    tabular: Optional[TabularTrainConfig] = None
    forecast: Optional[ForecastTrainConfig] = None


# ============================================================
# TRAIN RESPONSE
# ============================================================
class ModelTrainResponse(BaseModel):
    ok: bool
    model_id: str
    best: Dict[str, Any]
    leaderboard: List[Dict[str, Any]]


# ============================================================
# LIST MODELS
# ============================================================
class ModelListResponse(BaseModel):
    ok: bool
    models: List[Dict[str, Any]]


# ============================================================
# PREDICT REQUEST
# ============================================================
class ModelPredictRequest(BaseModel):
    # For tabular prediction: list of dicts
    records: Optional[List[Dict[str, Any]]] = None

    # For forecasting
    horizon: Optional[int] = None


# ============================================================
# PREDICT RESPONSE
# ============================================================
class TabularPredictResponse(BaseModel):
    ok: bool
    predictions: List[float]


class ForecastPredictResponse(BaseModel):
    ok: bool
    forecast: List[float]
    history_dates: List[str]
    history_values: List[float]


# ============================================================
# METRICS RESPONSE
# ============================================================
class ModelMetricsResponse(BaseModel):
    ok: bool
    metadata: Dict[str, Any]
