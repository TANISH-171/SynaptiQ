from pydantic import BaseModel
from typing import Any, Dict, List

class TaskStep(BaseModel):
    action: str                 # e.g., "ingest", "train", "forecast"
    params: Dict[str, Any] = {} # model="lightgbm", horizon=12, target="sales"

class TaskPlan(BaseModel):
    query: str
    steps: List[TaskStep]
