# backend/app/schemas/dataset.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class DatasetPreviewResponse(BaseModel):
    ok: bool = True
    dataset_id: str
    name: str
    rows: int
    cols: int
    columns: List[str]
    dtypes: Dict[str, str]
    data: List[Dict[str, Any]]


class ColumnMeta(BaseModel):
    name: str
    dtype: str
    missing: int
    missing_pct: float
    unique: int
    sample_values: List[str]


class CorrPayload(BaseModel):
    method: str
    matrix: List[List[Optional[float]]]
    columns: List[str]


class DatasetProfileResponse(BaseModel):
    ok: bool = True
    dataset_id: str
    name: str
    rows: int
    cols: int
    columns: List[ColumnMeta]
    missing_overall_pct: float
    numeric_cols: List[str]
    categorical_cols: List[str]
    numeric_summary: Dict[str, Dict[str, Optional[float]]]
    top_values: Dict[str, List[Dict[str, Any]]]
    corr: CorrPayload
