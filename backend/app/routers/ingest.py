# backend/app/routers/ingest.py
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from typing import Any

from ..services import registry
from ..services.profiling import load_dataframe_from_path, build_preview, build_profile
from ..schemas.dataset import DatasetPreviewResponse, DatasetProfileResponse

router = APIRouter(tags=["ingest"])


@router.get("/list")
def list_datasets() -> Any:
    """
    Return datasets currently registered.
    If a profile exists it will be included under each dataset object.
    """
    datasets = registry.list_datasets()
    return {"ok": True, "datasets": datasets}


@router.get("/preview/{dataset_id}", response_model=DatasetPreviewResponse)
def get_preview(dataset_id: str, n: int = Query(10, ge=1, le=1000)) -> DatasetPreviewResponse:
    """
    Return first N rows for a dataset, with column order and dtype map.
    """
    ds = registry.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    path = ds.get("path")
    name = ds.get("name", dataset_id)
    try:
        df = load_dataframe_from_path(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    payload = build_preview(df, n=n)
    return DatasetPreviewResponse(
        ok=True,
        dataset_id=dataset_id,
        name=name,
        rows=payload["rows"],
        cols=payload["cols"],
        columns=payload["columns"],
        dtypes=payload["dtypes"],
        data=payload["data"],
    )


@router.get("/profile/{dataset_id}", response_model=DatasetProfileResponse)
def get_profile(dataset_id: str) -> DatasetProfileResponse:
    """
    Compute (or return existing) profile for a dataset and persist to registry.
    """
    ds = registry.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    # If we already have a profile, return it (idempotent & fast)
    existing = registry.get_profile(dataset_id)
    if existing:
        return DatasetProfileResponse(ok=True, **existing)

    path = ds.get("path")
    name = ds.get("name", dataset_id)
    try:
        df = load_dataframe_from_path(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    try:
        profile = build_profile(dataset_id=dataset_id, name=name, df_full=df)
        # Persist profile
        registry.set_profile(dataset_id, profile)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute profile: {e}")

    return DatasetProfileResponse(ok=True, **profile)
