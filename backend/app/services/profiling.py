# backend/app/services/profiling.py
"""
Lightweight dataset profiling utilities for SynaptiQ 2.0, Phase 2.

- Loads datasets safely with sampling to avoid OOM.
- Computes compact stats using pandas only.
- Returns JSON-ready dictionaries (validated by Pydantic in routers).
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Any, List, Tuple

import pandas as pd
import numpy as np


MAX_SAMPLE_ROWS = 50_000  # Upper bound for in-memory profiling


def _infer_loader(path: str):
    """Return a callable that loads the file at `path` into a pandas DataFrame."""
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv"]:
        return lambda: pd.read_csv(path)
    if ext in [".tsv"]:
        return lambda: pd.read_csv(path, sep="\t")
    if ext in [".xlsx", ".xls"]:
        return lambda: pd.read_excel(path)
    if ext in [".json"]:
        # If JSON lines, try lines=True first, fallback to normal
        def _load_json():
            try:
                return pd.read_json(path, lines=True)
            except ValueError:
                return pd.read_json(path)
        return _load_json
    if ext in [".parquet"]:
        return lambda: pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {ext}")


def _safe_sample(df: pd.DataFrame, max_rows: int = MAX_SAMPLE_ROWS) -> pd.DataFrame:
    """Return `df` if small; otherwise take a stratified-ish random sample on index."""
    if len(df) <= max_rows:
        return df
    # Uniform random sample; we avoid groupby sampling to keep generic.
    return df.sample(n=max_rows, random_state=17).reset_index(drop=True)


def _to_py(obj: Any) -> Any:
    """Convert numpy/pandas scalars to plain Python types for JSON safety."""
    if isinstance(obj, (np.generic,)):
        return obj.item()
    return obj


def build_preview(df: pd.DataFrame, n: int = 10) -> Dict[str, Any]:
    """Return preview JSON: first N rows (records), column order, shape, and dtypes."""
    n = max(1, int(n))
    head_df = df.head(n)
    # Convert to records (JSON-serializable)
    records = head_df.replace({np.nan: None}).to_dict(orient="records")
    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    return {
        "rows": len(df),
        "cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": dtypes,
        "data": records,
    }


def build_profile(dataset_id: str, name: str, df_full: pd.DataFrame) -> Dict[str, Any]:
    """Compute the compact profile spec required by Phase 2."""
    # Work on sampled view for speed/memory
    df = _safe_sample(df_full)

    rows, cols = df.shape
    columns = list(df.columns)

    # Basic per-column stats
    col_meta: List[Dict[str, Any]] = []
    top_values: Dict[str, List[Dict[str, Any]]] = {}
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in columns:
        series = df[col]
        dtype_str = str(series.dtype)

        missing_count = int(series.isna().sum())
        missing_pct = (missing_count / rows * 100.0) if rows else 0.0
        unique_count = int(series.nunique(dropna=True))

        # Sample values (up to 5 distinct non-null values)
        sample_vals = (
            series.dropna().astype(str).head(5).tolist()
            if rows
            else []
        )

        col_meta.append({
            "name": col,
            "dtype": dtype_str,
            "missing": missing_count,
            "missing_pct": round(missing_pct, 3),
            "unique": unique_count,
            "sample_values": sample_vals,
        })

        # Top 5 values
        vc = series.value_counts(dropna=True).head(5)
        top_values[col] = [
            {"value": _to_py(idx), "count": int(ct)}
            for idx, ct in vc.items()
        ]

        # Type split (treat bool as categorical to avoid misleading correlations)
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    # Overall missing %
    if rows * cols > 0:
        missing_overall_pct = float(df.isna().sum().sum() / (rows * cols) * 100.0)
    else:
        missing_overall_pct = 0.0

    # Numeric summary
    numeric_summary: Dict[str, Dict[str, Any]] = {}
    if numeric_cols:
        desc = df[numeric_cols].describe(include=[np.number])
        # We want min, max, mean, std
        for col in numeric_cols:
            col_desc = {
                "min": _to_py(desc.loc["min", col]) if "min" in desc.index else None,
                "max": _to_py(desc.loc["max", col]) if "max" in desc.index else None,
                "mean": _to_py(desc.loc["mean", col]) if "mean" in desc.index else None,
                "std": _to_py(desc.loc["std", col]) if "std" in desc.index else None,
            }
            numeric_summary[col] = col_desc

    # Pearson correlation
    corr_payload = {"method": "pearson", "matrix": [], "columns": []}
    if len(numeric_cols) >= 2:
        corr_df = df[numeric_cols].corr(method="pearson")
        corr_payload["columns"] = list(corr_df.columns)
        corr_payload["matrix"] = [
            [(_to_py(v) if not pd.isna(v) else None) for v in row]
            for row in corr_df.values.tolist()
        ]

    return {
        "dataset_id": dataset_id,
        "name": name,
        "rows": rows,
        "cols": cols,
        "columns": col_meta,
        "missing_overall_pct": round(missing_overall_pct, 3),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_summary": numeric_summary,
        "top_values": top_values,
        "corr": corr_payload,
    }


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    """Load a dataset file at `path` to DataFrame, with a few pragmatic fallbacks."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")
    loader = _infer_loader(path)
    df = loader()

    # Basic normalization: keep column name strings; avoid object dtype explosion
    df.columns = [str(c) for c in df.columns]
    return df
