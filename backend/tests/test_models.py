# backend/tests/test_models.py

"""
Phase 3 Test Suite
------------------
Covers:
- Regression AutoML training
- Classification AutoML training
- Forecast training
- Model listing
- Model prediction
- Error handling

Tests use synthetic datasets created at runtime.
"""

import os
import json
import pandas as pd
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.registry import upsert_dataset, delete_model, list_models


client = TestClient(app)


# -----------------------------------------------------------
# Helpers
# -----------------------------------------------------------
def _create_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)


def _insert_dataset(tmpdir, df, name):
    """Insert dataset into registry + write underlying file."""
    dataset_id = f"ds_{name}"
    path = os.path.join(tmpdir, f"{dataset_id}.csv")
    df.to_csv(path, index=False)

    upsert_dataset({
        "id": dataset_id,
        "name": dataset_id,
        "path": path
    })
    return dataset_id, path


# -----------------------------------------------------------
# REGRESSION TRAINING TEST
# -----------------------------------------------------------
def test_train_regression(tmp_path):
    df = pd.DataFrame({
        "x": np.arange(0, 100),
        "y": np.arange(0, 100) * 2 + 5
        })
    dataset_id, path = _insert_dataset(tmp_path, df, "reg")

    payload = {
        "task_type": "regression",
        "tabular": {
            "dataset_id": dataset_id,
            "target": "y",
            "features": ["x"],
            "test_size": 0.2,
            "search_iter": 3
        }
    }

    r = client.post("/models/train", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["ok"] is True
    assert "model_id" in body
    assert len(body["leaderboard"]) > 0

    # Make sure model file exists
    meta = body
    model_id = body["model_id"]
    registry = list_models()
    assert model_id in registry


# -----------------------------------------------------------
# CLASSIFICATION TRAINING TEST
# -----------------------------------------------------------
def test_train_classification(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({
        "age": np.random.randint(18, 60, 200),
        "income": np.random.randint(20000, 90000, 200),
        "label": np.random.choice([0, 1], 200)
    })
    dataset_id, path = _insert_dataset(tmp_path, df, "clf")

    payload = {
        "task_type": "classification",
        "tabular": {
            "dataset_id": dataset_id,
            "target": "label",
            "features": ["age", "income"],
            "test_size": 0.2,
            "search_iter": 3
        }
    }

    r = client.post("/models/train", json=payload)
    assert r.status_code == 200
    body = r.json()

    assert body["ok"] is True
    assert "model_id" in body
    assert "best" in body


# -----------------------------------------------------------
# FORECAST TRAINING TEST
# -----------------------------------------------------------
def test_train_forecast(tmp_path):
    # Create simple synthetic time-series
    dates = pd.date_range(start="2023-01-01", periods=120, freq="D")
    values = np.linspace(10, 50, 120)

    df = pd.DataFrame({"date": dates, "sales": values})
    dataset_id, path = _insert_dataset(tmp_path, df, "ts")

    payload = {
        "task_type": "forecast",
        "forecast": {
            "dataset_id": dataset_id,
            "date_col": "date",
            "target": "sales",
            "horizon": 10,
            "freq": "D",
            "seasonal_period": 7
        }
    }

    r = client.post("/models/train", json=payload)
    assert r.status_code == 200
    body = r.json()

    assert "model_id" in body
    assert len(body["leaderboard"]) == 3


# -----------------------------------------------------------
# LIST MODELS TEST
# -----------------------------------------------------------
def test_list_models():
    r = client.get("/models/list")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body


# -----------------------------------------------------------
# PREDICT (TABULAR)
# -----------------------------------------------------------
def test_tabular_predict(tmp_path):
    df = pd.DataFrame({
        "x": np.arange(0, 50),
        "y": np.arange(0, 50) * 3 + 1
    })
    dataset_id, path = _insert_dataset(tmp_path, df, "pred_reg")

    # Train model
    payload = {
        "task_type": "regression",
        "tabular": {
            "dataset_id": dataset_id,
            "target": "y",
            "features": ["x"],
            "test_size": 0.2
        }
    }

    r = client.post("/models/train", json=payload)
    model_id = r.json()["model_id"]

    # Predict
    pred_payload = {
        "records": [{"x": 10}, {"x": 20}, {"x": 30}]
    }
    r = client.post(f"/models/{model_id}/predict", json=pred_payload)
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 3


# -----------------------------------------------------------
# PREDICT (FORECAST)
# -----------------------------------------------------------
def test_forecast_predict(tmp_path):
    # Time-series
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    values = np.linspace(100, 200, 60)

    df = pd.DataFrame({"date": dates, "sales": values})
    dataset_id, path = _insert_dataset(tmp_path, df, "pred_ts")

    # Train
    payload = {
        "task_type": "forecast",
        "forecast": {
            "dataset_id": dataset_id,
            "date_col": "date",
            "target": "sales",
            "horizon": 5,
            "freq": "D",
            "seasonal_period": 7
        }
    }
    r = client.post("/models/train", json=payload)
    model_id = r.json()["model_id"]

    # Predict
    pred_payload = {"horizon": 5}
    r = client.post(f"/models/{model_id}/predict", json=pred_payload)

    assert r.status_code == 200
    body = r.json()
    assert len(body["forecast"]) == 5


# -----------------------------------------------------------
# BAD REQUEST HANDLING
# -----------------------------------------------------------
def test_invalid_dataset():
    bad_payload = {
        "task_type": "regression",
        "tabular": {
            "dataset_id": "unknown123",
            "target": "y",
            "features": ["x"]
        }
    }
    r = client.post("/models/train", json=bad_payload)
    assert r.status_code == 404


def test_invalid_model_predict():
    pred_payload = {"records": [{"x": 1}]}
    r = client.post("/models/unknown_model/predict", json=pred_payload)
    assert r.status_code == 404
