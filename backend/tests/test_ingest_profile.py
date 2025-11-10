# backend/tests/test_ingest_profile.py
import os
import json
import tempfile
import pandas as pd
from fastapi.testclient import TestClient

# Wire up app depending on your project layout
# Assuming your FastAPI app is created in backend/app/main.py as `app`
from backend.app.main import app  # adjust if needed
from backend.app.services import registry


def _write_registry(tmp_path):
    os.environ["SYNAPTIQ_REGISTRY_PATH"] = os.path.join(tmp_path, "registry.json")


def test_preview_and_profile_happy_path(tmp_path):
    _write_registry(tmp_path)

    # Create a small CSV
    csv_path = os.path.join(tmp_path, "tiny.csv")
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [10.0, 9.5, 8.1, 7.2, 6.0],
        "c": ["x", "y", "y", "z", "x"]
    })
    df.to_csv(csv_path, index=False)

    # Register dataset
    ds = {
        "id": "ds1",
        "name": "Tiny",
        "path": os.path.abspath(csv_path),
        "mime": "text/csv"
    }
    registry.upsert_dataset(ds)

    client = TestClient(app)

    # Preview
    r = client.get("/ingest/preview/ds1?n=3")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["dataset_id"] == "ds1"
    assert body["rows"] == 5
    assert body["cols"] == 3
    assert len(body["data"]) == 3
    assert "a" in body["dtypes"]

    # Profile
    r2 = client.get("/ingest/profile/ds1")
    assert r2.status_code == 200
    prof = r2.json()
    assert prof["ok"] is True
    assert prof["rows"] == 5
    assert "numeric_cols" in prof
    assert "categorical_cols" in prof
    assert "columns" in prof and len(prof["columns"]) == 3
    assert "top_values" in prof
    assert "corr" in prof
    # corr should have columns either empty or size >= 2 depending on detected numeric cols
    assert "columns" in prof["corr"]


def test_invalid_dataset_id(tmp_path):
    _write_registry(tmp_path)
    client = TestClient(app)
    r = client.get("/ingest/preview/does_not_exist")
    assert r.status_code == 404
    r2 = client.get("/ingest/profile/does_not_exist")
    assert r2.status_code == 404
