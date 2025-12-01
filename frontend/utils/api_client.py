# frontend/utils/api_client.py

import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")


# -------------------------------
# Simple helper for handling errors
# -------------------------------
def _safe_request(method, url, **kwargs):
    try:
        r = requests.request(method, url, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise RuntimeError(f"API request failed: {e}")


# -------------------------------
# Health
# -------------------------------
def health():
    return _safe_request("GET", f"{API_URL}/health")


# -------------------------------
# Datasets (Phase 1 & 2)
# -------------------------------
def upload_dataset(file):
    files = {"file": (file.name, file.getvalue())}
    return _safe_request("POST", f"{API_URL}/ingest/upload", files=files)


def list_datasets():
    return _safe_request("GET", f"{API_URL}/ingest/list")


def route_nlq(query: str):
    return _safe_request("POST", f"{API_URL}/nlq/route", params={"query": query})


# -------------------------------
# APIClient class for Streamlit
# -------------------------------
class APIClient:
    def __init__(self, base_url: str | None = None, timeout: int = 25):
        self.base_url = (base_url or API_URL).rstrip("/")
        self.timeout = timeout

    # ---------- Health ----------
    def health(self):
        return _safe_request("GET", f"{self.base_url}/health")

    # ---------- Datasets ----------
    def upload_dataset(self, file):
        files = {"file": (file.name, file.getvalue())}
        return _safe_request("POST", f"{self.base_url}/ingest/upload", files=files)

    def list_datasets(self):
        return _safe_request("GET", f"{self.base_url}/ingest/list")

    def get_preview(self, dataset_id: str, n: int = 10):
        return _safe_request(
            "GET",
            f"{self.base_url}/ingest/preview/{dataset_id}",
            params={"n": n},
        )

    def get_profile(self, dataset_id: str):
        return _safe_request("GET", f"{self.base_url}/ingest/profile/{dataset_id}")

    def route_nlq(self, query: str):
        return _safe_request(
            "POST",
            f"{self.base_url}/nlq/route",
            params={"query": query},
        )

    # ---------- Models: Training ----------
    def train_model(self, payload: dict):
        """POST /models/train"""
        return _safe_request(
            "POST",
            f"{self.base_url}/models/train",
            json=payload,
        )

    # ---------- Models: List ----------
    def list_models(self):
        return _safe_request("GET", f"{self.base_url}/models/list")

    # ---------- Models: Metrics ----------
    def get_model_metrics(self, model_id: str):
        return _safe_request("GET", f"{self.base_url}/models/{model_id}/metrics")

    # ---------- Models: Prediction ----------
    def predict_model(self, model_id: str, payload: dict):
        return _safe_request(
            "POST",
            f"{self.base_url}/models/{model_id}/predict",
            json=payload,
        )

    # ---------- Models: Delete ----------
    def delete_model(self, model_id: str):
        return _safe_request("DELETE", f"{self.base_url}/models/{model_id}")
