# frontend/utils/api_client.py
import os
import requests

API_BASE = os.environ.get("SYNAPTIQ_API_BASE", "http://localhost:8000")

class APIClient:
    def __init__(self, base_url: str = API_BASE, timeout: int = 25):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # Existing helpers for /ingest/list and upload presumed present from Phase 1
    def list_datasets(self):
        url = f"{self.base_url}/ingest/list"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_preview(self, dataset_id: str, n: int = 10):
        url = f"{self.base_url}/ingest/preview/{dataset_id}"
        r = requests.get(url, params={"n": n}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_profile(self, dataset_id: str):
        url = f"{self.base_url}/ingest/profile/{dataset_id}"
        r = requests.get(url, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
