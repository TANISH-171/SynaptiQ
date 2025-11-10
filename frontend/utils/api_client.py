import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

def health():
    return requests.get(f"{API_URL}/health").json()

def upload_dataset(file):
    files = {"file": (file.name, file.getvalue())}
    r = requests.post(f"{API_URL}/ingest/upload", files=files, timeout=60)
    r.raise_for_status()
    return r.json()

def list_datasets():
    return requests.get(f"{API_URL}/ingest/list").json()

def route_nlq(query: str):
    return requests.post(f"{API_URL}/nlq/route", params={"query": query}).json()


# Provide a class-based client for pages expecting `APIClient`.
class APIClient:
    def __init__(self, base_url: str | None = None, timeout: int = 25):
        self.base_url = (base_url or API_URL).rstrip("/")
        self.timeout = timeout

    def health(self):
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def upload_dataset(self, file):
        files = {"file": (file.name, file.getvalue())}
        r = requests.post(f"{self.base_url}/ingest/upload", files=files, timeout=60)
        r.raise_for_status()
        return r.json()

    def list_datasets(self):
        r = requests.get(f"{self.base_url}/ingest/list", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def get_preview(self, dataset_id: str, n: int = 10):
        r = requests.get(
            f"{self.base_url}/ingest/preview/{dataset_id}", params={"n": n}, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()

    def get_profile(self, dataset_id: str):
        r = requests.get(f"{self.base_url}/ingest/profile/{dataset_id}", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def route_nlq(self, query: str):
        r = requests.post(f"{self.base_url}/nlq/route", params={"query": query}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()
