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
