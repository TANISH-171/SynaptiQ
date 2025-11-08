import json
from pathlib import Path
from typing import Dict, Optional

REGISTRY_PATH = Path(".data/registry.json")
DATA_DIR = Path(".data/datasets")
DATA_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load() -> Dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"datasets": {}}

def _save(data: Dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, indent=2))

def register_dataset(dataset_id: str, meta: Dict):
    data = _load()
    data["datasets"][dataset_id] = meta
    _save(data)

def get_dataset(dataset_id: str) -> Optional[Dict]:
    data = _load()
    return data["datasets"].get(dataset_id)

def list_datasets() -> Dict[str, Dict]:
    data = _load()
    return data["datasets"]
