# backend/app/services/registry.py

import os
import json
from threading import RLock
from typing import Dict, Any, Optional

_DEFAULT_REG_PATH = os.environ.get("SYNAPTIQ_REGISTRY_PATH", "registry.json")
_lock = RLock()


class Registry:
    def __init__(self, reg_path: str = _DEFAULT_REG_PATH):
        self.reg_path = reg_path
        self.data = self._load()

    # ----------------------------------------------------
    # Internal Load / Save
    # ----------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.reg_path):
            return {"datasets": {}, "models": {}}

        try:
            with open(self.reg_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"datasets": {}, "models": {}}

        data.setdefault("datasets", {})
        data.setdefault("models", {})
        return data

    def save(self):
        tmp = self.reg_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.reg_path)

    # ----------------------------------------------------
    # DATASET METHODS
    # ----------------------------------------------------
    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.data["datasets"].get(dataset_id)

    def upsert_dataset(self, entry: Dict[str, Any]):
        if not {"id", "name", "path"}.issubset(entry):
            raise ValueError("Dataset entry must include id, name, path")

        ds_id = entry["id"]
        existing = self.data["datasets"].get(ds_id, {})

        # keep previous profile if not overwritten
        if "profile" in existing and "profile" not in entry:
            entry["profile"] = existing["profile"]

        self.data["datasets"][ds_id] = entry
        self.save()

    def get_profile(self, dataset_id: str):
        ds = self.data["datasets"].get(dataset_id)
        if not ds:
            return None
        return ds.get("profile")

    def set_profile(self, dataset_id: str, profile: Dict[str, Any]):
        if dataset_id not in self.data["datasets"]:
            raise KeyError(f"Dataset '{dataset_id}' not found")
        self.data["datasets"][dataset_id]["profile"] = profile
        self.save()

    def list_datasets(self) -> Dict[str, Any]:
        """
        Return all registered datasets as a dict keyed by dataset_id.
        """
        return self.data.get("datasets", {})

    # ----------------------------------------------------
    # MODEL METHODS
    # ----------------------------------------------------
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.data["models"].get(model_id)

    def upsert_model(self, model_id: str, entry: Dict[str, Any]):
        self.data["models"][model_id] = entry
        self.save()

    def delete_model(self, model_id: str) -> bool:
        if model_id in self.data["models"]:
            del self.data["models"][model_id]
            self.save()
            return True
        return False

    def list_models(self):
        return self.data["models"]


# =============================================================
# GLOBAL REGISTRY INSTANCE
# =============================================================
registry = Registry()


# =============================================================
# Top-Level Helper Functions (required by tests & routers)
# =============================================================
def upsert_dataset(entry: Dict[str, Any]):
    return registry.upsert_dataset(entry)

def get_dataset(dataset_id: str):
    return registry.get_dataset(dataset_id)

def get_profile(dataset_id: str):
    return registry.get_profile(dataset_id)

def set_profile(dataset_id: str, profile: Dict[str, Any]):
    return registry.set_profile(dataset_id, profile)

def list_datasets():
    return registry.list_datasets()

def upsert_model(model_id: str, entry: Dict[str, Any]):
    return registry.upsert_model(model_id, entry)

def get_model(model_id: str):
    return registry.get_model(model_id)

def delete_model(model_id: str):
    return registry.delete_model(model_id)

def list_models():
    return registry.list_models()

def registry_path():
    return registry.reg_path
