# backend/app/services/registry.py
from __future__ import annotations

import json
import os
from typing import Dict, Any, Optional
from threading import RLock

_DEFAULT_REG_PATH = os.environ.get("SYNAPTIQ_REGISTRY_PATH", "registry.json")
_lock = RLock()


def _ensure_registry(reg_path: str) -> Dict[str, Any]:
    if not os.path.exists(reg_path):
        return {"datasets": {}}
    with open(reg_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {"datasets": {}}


def _write_registry(reg_path: str, data: Dict[str, Any]) -> None:
    tmp = reg_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, reg_path)


def list_datasets(reg_path: str = _DEFAULT_REG_PATH) -> Dict[str, Any]:
    with _lock:
        data = _ensure_registry(reg_path)
        return data.get("datasets", {})


def get_dataset(dataset_id: str, reg_path: str = _DEFAULT_REG_PATH) -> Optional[Dict[str, Any]]:
    with _lock:
        data = _ensure_registry(reg_path)
        return data.get("datasets", {}).get(dataset_id)


def upsert_dataset(entry: Dict[str, Any], reg_path: str = _DEFAULT_REG_PATH) -> Dict[str, Any]:
    if not {"id", "name", "path"}.issubset(entry.keys()):
        raise ValueError("Dataset entry must include 'id', 'name', and 'path'.")
    with _lock:
        data = _ensure_registry(reg_path)
        data.setdefault("datasets", {})
        existing = data["datasets"].get(entry["id"], {})
        if "profile" in existing and "profile" not in entry:
            entry["profile"] = existing["profile"]
        data["datasets"][entry["id"]] = entry
        _write_registry(reg_path, data)
        return entry


def set_profile(dataset_id: str, profile: Dict[str, Any], reg_path: str = _DEFAULT_REG_PATH) -> Dict[str, Any]:
    with _lock:
        data = _ensure_registry(reg_path)
        if dataset_id not in data.get("datasets", {}):
            raise KeyError(f"Dataset '{dataset_id}' not found in registry.")
        data["datasets"][dataset_id]["profile"] = profile
        _write_registry(reg_path, data)
        return data["datasets"][dataset_id]


def get_profile(dataset_id: str, reg_path: str = _DEFAULT_REG_PATH) -> Optional[Dict[str, Any]]:
    with _lock:
        data = _ensure_registry(reg_path)
        ds = data.get("datasets", {}).get(dataset_id)
        if not ds:
            return None
        return ds.get("profile")


def registry_path() -> str:
    return _DEFAULT_REG_PATH
