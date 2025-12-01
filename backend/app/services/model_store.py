# backend/app/services/model_store.py

"""
Model Artifact Storage (Phase 3)
--------------------------------
Compatible with the new class-based registry system.
Stores model pickle files under: artifacts/<model_id>.pkl

Provides:
    save_artifact(model_id, estimator)
    load_artifact(path)
    remove_artifact(path)
"""

import os
import pickle
from typing import Any

# New Phase-3 registry
from .registry import registry

ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Save model object
# ---------------------------------------------------------
def save_artifact(model_id: str, estimator: Any) -> str:
    """
    Save sklearn/statsmodels/prophet model to artifact file.
    Returns:
        str artifact_path
    """
    path = os.path.join(ARTIFACT_DIR, f"{model_id}.pkl")

    with open(path, "wb") as f:
        pickle.dump(estimator, f)

    return path


# ---------------------------------------------------------
# Load model object
# ---------------------------------------------------------
def load_artifact(path: str) -> Any:
    """
    Load model pickle file from disk.
    """
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------
# Remove model artifact
# ---------------------------------------------------------
def remove_artifact(path: str) -> bool:
    """
    Delete the artifact file.
    Returns:
        True if removed, False if file missing.
    """
    if path and os.path.exists(path):
        try:
            os.remove(path)
            return True
        except Exception:
            return False
    return False
