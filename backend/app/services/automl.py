# backend/app/services/automl.py

"""
AutoML Service
--------------
Provides tabular regression & classification model training with:
- Baseline predictor
- Linear Regression / Logistic Regression
- RandomForest (Regressor / Classifier)
- Optional XGBoost (if installed)
- Hyperparameter random search
- Train/test split + CV
- Leaderboard generation
- Full metrics suite for both tasks

Used by /models/train in FastAPI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier

import warnings
warnings.filterwarnings("ignore")


# Try to import XGBoost safely
try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------

def detect_task_type(y: pd.Series) -> str:
    """
    Auto-detects regression or classification.
    Classification if number of unique labels <= 20 and dtype is integer or string.
    """
    unique_vals = y.nunique()

    if y.dtype == object or y.dtype == bool:
        return "classification"

    if unique_vals <= 20:
        return "classification"

    return "regression"


# --------------------------------------------------------------------
# Metric Computation
# --------------------------------------------------------------------
def compute_regression_metrics(y_true, y_pred) -> Dict[str, float]:
    """Regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100) if np.all(y_true != 0) else None
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape) if mape is not None else None,
        "R2": float(r2)
    }


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Classification metrics."""
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # ROC-AUC only for binary classification
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["AUC"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["AUC"] = None
    else:
        metrics["AUC"] = None

    return metrics


# --------------------------------------------------------------------
# Model Training Helper
# --------------------------------------------------------------------
def train_single_model(task: str, model_name: str, model_obj, params_grid: dict,
                       X_train, y_train, X_test, y_test, search_iter: int = 10) -> Dict[str, Any]:
    """
    Train a single model with RandomizedSearchCV and return metrics + best estimator.
    """
    if params_grid:
        search = RandomizedSearchCV(
            estimator=model_obj,
            param_distributions=params_grid,
            n_iter=search_iter,
            scoring="neg_mean_absolute_error" if task == "regression" else "f1_macro",
            cv=3,
            random_state=42,
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_params = search.best_params_
    else:
        # no hyperparameters → simple fit
        model_obj.fit(X_train, y_train)
        best = model_obj
        best_params = {}

    # predictions
    y_pred = best.predict(X_test)
    y_proba = None

    if task == "classification" and hasattr(best, "predict_proba"):
        proba = best.predict_proba(X_test)
        if proba is not None:
            y_proba = proba[:, 1] if proba.shape[1] == 2 else None

    # compute metrics
    if task == "regression":
        metrics = compute_regression_metrics(y_test, y_pred)
        primary = metrics["MAE"]
    else:
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        primary = metrics["F1"]  # primary selection metric

    return {
        "model": model_name,
        "estimator": best,
        "params": best_params,
        "metrics": metrics,
        "primary_metric": primary
    }


# --------------------------------------------------------------------
# AUTO ML MAIN FUNCTION
# --------------------------------------------------------------------
def run_automl(
        df: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        test_size: float = 0.2,
        search_iter: int = 10,
        task_type: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run AutoML on a DataFrame for regression/classification.
    Returns:
        {
            "leaderboard": [...],
            "best_model": {...},
            "best_estimator": <sklearn model object>
        }
    """

    if features is None:
        features = [c for c in df.columns if c != target]

    X = df[features]
    y = df[target]

    if task_type is None:
        task_type = detect_task_type(y)

    # Convert categoricals or object into numeric using one-hot
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    leaderboard = []

    # --------------------------------------------------------
    # BASELINE MODEL
    # --------------------------------------------------------
    if task_type == "regression":
        baseline = DummyRegressor(strategy="mean")
    else:
        baseline = DummyClassifier(strategy="most_frequent")

    baseline_result = train_single_model(
        task=task_type,
        model_name="baseline",
        model_obj=baseline,
        params_grid={},
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        search_iter=search_iter,
    )
    leaderboard.append(baseline_result)

    # --------------------------------------------------------
    # LINEAR / LOGISTIC MODEL
    # --------------------------------------------------------
    if task_type == "regression":
        lin_model = LinearRegression()
        lin_params = {}
    else:
        lin_model = LogisticRegression(max_iter=500)
        lin_params = {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"]
        }

    lin_result = train_single_model(
        task=task_type,
        model_name="linear_model",
        model_obj=lin_model,
        params_grid=lin_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        search_iter=search_iter,
    )
    leaderboard.append(lin_result)

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------
    if task_type == "regression":
        rf = RandomForestRegressor(random_state=42)
        rf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }
    else:
        rf = RandomForestClassifier(random_state=42)
        rf_params = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        }

    rf_result = train_single_model(
        task=task_type,
        model_name="random_forest",
        model_obj=rf,
        params_grid=rf_params,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        search_iter=search_iter,
    )
    leaderboard.append(rf_result)

    # --------------------------------------------------------
    # XGBOOST (Optional)
    # --------------------------------------------------------
    if XGB_AVAILABLE:
        if task_type == "regression":
            xgb = XGBRegressor(random_state=42, tree_method="hist")
            xgb_params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2]
            }
        else:
            xgb = XGBClassifier(random_state=42, tree_method="hist")
            xgb_params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.05, 0.1, 0.2]
            }

        xgb_result = train_single_model(
            task=task_type,
            model_name="xgboost",
            model_obj=xgb,
            params_grid=xgb_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            search_iter=search_iter,
        )
        leaderboard.append(xgb_result)

    # --------------------------------------------------------
    # SELECT BEST MODEL BASED ON PRIMARY METRIC
    # --------------------------------------------------------
    if task_type == "regression":
        best = min(leaderboard, key=lambda x: x["primary_metric"])
    else:
        # For classification → higher F1 is better
        best = max(leaderboard, key=lambda x: x["primary_metric"])

    return {
        "leaderboard": leaderboard,
        "best_model": {
            "model": best["model"],
            "params": best["params"],
            "metrics": best["metrics"],
        },
        "best_estimator": best["estimator"],
        "task_type": task_type,
        "features": list(X.columns),
    }

# --------------------------------------------------------------------
# PHASE-3 WRAPPER REQUIRED BY ROUTERS: train_tabular()
# --------------------------------------------------------------------
def train_tabular(
    df: pd.DataFrame,
    target: str,
    features: Optional[List[str]] = None,
    test_size: float = 0.2,
    search_iter: int = 10,
    task_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Thin wrapper around run_automl().
    Required by the models router and the pytest suite.
    Ensures the output is JSON serializable.
    """

    result = run_automl(
        df=df,
        target=target,
        features=features,
        test_size=test_size,
        search_iter=search_iter,
        task_type=task_type,
    )

    # Convert leaderboard into JSON-safe format (remove estimators)
    leaderboard_serializable = []
    for entry in result["leaderboard"]:
        leaderboard_serializable.append({
            "model": entry["model"],
            "params": entry["params"],
            "metrics": entry["metrics"],
            "primary_metric": float(entry["primary_metric"]),
        })

    # Best model info
    best_info = result["best_model"]

    return {
        "leaderboard": leaderboard_serializable,
        "best_model": best_info,
        "estimator": result["best_estimator"],     # router will serialize separately
        "features": result["features"],
        "task_type": result["task_type"],
    }
