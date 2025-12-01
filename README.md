# 🧠 SynaptiQ 2.0 — Neuro-Symbolic Conversational Analytics Platform  
### Phase 3: AutoML + Forecasting + Model Registry

SynaptiQ 2.0 is an intelligent neuro-symbolic decision intelligence system designed to:
- Ingest datasets
- Profile, preview, and explore data
- Answer natural-language questions
- Train classical ML models automatically
- Run time-series forecasting
- Store & manage trained models
- Provide predictions through a conversational interface

Phase 3 introduces the **complete ML engine**:  
**AutoML + Time-Series Forecasting + Model Registry + Streamlit UI**.

---

# 🚀 Features Implemented in Phase 3

## 1️⃣ AutoML Engine (Regression & Classification)
- Linear Regression / Logistic Regression  
- RandomForest  
- Optional XGBoost (if installed)  
- Mean/Most-Frequent baseline  
- RandomizedSearchCV hyperparameter tuning  
- Automatic feature selection  
- Train/Test split  
- Leaderboard with primary metric selection  
- Artifact saving via `joblib`

## 2️⃣ Forecasting Engine
- Naive Forecast  
- Seasonal Naive  
- ARIMA (statsmodels)  
- Optional Prophet (auto-skipped if unavailable)  
- Rolling-window evaluation  
- MAE, RMSE, MAPE scoring  
- Smart model selection  
- JSON-based artifact storage  

## 3️⃣ Model Registry
Stores:
- model_id  
- dataset_id  
- target  
- task type  
- feature list  
- parameters  
- metrics  
- artifact path  
- created_at  

Backed by an extended `registry.json`.

## 4️⃣ Prediction APIs
Supports:
- Batch tabular prediction  
- Single-row prediction  
- Forecast horizon prediction  

## 5️⃣ Streamlit Model UI
- Dataset & target column selection  
- Task configuration  
- Training dashboard  
- Leaderboard visualization  
- Forecast plot (Plotly)  
- Prediction form  
- CSV upload prediction  
- Full model management

---

# 🏗 Backend Architecture

