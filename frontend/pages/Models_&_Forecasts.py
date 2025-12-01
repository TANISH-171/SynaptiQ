import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from utils.api_client import APIClient

st.set_page_config(page_title="Models & Forecasts", page_icon="📊")
st.header("📊 Models & Forecasts")

api = APIClient()


# ------------------------------------------------------------
# Helper UI widgets
# ------------------------------------------------------------
def load_dataset_options():
    """Safely load dataset options from backend."""
    resp = api.list_datasets()

    # Expected: {"ok": True, "datasets": {...}}
    if isinstance(resp, dict) and "datasets" in resp:
        datasets = resp["datasets"]
    else:
        return {}

    if not isinstance(datasets, dict):
        return {}

    # Build: {dataset_name: dataset_id}
    return {info.get("name", ds_id): ds_id for ds_id, info in datasets.items()}



def load_profile(dataset_id):
    try:
        return api.get_profile(dataset_id)
    except Exception:
        return None


# ------------------------------------------------------------
# Sidebar for dataset + task selection
# ------------------------------------------------------------
st.sidebar.subheader("Dataset & Task")

dataset_map = load_dataset_options()
dataset_name = st.sidebar.selectbox("Choose Dataset", list(dataset_map.keys()) if dataset_map else [])

if dataset_name:
    dataset_id = dataset_map[dataset_name]
    profile = load_profile(dataset_id)

    # Suggest target column from profile
    columns = profile["columns"] if profile and "columns" in profile else []

    target_col = st.sidebar.selectbox("Select Target Column", columns)
    task_type = st.sidebar.selectbox(
        "Task Type",
        ["regression", "classification", "forecast"]
    )


# ------------------------------------------------------------
# TABULAR TRAINING CONFIG
# ------------------------------------------------------------
if dataset_name and task_type in ("regression", "classification"):
    st.subheader("🔧 Tabular Model Training")

    # feature auto-detection (all except target)
    feature_cols = [c for c in columns if c != target_col]
    chosen_features = st.multiselect("Features", feature_cols, default=feature_cols)

    test_size = st.slider("Test Size (validation split)", 0.1, 0.4, 0.2, 0.05)
    search_iter = st.number_input("Search Iterations", min_value=1, max_value=25, value=10)


# ------------------------------------------------------------
# FORECAST TRAINING CONFIG
# ------------------------------------------------------------
if dataset_name and task_type == "forecast":
    st.subheader("🔧 Forecast Configuration")

    date_col = st.selectbox("Date Column", columns)
    horizon = st.number_input("Forecast Horizon", min_value=1, max_value=200, value=12)

    freq = st.text_input("Frequency (optional: 'D','M','H')", value="D")

    seasonal_period = st.number_input(
        "Seasonal Period (e.g., 7 for weekly seasonality)",
        min_value=1,
        max_value=400,
        value=7
    )


# ------------------------------------------------------------
# TRAIN MODEL BUTTON
# ------------------------------------------------------------
if dataset_name and st.button("🚀 Train Model"):
    with st.spinner("Training model... this may take a moment 🚀"):
        try:
            if task_type in ("regression", "classification"):
                payload = {
                    "task_type": task_type,
                    "tabular": {
                        "dataset_id": dataset_id,
                        "target": target_col,
                        "features": chosen_features,
                        "test_size": test_size,
                        "search_iter": int(search_iter)
                    }
                }
            else:  # forecast
                payload = {
                    "task_type": "forecast",
                    "forecast": {
                        "dataset_id": dataset_id,
                        "date_col": date_col,
                        "target": target_col,
                        "horizon": int(horizon),
                        "freq": freq or None,
                        "seasonal_period": int(seasonal_period)
                    }
                }

            result = api.train_model(payload)
            st.success("Training Completed Successfully!")

            st.session_state["last_model_id"] = result["model_id"]
            st.session_state["last_leaderboard"] = result["leaderboard"]
            st.session_state["last_best"] = result["best"]

        except Exception as e:
            st.error(f"Training failed: {e}")


# ------------------------------------------------------------
# RESULTS SECTION
# ------------------------------------------------------------
if "last_model_id" in st.session_state:
    st.subheader(f"🏆 Best Model (ID: {st.session_state['last_model_id']})")
    st.json(st.session_state["last_best"])

    st.subheader("🏁 Leaderboard")
    st.table(pd.DataFrame(st.session_state["last_leaderboard"]))


# ------------------------------------------------------------
# PREDICTION SECTION
# ------------------------------------------------------------
st.subheader("🔮 Make Predictions")

model_list = api.list_models().get("models", {})
model_names = [f"{mid} ({meta['task_type']})" for mid, meta in model_list.items()]

if model_names:
    chosen_model = st.selectbox("Choose Model", model_names)
    model_id = chosen_model.split(" ")[0]
    meta = model_list[model_id]
    mtype = meta["task_type"]

    st.info(f"Selected model type: **{mtype}**")

    # ----------- TABULAR PREDICTION ------------
    if mtype in ("regression", "classification"):
        st.subheader("Enter Record(s)")

        # Infer expected features
        feats = meta.get("features", [])
        inputs = {}
        for col in feats:
            inputs[col] = st.text_input(f"{col}", "")

        if st.button("Predict"):
            try:
                payload = {"records": [inputs]}
                result = api.predict_model(model_id, payload)
                st.success("Prediction:")
                st.json(result)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        st.subheader("OR Upload CSV")
        uploaded = st.file_uploader("CSV File", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if st.button("Predict from CSV"):
                try:
                    payload = {"records": df.to_dict(orient="records")}
                    result = api.predict_model(model_id, payload)
                    preds = result.get("predictions", [])
                    out_df = df.copy()
                    out_df["prediction"] = preds
                    st.dataframe(out_df)
                except Exception as e:
                    st.error(f"CSV Prediction failed: {e}")

    # ----------- FORECAST PREDICTION ------------
    elif mtype == "forecast":
        st.subheader("Forecast Horizon")
        h = st.number_input("Steps Ahead", min_value=1, max_value=200, value=12)

        if st.button("Forecast"):
            try:
                payload = {"horizon": int(h)}
                result = api.predict_model(model_id, payload)

                hist_dates = result["history_dates"]
                hist_vals = result["history_values"]
                forecast_vals = result["forecast"]

                # Plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist_dates, y=hist_vals, mode='lines', name='History'))
                future_x = list(range(len(hist_dates), len(hist_dates) + len(forecast_vals)))
                fig.add_trace(go.Scatter(x=future_x, y=forecast_vals, mode='lines', name='Forecast'))

                st.plotly_chart(fig, use_container_width=True)

                st.success("Forecast Complete")
                st.json(result)

            except Exception as e:
                st.error(f"Forecasting failed: {e}")
else:
    st.info("⚠ No models found. Train a model first!")
