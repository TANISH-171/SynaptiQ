# frontend/pages/1_📂_Datasets.py
import os
import json
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

from utils.api_client import APIClient

st.set_page_config(page_title="Datasets", layout="wide")

st.title("Datasets")

api = APIClient()

# -- Dataset selection / upload area (assuming upload endpoint exists from Phase 1)
st.subheader("Select a dataset")
datasets_resp = api.list_datasets()
datasets_map = datasets_resp.get("datasets", {})

if not datasets_map:
    st.info("No datasets found. Upload a dataset on this page (or via the upload widget if available).")
else:
    # Build options for selectbox
    options = []
    for ds_id, ds in datasets_map.items():
        label = f"{ds.get('name', ds_id)} ({ds_id})"
        options.append((label, ds_id))
    label_to_id = {lbl: did for (lbl, did) in options}
    choice = st.selectbox("Choose a dataset", [lbl for (lbl, _) in options])
    dataset_id = label_to_id[choice]

    # Controls
    n_preview = st.number_input("Rows to preview", min_value=1, max_value=1000, value=10, step=1)

    # Attempt preview
    with st.spinner("Loading preview..."):
        try:
            preview = api.get_preview(dataset_id, n=n_preview)
        except Exception as e:
            st.error(f"Failed to load preview: {e}")
            st.stop()

    rows = preview.get("rows", 0)
    cols = preview.get("cols", 0)
    dtypes_map = preview.get("dtypes", {})
    data_records = preview.get("data", [])
    columns = preview.get("columns", [])

    # Display metrics cards
    # Try to fetch profile (for Missing % and numeric/categorical counts)
    with st.spinner("Computing profile (first run may take a moment)..."):
        profile = None
        try:
            profile = api.get_profile(dataset_id)
        except Exception as e:
            st.warning(f"Profile not available yet: {e}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{rows:,}")
    c2.metric("Columns", f"{cols:,}")
    if profile:
        missing_overall = profile.get("missing_overall_pct", 0.0)
        num_count = len(profile.get("numeric_cols", []))
        cat_count = len(profile.get("categorical_cols", []))
    else:
        # Fallback estimation if profile failed
        missing_overall = 0.0
        # Heuristic from dtypes_map
        numeric_cols = [c for c, d in dtypes_map.items() if any(x in d for x in ["int", "float", "Int", "Float"])]
        num_count = len(numeric_cols)
        cat_count = cols - num_count

    c3.metric("Missing % (overall)", f"{missing_overall:.2f}%")
    c4.metric("# Numeric", f"{num_count}")
    c5.metric("# Categorical", f"{cat_count}")

    # Data preview table
    if data_records and columns:
        df_preview = pd.DataFrame.from_records(data_records, columns=columns)
        st.subheader("Preview")
        st.dataframe(df_preview, use_container_width=True, height=350)
    else:
        st.info("No preview data available.")

    # Dtypes / missing / unique table
    st.subheader("Schema & Nulls")
    if profile and profile.get("columns"):
        col_rows = []
        for cmeta in profile["columns"]:
            col_rows.append({
                "column": cmeta["name"],
                "dtype": cmeta["dtype"],
                "missing": cmeta["missing"],
                "missing_pct": round(cmeta["missing_pct"], 3),
                "unique": cmeta["unique"]
            })
        st.dataframe(pd.DataFrame(col_rows), use_container_width=True, height=300)
    else:
        # Fallback using dtypes only
        st.dataframe(pd.DataFrame([{"column": c, "dtype": d} for c, d in dtypes_map.items()]))

    # Top-5 values per column
    st.subheader("Top values")
    if profile and "top_values" in profile:
        with st.expander("Show top 5 values per column", expanded=False):
            tv = profile["top_values"]
            for col in columns:
                if col in tv and tv[col]:
                    tv_df = pd.DataFrame(tv[col])
                    st.markdown(f"**{col}**")
                    st.dataframe(tv_df, use_container_width=True, height=160)
                else:
                    st.markdown(f"**{col}** — no data / all unique")

    # Correlation heatmap
    st.subheader("Correlation (Pearson)")
    corr = None
    if profile and profile.get("corr", {}).get("columns"):
        corr_cols = profile["corr"]["columns"]
        corr_mat = profile["corr"]["matrix"]
        if corr_cols and corr_mat and len(corr_cols) >= 2:
            z = corr_mat
            heatmap = go.Heatmap(
                z=z,
                x=corr_cols,
                y=corr_cols,
                zmin=-1,
                zmax=1,
                colorbar=dict(title="r")
            )
            fig = go.Figure(data=[heatmap])
            fig.update_layout(height=500, width=None, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough numeric columns to compute correlation.")
    else:
        st.info("Correlation not available for this dataset (or only one numeric column).")
