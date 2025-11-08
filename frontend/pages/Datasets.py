import streamlit as st
from utils.api_client import upload_dataset, list_datasets

st.header("📂 Datasets")

file = st.file_uploader(
    "Drag & drop or browse CSV, Excel, JSON, or Parquet", 
    type=["csv","xlsx","xls","json","parquet"]
)
if file and st.button("Upload"):
    with st.spinner("Uploading..."):
        meta = upload_dataset(file)
    st.success(f"Uploaded: {meta['name']} (id={meta['dataset_id']})")

st.subheader("Registry")
ds = list_datasets()
st.json(ds)
