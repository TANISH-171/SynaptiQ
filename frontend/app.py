import streamlit as st
from utils.api_client import health

st.set_page_config(page_title="SynaptiQ 2.0", layout="wide")
st.title("SynaptiQ 2.0 — Neuro-Symbolic Conversational Analytics")

h = health()
st.success(f"API Health: {h.get('status','unknown')}")
st.caption("Use the left sidebar to navigate pages.")
