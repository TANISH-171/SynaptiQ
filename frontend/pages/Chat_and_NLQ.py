import streamlit as st
from utils.api_client import route_nlq

st.header("💬 Chat & NLQ")
query = st.text_area("Ask a question in plain English (e.g., 'Forecast next 12 months for sales')")

if st.button("Route Query"):
    plan = route_nlq(query)
    st.subheader("TaskPlan")
    st.json(plan)
    st.caption("In Phase 2+, these steps will trigger AutoML/Forecasting/Explainability pipelines.")
