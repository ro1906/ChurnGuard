"""ChurnGuard: explainable customer churn prediction and retention system.

Run with:  streamlit run app.py
"""

import streamlit as st

from churnguard.ui import batch, overview, performance, single
from churnguard.ui.common import get_predictor, inject_css

st.set_page_config(page_title="ChurnGuard", page_icon=":material/shield:", layout="wide")
inject_css()

pages = [
    st.Page(
        overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True
    ),
    st.Page(
        single.render, title="Predict a customer", icon=":material/person_search:", url_path="predict"
    ),
    st.Page(batch.render, title="Batch scoring", icon=":material/upload_file:", url_path="batch"),
    st.Page(
        performance.render,
        title="Model performance",
        icon=":material/monitoring:",
        url_path="performance",
    ),
]
navigation = st.navigation(pages)

predictor = get_predictor()
st.sidebar.markdown("### ChurnGuard")
st.sidebar.caption(
    f"Model: {predictor.model_name}\n\nDecision threshold: {predictor.threshold:.2f}\n\n"
    "Explainable churn prediction with SHAP."
)
navigation.run()
