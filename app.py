"""ChurnGuard: explainable customer churn prediction and retention system.

Run with:  streamlit run app.py
"""

import streamlit as st

from churnguard.ui import batch, overview, performance, single
from churnguard.ui.common import get_predictor, inject_css
from churnguard.ui.theme import band

st.set_page_config(
    page_title="ChurnGuard",
    page_icon=":material/water_drop:",
    layout="wide",
    initial_sidebar_state="collapsed",
)
inject_css()
st.logo("static/logo.svg", size="large")

pages = {
    "overview": st.Page(
        overview.render, title="Overview", icon=":material/water_drop:", url_path="overview", default=True
    ),
    "predict": st.Page(single.render, title="Check a customer", icon=":material/person_search:", url_path="predict"),
    "batch": st.Page(batch.render, title="Score a list", icon=":material/upload_file:", url_path="batch"),
    "performance": st.Page(
        performance.render, title="Model quality", icon=":material/monitoring:", url_path="performance"
    ),
}
st.session_state["_pages"] = pages  # lets pages build links to each other
navigation = st.navigation(list(pages.values()), position="top")
navigation.run()

predictor = get_predictor()
with band("footer", "tide"):
    st.html(
        f'<div class="cg-footer"><b>ChurnGuard</b><span>Model: {predictor.model_name}</span>'
        f"<span>Decision threshold: {predictor.threshold:.2f}</span>"
        "<span>Built on the IBM Telco sample dataset. SHAP explains every prediction.</span></div>"
    )
