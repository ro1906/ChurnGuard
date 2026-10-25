"""Batch scoring page: upload a CSV, get risk levels and reasons for every customer."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from churnguard import config
from churnguard.predict import ValidationError
from churnguard.ui.common import (
    RISK_COLORS,
    RISK_ORDER,
    get_dataset,
    get_predictor,
    header,
    style_figure,
)

MAX_ROWS = 50_000
SOURCE_UPLOAD = "Upload a CSV"
SOURCE_SAMPLE = "Use 500 sample customers"


def _load_input() -> pd.DataFrame | None:
    source = st.radio("Customer data", [SOURCE_UPLOAD, SOURCE_SAMPLE], horizontal=True)
    if source == SOURCE_SAMPLE:
        return get_dataset().drop(columns=[config.TARGET]).sample(500, random_state=7)

    sample = get_dataset().drop(columns=[config.TARGET]).head(5)
    st.download_button(
        "Download a sample CSV to see the expected format",
        sample.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
    if uploaded is None:
        st.info(
            "Required columns: " + ", ".join(config.REQUIRED_COLS) + ". "
            "TotalCharges and customerID are optional."
        )
        return None
    try:
        df = pd.read_csv(uploaded)
    except Exception:  # noqa: BLE001 - any parser failure is a user-facing "bad file" message
        st.error("This file could not be read as a CSV. Check the delimiter and encoding.")
        return None
    if len(df) > MAX_ROWS:
        st.error(f"File has {len(df):,} rows; the limit is {MAX_ROWS:,}.")
        return None
    return df


def render() -> None:
    header(
        "Batch scoring",
        "Score a whole customer list and download a prioritised retention call list.",
    )
    predictor = get_predictor()
    df = _load_input()
    if df is None:
        return

    try:
        scored = predictor.score(df)
    except ValidationError as exc:
        st.error("The data could not be scored:")
        st.code(str(exc), language=None)
        return

    high = scored[scored["risk_level"] == "High"]
    cols = st.columns(4)
    cols[0].metric("Customers scored", f"{len(scored):,}")
    cols[1].metric("High risk", f"{len(high):,}", f"{len(high) / len(scored):.0%}", delta_color="off")
    cols[2].metric("Predicted to churn", f"{int(scored['predicted_churn'].sum()):,}")
    cols[3].metric("Monthly revenue at high risk", f"{high['MonthlyCharges'].sum():,.0f}")

    counts = scored["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
    counts.columns = ["risk_level", "customers"]
    fig = px.bar(
        counts, x="customers", y="risk_level", orientation="h", color="risk_level",
        color_discrete_map=RISK_COLORS, text="customers",
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=RISK_ORDER[::-1])
    st.plotly_chart(style_figure(fig, height=200, title="Customers by risk level"), width="stretch")

    levels = st.multiselect("Show risk levels", RISK_ORDER, default=["High", "Medium"])
    view = scored[scored["risk_level"].isin(levels)].sort_values("churn_probability", ascending=False)

    front = [c for c in (config.ID_COL, "risk_level", "churn_probability", "top_risk_driver") if c in view]
    rest = [c for c in view.columns if c not in front]
    st.dataframe(
        view[front + rest],
        hide_index=True,
        width="stretch",
        column_config={
            "churn_probability": st.column_config.ProgressColumn(
                "Churn probability", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "risk_level": "Risk level",
            "top_risk_driver": "Top risk driver",
        },
    )
    st.download_button(
        f"Download {len(view):,} rows as CSV",
        view.to_csv(index=False),
        file_name="churn_scores.csv",
        mime="text/csv",
        disabled=view.empty,
    )
