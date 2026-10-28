"""Score a list: upload a CSV, get risk levels and reasons for every customer."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churnguard import config
from churnguard.predict import ValidationError
from churnguard.ui.common import (
    RISK_COLORS,
    RISK_ORDER,
    get_dataset,
    get_predictor,
    style_figure,
)
from churnguard.ui.theme import CORAL, band, money, page_head

MAX_ROWS = 50_000
SOURCE_UPLOAD = "Upload a CSV"
SOURCE_SAMPLE = "Use 500 sample customers"


def _load_input() -> pd.DataFrame | None:
    source = st.radio("Customer data", [SOURCE_UPLOAD, SOURCE_SAMPLE], horizontal=True)
    if source == SOURCE_SAMPLE:
        return get_dataset().drop(columns=[config.TARGET]).sample(500, random_state=7)

    sample = get_dataset().drop(columns=[config.TARGET]).head(5)
    uploaded = st.file_uploader("Drop your customer CSV here", type=["csv"])
    st.download_button(
        "Download a sample CSV to see the format",
        sample.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    if uploaded is None:
        st.caption(
            "Required columns: " + ", ".join(config.REQUIRED_COLS) + ". TotalCharges and customerID are optional."
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


def _mix_donut(scored: pd.DataFrame) -> go.Figure:
    counts = scored["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0)
    fig = go.Figure(
        go.Pie(
            labels=list(counts.index),
            values=list(counts.values),
            hole=0.64,
            sort=False,
            marker={"colors": [RISK_COLORS[level] for level in counts.index], "line": {"color": "#FBFDFC", "width": 3}},
            textinfo="percent",
            textfont={"size": 15, "color": "#0C2E3B"},
            hovertemplate="%{label}: %{value:,} customers<extra></extra>",
        )
    )
    fig.add_annotation(text=f"<b>{len(scored):,}</b><br>customers", showarrow=False, font={"size": 18})
    fig.update_layout(showlegend=True, legend={"orientation": "h", "y": -0.05, "x": 0.5, "xanchor": "center"})
    return style_figure(fig, height=320, title="How the list splits by risk")


def _driver_bars(scored: pd.DataFrame) -> go.Figure | None:
    high = scored[scored["risk_level"] == "High"]
    names = high["top_risk_driver"].str.split(":").str[0]
    drivers = names[names != "None"].value_counts().head(6)
    if drivers.empty:
        return None
    drivers = drivers.iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=drivers.values,
            y=drivers.index,
            orientation="h",
            marker_color=CORAL,
            text=drivers.values,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x} high-risk customers<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(title=None)
    return style_figure(fig, height=320, title="What pushes high-risk customers out")


def _results(scored: pd.DataFrame) -> None:
    high = scored[scored["risk_level"] == "High"]
    cols = st.columns(4)
    cols[0].metric("Customers scored", f"{len(scored):,}")
    cols[1].metric("High risk", f"{len(high):,}", f"{len(high) / len(scored):.0%}", delta_color="off")
    cols[2].metric("Predicted to churn", f"{int(scored['predicted_churn'].sum()):,}")
    cols[3].metric("Monthly revenue at high risk", money(high["MonthlyCharges"].sum()))

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="card-mix"):
        st.plotly_chart(_mix_donut(scored), width="stretch", config={"displayModeBar": False})
    with right, st.container(border=True, key="card-reasons"):
        bars = _driver_bars(scored)
        if bars is None:
            st.info("No high-risk customers in this list.")
        else:
            st.plotly_chart(bars, width="stretch", config={"displayModeBar": False})

    st.subheader("Your call list")
    levels = st.pills(
        "Show risk levels", RISK_ORDER, selection_mode="multi", default=["High", "Medium"], key="batch_levels"
    )
    view = scored[scored["risk_level"].isin(levels or [])].sort_values("churn_probability", ascending=False)

    front = [c for c in (config.ID_COL, "risk_level", "churn_probability", "top_risk_driver") if c in view]
    rest = [c for c in view.columns if c not in front]
    ordered = view[front + rest]  # the table and the download share this column order
    st.dataframe(
        ordered,
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
        ordered.to_csv(index=False),
        file_name="churn_scores.csv",
        mime="text/csv",
        disabled=view.empty,
        icon=":material/download:",
    )


def render() -> None:
    predictor = get_predictor()
    with band("head", "tide", first=True):
        page_head(
            "Score a whole list",
            "Upload your customers and get a prioritised call list: who to contact first, and why.",
        )
    with band("work", "paper"):
        df = _load_input()
        if df is None:
            return
        try:
            scored = predictor.score(df)
        except ValidationError as exc:
            st.error("The data could not be scored:")
            st.code(str(exc), language=None)
            return
        _results(scored)
