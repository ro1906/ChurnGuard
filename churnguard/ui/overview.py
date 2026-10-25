"""Overview dashboard: churn patterns and risk across the customer base."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from churnguard import config
from churnguard.ui.common import (
    ACCENT,
    DOWN_COLOR,
    RISK_COLORS,
    RISK_ORDER,
    UP_COLOR,
    get_scored_dataset,
    header,
    style_figure,
)

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6", "7-12", "13-24", "25-48", "49+"]


def _rate_bar(df: pd.DataFrame, column: str, title: str, order: list[str] | None = None):
    rates = df.groupby(column, observed=True)[config.TARGET].mean().mul(100).reset_index()
    rates.columns = [column, "churn_rate"]
    fig = px.bar(
        rates,
        x=column,
        y="churn_rate",
        text=rates["churn_rate"].map("{:.0f}%".format),
        category_orders={column: order} if order else None,
    )
    fig.update_traces(marker_color=ACCENT, textposition="outside", cliponaxis=False)
    fig.update_yaxes(ticksuffix="%", title=None, range=[0, rates["churn_rate"].max() * 1.2])
    fig.update_xaxes(title=None)
    return style_figure(fig, title=title)


def render() -> None:
    header(
        "Customer base overview",
        "Where churn happens and how much recurring revenue is currently at risk.",
    )
    df = get_scored_dataset()
    high = df[df["risk_level"] == "High"]

    cols = st.columns(5)
    cols[0].metric("Customers", f"{len(df):,}")
    cols[1].metric("Historical churn rate", f"{df[config.TARGET].mean():.1%}")
    cols[2].metric("Average tenure", f"{df['tenure'].mean():.0f} months")
    cols[3].metric(
        "High-risk customers", f"{len(high):,}", f"{len(high) / len(df):.0%} of base", delta_color="off"
    )
    cols[4].metric("Monthly revenue at high risk", f"{high['MonthlyCharges'].sum():,.0f}")

    st.write("")
    df = df.assign(tenure_group=pd.cut(df["tenure"], TENURE_BINS, labels=TENURE_LABELS))

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_rate_bar(df, "Contract", "Churn rate by contract type"), width="stretch")
        st.plotly_chart(
            _rate_bar(df, "tenure_group", "Churn rate by tenure (months)", TENURE_LABELS),
            width="stretch",
        )
    with right:
        st.plotly_chart(
            _rate_bar(df, "InternetService", "Churn rate by internet service"), width="stretch"
        )
        st.plotly_chart(
            _rate_bar(df, "PaymentMethod", "Churn rate by payment method"), width="stretch"
        )

    left, right = st.columns(2)
    with left:
        counts = df["risk_level"].value_counts().reindex(RISK_ORDER).fillna(0).reset_index()
        counts.columns = ["risk_level", "customers"]
        fig = px.bar(
            counts,
            x="risk_level",
            y="customers",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            text="customers",
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title=None)
        st.plotly_chart(style_figure(fig, title="Customers by predicted risk level"), width="stretch")
    with right:
        hist = df.assign(Status=df[config.TARGET].map({1: "Churned", 0: "Stayed"}))
        fig = px.histogram(
            hist,
            x="MonthlyCharges",
            color="Status",
            barmode="overlay",
            nbins=40,
            opacity=0.7,
            color_discrete_map={"Churned": UP_COLOR, "Stayed": DOWN_COLOR},
        )
        fig.update_xaxes(title="Monthly charges")
        fig.update_yaxes(title=None)
        fig.update_layout(legend={"title": None, "orientation": "h", "y": 1.12})
        st.plotly_chart(style_figure(fig, title="Monthly charges: churned vs. stayed"), width="stretch")

    st.caption(
        "Risk levels come from the deployed model scoring every customer. The model was refit on "
        "all customers after validation, so this view is descriptive; see Model performance for "
        "honest held-out accuracy."
    )
