"""Shared UI helpers: cached loaders, colours, styling."""

from __future__ import annotations

import json
import logging

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from churnguard import config
from churnguard.data import load_dataset
from churnguard.predict import ChurnPredictor, load_predictor

logger = logging.getLogger(__name__)

RISK_COLORS = {"Low": "#2a9d8f", "Medium": "#e9a23b", "High": "#d1495b"}
RISK_ORDER = config.RISK_LEVELS
UP_COLOR = "#d1495b"  # raises churn risk
DOWN_COLOR = "#3a7ca5"  # lowers churn risk
ACCENT = "#3a7ca5"


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; max-width: 1250px;}
        [data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border-radius: 10px;
            padding: 12px 16px;
        }
        .risk-badge {
            display: inline-block; padding: 4px 14px; border-radius: 999px;
            color: white; font-weight: 600; font-size: 0.95rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def risk_badge(level: str) -> str:
    """HTML pill for a risk level. Only ever fed our own fixed level names."""
    color = RISK_COLORS.get(level, "#888")
    return f'<span class="risk-badge" style="background:{color}">{level} risk</span>'


def style_figure(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 50 if title else 20, "b": 10},
        title={"text": title, "font": {"size": 15}} if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> ChurnPredictor:
    """Load the saved model; train one automatically if it is missing or incompatible.

    A model pickled under different library versions can fail to load on a fresh deploy,
    so retraining (about 30 seconds) is the safe fallback instead of crashing.
    """
    try:
        return load_predictor()
    except Exception:  # noqa: BLE001 - missing file or version-incompatible pickle
        logger.warning("Saved model unavailable; training a new one", exc_info=True)
        from churnguard.train import train

        train()
        return load_predictor()


@st.cache_data(show_spinner="Loading customer data...")
def get_dataset() -> pd.DataFrame:
    return load_dataset()


@st.cache_data(show_spinner="Scoring customers...")
def get_scored_dataset() -> pd.DataFrame:
    """The whole customer base scored by the model, with the historical Churn label attached."""
    data = get_dataset()
    scored = get_predictor().score(data.drop(columns=[config.TARGET]), with_drivers=False)
    scored[config.TARGET] = data[config.TARGET].to_numpy()
    return scored


@st.cache_data
def get_metrics() -> dict:
    get_predictor()  # makes sure a model (and metrics file) exists
    with open(config.METRICS_PATH, encoding="utf-8") as fh:
        return json.load(fh)
