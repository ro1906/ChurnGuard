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
from churnguard.ui import theme
from churnguard.ui.theme import inject_css, risk_badge  # noqa: F401 - re-exported for the pages

logger = logging.getLogger(__name__)

RISK_COLORS = {"Low": theme.KELP, "Medium": theme.AMBER, "High": theme.CORAL}
RISK_ORDER = config.RISK_LEVELS
UP_COLOR = theme.CORAL  # raises churn risk
DOWN_COLOR = theme.WATER  # lowers churn risk
ACCENT = theme.WATER

FONT = "Figtree, system-ui, sans-serif"


def style_figure(fig: go.Figure, height: int = 320, title: str | None = None, dark: bool = False) -> go.Figure:
    """Apply the ChurnGuard look to a Plotly figure (light paper or dark tide background)."""
    text = "#DCEBE8" if dark else theme.INK
    grid = "rgba(255,255,255,0.10)" if dark else "rgba(12,46,59,0.08)"
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        height=height,
        margin={"l": 10, "r": 10, "t": 54 if title else 20, "b": 10},
        title={"text": title, "font": {"size": 17, "family": "Bricolage Grotesque, " + FONT}} if title else None,
        font={"family": FONT, "color": text},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel={"font": {"family": FONT}, "bordercolor": "rgba(0,0,0,0)"},
    )
    tick = {"size": 14, "color": text}
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=tick)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid, linecolor=grid, tickfont=tick)
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
