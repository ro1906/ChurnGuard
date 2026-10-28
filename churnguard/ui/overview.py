"""Overview: the animated leaky-bucket hero, where the leaks are, and the customer map."""

from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churnguard import config
from churnguard.predict import ChurnPredictor, ValidationError
from churnguard.ui.batch import MAX_ROWS
from churnguard.ui.common import (
    RISK_COLORS,
    RISK_ORDER,
    get_dataset,
    get_predictor,
    get_scored_dataset,
    style_figure,
)
from churnguard.ui.hero import hero_html
from churnguard.ui.theme import CORAL, band, money

TENURE_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_LABELS = ["0-6 months", "7-12", "13-24", "25-48", "49+"]
LENSES = {
    "Contract": ("Contract", "Contract type"),
    "Tenure": ("tenure_group", "Time as a customer"),
    "Internet": ("InternetService", "Internet service"),
    "Payment": ("PaymentMethod", "How they pay"),
}


SOURCE_DEMO = "Demo customers"
SOURCE_OWN = "My own customers"
_YES_NO = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, "1": 1, "0": 0}


def prepare_own_data(raw: pd.DataFrame, predictor: ChurnPredictor) -> tuple[pd.DataFrame, bool]:
    """Score an uploaded customer list for the Overview.

    Returns (scored, has_history). has_history is True when the file has a usable past "Churn"
    column (Yes/No), so the leak charts can show who really left. Without it, the charts use the
    model's predicted risk instead, and the Churn column is filled with that probability.
    Raises ValidationError for a file that is empty, too large, or has bad values.
    """
    if len(raw) > MAX_ROWS:
        raise ValidationError(f"File has {len(raw):,} rows; the limit is {MAX_ROWS:,}.")
    raw = raw.copy()
    history = False
    if config.TARGET in raw.columns:
        mapped = raw[config.TARGET].map(_YES_NO)
        if mapped.notna().all():
            raw[config.TARGET] = mapped.astype(int)
            history = True
        else:
            raw = raw.drop(columns=[config.TARGET])
    scored = predictor.score(raw, with_drivers=False)
    if not history:
        scored[config.TARGET] = scored["churn_probability"]
    return scored, history


def _own_customers() -> tuple[pd.DataFrame, bool] | None:
    """The upload box. Scored data stays in this visitor's session only (never saved)."""
    with st.container(border=True, key="card-upload"):
        st.html(
            "<b>Use your own customers</b><br>Upload a CSV of your active customers and every chart on "
            "this page updates. Your file is scored in memory and not stored."
        )
        uploaded = st.file_uploader("Customer CSV", type=["csv"], key="overview_upload", label_visibility="collapsed")
        sample = get_dataset().drop(columns=[config.TARGET]).head(5)
        st.download_button(
            "Download a sample CSV to see the format",
            sample.to_csv(index=False),
            file_name="sample_customers.csv",
            mime="text/csv",
            icon=":material/download:",
            key="overview_sample",
        )
        if uploaded is None:
            st.session_state.pop("_own_overview", None)
            st.caption("Until you upload a file, the page shows the demo customers.")
            return None
        cached = st.session_state.get("_own_overview")
        if cached and cached[0] == uploaded.file_id:
            return cached[1], cached[2]
        try:
            raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        except Exception:  # noqa: BLE001 - any parser failure becomes a plain message
            st.error("This file could not be read as a CSV. Check the delimiter and encoding.")
            return None
        try:
            scored, history = prepare_own_data(raw, get_predictor())
        except ValidationError as exc:
            st.error("Your file could not be scored. Fix these and upload it again:")
            st.code(str(exc), language=None)
            return None
        st.session_state["_own_overview"] = (uploaded.file_id, scored, history)
        return scored, history


def _choose_data() -> tuple[pd.DataFrame, bool, bool]:
    """Returns (scored customers, has_history, is_own_file)."""
    source = st.segmented_control(
        "Customer data", [SOURCE_DEMO, SOURCE_OWN], default=SOURCE_DEMO, key="overview_source",
        label_visibility="collapsed",
    ) or SOURCE_DEMO
    if source == SOURCE_OWN:
        own = _own_customers()
        if own is not None:
            return own[0], own[1], True
    return get_scored_dataset(), True, False


def _hero_stats(df: pd.DataFrame) -> dict:
    high = df[df["risk_level"] == "High"]
    return {
        "high": int(len(high)),
        "medium": int((df["risk_level"] == "Medium").sum()),
        "low": int((df["risk_level"] == "Low").sum()),
        "high_revenue": float(high["MonthlyCharges"].sum()),
        "avg_high_charge": float(high["MonthlyCharges"].mean()) if len(high) else 0.0,
    }


def _lens_rates(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = df.groupby(column, observed=True)
    rates = grouped[config.TARGET].agg(["mean", "size"]).reset_index()
    rates.columns = [column, "rate", "customers"]
    rates["rate"] *= 100
    return rates


def _lens_chart(rates: pd.DataFrame, column: str, ordered: bool, history: bool = True) -> go.Figure:
    data = rates if ordered else rates.sort_values("rate")
    top = data["rate"].max()
    fig = go.Figure(
        go.Bar(
            x=data["rate"],
            y=data[column].astype(str),
            orientation="h",
            marker_color=[CORAL if r == top else "#3F8C9C" for r in data["rate"]],
            text=[f"{r:.0f}%" for r in data["rate"]],
            textposition="outside",
            cliponaxis=False,
            customdata=data["customers"],
            hovertemplate="%{y}<br>%{x:.1f}% " + ("left" if history else "expected to leave")
            + "<br>%{customdata:,} customers<extra></extra>",
        )
    )
    fig.update_xaxes(visible=False, range=[0, top * 1.18])
    fig.update_yaxes(title=None, autorange="reversed" if ordered else True)
    return style_figure(fig, height=330, dark=True)


def _leaks_band(df: pd.DataFrame, history: bool = True) -> None:
    left, right = st.columns([1, 1.35], gap="large", vertical_alignment="center")
    with left:
        when = "have historically left the most" if history else "the model expects to leave the most"
        st.html(
            f'<h2>Where the leaks are</h2><p class="cg-lede">Pick a lens to see which groups of customers {when}.</p>'
        )
        lens = (
            st.segmented_control(
                "Look at churn by", list(LENSES), default="Contract", key="lens", label_visibility="collapsed"
            )
            or "Contract"
        )
        column, label = LENSES[lens]
        rates = _lens_rates(df, column)
        hi, lo = rates.loc[rates["rate"].idxmax()], rates.loc[rates["rate"].idxmin()]
        leave = "leave" if history else "are expected to leave"
        if lo["rate"] > 0:
            ratio = hi["rate"] / lo["rate"]
            big = f"{ratio:.0f}x"
            line = (
                f'<b style="color:#fff">{hi[column]}</b> customers {leave} {ratio:.0f} times as often as '
                f'<b style="color:#fff">{lo[column]}</b> customers ({hi["rate"]:.0f}% against {lo["rate"]:.0f}%).'
            )
        else:  # the safest group has none, so a ratio would be infinite
            big = f"{hi['rate']:.0f}%"
            line = (
                f'<b style="color:#fff">{hi[column]}</b> customers {leave}; '
                f'<b style="color:#fff">{lo[column]}</b> customers hardly ever do.'
            )
        st.html(
            f'<div style="margin-top:.6rem"><span style="font-family:Bricolage Grotesque;font-weight:800;'
            f'font-size:4.2rem;letter-spacing:-.04em;line-height:1;color:{CORAL}">{big}</span>'
            f'<p class="cg-lede" style="margin-top:.4rem">{line}</p></div>'
        )
    with right:
        st.plotly_chart(
            _lens_chart(rates, column, ordered=column == "tenure_group", history=history),
            width="stretch",
            config={"displayModeBar": False},
        )
        if history:
            st.caption(f"Share of customers who left, by {label.lower()}. Historical data, all {len(df):,} customers.")
        else:
            st.caption(f"Average predicted churn risk, by {label.lower()}, across your {len(df):,} customers.")


def _map_figure(df: pd.DataFrame, levels: list[str]) -> go.Figure:
    view = df[df["risk_level"].isin(levels)]
    if len(view) > 2500:
        view = view.sample(2500, random_state=3)
    fig = px.scatter(
        view,
        x="tenure",
        y="MonthlyCharges",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_level": RISK_ORDER},
        custom_data=["Contract", "InternetService", "churn_probability"],
        opacity=0.78,
    )
    fig.update_traces(
        marker={"size": 7, "line": {"width": 0}},
        hovertemplate="<b>%{customdata[0]}</b>, %{customdata[1]}<br>%{x} months, "
        + config.CURRENCY
        + "%{y:.0f}/month<br>Churn risk %{customdata[2]:.0%}<extra></extra>",
    )
    fig.update_xaxes(title="Months as a customer")
    fig.update_yaxes(title=f"Monthly charges ({config.CURRENCY})")
    fig.update_layout(legend={"title": None, "orientation": "h", "y": 1.08, "x": 0})
    return style_figure(fig, height=430)


def _map_band(df: pd.DataFrame, history: bool = True, own: bool = False) -> None:
    high = df[df["risk_level"] == "High"]
    st.html(
        '<h2>Your customer base, one dot per customer</h2><p class="cg-lede">Every customer is '
        "scored by the model. Filter the colours to focus on who needs attention.</p>"
    )
    cols = st.columns(5)
    cols[0].metric("Customers", f"{len(df):,}")
    cols[1].metric("Historical churn rate" if history else "Average churn risk", f"{df[config.TARGET].mean():.1%}")
    cols[2].metric("Average tenure", f"{df['tenure'].mean():.0f} months")
    cols[3].metric("High-risk customers", f"{len(high):,}", f"{len(high) / len(df):.0%} of base", delta_color="off")
    cols[4].metric("Monthly revenue at high risk", money(high["MonthlyCharges"].sum()))

    levels = st.pills("Show", RISK_ORDER, selection_mode="multi", default=RISK_ORDER, key="map_levels")
    st.plotly_chart(_map_figure(df, list(levels) or RISK_ORDER), width="stretch", config={"displayModeBar": False})

    rest = df[df["risk_level"] != "High"]
    if high.empty or rest.empty:
        st.html(
            '<p class="cg-lede">Every customer in this list falls in the same risk group, '
            "so there is nothing to compare.</p>"
        )
    else:
        st.html(
            f'<p class="cg-lede">The typical high-risk customer has been with you <b>{high["tenure"].median():.0f} '
            f"months</b> and pays <b>{money(high['MonthlyCharges'].median())}</b> a month. Everyone else: "
            f"<b>{rest['tenure'].median():.0f} months</b> and <b>{money(rest['MonthlyCharges'].median())}</b>.</p>"
        )
    if own:
        st.caption(
            "Risk levels come from the model scoring your file. The model was trained on the IBM sample data, "
            "so treat the numbers as a first look until it is retrained on your own history."
        )
    else:
        st.caption(
            "Risk levels come from the deployed model scoring every customer. It was refit on all customers "
            "after validation, so this view is descriptive. See Model performance for held-out accuracy."
        )


CTAS = [
    ("predict", "Check one customer", "Enter a profile, see the risk, the reasons, and test a retention move."),
    ("batch", "Score a whole list", "Upload a CSV and get a prioritised call list in seconds."),
    ("performance", "See how good it is", "Accuracy on customers the model never saw, in plain language."),
]


def _go_band() -> None:
    st.html('<h2>Try it on a real customer</h2><p class="cg-lede">Three ways to put the model to work.</p>')
    pages = st.session_state.get("_pages", {})
    for col, (key, title, text) in zip(st.columns(3, gap="medium"), CTAS, strict=True):
        with col, st.container(border=True, key=f"card-go-{key}"):
            st.html(f'<div class="cg-cta"><h3>{title}</h3><p>{text}</p></div>')
            if key in pages:
                st.page_link(pages[key], label="Open", icon=":material/arrow_forward:")


def render() -> None:
    with band("hero", "paper", first=True):
        df, history, own = _choose_data()
        df = df.assign(tenure_group=pd.cut(df["tenure"], TENURE_BINS, labels=TENURE_LABELS))
        st.iframe(hero_html(_hero_stats(df), config.CURRENCY))
    with band("leaks", "tide"):
        _leaks_band(df, history)
    with band("map", "paper"):
        _map_band(df, history, own)
    with band("go", "water"):
        _go_band()
