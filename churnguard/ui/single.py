"""Check a customer: risk gauge, the reasons behind it, and retention moves you can test."""

from __future__ import annotations

import base64
import html

import pandas as pd
import streamlit as st

from churnguard import config
from churnguard.predict import ValidationError
from churnguard.recommend import recommend
from churnguard.ui.common import get_predictor
from churnguard.ui.theme import band, money, page_head, risk_badge, risk_color

PRESETS = {
    "High-risk example": {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
    },
    "Stable customer": {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 24,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 55.0,
    },
    "Loyal low-risk example": {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 65.0,
    },
}
YES_NO = ["No", "Yes"]
ARC_LENGTH = 377.0  # length of the gauge's half circle (pi * 120)
ARC = "M30 150 A120 120 0 0 1 270 150"


def _apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
        st.session_state[f"in_{key}"] = value


def _apply_changes(changes: dict) -> None:
    """Callback for the "Apply" buttons: write a retention move into the profile form."""
    for key, value in changes.items():
        st.session_state[f"in_{key}"] = value


def _select(label: str, column: str, options: list[str], **kwargs) -> str:
    return st.selectbox(label, options, key=f"in_{column}", **kwargs)


def _customer_from_state() -> dict:
    s = st.session_state
    customer = {col: s[f"in_{col}"] for col in config.CATEGORICAL_COLS}
    customer["SeniorCitizen"] = 1 if s["in_SeniorCitizen"] == "Yes" else 0
    customer["tenure"] = int(s["in_tenure"])
    customer["MonthlyCharges"] = float(s["in_MonthlyCharges"])
    # Keep the profile internally consistent, exactly as the training data is.
    if customer["InternetService"] == "No":
        for col in config.ADDON_COLS:
            customer[col] = "No internet service"
    if customer["PhoneService"] == "No":
        customer["MultipleLines"] = "No phone service"
    return customer


def _inputs() -> None:
    choices = config.CATEGORICAL_CHOICES
    with st.container(border=True, key="card-account"):
        st.subheader("Account and billing")
        a, b = st.columns(2)
        with a:
            _select("Gender", "gender", choices["gender"])
            _select("Partner", "Partner", YES_NO)
            _select("Contract", "Contract", choices["Contract"])
            _select("Payment method", "PaymentMethod", choices["PaymentMethod"])
            st.number_input("Monthly charges", 0.0, 200.0, step=0.5, key="in_MonthlyCharges")
        with b:
            _select("Senior citizen", "SeniorCitizen", YES_NO)
            _select("Dependents", "Dependents", YES_NO)
            st.number_input("Tenure (months)", 0, 72, step=1, key="in_tenure")
            _select("Paperless billing", "PaperlessBilling", YES_NO)

    with st.container(border=True, key="card-services"):
        st.subheader("Services")
        a, b = st.columns(2)
        with a:
            phone = _select("Phone service", "PhoneService", YES_NO)
            internet = _select("Internet service", "InternetService", choices["InternetService"])
        with b:
            _select("Multiple lines", "MultipleLines", YES_NO, disabled=phone == "No")
        no_internet = internet == "No"
        a, b = st.columns(2)
        labels = {
            "OnlineSecurity": "Online security",
            "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection",
            "TechSupport": "Tech support",
            "StreamingTV": "Streaming TV",
            "StreamingMovies": "Streaming movies",
        }
        for i, (col, label) in enumerate(labels.items()):
            with a if i % 2 == 0 else b:
                _select(label, col, YES_NO, disabled=no_internet)
        if no_internet:
            st.caption("Add-on services are not available without an internet plan.")


def _gauge_html(probability: float, level: str, decision: str, threshold: float, revenue: float) -> str:
    """The verdict card: a half-circle gauge that sweeps to the churn probability."""
    colour = risk_color(level)
    deg = -90 + 180 * probability
    rest = ARC_LENGTH * (1 - probability)

    def tick(p: float, bold: bool = False) -> str:
        angle = -90 + 180 * p
        return (
            f'<line x1="150" y1="14" x2="150" y2="{36 if bold else 30}" stroke="#fff" '
            f'stroke-width="{4 if bold else 2}" stroke-linecap="round" opacity="{1 if bold else 0.45}" '
            f'transform="rotate({angle:.1f} 150 150)"/>'
        )

    ticks = tick(config.LOW_RISK_MAX) + tick(config.HIGH_RISK_MIN) + tick(threshold, bold=True)
    # Streamlit's HTML sanitiser drops inline <svg>, so the gauge is an image whose animation
    # lives inside the SVG itself.
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 300 175"><style>'
        f".arc{{stroke-dasharray:{ARC_LENGTH};stroke-dashoffset:{ARC_LENGTH};"
        f"animation:f 1.1s cubic-bezier(.2,.8,.2,1) forwards}}"
        f"@keyframes f{{to{{stroke-dashoffset:{rest:.1f}}}}}"
        f".needle{{transform-origin:150px 150px;transform:rotate({deg:.1f}deg);"
        f"animation:s 1.1s cubic-bezier(.2,.8,.2,1)}}"
        f"@keyframes s{{from{{transform:rotate(-90deg)}}to{{transform:rotate({deg:.1f}deg)}}}}"
        f"@media (prefers-reduced-motion:reduce){{.arc,.needle{{animation:none}}.arc{{stroke-dashoffset:{rest:.1f}}}}}"
        "</style>"
        f'<path d="{ARC}" fill="none" stroke="rgba(255,255,255,.13)" stroke-width="22" stroke-linecap="round"/>'
        f'<path class="arc" d="{ARC}" fill="none" stroke="{colour}" stroke-width="22" stroke-linecap="round"/>'
        f"{ticks}"
        '<g class="needle"><line x1="150" y1="150" x2="150" y2="62" stroke="#fff" stroke-width="5" '
        'stroke-linecap="round"/></g>'
        '<circle cx="150" cy="150" r="11" fill="#fff"/></svg>'
    )
    uri = "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()
    return f"""
    <div class="cg-verdict">
      <img class="cg-gauge" src="{uri}" alt="Churn probability {probability:.0%}, {level} risk">
      <div class="pct">{probability:.0%}</div>
      <div class="sub">chance this customer leaves</div>
      <div style="text-align:center;margin-bottom:1rem">{risk_badge(level)}</div>
      <div class="row">
        <div><b>{html.escape(decision)}</b>Model decision</div>
        <div><b>{money(revenue)}</b>Revenue a month</div>
        <div><b>{threshold:.0%}</b>Decision line</div>
      </div>
    </div>"""


def _drivers_html(drivers: list[dict]) -> str:
    top = drivers[:7]
    biggest = max((abs(d["shap"]) for d in top), default=1.0) or 1.0
    rows = []
    for d in top:
        side = "up" if d["shap"] > 0 else "down"
        width = abs(d["shap"]) / biggest * 100
        rows.append(
            f'<div class="cg-driver"><div class="name">{html.escape(str(d["value"]))}'
            f"<small>{html.escape(d['feature'])}</small></div>"
            f'<div class="cg-bar"><i class="{side}" style="--w:{width / 2:.1f}%"></i></div></div>'
        )
    return "".join(rows) + '<div class="cg-axis"><span>Keeps them</span><span>Pushes them to leave</span></div>'


def _what_if(predictor, customer: dict, changes: dict) -> float:
    """The model's churn probability if the customer's profile were changed as given."""
    return float(predictor.predict_proba(pd.DataFrame([{**customer, **changes}]))[0])


def _shift_html(before: float, after: float) -> str:
    delta = (after - before) * 100
    good = delta < -0.5
    note = f"{abs(delta):.0f} points {'lower' if delta < 0 else 'higher'}" if abs(delta) >= 0.5 else "no change"
    return (
        f'<div class="cg-shift"><span class="from">{before:.0%}</span>'
        f'<span class="to {"good" if good else ""}">{after:.0%}</span><span class="note">{note}</span></div>'
    )


def _plays(predictor, customer: dict, level: str, drivers: list[dict]) -> None:
    recs = recommend(customer, level, drivers)
    before = predictor.predict_proba(pd.DataFrame([customer]))[0]
    for i, rec in enumerate(recs):
        with st.container(border=True, key=f"card-play-{i}"):
            tag = (
                '<span class="cg-tag">The model flags this</span>'
                if rec["model_flagged"]
                else '<span class="cg-tag plain">Fits this customer</span>'
            )
            body = f'<div class="cg-play">{tag}<h4>{html.escape(rec["title"])}</h4><p>{html.escape(rec["action"])}</p>'
            if rec["changes"]:
                after = _what_if(predictor, customer, rec["changes"])
                body += f'<p style="margin:.2rem 0 .5rem"><b>{html.escape(rec["what"])}</b> in the model:</p>'
                body += _shift_html(before, after)
            st.html(body + "</div>")
            if rec["changes"]:
                st.button(
                    "Apply to this profile",
                    key=f"apply-{i}",
                    on_click=_apply_changes,
                    args=(rec["changes"],),
                    icon=":material/play_arrow:",
                )

    combined: dict = {}
    for rec in recs:
        combined.update(rec["changes"])
    if len(recs) > 1 and combined:
        after = _what_if(predictor, customer, combined)
        with st.container(border=True, key="card-play-all"):
            st.html('<div class="cg-play"><h4>Do all of them</h4>' + _shift_html(before, after) + "</div>")
            st.button(
                "Apply all to this profile",
                key="apply-all",
                on_click=_apply_changes,
                args=(combined,),
                icon=":material/done_all:",
            )
    st.caption(
        "Before-and-after numbers are the model's estimate from patterns in past customers. They are not "
        "a promise: test an offer on a small group before rolling it out."
    )


def render() -> None:
    predictor = get_predictor()
    if "in_tenure" not in st.session_state:
        _apply_preset("High-risk example")

    with band("head", "tide", first=True):
        page_head(
            "Check a customer",
            "Describe one customer. See how likely they are to leave, why, and what could change that.",
        )
        cols = st.columns(len(PRESETS))
        for col, name in zip(cols, PRESETS, strict=True):
            col.button(name, on_click=_apply_preset, args=(name,), width="stretch")

    with band("work", "paper"):
        left, right = st.columns([5, 6], gap="large")
        with left, st.container(key="sticky-form"):
            _inputs()
        customer = _customer_from_state()

        with right:
            try:
                result = predictor.predict_one(customer, top_k=30)
            except ValidationError as exc:
                st.error(str(exc))
                return

            level = result["risk_level"]
            decision = "Likely to churn" if result["predicted_churn"] else "Likely to stay"
            st.html(
                _gauge_html(result["probability"], level, decision, predictor.threshold, customer["MonthlyCharges"])
            )
            with st.container(border=True, key="card-why"):
                st.subheader("Why the model says this")
                st.html(_drivers_html(result["drivers"]))
            st.subheader("What to do about it")
            _plays(predictor, customer, level, result["drivers"])
