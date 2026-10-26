"""Single customer prediction page: probability, risk level, SHAP reasons, recommendations."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from churnguard import config
from churnguard.predict import ValidationError
from churnguard.recommend import recommend
from churnguard.ui.common import (
    DOWN_COLOR,
    RISK_COLORS,
    UP_COLOR,
    get_predictor,
    header,
    risk_badge,
    style_figure,
)

PRESETS = {
    "High-risk example": {
        "gender": "Female", "SeniorCitizen": "No", "Partner": "No", "Dependents": "No",
        "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0,
    },
    "Stable customer": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
        "tenure": 24, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "One year", "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 55.0,
    },
    "Loyal low-risk example": {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
        "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0,
    },
}
YES_NO = ["No", "Yes"]


def _apply_preset(name: str) -> None:
    for key, value in PRESETS[name].items():
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
    with st.expander("Account and billing", expanded=True):
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

    with st.expander("Services", expanded=True):
        a, b = st.columns(2)
        with a:
            phone = _select("Phone service", "PhoneService", YES_NO)
            internet = _select("Internet service", "InternetService", choices["InternetService"])
        with b:
            _select("Multiple lines", "MultipleLines", YES_NO, disabled=phone == "No")
        no_internet = internet == "No"
        a, b = st.columns(2)
        labels = {
            "OnlineSecurity": "Online security", "OnlineBackup": "Online backup",
            "DeviceProtection": "Device protection", "TechSupport": "Tech support",
            "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies",
        }
        for i, (col, label) in enumerate(labels.items()):
            with (a if i % 2 == 0 else b):
                _select(label, col, YES_NO, disabled=no_internet)
        if no_internet:
            st.caption("Add-on services are not available without an internet plan.")


def _gauge(probability: float, level: str, threshold: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "ticksuffix": "%"},
                "bar": {"color": RISK_COLORS[level], "thickness": 0.3},
                "steps": [
                    {"range": [0, config.LOW_RISK_MAX * 100], "color": "rgba(42,157,143,0.18)"},
                    {
                        "range": [config.LOW_RISK_MAX * 100, config.HIGH_RISK_MIN * 100],
                        "color": "rgba(233,162,59,0.20)",
                    },
                    {"range": [config.HIGH_RISK_MIN * 100, 100], "color": "rgba(209,73,91,0.18)"},
                ],
                "threshold": {"line": {"color": "#333", "width": 3}, "value": threshold * 100},
            },
        )
    )
    return style_figure(fig, height=250)


def _drivers_chart(drivers: list[dict]) -> go.Figure:
    top = sorted(drivers[:8], key=lambda d: d["shap"])
    fig = go.Figure(
        go.Bar(
            x=[d["shap"] for d in top],
            y=[f"{d['feature']}: {d['value']}" for d in top],
            orientation="h",
            marker_color=[UP_COLOR if d["shap"] > 0 else DOWN_COLOR for d in top],
            hovertemplate="%{y}<br>Impact: %{x:+.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Lowers risk  <-  impact on churn (log-odds)  ->  Raises risk", zeroline=True)
    return style_figure(fig, height=340, title="Why the model predicts this")


def render() -> None:
    header(
        "Predict a customer",
        "Enter a customer profile to get churn probability, the reasons behind it, and what to do.",
    )
    predictor = get_predictor()

    if "in_tenure" not in st.session_state:
        _apply_preset("High-risk example")

    st.write("Load an example:")
    cols = st.columns(len(PRESETS))
    for col, name in zip(cols, PRESETS, strict=True):
        col.button(name, on_click=_apply_preset, args=(name,), width="stretch")

    left, right = st.columns([2, 3], gap="large")
    with left:
        _inputs()
    customer = _customer_from_state()

    with right:
        try:
            result = predictor.predict_one(customer, top_k=30)
        except ValidationError as exc:
            st.error(str(exc))
            return

        level = result["risk_level"]
        st.plotly_chart(_gauge(result["probability"], level, predictor.threshold), width="stretch")
        decision = "Likely to churn" if result["predicted_churn"] else "Likely to stay"
        st.markdown(risk_badge(level), unsafe_allow_html=True)
        a, b = st.columns(2)  # two wide columns so the decision text is never cut off
        a.metric("Model decision", decision)
        b.metric("Monthly revenue at stake", f"{customer['MonthlyCharges']:,.2f}")
        st.caption(
            f"Risk bands: Low below {config.LOW_RISK_MAX:.0%}, High from {config.HIGH_RISK_MIN:.0%}. "
            f"The black line marks the model's decision threshold ({predictor.threshold:.0%})."
        )

        st.plotly_chart(_drivers_chart(result["drivers"]), width="stretch")
        st.caption("Red bars push this customer's churn risk up; blue bars push it down (SHAP values).")

        st.subheader("Recommended retention actions")
        for rec in recommend(customer, level, result["drivers"]):
            with st.container(border=True):
                st.markdown(f"**{rec['title']}**")
                st.write(rec["action"])
                st.caption(rec["reason"])
