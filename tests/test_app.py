"""Smoke tests that run each Streamlit page headlessly and interact with it."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_FILE = str(Path(__file__).resolve().parent.parent / "app.py")


def _page(name: str) -> AppTest:
    return AppTest.from_string(f"from churnguard.ui import {name}\n{name}.render()", default_timeout=180)


@pytest.fixture(scope="module", autouse=True)
def _ensure_default_model():
    """The UI loads models/churn_model.joblib; train it once if the repo does not ship one."""
    from churnguard import config

    if not config.MODEL_PATH.exists():
        from churnguard.train import train

        train(tune=False)


def test_main_app_boots_and_shows_overview():
    at = AppTest.from_file(APP_FILE, default_timeout=180).run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Customers" in labels and "Monthly revenue at high risk" in labels


def _html(at) -> str:
    """All st.html bodies on the page, joined (the redesigned pages render through st.html)."""
    return " ".join(el.proto.body for el in at.get("html"))


def test_predict_page_starts_with_a_high_risk_example():
    at = _page("single").run()
    assert not at.exception
    assert "High risk" in _html(at)


def test_predict_page_updates_when_loading_a_loyal_example():
    at = _page("single").run()
    at.button[2].click().run()
    assert not at.exception
    assert "Low risk" in _html(at)
    assert "No retention action needed" in _html(at)


def test_addons_are_disabled_without_internet_service():
    at = _page("single").run()
    at.selectbox(key="in_InternetService").set_value("No").run()
    assert not at.exception
    assert at.selectbox(key="in_TechSupport").disabled


def test_batch_page_defaults_to_all_demo_customers():
    at = _page("batch").run()
    assert not at.exception
    assert at.radio[0].value.startswith("Use all")
    assert any(m.label == "Customers scored" and m.value == "7,043" for m in at.metric)


def test_batch_page_scores_the_sample_customers():
    at = _page("batch").run()
    at.radio[0].set_value("Use 500 sample customers").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Customers scored"] == "500"


def test_performance_page_reports_held_out_metrics():
    at = _page("performance").run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Selected model"] in {"XGBoost", "Random Forest", "Logistic Regression"}
