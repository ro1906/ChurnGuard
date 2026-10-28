import pandas as pd
import pytest

from churnguard import config
from churnguard.predict import ValidationError
from churnguard.ui.overview import prepare_own_data


def _customers(dataset, n=60):
    return dataset.sample(n, random_state=1).copy()


def test_file_with_past_churn_column_keeps_real_history(dataset, predictor):
    raw = _customers(dataset)
    raw[config.TARGET] = raw[config.TARGET].map({1: "Yes", 0: "No"})
    scored, history = prepare_own_data(raw, predictor)
    assert history is True
    assert set(scored[config.TARGET].unique()) <= {0, 1}
    assert {"churn_probability", "risk_level"} <= set(scored.columns)


def test_file_without_churn_column_uses_predicted_risk(dataset, predictor):
    raw = _customers(dataset).drop(columns=[config.TARGET])
    scored, history = prepare_own_data(raw, predictor)
    assert history is False
    assert scored[config.TARGET].equals(scored["churn_probability"])


def test_unusable_churn_column_is_ignored_not_fatal(dataset, predictor):
    raw = _customers(dataset)
    raw[config.TARGET] = "maybe"
    _, history = prepare_own_data(raw, predictor)
    assert history is False


def test_bad_file_gives_a_readable_error(dataset, predictor):
    raw = _customers(dataset).drop(columns=["Contract"])
    with pytest.raises(ValidationError, match="Contract"):
        prepare_own_data(raw, predictor)


def test_oversized_file_is_refused(dataset, predictor, monkeypatch):
    monkeypatch.setattr("churnguard.ui.overview.MAX_ROWS", 10)
    with pytest.raises(ValidationError, match="limit"):
        prepare_own_data(_customers(dataset, 30), predictor)


def test_empty_file_is_refused(dataset, predictor):
    with pytest.raises(ValidationError):
        prepare_own_data(pd.DataFrame(columns=_customers(dataset).columns), predictor)
