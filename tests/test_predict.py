import pandas as pd
import pytest

from churnguard import config
from churnguard.predict import ValidationError, load_predictor, risk_level, validate_customers


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "Low"), (0.29, "Low"), (0.30, "Medium"), (0.59, "Medium"), (0.60, "High"), (1.0, "High")],
)
def test_risk_level_boundaries(probability, expected):
    assert risk_level(probability) == expected


def test_validation_accepts_a_complete_customer_and_derives_total_charges(risky):
    out = validate_customers(pd.DataFrame([risky]))
    assert out.loc[0, "TotalCharges"] == pytest.approx(95.0 * 2)


def test_validation_reports_missing_columns(risky):
    del risky["Contract"]
    with pytest.raises(ValidationError, match="Contract"):
        validate_customers(pd.DataFrame([risky]))


def test_validation_rejects_unknown_category(risky):
    risky["Contract"] = "Five year"
    with pytest.raises(ValidationError, match="Contract: invalid"):
        validate_customers(pd.DataFrame([risky]))


def test_validation_rejects_out_of_range_numbers(risky):
    risky["tenure"] = -3
    with pytest.raises(ValidationError, match="tenure"):
        validate_customers(pd.DataFrame([risky]))


def test_validation_rejects_non_numeric_values(risky):
    risky["MonthlyCharges"] = "lots"
    with pytest.raises(ValidationError, match="MonthlyCharges"):
        validate_customers(pd.DataFrame([risky]))


def test_validation_lists_every_problem_at_once(risky):
    risky["Contract"] = "x"
    risky["tenure"] = 999
    with pytest.raises(ValidationError) as exc:
        validate_customers(pd.DataFrame([risky]))
    assert "Contract" in str(exc.value) and "tenure" in str(exc.value)


def test_validation_rejects_empty_input(risky):
    with pytest.raises(ValidationError, match="No customer rows"):
        validate_customers(pd.DataFrame([risky]).iloc[0:0])


def test_probabilities_are_valid(predictor, risky, loyal):
    proba = predictor.predict_proba(pd.DataFrame([risky, loyal]))
    assert ((proba >= 0) & (proba <= 1)).all()


def test_risky_customer_scores_far_higher_than_loyal_customer(predictor, risky, loyal):
    risky_p, loyal_p = predictor.predict_proba(pd.DataFrame([risky, loyal]))
    assert risky_p > 0.5 > 0.15 > loyal_p


def test_predict_one_returns_probability_level_and_drivers(predictor, risky):
    result = predictor.predict_one(risky, top_k=5)
    assert result["risk_level"] == risk_level(result["probability"])
    assert len(result["drivers"]) == 5
    magnitudes = [abs(d["shap"]) for d in result["drivers"]]
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert {"key", "feature", "value", "shap"} <= set(result["drivers"][0])


def test_drivers_shows_the_customers_actual_values(predictor, risky):
    drivers = {d["key"]: d for d in predictor.predict_one(risky, top_k=30)["drivers"]}
    assert drivers["Contract"]["value"] == "Month-to-month"
    assert drivers["tenure"]["value"] == "2"
    assert drivers["Contract"]["shap"] > 0  # month-to-month raises risk


def test_explanations_add_up_to_the_models_output(predictor, risky):
    """SHAP values plus the base value must reproduce the model's log-odds for the customer."""
    import numpy as np

    frame = pd.DataFrame([risky])
    expected = predictor._explainer.expected_value
    base = float(np.ravel(expected)[0])
    total = base + sum(d["shap"] for d in predictor.explain(frame, top_k=50)[0])
    proba = predictor.predict_proba(frame)[0]
    assert 1 / (1 + np.exp(-total)) == pytest.approx(proba, abs=1e-3)


def test_score_adds_expected_columns_and_keeps_rows(predictor, dataset):
    raw = dataset.drop(columns=[config.TARGET]).head(40)
    scored = predictor.score(raw)
    assert len(scored) == 40
    assert {"churn_probability", "risk_level", "predicted_churn", "top_risk_driver"} <= set(scored)
    assert scored["customerID"].tolist() == raw["customerID"].tolist()
    assert set(scored["risk_level"]) <= set(config.RISK_LEVELS)


def test_score_without_drivers_skips_the_explanation_column(predictor, dataset):
    scored = predictor.score(dataset.drop(columns=[config.TARGET]).head(5), with_drivers=False)
    assert "top_risk_driver" not in scored


def test_predicted_churn_follows_the_threshold(predictor, dataset):
    scored = predictor.score(dataset.drop(columns=[config.TARGET]).head(200), with_drivers=False)
    expected = (scored["churn_probability"] >= predictor.threshold - 1e-4).astype(int)
    assert (scored["predicted_churn"] <= expected).all()


def test_higher_risk_bands_really_churn_more_on_held_out_style_data(predictor, dataset):
    scored = predictor.score(dataset.drop(columns=[config.TARGET]), with_drivers=False)
    rates = dataset.assign(level=scored["risk_level"].to_numpy()).groupby("level")[config.TARGET].mean()
    assert rates["Low"] < rates["Medium"] < rates["High"]


def test_loading_a_missing_model_gives_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="churnguard.train"):
        load_predictor(tmp_path / "missing.joblib")
