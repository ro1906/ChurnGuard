from churnguard.recommend import recommend


def _driver(key, shap):
    return {"key": key, "feature": key, "value": "x", "shap": shap}


def test_low_risk_customers_get_no_retention_action(loyal):
    recs = recommend(loyal, "Low", [])
    assert len(recs) == 1
    assert recs[0]["title"] == "No retention action needed"


def test_model_flagged_actions_rank_first_strongest_first(risky):
    drivers = [_driver("Contract", 0.9), _driver("tenure", 0.2), _driver("PaymentMethod", 0.1)]
    recs = recommend(risky, "High", drivers)
    assert recs[0]["title"] == "Offer a longer-term contract"
    assert recs[0]["model_flagged"] is True
    assert [r["impact"] for r in recs] == sorted([r["impact"] for r in recs], reverse=True)


def test_actions_not_flagged_by_the_model_are_still_offered_after_flagged_ones(risky):
    recs = recommend(risky, "High", [_driver("Contract", 0.5)], max_items=10)
    flags = [r["model_flagged"] for r in recs]
    assert flags == sorted(flags, reverse=True)
    assert len(recs) > 1


def test_negative_drivers_do_not_flag_an_action(risky):
    recs = recommend(risky, "High", [_driver("Contract", -0.8)])
    assert not any(r["model_flagged"] for r in recs)


def test_max_items_is_respected(risky):
    assert len(recommend(risky, "High", [], max_items=2)) == 2


def test_inapplicable_rules_are_not_suggested(loyal):
    titles = [r["title"] for r in recommend(loyal, "Medium", [], max_items=10)]
    assert "Offer a longer-term contract" not in titles  # already on a two-year contract


def test_fallback_when_no_rule_applies(loyal):
    loyal.update(
        {"InternetService": "No", "tenure": 50, "MonthlyCharges": 30.0,
         "OnlineSecurity": "No internet service", "TechSupport": "No internet service"}
    )
    recs = recommend(loyal, "Medium", [])
    assert recs[0]["title"] == "Proactive check-in call"
