import numpy as np
import pandas as pd
import pytest

from churnguard import config
from churnguard.explain import group_shap, source_feature
from churnguard.features import add_features, build_preprocessor


def _frame(risky, **overrides):
    row = {**risky, "TotalCharges": risky["MonthlyCharges"] * risky["tenure"], **overrides}
    return pd.DataFrame([row])


def test_num_addons_counts_only_yes(risky):
    assert add_features(_frame(risky))["num_addons"].iloc[0] == 2  # streaming TV + movies
    assert add_features(_frame(risky, TechSupport="Yes"))["num_addons"].iloc[0] == 3


def test_autopay_flag(risky):
    assert add_features(_frame(risky))["is_autopay"].iloc[0] == 0
    out = add_features(_frame(risky, PaymentMethod="Credit card (automatic)"))
    assert out["is_autopay"].iloc[0] == 1


def test_new_customer_flag(risky):
    assert add_features(_frame(risky, tenure=3))["is_new_customer"].iloc[0] == 1
    assert add_features(_frame(risky, tenure=30))["is_new_customer"].iloc[0] == 0


def test_spend_gap_is_zero_for_brand_new_customers(risky):
    out = add_features(_frame(risky, tenure=0, TotalCharges=0.0))
    assert out["spend_gap"].iloc[0] == 0


def test_add_features_does_not_mutate_input(risky):
    frame = _frame(risky)
    before = frame.copy()
    add_features(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_preprocessor_handles_unseen_category(dataset):
    prep = build_preprocessor(scale_numeric=True)
    X = add_features(dataset[config.MODEL_INPUT_COLS])
    prep.fit(X)
    odd = X.head(3).copy()
    odd["Contract"] = "Five year"
    out = prep.transform(odd)
    assert out.shape == (3, len(prep.get_feature_names_out()))


def test_source_feature_maps_one_hot_columns_back():
    assert source_feature("cat__Contract_Two year") == "Contract"
    assert source_feature("cat__PaymentMethod_Electronic check") == "PaymentMethod"
    assert source_feature("num__tenure") == "tenure"


def test_group_shap_preserves_the_total_contribution():
    names = ["cat__Contract_Month-to-month", "cat__Contract_One year", "num__tenure"]
    values = np.array([[0.4, -0.1, 0.3], [-0.2, 0.0, 0.1]])
    grouped = group_shap(values, names)
    assert set(grouped.columns) == {"Contract", "tenure"}
    np.testing.assert_allclose(grouped.sum(axis=1), values.sum(axis=1))
    assert grouped.loc[0, "Contract"] == pytest.approx(0.3)
