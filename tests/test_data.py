import pandas as pd
import pytest

from churnguard import config
from churnguard.data import clean, load_dataset


def _raw_row(**overrides):
    row = {
        "customerID": "0001-AAAAA", "tenure": "0", "MonthlyCharges": "52.5",
        "TotalCharges": " ", "SeniorCitizen": 0, "Churn": "No",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_blank_total_charges_becomes_numeric_zero_for_new_customers():
    out = clean(_raw_row())
    assert out.loc[0, "TotalCharges"] == 0
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])


def test_churn_is_encoded_as_binary():
    out = clean(pd.concat([_raw_row(Churn="Yes", customerID="a"), _raw_row(Churn="No", customerID="b")]))
    assert out[config.TARGET].tolist() == [1, 0]


def test_duplicate_customers_are_dropped():
    out = clean(pd.concat([_raw_row(), _raw_row()]))
    assert len(out) == 1


def test_whitespace_is_stripped_from_text_values():
    out = clean(_raw_row(customerID="  0001-AAAAA  "))
    assert out.loc[0, "customerID"] == "0001-AAAAA"


def test_real_dataset_loads_cleanly(dataset):
    assert len(dataset) == 7043
    assert 0.25 < dataset[config.TARGET].mean() < 0.28
    assert dataset[config.MODEL_INPUT_COLS].notna().all().all()


def test_every_category_in_data_is_declared_in_config(dataset):
    for col, allowed in config.CATEGORICAL_CHOICES.items():
        assert set(dataset[col].unique()) <= set(allowed), col


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="curl"):
        load_dataset(tmp_path / "nope.csv")


def test_missing_columns_are_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"customerID": ["x"], "Churn": ["No"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)
