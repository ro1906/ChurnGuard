"""Feature engineering and the preprocessing step of the sklearn pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churnguard import config


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-meaningful engineered features to a validated customer dataframe."""
    out = df.copy()
    out["num_addons"] = (out[config.ADDON_COLS] == "Yes").sum(axis=1).astype(int)
    out["is_autopay"] = (
        out["PaymentMethod"].astype(str).str.contains("automatic", case=False).astype(int)
    )
    # How far this month's bill is from the customer's historical average monthly spend.
    avg_spend = np.where(
        out["tenure"] > 0,
        out["TotalCharges"] / out["tenure"].clip(lower=1),
        out["MonthlyCharges"],
    )
    out["spend_gap"] = (out["MonthlyCharges"] - avg_spend).round(4)
    out["is_new_customer"] = (out["tenure"] <= config.NEW_CUSTOMER_MONTHS).astype(int)
    return out


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """One-hot encode categoricals; optionally standardise numerics (needed for linear models)."""
    numeric = config.NUMERIC_COLS + config.ENGINEERED_COLS
    encoder = OneHotEncoder(
        categories=[config.CATEGORICAL_CHOICES[c] for c in config.CATEGORICAL_COLS],
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        [
            ("cat", encoder, config.CATEGORICAL_COLS),
            ("num", StandardScaler() if scale_numeric else "passthrough", numeric),
        ],
        verbose_feature_names_out=True,
    )


def feature_step() -> FunctionTransformer:
    return FunctionTransformer(add_features, validate=False)
