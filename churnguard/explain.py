"""SHAP helpers shared by training (global importance) and inference (per-customer reasons)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

from churnguard import config

DISPLAY_NAMES = {
    "tenure": "Tenure (months)",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "SeniorCitizen": "Senior citizen",
    "num_addons": "Number of add-on services",
    "is_autopay": "Auto-pay enabled",
    "spend_gap": "This month vs. average spend",
    "is_new_customer": "New customer (6 months or less)",
    "InternetService": "Internet service",
    "PaymentMethod": "Payment method",
    "PaperlessBilling": "Paperless billing",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "MultipleLines": "Multiple lines",
    "PhoneService": "Phone service",
    "Contract": "Contract type",
    "gender": "Gender",
    "Partner": "Has partner",
    "Dependents": "Has dependents",
}


def display_name(feature: str) -> str:
    return DISPLAY_NAMES.get(feature, feature)


def source_feature(transformed_name: str) -> str:
    """Map a transformed column such as 'cat__Contract_Two year' back to 'Contract'."""
    name = transformed_name.split("__", 1)[-1]
    for col in config.CATEGORICAL_COLS:
        if name.startswith(f"{col}_"):
            return col
    return name


def build_explainer(model, background: np.ndarray):
    """Pick the right SHAP explainer for the fitted model type."""
    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    return shap.TreeExplainer(model)


def shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """SHAP values for the positive (churn) class as a 2-D array of shape (rows, features)."""
    values = explainer.shap_values(X)
    if isinstance(values, list):  # older shap: one array per class
        values = values[1]
    values = np.asarray(values)
    if values.ndim == 3:  # newer shap: (rows, features, classes)
        values = values[:, :, 1]
    return values


def group_shap(values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Sum one-hot SHAP columns back to their original feature, one column per customer attribute.

    One-hot columns of the same attribute belong together: the customer's contract type is one
    reason, not three. Grouping makes explanations readable ("Contract type: Month-to-month").
    """
    sources = [source_feature(n) for n in feature_names]
    return pd.DataFrame(values, columns=sources).T.groupby(level=0).sum().T
