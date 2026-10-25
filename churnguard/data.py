"""Loading and cleaning the customer dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from churnguard import config

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of a raw Telco-style dataframe.

    - strips whitespace from column names and text values
    - converts TotalCharges to numeric (the raw file has blank strings for brand-new customers)
    - encodes the Churn target as 1/0 when it is present
    - removes duplicate customer ids
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    for col in out.columns:
        if out[col].dtype == object or str(out[col].dtype).startswith("str"):
            out[col] = out[col].astype("string").str.strip().astype(object)

    if "tenure" in out.columns:
        out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    if "MonthlyCharges" in out.columns:
        out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")

    if "TotalCharges" in out.columns:
        total = pd.to_numeric(out["TotalCharges"], errors="coerce")
        # Blank TotalCharges only happens for customers with tenure 0 (no bill yet).
        fallback = out["MonthlyCharges"] * out["tenure"]
        out["TotalCharges"] = total.fillna(fallback)

    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = pd.to_numeric(out["SeniorCitizen"], errors="coerce")

    if config.TARGET in out.columns:
        out[config.TARGET] = out[config.TARGET].map({"Yes": 1, "No": 0, 1: 1, 0: 0})

    if config.ID_COL in out.columns:
        before = len(out)
        out = out.drop_duplicates(subset=config.ID_COL)
        if len(out) != before:
            logger.warning("Dropped %d duplicate customer rows", before - len(out))

    return out.reset_index(drop=True)


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    """Load and clean the training dataset from CSV."""
    path = Path(path) if path else config.DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the IBM Telco Customer Churn CSV with:\n"
            f"  curl -L -o {config.DATA_PATH} {config.DATA_SOURCE_URL}"
        )
    df = clean(pd.read_csv(path))

    missing = [c for c in config.MODEL_INPUT_COLS + [config.TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    if df[config.TARGET].isna().any():
        raise ValueError("Churn column contains values other than Yes/No")
    logger.info("Loaded %d customers from %s", len(df), path)
    return df
