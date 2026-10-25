"""Inference: validation, churn probability, risk level and per-customer SHAP explanations."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnguard import config
from churnguard.explain import build_explainer, display_name, group_shap, shap_values
from churnguard.features import add_features

logger = logging.getLogger(__name__)

YES_NO_FEATURES = {"SeniorCitizen", "is_autopay", "is_new_customer"}


class ValidationError(ValueError):
    """Raised when customer data is missing columns or contains invalid values."""


def risk_level(probability: float) -> str:
    """Convert a churn probability into a business risk band."""
    if probability < config.LOW_RISK_MAX:
        return "Low"
    if probability < config.HIGH_RISK_MIN:
        return "Medium"
    return "High"


def validate_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Check that customer data is complete and valid; return a cleaned copy.

    Raises ValidationError with a readable message listing every problem found.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    missing = [c for c in config.REQUIRED_COLS if c not in out.columns]
    if missing:
        raise ValidationError(f"Missing required column(s): {', '.join(missing)}")
    if out.empty:
        raise ValidationError("No customer rows to score.")

    errors: list[str] = []
    for col, allowed in config.CATEGORICAL_CHOICES.items():
        out[col] = out[col].astype(str).str.strip()
        bad = sorted(set(out[col]) - set(allowed))
        if bad:
            errors.append(f"{col}: invalid value(s) {bad[:3]}. Allowed: {allowed}")

    for col in config.NUMERIC_COLS:
        if col not in out.columns:
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce")
        if col == "TotalCharges":
            continue
        if out[col].isna().any():
            errors.append(f"{col}: contains missing or non-numeric values")
            continue
        lo, hi = config.NUMERIC_RANGES[col]
        if ((out[col] < lo) | (out[col] > hi)).any():
            errors.append(f"{col}: values must be between {lo} and {hi}")

    if errors:
        raise ValidationError("\n".join(errors))

    derived = out["MonthlyCharges"] * out["tenure"]
    if "TotalCharges" not in out.columns:
        out["TotalCharges"] = derived
    else:
        out["TotalCharges"] = out["TotalCharges"].fillna(derived)
    lo, hi = config.NUMERIC_RANGES["TotalCharges"]
    if ((out["TotalCharges"] < lo) | (out["TotalCharges"] > hi)).any():
        raise ValidationError(f"TotalCharges: values must be between {lo} and {hi}")
    return out


def _format_value(feature: str, value) -> str:
    if feature in YES_NO_FEATURES:
        return "Yes" if int(value) == 1 else "No"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(int(value)) if float(value).is_integer() else f"{value:.2f}"
    return str(value)


class ChurnPredictor:
    """Wraps the trained pipeline, decision threshold and SHAP explainer."""

    def __init__(self, bundle: dict):
        self.pipeline = bundle["pipeline"]
        self.model_name: str = bundle["model_name"]
        self.threshold: float = float(bundle["threshold"])
        self.feature_names: list[str] = list(bundle["feature_names"])
        self.trained_at: str = bundle.get("trained_at", "")
        self.model = self.pipeline.named_steps["model"]
        self._explainer = build_explainer(self.model, bundle["background"])

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = validate_customers(df)[config.MODEL_INPUT_COLS]
        return self.pipeline.predict_proba(X)[:, 1]

    def explain(self, df: pd.DataFrame, top_k: int = 8) -> list[list[dict]]:
        """Top SHAP drivers for every row. Positive 'shap' pushes churn risk up."""
        X = validate_customers(df)[config.MODEL_INPUT_COLS]
        grouped = group_shap(
            shap_values(self._explainer, self.pipeline[:-1].transform(X)), self.feature_names
        )
        shown = add_features(X).reset_index(drop=True)
        results: list[list[dict]] = []
        for i in range(len(grouped)):
            row = grouped.iloc[i]
            order = row.abs().sort_values(ascending=False).index[:top_k]
            results.append(
                [
                    {
                        "key": key,
                        "feature": display_name(key),
                        "value": _format_value(key, shown.at[i, key]),
                        "shap": float(row[key]),
                    }
                    for key in order
                ]
            )
        return results

    def predict_one(self, customer: dict, top_k: int = 8) -> dict:
        """Full prediction for a single customer, including explanation drivers."""
        frame = pd.DataFrame([customer])
        probability = float(self.predict_proba(frame)[0])
        return {
            "probability": probability,
            "risk_level": risk_level(probability),
            "predicted_churn": probability >= self.threshold,
            "drivers": self.explain(frame, top_k=top_k)[0],
        }

    def score(self, df: pd.DataFrame, with_drivers: bool = True) -> pd.DataFrame:
        """Score many customers; returns the input plus probability, risk level and top driver."""
        validated = validate_customers(df)
        proba = self.pipeline.predict_proba(validated[config.MODEL_INPUT_COLS])[:, 1]
        out = validated.copy()
        out["churn_probability"] = np.round(proba, 4)
        out["risk_level"] = [risk_level(p) for p in proba]
        out["predicted_churn"] = (proba >= self.threshold).astype(int)
        if with_drivers:
            drivers = self.explain(validated, top_k=3)
            out["top_risk_driver"] = [
                next((f"{d['feature']}: {d['value']}" for d in row if d["shap"] > 0), "None")
                for row in drivers
            ]
        return out


def load_predictor(model_path: str | Path | None = None) -> ChurnPredictor:
    path = Path(model_path) if model_path else config.MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Train one first with:  python -m churnguard.train"
        )
    logger.info("Loading model from %s", path)
    return ChurnPredictor(joblib.load(path))
