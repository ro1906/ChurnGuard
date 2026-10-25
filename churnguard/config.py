"""Central configuration: paths, schema, thresholds. Change things here, not in the code."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

DATA_PATH = DATA_DIR / "telco_churn.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

DATA_SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Model selection: pick the best cross-validated ROC-AUC. If other models are within this
# margin of the best (a statistical tie on this dataset), prefer the model below because it
# captures feature interactions and has exact TreeSHAP explanations.
SELECTION_TIE_MARGIN = 0.005
PREFERRED_MODEL_ON_TIE = "XGBoost"
TUNING_ITERATIONS = 20

TARGET = "Churn"
ID_COL = "customerID"

# Allowed values for every categorical column. Used for validation, one-hot encoding
# (so the encoded columns are always identical) and for the dropdowns in the UI.
CATEGORICAL_CHOICES: dict[str, list[str]] = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No phone service", "No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No internet service", "No", "Yes"],
    "OnlineBackup": ["No internet service", "No", "Yes"],
    "DeviceProtection": ["No internet service", "No", "Yes"],
    "TechSupport": ["No internet service", "No", "Yes"],
    "StreamingTV": ["No internet service", "No", "Yes"],
    "StreamingMovies": ["No internet service", "No", "Yes"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_COLS = list(CATEGORICAL_CHOICES)

NUMERIC_COLS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 100_000),
}

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ENGINEERED_COLS = ["num_addons", "is_autopay", "spend_gap", "is_new_customer"]

# Columns a caller must provide (TotalCharges is optional: derived if missing).
REQUIRED_COLS = CATEGORICAL_COLS + [c for c in NUMERIC_COLS if c != "TotalCharges"]
MODEL_INPUT_COLS = CATEGORICAL_COLS + NUMERIC_COLS

# Probability bands used to turn a churn probability into a business risk level.
LOW_RISK_MAX = 0.30
HIGH_RISK_MIN = 0.60
RISK_LEVELS = ["Low", "Medium", "High"]

NEW_CUSTOMER_MONTHS = 6
