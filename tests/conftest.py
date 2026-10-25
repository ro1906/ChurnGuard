import pytest

from churnguard.data import load_dataset
from churnguard.predict import load_predictor
from churnguard.train import train

RISKY_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0,
}
LOYAL_CUSTOMER = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "tenure": 60,
    "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL",
    "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes",
    "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0,
}


@pytest.fixture(scope="session")
def dataset():
    return load_dataset()


@pytest.fixture(scope="session")
def trained(tmp_path_factory, dataset):
    """Train a small, fast model once for the whole test session."""
    model_dir = tmp_path_factory.mktemp("models")
    metrics = train(df=dataset.sample(2500, random_state=0), model_dir=model_dir, tune=False)
    return metrics, model_dir


@pytest.fixture(scope="session")
def predictor(trained):
    _, model_dir = trained
    return load_predictor(model_dir / "churn_model.joblib")


@pytest.fixture
def risky():
    return dict(RISKY_CUSTOMER)


@pytest.fixture
def loyal():
    return dict(LOYAL_CUSTOMER)
