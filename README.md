# ChurnGuard: Explainable Customer Churn Prediction and Retention System

ChurnGuard predicts which telecom customers are likely to cancel, explains **why** with SHAP,
assigns a Low / Medium / High risk level, and recommends retention actions. It is an end-to-end
web application: data cleaning, model comparison and tuning, an explainability layer, a
Streamlit dashboard (single-customer and batch scoring), automated tests, Docker and CI.

**Stack:** Python, scikit-learn, XGBoost, SHAP, Streamlit, Plotly, pytest, Docker, GitHub Actions

## What the app does

| Page | What you can do |
| --- | --- |
| **Overview** | See churn patterns by contract, tenure, internet service and payment method, plus how many customers and how much monthly revenue sit at high risk. |
| **Predict a customer** | Enter a customer profile and get the churn probability, a risk level, a SHAP chart showing the reasons, and ranked retention actions. Three one-click example customers included. |
| **Batch scoring** | Upload a CSV (or use sample data), get risk levels and each customer's top risk driver, filter by risk level and download a prioritised call list. Invalid files produce a clear list of problems. |
| **Model performance** | Model comparison, ROC curve, confusion matrix, global SHAP importance and the method used. |

## Results

Dataset: IBM Telco Customer Churn sample (7,043 customers, 19 raw attributes, 26.5% churn).
80/20 stratified split. Model selection used 5-fold cross-validation on the training split only.

| Model | CV ROC-AUC | Test ROC-AUC | Test PR-AUC |
| --- | --- | --- | --- |
| Logistic Regression | 0.847 +/- 0.011 | 0.846 | 0.654 |
| Random Forest | 0.844 +/- 0.009 | 0.841 | 0.654 |
| **XGBoost (tuned, selected)** | **0.851 +/- 0.012** | **0.847** | **0.665** |

At the tuned decision threshold of 0.35 the selected model finds **73% of customers who churn**
(recall) with **56% precision** on 1,409 held-out customers. The default 0.5 threshold finds only
52% of churners, which is why the threshold is tuned. The three models are close on this
dataset; the differences are smaller than the fold-to-fold standard deviation.

Top churn drivers (mean absolute SHAP): contract type, tenure, internet service, online
security, monthly charges, payment method.

## Architecture

```
data/telco_churn.csv
        |
  data.py  ->  clean (blank TotalCharges, target encoding, duplicates)
        |
  features.py  ->  engineered features + one-hot encoding (inside the sklearn Pipeline)
        |
  train.py  ->  compare LR / RF / XGBoost (5-fold CV), tune XGBoost, pick model,
        |         tune threshold, evaluate on held-out set, refit on all data
        v
  models/churn_model.joblib + models/metrics.json
        |
  predict.py  ->  validation, probability, risk level, SHAP reasons (explain.py)
  recommend.py -> retention rules ranked by SHAP impact
        |
  app.py + churnguard/ui/  ->  Streamlit dashboard
```

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
python -m churnguard.train        # about 30 seconds; writes models/
streamlit run app.py              # opens http://localhost:8501
pytest                            # 64 tests
ruff check .                      # lint
```

The repository ships with the dataset and a trained model. If the model file is missing or was
saved under different library versions, the app trains a fresh one automatically on start.

The dataset can be re-downloaded with:

```bash
curl -L -o data/telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Docker

```bash
docker build -t churnguard .
docker run --rm -p 8501:8501 churnguard
```

The image trains the model at build time so it always matches the installed library versions,
runs as a non-root user and has a health check. The `PORT` environment variable is honoured.

## Deploy

**Streamlit Community Cloud (free, easiest):** push the repo to GitHub, go to
share.streamlit.io, choose the repo, set the main file to `app.py`, deploy.

**Render / Hugging Face Spaces (Docker):** create a new Docker web service or Space from the
repo. On Spaces add `sdk: docker` and `app_port: 8501` to the README front matter.

## Project structure

```
app.py                     Streamlit entry point and navigation
churnguard/
  config.py                paths, schema, thresholds (single source of truth)
  data.py                  loading and cleaning
  features.py              feature engineering and preprocessing
  train.py                 model comparison, tuning, threshold, evaluation, saving
  explain.py               SHAP helpers (grouped per customer attribute)
  predict.py               validation, scoring, explanations
  recommend.py             retention rules ranked by SHAP impact
  ui/                      one module per page
tests/                     64 pytest tests (data, features, training, inference, UI)
.github/workflows/ci.yml   lint, train, test, build and health-check the Docker image
```

## Design decisions

- **Everything is inside one sklearn `Pipeline`.** Feature engineering and encoding are fitted
  only on training folds, so cross-validation has no preprocessing leakage, and the saved
  artifact accepts raw customer rows.
- **Imbalance handled with stratification, PR-AUC and threshold tuning**, not resampling, so the
  predicted probabilities stay meaningful and the Low / Medium / High bands can be read as risk.
- **Model selection never sees the test set.** A tie-break rule (documented in `config.py`)
  prefers XGBoost when models are statistically tied.
- **SHAP values are grouped by customer attribute.** "Contract type: Month-to-month" is one
  reason, not three one-hot columns. A test checks that SHAP values plus the base value
  reproduce the model output.
- **Strict input validation** reports every problem in an uploaded file at once.

## Limitations and next steps

- The dataset is a public sample, not live company data; there is no drift monitoring.
- Recommendations are rule-based and ranked by model impact; they are not causal. An A/B test
  would be needed to measure which retention offers actually work.
- The decision threshold maximises F1. A real deployment should choose it from the relative cost
  of a lost customer versus a wasted retention offer.
- Next: split into a FastAPI prediction service with a React frontend, store scores in
  PostgreSQL, add model monitoring and scheduled retraining.

## Data source

IBM Telco Customer Churn sample dataset, from the IBM `telco-customer-churn-on-icp4d` repository.
