# ChurnGuard: Interview Prep

The interviewer said: *"Use AI tools and whatever you want, but the project should be built and
fully working."* Using AI to build it is fine and expected. What they will test is whether **you
understand it**. This guide is built around that. Read the code in the order below, run it, break
it, and then use the answers here as a starting point in your own words.

## 1. Study plan

**Sunday (today): get it running and read the core.**
1. Run it: `pip install -r requirements-dev.txt`, `python -m churnguard.train`, `streamlit run app.py`, `pytest`.
2. Read in this order, about 20 minutes each: `config.py`, `data.py`, `features.py`, `train.py`.

**Monday: read the rest and make a change yourself.**
3. Read `explain.py`, `predict.py`, `recommend.py`, then `ui/single.py`.
4. Make one small change on your own and see it work. Ideas: change the risk bands in
   `config.py`, add a new recommendation rule in `recommend.py`, add a feature in `features.py`
   and retrain. Being able to say "I changed X and saw Y" is the most convincing thing you can do.
5. Push to GitHub and deploy to Streamlit Community Cloud so you have a live link.

**Tuesday morning:** rehearse the 60-second pitch and the demo out loud, twice.

## 2. The 60-second pitch

> "ChurnGuard predicts which telecom customers are likely to cancel and, more importantly,
> explains why, so a retention team knows what to do. I used the IBM Telco dataset of about
> 7,000 customers. I compared logistic regression, random forest and XGBoost with 5-fold
> cross-validation, tuned XGBoost, and tuned the decision threshold, because only about a quarter
> of customers churn and the default threshold misses too many of them. At the chosen threshold
> it catches 73% of churners with 56% precision on held-out data, ROC-AUC about 0.85. Every
> prediction is explained with SHAP, grouped by customer attribute, and a rule layer turns the
> top drivers into retention actions. It's a Streamlit app with single-customer and batch
> scoring, 64 automated tests, a Docker image and a CI pipeline."

## 3. Demo script (2 minutes)

1. **Overview:** "High-risk customers and the monthly revenue tied to them. Month-to-month contracts and short tenure clearly churn more."
2. **Predict a customer:** the high-risk example is loaded. Point at the gauge, the risk badge, then the SHAP chart: "Red bars push risk up: month-to-month contract, tenure of 2 months, fiber optic."
3. Click **Loyal low-risk example**: "Two-year contract and long tenure push risk down, and no action is recommended."
4. Change the contract on the high-risk example to Two year and watch the probability drop: "That's a what-if."
5. **Batch scoring:** use the sample data, filter High, show the call list and the download.
6. **Model performance:** model comparison, then say the honest line: "The three models are close; I picked XGBoost because it had the best cross-validated score and exact SHAP support."

## 4. Numbers to know

| Fact | Value |
| --- | --- |
| Customers / raw features / after encoding | 7,043 / 19 / 49 |
| Churn rate | 26.5% |
| Split | 80% train (5,634) / 20% test (1,409), stratified |
| XGBoost CV ROC-AUC | 0.851 (+/- 0.012) |
| XGBoost test ROC-AUC / PR-AUC | 0.847 / 0.665 |
| Logistic Regression test ROC-AUC | 0.846 |
| Random Forest test ROC-AUC | 0.841 |
| Threshold | 0.35 (tuned for F1 on out-of-fold predictions) |
| At 0.35: recall / precision / F1 | 73% / 56% / 0.63 |
| At 0.5: recall / precision | 52% / 67% |
| Risk bands | Low below 30%, Medium 30-60%, High 60%+ |
| Top SHAP drivers | contract type, tenure, internet service, online security, monthly charges |
| Tests | 64 |

## 5. Likely questions and answers

### Machine learning

**Why did you pick XGBoost?**
It had the best cross-validated ROC-AUC after tuning (0.851 vs 0.847 for logistic regression).
Honestly, the models are close on this dataset, and the differences are smaller than the
fold-to-fold noise. XGBoost also captures interactions, like fiber optic on a month-to-month
contract, and SHAP's TreeExplainer gives exact explanations for it. In production I would also
consider logistic regression because it is simpler to maintain.

**How did you handle class imbalance?**
Churn is 26.5%, so it is moderately imbalanced. I used stratified splits and folds so every fold
has the same churn rate, evaluated with PR-AUC as well as ROC-AUC, and tuned the decision
threshold. I did not use SMOTE or class weights because they distort predicted probabilities, and
I want the probabilities to be usable as risk levels.

**Why tune the threshold? Why 0.35?**
At the default 0.5 the model finds only 52% of churners. Missing a churner usually costs more than
a wasted discount, so I picked the threshold that maximises F1 on out-of-fold predictions from the
training data: 0.35. That raises recall to 73% at 56% precision. A real deployment should choose
the threshold from actual costs: lost customer lifetime value versus the cost of a retention offer.

**Which metrics matter and why?**
Accuracy is misleading because 73.5% of customers don't churn, so "predict nobody churns" already
scores 73.5%. I report ROC-AUC (how well customers are ranked), PR-AUC (better when the positive class
is rarer), and precision and recall at the chosen threshold.

**How did you avoid data leakage?**
Feature engineering and encoding live inside one scikit-learn Pipeline, so they are fitted only on
training folds. The test set was never used for model selection or threshold tuning; those use
cross-validation on the training split. `customerID` is not a feature. Hyperparameter tuning used
the training split only.

**Is your CV score optimistic?**
Slightly. I tuned XGBoost on the same training folds I report, so 0.851 is a little flattering.
The test ROC-AUC of 0.847 on unseen customers is the number to trust.

**Why refit on all the data at the end?**
After validation, the final model can use the extra 20% of data. The reported metrics come from
the held-out evaluation before the refit.

**What does SHAP tell you that feature importance doesn't?**
Global feature importance says which features matter on average. SHAP gives a per-customer
breakdown: how much each attribute pushed this customer's risk up or down. That is what a
retention agent needs. I grouped one-hot columns back to their attribute so the reason reads
"Contract type: Month-to-month" instead of three separate dummy columns. A test confirms the SHAP
values plus the base value reproduce the model's output.

**What are SHAP values measured in?**
Log-odds of churn, not percentage points. Positive pushes risk up, negative pushes it down.

**Is the model calibrated?**
I did not explicitly calibrate it. Brier score is 0.135, and without class weighting or resampling
the probabilities are reasonable, but for a production system I would check a calibration curve
and add Platt or isotonic calibration if needed.

**What features did you engineer?**
Number of add-on services, auto-pay flag, new-customer flag (tenure of 6 months or less) and the
gap between this month's charge and the customer's average monthly spend. I then tested whether
they help by running 5-fold CV with and without them on three different random seeds: the
ROC-AUC difference was within +/- 0.0003, so there is no measurable accuracy gain for XGBoost,
which can already learn those combinations from the raw columns. I kept them because they make
explanations more readable (for example "Number of add-on services" as a reason), but I would not
claim they improved the model. (I did not test them on logistic regression.)

**What would you do to improve the model?**
Real behavioural data (usage, support calls, late payments), time-based validation instead of a
random split, calibration, a cost-sensitive threshold, and monitoring for drift.

### Engineering

**Why Streamlit?**
It is the fastest way to ship a data application in Python. For a larger product I would split it
into a FastAPI prediction service and a React frontend, which is what I am learning now.

**How is the code organised?**
A `churnguard` package with one responsibility per module: config, data, features, training,
explanation, prediction, recommendations and a `ui` package with one file per page. `config.py`
is the single source of truth for the schema and thresholds.

**How do you validate inputs?**
`validate_customers` checks required columns, allowed category values, numeric ranges and
non-numeric values, and reports every problem at once. The same function guards the single form
and CSV uploads.

**How is it tested?**
64 pytest tests: data cleaning, feature engineering, model selection logic, training outputs,
inference, validation, recommendations, and headless Streamlit tests that load every page and
click through it. CI runs lint, trains the model, runs the tests and builds the Docker image.

**How would you deploy and scale it?**
Docker image to Render, Hugging Face Spaces or any container platform. To scale, move model
serving behind a FastAPI service and keep Streamlit as a thin client. Batch scoring would run as
a scheduled job writing to a database.

**What happens when the model file is missing or incompatible?**
The app catches the load error and trains a fresh model automatically (about 30 seconds). Library
versions are pinned in `requirements.txt` so this is rare.

**How would you monitor it in production?**
Track the input distribution (drift), the score distribution, and, once real outcomes arrive,
actual churn rate by risk band versus predicted. Retrain on a schedule or when drift is detected.

### Your resume and honesty

**Did you use AI to build this?**
Yes. Say it directly, because your resume already does: "I used AI coding tools to move faster,
then read, ran, debugged and tested the output myself." Then prove it: walk through a file, explain
a decision, and describe a change you made.

**What was the hardest part?**
Pick a real one from your own experience while building and running it. Good honest candidates:
choosing between models that perform almost identically, deciding the threshold, or turning SHAP
output into something readable.

**What would you do differently?**
Use real company data with time-based validation, a cost-based threshold, and measure whether the
retention actions actually reduce churn with an A/B test.

**How do the recommendations work? Are they causal?**
No. They are business rules (for example, offer a longer contract to month-to-month customers),
ranked by how much the model says that attribute raises this customer's risk. The model shows
association, not causation, so I would A/B test each offer before trusting it.

## 6. Things to be upfront about

- The dataset is a public IBM sample, not data from a real company.
- The three models perform within noise of each other on this dataset.
- The engineered features did not measurably improve XGBoost's accuracy (tested, see above).
- The 0.35 threshold maximises F1, not business profit.
- The recommendations are rules, not learned from outcomes.
- No live drift monitoring exists.
- Only claim Docker and CI after you have run `docker build` and seen the GitHub Action pass.

## 7. Optional resume bullets

Your current ChurnGuard bullets are accurate. Once you have run everything yourself, you could
make them more concrete:

- Built an end-to-end churn prediction system on 7,043 telecom customers: compared Logistic
  Regression, Random Forest and XGBoost with 5-fold cross-validation, tuned XGBoost and the
  decision threshold, reaching 0.85 ROC-AUC and 73% recall on held-out data.
- Explained each prediction with SHAP and built a Streamlit app with single-customer and batch
  scoring, risk levels and ranked retention recommendations.
- Added 64 pytest tests, a Dockerfile and a GitHub Actions CI pipeline.
