import json

import joblib

from churnguard import config
from churnguard.train import best_f1_threshold, select_model


def _results(**cv):
    return {name: {"cv_roc_auc_mean": score} for name, score in cv.items()}


def test_training_writes_model_and_metrics(trained):
    metrics, model_dir = trained
    assert (model_dir / "churn_model.joblib").exists()
    on_disk = json.loads((model_dir / "metrics.json").read_text())
    assert on_disk["best_model"] == metrics["best_model"]


def test_all_three_models_are_compared(trained):
    metrics, _ = trained
    assert set(metrics["models"]) == {"Logistic Regression", "Random Forest", "XGBoost"}


def test_model_beats_chance_on_held_out_data(trained):
    metrics, _ = trained
    assert metrics["best_at_threshold"]["roc_auc"] > 0.75


def test_threshold_is_in_search_range_and_improves_recall_over_default(trained):
    metrics, _ = trained
    assert 0.15 <= metrics["threshold"] <= 0.70
    default_recall = metrics["models"][metrics["best_model"]]["test"]["recall"]
    assert metrics["best_at_threshold"]["recall"] >= default_recall


def test_confusion_matrix_accounts_for_every_test_row(trained):
    metrics, _ = trained
    total = sum(sum(row) for row in metrics["best_at_threshold"]["confusion_matrix"])
    assert total == metrics["data"]["test_rows"]


def test_feature_importance_is_sorted_and_named(trained):
    metrics, _ = trained
    values = [f["mean_abs_shap"] for f in metrics["feature_importance"]]
    assert values == sorted(values, reverse=True)
    assert all(f["feature"] for f in metrics["feature_importance"])


def test_saved_bundle_contains_what_inference_needs(trained):
    _, model_dir = trained
    bundle = joblib.load(model_dir / "churn_model.joblib")
    assert {"pipeline", "model_name", "threshold", "feature_names", "background"} <= set(bundle)


def test_best_f1_threshold_finds_the_separating_point():
    import numpy as np

    y = np.array([0] * 50 + [1] * 50)
    proba = np.array([0.1] * 50 + [0.6] * 50)
    assert 0.1 < best_f1_threshold(y, proba) <= 0.6


def test_select_model_prefers_xgboost_when_scores_are_tied():
    name, reason = select_model(
        _results(**{"Logistic Regression": 0.8474, "Random Forest": 0.84, "XGBoost": 0.8458})
    )
    assert name == "XGBoost"
    assert "tied" in reason


def test_select_model_picks_clear_winner_over_preferred():
    name, _ = select_model(
        _results(**{"Logistic Regression": 0.88, "Random Forest": 0.84, "XGBoost": 0.84})
    )
    assert name == "Logistic Regression"


def test_select_model_when_preferred_is_already_best():
    name, reason = select_model(
        _results(**{"Logistic Regression": 0.84, "Random Forest": 0.84, "XGBoost": 0.85})
    )
    assert name == "XGBoost"
    assert reason == "Highest cross-validated ROC-AUC."


def test_tie_margin_is_small_relative_to_a_meaningful_difference():
    assert config.SELECTION_TIE_MARGIN <= 0.01
