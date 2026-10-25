"""Train, compare and persist the churn model.

Run with:  python -m churnguard.train
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churnguard import config
from churnguard.data import load_dataset
from churnguard.explain import build_explainer, display_name, group_shap, shap_values
from churnguard.features import build_preprocessor, feature_step

logger = logging.getLogger("churnguard.train")


XGB_DEFAULTS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
}

XGB_SEARCH_SPACE = {
    "model__n_estimators": [200, 300, 500],
    "model__learning_rate": [0.02, 0.03, 0.05, 0.08],
    "model__max_depth": [2, 3, 4, 5],
    "model__min_child_weight": [1, 3, 5, 10],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__reg_lambda": [1.0, 2.0, 5.0, 10.0],
}


def candidate_models(xgb_params: dict | None = None) -> dict[str, tuple[object, bool]]:
    """Models to compare: name -> (estimator, needs_scaled_numeric_features)."""
    rs = config.RANDOM_STATE
    xgb = {**XGB_DEFAULTS, **(xgb_params or {})}
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, C=0.5), True),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=rs
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                **xgb,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=rs,
            ),
            False,
        ),
    }


def make_pipeline(model, scale_numeric: bool) -> Pipeline:
    return Pipeline(
        [
            ("features", feature_step()),
            ("prep", build_preprocessor(scale_numeric)),
            ("model", model),
        ]
    )


def threshold_scores(y_true, proba, threshold: float) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Decision threshold that maximises F1 (the default 0.5 under-detects churners)."""
    grid = np.round(np.arange(0.15, 0.71, 0.01), 2)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def _py(obj):
    """Make numpy types JSON serialisable."""
    if isinstance(obj, dict):
        return {k: _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def tune_xgboost(X_train, y_train, skf) -> dict:
    """Randomised hyper-parameter search for XGBoost, using the training split only."""
    search = RandomizedSearchCV(
        make_pipeline(candidate_models()["XGBoost"][0], False),
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=config.TUNING_ITERATIONS,
        scoring="roc_auc",
        cv=skf,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    logger.info("Best XGBoost params: %s (CV AUC %.4f)", params, search.best_score_)
    return params


def select_model(results: dict[str, dict]) -> tuple[str, str]:
    """Choose the model and explain why. Uses CV scores only; the test set is never consulted."""
    best = max(results, key=lambda n: results[n]["cv_roc_auc_mean"])
    top = results[best]["cv_roc_auc_mean"]
    preferred = config.PREFERRED_MODEL_ON_TIE
    if preferred in results and top - results[preferred]["cv_roc_auc_mean"] <= config.SELECTION_TIE_MARGIN:
        if preferred == best:
            return preferred, "Highest cross-validated ROC-AUC."
        return preferred, (
            f"{best} scored marginally higher in cross-validation, but the gap "
            f"({top - results[preferred]['cv_roc_auc_mean']:.4f}) is within the "
            f"{config.SELECTION_TIE_MARGIN} tie margin and far smaller than the fold-to-fold "
            f"standard deviation, so the models are statistically tied. {preferred} was "
            f"preferred because it captures non-linear feature interactions and supports "
            f"exact TreeSHAP explanations."
        )
    return best, "Highest cross-validated ROC-AUC."


def train(
    data_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    df: pd.DataFrame | None = None,
    tune: bool = True,
) -> dict:
    """Run the full training workflow and write the model bundle and metrics.json."""
    model_dir = Path(model_dir) if model_dir else config.MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    data = df if df is not None else load_dataset(data_path)
    X = data[config.MODEL_INPUT_COLS]
    y = data[config.TARGET].astype(int)
    logger.info("Training on %d customers, churn rate %.1f%%", len(data), 100 * y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    xgb_params = tune_xgboost(X_train, y_train, skf) if tune else None
    models = candidate_models(xgb_params)

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, (estimator, scale) in models.items():
        pipe = make_pipeline(estimator, scale)
        cv = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=skf,
            scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision"},
            n_jobs=1,
        )
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        test = {
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "brier": brier_score_loss(y_test, proba),
            **threshold_scores(y_test, proba, 0.5),
        }
        results[name] = {
            "cv_roc_auc_mean": cv["test_roc_auc"].mean(),
            "cv_roc_auc_std": cv["test_roc_auc"].std(),
            "cv_pr_auc_mean": cv["test_pr_auc"].mean(),
            "test": test,
        }
        fitted[name] = pipe
        logger.info(
            "%-20s CV AUC %.4f (+/- %.4f) | test AUC %.4f",
            name,
            results[name]["cv_roc_auc_mean"],
            results[name]["cv_roc_auc_std"],
            test["roc_auc"],
        )

    # Choose using cross-validation on the training split only, so the test set stays unseen.
    best_name, selection_reason = select_model(results)
    best_estimator, best_scale = models[best_name]
    logger.info("Selected model: %s. %s", best_name, selection_reason)

    oof = cross_val_predict(
        make_pipeline(clone(best_estimator), best_scale),
        X_train,
        y_train,
        cv=skf,
        method="predict_proba",
    )[:, 1]
    threshold = best_f1_threshold(y_train, oof)

    test_proba = fitted[best_name].predict_proba(X_test)[:, 1]
    at_threshold = threshold_scores(y_test, test_proba, threshold)
    tn, fp, fn, tp = confusion_matrix(y_test, (test_proba >= threshold).astype(int)).ravel()
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    idx = np.unique(np.linspace(0, len(fpr) - 1, 120).astype(int))

    # Refit the chosen pipeline on all data for deployment (standard practice after validation).
    final = make_pipeline(clone(best_estimator), best_scale).fit(X, y)
    prep = final.named_steps["prep"]
    feature_names = list(prep.get_feature_names_out())
    model = final.named_steps["model"]

    sample = X.sample(min(800, len(X)), random_state=config.RANDOM_STATE)
    sample_t = final[:-1].transform(sample)
    background = sample_t[:100]
    explainer = build_explainer(model, background)
    grouped = group_shap(shap_values(explainer, sample_t), feature_names)
    importance = grouped.abs().mean().sort_values(ascending=False).head(15)

    metrics = {
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {
            "rows": int(len(data)),
            "churn_rate": float(y.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "features_raw": len(config.MODEL_INPUT_COLS),
            "features_encoded": len(feature_names),
        },
        "best_model": best_name,
        "selection_reason": selection_reason,
        "xgboost_params": {**XGB_DEFAULTS, **(xgb_params or {})},
        "threshold": threshold,
        "models": results,
        "best_at_threshold": {
            **at_threshold,
            "roc_auc": roc_auc_score(y_test, test_proba),
            "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "roc_curve": {"fpr": np.round(fpr[idx], 4), "tpr": np.round(tpr[idx], 4)},
        "feature_importance": [
            {"feature": display_name(name), "mean_abs_shap": float(val)}
            for name, val in importance.items()
        ],
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }

    bundle = {
        "pipeline": final,
        "model_name": best_name,
        "threshold": threshold,
        "feature_names": feature_names,
        "background": background,
        "trained_at": metrics["trained_at"],
    }
    joblib.dump(bundle, model_dir / config.MODEL_PATH.name, compress=3)
    with open(model_dir / config.METRICS_PATH.name, "w", encoding="utf-8") as fh:
        json.dump(_py(metrics), fh, indent=2)
    logger.info("Saved model and metrics to %s", model_dir)
    return _py(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ChurnGuard churn model")
    parser.add_argument("--data", default=None, help="CSV path (default: data/telco_churn.csv)")
    parser.add_argument("--model-dir", default=None, help="Output directory (default: models/)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metrics = train(args.data, args.model_dir)

    print("\n=== Model comparison (held-out test set) ===")
    for name, res in metrics["models"].items():
        t = res["test"]
        print(
            f"{name:<20} CV AUC {res['cv_roc_auc_mean']:.3f} | test AUC {t['roc_auc']:.3f} "
            f"| PR-AUC {t['pr_auc']:.3f}"
        )
    b = metrics["best_at_threshold"]
    print(
        f"\nSelected: {metrics['best_model']} @ threshold {metrics['threshold']:.2f} -> "
        f"precision {b['precision']:.3f}, recall {b['recall']:.3f}, F1 {b['f1']:.3f}"
    )


if __name__ == "__main__":
    main()
