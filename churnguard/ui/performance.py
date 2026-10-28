"""Model quality: how the model was built and how well it works on customers it never saw."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from churnguard.ui.common import ACCENT, UP_COLOR, get_metrics, get_predictor, style_figure
from churnguard.ui.theme import band, page_head


def _comparison_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for name, res in metrics["models"].items():
        test = res["test"]
        rows.append(
            {
                "Model": name + ("  (selected)" if name == metrics["best_model"] else ""),
                "CV ROC-AUC": f"{res['cv_roc_auc_mean']:.3f} +/- {res['cv_roc_auc_std']:.3f}",
                "Test ROC-AUC": round(test["roc_auc"], 3),
                "Test PR-AUC": round(test["pr_auc"], 3),
                "Precision @0.5": round(test["precision"], 3),
                "Recall @0.5": round(test["recall"], 3),
                "F1 @0.5": round(test["f1"], 3),
            }
        )
    return pd.DataFrame(rows)


def _roc_figure(metrics: dict) -> go.Figure:
    roc = metrics["roc_curve"]
    auc = metrics["best_at_threshold"]["roc_auc"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=roc["fpr"],
            y=roc["tpr"],
            mode="lines",
            name=f"Model (AUC {auc:.3f})",
            line={"color": ACCENT, "width": 3.5},
            fill="tozeroy",
            fillcolor="rgba(18,120,138,0.14)",
        )
    )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line={"color": "#8AA0A6", "dash": "dash"})
    )
    fig.update_xaxes(title="False alarms (share of stayers flagged)")
    fig.update_yaxes(title="Churners caught")
    fig.update_layout(legend={"orientation": "h", "y": -0.25})
    return style_figure(fig, height=360, title="ROC curve (held-out test set)")


def _confusion_figure(metrics: dict) -> go.Figure:
    (tn, fp), (fn, tp) = metrics["best_at_threshold"]["confusion_matrix"]
    fig = px.imshow(
        [[tn, fp], [fn, tp]],
        x=["Predicted stay", "Predicted churn"],
        y=["Actually stayed", "Actually churned"],
        text_auto=True,
        color_continuous_scale=["#EAF2EE", ACCENT],
    )
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textfont={"size": 22})
    return style_figure(fig, height=360, title="Confusion matrix (held-out test set)")


def _waffle(share: float, coral: bool = False) -> str:
    """100 dots, `share` of them filled: 'out of 100 customers, this many'."""
    on = round(share * 100)
    dots = "".join(f'<i class="{"on pop" if i < on else ""}" style="--n:{i}"></i>' for i in range(100))
    return f'<div class="cg-waffle{" coral" if coral else ""}" role="img" aria-label="{on} out of 100">{dots}</div>'


def _in_plain_words(best: dict) -> None:
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True, key="card-recall"):
        st.html(
            f'<div class="cg-play"><h4>{best["recall"]:.0%} of leavers are caught</h4>'
            "<p>Of every 100 customers who really leave, this many are flagged before they go.</p></div>"
            + _waffle(best["recall"])
        )
    with right, st.container(border=True, key="card-precision"):
        st.html(
            f'<div class="cg-play"><h4>{best["precision"]:.0%} of flags are right</h4>'
            "<p>Of every 100 customers the model flags, this many really were going to leave.</p></div>"
            + _waffle(best["precision"], coral=True)
        )


def render() -> None:
    metrics = get_metrics()
    predictor = get_predictor()
    best = metrics["best_at_threshold"]
    data = metrics["data"]

    with band("head", "tide", first=True):
        page_head(
            "Model quality",
            "How the model was built, compared, and how well it works on customers it has never seen.",
        )
    with band("work", "paper"):
        cols = st.columns(4)
        cols[0].metric("Selected model", metrics["best_model"])
        cols[1].metric("Test ROC-AUC", f"{best['roc_auc']:.3f}")
        cols[2].metric("Recall (churners caught)", f"{best['recall']:.0%}")
        cols[3].metric("Precision", f"{best['precision']:.0%}")
        st.caption(
            f"Checked on {data['test_rows']:,} customers never used for training or model selection, "
            f"at a decision threshold of {metrics['threshold']:.2f}."
        )
        _in_plain_words(best)

        tab_compare, tab_eval, tab_drivers, tab_method = st.tabs(
            ["Model comparison", "Evaluation", "What drives churn", "Method"]
        )

        with tab_compare:
            st.dataframe(_comparison_table(metrics), hide_index=True, width="stretch")
            st.info(metrics.get("selection_reason", ""))
            st.caption(
                "CV = 5-fold stratified cross-validation on the training split. ROC-AUC measures how "
                "well customers are ranked by risk; PR-AUC is more informative because only about a "
                "quarter of customers churn."
            )

        with tab_eval:
            left, right = st.columns(2)
            left.plotly_chart(_roc_figure(metrics), width="stretch", config={"displayModeBar": False})
            right.plotly_chart(_confusion_figure(metrics), width="stretch", config={"displayModeBar": False})
            st.write(
                f"At the chosen threshold of **{metrics['threshold']:.2f}** the model finds "
                f"**{best['recall']:.0%}** of customers who actually churn, and "
                f"**{best['precision']:.0%}** of the customers it flags really do churn. "
                "The default 0.5 threshold would miss many more churners."
            )

        with tab_drivers:
            imp = pd.DataFrame(metrics["feature_importance"]).sort_values("mean_abs_shap")
            fig = px.bar(imp, x="mean_abs_shap", y="feature", orientation="h")
            fig.update_traces(marker_color=UP_COLOR)
            fig.update_xaxes(title="Average impact on the prediction (mean |SHAP|)")
            fig.update_yaxes(title=None)
            st.plotly_chart(
                style_figure(fig, height=480, title="Most important customer attributes"),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption("Computed with SHAP across a sample of customers, with one-hot columns grouped by attribute.")

        with tab_method:
            st.markdown(
                f"""
                **Data.** IBM Telco Customer Churn sample: {data["rows"]:,} customers, {data["features_raw"]}
                raw attributes (demographics, services, account and billing), {data["churn_rate"]:.1%} churn rate.

                **Preprocessing.** Cleaning (blank `TotalCharges` for brand-new customers), one-hot encoding,
                and four engineered features: number of add-on services, auto-pay flag, new-customer flag and
                the gap between this month's bill and the customer's average monthly spend
                ({data["features_encoded"]} model inputs after encoding).

                **Class imbalance.** Stratified train/test split and cross-validation, evaluation with PR-AUC,
                and a decision threshold tuned for F1 on out-of-fold predictions.

                **Selection.** Logistic Regression, Random Forest and XGBoost are compared with 5-fold
                cross-validation; XGBoost is tuned with a randomised search on the training split only.

                **Deployment.** The selected pipeline is refit on all customers. Trained {predictor.trained_at}.

                **Explainability.** SHAP values explain every individual prediction.
                """
            )
