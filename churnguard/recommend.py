"""Retention recommendations: business rules ranked by what the model says drives churn."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    keys: tuple[str, ...]  # SHAP feature keys that justify this action
    title: str
    action: str
    applies: Callable[[Mapping], bool]
    # What the action changes in the customer's profile, so the app can ask the model
    # "what would the risk be then?". None for actions the model has no input for.
    changes: Callable[[Mapping], dict] | None = None
    what: str = ""


RULES: tuple[Rule, ...] = (
    Rule(
        ("Contract",),
        "Offer a longer-term contract",
        "Offer a 10-15% discount or a free add-on in return for a 12-month contract. "
        "Month-to-month customers can leave at any time, so this is the highest-leverage action.",
        lambda c: c["Contract"] == "Month-to-month",
        changes=lambda c: {"Contract": "One year"},
        what="Moves to a one-year contract",
    ),
    Rule(
        ("TechSupport", "OnlineSecurity", "num_addons"),
        "Bundle support and security",
        "Give 3 months of free tech support and online security. Customers with these services "
        "are more engaged and churn less.",
        lambda c: c["InternetService"] != "No" and (c["TechSupport"] == "No" or c["OnlineSecurity"] == "No"),
        changes=lambda c: {"TechSupport": "Yes", "OnlineSecurity": "Yes"},
        what="Adds tech support and online security",
    ),
    Rule(
        ("PaymentMethod", "is_autopay"),
        "Move to automatic payment",
        "Offer a one-time bill credit for switching from manual payment to card or bank auto-pay. "
        "Removing payment friction lowers churn.",
        lambda c: c["PaymentMethod"] in ("Electronic check", "Mailed check"),
        changes=lambda c: {"PaymentMethod": "Bank transfer (automatic)"},
        what="Pays by automatic bank transfer",
    ),
    Rule(
        ("MonthlyCharges", "spend_gap", "InternetService"),
        "Review pricing and plan fit",
        "Run a plan right-sizing call: offer a cheaper plan that matches actual usage, or a "
        "time-limited price lock if the customer is price sensitive.",
        lambda c: float(c["MonthlyCharges"]) >= 70,
        changes=lambda c: {"MonthlyCharges": round(float(c["MonthlyCharges"]) * 0.85, 2)},
        what="Monthly price 15% lower",
    ),
    Rule(
        ("InternetService",),
        "Run a fibre quality check",
        "Fibre customers churn more often. Check for speed or outage issues and offer a free "
        "technician visit before the customer complains.",
        lambda c: c["InternetService"] == "Fiber optic",
    ),
    Rule(
        ("tenure", "is_new_customer"),
        "Strengthen early-life onboarding",
        "Schedule a welcome call within the first 90 days, confirm the service works as sold, "
        "and offer a loyalty reward at month 6.",
        lambda c: float(c["tenure"]) <= 12,
    ),
    Rule(
        ("SeniorCitizen",),
        "Offer assisted support",
        "Provide a dedicated support line or in-store assistance, and simplify billing.",
        lambda c: int(c["SeniorCitizen"]) == 1,
    ),
)


def recommend(customer: Mapping, risk: str, drivers: list[dict], max_items: int = 4) -> list[dict]:
    """Return ranked retention actions for one customer.

    Actions the model itself flagged as risk drivers (positive SHAP value for the matching
    feature) are ranked first, strongest driver first. Remaining applicable actions follow.
    """
    if risk == "Low":
        return [
            {
                "title": "No retention action needed",
                "action": "Keep the customer in standard communications and re-score if their "
                "plan, billing or usage changes.",
                "reason": "Churn risk is low.",
                "model_flagged": False,
                "impact": 0.0,
                "changes": {},
                "what": "",
            }
        ]

    positive = {d["key"]: d["shap"] for d in drivers if d["shap"] > 0}
    scored: list[dict] = []
    for rule in RULES:
        if not rule.applies(customer):
            continue
        impact = max((positive.get(k, 0.0) for k in rule.keys), default=0.0)
        flagged = impact > 0
        scored.append(
            {
                "title": rule.title,
                "action": rule.action,
                "reason": "The model flags this customer attribute as raising churn risk."
                if flagged
                else "Applies to this customer's profile.",
                "model_flagged": flagged,
                "impact": impact,
                "changes": rule.changes(customer) if rule.changes else {},
                "what": rule.what,
            }
        )

    scored.sort(key=lambda r: r["impact"], reverse=True)
    if not scored:
        scored.append(
            {
                "title": "Proactive check-in call",
                "action": "Have an account manager call to understand satisfaction and offer a "
                "tailored retention offer.",
                "reason": f"{risk} churn risk with no specific rule triggered.",
                "model_flagged": False,
                "impact": 0.0,
                "changes": {},
                "what": "",
            }
        )
    return scored[:max_items]
