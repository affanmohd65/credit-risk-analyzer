from __future__ import annotations

from typing import Any

import pandas as pd


def global_feature_importance(model: Any) -> list[dict[str, float | str]]:
    """Return native tree importances when supported; SHAP is used by the dashboard if installed."""
    estimator = getattr(model, "estimator", model)
    if not hasattr(estimator, "feature_importances_"):
        return []
    names = estimator.named_steps["preprocessor"].get_feature_names_out()
    values = estimator.named_steps["classifier"].feature_importances_
    return sorted(({"feature": str(name), "importance": float(value)} for name, value in zip(names, values)), key=lambda row: row["importance"], reverse=True)[:20]


def individual_risk_factors(application: dict[str, Any]) -> dict[str, list[str]]:
    increases, reduces = [], []
    statuses = [application.get(f"pay_status_{month}", 0) for month in [0, 2, 3, 4, 5, 6]]
    if max(statuses) > 0: increases.append("Recent delayed payment status")
    if application.get("bill_amount_1", 0) > application.get("credit_limit", 1) * .70: increases.append("High current balance relative to credit limit")
    if application.get("payment_amount_1", 0) < application.get("bill_amount_1", 0) * .05: increases.append("Low recent payment relative to statement balance")
    if max(statuses) <= 0: reduces.append("No recent recorded payment delays")
    if application.get("bill_amount_1", 0) < application.get("credit_limit", 1) * .30: reduces.append("Lower current balance relative to credit limit")
    if application.get("payment_amount_1", 0) >= application.get("bill_amount_1", 0) * .25: reduces.append("Meaningful recent payment")
    return {"risk_factors": increases, "positive_factors": reduces}