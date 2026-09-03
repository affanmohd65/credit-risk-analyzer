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
    if application.get("bureau_score", 900) < 650: increases.append("Lower bureau score")
    if application.get("foir", 0) > .50: increases.append("High fixed-obligation-to-income ratio")
    if application.get("overdue_accounts", 0) > 0: increases.append("Existing overdue accounts")
    if application.get("credit_inquiries_6m", 0) >= 4: increases.append("High recent credit inquiry count")
    if application.get("employment_type") == "Contract": increases.append("Contract employment profile")
    if application.get("bureau_score", 0) >= 750: reduces.append("Strong bureau score")
    if application.get("foir", 1) < .35: reduces.append("Comfortable obligation-to-income ratio")
    if application.get("bank_balance_inr", 0) >= application.get("loan_amount_inr", 1) * .50: reduces.append("Strong bank-balance buffer")
    return {"risk_factors": increases, "positive_factors": reduces}