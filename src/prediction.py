from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import RISK_RULES, RiskRules
from src.feature_engineering import model_features


def risk_category(probability: float, rules: RiskRules = RISK_RULES) -> str:
    if probability < rules.low_max:
        return "LOW"
    if probability < rules.medium_max:
        return "MEDIUM"
    if probability < rules.high_max:
        return "HIGH"
    return "VERY_HIGH"


def loan_decision(category: str) -> str:
    return {"LOW": "APPROVE", "MEDIUM": "MANUAL_REVIEW", "HIGH": "MANUAL_REVIEW", "VERY_HIGH": "REJECT"}[category]


def predict_records(model: Any, applications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = model_features(pd.DataFrame(applications))
    probabilities = model.predict_proba(features)[:, 1]
    results = []
    for probability in probabilities:
        probability = float(probability)
        category = risk_category(probability)
        results.append({"default_probability": round(probability, 4), "risk_score": round(probability * 100),
                "risk_category": category, "decision": loan_decision(category)})
    return results