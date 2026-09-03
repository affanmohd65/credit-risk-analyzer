from __future__ import annotations

import numpy as np
import pandas as pd


ID_COLUMNS = ["application_id"]
TARGET_COLUMN = "default"


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create ratios using only application-time attributes."""
    output = frame.copy()
    monthly_income = output["annual_income_inr"].clip(lower=1) / 12
    output["emi_to_income_ratio"] = output["proposed_emi_inr"] / monthly_income
    output["existing_emi_to_income_ratio"] = output["existing_emi_inr"] / monthly_income
    output["total_emi_to_income_ratio"] = (output["existing_emi_inr"] + output["proposed_emi_inr"]) / monthly_income
    output["loan_to_income_ratio_derived"] = output["loan_amount_inr"] / output["annual_income_inr"].clip(lower=1)
    output["balance_to_loan_ratio"] = output["bank_balance_inr"] / output["loan_amount_inr"].clip(lower=1)
    output["bureau_risk_gap"] = np.clip((750 - output["bureau_score"]) / 450, 0, 1)
    output["financial_stress_score"] = np.clip(output["foir"] * (1 + output["overdue_accounts"]) * (1 + output["credit_inquiries_6m"] / 10), 0, 10)
    return output


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(frame).drop(columns=ID_COLUMNS + [TARGET_COLUMN], errors="ignore")