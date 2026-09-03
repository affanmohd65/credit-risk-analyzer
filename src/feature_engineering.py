from __future__ import annotations

import numpy as np
import pandas as pd


ID_COLUMNS = ["account_id"]
TARGET_COLUMN = "default"


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create ratios using only application-time attributes."""
    output = frame.copy()
    bill_columns = [f"bill_amount_{index}" for index in range(1, 7)]
    payment_columns = [f"payment_amount_{index}" for index in range(1, 7)]
    status_columns = ["pay_status_0", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6"]
    output["average_bill_amount"] = output[bill_columns].mean(axis=1)
    output["average_payment_amount"] = output[payment_columns].mean(axis=1)
    output["payment_to_bill_ratio"] = output["average_payment_amount"] / output["average_bill_amount"].abs().clip(lower=1)
    output["credit_utilization_proxy"] = output["bill_amount_1"].clip(lower=0) / output["credit_limit"].clip(lower=1)
    output["average_repayment_delay"] = output[status_columns].mean(axis=1)
    output["late_payment_months"] = (output[status_columns] > 0).sum(axis=1)
    output["balance_trend"] = output["bill_amount_1"] - output["bill_amount_6"]
    output["financial_stress_score"] = np.clip(output["credit_utilization_proxy"] * (1 + output["average_repayment_delay"].clip(lower=0)), 0, 10)
    return output


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(frame).drop(columns=ID_COLUMNS + [TARGET_COLUMN], errors="ignore")