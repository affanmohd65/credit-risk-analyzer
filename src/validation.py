from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass
class ValidationReport:
    rows: int
    duplicates: int
    missing_values: dict[str, int]
    invalid_counts: dict[str, int]
    default_rate: float
    outlier_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_data(frame: pd.DataFrame) -> ValidationReport:
    checks = {
        "invalid_age": int(((frame["age"] < 18) | (frame["age"] > 100)).sum()),
        "nonpositive_credit_limit": int((frame["credit_limit"] <= 0).sum()),
        "invalid_payment_status": int(((frame[["pay_status_0", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6"]] < -2) | (frame[["pay_status_0", "pay_status_2", "pay_status_3", "pay_status_4", "pay_status_5", "pay_status_6"]] > 9)).sum().sum()),
    }
    outliers = {}
    for column in ["credit_limit", "bill_amount_1", "payment_amount_1"]:
        q1, q3 = frame[column].quantile([.25, .75])
        outliers[column] = int(((frame[column] < q1 - 1.5 * (q3 - q1)) | (frame[column] > q3 + 1.5 * (q3 - q1))).sum())
    return ValidationReport(len(frame), int(frame.duplicated().sum()), frame.isna().sum().astype(int).to_dict(), checks, float(frame["default"].mean()), outliers)