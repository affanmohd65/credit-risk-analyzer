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
        "nonpositive_income": int((frame["annual_income_inr"] <= 0).sum()),
        "nonpositive_loan_amount": int((frame["loan_amount_inr"] <= 0).sum()),
        "invalid_bureau_score": int(((frame["bureau_score"] < 300) | (frame["bureau_score"] > 900)).sum()),
        "invalid_foir": int(((frame["foir"] < 0) | (frame["foir"] > 2)).sum()),
    }
    outliers = {}
    for column in ["annual_income_inr", "loan_amount_inr", "foir"]:
        q1, q3 = frame[column].quantile([.25, .75])
        outliers[column] = int(((frame[column] < q1 - 1.5 * (q3 - q1)) | (frame[column] > q3 + 1.5 * (q3 - q1))).sum())
    return ValidationReport(len(frame), int(frame.duplicated().sum()), frame.isna().sum().astype(int).to_dict(), checks, float(frame["default"].mean()), outliers)