from __future__ import annotations

import pandas as pd


def business_insights(frame: pd.DataFrame) -> list[str]:
    """Generate transparent, descriptive portfolio observations from observed outcomes."""
    score_rates = frame.groupby(pd.qcut(frame["credit_score"], 4, duplicates="drop"), observed=True)["default"].mean()
    dti_rates = frame.groupby(pd.qcut(frame["debt_to_income_ratio"], 4, duplicates="drop"), observed=True)["default"].mean()
    previous = frame.groupby("previous_defaults")["default"].mean()
    return [
        f"Lowest credit-score quartile default rate: {score_rates.iloc[0]:.1%}; highest quartile: {score_rates.iloc[-1]:.1%}.",
        f"Lowest DTI quartile default rate: {dti_rates.iloc[0]:.1%}; highest quartile: {dti_rates.iloc[-1]:.1%}.",
        f"Applicants with prior defaults have an observed default rate of {previous[previous.index > 0].mean():.1%}, versus {previous.get(0, 0):.1%} with none.",
    ]