"""Reproducible India-focused retail-loan portfolio generator."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED


def generate_india_retail_loans(n_rows: int = 100_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate correlated INR loan applications and feature-dependent defaults."""
    rng = np.random.default_rng(seed)
    cities = rng.choice(["Mumbai", "Delhi NCR", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Kolkata", "Tier 2/3"], n_rows, p=[.13, .15, .12, .08, .09, .08, .07, .06, .22])
    employment = rng.choice(["Salaried", "Self Employed", "Professional", "Contract"], n_rows, p=[.62, .22, .09, .07])
    loan_type = rng.choice(["Personal Loan", "Two Wheeler", "Consumer Durable", "Home Improvement", "Business Loan"], n_rows, p=[.38, .17, .20, .12, .13])
    age = rng.integers(21, 66, n_rows)
    employment_years = np.clip(age - 20 + rng.normal(-9, 5, n_rows), 0, 40).round(1)
    multiplier = pd.Series(cities).map({"Mumbai": 1.27, "Delhi NCR": 1.20, "Bengaluru": 1.22, "Chennai": 1.05, "Hyderabad": 1.08, "Pune": 1.11, "Ahmedabad": .94, "Kolkata": .90, "Tier 2/3": .72}).to_numpy()
    income = np.exp(rng.normal(13.15, .62, n_rows)) * multiplier * (1 + employment_years / 80)
    income *= np.where(employment == "Professional", 1.35, np.where(employment == "Contract", .72, 1))
    income = np.clip(income, 120_000, 10_000_000).round(0)
    bureau_score = np.clip(590 + .000020 * income + 3.8 * employment_years + rng.normal(0, 52, n_rows), 300, 900).round()
    inquiries = rng.poisson(np.clip((730 - bureau_score) / 85, .25, 3.5)).clip(0, 12)
    overdue = rng.poisson(np.clip((690 - bureau_score) / 100, .04, 2.4)).clip(0, 12)
    existing_emi = np.clip(income / 12 * rng.beta(1.5, 5.0, n_rows) + overdue * 2_500, 0, 200_000)
    term = rng.choice([12, 18, 24, 36, 48, 60], n_rows, p=[.12, .11, .30, .28, .08, .11])
    loan_amount = np.clip(income * rng.lognormal(-1.2, .7, n_rows), 20_000, 3_000_000)
    loan_amount *= np.where(loan_type == "Business Loan", 1.55, np.where(loan_type == "Two Wheeler", .55, 1))
    loan_amount = np.clip(loan_amount, 20_000, 3_000_000)
    rate = np.clip(10.5 + (760 - bureau_score) / 24 + overdue * 1.1 + inquiries * .38 + rng.normal(0, 1.2, n_rows), 8.0, 30.0)
    monthly_rate = rate / 1200
    proposed_emi = loan_amount * monthly_rate * (1 + monthly_rate) ** term / ((1 + monthly_rate) ** term - 1)
    foir = np.clip((existing_emi + proposed_emi) / (income / 12), .02, 1.25)
    bank_balance = np.clip(income / 12 * rng.lognormal(.25, .75, n_rows), 0, 4_000_000)
    residence = rng.choice(["Owned", "Rented", "Family Owned"], n_rows, p=[.31, .45, .24])
    loan_to_income = loan_amount / income
    logit = (-6.25 + (700 - bureau_score) / 72 + 2.45 * foir + 1.0 * overdue + .26 * inquiries + .32 * (employment == "Contract") + .20 * (employment == "Self Employed") + .0000006 * loan_amount - .00000035 * bank_balance + .26 * (residence == "Rented") + rng.normal(0, .48, n_rows))
    default_probability = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, default_probability)
    data = pd.DataFrame({"application_id": [f"INR{index:07d}" for index in range(n_rows)], "age": age, "city_tier": cities, "employment_type": employment, "employment_years": employment_years, "annual_income_inr": income, "residence_type": residence, "bureau_score": bureau_score, "credit_inquiries_6m": inquiries, "overdue_accounts": overdue, "existing_emi_inr": existing_emi.round(0), "bank_balance_inr": bank_balance.round(0), "loan_type": loan_type, "loan_amount_inr": loan_amount.round(0), "loan_term_months": term, "interest_rate": rate.round(2), "proposed_emi_inr": proposed_emi.round(0), "foir": foir.round(4), "loan_to_income_ratio": loan_to_income.round(4), "default": default})
    for column in ["annual_income_inr", "bureau_score", "bank_balance_inr"]:
        data.loc[rng.choice(data.index, max(1, n_rows // 100), replace=False), column] = np.nan
    return data