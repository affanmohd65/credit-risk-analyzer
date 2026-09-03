import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.feature_engineering import engineer_features
from src.prediction import loan_decision, risk_category
from src.validation import validate_data


def loan_frame() -> pd.DataFrame:
    return pd.DataFrame({"application_id": ["INR0000001", "INR0000002"], "age": [35, 40], "city_tier": ["Mumbai", "Pune"], "employment_type": ["Salaried", "Contract"], "employment_years": [8, 2], "annual_income_inr": [900_000, 450_000], "residence_type": ["Owned", "Rented"], "bureau_score": [760, 590], "credit_inquiries_6m": [1, 5], "overdue_accounts": [0, 2], "existing_emi_inr": [10_000, 25_000], "bank_balance_inr": [300_000, 15_000], "loan_type": ["Personal Loan", "Business Loan"], "loan_amount_inr": [250_000, 700_000], "loan_term_months": [24, 36], "interest_rate": [12.5, 20.0], "proposed_emi_inr": [12_000, 30_000], "foir": [.29, .85], "loan_to_income_ratio": [.28, 1.56], "default": [0, 1]})


def test_validation_and_features():
    data = loan_frame()
    data.loc[0, "age"] = 12
    assert validate_data(data).invalid_counts["invalid_age"] == 1
    assert {"emi_to_income_ratio", "financial_stress_score"}.issubset(engineer_features(data).columns)


def test_risk_decision_rules():
    assert risk_category(.09) == "LOW"
    assert risk_category(.51) == "VERY_HIGH"
    assert loan_decision("VERY_HIGH") == "REJECT"


def test_api_health_and_rejects_invalid_input():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json={}).status_code == 422