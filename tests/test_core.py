import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from src.feature_engineering import engineer_features
from src.prediction import loan_decision, risk_category
from src.validation import validate_data


def account_frame() -> pd.DataFrame:
    values = {"account_id": [1, 2], "credit_limit": [50_000, 80_000], "sex": [1, 2], "education": [2, 1], "marital_status": [1, 2], "age": [35, 40], "default": [0, 1]}
    for month in [0, 2, 3, 4, 5, 6]:
        values[f"pay_status_{month}"] = [0, 2]
    for index in range(1, 7):
        values[f"bill_amount_{index}"] = [15_000, 70_000]
        values[f"payment_amount_{index}"] = [5_000, 500]
    return pd.DataFrame(values)


def test_validation_and_features():
    data = account_frame()
    data.loc[0, "age"] = 12
    assert validate_data(data).invalid_counts["invalid_age"] == 1
    assert {"average_repayment_delay", "financial_stress_score"}.issubset(engineer_features(data).columns)


def test_risk_decision_rules():
    assert risk_category(.09) == "LOW"
    assert risk_category(.51) == "VERY_HIGH"
    assert loan_decision("VERY_HIGH") == "REJECT"


def test_api_health_and_rejects_invalid_input():
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.post("/predict", json={}).status_code == 422