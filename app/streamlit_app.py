import os
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import METRICS_PATH, MODEL_PATH, RAW_DATA_PATH
from src.feature_engineering import engineer_features
from src.explainability import individual_risk_factors
from src.prediction import predict_records

st.set_page_config(page_title="Credit Risk Analyzer", page_icon="CR", layout="wide")
API_URL = os.getenv("API_URL", "")


@st.cache_data
def load_data() -> pd.DataFrame:
    return engineer_features(pd.read_csv(RAW_DATA_PATH))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def make_prediction(values: dict) -> dict:
    if API_URL:
        response = requests.post(f"{API_URL}/predict", json=values, timeout=20)
        response.raise_for_status()
        return response.json()
    result = predict_records(load_model(), [values])[0]
    result["explanation"] = individual_risk_factors(values)
    return result


def dashboard(data: pd.DataFrame) -> None:
    st.title("Credit Risk Analyzer")
    st.caption("India-focused retail lending analytics and default prediction dashboard.")
    values = [f"{len(data):,}", f"{data.default.mean():.1%}", f"INR {data.loan_amount_inr.mean():,.0f}", f"{data.bureau_score.mean():.0f}", f"{data.foir.mean():.1%}", f"INR {data.proposed_emi_inr.mean():,.0f}"]
    for box, label, value in zip(st.columns(6), ["Applications", "Default rate", "Average loan", "Average bureau score", "Average FOIR", "Average proposed EMI"], values):
        box.metric(label, value)
    left, right = st.columns(2)
    left.plotly_chart(px.bar(data.groupby("city_tier").default.mean().reset_index(), x="city_tier", y="default", title="Default rate by city"), width="stretch")
    right.plotly_chart(px.bar(data.groupby("loan_type").default.mean().reset_index(), x="loan_type", y="default", title="Default rate by loan type"), width="stretch")
    st.plotly_chart(px.histogram(data, x="loan_amount_inr", color="default", nbins=50, title="Loan amount distribution"), width="stretch")


def prediction_page() -> None:
    st.title("Indian Retail Loan Default Prediction")
    with st.form("account"):
        profile, credit, loan = st.columns(3)
        values = {"city_tier": profile.selectbox("City", ["Mumbai", "Delhi NCR", "Bengaluru", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Kolkata", "Tier 2/3"]), "employment_type": profile.selectbox("Employment type", ["Salaried", "Self Employed", "Professional", "Contract"]), "residence_type": profile.selectbox("Residence type", ["Owned", "Rented", "Family Owned"]), "age": profile.number_input("Age", 21, 70, 35), "employment_years": profile.number_input("Employment years", 0.0, 40.0, 5.0), "annual_income_inr": credit.number_input("Annual income (INR)", 120_000, 10_000_000, 750_000), "bureau_score": credit.number_input("Bureau score", 300, 900, 700), "credit_inquiries_6m": credit.number_input("Credit inquiries in last 6 months", 0, 12, 1), "overdue_accounts": credit.number_input("Overdue accounts", 0, 12, 0), "existing_emi_inr": credit.number_input("Existing monthly EMI (INR)", 0, 200_000, 10_000), "bank_balance_inr": loan.number_input("Bank balance (INR)", 0, 4_000_000, 100_000), "loan_type": loan.selectbox("Loan type", ["Personal Loan", "Two Wheeler", "Consumer Durable", "Home Improvement", "Business Loan"]), "loan_amount_inr": loan.number_input("Loan amount (INR)", 20_000, 3_000_000, 300_000), "loan_term_months": loan.selectbox("Loan term", [12, 18, 24, 36, 48, 60], index=2), "interest_rate": loan.number_input("Interest rate", 8.0, 30.0, 14.0), "proposed_emi_inr": loan.number_input("Proposed monthly EMI (INR)", 1_000, 300_000, 15_000), "foir": loan.slider("Fixed obligation to income ratio", 0.0, 1.25, 0.35), "loan_to_income_ratio": loan.slider("Loan to income ratio", 0.0, 10.0, 0.40)}
        submitted = st.form_submit_button("Estimate default risk")
    if submitted:
        try:
            result = make_prediction(values)
            for box, label, value in zip(st.columns(4), ["Default probability", "Risk score", "Risk category", "Decision"], [f"{result['default_probability']:.1%}", f"{result['risk_score']}/100", result["risk_category"], result["decision"]]):
                box.metric(label, value)
            st.subheader("Factors associated with the estimate")
            st.write("Risk factors", result["explanation"]["risk_factors"] or ["No prominent rule-based risk factors"])
            st.write("Protective factors", result["explanation"]["positive_factors"] or ["No prominent rule-based protective factors"])
        except requests.RequestException as error:
            st.error(f"Prediction service is unavailable: {error}")


def portfolio(data: pd.DataFrame) -> None:
    st.title("Exposure Analytics")
    lgd = st.slider("Loss given default", .05, .90, .45)
    expected_loss = (data.default * data.loan_amount_inr * lgd).sum()
    for box, label, value in zip(st.columns(4), ["Total loan exposure", "Observed defaults", "Expected loss", "Observed default rate"], [f"INR {data.loan_amount_inr.sum():,.0f}", f"{data.default.sum():,.0f}", f"INR {expected_loss:,.0f}", f"{data.default.mean():.1%}"]): box.metric(label, value)
    st.caption("Expected loss is calculated from default outcomes, loan exposure, and the selected LGD.")


def load_metrics() -> dict:
    if API_URL:
        response = requests.get(f"{API_URL}/model-info", timeout=10)
        response.raise_for_status()
        return response.json()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def model_performance() -> None:
    st.title("Model Performance")
    try:
        metrics = load_metrics()
    except requests.RequestException as error:
        st.error(f"Model metrics are unavailable: {error}")
        return
    metric_options = {"PR-AUC": "pr_auc", "ROC-AUC": "roc_auc", "F1 Score": "f1", "Recall": "recall", "Precision": "precision"}
    selected_label = st.selectbox("Compare models by", list(metric_options))
    selected_metric = metric_options[selected_label]
    comparison = pd.DataFrame(metrics["validation"]).T.reset_index(names="model")
    comparison = comparison.sort_values(selected_metric, ascending=False)
    st.metric("Selected model", metrics["selected_model"].replace("_", " ").title(), f"Test {selected_label}: {metrics['test'][selected_metric]:.3f}")
    st.plotly_chart(px.bar(comparison, x="model", y=selected_metric, color=selected_metric, title=f"Validation {selected_label} by model", text_auto=".3f"), width="stretch")
    visible_columns = ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "specificity", "false_negative_rate"]
    st.dataframe(comparison[visible_columns].style.format({column: "{:.3f}" for column in visible_columns[1:]}), width="stretch", hide_index=True)


def risk_analysis(data: pd.DataFrame) -> None:
    st.title("Risk Analysis")
    age_range = st.slider("Age range", int(data.age.min()), int(data.age.max()), (int(data.age.min()), int(data.age.max())))
    loan_range = st.slider("Loan amount range (INR)", int(data.loan_amount_inr.min()), int(data.loan_amount_inr.quantile(.99)), (int(data.loan_amount_inr.min()), int(data.loan_amount_inr.quantile(.99))))
    selected_types = st.multiselect("Loan types", sorted(data.loan_type.unique()), default=sorted(data.loan_type.unique()))
    filtered = data[data.age.between(*age_range) & data.loan_amount_inr.between(*loan_range) & data.loan_type.isin(selected_types)]
    st.caption(f"Showing {len(filtered):,} selected applications.")
    left, right = st.columns(2)
    left.plotly_chart(px.box(filtered, x="default", y="foir", title="FOIR by observed outcome"), width="stretch")
    right.plotly_chart(px.histogram(filtered, x="bureau_score", color="default", nbins=40, barmode="overlay", title="Bureau score by observed outcome"), width="stretch")


data = load_data()
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "Exposure Analytics", "Model Performance", "Risk Analysis", "Data Explorer", "About"])
if page == "Dashboard": dashboard(data)
elif page == "Prediction": prediction_page()
elif page == "Exposure Analytics": portfolio(data)
elif page == "Model Performance": model_performance()
elif page == "Risk Analysis": risk_analysis(data)
elif page == "Data Explorer": st.dataframe(data.head(2_000), width="stretch")
else: st.markdown("## About\nAn end-to-end credit risk analytics project covering public-data ingestion, validation, feature engineering, model training, calibrated probability estimates, FastAPI services, monitoring, and interactive portfolio analytics.")