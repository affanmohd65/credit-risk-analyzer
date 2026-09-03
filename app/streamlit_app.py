import os

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

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
    st.caption("Analysis of the UCI Default of Credit Card Clients dataset (Yeh, 2009).")
    values = [f"{len(data):,}", f"{data.default.mean():.1%}", f"NT${data.credit_limit.mean():,.0f}", f"{data.age.mean():.0f}", f"{data.average_repayment_delay.mean():.2f}", f"{data.late_payment_months.mean():.1f}"]
    for box, label, value in zip(st.columns(6), ["Accounts", "Default rate", "Mean credit limit", "Mean age", "Mean repayment delay", "Mean late months"], values):
        box.metric(label, value)
    left, right = st.columns(2)
    age_rates = data.assign(age_band=pd.cut(data.age, [20, 30, 40, 50, 60, 80], labels=["20-29", "30-39", "40-49", "50-59", "60+"]).astype(str)).groupby("age_band", observed=True).default.mean().reset_index()
    left.plotly_chart(px.bar(age_rates, x="age_band", y="default", title="Observed default rate by age band"), width="stretch")
    right.plotly_chart(px.bar(data.groupby("education").default.mean().reset_index(), x="education", y="default", title="Observed default rate by education code"), width="stretch")
    st.plotly_chart(px.histogram(data, x="credit_limit", color="default", nbins=50, title="Credit-limit distribution"), width="stretch")


def prediction_page() -> None:
    st.title("Credit Card Default Prediction")
    st.caption("Payment status: -1 means paid on time; positive values represent months of payment delay in the UCI data dictionary.")
    with st.form("account"):
        profile, payment, balances = st.columns(3)
        values = {"credit_limit": profile.number_input("Credit limit (NT$)", 10_000, 2_000_000, 200_000), "age": profile.number_input("Age", 18, 100, 35), "sex": profile.selectbox("Sex code", [1, 2]), "education": profile.selectbox("Education code", [1, 2, 3, 4]), "marital_status": profile.selectbox("Marital-status code", [1, 2, 3]), "pay_status_0": payment.slider("September payment status", -2, 9, 0), "pay_status_2": payment.slider("August payment status", -2, 9, 0), "pay_status_3": payment.slider("July payment status", -2, 9, 0), "pay_status_4": payment.slider("June payment status", -2, 9, 0), "pay_status_5": payment.slider("May payment status", -2, 9, 0), "pay_status_6": payment.slider("April payment status", -2, 9, 0), "bill_amount_1": balances.number_input("September statement balance", -500_000, 2_000_000, 75_000), "payment_amount_1": balances.number_input("September payment amount", 0, 2_000_000, 15_000)}
        submitted = st.form_submit_button("Estimate default risk")
    if submitted:
        for month in range(2, 7):
            values[f"bill_amount_{month}"] = values["bill_amount_1"]
            values[f"payment_amount_{month}"] = values["payment_amount_1"]
        try:
            result = make_prediction(values)
            for box, label, value in zip(st.columns(4), ["Default probability", "Risk score", "Risk category", "Decision"], [f"{result['default_probability']:.1%}", f"{result['risk_score']}/100", result["risk_category"], result["decision"]]):
                box.metric(label, value)
            st.subheader("Factors associated with the estimate")
            st.write("Risk factors", result["explanation"]["risk_factors"] or ["No prominent rule-based risk factors"])
            st.write("Protective factors", result["explanation"]["positive_factors"] or ["No prominent rule-based protective factors"])
            st.caption(result["business_rule_notice"])
        except requests.RequestException as error:
            st.error(f"Prediction service is unavailable: {error}")


def portfolio(data: pd.DataFrame) -> None:
    st.title("Exposure Analytics")
    lgd = st.slider("Loss given default", .05, .90, .45)
    expected_loss = (data.default * data.credit_limit * lgd).sum()
    for box, label, value in zip(st.columns(4), ["Total credit exposure", "Observed defaults", "Illustrative loss", "Observed default rate"], [f"NT${data.credit_limit.sum():,.0f}", f"{data.default.sum():,.0f}", f"NT${expected_loss:,.0f}", f"{data.default.mean():.1%}"]): box.metric(label, value)
    st.caption("Illustrative expected-loss view using observed default outcomes, credit limit as exposure, and the selected LGD.")


data = load_data()
page = st.sidebar.radio("Navigate", ["Dashboard", "Prediction", "Exposure Analytics", "Model Performance", "Risk Analysis", "Data Explorer", "About"])
if page == "Dashboard": dashboard(data)
elif page == "Prediction": prediction_page()
elif page == "Exposure Analytics": portfolio(data)
elif page == "Model Performance":
    st.title("Model Performance")
    if API_URL:
        try: st.json(requests.get(f"{API_URL}/model-info", timeout=10).json())
        except requests.RequestException as error: st.error(f"Prediction service is unavailable: {error}")
    else:
        import json
        st.json(json.loads(METRICS_PATH.read_text(encoding="utf-8")))
elif page == "Risk Analysis": st.plotly_chart(px.box(data, x="default", y="average_repayment_delay", title="Repayment delay by observed outcome"), width="stretch")
elif page == "Data Explorer": st.dataframe(data.head(2_000), width="stretch")
else: st.markdown("## About\nPublic data source: UCI Default of Credit Card Clients, Yeh (2009), CC BY 4.0. This application is for analysis and learning, not credit approval.")