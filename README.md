# Credit Risk Analyzer

An end-to-end credit risk analytics project using the public **UCI Default of Credit Card Clients** dataset. It ingests and validates data, engineers features, trains and calibrates multiple classifiers, exposes predictions through FastAPI, and presents results in Streamlit.

## Data Source

The project downloads 30,000 account records from the UCI Machine Learning Repository:

- Yeh, I. (2009). *Default of Credit Card Clients*. UCI Machine Learning Repository. DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H)
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

The target is default payment in the next month. Predictors include credit limit, demographic codes, six months of repayment status, six statement balances, and six payment amounts. The repository includes the downloaded public data and trained artifact so Streamlit Cloud can serve predictions immediately.

## Methodology

- Data validation: missingness, duplicate rows, age, credit-limit, repayment-status, and outlier checks.
- Features: average statement balance, average payment, payment-to-bill ratio, current balance-to-limit ratio, repayment delay, delayed-payment months, balance trend, and financial stress.
- Evaluation: stratified train/validation/test split, class weighting, PR-AUC model selection, ROC-AUC, precision, recall, F1, specificity, FPR, FNR, confusion counts, threshold tuning, and sigmoid probability calibration.
- Models: Logistic Regression, Decision Tree, Random Forest, and XGBoost.
- Monitoring: Population Stability Index (PSI) with Normal, Warning, and Critical bands.

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
python src/train.py
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for API documentation. The dashboard uses the local trained model by default. To send predictions through a separately hosted FastAPI service, set `API_URL` to its public base URL.

## Docker

```powershell
docker compose up --build
```

The API is exposed on port 8000 and the dashboard on port 8501.

## Streamlit Cloud

1. Create a Streamlit Cloud app from `affanmohd65/credit-risk-analyzer`.
2. Select `main` as the branch and `app/streamlit_app.py` as the entry point.
3. Deploy. No secrets are required for the self-contained Cloud application because the public data and fitted model are included.

## Tests

```powershell
pytest -q
```

Tests cover validation, feature engineering, risk rules, and FastAPI input validation.

## Limitations

The UCI data represents a specific historical population and period. Before any real-world use, conduct temporal validation, fairness analysis, economic stress testing, privacy and security review, model governance, monitoring against observed outcomes, and regulatory review.