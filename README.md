# India Credit Risk Analyzer

An end-to-end machine-learning project for India-focused retail-loan default prediction. The application creates an INR lending portfolio, validates data quality, engineers lending features, compares and calibrates classifiers, provides FastAPI prediction endpoints, and presents portfolio insights in Streamlit.

## India Lending Portfolio

The reproducible generator creates 100,000 retail-loan applications under a fixed random seed. The data model reflects common Indian retail-credit inputs:

- City, employment type, employment tenure, and residence type
- Annual income in INR, bank balance, existing EMI, and proposed EMI
- Bureau score, overdue accounts, and credit inquiries from the previous six months
- Loan type, amount, term, interest rate, FOIR, and loan-to-income ratio

Default outcomes are driven by bureau score, repayment obligations, overdue accounts, inquiry frequency, employment profile, loan exposure, bank balance, and residence type. The generated portfolio is designed to demonstrate Indian credit-risk workflows and does not contain lender customer records.

## Technology Stack

- Python 3.12, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, joblib
- FastAPI, Pydantic, Uvicorn
- Streamlit, Plotly, requests
- pytest, Docker, Docker Compose, GitHub Actions

## Modeling Workflow

- Validation for missing values, duplicate records, invalid age, INR values, bureau score, FOIR, class distribution, and outliers
- Features for EMI-to-income, total EMI-to-income, loan-to-income, balance-to-loan, bureau risk gap, and financial stress
- Stratified train, validation, and test split
- Logistic Regression, Decision Tree, Random Forest, and XGBoost comparison
- Class weighting and `scale_pos_weight` for class imbalance
- PR-AUC model selection, F1-based threshold selection, and sigmoid probability calibration
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, specificity, FPR, FNR, and confusion-matrix metrics
- Risk categories, expected-loss analytics, and PSI monitoring

## Run Locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/generate_india_data.py
python src/train.py
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501` for the dashboard and `http://localhost:8000/docs` for the FastAPI documentation.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service readiness |
| `POST` | `/predict` | One Indian retail-loan prediction |
| `POST` | `/batch-predict` | Batch default probability predictions |
| `GET` | `/model-info` | Model selection and evaluation metrics |
| `GET` | `/risk-rules` | Risk-band thresholds |

## Docker

```powershell
docker compose up --build
```

The Docker image generates the Indian loan portfolio and trains the model. FastAPI is exposed on port 8000 and Streamlit on port 8501.

## Tests

```powershell
pytest -q
```

Tests cover Indian lending validation, feature engineering, risk categorization, decision logic, and FastAPI request validation.

## Streamlit Cloud

Deploy `app/streamlit_app.py` from the `main` branch. The Streamlit application loads the fitted model directly by default. Set `API_URL` only when using a separately deployed FastAPI service.