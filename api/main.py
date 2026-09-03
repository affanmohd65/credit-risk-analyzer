import json
import logging
from functools import lru_cache

import joblib
from fastapi import FastAPI, HTTPException

from api.schemas import BatchRequest, CreditAccount
from src.config import METRICS_PATH, MODEL_PATH, RISK_RULES
from src.explainability import individual_risk_factors
from src.prediction import predict_records

logger = logging.getLogger(__name__)
app = FastAPI(title="Credit Risk API", version="1.0.0", description="Credit-card default probability API using UCI public data.")


@lru_cache
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model missing. Run: python src/train.py")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy" if MODEL_PATH.exists() else "model_not_trained"}


@app.post("/predict")
def predict(application: CreditAccount) -> dict:
    try:
        values = application.model_dump()
        prediction = predict_records(get_model(), [values])[0]
        prediction["explanation"] = individual_risk_factors(values)
        return prediction
    except Exception as error:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/batch-predict")
def batch_predict(request: BatchRequest) -> dict:
    try:
        return {"predictions": predict_records(get_model(), [item.model_dump() for item in request.applications])}
    except Exception as error:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/model-info")
def model_info() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="No training metrics found")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@app.get("/risk-rules")
def risk_rules() -> dict:
    return {"thresholds": {"low_max": RISK_RULES.low_max, "medium_max": RISK_RULES.medium_max, "high_max": RISK_RULES.high_max}, "rules_notice": "Risk categories are illustrative and must not be used as lending decisions."}