from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import METRICS_PATH, MODEL_PATH, PROCESSED_DIR, RANDOM_SEED, RAW_DATA_PATH
from src.evaluate import evaluate_probabilities, optimize_threshold
from src.feature_engineering import TARGET_COLUMN, model_features
from src.preprocessing import make_preprocessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _models(positive_weight: float) -> dict[str, object]:
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "decision_tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=40, class_weight="balanced", random_state=RANDOM_SEED),
        "random_forest": RandomForestClassifier(n_estimators=180, max_depth=14, min_samples_leaf=12, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
    }
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=.06, subsample=.8, colsample_bytree=.8, scale_pos_weight=positive_weight, eval_metric="logloss", random_state=RANDOM_SEED)
    except ImportError:
        logging.info("XGBoost unavailable; comparison continues with sklearn models.")
    return models


def main() -> None:
    frame = pd.read_csv(RAW_DATA_PATH)
    train, remainder = train_test_split(frame, test_size=.30, stratify=frame[TARGET_COLUMN], random_state=RANDOM_SEED)
    validation, test = train_test_split(remainder, test_size=.50, stratify=remainder[TARGET_COLUMN], random_state=RANDOM_SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("validation", validation), ("test", test)]: split.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
    x_train, x_validation, x_test = map(model_features, [train, validation, test])
    y_train, y_validation, y_test = train[TARGET_COLUMN], validation[TARGET_COLUMN], test[TARGET_COLUMN]
    weight = (1 - y_train.mean()) / y_train.mean()
    comparison, fitted = {}, {}
    for name, classifier in _models(weight).items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(x_train)), ("classifier", classifier)])
        pipeline.fit(x_train, y_train)
        probability = pipeline.predict_proba(x_validation)[:, 1]
        comparison[name] = evaluate_probabilities(y_validation, probability)
        fitted[name] = pipeline
        logging.info("%s PR-AUC %.3f", name, comparison[name]["pr_auc"])
    selected = max(comparison, key=lambda name: comparison[name]["pr_auc"])
    threshold = optimize_threshold(y_validation, fitted[selected].predict_proba(x_validation)[:, 1])
    calibrated = CalibratedClassifierCV(fitted[selected], method="sigmoid", cv=3)
    calibrated.fit(pd.concat([x_train, x_validation]), pd.concat([y_train, y_validation]))
    test_probability = calibrated.predict_proba(x_test)[:, 1]
    metrics = {"selection_metric": "pr_auc", "selected_model": selected, "threshold": threshold, "validation": comparison,
               "test": evaluate_probabilities(y_test, test_probability, threshold)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logging.info("Saved calibrated %s model. Test PR-AUC %.3f", selected, metrics["test"]["pr_auc"])


if __name__ == "__main__":
    main()