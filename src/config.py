from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = ROOT / "data" / "raw" / "uci_credit_card_default.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metrics.json"
QUALITY_REPORT_PATH = ROOT / "models" / "data_quality_report.json"
RANDOM_SEED = 42


@dataclass(frozen=True)
class RiskRules:
    low_max: float = 0.10
    medium_max: float = 0.25
    high_max: float = 0.50


RISK_RULES = RiskRules()