import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import QUALITY_REPORT_PATH, RAW_DATA_PATH
from src.india_data import generate_india_retail_loans
from src.validation import validate_data

if __name__ == "__main__":
    data = generate_india_retail_loans()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DATA_PATH, index=False)
    QUALITY_REPORT_PATH.write_text(json.dumps(validate_data(data).to_dict() | {"source": "India-focused generated retail lending portfolio"}, indent=2), encoding="utf-8")
    print(f"Wrote {len(data):,} Indian retail-loan applications to {RAW_DATA_PATH}")
    print(f"Default rate: {data.default.mean():.2%}")