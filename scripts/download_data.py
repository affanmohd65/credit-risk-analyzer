from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import QUALITY_REPORT_PATH, RAW_DATA_PATH
from src.public_data import CITATION, download_public_data
from src.validation import validate_data


if __name__ == "__main__":
    data = download_public_data()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_DATA_PATH, index=False)
    report = validate_data(data).to_dict() | {"source": CITATION}
    QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {len(data):,} public records to {RAW_DATA_PATH}")