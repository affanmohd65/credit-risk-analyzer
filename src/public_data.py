"""Public UCI Default of Credit Card Clients dataset ingestion."""
from __future__ import annotations

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

UCI_DATA_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
CITATION = "Yeh, I. (2009). Default of Credit Card Clients. UCI Machine Learning Repository. DOI: 10.24432/C55S3H"

RENAME_COLUMNS = {
    "ID": "account_id", "LIMIT_BAL": "credit_limit", "SEX": "sex", "EDUCATION": "education",
    "MARRIAGE": "marital_status", "AGE": "age", "PAY_0": "pay_status_0", "PAY_2": "pay_status_2",
    "PAY_3": "pay_status_3", "PAY_4": "pay_status_4", "PAY_5": "pay_status_5", "PAY_6": "pay_status_6",
    "BILL_AMT1": "bill_amount_1", "BILL_AMT2": "bill_amount_2", "BILL_AMT3": "bill_amount_3",
    "BILL_AMT4": "bill_amount_4", "BILL_AMT5": "bill_amount_5", "BILL_AMT6": "bill_amount_6",
    "PAY_AMT1": "payment_amount_1", "PAY_AMT2": "payment_amount_2", "PAY_AMT3": "payment_amount_3",
    "PAY_AMT4": "payment_amount_4", "PAY_AMT5": "payment_amount_5", "PAY_AMT6": "payment_amount_6",
    "default payment next month": "default",
}


def download_public_data() -> pd.DataFrame:
    """Download, standardize, and return the CC BY 4.0 UCI dataset."""
    with urlopen(UCI_DATA_URL, timeout=60) as response:
        archive = BytesIO(response.read())
    with ZipFile(archive) as zipped:
        workbook_name = next(name for name in zipped.namelist() if name.endswith(".xls"))
        frame = pd.read_excel(BytesIO(zipped.read(workbook_name)), sheet_name=0, header=1)
    frame = frame.rename(columns=RENAME_COLUMNS)
    required = set(RENAME_COLUMNS.values())
    if not required.issubset(frame.columns):
        raise ValueError("UCI dataset columns did not match the documented schema.")
    return frame