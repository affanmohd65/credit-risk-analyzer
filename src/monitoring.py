from __future__ import annotations

import numpy as np
import pandas as pd


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    edges = np.unique(np.nanquantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    expected, _ = np.histogram(reference.dropna(), bins=edges)
    actual, _ = np.histogram(current.dropna(), bins=edges)
    expected = np.maximum(expected / max(expected.sum(), 1), .0001)
    actual = np.maximum(actual / max(actual.sum(), 1), .0001)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_status(psi: float) -> str:
    return "NORMAL" if psi < .10 else "WARNING" if psi < .25 else "CRITICAL"


def monitoring_summary(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float | str]]:
    return {column: {"psi": round(population_stability_index(reference[column], current[column]), 4),
                     "status": drift_status(population_stability_index(reference[column], current[column]))} for column in features}