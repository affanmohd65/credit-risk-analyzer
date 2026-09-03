from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def evaluate_probabilities(y_true: Any, probabilities: np.ndarray, threshold: float = .5) -> dict[str, float]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {"accuracy": accuracy_score(y_true, predicted), "precision": precision_score(y_true, predicted, zero_division=0),
            "recall": recall_score(y_true, predicted, zero_division=0), "f1": f1_score(y_true, predicted, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probabilities), "pr_auc": average_precision_score(y_true, probabilities),
            "specificity": tn / max(tn + fp, 1), "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(fn + tp, 1), "true_negatives": int(tn), "false_positives": int(fp),
            "false_negatives": int(fn), "true_positives": int(tp), "threshold": threshold}


def optimize_threshold(y_true: Any, probabilities: np.ndarray) -> float:
    thresholds = np.arange(.10, .71, .01)
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])