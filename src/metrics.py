from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece += selected.mean() * abs(correct[selected].mean() - confidence[selected].mean())
    return float(ece)


def classification_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predictions = probabilities.argmax(axis=1)
    chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
    return {
        "accuracy": float(accuracy_score(labels, predictions) * 100.0),
        "macro_f1": float(f1_score(labels, predictions, average="macro") * 100.0),
        "nll": float(-np.log(chosen).mean()),
        "ece": expected_calibration_error(probabilities, labels),
    }

