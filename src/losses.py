from __future__ import annotations

import numpy as np


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-8
    y_true = y_true.reshape(-1, 1).astype(float)
    y_pred = np.clip(y_pred.reshape(-1, 1), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))
