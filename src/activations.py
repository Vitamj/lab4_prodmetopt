from __future__ import annotations

import numpy as np


def brelu(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def brelu_derivative(x: np.ndarray) -> np.ndarray:
    return ((x > 0.0) & (x < 1.0)).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))

