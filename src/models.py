from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .activations import brelu, brelu_derivative, sigmoid


@dataclass
class ForwardCache:
    z1: np.ndarray
    a1: np.ndarray
    z2: np.ndarray | None = None
    y_prob: np.ndarray | None = None


class BaseModel:
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        raise NotImplementedError

    def backward(self, X: np.ndarray, y: np.ndarray, cache: ForwardCache) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def parameters(self) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        raise NotImplementedError

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        y_prob, _ = self.forward(X)
        return y_prob


class SingleLayerClassifier(BaseModel):
    def __init__(self, input_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0.0, 0.35, size=(input_dim, 1))
        self.b = np.zeros((1,))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        z = X @ self.W + self.b
        a = brelu(z)
        y_prob = sigmoid(a)
        return y_prob, ForwardCache(z1=z, a1=a, y_prob=y_prob)

    def backward(self, X: np.ndarray, y: np.ndarray, cache: ForwardCache) -> dict[str, np.ndarray]:
        y = y.reshape(-1, 1)
        m = len(X)
        d_logits = (cache.y_prob - y) / m
        d_brelu = d_logits * cache.y_prob * (1.0 - cache.y_prob)
        dz = d_brelu * brelu_derivative(cache.z1)
        return {
            "W": X.T @ dz,
            "b": dz.sum(axis=0),
        }

    def parameters(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "b": self.b}

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.W = params["W"]
        self.b = params["b"]


class HiddenLayerPerceptron(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0.0, 0.35, size=(input_dim, hidden_dim))
        self.b1 = np.zeros((hidden_dim,))
        self.W2 = rng.normal(0.0, 0.35, size=(hidden_dim, 1))
        self.b2 = np.zeros((1,))

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, ForwardCache]:
        z1 = X @ self.W1 + self.b1
        a1 = brelu(z1)
        z2 = a1 @ self.W2 + self.b2
        y_prob = sigmoid(z2)
        return y_prob, ForwardCache(z1=z1, a1=a1, z2=z2, y_prob=y_prob)

    def backward(self, X: np.ndarray, y: np.ndarray, cache: ForwardCache) -> dict[str, np.ndarray]:
        y = y.reshape(-1, 1)
        m = len(X)
        dz2 = (cache.y_prob - y) / m
        dW2 = cache.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * brelu_derivative(cache.z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def parameters(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_parameters(self, params: dict[str, np.ndarray]) -> None:
        self.W1 = params["W1"]
        self.b1 = params["b1"]
        self.W2 = params["W2"]
        self.b2 = params["b2"]

