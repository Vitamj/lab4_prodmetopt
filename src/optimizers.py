from __future__ import annotations

import numpy as np


class SGDOptimizer:
    def __init__(self, learning_rate: float, momentum: float = 0.0, nesterov: bool = False) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        updated = {}
        for key, value in params.items():
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(value)
            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            if self.nesterov:
                update = self.momentum * self.velocity[key] - self.learning_rate * grads[key]
            else:
                update = self.velocity[key]
            updated[key] = value + update
        return updated


class AdamLikeOptimizer:
    # Momentum/Nesterov + RMSProp-style adaptive scaling.
    def __init__(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.99,
        epsilon: float = 1e-8,
        nesterov: bool = True,
    ) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.nesterov = nesterov
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        self.t += 1
        updated = {}
        for key, value in params.items():
            if key not in self.m:
                self.m[key] = np.zeros_like(value)
                self.v[key] = np.zeros_like(value)
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (grads[key] ** 2)

            m_hat = self.m[key] / (1.0 - self.beta1**self.t)
            v_hat = self.v[key] / (1.0 - self.beta2**self.t)
            if self.nesterov:
                lookahead = self.beta1 * m_hat + (1.0 - self.beta1) * grads[key] / (1.0 - self.beta1**self.t)
            else:
                lookahead = m_hat
            updated[key] = value - self.learning_rate * lookahead / (np.sqrt(v_hat) + self.epsilon)
        return updated

