from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import classification_metrics


@dataclass
class TrainingResult:
    best_params: dict[str, np.ndarray]
    history: list[dict[str, float]]
    best_epoch: int
    best_val_accuracy: float
    best_val_loss: float


def batch_iterator(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    indices = rng.permutation(len(X))
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


def train_model(
    model,
    optimizer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
    patience: int = 25,
) -> TrainingResult:
    rng = np.random.default_rng(seed)
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    best_params = {k: v.copy() for k, v in model.parameters().items()}
    history: list[dict[str, float]] = []
    stalled = 0

    for epoch in range(1, epochs + 1):
        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size, rng):
            y_prob, cache = model.forward(X_batch)
            grads = model.backward(X_batch, y_batch, cache)
            new_params = optimizer.step(model.parameters(), grads)
            model.set_parameters(new_params)

        train_metrics = classification_metrics(y_train, model.predict_proba(X_train))
        val_metrics = classification_metrics(y_val, model.predict_proba(X_val))
        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)

        improved = (
            val_metrics["accuracy"] > best_val_accuracy
            or (
                abs(val_metrics["accuracy"] - best_val_accuracy) < 1e-9
                and val_metrics["loss"] < best_val_loss
            )
        )
        if improved:
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            best_params = {k: v.copy() for k, v in model.parameters().items()}
            stalled = 0
        else:
            stalled += 1
            if stalled >= patience:
                break

    model.set_parameters(best_params)
    return TrainingResult(best_params, history, best_epoch, best_val_accuracy, best_val_loss)

