from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_classification, make_moons

from .config import ISU_ID


@dataclass
class DatasetBundle:
    name: str
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def _split_dataset(X: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X = X[indices]
    y = y[indices]

    train_end = int(0.6 * len(X))
    val_end = int(0.8 * len(X))
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_datasets(seed: int = ISU_ID) -> list[DatasetBundle]:
    moons_X, moons_y = make_moons(n_samples=400, noise=0.15, random_state=seed)
    cls_X, cls_y = make_classification(
        n_samples=200,
        n_features=5,
        n_redundant=2,
        random_state=seed,
        n_informative=2,
        n_clusters_per_class=2,
        n_classes=2,
    )

    return [
        DatasetBundle("moons", *_split_dataset(moons_X.astype(float), moons_y.astype(int), seed + 11)),
        DatasetBundle(
            "classification5d",
            *_split_dataset(cls_X.astype(float), cls_y.astype(int), seed + 29),
        ),
    ]

