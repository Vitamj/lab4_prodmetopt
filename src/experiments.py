from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import ISU_ID
from .data import build_datasets
from .metrics import classification_metrics
from .models import HiddenLayerPerceptron, SingleLayerClassifier
from .optimizers import AdamLikeOptimizer, SGDOptimizer
from .training import train_model


ROOT = Path(__file__).resolve().parents[1]
IMG_DIR = ROOT / "img"


def make_model(model_name: str, input_dim: int, seed: int):
    if model_name == "single_layer":
        return SingleLayerClassifier(input_dim=input_dim, seed=seed)
    if model_name == "hidden_layer_perceptron":
        return HiddenLayerPerceptron(input_dim=input_dim, hidden_dim=12, seed=seed)
    raise ValueError(model_name)


def make_optimizer(optimizer_name: str, learning_rate: float):
    if optimizer_name == "sgd":
        return SGDOptimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    if optimizer_name == "adam_like":
        return AdamLikeOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99, nesterov=True)
    raise ValueError(optimizer_name)


def hyperparameter_grid(dataset_name: str, model_name: str) -> list[dict[str, float | int | str]]:
    if dataset_name == "moons":
        if model_name == "single_layer":
            return [
                {"optimizer": "sgd", "lr": 0.03, "batch_size": 16, "epochs": 220},
                {"optimizer": "adam_like", "lr": 0.02, "batch_size": 16, "epochs": 220},
            ]
        return [
            {"optimizer": "sgd", "lr": 0.05, "batch_size": 16, "epochs": 260},
            {"optimizer": "adam_like", "lr": 0.01, "batch_size": 16, "epochs": 260},
        ]
    if model_name == "single_layer":
        return [
            {"optimizer": "sgd", "lr": 0.03, "batch_size": 16, "epochs": 220},
            {"optimizer": "adam_like", "lr": 0.01, "batch_size": 16, "epochs": 220},
        ]
    return [
        {"optimizer": "sgd", "lr": 0.03, "batch_size": 16, "epochs": 260},
        {"optimizer": "adam_like", "lr": 0.008, "batch_size": 16, "epochs": 260},
    ]


def choose_best_configuration(dataset) -> tuple[dict[str, object], list[dict[str, object]]]:
    all_runs: list[dict[str, object]] = []
    best_run: dict[str, object] | None = None

    for model_name in ("single_layer", "hidden_layer_perceptron"):
        for idx, cfg in enumerate(hyperparameter_grid(dataset.name, model_name)):
            seed = ISU_ID + 100 * (idx + 1) + (0 if model_name == "single_layer" else 50)
            model = make_model(model_name, dataset.X_train.shape[1], seed)
            optimizer = make_optimizer(str(cfg["optimizer"]), float(cfg["lr"]))
            training = train_model(
                model=model,
                optimizer=optimizer,
                X_train=dataset.X_train,
                y_train=dataset.y_train,
                X_val=dataset.X_val,
                y_val=dataset.y_val,
                epochs=int(cfg["epochs"]),
                batch_size=int(cfg["batch_size"]),
                seed=seed,
            )
            val_metrics = classification_metrics(dataset.y_val, model.predict_proba(dataset.X_val))
            run = {
                "dataset": dataset.name,
                "model": model_name,
                "optimizer": cfg["optimizer"],
                "learning_rate": cfg["lr"],
                "batch_size": cfg["batch_size"],
                "epochs_ran": len(training.history),
                "best_epoch": training.best_epoch,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "history": training.history,
                "model_obj": model,
            }
            all_runs.append(run)
            if best_run is None or (
                run["val_accuracy"] > best_run["val_accuracy"]
                or (
                    abs(run["val_accuracy"] - best_run["val_accuracy"]) < 1e-9
                    and run["val_loss"] < best_run["val_loss"]
                )
            ):
                best_run = run
    assert best_run is not None
    return best_run, all_runs


def _save_learning_curve(dataset_name: str, all_runs: list[dict[str, object]]) -> None:
    plt.figure(figsize=(8, 5))
    for run in all_runs:
        history = pd.DataFrame(run["history"])
        label = f"{run['model']} + {run['optimizer']}"
        plt.plot(history["epoch"], history["val_accuracy"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title(f"Validation accuracy: {dataset_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"learning_curve_{dataset_name}.png", dpi=180)
    plt.close()


def _save_decision_boundary(dataset, model, dataset_name: str) -> None:
    if dataset.X_train.shape[1] != 2:
        return
    X_all = np.vstack([dataset.X_train, dataset.X_val, dataset.X_test])
    y_all = np.concatenate([dataset.y_train, dataset.y_val, dataset.y_test])
    x_min, x_max = X_all[:, 0].min() - 0.7, X_all[:, 0].max() + 0.7
    y_min, y_max = X_all[:, 1].min() - 0.7, X_all[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 250), np.linspace(y_min, y_max, 250))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict_proba(grid).reshape(xx.shape)

    plt.figure(figsize=(7, 6))
    plt.contourf(xx, yy, preds, levels=20, cmap="RdBu", alpha=0.65)
    plt.scatter(X_all[:, 0], X_all[:, 1], c=y_all, cmap="bwr", edgecolor="k", s=18)
    plt.title(f"Decision boundary: {dataset_name}")
    plt.tight_layout()
    plt.savefig(IMG_DIR / f"decision_boundary_{dataset_name}.png", dpi=180)
    plt.close()


def run_all_experiments() -> list[dict[str, object]]:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    datasets = build_datasets()
    rows: list[dict[str, object]] = []
    full_report: list[dict[str, object]] = []

    for dataset in datasets:
        best_run, all_runs = choose_best_configuration(dataset)
        model = best_run["model_obj"]
        train_metrics = classification_metrics(dataset.y_train, model.predict_proba(dataset.X_train))
        val_metrics = classification_metrics(dataset.y_val, model.predict_proba(dataset.X_val))
        test_metrics = classification_metrics(dataset.y_test, model.predict_proba(dataset.X_test))

        row = {
            "dataset": dataset.name,
            "best_model": best_run["model"],
            "best_optimizer": best_run["optimizer"],
            "learning_rate": best_run["learning_rate"],
            "batch_size": best_run["batch_size"],
            "epochs_ran": best_run["epochs_ran"],
            "best_epoch": best_run["best_epoch"],
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
        }
        rows.append(row)
        full_report.append(
            {
                "summary": row,
                "all_runs": [
                    {k: v for k, v in run.items() if k not in {"model_obj"}}
                    for run in all_runs
                ],
            }
        )
        _save_learning_curve(dataset.name, all_runs)
        _save_decision_boundary(dataset, model, dataset.name)

    with (IMG_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (IMG_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(full_report, handle, ensure_ascii=False, indent=2)

    return rows


if __name__ == "__main__":
    run_all_experiments()

