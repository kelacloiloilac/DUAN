"""Reusable training and evaluation loops for the notebooks."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


class ExperimentLogger:
    """Append structured run events so notebook experiments are auditable."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **event) -> None:
        event = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one train or evaluation epoch and return loss, accuracy, macro-F1."""
    training = optimizer is not None
    model.train(training)
    losses, targets, predictions = [], [], []
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            losses.append(loss.item() * labels.size(0))
            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(logits.argmax(1).detach().cpu().numpy())
    return {
        "loss": float(np.sum(losses) / len(loader.dataset)),
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, average="macro", zero_division=0),
    }


def fit(
    model, loaders: Dict, epochs: int, device, learning_rate: float = 1e-3,
    checkpoint_path: str | Path | None = None,
    logger: ExperimentLogger | None = None,
    run_id: str | None = None,
    experiment_name: str | None = None,
    wandb_run=None,
):
    """Train using validation Macro-F1 for model selection."""
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate, weight_decay=1e-4,
    )
    history = []
    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())
    if logger is not None:
        logger.log(
            event="run_started", run_id=run_id, experiment=experiment_name,
            epochs=epochs, learning_rate=learning_rate, device=str(device),
        )
    if wandb_run is not None:
        wandb_run.log({
            "status/run_started": 1,
            "experiment/epoch_limit": epochs,
            "experiment/learning_rate": learning_rate,
        })
    try:
        for epoch in range(1, epochs + 1):
            train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_metrics = run_epoch(model, loaders["val"], criterion, device)
            row = {"epoch": epoch}
            row.update({f"train_{key}": value for key, value in train_metrics.items()})
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            history.append(row)
            print(
                f"Epoch {epoch:02d}: train F1={train_metrics['macro_f1']:.4f}, "
                f"val F1={val_metrics['macro_f1']:.4f}"
            )
            if logger is not None:
                logger.log(
                    event="epoch_completed", run_id=run_id, experiment=experiment_name,
                    **row,
                )
            if wandb_run is not None:
                wandb_run.log(row)
            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                best_state = deepcopy(model.state_dict())
                if checkpoint_path is not None:
                    path = Path(checkpoint_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"model_state": best_state, "epoch": epoch, "val_macro_f1": best_f1}, path)
        if logger is not None:
            logger.log(
                event="run_completed", run_id=run_id, experiment=experiment_name,
                best_val_macro_f1=best_f1,
            )
        if wandb_run is not None:
            wandb_run.log({
                "status/run_completed": 1,
                "val/best_macro_f1": best_f1,
            })
    except Exception as error:
        traceback_text = traceback.format_exc()
        if logger is not None:
            logger.log(
                event="run_failed", run_id=run_id, experiment=experiment_name,
                error_type=type(error).__name__, error_message=str(error),
                traceback=traceback_text,
            )
        if wandb_run is not None:
            wandb_run.log({
                "status/run_failed": 1,
                "error/error_type": type(error).__name__,
                "error/error_message": str(error),
            })
            wandb_run.summary["error_type"] = type(error).__name__
            wandb_run.summary["error_message"] = str(error)
            wandb_run.summary["traceback"] = traceback_text
        raise
    model.load_state_dict(best_state)
    return model, history