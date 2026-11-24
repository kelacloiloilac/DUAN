"""Reusable training and evaluation loops for the notebooks."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import traceback
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm


class ExperimentLogger:
    """Append structured run events to a JSON Lines file so experiments are auditable."""

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


def compute_class_weights(metadata: pd.DataFrame) -> torch.Tensor:
    """Compute inverse-frequency class weights from training split metadata."""
    train_meta = metadata[metadata["split"] == "train"]
    counts = train_meta["label"].value_counts().sort_index().values
    total_samples = len(train_meta)
    num_classes = len(counts)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Run one train or evaluation epoch; return loss, accuracy, macro-F1."""
    training = optimizer is not None
    model.train(training)
    losses, targets, predictions = [], [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, leave=False, desc="train" if training else "eval"):
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)

            # Phát hiện NaN loss sớm để ghi lỗi có ý nghĩa
            if torch.isnan(loss):
                raise ValueError(
                    f"Loss is NaN during {'training' if training else 'evaluation'}. "
                    "Kiểm tra learning rate, dữ liệu đầu vào và giá trị bất thường."
                )

            if training:
                loss.backward()
                optimizer.step()

            losses.append(loss.item() * labels.size(0))
            targets.extend(labels.detach().cpu().numpy())
            predictions.extend(logits.argmax(1).detach().cpu().numpy())

    n = len(loader.dataset)
    return {
        "loss"     : float(np.sum(losses) / n),
        "accuracy" : float(accuracy_score(targets, predictions)),
        "macro_f1" : float(f1_score(targets, predictions,
                                    average="macro", zero_division=0)),
    }


def fit(
    model: nn.Module,
    loaders: Dict,
    epochs: int,
    device: torch.device,
    learning_rate: float = 1e-3,
    criterion: nn.Module | None = None,
    class_weights: torch.Tensor | None = None,
    checkpoint_path: str | Path | None = None,
    logger: ExperimentLogger | None = None,
    run_id: str | None = None,
    experiment_name: str | None = None,
    wandb_run=None,
):
    """
    Train model, select checkpoint by validation Macro-F1.

    Args:
        model            : PyTorch nn.Module chưa .to(device)
        loaders          : dict với keys 'train' và 'val'
        epochs           : số epoch tối đa
        device           : torch.device
        learning_rate    : learning rate cho AdamW
        criterion        : custom loss function (mặc định CrossEntropyLoss)
        class_weights    : tensor trọng số lớp nghịch đảo cho Weighted Cross-Entropy
        checkpoint_path  : nơi lưu checkpoint tốt nhất (None = không lưu)
        logger           : ExperimentLogger, None = không ghi JSONL
        run_id           : id của run (ví dụ 'E4_resnet_finetune_wce_seed42')
        experiment_name  : tên experiment (ví dụ 'E4_resnet_finetune_wce')
        wandb_run        : wandb run object, None = không log W&B

    Returns:
        (model, history) – model đã load lại best checkpoint; history là list of dicts.
    """
    model.to(device)

    # Khởi tạo hàm mất mát tương thích class-weights
    if criterion is None:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )

    history = []
    best_f1 = -1.0
    best_state = deepcopy(model.state_dict())

    # Log bắt đầu run
    _meta = dict(
        event="run_started",
        run_id=run_id,
        experiment=experiment_name,
        epochs=epochs,
        learning_rate=learning_rate,
        weighted_loss=class_weights is not None,
        device=str(device),
    )
    if logger:
        logger.log(**_meta)
    if wandb_run is not None:
        wandb_run.log({
            "status/run_started": 1,
            "config/epoch_limit": epochs,
            "config/learning_rate": learning_rate,
            "config/weighted_loss": class_weights is not None,
        })

    try:
        for epoch in range(1, epochs + 1):
            train_m = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_m   = run_epoch(model, loaders["val"],   criterion, device)

            row = {"epoch": epoch}
            row.update({f"train_{k}": v for k, v in train_m.items()})
            row.update({f"val_{k}":   v for k, v in val_m.items()})
            history.append(row)

            print(
                f"  Epoch {epoch:02d}/{epochs}: "
                f"train_loss={train_m['loss']:.4f}  train_f1={train_m['macro_f1']:.4f} | "
                f"val_loss={val_m['loss']:.4f}  val_f1={val_m['macro_f1']:.4f}"
            )

            if logger:
                logger.log(event="epoch_completed",
                           run_id=run_id, experiment=experiment_name, **row)
            if wandb_run is not None:
                wandb_run.log(row)

            # Checkpoint tốt nhất theo val Macro-F1
            if val_m["macro_f1"] > best_f1:
                best_f1 = val_m["macro_f1"]
                best_state = deepcopy(model.state_dict())
                if checkpoint_path is not None:
                    ckpt = Path(checkpoint_path)
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "model_state": best_state,
                            "epoch": epoch,
                            "val_macro_f1": best_f1,
                            "run_id": run_id,
                            "experiment": experiment_name,
                        },
                        ckpt,
                    )
                    if wandb_run is not None:
                        wandb_run.summary["best_checkpoint_epoch"] = epoch

        # Log hoàn thành run
        if logger:
            logger.log(event="run_completed",
                       run_id=run_id, experiment=experiment_name,
                       best_val_macro_f1=best_f1)
        if wandb_run is not None:
            wandb_run.log({"status/run_completed": 1,
                           "val/best_macro_f1": best_f1})

    except Exception as error:
        tb_text = traceback.format_exc()
        if logger:
            logger.log(
                event="run_failed",
                run_id=run_id,
                experiment=experiment_name,
                error_type=type(error).__name__,
                error_message=str(error),
                traceback=tb_text,
            )
        if wandb_run is not None:
            wandb_run.log({"status/run_failed": 1,
                           "error/error_type": type(error).__name__,
                           "error/message": str(error)[:500]})
            wandb_run.summary["status"]     = "failed"
            wandb_run.summary["error_type"] = type(error).__name__
            wandb_run.summary["error_msg"]  = str(error)[:500]
            wandb_run.summary["traceback"]  = tb_text[:3000]
        raise

    model.load_state_dict(best_state)
    return model, history
