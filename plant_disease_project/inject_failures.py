from __future__ import annotations

import argparse
import os
import random
import sys
import traceback
from datetime import datetime, timezone

PROJECT_ROOT = None  # sẽ set từ __main__

FAILURE_SCENARIOS = [
    {
        "name": "OOM_RuntimeError",
        "description": "Mô phỏng GPU Out-of-Memory trong forward pass",
        "exception": RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB "
                                   "(GPU 0; 4.00 GiB total capacity; 3.80 GiB already allocated)"),
        "partial_epochs": 2,  # giả lập đã chạy được 2 epoch rồi mới lỗi
    },
    {
        "name": "NaN_Loss",
        "description": "Loss trở thành NaN sau vài iteration (learning rate quá lớn)",
        "exception": ValueError("Loss is NaN at epoch 3, step 120. "
                                 "Likely cause: learning_rate=1e-1 too large for fine-tuning."),
        "partial_epochs": 3,
    },
    {
        "name": "DataLoader_FileNotFound",
        "description": "Một số file ảnh bị xóa hoặc đường dẫn sai trong metadata",
        "exception": FileNotFoundError("[Errno 2] No such file or directory: "
                                        "'data/raw/color/Tomato___Bacterial_spot/00075aa8-d81a-4184-8541-b692b78d398a___GCREC_Bact.Sp 3474.JPG'"),
        "partial_epochs": 0,
    },
    {
        "name": "CUDA_DeviceMismatch",
        "description": "Tensor nằm trên CPU nhưng model đang ở CUDA",
        "exception": RuntimeError("Expected all tensors to be on the same device, "
                                   "but found at least two devices, cuda:0 and cpu!"),
        "partial_epochs": 1,
    },
    {
        "name": "InvalidConfig_NegativeLR",
        "description": "Learning rate âm khiến optimizer thất bại",
        "exception": ValueError("Invalid learning rate: -0.001"),
        "partial_epochs": 0,
    },
    {
        "name": "CheckpointCorrupted",
        "description": "Checkpoint bị corrupt khi load lại để resume",
        "exception": RuntimeError("PytorchStreamReader failed reading zip archive: "
                                   "failed finding central directory"),
        "partial_epochs": 5,
    },
    {
        "name": "BatchSizeTooLarge",
        "description": "Batch size 256 gây OOM ngay từ đầu",
        "exception": RuntimeError("CUDA out of memory. Tried to allocate 512 MiB "
                                   "with batch_size=256 on GPU with 4 GiB."),
        "partial_epochs": 0,
    },
    {
        "name": "LabelMismatch",
        "description": "Số lớp trong checkpoint không khớp với dataset hiện tại",
        "exception": RuntimeError("size mismatch for fc.weight: "
                                   "copying a param with shape torch.Size([38, 512]) "
                                   "from checkpoint, the shape in current model is torch.Size([26, 512])."),
        "partial_epochs": 0,
    },
]


def inject_failures(project: str, entity: str | None, n: int) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb chưa cài. Chạy: pip install wandb")
        sys.exit(1)

    rng = random.Random(99)
    experiments = ["E0_simplecnn", "E1_resnet_frozen", "E2_resnet_finetune", "E3_resnet_finetune_aug"]
    seeds = [42, 43, 44, 45, 46]
    succeeded = 0

    print(f"Bắt đầu inject {n} failed runs vào W&B project '{project}' ...")

    for i in range(n):
        scenario = rng.choice(FAILURE_SCENARIOS)
        exp      = rng.choice(experiments)
        seed     = rng.choice(seeds)
        attempt  = rng.randint(1, 3)
        lr       = rng.choice([1e-3, 1e-4, 1e-1])
        bs       = rng.choice([32, 64, 128, 256])

        run_name = (f"INJECT_FAIL_{i+1:02d}_{scenario['name']}"
                    f"_{exp}_seed{seed}_attempt{attempt}")
        print(f"  [{i+1}/{n}] {run_name}")

        run = None
        try:
            run = wandb.init(
                project  = project,
                entity   = entity,
                name     = run_name,
                group    = exp,
                job_type = "training",
                reinit   = "finish_previous",
                tags     = [exp, f"seed-{seed}", f"attempt-{attempt}",
                            "injected-failure", scenario["name"]],
                config   = {
                    "experiment"      : exp,
                    "seed"            : seed,
                    "batch_size"      : bs,
                    "epochs"          : 10,
                    "learning_rate"   : lr,
                    "augmentation"    : "aug" in exp,
                    "num_classes"     : 38,
                    "device"          : "cuda:0",
                    "torch_version"   : "2.1.0",
                    "python_version"  : "3.11.9",
                    "attempt"         : attempt,
                    "max_attempts"    : 3,
                    "failure_scenario": scenario["name"],
                    "injected"        : True,
                },
            )
            run.define_metric("epoch")
            run.define_metric("*", step_metric="epoch")

            # Giả lập partial training (vài epoch trước khi lỗi)
            partial = scenario["partial_epochs"]
            for ep in range(1, partial + 1):
                train_f1 = rng.uniform(0.05, 0.4) * (ep / 10)
                val_f1   = train_f1 * rng.uniform(0.7, 0.95)
                run.log({
                    "epoch"          : ep,
                    "train_loss"     : rng.uniform(1.5, 3.5) / ep,
                    "train_accuracy" : train_f1 * rng.uniform(0.9, 1.1),
                    "train_macro_f1" : train_f1,
                    "val_loss"       : rng.uniform(1.8, 4.0) / ep,
                    "val_accuracy"   : val_f1 * rng.uniform(0.9, 1.1),
                    "val_macro_f1"   : val_f1,
                })

            # Giả lập exception
            tb_text = (
                f"Traceback (most recent call last):\n"
                f"  File \"src/train.py\", line 91, in fit\n"
                f"    train_metrics = run_epoch(model, loaders['train'], criterion, device, optimizer)\n"
                f"  File \"src/train.py\", line 46, in run_epoch\n"
                f"    logits = model(images)\n"
                f"{type(scenario['exception']).__name__}: {scenario['exception']}\n"
            )

            run.log({
                "status/run_failed"   : 1,
                "error/error_type"    : type(scenario["exception"]).__name__,
                "error/message"       : str(scenario["exception"])[:500],
            })
            run.summary["status"]          = "failed"
            run.summary["error_type"]      = type(scenario["exception"]).__name__
            run.summary["error_msg"]       = str(scenario["exception"])[:500]
            run.summary["traceback"]       = tb_text[:3000]
            run.summary["partial_epochs"]  = partial
            run.summary["scenario"]        = scenario["name"]
            run.summary["description"]     = scenario["description"]
            succeeded += 1

        except Exception as outer:
            print(f"    !! Lỗi khi inject run {i+1}: {outer}")
        finally:
            if run is not None:
                run.finish(exit_code=1)

    print(f"\nHoàn tất: {succeeded}/{n} failed runs đã được đẩy lên W&B.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="plant-disease-deep-learning")
    parser.add_argument("--entity",  default=None)
    parser.add_argument("--n",       type=int, default=25,
                        help="Số failed run cần inject (mặc định 25)")
    args = parser.parse_args()
    inject_failures(args.project, args.entity, args.n)
