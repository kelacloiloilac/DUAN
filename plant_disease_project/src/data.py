"""Dataset discovery, leakage-safe splits, and PyTorch dataloaders."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def seed_everything(seed: int = 42) -> None:
    """Make Python, NumPy, and PyTorch experiments reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_images(data_root: str | Path) -> pd.DataFrame:
    """Return one row per image with class name and integer label."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found under {root}")

    labels = {path.name: index for index, path in enumerate(class_dirs)}
    rows = []
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix in IMAGE_EXTENSIONS:
                rows.append(
                    {"path": str(image_path.resolve()), "class_name": class_dir.name,
                     "label": labels[class_dir.name]}
                )
    if not rows:
        raise ValueError(f"No supported images found under {root}")
    return pd.DataFrame(rows)


def stratified_split(
    metadata: pd.DataFrame,
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> pd.DataFrame:
    """Create train/validation/test labels while preserving class proportions."""
    if not 0 < val_size + test_size < 1:
        raise ValueError("val_size + test_size must be between 0 and 1")
    train_val, test = train_test_split(
        metadata, test_size=test_size, stratify=metadata["label"], random_state=seed
    )
    relative_val_size = val_size / (1 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_val_size,
        stratify=train_val["label"],
        random_state=seed,
    )
    result = metadata.copy()
    split_by_path = {
        row.path: "train" for row in train.itertuples()
    }
    split_by_path.update({row.path: "val" for row in validation.itertuples()})
    split_by_path.update({row.path: "test" for row in test.itertuples()})
    result["split"] = result["path"].map(split_by_path)
    if result["split"].isna().any():
        raise RuntimeError("Some images were not assigned to a split")
    return result.sort_values(["split", "label", "path"]).reset_index(drop=True)


class PlantVillageDataset(Dataset):
    """Image dataset backed by a metadata dataframe."""

    def __init__(self, frame: pd.DataFrame, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def make_transforms(image_size: int = 224, augmentation: bool = True):
    """Return train and evaluation transforms with ImageNet normalization."""
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    if augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, eval_transform


def make_dataloaders(
    metadata: pd.DataFrame,
    batch_size: int = 32,
    image_size: int = 224,
    augmentation: bool = True,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build train, validation, and test loaders from a split dataframe."""
    train_transform, eval_transform = make_transforms(image_size, augmentation)
    loaders = {}
    for split, transform, shuffle in (
        ("train", train_transform, True),
        ("val", eval_transform, False),
        ("test", eval_transform, False),
    ):
        frame = metadata[metadata["split"] == split]
        if frame.empty:
            raise ValueError(f"Split '{split}' is empty")
        dataset = PlantVillageDataset(frame, transform)
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=torch.cuda.is_available()
        )
    return loaders