"""CLI utility for generating a reproducible metadata split."""

from __future__ import annotations

import argparse
from pathlib import Path

from .data import discover_images, seed_everything, stratified_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Directory containing class folders")
    parser.add_argument("--output", default="data/metadata.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    args = parser.parse_args()

    seed_everything(args.seed)
    metadata = stratified_split(
        discover_images(args.data_root), args.seed, args.val_size, args.test_size
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output, index=False)
    print(f"Saved {len(metadata):,} rows to {output}")
    print(metadata.groupby(["split", "class_name"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()