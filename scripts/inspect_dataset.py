#!/usr/bin/env python
"""
inspect_dataset.py
====================

Inspect a dataset stored on the Hugging Face Hub or accessible via LeRobot's
API. This script prints the top-level keys/features and a preview of examples
for manual inspection.
"""

import argparse
import json
from typing import Any


def try_import_datasets() -> Any:
    try:
        import datasets  # type: ignore

        return datasets
    except ImportError:
        return None


def try_import_lerobot_dataset() -> Any:
    try:
        from lerobot.dataset import LeRobotDataset  # type: ignore

        return LeRobotDataset
    except Exception:
        return None


def _safe_preview(example: Any, max_chars: int = 1000) -> str:
    try:
        text = json.dumps(example, indent=2, default=str)
    except Exception:
        text = str(example)
    return text[:max_chars]


def inspect(dataset_id: str, n: int) -> int:
    ds_lib = try_import_datasets()
    lr_ds = try_import_lerobot_dataset()
    dataset: Any = None

    if lr_ds is not None:
        try:
            dataset = lr_ds.load(dataset_id)
        except Exception:
            dataset = None

    if dataset is None and ds_lib is not None:
        try:
            dataset = ds_lib.load_dataset(dataset_id, split="train")
        except Exception:
            dataset = None

    if dataset is None:
        print("Unable to load dataset. Please ensure datasets or LeRobot is installed.")
        return 1

    print(f"Dataset type: {type(dataset)}")
    print("First keys or features:")
    if hasattr(dataset, "features"):
        print(dataset.features)
    elif hasattr(dataset, "column_names"):
        print(dataset.column_names)
    else:
        print("(unable to determine features)")

    if n <= 0:
        print("n <= 0, skipping example preview.")
        return 0

    print(f"\nShowing up to {n} examples:")
    try:
        total = len(dataset)
    except Exception:
        total = n

    count = min(n, total)
    for idx in range(count):
        try:
            example = dataset[idx]
            print(_safe_preview(example))
        except Exception as exc:
            print(f"Error at sample {idx}: {exc}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a dataset on the Hub or via LeRobot.")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset ID or local path")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to display")
    args = parser.parse_args()
    return inspect(args.dataset, args.n)


if __name__ == "__main__":
    raise SystemExit(main())
