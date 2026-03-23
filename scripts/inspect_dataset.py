#!/usr/bin/env python
"""
inspect_dataset.py
====================

Inspect a dataset stored on the Hugging Face Hub or accessible via LeRobot's
dataset API.  This script prints the top‐level keys and the first few
examples from the dataset for manual inspection.  Use this utility to
understand the structure of a new dataset before writing custom
processors.

Example usage::

    python scripts/inspect_dataset.py --dataset HuggingFaceVLA/smol-libero --n 5

The script gracefully handles missing dependencies by printing a helpful
message instead of crashing.
"""

import argparse
import json
from typing import Any

def try_import_datasets():
    try:
        import datasets  # type: ignore
        return datasets
    except ImportError:
        return None


def try_import_lerobot_dataset():
    try:
        from lerobot.dataset import LeRobotDataset  # type: ignore
        return LeRobotDataset
    except Exception:
        return None


def inspect(dataset_id: str, n: int) -> None:
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
        print("Unable to load dataset.  Please ensure the datasets library or LeRobot is installed.")
        return

    print(f"Dataset type: {type(dataset)}")
    print("First keys or features:")
    if hasattr(dataset, "features"):
        print(dataset.features)
    elif hasattr(dataset, "column_names"):
        print(dataset.column_names)
    else:
        print("(unable to determine features)")

    print(f"\nShowing {n} examples:")
    try:
        for i in range(min(n, len(dataset))):
            example = dataset[i]
            print(json.dumps(example, indent=2)[:1000])
    except Exception as e:
        print(f"Error while iterating dataset: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a dataset on the Hub or via LeRobot.")
    parser.add_argument("--dataset", type=str, required=True, help="Hugging Face dataset ID or local path")
    parser.add_argument("--n", type=int, default=5, help="Number of examples to display")
    args = parser.parse_args()
    inspect(args.dataset, args.n)


if __name__ == "__main__":
    main()