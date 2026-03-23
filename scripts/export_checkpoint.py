#!/usr/bin/env python
"""
export_checkpoint.py
====================

Export a LeRobot checkpoint directory into a standalone artifact.  The
script copies the `config.json` and `model.safetensors` files from the
source checkpoint directory into a specified destination directory.
This can be helpful when you want to upload your model to the Hugging
Face Hub or share it with collaborators.

Example usage::

    python scripts/export_checkpoint.py \
      --src outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
      --dst exports/phaseqflow_smol_local

If the source files are missing or the destination cannot be created,
the script prints an error instead of raising.
"""

import argparse
import os
import shutil

def export_checkpoint(src: str, dst: str) -> None:
    if not os.path.isdir(src):
        print(f"Source path does not exist: {src}")
        return
    os.makedirs(dst, exist_ok=True)
    files_to_copy = ["config.json", "model.safetensors"]
    for fname in files_to_copy:
        src_file = os.path.join(src, fname)
        if not os.path.isfile(src_file):
            print(f"File not found: {src_file}")
            continue
        dst_file = os.path.join(dst, fname)
        try:
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} -> {dst_file}")
        except Exception as e:
            print(f"Error copying {src_file}: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Export a LeRobot checkpoint directory")
    parser.add_argument("--src", type=str, required=True, help="Source checkpoint directory")
    parser.add_argument("--dst", type=str, required=True, help="Destination directory")
    args = parser.parse_args()
    export_checkpoint(args.src, args.dst)

if __name__ == "__main__":
    main()