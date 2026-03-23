#!/usr/bin/env bash
#
# Launch a multi‑GPU PhaseQFlow training run using the accelerate CLI.  This
# script is intended for cloud environments where multiple GPUs are available.

set -e

OUTPUT_DIR="outputs/train/phaseqflow_libero_cloud"
mkdir -p "$OUTPUT_DIR"

# Adjust the number of processes (--num_processes) and other accelerate
# parameters to match your hardware.  The environment variable
# CUDA_VISIBLE_DEVICES should enumerate the GPUs you wish to use.

accelerate launch \
  lerobot-train \
  --output_dir "$OUTPUT_DIR" \
  --policy.type phaseqflow \
  --dataset.repo_id HuggingFaceVLA/libero \
  --env.type libero \
  --batch_size 32 \
  --steps 100000 \
  --eval_freq 5000 \
  --policy.device cuda \
  --policy.use_amp true \
  --policy.num_phases 4 \
  --policy.use_quality_weight true \
  --policy.flow_matching true

echo "PhaseQFlow cloud training started.  Monitor logs for progress."