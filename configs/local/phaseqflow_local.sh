#!/usr/bin/env bash
#
# Launch a PhaseQFlow training run on the Smol‑LIBERO dataset.  This script
# trains the custom phase‑aware and quality‑weighted policy implemented in
# `lerobot_policy_phaseqflow`.  It is tuned for small‑GPU debugging.

set -e

OUTPUT_DIR="outputs/train/phaseqflow_smol_local"
mkdir -p "$OUTPUT_DIR"

# The policy.type value corresponds to the entry point registered by the
# `lerobot_policy_phaseqflow` package.  Make sure you have installed the
# policy package in editable mode: `pip install -e ./lerobot_policy_phaseqflow`.

lerobot-train \
  --output_dir "$OUTPUT_DIR" \
  --policy.type phaseqflow \
  --dataset.repo_id HuggingFaceVLA/smol-libero \
  --env.type libero \
  --batch_size 8 \
  --steps 5000 \
  --eval_freq 1000 \
  --policy.device cuda \
  --policy.use_amp false \
  --policy.num_phases 4 \
  --policy.use_quality_weight true

echo "PhaseQFlow local training completed.  Check $OUTPUT_DIR for checkpoints."