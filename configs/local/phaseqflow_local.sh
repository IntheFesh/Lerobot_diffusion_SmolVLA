#!/usr/bin/env bash
#
# Launch a PhaseQFlow training run on the Smol-LIBERO dataset.
# Supports environment-variable overrides for smoke tests.

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/phaseqflow_smol_local}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/smol-libero}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-5000}"
EVAL_FREQ="${EVAL_FREQ:-1000}"
NUM_PHASES="${NUM_PHASES:-4}"
USE_QUALITY_WEIGHT="${USE_QUALITY_WEIGHT:-true}"

mkdir -p "$OUTPUT_DIR"

if command -v lerobot-train >/dev/null 2>&1; then
  TRAIN_CMD=(lerobot-train)
elif python -c "import lerobot" >/dev/null 2>&1; then
  TRAIN_CMD=(python -m lerobot.scripts.train)
else
  echo "Error: LeRobot training entrypoint not found."
  echo "Install LeRobot first, then re-run."
  exit 127
fi

"${TRAIN_CMD[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --policy.type phaseqflow \
  --dataset.repo_id "$DATASET_REPO_ID" \
  --env.type libero \
  --batch_size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --eval_freq "$EVAL_FREQ" \
  --policy.device "$DEVICE" \
  --policy.use_amp false \
  --policy.num_phases "$NUM_PHASES" \
  --policy.use_quality_weight "$USE_QUALITY_WEIGHT"

echo "PhaseQFlow training completed. Check $OUTPUT_DIR for checkpoints."
