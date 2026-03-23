#!/usr/bin/env bash
#
# Evaluate a policy checkpoint on the LIBERO benchmark using the LeRobot
# evaluation script.  Pass the checkpoint directory as the first
# argument.  Additional evaluation options can be supplied via
# environment variables or by editing this script.

set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 <policy_path> [suite_list]"
  echo "Example: $0 outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model all"
  exit 1
fi

POLICY_PATH="$1"
SUITES="${2:-libero_task_0}"
OUTPUT_DIR="outputs/eval/$(basename "$POLICY_PATH")"
mkdir -p "$OUTPUT_DIR"

lerobot-eval \
  --policy.path "$POLICY_PATH" \
  --env.type libero \
  --env.suite "$SUITES" \
  --eval.n_episodes 50 \
  --eval.batch_size 10 \
  --policy.device cuda \
  --policy.use_amp false \
  --output_dir "$OUTPUT_DIR"

echo "Evaluation complete.  Logs and metrics written to $OUTPUT_DIR"