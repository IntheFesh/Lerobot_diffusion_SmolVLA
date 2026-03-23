#!/usr/bin/env bash
#
# Evaluate a policy on the PushT toy environment.  This is useful for
# sanity checking that your installation and policy plugin work
# correctly before moving on to larger benchmarks.  Pass the policy
# path as the first argument; if omitted, the script will fall back
# to the built‑in diffusion model for PushT provided by LeRobot.

set -e

POLICY_PATH=${1:-lerobot/diffusion_pusht}
OUTPUT_DIR="outputs/eval/pusht_$(basename "$POLICY_PATH")"
mkdir -p "$OUTPUT_DIR"

lerobot-eval \
  --policy.path "$POLICY_PATH" \
  --env.type pusht \
  --eval.n_episodes 20 \
  --eval.batch_size 10 \
  --policy.device cuda \
  --policy.use_amp false \
  --output_dir "$OUTPUT_DIR"

echo "PushT evaluation complete.  Results in $OUTPUT_DIR"