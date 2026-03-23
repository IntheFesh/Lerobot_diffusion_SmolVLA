#!/usr/bin/env python
"""
benchmark_latency.py
===================

Measure inference latency of a trained policy.  The script loads a
pretrained policy checkpoint and repeatedly calls its `predict` method
on dummy inputs to estimate the average execution time per call.  This
metric is helpful for comparing policies with different architectures.

Example usage::

    python scripts/benchmark_latency.py \
      --policy.path outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
      --n_iters 100

The script reports the average latency in milliseconds.  If the policy
cannot be loaded, a message is printed instead of an exception.
"""

import argparse
import time
from typing import Any

def try_import_lerobot_policy():
    try:
        from lerobot.policies import PreTrainedPolicy  # type: ignore
        return PreTrainedPolicy
    except Exception:
        return None

def load_policy(path: str) -> Any:
    PolicyBase = try_import_lerobot_policy()
    if PolicyBase is None:
        print("LeRobot is not installed or unavailable.  Cannot load policy.")
        return None
    try:
        policy = PolicyBase.from_pretrained(path)
        return policy
    except Exception as e:
        print(f"Failed to load policy: {e}")
        return None

def benchmark(policy: Any, n_iters: int) -> None:
    import numpy as np
    # Construct dummy observation and history arrays.  The shape should
    # correspond to what the policy expects; here we make reasonable
    # assumptions.  Adjust these shapes if they differ for your policy.
    dummy_obs = {
        "observation": np.zeros((1, 3, 84, 84), dtype=np.float32),
    }
    # Warm up
    try:
        policy.reset(1)
    except Exception:
        pass
    _ = policy(dummy_obs)
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        _ = policy(dummy_obs)
        end = time.perf_counter()
        times.append((end - start) * 1000.0)
    avg_ms = sum(times) / len(times)
    print(f"Average inference latency: {avg_ms:.3f} ms per call over {n_iters} iterations")

def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark policy inference latency")
    parser.add_argument("--policy.path", type=str, required=True, help="Path to pretrained policy directory")
    parser.add_argument("--n_iters", type=int, default=100, help="Number of iterations")
    args = parser.parse_args()
    policy = load_policy(args.policy_path)
    if policy is None:
        return
    benchmark(policy, args.n_iters)

if __name__ == "__main__":
    main()