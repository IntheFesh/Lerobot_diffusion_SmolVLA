#!/usr/bin/env python
"""
benchmark_latency.py
===================

Measure inference latency of a trained policy. The script loads a
pretrained policy checkpoint and repeatedly calls its forward/predict
path on dummy inputs to estimate average execution time per call.
"""

import argparse
import time
from typing import Any, Optional


def try_import_lerobot_policy():
    try:
        # Depending on LeRobot version, import path may differ.
        from lerobot.policies import PreTrainedPolicy  # type: ignore
        return PreTrainedPolicy
    except Exception:
        return None


def load_policy(path: str) -> Optional[Any]:
    policy_base = try_import_lerobot_policy()
    if policy_base is None:
        print("LeRobot is not installed or unavailable. Cannot load policy.")
        return None
    try:
        policy = policy_base.from_pretrained(path)
        return policy
    except Exception as e:
        print(f"Failed to load policy from '{path}': {e}")
        return None


def _infer_call(policy: Any, dummy_obs: Any) -> Any:
    """
    Try common inference entrypoints in a robust order.
    """
    if hasattr(policy, "predict"):
        return policy.predict(dummy_obs)  # type: ignore[attr-defined]
    return policy(dummy_obs)


def benchmark(policy: Any, n_iters: int) -> None:
    import numpy as np

    if n_iters <= 0:
        raise ValueError("n_iters must be > 0")

    # Dummy input shape may need adjustment for your policy.
    dummy_obs = {
        "observation": np.zeros((1, 3, 84, 84), dtype=np.float32),
    }

    # Warm-up
    try:
        if hasattr(policy, "reset"):
            policy.reset(1)
    except Exception:
        pass

    try:
        _ = _infer_call(policy, dummy_obs)
    except Exception as e:
        print(f"Warm-up inference failed: {e}")
        return

    times_ms = []
    for _ in range(n_iters):
        start = time.perf_counter()
        _ = _infer_call(policy, dummy_obs)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)

    avg_ms = sum(times_ms) / len(times_ms)
    p50 = sorted(times_ms)[len(times_ms) // 2]
    p95 = sorted(times_ms)[max(0, int(len(times_ms) * 0.95) - 1)]

    print(f"Average inference latency: {avg_ms:.3f} ms")
    print(f"P50 latency: {p50:.3f} ms")
    print(f"P95 latency: {p95:.3f} ms")
    print(f"Iterations: {n_iters}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark policy inference latency")

    # New preferred flags
    parser.add_argument("--policy-path", type=str, default=None, help="Path to pretrained policy directory")
    parser.add_argument("--n-iters", type=int, default=100, help="Number of benchmark iterations")

    # Backward-compatible legacy aliases
    parser.add_argument("--policy.path", dest="policy_path_legacy", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--n_iters", dest="n_iters_legacy", type=int, default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    policy_path = args.policy_path if args.policy_path is not None else args.policy_path_legacy
    n_iters = args.n_iters if args.n_iters_legacy is None else args.n_iters_legacy

    if not policy_path:
        parser.error("Please provide --policy-path (or legacy --policy.path).")

    policy = load_policy(policy_path)
    if policy is None:
        return
    benchmark(policy, n_iters)


if __name__ == "__main__":
    main()
