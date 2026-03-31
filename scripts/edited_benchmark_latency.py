#!/usr/bin/env python
"""Compatibility wrapper for benchmark latency script.

This file is kept for backward compatibility. It delegates to
`scripts/benchmark_latency.py` so there is a single implementation to maintain.
"""

from benchmark_latency import main


if __name__ == "__main__":
    raise SystemExit(main())
