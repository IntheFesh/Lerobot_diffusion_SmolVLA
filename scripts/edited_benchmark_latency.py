#!/usr/bin/env python
"""Legacy filename wrapper for PhaseQFlow latency benchmark.

Kept only for compatibility with old command lines.
"""

from benchmark_latency import main


if __name__ == "__main__":
    raise SystemExit(main())
