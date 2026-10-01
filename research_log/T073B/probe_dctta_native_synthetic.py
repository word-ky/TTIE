"""Synthetic-only driver: run_dctta_native_batch on a small synthetic native-size low set.

Usage: probe_dctta_native_synthetic.py COUNT <run_dctta_native_batch arguments>
Only the expected receipt count changes; every other code path is the batch runner's.
"""

import sys

import run_dctta_native_batch as runner

if __name__ == "__main__":
    runner.EXPECTED_COUNT = int(sys.argv.pop(1))
    runner.main()
