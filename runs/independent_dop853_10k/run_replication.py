#!/usr/bin/env python3
"""Run a locked independent SciPy DOP853 population replication."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


# Prevent hidden BLAS thread pools from multiplying the contracted worker count.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from jxplanetx.independent_dop853 import run_independent_replication  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    result = run_independent_replication(
        arguments.contract,
        arguments.run_dir,
        arguments.output,
        arguments.workers,
    )
    print(
        json.dumps(
            {
                "verdict": result["verdict"],
                "independent_effect": result["independent_effect"][
                    "source_minus_control_injection_fraction"
                ],
                "reference_effect": result["reference_effect"][
                    "source_minus_control_injection_fraction"
                ],
                "absolute_effect_difference": result[
                    "absolute_source_control_effect_difference"
                ],
                "result": str(arguments.output.resolve()),
            }
        ),
        flush=True,
    )
    return 0 if result["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
