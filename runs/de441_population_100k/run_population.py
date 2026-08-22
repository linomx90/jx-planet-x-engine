#!/usr/bin/env python3
"""Execute a locked JX real-epoch population contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
VENDOR = PROJECT.parent / ".vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(PROJECT / "src"))

from jxplanetx.de441_population import run_de441_population


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    arguments = parser.parse_args()
    result = run_de441_population(
        arguments.contract,
        arguments.run_dir,
        arguments.output,
        arguments.workers,
    )
    effect = result["population_screening"]["source_minus_control"]
    print(
        json.dumps(
            {
                "science_verdict": result["science_verdict"],
                "effect_classification": effect["effect_classification"],
                "injection_fraction_difference": effect["sampled_injection_fraction"],
                "bootstrap_95_percent_CI": effect["paired_block_bootstrap_95_percent_CI"],
                "result": str(arguments.output.resolve()),
            }
        ),
        flush=True,
    )
    return 0 if result["science_verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
