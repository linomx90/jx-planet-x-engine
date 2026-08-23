#!/usr/bin/env python3
"""Execute the prelocked JX/DE441 ten-year compatibility validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
LOCAL_VENDOR = PROJECT_ROOT.parent / ".vendor"
if LOCAL_VENDOR.is_dir():
    sys.path.insert(0, str(LOCAL_VENDOR))
sys.path.insert(0, str(SOURCE_ROOT))

from jxplanetx.de441_horizons import run_validation, write_blocked_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        type=Path,
        default=SCRIPT_PATH.with_name("contract_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_PATH.with_name("result_v1.json"),
    )
    parser.add_argument(
        "--states",
        type=Path,
        default=SCRIPT_PATH.with_name("jx_states_v1.csv"),
    )
    parser.add_argument(
        "--residuals",
        type=Path,
        default=SCRIPT_PATH.with_name("residuals_v1.csv"),
    )
    arguments = parser.parse_args()
    try:
        result = run_validation(
            arguments.contract,
            PROJECT_ROOT,
            arguments.output,
            arguments.states,
            arguments.residuals,
        )
    except BaseException as error:
        result = write_blocked_result(arguments.output, arguments.contract, error)
        print(json.dumps({"science_verdict": "BLOCKED", "error": str(error)}))
        return 2
    print(
        json.dumps(
            {
                "science_verdict": result["science_verdict"],
                "result": str(arguments.output.resolve()),
                "outer_max_position_km": result["derived_outer_gate_units"]["max_position_residual_km"],
                "outer_max_velocity_m_per_s": result["derived_outer_gate_units"]["max_velocity_residual_m_per_s"],
            }
        )
    )
    return 0 if result["science_verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
