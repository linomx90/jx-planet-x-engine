#!/usr/bin/env python3
"""Run the prelocked exact-zeta corrective replay of the official v2 pools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jxplanetx.provenance import sha256_file
from jxplanetx.survey_selection_v3 import finalize_survey_selection


EXPECTED_V2_RESULT_SHA256 = "9f0d86d6365b776a64333f55a431c143d36793f489f9a9297c3bbaa054e5c9bc"
EXPECTED_CORRECT_POOL_SHA256 = "b2b99a91ca52fe819b8e2a5d0a860488d0b650664cbe5abb5a6d0b773b3f8297"
EXPECTED_WRONG_POOL_SHA256 = "85c55eca72c8aaac0292c1f25d6ed3aed1b1d03efa194e35cde023396b8716fd"


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} hash mismatch: expected {expected}, observed {observed}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--contract", required=True, type=Path)
    command.add_argument("--v2-result", required=True, type=Path)
    command.add_argument("--correct-manifest", required=True, type=Path)
    command.add_argument("--wrong-manifest", required=True, type=Path)
    command.add_argument("--output", required=True, type=Path)
    return command


def main() -> int:
    args = parser().parse_args()
    _require_hash(args.v2_result, EXPECTED_V2_RESULT_SHA256, "v2 result")
    _require_hash(args.correct_manifest, EXPECTED_CORRECT_POOL_SHA256, "correct pool")
    _require_hash(args.wrong_manifest, EXPECTED_WRONG_POOL_SHA256, "wrong pool")
    v2_result = json.loads(args.v2_result.read_text(encoding="utf-8"))
    if v2_result.get("verdict") != "INVALID" or v2_result.get("invalid_reasons") != [
        {
            "code": "seed_block_verdict_instability",
            "message": "a leave-one-block-out verdict changed",
        }
    ]:
        raise RuntimeError("v3 input is not the declared immutable v2 failure")
    result = finalize_survey_selection(
        args.contract,
        args.correct_manifest,
        args.wrong_manifest,
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["verdict"] == "PASSED":
        return 0
    return 2 if result["verdict"] == "INVALID" else 3


if __name__ == "__main__":
    raise SystemExit(main())

