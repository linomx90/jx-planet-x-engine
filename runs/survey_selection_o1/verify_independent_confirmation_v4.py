#!/usr/bin/env python3
"""Verify an immutable JX-O1 V4 result against a fresh deterministic replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from jxplanetx.survey_selection_v4 import _finalize_impl as replay_survey_selection


def _verify_existing_result(
    result_path: Path,
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Return an immutable result only when it matches a fresh replay."""

    try:
        observed = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot verify immutable result {result_path}: {exc}") from exc
    if observed != replay:
        raise RuntimeError(
            f"immutable result does not match deterministic replay: {result_path}"
        )
    return observed


def verify_v4_result(
    contract_path: Path,
    correct_manifest_path: Path,
    wrong_manifest_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Replay finalization from the registered pools and verify the saved result."""

    replay = replay_survey_selection(
        contract_path,
        correct_manifest_path,
        wrong_manifest_path,
    )
    return _verify_existing_result(result_path, replay)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--contract", required=True, type=Path)
    command.add_argument("--correct-manifest", required=True, type=Path)
    command.add_argument("--wrong-manifest", required=True, type=Path)
    command.add_argument("--result", required=True, type=Path)
    return command


def main() -> int:
    args = parser().parse_args()
    result = verify_v4_result(
        args.contract,
        args.correct_manifest,
        args.wrong_manifest,
        args.result,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
