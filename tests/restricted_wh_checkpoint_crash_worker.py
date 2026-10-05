"""Abrupt-process test helper for restricted-WH checkpoint publication."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import jxplanetx.restricted_wh_checkpoint as checkpoint_module
from jxplanetx.restricted_wh_lifecycle import initialize_restricted_wh_lifecycle

from tests.test_cuda_restricted_wh import _state
from tests.test_restricted_wh_checkpoint import _dump, _namespace, _run_segment


_EXIT_CODES = {
    "before_array_link": 91,
    "after_array_link": 92,
    "before_record_link": 93,
    "after_record_link": 94,
}


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] not in _EXIT_CODES:
        raise SystemExit("usage: crash_worker PHASE CHECKPOINT_ROOT")
    phase = sys.argv[1]
    root = Path(sys.argv[2])
    initial = initialize_restricted_wh_lifecycle(_state(1))
    first, _tile0, _spec0 = _run_segment(initial, start=0, steps=1)
    result, tile, spec = _run_segment(first.state, start=1, steps=1)
    record_path = _namespace(root) / "checkpoint_00000000000000000000.json"
    predecessor = json.loads(record_path.read_text(encoding="utf-8"))[
        "checkpoint_content_sha256"
    ]
    original = checkpoint_module._commit_create_only

    def commit_then_die(temporary: Path, destination: Path) -> None:
        if destination.suffix == ".npz":
            if phase == "before_array_link":
                os._exit(_EXIT_CODES[phase])
            if phase == "after_array_link":
                os.link(temporary, destination)
                os._exit(_EXIT_CODES[phase])
        if destination.suffix == ".json":
            if phase == "before_record_link":
                os._exit(_EXIT_CODES[phase])
            if phase == "after_record_link":
                os.link(temporary, destination)
                os._exit(_EXIT_CODES[phase])
        original(temporary, destination)

    checkpoint_module._commit_create_only = commit_then_die
    _dump(
        root,
        index=1,
        genesis_state=initial,
        input_state=first.state,
        result=result,
        tile=tile,
        spec=spec,
        predecessor=predecessor,
    )
    raise AssertionError("the injected crash boundary was not reached")


if __name__ == "__main__":
    raise SystemExit(main())
