#!/usr/bin/env python3
"""Exact-wheel rc8 public JX CPU versus REBOUND WHFast screen.

Rc8 intentionally retains the supported rc7 numerical kernel.  This runner
gives the newly built wheel its own prospective timing protocol and report
identity while reusing the already audited rc6/rc7 fixture and gate machinery.
It refuses editable-source execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jxplanetx as jx_package

from benchmarks import jx_fast_wisdom_holman_rc7_rebound_race as parent


SCHEMA = "jxplanetx.fast-wh-rc8-rebound-race.report.v1"
PROTOCOL_SCHEMA = "jxplanetx.fast-wh-rc8-rebound-race.protocol.v1"
BENCHMARK_ID = "jx.fast-wh-rc8.rebound-whfast.100-period.v1"
REQUIRED_JX_VERSION = "0.6.0rc8"
REQUIRED_WHEEL = {
    "filename": "jxplanetx-0.6.0rc8-cp314-cp314-linux_x86_64.whl",
    "size_bytes": 847708,
    "sha256": "bb9ebbb81c8b847a61d947d78c47c3f78d1c4a67f0336b1a8ae0a8756ea52af5",
}
LANE_NAMES = ("jx_rc8_single_pass", "jx_rc8_replay", "rebound_whfast")
STATUS = "LOCKED_AFTER_DISCLOSED_RESULTS_BEFORE_EXACT_RC8_WHEEL_TIMING"


class RC8RaceError(RuntimeError):
    """The exact-wheel rc8 screen failed closed."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _relabel(value: Any) -> Any:
    if type(value) is dict:
        return {
            str(key).replace("jx_rc7", "jx_rc8"): _relabel(item)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_relabel(item) for item in value]
    if type(value) is str:
        return value.replace("jx_rc7", "jx_rc8")
    return value


def read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RC8RaceError("protocol is not valid UTF-8 JSON") from exc
    fixed = {
        "schema": PROTOCOL_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": STATUS,
        "periods": 100,
        "step_divisor": 64,
        "samples_per_period": 4,
        "timed_repetitions": 9,
        "native_replay_count": 1,
        "speed_resolution_fraction": 0.05,
        "required_jx_version": REQUIRED_JX_VERSION,
        "required_rebound_version": parent.base.REQUIRED_REBOUND_VERSION,
        "required_rebound_githash": parent.base.REQUIRED_REBOUND_GITHASH,
        "required_release_wheel": REQUIRED_WHEEL,
    }
    if type(protocol) is not dict or any(
        protocol.get(name) != value for name, value in fixed.items()
    ):
        raise RC8RaceError("protocol fixed contract changed")
    if protocol.get("lane_names") != list(LANE_NAMES):
        raise RC8RaceError("protocol lane roster changed")
    orders = protocol.get("execution_order")
    if (
        type(orders) is not list
        or len(orders) != 9
        or any(
            type(order) is not list
            or len(order) != len(LANE_NAMES)
            or set(order) != set(LANE_NAMES)
            for order in orders
        )
    ):
        raise RC8RaceError("protocol execution order changed")
    if protocol.get("accuracy_and_conservation_gate") != dict(
        parent.base.fixture.LONG_DESCRIPTIVE_ENVELOPE
    ):
        raise RC8RaceError("protocol accuracy gate changed")
    roster = protocol.get("source_roster")
    if type(roster) is not list or not roster:
        raise RC8RaceError("protocol source roster is missing")
    for expected in roster:
        if parent._file_identity(root / expected["path"], root) != expected:
            raise RC8RaceError(
                f"protocol source binding changed: {expected.get('path')}"
            )
    return protocol, {
        "path": path.resolve(strict=True).as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    parent.SCHEMA = SCHEMA
    parent.BENCHMARK_ID = BENCHMARK_ID
    parent.REQUIRED_JX_VERSION = REQUIRED_JX_VERSION
    converted = dict(protocol)
    converted["execution_order"] = [
        [name.replace("jx_rc8", "jx_rc7") for name in order]
        for order in protocol["execution_order"]
    ]
    return _relabel(parent.run(converted))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--release-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, protocol_identity = read_protocol(args.protocol, root)
    package_path = Path(jx_package.__file__).resolve(strict=True)
    if package_path.is_relative_to(root):
        raise RC8RaceError(
            "final rc8 race requires an installed wheel, not editable source"
        )
    wheel_identity = parent.base.verify_release_wheel(
        args.release_wheel, REQUIRED_WHEEL
    )
    report = run(protocol)
    report["protocol"] = protocol_identity
    report["release_wheel"] = wheel_identity
    report["installed_package_path"] = package_path.as_posix()
    report["protocol_prior_evidence_disclosure"] = protocol[
        "prior_evidence_disclosure"
    ]
    report["execution_order"] = protocol["execution_order"]
    report["semantic_content_sha256"] = _sha256(
        b"jx.fast-wh-rc8-rebound-race.report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
