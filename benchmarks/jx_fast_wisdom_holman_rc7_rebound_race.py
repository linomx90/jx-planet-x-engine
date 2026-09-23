#!/usr/bin/env python3
"""Exact-wheel rc7 public JX CPU versus REBOUND WHFast screen.

This runner reuses the already audited numerical and accuracy machinery from
the immutable rc6 runner while giving rc7 its own protocol, wheel identity,
lane labels, source binding, and report domain.  It must execute against an
installed wheel outside the repository, never the editable source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import jxplanetx as jx_package

from benchmarks import jx_fast_wisdom_holman_rc6_rebound_race as base


SCHEMA = "jxplanetx.fast-wh-rc7-rebound-race.report.v1"
PROTOCOL_SCHEMA = "jxplanetx.fast-wh-rc7-rebound-race.protocol.v1"
BENCHMARK_ID = "jx.fast-wh-rc7.rebound-whfast.100-period.v1"
REQUIRED_JX_VERSION = "0.6.0rc7"
REQUIRED_WHEEL = {
    "filename": "jxplanetx-0.6.0rc7-cp314-cp314-linux_x86_64.whl",
    "size_bytes": 847613,
    "sha256": "17663fa6c971be36be5c702766d65be604a6a23d11860768d52c6d21df7a54b4",
}
LANE_NAMES = (
    "jx_rc7_single_pass",
    "jx_rc7_replay",
    "rebound_whfast",
)
_INTERNAL_LANES = (
    "jx_rc6_single_pass",
    "jx_rc6_replay",
    "rebound_whfast",
)
_TO_INTERNAL = dict(zip(LANE_NAMES, _INTERNAL_LANES, strict=True))
STATUS = "LOCKED_AFTER_DISCLOSED_RESULTS_BEFORE_EXACT_RC7_WHEEL_TIMING"


class RC7RaceError(RuntimeError):
    """The exact-wheel rc7 screen failed closed."""


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


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise RC7RaceError(f"required source is not regular: {path}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise RC7RaceError(f"required source escapes root: {path}") from exc
    raw = resolved.read_bytes()
    if not raw:
        raise RC7RaceError(f"required source is empty: {path}")
    return {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}


def read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RC7RaceError("protocol is not valid UTF-8 JSON") from exc
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
        "required_rebound_version": base.REQUIRED_REBOUND_VERSION,
        "required_rebound_githash": base.REQUIRED_REBOUND_GITHASH,
        "required_release_wheel": REQUIRED_WHEEL,
    }
    if type(protocol) is not dict or any(
        protocol.get(name) != value for name, value in fixed.items()
    ):
        raise RC7RaceError("protocol fixed contract changed")
    if protocol.get("lane_names") != list(LANE_NAMES):
        raise RC7RaceError("protocol lane roster changed")
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
        raise RC7RaceError("protocol execution order changed")
    if protocol.get("accuracy_and_conservation_gate") != dict(
        base.fixture.LONG_DESCRIPTIVE_ENVELOPE
    ):
        raise RC7RaceError("protocol accuracy gate changed")
    roster = protocol.get("source_roster")
    if type(roster) is not list or not roster:
        raise RC7RaceError("protocol source roster is missing")
    for expected in roster:
        if _file_identity(root / expected["path"], root) != expected:
            raise RC7RaceError(
                f"protocol source binding changed: {expected.get('path')}"
            )
    return protocol, {
        "path": path.resolve(strict=True).as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _rc7_labels(value: Any) -> Any:
    if type(value) is dict:
        return {
            str(key).replace("jx_rc6", "jx_rc7"): _rc7_labels(item)
            for key, item in value.items()
        }
    if type(value) is list:
        return [_rc7_labels(item) for item in value]
    if type(value) is str:
        return value.replace("jx_rc6", "jx_rc7")
    return value


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    base.SCHEMA = SCHEMA
    base.BENCHMARK_ID = BENCHMARK_ID
    base.REQUIRED_JX_VERSION = REQUIRED_JX_VERSION
    internal_orders = tuple(
        tuple(_TO_INTERNAL[name] for name in order)
        for order in protocol["execution_order"]
    )
    report = base.run(
        periods=protocol["periods"],
        step_divisor=protocol["step_divisor"],
        repetitions=protocol["timed_repetitions"],
        orders=internal_orders,
        require_public_parity=True,
    )
    return _rc7_labels(report)


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
        raise RC7RaceError(
            "final rc7 race requires an installed wheel, not editable source"
        )
    wheel_identity = base.verify_release_wheel(
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
        b"jx.fast-wh-rc7-rebound-race.report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
