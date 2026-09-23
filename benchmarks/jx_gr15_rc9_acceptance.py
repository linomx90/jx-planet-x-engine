#!/usr/bin/env python3
"""Exact-artifact GR15 acceptance for the JX 0.6.0rc9 candidate.

Run this script with the interpreter from a fresh installation of the locked
wheel.  The repository supplies only the source-bound qualification harnesses;
the imported ``jxplanetx`` package must come from that fresh environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
from typing import Any
import zipfile


SCHEMA = "jxplanetx.gr15-rc9.artifact-acceptance-report.v1"
PROTOCOL_SCHEMA = "jxplanetx.gr15-rc9.artifact-acceptance-protocol.v1"
BENCHMARK_ID = "jx.gr15-v3.rc9.exact-artifact-acceptance.v1"


class RC9AcceptanceError(RuntimeError):
    """The exact rc9 artifact acceptance boundary failed closed."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _path_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise RC9AcceptanceError(f"required artifact is not a regular file: {path}")
    raw = resolved.read_bytes()
    return {
        "filename": resolved.name,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _source_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise RC9AcceptanceError(f"required source is not regular: {path}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise RC9AcceptanceError(f"required source escapes repository: {path}") from exc
    raw = resolved.read_bytes()
    return {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}


def _read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RC9AcceptanceError("protocol is not valid UTF-8 JSON") from exc
    if (
        type(protocol) is not dict
        or protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("benchmark_id") != BENCHMARK_ID
        or protocol.get("status") != "LOCKED_BEFORE_FIRST_EXACT_RC9_EXECUTION"
        or protocol.get("scientific_claim_state") != "SCREENING_ONLY"
        or protocol.get("package_version") != "0.6.0rc9"
        or protocol.get("rebound_version") != "5.1.1"
        or protocol.get("rebound_githash")
        != "33549d1d50d616a95a6d6a79e5e2c9c3b3730b1f"
        or protocol.get("gr15_core_source_sha256")
        != "036d1dc7e4e2e26c908c6dde9ff27bdf6a5804b88aa3f3e374f1f512642c1e41"
        or protocol.get("claim_controls")
        != {
            "general_superiority_claim_authorized": False,
            "production_ephemeris_claim_authorized": False,
            "publication_authorized_by_this_report": False,
            "screening_only": True,
        }
    ):
        raise RC9AcceptanceError("fixed rc9 acceptance protocol changed")
    roster = protocol.get("source_roster")
    if type(roster) is not list or not roster:
        raise RC9AcceptanceError("source roster is missing")
    for expected in roster:
        if _source_identity(root / expected["path"], root) != expected:
            raise RC9AcceptanceError(
                f"source binding changed: {expected.get('path')}"
            )
    return protocol, {
        "path": resolved.as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _urlsafe_sha256(raw: bytes) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode("ascii")
    return encoded.rstrip("=")


def _verify_installed_wheel(wheel: Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual = _path_identity(wheel)
    if actual != expected:
        raise RC9AcceptanceError("wheel identity differs from the locked artifact")
    distribution = importlib.metadata.distribution("jxplanetx")
    if distribution.version != "0.6.0rc9":
        raise RC9AcceptanceError("installed distribution version is not rc9")
    checked = 0
    with zipfile.ZipFile(wheel, "r") as archive:
        records = [name for name in archive.namelist() if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RC9AcceptanceError("wheel must contain exactly one RECORD")
        rows = csv.reader(io.StringIO(archive.read(records[0]).decode("utf-8")))
        for relative, digest, size in rows:
            if not digest:
                continue
            if not digest.startswith("sha256=") or not size.isdigit():
                raise RC9AcceptanceError("wheel RECORD uses an unsupported digest")
            wheel_bytes = archive.read(relative)
            expected_digest = digest.removeprefix("sha256=")
            if (
                len(wheel_bytes) != int(size)
                or _urlsafe_sha256(wheel_bytes) != expected_digest
            ):
                raise RC9AcceptanceError(f"wheel member digest failed: {relative}")
            installed = Path(distribution.locate_file(relative)).resolve(strict=True)
            installed_bytes = installed.read_bytes()
            if (
                len(installed_bytes) != int(size)
                or _urlsafe_sha256(installed_bytes) != expected_digest
            ):
                raise RC9AcceptanceError(
                    f"installed file differs from exact wheel: {relative}"
                )
            checked += 1
    if checked < 20:
        raise RC9AcceptanceError("too few installed wheel members were verified")
    return {
        **actual,
        "installed_distribution_root": str(Path(distribution.locate_file(".")).resolve()),
        "record_hashed_files_verified": checked,
    }


def _load_base_protocol(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    actual = _source_identity(path, path.resolve().parents[1])
    if actual != expected:
        raise RC9AcceptanceError(f"base qualification protocol changed: {path.name}")
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RC9AcceptanceError(f"base protocol is invalid: {path.name}") from exc
    if result.get("scientific_claim_state") != "SCREENING_ONLY":
        raise RC9AcceptanceError("base protocol claim state changed")
    result["package_version"] = "0.6.0rc9"
    return result


def run(
    protocol: dict[str, Any],
    root: Path,
    wheel: Path,
    sdist: Path,
) -> dict[str, Any]:
    wheel_record = _verify_installed_wheel(wheel, protocol["artifacts"]["wheel"])
    sdist_record = _path_identity(sdist)
    if sdist_record != protocol["artifacts"]["sdist"]:
        raise RC9AcceptanceError("source archive identity differs from the lock")

    source_root = (root / "src").resolve()
    import jxplanetx
    from jxplanetx.gr15 import gr15_runtime_identity

    package_path = Path(jxplanetx.__file__).resolve(strict=True)
    if package_path.is_relative_to(source_root):
        raise RC9AcceptanceError("acceptance imported the source tree, not the wheel")
    identity = gr15_runtime_identity()
    if (
        jxplanetx.__version__ != protocol["package_version"]
        or identity["method_id"] != "JX_GAUSS_RADAU15_V3"
        or identity["source_sha256"] != protocol["gr15_core_source_sha256"]
        or identity["scientific_claim_state"] != "SCREENING_ONLY"
    ):
        raise RC9AcceptanceError("installed public GR15 identity changed")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from benchmarks import jx_gr15_ias15 as direct
    from benchmarks import jx_gr15_v3_adversarial as adversarial

    direct_protocol = _load_base_protocol(
        root / protocol["base_protocols"]["direct"]["path"],
        protocol["base_protocols"]["direct"],
    )
    adversarial_protocol = _load_base_protocol(
        root / protocol["base_protocols"]["adversarial"]["path"],
        protocol["base_protocols"]["adversarial"],
    )
    direct_report = direct.run(direct_protocol)
    adversarial_report = adversarial.run(adversarial_protocol)
    passed = (
        direct_report["status"] == "PASS_SCREENING_ONLY"
        and adversarial_report["status"] == "PASS_SCREENING_ONLY"
    )
    return {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": "PASS_SCREENING_ONLY" if passed else "FAIL_FIXED_GATE",
        "scientific_claim_state": "SCREENING_ONLY",
        "claim_controls": protocol["claim_controls"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "package_path": str(package_path),
            "jxplanetx": jxplanetx.__version__,
            "gr15": identity,
        },
        "artifacts": {"wheel": wheel_record, "sdist": sdist_record},
        "direct_ias15": direct_report,
        "adversarial": adversarial_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, protocol_identity = _read_protocol(args.protocol, root)
    if args.validate_only:
        print(json.dumps(protocol_identity, sort_keys=True))
        return 0
    report = run(protocol, root, args.wheel, args.sdist)
    report["protocol"] = protocol_identity
    report["semantic_content_sha256"] = _sha256(
        b"jx.gr15-rc9.artifact-acceptance-report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"] == "PASS_SCREENING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
