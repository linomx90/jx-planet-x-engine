#!/usr/bin/env python3
"""Export the evidence-aware JX General Dynamics capability registry."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from jxplanetx.general_dynamics import registry_summary, verify_evidence


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def build_report(repository_root: Path) -> dict[str, object]:
    checks = verify_evidence(repository_root)
    if not all(checks.values()):
        failed = sorted(key for key, valid in checks.items() if not valid)
        raise RuntimeError(f"evidence verification failed: {failed}")
    report = registry_summary()
    report["evidence_verification"] = checks
    report["all_bound_evidence_verified"] = True
    return report


def write_new(path: Path, payload: bytes) -> None:
    parent = path.parent.resolve(strict=True)
    if parent.is_symlink():
        raise RuntimeError("output parent must not be a symlink")
    target = parent / path.name
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    write_new(arguments.output, canonical(build_report(repository_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
