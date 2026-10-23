#!/usr/bin/env python3
"""Source-bound input constructor for the registered V9 hierarchical holdout.

Importing this module is deliberately standard-library only.  The beacon and
input-manifest commands cannot import NumPy, JX, SciPy, or REBOUND.  Candidate
and oracle execution lives in a separately registered program.

This program never fetches the NIST beacon.  It accepts exact response bytes
retained by an independent administrator, creates canonical artifacts with
``O_EXCL``, and has no overwrite or case-replacement mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import sysconfig
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "runs/jx_wh_v9_hierarchical_holdout_v1/protocol_v1.json"
PROTOCOL_REGISTRATION_COMMIT = "23936a97adc76419216d83fd2f45b79b7b7e12be"
PROTOCOL_RAW_SHA256 = "b85d7999d49efc18622c2df4493b44029998136b1413af807efce42aaae7cb32"
PROTOCOL_RAW_SIZE = 66090
PROTOCOL_CONTENT_SHA256 = "c37e1b8360e4b98601049df5abaaa8ec52c9ac4ae3de63272bf39bfe26da4540"
PROTOCOL_CANONICAL_SHA256 = "404de5f21b7061b07ea8768292c167b03a0997f8625d73d07914e5886111f327"
PROTOCOL_CANONICAL_SIZE = 59483

BEACON_SCHEMA = "jxplanetx.wisdom-holman-v9-hierarchical-holdout-beacon-binding.v1"
BEACON_DOMAIN = "jxplanetx.wh-v9-holdout.beacon-binding.v1"
MANIFEST_SCHEMA = "jxplanetx.wisdom-holman-v9-hierarchical-holdout-inputs.v1"
MANIFEST_DOMAIN = "jxplanetx.wh-v9-holdout.input-manifest.v1"
CASE_DOMAIN = "jxplanetx.wh-v9-holdout.case-input.v1"
CONTROL_DOMAIN = "jxplanetx.wh-v9-holdout.guard-control-input.v1"
CUMULATIVE_GM_DOMAIN = "jxplanetx.wh-v9-holdout.cumulative-gm.v1"
SOURCE_REGISTRATION_SCHEMA = "jxplanetx.wisdom-holman-v9-hierarchical-holdout-source-registration.v1"
SOURCE_REGISTRATION_DOMAIN = "jxplanetx.wh-v9-holdout.source-registration.v1"
MPMATH_TREE_DOMAIN = "jxplanetx.wh-v9-holdout.mpmath-package-tree.v1"
SOURCE_REGISTRATION_PATH = ROOT / "runs/jx_wh_v9_hierarchical_holdout_v1/source_registration_v1.json"
BEACON_RAW_PATH = ROOT / "runs/jx_wh_v9_hierarchical_holdout_v1/beacon_raw_v1.json"
BEACON_BINDING_PATH = ROOT / "runs/jx_wh_v9_hierarchical_holdout_v1/beacon_binding_v1.json"
INPUT_MANIFEST_PATH = ROOT / "runs/jx_wh_v9_hierarchical_holdout_v1/input_manifest_v1.json"
RUNNER_PATH = "benchmarks/jx_wh_v9_hierarchical_holdout.py"
RUNNER_TEST_PATH = "tests/test_jx_wh_v9_hierarchical_holdout.py"
EXECUTOR_PATH = "benchmarks/jx_wh_v9_hierarchical_holdout_execute.py"
EXECUTOR_TEST_PATH = "tests/test_jx_wh_v9_hierarchical_holdout_execute.py"
PROTOCOL_TEST_PATH = "tests/test_jx_wh_v9_hierarchical_holdout_protocol.py"

MAX_JSON_BYTES = 128 * 1024 * 1024
MAX_BEACON_BYTES = 4 * 1024 * 1024
MAX_JSON_DEPTH = 96
MAX_JSON_NODES = 2_000_000
FORBIDDEN_PRE_FREEZE_MODULES = ("jxplanetx", "scipy", "rebound")
EPSILON = 2.220446049250313e-16
PI = float(3.141592653589793238462643383279502884)


class HoldoutError(RuntimeError):
    """The frozen holdout contract could not be satisfied."""


class CalibrationUnavailable(HoldoutError):
    """A registered calibration has no selectable binary64 candidate."""

    def __init__(self, message: str, branches: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.branches = branches


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    try:
        raw = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise HoldoutError("value is not canonical ASCII JSON") from exc
    return raw + (b"\n" if newline else b"")


def _domain_hash(domain: str, value: Any) -> str:
    return _sha256(domain.encode("ascii") + b"\0" + _canonical_bytes(value))


def _framed_hash(domain: str, identifier: str, value: Any) -> str:
    return _sha256(
        domain.encode("ascii")
        + b"\0"
        + identifier.encode("ascii")
        + b"\0"
        + _canonical_bytes(value)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise HoldoutError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _validate_json_tree(value: Any) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_JSON_NODES or depth > MAX_JSON_DEPTH:
            raise HoldoutError("JSON resource limits exceeded")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise HoldoutError("JSON object key is not a string")
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise HoldoutError("unsupported JSON value type")
        elif type(item) is float and not math.isfinite(item):
            raise HoldoutError("nonfinite JSON number")


def _read_regular(
    path: Path, maximum: int, label: str, *, allow_empty: bool = False
) -> bytes:
    if not isinstance(path, Path):
        raise HoldoutError(f"{label} path type changed")
    descriptor: int | None = None
    parent_fd: int | None = None
    try:
        parent_fd, leaf = _open_parent_nofollow(path)
        descriptor = os.open(
            leaf,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < (0 if allow_empty else 1)
            or before.st_size > maximum
        ):
            raise HoldoutError(f"{label} must be a bounded regular file")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise HoldoutError(f"{label} changed while reading")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise HoldoutError(f"{label} grew while reading")
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise HoldoutError(f"{label} changed while reading")
        return b"".join(chunks)
    except OSError as exc:
        raise HoldoutError(f"{label} is unavailable") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if parent_fd is not None:
            os.close(parent_fd)


def _parse_json(data: bytes, label: str) -> Any:
    try:
        value = json.loads(data.decode("ascii"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise HoldoutError(f"{label} is not strict ASCII JSON") from exc
    _validate_json_tree(value)
    return value


def _load_canonical_json(path: Path, maximum: int, label: str) -> tuple[dict[str, Any], bytes]:
    data = _read_regular(path, maximum, label)
    value = _parse_json(data, label)
    if type(value) is not dict or data not in (
        _canonical_bytes(value),
        _canonical_bytes(value, newline=True),
    ):
        raise HoldoutError(f"{label} is not canonical sorted JSON")
    return value, data


def _protocol() -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(PROTOCOL_PATH, PROTOCOL_RAW_SIZE, "registered protocol")
    if len(raw) != PROTOCOL_RAW_SIZE or _sha256(raw) != PROTOCOL_RAW_SHA256:
        raise HoldoutError("registered revision-5 protocol bytes changed")
    value = _parse_json(raw, "registered protocol")
    if (
        type(value) is not dict
        or value.get("protocol_revision") != 5
        or value.get("holdout_id") != "JX_WH_V9_HIERARCHICAL_HOLDOUT_V1"
        or value.get("content_integrity", {}).get("content_sha256")
        != PROTOCOL_CONTENT_SHA256
    ):
        raise HoldoutError("registered revision-5 protocol identity changed")
    content = dict(value)
    content.pop("content_integrity", None)
    if _domain_hash(value["content_integrity"]["domain"], content) != PROTOCOL_CONTENT_SHA256:
        raise HoldoutError("registered protocol content digest changed")
    canonical = _canonical_bytes(value)
    if len(canonical) != PROTOCOL_CANONICAL_SIZE or _sha256(canonical) != PROTOCOL_CANONICAL_SHA256:
        raise HoldoutError("registered protocol canonical bytes changed")
    return value, canonical


def _open_parent_nofollow(path: Path) -> tuple[int, str]:
    absolute = os.path.abspath(os.fspath(path))
    parent, leaf = os.path.split(absolute)
    if not leaf or "\0" in absolute:
        raise HoldoutError("output path is invalid")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.path.sep, flags)
    try:
        for component in Path(parent).parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor, leaf
    except BaseException:
        os.close(descriptor)
        raise


def _write_new_file(path: Path, data: bytes) -> None:
    if not isinstance(path, Path) or type(data) is not bytes:
        raise HoldoutError("output path/data type changed")
    parent_fd: int | None = None
    output_fd: int | None = None
    created: tuple[int, int] | None = None
    leaf = ""
    try:
        parent_fd, leaf = _open_parent_nofollow(path)
        try:
            os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise HoldoutError("output path must not exist")
        output_fd = os.open(
            leaf,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        opened = os.fstat(output_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise OSError("created output is not regular")
        created = (opened.st_dev, opened.st_ino)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            written = os.write(output_fd, view[offset:])
            if written <= 0:
                raise OSError("short output write")
            offset += written
        os.fsync(output_fd)
        os.fsync(parent_fd)
        retained = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        if (
            (retained.st_dev, retained.st_ino) != created
            or retained.st_size != len(data)
        ):
            raise OSError("output identity changed")
    except HoldoutError:
        raise
    except OSError as exc:
        if parent_fd is not None and created is not None:
            try:
                retained = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
                if (retained.st_dev, retained.st_ino) == created:
                    os.unlink(leaf, dir_fd=parent_fd)
                    os.fsync(parent_fd)
            except OSError:
                pass
        raise HoldoutError("create-only artifact publication failed") from exc
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if parent_fd is not None:
            os.close(parent_fd)


def _assert_pre_freeze_import_boundary() -> None:
    found = sorted(
        name
        for name in sys.modules
        if any(name == root or name.startswith(root + ".") for root in FORBIDDEN_PRE_FREEZE_MODULES)
    )
    if found:
        raise HoldoutError(
            "candidate/oracle module was imported before manifest freeze: " + found[0]
        )


def _content_record(path: Path, maximum: int, label: str) -> dict[str, Any]:
    data = _read_regular(path, maximum, label)
    return {"path": path.as_posix(), "size_bytes": len(data), "sha256": _sha256(data)}


def _validate_content_integrity(
    value: dict[str, Any],
    expected_schema: str | None = None,
    expected_domain: str | None = None,
) -> None:
    if expected_schema is not None and value.get("schema") != expected_schema:
        raise HoldoutError("artifact schema changed")
    integrity = value.get("content_integrity")
    if (
        type(integrity) is not dict
        or set(integrity) != {"algorithm", "canonicalization", "domain", "content_sha256"}
        or integrity["algorithm"] != "SHA-256"
        or integrity["canonicalization"]
        != "SORTED_KEYS_COMPACT_ASCII_EXCLUDING_CONTENT_INTEGRITY"
        or not _is_sha256(integrity["content_sha256"])
        or type(integrity["domain"]) is not str
        or (expected_domain is not None and integrity["domain"] != expected_domain)
    ):
        raise HoldoutError("artifact content-integrity schema changed")
    core = dict(value)
    core.pop("content_integrity")
    if _domain_hash(integrity["domain"], core) != integrity["content_sha256"]:
        raise HoldoutError("artifact content digest changed")


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        capture_output=True,
        text=not binary,
        check=False,
    )
    if completed.returncode:
        raise HoldoutError("Git custody command failed: " + " ".join(arguments))
    return completed.stdout if binary else completed.stdout.strip()


def _source_registration(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.resolve(strict=False) != SOURCE_REGISTRATION_PATH.resolve(strict=False):
        raise HoldoutError("source registration must use the fixed repository path")
    value, raw = _load_canonical_json(path, MAX_JSON_BYTES, "source registration")
    _validate_content_integrity(
        value, SOURCE_REGISTRATION_SCHEMA, SOURCE_REGISTRATION_DOMAIN
    )
    expected_keys = {
        "schema", "holdout_id", "protocol", "source_freeze", "bound_files",
        "constructor_dependency", "routes", "claim_controls", "content_integrity",
    }
    protocol_record = {
        "registration_commit": PROTOCOL_REGISTRATION_COMMIT,
        "raw_size_bytes": PROTOCOL_RAW_SIZE,
        "raw_sha256": PROTOCOL_RAW_SHA256,
        "content_sha256": PROTOCOL_CONTENT_SHA256,
        "canonical_size_bytes": PROTOCOL_CANONICAL_SIZE,
        "canonical_sha256": PROTOCOL_CANONICAL_SHA256,
    }
    routes = {
        "source_registration": SOURCE_REGISTRATION_PATH.relative_to(ROOT).as_posix(),
        "beacon_raw": BEACON_RAW_PATH.relative_to(ROOT).as_posix(),
        "beacon_binding": BEACON_BINDING_PATH.relative_to(ROOT).as_posix(),
        "input_manifest": INPUT_MANIFEST_PATH.relative_to(ROOT).as_posix(),
    }
    controls = {
        "registered_before_beacon_pulse": True,
        "constructor_source_change_after_pulse_allowed": False,
        "executor_source_change_after_pulse_allowed": False,
        "candidate_executions_before_registration": 0,
        "oracle_executions_before_registration": 0,
        "scientific_outcomes_generated": False,
    }
    if (
        set(value) != expected_keys
        or value["holdout_id"] != "JX_WH_V9_HIERARCHICAL_HOLDOUT_V1"
        or value["protocol"] != protocol_record
        or value["routes"] != routes
        or value["claim_controls"] != controls
    ):
        raise HoldoutError("source-registration contract changed")
    _validate_mpmath_registration(value["constructor_dependency"])
    freeze = value["source_freeze"]
    if type(freeze) is not dict or set(freeze) != {"commit", "tree", "parent"} or any(
        type(freeze[name]) is not str or len(freeze[name]) != 40
        or any(character not in "0123456789abcdef" for character in freeze[name])
        for name in ("commit", "tree", "parent")
    ):
        raise HoldoutError("source-freeze Git identity is invalid")
    if _git("show", "-s", "--format=%T", freeze["commit"]) != freeze["tree"] or _git(
        "show", "-s", "--format=%P", freeze["commit"]
    ).split() != [freeze["parent"]]:
        raise HoldoutError("source-freeze Git tree/parent binding changed")
    ancestor = subprocess.run(
        ("git", "merge-base", "--is-ancestor", PROTOCOL_REGISTRATION_COMMIT, freeze["commit"]),
        cwd=ROOT,
        check=False,
    )
    if ancestor.returncode != 0:
        raise HoldoutError("source freeze does not descend from revision-5 protocol")
    bound = value["bound_files"]
    expected_bound = {
        "runner": RUNNER_PATH,
        "runner_test": RUNNER_TEST_PATH,
        "executor": EXECUTOR_PATH,
        "executor_test": EXECUTOR_TEST_PATH,
        "protocol_test": PROTOCOL_TEST_PATH,
    }
    if type(bound) is not dict or set(bound) != set(expected_bound):
        raise HoldoutError("source-registration bound-file roster changed")
    for key, relative in expected_bound.items():
        item = bound[key]
        if type(item) is not dict or set(item) != {
            "path", "size_bytes", "sha256", "git_blob_sha1"
        } or item["path"] != relative or type(item["size_bytes"]) is not int \
            or item["size_bytes"] <= 0 or not _is_sha256(item["sha256"]) \
            or type(item["git_blob_sha1"]) is not str or len(item["git_blob_sha1"]) != 40:
            raise HoldoutError(f"source-registration record changed: {key}")
        committed = _git("show", f"{freeze['commit']}:{relative}", binary=True)
        if type(committed) is not bytes:
            raise HoldoutError("Git returned non-binary bound-source bytes")
        current = _read_regular(ROOT / relative, MAX_JSON_BYTES, f"bound source {relative}")
        blob = _git("rev-parse", f"{freeze['commit']}:{relative}")
        if (
            committed != current
            or len(committed) != item["size_bytes"]
            or _sha256(committed) != item["sha256"]
            or blob != item["git_blob_sha1"]
        ):
            raise HoldoutError(f"bound source differs from freeze: {relative}")
    registration_relative = SOURCE_REGISTRATION_PATH.relative_to(ROOT).as_posix()
    registration_commit = _git("log", "-1", "--format=%H", "--", registration_relative)
    if type(registration_commit) is not str or len(registration_commit) != 40:
        raise HoldoutError("source registration is not committed")
    committed_registration = _git("show", f"{registration_commit}:{registration_relative}", binary=True)
    if committed_registration != raw:
        raise HoldoutError("source registration differs from its publication commit")
    commit_seconds = int(_git("show", "-s", "--format=%ct", registration_commit))
    freeze_seconds = int(_git("show", "-s", "--format=%ct", freeze["commit"]))
    if max(commit_seconds, freeze_seconds) * 1000 >= 1792728000000:
        raise HoldoutError("source registration was not committed before beacon pulse")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", freeze["commit"], registration_commit),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise HoldoutError("source registration does not descend from source freeze")
    head = _git("rev-parse", "HEAD")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", registration_commit, head),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise HoldoutError("source-registration commit is not an ancestor of HEAD")
    record = {
        "path": registration_relative,
        "publication_commit": registration_commit,
        "source_freeze_commit": freeze["commit"],
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "content_sha256": value["content_integrity"]["content_sha256"],
    }
    return value, record


def _pulse_object(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise HoldoutError("beacon response root is invalid")
    pulse = value.get("pulse")
    if type(pulse) is not dict:
        raise HoldoutError("beacon response has no pulse object")
    return pulse


def create_beacon_binding(raw_response: Path, source_registration: Path) -> dict[str, Any]:
    _assert_pre_freeze_import_boundary()
    if raw_response.resolve(strict=False) != BEACON_RAW_PATH.resolve(strict=False):
        raise HoldoutError("beacon response must use the registered fixed path")
    protocol, canonical_protocol = _protocol()
    _, source_record = _source_registration(source_registration)
    raw = _read_regular(raw_response, MAX_BEACON_BYTES, "NIST beacon raw response")
    response = _parse_json(raw, "NIST beacon raw response")
    pulse = _pulse_object(response)
    registered = protocol["future_public_randomness"]
    chain_index = pulse.get("chainIndex")
    timestamp = pulse.get("timeStamp")
    output_value = pulse.get("outputValue")
    status_code = pulse.get("statusCode")
    if type(chain_index) is str and chain_index.isascii() and chain_index.isdecimal():
        chain_index = int(chain_index)
    if chain_index != registered["chain_index"] or timestamp != registered["pulse_timestamp_utc"] or type(status_code) is not int or status_code != 0:
        raise HoldoutError("beacon chain or timestamp differs from registration")
    if (
        type(output_value) is not str
        or len(output_value) != 128
        or any(character not in "0123456789abcdefABCDEF" for character in output_value)
    ):
        raise HoldoutError("beacon outputValue is not exact 512-bit hexadecimal")
    try:
        output_bytes = bytes.fromhex(output_value)
    except ValueError as exc:
        raise HoldoutError("beacon outputValue decoding failed") from exc
    seed_domain = protocol["public_case_derivation"]["seed_domain"]
    seed_sha256 = _sha256(
        seed_domain.encode("ascii") + b"\0" + canonical_protocol + b"\0" + output_bytes
    )
    core = {
        "schema": BEACON_SCHEMA,
        "holdout_id": protocol["holdout_id"],
        "protocol": {
            "registration_commit": PROTOCOL_REGISTRATION_COMMIT,
            "raw_size_bytes": PROTOCOL_RAW_SIZE,
            "raw_sha256": PROTOCOL_RAW_SHA256,
            "content_sha256": PROTOCOL_CONTENT_SHA256,
            "canonical_size_bytes": len(canonical_protocol),
            "canonical_sha256": _sha256(canonical_protocol),
        },
        "source_registration": source_record,
        "beacon": {
            "provider": registered["provider"],
            "chain_index": chain_index,
            "pulse_timestamp_utc": timestamp,
            "pulse_timestamp_unix_milliseconds": registered["pulse_timestamp_unix_milliseconds"],
            "api_path": registered["api_path"],
            "raw_response_path": BEACON_RAW_PATH.relative_to(ROOT).as_posix(),
            "raw_response_size_bytes": len(raw),
            "raw_response_sha256": _sha256(raw),
            "output_value": output_value,
            "status_code": status_code,
        },
        "seed_sha256": seed_sha256,
        "claim_controls": {
            "candidate_or_oracle_imported": False,
            "candidate_or_oracle_executed": False,
            "scientific_outcome_generated": False,
        },
    }
    return core | {"content_integrity": {"algorithm": "SHA-256", "canonicalization": "SORTED_KEYS_COMPACT_ASCII_EXCLUDING_CONTENT_INTEGRITY", "domain": BEACON_DOMAIN, "content_sha256": _domain_hash(BEACON_DOMAIN, core)}}


def validate_beacon_binding(value: Any, source_registration: Path) -> dict[str, Any]:
    if type(value) is not dict:
        raise HoldoutError("beacon binding must be an object")
    _validate_content_integrity(value, BEACON_SCHEMA, BEACON_DOMAIN)
    protocol, canonical = _protocol()
    _, source_record = _source_registration(source_registration)
    expected_protocol = {
        "registration_commit": PROTOCOL_REGISTRATION_COMMIT,
        "raw_size_bytes": PROTOCOL_RAW_SIZE,
        "raw_sha256": PROTOCOL_RAW_SHA256,
        "content_sha256": PROTOCOL_CONTENT_SHA256,
        "canonical_size_bytes": len(canonical),
        "canonical_sha256": _sha256(canonical),
    }
    beacon = value.get("beacon")
    if (
        set(value) != {
            "schema", "holdout_id", "protocol", "source_registration",
            "beacon", "seed_sha256", "claim_controls", "content_integrity",
        }
        or
        value.get("holdout_id") != protocol["holdout_id"]
        or value.get("protocol") != expected_protocol
        or value.get("source_registration") != source_record
        or value.get("claim_controls")
        != {"candidate_or_oracle_imported": False, "candidate_or_oracle_executed": False, "scientific_outcome_generated": False}
        or type(beacon) is not dict
        or set(beacon) != {
            "provider", "chain_index", "pulse_timestamp_utc",
            "pulse_timestamp_unix_milliseconds", "api_path",
            "raw_response_path", "raw_response_size_bytes",
            "raw_response_sha256", "output_value", "status_code",
        }
        or beacon.get("provider") != protocol["future_public_randomness"]["provider"]
        or beacon.get("chain_index") != 2
        or beacon.get("pulse_timestamp_utc") != protocol["future_public_randomness"]["pulse_timestamp_utc"]
        or beacon.get("pulse_timestamp_unix_milliseconds") != protocol["future_public_randomness"]["pulse_timestamp_unix_milliseconds"]
        or beacon.get("api_path") != protocol["future_public_randomness"]["api_path"]
        or beacon.get("raw_response_path") != BEACON_RAW_PATH.relative_to(ROOT).as_posix()
        or type(beacon.get("raw_response_size_bytes")) is not int
        or beacon["raw_response_size_bytes"] <= 0
        or not _is_sha256(beacon.get("raw_response_sha256"))
        or type(beacon.get("status_code")) is not int
        or beacon["status_code"] != 0
        or type(beacon.get("output_value")) is not str
        or len(beacon["output_value"]) != 128
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in beacon["output_value"]
        )
        or not _is_sha256(value.get("seed_sha256"))
    ):
        raise HoldoutError("beacon binding differs from frozen registration")
    raw = _read_regular(BEACON_RAW_PATH, MAX_BEACON_BYTES, "NIST beacon raw response")
    if (
        len(raw) != beacon["raw_response_size_bytes"]
        or _sha256(raw) != beacon["raw_response_sha256"]
    ):
        raise HoldoutError("retained beacon response bytes changed")
    pulse = _pulse_object(_parse_json(raw, "NIST beacon raw response"))
    raw_chain = pulse.get("chainIndex")
    if type(raw_chain) is str and raw_chain.isascii() and raw_chain.isdecimal():
        raw_chain = int(raw_chain)
    if (
        raw_chain != beacon["chain_index"]
        or pulse.get("timeStamp") != beacon["pulse_timestamp_utc"]
        or pulse.get("statusCode") != beacon["status_code"]
        or pulse.get("outputValue") != beacon["output_value"]
    ):
        raise HoldoutError("retained beacon response content changed")
    seed = _sha256(
        protocol["public_case_derivation"]["seed_domain"].encode("ascii")
        + b"\0"
        + canonical
        + b"\0"
        + bytes.fromhex(beacon["output_value"])
    )
    if value["seed_sha256"] != seed:
        raise HoldoutError("public case seed binding changed")
    return value


def _beacon_publication_record(
    value: dict[str, Any], source_record: dict[str, Any]
) -> dict[str, Any]:
    raw = _read_regular(BEACON_BINDING_PATH, MAX_JSON_BYTES, "beacon binding")
    if raw not in (_canonical_bytes(value), _canonical_bytes(value, newline=True)):
        raise HoldoutError("beacon binding bytes differ from validated value")
    relative = BEACON_BINDING_PATH.relative_to(ROOT).as_posix()
    publication_commit = _git("log", "-1", "--format=%H", "--", relative)
    if type(publication_commit) is not str or len(publication_commit) != 40:
        raise HoldoutError("beacon binding is not additively registered")
    committed = _git("show", f"{publication_commit}:{relative}", binary=True)
    if committed != raw:
        raise HoldoutError("beacon binding differs from its publication commit")
    raw_relative = BEACON_RAW_PATH.relative_to(ROOT).as_posix()
    raw_commit = _git("log", "-1", "--format=%H", "--", raw_relative)
    retained_raw = _read_regular(
        BEACON_RAW_PATH, MAX_BEACON_BYTES, "NIST beacon raw response"
    )
    if (
        type(raw_commit) is not str
        or len(raw_commit) != 40
        or _git("show", f"{raw_commit}:{raw_relative}", binary=True) != retained_raw
    ):
        raise HoldoutError("raw beacon response is not additively registered")
    for ancestor in (
        PROTOCOL_REGISTRATION_COMMIT,
        source_record["publication_commit"],
        raw_commit,
    ):
        if subprocess.run(
            ("git", "merge-base", "--is-ancestor", ancestor, publication_commit),
            cwd=ROOT,
            check=False,
        ).returncode:
            raise HoldoutError("beacon publication ancestry changed")
    if int(_git("show", "-s", "--format=%ct", publication_commit)) * 1000 < 1792728000000:
        raise HoldoutError("beacon binding publication predates the registered pulse")
    head = _git("rev-parse", "HEAD")
    if subprocess.run(
        ("git", "merge-base", "--is-ancestor", publication_commit, head),
        cwd=ROOT,
        check=False,
    ).returncode:
        raise HoldoutError("beacon publication commit is not an ancestor of HEAD")
    return {
        "path": relative,
        "publication_commit": publication_commit,
        "raw_response_publication_commit": raw_commit,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
        "content_sha256": value["content_integrity"]["content_sha256"],
    }


def _draw_digest(protocol: dict[str, Any], seed: str, case_id: str, body_token: str, field: str) -> bytes:
    table = protocol["public_case_derivation"]["literal_field_name_table"]
    allowed = table["case_level_with_body_index_dash"] if body_token == "-" else table["body_level_with_canonical_integer_index"]
    if field not in allowed:
        raise HoldoutError("unknown deterministic draw field")
    if body_token != "-" and (not body_token.isdecimal() or str(int(body_token)) != body_token or int(body_token) < 1):
        raise HoldoutError("invalid deterministic draw body token")
    parts = (protocol["public_case_derivation"]["draw_domain"], seed, case_id, body_token, field)
    try:
        preimage = b"\0".join(part.encode("ascii") for part in parts)
    except UnicodeEncodeError as exc:
        raise HoldoutError("draw token is not ASCII") from exc
    return hashlib.sha256(preimage).digest()


def _draw_integer(protocol: dict[str, Any], seed: str, case_id: str, body_token: str, field: str) -> int:
    return int.from_bytes(_draw_digest(protocol, seed, case_id, body_token, field)[:8], "big")


def _uniform53(protocol: dict[str, Any], seed: str, case_id: str, body: int, field: str) -> tuple[int, int]:
    digest = _draw_digest(protocol, seed, case_id, str(body), field)
    return int.from_bytes(digest[:8], "big") >> 11, 1 << 53


def _body_ids(count: int) -> list[str]:
    if type(count) is not int or count < 2 or count > 33:
        raise HoldoutError("constructor body count is outside frozen bounds")
    return ["primary", *(f"secondary_{index:02d}" for index in range(1, count))]


def _mp_from_float(mp: Any, value: float) -> Any:
    numerator, denominator = value.as_integer_ratio()
    return mp.mpf(numerator) / denominator


def _validate_mpmath_registration(value: Any) -> dict[str, Any]:
    expected_keys = {
        "name",
        "version",
        "package_root",
        "file_count",
        "total_size_bytes",
        "tree_domain",
        "tree_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value["name"] != "mpmath"
        or value["version"] != "1.3.0"
        or value["package_root"] != "mpmath"
        or type(value["file_count"]) is not int
        or value["file_count"] <= 0
        or type(value["total_size_bytes"]) is not int
        or value["total_size_bytes"] <= 0
        or value["tree_domain"] != MPMATH_TREE_DOMAIN
        or not _is_sha256(value["tree_sha256"])
    ):
        raise HoldoutError("registered mpmath package-tree binding changed")
    return value


def _installed_mpmath_record() -> tuple[dict[str, Any], Path]:
    try:
        distribution = importlib.metadata.distribution("mpmath")
    except importlib.metadata.PackageNotFoundError as exc:
        raise HoldoutError("input construction requires mpmath 1.3.0") from exc
    if distribution.version != "1.3.0":
        raise HoldoutError("input construction requires exact mpmath 1.3.0")
    specification = importlib.util.find_spec("mpmath")
    if (
        specification is None
        or specification.origin is None
        or specification.submodule_search_locations is None
    ):
        raise HoldoutError("mpmath must resolve to a regular site-packages package")
    origin = Path(specification.origin).absolute()
    package_root = origin.parent
    site_roots = {
        Path(value).absolute()
        for key, value in sysconfig.get_paths().items()
        if key in ("purelib", "platlib") and value
    }
    matching_roots = []
    for site_root in site_roots:
        try:
            relative = origin.relative_to(site_root)
        except ValueError:
            continue
        if relative.as_posix() == "mpmath/__init__.py":
            matching_roots.append(site_root)
    if len(matching_roots) != 1:
        raise HoldoutError("mpmath did not resolve under the interpreter site-packages")
    site_root = matching_roots[0]
    for forbidden_root in (ROOT.absolute(), Path.cwd().absolute()):
        try:
            origin.relative_to(forbidden_root)
        except ValueError:
            pass
        else:
            raise HoldoutError("repository/cwd mpmath shadowing is forbidden")
    search_locations = [Path(item).absolute() for item in specification.submodule_search_locations]
    if search_locations != [package_root]:
        raise HoldoutError("mpmath package search path changed")
    distribution_files = distribution.files
    if distribution_files is None or not any(
        Path(str(item)).as_posix() == "mpmath/__init__.py"
        for item in distribution_files
    ):
        raise HoldoutError("mpmath distribution RECORD does not own the imported package")
    located_init = Path(distribution.locate_file("mpmath/__init__.py")).absolute()
    if located_init != origin:
        raise HoldoutError("mpmath distribution/import origins disagree")

    records: list[dict[str, Any]] = []
    for directory, child_directories, filenames in os.walk(
        package_root, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(child_directories):
            child = directory_path / name
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise HoldoutError("mpmath package tree changed while scanning") from exc
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise HoldoutError("mpmath package tree contains a non-directory link")
            if name == "__pycache__":
                raise HoldoutError("mpmath bytecode caches are forbidden")
            retained_directories.append(name)
        child_directories[:] = retained_directories
        for name in sorted(filenames):
            path = directory_path / name
            try:
                metadata = path.lstat()
            except OSError as exc:
                raise HoldoutError("mpmath package tree changed while scanning") from exc
            if not stat.S_ISREG(metadata.st_mode):
                raise HoldoutError("mpmath package tree contains a non-regular file")
            if path.suffix == ".pyc":
                raise HoldoutError("mpmath precompiled bytecode is forbidden")
            relative = path.relative_to(site_root).as_posix()
            try:
                relative.encode("ascii")
            except UnicodeEncodeError as exc:
                raise HoldoutError("mpmath package path is not ASCII") from exc
            data = _read_regular(
                path,
                MAX_JSON_BYTES,
                f"mpmath package file {relative}",
                allow_empty=True,
            )
            records.append(
                {"path": relative, "size_bytes": len(data), "sha256": _sha256(data)}
            )
    records.sort(key=lambda item: item["path"].encode("ascii"))
    if not records or records[0]["path"].split("/", 1)[0] != "mpmath":
        raise HoldoutError("mpmath package tree is empty or misplaced")
    record = {
        "name": "mpmath",
        "version": "1.3.0",
        "package_root": "mpmath",
        "file_count": len(records),
        "total_size_bytes": sum(item["size_bytes"] for item in records),
        "tree_domain": MPMATH_TREE_DOMAIN,
        "tree_sha256": _domain_hash(MPMATH_TREE_DOMAIN, records),
    }
    return record, origin


def _load_mpmath(expected_record: dict[str, Any]) -> Any:
    _assert_pre_freeze_import_boundary()
    if any(
        name == "mpmath" or name.startswith("mpmath.") for name in sys.modules
    ):
        raise HoldoutError("mpmath was imported before package-tree validation")
    sys.dont_write_bytecode = True
    sys.pycache_prefix = "/dev/null/jxplanetx-wh-v9-holdout-no-pyc"
    expected = _validate_mpmath_registration(expected_record)
    before, expected_origin = _installed_mpmath_record()
    if before != expected:
        raise HoldoutError("installed mpmath package tree differs from source registration")
    try:
        module = importlib.import_module("mpmath")
    except Exception as exc:
        raise HoldoutError("input construction requires mpmath 1.3.0") from exc
    _assert_pre_freeze_import_boundary()
    if (
        importlib.metadata.version("mpmath") != "1.3.0"
        or getattr(module, "__version__", None) != "1.3.0"
        or Path(getattr(module, "__file__", "")).absolute() != expected_origin
    ):
        raise HoldoutError("input construction requires exact mpmath 1.3.0")
    after, after_origin = _installed_mpmath_record()
    if after != expected or after_origin != expected_origin:
        raise HoldoutError("mpmath package tree changed during import")
    package_root = expected_origin.parent
    loaded = [
        (name, loaded_module)
        for name, loaded_module in sys.modules.items()
        if name == "mpmath" or name.startswith("mpmath.")
    ]
    if not loaded:
        raise HoldoutError("mpmath import did not retain a validated module")
    for name, loaded_module in loaded:
        loaded_file = getattr(loaded_module, "__file__", None)
        if type(loaded_file) is not str:
            raise HoldoutError(f"loaded mpmath module has no source file: {name}")
        path = Path(loaded_file).absolute()
        try:
            path.relative_to(package_root)
        except ValueError as exc:
            raise HoldoutError(f"loaded mpmath module escaped package root: {name}") from exc
        if path.suffix != ".py":
            raise HoldoutError(f"loaded mpmath module is not source-only: {name}")
        _read_regular(
            path, MAX_JSON_BYTES, f"loaded mpmath source {name}", allow_empty=True
        )
    if (
        not sys.dont_write_bytecode
        or sys.pycache_prefix != "/dev/null/jxplanetx-wh-v9-holdout-no-pyc"
    ):
        raise HoldoutError("source-only mpmath import controls changed")
    _assert_pre_freeze_import_boundary()
    return module


def _solve_kepler(mp: Any, eccentricity: Any, mean_anomaly: Any) -> Any:
    two_pi = 2 * mp.pi
    mean = mp.fmod(mean_anomaly, two_pi)
    if mean < 0:
        mean += two_pi
    low = mp.mpf("0")
    high = two_pi
    current = mean
    target_width = mp.power(2, -400)
    for _ in range(2048):
        residual = current - eccentricity * mp.sin(current) - mean
        if residual < 0:
            low = current
        else:
            high = current
        if high - low < target_width:
            return (low + high) / 2
        derivative = 1 - eccentricity * mp.cos(current)
        candidate = current - residual / derivative
        current = candidate if low < candidate < high else (low + high) / 2
    raise HoldoutError("high-precision Kepler construction did not converge")


def _rotate(mp: Any, vector: tuple[Any, Any, Any], node: Any, inclination: Any, periapsis: Any) -> tuple[Any, Any, Any]:
    x, y, z = vector
    c, s = mp.cos(periapsis), mp.sin(periapsis)
    x, y = c * x - s * y, s * x + c * y
    c, s = mp.cos(inclination), mp.sin(inclination)
    y, z = c * y - s * z, s * y + c * z
    c, s = mp.cos(node), mp.sin(node)
    return c * x - s * y, s * x + c * y, z


def _cumulative_gm(gm: list[float]) -> list[float]:
    if not gm or any(type(value) is not float or not math.isfinite(value) or value <= 0.0 for value in gm):
        raise HoldoutError("constructed GM vector is invalid")
    cumulative = [gm[0]]
    for value in gm[1:]:
        retained = float(cumulative[-1] + value)
        if not math.isfinite(retained) or not retained > cumulative[-1]:
            raise HoldoutError("candidate-order cumulative GM did not strictly increase")
        cumulative.append(retained)
    return cumulative


def _construct_cartesian(
    mp: Any,
    gm: list[float],
    axes: list[Any],
    eccentricities: list[Any],
    inclinations: list[Any],
    nodes: list[Any],
    periapses: list[Any],
    anomalies: list[Any],
) -> tuple[list[list[float]], list[list[float]]]:
    count = len(gm)
    cumulative = _cumulative_gm(gm)
    q: list[tuple[Any, Any, Any]] = []
    u: list[tuple[Any, Any, Any]] = []
    for index in range(1, count):
        axis = axes[index - 1]
        eccentricity = eccentricities[index - 1]
        gravitational_parameter = _mp_from_float(mp, cumulative[index])
        anomaly = _solve_kepler(mp, eccentricity, anomalies[index - 1])
        root = mp.sqrt(1 - eccentricity * eccentricity)
        cosine, sine = mp.cos(anomaly), mp.sin(anomaly)
        denominator = 1 - eccentricity * cosine
        mean_motion = mp.sqrt(gravitational_parameter / (axis * axis * axis))
        planar_q = (axis * (cosine - eccentricity), axis * root * sine, mp.mpf("0"))
        planar_u = (-axis * mean_motion * sine / denominator, axis * mean_motion * root * cosine / denominator, mp.mpf("0"))
        q.append(_rotate(mp, planar_q, nodes[index - 1], inclinations[index - 1], periapses[index - 1]))
        u.append(_rotate(mp, planar_u, nodes[index - 1], inclinations[index - 1], periapses[index - 1]))

    # Solve the ordered-Jacobi inverse backward from the exact total
    # barycenter R_(N-1)=0.  This is algebraically identical to an ascending
    # inverse followed by a global center shift, but it preserves structural
    # zero components instead of manufacturing precision-dependent
    # cancellation residuals.
    prefix_p = [
        [mp.mpf("0"), mp.mpf("0"), mp.mpf("0")] for _ in range(count)
    ]
    prefix_v = [
        [mp.mpf("0"), mp.mpf("0"), mp.mpf("0")] for _ in range(count)
    ]
    for index in range(count - 1, 0, -1):
        coefficient = _mp_from_float(mp, gm[index]) / _mp_from_float(
            mp, cumulative[index]
        )
        for component in range(3):
            prefix_p[index - 1][component] = (
                prefix_p[index][component]
                - coefficient * q[index - 1][component]
            )
            prefix_v[index - 1][component] = (
                prefix_v[index][component]
                - coefficient * u[index - 1][component]
            )
    positions: list[list[Any]] = [list(prefix_p[0])]
    velocities: list[list[Any]] = [list(prefix_v[0])]
    for index in range(1, count):
        positions.append(
            [
                prefix_p[index - 1][component] + q[index - 1][component]
                for component in range(3)
            ]
        )
        velocities.append(
            [
                prefix_v[index - 1][component] + u[index - 1][component]
                for component in range(3)
            ]
        )
    binary_p = [[float(row[component]) for component in range(3)] for row in positions]
    binary_v = [[float(row[component]) for component in range(3)] for row in velocities]

    # Registered one-time binary64 math.fsum correction.
    denominator = math.fsum(gm)
    correction_p = [math.fsum(gm[i] * binary_p[i][c] for i in range(count)) / denominator for c in range(3)]
    correction_v = [math.fsum(gm[i] * binary_v[i][c] for i in range(count)) / denominator for c in range(3)]
    for index in range(count):
        binary_p[index] = [float(binary_p[index][c] - correction_p[c]) for c in range(3)]
        binary_v[index] = [float(binary_v[index][c] - correction_v[c]) for c in range(3)]
    if any(not math.isfinite(value) for rows in (binary_p, binary_v) for row in rows for value in row):
        raise HoldoutError("constructed Cartesian state is nonfinite")
    return binary_p, binary_v


def _dot3(left: list[float], right: list[float]) -> float:
    first = left[0] * right[0]
    second = left[1] * right[1]
    third = left[2] * right[2]
    return (first + second) + third


def _norm3(value: list[float]) -> float:
    squared = _dot3(value, value)
    if not math.isfinite(squared) or squared < 0.0:
        raise HoldoutError("standalone guard norm is invalid")
    result = math.sqrt(squared)
    if not math.isfinite(result):
        raise HoldoutError("standalone guard norm is nonfinite")
    return result


def _standalone_initial_guard(
    positions: list[list[float]], velocities: list[list[float]], gm: list[float], radii: list[float], fixed_step: float, minimum_encounter: float
) -> dict[str, Any]:
    count = len(gm)
    cumulative = _cumulative_gm(gm)
    inertias = [cumulative[-1]] + [gm[i] * cumulative[i - 1] / cumulative[i] for i in range(1, count)]
    weighted_p = [gm[0] * positions[0][c] for c in range(3)]
    weighted_v = [gm[0] * velocities[0][c] for c in range(3)]
    coordinates = [[0.0, 0.0, 0.0] for _ in range(count)]
    momenta = [[0.0, 0.0, 0.0] for _ in range(count)]
    relative_v = [[0.0, 0.0, 0.0] for _ in range(count)]
    for index in range(1, count):
        prefix = cumulative[index - 1]
        for component in range(3):
            prefix_p = weighted_p[component] / prefix
            prefix_v = weighted_v[component] / prefix
            coordinates[index][component] = positions[index][component] - prefix_p
            momenta[index][component] = inertias[index] * (
                velocities[index][component] - prefix_v
            )
            weighted_p[component] = weighted_p[component] + gm[index] * positions[index][component]
            weighted_v[component] = weighted_v[component] + gm[index] * velocities[index][component]
    coordinates[0] = [weighted_p[c] / cumulative[-1] for c in range(3)]
    momenta[0] = list(weighted_v)
    relative_v[0] = [momenta[0][c] / cumulative[-1] for c in range(3)]
    for index in range(1, count):
        relative_v[index] = [
            momenta[index][component] / inertias[index]
            for component in range(3)
        ]

    axes: list[float] = []
    eccentricities: list[float] = []
    periods: list[float] = []
    periapse_times: list[float] = []
    periapses: list[float] = []
    for index in range(1, count):
        position = coordinates[index]
        velocity = relative_v[index]
        radius = _norm3(position)
        speed_squared = _dot3(velocity, velocity)
        radial_dot = _dot3(position, velocity)
        parameter = cumulative[index]
        beta = 2.0 * parameter / radius - speed_squared
        axis = parameter / beta
        coefficient = speed_squared - parameter / radius
        eccentricity_vector = [(coefficient * position[c] - radial_dot * velocity[c]) / parameter for c in range(3)]
        eccentricity = _norm3(eccentricity_vector)
        periapse = axis * (1.0 - eccentricity)
        period = 2.0 * PI * math.sqrt(((axis * axis) * axis) / parameter)
        tau = 2.0 * PI * math.sqrt((((axis * axis) * axis) / parameter) * ((((1.0 - eccentricity) * (1.0 - eccentricity)) * (1.0 - eccentricity)) / (1.0 + eccentricity)))
        if not all(math.isfinite(value) and value > 0.0 for value in (axis, periapse, period, tau)) or eccentricity < 0.0:
            raise HoldoutError("standalone reconstructed orbit is invalid")
        axes.append(axis)
        eccentricities.append(eccentricity)
        periods.append(period)
        periapse_times.append(tau)
        periapses.append(periapse)

    acceleration = [[0.0, 0.0, 0.0] for _ in range(count)]
    for target in range(count):
        for source in range(count):
            if source == target:
                continue
            dx = positions[source][0] - positions[target][0]
            dy = positions[source][1] - positions[target][1]
            dz = positions[source][2] - positions[target][2]
            squared = (dx * dx + dy * dy) + dz * dz
            inverse_cube = 1.0 / (squared * math.sqrt(squared))
            weight = gm[source] * inverse_cube
            acceleration[target][0] += weight * dx
            acceleration[target][1] += weight * dy
            acceleration[target][2] += weight * dz

    matrix = [[0.0 for _ in range(count)] for _ in range(count)]
    for row in range(count):
        matrix[row][0] = 1.0
    for column in range(1, count):
        coefficient = -gm[column] / cumulative[column]
        for row in range(column):
            matrix[row][column] = coefficient
        matrix[column][column] = cumulative[column - 1] / cumulative[column]
    cartesian_force = [[gm[i] * acceleration[i][c] for c in range(3)] for i in range(count)]
    force_scale = 0.0
    for row in cartesian_force:
        force_scale = max(force_scale, _norm3(row))
    full_force = [[0.0, 0.0, 0.0] for _ in range(count)]
    for coordinate in range(count):
        for component in range(3):
            total = 0.0
            for body in range(count):
                total = total + matrix[body][coordinate] * cartesian_force[body][component]
            full_force[coordinate][component] = total
    translation_residual = _norm3(full_force[0])
    translation_cap = 256.0 * float(count) * EPSILON * force_scale
    interaction_ratios: list[float] = []
    for index in range(1, count):
        radius = _norm3(coordinates[index])
        coefficient = -(inertias[index] * cumulative[index]) / ((radius * radius) * radius)
        interaction = [full_force[index][c] - coefficient * coordinates[index][c] for c in range(3)]
        interaction_acceleration = _norm3(interaction) / inertias[index]
        acceleration_scale = cumulative[index] / (radius * radius)
        interaction_ratios.append(interaction_acceleration / acceleration_scale)

    node_utilizations: list[float] = []
    node_pair_utilizations: list[dict[str, Any]] = []
    hill_separations: list[float] = []
    for left in range(count - 1):
        for right in range(left + 1, count):
            separation = _norm3([positions[right][c] - positions[left][c] for c in range(3)])
            hill = 0.0
            if left and right:
                average_axis = 0.5 * (axes[left - 1] + axes[right - 1])
                ratio = (gm[left] + gm[right]) / (3.0 * gm[0])
                hill = average_axis * math.pow(ratio, 1.0 / 3.0)
            hill_floor = 3.0 * hill if hill > 0.0 else 0.0
            floor = max(radii[left] + radii[right], max(minimum_encounter, hill_floor))
            margin = 256.0 * float(count) * EPSILON * max(separation, floor)
            utilization = floor / (separation - margin)
            node_utilizations.append(utilization)
            node_pair_utilizations.append(
                {"left": left, "right": right, "utilization": utilization}
            )
            if hill > 0.0:
                hill_separations.append(separation / hill)

    position_scale = 0.0
    velocity_scale = 0.0
    for left in range(count - 1):
        for right in range(left + 1, count):
            position_scale = max(
                position_scale,
                _norm3(
                    [
                        positions[right][component]
                        - positions[left][component]
                        for component in range(3)
                    ]
                ),
            )
            velocity_scale = max(
                velocity_scale,
                _norm3(
                    [
                        velocities[right][component]
                        - velocities[left][component]
                        for component in range(3)
                    ]
                ),
            )
    derived_position_cap = 256.0 * float(count) * EPSILON * position_scale
    derived_velocity_cap = 256.0 * float(count) * EPSILON * velocity_scale
    orbit_fractions = [abs(fixed_step) / value for value in periods]
    periapse_fractions = [abs(fixed_step) / value for value in periapse_times]
    secondary_sum = 0.0
    for value in gm[1:]:
        secondary_sum = float(secondary_sum + value)

    return {
        "cumulative": cumulative,
        "coordinates": coordinates,
        "momenta": momenta,
        "relative_velocities": relative_v,
        "semimajor_axes": axes,
        "eccentricities": eccentricities,
        "periods": periods,
        "periapses": periapses,
        "periapse_timescales": periapse_times,
        "interaction_ratios": interaction_ratios,
        "orbit_fractions": orbit_fractions,
        "periapse_fractions": periapse_fractions,
        "node_pair_utilizations": node_pair_utilizations,
        "maximum_interaction_ratio": max(interaction_ratios, default=0.0),
        "maximum_interaction_index": 1 + min((i for i, value in enumerate(interaction_ratios) if value == max(interaction_ratios, default=0.0)), default=0),
        "maximum_eccentricity": max(eccentricities),
        "minimum_periapse": min(periapses),
        "maximum_orbit_fraction": max(orbit_fractions),
        "maximum_periapse_fraction": max(periapse_fractions),
        "maximum_node_utilization": max(node_utilizations),
        "minimum_mutual_hill_separation": min(hill_separations, default=0.0),
        "barycenter_position_norm": _norm3(coordinates[0]),
        "barycenter_velocity_norm": _norm3(relative_v[0]),
        "derived_barycenter_position_cap": derived_position_cap,
        "derived_barycenter_velocity_cap": derived_velocity_cap,
        "translation_force_residual": translation_residual,
        "translation_force_cap": translation_cap,
        "total_secondary_ratio": secondary_sum / gm[0],
    }


def _float_hex_rows(rows: list[list[float]]) -> list[list[str]]:
    return [[value.hex() for value in row] for row in rows]


def _cumulative_binding(identifier: str, cumulative_hex: list[str]) -> str:
    return _framed_hash(CUMULATIVE_GM_DOMAIN, identifier, cumulative_hex)


def _interior_cells(protocol: dict[str, Any], seed: str) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for count in protocol["interior_case_family"]["body_counts"]:
        records = [record for record in protocol["case_roster"] if record["class"] == "INTERIOR" and record["body_count"] == count]
        ordered = sorted(records, key=lambda record: (_draw_digest(protocol, seed, record["case_id"], "-", "interior_cell_rank"), record["case_id"].encode("ascii")))
        for cell, record in enumerate(ordered):
            assignments[record["case_id"]] = cell
    return assignments


def _base_cell(protocol: dict[str, Any], seed: str, record: dict[str, Any], interiors: dict[str, int]) -> int:
    if record["class"] == "INTERIOR":
        return interiors[record["case_id"]]
    return _draw_integer(protocol, seed, record["case_id"], "-", "boundary_base_cell") % 4


def _case_mass_vector(mp: Any, record: dict[str, Any], cell: int, protocol: dict[str, Any]) -> list[float]:
    count = record["body_count"]
    stress = record["stress"]
    totals = protocol["interior_case_family"]["total_secondary_to_primary_gm_ratio_by_cell"]
    total = mp.mpf(str(totals[cell]))
    if stress in ("MASS_RATIO_0_99", "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION"):
        total = mp.mpf("0.0099")
    values: list[float] = [1.0]
    target_binary = float(total)
    secondary_sum = 0.0
    if stress == "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION":
        other_count = count - 2
        other_share = mp.mpf("0.0009") / other_count
        for index in range(1, count):
            if index == count - 1:
                retained = float(target_binary - secondary_sum)
            elif index == 2:
                retained = float(mp.mpf("0.009"))
            else:
                retained = float(other_share)
            values.append(retained)
            secondary_sum = float(secondary_sum + retained)
    else:
        share = total / (count - 1)
        for index in range(1, count):
            if index == count - 1:
                retained = float(target_binary - secondary_sum)
            else:
                retained = float(share)
            values.append(retained)
            secondary_sum = float(secondary_sum + retained)
    if secondary_sum != target_binary or any(value <= 0.0 for value in values):
        raise HoldoutError("constructed secondary mass recurrence missed frozen total")
    _cumulative_gm(values)
    return values


def _nominal_elements(mp: Any, protocol: dict[str, Any], seed: str, record: dict[str, Any], cell: int) -> dict[str, list[Any]]:
    count = record["body_count"]
    family = protocol["interior_case_family"]
    ratio = mp.mpf(str(family["adjacent_jacobi_semimajor_axis_ratio_by_body_count_and_cell"][str(count)][cell]))
    emax = mp.mpf(str(family["eccentricity_max_by_body_count_and_cell"][str(count)][cell]))
    axes = [mp.power(ratio, index - 1) for index in range(1, count)]
    eccentricities: list[Any] = []
    inclinations: list[Any] = []
    nodes: list[Any] = []
    periapses: list[Any] = []
    anomalies: list[Any] = []
    for index in range(1, count):
        def uniform(field: str) -> Any:
            numerator, denominator = _uniform53(protocol, seed, record["case_id"], index, field)
            return mp.mpf(numerator) / denominator
        eccentricities.append(emax * (mp.mpf("0.25") + mp.mpf("0.75") * uniform("eccentricity_u")))
        inclinations.append(mp.mpf("0.05") * uniform("inclination_u"))
        nodes.append(2 * mp.pi * uniform("longitude_of_ascending_node_u"))
        periapses.append(2 * mp.pi * uniform("argument_of_periapsis_u"))
        anomalies.append(2 * mp.pi * uniform("mean_anomaly_u"))
    stress = record["stress"]
    if stress == "ECCENTRICITY_DYNAMIC_SAFE_0_80":
        eccentricities[-1] = mp.mpf("0.8")
        axes[-1] = axes[-2] * 20
    elif stress == "HILL_SWEPT_CLEARANCE_JUST_INSIDE":
        for sequence in (eccentricities, inclinations, nodes, periapses, anomalies):
            sequence[-2] = mp.mpf("0")
            sequence[-1] = mp.mpf("0")
    elif stress == "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION":
        for sequence in (eccentricities, inclinations, nodes, periapses, anomalies):
            sequence[0] = mp.mpf("0")
            sequence[1] = mp.mpf("0")
        axes[0] = mp.mpf("1")
    return {"axes": axes, "eccentricities": eccentricities, "inclinations": inclinations, "nodes": nodes, "periapses": periapses, "anomalies": anomalies}


def _copy_elements(value: dict[str, list[Any]]) -> dict[str, list[Any]]:
    return {name: list(items) for name, items in value.items()}


def _base_divisor(guard: dict[str, Any]) -> int:
    inner_period = guard["periods"][0]
    maximum = 0.0
    for period, eccentricity in zip(
        guard["periods"], guard["eccentricities"], strict=True
    ):
        one_minus = 1.0 - eccentricity
        denominator = (one_minus * one_minus) * one_minus
        factor = (inner_period / period) * math.sqrt(
            (1.0 + eccentricity) / denominator
        )
        if factor > maximum:
            maximum = factor
    target = 32.0 * maximum
    divisor = 64
    while float(divisor) < target:
        divisor = 2 * divisor
        if divisor > (1 << 30):
            raise HoldoutError("base divisor exceeded constructor bound")
    return divisor


def _node_pair_utilization(
    guard: dict[str, Any], left: int, right: int
) -> float:
    matches = [
        item["utilization"]
        for item in guard["node_pair_utilizations"]
        if item["left"] == left and item["right"] == right
    ]
    if len(matches) != 1:
        raise HoldoutError("standalone node-pair metric roster changed")
    return matches[0]


def _guard_inside(
    guard: dict[str, Any], *, allow_target: str | None = None, strict: bool = False
) -> bool:
    def upper(value: float, limit: float) -> bool:
        return value < limit if strict else value <= limit

    checks = {
        "eccentricity": upper(guard["maximum_eccentricity"], 0.9),
        "interaction": upper(guard["maximum_interaction_ratio"], 0.1),
        "orbit": upper(guard["maximum_orbit_fraction"], 0.05),
        "step": upper(guard["maximum_periapse_fraction"], 0.0625),
        "node": guard["maximum_node_utilization"] < 1.0,
        "mass": upper(guard["total_secondary_ratio"], 0.01),
        "periapse": (
            guard["minimum_periapse"] > float.fromhex("0x1.0000000000000p-12")
            if strict
            else guard["minimum_periapse"]
            >= float.fromhex("0x1.0000000000000p-12")
        ),
        "hierarchy": all(
            right > left
            for left, right in zip(
                guard["semimajor_axes"], guard["semimajor_axes"][1:]
            )
        ),
        "radial": all(
            left_axis * (1.0 + left_eccentricity)
            < right_axis * (1.0 - right_eccentricity)
            for left_axis, right_axis, left_eccentricity, right_eccentricity
            in zip(
                guard["semimajor_axes"],
                guard["semimajor_axes"][1:],
                guard["eccentricities"],
                guard["eccentricities"][1:],
            )
        ),
        "barycenter_position": (
            guard["barycenter_position_norm"]
            < min(1.0, guard["derived_barycenter_position_cap"])
            if strict
            else guard["barycenter_position_norm"]
            <= min(1.0, guard["derived_barycenter_position_cap"])
        ),
        "barycenter_velocity": (
            guard["barycenter_velocity_norm"]
            < min(1.0, guard["derived_barycenter_velocity_cap"])
            if strict
            else guard["barycenter_velocity_norm"]
            <= min(1.0, guard["derived_barycenter_velocity_cap"])
        ),
        "translation": (
            guard["translation_force_residual"] < guard["translation_force_cap"]
            if strict
            else guard["translation_force_residual"]
            <= guard["translation_force_cap"]
        ),
        "body_count": len(guard["cumulative"]) <= 32,
    }
    return all(passed for name, passed in checks.items() if name != allow_target)


def _state_and_guard(
    mp: Any,
    gm: list[float],
    elements: dict[str, list[Any]],
    fixed_step_rule: int | None = None,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any], float]:
    positions, velocities = _construct_cartesian(
        mp,
        gm,
        elements["axes"],
        elements["eccentricities"],
        elements["inclinations"],
        elements["nodes"],
        elements["periapses"],
        elements["anomalies"],
    )
    radii = [0.0] * len(gm)
    preliminary = _standalone_initial_guard(
        positions,
        velocities,
        gm,
        radii,
        1.0e-300,
        float.fromhex("0x1.0000000000000p-40"),
    )
    divisor = fixed_step_rule if fixed_step_rule is not None else _base_divisor(preliminary)
    step = float(preliminary["periods"][0] / divisor)
    guard = _standalone_initial_guard(
        positions,
        velocities,
        gm,
        radii,
        step,
        float.fromhex("0x1.0000000000000p-40"),
    )
    return positions, velocities, guard, step


def _calibrate_parameter(
    mp: Any,
    low_text: str,
    high_text: str,
    target: float,
    band: tuple[float, float],
    evaluator: Callable[[Any], tuple[float, Any]],
    *,
    decreasing: bool,
    label: str,
    final_predicate: Callable[[Any], bool] | None = None,
) -> tuple[float, Any, tuple[str, ...]]:
    low = mp.mpf(low_text)
    high = mp.mpf(high_text)
    low_value, _ = evaluator(low)
    high_value, _ = evaluator(high)
    if not all(math.isfinite(value) for value in (low_value, high_value)):
        raise HoldoutError(f"{label} calibration endpoint is nonfinite")
    if decreasing:
        bracketed = low_value >= target >= high_value
    else:
        bracketed = low_value <= target <= high_value
    if not bracketed:
        raise CalibrationUnavailable(
            f"{label} calibration endpoints do not bracket target"
        )
    branches: list[str] = []
    midpoint = (low + high) / 2
    for _ in range(80):
        midpoint = (low + high) / 2
        value, _ = evaluator(midpoint)
        if value == target:
            high = midpoint
            branches.append("EQUAL_TO_HIGH")
        elif value > target if decreasing else value < target:
            low = midpoint
            branches.append("LOW")
        else:
            high = midpoint
            branches.append("HIGH")
    candidates = sorted({float(low), float(midpoint), float(high)})
    for candidate in candidates:
        value, payload = evaluator(_mp_from_float(mp, candidate))
        if band[0] <= value <= band[1] and (
            final_predicate is None or final_predicate(payload)
        ):
            return candidate, payload, tuple(branches)
    raise CalibrationUnavailable(
        f"{label} calibration has no frozen-band candidate", tuple(branches)
    )


def _calibrated_case(
    mp: Any,
    protocol: dict[str, Any],
    gm: list[float],
    elements: dict[str, list[Any]],
    stress: str | None,
) -> tuple[list[list[float]], list[list[float]], dict[str, Any], float, tuple[str, ...]]:
    trace: list[str] = []
    if stress is None or stress in (
        "MASS_RATIO_0_99",
        "ECCENTRICITY_DYNAMIC_SAFE_0_80",
        "MAXIMUM_BODY_COUNT",
    ):
        positions, velocities, guard, step = _state_and_guard(mp, gm, elements)
        if not _guard_inside(guard):
            raise HoldoutError("uncalibrated case fails a registered input guard")
        return positions, velocities, guard, step, tuple(trace)

    if stress == "PERIAPSE_STEP_DYNAMIC_SAFE":
        rule = protocol["boundary_stress_construction"][stress]
        for divisor in rule["D0_candidates_in_ascending_fall_through_order"]:
            def evaluate(parameter: Any) -> tuple[float, Any]:
                selected = _copy_elements(elements)
                selected["eccentricities"][0] = parameter
                positions, velocities, guard, step = _state_and_guard(
                    mp, gm, selected, divisor
                )
                return guard["periapse_fractions"][0], (
                    selected,
                    positions,
                    velocities,
                    guard,
                    step,
                )

            try:
                chosen, payload, branches = _calibrate_parameter(
                    mp,
                    "0.0",
                    "0.8",
                    0.03125,
                    (0.03025, 0.03125),
                    evaluate,
                    decreasing=False,
                    label=f"periapse-step D0={divisor}",
                    final_predicate=lambda item: _base_divisor(item[3]) == divisor
                    and _guard_inside(item[3]),
                )
            except CalibrationUnavailable as exc:
                trace.append(f"D0={divisor}:NO_SELECTION")
                trace.extend(
                    f"D0={divisor}:{branch}" for branch in exc.branches
                )
                continue
            selected, positions, velocities, guard, step = payload
            elements.clear()
            elements.update(selected)
            trace.extend((f"D0={divisor}:SELECT:{chosen.hex()}", *branches))
            return positions, velocities, guard, step, tuple(trace)
        raise HoldoutError("no frozen D0 candidate constructs periapse-step stress")

    if stress == "HILL_SWEPT_CLEARANCE_JUST_INSIDE":
        rule = protocol["boundary_stress_construction"][stress]
        target = float(rule["initial_node_utilization_target"])
        band = tuple(float(value) for value in rule["reconstructed_target_band"])

        def evaluate(parameter: Any) -> tuple[float, Any]:
            selected = _copy_elements(elements)
            selected["axes"][-1] = selected["axes"][-2] * parameter
            result = _state_and_guard(mp, gm, selected)
            guard = result[2]
            return _node_pair_utilization(
                guard, len(gm) - 2, len(gm) - 1
            ), (selected, *result)

        chosen, payload, branches = _calibrate_parameter(
            mp,
            "1.0001",
            "4.0",
            target,
            band,
            evaluate,
            decreasing=True,
            label="Hill/node stress",
            final_predicate=lambda item: _guard_inside(item[3], strict=True),
        )
        selected, positions, velocities, guard, step = payload
        elements.clear()
        elements.update(selected)
        return positions, velocities, guard, step, (
            f"SELECT:{chosen.hex()}",
            *branches,
        )

    if stress == "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION":
        endpoint_values: list[float] = []

        def evaluate(parameter: Any) -> tuple[float, Any]:
            selected = _copy_elements(elements)
            selected["axes"][1] = parameter
            result = _state_and_guard(mp, gm, selected)
            return result[2]["maximum_interaction_ratio"], (selected, *result)

        for endpoint in (mp.mpf("1.0001"), mp.mpf("2.0")):
            endpoint_values.append(evaluate(endpoint)[0])
        if not (endpoint_values[0] - 0.015 > 0.0 and endpoint_values[1] - 0.015 < 0.0):
            raise HoldoutError("interaction stress endpoint residual signs changed")
        chosen, payload, branches = _calibrate_parameter(
            mp,
            "1.0001",
            "2.0",
            0.015,
            (0.0145, 0.0155),
            evaluate,
            decreasing=True,
            label="interaction stress",
            final_predicate=lambda item: _guard_inside(item[3], strict=True)
            and item[3]["minimum_mutual_hill_separation"] >= 3.5,
        )
        selected, positions, velocities, guard, step = payload
        elements.clear()
        elements.update(selected)
        return positions, velocities, guard, step, (
            f"ENDPOINT_LOW:{endpoint_values[0].hex()}",
            f"ENDPOINT_HIGH:{endpoint_values[1].hex()}",
            f"SELECT:{chosen.hex()}",
            *branches,
        )
    raise HoldoutError(f"unknown boundary stress: {stress}")


def _orbital_records(guard: dict[str, Any], body_ids: list[str]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for offset, body_id in enumerate(body_ids[1:]):
        result.append(
            {
                "body_id": body_id,
                "semimajor_axis_float_hex": guard["semimajor_axes"][offset].hex(),
                "eccentricity_float_hex": guard["eccentricities"][offset].hex(),
                "periapse_float_hex": guard["periapses"][offset].hex(),
                "period_float_hex": guard["periods"][offset].hex(),
                "periapse_timescale_float_hex": guard["periapse_timescales"][offset].hex(),
            }
        )
    return result


def _guard_record(
    guard: dict[str, Any], *, allow_target: str | None = None
) -> dict[str, Any]:
    return {
        "maximum_eccentricity_float_hex": guard["maximum_eccentricity"].hex(),
        "minimum_periapse_float_hex": guard["minimum_periapse"].hex(),
        "maximum_interaction_ratio_float_hex": guard["maximum_interaction_ratio"].hex(),
        "maximum_interaction_secondary_index": guard["maximum_interaction_index"],
        "maximum_orbit_fraction_float_hex": guard["maximum_orbit_fraction"].hex(),
        "maximum_periapse_step_fraction_float_hex": guard["maximum_periapse_fraction"].hex(),
        "maximum_initial_node_utilization_float_hex": guard["maximum_node_utilization"].hex(),
        "minimum_mutual_hill_separation_float_hex": guard["minimum_mutual_hill_separation"].hex(),
        "barycenter_position_norm_float_hex": guard["barycenter_position_norm"].hex(),
        "barycenter_velocity_norm_float_hex": guard["barycenter_velocity_norm"].hex(),
        "derived_barycenter_position_cap_float_hex": guard[
            "derived_barycenter_position_cap"
        ].hex(),
        "derived_barycenter_velocity_cap_float_hex": guard[
            "derived_barycenter_velocity_cap"
        ].hex(),
        "translation_force_residual_float_hex": guard[
            "translation_force_residual"
        ].hex(),
        "translation_force_cap_float_hex": guard["translation_force_cap"].hex(),
        "total_secondary_to_primary_gm_ratio_float_hex": guard["total_secondary_ratio"].hex(),
        "orbit_fraction_by_secondary_float_hex": [
            value.hex() for value in guard["orbit_fractions"]
        ],
        "periapse_step_fraction_by_secondary_float_hex": [
            value.hex() for value in guard["periapse_fractions"]
        ],
        "initial_node_pair_utilizations_float_hex": [
            {
                "left_body_index": item["left"],
                "right_body_index": item["right"],
                "utilization_float_hex": item["utilization"].hex(),
            }
            for item in guard["node_pair_utilizations"]
        ],
        "all_non_target_initial_guards_inside": _guard_inside(
            guard, allow_target=allow_target
        ),
    }


def _construct_case_at_precision(
    mpmath: Any,
    protocol: dict[str, Any],
    seed: str,
    roster: dict[str, Any],
    cell: int,
    precision: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    with mpmath.workdps(precision):
        mp = mpmath.mp
        gm = _case_mass_vector(mp, roster, cell, protocol)
        elements = _nominal_elements(mp, protocol, seed, roster, cell)
        positions, velocities, guard, coarse_step, trace = _calibrated_case(
            mp, protocol, gm, elements, roster["stress"]
        )
        divisor = _base_divisor(guard)
        # A calibrated step case must preserve the selected divisor.
        if roster["stress"] == "PERIAPSE_STEP_DYNAMIC_SAFE" and float(
            guard["periods"][0] / divisor
        ).hex() != coarse_step.hex():
            raise HoldoutError("calibrated periapse-step divisor changed")
        coarse_step = float(guard["periods"][0] / divisor)
        guard = _standalone_initial_guard(
            positions,
            velocities,
            gm,
            [0.0] * len(gm),
            coarse_step,
            float.fromhex("0x1.0000000000000p-40"),
        )
        stress = roster["stress"]
        if not _guard_inside(
            guard,
            strict=stress
            in (
                "HILL_SWEPT_CLEARANCE_JUST_INSIDE",
                "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION",
            ),
        ):
            raise HoldoutError("final case reconstruction fails an input guard")
        if stress == "MASS_RATIO_0_99" and guard["total_secondary_ratio"] != float(
            protocol["boundary_stress_construction"][stress][
                "total_secondary_to_primary_gm_ratio"
            ]
        ):
            raise HoldoutError("mass-stress target changed")
        if stress == "ECCENTRICITY_DYNAMIC_SAFE_0_80":
            rule = protocol["boundary_stress_construction"][stress]
            if not (
                rule["reconstructed_binary64_eccentricity_min"]
                <= guard["eccentricities"][-1]
                <= rule["reconstructed_binary64_eccentricity_max"]
            ):
                raise HoldoutError("eccentricity-stress reconstruction left its band")
        if stress == "PERIAPSE_STEP_DYNAMIC_SAFE":
            rule = protocol["boundary_stress_construction"][stress]
            lower = rule["target_periapse_step_fraction"] - rule[
                "guard_utilization_tolerance_absolute"
            ]
            if not lower <= guard["periapse_fractions"][0] <= rule[
                "target_periapse_step_fraction"
            ]:
                raise HoldoutError("periapse-step target coordinate left its band")
        if stress == "HILL_SWEPT_CLEARANCE_JUST_INSIDE":
            rule = protocol["boundary_stress_construction"][stress]
            value = _node_pair_utilization(guard, len(gm) - 2, len(gm) - 1)
            if not rule["reconstructed_target_band"][0] <= value <= rule[
                "reconstructed_target_band"
            ][1]:
                raise HoldoutError("Hill-stress target pair left its band")
        if stress == "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION":
            rule = protocol["boundary_stress_construction"][stress]
            if not (
                rule["reconstructed_target_band"][0]
                <= guard["maximum_interaction_ratio"]
                <= rule["reconstructed_target_band"][1]
                and guard["minimum_mutual_hill_separation"]
                >= rule["minimum_initial_adjacent_mutual_hill_separation"]
            ):
                raise HoldoutError("interaction-stress feasibility proof failed")
        body_ids = _body_ids(roster["body_count"])
        cumulative_hex = [value.hex() for value in _cumulative_gm(gm)]
        divisors = [divisor, 2 * divisor, 4 * divisor]
        steps = [float(guard["periods"][0] / value) for value in divisors]
        checkpoints = [
            {
                "divisor": value,
                "indices": list(
                    range(
                        0,
                        roster["horizon_inner_periods"] * value + 1,
                        value // 4,
                    )
                ),
            }
            for value in divisors
        ]
        core = {
            "case_id": roster["case_id"],
            "body_count": roster["body_count"],
            "body_ids": body_ids,
            "jacobi_body_order": body_ids,
            "initial_epoch_float_hex": "0x0.0p+0",
            "positions_float_hex": _float_hex_rows(positions),
            "velocities_float_hex": _float_hex_rows(velocities),
            "gm_float_hex": [value.hex() for value in gm],
            "cumulative_gm_float_hex": cumulative_hex,
            "cumulative_gm_binding_sha256": _cumulative_binding(
                roster["case_id"], cumulative_hex
            ),
            "radii_float_hex": ["0x0.0p+0"] * len(gm),
            "minimum_encounter_pair_separation_float_hex": "0x1.0000000000000p-40",
            "minimum_jacobi_periapse_float_hex": "0x1.0000000000000p-12",
            "maximum_initial_barycenter_position_norm_float_hex": "0x1.0000000000000p+0",
            "maximum_initial_barycenter_velocity_norm_float_hex": "0x1.0000000000000p+0",
            "inner_period_float_hex": guard["periods"][0].hex(),
            "base_step_divisor": divisor,
            "step_divisors": divisors,
            "steps_float_hex": [value.hex() for value in steps],
            "checkpoint_indices_by_divisor": checkpoints,
            "derived_orbital_elements_float_hex": _orbital_records(
                guard, body_ids
            ),
            "input_guard_utilizations": _guard_record(guard),
        }
        return core | {
            "case_content_sha256": _framed_hash(
                CASE_DOMAIN, roster["case_id"], core
            )
        }, trace


def _control_mass_vector(mp: Any, record: dict[str, Any]) -> list[float]:
    count = record["body_count"]
    if "secondary_gm" in record:
        return [1.0, float.fromhex(record["secondary_gm"])]
    if record["control_id"] in ("JXHHO-C-N32-IN", "JXHHO-C-N33-OUT"):
        total = mp.power(2, -10)
    else:
        total = mp.mpf(str(record.get("total_secondary_to_primary_gm_ratio", "0.001")))
    share = total / (count - 1)
    target_binary = float(total)
    secondary_sum = 0.0
    values = [1.0]
    for index in range(1, count):
        retained = (
            float(target_binary - secondary_sum)
            if index == count - 1
            else float(share)
        )
        values.append(retained)
        secondary_sum = float(secondary_sum + retained)
    if secondary_sum != target_binary or any(value <= 0.0 for value in values):
        raise HoldoutError("guard-control mass recurrence missed frozen total")
    return values


def _control_elements(mp: Any, count: int) -> dict[str, list[Any]]:
    return {
        "axes": [mp.power(4, index - 1) for index in range(1, count)],
        "eccentricities": [mp.mpf("0") for _ in range(1, count)],
        "inclinations": [mp.mpf("0") for _ in range(1, count)],
        "nodes": [mp.mpf("0") for _ in range(1, count)],
        "periapses": [mp.mpf("0") for _ in range(1, count)],
        "anomalies": [mp.mpf("0") for _ in range(1, count)],
    }


def _construct_control_at_precision(
    mpmath: Any,
    protocol: dict[str, Any],
    record: dict[str, Any],
    precision: int,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    with mpmath.workdps(precision):
        mp = mpmath.mp
        gm = _control_mass_vector(mp, record)
        elements = _control_elements(mp, record["body_count"])
        if record["control_id"] in ("JXHHO-C-N32-IN", "JXHHO-C-N33-OUT"):
            elements["axes"] = [mp.power(2, index - 1) for index in range(1, record["body_count"])]
        trace: tuple[str, ...] = ()

        def state(divisor: int) -> tuple[list[list[float]], list[list[float]], dict[str, Any], float]:
            return _state_and_guard(mp, gm, elements, divisor)

        pair = record["pair_id"]
        if pair == "ECCENTRICITY_GUARD":
            low, high = record["closed_interval"]
            if record["expected"] == "ACCEPT":
                target = record["target"]
                band = tuple(record["reconstructed_acceptance_band"])
            else:
                target = record["reconstructed_target"]
                band = (
                    record["reconstructed_acceptance_band_min"],
                    record["reconstructed_acceptance_band_max"],
                )

            def evaluate(parameter: Any) -> tuple[float, Any]:
                elements["eccentricities"][0] = parameter
                result = state(4096)
                return result[2]["maximum_eccentricity"], result

            chosen, payload, branches = _calibrate_parameter(
                mp, str(low), str(high), target, band, evaluate,
                decreasing=False,
                label=record["control_id"],
                final_predicate=lambda item: _guard_inside(
                    item[2], allow_target="eccentricity"
                ),
            )
            elements["eccentricities"][0] = _mp_from_float(mp, chosen)
            positions, velocities, guard, fixed_step = payload
            trace = (f"SELECT:{chosen.hex()}", *branches)
        elif pair == "PERIAPSE_STEP_GUARD":
            target = (
                record.get("target_periapse_step_fraction")
                or record["reconstructed_guard_utilization_target"] * 0.0625
            )
            band = (
                tuple(record["reconstructed_guard_utilization_band"])
                if "reconstructed_guard_utilization_band" in record
                else (
                    record["reconstructed_guard_utilization_band_min"],
                    record["reconstructed_guard_utilization_band_max"],
                )
            )
            fraction_band = (band[0] * 0.0625, band[1] * 0.0625)

            def evaluate(parameter: Any) -> tuple[float, Any]:
                elements["eccentricities"][0] = parameter
                result = state(64)
                return result[2]["maximum_periapse_fraction"], result

            chosen, payload, branches = _calibrate_parameter(
                mp, "0.0", "0.899", float(target), fraction_band, evaluate,
                decreasing=False,
                label=record["control_id"],
                final_predicate=lambda item: _guard_inside(
                    item[2], allow_target="step"
                ),
            )
            elements["eccentricities"][0] = _mp_from_float(mp, chosen)
            positions, velocities, guard, fixed_step = payload
            trace = (f"SELECT:{chosen.hex()}", *branches)
        elif pair == "INITIAL_NODE_HILL_GUARD":
            target = record.get("target_initial_node_utilization", record.get("reconstructed_initial_node_utilization_target"))
            band = (
                tuple(record["reconstructed_acceptance_band"])
                if "reconstructed_acceptance_band" in record
                else (
                    record["reconstructed_initial_node_utilization_band_min"],
                    record["reconstructed_initial_node_utilization_band_max"],
                )
            )

            def evaluate(parameter: Any) -> tuple[float, Any]:
                elements["axes"][1] = parameter
                result = state(1048576)
                return _node_pair_utilization(result[2], 1, 2), result

            chosen, payload, branches = _calibrate_parameter(
                mp, "1.0001", "4.0", float(target), band, evaluate,
                decreasing=True,
                label=record["control_id"],
                final_predicate=lambda item: _guard_inside(
                    item[2], allow_target="node"
                ),
            )
            elements["axes"][1] = _mp_from_float(mp, chosen)
            positions, velocities, guard, fixed_step = payload
            trace = (f"SELECT:{chosen.hex()}", *branches)
        else:
            divisor = 64
            positions, velocities, guard, fixed_step = state(divisor)

        target_name = {
            "ECCENTRICITY_GUARD": "eccentricity",
            "PERIAPSE_STEP_GUARD": "step",
            "TOTAL_SECONDARY_GM_GUARD": "mass",
            "INITIAL_NODE_HILL_GUARD": "node",
            "MAXIMUM_BODY_COUNT_GUARD": "body_count",
        }[pair]
        target_passes = {
            "eccentricity": guard["maximum_eccentricity"] <= 0.9,
            "step": guard["maximum_periapse_fraction"] <= 0.0625,
            "mass": guard["total_secondary_ratio"] <= 0.01,
            "node": guard["maximum_node_utilization"] < 1.0,
            "body_count": record["body_count"] <= 32,
        }[target_name]
        if target_passes != (record["expected"] == "ACCEPT"):
            raise HoldoutError(
                f"{record['control_id']} does not reconstruct its target guard semantics"
            )
        if not _guard_inside(guard, allow_target=target_name):
            raise HoldoutError(
                f"{record['control_id']} fails a non-target initial guard"
            )

        body_ids = _body_ids(record["body_count"])
        cumulative_hex = [value.hex() for value in _cumulative_gm(gm)]
        core = {
            "control_id": record["control_id"],
            "pair_id": record["pair_id"],
            "expected": record["expected"],
            "execution_mode": record["execution_mode"],
            "body_ids": body_ids,
            "jacobi_body_order": body_ids,
            "initial_epoch_float_hex": "0x0.0p+0",
            "positions_float_hex": _float_hex_rows(positions),
            "velocities_float_hex": _float_hex_rows(velocities),
            "gm_float_hex": [value.hex() for value in gm],
            "cumulative_gm_float_hex": cumulative_hex,
            "cumulative_gm_binding_sha256": _cumulative_binding(
                record["control_id"], cumulative_hex
            ),
            "radii_float_hex": ["0x0.0p+0"] * len(gm),
            "fixed_step_float_hex": fixed_step.hex(),
            "checkpoint_step_indices": [0, 1],
            "minimum_encounter_pair_separation_float_hex": "0x1.0000000000000p-40",
            "minimum_jacobi_periapse_float_hex": "0x1.0000000000000p-12",
            "maximum_initial_barycenter_position_norm_float_hex": "0x1.0000000000000p+0",
            "maximum_initial_barycenter_velocity_norm_float_hex": "0x1.0000000000000p+0",
            "reconstructed_initial_guard_metrics_float_hex": _guard_record(
                guard, allow_target=target_name
            ),
        }
        return core | {
            "control_content_sha256": _framed_hash(
                CONTROL_DOMAIN, record["control_id"], core
            )
        }, trace


def _compare_precision_results(
    low: tuple[dict[str, Any], tuple[str, ...]],
    high: tuple[dict[str, Any], tuple[str, ...]],
    label: str,
) -> dict[str, Any]:
    if _canonical_bytes(low[0]) != _canonical_bytes(high[0]) or low[1] != high[1]:
        raise HoldoutError(f"{label} is not stable at 160/220 digits")
    return low[0]


def construct_input_manifest(
    beacon_binding: dict[str, Any], source_registration: Path
) -> dict[str, Any]:
    _assert_pre_freeze_import_boundary()
    protocol, _ = _protocol()
    binding = validate_beacon_binding(beacon_binding, source_registration)
    source_value, source_record = _source_registration(source_registration)
    beacon_record = _beacon_publication_record(binding, source_record)
    mpmath_record = _validate_mpmath_registration(
        source_value["constructor_dependency"]
    )
    mpmath = _load_mpmath(mpmath_record)
    interiors = _interior_cells(protocol, binding["seed_sha256"])
    cases: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for roster in protocol["case_roster"]:
        cell = _base_cell(protocol, binding["seed_sha256"], roster, interiors)
        low = _construct_case_at_precision(
            mpmath, protocol, binding["seed_sha256"], roster, cell, 160
        )
        high = _construct_case_at_precision(
            mpmath, protocol, binding["seed_sha256"], roster, cell, 220
        )
        cases.append(_compare_precision_results(low, high, roster["case_id"]))
        traces.append(
            {
                "id": roster["case_id"],
                "cell_index": cell,
                "branch_trace_sha256": _domain_hash(
                    "jxplanetx.wh-v9-holdout.constructor-branches.v1", low[1]
                ),
            }
        )
    controls: list[dict[str, Any]] = []
    for record in protocol["guard_semantics_controls"]["controls"]:
        low = _construct_control_at_precision(mpmath, protocol, record, 160)
        high = _construct_control_at_precision(mpmath, protocol, record, 220)
        controls.append(_compare_precision_results(low, high, record["control_id"]))
        traces.append(
            {
                "id": record["control_id"],
                "cell_index": None,
                "branch_trace_sha256": _domain_hash(
                    "jxplanetx.wh-v9-holdout.constructor-branches.v1", low[1]
                ),
            }
        )
    _assert_pre_freeze_import_boundary()
    core = {
        "schema": MANIFEST_SCHEMA,
        "holdout_id": protocol["holdout_id"],
        "protocol": binding["protocol"],
        "source_registration": source_record,
        "beacon_binding": beacon_record,
        "seed_sha256": binding["seed_sha256"],
        "constructor": {
            "mpmath_version": "1.3.0",
            "mpmath_package_tree": mpmath_record,
            "construction_precision_decimal_digits": 160,
            "verification_precision_decimal_digits": 220,
            "dual_precision_bitwise_stable": True,
            "candidate_or_oracle_imported": False,
            "candidate_or_oracle_executed": False,
            "trace_records": traces,
        },
        "cases": cases,
        "guard_controls": controls,
        "claim_controls": {
            "input_manifest_frozen": True,
            "holdout_execution_completed": False,
            "scientific_outcome_generated": False,
            "scientific_approval_granted": False,
        },
    }
    return core | {
        "content_integrity": {
            "algorithm": "SHA-256",
            "canonicalization": "SORTED_KEYS_COMPACT_ASCII_EXCLUDING_CONTENT_INTEGRITY",
            "domain": MANIFEST_DOMAIN,
            "content_sha256": _domain_hash(MANIFEST_DOMAIN, core),
        }
    }


def _finite_float_hex(value: Any, label: str) -> float:
    if type(value) is not str:
        raise HoldoutError(f"{label} is not a float.hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise HoldoutError(f"{label} is not a float.hex string") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise HoldoutError(f"{label} is not canonical finite binary64")
    return result


def _float_hex_vector(value: Any, length: int, label: str) -> list[float]:
    if type(value) is not list or len(value) != length:
        raise HoldoutError(f"{label} shape changed")
    return [_finite_float_hex(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _float_hex_matrix(value: Any, rows: int, label: str) -> list[list[float]]:
    if type(value) is not list or len(value) != rows:
        raise HoldoutError(f"{label} shape changed")
    return [
        _float_hex_vector(row, 3, f"{label}[{index}]")
        for index, row in enumerate(value)
    ]


def _validate_guard_record(value: Any, body_count: int, label: str) -> None:
    scalar_names = {
        "maximum_eccentricity_float_hex",
        "minimum_periapse_float_hex",
        "maximum_interaction_ratio_float_hex",
        "maximum_orbit_fraction_float_hex",
        "maximum_periapse_step_fraction_float_hex",
        "maximum_initial_node_utilization_float_hex",
        "minimum_mutual_hill_separation_float_hex",
        "barycenter_position_norm_float_hex",
        "barycenter_velocity_norm_float_hex",
        "derived_barycenter_position_cap_float_hex",
        "derived_barycenter_velocity_cap_float_hex",
        "translation_force_residual_float_hex",
        "translation_force_cap_float_hex",
        "total_secondary_to_primary_gm_ratio_float_hex",
    }
    expected = scalar_names | {
        "maximum_interaction_secondary_index",
        "orbit_fraction_by_secondary_float_hex",
        "periapse_step_fraction_by_secondary_float_hex",
        "initial_node_pair_utilizations_float_hex",
        "all_non_target_initial_guards_inside",
    }
    if type(value) is not dict or set(value) != expected:
        raise HoldoutError(f"{label} field roster changed")
    for name in scalar_names:
        _finite_float_hex(value[name], f"{label}.{name}")
    if (
        type(value["maximum_interaction_secondary_index"]) is not int
        or not 1 <= value["maximum_interaction_secondary_index"] < body_count
        or type(value["all_non_target_initial_guards_inside"]) is not bool
    ):
        raise HoldoutError(f"{label} discrete metric changed")
    _float_hex_vector(
        value["orbit_fraction_by_secondary_float_hex"],
        body_count - 1,
        f"{label}.orbit_fraction_by_secondary_float_hex",
    )
    _float_hex_vector(
        value["periapse_step_fraction_by_secondary_float_hex"],
        body_count - 1,
        f"{label}.periapse_step_fraction_by_secondary_float_hex",
    )
    pairs = value["initial_node_pair_utilizations_float_hex"]
    if type(pairs) is not list or len(pairs) != body_count * (body_count - 1) // 2:
        raise HoldoutError(f"{label} node-pair roster changed")
    expected_pairs = [
        (left, right)
        for left in range(body_count - 1)
        for right in range(left + 1, body_count)
    ]
    for item, pair in zip(pairs, expected_pairs, strict=True):
        if (
            type(item) is not dict
            or set(item)
            != {"left_body_index", "right_body_index", "utilization_float_hex"}
            or (item["left_body_index"], item["right_body_index"]) != pair
        ):
            raise HoldoutError(f"{label} node-pair order changed")
        _finite_float_hex(item["utilization_float_hex"], f"{label}.node_pair")


def _validate_common_input_record(
    record: dict[str, Any], identifier: str, body_count: int, label: str
) -> None:
    body_ids = _body_ids(body_count)
    if record["body_ids"] != body_ids or record["jacobi_body_order"] != body_ids:
        raise HoldoutError(f"{label} body identity/order changed")
    if _finite_float_hex(record["initial_epoch_float_hex"], f"{label}.epoch") != 0.0:
        raise HoldoutError(f"{label} initial epoch changed")
    _float_hex_matrix(record["positions_float_hex"], body_count, f"{label}.positions")
    _float_hex_matrix(record["velocities_float_hex"], body_count, f"{label}.velocities")
    gm = _float_hex_vector(record["gm_float_hex"], body_count, f"{label}.gm")
    if any(value <= 0.0 for value in gm):
        raise HoldoutError(f"{label} GM is not positive")
    cumulative = _float_hex_vector(
        record["cumulative_gm_float_hex"], body_count, f"{label}.cumulative_gm"
    )
    recomputed = _cumulative_gm(gm)
    if cumulative != recomputed or record["cumulative_gm_binding_sha256"] != _cumulative_binding(
        identifier, [value.hex() for value in recomputed]
    ):
        raise HoldoutError(f"{label} cumulative-GM binding changed")
    radii = _float_hex_vector(record["radii_float_hex"], body_count, f"{label}.radii")
    if any(value < 0.0 for value in radii):
        raise HoldoutError(f"{label} radius is negative")
    for name in (
        "minimum_encounter_pair_separation_float_hex",
        "minimum_jacobi_periapse_float_hex",
        "maximum_initial_barycenter_position_norm_float_hex",
        "maximum_initial_barycenter_velocity_norm_float_hex",
    ):
        if _finite_float_hex(record[name], f"{label}.{name}") <= 0.0:
            raise HoldoutError(f"{label}.{name} is not positive")


def _validate_case_record(
    value: Any, roster: dict[str, Any], required: list[str]
) -> None:
    label = roster["case_id"]
    if type(value) is not dict or set(value) != set(required):
        raise HoldoutError(f"{label} case field roster changed")
    if value["case_id"] != label or value["body_count"] != roster["body_count"]:
        raise HoldoutError(f"{label} case identity changed")
    count = roster["body_count"]
    _validate_common_input_record(value, label, count, label)
    inner_period = _finite_float_hex(value["inner_period_float_hex"], f"{label}.inner_period")
    divisor = value["base_step_divisor"]
    if type(divisor) is not int or divisor < 64 or divisor & (divisor - 1):
        raise HoldoutError(f"{label} base divisor changed")
    if value["step_divisors"] != [divisor, 2 * divisor, 4 * divisor]:
        raise HoldoutError(f"{label} step-divisor ladder changed")
    steps = _float_hex_vector(value["steps_float_hex"], 3, f"{label}.steps")
    if steps != [float(inner_period / item) for item in value["step_divisors"]]:
        raise HoldoutError(f"{label} fixed-step values changed")
    checkpoints = value["checkpoint_indices_by_divisor"]
    if type(checkpoints) is not list or len(checkpoints) != 3:
        raise HoldoutError(f"{label} checkpoint roster changed")
    for item, step_divisor in zip(checkpoints, value["step_divisors"], strict=True):
        expected_indices = list(
            range(
                0,
                roster["horizon_inner_periods"] * step_divisor + 1,
                step_divisor // 4,
            )
        )
        if item != {"divisor": step_divisor, "indices": expected_indices}:
            raise HoldoutError(f"{label} checkpoint grid changed")
    orbitals = value["derived_orbital_elements_float_hex"]
    orbital_keys = {
        "body_id", "semimajor_axis_float_hex", "eccentricity_float_hex",
        "periapse_float_hex", "period_float_hex", "periapse_timescale_float_hex",
    }
    if type(orbitals) is not list or len(orbitals) != count - 1:
        raise HoldoutError(f"{label} orbital record shape changed")
    for body_id, item in zip(value["body_ids"][1:], orbitals, strict=True):
        if type(item) is not dict or set(item) != orbital_keys or item["body_id"] != body_id:
            raise HoldoutError(f"{label} orbital record order changed")
        for name in orbital_keys - {"body_id"}:
            _finite_float_hex(item[name], f"{label}.{name}")
    _validate_guard_record(value["input_guard_utilizations"], count, f"{label}.guard")
    core = dict(value)
    digest = core.pop("case_content_sha256")
    if not _is_sha256(digest) or digest != _framed_hash(CASE_DOMAIN, label, core):
        raise HoldoutError(f"{label} case digest changed")


def _validate_control_record(
    value: Any, registered: dict[str, Any], required: list[str]
) -> None:
    label = registered["control_id"]
    if type(value) is not dict or set(value) != set(required):
        raise HoldoutError(f"{label} control field roster changed")
    if (
        value["control_id"] != label
        or value["pair_id"] != registered["pair_id"]
        or value["expected"] != registered["expected"]
        or value["execution_mode"] != registered["execution_mode"]
    ):
        raise HoldoutError(f"{label} control identity changed")
    count = registered["body_count"]
    _validate_common_input_record(value, label, count, label)
    if _finite_float_hex(value["fixed_step_float_hex"], f"{label}.fixed_step") <= 0.0:
        raise HoldoutError(f"{label} fixed step changed")
    if value["checkpoint_step_indices"] != [0, 1]:
        raise HoldoutError(f"{label} checkpoint schedule changed")
    target = {
        "ECCENTRICITY_GUARD": "eccentricity",
        "PERIAPSE_STEP_GUARD": "step",
        "TOTAL_SECONDARY_GM_GUARD": "mass",
        "INITIAL_NODE_HILL_GUARD": "node",
        "MAXIMUM_BODY_COUNT_GUARD": "body_count",
    }[registered["pair_id"]]
    guard = value["reconstructed_initial_guard_metrics_float_hex"]
    _validate_guard_record(guard, count, f"{label}.guard")
    if guard["all_non_target_initial_guards_inside"] is not True:
        raise HoldoutError(f"{label} has a failed non-target input guard")
    core = dict(value)
    digest = core.pop("control_content_sha256")
    if not _is_sha256(digest) or digest != _framed_hash(CONTROL_DOMAIN, label, core):
        raise HoldoutError(f"{label} control digest changed")
    del target


def validate_input_manifest(
    value: Any,
    beacon_binding: dict[str, Any],
    source_registration: Path,
    *,
    recompute: bool = True,
) -> dict[str, Any]:
    _assert_pre_freeze_import_boundary()
    if type(value) is not dict:
        raise HoldoutError("input manifest must be an object")
    _validate_content_integrity(value, MANIFEST_SCHEMA, MANIFEST_DOMAIN)
    protocol, _ = _protocol()
    binding = validate_beacon_binding(beacon_binding, source_registration)
    source_value, source_record = _source_registration(source_registration)
    beacon_record = _beacon_publication_record(binding, source_record)
    if (
        set(value)
        != {
            "schema", "holdout_id", "protocol", "source_registration",
            "beacon_binding", "seed_sha256", "constructor",
            "cases", "guard_controls", "claim_controls", "content_integrity",
        }
        or value["holdout_id"] != protocol["holdout_id"]
        or value["protocol"] != binding["protocol"]
        or value["source_registration"] != source_record
        or value["beacon_binding"] != beacon_record
        or value["seed_sha256"] != binding["seed_sha256"]
        or value["claim_controls"]
        != {
            "input_manifest_frozen": True,
            "holdout_execution_completed": False,
            "scientific_outcome_generated": False,
            "scientific_approval_granted": False,
        }
    ):
        raise HoldoutError("input manifest custody binding changed")
    constructor = value["constructor"]
    if (
        type(constructor) is not dict
        or set(constructor)
        != {
            "mpmath_version", "construction_precision_decimal_digits",
            "mpmath_package_tree",
            "verification_precision_decimal_digits", "dual_precision_bitwise_stable",
            "candidate_or_oracle_imported", "candidate_or_oracle_executed",
            "trace_records",
        }
        or constructor["mpmath_version"] != "1.3.0"
        or constructor["mpmath_package_tree"]
        != source_value["constructor_dependency"]
        or constructor["construction_precision_decimal_digits"] != 160
        or constructor["verification_precision_decimal_digits"] != 220
        or constructor["dual_precision_bitwise_stable"] is not True
        or constructor["candidate_or_oracle_imported"] is not False
        or constructor["candidate_or_oracle_executed"] is not False
    ):
        raise HoldoutError("input constructor declaration changed")
    freeze = protocol["input_manifest_freeze"]
    cases = value["cases"]
    controls = value["guard_controls"]
    if (
        type(cases) is not list
        or len(cases) != freeze["required_case_count"]
        or type(controls) is not list
        or len(controls) != freeze["required_guard_control_count"]
    ):
        raise HoldoutError("input manifest case/control count changed")
    for case, roster in zip(cases, protocol["case_roster"], strict=True):
        _validate_case_record(case, roster, freeze["required_fields_per_case"])
    registered_controls = protocol["guard_semantics_controls"]["controls"]
    for control, registered in zip(controls, registered_controls, strict=True):
        _validate_control_record(
            control, registered, freeze["required_fields_per_guard_control"]
        )
    expected_ids = [item["case_id"] for item in protocol["case_roster"]] + [
        item["control_id"] for item in registered_controls
    ]
    traces = constructor["trace_records"]
    if type(traces) is not list or len(traces) != len(expected_ids):
        raise HoldoutError("constructor trace roster changed")
    for trace, expected_id in zip(traces, expected_ids, strict=True):
        if (
            type(trace) is not dict
            or set(trace) != {"id", "cell_index", "branch_trace_sha256"}
            or trace["id"] != expected_id
            or not _is_sha256(trace["branch_trace_sha256"])
            or (
                trace["cell_index"] is not None
                and (type(trace["cell_index"]) is not int or not 0 <= trace["cell_index"] <= 3)
            )
        ):
            raise HoldoutError("constructor trace record changed")
    if recompute:
        expected = construct_input_manifest(binding, source_registration)
        if _canonical_bytes(value) != _canonical_bytes(expected):
            raise HoldoutError("input manifest differs from full deterministic reconstruction")
    return value


def _require_fixed_path(value: Path, expected: Path, label: str) -> Path:
    if value.resolve(strict=False) != expected.resolve(strict=False):
        raise HoldoutError(f"{label} must use the registered fixed repository path")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Construct and verify the source-bound V9 hierarchical holdout inputs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    bind = subparsers.add_parser("bind-beacon")
    bind.add_argument("--raw-response", type=Path, required=True)
    bind.add_argument("--source-registration", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    construct = subparsers.add_parser("construct-inputs")
    construct.add_argument("--beacon-binding", type=Path, required=True)
    construct.add_argument("--source-registration", type=Path, required=True)
    construct.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify-inputs")
    verify.add_argument("--beacon-binding", type=Path, required=True)
    verify.add_argument("--source-registration", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    return parser


def main(arguments: list[str] | None = None) -> int:
    namespace = _parser().parse_args(arguments)
    _assert_pre_freeze_import_boundary()
    if namespace.command == "bind-beacon":
        output = _require_fixed_path(namespace.output, BEACON_BINDING_PATH, "beacon binding output")
        binding = create_beacon_binding(
            _require_fixed_path(namespace.raw_response, BEACON_RAW_PATH, "beacon response"),
            _require_fixed_path(
                namespace.source_registration,
                SOURCE_REGISTRATION_PATH,
                "source registration",
            ),
        )
        _write_new_file(output, _canonical_bytes(binding, newline=True))
        print(binding["content_integrity"]["content_sha256"])
        return 0
    binding_path = _require_fixed_path(
        namespace.beacon_binding, BEACON_BINDING_PATH, "beacon binding"
    )
    binding, _ = _load_canonical_json(
        binding_path, MAX_JSON_BYTES, "beacon binding"
    )
    source_path = _require_fixed_path(
        namespace.source_registration,
        SOURCE_REGISTRATION_PATH,
        "source registration",
    )
    if namespace.command == "construct-inputs":
        output = _require_fixed_path(namespace.output, INPUT_MANIFEST_PATH, "input manifest output")
        manifest = construct_input_manifest(binding, source_path)
        validate_input_manifest(
            manifest, binding, source_path, recompute=False
        )
        _write_new_file(output, _canonical_bytes(manifest, newline=True))
        print(manifest["content_integrity"]["content_sha256"])
        return 0
    manifest_path = _require_fixed_path(
        namespace.manifest, INPUT_MANIFEST_PATH, "input manifest"
    )
    manifest, _ = _load_canonical_json(
        manifest_path, MAX_JSON_BYTES, "input manifest"
    )
    validate_input_manifest(manifest, binding, source_path, recompute=True)
    print(manifest["content_integrity"]["content_sha256"])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except HoldoutError as error:
        print(f"INVALID_INPUT_CONSTRUCTION: {error}", file=sys.stderr)
        raise SystemExit(2) from None
