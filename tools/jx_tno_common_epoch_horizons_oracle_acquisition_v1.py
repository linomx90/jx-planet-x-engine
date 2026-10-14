#!/usr/bin/env python3
"""Acquire the preregistered all-12 common-epoch Horizons oracle.

This executor performs the single frozen HTTPS pass, retains exact response
entity bytes, and publishes a domain-bound custody record.  It does not run
dynamics, compare residuals, classify objects, or produce a science result.
The ``validate-result`` command is a read-only, no-network replay validator.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import queue
import re
import socket
import ssl
import stat
import subprocess
import sys
import threading
import time
import types
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_RELATIVE_PATH = "runs/tno_common_epoch_horizons_oracle_v1_result"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / RESULT_RELATIVE_PATH
PREREG_SOURCE_PATH = (
    PROJECT_ROOT / "tools" / "jx_tno_common_epoch_horizons_oracle_v1.py"
)
PREREG_TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_tno_common_epoch_horizons_oracle_v1.py"
)
PREREG_PROTOCOL_PATH = (
    PROJECT_ROOT / "runs" / "tno_common_epoch_horizons_oracle_v1" / "PROTOCOL.json"
)
MATRIX_MANIFEST_PATH = PROJECT_ROOT / "tools" / "test_matrix_manifest.json"
SUCCESSOR_SOURCE_PATH = (
    PROJECT_ROOT
    / "tools"
    / "jx_tno_common_epoch_horizons_oracle_acquisition_v1.py"
)
SUCCESSOR_TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_tno_common_epoch_horizons_oracle_acquisition_v1.py"
)
SUCCESSOR_BRANCH = (
    "research/tno-common-epoch-horizons-oracle-acquisition-v1-20261014"
)
SUCCESSOR_SOURCE_RELATIVE_PATH = (
    "tools/jx_tno_common_epoch_horizons_oracle_acquisition_v1.py"
)
SUCCESSOR_TEST_RELATIVE_PATH = (
    "tests/test_tno_common_epoch_horizons_oracle_acquisition_v1.py"
)

PREREG_TAG = "tno-common-epoch-horizons-oracle-v1-preregistered-20261014"
PREREG_TAG_OBJECT = "cb5bc0da8bd761d9ca400d9021937dd7b795bcac"
PREREG_COMMIT = "3a2df6013e60efc54e4ab8ef566fbd4d80370fa9"
PREREG_TREE = "a1dd4b80d38668f58a5c2bb0c346eb75cdacfd5e"
PREREG_TAG_PAYLOAD_BYTES = 254
PREREG_TAG_PAYLOAD_SHA256 = (
    "afb257367709d9dafc18a09630d001c8a7ec3a08d4d66f2341e634dc7eaf3ec4"
)

PREREG_SOURCE_RELATIVE_PATH = "tools/jx_tno_common_epoch_horizons_oracle_v1.py"
PREREG_SOURCE_BYTES = 74_892
PREREG_SOURCE_SHA256 = (
    "0fce70c8f8eeaad60e35db53c64aa446503b48c6e4b39648df369655d11147d0"
)
PREREG_TEST_RELATIVE_PATH = "tests/test_tno_common_epoch_horizons_oracle_v1.py"
PREREG_TEST_BYTES = 28_052
PREREG_TEST_SHA256 = (
    "cff7e39e2833e1bec388d3d1d6d791f10063b7bc348618e5f12f66dc2cd86f1c"
)
PREREG_PROTOCOL_RELATIVE_PATH = (
    "runs/tno_common_epoch_horizons_oracle_v1/PROTOCOL.json"
)
PREREG_PROTOCOL_BYTES = 52_597
PREREG_PROTOCOL_SHA256 = (
    "69b445578e38ae5d4605b78f5d4a83d14bdf6ba28a9bab1709b63ecf44c1f5a8"
)
PREREG_PROTOCOL_CONTENT_SHA256 = (
    "06aee564cdc8c41be671129da96ed9a531423f68238de972c78533c32d1036cc"
)
PREREG_MANIFEST_RELATIVE_PATH = "tools/test_matrix_manifest.json"
PREREG_MANIFEST_BYTES = 13_067
PREREG_MANIFEST_SHA256 = (
    "eccd3a1173b4304978cb9eee261eba4d182e792b4f284cff9249795dc2983187"
)
SUCCESSOR_CHANGESET = frozenset(
    {
        ("A", SUCCESSOR_SOURCE_RELATIVE_PATH),
        ("A", SUCCESSOR_TEST_RELATIVE_PATH),
        ("M", PREREG_MANIFEST_RELATIVE_PATH),
    }
)
MAX_SUCCESSOR_SOURCE_BYTES = 4_194_304
MAX_SUCCESSOR_TEST_BYTES = 4_194_304
MAX_SUCCESSOR_MANIFEST_BYTES = 1_048_576

CUSTODY_SCHEMA = "jx.tno.common-epoch-horizons-oracle.custody.v1"
CUSTODY_STATUS = (
    "RETAINED_COMMON_EPOCH_HORIZONS_NOMINAL_ORACLE_DIAGNOSTIC_INPUT_ONLY"
)
CUSTODY_DOMAIN = b"jx.tno.common-epoch-horizons-oracle.custody.v1\0"
MAX_RESPONSE_BYTES = 1_048_576
# The frozen raw ceiling applies independently to all 12 responses.  A retained
# response byte can expand by at most 6x when canonical JSON escapes literal
# ASCII DEL as ``\u007f``.  Seventy-two raw ceilings therefore cover all
# response-derived text; eight more cover the exact fixed schema and metadata.
MAX_CUSTODY_BYTES = 80 * MAX_RESPONSE_BYTES
STREAM_CHUNK_BYTES = 65_536
CONNECT_TIMEOUT_SECONDS = 30.0
READ_IDLE_TIMEOUT_SECONDS = 30.0
TOTAL_TIMEOUT_SECONDS = 120.0
EXPECTED_HOST = "ssd.jpl.nasa.gov"
EXPECTED_PATH = "/api/horizons.api"
REQUEST_HEADERS = (
    ("Accept", "application/json"),
    ("Accept-Encoding", "identity"),
    ("User-Agent", "JX-TNO-common-epoch-horizons-oracle-v1"),
)

_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_LENGTH = re.compile(r"^(?:0|[1-9][0-9]*)$")
_PREREG_MODULE: types.ModuleType | None = None


class AcquisitionError(RuntimeError):
    """A preregistration, transport, response, or publication gate failed."""


def _canonical_json(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise AcquisitionError("value is not strict canonical JSON") from exc
    return text.encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise AcquisitionError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise AcquisitionError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise AcquisitionError(f"JSON contains non-finite constant {token}")


def _strict_json(payload: bytes, label: str, maximum_bytes: int) -> dict[str, Any]:
    if (
        type(payload) is not bytes
        or type(maximum_bytes) is not int
        or maximum_bytes <= 0
        or not 0 < len(payload) <= maximum_bytes
    ):
        raise AcquisitionError(f"{label} has an invalid bounded byte length")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except AcquisitionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise AcquisitionError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise AcquisitionError(f"{label} root must be an exact object")
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_atime_ns,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _close_fd(descriptor: int) -> None:
    """Close once; an ambiguous close error must not block later cleanup."""
    try:
        os.close(descriptor)
    except OSError:
        pass


def _read_regular_path(path: Path, label: str, maximum_bytes: int) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOATIME", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        path_before = path.lstat()
    except OSError as exc:
        raise AcquisitionError(f"cannot inspect {label}") from exc
    if (
        not stat.S_ISREG(path_before.st_mode)
        or path_before.st_nlink != 1
        or not 0 < path_before.st_size <= maximum_bytes
    ):
        raise AcquisitionError(f"{label} is linked, nonregular, empty, or oversized")
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AcquisitionError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise AcquisitionError(f"{label} is linked, nonregular, empty, or oversized")
        if _stat_identity(path_before) != _stat_identity(before):
            raise AcquisitionError(f"{label} path identity changed while opened")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.pread(
                descriptor,
                min(STREAM_CHUNK_BYTES, before.st_size - len(payload)),
                len(payload),
            )
            if not block:
                raise AcquisitionError(f"{label} ended before its recorded length")
            payload.extend(block)
        after = os.fstat(descriptor)
    finally:
        _close_fd(descriptor)
    try:
        path_after = path.lstat()
    except OSError as exc:
        raise AcquisitionError(f"{label} path disappeared") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise AcquisitionError(f"{label} changed while held")
    if _stat_identity(before) != _stat_identity(path_after):
        raise AcquisitionError(f"{label} path identity changed")
    return bytes(payload)


def _require_exact_bytes(
    payload: bytes, expected_bytes: int, expected_sha256: str, label: str
) -> None:
    if len(payload) != expected_bytes or _sha256(payload) != expected_sha256:
        raise AcquisitionError(f"{label} bytes differ from the preregistration")


def _git(*arguments: str) -> bytes:
    environment = dict(os.environ)
    unsafe_git_environment = {
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_CONFIG",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_NAMESPACE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    }
    for key in tuple(environment):
        if (
            key in unsafe_git_environment
            or key == "GIT_CONFIG_COUNT"
            or key.startswith("GIT_CONFIG_KEY_")
            or key.startswith("GIT_CONFIG_VALUE_")
        ):
            environment.pop(key, None)
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "LC_ALL": "C",
        }
    )
    try:
        completed = subprocess.run(
            (
                "git",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "core.untrackedCache=false",
                *arguments,
            ),
            cwd=PROJECT_ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AcquisitionError("local git identity inspection failed") from exc
    if completed.returncode != 0:
        raise AcquisitionError("local git identity inspection failed")
    return completed.stdout


def _preregistration_identity() -> dict[str, object]:
    return {
        "commit": PREREG_COMMIT,
        "protocol_content_sha256": PREREG_PROTOCOL_CONTENT_SHA256,
        "protocol_file_bytes": PREREG_PROTOCOL_BYTES,
        "protocol_file_sha256": PREREG_PROTOCOL_SHA256,
        "source_bytes": PREREG_SOURCE_BYTES,
        "source_sha256": PREREG_SOURCE_SHA256,
        "tag": PREREG_TAG,
        "tag_object": PREREG_TAG_OBJECT,
        "tag_payload_bytes": PREREG_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": PREREG_TAG_PAYLOAD_SHA256,
        "test_bytes": PREREG_TEST_BYTES,
        "test_sha256": PREREG_TEST_SHA256,
        "tree": PREREG_TREE,
    }


def _prereg_module() -> types.ModuleType:
    global _PREREG_MODULE
    payload = _read_regular_path(
        PREREG_SOURCE_PATH, "preregistration source", PREREG_SOURCE_BYTES
    )
    _require_exact_bytes(
        payload, PREREG_SOURCE_BYTES, PREREG_SOURCE_SHA256, "preregistration source"
    )
    if _PREREG_MODULE is not None:
        return _PREREG_MODULE
    module_name = "_jx_tno_common_epoch_horizons_oracle_v1_bound"
    module = types.ModuleType(module_name)
    module.__file__ = str(PREREG_SOURCE_PATH)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(payload, str(PREREG_SOURCE_PATH), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    _PREREG_MODULE = module
    return module


def _verify_preregistration() -> tuple[dict[str, object], types.ModuleType]:
    if _git("cat-file", "-t", PREREG_TAG_OBJECT) != b"tag\n":
        raise AcquisitionError("preregistration object is not an annotated tag")
    if _git("rev-parse", f"refs/tags/{PREREG_TAG}").strip() != PREREG_TAG_OBJECT.encode():
        raise AcquisitionError("preregistration tag ref differs")
    tag_payload = _git("cat-file", "tag", PREREG_TAG_OBJECT)
    _require_exact_bytes(
        tag_payload,
        PREREG_TAG_PAYLOAD_BYTES,
        PREREG_TAG_PAYLOAD_SHA256,
        "annotated tag payload",
    )
    if _git("rev-parse", f"{PREREG_TAG}^{{}}").strip() != PREREG_COMMIT.encode():
        raise AcquisitionError("preregistration peeled commit differs")
    if _git("cat-file", "-t", PREREG_COMMIT) != b"commit\n":
        raise AcquisitionError("preregistration target is not a commit")
    if _git("rev-parse", f"{PREREG_COMMIT}^{{tree}}").strip() != PREREG_TREE.encode():
        raise AcquisitionError("preregistration tree differs")

    bindings = (
        (
            PREREG_SOURCE_RELATIVE_PATH,
            PREREG_SOURCE_PATH,
            PREREG_SOURCE_BYTES,
            PREREG_SOURCE_SHA256,
            True,
        ),
        (
            PREREG_TEST_RELATIVE_PATH,
            PREREG_TEST_PATH,
            PREREG_TEST_BYTES,
            PREREG_TEST_SHA256,
            True,
        ),
        (
            PREREG_PROTOCOL_RELATIVE_PATH,
            PREREG_PROTOCOL_PATH,
            PREREG_PROTOCOL_BYTES,
            PREREG_PROTOCOL_SHA256,
            True,
        ),
        (
            PREREG_MANIFEST_RELATIVE_PATH,
            MATRIX_MANIFEST_PATH,
            PREREG_MANIFEST_BYTES,
            PREREG_MANIFEST_SHA256,
            False,
        ),
    )
    current_payloads: dict[str, bytes] = {}
    for relative_path, current_path, expected_bytes, expected_hash, require_current in bindings:
        tagged = _git("cat-file", "blob", f"{PREREG_COMMIT}:{relative_path}")
        _require_exact_bytes(tagged, expected_bytes, expected_hash, f"tagged {relative_path}")
        if require_current:
            current = _read_regular_path(current_path, relative_path, expected_bytes)
            _require_exact_bytes(current, expected_bytes, expected_hash, relative_path)
            current_payloads[relative_path] = current

    protocol_payload = current_payloads[PREREG_PROTOCOL_RELATIVE_PATH]
    protocol = _strict_json(
        protocol_payload, "preregistration protocol", PREREG_PROTOCOL_BYTES
    )
    if protocol_payload != _canonical_json(protocol):
        raise AcquisitionError("preregistration protocol is not exact canonical JSON")
    if protocol.get("content_sha256") != PREREG_PROTOCOL_CONTENT_SHA256:
        raise AcquisitionError("preregistration protocol content hash differs")

    module = _prereg_module()
    try:
        module._validate_protocol(protocol)
    except module.HorizonsOracleProtocolError as exc:
        raise AcquisitionError("preregistration protocol semantics differ") from exc
    transport = protocol.get("future_transport_contract")
    acquisition = protocol.get("future_acquisition_contract")
    custody = protocol.get("future_custody_contract")
    publication = protocol.get("future_result_publication_contract")
    replay = protocol.get("future_offline_replay_validator")
    streaming = transport.get("streaming_read") if type(transport) is dict else None
    timeouts = (
        transport.get("timeouts_seconds_per_request")
        if type(transport) is dict
        else None
    )
    if (
        type(transport) is not dict
        or transport.get("ordered_exact_urls")
        != [target.query_url for target in module.TARGETS]
        or transport.get("request_count") != len(module.TARGETS)
        or transport.get("request_order") != list(range(len(module.TARGETS)))
        or transport.get("method") != "GET"
        or transport.get("requests_per_url") != 1
        or transport.get("retries_for_any_reason") != 0
        or transport.get("parallel_requests_forbidden") is not True
        or transport.get("application_request_headers") != dict(REQUEST_HEADERS)
        or type(streaming) is not dict
        or streaming.get("maximum_buffered_bytes") != MAX_RESPONSE_BYTES + 1
        or streaming.get("maximum_chunk_bytes") != STREAM_CHUNK_BYTES
        or type(timeouts) is not dict
        or timeouts.get("connect") != CONNECT_TIMEOUT_SECONDS
        or timeouts.get("read_idle") != READ_IDLE_TIMEOUT_SECONDS
        or timeouts.get("total_wall_clock") != TOTAL_TIMEOUT_SECONDS
        or type(acquisition) is not dict
        or acquisition.get("result_relative_path") != RESULT_RELATIVE_PATH
        or acquisition.get("successor_commit_required") is not True
        or acquisition.get("maximum_raw_response_bytes_each_inclusive")
        != MAX_RESPONSE_BYTES
        or acquisition.get("raw_response_count") != len(module.TARGETS)
        or acquisition.get("raw_response_roster")
        != [f"raw/{target.spk_id}.json" for target in module.TARGETS]
        or type(custody) is not dict
        or custody.get("schema") != CUSTODY_SCHEMA
        or type(publication) is not dict
        or publication.get("custody_is_last_file_and_commit_marker") is not True
        or type(replay) is not dict
        or replay.get("required_successor_cli") != "validate-result"
        or replay.get("network_access") is not False
        or replay.get("read_only") is not True
    ):
        raise AcquisitionError("frozen successor contract differs")
    return _preregistration_identity(), module


def _successor_manifest_bytes(preregistered_manifest: bytes) -> bytes:
    anchor = b'        "tests/test_tno_common_epoch_horizons_oracle_v1.py",\n'
    insertion = (
        b'        "tests/test_tno_common_epoch_horizons_oracle_acquisition_v1.py",\n'
    )
    if (
        preregistered_manifest.count(anchor) != 1
        or insertion in preregistered_manifest
    ):
        raise AcquisitionError("preregistered matrix insertion anchor differs")
    return preregistered_manifest.replace(anchor, insertion + anchor, 1)


def _parse_successor_changes(payload: bytes) -> frozenset[tuple[str, str]]:
    try:
        lines = payload.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise AcquisitionError("successor changeset is not strict ASCII") from exc
    changes: list[tuple[str, str]] = []
    for line in lines:
        fields = line.split("\t")
        if len(fields) != 2 or fields[0] not in {"A", "M", "D", "T"}:
            raise AcquisitionError("successor changeset has an invalid entry")
        changes.append((fields[0], fields[1]))
    if len(changes) != len(set(changes)):
        raise AcquisitionError("successor changeset contains a duplicate entry")
    return frozenset(changes)


def _parse_raw_successor_commit(payload: bytes, expected_sha1: str) -> str:
    if type(payload) is not bytes or _GIT_SHA1.fullmatch(expected_sha1) is None:
        raise AcquisitionError("successor raw commit contract is invalid")
    object_bytes = b"commit " + str(len(payload)).encode("ascii") + b"\0" + payload
    if hashlib.sha1(object_bytes).hexdigest() != expected_sha1:
        raise AcquisitionError("successor raw commit object hash differs")
    header, separator, _message = payload.partition(b"\n\n")
    if separator != b"\n\n" or not header or b"\0" in header or b"\r" in header:
        raise AcquisitionError("successor raw commit header is malformed")
    lines = header.split(b"\n")
    if any(not line for line in lines):
        raise AcquisitionError("successor raw commit header is malformed")
    top_level: list[tuple[bytes, bytes]] = []
    for line in lines:
        if line.startswith(b" "):
            if not top_level:
                raise AcquisitionError("successor raw commit header is malformed")
            continue
        key, delimiter, value = line.partition(b" ")
        if (
            delimiter != b" "
            or not value
            or re.fullmatch(rb"[a-z][a-z0-9-]*", key) is None
        ):
            raise AcquisitionError("successor raw commit header is malformed")
        top_level.append((key, value))
    expected_parent = b"parent " + PREREG_COMMIT.encode("ascii")
    parent_lines = [line for line in lines if line.startswith(b"parent ")]
    if len(lines) < 4 or lines[1] != expected_parent or parent_lines != [expected_parent]:
        raise AcquisitionError(
            "successor raw commit must be the direct single-parent child of the preregistration"
        )
    if (
        top_level[0][0] != b"tree"
        or sum(key == b"tree" for key, _value in top_level) != 1
        or sum(key == b"author" for key, _value in top_level) != 1
        or sum(key == b"committer" for key, _value in top_level) != 1
    ):
        raise AcquisitionError("successor raw commit header is malformed")
    try:
        tree = top_level[0][1].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise AcquisitionError("successor raw tree identity is not strict ASCII") from exc
    if _GIT_SHA1.fullmatch(tree) is None:
        raise AcquisitionError("successor raw tree identity differs")
    return tree


def _verify_successor_commit() -> dict[str, object]:
    if _git("status", "--porcelain=v1", "--untracked-files=all") != b"":
        raise AcquisitionError(
            "successor worktree is not clean; request zero is forbidden"
        )

    try:
        branch = _git("symbolic-ref", "--quiet", "--short", "HEAD").decode(
            "ascii", errors="strict"
        ).strip()
        head = _git("rev-parse", "--verify", "HEAD^{commit}").decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeDecodeError as exc:
        raise AcquisitionError("successor git identity is not strict ASCII") from exc
    if branch != SUCCESSOR_BRANCH:
        raise AcquisitionError("successor branch identity differs")
    if _GIT_SHA1.fullmatch(head) is None:
        raise AcquisitionError("successor commit identity differs")
    tree = _parse_raw_successor_commit(_git("cat-file", "commit", head), head)

    changes = _parse_successor_changes(
        _git(
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-r",
            PREREG_COMMIT,
            head,
        )
    )
    if changes != SUCCESSOR_CHANGESET:
        raise AcquisitionError("successor commit changeset differs")

    preregistered_manifest = _git(
        "cat-file", "blob", f"{PREREG_COMMIT}:{PREREG_MANIFEST_RELATIVE_PATH}"
    )
    _require_exact_bytes(
        preregistered_manifest,
        PREREG_MANIFEST_BYTES,
        PREREG_MANIFEST_SHA256,
        "tagged matrix manifest",
    )
    expected_manifest = _successor_manifest_bytes(preregistered_manifest)
    bindings = (
        (
            SUCCESSOR_SOURCE_RELATIVE_PATH,
            SUCCESSOR_SOURCE_PATH,
            MAX_SUCCESSOR_SOURCE_BYTES,
            None,
        ),
        (
            SUCCESSOR_TEST_RELATIVE_PATH,
            SUCCESSOR_TEST_PATH,
            MAX_SUCCESSOR_TEST_BYTES,
            None,
        ),
        (
            PREREG_MANIFEST_RELATIVE_PATH,
            MATRIX_MANIFEST_PATH,
            MAX_SUCCESSOR_MANIFEST_BYTES,
            expected_manifest,
        ),
    )
    verified: dict[str, object] = {
        "branch": branch,
        "commit": head,
        "parent": PREREG_COMMIT,
    }
    for relative_path, current_path, maximum_bytes, exact_payload in bindings:
        committed = _git("cat-file", "blob", f"{head}:{relative_path}")
        if not 0 < len(committed) <= maximum_bytes:
            raise AcquisitionError(f"committed {relative_path} has an invalid size")
        if exact_payload is not None and committed != exact_payload:
            raise AcquisitionError("successor matrix modification differs")
        current = _read_regular_path(current_path, relative_path, maximum_bytes)
        if current != committed:
            raise AcquisitionError(
                f"held worktree bytes differ from committed {relative_path}"
            )
        stem = (
            "source"
            if relative_path == SUCCESSOR_SOURCE_RELATIVE_PATH
            else "test"
            if relative_path == SUCCESSOR_TEST_RELATIVE_PATH
            else "manifest"
        )
        verified[f"{stem}_bytes"] = len(committed)
        verified[f"{stem}_sha256"] = _sha256(committed)
    if (
        _git("status", "--porcelain=v1", "--untracked-files=all") != b""
        or _git("rev-parse", "--verify", "HEAD^{commit}").strip()
        != head.encode("ascii")
    ):
        raise AcquisitionError("successor worktree or HEAD changed during verification")
    verified["tree"] = tree
    return verified


def _verify_execution_authority(
) -> tuple[dict[str, object], types.ModuleType, dict[str, object]]:
    identity, module = _verify_preregistration()
    successor = _verify_successor_commit()
    return identity, module, successor


def audit_prerequisites() -> dict[str, object]:
    identity, module, successor = _verify_execution_authority()
    return {
        "identity": identity,
        "manifest_bytes": PREREG_MANIFEST_BYTES,
        "manifest_sha256": PREREG_MANIFEST_SHA256,
        "request_count": len(module.TARGETS),
        "status": "PASSED_BEFORE_REQUEST_ZERO",
        "successor": successor,
    }


def _response_record(payload: bytes, target: object, module: types.ModuleType) -> dict[str, object]:
    try:
        record = module.validate_horizons_response(payload, target)
    except module.HorizonsOracleProtocolError as exc:
        raise AcquisitionError(str(exc)) from exc
    except Exception as exc:
        raise AcquisitionError("Horizons response validator failed closed") from exc
    if type(record) is not dict:
        raise AcquisitionError("response validator returned a non-object")
    return record


def _make_custody(
    records: list[dict[str, object]], module: types.ModuleType
) -> dict[str, object]:
    try:
        vector_digest = module.ordered_vector_sha256(records)
    except module.HorizonsOracleProtocolError as exc:
        raise AcquisitionError(str(exc)) from exc
    partial: dict[str, object] = {
        "claims": dict(module.CUSTODY_CLAIMS),
        "nonclaims": dict(module.CUSTODY_NONCLAIMS),
        "ordered_vector_sha256": vector_digest,
        "preregistration_identity": _preregistration_identity(),
        "protocol_content_sha256": PREREG_PROTOCOL_CONTENT_SHA256,
        "protocol_file_sha256": PREREG_PROTOCOL_SHA256,
        "record_count": len(records),
        "records": records,
        "schema": CUSTODY_SCHEMA,
        "status": CUSTODY_STATUS,
    }
    content_record = dict(partial)
    content_sha256 = _sha256(CUSTODY_DOMAIN + _canonical_json(content_record))
    complete = dict(partial)
    complete["content_sha256"] = content_sha256
    return {key: complete[key] for key in module.CUSTODY_TOP_LEVEL_KEYS}


def build_custody_from_payloads(
    payloads: Mapping[int, bytes], module: types.ModuleType | None = None
) -> dict[str, object]:
    bound = _prereg_module() if module is None else module
    expected_ids = {target.spk_id for target in bound.TARGETS}
    if type(payloads) is not dict or set(payloads) != expected_ids:
        raise AcquisitionError("raw response mapping roster differs")
    records: list[dict[str, object]] = []
    for target in bound.TARGETS:
        payload = payloads[target.spk_id]
        if type(payload) is not bytes:
            raise AcquisitionError("raw response mapping values must be exact bytes")
        records.append(_response_record(payload, target, bound))
    return _make_custody(records, bound)


def _remaining_seconds(started: float, clock: Callable[[], float]) -> float:
    observed = clock()
    if type(observed) not in (int, float) or not math.isfinite(float(observed)):
        raise AcquisitionError("monotonic clock returned an invalid value")
    elapsed = float(observed) - started
    if elapsed < 0.0:
        raise AcquisitionError("monotonic clock moved backwards")
    remaining = TOTAL_TIMEOUT_SECONDS - elapsed
    if remaining <= 0.0:
        raise AcquisitionError("Horizons request exceeded the total timeout")
    return remaining


def _remaining_connect_seconds(
    started: float, clock: Callable[[], float]
) -> float:
    observed = clock()
    if type(observed) not in (int, float) or not math.isfinite(float(observed)):
        raise AcquisitionError("monotonic clock returned an invalid value")
    elapsed = float(observed) - started
    if elapsed < 0.0:
        raise AcquisitionError("monotonic clock moved backwards")
    remaining = CONNECT_TIMEOUT_SECONDS - elapsed
    if remaining <= 0.0:
        raise AcquisitionError("Horizons connection exceeded the connect timeout")
    return remaining


def _bounded_resolve(
    host: str,
    port: int,
    timeout_seconds: float,
    *,
    resolver: Callable[..., object] = socket.getaddrinfo,
) -> list[tuple[Any, ...]]:
    if (
        type(host) is not str
        or type(port) is not int
        or type(timeout_seconds) not in (int, float)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
        or not callable(resolver)
    ):
        raise AcquisitionError("DNS resolver contract is invalid")
    outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def resolve_once() -> None:
        try:
            value = resolver(
                host,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
            )
        except BaseException as exc:
            outcome.put((False, exc))
        else:
            outcome.put((True, value))

    worker = threading.Thread(target=resolve_once, daemon=True)
    worker.start()
    worker.join(float(timeout_seconds))
    if worker.is_alive():
        raise AcquisitionError("Horizons DNS resolution exceeded the connect timeout")
    try:
        succeeded, value = outcome.get_nowait()
    except queue.Empty as exc:
        raise AcquisitionError("Horizons DNS resolution returned no result") from exc
    if not succeeded:
        if isinstance(value, BaseException):
            raise AcquisitionError("Horizons DNS resolution failed without retry") from value
        raise AcquisitionError("Horizons DNS resolution failed without retry")
    if type(value) is not list or not value:
        raise AcquisitionError("Horizons DNS resolution returned no addresses")
    addresses: list[tuple[Any, ...]] = []
    for item in value:
        if type(item) is not tuple or len(item) != 5:
            raise AcquisitionError("Horizons DNS resolution returned a malformed address")
        family, socket_type, protocol, canonical_name, socket_address = item
        if (
            family not in (socket.AF_INET, socket.AF_INET6)
            or socket_type != socket.SOCK_STREAM
            or protocol not in (0, socket.IPPROTO_TCP)
            or type(canonical_name) is not str
            or type(socket_address) is not tuple
        ):
            raise AcquisitionError("Horizons DNS resolution returned a malformed address")
        addresses.append(item)
    return addresses


def _single_address_connector(
    address_record: tuple[Any, ...],
) -> Callable[..., socket.socket]:
    family, socket_type, protocol, _canonical_name, socket_address = address_record

    def connect_once(
        address: tuple[str, int],
        timeout: object = socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        if address != (EXPECTED_HOST, 443):
            raise AcquisitionError("HTTPS connection authority differs")
        connected = socket.socket(family, socket_type, protocol)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                connected.settimeout(float(timeout))
            if source_address is not None:
                connected.bind(source_address)
            connected.connect(socket_address)
        except BaseException:
            connected.close()
            raise
        return connected

    return connect_once


def _bounded_connection_setup(connection: object, timeout_seconds: float) -> None:
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
        or not callable(getattr(connection, "connect", None))
        or not callable(getattr(connection, "close", None))
    ):
        raise AcquisitionError("HTTPS connection setup contract is invalid")
    outcome: queue.Queue[tuple[bool, object | None]] = queue.Queue(maxsize=1)
    cancelled = threading.Event()

    def connect_once() -> None:
        try:
            connection.connect()
        except BaseException as exc:
            outcome.put((False, exc))
            return
        if cancelled.is_set():
            try:
                connection.close()
            except Exception:
                pass
            return
        outcome.put((True, None))

    worker = threading.Thread(target=connect_once, daemon=True)
    worker.start()
    worker.join(float(timeout_seconds))
    if worker.is_alive():
        cancelled.set()
        try:
            connection.close()
        except Exception:
            pass
        raise AcquisitionError("Horizons connection exceeded the connect timeout")
    try:
        succeeded, error = outcome.get_nowait()
    except queue.Empty as exc:
        raise AcquisitionError("Horizons connection setup returned no result") from exc
    if not succeeded:
        if isinstance(error, BaseException):
            raise AcquisitionError("Horizons connection failed without retry") from error
        raise AcquisitionError("Horizons connection failed without retry")


def _bounded_initial_transport_setup(
    connection_factory: Callable[..., object], timeout_seconds: float
) -> tuple[ssl.SSLContext, object]:
    if (
        not callable(connection_factory)
        or type(timeout_seconds) not in (int, float)
        or not math.isfinite(float(timeout_seconds))
        or float(timeout_seconds) <= 0.0
    ):
        raise AcquisitionError("HTTPS initial transport setup contract is invalid")
    outcome: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)
    cancelled = threading.Event()

    def construct_once() -> None:
        connection: object | None = None
        try:
            context = ssl.create_default_context()
            if cancelled.is_set():
                return
            connection = connection_factory(
                EXPECTED_HOST,
                port=443,
                timeout=CONNECT_TIMEOUT_SECONDS,
                context=context,
            )
        except BaseException as exc:
            outcome.put((False, exc))
            return
        if cancelled.is_set():
            try:
                connection.close()
            except Exception:
                pass
            return
        outcome.put((True, (context, connection)))

    worker = threading.Thread(target=construct_once, daemon=True)
    worker.start()
    worker.join(float(timeout_seconds))
    if worker.is_alive():
        cancelled.set()
        raise AcquisitionError("Horizons TLS setup exceeded the total timeout")
    try:
        succeeded, value = outcome.get_nowait()
    except queue.Empty as exc:
        raise AcquisitionError("Horizons TLS setup returned no result") from exc
    if not succeeded:
        if isinstance(value, BaseException):
            raise AcquisitionError("Horizons TLS setup failed without retry") from value
        raise AcquisitionError("Horizons TLS setup failed without retry")
    if type(value) is not tuple or len(value) != 2:
        raise AcquisitionError("Horizons TLS setup returned an invalid result")
    context, connection = value
    if not isinstance(context, ssl.SSLContext):
        try:
            connection.close()
        except Exception:
            pass
        raise AcquisitionError("default TLS verification is unavailable")
    return context, connection


def _response_header_values(response: object, name: str) -> list[str]:
    try:
        pairs = response.getheaders()
    except (AttributeError, http.client.HTTPException) as exc:
        raise AcquisitionError("Horizons response headers are unavailable") from exc
    if type(pairs) is not list:
        pairs = list(pairs)
    values: list[str] = []
    for pair in pairs:
        if type(pair) not in (tuple, list) or len(pair) != 2:
            raise AcquisitionError("Horizons response contains a malformed header")
        key, value = pair
        if type(key) is not str or type(value) is not str:
            raise AcquisitionError("Horizons response contains a malformed header")
        if key.casefold() == name.casefold():
            values.append(value)
    return values


def _response_socket(connection: object, response: object) -> object:
    candidate = getattr(connection, "sock", None)
    if candidate is None:
        try:
            candidate = response.fp.raw._sock
        except AttributeError as exc:
            raise AcquisitionError("cannot enforce the Horizons read timeout") from exc
    if not callable(getattr(candidate, "settimeout", None)):
        raise AcquisitionError("cannot enforce the Horizons read timeout")
    return candidate


def _fetch(
    url: str,
    *,
    connection_factory: Callable[..., object] = http.client.HTTPSConnection,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """Perform exactly one direct HTTPS GET and retain decoded entity octets."""
    module = _prereg_module()
    exact_urls = tuple(target.query_url for target in module.TARGETS)
    if type(url) is not str or url not in exact_urls:
        raise AcquisitionError("request URL is not in the frozen exact roster")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != EXPECTED_HOST
        or parsed.hostname != EXPECTED_HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != EXPECTED_PATH
        or not parsed.query
        or parsed.fragment
    ):
        raise AcquisitionError("request URL authority or path differs")
    request_target = parsed.path + "?" + parsed.query
    started_value = clock()
    if type(started_value) not in (int, float) or not math.isfinite(float(started_value)):
        raise AcquisitionError("monotonic clock returned an invalid value")
    started = float(started_value)
    connection: object | None = None
    response: object | None = None
    total_timer: threading.Timer | None = None
    total_expired = threading.Event()
    connection_holder: list[object] = []
    try:
        def expire_total_deadline() -> None:
            total_expired.set()
            active = connection_holder[0] if connection_holder else None
            candidate = getattr(active, "sock", None)
            if candidate is not None:
                try:
                    candidate.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    candidate.close()
                except Exception:
                    pass
            if active is not None:
                try:
                    active.close()
                except Exception:
                    pass

        timer_budget = _remaining_seconds(started, clock)
        total_timer = threading.Timer(timer_budget, expire_total_deadline)
        total_timer.daemon = True
        total_timer.start()
        context, connection = _bounded_initial_transport_setup(
            connection_factory, timer_budget
        )
        connection_holder.append(connection)
        if context.check_hostname is not True or context.verify_mode != ssl.CERT_REQUIRED:
            raise AcquisitionError("default TLS verification is unavailable")
        try:
            connection.auto_open = False
        except (AttributeError, TypeError) as exc:
            raise AcquisitionError("cannot disable implicit HTTPS reconnect") from exc
        if getattr(connection, "auto_open", None) is not False:
            raise AcquisitionError("cannot disable implicit HTTPS reconnect")
        _remaining_seconds(started, clock)
        if total_expired.is_set():
            raise AcquisitionError("Horizons request exceeded the total timeout")
        if connection_factory is http.client.HTTPSConnection:
            resolve_budget = min(
                _remaining_connect_seconds(started, clock),
                _remaining_seconds(started, clock),
            )
            addresses = _bounded_resolve(EXPECTED_HOST, 443, resolve_budget)
            connection._create_connection = _single_address_connector(addresses[0])
            connection.timeout = min(
                _remaining_connect_seconds(started, clock),
                _remaining_seconds(started, clock),
            )
            _bounded_connection_setup(
                connection,
                min(
                    _remaining_connect_seconds(started, clock),
                    _remaining_seconds(started, clock),
                ),
            )
            _remaining_connect_seconds(started, clock)
        else:
            connection.connect()
        socket_object = getattr(connection, "sock", None)
        if socket_object is None or not callable(getattr(socket_object, "settimeout", None)):
            raise AcquisitionError("cannot enforce the Horizons connect timeout")
        remaining_after_connect = _remaining_seconds(started, clock)
        if total_expired.is_set():
            raise AcquisitionError("Horizons request exceeded the total timeout")
        socket_object.settimeout(min(READ_IDLE_TIMEOUT_SECONDS, remaining_after_connect))
        connection.putrequest("GET", request_target, skip_accept_encoding=True)
        for key, value in REQUEST_HEADERS:
            connection.putheader(key, value)
        _remaining_seconds(started, clock)
        if total_expired.is_set():
            raise AcquisitionError("Horizons request exceeded the total timeout")
        connection.endheaders()
        _remaining_seconds(started, clock)
        if total_expired.is_set():
            raise AcquisitionError("Horizons request exceeded the total timeout")
        response = connection.getresponse()
        if type(getattr(response, "status", None)) is not int or response.status != 200:
            raise AcquisitionError(
                f"Horizons returned terminal HTTP status {getattr(response, 'status', None)}"
            )

        encodings = _response_header_values(response, "Content-Encoding")
        if len(encodings) > 1 or (encodings and encodings[0] != "identity"):
            raise AcquisitionError("Horizons Content-Encoding is not absent or identity")
        transfer_encodings = _response_header_values(response, "Transfer-Encoding")
        if len(transfer_encodings) > 1 or (
            transfer_encodings and transfer_encodings[0].casefold() != "chunked"
        ):
            raise AcquisitionError("Horizons Transfer-Encoding cannot be decoded exactly")
        lengths = _response_header_values(response, "Content-Length")
        expected_length: int | None = None
        if lengths:
            if len(lengths) != 1 or _CONTENT_LENGTH.fullmatch(lengths[0]) is None:
                raise AcquisitionError("Horizons Content-Length is duplicate or malformed")
            length_token = lengths[0]
            ceiling_token = str(MAX_RESPONSE_BYTES)
            if len(length_token) > len(ceiling_token) or (
                len(length_token) == len(ceiling_token)
                and length_token > ceiling_token
            ):
                raise AcquisitionError("Horizons Content-Length exceeds the byte ceiling")
            expected_length = int(length_token)

        response_socket = _response_socket(connection, response)
        reader = getattr(response, "read1", None)
        if not callable(reader):
            raise AcquisitionError("Horizons response lacks bounded streaming read1")
        retained = bytearray()
        while True:
            remaining = _remaining_seconds(started, clock)
            response_socket.settimeout(min(READ_IDLE_TIMEOUT_SECONDS, remaining))
            requested = min(
                STREAM_CHUNK_BYTES,
                MAX_RESPONSE_BYTES + 1 - len(retained),
            )
            if requested <= 0:
                raise AcquisitionError("Horizons response exceeds the 1048576-byte ceiling")
            block = reader(requested)
            if type(block) is not bytes:
                raise AcquisitionError("Horizons streaming read did not return exact bytes")
            _remaining_seconds(started, clock)
            if not block:
                break
            if len(block) > requested:
                raise AcquisitionError("Horizons streaming read exceeded its requested bound")
            retained.extend(block)
            if len(retained) > MAX_RESPONSE_BYTES:
                raise AcquisitionError("Horizons response exceeds the 1048576-byte ceiling")
        payload = bytes(retained)
        _remaining_seconds(started, clock)
        if total_expired.is_set():
            raise AcquisitionError("Horizons request exceeded the total timeout")
        if expected_length is not None and len(payload) != expected_length:
            raise AcquisitionError("Horizons Content-Length differs from entity byte count")
        return payload
    except AcquisitionError:
        raise
    except Exception as exc:
        if total_expired.is_set():
            raise AcquisitionError("Horizons request exceeded the total timeout") from exc
        raise AcquisitionError("Horizons request failed without retry") from exc
    finally:
        if total_timer is not None:
            total_timer.cancel()
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in reversed((absolute, *absolute.parents)):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise AcquisitionError(f"cannot inspect {label} ancestors") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise AcquisitionError(f"{label} has a symlink ancestor")


def _normalize_output_root(root: Path) -> Path:
    absolute = Path(os.path.abspath(os.fspath(root)))
    if not absolute.name or absolute.name in (".", ".."):
        raise AcquisitionError("output root must name an absolute child directory")
    try:
        absolute.name.encode("ascii")
    except UnicodeEncodeError as exc:
        raise AcquisitionError("output root name must be ASCII") from exc
    _require_no_symlink_ancestors(absolute.parent, "output root")
    return absolute


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOATIME", 0)
    )


def _traversal_directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_anchored_directory_path(path: Path, label: str) -> list[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if not absolute.is_absolute():
        raise AcquisitionError(f"{label} path is not absolute")
    descriptors: list[int] = []
    try:
        root_fd = os.open(os.path.sep, _traversal_directory_flags())
        descriptors.append(root_fd)
        current_fd = root_fd
        for component in absolute.parts[1:]:
            path_metadata = os.stat(
                component, dir_fd=current_fd, follow_symlinks=False
            )
            if not stat.S_ISDIR(path_metadata.st_mode):
                raise AcquisitionError(f"{label} path contains a non-directory")
            child_fd = os.open(
                component, _traversal_directory_flags(), dir_fd=current_fd
            )
            try:
                held = os.fstat(child_fd)
                if (
                    not stat.S_ISDIR(held.st_mode)
                    or (held.st_dev, held.st_ino)
                    != (path_metadata.st_dev, path_metadata.st_ino)
                ):
                    raise AcquisitionError(
                        f"{label} path identity changed while opened"
                    )
            except BaseException:
                _close_fd(child_fd)
                raise
            descriptors.append(child_fd)
            current_fd = child_fd
    except OSError as exc:
        for descriptor in reversed(descriptors):
            _close_fd(descriptor)
        raise AcquisitionError(f"cannot open anchored {label} path") from exc
    except BaseException:
        for descriptor in reversed(descriptors):
            _close_fd(descriptor)
        raise
    try:
        _reanchor_directory_path(absolute, descriptors, label)
    except BaseException:
        for descriptor in reversed(descriptors):
            _close_fd(descriptor)
        raise
    return descriptors


def _reanchor_directory_path(path: Path, descriptors: Sequence[int], label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    components = absolute.parts[1:]
    if len(descriptors) != len(components) + 1:
        raise AcquisitionError(f"held {label} path depth differs")
    try:
        root_named = os.stat(os.path.sep, follow_symlinks=False)
        root_held = os.fstat(descriptors[0])
        if (
            not stat.S_ISDIR(root_named.st_mode)
            or not stat.S_ISDIR(root_held.st_mode)
            or (root_named.st_dev, root_named.st_ino)
            != (root_held.st_dev, root_held.st_ino)
        ):
            raise AcquisitionError(f"held {label} root identity differs")
        for index, component in enumerate(components):
            named = os.stat(
                component,
                dir_fd=descriptors[index],
                follow_symlinks=False,
            )
            held = os.fstat(descriptors[index + 1])
            if (
                not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(held.st_mode)
                or (named.st_dev, named.st_ino) != (held.st_dev, held.st_ino)
            ):
                raise AcquisitionError(
                    f"{label} path no longer identifies the held directory chain"
                )
    except OSError as exc:
        raise AcquisitionError(f"cannot re-anchor {label} path") from exc


def _close_directory_path(descriptors: Sequence[int]) -> None:
    for descriptor in reversed(descriptors):
        _close_fd(descriptor)


def _validate_private_directory(metadata: os.stat_result, label: str) -> None:
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise AcquisitionError(f"{label} is not a real owner-private 0700 directory")


def _reanchor_directory(
    parent_fd: int, name: str, held_fd: int, label: str
) -> None:
    try:
        held = os.fstat(held_fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionError(f"cannot re-anchor {label}") from exc
    _validate_private_directory(held, f"held {label}")
    _validate_private_directory(named, f"named {label}")
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise AcquisitionError(f"{label} dirent no longer identifies the held directory")


def _open_directory_at(
    parent_fd: int, name: str, label: str
) -> tuple[int, os.stat_result]:
    try:
        path_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot open {label} without following links") from exc
    try:
        held = os.fstat(descriptor)
        _validate_private_directory(held, label)
        if _stat_identity(path_metadata) != _stat_identity(held):
            raise AcquisitionError(f"{label} path identity changed while opened")
    except BaseException:
        _close_fd(descriptor)
        raise
    return descriptor, held


def _open_held_file_at(
    directory_fd: int, name: str, label: str, maximum_bytes: int
) -> tuple[int, os.stat_result, bytes]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOATIME", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        path_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionError(f"cannot inspect {label} without following links") from exc
    if (
        not stat.S_ISREG(path_metadata.st_mode)
        or stat.S_IMODE(path_metadata.st_mode) != 0o600
        or path_metadata.st_uid != os.geteuid()
        or path_metadata.st_nlink != 1
        or not 0 < path_metadata.st_size <= maximum_bytes
    ):
        raise AcquisitionError(
            f"{label} is not an owned regular nlink-one bounded 0600 file"
        )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot open {label} without following links") from exc
    try:
        before = os.fstat(descriptor)
        if _stat_identity(path_metadata) != _stat_identity(before):
            raise AcquisitionError(f"{label} path identity changed while opened")
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise AcquisitionError(
                f"{label} is not an owned regular nlink-one bounded 0600 file"
            )
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.pread(
                descriptor,
                min(STREAM_CHUNK_BYTES, before.st_size - len(payload)),
                len(payload),
            )
            if not block:
                raise AcquisitionError(f"{label} ended before its recorded length")
            payload.extend(block)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise AcquisitionError(f"{label} changed while read")
    except BaseException:
        _close_fd(descriptor)
        raise
    return descriptor, before, bytes(payload)


def _recheck_held_file(
    directory_fd: int,
    name: str,
    descriptor: int,
    identity: os.stat_result,
    label: str,
) -> None:
    try:
        held = os.fstat(descriptor)
        path_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionError(f"cannot recheck held {label}") from exc
    expected = _stat_identity(identity)
    if _stat_identity(held) != expected or _stat_identity(path_metadata) != expected:
        raise AcquisitionError(f"held {label} identity changed")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise AcquisitionError("artifact write made no progress")
        view = view[written:]


def _rollback_created_file_at(
    directory_fd: int, name: str, descriptor: int, label: str
) -> None:
    """Unlink only the still-named inode held by *descriptor*."""
    try:
        held = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionError(f"cannot inspect failed {label} for rollback") from exc
    if (
        not stat.S_ISREG(held.st_mode)
        or not stat.S_ISREG(named.st_mode)
        or held.st_uid != os.geteuid()
        or named.st_uid != os.geteuid()
        or held.st_nlink != 1
        or named.st_nlink != 1
        or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise AcquisitionError(f"refused replaced or unsafe failed {label} rollback")
    try:
        os.unlink(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot roll back failed {label}") from exc


def _write_create_only_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    identities: dict[str, os.stat_result],
) -> os.stat_result:
    if type(payload) is not bytes or not payload:
        raise AcquisitionError(f"create-only artifact {name} has invalid bytes")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot create artifact {name}") from exc
    metadata: os.stat_result
    try:
        created = os.fstat(descriptor)
        if (
            not stat.S_ISREG(created.st_mode)
            or created.st_uid != os.geteuid()
            or created.st_nlink != 1
            or created.st_size != 0
        ):
            raise AcquisitionError(f"new artifact {name} identity differs")
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or metadata.st_size != len(payload)
        ):
            raise AcquisitionError(f"created artifact {name} identity differs")
        try:
            path_metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise AcquisitionError(f"created artifact {name} disappeared") from exc
        if _stat_identity(path_metadata) != _stat_identity(metadata):
            raise AcquisitionError(f"created artifact {name} path identity changed")
        identities[name] = metadata
    except BaseException as original:
        try:
            _rollback_created_file_at(
                directory_fd, name, descriptor, f"artifact {name}"
            )
        except BaseException as cleanup_error:
            raise AcquisitionError(
                f"failed artifact {name} could not be transactionally rolled back"
            ) from cleanup_error
        identities.pop(name, None)
        raise original
    finally:
        # All content, metadata, fsync, and path-identity gates have either
        # completed or helper-local rollback has already run.  POSIX close
        # errors are state-ambiguous and must not suppress that outcome.
        _close_fd(descriptor)
    return metadata


def _rollback_created_directory_at(
    parent_fd: int,
    name: str,
    descriptor: int,
    created_identity: os.stat_result | None,
    label: str,
) -> None:
    """Remove an empty just-created directory without touching a replacement."""
    held_fd = descriptor
    opened_for_cleanup = False
    try:
        if held_fd < 0:
            held_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
            opened_for_cleanup = True
        held = os.fstat(held_fd)
        named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(held.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or held.st_uid != os.geteuid()
            or named.st_uid != os.geteuid()
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
            or (
                created_identity is not None
                and (held.st_dev, held.st_ino)
                != (created_identity.st_dev, created_identity.st_ino)
            )
            or os.listdir(held_fd)
        ):
            raise AcquisitionError(f"refused replaced or unsafe failed {label} rollback")
        os.rmdir(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot roll back failed {label}") from exc
    finally:
        if opened_for_cleanup and held_fd >= 0:
            _close_fd(held_fd)


def _create_private_staging(parent_fd: int, final_root: Path) -> tuple[Path, int]:
    prefix = f".{final_root.name}.staging-"
    for _attempt in range(128):
        try:
            name = prefix + os.urandom(16).hex()
        except OSError as exc:
            raise AcquisitionError("cannot generate a private staging name") from exc
        try:
            os.mkdir(name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            continue
        except OSError as exc:
            raise AcquisitionError("cannot create private staging root") from exc
        descriptor = -1
        created_identity: os.stat_result | None = None
        try:
            path_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            created_identity = path_metadata
            descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
            os.fchmod(descriptor, 0o700)
            held = os.fstat(descriptor)
            _validate_private_directory(held, "private staging root")
            if (
                not stat.S_ISDIR(path_metadata.st_mode)
                or (path_metadata.st_dev, path_metadata.st_ino)
                != (held.st_dev, held.st_ino)
            ):
                raise AcquisitionError(
                    "private staging path identity changed while opened"
                )
            _reanchor_directory(parent_fd, name, descriptor, "private staging root")
        except BaseException as original:
            try:
                _rollback_created_directory_at(
                    parent_fd,
                    name,
                    descriptor,
                    created_identity,
                    "private staging root",
                )
            except BaseException as cleanup_error:
                if descriptor >= 0:
                    _close_fd(descriptor)
                raise AcquisitionError(
                    "failed private staging root could not be transactionally rolled back"
                ) from cleanup_error
            if descriptor >= 0:
                _close_fd(descriptor)
            raise original
        return final_root.parent / name, descriptor
    raise AcquisitionError("could not reserve an exclusive private staging name")


def _create_private_directory_at(parent_fd: int, name: str, label: str) -> int:
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot create {label}") from exc
    descriptor = -1
    created_identity: os.stat_result | None = None
    try:
        path_metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        created_identity = path_metadata
        descriptor = os.open(name, _directory_flags(), dir_fd=parent_fd)
        os.fchmod(descriptor, 0o700)
        held = os.fstat(descriptor)
        _validate_private_directory(held, label)
        if (
            not stat.S_ISDIR(path_metadata.st_mode)
            or (path_metadata.st_dev, path_metadata.st_ino)
            != (held.st_dev, held.st_ino)
        ):
            raise AcquisitionError(f"{label} path identity changed while opened")
        _reanchor_directory(parent_fd, name, descriptor, label)
        return descriptor
    except BaseException as original:
        try:
            _rollback_created_directory_at(
                parent_fd, name, descriptor, created_identity, label
            )
        except BaseException as cleanup_error:
            if descriptor >= 0:
                _close_fd(descriptor)
            raise AcquisitionError(
                f"failed {label} could not be transactionally rolled back"
            ) from cleanup_error
        if descriptor >= 0:
            _close_fd(descriptor)
        raise original


def _rename_noreplace(
    source_directory_fd: int,
    source: str,
    destination_directory_fd: int,
    destination: str,
) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise AcquisitionError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        source_directory_fd,
        source.encode("ascii"),
        destination_directory_fd,
        destination.encode("ascii"),
        1,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise AcquisitionError("create-only publication destination already exists")
        raise AcquisitionError(f"renameat2 failed: {os.strerror(observed_errno)}")


def _rename_custody_transactionally(
    stage_fd: int,
    stage_identities: dict[str, os.stat_result],
) -> os.stat_result:
    source = ".CUSTODY.json.partial"
    destination = "CUSTODY.json"
    if source not in stage_identities or destination in stage_identities:
        raise AcquisitionError("staged custody identity roster differs before rename")
    expected = stage_identities[source]
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOATIME", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags, dir_fd=stage_fd)
    except OSError as exc:
        raise AcquisitionError("cannot hold staged custody across rename") from exc
    try:
        held = os.fstat(descriptor)
        named = os.stat(source, dir_fd=stage_fd, follow_symlinks=False)
        if (
            _stat_identity(held) != _stat_identity(expected)
            or _stat_identity(named) != _stat_identity(expected)
        ):
            raise AcquisitionError("staged custody identity changed before rename")
        try:
            _rename_noreplace(stage_fd, source, stage_fd, destination)
            destination_metadata = os.stat(
                destination, dir_fd=stage_fd, follow_symlinks=False
            )
            if (
                not stat.S_ISREG(destination_metadata.st_mode)
                or stat.S_IMODE(destination_metadata.st_mode) != 0o600
                or destination_metadata.st_uid != os.geteuid()
                or destination_metadata.st_nlink != 1
                or (
                    destination_metadata.st_dev,
                    destination_metadata.st_ino,
                    destination_metadata.st_size,
                )
                != (held.st_dev, held.st_ino, held.st_size)
            ):
                raise AcquisitionError("staged custody rename identity differs")
            # Destination first: an asynchronous exception can leave both
            # expected names, but never a renamed file absent from the roster.
            stage_identities[destination] = destination_metadata
            stage_identities.pop(source, None)
            return destination_metadata
        except BaseException as original:
            matching_names: list[str] = []
            for candidate in (source, destination):
                try:
                    candidate_metadata = os.stat(
                        candidate, dir_fd=stage_fd, follow_symlinks=False
                    )
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    raise AcquisitionError(
                        "cannot inspect failed staged custody rename"
                    ) from exc
                if (candidate_metadata.st_dev, candidate_metadata.st_ino) == (
                    held.st_dev,
                    held.st_ino,
                ):
                    if (
                        not stat.S_ISREG(candidate_metadata.st_mode)
                        or stat.S_IMODE(candidate_metadata.st_mode) != 0o600
                        or candidate_metadata.st_uid != os.geteuid()
                        or candidate_metadata.st_nlink != 1
                        or candidate_metadata.st_size != held.st_size
                    ):
                        raise AcquisitionError(
                            "refused unsafe failed staged custody rollback"
                        )
                    matching_names.append(candidate)
            if len(matching_names) != 1:
                raise AcquisitionError(
                    "refused replaced or ambiguous failed staged custody rollback"
                ) from original
            matched = matching_names[0]
            try:
                os.unlink(matched, dir_fd=stage_fd)
                os.fsync(stage_fd)
            except OSError as exc:
                raise AcquisitionError(
                    "cannot roll back failed staged custody rename"
                ) from exc
            stage_identities.pop(matched, None)
            raise original
    finally:
        _close_fd(descriptor)


def _is_sha256(value: object) -> bool:
    return type(value) is str and _HEX_SHA256.fullmatch(value) is not None


def _canonical_binary64_hex(value: object) -> bool:
    if type(value) is not str:
        return False
    try:
        number = float.fromhex(value)
    except ValueError:
        return False
    return math.isfinite(number) and number.hex() == value


def _validate_record_structure(
    record: object, target: object, module: types.ModuleType
) -> None:
    if type(record) is not dict or tuple(sorted(record)) != module.CUSTODY_RECORD_KEYS:
        raise AcquisitionError("custody record key roster differs")
    string_keys = (
        "calendar_tdb",
        "center",
        "corrections",
        "epoch_jd_tdb",
        "frame",
        "full_name",
        "object_id",
        "planetary_ephemeris",
        "primary_designation",
        "query_url",
        "raw_sha256",
        "retained_relative_path",
        "small_body_perturbers",
    )
    if any(type(record.get(key)) is not str for key in string_keys):
        raise AcquisitionError("custody record scalar string type differs")
    if (
        type(record.get("order")) is not int
        or type(record.get("raw_bytes")) is not int
        or type(record.get("solution_id")) is not int
        or type(record.get("spk_id")) is not int
        or type(record.get("tp_binary64_ulp_distance")) is not int
    ):
        raise AcquisitionError("custody record integer type differs")
    if (
        record["order"] != target.order
        or record["object_id"] != target.object_id
        or record["spk_id"] != target.spk_id
        or record["primary_designation"] != target.primary_designation
        or record["full_name"] != target.full_name
        or record["solution_id"] != target.solution_id
        or record["query_url"] != target.query_url
        or record["retained_relative_path"] != f"raw/{target.spk_id}.json"
        or record["center"] != "SUN_NAIF_10"
        or record["corrections"] != "GEOMETRIC_NONE"
        or record["epoch_jd_tdb"] != "2461200.5"
        or record["frame"] != "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000"
        or record["planetary_ephemeris"] != "DE441"
        or record["small_body_perturbers"] != "SB441-N16"
        or not 0 <= record["raw_bytes"] <= MAX_RESPONSE_BYTES
        or not _is_sha256(record["raw_sha256"])
        or record["tp_binary64_ulp_distance"] not in (0, 1)
    ):
        raise AcquisitionError("custody record frozen scalar value differs")
    arrays = (
        ("position_AU_binary64_hex", 3),
        ("position_AU_decimal_tokens", 3),
        ("printed_element_binary64_hex", 6),
        ("printed_element_decimal_tokens", 6),
        ("printed_element_labels", 6),
        ("vector_component_labels", 6),
        ("velocity_AU_per_day_binary64_hex", 3),
        ("velocity_AU_per_day_decimal_tokens", 3),
    )
    for key, length in arrays:
        value = record.get(key)
        if type(value) is not list or len(value) != length or any(
            type(item) is not str for item in value
        ):
            raise AcquisitionError(f"custody record {key} array differs")
    if record["printed_element_labels"] != list(module.ELEMENT_LABELS):
        raise AcquisitionError("custody printed element labels differ")
    if record["vector_component_labels"] != list(module.VECTOR_COMPONENT_LABELS):
        raise AcquisitionError("custody vector component labels differ")
    for key in (
        "position_AU_binary64_hex",
        "printed_element_binary64_hex",
        "velocity_AU_per_day_binary64_hex",
    ):
        if not all(_canonical_binary64_hex(item) for item in record[key]):
            raise AcquisitionError(f"custody record {key} value differs")


def _validate_custody_structure(
    custody: dict[str, Any], module: types.ModuleType
) -> None:
    if tuple(sorted(custody)) != module.CUSTODY_TOP_LEVEL_KEYS:
        raise AcquisitionError("custody top-level key roster differs")
    if custody.get("schema") != CUSTODY_SCHEMA or custody.get("status") != CUSTODY_STATUS:
        raise AcquisitionError("custody schema or status differs")
    if custody.get("claims") != module.CUSTODY_CLAIMS:
        raise AcquisitionError("custody claims differ")
    if custody.get("nonclaims") != module.CUSTODY_NONCLAIMS:
        raise AcquisitionError("custody nonclaims differ")
    if custody.get("preregistration_identity") != _preregistration_identity():
        raise AcquisitionError("custody preregistration identity differs")
    if (
        custody.get("protocol_content_sha256") != PREREG_PROTOCOL_CONTENT_SHA256
        or custody.get("protocol_file_sha256") != PREREG_PROTOCOL_SHA256
        or type(custody.get("record_count")) is not int
        or custody.get("record_count") != len(module.TARGETS)
        or not _is_sha256(custody.get("ordered_vector_sha256"))
        or not _is_sha256(custody.get("content_sha256"))
    ):
        raise AcquisitionError("custody scalar binding differs")
    records = custody.get("records")
    if type(records) is not list or len(records) != len(module.TARGETS):
        raise AcquisitionError("custody record array differs")
    for target, record in zip(module.TARGETS, records, strict=True):
        _validate_record_structure(record, target, module)
    without_content = dict(custody)
    without_content.pop("content_sha256")
    expected_content = _sha256(CUSTODY_DOMAIN + _canonical_json(without_content))
    if custody["content_sha256"] != expected_content:
        raise AcquisitionError("custody domain-separated content hash differs")


def _open_result_root(
    root: Path,
) -> tuple[Path, list[int], int, os.stat_result]:
    absolute = _normalize_output_root(root)
    _require_no_symlink_ancestors(absolute, "result root")
    parent_path = _open_anchored_directory_path(absolute.parent, "result parent")
    parent_fd = parent_path[-1]
    try:
        root_fd, root_identity = _open_directory_at(
            parent_fd, absolute.name, "result root"
        )
    except AcquisitionError as exc:
        _close_directory_path(parent_path)
        try:
            absolute.lstat()
        except FileNotFoundError:
            raise AcquisitionError("no committed result exists") from exc
        raise
    try:
        _reanchor_directory_path(absolute.parent, parent_path, "result parent")
    except BaseException:
        _close_fd(root_fd)
        _close_directory_path(parent_path)
        raise
    return absolute, parent_path, root_fd, root_identity


def _close_held_result_files(
    held_files: Sequence[tuple[int, str, int, os.stat_result, str]],
) -> None:
    for _directory_fd, _name, descriptor, _identity, _label in held_files:
        _close_fd(descriptor)


def _recheck_held_result_files(
    held_files: Sequence[tuple[int, str, int, os.stat_result, str]],
) -> None:
    for directory_fd, name, descriptor, identity, label in held_files:
        _recheck_held_file(directory_fd, name, descriptor, identity, label)


def _validate_result_fds(
    root_fd: int,
    root_identity: os.stat_result,
    raw_fd: int,
    raw_identity: os.stat_result,
    module: types.ModuleType,
) -> tuple[
    dict[str, Any],
    list[tuple[int, str, int, os.stat_result, str]],
]:
    held_files: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        root_roster = frozenset(os.listdir(root_fd))
        if "CUSTODY.json" not in root_roster:
            raise AcquisitionError("no committed result exists: CUSTODY.json is absent")
        if root_roster != frozenset(module.RESULT_ROOT_ROSTER):
            raise AcquisitionError("result root roster differs")
        if frozenset(os.listdir(raw_fd)) != frozenset(module.RESULT_RAW_ROSTER):
            raise AcquisitionError("raw response roster differs")

        custody_fd, custody_identity, custody_payload = _open_held_file_at(
            root_fd, "CUSTODY.json", "CUSTODY.json", MAX_CUSTODY_BYTES
        )
        held_files.append(
            (root_fd, "CUSTODY.json", custody_fd, custody_identity, "CUSTODY.json")
        )
        custody = _strict_json(custody_payload, "CUSTODY.json", MAX_CUSTODY_BYTES)
        if custody_payload != _canonical_json(custody):
            raise AcquisitionError("CUSTODY.json is not exact canonical JSON")
        _validate_custody_structure(custody, module)

        rebuilt_records: list[dict[str, object]] = []
        records = custody["records"]
        for target, retained_record in zip(module.TARGETS, records, strict=True):
            filename = f"{target.spk_id}.json"
            descriptor, identity, payload = _open_held_file_at(
                raw_fd,
                filename,
                f"raw/{filename}",
                MAX_RESPONSE_BYTES,
            )
            held_files.append(
                (raw_fd, filename, descriptor, identity, f"raw/{filename}")
            )
            if (
                len(payload) != retained_record["raw_bytes"]
                or _sha256(payload) != retained_record["raw_sha256"]
            ):
                raise AcquisitionError("raw response byte binding differs")
            rebuilt = _response_record(payload, target, module)
            if rebuilt != retained_record:
                raise AcquisitionError("rebuilt response record differs from custody")
            rebuilt_records.append(rebuilt)

        try:
            vector_digest = module.ordered_vector_sha256(rebuilt_records)
        except module.HorizonsOracleProtocolError as exc:
            raise AcquisitionError(str(exc)) from exc
        if vector_digest != custody["ordered_vector_sha256"]:
            raise AcquisitionError("ordered 12x6 vector digest differs")
        expected_custody = _make_custody(rebuilt_records, module)
        if custody_payload != _canonical_json(expected_custody):
            raise AcquisitionError("CUSTODY.json differs from exact replay")

        if frozenset(os.listdir(root_fd)) != frozenset(module.RESULT_ROOT_ROSTER):
            raise AcquisitionError("result root roster changed during validation")
        if frozenset(os.listdir(raw_fd)) != frozenset(module.RESULT_RAW_ROSTER):
            raise AcquisitionError("raw response roster changed during validation")
        _recheck_held_result_files(held_files)
        _reanchor_directory(root_fd, "raw", raw_fd, "raw directory")
        if _stat_identity(os.fstat(raw_fd)) != _stat_identity(raw_identity):
            raise AcquisitionError("held raw directory identity changed")
        if _stat_identity(os.fstat(root_fd)) != _stat_identity(root_identity):
            raise AcquisitionError("held result root identity changed")
        return expected_custody, held_files
    except OSError as exc:
        _close_held_result_files(held_files)
        raise AcquisitionError("offline result replay failed closed") from exc
    except BaseException:
        _close_held_result_files(held_files)
        raise


def validate_result(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Replay a committed result read-only without DNS, TLS, or HTTP use."""
    module = _prereg_module()
    absolute, parent_path, root_fd, root_identity = _open_result_root(root)
    parent_fd = parent_path[-1]
    raw_fd = -1
    held_files: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        root_roster = frozenset(os.listdir(root_fd))
        if "CUSTODY.json" not in root_roster:
            raise AcquisitionError("no committed result exists: CUSTODY.json is absent")
        if root_roster != frozenset(module.RESULT_ROOT_ROSTER):
            raise AcquisitionError("result root roster differs")
        raw_fd, raw_identity = _open_directory_at(root_fd, "raw", "raw directory")
        expected_custody, held_files = _validate_result_fds(
            root_fd,
            root_identity,
            raw_fd,
            raw_identity,
            module,
        )
        path_identity = os.stat(
            absolute.name, dir_fd=parent_fd, follow_symlinks=False
        )
        if _stat_identity(path_identity) != _stat_identity(root_identity):
            raise AcquisitionError("result root path identity changed")
        _reanchor_directory_path(absolute.parent, parent_path, "result parent")
        _recheck_held_result_files(held_files)
        return expected_custody
    except OSError as exc:
        raise AcquisitionError("offline result replay failed closed") from exc
    finally:
        _close_held_result_files(held_files)
        if raw_fd >= 0:
            _close_fd(raw_fd)
        _close_fd(root_fd)
        _close_directory_path(parent_path)


def _read_staged_payloads(
    raw_fd: int, module: types.ModuleType
) -> dict[int, bytes]:
    if frozenset(os.listdir(raw_fd)) != frozenset(module.RESULT_RAW_ROSTER):
        raise AcquisitionError("staged raw response roster differs")
    payloads: dict[int, bytes] = {}
    for target in module.TARGETS:
        filename = f"{target.spk_id}.json"
        descriptor, identity, payload = _open_held_file_at(
            raw_fd, filename, f"staged raw/{filename}", MAX_RESPONSE_BYTES
        )
        try:
            _recheck_held_file(
                raw_fd, filename, descriptor, identity, f"staged raw/{filename}"
            )
        finally:
            _close_fd(descriptor)
        payloads[target.spk_id] = payload
    return payloads


def _remove_private_staging(
    staging: Path,
    final_root: Path,
    parent_fd: int,
    stage_fd: int,
    raw_fd: int,
    module: types.ModuleType,
    raw_identities: Mapping[str, os.stat_result],
    stage_identities: Mapping[str, os.stat_result],
) -> None:
    if (
        staging.parent != final_root.parent
        or not staging.name.startswith(f".{final_root.name}.staging-")
        or staging == final_root
        or stage_fd < 0
    ):
        raise AcquisitionError("refused unsafe staging cleanup target")

    def require_cleanup_file(
        directory_fd: int,
        name: str,
        identities: Mapping[str, os.stat_result],
        label: str,
    ) -> None:
        if name not in identities:
            raise AcquisitionError(f"refused unexpected entry during {label} cleanup")
        try:
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise AcquisitionError(f"cannot inspect {label} cleanup entry") from exc
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
            or _stat_identity(metadata) != _stat_identity(identities[name])
        ):
            raise AcquisitionError(
                f"refused replaced or unsafe {label} cleanup entry"
            )

    _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
    allowed_raw = frozenset(f"{target.spk_id}.json" for target in module.TARGETS)
    if not set(raw_identities).issubset(allowed_raw):
        raise AcquisitionError("refused invalid raw cleanup identity roster")
    if raw_fd >= 0:
        _reanchor_directory(stage_fd, "raw", raw_fd, "private staging raw directory")
        try:
            raw_names = tuple(os.listdir(raw_fd))
        except OSError as exc:
            raise AcquisitionError("cannot list failed staging raw directory") from exc
        for name in raw_names:
            require_cleanup_file(raw_fd, name, raw_identities, "raw")
        for name in raw_names:
            try:
                os.unlink(name, dir_fd=raw_fd)
            except OSError as exc:
                raise AcquisitionError("could not remove failed staged raw file") from exc
        os.fsync(raw_fd)
        _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
        _reanchor_directory(stage_fd, "raw", raw_fd, "private staging raw directory")
        try:
            os.rmdir("raw", dir_fd=stage_fd)
        except OSError as exc:
            raise AcquisitionError("could not remove failed staging raw directory") from exc

    allowed_stage = frozenset({".CUSTODY.json.partial", "CUSTODY.json"})
    if not set(stage_identities).issubset(allowed_stage):
        raise AcquisitionError("refused invalid staging cleanup identity roster")
    try:
        stage_names = tuple(os.listdir(stage_fd))
    except OSError as exc:
        raise AcquisitionError("cannot list failed staging root") from exc
    for name in stage_names:
        require_cleanup_file(stage_fd, name, stage_identities, "staging")
    for name in stage_names:
        try:
            os.unlink(name, dir_fd=stage_fd)
        except OSError as exc:
            raise AcquisitionError("could not remove failed staged custody file") from exc
    os.fsync(stage_fd)
    _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
    try:
        os.rmdir(staging.name, dir_fd=parent_fd)
    except OSError as exc:
        raise AcquisitionError("could not remove failed private staging tree") from exc


def _acquire_to_root(
    root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Execute the one-pass acquisition and atomically publish a complete result."""
    _identity, module, _successor = _verify_execution_authority()
    final_root = _normalize_output_root(root)
    parent_path = _open_anchored_directory_path(final_root.parent, "output parent")
    parent_fd = parent_path[-1]
    staging: Path | None = None
    stage_fd = -1
    raw_fd = -1
    held_files: list[tuple[int, str, int, os.stat_result, str]] = []
    raw_identities: dict[str, os.stat_result] = {}
    stage_identities: dict[str, os.stat_result] = {}
    published = False
    try:
        _reanchor_directory_path(final_root.parent, parent_path, "output parent")
        try:
            os.stat(final_root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise AcquisitionError("cannot inspect final output root") from exc
        else:
            raise AcquisitionError("final output root already exists")

        staging, stage_fd = _create_private_staging(parent_fd, final_root)
        raw_fd = _create_private_directory_at(
            stage_fd, "raw", "private staging raw directory"
        )
        _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
        _reanchor_directory(
            stage_fd, "raw", raw_fd, "private staging raw directory"
        )
        _reanchor_directory_path(final_root.parent, parent_path, "output parent")

        for target in module.TARGETS:
            payload = _fetch(target.query_url)
            if type(payload) is not bytes:
                raise AcquisitionError("transport returned a non-bytes response")
            if len(payload) > MAX_RESPONSE_BYTES:
                raise AcquisitionError(
                    "Horizons response exceeds the 1048576-byte ceiling"
                )
            _response_record(payload, target, module)
            filename = f"{target.spk_id}.json"
            _write_create_only_at(raw_fd, filename, payload, raw_identities)
        os.fsync(raw_fd)

        staged_payloads = _read_staged_payloads(raw_fd, module)
        custody = build_custody_from_payloads(staged_payloads, module)
        custody_payload = _canonical_json(custody)
        _write_create_only_at(
            stage_fd,
            ".CUSTODY.json.partial",
            custody_payload,
            stage_identities,
        )
        _rename_custody_transactionally(stage_fd, stage_identities)
        os.fsync(stage_fd)
        _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
        _reanchor_directory(
            stage_fd, "raw", raw_fd, "private staging raw directory"
        )
        _reanchor_directory_path(final_root.parent, parent_path, "output parent")
        validated, held_files = _validate_result_fds(
            stage_fd,
            os.fstat(stage_fd),
            raw_fd,
            os.fstat(raw_fd),
            module,
        )
        if validated != custody:
            raise AcquisitionError("staged custody replay differs")
        _recheck_held_result_files(held_files)
        _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
        _reanchor_directory(
            stage_fd, "raw", raw_fd, "private staging raw directory"
        )

        try:
            os.stat(final_root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise AcquisitionError("cannot recheck final output root") from exc
        else:
            raise AcquisitionError("final output root appeared during acquisition")
        _recheck_held_result_files(held_files)
        _reanchor_directory(parent_fd, staging.name, stage_fd, "private staging root")
        _reanchor_directory(
            stage_fd, "raw", raw_fd, "private staging raw directory"
        )
        _reanchor_directory_path(final_root.parent, parent_path, "output parent")
        _rename_noreplace(
            parent_fd, staging.name, parent_fd, final_root.name
        )
        published = True
        _recheck_held_result_files(held_files)
        _reanchor_directory(parent_fd, final_root.name, stage_fd, "final result root")
        _reanchor_directory(
            stage_fd, "raw", raw_fd, "final result raw directory"
        )
        _reanchor_directory_path(final_root.parent, parent_path, "output parent")
        staging = None
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise AcquisitionError(
                "final result is structurally complete but durability is indeterminate; "
                "run validate-result before use"
            ) from exc
        _recheck_held_result_files(held_files)
        _reanchor_directory(parent_fd, final_root.name, stage_fd, "final result root")
        _reanchor_directory(
            stage_fd, "raw", raw_fd, "final result raw directory"
        )
        _reanchor_directory_path(final_root.parent, parent_path, "output parent")
        return validated
    except (AcquisitionError, OSError):
        # Process-control BaseExceptions are crash semantics.  They deliberately
        # bypass handled-failure removal; the frozen contract permits only the
        # private staging tree (or an already complete final rename) to remain.
        _close_held_result_files(held_files)
        held_files = []
        if staging is not None and not published:
            _remove_private_staging(
                staging,
                final_root,
                parent_fd,
                stage_fd,
                raw_fd,
                module,
                raw_identities,
                stage_identities,
            )
            try:
                os.fsync(parent_fd)
            except OSError as exc:
                raise AcquisitionError(
                    "failed staging cleanup durability is indeterminate"
                ) from exc
        raise
    finally:
        _close_held_result_files(held_files)
        if raw_fd >= 0:
            _close_fd(raw_fd)
        if stage_fd >= 0:
            _close_fd(stage_fd)
        _close_directory_path(parent_path)


def acquire() -> dict[str, Any]:
    """Acquire only to the exact preregistered production result path."""
    return _acquire_to_root(DEFAULT_OUTPUT_ROOT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-prerequisites")
    commands.add_parser("acquire")
    validate = commands.add_parser("validate-result")
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "audit-prerequisites":
            value = audit_prerequisites()
        elif arguments.command == "acquire":
            value = acquire()
        else:
            value = validate_result(arguments.output_root)
    except AcquisitionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
