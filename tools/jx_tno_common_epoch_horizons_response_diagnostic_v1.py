#!/usr/bin/env python3
"""Preregister one evidence-preserving Horizons response diagnostic.

This artifact performs no network access.  It freezes a future successor that
may issue exactly one copy of the first failed v1 request, retain the complete
bounded HTTP entity body before semantic diagnosis, and publish diagnostic-only
custody.  The retained response can never qualify an oracle or a bridge.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlencode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BRANCH = "research/tno-common-epoch-horizons-response-diagnostic-v1r1-20261014"
PROTOCOL_ROOT = PROJECT_ROOT / "runs" / "tno_common_epoch_horizons_response_diagnostic_v1"
PROTOCOL_PATH = PROTOCOL_ROOT / "PROTOCOL.json"

PROTOCOL_SCHEMA = "jx.tno.common-epoch-horizons-response-diagnostic.protocol.v1r1"
PROTOCOL_STATUS = (
    "PREPARED_SINGLE_RESPONSE_HORIZONS_DIAGNOSTIC_REPAIR_REVISION_1_NO_NETWORK_NO_ACQUISITION"
)
PROTOCOL_DOMAIN = b"jx.tno.common-epoch-horizons-response-diagnostic.protocol.v1r1\0"

PARENT_BRANCH = "research/tno-common-epoch-horizons-oracle-acquisition-v1-20261014"
PARENT_COMMIT = "35bd04a2e12c6f4a4a4c3527974c6954544fd84a"
PARENT_TREE = "8ca88c837257a04289c208c99b0395a23cc49b6a"
PARENT_TAG = "tno-common-epoch-horizons-oracle-acquisition-v1-preregistered-20261014"
PARENT_TAG_OBJECT = "f2151e8dd4cfc9a74f82f8ed5116a4b1aa8dd129"
PARENT_TAG_PAYLOAD_BYTES = 272
PARENT_TAG_PAYLOAD_SHA256 = (
    "2756945270f938dda0b7326ce55f66dd73a6895e49c9b6d3cf25a6a4f3c194b8"
)

V1_PREP_TAG = "tno-common-epoch-horizons-oracle-v1-preregistered-20261014"
V1_PREP_TAG_OBJECT = "cb5bc0da8bd761d9ca400d9021937dd7b795bcac"
V1_PREP_COMMIT = "3a2df6013e60efc54e4ab8ef566fbd4d80370fa9"
V1_PREP_TREE = "a1dd4b80d38668f58a5c2bb0c346eb75cdacfd5e"
V1_PREP_TAG_PAYLOAD_BYTES = 254
V1_PREP_TAG_PAYLOAD_SHA256 = (
    "afb257367709d9dafc18a09630d001c8a7ec3a08d4d66f2341e634dc7eaf3ec4"
)

SUPERSEDED_TAG = (
    "tno-common-epoch-horizons-response-diagnostic-v1-preregistered-20261014"
)
SUPERSEDED_TAG_OBJECT = "2ef49468359f030c8d836eb6f31b8e494a195869"
SUPERSEDED_COMMIT = "3d09f424d24eab41b3b3dd38f62bfb099b1f8bb5"
SUPERSEDED_TREE = "3a4b5e64d188d4df7fd89159b9e1560d8ecbb104"
SUPERSEDED_TAG_PAYLOAD_BYTES = 271
SUPERSEDED_TAG_PAYLOAD_SHA256 = (
    "8d29e92aa6e4d41f3829e16917d1646d98a52ea849a958634821960fcd8f46c3"
)
SUPERSEDED_SOURCE_BYTES = 50_582
SUPERSEDED_SOURCE_SHA256 = (
    "d89e4d6a8ad64e3dbb00e34e9353fdaa63780b40ee92a03bac5e5856d0fb7987"
)
SUPERSEDED_TEST_BYTES = 23_677
SUPERSEDED_TEST_SHA256 = (
    "f9fefa4272f43139bf8e3a2e5fedb2e346fe17e047772fe075e7d8f2f136615c"
)
SUPERSEDED_PROTOCOL_BYTES = 24_871
SUPERSEDED_PROTOCOL_SHA256 = (
    "5203a567a73827038873dd6d1f3b6c8064b258228a1ca1ac01d628bf8e5706cb"
)
SUPERSEDED_PROTOCOL_CONTENT_SHA256 = (
    "63a2c10694b66daead3e8e2a9841b8c1a8971832d524a9f5be2b61ddc27ea438"
)
SUPERSEDED_MANIFEST_BYTES = 13_214
SUPERSEDED_MANIFEST_SHA256 = (
    "f5c70d571ac91ca32d3c79cea1d434f1de52a2538d71e050ab193571c2f49742"
)

FUTURE_TAG = (
    "tno-common-epoch-horizons-response-diagnostic-v1r1-preregistered-20261014"
)
FUTURE_EXECUTOR_TAG = (
    "tno-common-epoch-horizons-response-diagnostic-executor-v1r1-preregistered-20261014"
)
FUTURE_RESULT_TAG = (
    "tno-common-epoch-horizons-response-diagnostic-v1r1-result-20261014"
)

MAX_RESPONSE_BYTES = 1_048_576
MAX_CUSTODY_BYTES = 8 * MAX_RESPONSE_BYTES
STREAM_CHUNK_BYTES = 65_536
TARGET_JD_TDB = "2461200.5"
TARGET_ORDER = 0
OBJECT_ID = "jpl-20090377"
SPK_ID = 20_090_377
SOLUTION_ID = 51
FULL_NAME = "90377 Sedna (2003 VB12)"
HORIZONS_ENDPOINT = "https://ssd.jpl.nasa.gov/api/horizons.api"
QUERY_PAIRS = (
    ("format", "json"),
    ("COMMAND", "'DES=20090377;'"),
    ("OBJ_DATA", "'NO'"),
    ("MAKE_EPHEM", "'YES'"),
    ("EPHEM_TYPE", "'VECTORS'"),
    ("CENTER", "'500@10'"),
    ("TLIST", "'2461200.5'"),
    ("TLIST_TYPE", "'JD'"),
    ("TIME_TYPE", "'TDB'"),
    ("REF_PLANE", "'ECLIPTIC'"),
    ("REF_SYSTEM", "'J2000'"),
    ("OUT_UNITS", "'AU-D'"),
    ("VEC_TABLE", "'2'"),
    ("VEC_CORR", "'NONE'"),
    ("CSV_FORMAT", "'YES'"),
    ("VEC_LABELS", "'YES'"),
    ("TIME_DIGITS", "'FRACSEC'"),
)
QUERY_URL = HORIZONS_ENDPOINT + "?" + urlencode(QUERY_PAIRS)
EXPECTED_QUERY_URL = (
    "https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND=%27DES%3D20090377%3B%27"
    "&OBJ_DATA=%27NO%27&MAKE_EPHEM=%27YES%27&EPHEM_TYPE=%27VECTORS%27"
    "&CENTER=%27500%4010%27&TLIST=%272461200.5%27&TLIST_TYPE=%27JD%27"
    "&TIME_TYPE=%27TDB%27&REF_PLANE=%27ECLIPTIC%27&REF_SYSTEM=%27J2000%27"
    "&OUT_UNITS=%27AU-D%27&VEC_TABLE=%272%27&VEC_CORR=%27NONE%27"
    "&CSV_FORMAT=%27YES%27&VEC_LABELS=%27YES%27&TIME_DIGITS=%27FRACSEC%27"
)
REQUEST_HEADERS = (
    ("Accept", "application/json"),
    ("Accept-Encoding", "identity"),
    ("User-Agent", "JX-TNO-common-epoch-horizons-oracle-v1"),
)

ATTEMPT_ROOT_RELATIVE = (
    "runs/tno_common_epoch_horizons_response_diagnostic_v1_attempt"
)
RESULT_ROOT_RELATIVE = (
    "runs/tno_common_epoch_horizons_response_diagnostic_v1_result"
)
RAW_RELATIVE = "raw/20090377.json"
ATTEMPT_SCHEMA = "jx.tno.common-epoch-horizons-response-diagnostic.attempt.v1r1"
CUSTODY_SCHEMA = "jx.tno.common-epoch-horizons-response-diagnostic.custody.v1r1"
CUSTODY_STATUS = (
    "RETAINED_SINGLE_HORIZONS_RESPONSE_REPAIR_REVISION_1_DIAGNOSTIC_ONLY"
)
ATTEMPT_DOMAIN_HEX = (
    b"jx.tno.common-epoch-horizons-response-diagnostic.attempt.v1r1\0".hex()
)
CUSTODY_DOMAIN_HEX = (
    b"jx.tno.common-epoch-horizons-response-diagnostic.custody.v1r1\0".hex()
)
PREREGISTRATION_IDENTITY_KEYS = (
    "commit",
    "manifest_bytes",
    "manifest_sha256",
    "protocol_bytes",
    "protocol_content_sha256",
    "protocol_sha256",
    "source_bytes",
    "source_sha256",
    "tag",
    "tag_object",
    "tag_payload_bytes",
    "tag_payload_sha256",
    "test_bytes",
    "test_sha256",
    "tree",
)
PREREGISTRATION_IDENTITY_TYPES = {
    "commit": "LOWERCASE_HEX_STRING_LENGTH_40",
    "manifest_bytes": "POSITIVE_INTEGER_NOT_BOOLEAN",
    "manifest_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
    "protocol_bytes": "POSITIVE_INTEGER_NOT_BOOLEAN",
    "protocol_content_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
    "protocol_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
    "source_bytes": "POSITIVE_INTEGER_NOT_BOOLEAN",
    "source_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
    "tag": "EXACT_STRING",
    "tag_object": "LOWERCASE_HEX_STRING_LENGTH_40",
    "tag_payload_bytes": "POSITIVE_INTEGER_NOT_BOOLEAN",
    "tag_payload_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
    "test_bytes": "POSITIVE_INTEGER_NOT_BOOLEAN",
    "test_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
    "tree": "LOWERCASE_HEX_STRING_LENGTH_40",
}

NATIVE_RAW_RELATIVE = "runs/tno_native_epoch_horizons_oracle_v1/raw/20090377.json"
NATIVE_RAW_BYTES = 6_150
NATIVE_RAW_SHA256 = "ceb546225c25013b72cb21a0fff9cd7f42180d1f496f5b6ab299829d9e3a095e"
SBDB_RAW_RELATIVE = "runs/tno_sbdb_covariance_acquisition_v1/raw/20090377.json"
SBDB_RAW_BYTES = 3_855
SBDB_RAW_SHA256 = "62bad94b65a7ab2472e20d771b5289003eafb9dedc6eca351935b98ba447bcc8"
NATIVE_ELEMENT_EPOCH = "2458411.5"


class DiagnosticProtocolError(RuntimeError):
    """A frozen input or prepare-only publication invariant failed."""


@dataclass(frozen=True, slots=True)
class Binding:
    relative_path: str
    bytes: int
    sha256: str
    role: str


V1_BINDINGS = (
    Binding(
        "tools/jx_tno_common_epoch_horizons_oracle_v1.py",
        74_892,
        "0fce70c8f8eeaad60e35db53c64aa446503b48c6e4b39648df369655d11147d0",
        "v1_prepare_source",
    ),
    Binding(
        "tests/test_tno_common_epoch_horizons_oracle_v1.py",
        28_052,
        "cff7e39e2833e1bec388d3d1d6d791f10063b7bc348618e5f12f66dc2cd86f1c",
        "v1_prepare_test",
    ),
    Binding(
        "runs/tno_common_epoch_horizons_oracle_v1/PROTOCOL.json",
        52_597,
        "69b445578e38ae5d4605b78f5d4a83d14bdf6ba28a9bab1709b63ecf44c1f5a8",
        "v1_prepare_protocol",
    ),
    Binding(
        "tools/jx_tno_common_epoch_horizons_oracle_acquisition_v1.py",
        95_391,
        "bb408c69031f1e3c3d9b94a116f5a55ff48423a501d32a4203d049240f6a7b8f",
        "v1_executor_source",
    ),
    Binding(
        "tests/test_tno_common_epoch_horizons_oracle_acquisition_v1.py",
        72_929,
        "d8244c18af17b7155d424ecea3db4f1d7e5774e276cc785644eed95e5b194ba9",
        "v1_executor_test",
    ),
    Binding(
        NATIVE_RAW_RELATIVE,
        NATIVE_RAW_BYTES,
        NATIVE_RAW_SHA256,
        "retained_same_solution_native_horizons_response",
    ),
    Binding(
        SBDB_RAW_RELATIVE,
        SBDB_RAW_BYTES,
        SBDB_RAW_SHA256,
        "retained_same_solution_sbdb_response",
    ),
)

ALLOWED_PREPARED_CHANGES = frozenset(
    {
        "tools/jx_tno_common_epoch_horizons_response_diagnostic_v1.py",
        "tests/test_tno_common_epoch_horizons_response_diagnostic_v1.py",
        "runs/tno_common_epoch_horizons_response_diagnostic_v1/PROTOCOL.json",
        "tools/test_matrix_manifest.json",
    }
)
PREPARED_PROTOCOL_RELATIVE_PATH = (
    "runs/tno_common_epoch_horizons_response_diagnostic_v1/PROTOCOL.json"
)
PREPARED_INPUT_CHANGES = ALLOWED_PREPARED_CHANGES - {
    PREPARED_PROTOCOL_RELATIVE_PATH
}

_NATIVE_EPOCH_LINE = re.compile(
    r"^\s*EPOCH=\s*2458411\.5\s*!.*\(TDB\).*$", re.MULTILINE
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise DiagnosticProtocolError("value is not strict canonical JSON") from exc


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise DiagnosticProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DiagnosticProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticProtocolError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise DiagnosticProtocolError(f"{label} is not a JSON object")
    return value


def diagnostic_observations(payload: bytes) -> dict[str, Any]:
    """Return total, non-authoritative byte-derived observations.

    Every exact ``bytes`` value at or below the frozen response ceiling maps to
    the same fixed-shape object.  In particular, JSON strings containing lone
    surrogate escapes are hashed as their ASCII canonical JSON string token;
    they are never re-encoded directly as UTF-8.
    """
    if type(payload) is not bytes or len(payload) > MAX_RESPONSE_BYTES:
        raise DiagnosticProtocolError("diagnostic payload is not bounded exact bytes")
    rejected = {
        "element_epoch_line_candidates": [],
        "json_envelope_status": "STRICT_JSON_REJECTED",
        "result_string_sha256": None,
        "state_vector_epoch_token_candidates": [],
    }
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except Exception:
        return rejected
    if type(value) is not dict:
        record = dict(rejected)
        record["json_envelope_status"] = "STRICT_NON_OBJECT"
        return record
    result = value.get("result")
    if type(result) is not str:
        record = dict(rejected)
        record["json_envelope_status"] = "STRICT_OBJECT"
        return record
    try:
        result_token = json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("ascii")
        epoch_lines = [
            line for line in result.splitlines() if line.lstrip().startswith("EPOCH=")
        ]
        vector_tokens: list[str] = []
        if result.count("$$SOE") == 1 and result.count("$$EOE") == 1:
            start = result.index("$$SOE") + len("$$SOE")
            stop = result.index("$$EOE")
            if start <= stop:
                vector_tokens = [
                    line.split(",", 1)[0].strip()
                    for line in result[start:stop].splitlines()
                    if line.strip()
                ]
        return {
            "element_epoch_line_candidates": epoch_lines,
            "json_envelope_status": "STRICT_OBJECT",
            "result_string_sha256": _sha256(result_token),
            "state_vector_epoch_token_candidates": vector_tokens,
        }
    except Exception:
        return rejected


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


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
        raise DiagnosticProtocolError("cannot execute local git") from exc
    if completed.returncode != 0:
        raise DiagnosticProtocolError("local git identity verification failed")
    return completed.stdout


def _read_binding(binding: Binding) -> bytes:
    path = PROJECT_ROOT / binding.relative_path
    try:
        before = path.lstat()
    except OSError as exc:
        raise DiagnosticProtocolError(f"cannot inspect {binding.role}") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise DiagnosticProtocolError(f"{binding.role} is not a single-link regular file")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise DiagnosticProtocolError(f"cannot read {binding.role}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise DiagnosticProtocolError(f"{binding.role} changed while read")
    if len(payload) != binding.bytes or _sha256(payload) != binding.sha256:
        raise DiagnosticProtocolError(f"{binding.role} byte binding differs")
    return payload


def _verify_tag(
    name: str,
    object_sha: str,
    commit: str,
    tree: str,
    payload_bytes: int,
    payload_sha256: str,
) -> None:
    if _git("cat-file", "-t", object_sha) != b"tag\n":
        raise DiagnosticProtocolError(f"{name} is not an annotated tag object")
    if _git("rev-parse", f"refs/tags/{name}").strip() != object_sha.encode("ascii"):
        raise DiagnosticProtocolError(f"{name} ref differs")
    payload = _git("cat-file", "tag", object_sha)
    if len(payload) != payload_bytes or _sha256(payload) != payload_sha256:
        raise DiagnosticProtocolError(f"{name} payload differs")
    if _git("rev-parse", f"{name}^{{}}").strip() != commit.encode("ascii"):
        raise DiagnosticProtocolError(f"{name} peeled commit differs")
    if _git("rev-parse", f"{commit}^{{tree}}").strip() != tree.encode("ascii"):
        raise DiagnosticProtocolError(f"{name} tree differs")


def _git_object_sha1(object_type: bytes, payload: bytes) -> str:
    if object_type not in {b"commit", b"tag", b"tree", b"blob"}:
        raise DiagnosticProtocolError("Git object type differs")
    framed = object_type + b" " + str(len(payload)).encode("ascii") + b"\0" + payload
    try:
        digest = hashlib.sha1(framed, usedforsecurity=False)
    except TypeError:
        digest = hashlib.sha1(framed)
    return digest.hexdigest()


def _raw_single_parent_commit_tree(
    commit: str, expected_parent: str, label: str
) -> str:
    if (
        re.fullmatch(r"[0-9a-f]{40}", commit) is None
        or re.fullmatch(r"[0-9a-f]{40}", expected_parent) is None
    ):
        raise DiagnosticProtocolError(f"{label} commit identity is malformed")
    payload = _git("cat-file", "commit", commit)
    if _git_object_sha1(b"commit", payload) != commit:
        raise DiagnosticProtocolError(f"{label} raw commit object hash differs")
    separator = payload.find(b"\n\n")
    if separator <= 0:
        raise DiagnosticProtocolError(f"{label} raw commit header is malformed")
    header_lines = payload[:separator].split(b"\n")
    tree_lines = [line for line in header_lines if line.startswith(b"tree ")]
    parent_lines = [line for line in header_lines if line.startswith(b"parent ")]
    if (
        len(header_lines) < 2
        or len(tree_lines) != 1
        or len(parent_lines) != 1
        or header_lines[0] != tree_lines[0]
        or header_lines[1] != parent_lines[0]
    ):
        raise DiagnosticProtocolError(
            f"{label} must encode exactly one tree and one first parent"
        )
    tree_bytes = tree_lines[0][len(b"tree ") :]
    parent_bytes = parent_lines[0][len(b"parent ") :]
    if (
        re.fullmatch(rb"[0-9a-f]{40}", tree_bytes) is None
        or re.fullmatch(rb"[0-9a-f]{40}", parent_bytes) is None
    ):
        raise DiagnosticProtocolError(f"{label} raw tree or parent identity is malformed")
    if parent_bytes != expected_parent.encode("ascii"):
        raise DiagnosticProtocolError(f"{label} raw parent differs")
    return tree_bytes.decode("ascii")


def _status_paths() -> frozenset[str]:
    payload = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not payload:
        return frozenset()
    fields = payload.split(b"\0")
    if fields[-1] != b"":
        raise DiagnosticProtocolError("git status output is not NUL terminated")
    paths: set[str] = set()
    index = 0
    while index < len(fields) - 1:
        record = fields[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise DiagnosticProtocolError("git status record is malformed")
        status_code = record[:2]
        try:
            path = record[3:].decode("utf-8")
        except UnicodeError as exc:
            raise DiagnosticProtocolError("git status path is not UTF-8") from exc
        paths.add(path)
        index += 1
        if b"R" in status_code or b"C" in status_code:
            raise DiagnosticProtocolError("git rename or copy status is forbidden")
    return frozenset(paths)


def _nul_paths(payload: bytes, label: str) -> frozenset[str]:
    if not payload:
        return frozenset()
    fields = payload.split(b"\0")
    if fields[-1] != b"" or any(field == b"" for field in fields[:-1]):
        raise DiagnosticProtocolError(f"{label} path output is malformed")
    try:
        decoded = [field.decode("utf-8") for field in fields[:-1]]
    except UnicodeError as exc:
        raise DiagnosticProtocolError(f"{label} path is not UTF-8") from exc
    if len(decoded) != len(set(decoded)):
        raise DiagnosticProtocolError(f"{label} path output contains duplicates")
    return frozenset(decoded)


def _read_live_preparation_path(relative_path: str) -> bytes:
    path = PROJECT_ROOT / relative_path
    try:
        before = path.lstat()
    except OSError as exc:
        raise DiagnosticProtocolError("cannot inspect a live preparation path") from exc
    if (
        path.is_symlink()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_uid != os.geteuid()
    ):
        raise DiagnosticProtocolError("live preparation path identity differs")
    try:
        payload = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise DiagnosticProtocolError("cannot read a live preparation path") from exc
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    if identity(before) != identity(after):
        raise DiagnosticProtocolError("live preparation path changed while read")
    return payload


def _preparation_invariant_projection() -> dict[str, Any]:
    return {
        "allowed_change_roster": sorted(ALLOWED_PREPARED_CHANGES),
        "expected_branch": EXPECTED_BRANCH,
        "parent_commit": PARENT_COMMIT,
        "parent_tree": PARENT_TREE,
        "sealed_lineage_proof": "RAW_GIT_CAT_FILE_COMMIT_BYTES_REHASHED_AS_SHA1_GIT_COMMIT_OBJECT_WITH_EXACTLY_ONE_ENCODED_PARENT_EQUAL_TO_PARENT_COMMIT_AND_ONE_FIRST_TREE_HEADER;NO_REVISION_WALK_OR_GRAFT_AUTHORITY",
        "sealed_changeset_proof": "REDUNDANT_EXPLICIT_PARENT_TREE_TO_RAW_CHILD_TREE_DIFF_TREE_AND_DIFF_EXACT_FOUR_PATH_ROSTER_WITH_RENAMES_DISABLED",
        "sealed_live_byte_proof": "EACH_LIVE_ALLOWED_PATH_IS_SINGLE_LINK_OWNER_REGULAR_STABLE_AND_EQUALS_THE_BLOB_RESOLVED_FROM_THE_RAW_BOUND_CHILD_TREE",
        "precommit_input_roster_before_protocol_publication": sorted(
            PREPARED_INPUT_CHANGES
        ),
        "validated_modes": [
            "PRECOMMIT_PARENT_HEAD_WITH_EXACT_THREE_INPUT_PATHS_AND_PROTOCOL_OUTPUT_ABSENT",
            "PRECOMMIT_PARENT_HEAD_WITH_EXACT_FOUR_PATH_WORKTREE_ROSTER_AND_PROTOCOL_OUTPUT_PRESENT",
            "SEALED_CLEAN_SINGLE_PARENT_DIRECT_CHILD_WITH_EXACT_FOUR_PATH_COMMIT_AND_LIVE_BLOB_EQUALITY",
        ],
    }


def _preparation_authority_record(
    authority_mode: str,
    observed_head: str,
    observed_tree: str,
    observed_worktree_changes: frozenset[str],
    observed_commit_changes: frozenset[str],
) -> dict[str, Any]:
    return {
        "authority_mode": authority_mode,
        "observed_branch": EXPECTED_BRANCH,
        "observed_commit_changes": sorted(observed_commit_changes),
        "observed_head": observed_head,
        "observed_tree": observed_tree,
        "observed_worktree_changes": sorted(observed_worktree_changes),
        "preparation_invariant_projection": _preparation_invariant_projection(),
    }


def _verify_preparation_authority() -> dict[str, Any]:
    branch = _git("branch", "--show-current").decode("ascii").strip()
    if branch != EXPECTED_BRANCH:
        raise DiagnosticProtocolError("preparation branch differs")
    head = _git("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise DiagnosticProtocolError("preparation HEAD identity is malformed")
    if (
        _git("rev-parse", f"{PARENT_COMMIT}^{{tree}}").strip()
        != PARENT_TREE.encode("ascii")
    ):
        raise DiagnosticProtocolError("sealed v1 executor tree differs")
    changed = _status_paths()
    if head == PARENT_COMMIT:
        try:
            PROTOCOL_PATH.lstat()
            protocol_output_present = True
        except FileNotFoundError:
            protocol_output_present = False
        except OSError as exc:
            raise DiagnosticProtocolError(
                "cannot inspect prepared protocol output path"
            ) from exc
        if changed == PREPARED_INPUT_CHANGES and not protocol_output_present:
            mode = "PRECOMMIT_PARENT_HEAD_WITH_EXACT_THREE_INPUT_PATHS_AND_PROTOCOL_OUTPUT_ABSENT"
        elif changed == ALLOWED_PREPARED_CHANGES and protocol_output_present:
            mode = "PRECOMMIT_PARENT_HEAD_WITH_EXACT_FOUR_PATH_WORKTREE_ROSTER_AND_PROTOCOL_OUTPUT_PRESENT"
        else:
            raise DiagnosticProtocolError(
                "precommit worktree changes or protocol output presence differ"
            )
        return _preparation_authority_record(
            mode,
            head,
            PARENT_TREE,
            changed,
            frozenset(),
        )
    if head == SUPERSEDED_COMMIT:
        raise DiagnosticProtocolError("superseded failed preparation commit is non-authoritative")
    if changed:
        raise DiagnosticProtocolError("sealed preparation worktree is not clean")
    observed_tree = _raw_single_parent_commit_tree(
        head, PARENT_COMMIT, "sealed preparation"
    )
    commit_paths = _nul_paths(
        _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "--no-renames",
            "-r",
            "-z",
            PARENT_TREE,
            observed_tree,
        ),
        "sealed preparation commit",
    )
    cumulative_paths = _nul_paths(
        _git(
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            PARENT_TREE,
            observed_tree,
            "--",
        ),
        "sealed preparation cumulative diff",
    )
    if (
        commit_paths != ALLOWED_PREPARED_CHANGES
        or cumulative_paths != ALLOWED_PREPARED_CHANGES
    ):
        raise DiagnosticProtocolError("sealed preparation changeset roster differs")
    for relative_path in sorted(ALLOWED_PREPARED_CHANGES):
        committed = _git("cat-file", "blob", f"{observed_tree}:{relative_path}")
        if _read_live_preparation_path(relative_path) != committed:
            raise DiagnosticProtocolError(
                "live preparation bytes differ from the sealed commit"
            )
    return _preparation_authority_record(
        "SEALED_CLEAN_SINGLE_PARENT_DIRECT_CHILD_WITH_EXACT_FOUR_PATH_COMMIT_AND_LIVE_BLOB_EQUALITY",
        head,
        observed_tree,
        changed,
        commit_paths,
    )


def _verify_superseded_candidate() -> dict[str, Any]:
    _verify_tag(
        SUPERSEDED_TAG,
        SUPERSEDED_TAG_OBJECT,
        SUPERSEDED_COMMIT,
        SUPERSEDED_TREE,
        SUPERSEDED_TAG_PAYLOAD_BYTES,
        SUPERSEDED_TAG_PAYLOAD_SHA256,
    )
    superseded_tree = _raw_single_parent_commit_tree(
        SUPERSEDED_COMMIT, PARENT_COMMIT, "superseded candidate"
    )
    if superseded_tree != SUPERSEDED_TREE:
        raise DiagnosticProtocolError("superseded candidate raw tree differs")
    bindings = (
        (
            "tools/jx_tno_common_epoch_horizons_response_diagnostic_v1.py",
            SUPERSEDED_SOURCE_BYTES,
            SUPERSEDED_SOURCE_SHA256,
        ),
        (
            "tests/test_tno_common_epoch_horizons_response_diagnostic_v1.py",
            SUPERSEDED_TEST_BYTES,
            SUPERSEDED_TEST_SHA256,
        ),
        (
            "runs/tno_common_epoch_horizons_response_diagnostic_v1/PROTOCOL.json",
            SUPERSEDED_PROTOCOL_BYTES,
            SUPERSEDED_PROTOCOL_SHA256,
        ),
        (
            "tools/test_matrix_manifest.json",
            SUPERSEDED_MANIFEST_BYTES,
            SUPERSEDED_MANIFEST_SHA256,
        ),
    )
    for relative_path, expected_bytes, expected_sha256 in bindings:
        payload = _git("cat-file", "blob", f"{SUPERSEDED_TREE}:{relative_path}")
        if len(payload) != expected_bytes or _sha256(payload) != expected_sha256:
            raise DiagnosticProtocolError("superseded candidate byte binding differs")
    return {
        "authoritative": False,
        "classification": "HISTORICAL_FAILED_PREPARATION_CANDIDATE_NEVER_EXECUTOR_AUTHORITY",
        "commit": SUPERSEDED_COMMIT,
        "failure": "PREPARATION_AUTHORITY_REQUIRED_HEAD_EQUAL_PARENT_AND_THEREFORE_REJECTED_ITS_OWN_SEALED_COMMIT",
        "manifest_bytes": SUPERSEDED_MANIFEST_BYTES,
        "manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
        "protocol_bytes": SUPERSEDED_PROTOCOL_BYTES,
        "protocol_content_sha256": SUPERSEDED_PROTOCOL_CONTENT_SHA256,
        "protocol_sha256": SUPERSEDED_PROTOCOL_SHA256,
        "source_bytes": SUPERSEDED_SOURCE_BYTES,
        "source_sha256": SUPERSEDED_SOURCE_SHA256,
        "tag": SUPERSEDED_TAG,
        "tag_object": SUPERSEDED_TAG_OBJECT,
        "tag_payload_bytes": SUPERSEDED_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": SUPERSEDED_TAG_PAYLOAD_SHA256,
        "test_bytes": SUPERSEDED_TEST_BYTES,
        "test_sha256": SUPERSEDED_TEST_SHA256,
        "tree": SUPERSEDED_TREE,
    }


def _verify_v1_protocol(payload: bytes) -> dict[str, Any]:
    protocol = _strict_json(payload, "v1 protocol")
    if protocol.get("schema") != "jx.tno.common-epoch-horizons-oracle.protocol.v1":
        raise DiagnosticProtocolError("v1 protocol schema differs")
    targets = protocol.get("inputs", {}).get("targets")
    if type(targets) is not list or not targets:
        raise DiagnosticProtocolError("v1 target roster is absent")
    first = targets[0]
    query = first.get("horizons_query") if type(first) is dict else None
    if (
        type(query) is not dict
        or query.get("url") != EXPECTED_QUERY_URL
        or query.get("ordered_parameter_pairs") != [list(pair) for pair in QUERY_PAIRS]
        or first.get("order") != TARGET_ORDER
        or first.get("object_id") != OBJECT_ID
        or first.get("spk_id") != SPK_ID
        or first.get("frozen_solution_id") != SOLUTION_ID
        or first.get("full_name") != FULL_NAME
    ):
        raise DiagnosticProtocolError("v1 first target or exact query differs")
    return protocol


def _verify_forensic_inputs(native_payload: bytes, sbdb_payload: bytes) -> dict[str, Any]:
    native = _strict_json(native_payload, "native Horizons response")
    result = native.get("result")
    if type(result) is not str or len(_NATIVE_EPOCH_LINE.findall(result)) != 1:
        raise DiagnosticProtocolError("retained native element epoch differs")
    sbdb = _strict_json(sbdb_payload, "SBDB response")
    orbit = sbdb.get("orbit")
    covariance = orbit.get("covariance") if type(orbit) is dict else None
    if (
        type(orbit) is not dict
        or orbit.get("epoch") != TARGET_JD_TDB
        or type(covariance) is not dict
        or covariance.get("epoch") != NATIVE_ELEMENT_EPOCH
    ):
        raise DiagnosticProtocolError("retained SBDB epoch lineage differs")
    return {
        "common_standard_orbit_epoch_jd_tdb": orbit["epoch"],
        "native_covariance_epoch_jd_tdb": covariance["epoch"],
        "same_solution_native_horizons_header_epoch_jd_tdb": NATIVE_ELEMENT_EPOCH,
    }


def audit_prerequisites() -> dict[str, Any]:
    if QUERY_URL != EXPECTED_QUERY_URL:
        raise DiagnosticProtocolError("constructed query URL differs from the frozen URL")
    authority = _verify_preparation_authority()
    _verify_tag(
        PARENT_TAG,
        PARENT_TAG_OBJECT,
        PARENT_COMMIT,
        PARENT_TREE,
        PARENT_TAG_PAYLOAD_BYTES,
        PARENT_TAG_PAYLOAD_SHA256,
    )
    _verify_tag(
        V1_PREP_TAG,
        V1_PREP_TAG_OBJECT,
        V1_PREP_COMMIT,
        V1_PREP_TREE,
        V1_PREP_TAG_PAYLOAD_BYTES,
        V1_PREP_TAG_PAYLOAD_SHA256,
    )
    superseded = _verify_superseded_candidate()
    payloads = {binding.role: _read_binding(binding) for binding in V1_BINDINGS}
    v1_protocol = _verify_v1_protocol(payloads["v1_prepare_protocol"])
    forensic = _verify_forensic_inputs(
        payloads["retained_same_solution_native_horizons_response"],
        payloads["retained_same_solution_sbdb_response"],
    )
    return {
        "status": "VALIDATED_SEALED_V1_FAILURE_CONTEXT_AND_REPAIRED_PREPARATION_AUTHORITY_WITH_NO_NETWORK",
        **authority,
        "bindings": [
            {
                "bytes": binding.bytes,
                "relative_path": binding.relative_path,
                "role": binding.role,
                "sha256": binding.sha256,
            }
            for binding in V1_BINDINGS
        ],
        "forensic_epoch_context": forensic,
        "superseded_failed_candidate": superseded,
        "v1_protocol_content_sha256": v1_protocol["content_sha256"],
    }


def _custody_contract() -> dict[str, Any]:
    claims = {
        "all_twelve_responses_retained": False,
        "bridge_execution_authorized": False,
        "bridge_input_authorized": False,
        "classification_generated": False,
        "common_epoch_nominal_oracle_bound": False,
        "dynamics_executed": False,
        "planet_nine_evidence": False,
        "population_inference": False,
        "response_accepted_as_oracle": False,
        "science_result": False,
        "v1_acquisition_succeeded": False,
    }
    nonclaims = {
        "covariance_posterior_or_probability": False,
        "executor_identity_cryptographically_proven_by_custody": False,
        "horizons_full_force_model_equivalence": False,
        "long_horizon_accuracy_or_execution_authority": False,
        "numerical_bridge_acceptance_or_qualification": False,
        "physical_stability_collision_or_ejection": False,
        "planet_nine_detection_exclusion_or_ranking": False,
        "provider_origin_cryptographically_proven": False,
        "provider_transport_independently_attested": False,
    }
    return {
        "schema": CUSTODY_SCHEMA,
        "status_exact": CUSTODY_STATUS,
        "content_hash": {
            "algorithm": "SHA-256",
            "construction": "SHA256(DOMAIN_SEPARATOR || CANONICAL_JSON_WITH_CONTENT_SHA256_REMOVED)",
            "domain_separator_hex": CUSTODY_DOMAIN_HEX,
        },
        "strict_canonical_json": "UTF8_ASCII_SUBSET_SORT_KEYS_COMPACT_SEPARATORS_ENSURE_ASCII_ALLOW_NAN_FALSE_ONE_TRAILING_LF",
        "maximum_canonical_custody_bytes_inclusive": MAX_CUSTODY_BYTES,
        "exact_top_level_keys_sorted": [
            "attempt",
            "claims",
            "content_sha256",
            "diagnosis",
            "nonclaims",
            "preregistration_identity",
            "protocol_content_sha256",
            "protocol_file_sha256",
            "request",
            "response",
            "schema",
            "status",
        ],
        "exact_top_level_types": {
            "attempt": "EXACT_OBJECT",
            "claims": "EXACT_OBJECT",
            "content_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
            "diagnosis": "EXACT_OBJECT",
            "nonclaims": "EXACT_OBJECT",
            "preregistration_identity": "EXACT_OBJECT",
            "protocol_content_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
            "protocol_file_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
            "request": "EXACT_OBJECT",
            "response": "EXACT_OBJECT",
            "schema": "EXACT_STRING",
            "status": "EXACT_STRING",
        },
        "top_level_fixed_values": {
            "schema": CUSTODY_SCHEMA,
            "status": CUSTODY_STATUS,
        },
        "claims_exact": claims,
        "nonclaims_exact": nonclaims,
        "preregistration_identity_exact_keys_sorted": list(PREREGISTRATION_IDENTITY_KEYS),
        "preregistration_identity_exact_types": dict(
            PREREGISTRATION_IDENTITY_TYPES
        ),
        "preregistration_identity_fixed_values": {"tag": FUTURE_TAG},
        "preregistration_identity_semantics": "ALL_VALUES_ARE_SUCCESSOR_COMPILE_TIME_LITERALS_BOUND_TO_THE_PUSHED_AND_EXPLICITLY_REFETCHED_ANNOTATED_PREPARE_TAG_BEFORE_REQUEST_ZERO",
        "identity_scope": {
            "embedded_identity": "PREPARE_PREREGISTRATION_ONLY",
            "executor_identity_embedded": False,
            "executor_identity_repository_binding": "SEPARATE_ANNOTATED_EXECUTOR_TAG_PUSHED_EXPLICITLY_REFETCHED_AND_VERIFIED_BEFORE_REQUEST_ZERO_THEN_BOUND_BY_THE_RESULT_COMMIT_AND_ANNOTATED_RESULT_TAG",
            "limitation": "CUSTODY_ALONE_DOES_NOT_CRYPTOGRAPHICALLY_ATTEST_EXECUTOR_IDENTITY_PROVIDER_ORIGIN_OR_PROVIDER_TRANSPORT",
        },
        "attempt_binding_exact_keys_sorted": [
            "content_sha256",
            "file_bytes",
            "file_sha256",
            "relative_path",
            "status",
        ],
        "attempt_binding_fixed_values": {
            "relative_path": f"{ATTEMPT_ROOT_RELATIVE}/ATTEMPT.json",
            "status": "ONE_REQUEST_BUDGET_IRREVERSIBLY_RESERVED",
        },
        "attempt_binding_dynamic_types": {
            "content_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
            "file_bytes": "POSITIVE_INTEGER_NOT_BOOLEAN_AT_OR_BELOW_1048576",
            "file_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
        },
        "request_exact_keys_sorted": [
            "completed_request_count",
            "headers",
            "method",
            "object_id",
            "retry_count",
            "solution_id",
            "spk_id",
            "target_order",
            "url",
        ],
        "request_fixed_values": {
            "completed_request_count": 1,
            "headers": [list(item) for item in REQUEST_HEADERS],
            "method": "GET",
            "object_id": OBJECT_ID,
            "retry_count": 0,
            "solution_id": SOLUTION_ID,
            "spk_id": SPK_ID,
            "target_order": TARGET_ORDER,
            "url": QUERY_URL,
        },
        "response_exact_keys_sorted": [
            "entity_byte_semantics",
            "http_status",
            "raw_bytes",
            "raw_relative_path",
            "raw_sha256",
        ],
        "response_fixed_values": {
            "entity_byte_semantics": "HTTP_ENTITY_OCTETS_AFTER_TRANSFER_CODING_REMOVAL_WITH_CONTENT_ENCODING_ABSENT_OR_IDENTITY",
            "http_status": 200,
            "raw_relative_path": RAW_RELATIVE,
        },
        "response_dynamic_types": {
            "raw_bytes": "INTEGER_NOT_BOOLEAN_0_THROUGH_1048576",
            "raw_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
        },
        "diagnosis_exact_keys_sorted": [
            "element_epoch_line_candidates",
            "json_envelope_status",
            "result_string_sha256",
            "state_vector_epoch_token_candidates",
            "v1_gate_error_message",
            "v1_gate_outcome",
        ],
        "diagnosis_exact_types": {
            "element_epoch_line_candidates": "ARRAY_OF_STRINGS",
            "json_envelope_status": "EXACT_ENUM_STRING",
            "result_string_sha256": "LOWERCASE_HEX_STRING_LENGTH_64_OR_NULL",
            "state_vector_epoch_token_candidates": "ARRAY_OF_STRINGS",
            "v1_gate_error_message": "STRING_OR_NULL",
            "v1_gate_outcome": "EXACT_ENUM_STRING",
        },
        "diagnosis_contract": {
            "authority": "PURE_BYTE_DERIVED_DIAGNOSTIC_ONLY_NEVER_AN_ACCEPTANCE_GATE",
            "element_epoch_line_candidates": "ALL_EXACT_RESULT_LINES_WHOSE_LEFT_STRIPPED_FORM_STARTS_WITH_EPOCH_EQUALS_IN_SOURCE_ORDER_OR_EMPTY",
            "json_envelope_status_values": [
                "STRICT_OBJECT",
                "STRICT_NON_OBJECT",
                "STRICT_JSON_REJECTED",
            ],
            "result_string_sha256": "SHA256_OF_ASCII_CANONICAL_JSON_STRING_TOKEN_USING_ENSURE_ASCII_TRUE_ALLOW_NAN_FALSE_COMPACT_SEPARATORS_OR_NULL;THIS_IS_TOTAL_FOR_LONE_SURROGATE_ESCAPES",
            "state_vector_epoch_token_candidates": "FIRST_COMMA_TOKEN_STRIPPED_FROM_EACH_NONEMPTY_LINE_BETWEEN_EXACTLY_ONE_SOE_THEN_EXACTLY_ONE_LATER_EOE_OR_EMPTY",
            "v1_gate_error_message": "EXACT_CONTROLLED_HORIZONS_ORACLE_PROTOCOL_ERROR_STRING_OR_NULL",
            "v1_gate_outcome_values": [
                "PASS_DIAGNOSTIC_ONLY",
                "REJECTED_BY_FROZEN_V1_GATE",
                "FROZEN_V1_VALIDATOR_INTERNAL_EXCEPTION",
            ],
            "totality": "EVERY_COMPLETE_BODY_AT_OR_BELOW_THE_CAP_PRODUCES_A_DIAGNOSIS_OBJECT;UTF8_JSON_DUPLICATE_NONFINITE_DEPTH_INTEGER_LIMIT_AND_ALL_OTHER_DECODE_FAILURES_MAP_TO_STRICT_JSON_REJECTED;SEMANTIC_FAILURE_NEVER_DELETES_OR_REJECTS_RAW_CUSTODY",
            "pure_observation_function": "diagnostic_observations",
        },
    }


def build_protocol() -> dict[str, Any]:
    prerequisites = audit_prerequisites()
    invariant_projection = prerequisites["preparation_invariant_projection"]
    if invariant_projection != _preparation_invariant_projection():
        raise DiagnosticProtocolError("preparation invariant projection differs")
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "status": PROTOCOL_STATUS,
        "authority": {
            "repair_revision": 1,
            "network_access": False,
            "provider_contacted": False,
            "execution_in_this_artifact": False,
            "acquisition_in_this_artifact": False,
            "expected_preparation_branch": EXPECTED_BRANCH,
            "public_cli": ["audit-prerequisites", "build", "prepare", "validate-prepared"],
            "explicitly_absent_public_cli": [
                "acquire",
                "acquire-diagnostic",
                "execute",
                "resume",
                "validate-result",
            ],
            "preparation_authority": invariant_projection,
            "precommit_and_sealed_build_bytes_must_be_identical": True,
        },
        "lineage": {
            "sealed_v1_executor": {
                "branch": PARENT_BRANCH,
                "commit": PARENT_COMMIT,
                "tree": PARENT_TREE,
                "tag": PARENT_TAG,
                "tag_object": PARENT_TAG_OBJECT,
                "tag_payload_bytes": PARENT_TAG_PAYLOAD_BYTES,
                "tag_payload_sha256": PARENT_TAG_PAYLOAD_SHA256,
            },
            "sealed_v1_prepare": {
                "commit": V1_PREP_COMMIT,
                "tree": V1_PREP_TREE,
                "tag": V1_PREP_TAG,
                "tag_object": V1_PREP_TAG_OBJECT,
                "tag_payload_bytes": V1_PREP_TAG_PAYLOAD_BYTES,
                "tag_payload_sha256": V1_PREP_TAG_PAYLOAD_SHA256,
                "protocol_content_sha256": prerequisites["v1_protocol_content_sha256"],
            },
            "superseded_failed_candidate": prerequisites[
                "superseded_failed_candidate"
            ],
            "input_bindings": prerequisites["bindings"],
        },
        "forensic_context": {
            "classification": "LOCAL_SEALED_EVIDENCE_AND_CONTROL_FLOW_INFERENCE_NOT_THE_MISSING_LIVE_BODY",
            "reported_v1_terminal_error": "Horizons element epoch or scale differs",
            "exact_live_response_retained_by_v1": False,
            "exact_live_element_epoch_known": False,
            "control_flow_inference": "THE_MATCHED_ELEMENT_SCALE_WAS_TDB_AND_THE_PARSED_DECIMAL_ELEMENT_EPOCH_DIFFERED_FROM_2461200_5",
            "reason_scale_is_excluded": "THE_FROZEN_V1_REGEX_CAN_REACH_THIS_ERROR_ONLY_AFTER_MATCHING_LITERAL_TDB",
            "sealed_same_solution_context": prerequisites["forensic_epoch_context"],
            "hypothesis_only": "HORIZONS_INITIAL_ELEMENT_HEADER_USES_THE_NATIVE_COVARIANCE_EPOCH_WHILE_TLIST_CONTROLS_THE_VECTOR_ROW_EPOCH",
            "hypothesis_is_not_accepted_without_the_NEW_RETAINED_BODY": True,
            "repair_scope": "AUTHORITY_LIFECYCLE_ONLY;SCIENTIFIC_QUERY_RETENTION_DIAGNOSIS_AND_CLAIM_CEILING_UNCHANGED_FROM_THE_SUPERSEDED_CANDIDATE",
        },
        "frozen_request": {
            "purpose": "ONE_RESPONSE_FAILURE_DIAGNOSTIC_ONLY",
            "target_order": TARGET_ORDER,
            "object_id": OBJECT_ID,
            "spk_id": SPK_ID,
            "solution_id": SOLUTION_ID,
            "full_name": FULL_NAME,
            "method": "HTTPS_GET",
            "endpoint": HORIZONS_ENDPOINT,
            "ordered_parameter_pairs": [list(item) for item in QUERY_PAIRS],
            "url": QUERY_URL,
            "headers": [list(item) for item in REQUEST_HEADERS],
            "same_exact_url_and_application_headers_as_failed_v1_request_zero": True,
            "request_count": 1,
            "requests_per_url": 1,
            "retries_for_any_reason": 0,
            "parallel_requests": 0,
        },
        "future_transport": {
            "implemented_here": False,
            "direct_connection_only": True,
            "proxy_cookies_authorization_forbidden": True,
            "redirects_forbidden": True,
            "http_status_required": 200,
            "tls": "DEFAULT_PLATFORM_CA_AND_HOSTNAME_VERIFICATION_REQUIRED_NO_BYPASS",
            "timeouts_seconds": {"connect": 30, "read_idle": 30, "total": 120},
            "content_encoding": "ABSENT_OR_EXACTLY_IDENTITY",
            "transfer_encoding": "ABSENT_OR_EXACTLY_ONE_CHUNKED_VALUE_DECODED_BY_HTTP_CLIENT",
            "content_length": "IF_PRESENT_EXACTLY_ONE_BASE10_NONNEGATIVE_VALUE_AT_OR_BELOW_CAP_AND_EQUAL_TO_FINAL_ENTITY_BYTES",
            "streaming": {
                "maximum_chunk_bytes": STREAM_CHUNK_BYTES,
                "maximum_complete_entity_bytes_inclusive": MAX_RESPONSE_BYTES,
                "maximum_buffered_bytes": MAX_RESPONSE_BYTES + 1,
                "oversize_action": "STOP_AT_BYTE_1048577_TRANSPORT_FAILURE_NO_COMPLETE_BODY_RESULT",
            },
            "transport_failure": "LEAVE_DURABLE_ATTEMPT_BUDGET_MARKER_PUBLISH_NO_RESPONSE_RESULT_AND_FORBID_ANY_SECOND_REQUEST",
        },
        "durable_one_request_budget": {
            "attempt_root_relative_path": ATTEMPT_ROOT_RELATIVE,
            "attempt_file_relative_path": f"{ATTEMPT_ROOT_RELATIVE}/ATTEMPT.json",
            "schema": ATTEMPT_SCHEMA,
            "status": "ONE_REQUEST_BUDGET_IRREVERSIBLY_RESERVED",
            "content_hash_domain_hex": ATTEMPT_DOMAIN_HEX,
            "created_and_parent_fsynced_before_request_bytes_can_be_sent": True,
            "atomic_create_only_noreplace": True,
            "existing_marker_action": "REFUSE_BEFORE_DNS_OR_SOCKET_CREATION",
            "marker_never_removed_or_reused": True,
            "crash_after_reservation": "REQUEST_BUDGET_REMAINS_CONSUMED_EVEN_IF_NO_REQUEST_WAS_SENT",
            "concurrency": "ONLY_THE_PROCESS_THAT_ATOMICALLY_PUBLISHES_THE_MARKER_MAY_CONTINUE",
            "maximum_request_invocations_across_PROCESS_RESTARTS": 1,
            "attempt_custody_contract": {
                "content_hash": {
                    "algorithm": "SHA-256",
                    "construction": "SHA256(DOMAIN_SEPARATOR || CANONICAL_JSON_WITH_CONTENT_SHA256_REMOVED)",
                    "domain_separator_hex": ATTEMPT_DOMAIN_HEX,
                },
                "exact_top_level_keys_sorted": [
                    "content_sha256",
                    "preregistration_identity",
                    "protocol_content_sha256",
                    "protocol_file_sha256",
                    "request_budget",
                    "schema",
                    "status",
                ],
                "exact_top_level_types": {
                    "content_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
                    "preregistration_identity": "EXACT_OBJECT",
                    "protocol_content_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
                    "protocol_file_sha256": "LOWERCASE_HEX_STRING_LENGTH_64",
                    "request_budget": "EXACT_OBJECT",
                    "schema": "EXACT_STRING",
                    "status": "EXACT_STRING",
                },
                "top_level_fixed_values": {
                    "schema": ATTEMPT_SCHEMA,
                    "status": "ONE_REQUEST_BUDGET_IRREVERSIBLY_RESERVED",
                },
                "preregistration_identity_exact_keys_sorted": list(
                    PREREGISTRATION_IDENTITY_KEYS
                ),
                "preregistration_identity_exact_types": dict(
                    PREREGISTRATION_IDENTITY_TYPES
                ),
                "preregistration_identity_fixed_values": {"tag": FUTURE_TAG},
                "request_budget_exact_keys_sorted": [
                    "headers",
                    "maximum_request_invocations",
                    "method",
                    "retries",
                    "url",
                ],
                "request_budget_exact_values": {
                    "headers": [list(item) for item in REQUEST_HEADERS],
                    "maximum_request_invocations": 1,
                    "method": "GET",
                    "retries": 0,
                    "url": QUERY_URL,
                },
                "strict_canonical_domain_hashed_json": True,
            },
            "attempt_publication_contract": {
                "exact_root_roster_sorted": ["ATTEMPT.json"],
                "directory_mode_octal": "0700",
                "file_mode_octal": "0600",
                "file_must_be_owned_regular_single_link": True,
                "symlinks_special_files_and_extra_entries": "FORBIDDEN",
                "maximum_attempt_file_bytes_inclusive": 1_048_576,
                "algorithm": "BUILD_AND_VALIDATE_PRIVATE_SIBLING_STAGING_ROOT;WRITE_AND_FSYNC_CANONICAL_ATTEMPT_JSON;FSYNC_STAGING_ROOT;ATOMIC_RENAMEAT2_RENAME_NOREPLACE_WHOLE_ROOT_TO_ATTEMPT_ROOT;FSYNC_PARENT",
                "prepublication_failure": "REMOVE_ONLY_PRIVATE_ATTEMPT_STAGING_AND_SEND_ZERO_REQUESTS",
                "postpublication_failure": "NEVER_REMOVE_OR_REPLACE_ATTEMPT_ROOT_AND_SEND_NO_SECOND_REQUEST",
            },
            "offline_validation": {
                "cli": "validate-attempt",
                "network_access": False,
                "read_only": True,
                "exact_canonical_replay_and_domain_hash": True,
                "verify_root_file_roster_mode_owner_nlink_and_held_fd_identity": True,
            },
        },
        "response_retention_and_diagnosis_order": [
            "VERIFY_ALL_PREREGISTRATION_EXECUTOR_GIT_INPUT_OUTPUT_AND_ABSENCE_GATES_BEFORE_REQUEST_ZERO",
            "CREATE_AND_HOLD_PRIVATE_RESULT_STAGING_ROOT_AND_RAW_DIRECTORY_BEFORE_REQUEST_ZERO",
            "ATOMICALLY_PUBLISH_AND_FSYNC_THE_COMPLETE_IRREVERSIBLE_ONE_REQUEST_ATTEMPT_ROOT",
            "ISSUE_EXACTLY_ONE_DIRECT_HTTPS_GET_WITH_NO_RETRY",
            "REQUIRE_HTTP_200_IDENTITY_ENTITY_AND_STREAM_TO_CLEAN_EOF_AT_OR_BELOW_1048576_BYTES",
            "WRITE_EXACT_ENTITY_BYTES_CREATE_ONLY_TO_RAW_20090377_JSON_AND_FSYNC_BEFORE_ANY_SEMANTIC_DIAGNOSIS",
            "RUN_TOTAL_NONAUTHORITATIVE_DIAGNOSIS_THAT_CONVERTS_ALL_SEMANTIC_FAILURES_TO_DATA",
            "BUILD_DOMAIN_HASHED_CANONICAL_CUSTODY_WITH_ALL_ORACLE_BRIDGE_AND_SCIENCE_CLAIMS_FALSE",
            "WRITE_FSYNC_AND_RENAME_CUSTODY_AS_THE_LAST_STAGED_COMMIT_MARKER",
            "OFFLINE_REPLAY_FROM_HELD_FILE_DESCRIPTORS",
            "ATOMIC_NOREPLACE_RENAME_COMPLETE_STAGING_ROOT_TO_FINAL_RESULT_AND_FSYNC_PARENT",
        ],
        "semantic_policy": {
            "semantic_gate_controls_retention": False,
            "v1_validator_rejection_is_a_successful_diagnostic_outcome": True,
            "provider_json_or_text_schema_change_is_retained_not_selected_AWAY": True,
            "post_response_query_gate_or_parser_tuning_before_publication": False,
            "surprising_v1_gate_pass_still_authorizes_oracle": False,
            "raw_bytes_are_the_only_authority_for_later_protocol_design": True,
        },
        "future_custody_contract": _custody_contract(),
        "future_result_publication": {
            "result_root_relative_path": RESULT_ROOT_RELATIVE,
            "result_root_must_not_exist": True,
            "exact_root_roster_sorted": ["CUSTODY.json", "raw"],
            "exact_raw_roster_sorted": ["20090377.json"],
            "directory_mode_octal": "0700",
            "file_mode_octal": "0600",
            "single_link_owned_regular_files_only": True,
            "held_file_and_directory_descriptors_rechecked_before_and_after_replay_and_final_rename": True,
            "symlinks_special_files_and_extra_entries": "FORBIDDEN",
            "private_staging_same_parent_filesystem": True,
            "private_staging_creation": "UNPREDICTABLE_EXCLUSIVE_OWNER_PRIVATE_SIBLING_CREATED_AND_HELD_BEFORE_ATTEMPT_RESERVATION_AND_REQUEST_ZERO",
            "custody_last_staged_file_and_commit_marker": True,
            "atomic_final_rename": "RENAMEAT2_RENAME_NOREPLACE",
            "publication_commit_point": "SUCCESSFUL_RENAMEAT2_RENAME_NOREPLACE_OF_THE_COMPLETE_PRIVATE_STAGING_ROOT_TO_THE_FINAL_RESULT_ROOT",
            "file_directory_and_parent_fsync": True,
            "pre_request_existing_attempt_or_result": "REFUSE_WITHOUT_NETWORK_OR_MUTATION",
            "pre_final_rename_existing_result": "REFUSE_PUBLICATION_WITHOUT_REPLACEMENT_DELETION_OR_SECOND_REQUEST",
            "required_attempt_at_final_rename": "EXACT_ALREADY_PUBLISHED_ATTEMPT_ROOT_MUST_REPLAY_AND_MATCH_CUSTODY_BINDING",
            "attempt_binding_in_custody": "EXACT_ATTEMPT_FILE_BYTE_LENGTH_SHA256_CONTENT_SHA256_RELATIVE_PATH_AND_STATUS",
            "semantic_failure_after_complete_body": "PUBLISH_RAW_AND_DIAGNOSTIC_CUSTODY",
            "transport_failure_before_complete_body": "PUBLISH_NO_RESULT_KEEP_ATTEMPT_MARKER",
            "handled_failure_before_final_rename": "IDENTITY_CONSTRAINED_REMOVE_ONLY_THE_PRIVATE_STAGING_TREE;FSYNC_PARENT;KEEP_ATTEMPT_MARKER;SEND_NO_SECOND_REQUEST",
            "crash_before_final_rename": "NO_FINAL_RESULT_EXISTS;PRIVATE_STAGING_MAY_REMAIN;KEEP_ATTEMPT_MARKER;SEND_NO_SECOND_REQUEST;VALIDATE_ATTEMPT_OFFLINE",
            "failure_after_final_rename_before_parent_fsync": "STRUCTURALLY_COMPLETE_FINAL_RESULT_WITH_DURABILITY_INDETERMINATE;NEVER_DELETE_REPLACE_OR_RETRY;RUN_READ_ONLY_VALIDATE_RESULT",
            "failure_after_parent_fsync_or_postpublication_recheck": "FINAL_RESULT_REMAINS_AUTHORITATIVE_ONLY_IF_READ_ONLY_VALIDATE_RESULT_PASSES;NEVER_DELETE_REPLACE_REPAIR_OR_RETRY",
            "process_control_baseexceptions": "CRASH_SEMANTICS_BYPASS_HANDLED_FAILURE_CLEANUP",
            "cleanup_identity_constraints": "REMOVE_ONLY_HELD_PRIVATE_STAGING_INODES_AFTER_EXACT_PARENT_ROOT_RAW_ROSTER_MODE_OWNER_NLINK_AND_NAME_REANCHOR_CHECKS;NEVER_PATH_RECURSIVELY_REMOVE_AN_UNVERIFIED_TREE",
            "partial_final_result": "FORBIDDEN_NEVER_COUNTS_AS_RESULT",
        },
        "future_offline_replay": {
            "network_access": False,
            "read_only": True,
            "required_cli": ["validate-attempt", "validate-result"],
            "rebuild_exact_custody_from_retained_raw": True,
            "recompute_diagnosis_from_retained_raw": True,
            "verify_attempt_marker_binding": True,
            "enforce_raw_and_custody_byte_caps_before_decoding": True,
            "verify_domain_hashes_canonical_bytes_rosters_modes_ownership_nlink_and_held_fd_identity": True,
            "mismatch_action": "FAIL_CLOSED_NO_REPAIR_REWRITE_OR_PARTIAL_ACCEPTANCE",
            "transport_failure_validation": "VALIDATE_ATTEMPT_OFFLINE_WITH_RESULT_ROOT_ABSENT",
        },
        "future_preregistration_and_execution": {
            "prepare_tag": FUTURE_TAG,
            "executor_tag": FUTURE_EXECUTOR_TAG,
            "result_tag": FUTURE_RESULT_TAG,
            "annotated_tags_required": True,
            "push_and_explicit_refetch_verification_required_before_successor_and_before_request": True,
            "successor_must_be_one_clean_direct_child_of_prepare_commit": True,
            "successor_allowed_diff": [
                "tests/test_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py",
                "tools/jx_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py",
                "tools/test_matrix_manifest.json",
            ],
            "successor_must_bind_prepare_tag_object_commit_tree_payload_and_exact_source_test_protocol_manifest_bytes": True,
            "successor_inputs_are_opened_without_symlink_following_and_identity_rechecked_before_request_and_publication": True,
            "custody_embeds_prepare_identity_not_executor_identity": True,
            "executor_identity_binding_is_repository_external_to_custody": "ANNOTATED_EXECUTOR_TAG_VERIFIED_BEFORE_REQUEST_AND_RESULT_COMMIT_PLUS_ANNOTATED_RESULT_TAG;CUSTODY_DOES_NOT_CLAIM_EXECUTOR_CRYPTOGRAPHIC_ATTESTATION",
            "request_count_before_all_bindings_and_attempt_reservation_pass": 0,
            "v1_result_path_must_remain_absent_and_unmodified": True,
        },
        "administrative_claims": {
            "authority_lifecycle_repair_revision_1": True,
            "prepare_only_contract_frozen": True,
            "one_exact_query_frozen": True,
            "future_raw_first_retention_order_frozen": True,
            "future_durable_one_request_budget_frozen": True,
            "future_diagnostic_custody_schema_frozen": True,
        },
        "claims": {
            "all_twelve_responses_acquired": False,
            "bridge_executed": False,
            "bridge_input_authorized": False,
            "bridge_qualified": False,
            "classification_generated": False,
            "common_epoch_nominal_oracle_bound": False,
            "diagnostic_response_acquired": False,
            "dynamics_executed": False,
            "planet_nine_evidence": False,
            "population_inference": False,
            "provider_contacted": False,
            "science_result": False,
            "v1_acquisition_retried_or_reclassified": False,
        },
        "claim_ceiling": "PREPARED_FUTURE_ONE_RESPONSE_DIAGNOSTIC_CONTRACT_ONLY_NO_PROVIDER_RESPONSE_NO_ORACLE_NO_BRIDGE_NO_SCIENCE",
        "publication": {
            "prepared_roster": ["PROTOCOL.json"],
            "root_must_not_exist": True,
            "create_only_no_overwrite": True,
            "directory_mode_octal": "0700",
            "file_mode_octal": "0600",
            "file_and_directory_fsync_required": True,
            "validation_is_read_only": True,
        },
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(descriptor, payload[offset:])
        except OSError as exc:
            raise DiagnosticProtocolError("cannot write prepared protocol") from exc
        if written <= 0:
            raise DiagnosticProtocolError("short write while preparing protocol")
        offset += written


def _validate_prepared_path(root: Path) -> dict[str, Any]:
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise DiagnosticProtocolError("prepared root is absent") from exc
    if root.is_symlink() or not stat.S_ISDIR(root_stat.st_mode):
        raise DiagnosticProtocolError("prepared root is not a real directory")
    if stat.S_IMODE(root_stat.st_mode) != 0o700 or root_stat.st_uid != os.geteuid():
        raise DiagnosticProtocolError("prepared root ownership or mode differs")
    try:
        roster = sorted(item.name for item in root.iterdir())
    except OSError as exc:
        raise DiagnosticProtocolError("cannot list prepared root") from exc
    if roster != ["PROTOCOL.json"]:
        raise DiagnosticProtocolError("prepared root roster differs")
    path = root / "PROTOCOL.json"
    metadata = path.lstat()
    if (
        path.is_symlink()
        or not stat.S_ISREG(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise DiagnosticProtocolError("prepared protocol file identity differs")
    payload = path.read_bytes()
    if len(payload) > MAX_RESPONSE_BYTES:
        raise DiagnosticProtocolError("prepared protocol unexpectedly exceeds 1 MiB")
    value = _strict_json(payload, "prepared protocol")
    if payload != _canonical_json(value):
        raise DiagnosticProtocolError("prepared protocol is not exact canonical JSON")
    expected = build_protocol()
    if value != expected or payload != _canonical_json(expected):
        raise DiagnosticProtocolError("prepared protocol differs from deterministic build")
    if value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value):
        raise DiagnosticProtocolError("prepared protocol content hash differs")
    return value


def prepare(root: Path = PROTOCOL_ROOT) -> dict[str, Any]:
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    parent = root.parent
    if root.exists() or root.is_symlink():
        raise DiagnosticProtocolError("prepared root already exists")
    try:
        parent_metadata = parent.lstat()
    except OSError as exc:
        raise DiagnosticProtocolError("prepared parent is absent") from exc
    if parent.is_symlink() or not stat.S_ISDIR(parent_metadata.st_mode):
        raise DiagnosticProtocolError("prepared parent is not a real directory")
    created = False
    descriptor = -1
    try:
        os.mkdir(root, 0o700)
        created = True
        os.chmod(root, 0o700, follow_symlinks=False)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(root / "PROTOCOL.json", flags, 0o600)
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return _validate_prepared_path(root)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if created:
            try:
                protocol_path = root / "PROTOCOL.json"
                if protocol_path.exists() and not protocol_path.is_symlink():
                    protocol_path.unlink()
                root.rmdir()
            except OSError:
                pass
        raise


def validate_prepared(root: Path = PROTOCOL_ROOT) -> dict[str, Any]:
    return _validate_prepared_path(root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-prerequisites")
    commands.add_parser("build")
    commands.add_parser("prepare")
    validate = commands.add_parser("validate-prepared")
    validate.add_argument("--root", type=Path, default=PROTOCOL_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "audit-prerequisites":
            value = audit_prerequisites()
        elif arguments.command == "build":
            value = build_protocol()
        elif arguments.command == "prepare":
            value = prepare()
        else:
            value = validate_prepared(arguments.root)
    except DiagnosticProtocolError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    sys.stdout.buffer.write(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
