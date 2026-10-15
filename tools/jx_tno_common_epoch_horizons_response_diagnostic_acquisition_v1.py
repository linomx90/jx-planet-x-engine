#!/usr/bin/env python3
"""Acquire one preregistered Horizons response for diagnosis only.

The durable ATTEMPT marker consumes the one-request budget before DNS or a
socket can be used.  A complete HTTP-200 entity is retained and fsynced before
the frozen v1 semantic gate is evaluated.  Semantic rejection is data: it does
not suppress raw custody.  This executor never creates an oracle, bridge input,
classification, dynamics run, or science result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types
from typing import Any, Callable, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREPARE_TAG = "tno-common-epoch-horizons-response-diagnostic-v1r1-preregistered-20261014"
PREPARE_TAG_OBJECT = "0ab84a65fee25b90b66be8d9946813818d868293"
PREPARE_COMMIT = "1fcea5691d9dc2e15f7c95185424eecafb16d19d"
PREPARE_TREE = "f1165051dc87baa65ac3dac7db223db33d1b6ae8"
PREPARE_PARENT = "35bd04a2e12c6f4a4a4c3527974c6954544fd84a"
PREPARE_TAG_PAYLOAD_BYTES = 276
PREPARE_TAG_PAYLOAD_SHA256 = (
    "bb64008d2fefc21f4a262f45b0798fff93db31f5cb569bea1539b4a670e703c0"
)

PREPARE_SOURCE_RELATIVE = "tools/jx_tno_common_epoch_horizons_response_diagnostic_v1.py"
PREPARE_TEST_RELATIVE = "tests/test_tno_common_epoch_horizons_response_diagnostic_v1.py"
PREPARE_PROTOCOL_RELATIVE = (
    "runs/tno_common_epoch_horizons_response_diagnostic_v1/PROTOCOL.json"
)
MANIFEST_RELATIVE = "tools/test_matrix_manifest.json"
PREPARE_SOURCE_BYTES = 64_411
PREPARE_SOURCE_SHA256 = (
    "12724a2d15defc8f2ce940590d82c2562ae6a4b133d90dc328f319048f60bf37"
)
PREPARE_TEST_BYTES = 41_555
PREPARE_TEST_SHA256 = (
    "74acab11b74179343b1a9644c0961db586986d9365457dabef09136ea5a36517"
)
PREPARE_PROTOCOL_BYTES = 27_806
PREPARE_PROTOCOL_SHA256 = (
    "62dbe4cc3a2187c869447dc3b107db4fcdb95c9c69d9e5f819e40ad2c7ba9307"
)
PREPARE_PROTOCOL_CONTENT_SHA256 = (
    "7528b32f3b45f107f050697abc395b601d27b0383e8fd780827b2b9d97343621"
)
PREPARE_MANIFEST_BYTES = 13_214
PREPARE_MANIFEST_SHA256 = (
    "f5c70d571ac91ca32d3c79cea1d434f1de52a2538d71e050ab193571c2f49742"
)

BASE_EXECUTOR_RELATIVE = "tools/jx_tno_common_epoch_horizons_oracle_acquisition_v1.py"
BASE_EXECUTOR_BYTES = 95_391
BASE_EXECUTOR_SHA256 = (
    "bb408c69031f1e3c3d9b94a116f5a55ff48423a501d32a4203d049240f6a7b8f"
)

SUCCESSOR_BRANCH = (
    "research/tno-common-epoch-horizons-response-diagnostic-executor-v1-20261014"
)
EXECUTOR_TAG = (
    "tno-common-epoch-horizons-response-diagnostic-executor-v1r1-preregistered-20261014"
)
SUCCESSOR_SOURCE_RELATIVE = (
    "tools/jx_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py"
)
SUCCESSOR_TEST_RELATIVE = (
    "tests/test_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py"
)
SUCCESSOR_PATHS = frozenset(
    {SUCCESSOR_SOURCE_RELATIVE, SUCCESSOR_TEST_RELATIVE, MANIFEST_RELATIVE}
)
MAX_SOURCE_BYTES = 4_194_304
MAX_TEST_BYTES = 4_194_304
MAX_MANIFEST_BYTES = 1_048_576

ATTEMPT_RELATIVE = "runs/tno_common_epoch_horizons_response_diagnostic_v1_attempt"
RESULT_RELATIVE = "runs/tno_common_epoch_horizons_response_diagnostic_v1_result"
V1_RESULT_RELATIVE = "runs/tno_common_epoch_horizons_oracle_v1_result"
DEFAULT_ATTEMPT_ROOT = PROJECT_ROOT / ATTEMPT_RELATIVE
DEFAULT_RESULT_ROOT = PROJECT_ROOT / RESULT_RELATIVE
V1_RESULT_ROOT = PROJECT_ROOT / V1_RESULT_RELATIVE
RAW_NAME = "20090377.json"
RAW_RELATIVE = f"raw/{RAW_NAME}"

ATTEMPT_SCHEMA = "jx.tno.common-epoch-horizons-response-diagnostic.attempt.v1r1"
ATTEMPT_STATUS = "ONE_REQUEST_BUDGET_IRREVERSIBLY_RESERVED"
ATTEMPT_DOMAIN = (
    b"jx.tno.common-epoch-horizons-response-diagnostic.attempt.v1r1\0"
)
CUSTODY_SCHEMA = "jx.tno.common-epoch-horizons-response-diagnostic.custody.v1r1"
CUSTODY_STATUS = (
    "RETAINED_SINGLE_HORIZONS_RESPONSE_REPAIR_REVISION_1_DIAGNOSTIC_ONLY"
)
CUSTODY_DOMAIN = (
    b"jx.tno.common-epoch-horizons-response-diagnostic.custody.v1r1\0"
)
MAX_RESPONSE_BYTES = 1_048_576
MAX_ATTEMPT_BYTES = 1_048_576
MAX_CUSTODY_BYTES = 8_388_608
MAX_EXECUTOR_TAG_BYTES = 1_048_576

QUERY_URL = (
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
OBJECT_ID = "jpl-20090377"
SPK_ID = 20_090_377
SOLUTION_ID = 51
TARGET_ORDER = 0

CLAIMS = {
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
NONCLAIMS = {
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

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_PREPARE_MODULE: types.ModuleType | None = None
_BASE_MODULE: types.ModuleType | None = None


class AcquisitionError(RuntimeError):
    """A frozen authority, transport, custody, or publication gate failed."""


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise AcquisitionError("SHA-256 input must be exact bytes")
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
        raise AcquisitionError("value is not strict canonical JSON") from exc


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if type(key) is not str or key in value:
            raise AcquisitionError("JSON contains a duplicate or invalid key")
        value[key] = item
    return value


def _reject_constant(token: str) -> None:
    raise AcquisitionError(f"JSON contains non-finite constant {token}")


def _strict_json(payload: bytes, label: str, maximum: int) -> dict[str, Any]:
    if (
        type(payload) is not bytes
        or type(maximum) is not int
        or maximum <= 0
        or not 0 < len(payload) <= maximum
    ):
        raise AcquisitionError(f"{label} has an invalid bounded byte length")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except AcquisitionError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise AcquisitionError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise AcquisitionError(f"{label} root must be an exact object")
    return value


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    partial = dict(value)
    partial.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(partial))


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
    try:
        os.close(descriptor)
    except OSError:
        pass


def _read_regular_path(
    path: Path,
    label: str,
    maximum: int,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    descriptor, identity, payload = _hold_regular_path(path, label, maximum)
    try:
        if expected_bytes is not None and len(payload) != expected_bytes:
            raise AcquisitionError(f"{label} byte length differs")
        if expected_sha256 is not None and _sha256(payload) != expected_sha256:
            raise AcquisitionError(f"{label} SHA-256 differs")
        _recheck_held_path(path, descriptor, identity, label)
        return payload
    finally:
        _close_fd(descriptor)


def _hold_regular_path(
    path: Path, label: str, maximum: int
) -> tuple[int, os.stat_result, bytes]:
    if type(maximum) is not int or maximum <= 0:
        raise AcquisitionError(f"{label} byte ceiling is invalid")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOATIME", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        named = path.lstat()
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AcquisitionError(f"cannot open {label} without following links") from exc
    try:
        held = os.fstat(descriptor)
        if (
            not stat.S_ISREG(named.st_mode)
            or not stat.S_ISREG(held.st_mode)
            or named.st_uid != os.geteuid()
            or held.st_uid != os.geteuid()
            or named.st_nlink != 1
            or held.st_nlink != 1
            or not 0 < held.st_size <= maximum
            or _stat_identity(named) != _stat_identity(held)
        ):
            raise AcquisitionError(f"{label} is not a stable owned single-link file")
        payload = bytearray()
        while len(payload) < held.st_size:
            block = os.pread(descriptor, min(65_536, held.st_size - len(payload)), len(payload))
            if not block:
                raise AcquisitionError(f"{label} ended before its recorded length")
            payload.extend(block)
        after = os.fstat(descriptor)
        if _stat_identity(after) != _stat_identity(held):
            raise AcquisitionError(f"{label} changed while read")
        return descriptor, held, bytes(payload)
    except BaseException:
        _close_fd(descriptor)
        raise


def _recheck_held_path(
    path: Path,
    descriptor: int,
    identity: os.stat_result,
    label: str,
) -> None:
    try:
        held = os.fstat(descriptor)
        named = path.lstat()
    except OSError as exc:
        raise AcquisitionError(f"cannot recheck held {label}") from exc
    expected = _stat_identity(identity)
    if _stat_identity(held) != expected or _stat_identity(named) != expected:
        raise AcquisitionError(f"held {label} identity changed")


def _git(*arguments: str) -> bytes:
    environment = dict(os.environ)
    unsafe = {
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
            key in unsafe
            or key == "GIT_CONFIG_COUNT"
            or key.startswith("GIT_CONFIG_KEY_")
            or key.startswith("GIT_CONFIG_VALUE_")
        ):
            environment.pop(key, None)
    environment.update(
        {"GIT_NO_REPLACE_OBJECTS": "1", "GIT_OPTIONAL_LOCKS": "0", "LC_ALL": "C"}
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
        raise AcquisitionError("local Git identity inspection failed") from exc
    if completed.returncode != 0:
        raise AcquisitionError("local Git identity inspection failed")
    return completed.stdout


def _git_object_sha1(kind: bytes, payload: bytes) -> str:
    framed = kind + b" " + str(len(payload)).encode("ascii") + b"\0" + payload
    try:
        return hashlib.sha1(framed, usedforsecurity=False).hexdigest()
    except TypeError:
        return hashlib.sha1(framed).hexdigest()


def _raw_direct_child_tree(commit: str, parent: str, label: str) -> str:
    if _HEX40.fullmatch(commit) is None or _HEX40.fullmatch(parent) is None:
        raise AcquisitionError(f"{label} commit identity is malformed")
    payload = _git("cat-file", "commit", commit)
    if _git_object_sha1(b"commit", payload) != commit:
        raise AcquisitionError(f"{label} raw commit object hash differs")
    separator = payload.find(b"\n\n")
    if separator <= 0 or b"\r" in payload[:separator] or b"\0" in payload[:separator]:
        raise AcquisitionError(f"{label} raw commit header is malformed")
    lines = payload[:separator].split(b"\n")
    trees = [line for line in lines if line.startswith(b"tree ")]
    parents = [line for line in lines if line.startswith(b"parent ")]
    if (
        len(lines) < 4
        or len(trees) != 1
        or len(parents) != 1
        or lines[0] != trees[0]
        or lines[1] != parents[0]
    ):
        raise AcquisitionError(f"{label} must encode one first tree and one first parent")
    tree = trees[0][5:]
    encoded_parent = parents[0][7:]
    if re.fullmatch(rb"[0-9a-f]{40}", tree) is None:
        raise AcquisitionError(f"{label} raw tree identity is malformed")
    if encoded_parent != parent.encode("ascii"):
        raise AcquisitionError(f"{label} raw parent differs")
    return tree.decode("ascii")


def _status_paths() -> frozenset[str]:
    payload = _git("status", "--porcelain=v1", "-z", "--untracked-files=all")
    if not payload:
        return frozenset()
    fields = payload.split(b"\0")
    if fields[-1] != b"":
        raise AcquisitionError("Git status output is not NUL terminated")
    paths: set[str] = set()
    index = 0
    while index < len(fields) - 1:
        record = fields[index]
        if len(record) < 4 or record[2:3] != b" ":
            raise AcquisitionError("Git status output is malformed")
        code = record[:2]
        if b"R" in code or b"C" in code:
            raise AcquisitionError("Git rename or copy status is forbidden")
        try:
            path = record[3:].decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise AcquisitionError("Git status path is not UTF-8") from exc
        paths.add(path)
        index += 1
    return frozenset(paths)


def _nul_paths(payload: bytes, label: str) -> frozenset[str]:
    if not payload:
        return frozenset()
    fields = payload.split(b"\0")
    if fields[-1] != b"" or any(not item for item in fields[:-1]):
        raise AcquisitionError(f"{label} path output is malformed")
    try:
        paths = [item.decode("utf-8", errors="strict") for item in fields[:-1]]
    except UnicodeError as exc:
        raise AcquisitionError(f"{label} path is not UTF-8") from exc
    if len(paths) != len(set(paths)):
        raise AcquisitionError(f"{label} path output contains duplicates")
    return frozenset(paths)


def _expected_manifest(preregistered: bytes) -> bytes:
    anchor = b'        "tests/test_tno_common_epoch_horizons_response_diagnostic_v1.py",\n'
    insertion = (
        b'        "tests/test_tno_common_epoch_horizons_response_diagnostic_acquisition_v1.py",\n'
    )
    if preregistered.count(anchor) != 1 or insertion in preregistered:
        raise AcquisitionError("preregistered matrix insertion anchor differs")
    return preregistered.replace(anchor, insertion + anchor, 1)


def _preregistration_identity() -> dict[str, object]:
    return {
        "commit": PREPARE_COMMIT,
        "manifest_bytes": PREPARE_MANIFEST_BYTES,
        "manifest_sha256": PREPARE_MANIFEST_SHA256,
        "protocol_bytes": PREPARE_PROTOCOL_BYTES,
        "protocol_content_sha256": PREPARE_PROTOCOL_CONTENT_SHA256,
        "protocol_sha256": PREPARE_PROTOCOL_SHA256,
        "source_bytes": PREPARE_SOURCE_BYTES,
        "source_sha256": PREPARE_SOURCE_SHA256,
        "tag": PREPARE_TAG,
        "tag_object": PREPARE_TAG_OBJECT,
        "tag_payload_bytes": PREPARE_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": PREPARE_TAG_PAYLOAD_SHA256,
        "test_bytes": PREPARE_TEST_BYTES,
        "test_sha256": PREPARE_TEST_SHA256,
        "tree": PREPARE_TREE,
    }


def _load_exact_module(
    path: Path,
    label: str,
    size: int,
    digest: str,
    module_name: str,
) -> types.ModuleType:
    payload = _read_regular_path(
        path, label, size, expected_bytes=size, expected_sha256=digest
    )
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(payload, str(path), "exec"), module.__dict__)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    return module


def _prepare_module() -> types.ModuleType:
    global _PREPARE_MODULE
    if _PREPARE_MODULE is None:
        _PREPARE_MODULE = _load_exact_module(
            PROJECT_ROOT / PREPARE_SOURCE_RELATIVE,
            "prepare source",
            PREPARE_SOURCE_BYTES,
            PREPARE_SOURCE_SHA256,
            "_jx_horizons_response_diagnostic_prepare_v1r1_bound",
        )
    return _PREPARE_MODULE


def _base_module() -> types.ModuleType:
    global _BASE_MODULE
    if _BASE_MODULE is None:
        module = _load_exact_module(
            PROJECT_ROOT / BASE_EXECUTOR_RELATIVE,
            "sealed v1 executor source",
            BASE_EXECUTOR_BYTES,
            BASE_EXECUTOR_SHA256,
            "_jx_horizons_oracle_acquisition_v1_bound",
        )
        module.AcquisitionError = AcquisitionError
        _BASE_MODULE = module
    return _BASE_MODULE


def _verify_prepare_tag_and_bytes() -> tuple[dict[str, object], bytes]:
    if _git("cat-file", "-t", PREPARE_TAG_OBJECT) != b"tag\n":
        raise AcquisitionError("prepare object is not an annotated tag")
    if _git("rev-parse", f"refs/tags/{PREPARE_TAG}").strip() != PREPARE_TAG_OBJECT.encode():
        raise AcquisitionError("prepare tag ref differs")
    tag_payload = _git("cat-file", "tag", PREPARE_TAG_OBJECT)
    if (
        len(tag_payload) != PREPARE_TAG_PAYLOAD_BYTES
        or _sha256(tag_payload) != PREPARE_TAG_PAYLOAD_SHA256
        or _git_object_sha1(b"tag", tag_payload) != PREPARE_TAG_OBJECT
    ):
        raise AcquisitionError("prepare annotated tag payload differs")
    if _git("rev-parse", f"{PREPARE_TAG}^{{}}").strip() != PREPARE_COMMIT.encode():
        raise AcquisitionError("prepare peeled commit differs")
    tree = _raw_direct_child_tree(PREPARE_COMMIT, PREPARE_PARENT, "prepare commit")
    if tree != PREPARE_TREE:
        raise AcquisitionError("prepare raw tree differs")

    bindings = (
        (PREPARE_SOURCE_RELATIVE, PREPARE_SOURCE_BYTES, PREPARE_SOURCE_SHA256),
        (PREPARE_TEST_RELATIVE, PREPARE_TEST_BYTES, PREPARE_TEST_SHA256),
        (PREPARE_PROTOCOL_RELATIVE, PREPARE_PROTOCOL_BYTES, PREPARE_PROTOCOL_SHA256),
        (MANIFEST_RELATIVE, PREPARE_MANIFEST_BYTES, PREPARE_MANIFEST_SHA256),
    )
    committed: dict[str, bytes] = {}
    for relative, size, digest in bindings:
        payload = _git("cat-file", "blob", f"{PREPARE_TREE}:{relative}")
        if len(payload) != size or _sha256(payload) != digest:
            raise AcquisitionError(f"prepare committed {relative} bytes differ")
        committed[relative] = payload
    for relative, size, digest in bindings[:3]:
        current = _read_regular_path(
            PROJECT_ROOT / relative,
            relative,
            size,
            expected_bytes=size,
            expected_sha256=digest,
        )
        if current != committed[relative]:
            raise AcquisitionError(f"live {relative} differs from prepare commit")

    protocol_payload = committed[PREPARE_PROTOCOL_RELATIVE]
    protocol = _strict_json(protocol_payload, "prepare protocol", PREPARE_PROTOCOL_BYTES)
    if protocol_payload != _canonical_json(protocol):
        raise AcquisitionError("prepare protocol is not exact canonical JSON")
    if protocol.get("content_sha256") != PREPARE_PROTOCOL_CONTENT_SHA256:
        raise AcquisitionError("prepare protocol content SHA-256 differs")
    _verify_protocol_contract(protocol)
    if _git("rev-parse", f"refs/tags/{PREPARE_TAG}").strip() != PREPARE_TAG_OBJECT.encode():
        raise AcquisitionError("prepare tag ref moved during verification")
    return _preregistration_identity(), committed[MANIFEST_RELATIVE]


def _verify_protocol_contract(protocol: dict[str, Any]) -> None:
    try:
        request = protocol["frozen_request"]
        budget = protocol["durable_one_request_budget"]
        custody = protocol["future_custody_contract"]
        transport = protocol["future_transport"]
        publication = protocol["future_result_publication"]
        successor = protocol["future_preregistration_and_execution"]
    except (KeyError, TypeError) as exc:
        raise AcquisitionError("prepare protocol successor contract is absent") from exc
    if (
        protocol.get("schema")
        != "jx.tno.common-epoch-horizons-response-diagnostic.protocol.v1r1"
        or request.get("url") != QUERY_URL
        or request.get("headers") != [list(item) for item in REQUEST_HEADERS]
        or request.get("request_count") != 1
        or request.get("retries_for_any_reason") != 0
        or budget.get("attempt_root_relative_path") != ATTEMPT_RELATIVE
        or budget.get("schema") != ATTEMPT_SCHEMA
        or budget.get("status") != ATTEMPT_STATUS
        or custody.get("schema") != CUSTODY_SCHEMA
        or custody.get("status_exact") != CUSTODY_STATUS
        or custody.get("claims_exact") != CLAIMS
        or custody.get("nonclaims_exact") != NONCLAIMS
        or transport.get("http_status_required") != 200
        or transport.get("content_encoding") != "ABSENT_OR_EXACTLY_IDENTITY"
        or publication.get("result_root_relative_path") != RESULT_RELATIVE
        or publication.get("semantic_failure_after_complete_body")
        != "PUBLISH_RAW_AND_DIAGNOSTIC_CUSTODY"
        or successor.get("prepare_tag") != PREPARE_TAG
        or successor.get("executor_tag") != EXECUTOR_TAG
        or successor.get("successor_allowed_diff") != sorted(SUCCESSOR_PATHS)
    ):
        raise AcquisitionError("prepare protocol successor contract differs")


def _verify_executor_tag(head: str) -> dict[str, object]:
    reference = _git("rev-parse", f"refs/tags/{EXECUTOR_TAG}").strip()
    try:
        object_id = reference.decode("ascii", errors="strict")
    except UnicodeError as exc:
        raise AcquisitionError("executor tag object identity is not ASCII") from exc
    if _HEX40.fullmatch(object_id) is None:
        raise AcquisitionError("executor tag object identity is malformed")
    if _git("cat-file", "-t", object_id) != b"tag\n":
        raise AcquisitionError("executor authority is not an annotated tag")
    payload = _git("cat-file", "tag", object_id)
    if (
        not 0 < len(payload) <= MAX_EXECUTOR_TAG_BYTES
        or _git_object_sha1(b"tag", payload) != object_id
    ):
        raise AcquisitionError("executor raw annotated tag hash differs")
    separator = payload.find(b"\n\n")
    if separator <= 0:
        raise AcquisitionError("executor annotated tag header is malformed")
    lines = payload[:separator].split(b"\n")
    expected = [
        b"object " + head.encode("ascii"),
        b"type commit",
        b"tag " + EXECUTOR_TAG.encode("ascii"),
    ]
    if len(lines) < 4 or lines[:3] != expected:
        raise AcquisitionError("executor annotated tag target or name differs")
    if _git("rev-parse", f"{EXECUTOR_TAG}^{{}}").strip() != head.encode():
        raise AcquisitionError("executor annotated tag peeled commit differs")
    if _git("rev-parse", f"refs/tags/{EXECUTOR_TAG}").strip() != object_id.encode():
        raise AcquisitionError("executor annotated tag ref moved during verification")
    return {
        "tag": EXECUTOR_TAG,
        "tag_object": object_id,
        "tag_payload_bytes": len(payload),
        "tag_payload_sha256": _sha256(payload),
    }


def _open_authority_inputs(
    expected_manifest: bytes,
    successor_blobs: Mapping[str, bytes],
) -> list[tuple[Path, int, os.stat_result, str]]:
    specifications = (
        (PREPARE_SOURCE_RELATIVE, PREPARE_SOURCE_BYTES, PREPARE_SOURCE_SHA256),
        (PREPARE_TEST_RELATIVE, PREPARE_TEST_BYTES, PREPARE_TEST_SHA256),
        (PREPARE_PROTOCOL_RELATIVE, PREPARE_PROTOCOL_BYTES, PREPARE_PROTOCOL_SHA256),
        (BASE_EXECUTOR_RELATIVE, BASE_EXECUTOR_BYTES, BASE_EXECUTOR_SHA256),
        (SUCCESSOR_SOURCE_RELATIVE, MAX_SOURCE_BYTES, None),
        (SUCCESSOR_TEST_RELATIVE, MAX_TEST_BYTES, None),
        (MANIFEST_RELATIVE, MAX_MANIFEST_BYTES, None),
    )
    held: list[tuple[Path, int, os.stat_result, str]] = []
    try:
        for relative, maximum, fixed_digest in specifications:
            path = PROJECT_ROOT / relative
            descriptor, identity, payload = _hold_regular_path(path, relative, maximum)
            if fixed_digest is not None and _sha256(payload) != fixed_digest:
                raise AcquisitionError(f"held {relative} bytes differ")
            if relative == MANIFEST_RELATIVE and payload != expected_manifest:
                raise AcquisitionError("held successor manifest bytes differ")
            if relative in successor_blobs and payload != successor_blobs[relative]:
                raise AcquisitionError(f"held successor {relative} differs from commit")
            held.append((path, descriptor, identity, relative))
        return held
    except BaseException:
        _close_authority_inputs(held)
        raise


def _recheck_authority_inputs(
    held: Sequence[tuple[Path, int, os.stat_result, str]]
) -> None:
    for path, descriptor, identity, label in held:
        _recheck_held_path(path, descriptor, identity, label)
    if V1_RESULT_ROOT.exists() or V1_RESULT_ROOT.is_symlink():
        raise AcquisitionError("failed v1 result path must remain absent")


def _close_authority_inputs(
    held: Sequence[tuple[Path, int, os.stat_result, str]]
) -> None:
    for _path, descriptor, _identity, _label in held:
        _close_fd(descriptor)


def _verify_execution_authority(
    *, require_sealed: bool, require_executor_tag: bool
) -> tuple[dict[str, Any], list[tuple[Path, int, os.stat_result, str]]]:
    identity, prepare_manifest = _verify_prepare_tag_and_bytes()
    expected_manifest = _expected_manifest(prepare_manifest)
    try:
        branch = _git("branch", "--show-current").decode(
            "ascii", errors="strict"
        ).strip()
        head = _git("rev-parse", "--verify", "HEAD^{commit}").decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeError as exc:
        raise AcquisitionError("successor branch or HEAD identity is not ASCII") from exc
    if branch != SUCCESSOR_BRANCH or _HEX40.fullmatch(head) is None:
        raise AcquisitionError("successor branch or HEAD identity differs")
    changed = _status_paths()
    successor_blobs: dict[str, bytes] = {}
    executor_tag: dict[str, object] | None = None
    if head == PREPARE_COMMIT:
        if require_sealed:
            raise AcquisitionError("acquisition requires a sealed successor commit")
        if changed != SUCCESSOR_PATHS:
            raise AcquisitionError("precommit successor worktree roster differs")
        current_manifest = _read_regular_path(
            PROJECT_ROOT / MANIFEST_RELATIVE, MANIFEST_RELATIVE, MAX_MANIFEST_BYTES
        )
        if current_manifest != expected_manifest:
            raise AcquisitionError("precommit successor manifest differs")
        mode = "PRECOMMIT_PREPARE_HEAD_WITH_EXACT_THREE_PATH_WORKTREE_ROSTER"
        tree = PREPARE_TREE
        commit_paths = frozenset()
    else:
        if changed:
            raise AcquisitionError("sealed successor worktree is not clean")
        tree = _raw_direct_child_tree(head, PREPARE_COMMIT, "sealed successor")
        commit_paths = _nul_paths(
            _git(
                "diff-tree",
                "--no-commit-id",
                "--name-only",
                "--no-renames",
                "-r",
                "-z",
                PREPARE_TREE,
                tree,
            ),
            "successor commit",
        )
        cumulative_paths = _nul_paths(
            _git(
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                PREPARE_TREE,
                tree,
                "--",
            ),
            "successor cumulative diff",
        )
        if commit_paths != SUCCESSOR_PATHS or cumulative_paths != SUCCESSOR_PATHS:
            raise AcquisitionError("sealed successor changeset differs")
        for relative, maximum in (
            (SUCCESSOR_SOURCE_RELATIVE, MAX_SOURCE_BYTES),
            (SUCCESSOR_TEST_RELATIVE, MAX_TEST_BYTES),
            (MANIFEST_RELATIVE, MAX_MANIFEST_BYTES),
        ):
            payload = _git("cat-file", "blob", f"{tree}:{relative}")
            if not 0 < len(payload) <= maximum:
                raise AcquisitionError(f"committed successor {relative} size differs")
            successor_blobs[relative] = payload
        if successor_blobs[MANIFEST_RELATIVE] != expected_manifest:
            raise AcquisitionError("committed successor manifest differs")
        mode = "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY"
        if require_executor_tag:
            executor_tag = _verify_executor_tag(head)

    held = _open_authority_inputs(expected_manifest, successor_blobs)
    try:
        _recheck_authority_inputs(held)
        record = {
            "authority_mode": mode,
            "branch": branch,
            "commit_paths": sorted(commit_paths),
            "executor_tag": executor_tag,
            "head": head,
            "prepare_identity": identity,
            "request_count": 1,
            "successor_identity": {
                relative: {
                    "bytes": len(payload),
                    "sha256": _sha256(payload),
                }
                for relative, payload in sorted(successor_blobs.items())
            },
            "tree": tree,
        }
        return record, held
    except BaseException:
        _close_authority_inputs(held)
        raise


def _recheck_sealed_git_authority(record: Mapping[str, Any]) -> None:
    """Repeat immutable Git/tag authority without consulting dirty status.

    The attempt/result roots intentionally dirty the worktree after reservation,
    so this recheck binds the branch, raw HEAD object, explicit tree diffs,
    annotated executor tag, prepare tag, and committed successor blobs directly.
    """
    if record.get("authority_mode") != (
        "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY"
    ):
        raise AcquisitionError("sealed Git authority mode differs")
    try:
        branch = _git("branch", "--show-current").decode("ascii", errors="strict").strip()
        head = _git("rev-parse", "--verify", "HEAD^{commit}").decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeError as exc:
        raise AcquisitionError("sealed Git identity is not ASCII") from exc
    if branch != SUCCESSOR_BRANCH or head != record.get("head"):
        raise AcquisitionError("sealed branch or HEAD moved after initial authority")
    tree = _raw_direct_child_tree(head, PREPARE_COMMIT, "rechecked sealed successor")
    if tree != record.get("tree"):
        raise AcquisitionError("sealed successor tree moved after initial authority")
    commit_paths = _nul_paths(
        _git(
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "--no-renames",
            "-r",
            "-z",
            PREPARE_TREE,
            tree,
        ),
        "rechecked successor commit",
    )
    cumulative_paths = _nul_paths(
        _git(
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            PREPARE_TREE,
            tree,
            "--",
        ),
        "rechecked successor cumulative diff",
    )
    if commit_paths != SUCCESSOR_PATHS or cumulative_paths != SUCCESSOR_PATHS:
        raise AcquisitionError("sealed successor changeset moved after initial authority")
    prepare_identity, _manifest = _verify_prepare_tag_and_bytes()
    if prepare_identity != record.get("prepare_identity"):
        raise AcquisitionError("prepare authority moved after initial authority")
    observed_tag = _verify_executor_tag(head)
    if observed_tag != record.get("executor_tag"):
        raise AcquisitionError("executor annotated tag moved after initial authority")
    successor_identity = record.get("successor_identity")
    if type(successor_identity) is not dict or set(successor_identity) != SUCCESSOR_PATHS:
        raise AcquisitionError("successor committed identity roster differs")
    for relative in sorted(SUCCESSOR_PATHS):
        expected = successor_identity[relative]
        payload = _git("cat-file", "blob", f"{tree}:{relative}")
        if (
            type(expected) is not dict
            or tuple(sorted(expected)) != ("bytes", "sha256")
            or type(expected.get("bytes")) is not int
            or expected["bytes"] <= 0
            or not _is_sha(expected.get("sha256"), 64)
            or len(payload) != expected["bytes"]
            or _sha256(payload) != expected["sha256"]
        ):
            raise AcquisitionError("successor committed bytes moved after initial authority")
    try:
        final_branch = _git("branch", "--show-current").decode(
            "ascii", errors="strict"
        ).strip()
    except UnicodeError as exc:
        raise AcquisitionError("sealed branch moved to a non-ASCII identity") from exc
    executor_tag = record.get("executor_tag")
    expected_executor_tag_object = (
        executor_tag.get("tag_object") if type(executor_tag) is dict else None
    )
    if (
        final_branch != SUCCESSOR_BRANCH
        or _git("rev-parse", "--verify", "HEAD^{commit}").strip()
        != head.encode("ascii")
        or _git("rev-parse", f"refs/tags/{PREPARE_TAG}").strip()
        != PREPARE_TAG_OBJECT.encode("ascii")
        or not _is_sha(expected_executor_tag_object, 40)
        or _git("rev-parse", f"refs/tags/{EXECUTOR_TAG}").strip()
        != expected_executor_tag_object.encode("ascii")
    ):
        raise AcquisitionError(
            "sealed branch, HEAD, or annotated tag moved during authority recheck"
        )


def audit_prerequisites() -> dict[str, Any]:
    record, held = _verify_execution_authority(
        require_sealed=False, require_executor_tag=False
    )
    try:
        _recheck_authority_inputs(held)
        return {"authority": record, "status": "PASSED_WITH_ZERO_REQUESTS"}
    finally:
        _close_authority_inputs(held)


def _request_budget() -> dict[str, object]:
    return {
        "headers": [list(item) for item in REQUEST_HEADERS],
        "maximum_request_invocations": 1,
        "method": "GET",
        "retries": 0,
        "url": QUERY_URL,
    }


def _make_attempt() -> dict[str, Any]:
    value: dict[str, Any] = {
        "preregistration_identity": _preregistration_identity(),
        "protocol_content_sha256": PREPARE_PROTOCOL_CONTENT_SHA256,
        "protocol_file_sha256": PREPARE_PROTOCOL_SHA256,
        "request_budget": _request_budget(),
        "schema": ATTEMPT_SCHEMA,
        "status": ATTEMPT_STATUS,
    }
    value["content_sha256"] = _domain_hash(ATTEMPT_DOMAIN, value)
    return value


def _is_sha(value: object, length: int) -> bool:
    pattern = _HEX40 if length == 40 else _HEX64
    return type(value) is str and pattern.fullmatch(value) is not None


def _positive_int(value: object, maximum: int | None = None) -> bool:
    return (
        type(value) is int
        and value > 0
        and (maximum is None or value <= maximum)
    )


def _validate_identity(value: object) -> None:
    expected = _preregistration_identity()
    if type(value) is not dict or value != expected:
        raise AcquisitionError("preregistration identity differs")
    for key in ("commit", "tag_object", "tree"):
        if not _is_sha(value[key], 40):
            raise AcquisitionError("preregistration Git identity type differs")
    for key in (
        "manifest_sha256",
        "protocol_content_sha256",
        "protocol_sha256",
        "source_sha256",
        "tag_payload_sha256",
        "test_sha256",
    ):
        if not _is_sha(value[key], 64):
            raise AcquisitionError("preregistration SHA-256 type differs")
    for key in (
        "manifest_bytes",
        "protocol_bytes",
        "source_bytes",
        "tag_payload_bytes",
        "test_bytes",
    ):
        if not _positive_int(value[key]):
            raise AcquisitionError("preregistration byte count type differs")


def _validate_attempt_record(value: dict[str, Any]) -> None:
    if tuple(sorted(value)) != (
        "content_sha256",
        "preregistration_identity",
        "protocol_content_sha256",
        "protocol_file_sha256",
        "request_budget",
        "schema",
        "status",
    ):
        raise AcquisitionError("ATTEMPT top-level key roster differs")
    if value.get("schema") != ATTEMPT_SCHEMA or value.get("status") != ATTEMPT_STATUS:
        raise AcquisitionError("ATTEMPT schema or status differs")
    _validate_identity(value.get("preregistration_identity"))
    if (
        value.get("protocol_content_sha256") != PREPARE_PROTOCOL_CONTENT_SHA256
        or value.get("protocol_file_sha256") != PREPARE_PROTOCOL_SHA256
        or value.get("request_budget") != _request_budget()
        or not _is_sha(value.get("content_sha256"), 64)
        or value.get("content_sha256") != _domain_hash(ATTEMPT_DOMAIN, value)
    ):
        raise AcquisitionError("ATTEMPT scalar or domain binding differs")


def _primitive(name: str) -> Any:
    value = getattr(_base_module(), name, None)
    if not callable(value):
        raise AcquisitionError(f"sealed publication primitive {name} is unavailable")
    return value


def _normalize_root(root: Path) -> Path:
    return _primitive("_normalize_output_root")(root)


def _write_create_only(
    directory_fd: int,
    name: str,
    payload: bytes,
    identities: dict[str, os.stat_result],
    *,
    allow_empty: bool,
) -> os.stat_result:
    if type(payload) is not bytes or (not payload and not allow_empty):
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
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AcquisitionError(f"artifact {name} write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        final = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(final.st_mode)
            or stat.S_IMODE(final.st_mode) != 0o600
            or final.st_uid != os.geteuid()
            or final.st_nlink != 1
            or final.st_size != len(payload)
            or _stat_identity(final) != _stat_identity(named)
        ):
            raise AcquisitionError(f"created artifact {name} identity differs")
        identities[name] = final
        return final
    except BaseException as original:
        try:
            held = os.fstat(descriptor)
            named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(held.st_mode)
                or not stat.S_ISREG(named.st_mode)
                or held.st_uid != os.geteuid()
                or held.st_nlink != 1
                or _stat_identity(held) != _stat_identity(named)
            ):
                raise AcquisitionError(f"refused unsafe artifact {name} rollback")
            os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        except BaseException as cleanup:
            raise AcquisitionError(f"failed artifact {name} rollback is unsafe") from cleanup
        identities.pop(name, None)
        raise original
    finally:
        _close_fd(descriptor)


def _open_held_file(
    directory_fd: int,
    name: str,
    label: str,
    maximum: int,
    *,
    allow_empty: bool,
) -> tuple[int, os.stat_result, bytes]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOATIME", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot open {label} without following links") from exc
    try:
        held = os.fstat(descriptor)
        size_ok = 0 <= held.st_size <= maximum if allow_empty else 0 < held.st_size <= maximum
        if (
            not stat.S_ISREG(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0o600
            or held.st_uid != os.geteuid()
            or held.st_nlink != 1
            or not size_ok
            or _stat_identity(held) != _stat_identity(named)
        ):
            raise AcquisitionError(f"{label} is not an exact bounded held 0600 file")
        payload = bytearray()
        while len(payload) < held.st_size:
            block = os.pread(descriptor, min(65_536, held.st_size - len(payload)), len(payload))
            if not block:
                raise AcquisitionError(f"{label} ended before its recorded length")
            payload.extend(block)
        if _stat_identity(os.fstat(descriptor)) != _stat_identity(held):
            raise AcquisitionError(f"{label} changed while read")
        return descriptor, held, bytes(payload)
    except BaseException:
        _close_fd(descriptor)
        raise


def _recheck_held_file(
    directory_fd: int,
    name: str,
    descriptor: int,
    identity: os.stat_result,
    label: str,
) -> None:
    _primitive("_recheck_held_file")(directory_fd, name, descriptor, identity, label)


def _open_validated_attempt(
    root: Path,
) -> tuple[
    dict[str, Any],
    bytes,
    Path,
    list[int],
    int,
    os.stat_result,
    int,
    os.stat_result,
]:
    absolute = _normalize_root(root)
    parent_path = _primitive("_open_anchored_directory_path")(
        absolute.parent, "attempt parent"
    )
    parent_fd = parent_path[-1]
    root_fd = -1
    file_fd = -1
    try:
        root_fd, root_identity = _primitive("_open_directory_at")(
            parent_fd, absolute.name, "attempt root"
        )
        if frozenset(os.listdir(root_fd)) != frozenset({"ATTEMPT.json"}):
            raise AcquisitionError("attempt root roster differs")
        file_fd, file_identity, payload = _open_held_file(
            root_fd,
            "ATTEMPT.json",
            "ATTEMPT.json",
            MAX_ATTEMPT_BYTES,
            allow_empty=False,
        )
        value = _strict_json(payload, "ATTEMPT.json", MAX_ATTEMPT_BYTES)
        if payload != _canonical_json(value):
            raise AcquisitionError("ATTEMPT.json is not exact canonical JSON")
        _validate_attempt_record(value)
        _recheck_held_file(root_fd, "ATTEMPT.json", file_fd, file_identity, "ATTEMPT.json")
        _primitive("_reanchor_directory")(
            parent_fd, absolute.name, root_fd, "attempt root"
        )
        _primitive("_reanchor_directory_path")(
            absolute.parent, parent_path, "attempt parent"
        )
        return (
            value,
            payload,
            absolute,
            parent_path,
            root_fd,
            root_identity,
            file_fd,
            file_identity,
        )
    except BaseException:
        if file_fd >= 0:
            _close_fd(file_fd)
        if root_fd >= 0:
            _close_fd(root_fd)
        _primitive("_close_directory_path")(parent_path)
        raise


def _close_attempt_hold(
    hold: tuple[
        dict[str, Any], bytes, Path, list[int], int, os.stat_result, int, os.stat_result
    ]
) -> None:
    _value, _payload, _absolute, parent_path, root_fd, _root_id, file_fd, _file_id = hold
    _close_fd(file_fd)
    _close_fd(root_fd)
    _primitive("_close_directory_path")(parent_path)


def _recheck_attempt_hold(
    hold: tuple[
        dict[str, Any], bytes, Path, list[int], int, os.stat_result, int, os.stat_result
    ]
) -> None:
    _value, _payload, absolute, parent_path, root_fd, root_id, file_fd, file_id = hold
    parent_fd = parent_path[-1]
    if frozenset(os.listdir(root_fd)) != frozenset({"ATTEMPT.json"}):
        raise AcquisitionError("attempt root roster changed")
    _recheck_held_file(root_fd, "ATTEMPT.json", file_fd, file_id, "ATTEMPT.json")
    if _stat_identity(os.fstat(root_fd)) != _stat_identity(root_id):
        raise AcquisitionError("attempt root identity changed")
    _primitive("_reanchor_directory")(parent_fd, absolute.name, root_fd, "attempt root")
    _primitive("_reanchor_directory_path")(absolute.parent, parent_path, "attempt parent")


def validate_attempt(root: Path = DEFAULT_ATTEMPT_ROOT) -> dict[str, Any]:
    """Replay the irreversible request-budget marker without network access."""
    hold = _open_validated_attempt(root)
    try:
        _recheck_attempt_hold(hold)
        return hold[0]
    finally:
        _close_attempt_hold(hold)


def _attempt_binding(payload: bytes, value: Mapping[str, Any]) -> dict[str, object]:
    return {
        "content_sha256": value["content_sha256"],
        "file_bytes": len(payload),
        "file_sha256": _sha256(payload),
        "relative_path": f"{ATTEMPT_RELATIVE}/ATTEMPT.json",
        "status": ATTEMPT_STATUS,
    }


def _v1_gate(payload: bytes) -> tuple[str, str | None]:
    base = _base_module()
    module = base._prereg_module()
    targets = [target for target in module.TARGETS if target.spk_id == SPK_ID]
    if len(targets) != 1 or targets[0].order != TARGET_ORDER:
        raise AcquisitionError("frozen v1 Sedna target binding differs")
    try:
        module.validate_horizons_response(payload, targets[0])
    except module.HorizonsOracleProtocolError as exc:
        return "REJECTED_BY_FROZEN_V1_GATE", str(exc)
    except Exception:
        return "FROZEN_V1_VALIDATOR_INTERNAL_EXCEPTION", None
    return "PASS_DIAGNOSTIC_ONLY", None


def _diagnosis(payload: bytes) -> dict[str, Any]:
    module = _prepare_module()
    try:
        observed = module.diagnostic_observations(payload)
    except Exception as exc:
        raise AcquisitionError("total diagnostic observation function failed") from exc
    if type(observed) is not dict:
        raise AcquisitionError("diagnostic observation returned a non-object")
    outcome, message = _v1_gate(payload)
    complete = dict(observed)
    complete["v1_gate_error_message"] = message
    complete["v1_gate_outcome"] = outcome
    _validate_diagnosis(complete)
    return complete


def _validate_diagnosis(value: object) -> None:
    if type(value) is not dict or tuple(sorted(value)) != (
        "element_epoch_line_candidates",
        "json_envelope_status",
        "result_string_sha256",
        "state_vector_epoch_token_candidates",
        "v1_gate_error_message",
        "v1_gate_outcome",
    ):
        raise AcquisitionError("diagnosis key roster differs")
    for key in ("element_epoch_line_candidates", "state_vector_epoch_token_candidates"):
        if type(value[key]) is not list or any(type(item) is not str for item in value[key]):
            raise AcquisitionError("diagnosis string-array type differs")
    if value["json_envelope_status"] not in {
        "STRICT_OBJECT",
        "STRICT_NON_OBJECT",
        "STRICT_JSON_REJECTED",
    }:
        raise AcquisitionError("diagnosis JSON status differs")
    if value["v1_gate_outcome"] not in {
        "PASS_DIAGNOSTIC_ONLY",
        "REJECTED_BY_FROZEN_V1_GATE",
        "FROZEN_V1_VALIDATOR_INTERNAL_EXCEPTION",
    }:
        raise AcquisitionError("diagnosis v1 gate outcome differs")
    digest = value["result_string_sha256"]
    if digest is not None and not _is_sha(digest, 64):
        raise AcquisitionError("diagnosis result string digest type differs")
    message = value["v1_gate_error_message"]
    if message is not None and type(message) is not str:
        raise AcquisitionError("diagnosis v1 error message type differs")
    if (
        value["v1_gate_outcome"] == "REJECTED_BY_FROZEN_V1_GATE"
        and type(message) is not str
    ) or (
        value["v1_gate_outcome"] != "REJECTED_BY_FROZEN_V1_GATE"
        and message is not None
    ):
        raise AcquisitionError("diagnosis v1 error message binding differs")


def _request_record() -> dict[str, object]:
    return {
        "completed_request_count": 1,
        "headers": [list(item) for item in REQUEST_HEADERS],
        "method": "GET",
        "object_id": OBJECT_ID,
        "retry_count": 0,
        "solution_id": SOLUTION_ID,
        "spk_id": SPK_ID,
        "target_order": TARGET_ORDER,
        "url": QUERY_URL,
    }


def _make_custody(
    raw: bytes,
    diagnosis: dict[str, Any],
    attempt_payload: bytes,
    attempt: dict[str, Any],
) -> dict[str, Any]:
    _validate_attempt_record(attempt)
    if attempt_payload != _canonical_json(attempt):
        raise AcquisitionError("attempt payload differs from exact canonical replay")
    partial: dict[str, Any] = {
        "attempt": _attempt_binding(attempt_payload, attempt),
        "claims": dict(CLAIMS),
        "diagnosis": diagnosis,
        "nonclaims": dict(NONCLAIMS),
        "preregistration_identity": _preregistration_identity(),
        "protocol_content_sha256": PREPARE_PROTOCOL_CONTENT_SHA256,
        "protocol_file_sha256": PREPARE_PROTOCOL_SHA256,
        "request": _request_record(),
        "response": {
            "entity_byte_semantics": "HTTP_ENTITY_OCTETS_AFTER_TRANSFER_CODING_REMOVAL_WITH_CONTENT_ENCODING_ABSENT_OR_IDENTITY",
            "http_status": 200,
            "raw_bytes": len(raw),
            "raw_relative_path": RAW_RELATIVE,
            "raw_sha256": _sha256(raw),
        },
        "schema": CUSTODY_SCHEMA,
        "status": CUSTODY_STATUS,
    }
    partial["content_sha256"] = _domain_hash(CUSTODY_DOMAIN, partial)
    ordered = {key: partial[key] for key in sorted(partial)}
    _validate_custody(ordered)
    if len(_canonical_json(ordered)) > MAX_CUSTODY_BYTES:
        raise AcquisitionError("canonical custody exceeds the frozen byte ceiling")
    return ordered


def _validate_custody(value: dict[str, Any]) -> None:
    if tuple(sorted(value)) != (
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
    ):
        raise AcquisitionError("custody top-level key roster differs")
    if value.get("schema") != CUSTODY_SCHEMA or value.get("status") != CUSTODY_STATUS:
        raise AcquisitionError("custody schema or status differs")
    if value.get("claims") != CLAIMS or value.get("nonclaims") != NONCLAIMS:
        raise AcquisitionError("custody claims or nonclaims differ")
    _validate_identity(value.get("preregistration_identity"))
    if (
        value.get("protocol_content_sha256") != PREPARE_PROTOCOL_CONTENT_SHA256
        or value.get("protocol_file_sha256") != PREPARE_PROTOCOL_SHA256
        or value.get("request") != _request_record()
        or not _is_sha(value.get("content_sha256"), 64)
        or value.get("content_sha256") != _domain_hash(CUSTODY_DOMAIN, value)
    ):
        raise AcquisitionError("custody scalar or domain binding differs")
    attempt = value.get("attempt")
    if type(attempt) is not dict or tuple(sorted(attempt)) != (
        "content_sha256",
        "file_bytes",
        "file_sha256",
        "relative_path",
        "status",
    ):
        raise AcquisitionError("custody attempt binding roster differs")
    if (
        not _is_sha(attempt.get("content_sha256"), 64)
        or not _positive_int(attempt.get("file_bytes"), MAX_ATTEMPT_BYTES)
        or not _is_sha(attempt.get("file_sha256"), 64)
        or attempt.get("relative_path") != f"{ATTEMPT_RELATIVE}/ATTEMPT.json"
        or attempt.get("status") != ATTEMPT_STATUS
    ):
        raise AcquisitionError("custody attempt binding differs")
    response = value.get("response")
    if type(response) is not dict or tuple(sorted(response)) != (
        "entity_byte_semantics",
        "http_status",
        "raw_bytes",
        "raw_relative_path",
        "raw_sha256",
    ):
        raise AcquisitionError("custody response roster differs")
    if (
        response.get("entity_byte_semantics")
        != "HTTP_ENTITY_OCTETS_AFTER_TRANSFER_CODING_REMOVAL_WITH_CONTENT_ENCODING_ABSENT_OR_IDENTITY"
        or response.get("http_status") != 200
        or type(response.get("raw_bytes")) is not int
        or not 0 <= response["raw_bytes"] <= MAX_RESPONSE_BYTES
        or response.get("raw_relative_path") != RAW_RELATIVE
        or not _is_sha(response.get("raw_sha256"), 64)
    ):
        raise AcquisitionError("custody response binding differs")
    _validate_diagnosis(value.get("diagnosis"))


def _require_absent_at(parent_fd: int, name: str, label: str) -> None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise AcquisitionError(f"cannot inspect {label}") from exc
    raise AcquisitionError(f"{label} already exists")


def _directory_anchor_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_uid,
        value.st_gid,
    )


def _recheck_result_tree(
    stage_fd: int,
    stage_identity: os.stat_result,
    raw_fd: int,
    raw_identity: os.stat_result,
    *,
    complete: bool,
) -> None:
    expected_root = frozenset({"raw", "CUSTODY.json"}) if complete else frozenset({"raw"})
    expected_raw = frozenset({RAW_NAME}) if complete else frozenset()
    try:
        held_stage = os.fstat(stage_fd)
        held_raw = os.fstat(raw_fd)
        root_roster = frozenset(os.listdir(stage_fd))
        raw_roster = frozenset(os.listdir(raw_fd))
    except OSError as exc:
        raise AcquisitionError("cannot recheck diagnostic result tree") from exc
    if (
        _directory_anchor_identity(held_stage)
        != _directory_anchor_identity(stage_identity)
        or _directory_anchor_identity(held_raw)
        != _directory_anchor_identity(raw_identity)
    ):
        raise AcquisitionError("diagnostic result directory identity changed")
    if root_roster != expected_root or raw_roster != expected_raw:
        raise AcquisitionError("diagnostic result tree roster differs")


def _safe_unlink_known(
    directory_fd: int,
    name: str,
    identity: os.stat_result,
    label: str,
) -> None:
    try:
        named = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise AcquisitionError(f"cannot inspect {label} for cleanup") from exc
    if (
        not stat.S_ISREG(named.st_mode)
        or named.st_uid != os.geteuid()
        or named.st_nlink != 1
        or _stat_identity(named) != _stat_identity(identity)
    ):
        raise AcquisitionError(f"refused replaced or unsafe {label} cleanup")
    try:
        os.unlink(name, dir_fd=directory_fd)
    except OSError as exc:
        raise AcquisitionError(f"cannot remove {label}") from exc


def _remove_attempt_staging(
    staging: Path,
    final: Path,
    parent_fd: int,
    stage_fd: int,
    identities: Mapping[str, os.stat_result],
) -> None:
    if (
        staging.parent != final.parent
        or not staging.name.startswith(f".{final.name}.staging-")
        or staging == final
    ):
        raise AcquisitionError("refused unsafe attempt staging cleanup target")
    _primitive("_reanchor_directory")(parent_fd, staging.name, stage_fd, "attempt staging")
    names = tuple(os.listdir(stage_fd))
    if set(names) != set(identities) or not set(names).issubset({"ATTEMPT.json"}):
        raise AcquisitionError("refused unexpected attempt staging cleanup roster")
    for name in names:
        _safe_unlink_known(stage_fd, name, identities[name], "attempt staging file")
    os.fsync(stage_fd)
    _primitive("_reanchor_directory")(parent_fd, staging.name, stage_fd, "attempt staging")
    try:
        os.rmdir(staging.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise AcquisitionError("cannot remove attempt staging root") from exc


def _remove_result_staging(
    staging: Path,
    final: Path,
    parent_fd: int,
    stage_fd: int,
    raw_fd: int,
    raw_identities: Mapping[str, os.stat_result],
    stage_identities: Mapping[str, os.stat_result],
) -> None:
    if (
        staging.parent != final.parent
        or not staging.name.startswith(f".{final.name}.staging-")
        or staging == final
    ):
        raise AcquisitionError("refused unsafe result staging cleanup target")
    _primitive("_reanchor_directory")(parent_fd, staging.name, stage_fd, "result staging")
    if raw_fd >= 0:
        _primitive("_reanchor_directory")(stage_fd, "raw", raw_fd, "result staging raw")
        raw_names = tuple(os.listdir(raw_fd))
        if set(raw_names) != set(raw_identities) or not set(raw_names).issubset({RAW_NAME}):
            raise AcquisitionError("refused unexpected staged raw cleanup roster")
        for name in raw_names:
            _safe_unlink_known(raw_fd, name, raw_identities[name], "staged raw file")
        os.fsync(raw_fd)
        _primitive("_reanchor_directory")(stage_fd, "raw", raw_fd, "result staging raw")
        try:
            os.rmdir("raw", dir_fd=stage_fd)
        except OSError as exc:
            raise AcquisitionError("cannot remove staged raw directory") from exc
    stage_names = tuple(os.listdir(stage_fd))
    if set(stage_names) != set(stage_identities) or not set(stage_names).issubset(
        {".CUSTODY.json.partial", "CUSTODY.json"}
    ):
        raise AcquisitionError("refused unexpected result staging cleanup roster")
    for name in stage_names:
        _safe_unlink_known(stage_fd, name, stage_identities[name], "staged custody file")
    os.fsync(stage_fd)
    _primitive("_reanchor_directory")(parent_fd, staging.name, stage_fd, "result staging")
    try:
        os.rmdir(staging.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except OSError as exc:
        raise AcquisitionError("cannot remove result staging root") from exc


def _publish_attempt(
    parent_fd: int,
    attempt_root: Path,
    attempt_payload: bytes,
) -> None:
    staging, stage_fd = _primitive("_create_private_staging")(parent_fd, attempt_root)
    identities: dict[str, os.stat_result] = {}
    file_fd = -1
    published = False
    try:
        _write_create_only(
            stage_fd,
            "ATTEMPT.json",
            attempt_payload,
            identities,
            allow_empty=False,
        )
        os.fsync(stage_fd)
        if frozenset(os.listdir(stage_fd)) != frozenset({"ATTEMPT.json"}):
            raise AcquisitionError("attempt staging roster differs")
        file_fd, file_identity, observed_payload = _open_held_file(
            stage_fd,
            "ATTEMPT.json",
            "staged ATTEMPT.json",
            MAX_ATTEMPT_BYTES,
            allow_empty=False,
        )
        if (
            observed_payload != attempt_payload
            or _stat_identity(file_identity)
            != _stat_identity(identities["ATTEMPT.json"])
        ):
            raise AcquisitionError("staged ATTEMPT.json identity or bytes differ")
        stage_identity = os.fstat(stage_fd)
        _primitive("_reanchor_directory")(
            parent_fd, staging.name, stage_fd, "attempt staging"
        )
        _require_absent_at(parent_fd, attempt_root.name, "attempt root")
        _primitive("_rename_noreplace")(
            parent_fd, staging.name, parent_fd, attempt_root.name
        )
        published = True
        staging = attempt_root
        _primitive("_reanchor_directory")(
            parent_fd, attempt_root.name, stage_fd, "published attempt root"
        )
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise AcquisitionError(
                "attempt marker exists but parent durability is indeterminate; no request sent"
            ) from exc
        _primitive("_reanchor_directory")(
            parent_fd, attempt_root.name, stage_fd, "published attempt root"
        )
        if (
            _directory_anchor_identity(os.fstat(stage_fd))
            != _directory_anchor_identity(stage_identity)
            or frozenset(os.listdir(stage_fd)) != frozenset({"ATTEMPT.json"})
        ):
            raise AcquisitionError("published attempt tree identity or roster differs")
        _recheck_held_file(
            stage_fd,
            "ATTEMPT.json",
            file_fd,
            file_identity,
            "published ATTEMPT.json",
        )
    except (AcquisitionError, OSError):
        if not published:
            _remove_attempt_staging(
                staging, attempt_root, parent_fd, stage_fd, identities
            )
        raise
    finally:
        if file_fd >= 0:
            _close_fd(file_fd)
        _close_fd(stage_fd)


def _open_result_for_replay(
    root: Path,
) -> tuple[Path, list[int], int, os.stat_result, int, os.stat_result]:
    absolute = _normalize_root(root)
    parent_path = _primitive("_open_anchored_directory_path")(
        absolute.parent, "result parent"
    )
    parent_fd = parent_path[-1]
    root_fd = -1
    raw_fd = -1
    try:
        root_fd, root_identity = _primitive("_open_directory_at")(
            parent_fd, absolute.name, "result root"
        )
        if frozenset(os.listdir(root_fd)) != frozenset({"CUSTODY.json", "raw"}):
            raise AcquisitionError("result root roster differs")
        raw_fd, raw_identity = _primitive("_open_directory_at")(
            root_fd, "raw", "result raw directory"
        )
        return absolute, parent_path, root_fd, root_identity, raw_fd, raw_identity
    except BaseException:
        if raw_fd >= 0:
            _close_fd(raw_fd)
        if root_fd >= 0:
            _close_fd(root_fd)
        _primitive("_close_directory_path")(parent_path)
        raise


def _validate_result_fds(
    root_fd: int,
    root_identity: os.stat_result,
    raw_fd: int,
    raw_identity: os.stat_result,
    attempt_hold: tuple[
        dict[str, Any], bytes, Path, list[int], int, os.stat_result, int, os.stat_result
    ],
    *,
    expected_raw_identities: Mapping[str, os.stat_result] | None = None,
    expected_stage_identities: Mapping[str, os.stat_result] | None = None,
) -> tuple[dict[str, Any], list[tuple[int, str, int, os.stat_result, str]]]:
    held: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        if frozenset(os.listdir(root_fd)) != frozenset({"CUSTODY.json", "raw"}):
            raise AcquisitionError("result root roster differs")
        if frozenset(os.listdir(raw_fd)) != frozenset({RAW_NAME}):
            raise AcquisitionError("raw response roster differs")
        custody_fd, custody_id, custody_payload = _open_held_file(
            root_fd,
            "CUSTODY.json",
            "CUSTODY.json",
            MAX_CUSTODY_BYTES,
            allow_empty=False,
        )
        held.append((root_fd, "CUSTODY.json", custody_fd, custody_id, "CUSTODY.json"))
        raw_file_fd, raw_id, raw = _open_held_file(
            raw_fd, RAW_NAME, RAW_RELATIVE, MAX_RESPONSE_BYTES, allow_empty=True
        )
        held.append((raw_fd, RAW_NAME, raw_file_fd, raw_id, RAW_RELATIVE))
        if expected_raw_identities is not None:
            if set(expected_raw_identities) != {RAW_NAME} or _stat_identity(
                raw_id
            ) != _stat_identity(expected_raw_identities[RAW_NAME]):
                raise AcquisitionError("created raw response identity differs at replay")
        if expected_stage_identities is not None:
            if set(expected_stage_identities) != {"CUSTODY.json"} or _stat_identity(
                custody_id
            ) != _stat_identity(expected_stage_identities["CUSTODY.json"]):
                raise AcquisitionError("created custody identity differs at replay")
        custody = _strict_json(custody_payload, "CUSTODY.json", MAX_CUSTODY_BYTES)
        if custody_payload != _canonical_json(custody):
            raise AcquisitionError("CUSTODY.json is not exact canonical JSON")
        _validate_custody(custody)
        attempt, attempt_payload = attempt_hold[0], attempt_hold[1]
        if custody["attempt"] != _attempt_binding(attempt_payload, attempt):
            raise AcquisitionError("custody attempt marker binding differs")
        if (
            custody["response"]["raw_bytes"] != len(raw)
            or custody["response"]["raw_sha256"] != _sha256(raw)
        ):
            raise AcquisitionError("custody raw response binding differs")
        expected = _make_custody(raw, _diagnosis(raw), attempt_payload, attempt)
        if custody != expected or custody_payload != _canonical_json(expected):
            raise AcquisitionError("custody differs from exact offline replay")
        for directory_fd, name, descriptor, identity, label in held:
            _recheck_held_file(directory_fd, name, descriptor, identity, label)
        _recheck_attempt_hold(attempt_hold)
        _primitive("_reanchor_directory")(root_fd, "raw", raw_fd, "result raw directory")
        if _directory_anchor_identity(os.fstat(root_fd)) != _directory_anchor_identity(
            root_identity
        ):
            raise AcquisitionError("result root identity changed")
        if _directory_anchor_identity(os.fstat(raw_fd)) != _directory_anchor_identity(
            raw_identity
        ):
            raise AcquisitionError("result raw directory identity changed")
        return expected, held
    except BaseException:
        for _directory, _name, descriptor, _identity, _label in held:
            _close_fd(descriptor)
        raise


def validate_result(
    root: Path = DEFAULT_RESULT_ROOT,
    attempt_root: Path = DEFAULT_ATTEMPT_ROOT,
) -> dict[str, Any]:
    """Replay raw bytes, diagnosis, custody, and attempt binding without network."""
    attempt_hold = _open_validated_attempt(attempt_root)
    result_hold: tuple[Path, list[int], int, os.stat_result, int, os.stat_result] | None = None
    files: list[tuple[int, str, int, os.stat_result, str]] = []
    try:
        result_hold = _open_result_for_replay(root)
        absolute, parent_path, root_fd, root_id, raw_fd, raw_id = result_hold
        expected, files = _validate_result_fds(
            root_fd, root_id, raw_fd, raw_id, attempt_hold
        )
        parent_fd = parent_path[-1]
        _primitive("_reanchor_directory")(parent_fd, absolute.name, root_fd, "result root")
        _primitive("_reanchor_directory_path")(
            absolute.parent, parent_path, "result parent"
        )
        for directory_fd, name, descriptor, identity, label in files:
            _recheck_held_file(directory_fd, name, descriptor, identity, label)
        _recheck_attempt_hold(attempt_hold)
        _primitive("_reanchor_directory")(
            parent_fd, absolute.name, root_fd, "result root"
        )
        _primitive("_reanchor_directory")(
            root_fd, "raw", raw_fd, "result raw directory"
        )
        _recheck_result_tree(
            root_fd,
            root_id,
            raw_fd,
            raw_id,
            complete=True,
        )
        _primitive("_reanchor_directory_path")(
            absolute.parent, parent_path, "result parent"
        )
        for directory_fd, name, descriptor, identity, label in files:
            _recheck_held_file(directory_fd, name, descriptor, identity, label)
        _recheck_attempt_hold(attempt_hold)
        return expected
    finally:
        for _directory, _name, descriptor, _identity, _label in files:
            _close_fd(descriptor)
        if result_hold is not None:
            _absolute, parent_path, root_fd, _root_id, raw_fd, _raw_id = result_hold
            _close_fd(raw_fd)
            _close_fd(root_fd)
            _primitive("_close_directory_path")(parent_path)
        _close_attempt_hold(attempt_hold)


def _transport(url: str) -> bytes:
    if url != QUERY_URL:
        raise AcquisitionError("transport URL differs from frozen request")
    payload = _base_module()._fetch(url)
    if type(payload) is not bytes or len(payload) > MAX_RESPONSE_BYTES:
        raise AcquisitionError("transport returned an invalid bounded entity")
    return payload


def _acquire_to_roots(
    result_root: Path,
    attempt_root: Path,
    *,
    transport: Callable[[str], bytes] = _transport,
    authority: tuple[
        dict[str, Any], list[tuple[Path, int, os.stat_result, str]]
    ]
    | None = None,
) -> dict[str, Any]:
    """Execute one already-preregistered request and publish diagnostic custody."""
    acquired_authority = authority is None
    authority_record, authority_files = (
        _verify_execution_authority(require_sealed=True, require_executor_tag=True)
        if authority is None
        else authority
    )
    if authority_record.get("authority_mode") != (
        "SEALED_CLEAN_DIRECT_CHILD_EXACT_THREE_PATH_COMMIT_LIVE_BLOB_EQUALITY"
    ):
        if acquired_authority:
            _close_authority_inputs(authority_files)
        raise AcquisitionError("request zero requires sealed executor authority")
    try:
        result = _normalize_root(result_root)
        attempt = _normalize_root(attempt_root)
        if result.parent != attempt.parent or result == attempt:
            raise AcquisitionError("attempt and result must be distinct siblings")
        parent_path = _primitive("_open_anchored_directory_path")(
            result.parent, "diagnostic output parent"
        )
    except BaseException:
        if acquired_authority:
            _close_authority_inputs(authority_files)
        raise
    parent_fd = parent_path[-1]
    staging: Path | None = None
    stage_fd = -1
    raw_fd = -1
    stage_directory_identity: os.stat_result | None = None
    raw_directory_identity: os.stat_result | None = None
    raw_identities: dict[str, os.stat_result] = {}
    stage_identities: dict[str, os.stat_result] = {}
    replay_files: list[tuple[int, str, int, os.stat_result, str]] = []
    attempt_hold = None
    published = False
    request_invocations = 0
    try:
        _recheck_authority_inputs(authority_files)
        _primitive("_reanchor_directory_path")(
            result.parent, parent_path, "diagnostic output parent"
        )
        _require_absent_at(parent_fd, result.name, "diagnostic result root")
        _require_absent_at(parent_fd, attempt.name, "diagnostic attempt root")
        staging, stage_fd = _primitive("_create_private_staging")(parent_fd, result)
        raw_fd = _primitive("_create_private_directory_at")(
            stage_fd, "raw", "private diagnostic raw directory"
        )
        stage_directory_identity = os.fstat(stage_fd)
        raw_directory_identity = os.fstat(raw_fd)
        _primitive("_reanchor_directory")(
            parent_fd, staging.name, stage_fd, "private diagnostic result staging"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "private diagnostic raw directory"
        )
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=False,
        )

        attempt_value = _make_attempt()
        attempt_payload = _canonical_json(attempt_value)
        _publish_attempt(parent_fd, attempt, attempt_payload)
        attempt_hold = _open_validated_attempt(attempt)
        _recheck_attempt_hold(attempt_hold)
        _recheck_authority_inputs(authority_files)
        _primitive("_reanchor_directory")(
            parent_fd, staging.name, stage_fd, "private diagnostic result staging"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "private diagnostic raw directory"
        )
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=False,
        )
        if acquired_authority:
            _recheck_sealed_git_authority(authority_record)
        _recheck_authority_inputs(authority_files)
        _recheck_attempt_hold(attempt_hold)
        _primitive("_reanchor_directory")(
            parent_fd, staging.name, stage_fd, "private diagnostic result staging"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "private diagnostic raw directory"
        )
        _primitive("_reanchor_directory_path")(
            result.parent, parent_path, "diagnostic output parent"
        )
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=False,
        )
        _require_absent_at(parent_fd, result.name, "diagnostic result root")

        request_invocations += 1
        if request_invocations != 1:
            raise AcquisitionError("one-request budget accounting differs")
        raw = transport(QUERY_URL)
        if type(raw) is not bytes or len(raw) > MAX_RESPONSE_BYTES:
            raise AcquisitionError("transport returned an invalid complete entity")
        _write_create_only(
            raw_fd, RAW_NAME, raw, raw_identities, allow_empty=True
        )
        os.fsync(raw_fd)

        # No semantic function is invoked before the exact raw file and its
        # containing raw directory have both reached fsync successfully.
        diagnosis = _diagnosis(raw)
        custody = _make_custody(raw, diagnosis, attempt_payload, attempt_value)
        custody_payload = _canonical_json(custody)
        _write_create_only(
            stage_fd,
            ".CUSTODY.json.partial",
            custody_payload,
            stage_identities,
            allow_empty=False,
        )
        _primitive("_rename_custody_transactionally")(
            stage_fd, stage_identities
        )
        os.fsync(stage_fd)

        _recheck_authority_inputs(authority_files)
        _recheck_attempt_hold(attempt_hold)
        validated, replay_files = _validate_result_fds(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            attempt_hold,
            expected_raw_identities=raw_identities,
            expected_stage_identities=stage_identities,
        )
        if validated != custody:
            raise AcquisitionError("staged result replay differs")
        _primitive("_reanchor_directory")(
            parent_fd, staging.name, stage_fd, "private diagnostic result staging"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "private diagnostic raw directory"
        )
        _require_absent_at(parent_fd, result.name, "diagnostic result root")
        _recheck_authority_inputs(authority_files)
        _recheck_attempt_hold(attempt_hold)
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=True,
        )
        if acquired_authority:
            _recheck_sealed_git_authority(authority_record)
        _recheck_authority_inputs(authority_files)
        _recheck_attempt_hold(attempt_hold)
        _primitive("_reanchor_directory")(
            parent_fd, staging.name, stage_fd, "private diagnostic result staging"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "private diagnostic raw directory"
        )
        _primitive("_reanchor_directory_path")(
            result.parent, parent_path, "diagnostic output parent"
        )
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=True,
        )
        _require_absent_at(parent_fd, result.name, "diagnostic result root")
        _primitive("_reanchor_directory_path")(
            result.parent, parent_path, "diagnostic output parent"
        )
        _primitive("_rename_noreplace")(parent_fd, staging.name, parent_fd, result.name)
        published = True
        staging = None
        _primitive("_reanchor_directory")(
            parent_fd, result.name, stage_fd, "published diagnostic result root"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "published diagnostic raw directory"
        )
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=True,
        )
        try:
            os.fsync(parent_fd)
        except OSError as exc:
            raise AcquisitionError(
                "complete result exists but parent durability is indeterminate; run validate-result"
            ) from exc
        _recheck_authority_inputs(authority_files)
        _recheck_attempt_hold(attempt_hold)
        _primitive("_reanchor_directory")(
            parent_fd, result.name, stage_fd, "published diagnostic result root"
        )
        _primitive("_reanchor_directory")(
            stage_fd, "raw", raw_fd, "published diagnostic raw directory"
        )
        _recheck_result_tree(
            stage_fd,
            stage_directory_identity,
            raw_fd,
            raw_directory_identity,
            complete=True,
        )
        _primitive("_reanchor_directory_path")(
            result.parent, parent_path, "diagnostic output parent"
        )
        for directory_fd, name, descriptor, identity, label in replay_files:
            _recheck_held_file(directory_fd, name, descriptor, identity, label)
        return validated
    except (AcquisitionError, OSError):
        for _directory, _name, descriptor, _identity, _label in replay_files:
            _close_fd(descriptor)
        replay_files = []
        if staging is not None and not published and stage_fd >= 0:
            _remove_result_staging(
                staging,
                result,
                parent_fd,
                stage_fd,
                raw_fd,
                raw_identities,
                stage_identities,
            )
        raise
    finally:
        for _directory, _name, descriptor, _identity, _label in replay_files:
            _close_fd(descriptor)
        if attempt_hold is not None:
            _close_attempt_hold(attempt_hold)
        if raw_fd >= 0:
            _close_fd(raw_fd)
        if stage_fd >= 0:
            _close_fd(stage_fd)
        _primitive("_close_directory_path")(parent_path)
        if acquired_authority:
            _close_authority_inputs(authority_files)


def acquire() -> dict[str, Any]:
    """Acquire only into the exact preregistered production roots."""
    return _acquire_to_roots(DEFAULT_RESULT_ROOT, DEFAULT_ATTEMPT_ROOT)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit-prerequisites")
    commands.add_parser("acquire")
    attempt = commands.add_parser("validate-attempt")
    attempt.add_argument("--attempt-root", type=Path, default=DEFAULT_ATTEMPT_ROOT)
    result = commands.add_parser("validate-result")
    result.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    result.add_argument("--attempt-root", type=Path, default=DEFAULT_ATTEMPT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "audit-prerequisites":
            value = audit_prerequisites()
        elif arguments.command == "acquire":
            value = acquire()
        elif arguments.command == "validate-attempt":
            value = validate_attempt(arguments.attempt_root)
        else:
            value = validate_result(arguments.result_root, arguments.attempt_root)
    except AcquisitionError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
