#!/usr/bin/env python3
"""Preregister the native-epoch full129 H64/H128 numerical precursor.

This module freezes a bounded, 1,000-Julian-year qualification design for all
twelve retained TNO batches and all 129 nominal/covariance states per batch.
It inherits the exact short-horizon predicates from the annotated 3-Gyr
preregistration and changes no threshold.  The two candidate resolutions are
binary64 H64 and H128; ``H128`` means 128 fixed steps per Julian year, not
128-bit arithmetic.

The retained states remain at their twelve native covariance epochs.  This is
therefore an engineering precursor, not the common-epoch gate required by the
3-Gyr experiment.  The module is deliberately prepare-only: it imports no JX,
CuPy, SciPy, or REBOUND runtime, exposes no execute/resume command, performs no
dynamics, and produces no model output.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = (
    PROJECT_ROOT
    / "benchmarks/jx_tno_1kyr_full129_h64_h128_qualification_v1.py"
)
TEST_PATH = (
    PROJECT_ROOT
    / "tests/test_tno_1kyr_full129_h64_h128_qualification_v1.py"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs/tno_1kyr_full129_h64_h128_qualification_v1"
)

EXPECTED_BRANCH = "research/tno-3gyr-h128-qualification-v1-20261014"
PREREGISTRATION_TAG = (
    "tno-1kyr-full129-h64-h128-qualification-v1-preregistered-20261014"
)
FUTURE_RESULT_TAG = (
    "tno-1kyr-full129-h64-h128-qualification-v1-result-20261014"
)

PARENT_TAG = "tno-3gyr-full129-robustness-v1-preregistered-20261014"
PARENT_TAG_OBJECT = "1504e185aceea8093ae47250374483f0af6adc3f"
PARENT_TAG_PAYLOAD_BYTES = 288
PARENT_TAG_PAYLOAD_SHA256 = (
    "e29988471b6458d914dd16a3ba9a512393d8d95a31b65ae60032e0026e7abde2"
)
PARENT_COMMIT = "5acedb9dbc7cdb6a477bba34fe21de295899adb9"
PARENT_TREE = "c6c41ed6581bd72e725d4700716c3fcc67579775"
PARENT_PROTOCOL_CONTENT_SHA256 = (
    "85d6058563fd67bbb77df095e707b0f155123b291959488c31274f2439f86bea"
)

PROTOCOL_SCHEMA = "jx.tno.1kyr-full129-h64-h128-qualification.protocol.v1"
PROTOCOL_DOMAIN = (
    b"jx.tno.1kyr-full129-h64-h128-qualification.protocol.v1\0"
)
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})
FILESYSTEM_TRUST_BOUNDARY = "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR"

HORIZON_JULIAN_YEARS = 1_000
CHECKPOINT_INTERVAL_JULIAN_YEARS = 1
CHECKPOINT_COUNT_ENDPOINT_INCLUSIVE = 1_001
BATCH_COUNT = 12
TRACERS_PER_BATCH = 129
TOTAL_TRACER_STATES = BATCH_COUNT * TRACERS_PER_BATCH
H64_STEPS_PER_JULIAN_YEAR = 64
H128_STEPS_PER_JULIAN_YEAR = 128
H64_TOTAL_STEPS_PER_BATCH = HORIZON_JULIAN_YEARS * H64_STEPS_PER_JULIAN_YEAR
H128_TOTAL_STEPS_PER_BATCH = HORIZON_JULIAN_YEARS * H128_STEPS_PER_JULIAN_YEAR
H64_STEP_HEX = (1.0 / H64_STEPS_PER_JULIAN_YEAR).hex()
H128_STEP_HEX = (1.0 / H128_STEPS_PER_JULIAN_YEAR).hex()


class TNOH64H128QualificationError(RuntimeError):
    """The prepare-only native-epoch qualification boundary failed closed."""


@dataclass(frozen=True)
class FilePin:
    role: str
    relative_path: str
    byte_length: int
    sha256: str


FILE_PINS = (
    FilePin(
        "parent_source",
        "benchmarks/jx_tno_3gyr_full129_robustness_v1.py",
        66_757,
        "3c3e4216cfaff611e742c3c560bc8da6d10ce55683a11facb13ff5fa2ffd0608",
    ),
    FilePin(
        "parent_test",
        "tests/test_tno_3gyr_full129_robustness_v1.py",
        64_756,
        "96c05a125069cbd800e1a13b5579db88c7045809f5aa809075e491ff51067296",
    ),
    FilePin(
        "parent_protocol",
        "runs/tno_3gyr_full129_robustness_v1/PROTOCOL.json",
        37_244,
        "d17f99472ed4580ce18dbb7af1ec4248052c98197beec365ea5e3fd400a0f7a9",
    ),
    FilePin(
        "native_epoch_full129_bridge",
        "runs/tno_full128_reduced5_bridge_v1/BRIDGE.json",
        1_208_834,
        "0570e7d81917caad871de2b2a143d4fc083851894d3c72aab1b77f8e9591fcec",
    ),
    FilePin(
        "cpu_massive_source",
        "src/jxplanetx/restricted_wh_source.py",
        61_803,
        "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52",
    ),
    FilePin(
        "cpu_massive_map",
        "src/jxplanetx/fast_wisdom_holman.py",
        36_182,
        "99da08305f5c0a191dd53e64d02a9a396b471aaf9f7b3a062545b1de220f7a72",
    ),
    FilePin(
        "cpu_massive_native_extension",
        "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so",
        42_520,
        "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3",
    ),
    FilePin(
        "gpu_quiet_tracer",
        "src/jxplanetx/cuda_restricted_wh.py",
        74_758,
        "3c22c3d717fe5b07eb4ed929e5d32be591838fca8c1adbf02dec1263d3150927",
    ),
    FilePin(
        "gpu_all_conic_kepler",
        "src/jxplanetx/_all_conic_kepler.py",
        37_373,
        "3d38c569764957bc0c781a4c2a7c767ccf2b630a38b0c88688e9716ac2b79691",
    ),
    FilePin(
        "gpu_cpu_hybrid",
        "src/jxplanetx/cuda_restricted_wh_hybrid.py",
        37_420,
        "6b3ea46436dbdb7dc2c434cff005a88395fea6e00cd97c7be104650c75769b10",
    ),
    FilePin(
        "cpu_dop853_route",
        "src/jxplanetx/restricted_wh_route.py",
        38_194,
        "7eae131cda32ee3feed525b13226b1b271b0fc54c1dd5d6a82927b4dc4b94075",
    ),
    FilePin(
        "cpu_dop853_event_locator",
        "src/jxplanetx/cuda_tracer_events.py",
        67_504,
        "a3c5f1f9de29f504420aa892bd50f8a947b24d6df21cadf91dc243c7976dad89",
    ),
    FilePin(
        "fixed_lane_lifecycle",
        "src/jxplanetx/restricted_wh_lifecycle.py",
        38_996,
        "b0211778fcf51dbfef78049e74bafd05d5daee4db7dd18f743f4a5b5a14af346",
    ),
    FilePin(
        "replay_verified_checkpoint",
        "src/jxplanetx/restricted_wh_checkpoint.py",
        97_417,
        "83016a8c6ae4e6869494046f9c4ba5af63672792fa493406c35fe860056865d7",
    ),
    FilePin(
        "backend_contract",
        "src/jxplanetx/engine/contracts.py",
        16_390,
        "11c834620deedacd219a35bda46772f9be9f0d6243c644dee4fb02eb6c0c144d",
    ),
    FilePin(
        "backend_resolver",
        "src/jxplanetx/engine/backends.py",
        7_367,
        "c039ae04aab43c0809b200304a47e9d6e63bb7cbea76e0665724c75ab513deba",
    ),
    FilePin(
        "dependency_metadata",
        "pyproject.toml",
        4_793,
        "ffbce4075756ebdc5d0c8c04b2cc41676d828bf77192ca4793ecdd6766507fd8",
    ),
)


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise TNOH64H128QualificationError("SHA-256 input must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    except (TypeError, ValueError, RecursionError) as exc:
        raise TNOH64H128QualificationError(
            "value is not canonical-JSON encodable"
        ) from exc


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TNOH64H128QualificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_number(token: str) -> None:
    raise TNOH64H128QualificationError(
        f"floating or nonfinite JSON token rejected: {token}"
    )


def _strict_json(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TNOH64H128QualificationError("JSON payload must be exact bytes")
    try:
        value = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_strict_pairs,
            parse_float=_reject_number,
            parse_constant=_reject_number,
        )
    except TNOH64H128QualificationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise TNOH64H128QualificationError(
            "payload is not strict ASCII JSON"
        ) from exc
    if type(value) is not dict:
        raise TNOH64H128QualificationError("JSON root must be an object")
    return value


def _exact_equal(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _stat_identity(
    value: os.stat_result,
) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    chain = [absolute, *absolute.parents]
    for component in reversed(chain):
        if component.exists() and component.is_symlink():
            raise TNOH64H128QualificationError(
                f"{label} has a symlink ancestor"
            )


@contextmanager
def _held_directory(path: Path, label: str) -> Iterator[int]:
    _require_no_symlink_ancestors(path, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TNOH64H128QualificationError(f"cannot open {label}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise TNOH64H128QualificationError(f"{label} is not a directory")
        yield descriptor
    finally:
        os.close(descriptor)


def _read_file(
    path: Path,
    *,
    expected_bytes: int | None,
    expected_sha256: str | None,
    label: str,
) -> bytes:
    _require_no_symlink_ancestors(path, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise TNOH64H128QualificationError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise TNOH64H128QualificationError(
                f"{label} must be a single-link regular file"
            )
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise TNOH64H128QualificationError(f"{label} byte length differs")
        if before.st_size > 2_000_000:
            raise TNOH64H128QualificationError(
                f"{label} exceeds the bounded read ceiling"
            )
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                raise TNOH64H128QualificationError(f"{label} ended early")
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise TNOH64H128QualificationError(f"{label} changed while held")
    finally:
        os.close(descriptor)
    payload = bytes(chunks)
    if expected_sha256 is not None and _sha256(payload) != expected_sha256:
        raise TNOH64H128QualificationError(f"{label} SHA-256 differs")
    return payload


def _git(arguments: Sequence[str], label: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", os.fspath(PROJECT_ROOT), *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise TNOH64H128QualificationError(f"Git check failed for {label}")
    return completed.stdout


def _git_text(arguments: Sequence[str], label: str) -> str:
    try:
        return _git(arguments, label).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise TNOH64H128QualificationError(
            f"Git output is not ASCII for {label}"
        ) from exc


def _verify_git_authority() -> dict[str, Any]:
    branch = _git_text(["symbolic-ref", "--short", "HEAD"], "current branch")
    if branch != EXPECTED_BRANCH:
        raise TNOH64H128QualificationError(
            "current branch differs from the qualification branch"
        )
    if _git_text(["cat-file", "-t", PARENT_TAG], "parent tag type") != "tag":
        raise TNOH64H128QualificationError("parent tag is not annotated")
    if _git_text(["rev-parse", PARENT_TAG], "parent tag object") != PARENT_TAG_OBJECT:
        raise TNOH64H128QualificationError("parent tag object differs")
    tag_payload = _git(["cat-file", "tag", PARENT_TAG], "parent tag payload")
    if (
        len(tag_payload) != PARENT_TAG_PAYLOAD_BYTES
        or _sha256(tag_payload) != PARENT_TAG_PAYLOAD_SHA256
    ):
        raise TNOH64H128QualificationError("parent tag payload differs")
    if (
        _git_text(["rev-parse", f"{PARENT_TAG}^{{commit}}"], "parent commit")
        != PARENT_COMMIT
    ):
        raise TNOH64H128QualificationError("parent tag commit differs")
    if (
        _git_text(["rev-parse", f"{PARENT_COMMIT}^{{tree}}"], "parent tree")
        != PARENT_TREE
    ):
        raise TNOH64H128QualificationError("parent tree differs")
    ancestor = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(PROJECT_ROOT),
            "merge-base",
            "--is-ancestor",
            PARENT_COMMIT,
            "HEAD",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if ancestor.returncode != 0:
        raise TNOH64H128QualificationError(
            "parent commit is not an ancestor of HEAD"
        )
    return {
        "expected_branch": EXPECTED_BRANCH,
        "parent_tag": PARENT_TAG,
        "parent_tag_object": PARENT_TAG_OBJECT,
        "parent_tag_payload_bytes": PARENT_TAG_PAYLOAD_BYTES,
        "parent_tag_payload_sha256": PARENT_TAG_PAYLOAD_SHA256,
        "parent_commit": PARENT_COMMIT,
        "parent_tree": PARENT_TREE,
    }


def _verify_file_pin(pin: FilePin) -> tuple[dict[str, Any], bytes]:
    relative = Path(pin.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise TNOH64H128QualificationError("file pin path is not contained")
    live = _read_file(
        PROJECT_ROOT / relative,
        expected_bytes=pin.byte_length,
        expected_sha256=pin.sha256,
        label=pin.role,
    )
    committed = _git(["show", f"{PARENT_COMMIT}:{pin.relative_path}"], pin.role)
    if committed != live:
        raise TNOH64H128QualificationError(
            f"{pin.role} live bytes differ from the parent commit"
        )
    return (
        {
            "role": pin.role,
            "relative_path": pin.relative_path,
            "byte_length": pin.byte_length,
            "sha256": pin.sha256,
            "parent_commit": PARENT_COMMIT,
            "live_equals_parent_commit": True,
        },
        live,
    )


def _validate_parent_protocol(payload: bytes) -> dict[str, Any]:
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise TNOH64H128QualificationError("parent protocol is not canonical")
    if (
        value.get("schema") != "jx.tno.3gyr-full129-robustness.protocol.v1"
        or value.get("evidence_class")
        != "PREPARED_PREREGISTRATION_NO_MODEL_OUTPUT"
        or value.get("content_sha256") != PARENT_PROTOCOL_CONTENT_SHA256
    ):
        raise TNOH64H128QualificationError("parent protocol identity differs")
    integrator = value.get("cpu_gpu_integrator")
    sealed = (
        integrator.get("short_horizon_qualification_contract")
        if type(integrator) is dict
        else None
    )
    scope = sealed.get("scope") if type(sealed) is dict else None
    if (
        type(sealed) is not dict
        or type(scope) is not dict
        or scope.get("horizon_julian_years") != HORIZON_JULIAN_YEARS
        or scope.get("batch_orders") != list(range(BATCH_COUNT))
        or scope.get("tracer_orders") != list(range(TRACERS_PER_BATCH))
        or scope.get("checkpoint_count_endpoint_inclusive")
        != CHECKPOINT_COUNT_ENDPOINT_INCLUSIVE
        or scope.get("checkpoint_interval_julian_years")
        != CHECKPOINT_INTERVAL_JULIAN_YEARS
        or scope.get("common_epoch_bridge_must_pass_first") is not True
        or scope.get("reference_integrator")
        != "INDEPENDENT_CPU_IAS15_EXACT_IMPLEMENTATION_AND_RUNTIME_TO_BE_PINNED"
    ):
        raise TNOH64H128QualificationError(
            "parent short-horizon qualification seal differs"
        )
    return value


def _validate_bridge(payload: bytes) -> list[dict[str, Any]]:
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise TNOH64H128QualificationError("full129 bridge is not canonical")
    if (
        value.get("schema") != "jx.tno.full128-reduced5-bridge.states.v1"
        or value.get("status")
        != "TWELVE_FULL128_NATIVE_EPOCH_REDUCED5_BATCHES_PREPARED_NO_DYNAMICS"
        or value.get("batch_count") != BATCH_COUNT
        or type(value.get("batch_count")) is not int
        or value.get("tracer_count_per_batch") != TRACERS_PER_BATCH
        or type(value.get("tracer_count_per_batch")) is not int
        or value.get("tracer_state_rows_total") != TOTAL_TRACER_STATES
        or type(value.get("tracer_state_rows_total")) is not int
        or value.get("execution_authorized") is not False
        or value.get("outcomes_generated") is not False
    ):
        raise TNOH64H128QualificationError("full129 bridge identity differs")
    batches = value.get("batches")
    if type(batches) is not list or len(batches) != BATCH_COUNT:
        raise TNOH64H128QualificationError("full129 bridge batch roster differs")
    records: list[dict[str, Any]] = []
    for batch_order, batch in enumerate(batches):
        if (
            type(batch) is not dict
            or batch.get("batch_order") != batch_order
            or type(batch.get("batch_order")) is not int
            or type(batch.get("object_id")) is not str
            or not batch["object_id"]
            or type(batch.get("spk_id")) is not int
            or type(batch.get("covariance_epoch_jd_tdb_hex")) is not str
            or batch.get("massive_body_count") != 5
            or type(batch.get("massive_body_count")) is not int
            or batch.get("tracer_count") != TRACERS_PER_BATCH
            or type(batch.get("tracer_count")) is not int
            or not _is_sha256(batch.get("binary_payload_sha256"))
            or not _is_sha256(batch.get("content_sha256"))
        ):
            raise TNOH64H128QualificationError(
                "full129 bridge batch identity differs"
            )
        records.append(
            {
                "batch_order": batch_order,
                "object_id": batch["object_id"],
                "spk_id": batch["spk_id"],
                "covariance_epoch_jd_tdb_hex": batch[
                    "covariance_epoch_jd_tdb_hex"
                ],
                "massive_body_count": 5,
                "tracer_count": TRACERS_PER_BATCH,
                "tracer_orders": list(range(TRACERS_PER_BATCH)),
                "bridge_binary_payload_sha256": batch[
                    "binary_payload_sha256"
                ],
                "bridge_batch_content_sha256": batch["content_sha256"],
            }
        )
    if len({record["object_id"] for record in records}) != BATCH_COUNT:
        raise TNOH64H128QualificationError("full129 bridge object IDs repeat")
    return records


def validate_bound_prerequisites() -> dict[str, Any]:
    git_authority = _verify_git_authority()
    records: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for pin in FILE_PINS:
        record, payload = _verify_file_pin(pin)
        records.append(record)
        payloads[pin.role] = payload
    parent = _validate_parent_protocol(payloads["parent_protocol"])
    roster = _validate_bridge(payloads["native_epoch_full129_bridge"])
    return {
        "git_authority": git_authority,
        "exact_parent_files": records,
        "parent_protocol_content_sha256": parent["content_sha256"],
        "native_epoch_roster": roster,
        "authority_chain": (
            "ANNOTATED_3GYR_PREREGISTRATION_AND_EXACT_PARENT_COMMIT_BYTES_"
            "AUTHORIZE_ONLY_THIS_PREPARE_ONLY_NATIVE_EPOCH_PRECURSOR"
        ),
    }


def _implementation_record(path: Path, role: str) -> dict[str, Any]:
    payload = _read_file(
        path,
        expected_bytes=None,
        expected_sha256=None,
        label=role,
    )
    return {
        "role": role,
        "relative_path": path.relative_to(PROJECT_ROOT).as_posix(),
        "byte_length": len(payload),
        "sha256": _sha256(payload),
    }


def _claims() -> dict[str, bool]:
    return {
        "prepare_only_protocol_frozen": True,
        "execution_implemented": False,
        "execution_authorized": False,
        "dynamics_executed": False,
        "cpu_source_executed": False,
        "gpu_tracers_executed": False,
        "ias15_reference_executed": False,
        "native_epoch_all12_full129_h64_executed": False,
        "native_epoch_all12_full129_h128_executed": False,
        "native_epoch_all12_full129_numerical_precursor_passed": False,
        "common_epoch_full_covariance_bridge_qualified": False,
        "three_gyr_short_horizon_gate_satisfied": False,
        "all_12_h64_qualification_passed": False,
        "all_12_h128_qualification_passed": False,
        "independent_all129_ias15_reference_pinned": False,
        "quiet_gpu_map_qualified": False,
        "guarded_cpu_route_qualified": False,
        "active_routed_removed_crashed_restart_qualified": False,
        "general_disk_checkpoint_restart_qualified": False,
        "external_forcing_qualified": False,
        "three_gyr_execution_authorized": False,
        "classification_generated": False,
        "population_inference": False,
        "planet_nine_evidence": False,
        "science_result": False,
    }


def _resolution_record(
    name: str,
    *,
    steps_per_year: int,
    total_steps_per_batch: int,
    step_hex: str,
) -> dict[str, Any]:
    if (
        name not in ("H64", "H128")
        or type(steps_per_year) is not int
        or type(total_steps_per_batch) is not int
        or type(step_hex) is not str
    ):
        raise TNOH64H128QualificationError("resolution construction differs")
    return {
        "resolution_id": name,
        "arithmetic_dtype": "IEEE754_BINARY64",
        "resolution_label_encodes_steps_per_julian_year_not_bit_width": True,
        "source_step_julian_years_binary64_hex": step_hex,
        "steps_per_julian_year": steps_per_year,
        "horizon_julian_years": HORIZON_JULIAN_YEARS,
        "fixed_steps_per_batch": total_steps_per_batch,
        "batch_count": BATCH_COUNT,
        "tracer_count_per_batch": TRACERS_PER_BATCH,
        "trajectory_count": TOTAL_TRACER_STATES,
        "charged_gpu_lane_steps": TOTAL_TRACER_STATES * total_steps_per_batch,
        "cpu_source_primary_steps": BATCH_COUNT * total_steps_per_batch,
        "cpu_source_replay_steps": BATCH_COUNT * total_steps_per_batch,
        "source_steps_per_gpu_map_step": 1,
        "annual_checkpoint_years_inclusive": list(
            range(HORIZON_JULIAN_YEARS + 1)
        ),
        "annual_checkpoint_count_per_trajectory": (
            CHECKPOINT_COUNT_ENDPOINT_INCLUSIVE
        ),
        "total_tracer_checkpoint_states": (
            TOTAL_TRACER_STATES * CHECKPOINT_COUNT_ENDPOINT_INCLUSIVE
        ),
    }


def build_protocol() -> dict[str, Any]:
    prerequisites = validate_bound_prerequisites()
    parent = _validate_parent_protocol(
        next(
            payload
            for pin, payload in (
                (
                    pin,
                    _read_file(
                        PROJECT_ROOT / pin.relative_path,
                        expected_bytes=pin.byte_length,
                        expected_sha256=pin.sha256,
                        label=pin.role,
                    ),
                )
                for pin in FILE_PINS
                if pin.role == "parent_protocol"
            )
        )
    )
    sealed = parent["cpu_gpu_integrator"][
        "short_horizon_qualification_contract"
    ]
    roster = prerequisites["native_epoch_roster"]
    if [record["batch_order"] for record in roster] != list(range(BATCH_COUNT)):
        raise TNOH64H128QualificationError("native-epoch roster order differs")

    h64 = _resolution_record(
        "H64",
        steps_per_year=H64_STEPS_PER_JULIAN_YEAR,
        total_steps_per_batch=H64_TOTAL_STEPS_PER_BATCH,
        step_hex=H64_STEP_HEX,
    )
    h128 = _resolution_record(
        "H128",
        steps_per_year=H128_STEPS_PER_JULIAN_YEAR,
        total_steps_per_batch=H128_TOTAL_STEPS_PER_BATCH,
        step_hex=H128_STEP_HEX,
    )

    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "title": (
            "JX native-epoch full129 H64/H128 1-kyr numerical qualification "
            "precursor v1"
        ),
        "status": "PREPARED_NATIVE_EPOCH_QUALIFICATION_NO_EXECUTION",
        "evidence_class": (
            "PREPARED_1KYR_AUTONOMOUS_REDUCED5_ALL12_FULL129_"
            "NUMERICAL_QUALIFICATION_NO_MODEL_OUTPUT"
        ),
        "objective": (
            "Freeze an all-twelve, all-129, annual-checkpoint H64/H128 "
            "autonomous reduced-five numerical precursor without claiming the "
            "common-epoch gate required by the 3-Gyr experiment."
        ),
        "authority": prerequisites,
        "implementation": {
            "source": _implementation_record(
                SOURCE_PATH, "native-epoch qualification source"
            ),
            "test": _implementation_record(
                TEST_PATH, "native-epoch qualification test"
            ),
        },
        "scope": {
            "initial_epoch_policy": "TWELVE_NATIVE_COVARIANCE_EPOCHS",
            "common_epoch_propagation_performed": False,
            "common_epoch_gate_satisfied": False,
            "model": "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS",
            "batch_count": BATCH_COUNT,
            "batch_orders": list(range(BATCH_COUNT)),
            "tracer_count_per_batch": TRACERS_PER_BATCH,
            "tracer_orders": list(range(TRACERS_PER_BATCH)),
            "total_tracer_states_per_resolution": TOTAL_TRACER_STATES,
            "horizon_julian_years": HORIZON_JULIAN_YEARS,
            "checkpoint_interval_julian_years": (
                CHECKPOINT_INTERVAL_JULIAN_YEARS
            ),
            "checkpoint_count_endpoint_inclusive": (
                CHECKPOINT_COUNT_ENDPOINT_INCLUSIVE
            ),
            "native_epoch_batches": roster,
        },
        "resolutions": {"h64_control": h64, "h128_candidate": h128},
        "exact_arithmetic": {
            "h64_fixed_steps_per_batch": H64_TOTAL_STEPS_PER_BATCH,
            "h128_fixed_steps_per_batch": H128_TOTAL_STEPS_PER_BATCH,
            "dual_fixed_steps_per_batch": (
                H64_TOTAL_STEPS_PER_BATCH + H128_TOTAL_STEPS_PER_BATCH
            ),
            "dual_charged_gpu_lane_steps": (
                h64["charged_gpu_lane_steps"] + h128["charged_gpu_lane_steps"]
            ),
            "dual_cpu_source_primary_steps": (
                h64["cpu_source_primary_steps"]
                + h128["cpu_source_primary_steps"]
            ),
            "dual_cpu_source_primary_plus_replay_steps": (
                h64["cpu_source_primary_steps"]
                + h64["cpu_source_replay_steps"]
                + h128["cpu_source_primary_steps"]
                + h128["cpu_source_replay_steps"]
            ),
            "dual_tracer_checkpoint_states": (
                h64["total_tracer_checkpoint_states"]
                + h128["total_tracer_checkpoint_states"]
            ),
        },
        "sealed_parent_short_horizon_contract": sealed,
        "native_epoch_precursor_interpretation": {
            "all_parent_numerical_thresholds_reused_without_change": True,
            "all_individual_predicates_must_pass": True,
            "comparison_support": (
                "SAME_BATCH_SAME_TRACER_SAME_NATIVE_EPOCH_SAME_1001_ANNUAL_"
                "CHECKPOINTS"
            ),
            "h64_h128_pairing": "EXACT_BATCH_AND_TRACER_IDENTITY",
            "h128_ias15_pairing": "EXACT_BATCH_AND_TRACER_IDENTITY",
            "autonomous_energy_and_angular_gates_apply": True,
            "external_forcing_predicates_apply": False,
            "promotion_rule": (
                "PASS_IS_NATIVE_EPOCH_ENGINEERING_EVIDENCE_ONLY;REPEAT_AFTER_"
                "QUALIFIED_COMMON_EPOCH_BRIDGE_BEFORE_SETTING_ANY_PARENT_"
                "ALL12_H64_OR_H128_QUALIFICATION_CLAIM_TRUE"
            ),
        },
        "integrator_contract": {
            "massive_source": {
                "backend": "numpy_cpu",
                "method": "COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN",
                "callable": (
                    "jxplanetx.restricted_wh_source."
                    "build_wisdom_holman_source_tile"
                ),
                "source_body_count": 5,
                "same_runtime_complete_bitwise_replay_required": True,
                "maximum_existing_source_steps_per_tile": 1_000_000,
                "h64_and_h128_1kyr_fit_one_existing_tile": True,
            },
            "quiet_tracer": {
                "backend": "cupy_cuda",
                "method": "CUDA_RESTRICTED_WH_FIXED_STEP_FUSED",
                "callable": (
                    "jxplanetx.cuda_restricted_wh."
                    "advance_cuda_restricted_wh_fused"
                ),
                "dtype": "float64",
                "threads_per_block": 128,
                "source_steps_per_map_step": 1,
                "cpu_fallback_allowed": False,
                "deterministic_reductions": True,
                "fast_math": False,
                "cuda_compiler_options": ["--std=c++11", "--fmad=false"],
            },
            "guarded_route": {
                "backend": "numpy_scipy_cpu_workers",
                "method": "CONTINUOUS_RESTRICTED_FRAME_DOP853",
                "callable": (
                    "jxplanetx.cuda_restricted_wh_hybrid."
                    "advance_cuda_restricted_wh_hybrid"
                ),
                "discard_provisional_gpu_step_and_redo_only_guarded_lane": True,
                "outer_boundary_AU_hex": float(4096.0).hex(),
                "outer_boundary_label": "MODEL_DOMAIN_EXIT_NOT_PHYSICAL_EJECTION",
                "physical_collision_claim": False,
            },
            "reference": {
                "method": (
                    "INDEPENDENT_CPU_IAS15_EXACT_IMPLEMENTATION_AND_RUNTIME_"
                    "TO_BE_PINNED"
                ),
                "all_129_tracers_per_batch_required": True,
                "all_twelve_batches_required": True,
                "annual_checkpoint_count": CHECKPOINT_COUNT_ENDPOINT_INCLUSIVE,
                "rebound_5_1_1_may_satisfy_independent_requirement_if_exactly_pinned_and_audited": True,
                "implementation_and_runtime_currently_pinned": False,
            },
        },
        "restart_qualification_contract": {
            "monolithic_and_fresh_process_suffixes_must_match_bitwise": True,
            "restart_year": 500,
            "h64_restart_global_step": 500 * H64_STEPS_PER_JULIAN_YEAR,
            "h128_restart_global_step": 500 * H128_STEPS_PER_JULIAN_YEAR,
            "all_twelve_batches_and_all_129_lanes_required": True,
            "active_lane_case_required": True,
            "deterministic_guarded_route_fixture_required": True,
            "deterministic_removed_lane_fixture_required": True,
            "precommit_interruption_fixture_required": True,
            "route_removal_source_runtime_and_predecessor_digests_required": True,
            "existing_batch0_all_active_h64_restart_is_not_general_authority": True,
            "existing_replay_every_commit_checkpoint_is_not_a_3gyr_scalable_writer": True,
        },
        "runtime_contract": {
            "python_abi_required_by_tracked_native_extension": (
                "CPYTHON_3_14_X86_64_LINUX_GNU"
            ),
            "numpy_requirement": "ONE_EXACT_VERSION_TO_BE_PINNED_BY_EXECUTION_SUCCESSOR",
            "scipy_requirement": "ONE_EXACT_VERSION_TO_BE_PINNED_BY_EXECUTION_SUCCESSOR",
            "cupy_requirement": "ONE_EXACT_CUPY_CUDA12_OR_CUDA13_VERSION_TO_BE_PINNED",
            "ias15_requirement": "ONE_EXACT_REBOUND_5_1_1_RUNTIME_TO_BE_PINNED_AND_AUDITED",
            "observed_setup_diagnostic_only": {
                "python": "3.14.4",
                "numpy": "2.5.3",
                "cupy": "14.2.0",
                "gpu_model": "NVIDIA GeForce RTX 5060 Ti",
                "gpu_uuid_without_prefix": "66041ce7445ea391aadd703ae6ebc672",
                "cuda_driver_and_runtime_integer": 13020,
                "target_stack_qualified_by_this_observation": False,
            },
            "visible_cuda_device_required_for_future_execution": True,
            "gpu_uuid_driver_runtime_nvrtc_and_kernel_receipts_required": True,
            "native_extension_binary_is_platform_specific": True,
            "different_host_requires_rebuild_and_new_binary_receipt": True,
            "runtime_migration_without_separate_bitwise_qualification_allowed": False,
        },
        "execution_blockers": [
            "COMMON_EPOCH_FULL_COVARIANCE_PROPAGATION_NOT_QUALIFIED_FOR_PARENT_GATE",
            "INDEPENDENT_ALL129_CPU_IAS15_IMPLEMENTATION_AND_RUNTIME_NOT_PINNED",
            "EXACT_CUPY_CUDA_DRIVER_DEVICE_AND_NVRTC_RUNTIME_NOT_PINNED",
            "ALL12_FULL129_H64_H128_GPU_AND_CPU_ROUTE_NOT_YET_EXECUTED",
            "ACTIVE_ROUTED_REMOVED_AND_CRASHED_FRESH_PROCESS_RESTART_NOT_QUALIFIED",
            "QUALIFICATION_RESULT_SCHEMA_AND_EXECUTION_SUCCESSOR_NOT_PREREGISTERED",
        ],
        "unsafe_reuse_exclusions": [
            "DO_NOT_TREAT_UNPINNED_OR_REUSED_REBOUND_OUTPUT_AS_THE_TARGET_REFERENCE",
            "DO_NOT_REUSE_SELECTED9_STAGE1_RESULTS_AS_FULL129_EVIDENCE",
            "DO_NOT_REUSE_BATCH0_H64_RESTART_AS_ALL12_H64_H128_RESTART_AUTHORITY",
            "DO_NOT_REUSE_10MYR_H64_1000_SHARD_GRID_OR_WRITER_GEOMETRY_FOR_H128",
            "DO_NOT_USE_NATIVE_EPOCH_PASS_TO_SET_COMMON_EPOCH_PARENT_CLAIMS_TRUE",
            "DO_NOT_IMPORT_OR_EXECUTE_OPTIONAL_DYNAMICS_RUNTIMES_DURING_PREPARATION",
        ],
        "claim_ceiling": {
            "classification": "ENGINEERING_PREREGISTRATION_ONLY",
            "native_epoch_pass_if_future_execution_succeeds": (
                "NUMERICAL_PRECURSOR_ONLY_NOT_3GYR_EXECUTION_AUTHORITY"
            ),
            "no_dynamical_classification": True,
            "no_physical_stability_or_ejection_claim": True,
            "no_population_or_probability_interpretation": True,
            "no_planet_nine_detection_exclusion_or_ranking": True,
        },
        "preregistration": {
            "expected_branch": EXPECTED_BRANCH,
            "annotated_tag": PREREGISTRATION_TAG,
            "future_result_tag": FUTURE_RESULT_TAG,
            "prepared_default_root": DEFAULT_OUTPUT_ROOT.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "prepared_roster": ["PROTOCOL.json"],
            "execution_successor_must_pin": [
                "ANNOTATED_PREREGISTRATION_TAG_OBJECT",
                "TAG_PEELED_COMMIT_AND_TREE",
                "SOURCE_TEST_AND_PROTOCOL_BYTES_AND_SHA256",
                "EXACT_INDEPENDENT_IAS15_IMPLEMENTATION_AND_RUNTIME",
                "EXACT_NUMPY_SCIPY_CUPY_CUDA_DRIVER_DEVICE_NVRTC_RUNTIME",
                "RESULT_SCHEMA_OUTPUT_ROSTER_AND_CREATE_ONLY_WRITER",
            ],
            "no_automatic_git_network_gpu_or_execution_action": True,
        },
        "filesystem_trust_boundary": FILESYSTEM_TRUST_BOUNDARY,
        "public_cli": ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        "explicitly_absent_public_cli": [
            "execute",
            "execute-worker",
            "resume",
            "validate-result",
        ],
        "claims": _claims(),
        "content_sha256": "",
    }
    protocol["content_sha256"] = _domain_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise TNOH64H128QualificationError("protocol must be an exact object")
    expected = build_protocol()
    if (
        not _exact_equal(value, expected)
        or value.get("content_sha256") != _domain_hash(PROTOCOL_DOMAIN, value)
        or not _exact_equal(value.get("claims"), _claims())
    ):
        raise TNOH64H128QualificationError(
            "protocol bytes, semantics, or claim ceiling differ"
        )
    return value["content_sha256"]


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise TNOH64H128QualificationError(
                "protocol write made no progress"
            )
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise TNOH64H128QualificationError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        directory_fd,
        source.encode("ascii"),
        directory_fd,
        destination.encode("ascii"),
        1,
    )
    if result != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise TNOH64H128QualificationError(
                "protocol destination already exists"
            )
        raise TNOH64H128QualificationError(
            f"renameat2 failed: {os.strerror(observed_errno)}"
        )


def _validate_prepared_directory(
    root_fd: int,
    expected_payload: bytes,
) -> dict[str, Any]:
    root_metadata = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o700
        or root_metadata.st_uid != os.geteuid()
    ):
        raise TNOH64H128QualificationError(
            "prepared root is not an owner-private directory"
        )
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise TNOH64H128QualificationError("prepared root roster differs")
    metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected_payload)
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        raise TNOH64H128QualificationError("prepared protocol identity differs")
    descriptor = os.open(
        "PROTOCOL.json",
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                raise TNOH64H128QualificationError(
                    "prepared protocol ended early"
                )
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise TNOH64H128QualificationError(
                "prepared protocol changed while held"
            )
    finally:
        os.close(descriptor)
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise TNOH64H128QualificationError(
            "prepared roster changed while held"
        )
    payload = bytes(chunks)
    if payload != expected_payload:
        raise TNOH64H128QualificationError("prepared protocol bytes differ")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise TNOH64H128QualificationError(
            "prepared protocol is not canonical"
        )
    validate_protocol(value)
    return value


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise TNOH64H128QualificationError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = os.open(root.parent, flags)
    created = False
    prepared: dict[str, Any] | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise TNOH64H128QualificationError(
                "prepared root already exists; create-only publication refused"
            ) from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            os.fchmod(root_fd, 0o700)
            if os.listdir(root_fd):
                raise TNOH64H128QualificationError(
                    "new prepared root is not empty"
                )
            partial = ".PROTOCOL.json.partial"
            descriptor = os.open(
                partial,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
            prepared = _validate_prepared_directory(root_fd, payload)
            os.fsync(parent_fd)
        except Exception:
            for name in tuple(os.listdir(root_fd)):
                metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode):
                    os.unlink(name, dir_fd=root_fd)
            os.fsync(root_fd)
            raise
        finally:
            os.close(root_fd)
    except Exception:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        raise
    finally:
        os.close(parent_fd)
    if prepared is None:
        raise TNOH64H128QualificationError(
            "prepared protocol validation did not complete"
        )
    return prepared


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    expected_payload = _canonical_json(build_protocol())
    root = Path(os.path.abspath(os.fspath(output_root)))
    with _held_directory(root, "prepared native-epoch root") as root_fd:
        return _validate_prepared_directory(root_fd, expected_payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("build", "audit-prerequisites", "prepare", "validate-prepared"),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        value = build_protocol()
    elif arguments.command == "audit-prerequisites":
        value = validate_bound_prerequisites()
    elif arguments.command == "prepare":
        value = prepare(arguments.output_root)
    else:
        value = validate_prepared(arguments.output_root)
    sys.stdout.buffer.write(_canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
