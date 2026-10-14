#!/usr/bin/env python3
"""Execute the sealed reduced-five H64/H128 source preflight V1.

This is the successor implementation authorized by the prepare-only protocol
sealed at commit ``e8cae5a``.  It has exactly one production execution path:

* verify the sealed protocol, annotated tag, every input binding, and the
  pinned single-thread CPU runtime before lane zero;
* run the compiled ordered-Jacobi source map at H64 and H128, including each
  builder's complete exact replay;
* retain every JX source node and every-node massive-source invariant;
* run the independently pinned REBOUND 5.1.1 IAS15 annual reference;
* evaluate every applicable frozen source predicate without rounded decision
  arithmetic; and
* publish the exact nine-file success roster atomically and create-only.

Failure publishes no result.  There is no resume, threshold override, reduced
horizon, tracer, GPU, external-forcing, classification, or inference surface.
A passing artifact qualifies only this bounded autonomous reduced-five source
preflight and is not a scientific result.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import csv
from dataclasses import dataclass
import errno
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any, Iterable, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "benchmarks"
    / "jx_tno_1kyr_reduced5_h128_source_preflight_executor_v1.py"
)
EXECUTOR_TEST_RELATIVE_PATH = (
    "tests/test_tno_1kyr_reduced5_h128_source_preflight_executor_v1.py"
)
EXECUTOR_TEST_PATH = PROJECT_ROOT / EXECUTOR_TEST_RELATIVE_PATH
SEALED_PROTOCOL_RELATIVE_PATH = (
    "runs/tno_1kyr_reduced5_h128_source_preflight_v1/PROTOCOL.json"
)
SEALED_PROTOCOL_PATH = PROJECT_ROOT / SEALED_PROTOCOL_RELATIVE_PATH
SEALED_INPUT_RELATIVE_PATHS = (
    "src/jxplanetx/long_horizon/physical_models.py",
    (
        "src/jxplanetx/data/long_horizon_known_system_v2/"
        "horizons_de441_barycentric_icrf_jd2461200p5.csv"
    ),
    (
        "src/jxplanetx/data/long_horizon_known_system_v2/"
        "gm_de440_major_barycenters.csv"
    ),
    (
        "src/jxplanetx/data/long_horizon_known_system_v2/"
        "known_system_primary_relative_ecliptic_float_hex.csv"
    ),
    "src/jxplanetx/data/long_horizon_known_system_v2/MANIFEST.json",
    "src/jxplanetx/restricted_wh_source.py",
    "src/jxplanetx/fast_wisdom_holman.py",
    "src/jxplanetx/engine/wisdom_holman_contracts.py",
    "src/jxplanetx/engine/wisdom_holman.py",
    "src/jxplanetx/engine/trajectory.py",
    "src/jxplanetx/_wisdom_holman_cpu_v3_module.c",
    "src/jxplanetx/_wisdom_holman_loop_v3.c",
    "src/jxplanetx/_wisdom_holman_loop_v2.c",
    "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so",
    "runs/tno_stage1_rebound511_runtime_v1/rebound-5.1.1-cp314-manylinux_x86_64.whl",
)
SEALED_PREPARE_BINDINGS = {
    "benchmarks/jx_tno_1kyr_reduced5_h128_source_preflight_v1.py": (
        47_198,
        "87d1dbb135e95b0880957c83a8d2e1064426bd8b87b3cc4d241c57877b51d57c",
    ),
    "tests/test_tno_1kyr_reduced5_h128_source_preflight_v1.py": (
        17_556,
        "a0cb1a7022a93694a1e765e6c623a9e427df8dd448506d69be0bc0cdb90953d1",
    ),
}
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "runs" / "tno_1kyr_reduced5_h128_source_preflight_v1_result"
)

PROTOCOL_SCHEMA = "jx.tno.1kyr-reduced5-h128-source-preflight.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.1kyr-reduced5-h128-source-preflight.protocol.v1\0"
PROTOCOL_BYTES = 20_740
PROTOCOL_FILE_SHA256 = (
    "e37730efc8cee79fadfc8324d8e33b1a97b9732df92b0cb1162529c439601141"
)
PROTOCOL_CONTENT_SHA256 = (
    "1c275624d21f815c7a2b9c9a8e18f77f63caf9356cf383bd1de3a6ad7228ae66"
)
PREREGISTRATION_COMMIT = "e8cae5a284967d8b752031567002bfbcb4227ec2"
PREREGISTRATION_TREE = "b4ad9ac3b2047584238ff76c29b120cb283ed763"
PREREGISTRATION_TAG = (
    "tno-1kyr-reduced5-h128-source-preflight-v1-preregistered-20261014"
)
PREREGISTRATION_TAG_OBJECT = "e6b0d930eb15c9e640476b5592f2594f3f79f0cc"
PREREGISTRATION_TAG_PAYLOAD_BYTES = 259
PREREGISTRATION_TAG_PAYLOAD_SHA256 = (
    "5e857f9230c4737698f29bcd2ec6819796f89a46f00bd696b13c50666a0af4a2"
)

RESULT_SCHEMA = "jx.tno.1kyr-reduced5-h128-source-preflight.result.v1"
RESULT_DOMAIN = b"jx.tno.1kyr-reduced5-h128-source-preflight.result.v1\0"
RUNTIME_SCHEMA = "jx.tno.1kyr-reduced5-h128-source-preflight.runtime.v1"
RUNTIME_DOMAIN = b"jx.tno.1kyr-reduced5-h128-source-preflight.runtime.v1\0"
SOURCE_TILE_DOMAIN = b"jxplanetx.cuda-restricted-wh.source-tile.v1\0"

SUCCESS_ROSTER = (
    "PROTOCOL.json",
    "SOURCE_GM.bin",
    "JX_H64_SOURCE_NODES.bin",
    "JX_H128_SOURCE_NODES.bin",
    "JX_H64_ALL_NODE_INVARIANTS.bin",
    "JX_H128_ALL_NODE_INVARIANTS.bin",
    "IAS15_ANNUAL_SOURCE_STATES.bin",
    "RUNTIME.json",
    "RESULT.json",
)
EVIDENCE_ROSTER = SUCCESS_ROSTER[:-1]
SOURCE_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
SOURCE_COUNT = 5
SOURCE_MAP_METRIC_NAMES = (
    "minimum_jacobi_periapse",
    "maximum_jacobi_eccentricity",
    "maximum_interaction_force_ratio",
    "maximum_orbit_step_fraction",
    "maximum_periapse_step_fraction",
    "minimum_pair_endpoint",
    "minimum_pair_path_lower_bound",
    "minimum_pair_clearance_after_margin",
    "minimum_secondary_hill_floor_ratio",
    "maximum_barycenter_position_norm",
    "maximum_barycenter_velocity_norm",
    "maximum_translation_force_residual",
    "maximum_kepler_time_residual",
    "maximum_kepler_residual_tolerance",
    "maximum_kepler_lagrange_identity_error",
    "maximum_kepler_energy_error",
    "maximum_kepler_angular_momentum_error",
)
HORIZON_JULIAN_YEARS = 1_000
CHECKPOINT_COUNT = 1_001
LANE_STEPS = (64, 128)
NODE_COUNTS = {64: 64_001, 128: 128_001}
SOURCE_STATE_WIDTH = SOURCE_COUNT * 6
INVARIANT_WIDTH = 4
EXPECTED_BINARY_BYTES = {
    "SOURCE_GM.bin": 40,
    "JX_H64_SOURCE_NODES.bin": 15_360_240,
    "JX_H128_SOURCE_NODES.bin": 30_720_240,
    "JX_H64_ALL_NODE_INVARIANTS.bin": 2_048_032,
    "JX_H128_ALL_NODE_INVARIANTS.bin": 4_096_032,
    "IAS15_ANNUAL_SOURCE_STATES.bin": 240_240,
}
FIXED_BINARY_EVIDENCE_BYTES = 52_464_824
MAXIMUM_JSON_RECEIPT_BYTES = 1 << 20
ARTIFACT_ROLES = {
    "PROTOCOL.json": "exact sealed prepare-only protocol",
    "SOURCE_GM.bin": "source-order gravitational parameters",
    "JX_H64_SOURCE_NODES.bin": "complete retained H64 source nodes",
    "JX_H128_SOURCE_NODES.bin": "complete retained H128 source nodes",
    "JX_H64_ALL_NODE_INVARIANTS.bin": "H64 all-node energy and angular vector",
    "JX_H128_ALL_NODE_INVARIANTS.bin": "H128 all-node energy and angular vector",
    "IAS15_ANNUAL_SOURCE_STATES.bin": "pinned REBOUND 5.1.1 annual source states",
    "RUNTIME.json": "pre-lane-zero pinned CPU runtime receipt",
}

PINNED_PYTHON_BYTES = 7_477_160
PINNED_PYTHON_SHA256 = (
    "52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41"
)
PINNED_PYTHON_VERSION = "3.14.4"
PINNED_PYTHON_RESOLVED_PATH = Path("/usr/bin/python3.14")
PINNED_PYTHON_RESOLVED_PATH_SHA256 = (
    "c03c93ec8cafc307d5de617b0988d1c401bf0aae3f19e8b03174a9bc0f3cb5d7"
)
PINNED_NUMPY_VERSION = "2.5.3"
PINNED_NUMPY_RECORD_BYTES = 111_517
PINNED_NUMPY_RECORD_SHA256 = (
    "9d23d1a798b32c85a52dd64d7f484ef38cc29b84692b05c64c23a3a1aada2ca9"
)
PINNED_NUMPY_INIT_BYTES = 25_220
PINNED_NUMPY_INIT_SHA256 = (
    "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11"
)
NATIVE_EXTENSION_RELATIVE_PATH = (
    "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so"
)
NATIVE_EXTENSION_SHA256 = (
    "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3"
)
NATIVE_EXTENSION_BYTES = 42_520
REBOUND_WHEEL_RELATIVE_PATH = (
    "runs/tno_stage1_rebound511_runtime_v1/"
    "rebound-5.1.1-cp314-manylinux_x86_64.whl"
)
REBOUND_WHEEL_SHA256 = (
    "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19"
)
REBOUND_WHEEL_BYTES = 903_973
REBOUND_INIT_SHA256 = (
    "924e19bfddb22bca9cbff0c641eaf3b2f791287f33df2d8d852f80aac78f8cdd"
)
REBOUND_INIT_BYTES = 3_824
REBOUND_LIBRARY_SHA256 = (
    "52df4fe1050b98d6b0db11f31c18ec14ed00697a987e904bd7587cce4a0ccdef"
)
REBOUND_LIBRARY_BYTES = 2_104_024

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
RUNTIME_PREFIXES = ("numpy", "rebound", "jxplanetx", "cupy")

CLAIMS = {
    "prepare_only_protocol_frozen": True,
    "dynamics_executed": True,
    "source_preflight_executed": True,
    "source_preflight_passed": True,
    "common_epoch_full_covariance_bridge_qualified": False,
    "h128_tracer_route_qualified": False,
    "all12_full129_qualification_passed": False,
    "external_forcing_qualified": False,
    "three_gyr_execution_authorized": False,
    "three_gyr_execution_completed": False,
    "classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}
NONCLAIMS = (
    "NO_TRACER_OR_COVARIANCE_PROPAGATION",
    "NO_COMMON_EPOCH_FULL_COVARIANCE_BRIDGE_QUALIFICATION",
    "NO_GPU_EXECUTION_OR_CPU_GPU_ROUTE_QUALIFICATION",
    "NO_GALACTIC_TIDE_STELLAR_COLLISION_OR_OUTER_BOUNDARY_MODEL",
    "NO_LONG_HORIZON_ACCURACY_OR_STABILITY_CLAIM",
    "NO_TNO_CLASSIFICATION_POPULATION_INFERENCE_OR_PLANET_NINE_EVIDENCE",
    "NO_THREE_GYR_EXECUTION_AUTHORIZATION",
)


class SourcePreflightExecutionError(RuntimeError):
    """The sealed source-preflight successor failed closed."""


@dataclass(frozen=True)
class RuntimeContext:
    np: Any
    rebound: Any
    source_module: Any
    provenance_type: type
    runtime_record: dict[str, Any]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError, OverflowError) as exc:
        raise SourcePreflightExecutionError(
            "record is not strict canonical JSON"
        ) from exc


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise SourcePreflightExecutionError(
                "JSON contains a duplicate or invalid key"
            )
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise SourcePreflightExecutionError(
        f"JSON contains non-finite constant {value}"
    )


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except SourcePreflightExecutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePreflightExecutionError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if type(value) is not dict:
        raise SourcePreflightExecutionError(f"{label} root is not an object")
    return value


def _domain_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(record))


def _regular_bytes(
    path: Path,
    label: str,
    *,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
) -> bytes:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise SourcePreflightExecutionError(f"cannot stat {label}") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or absolute.is_symlink()
        or metadata.st_nlink < 1
    ):
        raise SourcePreflightExecutionError(f"{label} is not a direct regular file")
    if expected_bytes is not None and metadata.st_size != expected_bytes:
        raise SourcePreflightExecutionError(f"{label} byte length changed")
    try:
        payload = absolute.read_bytes()
    except OSError as exc:
        raise SourcePreflightExecutionError(f"cannot read {label}") from exc
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise SourcePreflightExecutionError(f"{label} byte length changed")
    if expected_sha256 is not None and _sha256(payload) != expected_sha256:
        raise SourcePreflightExecutionError(f"{label} SHA-256 changed")
    return payload


def _git(*arguments: str, input_payload: bytes | None = None) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=PROJECT_ROOT,
            input=input_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise SourcePreflightExecutionError("cannot execute git authority check") from exc
    if completed.returncode != 0:
        raise SourcePreflightExecutionError(
            "git authority check failed: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return completed.stdout


def _git_text(*arguments: str) -> str:
    try:
        return _git(*arguments).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise SourcePreflightExecutionError(
            "git authority output is not ASCII"
        ) from exc


def _load_sealed_authority() -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    payload = _regular_bytes(
        SEALED_PROTOCOL_PATH,
        "sealed source-preflight protocol",
        expected_bytes=PROTOCOL_BYTES,
        expected_sha256=PROTOCOL_FILE_SHA256,
    )
    protocol = _strict_json(payload, "sealed source-preflight protocol")
    if payload != _canonical_json(protocol):
        raise SourcePreflightExecutionError(
            "sealed source-preflight protocol is not canonical JSON"
        )
    if (
        protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("content_sha256") != PROTOCOL_CONTENT_SHA256
        or _domain_hash(PROTOCOL_DOMAIN, protocol) != PROTOCOL_CONTENT_SHA256
    ):
        raise SourcePreflightExecutionError(
            "sealed source-preflight protocol content identity changed"
        )
    if (
        _git_text("rev-parse", f"refs/tags/{PREREGISTRATION_TAG}")
        != PREREGISTRATION_TAG_OBJECT
        or _git_text("cat-file", "-t", PREREGISTRATION_TAG_OBJECT) != "tag"
        or _git_text("rev-parse", f"{PREREGISTRATION_TAG}^{{}}")
        != PREREGISTRATION_COMMIT
        or _git_text("rev-parse", f"{PREREGISTRATION_COMMIT}^{{tree}}")
        != PREREGISTRATION_TREE
    ):
        raise SourcePreflightExecutionError(
            "annotated preregistration authority changed"
        )
    tag_payload = _git("cat-file", "tag", PREREGISTRATION_TAG_OBJECT)
    if (
        len(tag_payload) != PREREGISTRATION_TAG_PAYLOAD_BYTES
        or _sha256(tag_payload) != PREREGISTRATION_TAG_PAYLOAD_SHA256
    ):
        raise SourcePreflightExecutionError(
            "annotated preregistration tag payload changed"
        )
    tagged_protocol = _git(
        "show", f"{PREREGISTRATION_COMMIT}:{SEALED_PROTOCOL_RELATIVE_PATH}"
    )
    if tagged_protocol != payload:
        raise SourcePreflightExecutionError(
            "live sealed protocol differs from the preregistration commit"
        )
    implementation = protocol.get("implementation")
    if (
        type(implementation) is not dict
        or set(implementation)
        != {"source", "test", "prepare_imports_numpy_or_dynamics_runtime"}
        or implementation.get("prepare_imports_numpy_or_dynamics_runtime") is not False
    ):
        raise SourcePreflightExecutionError(
            "sealed prepare implementation roster changed"
        )
    for kind, role in (("source", "prepare-only source"), ("test", "prepare-only tests")):
        binding = implementation.get(kind)
        if (
            type(binding) is not dict
            or set(binding) != {"relative_path", "byte_length", "sha256", "role"}
            or binding.get("role") != role
            or binding.get("relative_path") not in SEALED_PREPARE_BINDINGS
            or SEALED_PREPARE_BINDINGS[binding["relative_path"]]
            != (binding.get("byte_length"), binding.get("sha256"))
        ):
            raise SourcePreflightExecutionError(
                "sealed prepare implementation binding changed"
            )
        relative = binding["relative_path"]
        live = _regular_bytes(
            PROJECT_ROOT / relative,
            f"sealed {role}",
            expected_bytes=binding["byte_length"],
            expected_sha256=binding["sha256"],
        )
        if _git("show", f"{PREREGISTRATION_COMMIT}:{relative}") != live:
            raise SourcePreflightExecutionError(
                f"sealed prepare implementation is not preregistered: {relative}"
            )
    bindings = protocol.get("input_bindings")
    if type(bindings) is not list or not bindings:
        raise SourcePreflightExecutionError("sealed input-binding roster is absent")
    observed_binding_paths: list[str] = []
    for binding in bindings:
        if type(binding) is not dict or set(binding) != {
            "relative_path", "byte_length", "sha256", "role"
        }:
            raise SourcePreflightExecutionError("sealed input binding shape changed")
        relative = binding["relative_path"]
        length = binding["byte_length"]
        digest = binding["sha256"]
        role = binding["role"]
        if (
            type(relative) is not str
            or type(length) is not int
            or length <= 0
            or type(digest) is not str
            or len(digest) != 64
            or type(role) is not str
        ):
            raise SourcePreflightExecutionError("sealed input binding is invalid")
        observed_binding_paths.append(relative)
        live = _regular_bytes(
            PROJECT_ROOT / relative,
            f"sealed input {role}",
            expected_bytes=length,
            expected_sha256=digest,
        )
        if _git("show", f"{PREREGISTRATION_COMMIT}:{relative}") != live:
            raise SourcePreflightExecutionError(
                f"sealed input is not the preregistered blob: {relative}"
            )
    if tuple(observed_binding_paths) != SEALED_INPUT_RELATIVE_PATHS:
        raise SourcePreflightExecutionError("sealed input-binding roster changed")
    authority = {
        "protocol_relative_path": SEALED_PROTOCOL_RELATIVE_PATH,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_content_sha256": PROTOCOL_CONTENT_SHA256,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "preregistration_tree": PREREGISTRATION_TREE,
        "preregistration_tag": PREREGISTRATION_TAG,
        "preregistration_tag_object": PREREGISTRATION_TAG_OBJECT,
        "preregistration_tag_payload_bytes": PREREGISTRATION_TAG_PAYLOAD_BYTES,
        "preregistration_tag_payload_sha256": PREREGISTRATION_TAG_PAYLOAD_SHA256,
        "input_binding_count": len(bindings),
        "every_input_binding_verified": True,
    }
    return protocol, payload, authority


def _successor_source_receipt() -> dict[str, Any]:
    payload = _regular_bytes(SCRIPT_PATH, "execution successor source")
    test_payload = _regular_bytes(
        EXECUTOR_TEST_PATH,
        "execution successor test",
    )
    relative = SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix()
    head = _git_text("rev-parse", "HEAD")
    tree = _git_text("rev-parse", "HEAD^{tree}")
    if head == PREREGISTRATION_COMMIT:
        raise SourcePreflightExecutionError(
            "execution implementation is not a successor commit"
        )
    _git("merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, head)
    _verify_successor_preserves_sealed_authority(head)
    test_receipt = _verify_successor_add_only_scope(head, test_payload)
    try:
        tagged = _git("show", f"{head}:{relative}")
    except SourcePreflightExecutionError as exc:
        raise SourcePreflightExecutionError(
            "execution successor must be committed before execution"
        ) from exc
    if tagged != payload:
        raise SourcePreflightExecutionError(
            "execution successor source differs from the current commit"
        )
    return {
        "commit": head,
        "tree": tree,
        "preregistration_ancestor": PREREGISTRATION_COMMIT,
        "relative_path": relative,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "tree_change_policy": "EXACTLY_TWO_ADDED_EXECUTOR_AND_TEST_PATHS",
        **test_receipt,
    }


def _verify_successor_preserves_sealed_authority(successor_commit: str) -> None:
    """Prove the successor commit did not rewrite any frozen authority blob."""

    if not _is_git_sha1(successor_commit) or successor_commit == PREREGISTRATION_COMMIT:
        raise SourcePreflightExecutionError(
            "execution-successor commit identity is invalid"
        )
    relative_paths = (
        SEALED_PROTOCOL_RELATIVE_PATH,
        *tuple(sorted(SEALED_PREPARE_BINDINGS)),
        *tuple(sorted(SEALED_INPUT_RELATIVE_PATHS)),
    )
    for relative in relative_paths:
        try:
            preregistered = _git(
                "show", f"{PREREGISTRATION_COMMIT}:{relative}"
            )
            successor = _git("show", f"{successor_commit}:{relative}")
        except SourcePreflightExecutionError as exc:
            raise SourcePreflightExecutionError(
                "execution successor omits sealed authority blob " + relative
            ) from exc
        if successor != preregistered:
            raise SourcePreflightExecutionError(
                "execution successor changed sealed authority blob " + relative
            )


def _verify_successor_add_only_scope(
    successor_commit: str,
    live_test_payload: bytes | None = None,
) -> dict[str, Any]:
    """Require the cumulative successor tree diff to add only executor + test."""

    if not _is_git_sha1(successor_commit) or successor_commit == PREREGISTRATION_COMMIT:
        raise SourcePreflightExecutionError(
            "execution-successor commit identity is invalid"
        )
    executor_relative = SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix()
    expected = {
        ("A", executor_relative),
        ("A", EXECUTOR_TEST_RELATIVE_PATH),
    }
    try:
        changed = _git(
            "diff-tree",
            "--no-commit-id",
            "-r",
            "--name-status",
            "--no-renames",
            PREREGISTRATION_COMMIT,
            successor_commit,
        ).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourcePreflightExecutionError(
            "execution-successor tree diff is not UTF-8"
        ) from exc
    observed: set[tuple[str, str]] = set()
    for line in changed.splitlines():
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise SourcePreflightExecutionError(
                "execution-successor tree diff shape changed"
            )
        observed.add((fields[0], fields[1]))
    if observed != expected or len(changed.splitlines()) != len(expected):
        raise SourcePreflightExecutionError(
            "execution successor must add only the exact executor and test"
        )
    if live_test_payload is None:
        live_test_payload = _regular_bytes(
            EXECUTOR_TEST_PATH,
            "execution successor test",
        )
    committed_test = _git(
        "show", f"{successor_commit}:{EXECUTOR_TEST_RELATIVE_PATH}"
    )
    if committed_test != live_test_payload:
        raise SourcePreflightExecutionError(
            "execution successor test differs from the current commit"
        )
    return {
        "test_relative_path": EXECUTOR_TEST_RELATIVE_PATH,
        "test_bytes": len(live_test_payload),
        "test_sha256": _sha256(live_test_payload),
    }


def _require_single_thread_environment() -> dict[str, str]:
    receipt: dict[str, str] = {}
    for name in THREAD_ENVIRONMENT:
        value = os.environ.get(name)
        if value != "1":
            raise SourcePreflightExecutionError(
                f"{name} must equal '1' before importing the numerical runtime"
            )
        receipt[name] = value
    return receipt


def _reject_preloaded_runtime() -> None:
    loaded = sorted(
        name
        for name in sys.modules
        if type(name) is str
        and any(name == prefix or name.startswith(prefix + ".") for prefix in RUNTIME_PREFIXES)
    )
    if loaded:
        raise SourcePreflightExecutionError(
            "numerical runtime was imported before custody verification: "
            + ",".join(loaded[:8])
        )


def _file_receipt(path: Path, locator: str) -> dict[str, Any]:
    payload = _regular_bytes(path, locator)
    return {"locator": locator, "bytes": len(payload), "sha256": _sha256(payload)}


def _wheel_members() -> tuple[dict[str, tuple[int, str]], str]:
    wheel_path = PROJECT_ROOT / REBOUND_WHEEL_RELATIVE_PATH
    wheel_payload = _regular_bytes(
        wheel_path,
        "retained REBOUND wheel",
        expected_bytes=REBOUND_WHEEL_BYTES,
        expected_sha256=REBOUND_WHEEL_SHA256,
    )
    members: dict[str, tuple[int, str]] = {}
    library_name = ""
    try:
        with zipfile.ZipFile(wheel_path, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise SourcePreflightExecutionError(
                    "retained REBOUND wheel has duplicate members"
                )
            for name in names:
                path = Path(name)
                if path.is_absolute() or ".." in path.parts or "\\" in name:
                    raise SourcePreflightExecutionError(
                        "retained REBOUND wheel has an unsafe member"
                    )
                if name.endswith("/") or not (
                    (name.startswith("rebound/") and name.endswith(".py"))
                    or (name.startswith("librebound") and name.endswith(".so"))
                ):
                    continue
                payload = archive.read(name)
                members[name] = (len(payload), _sha256(payload))
                if name.startswith("librebound"):
                    if library_name:
                        raise SourcePreflightExecutionError(
                            "retained REBOUND wheel has multiple libraries"
                        )
                    library_name = name
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise SourcePreflightExecutionError(
            "cannot verify retained REBOUND wheel"
        ) from exc
    if (
        members.get("rebound/__init__.py")
        != (REBOUND_INIT_BYTES, REBOUND_INIT_SHA256)
        or not library_name
        or members.get(library_name)
        != (REBOUND_LIBRARY_BYTES, REBOUND_LIBRARY_SHA256)
    ):
        raise SourcePreflightExecutionError(
            "retained REBOUND wheel member identities changed"
        )
    if _sha256(wheel_payload) != REBOUND_WHEEL_SHA256:
        raise SourcePreflightExecutionError("retained REBOUND wheel changed")
    return members, library_name


def _parse_numpy_record_manifest(
    record_payload: bytes,
) -> dict[str, tuple[int, str]]:
    record_manifest: dict[str, tuple[int, str]] = {}
    try:
        rows = csv.reader(record_payload.decode("utf-8").splitlines())
        for row in rows:
            if len(row) != 3 or not row[0]:
                raise SourcePreflightExecutionError("NumPy RECORD row changed")
            relative, digest_field, size_field = row
            # Wheel RECORD may also name scripts installed outside
            # site-packages (for example ../../../bin/f2py).  Retain both
            # the importable package tree and NumPy's wheel-bundled native
            # dependencies because both can enter the loaded process image.
            if not relative.startswith(("numpy/", "numpy.libs/")):
                continue
            if not digest_field or not size_field:
                continue
            algorithm, separator, encoded = digest_field.partition("=")
            if algorithm != "sha256" or separator != "=" or not encoded:
                raise SourcePreflightExecutionError(
                    "NumPy RECORD uses an unsupported digest"
                )
            try:
                digest = base64.urlsafe_b64decode(
                    encoded + "=" * (-len(encoded) % 4)
                ).hex()
                size = int(size_field)
            except (ValueError, TypeError) as exc:
                raise SourcePreflightExecutionError(
                    "NumPy RECORD digest or size is invalid"
                ) from exc
            if (
                relative in record_manifest
                or len(digest) != 64
                or size < 0
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise SourcePreflightExecutionError("NumPy RECORD member changed")
            record_manifest[relative] = (size, digest)
    except UnicodeDecodeError as exc:
        raise SourcePreflightExecutionError("NumPy RECORD is not UTF-8") from exc
    if "numpy/__init__.py" not in record_manifest:
        raise SourcePreflightExecutionError("NumPy RECORD lacks its initializer")
    return record_manifest


def _numpy_distribution_receipt(
    np: Any,
) -> tuple[dict[str, Any], Path, dict[str, tuple[int, str]]]:
    if getattr(np, "__version__", None) != PINNED_NUMPY_VERSION:
        raise SourcePreflightExecutionError("NumPy must be exactly 2.5.3")
    init_path = Path(str(getattr(np, "__file__", ""))).resolve(strict=True)
    init_payload = _regular_bytes(
        init_path,
        "NumPy package initializer",
        expected_bytes=PINNED_NUMPY_INIT_BYTES,
        expected_sha256=PINNED_NUMPY_INIT_SHA256,
    )
    candidates = []
    for distribution in importlib.metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name == "numpy" and distribution.version == PINNED_NUMPY_VERSION:
            candidates.append(distribution)
    owners = []
    for distribution in candidates:
        root = Path(distribution.locate_file("")).resolve()
        try:
            init_path.relative_to(root)
        except ValueError:
            continue
        owners.append((distribution, root))
    if len(owners) != 1:
        raise SourcePreflightExecutionError(
            "exactly one pinned NumPy distribution must own the imported module"
        )
    distribution, root = owners[0]
    dist_path = Path(str(getattr(distribution, "_path", ""))).resolve(strict=True)
    record_path = dist_path / "RECORD"
    record_payload = _regular_bytes(
        record_path,
        "NumPy RECORD",
        expected_bytes=PINNED_NUMPY_RECORD_BYTES,
        expected_sha256=PINNED_NUMPY_RECORD_SHA256,
    )
    record_manifest = _parse_numpy_record_manifest(record_payload)
    return (
        {
            "version": PINNED_NUMPY_VERSION,
            "initializer": {
                "locator": "NUMPY_DISTRIBUTION:numpy/__init__.py",
                "bytes": len(init_payload),
                "sha256": _sha256(init_payload),
            },
            "record": {
                "locator": f"NUMPY_DISTRIBUTION:{dist_path.name}/RECORD",
                "bytes": len(record_payload),
                "sha256": _sha256(record_payload),
            },
            "record_base64": base64.b64encode(record_payload).decode("ascii"),
        },
        root,
        record_manifest,
    )


def _module_locator(
    path: Path,
    *,
    numpy_root: Path,
    rebound_root: Path,
) -> str:
    absolute = path.resolve(strict=True)
    for prefix, root in (
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("NUMPY_DISTRIBUTION", numpy_root),
        ("REBOUND_WHEEL", rebound_root),
    ):
        try:
            relative = absolute.relative_to(root.resolve())
        except ValueError:
            continue
        return f"{prefix}:{relative.as_posix()}"
    return f"SYSTEM_RUNTIME:{absolute.name}"


def _module_manifest(
    *,
    numpy_root: Path,
    rebound_root: Path,
    numpy_record_manifest: Mapping[str, tuple[int, str]],
    rebound_wheel_members: Mapping[str, tuple[int, str]],
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for name in sorted(sys.modules):
        if not any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in ("numpy", "rebound", "jxplanetx")
        ):
            continue
        module = sys.modules.get(name)
        if type(module) is not types.ModuleType:
            continue
        file_value = getattr(module, "__file__", None)
        if type(file_value) is not str:
            continue
        path = Path(file_value)
        payload = _regular_bytes(path, f"loaded runtime module {name}")
        locator = _module_locator(
            path, numpy_root=numpy_root, rebound_root=rebound_root
        )
        if name == "numpy" or name.startswith("numpy."):
            if not locator.startswith("NUMPY_DISTRIBUTION:"):
                raise SourcePreflightExecutionError(
                    f"loaded NumPy module is outside its pinned distribution: {name}"
                )
            relative = locator.removeprefix("NUMPY_DISTRIBUTION:")
            expected = numpy_record_manifest.get(relative)
            if expected != (len(payload), _sha256(payload)):
                raise SourcePreflightExecutionError(
                    f"loaded NumPy module differs from its pinned RECORD: {name}"
                )
        elif name == "rebound" or name.startswith("rebound."):
            if not locator.startswith("REBOUND_WHEEL:"):
                raise SourcePreflightExecutionError(
                    f"loaded REBOUND module is outside its retained wheel: {name}"
                )
            relative = locator.removeprefix("REBOUND_WHEEL:")
            expected = rebound_wheel_members.get(relative)
            if expected != (len(payload), _sha256(payload)):
                raise SourcePreflightExecutionError(
                    f"loaded REBOUND module differs from its retained wheel: {name}"
                )
        elif name == "jxplanetx" or name.startswith("jxplanetx."):
            if not locator.startswith("PROJECT_ROOT:"):
                raise SourcePreflightExecutionError(
                    f"loaded JX module is outside the execution checkout: {name}"
                )
            relative = locator.removeprefix("PROJECT_ROOT:")
            if _git("show", f"HEAD:{relative}") != payload:
                raise SourcePreflightExecutionError(
                    f"loaded JX module differs from the execution commit: {name}"
                )
        manifest.append(
            {
                "module": name,
                "locator": locator,
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    required = {
        "numpy",
        "numpy._core._multiarray_umath",
        "rebound",
        "rebound.simulation",
        "jxplanetx.restricted_wh_source",
        "jxplanetx.fast_wisdom_holman",
        "jxplanetx._wisdom_holman_cpu_v3",
    }
    observed = {item["module"] for item in manifest}
    if not required.issubset(observed):
        raise SourcePreflightExecutionError(
            "required loaded runtime module is absent from the manifest"
        )
    if any(name == "cupy" or name.startswith("cupy.") for name in sys.modules):
        raise SourcePreflightExecutionError(
            "GPU runtime import is forbidden in the CPU-only source preflight"
        )
    return manifest


def _shared_library_manifest() -> list[dict[str, Any]]:
    maps_path = Path("/proc/self/maps")
    payload = _regular_bytes(maps_path, "process map")
    paths: set[Path] = set()
    for raw_line in payload.decode("utf-8", "strict").splitlines():
        fields = raw_line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        raw_path = fields[5]
        if raw_path.startswith("/") and ".so" in Path(raw_path).name:
            paths.add(Path(raw_path))
    manifest = []
    seen_locator: set[tuple[str, str]] = set()
    for path in sorted(paths, key=lambda item: os.fspath(item)):
        try:
            data = _regular_bytes(path, "loaded shared library")
        except SourcePreflightExecutionError:
            raise
        key = (path.name, _sha256(data))
        if key in seen_locator:
            continue
        seen_locator.add(key)
        manifest.append(
            {
                "locator": f"LOADED_SHARED_LIBRARY:{path.name}",
                "bytes": len(data),
                "sha256": key[1],
            }
        )
    if not manifest:
        raise SourcePreflightExecutionError("loaded shared-library manifest is empty")
    return sorted(
        manifest,
        key=lambda item: (item["locator"], item["sha256"]),
    )


def _validate_numpy_bundled_shared_libraries(
    libraries: list[dict[str, Any]],
    numpy_record_manifest: Mapping[str, tuple[int, str]],
) -> None:
    """Bind every wheel-bundled NumPy shared object to the pinned RECORD."""

    expected: dict[str, tuple[int, str]] = {}
    for relative, identity in numpy_record_manifest.items():
        if not relative.startswith("numpy.libs/"):
            continue
        basename = Path(relative).name
        if ".so" not in basename:
            continue
        locator = f"LOADED_SHARED_LIBRARY:{basename}"
        if locator in expected:
            raise SourcePreflightExecutionError(
                "NumPy RECORD has ambiguous bundled shared-library names"
            )
        expected[locator] = identity
    if not expected:
        raise SourcePreflightExecutionError(
            "NumPy RECORD lacks bundled shared-library identities"
        )
    observed = {
        item["locator"]: (item["bytes"], item["sha256"])
        for item in libraries
    }
    for locator, identity in expected.items():
        if observed.get(locator) != identity:
            raise SourcePreflightExecutionError(
                "NumPy bundled shared library differs from its pinned RECORD"
            )


def _runtime_preflight(authority: Mapping[str, Any]) -> RuntimeContext:
    thread_environment = _require_single_thread_environment()
    _reject_preloaded_runtime()
    executable = Path(sys.executable).resolve(strict=True)
    executable_payload = _regular_bytes(
        executable,
        "Python executable",
        expected_bytes=PINNED_PYTHON_BYTES,
        expected_sha256=PINNED_PYTHON_SHA256,
    )
    if (
        platform.python_version() != PINNED_PYTHON_VERSION
        or executable != PINNED_PYTHON_RESOLVED_PATH
    ):
        raise SourcePreflightExecutionError("Python must be exactly 3.14.4")
    members, library_name = _wheel_members()
    try:
        np = importlib.import_module("numpy")
        importlib.import_module("numpy._core._multiarray_umath")
        source_module = importlib.import_module("jxplanetx.restricted_wh_source")
        importlib.import_module("jxplanetx.fast_wisdom_holman")
        native_module = importlib.import_module("jxplanetx._wisdom_holman_cpu_v3")
        contracts_module = importlib.import_module("jxplanetx.engine.contracts")
        rebound = importlib.import_module("rebound")
        importlib.import_module("rebound.simulation")
    except Exception as exc:
        raise SourcePreflightExecutionError(
            "pinned CPU numerical runtime is unavailable"
        ) from exc
    numpy_receipt, numpy_root, numpy_record_manifest = _numpy_distribution_receipt(np)
    native_path = Path(str(getattr(native_module, "__file__", ""))).resolve(strict=True)
    native_payload = _regular_bytes(
        native_path,
        "compiled ordered-Jacobi extension",
        expected_bytes=NATIVE_EXTENSION_BYTES,
        expected_sha256=NATIVE_EXTENSION_SHA256,
    )
    if native_path != (PROJECT_ROOT / NATIVE_EXTENSION_RELATIVE_PATH).resolve(strict=True):
        raise SourcePreflightExecutionError(
            "compiled ordered-Jacobi extension was imported from the wrong source"
        )
    if getattr(rebound, "__version__", None) != "5.1.1":
        raise SourcePreflightExecutionError("REBOUND must be exactly 5.1.1")
    rebound_init = Path(str(getattr(rebound, "__file__", ""))).resolve(strict=True)
    if rebound_init.name != "__init__.py" or rebound_init.parent.name != "rebound":
        raise SourcePreflightExecutionError("imported REBOUND package layout changed")
    rebound_root = rebound_init.parent.parent
    init_payload = _regular_bytes(
        rebound_init,
        "imported REBOUND initializer",
        expected_bytes=members["rebound/__init__.py"][0],
        expected_sha256=REBOUND_INIT_SHA256,
    )
    library_value = getattr(rebound, "__libpath__", None)
    if type(library_value) is not str:
        library_value = getattr(getattr(rebound, "clibrebound", None), "_name", None)
    if type(library_value) is not str:
        raise SourcePreflightExecutionError("REBOUND library locator is absent")
    library_path = Path(library_value).resolve(strict=True)
    if library_path.name != library_name or library_path.parent != rebound_root:
        raise SourcePreflightExecutionError(
            "imported REBOUND library is not the retained wheel member"
        )
    library_payload = _regular_bytes(
        library_path,
        "imported REBOUND library",
        expected_bytes=members[library_name][0],
        expected_sha256=REBOUND_LIBRARY_SHA256,
    )
    module_manifest = _module_manifest(
        numpy_root=numpy_root,
        rebound_root=rebound_root,
        numpy_record_manifest=numpy_record_manifest,
        rebound_wheel_members=members,
    )
    successor = _successor_source_receipt()
    record: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "authority": dict(authority),
        "successor_implementation": successor,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "resolved_path_sha256": _sha256(os.fsencode(executable)),
            "executable": {
                "locator": "PINNED_PYTHON_EXECUTABLE",
                "bytes": len(executable_payload),
                "sha256": _sha256(executable_payload),
            },
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
            "float_mant_dig": sys.float_info.mant_dig,
            "float_radix": sys.float_info.radix,
        },
        "thread_environment": thread_environment,
        "numpy_distribution": numpy_receipt,
        "native_extension": {
            "locator": f"PROJECT_ROOT:{NATIVE_EXTENSION_RELATIVE_PATH}",
            "bytes": len(native_payload),
            "sha256": _sha256(native_payload),
        },
        "rebound_distribution": {
            "version": "5.1.1",
            "wheel": {
                "locator": f"PROJECT_ROOT:{REBOUND_WHEEL_RELATIVE_PATH}",
                "bytes": REBOUND_WHEEL_BYTES,
                "sha256": REBOUND_WHEEL_SHA256,
            },
            "initializer": {
                "locator": "REBOUND_WHEEL:rebound/__init__.py",
                "bytes": len(init_payload),
                "sha256": _sha256(init_payload),
            },
            "library": {
                "locator": f"REBOUND_WHEEL:{library_name}",
                "bytes": len(library_payload),
                "sha256": _sha256(library_payload),
            },
        },
        "loaded_module_manifest": module_manifest,
        "loaded_shared_library_manifest": _shared_library_manifest(),
        "gpu_runtime_imported": False,
        "cuda_device_queried": False,
        "dynamics_executed_at_receipt_time": False,
    }
    record["content_sha256"] = _domain_hash(RUNTIME_DOMAIN, record)
    provenance_type = getattr(contracts_module, "Provenance", None)
    if type(provenance_type) is not type:
        raise SourcePreflightExecutionError("JX Provenance type is unavailable")
    _validate_runtime_record(record, authority)
    return RuntimeContext(np, rebound, source_module, provenance_type, record)


def _float_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise SourcePreflightExecutionError(f"{label} is not float-hex text")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise SourcePreflightExecutionError(f"{label} is not valid float-hex") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise SourcePreflightExecutionError(
            f"{label} is not canonical finite binary64"
        )
    return result


def _validate_source_map_metrics_hex(metrics: object) -> tuple[float, ...]:
    """Validate the pinned provider's complete, ordered metric receipt."""

    if type(metrics) is not list or len(metrics) != len(SOURCE_MAP_METRIC_NAMES):
        raise SourcePreflightExecutionError("JX source-map metric roster changed")
    values: list[float] = []
    for expected_name, metric in zip(SOURCE_MAP_METRIC_NAMES, metrics, strict=True):
        if (
            type(metric) is not dict
            or set(metric) != {"name", "value_hex"}
            or metric.get("name") != expected_name
        ):
            raise SourcePreflightExecutionError("JX source-map metric changed")
        value = _float_hex(metric.get("value_hex"), "JX source-map metric")
        if value < 0.0:
            raise SourcePreflightExecutionError("JX source-map metric is negative")
        values.append(value)
    for name in (
        "minimum_jacobi_periapse",
        "minimum_pair_endpoint",
        "minimum_pair_path_lower_bound",
        "minimum_pair_clearance_after_margin",
    ):
        if values[SOURCE_MAP_METRIC_NAMES.index(name)] <= 0.0:
            raise SourcePreflightExecutionError(
                "JX source-map minimum metric is not positive"
            )
    if (
        values[SOURCE_MAP_METRIC_NAMES.index("maximum_kepler_time_residual")]
        > values[
            SOURCE_MAP_METRIC_NAMES.index("maximum_kepler_residual_tolerance")
        ]
    ):
        raise SourcePreflightExecutionError(
            "JX Kepler residual exceeds its certified tolerance"
        )
    return tuple(values)


def _initial_arrays(np: Any, protocol: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    state = protocol.get("initial_state")
    if type(state) is not dict or state.get("source_ids") != list(SOURCE_IDS):
        raise SourcePreflightExecutionError("sealed initial-state roster changed")
    gm_values = state.get("gm_AU3_per_julian_year2_hex")
    position_values = state.get("primary_relative_positions_AU_hex")
    velocity_values = state.get("primary_relative_velocities_AU_per_julian_year_hex")
    if (
        type(gm_values) is not list
        or len(gm_values) != SOURCE_COUNT
        or type(position_values) is not list
        or len(position_values) != SOURCE_COUNT
        or type(velocity_values) is not list
        or len(velocity_values) != SOURCE_COUNT
    ):
        raise SourcePreflightExecutionError("sealed initial-state geometry changed")
    gm = np.array(
        [_float_hex(value, "source GM") for value in gm_values],
        dtype=np.float64,
        order="C",
        copy=True,
    )
    positions = np.array(
        [
            [_float_hex(value, "source position") for value in row]
            for row in position_values
        ],
        dtype=np.float64,
        order="C",
        copy=True,
    )
    velocities = np.array(
        [
            [_float_hex(value, "source velocity") for value in row]
            for row in velocity_values
        ],
        dtype=np.float64,
        order="C",
        copy=True,
    )
    if (
        gm.shape != (SOURCE_COUNT,)
        or positions.shape != (SOURCE_COUNT, 3)
        or velocities.shape != (SOURCE_COUNT, 3)
        or not gm.flags.owndata
        or not positions.flags.owndata
        or not velocities.flags.owndata
        or not bool(np.all(np.isfinite(gm)))
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(velocities)))
        or not bool(np.all(gm > 0.0))
        or not bool(np.all(positions[0] == 0.0))
        or not bool(np.all(velocities[0] == 0.0))
    ):
        raise SourcePreflightExecutionError("sealed initial-state arrays are invalid")
    return gm, positions, velocities


def _hex_record(value: object) -> object:
    """Convert a runtime receipt to canonical JSON without decimal floats."""

    if type(value) is float:
        if not math.isfinite(value):
            raise SourcePreflightExecutionError("runtime receipt contains nonfinite data")
        return value.hex()
    if type(value) in (str, int, bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(key): _hex_record(item) for key, item in value.items()}
    if type(value) in (list, tuple):
        return [_hex_record(item) for item in value]
    raise SourcePreflightExecutionError(
        f"runtime receipt contains unsupported type {type(value).__name__}"
    )


def _path_free_source_runtime_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SourcePreflightExecutionError(
            "JX source runtime identity is not a mapping"
        )
    normalized = _hex_record(dict(value))
    if type(normalized) is not dict:
        raise SourcePreflightExecutionError(
            "JX source runtime identity changed"
        )
    source_map = normalized.get("source_map_runtime_identity")
    extension = source_map.get("extension_binary") if type(source_map) is dict else None
    if (
        type(extension) is not dict
        or set(extension) != {"path", "size_bytes", "sha256"}
        or type(extension.get("path")) is not str
    ):
        raise SourcePreflightExecutionError(
            "JX native runtime path receipt changed"
        )
    try:
        observed_path = Path(extension["path"]).resolve(strict=True)
        expected_path = (
            PROJECT_ROOT / NATIVE_EXTENSION_RELATIVE_PATH
        ).resolve(strict=True)
    except OSError as exc:
        raise SourcePreflightExecutionError(
            "JX native runtime path cannot be resolved"
        ) from exc
    if (
        observed_path != expected_path
        or extension.get("size_bytes") != NATIVE_EXTENSION_BYTES
        or extension.get("sha256") != NATIVE_EXTENSION_SHA256
    ):
        raise SourcePreflightExecutionError(
            "JX native runtime binary identity changed"
        )
    source_map["extension_binary"] = {
        "locator": f"PROJECT_ROOT:{NATIVE_EXTENSION_RELATIVE_PATH}",
        "size_bytes": NATIVE_EXTENSION_BYTES,
        "sha256": NATIVE_EXTENSION_SHA256,
    }

    def reject_absolute_path_strings(item: object) -> None:
        if type(item) is str:
            if item.startswith("/"):
                raise SourcePreflightExecutionError(
                    "JX runtime receipt retained an absolute workstation path"
                )
            return
        if type(item) is dict:
            for key, nested in item.items():
                reject_absolute_path_strings(key)
                reject_absolute_path_strings(nested)
        elif type(item) is list:
            for nested in item:
                reject_absolute_path_strings(nested)

    reject_absolute_path_strings(normalized)
    return normalized


def _run_jx_lane(
    context: RuntimeContext,
    protocol: Mapping[str, Any],
    steps_per_year: int,
) -> tuple[Any, dict[str, Any]]:
    if steps_per_year not in LANE_STEPS:
        raise SourcePreflightExecutionError("JX lane step count is not sealed")
    np = context.np
    execution = protocol.get("execution_specification")
    if type(execution) is not dict:
        raise SourcePreflightExecutionError("sealed execution specification changed")
    lanes = execution.get("jx_lanes")
    if type(lanes) is not list or len(lanes) != 2:
        raise SourcePreflightExecutionError("sealed JX lane roster changed")
    selected = [lane for lane in lanes if lane.get("steps_per_julian_year") == steps_per_year]
    if len(selected) != 1:
        raise SourcePreflightExecutionError("sealed JX lane identity changed")
    lane = selected[0]
    schedule = lane.get("schedule")
    if type(schedule) is not dict:
        raise SourcePreflightExecutionError("sealed JX source schedule changed")
    gm, raw_positions, raw_velocities = _initial_arrays(np, protocol)
    total_gm = math.fsum(float(value) for value in gm)
    acceleration_bound = math.nextafter(2.0 * total_gm, math.inf)
    if not math.isfinite(acceleration_bound) or acceleration_bound <= 0.0:
        raise SourcePreflightExecutionError(
            "source acceleration-bound certificate is invalid"
        )
    radii = np.ones(SOURCE_COUNT, dtype=np.float64)
    bounds = np.full(SOURCE_COUNT, acceleration_bound, dtype=np.float64)
    provenance = context.provenance_type(
        source_id=f"jx-tno-1kyr-reduced5-source-h{steps_per_year}",
        citation=(
            "Sealed common-epoch reduced-five source preflight protocol "
            + PROTOCOL_CONTENT_SHA256
        ),
        version="source-preflight-executor-v1",
        sha256=str(schedule.get("content_sha256")),
    )
    spec_type = getattr(context.source_module, "WisdomHolmanSourceTileSpec", None)
    builder = getattr(context.source_module, "build_wisdom_holman_source_tile", None)
    if type(spec_type) is not type or not callable(builder):
        raise SourcePreflightExecutionError("sealed JX source builder is unavailable")
    spec = spec_type(
        source_step_years=1.0 / steps_per_year,
        source_step_count=HORIZON_JULIAN_YEARS * steps_per_year,
        minimum_encounter_pair_separation_AU=1.0,
        minimum_jacobi_periapse_AU=0.05,
        maximum_initial_barycenter_position_norm_AU=1.0e-12,
        maximum_initial_barycenter_velocity_norm_AU_per_year=1.0e-12,
    )
    def build_once() -> Any:
        # Every call receives separately owned arrays.  The public builder also
        # performs its own full internal bitwise replay, so the two calls below
        # provide both the sealed fresh-workspace replay and implementation's
        # internal primary/replay certificate.
        try:
            return builder(
                tile_id=f"tno-1kyr-reduced5-source-h{steps_per_year}",
                epoch_origin_years=_float_hex(
                    execution["epoch_origin_julian_years_since_j2000_hex"],
                    "source epoch origin",
                ),
                global_source_step=0,
                time_scale="TDB",
                frame=str(protocol["initial_state"]["frame"]),
                axes="IAU76_80_ECLIPTIC_J2000",
                source_ids=SOURCE_IDS,
                primary_source_id=SOURCE_IDS[0],
                source_schedule_sha256=str(schedule["content_sha256"]),
                initial_source_relative_positions_AU=raw_positions.copy(order="C"),
                initial_source_relative_velocities_AU_per_year=raw_velocities.copy(order="C"),
                source_gm_AU3_per_year2=gm.copy(order="C"),
                source_encounter_radii_AU=radii.copy(order="C"),
                certified_relative_acceleration_bounds_AU_per_year2=bounds.copy(order="C"),
                provenance=provenance,
                spec=spec,
            )
        except Exception as exc:
            raise SourcePreflightExecutionError(
                f"JX H{steps_per_year} source lane failed closed"
            ) from exc

    build = build_once()
    fresh_replay = build_once()
    post_init = getattr(build, "__post_init__", None)
    if callable(post_init):
        post_init()
    tile = getattr(build, "tile", None)
    positions = getattr(tile, "source_relative_positions_AU", None)
    velocities = getattr(tile, "source_relative_velocities_AU_per_year", None)
    observed_gm = getattr(tile, "source_gm_AU3_per_year2", None)
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or type(observed_gm) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or observed_gm.dtype != np.dtype(np.float64)
        or positions.shape != (NODE_COUNTS[steps_per_year], SOURCE_COUNT, 3)
        or velocities.shape != positions.shape
        or observed_gm.shape != (SOURCE_COUNT,)
        or not bool(np.all(np.isfinite(positions)))
        or not bool(np.all(np.isfinite(velocities)))
        or not bool(np.array_equal(observed_gm.view(np.uint64), gm.view(np.uint64)))
        or not bool(np.all(positions[:, 0, :] == 0.0))
        or not bool(np.all(velocities[:, 0, :] == 0.0))
    ):
        raise SourcePreflightExecutionError(
            f"JX H{steps_per_year} retained source geometry changed"
        )
    runtime = getattr(build, "runtime_identity", None)
    canonicalization = getattr(build, "coordinate_canonicalization", None)
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("exact_replay_required") is not True
        or canonicalization is None
        or getattr(canonicalization, "canonicalization_idempotent", None) is not True
        or getattr(canonicalization, "tile_node_zero_exact", None) is not True
        or getattr(canonicalization, "source_tile_sha256", None)
        != getattr(tile, "content_sha256", None)
    ):
        raise SourcePreflightExecutionError(
            f"JX H{steps_per_year} exact replay/canonicalization receipt changed"
        )
    metrics = getattr(build, "source_map_metrics", None)
    if (
        type(metrics) is not tuple
        or tuple(item[0] for item in metrics if type(item) is tuple and item)
        != SOURCE_MAP_METRIC_NAMES
        or any(
            type(item) is not tuple
            or len(item) != 2
            or type(item[0]) is not str
            or type(item[1]) is not float
            or not math.isfinite(item[1])
            or item[1] < 0.0
            for item in metrics
        )
    ):
        raise SourcePreflightExecutionError(
            f"JX H{steps_per_year} map metrics changed"
        )
    replay_tile = getattr(fresh_replay, "tile", None)
    replay_runtime_identity = _path_free_source_runtime_identity(
        fresh_replay.runtime_identity
    )
    primary_runtime_identity = _path_free_source_runtime_identity(runtime)
    replay_non_timing = {
        "tile_content_sha256": getattr(replay_tile, "content_sha256", None),
        "input_content_sha256": getattr(
            fresh_replay, "input_content_sha256", None
        ),
        "source_map_input_content_sha256": getattr(
            fresh_replay, "source_map_input_content_sha256", None
        ),
        "source_map_result_content_sha256": getattr(
            fresh_replay, "source_map_result_content_sha256", None
        ),
        "source_map_verification_content_sha256": getattr(
            fresh_replay, "source_map_verification_content_sha256", None
        ),
        "source_map_metrics": _hex_record(
            getattr(fresh_replay, "source_map_metrics", None)
        ),
        "coordinate_canonicalization": _hex_record(
            dict(fresh_replay.coordinate_canonicalization.as_record())
        ),
        "runtime_identity": replay_runtime_identity,
    }
    primary_non_timing = {
        "tile_content_sha256": getattr(tile, "content_sha256", None),
        "input_content_sha256": build.input_content_sha256,
        "source_map_input_content_sha256": build.source_map_input_content_sha256,
        "source_map_result_content_sha256": build.source_map_result_content_sha256,
        "source_map_verification_content_sha256": (
            build.source_map_verification_content_sha256
        ),
        "source_map_metrics": _hex_record(metrics),
        "coordinate_canonicalization": _hex_record(
            dict(canonicalization.as_record())
        ),
        "runtime_identity": primary_runtime_identity,
    }
    if (
        _canonical_json(primary_non_timing) != _canonical_json(replay_non_timing)
        or not _array_bitwise_equal(
            np,
            positions,
            getattr(replay_tile, "source_relative_positions_AU", None),
        )
        or not _array_bitwise_equal(
            np,
            velocities,
            getattr(replay_tile, "source_relative_velocities_AU_per_year", None),
        )
        or not _array_bitwise_equal(
            np,
            observed_gm,
            getattr(replay_tile, "source_gm_AU3_per_year2", None),
        )
    ):
        raise SourcePreflightExecutionError(
            f"JX H{steps_per_year} fresh-workspace replay differs bitwise"
        )
    replay_receipt_domain = (
        b"jx.tno.1kyr-reduced5-h128-source-preflight.fresh-workspace-replay.v1\0"
    )
    primary_fresh_receipt_sha256 = _sha256(
        replay_receipt_domain + _canonical_json(primary_non_timing)
    )
    replay_fresh_receipt_sha256 = _sha256(
        replay_receipt_domain + _canonical_json(replay_non_timing)
    )
    canonical_record = _hex_record(dict(canonicalization.as_record()))
    receipt = {
        "lane_id": f"jx_ordered_jacobi_h{steps_per_year}",
        "engine": "JX_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN",
        "steps_per_julian_year": steps_per_year,
        "fixed_step_julian_years_hex": (1.0 / steps_per_year).hex(),
        "source_step_count": HORIZON_JULIAN_YEARS * steps_per_year,
        "source_node_count": NODE_COUNTS[steps_per_year],
        "schedule_content_sha256": schedule["content_sha256"],
        "tile_content_sha256": tile.content_sha256,
        "raw_relative_input_sha256": build.input_content_sha256,
        "source_map_input_content_sha256": build.source_map_input_content_sha256,
        "source_map_result_content_sha256": build.source_map_result_content_sha256,
        "source_map_verification_content_sha256": (
            build.source_map_verification_content_sha256
        ),
        "source_map_metrics_hex": [
            {"name": name, "value_hex": value.hex()} for name, value in metrics
        ],
        "coordinate_canonicalization": canonical_record,
        "coordinate_canonicalization_content_sha256": (
            canonicalization.content_sha256
        ),
        "acceleration_bound_certificate": {
            "formula": "NEXTUP_OF_2_TIMES_TOTAL_REDUCED_GM_DIVIDED_BY_ONE_AU_SQUARED",
            "value_in_source_order_hex": [acceleration_bound.hex()] * SOURCE_COUNT,
        },
        "exact_replay_verified": True,
        "fresh_workspace_replay_verified": True,
        "fresh_workspace_receipt_domain": (
            "jx.tno.1kyr-reduced5-h128-source-preflight."
            "fresh-workspace-replay.v1"
        ),
        "fresh_workspace_primary_receipt": primary_non_timing,
        "fresh_workspace_replay_receipt": replay_non_timing,
        "fresh_workspace_primary_receipt_sha256": primary_fresh_receipt_sha256,
        "fresh_workspace_replay_receipt_sha256": replay_fresh_receipt_sha256,
        "public_builder_call_count": 2,
        "internal_primary_execution_count": 2,
        "internal_replay_execution_count": 2,
        "total_native_map_execution_count": 4,
        "retained_stream": (
            "FIRST_FRESH_WORKSPACE_PRIMARY_AFTER_INTERNAL_AND_OUTER_BITWISE_REPLAY"
        ),
    }
    return build, receipt


def _array_bitwise_equal(np: Any, left: Any, right: Any) -> bool:
    return bool(
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.shape == right.shape
        and left.dtype == np.dtype(np.float64)
        and right.dtype == np.dtype(np.float64)
        and np.array_equal(left.view(np.uint64), right.view(np.uint64))
    )


def _bind_rebound_heartbeat(
    simulation: Any,
    rebound: Any,
    heartbeat: Any,
) -> Any:
    simulation_module = getattr(rebound, "simulation", None)
    callback_factory = getattr(simulation_module, "AFF", None)
    if not callable(callback_factory):
        raise SourcePreflightExecutionError(
            "pinned REBOUND heartbeat callback factory is unavailable"
        )
    try:
        callback = callback_factory(heartbeat)
        simulation._heartbeat = callback
    except BaseException as exc:
        raise SourcePreflightExecutionError(
            "pinned REBOUND C heartbeat binding failed"
        ) from exc
    if not callable(getattr(simulation, "_heartbeat", None)):
        raise SourcePreflightExecutionError(
            "pinned REBOUND C heartbeat field is not callable after binding"
        )
    return callback


def _rebound_state(np: Any, simulation: Any) -> tuple[Any, Any]:
    if int(simulation.N) != SOURCE_COUNT or int(simulation.N_active) != SOURCE_COUNT:
        raise SourcePreflightExecutionError("IAS15 source roster changed")
    positions = np.empty((SOURCE_COUNT, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    for index in range(SOURCE_COUNT):
        particle = simulation.particles[index]
        positions[index] = (float(particle.x), float(particle.y), float(particle.z))
        velocities[index] = (float(particle.vx), float(particle.vy), float(particle.vz))
    if not bool(np.all(np.isfinite(positions))) or not bool(
        np.all(np.isfinite(velocities))
    ):
        raise SourcePreflightExecutionError("IAS15 produced nonfinite source state")
    return positions, velocities


def _rebound_additional_forces_pointer_is_null(simulation: Any) -> bool:
    sentinel = object()
    pointer = getattr(simulation, "_additional_forces", sentinel)
    if pointer is sentinel:
        raise SourcePreflightExecutionError(
            "pinned REBOUND additional-forces C field is absent"
        )
    try:
        is_null = not bool(pointer)
    except BaseException as exc:
        raise SourcePreflightExecutionError(
            "cannot inspect pinned REBOUND additional-forces C field"
        ) from exc
    return is_null


def _run_ias15(
    context: RuntimeContext,
    canonical_positions: Any,
    canonical_velocities: Any,
    gm: Any,
) -> tuple[Any, dict[str, Any]]:
    np = context.np
    rebound = context.rebound
    if (
        not _array_bitwise_equal(np, canonical_positions, canonical_positions.copy())
        or not _array_bitwise_equal(np, canonical_velocities, canonical_velocities.copy())
        or type(gm) is not np.ndarray
        or gm.shape != (SOURCE_COUNT,)
    ):
        raise SourcePreflightExecutionError("canonical IAS15 start state is invalid")
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "ias15"
    simulation.dt = 1.0 / 32.0
    simulation.integrator.epsilon = 1.0e-13
    simulation.integrator.min_dt = 0.0
    simulation.integrator.adaptive_mode = "PRS23"
    for index in range(SOURCE_COUNT):
        position = canonical_positions[index]
        velocity = canonical_velocities[index]
        simulation.add(
            m=float(gm[index]),
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            vx=float(velocity[0]),
            vy=float(velocity[1]),
            vz=float(velocity[2]),
        )
    simulation.N_active = SOURCE_COUNT
    simulation.testparticle_type = 0
    initial_configuration = {
        "integrator": str(simulation.integrator),
        "epsilon_hex": float(simulation.integrator.epsilon).hex(),
        "minimum_dt_julian_years_hex": float(simulation.integrator.min_dt).hex(),
        "adaptive_mode": str(simulation.integrator.adaptive_mode),
        "initial_dt_julian_years_hex": float(simulation.dt).hex(),
        "initial_time_julian_years_hex": float(simulation.t).hex(),
        "G_hex": float(simulation.G).hex(),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "softening_hex": float(simulation.softening).hex(),
        "additional_forces_c_pointer_is_null": (
            _rebound_additional_forces_pointer_is_null(simulation)
        ),
        "force_is_velocity_dependent": int(
            simulation.force_is_velocity_dependent
        ),
        "exit_min_distance_AU_hex": float(simulation.exit_min_distance).hex(),
        "exit_max_distance_AU_hex": float(simulation.exit_max_distance).hex(),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "testparticle_type": int(simulation.testparticle_type),
    }
    expected_initial = {
        "integrator": "ias15",
        "epsilon_hex": (1.0e-13).hex(),
        "minimum_dt_julian_years_hex": "0x0.0p+0",
        "adaptive_mode": "PRS23",
        "initial_dt_julian_years_hex": (1.0 / 32.0).hex(),
        "initial_time_julian_years_hex": "0x0.0p+0",
        "G_hex": "0x1.0000000000000p+0",
        "gravity": "basic",
        "collision": "none",
        "boundary": "none",
        "softening_hex": "0x0.0p+0",
        "additional_forces_c_pointer_is_null": True,
        "force_is_velocity_dependent": 0,
        "exit_min_distance_AU_hex": "0x0.0p+0",
        "exit_max_distance_AU_hex": "0x0.0p+0",
        "N": SOURCE_COUNT,
        "N_active": SOURCE_COUNT,
        "testparticle_type": 0,
    }
    if initial_configuration != expected_initial:
        raise SourcePreflightExecutionError("IAS15 effective initial configuration changed")
    annual = np.empty((CHECKPOINT_COUNT, SOURCE_COUNT, 6), dtype=np.float64)
    initial_position, initial_velocity = _rebound_state(np, simulation)
    annual[0, :, :3] = initial_position
    annual[0, :, 3:] = initial_velocity
    monitor: dict[str, Any] = {
        "heartbeat_callback_count": 0,
        "observation_count": 1,
        "callback_exception": None,
    }

    def heartbeat(pointer: Any) -> None:
        current = pointer.contents if hasattr(pointer, "contents") else pointer
        try:
            _rebound_state(np, current)
            monitor["heartbeat_callback_count"] += 1
            monitor["observation_count"] += 1
        except BaseException as exc:  # ctypes callbacks otherwise swallow failures.
            monitor["callback_exception"] = (
                f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
            )

    callback = _bind_rebound_heartbeat(simulation, rebound, heartbeat)
    for year in range(1, CHECKPOINT_COUNT):
        simulation.integrate(float(year), exact_finish_time=1)
        simulation.synchronize()
        if monitor["callback_exception"] is not None:
            raise SourcePreflightExecutionError("IAS15 heartbeat guard failed")
        if float(simulation.t).hex() != float(year).hex():
            raise SourcePreflightExecutionError(
                "IAS15 did not stop at an exact annual checkpoint"
            )
        positions, velocities = _rebound_state(np, simulation)
        annual[year, :, :3] = positions
        annual[year, :, 3:] = velocities
    if callback is None or monitor["heartbeat_callback_count"] <= 0:
        raise SourcePreflightExecutionError("IAS15 heartbeat evidence is absent")
    if int(simulation.steps_done) <= 0:
        raise SourcePreflightExecutionError("IAS15 accepted no internal steps")
    if any(name == "cupy" or name.startswith("cupy.") for name in sys.modules):
        raise SourcePreflightExecutionError("GPU runtime entered a CPU-only execution")
    receipt = {
        "lane_id": "rebound511_ias15_reduced5",
        "engine": "REBOUND_5.1.1_IAS15",
        "model": "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS",
        "initial_effective_configuration": initial_configuration,
        "final_effective_configuration": {
            "integrator": str(simulation.integrator),
            "epsilon_hex": float(simulation.integrator.epsilon).hex(),
            "minimum_dt_julian_years_hex": float(simulation.integrator.min_dt).hex(),
            "adaptive_mode": str(simulation.integrator.adaptive_mode),
            "final_time_julian_years_hex": float(simulation.t).hex(),
            "steps_done": int(simulation.steps_done),
            "G_hex": float(simulation.G).hex(),
            "gravity": str(simulation.gravity),
            "collision": str(simulation.collision),
            "boundary": str(simulation.boundary),
            "softening_hex": float(simulation.softening).hex(),
            "additional_forces_c_pointer_is_null": (
                _rebound_additional_forces_pointer_is_null(simulation)
            ),
            "force_is_velocity_dependent": int(
                simulation.force_is_velocity_dependent
            ),
            "exit_min_distance_AU_hex": float(
                simulation.exit_min_distance
            ).hex(),
            "exit_max_distance_AU_hex": float(
                simulation.exit_max_distance
            ).hex(),
            "N": int(simulation.N),
            "N_active": int(simulation.N_active),
            "testparticle_type": int(simulation.testparticle_type),
        },
        "integration_call_count": HORIZON_JULIAN_YEARS,
        "annual_checkpoint_count": CHECKPOINT_COUNT,
        "heartbeat_binding": (
            "PINNED_REBOUND_5_1_1_C_FIELD_WITH_LOCAL_CALLBACK_REFERENCE_DUE_MISSING_HB_SLOT"
        ),
        "heartbeat_callback_count": monitor["heartbeat_callback_count"],
        "finite_roster_observation_count": monitor["observation_count"],
        "heartbeat_callback_exception": None,
        "force_and_step_implementation": "INDEPENDENT_REBOUND_5_1_1_NO_JX_FORCE_OR_STEP_CODE",
    }
    if (
        receipt["final_effective_configuration"]["integrator"] != "ias15"
        or receipt["final_effective_configuration"]["epsilon_hex"]
        != (1.0e-13).hex()
        or receipt["final_effective_configuration"]["minimum_dt_julian_years_hex"]
        != "0x0.0p+0"
        or receipt["final_effective_configuration"]["adaptive_mode"] != "PRS23"
        or receipt["final_effective_configuration"]["final_time_julian_years_hex"]
        != float(HORIZON_JULIAN_YEARS).hex()
        or receipt["final_effective_configuration"]["G_hex"]
        != "0x1.0000000000000p+0"
        or receipt["final_effective_configuration"]["gravity"] != "basic"
        or receipt["final_effective_configuration"]["collision"] != "none"
        or receipt["final_effective_configuration"]["boundary"] != "none"
        or receipt["final_effective_configuration"]["softening_hex"]
        != "0x0.0p+0"
        or receipt["final_effective_configuration"][
            "additional_forces_c_pointer_is_null"
        ]
        is not True
        or receipt["final_effective_configuration"][
            "force_is_velocity_dependent"
        ]
        != 0
        or receipt["final_effective_configuration"]["exit_min_distance_AU_hex"]
        != "0x0.0p+0"
        or receipt["final_effective_configuration"]["exit_max_distance_AU_hex"]
        != "0x0.0p+0"
        or receipt["final_effective_configuration"]["N"] != SOURCE_COUNT
        or receipt["final_effective_configuration"]["N_active"] != SOURCE_COUNT
        or receipt["final_effective_configuration"]["testparticle_type"] != 0
    ):
        raise SourcePreflightExecutionError("IAS15 effective final configuration changed")
    return annual, receipt


def _barycentric_node(
    np: Any,
    positions: Any,
    velocities: Any,
    gm: Any,
) -> tuple[Any, Any]:
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or type(gm) is not np.ndarray
        or positions.shape != (SOURCE_COUNT, 3)
        or velocities.shape != (SOURCE_COUNT, 3)
        or gm.shape != (SOURCE_COUNT,)
    ):
        raise SourcePreflightExecutionError("barycentric node geometry changed")
    total = math.fsum(float(gm[index]) for index in range(SOURCE_COUNT))
    if not math.isfinite(total) or total <= 0.0:
        raise SourcePreflightExecutionError("common GM is not finite and positive")
    center_position = tuple(
        math.fsum(
            float(gm[index]) * float(positions[index, component])
            for index in range(SOURCE_COUNT)
        )
        / total
        for component in range(3)
    )
    center_velocity = tuple(
        math.fsum(
            float(gm[index]) * float(velocities[index, component])
            for index in range(SOURCE_COUNT)
        )
        / total
        for component in range(3)
    )
    barycentric_positions = np.empty((SOURCE_COUNT, 3), dtype=np.float64)
    barycentric_velocities = np.empty_like(barycentric_positions)
    for index in range(SOURCE_COUNT):
        for component in range(3):
            barycentric_positions[index, component] = (
                float(positions[index, component]) - center_position[component]
            )
            barycentric_velocities[index, component] = (
                float(velocities[index, component]) - center_velocity[component]
            )
    if not bool(np.all(np.isfinite(barycentric_positions))) or not bool(
        np.all(np.isfinite(barycentric_velocities))
    ):
        raise SourcePreflightExecutionError(
            "barycentric reconstruction produced nonfinite state"
        )
    return barycentric_positions, barycentric_velocities


def _invariant_row(
    np: Any,
    positions: Any,
    velocities: Any,
    gm: Any,
) -> tuple[float, float, float, float]:
    barycentric_positions, barycentric_velocities = _barycentric_node(
        np, positions, velocities, gm
    )
    kinetic = math.fsum(
        0.5
        * float(gm[index])
        * math.fsum(
            float(barycentric_velocities[index, component]) ** 2
            for component in range(3)
        )
        for index in range(SOURCE_COUNT)
    )
    potential_terms = []
    for left in range(SOURCE_COUNT):
        for right in range(left + 1, SOURCE_COUNT):
            separation = math.dist(
                tuple(float(value) for value in barycentric_positions[left]),
                tuple(float(value) for value in barycentric_positions[right]),
            )
            if not math.isfinite(separation) or separation <= 0.0:
                raise SourcePreflightExecutionError(
                    "massive-source invariant encountered a singular separation"
                )
            potential_terms.append(
                -float(gm[left]) * float(gm[right]) / separation
            )
    energy = kinetic + math.fsum(potential_terms)
    angular = []
    for component in range(3):
        first = (component + 1) % 3
        second = (component + 2) % 3
        angular.append(
            math.fsum(
                float(gm[index])
                * (
                    float(barycentric_positions[index, first])
                    * float(barycentric_velocities[index, second])
                    - float(barycentric_positions[index, second])
                    * float(barycentric_velocities[index, first])
                )
                for index in range(SOURCE_COUNT)
            )
        )
    row = (energy, angular[0], angular[1], angular[2])
    if not all(math.isfinite(value) for value in row):
        raise SourcePreflightExecutionError(
            "massive-source invariant is nonfinite"
        )
    return row


def _invariant_stream(
    np: Any,
    positions: Any,
    velocities: Any,
    gm: Any,
) -> tuple[Any, dict[str, Any]]:
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or positions.ndim != 3
        or positions.shape != velocities.shape
        or positions.shape[1:] != (SOURCE_COUNT, 3)
    ):
        raise SourcePreflightExecutionError("source stream geometry changed")
    invariants = np.empty((positions.shape[0], INVARIANT_WIDTH), dtype=np.float64)
    for index in range(positions.shape[0]):
        invariants[index] = _invariant_row(
            np, positions[index], velocities[index], gm
        )
    initial_energy = float(invariants[0, 0])
    initial_angular = tuple(float(value) for value in invariants[0, 1:])
    energy_denominator = abs(initial_energy)
    angular_denominator = math.hypot(*initial_angular)
    if (
        not math.isfinite(energy_denominator)
        or energy_denominator == 0.0
        or not math.isfinite(angular_denominator)
        or angular_denominator == 0.0
    ):
        raise SourcePreflightExecutionError(
            "massive-source invariant denominator is zero or nonfinite"
        )
    maximum_energy_exact = Fraction(-1, 1)
    maximum_energy_node = -1
    maximum_angular_squared_exact = Fraction(-1, 1)
    maximum_angular_node = -1
    for index in range(invariants.shape[0]):
        energy_drift_exact = _exact_energy_relative_drift(
            float(invariants[index, 0]), initial_energy
        )
        angular_drift_squared_exact = _exact_angular_relative_drift_squared(
            tuple(float(value) for value in invariants[index, 1:]),
            initial_angular,
        )
        if energy_drift_exact > maximum_energy_exact:
            maximum_energy_exact = energy_drift_exact
            maximum_energy_node = index
        if angular_drift_squared_exact > maximum_angular_squared_exact:
            maximum_angular_squared_exact = angular_drift_squared_exact
            maximum_angular_node = index
    maximum_energy = _fraction_to_outward_float(maximum_energy_exact)
    maximum_angular = _sqrt_fraction_to_outward_float(
        maximum_angular_squared_exact
    )
    summary = {
        "node_count": int(invariants.shape[0]),
        "initial_energy_hex": initial_energy.hex(),
        "initial_angular_momentum_vector_hex": [value.hex() for value in initial_angular],
        "initial_angular_momentum_norm_hex": angular_denominator.hex(),
        "maximum_relative_energy_drift_hex": maximum_energy.hex(),
        "maximum_relative_energy_drift_node": maximum_energy_node,
        "maximum_relative_angular_vector_drift_hex": maximum_angular.hex(),
        "maximum_relative_angular_vector_drift_node": maximum_angular_node,
    }
    return invariants, summary


def _exact_squared(vector: Sequence[float]) -> Fraction:
    if type(vector) not in (tuple, list) or len(vector) != 3:
        raise SourcePreflightExecutionError("exact vector must contain three components")
    if any(type(value) is not float or not math.isfinite(value) for value in vector):
        raise SourcePreflightExecutionError("exact vector contains nonfinite data")
    return sum(
        (Fraction.from_float(value) ** 2 for value in vector),
        Fraction(0, 1),
    )


def _exact_squared_vector_difference(
    left: Sequence[float],
    right: Sequence[float],
) -> Fraction:
    if (
        type(left) not in (tuple, list)
        or type(right) not in (tuple, list)
        or len(left) != 3
        or len(right) != 3
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in (*left, *right)
        )
    ):
        raise SourcePreflightExecutionError(
            "exact comparison vectors must contain three finite floats"
        )
    return sum(
        (
            (
                Fraction.from_float(left[index])
                - Fraction.from_float(right[index])
            )
            ** 2
            for index in range(3)
        ),
        Fraction(0, 1),
    )


def _exact_squared_reference_ulp_floor(reference: Sequence[float]) -> Fraction:
    if (
        type(reference) not in (tuple, list)
        or len(reference) != 3
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in reference
        )
    ):
        raise SourcePreflightExecutionError(
            "ULP reference vector must contain three finite floats"
        )
    return sum(
        (
            (64 * Fraction.from_float(math.ulp(value))) ** 2
            for value in reference
        ),
        Fraction(0, 1),
    )


def _sqrt_fraction_to_outward_float(value: Fraction) -> float:
    if type(value) is not Fraction or value < 0:
        raise SourcePreflightExecutionError(
            "outward square root requires a nonnegative exact fraction"
        )
    if value == 0:
        return 0.0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    even_exponent = exponent if exponent % 2 == 0 else exponent - 1
    scaled = (
        value / Fraction(1 << even_exponent, 1)
        if even_exponent >= 0
        else value * Fraction(1 << (-even_exponent), 1)
    )
    result = math.ldexp(math.sqrt(float(scaled)), even_exponent // 2)
    if not math.isfinite(result):
        raise SourcePreflightExecutionError("outward square root overflowed")
    while Fraction.from_float(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    while result > 0.0:
        predecessor = math.nextafter(result, -math.inf)
        if Fraction.from_float(predecessor) ** 2 < value:
            break
        result = predecessor
    return result


def _fraction_to_outward_float(value: Fraction) -> float:
    if type(value) is not Fraction or value < 0:
        raise SourcePreflightExecutionError(
            "outward conversion requires a nonnegative exact fraction"
        )
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise SourcePreflightExecutionError("outward conversion overflowed") from exc
    if not math.isfinite(result):
        raise SourcePreflightExecutionError("outward conversion overflowed")
    while Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    while result > 0.0:
        predecessor = math.nextafter(result, -math.inf)
        if Fraction.from_float(predecessor) < value:
            break
        result = predecessor
    return result


def _exact_vector_threshold(
    difference: Sequence[float],
    *,
    numerator: int,
    denominator: int,
    divisor: int = 1,
) -> bool:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or type(divisor) is not int
        or numerator <= 0
        or denominator <= 0
        or divisor <= 0
    ):
        raise SourcePreflightExecutionError("exact vector threshold is invalid")
    return (
        _exact_squared(difference) * denominator * denominator
        <= numerator * numerator * divisor * divisor
    )


def _exact_vector_contraction(
    coarse_difference: Sequence[float],
    fine_difference: Sequence[float],
    reference: Sequence[float],
) -> bool:
    coarse_squared = _exact_squared(coarse_difference)
    fine_squared = _exact_squared(fine_difference)
    if type(reference) not in (tuple, list) or len(reference) != 3:
        raise SourcePreflightExecutionError("reference vector geometry changed")
    floor_squared = sum(
        (
            (64 * Fraction.from_float(math.ulp(value))) ** 2
            for value in reference
            if type(value) is float and math.isfinite(value)
        ),
        Fraction(0, 1),
    )
    if any(type(value) is not float or not math.isfinite(value) for value in reference):
        raise SourcePreflightExecutionError("reference vector is nonfinite")
    return fine_squared <= floor_squared or coarse_squared >= 9 * fine_squared


def _exact_vector_threshold_between(
    left: Sequence[float],
    right: Sequence[float],
    *,
    numerator: int,
    denominator: int,
    divisor: int = 1,
) -> bool:
    if (
        type(numerator) is not int
        or type(denominator) is not int
        or type(divisor) is not int
        or numerator <= 0
        or denominator <= 0
        or divisor <= 0
    ):
        raise SourcePreflightExecutionError("exact vector threshold is invalid")
    squared = _exact_squared_vector_difference(left, right)
    return (
        squared * denominator * denominator
        <= numerator * numerator * divisor * divisor
    )


def _exact_vector_contraction_between(
    coarse: Sequence[float],
    fine: Sequence[float],
    reference: Sequence[float],
) -> bool:
    coarse_squared = _exact_squared_vector_difference(coarse, reference)
    fine_squared = _exact_squared_vector_difference(fine, reference)
    floor_squared = _exact_squared_reference_ulp_floor(reference)
    return fine_squared <= floor_squared or coarse_squared >= 9 * fine_squared


def _maximum_residual_record(
    squared_value: Fraction,
    year: int,
    source_index: int,
) -> dict[str, Any]:
    return {
        "value_hex": _sqrt_fraction_to_outward_float(squared_value).hex(),
        "year_index": year,
        "source_index": source_index,
        "source_id": SOURCE_IDS[source_index],
    }


def _exact_energy_relative_drift(
    energy: float,
    initial_energy: float,
) -> Fraction:
    if (
        type(energy) is not float
        or type(initial_energy) is not float
        or not math.isfinite(energy)
        or not math.isfinite(initial_energy)
        or initial_energy == 0.0
    ):
        raise SourcePreflightExecutionError(
            "energy drift operands are zero or nonfinite"
        )
    exact_initial = Fraction.from_float(initial_energy)
    return abs(Fraction.from_float(energy) - exact_initial) / abs(exact_initial)


def _exact_angular_relative_drift_squared(
    angular: Sequence[float],
    initial_angular: Sequence[float],
) -> Fraction:
    if (
        type(angular) not in (tuple, list)
        or type(initial_angular) not in (tuple, list)
        or len(angular) != 3
        or len(initial_angular) != 3
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in (*angular, *initial_angular)
        )
    ):
        raise SourcePreflightExecutionError(
            "angular drift operands are nonfinite or malformed"
        )
    exact_initial = tuple(Fraction.from_float(value) for value in initial_angular)
    denominator_squared = sum(
        (value * value for value in exact_initial), Fraction(0, 1)
    )
    if denominator_squared == 0:
        raise SourcePreflightExecutionError(
            "angular drift denominator is zero"
        )
    numerator_squared = sum(
        (
            (Fraction.from_float(value) - baseline) ** 2
            for value, baseline in zip(angular, exact_initial, strict=True)
        ),
        Fraction(0, 1),
    )
    return numerator_squared / denominator_squared


def _evaluate_numeric_gates(
    np: Any,
    *,
    gm: Any,
    h64_positions: Any,
    h64_velocities: Any,
    h128_positions: Any,
    h128_velocities: Any,
    h64_invariants: Any,
    h128_invariants: Any,
    ias15_annual: Any,
    h64_stride: int = 64,
    h128_stride: int = 128,
) -> dict[str, Any]:
    annual_count = int(ias15_annual.shape[0])
    if (
        type(h64_stride) is not int
        or type(h128_stride) is not int
        or h64_stride <= 0
        or h128_stride <= 0
        or type(ias15_annual) is not np.ndarray
        or ias15_annual.dtype != np.dtype(np.float64)
        or ias15_annual.ndim != 3
        or ias15_annual.shape[1:] != (SOURCE_COUNT, 6)
        or h64_positions.shape != h64_velocities.shape
        or h128_positions.shape != h128_velocities.shape
        or h64_positions.shape[0] != (annual_count - 1) * h64_stride + 1
        or h128_positions.shape[0] != (annual_count - 1) * h128_stride + 1
        or h64_invariants.shape != (h64_positions.shape[0], INVARIANT_WIDTH)
        or h128_invariants.shape != (h128_positions.shape[0], INVARIANT_WIDTH)
    ):
        raise SourcePreflightExecutionError("numeric gate geometry changed")
    if any(
        not bool(np.all(np.isfinite(value)))
        for value in (
            h64_positions,
            h64_velocities,
            h128_positions,
            h128_velocities,
            h64_invariants,
            h128_invariants,
            ias15_annual,
        )
    ):
        raise SourcePreflightExecutionError("numeric gate input is nonfinite")

    maxima = {
        "richardson_position": (Fraction(-1, 1), -1, -1),
        "richardson_velocity": (Fraction(-1, 1), -1, -1),
        "h128_ias15_position": (Fraction(-1, 1), -1, -1),
        "h128_ias15_velocity": (Fraction(-1, 1), -1, -1),
    }
    decisions = {
        "richardson_position": True,
        "richardson_velocity": True,
        "h128_ias15_position": True,
        "h128_ias15_velocity": True,
        "position_vector_contraction": True,
        "velocity_vector_contraction": True,
        "canonical_year_zero_identity": True,
    }
    contraction_failure_counts = {"position": 0, "velocity": 0}
    for year in range(annual_count):
        coarse_position, coarse_velocity = _barycentric_node(
            np,
            h64_positions[year * h64_stride],
            h64_velocities[year * h64_stride],
            gm,
        )
        fine_position, fine_velocity = _barycentric_node(
            np,
            h128_positions[year * h128_stride],
            h128_velocities[year * h128_stride],
            gm,
        )
        reference_position, reference_velocity = _barycentric_node(
            np,
            ias15_annual[year, :, :3],
            ias15_annual[year, :, 3:],
            gm,
        )
        if year == 0 and not (
            _array_bitwise_equal(np, coarse_position, fine_position)
            and _array_bitwise_equal(np, coarse_velocity, fine_velocity)
            and _array_bitwise_equal(np, fine_position, reference_position)
            and _array_bitwise_equal(np, fine_velocity, reference_velocity)
        ):
            decisions["canonical_year_zero_identity"] = False
        for source_index in range(SOURCE_COUNT):
            vectors = (
                ("position", coarse_position[source_index], fine_position[source_index], reference_position[source_index], 1, 500),
                ("velocity", coarse_velocity[source_index], fine_velocity[source_index], reference_velocity[source_index], 1, 50_000),
            )
            for kind, coarse, fine, reference, numerator, denominator in vectors:
                coarse_tuple = tuple(float(value) for value in coarse)
                fine_tuple = tuple(float(value) for value in fine)
                reference_tuple = tuple(float(value) for value in reference)
                coarse_fine_squared = _exact_squared_vector_difference(
                    coarse_tuple, fine_tuple
                )
                fine_reference_squared = _exact_squared_vector_difference(
                    fine_tuple, reference_tuple
                )
                richardson_key = f"richardson_{kind}"
                reference_key = f"h128_ias15_{kind}"
                richardson_squared = coarse_fine_squared / 9
                if richardson_squared > maxima[richardson_key][0]:
                    maxima[richardson_key] = (
                        richardson_squared,
                        year,
                        source_index,
                    )
                if fine_reference_squared > maxima[reference_key][0]:
                    maxima[reference_key] = (
                        fine_reference_squared,
                        year,
                        source_index,
                    )
                if not _exact_vector_threshold_between(
                    coarse_tuple,
                    fine_tuple,
                    numerator=numerator,
                    denominator=denominator,
                    divisor=3,
                ):
                    decisions[richardson_key] = False
                if not _exact_vector_threshold_between(
                    fine_tuple,
                    reference_tuple,
                    numerator=numerator,
                    denominator=denominator,
                ):
                    decisions[reference_key] = False
                if not _exact_vector_contraction_between(
                    coarse_tuple,
                    fine_tuple,
                    reference_tuple,
                ):
                    decisions[f"{kind}_vector_contraction"] = False
                    contraction_failure_counts[kind] += 1

    def drift_summary(
        invariants: Any,
    ) -> tuple[Fraction, int, Fraction, int, float, float]:
        initial_energy = float(invariants[0, 0])
        initial_angular = tuple(float(value) for value in invariants[0, 1:])
        maximum_energy = Fraction(-1, 1)
        maximum_energy_node = -1
        maximum_angular_squared = Fraction(-1, 1)
        maximum_angular_node = -1
        for index, row in enumerate(invariants):
            energy = _exact_energy_relative_drift(
                float(row[0]), initial_energy
            )
            angular_squared = _exact_angular_relative_drift_squared(
                tuple(float(value) for value in row[1:]), initial_angular
            )
            if energy > maximum_energy:
                maximum_energy = energy
                maximum_energy_node = index
            if angular_squared > maximum_angular_squared:
                maximum_angular_squared = angular_squared
                maximum_angular_node = index
        maximum_energy_float = _fraction_to_outward_float(maximum_energy)
        maximum_angular_float = _sqrt_fraction_to_outward_float(
            maximum_angular_squared
        )
        return (
            maximum_energy,
            maximum_energy_node,
            maximum_angular_squared,
            maximum_angular_node,
            maximum_energy_float,
            maximum_angular_float,
        )

    (
        coarse_energy_exact,
        coarse_energy_node,
        coarse_angular_squared_exact,
        coarse_angular_node,
        coarse_energy,
        coarse_angular,
    ) = drift_summary(h64_invariants)
    (
        fine_energy_exact,
        fine_energy_node,
        fine_angular_squared_exact,
        fine_angular_node,
        fine_energy,
        fine_angular,
    ) = drift_summary(h128_invariants)
    fine_energy_pass = fine_energy_exact <= Fraction(1, 100_000_000)
    fine_angular_pass = fine_angular_squared_exact <= Fraction(
        1, 10_000_000_000
    ) ** 2
    energy_contraction_pass = fine_energy_exact <= max(
        coarse_energy_exact / 3,
        Fraction(1, 1 << 46),
    )

    gate_records: dict[str, Any] = {
        "canonical_year_zero_identity": {
            "pass": decisions["canonical_year_zero_identity"],
            "rule": "H64_H128_IAS15_COMMON_GM_BARYCENTRIC_YEAR_ZERO_BITWISE_IDENTICAL",
        },
        "richardson_position": {
            "pass": decisions["richardson_position"],
            "threshold": "1/500_AU",
            "divisor": 3,
            "maximum": _maximum_residual_record(*maxima["richardson_position"]),
            "decision": "EXACT_SQUARED_RATIONAL",
        },
        "richardson_velocity": {
            "pass": decisions["richardson_velocity"],
            "threshold": "1/50000_AU_PER_JULIAN_YEAR",
            "divisor": 3,
            "maximum": _maximum_residual_record(*maxima["richardson_velocity"]),
            "decision": "EXACT_SQUARED_RATIONAL",
        },
        "h128_vs_ias15_position": {
            "pass": decisions["h128_ias15_position"],
            "threshold": "1/500_AU",
            "maximum": _maximum_residual_record(*maxima["h128_ias15_position"]),
            "decision": "EXACT_SQUARED_RATIONAL",
        },
        "h128_vs_ias15_velocity": {
            "pass": decisions["h128_ias15_velocity"],
            "threshold": "1/50000_AU_PER_JULIAN_YEAR",
            "maximum": _maximum_residual_record(*maxima["h128_ias15_velocity"]),
            "decision": "EXACT_SQUARED_RATIONAL",
        },
        "position_vector_contraction": {
            "pass": decisions["position_vector_contraction"],
            "failure_count": contraction_failure_counts["position"],
            "rule": "FINE_SQUARED_LE_64_ULP_VECTOR_FLOOR_OR_COARSE_SQUARED_GE_9_FINE_SQUARED",
        },
        "velocity_vector_contraction": {
            "pass": decisions["velocity_vector_contraction"],
            "failure_count": contraction_failure_counts["velocity"],
            "rule": "FINE_SQUARED_LE_64_ULP_VECTOR_FLOOR_OR_COARSE_SQUARED_GE_9_FINE_SQUARED",
        },
        "h128_relative_energy_drift": {
            "pass": fine_energy_pass,
            "threshold": "1/100000000",
            "maximum_value_hex": fine_energy.hex(),
            "maximum_node": fine_energy_node,
        },
        "h128_relative_angular_vector_drift": {
            "pass": fine_angular_pass,
            "threshold": "1/10000000000",
            "maximum_value_hex": fine_angular.hex(),
            "maximum_node": fine_angular_node,
        },
        "energy_contraction": {
            "pass": energy_contraction_pass,
            "rule": "FINE_LE_MAX_COARSE_DIV_3_AND_2_POW_MINUS_46",
            "coarse_maximum_value_hex": coarse_energy.hex(),
            "coarse_maximum_node": coarse_energy_node,
            "fine_maximum_value_hex": fine_energy.hex(),
            "fine_maximum_node": fine_energy_node,
            "floor_hex": float.fromhex("0x1.0000000000000p-46").hex(),
        },
    }
    gate_records["all_applicable_source_predicates"] = {
        "pass": all(record["pass"] is True for record in gate_records.values()),
        "individual_gate_count": len(gate_records),
        "all_individual_predicates_required": True,
    }
    return gate_records


def _source_state_payload(np: Any, positions: Any, velocities: Any) -> bytes:
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or positions.shape != velocities.shape
        or positions.ndim != 3
        or positions.shape[1:] != (SOURCE_COUNT, 3)
    ):
        raise SourcePreflightExecutionError("source-state payload geometry changed")
    combined = np.empty(
        (positions.shape[0], SOURCE_COUNT, 6), dtype=np.dtype("<f8"), order="C"
    )
    combined[:, :, :3] = positions
    combined[:, :, 3:] = velocities
    if not bool(np.all(np.isfinite(combined))):
        raise SourcePreflightExecutionError("source-state payload is nonfinite")
    return combined.tobytes(order="C")


def _float64_payload(np: Any, value: Any, shape: tuple[int, ...]) -> bytes:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not bool(np.all(np.isfinite(value)))
    ):
        raise SourcePreflightExecutionError("binary64 payload geometry changed")
    canonical = np.ascontiguousarray(value, dtype=np.dtype("<f8"))
    return canonical.tobytes(order="C")


def _artifact_receipt(payload: bytes, role: str) -> dict[str, Any]:
    return {"bytes": len(payload), "sha256": _sha256(payload), "role": role}


def _secure_root(path: Path, *, expected_roster: Sequence[str]) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise SourcePreflightExecutionError("cannot stat result root") from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or absolute.is_symlink()
        or stat.S_IMODE(metadata.st_mode) != 0o700
        or metadata.st_uid != os.geteuid()
    ):
        raise SourcePreflightExecutionError(
            "result root is not an owner-private direct directory"
        )
    try:
        observed = sorted(item.name for item in absolute.iterdir())
    except OSError as exc:
        raise SourcePreflightExecutionError("cannot enumerate result root") from exc
    if observed != sorted(expected_roster):
        raise SourcePreflightExecutionError("result root roster changed")


def _secure_result_bytes(
    root: Path,
    name: str,
    *,
    expected_bytes: int | None = None,
) -> bytes:
    if type(name) is not str or not name or "/" in name or name in (".", ".."):
        raise SourcePreflightExecutionError("result member name is invalid")
    path = root / name
    try:
        before_path = path.lstat()
    except OSError as exc:
        raise SourcePreflightExecutionError(f"cannot stat result member {name}") from exc
    if (
        not stat.S_ISREG(before_path.st_mode)
        or path.is_symlink()
        or before_path.st_nlink != 1
        or stat.S_IMODE(before_path.st_mode) != 0o600
        or before_path.st_uid != os.geteuid()
    ):
        raise SourcePreflightExecutionError(
            f"result member {name} custody changed"
        )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourcePreflightExecutionError(f"cannot open result member {name}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            before.st_dev != before_path.st_dev
            or before.st_ino != before_path.st_ino
            or not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
        ):
            raise SourcePreflightExecutionError(
                f"result member {name} changed before read"
            )
        if (
            (expected_bytes is not None and before.st_size != expected_bytes)
            or (
                expected_bytes is None
                and before.st_size > MAXIMUM_JSON_RECEIPT_BYTES
            )
        ):
            raise SourcePreflightExecutionError(
                f"result member {name} byte length changed"
            )
        chunks = bytearray()
        while len(chunks) < before.st_size:
            chunk = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(chunks)),
                len(chunks),
            )
            if not chunk:
                raise SourcePreflightExecutionError(
                    f"result member {name} ended early"
                )
            chunks.extend(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after:
            raise SourcePreflightExecutionError(
                f"result member {name} changed while held"
            )
    finally:
        os.close(descriptor)
    payload = bytes(chunks)
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise SourcePreflightExecutionError(
            f"result member {name} byte length changed"
        )
    return payload


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SourcePreflightExecutionError(
            f"cannot create result member {path.name}"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise SourcePreflightExecutionError(
                    f"result member {path.name} write made no progress"
                )
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise SourcePreflightExecutionError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result != 0:
        observed = ctypes.get_errno()
        if observed == errno.EEXIST:
            raise SourcePreflightExecutionError(
                "result destination already exists; create-only publication refused"
            )
        raise SourcePreflightExecutionError(
            f"create-only result publication failed: {os.strerror(observed)}"
        )


def _require_safe_absent_output(output_root: Path) -> Path:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise SourcePreflightExecutionError("result root name is invalid")
    for component in reversed([root.parent, *root.parent.parents]):
        if component.exists() and component.is_symlink():
            raise SourcePreflightExecutionError("result root has a symlink ancestor")
    try:
        parent_metadata = root.parent.lstat()
    except OSError as exc:
        raise SourcePreflightExecutionError("result parent does not exist") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode) or root.parent.is_symlink():
        raise SourcePreflightExecutionError("result parent is not a direct directory")
    if root.exists() or root.is_symlink():
        raise SourcePreflightExecutionError(
            "result root already exists; create-only execution refused before lane zero"
        )
    try:
        parent_fd = os.open(
            root.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError as exc:
        raise SourcePreflightExecutionError(
            "result parent cannot provide durable create-only publication"
        ) from exc
    return root


def _publish_staged_directory(stage: Path, root: Path) -> None:
    root_visible = False
    try:
        _rename_directory_noreplace(stage, root)
        root_visible = True
        parent_fd = os.open(
            root.parent,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException as publication_error:
        if root_visible:
            try:
                _rename_directory_noreplace(root, stage)
            except BaseException as rollback_error:
                raise SourcePreflightExecutionError(
                    "result publication durability failed and atomic rollback "
                    "could not restore the unpublished staging name"
                ) from rollback_error
        raise SourcePreflightExecutionError(
            "result publication durability failed; atomic publication was rolled back"
        ) from publication_error


def _build_result(
    *,
    authority: Mapping[str, Any],
    runtime: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    h64_receipt: Mapping[str, Any],
    h128_receipt: Mapping[str, Any],
    ias15_receipt: Mapping[str, Any],
    h64_invariant_summary: Mapping[str, Any],
    h128_invariant_summary: Mapping[str, Any],
    gates: Mapping[str, Any],
    initial_state_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if gates.get("all_applicable_source_predicates", {}).get("pass") is not True:
        raise SourcePreflightExecutionError(
            "applicable source predicates did not all pass"
        )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": "PASSED_BOUNDED_AUTONOMOUS_REDUCED5_SOURCE_PREFLIGHT",
        "evidence_class": (
            "PASSED_1KYR_AUTONOMOUS_REDUCED5_SOURCE_NUMERICAL_PREFLIGHT_"
            "NO_TRACER_NO_SCIENCE_RESULT"
        ),
        "claim_ceiling": (
            "BOUNDED_SOURCE_PREFLIGHT_PASS_NO_TRACER_QUALIFICATION_NO_LONG_"
            "HORIZON_AUTHORITY_NO_SCIENCE_RESULT"
        ),
        "authority": dict(authority),
        "successor_implementation": runtime["successor_implementation"],
        "runtime_content_sha256": runtime["content_sha256"],
        "initial_state_binding": dict(initial_state_binding),
        "lanes": {
            "jx_h64": dict(h64_receipt),
            "jx_h128": dict(h128_receipt),
            "rebound511_ias15": dict(ias15_receipt),
        },
        "invariants": {
            "jx_h64": dict(h64_invariant_summary),
            "jx_h128": dict(h128_invariant_summary),
            "sampling": "EVERY_FIXED_NODE_INCLUDING_BOTH_ENDPOINTS",
        },
        "gates": dict(gates),
        "artifacts": dict(artifacts),
        "claims": dict(CLAIMS),
        "nonclaims": list(NONCLAIMS),
    }
    result["content_sha256"] = _domain_hash(RESULT_DOMAIN, result)
    return result


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha1(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_file_receipt(value: object, *, locator: str | None = None) -> None:
    if (
        type(value) is not dict
        or set(value) != {"locator", "bytes", "sha256"}
        or (locator is not None and value.get("locator") != locator)
        or type(value.get("locator")) is not str
        or not value["locator"]
        or value["locator"].startswith("/")
        or type(value.get("bytes")) is not int
        or value["bytes"] <= 0
        or not _is_sha256(value.get("sha256"))
    ):
        raise SourcePreflightExecutionError("runtime file receipt changed")


def _validate_runtime_record(
    runtime: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema",
        "authority",
        "successor_implementation",
        "python",
        "platform",
        "thread_environment",
        "numpy_distribution",
        "native_extension",
        "rebound_distribution",
        "loaded_module_manifest",
        "loaded_shared_library_manifest",
        "gpu_runtime_imported",
        "cuda_device_queried",
        "dynamics_executed_at_receipt_time",
        "content_sha256",
    }
    if (
        type(runtime) is not dict
        or set(runtime) != expected_keys
        or runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("authority") != authority
        or runtime.get("content_sha256") != _domain_hash(RUNTIME_DOMAIN, runtime)
        or runtime.get("gpu_runtime_imported") is not False
        or runtime.get("cuda_device_queried") is not False
        or runtime.get("dynamics_executed_at_receipt_time") is not False
    ):
        raise SourcePreflightExecutionError("runtime receipt schema changed")
    python = runtime.get("python")
    if (
        type(python) is not dict
        or set(python)
        != {
            "version", "implementation", "resolved_path_sha256", "executable"
        }
        or python.get("version") != PINNED_PYTHON_VERSION
        or python.get("implementation") != "CPython"
        or python.get("resolved_path_sha256")
        != PINNED_PYTHON_RESOLVED_PATH_SHA256
    ):
        raise SourcePreflightExecutionError("runtime Python receipt changed")
    _validate_file_receipt(
        python["executable"], locator="PINNED_PYTHON_EXECUTABLE"
    )
    if (
        python["executable"]["bytes"] != PINNED_PYTHON_BYTES
        or python["executable"]["sha256"] != PINNED_PYTHON_SHA256
    ):
        raise SourcePreflightExecutionError("runtime Python identity changed")
    platform_record = runtime.get("platform")
    if platform_record != {
        "system": "Linux",
        "machine": "x86_64",
        "byteorder": "little",
        "float_mant_dig": 53,
        "float_radix": 2,
    }:
        raise SourcePreflightExecutionError("runtime platform receipt changed")
    if runtime.get("thread_environment") != {
        name: "1" for name in THREAD_ENVIRONMENT
    }:
        raise SourcePreflightExecutionError("runtime thread environment changed")
    numpy_record = runtime.get("numpy_distribution")
    if (
        type(numpy_record) is not dict
        or set(numpy_record)
        != {"version", "initializer", "record", "record_base64"}
        or numpy_record.get("version") != PINNED_NUMPY_VERSION
        or type(numpy_record.get("record_base64")) is not str
    ):
        raise SourcePreflightExecutionError("runtime NumPy receipt changed")
    _validate_file_receipt(
        numpy_record["initializer"],
        locator="NUMPY_DISTRIBUTION:numpy/__init__.py",
    )
    if (
        numpy_record["initializer"]["bytes"] != PINNED_NUMPY_INIT_BYTES
        or numpy_record["initializer"]["sha256"] != PINNED_NUMPY_INIT_SHA256
    ):
        raise SourcePreflightExecutionError("runtime NumPy initializer changed")
    _validate_file_receipt(numpy_record["record"])
    if (
        not numpy_record["record"]["locator"].startswith("NUMPY_DISTRIBUTION:")
        or not numpy_record["record"]["locator"].endswith("/RECORD")
        or numpy_record["record"]["bytes"] != PINNED_NUMPY_RECORD_BYTES
        or numpy_record["record"]["sha256"] != PINNED_NUMPY_RECORD_SHA256
    ):
        raise SourcePreflightExecutionError("runtime NumPy RECORD changed")
    try:
        embedded_numpy_record = base64.b64decode(
            numpy_record["record_base64"],
            validate=True,
        )
    except (ValueError, TypeError) as exc:
        raise SourcePreflightExecutionError(
            "embedded NumPy RECORD is not strict base64"
        ) from exc
    if (
        len(embedded_numpy_record) != PINNED_NUMPY_RECORD_BYTES
        or _sha256(embedded_numpy_record) != PINNED_NUMPY_RECORD_SHA256
    ):
        raise SourcePreflightExecutionError("embedded NumPy RECORD changed")
    numpy_record_manifest = _parse_numpy_record_manifest(embedded_numpy_record)
    native = runtime.get("native_extension")
    _validate_file_receipt(
        native,
        locator=f"PROJECT_ROOT:{NATIVE_EXTENSION_RELATIVE_PATH}",
    )
    if native["sha256"] != NATIVE_EXTENSION_SHA256:
        raise SourcePreflightExecutionError("runtime native extension changed")
    if native["bytes"] != NATIVE_EXTENSION_BYTES:
        raise SourcePreflightExecutionError("runtime native extension size changed")
    rebound = runtime.get("rebound_distribution")
    if (
        type(rebound) is not dict
        or set(rebound) != {"version", "wheel", "initializer", "library"}
        or rebound.get("version") != "5.1.1"
    ):
        raise SourcePreflightExecutionError("runtime REBOUND receipt changed")
    for name in ("wheel", "initializer", "library"):
        _validate_file_receipt(rebound[name])
    if (
        rebound["wheel"]["locator"] != f"PROJECT_ROOT:{REBOUND_WHEEL_RELATIVE_PATH}"
        or rebound["wheel"]["bytes"] != REBOUND_WHEEL_BYTES
        or rebound["wheel"]["sha256"] != REBOUND_WHEEL_SHA256
        or rebound["initializer"]
        != {
            "locator": "REBOUND_WHEEL:rebound/__init__.py",
            "bytes": REBOUND_INIT_BYTES,
            "sha256": REBOUND_INIT_SHA256,
        }
        or not rebound["library"]["locator"].startswith(
            "REBOUND_WHEEL:librebound"
        )
        or rebound["library"]["sha256"] != REBOUND_LIBRARY_SHA256
        or rebound["library"]["bytes"] != REBOUND_LIBRARY_BYTES
    ):
        raise SourcePreflightExecutionError("runtime REBOUND identity changed")
    successor = runtime.get("successor_implementation")
    expected_relative = SCRIPT_PATH.relative_to(PROJECT_ROOT).as_posix()
    if (
        type(successor) is not dict
        or set(successor)
        != {
            "commit", "tree", "preregistration_ancestor", "relative_path",
            "bytes", "sha256", "tree_change_policy", "test_relative_path",
            "test_bytes", "test_sha256",
        }
        or not _is_git_sha1(successor.get("commit"))
        or not _is_git_sha1(successor.get("tree"))
        or successor.get("commit") == PREREGISTRATION_COMMIT
        or successor.get("preregistration_ancestor") != PREREGISTRATION_COMMIT
        or successor.get("relative_path") != expected_relative
        or successor.get("tree_change_policy")
        != "EXACTLY_TWO_ADDED_EXECUTOR_AND_TEST_PATHS"
        or successor.get("test_relative_path") != EXECUTOR_TEST_RELATIVE_PATH
        or type(successor.get("bytes")) is not int
        or successor["bytes"] <= 0
        or not _is_sha256(successor.get("sha256"))
        or type(successor.get("test_bytes")) is not int
        or successor["test_bytes"] <= 0
        or not _is_sha256(successor.get("test_sha256"))
        or _git_text("rev-parse", f"{successor['commit']}^{{tree}}")
        != successor["tree"]
    ):
        raise SourcePreflightExecutionError(
            "execution-successor commit receipt changed"
        )
    _git(
        "merge-base",
        "--is-ancestor",
        PREREGISTRATION_COMMIT,
        successor["commit"],
    )
    _verify_successor_preserves_sealed_authority(successor["commit"])
    test_receipt = _verify_successor_add_only_scope(successor["commit"])
    if any(successor.get(key) != value for key, value in test_receipt.items()):
        raise SourcePreflightExecutionError(
            "execution-successor test blob changed"
        )
    successor_payload = _git(
        "show", f"{successor['commit']}:{successor['relative_path']}"
    )
    if (
        len(successor_payload) != successor["bytes"]
        or _sha256(successor_payload) != successor["sha256"]
    ):
        raise SourcePreflightExecutionError(
            "execution-successor source blob changed"
        )
    modules = runtime.get("loaded_module_manifest")
    if type(modules) is not list or not modules:
        raise SourcePreflightExecutionError("loaded module manifest changed")
    observed_names: set[str] = set()
    rebound_members, _ = _wheel_members()
    for item in modules:
        if (
            type(item) is not dict
            or set(item) != {"module", "locator", "bytes", "sha256"}
            or type(item.get("module")) is not str
            or item["module"] in observed_names
        ):
            raise SourcePreflightExecutionError("loaded module receipt changed")
        observed_names.add(item["module"])
        _validate_file_receipt(
            {key: item[key] for key in ("locator", "bytes", "sha256")}
        )
        locator = item["locator"]
        module_name = item["module"]
        if module_name == "jxplanetx" or module_name.startswith("jxplanetx."):
            if not locator.startswith("PROJECT_ROOT:"):
                raise SourcePreflightExecutionError(
                    "JX module namespace locator changed"
                )
            relative = locator.removeprefix("PROJECT_ROOT:")
            payload = _git("show", f"{successor['commit']}:{relative}")
            if len(payload) != item["bytes"] or _sha256(payload) != item["sha256"]:
                raise SourcePreflightExecutionError("committed JX module receipt changed")
        elif module_name == "rebound" or module_name.startswith("rebound."):
            if not locator.startswith("REBOUND_WHEEL:rebound/"):
                raise SourcePreflightExecutionError(
                    "REBOUND module namespace locator changed"
                )
            relative = locator.removeprefix("REBOUND_WHEEL:")
            if rebound_members.get(relative) != (item["bytes"], item["sha256"]):
                raise SourcePreflightExecutionError("REBOUND module receipt changed")
        elif module_name == "numpy" or module_name.startswith("numpy."):
            if not locator.startswith("NUMPY_DISTRIBUTION:numpy/"):
                raise SourcePreflightExecutionError(
                    "NumPy module namespace locator changed"
                )
            relative = locator.removeprefix("NUMPY_DISTRIBUTION:")
            if numpy_record_manifest.get(relative) != (
                item["bytes"],
                item["sha256"],
            ):
                raise SourcePreflightExecutionError(
                    "NumPy module differs from embedded pinned RECORD"
                )
        else:
            raise SourcePreflightExecutionError(
                "loaded numerical module has an unauthorized namespace"
            )
    required_modules = {
        "numpy",
        "numpy._core._multiarray_umath",
        "rebound",
        "rebound.simulation",
        "jxplanetx.restricted_wh_source",
        "jxplanetx.fast_wisdom_holman",
        "jxplanetx._wisdom_holman_cpu_v3",
    }
    if not required_modules.issubset(observed_names):
        raise SourcePreflightExecutionError("required loaded module receipt is absent")
    if [item["module"] for item in modules] != sorted(observed_names):
        raise SourcePreflightExecutionError(
            "loaded module manifest order changed"
        )
    libraries = runtime.get("loaded_shared_library_manifest")
    if type(libraries) is not list or not libraries:
        raise SourcePreflightExecutionError("loaded shared-library manifest changed")
    library_identities: list[tuple[str, str]] = []
    for item in libraries:
        _validate_file_receipt(item)
        if not item["locator"].startswith("LOADED_SHARED_LIBRARY:"):
            raise SourcePreflightExecutionError(
                "loaded shared-library locator changed"
            )
        library_identities.append((item["locator"], item["sha256"]))
    if (
        library_identities != sorted(library_identities)
        or len(library_identities) != len(set(library_identities))
        or len({locator for locator, _ in library_identities})
        != len(library_identities)
    ):
        raise SourcePreflightExecutionError(
            "loaded shared-library manifest order or uniqueness changed"
        )
    library_hashes = {digest for _, digest in library_identities}
    required_library_hashes = {
        NATIVE_EXTENSION_SHA256,
        REBOUND_LIBRARY_SHA256,
        *(
            item["sha256"]
            for item in modules
            if item["module"].startswith("numpy.")
            and item["locator"].endswith(".so")
        ),
    }
    if not required_library_hashes.issubset(library_hashes):
        raise SourcePreflightExecutionError(
            "required native runtime library receipt is absent"
        )
    _validate_numpy_bundled_shared_libraries(
        libraries,
        numpy_record_manifest,
    )


def _validate_coordinate_receipt(coordinate: object) -> None:
    float_fields = (
        "forward_error_ulp_factor",
        "maximum_allowed_input_relative_error_in_eps",
        "maximum_allowed_input_relative_error",
        "position_maximum_component_ulps",
        "velocity_maximum_component_ulps",
        "position_maximum_forward_error_bound_fraction",
        "velocity_maximum_forward_error_bound_fraction",
        "position_maximum_input_relative_error",
        "velocity_maximum_input_relative_error",
        "position_maximum_absolute_delta_AU",
        "velocity_maximum_absolute_delta_AU_per_year",
    )
    count_fields = (
        "position_changed_component_count",
        "velocity_changed_component_count",
        "fixed_point_iteration_count",
        "maximum_fixed_point_iteration_count",
    )
    digest_fields = (
        "raw_relative_input_sha256",
        "canonical_relative_input_sha256",
        "canonical_barycentric_input_sha256",
        "source_tile_sha256",
        "content_sha256",
    )
    expected_keys = {
        "schema",
        "policy",
        "forward_error_bound",
        *float_fields,
        *count_fields,
        *digest_fields,
        "canonicalization_idempotent",
        "tile_node_zero_exact",
    }
    if (
        type(coordinate) is not dict
        or set(coordinate) != expected_keys
        or coordinate.get("schema")
        != "jx.restricted-wh.source-coordinate-canonicalization.v2"
        or coordinate.get("policy")
        != "PRIMARY_RELATIVE_TO_GM_BARYCENTRIC_TO_PRIMARY_RELATIVE_V2"
        or coordinate.get("forward_error_bound")
        != "SUM_OF_FULL_ULPS_OF_INTERMEDIATE_PRIMARY_AND_RESULT"
        or coordinate.get("canonicalization_idempotent") is not True
        or coordinate.get("tile_node_zero_exact") is not True
    ):
        raise SourcePreflightExecutionError(
            "JX coordinate-canonicalization receipt schema changed"
        )
    floats = {
        name: _float_hex(coordinate[name], f"coordinate receipt {name}")
        for name in float_fields
    }
    if (
        floats["forward_error_ulp_factor"] != 1.0
        or floats["maximum_allowed_input_relative_error_in_eps"] != 512.0
        or floats["maximum_allowed_input_relative_error"]
        != 512.0 * sys.float_info.epsilon
        or any(floats[name] < 0.0 for name in float_fields)
        or floats["position_maximum_forward_error_bound_fraction"] > 1.0
        or floats["velocity_maximum_forward_error_bound_fraction"] > 1.0
        or floats["position_maximum_input_relative_error"]
        > floats["maximum_allowed_input_relative_error"]
        or floats["velocity_maximum_input_relative_error"]
        > floats["maximum_allowed_input_relative_error"]
    ):
        raise SourcePreflightExecutionError(
            "JX coordinate-canonicalization numerical bounds changed"
        )
    for name in count_fields:
        if type(coordinate[name]) is not int or coordinate[name] < 0:
            raise SourcePreflightExecutionError(
                "JX coordinate-canonicalization count changed"
            )
    if (
        coordinate["maximum_fixed_point_iteration_count"] != 4
        or not 1 <= coordinate["fixed_point_iteration_count"] <= 4
        or any(
            not _is_sha256(coordinate[name]) or coordinate[name] == "0" * 64
            for name in digest_fields
        )
    ):
        raise SourcePreflightExecutionError(
            "JX coordinate-canonicalization identity changed"
        )
    canonical: dict[str, Any] = {
        "schema": coordinate["schema"],
        "policy": coordinate["policy"],
        "forward_error_bound": coordinate["forward_error_bound"],
    }
    for name in float_fields:
        canonical[f"{name}_hex"] = coordinate[name]
    for name in count_fields:
        canonical[name] = coordinate[name]
    for name in digest_fields[:-1]:
        canonical[name] = coordinate[name]
    canonical["canonicalization_idempotent"] = True
    canonical["tile_node_zero_exact"] = True
    canonical_payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    expected_digest = _sha256(
        b"jx.restricted-wh.source-coordinate-canonicalization.v2\0"
        + canonical_payload
    )
    if coordinate["content_sha256"] != expected_digest:
        raise SourcePreflightExecutionError(
            "JX coordinate-canonicalization content digest changed"
        )


def _provider_coordinate_state_sha256(
    np: Any,
    domain: bytes,
    positions: Any,
    velocities: Any,
    gm: Any,
) -> str:
    if (
        type(domain) is not bytes
        or not domain
        or type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or type(gm) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or gm.dtype != np.dtype(np.float64)
        or positions.shape != (SOURCE_COUNT, 3)
        or velocities.shape != (SOURCE_COUNT, 3)
        or gm.shape != (SOURCE_COUNT,)
    ):
        raise SourcePreflightExecutionError(
            "provider coordinate digest geometry changed"
        )
    digest = hashlib.sha256()
    digest.update(domain + b"\0")
    for source_id in SOURCE_IDS:
        encoded = source_id.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    for value in (positions, velocities, gm):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _recompute_coordinate_custody(
    np: Any,
    raw_positions: Any,
    raw_velocities: Any,
    gm: Any,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Reproduce the pinned provider's bounded coordinate fixed point."""

    if (
        type(raw_positions) is not np.ndarray
        or type(raw_velocities) is not np.ndarray
        or type(gm) is not np.ndarray
        or raw_positions.dtype != np.dtype(np.float64)
        or raw_velocities.dtype != np.dtype(np.float64)
        or gm.dtype != np.dtype(np.float64)
        or raw_positions.shape != (SOURCE_COUNT, 3)
        or raw_velocities.shape != raw_positions.shape
        or gm.shape != (SOURCE_COUNT,)
    ):
        raise SourcePreflightExecutionError(
            "coordinate-custody input geometry changed"
        )
    total_gm = np.sum(gm, dtype=np.float64)
    current_positions = np.ascontiguousarray(raw_positions.copy())
    current_velocities = np.ascontiguousarray(raw_velocities.copy())
    position_forward_bounds = np.zeros_like(raw_positions)
    velocity_forward_bounds = np.zeros_like(raw_velocities)
    seen: set[tuple[bytes, bytes]] = set()
    barycentric_positions: Any = None
    barycentric_velocities: Any = None
    canonical_positions: Any = None
    canonical_velocities: Any = None
    iteration_count = 0
    for iteration_count in range(1, 5):
        state_key = (
            current_positions.tobytes(order="C"),
            current_velocities.tobytes(order="C"),
        )
        if state_key in seen:
            raise SourcePreflightExecutionError(
                "coordinate canonicalization entered a cycle"
            )
        seen.add(state_key)
        barycenter_position = (
            np.sum(current_positions * gm[:, None], axis=0, dtype=np.float64)
            / total_gm
        )
        barycenter_velocity = (
            np.sum(current_velocities * gm[:, None], axis=0, dtype=np.float64)
            / total_gm
        )
        barycentric_positions = np.ascontiguousarray(
            current_positions - barycenter_position
        )
        barycentric_velocities = np.ascontiguousarray(
            current_velocities - barycenter_velocity
        )
        canonical_positions = np.ascontiguousarray(
            barycentric_positions - barycentric_positions[0]
        )
        canonical_velocities = np.ascontiguousarray(
            barycentric_velocities - barycentric_velocities[0]
        )
        for source_index in range(SOURCE_COUNT):
            for component in range(3):
                position_forward_bounds[source_index, component] += (
                    math.ulp(barycentric_positions[source_index, component])
                    + math.ulp(barycentric_positions[0, component])
                    + math.ulp(canonical_positions[source_index, component])
                )
                velocity_forward_bounds[source_index, component] += (
                    math.ulp(barycentric_velocities[source_index, component])
                    + math.ulp(barycentric_velocities[0, component])
                    + math.ulp(canonical_velocities[source_index, component])
                )
        if _array_bitwise_equal(
            np, canonical_positions, current_positions
        ) and _array_bitwise_equal(
            np, canonical_velocities, current_velocities
        ):
            break
        current_positions = canonical_positions
        current_velocities = canonical_velocities
    else:
        raise SourcePreflightExecutionError(
            "coordinate canonicalization did not reach a fixed point"
        )

    def metrics(
        original: Any,
        canonical: Any,
        bounds: Any,
        label: str,
    ) -> tuple[float, float, float]:
        maximum_ulps = 0.0
        maximum_forward_fraction = 0.0
        maximum_relative_error = 0.0
        for original_value, canonical_value, bound_value in zip(
            original.flat,
            canonical.flat,
            bounds.flat,
            strict=True,
        ):
            before = float(original_value)
            after = float(canonical_value)
            if before == after:
                continue
            correction = abs(after - before)
            forward_bound = float(bound_value)
            if before == 0.0:
                raise SourcePreflightExecutionError(
                    f"{label} canonicalization changed an exact zero"
                )
            relative_error = correction / abs(before)
            if (
                not math.isfinite(correction)
                or not math.isfinite(forward_bound)
                or correction > forward_bound
                or not math.isfinite(relative_error)
                or relative_error > 512.0 * sys.float_info.epsilon
            ):
                raise SourcePreflightExecutionError(
                    f"{label} coordinate custody exceeded its error bound"
                )
            maximum_ulps = max(
                maximum_ulps,
                correction / math.ulp(before),
            )
            maximum_forward_fraction = max(
                maximum_forward_fraction,
                correction / forward_bound,
            )
            maximum_relative_error = max(
                maximum_relative_error,
                relative_error,
            )
        return (
            float(maximum_ulps),
            float(maximum_forward_fraction),
            float(maximum_relative_error),
        )

    position_metrics = metrics(
        raw_positions,
        canonical_positions,
        position_forward_bounds,
        "position",
    )
    velocity_metrics = metrics(
        raw_velocities,
        canonical_velocities,
        velocity_forward_bounds,
        "velocity",
    )
    expected_receipt_fields = {
        "forward_error_ulp_factor": 1.0.hex(),
        "maximum_allowed_input_relative_error_in_eps": 512.0.hex(),
        "maximum_allowed_input_relative_error": (
            512.0 * sys.float_info.epsilon
        ).hex(),
        "position_maximum_component_ulps": position_metrics[0].hex(),
        "velocity_maximum_component_ulps": velocity_metrics[0].hex(),
        "position_maximum_forward_error_bound_fraction": (
            position_metrics[1].hex()
        ),
        "velocity_maximum_forward_error_bound_fraction": (
            velocity_metrics[1].hex()
        ),
        "position_maximum_input_relative_error": position_metrics[2].hex(),
        "velocity_maximum_input_relative_error": velocity_metrics[2].hex(),
        "position_changed_component_count": int(
            np.count_nonzero(
                raw_positions.view(np.uint64)
                != canonical_positions.view(np.uint64)
            )
        ),
        "velocity_changed_component_count": int(
            np.count_nonzero(
                raw_velocities.view(np.uint64)
                != canonical_velocities.view(np.uint64)
            )
        ),
        "fixed_point_iteration_count": iteration_count,
        "maximum_fixed_point_iteration_count": 4,
        "position_maximum_absolute_delta_AU": float(
            np.max(np.abs(canonical_positions - raw_positions))
        ).hex(),
        "velocity_maximum_absolute_delta_AU_per_year": float(
            np.max(np.abs(canonical_velocities - raw_velocities))
        ).hex(),
    }
    return (
        barycentric_positions,
        barycentric_velocities,
        canonical_positions,
        canonical_velocities,
        expected_receipt_fields,
    )


def _recompute_source_tile_sha256(
    np: Any,
    protocol: Mapping[str, Any],
    steps_per_year: int,
    positions: Any,
    velocities: Any,
    gm: Any,
) -> str:
    """Recompute the provider tile digest from retained bytes, without JX imports."""

    if steps_per_year not in LANE_STEPS:
        raise SourcePreflightExecutionError("source-tile lane is not sealed")
    node_count = NODE_COUNTS[steps_per_year]
    if (
        type(positions) is not np.ndarray
        or type(velocities) is not np.ndarray
        or type(gm) is not np.ndarray
        or positions.dtype != np.dtype(np.float64)
        or velocities.dtype != np.dtype(np.float64)
        or gm.dtype != np.dtype(np.float64)
        or positions.shape != (node_count, SOURCE_COUNT, 3)
        or velocities.shape != positions.shape
        or gm.shape != (SOURCE_COUNT,)
        or not positions.flags.c_contiguous
        or not velocities.flags.c_contiguous
        or not gm.flags.c_contiguous
    ):
        raise SourcePreflightExecutionError("retained source-tile geometry changed")
    lanes = protocol.get("execution_specification", {}).get("jx_lanes")
    if type(lanes) is not list:
        raise SourcePreflightExecutionError("sealed source-tile lanes changed")
    selected = [
        lane for lane in lanes
        if type(lane) is dict
        and lane.get("steps_per_julian_year") == steps_per_year
    ]
    if len(selected) != 1 or type(selected[0].get("schedule")) is not dict:
        raise SourcePreflightExecutionError("sealed source-tile schedule changed")
    execution = protocol["execution_specification"]
    schedule_sha256 = selected[0]["schedule"].get("content_sha256")
    if not _is_sha256(schedule_sha256):
        raise SourcePreflightExecutionError("sealed source-tile schedule changed")
    metadata = {
        "axes": "IAU76_80_ECLIPTIC_J2000",
        "coordinate_system": "PRIMARY_RELATIVE_NONROTATING",
        "epoch_origin_years_hex": _float_hex(
            execution["epoch_origin_julian_years_since_j2000_hex"],
            "source epoch origin",
        ).hex(),
        "frame": protocol["initial_state"]["frame"],
        "global_source_step": 0,
        "length_unit": "AU",
        "primary_source_id": SOURCE_IDS[0],
        "provenance": {
            "citation": (
                "Sealed common-epoch reduced-five source preflight protocol "
                + PROTOCOL_CONTENT_SHA256
            ),
            "sha256": schedule_sha256,
            "source_id": f"jx-tno-1kyr-reduced5-source-h{steps_per_year}",
            "version": "source-preflight-executor-v1",
        },
        "source_ids": list(SOURCE_IDS),
        "source_schedule_sha256": schedule_sha256,
        "source_step_years_hex": (1.0 / steps_per_year).hex(),
        "tile_id": f"tno-1kyr-reduced5-source-h{steps_per_year}",
        "time_scale": "TDB",
        "time_unit": "year",
        "unit_system_id": "AU-YEAR",
    }
    metadata_payload = json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    total_gm = math.fsum(float(value) for value in gm)
    acceleration_bound = math.nextafter(2.0 * total_gm, math.inf)
    arrays = (
        ("source_relative_positions_AU", positions),
        ("source_relative_velocities_AU_per_year", velocities),
        (
            "source_gm_AU3_per_year2",
            np.ascontiguousarray(gm, dtype=np.float64),
        ),
        (
            "source_encounter_radii_AU",
            np.ones(SOURCE_COUNT, dtype=np.float64),
        ),
        (
            "certified_relative_acceleration_bounds_AU_per_year2",
            np.full(SOURCE_COUNT, acceleration_bound, dtype=np.float64),
        ),
    )
    digest = hashlib.sha256()
    digest.update(SOURCE_TILE_DOMAIN)
    digest.update(len(metadata_payload).to_bytes(8, "little"))
    digest.update(metadata_payload)
    for label, array in arrays:
        encoded_label = label.encode("ascii")
        digest.update(len(encoded_label).to_bytes(4, "little"))
        digest.update(encoded_label)
        digest.update(array.ndim.to_bytes(4, "little"))
        for dimension in array.shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        raw = array.tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_coordinate_custody_fields(
    coordinate: object,
    expected_fields: Mapping[str, Any],
) -> None:
    if type(coordinate) is not dict or any(
        coordinate.get(name) != expected
        for name, expected in expected_fields.items()
    ):
        raise SourcePreflightExecutionError(
            "JX coordinate-canonicalization derived metrics changed"
        )


def _validate_lane_receipts(
    np: Any,
    result: Mapping[str, Any],
    protocol: Mapping[str, Any],
    gm: Any,
    canonical_positions: Any,
    canonical_velocities: Any,
    h64_positions: Any,
    h64_velocities: Any,
    h128_positions: Any,
    h128_velocities: Any,
) -> None:
    lanes = result.get("lanes")
    if type(lanes) is not dict or set(lanes) != {
        "jx_h64", "jx_h128", "rebound511_ias15"
    }:
        raise SourcePreflightExecutionError("result lane roster changed")
    protocol_lanes = {
        lane["steps_per_julian_year"]: lane
        for lane in protocol["execution_specification"]["jx_lanes"]
    }
    binding_digests = {
        binding["relative_path"]: binding["sha256"]
        for binding in protocol["input_bindings"]
    }
    expected_jx_keys = {
        "lane_id", "engine", "steps_per_julian_year",
        "fixed_step_julian_years_hex", "source_step_count",
        "source_node_count", "schedule_content_sha256",
        "tile_content_sha256", "raw_relative_input_sha256",
        "source_map_input_content_sha256", "source_map_result_content_sha256",
        "source_map_verification_content_sha256", "source_map_metrics_hex",
        "coordinate_canonicalization",
        "coordinate_canonicalization_content_sha256",
        "acceleration_bound_certificate", "exact_replay_verified",
        "fresh_workspace_replay_verified", "fresh_workspace_receipt_domain",
        "fresh_workspace_primary_receipt", "fresh_workspace_replay_receipt",
        "fresh_workspace_primary_receipt_sha256",
        "fresh_workspace_replay_receipt_sha256", "public_builder_call_count",
        "internal_primary_execution_count", "internal_replay_execution_count",
        "total_native_map_execution_count", "retained_stream",
    }
    total_gm = math.fsum(float(value) for value in gm)
    expected_bound = math.nextafter(2.0 * total_gm, math.inf).hex()
    raw_gm, raw_positions, raw_velocities = _initial_arrays(np, protocol)
    raw_input_sha256 = _provider_coordinate_state_sha256(
        np,
        b"jx.restricted-wh.source-provider.input.v1",
        raw_positions,
        raw_velocities,
        raw_gm,
    )
    (
        provider_barycentric_positions,
        provider_barycentric_velocities,
        recomputed_canonical_positions,
        recomputed_canonical_velocities,
        expected_coordinate_fields,
    ) = _recompute_coordinate_custody(
        np,
        raw_positions,
        raw_velocities,
        raw_gm,
    )
    if not (
        _array_bitwise_equal(
            np, canonical_positions, recomputed_canonical_positions
        )
        and _array_bitwise_equal(
            np, canonical_velocities, recomputed_canonical_velocities
        )
    ):
        raise SourcePreflightExecutionError(
            "retained node zero differs from provider coordinate fixed point"
        )
    canonical_relative_sha256 = _provider_coordinate_state_sha256(
        np,
        b"jx.restricted-wh.source-provider.input.v1",
        recomputed_canonical_positions,
        recomputed_canonical_velocities,
        gm,
    )
    canonical_barycentric_sha256 = _provider_coordinate_state_sha256(
        np,
        b"jx.restricted-wh.source-provider.canonical-barycentric-input.v1",
        provider_barycentric_positions,
        provider_barycentric_velocities,
        gm,
    )
    retained_streams = {
        64: (h64_positions, h64_velocities),
        128: (h128_positions, h128_velocities),
    }
    expected_tile_sha256 = {
        steps: _recompute_source_tile_sha256(
            np,
            protocol,
            steps,
            retained_streams[steps][0],
            retained_streams[steps][1],
            gm,
        )
        for steps in LANE_STEPS
    }
    jx_records: dict[int, dict[str, Any]] = {}
    for steps, key in ((64, "jx_h64"), (128, "jx_h128")):
        lane = lanes[key]
        sealed = protocol_lanes[steps]
        if (
            type(lane) is not dict
            or set(lane) != expected_jx_keys
            or lane.get("lane_id") != f"jx_ordered_jacobi_h{steps}"
            or lane.get("engine") != "JX_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN"
            or lane.get("steps_per_julian_year") != steps
            or lane.get("fixed_step_julian_years_hex") != (1.0 / steps).hex()
            or lane.get("source_step_count") != HORIZON_JULIAN_YEARS * steps
            or lane.get("source_node_count") != NODE_COUNTS[steps]
            or lane.get("schedule_content_sha256")
            != sealed["schedule"]["content_sha256"]
            or lane.get("tile_content_sha256") != expected_tile_sha256[steps]
            or lane.get("exact_replay_verified") is not True
            or lane.get("fresh_workspace_replay_verified") is not True
            or lane.get("fresh_workspace_receipt_domain")
            != "jx.tno.1kyr-reduced5-h128-source-preflight.fresh-workspace-replay.v1"
            or lane.get("public_builder_call_count") != 2
            or lane.get("internal_primary_execution_count") != 2
            or lane.get("internal_replay_execution_count") != 2
            or lane.get("total_native_map_execution_count") != 4
            or lane.get("retained_stream")
            != "FIRST_FRESH_WORKSPACE_PRIMARY_AFTER_INTERNAL_AND_OUTER_BITWISE_REPLAY"
        ):
            raise SourcePreflightExecutionError("JX execution receipt changed")
        for digest_key in (
            "tile_content_sha256", "raw_relative_input_sha256",
            "source_map_input_content_sha256", "source_map_result_content_sha256",
            "source_map_verification_content_sha256",
            "coordinate_canonicalization_content_sha256",
            "fresh_workspace_primary_receipt_sha256",
            "fresh_workspace_replay_receipt_sha256",
        ):
            if not _is_sha256(lane.get(digest_key)):
                raise SourcePreflightExecutionError("JX execution digest changed")
        if (
            lane["fresh_workspace_primary_receipt_sha256"]
            != lane["fresh_workspace_replay_receipt_sha256"]
        ):
            raise SourcePreflightExecutionError("fresh-workspace receipts differ")
        primary_fresh = lane.get("fresh_workspace_primary_receipt")
        replay_fresh = lane.get("fresh_workspace_replay_receipt")
        replay_domain = (
            b"jx.tno.1kyr-reduced5-h128-source-preflight."
            b"fresh-workspace-replay.v1\0"
        )
        if (
            type(primary_fresh) is not dict
            or type(replay_fresh) is not dict
            or primary_fresh != replay_fresh
            or _sha256(replay_domain + _canonical_json(primary_fresh))
            != lane["fresh_workspace_primary_receipt_sha256"]
            or _sha256(replay_domain + _canonical_json(replay_fresh))
            != lane["fresh_workspace_replay_receipt_sha256"]
        ):
            raise SourcePreflightExecutionError(
                "fresh-workspace replay evidence changed"
            )
        metrics = lane.get("source_map_metrics_hex")
        _validate_source_map_metrics_hex(metrics)
        coordinate = lane.get("coordinate_canonicalization")
        _validate_coordinate_receipt(coordinate)
        _validate_coordinate_custody_fields(
            coordinate,
            expected_coordinate_fields,
        )
        if (
            coordinate.get("content_sha256")
            != lane["coordinate_canonicalization_content_sha256"]
            or coordinate.get("source_tile_sha256") != lane["tile_content_sha256"]
            or coordinate.get("raw_relative_input_sha256")
            != lane["raw_relative_input_sha256"]
            or coordinate.get("canonicalization_idempotent") is not True
            or coordinate.get("tile_node_zero_exact") is not True
            or not _is_sha256(coordinate.get("canonical_relative_input_sha256"))
            or not _is_sha256(coordinate.get("canonical_barycentric_input_sha256"))
            or coordinate.get("raw_relative_input_sha256")
            != raw_input_sha256
            or coordinate.get("canonical_relative_input_sha256")
            != canonical_relative_sha256
            or coordinate.get("canonical_barycentric_input_sha256")
            != canonical_barycentric_sha256
        ):
            raise SourcePreflightExecutionError(
                "JX coordinate-canonicalization receipt changed"
            )
        expected_fresh_keys = {
            "tile_content_sha256", "input_content_sha256",
            "source_map_input_content_sha256",
            "source_map_result_content_sha256",
            "source_map_verification_content_sha256", "source_map_metrics",
            "coordinate_canonicalization", "runtime_identity",
        }
        expected_metric_pairs = [
            [metric["name"], metric["value_hex"]] for metric in metrics
        ]
        runtime_identity = primary_fresh.get("runtime_identity")
        source_map_runtime = (
            runtime_identity.get("source_map_runtime_identity")
            if type(runtime_identity) is dict
            else None
        )
        expected_source_map_runtime = {
            "kernel_id": (
                "jx.wisdom-holman.native-complete-map."
                "checkpoint-sync.prototype.v3"
            ),
            "g_kernel_id": (
                "jx.wisdom-holman.native-g-bundle.early-stop.prototype.v2"
            ),
            "exact_fallback_kernel_id": (
                "jx.wisdom-holman.native-g-bundle.prototype.v1"
            ),
            "scientific_claim_state": "SCREENING_ONLY",
            "maximum_bodies": 32,
            "metric_count": 17,
            "counter_count": 12,
            "compiler_version": "15.2.0",
            "compiler_flags": "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm",
            "release_wrapper_source_sha256": binding_digests[
                "src/jxplanetx/_wisdom_holman_cpu_v3_module.c"
            ],
            "release_loop_source_sha256": binding_digests[
                "src/jxplanetx/_wisdom_holman_loop_v3.c"
            ],
            "release_base_loop_source_sha256": binding_digests[
                "src/jxplanetx/_wisdom_holman_loop_v2.c"
            ],
            "extension_binary": {
                "locator": f"PROJECT_ROOT:{NATIVE_EXTENSION_RELATIVE_PATH}",
                "size_bytes": NATIVE_EXTENSION_BYTES,
                "sha256": NATIVE_EXTENSION_SHA256,
            },
        }

        def contains_absolute_path(item: object) -> bool:
            if type(item) is str:
                return item.startswith("/")
            if type(item) is dict:
                return any(
                    contains_absolute_path(key)
                    or contains_absolute_path(nested)
                    for key, nested in item.items()
                )
            if type(item) is list:
                return any(contains_absolute_path(nested) for nested in item)
            return False

        if (
            set(primary_fresh) != expected_fresh_keys
            or primary_fresh["tile_content_sha256"]
            != lane["tile_content_sha256"]
            or primary_fresh["input_content_sha256"]
            != lane["raw_relative_input_sha256"]
            or primary_fresh["source_map_input_content_sha256"]
            != lane["source_map_input_content_sha256"]
            or primary_fresh["source_map_result_content_sha256"]
            != lane["source_map_result_content_sha256"]
            or primary_fresh["source_map_verification_content_sha256"]
            != lane["source_map_verification_content_sha256"]
            or primary_fresh["source_map_metrics"] != expected_metric_pairs
            or primary_fresh["coordinate_canonicalization"] != coordinate
            or type(runtime_identity) is not dict
            or set(runtime_identity)
            != {
                "provider_id", "provider_implementation_source_sha256",
                "python_version", "platform", "numpy_version",
                "source_map_runtime_identity", "exact_replay_required",
                "coordinate_canonicalization",
                "coordinate_canonicalization_receipt_sha256",
                "source_tile_sha256",
            }
            or runtime_identity.get("provider_id")
            != "jx.restricted-wh.source-provider.ordered-jacobi.v1"
            or runtime_identity.get("provider_implementation_source_sha256")
            != binding_digests["src/jxplanetx/restricted_wh_source.py"]
            or runtime_identity.get("python_version") != PINNED_PYTHON_VERSION
            or type(runtime_identity.get("platform")) is not str
            or not runtime_identity["platform"]
            or runtime_identity.get("numpy_version") != PINNED_NUMPY_VERSION
            or source_map_runtime != expected_source_map_runtime
            or runtime_identity.get("exact_replay_required") is not True
            or runtime_identity.get("coordinate_canonicalization") != coordinate
            or runtime_identity.get("source_tile_sha256")
            != lane["tile_content_sha256"]
            or runtime_identity.get(
                "coordinate_canonicalization_receipt_sha256"
            )
            != lane["coordinate_canonicalization_content_sha256"]
            or contains_absolute_path(runtime_identity)
        ):
            raise SourcePreflightExecutionError(
                "fresh-workspace replay linkage changed"
            )
        acceleration = lane.get("acceleration_bound_certificate")
        if acceleration != {
            "formula": "NEXTUP_OF_2_TIMES_TOTAL_REDUCED_GM_DIVIDED_BY_ONE_AU_SQUARED",
            "value_in_source_order_hex": [expected_bound] * SOURCE_COUNT,
        }:
            raise SourcePreflightExecutionError(
                "JX acceleration-bound certificate changed"
            )
        jx_records[steps] = lane
    h64_coordinate = jx_records[64]["coordinate_canonicalization"]
    h128_coordinate = jx_records[128]["coordinate_canonicalization"]
    if any(
        h64_coordinate[name] != h128_coordinate[name]
        for name in (
            "raw_relative_input_sha256",
            "canonical_relative_input_sha256",
            "canonical_barycentric_input_sha256",
        )
    ):
        raise SourcePreflightExecutionError("H64/H128 canonical input receipt differs")
    ias = lanes["rebound511_ias15"]
    expected_ias_keys = {
        "lane_id", "engine", "model", "initial_effective_configuration",
        "final_effective_configuration", "integration_call_count",
        "annual_checkpoint_count", "heartbeat_binding",
        "heartbeat_callback_count", "finite_roster_observation_count",
        "heartbeat_callback_exception", "force_and_step_implementation",
    }
    if (
        type(ias) is not dict
        or set(ias) != expected_ias_keys
        or ias.get("lane_id") != "rebound511_ias15_reduced5"
        or ias.get("engine") != "REBOUND_5.1.1_IAS15"
        or ias.get("model") != "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS"
        or ias.get("integration_call_count") != HORIZON_JULIAN_YEARS
        or ias.get("annual_checkpoint_count") != CHECKPOINT_COUNT
        or ias.get("heartbeat_binding")
        != "PINNED_REBOUND_5_1_1_C_FIELD_WITH_LOCAL_CALLBACK_REFERENCE_DUE_MISSING_HB_SLOT"
        or type(ias.get("heartbeat_callback_count")) is not int
        or ias["heartbeat_callback_count"] <= 0
        or type(ias.get("finite_roster_observation_count")) is not int
        or ias["finite_roster_observation_count"]
        != ias["heartbeat_callback_count"] + 1
        or ias.get("heartbeat_callback_exception") is not None
        or ias.get("force_and_step_implementation")
        != "INDEPENDENT_REBOUND_5_1_1_NO_JX_FORCE_OR_STEP_CODE"
    ):
        raise SourcePreflightExecutionError("IAS15 execution receipt changed")
    expected_initial = {
        "integrator": "ias15", "epsilon_hex": (1.0e-13).hex(),
        "minimum_dt_julian_years_hex": "0x0.0p+0", "adaptive_mode": "PRS23",
        "initial_dt_julian_years_hex": (1.0 / 32.0).hex(),
        "initial_time_julian_years_hex": "0x0.0p+0",
        "G_hex": "0x1.0000000000000p+0", "gravity": "basic",
        "collision": "none", "boundary": "none", "softening_hex": "0x0.0p+0",
        "additional_forces_c_pointer_is_null": True,
        "force_is_velocity_dependent": 0,
        "exit_min_distance_AU_hex": "0x0.0p+0",
        "exit_max_distance_AU_hex": "0x0.0p+0",
        "N": SOURCE_COUNT, "N_active": SOURCE_COUNT, "testparticle_type": 0,
    }
    if ias.get("initial_effective_configuration") != expected_initial:
        raise SourcePreflightExecutionError("IAS15 initial configuration changed")
    final = ias.get("final_effective_configuration")
    if (
        type(final) is not dict
        or set(final) != {
            "integrator", "epsilon_hex", "minimum_dt_julian_years_hex",
            "adaptive_mode", "final_time_julian_years_hex", "steps_done",
            "G_hex", "gravity", "collision", "boundary", "softening_hex",
            "additional_forces_c_pointer_is_null", "force_is_velocity_dependent",
            "exit_min_distance_AU_hex", "exit_max_distance_AU_hex",
            "N", "N_active", "testparticle_type",
        }
        or final.get("integrator") != "ias15"
        or final.get("epsilon_hex") != (1.0e-13).hex()
        or final.get("minimum_dt_julian_years_hex") != "0x0.0p+0"
        or final.get("adaptive_mode") != "PRS23"
        or final.get("final_time_julian_years_hex")
        != float(HORIZON_JULIAN_YEARS).hex()
        or type(final.get("steps_done")) is not int
        or final["steps_done"] <= 0
        or final.get("G_hex") != "0x1.0000000000000p+0"
        or final.get("gravity") != "basic"
        or final.get("collision") != "none"
        or final.get("boundary") != "none"
        or final.get("softening_hex") != "0x0.0p+0"
        or final.get("additional_forces_c_pointer_is_null") is not True
        or final.get("force_is_velocity_dependent") != 0
        or final.get("exit_min_distance_AU_hex") != "0x0.0p+0"
        or final.get("exit_max_distance_AU_hex") != "0x0.0p+0"
        or final.get("N") != SOURCE_COUNT
        or final.get("N_active") != SOURCE_COUNT
        or final.get("testparticle_type") != 0
    ):
        raise SourcePreflightExecutionError("IAS15 final configuration changed")
    initial_binding = result.get("initial_state_binding")
    expected_binding = {
        "preregistered_initial_state_content_sha256": protocol["initial_state"][
            "content_sha256"
        ],
        "raw_relative_input_sha256": h128_coordinate["raw_relative_input_sha256"],
        "canonical_relative_input_sha256": h128_coordinate[
            "canonical_relative_input_sha256"
        ],
        "canonical_barycentric_input_sha256": h128_coordinate[
            "canonical_barycentric_input_sha256"
        ],
        "h64_coordinate_canonicalization_receipt_sha256": jx_records[64][
            "coordinate_canonicalization_content_sha256"
        ],
        "h128_coordinate_canonicalization_receipt_sha256": jx_records[128][
            "coordinate_canonicalization_content_sha256"
        ],
        "h64_h128_node_zero_bitwise_identical": True,
        "ias15_initialized_from_owned_canonical_node_zero_copies": True,
        "h64_h128_ias15_year_zero_barycentric_bitwise_identical": True,
    }
    if initial_binding != expected_binding:
        raise SourcePreflightExecutionError("canonical initial-state binding changed")


def validate_result(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    verify_repository_authority: bool = True,
) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    _secure_root(root, expected_roster=SUCCESS_ROSTER)
    if verify_repository_authority:
        protocol, sealed_payload, authority = _load_sealed_authority()
    else:
        sealed_payload = _secure_result_bytes(root, "PROTOCOL.json")
        protocol = _strict_json(sealed_payload, "embedded protocol")
        authority = {
            "protocol_relative_path": SEALED_PROTOCOL_RELATIVE_PATH,
            "protocol_file_sha256": PROTOCOL_FILE_SHA256,
            "protocol_content_sha256": PROTOCOL_CONTENT_SHA256,
            "preregistration_commit": PREREGISTRATION_COMMIT,
            "preregistration_tree": PREREGISTRATION_TREE,
            "preregistration_tag": PREREGISTRATION_TAG,
            "preregistration_tag_object": PREREGISTRATION_TAG_OBJECT,
            "preregistration_tag_payload_bytes": PREREGISTRATION_TAG_PAYLOAD_BYTES,
            "preregistration_tag_payload_sha256": PREREGISTRATION_TAG_PAYLOAD_SHA256,
            "input_binding_count": len(protocol.get("input_bindings", [])),
            "every_input_binding_verified": True,
        }
    embedded_protocol = _secure_result_bytes(
        root, "PROTOCOL.json", expected_bytes=PROTOCOL_BYTES
    )
    if embedded_protocol != sealed_payload or _sha256(embedded_protocol) != PROTOCOL_FILE_SHA256:
        raise SourcePreflightExecutionError("embedded protocol differs from authority")
    runtime_payload = _secure_result_bytes(root, "RUNTIME.json")
    runtime = _strict_json(runtime_payload, "runtime receipt")
    if (
        runtime_payload != _canonical_json(runtime)
        or runtime.get("schema") != RUNTIME_SCHEMA
        or runtime.get("content_sha256") != _domain_hash(RUNTIME_DOMAIN, runtime)
        or runtime.get("authority") != authority
        or runtime.get("gpu_runtime_imported") is not False
        or runtime.get("cuda_device_queried") is not False
        or runtime.get("dynamics_executed_at_receipt_time") is not False
    ):
        raise SourcePreflightExecutionError("runtime receipt differs")
    _validate_runtime_record(runtime, authority)
    result_payload = _secure_result_bytes(root, "RESULT.json")
    result = _strict_json(result_payload, "source-preflight result")
    if (
        result_payload != _canonical_json(result)
        or set(result) != {
            "schema", "status", "evidence_class", "claim_ceiling",
            "authority", "successor_implementation", "runtime_content_sha256",
            "initial_state_binding", "lanes", "invariants", "gates",
            "artifacts", "claims", "nonclaims", "content_sha256",
        }
        or result.get("schema") != RESULT_SCHEMA
        or result.get("status")
        != "PASSED_BOUNDED_AUTONOMOUS_REDUCED5_SOURCE_PREFLIGHT"
        or result.get("content_sha256") != _domain_hash(RESULT_DOMAIN, result)
        or result.get("authority") != authority
        or result.get("runtime_content_sha256") != runtime.get("content_sha256")
        or result.get("claims") != CLAIMS
        or result.get("successor_implementation")
        != runtime.get("successor_implementation")
        or result.get("evidence_class")
        != "PASSED_1KYR_AUTONOMOUS_REDUCED5_SOURCE_NUMERICAL_PREFLIGHT_NO_TRACER_NO_SCIENCE_RESULT"
        or result.get("claim_ceiling")
        != "BOUNDED_SOURCE_PREFLIGHT_PASS_NO_TRACER_QUALIFICATION_NO_LONG_HORIZON_AUTHORITY_NO_SCIENCE_RESULT"
        or result.get("nonclaims") != list(NONCLAIMS)
    ):
        raise SourcePreflightExecutionError("source-preflight result identity differs")
    artifacts = result.get("artifacts")
    if type(artifacts) is not dict or set(artifacts) != set(EVIDENCE_ROSTER):
        raise SourcePreflightExecutionError("result artifact manifest changed")
    payloads: dict[str, bytes] = {"PROTOCOL.json": embedded_protocol, "RUNTIME.json": runtime_payload}
    for name, expected_bytes in EXPECTED_BINARY_BYTES.items():
        payloads[name] = _secure_result_bytes(root, name, expected_bytes=expected_bytes)
    for name in EVIDENCE_ROSTER:
        payload = payloads[name]
        receipt = artifacts.get(name)
        if (
            type(receipt) is not dict
            or set(receipt) != {"bytes", "sha256", "role"}
            or receipt.get("bytes") != len(payload)
            or receipt.get("sha256") != _sha256(payload)
            or receipt.get("role") != ARTIFACT_ROLES[name]
        ):
            raise SourcePreflightExecutionError(
                f"result artifact receipt changed for {name}"
            )
    if sum(len(payloads[name]) for name in EXPECTED_BINARY_BYTES) != FIXED_BINARY_EVIDENCE_BYTES:
        raise SourcePreflightExecutionError("fixed binary evidence byte total changed")
    try:
        np = importlib.import_module("numpy")
    except Exception as exc:
        raise SourcePreflightExecutionError(
            "NumPy is required for offline evidence validation"
        ) from exc
    gm = np.frombuffer(payloads["SOURCE_GM.bin"], dtype=np.dtype("<f8")).copy()

    def source_stream(name: str, count: int) -> tuple[Any, Any]:
        values = np.frombuffer(payloads[name], dtype=np.dtype("<f8"))
        if values.size != count * SOURCE_STATE_WIDTH:
            raise SourcePreflightExecutionError(f"{name} element count changed")
        states = values.reshape(count, SOURCE_COUNT, 6)
        return (
            np.ascontiguousarray(states[:, :, :3], dtype=np.float64),
            np.ascontiguousarray(states[:, :, 3:], dtype=np.float64),
        )

    h64_positions, h64_velocities = source_stream(
        "JX_H64_SOURCE_NODES.bin", NODE_COUNTS[64]
    )
    h128_positions, h128_velocities = source_stream(
        "JX_H128_SOURCE_NODES.bin", NODE_COUNTS[128]
    )
    ias_positions, ias_velocities = source_stream(
        "IAS15_ANNUAL_SOURCE_STATES.bin", CHECKPOINT_COUNT
    )
    ias15_annual = np.empty((CHECKPOINT_COUNT, SOURCE_COUNT, 6), dtype=np.float64)
    ias15_annual[:, :, :3] = ias_positions
    ias15_annual[:, :, 3:] = ias_velocities
    h64_invariants = np.frombuffer(
        payloads["JX_H64_ALL_NODE_INVARIANTS.bin"], dtype=np.dtype("<f8")
    ).reshape(NODE_COUNTS[64], INVARIANT_WIDTH).copy()
    h128_invariants = np.frombuffer(
        payloads["JX_H128_ALL_NODE_INVARIANTS.bin"], dtype=np.dtype("<f8")
    ).reshape(NODE_COUNTS[128], INVARIANT_WIDTH).copy()
    expected_gm, _, _ = _initial_arrays(np, protocol)
    if not _array_bitwise_equal(np, gm, expected_gm):
        raise SourcePreflightExecutionError("retained source GM differs")
    _validate_lane_receipts(
        np,
        result,
        protocol,
        gm,
        h64_positions[0],
        h64_velocities[0],
        h64_positions,
        h64_velocities,
        h128_positions,
        h128_velocities,
    )
    if not (
        _array_bitwise_equal(np, h64_positions[0], h128_positions[0])
        and _array_bitwise_equal(np, h64_velocities[0], h128_velocities[0])
        and _array_bitwise_equal(np, h64_positions[0], ias15_annual[0, :, :3])
        and _array_bitwise_equal(np, h64_velocities[0], ias15_annual[0, :, 3:])
    ):
        raise SourcePreflightExecutionError(
            "retained lanes do not share the canonical node-zero state"
        )
    recomputed_h64, h64_summary = _invariant_stream(
        np, h64_positions, h64_velocities, gm
    )
    recomputed_h128, h128_summary = _invariant_stream(
        np, h128_positions, h128_velocities, gm
    )
    if not _array_bitwise_equal(np, recomputed_h64, h64_invariants) or not _array_bitwise_equal(
        np, recomputed_h128, h128_invariants
    ):
        raise SourcePreflightExecutionError(
            "retained all-node invariants differ from source evidence"
        )
    gates = _evaluate_numeric_gates(
        np,
        gm=gm,
        h64_positions=h64_positions,
        h64_velocities=h64_velocities,
        h128_positions=h128_positions,
        h128_velocities=h128_velocities,
        h64_invariants=h64_invariants,
        h128_invariants=h128_invariants,
        ias15_annual=ias15_annual,
    )
    if (
        result.get("gates") != gates
        or gates["all_applicable_source_predicates"]["pass"] is not True
        or result.get("invariants")
        != {
            "jx_h64": h64_summary,
            "jx_h128": h128_summary,
            "sampling": "EVERY_FIXED_NODE_INCLUDING_BOTH_ENDPOINTS",
        }
    ):
        raise SourcePreflightExecutionError(
            "offline gate or invariant re-evaluation differs"
        )
    for lane_name in ("jx_h64", "jx_h128"):
        lane = result.get("lanes", {}).get(lane_name)
        if (
            type(lane) is not dict
            or lane.get("exact_replay_verified") is not True
            or lane.get("fresh_workspace_replay_verified") is not True
            or lane.get("public_builder_call_count") != 2
            or lane.get("total_native_map_execution_count") != 4
        ):
            raise SourcePreflightExecutionError("JX replay receipt changed")
    ias_lane = result.get("lanes", {}).get("rebound511_ias15")
    if (
        type(ias_lane) is not dict
        or ias_lane.get("engine") != "REBOUND_5.1.1_IAS15"
        or ias_lane.get("integration_call_count") != HORIZON_JULIAN_YEARS
        or ias_lane.get("annual_checkpoint_count") != CHECKPOINT_COUNT
        or ias_lane.get("heartbeat_callback_exception") is not None
        or type(ias_lane.get("heartbeat_callback_count")) is not int
        or ias_lane["heartbeat_callback_count"] <= 0
    ):
        raise SourcePreflightExecutionError("IAS15 execution receipt changed")
    _secure_root(root, expected_roster=SUCCESS_ROSTER)
    return result


def execute(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _require_safe_absent_output(output_root)
    protocol, protocol_payload, authority = _load_sealed_authority()
    context = _runtime_preflight(authority)
    h64_build, h64_receipt = _run_jx_lane(context, protocol, 64)
    h128_build, h128_receipt = _run_jx_lane(context, protocol, 128)
    np = context.np
    h64_tile = h64_build.tile
    h128_tile = h128_build.tile
    gm = np.ascontiguousarray(h64_tile.source_gm_AU3_per_year2, dtype=np.float64)
    if not (
        _array_bitwise_equal(np, gm, h128_tile.source_gm_AU3_per_year2)
        and _array_bitwise_equal(
            np,
            h64_tile.source_relative_positions_AU[0],
            h128_tile.source_relative_positions_AU[0],
        )
        and _array_bitwise_equal(
            np,
            h64_tile.source_relative_velocities_AU_per_year[0],
            h128_tile.source_relative_velocities_AU_per_year[0],
        )
        and h64_build.coordinate_canonicalization.canonical_relative_input_sha256
        == h128_build.coordinate_canonicalization.canonical_relative_input_sha256
        and h64_build.coordinate_canonicalization.raw_relative_input_sha256
        == h128_build.coordinate_canonicalization.raw_relative_input_sha256
    ):
        raise SourcePreflightExecutionError(
            "H64 and H128 do not share one canonical node-zero state"
        )
    canonical_positions = np.array(
        h128_tile.source_relative_positions_AU[0],
        dtype=np.float64,
        order="C",
        copy=True,
    )
    canonical_velocities = np.array(
        h128_tile.source_relative_velocities_AU_per_year[0],
        dtype=np.float64,
        order="C",
        copy=True,
    )
    ias15_annual, ias15_receipt = _run_ias15(
        context,
        canonical_positions.copy(order="C"),
        canonical_velocities.copy(order="C"),
        gm.copy(order="C"),
    )
    if not (
        _array_bitwise_equal(np, canonical_positions, ias15_annual[0, :, :3])
        and _array_bitwise_equal(np, canonical_velocities, ias15_annual[0, :, 3:])
    ):
        raise SourcePreflightExecutionError(
            "IAS15 year-zero state differs from the canonical JX node-zero state"
        )
    h64_invariants, h64_invariant_summary = _invariant_stream(
        np,
        h64_tile.source_relative_positions_AU,
        h64_tile.source_relative_velocities_AU_per_year,
        gm,
    )
    h128_invariants, h128_invariant_summary = _invariant_stream(
        np,
        h128_tile.source_relative_positions_AU,
        h128_tile.source_relative_velocities_AU_per_year,
        gm,
    )
    gates = _evaluate_numeric_gates(
        np,
        gm=gm,
        h64_positions=h64_tile.source_relative_positions_AU,
        h64_velocities=h64_tile.source_relative_velocities_AU_per_year,
        h128_positions=h128_tile.source_relative_positions_AU,
        h128_velocities=h128_tile.source_relative_velocities_AU_per_year,
        h64_invariants=h64_invariants,
        h128_invariants=h128_invariants,
        ias15_annual=ias15_annual,
    )
    if gates["all_applicable_source_predicates"]["pass"] is not True:
        raise SourcePreflightExecutionError(
            "source preflight did not pass every frozen predicate; no result published"
        )
    initial_state_binding = {
        "preregistered_initial_state_content_sha256": protocol["initial_state"][
            "content_sha256"
        ],
        "raw_relative_input_sha256": (
            h128_build.coordinate_canonicalization.raw_relative_input_sha256
        ),
        "canonical_relative_input_sha256": (
            h128_build.coordinate_canonicalization.canonical_relative_input_sha256
        ),
        "canonical_barycentric_input_sha256": (
            h128_build.coordinate_canonicalization.canonical_barycentric_input_sha256
        ),
        "h64_coordinate_canonicalization_receipt_sha256": (
            h64_build.coordinate_canonicalization.content_sha256
        ),
        "h128_coordinate_canonicalization_receipt_sha256": (
            h128_build.coordinate_canonicalization.content_sha256
        ),
        "h64_h128_node_zero_bitwise_identical": True,
        "ias15_initialized_from_owned_canonical_node_zero_copies": True,
        "h64_h128_ias15_year_zero_barycentric_bitwise_identical": True,
    }
    binary_payloads = {
        "SOURCE_GM.bin": _float64_payload(np, gm, (SOURCE_COUNT,)),
        "JX_H64_SOURCE_NODES.bin": _source_state_payload(
            np,
            h64_tile.source_relative_positions_AU,
            h64_tile.source_relative_velocities_AU_per_year,
        ),
        "JX_H128_SOURCE_NODES.bin": _source_state_payload(
            np,
            h128_tile.source_relative_positions_AU,
            h128_tile.source_relative_velocities_AU_per_year,
        ),
        "JX_H64_ALL_NODE_INVARIANTS.bin": _float64_payload(
            np, h64_invariants, (NODE_COUNTS[64], INVARIANT_WIDTH)
        ),
        "JX_H128_ALL_NODE_INVARIANTS.bin": _float64_payload(
            np, h128_invariants, (NODE_COUNTS[128], INVARIANT_WIDTH)
        ),
        "IAS15_ANNUAL_SOURCE_STATES.bin": _float64_payload(
            np, ias15_annual, (CHECKPOINT_COUNT, SOURCE_COUNT, 6)
        ),
    }
    if (
        {name: len(payload) for name, payload in binary_payloads.items()}
        != EXPECTED_BINARY_BYTES
        or sum(len(payload) for payload in binary_payloads.values())
        != FIXED_BINARY_EVIDENCE_BYTES
    ):
        raise SourcePreflightExecutionError("fixed binary evidence geometry changed")
    runtime_payload = _canonical_json(context.runtime_record)
    staged_payloads: dict[str, bytes] = {
        "PROTOCOL.json": protocol_payload,
        **binary_payloads,
        "RUNTIME.json": runtime_payload,
    }
    artifacts = {
        name: _artifact_receipt(staged_payloads[name], ARTIFACT_ROLES[name])
        for name in EVIDENCE_ROSTER
    }
    result = _build_result(
        authority=authority,
        runtime=context.runtime_record,
        artifacts=artifacts,
        h64_receipt=h64_receipt,
        h128_receipt=h128_receipt,
        ias15_receipt=ias15_receipt,
        h64_invariant_summary=h64_invariant_summary,
        h128_invariant_summary=h128_invariant_summary,
        gates=gates,
        initial_state_binding=initial_state_binding,
    )
    staged_payloads["RESULT.json"] = _canonical_json(result)
    stage = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.partial-", dir=root.parent)
    )
    publication_committed = False
    try:
        os.chmod(stage, 0o700)
        for name in SUCCESS_ROSTER:
            _write_create_only(stage / name, staged_payloads[name])
        stage_fd = os.open(stage, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)
        validated_result = validate_result(stage, verify_repository_authority=True)
        _publish_staged_directory(stage, root)
        publication_committed = True
        return validated_result
    finally:
        if (
            not publication_committed
            and stage.exists()
            and not stage.is_symlink()
        ):
            shutil.rmtree(stage)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    execute_parser = subparsers.add_parser(
        "execute",
        help="execute the exact sealed CPU-only source preflight create-only",
    )
    execute_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser(
        "validate-result",
        help="offline revalidate the complete retained result evidence",
    )
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "execute":
        result = execute(arguments.output_root)
    elif arguments.command == "validate-result":
        result = validate_result(arguments.output_root)
    else:  # pragma: no cover - argparse owns this boundary.
        raise SourcePreflightExecutionError("unsupported command")
    print(
        json.dumps(
            {
                "content_sha256": result["content_sha256"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourcePreflightExecutionError as exc:
        print(f"source preflight execution error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
