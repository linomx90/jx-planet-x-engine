#!/usr/bin/env python3
"""Bounded post-v4 JX h=1/64 numerical-closure experiment.

This successor was designed after the retained Stage-1 v4 outcome was known.
It does not alter any numerical threshold and it does not repeat REBOUND or
DE440S/DE441 dynamics.  It binds the exact validated v4 artifact, runs only
the 24 missing JX reduced-five-body h=1/64 lanes (12 native epochs for each
of DE440S and DE441), and re-evaluates the Stage-1 gates with h=1/32 as the
coarse convergence lane and h=1/64 as the production-fine lane.

The absolute JX energy ceiling applies to the production-fine lane only.
The retained h=1/32 lane remains outcome-bearing through the unchanged
contraction and Richardson rules.  This is an explicitly retrospective
successor policy, not a claim that v4 passed.  No dynamical classification,
population inference, or Planet Nine result can be emitted by this module.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import types
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_jx_h64_closure_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage1_jx_h64_closure_v1.py"
V4_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v4.py"
V4_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage1_dynamics_qualification_v4.py"
V4_ROOT = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_stage1_jx_h64_closure_v1"
DE441_ROOT = PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1"
DEFAULT_REBOUND_TARGET = Path("/tmp/jx-m4-rebound511")
PREREGISTRATION_BRANCH = "research/tno-stage1-h64-closure-v1-preregistered-20261010"
PREREGISTRATION_TAG = "tno-stage1-h64-closure-v1-preregistered-20261010"

V4_SOURCE_SHA256 = "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"
V4_TEST_SHA256 = "dc71007054e9afd1cb4d4182de253ac17f495ff835cc6b8677cdf2f36b5c2f8d"
V4_SOURCE_BYTES = 436_125
V4_TEST_BYTES = 89_619
V4_PROTOCOL_CONTENT_SHA256 = "3ea78828498b71e688ad12e6210f3c532689265a6ba202a4d5a6bae5d08f3647"
V4_RESULT_CONTENT_SHA256 = "df4e089aad8d436e99871c943343ea8b631ca714c946edb388387307c75d87d2"
V4_GIT_COMMIT = "00636a7e71d946522b3f5022a5cfd05b170dd61c"
V4_GIT_TAG = "tno-stage1-v4-preregistered-20261009"

V4_FILES = {
    "PROTOCOL.json": (74_186, "5aaa8a2db5808c1b343fb25473a50aa2bcbc28059e2d724170cd94063f72c05d"),
    "JX_MIDPOINTS.bin": (20_736, "491adb7e2a9062816ff200c8cea63018cd97a59bcd43f213b0d1180da54a6455"),
    "JX_RESTART_STREAMS.bin": (1_731_456, "26dc3b0b29433dacfb9ab5baf089c6407ab61a099eb532ea890608df84a33d6c"),
    "JX_SOURCE_NODES.bin": (276_491_560, "227211a173a80b1fa5e4534d51efebdbb7eba6ba70908522408780a7eac83294"),
    "ARRAYS.bin": (71_495_424, "e7ce12bed175e51831b4d55e558b3e813fbd555fd18c5833f91203feb2921346"),
    "IAS15_ACCEPTED_STEPS.bin": (25_665_735_616, "c5678695f709478cde083aaec165c6797b59bc55bfe6ef243405384ed228bd32"),
    "RESULT.json": (10_602_296, "bebd9186da0ef8943e05a8c4afc057073d48a8a47e9a4d006dae6e59921d5c42"),
}

PROTOCOL_SCHEMA = "jx.tno.stage1-jx-h64-closure.protocol.v1"
RESULT_SCHEMA = "jx.tno.stage1-jx-h64-closure.result.v1"
ARRAY_SCHEMA = "jx.tno.stage1-jx-h64-closure.cartesian-arrays.v1"
MIDPOINT_SCHEMA = "jx.tno.stage1-jx-h64-closure.midpoints.v1"
RESTART_STREAM_SCHEMA = "jx.tno.stage1-jx-h64-closure.restart-streams.v1"
JX_SOURCE_SCHEMA = "jx.tno.stage1-jx-h64-closure.source-nodes.v1"
IMPORT_PROBE_SCHEMA = "jx.tno.stage1-jx-h64-closure.import-probe.v1"
RESTART_RECEIPT_SCHEMA = "jx.tno.stage1-jx-h64-closure.restart-receipt.v1"
SOURCE_PREFLIGHT_SCHEMA = "jx.tno.stage1-jx-h64-closure.source-preflight.v1"

PROTOCOL_DOMAIN = b"jx.tno.stage1-jx-h64-closure.protocol.v1\0"
RESULT_DOMAIN = b"jx.tno.stage1-jx-h64-closure.result.v1\0"
ARRAY_DOMAIN = b"jx.tno.stage1-jx-h64-closure.cartesian-arrays.v1\0"
MIDPOINT_DOMAIN = b"jx.tno.stage1-jx-h64-closure.midpoints.v1\0"
RESTART_STREAM_DOMAIN = b"jx.tno.stage1-jx-h64-closure.restart-streams.v1\0"
JX_SOURCE_DOMAIN = b"jx.tno.stage1-jx-h64-closure.source-nodes.v1\0"
IMPORT_PROBE_DOMAIN = b"jx.tno.stage1-jx-h64-closure.import-probe.v1\0"
RESTART_DOMAIN = b"jx.tno.stage1-jx-h64-closure.restart-receipt.v1\0"
SCHEDULE_DOMAIN = b"jx.tno.stage1-jx-h64-closure.schedule.v1\0"
SOURCE_PREFLIGHT_DOMAIN = b"jx.tno.stage1-jx-h64-closure.source-preflight.v1\0"

HORIZON_YEARS = 1_000
OBJECT_COUNT = 12
OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
TRACERS_PER_BATCH = 9
STEPS_PER_YEAR = 64
LANE_ID = "jx_reduced5_h_1_64"
MIDPOINT_YEAR = 500
MIDPOINT_RECORD_BYTES = TRACERS_PER_BATCH * 6 * 8
MIDPOINT_RECORD_COUNT = OBJECT_COUNT * 2
EXPECTED_MIDPOINT_BYTES = MIDPOINT_RECORD_COUNT * MIDPOINT_RECORD_BYTES
SOURCE_COUNT = 5
SOURCE_GM_BYTES = SOURCE_COUNT * 8
SOURCE_NODE_COUNT = HORIZON_YEARS * STEPS_PER_YEAR + 1
SOURCE_LANE_BYTES = SOURCE_NODE_COUNT * SOURCE_COUNT * 3 * 2 * 8
EXPECTED_SOURCE_BYTES = SOURCE_GM_BYTES + MIDPOINT_RECORD_COUNT * SOURCE_LANE_BYTES
EXPECTED_ARRAY_BYTES = MIDPOINT_RECORD_COUNT * (HORIZON_YEARS + 1) * 6 * (SOURCE_COUNT + TRACERS_PER_BATCH) * 8
RESTART_CASES = (
    ("DE440S", STEPS_PER_YEAR, "jpl-50666516"),
    ("DE441", STEPS_PER_YEAR, "jpl-20042301"),
)
RESTART_SUFFIX_RECORDS = HORIZON_YEARS - MIDPOINT_YEAR + 1
RESTART_SUFFIX_BYTES = RESTART_SUFFIX_RECORDS * MIDPOINT_RECORD_BYTES
EXPECTED_RESTART_STREAM_BYTES = len(RESTART_CASES) * 2 * RESTART_SUFFIX_BYTES

CLAIMS = {
    "retrospective_outcome_seen": True,
    "v4_reclassified_as_pass": False,
    "thresholds_changed_after_outcome": False,
    "threshold_application_roster_changed_after_outcome": True,
    "rebound_dynamics_reexecuted": False,
    "retained_rebound_de440s_de441_dynamics_reexecuted": False,
    "jx_h64_dynamics_executed": False,
    "h64_refinement_closure_established": False,
    "dynamical_classification_generated": False,
    "resonance_classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class ClosureError(ValueError):
    """The bounded h=1/64 successor failed closed."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii") + b"\n"
    except (TypeError, ValueError) as exc:
        raise ClosureError("value is not canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise ClosureError("SHA-256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise ClosureError("content-addressed value must be an exact dictionary")
    copy = dict(value)
    copy.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(copy))


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if type(key) is not str or key in result:
                raise ClosureError(f"{label} has a duplicate or non-string key")
            result[key] = value
        return result
    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClosureError(f"{label} is not strict ASCII JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise ClosureError(f"{label} is not canonical")
    return value


@contextmanager
def _stable_file(path: Path, label: str, expected_size: int) -> Iterable[int]:
    absolute = Path(os.path.abspath(os.fspath(path)))
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_size != expected_size
        ):
            raise ClosureError(f"{label} has a changed file identity")
        yield descriptor
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev or after.st_ino != before.st_ino
            or after.st_size != before.st_size or after.st_nlink != before.st_nlink
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
        ):
            raise ClosureError(f"{label} changed while open")
    except OSError as exc:
        raise ClosureError(f"cannot open {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _hash_stable_file(path: Path, label: str, expected_size: int) -> str:
    digest = hashlib.sha256()
    with _stable_file(path, label, expected_size) as descriptor:
        offset = 0
        while offset < expected_size:
            block = os.pread(descriptor, min(8 * 1024 * 1024, expected_size - offset), offset)
            if not block:
                raise ClosureError(f"{label} ended early")
            digest.update(block)
            offset += len(block)
    return digest.hexdigest()


def _read_small(path: Path, label: str, expected_size: int) -> bytes:
    if expected_size > 20_000_000:
        raise ClosureError(f"{label} is not a small retained file")
    with _stable_file(path, label, expected_size) as descriptor:
        payload = bytearray()
        while len(payload) < expected_size:
            block = os.pread(descriptor, expected_size - len(payload), len(payload))
            if not block:
                raise ClosureError(f"{label} ended early")
            payload.extend(block)
    return bytes(payload)


def _read_stable_bytes(path: Path, label: str, expected_size: int) -> bytes:
    with _stable_file(path, label, expected_size) as descriptor:
        payload = bytearray()
        while len(payload) < expected_size:
            block = os.pread(
                descriptor,
                min(8 * 1024 * 1024, expected_size - len(payload)),
                len(payload),
            )
            if not block:
                raise ClosureError(f"{label} ended early")
            payload.extend(block)
    return bytes(payload)


def _selected_restart_cases() -> tuple[tuple[str, int, str], ...]:
    domain = b"jx.tno.stage1-v5.restart-selection.v1\0"
    selected: list[tuple[str, int, str]] = []
    for ephemeris in ("DE440S", "DE441"):
        object_id = min(
            OBJECT_IDS,
            key=lambda value: hashlib.sha256(
                domain
                + ephemeris.encode("ascii")
                + b"\0"
                + value.encode("ascii")
            ).digest(),
        )
        selected.append((ephemeris, STEPS_PER_YEAR, object_id))
    return tuple(selected)


def _load_v4_module(name: str) -> types.ModuleType:
    payload = _read_small(V4_SOURCE_PATH, "frozen v4 source", V4_SOURCE_BYTES)
    if _sha256(payload) != V4_SOURCE_SHA256:
        raise ClosureError("frozen v4 source digest changed")
    module = types.ModuleType(name)
    module.__file__ = str(V4_SOURCE_PATH)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(payload, str(V4_SOURCE_PATH), "exec", dont_inherit=True, optimize=0), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


_BASE: types.ModuleType | None = None


def _base() -> types.ModuleType:
    global _BASE
    if _BASE is None:
        _BASE = _load_v4_module("_jx_tno_stage1_v4_h64_base")
    return _BASE


def _delegate() -> types.ModuleType:
    d = _load_v4_module("_jx_tno_stage1_v4_h64_delegate")
    replacements = {
        "SCRIPT_PATH": SCRIPT_PATH,
        "DEFAULT_OUTPUT_ROOT": DEFAULT_OUTPUT_ROOT,
        "PROTOCOL_SCHEMA": PROTOCOL_SCHEMA,
        "RESULT_SCHEMA": RESULT_SCHEMA,
        "IMPORT_PROBE_SCHEMA": IMPORT_PROBE_SCHEMA,
        "ARRAY_SCHEMA": ARRAY_SCHEMA,
        "MIDPOINT_SCHEMA": MIDPOINT_SCHEMA,
        "RESTART_STREAM_SCHEMA": RESTART_STREAM_SCHEMA,
        "JX_SOURCE_SCHEMA": JX_SOURCE_SCHEMA,
        "RESTART_RECEIPT_SCHEMA": RESTART_RECEIPT_SCHEMA,
        "SOURCE_BUILDER_PREFLIGHT_SCHEMA": SOURCE_PREFLIGHT_SCHEMA,
        "PROTOCOL_DOMAIN": PROTOCOL_DOMAIN,
        "RESULT_DOMAIN": RESULT_DOMAIN,
        "ARRAY_DOMAIN": ARRAY_DOMAIN,
        "MIDPOINT_DOMAIN": MIDPOINT_DOMAIN,
        "RESTART_STREAM_DOMAIN": RESTART_STREAM_DOMAIN,
        "JX_SOURCE_DOMAIN": JX_SOURCE_DOMAIN,
        "RESTART_DOMAIN": RESTART_DOMAIN,
        "IMPORT_PROBE_DOMAIN": IMPORT_PROBE_DOMAIN,
        "SCHEDULE_DOMAIN": SCHEDULE_DOMAIN,
        "SOURCE_BUILDER_PREFLIGHT_DOMAIN": SOURCE_PREFLIGHT_DOMAIN,
        "CLASSIFICATION": "TNO_STAGE1_POST_V4_JX_H64_CLOSURE_ONLY",
        "CLAIM_CEILING": "RETROSPECTIVE_NUMERICAL_REFINEMENT_CLOSURE_NOT_CLASSIFICATION",
        "JX_PROVENANCE_VERSION": "v5-h64-closure-v1",
        "JX_TILE_ID_PREFIX": "stage1-v5-h64",
        "JX_PROVENANCE_ID_PREFIX": "jx-tno-stage1-v5-h64",
        "JX_STEP_YEARS": (1.0 / STEPS_PER_YEAR,),
        "JX_STEPS_PER_YEAR": (STEPS_PER_YEAR,),
        "JX_FRESH_PROCESS_RESTART_CASES": RESTART_CASES,
        "JX_MIDPOINT_RECORD_COUNT": MIDPOINT_RECORD_COUNT,
        "EXPECTED_JX_MIDPOINT_BYTES": EXPECTED_MIDPOINT_BYTES,
        "JX_RESTART_STREAM_RECORD_COUNT": len(RESTART_CASES) * 2,
        "EXPECTED_JX_RESTART_STREAM_BYTES": EXPECTED_RESTART_STREAM_BYTES,
        "JX_SOURCE_RECORD_COUNT": MIDPOINT_RECORD_COUNT,
        "EXPECTED_JX_SOURCE_BYTES": EXPECTED_SOURCE_BYTES,
        "EXPECTED_CARTESIAN_ARRAY_BYTES": EXPECTED_ARRAY_BYTES,
        "LANE_SPECS": ({
            "lane_id": LANE_ID,
            "engine": "JX_RESTRICTED_WH",
            "model": "REDUCED_5_BODY",
            "integration_step_julian_years_hex": (1.0 / STEPS_PER_YEAR).hex(),
            "steps_per_julian_year": STEPS_PER_YEAR,
            "total_fixed_steps": HORIZON_YEARS * STEPS_PER_YEAR,
        },),
    }
    for key, value in replacements.items():
        setattr(d, key, value)
    d.build_protocol = build_protocol
    return d


def _implementation_record(path: Path, label: str) -> dict[str, object]:
    try:
        expected_size = path.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise ClosureError(f"cannot stat {label}") from exc
    payload = _read_small(path, label, expected_size)
    return {"label": label, "relative_path": str(path.relative_to(PROJECT_ROOT)), "bytes": len(payload), "sha256": _sha256(payload)}


def _v4_bindings() -> list[dict[str, object]]:
    return [
        {"relative_path": f"runs/tno_stage1_dynamics_qualification_v4/{name}", "bytes": size, "sha256": digest}
        for name, (size, digest) in V4_FILES.items()
    ]


def build_protocol() -> dict[str, object]:
    if _selected_restart_cases() != RESTART_CASES:
        raise ClosureError("deterministic h64 restart selection changed")
    source = _implementation_record(SCRIPT_PATH, "h64_closure_implementation")
    test = _implementation_record(TEST_PATH, "h64_closure_test")
    protocol: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": "POST_OUTCOME_BOUNDED_JX_H64_REFINEMENT_CLOSURE",
        "design_status": "RETROSPECTIVE_SUCCESSOR_DESIGNED_AFTER_V4_FAILURE",
        "objective": "RUN_ONLY_24_JX_H64_LANES_AND_REEVALUATE_UNCHANGED_STAGE1_THRESHOLDS",
        "preregistration": {
            "branch": PREREGISTRATION_BRANCH,
            "tag": PREREGISTRATION_TAG,
            "execution_allowed_only_after_commit_and_tag_are_pushed": True,
            "commit_sha_is_external_git_custody_not_self_referential_json": True,
        },
        "implementation": source,
        "test": test,
        "predecessor_v4": {
            "git_commit": V4_GIT_COMMIT,
            "git_tag": V4_GIT_TAG,
            "source_sha256": V4_SOURCE_SHA256,
            "test_sha256": V4_TEST_SHA256,
            "protocol_content_sha256": V4_PROTOCOL_CONTENT_SHA256,
            "result_content_sha256": V4_RESULT_CONTENT_SHA256,
            "status": "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS",
            "failed_gate_names": [
                "jx_reduced_max_relative_energy_drift",
                "jx_reduced_source_fine_vs_ias15_velocity_AU_per_julian_year",
                "jx_reduced_source_richardson_velocity_tail_AU_per_julian_year",
            ],
            "bindings": _v4_bindings(),
            "semantic_revalidation_policy": "EXACT_BYTES_AND_CONTENT_REBOUND;_V4_FULL_25_7GB_SEMANTIC_REPLAY_NOT_REPEATED",
        },
        "execution_matrix": {
            "ephemerides": ["DE440S", "DE441"],
            "object_count_per_ephemeris": OBJECT_COUNT,
            "tracers_per_batch": TRACERS_PER_BATCH,
            "new_lane_id": LANE_ID,
            "new_lane_count": MIDPOINT_RECORD_COUNT,
            "step_julian_years_hex": (1.0 / STEPS_PER_YEAR).hex(),
            "steps_per_julian_year": STEPS_PER_YEAR,
            "horizon_julian_years": HORIZON_YEARS,
            "annual_checkpoint_count": HORIZON_YEARS + 1,
            "new_fixed_source_steps": MIDPOINT_RECORD_COUNT * HORIZON_YEARS * STEPS_PER_YEAR,
            "new_tracer_step_updates": MIDPOINT_RECORD_COUNT * HORIZON_YEARS * STEPS_PER_YEAR * TRACERS_PER_BATCH,
            "rebound_dynamics_reexecuted": False,
            "ias15_trace_reused_by_exact_v4_binding": True,
        },
        "gate_policy": {
            "thresholds_changed": False,
            "coarse_lane": "v4:jx_reduced5_h_1_32",
            "production_fine_lane": LANE_ID,
            "independent_reference_lanes": [
                "v4:rebound511_ias15_reduced5",
                "v4:rebound511_ias15_resolved11",
            ],
            "fine_absolute_energy_ceiling_rational": "1/100000000",
            "source_velocity_ceiling_AU_per_julian_year_rational": "1/50000",
            "richardson_divisor": 3,
            "contraction_minimum_factor": 3,
            "coarse_absolute_energy": "DIAGNOSTIC_ONLY_NOT_INCLUDED_IN_FINE_ABSOLUTE_GATE",
            "post_outcome_policy_disclosure": "ABSOLUTE_ENERGY_ACCEPTANCE_ROSTER_CHANGED_FROM_ALL_JX_LANES_TO_PRODUCTION_FINE_H64_ONLY;_NUMERICAL_THRESHOLD_UNCHANGED",
            "all_other_v4_gate_formulas_and_thresholds": "UNCHANGED",
            "failure_rule": "ANY_ACCEPTANCE_GATE_FAILURE_MEANS_H64_REFINEMENT_CLOSURE_NOT_ESTABLISHED",
        },
        "restart_policy": {
            "selection_domain": "jx.tno.stage1-v5.restart-selection.v1",
            "selection_rule": "FOR_EACH_EPHEMERIS_ARGMIN_LEXICOGRAPHIC_SHA256(DOMAIN_NUL||EPHEMERIS_ASCII||NUL||OBJECT_ID_ASCII)",
            "object_id_roster": list(OBJECT_IDS),
            "cases": [list(item) for item in RESTART_CASES],
            "fresh_process_bitwise_suffix_required": True,
        },
        "expected_artifacts": {
            "new_lane_count": MIDPOINT_RECORD_COUNT,
            "arrays_bytes": EXPECTED_ARRAY_BYTES,
            "midpoint_bytes": EXPECTED_MIDPOINT_BYTES,
            "restart_stream_bytes": EXPECTED_RESTART_STREAM_BYTES,
            "source_node_bytes": EXPECTED_SOURCE_BYTES,
            "fixed_binary_evidence_bytes": (
                EXPECTED_ARRAY_BYTES + EXPECTED_MIDPOINT_BYTES
                + EXPECTED_RESTART_STREAM_BYTES + EXPECTED_SOURCE_BYTES
            ),
            "result_linked_last": True,
        },
        "publication_policy": {
            "create_only": True,
            "replacement_allowed": False,
            "result_linked_last": True,
            "multi_file_atomicity_claimed": False,
            "interrupted_or_partial_root": "INVALID_REQUIRES_MANUAL_INSPECTION_AND_NEW_ROOT",
        },
        "claims": CLAIMS,
        "outcomes_generated": False,
        "content_sha256": "",
    }
    protocol["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, protocol)
    return protocol


def _validate_v4_artifacts(*, full_hashes: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if not V4_ROOT.is_dir() or V4_ROOT.is_symlink() or {item.name for item in V4_ROOT.iterdir()} != set(V4_FILES):
        raise ClosureError("v4 retained root roster changed")
    for name, (size, digest) in V4_FILES.items():
        path = V4_ROOT / name
        if full_hashes and _hash_stable_file(path, f"v4 {name}", size) != digest:
            raise ClosureError(f"v4 {name} digest changed")
        if not full_hashes:
            with _stable_file(path, f"v4 {name}", size):
                pass
    protocol_payload = _read_small(V4_ROOT / "PROTOCOL.json", "v4 protocol", V4_FILES["PROTOCOL.json"][0])
    result_payload = _read_small(V4_ROOT / "RESULT.json", "v4 result", V4_FILES["RESULT.json"][0])
    protocol = _strict_json(protocol_payload, "v4 protocol")
    result = _strict_json(result_payload, "v4 result")
    base = _base()
    failed = [name for name, gate in result.get("qualification_gates", {}).get("gates", {}).items() if type(gate) is dict and gate.get("passed") is not True]
    if (
        protocol.get("content_sha256") != V4_PROTOCOL_CONTENT_SHA256
        or result.get("content_sha256") != V4_RESULT_CONTENT_SHA256
        or result.get("content_sha256") != base._content_sha256(base.RESULT_DOMAIN, result)
        or result.get("status") != "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
        or failed != [
            "jx_reduced_max_relative_energy_drift",
            "jx_reduced_source_fine_vs_ias15_velocity_AU_per_julian_year",
            "jx_reduced_source_richardson_velocity_tail_AU_per_julian_year",
        ]
        or result.get("science_result") is not False
        or result.get("dynamical_classification_generated") is not False
    ):
        raise ClosureError("v4 retained semantic identity changed")
    return protocol, result


def _write_create_only(path: Path, payload: bytes) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            if count <= 0:
                raise ClosureError("create-only write made no progress")
            view = view[count:]
        os.fsync(descriptor)
    except OSError as exc:
        raise ClosureError(f"create-only write failed: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        os.fsync(descriptor)
    except OSError as exc:
        raise ClosureError(f"cannot durably sync directory: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publish_result_artifacts(
    root: Path, artifacts: Sequence[tuple[str, bytes]]
) -> None:
    """Create-only durable publication with RESULT linked last.

    A caught or abrupt failure can leave a visibly incomplete, invalid root;
    no destination pathname is ever removed automatically.
    """

    expected_names = (
        "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin",
        "JX_SOURCE_NODES.bin", "ARRAYS.bin", "RESULT.json",
    )
    if (
        not root.is_dir() or root.is_symlink()
        or {item.name for item in root.iterdir()} != {"PROTOCOL.json"}
        or type(artifacts) not in (list, tuple)
        or tuple(name for name, _payload in artifacts) != expected_names
        or any(type(payload) is not bytes or not payload for _name, payload in artifacts)
    ):
        raise ClosureError("h64 publication precondition changed")
    stage = Path(tempfile.mkdtemp(prefix=".h64-publish-", dir=root))
    staged: list[Path] = []
    try:
        for name, payload in artifacts:
            path = stage / name
            _write_create_only(path, payload)
            staged.append(path)
        _fsync_directory(stage)
        for path in staged:
            destination = root / path.name
            try:
                os.link(path, destination, follow_symlinks=False)
            except OSError as exc:
                raise ClosureError(
                    f"no-replace h64 publication failed: {destination}"
                ) from exc
            source_identity = path.lstat()
            destination_identity = destination.lstat()
            if (
                not stat.S_ISREG(destination_identity.st_mode)
                or destination_identity.st_dev != source_identity.st_dev
                or destination_identity.st_ino != source_identity.st_ino
            ):
                raise ClosureError("h64 publication hard-link identity changed")
        _fsync_directory(root)
    except Exception:
        try:
            _fsync_directory(root)
        except ClosureError:
            pass
        raise
    finally:
        for path in staged:
            try:
                path.unlink()
            except OSError:
                pass
        try:
            stage.rmdir()
        except OSError:
            pass
    if {item.name for item in root.iterdir()} != {"PROTOCOL.json", *expected_names}:
        raise ClosureError("h64 publication final roster changed")


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root.exists() or root.is_symlink():
        raise ClosureError("v5 output root must start absent")
    _validate_v4_artifacts(full_hashes=False)
    protocol = build_protocol()
    root.mkdir(mode=0o700)
    # Never unlink pathnames after a failed create-only publication.  A
    # concurrent actor could replace a name between observation and cleanup.
    # A partial root is deliberately invalid and requires manual inspection.
    _write_create_only(root / "PROTOCOL.json", _canonical_json(protocol))
    return protocol


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.is_dir() or root.is_symlink() or {item.name for item in root.iterdir()} != {"PROTOCOL.json"}:
        raise ClosureError("prepared v5 root roster changed")
    expected = build_protocol()
    expected_payload = _canonical_json(expected)
    payload = _read_small(
        root / "PROTOCOL.json", "prepared h64 protocol", len(expected_payload)
    )
    if payload != expected_payload:
        raise ClosureError("prepared v5 protocol differs from reconstruction")
    return expected


def _load_v4_gate_lanes() -> tuple[dict[tuple[str, int, str], dict[str, Any]], bytes]:
    _protocol, result = _validate_v4_artifacts(full_hashes=False)
    arrays = _read_stable_bytes(
        V4_ROOT / "ARRAYS.bin", "v4 Cartesian arrays",
        V4_FILES["ARRAYS.bin"][0],
    )
    if len(arrays) != V4_FILES["ARRAYS.bin"][0] or _sha256(arrays) != V4_FILES["ARRAYS.bin"][1]:
        raise ClosureError("v4 Cartesian arrays changed")
    base = _base()
    bridge, custody, _bindings = base._verify_input_documents()
    templates = {
        (ephemeris, batch.batch_order): batch
        for ephemeris in ("DE440S", "DE441")
        for batch in base._batch_states(bridge, custody, ephemeris)
    }
    wanted = {
        "jx_reduced5_h_1_32",
        "rebound511_ias15_reduced5",
        "rebound511_ias15_resolved11",
    }
    lanes: dict[tuple[str, int, str], dict[str, Any]] = {}
    for record in result["batch_lanes"]:
        lane_id = record["lane"]["lane_id"]
        if lane_id not in wanted:
            continue
        key = (record["ephemeris"], record["batch_order"], lane_id)
        lanes[key] = base._unpack_lane_arrays(arrays, record, templates[key[:2]])
    if len(lanes) != OBJECT_COUNT * 2 * len(wanted):
        raise ClosureError("v4 reusable lane roster changed")
    return lanes, arrays


def _alias_lane(record: Mapping[str, Any], lane_id: str) -> dict[str, Any]:
    copied = json.loads(_canonical_json(dict(record)).decode("ascii"))
    copied["lane"]["lane_id"] = lane_id
    return copied


def _combined_gate_result(new_lanes: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    base = _base()
    old, _arrays = _load_v4_gate_lanes()
    new_index = {
        (row["ephemeris"], row["batch_order"]): row
        for row in new_lanes
    }
    if len(new_index) != OBJECT_COUNT * 2:
        raise ClosureError("new h64 lane roster changed")
    combined: list[dict[str, Any]] = []
    fine_energy: list[Fraction] = []
    coarse_energy: list[Fraction] = []
    for ephemeris in ("DE440S", "DE441"):
        for batch_order in range(OBJECT_COUNT):
            coarse = old[(ephemeris, batch_order, "jx_reduced5_h_1_32")]
            fine = new_index[(ephemeris, batch_order)]
            reduced = old[(ephemeris, batch_order, "rebound511_ias15_reduced5")]
            resolved = old[(ephemeris, batch_order, "rebound511_ias15_resolved11")]
            combined.extend((
                _alias_lane(coarse, "jx_reduced5_h_1_16"),
                _alias_lane(fine, "jx_reduced5_h_1_32"),
                reduced,
                resolved,
            ))
            coarse_energy.append(base._fraction_from_record(
                coarse["lane"]["summary"]["maximum_relative_massive_energy_drift_exact"],
                "retained h32 energy drift",
            ))
            fine_energy.append(base._fraction_from_record(
                fine["lane"]["summary"]["maximum_relative_massive_energy_drift_exact"],
                "new h64 energy drift",
            ))
    result = base.evaluate_qualification_gates(combined)
    result["gates"]["jx_reduced_max_relative_energy_drift"] = base._exact_fraction_gate(
        fine_energy, Fraction(1, 100_000_000)
    )
    result["all_predeclared_gates_passed"] = all(
        gate["passed"] for gate in result["gates"].values()
    )
    result["successor_refinement_mapping"] = {
        "coarse_alias_in_v4_formula": "RETAINED_H32",
        "fine_alias_in_v4_formula": "NEW_H64",
        "absolute_energy_acceptance_roster": "NEW_H64_ONLY",
        "thresholds_changed": False,
    }
    result["retained_h32_absolute_energy_diagnostic"] = base._exact_fraction_gate(
        coarse_energy, Fraction(1, 100_000_000)
    )
    result["retained_h32_absolute_energy_diagnostic"]["used_for_acceptance"] = False
    return result


def _extract_jx_lane(
    d: types.ModuleType,
    lane: dict[str, object],
    ephemeris: str,
    batch: Any,
    source_handle: Any,
    common_gm: bytes | None,
    source_records: list[dict[str, object]],
    midpoint_builder: bytearray,
    midpoint_records: list[dict[str, object]],
    selected_suffixes: dict[tuple[str, int, str], tuple[bytes, Mapping[str, Any]]],
    fresh_registry: Mapping[str, Any],
) -> bytes:
    midpoint = lane.pop("_jx_midpoint_payload", None)
    midpoint_checked = lane.pop("_jx_midpoint_revalidated_before_continuation", None)
    suffix = lane.pop("_jx_selected_suffix_payload", None)
    ledger = lane.pop("_jx_selected_suffix_ledger", None)
    source = lane.pop("_jx_source_evidence_payload", None)
    contract = lane.pop("_jx_source_contract", None)
    gm = lane.pop("_jx_source_gm_payload", None)
    if not (type(midpoint) is bytes and midpoint_checked is True and type(source) is bytes and type(contract) is dict and type(gm) is bytes):
        raise ClosureError("new JX lane omitted required retained evidence")
    if common_gm is None:
        common_gm = gm
        if len(gm) != SOURCE_GM_BYTES:
            raise ClosureError("new JX source GM width changed")
        source_handle.write(gm)
    elif gm != common_gm:
        raise ClosureError("new JX lanes used different GM bytes")
    source_offset = source_handle.tell()
    source_handle.write(source)
    block_bytes = SOURCE_NODE_COUNT * SOURCE_COUNT * 3 * 8
    record: dict[str, object] = {
        "source_order": len(source_records), "ephemeris": ephemeris,
        "batch_order": batch.batch_order, "object_id": batch.object_id,
        "lane_id": LANE_ID, "steps_per_year": STEPS_PER_YEAR,
        "node_count": SOURCE_NODE_COUNT, "source_count": SOURCE_COUNT,
        "byte_offset": source_offset, "byte_length": len(source),
        "positions_block_bytes": block_bytes, "velocities_block_bytes": block_bytes,
        "lane_payload_sha256": _sha256(source),
        "positions_sha256": _sha256(source[:block_bytes]),
        "velocities_sha256": _sha256(source[block_bytes:]),
        "contract": contract, "full_source_tile_sha256": lane["full_source_tile_sha256"],
        "annual_source_slice_sha256": lane["annual_source_slice_sha256"],
    }
    record["record_content_sha256"] = _sha256(JX_SOURCE_DOMAIN + _canonical_json(record) + source)
    source_records.append(record)
    lane["source_node_evidence"] = {
        "source_order": record["source_order"], "byte_offset": source_offset,
        "byte_length": len(source), "lane_payload_sha256": record["lane_payload_sha256"],
        "record_content_sha256": record["record_content_sha256"],
    }
    midpoint_offset = len(midpoint_builder)
    midpoint_builder.extend(midpoint)
    identity: dict[str, object] = {
        "midpoint_order": len(midpoint_records), "ephemeris": ephemeris,
        "batch_order": batch.batch_order, "object_id": batch.object_id,
        "lane_id": LANE_ID, "steps_per_year": STEPS_PER_YEAR,
        "midpoint_year": MIDPOINT_YEAR,
        "global_source_step": MIDPOINT_YEAR * STEPS_PER_YEAR,
        "byte_offset": midpoint_offset, "byte_length": MIDPOINT_RECORD_BYTES,
        "record_sha256": _sha256(midpoint),
        "revalidated_before_continuation": True,
    }
    identity["record_content_sha256"] = _sha256(MIDPOINT_DOMAIN + _canonical_json(identity) + midpoint)
    midpoint_records.append(identity)
    selection = (ephemeris, STEPS_PER_YEAR, batch.object_id)
    if selection in RESTART_CASES:
        if type(suffix) is not bytes or type(ledger) is not dict:
            raise ClosureError("selected h64 restart evidence is missing")
        retained_ledger = dict(ledger)
        retained_ledger["fresh_process_registry"] = dict(fresh_registry)
        selected_suffixes[selection] = (suffix, retained_ledger)
    elif suffix is not None or ledger is not None:
        raise ClosureError("unselected h64 lane emitted restart evidence")
    return common_gm


def _runtime_and_lanes(
    d: types.ModuleType,
    root: Path,
    custody: Mapping[str, Any],
    import_probe: Mapping[str, Any],
    fresh_registry: Mapping[str, Any],
    fresh_child: Mapping[str, Any],
    rebound_target: Path,
) -> tuple[list[dict[str, object]], bytes, list[dict[str, object]], bytes, list[dict[str, object]], bytes, list[dict[str, object]], dict[str, object]]:
    bridge = d._load_json(d.BRIDGE_STATES_PATH, "bridge states")[1]
    lanes: list[dict[str, object]] = []
    midpoint_builder = bytearray()
    midpoint_records: list[dict[str, object]] = []
    selected: dict[tuple[str, int, str], tuple[bytes, Mapping[str, Any]]] = {}
    source_records: list[dict[str, object]] = []
    common_gm: bytes | None = None
    with tempfile.TemporaryDirectory(prefix=".tno-h64-evidence-", dir=root.parent) as temporary:
        source_path = Path(temporary) / "JX_SOURCE_NODES.bin"
        with source_path.open("xb") as source_handle:
            with d._source_only_runtime_import_guard(root / ".h64-runtime-pycache-must-remain-absent") as guard:
                cp, rebound, target_receipt = d._load_dynamics_runtime(rebound_target)
                for ephemeris in ("DE440S", "DE441"):
                    for batch in d._batch_states(bridge, custody, ephemeris):
                        lane = d._run_jx_lane(batch, STEPS_PER_YEAR, cp)
                        common_gm = _extract_jx_lane(
                            d, lane, ephemeris, batch, source_handle, common_gm,
                            source_records, midpoint_builder, midpoint_records, selected,
                            fresh_registry,
                        )
                        lanes.append({
                            "ephemeris": ephemeris, "batch_order": batch.batch_order,
                            "object_id": batch.object_id,
                            "native_epoch_jd_tdb_hex": batch.epoch_jd_tdb.hex(),
                            "lane": lane,
                        })
                midpoint_payload = bytes(midpoint_builder)
                restart_receipts, restart_payload, restart_records = d._run_fresh_process_restart_checks(
                    midpoint_payload, midpoint_records, selected
                )
                runtime = d._runtime_identity(
                    rebound, cp, target_receipt, guard, fresh_registry,
                    fresh_child, import_probe,
                )
            source_handle.flush()
            os.fsync(source_handle.fileno())
        source_payload = _read_stable_bytes(
            source_path, "temporary h64 source evidence", EXPECTED_SOURCE_BYTES
        )
    if (
        common_gm is None or len(lanes) != MIDPOINT_RECORD_COUNT
        or len(midpoint_payload) != EXPECTED_MIDPOINT_BYTES
        or len(source_payload) != EXPECTED_SOURCE_BYTES
        or len(restart_payload) != EXPECTED_RESTART_STREAM_BYTES
    ):
        raise ClosureError("new h64 evidence geometry changed")
    return (
        lanes, midpoint_payload, midpoint_records, restart_payload,
        restart_records, source_payload, source_records, runtime,
    )


def _execute_worker(
    output_root: Path,
    de441_root: Path,
    rebound_target: Path,
    import_probe_path: Path,
) -> dict[str, object]:
    d = _delegate()
    fresh_child = d._fresh_child_interpreter_receipt()
    fresh_registry = d._reject_preloaded_runtime_modules()
    _probe_payload, import_probe = d._load_json(import_probe_path, "h64 import probe")
    d._validate_import_probe_receipt(import_probe, expected_path=import_probe_path)
    root = Path(os.path.abspath(os.fspath(output_root)))
    validate_prepared(root)
    _validate_v4_artifacts(full_hashes=True)
    custody = d.validate_execution_prerequisites(de441_root)
    (
        lanes, midpoint_payload, midpoint_records, restart_payload,
        restart_records, source_payload, source_records, runtime,
    ) = _runtime_and_lanes(
        d, root, custody, import_probe, fresh_registry, fresh_child,
        rebound_target,
    )
    arrays_payload, compact_lanes, gate_lanes = d.materialize_cartesian_payload(lanes)
    if len(arrays_payload) != EXPECTED_ARRAY_BYTES:
        raise ClosureError("new h64 Cartesian payload size changed")
    gates = _combined_gate_result(gate_lanes)
    protocol = build_protocol()
    midpoint_metadata = {
        "schema": MIDPOINT_SCHEMA, "relative_path": "JX_MIDPOINTS.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_PRIMARY_RELATIVE_TRACER_POSITIONS_THEN_VELOCITIES",
        "midpoint_year": MIDPOINT_YEAR,
        "byte_length": len(midpoint_payload), "file_sha256": _sha256(midpoint_payload),
        "content_sha256": _sha256(MIDPOINT_DOMAIN + midpoint_payload),
        "record_count": len(midpoint_records), "record_bytes": MIDPOINT_RECORD_BYTES,
        "record_order": "EPHEMERIS_THEN_BATCH_THEN_JX_STEP_64",
        "records": midpoint_records, "fresh_process_restart_cases": restart_receipts,
    }
    restart_metadata = {
        "schema": RESTART_STREAM_SCHEMA, "relative_path": "JX_RESTART_STREAMS.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64",
        "byte_length": len(restart_payload), "file_sha256": _sha256(restart_payload),
        "content_sha256": _sha256(RESTART_STREAM_DOMAIN + restart_payload),
        "stream_count": len(restart_records), "record_bytes": MIDPOINT_RECORD_BYTES,
        "records_per_stream": RESTART_SUFFIX_RECORDS,
        "stream_bytes": RESTART_SUFFIX_BYTES,
        "order": "CASE_THEN_MONOLITHIC_RESUMED_THEN_YEAR_500_THROUGH_1000_THEN_TRACER_POSITIONS_VELOCITIES",
        "records": restart_records,
    }
    source_metadata = {
        "schema": JX_SOURCE_SCHEMA, "relative_path": "JX_SOURCE_NODES.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64",
        "byte_length": len(source_payload), "file_sha256": _sha256(source_payload),
        "content_sha256": _sha256(JX_SOURCE_DOMAIN + source_payload),
        "common_gm_byte_offset": 0, "common_gm_byte_length": SOURCE_GM_BYTES,
        "common_gm_sha256": _sha256(source_payload[:SOURCE_GM_BYTES]),
        "lane_count": len(source_records),
        "order": "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_64_POSITIONS_THEN_VELOCITIES",
        "records": source_records,
    }
    arrays_metadata = {
        "schema": ARRAY_SCHEMA, "relative_path": "ARRAYS.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
        "byte_length": len(arrays_payload), "file_sha256": _sha256(arrays_payload),
        "content_sha256": _sha256(ARRAY_DOMAIN + arrays_payload),
        "batch_lane_count": len(compact_lanes),
        "ordering": "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL",
        "all_vectors_origin": "LANE_INSTANTANEOUS_MASSIVE_SYSTEM_COMMON_GM_BARYCENTER",
        "position_unit": "AU", "velocity_unit": "AU_PER_JULIAN_YEAR",
    }
    passed = gates["all_predeclared_gates_passed"] is True
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "classification": "POST_OUTCOME_BOUNDED_JX_H64_REFINEMENT_CLOSURE",
        "status": "H64_REFINEMENT_CLOSURE_PASS" if passed else "H64_REFINEMENT_CLOSURE_GATES_NOT_ALL_PASS",
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_canonical_json(protocol)),
        "predecessor_v4_result_content_sha256": V4_RESULT_CONTENT_SHA256,
        "runtime_identity": runtime,
        "new_lane_count": len(compact_lanes),
        "cartesian_arrays": arrays_metadata,
        "jx_midpoint_checkpoints": midpoint_metadata,
        "jx_restart_streams": restart_metadata,
        "jx_source_nodes": source_metadata,
        "batch_lanes": compact_lanes,
        "qualification_gates": gates,
        "claims": {
            **CLAIMS,
            "jx_h64_dynamics_executed": True,
            "h64_refinement_closure_established": passed,
        },
        "outcomes_generated": True,
        "dynamical_classification_generated": False,
        "resonance_classification_generated": False,
        "science_result": False,
        "content_sha256": "",
    }
    result["content_sha256"] = _content_sha256(RESULT_DOMAIN, result)
    _publish_result_artifacts(root, (
        ("JX_MIDPOINTS.bin", midpoint_payload),
        ("JX_RESTART_STREAMS.bin", restart_payload),
        ("JX_SOURCE_NODES.bin", source_payload),
        ("ARRAYS.bin", arrays_payload),
        ("RESULT.json", _canonical_json(result)),
    ))
    return result


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    de441_root: Path = DE441_ROOT,
    rebound_target: Path = DEFAULT_REBOUND_TARGET,
) -> dict[str, object]:
    d = _delegate()
    d._reject_preloaded_runtime_modules()
    root = Path(os.path.abspath(os.fspath(output_root)))
    validate_prepared(root)
    _validate_v4_artifacts(full_hashes=True)
    d.validate_execution_prerequisites(de441_root)
    with tempfile.TemporaryDirectory(prefix="jx-tno-h64-import-probe-") as temporary:
        probe_path, _receipt = d._run_import_only_preflight(rebound_target, Path(temporary))
        completed = subprocess.run(
            [
                str(d.PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH), "execute-worker",
                "--output-root", str(root), "--de441-root", str(de441_root),
                "--rebound-target", str(rebound_target), "--import-probe-receipt", str(probe_path),
            ],
            cwd=PROJECT_ROOT, env=d._production_child_environment(),
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            check=False, timeout=604_800,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-8_000:]
        raise ClosureError(f"fresh h64 execution worker failed: {detail}")
    return validate_output(root)


def _reconstruct_new_lanes(
    d: types.ModuleType,
    result: Mapping[str, Any],
    arrays_payload: bytes,
) -> tuple[list[dict[str, Any]], Mapping[str, Any], Mapping[str, Any]]:
    bridge, custody, _bindings = d._verify_input_documents()
    templates = {
        (ephemeris, batch.batch_order): batch
        for ephemeris in ("DE440S", "DE441")
        for batch in d._batch_states(bridge, custody, ephemeris)
    }
    reconstructed = [
        d._unpack_lane_arrays(
            arrays_payload, record,
            templates[(record["ephemeris"], record["batch_order"])],
        )
        for record in result["batch_lanes"]
    ]
    return reconstructed, bridge, custody


def _validate_array_metadata(metadata: object, payload: bytes) -> None:
    expected_keys = {
        "schema", "relative_path", "encoding", "byte_length", "file_sha256",
        "content_sha256", "batch_lane_count", "ordering", "all_vectors_origin",
        "position_unit", "velocity_unit",
    }
    if (
        type(metadata) is not dict or set(metadata) != expected_keys
        or type(payload) is not bytes or len(payload) != EXPECTED_ARRAY_BYTES
        or metadata.get("schema") != ARRAY_SCHEMA
        or metadata.get("relative_path") != "ARRAYS.bin"
        or metadata.get("encoding")
        != "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER"
        or metadata.get("byte_length") != len(payload)
        or metadata.get("file_sha256") != _sha256(payload)
        or metadata.get("content_sha256") != _sha256(ARRAY_DOMAIN + payload)
        or metadata.get("batch_lane_count") != MIDPOINT_RECORD_COUNT
        or metadata.get("ordering")
        != "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL"
        or metadata.get("all_vectors_origin")
        != "LANE_INSTANTANEOUS_MASSIVE_SYSTEM_COMMON_GM_BARYCENTER"
        or metadata.get("position_unit") != "AU"
        or metadata.get("velocity_unit") != "AU_PER_JULIAN_YEAR"
    ):
        raise ClosureError("h64 Cartesian metadata changed")


def _validate_midpoint_metadata(
    d: types.ModuleType,
    metadata: object,
    payload: bytes,
    bridge: Mapping[str, Any],
) -> None:
    expected_keys = {
        "schema", "relative_path", "encoding", "midpoint_year", "byte_length",
        "file_sha256", "content_sha256", "record_count", "record_bytes",
        "record_order", "records", "fresh_process_restart_cases",
    }
    if (
        type(metadata) is not dict or set(metadata) != expected_keys
        or len(payload) != EXPECTED_MIDPOINT_BYTES
        or metadata.get("schema") != MIDPOINT_SCHEMA
        or metadata.get("relative_path") != "JX_MIDPOINTS.bin"
        or metadata.get("encoding")
        != "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_PRIMARY_RELATIVE_TRACER_POSITIONS_THEN_VELOCITIES"
        or metadata.get("midpoint_year") != MIDPOINT_YEAR
        or metadata.get("byte_length") != len(payload)
        or metadata.get("file_sha256") != _sha256(payload)
        or metadata.get("content_sha256") != _sha256(MIDPOINT_DOMAIN + payload)
        or metadata.get("record_count") != MIDPOINT_RECORD_COUNT
        or metadata.get("record_bytes") != MIDPOINT_RECORD_BYTES
        or metadata.get("record_order") != "EPHEMERIS_THEN_BATCH_THEN_JX_STEP_64"
    ):
        raise ClosureError("h64 midpoint metadata changed")
    records = metadata.get("records")
    expected_order = [
        (ephemeris, batch_order)
        for ephemeris in ("DE440S", "DE441")
        for batch_order in range(OBJECT_COUNT)
    ]
    if type(records) is not list or len(records) != MIDPOINT_RECORD_COUNT:
        raise ClosureError("h64 midpoint roster changed")
    offset = 0
    for midpoint_order, (record, identity) in enumerate(
        zip(records, expected_order, strict=True)
    ):
        ephemeris, batch_order = identity
        batch = bridge["batches"][batch_order]
        expected_record_keys = {
            "midpoint_order", "ephemeris", "batch_order", "object_id", "lane_id",
            "steps_per_year", "midpoint_year", "global_source_step", "byte_offset",
            "byte_length", "record_sha256", "revalidated_before_continuation",
            "record_content_sha256",
        }
        if (
            type(record) is not dict or set(record) != expected_record_keys
            or record.get("midpoint_order") != midpoint_order
            or record.get("ephemeris") != ephemeris
            or record.get("batch_order") != batch_order
            or record.get("object_id") != batch["object_id"]
            or record.get("lane_id") != LANE_ID
            or record.get("steps_per_year") != STEPS_PER_YEAR
            or record.get("midpoint_year") != MIDPOINT_YEAR
            or record.get("global_source_step") != MIDPOINT_YEAR * STEPS_PER_YEAR
            or record.get("byte_offset") != offset
            or record.get("byte_length") != MIDPOINT_RECORD_BYTES
            or record.get("revalidated_before_continuation") is not True
        ):
            raise ClosureError("h64 midpoint record identity changed")
        record_payload = payload[offset : offset + MIDPOINT_RECORD_BYTES]
        without_content = dict(record)
        content = without_content.pop("record_content_sha256", None)
        if (
            record.get("record_sha256") != _sha256(record_payload)
            or content
            != _sha256(MIDPOINT_DOMAIN + _canonical_json(without_content) + record_payload)
        ):
            raise ClosureError("h64 midpoint record digest changed")
        d._unpack_jx_internal_tracer_state(record_payload)
        offset += MIDPOINT_RECORD_BYTES
    if offset != len(payload):
        raise ClosureError("h64 midpoint payload has trailing bytes")


def _validate_h64_source_evidence(
    d: types.ModuleType,
    handle: Any,
    metadata: object,
    reconstructed: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
    numpy_module: types.ModuleType,
) -> None:
    if (
        type(metadata) is not dict
        or metadata.get("order")
        != "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_64_POSITIONS_THEN_VELOCITIES"
    ):
        raise ClosureError("h64 source ordering disclosure changed")
    # The frozen v4 semantic validator derives its expected lane geometry from
    # the overridden JX_STEPS_PER_YEAR tuple, but its human-readable order
    # token is a literal.  Feed it a non-retained compatibility projection;
    # the retained value above remains truthful and is independently gated.
    compatibility = dict(metadata)
    compatibility["order"] = (
        "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_16_32_POSITIONS_THEN_VELOCITIES"
    )
    d._validate_jx_source_evidence(
        handle, compatibility, reconstructed, bridge, custody, numpy_module
    )


def _validate_restart_receipts(
    d: types.ModuleType,
    midpoint_metadata: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> None:
    records = midpoint_metadata.get("records")
    receipts = midpoint_metadata.get("fresh_process_restart_cases")
    if type(records) is not list or type(receipts) is not list:
        raise ClosureError("h64 restart receipt roster is absent")
    record_index = {
        (row.get("ephemeris"), row.get("steps_per_year"), row.get("object_id")): row
        for row in records if type(row) is dict
    }
    expected_outer_keys = {
        "restart_order", "ephemeris", "steps_per_year", "object_id",
        "batch_order", "midpoint_record_sha256", "worker_receipt_file_sha256",
        "worker_receipt_content_sha256", "suffix_file_sha256",
        "final_state_sha256", "suffix_checkpoint_count", "worker_request",
        "worker_receipt", "bitwise_midpoint_through_year_1000_match",
        "bitwise_final_state_match", "ledger_match", "fresh_process",
    }
    if len(receipts) != len(RESTART_CASES):
        raise ClosureError("h64 restart receipt count changed")
    for restart_order, (outer, selection) in enumerate(
        zip(receipts, RESTART_CASES, strict=True)
    ):
        midpoint = record_index.get(selection)
        worker = outer.get("worker_receipt") if type(outer) is dict else None
        request = outer.get("worker_request") if type(outer) is dict else None
        if (
            type(outer) is not dict or set(outer) != expected_outer_keys
            or type(midpoint) is not dict or type(worker) is not dict
            or type(request) is not dict or set(request) != d.RESTART_REQUEST_KEYS
            or outer.get("restart_order") != restart_order
            or (outer.get("ephemeris"), outer.get("steps_per_year"), outer.get("object_id"))
            != selection
            or outer.get("batch_order") != midpoint["batch_order"]
            or outer.get("midpoint_record_sha256") != midpoint["record_sha256"]
            or outer.get("suffix_checkpoint_count") != RESTART_SUFFIX_RECORDS
            or outer.get("bitwise_midpoint_through_year_1000_match") is not True
            or outer.get("bitwise_final_state_match") is not True
            or outer.get("ledger_match") is not True
            or outer.get("fresh_process") is not True
            or request.get("schema")
            != "jx.tno.stage1-dynamics-qualification.restart-request.v4"
            or request.get("ephemeris") != selection[0]
            or request.get("batch_order") != midpoint["batch_order"]
            or request.get("object_id") != selection[2]
            or request.get("steps_per_year") != selection[1]
            or request.get("midpoint_year") != MIDPOINT_YEAR
            or request.get("midpoint_relative_path") != "JX_MIDPOINTS.bin"
            or request.get("midpoint_byte_offset") != midpoint["byte_offset"]
            or request.get("midpoint_byte_length") != MIDPOINT_RECORD_BYTES
            or request.get("midpoint_record_sha256") != midpoint["record_sha256"]
            or request.get("content_sha256")
            != _content_sha256(RESTART_DOMAIN, request)
        ):
            raise ClosureError("h64 restart request or outer receipt changed")
        d._validate_restart_worker_receipt(worker, runtime_identity)
        if (
            worker.get("request_content_sha256") != request["content_sha256"]
            or worker.get("request_file_sha256") != _sha256(_canonical_json(request))
            or worker.get("midpoint_record_sha256") != midpoint["record_sha256"]
            or worker.get("suffix_file_sha256") != outer["suffix_file_sha256"]
            or worker.get("final_state_sha256") != outer["final_state_sha256"]
            or worker.get("suffix_checkpoint_count")
            != outer["suffix_checkpoint_count"]
            or worker.get("content_sha256")
            != outer["worker_receipt_content_sha256"]
            or _sha256(_canonical_json(worker))
            != outer["worker_receipt_file_sha256"]
        ):
            raise ClosureError("h64 restart worker receipt cross-binding changed")


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    expected_names = {
        "PROTOCOL.json", "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin",
        "JX_SOURCE_NODES.bin", "ARRAYS.bin", "RESULT.json",
    }
    if not root.is_dir() or root.is_symlink() or {item.name for item in root.iterdir()} != expected_names:
        raise ClosureError("executed v5 root roster changed")
    _validate_v4_artifacts(full_hashes=True)
    expected_protocol = _canonical_json(build_protocol())
    protocol_payload = _read_small(
        root / "PROTOCOL.json", "h64 protocol", len(expected_protocol)
    )
    if protocol_payload != expected_protocol:
        raise ClosureError("executed v5 protocol changed")
    midpoint = _read_small(
        root / "JX_MIDPOINTS.bin", "h64 midpoint", EXPECTED_MIDPOINT_BYTES
    )
    restart = _read_small(
        root / "JX_RESTART_STREAMS.bin", "h64 restart", EXPECTED_RESTART_STREAM_BYTES
    )
    source_path = root / "JX_SOURCE_NODES.bin"
    arrays = _read_small(root / "ARRAYS.bin", "h64 arrays", EXPECTED_ARRAY_BYTES)
    result_size = (root / "RESULT.json").stat(follow_symlinks=False).st_size
    result_payload = _read_small(root / "RESULT.json", "h64 result", result_size)
    result = _strict_json(result_payload, "v5 result")
    d = _delegate()
    d._guard_exact_json_integer_identities(result, "h64 result")
    expected_result_keys = {
        "schema", "classification", "status", "protocol_content_sha256",
        "protocol_file_sha256", "predecessor_v4_result_content_sha256",
        "runtime_identity", "new_lane_count", "cartesian_arrays",
        "jx_midpoint_checkpoints", "jx_restart_streams", "jx_source_nodes",
        "batch_lanes", "qualification_gates", "claims", "outcomes_generated",
        "dynamical_classification_generated", "resonance_classification_generated",
        "science_result", "content_sha256",
    }
    if (
        set(result) != expected_result_keys
        or
        len(midpoint) != EXPECTED_MIDPOINT_BYTES
        or len(restart) != EXPECTED_RESTART_STREAM_BYTES
        or len(arrays) != EXPECTED_ARRAY_BYTES
        or result.get("schema") != RESULT_SCHEMA
        or result.get("classification")
        != "POST_OUTCOME_BOUNDED_JX_H64_REFINEMENT_CLOSURE"
        or result.get("protocol_content_sha256")
        != build_protocol()["content_sha256"]
        or result.get("protocol_file_sha256") != _sha256(expected_protocol)
        or result.get("content_sha256") != _content_sha256(RESULT_DOMAIN, result)
        or result.get("predecessor_v4_result_content_sha256") != V4_RESULT_CONTENT_SHA256
        or result.get("new_lane_count") != MIDPOINT_RECORD_COUNT
        or result.get("science_result") is not False
        or result.get("dynamical_classification_generated") is not False
        or result.get("resonance_classification_generated") is not False
    ):
        raise ClosureError("v5 retained result identity changed")
    d._validate_runtime_identity(result.get("runtime_identity"))
    _validate_array_metadata(result.get("cartesian_arrays"), arrays)
    reconstructed, bridge, custody = _reconstruct_new_lanes(d, result, arrays)
    _validate_midpoint_metadata(
        d, result.get("jx_midpoint_checkpoints"), midpoint, bridge
    )
    retained_numpy = result["runtime_identity"]["numpy_runtime_identity"]
    with d._bound_offline_numpy_runtime(retained_numpy) as numpy_module:
        with d._stable_read_handle(source_path, "v5 source nodes", maximum=EXPECTED_SOURCE_BYTES) as source_handle:
            _validate_h64_source_evidence(
                d,
                source_handle, result.get("jx_source_nodes"), reconstructed,
                bridge, custody, numpy_module,
            )
            d._validate_restart_stream_evidence(
                restart, result.get("jx_restart_streams"), midpoint,
                result.get("jx_midpoint_checkpoints"), source_handle,
                result.get("jx_source_nodes"), reconstructed, bridge, custody,
            )
    _validate_restart_receipts(
        d, result["jx_midpoint_checkpoints"], result["runtime_identity"]
    )
    gates = _combined_gate_result(reconstructed)
    if _canonical_json(gates) != _canonical_json(result.get("qualification_gates")):
        raise ClosureError("v5 gates differ from retained observables")
    passed = gates["all_predeclared_gates_passed"] is True
    expected_status = "H64_REFINEMENT_CLOSURE_PASS" if passed else "H64_REFINEMENT_CLOSURE_GATES_NOT_ALL_PASS"
    expected_claims = {
        **CLAIMS,
        "jx_h64_dynamics_executed": True,
        "h64_refinement_closure_established": passed,
    }
    if (
        result.get("status") != expected_status
        or result.get("claims") != expected_claims
        or result.get("outcomes_generated") is not True
    ):
        raise ClosureError("v5 status or claims differ from recomputed gates")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "protocol", "prepare", "validate-prepared", "execute", "execute-worker",
        "validate", "resume-worker", "import-probe-worker",
    ))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--de441-root", type=Path, default=DE441_ROOT)
    parser.add_argument("--rebound-target", type=Path, default=DEFAULT_REBOUND_TARGET)
    parser.add_argument("--import-probe-receipt", type=Path)
    parser.add_argument("--probe-output", type=Path)
    parser.add_argument("--resume-request", type=Path)
    parser.add_argument("--resume-output-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "protocol":
            value = build_protocol()
        elif args.command == "prepare":
            value = prepare(args.output_root)
        elif args.command == "validate-prepared":
            value = validate_prepared(args.output_root)
        elif args.command == "execute":
            value = execute(args.output_root, de441_root=args.de441_root, rebound_target=args.rebound_target)
        elif args.command == "execute-worker":
            if args.import_probe_receipt is None:
                raise ClosureError("execute-worker requires import probe receipt")
            value = _execute_worker(
                args.output_root, args.de441_root, args.rebound_target,
                args.import_probe_receipt,
            )
        elif args.command == "validate":
            value = validate_output(args.output_root)
        elif args.command == "resume-worker":
            if args.resume_request is None or args.resume_output_root is None:
                raise ClosureError("resume-worker requires request and output root")
            value = _delegate()._run_jx_restart_worker(
                args.resume_request, args.resume_output_root
            )
        else:
            if args.probe_output is None:
                raise ClosureError("import-probe-worker requires output path")
            value = _delegate()._import_probe_worker(
                args.rebound_target, args.probe_output
            )
    except (ClosureError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    summary = {
        "schema": value.get("schema"),
        "status": value.get("status", "OK"),
        "content_sha256": value.get("content_sha256"),
        "all_predeclared_gates_passed": value.get("qualification_gates", {}).get("all_predeclared_gates_passed"),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
