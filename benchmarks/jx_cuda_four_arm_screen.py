#!/usr/bin/env python3
"""Run the resumable M2 four-arm CUDA mechanism screen.

This runner is intentionally narrower than a Planet Nine inference pipeline.
It compares four closed, fully self-gravitating source rosters against one
shared *synthetic* massless-tracer population:

* the frozen current-epoch DE441 outer-system control;
* that control plus a circular distant-planet sensitivity fixture;
* that control plus a synthetic passing star; and
* that control plus a synthetic unbound rogue flyby.

The acceptance-v3 tracer fixture is translated into the DE441 Sun-relative
frame without changing any tracer's Sun-relative position or velocity.  A
domain-separated receipt binds the input bytes, transform, output arrays, and
selection rule.  It is not an observed, Rubin-selected, debiased, or primordial
population.

The supported stages are a short mechanics qualification, a full-population
100-kyr route-capacity pilot, and a 1-Myr screening rung.  Every integration
segment is committed through the engine's immutable checkpoint chain, so an
interrupted arm resumes from its latest verified endpoint.  Results remain
``MODEL_OUTPUT / SCREENING_ONLY`` and make no claim that any candidate is
detected, excluded, preferred, or historically real.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

import jxplanetx.cuda_massless_tracer as cuda_tracer_module  # noqa: E402
from jxplanetx.cuda_massless_tracer import (  # noqa: E402
    EVENT_LOCATOR_SOURCE_SHA256,
    IMPLEMENTATION_SOURCE_SHA256,
    KERNEL_SOURCE_SHA256,
    LIFECYCLE_ACTIVE,
    LIFECYCLE_NAMES,
    MAXIMUM_EVENT_LOCATOR_ACCEPTED_STEPS,
    MAXIMUM_TOTAL_ENCOUNTER_ROUTES,
    METHOD_ID,
    OUTWARD_EVENT_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR,
    CUDAMasslessTracerResult,
    CUDAMasslessTracerState,
    FixedStepCUDAMasslessTracerSpec,
    advance_cuda_massless_tracers,
    cuda_massless_tracer_runtime_identity,
    dump_cuda_massless_tracer_checkpoint,
    load_latest_cuda_massless_tracer_checkpoint,
)
from jxplanetx.cuda_mechanism_schedules import (  # noqa: E402
    ARM_KINDS,
    CONTROL_ARM,
    DE441_CONTROL_EPOCH_JD_TDB,
    DISTANT_PLANET_ARM,
    RESULT_CLAIMS,
    SCIENTIFIC_CLASSIFICATION,
    STELLAR_FLYBY_ARM,
    TEMPORARY_ROGUE_ARM,
    MatchedMechanismScheduleSet,
    MechanismArm,
    build_de441_four_arm_schedule_set,
)
from jxplanetx.engine import BackendSpec, Provenance  # noqa: E402


SCHEMA = "jxplanetx.cuda-four-arm-mechanism-screen.v1"
PROTOCOL_SCHEMA = "jxplanetx.cuda-four-arm-mechanism-screen-protocol.v1"
TRANSFORM_SCHEMA = "jxplanetx.cuda-four-arm-population-transform.v1"
ARM_RESULT_SCHEMA = "jxplanetx.cuda-four-arm-result.v1"
SUMMARY_SCHEMA = "jxplanetx.cuda-four-arm-summary.v1"
FAILURE_SCHEMA = "jxplanetx.cuda-four-arm-failure.v1"
RUNTIME_RECEIPT_SCHEMA = "jxplanetx.cuda-four-arm-runtime-receipt.v1"
CAPACITY_ADMISSION_SCHEMA = "jxplanetx.cuda-four-arm-capacity-admission.v1"

PROTOCOL_DOMAIN = b"jxplanetx.cuda-four-arm-mechanism-screen-protocol.v1\x00"
TRANSFORM_DOMAIN = b"jxplanetx.cuda-four-arm-population-transform.v1\x00"
ARRAY_DOMAIN = b"jxplanetx.cuda-four-arm-array-set.v1\x00"
PROVENANCE_DOMAIN = b"jxplanetx.cuda-four-arm-provenance.v1\x00"
REPORT_DOMAIN = b"jxplanetx.cuda-four-arm-report.v1\x00"
RUNTIME_RECEIPT_DOMAIN = b"jxplanetx.cuda-four-arm-runtime-receipt.v1\x00"
EXECUTION_EVIDENCE_DOMAIN = b"jxplanetx.cuda-four-arm-execution-evidence.v1\x00"
EXECUTION_EVIDENCE_SCHEMA = "jxplanetx.cuda-four-arm-execution-evidence.v1"
CAPACITY_ADMISSION_DOMAIN = b"jxplanetx.cuda-four-arm-capacity-admission.v1\x00"
CAPACITY_PREFIX_PHYSICS_DOMAIN = (
    b"jxplanetx.cuda-four-arm-capacity-prefix-physics.v1\x00"
)

ACCEPTANCE_INITIAL_STATE_SHA256 = (
    "ebb4d6b8952e78a6eb91e9591c2ddff36831f3f92b98952a9f50da802e82f048"
)
ACCEPTANCE_PROTOCOL_SHA256 = (
    "0beda4fdfb98394dd7d0fc505b46da94b01ea8adcbaac9769251294ca1b3e75b"
)
DE441_CONTROL_SHA256 = (
    "edfbfd50ca29b28b46fe43cae8cb99bd73a9303a5f2e5babf7bf9ad2a820ff14"
)

DEFAULT_ACCEPTANCE_STATE = (
    ROOT / "runs/cuda_massless_tracer_acceptance_v3/INITIAL_STATE.npz"
)
DEFAULT_ACCEPTANCE_PROTOCOL = (
    ROOT / "runs/cuda_massless_tracer_acceptance_v3/PROTOCOL.json"
)
DEFAULT_DE441_CONTROL = (
    ROOT / "runs/de441_population_100k/states/de441_control_state.csv"
)
DEFAULT_OUTPUT_ROOT = ROOT / "runs/cuda_four_arm_screen_v1"

STEP_YEARS = 0.125
PRIMARY_OUTER_BOUNDARY_AU = 10_000.0
RELATIVE_ACCELERATION_BOUND_AU_PER_YEAR2 = 10.0
MAXIMUM_PRIMARY_DYNAMICAL_PHASE = 0.05
EVENT_TIME_TOLERANCE_FRACTION = 1.0e-12
MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER = 16_384
MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM = MAXIMUM_TOTAL_ENCOUNTER_ROUTES
MAXIMUM_TOTAL_ROUTE_RECORDS_PER_STAGE = (
    len(ARM_KINDS) * MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
)
MINIMUM_FREE_STORAGE_BYTES = 16 * 1024**3
SOURCE_DIAGNOSTIC_YEARS = 1_000.0
FULL_SCREEN_PASSAGE_YEARS = 50_000.0

# Route episodes are maximal runs of routed macrosteps for one tracer.  Source
# or reason changes do not split a run, and even one quiet macrostep closes it.
# This definition deliberately counts exact route records rather than a
# physically interpreted encounter duration.
ROUTE_EPISODE_RESERVED_RECORDS = 512

# The admission ceiling leaves a fixed 25% of the unchanged hard engine cap
# unused.  It is a fail-closed engineering resource margin, not a prediction
# that the target integration will complete.
ROUTE_CAPACITY_REQUIRED_HEADROOM = 4_096
ROUTE_CAPACITY_ADMISSION_LIMIT = (
    MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER - ROUTE_CAPACITY_REQUIRED_HEADROOM
)

# Only a full-population 100-kyr prefix of the exact one-Myr physical initial
# bytes may authorize the one-Myr rung.  The pure analyzer can still describe
# shorter mechanics tests, but they fail this identity gate.
ROUTE_CAPACITY_PILOT_STEPS = 800_000

INTERPRETATION_LIMITS = (
    "This is a current-epoch intervention screen, not a primordial reconstruction.",
    "The tracer population is a deterministic synthetic engineering fixture, not observed, Rubin-selected, survey-debiased, or fitted.",
    "Each non-control arm is one synthetic point assumption, not a prior distribution or population model.",
    "A passing source is active and mutually gravitating for the complete finite integration; the nominal passage window is diagnostic only.",
    "The fixed 0.125-year lattice has not established timestep convergence for 1-Myr candidate contrasts.",
    "The finite passing-source fixtures omit the Galactic tide and the Galactic/stellar environment before and after the modeled interval.",
    "Removal surfaces are operational numerical boundaries, not physical impact or definitive ejection classifications.",
    "The screen cannot detect, exclude, or prefer Planet Nine, a stellar flyby, or a rogue planet.",
    "No result from this runner qualifies a 4.5-Gyr history or accuracy claim.",
)


@dataclass(frozen=True)
class Stage:
    stage_id: str
    duration_years: float
    tracer_count: int
    checkpoint_years: float
    completion_status: str

    @property
    def total_steps(self) -> int:
        return _years_to_steps(self.duration_years, "duration_years")

    @property
    def checkpoint_steps(self) -> int:
        return _years_to_steps(self.checkpoint_years, "checkpoint_years")


STAGES = {
    "qualification": Stage(
        stage_id="qualification",
        duration_years=1_000.0,
        tracer_count=2_048,
        checkpoint_years=250.0,
        completion_status="PASSED_QUALIFICATION_SCREEN_NOT_SCIENCE",
    ),
    "capacity_pilot": Stage(
        stage_id="capacity_pilot",
        duration_years=100_000.0,
        tracer_count=16_800,
        checkpoint_years=25_000.0,
        completion_status="PASSED_CAPACITY_PILOT_NOT_SCIENCE",
    ),
    "one_myr": Stage(
        stage_id="one_myr",
        duration_years=1_000_000.0,
        tracer_count=16_800,
        checkpoint_years=100_000.0,
        completion_status="COMPLETED_ONE_MYR_SCREEN_NOT_INFERENCE",
    ),
}

ARM_DIRECTORY_NAMES = {
    CONTROL_ARM: "control",
    DISTANT_PLANET_ARM: "circular_distant_planet_sensitivity",
    STELLAR_FLYBY_ARM: "stellar_flyby",
    TEMPORARY_ROGUE_ARM: "unbound_rogue_flyby",
}

REQUIRED_ARM_ARTIFACT_NAMES = frozenset(
    {
        "FORCE_SCHEDULE.json",
        "SOURCE_STATE.json",
        "SOURCE_STATE.npz",
        "INITIAL_STATE.npz",
        "ENDPOINT_STATE.npz",
        "ENCOUNTER_ROUTES.json",
        "REMOVAL_EVENTS.json",
        "SOURCE_PATH_AUDIT.json",
        "SOURCE_PATH_AUDIT_ENDPOINT.npz",
        "ROUTE_SOURCE_MEMBERSHIP_AUDIT.json",
        "ROUTED_TRACER_REPLAY_AUDIT.json",
    }
)

# Operational source-radius removal surfaces in AU.  The Sun entry is a
# 16-AU inner-model-domain exit, not a physical radius or collision.  The other
# values are approximate physical-size scales, but their role here is still
# solely to define reproducible numerical terminal surfaces; the runner never
# implements or claims collision physics.
SOURCE_COLLISION_RADII_AU = {
    # This is deliberately *not* the solar radius.  The reduced outer-system
    # model is outside its declared domain once a tracer enters 16 AU.
    "Sun": 16.0,
    "Jupiter": 0.000477895,
    "Saturn": 0.000402866,
    "Uranus": 0.000170851,
    "Neptune": 0.000165537,
    "Pluto": 0.00000794,
    "P9_SYNTHETIC": 0.0001,
    "FLYBY_STAR_SYNTHETIC": 0.00465047,
    "ROGUE_PLANET_SYNTHETIC": 0.0001,
}


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_set_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256(ARRAY_DOMAIN)
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(value.dtype.str.encode("ascii") + b"\x00")
        digest.update(_canonical(list(value.shape)) + b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _content_hash(domain: bytes, value: Mapping[str, Any]) -> str:
    return hashlib.sha256(domain + _canonical(dict(value))).hexdigest()


def _verify_content_hash(
    value: Mapping[str, Any],
    domain: bytes,
    *,
    schema: str,
    label: str,
) -> None:
    if value.get("schema") != schema:
        raise RuntimeError(f"{label} schema differs from the closed contract")
    observed = value.get("content_sha256")
    if type(observed) is not str or len(observed) != 64:
        raise RuntimeError(f"{label} lacks a valid content SHA-256")
    payload = dict(value)
    payload.pop("content_sha256", None)
    expected = _content_hash(domain, payload)
    if observed != expected:
        raise RuntimeError(f"{label} content SHA-256 failed")


def _require_runtime_identity(
    observed: object,
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    if type(observed) is not dict or observed != dict(expected):
        raise RuntimeError(f"{label} differs from the frozen stage runtime identity")


def _current_runtime_identity(
    backend: BackendSpec,
    *,
    expected: Mapping[str, Any] | None = None,
    label: str = "current CUDA/CPU runtime identity",
) -> dict[str, Any]:
    runtime = dict(cuda_massless_tracer_runtime_identity(backend))
    if (
        runtime.get("method_id") != METHOD_ID
        or runtime.get("kernel_source_sha256") != KERNEL_SOURCE_SHA256
        or runtime.get("implementation_source_sha256")
        != IMPLEMENTATION_SOURCE_SHA256
        or runtime.get("event_locator_source_sha256")
        != EVENT_LOCATOR_SOURCE_SHA256
    ):
        raise RuntimeError("CUDA runtime implementation identity differs from the runner")
    if expected is not None:
        _require_runtime_identity(runtime, expected, label=label)
    return runtime


def _freeze_stage_runtime_identity(
    stage_directory: Path,
    *,
    protocol_sha256: str,
    current_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze one runtime for the stage and reject heterogeneous cached evidence."""

    receipt_path = stage_directory / "RUNTIME_IDENTITY.json"
    if receipt_path.is_file():
        receipt = _load_json(receipt_path)
        _verify_content_hash(
            receipt,
            RUNTIME_RECEIPT_DOMAIN,
            schema=RUNTIME_RECEIPT_SCHEMA,
            label="stage runtime receipt",
        )
        if receipt.get("protocol_sha256") != protocol_sha256:
            raise RuntimeError("stage runtime receipt binds a different protocol")
        _require_runtime_identity(
            receipt.get("runtime_identity"),
            current_runtime,
            label="stage runtime receipt",
        )
    elif receipt_path.exists():
        raise RuntimeError("stage runtime receipt path is not a regular file")
    else:
        receipt = {
            "schema": RUNTIME_RECEIPT_SCHEMA,
            "protocol_sha256": protocol_sha256,
            "runtime_identity": dict(current_runtime),
            "semantics": (
                "EXACT_STAGE_WIDE_CUDA_GPU_CPU_PYTHON_NUMPY_SCIPY_RUNTIME; "
                "ANY_CHANGE_REQUIRES_A_NEW_STAGE"
            ),
        }
        receipt["content_sha256"] = _content_hash(RUNTIME_RECEIPT_DOMAIN, receipt)

    # A pre-receipt partial stage may exist from an earlier runner.  Validate
    # every durable runtime-bearing artifact before publishing or accepting the
    # receipt, so a resumed invocation cannot mix environments and fail late.
    cached_documents = (
        (
            "arms/*/SOURCE_PATH_AUDIT.json",
            "jxplanetx.cuda-four-arm-source-path-audit.v1",
            "cached source-path audit",
        ),
        ("arms/*/ARM_RESULT.json", ARM_RESULT_SCHEMA, "cached arm result"),
        ("SUMMARY.json", SUMMARY_SCHEMA, "cached stage summary"),
    )
    for pattern, schema, label in cached_documents:
        for path in sorted(stage_directory.glob(pattern)):
            document = _load_json(path)
            _verify_content_hash(
                document,
                REPORT_DOMAIN,
                schema=schema,
                label=f"{label} {path.relative_to(stage_directory)}",
            )
            _require_runtime_identity(
                document.get("runtime_identity"),
                current_runtime,
                label=f"{label} {path.relative_to(stage_directory)}",
            )
            if (
                document.get("runtime_identity_receipt_sha256")
                != receipt["content_sha256"]
            ):
                raise RuntimeError(
                    f"{label} {path.relative_to(stage_directory)} binds a "
                    "different stage runtime receipt"
                )

    _publish_json_exact(receipt_path, receipt)
    return receipt


def _verify_artifact_records(
    records: object,
    *,
    stage_directory: Path,
    label: str,
) -> None:
    if type(records) is not dict or not records:
        raise RuntimeError(f"{label} artifact roster is invalid")
    stage_root = stage_directory.resolve()
    for key, record in records.items():
        if type(key) is not str or type(record) is not dict:
            raise RuntimeError(f"{label} artifact roster contains an invalid row")
        relative = record.get("path")
        if type(relative) is not str or not relative:
            raise RuntimeError(f"{label} artifact path is invalid")
        path = (stage_directory / relative).resolve()
        try:
            path.relative_to(stage_root)
        except ValueError as exc:
            raise RuntimeError(f"{label} artifact escapes the stage directory") from exc
        if not path.is_file():
            raise RuntimeError(f"{label} artifact is absent: {relative}")
        if record.get("size_bytes") != path.stat().st_size:
            raise RuntimeError(f"{label} artifact size failed: {relative}")
        if record.get("sha256") != _sha256_file(path):
            raise RuntimeError(f"{label} artifact SHA-256 failed: {relative}")


def _years_to_steps(years: float, label: str) -> int:
    if not math.isfinite(years) or years <= 0.0:
        raise ValueError(f"{label} must be positive and finite")
    quotient = years / STEP_YEARS
    rounded = round(quotient)
    if not math.isclose(quotient, rounded, rel_tol=0.0, abs_tol=1.0e-10):
        raise ValueError(f"{label} must be an exact multiple of {STEP_YEARS}")
    return int(rounded)


def _route_storage_guard(
    path: Path,
    *,
    phase: str,
    cumulative_route_records: int,
) -> dict[str, Any]:
    """Fail closed before route-ledger work can consume the recovery reserve."""

    if type(cumulative_route_records) is not int or cumulative_route_records < 0:
        raise RuntimeError("cumulative route-record count is invalid")
    if cumulative_route_records > MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM:
        raise RuntimeError(
            "arm cumulative route-record budget was exceeded before checkpoint commit"
        )
    usage = shutil.disk_usage(path)
    free_bytes = int(usage.free)
    if free_bytes < MINIMUM_FREE_STORAGE_BYTES:
        raise RuntimeError(
            f"insufficient free storage before {phase}: {free_bytes} bytes available; "
            f"{MINIMUM_FREE_STORAGE_BYTES} required"
        )
    return {
        "phase": phase,
        "cumulative_route_records": cumulative_route_records,
        "maximum_total_route_records_per_arm": (
            MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
        ),
        "observed_free_storage_bytes": free_bytes,
        "required_minimum_free_storage_bytes": MINIMUM_FREE_STORAGE_BYTES,
        "passed": True,
    }


def _commit_create_only(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite artifact: {destination}") from None
    else:
        temporary.unlink()
        descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_bytes_create(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _publish_json_exact(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_payload(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing artifact differs from requested bytes: {path}")
        return
    _atomic_bytes_create(path, payload)


def _publish_npz_exact(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    normalized = {
        name: np.ascontiguousarray(value) for name, value in arrays.items()
    }
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != set(normalized):
                    raise RuntimeError(f"existing NPZ member roster differs: {path}")
                for name, expected in normalized.items():
                    if not np.array_equal(archive[name], expected):
                        raise RuntimeError(
                            f"existing NPZ member differs: {path}:{name}"
                        )
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"existing NPZ is unreadable: {path}") from exc
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **normalized)
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise RuntimeError(f"JSON artifact must contain an object: {path}")
    return value


def _artifact(path: Path, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _git(command: Iterable[str]) -> str:
    process = subprocess.run(
        ["git", *command],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return process.stdout.strip()


def _git_identity() -> dict[str, Any]:
    return {
        "head": _git(("rev-parse", "HEAD")),
        "branch": _git(("branch", "--show-current")),
        "main": _git(("rev-parse", "main")),
        # Exact executable/source hashes below bind the working implementation.
        # Do not bind the mutable worktree-status listing: the run's own
        # create-only evidence directory may itself be untracked and would make
        # an otherwise identical checkpoint resume impossible.
        "clean_worktree_claimed": False,
    }


def _verify_frozen_inputs(
    acceptance_state: Path,
    acceptance_protocol: Path,
    de441_control: Path,
) -> None:
    expected = {
        acceptance_state: ACCEPTANCE_INITIAL_STATE_SHA256,
        acceptance_protocol: ACCEPTANCE_PROTOCOL_SHA256,
        de441_control: DE441_CONTROL_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file():
            raise FileNotFoundError(f"required frozen input is absent: {path}")
        observed = _sha256_file(path)
        if observed != digest:
            raise RuntimeError(
                f"frozen input digest mismatch for {path}: {observed} != {digest}"
            )


def _derive_population(
    acceptance_state: Path,
    schedule_set: MatchedMechanismScheduleSet,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(acceptance_state, allow_pickle=False) as archive:
        required = {
            "source_positions",
            "source_velocities",
            "tracer_positions",
            "tracer_velocities",
            "tracer_indices",
        }
        if not required.issubset(archive.files):
            raise RuntimeError("acceptance-v3 state lacks required population arrays")
        source_positions = np.ascontiguousarray(archive["source_positions"], dtype="<f8")
        source_velocities = np.ascontiguousarray(archive["source_velocities"], dtype="<f8")
        input_positions = np.ascontiguousarray(archive["tracer_positions"], dtype="<f8")
        input_velocities = np.ascontiguousarray(archive["tracer_velocities"], dtype="<f8")
        indices = np.ascontiguousarray(archive["tracer_indices"], dtype="<u8")
    if source_positions.shape[1:] != (3,) or source_velocities.shape != source_positions.shape:
        raise RuntimeError("acceptance source arrays have an invalid shape")
    if input_positions.shape != input_velocities.shape or input_positions.shape[1:] != (3,):
        raise RuntimeError("acceptance tracer phase-space arrays have an invalid shape")
    if indices.shape != (input_positions.shape[0],) or len(np.unique(indices)) != len(indices):
        raise RuntimeError("acceptance tracer identities are invalid")

    target = schedule_set.control.initial_state
    relative_positions = input_positions - source_positions[0]
    relative_velocities = input_velocities - source_velocities[0]
    output_positions = np.ascontiguousarray(
        relative_positions + target.positions_AU[0], dtype="<f8"
    )
    output_velocities = np.ascontiguousarray(
        relative_velocities + target.velocities_AU_per_year[0], dtype="<f8"
    )
    arrays = {
        "tracer_positions": output_positions,
        "tracer_velocities": output_velocities,
        "tracer_indices": indices,
    }
    input_arrays = {
        "acceptance_sun_position": source_positions[0],
        "acceptance_sun_velocity": source_velocities[0],
        "tracer_positions": input_positions,
        "tracer_velocities": input_velocities,
        "tracer_indices": indices,
    }
    receipt: dict[str, Any] = {
        "schema": TRANSFORM_SCHEMA,
        "classification": "MODEL_OUTPUT / SYNTHETIC_POPULATION_REFRAME_ONLY",
        "input_artifact": {
            "artifact_id": "cuda-massless-tracer-acceptance-v3-initial-state",
            "sha256": ACCEPTANCE_INITIAL_STATE_SHA256,
            "population_description": (
                "deterministic synthetic a=100..1000 AU, q=31..80 AU; "
                "not observed, primordial, Rubin-selected, or survey-debiased"
            ),
            "semantic_array_sha256": _array_set_sha256(input_arrays),
        },
        "target_source_state": {
            "artifact_sha256": target.base_artifact_sha256,
            "source_state_sha256": target.state_sha256,
            "epoch_jd_tdb_hex": target.epoch_jd_tdb.hex(),
            "frame": target.frame,
            "origin": target.origin,
            "sun_position_AU_hex": [float(value).hex() for value in target.positions_AU[0]],
            "sun_velocity_AU_per_year_hex": [
                float(value).hex() for value in target.velocities_AU_per_year[0]
            ],
        },
        "transform": {
            "position": "r_out=(r_in-r_acceptance_sun)+r_de441_sun",
            "velocity": "v_out=(v_in-v_acceptance_sun)+v_de441_sun",
            "rotation": "NONE",
            "orientation_assumption": (
                "the synthetic fixture reference plane is identified with the DE441 "
                "ecliptic axes; this arbitrary engineering convention carries no "
                "observational orientation information"
            ),
            "resampling": "NONE",
            "selection": "IMMUTABLE_PREFIX_BY_ACCEPTANCE_ARRAY_ORDER",
            "identity_preservation": "TRACER_UINT64_IDS_UNCHANGED",
        },
        "output": {
            "tracer_count": int(indices.size),
            "semantic_array_sha256": _array_set_sha256(arrays),
            "sun_relative_phase_space_preserved": True,
        },
        "claims": {
            "observed_population": False,
            "rubin_selected": False,
            "survey_debiased": False,
            "primordial_population": False,
            "coordinate_transform_adds_information": False,
        },
    }
    receipt["content_sha256"] = _content_hash(TRANSFORM_DOMAIN, receipt)
    return arrays, receipt


def _source_safety_arrays(arm: MechanismArm) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = arm.initial_state
    count = len(state.source_ids)
    encounter = np.empty(count, dtype="<f8")
    collision = np.empty(count, dtype="<f8")
    primary_gm = float(state.gm_AU3_per_year2[0])
    primary_position = state.positions_AU[0]
    for index, source_id in enumerate(state.source_ids):
        try:
            collision[index] = SOURCE_COLLISION_RADII_AU[source_id]
        except KeyError as exc:
            raise RuntimeError(f"no frozen collision radius for {source_id}") from exc
        if index == 0:
            encounter[index] = 5.0
        elif source_id == "FLYBY_STAR_SYNTHETIC":
            encounter[index] = 25.0
        elif source_id == "ROGUE_PLANET_SYNTHETIC":
            encounter[index] = 10.0
        else:
            distance = float(np.linalg.norm(state.positions_AU[index] - primary_position))
            hill = distance * (
                float(state.gm_AU3_per_year2[index]) / (3.0 * primary_gm)
            ) ** (1.0 / 3.0)
            encounter[index] = 3.0 * hill
        encounter[index] = max(encounter[index], collision[index])
    acceleration = np.full(
        count,
        RELATIVE_ACCELERATION_BOUND_AU_PER_YEAR2,
        dtype="<f8",
    )
    return encounter, collision, acceleration


def _source_state_artifacts(arm: MechanismArm) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    state = arm.initial_state
    metadata = state.canonical_metadata()
    metadata["state_sha256"] = state.state_sha256
    metadata["base_prefix_sha256"] = state.base_prefix_sha256
    arrays = {
        "source_positions": state.positions_AU,
        "source_velocities": state.velocities_AU_per_year,
        "source_gm": state.gm_AU3_per_year2,
    }
    return metadata, arrays


def _source_invariants(
    cp: Any,
    positions: Any,
    velocities: Any,
    gm: Any,
) -> tuple[Any, Any, Any]:
    """Return GM-scaled Newtonian energy, momentum, and angular momentum."""

    kinetic = 0.5 * cp.sum(
        gm[None, :] * cp.sum(velocities * velocities, axis=2), axis=1
    )
    potential = cp.zeros(positions.shape[0], dtype=cp.float64)
    for left in range(int(gm.shape[0]) - 1):
        delta = positions[:, left + 1 :, :] - positions[:, left : left + 1, :]
        distance = cp.linalg.norm(delta, axis=2)
        potential -= cp.sum(
            gm[left] * gm[None, left + 1 :] / distance,
            axis=1,
        )
    momentum = cp.sum(gm[None, :, None] * velocities, axis=1)
    angular_momentum = cp.sum(
        gm[None, :, None] * cp.cross(positions, velocities), axis=1
    )
    return kinetic + potential, momentum, angular_momentum


def _source_state_digest(positions: np.ndarray, velocities: np.ndarray) -> str:
    return _array_set_sha256(
        {
            "source_positions": np.ascontiguousarray(positions),
            "source_velocities": np.ascontiguousarray(velocities),
        }
    )


def _source_path_audit(
    *,
    cp: Any,
    backend: BackendSpec,
    stage: Stage,
    checkpoint_years: float,
    arm: MechanismArm,
    arm_directory: Path,
    protocol_sha256: str,
    runtime_identity: Mapping[str, Any],
    runtime_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Audit the exact massive-source KDK path independently of tracers.

    This intentionally uses the same private source-path kernel bound by the
    engine implementation hash.  It samples conservation diagnostics every
    1,000 years, but it evaluates every lattice node for closest approaches.
    Source chunk boundaries exactly mirror the tracer run's checkpoint and
    65,536-step chunk policy so checkpoint endpoint hashes can be compared.
    """

    _current_runtime_identity(
        backend,
        expected=runtime_identity,
        label=f"current runtime before {arm.schedule.arm_kind} source-path audit",
    )
    audit_path = arm_directory / "SOURCE_PATH_AUDIT.json"
    endpoint_path = arm_directory / "SOURCE_PATH_AUDIT_ENDPOINT.npz"
    if audit_path.exists() or endpoint_path.exists():
        if not (audit_path.is_file() and endpoint_path.is_file()):
            raise RuntimeError("source-path audit artifact pair is incomplete")
        document = _load_json(audit_path)
        _verify_content_hash(
            document,
            REPORT_DOMAIN,
            schema="jxplanetx.cuda-four-arm-source-path-audit.v1",
            label="source-path audit",
        )
        if (
            document.get("protocol_sha256") != protocol_sha256
            or document.get("schedule_sha256")
            != arm.schedule.schedule_sha256
            or document.get("arm_kind") != arm.schedule.arm_kind
            or document.get("method_id") != METHOD_ID
            or document.get("kernel_source_sha256") != KERNEL_SOURCE_SHA256
            or document.get("implementation_source_sha256")
            != IMPLEMENTATION_SOURCE_SHA256
            or document.get("runtime_identity_receipt_sha256")
            != runtime_receipt_sha256
            or document.get("step_years") != STEP_YEARS
            or document.get("total_steps") != stage.total_steps
        ):
            raise RuntimeError("source-path audit identity differs from this run")
        _require_runtime_identity(
            document.get("runtime_identity"),
            runtime_identity,
            label=f"cached {arm.schedule.arm_kind} source-path audit",
        )
        with np.load(endpoint_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "source_positions",
                "source_velocities",
            }:
                raise RuntimeError("source-path audit endpoint roster differs")
            endpoint = {
                name: np.ascontiguousarray(archive[name])
                for name in archive.files
            }
        if _array_set_sha256(endpoint) != document.get(
            "endpoint_semantic_array_sha256"
        ):
            raise RuntimeError("source-path audit endpoint hash failed")
        checkpoint_steps = _years_to_steps(checkpoint_years, "checkpoint_years")
        expected_checkpoint_steps: list[str] = []
        cursor = 0
        while cursor < stage.total_steps:
            cursor = min(stage.total_steps, cursor + checkpoint_steps)
            expected_checkpoint_steps.append(str(cursor))
        checkpoint_hashes = document.get("checkpoint_source_state_sha256")
        if (
            type(checkpoint_hashes) is not dict
            or sorted(checkpoint_hashes, key=int) != expected_checkpoint_steps
            or any(
                type(value) is not str or len(value) != 64
                for value in checkpoint_hashes.values()
            )
        ):
            raise RuntimeError("source-path audit checkpoint hash table is invalid")
        if checkpoint_hashes[expected_checkpoint_steps[-1]] != _source_state_digest(
            endpoint["source_positions"], endpoint["source_velocities"]
        ):
            raise RuntimeError("source-path final checkpoint hash differs from endpoint")
        return document, endpoint

    resolved = cuda_tracer_module.resolve_backend(backend)
    source = arm.initial_state
    checkpoint_steps = _years_to_steps(checkpoint_years, "checkpoint_years")
    diagnostic_steps = _years_to_steps(
        SOURCE_DIAGNOSTIC_YEARS, "source_diagnostic_years"
    )
    source_count = len(source.source_ids)
    pair_minima: dict[tuple[int, int], tuple[float, int]] = {
        (left, right): (math.inf, -1)
        for left in range(source_count - 1)
        for right in range(left + 1, source_count)
    }
    diagnostic_rows: dict[int, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, str] = {}

    with resolved.activate():
        source_kernel, _ = cuda_tracer_module._path_kernels(resolved)
        gm = cp.asarray(source.gm_AU3_per_year2)
        current_position = cp.asarray(source.positions_AU)
        current_velocity = cp.asarray(source.velocities_AU_per_year)
        initial_energy_device, initial_momentum_device, initial_angular_device = (
            _source_invariants(
                cp,
                current_position[None, :, :],
                current_velocity[None, :, :],
                gm,
            )
        )
        initial_energy = float(cp.asnumpy(initial_energy_device[0]))
        initial_momentum = cp.asnumpy(initial_momentum_device[0])
        initial_angular = cp.asnumpy(initial_angular_device[0])
        energy_scale = max(abs(initial_energy), np.finfo(np.float64).tiny)
        angular_scale = max(
            float(np.linalg.norm(initial_angular)),
            np.finfo(np.float64).tiny,
        )
        maximum_energy_drift = 0.0
        maximum_angular_drift = 0.0
        maximum_momentum_delta = 0.0
        processed = 0
        while processed < stage.total_steps:
            segment_end = min(
                stage.total_steps,
                ((processed // checkpoint_steps) + 1) * checkpoint_steps,
            )
            while processed < segment_end:
                chunk_steps = min(65_536, segment_end - processed)
                path = cuda_tracer_module._build_source_path(
                    resolved,
                    source_kernel,
                    current_position,
                    current_velocity,
                    gm,
                    step_count=chunk_steps,
                    step=STEP_YEARS,
                )
                for pair in pair_minima:
                    left, right = pair
                    distance = cp.linalg.norm(
                        path.positions[:, right, :] - path.positions[:, left, :],
                        axis=1,
                    )
                    local_index = int(cp.asnumpy(cp.argmin(distance)))
                    local_distance = float(cp.asnumpy(distance[local_index]))
                    if local_distance < pair_minima[pair][0]:
                        pair_minima[pair] = (
                            local_distance,
                            processed + local_index,
                        )

                path_energy, path_momentum, path_angular = _source_invariants(
                    cp, path.positions, path.velocities, gm
                )
                chunk_energy_drift = float(
                    cp.asnumpy(
                        cp.max(cp.abs(path_energy - initial_energy) / energy_scale)
                    )
                )
                chunk_angular_drift = float(
                    cp.asnumpy(
                        cp.max(
                            cp.linalg.norm(
                                path_angular - cp.asarray(initial_angular)[None, :],
                                axis=1,
                            )
                            / angular_scale
                        )
                    )
                )
                chunk_momentum_delta = float(
                    cp.asnumpy(
                        cp.max(
                            cp.linalg.norm(
                                path_momentum - cp.asarray(initial_momentum)[None, :],
                                axis=1,
                            )
                        )
                    )
                )
                maximum_energy_drift = max(
                    maximum_energy_drift, chunk_energy_drift
                )
                maximum_angular_drift = max(
                    maximum_angular_drift, chunk_angular_drift
                )
                maximum_momentum_delta = max(
                    maximum_momentum_delta, chunk_momentum_delta
                )

                first_sample = (
                    (processed + diagnostic_steps - 1) // diagnostic_steps
                ) * diagnostic_steps
                sample_steps = np.arange(
                    first_sample,
                    processed + chunk_steps + 1,
                    diagnostic_steps,
                    dtype=np.int64,
                )
                if sample_steps.size:
                    local = cp.asarray(sample_steps - processed, dtype=cp.int64)
                    sample_positions = path.positions[local]
                    sample_velocities = path.velocities[local]
                    energy, momentum, angular = _source_invariants(
                        cp, sample_positions, sample_velocities, gm
                    )
                    energy_host = cp.asnumpy(energy)
                    momentum_host = cp.asnumpy(momentum)
                    angular_host = cp.asnumpy(angular)
                    for row_index, absolute_step in enumerate(sample_steps):
                        step_value = int(absolute_step)
                        diagnostic_rows[step_value] = {
                            "global_step": step_value,
                            "elapsed_years": step_value * STEP_YEARS,
                            "scaled_source_energy": float(energy_host[row_index]),
                            "relative_energy_drift": float(
                                (energy_host[row_index] - initial_energy) / energy_scale
                            ),
                            "gm_weighted_linear_momentum": [
                                float(value) for value in momentum_host[row_index]
                            ],
                            "linear_momentum_delta_norm": float(
                                np.linalg.norm(momentum_host[row_index] - initial_momentum)
                            ),
                            "gm_weighted_angular_momentum": [
                                float(value) for value in angular_host[row_index]
                            ],
                            "relative_angular_momentum_vector_drift": float(
                                np.linalg.norm(angular_host[row_index] - initial_angular)
                                / angular_scale
                            ),
                        }
                current_position = path.positions[-1].copy()
                current_velocity = path.velocities[-1].copy()
                processed += chunk_steps
            checkpoint_position = cp.asnumpy(current_position)
            checkpoint_velocity = cp.asnumpy(current_velocity)
            checkpoint_hashes[str(processed)] = _source_state_digest(
                checkpoint_position, checkpoint_velocity
            )
        cp.cuda.runtime.deviceSynchronize()
        endpoint = {
            "source_positions": cp.asnumpy(current_position),
            "source_velocities": cp.asnumpy(current_velocity),
        }

    _current_runtime_identity(
        backend,
        expected=runtime_identity,
        label=f"current runtime after {arm.schedule.arm_kind} source-path audit",
    )
    pair_records = []
    primary_records = []
    for (left, right), (minimum, global_step) in sorted(pair_minima.items()):
        record = {
            "left_source_index": left,
            "left_source_id": source.source_ids[left],
            "right_source_index": right,
            "right_source_id": source.source_ids[right],
            "minimum_lattice_distance_AU": minimum,
            "minimum_global_step": global_step,
            "minimum_elapsed_years": global_step * STEP_YEARS,
            "sampling_semantics": "ALL_FIXED_STEP_LATTICE_NODES_NOT_CONTINUOUS_ROOT",
        }
        pair_records.append(record)
        if left == 0:
            primary_records.append(record)
    rows = [diagnostic_rows[key] for key in sorted(diagnostic_rows)]
    audit: dict[str, Any] = {
        "schema": "jxplanetx.cuda-four-arm-source-path-audit.v1",
        "arm_kind": arm.schedule.arm_kind,
        "protocol_sha256": protocol_sha256,
        "schedule_sha256": arm.schedule.schedule_sha256,
        "method_id": METHOD_ID,
        "kernel_source_sha256": KERNEL_SOURCE_SHA256,
        "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_receipt_sha256": runtime_receipt_sha256,
        "step_years": STEP_YEARS,
        "total_steps": stage.total_steps,
        "all_lattice_nodes_scanned_for_minima": True,
        "source_diagnostic_cadence_years": SOURCE_DIAGNOSTIC_YEARS,
        "source_pair_minima": pair_records,
        "primary_relative_source_minima": primary_records,
        "conservation_diagnostics": rows,
        "maximum_absolute_relative_energy_drift": maximum_energy_drift,
        "maximum_relative_angular_momentum_vector_drift": maximum_angular_drift,
        "maximum_linear_momentum_delta_norm": maximum_momentum_delta,
        "conservation_maximum_semantics": "ALL_FIXED_STEP_LATTICE_NODES",
        "checkpoint_source_state_sha256": checkpoint_hashes,
        "endpoint_semantic_array_sha256": _array_set_sha256(endpoint),
        "interpretation": (
            "massive-source numerical audit for this finite closed schedule; "
            "not an observed passage reconstruction"
        ),
    }
    audit["content_sha256"] = _content_hash(REPORT_DOMAIN, audit)
    _publish_npz_exact(endpoint_path, endpoint)
    _publish_json_exact(audit_path, audit)
    return audit, endpoint


def _route_source_membership_audit(
    *,
    cp: Any,
    backend: BackendSpec,
    stage: Stage,
    arm: MechanismArm,
    arm_directory: Path,
    protocol_sha256: str,
    routes: tuple[Any, ...],
) -> dict[str, Any]:
    """Bind every route source witness to the regenerated KDK source path."""

    audit_path = arm_directory / "ROUTE_SOURCE_MEMBERSHIP_AUDIT.json"
    if audit_path.exists():
        document = _load_json(audit_path)
        _verify_content_hash(
            document,
            REPORT_DOMAIN,
            schema="jxplanetx.cuda-four-arm-route-source-membership.v1",
            label="route source-membership audit",
        )
        if (
            document.get("protocol_sha256") != protocol_sha256
            or document.get("schedule_sha256")
            != arm.schedule.schedule_sha256
            or document.get("arm_kind") != arm.schedule.arm_kind
            or document.get("method_id") != METHOD_ID
            or document.get("kernel_source_sha256") != KERNEL_SOURCE_SHA256
            or document.get("implementation_source_sha256")
            != IMPLEMENTATION_SOURCE_SHA256
            or document.get("step_years") != STEP_YEARS
            or document.get("total_steps") != stage.total_steps
            or document.get("route_count") != len(routes)
            or document.get("all_route_source_segments_exactly_matched")
            is not True
        ):
            raise RuntimeError(
                "route source-membership audit identity or result differs"
            )
        return document

    routes_by_step: dict[int, list[Any]] = {}
    for route in routes:
        if not 0 <= route.global_step < stage.total_steps:
            raise RuntimeError("route step is outside the stage source path")
        routes_by_step.setdefault(route.global_step, []).append(route)

    resolved = cuda_tracer_module.resolve_backend(backend)
    source = arm.initial_state
    with resolved.activate():
        source_kernel, _ = cuda_tracer_module._path_kernels(resolved)
        gm = cp.asarray(source.gm_AU3_per_year2)
        gm_host = np.ascontiguousarray(source.gm_AU3_per_year2)
        current_position = cp.asarray(source.positions_AU)
        current_velocity = cp.asarray(source.velocities_AU_per_year)
        processed = 0
        matched = 0
        unique_steps = 0
        while processed < stage.total_steps:
            chunk_steps = min(65_536, stage.total_steps - processed)
            path = cuda_tracer_module._build_source_path(
                resolved,
                source_kernel,
                current_position,
                current_velocity,
                gm,
                step_count=chunk_steps,
                step=STEP_YEARS,
            )
            local_steps = sorted(
                step
                for step in routes_by_step
                if processed <= step < processed + chunk_steps
            )
            for absolute_step in local_steps:
                local = absolute_step - processed
                position_0 = cp.asnumpy(path.positions[local])
                velocity_0 = cp.asnumpy(path.velocities[local])
                position_1 = cp.asnumpy(path.positions[local + 1])
                velocity_1 = cp.asnumpy(path.velocities[local + 1])
                segment_sha256 = cuda_tracer_module._source_segment_sha256(
                    position_0,
                    velocity_0,
                    position_1,
                    velocity_1,
                    gm_host,
                    STEP_YEARS,
                )
                for route in routes_by_step[absolute_step]:
                    if (
                        route.source_segment_sha256 != segment_sha256
                        or route.source_segment_step_years != STEP_YEARS
                        or not np.array_equal(
                            np.asarray(
                                route.source_segment_position_0_AU,
                                dtype=np.float64,
                            ),
                            position_0,
                        )
                        or not np.array_equal(
                            np.asarray(
                                route.source_segment_velocity_0_AU_per_year,
                                dtype=np.float64,
                            ),
                            velocity_0,
                        )
                        or not np.array_equal(
                            np.asarray(
                                route.source_segment_position_1_AU,
                                dtype=np.float64,
                            ),
                            position_1,
                        )
                        or not np.array_equal(
                            np.asarray(
                                route.source_segment_velocity_1_AU_per_year,
                                dtype=np.float64,
                            ),
                            velocity_1,
                        )
                        or not np.array_equal(
                            np.asarray(
                                route.source_segment_gm_AU3_per_year2,
                                dtype=np.float64,
                            ),
                            gm_host,
                        )
                    ):
                        raise RuntimeError(
                            "a route source witness is not a member of the exact "
                            f"integrated source path at step {absolute_step}"
                        )
                    matched += 1
                unique_steps += 1
            current_position = path.positions[-1].copy()
            current_velocity = path.velocities[-1].copy()
            processed += chunk_steps
        cp.cuda.runtime.deviceSynchronize()
        endpoint_sha256 = _source_state_digest(
            cp.asnumpy(current_position), cp.asnumpy(current_velocity)
        )
    if matched != len(routes):
        raise RuntimeError("not every route source witness was audited")
    document: dict[str, Any] = {
        "schema": "jxplanetx.cuda-four-arm-route-source-membership.v1",
        "arm_kind": arm.schedule.arm_kind,
        "protocol_sha256": protocol_sha256,
        "schedule_sha256": arm.schedule.schedule_sha256,
        "method_id": METHOD_ID,
        "kernel_source_sha256": KERNEL_SOURCE_SHA256,
        "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
        "step_years": STEP_YEARS,
        "total_steps": stage.total_steps,
        "route_count": len(routes),
        "unique_routed_source_steps": unique_steps,
        "all_route_source_segments_exactly_matched": True,
        "regenerated_source_endpoint_sha256": endpoint_sha256,
        "scope": (
            "exact membership of persisted route source segments in the same-code "
            "KDK source path; not independent reproduction"
        ),
    }
    document["content_sha256"] = _content_hash(REPORT_DOMAIN, document)
    _publish_json_exact(audit_path, document)
    return document


def _routed_tracer_replay_audit(
    *,
    cp: Any,
    backend: BackendSpec,
    stage: Stage,
    arm: MechanismArm,
    arm_directory: Path,
    schedule_set: MatchedMechanismScheduleSet,
    population: Mapping[str, np.ndarray],
    receipt_sha256: str,
    protocol_sha256: str,
    main_result: CUDAMasslessTracerResult,
    main_endpoint: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Replay each routed lane from its exact initial state for lineage audit."""

    audit_path = arm_directory / "ROUTED_TRACER_REPLAY_AUDIT.json"
    routed_ids = tuple(sorted({row.tracer_index for row in main_result.encounter_routes}))
    ledger_commitment = hashlib.sha256(
        REPORT_DOMAIN
        + _canonical(
            {
                "routes": [asdict(row) for row in main_result.encounter_routes],
                "removals": [asdict(row) for row in main_result.removal_events],
            }
        )
    ).hexdigest()
    if audit_path.exists():
        document = _load_json(audit_path)
        _verify_content_hash(
            document,
            REPORT_DOMAIN,
            schema="jxplanetx.cuda-four-arm-routed-tracer-replay.v1",
            label="routed-tracer replay audit",
        )
        if (
            document.get("protocol_sha256") != protocol_sha256
            or document.get("schedule_sha256") != arm.schedule.schedule_sha256
            or document.get("arm_kind") != arm.schedule.arm_kind
            or document.get("method_id") != METHOD_ID
            or document.get("kernel_source_sha256") != KERNEL_SOURCE_SHA256
            or document.get("implementation_source_sha256")
            != IMPLEMENTATION_SOURCE_SHA256
            or document.get("event_locator_source_sha256")
            != EVENT_LOCATOR_SOURCE_SHA256
            or document.get("main_route_ledger_sha256") != ledger_commitment
            or document.get("routed_tracer_count") != len(routed_ids)
            or document.get("exact_replay_passed") is not True
        ):
            raise RuntimeError("routed-tracer replay audit identity differs")
        return document

    if routed_ids:
        selected_ids = np.asarray(routed_ids, dtype=np.uint64)
        population_ids = np.asarray(
            population["tracer_indices"][: stage.tracer_count], dtype=np.uint64
        )
        slot_for_id = {
            int(tracer_id): slot
            for slot, tracer_id in enumerate(population_ids)
        }
        try:
            selected_slots = np.asarray(
                [slot_for_id[int(tracer_id)] for tracer_id in selected_ids],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise RuntimeError("a routed tracer is outside the frozen population") from exc
        subset = {
            "tracer_positions": np.ascontiguousarray(
                population["tracer_positions"][selected_slots]
            ),
            "tracer_velocities": np.ascontiguousarray(
                population["tracer_velocities"][selected_slots]
            ),
            "tracer_indices": np.ascontiguousarray(selected_ids),
        }
        replay_state = _build_device_state(
            cp,
            arm,
            schedule_set,
            subset,
            receipt_sha256,
            len(routed_ids),
        )
        replay_spec = FixedStepCUDAMasslessTracerSpec(
            step_years=STEP_YEARS,
            step_count=stage.total_steps,
            backend=backend,
            maximum_encounter_routes_per_tracer=(
                MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
            ),
            maximum_total_encounter_routes=(
                MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
            ),
            encounter_route_method=(
                "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2"
            ),
            encounter_route_rtol=1.0e-12,
            encounter_route_atol=1.0e-14,
            encounter_route_maximum_substep_fraction=0.125,
            maximum_primary_dynamical_phase=MAXIMUM_PRIMARY_DYNAMICAL_PHASE,
            event_time_tolerance_fraction=EVENT_TIME_TOLERANCE_FRACTION,
            source_path_steps_per_chunk=65_536,
            threads_per_block=128,
        )
        replay = advance_cuda_massless_tracers(replay_state, replay_spec)
        replay_endpoint = _endpoint_arrays(cp, replay)
        routed_id_set = set(routed_ids)
        main_routes = tuple(
            row
            for row in main_result.encounter_routes
            if row.tracer_index in routed_id_set
        )
        main_removals = tuple(
            row
            for row in main_result.removal_events
            if row.tracer_index in routed_id_set
        )
        if replay.encounter_routes != main_routes or replay.removal_events != main_removals:
            raise RuntimeError(
                "routed-tracer replay ledger differs from the main trajectory"
            )
        for key in (
            "tracer_positions",
            "tracer_velocities",
            "minimum_source_distance_AU",
            "ever_unbound",
            "encounter_route_counts",
            "tracer_lifecycle",
        ):
            if not np.array_equal(
                replay_endpoint[key], main_endpoint[key][selected_slots]
            ):
                raise RuntimeError(
                    f"routed-tracer replay endpoint differs for {key}"
                )
        if not np.array_equal(
            replay_endpoint["source_positions"],
            main_endpoint["source_positions"],
        ) or not np.array_equal(
            replay_endpoint["source_velocities"],
            main_endpoint["source_velocities"],
        ):
            raise RuntimeError("routed-tracer replay source endpoint differs")

    document: dict[str, Any] = {
        "schema": "jxplanetx.cuda-four-arm-routed-tracer-replay.v1",
        "arm_kind": arm.schedule.arm_kind,
        "protocol_sha256": protocol_sha256,
        "schedule_sha256": arm.schedule.schedule_sha256,
        "method_id": METHOD_ID,
        "kernel_source_sha256": KERNEL_SOURCE_SHA256,
        "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
        "event_locator_source_sha256": EVENT_LOCATOR_SOURCE_SHA256,
        "main_route_ledger_sha256": ledger_commitment,
        "routed_tracer_count": len(routed_ids),
        "routed_tracer_ids": list(routed_ids),
        "exact_replay_passed": True,
        "scope": (
            "same-code full-stage replay of every routed tracer from its frozen "
            "initial state; tracer-count independence check, not independent reproduction"
        ),
    }
    document["content_sha256"] = _content_hash(REPORT_DOMAIN, document)
    _publish_json_exact(audit_path, document)
    return document


def _schedule_documents(schedule_set: MatchedMechanismScheduleSet) -> dict[str, Any]:
    return {
        "matched_set": {
            **schedule_set.canonical_record(),
            "matched_set_sha256": schedule_set.matched_set_sha256,
        },
        "arms": {
            arm.schedule.arm_kind: {
                **arm.schedule.canonical_record(),
                "schedule_sha256": arm.schedule.schedule_sha256,
            }
            for arm in schedule_set.arms
        },
    }


def _stage_nominal_passage_years(stage: Stage) -> float:
    if stage.stage_id in {"capacity_pilot", "one_myr"}:
        return FULL_SCREEN_PASSAGE_YEARS
    return 0.5 * stage.duration_years


def _build_stage_schedule_set(
    stage: Stage,
    de441_control: Path,
) -> MatchedMechanismScheduleSet:
    nominal_passage_step = _years_to_steps(
        _stage_nominal_passage_years(stage), "nominal_passage_years"
    )
    return build_de441_four_arm_schedule_set(
        de441_control,
        step_years=STEP_YEARS,
        total_steps=stage.total_steps,
        matched_set_id=(
            f"jx-de441-current-epoch-four-mechanism-{stage.stage_id}-v1"
        ),
        nominal_passage_global_step=nominal_passage_step,
        diagnostic_half_window_steps=_years_to_steps(
            min(10_000.0, 0.1 * stage.duration_years),
            "diagnostic_half_window_years",
        ),
    )


def _schedule_prefix_physics_binding(
    schedule_set: MatchedMechanismScheduleSet,
) -> dict[str, Any]:
    """Normalize only nonphysical stage identifiers and horizon endpoints."""

    arms: dict[str, Any] = {}
    observed_order: list[str] = []
    for arm in schedule_set.arms:
        schedule = arm.schedule
        observed_order.append(schedule.arm_kind)
        record = dict(schedule.canonical_record())
        clock = record.pop("clock")
        if (
            type(clock) is not dict
            or clock.get("start_global_step") != 0
            or clock.get("end_global_step_exclusive") != schedule.total_steps
        ):
            raise RuntimeError("capacity schedule clock is not a full-stage interval")
        activity = record.pop("source_activity")
        expected_activity = [
            {
                "source_id": source_id,
                "start_global_step": 0,
                "end_global_step_exclusive": schedule.total_steps,
            }
            for source_id in schedule.source_ids
        ]
        if activity != expected_activity:
            raise RuntimeError("capacity schedule source activity is not full-stage")
        if record.pop("schedule_id", None) != schedule.schedule_id:
            raise RuntimeError("capacity schedule identifier canonicalization failed")
        if record.pop("matched_set_id", None) != schedule.matched_set_id:
            raise RuntimeError("capacity matched-set identifier canonicalization failed")
        record["clock"] = {
            "start_global_step": 0,
            "step_years_hex": clock.get("step_years_hex"),
            "direction": clock.get("direction"),
            "end_semantics": "STAGE_END_EXCLUSIVE",
        }
        record["source_activity"] = [
            {
                "source_id": source_id,
                "activity_semantics": "ACTIVE_FROM_STEP_ZERO_TO_STAGE_END_EXCLUSIVE",
            }
            for source_id in schedule.source_ids
        ]
        arms[schedule.arm_kind] = record
    if tuple(observed_order) != ARM_KINDS:
        raise RuntimeError("capacity prefix-physics arm order is not canonical")
    document: dict[str, Any] = {
        "schema": "jxplanetx.cuda-four-arm-capacity-prefix-physics.v1",
        "arm_order": list(ARM_KINDS),
        "arms": arms,
    }
    document["content_sha256"] = _content_hash(
        CAPACITY_PREFIX_PHYSICS_DOMAIN, document
    )
    return document


def _initial_state_semantic_hashes(
    schedule_set: MatchedMechanismScheduleSet,
    population: Mapping[str, np.ndarray],
    tracer_count: int,
) -> dict[str, str]:
    return {
        arm.schedule.arm_kind: _array_set_sha256(
            _initial_artifact_arrays(arm, population, tracer_count)
        )
        for arm in schedule_set.arms
    }


def _source_activity_is_exact_prefix(
    pilot: MatchedMechanismScheduleSet,
    target: MatchedMechanismScheduleSet,
) -> bool:
    if tuple(arm.schedule.arm_kind for arm in pilot.arms) != ARM_KINDS:
        return False
    if tuple(arm.schedule.arm_kind for arm in target.arms) != ARM_KINDS:
        return False
    if pilot.control.schedule.total_steps >= target.control.schedule.total_steps:
        return False
    for pilot_arm, target_arm in zip(pilot.arms, target.arms, strict=True):
        if pilot_arm.schedule.source_ids != target_arm.schedule.source_ids:
            return False
        expected_pilot = tuple(
            (source_id, 0, pilot_arm.schedule.total_steps)
            for source_id in pilot_arm.schedule.source_ids
        )
        expected_target = tuple(
            (source_id, 0, target_arm.schedule.total_steps)
            for source_id in target_arm.schedule.source_ids
        )
        observed_pilot = tuple(
            (
                row.source_id,
                row.start_global_step,
                row.end_global_step_exclusive,
            )
            for row in pilot_arm.schedule.source_activity
        )
        observed_target = tuple(
            (
                row.source_id,
                row.start_global_step,
                row.end_global_step_exclusive,
            )
            for row in target_arm.schedule.source_activity
        )
        if observed_pilot != expected_pilot or observed_target != expected_target:
            return False
    return True


def _capacity_target_context(
    *,
    pilot_stage: Stage,
    pilot_schedule_set: MatchedMechanismScheduleSet,
    pilot_population: Mapping[str, np.ndarray],
    pilot_receipt: Mapping[str, Any],
    de441_control: Path,
    acceptance_state: Path,
) -> dict[str, Any]:
    if pilot_stage.stage_id != "capacity_pilot":
        raise ValueError("capacity target context requires the capacity_pilot stage")
    target_stage = STAGES["one_myr"]
    target_schedule_set = _build_stage_schedule_set(target_stage, de441_control)
    target_population, target_receipt = _derive_population(
        acceptance_state, target_schedule_set
    )
    pilot_population_keys = tuple(sorted(pilot_population))
    target_population_keys = tuple(sorted(target_population))
    population_arrays_equal = (
        pilot_population_keys == target_population_keys
        and all(
            np.array_equal(pilot_population[name], target_population[name])
            for name in pilot_population_keys
        )
    )
    pilot_physics = _schedule_prefix_physics_binding(pilot_schedule_set)
    target_physics = _schedule_prefix_physics_binding(target_schedule_set)
    pilot_initial_hashes = _initial_state_semantic_hashes(
        pilot_schedule_set, pilot_population, pilot_stage.tracer_count
    )
    target_initial_hashes = _initial_state_semantic_hashes(
        target_schedule_set, target_population, target_stage.tracer_count
    )
    pilot_source_hashes = {
        arm.schedule.arm_kind: arm.initial_state.state_sha256
        for arm in pilot_schedule_set.arms
    }
    target_source_hashes = {
        arm.schedule.arm_kind: arm.initial_state.state_sha256
        for arm in target_schedule_set.arms
    }
    gates = {
        "exact_stage_pair": (
            pilot_stage.total_steps == ROUTE_CAPACITY_PILOT_STEPS
            and target_stage.total_steps == 8_000_000
            and pilot_stage.tracer_count == target_stage.tracer_count == 16_800
        ),
        "exact_arm_rosters": (
            tuple(arm.schedule.arm_kind for arm in pilot_schedule_set.arms)
            == ARM_KINDS
            and tuple(arm.schedule.arm_kind for arm in target_schedule_set.arms)
            == ARM_KINDS
        ),
        "exact_source_activity_prefix": _source_activity_is_exact_prefix(
            pilot_schedule_set, target_schedule_set
        ),
        "exact_prefix_physics": pilot_physics == target_physics,
        "exact_source_initial_states": pilot_source_hashes == target_source_hashes,
        "exact_population_receipt": (
            pilot_receipt.get("content_sha256")
            == target_receipt.get("content_sha256")
        ),
        "exact_population_semantic_arrays": (
            population_arrays_equal
            and pilot_receipt.get("output", {}).get("semantic_array_sha256")
            == target_receipt.get("output", {}).get("semantic_array_sha256")
        ),
        "exact_full_initial_states": pilot_initial_hashes == target_initial_hashes,
    }
    return {
        "pilot_stage_id": pilot_stage.stage_id,
        "pilot_total_steps": pilot_stage.total_steps,
        "target_stage_id": target_stage.stage_id,
        "target_total_steps": target_stage.total_steps,
        "tracer_count_per_arm": target_stage.tracer_count,
        "pilot_population_array_names": list(pilot_population_keys),
        "target_population_array_names": list(target_population_keys),
        "pilot_population_transform_receipt_sha256": pilot_receipt.get(
            "content_sha256"
        ),
        "target_population_transform_receipt_sha256": target_receipt.get(
            "content_sha256"
        ),
        "population_semantic_array_sha256": target_receipt.get("output", {}).get(
            "semantic_array_sha256"
        ),
        "pilot_schedule_prefix_physics": pilot_physics,
        "target_schedule_prefix_physics": target_physics,
        "pilot_source_initial_state_sha256": pilot_source_hashes,
        "target_source_initial_state_sha256": target_source_hashes,
        "pilot_full_initial_state_semantic_sha256": pilot_initial_hashes,
        "target_full_initial_state_semantic_sha256": target_initial_hashes,
        "gates": {**gates, "all_pass": all(gates.values())},
    }


def _build_protocol(
    stage: Stage,
    checkpoint_years: float,
    schedule_set: MatchedMechanismScheduleSet,
    receipt: Mapping[str, Any],
    device: str,
    capacity_target_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_steps = _years_to_steps(checkpoint_years, "checkpoint_years")
    if checkpoint_steps > stage.total_steps:
        raise ValueError("checkpoint cadence cannot exceed the modeled horizon")
    documents = _schedule_documents(schedule_set)
    protocol: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "run_family": SCHEMA,
        "stage": {
            "stage_id": stage.stage_id,
            "duration_years_hex": stage.duration_years.hex(),
            "step_years_hex": STEP_YEARS.hex(),
            "total_steps": stage.total_steps,
            "tracer_count": stage.tracer_count,
            "checkpoint_years_hex": float(checkpoint_years).hex(),
            "checkpoint_steps": checkpoint_steps,
            "checkpoint_policy": (
                "EVERY_ENGINE_SEGMENT_IMMUTABLE_HASH_CHAIN; final short segment allowed"
            ),
            "engine_epoch_at_global_step_zero_years_hex": 0.0.hex(),
            "absolute_initial_epoch_jd_tdb_hex": DE441_CONTROL_EPOCH_JD_TDB.hex(),
            "clock_semantics": (
                "engine epoch is modeled elapsed years since the separately recorded "
                "absolute JD_TDB; JD is never inserted into the year-valued engine clock"
            ),
        },
        "backend": {
            "backend_id": "cupy",
            "device": device,
            "threads_per_block": 128,
            "source_path_steps_per_chunk": 65_536,
        },
        "integrator": {
            "method_id": METHOD_ID,
            "step_years_hex": STEP_YEARS.hex(),
            "maximum_encounter_routes_per_tracer": MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER,
            "maximum_total_encounter_routes": MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM,
            "encounter_route_method": "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2",
            "encounter_route_rtol_hex": (1.0e-12).hex(),
            "encounter_route_atol_hex": (1.0e-14).hex(),
            "encounter_route_maximum_substep_fraction_hex": (0.125).hex(),
            "maximum_primary_dynamical_phase_hex": MAXIMUM_PRIMARY_DYNAMICAL_PHASE.hex(),
            "event_time_tolerance_fraction_hex": EVENT_TIME_TOLERANCE_FRACTION.hex(),
            "event_distance_absolute_tolerance_floor_AU_hex": (1.0e-11).hex(),
            "event_distance_effective_tolerance_semantics": (
                "MAX(ABSOLUTE_FLOOR,256*BINARY64_EPS*MAX_COORDINATE_SCALE)"
            ),
            "outward_event_radial_velocity_tolerance_AU_per_year_hex": (
                OUTWARD_EVENT_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR.hex()
            ),
            "maximum_event_locator_accepted_steps": (
                MAXIMUM_EVENT_LOCATOR_ACCEPTED_STEPS
            ),
        },
        "safety": {
            "primary_outer_boundary_AU_hex": PRIMARY_OUTER_BOUNDARY_AU.hex(),
            "relative_acceleration_bound_AU_per_year2_hex": (
                RELATIVE_ACCELERATION_BOUND_AU_PER_YEAR2.hex()
            ),
            "primary_source_radius_semantics": "16_AU_INNER_MODEL_DOMAIN_EXIT_NEVER_COLLISION",
            "other_source_radius_semantics": "OPERATIONAL_REMOVAL_NOT_IMPACT_PHYSICS",
            "outer_surface_semantics": "10000_AU_MODEL_DOMAIN_EXIT_NOT_DEFINITIVE_EJECTION",
            "removed_lane_semantics": "FROZEN_AT_CONTINUOUSLY_LOCATED_EVENT_STATE",
        },
        "population": {
            "transform_receipt_sha256": receipt["content_sha256"],
            "transformed_population_sha256": receipt["output"]["semantic_array_sha256"],
            "selection": f"prefix [0:{stage.tracer_count}] in immutable acceptance order",
        },
        "schedule_set": {
            "matched_set_id": schedule_set.matched_set_id,
            "matched_set_sha256": schedule_set.matched_set_sha256,
            "arm_schedule_sha256": documents["matched_set"]["arm_schedule_sha256"],
            "arm_initial_state_sha256": documents["matched_set"]["arm_initial_state_sha256"],
            "nominal_passing_source_event_years": (
                schedule_set.stellar_flyby.schedule.diagnostic_window.nominal_event_global_step
                * STEP_YEARS
            ),
            "passing_source_environment_limit": (
                "finite-start engineering fixture; Galactic tide and Galactic/stellar "
                "environment are omitted"
            ),
        },
        "capacity_target_binding": (
            None
            if capacity_target_context is None
            else dict(capacity_target_context)
        ),
        "source_path_audit": {
            "all_lattice_nodes_used_for_pairwise_minima": True,
            "diagnostic_cadence_years_hex": SOURCE_DIAGNOSTIC_YEARS.hex(),
            "diagnostic_cadence_steps": _years_to_steps(
                SOURCE_DIAGNOSTIC_YEARS, "source_diagnostic_years"
            ),
            "quantities": [
                "all_source_pair_minimum_distance_and_lattice_time",
                "primary_relative_minimum_distance_and_lattice_time_per_source",
                "scaled_newtonian_source_energy_drift",
                "gm_weighted_angular_momentum_drift",
                "gm_weighted_linear_momentum_drift",
                "source_endpoint_bitwise_identity_against_tracer_run",
                "exact_route_source_segment_membership",
                "full_stage_same_code_replay_of_every_routed_tracer",
            ],
        },
        "bindings": {
            "runner_sha256": _sha256_file(Path(__file__)),
            "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
            "kernel_source_sha256": KERNEL_SOURCE_SHA256,
            "event_locator_source_sha256": EVENT_LOCATOR_SOURCE_SHA256,
            "mechanism_schedule_source_sha256": _sha256_file(
                ROOT / "src/jxplanetx/cuda_mechanism_schedules.py"
            ),
            "acceptance_state_sha256": ACCEPTANCE_INITIAL_STATE_SHA256,
            "acceptance_protocol_sha256": ACCEPTANCE_PROTOCOL_SHA256,
            "de441_control_sha256": DE441_CONTROL_SHA256,
            "git": _git_identity(),
        },
        "qualification_gates": {
            "all_requested_arms_reach_exact_endpoint": True,
            "all_endpoint_arrays_finite": True,
            "route_ledger_matches_route_counts": True,
            "removal_ledger_matches_terminal_lifecycle": True,
            "runtime_method_kernel_and_implementation_bound": True,
            "source_endpoint_matches_all_horizon_source_path_audit": True,
            "every_route_source_segment_matches_exact_source_path": True,
            "every_routed_tracer_exact_full_stage_replay_passed": True,
            "committed_checkpoint_segment_roster_hashes_and_chain_verified": True,
            "uncommitted_checkpoint_array_orphans_preserved_separately": True,
            "matched_initial_tracer_bytes": True,
            "matched_base_source_prefix_bytes": True,
        },
        "route_ledger_budget": {
            "maximum_routes_per_tracer": MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER,
            "maximum_total_route_records_per_arm": (
                MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
            ),
            "maximum_total_route_records_across_four_arms": (
                MAXIMUM_TOTAL_ROUTE_RECORDS_PER_STAGE
            ),
            "minimum_free_storage_bytes_before_every_segment": (
                MINIMUM_FREE_STORAGE_BYTES
            ),
            "minimum_free_storage_bytes_before_every_checkpoint": (
                MINIMUM_FREE_STORAGE_BYTES
            ),
            "storage_guard_semantics": (
                "HARD_ENGINE_CUMULATIVE_ARM_ROUTE_CAP_AND_HOST_FREE_SPACE_"
                "PREFLIGHT; FAILS_BEFORE_EACH_SEGMENT_AND_CHECKPOINT_COMMIT; "
                "THE_PRE_WRITE_THRESHOLD_IS_INTENDED_TO_LEAVE_RECOVERY_MARGIN"
            ),
            "capacity_admission_policy": {
                "episode_definition": (
                    "MAXIMAL_CONSECUTIVE_ROUTED_MACROSTEPS_PER_TRACER; "
                    "SOURCE_AND_REASON_CHANGES_DO_NOT_SPLIT; ONE_QUIET_STEP_SPLITS"
                ),
                "reserved_route_records_per_future_episode": (
                    ROUTE_EPISODE_RESERVED_RECORDS
                ),
                "required_runtime_headroom_per_tracer": (
                    ROUTE_CAPACITY_REQUIRED_HEADROOM
                ),
                "admission_limit_per_tracer": ROUTE_CAPACITY_ADMISSION_LIMIT,
                "capacity_pilot_stage_id": "capacity_pilot",
                "required_capacity_pilot_steps": ROUTE_CAPACITY_PILOT_STEPS,
                "required_capacity_pilot_tracer_count_per_arm": 16_800,
                "required_target_stage_id": "one_myr",
                "required_target_steps": STAGES["one_myr"].total_steps,
                "required_arm_order": list(ARM_KINDS),
                "future_episode_slot_formula": (
                    "ACTIVE: CEIL((COMPLETED_EPISODES+1)*"
                    "(TARGET_STEPS-OBSERVED_STEPS)/OBSERVED_STEPS); TERMINAL: 0"
                ),
                "right_frontier_policy": (
                    "ACTIVE_REJOINED_EPISODE_AT_FINAL_OBSERVED_STEP_FAILS_CLOSED"
                ),
                "claim_limit": (
                    "ENGINEERING_RESOURCE_ADMISSION_NOT_COMPLETION_OR_DYNAMICAL_"
                    "RECURRENCE_PREDICTION"
                ),
            },
            "semantics": (
                "fail-closed capacity ceiling, not an expected count; observed maximum "
                "and remaining per-tracer headroom are reported at every checkpoint"
            ),
        },
        "numerical_scope": {
            "this_run_is_a_single_timestep_screen": True,
            "timestep_convergence_claimed": False,
            "promotion_rule": (
                "Before interpreting any selected contrast, replay a frozen shared subset of the "
                "control and selected arm at 0.0625 year with identical physical initial bytes and "
                "compare predeclared endpoint/tail metrics."
            ),
        },
        "classification": f"{SCIENTIFIC_CLASSIFICATION} / SCREENING_ONLY",
        "interpretation_limits": list(INTERPRETATION_LIMITS),
        "claims": dict(RESULT_CLAIMS),
    }
    protocol["content_sha256"] = _content_hash(PROTOCOL_DOMAIN, protocol)
    return protocol


def _provenance(receipt_sha256: str, arm: MechanismArm) -> Provenance:
    digest = hashlib.sha256(PROVENANCE_DOMAIN)
    digest.update(receipt_sha256.encode("ascii"))
    digest.update(b"\x00")
    digest.update(arm.initial_state.state_sha256.encode("ascii"))
    digest.update(b"\x00")
    digest.update(arm.schedule.schedule_sha256.encode("ascii"))
    return Provenance(
        source_id="jx.cuda-four-arm-current-epoch-screen-input.v1",
        citation=(
            "frozen DE441 outer-control artifact plus acceptance-v3 synthetic "
            "tracer transform receipt"
        ),
        version="m2-screen-v1",
        sha256=digest.hexdigest(),
    )


def _build_device_state(
    cp: Any,
    arm: MechanismArm,
    schedule_set: MatchedMechanismScheduleSet,
    population: Mapping[str, np.ndarray],
    receipt_sha256: str,
    tracer_count: int,
) -> CUDAMasslessTracerState:
    source = arm.initial_state
    encounter, collision, acceleration = _source_safety_arrays(arm)
    return CUDAMasslessTracerState(
        snapshot_id=f"{arm.schedule.schedule_id}:t0",
        epoch=0.0,
        global_step=0,
        time_scale=f"TDB_YEAR_OFFSET_FROM_JD_{DE441_CONTROL_EPOCH_JD_TDB:.1f}",
        frame=source.frame,
        origin=source.origin,
        axes="RIGHT_HANDED_ECLIPTIC_J2000",
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=source.source_ids,
        primary_source_id=source.primary_source_id,
        source_positions=cp.asarray(source.positions_AU),
        source_velocities=cp.asarray(source.velocities_AU_per_year),
        source_gm=cp.asarray(source.gm_AU3_per_year2),
        source_encounter_radii=cp.asarray(encounter),
        source_collision_radii=cp.asarray(collision),
        relative_acceleration_bounds=cp.asarray(acceleration),
        tracer_positions=cp.asarray(population["tracer_positions"][:tracer_count]),
        tracer_velocities=cp.asarray(population["tracer_velocities"][:tracer_count]),
        tracer_indices=cp.asarray(population["tracer_indices"][:tracer_count]),
        provenance=_provenance(receipt_sha256, arm),
        force_schedule_sha256=arm.schedule.schedule_sha256,
        matched_schedule_set_sha256=schedule_set.matched_set_sha256,
        primary_outer_boundary_AU=PRIMARY_OUTER_BOUNDARY_AU,
    )


def _initial_artifact_arrays(
    arm: MechanismArm,
    population: Mapping[str, np.ndarray],
    tracer_count: int,
) -> dict[str, np.ndarray]:
    encounter, collision, acceleration = _source_safety_arrays(arm)
    return {
        "source_positions": arm.initial_state.positions_AU,
        "source_velocities": arm.initial_state.velocities_AU_per_year,
        "source_gm": arm.initial_state.gm_AU3_per_year2,
        "source_encounter_radii": encounter,
        "source_collision_radii": collision,
        "relative_acceleration_bounds": acceleration,
        "tracer_positions": population["tracer_positions"][:tracer_count],
        "tracer_velocities": population["tracer_velocities"][:tracer_count],
        "tracer_indices": population["tracer_indices"][:tracer_count],
    }


def _latest_checkpoint_digest(
    checkpoint_root: Path,
    run_id: str,
    protocol_sha256: str,
) -> tuple[int, str] | None:
    matches: list[tuple[int, str]] = []
    if not checkpoint_root.exists():
        return None
    for path in checkpoint_root.glob("run-*/checkpoint_*.json"):
        document = _load_json(path)
        if (
            document.get("run_id") == run_id
            and document.get("protocol_sha256") == protocol_sha256
        ):
            index = document.get("checkpoint_index")
            digest = document.get("checkpoint_content_sha256")
            if type(index) is not int or type(digest) is not str:
                raise RuntimeError(f"invalid checkpoint identity: {path}")
            matches.append((index, digest))
    if not matches:
        return None
    matches.sort()
    if len({index for index, _ in matches}) != len(matches):
        raise RuntimeError("duplicate checkpoint indices were found")
    return matches[-1]


def _endpoint_arrays(cp: Any, result: CUDAMasslessTracerResult) -> dict[str, np.ndarray]:
    state = result.state
    return {
        "source_positions": cp.asnumpy(state.source_positions),
        "source_velocities": cp.asnumpy(state.source_velocities),
        "source_gm": cp.asnumpy(state.source_gm),
        "source_encounter_radii": cp.asnumpy(state.source_encounter_radii),
        "source_collision_radii": cp.asnumpy(state.source_collision_radii),
        "relative_acceleration_bounds": cp.asnumpy(state.relative_acceleration_bounds),
        "tracer_positions": cp.asnumpy(state.tracer_positions),
        "tracer_velocities": cp.asnumpy(state.tracer_velocities),
        "tracer_indices": cp.asnumpy(state.tracer_indices),
        "minimum_source_distance_AU": cp.asnumpy(result.minimum_source_distance_AU),
        "ever_unbound": cp.asnumpy(result.ever_unbound),
        "encounter_route_counts": cp.asnumpy(result.encounter_route_counts),
        "tracer_lifecycle": cp.asnumpy(result.tracer_lifecycle),
    }


def _finite_endpoint(arrays: Mapping[str, np.ndarray]) -> bool:
    for value in arrays.values():
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            return False
    return True


def _pair_minimum(positions: np.ndarray) -> float:
    if positions.shape[0] < 2:
        return math.inf
    values = []
    for left in range(positions.shape[0] - 1):
        delta = positions[left + 1 :] - positions[left]
        values.append(float(np.min(np.linalg.norm(delta, axis=1))))
    return min(values)


def _orbital_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    position = arrays["tracer_positions"] - arrays["source_positions"][0]
    velocity = arrays["tracer_velocities"] - arrays["source_velocities"][0]
    mu = float(arrays["source_gm"][0])
    radius = np.linalg.norm(position, axis=1)
    speed2 = np.einsum("ij,ij->i", velocity, velocity)
    energy = 0.5 * speed2 - mu / radius
    angular = np.cross(position, velocity)
    angular_norm = np.linalg.norm(angular, axis=1)
    eccentricity_vector = np.cross(velocity, angular) / mu - position / radius[:, None]
    eccentricity = np.linalg.norm(eccentricity_vector, axis=1)
    perihelion = angular_norm**2 / (mu * (1.0 + eccentricity))
    semimajor_axis = np.full(radius.shape, np.nan, dtype=np.float64)
    bound = energy < 0.0
    semimajor_axis[bound] = -mu / (2.0 * energy[bound])
    cos_i = np.divide(
        angular[:, 2],
        angular_norm,
        out=np.ones_like(angular_norm),
        where=angular_norm > 0.0,
    )
    inclination = np.arccos(np.clip(cos_i, -1.0, 1.0))
    return {
        "specific_energy_AU2_per_year2": energy,
        "semimajor_axis_AU": semimajor_axis,
        "eccentricity": eccentricity,
        "perihelion_AU": perihelion,
        "inclination_rad": inclination,
        "bound": bound,
    }


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {"minimum": None, "q05": None, "median": None, "q95": None, "maximum": None}
    q = np.quantile(finite, (0.0, 0.05, 0.5, 0.95, 1.0))
    return {
        "minimum": float(q[0]),
        "q05": float(q[1]),
        "median": float(q[2]),
        "q95": float(q[3]),
        "maximum": float(q[4]),
    }


def _endpoint_measurements(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    orbital = _orbital_arrays(arrays)
    lifecycle = arrays["tracer_lifecycle"]
    route_counts = arrays["encounter_route_counts"]
    maximum_route_count = int(np.max(route_counts)) if route_counts.size else 0
    active = lifecycle == LIFECYCLE_ACTIVE
    lifecycle_counts = Counter(int(value) for value in lifecycle)
    return {
        "orbital_element_semantics": (
            "INSTANTANEOUS_PRIMARY_RELATIVE_TWO_BODY_OSCULATING_PROXIES; "
            "OTHER_MASSIVE_SOURCES_OMITTED_FROM_ELEMENT_CONVERSION; "
            "SEMIMAJOR_AXIS_REPORTED_ONLY_FOR_BOUND_STATES"
        ),
        "tracer_count": int(lifecycle.size),
        "active_tracer_count": int(np.count_nonzero(active)),
        "lifecycle_counts": {
            LIFECYCLE_NAMES.get(code, f"UNKNOWN_{code}"): count
            for code, count in sorted(lifecycle_counts.items())
        },
        "ever_unbound_count": int(np.count_nonzero(arrays["ever_unbound"])),
        "encounter_route_count": int(np.sum(arrays["encounter_route_counts"])),
        "routed_tracer_count": int(np.count_nonzero(arrays["encounter_route_counts"])),
        "maximum_route_count_for_one_tracer": maximum_route_count,
        "route_capacity_per_tracer": MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER,
        "minimum_remaining_route_capacity_for_any_tracer": (
            MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER - maximum_route_count
        ),
        "maximum_route_capacity_fraction_used": (
            maximum_route_count / MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
        ),
        "minimum_evaluated_source_distance_AU": float(
            np.min(arrays["minimum_source_distance_AU"])
        ),
        "endpoint_minimum_massive_source_pair_distance_AU": _pair_minimum(
            arrays["source_positions"]
        ),
        "active_bound_count": int(np.count_nonzero(active & orbital["bound"])),
        "active_perihelion_AU": _quantiles(orbital["perihelion_AU"][active]),
        "active_semimajor_axis_AU": _quantiles(orbital["semimajor_axis_AU"][active]),
        "active_eccentricity": _quantiles(orbital["eccentricity"][active]),
        "active_inclination_rad": _quantiles(orbital["inclination_rad"][active]),
        "exploratory_tail_counts": {
            "active_q_ge_40_AU": int(np.count_nonzero(active & (orbital["perihelion_AU"] >= 40.0))),
            "active_q_ge_50_AU": int(np.count_nonzero(active & (orbital["perihelion_AU"] >= 50.0))),
            "active_a_ge_250_AU_q_ge_40_AU": int(
                np.count_nonzero(
                    active
                    & (orbital["semimajor_axis_AU"] >= 250.0)
                    & (orbital["perihelion_AU"] >= 40.0)
                )
            ),
        },
    }


def _segment_report(
    result: CUDAMasslessTracerResult,
    arrays: Mapping[str, np.ndarray],
    elapsed: float,
    checkpoint: Mapping[str, Any],
    route_storage_guard: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "jxplanetx.cuda-four-arm-segment.v1",
        "completed_utc": _utc_now(),
        "global_step": result.state.global_step,
        "epoch_years": result.state.epoch,
        "elapsed_seconds": elapsed,
        "checkpoint_content_sha256": checkpoint["checkpoint_content_sha256"],
        "state_sha256": checkpoint["state_sha256"],
        "input_state_sha256": result.input_state_sha256,
        "measurements": _endpoint_measurements(arrays),
        "route_storage_guard": dict(route_storage_guard),
        "classification": "MODEL_OUTPUT / SCREENING_ONLY",
    }


def _expected_checkpoint_indices(
    total_steps: int,
    checkpoint_steps: int,
) -> tuple[int, ...]:
    if type(total_steps) is not int or total_steps <= 0:
        raise RuntimeError("execution evidence total_steps is invalid")
    if type(checkpoint_steps) is not int or not 0 < checkpoint_steps <= total_steps:
        raise RuntimeError("execution evidence checkpoint_steps is invalid")
    indices: list[int] = []
    cursor = 0
    while cursor < total_steps:
        cursor = min(total_steps, cursor + checkpoint_steps)
        indices.append(cursor)
    return tuple(indices)


def _is_sha256_text(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _execution_artifact_map(
    paths: Iterable[Path],
    *,
    stage_directory: Path,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("execution evidence artifact is not a regular file")
        try:
            path.resolve().relative_to(stage_directory.resolve())
        except ValueError as exc:
            raise RuntimeError("execution evidence artifact escapes the stage") from exc
        relative = path.relative_to(stage_directory).as_posix()
        if relative in records:
            raise RuntimeError("execution evidence contains a duplicate artifact path")
        record = _artifact(path, stage_directory)
        if record["path"] != relative:
            raise RuntimeError("execution evidence artifact path is noncanonical")
        records[relative] = record
    return records


def _collect_execution_evidence(
    *,
    stage_directory: Path,
    arm_directory: Path,
    run_id: str,
    protocol_sha256: str,
    total_steps: int,
    checkpoint_steps: int,
) -> dict[str, Any]:
    """Bind the exact committed checkpoint chain and its segment reports.

    A content-addressed NPZ may survive a process interruption before its JSON
    record is atomically installed.  Such files are retained as explicit
    negative evidence, but they are never counted as committed checkpoints.
    """

    expected_indices = _expected_checkpoint_indices(total_steps, checkpoint_steps)
    checkpoint_root = arm_directory / "checkpoints"
    segment_root = arm_directory / "segments"
    if not checkpoint_root.is_dir() or not segment_root.is_dir():
        raise RuntimeError("completed arm lacks checkpoint or segment evidence")

    namespace_directories = sorted(
        path for path in checkpoint_root.iterdir() if path.is_dir()
    )
    root_files = sorted(path for path in checkpoint_root.iterdir() if path.is_file())
    if root_files or len(namespace_directories) != 1:
        raise RuntimeError("checkpoint evidence has a noncanonical namespace roster")
    namespace = namespace_directories[0]
    namespace_name = namespace.name
    if (
        not namespace_name.startswith("run-")
        or not _is_sha256_text(namespace_name.removeprefix("run-"))
    ):
        raise RuntimeError("checkpoint evidence namespace is invalid")

    committed_paths: list[Path] = []
    committed_set: set[Path] = set()
    latest_record: dict[str, Any] | None = None
    previous_route_record_count = 0
    for index in expected_indices:
        record_path = namespace / f"checkpoint_{index:020d}.json"
        if not record_path.is_file():
            raise RuntimeError(f"checkpoint evidence is missing index {index}")
        record = _load_json(record_path)
        if (
            record.get("run_id") != run_id
            or record.get("protocol_sha256") != protocol_sha256
            or record.get("checkpoint_namespace") != namespace_name
            or record.get("checkpoint_index") != index
            or not _is_sha256_text(record.get("checkpoint_content_sha256"))
            or not _is_sha256_text(record.get("state_sha256"))
        ):
            raise RuntimeError(f"checkpoint evidence identity failed at index {index}")
        array_digest = record.get("array_file_sha256")
        array_name = record.get("array_file")
        expected_array_name = f"checkpoint_{index:020d}.array-{array_digest}.npz"
        if (
            not _is_sha256_text(array_digest)
            or type(array_name) is not str
            or array_name != expected_array_name
            or Path(array_name).name != array_name
        ):
            raise RuntimeError(
                f"checkpoint array path identity failed at index {index}"
            )
        array_path = namespace / array_name
        if (
            not array_path.is_file()
            or array_path.stat().st_size != record.get("array_file_size_bytes")
            or _sha256_file(array_path) != array_digest
        ):
            raise RuntimeError(f"checkpoint array failed at index {index}")

        segment_path = segment_root / f"segment-{index:020d}.json"
        if not segment_path.is_file():
            raise RuntimeError(f"segment evidence is missing index {index}")
        segment = _load_json(segment_path)
        if (
            segment.get("schema") != "jxplanetx.cuda-four-arm-segment.v1"
            or segment.get("global_step") != index
            or segment.get("checkpoint_content_sha256")
            != record.get("checkpoint_content_sha256")
            or segment.get("state_sha256") != record.get("state_sha256")
        ):
            raise RuntimeError(f"segment/checkpoint binding failed at index {index}")
        storage_guard = segment.get("route_storage_guard")
        measurements = segment.get("measurements")
        current_route_record_count = (
            measurements.get("encounter_route_count")
            if type(measurements) is dict
            else None
        )
        if (
            type(storage_guard) is not dict
            or type(current_route_record_count) is not int
            or current_route_record_count < previous_route_record_count
        ):
            raise RuntimeError(f"segment storage guard is invalid at index {index}")
        for guard_name, expected_count in (
            ("pre_segment", previous_route_record_count),
            ("pre_checkpoint", current_route_record_count),
        ):
            guard = storage_guard.get(guard_name)
            if (
                type(guard) is not dict
                or guard.get("cumulative_route_records") != expected_count
                or guard.get("maximum_total_route_records_per_arm")
                != MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
                or guard.get("required_minimum_free_storage_bytes")
                != MINIMUM_FREE_STORAGE_BYTES
                or type(guard.get("observed_free_storage_bytes")) is not int
                or guard["observed_free_storage_bytes"]
                < MINIMUM_FREE_STORAGE_BYTES
                or guard.get("passed") is not True
            ):
                raise RuntimeError(
                    f"segment {guard_name} storage guard failed at index {index}"
                )
        previous_route_record_count = current_route_record_count

        for path in (record_path, array_path, segment_path):
            if path in committed_set:
                raise RuntimeError("execution evidence reuses an artifact path")
            committed_set.add(path)
            committed_paths.append(path)
        latest_record = record

    observed_segments = {
        path for path in segment_root.rglob("*") if path.is_file()
    }
    expected_segments = {
        segment_root / f"segment-{index:020d}.json" for index in expected_indices
    }
    if observed_segments != expected_segments:
        raise RuntimeError("segment artifact roster differs from the checkpoint cadence")

    observed_checkpoint_files = {
        path for path in checkpoint_root.rglob("*") if path.is_file()
    }
    committed_checkpoint_files = {
        path for path in committed_set if checkpoint_root in path.parents
    }
    orphan_paths: list[Path] = []
    for path in sorted(observed_checkpoint_files - committed_checkpoint_files):
        if path.parent != namespace:
            raise RuntimeError("checkpoint evidence contains a file outside its namespace")
        name = path.name
        stem, separator, digest_suffix = name.partition(".array-")
        digest, extension = (
            digest_suffix.rsplit(".", 1)
            if "." in digest_suffix
            else ("", "")
        )
        index_text = stem.removeprefix("checkpoint_")
        if (
            separator != ".array-"
            or not stem.startswith("checkpoint_")
            or len(index_text) != 20
            or not index_text.isdigit()
            or int(index_text) not in expected_indices
            or extension != "npz"
            or not _is_sha256_text(digest)
            or _sha256_file(path) != digest
        ):
            raise RuntimeError(
                "checkpoint evidence contains an unknown or malformed extra file"
            )
        orphan_paths.append(path)

    if latest_record is None:
        raise RuntimeError("execution evidence did not yield a terminal checkpoint")
    committed_artifacts = _execution_artifact_map(
        committed_paths,
        stage_directory=stage_directory,
    )
    orphan_artifacts = _execution_artifact_map(
        orphan_paths,
        stage_directory=stage_directory,
    )
    evidence: dict[str, Any] = {
        "schema": EXECUTION_EVIDENCE_SCHEMA,
        "run_id": run_id,
        "protocol_sha256": protocol_sha256,
        "total_steps": total_steps,
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_indices": list(expected_indices),
        "checkpoint_namespace": namespace_name,
        "latest_checkpoint_content_sha256": latest_record[
            "checkpoint_content_sha256"
        ],
        "latest_state_sha256": latest_record["state_sha256"],
        "committed_artifact_count": len(committed_artifacts),
        "committed_artifacts": committed_artifacts,
        "uncommitted_orphan_artifact_count": len(orphan_artifacts),
        "uncommitted_orphan_artifacts": orphan_artifacts,
        "uncommitted_orphan_semantics": (
            "CONTENT_ADDRESSED_NPZ_PRESERVED_AFTER_INTERRUPTED_PRE_RECORD_COMMIT; "
            "NOT_A_COMMITTED_CHECKPOINT_AND_NOT_IN_THE_HASH_CHAIN"
        ),
    }
    evidence["content_sha256"] = _content_hash(
        EXECUTION_EVIDENCE_DOMAIN, evidence
    )
    return evidence


def _verify_loaded_execution_endpoint(
    *,
    backend: BackendSpec,
    stage: Stage,
    arm_directory: Path,
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    endpoint: Mapping[str, np.ndarray],
    route_records: list[dict[str, Any]],
    removal_records: list[dict[str, Any]],
) -> None:
    checkpoint_root = arm_directory / "checkpoints"
    loaded = load_latest_cuda_massless_tracer_checkpoint(
        checkpoint_root,
        run_id=str(report["run_id"]),
        protocol_sha256=str(report["protocol_sha256"]),
        backend=backend,
    )
    if loaded is None:
        raise RuntimeError("committed checkpoint evidence did not reload")
    latest = _latest_checkpoint_digest(
        checkpoint_root,
        str(report["run_id"]),
        str(report["protocol_sha256"]),
    )
    if latest != (
        stage.total_steps,
        evidence.get("latest_checkpoint_content_sha256"),
    ):
        raise RuntimeError("reloaded checkpoint terminal digest differs")
    if (
        loaded.state.global_step != stage.total_steps
        or loaded.state.epoch != stage.duration_years
        or report.get("endpoint_global_step") != stage.total_steps
        or report.get("endpoint_epoch_years") != stage.duration_years
    ):
        raise RuntimeError("reloaded checkpoint did not reach the exact stage endpoint")
    _require_runtime_identity(
        dict(loaded.runtime_identity),
        report.get("runtime_identity", {}),
        label="reloaded checkpoint/arm result runtime",
    )
    resolved = cuda_tracer_module.resolve_backend(backend)
    with resolved.activate():
        loaded_endpoint = _endpoint_arrays(resolved.xp, loaded)
    if set(loaded_endpoint) != set(endpoint) or any(
        not np.array_equal(loaded_endpoint[key], endpoint[key])
        for key in loaded_endpoint
    ):
        raise RuntimeError("reloaded checkpoint endpoint differs from ENDPOINT_STATE")
    if _canonical([asdict(row) for row in loaded.encounter_routes]) != _canonical(
        route_records
    ):
        raise RuntimeError("reloaded checkpoint route ledger differs")
    if _canonical([asdict(row) for row in loaded.removal_events]) != _canonical(
        removal_records
    ):
        raise RuntimeError("reloaded checkpoint removal ledger differs")


def _verify_execution_evidence(
    *,
    stage_directory: Path,
    arm_directory: Path,
    stage: Stage,
    checkpoint_years: float,
    report: Mapping[str, Any],
    endpoint: Mapping[str, np.ndarray],
    route_records: list[dict[str, Any]],
    removal_records: list[dict[str, Any]],
    backend: BackendSpec | None,
) -> dict[str, Any]:
    checkpoint_steps = _years_to_steps(checkpoint_years, "checkpoint_years")
    observed = report.get("execution_evidence")
    if type(observed) is not dict:
        raise RuntimeError("arm result lacks execution evidence")
    _verify_content_hash(
        observed,
        EXECUTION_EVIDENCE_DOMAIN,
        schema=EXECUTION_EVIDENCE_SCHEMA,
        label="arm execution evidence",
    )
    recomputed = _collect_execution_evidence(
        stage_directory=stage_directory,
        arm_directory=arm_directory,
        run_id=str(report.get("run_id")),
        protocol_sha256=str(report.get("protocol_sha256")),
        total_steps=stage.total_steps,
        checkpoint_steps=checkpoint_steps,
    )
    if observed != recomputed:
        raise RuntimeError("arm execution evidence differs from its exact artifact roster")
    if backend is not None:
        _verify_loaded_execution_endpoint(
            backend=backend,
            stage=stage,
            arm_directory=arm_directory,
            report=report,
            evidence=recomputed,
            endpoint=endpoint,
            route_records=route_records,
            removal_records=removal_records,
        )
    return recomputed


def _arm_run_id(stage: Stage, arm: MechanismArm) -> str:
    return f"jx-m2-{stage.stage_id}-{ARM_DIRECTORY_NAMES[arm.schedule.arm_kind]}-v1"


def _arm_result_path(stage_directory: Path, arm: MechanismArm) -> Path:
    return stage_directory / "arms" / ARM_DIRECTORY_NAMES[arm.schedule.arm_kind] / "ARM_RESULT.json"


def _load_verified_arm_result(
    path: Path,
    *,
    backend: BackendSpec,
    stage: Stage,
    checkpoint_years: float,
    stage_directory: Path,
    arm: MechanismArm,
    schedule_set: MatchedMechanismScheduleSet,
    protocol_sha256: str,
    receipt_sha256: str,
    runtime_identity: Mapping[str, Any],
    runtime_receipt_sha256: str,
) -> dict[str, Any]:
    report = _load_json(path)
    _verify_content_hash(
        report,
        REPORT_DOMAIN,
        schema=ARM_RESULT_SCHEMA,
        label=f"arm result {arm.schedule.arm_kind}",
    )
    if (
        report.get("arm_kind") != arm.schedule.arm_kind
        or report.get("protocol_sha256") != protocol_sha256
        or report.get("schedule_sha256") != arm.schedule.schedule_sha256
        or report.get("source_state_sha256") != arm.initial_state.state_sha256
        or report.get("matched_schedule_set_sha256")
        != schedule_set.matched_set_sha256
        or report.get("population_transform_receipt_sha256") != receipt_sha256
        or report.get("runtime_identity_receipt_sha256")
        != runtime_receipt_sha256
        or report.get("claims") != dict(RESULT_CLAIMS)
        or report.get("classification") != "MODEL_OUTPUT / SCREENING_ONLY"
    ):
        raise RuntimeError("cached arm result identity or claim ceiling differs")
    _require_runtime_identity(
        report.get("runtime_identity"),
        runtime_identity,
        label=f"cached {arm.schedule.arm_kind} arm result",
    )
    arm_relative = Path("arms") / ARM_DIRECTORY_NAMES[arm.schedule.arm_kind]
    artifact_records = report.get("artifacts")
    if (
        type(artifact_records) is not dict
        or set(artifact_records) != REQUIRED_ARM_ARTIFACT_NAMES
    ):
        raise RuntimeError("cached arm result artifact roster is incomplete")
    for name, record in artifact_records.items():
        if (
            type(record) is not dict
            or record.get("path") != str(arm_relative / name)
        ):
            raise RuntimeError("cached arm result artifact path is noncanonical")
    _verify_artifact_records(
        artifact_records,
        stage_directory=stage_directory,
        label=f"arm result {arm.schedule.arm_kind}",
    )
    arm_directory = path.parent
    endpoint_path = arm_directory / "ENDPOINT_STATE.npz"
    with np.load(endpoint_path, allow_pickle=False) as archive:
        endpoint = {
            name: np.ascontiguousarray(archive[name]) for name in archive.files
        }
    if _array_set_sha256(endpoint) != report.get("endpoint_semantic_array_sha256"):
        raise RuntimeError("cached arm endpoint semantic hash failed")
    route_document = _load_json(arm_directory / "ENCOUNTER_ROUTES.json")
    removal_document = _load_json(arm_directory / "REMOVAL_EVENTS.json")
    if (
        route_document.get("schema")
        != "jxplanetx.cuda-four-arm-route-ledger.v1"
        or route_document.get("arm_kind") != arm.schedule.arm_kind
        or type(route_document.get("records")) is not list
        or route_document.get("count") != len(route_document["records"])
        or removal_document.get("schema")
        != "jxplanetx.cuda-four-arm-removal-ledger.v1"
        or removal_document.get("arm_kind") != arm.schedule.arm_kind
        or type(removal_document.get("records")) is not list
        or removal_document.get("count") != len(removal_document["records"])
    ):
        raise RuntimeError("cached arm route/removal ledger is invalid")
    final_storage_guards = report.get("route_storage_guards")
    if type(final_storage_guards) is not list or len(final_storage_guards) != 4:
        raise RuntimeError("cached arm final storage-guard roster is invalid")
    for guard in final_storage_guards:
        if (
            type(guard) is not dict
            or guard.get("cumulative_route_records") != route_document["count"]
            or guard.get("maximum_total_route_records_per_arm")
            != MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
            or guard.get("required_minimum_free_storage_bytes")
            != MINIMUM_FREE_STORAGE_BYTES
            or type(guard.get("observed_free_storage_bytes")) is not int
            or guard["observed_free_storage_bytes"] < MINIMUM_FREE_STORAGE_BYTES
            or guard.get("passed") is not True
        ):
            raise RuntimeError("cached arm final storage guard is invalid")
    if route_document["count"] != int(np.sum(endpoint["encounter_route_counts"])):
        raise RuntimeError("cached arm route ledger disagrees with endpoint counts")
    if removal_document["count"] != int(
        np.count_nonzero(endpoint["tracer_lifecycle"] != LIFECYCLE_ACTIVE)
    ):
        raise RuntimeError("cached arm removal ledger disagrees with endpoint lifecycle")
    source_audit = _load_json(arm_directory / "SOURCE_PATH_AUDIT.json")
    _verify_content_hash(
        source_audit,
        REPORT_DOMAIN,
        schema="jxplanetx.cuda-four-arm-source-path-audit.v1",
        label=f"source-path audit {arm.schedule.arm_kind}",
    )
    if (
        source_audit.get("protocol_sha256") != protocol_sha256
        or source_audit.get("schedule_sha256") != arm.schedule.schedule_sha256
        or source_audit.get("method_id") != METHOD_ID
        or source_audit.get("kernel_source_sha256") != KERNEL_SOURCE_SHA256
        or source_audit.get("implementation_source_sha256")
        != IMPLEMENTATION_SOURCE_SHA256
        or source_audit.get("runtime_identity_receipt_sha256")
        != runtime_receipt_sha256
    ):
        raise RuntimeError("cached source-path audit binding differs")
    _require_runtime_identity(
        source_audit.get("runtime_identity"),
        runtime_identity,
        label=f"cached {arm.schedule.arm_kind} source-path audit",
    )
    _verify_execution_evidence(
        stage_directory=stage_directory,
        arm_directory=arm_directory,
        stage=stage,
        checkpoint_years=checkpoint_years,
        report=report,
        endpoint=endpoint,
        route_records=route_document["records"],
        removal_records=removal_document["records"],
        backend=backend,
    )
    return report


def _verify_stage_sha256sums(stage_directory: Path) -> None:
    checksum_path = stage_directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise RuntimeError("completed stage is missing SHA256SUMS")
    observed: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or not parts[1]
            or parts[1] in observed
        ):
            raise RuntimeError("completed stage SHA256SUMS has an invalid row")
        relative = Path(parts[1])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("completed stage SHA256SUMS path is unsafe")
        observed[parts[1]] = parts[0]
    expected_paths = sorted(
        path.relative_to(stage_directory).as_posix()
        for path in stage_directory.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    if sorted(observed) != expected_paths:
        raise RuntimeError("completed stage SHA256SUMS roster differs")
    for relative, digest in observed.items():
        if _sha256_file(stage_directory / relative) != digest:
            raise RuntimeError(f"completed stage checksum failed: {relative}")


def _finalized_summary_gates(stage: Stage, status: str) -> dict[str, bool]:
    gates = {
        "all_four_arms_complete": True,
        "all_arm_gates_pass": True,
    }
    if stage.stage_id == "capacity_pilot":
        gates.update(
            {
                "route_episode_capacity_gate_required": True,
                "route_episode_capacity_gate_passed": (
                    status == stage.completion_status
                ),
            }
        )
    if status == stage.completion_status:
        gates["all_pass"] = True
        return gates
    if stage.stage_id == "capacity_pilot" and status == "FAILED_ROUTE_CAPACITY_GATE":
        gates["all_pass"] = False
        return gates
    raise RuntimeError("stage summary does not have a supported finalized status")


def _load_partial_summary_commit(path: Path) -> dict[str, Any]:
    """Load a create-only summary left just before its SHA256SUMS commit."""

    if not path.is_file() or (path.parent / "SHA256SUMS").exists():
        raise RuntimeError("stage is not in the recoverable summary-only state")
    summary = _load_json(path)
    _verify_content_hash(
        summary,
        REPORT_DOMAIN,
        schema=SUMMARY_SCHEMA,
        label="partial stage summary",
    )
    completed_utc = summary.get("completed_utc")
    if type(completed_utc) is not str or not completed_utc:
        raise RuntimeError("partial stage summary has an invalid completion time")
    return summary


def _load_verified_stage_summary(
    path: Path,
    *,
    allowed_statuses: tuple[str, ...],
    expected_runtime_identity: Mapping[str, Any] | None = None,
    backend: BackendSpec | None = None,
) -> dict[str, Any]:
    summary = _load_json(path)
    _verify_content_hash(
        summary,
        REPORT_DOMAIN,
        schema=SUMMARY_SCHEMA,
        label="stage summary",
    )
    if (
        summary.get("status") not in allowed_statuses
        or summary.get("claims") != dict(RESULT_CLAIMS)
        or summary.get("classification") != "MODEL_OUTPUT / SCREENING_ONLY"
    ):
        raise RuntimeError("completed stage summary status/claim gates are invalid")
    stage_id = summary.get("stage_id")
    stage = STAGES.get(stage_id) if type(stage_id) is str else None
    checkpoint_years = summary.get("checkpoint_years")
    if (
        stage is None
        or summary.get("duration_years") != stage.duration_years
        or summary.get("step_years") != STEP_YEARS
        or summary.get("total_steps") != stage.total_steps
        or summary.get("tracer_count") != stage.tracer_count
        or isinstance(checkpoint_years, bool)
        or not isinstance(checkpoint_years, (int, float))
    ):
        raise RuntimeError("completed stage summary does not name a canonical stage")
    expected_gates = _finalized_summary_gates(stage, str(summary["status"]))
    if summary.get("gates") != expected_gates:
        raise RuntimeError("completed stage summary gates are invalid")
    checkpoint_years = float(checkpoint_years)
    checkpoint_steps = _years_to_steps(
        checkpoint_years, "completed checkpoint_years"
    )
    stage_directory = path.parent
    protocol = _load_json(stage_directory / "PROTOCOL.json")
    _verify_content_hash(
        protocol,
        PROTOCOL_DOMAIN,
        schema=PROTOCOL_SCHEMA,
        label="completed stage protocol",
    )
    protocol_stage = protocol.get("stage")
    if (
        protocol.get("content_sha256") != summary.get("protocol_sha256")
        or type(protocol_stage) is not dict
        or protocol_stage.get("stage_id") != stage.stage_id
        or protocol_stage.get("total_steps") != stage.total_steps
        or protocol_stage.get("tracer_count") != stage.tracer_count
        or protocol_stage.get("checkpoint_steps") != checkpoint_steps
        or protocol_stage.get("checkpoint_years_hex") != checkpoint_years.hex()
    ):
        raise RuntimeError("completed stage protocol/cadence binding differs")
    runtime_receipt = _load_json(stage_directory / "RUNTIME_IDENTITY.json")
    _verify_content_hash(
        runtime_receipt,
        RUNTIME_RECEIPT_DOMAIN,
        schema=RUNTIME_RECEIPT_SCHEMA,
        label="completed stage runtime receipt",
    )
    if (
        runtime_receipt.get("protocol_sha256") != summary.get("protocol_sha256")
        or summary.get("runtime_identity_receipt_sha256")
        != runtime_receipt.get("content_sha256")
    ):
        raise RuntimeError("completed stage runtime receipt binding differs")
    frozen_runtime = runtime_receipt.get("runtime_identity")
    if type(frozen_runtime) is not dict:
        raise RuntimeError("completed stage runtime receipt identity is invalid")
    _require_runtime_identity(
        summary.get("runtime_identity"),
        frozen_runtime,
        label="completed stage summary",
    )
    if expected_runtime_identity is not None:
        _require_runtime_identity(
            frozen_runtime,
            expected_runtime_identity,
            label="completed stage runtime receipt/current runtime",
        )
    result_hashes = summary.get("arm_result_sha256")
    if type(result_hashes) is not dict or set(result_hashes) != set(ARM_KINDS):
        raise RuntimeError("completed stage summary arm-result roster is invalid")
    for arm_kind in ARM_KINDS:
        result_path = (
            stage_directory
            / "arms"
            / ARM_DIRECTORY_NAMES[arm_kind]
            / "ARM_RESULT.json"
        )
        report = _load_json(result_path)
        _verify_content_hash(
            report,
            REPORT_DOMAIN,
            schema=ARM_RESULT_SCHEMA,
            label=f"stage arm result {arm_kind}",
        )
        if (
            report.get("arm_kind") != arm_kind
            or report.get("stage_id") != stage.stage_id
            or report.get("protocol_sha256") != summary.get("protocol_sha256")
            or report.get("endpoint_global_step") != stage.total_steps
            or report.get("endpoint_epoch_years") != stage.duration_years
            or report.get("content_sha256") != result_hashes[arm_kind]
            or report.get("claims") != dict(RESULT_CLAIMS)
            or report.get("classification") != "MODEL_OUTPUT / SCREENING_ONLY"
            or report.get("gates", {}).get("all_pass") is not True
            or report.get("runtime_identity_receipt_sha256")
            != runtime_receipt.get("content_sha256")
        ):
            raise RuntimeError("completed stage arm result binding is invalid")
        _require_runtime_identity(
            report.get("runtime_identity"),
            frozen_runtime,
            label=f"completed stage arm result {arm_kind}",
        )
        artifact_records = report.get("artifacts")
        if (
            type(artifact_records) is not dict
            or set(artifact_records) != REQUIRED_ARM_ARTIFACT_NAMES
        ):
            raise RuntimeError("completed stage arm artifact roster is incomplete")
        arm_relative = Path("arms") / ARM_DIRECTORY_NAMES[arm_kind]
        if any(
            type(record) is not dict
            or record.get("path") != str(arm_relative / name)
            for name, record in artifact_records.items()
        ):
            raise RuntimeError("completed stage arm artifact path is noncanonical")
        _verify_artifact_records(
            artifact_records,
            stage_directory=stage_directory,
            label=f"stage arm result {arm_kind}",
        )
        source_audit = _load_json(result_path.parent / "SOURCE_PATH_AUDIT.json")
        _verify_content_hash(
            source_audit,
            REPORT_DOMAIN,
            schema="jxplanetx.cuda-four-arm-source-path-audit.v1",
            label=f"completed stage source-path audit {arm_kind}",
        )
        if (
            source_audit.get("runtime_identity_receipt_sha256")
            != runtime_receipt.get("content_sha256")
        ):
            raise RuntimeError(
                "completed stage source-path audit runtime receipt binding differs"
            )
        _require_runtime_identity(
            source_audit.get("runtime_identity"),
            frozen_runtime,
            label=f"completed stage source-path audit {arm_kind}",
        )
        endpoint_path = result_path.parent / "ENDPOINT_STATE.npz"
        with np.load(endpoint_path, allow_pickle=False) as archive:
            endpoint = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
        route_document = _load_json(result_path.parent / "ENCOUNTER_ROUTES.json")
        removal_document = _load_json(result_path.parent / "REMOVAL_EVENTS.json")
        if (
            type(route_document.get("records")) is not list
            or type(removal_document.get("records")) is not list
        ):
            raise RuntimeError("completed stage arm ledgers are invalid")
        _verify_execution_evidence(
            stage_directory=stage_directory,
            arm_directory=result_path.parent,
            stage=stage,
            checkpoint_years=checkpoint_years,
            report=report,
            endpoint=endpoint,
            route_records=route_document["records"],
            removal_records=removal_document["records"],
            backend=backend,
        )
    _verify_stage_sha256sums(stage_directory)
    if stage.stage_id == "capacity_pilot":
        admission = _load_verified_capacity_admission(
            stage_directory / "CAPACITY_ADMISSION.json",
            expected_authorized=(
                summary.get("status") == stage.completion_status
            ),
        )
        if (
            summary.get("capacity_admission_sha256")
            != admission.get("content_sha256")
            or admission.get("source_protocol_sha256")
            != summary.get("protocol_sha256")
            or admission.get("source_runtime_identity_receipt_sha256")
            != summary.get("runtime_identity_receipt_sha256")
            or admission.get("episode_capacity_screen")
            != summary.get("route_episode_capacity_screen")
            or admission.get("arm_result_sha256")
            != summary.get("arm_result_sha256")
        ):
            raise RuntimeError(
                "capacity pilot summary/admission receipt binding differs"
            )
    return summary


def _write_failure(stage_directory: Path, arm_kind: str, exception: BaseException) -> None:
    failure_directory = stage_directory / "failures"
    failure_directory.mkdir(parents=True, exist_ok=True)
    identity = f"{time.time_ns()}-{os.getpid()}"
    document = {
        "schema": FAILURE_SCHEMA,
        "failed_utc": _utc_now(),
        "arm_kind": arm_kind,
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "traceback": traceback.format_exc(),
        "classification": "NEGATIVE_OR_FAILURE_RESULT_PRESERVED",
    }
    _atomic_bytes_create(failure_directory / f"failure-{identity}.json", _json_payload(document))


def _has_finalized_stage_manifest(stage_directory: Path) -> bool:
    return (
        (stage_directory / "SUMMARY.json").is_file()
        and (stage_directory / "SHA256SUMS").is_file()
    )


def _has_stage_finalization_artifact(stage_directory: Path) -> bool:
    return (
        (stage_directory / "SUMMARY.json").is_file()
        or (stage_directory / "SHA256SUMS").is_file()
    )


def _run_arm(
    *,
    cp: Any,
    backend: BackendSpec,
    stage: Stage,
    checkpoint_years: float,
    stage_directory: Path,
    schedule_set: MatchedMechanismScheduleSet,
    arm: MechanismArm,
    population: Mapping[str, np.ndarray],
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    runtime_receipt_sha256: str,
    stop_after_checkpoints: int | None,
) -> dict[str, Any] | None:
    arm_directory = stage_directory / "arms" / ARM_DIRECTORY_NAMES[arm.schedule.arm_kind]
    result_path = arm_directory / "ARM_RESULT.json"
    protocol_sha256 = str(protocol["content_sha256"])
    if result_path.exists():
        return _load_verified_arm_result(
            result_path,
            backend=backend,
            stage=stage,
            checkpoint_years=checkpoint_years,
            stage_directory=stage_directory,
            arm=arm,
            schedule_set=schedule_set,
            protocol_sha256=protocol_sha256,
            receipt_sha256=str(receipt["content_sha256"]),
            runtime_identity=runtime_identity,
            runtime_receipt_sha256=runtime_receipt_sha256,
        )
    arm_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_root = arm_directory / "checkpoints"
    run_id = _arm_run_id(stage, arm)

    schedule_document = {
        **arm.schedule.canonical_record(),
        "schedule_sha256": arm.schedule.schedule_sha256,
    }
    state_document, source_arrays = _source_state_artifacts(arm)
    initial_arrays = _initial_artifact_arrays(arm, population, stage.tracer_count)
    _publish_json_exact(arm_directory / "FORCE_SCHEDULE.json", schedule_document)
    _publish_json_exact(arm_directory / "SOURCE_STATE.json", state_document)
    _publish_npz_exact(arm_directory / "SOURCE_STATE.npz", source_arrays)
    _publish_npz_exact(arm_directory / "INITIAL_STATE.npz", initial_arrays)
    source_audit, source_audit_endpoint = _source_path_audit(
        cp=cp,
        backend=backend,
        stage=stage,
        checkpoint_years=checkpoint_years,
        arm=arm,
        arm_directory=arm_directory,
        protocol_sha256=protocol_sha256,
        runtime_identity=runtime_identity,
        runtime_receipt_sha256=runtime_receipt_sha256,
    )

    result = load_latest_cuda_massless_tracer_checkpoint(
        checkpoint_root,
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        backend=backend,
    )
    latest = _latest_checkpoint_digest(checkpoint_root, run_id, protocol_sha256)
    if result is None:
        if latest is not None:
            raise RuntimeError("checkpoint records exist but the loader returned no state")
        state = _build_device_state(
            cp,
            arm,
            schedule_set,
            population,
            str(receipt["content_sha256"]),
            stage.tracer_count,
        )
        previous_digest = None
    else:
        state = result.state
        if latest is None or latest[0] != state.global_step:
            raise RuntimeError("latest checkpoint identity disagrees with loaded endpoint")
        previous_digest = latest[1]

    checkpoint_steps = _years_to_steps(checkpoint_years, "checkpoint_years")
    completed_this_invocation = 0
    while state.global_step < stage.total_steps:
        pre_segment_storage = _route_storage_guard(
            arm_directory,
            phase=f"segment starting at global step {state.global_step}",
            cumulative_route_records=len(state.prior_encounter_routes),
        )
        segment_steps = min(checkpoint_steps, stage.total_steps - state.global_step)
        spec = FixedStepCUDAMasslessTracerSpec(
            step_years=STEP_YEARS,
            step_count=segment_steps,
            backend=backend,
            maximum_encounter_routes_per_tracer=MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER,
            maximum_total_encounter_routes=MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM,
            encounter_route_method=(
                "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2"
            ),
            encounter_route_rtol=1.0e-12,
            encounter_route_atol=1.0e-14,
            encounter_route_maximum_substep_fraction=0.125,
            maximum_primary_dynamical_phase=MAXIMUM_PRIMARY_DYNAMICAL_PHASE,
            event_time_tolerance_fraction=EVENT_TIME_TOLERANCE_FRACTION,
            source_path_steps_per_chunk=65_536,
            threads_per_block=128,
        )
        cp.cuda.runtime.deviceSynchronize()
        started = time.perf_counter()
        result = advance_cuda_massless_tracers(state, spec)
        cp.cuda.runtime.deviceSynchronize()
        elapsed = time.perf_counter() - started
        pre_checkpoint_storage = _route_storage_guard(
            arm_directory,
            phase=f"checkpoint at global step {result.state.global_step}",
            cumulative_route_records=len(result.encounter_routes),
        )
        checkpoint = dump_cuda_massless_tracer_checkpoint(
            checkpoint_root,
            run_id=run_id,
            protocol_sha256=protocol_sha256,
            checkpoint_index=result.state.global_step,
            result=result,
            previous_checkpoint_content_sha256=previous_digest,
        )
        previous_digest = str(checkpoint["checkpoint_content_sha256"])
        arrays = _endpoint_arrays(cp, result)
        observed_source_digest = _source_state_digest(
            arrays["source_positions"], arrays["source_velocities"]
        )
        expected_source_digest = source_audit[
            "checkpoint_source_state_sha256"
        ].get(str(result.state.global_step))
        if observed_source_digest != expected_source_digest:
            raise RuntimeError(
                "tracer-run source endpoint differs from the independent "
                "source-path audit at checkpoint "
                f"{result.state.global_step}"
            )
        segment_path = arm_directory / "segments" / f"segment-{result.state.global_step:020d}.json"
        _publish_json_exact(
            segment_path,
            _segment_report(
                result,
                arrays,
                elapsed,
                checkpoint,
                {
                    "pre_segment": pre_segment_storage,
                    "pre_checkpoint": pre_checkpoint_storage,
                },
            ),
        )
        state = result.state
        completed_this_invocation += 1
        if (
            stop_after_checkpoints is not None
            and completed_this_invocation >= stop_after_checkpoints
            and state.global_step < stage.total_steps
        ):
            return None

    if result is None:
        raise RuntimeError("completed checkpoint chain did not yield a result")
    endpoint = _endpoint_arrays(cp, result)
    final_storage_guards = [
        _route_storage_guard(
            arm_directory,
            phase="final endpoint artifact",
            cumulative_route_records=len(result.encounter_routes),
        )
    ]
    _publish_npz_exact(arm_directory / "ENDPOINT_STATE.npz", endpoint)
    route_rows = [asdict(row) for row in result.encounter_routes]
    removal_rows = [asdict(row) for row in result.removal_events]
    final_storage_guards.append(
        _route_storage_guard(
            arm_directory,
            phase="route-source membership audit",
            cumulative_route_records=len(route_rows),
        )
    )
    route_source_membership = _route_source_membership_audit(
        cp=cp,
        backend=backend,
        stage=stage,
        arm=arm,
        arm_directory=arm_directory,
        protocol_sha256=protocol_sha256,
        routes=result.encounter_routes,
    )
    final_storage_guards.append(
        _route_storage_guard(
            arm_directory,
            phase="routed-tracer replay audit",
            cumulative_route_records=len(route_rows),
        )
    )
    routed_tracer_replay = _routed_tracer_replay_audit(
        cp=cp,
        backend=backend,
        stage=stage,
        arm=arm,
        arm_directory=arm_directory,
        schedule_set=schedule_set,
        population=population,
        receipt_sha256=str(receipt["content_sha256"]),
        protocol_sha256=protocol_sha256,
        main_result=result,
        main_endpoint=endpoint,
    )
    final_storage_guards.append(
        _route_storage_guard(
            arm_directory,
            phase="final route and removal ledgers",
            cumulative_route_records=len(route_rows),
        )
    )
    _publish_json_exact(
        arm_directory / "ENCOUNTER_ROUTES.json",
        {
            "schema": "jxplanetx.cuda-four-arm-route-ledger.v1",
            "arm_kind": arm.schedule.arm_kind,
            "count": len(route_rows),
            "records": route_rows,
        },
    )
    _publish_json_exact(
        arm_directory / "REMOVAL_EVENTS.json",
        {
            "schema": "jxplanetx.cuda-four-arm-removal-ledger.v1",
            "arm_kind": arm.schedule.arm_kind,
            "count": len(removal_rows),
            "records": removal_rows,
            "semantics": "OPERATIONAL_REMOVALS_NOT_PHYSICAL_EVENT_CLASSIFICATION",
        },
    )
    checkpoint_steps = _years_to_steps(checkpoint_years, "checkpoint_years")
    execution_evidence = _collect_execution_evidence(
        stage_directory=stage_directory,
        arm_directory=arm_directory,
        run_id=run_id,
        protocol_sha256=protocol_sha256,
        total_steps=stage.total_steps,
        checkpoint_steps=checkpoint_steps,
    )
    _verify_loaded_execution_endpoint(
        backend=backend,
        stage=stage,
        arm_directory=arm_directory,
        report={
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "runtime_identity": dict(result.runtime_identity),
            "endpoint_global_step": result.state.global_step,
            "endpoint_epoch_years": result.state.epoch,
        },
        evidence=execution_evidence,
        endpoint=endpoint,
        route_records=route_rows,
        removal_records=removal_rows,
    )
    route_count = int(np.sum(endpoint["encounter_route_counts"]))
    terminal_count = int(np.count_nonzero(endpoint["tracer_lifecycle"] != LIFECYCLE_ACTIVE))
    source_endpoint_matches_audit = np.array_equal(
        endpoint["source_positions"], source_audit_endpoint["source_positions"]
    ) and np.array_equal(
        endpoint["source_velocities"], source_audit_endpoint["source_velocities"]
    )
    runtime = dict(result.runtime_identity)
    _require_runtime_identity(
        runtime,
        runtime_identity,
        label=f"fresh {arm.schedule.arm_kind} arm result",
    )
    gates = {
        "reached_exact_endpoint": result.state.global_step == stage.total_steps,
        "all_endpoint_arrays_finite": _finite_endpoint(endpoint),
        "route_ledger_matches_route_counts": len(route_rows) == route_count,
        "removal_ledger_matches_terminal_lifecycle": len(removal_rows) == terminal_count,
        "per_tracer_route_capacity_not_exhausted": int(
            np.max(endpoint["encounter_route_counts"])
        ) < MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER,
        "total_route_ledger_capacity_not_exhausted": (
            len(route_rows) < MAXIMUM_TOTAL_ROUTE_RECORDS_PER_ARM
        ),
        "source_endpoint_matches_all_horizon_source_path_audit": (
            source_endpoint_matches_audit
        ),
        "every_route_source_segment_matches_exact_source_path": (
            route_source_membership.get(
                "all_route_source_segments_exactly_matched"
            )
            is True
            and route_source_membership.get("route_count") == len(route_rows)
            and route_source_membership.get(
                "regenerated_source_endpoint_sha256"
            )
            == source_audit.get("endpoint_semantic_array_sha256")
        ),
        "every_routed_tracer_exact_full_stage_replay_passed": (
            routed_tracer_replay.get("exact_replay_passed") is True
            and routed_tracer_replay.get("routed_tracer_count")
            == len({row.tracer_index for row in result.encounter_routes})
        ),
        "runtime_method_kernel_and_implementation_bound": (
            runtime.get("method_id") == METHOD_ID
            and runtime.get("kernel_source_sha256") == KERNEL_SOURCE_SHA256
            and runtime.get("implementation_source_sha256")
            == IMPLEMENTATION_SOURCE_SHA256
            and runtime.get("event_locator_source_sha256")
            == EVENT_LOCATOR_SOURCE_SHA256
        ),
        "committed_checkpoint_segment_roster_hashes_and_chain_verified": True,
        "uncommitted_checkpoint_array_orphans_preserved_separately": True,
    }
    artifact_paths = (
        arm_directory / "FORCE_SCHEDULE.json",
        arm_directory / "SOURCE_STATE.json",
        arm_directory / "SOURCE_STATE.npz",
        arm_directory / "INITIAL_STATE.npz",
        arm_directory / "ENDPOINT_STATE.npz",
        arm_directory / "ENCOUNTER_ROUTES.json",
        arm_directory / "REMOVAL_EVENTS.json",
        arm_directory / "SOURCE_PATH_AUDIT.json",
        arm_directory / "SOURCE_PATH_AUDIT_ENDPOINT.npz",
        arm_directory / "ROUTE_SOURCE_MEMBERSHIP_AUDIT.json",
        arm_directory / "ROUTED_TRACER_REPLAY_AUDIT.json",
    )
    report: dict[str, Any] = {
        "schema": ARM_RESULT_SCHEMA,
        "completed_utc": _utc_now(),
        "status": "COMPLETED_ARM_SCREEN_NOT_INFERENCE" if all(gates.values()) else "FAILED_ARM_GATE",
        "stage_id": stage.stage_id,
        "arm_kind": arm.schedule.arm_kind,
        "run_id": run_id,
        "protocol_sha256": protocol_sha256,
        "schedule_sha256": arm.schedule.schedule_sha256,
        "source_state_sha256": arm.initial_state.state_sha256,
        "matched_schedule_set_sha256": schedule_set.matched_set_sha256,
        "population_transform_receipt_sha256": receipt["content_sha256"],
        "runtime_identity_receipt_sha256": runtime_receipt_sha256,
        "endpoint_global_step": result.state.global_step,
        "endpoint_epoch_years": result.state.epoch,
        "endpoint_semantic_array_sha256": _array_set_sha256(endpoint),
        "measurements": _endpoint_measurements(endpoint),
        "route_storage_guards": final_storage_guards,
        "route_reason_counts": dict(sorted(Counter(row.reason for row in result.encounter_routes).items())),
        "removal_event_counts": dict(sorted(Counter(row.event_type for row in result.removal_events).items())),
        "source_path_audit": {
            "content_sha256": source_audit["content_sha256"],
            "primary_relative_source_minima": source_audit[
                "primary_relative_source_minima"
            ],
            "maximum_absolute_relative_energy_drift": source_audit[
                "maximum_absolute_relative_energy_drift"
            ],
            "maximum_relative_angular_momentum_vector_drift": source_audit[
                "maximum_relative_angular_momentum_vector_drift"
            ],
            "maximum_linear_momentum_delta_norm": source_audit[
                "maximum_linear_momentum_delta_norm"
            ],
        },
        "route_source_membership_audit": {
            "content_sha256": route_source_membership["content_sha256"],
            "route_count": route_source_membership["route_count"],
            "unique_routed_source_steps": route_source_membership[
                "unique_routed_source_steps"
            ],
            "all_route_source_segments_exactly_matched": (
                route_source_membership[
                    "all_route_source_segments_exactly_matched"
                ]
            ),
        },
        "routed_tracer_replay_audit": {
            "content_sha256": routed_tracer_replay["content_sha256"],
            "routed_tracer_count": routed_tracer_replay[
                "routed_tracer_count"
            ],
            "exact_replay_passed": routed_tracer_replay[
                "exact_replay_passed"
            ],
        },
        "execution_evidence": execution_evidence,
        "runtime_identity": runtime,
        "gates": {**gates, "all_pass": all(gates.values())},
        "artifacts": {
            path.name: _artifact(path, stage_directory) for path in artifact_paths
        },
        "classification": "MODEL_OUTPUT / SCREENING_ONLY",
        "interpretation_limits": list(INTERPRETATION_LIMITS),
        "claims": dict(RESULT_CLAIMS),
    }
    report["content_sha256"] = _content_hash(REPORT_DOMAIN, report)
    _publish_json_exact(result_path, report)
    return report


def _comparison(
    control_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    with np.load(control_path, allow_pickle=False) as left_archive:
        left = {name: np.ascontiguousarray(left_archive[name]) for name in left_archive.files}
    with np.load(candidate_path, allow_pickle=False) as right_archive:
        right = {name: np.ascontiguousarray(right_archive[name]) for name in right_archive.files}
    if not np.array_equal(left["tracer_indices"], right["tracer_indices"]):
        raise RuntimeError("candidate/control endpoint tracer identities differ")
    left_orbit = _orbital_arrays(left)
    right_orbit = _orbital_arrays(right)
    common_active = (
        (left["tracer_lifecycle"] == LIFECYCLE_ACTIVE)
        & (right["tracer_lifecycle"] == LIFECYCLE_ACTIVE)
    )
    common_bound = common_active & left_orbit["bound"] & right_orbit["bound"]
    delta_q = right_orbit["perihelion_AU"] - left_orbit["perihelion_AU"]
    delta_e = right_orbit["eccentricity"] - left_orbit["eccentricity"]
    delta_i = right_orbit["inclination_rad"] - left_orbit["inclination_rad"]
    delta_a = right_orbit["semimajor_axis_AU"] - left_orbit["semimajor_axis_AU"]
    return {
        "comparison_semantics": "PAIRED_ENDPOINT_DIFFERENCE_CANDIDATE_MINUS_CONTROL",
        "orbital_element_semantics": (
            "INSTANTANEOUS_PRIMARY_RELATIVE_TWO_BODY_OSCULATING_PROXY_DIFFERENCES; "
            "NOT_FITTED_OR_FULL_SYSTEM_INVARIANTS; SEMIMAJOR_AXIS_COMMON_BOUND_ONLY"
        ),
        "common_active_count": int(np.count_nonzero(common_active)),
        "common_active_bound_count": int(np.count_nonzero(common_bound)),
        "delta_perihelion_AU": _quantiles(delta_q[common_active]),
        "absolute_delta_perihelion_AU": _quantiles(np.abs(delta_q[common_active])),
        "delta_semimajor_axis_AU_common_bound": _quantiles(delta_a[common_bound]),
        "absolute_delta_semimajor_axis_AU_common_bound": _quantiles(np.abs(delta_a[common_bound])),
        "delta_eccentricity": _quantiles(delta_e[common_active]),
        "absolute_delta_inclination_rad": _quantiles(np.abs(delta_i[common_active])),
        "candidate_only_terminal_count": int(
            np.count_nonzero(
                (left["tracer_lifecycle"] == LIFECYCLE_ACTIVE)
                & (right["tracer_lifecycle"] != LIFECYCLE_ACTIVE)
            )
        ),
        "control_only_terminal_count": int(
            np.count_nonzero(
                (left["tracer_lifecycle"] != LIFECYCLE_ACTIVE)
                & (right["tracer_lifecycle"] == LIFECYCLE_ACTIVE)
            )
        ),
        "interpretation": "DESCRIPTIVE_SCREEN_METRICS_ONLY_NOT_A_LIKELIHOOD_OR_MODEL_PREFERENCE",
    }


def _ceil_div(numerator: int, denominator: int) -> int:
    if type(numerator) is not int or numerator < 0:
        raise ValueError("ceil-div numerator must be a nonnegative integer")
    if type(denominator) is not int or denominator <= 0:
        raise ValueError("ceil-div denominator must be a positive integer")
    return (numerator + denominator - 1) // denominator


def _route_episode_capacity_screen(
    *,
    route_records_by_arm: Mapping[str, Iterable[Mapping[str, Any]]],
    tracer_ids_by_arm: Mapping[str, Iterable[int]],
    terminal_tracer_ids_by_arm: Mapping[str, Iterable[int]],
    observed_steps: int,
    target_steps: int,
    expected_arm_kinds: tuple[str, ...],
    required_pilot_steps: int = ROUTE_CAPACITY_PILOT_STEPS,
    required_tracer_count: int | None = None,
) -> dict[str, Any]:
    """Audit routed-step episodes and compute the long-run admission budget.

    An episode is a maximal run of consecutive routed macrosteps for one
    tracer.  Trigger-source or trigger-reason changes do not split it.  Thus
    ``route_record_count`` is both the episode length and the exact amount
    consumed from the engine's hard cumulative per-tracer route counter.

    For an active lane with ``e`` completed episodes and ``r`` observed route
    records after ``s`` pilot steps, the remaining target exposure reserves::

        future_slots = ceil((e + 1) * (S - s) / s)
        planned_routes = r + 512 * future_slots

    The add-one term protects lanes with no observed episode.  Admission
    requires every active observed episode to fit the 512-record reservation,
    no active REJOINED episode to end at the pilot frontier, and every lane's
    planned total to remain at or below 12,288.  The unchanged engine ceiling
    is 16,384, so 4,096 records remain as runtime headroom.  This is an
    engineering capacity screen, not a stochastic confidence interval, a
    dynamical recurrence model, or a guarantee that the target run completes.
    """

    if type(observed_steps) is not int or observed_steps <= 0:
        raise ValueError("observed_steps must be a positive integer")
    if type(target_steps) is not int or target_steps < observed_steps:
        raise ValueError("target_steps must be an integer no smaller than observed_steps")
    if type(required_pilot_steps) is not int or required_pilot_steps <= 0:
        raise ValueError("required_pilot_steps must be a positive integer")
    if (
        required_tracer_count is not None
        and (type(required_tracer_count) is not int or required_tracer_count <= 0)
    ):
        raise ValueError("required_tracer_count must be a positive integer or None")
    if (
        type(expected_arm_kinds) is not tuple
        or not expected_arm_kinds
        or any(type(value) is not str or not value for value in expected_arm_kinds)
        or len(set(expected_arm_kinds)) != len(expected_arm_kinds)
    ):
        raise ValueError("expected_arm_kinds must be a unique nonempty string tuple")
    expected_arm_set = set(expected_arm_kinds)
    for label, value in (
        ("route_records_by_arm", route_records_by_arm),
        ("tracer_ids_by_arm", tracer_ids_by_arm),
        ("terminal_tracer_ids_by_arm", terminal_tracer_ids_by_arm),
    ):
        if not isinstance(value, Mapping) or set(value) != expected_arm_set:
            raise ValueError(f"{label} must have the exact expected arm roster")

    terminal_outcomes = {
        "REMOVED_SOURCE_RADIUS",
        "REMOVED_OUTER_BOUNDARY",
    }
    allowed_outcomes = terminal_outcomes | {"REJOINED"}
    remaining_steps = target_steps - observed_steps
    all_episode_descriptors: list[dict[str, Any]] = []
    arm_diagnostics: dict[str, Any] = {}
    total_route_records = 0
    total_episodes = 0
    total_completed_episodes = 0
    right_censored_rejoined_episodes = 0
    longest_episode_records = 0
    maximum_observed_routes_for_one_tracer = 0
    maximum_planned_routes_for_one_tracer = 0
    minimum_planned_headroom = MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
    active_episode_reservation_bound_passed = True
    planned_budget_passed = True
    worst_lane: dict[str, Any] | None = None
    required_tracer_count_passed = required_tracer_count is not None
    reference_tracer_ids: tuple[int, ...] | None = None

    for arm_kind in expected_arm_kinds:
        raw_tracer_ids = tuple(tracer_ids_by_arm[arm_kind])
        if any(type(value) is not int or value < 0 for value in raw_tracer_ids):
            raise ValueError(f"{arm_kind} tracer ids must be nonnegative integers")
        if len(set(raw_tracer_ids)) != len(raw_tracer_ids):
            raise ValueError(f"{arm_kind} tracer ids must be unique")
        tracer_ids = tuple(sorted(raw_tracer_ids))
        if reference_tracer_ids is None:
            reference_tracer_ids = tracer_ids
        elif tracer_ids != reference_tracer_ids:
            raise ValueError("tracer rosters must be identical across arms")
        tracer_id_set = set(tracer_ids)
        raw_terminal_ids = tuple(terminal_tracer_ids_by_arm[arm_kind])
        if any(type(value) is not int or value < 0 for value in raw_terminal_ids):
            raise ValueError(
                f"{arm_kind} terminal tracer ids must be nonnegative integers"
            )
        terminal_ids = set(raw_terminal_ids)
        if len(terminal_ids) != len(raw_terminal_ids):
            raise ValueError(f"{arm_kind} terminal tracer ids must be unique")
        if not terminal_ids <= tracer_id_set:
            raise ValueError(f"{arm_kind} terminal tracer ids are outside the roster")
        if required_tracer_count is not None:
            required_tracer_count_passed = (
                required_tracer_count_passed
                and len(tracer_ids) == required_tracer_count
            )

        normalized: list[dict[str, Any]] = []
        seen_lane_steps: set[tuple[int, int]] = set()
        for record_index, record in enumerate(route_records_by_arm[arm_kind]):
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"{arm_kind} route record {record_index} is not a mapping"
                )
            tracer_index = record.get("tracer_index")
            global_step = record.get("global_step")
            source_id = record.get("source_id")
            reason = record.get("reason")
            outcome = record.get("outcome")
            if type(tracer_index) is not int or tracer_index not in tracer_id_set:
                raise ValueError(
                    f"{arm_kind} route record {record_index} has an invalid tracer"
                )
            if (
                type(global_step) is not int
                or not 0 <= global_step < observed_steps
            ):
                raise ValueError(
                    f"{arm_kind} route record {record_index} has an invalid step"
                )
            if type(source_id) is not str or not source_id:
                raise ValueError(
                    f"{arm_kind} route record {record_index} has an invalid source"
                )
            if type(reason) is not str or not reason:
                raise ValueError(
                    f"{arm_kind} route record {record_index} has an invalid reason"
                )
            if type(outcome) is not str or outcome not in allowed_outcomes:
                raise ValueError(
                    f"{arm_kind} route record {record_index} has an invalid outcome"
                )
            lane_step = (tracer_index, global_step)
            if lane_step in seen_lane_steps:
                raise ValueError(
                    f"{arm_kind} has duplicate routes for tracer-step {lane_step}"
                )
            seen_lane_steps.add(lane_step)
            normalized.append(
                {
                    "tracer_index": tracer_index,
                    "global_step": global_step,
                    "source_id": source_id,
                    "reason": reason,
                    "outcome": outcome,
                }
            )
        normalized.sort(
            key=lambda row: (
                row["tracer_index"],
                row["global_step"],
                row["source_id"],
                row["reason"],
                row["outcome"],
            )
        )
        records_by_tracer: dict[int, list[dict[str, Any]]] = {
            tracer_index: [] for tracer_index in tracer_ids
        }
        for record in normalized:
            records_by_tracer[record["tracer_index"]].append(record)

        arm_route_records = len(normalized)
        arm_episode_count = 0
        arm_completed_episode_count = 0
        arm_right_censored_count = 0
        arm_longest_episode = 0
        arm_maximum_observed_routes = 0
        arm_maximum_planned_routes = 0
        arm_minimum_planned_headroom = MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
        arm_episode_source_counts: Counter[str] = Counter()
        arm_future_slot_counts: Counter[int] = Counter()

        for tracer_index in tracer_ids:
            lane_records = records_by_tracer[tracer_index]
            is_terminal = tracer_index in terminal_ids
            terminal_positions = tuple(
                index
                for index, record in enumerate(lane_records)
                if record["outcome"] in terminal_outcomes
            )
            if is_terminal:
                if not lane_records or terminal_positions != (len(lane_records) - 1,):
                    raise ValueError(
                        f"{arm_kind} terminal tracer {tracer_index} lacks one final "
                        "terminal route outcome"
                    )
            elif terminal_positions:
                raise ValueError(
                    f"{arm_kind} active tracer {tracer_index} has a terminal route outcome"
                )

            lane_episodes: list[list[dict[str, Any]]] = []
            current_episode: list[dict[str, Any]] = []
            for record in lane_records:
                if (
                    current_episode
                    and record["global_step"]
                    != current_episode[-1]["global_step"] + 1
                ):
                    lane_episodes.append(current_episode)
                    current_episode = []
                current_episode.append(record)
            if current_episode:
                lane_episodes.append(current_episode)

            completed_episode_count = 0
            lane_right_censored_count = 0
            lane_longest_episode = 0
            for episode_index, episode in enumerate(lane_episodes):
                start_step = episode[0]["global_step"]
                end_step = episode[-1]["global_step"]
                route_record_count = len(episode)
                right_censored = (
                    not is_terminal
                    and end_step == observed_steps - 1
                    and episode[-1]["outcome"] == "REJOINED"
                )
                if not right_censored:
                    completed_episode_count += 1
                else:
                    lane_right_censored_count += 1
                lane_longest_episode = max(
                    lane_longest_episode, route_record_count
                )
                sources = tuple(sorted({row["source_id"] for row in episode}))
                reasons = tuple(sorted({row["reason"] for row in episode}))
                for source_id in sources:
                    arm_episode_source_counts[source_id] += 1
                all_episode_descriptors.append(
                    {
                        "arm_kind": arm_kind,
                        "tracer_index": tracer_index,
                        "episode_index_for_tracer": episode_index,
                        "start_global_step": start_step,
                        "end_global_step": end_step,
                        "route_record_count": route_record_count,
                        "source_ids": list(sources),
                        "reasons": list(reasons),
                        "terminal_tracer": is_terminal,
                        "right_censored_rejoined": right_censored,
                        "last_outcome": episode[-1]["outcome"],
                    }
                )

            route_count = len(lane_records)
            future_episode_slots = (
                0
                if is_terminal
                else _ceil_div(
                    (completed_episode_count + 1) * remaining_steps,
                    observed_steps,
                )
            )
            planned_routes = (
                route_count
                + ROUTE_EPISODE_RESERVED_RECORDS * future_episode_slots
            )
            planned_headroom = (
                MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER - planned_routes
            )
            lane_reservation_passed = (
                is_terminal
                or lane_longest_episode <= ROUTE_EPISODE_RESERVED_RECORDS
            )
            lane_budget_passed = planned_routes <= ROUTE_CAPACITY_ADMISSION_LIMIT
            active_episode_reservation_bound_passed = (
                active_episode_reservation_bound_passed
                and lane_reservation_passed
            )
            planned_budget_passed = planned_budget_passed and lane_budget_passed
            arm_future_slot_counts[future_episode_slots] += 1
            arm_episode_count += len(lane_episodes)
            arm_completed_episode_count += completed_episode_count
            arm_right_censored_count += lane_right_censored_count
            arm_longest_episode = max(arm_longest_episode, lane_longest_episode)
            arm_maximum_observed_routes = max(
                arm_maximum_observed_routes, route_count
            )
            arm_maximum_planned_routes = max(
                arm_maximum_planned_routes, planned_routes
            )
            arm_minimum_planned_headroom = min(
                arm_minimum_planned_headroom, planned_headroom
            )
            lane_summary = {
                "arm_kind": arm_kind,
                "tracer_index": tracer_index,
                "terminal_tracer": is_terminal,
                "observed_route_records": route_count,
                "completed_episode_count": completed_episode_count,
                "right_censored_rejoined_episode_count": (
                    lane_right_censored_count
                ),
                "longest_episode_route_records": lane_longest_episode,
                "reserved_future_episode_slots": future_episode_slots,
                "planned_route_records": planned_routes,
                "planned_headroom_to_hard_cap": planned_headroom,
            }
            if (
                worst_lane is None
                or planned_routes > worst_lane["planned_route_records"]
            ):
                worst_lane = lane_summary

        total_route_records += arm_route_records
        total_episodes += arm_episode_count
        total_completed_episodes += arm_completed_episode_count
        right_censored_rejoined_episodes += arm_right_censored_count
        longest_episode_records = max(
            longest_episode_records, arm_longest_episode
        )
        maximum_observed_routes_for_one_tracer = max(
            maximum_observed_routes_for_one_tracer,
            arm_maximum_observed_routes,
        )
        maximum_planned_routes_for_one_tracer = max(
            maximum_planned_routes_for_one_tracer,
            arm_maximum_planned_routes,
        )
        minimum_planned_headroom = min(
            minimum_planned_headroom,
            arm_minimum_planned_headroom,
        )
        arm_diagnostics[arm_kind] = {
            "tracer_count": len(tracer_ids),
            "active_tracer_count": len(tracer_ids) - len(terminal_ids),
            "terminal_tracer_count": len(terminal_ids),
            "route_record_count": arm_route_records,
            "episode_count": arm_episode_count,
            "completed_episode_count": arm_completed_episode_count,
            "right_censored_rejoined_episode_count": arm_right_censored_count,
            "longest_episode_route_records": arm_longest_episode,
            "maximum_observed_routes_for_one_tracer": (
                arm_maximum_observed_routes
            ),
            "maximum_planned_routes_for_one_tracer": (
                arm_maximum_planned_routes
            ),
            "minimum_planned_headroom_to_hard_cap": (
                arm_minimum_planned_headroom
            ),
            "episode_source_membership_counts": dict(
                sorted(arm_episode_source_counts.items())
            ),
            "future_episode_slot_distribution": {
                str(slot_count): lane_count
                for slot_count, lane_count in sorted(
                    arm_future_slot_counts.items()
                )
            },
        }

    required_pilot_step_count_passed = observed_steps == required_pilot_steps
    right_censoring_gate_passed = right_censored_rejoined_episodes == 0
    capacity_screen_passed = (
        required_pilot_step_count_passed
        and required_tracer_count_passed
        and right_censoring_gate_passed
        and active_episode_reservation_bound_passed
        and planned_budget_passed
    )
    episode_descriptor_sha256 = hashlib.sha256(
        b"jxplanetx.cuda-route-episode-descriptors.v1\x00"
        + _canonical(all_episode_descriptors)
    ).hexdigest()
    return {
        "schema": "jxplanetx.cuda-route-episode-capacity-screen.v1",
        "semantics": (
            "ADD_ONE_EXPOSURE_SCALED_EPISODE_RESOURCE_RESERVATION; "
            "FAIL_CLOSED_ADMISSION_SCREEN_NOT_A_COMPLETION_OR_DYNAMICAL_PREDICTION"
        ),
        "episode_definition": (
            "MAXIMAL_CONSECUTIVE_ROUTED_MACROSTEPS_PER_TRACER; "
            "SOURCE_AND_REASON_CHANGES_DO_NOT_SPLIT; ONE_QUIET_STEP_SPLITS"
        ),
        "observed_steps": observed_steps,
        "target_steps": target_steps,
        "remaining_target_steps": remaining_steps,
        "required_pilot_steps": required_pilot_steps,
        "required_tracer_count_per_arm": required_tracer_count,
        "future_episode_slot_formula": (
            "ACTIVE: CEIL((COMPLETED_EPISODES+1)*(TARGET_STEPS-OBSERVED_STEPS)"
            "/OBSERVED_STEPS); TERMINAL: 0"
        ),
        "reserved_route_records_per_future_episode": (
            ROUTE_EPISODE_RESERVED_RECORDS
        ),
        "engine_hard_capacity_per_tracer": (
            MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
        ),
        "required_runtime_headroom_per_tracer": (
            ROUTE_CAPACITY_REQUIRED_HEADROOM
        ),
        "admission_limit_per_tracer": ROUTE_CAPACITY_ADMISSION_LIMIT,
        "total_route_record_count": total_route_records,
        "total_episode_count": total_episodes,
        "total_completed_episode_count": total_completed_episodes,
        "right_censored_rejoined_episode_count": (
            right_censored_rejoined_episodes
        ),
        "longest_episode_route_records": longest_episode_records,
        "maximum_observed_routes_for_one_tracer": (
            maximum_observed_routes_for_one_tracer
        ),
        "maximum_planned_routes_for_one_tracer": (
            maximum_planned_routes_for_one_tracer
        ),
        "minimum_planned_headroom_to_hard_cap": minimum_planned_headroom,
        "worst_lane": worst_lane,
        "episode_descriptor_sha256": episode_descriptor_sha256,
        "episode_descriptor_count": len(all_episode_descriptors),
        "arm_diagnostics": arm_diagnostics,
        "gates": {
            "required_capacity_pilot_step_count": (
                required_pilot_step_count_passed
            ),
            "required_tracer_count_in_every_arm": (
                required_tracer_count_passed
            ),
            "no_right_censored_rejoined_episode": right_censoring_gate_passed,
            "active_observed_episodes_fit_future_reservation": (
                active_episode_reservation_bound_passed
            ),
            "every_lane_planned_routes_at_or_below_admission_limit": (
                planned_budget_passed
            ),
            "one_myr_capacity_screen_passed": capacity_screen_passed,
        },
        "one_myr_capacity_screen_passed": capacity_screen_passed,
        "claim_limit": (
            "This finite pilot-based resource budget does not estimate physical "
            "encounter probabilities, bound the number or length of future "
            "encounters, or guarantee one-Myr completion. The unchanged engine "
            "hard cap remains authoritative and fails closed at runtime."
        ),
    }


def _artifact_initial_state_hashes(stage_directory: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for arm_kind in ARM_KINDS:
        path = (
            stage_directory
            / "arms"
            / ARM_DIRECTORY_NAMES[arm_kind]
            / "INITIAL_STATE.npz"
        )
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
        hashes[arm_kind] = _array_set_sha256(arrays)
    return hashes


def _capacity_admission_document(
    *,
    stage: Stage,
    checkpoint_years: float,
    stage_directory: Path,
    schedule_set: MatchedMechanismScheduleSet,
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    runtime_receipt_sha256: str,
    reports: Mapping[str, Mapping[str, Any]],
    route_episode_capacity_screen: Mapping[str, Any],
    target_context: Mapping[str, Any],
) -> dict[str, Any]:
    if stage.stage_id != "capacity_pilot":
        raise ValueError("capacity admission requires the capacity_pilot stage")
    initial_artifact_hashes = _artifact_initial_state_hashes(stage_directory)
    target_gates = target_context.get("gates")
    target_context_passed = (
        type(target_gates) is dict and target_gates.get("all_pass") is True
    )
    schedule_physics = _schedule_prefix_physics_binding(schedule_set)
    exact_passage = (
        protocol.get("schedule_set", {}).get(
            "nominal_passing_source_event_years"
        )
        == FULL_SCREEN_PASSAGE_YEARS
    )
    gates = {
        "exact_capacity_pilot_stage": (
            stage.total_steps == ROUTE_CAPACITY_PILOT_STEPS
            and stage.tracer_count == STAGES["one_myr"].tracer_count == 16_800
        ),
        "exact_25k_year_checkpoint_cadence": (
            checkpoint_years == STAGES["capacity_pilot"].checkpoint_years
            and _years_to_steps(checkpoint_years, "checkpoint_years") == 200_000
        ),
        "exact_50k_year_nominal_passage": exact_passage,
        "all_four_arm_results_passed": (
            set(reports) == set(ARM_KINDS)
            and all(
                report.get("gates", {}).get("all_pass") is True
                for report in reports.values()
            )
        ),
        "episode_capacity_screen_passed": (
            route_episode_capacity_screen.get("one_myr_capacity_screen_passed")
            is True
            and route_episode_capacity_screen.get("observed_steps")
            == ROUTE_CAPACITY_PILOT_STEPS
            and route_episode_capacity_screen.get("target_steps")
            == STAGES["one_myr"].total_steps
            and route_episode_capacity_screen.get("required_tracer_count_per_arm")
            == STAGES["one_myr"].tracer_count
            and type(route_episode_capacity_screen.get("arm_diagnostics")) is dict
            and set(route_episode_capacity_screen["arm_diagnostics"])
            == set(ARM_KINDS)
        ),
        "exact_target_prefix_physics_population_and_roster": (
            target_context_passed
            and target_context.get("target_stage_id") == "one_myr"
            and target_context.get("target_total_steps")
            == STAGES["one_myr"].total_steps
            and target_context.get("tracer_count_per_arm")
            == STAGES["one_myr"].tracer_count
            and target_context.get("pilot_schedule_prefix_physics")
            == schedule_physics
            and protocol.get("capacity_target_binding") == dict(target_context)
        ),
        "persisted_initial_states_match_bound_target_context": (
            initial_artifact_hashes
            == target_context.get("pilot_full_initial_state_semantic_sha256")
            == target_context.get("target_full_initial_state_semantic_sha256")
        ),
        "exact_population_receipt_and_semantic_arrays": (
            receipt.get("content_sha256")
            == target_context.get("pilot_population_transform_receipt_sha256")
            == target_context.get("target_population_transform_receipt_sha256")
            and receipt.get("output", {}).get("semantic_array_sha256")
            == target_context.get("population_semantic_array_sha256")
        ),
    }
    document: dict[str, Any] = {
        "schema": CAPACITY_ADMISSION_SCHEMA,
        "source_stage_id": stage.stage_id,
        "target_stage_id": "one_myr",
        "source_protocol_sha256": protocol.get("content_sha256"),
        "source_runtime_identity_receipt_sha256": runtime_receipt_sha256,
        "runtime_identity": dict(runtime_identity),
        "population_transform_receipt_sha256": receipt.get("content_sha256"),
        "population_semantic_array_sha256": receipt.get("output", {}).get(
            "semantic_array_sha256"
        ),
        "schedule_prefix_physics": schedule_physics,
        "full_initial_state_semantic_sha256": initial_artifact_hashes,
        "capacity_target_context": dict(target_context),
        "episode_capacity_screen": dict(route_episode_capacity_screen),
        "arm_result_sha256": {
            arm_kind: reports[arm_kind].get("content_sha256")
            for arm_kind in ARM_KINDS
        },
        "route_ledger_sha256": {
            arm_kind: _sha256_file(
                stage_directory
                / "arms"
                / ARM_DIRECTORY_NAMES[arm_kind]
                / "ENCOUNTER_ROUTES.json"
            )
            for arm_kind in ARM_KINDS
        },
        "execution_policy": {
            "backend": protocol.get("backend"),
            "integrator": protocol.get("integrator"),
            "safety": protocol.get("safety"),
            "route_ledger_budget": protocol.get("route_ledger_budget"),
        },
        "code_and_frozen_input_bindings": protocol.get("bindings"),
        "gates": {**gates, "all_pass": all(gates.values())},
        "one_myr_authorized": all(gates.values()),
        "checkpoint_reuse": False,
        "checkpoint_reuse_semantics": (
            "SEPARATE_PROTOCOL_AND_RUN_IDS; ONE_MYR_RESTARTS_FROM_IDENTICAL_INITIAL_"
            "BYTES_AND_RERUNS_THE_FIRST_100_KYR"
        ),
        "classification": "MODEL_OUTPUT / RESOURCE_ADMISSION_ONLY_NOT_SCIENCE",
        "claim_limit": (
            "A pass authorizes only an attempt at the one-Myr engineering screen. "
            "It is not a completion guarantee, an accuracy or stability result, a "
            "dynamical recurrence bound, a model preference, or a 4.5-Gyr authorization."
        ),
    }
    document["content_sha256"] = _content_hash(
        CAPACITY_ADMISSION_DOMAIN, document
    )
    return document


def _load_verified_capacity_admission(
    path: Path,
    *,
    expected_authorized: bool = True,
) -> dict[str, Any]:
    document = _load_json(path)
    _verify_content_hash(
        document,
        CAPACITY_ADMISSION_DOMAIN,
        schema=CAPACITY_ADMISSION_SCHEMA,
        label="capacity admission receipt",
    )
    gates = document.get("gates")
    gate_values = (
        [value for key, value in gates.items() if key != "all_pass"]
        if type(gates) is dict
        else []
    )
    calculated_authorized = bool(gate_values) and all(
        value is True for value in gate_values
    )
    if (
        type(gates) is not dict
        or any(type(value) is not bool for value in gates.values())
        or gates.get("all_pass") is not calculated_authorized
        or document.get("one_myr_authorized") is not calculated_authorized
        or calculated_authorized is not expected_authorized
        or document.get("source_stage_id") != "capacity_pilot"
        or document.get("target_stage_id") != "one_myr"
        or document.get("checkpoint_reuse") is not False
    ):
        raise RuntimeError("capacity admission receipt has an invalid terminal decision")
    return document


def _capacity_pilot_prerequisite(
    output_root: Path,
    population: Mapping[str, np.ndarray],
    receipt: Mapping[str, Any],
    schedule_set: MatchedMechanismScheduleSet,
    protocol: Mapping[str, Any],
    runtime_identity: Mapping[str, Any] | None,
    backend: BackendSpec | None,
) -> None:
    path = output_root / "capacity_pilot/SUMMARY.json"
    if not path.is_file():
        raise RuntimeError(
            "the exact full-population capacity_pilot must pass before one_myr"
        )
    summary = _load_verified_stage_summary(
        path,
        allowed_statuses=(STAGES["capacity_pilot"].completion_status,),
        expected_runtime_identity=runtime_identity,
        backend=backend,
    )
    admission_path = output_root / "capacity_pilot/CAPACITY_ADMISSION.json"
    admission = _load_verified_capacity_admission(admission_path)
    if summary.get("capacity_admission_sha256") != admission.get("content_sha256"):
        raise RuntimeError("capacity pilot summary binds a different admission receipt")
    if admission.get("source_protocol_sha256") != summary.get("protocol_sha256"):
        raise RuntimeError("capacity admission binds a different pilot protocol")

    current_physics = _schedule_prefix_physics_binding(schedule_set)
    current_initial_hashes = _initial_state_semantic_hashes(
        schedule_set, population, STAGES["one_myr"].tracer_count
    )
    context = admission.get("capacity_target_context")
    if type(context) is not dict:
        raise RuntimeError("capacity admission target context is invalid")
    if (
        context.get("target_stage_id") != "one_myr"
        or context.get("target_total_steps") != STAGES["one_myr"].total_steps
        or context.get("tracer_count_per_arm") != STAGES["one_myr"].tracer_count
        or context.get("target_schedule_prefix_physics") != current_physics
        or context.get("target_source_initial_state_sha256")
        != {
            arm.schedule.arm_kind: arm.initial_state.state_sha256
            for arm in schedule_set.arms
        }
        or context.get("target_full_initial_state_semantic_sha256")
        != current_initial_hashes
    ):
        raise RuntimeError(
            "current one_myr schedule physics or exact initial state differs "
            "from the capacity pilot target"
        )
    if (
        admission.get("population_transform_receipt_sha256")
        != receipt.get("content_sha256")
        or admission.get("population_semantic_array_sha256")
        != receipt.get("output", {}).get("semantic_array_sha256")
        or context.get("target_population_transform_receipt_sha256")
        != receipt.get("content_sha256")
    ):
        raise RuntimeError("capacity pilot used a different transformed population")
    execution_policy = admission.get("execution_policy")
    if execution_policy != {
        "backend": protocol.get("backend"),
        "integrator": protocol.get("integrator"),
        "safety": protocol.get("safety"),
        "route_ledger_budget": protocol.get("route_ledger_budget"),
    }:
        raise RuntimeError("capacity pilot execution policy differs from one_myr")

    pilot_bindings = admission.get("code_and_frozen_input_bindings")
    if type(pilot_bindings) is not dict:
        raise RuntimeError("capacity admission code/input bindings are invalid")
    current_bindings = protocol["bindings"]
    for key in (
        "runner_sha256",
        "implementation_source_sha256",
        "kernel_source_sha256",
        "event_locator_source_sha256",
        "mechanism_schedule_source_sha256",
        "acceptance_state_sha256",
        "acceptance_protocol_sha256",
        "de441_control_sha256",
    ):
        if pilot_bindings.get(key) != current_bindings.get(key):
            raise RuntimeError(
                f"capacity pilot binding {key} differs; rerun capacity_pilot"
            )
    if runtime_identity is not None:
        _require_runtime_identity(
            admission.get("runtime_identity"),
            runtime_identity,
            label="capacity pilot admission/current runtime",
        )


def _summarize(
    *,
    stage: Stage,
    checkpoint_years: float,
    stage_directory: Path,
    output_root: Path,
    schedule_set: MatchedMechanismScheduleSet,
    receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    runtime_receipt_sha256: str,
    reports: Mapping[str, Mapping[str, Any]],
    capacity_target_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    for arm_kind, report in reports.items():
        if report.get("runtime_identity_receipt_sha256") != runtime_receipt_sha256:
            raise RuntimeError(f"{arm_kind} arm result binds a different runtime receipt")
        _require_runtime_identity(
            report.get("runtime_identity"),
            runtime_identity,
            label=f"{arm_kind} arm result",
        )
    expected_arms = set(ARM_KINDS)
    complete = set(reports) == expected_arms
    all_arm_gates = complete and all(
        bool(report.get("gates", {}).get("all_pass")) for report in reports.values()
    )
    partial_summary_commit: dict[str, Any] | None = None
    summary_path = stage_directory / "SUMMARY.json"
    if complete and summary_path.is_file():
        partial_summary_commit = _load_partial_summary_commit(summary_path)
    comparisons: dict[str, Any] = {}
    if all_arm_gates:
        control_endpoint = (
            stage_directory / "arms/control/ENDPOINT_STATE.npz"
        )
        for arm_kind in ARM_KINDS[1:]:
            candidate_endpoint = (
                stage_directory
                / "arms"
                / ARM_DIRECTORY_NAMES[arm_kind]
                / "ENDPOINT_STATE.npz"
            )
            comparisons[arm_kind] = _comparison(control_endpoint, candidate_endpoint)
    if complete:
        route_records_by_arm: dict[str, Iterable[Mapping[str, Any]]] = {}
        tracer_ids_by_arm: dict[str, Iterable[int]] = {}
        terminal_tracer_ids_by_arm: dict[str, Iterable[int]] = {}
        for arm_kind in ARM_KINDS:
            arm_directory = (
                stage_directory / "arms" / ARM_DIRECTORY_NAMES[arm_kind]
            )
            route_document = _load_json(arm_directory / "ENCOUNTER_ROUTES.json")
            if (
                route_document.get("schema")
                != "jxplanetx.cuda-four-arm-route-ledger.v1"
                or route_document.get("arm_kind") != arm_kind
                or type(route_document.get("records")) is not list
                or route_document.get("count") != len(route_document["records"])
            ):
                raise RuntimeError(
                    f"{arm_kind} route ledger is invalid for episode screening"
                )
            route_records_by_arm[arm_kind] = route_document["records"]
            with np.load(
                arm_directory / "ENDPOINT_STATE.npz", allow_pickle=False
            ) as archive:
                tracer_indices = np.asarray(archive["tracer_indices"])
                tracer_lifecycle = np.asarray(archive["tracer_lifecycle"])
            if (
                tracer_indices.ndim != 1
                or tracer_lifecycle.shape != tracer_indices.shape
            ):
                raise RuntimeError(
                    f"{arm_kind} endpoint tracer roster is invalid for episode screening"
                )
            tracer_ids = tuple(int(value) for value in tracer_indices)
            terminal_ids = tuple(
                int(tracer_indices[index])
                for index in np.flatnonzero(
                    tracer_lifecycle != LIFECYCLE_ACTIVE
                )
            )
            tracer_ids_by_arm[arm_kind] = tracer_ids
            terminal_tracer_ids_by_arm[arm_kind] = terminal_ids
            measured_maximum = int(
                reports[arm_kind]
                .get("measurements", {})
                .get("maximum_route_count_for_one_tracer", -1)
            )
            ledger_counts = Counter(
                int(record["tracer_index"])
                for record in route_document["records"]
            )
            ledger_maximum = max(ledger_counts.values(), default=0)
            if measured_maximum != ledger_maximum:
                raise RuntimeError(
                    f"{arm_kind} episode ledger maximum disagrees with endpoint"
                )
        route_episode_capacity_screen = _route_episode_capacity_screen(
            route_records_by_arm=route_records_by_arm,
            tracer_ids_by_arm=tracer_ids_by_arm,
            terminal_tracer_ids_by_arm=terminal_tracer_ids_by_arm,
            observed_steps=stage.total_steps,
            target_steps=STAGES["one_myr"].total_steps,
            expected_arm_kinds=ARM_KINDS,
            required_pilot_steps=ROUTE_CAPACITY_PILOT_STEPS,
            required_tracer_count=STAGES["one_myr"].tracer_count,
        )
    else:
        route_episode_capacity_screen = {
            "schema": "jxplanetx.cuda-route-episode-capacity-screen.v1",
            "semantics": "NOT_EVALUATED_UNTIL_ALL_FOUR_ARMS_COMPLETE",
            "one_myr_capacity_screen_passed": False,
        }
    capacity_admission: dict[str, Any] | None = None
    capacity_gate_required = stage.stage_id == "capacity_pilot"
    capacity_gate_passed = not capacity_gate_required
    if complete and capacity_gate_required:
        if capacity_target_context is None:
            raise RuntimeError("capacity pilot lacks its exact one_myr target context")
        capacity_admission = _capacity_admission_document(
            stage=stage,
            checkpoint_years=checkpoint_years,
            stage_directory=stage_directory,
            schedule_set=schedule_set,
            receipt=receipt,
            protocol=protocol,
            runtime_identity=runtime_identity,
            runtime_receipt_sha256=runtime_receipt_sha256,
            reports=reports,
            route_episode_capacity_screen=route_episode_capacity_screen,
            target_context=capacity_target_context,
        )
        _publish_json_exact(
            stage_directory / "CAPACITY_ADMISSION.json", capacity_admission
        )
        capacity_gate_passed = (
            capacity_admission.get("one_myr_authorized") is True
        )
    stage_all_pass = all_arm_gates and capacity_gate_passed
    status = (
        "PARTIAL_RESUMABLE_SCREEN"
        if not complete
        else "FAILED_SCREEN_GATE"
        if not all_arm_gates
        else "FAILED_ROUTE_CAPACITY_GATE"
        if capacity_gate_required and not capacity_gate_passed
        else stage.completion_status
    )
    gates = {
        "all_four_arms_complete": complete,
        "all_arm_gates_pass": all_arm_gates,
    }
    if capacity_gate_required:
        gates.update(
            {
                "route_episode_capacity_gate_required": True,
                "route_episode_capacity_gate_passed": capacity_gate_passed,
            }
        )
    gates["all_pass"] = stage_all_pass
    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "completed_utc": (
            _utc_now()
            if partial_summary_commit is None
            else partial_summary_commit["completed_utc"]
        ),
        "status": status,
        "stage_id": stage.stage_id,
        "duration_years": stage.duration_years,
        "step_years": STEP_YEARS,
        "total_steps": stage.total_steps,
        "tracer_count": stage.tracer_count,
        "checkpoint_years": checkpoint_years,
        "protocol_sha256": protocol["content_sha256"],
        "matched_schedule_set_sha256": schedule_set.matched_set_sha256,
        "population_transform_receipt_sha256": receipt["content_sha256"],
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_receipt_sha256": runtime_receipt_sha256,
        "arm_status": {
            arm_kind: reports.get(arm_kind, {}).get("status", "NOT_RUN")
            for arm_kind in ARM_KINDS
        },
        "arm_result_sha256": {
            arm_kind: reports[arm_kind]["content_sha256"]
            for arm_kind in reports
        },
        "paired_endpoint_comparisons": comparisons,
        "route_episode_capacity_screen": route_episode_capacity_screen,
        "capacity_admission_sha256": (
            None
            if capacity_admission is None
            else capacity_admission["content_sha256"]
        ),
        "gates": gates,
        "bindings": protocol["bindings"],
        "classification": "MODEL_OUTPUT / SCREENING_ONLY",
        "interpretation_limits": list(INTERPRETATION_LIMITS),
        "claims": dict(RESULT_CLAIMS),
    }
    summary["content_sha256"] = _content_hash(REPORT_DOMAIN, summary)
    if complete:
        if partial_summary_commit is not None and summary != partial_summary_commit:
            raise RuntimeError(
                "partial stage summary differs from the recovered terminal result"
            )
        _publish_json_exact(summary_path, summary)
        files = sorted(
            path
            for path in stage_directory.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        checksum_text = "".join(
            f"{_sha256_file(path)}  {path.relative_to(stage_directory)}\n"
            for path in files
        )
        checksum_path = stage_directory / "SHA256SUMS"
        if checksum_path.exists():
            if checksum_path.read_text(encoding="utf-8") != checksum_text:
                raise RuntimeError("existing SHA256SUMS differs from current artifacts")
        else:
            _atomic_bytes_create(checksum_path, checksum_text.encode("utf-8"))
    return summary


def run(
    *,
    stage: Stage,
    output_root: Path,
    acceptance_state: Path,
    acceptance_protocol: Path,
    de441_control: Path,
    checkpoint_years: float,
    device: str,
    selected_arm_kinds: tuple[str, ...],
    stop_after_checkpoints: int | None,
    prepare_only: bool,
) -> dict[str, Any]:
    _verify_frozen_inputs(acceptance_state, acceptance_protocol, de441_control)
    schedule_set = _build_stage_schedule_set(stage, de441_control)
    population, receipt = _derive_population(acceptance_state, schedule_set)
    if population["tracer_indices"].size < stage.tracer_count:
        raise RuntimeError("the frozen population is smaller than the stage requirement")
    capacity_target_context = (
        _capacity_target_context(
            pilot_stage=stage,
            pilot_schedule_set=schedule_set,
            pilot_population=population,
            pilot_receipt=receipt,
            de441_control=de441_control,
            acceptance_state=acceptance_state,
        )
        if stage.stage_id == "capacity_pilot"
        else None
    )
    if (
        capacity_target_context is not None
        and capacity_target_context.get("gates", {}).get("all_pass") is not True
    ):
        raise RuntimeError(
            "capacity_pilot does not have an exact one_myr physical-prefix target"
        )
    protocol = _build_protocol(
        stage,
        checkpoint_years,
        schedule_set,
        receipt,
        device,
        capacity_target_context=capacity_target_context,
    )
    stage_directory = output_root / stage.stage_id
    if stage.stage_id == "one_myr":
        _capacity_pilot_prerequisite(
            output_root,
            population,
            receipt,
            schedule_set,
            protocol,
            None,
            None,
        )

    backend: BackendSpec | None = None
    runtime: Mapping[str, Any] | None = None
    if not prepare_only:
        backend = BackendSpec("cupy", device, 128)
        runtime = _current_runtime_identity(backend)
        if stage.stage_id == "one_myr":
            _capacity_pilot_prerequisite(
                output_root,
                population,
                receipt,
                schedule_set,
                protocol,
                runtime,
                backend,
            )

    completed_summary_path = stage_directory / "SUMMARY.json"
    completed_checksum_path = stage_directory / "SHA256SUMS"
    if completed_checksum_path.is_file() and not completed_summary_path.is_file():
        raise RuntimeError("stage has SHA256SUMS without SUMMARY.json")
    if (
        not prepare_only
        and completed_summary_path.is_file()
        and completed_checksum_path.is_file()
    ):
        assert backend is not None and runtime is not None
        allowed_statuses = (stage.completion_status,)
        if stage.stage_id == "capacity_pilot":
            allowed_statuses += ("FAILED_ROUTE_CAPACITY_GATE",)
        completed_summary = _load_verified_stage_summary(
            completed_summary_path,
            allowed_statuses=allowed_statuses,
            expected_runtime_identity=runtime,
            backend=None,
        )
        if (
            completed_summary.get("protocol_sha256")
            != protocol["content_sha256"]
            or completed_summary.get("matched_schedule_set_sha256")
            != schedule_set.matched_set_sha256
            or completed_summary.get("population_transform_receipt_sha256")
            != receipt["content_sha256"]
            or completed_summary.get("bindings") != protocol["bindings"]
        ):
            raise RuntimeError("completed stage summary has a different run identity")
        for arm in schedule_set.arms:
            report = _load_verified_arm_result(
                _arm_result_path(stage_directory, arm),
                backend=backend,
                stage=stage,
                checkpoint_years=checkpoint_years,
                stage_directory=stage_directory,
                arm=arm,
                schedule_set=schedule_set,
                protocol_sha256=str(protocol["content_sha256"]),
                receipt_sha256=str(receipt["content_sha256"]),
                runtime_identity=runtime,
                runtime_receipt_sha256=str(
                    completed_summary["runtime_identity_receipt_sha256"]
                ),
            )
            if (
                completed_summary["arm_result_sha256"].get(
                    arm.schedule.arm_kind
                )
                != report["content_sha256"]
            ):
                raise RuntimeError(
                    "completed stage summary arm digest differs from ARM_RESULT"
                )
        return completed_summary

    stage_directory.mkdir(parents=True, exist_ok=True)
    _publish_json_exact(stage_directory / "POPULATION_TRANSFORM_RECEIPT.json", receipt)
    _publish_npz_exact(stage_directory / "TRANSFORMED_TRACERS.npz", population)
    _publish_json_exact(stage_directory / "PROTOCOL.json", protocol)
    schedules = _schedule_documents(schedule_set)
    _publish_json_exact(
        stage_directory / "MATCHED_SCHEDULE_SET.json", schedules["matched_set"]
    )
    for arm_kind, document in schedules["arms"].items():
        _publish_json_exact(
            stage_directory / "schedules" / f"{ARM_DIRECTORY_NAMES[arm_kind]}.json",
            document,
        )

    if prepare_only:
        return {
            "status": "PREPARED_NOT_EXECUTED",
            "stage_id": stage.stage_id,
            "output_directory": str(stage_directory),
            "protocol_sha256": protocol["content_sha256"],
            "matched_schedule_set_sha256": schedule_set.matched_set_sha256,
            "population_transform_receipt_sha256": receipt["content_sha256"],
        }
    assert backend is not None and runtime is not None
    runtime_receipt = _freeze_stage_runtime_identity(
        stage_directory,
        protocol_sha256=str(protocol["content_sha256"]),
        current_runtime=runtime,
    )
    runtime_receipt_sha256 = str(runtime_receipt["content_sha256"])

    import cupy as cp

    cp.cuda.Device(int(runtime["device_index"])).use()

    arm_by_kind = {arm.schedule.arm_kind: arm for arm in schedule_set.arms}
    reports: dict[str, Mapping[str, Any]] = {}
    for arm_kind in selected_arm_kinds:
        _current_runtime_identity(
            backend,
            expected=runtime,
            label=f"current runtime before {arm_kind} arm",
        )
        arm = arm_by_kind[arm_kind]
        try:
            report = _run_arm(
                cp=cp,
                backend=backend,
                stage=stage,
                checkpoint_years=checkpoint_years,
                stage_directory=stage_directory,
                schedule_set=schedule_set,
                arm=arm,
                population=population,
                receipt=receipt,
                protocol=protocol,
                runtime_identity=runtime,
                runtime_receipt_sha256=runtime_receipt_sha256,
                stop_after_checkpoints=stop_after_checkpoints,
            )
        except BaseException as exc:
            _write_failure(stage_directory, arm_kind, exc)
            raise
        if report is not None:
            reports[arm_kind] = report

    # Include arms completed by earlier invocations so a final resumed arm can
    # close the matched summary without rerunning its predecessors.
    for arm in schedule_set.arms:
        path = _arm_result_path(stage_directory, arm)
        if path.is_file() and arm.schedule.arm_kind not in reports:
            reports[arm.schedule.arm_kind] = _load_verified_arm_result(
                path,
                backend=backend,
                stage=stage,
                checkpoint_years=checkpoint_years,
                stage_directory=stage_directory,
                arm=arm,
                schedule_set=schedule_set,
                protocol_sha256=str(protocol["content_sha256"]),
                receipt_sha256=str(receipt["content_sha256"]),
                runtime_identity=runtime,
                runtime_receipt_sha256=runtime_receipt_sha256,
            )
    _current_runtime_identity(
        backend,
        expected=runtime,
        label="current runtime before stage summary",
    )
    return _summarize(
        stage=stage,
        checkpoint_years=checkpoint_years,
        stage_directory=stage_directory,
        output_root=output_root,
        schedule_set=schedule_set,
        receipt=receipt,
        protocol=protocol,
        runtime_identity=runtime,
        runtime_receipt_sha256=runtime_receipt_sha256,
        reports=reports,
        capacity_target_context=capacity_target_context,
    )


def _parse_arms(values: list[str] | None) -> tuple[str, ...]:
    if not values or values == ["all"]:
        return ARM_KINDS
    inverse = {directory: arm for arm, directory in ARM_DIRECTORY_NAMES.items()}
    selected: list[str] = []
    for value in values:
        arm = inverse.get(value, value)
        if arm not in ARM_KINDS:
            raise ValueError(
                f"unknown arm {value!r}; choose all or {tuple(inverse)!r}"
            )
        if arm not in selected:
            selected.append(arm)
    return tuple(selected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGES), default="qualification")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--acceptance-state", type=Path, default=DEFAULT_ACCEPTANCE_STATE)
    parser.add_argument("--acceptance-protocol", type=Path, default=DEFAULT_ACCEPTANCE_PROTOCOL)
    parser.add_argument("--de441-control", type=Path, default=DEFAULT_DE441_CONTROL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--arm",
        action="append",
        help=(
            "all, control, circular_distant_planet_sensitivity, stellar_flyby, "
            "or unbound_rogue_flyby"
        ),
    )
    parser.add_argument(
        "--checkpoint-years",
        type=float,
        help="override the stage's immutable-segment cadence; recorded in the protocol",
    )
    parser.add_argument(
        "--stop-after-checkpoints",
        type=int,
        help="intentionally stop each selected arm after this many new checkpoints",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write and verify protocols, schedules, and population receipt without CUDA",
    )
    arguments = parser.parse_args()
    stage = STAGES[arguments.stage]
    checkpoint_years = (
        stage.checkpoint_years
        if arguments.checkpoint_years is None
        else arguments.checkpoint_years
    )
    if (
        arguments.stop_after_checkpoints is not None
        and arguments.stop_after_checkpoints <= 0
    ):
        parser.error("--stop-after-checkpoints must be positive")
    try:
        result = run(
            stage=stage,
            output_root=arguments.output_root.resolve(),
            acceptance_state=arguments.acceptance_state.resolve(),
            acceptance_protocol=arguments.acceptance_protocol.resolve(),
            de441_control=arguments.de441_control.resolve(),
            checkpoint_years=checkpoint_years,
            device=arguments.device,
            selected_arm_kinds=_parse_arms(arguments.arm),
            stop_after_checkpoints=arguments.stop_after_checkpoints,
            prepare_only=arguments.prepare_only,
        )
    except BaseException as exc:
        try:
            failure_stage_directory = (
                arguments.output_root.resolve() / stage.stage_id
            )
            target_preflight_failed_before_publish = (
                stage.stage_id == "one_myr"
                and not failure_stage_directory.exists()
            )
            if (
                not target_preflight_failed_before_publish
                and not _has_stage_finalization_artifact(
                    failure_stage_directory
                )
            ):
                _write_failure(
                    failure_stage_directory,
                    "RUN_LEVEL",
                    exc,
                )
        except BaseException:
            # Preserve the original failure on stderr even if the requested
            # evidence location itself is unavailable or unwritable.
            pass
        print(
            json.dumps(
                {
                    "status": "FAILED",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    status = result.get("status")
    return 1 if type(status) is str and status.startswith("FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
