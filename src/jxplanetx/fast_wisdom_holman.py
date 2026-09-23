#!/usr/bin/env python3
"""Supported fast CPU Wisdom--Holman screening interface.

This opt-in API keeps the public JX contract validation at the Python boundary,
then executes every fixed-step map operation and numerical guard inside one
compiled C call. Its default verification certificate binds the postconditions,
guards, and accounting evaluated across every native step without executing the
trajectory twice. Callers can still request a complete deterministic replay.
The interface remains screening-only: it is not a production backend,
scientific qualification, or general superiority claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import _wisdom_holman_cpu_v3 as _native
from .engine import wisdom_holman as _wh
from .engine.contracts import ForcePlan, StateSnapshot
from .engine.trajectory import TrajectoryDomainError
from .engine.wisdom_holman_contracts import FixedStepWisdomHolmanSpec


SCHEMA = "jxplanetx.fast-wisdom-holman.screening-api.v2"
API_ID = "jx.fast-wisdom-holman.screening-api.v2"
KERNEL_ID = "jx.wisdom-holman.native-complete-map.checkpoint-sync.prototype.v3"
G_KERNEL_ID = "jx.wisdom-holman.native-g-bundle.early-stop.prototype.v2"
EXACT_FALLBACK_KERNEL_ID = "jx.wisdom-holman.native-g-bundle.prototype.v1"
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"
EXECUTION_SCOPE = (
    "COMPLETE_FIXED_STEP_NUMERICAL_MAP_AND_GUARDS_IN_NATIVE_C_WITH_"
    "BINARY64_NO_CHANGE_SERIES_TERMINATION_AND_EXACT_V1_FALLBACK"
)
REPLAY_SCOPE = "EXPLICIT_OPTIONAL_COMPLETE_NATIVE_NUMERICAL_REPLAY"
CERTIFICATE_VERIFICATION_MODE = "COMPLETE_MAP_POSTCONDITION_CERTIFICATE"
CERTIFICATE_WITH_EXACT_REPLAY_MODE = (
    "COMPLETE_MAP_POSTCONDITION_CERTIFICATE_PLUS_EXACT_REPLAY"
)
CERTIFICATE_VERIFICATION_SCOPE = (
    "ALL_NATIVE_STEPS_KEPLER_POSTCONDITIONS_TRANSLATION_NODE_PATH_GUARDS_"
    "EXACT_ACCOUNTING_AND_SHA256_CUSTODY"
)
CERTIFICATE_WITH_EXACT_REPLAY_SCOPE = (
    CERTIFICATE_VERIFICATION_SCOPE
    + "_PLUS_SECOND_COMPLETE_BITWISE_NATIVE_EXECUTION"
)
_EXPECTED_COMPILER_FLAGS = (
    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"
)
NATIVE_WRAPPER_SOURCE_SHA256 = (
    "ee0e40a075b481516d8d30f8824b1fe31c9051c8c0de1dc8b321155882c411b3"
)
NATIVE_LOOP_SOURCE_SHA256 = (
    "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629"
)
NATIVE_BASE_LOOP_SOURCE_SHA256 = (
    "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750"
)

METRIC_NAMES = (
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

COUNTER_NAMES = (
    "completed_steps",
    "force_evaluations",
    "interaction_assemblies",
    "kepler_solves",
    "solver_iterations",
    "bracket_expansions",
    "g_evaluations",
    "series_terms",
    "node_guards",
    "path_guards",
    "forward_transforms",
    "inverse_transforms",
)

STATUS_NAMES = {
    0: "SUCCESS",
    1: "INPUT_DOMAIN",
    2: "NONFINITE",
    3: "FORCE_SINGULARITY",
    4: "TRANSLATION_RESIDUAL",
    5: "ORBITAL_GUARD",
    6: "NODE_PAIR_GUARD",
    7: "PATH_GUARD",
    8: "KEPLER_ROOT",
    9: "KEPLER_POSTCONDITION",
    10: "CHECKPOINT_SCHEDULE",
}


class FastWisdomHolmanLoopError(RuntimeError):
    """A fast Wisdom--Holman request failed its supported contract."""


@dataclass(frozen=True, eq=False)
class FastWisdomHolmanWorkspace:
    """Validated, owned inputs for repeatable native execution."""

    body_ids: tuple[str, ...]
    initial_epoch: float
    fixed_step: float
    checkpoint_steps: np.ndarray
    checkpoint_epochs: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    gravitational_parameters: np.ndarray
    radii: np.ndarray
    minimum_encounter_pair_separation: float
    minimum_jacobi_periapse: float
    maximum_initial_barycenter_position_norm: float
    maximum_initial_barycenter_velocity_norm: float
    input_content_sha256: str

    @property
    def body_count(self) -> int:
        return len(self.body_ids)

    @property
    def checkpoint_count(self) -> int:
        return int(self.checkpoint_steps.size)


@dataclass(frozen=True, eq=False)
class FastWisdomHolmanLoopResult:
    """Owned output and accounting from one primary run plus verification."""

    positions: np.ndarray
    velocities: np.ndarray
    checkpoint_epochs: np.ndarray
    metrics: tuple[tuple[str, float], ...]
    primary_counters: tuple[tuple[str, int], ...]
    replay_counters: tuple[tuple[str, int], ...]
    total_counters: tuple[tuple[str, int], ...]
    primary_seconds: float
    replay_seconds: float
    replay_count: int
    verification_mode: str
    verification_scope: str
    certified_steps: int
    certified_kepler_solves: int
    full_step_postcondition_coverage: bool
    exact_replay_verified: bool
    independent_numerical_method_verified: bool
    deterministic_implementation_defect_excluded: bool
    verification_content_sha256: str
    result_content_sha256: str
    input_content_sha256: str
    api_id: str = API_ID
    kernel_id: str = KERNEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    execution_scope: str = EXECUTION_SCOPE
    replay_scope: str = REPLAY_SCOPE
    registry_authorized: bool = False
    qualification_authorized: bool = False
    production_authorized: bool = False
    superiority_claimed: bool = False

    def __post_init__(self) -> None:
        if (
            self.api_id != API_ID
            or self.kernel_id != KERNEL_ID
            or self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE
            or self.registry_authorized
            or self.qualification_authorized
            or self.production_authorized
            or self.superiority_claimed
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman result identity or claim ceiling changed"
            )
        arrays = (self.positions, self.velocities, self.checkpoint_epochs)
        if any(
            type(value) is not np.ndarray
            or value.dtype != np.dtype(np.float64)
            or value.flags.writeable
            or not value.flags.c_contiguous
            or not np.all(np.isfinite(value))
            for value in arrays
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman results must be finite read-only float64 arrays"
            )
        if (
            self.positions.ndim != 3
            or self.positions.shape != self.velocities.shape
            or self.positions.shape[0] != self.checkpoint_epochs.size
            or self.positions.shape[2] != 3
            or self.checkpoint_epochs.ndim != 1
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman result shapes changed"
            )
        if self.replay_count not in (0, 1):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman replay count must be zero or one"
            )
        if self.verification_mode not in {
            CERTIFICATE_VERIFICATION_MODE,
            CERTIFICATE_WITH_EXACT_REPLAY_MODE,
        }:
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman verification mode is invalid"
            )
        expected_scope = {
            CERTIFICATE_VERIFICATION_MODE: CERTIFICATE_VERIFICATION_SCOPE,
            CERTIFICATE_WITH_EXACT_REPLAY_MODE: (
                CERTIFICATE_WITH_EXACT_REPLAY_SCOPE
            ),
        }[self.verification_mode]
        if self.verification_scope != expected_scope:
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman verification scope changed"
            )
        expected_replay = (
            self.verification_mode == CERTIFICATE_WITH_EXACT_REPLAY_MODE
        )
        if (
            self.replay_count != int(expected_replay)
            or self.exact_replay_verified is not expected_replay
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman replay verification accounting changed"
            )
        if (
            type(self.full_step_postcondition_coverage) is not bool
            or not self.full_step_postcondition_coverage
            or type(self.independent_numerical_method_verified) is not bool
            or self.independent_numerical_method_verified
            or type(self.deterministic_implementation_defect_excluded) is not bool
            or self.deterministic_implementation_defect_excluded
            or type(self.certified_steps) is not int
            or self.certified_steps <= 0
            or type(self.certified_kepler_solves) is not int
            or self.certified_kepler_solves <= 0
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman certificate claim boundary changed"
            )
        if (
            type(self.metrics) is not tuple
            or tuple(name for name, _ in self.metrics) != METRIC_NAMES
            or any(
                type(value) is not float or not math.isfinite(value)
                for _, value in self.metrics
            )
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman certificate metrics are invalid"
            )
        metric_values = dict(self.metrics)
        for name in METRIC_NAMES:
            if metric_values[name] < 0.0:
                raise FastWisdomHolmanLoopError(
                    "fast Wisdom--Holman certificate metric is negative"
                )
        for name in (
            "minimum_jacobi_periapse",
            "minimum_pair_endpoint",
            "minimum_pair_path_lower_bound",
            "minimum_pair_clearance_after_margin",
        ):
            if metric_values[name] <= 0.0:
                raise FastWisdomHolmanLoopError(
                    "fast Wisdom--Holman certificate minimum guard is invalid"
                )
        if (
            metric_values["maximum_kepler_time_residual"]
            > metric_values["maximum_kepler_residual_tolerance"]
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman Kepler residual certificate failed"
            )
        for name, counters in (
            ("primary", self.primary_counters),
            ("replay", self.replay_counters),
            ("total", self.total_counters),
        ):
            if (
                type(counters) is not tuple
                or tuple(key for key, _ in counters) != COUNTER_NAMES
                or any(type(value) is not int or value < 0 for _, value in counters)
            ):
                raise FastWisdomHolmanLoopError(
                    f"fast Wisdom--Holman {name} counters are invalid"
                )
        primary = dict(self.primary_counters)
        replay = dict(self.replay_counters)
        total = dict(self.total_counters)
        if (
            primary["completed_steps"] != self.certified_steps
            or primary["kepler_solves"] != self.certified_kepler_solves
            or any(
                total[name] != primary[name] + replay[name]
                for name in COUNTER_NAMES
            )
            or (
                expected_replay
                and any(replay[name] != primary[name] for name in COUNTER_NAMES)
            )
            or (
                not expected_replay
                and any(replay[name] != 0 for name in COUNTER_NAMES)
            )
        ):
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman certificate counter binding failed"
            )
        expected_certificate = _certificate_hash(
            input_content_sha256=self.input_content_sha256,
            result_content_sha256=self.result_content_sha256,
            verification_mode=self.verification_mode,
            certified_steps=self.certified_steps,
            certified_kepler_solves=self.certified_kepler_solves,
        )
        if self.verification_content_sha256 != expected_certificate:
            raise FastWisdomHolmanLoopError(
                "fast Wisdom--Holman certificate digest changed"
            )
        for digest in (
            self.result_content_sha256,
            self.input_content_sha256,
            self.verification_content_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise FastWisdomHolmanLoopError(
                    "fast Wisdom--Holman custody digest is invalid"
                )


@dataclass(frozen=True, eq=False)
class _RawRun:
    positions: np.ndarray
    velocities: np.ndarray
    metrics: np.ndarray
    counters: np.ndarray
    seconds: float
    content_sha256: str


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise FastWisdomHolmanLoopError(
            f"native prototype source is not a regular file: {path}"
        )
    raw = resolved.read_bytes()
    if not raw:
        raise FastWisdomHolmanLoopError(
            f"native prototype source is empty: {path}"
        )
    return {
        "path": resolved.as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def fast_wisdom_holman_runtime_identity() -> dict[str, Any]:
    """Return the packaged kernel and build identity."""

    if (
        _native.MAP_KERNEL_ID != KERNEL_ID
        or _native.KERNEL_ID != G_KERNEL_ID
        or _native.EXACT_FALLBACK_KERNEL_ID != EXACT_FALLBACK_KERNEL_ID
        or _native.MAP_MAX_BODIES != 32
        or _native.MAP_METRIC_COUNT != len(METRIC_NAMES)
        or _native.MAP_COUNTER_COUNT != len(COUNTER_NAMES)
        or _native.SCIENTIFIC_CLAIM_STATE != SCIENTIFIC_CLAIM_STATE
        or _native.COMPILER_FLAGS != _EXPECTED_COMPILER_FLAGS
    ):
        raise FastWisdomHolmanLoopError(
            "complete native Wisdom--Holman kernel identity changed"
        )
    return {
        "kernel_id": _native.MAP_KERNEL_ID,
        "g_kernel_id": _native.KERNEL_ID,
        "exact_fallback_kernel_id": _native.EXACT_FALLBACK_KERNEL_ID,
        "scientific_claim_state": _native.SCIENTIFIC_CLAIM_STATE,
        "maximum_bodies": _native.MAP_MAX_BODIES,
        "metric_count": _native.MAP_METRIC_COUNT,
        "counter_count": _native.MAP_COUNTER_COUNT,
        "compiler_version": _native.COMPILER_VERSION,
        "compiler_flags": _native.COMPILER_FLAGS,
        "release_wrapper_source_sha256": NATIVE_WRAPPER_SOURCE_SHA256,
        "release_loop_source_sha256": NATIVE_LOOP_SOURCE_SHA256,
        "release_base_loop_source_sha256": NATIVE_BASE_LOOP_SOURCE_SHA256,
        "extension_binary": _file_identity(Path(_native.__file__)),
    }


def _owned_readonly(value: Any, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True, subok=False)
    array.setflags(write=False)
    return array


def _update_array_hash(hasher: Any, label: str, array: np.ndarray) -> None:
    descriptor = json.dumps(
        {
            "label": label,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(len(descriptor).to_bytes(8, "big"))
    hasher.update(descriptor)
    raw = array.tobytes(order="C")
    hasher.update(len(raw).to_bytes(8, "big"))
    hasher.update(raw)


def _workspace_hash(
    *,
    body_ids: tuple[str, ...],
    initial_epoch: float,
    fixed_step: float,
    checkpoint_steps: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    gravitational_parameters: np.ndarray,
    radii: np.ndarray,
    guard_values: tuple[float, float, float, float],
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"jx.native-wh-complete-loop.input.v2\0")
    scalar = json.dumps(
        {
            "body_ids": list(body_ids),
            "initial_epoch_hex": initial_epoch.hex(),
            "fixed_step_hex": fixed_step.hex(),
            "guard_hex": [value.hex() for value in guard_values],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    hasher.update(len(scalar).to_bytes(8, "big"))
    hasher.update(scalar)
    for label, array in (
        ("checkpoint_steps", checkpoint_steps),
        ("positions", positions),
        ("velocities", velocities),
        ("gravitational_parameters", gravitational_parameters),
        ("radii", radii),
    ):
        _update_array_hash(hasher, label, array)
    return hasher.hexdigest()


def _result_hash(
    workspace: FastWisdomHolmanWorkspace,
    positions: np.ndarray,
    velocities: np.ndarray,
    metrics: np.ndarray,
    counters: np.ndarray,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"jx.native-wh-complete-loop.result.v3\0")
    hasher.update(bytes.fromhex(workspace.input_content_sha256))
    for label, array in (
        ("positions", positions),
        ("velocities", velocities),
        ("metrics", metrics),
        ("counters", counters),
    ):
        _update_array_hash(hasher, label, array)
    return hasher.hexdigest()


def _certificate_hash(
    *,
    input_content_sha256: str,
    result_content_sha256: str,
    verification_mode: str,
    certified_steps: int,
    certified_kepler_solves: int,
) -> str:
    payload = json.dumps(
        {
            "input_content_sha256": input_content_sha256,
            "result_content_sha256": result_content_sha256,
            "verification_mode": verification_mode,
            "certified_steps": certified_steps,
            "certified_kepler_solves": certified_kepler_solves,
            "full_step_postcondition_coverage": True,
            "independent_numerical_method_verified": False,
            "deterministic_implementation_defect_excluded": False,
            "native_loop_source_sha256": NATIVE_LOOP_SOURCE_SHA256,
            "native_base_loop_source_sha256": NATIVE_BASE_LOOP_SOURCE_SHA256,
            "native_wrapper_source_sha256": NATIVE_WRAPPER_SOURCE_SHA256,
        },
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return _sha256(b"jx.fast-wh.postcondition-certificate.v1\0" + payload)


def _validate_workspace_custody(workspace: FastWisdomHolmanWorkspace) -> None:
    if (
        type(workspace.body_ids) is not tuple
        or not 2 <= len(workspace.body_ids) <= _native.MAP_MAX_BODIES
        or any(type(value) is not str or not value for value in workspace.body_ids)
        or len(set(workspace.body_ids)) != len(workspace.body_ids)
    ):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace body identities are invalid"
        )
    scalar_values = (
        workspace.initial_epoch,
        workspace.fixed_step,
        workspace.minimum_encounter_pair_separation,
        workspace.minimum_jacobi_periapse,
        workspace.maximum_initial_barycenter_position_norm,
        workspace.maximum_initial_barycenter_velocity_norm,
    )
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in scalar_values
    ):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace scalars must be finite floats"
        )
    if workspace.fixed_step == 0.0 or any(value <= 0.0 for value in scalar_values[2:]):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace step and guards are invalid"
        )
    arrays = (
        (workspace.checkpoint_steps, np.dtype(np.int64)),
        (workspace.checkpoint_epochs, np.dtype(np.float64)),
        (workspace.positions, np.dtype(np.float64)),
        (workspace.velocities, np.dtype(np.float64)),
        (workspace.gravitational_parameters, np.dtype(np.float64)),
        (workspace.radii, np.dtype(np.float64)),
    )
    if any(
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.flags.writeable
        or not value.flags.c_contiguous
        for value, dtype in arrays
    ):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace arrays lost dtype or custody"
        )
    body_count = len(workspace.body_ids)
    checkpoint_count = workspace.checkpoint_steps.size
    if (
        workspace.checkpoint_steps.ndim != 1
        or checkpoint_count < 2
        or workspace.checkpoint_steps[0] != 0
        or np.any(np.diff(workspace.checkpoint_steps) <= 0)
        or workspace.checkpoint_epochs.shape != (checkpoint_count,)
        or workspace.positions.shape != (body_count, 3)
        or workspace.velocities.shape != (body_count, 3)
        or workspace.gravitational_parameters.shape != (body_count,)
        or workspace.radii.shape != (body_count,)
    ):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace array shapes are invalid"
        )
    if (
        not np.all(np.isfinite(workspace.checkpoint_epochs))
        or not np.all(np.isfinite(workspace.positions))
        or not np.all(np.isfinite(workspace.velocities))
        or not np.all(np.isfinite(workspace.gravitational_parameters))
        or not np.all(np.isfinite(workspace.radii))
        or np.any(workspace.gravitational_parameters <= 0.0)
        or np.any(workspace.radii < 0.0)
    ):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace numerical domain is invalid"
        )
    expected_epochs = np.asarray(
        [
            float(workspace.initial_epoch + workspace.fixed_step * int(index))
            for index in workspace.checkpoint_steps
        ],
        dtype=np.float64,
    )
    if not np.array_equal(workspace.checkpoint_epochs, expected_epochs):
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace epoch schedule changed"
        )
    guard_values = scalar_values[2:]
    expected_hash = _workspace_hash(
        body_ids=workspace.body_ids,
        initial_epoch=workspace.initial_epoch,
        fixed_step=workspace.fixed_step,
        checkpoint_steps=workspace.checkpoint_steps,
        positions=workspace.positions,
        velocities=workspace.velocities,
        gravitational_parameters=workspace.gravitational_parameters,
        radii=workspace.radii,
        guard_values=guard_values,
    )
    if workspace.input_content_sha256 != expected_hash:
        raise FastWisdomHolmanLoopError(
            "fast Wisdom--Holman workspace custody hash changed"
        )


def prepare_fast_wisdom_holman_workspace(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepWisdomHolmanSpec,
) -> FastWisdomHolmanWorkspace:
    """Apply the public boundary checks once and retain owned native inputs."""

    fast_wisdom_holman_runtime_identity()
    if type(snapshot) is not StateSnapshot:
        raise FastWisdomHolmanLoopError(
            "snapshot must be an exact StateSnapshot"
        )
    if type(plan) is not ForcePlan:
        raise FastWisdomHolmanLoopError("plan must be an exact ForcePlan")
    if type(spec) is not FixedStepWisdomHolmanSpec:
        raise FastWisdomHolmanLoopError(
            "spec must be an exact FixedStepWisdomHolmanSpec"
        )
    _wh._validate_snapshot_schema(snapshot, "snapshot")
    _wh._validate_force_plan_schema(plan, "plan")
    spec.__post_init__()
    _wh._validate_semantic_boundary(snapshot)
    model = _wh._validate_force_plan_scope(snapshot, plan, spec)
    _wh._validate_parameter_metadata_schema(model)
    _wh._validate_model_sequence(plan.models)
    _wh._validate_dependencies(plan.models)
    _wh._validate_parameter_contracts(snapshot, plan)
    backend = _wh.resolve_backend(plan.backend)
    if backend.name != "numpy" or backend.device != "cpu":
        raise FastWisdomHolmanLoopError(
            "complete native Wisdom--Holman requires the NumPy CPU boundary"
        )
    if len(snapshot.body_ids) > _native.MAP_MAX_BODIES:
        raise FastWisdomHolmanLoopError(
            f"complete native loop supports at most {_native.MAP_MAX_BODIES} bodies"
        )
    for name in (
        "positions",
        "velocities",
        "gravitational_parameters",
        "masses",
        "radii",
        "massive",
    ):
        if type(getattr(snapshot, name)) is not np.ndarray:
            raise FastWisdomHolmanLoopError(
                f"native Wisdom--Holman input {name} must be an exact ndarray"
            )

    all_epochs = _wh._step_epochs(snapshot.epoch, spec)
    checkpoint_epochs = tuple(
        all_epochs[index] for index in spec.checkpoint_step_indices
    )
    _wh._validate_metadata_range(
        plan,
        min(all_epochs[0], all_epochs[-1]),
        max(all_epochs[0], all_epochs[-1]),
    )
    with backend.activate():
        validated = _wh._validate_state(backend, snapshot)
        if bool(np.any(~validated[5])):
            raise FastWisdomHolmanLoopError(
                "complete native Wisdom--Holman requires all bodies active"
            )
        if bool(np.any(validated[2] <= np.float64(0.0))):
            raise FastWisdomHolmanLoopError(
                "complete native Wisdom--Holman requires positive GM"
            )
        secondary_ratio = _wh._sum_1d_fixed(validated[2], 1) / float(
            validated[2][0]
        )
        if (
            not math.isfinite(secondary_ratio)
            or secondary_ratio
            > _wh.WH_MAXIMUM_TOTAL_SECONDARY_TO_PRIMARY_GM_RATIO
        ):
            raise TrajectoryDomainError(
                "total secondary-to-primary GM ratio exceeds the Wisdom--Holman envelope"
            )
        positions = _owned_readonly(validated[0], np.dtype(np.float64))
        velocities = _owned_readonly(validated[1], np.dtype(np.float64))
        gravitational_parameters = _owned_readonly(
            validated[2], np.dtype(np.float64)
        )
        radii = _owned_readonly(validated[4], np.dtype(np.float64))

    checkpoint_steps = _owned_readonly(
        spec.checkpoint_step_indices, np.dtype(np.int64)
    )
    epoch_array = _owned_readonly(checkpoint_epochs, np.dtype(np.float64))
    guard_values = (
        spec.minimum_encounter_pair_separation,
        spec.minimum_jacobi_periapse,
        spec.maximum_initial_barycenter_position_norm,
        spec.maximum_initial_barycenter_velocity_norm,
    )
    input_hash = _workspace_hash(
        body_ids=snapshot.body_ids,
        initial_epoch=float(snapshot.epoch),
        fixed_step=spec.fixed_step,
        checkpoint_steps=checkpoint_steps,
        positions=positions,
        velocities=velocities,
        gravitational_parameters=gravitational_parameters,
        radii=radii,
        guard_values=guard_values,
    )
    return FastWisdomHolmanWorkspace(
        body_ids=snapshot.body_ids,
        initial_epoch=float(snapshot.epoch),
        fixed_step=spec.fixed_step,
        checkpoint_steps=checkpoint_steps,
        checkpoint_epochs=epoch_array,
        positions=positions,
        velocities=velocities,
        gravitational_parameters=gravitational_parameters,
        radii=radii,
        minimum_encounter_pair_separation=guard_values[0],
        minimum_jacobi_periapse=guard_values[1],
        maximum_initial_barycenter_position_norm=guard_values[2],
        maximum_initial_barycenter_velocity_norm=guard_values[3],
        input_content_sha256=input_hash,
    )


def _execute_once(workspace: FastWisdomHolmanWorkspace) -> _RawRun:
    _validate_workspace_custody(workspace)
    shape = (workspace.checkpoint_count, workspace.body_count, 3)
    positions = np.empty(shape, dtype=np.float64, order="C")
    velocities = np.empty(shape, dtype=np.float64, order="C")
    metrics = np.empty(len(METRIC_NAMES), dtype=np.float64, order="C")
    counters = np.empty(len(COUNTER_NAMES), dtype=np.int64, order="C")
    started = time.perf_counter()
    status = _native.integrate_map(
        workspace.body_count,
        workspace.checkpoint_count,
        workspace.positions,
        workspace.velocities,
        workspace.gravitational_parameters,
        workspace.radii,
        workspace.checkpoint_steps,
        workspace.fixed_step,
        workspace.minimum_encounter_pair_separation,
        workspace.minimum_jacobi_periapse,
        workspace.maximum_initial_barycenter_position_norm,
        workspace.maximum_initial_barycenter_velocity_norm,
        positions,
        velocities,
        metrics,
        counters,
    )
    seconds = time.perf_counter() - started
    if status != 0:
        status_name = STATUS_NAMES.get(status, "UNKNOWN")
        raise FastWisdomHolmanLoopError(
            f"complete native loop failed closed with status {status}: {status_name}"
        )
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(velocities))
        and np.all(np.isfinite(metrics))
    ):
        raise FastWisdomHolmanLoopError(
            "complete native loop returned nonfinite retained output"
        )
    expected_steps = int(workspace.checkpoint_steps[-1])
    expected = {
        "completed_steps": expected_steps,
        "force_evaluations": expected_steps + 1,
        "interaction_assemblies": expected_steps + 1,
        "kepler_solves": expected_steps * (workspace.body_count - 1),
        "node_guards": 2 * expected_steps + 1,
        "path_guards": expected_steps,
        "forward_transforms": 1,
        "inverse_transforms": expected_steps + workspace.checkpoint_count - 1,
    }
    observed = dict(zip(COUNTER_NAMES, map(int, counters), strict=True))
    for name, value in expected.items():
        if observed[name] != value:
            raise FastWisdomHolmanLoopError(
                f"native accounting mismatch for {name}: "
                f"expected {value}, observed {observed[name]}"
            )
    content_hash = _result_hash(
        workspace, positions, velocities, metrics, counters
    )
    return _RawRun(
        positions,
        velocities,
        metrics,
        counters,
        float(seconds),
        content_hash,
    )


def integrate_fast_wisdom_holman_workspace(
    workspace: FastWisdomHolmanWorkspace,
    *,
    exact_replay: bool = False,
) -> FastWisdomHolmanLoopResult:
    """Execute the native loop and return its complete postcondition certificate.

    Every call checks all native step postconditions, guards, counters, and
    custody. ``exact_replay=True`` additionally requests a complete bitwise
    deterministic replay using the same numerical implementation.
    """

    if type(workspace) is not FastWisdomHolmanWorkspace:
        raise FastWisdomHolmanLoopError(
            "workspace must be an exact FastWisdomHolmanWorkspace"
        )
    if type(exact_replay) is not bool:
        raise FastWisdomHolmanLoopError("exact_replay must be a built-in bool")
    primary = _execute_once(workspace)
    if exact_replay:
        replay = _execute_once(workspace)
        if (
            primary.content_sha256 != replay.content_sha256
            or not np.array_equal(primary.positions, replay.positions)
            or not np.array_equal(primary.velocities, replay.velocities)
            or not np.array_equal(primary.metrics, replay.metrics)
            or not np.array_equal(primary.counters, replay.counters)
        ):
            raise FastWisdomHolmanLoopError(
                "complete native deterministic replay differs bitwise"
            )
        replay_counters_array = replay.counters
        replay_seconds = replay.seconds
        replay_count = 1
        verification_mode = CERTIFICATE_WITH_EXACT_REPLAY_MODE
        verification_scope = CERTIFICATE_WITH_EXACT_REPLAY_SCOPE
    else:
        replay_counters_array = np.zeros_like(primary.counters)
        replay_seconds = 0.0
        replay_count = 0
        verification_mode = CERTIFICATE_VERIFICATION_MODE
        verification_scope = CERTIFICATE_VERIFICATION_SCOPE

    positions = _owned_readonly(primary.positions, np.dtype(np.float64))
    velocities = _owned_readonly(primary.velocities, np.dtype(np.float64))
    epochs = _owned_readonly(workspace.checkpoint_epochs, np.dtype(np.float64))
    metrics = tuple(
        (name, float(value))
        for name, value in zip(METRIC_NAMES, primary.metrics, strict=True)
    )
    primary_counters = tuple(
        (name, int(value))
        for name, value in zip(COUNTER_NAMES, primary.counters, strict=True)
    )
    replay_counters = tuple(
        (name, int(value))
        for name, value in zip(
            COUNTER_NAMES, replay_counters_array, strict=True
        )
    )
    total_counters = tuple(
        (name, int(primary_value + replay_value))
        for name, primary_value, replay_value in zip(
            COUNTER_NAMES,
            primary.counters,
            replay_counters_array,
            strict=True,
        )
    )
    primary_counter_map = dict(primary_counters)
    certified_steps = primary_counter_map["completed_steps"]
    certified_kepler_solves = primary_counter_map["kepler_solves"]
    verification_digest = _certificate_hash(
        input_content_sha256=workspace.input_content_sha256,
        result_content_sha256=primary.content_sha256,
        verification_mode=verification_mode,
        certified_steps=certified_steps,
        certified_kepler_solves=certified_kepler_solves,
    )
    return FastWisdomHolmanLoopResult(
        positions=positions,
        velocities=velocities,
        checkpoint_epochs=epochs,
        metrics=metrics,
        primary_counters=primary_counters,
        replay_counters=replay_counters,
        total_counters=total_counters,
        primary_seconds=primary.seconds,
        replay_seconds=float(replay_seconds),
        replay_count=replay_count,
        verification_mode=verification_mode,
        verification_scope=verification_scope,
        certified_steps=certified_steps,
        certified_kepler_solves=certified_kepler_solves,
        full_step_postcondition_coverage=True,
        exact_replay_verified=exact_replay,
        independent_numerical_method_verified=False,
        deterministic_implementation_defect_excluded=False,
        verification_content_sha256=verification_digest,
        result_content_sha256=primary.content_sha256,
        input_content_sha256=workspace.input_content_sha256,
    )


def integrate_fast_wisdom_holman_trajectory(
    snapshot: StateSnapshot,
    plan: ForcePlan,
    spec: FixedStepWisdomHolmanSpec,
    *,
    exact_replay: bool = False,
) -> FastWisdomHolmanLoopResult:
    """Validate public inputs, execute the native loop, and verify it."""

    workspace = prepare_fast_wisdom_holman_workspace(snapshot, plan, spec)
    return integrate_fast_wisdom_holman_workspace(workspace, exact_replay=exact_replay)


__all__ = [
    "API_ID",
    "CERTIFICATE_VERIFICATION_MODE",
    "CERTIFICATE_VERIFICATION_SCOPE",
    "CERTIFICATE_WITH_EXACT_REPLAY_MODE",
    "CERTIFICATE_WITH_EXACT_REPLAY_SCOPE",
    "COUNTER_NAMES",
    "EXECUTION_SCOPE",
    "EXACT_FALLBACK_KERNEL_ID",
    "G_KERNEL_ID",
    "KERNEL_ID",
    "METRIC_NAMES",
    "NATIVE_BASE_LOOP_SOURCE_SHA256",
    "NATIVE_LOOP_SOURCE_SHA256",
    "NATIVE_WRAPPER_SOURCE_SHA256",
    "FastWisdomHolmanLoopError",
    "FastWisdomHolmanLoopResult",
    "FastWisdomHolmanWorkspace",
    "REPLAY_SCOPE",
    "SCHEMA",
    "SCIENTIFIC_CLAIM_STATE",
    "integrate_fast_wisdom_holman_trajectory",
    "integrate_fast_wisdom_holman_workspace",
    "fast_wisdom_holman_runtime_identity",
    "prepare_fast_wisdom_holman_workspace",
]
