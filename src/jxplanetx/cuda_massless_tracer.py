"""Experimental CUDA restricted-N-body engine for massless tracer populations.

This module is deliberately additive: it does not widen the qualified public
Wisdom--Holman or encounter backends.  One mutually interacting point-mass
source system is advanced once per source-path chunk while independent, zero-mass
tracer lanes feel the sources without back-reacting on them.  The quiet map is
fixed-step inertial kick--drift--kick (velocity Verlet).

An encounter guard is transactional.  A guarded tracer step is not committed
on the GPU; that complete outer step is re-integrated on the CPU with DOP853
against the exact cubic-Hermite source segment, then the lane rejoins the GPU
at the next outer node.  This is an ``encounter_route``, not an array-backend
fallback and not evidence that a collision or physical encounter occurred.

The implementation and its results remain ``MODEL_OUTPUT / SCREENING_ONLY``.
In particular, this module does not qualify a 4.5-Gyr integration, a real TNO
population, or any Planet Nine, stellar-flyby, or rogue-flyby inference.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import cuda_tracer_events as _cuda_tracer_events
from .engine.backends import ArrayBackend, resolve_backend
from .engine.contracts import BackendSpec, ContractError, Provenance
from .cuda_tracer_events import (
    DOP853EventLocatorSpec,
    EVENT_PRIMARY_BOUNDARY_OUTWARD,
    EVENT_SOURCE_RADIUS_CONTACT,
    TracerEventError,
    cubic_hermite_source_state,
    locate_tracer_step_events,
)


def _import_time_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


METHOD_ID = "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V2"
KERNEL_ID = "jx.cuda.massless-tracer.restricted-kdk.v2"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
DETERMINISM_SCOPE = "SAME_PYTHON_NUMPY_SCIPY_CPU_AND_CUDA_RUNTIME_DEVICE"
IMPLEMENTATION_SOURCE_SHA256 = _import_time_file_sha256(Path(__file__))
EVENT_LOCATOR_SOURCE_SHA256 = _import_time_file_sha256(
    Path(_cuda_tracer_events.__file__)
)

STATUS_SUCCESS = 0
STATUS_ENCOUNTER_ROUTE = 1
STATUS_NONFINITE = 2
STATUS_FORCE_SINGULARITY = 3
STATUS_ACCELERATION_BOUND = 4
STATUS_REMOVED_SOURCE_RADIUS = 5
STATUS_REMOVED_OUTER_BOUNDARY = 6
STATUS_REMOVED_FROZEN = 7
STATUS_NAMES = {
    STATUS_SUCCESS: "SUCCESS",
    STATUS_ENCOUNTER_ROUTE: "ENCOUNTER_ROUTE",
    STATUS_NONFINITE: "NONFINITE",
    STATUS_FORCE_SINGULARITY: "FORCE_SINGULARITY",
    STATUS_ACCELERATION_BOUND: "ACCELERATION_BOUND",
    STATUS_REMOVED_SOURCE_RADIUS: "REMOVED_SOURCE_RADIUS",
    STATUS_REMOVED_OUTER_BOUNDARY: "REMOVED_OUTER_BOUNDARY",
    STATUS_REMOVED_FROZEN: "REMOVED_FROZEN",
}

REASON_NONE = 0
REASON_SWEPT_FLOOR = 1
REASON_ACCELERATION_BOUND = 2
REASON_COLLISION_SURFACE_POSSIBLE = 3
REASON_OUTER_BOUNDARY_POSSIBLE = 4
REASON_PRIMARY_PERIAPSE_RESOLUTION = 5
REASON_NAMES = {
    REASON_NONE: "NONE",
    REASON_SWEPT_FLOOR: "SWEPT_FLOOR",
    REASON_ACCELERATION_BOUND: "ACCELERATION_BOUND",
    REASON_COLLISION_SURFACE_POSSIBLE: "COLLISION_SURFACE_POSSIBLE",
    REASON_OUTER_BOUNDARY_POSSIBLE: "OUTER_BOUNDARY_POSSIBLE",
    REASON_PRIMARY_PERIAPSE_RESOLUTION: "PRIMARY_PERIAPSE_RESOLUTION",
}

LIFECYCLE_ACTIVE = 0
LIFECYCLE_REMOVED_SOURCE_RADIUS = 1
LIFECYCLE_REMOVED_OUTER_BOUNDARY = 2
LIFECYCLE_NAMES = {
    LIFECYCLE_ACTIVE: "ACTIVE",
    LIFECYCLE_REMOVED_SOURCE_RADIUS: "REMOVED_SOURCE_RADIUS",
    LIFECYCLE_REMOVED_OUTER_BOUNDARY: "REMOVED_OUTER_BOUNDARY",
}

CUDA_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)

MAXIMUM_SOURCE_COUNT = 32
MAXIMUM_THREADS_PER_BLOCK = 128
MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER = 16_384
MAXIMUM_TOTAL_ENCOUNTER_ROUTES = 100_000
MAXIMUM_EVENT_LOCATOR_ACCEPTED_STEPS = 100_000
MAXIMUM_METHOD_PRIMARY_DYNAMICAL_PHASE = 0.25
OUTWARD_EVENT_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR = 1.0e-13
CHECKPOINT_SCHEMA = "jxplanetx.cuda-massless-tracer-checkpoint.v4"
CHECKPOINT_DOMAIN = b"jxplanetx.cuda-massless-tracer-checkpoint.v4\x00"
CHECKPOINT_NAMESPACE_DOMAIN = b"jxplanetx.cuda-massless-tracer-namespace.v1\x00"
STATE_DOMAIN = b"jxplanetx.cuda-massless-tracer-state.v3\x00"
LINEAGE_IDENTITY_DOMAIN = b"jxplanetx.cuda-massless-tracer-lineage.v2\x00"
SOURCE_SEGMENT_DOMAIN = b"jxplanetx.cuda-massless-tracer-source-segment.v1\x00"
RESULT_CLAIMS = MappingProxyType(
    {
        "four_point_five_gyr_accuracy_qualified": False,
        "planet_nine_detected": False,
        "planet_nine_excluded": False,
        "stellar_flyby_inferred": False,
        "unbound_rogue_flyby_inferred": False,
        "close_encounter_confirmed": False,
        "collision_confirmed": False,
        "independently_reproduced": False,
    }
)


class CUDAMasslessTracerError(RuntimeError):
    """Base error for the experimental restricted tracer engine."""


class CUDAMasslessTracerContractError(CUDAMasslessTracerError, ValueError):
    """An input violates the explicit restricted-tracer contract."""


class CUDAMasslessTracerIntegrationError(CUDAMasslessTracerError):
    """A source or tracer lane failed closed during integration."""


class EncounterRouteError(CUDAMasslessTracerIntegrationError):
    """A guarded outer step could not be transactionally re-integrated."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise CUDAMasslessTracerContractError(
            f"{label} must be a nonempty, trimmed string"
        )
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CUDAMasslessTracerContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise CUDAMasslessTracerContractError(f"{label} must be a finite number")
    return result


def _finite_vector3(value: object, label: str) -> tuple[float, float, float]:
    if type(value) not in (tuple, list) or len(value) != 3:
        raise CUDAMasslessTracerContractError(
            f"{label} must be a length-three tuple/list"
        )
    checked = tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))
    return checked[0], checked[1], checked[2]


def _finite_matrix3(
    value: object, label: str
) -> tuple[tuple[float, float, float], ...]:
    if type(value) not in (tuple, list) or not value:
        raise CUDAMasslessTracerContractError(
            f"{label} must be a nonempty tuple/list of vectors"
        )
    return tuple(
        _finite_vector3(row, f"{label}[{index}]")
        for index, row in enumerate(value)
    )


def _finite_tuple(value: object, label: str) -> tuple[float, ...]:
    if type(value) not in (tuple, list) or not value:
        raise CUDAMasslessTracerContractError(
            f"{label} must be a nonempty tuple/list"
        )
    return tuple(_finite(item, f"{label}[{index}]") for index, item in enumerate(value))


def _sha256_text(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CUDAMasslessTracerContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    if value == "0" * 64:
        raise CUDAMasslessTracerContractError(f"{label} cannot be all zero")
    return value


@dataclass(frozen=True, eq=False)
class CUDAMasslessTracerState:
    """One device-resident restricted source/tracer state.

    Numerical buffers remain opaque in the constructor so creating a contract
    never causes an implicit host/device transfer.  ``advance`` performs strict
    native-CuPy, device, dtype, shape, contiguity, and domain validation.
    """

    snapshot_id: str
    epoch: float
    global_step: int
    time_scale: str
    frame: str
    origin: str
    axes: str
    length_unit: str
    time_unit: str
    unit_system_id: str
    source_ids: tuple[str, ...]
    primary_source_id: str
    source_positions: object
    source_velocities: object
    source_gm: object
    source_encounter_radii: object
    source_collision_radii: object
    relative_acceleration_bounds: object
    tracer_positions: object
    tracer_velocities: object
    tracer_indices: object
    provenance: Provenance
    force_schedule_sha256: str
    matched_schedule_set_sha256: str
    primary_outer_boundary_AU: float
    lattice_step_years: float | None = None
    minimum_source_distance_AU: object | None = None
    ever_unbound: object | None = None
    encounter_route_counts: object | None = None
    tracer_lifecycle: object | None = None
    prior_encounter_routes: tuple[Any, ...] = ()
    prior_removal_events: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _finite(self.epoch, "epoch")
        if type(self.global_step) is not int or self.global_step < 0:
            raise CUDAMasslessTracerContractError(
                "global_step must be a nonnegative integer"
            )
        for label in (
            "time_scale",
            "frame",
            "origin",
            "axes",
            "length_unit",
            "time_unit",
            "unit_system_id",
        ):
            _text(getattr(self, label), label)
        if type(self.source_ids) is not tuple or not self.source_ids:
            raise CUDAMasslessTracerContractError(
                "source_ids must be a nonempty immutable tuple"
            )
        for index, source_id in enumerate(self.source_ids):
            _text(source_id, f"source_ids[{index}]")
        if len(set(self.source_ids)) != len(self.source_ids):
            raise CUDAMasslessTracerContractError("source_ids must be unique")
        if len(self.source_ids) > MAXIMUM_SOURCE_COUNT:
            raise CUDAMasslessTracerContractError(
                f"at most {MAXIMUM_SOURCE_COUNT} sources are supported"
            )
        _text(self.primary_source_id, "primary_source_id")
        if self.primary_source_id != self.source_ids[0]:
            raise CUDAMasslessTracerContractError(
                "primary_source_id must explicitly identify source slot zero"
            )
        if (
            self.length_unit != "AU"
            or self.time_unit != "year"
            or self.unit_system_id != "AU-YEAR"
        ):
            raise CUDAMasslessTracerContractError(
                "v2 accepts only the canonical AU/year unit_system_id='AU-YEAR'"
            )
        _sha256_text(self.force_schedule_sha256, "force_schedule_sha256")
        _sha256_text(
            self.matched_schedule_set_sha256,
            "matched_schedule_set_sha256",
        )
        outer = _finite(
            self.primary_outer_boundary_AU, "primary_outer_boundary_AU"
        )
        if outer <= 0.0:
            raise CUDAMasslessTracerContractError(
                "primary_outer_boundary_AU must be positive"
            )
        if self.lattice_step_years is not None:
            lattice_step = _finite(
                self.lattice_step_years, "lattice_step_years"
            )
            if lattice_step == 0.0:
                raise CUDAMasslessTracerContractError(
                    "lattice_step_years cannot be zero"
                )
        for label in (
            "source_positions",
            "source_velocities",
            "source_gm",
            "source_encounter_radii",
            "source_collision_radii",
            "relative_acceleration_bounds",
            "tracer_positions",
            "tracer_velocities",
            "tracer_indices",
        ):
            if getattr(self, label) is None:
                raise CUDAMasslessTracerContractError(
                    f"{label} must be an explicit backend-native array"
                )
        if type(self.provenance) is not Provenance:
            raise CUDAMasslessTracerContractError("provenance must be Provenance")
        if type(self.prior_encounter_routes) is not tuple:
            raise CUDAMasslessTracerContractError(
                "prior_encounter_routes must be an immutable tuple"
            )
        if type(self.prior_removal_events) is not tuple:
            raise CUDAMasslessTracerContractError(
                "prior_removal_events must be an immutable tuple"
            )


@dataclass(frozen=True)
class FixedStepCUDAMasslessTracerSpec:
    """Fixed CUDA segment and guarded CPU encounter-route policy."""

    step_years: float
    step_count: int
    backend: BackendSpec
    maximum_encounter_routes_per_tracer: int = 256
    maximum_total_encounter_routes: int = MAXIMUM_TOTAL_ENCOUNTER_ROUTES
    encounter_route_method: str = "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2"
    encounter_route_rtol: float = 1.0e-12
    encounter_route_atol: float = 1.0e-14
    encounter_route_maximum_substep_fraction: float = 0.125
    maximum_primary_dynamical_phase: float = 0.05
    event_time_tolerance_fraction: float = 1.0e-12
    source_path_steps_per_chunk: int = 65_536
    threads_per_block: int = MAXIMUM_THREADS_PER_BLOCK

    def __post_init__(self) -> None:
        step = _finite(self.step_years, "step_years")
        if step == 0.0:
            raise CUDAMasslessTracerContractError("step_years cannot be zero")
        if type(self.step_count) is not int or self.step_count <= 0:
            raise CUDAMasslessTracerContractError(
                "step_count must be a positive integer"
            )
        if type(self.backend) is not BackendSpec:
            raise CUDAMasslessTracerContractError("backend must be BackendSpec")
        if self.backend.backend_id != "cupy":
            raise CUDAMasslessTracerContractError(
                "the CUDA tracer engine requires backend_id='cupy'"
            )
        if (
            type(self.maximum_encounter_routes_per_tracer) is not int
            or self.maximum_encounter_routes_per_tracer < 0
            or self.maximum_encounter_routes_per_tracer
            > MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER
        ):
            raise CUDAMasslessTracerContractError(
                "maximum_encounter_routes_per_tracer must be in "
                f"[0, {MAXIMUM_ENCOUNTER_ROUTES_PER_TRACER}]"
            )
        if (
            type(self.maximum_total_encounter_routes) is not int
            or self.maximum_total_encounter_routes < 0
            or self.maximum_total_encounter_routes
            > MAXIMUM_TOTAL_ENCOUNTER_ROUTES
        ):
            raise CUDAMasslessTracerContractError(
                "maximum_total_encounter_routes must be in "
                f"[0, {MAXIMUM_TOTAL_ENCOUNTER_ROUTES}]"
            )
        if (
            self.encounter_route_method
            != "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2"
        ):
            raise CUDAMasslessTracerContractError(
                "the only implemented encounter route is "
                "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2"
            )
        rtol = _finite(self.encounter_route_rtol, "encounter_route_rtol")
        if not 100.0 * np.finfo(np.float64).eps <= rtol < 1.0:
            raise CUDAMasslessTracerContractError(
                "encounter_route_rtol must be in [100 binary64 eps, 1)"
            )
        atol = _finite(self.encounter_route_atol, "encounter_route_atol")
        if atol <= 0.0:
            raise CUDAMasslessTracerContractError(
                "encounter_route_atol must be positive"
            )
        fraction = _finite(
            self.encounter_route_maximum_substep_fraction,
            "encounter_route_maximum_substep_fraction",
        )
        if not 0.0 < fraction <= 1.0:
            raise CUDAMasslessTracerContractError(
                "encounter_route_maximum_substep_fraction must be in (0, 1]"
            )
        maximum_phase = _finite(
            self.maximum_primary_dynamical_phase,
            "maximum_primary_dynamical_phase",
        )
        if not 0.0 < maximum_phase <= 0.25:
            raise CUDAMasslessTracerContractError(
                "maximum_primary_dynamical_phase must be in (0, 0.25]"
            )
        event_tolerance = _finite(
            self.event_time_tolerance_fraction,
            "event_time_tolerance_fraction",
        )
        if not (
            8.0 * np.finfo(np.float64).eps
            <= event_tolerance
            <= 1.0e-6
        ):
            raise CUDAMasslessTracerContractError(
                "event_time_tolerance_fraction must be in [8 binary64 eps, 1e-6]"
            )
        if (
            type(self.source_path_steps_per_chunk) is not int
            or self.source_path_steps_per_chunk <= 0
        ):
            raise CUDAMasslessTracerContractError(
                "source_path_steps_per_chunk must be a positive integer"
            )
        if (
            type(self.threads_per_block) is not int
            or not 32 <= self.threads_per_block <= MAXIMUM_THREADS_PER_BLOCK
            or self.threads_per_block % 32 != 0
        ):
            raise CUDAMasslessTracerContractError(
                "threads_per_block must be a multiple of 32 in [32, 128]"
            )


@dataclass(frozen=True)
class TracerEncounterRouteRecord:
    """Audit record for one discarded GPU step and full CPU redo."""

    tracer_index: int
    source_index: int
    source_id: str
    global_step: int
    reason: str
    source_segment_sha256: str
    source_segment_epoch_0: float
    source_segment_position_0_AU: tuple[tuple[float, float, float], ...]
    source_segment_velocity_0_AU_per_year: tuple[tuple[float, float, float], ...]
    source_segment_position_1_AU: tuple[tuple[float, float, float], ...]
    source_segment_velocity_1_AU_per_year: tuple[tuple[float, float, float], ...]
    source_segment_gm_AU3_per_year2: tuple[float, ...]
    source_segment_step_years: float
    tracer_position_0_AU: tuple[float, float, float]
    tracer_velocity_0_AU_per_year: tuple[float, float, float]
    tracer_position_1_AU: tuple[float, float, float]
    tracer_velocity_1_AU_per_year: tuple[float, float, float]
    route_relative_tolerance: float
    route_absolute_tolerance: float
    route_maximum_substep_fraction: float
    route_event_fraction_tolerance: float
    route_maximum_primary_dynamical_phase: float
    cpu_accepted_steps: int
    cpu_rejected_steps: int
    minimum_evaluated_stage_distance_AU: float
    minimum_committed_stage_distance_AU: float
    endpoint_unbound: bool
    outcome: str = "REJOINED"
    committed_step_fraction: float = 1.0
    maximum_accepted_primary_dynamical_phase: float = 0.0
    maximum_committed_primary_dynamical_phase: float = 0.0

    def __post_init__(self) -> None:
        if (
            type(self.tracer_index) is not int
            or not 0 <= self.tracer_index < 2**64
        ):
            raise CUDAMasslessTracerContractError(
                "route tracer_index must be a uint64-compatible integer"
            )
        if type(self.source_index) is not int or self.source_index < 0:
            raise CUDAMasslessTracerContractError(
                "route source_index must be nonnegative"
            )
        _text(self.source_id, "route source_id")
        if type(self.global_step) is not int or self.global_step < 0:
            raise CUDAMasslessTracerContractError(
                "route global_step must be nonnegative"
            )
        if self.reason not in REASON_NAMES.values() or self.reason == "NONE":
            raise CUDAMasslessTracerContractError("route reason is invalid")
        _sha256_text(self.source_segment_sha256, "route source_segment_sha256")
        _finite(self.source_segment_epoch_0, "route source_segment_epoch_0")
        segment_matrices = (
            "source_segment_position_0_AU",
            "source_segment_velocity_0_AU_per_year",
            "source_segment_position_1_AU",
            "source_segment_velocity_1_AU_per_year",
        )
        for label in segment_matrices:
            object.__setattr__(
                self,
                label,
                _finite_matrix3(getattr(self, label), f"route {label}"),
            )
        object.__setattr__(
            self,
            "source_segment_gm_AU3_per_year2",
            _finite_tuple(
                self.source_segment_gm_AU3_per_year2,
                "route source_segment_gm_AU3_per_year2",
            ),
        )
        if any(value <= 0.0 for value in self.source_segment_gm_AU3_per_year2):
            raise CUDAMasslessTracerContractError(
                "route source-segment GM values must be positive"
            )
        segment_count = len(self.source_segment_gm_AU3_per_year2)
        if any(
            len(getattr(self, label)) != segment_count
            for label in segment_matrices
        ):
            raise CUDAMasslessTracerContractError(
                "route source-segment arrays have inconsistent source counts"
            )
        if (
            _finite(
                self.source_segment_step_years,
                "route source_segment_step_years",
            )
            == 0.0
        ):
            raise CUDAMasslessTracerContractError(
                "route source-segment step cannot be zero"
            )
        for label in (
            "tracer_position_0_AU",
            "tracer_velocity_0_AU_per_year",
            "tracer_position_1_AU",
            "tracer_velocity_1_AU_per_year",
        ):
            object.__setattr__(
                self,
                label,
                _finite_vector3(getattr(self, label), f"route {label}"),
            )
        relative_tolerance = _finite(
            self.route_relative_tolerance, "route relative tolerance"
        )
        absolute_tolerance = _finite(
            self.route_absolute_tolerance, "route absolute tolerance"
        )
        maximum_substep = _finite(
            self.route_maximum_substep_fraction,
            "route maximum substep fraction",
        )
        event_tolerance = _finite(
            self.route_event_fraction_tolerance,
            "route event fraction tolerance",
        )
        maximum_primary_phase = _finite(
            self.route_maximum_primary_dynamical_phase,
            "route maximum primary dynamical phase",
        )
        if not 100.0 * np.finfo(np.float64).eps <= relative_tolerance < 1.0:
            raise CUDAMasslessTracerContractError(
                "route relative tolerance is outside the engine contract"
            )
        if absolute_tolerance <= 0.0 or not 0.0 < maximum_substep <= 1.0:
            raise CUDAMasslessTracerContractError(
                "route absolute tolerance or maximum substep is invalid"
            )
        if not 8.0 * np.finfo(np.float64).eps <= event_tolerance <= 1.0e-6:
            raise CUDAMasslessTracerContractError(
                "route event fraction tolerance is outside the engine contract"
            )
        if not 0.0 < maximum_primary_phase <= MAXIMUM_METHOD_PRIMARY_DYNAMICAL_PHASE:
            raise CUDAMasslessTracerContractError(
                "route primary phase policy is outside the engine contract"
            )
        if (
            type(self.cpu_accepted_steps) is not int
            or not 0 <= self.cpu_accepted_steps <= MAXIMUM_EVENT_LOCATOR_ACCEPTED_STEPS
        ):
            raise CUDAMasslessTracerContractError(
                "route cpu_accepted_steps is outside the locator resource ceiling"
            )
        if self.cpu_rejected_steps != -1:
            raise CUDAMasslessTracerContractError(
                "route cpu_rejected_steps must be -1 when unavailable"
            )
        distance = _finite(
            self.minimum_evaluated_stage_distance_AU,
            "route minimum_evaluated_stage_distance_AU",
        )
        if distance <= 0.0:
            raise CUDAMasslessTracerContractError(
                "route minimum evaluated stage distance must be positive"
            )
        committed_distance = _finite(
            self.minimum_committed_stage_distance_AU,
            "route minimum_committed_stage_distance_AU",
        )
        if committed_distance <= 0.0:
            raise CUDAMasslessTracerContractError(
                "route minimum committed stage distance must be positive"
            )
        if type(self.endpoint_unbound) is not bool:
            raise CUDAMasslessTracerContractError(
                "route endpoint_unbound must be bool"
            )
        if self.outcome not in {
            "REJOINED",
            "REMOVED_SOURCE_RADIUS",
            "REMOVED_OUTER_BOUNDARY",
        }:
            raise CUDAMasslessTracerContractError("route outcome is invalid")
        fraction = _finite(
            self.committed_step_fraction, "route committed_step_fraction"
        )
        if not 0.0 <= fraction <= 1.0:
            raise CUDAMasslessTracerContractError(
                "route committed_step_fraction must be in [0, 1]"
            )
        if self.outcome == "REJOINED" and fraction != 1.0:
            raise CUDAMasslessTracerContractError(
                "a rejoined route must commit the complete outer step"
            )
        phase = _finite(
            self.maximum_accepted_primary_dynamical_phase,
            "route maximum_accepted_primary_dynamical_phase",
        )
        if phase < 0.0:
            raise CUDAMasslessTracerContractError(
                "route maximum accepted primary phase cannot be negative"
            )
        committed_phase = _finite(
            self.maximum_committed_primary_dynamical_phase,
            "route maximum_committed_primary_dynamical_phase",
        )
        if committed_phase < 0.0:
            raise CUDAMasslessTracerContractError(
                "route maximum committed primary phase cannot be negative"
            )
        if committed_phase > MAXIMUM_METHOD_PRIMARY_DYNAMICAL_PHASE:
            raise CUDAMasslessTracerContractError(
                "route committed phase exceeds the method-wide ceiling"
            )
        diagnostic_slack = 64.0 * np.finfo(np.float64).eps * max(
            1.0, distance, committed_distance, phase, committed_phase
        )
        if distance > committed_distance + diagnostic_slack:
            raise CUDAMasslessTracerContractError(
                "route all-evaluated minimum cannot exceed committed minimum"
            )
        if committed_phase > phase + diagnostic_slack:
            raise CUDAMasslessTracerContractError(
                "route committed phase cannot exceed all-accepted phase"
            )
        if self.outcome == "REJOINED" and (
            committed_distance != distance or committed_phase != phase
        ):
            raise CUDAMasslessTracerContractError(
                "a rejoined route must commit all audited distance/phase work"
            )


@dataclass(frozen=True)
class TracerRemovalEventRecord:
    """One continuously located or committed-node operational removal."""

    tracer_index: int
    event_type: str
    global_step: int
    step_fraction: float
    source_index: int
    source_id: str
    source_segment_sha256: str
    source_segment_position_0_AU: tuple[tuple[float, float, float], ...]
    source_segment_velocity_0_AU_per_year: tuple[tuple[float, float, float], ...]
    source_segment_position_1_AU: tuple[tuple[float, float, float], ...]
    source_segment_velocity_1_AU_per_year: tuple[tuple[float, float, float], ...]
    source_segment_gm_AU3_per_year2: tuple[float, ...]
    source_segment_step_years: float
    surface_radius_AU: float
    located_distance_AU: float
    root_residual_AU: float
    surface_tolerance_AU: float
    position_AU: tuple[float, float, float]
    velocity_AU_per_year: tuple[float, float, float]
    surface_source_position_AU: tuple[float, float, float]
    surface_source_velocity_AU_per_year: tuple[float, float, float]
    primary_source_position_AU: tuple[float, float, float]
    primary_source_velocity_AU_per_year: tuple[float, float, float]
    primary_relative_specific_energy_AU2_per_year2: float
    primary_relative_radial_velocity_AU_per_year: float

    def __post_init__(self) -> None:
        if (
            type(self.tracer_index) is not int
            or not 0 <= self.tracer_index < 2**64
        ):
            raise CUDAMasslessTracerContractError(
                "removal tracer_index must be uint64-compatible"
            )
        if self.event_type not in {
            "SOURCE_RADIUS",
            "OUTER_BOUNDARY",
        }:
            raise CUDAMasslessTracerContractError(
                "removal event_type is invalid"
            )
        if type(self.global_step) is not int or self.global_step < 0:
            raise CUDAMasslessTracerContractError(
                "removal global_step must be nonnegative"
            )
        fraction = _finite(self.step_fraction, "removal step_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise CUDAMasslessTracerContractError(
                "removal step_fraction must be in [0, 1]"
            )
        if type(self.source_index) is not int or self.source_index < -1:
            raise CUDAMasslessTracerContractError(
                "removal source_index must be -1 or nonnegative"
            )
        if self.event_type == "SOURCE_RADIUS":
            if self.source_index < 0:
                raise CUDAMasslessTracerContractError(
                    "source-radius removal must identify a source"
                )
            _text(self.source_id, "removal source_id")
        elif self.source_index != -1 or self.source_id != "NONE":
            raise CUDAMasslessTracerContractError(
                "outer-boundary removal cannot identify a source"
            )
        _sha256_text(
            self.source_segment_sha256, "removal source_segment_sha256"
        )
        segment_matrices = (
            "source_segment_position_0_AU",
            "source_segment_velocity_0_AU_per_year",
            "source_segment_position_1_AU",
            "source_segment_velocity_1_AU_per_year",
        )
        for label in segment_matrices:
            object.__setattr__(
                self,
                label,
                _finite_matrix3(getattr(self, label), f"removal {label}"),
            )
        object.__setattr__(
            self,
            "source_segment_gm_AU3_per_year2",
            _finite_tuple(
                self.source_segment_gm_AU3_per_year2,
                "removal source_segment_gm_AU3_per_year2",
            ),
        )
        if any(value <= 0.0 for value in self.source_segment_gm_AU3_per_year2):
            raise CUDAMasslessTracerContractError(
                "removal source-segment GM values must be positive"
            )
        segment_count = len(self.source_segment_gm_AU3_per_year2)
        if any(len(getattr(self, label)) != segment_count for label in segment_matrices):
            raise CUDAMasslessTracerContractError(
                "removal source-segment arrays have inconsistent source counts"
            )
        if _finite(self.source_segment_step_years, "removal source_segment_step_years") == 0.0:
            raise CUDAMasslessTracerContractError(
                "removal source-segment step cannot be zero"
            )
        for label in (
            "surface_radius_AU",
            "located_distance_AU",
            "root_residual_AU",
            "surface_tolerance_AU",
        ):
            value = _finite(getattr(self, label), f"removal {label}")
            if value < 0.0:
                raise CUDAMasslessTracerContractError(
                    f"removal {label} cannot be negative"
                )
        if self.surface_tolerance_AU <= 0.0:
            raise CUDAMasslessTracerContractError(
                "removal surface_tolerance_AU must be positive"
            )
        if self.root_residual_AU > self.surface_tolerance_AU:
            raise CUDAMasslessTracerContractError(
                "removal root residual exceeds its effective surface tolerance"
            )
        for label in (
            "position_AU",
            "velocity_AU_per_year",
            "surface_source_position_AU",
            "surface_source_velocity_AU_per_year",
            "primary_source_position_AU",
            "primary_source_velocity_AU_per_year",
        ):
            object.__setattr__(
                self,
                label,
                _finite_vector3(getattr(self, label), f"removal {label}"),
            )
        _finite(
            self.primary_relative_specific_energy_AU2_per_year2,
            "removal primary-relative energy",
        )
        _finite(
            self.primary_relative_radial_velocity_AU_per_year,
            "removal primary-relative radial velocity",
        )


@dataclass(frozen=True, eq=False)
class CUDAMasslessTracerResult:
    """Device-resident result of one completely committed segment."""

    state: CUDAMasslessTracerState
    minimum_source_distance_AU: object
    ever_unbound: object
    encounter_route_counts: object
    tracer_lifecycle: object
    encounter_routes: tuple[TracerEncounterRouteRecord, ...]
    removal_events: tuple[TracerRemovalEventRecord, ...]
    runtime_identity: Mapping[str, Any]
    input_state_sha256: str
    method_id: str = METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE
    claims: Mapping[str, bool] = RESULT_CLAIMS

    def __post_init__(self) -> None:
        if type(self.state) is not CUDAMasslessTracerState:
            raise CUDAMasslessTracerContractError(
                "result state must be an exact CUDAMasslessTracerState"
            )
        if self.method_id != METHOD_ID:
            raise CUDAMasslessTracerContractError("result method_id is immutable")
        if self.scientific_classification != SCIENTIFIC_CLASSIFICATION:
            raise CUDAMasslessTracerContractError(
                "result scientific classification cannot be widened"
            )
        if self.project_scope != PROJECT_SCOPE:
            raise CUDAMasslessTracerContractError(
                "result project scope cannot be widened"
            )
        if dict(self.claims) != dict(RESULT_CLAIMS):
            raise CUDAMasslessTracerContractError(
                "result claim controls must remain the fixed all-false map"
            )
        _sha256_text(self.input_state_sha256, "result input_state_sha256")
        object.__setattr__(
            self, "claims", MappingProxyType(dict(RESULT_CLAIMS))
        )
        object.__setattr__(
            self, "runtime_identity", MappingProxyType(dict(self.runtime_identity))
        )


@dataclass(frozen=True)
class _ValidatedState:
    source_positions: Any
    source_velocities: Any
    source_gm: Any
    source_encounter_radii: Any
    source_collision_radii: Any
    relative_acceleration_bounds: Any
    tracer_positions: Any
    tracer_velocities: Any
    tracer_indices: Any
    source_count: int
    tracer_count: int


@dataclass(frozen=True)
class _ValidatedContinuation:
    minimum_source_distance_AU: Any
    ever_unbound: Any
    encounter_route_counts_host: np.ndarray
    tracer_lifecycle: Any
    encounter_routes: tuple[TracerEncounterRouteRecord, ...]
    removal_events: tuple[TracerRemovalEventRecord, ...]


@dataclass(frozen=True)
class _LaunchResult:
    source_positions: Any
    source_velocities: Any
    tracer_positions: Any
    tracer_velocities: Any
    minimum_source_distance: Any
    ever_unbound: Any
    status: Any
    reason: Any
    trigger_source: Any
    completed_steps: Any
    trigger_source_position_0: Any
    trigger_source_velocity_0: Any
    trigger_source_position_1: Any
    trigger_source_velocity_1: Any


@dataclass(frozen=True)
class _SourcePath:
    positions: Any
    velocities: Any
    accelerations: Any


@dataclass(frozen=True)
class _PathLaunchResult:
    tracer_positions: Any
    tracer_velocities: Any
    minimum_source_distance: Any
    ever_unbound: Any
    status: Any
    reason: Any
    trigger_source: Any
    completed_steps: Any


CUDA_SOURCE = r'''
#define JX_MAX_SOURCES 32
#define JX_MAX_COMPONENTS 96

#define JX_SUCCESS 0
#define JX_ENCOUNTER_ROUTE 1
#define JX_NONFINITE 2
#define JX_FORCE_SINGULARITY 3

#define JX_REASON_NONE 0
#define JX_REASON_SWEPT_FLOOR 1
#define JX_REASON_ACCELERATION_BOUND 2

__device__ __forceinline__ int
jx_finite3(const double x, const double y, const double z)
{
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__device__ __forceinline__ int
jx_source_acceleration(
    const double* position,
    const double* gm,
    const int source_count,
    const int target,
    double* ax,
    double* ay,
    double* az)
{
    const int i = target * 3;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int source = 0; source < source_count; ++source) {
        if (source == target) continue;
        const int j = source * 3;
        const double dx = position[j] - position[i];
        const double dy = position[j+1] - position[i+1];
        const double dz = position[j+2] - position[i+2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (!(d2 > 0.0) || !isfinite(d2)) return 0;
        const double inv3 = 1.0 / (d2 * sqrt(d2));
        x += gm[source] * dx * inv3;
        y += gm[source] * dy * inv3;
        z += gm[source] * dz * inv3;
    }
    *ax = x; *ay = y; *az = z;
    return jx_finite3(x, y, z);
}

__device__ __forceinline__ int
jx_tracer_acceleration(
    const double tx,
    const double ty,
    const double tz,
    const double* position,
    const double* gm,
    const int source_count,
    double* ax,
    double* ay,
    double* az)
{
    double x = 0.0, y = 0.0, z = 0.0;
    for (int source = 0; source < source_count; ++source) {
        const int j = source * 3;
        const double dx = position[j] - tx;
        const double dy = position[j+1] - ty;
        const double dz = position[j+2] - tz;
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (!(d2 > 0.0) || !isfinite(d2)) return 0;
        const double inv3 = 1.0 / (d2 * sqrt(d2));
        x += gm[source] * dx * inv3;
        y += gm[source] * dy * inv3;
        z += gm[source] * dz * inv3;
    }
    *ax = x; *ay = y; *az = z;
    return jx_finite3(x, y, z);
}

__device__ __forceinline__ int
jx_bound_to_primary(
    const double tx,
    const double ty,
    const double tz,
    const double tvx,
    const double tvy,
    const double tvz,
    const double* source_position,
    const double* source_velocity,
    const double primary_gm)
{
    const double rx = tx - source_position[0];
    const double ry = ty - source_position[1];
    const double rz = tz - source_position[2];
    const double vx = tvx - source_velocity[0];
    const double vy = tvy - source_velocity[1];
    const double vz = tvz - source_velocity[2];
    const double r2 = rx*rx + ry*ry + rz*rz;
    if (!(r2 > 0.0) || !isfinite(r2)) return 0;
    const double energy = 0.5*(vx*vx + vy*vy + vz*vz) - primary_gm/sqrt(r2);
    return isfinite(energy) && energy < 0.0;
}

extern "C" __global__
void jx_massless_tracer_restricted_kdk(
    const double* input_source_position,
    const double* input_source_velocity,
    const double* source_gm,
    const double* source_encounter_radius,
    const double* relative_acceleration_bound,
    const double* input_tracer_position,
    const double* input_tracer_velocity,
    const int source_count,
    const int tracer_count,
    const long long step_count,
    const double step,
    double* output_source_position,
    double* output_source_velocity,
    double* output_tracer_position,
    double* output_tracer_velocity,
    double* output_minimum_source_distance,
    int* output_ever_unbound,
    int* output_status,
    int* output_reason,
    int* output_trigger_source,
    long long* output_completed_steps,
    double* output_trigger_source_position_0,
    double* output_trigger_source_velocity_0,
    double* output_trigger_source_position_1,
    double* output_trigger_source_velocity_1)
{
    const int tid = (int)threadIdx.x;
    const int lane = (int)blockIdx.x * (int)blockDim.x + tid;
    const int tracer_active = lane < tracer_count;
    const int component_count = source_count * 3;

    __shared__ double sp[JX_MAX_COMPONENTS];
    __shared__ double sv[JX_MAX_COMPONENTS];
    __shared__ double sa[JX_MAX_COMPONENTS];
    __shared__ double sp0[JX_MAX_COMPONENTS];
    __shared__ double sv0[JX_MAX_COMPONENTS];
    __shared__ int source_status;

    if (tid == 0) source_status = JX_SUCCESS;
    if (tid < source_count) {
        const int i = tid * 3;
        sp[i] = input_source_position[i];
        sp[i+1] = input_source_position[i+1];
        sp[i+2] = input_source_position[i+2];
        sv[i] = input_source_velocity[i];
        sv[i+1] = input_source_velocity[i+1];
        sv[i+2] = input_source_velocity[i+2];
    }
    __syncthreads();

    if (tid < source_count) {
        const int i = tid * 3;
        double ax, ay, az;
        if (!jx_source_acceleration(sp, source_gm, source_count, tid, &ax, &ay, &az)) {
            atomicCAS(&source_status, JX_SUCCESS, JX_FORCE_SINGULARITY);
            ax = ay = az = 0.0;
        }
        sa[i] = ax; sa[i+1] = ay; sa[i+2] = az;
    }
    __syncthreads();

    double tx = 0.0, ty = 0.0, tz = 0.0;
    double tvx = 0.0, tvy = 0.0, tvz = 0.0;
    double tax = 0.0, tay = 0.0, taz = 0.0;
    double minimum_distance = 1.0/0.0;
    int ever_unbound = 0;
    int status = JX_SUCCESS;
    int reason = JX_REASON_NONE;
    int trigger_source = -1;
    long long completed_steps = 0;

    if (tracer_active) {
        const int i = lane * 3;
        tx = input_tracer_position[i];
        ty = input_tracer_position[i+1];
        tz = input_tracer_position[i+2];
        tvx = input_tracer_velocity[i];
        tvy = input_tracer_velocity[i+1];
        tvz = input_tracer_velocity[i+2];
        if (!jx_finite3(tx,ty,tz) || !jx_finite3(tvx,tvy,tvz)) {
            status = JX_NONFINITE;
        } else if (!jx_tracer_acceleration(
            tx,ty,tz,sp,source_gm,source_count,&tax,&tay,&taz)) {
            status = JX_FORCE_SINGULARITY;
        }
        for (int source = 0; source < source_count; ++source) {
            const int j = source * 3;
            const double dx = tx-sp[j], dy = ty-sp[j+1], dz = tz-sp[j+2];
            const double distance = sqrt(dx*dx + dy*dy + dz*dz);
            if (distance < minimum_distance) minimum_distance = distance;
        }
        ever_unbound = !jx_bound_to_primary(
            tx,ty,tz,tvx,tvy,tvz,sp,sv,source_gm[0]);
    }
    __syncthreads();

    const double absolute_step = fabs(step);
    for (long long iteration = 0; iteration < step_count; ++iteration) {
        int newly_triggered = 0;
        if (tracer_active && status == JX_SUCCESS) {
            double best_margin = 1.0/0.0;
            int best_source = -1;
            int best_reason = JX_REASON_NONE;
            for (int source = 0; source < source_count; ++source) {
                const int j = source * 3;
                const double rx = tx-sp[j], ry = ty-sp[j+1], rz = tz-sp[j+2];
                const double vx = tvx-sv[j], vy = tvy-sv[j+1], vz = tvz-sv[j+2];
                const double r2 = rx*rx + ry*ry + rz*rz;
                const double v2 = vx*vx + vy*vy + vz*vz;
                if (!(r2 > 0.0) || !isfinite(r2) || !isfinite(v2)) {
                    status = JX_FORCE_SINGULARITY;
                    break;
                }
                const double distance = sqrt(r2);
                if (distance < minimum_distance) minimum_distance = distance;
                const double rax = tax-sa[j], ray = tay-sa[j+1], raz = taz-sa[j+2];
                const double relative_acceleration = sqrt(rax*rax + ray*ray + raz*raz);
                const double acceleration_bound = relative_acceleration_bound[source];
                int local_reason = JX_REASON_NONE;
                double margin = 1.0/0.0;
                if (!isfinite(relative_acceleration) || relative_acceleration > acceleration_bound) {
                    local_reason = JX_REASON_ACCELERATION_BOUND;
                    margin = -1.0/0.0;
                } else {
                    const double certified_lower = distance - sqrt(v2)*absolute_step
                        - 0.5*acceleration_bound*absolute_step*absolute_step;
                    double tau = 0.0;
                    if (v2 > 0.0) {
                        tau = -(rx*vx + ry*vy + rz*vz) / (v2*step);
                        if (tau < 0.0) tau = 0.0;
                        if (tau > 1.0) tau = 1.0;
                    }
                    const double sx = rx + tau*step*vx;
                    const double sy = ry + tau*step*vy;
                    const double sz = rz + tau*step*vz;
                    const double linear_distance = sqrt(sx*sx + sy*sy + sz*sz);
                    const double guarded_distance = fmin(certified_lower, linear_distance);
                    margin = guarded_distance - source_encounter_radius[source];
                    if (!(margin > 0.0)) local_reason = JX_REASON_SWEPT_FLOOR;
                }
                if (local_reason != JX_REASON_NONE && margin < best_margin) {
                    best_margin = margin;
                    best_source = source;
                    best_reason = local_reason;
                }
            }
            if (best_source >= 0 && status == JX_SUCCESS) {
                status = JX_ENCOUNTER_ROUTE;
                reason = best_reason;
                trigger_source = best_source;
                completed_steps = iteration;
                newly_triggered = 1;
                const long long offset = (long long)lane * component_count;
                for (int component = 0; component < component_count; ++component) {
                    output_trigger_source_position_0[offset+component] = sp[component];
                    output_trigger_source_velocity_0[offset+component] = sv[component];
                }
            }
        }
        __syncthreads();

        if (tid < source_count) {
            const int i = tid * 3;
            sp0[i] = sp[i]; sp0[i+1] = sp[i+1]; sp0[i+2] = sp[i+2];
            sv0[i] = sv[i]; sv0[i+1] = sv[i+1]; sv0[i+2] = sv[i+2];
            const double vhx = sv[i] + 0.5*step*sa[i];
            const double vhy = sv[i+1] + 0.5*step*sa[i+1];
            const double vhz = sv[i+2] + 0.5*step*sa[i+2];
            sv0[i] = vhx; sv0[i+1] = vhy; sv0[i+2] = vhz;
            sp[i] += step*vhx;
            sp[i+1] += step*vhy;
            sp[i+2] += step*vhz;
        }
        __syncthreads();
        if (tid < source_count) {
            const int i = tid * 3;
            double ax, ay, az;
            if (!jx_source_acceleration(sp, source_gm, source_count, tid, &ax, &ay, &az)) {
                atomicCAS(&source_status, JX_SUCCESS, JX_FORCE_SINGULARITY);
                ax = ay = az = 0.0;
            }
            sa[i] = ax; sa[i+1] = ay; sa[i+2] = az;
        }
        __syncthreads();
        if (tid < source_count) {
            const int i = tid * 3;
            sv[i] = sv0[i] + 0.5*step*sa[i];
            sv[i+1] = sv0[i+1] + 0.5*step*sa[i+1];
            sv[i+2] = sv0[i+2] + 0.5*step*sa[i+2];
        }
        __syncthreads();

        if (tracer_active && newly_triggered) {
            const long long offset = (long long)lane * component_count;
            for (int component = 0; component < component_count; ++component) {
                output_trigger_source_position_1[offset+component] = sp[component];
                output_trigger_source_velocity_1[offset+component] = sv[component];
            }
        }
        if (tracer_active && status == JX_SUCCESS) {
            tvx += 0.5*step*tax;
            tvy += 0.5*step*tay;
            tvz += 0.5*step*taz;
            tx += step*tvx;
            ty += step*tvy;
            tz += step*tvz;
            if (!jx_tracer_acceleration(
                tx,ty,tz,sp,source_gm,source_count,&tax,&tay,&taz)) {
                status = JX_FORCE_SINGULARITY;
                completed_steps = iteration + 1;
            } else {
                tvx += 0.5*step*tax;
                tvy += 0.5*step*tay;
                tvz += 0.5*step*taz;
                if (!jx_finite3(tx,ty,tz) || !jx_finite3(tvx,tvy,tvz)) {
                    status = JX_NONFINITE;
                    completed_steps = iteration + 1;
                } else {
                    completed_steps = iteration + 1;
                    for (int source = 0; source < source_count; ++source) {
                        const int j = source * 3;
                        const double dx=tx-sp[j], dy=ty-sp[j+1], dz=tz-sp[j+2];
                        const double distance=sqrt(dx*dx+dy*dy+dz*dz);
                        if (distance < minimum_distance) minimum_distance=distance;
                    }
                    if (!jx_bound_to_primary(
                        tx,ty,tz,tvx,tvy,tvz,sp,sv,source_gm[0])) ever_unbound=1;
                }
            }
        }
        __syncthreads();
    }

    if (tid < source_count) {
        const int i = tid * 3;
        const long long offset = (long long)blockIdx.x * component_count + i;
        output_source_position[offset] = sp[i];
        output_source_position[offset+1] = sp[i+1];
        output_source_position[offset+2] = sp[i+2];
        output_source_velocity[offset] = sv[i];
        output_source_velocity[offset+1] = sv[i+1];
        output_source_velocity[offset+2] = sv[i+2];
    }
    if (tracer_active) {
        const int i = lane * 3;
        output_tracer_position[i] = tx;
        output_tracer_position[i+1] = ty;
        output_tracer_position[i+2] = tz;
        output_tracer_velocity[i] = tvx;
        output_tracer_velocity[i+1] = tvy;
        output_tracer_velocity[i+2] = tvz;
        output_minimum_source_distance[lane] = minimum_distance;
        output_ever_unbound[lane] = ever_unbound;
        output_status[lane] = source_status != JX_SUCCESS ? source_status : status;
        output_reason[lane] = reason;
        output_trigger_source[lane] = trigger_source;
        output_completed_steps[lane] = completed_steps;
    }
}
'''

SOURCE_PATH_CUDA_SOURCE = r'''
#define JX_MAX_SOURCES 32
#define JX_MAX_COMPONENTS 96
#define JX_SUCCESS 0
#define JX_FORCE_SINGULARITY 3

__device__ __forceinline__ int
jx_source_path_acceleration(
    const double* position,
    const double* gm,
    const int source_count,
    const int target,
    double* ax,
    double* ay,
    double* az)
{
    const int i = target * 3;
    double x = 0.0, y = 0.0, z = 0.0;
    for (int source = 0; source < source_count; ++source) {
        if (source == target) continue;
        const int j = source * 3;
        const double dx = position[j] - position[i];
        const double dy = position[j+1] - position[i+1];
        const double dz = position[j+2] - position[i+2];
        const double d2 = dx*dx + dy*dy + dz*dz;
        if (!(d2 > 0.0) || !isfinite(d2)) return 0;
        const double inv3 = 1.0 / (d2 * sqrt(d2));
        x += gm[source] * dx * inv3;
        y += gm[source] * dy * inv3;
        z += gm[source] * dz * inv3;
    }
    *ax = x; *ay = y; *az = z;
    return isfinite(x) && isfinite(y) && isfinite(z);
}

extern "C" __global__
void jx_build_massive_source_path(
    const double* input_position,
    const double* input_velocity,
    const double* gm,
    const int source_count,
    const long long step_count,
    const double step,
    double* output_position,
    double* output_velocity,
    double* output_acceleration,
    int* output_status)
{
    const int source = (int)threadIdx.x;
    __shared__ double position[JX_MAX_COMPONENTS];
    __shared__ double velocity[JX_MAX_COMPONENTS];
    __shared__ double acceleration[JX_MAX_COMPONENTS];
    __shared__ double half_velocity[JX_MAX_COMPONENTS];
    __shared__ int status;
    if (source == 0) status = JX_SUCCESS;
    if (source < source_count) {
        const int i = source * 3;
        for (int axis = 0; axis < 3; ++axis) {
            position[i+axis] = input_position[i+axis];
            velocity[i+axis] = input_velocity[i+axis];
        }
    }
    __syncthreads();
    if (source < source_count) {
        const int i = source * 3;
        double ax, ay, az;
        if (!jx_source_path_acceleration(
            position,gm,source_count,source,&ax,&ay,&az)) {
            atomicCAS(&status,JX_SUCCESS,JX_FORCE_SINGULARITY);
            ax=ay=az=0.0;
        }
        acceleration[i]=ax; acceleration[i+1]=ay; acceleration[i+2]=az;
    }
    __syncthreads();
    if (source < source_count) {
        const int i = source * 3;
        for (int axis = 0; axis < 3; ++axis) {
            output_position[i+axis] = position[i+axis];
            output_velocity[i+axis] = velocity[i+axis];
            output_acceleration[i+axis] = acceleration[i+axis];
        }
    }
    __syncthreads();
    for (long long iteration=0; iteration<step_count; ++iteration) {
        if (source < source_count) {
            const int i=source*3;
            for (int axis=0; axis<3; ++axis) {
                half_velocity[i+axis]=velocity[i+axis]+0.5*step*acceleration[i+axis];
                position[i+axis]+=step*half_velocity[i+axis];
            }
        }
        __syncthreads();
        if (source < source_count) {
            const int i=source*3;
            double ax,ay,az;
            if (!jx_source_path_acceleration(
                position,gm,source_count,source,&ax,&ay,&az)) {
                atomicCAS(&status,JX_SUCCESS,JX_FORCE_SINGULARITY);
                ax=ay=az=0.0;
            }
            acceleration[i]=ax; acceleration[i+1]=ay; acceleration[i+2]=az;
        }
        __syncthreads();
        if (source < source_count) {
            const int i=source*3;
            const long long offset=(iteration+1)*(long long)source_count*3+i;
            for (int axis=0; axis<3; ++axis) {
                velocity[i+axis]=half_velocity[i+axis]+0.5*step*acceleration[i+axis];
                output_position[offset+axis]=position[i+axis];
                output_velocity[offset+axis]=velocity[i+axis];
                output_acceleration[offset+axis]=acceleration[i+axis];
            }
        }
        __syncthreads();
    }
    if (source == 0) output_status[0]=status;
}
'''


TRACER_PATH_CUDA_SOURCE = r'''
#define JX_SUCCESS 0
#define JX_ENCOUNTER_ROUTE 1
#define JX_NONFINITE 2
#define JX_FORCE_SINGULARITY 3
#define JX_REMOVED_FROZEN 7
#define JX_REASON_NONE 0
#define JX_REASON_SWEPT_FLOOR 1
#define JX_REASON_ACCELERATION_BOUND 2
#define JX_REASON_COLLISION_SURFACE_POSSIBLE 3
#define JX_REASON_OUTER_BOUNDARY_POSSIBLE 4
#define JX_REASON_PRIMARY_PERIAPSE_RESOLUTION 5
#define JX_LIFECYCLE_ACTIVE 0

__device__ __forceinline__ int
jx_path_tracer_acceleration(
    const double tx,
    const double ty,
    const double tz,
    const double* source_position,
    const double* gm,
    const int source_count,
    double* ax,
    double* ay,
    double* az)
{
    double x=0.0,y=0.0,z=0.0;
    for (int source=0; source<source_count; ++source) {
        const int j=source*3;
        const double dx=source_position[j]-tx;
        const double dy=source_position[j+1]-ty;
        const double dz=source_position[j+2]-tz;
        const double d2=dx*dx+dy*dy+dz*dz;
        if (!(d2>0.0) || !isfinite(d2)) return 0;
        const double inv3=1.0/(d2*sqrt(d2));
        x+=gm[source]*dx*inv3;
        y+=gm[source]*dy*inv3;
        z+=gm[source]*dz*inv3;
    }
    *ax=x; *ay=y; *az=z;
    return isfinite(x) && isfinite(y) && isfinite(z);
}

__device__ __forceinline__ int
jx_path_bound_to_primary(
    const double tx,const double ty,const double tz,
    const double tvx,const double tvy,const double tvz,
    const double* source_position,const double* source_velocity,
    const double primary_gm)
{
    const double rx=tx-source_position[0];
    const double ry=ty-source_position[1];
    const double rz=tz-source_position[2];
    const double vx=tvx-source_velocity[0];
    const double vy=tvy-source_velocity[1];
    const double vz=tvz-source_velocity[2];
    const double r2=rx*rx+ry*ry+rz*rz;
    if (!(r2>0.0) || !isfinite(r2)) return 0;
    const double energy=0.5*(vx*vx+vy*vy+vz*vz)-primary_gm/sqrt(r2);
    return isfinite(energy) && energy<0.0;
}

extern "C" __global__
void jx_advance_tracers_on_source_path(
    const double* source_position_path,
    const double* source_velocity_path,
    const double* source_acceleration_path,
    const double* source_gm,
    const double* source_encounter_radius,
    const double* source_collision_radius,
    const double* relative_acceleration_bound,
    const double* input_tracer_position,
    const double* input_tracer_velocity,
    const int* input_tracer_lifecycle,
    const int source_count,
    const int tracer_count,
    const long long step_count,
    const double step,
    const double primary_outer_boundary,
    const double maximum_primary_dynamical_phase,
    double* output_tracer_position,
    double* output_tracer_velocity,
    double* output_minimum_source_distance,
    int* output_ever_unbound,
    int* output_status,
    int* output_reason,
    int* output_trigger_source,
    long long* output_completed_steps)
{
    const int lane=(int)blockIdx.x*(int)blockDim.x+(int)threadIdx.x;
    if (lane>=tracer_count) return;
    const int component_count=source_count*3;
    const int i=lane*3;
    double tx=input_tracer_position[i];
    double ty=input_tracer_position[i+1];
    double tz=input_tracer_position[i+2];
    double tvx=input_tracer_velocity[i];
    double tvy=input_tracer_velocity[i+1];
    double tvz=input_tracer_velocity[i+2];
    double tax=0.0,tay=0.0,taz=0.0;
    double minimum_distance=1.0/0.0;
    int ever_unbound=0;
    int status=JX_SUCCESS;
    int reason=JX_REASON_NONE;
    int trigger_source=-1;
    long long completed_steps=0;
    if (input_tracer_lifecycle[lane] != JX_LIFECYCLE_ACTIVE) {
        output_tracer_position[i]=tx;
        output_tracer_position[i+1]=ty;
        output_tracer_position[i+2]=tz;
        output_tracer_velocity[i]=tvx;
        output_tracer_velocity[i+1]=tvy;
        output_tracer_velocity[i+2]=tvz;
        output_minimum_source_distance[lane]=minimum_distance;
        output_ever_unbound[lane]=0;
        output_status[lane]=JX_REMOVED_FROZEN;
        output_reason[lane]=JX_REASON_NONE;
        output_trigger_source[lane]=-1;
        output_completed_steps[lane]=step_count;
        return;
    }
    const double* source_position=source_position_path;
    const double* source_velocity=source_velocity_path;
    const double* source_acceleration=source_acceleration_path;
    if (!isfinite(tx)||!isfinite(ty)||!isfinite(tz)
        ||!isfinite(tvx)||!isfinite(tvy)||!isfinite(tvz)) status=JX_NONFINITE;
    else if (!jx_path_tracer_acceleration(
        tx,ty,tz,source_position,source_gm,source_count,&tax,&tay,&taz))
        status=JX_FORCE_SINGULARITY;
    for (int source=0; source<source_count; ++source) {
        const int j=source*3;
        const double dx=tx-source_position[j];
        const double dy=ty-source_position[j+1];
        const double dz=tz-source_position[j+2];
        const double distance=sqrt(dx*dx+dy*dy+dz*dz);
        if (distance<minimum_distance) minimum_distance=distance;
    }
    ever_unbound=!jx_path_bound_to_primary(
        tx,ty,tz,tvx,tvy,tvz,source_position,source_velocity,source_gm[0]);
    const double absolute_step=fabs(step);
    for (long long iteration=0; iteration<step_count && status==JX_SUCCESS; ++iteration) {
        source_position=source_position_path+iteration*(long long)component_count;
        source_velocity=source_velocity_path+iteration*(long long)component_count;
        source_acceleration=source_acceleration_path+iteration*(long long)component_count;
        int best_priority=999;
        int best_source=-1;
        int best_reason=JX_REASON_NONE;
        for (int source=0; source<source_count; ++source) {
            const int j=source*3;
            const double rx=tx-source_position[j];
            const double ry=ty-source_position[j+1];
            const double rz=tz-source_position[j+2];
            const double vx=tvx-source_velocity[j];
            const double vy=tvy-source_velocity[j+1];
            const double vz=tvz-source_velocity[j+2];
            const double r2=rx*rx+ry*ry+rz*rz;
            const double v2=vx*vx+vy*vy+vz*vz;
            if (!(r2>0.0)||!isfinite(r2)||!isfinite(v2)) {
                status=JX_FORCE_SINGULARITY;
                break;
            }
            const double distance=sqrt(r2);
            if (distance<minimum_distance) minimum_distance=distance;
            const double rax=tax-source_acceleration[j];
            const double ray=tay-source_acceleration[j+1];
            const double raz=taz-source_acceleration[j+2];
            const double relative_acceleration=sqrt(rax*rax+ray*ray+raz*raz);
            const double acceleration_bound=relative_acceleration_bound[source];
            int local_reason=JX_REASON_NONE;
            int local_priority=999;
            if (!isfinite(relative_acceleration)||relative_acceleration>acceleration_bound) {
                local_reason=JX_REASON_ACCELERATION_BOUND;
                local_priority=0;
            } else {
                const double certified_lower=distance-sqrt(v2)*absolute_step
                    -0.5*acceleration_bound*absolute_step*absolute_step;
                const double certified_upper=distance+sqrt(v2)*absolute_step
                    +0.5*acceleration_bound*absolute_step*absolute_step;
                double tau=0.0;
                if (v2>0.0) {
                    tau=-(rx*vx+ry*vy+rz*vz)/(v2*step);
                    if (tau<0.0) tau=0.0;
                    if (tau>1.0) tau=1.0;
                }
                const double sx=rx+tau*step*vx;
                const double sy=ry+tau*step*vy;
                const double sz=rz+tau*step*vz;
                const double linear_distance=sqrt(sx*sx+sy*sy+sz*sz);
                const double guarded_distance=fmin(certified_lower,linear_distance);
                if (source_collision_radius[source]>0.0
                    && !(certified_lower>source_collision_radius[source])) {
                    local_reason=JX_REASON_COLLISION_SURFACE_POSSIBLE;
                    local_priority=1;
                } else if (source==0
                    && certified_lower<=primary_outer_boundary
                    && certified_upper>=primary_outer_boundary) {
                    local_reason=JX_REASON_OUTER_BOUNDARY_POSSIBLE;
                    local_priority=2;
                } else if (source==0
                    && (!(certified_lower>0.0)
                        || absolute_step*sqrt(source_gm[0]
                            /(certified_lower*certified_lower*certified_lower))
                            >= maximum_primary_dynamical_phase)) {
                    local_reason=JX_REASON_PRIMARY_PERIAPSE_RESOLUTION;
                    local_priority=3;
                } else if (!(guarded_distance>source_encounter_radius[source])) {
                    local_reason=JX_REASON_SWEPT_FLOOR;
                    local_priority=4;
                }
            }
            if (local_reason!=JX_REASON_NONE
                && (local_priority<best_priority
                    || (local_priority==best_priority && source<best_source))) {
                best_priority=local_priority;
                best_source=source;
                best_reason=local_reason;
            }
        }
        if (status!=JX_SUCCESS) break;
        if (best_source>=0) {
            status=JX_ENCOUNTER_ROUTE;
            reason=best_reason;
            trigger_source=best_source;
            completed_steps=iteration;
            break;
        }
        tvx+=0.5*step*tax; tvy+=0.5*step*tay; tvz+=0.5*step*taz;
        tx+=step*tvx; ty+=step*tvy; tz+=step*tvz;
        const double* next_position=source_position_path+(iteration+1)*(long long)component_count;
        const double* next_velocity=source_velocity_path+(iteration+1)*(long long)component_count;
        if (!jx_path_tracer_acceleration(
            tx,ty,tz,next_position,source_gm,source_count,&tax,&tay,&taz)) {
            status=JX_FORCE_SINGULARITY;
            completed_steps=iteration+1;
            break;
        }
        tvx+=0.5*step*tax; tvy+=0.5*step*tay; tvz+=0.5*step*taz;
        completed_steps=iteration+1;
        if (!isfinite(tx)||!isfinite(ty)||!isfinite(tz)
            ||!isfinite(tvx)||!isfinite(tvy)||!isfinite(tvz)) {
            status=JX_NONFINITE;
            break;
        }
        for (int source=0; source<source_count; ++source) {
            const int j=source*3;
            const double dx=tx-next_position[j];
            const double dy=ty-next_position[j+1];
            const double dz=tz-next_position[j+2];
            const double distance=sqrt(dx*dx+dy*dy+dz*dz);
            if (distance<minimum_distance) minimum_distance=distance;
        }
        if (!jx_path_bound_to_primary(
            tx,ty,tz,tvx,tvy,tvz,next_position,next_velocity,source_gm[0]))
            ever_unbound=1;
    }
    output_tracer_position[i]=tx;
    output_tracer_position[i+1]=ty;
    output_tracer_position[i+2]=tz;
    output_tracer_velocity[i]=tvx;
    output_tracer_velocity[i+1]=tvy;
    output_tracer_velocity[i+2]=tvz;
    output_minimum_source_distance[lane]=minimum_distance;
    output_ever_unbound[lane]=ever_unbound;
    output_status[lane]=status;
    output_reason[lane]=reason;
    output_trigger_source[lane]=trigger_source;
    output_completed_steps[lane]=completed_steps;
}
'''


KERNEL_SOURCE_SHA256 = hashlib.sha256(
    SOURCE_PATH_CUDA_SOURCE.encode("utf-8")
    + b"\x00"
    + TRACER_PATH_CUDA_SOURCE.encode("utf-8")
).hexdigest()
_KERNEL_CACHE: dict[tuple[Any, ...], Any] = {}


def _require_device_array(
    backend: ArrayBackend,
    value: object,
    label: str,
    *,
    dtype: Any,
    shape: tuple[int, ...] | None = None,
) -> Any:
    array = backend.require_native_array(value, label)
    if array.dtype != dtype:
        raise CUDAMasslessTracerContractError(
            f"{label} must have dtype {np.dtype(dtype).name}"
        )
    if shape is not None and array.shape != shape:
        raise CUDAMasslessTracerContractError(
            f"{label} must have shape {shape}; received {array.shape}"
        )
    if not bool(array.flags.c_contiguous):
        raise CUDAMasslessTracerContractError(f"{label} must be C-contiguous")
    return array


def _device_truth(backend: ArrayBackend, value: object, label: str) -> bool:
    return backend.scalar_bool(value, label)


def _validate_state(
    backend: ArrayBackend, state: CUDAMasslessTracerState
) -> _ValidatedState:
    if type(state) is not CUDAMasslessTracerState:
        raise CUDAMasslessTracerContractError(
            "state must be an exact CUDAMasslessTracerState"
        )
    xp = backend.xp
    source_count = len(state.source_ids)
    source_shape = (source_count, 3)
    source_positions = _require_device_array(
        backend, state.source_positions, "source_positions", dtype=xp.float64, shape=source_shape
    )
    source_velocities = _require_device_array(
        backend, state.source_velocities, "source_velocities", dtype=xp.float64, shape=source_shape
    )
    source_gm = _require_device_array(
        backend, state.source_gm, "source_gm", dtype=xp.float64, shape=(source_count,)
    )
    source_encounter_radii = _require_device_array(
        backend,
        state.source_encounter_radii,
        "source_encounter_radii",
        dtype=xp.float64,
        shape=(source_count,),
    )
    source_collision_radii = _require_device_array(
        backend,
        state.source_collision_radii,
        "source_collision_radii",
        dtype=xp.float64,
        shape=(source_count,),
    )
    relative_acceleration_bounds = _require_device_array(
        backend,
        state.relative_acceleration_bounds,
        "relative_acceleration_bounds",
        dtype=xp.float64,
        shape=(source_count,),
    )
    tracer_positions = _require_device_array(
        backend, state.tracer_positions, "tracer_positions", dtype=xp.float64
    )
    if tracer_positions.ndim != 2 or tracer_positions.shape[1:] != (3,):
        raise CUDAMasslessTracerContractError(
            "tracer_positions must have shape (tracer_count, 3)"
        )
    tracer_count = int(tracer_positions.shape[0])
    tracer_velocities = _require_device_array(
        backend,
        state.tracer_velocities,
        "tracer_velocities",
        dtype=xp.float64,
        shape=(tracer_count, 3),
    )
    tracer_indices = _require_device_array(
        backend,
        state.tracer_indices,
        "tracer_indices",
        dtype=xp.uint64,
        shape=(tracer_count,),
    )
    floating = (
        source_positions,
        source_velocities,
        source_gm,
        source_encounter_radii,
        source_collision_radii,
        relative_acceleration_bounds,
        tracer_positions,
        tracer_velocities,
    )
    if any(
        _device_truth(backend, xp.any(~xp.isfinite(array)), "finite-state validation")
        for array in floating
    ):
        raise CUDAMasslessTracerContractError("all state arrays must be finite")
    if _device_truth(backend, xp.any(source_gm <= 0.0), "source GM validation"):
        raise CUDAMasslessTracerContractError("source_gm must be strictly positive")
    source_delta = source_positions[:, None, :] - source_positions[None, :, :]
    source_distance2 = xp.sum(source_delta * source_delta, axis=2)
    source_distance2 = source_distance2 + xp.eye(
        source_count, dtype=xp.float64
    )
    if _device_truth(
        backend,
        xp.any(source_distance2 <= 0.0),
        "source-separation validation",
    ):
        raise CUDAMasslessTracerContractError(
            "distinct sources cannot occupy the same position"
        )
    if tracer_count:
        tracer_delta = (
            tracer_positions[:, None, :] - source_positions[None, :, :]
        )
        if _device_truth(
            backend,
            xp.any(xp.sum(tracer_delta * tracer_delta, axis=2) <= 0.0),
            "tracer-source separation validation",
        ):
            raise CUDAMasslessTracerContractError(
                "a tracer cannot begin at a point-mass source position"
            )
    if _device_truth(
        backend,
        xp.any(source_encounter_radii < 0.0),
        "encounter-radius validation",
    ):
        raise CUDAMasslessTracerContractError(
            "source_encounter_radii cannot be negative"
        )
    if _device_truth(
        backend,
        xp.any(source_collision_radii < 0.0),
        "collision-radius validation",
    ):
        raise CUDAMasslessTracerContractError(
            "source_collision_radii cannot be negative"
        )
    if _device_truth(
        backend,
        xp.any(source_collision_radii > source_encounter_radii),
        "collision/encounter-radius ordering validation",
    ):
        raise CUDAMasslessTracerContractError(
            "each source collision radius must not exceed its encounter radius"
        )
    if _device_truth(
        backend,
        source_collision_radii[0] >= state.primary_outer_boundary_AU,
        "primary terminal-surface ordering validation",
    ):
        raise CUDAMasslessTracerContractError(
            "the primary outer boundary must exceed its collision radius"
        )
    if _device_truth(
        backend,
        xp.any(relative_acceleration_bounds <= 0.0),
        "acceleration-bound validation",
    ):
        raise CUDAMasslessTracerContractError(
            "relative_acceleration_bounds must be strictly positive"
        )
    if tracer_count and int(xp.unique(tracer_indices).size) != tracer_count:
        raise CUDAMasslessTracerContractError(
            "tracer_indices must contain stable unique uint64 identities"
        )
    return _ValidatedState(
        source_positions=source_positions,
        source_velocities=source_velocities,
        source_gm=source_gm,
        source_encounter_radii=source_encounter_radii,
        source_collision_radii=source_collision_radii,
        relative_acceleration_bounds=relative_acceleration_bounds,
        tracer_positions=tracer_positions,
        tracer_velocities=tracer_velocities,
        tracer_indices=tracer_indices,
        source_count=source_count,
        tracer_count=tracer_count,
    )


def _validate_active_tracer_domain(
    backend: ArrayBackend,
    state: CUDAMasslessTracerState,
    validated: _ValidatedState,
    lifecycle: Any,
) -> None:
    """Reject ambiguous active lanes already on or beyond terminal surfaces."""

    if not validated.tracer_count:
        return
    cp = backend.xp
    active = lifecycle == LIFECYCLE_ACTIVE
    if not _device_truth(backend, cp.any(active), "active-tracer selection"):
        return
    active_positions = validated.tracer_positions[active]
    delta = (
        active_positions[:, None, :]
        - validated.source_positions[None, :, :]
    )
    distance = cp.sqrt(cp.sum(delta * delta, axis=2))
    positive_collision = validated.source_collision_radii > 0.0
    if _device_truth(
        backend,
        cp.any(
            (distance <= validated.source_collision_radii[None, :])
            & positive_collision[None, :]
        ),
        "active collision-surface domain validation",
    ):
        raise CUDAMasslessTracerContractError(
            "an active tracer must begin strictly outside every enabled "
            "source collision radius"
        )
    primary_distance = distance[:, 0]
    if _device_truth(
        backend,
        cp.any(primary_distance >= state.primary_outer_boundary_AU),
        "active outer-boundary domain validation",
    ):
        raise CUDAMasslessTracerContractError(
            "an active tracer must begin strictly inside the primary outer boundary"
        )


def _audit_float_equal(observed: float, expected: float, *scales: float) -> bool:
    scale = max(1.0, abs(observed), abs(expected), *(abs(value) for value in scales))
    return abs(observed - expected) <= 4096.0 * np.finfo(np.float64).eps * scale


def _host_point_mass_accelerations(
    positions: np.ndarray, gm: np.ndarray
) -> np.ndarray:
    count = int(positions.shape[0])
    result = np.zeros_like(positions)
    for target in range(count):
        for source in range(count):
            if source == target:
                continue
            delta = positions[source] - positions[target]
            distance2 = float(np.dot(delta, delta))
            if not distance2 > 0.0 or not math.isfinite(distance2):
                raise CUDAMasslessTracerContractError(
                    "route source preimage contains a point-mass singularity"
                )
            result[target] += (
                float(gm[source]) * delta / (distance2 * math.sqrt(distance2))
            )
    if not np.isfinite(result).all():
        raise CUDAMasslessTracerContractError(
            "route source preimage produced nonfinite acceleration"
        )
    return result


def _recompute_guard_trigger(
    state: CUDAMasslessTracerState,
    record: TracerEncounterRouteRecord,
    *,
    segment_position_0: np.ndarray,
    segment_velocity_0: np.ndarray,
    segment_gm: np.ndarray,
    encounter_radii: np.ndarray,
    collision_radii: np.ndarray,
    acceleration_bounds: np.ndarray,
) -> tuple[int, str]:
    tracer_position = np.asarray(record.tracer_position_0_AU, dtype=np.float64)
    tracer_velocity = np.asarray(
        record.tracer_velocity_0_AU_per_year, dtype=np.float64
    )
    source_acceleration = _host_point_mass_accelerations(
        segment_position_0, segment_gm
    )
    tracer_acceleration = np.zeros(3, dtype=np.float64)
    for source in range(segment_position_0.shape[0]):
        delta = segment_position_0[source] - tracer_position
        distance2 = float(np.dot(delta, delta))
        if not distance2 > 0.0 or not math.isfinite(distance2):
            raise CUDAMasslessTracerContractError(
                "route tracer preimage reaches a point-mass singularity"
            )
        tracer_acceleration += (
            float(segment_gm[source])
            * delta
            / (distance2 * math.sqrt(distance2))
        )
    best_priority = 999
    best_source = -1
    best_reason = "NONE"
    step = float(record.source_segment_step_years)
    absolute_step = abs(step)
    for source in range(segment_position_0.shape[0]):
        relative_position = tracer_position - segment_position_0[source]
        relative_velocity = tracer_velocity - segment_velocity_0[source]
        distance2 = float(np.dot(relative_position, relative_position))
        speed2 = float(np.dot(relative_velocity, relative_velocity))
        if not distance2 > 0.0 or not math.isfinite(distance2 + speed2):
            raise CUDAMasslessTracerContractError(
                "route guard preimage is singular or nonfinite"
            )
        distance = math.sqrt(distance2)
        relative_acceleration = float(
            np.linalg.norm(tracer_acceleration - source_acceleration[source])
        )
        bound = float(acceleration_bounds[source])
        local_reason = "NONE"
        local_priority = 999
        if not math.isfinite(relative_acceleration) or relative_acceleration > bound:
            local_reason = "ACCELERATION_BOUND"
            local_priority = 0
        else:
            certified_lower = (
                distance
                - math.sqrt(speed2) * absolute_step
                - 0.5 * bound * absolute_step * absolute_step
            )
            certified_upper = (
                distance
                + math.sqrt(speed2) * absolute_step
                + 0.5 * bound * absolute_step * absolute_step
            )
            tau = 0.0
            if speed2 > 0.0:
                tau = -float(np.dot(relative_position, relative_velocity)) / (
                    speed2 * step
                )
                tau = min(1.0, max(0.0, tau))
            swept = relative_position + tau * step * relative_velocity
            linear_distance = float(np.linalg.norm(swept))
            guarded_distance = min(certified_lower, linear_distance)
            if collision_radii[source] > 0.0 and not (
                certified_lower > collision_radii[source]
            ):
                local_reason = "COLLISION_SURFACE_POSSIBLE"
                local_priority = 1
            elif (
                source == 0
                and certified_lower <= state.primary_outer_boundary_AU
                and certified_upper >= state.primary_outer_boundary_AU
            ):
                local_reason = "OUTER_BOUNDARY_POSSIBLE"
                local_priority = 2
            elif source == 0 and (
                not certified_lower > 0.0
                or absolute_step
                * math.sqrt(
                    float(segment_gm[0]) / certified_lower**3
                )
                >= record.route_maximum_primary_dynamical_phase
            ):
                local_reason = "PRIMARY_PERIAPSE_RESOLUTION"
                local_priority = 3
            elif not guarded_distance > encounter_radii[source]:
                local_reason = "SWEPT_FLOOR"
                local_priority = 4
        if local_reason != "NONE" and (
            local_priority < best_priority
            or (local_priority == best_priority and source < best_source)
        ):
            best_priority = local_priority
            best_source = source
            best_reason = local_reason
    return best_source, best_reason


def _validate_route_witness(
    state: CUDAMasslessTracerState,
    validated: _ValidatedState,
    record: TracerEncounterRouteRecord,
    *,
    source_positions_host: np.ndarray,
    source_velocities_host: np.ndarray,
    source_gm_host: np.ndarray,
    encounter_radii_host: np.ndarray,
    collision_radii_host: np.ndarray,
    acceleration_bounds_host: np.ndarray,
) -> None:
    """Replay one persisted CPU route from its exact numerical preimage.

    The source barycentre and one-step KDK checks reject a displaced or locally
    impossible source witness.  The complete locator replay authenticates the
    tracer start/end state and the event/no-event outcome.  Membership in the
    full integrated source history is a runner-level audit; this replay is not
    an independent implementation.
    """

    segment_position_0 = np.ascontiguousarray(
        record.source_segment_position_0_AU, dtype=np.float64
    )
    segment_velocity_0 = np.ascontiguousarray(
        record.source_segment_velocity_0_AU_per_year, dtype=np.float64
    )
    segment_position_1 = np.ascontiguousarray(
        record.source_segment_position_1_AU, dtype=np.float64
    )
    segment_velocity_1 = np.ascontiguousarray(
        record.source_segment_velocity_1_AU_per_year, dtype=np.float64
    )
    segment_gm = np.ascontiguousarray(
        record.source_segment_gm_AU3_per_year2, dtype=np.float64
    )
    expected_shape = (validated.source_count, 3)
    if (
        segment_position_0.shape != expected_shape
        or segment_velocity_0.shape != expected_shape
        or segment_position_1.shape != expected_shape
        or segment_velocity_1.shape != expected_shape
        or segment_gm.shape != (validated.source_count,)
        or not np.array_equal(segment_gm, source_gm_host)
        or _source_segment_sha256(
            segment_position_0,
            segment_velocity_0,
            segment_position_1,
            segment_velocity_1,
            segment_gm,
            record.source_segment_step_years,
        )
        != record.source_segment_sha256
    ):
        raise CUDAMasslessTracerContractError(
            "a route does not reproduce its source-segment commitment"
        )
    expected_epoch_0 = state.epoch - (
        state.global_step - record.global_step
    ) * record.source_segment_step_years
    if not _audit_float_equal(
        record.source_segment_epoch_0,
        expected_epoch_0,
        state.epoch,
    ):
        raise CUDAMasslessTracerContractError(
            "a route source epoch is inconsistent with its lattice step"
        )
    initial_acceleration = _host_point_mass_accelerations(
        segment_position_0, segment_gm
    )
    half_velocity = (
        segment_velocity_0
        + 0.5 * record.source_segment_step_years * initial_acceleration
    )
    predicted_position_1 = (
        segment_position_0
        + record.source_segment_step_years * half_velocity
    )
    final_acceleration = _host_point_mass_accelerations(
        predicted_position_1, segment_gm
    )
    predicted_velocity_1 = (
        half_velocity
        + 0.5 * record.source_segment_step_years * final_acceleration
    )
    source_step_scale = max(
        1.0,
        float(np.max(np.abs(segment_position_1))),
        float(np.max(np.abs(segment_velocity_1))),
    )
    source_step_tolerance = max(
        1.0e-13,
        32768.0 * np.finfo(np.float64).eps * source_step_scale,
    )
    if (
        float(np.max(np.abs(predicted_position_1 - segment_position_1)))
        > source_step_tolerance
        or float(np.max(np.abs(predicted_velocity_1 - segment_velocity_1)))
        > source_step_tolerance
    ):
        raise CUDAMasslessTracerContractError(
            "a route source segment is not one valid KDK source step"
        )
    trigger_source, trigger_reason = _recompute_guard_trigger(
        state,
        record,
        segment_position_0=segment_position_0,
        segment_velocity_0=segment_velocity_0,
        segment_gm=segment_gm,
        encounter_radii=encounter_radii_host,
        collision_radii=collision_radii_host,
        acceleration_bounds=acceleration_bounds_host,
    )
    if trigger_source != record.source_index or trigger_reason != record.reason:
        raise CUDAMasslessTracerContractError(
            "a persisted route does not reproduce its CUDA guard decision"
        )

    total_gm = float(np.sum(source_gm_host))
    endpoint_barycentre_position = np.sum(
        source_positions_host * source_gm_host[:, None], axis=0
    ) / total_gm
    endpoint_barycentre_velocity = np.sum(
        source_velocities_host * source_gm_host[:, None], axis=0
    ) / total_gm
    segment_barycentre_position = np.sum(
        segment_position_0 * source_gm_host[:, None], axis=0
    ) / total_gm
    segment_barycentre_velocity = np.sum(
        segment_velocity_0 * source_gm_host[:, None], axis=0
    ) / total_gm
    elapsed = float(state.epoch - record.source_segment_epoch_0)
    expected_barycentre_position = (
        endpoint_barycentre_position - elapsed * endpoint_barycentre_velocity
    )
    barycentre_scale = max(
        1.0,
        float(np.max(np.abs(endpoint_barycentre_position))),
        float(np.max(np.abs(expected_barycentre_position))),
        abs(elapsed) * float(np.max(np.abs(endpoint_barycentre_velocity))),
    )
    position_tolerance = max(
        1.0e-8,
        32768.0 * np.finfo(np.float64).eps * barycentre_scale,
    )
    velocity_scale = max(
        1.0,
        float(np.max(np.abs(endpoint_barycentre_velocity))),
        float(np.max(np.abs(segment_barycentre_velocity))),
    )
    velocity_tolerance = max(
        1.0e-12,
        32768.0 * np.finfo(np.float64).eps * velocity_scale,
    )
    if (
        float(
            np.max(
                np.abs(
                    segment_barycentre_position
                    - expected_barycentre_position
                )
            )
        )
        > position_tolerance
        or float(
            np.max(
                np.abs(
                    segment_barycentre_velocity
                    - endpoint_barycentre_velocity
                )
            )
        )
        > velocity_tolerance
    ):
        raise CUDAMasslessTracerContractError(
            "a route source segment is not anchored to the endpoint barycentre"
        )

    locator_spec = DOP853EventLocatorSpec(
        relative_tolerance=record.route_relative_tolerance,
        absolute_tolerance=record.route_absolute_tolerance,
        maximum_substep_fraction=record.route_maximum_substep_fraction,
        event_fraction_tolerance=record.route_event_fraction_tolerance,
        outward_radial_velocity_tolerance_AU_per_year=(
            OUTWARD_EVENT_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR
        ),
        maximum_accepted_steps=MAXIMUM_EVENT_LOCATOR_ACCEPTED_STEPS,
    )
    try:
        replay = locate_tracer_step_events(
            np.ascontiguousarray(record.tracer_position_0_AU, dtype=np.float64),
            np.ascontiguousarray(
                record.tracer_velocity_0_AU_per_year, dtype=np.float64
            ),
            segment_position_0,
            segment_velocity_0,
            segment_position_1,
            segment_velocity_1,
            segment_gm,
            np.ascontiguousarray(collision_radii_host, dtype=np.float64),
            step_years=record.source_segment_step_years,
            primary_source_index=0,
            primary_boundary_radius_AU=state.primary_outer_boundary_AU,
            spec=locator_spec,
        )
    except TracerEventError as exc:
        raise CUDAMasslessTracerContractError(
            "a persisted route cannot be replayed from its numerical preimage"
        ) from exc

    expected_outcome = (
        "REMOVED_SOURCE_RADIUS"
        if replay.terminated_at_event
        and replay.earliest_event is not None
        and replay.earliest_event.event_type == EVENT_SOURCE_RADIUS_CONTACT
        else "REMOVED_OUTER_BOUNDARY"
        if replay.terminated_at_event
        else "REJOINED"
    )
    replay_position = replay.terminal_position_AU
    replay_velocity = replay.terminal_velocity_AU_per_year
    if (
        expected_outcome != record.outcome
        or replay.accepted_step_count != record.cpu_accepted_steps
        or not np.array_equal(
            replay_position,
            np.asarray(record.tracer_position_1_AU, dtype=np.float64),
        )
        or not np.array_equal(
            replay_velocity,
            np.asarray(record.tracer_velocity_1_AU_per_year, dtype=np.float64),
        )
        or not _audit_float_equal(
            replay.completed_fraction, record.committed_step_fraction
        )
        or not _audit_float_equal(
            replay.minimum_evaluated_stage_distance_AU,
            record.minimum_evaluated_stage_distance_AU,
        )
        or not _audit_float_equal(
            replay.minimum_committed_stage_distance_AU,
            record.minimum_committed_stage_distance_AU,
        )
        or not _audit_float_equal(
            replay.maximum_accepted_primary_dynamical_phase_radians,
            record.maximum_accepted_primary_dynamical_phase,
        )
        or not _audit_float_equal(
            replay.maximum_committed_primary_dynamical_phase_radians,
            record.maximum_committed_primary_dynamical_phase,
        )
        or replay.maximum_committed_primary_dynamical_phase_radians
        > record.route_maximum_primary_dynamical_phase
    ):
        raise CUDAMasslessTracerContractError(
            "a persisted route differs from deterministic locator replay"
        )
    if replay.terminated_at_event:
        event = replay.earliest_event
        assert event is not None
        replay_unbound = event.primary_specific_energy_AU2_per_year2 >= 0.0
    else:
        replay_unbound = _unbound_from_primary_host(
            replay_position,
            replay_velocity,
            segment_position_1[0],
            segment_velocity_1[0],
            float(segment_gm[0]),
        )
    if replay_unbound != record.endpoint_unbound:
        raise CUDAMasslessTracerContractError(
            "a persisted route has an incorrect endpoint-unbound diagnostic"
        )


def _validate_continuation(
    backend: ArrayBackend,
    state: CUDAMasslessTracerState,
    validated: _ValidatedState,
) -> _ValidatedContinuation:
    """Validate cumulative diagnostics and their immutable route ledger."""

    cp = backend.xp
    tracer_count = validated.tracer_count
    optional_values = (
        state.minimum_source_distance_AU,
        state.ever_unbound,
        state.encounter_route_counts,
        state.tracer_lifecycle,
    )
    missing = tuple(value is None for value in optional_values)
    if all(missing):
        if state.global_step != 0:
            raise CUDAMasslessTracerContractError(
                "cumulative diagnostics may be absent only at global_step zero"
            )
        if state.prior_encounter_routes:
            raise CUDAMasslessTracerContractError(
                "a new state cannot have a prior route ledger without diagnostics"
            )
        if state.prior_removal_events:
            raise CUDAMasslessTracerContractError(
                "a new state cannot have a removal ledger without diagnostics"
            )
        lifecycle = cp.zeros(tracer_count, dtype=cp.int32)
        _validate_active_tracer_domain(
            backend, state, validated, lifecycle
        )
        return _ValidatedContinuation(
            minimum_source_distance_AU=cp.full(
                tracer_count, cp.inf, dtype=cp.float64
            ),
            ever_unbound=cp.zeros(tracer_count, dtype=cp.bool_),
            encounter_route_counts_host=np.zeros(
                tracer_count, dtype=np.int64
            ),
            tracer_lifecycle=lifecycle,
            encounter_routes=(),
            removal_events=(),
        )
    if any(missing):
        raise CUDAMasslessTracerContractError(
            "continuation diagnostics must be supplied together"
        )
    minimum = _require_device_array(
        backend,
        state.minimum_source_distance_AU,
        "minimum_source_distance_AU",
        dtype=cp.float64,
        shape=(tracer_count,),
    ).copy()
    if _device_truth(
        backend,
        cp.any(~cp.isfinite(minimum) | (minimum <= 0.0)),
        "minimum-distance continuation validation",
    ):
        raise CUDAMasslessTracerContractError(
            "propagated minimum_source_distance_AU values must be finite and positive"
        )
    ever_unbound = _require_device_array(
        backend,
        state.ever_unbound,
        "ever_unbound",
        dtype=cp.bool_,
        shape=(tracer_count,),
    ).copy()
    route_counts = _require_device_array(
        backend,
        state.encounter_route_counts,
        "encounter_route_counts",
        dtype=cp.int64,
        shape=(tracer_count,),
    )
    if _device_truth(
        backend,
        cp.any(route_counts < 0),
        "encounter-route continuation validation",
    ):
        raise CUDAMasslessTracerContractError(
            "encounter_route_counts cannot be negative"
        )
    route_counts_host = cp.asnumpy(route_counts)
    lifecycle = _require_device_array(
        backend,
        state.tracer_lifecycle,
        "tracer_lifecycle",
        dtype=cp.int32,
        shape=(tracer_count,),
    ).copy()
    if _device_truth(
        backend,
        cp.any(
            (lifecycle != LIFECYCLE_ACTIVE)
            & (lifecycle != LIFECYCLE_REMOVED_SOURCE_RADIUS)
            & (lifecycle != LIFECYCLE_REMOVED_OUTER_BOUNDARY)
        ),
        "tracer-lifecycle validation",
    ):
        raise CUDAMasslessTracerContractError(
            "tracer_lifecycle contains an unknown state"
        )
    records = tuple(state.prior_encounter_routes)
    if any(type(row) is not TracerEncounterRouteRecord for row in records):
        raise CUDAMasslessTracerContractError(
            "prior_encounter_routes entries must be TracerEncounterRouteRecord"
        )
    canonical_records = tuple(
        sorted(
            records,
            key=lambda row: (row.global_step, row.tracer_index, row.source_index),
        )
    )
    if records != canonical_records:
        raise CUDAMasslessTracerContractError(
            "prior encounter routes must be in canonical order"
        )
    tracer_ids_host = cp.asnumpy(validated.tracer_indices)
    minimum_host = cp.asnumpy(minimum)
    ever_unbound_host = cp.asnumpy(ever_unbound)
    source_positions_host = cp.asnumpy(validated.source_positions)
    source_velocities_host = cp.asnumpy(validated.source_velocities)
    source_gm_host = cp.asnumpy(validated.source_gm)
    encounter_radii_host = cp.asnumpy(validated.source_encounter_radii)
    collision_radii_host = cp.asnumpy(validated.source_collision_radii)
    acceleration_bounds_host = cp.asnumpy(
        validated.relative_acceleration_bounds
    )
    tracer_positions_host = cp.asnumpy(validated.tracer_positions)
    tracer_velocities_host = cp.asnumpy(validated.tracer_velocities)
    lifecycle_host = cp.asnumpy(lifecycle)
    slot_for_tracer_id = {
        int(tracer_id): slot for slot, tracer_id in enumerate(tracer_ids_host)
    }
    active_slots = lifecycle_host == LIFECYCLE_ACTIVE
    if np.any(active_slots):
        active_positions = tracer_positions_host[active_slots]
        endpoint_distances = np.linalg.norm(
            active_positions[:, None, :] - source_positions_host[None, :, :],
            axis=2,
        )
        endpoint_minimum = np.min(endpoint_distances, axis=1)
        if np.any(minimum_host[active_slots] > endpoint_minimum):
            raise CUDAMasslessTracerContractError(
                "cumulative minimum excludes an active tracer endpoint"
            )
        primary_relative_position = (
            active_positions - source_positions_host[0]
        )
        primary_relative_velocity = (
            tracer_velocities_host[active_slots] - source_velocities_host[0]
        )
        primary_distance = np.linalg.norm(primary_relative_position, axis=1)
        endpoint_energy = (
            0.5 * np.sum(primary_relative_velocity**2, axis=1)
            - float(source_gm_host[0]) / primary_distance
        )
        if np.any(
            (endpoint_energy >= 0.0) & ~ever_unbound_host[active_slots]
        ):
            raise CUDAMasslessTracerContractError(
                "a nonnegative-energy active endpoint must retain ever_unbound"
            )
    ledger_counts = np.zeros(tracer_count, dtype=np.int64)
    seen_route_nodes: set[tuple[int, int]] = set()
    for record in records:
        slot = slot_for_tracer_id.get(record.tracer_index)
        if slot is None:
            raise CUDAMasslessTracerContractError(
                "a prior encounter route names a tracer outside this state"
            )
        if (
            record.source_index >= validated.source_count
            or state.source_ids[record.source_index] != record.source_id
        ):
            raise CUDAMasslessTracerContractError(
                "a prior encounter route has inconsistent source identity"
            )
        if record.global_step >= state.global_step:
            raise CUDAMasslessTracerContractError(
                "a prior encounter route is not earlier than the state endpoint"
            )
        _validate_route_witness(
            state,
            validated,
            record,
            source_positions_host=source_positions_host,
            source_velocities_host=source_velocities_host,
            source_gm_host=source_gm_host,
            encounter_radii_host=encounter_radii_host,
            collision_radii_host=collision_radii_host,
            acceleration_bounds_host=acceleration_bounds_host,
        )
        route_node = (record.global_step, record.tracer_index)
        if route_node in seen_route_nodes:
            raise CUDAMasslessTracerContractError(
                "prior encounter route ledger contains a duplicate tracer step"
            )
        seen_route_nodes.add(route_node)
        if record.endpoint_unbound and not bool(ever_unbound_host[slot]):
            raise CUDAMasslessTracerContractError(
                "route endpoint-unbound record contradicts cumulative diagnostics"
            )
        committed_route_minimum = (
            record.minimum_evaluated_stage_distance_AU
            if record.outcome == "REJOINED"
            else record.minimum_committed_stage_distance_AU
        )
        if minimum_host[slot] > committed_route_minimum:
            raise CUDAMasslessTracerContractError(
                "route committed minimum contradicts cumulative diagnostics"
            )
        ledger_counts[slot] += 1
    if not np.array_equal(ledger_counts, route_counts_host):
        raise CUDAMasslessTracerContractError(
            "encounter-route counts and immutable route ledger disagree"
        )
    removal_records = tuple(state.prior_removal_events)
    if any(type(row) is not TracerRemovalEventRecord for row in removal_records):
        raise CUDAMasslessTracerContractError(
            "prior_removal_events entries must be TracerRemovalEventRecord"
        )
    canonical_removals = tuple(
        sorted(
            removal_records,
            key=lambda row: (
                row.global_step,
                row.step_fraction,
                row.tracer_index,
                row.event_type,
                row.source_index,
            ),
        )
    )
    if removal_records != canonical_removals:
        raise CUDAMasslessTracerContractError(
            "prior removal events must be in canonical order"
        )
    removal_for_slot: dict[int, TracerRemovalEventRecord] = {}
    removal_event_minimum: dict[tuple[int, int], float] = {}
    for removal in removal_records:
        slot = slot_for_tracer_id.get(removal.tracer_index)
        if slot is None:
            raise CUDAMasslessTracerContractError(
                "a prior removal event names a tracer outside this state"
            )
        if slot in removal_for_slot:
            raise CUDAMasslessTracerContractError(
                "a tracer may have at most one terminal removal event"
            )
        if removal.global_step >= state.global_step:
            raise CUDAMasslessTracerContractError(
                "a prior removal event is not earlier than the state endpoint"
            )
        segment_position_0 = np.ascontiguousarray(
            removal.source_segment_position_0_AU, dtype=np.float64
        )
        segment_velocity_0 = np.ascontiguousarray(
            removal.source_segment_velocity_0_AU_per_year, dtype=np.float64
        )
        segment_position_1 = np.ascontiguousarray(
            removal.source_segment_position_1_AU, dtype=np.float64
        )
        segment_velocity_1 = np.ascontiguousarray(
            removal.source_segment_velocity_1_AU_per_year, dtype=np.float64
        )
        segment_gm = np.ascontiguousarray(
            removal.source_segment_gm_AU3_per_year2, dtype=np.float64
        )
        expected_segment_shape = (validated.source_count, 3)
        if (
            segment_position_0.shape != expected_segment_shape
            or segment_velocity_0.shape != expected_segment_shape
            or segment_position_1.shape != expected_segment_shape
            or segment_velocity_1.shape != expected_segment_shape
            or segment_gm.shape != (validated.source_count,)
            or not np.array_equal(segment_gm, source_gm_host)
            or _source_segment_sha256(
                segment_position_0,
                segment_velocity_0,
                segment_position_1,
                segment_velocity_1,
                segment_gm,
                removal.source_segment_step_years,
            )
            != removal.source_segment_sha256
        ):
            raise CUDAMasslessTracerContractError(
                "a removal event does not reproduce its source-segment commitment"
            )
        event_source_positions, event_source_velocities = (
            cubic_hermite_source_state(
                segment_position_0,
                segment_velocity_0,
                segment_position_1,
                segment_velocity_1,
                removal.step_fraction,
                removal.source_segment_step_years,
            )
        )
        if removal.event_type == "SOURCE_RADIUS":
            if (
                removal.source_index >= validated.source_count
                or state.source_ids[removal.source_index] != removal.source_id
            ):
                raise CUDAMasslessTracerContractError(
                    "a source-radius removal has inconsistent source identity"
                )
            expected_surface_radius = float(
                collision_radii_host[removal.source_index]
            )
            if expected_surface_radius <= 0.0:
                raise CUDAMasslessTracerContractError(
                    "a source-radius removal names a disabled surface"
                )
            surface_source_slot = removal.source_index
            expected_lifecycle = LIFECYCLE_REMOVED_SOURCE_RADIUS
        else:
            if state.primary_outer_boundary_AU is None:
                raise CUDAMasslessTracerContractError(
                    "an outer-boundary removal requires a configured boundary"
                )
            expected_surface_radius = float(state.primary_outer_boundary_AU)
            surface_source_slot = 0
            expected_lifecycle = LIFECYCLE_REMOVED_OUTER_BOUNDARY
        if (
            not np.array_equal(
                np.asarray(removal.surface_source_position_AU, dtype=np.float64),
                event_source_positions[surface_source_slot],
            )
            or not np.array_equal(
                np.asarray(
                    removal.surface_source_velocity_AU_per_year,
                    dtype=np.float64,
                ),
                event_source_velocities[surface_source_slot],
            )
            or not np.array_equal(
                np.asarray(removal.primary_source_position_AU, dtype=np.float64),
                event_source_positions[0],
            )
            or not np.array_equal(
                np.asarray(
                    removal.primary_source_velocity_AU_per_year,
                    dtype=np.float64,
                ),
                event_source_velocities[0],
            )
        ):
            raise CUDAMasslessTracerContractError(
                "a removal event source state differs from its Hermite segment"
            )
        scale = max(1.0, expected_surface_radius, removal.located_distance_AU)
        ledger_tolerance = max(
            removal.surface_tolerance_AU,
            64.0 * np.finfo(np.float64).eps * scale,
        )
        if abs(removal.surface_radius_AU - expected_surface_radius) > ledger_tolerance:
            raise CUDAMasslessTracerContractError(
                "a removal event surface differs from the configured policy"
            )
        expected_residual = abs(
            removal.located_distance_AU - removal.surface_radius_AU
        )
        if abs(removal.root_residual_AU - expected_residual) > ledger_tolerance:
            raise CUDAMasslessTracerContractError(
                "a removal event root residual is internally inconsistent"
            )
        event_position = np.asarray(removal.position_AU, dtype=np.float64)
        event_velocity = np.asarray(
            removal.velocity_AU_per_year, dtype=np.float64
        )
        if not np.array_equal(event_position, tracer_positions_host[slot]) or not np.array_equal(
            event_velocity, tracer_velocities_host[slot]
        ):
            raise CUDAMasslessTracerContractError(
                "a removed tracer state differs from its frozen event state"
            )
        event_minimum_distance = float(
            np.min(
                np.linalg.norm(
                    event_source_positions - event_position[None, :], axis=1
                )
            )
        )
        removal_event_minimum[
            (removal.global_step, removal.tracer_index)
        ] = event_minimum_distance
        surface_source_position = np.asarray(
            removal.surface_source_position_AU, dtype=np.float64
        )
        observed_surface_distance = float(
            np.linalg.norm(event_position - surface_source_position)
        )
        if (
            not math.isfinite(observed_surface_distance)
            or abs(observed_surface_distance - removal.located_distance_AU)
            > removal.surface_tolerance_AU
        ):
            raise CUDAMasslessTracerContractError(
                "a removal distance disagrees with its frozen event vectors"
            )
        coordinate_scale = max(
            1.0,
            removal.surface_radius_AU,
            removal.located_distance_AU,
            float(np.max(np.abs(source_positions_host))),
            float(np.max(np.abs(event_position))),
            float(np.max(np.abs(surface_source_position))),
            float(np.max(np.abs(removal.primary_source_position_AU))),
        )
        maximum_surface_tolerance = max(
            1.0e-11,
            4096.0 * np.finfo(np.float64).eps * coordinate_scale,
        )
        if removal.surface_tolerance_AU > maximum_surface_tolerance:
            raise CUDAMasslessTracerContractError(
                "a removal event claims an excessive surface tolerance"
            )
        primary_position = np.asarray(
            removal.primary_source_position_AU, dtype=np.float64
        )
        primary_velocity = np.asarray(
            removal.primary_source_velocity_AU_per_year, dtype=np.float64
        )
        primary_relative_position = event_position - primary_position
        primary_relative_velocity = event_velocity - primary_velocity
        primary_distance = float(np.linalg.norm(primary_relative_position))
        if not primary_distance > 0.0:
            raise CUDAMasslessTracerContractError(
                "a removal event reaches the primary point singularity"
            )
        recomputed_energy = float(
            0.5 * np.dot(primary_relative_velocity, primary_relative_velocity)
            - source_gm_host[0] / primary_distance
        )
        recomputed_radial_velocity = float(
            np.dot(primary_relative_position, primary_relative_velocity)
            / primary_distance
        )
        diagnostic_scale = max(
            1.0,
            abs(recomputed_energy),
            abs(removal.primary_relative_specific_energy_AU2_per_year2),
            abs(recomputed_radial_velocity),
            abs(removal.primary_relative_radial_velocity_AU_per_year),
        )
        diagnostic_tolerance = 4096.0 * np.finfo(np.float64).eps * diagnostic_scale
        if (
            abs(
                recomputed_energy
                - removal.primary_relative_specific_energy_AU2_per_year2
            )
            > diagnostic_tolerance
            or abs(
                recomputed_radial_velocity
                - removal.primary_relative_radial_velocity_AU_per_year
            )
            > diagnostic_tolerance
        ):
            raise CUDAMasslessTracerContractError(
                "a removal event's primary-relative diagnostics are inconsistent"
            )
        if (
            removal.event_type == "OUTER_BOUNDARY"
            and recomputed_radial_velocity
            <= OUTWARD_EVENT_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR
        ):
            raise CUDAMasslessTracerContractError(
                "an outer-boundary removal is not outward-moving"
            )
        if int(lifecycle_host[slot]) != expected_lifecycle:
            raise CUDAMasslessTracerContractError(
                "a removal event contradicts tracer_lifecycle"
            )
        removal_for_slot[slot] = removal
    for slot, lifecycle_value in enumerate(lifecycle_host):
        is_removed = int(lifecycle_value) != LIFECYCLE_ACTIVE
        if is_removed != (slot in removal_for_slot):
            raise CUDAMasslessTracerContractError(
                "each removed tracer must have exactly one terminal event"
            )
    terminal_routes = {
        (row.global_step, row.tracer_index): row
        for row in records
        if row.outcome != "REJOINED"
    }
    if len(terminal_routes) != len(removal_records):
        raise CUDAMasslessTracerContractError(
            "terminal route outcomes and removal events disagree"
        )
    terminal_step_for_tracer = {
        row.tracer_index: row.global_step
        for row in terminal_routes.values()
    }
    for row in records:
        terminal_step = terminal_step_for_tracer.get(row.tracer_index)
        if terminal_step is not None and row.global_step > terminal_step:
            raise CUDAMasslessTracerContractError(
                "an encounter route cannot follow a tracer's terminal route"
            )
    for removal in removal_records:
        route = terminal_routes.get((removal.global_step, removal.tracer_index))
        expected_outcome = (
            "REMOVED_SOURCE_RADIUS"
            if removal.event_type == "SOURCE_RADIUS"
            else "REMOVED_OUTER_BOUNDARY"
        )
        if route is None or route.outcome != expected_outcome:
            raise CUDAMasslessTracerContractError(
                "terminal route outcomes and removal events disagree"
            )
        if (
            abs(route.committed_step_fraction - removal.step_fraction) > 8.0e-12
            or route.source_segment_sha256 != removal.source_segment_sha256
            or route.endpoint_unbound
            != (
                removal.primary_relative_specific_energy_AU2_per_year2 >= 0.0
            )
        ):
            raise CUDAMasslessTracerContractError(
                "a terminal route is not exactly linked to its removal event"
            )
        event_minimum = removal_event_minimum[
            (removal.global_step, removal.tracer_index)
        ]
        if (
            route.minimum_committed_stage_distance_AU
            > event_minimum + removal.surface_tolerance_AU
        ):
            raise CUDAMasslessTracerContractError(
                "a terminal route minimum excludes its frozen event point"
            )
    _validate_active_tracer_domain(backend, state, validated, lifecycle)
    return _ValidatedContinuation(
        minimum_source_distance_AU=minimum,
        ever_unbound=ever_unbound,
        encounter_route_counts_host=route_counts_host,
        tracer_lifecycle=lifecycle,
        encounter_routes=records,
        removal_events=removal_records,
    )


def _kernel(backend: ArrayBackend) -> Any:
    del backend
    raise CUDAMasslessTracerContractError(
        "the legacy block-local CUDA kernel is retired and cannot be compiled"
    )


def _path_kernels(backend: ArrayBackend) -> tuple[Any, Any]:
    key = ("shared-source-path", int(backend.device_index or 0), KERNEL_SOURCE_SHA256)
    cached = _KERNEL_CACHE.get(key)
    if cached is None or not isinstance(cached, tuple):
        source_kernel = backend.xp.RawKernel(
            SOURCE_PATH_CUDA_SOURCE,
            "jx_build_massive_source_path",
            options=CUDA_OPTIONS,
        )
        tracer_kernel = backend.xp.RawKernel(
            TRACER_PATH_CUDA_SOURCE,
            "jx_advance_tracers_on_source_path",
            options=CUDA_OPTIONS,
        )
        cached = (source_kernel, tracer_kernel)
        _KERNEL_CACHE[key] = cached
    return cached


def _build_source_path(
    backend: ArrayBackend,
    kernel: Any,
    source_positions: Any,
    source_velocities: Any,
    source_gm: Any,
    *,
    step_count: int,
    step: float,
) -> _SourcePath:
    cp = backend.xp
    source_count = int(source_positions.shape[0])
    shape = (step_count + 1, source_count, 3)
    positions = cp.empty(shape, dtype=cp.float64)
    velocities = cp.empty(shape, dtype=cp.float64)
    accelerations = cp.empty(shape, dtype=cp.float64)
    status = cp.empty(1, dtype=cp.int32)
    kernel(
        (1,),
        (32,),
        (
            source_positions,
            source_velocities,
            source_gm,
            cp.int32(source_count),
            cp.int64(step_count),
            cp.float64(step),
            positions,
            velocities,
            accelerations,
            status,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    code = int(cp.asnumpy(status[0]))
    if code != STATUS_SUCCESS:
        raise CUDAMasslessTracerIntegrationError(
            f"massive source path failed with {STATUS_NAMES.get(code, code)}"
        )
    if not _device_truth(
        backend,
        cp.all(cp.isfinite(positions))
        & cp.all(cp.isfinite(velocities))
        & cp.all(cp.isfinite(accelerations)),
        "massive source path finite-state validation",
    ):
        raise CUDAMasslessTracerIntegrationError(
            "massive source path produced a nonfinite state"
        )
    return _SourcePath(positions, velocities, accelerations)


def _launch_on_source_path(
    backend: ArrayBackend,
    kernel: Any,
    source_path: _SourcePath,
    source_gm: Any,
    encounter_radii: Any,
    collision_radii: Any,
    acceleration_bounds: Any,
    tracer_positions: Any,
    tracer_velocities: Any,
    tracer_lifecycle: Any,
    *,
    step: float,
    primary_outer_boundary_AU: float,
    maximum_primary_dynamical_phase: float,
    threads: int,
) -> _PathLaunchResult:
    cp = backend.xp
    source_count = int(source_path.positions.shape[1])
    step_count = int(source_path.positions.shape[0] - 1)
    tracer_count = int(tracer_positions.shape[0])
    output_position = cp.empty_like(tracer_positions)
    output_velocity = cp.empty_like(tracer_velocities)
    minimum_distance = cp.empty(tracer_count, dtype=cp.float64)
    ever_unbound = cp.empty(tracer_count, dtype=cp.int32)
    status = cp.empty(tracer_count, dtype=cp.int32)
    reason = cp.empty(tracer_count, dtype=cp.int32)
    trigger_source = cp.empty(tracer_count, dtype=cp.int32)
    completed_steps = cp.empty(tracer_count, dtype=cp.int64)
    if tracer_count:
        blocks = math.ceil(tracer_count / threads)
        kernel(
            (blocks,),
            (threads,),
            (
                source_path.positions,
                source_path.velocities,
                source_path.accelerations,
                source_gm,
                encounter_radii,
                collision_radii,
                acceleration_bounds,
                tracer_positions,
                tracer_velocities,
                tracer_lifecycle,
                cp.int32(source_count),
                cp.int32(tracer_count),
                cp.int64(step_count),
                cp.float64(step),
                cp.float64(primary_outer_boundary_AU),
                cp.float64(maximum_primary_dynamical_phase),
                output_position,
                output_velocity,
                minimum_distance,
                ever_unbound,
                status,
                reason,
                trigger_source,
                completed_steps,
            ),
        )
        cp.cuda.runtime.deviceSynchronize()
        fatal = (
            (status != STATUS_SUCCESS)
            & (status != STATUS_ENCOUNTER_ROUTE)
            & (status != STATUS_REMOVED_FROZEN)
        )
        if _device_truth(backend, cp.any(fatal), "CUDA path fatal-status validation"):
            lane = int(cp.asnumpy(cp.flatnonzero(fatal)[0]))
            code = int(cp.asnumpy(status[lane]))
            raise CUDAMasslessTracerIntegrationError(
                f"CUDA tracer lane {lane} failed with {STATUS_NAMES.get(code, code)}"
            )
    return _PathLaunchResult(
        output_position,
        output_velocity,
        minimum_distance,
        ever_unbound,
        status,
        reason,
        trigger_source,
        completed_steps,
    )


def _cpu_runtime_fingerprint() -> Mapping[str, Any]:
    fields: dict[str, str] = {}
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            first_processor = cpuinfo.read_text(encoding="utf-8").split("\n\n", 1)[0]
        except (OSError, UnicodeError):
            first_processor = ""
        wanted = {
            "vendor_id",
            "cpu family",
            "model",
            "model name",
            "stepping",
            "microcode",
            "flags",
        }
        for line in first_processor.splitlines():
            key, separator, value = line.partition(":")
            normalized = key.strip()
            if separator and normalized in wanted:
                fields[normalized] = " ".join(value.split())
    if not fields:
        fields = {
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    encoded = json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return MappingProxyType(
        {
            "fields": fields,
            "sha256": hashlib.sha256(
                b"jxplanetx.cpu-runtime-fingerprint.v1\x00" + encoded
            ).hexdigest(),
        }
    )


def cuda_massless_tracer_runtime_identity(backend: BackendSpec) -> Mapping[str, Any]:
    """Return the exact GPU and CPU-route runtime identity for checkpoints."""

    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise CUDAMasslessTracerContractError(
            "runtime identity requires an exact CuPy BackendSpec"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    try:
        import cupy.cuda.compiler as cupy_cuda_compiler
        import scipy
        import scipy.integrate._ivp.base as scipy_ivp_base
        import scipy.integrate._ivp.common as scipy_ivp_common
        import scipy.integrate._ivp.dop853_coefficients as scipy_dop853_coefficients
        import scipy.integrate._ivp.rk as scipy_ivp_rk
    except (ImportError, OSError) as exc:  # pragma: no cover - environment failure
        raise CUDAMasslessTracerContractError(
            "SciPy is required by the declared encounter-route contract"
        ) from exc
    with resolved.activate():
        properties = cp.cuda.runtime.getDeviceProperties(resolved.device_index or 0)
        runtime_binary_paths = {
            "python_executable": Path(sys.executable),
            "numpy_multiarray_umath": Path(np._core._multiarray_umath.__file__),
            "cupy_cuda_compiler": Path(cupy_cuda_compiler.__file__),
            "cupy_cuda_runtime_extension": Path(cp.cuda.runtime.__file__),
            "cupy_nvrtc_extension": Path(cp.cuda.nvrtc.__file__),
            "scipy_ivp_base": Path(scipy_ivp_base.__file__),
            "scipy_ivp_common": Path(scipy_ivp_common.__file__),
            "scipy_dop853_coefficients": Path(
                scipy_dop853_coefficients.__file__
            ),
            "scipy_ivp_rk": Path(scipy_ivp_rk.__file__),
        }
        runtime_binary_sha256 = {
            label: _sha256_file(path)
            for label, path in runtime_binary_paths.items()
        }
        numpy_build_sha256 = hashlib.sha256(
            json.dumps(
                np.__config__.show(mode="dicts"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        scipy_build_sha256 = hashlib.sha256(
            json.dumps(
                scipy.show_config(mode="dicts"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()
        cpu_fingerprint = _cpu_runtime_fingerprint()
        value = {
            "method_id": METHOD_ID,
            "kernel_id": KERNEL_ID,
            "array_backend_id": backend.backend_id,
            "array_backend_device": resolved.device,
            "kernel_source_sha256": KERNEL_SOURCE_SHA256,
            "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
            "event_locator_source_sha256": EVENT_LOCATOR_SOURCE_SHA256,
            "cuda_options": list(CUDA_OPTIONS),
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(
                cp.cuda.get_local_runtime_version()
            ),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "nvrtc_version": [int(value) for value in cp.cuda.nvrtc.getVersion()],
            "device_index": int(resolved.device_index or 0),
            "device_name": properties["name"].decode("ascii"),
            "device_uuid": bytes(properties["uuid"]).hex(),
            "device_compute_capability": (
                f"{int(properties['major'])}.{int(properties['minor'])}"
            ),
            "device_pci_domain_bus_device": (
                f"{int(properties['pciDomainID']):04x}:"
                f"{int(properties['pciBusID']):02x}:"
                f"{int(properties['pciDeviceID']):02x}"
            ),
            "device_total_global_memory_bytes": int(properties["totalGlobalMem"]),
            "device_multiprocessor_count": int(properties["multiProcessorCount"]),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "runtime_binary_sha256": runtime_binary_sha256,
            "numpy_version": np.__version__,
            "numpy_build_configuration_sha256": numpy_build_sha256,
            "scipy_version": scipy.__version__,
            "scipy_build_configuration_sha256": scipy_build_sha256,
            "cpu_system": platform.system(),
            "cpu_release": platform.release(),
            "cpu_machine": platform.machine(),
            "cpu_processor": platform.processor(),
            "cpu_fingerprint": dict(cpu_fingerprint),
            "libc": list(platform.libc_ver()),
            "encounter_route_method": (
                "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2"
            ),
            "determinism_scope": DETERMINISM_SCOPE,
        }
    return MappingProxyType(value)


def _launch(
    backend: ArrayBackend,
    kernel: Any,
    source_positions: Any,
    source_velocities: Any,
    source_gm: Any,
    encounter_radii: Any,
    acceleration_bounds: Any,
    tracer_positions: Any,
    tracer_velocities: Any,
    *,
    step_count: int,
    step: float,
    threads: int,
) -> _LaunchResult:
    cp = backend.xp
    source_count = int(source_positions.shape[0])
    tracer_count = int(tracer_positions.shape[0])
    blocks = max(1, math.ceil(tracer_count / threads))
    output_source_position = cp.empty((blocks, source_count, 3), dtype=cp.float64)
    output_source_velocity = cp.empty_like(output_source_position)
    output_tracer_position = cp.empty_like(tracer_positions)
    output_tracer_velocity = cp.empty_like(tracer_velocities)
    minimum_distance = cp.empty(tracer_count, dtype=cp.float64)
    ever_unbound = cp.empty(tracer_count, dtype=cp.int32)
    status = cp.empty(tracer_count, dtype=cp.int32)
    reason = cp.empty(tracer_count, dtype=cp.int32)
    trigger_source = cp.empty(tracer_count, dtype=cp.int32)
    completed_steps = cp.empty(tracer_count, dtype=cp.int64)
    trigger_shape = (tracer_count, source_count, 3)
    trigger_p0 = cp.empty(trigger_shape, dtype=cp.float64)
    trigger_v0 = cp.empty(trigger_shape, dtype=cp.float64)
    trigger_p1 = cp.empty(trigger_shape, dtype=cp.float64)
    trigger_v1 = cp.empty(trigger_shape, dtype=cp.float64)
    kernel(
        (blocks,),
        (threads,),
        (
            source_positions,
            source_velocities,
            source_gm,
            encounter_radii,
            acceleration_bounds,
            tracer_positions,
            tracer_velocities,
            cp.int32(source_count),
            cp.int32(tracer_count),
            cp.int64(step_count),
            cp.float64(step),
            output_source_position,
            output_source_velocity,
            output_tracer_position,
            output_tracer_velocity,
            minimum_distance,
            ever_unbound,
            status,
            reason,
            trigger_source,
            completed_steps,
            trigger_p0,
            trigger_v0,
            trigger_p1,
            trigger_v1,
        ),
    )
    cp.cuda.runtime.deviceSynchronize()
    reference_p = output_source_position[0]
    reference_v = output_source_velocity[0]
    if not _device_truth(
        backend,
        cp.all(output_source_position == reference_p[None, :, :]),
        "source position no-backreaction invariant",
    ) or not _device_truth(
        backend,
        cp.all(output_source_velocity == reference_v[None, :, :]),
        "source velocity no-backreaction invariant",
    ):
        raise CUDAMasslessTracerIntegrationError(
            "source path differs between CUDA tracer blocks"
        )
    if tracer_count:
        fatal = (status != STATUS_SUCCESS) & (status != STATUS_ENCOUNTER_ROUTE)
        if _device_truth(backend, cp.any(fatal), "CUDA fatal-status validation"):
            lane = int(cp.asnumpy(cp.flatnonzero(fatal)[0]))
            code = int(cp.asnumpy(status[lane]))
            raise CUDAMasslessTracerIntegrationError(
                f"CUDA tracer lane {lane} failed with {STATUS_NAMES.get(code, code)}"
            )
    return _LaunchResult(
        reference_p.copy(),
        reference_v.copy(),
        output_tracer_position,
        output_tracer_velocity,
        minimum_distance,
        ever_unbound,
        status,
        reason,
        trigger_source,
        completed_steps,
        trigger_p0,
        trigger_v0,
        trigger_p1,
        trigger_v1,
    )


def _hermite_position(
    position_0: np.ndarray,
    velocity_0: np.ndarray,
    position_1: np.ndarray,
    velocity_1: np.ndarray,
    fraction: float,
    step: float,
) -> np.ndarray:
    s = fraction
    s2 = s * s
    s3 = s2 * s
    return (
        (2.0 * s3 - 3.0 * s2 + 1.0) * position_0
        + (s3 - 2.0 * s2 + s) * step * velocity_0
        + (-2.0 * s3 + 3.0 * s2) * position_1
        + (s3 - s2) * step * velocity_1
    )


def _unbound_from_primary_host(
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    primary_position: np.ndarray,
    primary_velocity: np.ndarray,
    primary_gm: float,
) -> bool:
    """Apply the CUDA diagnostic's node-sampled primary-energy definition."""

    relative_position = np.asarray(tracer_position) - np.asarray(primary_position)
    relative_velocity = np.asarray(tracer_velocity) - np.asarray(primary_velocity)
    distance2 = float(np.dot(relative_position, relative_position))
    speed2 = float(np.dot(relative_velocity, relative_velocity))
    if not distance2 > 0.0 or not math.isfinite(distance2 + speed2):
        raise EncounterRouteError(
            "encounter route endpoint cannot be classified against the primary"
        )
    energy = 0.5 * speed2 - float(primary_gm) / math.sqrt(distance2)
    if not math.isfinite(energy):
        raise EncounterRouteError(
            "encounter route endpoint energy is nonfinite"
        )
    return energy >= 0.0


def _source_segment_sha256(
    position_0: np.ndarray,
    velocity_0: np.ndarray,
    position_1: np.ndarray,
    velocity_1: np.ndarray,
    gm: np.ndarray,
    step: float,
) -> str:
    digest = hashlib.sha256(SOURCE_SEGMENT_DOMAIN)
    for label, value in (
        ("position_0", position_0),
        ("velocity_0", velocity_0),
        ("position_1", position_1),
        ("velocity_1", velocity_1),
        ("gm", gm),
    ):
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(label.encode("ascii") + b"\x00")
        digest.update(str(array.shape).encode("ascii") + b"\x00")
        digest.update(array.tobytes())
    digest.update(np.asarray([step], dtype="<f8").tobytes())
    return digest.hexdigest()


def _redo_encounter_step(
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    source_gm: np.ndarray,
    *,
    step: float,
    rtol: float,
    atol: float,
    maximum_substep_fraction: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Redo exactly one untouched outer step against its source segment."""

    try:
        from scipy.integrate import DOP853
    except (ImportError, OSError) as exc:  # pragma: no cover - optional environment
        raise EncounterRouteError(
            "SciPy is required only when the encounter route is exercised"
        ) from exc

    y0 = np.concatenate(
        (
            np.asarray(tracer_position, dtype=np.float64),
            np.asarray(tracer_velocity, dtype=np.float64),
        )
    )

    minimum_evaluated_stage_distance = math.inf

    def derivative(epoch: float, state: np.ndarray) -> np.ndarray:
        nonlocal minimum_evaluated_stage_distance
        fraction = epoch / step
        sources = _hermite_position(
            source_position_0,
            source_velocity_0,
            source_position_1,
            source_velocity_1,
            fraction,
            step,
        )
        delta = sources - state[:3]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        if (
            not np.isfinite(distance2).all()
            or bool(np.any(distance2 <= 0.0))
        ):
            raise EncounterRouteError(
                "encounter route reached a point-mass singularity"
            )
        minimum_evaluated_stage_distance = min(
            minimum_evaluated_stage_distance, float(np.sqrt(np.min(distance2)))
        )
        acceleration = np.sum(
            source_gm[:, None]
            * delta
            / (distance2 * np.sqrt(distance2))[:, None],
            axis=0,
        )
        return np.concatenate((state[3:], acceleration))

    solver = DOP853(
        derivative,
        0.0,
        y0,
        step,
        rtol=rtol,
        atol=atol,
        max_step=abs(step) * maximum_substep_fraction,
    )
    accepted = 0
    while solver.status == "running":
        solver.step()
        accepted += 1
    if solver.status != "finished" or not np.isfinite(solver.y).all():
        raise EncounterRouteError(
            f"DOP853 encounter route failed: {solver.status}"
        )
    # SciPy's low-level solver exposes function evaluations but not a rejected
    # step counter.  Preserve the observable ledger and mark rejections unknown
    # with -1 rather than inventing a value.
    return (
        solver.y[:3].copy(),
        solver.y[3:].copy(),
        accepted,
        -1,
        minimum_evaluated_stage_distance,
    )


def _copy_state_with_endpoint(
    state: CUDAMasslessTracerState,
    *,
    source_positions: Any,
    source_velocities: Any,
    tracer_positions: Any,
    tracer_velocities: Any,
    minimum_source_distance_AU: Any | None = None,
    ever_unbound: Any | None = None,
    encounter_route_counts: Any | None = None,
    tracer_lifecycle: Any | None = None,
    prior_encounter_routes: tuple[Any, ...] = (),
    prior_removal_events: tuple[Any, ...] = (),
    step_years: float,
    step_count: int,
) -> CUDAMasslessTracerState:
    return CUDAMasslessTracerState(
        snapshot_id=state.snapshot_id,
        epoch=state.epoch + step_years * step_count,
        global_step=state.global_step + step_count,
        time_scale=state.time_scale,
        frame=state.frame,
        origin=state.origin,
        axes=state.axes,
        length_unit=state.length_unit,
        time_unit=state.time_unit,
        unit_system_id=state.unit_system_id,
        source_ids=state.source_ids,
        primary_source_id=state.primary_source_id,
        source_positions=source_positions,
        source_velocities=source_velocities,
        source_gm=state.source_gm,
        source_encounter_radii=state.source_encounter_radii,
        source_collision_radii=state.source_collision_radii,
        relative_acceleration_bounds=state.relative_acceleration_bounds,
        tracer_positions=tracer_positions,
        tracer_velocities=tracer_velocities,
        tracer_indices=state.tracer_indices,
        provenance=state.provenance,
        force_schedule_sha256=state.force_schedule_sha256,
        matched_schedule_set_sha256=state.matched_schedule_set_sha256,
        primary_outer_boundary_AU=state.primary_outer_boundary_AU,
        lattice_step_years=(
            state.lattice_step_years
            if state.lattice_step_years is not None
            else step_years
        ),
        minimum_source_distance_AU=minimum_source_distance_AU,
        ever_unbound=ever_unbound,
        encounter_route_counts=encounter_route_counts,
        tracer_lifecycle=tracer_lifecycle,
        prior_encounter_routes=prior_encounter_routes,
        prior_removal_events=prior_removal_events,
    )


def _advance_block_local_reference(
    state: CUDAMasslessTracerState,
    spec: FixedStepCUDAMasslessTracerSpec,
) -> CUDAMasslessTracerResult:
    """Retired pre-M2 block-local reference path.

    It does not implement terminal-surface lifecycle semantics and therefore
    cannot be used with the v2 contract.  The implementation body is retained
    temporarily only as a historical comparison aid and is unreachable.
    """

    raise CUDAMasslessTracerContractError(
        "the block-local reference path is retired for the v2 lifecycle contract"
    )

    if type(spec) is not FixedStepCUDAMasslessTracerSpec:
        raise CUDAMasslessTracerContractError(
            "spec must be an exact FixedStepCUDAMasslessTracerSpec"
        )
    if (
        state.lattice_step_years is not None
        and state.lattice_step_years != spec.step_years
    ):
        raise CUDAMasslessTracerContractError(
            "step_years cannot change after the integration lattice is established"
        )
    backend = resolve_backend(spec.backend)
    cp = backend.xp
    with backend.activate():
        validated = _validate_state(backend, state)
        continuation = _validate_continuation(backend, state, validated)
        input_state_sha256 = _validated_state_sha256(
            backend, state, validated, continuation
        )
        kernel = _kernel(backend)
        main = _launch(
            backend,
            kernel,
            validated.source_positions,
            validated.source_velocities,
            validated.source_gm,
            validated.source_encounter_radii,
            validated.relative_acceleration_bounds,
            validated.tracer_positions,
            validated.tracer_velocities,
            step_count=spec.step_count,
            step=spec.step_years,
            threads=spec.threads_per_block,
        )
        final_position = cp.empty_like(validated.tracer_positions)
        final_velocity = cp.empty_like(validated.tracer_velocities)
        minimum_distance = cp.full(validated.tracer_count, cp.inf, dtype=cp.float64)
        ever_unbound = cp.zeros(validated.tracer_count, dtype=cp.bool_)
        route_counts_host = np.zeros(validated.tracer_count, dtype=np.int64)
        records: list[TracerEncounterRouteRecord] = []
        expected_source_position = main.source_positions
        expected_source_velocity = main.source_velocities
        gm_host: np.ndarray | None = None

        def commit_launch(
            launch: _LaunchResult,
            lane_slots: Any,
            tracer_indices: Any,
            *,
            absolute_start_step: int,
            steps_in_launch: int,
        ) -> None:
            nonlocal gm_host
            if not _device_truth(
                backend,
                cp.array_equal(launch.source_positions, expected_source_position),
                "routed source endpoint position identity",
            ) or not _device_truth(
                backend,
                cp.array_equal(launch.source_velocities, expected_source_velocity),
                "routed source endpoint velocity identity",
            ):
                raise EncounterRouteError(
                    "a routed lane did not rejoin the bitwise-identical source endpoint"
                )

            current_minimum = minimum_distance[lane_slots]
            minimum_distance[lane_slots] = cp.minimum(
                current_minimum, launch.minimum_source_distance
            )
            ever_unbound[lane_slots] = ever_unbound[lane_slots] | (
                launch.ever_unbound != 0
            )
            quiet = launch.status == STATUS_SUCCESS
            if _device_truth(backend, cp.any(quiet), "quiet-lane selection"):
                quiet_local = cp.flatnonzero(quiet)
                quiet_slots = lane_slots[quiet_local]
                final_position[quiet_slots] = launch.tracer_positions[quiet_local]
                final_velocity[quiet_slots] = launch.tracer_velocities[quiet_local]

            guarded_local_host = cp.asnumpy(
                cp.flatnonzero(launch.status == STATUS_ENCOUNTER_ROUTE)
            )
            for local_value in guarded_local_host:
                local = int(local_value)
                slot = int(cp.asnumpy(lane_slots[local]))
                route_counts_host[slot] += 1
                if (
                    route_counts_host[slot]
                    > spec.maximum_encounter_routes_per_tracer
                ):
                    raise EncounterRouteError(
                        f"tracer {slot} exceeded the encounter-route budget"
                    )
                completed = int(cp.asnumpy(launch.completed_steps[local]))
                source_index = int(cp.asnumpy(launch.trigger_source[local]))
                reason_code = int(cp.asnumpy(launch.reason[local]))
                if not 0 <= source_index < validated.source_count:
                    raise EncounterRouteError("CUDA returned an invalid trigger source")
                if reason_code not in REASON_NAMES or reason_code == REASON_NONE:
                    raise EncounterRouteError("CUDA returned an invalid trigger reason")
                p0 = cp.asnumpy(launch.trigger_source_position_0[local])
                v0 = cp.asnumpy(launch.trigger_source_velocity_0[local])
                p1 = cp.asnumpy(launch.trigger_source_position_1[local])
                v1 = cp.asnumpy(launch.trigger_source_velocity_1[local])
                tp0 = cp.asnumpy(launch.tracer_positions[local])
                tv0 = cp.asnumpy(launch.tracer_velocities[local])
                if gm_host is None:
                    gm_host = cp.asnumpy(validated.source_gm)
                tp1, tv1, accepted, rejected, routed_minimum = _redo_encounter_step(
                    tp0,
                    tv0,
                    p0,
                    v0,
                    p1,
                    v1,
                    gm_host,
                    step=spec.step_years,
                    rtol=spec.encounter_route_rtol,
                    atol=spec.encounter_route_atol,
                    maximum_substep_fraction=(
                        spec.encounter_route_maximum_substep_fraction
                    ),
                )
                tracer_index = int(cp.asnumpy(tracer_indices[local]))
                segment_sha = _source_segment_sha256(
                    p0, v0, p1, v1, gm_host, spec.step_years
                )
                distance = routed_minimum
                endpoint_unbound = _unbound_from_primary_host(
                    tp1, tv1, p1[0], v1[0], float(gm_host[0])
                )
                records.append(
                    TracerEncounterRouteRecord(
                        tracer_index=tracer_index,
                        source_index=source_index,
                        source_id=state.source_ids[source_index],
                        global_step=absolute_start_step + completed,
                        reason=REASON_NAMES[reason_code],
                        source_segment_sha256=segment_sha,
                        cpu_accepted_steps=accepted,
                        cpu_rejected_steps=rejected,
                        minimum_evaluated_stage_distance_AU=distance,
                        minimum_committed_stage_distance_AU=distance,
                        endpoint_unbound=endpoint_unbound,
                    )
                )
                minimum_distance[slot] = cp.minimum(
                    minimum_distance[slot], cp.float64(distance)
                )
                ever_unbound[slot] = ever_unbound[slot] | endpoint_unbound
                remaining = steps_in_launch - completed - 1
                if remaining < 0:
                    raise EncounterRouteError("negative routed-step remainder")
                if remaining == 0:
                    if not _device_truth(
                        backend,
                        cp.array_equal(cp.asarray(p1), expected_source_position),
                        "terminal routed source position identity",
                    ) or not _device_truth(
                        backend,
                        cp.array_equal(cp.asarray(v1), expected_source_velocity),
                        "terminal routed source velocity identity",
                    ):
                        raise EncounterRouteError(
                            "terminal routed lane missed the common source endpoint"
                        )
                    final_position[slot] = cp.asarray(tp1)
                    final_velocity[slot] = cp.asarray(tv1)
                    continue
                next_position = cp.asarray(tp1[None, :])
                next_velocity = cp.asarray(tv1[None, :])
                resumed = _launch(
                    backend,
                    kernel,
                    cp.asarray(p1),
                    cp.asarray(v1),
                    validated.source_gm,
                    validated.source_encounter_radii,
                    validated.relative_acceleration_bounds,
                    next_position,
                    next_velocity,
                    step_count=remaining,
                    step=spec.step_years,
                    threads=spec.threads_per_block,
                )
                commit_launch(
                    resumed,
                    cp.asarray([slot], dtype=cp.int64),
                    cp.asarray([tracer_index], dtype=cp.uint64),
                    absolute_start_step=absolute_start_step + completed + 1,
                    steps_in_launch=remaining,
                )

        if validated.tracer_count:
            commit_launch(
                main,
                cp.arange(validated.tracer_count, dtype=cp.int64),
                validated.tracer_indices,
                absolute_start_step=state.global_step,
                steps_in_launch=spec.step_count,
            )
        endpoint_state = _copy_state_with_endpoint(
            state,
            source_positions=expected_source_position,
            source_velocities=expected_source_velocity,
            tracer_positions=final_position,
            tracer_velocities=final_velocity,
            minimum_source_distance_AU=minimum_distance,
            ever_unbound=ever_unbound,
            encounter_route_counts=cp.asarray(route_counts_host),
            tracer_lifecycle=cp.zeros(
                validated.tracer_count, dtype=cp.int32
            ),
            prior_encounter_routes=tuple(
                sorted(records, key=lambda row: (row.global_step, row.tracer_index))
            ),
            step_years=spec.step_years,
            step_count=spec.step_count,
        )
        runtime = cuda_massless_tracer_runtime_identity(spec.backend)
        return CUDAMasslessTracerResult(
            state=endpoint_state,
            minimum_source_distance_AU=minimum_distance,
            ever_unbound=ever_unbound,
            encounter_route_counts=cp.asarray(route_counts_host),
            tracer_lifecycle=cp.zeros(
                validated.tracer_count, dtype=cp.int32
            ),
            encounter_routes=tuple(
                sorted(records, key=lambda row: (row.global_step, row.tracer_index))
            ),
            removal_events=(),
            runtime_identity=runtime,
            input_state_sha256=input_state_sha256,
        )


def advance_cuda_massless_tracers(
    state: CUDAMasslessTracerState,
    spec: FixedStepCUDAMasslessTracerSpec,
) -> CUDAMasslessTracerResult:
    """Advance tracers over one precomputed immutable source path per chunk.

    This is the production path for the experimental v2 module.  Massive
    sources are integrated exactly once per chunk, not once per tracer block.
    A guarded tracer lane stops at its untouched outer node, its complete step
    is redone by DOP853 against two nodes of that same source path, and only a
    successfully synchronized endpoint is committed.
    """

    if type(spec) is not FixedStepCUDAMasslessTracerSpec:
        raise CUDAMasslessTracerContractError(
            "spec must be an exact FixedStepCUDAMasslessTracerSpec"
        )
    if (
        state.lattice_step_years is not None
        and state.lattice_step_years != spec.step_years
    ):
        raise CUDAMasslessTracerContractError(
            "step_years cannot change after the integration lattice is established"
        )
    if not math.isfinite(state.epoch + spec.step_years * spec.step_count):
        raise CUDAMasslessTracerContractError(
            "the requested segment endpoint epoch is nonfinite"
        )
    backend = resolve_backend(spec.backend)
    cp = backend.xp
    with backend.activate():
        validated = _validate_state(backend, state)
        source_kernel, tracer_kernel = _path_kernels(backend)
        tracer_count = validated.tracer_count

        continuation = _validate_continuation(backend, state, validated)
        input_state_sha256 = _validated_state_sha256(
            backend, state, validated, continuation
        )
        cumulative_minimum = continuation.minimum_source_distance_AU
        cumulative_unbound = continuation.ever_unbound
        route_counts_host = continuation.encounter_route_counts_host
        cumulative_lifecycle = continuation.tracer_lifecycle
        records = list(continuation.encounter_routes)
        removal_records = list(continuation.removal_events)
        if bool(np.any(route_counts_host > spec.maximum_encounter_routes_per_tracer)):
            raise CUDAMasslessTracerContractError(
                "a cumulative encounter-route count exceeds the declared budget"
            )
        if len(records) > spec.maximum_total_encounter_routes:
            raise CUDAMasslessTracerContractError(
                "the cumulative encounter-route ledger exceeds the declared total budget"
            )

        current_source_position = validated.source_positions.copy()
        current_source_velocity = validated.source_velocities.copy()
        current_tracer_position = validated.tracer_positions.copy()
        current_tracer_velocity = validated.tracer_velocities.copy()
        processed_steps = 0
        gm_host: np.ndarray | None = None

        while processed_steps < spec.step_count:
            chunk_steps = min(
                spec.source_path_steps_per_chunk,
                spec.step_count - processed_steps,
            )
            source_path = _build_source_path(
                backend,
                source_kernel,
                current_source_position,
                current_source_velocity,
                validated.source_gm,
                step_count=chunk_steps,
                step=spec.step_years,
            )
            launch = _launch_on_source_path(
                backend,
                tracer_kernel,
                source_path,
                validated.source_gm,
                validated.source_encounter_radii,
                validated.source_collision_radii,
                validated.relative_acceleration_bounds,
                current_tracer_position,
                current_tracer_velocity,
                cumulative_lifecycle,
                step=spec.step_years,
                primary_outer_boundary_AU=state.primary_outer_boundary_AU,
                maximum_primary_dynamical_phase=(
                    spec.maximum_primary_dynamical_phase
                ),
                threads=spec.threads_per_block,
            )
            chunk_position = current_tracer_position.copy()
            chunk_velocity = current_tracer_velocity.copy()
            pending_launches: deque[tuple[Any, ...]] = deque()

            def commit_path_launch(
                path_launch: _PathLaunchResult,
                path: _SourcePath,
                lane_slots: Any,
                tracer_indices: Any,
                *,
                absolute_start_step: int,
            ) -> None:
                nonlocal gm_host
                local_steps = int(path.positions.shape[0] - 1)
                cumulative_minimum[lane_slots] = cp.minimum(
                    cumulative_minimum[lane_slots],
                    path_launch.minimum_source_distance,
                )
                cumulative_unbound[lane_slots] = cumulative_unbound[lane_slots] | (
                    path_launch.ever_unbound != 0
                )
                frozen = path_launch.status == STATUS_REMOVED_FROZEN
                if _device_truth(
                    backend, cp.any(frozen), "frozen path-lane selection"
                ):
                    frozen_local = cp.flatnonzero(frozen)
                    frozen_slots = lane_slots[frozen_local]
                    chunk_position[frozen_slots] = path_launch.tracer_positions[
                        frozen_local
                    ]
                    chunk_velocity[frozen_slots] = path_launch.tracer_velocities[
                        frozen_local
                    ]
                quiet = path_launch.status == STATUS_SUCCESS
                if _device_truth(backend, cp.any(quiet), "quiet path-lane selection"):
                    quiet_local = cp.flatnonzero(quiet)
                    quiet_slots = lane_slots[quiet_local]
                    chunk_position[quiet_slots] = path_launch.tracer_positions[
                        quiet_local
                    ]
                    chunk_velocity[quiet_slots] = path_launch.tracer_velocities[
                        quiet_local
                    ]
                guarded_local_host = cp.asnumpy(
                    cp.flatnonzero(
                        path_launch.status == STATUS_ENCOUNTER_ROUTE
                    )
                )
                for local_value in guarded_local_host:
                    local = int(local_value)
                    slot = int(cp.asnumpy(lane_slots[local]))
                    if len(records) >= spec.maximum_total_encounter_routes:
                        raise EncounterRouteError(
                            "integration exceeded the cumulative total encounter-route budget"
                        )
                    route_counts_host[slot] += 1
                    if (
                        route_counts_host[slot]
                        > spec.maximum_encounter_routes_per_tracer
                    ):
                        raise EncounterRouteError(
                            f"tracer {slot} exceeded the encounter-route budget"
                        )
                    completed = int(cp.asnumpy(path_launch.completed_steps[local]))
                    source_index = int(cp.asnumpy(path_launch.trigger_source[local]))
                    reason_code = int(cp.asnumpy(path_launch.reason[local]))
                    if not 0 <= completed < local_steps:
                        raise EncounterRouteError(
                            "CUDA returned an invalid guarded-step index"
                        )
                    if not 0 <= source_index < validated.source_count:
                        raise EncounterRouteError(
                            "CUDA returned an invalid trigger source"
                        )
                    if reason_code not in REASON_NAMES or reason_code == REASON_NONE:
                        raise EncounterRouteError(
                            "CUDA returned an invalid trigger reason"
                        )
                    p0 = cp.asnumpy(path.positions[completed])
                    v0 = cp.asnumpy(path.velocities[completed])
                    p1 = cp.asnumpy(path.positions[completed + 1])
                    v1 = cp.asnumpy(path.velocities[completed + 1])
                    tp0 = cp.asnumpy(path_launch.tracer_positions[local])
                    tv0 = cp.asnumpy(path_launch.tracer_velocities[local])
                    if gm_host is None:
                        gm_host = cp.asnumpy(validated.source_gm)
                    event_spec = DOP853EventLocatorSpec(
                        relative_tolerance=spec.encounter_route_rtol,
                        absolute_tolerance=spec.encounter_route_atol,
                        maximum_substep_fraction=(
                            spec.encounter_route_maximum_substep_fraction
                        ),
                        event_fraction_tolerance=(
                            spec.event_time_tolerance_fraction
                        ),
                        outward_radial_velocity_tolerance_AU_per_year=(
                            OUTWARD_EVENT_RADIAL_VELOCITY_TOLERANCE_AU_PER_YEAR
                        ),
                        maximum_accepted_steps=(
                            MAXIMUM_EVENT_LOCATOR_ACCEPTED_STEPS
                        ),
                    )
                    collision_radii_host = cp.asnumpy(
                        validated.source_collision_radii
                    )
                    try:
                        routed = locate_tracer_step_events(
                            np.ascontiguousarray(tp0, dtype=np.float64),
                            np.ascontiguousarray(tv0, dtype=np.float64),
                            np.ascontiguousarray(p0, dtype=np.float64),
                            np.ascontiguousarray(v0, dtype=np.float64),
                            np.ascontiguousarray(p1, dtype=np.float64),
                            np.ascontiguousarray(v1, dtype=np.float64),
                            np.ascontiguousarray(gm_host, dtype=np.float64),
                            np.ascontiguousarray(
                                collision_radii_host, dtype=np.float64
                            ),
                            step_years=spec.step_years,
                            primary_source_index=0,
                            primary_boundary_radius_AU=(
                                state.primary_outer_boundary_AU
                            ),
                            spec=event_spec,
                        )
                    except TracerEventError as exc:
                        raise EncounterRouteError(
                            "continuous encounter/event route failed closed"
                        ) from exc
                    if (
                        routed.maximum_committed_primary_dynamical_phase_radians
                        > spec.maximum_primary_dynamical_phase
                    ):
                        raise EncounterRouteError(
                            "CPU route exceeded the primary dynamical-phase ceiling"
                        )
                    tp1 = routed.terminal_position_AU
                    tv1 = routed.terminal_velocity_AU_per_year
                    accepted = routed.accepted_step_count
                    rejected = -1
                    routed_minimum = (
                        routed.minimum_evaluated_stage_distance_AU
                    )
                    committed_minimum = (
                        routed.minimum_committed_stage_distance_AU
                    )
                    tracer_index = int(cp.asnumpy(tracer_indices[local]))
                    distance = routed_minimum
                    segment_sha = _source_segment_sha256(
                        p0, v0, p1, v1, gm_host, spec.step_years
                    )
                    removal_event: TracerRemovalEventRecord | None = None
                    if routed.terminated_at_event:
                        located = routed.earliest_event
                        assert located is not None
                        event_source_position, event_source_velocity = (
                            cubic_hermite_source_state(
                                np.ascontiguousarray(p0, dtype=np.float64),
                                np.ascontiguousarray(v0, dtype=np.float64),
                                np.ascontiguousarray(p1, dtype=np.float64),
                                np.ascontiguousarray(v1, dtype=np.float64),
                                located.step_fraction,
                                spec.step_years,
                            )
                        )
                        endpoint_unbound = (
                            located.primary_specific_energy_AU2_per_year2 >= 0.0
                        )
                        if located.event_type == EVENT_SOURCE_RADIUS_CONTACT:
                            outcome = "REMOVED_SOURCE_RADIUS"
                            lifecycle_value = LIFECYCLE_REMOVED_SOURCE_RADIUS
                            event_type = "SOURCE_RADIUS"
                            event_source_index = located.source_index
                            event_source_id = state.source_ids[event_source_index]
                        elif located.event_type == EVENT_PRIMARY_BOUNDARY_OUTWARD:
                            outcome = "REMOVED_OUTER_BOUNDARY"
                            lifecycle_value = LIFECYCLE_REMOVED_OUTER_BOUNDARY
                            event_type = "OUTER_BOUNDARY"
                            event_source_index = -1
                            event_source_id = "NONE"
                        else:  # pragma: no cover - locator contract is closed
                            raise EncounterRouteError(
                                "continuous route returned an unknown event"
                            )
                        surface_source_slot = (
                            located.source_index
                            if located.event_type == EVENT_SOURCE_RADIUS_CONTACT
                            else 0
                        )
                        removal_event = TracerRemovalEventRecord(
                            tracer_index=tracer_index,
                            event_type=event_type,
                            global_step=absolute_start_step + completed,
                            step_fraction=located.step_fraction,
                            source_index=event_source_index,
                            source_id=event_source_id,
                            source_segment_sha256=segment_sha,
                            source_segment_position_0_AU=tuple(
                                tuple(float(value) for value in row) for row in p0
                            ),
                            source_segment_velocity_0_AU_per_year=tuple(
                                tuple(float(value) for value in row) for row in v0
                            ),
                            source_segment_position_1_AU=tuple(
                                tuple(float(value) for value in row) for row in p1
                            ),
                            source_segment_velocity_1_AU_per_year=tuple(
                                tuple(float(value) for value in row) for row in v1
                            ),
                            source_segment_gm_AU3_per_year2=tuple(
                                float(value) for value in gm_host
                            ),
                            source_segment_step_years=spec.step_years,
                            surface_radius_AU=located.surface_threshold_AU,
                            located_distance_AU=located.surface_distance_AU,
                            root_residual_AU=abs(located.surface_residual_AU),
                            surface_tolerance_AU=located.surface_tolerance_AU,
                            position_AU=tuple(
                                float(value) for value in located.position_AU
                            ),
                            velocity_AU_per_year=tuple(
                                float(value)
                                for value in located.velocity_AU_per_year
                            ),
                            surface_source_position_AU=tuple(
                                float(value)
                                for value in event_source_position[
                                    surface_source_slot
                                ]
                            ),
                            surface_source_velocity_AU_per_year=tuple(
                                float(value)
                                for value in event_source_velocity[
                                    surface_source_slot
                                ]
                            ),
                            primary_source_position_AU=tuple(
                                float(value)
                                for value in event_source_position[0]
                            ),
                            primary_source_velocity_AU_per_year=tuple(
                                float(value)
                                for value in event_source_velocity[0]
                            ),
                            primary_relative_specific_energy_AU2_per_year2=(
                                located.primary_specific_energy_AU2_per_year2
                            ),
                            primary_relative_radial_velocity_AU_per_year=(
                                located.primary_radial_velocity_AU_per_year
                            ),
                        )
                    else:
                        outcome = "REJOINED"
                        lifecycle_value = LIFECYCLE_ACTIVE
                        endpoint_unbound = _unbound_from_primary_host(
                            tp1, tv1, p1[0], v1[0], float(gm_host[0])
                        )
                    records.append(
                        TracerEncounterRouteRecord(
                            tracer_index=tracer_index,
                            source_index=source_index,
                            source_id=state.source_ids[source_index],
                            global_step=absolute_start_step + completed,
                            reason=REASON_NAMES[reason_code],
                            source_segment_sha256=segment_sha,
                            source_segment_epoch_0=(
                                state.epoch
                                + (
                                    absolute_start_step
                                    + completed
                                    - state.global_step
                                )
                                * spec.step_years
                            ),
                            source_segment_position_0_AU=tuple(
                                tuple(float(value) for value in row) for row in p0
                            ),
                            source_segment_velocity_0_AU_per_year=tuple(
                                tuple(float(value) for value in row) for row in v0
                            ),
                            source_segment_position_1_AU=tuple(
                                tuple(float(value) for value in row) for row in p1
                            ),
                            source_segment_velocity_1_AU_per_year=tuple(
                                tuple(float(value) for value in row) for row in v1
                            ),
                            source_segment_gm_AU3_per_year2=tuple(
                                float(value) for value in gm_host
                            ),
                            source_segment_step_years=spec.step_years,
                            tracer_position_0_AU=tuple(
                                float(value) for value in tp0
                            ),
                            tracer_velocity_0_AU_per_year=tuple(
                                float(value) for value in tv0
                            ),
                            tracer_position_1_AU=tuple(
                                float(value) for value in tp1
                            ),
                            tracer_velocity_1_AU_per_year=tuple(
                                float(value) for value in tv1
                            ),
                            route_relative_tolerance=spec.encounter_route_rtol,
                            route_absolute_tolerance=spec.encounter_route_atol,
                            route_maximum_substep_fraction=(
                                spec.encounter_route_maximum_substep_fraction
                            ),
                            route_event_fraction_tolerance=(
                                spec.event_time_tolerance_fraction
                            ),
                            route_maximum_primary_dynamical_phase=(
                                spec.maximum_primary_dynamical_phase
                            ),
                            cpu_accepted_steps=accepted,
                            cpu_rejected_steps=rejected,
                            minimum_evaluated_stage_distance_AU=distance,
                            minimum_committed_stage_distance_AU=(
                                committed_minimum
                            ),
                            endpoint_unbound=endpoint_unbound,
                            outcome=outcome,
                            committed_step_fraction=routed.completed_fraction,
                            maximum_accepted_primary_dynamical_phase=(
                                routed.maximum_accepted_primary_dynamical_phase_radians
                            ),
                            maximum_committed_primary_dynamical_phase=(
                                routed.maximum_committed_primary_dynamical_phase_radians
                            ),
                        )
                    )
                    cumulative_minimum[slot] = cp.minimum(
                        cumulative_minimum[slot],
                        cp.float64(committed_minimum),
                    )
                    cumulative_unbound[slot] = (
                        cumulative_unbound[slot] | endpoint_unbound
                    )
                    if removal_event is not None:
                        cumulative_lifecycle[slot] = cp.int32(lifecycle_value)
                        removal_records.append(removal_event)
                        chunk_position[slot] = cp.asarray(tp1)
                        chunk_velocity[slot] = cp.asarray(tv1)
                        continue
                    remaining = local_steps - completed - 1
                    if remaining == 0:
                        chunk_position[slot] = cp.asarray(tp1)
                        chunk_velocity[slot] = cp.asarray(tv1)
                        continue
                    resumed_path = _SourcePath(
                        path.positions[completed + 1 :],
                        path.velocities[completed + 1 :],
                        path.accelerations[completed + 1 :],
                    )
                    resumed = _launch_on_source_path(
                        backend,
                        tracer_kernel,
                        resumed_path,
                        validated.source_gm,
                        validated.source_encounter_radii,
                        validated.source_collision_radii,
                        validated.relative_acceleration_bounds,
                        cp.asarray(tp1[None, :]),
                        cp.asarray(tv1[None, :]),
                        cp.zeros(1, dtype=cp.int32),
                        step=spec.step_years,
                        primary_outer_boundary_AU=(
                            state.primary_outer_boundary_AU
                        ),
                        maximum_primary_dynamical_phase=(
                            spec.maximum_primary_dynamical_phase
                        ),
                        threads=spec.threads_per_block,
                    )
                    pending_launches.append(
                        (
                            resumed,
                            resumed_path,
                            cp.asarray([slot], dtype=cp.int64),
                            cp.asarray([tracer_index], dtype=cp.uint64),
                            absolute_start_step + completed + 1,
                        )
                    )

            if tracer_count:
                pending_launches.append(
                    (
                        launch,
                        source_path,
                        cp.arange(tracer_count, dtype=cp.int64),
                        validated.tracer_indices,
                        state.global_step + processed_steps,
                    )
                )
                while pending_launches:
                    (
                        pending_launch,
                        pending_path,
                        pending_slots,
                        pending_indices,
                        pending_start_step,
                    ) = pending_launches.popleft()
                    commit_path_launch(
                        pending_launch,
                        pending_path,
                        pending_slots,
                        pending_indices,
                        absolute_start_step=pending_start_step,
                    )
            current_source_position = source_path.positions[-1].copy()
            current_source_velocity = source_path.velocities[-1].copy()
            current_tracer_position = chunk_position
            current_tracer_velocity = chunk_velocity
            processed_steps += chunk_steps

        route_counts_device = cp.asarray(route_counts_host)
        # All gathers/scatters, diagnostic uploads, and CPU-route reinjections are part of the
        # transaction.  Surface asynchronous errors and complete the committed
        # endpoint before returning buffers to a caller or another stream.
        cp.cuda.runtime.deviceSynchronize()
        routes = tuple(
            sorted(records, key=lambda row: (row.global_step, row.tracer_index))
        )
        removals = tuple(
            sorted(
                removal_records,
                key=lambda row: (
                    row.global_step,
                    row.step_fraction,
                    row.tracer_index,
                    row.event_type,
                    row.source_index,
                ),
            )
        )
        endpoint_state = _copy_state_with_endpoint(
            state,
            source_positions=current_source_position,
            source_velocities=current_source_velocity,
            tracer_positions=current_tracer_position,
            tracer_velocities=current_tracer_velocity,
            minimum_source_distance_AU=cumulative_minimum,
            ever_unbound=cumulative_unbound,
            encounter_route_counts=route_counts_device,
            tracer_lifecycle=cumulative_lifecycle,
            prior_encounter_routes=routes,
            prior_removal_events=removals,
            step_years=spec.step_years,
            step_count=spec.step_count,
        )
        runtime = cuda_massless_tracer_runtime_identity(spec.backend)
        return CUDAMasslessTracerResult(
            state=endpoint_state,
            minimum_source_distance_AU=cumulative_minimum,
            ever_unbound=cumulative_unbound,
            encounter_route_counts=route_counts_device,
            tracer_lifecycle=cumulative_lifecycle,
            encounter_routes=routes,
            removal_events=removals,
            runtime_identity=runtime,
            input_state_sha256=input_state_sha256,
        )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _host_state_arrays(result: CUDAMasslessTracerResult) -> dict[str, np.ndarray]:
    """Perform the checkpoint API's explicit, documented device-to-host copy."""

    state = result.state
    try:
        import cupy as cp
    except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
        raise CUDAMasslessTracerContractError(
            "CuPy is required to dump a CUDA tracer checkpoint"
        ) from exc
    arrays = {
        "source_positions": cp.asnumpy(state.source_positions),
        "source_velocities": cp.asnumpy(state.source_velocities),
        "source_gm": cp.asnumpy(state.source_gm),
        "source_encounter_radii": cp.asnumpy(state.source_encounter_radii),
        "source_collision_radii": cp.asnumpy(state.source_collision_radii),
        "relative_acceleration_bounds": cp.asnumpy(
            state.relative_acceleration_bounds
        ),
        "tracer_positions": cp.asnumpy(state.tracer_positions),
        "tracer_velocities": cp.asnumpy(state.tracer_velocities),
        "tracer_indices": cp.asnumpy(state.tracer_indices),
        "minimum_source_distance_AU": cp.asnumpy(
            result.minimum_source_distance_AU
        ),
        "ever_unbound": cp.asnumpy(result.ever_unbound),
        "encounter_route_counts": cp.asnumpy(result.encounter_route_counts),
        "tracer_lifecycle": cp.asnumpy(result.tracer_lifecycle),
    }
    return {key: np.ascontiguousarray(value) for key, value in arrays.items()}


def _validated_state_sha256(
    backend: ArrayBackend,
    state: CUDAMasslessTracerState,
    validated: _ValidatedState,
    continuation: _ValidatedContinuation,
) -> str:
    """Hash the exact pre-segment numerical state for checkpoint lineage."""

    cp = backend.xp
    arrays = {
        "source_positions": cp.asnumpy(validated.source_positions),
        "source_velocities": cp.asnumpy(validated.source_velocities),
        "source_gm": cp.asnumpy(validated.source_gm),
        "source_encounter_radii": cp.asnumpy(
            validated.source_encounter_radii
        ),
        "source_collision_radii": cp.asnumpy(
            validated.source_collision_radii
        ),
        "relative_acceleration_bounds": cp.asnumpy(
            validated.relative_acceleration_bounds
        ),
        "tracer_positions": cp.asnumpy(validated.tracer_positions),
        "tracer_velocities": cp.asnumpy(validated.tracer_velocities),
        "tracer_indices": cp.asnumpy(validated.tracer_indices),
        "minimum_source_distance_AU": cp.asnumpy(
            continuation.minimum_source_distance_AU
        ),
        "ever_unbound": cp.asnumpy(continuation.ever_unbound),
        "encounter_route_counts": continuation.encounter_route_counts_host,
        "tracer_lifecycle": cp.asnumpy(continuation.tracer_lifecycle),
    }
    contiguous = {
        key: np.ascontiguousarray(value) for key, value in arrays.items()
    }
    return _state_sha256(_state_metadata(state), contiguous)


def _state_metadata(state: CUDAMasslessTracerState) -> dict[str, Any]:
    return {
        "snapshot_id": state.snapshot_id,
        "epoch": state.epoch,
        "global_step": state.global_step,
        "time_scale": state.time_scale,
        "frame": state.frame,
        "origin": state.origin,
        "axes": state.axes,
        "length_unit": state.length_unit,
        "time_unit": state.time_unit,
        "unit_system_id": state.unit_system_id,
        "source_ids": list(state.source_ids),
        "primary_source_id": state.primary_source_id,
        "force_schedule_sha256": state.force_schedule_sha256,
        "matched_schedule_set_sha256": state.matched_schedule_set_sha256,
        "primary_outer_boundary_AU": state.primary_outer_boundary_AU,
        "lattice_step_years": state.lattice_step_years,
        "provenance": asdict(state.provenance),
    }


def _state_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    digest = hashlib.sha256(STATE_DOMAIN)
    digest.update(_canonical_json(dict(metadata)))
    for key in sorted(arrays):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(value.dtype.str.encode("ascii") + b"\x00")
        digest.update(_canonical_json(list(value.shape)) + b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _lineage_identity_sha256(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    """Bind state attributes that must never change inside a run chain."""

    invariant_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"epoch", "global_step"}
    }
    digest = hashlib.sha256(LINEAGE_IDENTITY_DOMAIN)
    digest.update(_canonical_json(invariant_metadata))
    for key in (
        "source_gm",
        "source_encounter_radii",
        "source_collision_radii",
        "relative_acceleration_bounds",
        "tracer_indices",
    ):
        value = np.ascontiguousarray(arrays[key])
        digest.update(key.encode("ascii") + b"\x00")
        digest.update(value.dtype.str.encode("ascii") + b"\x00")
        digest.update(_canonical_json(list(value.shape)) + b"\x00")
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _checkpoint_run_directory(
    directory: Path, run_id: str, protocol_sha256: str
) -> Path:
    digest = hashlib.sha256(
        CHECKPOINT_NAMESPACE_DOMAIN
        + run_id.encode("utf-8")
        + b"\x00"
        + protocol_sha256.encode("ascii")
    ).hexdigest()
    return directory / f"run-{digest}"


def _checkpoint_record_path(directory: Path, index: int) -> Path:
    stem = f"checkpoint_{index:020d}"
    return directory / f"{stem}.json"


def _checkpoint_array_path_from_record(
    directory: Path,
    index: int,
    record: Mapping[str, Any],
) -> Path:
    name = record.get("array_file")
    digest = record.get("array_file_sha256")
    expected_name = f"checkpoint_{index:020d}.array-{digest}.npz"
    if type(name) is not str or name != expected_name or Path(name).name != name:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint array content-addressed path identity failed"
        )
    return directory / name


def _commit_create_only(temporary: Path, destination: Path) -> None:
    """Atomically publish a same-directory file without overwrite races."""

    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(
            f"refusing to overwrite committed artifact: {destination}"
        ) from None
    else:
        temporary.unlink()
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(destination.parent, flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)


def _atomic_json_create(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint record: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _commit_create_only(temporary, path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _expected_claim_ceiling() -> dict[str, Any]:
    return {
        "scientific_classification": SCIENTIFIC_CLASSIFICATION,
        "project_scope": PROJECT_SCOPE,
        "claims": dict(RESULT_CLAIMS),
    }


def _validate_checkpoint_record_invariants(
    record: Mapping[str, Any],
    current_runtime: Mapping[str, Any],
) -> None:
    """Fail closed if any historical record used a different implementation."""

    if record.get("schema") != CHECKPOINT_SCHEMA:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history schema changed"
        )
    if record.get("method_id") != METHOD_ID:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history method changed"
        )
    runtime_identity = record.get("runtime_identity")
    if (
        type(runtime_identity) is not dict
        or runtime_identity.get("method_id") != record.get("method_id")
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint method/runtime identity is inconsistent"
        )
    if runtime_identity != dict(current_runtime):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history runtime/device identity changed"
        )
    if record.get("claim_ceiling") != _expected_claim_ceiling():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history claim ceiling changed"
        )
    routes = record.get("encounter_routes")
    if type(routes) is not list or any(type(row) is not dict for row in routes):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history route ledger is invalid"
        )
    removals = record.get("removal_events")
    if type(removals) is not list or any(
        type(row) is not dict for row in removals
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history removal ledger is invalid"
        )
    try:
        _sha256_text(record.get("input_state_sha256"), "input_state_sha256")
        _sha256_text(
            record.get("lineage_identity_sha256"),
            "lineage_identity_sha256",
        )
    except CUDAMasslessTracerContractError as exc:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history lineage digest is invalid"
        ) from exc


def dump_cuda_massless_tracer_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    checkpoint_index: int,
    result: CUDAMasslessTracerResult,
    previous_checkpoint_content_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Create one immutable checkpoint pair after an explicit D2H transfer."""

    directory = Path(directory)
    _text(run_id, "run_id")
    _sha256_text(protocol_sha256, "protocol_sha256")
    if type(checkpoint_index) is not int or checkpoint_index < 0:
        raise CUDAMasslessTracerContractError(
            "checkpoint_index must be a nonnegative integer"
        )
    if type(result) is not CUDAMasslessTracerResult:
        raise CUDAMasslessTracerContractError(
            "result must be an exact CUDAMasslessTracerResult"
        )
    runtime_device_index = result.runtime_identity.get("device_index")
    if type(runtime_device_index) is not int or runtime_device_index < 0:
        raise CUDAMasslessTracerContractError(
            "result runtime identity lacks a valid CUDA device index"
        )
    checkpoint_backend = BackendSpec(
        "cupy", f"cuda:{runtime_device_index}", MAXIMUM_THREADS_PER_BLOCK
    )
    current_runtime = cuda_massless_tracer_runtime_identity(checkpoint_backend)
    if dict(result.runtime_identity) != dict(current_runtime):
        raise CUDAMasslessTracerContractError(
            "checkpoint result runtime identity differs from the current runtime"
        )
    resolved = resolve_backend(checkpoint_backend)
    cp = resolved.xp
    with resolved.activate():
        validated = _validate_state(resolved, result.state)
        continuation = _validate_continuation(resolved, result.state, validated)
        if not _device_truth(
            resolved,
            cp.array_equal(
                result.minimum_source_distance_AU,
                continuation.minimum_source_distance_AU,
            ),
            "checkpoint minimum-distance result/state identity",
        ) or not _device_truth(
            resolved,
            cp.array_equal(result.ever_unbound, continuation.ever_unbound),
            "checkpoint unbound result/state identity",
        ) or not np.array_equal(
            cp.asnumpy(result.encounter_route_counts),
            continuation.encounter_route_counts_host,
        ) or not _device_truth(
            resolved,
            cp.array_equal(
                result.tracer_lifecycle,
                continuation.tracer_lifecycle,
            ),
            "checkpoint lifecycle result/state identity",
        ):
            raise CUDAMasslessTracerContractError(
                "checkpoint result diagnostics differ from its state"
            )
        if result.encounter_routes != continuation.encounter_routes:
            raise CUDAMasslessTracerContractError(
                "checkpoint result route ledger differs from its state"
            )
        if result.removal_events != continuation.removal_events:
            raise CUDAMasslessTracerContractError(
                "checkpoint result removal ledger differs from its state"
            )
    if previous_checkpoint_content_sha256 is not None:
        _sha256_text(
            previous_checkpoint_content_sha256,
            "previous_checkpoint_content_sha256",
        )
    if directory.exists() and not directory.is_dir():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint root exists but is not a directory"
        )
    directory.mkdir(parents=True, exist_ok=True)
    expected_claim_ceiling = _expected_claim_ceiling()
    observed_claim_ceiling = {
        "scientific_classification": result.scientific_classification,
        "project_scope": result.project_scope,
        "claims": dict(result.claims),
    }
    if observed_claim_ceiling != expected_claim_ceiling:
        raise CUDAMasslessTracerContractError(
            "checkpoint result exceeds the immutable claim ceiling"
        )
    if result.method_id != METHOD_ID:
        raise CUDAMasslessTracerContractError(
            "checkpoint result method_id differs from this implementation"
        )
    arrays = _host_state_arrays(result)
    metadata = _state_metadata(result.state)
    state_sha256 = _state_sha256(metadata, arrays)
    lineage_identity_sha256 = _lineage_identity_sha256(metadata, arrays)
    route_rows = [asdict(row) for row in result.encounter_routes]
    removal_rows = [asdict(row) for row in result.removal_events]
    existing = _matching_checkpoint_records(directory, run_id, protocol_sha256)
    checkpoint_directory = _checkpoint_run_directory(
        directory, run_id, protocol_sha256
    )
    checkpoint_directory.mkdir(parents=False, exist_ok=True)
    previous_historical: tuple[int, Path, dict[str, Any]] | None = None
    for historical_index, historical_path, historical_record in existing:
        _validate_checkpoint_record_invariants(
            historical_record, current_runtime
        )
        historical_metadata, historical_arrays = _load_checkpoint_arrays(
            historical_path, historical_index, historical_record
        )
        if previous_historical is not None:
            prior_index, prior_path, prior_record = previous_historical
            historical_routes = historical_record.get("encounter_routes")
            if type(historical_routes) is not list:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint route ledger must be a JSON array"
                )
            _validate_checkpoint_successor(
                prior_path,
                prior_index,
                prior_record,
                current_metadata=historical_metadata,
                current_arrays=historical_arrays,
                current_routes=historical_routes,
                current_removals=historical_record.get(
                    "removal_events", []
                ),
                current_input_state_sha256=historical_record[
                    "input_state_sha256"
                ],
                current_lineage_identity_sha256=historical_record[
                    "lineage_identity_sha256"
                ],
            )
        previous_historical = (
            historical_index,
            historical_path,
            historical_record,
        )
    if existing:
        previous_index, previous_path, previous_record = existing[-1]
        if checkpoint_index <= previous_index:
            raise CUDAMasslessTracerContractError(
                "checkpoint indices must increase monotonically within a run"
            )
        if (
            previous_checkpoint_content_sha256
            != previous_record["checkpoint_content_sha256"]
        ):
            raise CUDAMasslessTracerContractError(
                "previous checkpoint digest must name the latest committed checkpoint"
            )
        previous_global_step = previous_record.get("state_metadata", {}).get(
            "global_step"
        )
        if (
            type(previous_global_step) is not int
            or result.state.global_step <= previous_global_step
        ):
            raise CUDAMasslessTracerContractError(
                "checkpoint state global_step must advance monotonically"
            )
        _validate_checkpoint_successor(
            previous_path,
            previous_index,
            previous_record,
            current_metadata=metadata,
            current_arrays=arrays,
            current_routes=route_rows,
            current_removals=removal_rows,
            current_input_state_sha256=result.input_state_sha256,
            current_lineage_identity_sha256=lineage_identity_sha256,
        )
    elif previous_checkpoint_content_sha256 is not None:
        raise CUDAMasslessTracerContractError(
            "the first checkpoint in a run cannot name a predecessor"
        )
    record_path = _checkpoint_record_path(checkpoint_directory, checkpoint_index)
    if record_path.exists():
        raise FileExistsError(
            f"refusing to overwrite checkpoint {checkpoint_index}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".checkpoint_{checkpoint_index:020d}.",
        suffix=".npz",
        dir=checkpoint_directory,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        array_sha256 = _sha256_file(temporary)
        array_path = checkpoint_directory / (
            f"checkpoint_{checkpoint_index:020d}.array-{array_sha256}.npz"
        )
        if array_path.exists():
            if _sha256_file(array_path) != array_sha256:
                raise CUDAMasslessTracerIntegrationError(
                    "content-addressed checkpoint array collision"
                )
            temporary.unlink()
        else:
            _commit_create_only(temporary, array_path)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    record: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": run_id,
        "protocol_sha256": protocol_sha256,
        "checkpoint_namespace": checkpoint_directory.name,
        "checkpoint_index": checkpoint_index,
        "array_file": array_path.name,
        "array_file_size_bytes": array_path.stat().st_size,
        "array_file_sha256": array_sha256,
        "state_metadata": metadata,
        "state_sha256": state_sha256,
        "input_state_sha256": result.input_state_sha256,
        "lineage_identity_sha256": lineage_identity_sha256,
        "method_id": result.method_id,
        "runtime_identity": dict(result.runtime_identity),
        "previous_checkpoint_content_sha256": previous_checkpoint_content_sha256,
        "encounter_routes": route_rows,
        "removal_events": removal_rows,
        "claim_ceiling": expected_claim_ceiling,
    }
    record["checkpoint_content_sha256"] = hashlib.sha256(
        CHECKPOINT_DOMAIN + _canonical_json(record)
    ).hexdigest()
    _atomic_json_create(record_path, record)
    return MappingProxyType(record)


def _load_checkpoint_record(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CUDAMasslessTracerIntegrationError(
            f"checkpoint record is unreadable: {path}"
        ) from exc
    if type(value) is not dict:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint record must be a JSON object"
        )
    recorded = value.pop("checkpoint_content_sha256", None)
    expected = hashlib.sha256(
        CHECKPOINT_DOMAIN + _canonical_json(value)
    ).hexdigest()
    if recorded != expected:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint record content hash failed"
        )
    value["checkpoint_content_sha256"] = recorded
    return value


_CHECKPOINT_ARRAY_KEYS = frozenset(
    {
        "source_positions",
        "source_velocities",
        "source_gm",
        "source_encounter_radii",
        "source_collision_radii",
        "relative_acceleration_bounds",
        "tracer_positions",
        "tracer_velocities",
        "tracer_indices",
        "minimum_source_distance_AU",
        "ever_unbound",
        "encounter_route_counts",
        "tracer_lifecycle",
    }
)


def _load_checkpoint_arrays(
    record_path: Path,
    index: int,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Verify and load one checkpoint's bound endpoint state."""

    array_path = _checkpoint_array_path_from_record(
        record_path.parent, index, record
    )
    if not array_path.is_file():
        raise CUDAMasslessTracerIntegrationError(
            "committed checkpoint record is missing its array file"
        )
    if array_path.stat().st_size != record.get("array_file_size_bytes"):
        raise CUDAMasslessTracerIntegrationError("checkpoint array size failed")
    if _sha256_file(array_path) != record.get("array_file_sha256"):
        raise CUDAMasslessTracerIntegrationError("checkpoint array hash failed")
    try:
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != _CHECKPOINT_ARRAY_KEYS:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint array member roster changed"
                )
            arrays = {
                key: np.ascontiguousarray(np.array(archive[key], copy=True))
                for key in _CHECKPOINT_ARRAY_KEYS
            }
    except (OSError, ValueError) as exc:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint array archive is unreadable"
        ) from exc
    metadata = record.get("state_metadata")
    if type(metadata) is not dict:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint state metadata is missing"
        )
    if _state_sha256(metadata, arrays) != record.get("state_sha256"):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint semantic state hash failed"
        )
    if _lineage_identity_sha256(metadata, arrays) != record.get(
        "lineage_identity_sha256"
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint immutable lineage identity failed"
        )
    return metadata, arrays


def _validate_checkpoint_successor(
    previous_record_path: Path,
    previous_index: int,
    previous_record: Mapping[str, Any],
    *,
    current_metadata: Mapping[str, Any],
    current_arrays: Mapping[str, np.ndarray],
    current_routes: list[dict[str, Any]],
    current_removals: list[dict[str, Any]],
    current_input_state_sha256: str,
    current_lineage_identity_sha256: str,
) -> None:
    """Verify that one endpoint is an actual continuation of its predecessor."""

    previous_metadata, previous_arrays = _load_checkpoint_arrays(
        previous_record_path, previous_index, previous_record
    )
    if current_input_state_sha256 != previous_record.get("state_sha256"):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor was not advanced from the prior endpoint state"
        )
    if (
        current_lineage_identity_sha256
        != previous_record.get("lineage_identity_sha256")
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor changed immutable state identity"
        )
    previous_step = previous_metadata.get("global_step")
    current_step = current_metadata.get("global_step")
    if (
        type(previous_step) is not int
        or type(current_step) is not int
        or current_step <= previous_step
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor global_step did not increase"
        )
    previous_routes = previous_record.get("encounter_routes")
    if type(previous_routes) is not list:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint predecessor route ledger is invalid"
        )
    if _canonical_json(current_routes[: len(previous_routes)]) != _canonical_json(
        previous_routes
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor route ledger is not an exact prefix extension"
        )
    suffix = current_routes[len(previous_routes) :]
    for row in suffix:
        route_step = row.get("global_step") if type(row) is dict else None
        if (
            type(route_step) is not int
            or route_step < previous_step
            or route_step >= current_step
        ):
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor route ledger has an out-of-segment step"
            )
    previous_counts = previous_arrays["encounter_route_counts"]
    current_counts = current_arrays["encounter_route_counts"]
    if previous_counts.shape != current_counts.shape or np.any(
        current_counts < previous_counts
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor encounter-route counts decreased"
        )
    tracer_ids = current_arrays["tracer_indices"]
    slot_for_id = {
        int(tracer_id): slot for slot, tracer_id in enumerate(tracer_ids)
    }
    suffix_counts = np.zeros_like(current_counts)
    for row in suffix:
        slot = slot_for_id.get(row.get("tracer_index"))
        if slot is None:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor route suffix names an unknown tracer"
            )
        suffix_counts[slot] += 1
    if not np.array_equal(
        current_counts - previous_counts, suffix_counts
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor route-count delta disagrees with its ledger"
        )
    previous_removals = previous_record.get("removal_events")
    if type(previous_removals) is not list:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint predecessor removal ledger is invalid"
        )
    if _canonical_json(
        current_removals[: len(previous_removals)]
    ) != _canonical_json(previous_removals):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor removal ledger is not an exact prefix extension"
        )
    removal_suffix = current_removals[len(previous_removals) :]
    previous_lifecycle = previous_arrays["tracer_lifecycle"]
    current_lifecycle = current_arrays["tracer_lifecycle"]
    if previous_lifecycle.shape != current_lifecycle.shape:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor lifecycle shape changed"
        )
    if np.any(
        (previous_lifecycle != LIFECYCLE_ACTIVE)
        & (current_lifecycle != previous_lifecycle)
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor changed a terminal tracer lifecycle"
        )
    new_removal_by_slot: dict[int, dict[str, Any]] = {}
    for row in removal_suffix:
        if type(row) is not dict:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor removal suffix is invalid"
            )
        event_step = row.get("global_step")
        if (
            type(event_step) is not int
            or event_step < previous_step
            or event_step >= current_step
        ):
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor removal ledger has an out-of-segment step"
            )
        slot = slot_for_id.get(row.get("tracer_index"))
        if slot is None or slot in new_removal_by_slot:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor removal suffix has an unknown or duplicate tracer"
            )
        new_removal_by_slot[slot] = row
    for slot in range(current_lifecycle.size):
        before = int(previous_lifecycle[slot])
        after = int(current_lifecycle[slot])
        event = new_removal_by_slot.get(slot)
        if before == LIFECYCLE_ACTIVE and after != LIFECYCLE_ACTIVE:
            if event is None:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint lifecycle transition lacks a removal event"
                )
            expected = (
                LIFECYCLE_REMOVED_SOURCE_RADIUS
                if event.get("event_type") == "SOURCE_RADIUS"
                else LIFECYCLE_REMOVED_OUTER_BOUNDARY
                if event.get("event_type") == "OUTER_BOUNDARY"
                else -1
            )
            if after != expected:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint lifecycle transition contradicts its event"
                )
        elif event is not None:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint removal event does not create a lifecycle transition"
            )
    previous_minimum = previous_arrays["minimum_source_distance_AU"]
    current_minimum = current_arrays["minimum_source_distance_AU"]
    if previous_minimum.shape != current_minimum.shape or np.any(
        current_minimum > previous_minimum
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor cumulative minimum increased"
        )
    previous_unbound = previous_arrays["ever_unbound"]
    current_unbound = current_arrays["ever_unbound"]
    if previous_unbound.shape != current_unbound.shape or np.any(
        previous_unbound & ~current_unbound
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor cleared a cumulative unbound flag"
        )
    previously_removed = previous_lifecycle != LIFECYCLE_ACTIVE
    for key in ("tracer_positions", "tracer_velocities"):
        if not np.array_equal(
            previous_arrays[key][previously_removed],
            current_arrays[key][previously_removed],
        ):
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint successor moved a terminal tracer"
            )
    if not np.array_equal(
        previous_minimum[previously_removed],
        current_minimum[previously_removed],
    ) or not np.array_equal(
        previous_unbound[previously_removed],
        current_unbound[previously_removed],
    ) or not np.array_equal(
        previous_counts[previously_removed],
        current_counts[previously_removed],
    ):
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint successor changed diagnostics for a terminal tracer"
        )


def _matching_checkpoint_records(
    directory: Path,
    run_id: str,
    protocol_sha256: str,
) -> tuple[tuple[int, Path, dict[str, Any]], ...]:
    """Load, identify, order, and hash-chain one run's committed records."""

    checkpoint_directory = _checkpoint_run_directory(
        directory, run_id, protocol_sha256
    )
    if not checkpoint_directory.exists():
        return ()
    if not checkpoint_directory.is_dir():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint run namespace exists but is not a directory"
        )
    matching: list[tuple[int, Path, dict[str, Any]]] = []
    for path in checkpoint_directory.glob("checkpoint_*.json"):
        stem = path.stem
        _, separator, suffix = stem.partition("_")
        if separator != "_" or not suffix.isdigit():
            continue
        index = int(suffix)
        record = _load_checkpoint_record(path)
        expected_path = _checkpoint_record_path(checkpoint_directory, index)
        identity = (
            record.get("schema"),
            record.get("run_id"),
            record.get("protocol_sha256"),
            record.get("checkpoint_namespace"),
            record.get("checkpoint_index"),
            path,
        )
        expected_identity = (
            CHECKPOINT_SCHEMA,
            run_id,
            protocol_sha256,
            checkpoint_directory.name,
            index,
            expected_path,
        )
        if identity != expected_identity:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint record identity or canonical path failed"
            )
        matching.append((index, path, record))
    matching.sort(key=lambda item: item[0])
    previous_digest: str | None = None
    previous_index = -1
    seen_indices: set[int] = set()
    for index, _, record in matching:
        if index in seen_indices or index <= previous_index:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint indices are duplicated or nonmonotonic"
            )
        if record.get("previous_checkpoint_content_sha256") != previous_digest:
            raise CUDAMasslessTracerIntegrationError(
                "checkpoint predecessor hash chain failed"
            )
        seen_indices.add(index)
        previous_index = index
        previous_digest = record["checkpoint_content_sha256"]
    return tuple(matching)


def load_latest_cuda_massless_tracer_checkpoint(
    directory: str | Path,
    *,
    run_id: str,
    protocol_sha256: str,
    backend: BackendSpec,
) -> CUDAMasslessTracerResult | None:
    """Load the numerically latest valid checkpoint on the same runtime/device.

    Numeric selection intentionally avoids the historical lexicographic bug at
    checkpoint 1,000 and beyond.  A malformed latest *record* fails closed;
    hidden temporary files and an array orphaned before its record was atomically
    installed do not displace the preceding committed checkpoint.
    """

    directory = Path(directory)
    _text(run_id, "run_id")
    _sha256_text(protocol_sha256, "protocol_sha256")
    if type(backend) is not BackendSpec or backend.backend_id != "cupy":
        raise CUDAMasslessTracerContractError(
            "checkpoint load requires an exact CuPy BackendSpec"
        )
    if not directory.exists():
        return None
    if not directory.is_dir():
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint root exists but is not a directory"
        )
    matching = _matching_checkpoint_records(directory, run_id, protocol_sha256)
    if not matching:
        return None
    current_runtime = dict(cuda_massless_tracer_runtime_identity(backend))
    previous_entry: tuple[int, Path, dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    arrays: dict[str, np.ndarray] | None = None
    for historical_index, historical_record_path, historical_record in matching:
        _validate_checkpoint_record_invariants(
            historical_record, current_runtime
        )
        metadata, arrays = _load_checkpoint_arrays(
            historical_record_path, historical_index, historical_record
        )
        if previous_entry is not None:
            previous_index, previous_path, previous_record = previous_entry
            routes = historical_record.get("encounter_routes")
            if type(routes) is not list:
                raise CUDAMasslessTracerIntegrationError(
                    "checkpoint route ledger must be a JSON array"
                )
            _validate_checkpoint_successor(
                previous_path,
                previous_index,
                previous_record,
                current_metadata=metadata,
                current_arrays=arrays,
                current_routes=routes,
                current_removals=historical_record.get(
                    "removal_events", []
                ),
                current_input_state_sha256=historical_record[
                    "input_state_sha256"
                ],
                current_lineage_identity_sha256=historical_record[
                    "lineage_identity_sha256"
                ],
            )
        previous_entry = (
            historical_index,
            historical_record_path,
            historical_record,
        )
    index, record_path, record = matching[-1]
    expected_identity = (
        CHECKPOINT_SCHEMA,
        run_id,
        protocol_sha256,
        index,
    )
    observed_identity = (
        record.get("schema"),
        record.get("run_id"),
        record.get("protocol_sha256"),
        record.get("checkpoint_index"),
    )
    if observed_identity != expected_identity:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint identity does not match the requested run"
        )
    expected_record_path = _checkpoint_record_path(record_path.parent, index)
    if record_path != expected_record_path:
        raise CUDAMasslessTracerIntegrationError("checkpoint path identity failed")
    if metadata is None or arrays is None:
        raise CUDAMasslessTracerIntegrationError(
            "checkpoint history did not yield an endpoint state"
        )
    resolved = resolve_backend(backend)
    cp = resolved.xp
    with resolved.activate():
        state = CUDAMasslessTracerState(
            snapshot_id=metadata["snapshot_id"],
            epoch=metadata["epoch"],
            global_step=metadata["global_step"],
            time_scale=metadata["time_scale"],
            frame=metadata["frame"],
            origin=metadata["origin"],
            axes=metadata["axes"],
            length_unit=metadata["length_unit"],
            time_unit=metadata["time_unit"],
            unit_system_id=metadata["unit_system_id"],
            source_ids=tuple(metadata["source_ids"]),
            primary_source_id=metadata["primary_source_id"],
            source_positions=cp.asarray(arrays["source_positions"]),
            source_velocities=cp.asarray(arrays["source_velocities"]),
            source_gm=cp.asarray(arrays["source_gm"]),
            source_encounter_radii=cp.asarray(
                arrays["source_encounter_radii"]
            ),
            source_collision_radii=cp.asarray(
                arrays["source_collision_radii"]
            ),
            relative_acceleration_bounds=cp.asarray(
                arrays["relative_acceleration_bounds"]
            ),
            tracer_positions=cp.asarray(arrays["tracer_positions"]),
            tracer_velocities=cp.asarray(arrays["tracer_velocities"]),
            tracer_indices=cp.asarray(arrays["tracer_indices"]),
            provenance=Provenance(**metadata["provenance"]),
            force_schedule_sha256=metadata["force_schedule_sha256"],
            matched_schedule_set_sha256=metadata[
                "matched_schedule_set_sha256"
            ],
            primary_outer_boundary_AU=metadata[
                "primary_outer_boundary_AU"
            ],
            lattice_step_years=metadata.get("lattice_step_years"),
            minimum_source_distance_AU=cp.asarray(
                arrays["minimum_source_distance_AU"]
            ),
            ever_unbound=cp.asarray(arrays["ever_unbound"]),
            encounter_route_counts=cp.asarray(
                arrays["encounter_route_counts"]
            ),
            tracer_lifecycle=cp.asarray(arrays["tracer_lifecycle"]),
            prior_encounter_routes=tuple(
                TracerEncounterRouteRecord(**row)
                for row in record.get("encounter_routes", [])
            ),
            prior_removal_events=tuple(
                TracerRemovalEventRecord(**row)
                for row in record.get("removal_events", [])
            ),
        )
        validated = _validate_state(resolved, state)
        continuation = _validate_continuation(resolved, state, validated)
        routes = continuation.encounter_routes
        removals = continuation.removal_events
        return CUDAMasslessTracerResult(
            state=state,
            minimum_source_distance_AU=state.minimum_source_distance_AU,
            ever_unbound=state.ever_unbound,
            encounter_route_counts=state.encounter_route_counts,
            tracer_lifecycle=state.tracer_lifecycle,
            encounter_routes=routes,
            removal_events=removals,
            runtime_identity=MappingProxyType(current_runtime),
            input_state_sha256=record["input_state_sha256"],
        )


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CUDA_OPTIONS",
    "CUDAMasslessTracerContractError",
    "CUDAMasslessTracerError",
    "CUDAMasslessTracerIntegrationError",
    "CUDAMasslessTracerResult",
    "CUDAMasslessTracerState",
    "DETERMINISM_SCOPE",
    "EncounterRouteError",
    "EVENT_LOCATOR_SOURCE_SHA256",
    "FixedStepCUDAMasslessTracerSpec",
    "IMPLEMENTATION_SOURCE_SHA256",
    "KERNEL_ID",
    "KERNEL_SOURCE_SHA256",
    "LIFECYCLE_ACTIVE",
    "LIFECYCLE_NAMES",
    "LIFECYCLE_REMOVED_OUTER_BOUNDARY",
    "LIFECYCLE_REMOVED_SOURCE_RADIUS",
    "MAXIMUM_TOTAL_ENCOUNTER_ROUTES",
    "METHOD_ID",
    "PROJECT_SCOPE",
    "SCIENTIFIC_CLASSIFICATION",
    "TracerEncounterRouteRecord",
    "TracerRemovalEventRecord",
    "advance_cuda_massless_tracers",
    "cuda_massless_tracer_runtime_identity",
    "dump_cuda_massless_tracer_checkpoint",
    "load_latest_cuda_massless_tracer_checkpoint",
]
