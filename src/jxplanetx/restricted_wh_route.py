"""Continuous CPU route for one guarded restricted-WH map step.

The CUDA restricted map stores source nodes and tracers in primary-relative
coordinates.  The already-qualified event locator integrates an inertial
Cartesian tracer against cubic-Hermite source segments.  This adapter closes
that coordinate gap without changing the existing locator: it reconstructs a
GM-barycentric source realization from each relative node, integrates every
fine source interval in order, and converts the released or terminal tracer
state back to the declared primary-relative frame.

This is an additive, direct-import M1 component.  It does not yet schedule
guarded CUDA lanes, persist a route ledger, or claim checkpoint/encounter
qualification.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from . import cuda_tracer_events as _event_module
from .cuda_restricted_wh import (
    MAXIMUM_PRIMARY_DYNAMICAL_PHASE,
    FrozenSourcePathTile,
    verify_source_path_tile,
)
from .cuda_tracer_events import (
    DOP853EventLocatorSpec,
    TracerEventRecord,
    TracerEventError,
    cubic_hermite_source_state,
    locate_tracer_step_events,
)


ROUTE_METHOD_ID = "JX_RESTRICTED_WH_DOP853_FINE_SOURCE_ROUTE_V2"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"
PROJECT_SCOPE = "SCREENING_ONLY"
IMPLEMENTATION_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
EVENT_LOCATOR_SOURCE_SHA256 = hashlib.sha256(
    Path(_event_module.__file__).read_bytes()
).hexdigest()
ROUTE_CLAIMS = MappingProxyType(
    {
        "production_qualified": False,
        "close_encounter_qualified": False,
        "collision_response_qualified": False,
        "disk_checkpoint_restart_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "science_result": False,
    }
)


class RestrictedWHRouteError(RuntimeError):
    """Base error for a guarded restricted-WH route."""


class RestrictedWHRouteContractError(RestrictedWHRouteError, ValueError):
    """The route request is outside the fixed M1 contract."""


class RestrictedWHRouteIntegrationError(RestrictedWHRouteError):
    """The route failed closed without releasing a state."""


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RestrictedWHRouteContractError(
            f"{label} must be a finite built-in float"
        )
    return value


def _vector(value: object, label: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float64)
        or value.shape != (3,)
        or not value.flags.c_contiguous
        or not bool(np.all(np.isfinite(value)))
    ):
        raise RestrictedWHRouteContractError(
            f"{label} must be a finite C-contiguous float64 vector"
        )
    owned = np.array(value, dtype=np.float64, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _freeze_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return MappingProxyType(
            {str(key): _freeze_tree(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_tree(item) for item in value)
    return value


def _barycentric_node(
    relative_positions: np.ndarray,
    relative_velocities: np.ndarray,
    source_gm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute sources plus the primary barycentric state.

    Scaling by the largest GM prevents overflow in the weight normalization.
    Only pairwise separations affect the route; this deterministic realization
    fixes the otherwise arbitrary common translation and boost.
    """

    maximum_gm = float(np.max(source_gm))
    scaled = source_gm / maximum_gm
    denominator = float(np.sum(scaled, dtype=np.float64))
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise RestrictedWHRouteIntegrationError(
            "source mass weights are outside the finite domain"
        )
    weights = scaled / denominator
    with np.errstate(over="raise", invalid="raise"):
        try:
            primary_position = -np.sum(
                weights[:, None] * relative_positions,
                axis=0,
                dtype=np.float64,
            )
            primary_velocity = -np.sum(
                weights[:, None] * relative_velocities,
                axis=0,
                dtype=np.float64,
            )
            positions = relative_positions + primary_position[None, :]
            velocities = relative_velocities + primary_velocity[None, :]
        except FloatingPointError as exc:
            raise RestrictedWHRouteIntegrationError(
                "barycentric source reconstruction is outside the finite domain"
            ) from exc
    if not all(
        bool(np.all(np.isfinite(array)))
        for array in (positions, velocities, primary_position, primary_velocity)
    ):
        raise RestrictedWHRouteIntegrationError(
            "barycentric source reconstruction is non-finite"
        )
    return (
        np.ascontiguousarray(positions),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(primary_position),
        np.ascontiguousarray(primary_velocity),
    )


def _segment_sha256(
    source_tile: FrozenSourcePathTile,
    source_node_index: int,
    source_steps_per_map_step: int,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.restricted-wh.route.source-segment.v2\0")
    digest.update(source_tile.source_step_years.hex().encode("ascii"))
    digest.update(b"\0")
    global_source_step = source_tile.global_source_step + source_node_index
    digest.update(global_source_step.to_bytes(8, "little", signed=False))
    digest.update(source_steps_per_map_step.to_bytes(8, "little", signed=False))
    digest.update(
        json.dumps(
            {
                "source_ids": source_tile.source_ids,
                "primary_source_id": source_tile.primary_source_id,
                "source_schedule_sha256": source_tile.source_schedule_sha256,
                "coordinate_system": source_tile.coordinate_system,
                "unit_system_id": source_tile.unit_system_id,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    )
    stop = source_node_index + source_steps_per_map_step + 1
    for array in (
        source_tile.source_relative_positions_AU[source_node_index:stop],
        source_tile.source_relative_velocities_AU_per_year[
            source_node_index:stop
        ],
        source_tile.source_gm_AU3_per_year2,
        source_tile.source_encounter_radii_AU,
    ):
        canonical = np.ascontiguousarray(array)
        digest.update(canonical.dtype.str.encode("ascii"))
        digest.update(str(canonical.shape).encode("ascii"))
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def restricted_wh_route_policy_identity(
    spec: RestrictedWHRouteSpec,
) -> Mapping[str, Any]:
    """Return the canonical numerical/event policy identity for a route."""

    if type(spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHRouteContractError(
            "spec must be an exact RestrictedWHRouteSpec"
        )
    event = spec.event_locator
    record = {
        "schema": "jx.restricted-wh.route-policy.v1",
        "maximum_primary_dynamical_phase_hex": (
            spec.maximum_primary_dynamical_phase.hex()
        ),
        "primary_outer_boundary_AU_hex": (
            None
            if spec.primary_outer_boundary_AU is None
            else spec.primary_outer_boundary_AU.hex()
        ),
        "event_locator": {
            "relative_tolerance_hex": event.relative_tolerance.hex(),
            "absolute_tolerance_hex": event.absolute_tolerance.hex(),
            "maximum_substep_fraction_hex": (
                event.maximum_substep_fraction.hex()
            ),
            "dense_polynomial_validation_points": (
                event.dense_polynomial_validation_points
            ),
            "event_fraction_tolerance_hex": (
                event.event_fraction_tolerance.hex()
            ),
            "event_distance_tolerance_AU_hex": (
                event.event_distance_tolerance_AU.hex()
            ),
            "outward_radial_velocity_tolerance_AU_per_year_hex": (
                event.outward_radial_velocity_tolerance_AU_per_year.hex()
            ),
            "maximum_accepted_steps": event.maximum_accepted_steps,
        },
    }
    payload = json.dumps(
        record,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    identity = dict(record)
    identity["content_sha256"] = hashlib.sha256(
        b"jx.restricted-wh.route-policy.v1\0" + payload
    ).hexdigest()
    return _freeze_tree(identity)


@dataclass(frozen=True)
class RestrictedWHRouteSpec:
    """Fixed policy for one guarded coarse-map redo."""

    event_locator: DOP853EventLocatorSpec = DOP853EventLocatorSpec()
    maximum_primary_dynamical_phase: float = 0.05
    primary_outer_boundary_AU: float | None = None

    def __post_init__(self) -> None:
        if type(self.event_locator) is not DOP853EventLocatorSpec:
            raise RestrictedWHRouteContractError(
                "event_locator must be an exact DOP853EventLocatorSpec"
            )
        phase = _finite_float(
            self.maximum_primary_dynamical_phase,
            "maximum_primary_dynamical_phase",
        )
        if not 0.0 < phase <= MAXIMUM_PRIMARY_DYNAMICAL_PHASE:
            raise RestrictedWHRouteContractError(
                "maximum_primary_dynamical_phase must be in (0, 0.25]"
            )
        if self.primary_outer_boundary_AU is not None:
            boundary = _finite_float(
                self.primary_outer_boundary_AU,
                "primary_outer_boundary_AU",
            )
            if boundary <= 0.0:
                raise RestrictedWHRouteContractError(
                    "primary_outer_boundary_AU must be positive"
                )


@dataclass(frozen=True, eq=False)
class RestrictedWHRouteEvent:
    """One first event expressed in the primary-relative route frame."""

    event_type: str
    source_index: int
    source_id: str
    map_step_fraction: float
    source_interval_offset: int
    interval_fraction: float
    relative_position_AU: np.ndarray
    relative_velocity_AU_per_year: np.ndarray
    surface_distance_AU: float
    surface_threshold_AU: float
    surface_residual_AU: float
    surface_tolerance_AU: float
    primary_specific_energy_AU2_per_year2: float
    primary_radial_velocity_AU_per_year: float
    surface_radial_velocity_AU_per_year: float
    location_method: str

    def __post_init__(self) -> None:
        if type(self.event_type) is not str or not self.event_type:
            raise RestrictedWHRouteContractError("event_type must be nonempty")
        if type(self.source_index) is not int or self.source_index < 0:
            raise RestrictedWHRouteContractError(
                "source_index must be nonnegative"
            )
        if type(self.source_id) is not str or not self.source_id:
            raise RestrictedWHRouteContractError("source_id must be nonempty")
        for value, label in (
            (self.map_step_fraction, "map_step_fraction"),
            (self.interval_fraction, "interval_fraction"),
        ):
            checked = _finite_float(value, label)
            if not 0.0 <= checked <= 1.0:
                raise RestrictedWHRouteContractError(f"{label} must be in [0, 1]")
        if type(self.source_interval_offset) is not int or self.source_interval_offset < 0:
            raise RestrictedWHRouteContractError(
                "source_interval_offset must be nonnegative"
            )
        object.__setattr__(
            self,
            "relative_position_AU",
            _vector(self.relative_position_AU, "relative_position_AU"),
        )
        object.__setattr__(
            self,
            "relative_velocity_AU_per_year",
            _vector(
                self.relative_velocity_AU_per_year,
                "relative_velocity_AU_per_year",
            ),
        )
        for label in (
            "surface_distance_AU",
            "surface_threshold_AU",
            "surface_residual_AU",
            "surface_tolerance_AU",
            "primary_specific_energy_AU2_per_year2",
            "primary_radial_velocity_AU_per_year",
            "surface_radial_velocity_AU_per_year",
        ):
            _finite_float(getattr(self, label), label)
        if type(self.location_method) is not str or not self.location_method:
            raise RestrictedWHRouteContractError(
                "location_method must be nonempty"
            )


@dataclass(frozen=True, eq=False)
class RestrictedWHRouteResult:
    """Event-free endpoint or terminal first event for one coarse map step."""

    tracer_index: int
    source_tile_sha256: str
    source_segment_sha256: str
    global_source_step: int
    source_steps_per_map_step: int
    events: tuple[RestrictedWHRouteEvent, ...]
    endpoint_relative_position_AU: np.ndarray | None
    endpoint_relative_velocity_AU_per_year: np.ndarray | None
    accepted_step_count: int
    derivative_evaluation_count: int
    minimum_evaluated_stage_distance_AU: float
    minimum_committed_stage_distance_AU: float
    maximum_accepted_primary_dynamical_phase_radians: float
    maximum_committed_primary_dynamical_phase_radians: float
    completed_map_fraction: float
    runtime_identity: Mapping[str, Any]
    claims: Mapping[str, bool] = ROUTE_CLAIMS
    method_id: str = ROUTE_METHOD_ID
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    project_scope: str = PROJECT_SCOPE

    def __post_init__(self) -> None:
        if type(self.tracer_index) is not int or self.tracer_index < 0:
            raise RestrictedWHRouteContractError(
                "tracer_index must be nonnegative"
            )
        for value, label in (
            (self.source_tile_sha256, "source_tile_sha256"),
            (self.source_segment_sha256, "source_segment_sha256"),
        ):
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise RestrictedWHRouteContractError(
                    f"{label} must be a lowercase SHA-256 digest"
                )
        if type(self.global_source_step) is not int or self.global_source_step < 0:
            raise RestrictedWHRouteContractError(
                "global_source_step must be nonnegative"
            )
        if (
            type(self.source_steps_per_map_step) is not int
            or self.source_steps_per_map_step <= 0
        ):
            raise RestrictedWHRouteContractError(
                "source_steps_per_map_step must be positive"
            )
        if type(self.events) is not tuple or any(
            type(item) is not RestrictedWHRouteEvent for item in self.events
        ):
            raise RestrictedWHRouteContractError(
                "events must be an immutable RestrictedWHRouteEvent tuple"
            )
        endpoint_pair = (
            self.endpoint_relative_position_AU,
            self.endpoint_relative_velocity_AU_per_year,
        )
        if self.events:
            if any(value is not None for value in endpoint_pair):
                raise RestrictedWHRouteContractError(
                    "terminal route cannot release an endpoint"
                )
        else:
            if any(value is None for value in endpoint_pair):
                raise RestrictedWHRouteContractError(
                    "event-free route must release an endpoint"
                )
            object.__setattr__(
                self,
                "endpoint_relative_position_AU",
                _vector(endpoint_pair[0], "endpoint_relative_position_AU"),
            )
            object.__setattr__(
                self,
                "endpoint_relative_velocity_AU_per_year",
                _vector(
                    endpoint_pair[1], "endpoint_relative_velocity_AU_per_year"
                ),
            )
        for value, label in (
            (self.accepted_step_count, "accepted_step_count"),
            (self.derivative_evaluation_count, "derivative_evaluation_count"),
        ):
            if type(value) is not int or value < 0:
                raise RestrictedWHRouteContractError(
                    f"{label} must be a nonnegative integer"
                )
        for label in (
            "minimum_evaluated_stage_distance_AU",
            "minimum_committed_stage_distance_AU",
            "maximum_accepted_primary_dynamical_phase_radians",
            "maximum_committed_primary_dynamical_phase_radians",
            "completed_map_fraction",
        ):
            _finite_float(getattr(self, label), label)
        if (
            self.minimum_evaluated_stage_distance_AU <= 0.0
            or self.minimum_committed_stage_distance_AU <= 0.0
            or self.maximum_accepted_primary_dynamical_phase_radians < 0.0
            or self.maximum_committed_primary_dynamical_phase_radians < 0.0
            or self.minimum_evaluated_stage_distance_AU
            > self.minimum_committed_stage_distance_AU
            + 64.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                self.minimum_evaluated_stage_distance_AU,
                self.minimum_committed_stage_distance_AU,
            )
            or self.maximum_committed_primary_dynamical_phase_radians
            > self.maximum_accepted_primary_dynamical_phase_radians
            + 64.0
            * np.finfo(np.float64).eps
            * max(
                1.0,
                self.maximum_accepted_primary_dynamical_phase_radians,
                self.maximum_committed_primary_dynamical_phase_radians,
            )
        ):
            raise RestrictedWHRouteContractError(
                "route diagnostics are outside their physical ordering"
            )
        if not 0.0 <= self.completed_map_fraction <= 1.0:
            raise RestrictedWHRouteContractError(
                "completed_map_fraction must be in [0, 1]"
            )
        if self.events and self.completed_map_fraction != self.events[0].map_step_fraction:
            raise RestrictedWHRouteContractError(
                "completed_map_fraction must identify the earliest event"
            )
        if not self.events and self.completed_map_fraction != 1.0:
            raise RestrictedWHRouteContractError(
                "event-free route must complete the full map step"
            )
        if (
            self.method_id != ROUTE_METHOD_ID
            or self.scientific_classification != SCIENTIFIC_CLASSIFICATION
            or self.project_scope != PROJECT_SCOPE
            or dict(self.claims) != dict(ROUTE_CLAIMS)
            or any(self.claims.values())
        ):
            raise RestrictedWHRouteContractError(
                "route identity or claim ceiling changed"
            )
        object.__setattr__(self, "claims", MappingProxyType(dict(ROUTE_CLAIMS)))
        object.__setattr__(self, "runtime_identity", _freeze_tree(self.runtime_identity))

    @property
    def terminated_at_event(self) -> bool:
        return bool(self.events)

    @property
    def terminal_relative_position_AU(self) -> np.ndarray:
        if self.events:
            return self.events[0].relative_position_AU
        assert self.endpoint_relative_position_AU is not None
        return self.endpoint_relative_position_AU

    @property
    def terminal_relative_velocity_AU_per_year(self) -> np.ndarray:
        if self.events:
            return self.events[0].relative_velocity_AU_per_year
        assert self.endpoint_relative_velocity_AU_per_year is not None
        return self.endpoint_relative_velocity_AU_per_year


def _relative_event(
    event: TracerEventRecord,
    *,
    source_ids: tuple[str, ...],
    source_interval_offset: int,
    source_steps_per_map_step: int,
    source_positions_0: np.ndarray,
    source_velocities_0: np.ndarray,
    source_positions_1: np.ndarray,
    source_velocities_1: np.ndarray,
    source_step_years: float,
) -> RestrictedWHRouteEvent:
    source_positions, source_velocities = cubic_hermite_source_state(
        source_positions_0,
        source_velocities_0,
        source_positions_1,
        source_velocities_1,
        event.step_fraction,
        source_step_years,
    )
    primary_position = source_positions[0]
    primary_velocity = source_velocities[0]
    map_fraction = (
        source_interval_offset + event.step_fraction
    ) / source_steps_per_map_step
    return RestrictedWHRouteEvent(
        event_type=event.event_type,
        source_index=event.source_index,
        source_id=source_ids[event.source_index],
        map_step_fraction=float(map_fraction),
        source_interval_offset=source_interval_offset,
        interval_fraction=event.step_fraction,
        relative_position_AU=np.ascontiguousarray(
            event.position_AU - primary_position
        ),
        relative_velocity_AU_per_year=np.ascontiguousarray(
            event.velocity_AU_per_year - primary_velocity
        ),
        surface_distance_AU=event.surface_distance_AU,
        surface_threshold_AU=event.surface_threshold_AU,
        surface_residual_AU=event.surface_residual_AU,
        surface_tolerance_AU=event.surface_tolerance_AU,
        primary_specific_energy_AU2_per_year2=(
            event.primary_specific_energy_AU2_per_year2
        ),
        primary_radial_velocity_AU_per_year=(
            event.primary_radial_velocity_AU_per_year
        ),
        surface_radial_velocity_AU_per_year=(
            event.surface_radial_velocity_AU_per_year
        ),
        location_method=event.location_method,
    )


def route_restricted_wh_map_step(
    tracer_relative_position_AU: np.ndarray,
    tracer_relative_velocity_AU_per_year: np.ndarray,
    *,
    tracer_index: int,
    source_tile: FrozenSourcePathTile,
    source_node_index: int,
    source_steps_per_map_step: int,
    spec: RestrictedWHRouteSpec = RestrictedWHRouteSpec(),
) -> RestrictedWHRouteResult:
    """Redo one guarded map step over every retained fine source segment."""

    position = _vector(
        tracer_relative_position_AU, "tracer_relative_position_AU"
    )
    velocity = _vector(
        tracer_relative_velocity_AU_per_year,
        "tracer_relative_velocity_AU_per_year",
    )
    if type(tracer_index) is not int or tracer_index < 0:
        raise RestrictedWHRouteContractError(
            "tracer_index must be a nonnegative integer"
        )
    if type(source_tile) is not FrozenSourcePathTile:
        raise RestrictedWHRouteContractError(
            "source_tile must be an exact FrozenSourcePathTile"
        )
    tile_digest = verify_source_path_tile(source_tile)
    if type(source_node_index) is not int or source_node_index < 0:
        raise RestrictedWHRouteContractError(
            "source_node_index must be nonnegative"
        )
    if (
        type(source_steps_per_map_step) is not int
        or source_steps_per_map_step <= 0
    ):
        raise RestrictedWHRouteContractError(
            "source_steps_per_map_step must be positive"
        )
    if source_node_index + source_steps_per_map_step >= source_tile.node_count:
        raise RestrictedWHRouteContractError(
            "requested route extends beyond the source tile"
        )
    if type(spec) is not RestrictedWHRouteSpec:
        raise RestrictedWHRouteContractError(
            "spec must be an exact RestrictedWHRouteSpec"
        )
    spec.__post_init__()

    accepted_steps = 0
    derivative_evaluations = 0
    minimum_evaluated = math.inf
    minimum_committed = math.inf
    maximum_accepted_phase = 0.0
    maximum_committed_phase = 0.0
    source_gm = source_tile.source_gm_AU3_per_year2
    source_radii = source_tile.source_encounter_radii_AU

    try:
        for offset in range(source_steps_per_map_step):
            node_0 = source_node_index + offset
            node_1 = node_0 + 1
            positions_0, velocities_0, primary_position_0, primary_velocity_0 = (
                _barycentric_node(
                    source_tile.source_relative_positions_AU[node_0],
                    source_tile.source_relative_velocities_AU_per_year[node_0],
                    source_gm,
                )
            )
            positions_1, velocities_1, primary_position_1, primary_velocity_1 = (
                _barycentric_node(
                    source_tile.source_relative_positions_AU[node_1],
                    source_tile.source_relative_velocities_AU_per_year[node_1],
                    source_gm,
                )
            )
            inertial_position = np.ascontiguousarray(position + primary_position_0)
            inertial_velocity = np.ascontiguousarray(velocity + primary_velocity_0)
            routed = locate_tracer_step_events(
                inertial_position,
                inertial_velocity,
                positions_0,
                velocities_0,
                positions_1,
                velocities_1,
                source_gm,
                source_radii,
                step_years=source_tile.source_step_years,
                primary_source_index=0,
                primary_boundary_radius_AU=spec.primary_outer_boundary_AU,
                spec=spec.event_locator,
            )
            accepted_steps += routed.accepted_step_count
            derivative_evaluations += routed.derivative_evaluation_count
            minimum_evaluated = min(
                minimum_evaluated,
                routed.minimum_evaluated_stage_distance_AU,
            )
            minimum_committed = min(
                minimum_committed,
                routed.minimum_committed_stage_distance_AU,
            )
            maximum_accepted_phase = max(
                maximum_accepted_phase,
                routed.maximum_accepted_primary_dynamical_phase_radians,
            )
            maximum_committed_phase = max(
                maximum_committed_phase,
                routed.maximum_committed_primary_dynamical_phase_radians,
            )
            phase_to_enforce = (
                routed.maximum_committed_primary_dynamical_phase_radians
                if routed.events
                else routed.maximum_accepted_primary_dynamical_phase_radians
            )
            if phase_to_enforce > spec.maximum_primary_dynamical_phase:
                raise RestrictedWHRouteIntegrationError(
                    "DOP853 route exceeded the committed primary phase ceiling"
                )
            if routed.events:
                events = tuple(
                    _relative_event(
                        event,
                        source_ids=source_tile.source_ids,
                        source_interval_offset=offset,
                        source_steps_per_map_step=source_steps_per_map_step,
                        source_positions_0=positions_0,
                        source_velocities_0=velocities_0,
                        source_positions_1=positions_1,
                        source_velocities_1=velocities_1,
                        source_step_years=source_tile.source_step_years,
                    )
                    for event in routed.events
                )
                return RestrictedWHRouteResult(
                    tracer_index=tracer_index,
                    source_tile_sha256=tile_digest,
                    source_segment_sha256=_segment_sha256(
                        source_tile,
                        source_node_index,
                        source_steps_per_map_step,
                    ),
                    global_source_step=(
                        source_tile.global_source_step + source_node_index
                    ),
                    source_steps_per_map_step=source_steps_per_map_step,
                    events=events,
                    endpoint_relative_position_AU=None,
                    endpoint_relative_velocity_AU_per_year=None,
                    accepted_step_count=accepted_steps,
                    derivative_evaluation_count=derivative_evaluations,
                    minimum_evaluated_stage_distance_AU=minimum_evaluated,
                    minimum_committed_stage_distance_AU=minimum_committed,
                    maximum_accepted_primary_dynamical_phase_radians=(
                        maximum_accepted_phase
                    ),
                    maximum_committed_primary_dynamical_phase_radians=(
                        maximum_committed_phase
                    ),
                    completed_map_fraction=events[0].map_step_fraction,
                    runtime_identity=_runtime_identity(source_tile, spec),
                )
            assert routed.endpoint_position_AU is not None
            assert routed.endpoint_velocity_AU_per_year is not None
            position = np.ascontiguousarray(
                routed.endpoint_position_AU - primary_position_1
            )
            velocity = np.ascontiguousarray(
                routed.endpoint_velocity_AU_per_year - primary_velocity_1
            )
    except RestrictedWHRouteError:
        raise
    except TracerEventError as exc:
        raise RestrictedWHRouteIntegrationError(
            "restricted-WH DOP853 route failed closed"
        ) from exc
    except Exception as exc:
        raise RestrictedWHRouteIntegrationError(
            "restricted-WH route failed closed"
        ) from exc

    if not all(
        math.isfinite(value) and value > 0.0
        for value in (minimum_evaluated, minimum_committed)
    ):
        raise RestrictedWHRouteIntegrationError(
            "route did not produce finite positive distance diagnostics"
        )
    return RestrictedWHRouteResult(
        tracer_index=tracer_index,
        source_tile_sha256=tile_digest,
        source_segment_sha256=_segment_sha256(
            source_tile,
            source_node_index,
            source_steps_per_map_step,
        ),
        global_source_step=source_tile.global_source_step + source_node_index,
        source_steps_per_map_step=source_steps_per_map_step,
        events=(),
        endpoint_relative_position_AU=position,
        endpoint_relative_velocity_AU_per_year=velocity,
        accepted_step_count=accepted_steps,
        derivative_evaluation_count=derivative_evaluations,
        minimum_evaluated_stage_distance_AU=minimum_evaluated,
        minimum_committed_stage_distance_AU=minimum_committed,
        maximum_accepted_primary_dynamical_phase_radians=maximum_accepted_phase,
        maximum_committed_primary_dynamical_phase_radians=(
            maximum_committed_phase
        ),
        completed_map_fraction=1.0,
        runtime_identity=_runtime_identity(source_tile, spec),
    )


def _runtime_identity(
    source_tile: FrozenSourcePathTile,
    spec: RestrictedWHRouteSpec,
) -> Mapping[str, Any]:
    try:
        import scipy

        scipy_version = scipy.__version__
    except (ImportError, OSError) as exc:
        raise RestrictedWHRouteIntegrationError(
            "SciPy runtime identity is unavailable"
        ) from exc
    return {
        "method_id": ROUTE_METHOD_ID,
        "implementation_source_sha256": IMPLEMENTATION_SOURCE_SHA256,
        "event_locator_source_sha256": EVENT_LOCATOR_SOURCE_SHA256,
        "source_tile_sha256": source_tile.content_sha256,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "route_policy": restricted_wh_route_policy_identity(spec),
        "determinism_scope": "SAME_PYTHON_NUMPY_SCIPY_CPU_RUNTIME",
    }


__all__ = [
    "EVENT_LOCATOR_SOURCE_SHA256",
    "IMPLEMENTATION_SOURCE_SHA256",
    "PROJECT_SCOPE",
    "ROUTE_CLAIMS",
    "ROUTE_METHOD_ID",
    "RestrictedWHRouteContractError",
    "RestrictedWHRouteError",
    "RestrictedWHRouteEvent",
    "RestrictedWHRouteIntegrationError",
    "RestrictedWHRouteResult",
    "RestrictedWHRouteSpec",
    "SCIENTIFIC_CLASSIFICATION",
    "restricted_wh_route_policy_identity",
    "route_restricted_wh_map_step",
]
