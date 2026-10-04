"""Continuous event location for one guarded restricted-tracer step.

This module is intentionally CPU-only and independent of the CUDA launcher.
It re-integrates exactly one signed outer step with SciPy's DOP853 solver while
the massive sources follow the caller-supplied cubic-Hermite endpoint segment.
Every accepted DOP853 interval is inspected through its dense output.  Source
surface contacts are found both by sign-changing root brackets and by bounded
minimization, so a grazing/tangent contact does not require a sign change.

The locator is a safety mechanism, not a collision-response model.  It stops
at the first event (retaining deterministic simultaneous-event ordering), or
returns the full-step endpoint when no event exists.  Any non-finite,
singular, ambiguous, or numerically unsuccessful operation fails closed and
returns no partial result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np


EVENT_SOURCE_RADIUS_CONTACT = "SOURCE_RADIUS_CONTACT"
EVENT_PRIMARY_BOUNDARY_OUTWARD = "PRIMARY_BOUNDARY_OUTWARD"
LOCATION_BRACKETED_ROOT = "BRACKETED_DENSE_ROOT"
LOCATION_DENSE_MINIMUM = "DENSE_OUTPUT_MINIMUM"
LOCATION_INITIAL_SURFACE = "INITIAL_SURFACE"
LOCATION_EXACT_DENSE_NODE = "EXACT_DENSE_NODE"

_EVENT_PRIORITY = {
    EVENT_SOURCE_RADIUS_CONTACT: 0,
    EVENT_PRIMARY_BOUNDARY_OUTWARD: 1,
}


class TracerEventError(RuntimeError):
    """Base error for the one-step event locator."""


class TracerEventContractError(TracerEventError, ValueError):
    """The event-locator input contract is invalid."""


class TracerEventIntegrationError(TracerEventError):
    """The event route or continuous event location failed closed."""


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TracerEventContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise TracerEventContractError(f"{label} must be a finite number")
    return result


def _immutable_vector(value: object, label: str) -> np.ndarray:
    array = np.asarray(value)
    if type(array) is not np.ndarray or array.dtype != np.dtype(np.float64):
        raise TracerEventContractError(f"{label} must be a NumPy float64 vector")
    if array.shape != (3,) or not array.flags.c_contiguous:
        raise TracerEventContractError(
            f"{label} must have shape (3,) and C-contiguous storage"
        )
    if not np.isfinite(array).all():
        raise TracerEventContractError(f"{label} must be finite")
    owned = np.array(array, dtype=np.float64, order="C", copy=True)
    owned.setflags(write=False)
    return owned


def _strict_array(
    value: object,
    label: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TracerEventContractError(f"{label} must be an exact NumPy array")
    if value.dtype != np.dtype(np.float64):
        raise TracerEventContractError(f"{label} must have float64 dtype")
    if value.shape != shape or not value.flags.c_contiguous:
        raise TracerEventContractError(
            f"{label} must have shape {shape} and C-contiguous storage"
        )
    if not np.isfinite(value).all():
        raise TracerEventContractError(f"{label} must be finite")
    return np.array(value, dtype=np.float64, order="C", copy=True)


@dataclass(frozen=True)
class DOP853EventLocatorSpec:
    """Numerical policy for one signed DOP853 event-location route."""

    relative_tolerance: float = 1.0e-12
    absolute_tolerance: float = 1.0e-14
    maximum_substep_fraction: float = 0.125
    dense_polynomial_validation_points: int = 17
    event_fraction_tolerance: float = 1.0e-12
    event_distance_tolerance_AU: float = 1.0e-11
    outward_radial_velocity_tolerance_AU_per_year: float = 1.0e-13
    maximum_accepted_steps: int = 100_000

    def __post_init__(self) -> None:
        relative = _finite_number(self.relative_tolerance, "relative_tolerance")
        absolute = _finite_number(self.absolute_tolerance, "absolute_tolerance")
        if relative < 100.0 * np.finfo(np.float64).eps:
            raise TracerEventContractError(
                "relative_tolerance must be at least 100 binary64 eps"
            )
        if relative >= 1.0 or absolute <= 0.0:
            raise TracerEventContractError(
                "relative_tolerance must be below one and absolute_tolerance positive"
            )
        maximum = _finite_number(
            self.maximum_substep_fraction, "maximum_substep_fraction"
        )
        if not 0.0 < maximum <= 1.0:
            raise TracerEventContractError(
                "maximum_substep_fraction must be in (0, 1]"
            )
        if (
            type(self.dense_polynomial_validation_points) is not int
            or not 9 <= self.dense_polynomial_validation_points <= 4096
        ):
            raise TracerEventContractError(
                "dense_polynomial_validation_points must be an integer in [9, 4096]"
            )
        fraction_tolerance = _finite_number(
            self.event_fraction_tolerance, "event_fraction_tolerance"
        )
        if not 8.0 * np.finfo(np.float64).eps <= fraction_tolerance <= 1.0e-3:
            raise TracerEventContractError(
                "event_fraction_tolerance must be in [8 eps, 1e-3]"
            )
        distance_tolerance = _finite_number(
            self.event_distance_tolerance_AU,
            "event_distance_tolerance_AU",
        )
        if distance_tolerance <= 0.0:
            raise TracerEventContractError(
                "event_distance_tolerance_AU must be positive"
            )
        radial_tolerance = _finite_number(
            self.outward_radial_velocity_tolerance_AU_per_year,
            "outward_radial_velocity_tolerance_AU_per_year",
        )
        if radial_tolerance < 0.0:
            raise TracerEventContractError(
                "outward radial-velocity tolerance cannot be negative"
            )
        if (
            type(self.maximum_accepted_steps) is not int
            or self.maximum_accepted_steps <= 0
        ):
            raise TracerEventContractError(
                "maximum_accepted_steps must be a positive integer"
            )


@dataclass(frozen=True, eq=False)
class TracerEventRecord:
    """One continuously located event on the accepted dense solution."""

    event_type: str
    source_index: int
    step_fraction: float
    local_time_years: float
    position_AU: np.ndarray
    velocity_AU_per_year: np.ndarray
    surface_distance_AU: float
    surface_threshold_AU: float
    surface_residual_AU: float
    surface_tolerance_AU: float
    primary_specific_energy_AU2_per_year2: float
    primary_radial_velocity_AU_per_year: float
    surface_radial_velocity_AU_per_year: float
    location_method: str

    def __post_init__(self) -> None:
        if self.event_type not in _EVENT_PRIORITY:
            raise TracerEventContractError("event_type is not recognized")
        if type(self.source_index) is not int or self.source_index < 0:
            raise TracerEventContractError("source_index must be nonnegative")
        fraction = _finite_number(self.step_fraction, "step_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise TracerEventContractError("step_fraction must be in [0, 1]")
        _finite_number(self.local_time_years, "local_time_years")
        object.__setattr__(
            self,
            "position_AU",
            _immutable_vector(self.position_AU, "position_AU"),
        )
        object.__setattr__(
            self,
            "velocity_AU_per_year",
            _immutable_vector(self.velocity_AU_per_year, "velocity_AU_per_year"),
        )
        distance = _finite_number(self.surface_distance_AU, "surface_distance_AU")
        threshold = _finite_number(
            self.surface_threshold_AU, "surface_threshold_AU"
        )
        residual = _finite_number(self.surface_residual_AU, "surface_residual_AU")
        surface_tolerance = _finite_number(
            self.surface_tolerance_AU, "surface_tolerance_AU"
        )
        _finite_number(
            self.primary_specific_energy_AU2_per_year2,
            "primary_specific_energy_AU2_per_year2",
        )
        _finite_number(
            self.primary_radial_velocity_AU_per_year,
            "primary_radial_velocity_AU_per_year",
        )
        _finite_number(
            self.surface_radial_velocity_AU_per_year,
            "surface_radial_velocity_AU_per_year",
        )
        if distance < 0.0 or threshold <= 0.0 or surface_tolerance <= 0.0:
            raise TracerEventContractError(
                "event distance must be nonnegative and threshold/tolerance positive"
            )
        if abs(residual) > surface_tolerance:
            raise TracerEventContractError(
                "event surface residual exceeds its effective tolerance"
            )
        if self.location_method not in {
            LOCATION_BRACKETED_ROOT,
            LOCATION_DENSE_MINIMUM,
            LOCATION_INITIAL_SURFACE,
            LOCATION_EXACT_DENSE_NODE,
        }:
            raise TracerEventContractError("location_method is not recognized")


@dataclass(frozen=True, eq=False)
class TracerStepEventResult:
    """First-event transaction or the event-free full-step endpoint."""

    events: tuple[TracerEventRecord, ...]
    endpoint_position_AU: np.ndarray | None
    endpoint_velocity_AU_per_year: np.ndarray | None
    accepted_step_count: int
    derivative_evaluation_count: int
    minimum_evaluated_stage_distance_AU: float
    minimum_committed_stage_distance_AU: float
    maximum_accepted_primary_dynamical_phase_radians: float
    maximum_committed_primary_dynamical_phase_radians: float
    completed_fraction: float

    def __post_init__(self) -> None:
        if type(self.events) is not tuple or any(
            type(item) is not TracerEventRecord for item in self.events
        ):
            raise TracerEventContractError(
                "events must be an immutable tuple of TracerEventRecord"
            )
        if (self.endpoint_position_AU is None) != (
            self.endpoint_velocity_AU_per_year is None
        ):
            raise TracerEventContractError(
                "endpoint position and velocity must both be present or absent"
            )
        if self.events:
            if self.endpoint_position_AU is not None:
                raise TracerEventContractError(
                    "an event transaction cannot also release a full-step endpoint"
                )
        else:
            if self.endpoint_position_AU is None:
                raise TracerEventContractError(
                    "an event-free transaction must release a full-step endpoint"
                )
            object.__setattr__(
                self,
                "endpoint_position_AU",
                _immutable_vector(self.endpoint_position_AU, "endpoint_position_AU"),
            )
            object.__setattr__(
                self,
                "endpoint_velocity_AU_per_year",
                _immutable_vector(
                    self.endpoint_velocity_AU_per_year,
                    "endpoint_velocity_AU_per_year",
                ),
            )
        if type(self.accepted_step_count) is not int or self.accepted_step_count < 0:
            raise TracerEventContractError("accepted_step_count must be nonnegative")
        if (
            type(self.derivative_evaluation_count) is not int
            or self.derivative_evaluation_count < 0
        ):
            raise TracerEventContractError(
                "derivative_evaluation_count must be nonnegative"
            )
        minimum = _finite_number(
            self.minimum_evaluated_stage_distance_AU,
            "minimum_evaluated_stage_distance_AU",
        )
        committed_minimum = _finite_number(
            self.minimum_committed_stage_distance_AU,
            "minimum_committed_stage_distance_AU",
        )
        phase = _finite_number(
            self.maximum_accepted_primary_dynamical_phase_radians,
            "maximum_accepted_primary_dynamical_phase_radians",
        )
        committed_phase = _finite_number(
            self.maximum_committed_primary_dynamical_phase_radians,
            "maximum_committed_primary_dynamical_phase_radians",
        )
        completed = _finite_number(self.completed_fraction, "completed_fraction")
        if (
            minimum <= 0.0
            or committed_minimum <= 0.0
            or phase < 0.0
            or committed_phase < 0.0
            or not 0.0 <= completed <= 1.0
        ):
            raise TracerEventContractError("result diagnostics are outside their domain")
        diagnostic_slack = 64.0 * np.finfo(np.float64).eps * max(
            1.0, minimum, committed_minimum, phase, committed_phase
        )
        if minimum > committed_minimum + diagnostic_slack:
            raise TracerEventContractError(
                "all-evaluated minimum cannot exceed the committed-prefix minimum"
            )
        if committed_phase > phase + diagnostic_slack:
            raise TracerEventContractError(
                "committed-prefix phase cannot exceed the all-accepted phase"
            )
        if self.events and abs(completed - self.events[0].step_fraction) > 8e-12:
            raise TracerEventContractError(
                "completed_fraction must identify the earliest event"
            )
        if not self.events and completed != 1.0:
            raise TracerEventContractError(
                "an event-free route must complete exactly one step"
            )

    @property
    def earliest_event(self) -> TracerEventRecord | None:
        return self.events[0] if self.events else None

    @property
    def terminated_at_event(self) -> bool:
        return bool(self.events)

    @property
    def terminal_position_AU(self) -> np.ndarray:
        if self.events:
            return self.events[0].position_AU
        assert self.endpoint_position_AU is not None
        return self.endpoint_position_AU

    @property
    def terminal_velocity_AU_per_year(self) -> np.ndarray:
        if self.events:
            return self.events[0].velocity_AU_per_year
        assert self.endpoint_velocity_AU_per_year is not None
        return self.endpoint_velocity_AU_per_year


def cubic_hermite_source_state(
    position_0: np.ndarray,
    velocity_0: np.ndarray,
    position_1: np.ndarray,
    velocity_1: np.ndarray,
    fraction: float,
    step_years: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate source position and velocity on an endpoint Hermite segment."""

    s = float(fraction)
    h = float(step_years)
    if not math.isfinite(s) or not math.isfinite(h) or h == 0.0:
        raise TracerEventIntegrationError("invalid Hermite evaluation coordinate")
    s2 = s * s
    s3 = s2 * s
    position = (
        (2.0 * s3 - 3.0 * s2 + 1.0) * position_0
        + (s3 - 2.0 * s2 + s) * h * velocity_0
        + (-2.0 * s3 + 3.0 * s2) * position_1
        + (s3 - s2) * h * velocity_1
    )
    velocity = (
        ((6.0 * s2 - 6.0 * s) / h) * position_0
        + (3.0 * s2 - 4.0 * s + 1.0) * velocity_0
        + ((-6.0 * s2 + 6.0 * s) / h) * position_1
        + (3.0 * s2 - 2.0 * s) * velocity_1
    )
    if not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise TracerEventIntegrationError("Hermite source evaluation is non-finite")
    return position, velocity


@dataclass(frozen=True)
class _SurfaceRoot:
    fraction: float
    method: str
    effective_tolerance_AU: float


def _effective_surface_tolerance(
    spec: DOP853EventLocatorSpec,
    threshold: float,
    *coordinate_vectors: np.ndarray,
) -> float:
    coordinate_scale = max(
        1.0,
        abs(float(threshold)),
        *(float(np.max(np.abs(value))) for value in coordinate_vectors),
    )
    return max(
        spec.event_distance_tolerance_AU,
        256.0 * np.finfo(np.float64).eps * coordinate_scale,
    )


def _distance_squared(
    dense_output: Callable[[float], np.ndarray],
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    step: float,
    fraction: float,
) -> float:
    state = np.asarray(dense_output(fraction * step), dtype=np.float64)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise TracerEventIntegrationError("DOP853 dense output is invalid")
    source_position, _ = cubic_hermite_source_state(
        source_position_0,
        source_velocity_0,
        source_position_1,
        source_velocity_1,
        fraction,
        step,
    )
    relative = state[:3] - source_position
    squared = float(np.dot(relative, relative))
    if not math.isfinite(squared) or squared < 0.0:
        raise TracerEventIntegrationError("dense relative distance is invalid")
    return squared


def _deduplicate_roots(
    candidates: list[_SurfaceRoot], tolerance: float
) -> tuple[_SurfaceRoot, ...]:
    candidates.sort(key=lambda item: (item.fraction, item.method))
    unique: list[_SurfaceRoot] = []
    method_priority = {
        LOCATION_BRACKETED_ROOT: 0,
        LOCATION_DENSE_MINIMUM: 1,
        LOCATION_EXACT_DENSE_NODE: 2,
        LOCATION_INITIAL_SURFACE: 3,
    }
    for candidate in candidates:
        if not unique or abs(candidate.fraction - unique[-1].fraction) > tolerance:
            unique.append(candidate)
            continue
        if method_priority[candidate.method] < method_priority[unique[-1].method]:
            unique[-1] = candidate
    return tuple(unique)


def _dense_distance_critical_fractions(
    dense_output: Callable[[float], np.ndarray],
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    *,
    step: float,
    fraction_0: float,
    fraction_1: float,
    spec: DOP853EventLocatorSpec,
) -> tuple[float, ...]:
    """Recover every stationary distance on one DOP853 dense interval.

    A SciPy DOP853 accepted dense component is a degree-seven polynomial.
    Subtracting the cubic-Hermite source leaves degree at most seven, so its
    squared norm has degree at most fourteen and at most thirteen stationary
    roots.  Chebyshev interpolation is used instead of depending on SciPy's
    private coefficient representation.  Independent evaluations certify the
    reconstructed degree-seven components before their stationary roots are
    trusted; a changed/unsupported dense representation therefore fails closed.
    """

    from numpy.polynomial import Chebyshev

    degree = 7
    span = fraction_1 - fraction_0
    if not span > 0.0:
        raise TracerEventIntegrationError("dense interval has no positive span")

    # Roots of T_8 mapped to the accepted interval give a well-conditioned
    # interpolation stencil with no duplicated endpoint evaluations.
    index = np.arange(degree + 1, dtype=np.float64)
    canonical = np.cos(math.pi * (index + 0.5) / (degree + 1))
    nodes = fraction_0 + 0.5 * span * (canonical + 1.0)

    def relative_position(fraction: float) -> np.ndarray:
        state = np.asarray(dense_output(float(fraction) * step), dtype=np.float64)
        if state.shape != (6,) or not np.isfinite(state).all():
            raise TracerEventIntegrationError("DOP853 dense output is invalid")
        source_position, _ = cubic_hermite_source_state(
            source_position_0,
            source_velocity_0,
            source_position_1,
            source_velocity_1,
            float(fraction),
            step,
        )
        relative = np.asarray(state[:3] - source_position, dtype=np.float64)
        if relative.shape != (3,) or not np.isfinite(relative).all():
            raise TracerEventIntegrationError(
                "dense relative-position polynomial is invalid"
            )
        return relative

    samples = np.asarray([relative_position(float(node)) for node in nodes])
    try:
        components = tuple(
            Chebyshev.fit(
                nodes,
                samples[:, component],
                degree,
                domain=(fraction_0, fraction_1),
            )
            for component in range(3)
        )
    except Exception as exc:
        raise TracerEventIntegrationError(
            "dense distance-polynomial reconstruction failed"
        ) from exc

    # Do not trust the degree assumption implicitly.  Probe a distinct,
    # deterministic set including both endpoints; the tolerance is generous
    # for binary64 interpolation yet tight enough to detect a representation
    # with terms above degree seven.
    validation_count = spec.dense_polynomial_validation_points
    validation = fraction_0 + span * (
        (np.arange(validation_count, dtype=np.float64) + 0.371)
        / validation_count
    )
    validation = np.concatenate(
        (np.asarray([fraction_0, fraction_1]), validation)
    )
    for fraction in validation:
        observed = relative_position(float(fraction))
        reconstructed = np.asarray(
            [component(float(fraction)) for component in components],
            dtype=np.float64,
        )
        if not np.isfinite(reconstructed).all():
            raise TracerEventIntegrationError(
                "reconstructed dense relative position is non-finite"
            )
        scale = max(
            1.0,
            float(np.max(np.abs(observed))),
            float(np.max(np.abs(reconstructed))),
        )
        tolerance = 65_536.0 * np.finfo(np.float64).eps * scale
        if float(np.max(np.abs(observed - reconstructed))) > tolerance:
            raise TracerEventIntegrationError(
                "DOP853 dense output is not a certified degree-seven polynomial"
            )

    squared_distance = components[0] * components[0]
    squared_distance = squared_distance + components[1] * components[1]
    squared_distance = squared_distance + components[2] * components[2]
    try:
        roots = np.asarray(squared_distance.deriv().roots())
    except Exception as exc:
        raise TracerEventIntegrationError(
            "dense distance stationary-root solution failed"
        ) from exc
    if roots.ndim != 1 or not np.isfinite(roots).all():
        raise TracerEventIntegrationError(
            "dense distance stationary roots are non-finite"
        )

    root_slack = 512.0 * np.finfo(np.float64).eps * max(
        1.0, abs(fraction_0), abs(fraction_1)
    )
    ambiguous_imaginary = 1.0e-7 * max(1.0, span)
    critical: list[float] = []
    for raw_root in roots:
        root = complex(raw_root)
        if abs(root.imag) <= root_slack:
            candidate = float(root.real)
        elif (
            fraction_0 - root_slack <= root.real <= fraction_1 + root_slack
            and abs(root.imag) <= ambiguous_imaginary
        ):
            # A nearly-real conjugate pair can be a numerically split repeated
            # stationary point.  Ignoring it could hide a grazing event.
            raise TracerEventIntegrationError(
                "dense distance has an ambiguous nearly-real stationary root"
            )
        else:
            continue
        if fraction_0 - root_slack <= candidate <= fraction_1 + root_slack:
            critical.append(min(fraction_1, max(fraction_0, candidate)))
    critical.sort()
    unique: list[float] = []
    for candidate in critical:
        if not unique or abs(candidate - unique[-1]) > root_slack:
            unique.append(candidate)
    if len(unique) > 13:
        raise TracerEventIntegrationError(
            "dense distance exceeds its stationary-root degree bound"
        )
    return tuple(unique)


def _refined_distance_minima(
    dense_output: Callable[[float], np.ndarray],
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    *,
    step: float,
    fraction_0: float,
    fraction_1: float,
    critical: tuple[float, ...],
    spec: DOP853EventLocatorSpec,
    minimize_scalar: Callable[..., object],
) -> tuple[tuple[float, float], ...]:
    """Refine every possible dense-distance minimum in an isolated basin."""

    def squared(fraction: float) -> float:
        return _distance_squared(
            dense_output,
            source_position_0,
            source_velocity_0,
            source_position_1,
            source_velocity_1,
            step,
            float(fraction),
        )

    minima: list[tuple[float, float]] = []
    for index, candidate in enumerate(critical):
        previous = critical[index - 1] if index else fraction_0
        following = critical[index + 1] if index + 1 < len(critical) else fraction_1
        left = 0.5 * (previous + candidate)
        right = 0.5 * (candidate + following)
        left = max(fraction_0, min(left, candidate))
        right = min(fraction_1, max(right, candidate))
        candidate_squared = squared(candidate)
        left_squared = squared(left)
        right_squared = squared(right)
        if candidate_squared > left_squared or candidate_squared > right_squared:
            # This stationary point is a maximum or a horizontal inflection.
            # It remains a partition landmark but is not a tangent candidate.
            continue
        if right - left <= 32.0 * np.finfo(np.float64).eps * max(
            1.0, abs(candidate)
        ):
            minima.append((candidate, candidate_squared))
            continue
        try:
            minimum = minimize_scalar(
                squared,
                bounds=(left, right),
                method="bounded",
                options={
                    "xatol": np.finfo(np.float64).eps,
                    "maxiter": 512,
                },
            )
        except Exception as exc:
            raise TracerEventIntegrationError(
                "dense-output stationary-point minimization failed"
            ) from exc
        if not bool(getattr(minimum, "success", False)):
            raise TracerEventIntegrationError(
                "dense-output stationary-point minimization did not converge"
            )
        minimum_fraction = float(getattr(minimum, "x", math.nan))
        minimum_squared = float(getattr(minimum, "fun", math.nan))
        if (
            not math.isfinite(minimum_fraction)
            or not math.isfinite(minimum_squared)
            or minimum_squared < 0.0
            or not left <= minimum_fraction <= right
        ):
            raise TracerEventIntegrationError(
                "dense-output stationary-point minimum is invalid"
            )
        if candidate_squared <= minimum_squared:
            # Bounded minimization has a sqrt(eps)-scale location floor in
            # SciPy.  The exhaustive polynomial stationary root can therefore
            # be the sharper representation of a very shallow tangent.  The
            # minimizer still independently establishes the isolated basin.
            minimum_fraction = candidate
            minimum_squared = candidate_squared
        minima.append((minimum_fraction, minimum_squared))
    return tuple(minima)


def _locate_surface_roots(
    dense_output: Callable[[float], np.ndarray],
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    *,
    step: float,
    fraction_0: float,
    fraction_1: float,
    threshold: float,
    spec: DOP853EventLocatorSpec,
    brentq: Callable[..., float],
    minimize_scalar: Callable[..., object],
) -> tuple[_SurfaceRoot, ...]:
    if not 0.0 <= fraction_0 < fraction_1 <= 1.0 + 16.0 * np.finfo(float).eps:
        raise TracerEventIntegrationError("accepted dense interval is invalid")
    fraction_1 = min(1.0, fraction_1)
    threshold_squared = threshold * threshold
    dense_position_0 = np.asarray(
        dense_output(fraction_0 * step), dtype=np.float64
    )[:3]
    dense_position_1 = np.asarray(
        dense_output(fraction_1 * step), dtype=np.float64
    )[:3]
    effective_tolerance = _effective_surface_tolerance(
        spec,
        threshold,
        source_position_0,
        source_position_1,
        dense_position_0,
        dense_position_1,
    )
    squared_tolerance = (
        2.0 * threshold * effective_tolerance
        + effective_tolerance**2
    )

    def squared(fraction: float) -> float:
        return _distance_squared(
            dense_output,
            source_position_0,
            source_velocity_0,
            source_position_1,
            source_velocity_1,
            step,
            float(fraction),
        )

    def residual_squared(fraction: float) -> float:
        return squared(fraction) - threshold_squared

    critical = _dense_distance_critical_fractions(
        dense_output,
        source_position_0,
        source_velocity_0,
        source_position_1,
        source_velocity_1,
        step=step,
        fraction_0=fraction_0,
        fraction_1=fraction_1,
        spec=spec,
    )
    minima = _refined_distance_minima(
        dense_output,
        source_position_0,
        source_velocity_0,
        source_position_1,
        source_velocity_1,
        step=step,
        fraction_0=fraction_0,
        fraction_1=fraction_1,
        critical=critical,
        spec=spec,
        minimize_scalar=minimize_scalar,
    )

    # Critical roots make the polynomial monotonic between landmarks.  Refined
    # minima are added as actual-function landmarks so a sub-threshold valley
    # brackets both contact roots even if its two accepted-step endpoints are
    # outside the surface.
    landmarks = [fraction_0, fraction_1, *critical]
    landmarks.extend(item[0] for item in minima)
    landmarks.sort()
    landmark_tolerance = 64.0 * np.finfo(np.float64).eps * max(
        1.0, abs(fraction_0), abs(fraction_1)
    )
    unique_landmarks: list[float] = []
    for landmark in landmarks:
        landmark = min(fraction_1, max(fraction_0, float(landmark)))
        if (
            not unique_landmarks
            or abs(landmark - unique_landmarks[-1]) > landmark_tolerance
        ):
            unique_landmarks.append(landmark)
    values = [residual_squared(item) for item in unique_landmarks]
    if not all(math.isfinite(item) for item in values):
        raise TracerEventIntegrationError(
            "stationary-partition surface values are non-finite"
        )
    candidates: list[_SurfaceRoot] = []
    # Root accuracy is governed by the surface-residual contract, not by the
    # looser fraction tolerance used only to deduplicate and order ties.
    root_relative_tolerance = 4.0 * np.finfo(np.float64).eps
    root_absolute_tolerance = np.nextafter(0.0, 1.0)

    for index, value in enumerate(values):
        if value == 0.0:
            candidates.append(
                _SurfaceRoot(
                    float(unique_landmarks[index]),
                    LOCATION_EXACT_DENSE_NODE,
                    effective_tolerance,
                )
            )
    for index in range(len(unique_landmarks) - 1):
        left = float(unique_landmarks[index])
        right = float(unique_landmarks[index + 1])
        left_value = float(values[index])
        right_value = float(values[index + 1])
        if left_value * right_value < 0.0:
            try:
                root = float(
                    brentq(
                        residual_squared,
                        left,
                        right,
                        xtol=root_absolute_tolerance,
                        rtol=root_relative_tolerance,
                        maxiter=128,
                    )
                )
            except Exception as exc:
                raise TracerEventIntegrationError(
                    "bracketed dense-output event root failed"
                ) from exc
            candidates.append(
                _SurfaceRoot(
                    root, LOCATION_BRACKETED_ROOT, effective_tolerance
                )
            )

    for minimum_fraction, minimum_squared in minima:
        minimum_value = minimum_squared - threshold_squared
        roundoff_envelope = 256.0 * np.finfo(np.float64).eps * max(
            1.0, threshold_squared, minimum_squared
        )
        if minimum_value > squared_tolerance + roundoff_envelope:
            continue
        # With no sign change, the minimum itself is the tangent/grazing event.
        if abs(minimum_value) <= squared_tolerance + roundoff_envelope:
            candidates.append(
                _SurfaceRoot(
                    minimum_fraction,
                    LOCATION_DENSE_MINIMUM,
                    effective_tolerance,
                )
            )

    checked: list[_SurfaceRoot] = []
    for candidate in _deduplicate_roots(
        candidates, spec.event_fraction_tolerance
    ):
        distance = math.sqrt(max(0.0, squared(candidate.fraction)))
        if abs(distance - threshold) <= candidate.effective_tolerance_AU:
            checked.append(candidate)
        else:
            raise TracerEventIntegrationError(
                "dense-output event candidate does not satisfy the surface residual contract"
            )
    return tuple(checked)


def _minimum_distance_on_interval(
    dense_output: Callable[[float], np.ndarray],
    source_position_0: np.ndarray,
    source_velocity_0: np.ndarray,
    source_position_1: np.ndarray,
    source_velocity_1: np.ndarray,
    *,
    step: float,
    fraction_0: float,
    fraction_1: float,
    spec: DOP853EventLocatorSpec,
    minimize_scalar: Callable[..., object],
) -> float:
    def squared(fraction: float) -> float:
        return _distance_squared(
            dense_output,
            source_position_0,
            source_velocity_0,
            source_position_1,
            source_velocity_1,
            step,
            float(fraction),
        )

    critical = _dense_distance_critical_fractions(
        dense_output,
        source_position_0,
        source_velocity_0,
        source_position_1,
        source_velocity_1,
        step=step,
        fraction_0=fraction_0,
        fraction_1=fraction_1,
        spec=spec,
    )
    minima = _refined_distance_minima(
        dense_output,
        source_position_0,
        source_velocity_0,
        source_position_1,
        source_velocity_1,
        step=step,
        fraction_0=fraction_0,
        fraction_1=fraction_1,
        critical=critical,
        spec=spec,
        minimize_scalar=minimize_scalar,
    )
    minimum_squared = min(squared(fraction_0), squared(fraction_1))
    if minima:
        minimum_squared = min(
            minimum_squared, min(value for _, value in minima)
        )
    distance = math.sqrt(minimum_squared)
    if not math.isfinite(distance) or distance <= 0.0:
        raise TracerEventIntegrationError(
            "accepted step reaches the primary point-mass singularity"
        )
    return distance


def _event_record(
    *,
    event_type: str,
    source_index: int,
    root: _SurfaceRoot,
    threshold: float,
    dense_output: Callable[[float], np.ndarray],
    position_0: np.ndarray,
    velocity_0: np.ndarray,
    position_1: np.ndarray,
    velocity_1: np.ndarray,
    source_gm: np.ndarray,
    primary_source_index: int,
    step: float,
) -> TracerEventRecord:
    state = np.asarray(dense_output(root.fraction * step), dtype=np.float64)
    if state.shape != (6,) or not np.isfinite(state).all():
        raise TracerEventIntegrationError("event state is invalid")
    source_position, source_velocity = cubic_hermite_source_state(
        position_0,
        velocity_0,
        position_1,
        velocity_1,
        root.fraction,
        step,
    )
    relative_surface_position = state[:3] - source_position[source_index]
    relative_surface_velocity = state[3:] - source_velocity[source_index]
    surface_distance = float(np.linalg.norm(relative_surface_position))
    if not surface_distance > 0.0 or not math.isfinite(surface_distance):
        raise TracerEventIntegrationError("event surface distance is invalid")
    surface_radial_velocity = float(
        np.dot(relative_surface_position, relative_surface_velocity)
        / surface_distance
    )
    relative_primary_position = state[:3] - source_position[primary_source_index]
    relative_primary_velocity = state[3:] - source_velocity[primary_source_index]
    primary_distance = float(np.linalg.norm(relative_primary_position))
    speed_squared = float(np.dot(relative_primary_velocity, relative_primary_velocity))
    if not primary_distance > 0.0 or not math.isfinite(speed_squared):
        raise TracerEventIntegrationError("primary event diagnostics are invalid")
    primary_energy = (
        0.5 * speed_squared
        - float(source_gm[primary_source_index]) / primary_distance
    )
    primary_radial_velocity = float(
        np.dot(relative_primary_position, relative_primary_velocity)
        / primary_distance
    )
    diagnostics = (
        surface_radial_velocity,
        primary_energy,
        primary_radial_velocity,
    )
    if not all(math.isfinite(item) for item in diagnostics):
        raise TracerEventIntegrationError("event diagnostics are non-finite")
    return TracerEventRecord(
        event_type=event_type,
        source_index=source_index,
        step_fraction=root.fraction,
        local_time_years=root.fraction * step,
        position_AU=np.ascontiguousarray(state[:3]),
        velocity_AU_per_year=np.ascontiguousarray(state[3:]),
        surface_distance_AU=surface_distance,
        surface_threshold_AU=threshold,
        surface_residual_AU=surface_distance - threshold,
        surface_tolerance_AU=root.effective_tolerance_AU,
        primary_specific_energy_AU2_per_year2=primary_energy,
        primary_radial_velocity_AU_per_year=primary_radial_velocity,
        surface_radial_velocity_AU_per_year=surface_radial_velocity,
        location_method=root.method,
    )


def locate_tracer_step_events(
    tracer_position_AU: np.ndarray,
    tracer_velocity_AU_per_year: np.ndarray,
    source_position_0_AU: np.ndarray,
    source_velocity_0_AU_per_year: np.ndarray,
    source_position_1_AU: np.ndarray,
    source_velocity_1_AU_per_year: np.ndarray,
    source_gm_AU3_per_year2: np.ndarray,
    source_radii_AU: np.ndarray,
    *,
    step_years: float,
    primary_source_index: int = 0,
    primary_boundary_radius_AU: float | None = None,
    spec: DOP853EventLocatorSpec | None = None,
) -> TracerStepEventResult:
    """Route and continuously inspect exactly one signed tracer step.

    The source endpoint states define their paths; they are never integrated or
    modified here.  A positive-radius source contact terminates the transaction.
    An optional primary boundary terminates only where the physical (not
    integration-direction) primary-relative radial velocity is outward.  This
    definition therefore remains valid for a negative ``step_years``.
    """

    if spec is None:
        spec = DOP853EventLocatorSpec()
    if type(spec) is not DOP853EventLocatorSpec:
        raise TracerEventContractError("spec must be DOP853EventLocatorSpec")
    step = _finite_number(step_years, "step_years")
    if step == 0.0:
        raise TracerEventContractError("step_years cannot be zero")
    tracer_position = _strict_array(tracer_position_AU, "tracer_position_AU", (3,))
    tracer_velocity = _strict_array(
        tracer_velocity_AU_per_year, "tracer_velocity_AU_per_year", (3,)
    )
    if type(source_position_0_AU) is not np.ndarray or source_position_0_AU.ndim != 2:
        raise TracerEventContractError(
            "source_position_0_AU must be an exact rank-two NumPy array"
        )
    source_count = int(source_position_0_AU.shape[0])
    if source_count <= 0:
        raise TracerEventContractError("at least one source is required")
    source_shape = (source_count, 3)
    position_0 = _strict_array(
        source_position_0_AU, "source_position_0_AU", source_shape
    )
    velocity_0 = _strict_array(
        source_velocity_0_AU_per_year,
        "source_velocity_0_AU_per_year",
        source_shape,
    )
    position_1 = _strict_array(
        source_position_1_AU, "source_position_1_AU", source_shape
    )
    velocity_1 = _strict_array(
        source_velocity_1_AU_per_year,
        "source_velocity_1_AU_per_year",
        source_shape,
    )
    source_gm = _strict_array(
        source_gm_AU3_per_year2,
        "source_gm_AU3_per_year2",
        (source_count,),
    )
    source_radii = _strict_array(
        source_radii_AU, "source_radii_AU", (source_count,)
    )
    if bool(np.any(source_gm <= 0.0)):
        raise TracerEventContractError("every source GM must be strictly positive")
    if bool(np.any(source_radii < 0.0)):
        raise TracerEventContractError("source radii cannot be negative")
    if (
        type(primary_source_index) is not int
        or not 0 <= primary_source_index < source_count
    ):
        raise TracerEventContractError("primary_source_index is outside the source table")
    boundary: float | None = None
    if primary_boundary_radius_AU is not None:
        boundary = _finite_number(
            primary_boundary_radius_AU, "primary_boundary_radius_AU"
        )
        if boundary <= 0.0:
            raise TracerEventContractError(
                "primary_boundary_radius_AU must be positive when supplied"
            )

    initial_delta = position_0 - tracer_position[None, :]
    initial_distance_squared = np.einsum("ij,ij->i", initial_delta, initial_delta)
    if (
        not np.isfinite(initial_distance_squared).all()
        or bool(np.any(initial_distance_squared <= 0.0))
    ):
        raise TracerEventIntegrationError(
            "initial tracer state reaches a source point-mass singularity"
        )
    initial_distance = np.sqrt(initial_distance_squared)
    initial_surface_tolerances = np.asarray(
        [
            _effective_surface_tolerance(
                spec,
                float(radius),
                position_0[source_index],
                tracer_position,
            )
            if radius > 0.0
            else spec.event_distance_tolerance_AU
            for source_index, radius in enumerate(source_radii)
        ],
        dtype=np.float64,
    )
    inside = (source_radii > 0.0) & (
        initial_distance < source_radii - initial_surface_tolerances
    )
    if bool(np.any(inside)):
        raise TracerEventIntegrationError(
            "initial tracer state is already inside a source event radius"
        )

    y0 = np.concatenate((tracer_position, tracer_velocity))
    initial_event_candidates: list[TracerEventRecord] = []

    # Resolve the initial surface transaction before constructing DOP853.
    # Solver construction itself evaluates trial derivatives away from t=0;
    # allowing that work first could make a valid t=0 terminal event fail or
    # contaminate its ledger with post-event stages.
    class _InitialDense:
        def __call__(self, _time: float) -> np.ndarray:
            return y0.copy()

    initial_dense = _InitialDense()
    for source_index, radius in enumerate(source_radii):
        if radius <= 0.0:
            continue
        if (
            abs(float(initial_distance[source_index] - radius))
            <= initial_surface_tolerances[source_index]
        ):
            initial_event_candidates.append(
                _event_record(
                    event_type=EVENT_SOURCE_RADIUS_CONTACT,
                    source_index=source_index,
                    root=_SurfaceRoot(
                        0.0,
                        LOCATION_INITIAL_SURFACE,
                        float(initial_surface_tolerances[source_index]),
                    ),
                    threshold=float(radius),
                    dense_output=initial_dense,
                    position_0=position_0,
                    velocity_0=velocity_0,
                    position_1=position_1,
                    velocity_1=velocity_1,
                    source_gm=source_gm,
                    primary_source_index=primary_source_index,
                    step=step,
                )
            )
    if boundary is not None:
        primary_distance = float(initial_distance[primary_source_index])
        boundary_tolerance = _effective_surface_tolerance(
            spec,
            boundary,
            position_0[primary_source_index],
            tracer_position,
        )
        if abs(primary_distance - boundary) <= boundary_tolerance:
            record = _event_record(
                event_type=EVENT_PRIMARY_BOUNDARY_OUTWARD,
                source_index=primary_source_index,
                root=_SurfaceRoot(
                    0.0,
                    LOCATION_INITIAL_SURFACE,
                    boundary_tolerance,
                ),
                threshold=boundary,
                dense_output=initial_dense,
                position_0=position_0,
                velocity_0=velocity_0,
                position_1=position_1,
                velocity_1=velocity_1,
                source_gm=source_gm,
                primary_source_index=primary_source_index,
                step=step,
            )
            if (
                record.primary_radial_velocity_AU_per_year
                > spec.outward_radial_velocity_tolerance_AU_per_year
            ):
                initial_event_candidates.append(record)
    if initial_event_candidates:
        initial_event_candidates.sort(
            key=lambda item: (_EVENT_PRIORITY[item.event_type], item.source_index)
        )
        return TracerStepEventResult(
            events=tuple(initial_event_candidates),
            endpoint_position_AU=None,
            endpoint_velocity_AU_per_year=None,
            accepted_step_count=0,
            derivative_evaluation_count=0,
            minimum_evaluated_stage_distance_AU=float(np.min(initial_distance)),
            minimum_committed_stage_distance_AU=float(np.min(initial_distance)),
            maximum_accepted_primary_dynamical_phase_radians=0.0,
            maximum_committed_primary_dynamical_phase_radians=0.0,
            completed_fraction=0.0,
        )

    try:
        from scipy.integrate import DOP853
        from scipy.optimize import brentq, minimize_scalar
    except (ImportError, OSError) as exc:  # pragma: no cover - optional environment
        raise TracerEventIntegrationError(
            "SciPy DOP853 and optimize are required for event location"
        ) from exc

    minimum_stage_distance = math.inf
    stage_distance_samples: list[tuple[float, float]] = []
    derivative_evaluations = 0

    def derivative(local_time: float, state: np.ndarray) -> np.ndarray:
        nonlocal minimum_stage_distance, derivative_evaluations
        derivative_evaluations += 1
        if state.shape != (6,) or not np.isfinite(state).all():
            raise TracerEventIntegrationError("DOP853 derivative state is non-finite")
        fraction = local_time / step
        if not math.isfinite(fraction):
            raise TracerEventIntegrationError("DOP853 stage fraction is non-finite")
        fraction_slack = 64.0 * np.finfo(np.float64).eps
        if not -fraction_slack <= fraction <= 1.0 + fraction_slack:
            raise TracerEventIntegrationError(
                "DOP853 stage escaped the cubic-Hermite source segment"
            )
        fraction = min(1.0, max(0.0, fraction))
        sources, _ = cubic_hermite_source_state(
            position_0,
            velocity_0,
            position_1,
            velocity_1,
            fraction,
            step,
        )
        delta = sources - state[:3]
        squared = np.einsum("ij,ij->i", delta, delta)
        if not np.isfinite(squared).all() or bool(np.any(squared <= 0.0)):
            raise TracerEventIntegrationError(
                "DOP853 stage reaches a source point-mass singularity"
            )
        stage_distance = float(np.sqrt(np.min(squared)))
        minimum_stage_distance = min(minimum_stage_distance, stage_distance)
        stage_distance_samples.append((fraction, stage_distance))
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                acceleration = np.sum(
                    source_gm[:, None]
                    * delta
                    / (squared * np.sqrt(squared))[:, None],
                    axis=0,
                )
            except FloatingPointError as exc:
                raise TracerEventIntegrationError(
                    "DOP853 stage acceleration is outside the finite domain"
                ) from exc
        result = np.concatenate((state[3:], acceleration))
        if not np.isfinite(result).all():
            raise TracerEventIntegrationError(
                "DOP853 derivative is non-finite"
            )
        return result

    try:
        solver = DOP853(
            derivative,
            0.0,
            y0,
            step,
            rtol=spec.relative_tolerance,
            atol=spec.absolute_tolerance,
            max_step=abs(step) * spec.maximum_substep_fraction,
        )
    except TracerEventIntegrationError:
        raise
    except Exception as exc:
        raise TracerEventIntegrationError("DOP853 initialization failed") from exc

    accepted_steps = 0
    maximum_primary_phase = 0.0
    while solver.status == "running":
        previous_time = float(solver.t)
        try:
            message = solver.step()
        except TracerEventIntegrationError:
            raise
        except Exception as exc:
            raise TracerEventIntegrationError("DOP853 accepted-step attempt failed") from exc
        if solver.status == "failed":
            raise TracerEventIntegrationError(
                f"DOP853 route failed: {message or 'no solver message'}"
            )
        accepted_steps += 1
        if accepted_steps > spec.maximum_accepted_steps:
            raise TracerEventIntegrationError(
                "DOP853 accepted-step resource ceiling exceeded"
            )
        current_time = float(solver.t)
        if (
            not math.isfinite(current_time)
            or current_time == previous_time
            or (current_time - previous_time) * step <= 0.0
        ):
            raise TracerEventIntegrationError("DOP853 accepted step made invalid progress")
        try:
            dense_output = solver.dense_output()
        except Exception as exc:
            raise TracerEventIntegrationError(
                "DOP853 did not provide accepted dense output"
            ) from exc
        fraction_0 = previous_time / step
        fraction_1 = current_time / step
        if fraction_1 > 1.0 and fraction_1 <= 1.0 + 32.0 * np.finfo(float).eps:
            fraction_1 = 1.0
        if not 0.0 <= fraction_0 < fraction_1 <= 1.0:
            raise TracerEventIntegrationError(
                "DOP853 accepted interval escaped the one-step domain"
            )

        primary_minimum_distance = _minimum_distance_on_interval(
            dense_output,
            position_0[primary_source_index],
            velocity_0[primary_source_index],
            position_1[primary_source_index],
            velocity_1[primary_source_index],
            step=step,
            fraction_0=fraction_0,
            fraction_1=fraction_1,
            spec=spec,
            minimize_scalar=minimize_scalar,
        )
        try:
            phase = abs(current_time - previous_time) * math.sqrt(
                float(source_gm[primary_source_index])
                / primary_minimum_distance**3
            )
        except (OverflowError, ValueError, ZeroDivisionError) as exc:
            raise TracerEventIntegrationError(
                "accepted primary dynamical phase is outside the finite domain"
            ) from exc
        if not math.isfinite(phase):
            raise TracerEventIntegrationError(
                "accepted primary dynamical phase is non-finite"
            )
        prior_maximum_primary_phase = maximum_primary_phase
        maximum_primary_phase = max(maximum_primary_phase, phase)

        candidates: list[TracerEventRecord] = []
        for source_index, radius in enumerate(source_radii):
            if radius <= 0.0:
                continue
            roots = _locate_surface_roots(
                dense_output,
                position_0[source_index],
                velocity_0[source_index],
                position_1[source_index],
                velocity_1[source_index],
                step=step,
                fraction_0=fraction_0,
                fraction_1=fraction_1,
                threshold=float(radius),
                spec=spec,
                brentq=brentq,
                minimize_scalar=minimize_scalar,
            )
            for root in roots:
                candidates.append(
                    _event_record(
                        event_type=EVENT_SOURCE_RADIUS_CONTACT,
                        source_index=source_index,
                        root=root,
                        threshold=float(radius),
                        dense_output=dense_output,
                        position_0=position_0,
                        velocity_0=velocity_0,
                        position_1=position_1,
                        velocity_1=velocity_1,
                        source_gm=source_gm,
                        primary_source_index=primary_source_index,
                        step=step,
                    )
                )
        if boundary is not None:
            roots = _locate_surface_roots(
                dense_output,
                position_0[primary_source_index],
                velocity_0[primary_source_index],
                position_1[primary_source_index],
                velocity_1[primary_source_index],
                step=step,
                fraction_0=fraction_0,
                fraction_1=fraction_1,
                threshold=boundary,
                spec=spec,
                brentq=brentq,
                minimize_scalar=minimize_scalar,
            )
            for root in roots:
                record = _event_record(
                    event_type=EVENT_PRIMARY_BOUNDARY_OUTWARD,
                    source_index=primary_source_index,
                    root=root,
                    threshold=boundary,
                    dense_output=dense_output,
                    position_0=position_0,
                    velocity_0=velocity_0,
                    position_1=position_1,
                    velocity_1=velocity_1,
                    source_gm=source_gm,
                    primary_source_index=primary_source_index,
                    step=step,
                )
                if (
                    record.primary_radial_velocity_AU_per_year
                    > spec.outward_radial_velocity_tolerance_AU_per_year
                ):
                    candidates.append(record)

        if candidates:
            earliest_fraction = min(item.step_fraction for item in candidates)
            simultaneous = [
                item
                for item in candidates
                if abs(item.step_fraction - earliest_fraction)
                <= spec.event_fraction_tolerance
            ]
            simultaneous.sort(
                key=lambda item: (
                    item.step_fraction,
                    _EVENT_PRIORITY[item.event_type],
                    item.source_index,
                )
            )
            committed_fraction = simultaneous[0].step_fraction
            committed_stage_values = [
                distance
                for fraction, distance in stage_distance_samples
                if fraction
                <= committed_fraction + spec.event_fraction_tolerance
            ]
            terminal_position = simultaneous[0].position_AU
            terminal_sources, _ = cubic_hermite_source_state(
                position_0,
                velocity_0,
                position_1,
                velocity_1,
                committed_fraction,
                step,
            )
            terminal_minimum = float(
                np.min(
                    np.linalg.norm(
                        terminal_sources - terminal_position[None, :], axis=1
                    )
                )
            )
            minimum_audited_distance = min(
                minimum_stage_distance, terminal_minimum
            )
            committed_stage_values.append(terminal_minimum)
            minimum_committed_stage_distance = min(committed_stage_values)
            if committed_fraction <= fraction_0 + spec.event_fraction_tolerance:
                committed_interval_phase = 0.0
            else:
                committed_primary_minimum = _minimum_distance_on_interval(
                    dense_output,
                    position_0[primary_source_index],
                    velocity_0[primary_source_index],
                    position_1[primary_source_index],
                    velocity_1[primary_source_index],
                    step=step,
                    fraction_0=fraction_0,
                    fraction_1=committed_fraction,
                    spec=spec,
                    minimize_scalar=minimize_scalar,
                )
                try:
                    committed_interval_phase = abs(
                        (committed_fraction - fraction_0) * step
                    ) * math.sqrt(
                        float(source_gm[primary_source_index])
                        / committed_primary_minimum**3
                    )
                except (OverflowError, ValueError, ZeroDivisionError) as exc:
                    raise TracerEventIntegrationError(
                        "committed primary dynamical phase is outside the finite domain"
                    ) from exc
            if not math.isfinite(committed_interval_phase):
                raise TracerEventIntegrationError(
                    "committed primary dynamical phase is non-finite"
                )
            maximum_committed_primary_phase = max(
                prior_maximum_primary_phase,
                committed_interval_phase,
            )
            return TracerStepEventResult(
                events=tuple(simultaneous),
                endpoint_position_AU=None,
                endpoint_velocity_AU_per_year=None,
                accepted_step_count=accepted_steps,
                derivative_evaluation_count=derivative_evaluations,
                minimum_evaluated_stage_distance_AU=minimum_audited_distance,
                minimum_committed_stage_distance_AU=(
                    minimum_committed_stage_distance
                ),
                maximum_accepted_primary_dynamical_phase_radians=maximum_primary_phase,
                maximum_committed_primary_dynamical_phase_radians=(
                    maximum_committed_primary_phase
                ),
                # Fractions order first; exact ties then use type/source order.
                completed_fraction=simultaneous[0].step_fraction,
            )

    if solver.status != "finished":
        raise TracerEventIntegrationError(
            f"DOP853 route ended in unexpected state {solver.status!r}"
        )
    if float(solver.t) != step:
        raise TracerEventIntegrationError(
            "DOP853 route did not land on the exact signed outer-step endpoint"
        )
    endpoint = np.asarray(solver.y, dtype=np.float64)
    if endpoint.shape != (6,) or not np.isfinite(endpoint).all():
        raise TracerEventIntegrationError("DOP853 endpoint is invalid")
    if not math.isfinite(minimum_stage_distance) or minimum_stage_distance <= 0.0:
        raise TracerEventIntegrationError("stage-distance diagnostic is invalid")
    return TracerStepEventResult(
        events=(),
        endpoint_position_AU=np.ascontiguousarray(endpoint[:3]),
        endpoint_velocity_AU_per_year=np.ascontiguousarray(endpoint[3:]),
        accepted_step_count=accepted_steps,
        derivative_evaluation_count=derivative_evaluations,
        minimum_evaluated_stage_distance_AU=minimum_stage_distance,
        minimum_committed_stage_distance_AU=minimum_stage_distance,
        maximum_accepted_primary_dynamical_phase_radians=maximum_primary_phase,
        maximum_committed_primary_dynamical_phase_radians=maximum_primary_phase,
        completed_fraction=1.0,
    )


__all__ = [
    "DOP853EventLocatorSpec",
    "EVENT_PRIMARY_BOUNDARY_OUTWARD",
    "EVENT_SOURCE_RADIUS_CONTACT",
    "LOCATION_BRACKETED_ROOT",
    "LOCATION_DENSE_MINIMUM",
    "LOCATION_EXACT_DENSE_NODE",
    "LOCATION_INITIAL_SURFACE",
    "TracerEventContractError",
    "TracerEventError",
    "TracerEventIntegrationError",
    "TracerEventRecord",
    "TracerStepEventResult",
    "cubic_hermite_source_state",
    "locate_tracer_step_events",
]
