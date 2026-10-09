"""Experimental Earth-pole policies for Step 5 force models.

The default remains the historical fixed J2000 positive-Z pole.  Callers may
instead select an epoch-specific pole once at construction time, or request a
fresh pole at each force-evaluation epoch.  This module selects orientation;
it does not add a gravity harmonic, integrate a trajectory, or qualify a
Solar-System model.

The DE440 long-term prescription follows Park et al. (2021), equations
20--26, with the ROTEX/ROTEY solution values retained by the post-V5 Step 5
study.  It requires PyERFA only when that provider is actually evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
from numbers import Real
from typing import Protocol


FIXED_J2000_POSITIVE_Z = "fixed_j2000_positive_z"
STATIC_AT_START = "static_at_start"
DYNAMIC = "dynamic"
EARTH_POLE_POLICIES = (
    FIXED_J2000_POSITIVE_Z,
    STATIC_AT_START,
    DYNAMIC,
)

J2000_POSITIVE_Z = (0.0, 0.0, 1.0)
SECONDS_PER_DAY = 86_400
JULIAN_CENTURY_DAYS = 36_525
JULIAN_CENTURY_SECONDS = JULIAN_CENTURY_DAYS * SECONDS_PER_DAY

RADIANS_PER_ARCSECOND = math.pi / (180.0 * 3_600.0)
OMEGA_0_ARCSECONDS = 450_160.280
OMEGA_1_ARCSECONDS = -6_962_890.539
OMEGA_2_ARCSECONDS = 7.455
OMEGA_3_ARCSECONDS = 0.008
DELTA_PSI_AMPLITUDE_ARCSECONDS = -17.1996
DELTA_EPSILON_AMPLITUDE_ARCSECONDS = 9.2025
MEAN_EPSILON_0_ARCSECONDS = 84_381.448
MEAN_EPSILON_1_ARCSECONDS = -46.815
MEAN_EPSILON_2_ARCSECONDS = -0.00059
MEAN_EPSILON_3_ARCSECONDS = 0.001813
PHI_X_ARCSECONDS = 5.3439916044148049e-3
PHI_Y_ARCSECONDS = -1.7128824840317532e-2


class EarthPoleError(ValueError):
    """An Earth-pole request is ambiguous or outside this narrow contract."""


class EarthPoleDependencyError(EarthPoleError):
    """The explicitly selected pole provider is unavailable."""


class EarthPoleProvider(Protocol):
    """Callable returning a three-component pole at absolute TDB-like ET."""

    def __call__(self, absolute_et: float) -> object: ...


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise EarthPoleError(f"{label} must be a finite built-in float")
    return value


def _vector3(value: object, label: str) -> tuple[float, float, float]:
    try:
        iterator = iter(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EarthPoleError(f"{label} must be an iterable three-vector") from exc
    items = tuple(islice(iterator, 4))
    if len(items) != 3:
        raise EarthPoleError(f"{label} must contain exactly three components")
    checked: list[float] = []
    for index, item in enumerate(items):
        if not isinstance(item, Real) or isinstance(item, bool):
            raise EarthPoleError(
                f"{label}[{index}] must be a finite real scalar"
            )
        component = float(item)
        if not math.isfinite(component):
            raise EarthPoleError(f"{label}[{index}] must be finite")
        checked.append(0.0 if component == 0.0 else component)
    return (checked[0], checked[1], checked[2])


def _unit_vector(value: object, label: str) -> tuple[float, float, float]:
    vector = _vector3(value, label)
    norm = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise EarthPoleError(f"{label} must have a positive finite norm")
    normalized = tuple(
        0.0 if component == 0.0 else component / norm for component in vector
    )
    if not all(math.isfinite(component) for component in normalized):
        raise EarthPoleError(f"{label} normalization became nonfinite")
    return normalized  # type: ignore[return-value]


def _rotation_x(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        (1.0, 0.0, 0.0),
        (0.0, cosine, sine),
        (0.0, -sine, cosine),
    )


def _rotation_y(angle: float) -> tuple[tuple[float, float, float], ...]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return (
        (cosine, 0.0, -sine),
        (0.0, 1.0, 0.0),
        (sine, 0.0, cosine),
    )


def _resolve_erfa(erfa_module: object | None) -> object:
    if erfa_module is not None:
        module = erfa_module
    else:
        try:
            import erfa as module
        except (ImportError, OSError) as exc:
            raise EarthPoleDependencyError(
                "the DE440 long-term pole provider requires PyERFA"
            ) from exc
    if not callable(getattr(module, "ltpequ", None)) or not callable(
        getattr(module, "ltpecl", None)
    ):
        raise EarthPoleDependencyError(
            "the Earth-pole provider requires callable ltpequ and ltpecl"
        )
    return module


def de440_long_term_earth_pole(
    absolute_et: float,
    *,
    erfa_module: object | None = None,
) -> tuple[float, float, float]:
    """Evaluate the registered DE440 long-term pole at absolute J2000 ET.

    ``absolute_et`` is a finite built-in float containing TDB-compatible SI
    seconds from the SPICE J2000 origin.  The returned vector is deliberately
    raw: :class:`PreparedEarthPolePolicy` exposes both raw and normalized
    views, while the Earth-J2 force performs the single force-facing NumPy
    normalization used by the archived experiment.
    """

    absolute_et = _finite_float(absolute_et, "absolute_et")
    erfa_module = _resolve_erfa(erfa_module)
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise EarthPoleDependencyError(
            "the DE440 long-term pole provider requires NumPy"
        ) from exc
    t = absolute_et / float(JULIAN_CENTURY_SECONDS)
    t_squared = t * t
    t_cubed = t_squared * t
    omega_arcseconds = (
        OMEGA_0_ARCSECONDS
        + OMEGA_1_ARCSECONDS * t
        + OMEGA_2_ARCSECONDS * t_squared
        + OMEGA_3_ARCSECONDS * t_cubed
    )
    omega = omega_arcseconds * RADIANS_PER_ARCSECOND
    delta_psi = (
        DELTA_PSI_AMPLITUDE_ARCSECONDS
        * math.sin(omega)
        * RADIANS_PER_ARCSECOND
    )
    delta_epsilon = (
        DELTA_EPSILON_AMPLITUDE_ARCSECONDS
        * math.cos(omega)
        * RADIANS_PER_ARCSECOND
    )
    mean_epsilon = (
        MEAN_EPSILON_0_ARCSECONDS
        + MEAN_EPSILON_1_ARCSECONDS * t
        + MEAN_EPSILON_2_ARCSECONDS * t_squared
        + MEAN_EPSILON_3_ARCSECONDS * t_cubed
    ) * RADIANS_PER_ARCSECOND
    true_epsilon = mean_epsilon + delta_epsilon
    pole_of_date = np.ascontiguousarray(
        [
            math.sin(delta_psi) * math.sin(true_epsilon),
            (
                math.cos(delta_psi)
                * math.sin(true_epsilon)
                * math.cos(mean_epsilon)
                - math.cos(true_epsilon) * math.sin(mean_epsilon)
            ),
            (
                math.cos(delta_psi)
                * math.sin(true_epsilon)
                * math.sin(mean_epsilon)
                + math.cos(true_epsilon) * math.cos(mean_epsilon)
            ),
        ],
        dtype=np.float64,
    )

    julian_epoch = 2000.0 + 100.0 * t
    mean_equator_pole = np.ascontiguousarray(
        erfa_module.ltpequ(julian_epoch),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    ecliptic_pole = np.ascontiguousarray(
        erfa_module.ltpecl(julian_epoch),  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    for value, label in (
        (mean_equator_pole, "ERFA mean-equator pole"),
        (ecliptic_pole, "ERFA ecliptic pole"),
    ):
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise EarthPoleError(f"{label} must be a finite three-vector")

    def unit_array(value: object, label: str):
        exact = np.ascontiguousarray(value, dtype=np.float64)
        norm = float(np.linalg.norm(exact))
        if exact.shape != (3,) or not math.isfinite(norm) or norm <= 0.0:
            raise EarthPoleError(f"{label} cannot be normalized")
        result = np.ascontiguousarray(exact / norm, dtype=np.float64)
        if not np.all(np.isfinite(result)):
            raise EarthPoleError(f"{label} normalization became nonfinite")
        return result

    first_axis = unit_array(
        np.cross(mean_equator_pole, ecliptic_pole),
        "Vondrak first axis",
    )
    second_axis = unit_array(
        np.cross(
            mean_equator_pole,
            np.cross(mean_equator_pole, ecliptic_pole),
        ),
        "Vondrak second axis",
    )
    third_axis = unit_array(mean_equator_pole, "Vondrak mean-equator pole")
    precession_matrix = np.ascontiguousarray(
        np.stack((first_axis, second_axis, third_axis)),
        dtype=np.float64,
    )
    rotation_x = np.ascontiguousarray(
        _rotation_x(-PHI_X_ARCSECONDS * RADIANS_PER_ARCSECOND),
        dtype=np.float64,
    )
    rotation_y = np.ascontiguousarray(
        _rotation_y(-PHI_Y_ARCSECONDS * RADIANS_PER_ARCSECOND),
        dtype=np.float64,
    )
    inertial_pole = np.ascontiguousarray(
        precession_matrix @ rotation_x @ rotation_y @ pole_of_date,
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(inertial_pole))
    if not math.isfinite(norm) or abs(norm - 1.0) > 8.0 * math.ulp(1.0):
        raise EarthPoleError("DE440 long-term Earth pole became invalid")
    return tuple(float(value) for value in inertial_pole)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class PreparedEarthPolePolicy:
    """A resolved pole-selection policy for one Step 5 trajectory start."""

    mode: str
    absolute_start_et: float
    raw_start_pole: tuple[float, float, float]
    effective_start_pole: tuple[float, float, float]
    _provider: EarthPoleProvider | None

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode not in EARTH_POLE_POLICIES:
            raise EarthPoleError("mode is not a supported Earth-pole policy")
        _finite_float(self.absolute_start_et, "absolute_start_et")
        for value, label in (
            (self.raw_start_pole, "raw_start_pole"),
            (self.effective_start_pole, "effective_start_pole"),
        ):
            if type(value) is not tuple or len(value) != 3 or any(
                type(component) is not float for component in value
            ):
                raise EarthPoleError(
                    f"{label} must be an exact tuple of three built-in floats"
                )
        raw = _vector3(self.raw_start_pole, "raw_start_pole")
        effective = _unit_vector(raw, "raw_start_pole")
        if effective != self.effective_start_pole:
            raise EarthPoleError("effective_start_pole is not the normalized raw pole")
        if self.mode == DYNAMIC:
            if not callable(self._provider):
                raise EarthPoleError("dynamic policy requires a callable pole provider")
        elif self._provider is not None:
            raise EarthPoleError("fixed policies cannot retain a dynamic provider")

    @property
    def is_time_dependent(self) -> bool:
        """Whether later evaluations call the provider at the requested epoch."""

        return self.mode == DYNAMIC

    def raw_pole_at_elapsed_time(self, elapsed_time: float) -> tuple[float, float, float]:
        """Return the provider pole before the force kernel's normalization."""

        elapsed_time = _finite_float(elapsed_time, "elapsed_time")
        if self.mode != DYNAMIC or elapsed_time == 0.0:
            return self.raw_start_pole
        absolute_et = self.absolute_start_et + elapsed_time
        if not math.isfinite(absolute_et):
            raise EarthPoleError("absolute Earth-pole epoch became nonfinite")
        provider = self._provider
        if provider is None:  # Defensive against unsafe object reconstruction.
            raise EarthPoleError("dynamic policy lost its pole provider")
        return _vector3(provider(absolute_et), "dynamic Earth pole")

    def pole_at_elapsed_time(self, elapsed_time: float) -> tuple[float, float, float]:
        """Return the force-effective pole at local elapsed trajectory time."""

        raw = self.raw_pole_at_elapsed_time(elapsed_time)
        if elapsed_time == 0.0:
            return self.effective_start_pole
        return _unit_vector(raw, "dynamic Earth pole")


def prepare_earth_pole_policy(
    mode: str = FIXED_J2000_POSITIVE_Z,
    *,
    absolute_start_et: float = 0.0,
    pole_provider: EarthPoleProvider | None = None,
) -> PreparedEarthPolePolicy:
    """Resolve one explicit Earth-pole policy without changing old defaults.

    ``STATIC_AT_START`` invokes ``pole_provider`` exactly once here and then
    holds both its raw value and a normalized inspection view.  ``DYNAMIC``
    also samples the start here but reevaluates nonzero elapsed epochs.  The
    default fixed policy neither imports PyERFA nor invokes a provider.
    """

    if type(mode) is not str or mode not in EARTH_POLE_POLICIES:
        raise EarthPoleError("mode is not a supported Earth-pole policy")
    absolute_start_et = _finite_float(absolute_start_et, "absolute_start_et")
    if mode == FIXED_J2000_POSITIVE_Z:
        if pole_provider is not None:
            raise EarthPoleError("the fixed +Z policy does not accept a provider")
        raw_start = J2000_POSITIVE_Z
        retained_provider = None
    else:
        if not callable(pole_provider):
            raise EarthPoleError(f"{mode} requires a callable pole provider")
        raw_start = _vector3(
            pole_provider(absolute_start_et),
            f"{mode} start pole",
        )
        retained_provider = pole_provider if mode == DYNAMIC else None
    return PreparedEarthPolePolicy(
        mode=mode,
        absolute_start_et=absolute_start_et,
        raw_start_pole=raw_start,
        effective_start_pole=_unit_vector(raw_start, "start pole"),
        _provider=retained_provider,
    )


__all__ = [
    "DYNAMIC",
    "EARTH_POLE_POLICIES",
    "EarthPoleDependencyError",
    "EarthPoleError",
    "EarthPoleProvider",
    "FIXED_J2000_POSITIVE_Z",
    "J2000_POSITIVE_Z",
    "PreparedEarthPolePolicy",
    "STATIC_AT_START",
    "de440_long_term_earth_pole",
    "prepare_earth_pole_policy",
]
