"""Correction-only static lunar degree-two gravity in a supplied PA frame.

This additive post-V5 module evaluates the constant unnormalized degree-two
coefficients printed in the retained DE440 technical comments.  The caller
must supply the Moon principal-axis orientation at every evaluation epoch.
The model contains no monopole, libration integration, time-variable lunar
deformation, tides, core dynamics, or qualification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


LUNAR_STATIC_DEGREE2_PAIR_MODEL_ID = (
    "solar-system.force.lunar-static-degree2-principal-axis-pair"
)
DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES = 1_738_000.0
DE440_TECH_COMMENTS_LUNAR_J2 = 2.0321436013500001e-4
DE440_TECH_COMMENTS_LUNAR_C21 = 2.2779569194479859e-10
DE440_TECH_COMMENTS_LUNAR_S21 = 1.3343450358392620e-9
DE440_TECH_COMMENTS_LUNAR_C22 = 2.2380845245752079e-5
DE440_TECH_COMMENTS_LUNAR_S22 = 0.0
DE440_TECH_COMMENTS_LUNAR_DEGREE2 = (
    DE440_TECH_COMMENTS_LUNAR_J2,
    DE440_TECH_COMMENTS_LUNAR_C21,
    DE440_TECH_COMMENTS_LUNAR_S21,
    DE440_TECH_COMMENTS_LUNAR_C22,
    DE440_TECH_COMMENTS_LUNAR_S22,
)
MODEL_OUTPUT = "MODEL_OUTPUT"


class LunarFigureError(ValueError):
    """A lunar-figure request violated the narrow correction contract."""


class LunarFigureDependencyError(LunarFigureError):
    """The explicitly requested NumPy force runtime is unavailable."""


class LunarOrientationProvider(Protocol):
    """Callable returning the inertial-to-principal-axis rotation matrix."""

    def __call__(self, absolute_et: float) -> object: ...


def _identifier(value: object, label: str) -> str:
    if type(value) is not str or not value or len(value) > 128:
        raise LunarFigureError(f"{label} must be bounded nonempty text")
    if value.strip() != value or any(
        ord(character) < 0x21 or ord(character) > 0x7E for character in value
    ):
        raise LunarFigureError(
            f"{label} must be trimmed printable ASCII without spaces"
        )
    return value


def _index(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise LunarFigureError(f"{label} must be a nonnegative built-in integer")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LunarFigureError(f"{label} must be a finite built-in float")
    return value


def _positive_float(value: object, label: str) -> float:
    checked = _finite_float(value, label)
    if checked <= 0.0:
        raise LunarFigureError(f"{label} must be positive")
    return checked


def _coefficients(value: object) -> tuple[float, float, float, float, float]:
    if type(value) is not tuple or len(value) != 5:
        raise LunarFigureError("coefficients must be an exact five-float tuple")
    checked: list[float] = []
    for index, item in enumerate(value):
        checked.append(_finite_float(item, f"coefficients[{index}]"))
    if checked[0] == 0.0:
        raise LunarFigureError("the registered lunar degree-two J2 must be nonzero")
    return (checked[0], checked[1], checked[2], checked[3], checked[4])


@dataclass(frozen=True, slots=True, eq=False)
class LunarStaticDegree2PairForce:
    """Immutable correction for the Moon figure interacting with one target."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    coefficients: tuple[float, float, float, float, float]
    reference_radius_metres: float
    absolute_start_et: float
    orientation_provider: LunarOrientationProvider
    orientation_frame: str = "MOON_PA_DE440"
    inertial_frame: str = "J2000"
    model_id: str = LUNAR_STATIC_DEGREE2_PAIR_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        if _identifier(self.source_id, "source_id") != "MOON":
            raise LunarFigureError("the lunar-figure source_id must be exact MOON")
        target_id = _identifier(self.target_id, "target_id")
        if target_id == self.source_id:
            raise LunarFigureError("source_id and target_id must be distinct")
        source_index = _index(self.source_index, "source_index")
        target_index = _index(self.target_index, "target_index")
        if source_index == target_index:
            raise LunarFigureError("source_index and target_index must be distinct")
        _coefficients(self.coefficients)
        _positive_float(self.reference_radius_metres, "reference_radius_metres")
        _finite_float(self.absolute_start_et, "absolute_start_et")
        if not callable(self.orientation_provider):
            raise LunarFigureError("orientation_provider must be callable")
        if self.orientation_frame != "MOON_PA_DE440":
            raise LunarFigureError("orientation_frame must be exact MOON_PA_DE440")
        if self.inertial_frame != "J2000":
            raise LunarFigureError("inertial_frame must be exact J2000")
        if self.model_id != LUNAR_STATIC_DEGREE2_PAIR_MODEL_ID:
            raise LunarFigureError("model_id is not the registered lunar figure")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarFigureError("lunar-figure output must remain MODEL_OUTPUT")
        for observed, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(observed) is not bool or observed:
                raise LunarFigureError(f"{label} must be exact false")


def prepare_lunar_static_degree2_pair_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    absolute_start_et: float,
    orientation_provider: LunarOrientationProvider,
    coefficients: tuple[float, float, float, float, float] = (
        DE440_TECH_COMMENTS_LUNAR_DEGREE2
    ),
    reference_radius_metres: float = (
        DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    ),
) -> LunarStaticDegree2PairForce:
    """Prepare one constant degree-two lunar-figure pair correction."""

    return LunarStaticDegree2PairForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        coefficients=_coefficients(coefficients),
        reference_radius_metres=reference_radius_metres,
        absolute_start_et=absolute_start_et,
        orientation_provider=orientation_provider,
    )


def _rotation_matrix(value: object):
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarFigureDependencyError(
            "lunar-figure evaluation requires NumPy"
        ) from exc
    if type(value) is not np.ndarray:
        raise LunarFigureError("orientation provider must return an exact NumPy array")
    if (
        value.dtype != np.dtype("float64")
        or value.shape != (3, 3)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarFigureError(
            "orientation matrix must be finite contiguous float64 with shape (3,3)"
        )
    gram = value @ value.T
    tolerance = 64.0 * float(np.finfo(np.float64).eps)
    if float(np.max(np.abs(gram - np.eye(3, dtype=np.float64)))) > tolerance:
        raise LunarFigureError("orientation matrix must be orthonormal")
    determinant = float(np.linalg.det(value))
    if not math.isfinite(determinant) or abs(determinant - 1.0) > tolerance:
        raise LunarFigureError("orientation matrix must be a proper rotation")
    return value


def _degree2_body_acceleration(
    relative_body: object,
    source_gm: float,
    reference_radius: float,
    coefficients: tuple[float, float, float, float, float],
):
    """Return point-target acceleration in the lunar principal-axis frame."""

    import numpy as np

    x = float(relative_body[0])
    y = float(relative_body[1])
    z = float(relative_body[2])
    radius_squared = x * x + y * y + z * z
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise LunarFigureError("lunar source and point target must be separated")
    radius = math.sqrt(radius_squared)
    j2, c21, s21, c22, s22 = coefficients

    harmonic = (
        0.5 * j2 * (x * x + y * y - 2.0 * z * z)
        + 3.0 * c21 * z * x
        + 3.0 * s21 * z * y
        + 3.0 * c22 * (x * x - y * y)
        + 6.0 * s22 * x * y
    )
    gradient = np.ascontiguousarray(
        [
            j2 * x + 3.0 * c21 * z + 6.0 * c22 * x + 6.0 * s22 * y,
            j2 * y + 3.0 * s21 * z - 6.0 * c22 * y + 6.0 * s22 * x,
            -2.0 * j2 * z + 3.0 * c21 * x + 3.0 * s21 * y,
        ],
        dtype=np.float64,
    )
    scale = source_gm * reference_radius * reference_radius
    inverse_radius5 = 1.0 / (radius_squared * radius_squared * radius)
    acceleration = scale * inverse_radius5 * (
        gradient
        - (5.0 * harmonic / radius_squared)
        * np.ascontiguousarray([x, y, z], dtype=np.float64)
    )
    result = np.ascontiguousarray(acceleration, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise LunarFigureError("lunar degree-two acceleration became nonfinite")
    return result


def evaluate_lunar_static_degree2_pair_correction(
    force: LunarStaticDegree2PairForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    gravitational_parameters: object,
    elapsed_time: float = 0.0,
):
    """Return the correction-only acceleration for the registered pair."""

    if type(force) is not LunarStaticDegree2PairForce:
        raise LunarFigureError("force must be an exact LunarStaticDegree2PairForce")
    if type(body_ids) is not tuple or not body_ids:
        raise LunarFigureError("body_ids must be a nonempty exact tuple")
    checked_ids = tuple(
        _identifier(body_id, f"body_ids[{index}]")
        for index, body_id in enumerate(body_ids)
    )
    if len(set(checked_ids)) != len(checked_ids):
        raise LunarFigureError("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise LunarFigureError("configured pair index is outside body_ids")
    if checked_ids[force.source_index] != force.source_id:
        raise LunarFigureError("source index does not match source_id")
    if checked_ids[force.target_index] != force.target_id:
        raise LunarFigureError("target index does not match target_id")

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarFigureDependencyError(
            "lunar-figure evaluation requires NumPy"
        ) from exc
    if type(positions) is not np.ndarray:
        raise LunarFigureError("positions must be an exact NumPy array")
    if (
        positions.dtype != np.dtype("float64")
        or positions.shape != (len(checked_ids), 3)
        or not positions.flags.c_contiguous
        or not np.all(np.isfinite(positions))
    ):
        raise LunarFigureError(
            "positions must be finite contiguous float64 with shape (body_count,3)"
        )
    if type(gravitational_parameters) is not np.ndarray:
        raise LunarFigureError(
            "gravitational_parameters must be an exact NumPy array"
        )
    if (
        gravitational_parameters.dtype != np.dtype("float64")
        or gravitational_parameters.shape != (len(checked_ids),)
        or not gravitational_parameters.flags.c_contiguous
        or not np.all(np.isfinite(gravitational_parameters))
        or not np.all(gravitational_parameters > 0.0)
    ):
        raise LunarFigureError(
            "gravitational_parameters must be finite positive contiguous float64"
        )
    elapsed_time = _finite_float(elapsed_time, "elapsed_time")
    absolute_et = force.absolute_start_et + elapsed_time
    if not math.isfinite(absolute_et):
        raise LunarFigureError("absolute orientation epoch became nonfinite")
    rotation = _rotation_matrix(force.orientation_provider(absolute_et))
    relative_inertial = np.ascontiguousarray(
        positions[force.target_index] - positions[force.source_index],
        dtype=np.float64,
    )
    relative_body = np.ascontiguousarray(rotation @ relative_inertial, dtype=np.float64)
    source_gm = float(gravitational_parameters[force.source_index])
    target_gm = float(gravitational_parameters[force.target_index])
    target_body_acceleration = _degree2_body_acceleration(
        relative_body,
        source_gm,
        force.reference_radius_metres,
        force.coefficients,
    )
    target_inertial_acceleration = np.ascontiguousarray(
        rotation.T @ target_body_acceleration,
        dtype=np.float64,
    )
    result = np.zeros_like(positions)
    result[force.target_index] += target_inertial_acceleration
    result[force.source_index] -= (
        target_gm / source_gm
    ) * target_inertial_acceleration
    if not np.all(np.isfinite(result)):
        raise LunarFigureError("lunar-figure correction became nonfinite")
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "DE440_TECH_COMMENTS_LUNAR_C21",
    "DE440_TECH_COMMENTS_LUNAR_C22",
    "DE440_TECH_COMMENTS_LUNAR_DEGREE2",
    "DE440_TECH_COMMENTS_LUNAR_J2",
    "DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES",
    "DE440_TECH_COMMENTS_LUNAR_S21",
    "DE440_TECH_COMMENTS_LUNAR_S22",
    "LUNAR_STATIC_DEGREE2_PAIR_MODEL_ID",
    "LunarFigureDependencyError",
    "LunarFigureError",
    "LunarOrientationProvider",
    "LunarStaticDegree2PairForce",
    "evaluate_lunar_static_degree2_pair_correction",
    "prepare_lunar_static_degree2_pair_force",
]
