"""Correction-only static lunar degree-three gravity in a supplied PA frame.

This additive post-V5 module evaluates only the constant unnormalized
degree-three coefficients printed in the retained DE440 technical comments.
The caller supplies the Moon principal-axis orientation at every evaluation
epoch.  The model contains no monopole, lower or higher harmonics, libration
integration, time-variable deformation, tides, or qualification claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import lunar_figure


LUNAR_STATIC_DEGREE3_PAIR_MODEL_ID = (
    "solar-system.force.lunar-static-degree3-principal-axis-pair"
)
DE440_TECH_COMMENTS_LUNAR_J3 = 8.4596458204206400e-6
DE440_TECH_COMMENTS_LUNAR_C31 = 2.8480784606000001e-5
DE440_TECH_COMMENTS_LUNAR_S31 = 5.8915728033016701e-6
DE440_TECH_COMMENTS_LUNAR_C32 = 4.8404881300954602e-6
DE440_TECH_COMMENTS_LUNAR_S32 = 1.6661542142396701e-6
DE440_TECH_COMMENTS_LUNAR_C33 = 1.7116515250000001e-6
DE440_TECH_COMMENTS_LUNAR_S33 = -2.4741804446906898e-7
DE440_TECH_COMMENTS_LUNAR_DEGREE3 = (
    DE440_TECH_COMMENTS_LUNAR_J3,
    DE440_TECH_COMMENTS_LUNAR_C31,
    DE440_TECH_COMMENTS_LUNAR_S31,
    DE440_TECH_COMMENTS_LUNAR_C32,
    DE440_TECH_COMMENTS_LUNAR_S32,
    DE440_TECH_COMMENTS_LUNAR_C33,
    DE440_TECH_COMMENTS_LUNAR_S33,
)
MODEL_OUTPUT = "MODEL_OUTPUT"


class LunarDegree3Error(ValueError):
    """A lunar degree-three request violated the narrow contract."""


class LunarDegree3DependencyError(LunarDegree3Error):
    """The explicitly requested NumPy force runtime is unavailable."""


def _coefficients(value: object) -> tuple[float, float, float, float, float, float, float]:
    if type(value) is not tuple or len(value) != 7:
        raise LunarDegree3Error("coefficients must be an exact seven-float tuple")
    checked: list[float] = []
    for index, item in enumerate(value):
        try:
            checked.append(lunar_figure._finite_float(item, f"coefficients[{index}]"))
        except lunar_figure.LunarFigureError as exc:
            raise LunarDegree3Error(str(exc)) from exc
    if not any(item != 0.0 for item in checked):
        raise LunarDegree3Error("the registered lunar degree-three field must be nonzero")
    return tuple(checked)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, eq=False)
class LunarStaticDegree3PairForce:
    """Immutable correction for the Moon degree-three figure and one target."""

    source_id: str
    target_id: str
    source_index: int
    target_index: int
    coefficients: tuple[float, float, float, float, float, float, float]
    reference_radius_metres: float
    absolute_start_et: float
    orientation_provider: lunar_figure.LunarOrientationProvider
    orientation_frame: str = "MOON_PA_DE440"
    inertial_frame: str = "J2000"
    model_id: str = LUNAR_STATIC_DEGREE3_PAIR_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        try:
            source_id = lunar_figure._identifier(self.source_id, "source_id")
            target_id = lunar_figure._identifier(self.target_id, "target_id")
            source_index = lunar_figure._index(self.source_index, "source_index")
            target_index = lunar_figure._index(self.target_index, "target_index")
            lunar_figure._positive_float(
                self.reference_radius_metres, "reference_radius_metres"
            )
            lunar_figure._finite_float(self.absolute_start_et, "absolute_start_et")
        except lunar_figure.LunarFigureError as exc:
            raise LunarDegree3Error(str(exc)) from exc
        if source_id != "MOON":
            raise LunarDegree3Error("the lunar-figure source_id must be exact MOON")
        if target_id == source_id or target_index == source_index:
            raise LunarDegree3Error("source and target must be distinct")
        _coefficients(self.coefficients)
        if not callable(self.orientation_provider):
            raise LunarDegree3Error("orientation_provider must be callable")
        if self.orientation_frame != "MOON_PA_DE440" or self.inertial_frame != "J2000":
            raise LunarDegree3Error("orientation frame contract changed")
        if self.model_id != LUNAR_STATIC_DEGREE3_PAIR_MODEL_ID:
            raise LunarDegree3Error("model_id is not the registered degree-three force")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarDegree3Error("degree-three output must remain MODEL_OUTPUT")
        for observed, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(observed) is not bool or observed:
                raise LunarDegree3Error(f"{label} must be exact false")


def prepare_lunar_static_degree3_pair_force(
    *,
    source_id: str,
    target_id: str,
    source_index: int,
    target_index: int,
    absolute_start_et: float,
    orientation_provider: lunar_figure.LunarOrientationProvider,
    coefficients: tuple[float, float, float, float, float, float, float] = (
        DE440_TECH_COMMENTS_LUNAR_DEGREE3
    ),
    reference_radius_metres: float = (
        lunar_figure.DE440_TECH_COMMENTS_LUNAR_REFERENCE_RADIUS_METRES
    ),
) -> LunarStaticDegree3PairForce:
    """Prepare one constant degree-three lunar-figure pair correction."""

    return LunarStaticDegree3PairForce(
        source_id=source_id,
        target_id=target_id,
        source_index=source_index,
        target_index=target_index,
        coefficients=_coefficients(coefficients),
        reference_radius_metres=reference_radius_metres,
        absolute_start_et=absolute_start_et,
        orientation_provider=orientation_provider,
    )


def _degree3_body_acceleration(
    relative_body: object,
    source_gm: float,
    reference_radius: float,
    coefficients: tuple[float, float, float, float, float, float, float],
):
    """Return point-target acceleration in the lunar principal-axis frame."""

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarDegree3DependencyError(
            "lunar degree-three evaluation requires NumPy"
        ) from exc

    x = float(relative_body[0])
    y = float(relative_body[1])
    z = float(relative_body[2])
    x2 = x * x
    y2 = y * y
    z2 = z * z
    radius_squared = x2 + y2 + z2
    if not math.isfinite(radius_squared) or radius_squared <= 0.0:
        raise LunarDegree3Error("lunar source and point target must be separated")
    radius = math.sqrt(radius_squared)
    j3, c31, s31, c32, s32, c33, s33 = coefficients

    linear31 = c31 * x + s31 * y
    shape31 = 4.0 * z2 - x2 - y2
    quadratic32 = c32 * (x2 - y2) + 2.0 * s32 * x * y
    harmonic = (
        0.5 * j3 * z * (3.0 * x2 + 3.0 * y2 - 2.0 * z2)
        + 1.5 * shape31 * linear31
        + 15.0 * z * quadratic32
        + 15.0
        * (
            c33 * (x * x2 - 3.0 * x * y2)
            + s33 * (3.0 * x2 * y - y * y2)
        )
    )
    gradient = np.ascontiguousarray(
        [
            3.0 * j3 * x * z
            + 1.5 * (-2.0 * x * linear31 + shape31 * c31)
            + 30.0 * z * (c32 * x + s32 * y)
            + 45.0 * (c33 * (x2 - y2) + 2.0 * s33 * x * y),
            3.0 * j3 * y * z
            + 1.5 * (-2.0 * y * linear31 + shape31 * s31)
            + 30.0 * z * (-c32 * y + s32 * x)
            + 45.0 * (-2.0 * c33 * x * y + s33 * (x2 - y2)),
            1.5 * j3 * (x2 + y2 - 2.0 * z2)
            + 12.0 * z * linear31
            + 15.0 * quadratic32,
        ],
        dtype=np.float64,
    )
    scale = source_gm * reference_radius * reference_radius * reference_radius
    inverse_radius7 = 1.0 / (
        radius_squared * radius_squared * radius_squared * radius
    )
    acceleration = scale * inverse_radius7 * (
        gradient
        - (7.0 * harmonic / radius_squared)
        * np.ascontiguousarray([x, y, z], dtype=np.float64)
    )
    result = np.ascontiguousarray(acceleration, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise LunarDegree3Error("lunar degree-three acceleration became nonfinite")
    return result


def evaluate_lunar_static_degree3_pair_correction(
    force: LunarStaticDegree3PairForce,
    *,
    body_ids: tuple[str, ...],
    positions: object,
    gravitational_parameters: object,
    elapsed_time: float = 0.0,
):
    """Return the correction-only acceleration for the registered pair."""

    if type(force) is not LunarStaticDegree3PairForce:
        raise LunarDegree3Error("force must be an exact LunarStaticDegree3PairForce")
    if type(body_ids) is not tuple or not body_ids:
        raise LunarDegree3Error("body_ids must be a nonempty exact tuple")
    try:
        checked_ids = tuple(
            lunar_figure._identifier(body_id, f"body_ids[{index}]")
            for index, body_id in enumerate(body_ids)
        )
    except lunar_figure.LunarFigureError as exc:
        raise LunarDegree3Error(str(exc)) from exc
    if len(set(checked_ids)) != len(checked_ids):
        raise LunarDegree3Error("body_ids must be unique")
    if force.source_index >= len(checked_ids) or force.target_index >= len(checked_ids):
        raise LunarDegree3Error("configured pair index is outside body_ids")
    if checked_ids[force.source_index] != force.source_id or checked_ids[force.target_index] != force.target_id:
        raise LunarDegree3Error("configured pair indices do not match body_ids")

    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarDegree3DependencyError(
            "lunar degree-three evaluation requires NumPy"
        ) from exc
    if type(positions) is not np.ndarray or positions.dtype != np.dtype("float64") or positions.shape != (len(checked_ids), 3) or not positions.flags.c_contiguous or not np.all(np.isfinite(positions)):
        raise LunarDegree3Error("positions must be finite contiguous float64 with shape (body_count,3)")
    if type(gravitational_parameters) is not np.ndarray or gravitational_parameters.dtype != np.dtype("float64") or gravitational_parameters.shape != (len(checked_ids),) or not gravitational_parameters.flags.c_contiguous or not np.all(np.isfinite(gravitational_parameters)) or not np.all(gravitational_parameters > 0.0):
        raise LunarDegree3Error("gravitational_parameters must be finite positive contiguous float64")
    try:
        elapsed_time = lunar_figure._finite_float(elapsed_time, "elapsed_time")
    except lunar_figure.LunarFigureError as exc:
        raise LunarDegree3Error(str(exc)) from exc
    absolute_et = force.absolute_start_et + elapsed_time
    if not math.isfinite(absolute_et):
        raise LunarDegree3Error("absolute orientation epoch became nonfinite")
    try:
        rotation = lunar_figure._rotation_matrix(force.orientation_provider(absolute_et))
    except lunar_figure.LunarFigureError as exc:
        raise LunarDegree3Error(str(exc)) from exc
    relative_inertial = np.ascontiguousarray(
        positions[force.target_index] - positions[force.source_index],
        dtype=np.float64,
    )
    relative_body = np.ascontiguousarray(rotation @ relative_inertial, dtype=np.float64)
    source_gm = float(gravitational_parameters[force.source_index])
    target_gm = float(gravitational_parameters[force.target_index])
    target_body_acceleration = _degree3_body_acceleration(
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
        raise LunarDegree3Error("lunar degree-three correction became nonfinite")
    return np.ascontiguousarray(result, dtype=np.float64)


__all__ = [
    "DE440_TECH_COMMENTS_LUNAR_C31",
    "DE440_TECH_COMMENTS_LUNAR_C32",
    "DE440_TECH_COMMENTS_LUNAR_C33",
    "DE440_TECH_COMMENTS_LUNAR_DEGREE3",
    "DE440_TECH_COMMENTS_LUNAR_J3",
    "DE440_TECH_COMMENTS_LUNAR_S31",
    "DE440_TECH_COMMENTS_LUNAR_S32",
    "DE440_TECH_COMMENTS_LUNAR_S33",
    "LUNAR_STATIC_DEGREE3_PAIR_MODEL_ID",
    "LunarDegree3DependencyError",
    "LunarDegree3Error",
    "LunarStaticDegree3PairForce",
    "evaluate_lunar_static_degree3_pair_correction",
    "prepare_lunar_static_degree3_pair_force",
]
