"""Staged translational-physics increments for coupled lunar screening.

The intended base is a simultaneous planetary point-mass trajectory that
already includes the historical fixed-+Z Earth--Moon J2 term.  Four explicit
arms are provided: unchanged planetary control, EIH 1PN only, dynamic-pole J2
replacement only, and both increments.  The dynamic-pole component is the
dynamic J2 acceleration minus the fixed-+Z J2 acceleration, so adding it to
the base replaces rather than duplicates Earth J2.

Inputs use kilometres, seconds, and km^3/s^2.  The public Earth-J2 primitive
uses metre units, so conversion is performed explicitly around that call.
All outputs remain ``SCREENING_ONLY`` and unregistered.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from . import earth_j2
from . import earth_pole
from . import eih_1pn


PLANETARY_CONTROL = "PLANETARY_CONTROL"
PLANETARY_PLUS_EIH = "PLANETARY_PLUS_EIH_1PN"
PLANETARY_PLUS_DYNAMIC_POLE = "PLANETARY_PLUS_DYNAMIC_EARTH_POLE_J2"
PLANETARY_PLUS_EIH_DYNAMIC_POLE = (
    "PLANETARY_PLUS_EIH_1PN_AND_DYNAMIC_EARTH_POLE_J2"
)
TRANSLATIONAL_PHYSICS_ARMS = (
    PLANETARY_CONTROL,
    PLANETARY_PLUS_EIH,
    PLANETARY_PLUS_DYNAMIC_POLE,
    PLANETARY_PLUS_EIH_DYNAMIC_POLE,
)
LUNAR_TRANSLATIONAL_PHYSICS_MODEL_ID = (
    "solar-system.experimental.lunar-planetary-eih-dynamic-earth-pole.v1"
)
SCIENTIFIC_CLAIM_STATE = "SCREENING_ONLY"


class LunarTranslationalPhysicsError(ValueError):
    """A staged translational increment violated its physical contract."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarTranslationalPhysicsError(
            "lunar translational screening requires NumPy"
        ) from exc
    return np


def _state_matrix(value: object, label: str, *, rows: int | None = None):
    np = _numpy()
    if type(value) is not np.ndarray:
        raise LunarTranslationalPhysicsError(
            f"{label} must be an exact NumPy array"
        )
    if value.dtype != np.dtype("float64") or value.ndim != 2 or value.shape[1] != 3:
        raise LunarTranslationalPhysicsError(
            f"{label} must have float64 shape (N,3)"
        )
    if rows is not None and value.shape[0] != rows:
        raise LunarTranslationalPhysicsError(f"{label} has the wrong body count")
    if not value.flags.c_contiguous or not np.all(np.isfinite(value)):
        raise LunarTranslationalPhysicsError(
            f"{label} must be finite and C-contiguous"
        )
    return value


def _gm_vector(value: object, rows: int):
    np = _numpy()
    if type(value) is not np.ndarray:
        raise LunarTranslationalPhysicsError(
            "gravitational_parameters_km3_s2 must be an exact NumPy array"
        )
    if value.dtype != np.dtype("float64") or value.shape != (rows,):
        raise LunarTranslationalPhysicsError(
            "gravitational_parameters_km3_s2 must have float64 shape (N,)"
        )
    if (
        not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
        or np.any(value <= 0.0)
    ):
        raise LunarTranslationalPhysicsError(
            "gravitational_parameters_km3_s2 must be finite, positive, and C-contiguous"
        )
    return value


def _body_ids(value: object, rows: int) -> tuple[str, ...]:
    if type(value) is not tuple or len(value) != rows:
        raise LunarTranslationalPhysicsError(
            "body_ids must be an exact tuple matching the state"
        )
    checked = []
    for index, item in enumerate(value):
        if type(item) is not str or not item or item.strip() != item:
            raise LunarTranslationalPhysicsError(
                f"body_ids[{index}] must be nonempty trimmed text"
            )
        checked.append(item)
    if len(set(checked)) != len(checked):
        raise LunarTranslationalPhysicsError("body_ids must be unique")
    return tuple(checked)


@dataclass(frozen=True, slots=True, eq=False)
class LunarTranslationalPhysicsParameters:
    """Bound EIH and fixed/dynamic Earth-J2 replacement contracts."""

    eih: eih_1pn.EIH1PNParameters
    fixed_earth_j2: earth_j2.EarthJ2PairForce
    dynamic_earth_j2: earth_j2.EarthJ2PairForce
    model_id: str = LUNAR_TRANSLATIONAL_PHYSICS_MODEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False
    production_ready: bool = False

    def __post_init__(self) -> None:
        if type(self.eih) is not eih_1pn.EIH1PNParameters:
            raise LunarTranslationalPhysicsError(
                "eih must be exact EIH1PNParameters"
            )
        if type(self.fixed_earth_j2) is not earth_j2.EarthJ2PairForce:
            raise LunarTranslationalPhysicsError(
                "fixed_earth_j2 must be exact EarthJ2PairForce"
            )
        if type(self.dynamic_earth_j2) is not earth_j2.EarthJ2PairForce:
            raise LunarTranslationalPhysicsError(
                "dynamic_earth_j2 must be exact EarthJ2PairForce"
            )
        fixed = self.fixed_earth_j2
        dynamic = self.dynamic_earth_j2
        if fixed.pole_mode != earth_pole.FIXED_J2000_POSITIVE_Z:
            raise LunarTranslationalPhysicsError(
                "fixed_earth_j2 must use the fixed J2000 +Z pole"
            )
        if dynamic.pole_mode != earth_pole.DYNAMIC:
            raise LunarTranslationalPhysicsError(
                "dynamic_earth_j2 must use a dynamic pole provider"
            )
        matching = (
            fixed.source_id == dynamic.source_id == "EARTH"
            and fixed.target_id == dynamic.target_id == "MOON"
            and fixed.source_index == dynamic.source_index
            and fixed.target_index == dynamic.target_index
            and fixed.j2 == dynamic.j2
            and fixed.reference_radius_metres == dynamic.reference_radius_metres
        )
        if not matching:
            raise LunarTranslationalPhysicsError(
                "fixed and dynamic Earth-J2 contracts must differ only by pole policy"
            )
        if self.model_id != LUNAR_TRANSLATIONAL_PHYSICS_MODEL_ID:
            raise LunarTranslationalPhysicsError("model_id changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise LunarTranslationalPhysicsError("scientific_claim_state changed")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
            (self.production_ready, "production_ready"),
        ):
            if type(value) is not bool or value:
                raise LunarTranslationalPhysicsError(f"{label} must be exact false")


@dataclass(frozen=True, slots=True, eq=False)
class LunarTranslationalPhysicsEvaluation:
    """Owned component accelerations for one selected protocol arm."""

    arm: str
    total_correction_km_s2: object
    eih_correction_km_s2: object
    dynamic_minus_fixed_earth_j2_km_s2: object
    maximum_eih_compactness: float
    maximum_eih_speed_fraction_squared: float
    model_id: str = LUNAR_TRANSLATIONAL_PHYSICS_MODEL_ID
    scientific_claim_state: str = SCIENTIFIC_CLAIM_STATE
    registry_authorized: bool = False
    qualification_authorized: bool = False
    production_ready: bool = False

    def __post_init__(self) -> None:
        np = _numpy()
        if self.arm not in TRANSLATIONAL_PHYSICS_ARMS:
            raise LunarTranslationalPhysicsError("arm is not registered")
        total = _state_matrix(self.total_correction_km_s2, "total_correction_km_s2")
        rows = total.shape[0]
        owned = []
        for value, label in (
            (total, "total_correction_km_s2"),
            (self.eih_correction_km_s2, "eih_correction_km_s2"),
            (
                self.dynamic_minus_fixed_earth_j2_km_s2,
                "dynamic_minus_fixed_earth_j2_km_s2",
            ),
        ):
            exact = _state_matrix(value, label, rows=rows)
            copied = np.array(exact, dtype=np.float64, order="C", copy=True)
            copied[copied == 0.0] = 0.0
            copied.setflags(write=False)
            owned.append(copied)
        object.__setattr__(self, "total_correction_km_s2", owned[0])
        object.__setattr__(self, "eih_correction_km_s2", owned[1])
        object.__setattr__(
            self,
            "dynamic_minus_fixed_earth_j2_km_s2",
            owned[2],
        )
        for value, label in (
            (self.maximum_eih_compactness, "maximum_eih_compactness"),
            (
                self.maximum_eih_speed_fraction_squared,
                "maximum_eih_speed_fraction_squared",
            ),
        ):
            if type(value) is not float or not math.isfinite(value) or value < 0.0:
                raise LunarTranslationalPhysicsError(
                    f"{label} must be a nonnegative finite float"
                )
        if self.model_id != LUNAR_TRANSLATIONAL_PHYSICS_MODEL_ID:
            raise LunarTranslationalPhysicsError("evaluation model_id changed")
        if self.scientific_claim_state != SCIENTIFIC_CLAIM_STATE:
            raise LunarTranslationalPhysicsError(
                "evaluation scientific_claim_state changed"
            )
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
            (self.production_ready, "production_ready"),
        ):
            if type(value) is not bool or value:
                raise LunarTranslationalPhysicsError(
                    f"evaluation {label} must be exact false"
                )


def _earth_j2_metres_to_kilometres(
    force: earth_j2.EarthJ2PairForce,
    body_ids: tuple[str, ...],
    positions_km,
    gm_km3_s2,
    elapsed_time_seconds: float,
):
    np = _numpy()
    positions_metres = np.ascontiguousarray(positions_km * 1.0e3, dtype=np.float64)
    gm_metres = np.ascontiguousarray(gm_km3_s2 * 1.0e9, dtype=np.float64)
    correction_metres = earth_j2.evaluate_earth_j2_pair_correction(
        force,
        body_ids=body_ids,
        positions=positions_metres,
        gravitational_parameters=gm_metres,
        elapsed_time=elapsed_time_seconds,
    )
    return np.ascontiguousarray(correction_metres / 1.0e3, dtype=np.float64)


def evaluate_lunar_translational_physics(
    parameters: LunarTranslationalPhysicsParameters,
    *,
    arm: str,
    body_ids: object,
    positions_km: object,
    velocities_km_s: object,
    gravitational_parameters_km3_s2: object,
    elapsed_time_seconds: float,
) -> LunarTranslationalPhysicsEvaluation:
    """Evaluate one preregistrable correction arm without changing the base."""

    np = _numpy()
    if type(parameters) is not LunarTranslationalPhysicsParameters:
        raise LunarTranslationalPhysicsError(
            "parameters must be exact LunarTranslationalPhysicsParameters"
        )
    if type(arm) is not str or arm not in TRANSLATIONAL_PHYSICS_ARMS:
        raise LunarTranslationalPhysicsError("arm is not registered")
    if type(elapsed_time_seconds) is not float or not math.isfinite(
        elapsed_time_seconds
    ):
        raise LunarTranslationalPhysicsError(
            "elapsed_time_seconds must be a finite built-in float"
        )
    positions = _state_matrix(positions_km, "positions_km")
    velocities = _state_matrix(
        velocities_km_s,
        "velocities_km_s",
        rows=positions.shape[0],
    )
    gm = _gm_vector(gravitational_parameters_km3_s2, positions.shape[0])
    ids = _body_ids(body_ids, positions.shape[0])
    fixed = parameters.fixed_earth_j2
    if (
        ids[fixed.source_index] != fixed.source_id
        or ids[fixed.target_index] != fixed.target_id
    ):
        raise LunarTranslationalPhysicsError(
            "body_ids do not match the bound Earth-Moon indices"
        )

    zero = np.zeros_like(positions, dtype=np.float64)
    eih_component = zero
    maximum_compactness = 0.0
    maximum_speed_fraction = 0.0
    if arm in (PLANETARY_PLUS_EIH, PLANETARY_PLUS_EIH_DYNAMIC_POLE):
        try:
            eih_result = eih_1pn.evaluate_eih_1pn_correction(
                positions,
                velocities,
                gm,
                parameters.eih,
            )
        except eih_1pn.EIH1PNError as exc:
            raise LunarTranslationalPhysicsError(str(exc)) from exc
        eih_component = np.ascontiguousarray(
            eih_result.correction_accelerations_km_s2,
            dtype=np.float64,
        )
        maximum_compactness = eih_result.maximum_compactness
        maximum_speed_fraction = eih_result.maximum_speed_fraction_squared

    pole_component = zero
    if arm in (
        PLANETARY_PLUS_DYNAMIC_POLE,
        PLANETARY_PLUS_EIH_DYNAMIC_POLE,
    ):
        try:
            dynamic = _earth_j2_metres_to_kilometres(
                parameters.dynamic_earth_j2,
                ids,
                positions,
                gm,
                elapsed_time_seconds,
            )
            fixed_value = _earth_j2_metres_to_kilometres(
                parameters.fixed_earth_j2,
                ids,
                positions,
                gm,
                elapsed_time_seconds,
            )
        except earth_j2.EarthJ2Error as exc:
            raise LunarTranslationalPhysicsError(str(exc)) from exc
        pole_component = np.ascontiguousarray(dynamic - fixed_value)

    return LunarTranslationalPhysicsEvaluation(
        arm=arm,
        total_correction_km_s2=np.ascontiguousarray(
            eih_component + pole_component
        ),
        eih_correction_km_s2=eih_component,
        dynamic_minus_fixed_earth_j2_km_s2=pole_component,
        maximum_eih_compactness=float(maximum_compactness),
        maximum_eih_speed_fraction_squared=float(maximum_speed_fraction),
    )


__all__ = [
    "LUNAR_TRANSLATIONAL_PHYSICS_MODEL_ID",
    "LunarTranslationalPhysicsError",
    "LunarTranslationalPhysicsEvaluation",
    "LunarTranslationalPhysicsParameters",
    "PLANETARY_CONTROL",
    "PLANETARY_PLUS_DYNAMIC_POLE",
    "PLANETARY_PLUS_EIH",
    "PLANETARY_PLUS_EIH_DYNAMIC_POLE",
    "SCIENTIFIC_CLAIM_STATE",
    "TRANSLATIONAL_PHYSICS_ARMS",
    "evaluate_lunar_translational_physics",
]
