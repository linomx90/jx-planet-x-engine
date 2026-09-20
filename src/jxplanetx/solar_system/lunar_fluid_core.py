"""Opt-in lunar mantle--fluid-core rotational coupling primitive.

This module promotes the already tested local coupling equations from the JX
lunar research probes into a reusable library boundary.  It does not choose a
core size, inertia, viscosity, initial core rate, or physical interpretation.
Every parameter and every state vector is supplied explicitly by the caller.

The primitive is intentionally not registered in a production force catalog
and is not qualified as a DE440/LLR reproduction.  Its narrow purpose is to
provide one well-validated implementation of the equal-and-opposite CMB
exchange and the coupled mantle/core angular-acceleration equations.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


LUNAR_FLUID_CORE_COUPLING_MODEL_ID = (
    "solar-system.rotation.lunar-mantle-fluid-core-coupling.experimental"
)
MODEL_OUTPUT = "MODEL_OUTPUT"
PARAMETER_STATUS = "CALLER_SUPPLIED_RESEARCH_PARAMETERS_NOT_ENGINE_DEFAULTS"
PHYSICAL_STATUS = "EXPERIMENTAL_PHYSICS_OUTCOME_UNDECIDED"
MANTLE_FRAME = "LUNAR_MANTLE_PRINCIPAL_AXES"
PRESSURE_TORQUE_FORMULA = "JX_AXISYMMETRIC_OBLATE_CORE_FIRST_ORDER_PRESSURE_TORQUE"


class LunarFluidCoreError(ValueError):
    """A lunar mantle--fluid-core request violated its narrow contract."""


class LunarFluidCoreDependencyError(LunarFluidCoreError):
    """The explicitly requested NumPy runtime is unavailable."""


def _numpy():
    try:
        import numpy as np
    except (ImportError, OSError) as exc:
        raise LunarFluidCoreDependencyError(
            "lunar fluid-core evaluation requires NumPy"
        ) from exc
    return np


def _finite_float(value: object, label: str, *, nonnegative: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LunarFluidCoreError(f"{label} must be a finite built-in float")
    if nonnegative and value < 0.0:
        raise LunarFluidCoreError(f"{label} must be nonnegative")
    return value


def _source_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 256
        or value.strip() != value
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        raise LunarFluidCoreError(
            "parameter_source_id must be bounded printable ASCII without spaces"
        )
    return value


def _tuple3(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise LunarFluidCoreError(f"{label} must be an exact three-float tuple")
    checked = tuple(_finite_float(item, f"{label}[{index}]") for index, item in enumerate(value))
    if not all(item > 0.0 for item in checked):
        raise LunarFluidCoreError(f"{label} must contain positive moments")
    return checked  # type: ignore[return-value]


def _vector3(value: object, label: str):
    np = _numpy()
    if type(value) is not np.ndarray or (
        value.dtype != np.dtype("float64")
        or value.shape != (3,)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
    ):
        raise LunarFluidCoreError(
            f"{label} must be finite contiguous float64 with shape (3,)"
        )
    return value


def _symmetric_matrix3(value: object, label: str, *, positive_definite: bool):
    np = _numpy()
    if type(value) is not np.ndarray or (
        value.dtype != np.dtype("float64")
        or value.shape != (3, 3)
        or not value.flags.c_contiguous
        or not np.all(np.isfinite(value))
        or not np.array_equal(value, value.T)
    ):
        raise LunarFluidCoreError(
            f"{label} must be finite symmetric contiguous float64 with shape (3,3)"
        )
    if positive_definite and float(np.min(np.linalg.eigvalsh(value))) <= 0.0:
        raise LunarFluidCoreError(f"{label} must be positive definite")
    return value


@dataclass(frozen=True, slots=True, eq=False)
class LunarFluidCoreCoupling:
    """Immutable, caller-sourced parameters for the experimental CMB exchange."""

    core_inertia_over_mr2: tuple[float, float, float]
    viscous_coefficient_over_mr2_per_second: float
    parameter_source_id: str
    mantle_frame: str = MANTLE_FRAME
    pressure_torque_formula: str = PRESSURE_TORQUE_FORMULA
    parameter_status: str = PARAMETER_STATUS
    physical_status: str = PHYSICAL_STATUS
    model_id: str = LUNAR_FLUID_CORE_COUPLING_MODEL_ID
    evidence_class: str = MODEL_OUTPUT
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        core = _tuple3(self.core_inertia_over_mr2, "core_inertia_over_mr2")
        if not (core[0] == core[1] <= core[2]):
            raise LunarFluidCoreError(
                "core inertia must be axisymmetric and oblate or spherical"
            )
        _finite_float(
            self.viscous_coefficient_over_mr2_per_second,
            "viscous_coefficient_over_mr2_per_second",
            nonnegative=True,
        )
        _source_id(self.parameter_source_id)
        if self.mantle_frame != MANTLE_FRAME:
            raise LunarFluidCoreError("mantle_frame changed")
        if self.pressure_torque_formula != PRESSURE_TORQUE_FORMULA:
            raise LunarFluidCoreError("pressure_torque_formula changed")
        if self.parameter_status != PARAMETER_STATUS:
            raise LunarFluidCoreError("parameter_status changed")
        if self.physical_status != PHYSICAL_STATUS:
            raise LunarFluidCoreError("physical_status changed")
        if self.model_id != LUNAR_FLUID_CORE_COUPLING_MODEL_ID:
            raise LunarFluidCoreError("model_id changed")
        if self.evidence_class != MODEL_OUTPUT:
            raise LunarFluidCoreError("evidence_class changed")
        for value, label in (
            (self.registry_authorized, "registry_authorized"),
            (self.qualification_authorized, "qualification_authorized"),
        ):
            if type(value) is not bool or value:
                raise LunarFluidCoreError(f"{label} must be exact false")


@dataclass(frozen=True, slots=True, eq=False)
class LunarFluidCoreEvaluation:
    """One evaluated rotational derivative and its CMB torque components."""

    mantle_angular_acceleration_body: object
    core_angular_acceleration_body: object
    cmb_torque_on_mantle_body: object
    viscous_torque_on_mantle_body: object
    pressure_torque_on_mantle_body: object


def prepare_lunar_fluid_core_coupling(
    *,
    core_inertia_over_mr2: tuple[float, float, float],
    viscous_coefficient_over_mr2_per_second: float,
    parameter_source_id: str,
) -> LunarFluidCoreCoupling:
    """Create an unregistered, unqualified coupling with explicit provenance."""

    return LunarFluidCoreCoupling(
        core_inertia_over_mr2=core_inertia_over_mr2,
        viscous_coefficient_over_mr2_per_second=(
            viscous_coefficient_over_mr2_per_second
        ),
        parameter_source_id=parameter_source_id,
    )


def lunar_cmb_torque_components(
    coupling: LunarFluidCoreCoupling,
    *,
    mantle_angular_velocity_body: object,
    core_angular_velocity_body: object,
) -> tuple[object, object, object]:
    """Return total, viscous and pressure CMB torque acting on the mantle.

    The equal-and-opposite torque acts on the fluid core.  The relative-power
    identity for the viscous component is nonpositive:
    ``(omega_m-omega_c) dot N_viscous = -K_v |omega_m-omega_c|^2``.
    """

    if type(coupling) is not LunarFluidCoreCoupling:
        raise LunarFluidCoreError("coupling must be an exact LunarFluidCoreCoupling")
    np = _numpy()
    wm = _vector3(mantle_angular_velocity_body, "mantle_angular_velocity_body")
    wc = _vector3(core_angular_velocity_body, "core_angular_velocity_body")
    core = np.asarray(coupling.core_inertia_over_mr2, dtype=np.float64)
    kv = coupling.viscous_coefficient_over_mr2_per_second
    viscous = np.ascontiguousarray(kv * (wc - wm), dtype=np.float64)
    z_axis = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    pressure = np.ascontiguousarray(
        (core[2] - core[0]) * wm[2] * np.cross(z_axis, wc),
        dtype=np.float64,
    )
    total = np.ascontiguousarray(viscous + pressure, dtype=np.float64)
    for value in (total, viscous, pressure):
        value[value == 0.0] = 0.0
        if not np.all(np.isfinite(value)):
            raise LunarFluidCoreError("CMB torque became nonfinite")
    return total, viscous, pressure


def lunar_mantle_core_angular_accelerations(
    coupling: LunarFluidCoreCoupling,
    *,
    mantle_angular_velocity_body: object,
    core_angular_velocity_body: object,
    mantle_inertia_over_mr2: object,
    mantle_inertia_rate_over_mr2_per_second: object,
    external_torque_over_mr2: object,
    geodetic_precession_body: object,
) -> LunarFluidCoreEvaluation:
    """Evaluate the coupled mantle and fluid-core rotational derivatives.

    All vectors and tensors use the lunar mantle principal-axis frame.  The
    mantle inertia may include separately computed tide/spin deformation and
    its supplied time derivative.  This function does not own delay history,
    ephemerides, quaternion integration, or any reference scorer.
    """

    if type(coupling) is not LunarFluidCoreCoupling:
        raise LunarFluidCoreError("coupling must be an exact LunarFluidCoreCoupling")
    np = _numpy()
    wm = _vector3(mantle_angular_velocity_body, "mantle_angular_velocity_body")
    wc = _vector3(core_angular_velocity_body, "core_angular_velocity_body")
    inertia = _symmetric_matrix3(
        mantle_inertia_over_mr2,
        "mantle_inertia_over_mr2",
        positive_definite=True,
    )
    inertia_rate = _symmetric_matrix3(
        mantle_inertia_rate_over_mr2_per_second,
        "mantle_inertia_rate_over_mr2_per_second",
        positive_definite=False,
    )
    external = _vector3(external_torque_over_mr2, "external_torque_over_mr2")
    geodetic = _vector3(geodetic_precession_body, "geodetic_precession_body")
    core = np.asarray(coupling.core_inertia_over_mr2, dtype=np.float64)
    cmb, viscous, pressure = lunar_cmb_torque_components(
        coupling,
        mantle_angular_velocity_body=wm,
        core_angular_velocity_body=wc,
    )
    relative_frame_rate = np.ascontiguousarray(wm - geodetic, dtype=np.float64)
    mantle_rhs = np.ascontiguousarray(
        external
        - inertia_rate @ wm
        - np.cross(relative_frame_rate, inertia @ wm)
        + cmb,
        dtype=np.float64,
    )
    mantle_acceleration = np.ascontiguousarray(
        np.linalg.solve(inertia, mantle_rhs), dtype=np.float64
    )
    core_acceleration = np.ascontiguousarray(
        (-np.cross(relative_frame_rate, core * wc) - cmb) / core,
        dtype=np.float64,
    )
    if not np.all(np.isfinite(mantle_acceleration)) or not np.all(
        np.isfinite(core_acceleration)
    ):
        raise LunarFluidCoreError("coupled angular acceleration became nonfinite")
    return LunarFluidCoreEvaluation(
        mantle_angular_acceleration_body=mantle_acceleration,
        core_angular_acceleration_body=core_acceleration,
        cmb_torque_on_mantle_body=cmb,
        viscous_torque_on_mantle_body=viscous,
        pressure_torque_on_mantle_body=pressure,
    )


__all__ = [
    "LUNAR_FLUID_CORE_COUPLING_MODEL_ID",
    "LunarFluidCoreCoupling",
    "LunarFluidCoreDependencyError",
    "LunarFluidCoreError",
    "LunarFluidCoreEvaluation",
    "lunar_cmb_torque_components",
    "lunar_mantle_core_angular_accelerations",
    "prepare_lunar_fluid_core_coupling",
]
