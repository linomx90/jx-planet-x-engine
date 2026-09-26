"""Provenance-bound adapters for supported Solar-System force kernels.

The underlying Earth and lunar kernels predate :class:`ForcePlan`.  These
immutable adapters promote them into the unified engine without copying their
equations or weakening the engine's unit, validity, and provenance contracts.
They are intentionally NumPy/CPU-only until equivalent device-native kernels
exist; requesting another backend fails closed in the evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass

from jxplanetx.solar_system.earth_zonal import (
    EARTH_ZONAL_PAIR_MODEL_ID,
    EarthZonalPairForce,
)
from jxplanetx.solar_system.lunar_figure import (
    LUNAR_STATIC_DEGREE2_PAIR_MODEL_ID,
    LunarStaticDegree2PairForce,
)
from jxplanetx.solar_system.lunar_figure_degree3 import (
    LUNAR_STATIC_DEGREE3_PAIR_MODEL_ID,
    LunarStaticDegree3PairForce,
)

from .contracts import ContractError, ParameterMetadata, _metadata, _text


EARTH_ZONAL_PARAMETER_IDS = (
    "reference_radius",
    "zonal_coefficients",
    "earth_pole_model",
)
LUNAR_DEGREE2_PARAMETER_IDS = (
    "reference_radius",
    "degree2_coefficients",
    "lunar_orientation_model",
)
LUNAR_DEGREE3_PARAMETER_IDS = (
    "reference_radius",
    "degree3_coefficients",
    "lunar_orientation_model",
)


def _provider_id(value: object, label: str) -> str:
    checked = _text(value, label)
    if len(checked) > 256:
        raise ContractError(f"{label} must contain at most 256 characters")
    return checked


def _require_non_authorizing_force(force: object, label: str) -> None:
    if (
        getattr(force, "evidence_class", None) != "MODEL_OUTPUT"
        or getattr(force, "registry_authorized", None) is not False
        or getattr(force, "qualification_authorized", None) is not False
    ):
        raise ContractError(f"{label} must remain nonauthorizing MODEL_OUTPUT")


@dataclass(frozen=True, slots=True, eq=False)
class EarthZonalJ2J5Force:
    """One bound Earth J2--J5 pair correction for the unified engine."""

    MODEL_ID = EARTH_ZONAL_PAIR_MODEL_ID

    force: EarthZonalPairForce
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]
    pole_provider_id: str

    def __post_init__(self) -> None:
        if type(self.force) is not EarthZonalPairForce:
            raise ContractError("force must be an exact EarthZonalPairForce")
        _text(self.unit_system_id, "unit_system_id")
        _metadata(self.parameter_metadata, EARTH_ZONAL_PARAMETER_IDS)
        _provider_id(self.pole_provider_id, "pole_provider_id")
        _require_non_authorizing_force(self.force, "Earth-zonal force")

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def source_ids(self) -> tuple[str, ...]:
        return (self.force.source_id,)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.force.target_id,)


@dataclass(frozen=True, slots=True, eq=False)
class LunarStaticDegree2Force:
    """One bound static lunar degree-two pair correction."""

    MODEL_ID = LUNAR_STATIC_DEGREE2_PAIR_MODEL_ID

    force: LunarStaticDegree2PairForce
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]
    orientation_provider_id: str

    def __post_init__(self) -> None:
        if type(self.force) is not LunarStaticDegree2PairForce:
            raise ContractError("force must be an exact LunarStaticDegree2PairForce")
        _text(self.unit_system_id, "unit_system_id")
        _metadata(self.parameter_metadata, LUNAR_DEGREE2_PARAMETER_IDS)
        _provider_id(self.orientation_provider_id, "orientation_provider_id")
        _require_non_authorizing_force(self.force, "lunar degree-two force")

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def source_ids(self) -> tuple[str, ...]:
        return (self.force.source_id,)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.force.target_id,)


@dataclass(frozen=True, slots=True, eq=False)
class LunarStaticDegree3Force:
    """One bound static lunar degree-three pair correction."""

    MODEL_ID = LUNAR_STATIC_DEGREE3_PAIR_MODEL_ID

    force: LunarStaticDegree3PairForce
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]
    orientation_provider_id: str

    def __post_init__(self) -> None:
        if type(self.force) is not LunarStaticDegree3PairForce:
            raise ContractError("force must be an exact LunarStaticDegree3PairForce")
        _text(self.unit_system_id, "unit_system_id")
        _metadata(self.parameter_metadata, LUNAR_DEGREE3_PARAMETER_IDS)
        _provider_id(self.orientation_provider_id, "orientation_provider_id")
        _require_non_authorizing_force(self.force, "lunar degree-three force")

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def source_ids(self) -> tuple[str, ...]:
        return (self.force.source_id,)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return (self.force.target_id,)


def bind_earth_zonal_j2_j5_force(
    force: EarthZonalPairForce,
    *,
    unit_system_id: str,
    parameter_metadata: tuple[ParameterMetadata, ...],
    pole_provider_id: str,
) -> EarthZonalJ2J5Force:
    """Bind a validated Earth-zonal kernel to engine metadata."""

    return EarthZonalJ2J5Force(
        force=force,
        unit_system_id=unit_system_id,
        parameter_metadata=parameter_metadata,
        pole_provider_id=pole_provider_id,
    )


def bind_lunar_static_degree2_force(
    force: LunarStaticDegree2PairForce,
    *,
    unit_system_id: str,
    parameter_metadata: tuple[ParameterMetadata, ...],
    orientation_provider_id: str,
) -> LunarStaticDegree2Force:
    """Bind a validated lunar degree-two kernel to engine metadata."""

    return LunarStaticDegree2Force(
        force=force,
        unit_system_id=unit_system_id,
        parameter_metadata=parameter_metadata,
        orientation_provider_id=orientation_provider_id,
    )


def bind_lunar_static_degree3_force(
    force: LunarStaticDegree3PairForce,
    *,
    unit_system_id: str,
    parameter_metadata: tuple[ParameterMetadata, ...],
    orientation_provider_id: str,
) -> LunarStaticDegree3Force:
    """Bind a validated lunar degree-three kernel to engine metadata."""

    return LunarStaticDegree3Force(
        force=force,
        unit_system_id=unit_system_id,
        parameter_metadata=parameter_metadata,
        orientation_provider_id=orientation_provider_id,
    )


def canonical_physical_force_record(model: object) -> dict[str, object]:
    """Return a callable-free record for the RKF78 retained-content digest."""

    if type(model) is EarthZonalJ2J5Force:
        force = model.force
        policy = force.pole_policy
        physical = {
            "source_id": force.source_id,
            "target_id": force.target_id,
            "source_index": force.source_index,
            "target_index": force.target_index,
            "zonal_coefficients": force.zonal_coefficients,
            "reference_radius_metres": force.reference_radius_metres,
            "pole_mode": policy.mode,
            "absolute_start_et": policy.absolute_start_et,
            "raw_start_pole": policy.raw_start_pole,
            "effective_start_pole": policy.effective_start_pole,
            "pole_provider_id": model.pole_provider_id,
        }
    elif type(model) is LunarStaticDegree2Force:
        force = model.force
        physical = {
            "source_id": force.source_id,
            "target_id": force.target_id,
            "source_index": force.source_index,
            "target_index": force.target_index,
            "coefficients": force.coefficients,
            "reference_radius_metres": force.reference_radius_metres,
            "absolute_start_et": force.absolute_start_et,
            "orientation_frame": force.orientation_frame,
            "inertial_frame": force.inertial_frame,
            "orientation_provider_id": model.orientation_provider_id,
        }
    elif type(model) is LunarStaticDegree3Force:
        force = model.force
        physical = {
            "source_id": force.source_id,
            "target_id": force.target_id,
            "source_index": force.source_index,
            "target_index": force.target_index,
            "coefficients": force.coefficients,
            "reference_radius_metres": force.reference_radius_metres,
            "absolute_start_et": force.absolute_start_et,
            "orientation_frame": force.orientation_frame,
            "inertial_frame": force.inertial_frame,
            "orientation_provider_id": model.orientation_provider_id,
        }
    else:
        raise ContractError("model is not a physical-force binding")
    return {
        "model_id": model.model_id,
        "unit_system_id": model.unit_system_id,
        "parameter_metadata": model.parameter_metadata,
        "physical": physical,
    }


__all__ = [
    "EARTH_ZONAL_PARAMETER_IDS",
    "LUNAR_DEGREE2_PARAMETER_IDS",
    "LUNAR_DEGREE3_PARAMETER_IDS",
    "EarthZonalJ2J5Force",
    "LunarStaticDegree2Force",
    "LunarStaticDegree3Force",
    "bind_earth_zonal_j2_j5_force",
    "bind_lunar_static_degree2_force",
    "bind_lunar_static_degree3_force",
    "canonical_physical_force_record",
]
