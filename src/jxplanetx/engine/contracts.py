"""Immutable, fail-closed contracts for the public experimental JX engine.

The contracts in this module describe inputs; they do not grant scientific or
registry authority.  Physical constants are always caller supplied.  The
initial numerical surface is deliberately restricted to binary64 arithmetic.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar


Vec3 = tuple[float, float, float]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContractError(ValueError):
    """An engine input is incomplete, ambiguous, or internally inconsistent."""


class CapabilityUnavailableError(RuntimeError):
    """A declared capability was requested without an implementation."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ContractError(f"{label} must be a nonempty, trimmed string")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{label} must be a finite real number")
    if positive and result <= 0.0:
        raise ContractError(f"{label} must be positive")
    return result


def _vec3(value: object, label: str) -> Vec3:
    if type(value) is not tuple or len(value) != 3:
        raise ContractError(f"{label} must be an immutable 3-tuple")
    return tuple(_number(component, f"{label}[{index}]") for index, component in enumerate(value))  # type: ignore[return-value]


def _ids(values: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if type(values) is not tuple or (not values and not allow_empty):
        qualifier = "an immutable tuple" if allow_empty else "a nonempty immutable tuple"
        raise ContractError(f"{label} must be {qualifier}")
    checked = tuple(_text(value, f"{label}[{index}]") for index, value in enumerate(values))
    if len(set(checked)) != len(checked):
        raise ContractError(f"{label} must not contain duplicates")
    return checked


@dataclass(frozen=True)
class Provenance:
    """Identity of the source from which a state or parameter set was derived."""

    source_id: str
    citation: str
    version: str
    sha256: str

    def __post_init__(self) -> None:
        _text(self.source_id, "source_id")
        _text(self.citation, "citation")
        _text(self.version, "version")
        if type(self.sha256) is not str or _SHA256.fullmatch(self.sha256) is None:
            raise ContractError("sha256 must be a lowercase SHA-256 digest")
        if self.sha256 == "0" * 64:
            raise ContractError("sha256 cannot be an all-zero placeholder")


@dataclass(frozen=True)
class ParameterMetadata:
    """Units, uncertainty, covariance, provenance, and validity for a parameter."""

    parameter_id: str
    units: str
    provenance: Provenance
    uncertainty: float | None
    covariance_group: str | None
    validity_start: float
    validity_end: float

    def __post_init__(self) -> None:
        _text(self.parameter_id, "parameter_id")
        _text(self.units, "units")
        if type(self.provenance) is not Provenance:
            raise ContractError("provenance must be Provenance")
        if self.uncertainty is not None and _number(self.uncertainty, "uncertainty") < 0.0:
            raise ContractError("uncertainty cannot be negative")
        if self.covariance_group is not None:
            _text(self.covariance_group, "covariance_group")
        start = _number(self.validity_start, "validity_start")
        end = _number(self.validity_end, "validity_end")
        if start > end:
            raise ContractError("validity_start cannot exceed validity_end")


@dataclass(frozen=True)
class BackendSpec:
    """Requested array backend and arithmetic policy; fallback is forbidden."""

    backend_id: str
    device: str
    tile_size: int
    dtype: str = "float64"
    allow_fallback: bool = False
    deterministic_reductions: bool = True
    fast_math: bool = False
    determinism_scope: str = "SAME_RUNTIME_DEVICE"

    def __post_init__(self) -> None:
        _text(self.backend_id, "backend_id")
        _text(self.device, "device")
        if type(self.tile_size) is not int or self.tile_size <= 0:
            raise ContractError("tile_size must be a positive integer")
        if type(self.dtype) is not str or self.dtype != "float64":
            raise ContractError("the initial JX engine supports only float64")
        if type(self.allow_fallback) is not bool or self.allow_fallback:
            raise ContractError("backend fallback must remain explicitly disabled")
        if (
            type(self.deterministic_reductions) is not bool
            or not self.deterministic_reductions
        ):
            raise ContractError(
                "deterministic_reductions must be enabled in the initial engine"
            )
        if type(self.fast_math) is not bool or self.fast_math:
            raise ContractError("fast_math must remain disabled in the initial engine")
        if self.determinism_scope != "SAME_RUNTIME_DEVICE":
            raise ContractError(
                "the initial determinism claim is scoped only to SAME_RUNTIME_DEVICE"
            )


@dataclass(frozen=True, eq=False)
class StateSnapshot:
    """One Cartesian snapshot descriptor with complete context.

    The descriptor is immutable and compares by identity.  Its caller-owned
    backend buffers are intentionally not copied; callers must keep them stable
    for the duration of one evaluation.
    """

    snapshot_id: str
    epoch: float
    time_scale: str
    frame: str
    origin: str
    axes: str
    length_unit: str
    time_unit: str
    mass_unit: str
    unit_system_id: str
    body_ids: tuple[str, ...]
    positions: object
    velocities: object
    gravitational_parameters: object
    masses: object
    radii: object
    massive: object
    provenance: Provenance

    def __post_init__(self) -> None:
        _text(self.snapshot_id, "snapshot_id")
        _number(self.epoch, "epoch")
        for label in (
            "time_scale",
            "frame",
            "origin",
            "axes",
            "length_unit",
            "time_unit",
            "mass_unit",
            "unit_system_id",
        ):
            _text(getattr(self, label), label)
        body_ids = _ids(self.body_ids, "body_ids")
        # Numerical buffers deliberately remain opaque here.  Inspecting or
        # materializing them could trigger an implicit GPU-to-host transfer.
        # The selected backend validates ownership, float64 dtype, shape,
        # finiteness, and physical ranges immediately before evaluation.
        for label in (
            "positions",
            "velocities",
            "gravitational_parameters",
            "masses",
            "radii",
            "massive",
        ):
            if getattr(self, label) is None:
                raise ContractError(f"{label} must be a backend-native array")
        if type(self.provenance) is not Provenance:
            raise ContractError("provenance must be Provenance")

    def index_of(self, body_id: str) -> int:
        """Return a body index or fail with a contract-level error."""

        _text(body_id, "body_id")
        try:
            return self.body_ids.index(body_id)
        except ValueError as exc:
            raise ContractError(f"unknown body_id {body_id!r}") from exc


def _metadata(values: object, required_order: tuple[str, ...]) -> tuple[ParameterMetadata, ...]:
    if type(values) is not tuple:
        raise ContractError("parameter_metadata must be an immutable tuple")
    for value in values:
        if type(value) is not ParameterMetadata:
            raise ContractError("parameter_metadata entries must be ParameterMetadata")
    identifiers = tuple(value.parameter_id for value in values)
    if len(set(identifiers)) != len(identifiers):
        raise ContractError("parameter_metadata cannot repeat a parameter_id")
    if identifiers != required_order:
        raise ContractError(
            "parameter_metadata must have exact ordered identifiers "
            f"{required_order!r}; received {identifiers!r}"
        )
    return values


@dataclass(frozen=True)
class NewtonianPointMass:
    """Direct, unsoftened Newtonian point-mass acceleration."""

    MODEL_ID: ClassVar[str] = "force.newtonian.point_mass"
    source_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]

    def __post_init__(self) -> None:
        sources = _ids(self.source_ids, "source_ids")
        targets = _ids(self.target_ids, "target_ids")
        _text(self.unit_system_id, "unit_system_id")
        if not set(sources) | set(targets):
            raise ContractError("Newtonian force configuration cannot be empty")
        _metadata(self.parameter_metadata, ("state.gravitational_parameters",))

    @property
    def model_id(self) -> str:
        return self.MODEL_ID


@dataclass(frozen=True)
class RestrictedStaticCentral1PN:
    """Correction-only Schwarzschild 1PN model about one static source."""

    MODEL_ID: ClassVar[str] = "relativity.solar_schwarzschild_test_particle_1pn"
    central_source_id: str
    target_ids: tuple[str, ...]
    speed_of_light: float
    maximum_compactness: float
    maximum_speed_fraction_squared: float
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]

    def __post_init__(self) -> None:
        source = _text(self.central_source_id, "central_source_id")
        targets = _ids(self.target_ids, "target_ids")
        _text(self.unit_system_id, "unit_system_id")
        if source in targets:
            raise ContractError("the central source cannot be a 1PN target")
        _number(self.speed_of_light, "speed_of_light", positive=True)
        compactness = _number(self.maximum_compactness, "maximum_compactness", positive=True)
        speed_fraction = _number(
            self.maximum_speed_fraction_squared,
            "maximum_speed_fraction_squared",
            positive=True,
        )
        if compactness >= 1.0 or speed_fraction >= 1.0:
            raise ContractError("1PN weak-field and speed bounds must be below one")
        _metadata(
            self.parameter_metadata,
            (
                "speed_of_light",
                "maximum_compactness",
                "maximum_speed_fraction_squared",
            ),
        )

    @property
    def model_id(self) -> str:
        return self.MODEL_ID


@dataclass(frozen=True)
class MutualEIH1PN:
    """Correction-only mutual point-mass EIH 1PN model in barycentric coordinates."""

    MODEL_ID: ClassVar[str] = "force.relativity.eih_1pn_gr"
    body_ids: tuple[str, ...]
    speed_of_light: float
    maximum_compactness: float
    maximum_speed_fraction_squared: float
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]

    def __post_init__(self) -> None:
        bodies = _ids(self.body_ids, "body_ids")
        if len(bodies) < 2:
            raise ContractError("mutual EIH 1PN requires at least two bodies")
        _text(self.unit_system_id, "unit_system_id")
        _number(self.speed_of_light, "speed_of_light", positive=True)
        compactness = _number(
            self.maximum_compactness,
            "maximum_compactness",
            positive=True,
        )
        speed_fraction = _number(
            self.maximum_speed_fraction_squared,
            "maximum_speed_fraction_squared",
            positive=True,
        )
        if compactness >= 1.0 or speed_fraction >= 1.0:
            raise ContractError("EIH 1PN weak-field and speed bounds must be below one")
        _metadata(
            self.parameter_metadata,
            (
                "speed_of_light",
                "maximum_compactness",
                "maximum_speed_fraction_squared",
            ),
        )

    @property
    def model_id(self) -> str:
        return self.MODEL_ID


@dataclass(frozen=True, eq=False)
class CannonballSRP:
    """Unshadowed radial cannonball solar-radiation-pressure correction.

    Per-target buffers compare by configuration identity and must remain stable
    for the duration of evaluation.
    """

    MODEL_ID: ClassVar[str] = "force.nongrav.srp_cannonball"
    radiation_source_id: str
    target_ids: tuple[str, ...]
    reference_pressure: float
    reference_distance: float
    area_to_mass: object
    radiation_pressure_coefficient: object
    coefficient_convention: str
    attitude_model: str
    shadow_model: str
    unit_system_id: str
    parameter_metadata: tuple[ParameterMetadata, ...]

    def __post_init__(self) -> None:
        source = _text(self.radiation_source_id, "radiation_source_id")
        targets = _ids(self.target_ids, "target_ids")
        _text(self.unit_system_id, "unit_system_id")
        if source in targets:
            raise ContractError("the radiation source cannot be an SRP target")
        _number(self.reference_pressure, "reference_pressure", positive=True)
        _number(self.reference_distance, "reference_distance", positive=True)
        for label in ("area_to_mass", "radiation_pressure_coefficient"):
            if getattr(self, label) is None:
                raise ContractError(f"{label} must be a backend-native array aligned with target_ids")
        if self.coefficient_convention not in {"QPR", "CR"}:
            raise ContractError("coefficient_convention must be explicitly QPR or CR")
        if self.attitude_model != "ISOTROPIC_CANNONBALL":
            raise ContractError("the initial SRP implementation requires ISOTROPIC_CANNONBALL")
        if self.shadow_model != "NONE":
            raise ContractError("the initial SRP implementation is explicitly unshadowed")
        _metadata(
            self.parameter_metadata,
            (
                "reference_pressure",
                "reference_distance",
                "area_to_mass",
                "radiation_pressure_coefficient",
            ),
        )

    @property
    def model_id(self) -> str:
        return self.MODEL_ID


ImplementedForceConfig = (
    NewtonianPointMass | RestrictedStaticCentral1PN | MutualEIH1PN | CannonballSRP
)


@dataclass(frozen=True, eq=False)
class ForcePlan:
    """Ordered force composition with immutable nonauthorizing claim controls."""

    plan_id: str
    backend: BackendSpec
    models: tuple[object, ...]
    evidence_class: str = "MODEL_OUTPUT"
    registry_authorized: bool = False
    qualification_authorized: bool = False

    def __post_init__(self) -> None:
        _text(self.plan_id, "plan_id")
        if type(self.backend) is not BackendSpec:
            raise ContractError("backend must be BackendSpec")
        if type(self.models) is not tuple or not self.models:
            raise ContractError("models must be a nonempty immutable ordered tuple")
        model_ids: list[str] = []
        for index, model in enumerate(self.models):
            model_id = getattr(model, "model_id", None)
            if type(model_id) is not str:
                raise ContractError(f"models[{index}] does not expose a model_id")
            model_ids.append(_text(model_id, f"models[{index}].model_id"))
        if len(set(model_ids)) != len(model_ids):
            raise ContractError("a force plan cannot repeat a model_id")
        if self.evidence_class != "MODEL_OUTPUT":
            raise ContractError("the experimental engine emits only MODEL_OUTPUT")
        if type(self.registry_authorized) is not bool or self.registry_authorized:
            raise ContractError("the experimental engine cannot authorize a registry")
        if type(self.qualification_authorized) is not bool or self.qualification_authorized:
            raise ContractError("the experimental engine is unqualified")


__all__ = [
    "BackendSpec",
    "CannonballSRP",
    "CapabilityUnavailableError",
    "ContractError",
    "ForcePlan",
    "ImplementedForceConfig",
    "MutualEIH1PN",
    "NewtonianPointMass",
    "ParameterMetadata",
    "Provenance",
    "RestrictedStaticCentral1PN",
    "StateSnapshot",
    "Vec3",
]
