"""Inspectable engine-native force ABI registry.

This registry is the single exact-type and ordering roster consumed by the
unified force evaluator.  It describes execution compatibility; it does not
turn Python callables into a C ABI and does not authorize scientific claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .contracts import (
    CannonballSRP,
    MutualEIH1PN,
    NewtonianPointMass,
    RestrictedStaticCentral1PN,
)
from .physical_forces import (
    EarthZonalJ2J5Force,
    LunarStaticDegree2Force,
    LunarStaticDegree3Force,
)


FORCE_ABI_VERSION = 1


@dataclass(frozen=True, slots=True)
class ForceABISpec:
    """Immutable dispatch and compatibility identity for one force type."""

    model_id: str
    config_type: type
    canonical_order: int
    supported_backend_ids: tuple[str, ...]
    compatible_integrator_classes: tuple[str, ...]
    acceleration_semantics: str


_SPECS = (
    ForceABISpec(
        NewtonianPointMass.MODEL_ID,
        NewtonianPointMass,
        0,
        ("numpy", "cupy"),
        ("GENERAL_FIRST_ORDER", "POSITION_FORCE_KICK_DRIFT"),
        "BASE_ACCELERATION",
    ),
    ForceABISpec(
        RestrictedStaticCentral1PN.MODEL_ID,
        RestrictedStaticCentral1PN,
        10,
        ("numpy", "cupy"),
        ("GENERAL_FIRST_ORDER", "IMPLICIT_VELOCITY_DEPENDENT"),
        "ACCELERATION_CORRECTION",
    ),
    ForceABISpec(
        MutualEIH1PN.MODEL_ID,
        MutualEIH1PN,
        10,
        ("numpy", "cupy"),
        ("GENERAL_FIRST_ORDER", "IMPLICIT_VELOCITY_DEPENDENT"),
        "ACCELERATION_CORRECTION",
    ),
    ForceABISpec(
        EarthZonalJ2J5Force.MODEL_ID,
        EarthZonalJ2J5Force,
        20,
        ("numpy",),
        ("GENERAL_FIRST_ORDER",),
        "ACCELERATION_CORRECTION",
    ),
    ForceABISpec(
        LunarStaticDegree2Force.MODEL_ID,
        LunarStaticDegree2Force,
        20,
        ("numpy",),
        ("GENERAL_FIRST_ORDER",),
        "ACCELERATION_CORRECTION",
    ),
    ForceABISpec(
        LunarStaticDegree3Force.MODEL_ID,
        LunarStaticDegree3Force,
        20,
        ("numpy",),
        ("GENERAL_FIRST_ORDER",),
        "ACCELERATION_CORRECTION",
    ),
    ForceABISpec(
        CannonballSRP.MODEL_ID,
        CannonballSRP,
        30,
        ("numpy", "cupy"),
        ("GENERAL_FIRST_ORDER",),
        "ACCELERATION_CORRECTION",
    ),
)

FORCE_ABI_REGISTRY: Mapping[str, ForceABISpec] = MappingProxyType(
    {spec.model_id: spec for spec in _SPECS}
)
_TYPE_REGISTRY: Mapping[type, ForceABISpec] = MappingProxyType(
    {spec.config_type: spec for spec in _SPECS}
)


def list_force_abis() -> tuple[ForceABISpec, ...]:
    """Return the stable force ABI roster in canonical registration order."""

    return _SPECS


def get_force_abi(model_or_id: object) -> ForceABISpec:
    """Resolve an exact config instance/type or exact model identifier."""

    if type(model_or_id) is str:
        try:
            return FORCE_ABI_REGISTRY[model_or_id]
        except KeyError as exc:
            raise KeyError(f"unknown force ABI model_id {model_or_id!r}") from exc
    config_type = model_or_id if type(model_or_id) is type else type(model_or_id)
    try:
        return _TYPE_REGISTRY[config_type]
    except KeyError as exc:
        raise KeyError(
            f"unsupported force ABI config type {config_type.__name__!r}"
        ) from exc


__all__ = [
    "FORCE_ABI_REGISTRY",
    "FORCE_ABI_VERSION",
    "ForceABISpec",
    "get_force_abi",
    "list_force_abis",
]
