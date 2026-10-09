"""Sealed declarations of provider-native geometric Cartesian state values.

This unpublished contract stores caller-supplied binary64 values in the exact
target order, frame, epoch, and native unit system declared by an M1
``EphemerisQuerySpec``.  It does not execute a provider, open an artifact,
validate a target/observer segment chain, convert units or time scales,
translate an origin, rotate axes, recenter a system, or prepare engine arrays.

The retained digest supplies unauthenticated content integrity only.  A future
artifact-at-use receipt and provider-specific semantic replay are necessary
but not sufficient to turn this caller-supplied batch into provider execution
evidence; a future provider execution receipt is also required.  This contract
does not prove target-specific coverage, interpolation or physical-accuracy
bounds, enforce network/fallback/extrapolation policy, or supply license,
registry, or qualification authority.  In particular, the query epoch's
coordinate unit is not the unit of the velocity derivative: velocity is
expressed per the provider-native output unit system's time unit.  For query
target ``i``, the slices
``positions[3*i:3*i+3]`` and ``velocities[3*i:3*i+3]`` are its X/Y/Z values;
``(N, 3)`` is the logical shape of each separately flattened block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import ClassVar

from .contracts import (
    MAXIMUM_BODY_COUNT,
    EphemerisQuerySpec,
    SolarSystemContractError,
    validate_integrity,
)
from .serialization import (
    domain_sha256,
    validate_sha256,
)


_STATE_STAGE = (
    "DECLARED_PROVIDER_NATIVE_GEOMETRIC_TARGET_RELATIVE_OBSERVER_STATE_"
    "NO_JX_CONVERSION"
)
_EXECUTION_EVIDENCE_STATUS = (
    "CALLER_SUPPLIED_UNAUTHENTICATED_NO_EXECUTION_RECEIPT"
)
_COMPONENT_ORDER = ("X", "Y", "Z")
_VELOCITY_SEMANTICS = "COORDINATE_DERIVATIVE_PER_PROVIDER_NATIVE_TIME_UNIT"
_ARTIFACT_CUSTODY_STATUS = "NO_ARTIFACT_AT_USE_CUSTODY_EVIDENCE"
_SEMANTIC_REPLAY_STATUS = "REQUIRES_PROVIDER_SPECIFIC_SEMANTIC_REPLAY"
_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"

_BATCH_DOMAIN = (
    "jxplanetx.solar-system.states.provider-native-state-batch."
    "content-integrity.v1"
)
_BATCH_SCHEMA = "ProviderNativeStateBatch.v1"
_BATCH_QUALIFIED_NAME = (
    "jxplanetx.solar_system.states.ProviderNativeStateBatch"
)

_MAXIMUM_IDENTIFIER_CODEPOINTS = 256
_MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE = (1 << 31) - 1


def _text(
    value: object,
    label: str,
    maximum_codepoints: int = _MAXIMUM_IDENTIFIER_CODEPOINTS,
) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > maximum_codepoints:
        raise SolarSystemContractError(f"{label} exceeds the text cap")
    if not value or value.strip() != value:
        raise SolarSystemContractError(f"{label} must be nonempty trimmed text")
    if any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise SolarSystemContractError(f"{label} must use printable ASCII only")
    return value


def _exact_token(value: object, expected: str, label: str) -> str:
    result = _text(value, label)
    if result != expected:
        raise SolarSystemContractError(f"{label} is not the exact required token")
    return result


def _batch_payload(value: ProviderNativeStateBatch) -> tuple[object, ...]:
    return (
        _BATCH_QUALIFIED_NAME,
        tuple(
            (descriptor.name, getattr(value, descriptor.name))
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _batch_digest(value: ProviderNativeStateBatch) -> str:
    return domain_sha256(_BATCH_DOMAIN, _BATCH_SCHEMA, _batch_payload(value))


@dataclass(frozen=True, slots=True, eq=False)
class ProviderNativeStateBatch:
    """Caller-supplied provider-native values without execution authority."""

    batch_id: str
    query: EphemerisQuerySpec
    state_stage: str
    execution_evidence_status: str
    target_naif_ids: tuple[int, ...]
    component_order: tuple[str, str, str]
    state_shape: tuple[int, int]
    position_unit_id: str
    velocity_length_unit_id: str
    velocity_time_unit_id: str
    velocity_semantics: str
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    target_availability_status: str
    artifact_custody_status: str
    semantic_replay_status: str
    content_integrity_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _BATCH_DOMAIN
    _SCHEMA: ClassVar[str] = _BATCH_SCHEMA

    def __post_init__(self) -> None:
        if type(self.content_sha256) is not str:
            raise SolarSystemContractError(
                "content_sha256 must be an exact string"
            )
        if self.content_sha256 != "":
            validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _batch_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
            return
        if self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact provider-native state batch"
            )

    def _validate(self) -> None:
        if type(self) is not ProviderNativeStateBatch:
            raise SolarSystemContractError(
                "provider-native state batch must have its exact concrete type"
            )
        _text(self.batch_id, "batch_id")
        if type(self.query) is not EphemerisQuerySpec:
            raise SolarSystemContractError(
                "query must be an exact EphemerisQuerySpec"
            )
        validate_integrity(self.query)

        _exact_token(self.state_stage, _STATE_STAGE, "state_stage")
        _exact_token(
            self.execution_evidence_status,
            _EXECUTION_EVIDENCE_STATUS,
            "execution_evidence_status",
        )

        if type(self.target_naif_ids) is not tuple:
            raise SolarSystemContractError("target_naif_ids must be an exact tuple")
        target_count = len(self.query.target_naif_ids)
        if len(self.target_naif_ids) != target_count:
            raise SolarSystemContractError(
                "target_naif_ids must have the exact query target count"
            )
        if target_count < 1 or target_count > MAXIMUM_BODY_COUNT:
            raise SolarSystemContractError("query target count is outside the hard cap")
        for index, target in enumerate(self.target_naif_ids):
            if type(target) is not int:
                raise SolarSystemContractError(
                    f"target_naif_ids[{index}] must be an exact integer"
                )
            if abs(target) > _MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE:
                raise SolarSystemContractError(
                    f"target_naif_ids[{index}] is outside the exact NAIF-ID cap"
                )
        if self.target_naif_ids != self.query.target_naif_ids:
            raise SolarSystemContractError(
                "target_naif_ids must preserve exact query order"
            )

        if type(self.component_order) is not tuple or len(self.component_order) != 3:
            raise SolarSystemContractError(
                "component_order must be an exact three-item tuple"
            )
        for index, component in enumerate(self.component_order):
            _text(component, f"component_order[{index}]")
        if self.component_order != _COMPONENT_ORDER:
            raise SolarSystemContractError("component_order must be exactly X, Y, Z")

        if type(self.state_shape) is not tuple or len(self.state_shape) != 2:
            raise SolarSystemContractError(
                "state_shape must be an exact two-item tuple"
            )
        for index, extent in enumerate(self.state_shape):
            if type(extent) is not int:
                raise SolarSystemContractError(
                    f"state_shape[{index}] must be an exact integer"
                )
            if extent < 0 or extent > MAXIMUM_BODY_COUNT:
                raise SolarSystemContractError(
                    f"state_shape[{index}] is outside the exact extent cap"
                )
        if self.state_shape != (target_count, 3):
            raise SolarSystemContractError(
                "state_shape must be exactly (query target count, 3)"
            )

        units = self.query.output_unit_system
        position_unit = _text(self.position_unit_id, "position_unit_id")
        velocity_length_unit = _text(
            self.velocity_length_unit_id, "velocity_length_unit_id"
        )
        velocity_time_unit = _text(
            self.velocity_time_unit_id, "velocity_time_unit_id"
        )
        if position_unit != units.length.unit_id:
            raise SolarSystemContractError(
                "position_unit_id must equal the provider-native output length unit"
            )
        if velocity_length_unit != units.length.unit_id:
            raise SolarSystemContractError(
                "velocity length unit must equal the provider-native output length unit"
            )
        if velocity_time_unit != units.time.unit_id:
            raise SolarSystemContractError(
                "velocity time unit must equal the provider-native output time unit"
            )
        _exact_token(
            self.velocity_semantics,
            _VELOCITY_SEMANTICS,
            "velocity_semantics",
        )

        component_count = target_count * 3
        for label, block in (
            ("positions", self.positions),
            ("velocities", self.velocities),
        ):
            if type(block) is not tuple:
                raise SolarSystemContractError(f"{label} must be an exact tuple")
            if len(block) != component_count:
                raise SolarSystemContractError(
                    f"{label} must contain exactly three components per target"
                )
            for index, component in enumerate(block):
                if type(component) is not float or not math.isfinite(component):
                    raise SolarSystemContractError(
                        f"{label}[{index}] must be an exact finite binary64 value"
                    )

        target_status = _text(
            self.target_availability_status, "target_availability_status"
        )
        if target_status != self.query.target_chain_availability_status:
            raise SolarSystemContractError(
                "target availability status must equal the nested query declaration"
            )
        _exact_token(
            self.artifact_custody_status,
            _ARTIFACT_CUSTODY_STATUS,
            "artifact_custody_status",
        )
        _exact_token(
            self.semantic_replay_status,
            _SEMANTIC_REPLAY_STATUS,
            "semantic_replay_status",
        )
        _exact_token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )

    def validate_integrity(self) -> None:
        """Revalidate exact fields, the nested M1 query, and the own seal."""

        self._validate()
        validate_sha256(self.content_sha256, "content_sha256")
        if self.content_sha256 != _batch_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact provider-native state batch"
            )


def validate_provider_native_state_batch(value: ProviderNativeStateBatch) -> None:
    """Validate one exact state-batch contract without executing a provider."""

    if type(value) is not ProviderNativeStateBatch:
        raise SolarSystemContractError(
            "value must be an exact ProviderNativeStateBatch"
        )
    value.validate_integrity()


__all__ = [
    "ProviderNativeStateBatch",
    "validate_provider_native_state_batch",
]
