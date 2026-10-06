"""Backend-neutral contracts for experimental long-horizon integrations.

These records are immutable metadata receipts.  They do not replace an
integrator's authoritative state or checkpoint object, and they make no
scientific or production qualification claim.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping


PUBLIC_STABILITY = "EXPERIMENTAL"
PROJECT_SCOPE = "SCREENING_ONLY"
SCIENTIFIC_CLASSIFICATION = "MODEL_OUTPUT"

EVENT_SCHEMA = "jxplanetx.long-horizon.event.v1"
RESULT_SCHEMA = "jxplanetx.long-horizon.result.v2"
CHECKPOINT_SCHEMA = "jxplanetx.long-horizon.checkpoint.v1"
ARCHITECTURE_SCHEMA = "jxplanetx.long-horizon.architecture.v1"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_RESULT_STATUSES = frozenset({"COMPLETED", "TERMINATED"})
_CHECKPOINT_STATUSES = frozenset({"VERIFIED"})

CLAIM_CEILING = MappingProxyType(
    {
        "production_qualified": False,
        "science_result": False,
        "scientific_inference_qualified": False,
        "long_horizon_accuracy_qualified": False,
        "physical_collision_inferred": False,
        "physical_ejection_inferred": False,
        "cross_runtime_reproducible": False,
        "globally_symplectic": False,
    }
)


class LongHorizonContractError(ValueError):
    """A shared long-horizon metadata record violates its contract."""


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise LongHorizonContractError(f"{label} must be a nonempty trimmed string")
    return value


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise LongHorizonContractError(f"{label} must be a nonnegative integer")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise LongHorizonContractError(f"{label} must be a finite built-in float")
    return value


def _digest(value: object, label: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise LongHorizonContractError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _vector(value: object, label: str) -> tuple[float, float, float]:
    if type(value) is not tuple or len(value) != 3:
        raise LongHorizonContractError(
            f"{label} must be an exact three-float tuple"
        )
    for index, item in enumerate(value):
        _finite(item, f"{label}[{index}]")
    return value


def _freeze_tree(value: Any, label: str = "metadata") -> Any:
    if isinstance(value, MappingABC):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            _text(key, f"{label} key")
            if key in frozen:
                raise LongHorizonContractError(f"{label} contains a duplicate key")
            frozen[key] = _freeze_tree(item, f"{label}.{key}")
        return MappingProxyType(frozen)
    if type(value) in (list, tuple):
        return tuple(
            _freeze_tree(item, f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return _finite(value, label)
    raise LongHorizonContractError(
        f"{label} contains unsupported value type {type(value).__name__}"
    )


def _freeze_mapping(
    value: object,
    label: str,
    *,
    allow_empty: bool = True,
) -> Mapping[str, Any]:
    if not isinstance(value, MappingABC) or (not allow_empty and not value):
        qualifier = "a mapping" if allow_empty else "a nonempty mapping"
        raise LongHorizonContractError(f"{label} must be {qualifier}")
    return _freeze_tree(value, label)


def _plain_tree(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return {key: _plain_tree(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain_tree(item) for item in value]
    if type(value) is float:
        return {"float_hex": value.hex()}
    return value


def _claims(value: Mapping[str, bool]) -> Mapping[str, bool]:
    if not isinstance(value, MappingABC) or dict(value) != dict(CLAIM_CEILING):
        raise LongHorizonContractError("the long-horizon claim ceiling changed")
    if any(type(item) is not bool or item for item in value.values()):
        raise LongHorizonContractError("all long-horizon claims must be exact false")
    return MappingProxyType(dict(CLAIM_CEILING))


def _content_sha256(domain: bytes, record: Mapping[str, Any]) -> str:
    payload = json.dumps(
        _plain_tree(record),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(domain + payload).hexdigest()


@dataclass(frozen=True, eq=False)
class LongHorizonArchitectureContract:
    """Declaration of one canonical but explicitly experimental architecture."""

    architecture_id: str
    family: str
    components: tuple[str, ...]
    entry_points: Mapping[str, str]
    canonical: bool = True
    public_stability: str = PUBLIC_STABILITY
    project_scope: str = PROJECT_SCOPE
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)
    schema: str = ARCHITECTURE_SCHEMA

    def __post_init__(self) -> None:
        _text(self.architecture_id, "architecture_id")
        _text(self.family, "family")
        if (
            type(self.components) is not tuple
            or not self.components
            or len(set(self.components)) != len(self.components)
        ):
            raise LongHorizonContractError(
                "components must be a nonempty unique exact tuple"
            )
        for index, component in enumerate(self.components):
            _text(component, f"components[{index}]")
        if self.canonical is not True:
            raise LongHorizonContractError("the declared architecture must be canonical")
        _validate_ceiling(self)
        entry_points = _freeze_mapping(
            self.entry_points, "entry_points", allow_empty=False
        )
        if any(type(item) is not str for item in entry_points.values()):
            raise LongHorizonContractError(
                "entry point values must be import-path strings"
            )
        object.__setattr__(self, "entry_points", entry_points)
        object.__setattr__(self, "claims", _claims(self.claims))


@dataclass(frozen=True, eq=False)
class LongHorizonEventContract:
    """Backend-neutral, immutable event metadata."""

    event_type: str
    tracer_index: int
    global_step: int
    step_fraction: float
    relative_position: tuple[float, float, float]
    relative_velocity: tuple[float, float, float]
    coordinate_system: str
    frame: str
    length_unit: str
    time_unit: str
    location_method: str
    origin_method_id: str
    terminal: bool
    source_index: int | None = None
    source_id: str | None = None
    lane_index: int | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    public_stability: str = PUBLIC_STABILITY
    project_scope: str = PROJECT_SCOPE
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)
    schema: str = EVENT_SCHEMA
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.event_type, "event_type"),
            (self.coordinate_system, "coordinate_system"),
            (self.frame, "frame"),
            (self.length_unit, "length_unit"),
            (self.time_unit, "time_unit"),
            (self.location_method, "location_method"),
            (self.origin_method_id, "origin_method_id"),
        ):
            _text(value, label)
        _nonnegative(self.tracer_index, "tracer_index")
        _nonnegative(self.global_step, "global_step")
        fraction = _finite(self.step_fraction, "step_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise LongHorizonContractError("step_fraction must be in [0, 1]")
        _vector(self.relative_position, "relative_position")
        _vector(self.relative_velocity, "relative_velocity")
        if type(self.terminal) is not bool:
            raise LongHorizonContractError("terminal must be an exact bool")
        if (self.source_index is None) != (self.source_id is None):
            raise LongHorizonContractError(
                "source_index and source_id must either both be present or both absent"
            )
        if self.source_index is not None:
            _nonnegative(self.source_index, "source_index")
            _text(self.source_id, "source_id")
        if self.lane_index is not None:
            _nonnegative(self.lane_index, "lane_index")
        _validate_ceiling(self)
        object.__setattr__(
            self, "diagnostics", _freeze_mapping(self.diagnostics, "diagnostics")
        )
        object.__setattr__(self, "claims", _claims(self.claims))
        actual = _content_sha256(
            b"jxplanetx.long-horizon.event.v1\0", self._content_record()
        )
        if self.content_sha256 is not None:
            _digest(self.content_sha256, "content_sha256")
            if self.content_sha256 != actual:
                raise LongHorizonContractError("event content_sha256 does not match")
        object.__setattr__(self, "content_sha256", actual)

    def _content_record(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "event_type": self.event_type,
            "tracer_index": self.tracer_index,
            "lane_index": self.lane_index,
            "global_step": self.global_step,
            "step_fraction": self.step_fraction,
            "source_index": self.source_index,
            "source_id": self.source_id,
            "relative_position": self.relative_position,
            "relative_velocity": self.relative_velocity,
            "coordinate_system": self.coordinate_system,
            "frame": self.frame,
            "length_unit": self.length_unit,
            "time_unit": self.time_unit,
            "location_method": self.location_method,
            "origin_method_id": self.origin_method_id,
            "terminal": self.terminal,
            "diagnostics": self.diagnostics,
            "public_stability": self.public_stability,
            "project_scope": self.project_scope,
            "scientific_classification": self.scientific_classification,
            "claims": self.claims,
        }


@dataclass(frozen=True, eq=False)
class LongHorizonResultContract:
    """Normalized receipt for a fully committed integration segment."""

    method_id: str
    status: str
    global_step: int
    epoch: float
    source_tile_sha256: str
    output_state_sha256: str
    tracer_count: int
    active_tracer_count: int
    route_count: int
    route_ledger_sha256: str
    events: tuple[LongHorizonEventContract, ...]
    runtime_identity: Mapping[str, Any]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    public_stability: str = PUBLIC_STABILITY
    project_scope: str = PROJECT_SCOPE
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)
    schema: str = RESULT_SCHEMA
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.method_id, "method_id")
        if self.status not in _RESULT_STATUSES:
            raise LongHorizonContractError("status is not a supported result status")
        _nonnegative(self.global_step, "global_step")
        _finite(self.epoch, "epoch")
        _digest(self.source_tile_sha256, "source_tile_sha256")
        _digest(self.output_state_sha256, "output_state_sha256")
        count = _nonnegative(self.tracer_count, "tracer_count")
        active = _nonnegative(self.active_tracer_count, "active_tracer_count")
        if active > count:
            raise LongHorizonContractError("active_tracer_count exceeds tracer_count")
        _nonnegative(self.route_count, "route_count")
        _digest(self.route_ledger_sha256, "route_ledger_sha256")
        if type(self.events) is not tuple or any(
            type(item) is not LongHorizonEventContract for item in self.events
        ):
            raise LongHorizonContractError(
                "events must be an exact tuple of LongHorizonEventContract"
            )
        event_keys = tuple(_event_order_key(item) for item in self.events)
        if event_keys != tuple(sorted(event_keys)) or len(
            {item.content_sha256 for item in self.events}
        ) != len(self.events):
            raise LongHorizonContractError("events must be unique and canonically ordered")
        if any(
            item.global_step > self.global_step
            or (
                item.global_step == self.global_step
                and item.step_fraction != 0.0
            )
            for item in self.events
        ):
            raise LongHorizonContractError("an event is beyond the result state")
        _validate_ceiling(self)
        object.__setattr__(
            self,
            "runtime_identity",
            _freeze_mapping(
                self.runtime_identity, "runtime_identity", allow_empty=False
            ),
        )
        object.__setattr__(
            self, "diagnostics", _freeze_mapping(self.diagnostics, "diagnostics")
        )
        object.__setattr__(self, "claims", _claims(self.claims))
        actual = _content_sha256(
            b"jxplanetx.long-horizon.result.v2\0", self._content_record()
        )
        if self.content_sha256 is not None:
            _digest(self.content_sha256, "content_sha256")
            if self.content_sha256 != actual:
                raise LongHorizonContractError("result content_sha256 does not match")
        object.__setattr__(self, "content_sha256", actual)

    def _content_record(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "status": self.status,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "source_tile_sha256": self.source_tile_sha256,
            "output_state_sha256": self.output_state_sha256,
            "tracer_count": self.tracer_count,
            "active_tracer_count": self.active_tracer_count,
            "route_count": self.route_count,
            "route_ledger_sha256": self.route_ledger_sha256,
            "events": tuple(item.content_sha256 for item in self.events),
            "runtime_identity": self.runtime_identity,
            "diagnostics": self.diagnostics,
            "public_stability": self.public_stability,
            "project_scope": self.project_scope,
            "scientific_classification": self.scientific_classification,
            "claims": self.claims,
        }


@dataclass(frozen=True, eq=False)
class LongHorizonCheckpointContract:
    """Normalized receipt for an already verified checkpoint head."""

    method_id: str
    status: str
    checkpoint_index: int
    run_id: str
    protocol_sha256: str
    checkpoint_content_sha256: str
    input_state_sha256: str
    state_sha256: str
    global_step: int
    epoch: float
    lineage: Mapping[str, Any] = field(default_factory=dict)
    public_stability: str = PUBLIC_STABILITY
    project_scope: str = PROJECT_SCOPE
    scientific_classification: str = SCIENTIFIC_CLASSIFICATION
    claims: Mapping[str, bool] = field(default_factory=lambda: CLAIM_CEILING)
    schema: str = CHECKPOINT_SCHEMA
    content_sha256: str | None = None

    def __post_init__(self) -> None:
        _text(self.method_id, "method_id")
        if self.status not in _CHECKPOINT_STATUSES:
            raise LongHorizonContractError(
                "status is not a supported checkpoint status"
            )
        _nonnegative(self.checkpoint_index, "checkpoint_index")
        _text(self.run_id, "run_id")
        for value, label in (
            (self.protocol_sha256, "protocol_sha256"),
            (self.checkpoint_content_sha256, "checkpoint_content_sha256"),
            (self.input_state_sha256, "input_state_sha256"),
            (self.state_sha256, "state_sha256"),
        ):
            _digest(value, label)
        _nonnegative(self.global_step, "global_step")
        _finite(self.epoch, "epoch")
        _validate_ceiling(self)
        object.__setattr__(
            self,
            "lineage",
            _freeze_mapping(self.lineage, "lineage", allow_empty=False),
        )
        object.__setattr__(self, "claims", _claims(self.claims))
        actual = _content_sha256(
            b"jxplanetx.long-horizon.checkpoint.v1\0", self._content_record()
        )
        if self.content_sha256 is not None:
            _digest(self.content_sha256, "content_sha256")
            if self.content_sha256 != actual:
                raise LongHorizonContractError("checkpoint content_sha256 does not match")
        object.__setattr__(self, "content_sha256", actual)

    def _content_record(self) -> Mapping[str, Any]:
        return {
            "schema": self.schema,
            "method_id": self.method_id,
            "status": self.status,
            "checkpoint_index": self.checkpoint_index,
            "run_id": self.run_id,
            "protocol_sha256": self.protocol_sha256,
            "checkpoint_content_sha256": self.checkpoint_content_sha256,
            "input_state_sha256": self.input_state_sha256,
            "state_sha256": self.state_sha256,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "lineage": self.lineage,
            "public_stability": self.public_stability,
            "project_scope": self.project_scope,
            "scientific_classification": self.scientific_classification,
            "claims": self.claims,
        }


def _validate_ceiling(value: object) -> None:
    if getattr(value, "public_stability", None) != PUBLIC_STABILITY:
        raise LongHorizonContractError("public stability must remain EXPERIMENTAL")
    if getattr(value, "project_scope", None) != PROJECT_SCOPE:
        raise LongHorizonContractError("project scope must remain SCREENING_ONLY")
    if getattr(value, "scientific_classification", None) != SCIENTIFIC_CLASSIFICATION:
        raise LongHorizonContractError(
            "scientific classification must remain MODEL_OUTPUT"
        )
    expected = {
        LongHorizonArchitectureContract: ARCHITECTURE_SCHEMA,
        LongHorizonEventContract: EVENT_SCHEMA,
        LongHorizonResultContract: RESULT_SCHEMA,
        LongHorizonCheckpointContract: CHECKPOINT_SCHEMA,
    }.get(type(value))
    if expected is None or getattr(value, "schema", None) != expected:
        raise LongHorizonContractError("long-horizon schema identity changed")


def _event_order_key(event: LongHorizonEventContract) -> tuple[Any, ...]:
    return (
        event.global_step,
        event.step_fraction,
        -1 if event.lane_index is None else event.lane_index,
        event.tracer_index,
        event.event_type,
        event.content_sha256,
    )


__all__ = [
    "ARCHITECTURE_SCHEMA",
    "CHECKPOINT_SCHEMA",
    "CLAIM_CEILING",
    "EVENT_SCHEMA",
    "LongHorizonArchitectureContract",
    "LongHorizonCheckpointContract",
    "LongHorizonContractError",
    "LongHorizonEventContract",
    "LongHorizonResultContract",
    "PROJECT_SCOPE",
    "PUBLIC_STABILITY",
    "RESULT_SCHEMA",
    "SCIENTIFIC_CLASSIFICATION",
]
