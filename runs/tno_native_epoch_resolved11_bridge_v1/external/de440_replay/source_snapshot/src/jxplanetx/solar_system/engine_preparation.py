"""Prepare exact DE440 resolved-eleven initial inputs for the frozen engine.

This opt-in module verifies and parses a separately retained GM parameter
artifact, recomputes the primary and replay preparation paths, then allocates
fresh owned read-only NumPy arrays for one direct mutual Newtonian force plan.
It does not modify or publish the engine, run a trajectory, authorize the
parameter sources, or claim continuous filesystem/array custody.
The M4C2B state receipt and the later parameter reads are selected point
observations, not one atomic snapshot.  Byte/call caps do not bound blocking
filesystem latency, and this boundary does not defend against same-UID, root,
mount, hard-link, or mutate-and-restore interference.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jxplanetx.engine.contracts import ForcePlan, StateSnapshot

from .artifacts import verify_local_artifact_bytes
from .contracts import (
    SolarSystemContractError,
    SolarSystemDataError,
    SolarSystemDependencyUnavailableError,
)
from .cspice_execution_contracts import CspiceSpkgeoExecutionReceipt
from .engine_preparation_contracts import (
    De440GmProjection,
    De440NewtonianInitialState,
    De440NewtonianPreparationReceipt,
    De440ResolvedEarthMoonNewtonianSpec,
    TdbEngineEpochBinding,
    _BODY_COUNT,
    _BODY_IDS,
    _BODY_NAIF_IDS,
    _BODY_ROWS,
    _CONTENT_INTEGRITY_CLASS,
    _ENGINE_ALIAS_POLICY,
    _EPOCH_FUTURE_STATUS,
    _EPOCH_MAPPING_POLICY,
    _EVIDENCE_CLASS,
    _FORCE_SCOPE,
    _INTEGRATOR_SCOPE,
    _MASS_POLICY,
    _OMITTED_PHYSICS,
    _ORIGIN_POLICY,
    _PARAMETER_READ_STATUS,
    _PREPARATION_REPLAY_SCOPE,
    _PREPARATION_REPLAY_STATUS,
    _PREPARATION_STATUS,
    _PROJECTION_POLICY,
    _RADIUS_POLICY,
    _RECENTER_POLICY,
    _ROSTER_POLICY,
    _RULES_SHA256,
    _SELECTED_LANE_POLICY,
    _SOURCE_GM_UNIT,
    _STATE_CONVERSION_POLICY,
    _STATE_STAGE,
    _TARGET_GM_UNIT,
    _TARGET_UNIT_SYSTEM,
    _artifact_binding,
    _body_parameter,
    _lane_preparation_semantic_digest,
    _lane_preparation_values,
    _physical_constant,
    _require_source_execution,
    _target_frame,
    validate_de440_newtonian_initial_state,
    validate_de440_newtonian_preparation_receipt,
)
from .units import KILOGRAM, METRE, SECOND, convert_fraction_to_binary64


_MAXIMUM_ROOT_FD = (1 << 31) - 1
_MAXIMUM_COMPONENTS = 64
_MAXIMUM_COMPONENT_CODEPOINTS = 256
_MAXIMUM_LOCATOR_CODEPOINTS = 1_024
_MAXIMUM_PARAMETER_BYTES = 65_536
_MAXIMUM_READ_CALLS = 8_192
_READ_CHUNK = 65_536
_ENGINE_GM_METADATA_UNITS = (
    f"{METRE.unit_id}^3/{SECOND.unit_id}^2"
)
_ROOT_FIELDS = ("st_dev", "st_ino", "st_mode")
_FILE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_ASSIGNMENT = re.compile(
    r"^[ \t]+(BODY(?:0|[1-9][0-9]*)_GM)[ \t]*=[ \t]*"
    r"\([ \t]*([0-9]+\.[0-9]+[DE][+-][0-9]+)[ \t]*\)[ \t]*$",
    re.MULTILINE,
)
_PROVENANCE_SOURCE_ID = "jxplanetx.de440-resolved-earth-moon-newtonian-preparation.v1"
_PROVENANCE_CITATION = (
    "DE440s SPKGEO geometric initial state and NAIF gm_de440.tpc; exact SI conversion "
    "and operational-GM-weighted resolved-eleven model-barycenter reduction"
)
_PROVENANCE_VERSION = "1"


def _load_engine_contracts():
    """Import the optional engine dependency only at materialization/use time."""

    try:
        from jxplanetx.engine import contracts as engine_contracts
    except (ImportError, OSError) as exc:
        raise SolarSystemDependencyUnavailableError(
            "the frozen engine runtime and its numerical dependencies are unavailable"
        ) from exc
    return engine_contracts


def _identifier(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if not value or len(value) > 256:
        raise SolarSystemContractError(f"{label} is outside its exact text cap")
    if value.strip() != value or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value):
        raise SolarSystemContractError(f"{label} must be trimmed printable ASCII")
    return value


def _components(locator: str) -> tuple[str, ...]:
    if type(locator) is not str or not locator or len(locator) > _MAXIMUM_LOCATOR_CODEPOINTS:
        raise SolarSystemContractError("parameter locator is outside its exact text cap")
    path = PurePosixPath(locator)
    parts = path.parts
    if path.is_absolute() or path.as_posix() != locator or not parts or len(parts) > _MAXIMUM_COMPONENTS:
        raise SolarSystemContractError("parameter locator is not normalized relative POSIX form")
    for part in parts:
        if part in ("", ".", "..") or len(part) > _MAXIMUM_COMPONENT_CODEPOINTS:
            raise SolarSystemContractError("parameter locator component is outside its cap")
        try:
            encoded = part.encode("ascii", "strict")
        except UnicodeEncodeError as exc:
            raise SolarSystemContractError("parameter locator component is not ASCII") from exc
        if len(encoded) > _MAXIMUM_COMPONENT_CODEPOINTS:
            raise SolarSystemContractError("parameter locator component exceeds its byte cap")
    return parts


def _stat_tuple(value: os.stat_result, names: tuple[str, ...], label: str) -> tuple[int, ...]:
    result: list[int] = []
    for name in names:
        item = getattr(value, name, None)
        if type(item) is not int or item.bit_length() > 128:
            raise SolarSystemDataError(f"{label}.{name} is not an exact bounded integer")
        result.append(item)
    return tuple(result)


def _walk_directory(root_fd: int, parts: tuple[str, ...]) -> int:
    current = os.dup(root_fd)
    transferred = False
    try:
        for part in parts:
            following = os.open(
                part,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=current,
            )
            os.close(current)
            current = following
        transferred = True
        return current
    finally:
        if not transferred:
            os.close(current)


def _read_held_artifact(root_fd: int, artifact, parser):
    parts = _components(artifact.logical_locator)
    if artifact.byte_length > _MAXIMUM_PARAMETER_BYTES:
        raise SolarSystemContractError("parameter artifact exceeds the fixed read cap")
    with ExitStack() as owned:
        parent = _walk_directory(root_fd, parts[:-1])
        owned.callback(os.close, parent)
        pre_open = _stat_tuple(
            os.stat(parts[-1], dir_fd=parent, follow_symlinks=False),
            _FILE_FIELDS,
            "pre_open",
        )
        if (
            not stat.S_ISREG(pre_open[2])
            or pre_open[3] < 1
            or pre_open[6] != artifact.byte_length
        ):
            raise SolarSystemDataError("parameter leaf is not the exact declared regular file")
        flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC
        if hasattr(os, "O_NOCTTY"):
            flags |= os.O_NOCTTY
        file_fd = os.open(parts[-1], flags, dir_fd=parent)
        owned.callback(os.close, file_fd)
        pre_read = _stat_tuple(os.fstat(file_fd), _FILE_FIELDS, "pre_read")
        if pre_read != pre_open:
            raise SolarSystemDataError("held parameter file differs from pre-open observation")
        remaining = artifact.byte_length
        retained = bytearray()
        digest = hashlib.sha256()
        read_calls = 0
        while remaining:
            request = min(remaining, _READ_CHUNK)
            chunk = os.read(file_fd, request)
            read_calls += 1
            if not chunk:
                raise SolarSystemDataError("parameter file ended before declared length")
            retained.extend(chunk)
            digest.update(chunk)
            remaining -= len(chunk)
            if read_calls > _MAXIMUM_READ_CALLS:
                raise SolarSystemDataError("parameter read-call cap exceeded")
        if os.read(file_fd, 1) != b"":
            raise SolarSystemDataError("parameter file contains undeclared trailing bytes")
        if digest.hexdigest() != artifact.artifact_sha256:
            raise SolarSystemDataError("held parameter bytes differ from exact SHA-256")
        post_read = _stat_tuple(os.fstat(file_fd), _FILE_FIELDS, "post_read")
        if post_read != pre_read:
            raise SolarSystemDataError("selected parameter metadata changed during read")
        immutable_bytes = bytes(retained)
        parsed = parser(immutable_bytes)
        second_parent = _walk_directory(root_fd, parts[:-1])
        owned.callback(os.close, second_parent)
        second = _stat_tuple(
            os.stat(parts[-1], dir_fd=second_parent, follow_symlinks=False),
            _FILE_FIELDS,
            "second_resolution",
        )
        if second != pre_read:
            raise SolarSystemDataError("second parameter resolution differs from held file identity")
        return parsed


def _parse_gm_bytes(data: bytes) -> tuple[str, ...]:
    if type(data) is not bytes or len(data) > _MAXIMUM_PARAMETER_BYTES:
        raise SolarSystemDataError("GM parser requires exact bounded bytes")
    try:
        text = data.decode("ascii", "strict")
    except UnicodeDecodeError as exc:
        raise SolarSystemDataError("GM parameter bytes are not exact ASCII") from exc
    matches: dict[str, list[str]] = {}
    for match in _ASSIGNMENT.finditer(text):
        matches.setdefault(match.group(1), []).append(match.group(2))
    result: list[str] = []
    for row in _BODY_ROWS:
        keyword = str(row[3])
        expected = str(row[4])
        if matches.get(keyword) != [expected]:
            raise SolarSystemDataError(
                f"{keyword} is not present exactly once with its frozen source lexeme"
            )
        result.append(expected)
    return tuple(result)


def _parse_rules_bytes(data: bytes) -> str:
    if type(data) is not bytes or len(data) > _MAXIMUM_PARAMETER_BYTES:
        raise SolarSystemDataError("Rules parser requires exact bounded bytes")
    if hashlib.sha256(data).hexdigest() != _RULES_SHA256:
        raise SolarSystemDataError("Rules bytes differ from the exact retained leaf")
    try:
        data.decode("ascii", "strict")
    except UnicodeDecodeError as exc:
        raise SolarSystemDataError("Rules bytes are not exact ASCII") from exc
    return "EXACT_RAW_NAIF_RULES_BYTES_OBSERVED_NONAUTHORIZING"


def _build_projections(lexemes: tuple[str, ...]) -> tuple[De440GmProjection, ...]:
    if type(lexemes) is not tuple or len(lexemes) != _BODY_COUNT:
        raise SolarSystemDataError("parsed GM lexeme roster has the wrong exact count")
    result: list[De440GmProjection] = []
    for index, row in enumerate(_BODY_ROWS):
        if lexemes[index] != row[4]:
            raise SolarSystemDataError("parsed GM lexeme order differs from the closed roster")
        exact_km = Fraction(Decimal(str(row[5])))
        conversion = convert_fraction_to_binary64(exact_km, _SOURCE_GM_UNIT, _TARGET_GM_UNIT)
        result.append(
            De440GmProjection(
                naif_id=int(row[0]),
                kernel_keyword=str(row[3]),
                kernel_value_lexeme=str(row[4]),
                source_gravitational_parameter=_physical_constant(row, target=False),
                si_conversion=conversion,
                target_gravitational_parameter=_physical_constant(row, target=True),
                projection_policy=_PROJECTION_POLICY,
                content_integrity_class=_CONTENT_INTEGRITY_CLASS,
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True, eq=False)
class PreparedDe440NewtonianInputs:
    preparation_receipt: De440NewtonianPreparationReceipt
    initial_state: De440NewtonianInitialState
    snapshot: "StateSnapshot"
    force_plan: "ForcePlan"


def _require_array(array, expected_shape: tuple[int, ...], expected_dtype, label: str, np):
    if type(array) is not np.ndarray:
        raise SolarSystemDataError(f"{label} must be an exact NumPy ndarray")
    if array.dtype != expected_dtype or array.shape != expected_shape:
        raise SolarSystemDataError(f"{label} has the wrong exact dtype or shape")
    if not array.flags.c_contiguous or not array.flags.owndata or array.base is not None:
        raise SolarSystemDataError(f"{label} must be fresh owned C-contiguous storage")
    if array.flags.writeable:
        raise SolarSystemDataError(f"{label} must be read-only at validation time")
    return array


def _exact_runtime_text(value: object, expected: str, label: str) -> None:
    if (
        type(value) is not str
        or len(value) != len(expected)
        or value != expected
    ):
        raise SolarSystemDataError(f"{label} differs from the exact sealed runtime text")


def validate_prepared_de440_newtonian_inputs(value: object) -> None:
    if type(value) is not PreparedDe440NewtonianInputs:
        raise SolarSystemContractError("value must be an exact PreparedDe440NewtonianInputs")
    validate_de440_newtonian_preparation_receipt(value.preparation_receipt)
    validate_de440_newtonian_initial_state(value.initial_state)
    if value.initial_state.preparation_receipt.content_sha256 != value.preparation_receipt.content_sha256:
        raise SolarSystemContractError("initial state does not bind the wrapper preparation receipt")
    engine_contracts = _load_engine_contracts()
    if type(value.snapshot) is not engine_contracts.StateSnapshot:
        raise SolarSystemContractError("snapshot must be an exact StateSnapshot")
    if type(value.force_plan) is not engine_contracts.ForcePlan:
        raise SolarSystemContractError("force_plan must be an exact ForcePlan")
    snapshot = value.snapshot
    plan = value.force_plan
    state = value.initial_state
    if type(snapshot.body_ids) is not tuple or len(snapshot.body_ids) != _BODY_COUNT:
        raise SolarSystemDataError("snapshot body_ids must be the exact bounded roster")
    if any(type(item) is not str or len(item) > 128 for item in snapshot.body_ids):
        raise SolarSystemDataError("snapshot body_ids entries must be exact bounded strings")
    if type(snapshot.provenance) is not engine_contracts.Provenance:
        raise SolarSystemDataError("snapshot provenance has the wrong exact type")
    if type(plan.backend) is not engine_contracts.BackendSpec:
        raise SolarSystemDataError("force plan backend has the wrong exact type")
    if type(plan.models) is not tuple or len(plan.models) != 1:
        raise SolarSystemDataError("force plan models must be an exact singleton tuple")
    model = plan.models[0]
    if type(model) is not engine_contracts.NewtonianPointMass:
        raise SolarSystemDataError("force plan model must be exact NewtonianPointMass")
    if (
        type(model.source_ids) is not tuple
        or len(model.source_ids) != _BODY_COUNT
        or type(model.target_ids) is not tuple
        or len(model.target_ids) != _BODY_COUNT
    ):
        raise SolarSystemDataError("Newtonian source/target rosters must be exact bounded tuples")
    if any(
        type(item) is not str or len(item) > 128
        for item in model.source_ids + model.target_ids
    ):
        raise SolarSystemDataError("Newtonian roster entries must be exact bounded strings")
    if type(model.parameter_metadata) is not tuple or len(model.parameter_metadata) != 1:
        raise SolarSystemDataError("Newtonian parameter metadata must be an exact singleton tuple")
    metadata = model.parameter_metadata[0]
    if type(metadata) is not engine_contracts.ParameterMetadata:
        raise SolarSystemDataError("Newtonian parameter metadata has the wrong exact type")
    if type(metadata.provenance) is not engine_contracts.Provenance:
        raise SolarSystemDataError("parameter provenance has the wrong exact type")
    if type(snapshot.epoch) is not float or snapshot.epoch.hex() != state.engine_epoch.hex():
        raise SolarSystemDataError("snapshot epoch differs from the sealed initial state")
    _exact_runtime_text(snapshot.snapshot_id, state.state_id, "snapshot.snapshot_id")
    for label, observed, expected in (
        ("snapshot.time_scale", snapshot.time_scale, state.engine_time_scale),
        ("snapshot.frame", snapshot.frame, state.engine_frame),
        ("snapshot.origin", snapshot.origin, state.engine_origin),
        ("snapshot.axes", snapshot.axes, state.engine_axes),
        ("snapshot.length_unit", snapshot.length_unit, state.engine_length_unit),
        ("snapshot.time_unit", snapshot.time_unit, state.engine_time_unit),
        ("snapshot.mass_unit", snapshot.mass_unit, state.engine_mass_unit),
        ("snapshot.unit_system_id", snapshot.unit_system_id, state.engine_unit_system_id),
        ("plan.plan_id", plan.plan_id, state.force_plan_id),
    ):
        _exact_runtime_text(observed, expected, label)
    if snapshot.body_ids != state.body_ids:
        raise SolarSystemDataError("snapshot body_ids differ from the sealed initial state")
    provenance = snapshot.provenance
    for label, observed, expected in (
        ("provenance.source_id", provenance.source_id, _PROVENANCE_SOURCE_ID),
        ("provenance.citation", provenance.citation, _PROVENANCE_CITATION),
        ("provenance.version", provenance.version, _PROVENANCE_VERSION),
        (
            "provenance.sha256",
            provenance.sha256,
            value.preparation_receipt.content_sha256,
        ),
    ):
        _exact_runtime_text(observed, expected, label)
    backend = plan.backend
    if (
        type(backend.tile_size) is not int
        or backend.tile_size != 1
        or type(backend.allow_fallback) is not bool
        or backend.allow_fallback
        or type(backend.deterministic_reductions) is not bool
        or not backend.deterministic_reductions
        or type(backend.fast_math) is not bool
        or backend.fast_math
    ):
        raise SolarSystemDataError("backend scalar policy differs from the exact tile-one profile")
    for label, observed, expected in (
        ("backend.backend_id", backend.backend_id, "numpy"),
        ("backend.device", backend.device, "cpu"),
        ("backend.dtype", backend.dtype, "float64"),
        ("backend.determinism_scope", backend.determinism_scope, "SAME_RUNTIME_DEVICE"),
        (
            "NewtonianPointMass.MODEL_ID",
            engine_contracts.NewtonianPointMass.MODEL_ID,
            "force.newtonian.point_mass",
        ),
        ("model.model_id", model.model_id, "force.newtonian.point_mass"),
        ("model.unit_system_id", model.unit_system_id, state.engine_unit_system_id),
        ("metadata.parameter_id", metadata.parameter_id, "state.gravitational_parameters"),
        ("metadata.units", metadata.units, _ENGINE_GM_METADATA_UNITS),
    ):
        _exact_runtime_text(observed, expected, label)
    if model.source_ids != state.body_ids or model.target_ids != state.body_ids:
        raise SolarSystemDataError("Newtonian source/target rosters differ from the sealed state")
    if (
        type(metadata.validity_start) is not float
        or metadata.validity_start.hex() != "0x0.0p+0"
        or type(metadata.validity_end) is not float
        or metadata.validity_end.hex() != "0x0.0p+0"
        or metadata.uncertainty is not None
        or metadata.covariance_group is not None
    ):
        raise SolarSystemDataError("parameter metadata scalar policy differs from point validity")
    for label, observed, expected in (
        ("metadata.provenance.source_id", metadata.provenance.source_id, _PROVENANCE_SOURCE_ID),
        ("metadata.provenance.citation", metadata.provenance.citation, _PROVENANCE_CITATION),
        ("metadata.provenance.version", metadata.provenance.version, _PROVENANCE_VERSION),
        (
            "metadata.provenance.sha256",
            metadata.provenance.sha256,
            value.preparation_receipt.content_sha256,
        ),
    ):
        _exact_runtime_text(observed, expected, label)
    if (
        type(plan.evidence_class) is not str
        or plan.evidence_class != _EVIDENCE_CLASS
        or type(plan.registry_authorized) is not bool
        or plan.registry_authorized
        or type(plan.qualification_authorized) is not bool
        or plan.qualification_authorized
    ):
        raise SolarSystemDataError("force plan authority fields differ from the unqualified profile")
    try:
        engine_contracts.StateSnapshot.__post_init__(snapshot)
        engine_contracts.Provenance.__post_init__(snapshot.provenance)
        engine_contracts.ForcePlan.__post_init__(plan)
        engine_contracts.BackendSpec.__post_init__(plan.backend)
        engine_contracts.NewtonianPointMass.__post_init__(model)
        engine_contracts.ParameterMetadata.__post_init__(metadata)
        engine_contracts.Provenance.__post_init__(metadata.provenance)
    except (
        engine_contracts.ContractError,
        AttributeError,
        OverflowError,
        TypeError,
        ValueError,
    ) as exc:
        raise SolarSystemDataError("engine runtime object contracts changed after construction") from exc
    try:
        import numpy as np
    except ImportError as exc:
        raise SolarSystemDependencyUnavailableError("NumPy is required for engine preparation") from exc
    if (
        snapshot.snapshot_id != state.state_id
        or snapshot.body_ids != state.body_ids
        or snapshot.epoch.hex() != state.engine_epoch.hex()
        or snapshot.time_scale != state.engine_time_scale
        or snapshot.frame != state.engine_frame
        or snapshot.origin != state.engine_origin
        or snapshot.axes != state.engine_axes
        or snapshot.length_unit != state.engine_length_unit
        or snapshot.time_unit != state.engine_time_unit
        or snapshot.mass_unit != state.engine_mass_unit
        or snapshot.unit_system_id != state.engine_unit_system_id
    ):
        raise SolarSystemDataError("snapshot metadata differs from the sealed initial state")
    positions = _require_array(snapshot.positions, (_BODY_COUNT, 3), np.dtype(np.float64), "positions", np)
    velocities = _require_array(snapshot.velocities, (_BODY_COUNT, 3), np.dtype(np.float64), "velocities", np)
    gm = _require_array(snapshot.gravitational_parameters, (_BODY_COUNT,), np.dtype(np.float64), "gravitational_parameters", np)
    masses = _require_array(snapshot.masses, (_BODY_COUNT,), np.dtype(np.float64), "masses", np)
    radii = _require_array(snapshot.radii, (_BODY_COUNT,), np.dtype(np.float64), "radii", np)
    massive = _require_array(snapshot.massive, (_BODY_COUNT,), np.dtype(np.bool_), "massive", np)
    arrays = (positions, velocities, gm, masses, radii, massive)
    for left_index, left in enumerate(arrays):
        for right in arrays[left_index + 1 :]:
            if np.shares_memory(left, right):
                raise SolarSystemDataError("prepared arrays must not share memory")
    for label, observed, expected in (
        ("positions", positions.ravel(order="C"), state.positions),
        ("velocities", velocities.ravel(order="C"), state.velocities),
        ("gravitational_parameters", gm, state.gravitational_parameters),
        ("masses", masses, state.masses),
        ("radii", radii, state.radii),
    ):
        if len(observed) != len(expected) or any(
            float(observed[index]).hex() != expected[index].hex()
            for index in range(len(expected))
        ):
            raise SolarSystemDataError(f"{label} bytes differ from the sealed initial state")
    if tuple(bool(item) for item in massive) != state.massive:
        raise SolarSystemDataError("massive mask differs from the sealed initial state")
    provenance = snapshot.provenance
    if (
        provenance.source_id != _PROVENANCE_SOURCE_ID
        or provenance.citation != _PROVENANCE_CITATION
        or provenance.version != _PROVENANCE_VERSION
        or provenance.sha256 != value.preparation_receipt.content_sha256
    ):
        raise SolarSystemDataError("snapshot provenance does not bind the preparation receipt")
    if plan.plan_id != state.force_plan_id:
        raise SolarSystemDataError("force plan identifier differs from the sealed initial state")
    backend = plan.backend
    if (
        type(backend) is not engine_contracts.BackendSpec
        or backend.backend_id != "numpy"
        or backend.device != "cpu"
        or backend.tile_size != 1
        or backend.dtype != "float64"
        or backend.allow_fallback
        or not backend.deterministic_reductions
        or backend.fast_math
        or backend.determinism_scope != "SAME_RUNTIME_DEVICE"
    ):
        raise SolarSystemDataError("force plan backend differs from the fixed tile-one profile")
    if (
        plan.plan_id == ""
        or len(plan.models) != 1
        or type(plan.models[0]) is not engine_contracts.NewtonianPointMass
    ):
        raise SolarSystemDataError("force plan must contain exactly one Newtonian model")
    model = plan.models[0]
    if (
        model.source_ids != state.body_ids
        or model.target_ids != state.body_ids
        or model.unit_system_id != state.engine_unit_system_id
        or len(model.parameter_metadata) != 1
    ):
        raise SolarSystemDataError("Newtonian model differs from the sealed resolved-eleven state")
    metadata = model.parameter_metadata[0]
    if (
        metadata.parameter_id != "state.gravitational_parameters"
        or metadata.units != _ENGINE_GM_METADATA_UNITS
        or metadata.uncertainty is not None
        or metadata.covariance_group is not None
        or metadata.validity_start.hex() != "0x0.0p+0"
        or metadata.validity_end.hex() != "0x0.0p+0"
        or metadata.provenance != provenance
    ):
        raise SolarSystemDataError("Newtonian parameter metadata is not the exact point-valid profile")
    if plan.evidence_class != _EVIDENCE_CLASS or plan.registry_authorized or plan.qualification_authorized:
        raise SolarSystemDataError("force plan authority controls differ from the unqualified profile")


def prepare_de440_newtonian_engine_inputs(
    source_execution: CspiceSpkgeoExecutionReceipt,
    *,
    parameter_root_directory_fd: int,
    receipt_id: str,
    state_id: str,
    plan_id: str,
) -> PreparedDe440NewtonianInputs:
    checked_receipt_id = _identifier(receipt_id, "receipt_id")
    checked_state_id = _identifier(state_id, "state_id")
    checked_plan_id = _identifier(plan_id, "plan_id")
    if len({checked_receipt_id, checked_state_id, checked_plan_id}) != 3:
        raise SolarSystemContractError("receipt, state, and plan identifiers must be distinct")
    if type(parameter_root_directory_fd) is not int or not 0 <= parameter_root_directory_fd <= _MAXIMUM_ROOT_FD:
        raise SolarSystemContractError("parameter_root_directory_fd must be an exact in-range integer")
    if type(source_execution) is not CspiceSpkgeoExecutionReceipt:
        raise SolarSystemContractError("source_execution must be an exact CspiceSpkgeoExecutionReceipt")
    _require_source_execution(source_execution)

    constants_artifact = _artifact_binding("CONSTANTS")
    rules_artifact = _artifact_binding("LICENSE")
    try:
        root_fd = os.dup(parameter_root_directory_fd)
    except (OSError, OverflowError) as exc:
        raise SolarSystemDataError("parameter root directory fd cannot be duplicated") from exc
    try:
        pre_root = _stat_tuple(os.fstat(root_fd), _ROOT_FIELDS, "pre_root")
        if not stat.S_ISDIR(pre_root[2]):
            raise SolarSystemDataError("parameter root capability is not a directory")
        gm_verification = verify_local_artifact_bytes(constants_artifact, root_fd)
        rules_verification = verify_local_artifact_bytes(rules_artifact, root_fd)
        lexemes = _read_held_artifact(root_fd, constants_artifact, _parse_gm_bytes)
        _read_held_artifact(root_fd, rules_artifact, _parse_rules_bytes)
        post_root = _stat_tuple(os.fstat(root_fd), _ROOT_FIELDS, "post_root")
        if post_root != pre_root:
            raise SolarSystemDataError("parameter root identity changed during preparation")
    except (SolarSystemContractError, SolarSystemDataError, SolarSystemDependencyUnavailableError):
        raise
    except (OSError, OverflowError) as exc:
        raise SolarSystemDataError("parameter artifact acquisition failed") from exc
    finally:
        os.close(root_fd)

    try:
        projections = _build_projections(lexemes)
        query = source_execution.primary_lane.projection.effective_query
        spec = De440ResolvedEarthMoonNewtonianSpec(
            spec_id="spec.de440-resolved-earth-moon-newtonian-initial-state.v1",
            constants_artifact=constants_artifact,
            constants_license_artifact=rules_artifact,
            ordered_gm_projections=projections,
            ordered_body_parameters=tuple(
                _body_parameter(row, projection)
                for row, projection in zip(_BODY_ROWS, projections)
            ),
            source_frame=query.frame,
            target_frame=_target_frame(query.frame),
            source_unit_system=query.output_unit_system,
            target_unit_system=_TARGET_UNIT_SYSTEM,
            roster_policy=_ROSTER_POLICY,
            origin_policy=_ORIGIN_POLICY,
            force_scope=_FORCE_SCOPE,
            integrator_scope=_INTEGRATOR_SCOPE,
            mass_policy=_MASS_POLICY,
            radius_policy=_RADIUS_POLICY,
            engine_alias_policy=_ENGINE_ALIAS_POLICY,
            omitted_physics=_OMITTED_PHYSICS,
            evidence_class=_EVIDENCE_CLASS,
            registry_authorized=False,
            qualification_authorized=False,
            content_integrity_class=_CONTENT_INTEGRITY_CLASS,
        )
        projected = source_execution.primary_lane.projection
        epoch_binding = TdbEngineEpochBinding(
            binding_id="epoch-binding.de440-resolved11.initial-origin.v1",
            effective_epoch=projected.effective_query.epoch,
            effective_binary64_et=projected.binary64_projection.rounded_value,
            engine_time_scale="TDB",
            engine_time_unit_id=SECOND.unit_id,
            engine_epoch_at_origin=0.0,
            mapping_policy=_EPOCH_MAPPING_POLICY,
            future_mapping_status=_EPOCH_FUTURE_STATUS,
            content_integrity_class=_CONTENT_INTEGRITY_CLASS,
        )
        primary_values = _lane_preparation_values(source_execution.primary_lane, spec)
        replay_values = _lane_preparation_values(source_execution.replay_lane, spec)
        primary_semantic = _lane_preparation_semantic_digest(
            source_execution.primary_lane,
            spec,
            epoch_binding,
            gm_verification,
            rules_verification,
            primary_values,
        )
        replay_semantic = _lane_preparation_semantic_digest(
            source_execution.replay_lane,
            spec,
            epoch_binding,
            gm_verification,
            rules_verification,
            replay_values,
        )
        receipt = De440NewtonianPreparationReceipt(
            receipt_id=checked_receipt_id,
            source_execution=source_execution,
            model_spec=spec,
            selected_lane_policy=_SELECTED_LANE_POLICY,
            source_state_payload_byte_length=int(primary_values["source_payload_byte_length"]),
            source_state_payload_sha256=str(primary_values["source_payload_sha256"]),
            converted_ssb_payload_sha256=str(primary_values["converted_payload_sha256"]),
            state_conversion_policy=_STATE_CONVERSION_POLICY,
            centroid_position_exact_pairs=primary_values["centroid_position_pairs"],
            centroid_velocity_exact_pairs=primary_values["centroid_velocity_pairs"],
            recenter_policy=_RECENTER_POLICY,
            normalized_position_residual_exact_pairs=primary_values["position_residual_pairs"],
            normalized_velocity_residual_exact_pairs=primary_values["velocity_residual_pairs"],
            recentered_payload_sha256=str(primary_values["recentered_payload_sha256"]),
            gm_payload_sha256=str(primary_values["gm_payload_sha256"]),
            epoch_binding=epoch_binding,
            gm_artifact_verification=gm_verification,
            gm_license_verification=rules_verification,
            parameter_read_status=_PARAMETER_READ_STATUS,
            primary_preparation_semantic_sha256=primary_semantic,
            replay_preparation_semantic_sha256=replay_semantic,
            semantic_replay_scope=_PREPARATION_REPLAY_SCOPE,
            semantic_replay_status=_PREPARATION_REPLAY_STATUS,
            preparation_status=_PREPARATION_STATUS,
            evidence_class=_EVIDENCE_CLASS,
            content_integrity_class=_CONTENT_INTEGRITY_CLASS,
        )
        initial = De440NewtonianInitialState(
            state_id=checked_state_id,
            force_plan_id=checked_plan_id,
            preparation_receipt=receipt,
            body_ids=_BODY_IDS,
            naif_ids=_BODY_NAIF_IDS,
            component_order=("X", "Y", "Z"),
            state_shape=(_BODY_COUNT, 3),
            positions=primary_values["positions"],
            velocities=primary_values["velocities"],
            gravitational_parameters=primary_values["gm_values"],
            masses=(0.0,) * _BODY_COUNT,
            radii=(0.0,) * _BODY_COUNT,
            massive=(True,) * _BODY_COUNT,
            engine_epoch=0.0,
            engine_time_scale="TDB",
            engine_frame="BARYCENTRIC_INERTIAL",
            engine_origin="BARYCENTER",
            engine_axes="CARTESIAN_RIGHT_HANDED",
            engine_length_unit=METRE.unit_id,
            engine_time_unit=SECOND.unit_id,
            engine_mass_unit=KILOGRAM.unit_id,
            engine_unit_system_id=_TARGET_UNIT_SYSTEM.unit_system_id,
            state_stage=_STATE_STAGE,
            evidence_class=_EVIDENCE_CLASS,
            registry_authorized=False,
            qualification_authorized=False,
            content_integrity_class=_CONTENT_INTEGRITY_CLASS,
        )
    except SolarSystemContractError as exc:
        raise SolarSystemDataError("verified source data could not construct the exact preparation") from exc

    engine_contracts = _load_engine_contracts()
    try:
        import numpy as np
    except ImportError as exc:
        raise SolarSystemDependencyUnavailableError("NumPy is required for engine preparation") from exc
    try:
        positions = np.empty((_BODY_COUNT, 3), dtype=np.float64)
        positions.flat[:] = initial.positions
        velocities = np.empty((_BODY_COUNT, 3), dtype=np.float64)
        velocities.flat[:] = initial.velocities
        gm = np.array(initial.gravitational_parameters, dtype=np.float64, copy=True)
        masses = np.zeros(_BODY_COUNT, dtype=np.float64)
        radii = np.zeros(_BODY_COUNT, dtype=np.float64)
        massive = np.ones(_BODY_COUNT, dtype=np.bool_)
        for array in (positions, velocities, gm, masses, radii, massive):
            array.flags.writeable = False
        provenance = engine_contracts.Provenance(
            source_id=_PROVENANCE_SOURCE_ID,
            citation=_PROVENANCE_CITATION,
            version=_PROVENANCE_VERSION,
            sha256=receipt.content_sha256,
        )
        snapshot = engine_contracts.StateSnapshot(
            snapshot_id=checked_state_id,
            epoch=0.0,
            time_scale="TDB",
            frame="BARYCENTRIC_INERTIAL",
            origin="BARYCENTER",
            axes="CARTESIAN_RIGHT_HANDED",
            length_unit=METRE.unit_id,
            time_unit=SECOND.unit_id,
            mass_unit=KILOGRAM.unit_id,
            unit_system_id=_TARGET_UNIT_SYSTEM.unit_system_id,
            body_ids=_BODY_IDS,
            positions=positions,
            velocities=velocities,
            gravitational_parameters=gm,
            masses=masses,
            radii=radii,
            massive=massive,
            provenance=provenance,
        )
        metadata = engine_contracts.ParameterMetadata(
            parameter_id="state.gravitational_parameters",
            # The frozen evaluator derives this label from the snapshot's
            # length/time strings.  The exact M2 GM unit remains sealed in the
            # preparation records rather than being relabelled here.
            units=_ENGINE_GM_METADATA_UNITS,
            provenance=provenance,
            uncertainty=None,
            covariance_group=None,
            validity_start=0.0,
            validity_end=0.0,
        )
        model = engine_contracts.NewtonianPointMass(
            source_ids=_BODY_IDS,
            target_ids=_BODY_IDS,
            unit_system_id=_TARGET_UNIT_SYSTEM.unit_system_id,
            parameter_metadata=(metadata,),
        )
        plan = engine_contracts.ForcePlan(
            plan_id=checked_plan_id,
            backend=engine_contracts.BackendSpec(
                backend_id="numpy",
                device="cpu",
                tile_size=1,
                dtype="float64",
                allow_fallback=False,
                deterministic_reductions=True,
                fast_math=False,
                determinism_scope="SAME_RUNTIME_DEVICE",
            ),
            models=(model,),
            evidence_class=_EVIDENCE_CLASS,
            registry_authorized=False,
            qualification_authorized=False,
        )
    except (engine_contracts.ContractError, ValueError, TypeError) as exc:
        raise SolarSystemDataError("engine runtime objects could not be constructed exactly") from exc
    prepared = PreparedDe440NewtonianInputs(receipt, initial, snapshot, plan)
    validate_prepared_de440_newtonian_inputs(prepared)
    return prepared


__all__ = [
    "PreparedDe440NewtonianInputs",
    "prepare_de440_newtonian_engine_inputs",
    "validate_prepared_de440_newtonian_inputs",
]
