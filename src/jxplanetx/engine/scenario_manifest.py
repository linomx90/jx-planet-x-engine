"""Canonical portable manifests for public JX RKF78 dynamics scenarios.

V1 is intentionally restricted to NumPy/CPU binary64.  Floating-point values
are serialized with Python's canonical ``float.hex()`` representation, arrays
use explicit C-order value streams, and loading requires an external SHA-256
identity.  The manifest preserves a scenario; it does not qualify its physics
or numerical accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .backends import resolve_backend
from .contracts import (
    BackendSpec,
    CannonballSRP,
    ForcePlan,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    RestrictedStaticCentral1PN,
    StateSnapshot,
)
from .scenario import DynamicsScenario, ScenarioContractError
from .trajectory_contracts import AdaptiveRKF78Spec


PORTABLE_SCENARIO_MANIFEST_SCHEMA = "jxplanetx.dynamics-scenario-manifest.v1"
PORTABLE_SCENARIO_MANIFEST_SCOPE = "NUMPY_CPU_FLOAT64_ADAPTIVE_RKF78_ONLY"
PORTABLE_SCENARIO_FLOAT_ENCODING = "PYTHON_FLOAT_HEX_BINARY64"
PORTABLE_SCENARIO_ARRAY_ORDER = "C_ROW_MAJOR"
PORTABLE_SCENARIO_HASH_ALGORITHM = "SHA256_CANONICAL_JSON_BYTES"
PORTABLE_SCENARIO_MANIFEST_MAX_BYTES = 64 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ScenarioManifestError(ScenarioContractError):
    """A portable scenario manifest is malformed or has lost identity."""


def _expect_keys(value: object, keys: tuple[str, ...], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ScenarioManifestError(f"{label} must be an object")
    if set(value) != set(keys) or len(value) != len(keys):
        raise ScenarioManifestError(f"{label} must contain exactly {keys!r}")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ScenarioManifestError(f"{label} must be a nonempty, trimmed string")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _integer(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ScenarioManifestError(f"{label} must be an exact integer")
    if minimum is not None and value < minimum:
        raise ScenarioManifestError(f"{label} must be at least {minimum}")
    return value


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise ScenarioManifestError(f"{label} must be an exact boolean")
    return value


def _string_tuple(value: object, label: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if type(value) is not list or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a nonempty list"
        raise ScenarioManifestError(f"{label} must be {qualifier}")
    checked = tuple(
        _text(item, f"{label}[{index}]") for index, item in enumerate(value)
    )
    if len(set(checked)) != len(checked):
        raise ScenarioManifestError(f"{label} must not contain duplicates")
    return checked


def _float_hex(value: object, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScenarioManifestError(f"{label} must be a finite real number")
    checked = float(value)
    if not math.isfinite(checked):
        raise ScenarioManifestError(f"{label} must be a finite real number")
    return checked.hex()


def _from_float_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise ScenarioManifestError(f"{label} must be canonical binary64 hex")
    try:
        checked = float.fromhex(value)
    except ValueError as exc:
        raise ScenarioManifestError(
            f"{label} must be canonical binary64 hex"
        ) from exc
    if not math.isfinite(checked) or checked.hex() != value:
        raise ScenarioManifestError(f"{label} must be canonical finite binary64 hex")
    return checked


def _canonical(document: object) -> bytes:
    try:
        text = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ScenarioManifestError("manifest cannot be serialized canonically") from exc
    return (text + "\n").encode("ascii")


def _strict_json(raw: bytes) -> object:
    duplicate: str | None = None

    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        nonlocal duplicate
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result and duplicate is None:
                duplicate = key
            result[key] = value
        return result

    def reject_float(token: str) -> object:
        raise ScenarioManifestError(
            f"JSON floating token {token!r} is forbidden; use canonical binary64 hex"
        )

    def reject_constant(token: str) -> object:
        raise ScenarioManifestError(f"nonfinite JSON constant {token!r} is forbidden")

    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=object_pairs,
            parse_float=reject_float,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScenarioManifestError("manifest must be canonical ASCII JSON") from exc
    if duplicate is not None:
        raise ScenarioManifestError(f"manifest repeats object key {duplicate!r}")
    return document


def dynamics_scenario_manifest_sha256(manifest_bytes: bytes) -> str:
    """Return the raw canonical-manifest SHA-256 identity."""

    if type(manifest_bytes) is not bytes:
        raise ScenarioManifestError("manifest_bytes must be exact bytes")
    return hashlib.sha256(manifest_bytes).hexdigest()


def _provenance_document(value: Provenance) -> dict[str, object]:
    if type(value) is not Provenance:
        raise ScenarioManifestError("provenance must be Provenance")
    return {
        "citation": value.citation,
        "sha256": value.sha256,
        "source_id": value.source_id,
        "version": value.version,
    }


def _provenance_from_document(value: object, label: str) -> Provenance:
    document = _expect_keys(
        value,
        ("citation", "sha256", "source_id", "version"),
        label,
    )
    return Provenance(
        source_id=_text(document["source_id"], f"{label}.source_id"),
        citation=_text(document["citation"], f"{label}.citation"),
        version=_text(document["version"], f"{label}.version"),
        sha256=_text(document["sha256"], f"{label}.sha256"),
    )


def _metadata_document(value: ParameterMetadata) -> dict[str, object]:
    if type(value) is not ParameterMetadata:
        raise ScenarioManifestError("parameter metadata must be ParameterMetadata")
    uncertainty = (
        None
        if value.uncertainty is None
        else _float_hex(value.uncertainty, "uncertainty")
    )
    return {
        "covariance_group": value.covariance_group,
        "parameter_id": value.parameter_id,
        "provenance": _provenance_document(value.provenance),
        "uncertainty_hex": uncertainty,
        "units": value.units,
        "validity_end_hex": _float_hex(value.validity_end, "validity_end"),
        "validity_start_hex": _float_hex(value.validity_start, "validity_start"),
    }


def _metadata_from_document(value: object, label: str) -> ParameterMetadata:
    document = _expect_keys(
        value,
        (
            "covariance_group",
            "parameter_id",
            "provenance",
            "uncertainty_hex",
            "units",
            "validity_end_hex",
            "validity_start_hex",
        ),
        label,
    )
    raw_uncertainty = document["uncertainty_hex"]
    uncertainty = (
        None
        if raw_uncertainty is None
        else _from_float_hex(raw_uncertainty, f"{label}.uncertainty_hex")
    )
    return ParameterMetadata(
        parameter_id=_text(document["parameter_id"], f"{label}.parameter_id"),
        units=_text(document["units"], f"{label}.units"),
        provenance=_provenance_from_document(
            document["provenance"], f"{label}.provenance"
        ),
        uncertainty=uncertainty,
        covariance_group=_optional_text(
            document["covariance_group"], f"{label}.covariance_group"
        ),
        validity_start=_from_float_hex(
            document["validity_start_hex"], f"{label}.validity_start_hex"
        ),
        validity_end=_from_float_hex(
            document["validity_end_hex"], f"{label}.validity_end_hex"
        ),
    )


def _metadata_tuple_document(
    values: tuple[ParameterMetadata, ...],
) -> list[dict[str, object]]:
    if type(values) is not tuple:
        raise ScenarioManifestError("parameter_metadata must be an immutable tuple")
    return [_metadata_document(value) for value in values]


def _metadata_tuple_from_document(
    value: object,
    label: str,
) -> tuple[ParameterMetadata, ...]:
    if type(value) is not list:
        raise ScenarioManifestError(f"{label} must be a list")
    return tuple(
        _metadata_from_document(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _float_array_document(
    value: object,
    expected_shape: tuple[int, ...],
    label: str,
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> dict[str, object]:
    backend = resolve_backend("numpy")
    if type(value) is not backend.array_type:
        raise ScenarioManifestError(f"{label} must be an exact NumPy array")
    array = backend.require_native_array(value, label)
    if array.dtype != backend.xp.dtype("float64"):
        raise ScenarioManifestError(f"{label} must have dtype float64")
    if array.shape != expected_shape:
        raise ScenarioManifestError(f"{label} must have shape {expected_shape!r}")
    if not bool(backend.xp.all(backend.xp.isfinite(array)).item()):
        raise ScenarioManifestError(f"{label} must contain only finite values")
    if nonnegative and bool(backend.xp.any(array < 0.0).item()):
        raise ScenarioManifestError(f"{label} cannot contain negative values")
    if positive and bool(backend.xp.any(array <= 0.0).item()):
        raise ScenarioManifestError(f"{label} must be strictly positive")
    return {
        "dtype": "float64",
        "shape": list(expected_shape),
        "values_hex": [float(item).hex() for item in array.flat],
    }


def _float_array_from_document(
    value: object,
    expected_shape: tuple[int, ...],
    label: str,
) -> object:
    document = _expect_keys(value, ("dtype", "shape", "values_hex"), label)
    if document["dtype"] != "float64":
        raise ScenarioManifestError(f"{label}.dtype must equal 'float64'")
    shape = document["shape"]
    if type(shape) is not list or tuple(shape) != expected_shape or any(
        type(dimension) is not int for dimension in shape
    ):
        raise ScenarioManifestError(f"{label}.shape must equal {expected_shape!r}")
    values = document["values_hex"]
    expected_size = math.prod(expected_shape)
    if type(values) is not list or len(values) != expected_size:
        raise ScenarioManifestError(
            f"{label}.values_hex must contain exactly {expected_size} values"
        )
    floats = tuple(
        _from_float_hex(item, f"{label}.values_hex[{index}]")
        for index, item in enumerate(values)
    )
    import numpy as np

    array = np.asarray(floats, dtype=np.float64).reshape(expected_shape)
    array.setflags(write=False)
    return array


def _bool_array_document(
    value: object,
    expected_shape: tuple[int, ...],
    label: str,
) -> dict[str, object]:
    backend = resolve_backend("numpy")
    if type(value) is not backend.array_type:
        raise ScenarioManifestError(f"{label} must be an exact NumPy array")
    array = backend.require_native_array(value, label)
    if array.dtype != backend.xp.dtype("bool") or array.shape != expected_shape:
        raise ScenarioManifestError(
            f"{label} must be a Boolean array with shape {expected_shape!r}"
        )
    return {
        "dtype": "bool",
        "shape": list(expected_shape),
        "values": [bool(item) for item in array.flat],
    }


def _bool_array_from_document(
    value: object,
    expected_shape: tuple[int, ...],
    label: str,
) -> object:
    document = _expect_keys(value, ("dtype", "shape", "values"), label)
    if document["dtype"] != "bool":
        raise ScenarioManifestError(f"{label}.dtype must equal 'bool'")
    shape = document["shape"]
    if type(shape) is not list or tuple(shape) != expected_shape or any(
        type(dimension) is not int for dimension in shape
    ):
        raise ScenarioManifestError(f"{label}.shape must equal {expected_shape!r}")
    values = document["values"]
    expected_size = math.prod(expected_shape)
    if (
        type(values) is not list
        or len(values) != expected_size
        or any(type(item) is not bool for item in values)
    ):
        raise ScenarioManifestError(
            f"{label}.values must contain exactly {expected_size} booleans"
        )
    import numpy as np

    array = np.asarray(values, dtype=np.bool_).reshape(expected_shape)
    array.setflags(write=False)
    return array


def _backend_document(value: BackendSpec) -> dict[str, object]:
    if type(value) is not BackendSpec:
        raise ScenarioManifestError("backend must be BackendSpec")
    if value.backend_id != "numpy" or value.device != "cpu":
        raise ScenarioManifestError(
            "portable scenario manifest V1 supports only NumPy/CPU"
        )
    return {
        "allow_fallback": value.allow_fallback,
        "backend_id": value.backend_id,
        "determinism_scope": value.determinism_scope,
        "deterministic_reductions": value.deterministic_reductions,
        "device": value.device,
        "dtype": value.dtype,
        "fast_math": value.fast_math,
        "tile_size": value.tile_size,
    }


def _backend_from_document(value: object) -> BackendSpec:
    label = "scenario.force_plan.backend"
    document = _expect_keys(
        value,
        (
            "allow_fallback",
            "backend_id",
            "determinism_scope",
            "deterministic_reductions",
            "device",
            "dtype",
            "fast_math",
            "tile_size",
        ),
        label,
    )
    backend = BackendSpec(
        backend_id=_text(document["backend_id"], f"{label}.backend_id"),
        device=_text(document["device"], f"{label}.device"),
        tile_size=_integer(document["tile_size"], f"{label}.tile_size", minimum=1),
        dtype=_text(document["dtype"], f"{label}.dtype"),
        allow_fallback=_boolean(
            document["allow_fallback"], f"{label}.allow_fallback"
        ),
        deterministic_reductions=_boolean(
            document["deterministic_reductions"],
            f"{label}.deterministic_reductions",
        ),
        fast_math=_boolean(document["fast_math"], f"{label}.fast_math"),
        determinism_scope=_text(
            document["determinism_scope"], f"{label}.determinism_scope"
        ),
    )
    if backend.backend_id != "numpy" or backend.device != "cpu":
        raise ScenarioManifestError(
            "portable scenario manifest V1 supports only NumPy/CPU"
        )
    return backend


def _snapshot_document(value: StateSnapshot) -> dict[str, object]:
    if type(value) is not StateSnapshot:
        raise ScenarioManifestError("initial_snapshot must be StateSnapshot")
    body_count = len(value.body_ids)
    gm = _float_array_document(
        value.gravitational_parameters,
        (body_count,),
        "gravitational_parameters",
        nonnegative=True,
    )
    masses = _float_array_document(
        value.masses,
        (body_count,),
        "masses",
        nonnegative=True,
    )
    radii = _float_array_document(
        value.radii,
        (body_count,),
        "radii",
        nonnegative=True,
    )
    massive = _bool_array_document(value.massive, (body_count,), "massive")
    for index, is_massive in enumerate(massive["values"]):
        if is_massive and float.fromhex(gm["values_hex"][index]) <= 0.0:
            raise ScenarioManifestError("every massive body must have positive GM")
    return {
        "axes": value.axes,
        "body_ids": list(value.body_ids),
        "epoch_hex": _float_hex(value.epoch, "snapshot.epoch"),
        "frame": value.frame,
        "gravitational_parameters": gm,
        "length_unit": value.length_unit,
        "mass_unit": value.mass_unit,
        "masses": masses,
        "massive": massive,
        "origin": value.origin,
        "positions": _float_array_document(
            value.positions, (body_count, 3), "positions"
        ),
        "provenance": _provenance_document(value.provenance),
        "radii": radii,
        "snapshot_id": value.snapshot_id,
        "time_scale": value.time_scale,
        "time_unit": value.time_unit,
        "unit_system_id": value.unit_system_id,
        "velocities": _float_array_document(
            value.velocities, (body_count, 3), "velocities"
        ),
    }


def _snapshot_from_document(value: object) -> StateSnapshot:
    label = "scenario.initial_snapshot"
    keys = (
        "axes",
        "body_ids",
        "epoch_hex",
        "frame",
        "gravitational_parameters",
        "length_unit",
        "mass_unit",
        "masses",
        "massive",
        "origin",
        "positions",
        "provenance",
        "radii",
        "snapshot_id",
        "time_scale",
        "time_unit",
        "unit_system_id",
        "velocities",
    )
    document = _expect_keys(value, keys, label)
    body_ids = _string_tuple(document["body_ids"], f"{label}.body_ids")
    body_count = len(body_ids)
    return StateSnapshot(
        snapshot_id=_text(document["snapshot_id"], f"{label}.snapshot_id"),
        epoch=_from_float_hex(document["epoch_hex"], f"{label}.epoch_hex"),
        time_scale=_text(document["time_scale"], f"{label}.time_scale"),
        frame=_text(document["frame"], f"{label}.frame"),
        origin=_text(document["origin"], f"{label}.origin"),
        axes=_text(document["axes"], f"{label}.axes"),
        length_unit=_text(document["length_unit"], f"{label}.length_unit"),
        time_unit=_text(document["time_unit"], f"{label}.time_unit"),
        mass_unit=_text(document["mass_unit"], f"{label}.mass_unit"),
        unit_system_id=_text(
            document["unit_system_id"], f"{label}.unit_system_id"
        ),
        body_ids=body_ids,
        positions=_float_array_from_document(
            document["positions"], (body_count, 3), f"{label}.positions"
        ),
        velocities=_float_array_from_document(
            document["velocities"], (body_count, 3), f"{label}.velocities"
        ),
        gravitational_parameters=_float_array_from_document(
            document["gravitational_parameters"],
            (body_count,),
            f"{label}.gravitational_parameters",
        ),
        masses=_float_array_from_document(
            document["masses"], (body_count,), f"{label}.masses"
        ),
        radii=_float_array_from_document(
            document["radii"], (body_count,), f"{label}.radii"
        ),
        massive=_bool_array_from_document(
            document["massive"], (body_count,), f"{label}.massive"
        ),
        provenance=_provenance_from_document(
            document["provenance"], f"{label}.provenance"
        ),
    )


def _model_document(value: object) -> dict[str, object]:
    if type(value) is NewtonianPointMass:
        return {
            "kind": "NEWTONIAN_POINT_MASS",
            "model_id": value.model_id,
            "parameter_metadata": _metadata_tuple_document(value.parameter_metadata),
            "source_ids": list(value.source_ids),
            "target_ids": list(value.target_ids),
            "unit_system_id": value.unit_system_id,
        }
    if type(value) is RestrictedStaticCentral1PN:
        return {
            "central_source_id": value.central_source_id,
            "kind": "RESTRICTED_STATIC_CENTRAL_1PN",
            "maximum_compactness_hex": _float_hex(
                value.maximum_compactness, "maximum_compactness"
            ),
            "maximum_speed_fraction_squared_hex": _float_hex(
                value.maximum_speed_fraction_squared,
                "maximum_speed_fraction_squared",
            ),
            "model_id": value.model_id,
            "parameter_metadata": _metadata_tuple_document(value.parameter_metadata),
            "speed_of_light_hex": _float_hex(value.speed_of_light, "speed_of_light"),
            "target_ids": list(value.target_ids),
            "unit_system_id": value.unit_system_id,
        }
    if type(value) is CannonballSRP:
        target_count = len(value.target_ids)
        return {
            "area_to_mass": _float_array_document(
                value.area_to_mass,
                (target_count,),
                "area_to_mass",
                nonnegative=True,
            ),
            "attitude_model": value.attitude_model,
            "coefficient_convention": value.coefficient_convention,
            "kind": "CANNONBALL_SRP",
            "model_id": value.model_id,
            "parameter_metadata": _metadata_tuple_document(value.parameter_metadata),
            "radiation_pressure_coefficient": _float_array_document(
                value.radiation_pressure_coefficient,
                (target_count,),
                "radiation_pressure_coefficient",
                nonnegative=True,
            ),
            "radiation_source_id": value.radiation_source_id,
            "reference_distance_hex": _float_hex(
                value.reference_distance, "reference_distance"
            ),
            "reference_pressure_hex": _float_hex(
                value.reference_pressure, "reference_pressure"
            ),
            "shadow_model": value.shadow_model,
            "target_ids": list(value.target_ids),
            "unit_system_id": value.unit_system_id,
        }
    raise ScenarioManifestError(
        "portable scenario manifest V1 supports only implemented force types"
    )


def _model_from_document(value: object, index: int) -> object:
    label = f"scenario.force_plan.models[{index}]"
    if type(value) is not dict:
        raise ScenarioManifestError(f"{label} must be an object")
    kind = value.get("kind")
    if kind == "NEWTONIAN_POINT_MASS":
        document = _expect_keys(
            value,
            (
                "kind",
                "model_id",
                "parameter_metadata",
                "source_ids",
                "target_ids",
                "unit_system_id",
            ),
            label,
        )
        model = NewtonianPointMass(
            source_ids=_string_tuple(document["source_ids"], f"{label}.source_ids"),
            target_ids=_string_tuple(document["target_ids"], f"{label}.target_ids"),
            unit_system_id=_text(
                document["unit_system_id"], f"{label}.unit_system_id"
            ),
            parameter_metadata=_metadata_tuple_from_document(
                document["parameter_metadata"], f"{label}.parameter_metadata"
            ),
        )
    elif kind == "RESTRICTED_STATIC_CENTRAL_1PN":
        document = _expect_keys(
            value,
            (
                "central_source_id",
                "kind",
                "maximum_compactness_hex",
                "maximum_speed_fraction_squared_hex",
                "model_id",
                "parameter_metadata",
                "speed_of_light_hex",
                "target_ids",
                "unit_system_id",
            ),
            label,
        )
        model = RestrictedStaticCentral1PN(
            central_source_id=_text(
                document["central_source_id"], f"{label}.central_source_id"
            ),
            target_ids=_string_tuple(document["target_ids"], f"{label}.target_ids"),
            speed_of_light=_from_float_hex(
                document["speed_of_light_hex"], f"{label}.speed_of_light_hex"
            ),
            maximum_compactness=_from_float_hex(
                document["maximum_compactness_hex"],
                f"{label}.maximum_compactness_hex",
            ),
            maximum_speed_fraction_squared=_from_float_hex(
                document["maximum_speed_fraction_squared_hex"],
                f"{label}.maximum_speed_fraction_squared_hex",
            ),
            unit_system_id=_text(
                document["unit_system_id"], f"{label}.unit_system_id"
            ),
            parameter_metadata=_metadata_tuple_from_document(
                document["parameter_metadata"], f"{label}.parameter_metadata"
            ),
        )
    elif kind == "CANNONBALL_SRP":
        document = _expect_keys(
            value,
            (
                "area_to_mass",
                "attitude_model",
                "coefficient_convention",
                "kind",
                "model_id",
                "parameter_metadata",
                "radiation_pressure_coefficient",
                "radiation_source_id",
                "reference_distance_hex",
                "reference_pressure_hex",
                "shadow_model",
                "target_ids",
                "unit_system_id",
            ),
            label,
        )
        target_ids = _string_tuple(document["target_ids"], f"{label}.target_ids")
        target_count = len(target_ids)
        model = CannonballSRP(
            radiation_source_id=_text(
                document["radiation_source_id"], f"{label}.radiation_source_id"
            ),
            target_ids=target_ids,
            reference_pressure=_from_float_hex(
                document["reference_pressure_hex"],
                f"{label}.reference_pressure_hex",
            ),
            reference_distance=_from_float_hex(
                document["reference_distance_hex"],
                f"{label}.reference_distance_hex",
            ),
            area_to_mass=_float_array_from_document(
                document["area_to_mass"],
                (target_count,),
                f"{label}.area_to_mass",
            ),
            radiation_pressure_coefficient=_float_array_from_document(
                document["radiation_pressure_coefficient"],
                (target_count,),
                f"{label}.radiation_pressure_coefficient",
            ),
            coefficient_convention=_text(
                document["coefficient_convention"],
                f"{label}.coefficient_convention",
            ),
            attitude_model=_text(
                document["attitude_model"], f"{label}.attitude_model"
            ),
            shadow_model=_text(document["shadow_model"], f"{label}.shadow_model"),
            unit_system_id=_text(
                document["unit_system_id"], f"{label}.unit_system_id"
            ),
            parameter_metadata=_metadata_tuple_from_document(
                document["parameter_metadata"], f"{label}.parameter_metadata"
            ),
        )
    else:
        raise ScenarioManifestError(f"{label}.kind is unsupported")
    if document["model_id"] != model.model_id:
        raise ScenarioManifestError(f"{label}.model_id does not match kind")
    return model


def _force_plan_document(value: ForcePlan) -> dict[str, object]:
    if type(value) is not ForcePlan:
        raise ScenarioManifestError("force_plan must be ForcePlan")
    return {
        "backend": _backend_document(value.backend),
        "evidence_class": value.evidence_class,
        "models": [_model_document(model) for model in value.models],
        "plan_id": value.plan_id,
        "qualification_authorized": value.qualification_authorized,
        "registry_authorized": value.registry_authorized,
    }


def _force_plan_from_document(value: object) -> ForcePlan:
    label = "scenario.force_plan"
    document = _expect_keys(
        value,
        (
            "backend",
            "evidence_class",
            "models",
            "plan_id",
            "qualification_authorized",
            "registry_authorized",
        ),
        label,
    )
    raw_models = document["models"]
    if type(raw_models) is not list or not raw_models:
        raise ScenarioManifestError(f"{label}.models must be a nonempty list")
    return ForcePlan(
        plan_id=_text(document["plan_id"], f"{label}.plan_id"),
        backend=_backend_from_document(document["backend"]),
        models=tuple(
            _model_from_document(model, index)
            for index, model in enumerate(raw_models)
        ),
        evidence_class=_text(
            document["evidence_class"], f"{label}.evidence_class"
        ),
        registry_authorized=_boolean(
            document["registry_authorized"], f"{label}.registry_authorized"
        ),
        qualification_authorized=_boolean(
            document["qualification_authorized"],
            f"{label}.qualification_authorized",
        ),
    )


def _integration_spec_document(value: AdaptiveRKF78Spec) -> dict[str, object]:
    if type(value) is not AdaptiveRKF78Spec:
        raise ScenarioManifestError("integration_spec must be AdaptiveRKF78Spec")
    checkpoint_count = len(value.checkpoint_epochs)
    if checkpoint_count < 2:
        raise ScenarioManifestError("integration_spec requires at least two checkpoints")
    body_count = value.position_atol.shape[0] if hasattr(value.position_atol, "shape") else -1
    return {
        "accepted_distinguishing_stages": list(value.accepted_distinguishing_stages),
        "accepted_order": value.accepted_order,
        "accepted_solution": value.accepted_solution,
        "accepted_state_accumulation": value.accepted_state_accumulation,
        "checkpoint_epochs_hex": [
            _float_hex(epoch, f"checkpoint_epochs[{index}]")
            for index, epoch in enumerate(value.checkpoint_epochs)
        ],
        "checkpoint_policy": value.checkpoint_policy,
        "checkpoint_proposal_policy": value.checkpoint_proposal_policy,
        "controller_exponent_hex": _float_hex(
            value.controller_exponent, "controller_exponent"
        ),
        "defect_orientation": value.defect_orientation,
        "dense_output": value.dense_output,
        "dtype": value.dtype,
        "embedded_distinguishing_stages": list(value.embedded_distinguishing_stages),
        "embedded_order": value.embedded_order,
        "embedded_solution": value.embedded_solution,
        "error_norm": value.error_norm,
        "error_scale": value.error_scale,
        "evidence_class": value.evidence_class,
        "force_evaluations_per_attempt": value.force_evaluations_per_attempt,
        "initial_step_hex": _float_hex(value.initial_step, "initial_step"),
        "maximum_rejections": value.maximum_rejections,
        "maximum_scale_factor_hex": _float_hex(
            value.maximum_scale_factor, "maximum_scale_factor"
        ),
        "maximum_step_hex": _float_hex(value.maximum_step, "maximum_step"),
        "maximum_steps": value.maximum_steps,
        "method_class": value.method_class,
        "method_id": value.method_id,
        "minimum_scale_factor_hex": _float_hex(
            value.minimum_scale_factor, "minimum_scale_factor"
        ),
        "minimum_step_failure_policy": value.minimum_step_failure_policy,
        "minimum_step_hex": _float_hex(value.minimum_step, "minimum_step"),
        "position_atol": _float_array_document(
            value.position_atol,
            (body_count, 3),
            "position_atol",
            positive=True,
        ),
        "position_rtol_hex": _float_hex(value.position_rtol, "position_rtol"),
        "principal_order": value.principal_order,
        "qualification_authorized": value.qualification_authorized,
        "registry_authorized": value.registry_authorized,
        "safety_factor_hex": _float_hex(value.safety_factor, "safety_factor"),
        "source_author": value.source_author,
        "source_document_id": value.source_document_id,
        "source_publication_date": value.source_publication_date,
        "source_report": value.source_report,
        "source_title": value.source_title,
        "source_url": value.source_url,
        "stage_count": value.stage_count,
        "supports_velocity_dependent_forces": value.supports_velocity_dependent_forces,
        "time_step_representation": value.time_step_representation,
        "velocity_atol": _float_array_document(
            value.velocity_atol,
            (body_count, 3),
            "velocity_atol",
            positive=True,
        ),
        "velocity_rtol_hex": _float_hex(value.velocity_rtol, "velocity_rtol"),
        "zero_defect_scale_policy": value.zero_defect_scale_policy,
    }


def _integration_spec_from_document(
    value: object,
    body_count: int,
) -> AdaptiveRKF78Spec:
    label = "scenario.integration_spec"
    if type(value) is not dict:
        raise ScenarioManifestError(f"{label} must be an object")
    raw_epochs = value.get("checkpoint_epochs_hex")
    if type(raw_epochs) is not list:
        raise ScenarioManifestError(f"{label}.checkpoint_epochs_hex must be a list")
    checkpoints = tuple(
        _from_float_hex(item, f"{label}.checkpoint_epochs_hex[{index}]")
        for index, item in enumerate(raw_epochs)
    )
    request = AdaptiveRKF78Spec(
        checkpoint_epochs=checkpoints,
        initial_step=_from_float_hex(
            value.get("initial_step_hex"), f"{label}.initial_step_hex"
        ),
        minimum_step=_from_float_hex(
            value.get("minimum_step_hex"), f"{label}.minimum_step_hex"
        ),
        maximum_step=_from_float_hex(
            value.get("maximum_step_hex"), f"{label}.maximum_step_hex"
        ),
        position_atol=_float_array_from_document(
            value.get("position_atol"), (body_count, 3), f"{label}.position_atol"
        ),
        position_rtol=_from_float_hex(
            value.get("position_rtol_hex"), f"{label}.position_rtol_hex"
        ),
        velocity_atol=_float_array_from_document(
            value.get("velocity_atol"), (body_count, 3), f"{label}.velocity_atol"
        ),
        velocity_rtol=_from_float_hex(
            value.get("velocity_rtol_hex"), f"{label}.velocity_rtol_hex"
        ),
        maximum_steps=_integer(
            value.get("maximum_steps"), f"{label}.maximum_steps", minimum=1
        ),
        maximum_rejections=_integer(
            value.get("maximum_rejections"),
            f"{label}.maximum_rejections",
            minimum=0,
        ),
        safety_factor=_from_float_hex(
            value.get("safety_factor_hex"), f"{label}.safety_factor_hex"
        ),
        minimum_scale_factor=_from_float_hex(
            value.get("minimum_scale_factor_hex"),
            f"{label}.minimum_scale_factor_hex",
        ),
        maximum_scale_factor=_from_float_hex(
            value.get("maximum_scale_factor_hex"),
            f"{label}.maximum_scale_factor_hex",
        ),
    )
    expected = _integration_spec_document(request)
    if value != expected:
        raise ScenarioManifestError(
            "integration_spec does not match the complete fixed RKF78 contract"
        )
    return request


def _scenario_document(value: DynamicsScenario) -> dict[str, object]:
    if type(value) is not DynamicsScenario:
        raise ScenarioManifestError("scenario must be an exact DynamicsScenario")
    if value.force_plan.backend.backend_id != "numpy" or value.force_plan.backend.device != "cpu":
        raise ScenarioManifestError(
            "portable scenario manifest V1 supports only NumPy/CPU"
        )
    if value.integration_spec.position_atol is None:
        raise ScenarioManifestError("integration_spec position_atol is missing")
    return {
        "accuracy_claimed": value.accuracy_claimed,
        "comparison_id": value.comparison_id,
        "description": value.description,
        "evidence_class": value.evidence_class,
        "force_plan": _force_plan_document(value.force_plan),
        "improvement_claimed": value.improvement_claimed,
        "initial_snapshot": _snapshot_document(value.initial_snapshot),
        "integration_spec": _integration_spec_document(value.integration_spec),
        "physics_claimed": value.physics_claimed,
        "qualification_authorized": value.qualification_authorized,
        "registry_authorized": value.registry_authorized,
        "role": value.role,
        "scenario_id": value.scenario_id,
        "scope": value.scope,
    }


def _scenario_from_document(value: object) -> DynamicsScenario:
    label = "scenario"
    document = _expect_keys(
        value,
        (
            "accuracy_claimed",
            "comparison_id",
            "description",
            "evidence_class",
            "force_plan",
            "improvement_claimed",
            "initial_snapshot",
            "integration_spec",
            "physics_claimed",
            "qualification_authorized",
            "registry_authorized",
            "role",
            "scenario_id",
            "scope",
        ),
        label,
    )
    snapshot = _snapshot_from_document(document["initial_snapshot"])
    return DynamicsScenario(
        scenario_id=_text(document["scenario_id"], f"{label}.scenario_id"),
        role=_text(document["role"], f"{label}.role"),
        description=_text(document["description"], f"{label}.description"),
        comparison_id=_optional_text(
            document["comparison_id"], f"{label}.comparison_id"
        ),
        initial_snapshot=snapshot,
        force_plan=_force_plan_from_document(document["force_plan"]),
        integration_spec=_integration_spec_from_document(
            document["integration_spec"], len(snapshot.body_ids)
        ),
        scope=_text(document["scope"], f"{label}.scope"),
        evidence_class=_text(
            document["evidence_class"], f"{label}.evidence_class"
        ),
        registry_authorized=_boolean(
            document["registry_authorized"], f"{label}.registry_authorized"
        ),
        qualification_authorized=_boolean(
            document["qualification_authorized"],
            f"{label}.qualification_authorized",
        ),
        accuracy_claimed=_boolean(
            document["accuracy_claimed"], f"{label}.accuracy_claimed"
        ),
        improvement_claimed=_boolean(
            document["improvement_claimed"], f"{label}.improvement_claimed"
        ),
        physics_claimed=_boolean(
            document["physics_claimed"], f"{label}.physics_claimed"
        ),
    )


def dump_dynamics_scenario_manifest(scenario: DynamicsScenario) -> bytes:
    """Return canonical, newline-terminated ASCII JSON for one scenario."""

    document = {
        "array_order": PORTABLE_SCENARIO_ARRAY_ORDER,
        "float_encoding": PORTABLE_SCENARIO_FLOAT_ENCODING,
        "hash_algorithm": PORTABLE_SCENARIO_HASH_ALGORITHM,
        "portability_scope": PORTABLE_SCENARIO_MANIFEST_SCOPE,
        "scenario": _scenario_document(scenario),
        "schema": PORTABLE_SCENARIO_MANIFEST_SCHEMA,
    }
    raw = _canonical(document)
    if len(raw) > PORTABLE_SCENARIO_MANIFEST_MAX_BYTES:
        raise ScenarioManifestError("manifest exceeds the V1 byte ceiling")
    return raw


def load_dynamics_scenario_manifest(
    manifest_bytes: bytes,
    expected_sha256: str,
) -> DynamicsScenario:
    """Verify identity and rebuild one read-only NumPy/CPU scenario."""

    if type(manifest_bytes) is not bytes:
        raise ScenarioManifestError("manifest_bytes must be exact bytes")
    if len(manifest_bytes) > PORTABLE_SCENARIO_MANIFEST_MAX_BYTES:
        raise ScenarioManifestError("manifest exceeds the V1 byte ceiling")
    if type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None:
        raise ScenarioManifestError("expected_sha256 must be lowercase SHA-256 hex")
    actual_sha256 = dynamics_scenario_manifest_sha256(manifest_bytes)
    if actual_sha256 != expected_sha256:
        raise ScenarioManifestError("manifest SHA-256 identity mismatch")
    document = _strict_json(manifest_bytes)
    root = _expect_keys(
        document,
        (
            "array_order",
            "float_encoding",
            "hash_algorithm",
            "portability_scope",
            "scenario",
            "schema",
        ),
        "manifest",
    )
    exact = {
        "array_order": PORTABLE_SCENARIO_ARRAY_ORDER,
        "float_encoding": PORTABLE_SCENARIO_FLOAT_ENCODING,
        "hash_algorithm": PORTABLE_SCENARIO_HASH_ALGORITHM,
        "portability_scope": PORTABLE_SCENARIO_MANIFEST_SCOPE,
        "schema": PORTABLE_SCENARIO_MANIFEST_SCHEMA,
    }
    for field, expected in exact.items():
        if type(root[field]) is not str or root[field] != expected:
            raise ScenarioManifestError(
                f"manifest.{field} must equal {expected!r}"
            )
    if _canonical(root) != manifest_bytes:
        raise ScenarioManifestError("manifest bytes are not canonical")
    try:
        scenario = _scenario_from_document(root["scenario"])
    except ScenarioManifestError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ScenarioManifestError("manifest scenario reconstruction failed") from exc
    if dump_dynamics_scenario_manifest(scenario) != manifest_bytes:
        raise ScenarioManifestError("manifest is not an exact scenario round trip")
    return scenario


__all__ = [
    "PORTABLE_SCENARIO_ARRAY_ORDER",
    "PORTABLE_SCENARIO_FLOAT_ENCODING",
    "PORTABLE_SCENARIO_HASH_ALGORITHM",
    "PORTABLE_SCENARIO_MANIFEST_MAX_BYTES",
    "PORTABLE_SCENARIO_MANIFEST_SCHEMA",
    "PORTABLE_SCENARIO_MANIFEST_SCOPE",
    "ScenarioManifestError",
    "dump_dynamics_scenario_manifest",
    "dynamics_scenario_manifest_sha256",
    "load_dynamics_scenario_manifest",
]
