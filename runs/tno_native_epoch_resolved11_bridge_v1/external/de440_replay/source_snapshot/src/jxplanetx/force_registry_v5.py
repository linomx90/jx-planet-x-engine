"""Fail-closed inspection and loading for the JX V5 physics registry.

This module is specification infrastructure, not a force implementation.  It
allows draft registries to be inspected while categorically refusing to turn
unresolved declarations into an executable dynamics configuration.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .provenance import canonical_json, sha256_data, sha256_file


REGISTRY_SCHEMA = "jx-force-parameter-registry/v5"
INSPECTION_SCHEMA = "jx-force-parameter-registry-inspection/v5"
DEFAULT_SCHEMA_NAME = "jx-force-parameter-registry-v5.schema.json"
BLOCKED = "TBD_BLOCKED"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EVIDENCE_CLASSES = (
    "MEASURED",
    "RECONSTRUCTED",
    "MODEL_OUTPUT",
    "ASSUMPTION",
    "FORECAST",
    "SPECULATION",
)


class ForceRegistryError(ValueError):
    """Stable, machine-readable V5 registry validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise ForceRegistryError(code, message)


def _reject_json_constant(value: str) -> None:
    _fail("nonfinite_json", f"JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _load_json(path: str | Path, context: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except ForceRegistryError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail("invalid_json", f"cannot load {context} {source}: {exc}")
    if not isinstance(value, dict):
        _fail("invalid_json_root", f"{context} root must be an object")
    return value


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError, RecursionError):
        return False


def _schema_type_matches(value: Any, declared: str) -> bool:
    if declared == "object":
        return isinstance(value, dict)
    if declared == "array":
        return isinstance(value, list)
    if declared == "string":
        return isinstance(value, str)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared == "null":
        return value is None
    _fail("unsupported_schema_keyword", f"unsupported JSON Schema type {declared!r}")


def _resolve_local_ref(root_schema: Mapping[str, Any], reference: str) -> Mapping[str, Any]:
    if not reference.startswith("#/"):
        _fail("unsupported_schema_reference", f"only local schema references are supported: {reference!r}")
    current: Any = root_schema
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            _fail("invalid_schema_reference", f"schema reference does not resolve: {reference!r}")
        current = current[key]
    if not isinstance(current, Mapping):
        _fail("invalid_schema_reference", f"schema reference is not an object: {reference!r}")
    return current


def _schema_matches(value: Any, schema: Mapping[str, Any], root_schema: Mapping[str, Any]) -> bool:
    try:
        _validate_schema_value(value, schema, root_schema, "conditional")
    except ForceRegistryError:
        return False
    return True


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    context: str,
) -> None:
    if "$ref" in schema:
        _validate_schema_value(value, _resolve_local_ref(root_schema, schema["$ref"]), root_schema, context)
        return

    declared_type = schema.get("type")
    if declared_type is not None and not _schema_type_matches(value, declared_type):
        _fail("schema_type", f"{context} must have JSON type {declared_type}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        _fail("schema_const", f"{context} must equal the schema constant")
    if "enum" in schema and not any(_json_equal(value, candidate) for candidate in schema["enum"]):
        _fail("schema_enum", f"{context} is outside the allowed enumeration")

    if isinstance(value, str):
        minimum = schema.get("minLength")
        if minimum is not None and len(value) < minimum:
            _fail("schema_min_length", f"{context} is shorter than {minimum}")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            _fail("schema_pattern", f"{context} does not match {pattern!r}")

    if isinstance(value, list):
        minimum_items = schema.get("minItems")
        if minimum_items is not None and len(value) < minimum_items:
            _fail("schema_min_items", f"{context} requires at least {minimum_items} items")
        if schema.get("uniqueItems"):
            encoded = [canonical_json(item) for item in value]
            if len(set(encoded)) != len(encoded):
                _fail("schema_unique_items", f"{context} items must be unique")
        item_schema = schema.get("items")
        if item_schema is not None:
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, root_schema, f"{context}[{index}]")

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            _fail("schema_required", f"{context} is missing fields {sorted(missing)}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extras = sorted(set(value) - set(properties))
            if extras:
                _fail("schema_unknown_field", f"{context} contains unknown fields {extras}")
        for key, property_schema in properties.items():
            if key in value:
                _validate_schema_value(value[key], property_schema, root_schema, f"{context}.{key}")

    for sub_schema in schema.get("allOf", []):
        _validate_schema_value(value, sub_schema, root_schema, context)
    condition = schema.get("if")
    if condition is not None:
        branch = schema.get("then") if _schema_matches(value, condition, root_schema) else schema.get("else")
        if branch is not None:
            _validate_schema_value(value, branch, root_schema, context)


def validate_against_bundled_schema(registry: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Validate the structural subset used by the checked-in V5 schema."""

    _validate_schema_value(registry, schema, schema, "registry")


def _safe_id(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        _fail("invalid_identifier", f"{context} must be a stable nonempty identifier")
    return value


def _string_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        _fail("invalid_string_list", f"{context} must be a list of nonempty strings")
    if len(set(value)) != len(value):
        _fail("duplicate_list_item", f"{context} must not contain duplicates")
    return tuple(value)


def _index(records: Any, id_field: str, context: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list) or not records:
        _fail("empty_registry_section", f"{context} must be a nonempty list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail("invalid_registry_record", f"{context}[{position}] must be an object")
        identifier = _safe_id(record.get(id_field), f"{context}[{position}].{id_field}")
        if identifier in indexed:
            _fail("duplicate_identifier", f"duplicate {context} identifier {identifier!r}")
        indexed[identifier] = record
    return indexed


def _check_refs(
    references: Any,
    available: Mapping[str, Any],
    context: str,
    *,
    allow_blocked: bool = True,
) -> None:
    for reference in _string_list(references, context):
        if allow_blocked and reference in {BLOCKED, "NOT_APPLICABLE"}:
            continue
        if reference not in available:
            _fail("unknown_reference", f"{context} references unknown identifier {reference!r}")


def _check_ref(value: Any, available: Mapping[str, Any], context: str) -> None:
    reference = _safe_id(value, context)
    if reference not in {BLOCKED, "NOT_APPLICABLE"} and reference not in available:
        _fail("unknown_reference", f"{context} references unknown identifier {reference!r}")


def _canonical_decimal(value: Any, context: str) -> Decimal:
    if not isinstance(value, str) or not value:
        _fail("invalid_decimal", f"{context} must be a canonical decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation:
        _fail("invalid_decimal", f"{context} is not a decimal")
    if not number.is_finite() or number.is_zero() and value.startswith("-"):
        _fail("invalid_decimal", f"{context} must be finite and must not be negative zero")
    if "e" in value.lower() or value.startswith("+"):
        _fail("noncanonical_decimal", f"{context} must not use exponent or leading plus notation")
    integer, dot, fraction = value.partition(".")
    if not re.fullmatch(r"-?(0|[1-9][0-9]*)", integer):
        _fail("noncanonical_decimal", f"{context} has noncanonical integer digits")
    if dot and (not fraction or fraction.endswith("0") or not fraction.isdigit()):
        _fail("noncanonical_decimal", f"{context} has noncanonical fractional digits")
    return number


def _contains_binary_float(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, list):
        return any(_contains_binary_float(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_binary_float(item) for item in value.values())
    return False


def _contains_blocked(value: Any) -> bool:
    if value == BLOCKED:
        return True
    if isinstance(value, list):
        return any(_contains_blocked(item) for item in value)
    if isinstance(value, Mapping):
        return any(_contains_blocked(item) for item in value.values())
    return False


def _validate_project_binding(registry: Mapping[str, Any], project_root: Path | None) -> None:
    if project_root is None:
        return
    framework = registry["framework"]
    relative = Path(framework["scientific_contract_path"])
    if relative.is_absolute() or ".." in relative.parts:
        _fail("unsafe_contract_path", "scientific contract path must be repository-relative")
    path = project_root / relative
    if not path.is_file() or path.is_symlink():
        _fail("missing_scientific_contract", f"scientific contract is missing: {path}")
    if sha256_file(path) != framework["scientific_contract_sha256"]:
        _fail("scientific_contract_hash_mismatch", "scientific contract hash does not match live bytes")
    if str(path.stat().st_size) != framework["scientific_contract_size_bytes"]:
        _fail("scientific_contract_size_mismatch", "scientific contract size does not match live bytes")


def _validate_parameter_resolution(parameter: Mapping[str, Any]) -> None:
    identifier = parameter["parameter_id"]
    resolution = parameter["resolution"]
    uncertainty = parameter["uncertainty"]
    state = resolution["state"]
    value = resolution["value"]
    if state == BLOCKED:
        if value != BLOCKED or not resolution["reason_code"] or not resolution["rationale"]:
            _fail("invalid_tbd_parameter", f"parameter {identifier} must use an explicit blocked value and rationale")
    elif state == "NOT_APPLICABLE":
        if value != "NOT_APPLICABLE" or not resolution["evidence_refs"]:
            _fail("invalid_not_applicable", f"parameter {identifier} requires evidence and no numeric value")
    elif state == "VALUE":
        if parameter["value_kind"] == "SCALAR":
            _canonical_decimal(value, f"parameter {identifier} value")
        elif value in {BLOCKED, "NOT_APPLICABLE"}:
            _fail("invalid_parameter_value", f"parameter {identifier} has no resolved value")
        if uncertainty["state"] == BLOCKED:
            _fail("unresolved_uncertainty", f"parameter {identifier} has unresolved uncertainty")
    else:
        _fail("invalid_resolution_state", f"parameter {identifier} has unsupported resolution state")

    if parameter["required_for_execution"] and state == BLOCKED:
        return
    if uncertainty["state"] == "VALUE" and parameter["value_kind"] == "SCALAR":
        _canonical_decimal(uncertainty["value"], f"parameter {identifier} uncertainty")


def _validate_covariance(block: Mapping[str, Any], parameters: Mapping[str, Any]) -> None:
    identifier = block["covariance_block_id"]
    parameter_ids = _string_list(block["parameter_ids"], f"covariance {identifier}.parameter_ids")
    _check_refs(list(parameter_ids), parameters, f"covariance {identifier}.parameter_ids", allow_blocked=False)
    if block["status"] == BLOCKED:
        if block["blocking_issue"] == "":
            _fail("missing_blocking_issue", f"covariance {identifier} requires a blocking issue")
        return
    if block["status"] != "VALUE":
        return
    sigmas = block["sigmas"]
    if len(sigmas) != len(parameter_ids):
        _fail("covariance_shape", f"covariance {identifier} sigma count is wrong")
    sigma_ids = [sigma["parameter_id"] for sigma in sigmas]
    if sigma_ids != list(parameter_ids):
        _fail("covariance_order", f"covariance {identifier} sigma order must match parameter_ids")
    for sigma in sigmas:
        if _canonical_decimal(sigma["value"], f"covariance {identifier} sigma") <= 0:
            _fail("covariance_sigma", f"covariance {identifier} sigmas must be positive")
    matrix = block["correlation_matrix"]
    if len(matrix) != len(parameter_ids) or any(len(row) != len(parameter_ids) for row in matrix):
        _fail("covariance_shape", f"covariance {identifier} matrix must be square")
    numeric = [[_canonical_decimal(value, f"covariance {identifier} correlation") for value in row] for row in matrix]
    for i, row in enumerate(numeric):
        if row[i] != 1:
            _fail("covariance_diagonal", f"covariance {identifier} correlation diagonal must be one")
        for j, value in enumerate(row):
            if value < -1 or value > 1 or value != numeric[j][i]:
                _fail("covariance_symmetry", f"covariance {identifier} correlation matrix is invalid")


@dataclass(frozen=True)
class RegistryInspection:
    schema: str
    registry_id: str
    registry_state: str
    registry_sha256: str
    execution_authorized: bool
    force_model_count: int
    force_families: tuple[str, ...]
    unresolved_parameter_ids: tuple[str, ...]
    blocked_force_model_ids: tuple[str, ...]
    collision_capability_states: tuple[str, ...]
    measurement_capability_states: tuple[str, ...]
    claim_state: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_registry_semantics(
    registry: Mapping[str, Any],
    *,
    project_root: str | Path | None = None,
) -> RegistryInspection:
    """Validate V5 cross-references, resolution semantics, and claim boundary."""

    if registry.get("schema") != REGISTRY_SCHEMA:
        _fail("schema_mismatch", f"registry schema must be {REGISTRY_SCHEMA}")
    if _contains_binary_float(registry):
        _fail("binary_float_forbidden", "registry scientific values must not use JSON binary floats")
    _safe_id(registry["registry_id"], "registry_id")

    framework = registry["framework"]
    if tuple(framework["evidence_classes"]) != _EVIDENCE_CLASSES:
        _fail("evidence_class_roster", "framework evidence classes differ from the scientific contract")
    _validate_project_binding(registry, None if project_root is None else Path(project_root).resolve())

    lineage = registry["lineage"]
    if not lineage["v4_immutable"] or any(
        lineage[field] for field in ("v4_may_be_read", "v4_may_be_ingested", "v4_may_be_modified")
    ):
        _fail("lineage_policy", "V4 must remain immutable, unread, un-ingested, and unmodified")
    if lineage["v4_artifact_references"]:
        _fail("lineage_reference", "the V5 foundation must not reference V4 artifacts")

    policy = registry["execution_policy"]
    for field in (
        "tbd_blocks_execution",
        "not_applicable_requires_evidence",
        "bounded_omission_requires_error_budget",
        "unknown_keys_rejected",
        "implicit_defaults_forbidden",
    ):
        if policy[field] is not True:
            _fail("unsafe_execution_policy", f"execution policy {field} must be true")

    units = _index(registry["unit_definitions"], "unit_id", "unit_definitions")
    frames = _index(registry["frames"], "frame_id", "frames")
    validity = _index(registry["validity_intervals"], "validity_id", "validity_intervals")
    bodies = _index(registry["bodies"], "body_id", "bodies")
    sources = _index(registry["provenance_sources"], "provenance_id", "provenance_sources")
    parameters = _index(registry["parameters"], "parameter_id", "parameters")
    covariances = _index(registry["covariance_blocks"], "covariance_block_id", "covariance_blocks")
    models = _index(registry["force_models"], "model_id", "force_models")
    collisions = _index(registry["collision_capabilities"], "capability_id", "collision_capabilities")
    measurements = _index(registry["measurement_capabilities"], "capability_id", "measurement_capabilities")

    for identifier, unit in units.items():
        _check_refs(unit["provenance_refs"], sources, f"unit {identifier}.provenance_refs")
        if unit["definition_status"] == "DEFINED" and not unit["provenance_refs"]:
            _fail("unprovenanced_unit", f"defined unit {identifier} requires provenance")

    for identifier, interval in validity.items():
        _check_refs(interval["provenance_refs"], sources, f"validity {identifier}.provenance_refs")

    for identifier, frame in frames.items():
        _check_ref(frame["validity_ref"], validity, f"frame {identifier}.validity_ref")
        _check_refs(frame["provenance_refs"], sources, f"frame {identifier}.provenance_refs")

    for identifier, source in sources.items():
        if source["status"] == "VERIFIED":
            if _SHA256.fullmatch(source["sha256"]) is None or source["sha256"] == "0" * 64:
                _fail("invalid_provenance_hash", f"verified source {identifier} requires a real SHA-256")
            if not source["size_bytes"].isdigit() or int(source["size_bytes"]) <= 0:
                _fail("invalid_provenance_size", f"verified source {identifier} requires a positive byte size")
        elif source["sha256"] != BLOCKED or source["size_bytes"] != BLOCKED:
            _fail("premature_provenance_value", f"blocked source {identifier} must not claim resolved bytes")

    for identifier, body in bodies.items():
        _check_ref(body["gm_parameter_ref"], parameters, f"body {identifier}.gm_parameter_ref")
        _check_ref(body["radius_parameter_ref"], parameters, f"body {identifier}.radius_parameter_ref")
        _check_refs(body["spin_parameter_refs"], parameters, f"body {identifier}.spin_parameter_refs")
        _check_ref(body["frame_ref"], frames, f"body {identifier}.frame_ref")
        _check_ref(body["validity_ref"], validity, f"body {identifier}.validity_ref")
        _check_refs(body["provenance_refs"], sources, f"body {identifier}.provenance_refs")

    for identifier, parameter in parameters.items():
        _check_ref(parameter["unit_id"], units, f"parameter {identifier}.unit_id")
        _check_ref(parameter["frame_ref"], frames, f"parameter {identifier}.frame_ref")
        _check_ref(parameter["validity_ref"], validity, f"parameter {identifier}.validity_ref")
        _check_refs(parameter["body_refs"], bodies, f"parameter {identifier}.body_refs")
        _check_refs(parameter["provenance_refs"], sources, f"parameter {identifier}.provenance_refs")
        _check_refs(parameter["resolution"]["evidence_refs"], sources, f"parameter {identifier}.resolution.evidence_refs")
        _check_ref(parameter["uncertainty"]["unit_id"], units, f"parameter {identifier}.uncertainty.unit_id")
        covariance_ref = parameter["uncertainty"]["covariance_block_ref"]
        if covariance_ref not in {BLOCKED, "NOT_APPLICABLE"} and covariance_ref not in covariances:
            _fail("unknown_reference", f"parameter {identifier} references unknown covariance {covariance_ref!r}")
        _check_refs(parameter["uncertainty"]["provenance_refs"], sources, f"parameter {identifier}.uncertainty.provenance_refs")
        _validate_parameter_resolution(parameter)

    for block in covariances.values():
        _check_ref(block["frame_ref"], frames, f"covariance {block['covariance_block_id']}.frame_ref")
        _check_ref(block["validity_ref"], validity, f"covariance {block['covariance_block_id']}.validity_ref")
        _check_refs(block["provenance_refs"], sources, f"covariance {block['covariance_block_id']}.provenance_refs")
        _validate_covariance(block, parameters)

    model_families: set[str] = set()
    for identifier, model in models.items():
        model_families.add(model["family"])
        _check_refs(model["parameter_refs"], parameters, f"model {identifier}.parameter_refs")
        _check_refs(model["source_body_refs"], bodies, f"model {identifier}.source_body_refs")
        _check_refs(model["target_body_refs"], bodies, f"model {identifier}.target_body_refs")
        _check_ref(model["frame_ref"], frames, f"model {identifier}.frame_ref")
        _check_ref(model["validity_ref"], validity, f"model {identifier}.validity_ref")
        _check_refs(model["provenance_refs"], sources, f"model {identifier}.provenance_refs")
        _check_refs(model["exclusive_with"], models, f"model {identifier}.exclusive_with", allow_blocked=False)
        if model["treatment"] == "MODELED" and (
            model["implementation_status"] != "IMPLEMENTED"
            or model["qualification_status"] != "QUALIFIED"
        ):
            if not model["blocking_reasons"]:
                _fail("unqualified_model", f"model {identifier} must be blocked with an explicit reason")

    required_families = {"NEWTONIAN_POINT_MASS", "RELATIVITY", "GRAVITY_HARMONICS", "NONGRAVITATIONAL"}
    missing_families = sorted(required_families - model_families)
    if missing_families:
        _fail("missing_force_family", f"registry omits required force families {missing_families}")

    applicability_keys: set[tuple[str, str, str]] = set()
    for position, row in enumerate(registry["applicability"]):
        _check_ref(row["model_id"], models, f"applicability[{position}].model_id")
        key = (row["model_id"], row["subject_kind"], row["subject_ref"])
        if key in applicability_keys:
            _fail("duplicate_applicability", f"duplicate applicability row {key}")
        applicability_keys.add(key)
        if row["subject_kind"] in {"BODY", "BODY_SET"}:
            _check_ref(row["subject_ref"], bodies, f"applicability[{position}].subject_ref")
        _check_refs(row["evidence_refs"], sources, f"applicability[{position}].evidence_refs")
        if row["status"] == "NOT_APPLICABLE" and not row["evidence_refs"]:
            _fail("unproven_not_applicable", f"applicability row {key} requires evidence")
        if row["treatment"] == "BOUNDED_OMISSION" and (
            row["error_bound_parameter_ref"] in {BLOCKED, "NOT_APPLICABLE"}
            or row["error_budget_ref"] in {BLOCKED, "NOT_APPLICABLE"}
        ):
            _fail("unbounded_omission", f"applicability row {key} lacks a bound or error budget")

    for identifier, capability in collisions.items():
        _check_refs(capability["radius_parameter_refs"], parameters, f"collision {identifier}.radius_parameter_refs")
        if capability["status"] != "IMPLEMENTED" and not capability["blocking_reasons"]:
            _fail("unexplained_capability_status", f"collision capability {identifier} requires a blocking reason")

    for identifier, capability in measurements.items():
        _check_refs(capability["required_correction_capability_ids"], measurements, f"measurement {identifier}.required_correction_capability_ids", allow_blocked=False)
        _check_refs(capability["input_frame_refs"], frames, f"measurement {identifier}.input_frame_refs")
        _check_ref(capability["output_frame_ref"], frames, f"measurement {identifier}.output_frame_ref")
        _check_refs(capability["parameter_refs"], parameters, f"measurement {identifier}.parameter_refs")
        _check_refs(capability["provenance_refs"], sources, f"measurement {identifier}.provenance_refs")
        if capability["status"] != "IMPLEMENTED" and not capability["blocking_reasons"]:
            _fail("unexplained_capability_status", f"measurement capability {identifier} requires a blocking reason")

    claims = registry["claim_boundaries"]
    if claims["simulation_output_evidence_class"] != "MODEL_OUTPUT" or claims["external_review_required"] is not True:
        _fail("claim_boundary", "simulation output must remain MODEL_OUTPUT and externally reviewed")
    if set(claims["permitted_claims"]) & set(claims["prohibited_claims"]):
        _fail("claim_boundary", "permitted and prohibited claims overlap")
    if not claims["prohibited_claims"]:
        _fail("claim_boundary", "registry must enumerate prohibited claims")

    unresolved = tuple(
        sorted(
            identifier
            for identifier, parameter in parameters.items()
            if parameter["resolution"]["state"] == BLOCKED
            or parameter["uncertainty"]["state"] == BLOCKED
        )
    )
    blocked_models = tuple(
        sorted(
            identifier
            for identifier, model in models.items()
            if model["implementation_status"] != "IMPLEMENTED"
            or model["qualification_status"] != "QUALIFIED"
            or model["treatment"] == "BLOCKED"
            or model["applicability_status"] == BLOCKED
        )
    )
    executable = (
        registry["state"] == "FROZEN_EXECUTABLE"
        and policy["executable"] is True
        and policy["blocking_state"] == "NONE"
        and not unresolved
        and not blocked_models
        and not _contains_blocked(registry)
    )
    if registry["state"] == "DRAFT_NONEXECUTABLE" and (
        policy["executable"] is not False or policy["blocking_state"] != BLOCKED
    ):
        _fail("draft_execution_policy", "draft registry must be nonexecutable and blocked")
    if registry["state"] == "FROZEN_EXECUTABLE" and not executable:
        _fail("false_executable_claim", "frozen registry contains unresolved or unqualified content")

    return RegistryInspection(
        schema=INSPECTION_SCHEMA,
        registry_id=registry["registry_id"],
        registry_state=registry["state"],
        registry_sha256=sha256_data(registry),
        execution_authorized=executable,
        force_model_count=len(models),
        force_families=tuple(sorted(model_families)),
        unresolved_parameter_ids=unresolved,
        blocked_force_model_ids=blocked_models,
        collision_capability_states=tuple(sorted({item["status"] for item in collisions.values()})),
        measurement_capability_states=tuple(sorted({item["status"] for item in measurements.values()})),
        claim_state=claims["registry_claim_state"],
    )


def inspect_registry_file(
    registry_path: str | Path,
    *,
    schema_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[dict[str, Any], RegistryInspection]:
    """Load and inspect a registry without granting execution authority."""

    registry_source = Path(registry_path).resolve()
    if schema_path is None:
        candidate_root = registry_source.parent.parent
        schema_source = candidate_root / "schemas" / DEFAULT_SCHEMA_NAME
    else:
        schema_source = Path(schema_path).resolve()
        candidate_root = schema_source.parent.parent
    root = candidate_root if project_root is None else Path(project_root).resolve()
    registry = _load_json(registry_source, "registry")
    schema = _load_json(schema_source, "registry schema")
    validate_against_bundled_schema(registry, schema)
    inspection = validate_registry_semantics(registry, project_root=root)
    normalized = json.loads(canonical_json(registry).decode("utf-8"))
    return normalized, inspection


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def load_executable_registry(
    registry_path: str | Path,
    *,
    schema_path: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Mapping[str, Any]:
    """Return an immutable registry only after all execution gates pass."""

    registry, inspection = inspect_registry_file(
        registry_path,
        schema_path=schema_path,
        project_root=project_root,
    )
    if not inspection.execution_authorized:
        _fail(
            "registry_not_executable",
            f"registry {inspection.registry_id!r} is {inspection.registry_state} and grants no execution authority",
        )
    return _freeze(registry)

