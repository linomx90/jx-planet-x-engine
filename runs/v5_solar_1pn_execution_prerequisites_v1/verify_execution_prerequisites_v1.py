"""Read-only verifier for the blocked V5 Solar 1PN prerequisite design."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from pathlib import Path
from typing import Any


PACKAGE_DIRECTORY = "runs/v5_solar_1pn_execution_prerequisites_v1"
REGISTRATION_PATH = f"{PACKAGE_DIRECTORY}/registration_v1.json"
PACKAGE_SCHEMA_REFERENCE = (
    "../../schemas/jx-v5-solar-1pn-execution-prerequisite-registration-v1.schema.json"
)
PACKAGE_SCHEMA_ID = "jx-v5-solar-1pn-execution-prerequisite-registration/v1"
PACKAGE_ID = "jx.v5.solar_1pn.execution_prerequisites.v1"
PACKAGE_NONCLAIM = (
    "This additive package is design-only blocked metadata. It contains no "
    "execution implementation, numerical outcome, scientific evidence, or authority."
)
Q2_SCHEMA_REFERENCE = "../../schemas/jx-v5-solar-1pn-q2-observer-registration-v1.schema.json"
Q2_SCHEMA_ID = "jx-v5-solar-1pn-q2-observer-registration/v1"
Q2_REGISTRATION_ID = "jx.v5.solar_1pn.q2_observer_registration.v1"
Q2_NONCLAIM = (
    "This null registration supplies no observer, runtime, rule completion, "
    "expectation, budget, result, or execution authority."
)
Q3_SCHEMA_REFERENCE = "../../schemas/jx-v5-solar-1pn-q3-oracle-registration-v1.schema.json"
Q3_SCHEMA_ID = "jx-v5-solar-1pn-q3-oracle-registration/v1"
Q3_REGISTRATION_ID = "jx.v5.solar_1pn.q3_oracle_registration.v1"
Q3_NONCLAIM = (
    "This null registration supplies no oracle, coefficients, controller, scales, "
    "norms, self-convergence result, expectation, or execution authority."
)
Q4_SCHEMA_REFERENCE = "../../schemas/jx-v5-solar-1pn-q4-eih-registration-v1.schema.json"
Q4_SCHEMA_ID = "jx-v5-solar-1pn-q4-eih-registration/v1"
Q4_REGISTRATION_ID = "jx.v5.solar_1pn.q4_eih_registration.v1"
Q4_NONCLAIM = (
    "This null registration supplies no EIH evaluator, runtime, fixtures, convention "
    "review, backreaction result, expectation, budget, or execution authority."
)
QUALIFICATION_ID = (
    "jx.v5.solar_1pn.qualification."
    "78d05102bac2588b7677ec5ef19c653c0102923a1ab499852194f423c8911781"
)
PREDECESSOR_PACKAGE_SHA256 = (
    "80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237"
)
PREDECESSOR_COMMIT = "975b1b7002e358a980457acb18c9beaa90aa1c9f"
BLOCKED_REASONS = (
    "Q2_OBSERVER_UNREGISTERED",
    "Q2_RULES_EXPECTATION_BUDGET_UNRESOLVED",
    "Q3_ORACLE_UNREGISTERED",
    "Q3_SCALES_NORMS_SELF_CONVERGENCE_UNRESOLVED",
    "Q4_EIH_EVALUATOR_UNREGISTERED",
    "Q4_FIXTURES_CONVENTION_BACKREACTION_BUDGET_UNRESOLVED",
    "EXTERNAL_INDEPENDENCE_UNATTESTED",
    "SEALED_EXPECTATIONS_UNAVAILABLE",
    "EXECUTION_NOT_AUTHORIZED",
)
CLAIM_CONTROL = {
    "permitted_claim": "DESIGN_ONLY_BLOCKED_NONAUTHORIZING",
    "prohibited_claims": [
        "NO_QUALIFICATION_OUTCOME",
        "NO_EXECUTION_AUTHORITY",
        "NO_REGISTRY_AUTHORITY",
        "NO_OBSERVER_IMPLEMENTATION",
        "NO_ORACLE_IMPLEMENTATION",
        "NO_EIH_IMPLEMENTATION",
        "NO_INDEPENDENCE_ATTESTATION",
        "NO_SEALED_EXPECTATION",
        "NO_ERROR_BUDGET",
        "NO_SCIENTIFIC_CLAIM",
    ],
}
SCHEMA_BINDINGS = (
    {
        "role": "package_schema",
        "schema": PACKAGE_SCHEMA_ID,
        "path": "schemas/jx-v5-solar-1pn-execution-prerequisite-registration-v1.schema.json",
        "sha256": "e9d04cae4ae5aef4f3586e2dbb531f979b0ba539028dc3ec3cdb6a65d0754ad5",
        "size_bytes": "4825",
        "canonical_sha256": "e9ac191d34fd1ad7242487ddd9684d03d567ab390dafc5b1c9ec2075b62c81dd",
    },
    {
        "role": "q2_schema",
        "schema": Q2_SCHEMA_ID,
        "path": "schemas/jx-v5-solar-1pn-q2-observer-registration-v1.schema.json",
        "sha256": "a8de8d0e19d91b47337d6b11eec280c19d2d4fe1d72e3f4348d14203803befc4",
        "size_bytes": "6883",
        "canonical_sha256": "c2226e2d3ee9c020b010da0c043b90da7238f34bd0c91e330effd96fb4181fe4",
    },
    {
        "role": "q3_schema",
        "schema": Q3_SCHEMA_ID,
        "path": "schemas/jx-v5-solar-1pn-q3-oracle-registration-v1.schema.json",
        "sha256": "52353353fd1d82a488245a9ac6d219dfb981cdfa66248553da1df432e013d2e3",
        "size_bytes": "8252",
        "canonical_sha256": "85db20ffbdbc91a218eac7bec7c0d57ea65f0a797fee0619eb898807dab7a4d2",
    },
    {
        "role": "q4_schema",
        "schema": Q4_SCHEMA_ID,
        "path": "schemas/jx-v5-solar-1pn-q4-eih-registration-v1.schema.json",
        "sha256": "94954937a5c4f3ec7d14475b1f2e5eb5d34c11fcab475b74121543aa1ee95b85",
        "size_bytes": "6522",
        "canonical_sha256": "0f56aa92f358085c010f1a0b6c76a505cf9fd142779d48809509231c766af407",
    },
)
SCHEMA_DOCUMENT_IDS = {
    "package_schema": "https://jx-planet-x-engine.invalid/schemas/jx-v5-solar-1pn-execution-prerequisite-registration-v1.schema.json",
    "q2_schema": "https://jx-planet-x-engine.invalid/schemas/jx-v5-solar-1pn-q2-observer-registration-v1.schema.json",
    "q3_schema": "https://jx-planet-x-engine.invalid/schemas/jx-v5-solar-1pn-q3-oracle-registration-v1.schema.json",
    "q4_schema": "https://jx-planet-x-engine.invalid/schemas/jx-v5-solar-1pn-q4-eih-registration-v1.schema.json",
}
SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "const",
        "enum",
        "items",
        "maxItems",
        "minItems",
        "minLength",
        "pattern",
        "properties",
        "required",
        "title",
        "type",
        "uniqueItems",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPOSITORY_PATH = re.compile(
    r"^(?!/)(?!.*//)(?!.*(?:^|/)\.{1,2}(?:/|$))"
    r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$"
)


class PrerequisiteVerificationError(ValueError):
    """The design registration is malformed, drifted, or overclaims readiness."""


def _reject_float(_: str) -> None:
    raise PrerequisiteVerificationError("binary-float JSON numbers are forbidden")


def _reject_constant(value: str) -> None:
    raise PrerequisiteVerificationError(f"nonfinite JSON constant is forbidden: {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PrerequisiteVerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PrerequisiteVerificationError(f"cannot read UTF-8 JSON: {path}") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, TypeError) as exc:
        raise PrerequisiteVerificationError(f"invalid strict JSON: {path}") from exc
    if type(value) is not dict:
        raise PrerequisiteVerificationError(f"JSON root must be an object: {path}")
    _reject_runtime_numbers(value, str(path))
    return value


def _reject_runtime_numbers(value: Any, context: str) -> None:
    if type(value) is float:
        if not math.isfinite(value):
            raise PrerequisiteVerificationError(f"nonfinite value at {context}")
        raise PrerequisiteVerificationError(f"binary float at {context}")
    if type(value) is dict:
        for key, item in value.items():
            _reject_runtime_numbers(item, f"{context}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _reject_runtime_numbers(item, f"{context}[{index}]")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        raise PrerequisiteVerificationError(
            f"{context} keys differ: missing={sorted(expected - actual)} "
            f"unknown={sorted(actual - expected)}"
        )


def _safe_file(root: Path, repository_path: Any, context: str) -> Path:
    if type(repository_path) is not str or _REPOSITORY_PATH.fullmatch(repository_path) is None:
        raise PrerequisiteVerificationError(f"{context} is not a safe repository path")
    root = root.resolve(strict=True)
    candidate = root.joinpath(*repository_path.split("/"))
    current = root
    for part in repository_path.split("/"):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise PrerequisiteVerificationError(f"{context} does not exist") from exc
        if stat.S_ISLNK(mode):
            raise PrerequisiteVerificationError(f"{context} traverses a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise PrerequisiteVerificationError(f"{context} escapes the repository") from exc
    if not resolved.is_file():
        raise PrerequisiteVerificationError(f"{context} is not a regular file")
    return resolved


def _verify_binding(
    root: Path, binding: dict[str, Any], context: str, *, canonical: bool
) -> tuple[Path, dict[str, Any] | None]:
    expected = {"path", "sha256", "size_bytes"}
    if canonical:
        expected.add("canonical_sha256")
    if type(binding) is not dict:
        raise PrerequisiteVerificationError(f"{context} binding must be an object")
    _keys(binding, expected, context)
    path = _safe_file(root, binding["path"], context)
    raw = path.read_bytes()
    if type(binding["sha256"]) is not str or _SHA256.fullmatch(binding["sha256"]) is None:
        raise PrerequisiteVerificationError(f"{context} SHA-256 is malformed")
    if _sha256_bytes(raw) != binding["sha256"]:
        raise PrerequisiteVerificationError(f"{context} raw SHA-256 differs")
    if binding["size_bytes"] != str(len(raw)):
        raise PrerequisiteVerificationError(f"{context} size differs")
    document = None
    if canonical:
        document = _strict_json(path)
        if _sha256_bytes(_canonical_bytes(document)) != binding["canonical_sha256"]:
            raise PrerequisiteVerificationError(f"{context} canonical SHA-256 differs")
    return path, document


def _schema_error(context: str, message: str) -> None:
    raise PrerequisiteVerificationError(f"{context}: {message}")


def _audit_schema_node(schema: Any, context: str) -> None:
    if type(schema) is not dict:
        _schema_error(context, "schema nodes must be objects")
    unknown = set(schema) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        _schema_error(context, f"unsupported schema keywords: {sorted(unknown)}")

    for keyword in ("$schema", "$id", "title"):
        if keyword in schema and type(schema[keyword]) is not str:
            _schema_error(context, f"{keyword} must be a string")
    if "$ref" in schema:
        reference = schema["$ref"]
        if type(reference) is not str or not reference.startswith("#/"):
            _schema_error(context, "$ref must be a local JSON Pointer")
    if "type" in schema:
        declared = schema["type"]
        if declared not in {
            "object",
            "array",
            "string",
            "integer",
            "number",
            "boolean",
            "null",
        }:
            _schema_error(context, f"unsupported schema type: {declared!r}")
    if "additionalProperties" in schema and type(schema["additionalProperties"]) is not bool:
        _schema_error(context, "additionalProperties must be boolean")
    if "required" in schema:
        required = schema["required"]
        if (
            type(required) is not list
            or any(type(item) is not str for item in required)
            or len(set(required)) != len(required)
        ):
            _schema_error(context, "required must be a unique string array")
    for keyword in ("properties", "$defs"):
        if keyword not in schema:
            continue
        entries = schema[keyword]
        if type(entries) is not dict or any(type(name) is not str for name in entries):
            _schema_error(context, f"{keyword} must be an object")
        for name, child in entries.items():
            _audit_schema_node(child, f"{context}.{keyword}.{name}")
    if "items" in schema:
        _audit_schema_node(schema["items"], f"{context}.items")
    if "allOf" in schema:
        branches = schema["allOf"]
        if type(branches) is not list or not branches:
            _schema_error(context, "allOf must be a nonempty schema array")
        for index, branch in enumerate(branches):
            _audit_schema_node(branch, f"{context}.allOf[{index}]")
    if "enum" in schema:
        values = schema["enum"]
        if type(values) is not list or not values:
            _schema_error(context, "enum must be a nonempty array")
        encoded = [_canonical_bytes(value) for value in values]
        if len(set(encoded)) != len(encoded):
            _schema_error(context, "enum values must be unique")
    if "pattern" in schema:
        pattern = schema["pattern"]
        if type(pattern) is not str:
            _schema_error(context, "pattern must be a string")
        try:
            re.compile(pattern)
        except re.error as exc:
            _schema_error(context, f"pattern is invalid: {exc}")
    for keyword in ("minLength", "minItems", "maxItems"):
        if keyword in schema and (
            type(schema[keyword]) is not int or schema[keyword] < 0
        ):
            _schema_error(context, f"{keyword} must be a nonnegative integer")
    if (
        "minItems" in schema
        and "maxItems" in schema
        and schema["minItems"] > schema["maxItems"]
    ):
        _schema_error(context, "minItems exceeds maxItems")
    if "uniqueItems" in schema and type(schema["uniqueItems"]) is not bool:
        _schema_error(context, "uniqueItems must be boolean")


def _resolve_local_schema_ref(
    root_schema: dict[str, Any], reference: str, context: str
) -> dict[str, Any]:
    current: Any = root_schema
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if type(current) is not dict or key not in current:
            _schema_error(context, f"schema reference does not resolve: {reference!r}")
        current = current[key]
    if type(current) is not dict:
        _schema_error(context, f"schema reference is not an object: {reference!r}")
    return current


def _schema_type_matches(value: Any, declared: str) -> bool:
    if declared == "object":
        return type(value) is dict
    if declared == "array":
        return type(value) is list
    if declared == "string":
        return type(value) is str
    if declared == "integer":
        return type(value) is int
    if declared == "number":
        return type(value) in {int, float}
    if declared == "boolean":
        return type(value) is bool
    if declared == "null":
        return value is None
    _schema_error("schema", f"unsupported type reached validation: {declared!r}")


def _json_equal(left: Any, right: Any) -> bool:
    try:
        return _canonical_bytes(left) == _canonical_bytes(right)
    except (TypeError, ValueError, UnicodeError, RecursionError):
        return False


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    context: str,
    reference_stack: tuple[str, ...] = (),
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if reference in reference_stack:
            _schema_error(context, f"cyclic schema reference is unsupported: {reference!r}")
        target = _resolve_local_schema_ref(root_schema, reference, context)
        _validate_schema_value(
            value,
            target,
            root_schema,
            context,
            (*reference_stack, reference),
        )

    declared = schema.get("type")
    if declared is not None and not _schema_type_matches(value, declared):
        _schema_error(context, f"must have JSON type {declared}")
    if "const" in schema and not _json_equal(value, schema["const"]):
        _schema_error(context, "must equal the schema constant")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        _schema_error(context, "is outside the allowed enumeration")

    if type(value) is str:
        if "minLength" in schema and len(value) < schema["minLength"]:
            _schema_error(context, f"is shorter than minLength {schema['minLength']}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            _schema_error(context, f"does not match pattern {schema['pattern']!r}")

    if type(value) is list:
        if "minItems" in schema and len(value) < schema["minItems"]:
            _schema_error(context, f"has fewer than minItems {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _schema_error(context, f"has more than maxItems {schema['maxItems']}")
        if schema.get("uniqueItems"):
            encoded = [_canonical_bytes(item) for item in value]
            if len(set(encoded)) != len(encoded):
                _schema_error(context, "items are not unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    schema["items"],
                    root_schema,
                    f"{context}[{index}]",
                    reference_stack,
                )

    if type(value) is dict:
        required = schema.get("required", [])
        missing = sorted(set(required) - set(value))
        if missing:
            _schema_error(context, f"is missing required fields: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unknown = sorted(set(value) - set(properties))
            if unknown:
                _schema_error(context, f"contains unknown fields: {unknown}")
        for key, property_schema in properties.items():
            if key in value:
                _validate_schema_value(
                    value[key],
                    property_schema,
                    root_schema,
                    f"{context}.{key}",
                    reference_stack,
                )

    for index, branch in enumerate(schema.get("allOf", [])):
        _validate_schema_value(
            value,
            branch,
            root_schema,
            f"{context}.allOf[{index}]",
            reference_stack,
        )


def _validate_schema(
    document: dict[str, Any], schema: dict[str, Any], context: str
) -> None:
    """Validate the exact closed keyword subset used by the four frozen schemas."""

    _audit_schema_node(schema, f"{context}.schema")
    _validate_schema_value(document, schema, schema, context)


def _awaiting(value: Any, expected_status: str, context: str) -> None:
    if type(value) is not dict:
        raise PrerequisiteVerificationError(f"{context} must be an awaiting object")
    _keys(value, {"status", "value"}, context)
    if value != {"status": expected_status, "value": None}:
        raise PrerequisiteVerificationError(f"{context} is not the frozen null placeholder")


def _common_null_registration(
    document: dict[str, Any],
    *,
    schema_reference: str,
    schema_id: str,
    registration_id: str,
    status: str,
    nonclaim: str,
    context: str,
) -> None:
    for key, expected in (
        ("$schema", schema_reference),
        ("schema", schema_id),
        ("registration_id", registration_id),
        ("status", status),
        ("design_only", True),
        ("outcomes_generated", False),
        ("execution_authorized", False),
        ("registry_authorized", False),
        ("evidence_class", "MODEL_OUTPUT"),
        ("ready", False),
        ("qualification_id", QUALIFICATION_ID),
        ("nonclaim", nonclaim),
    ):
        if document.get(key) != expected:
            raise PrerequisiteVerificationError(f"{context}.{key} differs")


def _verify_q2(document: dict[str, Any]) -> None:
    _keys(document, {"$schema", "schema", "registration_id", "qualification_id", "status", "design_only", "outcomes_generated", "execution_authorized", "registry_authorized", "evidence_class", "protocol_constants", "observer_rules", "external_bindings", "ready", "nonclaim"}, "q2")
    _common_null_registration(
        document,
        schema_reference=Q2_SCHEMA_REFERENCE,
        schema_id=Q2_SCHEMA_ID,
        registration_id=Q2_REGISTRATION_ID,
        status="AWAITING_EXTERNAL_Q2_OBSERVER_REGISTRATION",
        nonclaim=Q2_NONCLAIM,
        context="q2",
    )
    expected_constants = {
        "event_function": "R_DOT_V_ZERO_OUTWARD_PERIHELION",
        "event_direction": "NEGATIVE_TO_POSITIVE",
        "observer_class": "INDEPENDENT_ARBITRARY_PRECISION",
        "angle_class": "HIGH_PRECISION_ORIENTED_ATAN2",
        "binary_float_trigonometry_allowed": False,
        "linear_event_interpolation_allowed": False,
        "paired_force_execution_required": True,
        "step_grid": ["0.04", "0.02", "0.01"],
        "event_tolerance_grid": [
            "0.00000000000000000001",
            "0.000000000000000000000000000001",
        ],
        "eccentricity_roster": ["0.31", "0.47"],
        "orbit_count": 2,
        "analytic_leading_coefficient": "6*pi",
        "scaling_parameter_kind": "INVERSE_C_SQUARED",
    }
    if document["protocol_constants"] != expected_constants:
        raise PrerequisiteVerificationError("q2 protocol constants differ")
    rules = {
        "initial_event_exclusion_rule": "AWAITING_EXTERNAL_RULE",
        "event_bracket_acceptance_rule": "AWAITING_EXTERNAL_RULE",
        "root_method_id": "AWAITING_EXTERNAL_METHOD_ID",
        "root_termination_rule": "AWAITING_EXTERNAL_RULE",
        "orbital_orientation_rule": "AWAITING_EXTERNAL_RULE",
        "angle_unwrap_rule": "AWAITING_EXTERNAL_RULE",
        "orbit_index_rule": "AWAITING_EXTERNAL_RULE",
        "secular_regression_rule": "AWAITING_EXTERNAL_RULE",
        "step_extrapolation_rule": "AWAITING_EXTERNAL_RULE",
        "event_extrapolation_rule": "AWAITING_EXTERNAL_RULE",
        "inverse_c_squared_extrapolation_rule": "AWAITING_EXTERNAL_RULE",
        "analytic_target_expression": "AWAITING_EXTERNAL_EXPRESSION",
        "high_precision_pi_rule": "AWAITING_EXTERNAL_RULE",
    }
    _keys(document["observer_rules"], set(rules), "q2.observer_rules")
    for key, status in rules.items():
        _awaiting(document["observer_rules"][key], status, f"q2.observer_rules.{key}")
    bindings = {
        "observer_source_binding": "AWAITING_EXTERNAL_SOURCE",
        "observer_runtime_binding": "AWAITING_EXTERNAL_RUNTIME",
        "observer_configuration_binding": "AWAITING_EXTERNAL_CONFIGURATION",
        "observer_independence_binding": "AWAITING_EXTERNAL_INDEPENDENCE_ATTESTATION",
        "sealed_expectation_binding": "AWAITING_EXTERNAL_SEALED_EXPECTATION",
        "q2_budget_binding": "AWAITING_EXTERNAL_BUDGET",
    }
    _keys(document["external_bindings"], set(bindings), "q2.external_bindings")
    for key, status in bindings.items():
        _awaiting(document["external_bindings"][key], status, f"q2.external_bindings.{key}")


def _verify_q3(document: dict[str, Any]) -> None:
    _keys(document, {"$schema", "schema", "registration_id", "qualification_id", "status", "design_only", "outcomes_generated", "execution_authorized", "registry_authorized", "evidence_class", "protocol_constants", "method_requirements", "checkpoint_requirements", "self_convergence", "external_bindings", "ready", "nonclaim"}, "q3")
    _common_null_registration(
        document,
        schema_reference=Q3_SCHEMA_REFERENCE,
        schema_id=Q3_SCHEMA_ID,
        registration_id=Q3_REGISTRATION_ID,
        status="AWAITING_EXTERNAL_Q3_ORACLE_REGISTRATION",
        nonclaim=Q3_NONCLAIM,
        context="q3",
    )
    method = {
        "method_id": "AWAITING_EXTERNAL_METHOD_ID", "exact_order": "AWAITING_EXTERNAL_ORDER", "source_citation": "AWAITING_EXTERNAL_SOURCE_CITATION", "method_coefficients_binding": "AWAITING_EXTERNAL_COEFFICIENTS", "rhs_transcription_binding": "AWAITING_EXTERNAL_RHS_TRANSCRIPTION", "force_ledger_binding": "AWAITING_EXTERNAL_FORCE_LEDGER", "error_control_policy": "AWAITING_EXTERNAL_ERROR_CONTROL_POLICY", "dense_output_or_exact_checkpoint_policy": "AWAITING_EXTERNAL_CHECKPOINT_POLICY", "event_handling": "AWAITING_EXTERNAL_EVENT_HANDLING", "controller_binding": "AWAITING_EXTERNAL_CONTROLLER"
    }
    _keys(document["method_requirements"], set(method), "q3.method_requirements")
    for key, status in method.items():
        _awaiting(document["method_requirements"][key], status, f"q3.method_requirements.{key}")
    refs = [
        "schedule.q2.e031.c83.h001.one_pn",
        "schedule.q2.e031.c97.h001.one_pn",
        "schedule.q2.e047.c83.h001.one_pn",
        "schedule.q2.e047.c97.h001.one_pn",
        "schedule.q3.generic.h001.one_pn",
    ]
    checkpoint_requirements = {
        "checkpoint_roster": "AWAITING_EXTERNAL_CHECKPOINT_ROSTER",
        "per_checkpoint_metric_rows": "AWAITING_EXTERNAL_PER_CHECKPOINT_METRICS",
    }
    _keys(document["checkpoint_requirements"], set(checkpoint_requirements), "q3.checkpoint_requirements")
    for key, status in checkpoint_requirements.items():
        _awaiting(document["checkpoint_requirements"][key], status, f"q3.checkpoint_requirements.{key}")
    convergence = {"oracle_allocation_ref": "AWAITING_EXTERNAL_BUDGET", "tolerance_convergence_result_binding": "AWAITING_EXTERNAL_SELF_CONVERGENCE_RESULT", "precision_convergence_result_binding": "AWAITING_EXTERNAL_SELF_CONVERGENCE_RESULT", "pass_attestation_binding": "AWAITING_EXTERNAL_SELF_CONVERGENCE_ATTESTATION"}
    _keys(document["self_convergence"], set(convergence), "q3.self_convergence")
    for key, status in convergence.items():
        _awaiting(document["self_convergence"][key], status, f"q3.self_convergence.{key}")
    bindings = {"oracle_source_binding": "AWAITING_EXTERNAL_SOURCE", "oracle_runtime_binding": "AWAITING_EXTERNAL_RUNTIME", "dependency_lock_binding": "AWAITING_EXTERNAL_DEPENDENCY_LOCK", "oracle_configuration_binding": "AWAITING_EXTERNAL_CONFIGURATION", "external_independence_binding": "AWAITING_EXTERNAL_INDEPENDENCE_ATTESTATION", "sealed_expectation_binding": "AWAITING_EXTERNAL_SEALED_EXPECTATION"}
    _keys(document["external_bindings"], set(bindings), "q3.external_bindings")
    for key, status in bindings.items():
        _awaiting(document["external_bindings"][key], status, f"q3.external_bindings.{key}")
    expected_constants = {
        "minimum_method_order": 8,
        "arithmetic": "ARBITRARY_PRECISION",
        "adaptive": True,
        "no_jx_imports_required": True,
        "force_ledger": [
            "force.newtonian.point_mass",
            "relativity.solar_schwarzschild_test_particle_1pn",
        ],
        "rhs_equation": "dr/dt=v;dv/dt=-mu*r/r^3+mu/(c^2*r^3)*((4*mu/r-v^2)*r+4*(r_dot_v)*v)",
        "checkpoint_schedule_refs": refs,
        "per_checkpoint_required_fields": [
            "checkpoint_id",
            "checkpoint_epoch",
            "length_scale_lk",
            "velocity_scale_vk",
            "differential_signal_position_scale",
            "differential_signal_velocity_scale",
            "signal_floor",
            "component_treatment",
            "total_state_norm",
            "differential_signal_norm",
        ],
        "self_convergence_tolerance_levels": [
            "0.000000000000000000000000000001",
            "0.0000000000000000000000000000000000000001",
        ],
        "self_convergence_precision_context_refs": [
            "decimal.context.precision_50",
            "decimal.context.precision_70",
            "decimal.context.precision_90",
        ],
        "comparison_prohibited_until_self_convergence_passed": True,
    }
    if document["protocol_constants"] != expected_constants:
        raise PrerequisiteVerificationError("q3 protocol constants differ")


def _verify_q4(document: dict[str, Any]) -> None:
    _keys(document, {"$schema", "schema", "registration_id", "qualification_id", "status", "design_only", "outcomes_generated", "execution_authorized", "registry_authorized", "evidence_class", "protocol_constants", "evaluator_requirements", "fixture_slots", "external_bindings", "ready", "nonclaim"}, "q4")
    _common_null_registration(
        document,
        schema_reference=Q4_SCHEMA_REFERENCE,
        schema_id=Q4_SCHEMA_ID,
        registration_id=Q4_REGISTRATION_ID,
        status="AWAITING_EXTERNAL_Q4_EIH_REGISTRATION",
        nonclaim=Q4_NONCLAIM,
        context="q4",
    )
    requirements = {"complete_simultaneous_finite_mass_eih_identity_binding": "AWAITING_EXTERNAL_EIH_IDENTITY", "barycentric_state_policy_binding": "AWAITING_EXTERNAL_BARYCENTRIC_POLICY", "body_exchange_rule": "AWAITING_EXTERNAL_BODY_EXCHANGE_RULE", "source_backreaction_metric": "AWAITING_EXTERNAL_BACKREACTION_METRIC", "comparison_norm": "AWAITING_EXTERNAL_NORM", "q4_budget_binding": "AWAITING_EXTERNAL_BUDGET", "convention_review_binding": "AWAITING_EXTERNAL_CONVENTION_REVIEW"}
    _keys(document["evaluator_requirements"], set(requirements), "q4.evaluator_requirements")
    for key, status in requirements.items():
        _awaiting(document["evaluator_requirements"][key], status, f"q4.evaluator_requirements.{key}")
    roles = ["GENERIC", "RADIAL", "TRANSVERSE", "ROTATED", "BODY_EXCHANGED"]
    slots = document["fixture_slots"]
    if type(slots) is not list or [slot.get("role") for slot in slots] != roles:
        raise PrerequisiteVerificationError("q4 fixture roster differs")
    for index, slot in enumerate(slots):
        if slot != {"role": roles[index], "status": "AWAITING_EXTERNAL_FIXTURE", "binding": None}:
            raise PrerequisiteVerificationError(f"q4 fixture slot {index} is not null")
    bindings = {"eih_source_binding": "AWAITING_EXTERNAL_SOURCE", "eih_runtime_binding": "AWAITING_EXTERNAL_RUNTIME", "dependency_lock_binding": "AWAITING_EXTERNAL_DEPENDENCY_LOCK", "eih_configuration_binding": "AWAITING_EXTERNAL_CONFIGURATION", "external_independence_binding": "AWAITING_EXTERNAL_INDEPENDENCE_ATTESTATION", "sealed_expectation_binding": "AWAITING_EXTERNAL_SEALED_EXPECTATION"}
    _keys(document["external_bindings"], set(bindings), "q4.external_bindings")
    for key, status in bindings.items():
        _awaiting(document["external_bindings"][key], status, f"q4.external_bindings.{key}")
    expected_constants = {
        "coordinate_gauge": "HARMONIC",
        "beta": "1",
        "gamma": "1",
        "retained_order_policy": "NEWTONIAN_SUBSTITUTION_INSIDE_C_MINUS_2_TERMS",
        "fixed_total_mu": True,
        "correction_only_comparison": True,
        "newtonian_base_addition_policy": "ADD_EXACTLY_ONCE_AFTER_CORRECTION_ONLY_COMPARISON",
        "mass_ratio_definition": "nu=q/(1+q)^2",
        "finite_mass_relative_identity": "a_rel(nu)=-mu*n/r^2+mu/(c^2*r^2)*{[(4+2*nu)*mu/r-(1+3*nu)*v^2+(3/2)*nu*rdot^2]*n+(4-2*nu)*rdot*v}",
        "restricted_difference_identity": "a_rel(nu)-a_restricted=nu*mu/(c^2*r^2)*{[2*mu/r-3*v^2+(3/2)*rdot^2]*n-2*rdot*v}",
        "barycentric_position_policy": "x_source=-q*r/(1+q);x_target=r/(1+q)",
        "barycentric_velocity_policy": "v_source=-q*v/(1+q);v_target=v/(1+q)",
        "nu_ladder": ["0.25", "0.1", "0.01"],
        "required_fixture_roles": roles,
    }
    if document["protocol_constants"] != expected_constants:
        raise PrerequisiteVerificationError("q4 protocol constants differ")


def verify_package(project_root: Path | str) -> dict[str, Any]:
    """Verify the immutable design package without writing or executing science."""

    root = Path(project_root)
    loaded_schemas: dict[str, dict[str, Any]] = {}
    for expected_binding in SCHEMA_BINDINGS:
        role = expected_binding["role"]
        core = {
            key: value
            for key, value in expected_binding.items()
            if key not in {"role", "schema"}
        }
        _, schema_document = _verify_binding(
            root, core, f"frozen schema.{role}", canonical=True
        )
        assert schema_document is not None
        if schema_document.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
            raise PrerequisiteVerificationError(f"schema {role} draft differs")
        if schema_document.get("$id") != SCHEMA_DOCUMENT_IDS[role]:
            raise PrerequisiteVerificationError(f"schema {role} identity differs")
        if (
            schema_document.get("type") != "object"
            or schema_document.get("additionalProperties") is not False
        ):
            raise PrerequisiteVerificationError(f"schema {role} is not closed")
        loaded_schemas[role] = schema_document

    registration_path = _safe_file(root, REGISTRATION_PATH, "registration")
    registration = _strict_json(registration_path)
    _validate_schema(registration, loaded_schemas["package_schema"], "registration")
    _keys(registration, {"$schema", "schema", "prerequisite_package_id", "predecessor", "status", "outcomes_generated", "registry_authorized", "execution_authorized", "ready", "evidence_class", "scientific_evidence_artifact", "artifacts", "schemas", "verifier", "readme", "blocked_reasons", "claim_control", "nonclaim"}, "registration")
    expected_identity = {
        "$schema": PACKAGE_SCHEMA_REFERENCE,
        "schema": PACKAGE_SCHEMA_ID,
        "prerequisite_package_id": PACKAGE_ID,
        "nonclaim": PACKAGE_NONCLAIM,
    }
    for key, expected in expected_identity.items():
        if registration.get(key) != expected:
            raise PrerequisiteVerificationError(f"registration.{key} differs")
    expected_predecessor = {
        "qualification_id": QUALIFICATION_ID,
        "package_sha256": PREDECESSOR_PACKAGE_SHA256,
        "commit": PREDECESSOR_COMMIT,
        "plan": {"path": "runs/v5_solar_1pn_qualification/qualification_plan_v1.json", "sha256": "b511ecd55a759e9c8037ac74cdf9e68985cde6b4f66008d4c9531f10c5c3b3dc", "size_bytes": "92080", "canonical_sha256": "9e2a75c6ff7c4c220d31d14348e95f619437e966bcfe517763a2eb57d0953210"},
        "inputs": {"path": "runs/v5_solar_1pn_qualification/qualification_inputs_v1.json", "sha256": "30ae263fc402d2d0d0bf6318c77285be13f8c2a5daa9c8e5667c789a20ab1b7c", "size_bytes": "186336", "canonical_sha256": "57b64944d65d930d4dfcf0a60fd02e5945788a14483c3daa30040257433c4f79"},
        "registration": {"path": "runs/v5_solar_1pn_qualification/registration_v1.json", "sha256": "7e4b839f0842b6e5d23b6b2c1610eb3a1f3024062e4b696d09e1eb9d6b32f545", "size_bytes": "18490", "canonical_sha256": "319939d02d5764e527514fd346615b9b175ac120c194cf4e88435482b31a253c"},
    }
    if registration["predecessor"] != expected_predecessor:
        raise PrerequisiteVerificationError("predecessor binding differs")
    for name in ("plan", "inputs", "registration"):
        _verify_binding(root, registration["predecessor"][name], f"predecessor.{name}", canonical=True)
    expected_flags = {"status": "DESIGN_ONLY_BLOCKED", "outcomes_generated": False, "registry_authorized": False, "execution_authorized": False, "ready": False, "evidence_class": "MODEL_OUTPUT", "scientific_evidence_artifact": False}
    for key, expected in expected_flags.items():
        if registration.get(key) != expected:
            raise PrerequisiteVerificationError(f"registration.{key} differs")
    if tuple(registration["blocked_reasons"]) != BLOCKED_REASONS:
        raise PrerequisiteVerificationError("registration.blocked_reasons differs")
    if registration["claim_control"] != CLAIM_CONTROL:
        raise PrerequisiteVerificationError("registration.claim_control differs")
    if registration["schemas"] != list(SCHEMA_BINDINGS):
        raise PrerequisiteVerificationError("registered frozen schema bindings differ")
    artifact_roles = {
        "q2_observer_null_registration": (
            Q2_SCHEMA_ID,
            f"{PACKAGE_DIRECTORY}/q2_observer_registration_v1.json",
            "q2_schema",
            _verify_q2,
        ),
        "q3_oracle_null_registration": (
            Q3_SCHEMA_ID,
            f"{PACKAGE_DIRECTORY}/q3_oracle_registration_v1.json",
            "q3_schema",
            _verify_q3,
        ),
        "q4_eih_null_registration": (
            Q4_SCHEMA_ID,
            f"{PACKAGE_DIRECTORY}/q4_eih_registration_v1.json",
            "q4_schema",
            _verify_q4,
        ),
    }
    artifacts = registration["artifacts"]
    if type(artifacts) is not list or [item.get("role") for item in artifacts] != list(artifact_roles):
        raise PrerequisiteVerificationError("artifact roster differs")
    for binding in artifacts:
        role = binding["role"]
        expected_schema, expected_path, schema_role, checker = artifact_roles[role]
        if binding.get("schema") != expected_schema or binding.get("path") != expected_path:
            raise PrerequisiteVerificationError(f"artifact {role} schema differs")
        core = {key: value for key, value in binding.items() if key not in {"role", "schema"}}
        _, document = _verify_binding(root, core, f"artifact.{role}", canonical=True)
        assert document is not None
        _validate_schema(document, loaded_schemas[schema_role], f"artifact.{role}")
        checker(document)
    if registration["verifier"].get("path") != f"{PACKAGE_DIRECTORY}/verify_execution_prerequisites_v1.py":
        raise PrerequisiteVerificationError("verifier path differs")
    if registration["readme"].get("path") != f"{PACKAGE_DIRECTORY}/README.md":
        raise PrerequisiteVerificationError("README path differs")
    _verify_binding(root, registration["verifier"], "verifier", canonical=False)
    _verify_binding(root, registration["readme"], "readme", canonical=False)
    return {
        "schema": "jx-v5-solar-1pn-execution-prerequisite-inspection/v1",
        "prerequisite_package_id": PACKAGE_ID,
        "predecessor_package_sha256": PREDECESSOR_PACKAGE_SHA256,
        "registration_sha256": _sha256_bytes(registration_path.read_bytes()),
        "registration_canonical_sha256": _sha256_bytes(_canonical_bytes(registration)),
        **expected_flags,
    }
