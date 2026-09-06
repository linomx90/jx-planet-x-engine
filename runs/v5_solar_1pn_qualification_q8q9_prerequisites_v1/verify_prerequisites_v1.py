"""Read-only, fail-closed verifier for the design-only Q8/Q9 prerequisites.

This module does not expose a command-line entry point, run dynamics, create
outcomes, sign attestations, commit expectations, or authorize unblinding.
External artifacts are accepted only by the in-memory validators; none are
created by this package. Schema checks use the closed subset implemented in
this file; this module is not a general JSON Schema Draft 2020-12 validator.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


QUALIFICATION_ID = (
    "jx.v5.solar_1pn.qualification."
    "78d05102bac2588b7677ec5ef19c653c0102923a1ab499852194f423c8911781"
)
PREDECESSOR_PACKAGE_SHA256 = (
    "80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237"
)
PREDECESSOR_COMMIT = "975b1b7002e358a980457acb18c9beaa90aa1c9f"
PACKAGE_ID = "jx.v5.solar_1pn.q8q9_prerequisites.v1"
REGISTRATION_RELATIVE = Path(
    "runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1/"
    "registration_v1.json"
)
REGISTRATION_SCHEMA_RELATIVE = Path(
    "schemas/jx-v5-solar-1pn-q8q9-prerequisite-registration-v1.schema.json"
)
DOMAIN_SEPARATOR = b"JX-V5-SOLAR-1PN-Q8Q9-SEALED-EXPECTATIONS-V1|"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_PATTERN = re.compile(
    r"^(?:0|[1-9][0-9]*(?:\.[0-9]*[1-9])?|0\.[0-9]*[1-9])$"
)
SCHEMA_DIALECT_URI = "https://json-schema.org/draft/2020-12/schema"
SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "additionalProperties",
        "allOf",
        "const",
        "else",
        "enum",
        "if",
        "items",
        "maxItems",
        "minItems",
        "minLength",
        "oneOf",
        "pattern",
        "properties",
        "required",
        "then",
        "title",
        "type",
        "uniqueItems",
    }
)
SUPPORTED_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "object", "string"}
)

PREDECESSOR_BINDINGS = {
    "plan": {
        "path": "runs/v5_solar_1pn_qualification/qualification_plan_v1.json",
        "sha256": "b511ecd55a759e9c8037ac74cdf9e68985cde6b4f66008d4c9531f10c5c3b3dc",
        "size_bytes": "92080",
        "canonical_sha256": "9e2a75c6ff7c4c220d31d14348e95f619437e966bcfe517763a2eb57d0953210",
    },
    "inputs": {
        "path": "runs/v5_solar_1pn_qualification/qualification_inputs_v1.json",
        "sha256": "30ae263fc402d2d0d0bf6318c77285be13f8c2a5daa9c8e5667c789a20ab1b7c",
        "size_bytes": "186336",
        "canonical_sha256": "57b64944d65d930d4dfcf0a60fd02e5945788a14483c3daa30040257433c4f79",
    },
    "registration": {
        "path": "runs/v5_solar_1pn_qualification/registration_v1.json",
        "sha256": "7e4b839f0842b6e5d23b6b2c1610eb3a1f3024062e4b696d09e1eb9d6b32f545",
        "size_bytes": "18490",
        "canonical_sha256": "319939d02d5764e527514fd346615b9b175ac120c194cf4e88435482b31a253c",
    },
}
PREDECESSOR_PRIOR_MANIFEST_BINDING = {
    "path": "runs/v5_solar_1pn_qualification/prior_development_cases_v1.json",
    "sha256": "c7897e82143cf155933910103d3dd9cca3166b2033a36bbd0d72c3109fe3c8ce",
    "size_bytes": "132803",
    "canonical_sha256": "3444af9afbc39556148a7757e4e5cbf1a99d7be14335b126d6378b8ecae25140",
}

QUALIFICATION_STATES = (
    "SPECIFICATION_ONLY_BLOCKED",
    "READY_NONAUTHORIZING",
    "RUNNING_NONAUTHORIZING",
    "COMPLETE_NONAUTHORIZING",
    "INVALIDATED",
)
CUSTODY_STATES = (
    "DESIGN_ONLY_BLOCKED",
    "AWAITING_EXTERNAL_CUSTODIAN",
    "CUSTODY_ACCEPTED_EXPECTATIONS_NOT_COMMITTED",
    "EXPECTATIONS_COMMITTED_SEALED",
    "EXECUTION_PACKAGE_LOCKED_SEALED",
    "RUNNING_EXPECTATIONS_SEALED",
    "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
    "UNBLIND_REQUESTED",
    "UNBLINDED_FIRST_VALID",
    "ADJUDICATED_RETAINED",
    "INVALIDATED",
)
ERROR_BUDGET_STATES = (
    "UNRESOLVED",
    "DRAFT_UNREVIEWED",
    "FROZEN_AWAITING_EXTERNAL_REVIEW",
    "INDEPENDENTLY_REVIEWED_AND_FROZEN",
)
ERROR_BUDGET_DOCUMENT_STATES = (*ERROR_BUDGET_STATES, "INVALIDATED")
COMPONENTS = (
    "oracle",
    "step",
    "solve",
    "decimal",
    "event",
    "transform",
    "parameter",
    "analytic",
    "model",
)
OBSERVABLES = (
    "observable.q1.acceleration",
    "observable.q2.perihelion_coefficient",
    "observable.q3.total_state",
    "observable.q3.differential_signal",
    "observable.q4.eih_restricted_limit",
    "observable.q5.order_solver",
    "observable.q6.precision_plateau",
    "observable.q7.transform",
)
CASE_ROLES = (
    "GENERIC_3D",
    "ECCENTRIC",
    "LONG_ARC",
    "DOMAIN_INSIDE",
    "DOMAIN_BOUNDARY",
    "DOMAIN_OUTSIDE",
    "TRANSFORM_TWIN",
    "SIGNAL_SCALE",
    "EIH_LIMIT",
)
ADJUDICATION_VERDICTS = (
    "PASS",
    "FAIL",
    "INCONCLUSIVE",
)
PHASE_REQUIREMENTS = (
    {
        "phase": "BEFORE_EXECUTION",
        "required_artifact_roles": [
            "custody_attestation",
            "independence_attestation",
            "sealed_expectation_commitment",
            "error_budget",
        ],
    },
    {
        "phase": "AFTER_EXECUTION_BEFORE_UNBLIND",
        "required_artifact_roles": ["output_manifest_commitment"],
    },
    {
        "phase": "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
        "required_artifact_roles": ["unblinding_record"],
    },
)
BLOCKED_REASONS = (
    "AWAITING_EXTERNAL_CUSTODIAN_ATTESTATION",
    "AWAITING_EXTERNAL_INDEPENDENCE_ATTESTATION",
    "AWAITING_EXTERNAL_SEALED_EXPECTATION_COMMITMENT",
    "AWAITING_EXTERNAL_ERROR_BUDGETS",
    "EXECUTION_NOT_AUTHORIZED",
    "OUTCOMES_NOT_GENERATED",
)

SCHEMA_SPECS = {
    "prerequisite_registration_schema": (
        "jx-v5-solar-1pn-q8q9-prerequisite-registration/v1",
        "schemas/jx-v5-solar-1pn-q8q9-prerequisite-registration-v1.schema.json",
    ),
    "custody_attestation_schema": (
        "jx-v5-solar-1pn-q8q9-custody-attestation/v1",
        "schemas/jx-v5-solar-1pn-q8q9-custody-attestation-v1.schema.json",
    ),
    "independence_attestation_schema": (
        "jx-v5-solar-1pn-q8q9-independence-attestation/v1",
        "schemas/jx-v5-solar-1pn-q8q9-independence-attestation-v1.schema.json",
    ),
    "sealed_expectation_commitment_schema": (
        "jx-v5-solar-1pn-q8q9-sealed-expectation-commitment/v1",
        "schemas/jx-v5-solar-1pn-q8q9-sealed-expectation-commitment-v1.schema.json",
    ),
    "error_budget_schema": (
        "jx-v5-solar-1pn-q8q9-error-budget/v1",
        "schemas/jx-v5-solar-1pn-q8q9-error-budget-v1.schema.json",
    ),
    "unblinding_record_schema": (
        "jx-v5-solar-1pn-q8q9-unblinding-record/v1",
        "schemas/jx-v5-solar-1pn-q8q9-unblinding-record-v1.schema.json",
    ),
}
SCHEMA_CANONICAL_SHA256 = {
    "prerequisite_registration_schema": (
        "0877aaa1358abaaaf16eb4ca4ad40373ee630c1d932c618e09d42a96465f9ba4"
    ),
    "custody_attestation_schema": (
        "a1d2df0edca31c68f5d721a5d15214fe003ddbcaf609ee18a0fb3c5b30ef6b39"
    ),
    "independence_attestation_schema": (
        "7b5f70149b09c2553cca49e9594e84bc02c4d0df7932ae09ceffc79f7f12c8d2"
    ),
    "sealed_expectation_commitment_schema": (
        "772d1ece6d3673e8052215bb1cfd3947c2e10b004be123ed63decfa987077857"
    ),
    "error_budget_schema": (
        "c306badcf5e0889e95b9ec274df26e3b8e02c4bd9451852de98796e9a9c46e87"
    ),
    "unblinding_record_schema": (
        "df2743fc8a35da9f54d8be8e1888ea608d8c2df7b6b33274696c9b03eda99b3b"
    ),
}
SCHEMA_RAW_IDENTITY = {
    "prerequisite_registration_schema": (
        "1044f1564fdabfab43958270e765411319d03a3f38a40d6ef1c47b251377d53c",
        "11453",
    ),
    "custody_attestation_schema": (
        "0f4026f7a7190fa3b7cb450d5e3fbc1e12ce9390d23813e2b8023b180da5dc91",
        "7132",
    ),
    "independence_attestation_schema": (
        "da87bb8d0d2a09a8224709ff473acac22c143943a0998b02b1185a43d0d51503",
        "5289",
    ),
    "sealed_expectation_commitment_schema": (
        "f7fa4ca3950f5a53d2a6ff2cb74b52b6e724aba5a456a9eebfe70f1f0a158e18",
        "5356",
    ),
    "error_budget_schema": (
        "b186a99016eb8d65b245439028d95f6859d124358ff5954e72d82d329429ef91",
        "9266",
    ),
    "unblinding_record_schema": (
        "6b52bce95cb22e155793ea7c14b2e8e923585229ca5d901eeffa11db723f31c5",
        "6021",
    ),
}
EXTERNAL_DOCUMENT_CONSTANTS = {
    "custody_attestation_schema": {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-custody-attestation-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-custody-attestation/v1",
        "attestation_role": "EXTERNAL_HOLDOUT_CUSTODIAN",
        "phase_required": "BEFORE_EXECUTION",
    },
    "independence_attestation_schema": {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-independence-attestation-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-independence-attestation/v1",
        "attestation_role": "EXTERNAL_SCIENTIFIC_INDEPENDENCE_REVIEW",
        "phase_required": "BEFORE_EXECUTION",
    },
    "sealed_expectation_commitment_schema": {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-sealed-expectation-commitment-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-sealed-expectation-commitment/v1",
        "phase_required": "BEFORE_EXECUTION",
    },
    "error_budget_schema": {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-error-budget-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-error-budget/v1",
        "phase_required": "BEFORE_EXECUTION",
    },
    "unblinding_record_schema": {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-unblinding-record-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-unblinding-record/v1",
        "phase_required": "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
    },
}
REGISTRATION_NONCLAIM = (
    "This registration binds only a design and null external-artifact slots. "
    "It is not scientific evidence and grants no execution, unblinding, "
    "registry, or claim authority."
)
EXTERNAL_SLOTS_NONCLAIM = (
    "These are empty external-artifact slots only. No attestation, sealed "
    "expectation, unblinding record, identity, timestamp, signature, or outcome "
    "is present."
)
ERROR_BUDGET_TEMPLATE_NONCLAIM = (
    "This null template contains no numerical allocation or review. All norms, "
    "scopes, margins, derivations, and external reviews remain unresolved; it "
    "cannot support execution or a scientific claim."
)
ARTIFACT_SPECS = {
    "external_artifact_slots": (
        "jx-v5-solar-1pn-q8q9-external-artifact-slots/v1",
        "runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1/"
        "external_artifact_slots_v1.json",
    ),
    "error_budget_template": (
        "jx-v5-solar-1pn-q8q9-error-budget/v1",
        "runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1/"
        "error_budget_template_v1.json",
    ),
}
EXTERNAL_SLOT_SPECS = {
    "custody_attestation": (
        "jx-v5-solar-1pn-q8q9-custody-attestation/v1",
        "AWAITING_EXTERNAL_CUSTODIAN_ATTESTATION",
        "BEFORE_EXECUTION",
    ),
    "independence_attestation": (
        "jx-v5-solar-1pn-q8q9-independence-attestation/v1",
        "AWAITING_EXTERNAL_INDEPENDENCE_ATTESTATION",
        "BEFORE_EXECUTION",
    ),
    "sealed_expectation_commitment": (
        "jx-v5-solar-1pn-q8q9-sealed-expectation-commitment/v1",
        "AWAITING_EXTERNAL_SEALED_EXPECTATION_COMMITMENT",
        "BEFORE_EXECUTION",
    ),
    "unblinding_record": (
        "jx-v5-solar-1pn-q8q9-unblinding-record/v1",
        "NOT_APPLICABLE_PREEXECUTION",
        "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
    ),
}
FORBIDDEN_INSTANCE_NAMES = (
    "custody_attestation_v1.json",
    "independence_attestation_v1.json",
    "sealed_expectation_commitment_v1.json",
    "unblinding_record_v1.json",
)


class Q8Q9PrerequisiteError(ValueError):
    """Stable, machine-readable prerequisite validation failure."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _fail(code: str, message: str) -> None:
    raise Q8Q9PrerequisiteError(code, message)


def _reject_float(value: str) -> None:
    _fail("binary_float_forbidden", f"JSON float {value!r} is forbidden")


def _reject_constant(value: str) -> None:
    _fail("nonfinite_json", f"JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_binary_float_tree(value: Any, context: str = "document") -> None:
    if isinstance(value, float):
        _fail("binary_float_forbidden", f"{context} contains a binary float")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                _fail("noncanonical_json", f"{context} contains a non-string object key")
            _reject_binary_float_tree(item, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_binary_float_tree(item, f"{context}[{index}]")


def canonical_json(value: Any) -> bytes:
    _reject_binary_float_tree(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        _fail("noncanonical_json", f"cannot canonicalize JSON value: {exc}")


def sha256_data(value: Any) -> str:
    return sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail("artifact_read_failed", f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _load_json_bytes(raw: bytes, context: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except Q8Q9PrerequisiteError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        _fail("invalid_json", f"cannot load {context}: {exc}")
    if not isinstance(value, dict):
        _fail("invalid_json_root", f"{context} must have an object root")
    return value


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _fail("artifact_read_failed", f"cannot read {context}: {exc}")
    return _load_json_bytes(raw, context)


def _safe_repository_file(root: Path, relative: str | Path, context: str) -> Path:
    if not isinstance(relative, (str, Path)):
        _fail("invalid_repository_path", f"{context} path has an invalid type")
    candidate = Path(relative)
    if candidate.is_absolute() or not candidate.parts:
        _fail("invalid_repository_path", f"{context} path must be repository-relative")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        _fail("path_escape", f"{context} path contains a forbidden segment")
    current = root
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            _fail("symlink_forbidden", f"{context} crosses a symlink")
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        _fail("artifact_missing", f"cannot resolve {context}: {exc}")
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("path_escape", f"{context} escapes the project root")
    if not resolved.is_file():
        _fail("artifact_not_regular_file", f"{context} must be a regular file")
    return resolved


def _schema_json_equal(left: Any, right: Any) -> bool:
    try:
        return canonical_json(left) == canonical_json(right)
    except Q8Q9PrerequisiteError:
        return False


def _schema_type_matches(value: Any, declared: str) -> bool:
    if declared == "object":
        return isinstance(value, Mapping)
    if declared == "array":
        return isinstance(value, list)
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    _fail("unsupported_schema_type", f"schema type {declared!r} is unsupported")


def _resolve_local_schema_ref(
    root_schema: Mapping[str, Any], reference: Any
) -> Mapping[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        _fail(
            "unsupported_schema_reference",
            f"only local JSON-pointer schema references are supported: {reference!r}",
        )
    current: Any = root_schema
    for encoded in reference[2:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or key not in current:
            _fail(
                "invalid_schema_reference",
                f"schema reference does not resolve: {reference!r}",
            )
        current = current[key]
    if not isinstance(current, Mapping):
        _fail(
            "invalid_schema_reference",
            f"schema reference is not an object: {reference!r}",
        )
    return current


def _require_schema_count(value: Any, keyword: str, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        _fail(
            "invalid_schema_definition",
            f"{context}.{keyword} must be a positive integer",
        )
    return value


def _audit_schema_node(
    schema: Any,
    root_schema: Mapping[str, Any],
    context: str,
) -> None:
    if not isinstance(schema, Mapping):
        _fail("invalid_schema_definition", f"{context} must be a schema object")
    unknown = sorted(set(schema) - SUPPORTED_SCHEMA_KEYWORDS)
    if unknown:
        _fail(
            "unsupported_schema_keyword",
            f"{context} uses unsupported schema keywords {unknown}",
        )

    if "$schema" in schema and schema["$schema"] != SCHEMA_DIALECT_URI:
        _fail(
            "unsupported_schema_dialect",
            f"{context} declares an unsupported schema dialect",
        )
    for annotation in ("$id", "title"):
        if annotation in schema and (
            not isinstance(schema[annotation], str) or not schema[annotation]
        ):
            _fail(
                "invalid_schema_definition",
                f"{context}.{annotation} must be a nonempty string",
            )

    if "$ref" in schema:
        _resolve_local_schema_ref(root_schema, schema["$ref"])
        unsupported_siblings = set(schema) - {"$ref", "$defs"}
        if unsupported_siblings:
            _fail(
                "unsupported_schema_shape",
                f"{context} combines $ref with unsupported sibling constraints",
            )

    declared_type = schema.get("type")
    if declared_type is not None and (
        not isinstance(declared_type, str) or declared_type not in SUPPORTED_SCHEMA_TYPES
    ):
        _fail(
            "unsupported_schema_type",
            f"{context}.type {declared_type!r} is unsupported",
        )

    properties = schema.get("properties")
    required = schema.get("required")
    if declared_type == "object":
        if schema.get("additionalProperties") is not False:
            _fail(
                "schema_not_closed",
                f"{context} object schema must set additionalProperties to false",
            )
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            _fail(
                "schema_not_closed",
                f"{context} object schema must declare properties and required",
            )
        if (
            any(not isinstance(key, str) for key in required)
            or len(set(required)) != len(required)
            or set(required) != set(properties)
        ):
            _fail(
                "schema_not_closed",
                f"{context} must require exactly every declared property",
            )
    elif any(
        keyword in schema
        for keyword in ("additionalProperties", "properties", "required")
    ):
        _fail(
            "unsupported_schema_shape",
            f"{context} uses object keywords without type object",
        )

    if properties is not None:
        for key, child in properties.items():
            if not isinstance(key, str):
                _fail(
                    "invalid_schema_definition",
                    f"{context}.properties contains a non-string key",
                )
            _audit_schema_node(child, root_schema, f"{context}.properties.{key}")

    definitions = schema.get("$defs")
    if definitions is not None:
        if not isinstance(definitions, Mapping):
            _fail("invalid_schema_definition", f"{context}.$defs must be an object")
        for key, child in definitions.items():
            if not isinstance(key, str):
                _fail(
                    "invalid_schema_definition",
                    f"{context}.$defs contains a non-string key",
                )
            _audit_schema_node(child, root_schema, f"{context}.$defs.{key}")

    if "items" in schema:
        if declared_type != "array":
            _fail(
                "unsupported_schema_shape",
                f"{context}.items requires type array",
            )
        _audit_schema_node(schema["items"], root_schema, f"{context}.items")
    if "minItems" in schema:
        _require_schema_count(schema["minItems"], "minItems", context)
    if "maxItems" in schema:
        _require_schema_count(schema["maxItems"], "maxItems", context)
    if "minItems" in schema and "maxItems" in schema and schema["minItems"] > schema[
        "maxItems"
    ]:
        _fail(
            "invalid_schema_definition",
            f"{context}.minItems exceeds maxItems",
        )
    if "uniqueItems" in schema and schema["uniqueItems"] is not True:
        _fail(
            "schema_not_closed",
            f"{context}.uniqueItems may only be the strict value true",
        )
    if any(keyword in schema for keyword in ("minItems", "maxItems", "uniqueItems")) and (
        declared_type != "array"
    ):
        _fail(
            "unsupported_schema_shape",
            f"{context} uses array keywords without type array",
        )

    if "minLength" in schema:
        _require_schema_count(schema["minLength"], "minLength", context)
        if declared_type != "string":
            _fail(
                "unsupported_schema_shape",
                f"{context}.minLength requires type string",
            )
    if "pattern" in schema:
        if declared_type != "string" or not isinstance(schema["pattern"], str):
            _fail(
                "unsupported_schema_shape",
                f"{context}.pattern requires a string schema and string expression",
            )
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            _fail(
                "invalid_schema_pattern",
                f"{context}.pattern is invalid: {exc}",
            )

    if "enum" in schema:
        enumeration = schema["enum"]
        if not isinstance(enumeration, list) or not enumeration:
            _fail("invalid_schema_definition", f"{context}.enum must be nonempty")
        encoded = [canonical_json(item) for item in enumeration]
        if len(set(encoded)) != len(encoded):
            _fail("invalid_schema_definition", f"{context}.enum contains duplicates")

    for group_keyword in ("allOf", "oneOf"):
        if group_keyword in schema:
            group = schema[group_keyword]
            if not isinstance(group, list) or not group:
                _fail(
                    "invalid_schema_definition",
                    f"{context}.{group_keyword} must be a nonempty array",
                )
            for index, child in enumerate(group):
                _audit_schema_node(
                    child,
                    root_schema,
                    f"{context}.{group_keyword}[{index}]",
                )

    if "if" in schema:
        _audit_schema_node(schema["if"], root_schema, f"{context}.if")
        for branch in ("then", "else"):
            if branch in schema:
                _audit_schema_node(schema[branch], root_schema, f"{context}.{branch}")
    elif "then" in schema or "else" in schema:
        _fail(
            "invalid_schema_definition",
            f"{context} has then/else without if",
        )


def _schema_matches(
    value: Any,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
) -> bool:
    try:
        _validate_schema_value(value, schema, root_schema, "conditional", ())
    except Q8Q9PrerequisiteError:
        return False
    return True


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    context: str,
    reference_stack: tuple[str, ...],
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if reference in reference_stack:
            _fail("cyclic_schema_reference", f"{context} enters a cyclic schema reference")
        _validate_schema_value(
            value,
            _resolve_local_schema_ref(root_schema, reference),
            root_schema,
            context,
            (*reference_stack, reference),
        )
        return

    declared_type = schema.get("type")
    if declared_type is not None and not _schema_type_matches(value, declared_type):
        _fail("schema_type", f"{context} must have JSON type {declared_type}")
    if "const" in schema and not _schema_json_equal(value, schema["const"]):
        _fail("schema_const", f"{context} must equal the schema constant")
    if "enum" in schema and not any(
        _schema_json_equal(value, candidate) for candidate in schema["enum"]
    ):
        _fail("schema_enum", f"{context} is outside the allowed enumeration")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            _fail(
                "schema_min_length",
                f"{context} is shorter than {schema['minLength']}",
            )
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            _fail(
                "schema_pattern",
                f"{context} does not match {schema['pattern']!r}",
            )

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            _fail(
                "schema_min_items",
                f"{context} requires at least {schema['minItems']} items",
            )
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _fail(
                "schema_max_items",
                f"{context} permits at most {schema['maxItems']} items",
            )
        if schema.get("uniqueItems") is True:
            encoded = [canonical_json(item) for item in value]
            if len(set(encoded)) != len(encoded):
                _fail("schema_unique_items", f"{context} items must be unique")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema_value(
                    item,
                    schema["items"],
                    root_schema,
                    f"{context}[{index}]",
                    reference_stack,
                )

    if isinstance(value, Mapping):
        missing = sorted(set(schema.get("required", ())) - set(value))
        if missing:
            _fail("schema_required", f"{context} is missing fields {missing}")
        properties = schema.get("properties", {})
        extras = sorted(set(value) - set(properties))
        if schema.get("additionalProperties") is False and extras:
            _fail("schema_unknown_field", f"{context} contains unknown fields {extras}")
        for key, child_schema in properties.items():
            if key in value:
                _validate_schema_value(
                    value[key],
                    child_schema,
                    root_schema,
                    f"{context}.{key}",
                    reference_stack,
                )

    for child_schema in schema.get("allOf", ()):
        _validate_schema_value(
            value, child_schema, root_schema, context, reference_stack
        )
    if "oneOf" in schema:
        matches = sum(
            _schema_matches(value, child_schema, root_schema)
            for child_schema in schema["oneOf"]
        )
        if matches != 1:
            _fail(
                "schema_one_of",
                f"{context} must match exactly one oneOf branch, observed {matches}",
            )
    if "if" in schema:
        branch = (
            schema.get("then")
            if _schema_matches(value, schema["if"], root_schema)
            else schema.get("else")
        )
        if branch is not None:
            _validate_schema_value(value, branch, root_schema, context, reference_stack)


def _validate_schema(
    document: Mapping[str, Any], schema: Mapping[str, Any], context: str
) -> None:
    _reject_binary_float_tree(document, context)
    _reject_binary_float_tree(schema, f"{context} schema")
    _audit_schema_node(schema, schema, f"{context} schema")
    _validate_schema_value(document, schema, schema, context, ())


def _verify_binding(
    root: Path,
    binding: Mapping[str, Any],
    context: str,
    *,
    canonical: bool,
) -> tuple[Path, dict[str, Any] | None]:
    path = _safe_repository_file(root, binding.get("path"), context)
    if str(path.stat().st_size) != binding.get("size_bytes"):
        _fail("size_mismatch", f"{context} byte count differs from its binding")
    if sha256_file(path) != binding.get("sha256"):
        _fail("raw_digest_mismatch", f"{context} raw SHA-256 differs from its binding")
    document = _load_json(path, context) if canonical else None
    if canonical and sha256_data(document) != binding.get("canonical_sha256"):
        _fail(
            "canonical_digest_mismatch",
            f"{context} canonical SHA-256 differs from its binding",
        )
    return path, document


def _load_pinned_schema(
    root: Path, schema_role: str
) -> dict[str, Any]:
    if schema_role not in SCHEMA_SPECS:
        _fail("unknown_schema_role", f"unknown schema role {schema_role!r}")
    _, expected_path = SCHEMA_SPECS[schema_role]
    expected_raw_sha256, expected_size = SCHEMA_RAW_IDENTITY[schema_role]
    schema_path = _safe_repository_file(root, expected_path, f"schema {schema_role}")
    if str(schema_path.stat().st_size) != expected_size:
        _fail(
            "schema_identity_mismatch",
            f"schema {schema_role!r} byte count differs from the pinned identity",
        )
    if sha256_file(schema_path) != expected_raw_sha256:
        _fail(
            "schema_identity_mismatch",
            f"schema {schema_role!r} raw SHA-256 differs from the pinned identity",
        )
    schema = _load_json(schema_path, f"schema {schema_role}")
    if sha256_data(schema) != SCHEMA_CANONICAL_SHA256[schema_role]:
        _fail(
            "schema_identity_mismatch",
            f"schema {schema_role!r} canonical SHA-256 differs from the pinned identity",
        )
    _audit_schema_node(schema, schema, f"schema {schema_role}")
    expected_document_id = (
        "https://jx-planet-x-engine.invalid/schemas/" + Path(expected_path).name
    )
    if schema.get("$schema") != SCHEMA_DIALECT_URI or schema.get("$id") != expected_document_id:
        _fail(
            "schema_identity_mismatch",
            f"schema {schema_role!r} document identity differs",
        )
    return schema


def _enforce_external_document_constants(
    document: Mapping[str, Any], schema_role: str
) -> None:
    expected = EXTERNAL_DOCUMENT_CONSTANTS[schema_role]
    for field, value in expected.items():
        if document.get(field) != value:
            _fail(
                "external_constant_mismatch",
                f"{schema_role}.{field} differs from the frozen JX constant",
            )
    nonclaim = document.get("nonclaim")
    if not isinstance(nonclaim, str) or not nonclaim.strip():
        _fail(
            "external_nonclaim_missing",
            f"{schema_role}.nonclaim must remain an explicit nonclaim",
        )


def _require_exact_keys(
    indexed: Mapping[str, Any], expected: Sequence[str], context: str
) -> None:
    if set(indexed) != set(expected) or len(indexed) != len(expected):
        _fail("incomplete_roster", f"{context} must equal the frozen roster")


def _index_unique(
    records: Any, id_field: str, context: str
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(records, list):
        _fail("invalid_roster", f"{context} must be an array")
    indexed: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, Mapping):
            _fail("invalid_record", f"{context}[{position}] must be an object")
        identifier = record.get(id_field)
        if not isinstance(identifier, str) or not identifier:
            _fail("invalid_identifier", f"{context}[{position}].{id_field} is invalid")
        if identifier in indexed:
            _fail("duplicate_identifier", f"{context} repeats {identifier!r}")
        indexed[identifier] = record
    return indexed


def _parse_nonnegative_decimal(value: Any, context: str) -> Decimal:
    if not isinstance(value, str) or DECIMAL_PATTERN.fullmatch(value) is None:
        _fail("invalid_decimal", f"{context} must be a canonical nonnegative Decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation:
        _fail("invalid_decimal", f"{context} is not a Decimal value")
    if not number.is_finite() or number.is_signed():
        _fail("invalid_decimal", f"{context} must be finite and nonnegative")
    return number


def _exact_decimal_sum(values: Sequence[Decimal]) -> Decimal:
    maximum_fractional_places = max(
        (max(0, -value.as_tuple().exponent) for value in values), default=0
    )
    maximum_integer_places = max(
        (
            max(1, len(value.as_tuple().digits) + value.as_tuple().exponent)
            for value in values
        ),
        default=1,
    )
    exact_precision = (
        maximum_integer_places
        + maximum_fractional_places
        + len(str(max(1, len(values))))
        + 2
    )
    with localcontext() as arithmetic:
        arithmetic.prec = max(64, exact_precision)
        arithmetic.Emin = -999999999
        arithmetic.Emax = 999999999
        return sum(values, Decimal(0))


def _validate_error_budget_semantics(document: Mapping[str, Any]) -> None:
    if document.get("qualification_id") != QUALIFICATION_ID:
        _fail("predecessor_mismatch", "error budget qualification_id is wrong")
    if document.get("predecessor_package_sha256") != PREDECESSOR_PACKAGE_SHA256:
        _fail("predecessor_mismatch", "error budget predecessor digest is wrong")
    if tuple(document.get("component_order", ())) != COMPONENTS:
        _fail("component_roster_mismatch", "the nine Q9 components are not exact")
    observable_index = _index_unique(
        document.get("observables"), "observable_id", "error-budget observables"
    )
    _require_exact_keys(observable_index, OBSERVABLES, "error-budget observables")

    status = document.get("status")
    if status not in ERROR_BUDGET_DOCUMENT_STATES:
        _fail("unknown_budget_status", "error budget status is outside the frozen lifecycle")

    allocated_state_requirements = {
        "DRAFT_UNREVIEWED": (
            "DRAFT_UNREVIEWED",
            "ALLOCATED_UNREVIEWED",
            "AWAITING_EXTERNAL_REVIEW",
        ),
        "FROZEN_AWAITING_EXTERNAL_REVIEW": (
            "FROZEN_AWAITING_EXTERNAL_REVIEW",
            "FROZEN_AWAITING_EXTERNAL_REVIEW",
            "AWAITING_EXTERNAL_REVIEW",
        ),
        "INDEPENDENTLY_REVIEWED_AND_FROZEN": (
            "INDEPENDENTLY_REVIEWED_AND_FROZEN",
            "INDEPENDENTLY_REVIEWED_AND_FROZEN",
            "EXTERNAL_REVIEW_ACCEPTED",
        ),
    }

    for observable_id in OBSERVABLES:
        observable = observable_index[observable_id]
        if observable.get("unit_ref") != "unit.dimensionless":
            _fail("unit_mismatch", f"{observable_id} must retain its frozen unit")
        if observable.get("acceptance_relation") != (
            "DISCREPANCY_LE_TOTAL_AND_EACH_REALIZED_COMPONENT_LE_ALLOCATION"
        ):
            _fail(
                "acceptance_relation_mismatch",
                f"{observable_id} has the wrong acceptance relation",
            )
        component_index = _index_unique(
            observable.get("components"), "component", f"{observable_id}.components"
        )
        _require_exact_keys(component_index, COMPONENTS, f"{observable_id}.components")

        if status == "UNRESOLVED":
            unresolved_fields = (
                ("status", "UNRESOLVED"),
                ("norm_status", "AWAITING_EXTERNAL_NORM"),
                ("norm", None),
                ("scope_status", "AWAITING_EXTERNAL_SCOPE"),
                ("scope", None),
                ("discrimination_margin_status", "AWAITING_EXTERNAL_MARGIN"),
                ("discrimination_margin", None),
                ("total_status", "AWAITING_EXTERNAL_BUDGET"),
                ("total_allocation", None),
            )
            for field, expected in unresolved_fields:
                if observable.get(field) != expected:
                    _fail("premature_budget_value", f"{observable_id}.{field} is not unresolved")
            for component in component_index.values():
                expected = {
                    "status": "AWAITING_EXTERNAL_BUDGET",
                    "allocation": None,
                    "unit_ref": observable["unit_ref"],
                    "derivation_class": None,
                    "evidence_bindings": [],
                    "review_status": "AWAITING_EXTERNAL_REVIEW",
                    "exact_zero_scope_proof": None,
                }
                for field, value in expected.items():
                    if component.get(field) != value:
                        _fail(
                            "premature_budget_value",
                            f"{observable_id}.{component['component']}.{field} is not unresolved",
                        )
            continue

        if status == "INVALIDATED":
            invalidated_observable = {
                "status": "INVALIDATED",
                "norm_status": "INVALIDATED",
                "scope_status": "INVALIDATED",
                "discrimination_margin_status": "INVALIDATED",
                "total_status": "INVALIDATED",
            }
            for field, expected in invalidated_observable.items():
                if observable.get(field) != expected:
                    _fail(
                        "budget_state_mismatch",
                        f"{observable_id}.{field} must be {expected!r} in INVALIDATED",
                    )
            for component_name in COMPONENTS:
                component = component_index[component_name]
                if component.get("unit_ref") != observable["unit_ref"]:
                    _fail(
                        "unit_mismatch",
                        f"{observable_id}.{component_name} has the wrong unit",
                    )
                if component.get("status") != "INVALIDATED" or component.get(
                    "review_status"
                ) != "INVALIDATED":
                    _fail(
                        "budget_state_mismatch",
                        f"{observable_id}.{component_name} is not coherently invalidated",
                    )
            continue

        observable_status, component_status, review_status = allocated_state_requirements[
            status
        ]
        required_observable = {
            "status": observable_status,
            "norm_status": "FROZEN",
            "scope_status": "FROZEN",
            "discrimination_margin_status": "FROZEN",
            "total_status": "CONSERVATIVE_SUM_FROZEN",
        }
        for field, expected in required_observable.items():
            if observable.get(field) != expected:
                _fail(
                    "budget_state_mismatch",
                    f"{observable_id}.{field} is incoherent with {status}",
                )
        if not observable.get("norm") or not observable.get("scope"):
            _fail("budget_not_frozen", f"{observable_id} lacks a frozen norm or scope")
        _parse_nonnegative_decimal(
            observable.get("discrimination_margin"),
            f"{observable_id}.discrimination_margin",
        )

        allocations: list[Decimal] = []
        for component_name in COMPONENTS:
            component = component_index[component_name]
            if component.get("unit_ref") != observable["unit_ref"]:
                _fail("unit_mismatch", f"{observable_id}.{component_name} has the wrong unit")
            if component.get("status") != component_status:
                _fail(
                    "budget_state_mismatch",
                    f"{observable_id}.{component_name} is incoherent with {status}",
                )
            if component.get("review_status") != review_status:
                _fail(
                    "budget_state_mismatch",
                    f"{observable_id}.{component_name} review is incoherent with {status}",
                )
            allocation = _parse_nonnegative_decimal(
                component.get("allocation"), f"{observable_id}.{component_name}.allocation"
            )
            allocations.append(allocation)
            if allocation.is_zero():
                proof = component.get("exact_zero_scope_proof")
                if component.get("derivation_class") != "EXACT_ZERO_PROVED":
                    _fail("unjustified_zero", f"{observable_id}.{component_name} zero is unproved")
                if not isinstance(proof, Mapping) or proof.get("scope_class") != "SYNTHETIC_SAME_EQUATION":
                    _fail("unjustified_zero", f"{observable_id}.{component_name} lacks synthetic proof")
                if not proof.get("proof_statement") or not proof.get("evidence_bindings"):
                    _fail("unjustified_zero", f"{observable_id}.{component_name} proof is incomplete")
            elif component.get("derivation_class") in {None, "EXACT_ZERO_PROVED"}:
                _fail("missing_budget_derivation", f"{observable_id}.{component_name} lacks derivation")
            if not component.get("evidence_bindings"):
                _fail("missing_budget_evidence", f"{observable_id}.{component_name} lacks evidence")

        total = _parse_nonnegative_decimal(
            observable.get("total_allocation"), f"{observable_id}.total_allocation"
        )
        if total != _exact_decimal_sum(allocations):
            _fail("conservative_sum_mismatch", f"{observable_id} total is not the exact nine-term sum")

    external = document.get("external_scientific_review")
    if not isinstance(external, Mapping):
        _fail("invalid_external_review", "external scientific review must be an object")

    if status in {
        "UNRESOLVED",
        "DRAFT_UNREVIEWED",
        "FROZEN_AWAITING_EXTERNAL_REVIEW",
    }:
        expected_external = {
            "status": "AWAITING_EXTERNAL_BUDGET_REVIEW",
            "reviewer_identity": None,
            "reviewer_organization": None,
            "signing_key_id": None,
            "scientific_acceptance_statement": None,
            "reviewed_utc": None,
            "signature": None,
        }
        if external != expected_external:
            _fail(
                "fabricated_external_review",
                f"{status} budget contains external review data",
            )
    elif status == "INDEPENDENTLY_REVIEWED_AND_FROZEN":
        required = (
            "reviewer_identity",
            "reviewer_organization",
            "signing_key_id",
            "scientific_acceptance_statement",
            "reviewed_utc",
            "signature",
        )
        if external.get("status") != "EXTERNAL_BUDGET_REVIEW_ACCEPTED" or any(
            not external.get(field) for field in required
        ):
            _fail("budget_not_reviewed", "external budget review is incomplete")
        _parse_utc(external["reviewed_utc"], "external_scientific_review.reviewed_utc")
    elif external.get("status") != "INVALIDATED":
        _fail(
            "budget_state_mismatch",
            "invalidated budget must invalidate its external-review state",
        )

    expected_postrun_status = (
        "INVALIDATED" if status == "INVALIDATED" else "NOT_APPLICABLE_PREEXECUTION"
    )
    if document.get("postrun_realized_errors_status") != expected_postrun_status:
        _fail(
            "postrun_status_mismatch",
            "realized errors are a separate postrun artifact and cannot mutate this budget",
        )


def compute_sealed_commitment(expectations: Mapping[str, Any], nonce_hex: str) -> str:
    if not isinstance(nonce_hex, str) or SHA256_PATTERN.fullmatch(nonce_hex) is None:
        _fail("nonce_policy_violation", "commitment nonce must be a 256-bit lowercase hex secret")
    return sha256(DOMAIN_SEPARATOR + bytes.fromhex(nonce_hex) + canonical_json(expectations)).hexdigest()


def verify_sealed_reveal(
    commitment_sha256: str, nonce_hex: str, canonical_expectation_bytes: bytes
) -> dict[str, Any]:
    if not isinstance(commitment_sha256, str) or SHA256_PATTERN.fullmatch(commitment_sha256) is None:
        _fail("invalid_commitment", "commitment must be a lowercase SHA-256 digest")
    document = _load_json_bytes(canonical_expectation_bytes, "revealed expectations")
    if canonical_expectation_bytes != canonical_json(document):
        _fail("noncanonical_expectation_bytes", "revealed expectations are not canonical JSON bytes")
    observed = compute_sealed_commitment(document, nonce_hex)
    if observed != commitment_sha256:
        _fail("commitment_reveal_mismatch", "revealed expectations do not match the sealed commitment")
    return document


def transition_state(sequence_name: str, current: str, proposed: str) -> str:
    sequences = {
        "qualification": QUALIFICATION_STATES,
        "custody": CUSTODY_STATES,
        "error_budget": ERROR_BUDGET_STATES,
    }
    if sequence_name not in sequences:
        _fail("unknown_lifecycle", f"unknown lifecycle {sequence_name!r}")
    sequence = sequences[sequence_name]
    if current == "INVALIDATED":
        return "INVALIDATED"
    if current not in sequence:
        _fail("unknown_state", "lifecycle state is outside the frozen sequence")
    if proposed == "INVALIDATED":
        return "INVALIDATED"
    if proposed not in sequence:
        _fail("unknown_state", "lifecycle state is outside the frozen sequence")
    if proposed == current:
        if sequence_name == "custody" and current in {
            "UNBLIND_REQUESTED",
            "UNBLINDED_FIRST_VALID",
        }:
            return "INVALIDATED"
        return current
    position = sequence.index(current)
    if position + 1 < len(sequence) and proposed == sequence[position + 1]:
        return proposed
    return "INVALIDATED"


def validate_unblinding_attempt(custody_state: str, requested_ordinal: int) -> None:
    if custody_state != "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED":
        _fail("unblinding_before_output_commit", "unblinding requires committed outputs")
    if requested_ordinal != 1 or isinstance(requested_ordinal, bool):
        _fail("repeated_unblinding", "only the first valid unblinding is permitted")


def enforce_output_immutability(
    committed_sha256: str, observed_sha256: str, custody_state: str
) -> None:
    committed_position = CUSTODY_STATES.index("OUTPUTS_COMMITTED_EXPECTATIONS_SEALED")
    if custody_state not in CUSTODY_STATES:
        _fail("unknown_state", "custody state is unknown")
    if CUSTODY_STATES.index(custody_state) >= committed_position and committed_sha256 != observed_sha256:
        _fail("output_mutation_after_commit", "committed output bytes changed")


def enforce_budget_immutability(
    frozen_budget_sha256: str, proposed_budget_sha256: str, *, outcomes_generated: bool
) -> None:
    if outcomes_generated and frozen_budget_sha256 != proposed_budget_sha256:
        _fail("post_outcome_budget_edit", "error budgets cannot change after outcomes exist")


def validate_holdout_roster(
    cases: Sequence[Mapping[str, Any]], prior_fingerprints: Sequence[str]
) -> None:
    covered: set[str] = set()
    fingerprints: set[str] = set()
    prior = set(prior_fingerprints)
    for position, case in enumerate(cases):
        roles = case.get("case_roles")
        fingerprint = case.get("scientific_fingerprint_sha256")
        if not isinstance(roles, list) or any(role not in CASE_ROLES for role in roles):
            _fail("invalid_case_role", f"holdout case {position} has invalid roles")
        if not isinstance(fingerprint, str) or SHA256_PATTERN.fullmatch(fingerprint) is None:
            _fail("invalid_fingerprint", f"holdout case {position} lacks a SHA-256 fingerprint")
        if fingerprint in fingerprints:
            _fail("duplicate_holdout", "holdout cases repeat a scientific fingerprint")
        if fingerprint in prior:
            _fail("prior_case_collision", "holdout collides with disclosed development evidence")
        fingerprints.add(fingerprint)
        covered.update(roles)
    missing = set(CASE_ROLES) - covered
    if missing:
        _fail("holdout_role_gap", f"holdout roster omits roles {sorted(missing)}")


def validate_realized_error_record(
    observable_budget: Mapping[str, Any], realized: Mapping[str, Any]
) -> None:
    if realized.get("unit_ref") != observable_budget.get("unit_ref"):
        _fail("unit_mismatch", "realized error and budget units differ")
    if realized.get("norm") != observable_budget.get("norm"):
        _fail("norm_mismatch", "realized error and budget norms differ")
    budgets = _index_unique(
        observable_budget.get("components"), "component", "observable budget components"
    )
    errors = realized.get("components")
    if not isinstance(errors, Mapping) or set(errors) != set(COMPONENTS):
        _fail("component_roster_mismatch", "realized errors need all nine components")
    for name in COMPONENTS:
        allocation = _parse_nonnegative_decimal(budgets[name].get("allocation"), f"{name}.allocation")
        observed = _parse_nonnegative_decimal(errors[name], f"{name}.realized_error")
        if observed > allocation:
            _fail("component_overflow", f"{name} exceeds its individual allocation")
    total_error = _parse_nonnegative_decimal(realized.get("total_error"), "total_error")
    total_budget = _parse_nonnegative_decimal(
        observable_budget.get("total_allocation"), "total_allocation"
    )
    if total_error > total_budget:
        _fail("total_overflow", "realized discrepancy exceeds the total budget")


def _parse_utc(value: Any, context: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("invalid_timestamp", f"{context} must be a UTC timestamp")
    try:
        return datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        _fail("invalid_timestamp", f"{context} is not a valid timestamp")


def validate_external_artifact(
    document: Mapping[str, Any],
    *,
    schema_role: str,
    project_root: str | Path,
    execution_started_utc: str | None = None,
    signature_verifier: Callable[[Mapping[str, Any]], bool] | None = None,
) -> None:
    if schema_role not in SCHEMA_SPECS or schema_role == "prerequisite_registration_schema":
        _fail("unknown_external_schema", f"{schema_role!r} is not an external artifact schema")
    root = Path(project_root).resolve()
    schema = _load_pinned_schema(root, schema_role)
    _enforce_external_document_constants(document, schema_role)
    _validate_schema(document, schema, schema_role)
    if document.get("qualification_id") != QUALIFICATION_ID:
        _fail("predecessor_mismatch", "external artifact qualification_id is wrong")
    if document.get("predecessor_package_sha256") != PREDECESSOR_PACKAGE_SHA256:
        _fail("predecessor_mismatch", "external artifact predecessor digest is wrong")

    if schema_role == "custody_attestation_schema":
        required = (
            "custodian_identity",
            "custodian_organization",
            "signing_key_id",
            "independence_statement",
            "conflicts_of_interest_statement",
            "attested_utc",
            "signature",
        )
        if document.get("status") != "EXTERNAL_CUSTODY_ACCEPTED" or any(
            not document.get(field) for field in required
        ):
            _fail("unsigned_custody_attestation", "custody attestation is incomplete or unsigned")
        for field in (
            "no_prior_expectation_disclosure_attested",
            "holdout_freshness_attested",
            "first_valid_unblinding_only_attested",
            "no_selective_retry_or_deletion_attested",
        ):
            if document.get(field) is not True:
                _fail("custody_attestation_rejected", f"{field} was not affirmatively attested")
        required_case_roles = document.get("required_case_roles")
        covered_case_roles = document.get("covered_case_roles")
        if tuple(required_case_roles or ()) != CASE_ROLES:
            _fail("holdout_role_gap", "custody required_case_roles is not the frozen roster")
        if (
            not isinstance(covered_case_roles, list)
            or len(covered_case_roles) != len(CASE_ROLES)
            or set(covered_case_roles) != set(required_case_roles)
        ):
            _fail("holdout_role_gap", "custodian coverage does not exactly close the roster")
        if document.get("disclosed_prior_manifest_binding") != PREDECESSOR_PRIOR_MANIFEST_BINDING:
            _fail("prior_manifest_mismatch", "custody did not bind the disclosed prior-case manifest")
        holdout_binding = document.get("holdout_manifest_binding")
        if not isinstance(holdout_binding, Mapping) or any(
            not holdout_binding.get(field)
            for field in ("path", "size_bytes", "sha256", "canonical_sha256")
        ):
            _fail("holdout_manifest_unbound", "custody did not bind the external holdout manifest")
        external_artifact = document.get("external_attestation_artifact")
        if external_artifact.get("status") != "EXTERNAL_ATTESTATION_ARTIFACT_BOUND" or any(
            not external_artifact.get(field)
            for field in (
                "path",
                "size_bytes",
                "sha256",
                "canonical_sha256",
                "signed_utc",
                "signature",
            )
        ):
            _fail("unsigned_custody_attestation", "external custody artifact is not bound")
        if execution_started_utc is not None and _parse_utc(
            document.get("attested_utc"), "attested_utc"
        ) >= _parse_utc(execution_started_utc, "execution_started_utc"):
            _fail("late_attestation", "custody was attested after execution began")

    elif schema_role == "independence_attestation_schema":
        required = (
            "reviewer_identity",
            "reviewer_organization",
            "signing_key_id",
            "relationship_to_implementation_team",
            "conflicts_of_interest_statement",
            "attested_utc",
            "signature",
        )
        if document.get("status") != "EXTERNAL_INDEPENDENCE_ACCEPTED" or any(
            not document.get(field) for field in required
        ):
            _fail("unsigned_independence_attestation", "independence attestation is incomplete")
        for field in (
            "oracle_independence_accepted",
            "eih_independence_accepted",
            "budget_review_independence_accepted",
        ):
            if document.get(field) is not True:
                _fail("independence_not_accepted", f"{field} was not accepted")
        reviewed_sources = document.get("reviewed_source_bindings")
        if not isinstance(reviewed_sources, list) or not reviewed_sources:
            _fail(
                "missing_reviewed_source_binding",
                "accepted independence requires at least one reviewed source binding",
            )
        external_artifact = document.get("external_attestation_artifact")
        if external_artifact.get("status") != "EXTERNAL_ATTESTATION_ARTIFACT_BOUND" or any(
            not external_artifact.get(field)
            for field in (
                "path",
                "size_bytes",
                "sha256",
                "canonical_sha256",
                "signed_utc",
                "signature",
            )
        ):
            _fail("unsigned_independence_attestation", "external independence artifact is not bound")
        if execution_started_utc is not None and _parse_utc(
            document.get("attested_utc"), "attested_utc"
        ) >= _parse_utc(execution_started_utc, "execution_started_utc"):
            _fail("late_attestation", "independence was attested after execution began")

    elif schema_role == "sealed_expectation_commitment_schema":
        required = (
            "expectation_package_path",
            "expectation_package_size_bytes",
            "expectation_package_sha256",
            "expectation_package_canonical_sha256",
            "commitment_sha256",
            "custodian_signing_key_id",
            "committed_utc",
            "signature",
        )
        if document.get("status") != "EXPECTATIONS_COMMITTED_SEALED" or any(
            not document.get(field) for field in required
        ):
            _fail("unsealed_expectations", "expectation commitment is incomplete or unsigned")
        if set(document.get("required_case_roles", ())) != set(CASE_ROLES):
            _fail("holdout_role_gap", "expectation commitment omits required roles")
        if set(document.get("required_observable_ids", ())) != set(OBSERVABLES):
            _fail("observable_roster_gap", "expectation commitment omits observables")
        if execution_started_utc is not None and _parse_utc(
            document.get("committed_utc"), "committed_utc"
        ) >= _parse_utc(execution_started_utc, "execution_started_utc"):
            _fail("late_expectation_commitment", "expectations were committed after execution began")

    elif schema_role == "unblinding_record_schema":
        if document.get("status") not in {"UNBLINDED_FIRST_VALID", "ADJUDICATED_RETAINED"}:
            _fail("unblinding_incomplete", "unblinding record is not first-valid")
        if document.get("outcomes_generated") is not True:
            _fail("unblinding_before_output_commit", "unblinding record has no outcomes")
        if document.get("output_manifest_status") != "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED":
            _fail("unblinding_before_output_commit", "output manifest was not committed while sealed")
        if document.get("valid_unblinding_ordinal") != 1 or document.get("second_unblinding_attempted"):
            _fail("repeated_unblinding", "unblinding is not the sole first-valid attempt")
        if document.get("commitment_verified") is not True:
            _fail("commitment_reveal_mismatch", "unblinding commitment was not verified")
        required = (
            "output_manifest_path",
            "output_manifest_size_bytes",
            "output_manifest_sha256",
            "output_manifest_canonical_sha256",
            "output_committed_utc",
            "unblind_request_utc",
            "custodian_authorization_signature",
            "sealed_expectation_commitment_sha256",
            "revealed_nonce_hex",
            "revealed_expectation_package_path",
            "revealed_expectation_package_sha256",
            "revealed_expectation_package_canonical_sha256",
        )
        if any(not document.get(field) for field in required):
            _fail("unblinding_incomplete", "unblinding record lacks required external fields")
        output_committed_utc = _parse_utc(
            document["output_committed_utc"], "output_committed_utc"
        )
        unblind_request_utc = _parse_utc(
            document["unblind_request_utc"], "unblind_request_utc"
        )
        if unblind_request_utc <= output_committed_utc:
            _fail("unblinding_before_output_commit", "unblinding request did not follow output commit")

        _verify_binding(
            root,
            {
                "path": document["output_manifest_path"],
                "size_bytes": document["output_manifest_size_bytes"],
                "sha256": document["output_manifest_sha256"],
                "canonical_sha256": document["output_manifest_canonical_sha256"],
            },
            "committed output manifest",
            canonical=True,
        )

        revealed_path = _safe_repository_file(
            root,
            document["revealed_expectation_package_path"],
            "revealed expectation package",
        )
        if sha256_file(revealed_path) != document["revealed_expectation_package_sha256"]:
            _fail(
                "raw_digest_mismatch",
                "revealed expectation package raw SHA-256 differs from its binding",
            )
        try:
            revealed_bytes = revealed_path.read_bytes()
        except OSError as exc:
            _fail(
                "artifact_read_failed",
                f"cannot read revealed expectation package: {exc}",
            )
        revealed_expectations = verify_sealed_reveal(
            document["sealed_expectation_commitment_sha256"],
            document["revealed_nonce_hex"],
            revealed_bytes,
        )
        if sha256_data(revealed_expectations) != document[
            "revealed_expectation_package_canonical_sha256"
        ]:
            _fail(
                "canonical_digest_mismatch",
                "revealed expectation package canonical SHA-256 differs from its binding",
            )

        if document["status"] == "UNBLINDED_FIRST_VALID":
            if (
                document.get("adjudication_status") != "AWAITING_ADJUDICATION"
                or document.get("adjudication_verdict") is not None
                or document.get("adjudicated_utc") is not None
            ):
                _fail(
                    "adjudication_state_mismatch",
                    "first-valid unblinding must remain awaiting adjudication",
                )
        else:
            if document.get("adjudication_status") != "ADJUDICATED_RETAINED":
                _fail(
                    "adjudication_state_mismatch",
                    "retained adjudication status does not close the lifecycle",
                )
            if document.get("adjudication_verdict") not in ADJUDICATION_VERDICTS:
                _fail(
                    "invalid_adjudication_verdict",
                    "retained adjudication requires an allowed nonnull verdict",
                )
            adjudicated_utc = _parse_utc(document.get("adjudicated_utc"), "adjudicated_utc")
            if adjudicated_utc <= unblind_request_utc:
                _fail(
                    "adjudication_chronology_invalid",
                    "adjudication must follow the first-valid unblinding request",
                )

    elif schema_role == "error_budget_schema":
        _validate_error_budget_semantics(document)
        if document.get("status") != "INDEPENDENTLY_REVIEWED_AND_FROZEN":
            _fail(
                "budget_not_ready",
                "only an independently reviewed and frozen budget satisfies the prerequisite",
            )

    requires_signature_verification = schema_role != "error_budget_schema" or document.get(
        "status"
    ) != "UNRESOLVED"
    if requires_signature_verification:
        if signature_verifier is None:
            _fail(
                "signature_verification_required",
                "external artifact needs an independent signature verifier",
            )
        try:
            signature_valid = signature_verifier(document)
        except Exception as exc:
            _fail("signature_verification_failed", f"signature verifier failed: {exc}")
        if signature_valid is not True:
            _fail("invalid_external_signature", "external artifact signature did not verify")


@dataclass(frozen=True)
class Q8Q9PrerequisiteInspection:
    schema: str
    prerequisite_package_id: str
    predecessor_qualification_id: str
    predecessor_package_sha256: str
    predecessor_commit: str
    registration_file_sha256: str
    registration_canonical_sha256: str
    package_sha256: str
    status: str
    scientific_evidence_artifact: bool
    outcomes_generated: bool
    execution_authorized: bool
    ready_for_holdout_execution: bool
    unblinding_occurred: bool
    blocked_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_q8q9_prerequisite_package(
    registration_path: str | Path = REGISTRATION_RELATIVE,
    *,
    project_root: str | Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Q8Q9PrerequisiteInspection,
]:
    """Inspect the frozen design package without executing or authorizing work."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        _fail("invalid_project_root", "project_root must be an existing directory")
    registration_source = _safe_repository_file(root, registration_path, "registration")
    registration = _load_json(registration_source, "registration")
    registration_schema = _load_pinned_schema(root, "prerequisite_registration_schema")
    _validate_schema(registration, registration_schema, "registration")

    predecessor = registration["predecessor"]
    expected_predecessor = {
        "qualification_id": QUALIFICATION_ID,
        "package_sha256": PREDECESSOR_PACKAGE_SHA256,
        "commit": PREDECESSOR_COMMIT,
        **deepcopy(PREDECESSOR_BINDINGS),
    }
    if predecessor != expected_predecessor:
        _fail("predecessor_mismatch", "predecessor identity or bindings are not exact")
    for role, binding in PREDECESSOR_BINDINGS.items():
        _verify_binding(root, binding, f"predecessor {role}", canonical=True)

    expected_registration = {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-prerequisite-registration-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-prerequisite-registration/v1",
        "prerequisite_package_id": PACKAGE_ID,
        "status": "DESIGN_ONLY_BLOCKED",
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_authorized": False,
        "ready_for_holdout_execution": False,
        "unblinding_occurred": False,
        "nonclaim": REGISTRATION_NONCLAIM,
    }
    for field, expected in expected_registration.items():
        if registration.get(field) != expected:
            _fail("premature_authorization", f"registration.{field} must remain {expected!r}")
    if tuple(registration["lifecycle"]["qualification_states"]) != QUALIFICATION_STATES:
        _fail("lifecycle_mismatch", "qualification lifecycle is not exact")
    if tuple(registration["lifecycle"]["custody_states"]) != CUSTODY_STATES:
        _fail("lifecycle_mismatch", "custody lifecycle is not exact")
    if tuple(registration["lifecycle"]["error_budget_states"]) != ERROR_BUDGET_STATES:
        _fail("lifecycle_mismatch", "error-budget lifecycle is not exact")
    if tuple(registration["lifecycle"]["phase_requirements"]) != PHASE_REQUIREMENTS:
        _fail("phase_mismatch", "phase requirements are not exact")
    if tuple(registration["blocked_reasons"]) != BLOCKED_REASONS:
        _fail("blocked_reason_mismatch", "blocked reasons are not exact")
    if registration["claim_control"] != {
        "claim_emitted": False,
        "claim_text": None,
        "maximum_future_claim_effect": "ELIGIBLE_FOR_EXTERNAL_REVIEW_NONAUTHORIZING_ONLY",
    }:
        _fail("early_claim", "design package emitted or enlarged a claim")

    schema_bindings = _index_unique(registration["schemas"], "role", "schema bindings")
    _require_exact_keys(schema_bindings, tuple(SCHEMA_SPECS), "schema bindings")
    loaded_schemas: dict[str, dict[str, Any]] = {}
    for role, (schema_id, expected_path) in SCHEMA_SPECS.items():
        binding = schema_bindings[role]
        expected_raw_sha256, expected_size = SCHEMA_RAW_IDENTITY[role]
        expected_binding = {
            "role": role,
            "schema": schema_id,
            "path": expected_path,
            "sha256": expected_raw_sha256,
            "size_bytes": expected_size,
            "canonical_sha256": SCHEMA_CANONICAL_SHA256[role],
        }
        if binding != expected_binding:
            _fail("schema_binding_mismatch", f"schema binding {role!r} is wrong")
        loaded_schemas[role] = _load_pinned_schema(root, role)

    artifact_bindings = _index_unique(registration["artifacts"], "role", "artifact bindings")
    _require_exact_keys(artifact_bindings, tuple(ARTIFACT_SPECS), "artifact bindings")
    loaded_artifacts: dict[str, dict[str, Any]] = {}
    for role, (schema_id, expected_path) in ARTIFACT_SPECS.items():
        binding = artifact_bindings[role]
        if binding.get("schema") != schema_id or binding.get("path") != expected_path:
            _fail("artifact_binding_mismatch", f"artifact binding {role!r} is wrong")
        _, loaded = _verify_binding(root, binding, f"artifact {role}", canonical=True)
        assert loaded is not None
        loaded_artifacts[role] = loaded

    slot_wrapper = {
        "$ref": "#/$defs/externalArtifactSlotsDocument",
        "$defs": registration_schema["$defs"],
    }
    slots = loaded_artifacts["external_artifact_slots"]
    _validate_schema(slots, slot_wrapper, "external artifact slots")
    expected_slots_identity = {
        "$schema": (
            "../../schemas/jx-v5-solar-1pn-q8q9-prerequisite-registration-v1.schema.json"
            "#/$defs/externalArtifactSlotsDocument"
        ),
        "schema": "jx-v5-solar-1pn-q8q9-external-artifact-slots/v1",
        "status": "DESIGN_ONLY_BLOCKED",
        "nonclaim": EXTERNAL_SLOTS_NONCLAIM,
    }
    for field, expected in expected_slots_identity.items():
        if slots.get(field) != expected:
            _fail("external_slot_mismatch", f"external slots {field} differs")
    if slots.get("qualification_id") != QUALIFICATION_ID or slots.get(
        "predecessor_package_sha256"
    ) != PREDECESSOR_PACKAGE_SHA256:
        _fail("predecessor_mismatch", "external slots predecessor binding is wrong")
    slot_index = _index_unique(slots["slots"], "role", "external artifact slots")
    _require_exact_keys(slot_index, tuple(EXTERNAL_SLOT_SPECS), "external artifact slots")
    for role, (schema_id, status, phase) in EXTERNAL_SLOT_SPECS.items():
        slot = slot_index[role]
        if slot.get("target_schema") != schema_id or slot.get("status") != status:
            _fail("external_slot_mismatch", f"external slot {role!r} has the wrong status")
        if slot.get("phase_required") != phase:
            _fail("phase_mismatch", f"external slot {role!r} has the wrong phase")
        for field in (
            "path",
            "size_bytes",
            "sha256",
            "canonical_sha256",
            "signer_identity",
            "signing_key_id",
            "signed_utc",
            "signature",
        ):
            if slot.get(field) is not None:
                _fail("fabricated_external_artifact", f"external slot {role!r}.{field} must be null")

    budget = loaded_artifacts["error_budget_template"]
    _validate_schema(budget, loaded_schemas["error_budget_schema"], "error-budget template")
    expected_budget_identity = {
        "$schema": "../../schemas/jx-v5-solar-1pn-q8q9-error-budget-v1.schema.json",
        "schema": "jx-v5-solar-1pn-q8q9-error-budget/v1",
        "phase_required": "BEFORE_EXECUTION",
        "nonclaim": ERROR_BUDGET_TEMPLATE_NONCLAIM,
    }
    for field, expected in expected_budget_identity.items():
        if budget.get(field) != expected:
            _fail("budget_constant_mismatch", f"error-budget template {field} differs")
    _validate_error_budget_semantics(budget)

    package_directory = registration_source.parent
    for forbidden in FORBIDDEN_INSTANCE_NAMES:
        if (package_directory / forbidden).exists() or (package_directory / forbidden).is_symlink():
            _fail("fabricated_external_instance", f"external instance {forbidden!r} must not exist")

    for binding_name in ("verifier", "readme"):
        _verify_binding(root, registration[binding_name], binding_name, canonical=False)

    locked = [
        {
            "path": binding["path"],
            "sha256": binding["sha256"],
            "size_bytes": binding["size_bytes"],
            **(
                {"canonical_sha256": binding["canonical_sha256"]}
                if "canonical_sha256" in binding
                else {}
            ),
        }
        for binding in sorted(
            [*registration["artifacts"], *registration["schemas"], registration["verifier"], registration["readme"]],
            key=lambda item: item["path"],
        )
    ]
    inspection = Q8Q9PrerequisiteInspection(
        schema="jx-v5-solar-1pn-q8q9-prerequisite-inspection/v1",
        prerequisite_package_id=PACKAGE_ID,
        predecessor_qualification_id=QUALIFICATION_ID,
        predecessor_package_sha256=PREDECESSOR_PACKAGE_SHA256,
        predecessor_commit=PREDECESSOR_COMMIT,
        registration_file_sha256=sha256_file(registration_source),
        registration_canonical_sha256=sha256_data(registration),
        package_sha256=sha256_data(
            {
                "schema": "jx-v5-solar-1pn-q8q9-prerequisite-package-digest/v1",
                "predecessor_package_sha256": PREDECESSOR_PACKAGE_SHA256,
                "registration_canonical_sha256": sha256_data(registration),
                "locked_files": locked,
            }
        ),
        status="DESIGN_ONLY_BLOCKED",
        scientific_evidence_artifact=False,
        outcomes_generated=False,
        execution_authorized=False,
        ready_for_holdout_execution=False,
        unblinding_occurred=False,
        blocked_reasons=BLOCKED_REASONS,
    )
    return slots, budget, registration, inspection


__all__ = [
    "ADJUDICATION_VERDICTS",
    "ARTIFACT_SPECS",
    "BLOCKED_REASONS",
    "CASE_ROLES",
    "COMPONENTS",
    "CUSTODY_STATES",
    "DOMAIN_SEPARATOR",
    "ERROR_BUDGET_STATES",
    "ERROR_BUDGET_DOCUMENT_STATES",
    "OBSERVABLES",
    "PACKAGE_ID",
    "PHASE_REQUIREMENTS",
    "PREDECESSOR_COMMIT",
    "PREDECESSOR_PRIOR_MANIFEST_BINDING",
    "PREDECESSOR_PACKAGE_SHA256",
    "QUALIFICATION_ID",
    "QUALIFICATION_STATES",
    "Q8Q9PrerequisiteError",
    "Q8Q9PrerequisiteInspection",
    "REGISTRATION_RELATIVE",
    "SCHEMA_CANONICAL_SHA256",
    "SCHEMA_SPECS",
    "SCHEMA_DIALECT_URI",
    "SUPPORTED_SCHEMA_KEYWORDS",
    "SUPPORTED_SCHEMA_TYPES",
    "canonical_json",
    "compute_sealed_commitment",
    "enforce_budget_immutability",
    "enforce_output_immutability",
    "inspect_q8q9_prerequisite_package",
    "sha256_data",
    "sha256_file",
    "transition_state",
    "validate_external_artifact",
    "validate_holdout_roster",
    "validate_realized_error_record",
    "validate_unblinding_attempt",
    "verify_sealed_reveal",
]
