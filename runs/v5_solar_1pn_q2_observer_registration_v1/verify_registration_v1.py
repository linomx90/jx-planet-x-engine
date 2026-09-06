"""Read-only verifier for the blocked Q2 observer registration boundary.

This module validates metadata and external attestations only.  It imports no
JX, observer, oracle, dynamics, trajectory, registry, or holdout code and has
no command-line entry point.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence


PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_q2_observer_registration_v1")
REGISTRATION_RELATIVE = PACKAGE_RELATIVE / "registration_v1.json"
TEMPLATE_RELATIVE = PACKAGE_RELATIVE / "external_registration_template_v1.json"
README_RELATIVE = PACKAGE_RELATIVE / "README.md"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_registration_v1.py"
SCHEMA_RELATIVE = Path(
    "schemas/jx-v5-solar-1pn-q2-observer-registration-boundary-v1.schema.json"
)

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "jx-v5-solar-1pn-q2-observer-registration-boundary/v1"
EXTERNAL_SCHEMA_ID = "jx-v5-solar-1pn-q2-observer-external-registration/v1"
PACKAGE_ID = "jx.v5.solar_1pn.q2_observer_registration_boundary.v1"
QUALIFICATION_ID = (
    "jx.v5.solar_1pn.qualification."
    "78d05102bac2588b7677ec5ef19c653c0102923a1ab499852194f423c8911781"
)
QUALIFICATION_PACKAGE_SHA256 = (
    "80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237"
)
QUALIFICATION_COMMIT = "975b1b7002e358a980457acb18c9beaa90aa1c9f"
EXECUTION_PREREQUISITES_COMMIT = "dd7b22f64c04e0836c624f13b6a4a087caeeeb46"
OBSERVER_COMMIT = "d6344c49e7c6a734a337e84bd209ca5c11c23916"
OBSERVER_PARENT_COMMIT = EXECUTION_PREREQUISITES_COMMIT
OBSERVER_TREE = "4eb04188a0ec6e073e51f126b02eb8a394872aa2"
OBSERVER_ID = "jx.v5.solar_1pn.q2.perihelion_observer_candidate.v1"
OBSERVER_SOURCE_TUPLE_SHA256 = (
    "d9d02d25689bb4561594aae77d7673433ce3c8e008b785f314c33bed143f3e27"
)

# Patched only after the schema is frozen; the verifier refuses a rehashed,
# weakened schema.
SCHEMA_RAW_SHA256 = "696a51313f443bb9a139cb191b86e2676c7568a4e6244b157d94ba5590ab62ef"
SCHEMA_SIZE_BYTES = "22828"
SCHEMA_CANONICAL_SHA256 = "8f7fbdf1a4eee759885bb3ce95647a2632a6213ff39160c176b3e74784c70fb4"

PACKAGE_NONCLAIM = (
    "This additive package binds a candidate and an empty external handoff "
    "template only. It is not scientific evidence and grants no execution, "
    "registry, qualification, unblinding, or claim authority."
)
TEMPLATE_NONCLAIM = (
    "This null template binds only the frozen Q2 candidate identity. It supplies "
    "no runtime, dependency closure, configuration, independence attestation, "
    "custody artifact, expectation, budget, outcome, or authority."
)
EXTERNAL_NONCLAIM = (
    "This signed external registration binds source, runtime, dependencies, and "
    "observer configurations for review only. It remains nonauthorizing and is "
    "not a qualification result, custody completion, or execution permission."
)

IMPLEMENTED_SCOPE = (
    "INITIAL_EVENT_OUTWARD_THEN_INWARD_ARMING_STATE_MACHINE",
    "CUBIC_HERMITE_DENSE_STATE",
    "BERNSTEIN_DEGREE5_ROOT_ISOLATION",
    "NEGATIVE_TO_POSITIVE_ROOT_REFINEMENT_AND_TERMINATION",
    "INITIAL_R_ANGULAR_MOMENTUM_ORIENTATION_OFF_PLANE_CHECK",
    "DECIMAL_ATAN2_AND_PI",
    "EXACT_PI_TIE_REJECTING_UNWRAP",
    "ORBIT_INDICES_1_AND_2",
    "DECIMAL_OLS_SLOPE",
    "SIGNED_1PN_MINUS_NEWTONIAN_PAIRING",
)
UNRESOLVED_SCOPE = (
    "STEP_EXTRAPOLATION",
    "EVENT_TOLERANCE_EXTRAPOLATION",
    "INVERSE_C_SQUARED_EXTRAPOLATION",
    "ANALYTIC_6PI_COMPARISON",
    "QUALIFICATION_ERROR_BUDGET",
    "GATE_ADJUDICATION",
)
BLOCKED_REASONS = (
    "AWAITING_EXTERNAL_RUNTIME",
    "AWAITING_EXTERNAL_DEPENDENCY_MANIFEST",
    "AWAITING_EXTERNAL_CONFIGURATION",
    "AWAITING_EXTERNAL_Q2_INDEPENDENCE_ATTESTATION",
    "SIGNATURE_TRUST_PROFILE_AND_KEYRING_UNREGISTERED",
    "EXPECTATION_CONTENT_SCHEMA_UNREGISTERED",
    "WRAPPER_AUTHENTICATION_AND_SERIALIZATION_UNREGISTERED",
    "Q2_EXTRAPOLATION_ANALYTIC_6PI_AND_ADJUDICATION_UNRESOLVED",
    "QUALIFICATION_ERROR_BUDGET_UNRESOLVED",
    "Q3_ORACLE_UNRESOLVED",
    "Q4_EIH_UNRESOLVED",
    "Q8Q9_CUSTODY_AND_SEALED_EXPECTATIONS_UNRESOLVED",
    "FROZEN_QUALIFICATION_OMITS_CONCRETE_Q2_OBSERVER_IDENTITY",
    "FROZEN_QUALIFICATION_Q8_PHASE_CONTRADICTION_REQUIRES_CONTENT_DERIVED_SUCCESSOR",
    "HOLDOUT_EXECUTION_PROHIBITED",
    "EXECUTION_NOT_AUTHORIZED",
    "OUTCOMES_NOT_GENERATED",
)

DIRECT_STDLIB_IMPORTS = (
    "__future__",
    "dataclasses",
    "decimal",
    "enum",
    "hashlib",
    "json",
    "re",
    "typing",
)
REQUIRED_ROOT_TOLERANCES = ("1E-20", "1E-30")

FIXED_CONFIGURATION = {
    "observer_id": OBSERVER_ID,
    "context_id": "decimal.context.precision_90",
    "precision": 90,
    "rounding": "ROUND_HALF_EVEN",
    "emin": -999999,
    "emax": 999999,
    "capitals": 1,
    "clamp": 0,
    "maximum_root_iterations": 256,
    "certificate_precision": 4096,
    "expected_event_count": 2,
    "atan_guard_digits": 12,
    "maximum_atan_series_iterations": 10000,
    "epoch_unit_id": "unit.day",
    "root_tolerance_unit_id": "unit.day",
    "angle_unit_id": "unit.radian",
    "dense_state_method_id": (
        "decimal.cubic-hermite.position-with-derivative-velocity.v1"
    ),
    "root_method_id": "decimal.bisection.rdotv.negative-to-positive.v1",
    "root_isolation_method_id": "decimal.bernstein-degree5.sign-variation.v1",
    "angle_method_id": "decimal.oriented-atan2.dlmf-reduced-series.v1",
    "unwrap_method_id": "nearest-principal-branch.exact-pi-tie-rejected.v1",
    "regression_method_id": "decimal.ols.angle-on-integer-orbit-index.v1",
}

CANDIDATE_FILES = (
    {
        "role": "observer_readme",
        "path": "observer/v5_solar_1pn/README.md",
        "sha256": "3947c7aa4739414a7149f224158090dbd896c16a83a930755825dd18954275d4",
        "size_bytes": "4465",
        "git_blob": "aace364abb0df1c87e6dcf8d05904450ccf4954a",
    },
    {
        "role": "observer_package",
        "path": "observer/v5_solar_1pn/__init__.py",
        "sha256": "7c31721343f7dad6a8f8bc4a3d4fd6ab3339529627d38198edc3d7515adde6cb",
        "size_bytes": "181",
        "git_blob": "2766c51b52e92ebebbb03afdf4998ec0fde1ae4b",
    },
    {
        "role": "observer_source",
        "path": "observer/v5_solar_1pn/candidate.py",
        "sha256": "720e69b26be5a52d6a8279e036486242d14617631e42c9339f29832de44e3640",
        "size_bytes": "66910",
        "git_blob": "39076d74d035b0321e119efbf221d793690e13e3",
    },
    {
        "role": "observer_disclosed_tests",
        "path": "tests/test_v5_q2_perihelion_observer.py",
        "sha256": "b827522717cc6792ed11af998908e6c8b8fee329ad399dcb54d44a823da40b1b",
        "size_bytes": "36508",
        "git_blob": "a41158246e4a17b1e56f04662949d1823137db3c",
    },
)

PREDECESSORS = {
    "qualification": {
        "role": "qualification",
        "package_id": QUALIFICATION_ID,
        "commit": QUALIFICATION_COMMIT,
        "identity_kind": "QUALIFICATION_PACKAGE_SHA256",
        "identity_sha256": QUALIFICATION_PACKAGE_SHA256,
        "registration": {
            "path": "runs/v5_solar_1pn_qualification/registration_v1.json",
            "sha256": "7e4b839f0842b6e5d23b6b2c1610eb3a1f3024062e4b696d09e1eb9d6b32f545",
            "size_bytes": "18490",
            "canonical_sha256": "319939d02d5764e527514fd346615b9b175ac120c194cf4e88435482b31a253c",
        },
    },
    "execution_prerequisites": {
        "role": "execution_prerequisites",
        "package_id": "jx.v5.solar_1pn.execution_prerequisites.v1",
        "commit": EXECUTION_PREREQUISITES_COMMIT,
        "identity_kind": "REGISTRATION_CANONICAL_SHA256",
        "identity_sha256": "4aa7a82574cdd3127809d82202ca4e5a9e999f2f4d40b7fe93a01b388d002ea7",
        "registration": {
            "path": "runs/v5_solar_1pn_execution_prerequisites_v1/registration_v1.json",
            "sha256": "e64b267971353b54b88ce7c2e8132f14673a64c7068ce52193eb78c71ce70b11",
            "size_bytes": "5960",
            "canonical_sha256": "4aa7a82574cdd3127809d82202ca4e5a9e999f2f4d40b7fe93a01b388d002ea7",
        },
    },
    "q8q9_prerequisites": {
        "role": "q8q9_prerequisites",
        "package_id": "jx.v5.solar_1pn.q8q9_prerequisites.v1",
        "commit": EXECUTION_PREREQUISITES_COMMIT,
        "identity_kind": "Q8Q9_PACKAGE_SHA256",
        "identity_sha256": "1d9008ae6a3c3a79edd06c3b01f6ab2a662246633188339c0d7f608f5712080b",
        "registration": {
            "path": "runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1/registration_v1.json",
            "sha256": "b046b2c049c8a2c800382aed171306dd7eb6e8a813fd283ad55706b801ce109f",
            "size_bytes": "7446",
            "canonical_sha256": "b6e86925ea6a0ecc28423b36a795ae743400cc183c46b9fdaf570431788514f3",
        },
    },
}

Q8Q9_SCHEMA_BINDINGS = (
    {
        "role": "q8q9_custody_schema",
        "schema": "jx-v5-solar-1pn-q8q9-custody-attestation/v1",
        "path": "schemas/jx-v5-solar-1pn-q8q9-custody-attestation-v1.schema.json",
        "sha256": "0f4026f7a7190fa3b7cb450d5e3fbc1e12ce9390d23813e2b8023b180da5dc91",
        "size_bytes": "7132",
        "canonical_sha256": "a1d2df0edca31c68f5d721a5d15214fe003ddbcaf609ee18a0fb3c5b30ef6b39",
    },
    {
        "role": "q8q9_sealed_expectation_schema",
        "schema": "jx-v5-solar-1pn-q8q9-sealed-expectation-commitment/v1",
        "path": "schemas/jx-v5-solar-1pn-q8q9-sealed-expectation-commitment-v1.schema.json",
        "sha256": "f7fa4ca3950f5a53d2a6ff2cb74b52b6e724aba5a456a9eebfe70f1f0a158e18",
        "size_bytes": "5356",
        "canonical_sha256": "772d1ece6d3673e8052215bb1cfd3947c2e10b004be123ed63decfa987077857",
    },
    {
        "role": "q8q9_qualification_error_budget_schema",
        "schema": "jx-v5-solar-1pn-q8q9-error-budget/v1",
        "path": "schemas/jx-v5-solar-1pn-q8q9-error-budget-v1.schema.json",
        "sha256": "b186a99016eb8d65b245439028d95f6859d124358ff5954e72d82d329429ef91",
        "size_bytes": "9266",
        "canonical_sha256": "c306badcf5e0889e95b9ec274df26e3b8e02c4bd9451852de98796e9a9c46e87",
    },
)

DOWNSTREAM_SLOTS = (
    {
        "role": "custody_attestation",
        "target_schema": "jx-v5-solar-1pn-q8q9-custody-attestation/v1",
        "status": "AWAITING_EXTERNAL_CUSTODIAN_ATTESTATION",
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "canonical_sha256": None,
    },
    {
        "role": "sealed_expectation_commitment",
        "target_schema": "jx-v5-solar-1pn-q8q9-sealed-expectation-commitment/v1",
        "status": "AWAITING_EXTERNAL_SEALED_EXPECTATION_COMMITMENT",
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "canonical_sha256": None,
    },
    {
        "role": "qualification_error_budget",
        "target_schema": "jx-v5-solar-1pn-q8q9-error-budget/v1",
        "status": "AWAITING_EXTERNAL_QUALIFICATION_ERROR_BUDGET",
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "canonical_sha256": None,
    },
)

UNRESOLVED_SLOTS = {
    "signature_trust_profile_and_keyring": {
        "status": "AWAITING_EXTERNAL_SIGNATURE_TRUST_PROFILE_AND_KEYRING",
        "value": None,
    },
    "expectation_content_schema": {
        "status": "AWAITING_EXTERNAL_EXPECTATION_CONTENT_SCHEMA",
        "value": None,
    },
    "wrapper_authentication_and_serialization": {
        "status": "AWAITING_EXTERNAL_WRAPPER_AUTHENTICATION_AND_SERIALIZATION",
        "value": None,
    },
    "q2_extrapolation_analytic_and_adjudication": {
        "status": "AWAITING_EXTERNAL_Q2_EXTRAPOLATION_ANALYTIC_AND_ADJUDICATION",
        "value": None,
    },
    "qualification_error_budget": {
        "status": "AWAITING_EXTERNAL_QUALIFICATION_ERROR_BUDGET",
        "value": None,
    },
    "q3_oracle": {"status": "AWAITING_EXTERNAL_Q3_ORACLE", "value": None},
    "q4_eih": {"status": "AWAITING_EXTERNAL_Q4_EIH", "value": None},
    "qualification_successor": {
        "status": "AWAITING_EXTERNAL_CONTENT_DERIVED_QUALIFICATION_SUCCESSOR",
        "value": None,
    },
    "holdout_execution": {
        "status": "AWAITING_EXTERNAL_HOLDOUT_EXECUTION_AUTHORIZATION",
        "value": None,
    },
}

CLAIM_CONTROL = {
    "claim_emitted": False,
    "claim_text": None,
    "maximum_effect": "EXTERNAL_REVIEW_HANDOFF_NONAUTHORIZING_ONLY",
}

SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "title",
        "type",
        "additionalProperties",
        "required",
        "properties",
        "const",
        "enum",
        "pattern",
        "minLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "items",
        "if",
        "then",
        "else",
    }
)
SUPPORTED_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "integer", "boolean", "null"}
)


class Q2ObserverRegistrationError(ValueError):
    """The registration boundary is incomplete, drifted, or unsafe."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise Q2ObserverRegistrationError(code, message)


def _reject_float(value: str) -> None:
    _fail("binary_float_json", f"JSON binary-float token is forbidden: {value}")


def _reject_constant(value: str) -> None:
    _fail("nonfinite_json", f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_binary_float_tree(value: Any, context: str = "document") -> None:
    if isinstance(value, float):
        _fail("binary_float_value", f"{context} contains a binary float")
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            _fail("noncanonical_json", f"{context} has a non-string key")
        for key, child in value.items():
            _reject_binary_float_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_binary_float_tree(child, f"{context}[{index}]")


def canonical_json(value: Any) -> bytes:
    _reject_binary_float_tree(value)
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        _fail("noncanonical_json", f"cannot canonicalize value: {exc}")


def sha256_data(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail("artifact_read_failed", f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _load_json(path: Path, context: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except Q2ObserverRegistrationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("invalid_json", f"cannot load {context}: {exc}")
    if not isinstance(value, dict):
        _fail("invalid_json", f"{context} must be a JSON object")
    return value


def _safe_file(root: Path, relative: Any, context: str) -> Path:
    if not isinstance(relative, str) or not relative:
        _fail("invalid_repository_path", f"{context} path is invalid")
    raw_parts = relative.split("/")
    if relative.startswith("/") or any(part in {"", ".", ".."} for part in raw_parts):
        _fail("invalid_repository_path", f"{context} path is unsafe")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        _fail("invalid_repository_path", f"{context} path is unsafe")
    candidate = root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            _fail("symlink_forbidden", f"{context} traverses a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail("path_escape", f"cannot resolve {context}: {exc}")
    if not resolved.is_file():
        _fail("artifact_not_regular_file", f"{context} is not a regular file")
    return resolved


def _schema_equal(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _resolve_ref(root_schema: Mapping[str, Any], reference: Any) -> Mapping[str, Any]:
    if not isinstance(reference, str) or not reference.startswith("#/"):
        _fail("unsupported_schema_reference", "only local JSON-pointer refs are supported")
    current: Any = root_schema
    for encoded in reference[2:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or token not in current:
            _fail("invalid_schema_reference", f"schema ref {reference!r} does not resolve")
        current = current[token]
    if not isinstance(current, Mapping):
        _fail("invalid_schema_reference", f"schema ref {reference!r} is not an object")
    return current


def _audit_schema(schema: Any, root_schema: Mapping[str, Any], context: str) -> None:
    if not isinstance(schema, Mapping):
        _fail("invalid_schema_definition", f"{context} is not an object")
    unknown = set(schema) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        _fail("unsupported_schema_keyword", f"{context} has {sorted(unknown)}")
    if "$schema" in schema and schema["$schema"] != SCHEMA_DIALECT:
        _fail("unsupported_schema_dialect", f"{context} has the wrong dialect")
    if "$ref" in schema:
        _resolve_ref(root_schema, schema["$ref"])
        if set(schema) - {"$ref", "$defs"}:
            _fail("unsupported_schema_shape", f"{context} has unsupported ref siblings")
    declared = schema.get("type")
    if declared is not None and declared not in SUPPORTED_SCHEMA_TYPES:
        _fail("unsupported_schema_type", f"{context} has unsupported type")
    if declared == "object":
        if schema.get("additionalProperties") is not False:
            _fail("schema_not_closed", f"{context} must be closed")
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            _fail("schema_not_closed", f"{context} needs properties and required")
        if set(properties) != set(required) or len(required) != len(set(required)):
            _fail("schema_not_closed", f"{context} must require every property exactly once")
        for key, child in properties.items():
            _audit_schema(child, root_schema, f"{context}.properties.{key}")
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, Mapping):
        _fail("invalid_schema_definition", f"{context}.$defs is not an object")
    for key, child in definitions.items():
        _audit_schema(child, root_schema, f"{context}.$defs.{key}")
    if "items" in schema:
        _audit_schema(schema["items"], root_schema, f"{context}.items")
    for key in ("minItems", "maxItems", "minLength"):
        if key in schema and (not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0):
            _fail("invalid_schema_definition", f"{context}.{key} is invalid")
    if schema.get("uniqueItems", True) is not True:
        _fail("invalid_schema_definition", f"{context}.uniqueItems may only be true")
    if "pattern" in schema:
        if declared != "string" or not isinstance(schema["pattern"], str):
            _fail("unsupported_schema_shape", f"{context}.pattern requires a string schema")
        try:
            re.compile(schema["pattern"])
        except re.error as exc:
            _fail("invalid_schema_pattern", f"{context}.pattern is invalid: {exc}")
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not values:
            _fail("invalid_schema_definition", f"{context}.enum is invalid")
        if len({canonical_json(item) for item in values}) != len(values):
            _fail("invalid_schema_definition", f"{context}.enum repeats values")
    if "if" in schema:
        _audit_schema(schema["if"], root_schema, f"{context}.if")
        for branch in ("then", "else"):
            if branch in schema:
                _audit_schema(schema[branch], root_schema, f"{context}.{branch}")
    elif "then" in schema or "else" in schema:
        _fail("invalid_schema_definition", f"{context} has a branch without if")


def _type_matches(value: Any, declared: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }[declared]


def _schema_matches(value: Any, schema: Mapping[str, Any], root_schema: Mapping[str, Any]) -> bool:
    try:
        _validate_schema_value(value, schema, root_schema, "conditional", ())
        return True
    except Q2ObserverRegistrationError:
        return False


def _validate_schema_value(
    value: Any,
    schema: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    context: str,
    stack: tuple[str, ...],
) -> None:
    if "$ref" in schema:
        reference = schema["$ref"]
        if reference in stack:
            _fail("cyclic_schema_reference", f"{context} has a cyclic ref")
        _validate_schema_value(
            value, _resolve_ref(root_schema, reference), root_schema, context, (*stack, reference)
        )
        return
    declared = schema.get("type")
    if declared is not None and not _type_matches(value, declared):
        _fail("schema_type", f"{context} must have type {declared}")
    if "const" in schema and not _schema_equal(value, schema["const"]):
        _fail("schema_const", f"{context} differs from a schema constant")
    if "enum" in schema and not any(_schema_equal(value, item) for item in schema["enum"]):
        _fail("schema_enum", f"{context} is outside the allowed enum")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            _fail("schema_min_length", f"{context} is too short")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value) is None:
            _fail("schema_pattern", f"{context} does not match its pattern")
    if isinstance(value, Mapping) and declared == "object":
        properties = schema["properties"]
        if set(value) != set(properties):
            _fail("schema_properties", f"{context} keys differ from the closed schema")
        for key, child in properties.items():
            _validate_schema_value(value[key], child, root_schema, f"{context}.{key}", stack)
    if isinstance(value, list) and declared == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            _fail("schema_min_items", f"{context} has too few items")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            _fail("schema_max_items", f"{context} has too many items")
        if schema.get("uniqueItems") and len({canonical_json(item) for item in value}) != len(value):
            _fail("schema_unique_items", f"{context} repeats items")
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema_value(item, schema["items"], root_schema, f"{context}[{index}]", stack)
    if "if" in schema:
        branch = schema.get("then") if _schema_matches(value, schema["if"], root_schema) else schema.get("else")
        if branch is not None:
            _validate_schema_value(value, branch, root_schema, context, stack)


def _validate_schema(document: Mapping[str, Any], schema: Mapping[str, Any], context: str) -> None:
    _reject_binary_float_tree(document, context)
    _reject_binary_float_tree(schema, f"{context} schema")
    _audit_schema(schema, schema, f"{context} schema")
    _validate_schema_value(document, schema, schema, context, ())


def _load_pinned_schema(root: Path) -> dict[str, Any]:
    path = _safe_file(root, SCHEMA_RELATIVE.as_posix(), "boundary schema")
    if str(path.stat().st_size) != SCHEMA_SIZE_BYTES or sha256_file(path) != SCHEMA_RAW_SHA256:
        _fail("schema_identity_mismatch", "boundary schema raw identity drifted")
    schema = _load_json(path, "boundary schema")
    if sha256_data(schema) != SCHEMA_CANONICAL_SHA256:
        _fail("schema_identity_mismatch", "boundary schema canonical identity drifted")
    if schema.get("$schema") != SCHEMA_DIALECT:
        _fail("schema_identity_mismatch", "boundary schema dialect drifted")
    _audit_schema(schema, schema, "boundary schema")
    return schema


def _verify_binding(
    root: Path, binding: Mapping[str, Any], context: str, *, canonical: bool
) -> dict[str, Any] | None:
    path = _safe_file(root, binding.get("path"), context)
    if str(path.stat().st_size) != binding.get("size_bytes"):
        _fail("size_mismatch", f"{context} size differs")
    if sha256_file(path) != binding.get("sha256"):
        _fail("raw_digest_mismatch", f"{context} raw digest differs")
    if not canonical:
        return None
    document = _load_json(path, context)
    if sha256_data(document) != binding.get("canonical_sha256"):
        _fail("canonical_digest_mismatch", f"{context} canonical digest differs")
    return document


def _source_binding() -> dict[str, Any]:
    return {
        "status": "JX_CANDIDATE_BYTES_BOUND_PENDING_EXTERNAL_REVIEW",
        "observer_id": OBSERVER_ID,
        "review_status": "CANDIDATE_PENDING_INDEPENDENT_REVIEW",
        "commit": OBSERVER_COMMIT,
        "tree": OBSERVER_TREE,
        "source_tuple_sha256": OBSERVER_SOURCE_TUPLE_SHA256,
    }


def _claim_flags(document: Mapping[str, Any], context: str) -> None:
    for field in (
        "scientific_evidence_artifact",
        "outcomes_generated",
        "execution_authorized",
        "registry_authorized",
        "ready",
    ):
        if document.get(field) is not False:
            _fail("premature_authorization", f"{context}.{field} must remain false")
    if document.get("claim_control") != CLAIM_CONTROL:
        _fail("early_claim", f"{context} claim control is not exact")


def _template_semantics(document: Mapping[str, Any]) -> None:
    if document.get("observer_source_binding") != _source_binding():
        _fail("observer_source_mismatch", "template source identity is not exact")
    if document.get("status") != "DESIGN_ONLY_BLOCKED" or document.get("evidence_class") != "MODEL_OUTPUT":
        _fail("template_state_mismatch", "template must remain design-only model output")
    runtime = document.get("runtime_registration")
    if not isinstance(runtime, Mapping) or runtime.get("status") != "AWAITING_EXTERNAL_RUNTIME":
        _fail("fabricated_external_runtime", "template runtime status is not null")
    if any(value is not None for key, value in runtime.items() if key != "status"):
        _fail("fabricated_external_runtime", "template runtime fields must be null")
    dependencies = document.get("dependency_registration")
    if not isinstance(dependencies, Mapping) or dependencies.get("status") != "AWAITING_EXTERNAL_DEPENDENCY_MANIFEST":
        _fail("fabricated_dependency_manifest", "template dependency status is not null")
    for key, value in dependencies.items():
        expected = [] if key in {"direct_stdlib_imports", "third_party_distributions"} else None
        if key != "status" and value != expected:
            _fail("fabricated_dependency_manifest", f"template dependencies.{key} is filled")
    configurations = document.get("configuration_registration")
    if configurations != {"status": "AWAITING_EXTERNAL_CONFIGURATION", "configurations": []}:
        _fail("fabricated_external_configuration", "template configuration slot is filled")
    attestation = document.get("independence_attestation")
    if not isinstance(attestation, Mapping) or attestation.get("status") != "AWAITING_EXTERNAL_Q2_INDEPENDENCE_ATTESTATION":
        _fail("fabricated_independence_attestation", "template attestation status is wrong")
    if any(value is not None for key, value in attestation.items() if key != "status"):
        _fail("fabricated_independence_attestation", "template attestation fields must be null")
    if tuple(document.get("downstream_q8q9_slots", ())) != DOWNSTREAM_SLOTS:
        _fail("downstream_slot_mismatch", "template Q8/Q9 slots are not exact and null")
    if document.get("unresolved_prerequisites") != UNRESOLVED_SLOTS:
        _fail("unresolved_prerequisite_mismatch", "template blockers are not exact")
    if document.get("nonclaim") != TEMPLATE_NONCLAIM:
        _fail("nonclaim_mismatch", "template nonclaim drifted")
    _claim_flags(document, "template")


def _configuration_hash(configuration: Mapping[str, Any]) -> str:
    context = configuration["context"]
    encoded = {
        "schema": "jx-v5-q2-observer-configuration/v1",
        "angle_method_id": configuration["angle_method_id"],
        "angle_unit_id": configuration["angle_unit_id"],
        "atan_guard_digits": configuration["atan_guard_digits"],
        "configuration_id": configuration["configuration_id"],
        "context": {
            "capitals": context["capitals"],
            "clamp": context["clamp"],
            "context_id": context["context_id"],
            "emax": context["emax"],
            "emin": context["emin"],
            "precision": context["precision"],
            "rounding": context["rounding"],
        },
        "dense_state_method_id": configuration["dense_state_method_id"],
        "epoch_unit_id": configuration["epoch_unit_id"],
        "expected_event_count": configuration["expected_event_count"],
        "maximum_atan_series_iterations": configuration["maximum_atan_series_iterations"],
        "certificate_precision": configuration["certificate_precision"],
        "maximum_relative_off_plane": configuration["maximum_relative_off_plane"],
        "maximum_root_iterations": configuration["maximum_root_iterations"],
        "observer_id": configuration["observer_id"],
        "regression_method_id": configuration["regression_method_id"],
        "root_bracket_width_tolerance": configuration["root_bracket_width_tolerance"],
        "root_method_id": configuration["root_method_id"],
        "root_isolation_method_id": configuration["root_isolation_method_id"],
        "root_tolerance_unit_id": configuration["root_tolerance_unit_id"],
        "unwrap_method_id": configuration["unwrap_method_id"],
    }
    return sha256_data(encoded)


def _validate_configuration(configuration: Mapping[str, Any], tolerance: str) -> None:
    if configuration.get("root_bracket_width_tolerance") != tolerance:
        _fail("configuration_tolerance_mismatch", "configuration tolerance roster drifted")
    context = configuration.get("context")
    if not isinstance(context, Mapping):
        _fail("configuration_mismatch", "configuration context is missing")
    actual_fixed = {
        "observer_id": configuration.get("observer_id"),
        "context_id": context.get("context_id"),
        "precision": context.get("precision"),
        "rounding": context.get("rounding"),
        "emin": context.get("emin"),
        "emax": context.get("emax"),
        "capitals": context.get("capitals"),
        "clamp": context.get("clamp"),
        **{key: configuration.get(key) for key in FIXED_CONFIGURATION if key not in {"observer_id", "context_id", "precision", "rounding", "emin", "emax", "capitals", "clamp"}},
    }
    if actual_fixed != FIXED_CONFIGURATION:
        _fail("configuration_mismatch", "configuration fixed values drifted")
    off_plane_text = configuration.get("maximum_relative_off_plane")
    try:
        off_plane = Decimal(off_plane_text)
    except (InvalidOperation, TypeError):
        _fail("invalid_off_plane_bound", "maximum_relative_off_plane is invalid")
    if not off_plane.is_finite() or not (Decimal(0) <= off_plane < Decimal(1)):
        _fail("invalid_off_plane_bound", "maximum_relative_off_plane is outside [0,1)")
    if configuration.get("observer_configuration_sha256") != _configuration_hash(configuration):
        _fail("configuration_digest_mismatch", "observer configuration hash differs")


def _parse_utc(value: Any, context: str) -> datetime:
    if not isinstance(value, str):
        _fail("invalid_timestamp", f"{context} must be a UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        _fail("invalid_timestamp", f"{context} is not a real UTC timestamp")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        _fail("invalid_timestamp", f"{context} is not canonical UTC")
    return parsed


def external_signed_payload(document: Mapping[str, Any]) -> bytes:
    """Canonical bytes signed by the Q2-specific external reviewer."""
    payload = copy.deepcopy(dict(document))
    attestation = payload.get("independence_attestation")
    if not isinstance(attestation, dict):
        _fail("attestation_missing", "independence attestation is absent")
    attestation["signed_payload_sha256"] = None
    attestation["signature"] = None
    return canonical_json(payload)


def validate_external_registration(
    document: Mapping[str, Any],
    *,
    project_root: str | Path,
    signature_verifier: Callable[[bytes, Mapping[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """Validate a separately supplied completed copy; never authorize execution."""
    private_document = json.loads(canonical_json(document).decode("ascii"))
    if not isinstance(private_document, dict):
        _fail("invalid_external_registration", "external registration must be an object")
    document = private_document
    root = Path(project_root).resolve()
    package_inspection = verify_package(root)
    schema = _load_pinned_schema(root)
    wrapper = {"$ref": "#/$defs/externalRegistrationDocument", "$defs": schema["$defs"]}
    _validate_schema(document, wrapper, "external registration")
    constants = {
        "schema": EXTERNAL_SCHEMA_ID,
        "external_registration_id": "jx.v5.solar_1pn.q2_observer.external_registration.v1",
        "qualification_id": QUALIFICATION_ID,
        "observer_package_id": PACKAGE_ID,
        "observer_source_binding": _source_binding(),
        "status": "EXTERNAL_Q2_OBSERVER_REGISTERED_NONAUTHORIZING",
        "evidence_class": "EXTERNAL_ATTESTATION_METADATA",
        "phase_required": "BEFORE_ANY_Q2_TRAJECTORY_EXECUTION",
        "downstream_q8q9_slots": list(DOWNSTREAM_SLOTS),
        "unresolved_prerequisites": UNRESOLVED_SLOTS,
        "claim_control": CLAIM_CONTROL,
        "nonclaim": EXTERNAL_NONCLAIM,
    }
    for key, expected in constants.items():
        if document.get(key) != expected:
            _fail("external_registration_mismatch", f"external registration {key} is wrong")
    _claim_flags(document, "external registration")

    runtime = document["runtime_registration"]
    if runtime.get("status") != "EXTERNAL_RUNTIME_BOUND" or any(
        value is None for key, value in runtime.items() if key != "status"
    ):
        _fail("runtime_unbound", "external runtime binding is incomplete")
    if runtime.get("python_implementation") != "CPython":
        _fail("runtime_implementation_mismatch", "the registered runtime must be exact CPython")

    dependencies = document["dependency_registration"]
    if dependencies.get("status") != "EXTERNAL_DEPENDENCY_MANIFEST_BOUND" or any(
        dependencies.get(key) is None
        for key in (
            "manifest_id", "manifest_locator", "manifest_size_bytes",
            "manifest_sha256", "manifest_canonical_sha256",
        )
    ):
        _fail("dependency_manifest_unbound", "dependency manifest is incomplete")
    if tuple(dependencies.get("direct_stdlib_imports", ())) != DIRECT_STDLIB_IMPORTS:
        _fail("dependency_roster_mismatch", "direct stdlib import roster is not exact")
    if dependencies.get("third_party_distributions") != []:
        _fail("third_party_dependency_forbidden", "observer registration must remain stdlib-only")

    registration = document["configuration_registration"]
    configurations = registration.get("configurations")
    if registration.get("status") != "EXTERNAL_CONFIGURATION_ROSTER_BOUND" or not isinstance(configurations, list) or len(configurations) != 2:
        _fail("configuration_roster_unbound", "exactly two observer configurations are required")
    if len({item.get("configuration_id") for item in configurations}) != 2:
        _fail("duplicate_configuration_id", "configuration IDs must be unique")
    for configuration, tolerance in zip(configurations, REQUIRED_ROOT_TOLERANCES):
        _validate_configuration(configuration, tolerance)
    if (
        configurations[0]["maximum_relative_off_plane"]
        != configurations[1]["maximum_relative_off_plane"]
    ):
        _fail(
            "configuration_comparison_confounded",
            "the two configurations may differ only in tolerance and configuration ID",
        )

    attestation = document["independence_attestation"]
    if attestation.get("status") != "EXTERNAL_Q2_INDEPENDENCE_ATTESTED":
        _fail("independence_unattested", "Q2-specific independence is not attested")
    for field in (
        "reviewer_identity", "reviewer_organization",
        "relationship_to_implementation_team", "conflicts_of_interest_statement",
        "attested_utc", "signature_profile_id", "signing_key_id", "signature",
    ):
        if not attestation.get(field):
            _fail("independence_unattested", f"attestation.{field} is missing")
    _parse_utc(attestation["attested_utc"], "attestation.attested_utc")
    for field in (
        "reviewer_independence_accepted", "conflicts_of_interest_absent_attested",
        "source_isolation_accepted", "runtime_binding_reviewed",
        "dependency_closure_reviewed", "configuration_roster_reviewed",
        "no_holdout_access_attested", "no_execution_attested",
    ):
        if attestation.get(field) is not True:
            _fail("independence_not_accepted", f"attestation.{field} is not true")
    payload = external_signed_payload(document)
    if attestation.get("signed_payload_sha256") != hashlib.sha256(payload).hexdigest():
        _fail("signed_payload_mismatch", "signed payload digest differs")
    if signature_verifier is None:
        _fail("signature_verification_required", "a caller-supplied signature verifier is required")
    try:
        valid = signature_verifier(payload, copy.deepcopy(dict(attestation)))
    except Exception as exc:
        _fail("signature_verification_failed", f"signature verifier failed: {exc}")
    if valid is not True:
        _fail("invalid_external_signature", "external Q2 signature did not verify")
    post_signature_package = verify_package(root)
    if any(
        post_signature_package[field] != package_inspection[field]
        for field in ("package_sha256", "registration_sha256", "source_tuple_sha256")
    ):
        _fail(
            "package_changed_during_signature_verification",
            "the bound package identity changed during signature verification",
        )
    return {
        "schema": "jx-v5-solar-1pn-q2-observer-external-registration-inspection/v1",
        "status": "EXTERNAL_Q2_OBSERVER_REGISTERED_NONAUTHORIZING",
        "boundary_package_sha256": package_inspection["package_sha256"],
        "source_tuple_sha256": OBSERVER_SOURCE_TUPLE_SHA256,
        "runtime_id": runtime["runtime_id"],
        "configuration_sha256s": tuple(
            item["observer_configuration_sha256"] for item in configurations
        ),
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_authorized": False,
        "registry_authorized": False,
        "ready": False,
        "blocked_reasons": BLOCKED_REASONS[4:],
    }


def verify_package(
    project_root: str | Path,
    registration_path: str | Path = REGISTRATION_RELATIVE,
) -> dict[str, Any]:
    """Verify the immutable design package without executing observer code."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        _fail("invalid_project_root", "project_root must be an existing directory")
    schema = _load_pinned_schema(root)
    registration_file = _safe_file(root, str(registration_path), "package registration")
    registration = _load_json(registration_file, "package registration")
    _validate_schema(registration, schema, "package registration")
    expected_constants = {
        "schema": SCHEMA_ID,
        "package_id": PACKAGE_ID,
        "predecessors": PREDECESSORS,
        "implemented_scope": list(IMPLEMENTED_SCOPE),
        "unresolved_scope": list(UNRESOLVED_SCOPE),
        "status": "DESIGN_ONLY_BLOCKED",
        "evidence_class": "MODEL_OUTPUT",
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_authorized": False,
        "registry_authorized": False,
        "ready": False,
        "referenced_q8q9_schemas": list(Q8Q9_SCHEMA_BINDINGS),
        "blocked_reasons": list(BLOCKED_REASONS),
        "claim_control": CLAIM_CONTROL,
        "nonclaim": PACKAGE_NONCLAIM,
    }
    for key, expected in expected_constants.items():
        if registration.get(key) != expected:
            _fail("registration_mismatch", f"registration.{key} is not exact")
    expected_candidate = {
        "observer_id": OBSERVER_ID,
        "review_status": "CANDIDATE_PENDING_INDEPENDENT_REVIEW",
        "commit": OBSERVER_COMMIT,
        "parent_commit": OBSERVER_PARENT_COMMIT,
        "tree": OBSERVER_TREE,
        "source_tuple_sha256": OBSERVER_SOURCE_TUPLE_SHA256,
        "files": list(CANDIDATE_FILES),
        "configuration_contract": {
            "configuration_schema": "jx-v5-q2-observer-configuration/v1",
            "required_root_tolerances": list(REQUIRED_ROOT_TOLERANCES),
            "external_values_required": ["configuration_id", "maximum_relative_off_plane"],
            "fixed_values": FIXED_CONFIGURATION,
        },
    }
    if registration.get("observer_candidate") != expected_candidate:
        _fail("observer_candidate_mismatch", "candidate binding or method contract drifted")
    if sha256_data(
        {
            "schema": "jx-v5-solar-1pn-q2-observer-source-tuple/v1",
            "commit": OBSERVER_COMMIT,
            "tree": OBSERVER_TREE,
            "files": list(CANDIDATE_FILES),
        }
    ) != OBSERVER_SOURCE_TUPLE_SHA256:
        _fail("internal_source_tuple_mismatch", "verifier source tuple constant is inconsistent")
    for binding in CANDIDATE_FILES:
        _verify_binding(root, binding, f"candidate {binding['role']}", canonical=False)
    for predecessor in PREDECESSORS.values():
        _verify_binding(root, predecessor["registration"], f"predecessor {predecessor['role']}", canonical=True)
    for binding in Q8Q9_SCHEMA_BINDINGS:
        _verify_binding(root, binding, f"referenced schema {binding['role']}", canonical=True)
    schema_binding = registration["schema_binding"]
    expected_schema_binding = {
        "role": "boundary_schema",
        "schema": SCHEMA_ID,
        "path": SCHEMA_RELATIVE.as_posix(),
        "sha256": SCHEMA_RAW_SHA256,
        "size_bytes": SCHEMA_SIZE_BYTES,
        "canonical_sha256": SCHEMA_CANONICAL_SHA256,
    }
    if schema_binding != expected_schema_binding:
        _fail("schema_binding_mismatch", "registration schema binding is wrong")
    template = _verify_binding(root, registration["external_registration_template"], "external registration template", canonical=True)
    assert template is not None
    wrapper = {"$ref": "#/$defs/externalRegistrationDocument", "$defs": schema["$defs"]}
    _validate_schema(template, wrapper, "external registration template")
    _template_semantics(template)
    _verify_binding(root, registration["verifier"], "verifier", canonical=False)
    _verify_binding(root, registration["readme"], "README", canonical=False)
    package_directory = _safe_file(root, README_RELATIVE.as_posix(), "README").parent
    entries = tuple(package_directory.iterdir())
    observed_files = {item.name for item in entries}
    expected_files = {"README.md", "registration_v1.json", "external_registration_template_v1.json", "verify_registration_v1.py"}
    if observed_files != expected_files:
        _fail("package_roster_mismatch", "package contains an unexpected or missing file")
    if any(item.is_symlink() or not item.is_file() for item in entries):
        _fail("package_roster_mismatch", "every package entry must be a regular non-symlink file")
    _claim_flags(registration, "registration")
    locked = sorted(
        [
            *list(CANDIDATE_FILES),
            *list(Q8Q9_SCHEMA_BINDINGS),
            registration["external_registration_template"],
            registration["schema_binding"],
            registration["verifier"],
            registration["readme"],
        ],
        key=lambda item: item["path"],
    )
    return {
        "schema": "jx-v5-solar-1pn-q2-observer-registration-boundary-inspection/v1",
        "package_id": PACKAGE_ID,
        "qualification_id": QUALIFICATION_ID,
        "observer_commit": OBSERVER_COMMIT,
        "source_tuple_sha256": OBSERVER_SOURCE_TUPLE_SHA256,
        "registration_sha256": sha256_file(registration_file),
        "registration_canonical_sha256": sha256_data(registration),
        "package_sha256": sha256_data(
            {
                "schema": "jx-v5-solar-1pn-q2-observer-registration-boundary-package-digest/v1",
                "registration_canonical_sha256": sha256_data(registration),
                "locked_files": locked,
            }
        ),
        "status": "DESIGN_ONLY_BLOCKED",
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_authorized": False,
        "registry_authorized": False,
        "ready": False,
        "blocked_reasons": BLOCKED_REASONS,
    }


__all__ = (
    "BLOCKED_REASONS",
    "CANDIDATE_FILES",
    "DIRECT_STDLIB_IMPORTS",
    "FIXED_CONFIGURATION",
    "IMPLEMENTED_SCOPE",
    "OBSERVER_SOURCE_TUPLE_SHA256",
    "PACKAGE_ID",
    "QUALIFICATION_ID",
    "Q2ObserverRegistrationError",
    "REQUIRED_ROOT_TOLERANCES",
    "UNRESOLVED_SCOPE",
    "canonical_json",
    "external_signed_payload",
    "sha256_data",
    "sha256_file",
    "validate_external_registration",
    "verify_package",
)
