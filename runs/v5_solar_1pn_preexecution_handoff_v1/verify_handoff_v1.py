"""Read-only verifier for the blocked V5 pre-execution handoff request.

This module verifies metadata, custody-request structure, and exact byte
closures only.  It has no CLI and imports no JX, observer, oracle, dynamics,
registry, trajectory, runner, outcome, or holdout implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_preexecution_handoff_v1")
MANIFEST_RELATIVE = PACKAGE_RELATIVE / "registration_v1.json"
README_RELATIVE = PACKAGE_RELATIVE / "README.md"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_handoff_v1.py"
SCHEMA_RELATIVE = Path("schemas/jx-v5-solar-1pn-preexecution-handoff-v1.schema.json")
TEST_RELATIVE = Path("tests/test_v5_solar_1pn_preexecution_handoff_v1.py")

PREDECESSOR_PACKAGE_RELATIVE = Path(
    "runs/v5_solar_1pn_qualification_successor_v2"
)
PREDECESSOR_MANIFEST_RELATIVE = (
    PREDECESSOR_PACKAGE_RELATIVE / "qualification_successor_v2.json"
)
PREDECESSOR_README_RELATIVE = PREDECESSOR_PACKAGE_RELATIVE / "README.md"
PREDECESSOR_VERIFIER_RELATIVE = (
    PREDECESSOR_PACKAGE_RELATIVE / "verify_qualification_successor_v2.py"
)
PREDECESSOR_SCHEMA_RELATIVE = Path(
    "schemas/jx-v5-solar-1pn-qualification-successor-v2.schema.json"
)
PREDECESSOR_TEST_RELATIVE = Path(
    "tests/test_v5_solar_1pn_qualification_successor_v2.py"
)

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "jx-v5-solar-1pn-preexecution-handoff-request/v1"
SCHEMA_RAW_SHA256 = "2a1bceeea916be87e9d933a61123f88e8e29df245d82dbcd856b8e46896dc182"
SCHEMA_SIZE_BYTES = "21510"
SCHEMA_CANONICAL_SHA256 = (
    "69ba0cf87f4201cc8538faf3c87af8f14624084f3698ef58e2c8d1260aa518d9"
)

REQUEST_ID_PREFIX = "jx.v5.solar_1pn.preexecution_handoff_request."
IDENTITY_DOMAIN = "jx-v5-solar-1pn-preexecution-handoff-request-identity/v1"
IDENTITY_SENTINEL = "CONTENT_DIGEST_SENTINEL"
SOURCE_SLOT_DOMAIN = "jx-v5-solar-1pn-preexecution-handoff-source-slot-binding/v1"

PREDECESSOR_SPEC_ID = (
    "jx.v5.solar_1pn.qualification_successor_spec."
    "f402f13c924543633925a99734d800911b01a8812ca7b293f7426fea6e07030c"
)
PREDECESSOR_SPEC_ID_PREFIX = "jx.v5.solar_1pn.qualification_successor_spec."
PREDECESSOR_IDENTITY_DOMAIN = (
    "jx-v5-solar-1pn-qualification-successor-spec-identity/v2"
)
PREDECESSOR_MANIFEST_SIZE_BYTES = "39047"
PREDECESSOR_MANIFEST_SHA256 = (
    "43dd622e651a80d7d42d31082170792b51ce646e394a5850ff2064c09d1ba21c"
)
PREDECESSOR_MANIFEST_CANONICAL_SHA256 = (
    "1416385b2c46af458579b23c8ac86d43b16b467665c443ab7acbc19782ff9221"
)
PREDECESSOR_PACKAGE_SHA256 = (
    "6f53606a452012269a9f43c7a5e62e1c39ae91501b117f424220a8941d623542"
)

PACKAGE_FILES = frozenset(
    {"README.md", "registration_v1.json", "verify_handoff_v1.py"}
)
PREDECESSOR_PACKAGE_FILES = frozenset(
    {"README.md", "qualification_successor_v2.json", "verify_qualification_successor_v2.py"}
)

LOCKED_ROLE_PATHS = (
    ("handoff_readme", README_RELATIVE.as_posix(), False),
    ("handoff_schema", SCHEMA_RELATIVE.as_posix(), True),
    ("handoff_verifier", VERIFIER_RELATIVE.as_posix(), False),
    ("handoff_disclosed_tests", TEST_RELATIVE.as_posix(), False),
)

PREDECESSOR_LOCKED_FILES = (
    {
        "role": "successor_readme",
        "path": PREDECESSOR_README_RELATIVE.as_posix(),
        "sha256": "dd0709f2186e4b540a7e60a1f63657aba6e5cc47b8dc2ba9f8d8152d2f44204a",
        "size_bytes": "7728",
        "canonical_sha256": None,
    },
    {
        "role": "successor_schema",
        "path": PREDECESSOR_SCHEMA_RELATIVE.as_posix(),
        "sha256": "f4db0e9a654bd187c4e93a3ecef5f0a3058f2893bdea76f3fa68b74e6f330a86",
        "size_bytes": "25382",
        "canonical_sha256": "9027a9ad3a63f94ebdb9f0743c1cf0b903a840e981b294a671a21b4a33bf4909",
    },
    {
        "role": "successor_verifier",
        "path": PREDECESSOR_VERIFIER_RELATIVE.as_posix(),
        "sha256": "005431008790078bb6e54c970a5c18224fe182d53852d929be625a953a10c2ec",
        "size_bytes": "55349",
        "canonical_sha256": None,
    },
    {
        "role": "successor_disclosed_tests",
        "path": PREDECESSOR_TEST_RELATIVE.as_posix(),
        "sha256": "9e31e4e98ca62dc677b542bba10f47fa23a9b0cfa29e30d0c417efbc5469b991",
        "size_bytes": "23595",
        "canonical_sha256": None,
    },
)

ROLES = (
    "fresh_external_case_custody_and_claim_scope",
    "q2_content_derived_external_registration",
    "signature_trust_profile_and_keyring",
    "q2_execution_and_adjudication_contract",
    "q2_planarity_identifiability_and_remainder_policy",
    "q3_independent_oracle_execution_contract",
    "q4_external_eih_execution_contract",
    "typed_expectation_content_schema",
    "secrecy_preserving_expectation_commitment_schema",
    "jx_execution_implementation_output_authentication_serialization_and_attempt_contract",
    "q1_independent_equation_oracle_execution_contract",
    "qualification_error_budget",
    "provenance_custody_and_independence_attestations",
    "sealed_expectation_commitment",
)
TRUST_ROLE = ROLES[2]

DEPENDENCY_ROLES = {
    ROLES[0]: (ROLES[2],),
    ROLES[1]: (ROLES[2],),
    ROLES[2]: (),
    ROLES[3]: (ROLES[1], ROLES[2], ROLES[4]),
    ROLES[4]: (ROLES[2],),
    ROLES[5]: (ROLES[2],),
    ROLES[6]: (ROLES[2],),
    ROLES[7]: (ROLES[0], ROLES[2], ROLES[3], ROLES[5], ROLES[6], ROLES[9], ROLES[10]),
    ROLES[8]: (ROLES[2],),
    ROLES[9]: (ROLES[0], ROLES[1], ROLES[2], ROLES[3], ROLES[4], ROLES[5], ROLES[6], ROLES[10]),
    ROLES[10]: (ROLES[2],),
    ROLES[11]: (ROLES[0], ROLES[2], ROLES[3], ROLES[4], ROLES[5], ROLES[6], ROLES[7], ROLES[9], ROLES[10]),
    ROLES[12]: ROLES[:12],
    ROLES[13]: ROLES[:13],
}

RESPONSIBLE_PARTY_CLASSES = {
    ROLES[0]: "SCIENTIFIC_SPONSOR_AND_FRESH_BRANCH_EXTERNAL_CUSTODIAN_OR_DISCLOSED_ROSTER_AUTHORITY",
    ROLES[1]: "BOUND_Q2_SOURCE_OWNER_AND_EXTERNAL_INDEPENDENCE_REVIEWER",
    ROLES[2]: "SCIENTIFIC_SPONSOR_AND_INDEPENDENTLY_PINNED_ROOT_TRUST_AUTHORITY",
    ROLES[3]: "Q2_CONTRACT_OWNER_AND_EXTERNAL_INDEPENDENCE_REVIEWER",
    ROLES[4]: "SCIENTIFIC_SPONSOR_AND_EXTERNAL_Q2_REVIEWER",
    ROLES[5]: "EXTERNAL_INDEPENDENT_Q3_IMPLEMENTATION_TEAM",
    ROLES[6]: "EXTERNAL_INDEPENDENT_Q4_IMPLEMENTATION_TEAM",
    ROLES[7]: "EXPECTATION_SCHEMA_OWNER_AND_INDEPENDENT_REVIEWER",
    ROLES[8]: "EXTERNAL_CUSTODIAN_AND_CRYPTOGRAPHIC_SCHEMA_REVIEWER",
    ROLES[9]: "JX_IMPLEMENTATION_OWNER_AND_EXTERNAL_INDEPENDENCE_REVIEWER",
    ROLES[10]: "EXTERNAL_INDEPENDENT_Q1_IMPLEMENTATION_TEAM",
    ROLES[11]: "JX_SCIENTIFIC_BUDGET_OWNER_AND_DISTINCT_EXTERNAL_INDEPENDENT_REVIEWER",
    ROLES[12]: "EXTERNAL_CUSTODIAN_AND_PROVENANCE_INDEPENDENCE_REVIEWERS",
    ROLES[13]: "EXTERNAL_EXPECTATION_CUSTODIAN",
}

CONFIDENTIALITY_CLASSES = {role: "PUBLIC_SIGNED_METADATA" for role in ROLES}
CONFIDENTIALITY_CLASSES.update(
    {
        ROLES[0]: "BRANCH_DEPENDENT_SECRET_OR_PUBLIC",
        ROLES[2]: "PUBLIC_TRUST_METADATA_ONLY_AFTER_ROOT_ACCEPTANCE",
        ROLES[7]: "PUBLIC_SCHEMA_ONLY",
        ROLES[8]: "PUBLIC_SCHEMA_ONLY",
        ROLES[13]: "PUBLIC_HIDING_COMMITMENT_ONLY_SECRET_PAYLOAD_OUT_OF_BAND",
    }
)

IDENTITY_CONTRACT = {
    "algorithm": "SHA-256",
    "id_prefix": REQUEST_ID_PREFIX,
    "canonicalization": "UTF8_SORTED_KEYS_COMPACT_JSON_REJECT_DUPLICATE_KEYS_ALL_JSON_NUMBERS_AND_NONFINITE",
    "domain": IDENTITY_DOMAIN,
    "envelope_keys": ["schema", "sentinel", "handoff"],
    "payload_field": "handoff",
    "sentinel": IDENTITY_SENTINEL,
    "sentinelized_field": "handoff_request_id",
    "identity_scope": "EXTERNAL_HANDOFF_REQUEST_ONLY_NOT_RESPONSE_OR_EXECUTION_QUALIFICATION",
}

STATUS_FLAGS = {
    "status": "HANDOFF_REQUEST_ONLY_BLOCKED",
    "evidence_class": "MODEL_OUTPUT",
    "scientific_evidence_artifact": False,
    "outcomes_generated": False,
    "execution_authorized": False,
    "registry_authorized": False,
    "qualification_authorized": False,
    "adjudication_authorized": False,
    "ready_for_holdout_execution": False,
    "unblinding_occurred": False,
    "case_branch_selected": False,
    "execution_qualification_id": None,
    "sealed_expectation_commitment_present": False,
    "external_responses_present": False,
    "root_trust_accepted": False,
    "resolved_successor_slots": "0",
}

CLAIM_CONTROL = {
    "claim_emitted": False,
    "claim_text": None,
    "maximum_effect": "HANDOFF_REQUEST_IDENTITY_ONLY",
}

BLOCKED_REASONS = (
    "NO_EXTERNAL_ARTIFACT_RESPONSES_PRESENT",
    "NO_CASE_OR_CLAIM_BRANCH_SELECTED",
    "NO_OUT_OF_BAND_ROOT_TRUST_POLICY_OR_ACCEPTANCE_RECORD",
    "NO_ACCEPTED_SIGNATURE_TRUST_PROFILE_OR_KEYRING",
    "NO_ACCEPTED_Q1_Q2_Q3_OR_Q4_EXECUTION_CONTRACT_CLOSURE",
    "NO_ACCEPTED_NINE_COMPONENT_QUALIFICATION_ERROR_BUDGET",
    "NO_SEALED_EXPECTATION_COMMITMENT",
    "ALL_FOURTEEN_SOURCE_SUCCESSOR_SLOTS_REMAIN_UNRESOLVED",
    "NO_EXECUTION_QUALIFICATION_ID",
    "NO_EXECUTION_UNBLINDING_ADJUDICATION_OR_SCIENTIFIC_CLAIM_AUTHORITY",
)

NONCLAIM = (
    "This content-derived artifact is only an immutable blocked request for "
    "assigned-party pre-execution artifacts. It resolves zero successor slots, "
    "selects no case or claim branch, contains no external response or trust "
    "acceptance, and grants no execution, qualification, registry, unblinding, "
    "adjudication, outcome, or scientific-claim authority."
)

SUPPORTED_SCHEMA_KEYWORDS = frozenset(
    {
        "$schema", "$id", "$defs", "$ref", "title", "type",
        "additionalProperties", "required", "properties", "const", "enum",
        "pattern", "minLength", "minItems", "maxItems", "uniqueItems", "items",
    }
)
SUPPORTED_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "integer", "boolean", "null"}
)


class PreexecutionHandoffError(ValueError):
    """The pre-execution request or one of its byte closures is unsafe."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise PreexecutionHandoffError(code, message)


def _object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail("duplicate_json_key", f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_number(token: str) -> Any:
    _fail("json_number_forbidden", f"JSON number {token!r} is forbidden")


def _reject_float(token: str) -> Any:
    _fail("binary_float_forbidden", f"JSON float {token!r} is forbidden")


def _reject_constant(token: str) -> Any:
    _fail("nonfinite_json_constant", f"JSON constant {token!r} is forbidden")


def _decode_json(data: bytes, context: str, *, manifest: bool) -> dict[str, Any]:
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        _fail("invalid_utf8", f"{context} is not UTF-8: {exc}")
    try:
        document = json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_int=_reject_number if manifest else int,
            parse_float=_reject_number if manifest else _reject_float,
            parse_constant=_reject_constant,
        )
    except PreexecutionHandoffError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        _fail("invalid_json", f"{context} is invalid JSON: {exc}")
    if not isinstance(document, dict):
        _fail("json_root", f"{context} must be a JSON object")
    return document


def _load_json(path: Path, context: str, *, manifest: bool = False) -> dict[str, Any]:
    try:
        return _decode_json(path.read_bytes(), context, manifest=manifest)
    except OSError as exc:
        _fail("file_read", f"cannot read {context}: {exc}")


def _reject_binary_float_tree(value: Any, context: str) -> None:
    if type(value) is float:
        _fail("binary_float_forbidden", f"{context} contains a binary float")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_binary_float_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_binary_float_tree(child, f"{context}[{index}]")


def _reject_all_json_numbers_tree(value: Any, context: str) -> None:
    if type(value) in {int, float}:
        _fail("json_number_forbidden", f"{context} contains a JSON number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_all_json_numbers_tree(child, f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_all_json_numbers_tree(child, f"{context}[{index}]")


def canonical_json(value: Any) -> str:
    _reject_binary_float_tree(value, "canonical JSON")
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        _fail("canonical_json", f"value is not canonicalizable: {exc}")


def sha256_data(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _stable_read(path: Path, context: str) -> bytes:
    """Read one byte snapshot and reject a file that changes during the read."""
    try:
        before = path.stat()
        data = path.read_bytes()
        after = path.stat()
    except OSError as exc:
        _fail("file_read", f"cannot read {context}: {exc}")
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        _fail("concurrent_mutation", f"{context} changed while being read")
    if after.st_nlink != 1 or path.is_symlink():
        _fail("hardlink_forbidden", f"{context} is linked")
    if len(data) != after.st_size:
        _fail("concurrent_mutation", f"{context} byte count changed while being read")
    return data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(_stable_read(path, str(path))).hexdigest()


def _safe_file(root: Path, relative: Any, context: str) -> Path:
    if type(relative) is not str or not relative:
        _fail("unsafe_path", f"{context} path must be a nonempty string")
    if "\\" in relative or "//" in relative:
        _fail("unsafe_path", f"{context} path is not normalized POSIX")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or pure.as_posix() != relative:
        _fail("unsafe_path", f"{context} path is absolute or noncanonical")
    if any(part in {"", ".", ".."} for part in pure.parts):
        _fail("unsafe_path", f"{context} path traverses or is noncanonical")
    cursor = root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail("symlink_forbidden", f"{context} path traverses a symlink")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        _fail("unsafe_path", f"{context} path does not resolve inside root: {exc}")
    if not resolved.is_file() or resolved.is_symlink():
        _fail("not_regular_file", f"{context} is not a regular file")
    try:
        link_count = resolved.stat().st_nlink
    except OSError as exc:
        _fail("file_stat", f"cannot inspect {context}: {exc}")
    if link_count != 1:
        _fail("hardlink_forbidden", f"{context} has {link_count} hard links")
    return resolved


def _verify_exact_tree(directory: Path, expected_files: frozenset[str], context: str) -> None:
    try:
        entries = tuple(directory.rglob("*"))
    except OSError as exc:
        _fail("directory_read", f"cannot inspect {context}: {exc}")
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for entry in entries:
        relative = entry.relative_to(directory).as_posix()
        if entry.is_symlink():
            _fail("symlink_forbidden", f"{context} contains symlink {relative}")
        try:
            if entry.is_dir():
                observed_directories.add(relative)
            elif entry.is_file():
                if entry.stat().st_nlink != 1:
                    _fail("hardlink_forbidden", f"{context} file {relative} has multiple hard links")
                observed_files.add(relative)
            else:
                _fail("nonregular_tree_entry", f"{context} contains nonregular entry {relative}")
        except OSError as exc:
            _fail("file_stat", f"cannot inspect {context} entry {relative}: {exc}")
    if observed_files != set(expected_files) or observed_directories:
        _fail("package_roster", f"{context} roster differs: files={sorted(observed_files)!r}, dirs={sorted(observed_directories)!r}")


def _verify_binding(
    root: Path,
    binding: Mapping[str, Any],
    context: str,
    *,
    canonical_required: bool,
) -> tuple[Path, dict[str, Any] | None, bytes]:
    path = _safe_file(root, binding.get("path"), context)
    data = _stable_read(path, context)
    if str(len(data)) != binding.get("size_bytes"):
        _fail("size_mismatch", f"{context} size differs")
    if hashlib.sha256(data).hexdigest() != binding.get("sha256"):
        _fail("raw_digest_mismatch", f"{context} raw SHA-256 differs")
    expected_canonical = binding.get("canonical_sha256")
    if canonical_required and type(expected_canonical) is not str:
        _fail("canonical_binding_missing", f"{context} lacks a canonical digest")
    if expected_canonical is None:
        return path, None, data
    document = _decode_json(data, context, manifest=False)
    if sha256_data(document) != expected_canonical:
        _fail("canonical_digest_mismatch", f"{context} canonical SHA-256 differs")
    return path, document, data


def _schema_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _resolve_ref(root_schema: Mapping[str, Any], reference: Any) -> Mapping[str, Any]:
    if type(reference) is not str or not reference.startswith("#/$defs/"):
        _fail("unsupported_schema_ref", "only local direct $defs references are supported")
    name = reference[len("#/$defs/") :]
    definitions = root_schema.get("$defs")
    if not name or "/" in name or not isinstance(definitions, Mapping):
        _fail("unsupported_schema_ref", "schema reference is not a direct definition")
    target = definitions.get(name)
    if not isinstance(target, Mapping):
        _fail("unresolved_schema_ref", f"schema definition {name!r} is missing")
    return target


def _audit_schema_node(node: Mapping[str, Any], root_schema: Mapping[str, Any], context: str) -> None:
    unknown = set(node) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        _fail("unsupported_schema_keyword", f"{context} uses {sorted(unknown)!r}")
    if "$ref" in node:
        if set(node) != {"$ref"}:
            _fail("schema_ref_siblings", f"{context} has siblings beside $ref")
        _resolve_ref(root_schema, node["$ref"])
        return
    declared = node.get("type")
    declared_types: set[str] = set()
    if isinstance(declared, str):
        declared_types = {declared}
    elif isinstance(declared, list):
        if not declared or any(type(item) is not str for item in declared):
            _fail("invalid_schema_type", f"{context} has an invalid type list")
        declared_types = set(declared)
        if len(declared_types) != len(declared):
            _fail("invalid_schema_type", f"{context} repeats a type")
    elif declared is not None:
        _fail("invalid_schema_type", f"{context} type is invalid")
    if not declared_types.issubset(SUPPORTED_SCHEMA_TYPES):
        _fail("unsupported_schema_type", f"{context} has unsupported types")
    if "object" in declared_types:
        properties = node.get("properties")
        required = node.get("required")
        if node.get("additionalProperties") is not False or not isinstance(properties, Mapping):
            _fail("open_schema_object", f"{context} is not a closed object schema")
        if not isinstance(required, list) or any(type(item) is not str for item in required):
            _fail("schema_required", f"{context}.required is invalid")
        if len(set(required)) != len(required) or set(required) != set(properties):
            _fail("schema_required", f"{context} required/properties roster differs")
        for key, child in properties.items():
            if not isinstance(child, Mapping):
                _fail("invalid_schema", f"{context}.properties.{key} is invalid")
            _audit_schema_node(child, root_schema, f"{context}.properties.{key}")
    if "array" in declared_types and "items" in node:
        child = node["items"]
        if not isinstance(child, Mapping):
            _fail("invalid_schema", f"{context}.items is invalid")
        _audit_schema_node(child, root_schema, f"{context}.items")
    if "pattern" in node:
        if type(node["pattern"]) is not str:
            _fail("invalid_schema_pattern", f"{context}.pattern is not a string")
        try:
            re.compile(node["pattern"])
        except re.error as exc:
            _fail("invalid_schema_pattern", f"{context}.pattern is invalid: {exc}")
    for keyword in ("minLength", "minItems", "maxItems"):
        if keyword in node and (type(node[keyword]) is not int or node[keyword] < 0):
            _fail("invalid_schema_bound", f"{context}.{keyword} is invalid")
    definitions = node.get("$defs")
    if definitions is not None:
        if not isinstance(definitions, Mapping):
            _fail("invalid_schema", f"{context}.$defs is invalid")
        for name, child in definitions.items():
            if not isinstance(child, Mapping):
                _fail("invalid_schema", f"{context}.$defs.{name} is invalid")
            _audit_schema_node(child, root_schema, f"{context}.$defs.{name}")


def _audit_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("$schema") != SCHEMA_DIALECT:
        _fail("schema_dialect", "handoff schema dialect differs")
    _audit_schema_node(schema, schema, "handoff schema")


def _type_matches(value: Any, declared: str) -> bool:
    return {
        "object": isinstance(value, Mapping),
        "array": isinstance(value, list),
        "string": type(value) is str,
        "integer": type(value) is int,
        "boolean": type(value) is bool,
        "null": value is None,
    }[declared]


def _validate_schema_value(
    value: Any,
    node: Mapping[str, Any],
    root_schema: Mapping[str, Any],
    context: str,
    stack: tuple[str, ...],
) -> None:
    if "$ref" in node:
        reference = node["$ref"]
        if reference in stack:
            _fail("cyclic_schema_ref", f"{context} has a cyclic schema reference")
        _validate_schema_value(value, _resolve_ref(root_schema, reference), root_schema, context, (*stack, reference))
        return
    declared = node.get("type")
    if declared is not None:
        choices = [declared] if isinstance(declared, str) else declared
        if not any(_type_matches(value, choice) for choice in choices):
            _fail("schema_type", f"{context} has the wrong type")
    if "const" in node and not _schema_equal(value, node["const"]):
        _fail("schema_const", f"{context} differs from a schema constant")
    if "enum" in node and not any(_schema_equal(value, item) for item in node["enum"]):
        _fail("schema_enum", f"{context} is outside the schema enum")
    if isinstance(value, str):
        if "minLength" in node and len(value) < node["minLength"]:
            _fail("schema_min_length", f"{context} is too short")
        if "pattern" in node and re.fullmatch(node["pattern"], value) is None:
            _fail("schema_pattern", f"{context} does not match the schema pattern")
    if isinstance(value, Mapping) and node.get("type") == "object":
        properties = node["properties"]
        if set(value) != set(properties):
            _fail("schema_properties", f"{context} keys differ from the closed schema")
        for key, child in properties.items():
            _validate_schema_value(value[key], child, root_schema, f"{context}.{key}", stack)
    if isinstance(value, list) and node.get("type") == "array":
        if "minItems" in node and len(value) < node["minItems"]:
            _fail("schema_min_items", f"{context} has too few items")
        if "maxItems" in node and len(value) > node["maxItems"]:
            _fail("schema_max_items", f"{context} has too many items")
        if node.get("uniqueItems"):
            encoded = [canonical_json(item) for item in value]
            if len(set(encoded)) != len(encoded):
                _fail("schema_unique_items", f"{context} contains duplicates")
        if "items" in node:
            for index, item in enumerate(value):
                _validate_schema_value(item, node["items"], root_schema, f"{context}[{index}]", stack)


def _validate_schema(document: Mapping[str, Any], schema: Mapping[str, Any], context: str) -> None:
    _reject_binary_float_tree(document, context)
    _validate_schema_value(document, schema, schema, context, ())


def _count_string(value: Any, target: str) -> int:
    if type(value) is str:
        return int(value == target)
    if isinstance(value, Mapping):
        return sum(_count_string(child, target) for child in value.values())
    if isinstance(value, list):
        return sum(_count_string(child, target) for child in value)
    return 0


def handoff_request_identity_sha256(document: Mapping[str, Any]) -> str:
    """Return the digest of the exact sentinelized three-key request envelope."""
    _reject_all_json_numbers_tree(document, "handoff request identity manifest")
    normalized = copy.deepcopy(dict(document))
    actual = normalized.get("handoff_request_id")
    if type(actual) is not str or not actual.startswith(REQUEST_ID_PREFIX):
        _fail("handoff_request_id", "handoff_request_id has the wrong prefix")
    if _count_string(normalized, actual) != 1:
        _fail("identity_self_reference", "handoff_request_id occurs outside its field")
    if _count_string(normalized, IDENTITY_SENTINEL) != 1:
        _fail("identity_sentinel", "request identity sentinel occurs outside its contract")
    normalized["handoff_request_id"] = IDENTITY_SENTINEL
    return sha256_data(
        {"schema": IDENTITY_DOMAIN, "sentinel": IDENTITY_SENTINEL, "handoff": normalized}
    )


def _predecessor_identity_sha256(document: Mapping[str, Any]) -> str:
    _reject_all_json_numbers_tree(document, "predecessor successor identity manifest")
    normalized = copy.deepcopy(dict(document))
    actual = normalized.get("successor_spec_id")
    if type(actual) is not str or not actual.startswith(PREDECESSOR_SPEC_ID_PREFIX):
        _fail("predecessor_spec_id", "predecessor successor ID has the wrong prefix")
    if _count_string(normalized, actual) != 1:
        _fail("predecessor_identity_self_reference", "predecessor successor ID is repeated")
    if _count_string(normalized, IDENTITY_SENTINEL) != 1:
        _fail("predecessor_identity_sentinel", "predecessor sentinel count differs")
    normalized["successor_spec_id"] = IDENTITY_SENTINEL
    return sha256_data(
        {
            "schema": PREDECESSOR_IDENTITY_DOMAIN,
            "sentinel": IDENTITY_SENTINEL,
            "successor": normalized,
        }
    )


def _load_and_verify_schema(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    locked = manifest.get("locked_files")
    matches = [item for item in locked if isinstance(item, Mapping) and item.get("role") == "handoff_schema"] if isinstance(locked, list) else []
    exact = {
        "role": "handoff_schema",
        "path": SCHEMA_RELATIVE.as_posix(),
        "size_bytes": SCHEMA_SIZE_BYTES,
        "sha256": SCHEMA_RAW_SHA256,
        "canonical_sha256": SCHEMA_CANONICAL_SHA256,
    }
    if len(matches) != 1 or matches[0] != exact:
        _fail("schema_binding", "handoff schema binding is not exact")
    _, schema, _ = _verify_binding(root, matches[0], "handoff schema", canonical_required=True)
    assert schema is not None
    _audit_schema(schema)
    return schema


def _verify_predecessor(root: Path, declared: Mapping[str, Any]) -> dict[str, Any]:
    expected_declared = {
        "role": "frozen_qualification_successor_specification",
        "successor_spec_id": PREDECESSOR_SPEC_ID,
        "successor_package_sha256": PREDECESSOR_PACKAGE_SHA256,
        "manifest_path": PREDECESSOR_MANIFEST_RELATIVE.as_posix(),
        "manifest_size_bytes": PREDECESSOR_MANIFEST_SIZE_BYTES,
        "manifest_sha256": PREDECESSOR_MANIFEST_SHA256,
        "manifest_canonical_sha256": PREDECESSOR_MANIFEST_CANONICAL_SHA256,
        "closure_policy": "DEREFERENCE_AND_RECOMPUTE_MANIFEST_ID_PACKAGE_ID_AND_FOUR_LOCKED_FILES_WITH_EXACT_PACKAGE_ROSTER",
    }
    if declared != expected_declared:
        _fail("predecessor_binding", "declared predecessor binding differs")
    source, predecessor, source_bytes = _verify_binding(
        root,
        {
            "path": PREDECESSOR_MANIFEST_RELATIVE.as_posix(),
            "size_bytes": PREDECESSOR_MANIFEST_SIZE_BYTES,
            "sha256": PREDECESSOR_MANIFEST_SHA256,
            "canonical_sha256": PREDECESSOR_MANIFEST_CANONICAL_SHA256,
        },
        "predecessor successor manifest",
        canonical_required=True,
    )
    assert predecessor is not None
    strict = _decode_json(source_bytes, "predecessor successor manifest", manifest=True)
    if strict != predecessor:
        _fail("predecessor_manifest", "strict predecessor parse differs")
    expected_id = PREDECESSOR_SPEC_ID_PREFIX + _predecessor_identity_sha256(predecessor)
    if predecessor.get("successor_spec_id") != expected_id or expected_id != PREDECESSOR_SPEC_ID:
        _fail("predecessor_spec_id_digest", "predecessor successor ID does not bind its manifest")
    locked = predecessor.get("locked_files")
    if locked != list(PREDECESSOR_LOCKED_FILES):
        _fail("predecessor_locked_roster", "predecessor locked closure differs")
    for item in PREDECESSOR_LOCKED_FILES:
        _verify_binding(
            root,
            item,
            f"predecessor locked file {item['role']}",
            canonical_required=item["canonical_sha256"] is not None,
        )
    package_directory = _safe_file(root, PREDECESSOR_README_RELATIVE.as_posix(), "predecessor README").parent
    _verify_exact_tree(package_directory, PREDECESSOR_PACKAGE_FILES, "predecessor successor package")
    computed_package = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-qualification-successor-spec-package-digest/v2",
            "successor_spec_id": PREDECESSOR_SPEC_ID,
            "manifest_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "manifest_canonical_sha256": sha256_data(predecessor),
            "locked_files": sorted(locked, key=lambda item: item["path"]),
        }
    )
    if computed_package != PREDECESSOR_PACKAGE_SHA256:
        _fail("predecessor_package_digest", "predecessor package digest differs")
    return predecessor


def _verify_dependency_graph(requests: Sequence[Mapping[str, Any]]) -> None:
    graph = {item["role"]: tuple(item["dependency_roles"]) for item in requests}
    allowed = set(ROLES)
    for role, dependencies in graph.items():
        if role in dependencies or not set(dependencies).issubset(allowed):
            _fail("dependency_graph", f"{role} has a self or unknown dependency")
        if role != TRUST_ROLE and TRUST_ROLE not in dependencies:
            _fail("trust_dependency", f"{role} does not depend on preaccepted trust")
    visiting: set[str] = set()
    complete: set[str] = set()

    def visit(role: str) -> None:
        if role in visiting:
            _fail("dependency_cycle", f"dependency cycle reaches {role}")
        if role in complete:
            return
        visiting.add(role)
        for dependency in graph[role]:
            visit(dependency)
        visiting.remove(role)
        complete.add(role)

    for role in ROLES:
        visit(role)
    if graph != DEPENDENCY_ROLES:
        _fail("dependency_graph", "artifact dependency graph differs")


def _verify_artifact_requests(
    manifest: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> None:
    requests = manifest.get("artifact_requests")
    slots = predecessor.get("unresolved_artifact_slots")
    if not isinstance(requests, list) or len(requests) != 14:
        _fail("request_roster", "artifact request roster is not exactly fourteen")
    if not isinstance(slots, list) or len(slots) < 14:
        _fail("predecessor_slot_roster", "predecessor does not contain fourteen source slots")
    if tuple(item.get("role") for item in requests if isinstance(item, Mapping)) != ROLES:
        _fail("request_roster", "artifact request role order differs")
    null_response = {
        "content_derived_response_id": None,
        "payload_schema": None,
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "canonical_sha256": None,
        "signer_identity": None,
        "trust_profile_artifact_id": None,
        "signing_key_id": None,
        "signed_payload_domain": None,
        "signature": None,
        "submitted_utc": None,
    }
    for index, (request, slot) in enumerate(zip(requests, slots[:14])):
        role = ROLES[index]
        if request.get("request_ordinal") != f"{index + 1:02}":
            _fail("request_ordinal", f"request {role} ordinal differs")
        if request.get("role") != slot.get("role") or request.get("phase_required") != slot.get("phase_required"):
            _fail("source_slot_copy", f"request {role} role/phase differs from predecessor")
        if request.get("requirements") != slot.get("requirements"):
            _fail("source_slot_copy", f"request {role} requirements differ from predecessor")
        if request.get("source_selector") != f"unresolved_artifact_slots[{index}]":
            _fail("source_selector", f"request {role} selector differs")
        expected_binding = sha256_data(
            {
                "schema": SOURCE_SLOT_DOMAIN,
                "successor_spec_id": PREDECESSOR_SPEC_ID,
                "slot": slot,
            }
        )
        if request.get("source_slot_binding_sha256") != expected_binding:
            _fail("source_slot_digest", f"request {role} source-slot digest differs")
        if request.get("responsible_party_class") != RESPONSIBLE_PARTY_CLASSES[role]:
            _fail("responsible_party", f"request {role} party class differs")
        if request.get("confidentiality_class") != CONFIDENTIALITY_CLASSES[role]:
            _fail("confidentiality", f"request {role} confidentiality class differs")
        if request.get("request_status") != "AWAITING_ASSIGNED_PARTY_SUBMISSION":
            _fail("request_status", f"request {role} status differs")
        if request.get("selection") is not None or request.get("evidence") is not None:
            _fail("premature_response", f"request {role} has selection or evidence")
        if request.get("response") != null_response:
            _fail("premature_response", f"request {role} response fields are not exactly null")
        if request.get("slot_resolved") is not False or request.get("authority_granted") is not False:
            _fail("premature_authority", f"request {role} resolves a slot or grants authority")
    _verify_dependency_graph(requests)


def _verify_manifest_semantics(
    manifest: Mapping[str, Any], predecessor: Mapping[str, Any]
) -> None:
    if manifest.get("schema") != SCHEMA_ID:
        _fail("manifest_schema", "handoff request schema ID differs")
    if manifest.get("identity_contract") != IDENTITY_CONTRACT:
        _fail("identity_contract", "request identity contract differs")
    for field, expected in STATUS_FLAGS.items():
        if manifest.get(field) != expected or type(manifest.get(field)) is not type(expected):
            _fail("premature_authority", f"manifest.{field} differs")
    if manifest.get("claim_control") != CLAIM_CONTROL:
        _fail("claim_control", "claim control differs")
    blocked = manifest.get("blocked_reasons")
    if blocked != list(BLOCKED_REASONS):
        _fail("blocked_reasons", "blocked reason roster differs")
    if manifest.get("nonclaim") != NONCLAIM:
        _fail("nonclaim", "nonclaim differs")
    trust = manifest.get("trust_bootstrap")
    if not isinstance(trust, Mapping):
        _fail("trust_bootstrap", "trust bootstrap is absent")
    acceptance = trust.get("acceptance")
    if trust.get("accepted") is not False or not isinstance(acceptance, Mapping):
        _fail("trust_bootstrap", "root trust is prematurely accepted")
    if any(value is not None for value in acceptance.values()):
        _fail("trust_bootstrap", "root-trust acceptance fields must all be null")
    if any(
        trust.get(field) is not False
        for field in (
            "response_may_self_authorize",
            "response_may_self_sign_for_acceptance",
            "proposed_key_may_be_sole_acceptance_root",
        )
    ):
        _fail("trust_bootstrap_cycle", "trust profile can self-authorize")
    if trust.get("accepting_root_must_be_independently_pinned_out_of_band") is not True:
        _fail("trust_bootstrap_cycle", "independently pinned root is not required")
    if trust.get("all_other_responses_require_preaccepted_trust") is not True:
        _fail("trust_bootstrap_cycle", "other responses do not require preaccepted trust")
    response_contract = manifest.get("common_future_response_contract")
    if not isinstance(response_contract, Mapping):
        _fail("response_contract", "future response contract is absent")
    if response_contract.get("contract_status") != "DESCRIPTIVE_FROZEN_NO_RESPONSE_INSTANCE_OR_ACCEPTANCE_VALIDATOR":
        _fail("response_contract", "request purports to validate a response")
    if response_contract.get("request_mutation_permitted") is not False:
        _fail("response_contract", "future response can mutate the request")
    if response_contract.get("accepted_response_requires_new_successor_spec_id") is not True:
        _fail("response_contract", "response acceptance does not require a new successor")
    _verify_artifact_requests(manifest, predecessor)


def _verify_locked_files(root: Path, manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    locked = manifest.get("locked_files")
    if not isinstance(locked, list):
        _fail("locked_roster", "locked_files must be a list")
    observed = tuple(
        (
            item.get("role") if isinstance(item, Mapping) else None,
            item.get("path") if isinstance(item, Mapping) else None,
            isinstance(item, Mapping) and item.get("canonical_sha256") is not None,
        )
        for item in locked
    )
    if observed != LOCKED_ROLE_PATHS:
        _fail("locked_roster", "locked role/path/canonical roster differs")
    if any(item.get("path") == MANIFEST_RELATIVE.as_posix() for item in locked):
        _fail("identity_cycle", "request manifest must not bind itself")
    for item, (_, _, canonical_required) in zip(locked, LOCKED_ROLE_PATHS):
        _verify_binding(root, item, f"locked file {item['role']}", canonical_required=canonical_required)
    return tuple(locked)


def verify_package(
    project_root: str | Path,
    manifest_path: str | Path = MANIFEST_RELATIVE,
) -> dict[str, Any]:
    """Verify the immutable blocked request without executing scientific code."""
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as exc:
        _fail("invalid_project_root", f"project root does not resolve: {exc}")
    if not root.is_dir():
        _fail("invalid_project_root", "project root is not a directory")
    if str(manifest_path) != MANIFEST_RELATIVE.as_posix():
        _fail("manifest_path", "only the fixed handoff request manifest path is accepted")
    source = _safe_file(root, str(manifest_path), "handoff request manifest")
    source_bytes = _stable_read(source, "handoff request manifest")
    manifest = _decode_json(source_bytes, "handoff request manifest", manifest=True)
    schema = _load_and_verify_schema(root, manifest)
    _validate_schema(manifest, schema, "handoff request manifest")
    predecessor = _verify_predecessor(root, manifest["predecessor_successor"])
    _verify_manifest_semantics(manifest, predecessor)

    expected_id = REQUEST_ID_PREFIX + handoff_request_identity_sha256(manifest)
    if manifest["handoff_request_id"] != expected_id:
        _fail("handoff_request_id_digest", "handoff_request_id does not bind manifest")

    locked = _verify_locked_files(root, manifest)
    package_directory = _safe_file(root, README_RELATIVE.as_posix(), "handoff README").parent
    _verify_exact_tree(package_directory, PACKAGE_FILES, "pre-execution handoff package")

    # Revalidate all interpreted bytes and both exact rosters before returning.
    # This turns concurrent worktree mutation into a closed failure instead of
    # mixing metadata from different snapshots.
    final_source_bytes = _stable_read(source, "handoff request manifest final recheck")
    if final_source_bytes != source_bytes:
        _fail("concurrent_mutation", "handoff request manifest changed during inspection")
    final_predecessor = _verify_predecessor(root, manifest["predecessor_successor"])
    if final_predecessor != predecessor:
        _fail("concurrent_mutation", "predecessor manifest changed during inspection")
    final_locked = _verify_locked_files(root, manifest)
    if final_locked != locked:
        _fail("concurrent_mutation", "handoff locked-file roster changed during inspection")
    _verify_exact_tree(package_directory, PACKAGE_FILES, "pre-execution handoff package final recheck")

    return {
        "schema": "jx-v5-solar-1pn-preexecution-handoff-request-inspection/v1",
        "handoff_request_id": manifest["handoff_request_id"],
        "identity_sha256": expected_id[len(REQUEST_ID_PREFIX) :],
        "manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "manifest_canonical_sha256": sha256_data(manifest),
        "predecessor_successor_spec_id": PREDECESSOR_SPEC_ID,
        "predecessor_package_sha256": PREDECESSOR_PACKAGE_SHA256,
        "locked_file_count": str(len(locked)),
        "requested_role_count": "14",
        "resolved_successor_slots": "0",
        "status": "HANDOFF_REQUEST_ONLY_BLOCKED",
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_authorized": False,
        "registry_authorized": False,
        "qualification_authorized": False,
        "adjudication_authorized": False,
        "ready_for_holdout_execution": False,
        "unblinding_occurred": False,
        "case_branch_selected": False,
        "execution_qualification_id": None,
        "sealed_expectation_commitment_present": False,
        "external_responses_present": False,
        "root_trust_accepted": False,
    }


__all__ = (
    "IDENTITY_DOMAIN",
    "IDENTITY_SENTINEL",
    "MANIFEST_RELATIVE",
    "PreexecutionHandoffError",
    "REQUEST_ID_PREFIX",
    "canonical_json",
    "handoff_request_identity_sha256",
    "sha256_data",
    "sha256_file",
    "verify_package",
)
