"""Read-only verifier for the blocked V5 qualification successor spec.

The module validates metadata, identity, and exact byte bindings only.  It has
no CLI and imports no JX, observer, oracle, dynamics, registry, trajectory, or
holdout implementation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_RELATIVE = Path("runs/v5_solar_1pn_qualification_successor_v2")
MANIFEST_RELATIVE = PACKAGE_RELATIVE / "qualification_successor_v2.json"
README_RELATIVE = PACKAGE_RELATIVE / "README.md"
VERIFIER_RELATIVE = PACKAGE_RELATIVE / "verify_qualification_successor_v2.py"
SCHEMA_RELATIVE = Path(
    "schemas/jx-v5-solar-1pn-qualification-successor-v2.schema.json"
)
TEST_RELATIVE = Path("tests/test_v5_solar_1pn_qualification_successor_v2.py")

SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "jx-v5-solar-1pn-qualification-successor-spec/v2"
SCHEMA_RAW_SHA256 = "f4db0e9a654bd187c4e93a3ecef5f0a3058f2893bdea76f3fa68b74e6f330a86"
SCHEMA_SIZE_BYTES = "25382"
SCHEMA_CANONICAL_SHA256 = (
    "9027a9ad3a63f94ebdb9f0743c1cf0b903a840e981b294a671a21b4a33bf4909"
)
SPEC_ID_PREFIX = "jx.v5.solar_1pn.qualification_successor_spec."
IDENTITY_DOMAIN = "jx-v5-solar-1pn-qualification-successor-spec-identity/v2"
IDENTITY_SENTINEL = "CONTENT_DIGEST_SENTINEL"
Q2_BOUNDARY_PACKAGE_SHA256 = (
    "e4abdcf4b83b9a01b8b060ea8095b229d3ee8191eb73d2ca93ef9e4fbd57ba55"
)
Q2_BOUNDARY_COMMIT = "3102c53616e3b7134092191ca94c77077128ae64"
OBSERVER_COMMIT = "d6344c49e7c6a734a337e84bd209ca5c11c23916"
OBSERVER_TREE = "4eb04188a0ec6e073e51f126b02eb8a394872aa2"
OBSERVER_SOURCE_TUPLE_SHA256 = (
    "d9d02d25689bb4561594aae77d7673433ce3c8e008b785f314c33bed143f3e27"
)

SECTION_SHA256 = {
    "bound_packages": "fd17594d2d20ff6bcb629456acf2d9be472dc3c068cd55bf8ed78c3f81cd7b13",
    "effective_protocol": "77dc93c9f064e8ec66529fa2c78c12c18140afa08295ef0d6b8c0471d3be889c",
    "unresolved_artifact_slots": "a41572420d0d5af1db24bd1fa565b3c052d7cefa68dc573c3afdcebb2aadc9f2",
    "external_identity_policy": "002187d78cc9bd57772deed6ada5cc369e20233b82800054ffa1241ce9271f31",
    "execution_qualification": "dee0af57a64cc5587c22e0b2e39dba5e47f5b3e239917e4b1ee7c445dd547628",
    "blocked_reasons": "1da0c710cee3aa99661ae6adbda925c5fb50009d5367ef3e59f64ca7fa809cdf",
    "claim_control": "fb5750ee0f287f421633e757b0137c87abee34300eb46a4667683fdc211d350f",
}

IDENTITY_CONTRACT = {
    "algorithm": "SHA-256",
    "id_prefix": SPEC_ID_PREFIX,
    "canonicalization": (
        "UTF8_SORTED_KEYS_COMPACT_JSON_REJECT_DUPLICATE_KEYS_"
        "ALL_JSON_NUMBERS_AND_NONFINITE"
    ),
    "domain": IDENTITY_DOMAIN,
    "envelope_keys": ["schema", "sentinel", "successor"],
    "payload_field": "successor",
    "sentinel": IDENTITY_SENTINEL,
    "sentinelized_field": "successor_spec_id",
    "identity_scope": "BLOCKED_SPECIFICATION_NOT_EXECUTION_QUALIFICATION",
}

LOCKED_ROLE_PATHS = (
    ("successor_readme", README_RELATIVE.as_posix(), False),
    ("successor_schema", SCHEMA_RELATIVE.as_posix(), True),
    ("successor_verifier", VERIFIER_RELATIVE.as_posix(), False),
    ("successor_disclosed_tests", TEST_RELATIVE.as_posix(), False),
)

PACKAGE_FILES = frozenset(
    {"README.md", "qualification_successor_v2.json", "verify_qualification_successor_v2.py"}
)

FROZEN_QUALIFICATION_FILES = frozenset(
    {
        "README.md",
        "prior/equation_level_perihelion_4096_fingerprints_v1.json",
        "prior/foundation_test_solar_1pn.py",
        "prior/foundation_test_v5_reference_integrator.py",
        "prior_development_cases_v1.json",
        "qualification_inputs_v1.json",
        "qualification_plan_v1.json",
        "registration_v1.json",
        "sources/README.md",
        "sources/bipm_si_brochure_9_v4_01.pdf",
        "sources/gm_de440.tpc",
        "sources/iau_2006_resolution_b2.pdf",
        "sources/iau_2006_resolution_b3.pdf",
        "sources/iau_2012_resolutions_en.pdf",
        "sources/iers_tn36_chapter_10.pdf",
        "sources/jpl_astrodynamic_parameters_2026-08-29.html",
        "sources/jpl_de440_export_2026-08-29.html",
    }
)

FROZEN_QUALIFICATION_DIRECTORIES = frozenset({"prior", "sources"})

STATUS_FLAGS = {
    "status": "SPECIFICATION_ONLY_BLOCKED",
    "evidence_class": "MODEL_OUTPUT",
    "scientific_evidence_artifact": False,
    "outcomes_generated": False,
    "execution_authorized": False,
    "registry_authorized": False,
    "ready_for_holdout_execution": False,
    "unblinding_occurred": False,
}

CLAIM_CONTROL = {
    "claim_emitted": False,
    "claim_text": None,
    "maximum_effect": "BLOCKED_SUCCESSOR_SPECIFICATION_IDENTITY_ONLY",
}

NONCLAIM = (
    "This content-derived successor specification binds frozen inputs, the exact "
    "Q2 boundary, one chronology correction, two execution-eligibility and "
    "claim-boundary corrections, and an immutable null future-artifact roster. "
    "It is not an execution qualification or scientific result and grants no "
    "execution, registry, unblinding, adjudication, or claim authority."
)

PHASES = (
    "BEFORE_EXECUTION",
    "AFTER_EXECUTION_BEFORE_UNBLIND",
    "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
    "AFTER_ADJUDICATION",
)

POSTEXECUTION_ROLES = (
    "output_manifest_commitment",
    "unblinding_record",
    "final_adjudication_and_result_record",
)

EXCLUDED_FROZEN_PLACEHOLDER = "blocker.q8.unblinding_record"

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
    }
)
SUPPORTED_SCHEMA_TYPES = frozenset(
    {"object", "array", "string", "integer", "boolean", "null"}
)


class QualificationSuccessorError(ValueError):
    """The successor specification is drifted, incomplete, or unsafe."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


def _fail(code: str, message: str) -> None:
    raise QualificationSuccessorError(code, message)


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
    except QualificationSuccessorError:
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _fail("file_read", f"cannot hash {path}: {exc}")
    return digest.hexdigest()


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


def _verify_exact_tree(
    directory: Path,
    *,
    expected_files: frozenset[str],
    expected_directories: frozenset[str] = frozenset(),
    context: str,
) -> None:
    """Require an exact recursive regular-file tree with no links or caches."""
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
                    _fail(
                        "hardlink_forbidden",
                        f"{context} file {relative} has multiple hard links",
                    )
                observed_files.add(relative)
            else:
                _fail(
                    "nonregular_tree_entry",
                    f"{context} contains nonregular entry {relative}",
                )
        except OSError as exc:
            _fail("file_stat", f"cannot inspect {context} entry {relative}: {exc}")
    if observed_files != set(expected_files):
        _fail(
            "package_roster",
            f"{context} file roster differs: {sorted(observed_files)!r}",
        )
    if observed_directories != set(expected_directories):
        _fail(
            "package_roster",
            f"{context} directory roster differs: {sorted(observed_directories)!r}",
        )


def _verify_binding(
    root: Path,
    binding: Mapping[str, Any],
    context: str,
    *,
    canonical_required: bool,
) -> tuple[Path, dict[str, Any] | None]:
    path = _safe_file(root, binding.get("path"), context)
    if str(path.stat().st_size) != binding.get("size_bytes"):
        _fail("size_mismatch", f"{context} size differs")
    if sha256_file(path) != binding.get("sha256"):
        _fail("raw_digest_mismatch", f"{context} raw SHA-256 differs")
    expected_canonical = binding.get("canonical_sha256")
    if canonical_required and type(expected_canonical) is not str:
        _fail("canonical_binding_missing", f"{context} lacks a canonical digest")
    if expected_canonical is None:
        return path, None
    document = _load_json(path, context)
    if sha256_data(document) != expected_canonical:
        _fail("canonical_digest_mismatch", f"{context} canonical SHA-256 differs")
    return path, document


def _schema_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    return left == right


def _resolve_ref(root_schema: Mapping[str, Any], reference: Any) -> Mapping[str, Any]:
    if type(reference) is not str or not reference.startswith("#/$defs/"):
        _fail("unsupported_schema_ref", "only local #/$defs references are supported")
    name = reference[len("#/$defs/") :]
    if not name or "/" in name:
        _fail("unsupported_schema_ref", "schema reference is not a direct definition")
    definitions = root_schema.get("$defs")
    if not isinstance(definitions, Mapping) or not isinstance(definitions.get(name), Mapping):
        _fail("unresolved_schema_ref", f"schema definition {name!r} is missing")
    return definitions[name]


def _audit_schema_node(
    node: Mapping[str, Any], root_schema: Mapping[str, Any], context: str
) -> None:
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
        _fail("invalid_schema_type", f"{context} type must be a string or list")
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
                _fail("invalid_schema", f"{context}.properties.{key} is not an object")
            _audit_schema_node(child, root_schema, f"{context}.properties.{key}")
    if "array" in declared_types and "items" in node:
        child = node["items"]
        if not isinstance(child, Mapping):
            _fail("invalid_schema", f"{context}.items is not an object")
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
    if "$defs" in node:
        definitions = node["$defs"]
        if not isinstance(definitions, Mapping):
            _fail("invalid_schema", f"{context}.$defs is not an object")
        for name, child in definitions.items():
            if not isinstance(child, Mapping):
                _fail("invalid_schema", f"{context}.$defs.{name} is not an object")
            _audit_schema_node(child, root_schema, f"{context}.$defs.{name}")


def _audit_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("$schema") != SCHEMA_DIALECT:
        _fail("schema_dialect", "successor schema dialect differs")
    _audit_schema_node(schema, schema, "successor schema")


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
        _validate_schema_value(
            value,
            _resolve_ref(root_schema, reference),
            root_schema,
            context,
            (*stack, reference),
        )
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
            _validate_schema_value(
                value[key], child, root_schema, f"{context}.{key}", stack
            )
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
                _validate_schema_value(
                    item,
                    node["items"],
                    root_schema,
                    f"{context}[{index}]",
                    stack,
                )


def _validate_schema(
    document: Mapping[str, Any], schema: Mapping[str, Any], context: str
) -> None:
    _reject_binary_float_tree(document, context)
    _validate_schema_value(document, schema, schema, context, ())


def successor_spec_identity_sha256(document: Mapping[str, Any]) -> str:
    """Return the digest of the exact sentinelized three-key envelope."""
    _reject_all_json_numbers_tree(document, "successor identity manifest")
    normalized = copy.deepcopy(dict(document))
    actual = normalized.get("successor_spec_id")
    if type(actual) is not str or not actual.startswith(SPEC_ID_PREFIX):
        _fail("successor_spec_id", "successor_spec_id has the wrong prefix")

    def count_string(value: Any, target: str) -> int:
        if type(value) is str:
            return int(value == target)
        if isinstance(value, Mapping):
            return sum(count_string(child, target) for child in value.values())
        if isinstance(value, list):
            return sum(count_string(child, target) for child in value)
        return 0

    if count_string(normalized, actual) != 1:
        _fail("identity_self_reference", "successor_spec_id occurs outside its field")
    if count_string(normalized, IDENTITY_SENTINEL) != 1:
        _fail("identity_sentinel", "identity sentinel must occur only in its contract")
    normalized["successor_spec_id"] = IDENTITY_SENTINEL
    return sha256_data(
        {
            "schema": IDENTITY_DOMAIN,
            "sentinel": IDENTITY_SENTINEL,
            "successor": normalized,
        }
    )


def _verify_section_locks(manifest: Mapping[str, Any]) -> None:
    for name, expected in SECTION_SHA256.items():
        if name not in manifest or sha256_data(manifest[name]) != expected:
            _fail("semantic_section_mismatch", f"manifest.{name} is not exact")


def _verify_locked_files(
    root: Path, manifest: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
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
        _fail("identity_cycle", "manifest must not bind itself in locked_files")
    for item, (_, _, canonical_required) in zip(locked, LOCKED_ROLE_PATHS):
        _verify_binding(
            root,
            item,
            f"locked file {item['role']}",
            canonical_required=canonical_required,
        )
    return tuple(locked)


def _load_and_verify_schema(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    locked = manifest.get("locked_files")
    if not isinstance(locked, list):
        _fail("locked_roster", "locked_files must be a list")
    matches = [
        item
        for item in locked
        if isinstance(item, Mapping) and item.get("role") == "successor_schema"
    ]
    expected_binding = {
        "role": "successor_schema",
        "path": SCHEMA_RELATIVE.as_posix(),
        "sha256": SCHEMA_RAW_SHA256,
        "size_bytes": SCHEMA_SIZE_BYTES,
        "canonical_sha256": SCHEMA_CANONICAL_SHA256,
    }
    if len(matches) != 1 or matches[0] != expected_binding:
        _fail("schema_binding", "successor schema binding is not exact")
    _, schema = _verify_binding(
        root, matches[0], "successor schema", canonical_required=True
    )
    assert schema is not None
    _audit_schema(schema)
    return schema


def _verify_bound_packages(
    root: Path, manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    packages = manifest["bound_packages"]
    frozen = packages["frozen_qualification"]
    execution = packages["execution_prerequisites"]
    q8q9 = packages["q8q9_prerequisites"]
    q2 = packages["q2_observer_registration_boundary"]

    exact_identities = {
        "frozen_qualification": (
            "QUALIFICATION_PACKAGE_SHA256",
            "80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237",
            "975b1b7002e358a980457acb18c9beaa90aa1c9f",
        ),
        "execution_prerequisites": (
            "REGISTRATION_CANONICAL_SHA256",
            "4aa7a82574cdd3127809d82202ca4e5a9e999f2f4d40b7fe93a01b388d002ea7",
            "dd7b22f64c04e0836c624f13b6a4a087caeeeb46",
        ),
        "q8q9_prerequisites": (
            "Q8Q9_PACKAGE_SHA256",
            "1d9008ae6a3c3a79edd06c3b01f6ab2a662246633188339c0d7f608f5712080b",
            "dd7b22f64c04e0836c624f13b6a4a087caeeeb46",
        ),
        "q2_observer_registration_boundary": (
            "Q2_OBSERVER_BOUNDARY_PACKAGE_SHA256",
            Q2_BOUNDARY_PACKAGE_SHA256,
            Q2_BOUNDARY_COMMIT,
        ),
    }
    for name, (kind, identity, commit) in exact_identities.items():
        package = packages[name]
        if (
            package.get("identity_kind") != kind
            or package.get("identity_sha256") != identity
            or package.get("commit") != commit
        ):
            _fail("bound_package_identity", f"bound package {name} identity differs")

    documents: dict[str, dict[str, Any]] = {}
    for role in ("plan", "inputs", "registration"):
        _, document = _verify_binding(
            root,
            frozen[role],
            f"frozen qualification {role}",
            canonical_required=True,
        )
        assert document is not None
        documents[f"frozen_{role}"] = document
    for name, package in (
        ("execution_registration", execution),
        ("q8q9_registration", q8q9),
        ("q2_registration", q2),
    ):
        _, document = _verify_binding(
            root,
            package["registration"],
            name.replace("_", " "),
            canonical_required=True,
        )
        assert document is not None
        documents[name] = document

    if frozen.get("qualification_id") != frozen.get("package_id"):
        _fail("bound_package_identity", "frozen package and qualification IDs differ")
    if q2.get("tree") != "1a9d0c4b4389f15322a4633cbed4fcf25ef53c69":
        _fail("q2_boundary_tree", "Q2 boundary tree differs")
    observer = q2.get("observer")
    if not isinstance(observer, Mapping) or (
        observer.get("commit") != OBSERVER_COMMIT
        or observer.get("tree") != OBSERVER_TREE
        or observer.get("source_tuple_sha256") != OBSERVER_SOURCE_TUPLE_SHA256
        or observer.get("observer_id")
        != "jx.v5.solar_1pn.q2.perihelion_observer_candidate.v1"
    ):
        _fail("observer_binding", "exact Q2 observer binding differs")
    return documents


def _artifact_record(
    root: Path, binding: Mapping[str, Any], context: str
) -> dict[str, str]:
    path, _ = _verify_binding(root, binding, context, canonical_required=False)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": str(path.stat().st_size),
    }


def _verify_frozen_qualification_closure(
    root: Path,
    plan: Mapping[str, Any],
    inputs: Mapping[str, Any],
    registration: Mapping[str, Any],
) -> None:
    frozen_directory = _safe_file(
        root,
        "runs/v5_solar_1pn_qualification/README.md",
        "frozen qualification README",
    ).parent
    _verify_exact_tree(
        frozen_directory,
        expected_files=FROZEN_QUALIFICATION_FILES,
        expected_directories=FROZEN_QUALIFICATION_DIRECTORIES,
        context="frozen qualification package",
    )
    frozen_sources_readme = _safe_file(
        root,
        "runs/v5_solar_1pn_qualification/sources/README.md",
        "frozen provenance-source README",
    )
    if (
        frozen_sources_readme.stat().st_size != 1827
        or sha256_file(frozen_sources_readme)
        != "b16827bc9e3861faa574fb63e73e8e60b0f29733a0612f87a4e61e30a828f0f5"
    ):
        _fail(
            "frozen_source_readme",
            "frozen provenance-source README raw binding differs",
        )

    manifest_binding = registration.get("prior_development_manifest")
    if not isinstance(manifest_binding, Mapping):
        _fail("frozen_closure", "prior-development manifest binding is missing")
    _, prior_manifest = _verify_binding(
        root,
        manifest_binding,
        "frozen prior-development manifest",
        canonical_required=True,
    )
    assert prior_manifest is not None

    locked: dict[str, dict[str, str]] = {}

    def retain(binding: Mapping[str, Any], context: str) -> None:
        record = _artifact_record(root, binding, context)
        previous = locked.get(record["path"])
        if previous is not None and previous != record:
            _fail("frozen_closure", f"inconsistent binding for {record['path']}")
        locked[record["path"]] = record

    retain(registration["plan"], "frozen registered plan")
    retain(registration["inputs"], "frozen registered inputs")
    retain(manifest_binding, "frozen prior-development manifest")
    for binding in registration.get("schemas", []):
        retain(binding, f"frozen schema {binding.get('schema')}")
    bindings = plan.get("bindings")
    if not isinstance(bindings, Mapping):
        _fail("frozen_closure", "frozen plan bindings are missing")
    retain(bindings["registry"], "frozen plan registry")
    retain(bindings["scientific_contract"], "frozen scientific contract")
    for binding in bindings.get("locked_files", []):
        retain(binding, f"frozen locked file {binding.get('role')}")
    prior_evidence = plan.get("prior_development_evidence")
    if not isinstance(prior_evidence, Mapping):
        _fail("frozen_closure", "frozen prior-development evidence is missing")
    for binding in prior_evidence.get("known_files", []):
        retain(binding, f"frozen known development file {binding.get('role')}")
    for source in prior_manifest.get("foundation_sources", []):
        retain(
            source["foundation_copy"],
            f"frozen foundation copy {source.get('source_role')}",
        )
        retain(
            source["current_file"],
            f"frozen current source {source.get('source_role')}",
        )
    for family in prior_manifest.get("generated_state_families", []):
        retain(
            family["fingerprint_list"],
            f"frozen fingerprint family {family.get('family_id')}",
        )
    for source in inputs.get("provenance_sources", []):
        retain(source, f"frozen provenance {source.get('provenance_id')}")

    registration_path = _safe_file(
        root,
        "runs/v5_solar_1pn_qualification/registration_v1.json",
        "frozen qualification registration",
    )
    package_sha256 = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-qualification-package-digest/v1",
            "qualification_id": plan["qualification_id"],
            "plan_canonical_sha256": sha256_data(plan),
            "input_canonical_sha256": sha256_data(inputs),
            "registration_file_sha256": sha256_file(registration_path),
            "registration_size_bytes": str(registration_path.stat().st_size),
            "registration_canonical_sha256": sha256_data(registration),
            "locked_files": [locked[path] for path in sorted(locked)],
        }
    )
    if package_sha256 != (
        "80b4b6196167d95ce6cc72bf5a5ffba90e0fdc748d9144d2d9f83183b974f237"
    ):
        _fail("frozen_package_digest", "frozen qualification package digest differs")


def _verify_execution_prerequisite_closure(
    root: Path, registration: Mapping[str, Any]
) -> None:
    artifacts = registration.get("artifacts")
    schemas = registration.get("schemas")
    if not isinstance(artifacts, list) or len(artifacts) != 3:
        _fail("execution_closure", "execution artifact roster differs")
    if not isinstance(schemas, list) or len(schemas) != 4:
        _fail("execution_closure", "execution schema roster differs")
    for binding in artifacts:
        _verify_binding(
            root,
            binding,
            f"execution artifact {binding.get('role')}",
            canonical_required=True,
        )
    for binding in schemas:
        _verify_binding(
            root,
            binding,
            f"execution schema {binding.get('role')}",
            canonical_required=True,
        )
    _verify_binding(
        root, registration["verifier"], "execution verifier", canonical_required=False
    )
    _verify_binding(
        root, registration["readme"], "execution README", canonical_required=False
    )
    package = _safe_file(
        root,
        "runs/v5_solar_1pn_execution_prerequisites_v1/README.md",
        "execution README",
    ).parent
    expected = {
        "README.md",
        "q2_observer_registration_v1.json",
        "q3_oracle_registration_v1.json",
        "q4_eih_registration_v1.json",
        "registration_v1.json",
        "verify_execution_prerequisites_v1.py",
    }
    _verify_exact_tree(
        package,
        expected_files=frozenset(expected),
        context="execution prerequisite package",
    )


def _verify_q8q9_closure(
    root: Path, registration: Mapping[str, Any]
) -> None:
    predecessor = registration.get("predecessor")
    if not isinstance(predecessor, Mapping):
        _fail("q8q9_closure", "Q8/Q9 predecessor binding is missing")
    for role in ("plan", "inputs", "registration"):
        _verify_binding(
            root,
            predecessor[role],
            f"Q8/Q9 predecessor {role}",
            canonical_required=True,
        )
    artifacts = registration.get("artifacts")
    schemas = registration.get("schemas")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        _fail("q8q9_closure", "Q8/Q9 artifact roster differs")
    if not isinstance(schemas, list) or len(schemas) != 6:
        _fail("q8q9_closure", "Q8/Q9 schema roster differs")
    for binding in [*artifacts, *schemas]:
        _verify_binding(
            root,
            binding,
            f"Q8/Q9 bound file {binding.get('role')}",
            canonical_required=True,
        )
    _verify_binding(
        root, registration["verifier"], "Q8/Q9 verifier", canonical_required=False
    )
    _verify_binding(
        root, registration["readme"], "Q8/Q9 README", canonical_required=False
    )

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
            [*artifacts, *schemas, registration["verifier"], registration["readme"]],
            key=lambda item: item["path"],
        )
    ]
    package_sha256 = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-q8q9-prerequisite-package-digest/v1",
            "predecessor_package_sha256": registration["predecessor"][
                "package_sha256"
            ],
            "registration_canonical_sha256": sha256_data(registration),
            "locked_files": locked,
        }
    )
    if package_sha256 != (
        "1d9008ae6a3c3a79edd06c3b01f6ab2a662246633188339c0d7f608f5712080b"
    ):
        _fail("q8q9_package_digest", "Q8/Q9 package digest differs")

    package = _safe_file(
        root,
        "runs/v5_solar_1pn_qualification_q8q9_prerequisites_v1/README.md",
        "Q8/Q9 README",
    ).parent
    expected = {
        "README.md",
        "error_budget_template_v1.json",
        "external_artifact_slots_v1.json",
        "registration_v1.json",
        "verify_prerequisites_v1.py",
    }
    _verify_exact_tree(
        package,
        expected_files=frozenset(expected),
        context="Q8/Q9 prerequisite package",
    )
    for forbidden in (
        "custody_attestation_v1.json",
        "independence_attestation_v1.json",
        "sealed_expectation_commitment_v1.json",
        "unblinding_record_v1.json",
    ):
        candidate = package / forbidden
        if candidate.exists() or candidate.is_symlink():
            _fail("q8q9_closure", f"forbidden realized instance {forbidden} exists")


def _git_blob_sha1(data: bytes) -> str:
    header = b"blob " + str(len(data)).encode("ascii") + b"\0"
    return hashlib.sha1(header + data).hexdigest()


def _verify_q2_boundary(
    root: Path, registration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        registration.get("package_id")
        != "jx.v5.solar_1pn.q2_observer_registration_boundary.v1"
        or registration.get("status") != "DESIGN_ONLY_BLOCKED"
    ):
        _fail("q2_boundary", "Q2 boundary identity or state differs")
    for flag in (
        "scientific_evidence_artifact",
        "outcomes_generated",
        "execution_authorized",
        "registry_authorized",
        "ready",
    ):
        if registration.get(flag) is not False:
            _fail("q2_boundary_authority", f"Q2 boundary {flag} must be false")

    candidate = registration.get("observer_candidate")
    if not isinstance(candidate, Mapping) or (
        candidate.get("commit") != OBSERVER_COMMIT
        or candidate.get("tree") != OBSERVER_TREE
        or candidate.get("source_tuple_sha256") != OBSERVER_SOURCE_TUPLE_SHA256
    ):
        _fail("observer_binding", "Q2 candidate identity differs")
    files = candidate.get("files")
    if not isinstance(files, list) or len(files) != 4:
        _fail("observer_roster", "Q2 candidate file roster differs")
    for item in files:
        path, _ = _verify_binding(
            root, item, f"Q2 candidate {item.get('role')}", canonical_required=False
        )
        try:
            data = path.read_bytes()
        except OSError as exc:
            _fail("file_read", f"cannot read Q2 candidate blob: {exc}")
        if _git_blob_sha1(data) != item.get("git_blob"):
            _fail("git_blob_mismatch", f"Q2 candidate {item.get('role')} blob differs")
    source_tuple = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-q2-observer-source-tuple/v1",
            "commit": OBSERVER_COMMIT,
            "tree": OBSERVER_TREE,
            "files": files,
        }
    )
    if source_tuple != OBSERVER_SOURCE_TUPLE_SHA256:
        _fail("observer_source_tuple", "Q2 observer source tuple differs")

    predecessors = registration.get("predecessors")
    if not isinstance(predecessors, Mapping) or set(predecessors) != {
        "qualification",
        "execution_prerequisites",
        "q8q9_prerequisites",
    }:
        _fail("q2_predecessors", "Q2 predecessor roster differs")
    for item in predecessors.values():
        _verify_binding(
            root,
            item["registration"],
            f"Q2 predecessor {item.get('role')}",
            canonical_required=True,
        )

    q8_schemas = registration.get("referenced_q8q9_schemas")
    if not isinstance(q8_schemas, list) or len(q8_schemas) != 3:
        _fail("q2_schema_roster", "Q2 referenced Q8/Q9 schema roster differs")
    legacy_schema: dict[str, Any] | None = None
    for item in q8_schemas:
        _, document = _verify_binding(
            root,
            item,
            f"Q2 referenced schema {item.get('role')}",
            canonical_required=True,
        )
        if item.get("role") == "q8q9_sealed_expectation_schema":
            assert document is not None
            legacy_schema = document
    if legacy_schema is None:
        _fail("legacy_schema_binding", "legacy expectation schema is not bound")

    template_binding = registration["external_registration_template"]
    _, template = _verify_binding(
        root,
        template_binding,
        "Q2 external registration template",
        canonical_required=True,
    )
    assert template is not None
    if (
        template.get("external_registration_id")
        != "jx.v5.solar_1pn.q2_observer.external_registration.v1"
        or template.get("status") != "DESIGN_ONLY_BLOCKED"
    ):
        _fail("static_external_id", "Q2 static external template identity differs")

    schema_binding = registration["schema_binding"]
    _verify_binding(
        root, schema_binding, "Q2 boundary schema", canonical_required=True
    )
    _verify_binding(root, registration["verifier"], "Q2 verifier", canonical_required=False)
    _verify_binding(root, registration["readme"], "Q2 README", canonical_required=False)

    q2_directory = _safe_file(
        root,
        "runs/v5_solar_1pn_q2_observer_registration_v1/README.md",
        "Q2 README",
    ).parent
    expected_names = {
        "README.md",
        "registration_v1.json",
        "external_registration_template_v1.json",
        "verify_registration_v1.py",
    }
    _verify_exact_tree(
        q2_directory,
        expected_files=frozenset(expected_names),
        context="Q2 boundary package",
    )

    locked = sorted(
        [
            *files,
            *q8_schemas,
            template_binding,
            schema_binding,
            registration["verifier"],
            registration["readme"],
        ],
        key=lambda item: item["path"],
    )
    package_sha256 = sha256_data(
        {
            "schema": (
                "jx-v5-solar-1pn-q2-observer-registration-boundary-"
                "package-digest/v1"
            ),
            "registration_canonical_sha256": sha256_data(registration),
            "locked_files": locked,
        }
    )
    if package_sha256 != Q2_BOUNDARY_PACKAGE_SHA256:
        _fail("q2_package_digest", "Q2 boundary package SHA-256 differs")
    return template, legacy_schema


def _verify_frozen_and_effective_semantics(
    manifest: Mapping[str, Any],
    inputs: Mapping[str, Any],
    q8q9_registration: Mapping[str, Any],
    legacy_schema: Mapping[str, Any],
) -> None:
    protocol = manifest["effective_protocol"]
    corrections = protocol["chronology_corrections"]
    if len(corrections) != 1:
        _fail("chronology_correction", "exactly one chronology correction is required")
    correction = corrections[0]
    source = correction["frozen_source"]
    blocked = inputs.get("blocked_artifacts")
    if not isinstance(blocked, list):
        _fail("frozen_blocker_roster", "frozen blocked_artifacts is missing")
    matches = [
        item for item in blocked if item.get("placeholder_id") == EXCLUDED_FROZEN_PLACEHOLDER
    ]
    if len(matches) != 1:
        _fail("frozen_q8_source", "frozen Q8 unblinding placeholder is not unique")
    observed_fields = {
        key: matches[0].get(key) for key in source["expected_fields"]
    }
    if observed_fields != source["expected_fields"]:
        _fail("frozen_q8_source", "frozen Q8 contradictory source fields differ")
    rule = correction["effective_rule"]
    required_rule = {
        "preexecution_unblinding_status": "FORBIDDEN_NOT_APPLICABLE",
        "required_before_execution": False,
        "required_predecessor_artifact_role": "output_manifest_commitment",
        "required_predecessor_custody_state": "OUTPUTS_COMMITTED_EXPECTATIONS_SEALED",
        "phase_required": "AFTER_OUTPUT_COMMIT_BEFORE_ADJUDICATION",
        "first_valid_unblinding_only": True,
        "valid_unblinding_ordinal": "1",
        "invalid_or_aborted_requests": (
            "REVEAL_NOTHING_AND_DO_NOT_CHANGE_VALID_ORDINAL"
        ),
        "post_unblinding_tuning_allowed": False,
        "q8_readiness_effect": (
            "UNBLINDING_RECORD_EXCLUDED_FROM_PREEXECUTION_READINESS"
        ),
        "q8_postrun_adjudication_effect": (
            "FIRST_VALID_UNBLINDING_RECORD_REQUIRED"
        ),
    }
    if rule != required_rule:
        _fail("chronology_correction", "effective Q8 chronology rule differs")

    frozen_required = [
        (item.get("placeholder_id"), item.get("artifact_kind"))
        for item in blocked
        if item.get("required_before_execution") is True
        and item.get("placeholder_id") != EXCLUDED_FROZEN_PLACEHOLDER
    ]
    resolution_map = protocol["frozen_preexecution_blocker_resolution_map"]
    mapped = [(item["placeholder_id"], item["artifact_kind"]) for item in resolution_map]
    if len(frozen_required) != 25 or mapped != frozen_required:
        _fail(
            "frozen_blocker_resolution_map",
            "resolution map is not the exact frozen preexecution domain minus Q8 unblind",
        )
    if len({item["placeholder_id"] for item in resolution_map}) != 25:
        _fail("frozen_blocker_resolution_map", "resolution map repeats a placeholder")

    slots = manifest["unresolved_artifact_slots"]
    phase_requirements = protocol["phase_requirements"]
    if tuple(item["phase"] for item in phase_requirements) != PHASES:
        _fail("phase_roster", "phase order differs")
    for phase_item in phase_requirements:
        observed = [
            item["role"]
            for item in slots
            if item["phase_required"] == phase_item["phase"]
        ]
        if observed != phase_item["required_artifact_roles"]:
            _fail("phase_roster", f"{phase_item['phase']} role order differs")
    preexecution_roles = phase_requirements[0]["required_artifact_roles"]
    if len(preexecution_roles) != 14 or len(slots) != 17:
        _fail("phase_roster", "expected exact 14 preexecution plus 3 postrun roles")
    if tuple(item["role"] for item in slots[-3:]) != POSTEXECUTION_ROLES:
        _fail("phase_roster", "postrun child role order differs")
    if manifest["external_identity_policy"][
        "future_execution_identity_preexecution_role_roster"
    ] != preexecution_roles:
        _fail("execution_identity_roster", "future execution roster differs")
    for item in resolution_map:
        if not set(item["successor_roles"]).issubset(preexecution_roles):
            _fail("frozen_blocker_resolution_map", "mapping names a non-preexec role")
    for item in slots:
        for field in (
            "content_derived_artifact_id",
            "path",
            "size_bytes",
            "sha256",
            "canonical_sha256",
        ):
            if item[field] is not None:
                _fail("slot_filled_in_place", f"slot {item['role']}.{field} must stay null")

    lifecycle = q8q9_registration.get("lifecycle", {}).get("phase_requirements")
    if not isinstance(lifecycle, list) or [item.get("phase") for item in lifecycle] != list(
        PHASES[:3]
    ):
        _fail("q8q9_chronology", "bound Q8/Q9 lifecycle phases differ")
    if lifecycle[1].get("required_artifact_roles") != ["output_manifest_commitment"]:
        _fail("q8q9_chronology", "Q8/Q9 output commitment phase differs")
    if lifecycle[2].get("required_artifact_roles") != ["unblinding_record"]:
        _fail("q8q9_chronology", "Q8/Q9 unblinding phase differs")

    eligibility = protocol["eligibility_and_claim_corrections"]
    disclosed = [
        item["fixture_id"]
        for item in inputs.get("fixtures", [])
        if item.get("fixture_id", "").startswith("fixture.holdout.")
    ]
    if disclosed != eligibility[0]["expected_values"] or len(disclosed) != 12:
        _fail("disclosed_fixture_roster", "disclosed fixture correction roster differs")
    exposed = eligibility[1]["expected_values"]
    if not all(
        field in legacy_schema.get("required", [])
        and field in legacy_schema.get("properties", {})
        for field in exposed
    ):
        _fail("legacy_expectation_schema", "legacy exposed hash fields differ")
    if protocol["legacy_sealed_expectation_schema_execution_acceptable"] is not False:
        _fail("legacy_expectation_schema", "legacy expectation schema became acceptable")

    q2 = inputs.get("q2_perihelion_observer")
    if not isinstance(q2, Mapping) or len(q2.get("case_cells", [])) != 24:
        _fail("q2_cell_roster", "frozen Q2 does not contain exactly 24 cells")
    axes = (
        q2.get("eccentricity_roster"),
        q2.get("inverse_c_squared_ladder"),
        q2.get("step_grid"),
        q2.get("event_tolerance_grid"),
    )
    expected_cells = {
        (eccentricity, inverse_c_squared, step, tolerance)
        for eccentricity in axes[0]
        for inverse_c_squared in axes[1]
        for step in axes[2]
        for tolerance in axes[3]
    }
    observed_cells = {
        (
            item["eccentricity"],
            item["inverse_c_squared_value"],
            item["step_size"],
            item["event_tolerance"],
        )
        for item in q2["case_cells"]
    }
    if (
        len(expected_cells) != 24
        or observed_cells != expected_cells
        or q2.get("complete_cartesian_product") is not True
        or q2.get("orbit_count") != 2
        or len(q2.get("inverse_c_squared_ladder", [])) != 2
    ):
        _fail("q2_cell_roster", "frozen Q2 Cartesian or identifiability facts differ")


def _verify_manifest_semantics(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != SCHEMA_ID:
        _fail("manifest_schema", "successor manifest schema ID differs")
    if manifest.get("identity_contract") != IDENTITY_CONTRACT:
        _fail("identity_contract", "identity contract differs")
    _verify_section_locks(manifest)
    for field, expected in STATUS_FLAGS.items():
        if manifest.get(field) != expected or type(manifest.get(field)) is not type(expected):
            _fail("premature_authority", f"manifest.{field} differs")
    if manifest.get("claim_control") != CLAIM_CONTROL:
        _fail("claim_control", "claim control differs")
    if manifest.get("nonclaim") != NONCLAIM:
        _fail("nonclaim", "nonclaim differs")
    execution = manifest["execution_qualification"]
    if execution != {
        "status": "AWAITING_NEW_CONTENT_DERIVED_EXECUTION_ENVELOPE",
        "execution_qualification_id": None,
        "path": None,
        "size_bytes": None,
        "sha256": None,
        "canonical_sha256": None,
    }:
        _fail("execution_qualification", "execution qualification slot is not exact/null")


def verify_package(
    project_root: str | Path,
    manifest_path: str | Path = MANIFEST_RELATIVE,
) -> dict[str, Any]:
    """Verify the immutable blocked specification without executing science."""
    try:
        root = Path(project_root).resolve(strict=True)
    except OSError as exc:
        _fail("invalid_project_root", f"project root does not resolve: {exc}")
    if not root.is_dir():
        _fail("invalid_project_root", "project root is not a directory")
    if str(manifest_path) != MANIFEST_RELATIVE.as_posix():
        _fail("manifest_path", "only the fixed successor manifest path is accepted")
    source = _safe_file(root, str(manifest_path), "successor manifest")
    manifest = _load_json(source, "successor manifest", manifest=True)
    schema = _load_and_verify_schema(root, manifest)
    _validate_schema(manifest, schema, "successor manifest")
    _verify_manifest_semantics(manifest)

    expected_id = SPEC_ID_PREFIX + successor_spec_identity_sha256(manifest)
    if manifest["successor_spec_id"] != expected_id:
        _fail("successor_spec_id_digest", "successor_spec_id does not bind manifest")

    locked = _verify_locked_files(root, manifest)
    package_directory = _safe_file(
        root, README_RELATIVE.as_posix(), "successor README"
    ).parent
    _verify_exact_tree(
        package_directory,
        expected_files=PACKAGE_FILES,
        context="successor specification package",
    )

    documents = _verify_bound_packages(root, manifest)
    _verify_frozen_qualification_closure(
        root,
        documents["frozen_plan"],
        documents["frozen_inputs"],
        documents["frozen_registration"],
    )
    _verify_execution_prerequisite_closure(root, documents["execution_registration"])
    _verify_q8q9_closure(root, documents["q8q9_registration"])
    _, legacy_schema = _verify_q2_boundary(root, documents["q2_registration"])
    _verify_frozen_and_effective_semantics(
        manifest,
        documents["frozen_inputs"],
        documents["q8q9_registration"],
        legacy_schema,
    )

    registration_canonical = sha256_data(manifest)
    package_sha256 = sha256_data(
        {
            "schema": "jx-v5-solar-1pn-qualification-successor-spec-package-digest/v2",
            "successor_spec_id": manifest["successor_spec_id"],
            "manifest_file_sha256": sha256_file(source),
            "manifest_canonical_sha256": registration_canonical,
            "locked_files": sorted(locked, key=lambda item: item["path"]),
        }
    )
    return {
        "schema": "jx-v5-solar-1pn-qualification-successor-spec-inspection/v2",
        "successor_spec_id": manifest["successor_spec_id"],
        "identity_sha256": expected_id[len(SPEC_ID_PREFIX) :],
        "package_sha256": package_sha256,
        "manifest_sha256": sha256_file(source),
        "manifest_canonical_sha256": registration_canonical,
        "q2_boundary_package_sha256": Q2_BOUNDARY_PACKAGE_SHA256,
        "q2_observer_commit": OBSERVER_COMMIT,
        "status": "SPECIFICATION_ONLY_BLOCKED",
        "execution_qualification_id": None,
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_authorized": False,
        "registry_authorized": False,
        "ready_for_holdout_execution": False,
        "unblinding_occurred": False,
        "blocked_reasons": tuple(manifest["blocked_reasons"]),
    }


__all__ = (
    "IDENTITY_DOMAIN",
    "IDENTITY_SENTINEL",
    "MANIFEST_RELATIVE",
    "Q2_BOUNDARY_PACKAGE_SHA256",
    "QualificationSuccessorError",
    "SPEC_ID_PREFIX",
    "canonical_json",
    "sha256_data",
    "sha256_file",
    "successor_spec_identity_sha256",
    "verify_package",
)
