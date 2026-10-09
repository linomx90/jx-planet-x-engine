#!/usr/bin/env python3
"""Seal, but never execute, the 12-object TNO covariance pilot.

This module only verifies retained inputs and writes a create-only protocol.  It
has no dynamics, element-to-Cartesian conversion, or classification execution
surface.  Every scientific and execution claim remains false until separate,
reviewed implementations close all of the protocol's blockers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_covariance_classification_pilot_v1.py"
ACQUISITION_TOOL = PROJECT_ROOT / "tools" / "jx_tno_sbdb_covariance_acquisition_v1.py"
ACQUISITION_TEST = PROJECT_ROOT / "tests" / "test_tno_sbdb_covariance_acquisition_v1.py"
ACQUISITION_ROOT = PROJECT_ROOT / "runs" / "tno_sbdb_covariance_acquisition_v1"
CLONE_TOOL = PROJECT_ROOT / "tools" / "jx_tno_covariance_clone_materialization_v1.py"
CLONE_TEST = PROJECT_ROOT / "tests" / "test_tno_covariance_clone_materialization_v1.py"
CLONE_ROOT = PROJECT_ROOT / "runs" / "tno_covariance_clones_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "runs" / "tno_covariance_classification_pilot_v1" / "PROTOCOL.json"

SCHEMA = "jx.tno.covariance-classification-pilot.protocol.v1"
DOMAIN = b"jx.tno.covariance-classification-pilot.protocol.v1\0"
CLASSIFICATION = "TWELVE_OBJECT_COVARIANCE_CLASSIFICATION_PILOT_PREPARATION_ONLY"

ACQUISITION_BINDINGS = (
    (
        "implementation",
        "tools/jx_tno_sbdb_covariance_acquisition_v1.py",
        31_175,
        "51e96414e136e2870542383eb505c75b58c46b0e38f70fe0417b81791047eeca",
    ),
    (
        "verification_test",
        "tests/test_tno_sbdb_covariance_acquisition_v1.py",
        11_462,
        "a9c39da396c43f06b34c813e70ca2a267d1ff7007150f6e9ad171dcee8f352f4",
    ),
    (
        "protocol",
        "runs/tno_sbdb_covariance_acquisition_v1/PROTOCOL.json",
        6_261,
        "4f3de80ba98b43c907bb846bd35724a0f1bd0ef9ef152e7642e3af65a2229acc",
    ),
    (
        "custody",
        "runs/tno_sbdb_covariance_acquisition_v1/CUSTODY.json",
        16_520,
        "71e2af48651a8a3d7c6913c551c901705e699f79293f2fcd4b5303d0ac91d288",
    ),
)
CLONE_BINDINGS = (
    (
        "implementation",
        "tools/jx_tno_covariance_clone_materialization_v1.py",
        27_114,
        "49f00b856810225a495cbd16e34282db2199e2295d755c2bb2aaba4897b7ba53",
    ),
    (
        "verification_test",
        "tests/test_tno_covariance_clone_materialization_v1.py",
        9_607,
        "c2fdd4eddd85a9a5655e72bb8ad899bd6d3dc5db9c904b4259cee23b5cacbf0d",
    ),
    (
        "protocol",
        "runs/tno_covariance_clones_v1/PROTOCOL.json",
        3_358,
        "88745a6f8286cf1c6f717033973fa972457f8e088eebbfe1d6047b1286a1f21b",
    ),
    (
        "clones",
        "runs/tno_covariance_clones_v1/CLONES.json",
        531_609,
        "395ce9914df0efe235e2144c24abd157f5aa3ba87beba87fa39891ec42312014",
    ),
)

ACQUISITION_PROTOCOL_CONTENT_SHA256 = (
    "176b56c4784bef58a8eacc6e34d99edb17d21d70c559a245b7c0677cdb1a4a60"
)
ACQUISITION_CUSTODY_CONTENT_SHA256 = (
    "d326b424aa417a573c933bd0919a9e3b4a5ebe6b7f258f1f7a6302dabdf5841f"
)
ACQUISITION_RAW_ROSTER_SHA256 = (
    "605b191ec8df5ab52f12f5018f8376b4763e6c4e4b5aeb1998c8dae3018f8f46"
)
CLONE_PROTOCOL_CONTENT_SHA256 = (
    "939e6c3f8e3773f2af4675bab95369e702461ae00edf311b71c02f16785ab798"
)
CLONE_ARTIFACT_CONTENT_SHA256 = (
    "9f1c0df03a2001f6249aafd423c4c9feb8bde9a75467416d35b81b88bb2d59d8"
)
CLONE_BINARY_PAYLOAD_SHA256 = (
    "3f296d6bf9489598674e152fc7511641023c9d513fd4f6ca6d33ab6ec7293278"
)

EXPECTED_OBJECTS = (
    ("jpl-20090377", 20_090_377, "DETACHED_TNO_SCREEN_V1", "2458411.5"),
    ("jpl-20474640", 20_474_640, "DETACHED_TNO_SCREEN_V1", "2456187.5"),
    ("jpl-54526835", 54_526_835, "DETACHED_TNO_SCREEN_V1", "2460034.5"),
    ("jpl-20541132", 20_541_132, "DETACHED_TNO_SCREEN_V1", "2458052.5"),
    ("jpl-50666516", 50_666_516, "DETACHED_TNO_SCREEN_V1", "2456596.5"),
    ("jpl-20148209", 20_148_209, "DETACHED_TNO_SCREEN_V1", "2452568.5"),
    ("jpl-20574372", 20_574_372, "SCATTERING_DISK_CONTROL_V1", "2458299.5"),
    ("jpl-20303775", 20_303_775, "SCATTERING_DISK_CONTROL_V1", "2458767.5"),
    ("jpl-20026181", 20_026_181, "SCATTERING_DISK_CONTROL_V1", "2458156.5"),
    ("jpl-20042301", 20_042_301, "SCATTERING_DISK_CONTROL_V1", "2458324.5"),
    ("jpl-20026375", 20_026_375, "SCATTERING_DISK_CONTROL_V1", "2457961.5"),
    ("jpl-20015874", 20_015_874, "SCATTERING_DISK_CONTROL_V1", "2458098.5"),
)

RESOLVED_SOURCE_IDS = (
    "SUN",
    "MERCURY",
    "VENUS",
    "EARTH",
    "MOON",
    "MARS_BARYCENTER",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
    "PLUTO_BARYCENTER",
)
STAGE_1_HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)

BLOCKERS = (
    "EXACT_NATIVE_COVARIANCE_EPOCH_TIME_SCALE_FRAME_ORIGIN_GM_AND_ELEMENT_TO_CARTESIAN_TRANSFORM_NOT_IMPLEMENTED_OR_QUALIFIED",
    "DE441_NATIVE_RESOLVED_ELEVEN_BODY_SOURCE_NOT_BOUND_AND_NO_PREREGISTERED_DE440_DE441_SENSITIVITY_BRIDGE",
    "RESOLVED_ELEVEN_BODY_SOURCE_OR_SEPARATELY_QUALIFIED_REDUCTION_NOT_BOUND",
    "HYBRID_CLOSE_ENCOUNTER_ENGINE_NOT_IMPLEMENTED_AND_QUALIFIED_FOR_THIS_PILOT",
    "EXECUTABLE_CLASSIFIER_AND_RESONANCE_RULES_NOT_FROZEN",
    "INDEPENDENT_NUMERICAL_ERROR_BUDGET_AND_ACCEPTANCE_THRESHOLDS_NOT_FROZEN",
    "EXECUTION_RUNTIME_SOFTWARE_HARDWARE_AND_DEVICE_IDENTITY_NOT_BOUND",
)

CLAIMS = {
    "prepare_only_protocol_frozen": True,
    "sealed_inputs_revalidated_and_clone_payload_rederived": True,
    "retained_covariance_bytes_schema_conformant": True,
    "retained_covariance_bytes_transport_authenticated": False,
    "retained_covariance_bytes_authoritative_provider_origin_proven": False,
    "native_epoch_cartesian_transform_qualified": False,
    "de441_native_source_bound": False,
    "de440_de441_sensitivity_bridge_qualified": False,
    "resolved_known_planets_source_bound": False,
    "hybrid_close_encounter_engine_qualified": False,
    "executable_classifier_and_resonance_rules_frozen": False,
    "independent_numerical_budget_frozen": False,
    "execution_runtime_identity_bound": False,
    "future_execution_externally_preregistered": False,
    "pilot_execution_authorized": False,
    "pilot_executed": False,
    "dynamical_classifications_generated": False,
    "observational_likelihood_evaluated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class PilotProtocolError(ValueError):
    """A sealed input or prepare-only protocol failed closed."""


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256_bytes(DOMAIN + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise PilotProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PilotProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotProtocolError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise PilotProtocolError("JSON root must be an object")
    return value


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PilotProtocolError(f"required regular file cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PilotProtocolError(f"file is linked or nonregular: {path}")
        blocks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            blocks.append(block)
        payload = b"".join(blocks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        path_after = path.lstat()
    except FileNotFoundError as exc:
        raise PilotProtocolError(f"file disappeared while read: {path}") from exc
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_path = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
    )
    if identity_before != identity_after or identity_before != identity_path:
        raise PilotProtocolError(f"file changed while read: {path}")
    if len(payload) != before.st_size:
        raise PilotProtocolError(f"file length changed while read: {path}")
    return payload


def _load_canonical(path: Path) -> dict[str, Any]:
    payload = _read_regular_bytes(path)
    value = _strict_json(payload)
    if _canonical_json(value) != payload:
        raise PilotProtocolError(f"retained JSON is not canonical: {path}")
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        if set(left) != set(right):
            return False
        return all(_exact_equal(left[key], right[key]) for key in left)
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _bound_file_record(binding: tuple[str, str, int, str]) -> dict[str, object]:
    role, relative_path, expected_size, expected_sha256 = binding
    path = PROJECT_ROOT / relative_path
    payload = _read_regular_bytes(path)
    if len(payload) != expected_size:
        raise PilotProtocolError(f"bound file size differs: {relative_path}")
    if _sha256_bytes(payload) != expected_sha256:
        raise PilotProtocolError(f"bound file digest differs: {relative_path}")
    return {
        "role": role,
        "relative_path": relative_path,
        "bytes": expected_size,
        "file_sha256": expected_sha256,
    }


def _load_clone_module():
    # Compile the exact pinned bytes.  SourceFileLoader can otherwise execute
    # an adjacent timestamp-valid ``__pycache__`` entry that is not bound by
    # this protocol.
    role, relative_path, expected_size, expected_sha256 = CLONE_BINDINGS[0]
    if role != "implementation":
        raise PilotProtocolError("clone implementation binding role differs")
    payload = _read_regular_bytes(PROJECT_ROOT / relative_path)
    if len(payload) != expected_size or _sha256_bytes(payload) != expected_sha256:
        raise PilotProtocolError("clone implementation source binding differs")
    _bound_file_record(ACQUISITION_BINDINGS[0])
    name = "jx_tno_covariance_clone_materialization_v1_for_pilot"
    module = types.ModuleType(name)
    module.__file__ = str(CLONE_TOOL)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        code = compile(
            payload,
            str(CLONE_TOOL),
            "exec",
            dont_inherit=True,
            optimize=sys.flags.optimize,
        )
        exec(code, module.__dict__)
    except Exception as exc:
        sys.modules.pop(name, None)
        raise PilotProtocolError(
            "sealed clone verifier source failed to execute"
        ) from exc
    return module


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise PilotProtocolError(f"{label} differs from the frozen contract")


def _validate_and_rederive_inputs() -> dict[str, object]:
    acquisition_files = [_bound_file_record(row) for row in ACQUISITION_BINDINGS]
    clone_files = [_bound_file_record(row) for row in CLONE_BINDINGS]
    clones = _load_clone_module()
    try:
        custody = clones.ACQUISITION.validate_output(ACQUISITION_ROOT)
        retained_clone_protocol = _load_canonical(CLONE_ROOT / "PROTOCOL.json")
        rebuilt_clone_protocol = clones.build_protocol(ACQUISITION_ROOT)
        _require_exact(
            retained_clone_protocol,
            rebuilt_clone_protocol,
            "retained clone protocol reconstruction",
        )
        retained_clones = _load_canonical(CLONE_ROOT / "CLONES.json")
        rebuilt_clones = clones.build_clones(rebuilt_clone_protocol, ACQUISITION_ROOT)
        _require_exact(retained_clones, rebuilt_clones, "retained clone artifact reconstruction")
    except PilotProtocolError:
        raise
    except Exception as exc:
        raise PilotProtocolError("upstream custody or clone reconstruction failed") from exc

    _require_exact(custody.get("content_sha256"), ACQUISITION_CUSTODY_CONTENT_SHA256, "custody content digest")
    _require_exact(custody.get("raw_roster_sha256"), ACQUISITION_RAW_ROSTER_SHA256, "raw roster digest")
    _require_exact(custody.get("record_count"), 12, "custody record count")
    _require_exact(
        retained_clone_protocol.get("content_sha256"),
        CLONE_PROTOCOL_CONTENT_SHA256,
        "clone protocol content digest",
    )
    _require_exact(
        retained_clone_protocol.get("input", {}).get("protocol_content_sha256"),
        ACQUISITION_PROTOCOL_CONTENT_SHA256,
        "clone-to-acquisition protocol content binding",
    )
    _require_exact(
        retained_clones.get("content_sha256"),
        CLONE_ARTIFACT_CONTENT_SHA256,
        "clone artifact content digest",
    )
    _require_exact(
        retained_clones.get("binary_payload_sha256"),
        CLONE_BINARY_PAYLOAD_SHA256,
        "clone binary payload digest",
    )
    _require_exact(retained_clones.get("object_count"), 12, "clone object count")
    _require_exact(retained_clones.get("clones_per_object"), 128, "clones per object")
    _require_exact(retained_clones.get("total_clone_count"), 1_536, "total clone count")
    _require_exact(
        custody.get("classification"),
        "SBDB_SCHEMA_CONFORMANT_FULL_COVARIANCE_BYTES_INPUT_CUSTODY_ONLY",
        "covariance custody classification",
    )
    _require_exact(custody.get("claims", {}).get("provider_transport_authenticated"), False, "transport authentication claim")
    _require_exact(
        custody.get("claims", {}).get("authoritative_provider_origin_cryptographically_proven"),
        False,
        "provider-origin claim",
    )
    _require_exact(
        custody.get("claims", {}).get("sbdb_schema_conformant_raw_covariance_bytes_retained"),
        True,
        "schema-conformant custody claim",
    )
    _require_exact(retained_clones.get("execution_authorized"), False, "clone execution claim")
    _require_exact(retained_clones.get("outcomes_generated"), False, "clone outcome claim")

    actual_roster = [
        (
            row.get("object_id"),
            row.get("spk_id"),
            row.get("cohort"),
            row.get("covariance_epoch_jd_tdb"),
        )
        for row in retained_clones.get("objects", [])
        if type(row) is dict
    ]
    _require_exact(actual_roster, list(EXPECTED_OBJECTS), "clone object roster")

    # Close the ordinary concurrent-change window after all upstream code ran.
    _require_exact(
        [_bound_file_record(row) for row in ACQUISITION_BINDINGS],
        acquisition_files,
        "acquisition file bindings after reconstruction",
    )
    _require_exact(
        [_bound_file_record(row) for row in CLONE_BINDINGS],
        clone_files,
        "clone file bindings after reconstruction",
    )
    return {
        "acquisition": {
            "files": acquisition_files,
            "protocol_content_sha256": ACQUISITION_PROTOCOL_CONTENT_SHA256,
            "custody_content_sha256": ACQUISITION_CUSTODY_CONTENT_SHA256,
            "raw_roster_sha256": ACQUISITION_RAW_ROSTER_SHA256,
            "record_count": 12,
            "custody_classification": "SBDB_SCHEMA_CONFORMANT_FULL_COVARIANCE_BYTES_INPUT_CUSTODY_ONLY",
            "authentication_status": "SCHEMA_CONFORMANT_UNAUTHENTICATED_RETAINED_BYTES",
        },
        "clones": {
            "files": clone_files,
            "protocol_content_sha256": CLONE_PROTOCOL_CONTENT_SHA256,
            "artifact_content_sha256": CLONE_ARTIFACT_CONTENT_SHA256,
            "binary_payload_sha256": CLONE_BINARY_PAYLOAD_SHA256,
            "object_count": 12,
            "clones_per_object": 128,
            "total_clone_count": 1_536,
            "classification": "DETERMINISTIC_FULL_COVARIANCE_NATIVE_ELEMENT_CLONES_INPUT_ONLY",
        },
    }


def _object_roster() -> list[dict[str, object]]:
    return [
        {
            "object_id": object_id,
            "spk_id": spk_id,
            "cohort": cohort,
            "covariance_epoch_jd_tdb": epoch,
            "nominal_state_count": 1,
            "clone_state_count": 128,
        }
        for object_id, spk_id, cohort, epoch in EXPECTED_OBJECTS
    ]


def build_protocol() -> dict[str, object]:
    inputs = _validate_and_rederive_inputs()
    record: dict[str, object] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "status": "SEALED_PREPARE_ONLY_EXECUTION_BLOCKED",
        "claim_ceiling": "TWELVE_OBJECT_INPUT_QUALIFICATION_PROTOCOL_ONLY",
        "scope": {
            "purpose": "JX_12_OBJECT_COVARIANCE_DYNAMICAL_CLASSIFICATION_QUALIFICATION_PILOT",
            "object_count": 12,
            "detached_screen_count": 6,
            "scattering_control_count": 6,
            "final_36_object_science_sample": False,
            "population_study": False,
            "planet_nine_test": False,
            "object_roster": _object_roster(),
        },
        "inputs": inputs,
        "custody_interpretation": {
            "permitted_statement": "Retained raw covariance bytes passed the frozen SBDB schema and matrix checks.",
            "required_qualification": "The retained bytes are unauthenticated; cryptographic provider origin is not established.",
            "provider_transport_authenticated": False,
            "authoritative_provider_origin_cryptographically_proven": False,
            "clone_interpretation": "LOCAL_FORMAL_COVARIANCE_SENSITIVITY_DRAWS_NOT_POSTERIOR_SAMPLES_OR_LIKELIHOOD_DRAWS",
            "clone_count_limitation": "128_CLONES_PER_OBJECT_DOES_NOT_ESTABLISH_TAIL_PROBABILITIES_OR_POPULATION_PRECISION",
        },
        "known_planets_model": {
            "required_default_source_ids_in_exact_order": list(RESOLVED_SOURCE_IDS),
            "required_default_massive_source_count": 11,
            "tno_states_are_massless_and_noninteracting": True,
            "planet_nine_included": False,
            "stellar_encounters_included": False,
            "galactic_tide_included": False,
            "unqualified_five_body_or_inner_monopole_collapse_permitted": False,
            "reduction_exception": "ONLY_A_SEPARATELY_PREREGISTERED_AND_NUMERICALLY_QUALIFIED_CLASSIFICATION_EQUIVALENCE_CONTRACT_MAY_REPLACE_THE_RESOLVED_ELEVEN_BODY_SOURCE",
            "ephemeris_requirement": {
                "preferred": "DE441_NATIVE_STATES_AND_GM_VALUES_AT_EACH_OBJECT_NATIVE_COVARIANCE_EPOCH",
                "conditional_alternative": "DE440_STATES_ONLY_AFTER_A_PREREGISTERED_DE440_VERSUS_DE441_CLASSIFICATION_SENSITIVITY_BRIDGE_PASSES",
                "currently_satisfied": False,
            },
        },
        "stages": [
            {
                "stage": 0,
                "id": "NATIVE_EPOCH_TO_CARTESIAN_TRANSFORM_QUALIFICATION",
                "purpose": "QUALIFICATION_ONLY",
                "object_count": 12,
                "nominal_state_count": 12,
                "clone_state_count": 1_536,
                "total_tracer_state_count": 1_548,
                "required_input_basis_labels": ["e", "q", "tp", "node", "peri", "i"],
                "per_object_native_covariance_epoch_preserved": True,
                "required_semantics": [
                    "EXACT_INPUT_EPOCH_AND_TIME_SCALE",
                    "EXACT_ELEMENT_REFERENCE_FRAME_AND_AXES",
                    "EXACT_COORDINATE_ORIGIN_AND_CENTER",
                    "EXACT_CENTRAL_GM_AND_BODY_GM_SET",
                    "EXACT_TP_TO_CARTESIAN_PHASE_CONVENTION",
                    "EXACT_OUTPUT_FRAME_ORIGIN_UNITS_AND_TIME_SCALE",
                ],
                "input_semantics_to_bind": {
                    "covariance_epoch": "JULIAN_DATE_TDB",
                    "tp_coordinate": "ABSOLUTE_JULIAN_DATE_TDB_WITH_COVARIANCE_DIFFERENTIALS_IN_DAYS",
                    "element_frame": "HELIOCENTRIC_J2000_ECLIPTIC",
                    "binding_status": "REQUIRED_NOT_YET_IMPLEMENTED_OR_QUALIFIED",
                },
                "independent_transform_and_roundtrip_acceptance_budget_required": True,
                "implemented": False,
                "execution_authorized": False,
            },
            {
                "stage": 1,
                "id": "ONE_THOUSAND_YEAR_EIGHT_CLONE_PER_OBJECT_HOLDOUT",
                "purpose": "TRANSFORM_DYNAMICS_AND_CLASSIFIER_QUALIFICATION_ONLY",
                "horizon_julian_years": 1_000,
                "holdout_clone_indices_per_object": list(STAGE_1_HOLDOUT_CLONE_INDICES),
                "holdout_selection_rule": "NEAREST_INTEGER_OF_K_TIMES_127_OVER_7_FOR_K_0_THROUGH_7_FROZEN_BEFORE_TRANSFORM",
                "object_count": 12,
                "nominal_state_count": 12,
                "clone_state_count": 96,
                "total_tracer_state_count": 108,
                "primary_step_julian_years": 0.0625,
                "refinement_step_julian_years": 0.03125,
                "independent_reference_required": True,
                "science_output_permitted": False,
                "execution_authorized": False,
            },
            {
                "stage": 2,
                "id": "TEN_MILLION_YEAR_FULL_COVARIANCE_CLASSIFICATION_PILOT",
                "horizon_julian_years": 10_000_000,
                "object_count": 12,
                "nominal_state_count": 12,
                "clone_state_count": 1_536,
                "total_tracer_state_count": 1_548,
                "primary_step_julian_years": 0.5,
                "refinement_step_julian_years": 0.25,
                "known_planets_only": True,
                "hybrid_close_encounter_handling_required": True,
                "classifier_and_resonance_rules_must_be_frozen_before_execution": True,
                "independent_numerical_budget_must_be_frozen_before_execution": True,
                "execution_authorized": False,
            },
            {
                "stage": 3,
                "id": "MECHANICAL_ONE_HUNDRED_MILLION_YEAR_EXTENSION",
                "target_total_horizon_julian_years": 100_000_000,
                "additional_horizon_after_stage_2_julian_years": 90_000_000,
                "trigger_labels": [
                    "MIXED_UNDER_ORBIT_COVARIANCE",
                    "BORDERLINE_NUMERICAL_CLASSIFICATION",
                ],
                "selection_rule": "EXTEND_EVERY_AND_ONLY_STAGE_2_OBJECT_WITH_A_TRIGGER_LABEL_WITHOUT_POST_HOC_DISCRETION",
                "states_per_triggered_object": 129,
                "minimum_triggered_object_count": 0,
                "maximum_triggered_object_count": 12,
                "maximum_tracer_state_count": 1_548,
                "primary_step_julian_years": 0.5,
                "refinement_step_julian_years": 0.25,
                "known_planets_only": True,
                "hybrid_close_encounter_handling_required": True,
                "stage_2_label_must_remain_reported": True,
                "execution_authorized": False,
            },
        ],
        "classification_decision_rule": {
            "clone_denominator": 128,
            "minimum_agreeing_clones_for_robust_class": 123,
            "minimum_fraction_exact": "123/128",
            "minimum_fraction_decimal": "0.9609375",
            "nominal_must_equal_clone_consensus_class": True,
            "robust_rule": "ROBUST_CLASS_C_IFF_NOMINAL_CLASS_EQUALS_C_AND_AT_LEAST_123_OF_128_CLONES_EQUAL_C",
            "covariance_disagreement_label": "MIXED_UNDER_ORBIT_COVARIANCE",
            "covariance_disagreement_rule": "IF_NO_CLASS_SATISFIES_THE_ROBUST_RULE_AND_NUMERICAL_GATES_PASS",
            "numerical_borderline_label": "BORDERLINE_NUMERICAL_CLASSIFICATION",
            "numerical_borderline_rule": "IF_ANY_FROZEN_PRIMARY_REFINEMENT_INDEPENDENT_OR_CLASSIFIER_BOUNDARY_GATE_FAILS",
            "no_forced_secure_class": True,
        },
        "execution_gate": {
            "all_blockers_must_be_closed_by_content_addressed_reviewed_artifacts": True,
            "blockers": list(BLOCKERS),
            "ephemeris_mismatch_caution": "THE_RETAINED_SBDB_ORBIT_FITS_DECLARE_DE441; A_PLANNED_DE440_OR_DE440S_MASSIVE_BODY_SOURCE_IS_A_MODEL_MISMATCH_AND_REQUIRES_THE_PREREGISTERED_DE440_DE441_SENSITIVITY_BRIDGE",
            "external_preregistration_requirement": "A_FUTURE_EXECUTION_MAY_BE_CALLED_PREREGISTERED_ONLY_AFTER_THIS_PROTOCOL_AND_ALL_SUCCESSOR_GATE_ARTIFACTS_HAVE_AN_EXTERNAL_IMMUTABLE_COMMIT_AND_TRUSTED_TIMESTAMP_CREATED_BEFORE_EXECUTION",
            "external_immutable_commit_present": False,
            "trusted_preexecution_timestamp_present": False,
            "runtime_identity_must_bind": [
                "PROTOCOL_AND_ALL_INPUT_HASHES",
                "SOURCE_TREE_OR_INSTALL_ARTIFACT_HASHES",
                "PYTHON_AND_NATIVE_DEPENDENCY_VERSIONS",
                "OPERATING_SYSTEM_AND_ARCHITECTURE",
                "CPU_OR_GPU_MODEL_AND_DEVICE_IDENTITY",
                "NUMERICAL_BACKEND_AND_PRECISION_MODE",
            ],
            "execution_authorized": False,
        },
        "references": [
            {
                "title": "SBDB API",
                "publisher": "NASA/JPL Solar System Dynamics",
                "url": "https://ssd-api.jpl.nasa.gov/doc/sbdb.html",
                "role": "Official schema documentation for the retained covariance response; not proof of transport or provider origin.",
            },
            {
                "authors": "Bernardinelli, P. H.; Bernstein, G. M.; Sako, M.; et al.",
                "year": 2022,
                "title": "A Search of the Full Six Years of the Dark Energy Survey for Outer Solar System Objects",
                "journal": "The Astrophysical Journal Supplement Series",
                "volume": 258,
                "issue": 2,
                "article": 41,
                "doi": "10.3847/1538-4365/ac3914",
                "role": "Published DES TNO catalog and clone-based dynamical-classification context; not a claim that this pilot reproduces DES.",
            },
        ],
        "implementation": {
            "relative_path": "benchmarks/jx_tno_covariance_classification_pilot_v1.py",
            "file_sha256": _sha256_bytes(_read_regular_bytes(SCRIPT_PATH)),
            "cli_surface": "PREPARE_ONLY",
            "dynamics_imported": False,
        },
        "claims": dict(CLAIMS),
        "preparation_only": True,
        "execution_authorized": False,
        "outcomes_generated": False,
        "nonclaim": "This protocol is not an orbit transform, simulation, dynamical classification, 36-object result, population result, or Planet Nine result.",
    }
    record["content_sha256"] = _content_sha256(record)
    return record


def validate_protocol(value: Mapping[str, Any]) -> str:
    expected = build_protocol()
    if type(value) is not dict or not _exact_equal(value, expected):
        raise PilotProtocolError("protocol differs from exact frozen reconstruction")
    digest = _content_sha256(value)
    _require_exact(value.get("content_sha256"), digest, "protocol content digest")
    for key in (
        "pilot_execution_authorized",
        "pilot_executed",
        "dynamical_classifications_generated",
        "population_inference",
        "planet_nine_evidence",
        "science_result",
    ):
        _require_exact(value.get("claims", {}).get(key), False, f"claim {key}")
    _require_exact(value.get("execution_authorized"), False, "top-level execution claim")
    _require_exact(value.get("outcomes_generated"), False, "top-level outcome claim")
    return digest


def _write_create_only(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as exc:
        raise PilotProtocolError(f"protocol path is not create-only: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def prepare(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    parent = output.parent
    if parent.is_symlink():
        raise PilotProtocolError("output directory cannot be a symlink")
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir() or parent.is_symlink():
        raise PilotProtocolError("output directory must be a real directory")
    if any(parent.iterdir()):
        raise PilotProtocolError("dedicated output directory is not empty")
    if output.exists() or output.is_symlink():
        raise PilotProtocolError("protocol already exists")
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    _write_create_only(output, payload)
    retained = _load_canonical(output)
    validate_protocol(retained)
    return protocol


def validate_file(path: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    value = _load_canonical(path)
    validate_protocol(value)
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="verify/rederive sealed inputs and create the protocol; never run dynamics",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.prepare_only:
        parser.error("--prepare-only is required; this tool has no execution mode")
    try:
        protocol = prepare(args.output)
    except PilotProtocolError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
