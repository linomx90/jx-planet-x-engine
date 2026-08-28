from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


sys.dont_write_bytecode = True

EXPERIMENT_ID = "jx-o2-characterized-survey-model-comparison-design-v1"
PROTOCOL_ID = "jx-o2-g0-open-regeneration-prereg-v1"
CLAIM_CEILING = "OPEN_REGENERATION_PROTOCOL_ONLY"
PACKAGE_STATUS = "JX_O2_OPEN_REGEN_PREREG_VERIFIED_DESIGN_ONLY_BLOCKED"
REGISTRATION_SCHEMA = "jx-o2-g0-open-regeneration-registration/v1"

EXPECTED_FILES = {
    "README.md",
    "open_regeneration_preregistration_v1.json",
    "model_family_manifest_v1.json",
    "randomization_and_numerics_v1.json",
    "analysis_gate_protocol_v1.json",
    "prior_bindings_v1.json",
    "verify_preregistration.py",
    "test_preregistration.py",
    "registration_v1.json",
}
LOCKED_FILES = EXPECTED_FILES - {"registration_v1.json"}
ARTIFACT_FILES = {
    "main": "open_regeneration_preregistration_v1.json",
    "model": "model_family_manifest_v1.json",
    "randomization": "randomization_and_numerics_v1.json",
    "analysis": "analysis_gate_protocol_v1.json",
    "priors": "prior_bindings_v1.json",
}

# These recursive JSON-shape digests are filled only after the package structure
# is frozen. They are an additionalProperties=false equivalent at every depth.
EXPECTED_SHAPE_SHA256 = {
    "main": "39bc5ca6b0e11b39e21da571405e3b981d969871de6ff9c60daa6414cf8c549d",
    "model": "cad72526c5edbc02e53074f9e54c95e40c9a05affe2318322f68b55cda59bd9b",
    "randomization": "a3c187e4bd8fda7842a961bd206c3883ef5a81ddf347fa4d299d049159020a16",
    "analysis": "ed5df50a1e867ac439461baeb7ee86029f24f2e7806e93de07c8b1ba24de975e",
    "priors": "074b996bf0157682ce8d430f234b4711deb588aa6eb3bb4acbd5bfb91af473c7",
    "registration": "08ecca056154575c2f9b44c671b073c70b7e06309e84883ea16f03c33e7eb14b",
}

EXPECTED_REQUIREMENT_STATUSES = {
    **{f"OSSOS-A{index:02d}": "NOT_SATISFIED" for index in range(1, 9)},
    **{f"DES-A{index:02d}": "NOT_SATISFIED" for index in range(1, 9)},
    **{f"MODEL-A{index:02d}": "NOT_SATISFIED" for index in range(1, 16)},
}
for _requirement in ("OSSOS-A01", "OSSOS-A03", "OSSOS-A06", "OSSOS-A07"):
    EXPECTED_REQUIREMENT_STATUSES[_requirement] = "AWAITING_AUTHORITY"
EXPECTED_REQUIREMENT_STATUSES["DES-A06"] = "AWAITING_AUTHORITY"
EXPECTED_REQUIREMENT_STATUSES["MODEL-A01"] = "AWAITING_AUTHORITY_OR_OPEN_REGENERATION"
EXPECTED_REQUIREMENT_STATUSES["MODEL-A04"] = "AWAITING_AUTHORITY_OR_OPEN_REGENERATION"

EXPECTED_PHYSICAL_ROWS = (
    ("P01", "5/1", "367/1", "1/5", "20/1", "1/9"),
    ("P02", "5/1", "420/1", "7/20", "20/1", "1/9"),
    ("P03", "5/1", "480/1", "1/2", "20/1", "1/9"),
    ("P04", "707/100", "356/1", "1/5", "20/1", "1/9"),
    ("P05", "707/100", "433/1", "7/20", "20/1", "1/9"),
    ("P06", "707/100", "497/1", "1/2", "20/1", "1/9"),
    ("P07", "10/1", "356/1", "1/5", "20/1", "1/9"),
    ("P08", "10/1", "433/1", "7/20", "20/1", "1/9"),
    ("P09", "10/1", "540/1", "1/2", "20/1", "1/9"),
)

EXPECTED_PERMISSION_ACTIONS = [
    "GENERATE_SOURCE_POPULATION",
    "GENERATE_OR_SELECT_CHECKPOINT",
    "MATERIALIZE_COMPACT_BODY_STATES",
    "REALIZE_RANDOM_SEEDS",
    "RUN_PILOT_OR_DRY_RUN",
    "RUN_BENCHMARK_OR_PREFLIGHT_DYNAMICS",
    "RUN_SURVEY_ADAPTER",
    "RUN_SYNTHETIC_CALIBRATION",
    "ACCESS_OR_UNBLIND_HOLDOUT",
    "COMPUTE_OBSERVED_STATISTIC",
    "RUN_OBSERVED_MODEL_COMPARISON",
    "SUBMIT_CPU_OR_GPU_JOB",
    "CREATE_G1_EXECUTION_CONTRACT",
    "CREATE_ACTIVATION_RECEIPT",
    "MAKE_PLANET_X_CLAIM",
]

EXPECTED_SEED_PROTOCOL_COMMON = {
    "hash_algorithm": "SHA256",
    "child_formula": "SHA256(MASTER || LENGTH_PREFIXED_STREAM_LABEL || UINT64_BIG_ENDIAN_INDICES || LENGTH_PREFIXED_KEY_OR_COUNTER_TAG)",
    "string_encoding": "STRICT_ASCII",
    "field_encoding": "UINT32_BIG_ENDIAN_LENGTH_PREFIX_FOLLOWED_BY_EXACT_FIELD_BYTES",
    "counter_encoding": "UINT64_BIG_ENDIAN",
    "beacon_output_encoding": "LOWERCASE_HEX_VALIDATED_THEN_DECODED_TO_BYTES",
    "digest_to_rng_state": "KEY_USES_FIRST_16_BYTES_OF_KEY_CHILD; COUNTER_USES_ALL_32_BYTES_OF_COUNTER_CHILD; BOTH_BIG_ENDIAN",
    "rng_family": "PHILOX4X64_10",
    "rng_reference_implementation": "NUMPY_PHILOX_IN_NUMPY_2_3_5_WITH_EXACT_WRAPPER_AND_NATIVE_HASHES_REQUIRED",
    "caller_seed_override": False,
    "system_time_pid_hostname_or_unregistered_entropy": False,
    "seed_shopping": False,
}
EXPECTED_INPUT_SEED_DERIVATION = {
    "status": "FORMULA_AND_BEACON_SELECTION_RULE_LOCKED_VALUE_NOT_YET_AVAILABLE",
    "contract_identity_requirement": "EXACT_INPUT_GENERATION_CONTRACT_SCHEMA_ID_RAW_SHA256_AND_EXTERNAL_TIMESTAMP_REQUIRED",
    "domain_separator_ascii": "jx-o2-open-regeneration-input-v1",
    "beacon_source": "NIST_RANDOMNESS_BEACON_2_0",
    "beacon_event_rule": "FIRST_VALID_PULSE_WITH_TIMESTAMP_AT_LEAST_86400_SECONDS_AFTER_THE_EXTERNAL_INPUT_GENERATION_CONTRACT_TIMESTAMP",
    "beacon_fallback_rule": "STOP_WITHOUT_INPUT_SEEDS_IF_THE_SELECTED_SOURCE_HAS_NO_VALID_PULSE_WITHIN_7_DAYS; NO_ALTERNATE_SOURCE_OR_LATER_FAVORABLE_PULSE",
    "master_formula": "SHA256(LENGTH_PREFIXED_INPUT_DOMAIN_SEPARATOR || RAW_32_BYTE_INPUT_GENERATION_CONTRACT_SHA256 || SHA256(EXACT_SIGNED_INPUT_BEACON_RECORD_BYTES))",
    "may_be_realized_before_external_input_generation_contract_registration": False,
}
EXPECTED_ANALYSIS_SEED_DERIVATION = {
    "status": "FORMULA_AND_BEACON_SELECTION_RULE_LOCKED_VALUE_NOT_YET_AVAILABLE",
    "contract_identity_requirement": "EXACT_G1_EXECUTION_CONTRACT_SCHEMA_ID_RAW_SHA256_AND_EXTERNAL_TIMESTAMP_REQUIRED",
    "domain_separator_ascii": "jx-o2-open-regeneration-analysis-v1",
    "beacon_source": "NIST_RANDOMNESS_BEACON_2_0",
    "beacon_event_rule": "FIRST_VALID_PULSE_WITH_TIMESTAMP_AT_LEAST_86400_SECONDS_AFTER_THE_EXTERNAL_G1_EXECUTION_CONTRACT_TIMESTAMP",
    "beacon_fallback_rule": "STOP_WITHOUT_ANALYSIS_SEEDS_IF_THE_SELECTED_SOURCE_HAS_NO_VALID_PULSE_WITHIN_7_DAYS; NO_ALTERNATE_SOURCE_OR_LATER_FAVORABLE_PULSE",
    "master_formula": "SHA256(LENGTH_PREFIXED_ANALYSIS_DOMAIN_SEPARATOR || RAW_32_BYTE_G1_EXECUTION_CONTRACT_SHA256 || SHA256(EXACT_SIGNED_ANALYSIS_BEACON_RECORD_BYTES))",
    "may_be_realized_before_g0_completion_and_external_g1_registration": False,
}
EXPECTED_PRIMARY_SCORE_FORMULA = (
    "T_EQUALS_2_TIMES_OPEN_PAREN_LOG_PREDICTIVE_DENSITY_D_GIVEN_EQUAL_WEIGHT_M1_MIXTURE_"
    "MINUS_LOG_PREDICTIVE_DENSITY_D_GIVEN_M0_CLOSE_PAREN"
)
EXPECTED_CURRENT_STATEMENT = (
    "A_LOCAL_OPEN_REGENERATION_PROTOCOL_HAS_BEEN_CONTENT_HASHED_BUT_IS_NOT_EXTERNALLY_"
    "PREREGISTERED_OR_EXECUTABLE"
)

EXPECTED_PRIOR_BINDINGS = {
    "JX_O2_DESIGN_CONTRACT": (
        "work/jx-public-reconstruction-design/runs/planet_x_survey_model_comparison_v1/design_contract_v1.json",
        "ccd9631097a403d374ca1d6954ef32751027b1b3b755d5de7065ccce8017c971",
        "GOVERNING_EXPERIMENT_AND_STAGE_CONTRACT",
    ),
    "JX_O2_SOURCE_MODEL_MANIFEST": (
        "work/jx-public-reconstruction-design/runs/planet_x_survey_model_comparison_v1/source_models_manifest_v1.json",
        "d723c2dede9dee4bd490751ca4b5ad334397ecf8e913094039ddb8d624b6ac73",
        "GOVERNING_MODEL_INPUT_REQUIREMENTS",
    ),
    "JX_O2_SURVEY_INPUT_MANIFEST": (
        "work/jx-public-reconstruction-design/runs/planet_x_survey_model_comparison_v1/survey_inputs_manifest_v1.json",
        "d446ac5049add3409a6b5c0d0f3cd76ab1a1d7c9f1676f98069c1d9f5e42e3c4",
        "GOVERNING_SURVEY_INPUT_REQUIREMENTS",
    ),
    "JX_O2_DESIGN_REGISTRATION": (
        "work/jx-public-reconstruction-design/runs/planet_x_survey_model_comparison_v1/registration_design_v1.json",
        "d2978c56c492285d7e9f5cef6c9a8e4bdb927dca7c7a08699b4b45d1e8b2aa01",
        "GOVERNING_DESIGN_REGISTRATION",
    ),
    "JX_O2_G0_AUDIT": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_input_audit_v1/g0_audit_v1.json",
        "5f7a5f9cef4f6bd73f7c36235e610d96c8475bd646bf5807aa591c58fbf002b0",
        "GOVERNING_31_REQUIREMENT_G0_BASELINE",
    ),
    "JX_O2_G0_REGISTRATION": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_input_audit_v1/registration_g0_v1.json",
        "21634c974ffdff0f59b1f7b33c80bdabdfe9fc337f3476bf6f20be3ede00bd03",
        "GOVERNING_G0_REGISTRATION",
    ),
    "JX_O2_G0_SOURCE_INVENTORY": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_input_audit_v1/source_model_inventory_v1.json",
        "0ac5ef9bc1fbf84313a9ee5a8650a3685cba3d2d06eea1d29a547c7eda59da9f",
        "SOURCE_AVAILABILITY_AND_PROVENANCE_BOUNDARY",
    ),
    "PUBLIC_RECONSTRUCTION_DESIGN": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_public_reconstruction_design_v1/reconstruction_design_v1.json",
        "bf0035fe7595536b5c1fd96b447298654d2892be11426e6ea9774b2585aa9ac6",
        "PUBLIC_RECONSTRUCTION_BOUNDARY_AND_RESOLUTION_LANES",
    ),
    "PUBLIC_LITERATURE_GRID": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_public_reconstruction_design_v1/literature_prior_grid_v1.json",
        "b6fe87043937764976b26473b49004eecfefb2882178f5f41ccb98424534068c",
        "AUDITED_LITERATURE_INPUT_GRID_AND_EXPOSURE_RECORD",
    ),
    "PUBLIC_RECONSTRUCTION_REGISTRATION": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_public_reconstruction_design_v1/registration_reconstruction_design_v1.json",
        "de8295e0070745f251acc669d225e79fca7ae9f61e17af65c924eb24e02d9503",
        "PUBLIC_RECONSTRUCTION_REGISTRATION",
    ),
    "CANDIDATE_9118_RETIREMENT_RECEIPT": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_candidate_9118_retirement_v1/candidate_9118_retirement_v1.json",
        "01a041382aa716293db9f61ae8917d44bd80c62347cc2aaa7c03d0442f6009cd",
        "HISTORICAL_REFERENCE_ONLY_PERMANENT_NONUSE_POLICY",
    ),
    "CANDIDATE_9118_RETIREMENT_REGISTRATION": (
        "work/jx-public-reconstruction-design/audits/jx_o2_g0_candidate_9118_retirement_v1/registration_retirement_v1.json",
        "037c22ea0d1a9cb4c86eadc9056b8e1e605a7a5790cba60871daaa75aad1bbc5",
        "HIGHEST_PRECEDENCE_RETIREMENT_REGISTRATION",
    ),
    "LOCAL_G0_ACQUISITION_CHECKLIST": (
        "work/jx_o2_g0_local_acquisition_v1/acquisition_checklist_v1.json",
        "31f9dd113d291f5d8c0c573450b7d6e6384a76f8abebffe8fa205561767e2280",
        "CURRENT_31_REQUIREMENT_LOCAL_STATUS",
    ),
    "LOCAL_G0_CUSTODY_MANIFEST": (
        "work/jx_o2_g0_local_acquisition_v1/local_custody_manifest_v1.json",
        "defc6413a76f56f6320440412a6f663513d5e892a24caeca75611dd141ebcd73",
        "QUARANTINED_LOCAL_PUBLIC_BYTE_CUSTODY",
    ),
    "LOCAL_G0_ACQUISITION_REGISTRATION": (
        "work/jx_o2_g0_local_acquisition_v1/registration_v1.json",
        "ad94637de5b2062213b8d86726b15f256ae3a14522382ad0194e95b28ae337d7",
        "LOCAL_CUSTODY_CONTENT_HASH_ANCHOR",
    ),
    "JX_E2_LOCAL_CLOSURE": (
        "work/jx_e2_numerics/closure_v1.json",
        "4cec842a5f5b6f22a97f0b1b5dfaaf4bcddfe7efbe5c6836bf3772ef7724d901",
        "EXCLUDED_ENGINEERING_CONTEXT_NOT_MODEL_INPUT",
    ),
}

RETIREMENT_BINDING_IDS = {
    "CANDIDATE_9118_RETIREMENT_RECEIPT",
    "CANDIDATE_9118_RETIREMENT_REGISTRATION",
}
RETIREMENT_FORBIDDEN_STRINGS = {
    "p9_bb21_idx9118",
    "candidate_9118",
    "de441_source_9118_state.csv",
    "81a8cf50f2ce6d17e90369efcaa82cf82a0955665851fae0207e4c6cfae4b6cf",
    "509917e0093107464d9ee45ed2c8e9f403403b2bb0e94455fa3614825917f8b0",
    "050b68182ecbf7fd76f280d8cc43c0683d207499aba1488a68092b834420a422",
}
RETIREMENT_ORBIT = {
    Fraction(253, 50),
    Fraction(49519, 100),
    Fraction(59, 250),
    Fraction(507, 25),
    Fraction(28417, 100),
    Fraction(9687, 100),
    Fraction(12621, 100),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_float(value: str) -> None:
    raise ValueError(f"JSON floating-point values are forbidden; use an exact rational string: {value}")


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
        parse_float=reject_float,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


JSON_SCALAR_KIND = {type(None): "null", bool: "boolean", int: "integer", str: "string"}


def json_shape_nodes(value: Any, path: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    if type(value) is dict:
        nodes.append({"path": list(path), "kind": "object", "keys": sorted(value)})
        for key in sorted(value):
            nodes.extend(json_shape_nodes(value[key], (*path, key)))
    elif type(value) is list:
        nodes.append({"path": list(path), "kind": "array", "length": len(value)})
        for index, child in enumerate(value):
            nodes.extend(json_shape_nodes(child, (*path, index)))
    else:
        if type(value) not in JSON_SCALAR_KIND:
            raise TypeError(f"unexpected JSON scalar type: {type(value).__name__}")
        nodes.append({"path": list(path), "kind": JSON_SCALAR_KIND[type(value)]})
    return nodes


def json_shape_sha256(value: Any) -> str:
    return canonical_json_sha256(json_shape_nodes(value))


def canonical_regular(base: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or str(rel) != rel.as_posix():
        raise ValueError("path is not canonical workspace-relative POSIX text")
    candidate = base / rel
    if candidate.absolute() != candidate.resolve() or candidate.is_symlink():
        raise ValueError("path traverses a symlink or alias")
    mode = candidate.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("bound path is not a regular file")
    return candidate


def validate_inventory(root: Path) -> None:
    actual: set[str] = set()
    for entry in root.iterdir():
        if entry.is_symlink() or not stat.S_ISREG(entry.stat(follow_symlinks=False).st_mode):
            raise ValueError(f"non-regular or nested package entry: {entry.name}")
        if entry.stat().st_nlink != 1:
            raise ValueError(f"hard-linked package entry: {entry.name}")
        actual.add(entry.name)
    if actual != EXPECTED_FILES:
        raise ValueError(f"package file inventory changed: {sorted(actual ^ EXPECTED_FILES)}")


def validate_registration(root: Path, artifacts: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], str]:
    path = root / "registration_v1.json"
    registration = strict_json(path)
    if json_shape_sha256(registration) != EXPECTED_SHAPE_SHA256["registration"]:
        raise ValueError("registration recursive JSON shape changed")
    expected_permissions = {
        "local_hash_verification_authorized": True,
        "network_or_github_action_authorized": False,
        "generate_source_population_authorized": False,
        "generate_checkpoint_authorized": False,
        "realize_angles_or_seeds_authorized": False,
        "dynamics_or_pilot_authorized": False,
        "survey_adapter_authorized": False,
        "synthetic_calibration_authorized": False,
        "observed_data_access_authorized": False,
        "observed_model_comparison_authorized": False,
        "cpu_or_gpu_job_authorized": False,
        "g1_contract_authorized": False,
        "activation_receipt_authorized": False,
        "planet_x_claim_authorized": False,
    }
    if (
        registration["schema"] != REGISTRATION_SCHEMA
        or registration["artifact_id"] != "jx-o2-g0-open-regeneration-registration-v1"
        or registration["experiment_id"] != EXPERIMENT_ID
        or registration["protocol_id"] != PROTOCOL_ID
        or registration["registration_state"] != "LOCAL_DESIGN_HASH_LOCKED_NOT_EXTERNALLY_PREREGISTERED"
        or registration["timestamp_authority"] != "LOCAL_CONTENT_HASH_ONLY_NO_EXTERNAL_TIMESTAMP"
        or registration["externally_timestamped"] is not False
        or registration["scientific_evidence_artifact"] is not False
        or registration["outcomes_generated"] is not False
        or registration["g0_complete"] is not False
        or registration["eligible_for_g1"] is not False
        or registration["claim_ceiling"] != CLAIM_CEILING
        or registration["execution_permissions"] != expected_permissions
        or registration["mandatory_nonclaim"] != artifacts["main"]["mandatory_nonclaim"]
        or not isinstance(registration["recorded_at_utc"], str)
        or not registration["recorded_at_utc"].endswith("Z")
    ):
        raise ValueError("registration identity, status, or permission boundary changed")
    if set(registration["locked_files"]) != LOCKED_FILES:
        raise ValueError("registration locked-file set changed")
    for relative, expected in registration["locked_files"].items():
        locked = canonical_regular(root, relative)
        if sha256_file(locked) != expected:
            raise RuntimeError(f"registered file changed: {relative}")
    return registration, sha256_file(path)


def require_identity(artifact: dict[str, Any], schema: str, artifact_class: str) -> None:
    if (
        artifact["schema"] != schema
        or artifact["experiment_id"] != EXPERIMENT_ID
        or artifact["protocol_id"] != PROTOCOL_ID
        or artifact["artifact_class"] != artifact_class
        or artifact["claim_ceiling"] != CLAIM_CEILING
        or artifact["execution_authorized"] is not False
    ):
        raise ValueError(f"artifact identity or claim boundary changed: {schema}")


def validate_main(main: dict[str, Any]) -> None:
    require_identity(main, "jx-o2-g0-open-regeneration-preregistration/v1", "LOCAL_DESIGN_ONLY_OPEN_REGENERATION_PROTOCOL")
    required_false = (
        "externally_timestamped",
        "scientific_evidence_artifact",
        "outcomes_generated",
        "checkpoint_generated",
        "compact_body_states_realized",
        "seed_values_realized",
        "g0_complete",
        "eligible_for_g1",
        "execution_authorized",
        "observed_execution_authorized",
        "gpu_run_authorized",
    )
    if any(main[key] is not False for key in required_false):
        raise ValueError("main fail-closed status changed")
    if (
        main["artifact_state"] != "LOCAL_DESIGN_HASH_LOCKED_NOT_EXTERNALLY_PREREGISTERED"
        or main["timestamp_authority"] != "LOCAL_CLOCK_ONLY_NO_EXTERNAL_TIMESTAMP"
        or main["scientific_evidence_role"] != "DESIGN_AND_GATE_SPECIFICATION_NOT_MODEL_OR_OUTCOME_EVIDENCE"
        or main["scientific_objective"]["family_class"] != "INDEPENDENT_PUBLIC_MODEL_FAMILY"
        or main["scientific_objective"]["model_label"] != "CLUSTER2_LIKE_NEW_MODEL"
        or main["primary_lane"]["lane_id"] != "OPEN_REGENERATION_POST_CHECKPOINT_INTERVENTION"
        or main["primary_lane"]["checkpoint_time_myr_from_generator_time_zero"] != 300
        or main["primary_lane"]["post_insertion_integration_duration_gyr"] != 4
        or main["generator_boundary"]["production_history_count"] != 128
        or main["generator_boundary"]["disk_draws_shared_between_histories"] is not False
        or main["generator_boundary"]["stellar_schedules_shared_between_histories"] is not False
        or main["g0_disposition"]["all_requirement_count"] != 31
        or main["g0_disposition"]["requirements_semantically_verified_by_this_artifact"] != 0
        or main["g0_disposition"]["this_artifact_completes_g0"] is not False
    ):
        raise ValueError("main scientific boundary changed")
    if main["permission_ontology"]["state_changing_actions"] != EXPECTED_PERMISSION_ACTIONS:
        raise ValueError("closed permission action ontology changed")
    if main["permission_ontology"]["missing_or_unmapped_action"] != "DENY":
        raise ValueError("unmapped permission action is not denied")
    permissions = main["current_permissions"]
    if permissions["local_hash_verification"] is not True:
        raise ValueError("local verification permission changed")
    for key, value in permissions.items():
        if key != "local_hash_verification" and value is not False:
            raise ValueError(f"state-changing permission enabled: {key}")
    pairing = main["pairing_invariants"]
    if pairing["only_registered_additional_body_may_physically_differ"] is not True:
        raise ValueError("paired-arm physical difference invariant changed")
    if not all(value is True for value in pairing.values()):
        raise ValueError("paired-arm invariant disabled")


def validate_model(model: dict[str, Any]) -> None:
    require_identity(model, "jx-o2-open-regeneration-model-family/v1", "DESIGN_ONLY_FINITE_MODEL_FAMILY_MANIFEST")
    if model["outcomes_generated"] is not False or model["manifest_state"] != "SUPPORT_AND_PAIRING_POLICY_LOCKED_STATES_NOT_REALIZED":
        raise ValueError("model manifest state changed")
    rows = model["physical_support"]["rows"]
    actual = tuple(
        (row["member_id"], row["mass_earth"], row["a_au"], row["e"], row["i_deg"], row["weight"])
        for row in rows
    )
    if actual != EXPECTED_PHYSICAL_ROWS or sum(Fraction(row[5]) for row in actual) != 1:
        raise ValueError("nine-row physical support or exact weights changed")
    angular = model["angular_support"]
    if (
        angular["sample_count"] != 128
        or Fraction(angular["weight_per_realized_angle"]) != Fraction(1, 128)
        or angular["angle_values_realized"] is not False
        or angular["zero_fill_for_missing_angles"] is not False
        or angular["same_angle_triple_used_for_all_nine_physical_rows_within_history"] is not True
    ):
        raise ValueError("angular nuisance design changed")
    joint = model["joint_support"]
    if joint["count_after_seed_realization"] != 1152 or Fraction(joint["each_weight"]) != Fraction(1, 1152):
        raise ValueError("joint finite support changed")
    history = model["history_ensemble_and_assignment"]
    if (
        history["final_history_count"] != 128
        or history["convergence_block_count"] != 4
        or history["independent_joint_history_count_per_block"] != 32
        or history["independent_disk_draw_count"] != 128
        or history["independent_cluster_and_field_star_schedule_count"] != 128
        or "NO_CROSS_REUSE" not in history["assignment_rule"]
        or history["m0_run_count"] != 128
        or history["m1_run_count"] != 1152
        or history["history_count_realized"] is not False
        or history["outcome_based_history_selection"] is not False
        or history["histories_may_be_treated_as_independent_joint_draws"] is not True
        or history["convergence_blocks_may_be_treated_as_additional_independent_replicates"] is not False
        or history["inference_unit"] != "HISTORY"
    ):
        raise ValueError("paired history design changed")
    retired = model["retired_candidate_policy"]
    if retired != {
        "governing_retirement_registration": "037c22ea0d1a9cb4c86eadc9056b8e1e605a7a5790cba60871daaa75aad1bbc5",
        "eligible_as_member_source_angle_phase_weight_or_prior_influence": False,
        "partial_or_transformed_reuse_allowed": False,
        "automated_literal_scan_proves_absence_of_cartesian_or_encoded_derivatives": False,
        "independent_transformation_and_lineage_nonuse_audit_required_before_g0_resolution": True,
    }:
        raise ValueError("retired-candidate nonuse policy changed")


def validate_randomization(randomization: dict[str, Any]) -> None:
    require_identity(
        randomization,
        "jx-o2-open-regeneration-randomization-numerics/v1",
        "DESIGN_ONLY_RANDOMIZATION_AND_NUMERICAL_GATE_PROTOCOL",
    )
    if randomization["seed_values_realized"] is not False or randomization["numerical_outputs_generated"] is not False:
        raise ValueError("randomization artifact claims realized values")
    if randomization["seed_protocol_common"] != EXPECTED_SEED_PROTOCOL_COMMON:
        raise ValueError("common seed encoding or RNG policy changed")
    if randomization["input_seed_derivation"] != EXPECTED_INPUT_SEED_DERIVATION:
        raise ValueError("input-generation seed master or beacon policy changed")
    if randomization["analysis_seed_derivation"] != EXPECTED_ANALYSIS_SEED_DERIVATION:
        raise ValueError("G1 analysis seed master or beacon policy changed")
    namespace = randomization["stream_namespace"]
    input_labels = namespace["input_master_allowed_labels"]
    analysis_labels = namespace["analysis_master_allowed_labels"]
    if len(input_labels) != len(set(input_labels)) or len(analysis_labels) != len(set(analysis_labels)) or set(input_labels) & set(analysis_labels):
        raise ValueError("seed stream namespaces are duplicated or cross-master ambiguous")
    if namespace["unmapped_stream"] != "DENY" or namespace["cross_master_label_reuse"] != "INVALID":
        raise ValueError("unmapped or cross-master seed stream was allowed")
    balance = randomization["history_and_member_balance"]
    if (
        balance["production_history_count"] != 128
        or balance["convergence_block_count"] != 4
        or balance["independent_joint_history_count_per_block"] != 32
        or balance["independent_disk_draw_count"] != 128
        or balance["independent_cluster_and_field_star_schedule_count"] != 128
        or balance["disk_draw_or_stellar_schedule_reuse_between_histories"] is not False
    ):
        raise ValueError("randomization history balance changed")
    if randomization["input_generation_numerics"]["may_execute_under_this_artifact"] is not False:
        raise ValueError("input generation was authorized")
    numerics = randomization["post_checkpoint_numerics"]
    if (
        numerics["primary_method"] != "REBOUND_4_4_11_IAS15"
        or Fraction(numerics["primary_epsilon"]) != Fraction(1, 100000000000)
        or Fraction(numerics["precision_repeat_epsilon"]) != Fraction(1, 1000000000000)
        or numerics["independent_method_may_share_same_native_library_and_count_as_independent_scientific_implementation"] is not False
        or numerics["threshold_widening_after_failure"] != "FORBIDDEN"
    ):
        raise ValueError("proposed numerical protocol changed")


def validate_analysis(analysis: dict[str, Any]) -> None:
    require_identity(analysis, "jx-o2-open-regeneration-analysis-gates/v1", "DESIGN_ONLY_STAGE_GATE_AND_G0_TRACE_PROTOCOL")
    if (
        analysis["artifact_state"] != "ALL_G0_ACCEPTANCE_REQUIREMENTS_UNVERIFIED"
        or analysis["observed_execution_authorized"] is not False
        or analysis["outcomes_generated"] is not False
    ):
        raise ValueError("analysis gate state changed")
    trace = analysis["g0_requirement_trace"]
    if len(trace) != 31 or len({item["requirement_id"] for item in trace}) != 31:
        raise ValueError("G0 requirement trace is incomplete or duplicated")
    actual_statuses = {item["requirement_id"]: item["g0_acceptance_status"] for item in trace}
    if actual_statuses != EXPECTED_REQUIREMENT_STATUSES:
        raise ValueError("G0 acceptance statuses changed")
    if any(item["g0_acceptance_status"] == "SEMANTICALLY_VERIFIED" for item in trace):
        raise ValueError("a design recipe cannot semantically accept a G0 requirement")
    semantics = analysis["requirement_semantics"]
    if semantics["semantically_verified_count"] != 0 or semantics["design_resolution_status_is_not_g0_acceptance"] is not True:
        raise ValueError("design and G0 acceptance were conflated")
    activation = analysis["activation_sequence"]
    for key, value in activation.items():
        if key.endswith("_present") or key.endswith("_passed") or key.endswith("_allowed"):
            if value is not False:
                raise ValueError(f"future activation gate already open: {key}")
    power = analysis["power_and_calibration_policy"]
    if (
        power["fixed_history_count"] != 128
        or power["threshold_selection_calibration_pseudocatalog_count_under_m0"] != 100000
        or power["absolute_adequacy_calibration_pseudocatalog_count_per_family"] != 100000
        or power["per_physical_row_power_pseudocatalog_count"] != 100000
        or power["m0_audit_pseudocatalog_count"] != 100000
        or power["m1_equal_weight_mixture_audit_pseudocatalog_count"] != 100000
        or Fraction(power["minimum_simultaneous_power"]) != Fraction(9, 10)
        or Fraction(power["maximum_simultaneous_global_type1_error"]) != Fraction(1, 20)
        or "T_STRICTLY_GREATER_THAN_C" not in power["critical_value_selection_rule"]
        or power["critical_value_tie_rule"] != "PRIMARY_REJECTION_REQUIRES_T_STRICTLY_GREATER_THAN_C_EQUAL_VALUES_DO_NOT_REJECT"
        or power["absolute_adequacy_pass_rule"] != "P_STRICTLY_GREATER_THAN_1_OVER_20"
        or power["failed_power_gate_action"] != "STOP_NO_OBSERVED_EXECUTION"
    ):
        raise ValueError("calibration or simultaneous-power gate changed")
    policy = analysis["analysis_policy"]
    if (
        policy["primary_joint_score_formula"] != EXPECTED_PRIMARY_SCORE_FORMULA
        or policy["primary_hypothesis_direction"] != "ONE_SIDED_M1_IMPROVEMENT_OVER_M0"
        or policy["m1_physical_and_history_angle_weights_are_fixed_not_fitted"] is not True
        or policy["physical_row_or_angle_maximization_selection_or_ranking_affects_primary_score"] is not False
    ):
        raise ValueError("primary score formula or fixed-mixture policy changed")
    if analysis["claim_firewall"]["allowed_current_statement"] != EXPECTED_CURRENT_STATEMENT:
        raise ValueError("allowed current statement changed")
    if any(value is not False for key, value in analysis["claim_firewall"].items() if key != "allowed_current_statement"):
        raise ValueError("claim firewall opened")
    outcomes = analysis["future_closed_outcome_vocabulary"]
    if outcomes["allowed_terminal_states_after_their_relevant_registered_gate_evaluation"] != [
        "M1_FAMILY_BETTER_PREDICTIVE_FIT_WITHIN_LOCKED_MODELS",
        "INCONCLUSIVE_WITHIN_LOCKED_MODELS",
        "NO_REGISTERED_MODEL_ADEQUATE",
        "SENSITIVITY_NOT_ESTABLISHED",
    ]:
        raise ValueError("future terminal outcome vocabulary changed")


def validate_prior_bindings(priors: dict[str, Any], workspace: Path) -> list[tuple[Path, str]]:
    require_identity(priors, "jx-o2-open-regeneration-prior-bindings/v1", "DESIGN_ONLY_GOVERNING_PROVENANCE_BINDINGS")
    if priors["outcomes_generated"] is not False or priors["binding_root"] != "WORKSPACE_ROOT":
        raise ValueError("prior binding state changed")
    records = priors["bindings"]
    if len(records) != len(EXPECTED_PRIOR_BINDINGS) or len({record["binding_id"] for record in records}) != len(records):
        raise ValueError("prior binding set is incomplete or duplicated")
    actual = {
        record["binding_id"]: (record["path"], record["sha256"], record["role"])
        for record in records
    }
    if actual != EXPECTED_PRIOR_BINDINGS:
        raise ValueError("governing prior binding identity changed")
    checked: list[tuple[Path, str]] = []
    for record in records:
        path = canonical_regular(workspace, record["path"])
        if sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"governing source changed: {record['binding_id']}")
        checked.append((path, record["sha256"]))
    return checked


def collect_fraction_values(value: Any) -> set[Fraction]:
    result: set[Fraction] = set()
    if type(value) is int:
        result.add(Fraction(value, 1))
    elif type(value) is str:
        try:
            result.add(Fraction(value))
        except (ValueError, ZeroDivisionError):
            pass
    elif type(value) is list:
        for child in value:
            result.update(collect_fraction_values(child))
    elif type(value) is dict:
        for child in value.values():
            result.update(collect_fraction_values(child))
    return result


def validate_retirement_nonuse(artifacts: dict[str, dict[str, Any]]) -> None:
    scrubbed = copy.deepcopy(artifacts)
    records = scrubbed["priors"]["bindings"]
    retirement_records = [record for record in records if record["binding_id"] in RETIREMENT_BINDING_IDS]
    if len(retirement_records) != 2:
        raise ValueError("retirement governing bindings changed")
    scrubbed["priors"]["bindings"] = [record for record in records if record["binding_id"] not in RETIREMENT_BINDING_IDS]
    serialized = json.dumps(scrubbed, sort_keys=True, ensure_ascii=True).lower()
    if "9118" in serialized or any(token in serialized for token in RETIREMENT_FORBIDDEN_STRINGS):
        raise ValueError("retired candidate fingerprint appears outside its exact governing bindings")
    fraction_values = collect_fraction_values(scrubbed)
    if Fraction(9118, 1) in fraction_values:
        raise ValueError("retired candidate numeric index appears outside its exact governing bindings")
    if RETIREMENT_ORBIT.issubset(fraction_values):
        raise ValueError("retired candidate orbital fingerprint was reconstructed")


def verify(root: Path, expected_registration_sha256: str) -> dict[str, Any]:
    root = root.resolve()
    if Path(__file__).absolute() != Path(__file__).resolve() or root != Path(__file__).resolve().parent:
        raise ValueError("package root or verifier path is not canonical")
    validate_inventory(root)
    artifacts = {name: strict_json(root / filename) for name, filename in ARTIFACT_FILES.items()}
    for name, artifact in artifacts.items():
        if json_shape_sha256(artifact) != EXPECTED_SHAPE_SHA256[name]:
            raise ValueError(f"recursive JSON shape changed: {name}")
    registration, registration_sha256 = validate_registration(root, artifacts)
    if (
        len(expected_registration_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_registration_sha256)
        or registration_sha256 != expected_registration_sha256
    ):
        raise RuntimeError("registration does not match the caller-supplied external hash anchor")

    validate_main(artifacts["main"])
    validate_model(artifacts["model"])
    validate_randomization(artifacts["randomization"])
    validate_analysis(artifacts["analysis"])
    checked_sources = validate_prior_bindings(artifacts["priors"], root.parents[1])
    validate_retirement_nonuse(artifacts)

    linked = artifacts["main"]["linked_files"]
    if linked != {
        "model_family": ARTIFACT_FILES["model"],
        "randomization_and_numerics": ARTIFACT_FILES["randomization"],
        "analysis_gates": ARTIFACT_FILES["analysis"],
        "prior_bindings": ARTIFACT_FILES["priors"],
    }:
        raise ValueError("main linked-file set changed")

    # Second pass: detect ordinary mutation after the first validation pass.
    for relative, expected in registration["locked_files"].items():
        if sha256_file(canonical_regular(root, relative)) != expected:
            raise RuntimeError(f"registered file changed during verification: {relative}")
    for path, expected in checked_sources:
        if sha256_file(path) != expected:
            raise RuntimeError(f"governing source changed during verification: {path}")
    if sha256_file(root / "registration_v1.json") != expected_registration_sha256:
        raise RuntimeError("registration changed during verification")

    return {
        "status": PACKAGE_STATUS,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": PROTOCOL_ID,
        "registration_sha256": expected_registration_sha256,
        "g0_requirement_count": 31,
        "g0_requirements_semantically_verified": 0,
        "physical_support_count": 9,
        "paired_history_count": 128,
        "future_m1_member_count": 1152,
        "g0_complete": False,
        "eligible_for_g1": False,
        "execution_authorized": False,
        "observed_execution_authorized": False,
        "gpu_run_authorized": False,
        "claim_ceiling": CLAIM_CEILING,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the local JX-O2 open-regeneration preregistration package")
    parser.add_argument("--expected-registration-sha256", required=True)
    arguments = parser.parse_args()
    result = verify(Path(__file__).resolve().parent, arguments.expected_registration_sha256)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
