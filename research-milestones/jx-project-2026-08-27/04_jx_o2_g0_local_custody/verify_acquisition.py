from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ID = "jx-o2-characterized-survey-model-comparison-design-v1"
REGISTRATION_SCHEMA = "jx-o2-g0-local-acquisition-registration/v1"
LOCKED_FILES = {
    "README.md",
    "local_custody_manifest_v1.json",
    "acquisition_checklist_v1.json",
    "verify_acquisition.py",
    "test_acquisition.py",
}
sys.dont_write_bytecode = True

CHECKLIST_KEYS = {
    "schema", "artifact_id", "experiment_id", "artifact_class", "status",
    "audit_status", "g0_complete", "eligible_for_g1", "execution_authorized",
    "gpu_run_authorized", "observed_execution_authorized", "outcomes_generated",
    "claim_ceiling", "source_bindings", "local_custody_binding", "permissions",
    "survey_acceptance_requirements", "matched_model_acceptance_requirements",
    "acceptance_policy", "claim_firewall", "mandatory_nonclaim",
}
CUSTODY_KEYS = {
    "schema", "artifact_id", "experiment_id", "artifact_class", "custody_status",
    "recorded_at_utc", "timestamp_authority", "externally_timestamped",
    "execution_authorized", "outcomes_generated", "custody_root",
    "file_count_including_notice", "total_bytes_including_notice",
    "quarantine_notice", "survey_artifacts", "model_documentation_artifacts",
    "custody_assertions", "mandatory_nonclaim",
}
ALLOWED_ACTIVATION_STATUSES = {
    "DOCUMENTATION_ONLY_CHECKPOINT_ABSENT",
    "DOCUMENTATION_ONLY_FULL_STATES_CHECKPOINT_SEEDS_OUTPUTS_ABSENT",
    "DOCUMENTATION_ONLY_FULL_STATE_CHECKPOINT_SEEDS_OUTPUTS_ABSENT",
    "DOCUMENTATION_ONLY_NEVER_RECOVERED_2024_STATE",
    "DOCUMENTATION_ONLY_NOT_COMPLETE_SELECTION",
    "DOCUMENTATION_ONLY_PENDING_GLOBAL_SURVEY_RULES",
    "NOT_MATCHED_TNO_SOURCE_POPULATION_CANDIDATE_9118_RETIRED",
    "QUARANTINED_DEDUP_AND_SELECTION_RULES_UNLOCKED",
    "QUARANTINED_EXPOSURE_COUNT_AND_GRIZ_RULES_UNLOCKED",
    "QUARANTINED_EXPOSURE_COUNT_AND_SELECTION_RULES_UNLOCKED",
    "QUARANTINED_HR_RELEASE_SELECTION_UNLOCKED",
    "QUARANTINED_INCOMPLETE_LATER_OSSOS_COMPONENTS_ABSENT",
    "QUARANTINED_INCOMPLETE_NOT_TAGGED_RELEASE",
    "QUARANTINED_PAPER_ERA_VERSION_DEPENDENCIES_LICENSE_UNLOCKED",
    "QUARANTINED_PENDING_GLOBAL_SURVEY_RULES",
    "QUARANTINED_PENDING_LICENSE_AND_RULE_LOCK",
    "QUARANTINED_RELEASE_AND_ELIGIBILITY_RULES_UNLOCKED",
    "QUARANTINED_SELECTION_INCOMPLETE",
    "QUARANTINED_VERSION_DEPENDENCY_RNG_LICENSE_TESTS_UNLOCKED",
    "REFERENCE_ONLY_NOT_CHARACTERIZED_INPUT",
    "REFERENCE_ONLY_NOT_SOURCE_POPULATION",
    "REFERENCE_ONLY_UNBOUND_LOCAL_BYTES",
}
EXPECTED_SOURCE_BINDINGS_SHA256 = "177015a132d442a0bd1980b5926b5f759b98a4439e1b372f9f64ada5cace104e"
EXPECTED_CUSTODY_POLICY_SHA256 = "7ff6edb23146bf813119ea5dcaf5ee60bdb8eaca7cb64164dece0431637aa6c1"
EXPECTED_REQUIREMENTS_SHA256 = "b5082b2b0610c26b4d1605c5f2dc65d9fd103834d94dd041a478569a25adff1f"
EXPECTED_REQUIREMENT_IDS = (
    {f"OSSOS-A{index:02d}" for index in range(1, 9)},
    {f"DES-A{index:02d}" for index in range(1, 9)},
    {f"MODEL-A{index:02d}" for index in range(1, 16)},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number")
    if isinstance(value, dict):
        for child in value.values():
            reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            reject_nonfinite(child)


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    reject_nonfinite(value)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def canonical_regular(base: Path, relative: str) -> Path:
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError("path is not canonical relative")
    candidate = base / rel
    if candidate.absolute() != candidate.resolve() or candidate.is_symlink():
        raise ValueError("path traverses a symlink or alias")
    if not candidate.is_file():
        raise ValueError("bound path is not a regular file")
    return candidate


def artifact_records(custody: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield from custody["survey_artifacts"]["ossos"]
    yield from custody["survey_artifacts"]["des"]
    yield from custody["model_documentation_artifacts"]


def validate_registration(root: Path, checklist: dict[str, Any]) -> tuple[dict[str, Any], str]:
    registration_path = root / "registration_v1.json"
    registration = strict_json(registration_path)
    if set(registration) != {
        "schema",
        "artifact_id",
        "experiment_id",
        "registration_state",
        "recorded_at_utc",
        "timestamp_authority",
        "externally_timestamped",
        "scientific_evidence_artifact",
        "outcomes_generated",
        "execution_permissions",
        "locked_files",
        "mandatory_nonclaim",
    }:
        raise ValueError("registration top-level shape changed")
    expected_permissions = {
        "local_hash_verification_authorized": True,
        "network_or_github_action_authorized": False,
        "input_activation_authorized": False,
        "dynamics_authorized": False,
        "synthetic_calibration_authorized": False,
        "observed_data_access_authorized": False,
        "observed_model_comparison_authorized": False,
        "gpu_execution_authorized": False,
        "g1_contract_authorized": False,
        "planet_x_claim_authorized": False,
    }
    if (
        registration["schema"] != REGISTRATION_SCHEMA
        or registration["artifact_id"] != "jx-o2-g0-local-acquisition-registration-v1"
        or registration["experiment_id"] != EXPERIMENT_ID
        or registration["registration_state"] != "LOCAL_HASH_LOCK_COMPLETE_G0_REMAINS_BLOCKED"
        or registration["timestamp_authority"]
        != "LOCAL_CONTENT_HASH_REGISTRATION_ONLY_NO_EXTERNAL_TIMESTAMP"
        or registration["externally_timestamped"] is not False
        or registration["scientific_evidence_artifact"] is not False
        or registration["outcomes_generated"] is not False
        or registration["execution_permissions"] != expected_permissions
        or registration["mandatory_nonclaim"] != checklist["mandatory_nonclaim"]
        or not isinstance(registration["recorded_at_utc"], str)
        or not registration["recorded_at_utc"].endswith("Z")
    ):
        raise ValueError("registration identity or permission boundary changed")
    if set(registration["locked_files"]) != LOCKED_FILES:
        raise ValueError("registration locked-file set changed")
    for relative, expected in registration["locked_files"].items():
        locked_path = canonical_regular(root, relative)
        if sha256_file(locked_path) != expected:
            raise RuntimeError(f"registered file changed: {relative}")
    return registration, sha256_file(registration_path)


def verify(root: Path, expected_registration_sha256: str) -> dict[str, Any]:
    root = root.resolve()
    if root != Path(__file__).resolve().parent:
        raise ValueError("acquisition package root is not canonical")
    checklist = strict_json(root / "acquisition_checklist_v1.json")
    custody = strict_json(root / "local_custody_manifest_v1.json")
    registration, registration_sha256 = validate_registration(root, checklist)
    if (
        len(expected_registration_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_registration_sha256)
        or registration_sha256 != expected_registration_sha256
    ):
        raise RuntimeError("registration does not match the required external hash anchor")

    if set(checklist) != CHECKLIST_KEYS or set(custody) != CUSTODY_KEYS:
        raise ValueError("checklist or custody top-level shape changed")
    if checklist["schema"] != "jx-o2-g0-local-acquisition-checklist/v1":
        raise ValueError("checklist schema changed")
    if custody["schema"] != "jx-o2-g0-local-custody-manifest/v1":
        raise ValueError("custody schema changed")
    if checklist["experiment_id"] != EXPERIMENT_ID or custody["experiment_id"] != EXPERIMENT_ID:
        raise ValueError("experiment identity changed")
    if (
        checklist["status"] != "BLOCKED_PARTIAL_LOCAL_CUSTODY"
        or checklist["audit_status"] != "BLOCKED"
        or checklist["g0_complete"] is not False
        or checklist["eligible_for_g1"] is not False
        or checklist["execution_authorized"] is not False
        or checklist["gpu_run_authorized"] is not False
        or checklist["observed_execution_authorized"] is not False
        or checklist["outcomes_generated"] is not False
        or checklist["claim_ceiling"] != "ACQUISITION_CHECKLIST_ONLY"
        or custody["custody_status"] != "LOCAL_BYTES_HASH_VERIFIED_QUARANTINED"
        or custody["artifact_class"] != "LOCAL_PUBLIC_BYTES_CUSTODY_MANIFEST_ONLY"
        or custody["custody_root"] != "evidence"
        or custody["externally_timestamped"] is not False
        or custody["execution_authorized"] is not False
        or custody["outcomes_generated"] is not False
    ):
        raise ValueError("package fail-closed state changed")
    expected_permissions = {
        "preserve_and_hash_public_bytes_locally": True,
        "draft_author_or_custodian_requests": True,
        "network_or_github_action_authorized_by_this_artifact": False,
        "activate_quarantined_input": False,
        "generate_or_select_physical_checkpoint": False,
        "realize_angle_grid_or_prior": False,
        "realize_random_seeds": False,
        "run_dynamics": False,
        "run_synthetic_calibration": False,
        "access_or_unblind_untouched_holdout": False,
        "compute_observed_statistic": False,
        "run_observed_model_comparison": False,
        "start_large_gpu_job": False,
        "create_g1_execution_contract": False,
        "create_activation_receipt": False,
    }
    if checklist["permissions"] != expected_permissions:
        raise ValueError("checklist permissions changed")

    workspace = root.parents[1]
    if canonical_json_sha256(checklist["source_bindings"]) != EXPECTED_SOURCE_BINDINGS_SHA256:
        raise ValueError("source binding policy identity changed")
    source_ids: set[str] = set()
    for binding in checklist["source_bindings"]:
        if set(binding) != {"artifact_id", "path", "sha256"}:
            raise ValueError("source binding shape changed")
        if binding["artifact_id"] in source_ids:
            raise ValueError("duplicate source binding id")
        source_ids.add(binding["artifact_id"])
        path = canonical_regular(workspace, binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise RuntimeError(f"source binding changed: {binding['artifact_id']}")
    if len(source_ids) != 13:
        raise ValueError("source binding count changed")
    local_binding = checklist["local_custody_binding"]
    if local_binding["status"] != "BYTE_CUSTODY_VERIFIED_NOT_SEMANTICALLY_ACCEPTED":
        raise ValueError("local custody acceptance state changed")
    custody_path = canonical_regular(workspace, local_binding["path"])
    if custody_path != root / "local_custody_manifest_v1.json" or sha256_file(custody_path) != local_binding["sha256"]:
        raise RuntimeError("local custody manifest binding changed")

    expected_evidence: set[Path] = set()
    custody_policy = {
        "quarantine_notice": custody["quarantine_notice"],
        "survey_artifacts": custody["survey_artifacts"],
        "model_documentation_artifacts": custody["model_documentation_artifacts"],
    }
    if canonical_json_sha256(custody_policy) != EXPECTED_CUSTODY_POLICY_SHA256:
        raise ValueError("custody artifact policy identity changed")
    notice = custody["quarantine_notice"]
    notice_path = canonical_regular(root, notice["path"])
    if notice_path.stat().st_size != notice["byte_count"] or sha256_file(notice_path) != notice["sha256"]:
        raise RuntimeError("quarantine notice changed")
    expected_evidence.add(notice_path)
    artifact_ids: set[str] = set()
    total_bytes = notice_path.stat().st_size
    if (
        len(custody["survey_artifacts"]["ossos"]) != 10
        or len(custody["survey_artifacts"]["des"]) != 8
        or len(custody["model_documentation_artifacts"]) != 6
    ):
        raise ValueError("custody group counts changed")
    for record in artifact_records(custody):
        required = {"artifact_id", "path", "byte_count", "sha256", "role", "activation_status"}
        optional = {
            "record_count", "distinct_physical_object_count", "provenance_status",
            "canonical_source_locator", "retrieval_time_status", "license_status",
        }
        if not required.issubset(record) or not set(record).issubset(required | optional):
            raise ValueError("custody artifact shape incomplete")
        if record["artifact_id"] in artifact_ids:
            raise ValueError("duplicate custody artifact id")
        artifact_ids.add(record["artifact_id"])
        path = canonical_regular(root, record["path"])
        if path in expected_evidence:
            raise ValueError("duplicate custody artifact path")
        if path.stat().st_size != record["byte_count"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"custody artifact changed: {record['artifact_id']}")
        if record["activation_status"] not in ALLOWED_ACTIVATION_STATUSES:
            raise ValueError("custody artifact has an activating status")
        expected_unbound = {
            "A16_CDS_README": {
                "provenance_status": "UNBOUND_LOCAL_COPY_REQUIRES_REACQUISITION",
                "canonical_source_locator": "https://cdsarc.cds.unistra.fr/ftp/J/AJ/152/111/ReadMe",
                "retrieval_time_status": "NOT_RETAINED",
                "license_status": "NOT_AUDITED",
            },
            "HILAT_PUBLICATION_SOURCE_1608_02873": {
                "provenance_status": "UNBOUND_LOCAL_COPY_REQUIRES_REACQUISITION",
                "canonical_source_locator": "https://arxiv.org/src/1608.02873",
                "retrieval_time_status": "NOT_RETAINED",
                "license_status": "NOT_AUDITED",
            },
        }
        provenance_keys = set(next(iter(expected_unbound.values())))
        if record["artifact_id"] in expected_unbound:
            if (
                {key: record.get(key) for key in provenance_keys}
                != expected_unbound[record["artifact_id"]]
                or record["activation_status"] != "REFERENCE_ONLY_UNBOUND_LOCAL_BYTES"
            ):
                raise ValueError("unbound custody provenance status changed")
        elif provenance_keys & set(record):
            raise ValueError("unexpected custody provenance fields")
        expected_evidence.add(path)
        total_bytes += path.stat().st_size
    evidence_nodes = list((root / "evidence").rglob("*"))
    actual_evidence = {path for path in evidence_nodes if path.is_file()}
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in evidence_nodes):
        raise ValueError("evidence tree contains symlink or special entry")
    if actual_evidence != expected_evidence:
        raise ValueError("evidence file inventory changed")
    if (
        len(expected_evidence) != custody["file_count_including_notice"]
        or total_bytes != custody["total_bytes_including_notice"]
        or len(artifact_ids) != 24
    ):
        raise ValueError("custody file count or byte total changed")
    assertions = custody["custody_assertions"]
    if assertions != {
        "all_listed_bytes_sha256_verified_after_local_copy": True,
        "complete_authoritative_ossos_selection_bundle_present": False,
        "withdrawn_2019_deep_surveys_payload_preserved_or_eligible": False,
        "des_raw_selection_lineage_complete": False,
        "physical_cluster_checkpoint_present": False,
        "paired_physical_m0_m1_decks_present": False,
        "eligible_finite_full_state_m1_family_present": False,
        "candidate_9118_eligible_for_jx_o2": False,
        "this_manifest_satisfies_g0": False,
    }:
        raise ValueError("custody assertions changed")

    groups = [
        checklist["survey_acceptance_requirements"]["ossos"],
        checklist["survey_acceptance_requirements"]["des"],
        checklist["matched_model_acceptance_requirements"],
    ]
    requirements_policy = {
        "ossos": groups[0],
        "des": groups[1],
        "model": groups[2],
    }
    if canonical_json_sha256(requirements_policy) != EXPECTED_REQUIREMENTS_SHA256:
        raise ValueError("acceptance requirement policy identity changed")
    expected_counts = (8, 8, 15)
    requirement_ids: set[str] = set()
    for group, expected_count, expected_ids in zip(
        groups, expected_counts, EXPECTED_REQUIREMENT_IDS, strict=True
    ):
        if len(group) != expected_count:
            raise ValueError("acceptance requirement count changed")
        for requirement in group:
            if set(requirement) != {"requirement_id", "status", "required_evidence"}:
                raise ValueError("acceptance requirement shape changed")
            if requirement["requirement_id"] in requirement_ids:
                raise ValueError("duplicate acceptance requirement id")
            requirement_ids.add(requirement["requirement_id"])
            if requirement["status"] not in checklist["acceptance_policy"]["allowed_requirement_statuses"]:
                raise ValueError("unknown acceptance requirement status")
            if requirement["status"] == "SEMANTICALLY_VERIFIED":
                raise ValueError("current blocked checklist contains accepted requirement")
        if {item["requirement_id"] for item in group} != expected_ids:
            raise ValueError("acceptance requirement id set changed")
    if len(requirement_ids) != 31:
        raise ValueError("acceptance requirement total changed")
    policy = checklist["acceptance_policy"]
    expected_policy = {
        "allowed_requirement_statuses": [
            "AWAITING_AUTHORITY",
            "AWAITING_AUTHORITY_OR_OPEN_REGENERATION",
            "NOT_SATISFIED",
            "BYTE_ACQUIRED_UNAUDITED",
            "SEMANTICALLY_VERIFIED",
        ],
        "every_requirement_must_be_semantically_verified": True,
        "missing_unknown_or_placeholder_values_allowed_at_acceptance": False,
        "independent_audit_receipt_required": True,
        "license_audit_required": True,
        "durable_archive_required": True,
        "maintainer_comment_or_link_alone_satisfies_requirement": False,
        "withdrawn_2019_ossos_directory_satisfies_requirement": False,
        "partial_selection_bundle_satisfies_complete_survey_requirement": False,
        "software_license_inherited_by_data": False,
        "publication_summary_or_orbit_catalog_satisfies_physical_checkpoint_requirement": False,
        "engineering_surrogate_satisfies_matched_physical_model_requirement": False,
        "maximum_state_after_all_requirements_verified": "READY_TO_DRAFT_G1",
        "g1_execution_authorized_by_this_checklist": False,
        "observed_execution_authorized_after_g0_only": False,
    }
    if policy != expected_policy:
        raise ValueError("acceptance policy changed")
    expected_firewall = {
        "observed_statistic_or_model_preference_present": False,
        "planet_x_detection_exclusion_or_confirmation_claim_present": False,
        "more_particles_or_gpu_may_repair_missing_provenance": False,
        "byte_hash_match_alone_may_establish_semantic_eligibility": False,
        "candidate_9118_may_be_reintroduced": False,
    }
    if checklist["claim_firewall"] != expected_firewall:
        raise ValueError("claim firewall changed")
    if custody["mandatory_nonclaim"] != (
        "This manifest records local custody of public bytes only. It does not make the survey inputs complete, "
        "recover a physical checkpoint or matched model pair, authorize execution, or provide evidence for or against Planet X."
    ):
        raise ValueError("custody nonclaim changed")

    root_nodes = list(root.iterdir())
    if any(path.is_symlink() or (not path.is_file() and not path.is_dir()) for path in root_nodes):
        raise ValueError("package root contains symlink or special entry")
    top_level_files = {path.name for path in root_nodes if path.is_file()}
    expected_top_level = LOCKED_FILES | {"registration_v1.json"}
    if top_level_files != expected_top_level:
        raise ValueError("package top-level file set changed")
    expected_directories = {
        Path("evidence"),
        Path("evidence/ossos"),
        Path("evidence/des"),
        Path("evidence/literature"),
        Path("evidence/model_catalogs"),
    }
    actual_directories = {
        path.relative_to(root) for path in root.rglob("*") if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise ValueError("package directory set changed")
    return {
        "status": "JX_O2_G0_LOCAL_CUSTODY_VERIFIED_BLOCKED",
        "registration_sha256": registration_sha256,
        "evidence_file_count": len(expected_evidence),
        "evidence_total_bytes": total_bytes,
        "acceptance_requirement_count": len(requirement_ids),
        "g0_complete": False,
        "eligible_for_g1": False,
        "execution_authorized": False,
        "claim_ceiling": checklist["claim_ceiling"],
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_acquisition.py PACKAGE_ROOT EXPECTED_REGISTRATION_SHA256"
        )
    root = Path(sys.argv[1])
    print(json.dumps(verify(root, sys.argv[2]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
