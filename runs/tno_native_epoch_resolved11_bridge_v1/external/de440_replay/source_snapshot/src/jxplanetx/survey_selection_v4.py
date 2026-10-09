"""Independent JX-O1 confirmation using fresh official OSSOS pools.

V4 keeps the V3 exact finite-pool zeta calculation and every V2 scientific
threshold, but requires a new intrinsic-population key, a new resampling key,
and pool hashes that cannot match the V2 execution.  A passing V4 result is an
independent computational confirmation of the locked calibration workflow. It
is not evidence for or against Planet X.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from . import survey_selection as v1
from . import survey_selection_v2 as v2
from . import survey_selection_v3 as v3
from .provenance import sha256_data, sha256_file


CONTRACT_SCHEMA = v3.CONTRACT_SCHEMA
RESULT_SCHEMA = v3.RESULT_SCHEMA
SurveySelectionError = v3.SurveySelectionError
SurveySelectionVerdict = v3.SurveySelectionVerdict

validate_survey_contract = v3.validate_survey_contract
load_survey_contract = v3.load_survey_contract
exact_zeta_moments = v3.exact_zeta_moments

EXPERIMENT_ID = "jx-o1-ossos-b-telescope-selection-v4-independent-confirmation"
POPULATION_SEED = "jx-o1-independent-confirmation-population-2026-08-23-v4"
RESAMPLING_SEED = "jx-o1-independent-confirmation-resampling-2026-08-23-v4"
OFFICIAL_BACKEND = "OSSOS_SURVEY_SIMULATOR_F95_86CE093_V4_INDEPENDENT_14_FIELD"
EVIDENCE_RELATIONSHIP = "INDEPENDENT_CONFIRMATION_WITH_FRESH_OFFICIAL_POOLS"

V2_POPULATION_SEED = "jx-o1-intrinsic-population-2026-08-23-v2"
V2_RESAMPLING_SEED = "jx-o1-bootstrap-and-pit-2026-08-23-v2"
V2_RESULT_SHA256 = "9f0d86d6365b776a64333f55a431c143d36793f489f9a9297c3bbaa054e5c9bc"
V3_RESULT_SHA256 = "2256fd5899a1d869a66241754d65eb4ba01f28ba773188b19f7a4e4d96bc5879"
V2_POOL_SHA256 = {
    "correct": "b2b99a91ca52fe819b8e2a5d0a860488d0b650664cbe5abb5a6d0b773b3f8297",
    "wrong": "85c55eca72c8aaac0292c1f25d6ed3aed1b1d03efa194e35cde023396b8716fd",
}
V2_POOL_SEMANTIC_SHA256 = {
    "correct": "e1676f5fe4a9d5f67c4d4c5e24a9e96db2541b5593a14d26f8f0f3cdde20982b",
    "wrong": "024bc871e3347147d90f0a9fbaf9f573fcec126b7ee586415f21827349228895",
}


def independent_contract_audit(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the deterministic V4 seed and execution independence audit."""

    population_seed = contract.get("population", {}).get("seed_key")
    resampling_seed = contract.get("statistics", {}).get("resampling_seed")
    backend = contract.get("execution", {}).get("official_backend_name")
    checks = {
        "experiment_id_is_v4": contract.get("experiment_id") == EXPERIMENT_ID,
        "population_seed_is_predeclared_v4": population_seed == POPULATION_SEED,
        "population_seed_differs_from_v2": population_seed != V2_POPULATION_SEED,
        "resampling_seed_is_predeclared_v4": resampling_seed == RESAMPLING_SEED,
        "resampling_seed_differs_from_v2": resampling_seed != V2_RESAMPLING_SEED,
        "official_backend_is_v4": backend == OFFICIAL_BACKEND,
        "checkpoint_restart_required": contract.get("execution", {}).get(
            "checkpoint_restart_required"
        )
        is True,
        "exact_replay_required": contract.get("gates", {}).get("exact_replay_required")
        is True,
    }
    return {
        "schema": "jx-survey-selection-independence-audit/v4",
        "checks": checks,
        "all_passed": all(checks.values()),
        "baseline_v2_result_sha256": V2_RESULT_SHA256,
        "baseline_v3_corrective_result_sha256": V3_RESULT_SHA256,
    }


def _require_independent_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    audit = independent_contract_audit(contract)
    if not audit["all_passed"]:
        failed = sorted(name for name, passed in audit["checks"].items() if not passed)
        v1._fail(
            "not_independent_confirmation",
            "V4 contract independence checks failed: " + ", ".join(failed),
        )
    return audit


def independent_pool_audit(
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
) -> dict[str, Any]:
    """Prove that both registered pools differ from the official V2 pools."""

    paths = {
        "correct": Path(correct_manifest_path),
        "wrong": Path(wrong_manifest_path),
    }
    pools: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for model_id, path in paths.items():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            v1._fail("invalid_pool_manifest", f"cannot audit {path}: {exc}")
        manifest_sha = sha256_file(path)
        semantic_sha = manifest.get("detection_semantic_sha256")
        manifest_is_fresh = manifest_sha != V2_POOL_SHA256[model_id]
        semantic_is_fresh = semantic_sha != V2_POOL_SEMANTIC_SHA256[model_id]
        checks[f"{model_id}_manifest_differs_from_v2"] = manifest_is_fresh
        checks[f"{model_id}_semantic_pool_differs_from_v2"] = semantic_is_fresh
        pools[model_id] = {
            "manifest_sha256": manifest_sha,
            "detection_semantic_sha256": semantic_sha,
            "v2_manifest_sha256": V2_POOL_SHA256[model_id],
            "v2_detection_semantic_sha256": V2_POOL_SEMANTIC_SHA256[model_id],
        }
    checks["correct_and_wrong_manifests_are_distinct"] = (
        pools["correct"]["manifest_sha256"] != pools["wrong"]["manifest_sha256"]
    )
    checks["correct_and_wrong_semantic_pools_are_distinct"] = (
        pools["correct"]["detection_semantic_sha256"]
        != pools["wrong"]["detection_semantic_sha256"]
    )
    audit = {
        "schema": "jx-survey-selection-pool-independence-audit/v4",
        "checks": checks,
        "all_passed": all(checks.values()),
        "pools": pools,
    }
    if not audit["all_passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        v1._fail(
            "reused_official_pool",
            "V4 pool independence checks failed: " + ", ".join(failed),
        )
    return audit


def _finalize_impl(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
) -> dict[str, Any]:
    contract = load_survey_contract(contract_path)
    contract_audit = _require_independent_contract(contract)
    pool_audit = independent_pool_audit(correct_manifest_path, wrong_manifest_path)
    result = v3._finalize_impl(
        contract_path,
        correct_manifest_path,
        wrong_manifest_path,
    )
    result["evidence_relationship"] = EVIDENCE_RELATIONSHIP
    result["independence_audit"] = {
        "contract": contract_audit,
        "pools": pool_audit,
    }
    result.pop("replay_sha256", None)
    result["replay_sha256"] = sha256_data(result)
    return result


def finalize_survey_selection(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Emit an immutable independent V4 verdict."""

    target = Path(output)
    if target.exists():
        v1._fail("output_exists", f"refusing to overwrite immutable output: {target}")
    try:
        result = _finalize_impl(
            contract_path,
            correct_manifest_path,
            wrong_manifest_path,
        )
    except SurveySelectionError as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "milestone": "JX-O1_TELESCOPE_SELECTION_VALIDATION",
            "verdict": SurveySelectionVerdict.INVALID.value,
            "claim_decision": "INVALID",
            "invalid_reasons": [{"code": exc.code, "message": exc.message}],
            "blocked_reasons": [],
            "nonclaims": list(v2.NONCLAIMS),
            "evidence_relationship": "INDEPENDENT_CONFIRMATION_NOT_COMPLETED",
        }
        result["replay_sha256"] = sha256_data(result)
    v1._atomic_json(target, result)
    return result
