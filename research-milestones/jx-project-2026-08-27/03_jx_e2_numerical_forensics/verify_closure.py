from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any


CLOSURE_SCHEMA = "jx-e2-numerics-local-closure/v1"
EXPERIMENT_ID = "jx-e2-active-frame-integrator-50k-v1"
EXPECTED_BINDINGS = {
    "registration",
    "closure_verifier",
    "execution_a",
    "execution_b",
    "replay_receipt",
    "final_report",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def canonical_path(workspace: Path, relative: str, *, directory: bool = False) -> Path:
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise ValueError("closure path is not canonical workspace-relative")
    candidate = workspace / candidate_relative
    if candidate.absolute() != candidate.resolve() or candidate.is_symlink():
        raise ValueError("closure path traverses a symlink or alias")
    if directory:
        if not candidate.is_dir():
            raise ValueError("bound output root is not a directory")
    elif not candidate.is_file():
        raise ValueError("bound artifact is not a regular file")
    return candidate


def load_bound_verifier(path: Path, expected_sha256: str) -> types.ModuleType:
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise RuntimeError("registered verifier bytes changed")
    module = types.ModuleType("jx_e2_closure_bound_replay_verifier")
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module


def verify(closure_path: Path, expected_closure_sha256: str) -> dict[str, Any]:
    closure_path = closure_path.resolve()
    workspace = Path(__file__).resolve().parents[2]
    if closure_path != Path(__file__).resolve().parent / "closure_v1.json":
        raise ValueError("closure path is not canonical")
    closure = strict_json(closure_path)
    if (
        len(expected_closure_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_closure_sha256)
        or sha256_file(closure_path) != expected_closure_sha256
    ):
        raise RuntimeError("closure does not match the required external hash anchor")
    if set(closure) != {
        "schema",
        "experiment_id",
        "closure_state",
        "artifact_class",
        "recorded_at_utc",
        "timestamp_authority",
        "externally_timestamped",
        "scientific_evidence_artifact",
        "new_dynamics_executed",
        "path_base",
        "claim_ceiling",
        "replay_verdict",
        "overall_numerical_classification",
        "semantic_sha256",
        "bindings",
        "closure_policy",
        "verification_scope",
        "mandatory_nonclaim",
    }:
        raise ValueError("closure top-level shape changed")
    if (
        closure["schema"] != CLOSURE_SCHEMA
        or closure["experiment_id"] != EXPERIMENT_ID
        or closure["closure_state"] != "JX_E2_LOCAL_CLOSURE_COMPLETE"
        or closure["artifact_class"] != "LOCAL_CONTENT_HASH_CLOSURE_ONLY"
        or closure["timestamp_authority"]
        != "LOCAL_CONTENT_HASH_CLOSURE_ONLY_NO_EXTERNAL_TIMESTAMP"
        or closure["externally_timestamped"] is not False
        or closure["scientific_evidence_artifact"] is not False
        or closure["new_dynamics_executed"] is not False
        or closure["path_base"] != "WORKSPACE_ROOT"
        or closure["claim_ceiling"] != "NUMERICAL_METHOD_FORENSICS_ONLY"
        or closure["replay_verdict"] != "JX_E2_SEMANTIC_REPLAY_EXACT"
        or closure["overall_numerical_classification"] != "MIXED_OR_INCONCLUSIVE"
        or not isinstance(closure["recorded_at_utc"], str)
        or not closure["recorded_at_utc"].endswith("Z")
    ):
        raise ValueError("closure identity or claim boundary changed")
    if set(closure["bindings"]) != EXPECTED_BINDINGS:
        raise ValueError("closure binding set changed")
    bindings = closure["bindings"]
    if (
        set(bindings["registration"]) != {"relative_path", "sha256"}
        or set(bindings["closure_verifier"]) != {"relative_path", "sha256"}
        or set(bindings["replay_receipt"]) != {"relative_path", "sha256"}
        or set(bindings["final_report"]) != {"relative_path", "sha256", "binding_scope"}
        or set(bindings["execution_a"]) != {
            "root_relative_path", "run_manifest_sha256", "result_sha256",
            "output_tree_manifest_algorithm", "output_tree_manifest_sha256",
        }
        or set(bindings["execution_b"]) != {
            "root_relative_path", "run_manifest_sha256", "result_sha256",
            "output_tree_manifest_algorithm", "output_tree_manifest_sha256",
        }
        or bindings["final_report"]["binding_scope"] != "BYTE_IDENTITY_ONLY"
    ):
        raise ValueError("closure nested binding shape or scope changed")
    expected_policy = {
        "additional_jx_e2_execution_authorized": False,
        "threshold_change_or_post_outcome_rescue_authorized": False,
        "jx_o2_execution_authorized": False,
        "gpu_execution_authorized": False,
        "observed_data_execution_authorized": False,
        "scientific_planet_x_claim_authorized": False,
        "new_work_requires_separate_preregistration": True,
    }
    if closure["closure_policy"] != expected_policy:
        raise ValueError("closure policy changed")

    if closure["verification_scope"] != (
        "Content-hash closure only. It binds byte identity and the previously verified output-tree digests; "
        "it does not reclassify E2, reconstruct unretained samples, provide an external timestamp, or constitute "
        "independent scientific replication."
    ):
        raise ValueError("closure verification scope changed")
    registration_path = canonical_path(workspace, bindings["registration"]["relative_path"])
    verifier_path = canonical_path(workspace, bindings["closure_verifier"]["relative_path"])
    receipt_path = canonical_path(workspace, bindings["replay_receipt"]["relative_path"])
    report_path = canonical_path(workspace, bindings["final_report"]["relative_path"])
    for key, path in (
        ("registration", registration_path),
        ("closure_verifier", verifier_path),
        ("replay_receipt", receipt_path),
        ("final_report", report_path),
    ):
        if sha256_file(path) != bindings[key]["sha256"]:
            raise RuntimeError(f"closure file binding changed: {key}")

    registration = strict_json(registration_path)
    package_root = registration_path.parent
    contract_path = package_root / "contract_v1.json"
    replay_verifier_path = package_root / "verify_replay.py"
    replay_verifier = load_bound_verifier(
        replay_verifier_path,
        registration["locked_files"]["verify_replay.py"],
    )
    contract = replay_verifier.strict_json(contract_path)
    replay_verifier.validate_actual_runtime()
    replay_verifier.validate_registration(registration_path, contract_path)
    registration_sha256 = sha256_file(registration_path)

    verified: dict[str, Any] = {}
    for label, key in (("E2-A", "execution_a"), ("E2-B", "execution_b")):
        binding = bindings[key]
        root = canonical_path(workspace, binding["root_relative_path"], directory=True)
        result = replay_verifier.verify_output(contract, registration_sha256, root)
        if result["manifest"]["execution_label"] != label:
            raise RuntimeError("closure execution label changed")
        if sha256_file(root / "run_manifest.json") != binding["run_manifest_sha256"]:
            raise RuntimeError("closure run-manifest binding changed")
        if result["result_file_sha256"] != binding["result_sha256"]:
            raise RuntimeError("closure result binding changed")
        if result["output_manifest_sha256"] != binding["output_tree_manifest_sha256"]:
            raise RuntimeError("closure output-tree binding changed")
        if binding["output_tree_manifest_algorithm"] != "jx-e2-recursive-file-manifest-sha256/v1":
            raise ValueError("closure output-tree algorithm changed")
        verified[key] = result

    first = verified["execution_a"]
    second = verified["execution_b"]
    if (
        first["manifest"]["execution_instance_id"]
        == second["manifest"]["execution_instance_id"]
        or first["result"]["semantic"] != second["result"]["semantic"]
        or first["result"]["semantic_sha256"] != closure["semantic_sha256"]
        or second["result"]["semantic_sha256"] != closure["semantic_sha256"]
        or first["result"]["semantic"]["overall_numerical_classification"]
        != closure["overall_numerical_classification"]
        or second["result"]["semantic"]["overall_numerical_classification"]
        != closure["overall_numerical_classification"]
    ):
        raise RuntimeError("closure semantic replay identity changed")

    receipt = strict_json(receipt_path)
    if (
        set(receipt) != {
            "schema", "experiment_id", "contract_sha256", "registration_sha256",
            "runner_sha256", "verifier_sha256", "execution_a_result_sha256",
            "execution_b_result_sha256", "execution_a_output_manifest_sha256",
            "execution_b_output_manifest_sha256", "semantic_sha256", "verdict",
            "claim_ceiling", "overall_numerical_classification", "verification_scope",
            "mandatory_nonclaim",
        }
        or receipt["schema"] != "jx-e2-numerics-replay-receipt/v1"
        or receipt["experiment_id"] != EXPERIMENT_ID
        or receipt["verdict"] != closure["replay_verdict"]
        or receipt["claim_ceiling"] != closure["claim_ceiling"]
        or receipt["overall_numerical_classification"]
        != closure["overall_numerical_classification"]
        or receipt["semantic_sha256"] != closure["semantic_sha256"]
        or receipt["mandatory_nonclaim"] != closure["mandatory_nonclaim"]
        or contract["mandatory_nonclaim"] != closure["mandatory_nonclaim"]
        or receipt["execution_a_result_sha256"] != bindings["execution_a"]["result_sha256"]
        or receipt["execution_b_result_sha256"] != bindings["execution_b"]["result_sha256"]
        or receipt["execution_a_output_manifest_sha256"]
        != bindings["execution_a"]["output_tree_manifest_sha256"]
        or receipt["execution_b_output_manifest_sha256"]
        != bindings["execution_b"]["output_tree_manifest_sha256"]
        or receipt["registration_sha256"] != registration_sha256
        or receipt["contract_sha256"] != registration["locked_files"]["contract_v1.json"]
        or receipt["runner_sha256"] != registration["locked_files"]["run_numerics.py"]
        or receipt["verifier_sha256"] != registration["locked_files"]["verify_replay.py"]
        or receipt["verification_scope"]
        != (
            "The verifier independently checks locked identities, checkpoint containers and decoded states, "
            "structural completeness, stored classification arithmetic, and exact A/B semantic replay. Raw "
            "10-year state samples are not retained, so sampled maxima are content-bound by A/B equality but "
            "cannot be independently reconstructed from trajectories. This is not an independent scientific "
            "implementation."
        )
    ):
        raise RuntimeError("closure receipt or nonclaim changed")
    for key, path in (
        ("registration", registration_path),
        ("closure_verifier", verifier_path),
        ("replay_receipt", receipt_path),
        ("final_report", report_path),
    ):
        if sha256_file(path) != bindings[key]["sha256"]:
            raise RuntimeError(f"closure input changed during verification: {key}")
    if (
        replay_verifier.recursive_file_manifest_sha256(
            canonical_path(workspace, bindings["execution_a"]["root_relative_path"], directory=True)
        )
        != bindings["execution_a"]["output_tree_manifest_sha256"]
        or replay_verifier.recursive_file_manifest_sha256(
            canonical_path(workspace, bindings["execution_b"]["root_relative_path"], directory=True)
        )
        != bindings["execution_b"]["output_tree_manifest_sha256"]
        or sha256_file(closure_path) != expected_closure_sha256
    ):
        raise RuntimeError("closure inputs changed during final verification pass")
    return {
        "status": "JX_E2_LOCAL_CLOSURE_VERIFIED",
        "closure_sha256": expected_closure_sha256,
        "semantic_sha256": closure["semantic_sha256"],
        "classification": closure["overall_numerical_classification"],
        "claim_ceiling": closure["claim_ceiling"],
        "new_dynamics_executed": False,
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: verify_closure.py CLOSURE_PATH EXPECTED_CLOSURE_SHA256")
    closure_path = Path(sys.argv[1])
    print(json.dumps(verify(closure_path, sys.argv[2]), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
