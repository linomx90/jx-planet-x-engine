#!/usr/bin/env python3
"""Independently verify the hardened JX-E1 smoke replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import rebound

import run_pilot as engine
import run_pilot_v3 as hardened


def by_label(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {record["label"]: record for record in records}
    if len(result) != len(records):
        raise ValueError("duplicate run label")
    return result


def expected_labels(contract: dict[str, Any]) -> tuple[set[str], set[str]]:
    blocks = range(int(contract["tracer_design"]["blocks"]))
    primary = {f"M0-b{block:02d}" for block in blocks}
    primary.update(
        f"{model['id']}-{angle['id']}-b{block:02d}"
        for model in contract["model_grid"]
        for angle in contract["angle_grid"]
        for block in blocks
    )
    audit = {f"dt-half-M0-b{contract['timestep_audit']['block']:02d}"}
    audit.update(
        f"dt-half-{model_id}-{contract['timestep_audit']['angle_id']}-b{contract['timestep_audit']['block']:02d}"
        for model_id in contract["timestep_audit"]["model_ids"]
    )
    return primary, audit


def verify_one(
    result: dict[str, Any],
    output_dir: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    primary = by_label(result["run_records"])
    audit = by_label(result["audit_run_records"])
    expected_primary, expected_audit = expected_labels(contract)
    embedded_semantic = engine.sha256_bytes(engine.canonical_bytes(result["semantic"]))
    checkpoint_records = {**primary, **audit}
    checkpoint_paths = {path.stem: path for path in (output_dir / "checkpoints").glob("*.bin")}
    checkpoint_sets_exact = set(checkpoint_paths) == set(checkpoint_records)
    decoded_exact = checkpoint_sets_exact and all(
        hardened.decoded_state_digest(rebound.Simulation(str(checkpoint_paths[label])))
        == checkpoint_records[label]["checkpoint_decoded_state_sha256"]
        for label in checkpoint_records
    )
    return {
        "verdict_valid": result["verdict"] == "ENGINEERING_SMOKE_VALID",
        "all_declared_checks_true": all(result["checks"].values()),
        "semantic_hash_recomputed_exact": embedded_semantic == result["semantic_sha256"],
        "primary_run_key_set_exact": set(primary) == expected_primary,
        "audit_run_key_set_exact": set(audit) == expected_audit,
        "checkpoint_run_key_set_exact": checkpoint_sets_exact,
        "every_loaded_checkpoint_matches_recorded_decoded_hash": decoded_exact,
        "mercurius_r_crit_hill_readback_exact": all(
            record["mercurius_r_crit_hill"] == float(contract["dynamics"]["mercurius_hillfac"])
            for record in checkpoint_records.values()
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    verification_path = arguments.contract.resolve()
    verification = engine.strict_json(verification_path)
    if verification.get("schema") != "jx-e1-hardened-replay-verification-contract/v1":
        raise ValueError("unexpected verification schema")
    if verification["verifier_sha256"] != engine.sha256_file(Path(__file__).resolve()):
        raise ValueError("verifier hash mismatch")
    experiment_contract_path = (verification_path.parent / verification["experiment_contract_path"]).resolve()
    if engine.sha256_file(experiment_contract_path) != verification["experiment_contract_sha256"]:
        raise ValueError("experiment contract hash mismatch")
    experiment_contract = engine.strict_json(experiment_contract_path)
    results = []
    for specification in verification["outputs"]:
        directory = (verification_path.parent / specification["directory"]).resolve()
        result_path = directory / "result_v1.json"
        if engine.sha256_file(result_path) != specification["result_sha256"]:
            raise ValueError(f"result hash mismatch for {directory.name}")
        result = engine.strict_json(result_path)
        results.append((directory, result, verify_one(result, directory, experiment_contract)))
    first, second = results[0][1], results[1][1]
    cross_checks = {
        "semantic_sha256_exact_across_executions": first["semantic_sha256"] == second["semantic_sha256"],
        "base_deterministic_payload_sha256_exact_across_executions": (
            first["deterministic_payload_sha256"] == second["deterministic_payload_sha256"]
        ),
        "semantic_objects_exact_across_executions": first["semantic"] == second["semantic"],
        "primary_endpoint_hashes_exact_by_key": {
            record["label"]: record["endpoint_state_sha256"] for record in first["run_records"]
        } == {
            record["label"]: record["endpoint_state_sha256"] for record in second["run_records"]
        },
        "audit_endpoint_hashes_exact_by_key": {
            record["label"]: record["endpoint_state_sha256"] for record in first["audit_run_records"]
        } == {
            record["label"]: record["endpoint_state_sha256"] for record in second["audit_run_records"]
        },
    }
    valid = all(all(checks.values()) for _, _, checks in results) and all(cross_checks.values())
    receipt = {
        "schema": "jx-e1-hardened-replay-verification-receipt/v1",
        "experiment_id": "jx-e1-p9-9x4-smoke-v3-replay",
        "verdict": "HARDENED_SEMANTIC_REPLAY_EXACT" if valid else "HARDENED_REPLAY_INVALID",
        "claim_ceiling": "ENGINEERING_SURROGATE_ONLY",
        "verification_contract_sha256": engine.sha256_file(verification_path),
        "experiment_contract_sha256": verification["experiment_contract_sha256"],
        "semantic_sha256": first["semantic_sha256"] if valid else None,
        "per_execution_checks": {
            directory.name: checks for directory, _, checks in results
        },
        "cross_execution_checks": cross_checks,
        "raw_checkpoint_container_hashes_affect_verdict": False,
        "nonclaim": verification["mandatory_nonclaim"],
    }
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": receipt["verdict"], "output": str(output_path)}, indent=2))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
