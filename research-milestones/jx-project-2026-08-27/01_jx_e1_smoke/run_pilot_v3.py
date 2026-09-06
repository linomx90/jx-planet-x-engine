#!/usr/bin/env python3
"""Hardened JX-E1 smoke runner with decoded configuration-state hashing."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import rebound

import run_pilot as engine


BASE_RUNNER = Path(__file__).with_name("run_pilot.py").resolve()
BASE_CONTRACT = Path(__file__).with_name("contract_v2.json").resolve()
BASE_REPLAY = Path(__file__).with_name("replay_receipt_v1.json").resolve()
ORIGINAL_VALIDATOR = engine.validate_contract
ORIGINAL_BUILD = engine.build_simulation
ORIGINAL_DIGEST = engine.simulation_digest
ORIGINAL_RUN_ARM = engine.run_arm
ORIGINAL_PUBLIC_RECORD = engine.public_record
CAPTURED_RECORDS: list[dict[str, Any]] = []


def decoded_state_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx-e1-decoded-state/v2\0")
    configuration = {
        "t_hex": float(simulation.t).hex(),
        "G_hex": float(simulation.G).hex(),
        "dt_hex": float(simulation.dt).hex(),
        "N": int(simulation.N),
        "N_active": int(simulation.N_active),
        "integrator": str(simulation.integrator),
        "gravity": str(simulation.gravity),
        "collision": str(simulation.collision),
        "boundary": str(simulation.boundary),
        "testparticle_type": int(simulation.testparticle_type),
        "mercurius_r_crit_hill_hex": float(simulation.ri_mercurius.r_crit_hill).hex(),
        "mercurius_safe_mode": int(simulation.ri_mercurius.safe_mode),
        "mercurius_is_synchronized": int(simulation.ri_mercurius.is_synchronized),
        "mercurius_recalculate_coordinates_this_timestep": int(
            simulation.ri_mercurius.recalculate_coordinates_this_timestep
        ),
    }
    digest.update(engine.canonical_bytes(configuration))
    for index, particle in enumerate(simulation.particles):
        digest.update(
            struct.pack(
                "!II8d",
                index,
                int(particle.hash.value),
                particle.m,
                particle.r,
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            )
        )
    return digest.hexdigest()


def hardened_build(*args: Any, **kwargs: Any) -> tuple[Any, int, list[str]]:
    simulation, tracer_start, common_names = ORIGINAL_BUILD(*args, **kwargs)
    contract = args[0]
    required = float(contract["dynamics"]["mercurius_hillfac"])
    simulation.ri_mercurius.r_crit_hill = required
    if float(simulation.ri_mercurius.r_crit_hill) != required:
        raise RuntimeError("MERCURIUS r_crit_hill readback mismatch")
    return simulation, tracer_start, common_names


def hardened_run_arm(
    contract: dict[str, Any],
    output_dir: Path,
    label: str,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    record = ORIGINAL_RUN_ARM(contract, output_dir, label, *args, **kwargs)
    checkpoint = output_dir / "checkpoints" / f"{label}.bin"
    decoded = decoded_state_digest(rebound.Simulation(str(checkpoint)))
    container = record.pop("checkpoint_sha256")
    record["checkpoint_decoded_state_sha256"] = decoded
    record["_checkpoint_container_sha256"] = container
    record["mercurius_r_crit_hill"] = float(contract["dynamics"]["mercurius_hillfac"])
    CAPTURED_RECORDS.append(record)
    return record


def hardened_public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in ORIGINAL_PUBLIC_RECORD(record).items()
        if key != "checkpoint_sha256"
    }


def validate_contract_v3(contract: dict[str, Any], contract_path: Path) -> None:
    if contract.get("schema") != "jx-e1-synthetic-engineering-pilot-contract/v3":
        raise ValueError("unexpected hardened contract schema")
    if contract.get("experiment_id") != "jx-e1-p9-9x4-smoke-v3":
        raise ValueError("unexpected hardened experiment ID")
    expected_correction = {
        "base_contract_sha256": engine.sha256_file(BASE_CONTRACT),
        "base_replay_receipt_sha256": engine.sha256_file(BASE_REPLAY),
        "base_runner_sha256": engine.sha256_file(BASE_RUNNER),
        "reason": "The prior runner used a Python-only hillfac attribute instead of REBOUND's r_crit_hill field, and its deterministic payload included noncanonical checkpoint-container hashes.",
        "effect_on_prior_numerics": "The silent hillfac no-op left REBOUND's default r_crit_hill at 3.0, which equaled the prior contract; decoded prior states remain internally consistent, but enforcement was inadequate.",
        "remedy": "Set and read back r_crit_hill, hash decoded configuration plus state, and exclude raw checkpoint-container hashes and timing from the deterministic payload."
    }
    if contract.get("correction") != expected_correction:
        raise ValueError("hardened correction binding changed")
    if contract["runtime"].get("wrapper_sha256") != engine.sha256_file(Path(__file__).resolve()):
        raise ValueError("hardened wrapper hash mismatch")
    compatibility = copy.deepcopy(contract)
    compatibility["schema"] = "jx-e1-synthetic-engineering-pilot-contract/v1"
    compatibility["experiment_id"] = "jx-e1-p9-9x4-smoke-v1"
    compatibility["runtime"].pop("wrapper_sha256", None)
    compatibility["runtime"]["runner_sha256"] = engine.sha256_file(BASE_RUNNER)
    ORIGINAL_VALIDATOR(compatibility, Path("contract_v1.json"))
    if contract["timestep_audit"]["tracers"] != 32:
        raise ValueError("the complete 32-particle timestep block is required")


def checkpoint_manifest(output_dir: Path) -> list[dict[str, str]]:
    result = []
    for path in sorted((output_dir / "checkpoints").glob("*.bin")):
        result.append({
            "name": path.name,
            "container_sha256_provenance_only": engine.sha256_file(path),
            "decoded_configuration_state_sha256": decoded_state_digest(rebound.Simulation(str(path))),
        })
    return result


def canonical_semantic(result: dict[str, Any], audit_records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "jx-e1-synthetic-engineering-semantic/v3",
        "experiment_id": result["experiment_id"],
        "claim_ceiling": result["claim_ceiling"],
        "nonclaim": result["nonclaim"],
        "contract_sha256": result["contract_sha256"],
        "runtime": {
            key: value
            for key, value in result["runtime"].items()
            if key != "rebound_binary_path"
        },
        "design_counts": result["design_counts"],
        "checks": result["checks"],
        "maximum_observed_drifts": result["maximum_observed_drifts"],
        "timestep_audit": result["timestep_audit"],
        "primary_run_records": [
            {key: value for key, value in record.items() if key != "elapsed_seconds"}
            for record in result["run_records"]
        ],
        "audit_run_records": [
            {key: value for key, value in hardened_public_record(record).items() if key != "elapsed_seconds"}
            for record in audit_records
        ],
        "correction": result["correction"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    contract = engine.strict_json(contract_path)
    validate_contract_v3(contract, contract_path)
    runtime = engine.validate_runtime(contract)
    if arguments.validate_only:
        simulation, _, _ = hardened_build(contract, engine.make_tracers(contract, 0), None, None, contract["dynamics"]["dt_years"])
        print(json.dumps({
            "contract_sha256": engine.sha256_file(contract_path),
            "runtime": runtime,
            "r_crit_hill_readback": simulation.ri_mercurius.r_crit_hill,
            "initial_decoded_state_sha256": decoded_state_digest(simulation),
        }, indent=2))
        return 0

    original = (engine.validate_contract, engine.build_simulation, engine.simulation_digest, engine.run_arm, engine.public_record)
    CAPTURED_RECORDS.clear()
    try:
        engine.validate_contract = validate_contract_v3
        engine.build_simulation = hardened_build
        engine.simulation_digest = decoded_state_digest
        engine.run_arm = hardened_run_arm
        engine.public_record = hardened_public_record
        result = engine.execute(contract_path, arguments.output_dir.resolve())
    finally:
        (
            engine.validate_contract,
            engine.build_simulation,
            engine.simulation_digest,
            engine.run_arm,
            engine.public_record,
        ) = original
    result["schema"] = "jx-e1-synthetic-engineering-pilot-result/v3"
    result["correction"] = contract["correction"]
    audit_records = [record for record in CAPTURED_RECORDS if record["label"].startswith("dt-half-")]
    gates = contract["gates"]
    result["checks"].update({
        "all_audit_sampled_and_endpoint_states_finite": all(
            record["diagnostics"]["all_sampled_states_finite"]
            and record["all_endpoint_states_finite"]
            for record in audit_records
        ),
        "all_audit_checkpoint_serialization_state_exact": all(
            record["checkpoint_serialization_state_exact"] for record in audit_records
        ),
        "all_audit_checkpoint_continuation_state_exact": all(
            record["checkpoint_continuation_state_exact"] for record in audit_records
        ),
        "audit_active_energy_drift_within_gate": max(
            record["maximum_relative_active_energy_drift"] for record in audit_records
        ) <= float(gates["max_relative_active_energy_drift"]),
        "audit_active_angular_momentum_drift_within_gate": max(
            record["maximum_relative_active_angular_momentum_vector_drift"]
            for record in audit_records
        ) <= float(gates["max_relative_active_angular_momentum_vector_drift"]),
        "audit_common_initial_states_match_primary": all(
            record["initial_common_state_sha256"]
            == next(
                primary["initial_common_state_sha256"]
                for primary in CAPTURED_RECORDS
                if not primary["label"].startswith("dt-half-")
                and (
                    (record["model_id"] is None and primary["model_id"] is None)
                    or (
                        record["model_id"] == primary["model_id"]
                        and record["angle_id"] == primary["angle_id"]
                    )
                )
                and record["block"] == primary["block"]
            )
            for record in audit_records
        ),
    })
    result["verdict"] = "ENGINEERING_SMOKE_VALID" if all(result["checks"].values()) else "ENGINEERING_SMOKE_INVALID"
    result["audit_run_records"] = [hardened_public_record(record) for record in audit_records]
    result["maximum_observed_drifts"]["relative_active_energy_including_audits"] = max(
        record["maximum_relative_active_energy_drift"] for record in CAPTURED_RECORDS
    )
    result["maximum_observed_drifts"]["relative_active_angular_momentum_vector_including_audits"] = max(
        record["maximum_relative_active_angular_momentum_vector_drift"] for record in CAPTURED_RECORDS
    )
    result["semantic"] = canonical_semantic(result, audit_records)
    result["semantic_sha256"] = engine.sha256_bytes(engine.canonical_bytes(result["semantic"]))
    result["checkpoint_container_manifest"] = checkpoint_manifest(arguments.output_dir.resolve())
    result_path = arguments.output_dir.resolve() / "result_v1.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": result["verdict"],
        "deterministic_payload_sha256": result["deterministic_payload_sha256"],
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(result_path),
    }, indent=2))
    return 0 if result["verdict"] == "ENGINEERING_SMOKE_VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
