#!/usr/bin/env python3
"""Post-failure, no-new-dynamics audit of stored JX-E1 long-A checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from pathlib import Path
from typing import Any

import rebound

import run_long_pilot as long


EXPECTED = {
    "contract_sha256": "8ac4b0a54418e2ce1d4ce13e9c986177086ec433c900b6060537f15bf97a1187",
    "runner_sha256": "a5f4d74e0a9c03c28fdb1a9a54d8c4be676df2c5caca7325570e9049e73500d9",
    "verifier_sha256": "2dd3ab2651e20d2452cf21ba6df8cb41b47bc35f9b5879ca07e28ba6c9c7bece",
    "result_sha256": "d121033d412a8cd3739b79827e506e895a616d735dde8eb4769d1617e05fc559",
    "result_semantic_sha256": "beb1a2d5756b3c4ca565009b2e24922541565504cb6dad52276392088532b479",
}


def norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(math.fsum(value * value for value in vector))


def relative_vector_drift(
    current: tuple[float, float, float],
    initial: tuple[float, float, float],
) -> float:
    difference = tuple(current[index] - initial[index] for index in range(3))
    return norm(difference) / max(norm(initial), float.fromhex("0x1.0p-1022"))


def cross(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def active_invariants(simulation: Any) -> dict[str, tuple[float, float, float]]:
    particles = list(simulation.particles[: simulation.N_active])
    masses = [float(particle.m) for particle in particles]
    total_mass = math.fsum(masses)
    positions = [
        (float(particle.x), float(particle.y), float(particle.z))
        for particle in particles
    ]
    velocities = [
        (float(particle.vx), float(particle.vy), float(particle.vz))
        for particle in particles
    ]
    linear_ordinary = tuple(
        sum(mass * velocity[axis] for mass, velocity in zip(masses, velocities))
        for axis in range(3)
    )
    linear_compensated = tuple(
        math.fsum(mass * velocity[axis] for mass, velocity in zip(masses, velocities))
        for axis in range(3)
    )
    center_position = tuple(
        math.fsum(mass * position[axis] for mass, position in zip(masses, positions))
        / total_mass
        for axis in range(3)
    )
    center_velocity = tuple(
        math.fsum(mass * velocity[axis] for mass, velocity in zip(masses, velocities))
        / total_mass
        for axis in range(3)
    )
    origin_terms = [
        tuple(mass * component for component in cross(position, velocity))
        for mass, position, velocity in zip(masses, positions, velocities)
    ]
    origin_ordinary = tuple(sum(term[axis] for term in origin_terms) for axis in range(3))
    origin_compensated = tuple(
        math.fsum(term[axis] for term in origin_terms) for axis in range(3)
    )
    com_terms = []
    for mass, position, velocity in zip(masses, positions, velocities):
        relative_position = tuple(
            position[axis] - center_position[axis] for axis in range(3)
        )
        relative_velocity = tuple(
            velocity[axis] - center_velocity[axis] for axis in range(3)
        )
        com_terms.append(
            tuple(mass * component for component in cross(relative_position, relative_velocity))
        )
    com_compensated = tuple(
        math.fsum(term[axis] for term in com_terms) for axis in range(3)
    )
    return {
        "linear_ordinary": linear_ordinary,
        "linear_compensated": linear_compensated,
        "angular_origin_ordinary": origin_ordinary,
        "angular_origin_compensated": origin_compensated,
        "angular_com_compensated": com_compensated,
    }


def active_state_sha256(simulation: Any) -> str:
    digest = hashlib.sha256(b"jx-e1-post-failure-active-state/v1\0")
    digest.update(struct.pack("!dII", float(simulation.t), simulation.N, simulation.N_active))
    for index in range(simulation.N_active):
        particle = simulation.particles[index]
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


def audit(contract_path: Path, output_dir: Path, receipt_path: Path) -> dict[str, Any]:
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite audit receipt: {receipt_path}")
    script_path = Path(__file__).resolve()
    verifier_path = script_path.with_name("verify_long_replay.py")
    result_path = output_dir / "result_v1.json"
    actual_bindings = {
        "contract_sha256": long.sha256_file(contract_path),
        "runner_sha256": long.sha256_file(script_path.with_name("run_long_pilot.py")),
        "verifier_sha256": long.sha256_file(verifier_path),
        "result_sha256": long.sha256_file(result_path),
    }
    for key, expected in EXPECTED.items():
        if key == "result_semantic_sha256":
            continue
        if actual_bindings[key] != expected:
            raise ValueError(f"bound artifact changed: {key}")
    contract = long.strict_json(contract_path)
    long.validate_contract(contract, contract_path)
    result = long.strict_json(result_path)
    if result["verdict"] != "ENGINEERING_LONG_INVALID":
        raise ValueError("source result is not the locked invalid result")
    if result["semantic_sha256"] != EXPECTED["result_semantic_sha256"]:
        raise ValueError("source result semantic hash changed")
    if result["semantic_sha256"] != long.sha256_bytes(long.canonical_bytes(result["semantic"])):
        raise ValueError("source result semantic hash does not recompute")
    embedded = {
        item["run_key"]: item
        for item in result["semantic"]["timestep_audit_arms"]
    }
    _, audit_specs = long.build_matrix_specs(contract)
    if len(audit_specs) != 16:
        raise ValueError("audit matrix cardinality changed")
    by_base: dict[str, list[dict[str, Any]]] = {}
    verified_checkpoints = 0
    seen_checkpoint_paths: set[str] = set()
    expected_checkpoint_count = int(
        round(
            float(contract["dynamics"]["duration_years"])
            / float(contract["dynamics"]["checkpoint_cadence_years"])
        )
    )
    for spec in audit_specs:
        run_key = spec["run_key"]
        record_path = long.record_path(output_dir, run_key)
        record = long.strict_json(record_path)
        semantic = record["semantic"]
        if record["semantic_sha256"] != long.sha256_bytes(long.canonical_bytes(semantic)):
            raise ValueError(f"arm semantic hash changed: {run_key}")
        if semantic != embedded[run_key]:
            raise ValueError(f"result no longer embeds arm semantic exactly: {run_key}")
        initial_simulation, _, _ = long.build_simulation(
            contract,
            long.make_tracers(contract, spec["block"]),
            spec["model"],
            spec["angle"],
            spec["dt_years"],
        )
        initial = active_invariants(initial_simulation)
        base_key = spec["primary_run_key"].rsplit("-b", 1)[0]
        arm_row = {
            "run_key": run_key,
            "block": spec["block"],
            "stored_10yr_sampled_max_relative_energy_drift": semantic[
                "maximum_relative_active_energy_drift"
            ],
            "stored_10yr_sampled_max_relative_origin_angular_drift": semantic[
                "maximum_relative_active_angular_momentum_vector_drift"
            ],
            "stored_10yr_sampled_max_relative_linear_drift": semantic[
                "maximum_relative_active_linear_momentum_vector_drift"
            ],
            "stored_failed_checks": sorted(
                key for key, value in semantic["checks"].items() if value is not True
            ),
            "checkpointed_active_state_sha256": [],
            "checkpoint_max_relative_linear_drift_ordinary": 0.0,
            "checkpoint_max_relative_linear_drift_compensated": 0.0,
            "checkpoint_max_relative_origin_angular_drift_ordinary": 0.0,
            "checkpoint_max_relative_origin_angular_drift_compensated": 0.0,
            "checkpoint_max_relative_com_angular_drift_compensated": 0.0,
        }
        semantic_checkpoints = {
            item["checkpoint_index"]: item
            for item in semantic["checkpoint_decoded_states"]
        }
        containers = record["provenance"]["checkpoint_containers"]
        if len(containers) != expected_checkpoint_count:
            raise ValueError(f"checkpoint container count changed: {run_key}")
        if set(semantic_checkpoints) != set(range(1, expected_checkpoint_count + 1)):
            raise ValueError(f"semantic checkpoint indices changed: {run_key}")
        for expected_index, container in enumerate(containers, start=1):
            expected_time = float(contract["dynamics"]["checkpoint_cadence_years"]) * expected_index
            expected_relative_path = (
                f"checkpoints/{run_key}/checkpoint_{expected_index:02d}.bin"
            )
            if (
                container["checkpoint_index"] != expected_index
                or container["time_year"] != expected_time
                or container["relative_path"] != expected_relative_path
            ):
                raise ValueError(f"checkpoint ordering or identity changed: {run_key}")
            if container["relative_path"] in seen_checkpoint_paths:
                raise ValueError(f"duplicate checkpoint path: {container['relative_path']}")
            seen_checkpoint_paths.add(container["relative_path"])
            checkpoint_path = output_dir / container["relative_path"]
            if checkpoint_path.stat().st_size != container["container_bytes"]:
                raise ValueError(f"checkpoint size changed: {checkpoint_path}")
            if long.sha256_file(checkpoint_path) != container[
                "container_sha256_provenance_only"
            ]:
                raise ValueError(f"checkpoint container hash changed: {checkpoint_path}")
            simulation = rebound.Simulation(str(checkpoint_path))
            decoded = long.decoded_state_digest(simulation)
            if decoded != container["decoded_state_sha256"]:
                raise ValueError(f"checkpoint decoded state changed: {checkpoint_path}")
            if decoded != semantic_checkpoints[container["checkpoint_index"]][
                "decoded_state_sha256"
            ]:
                raise ValueError(f"checkpoint semantic binding changed: {checkpoint_path}")
            current = active_invariants(simulation)
            arm_row["checkpointed_active_state_sha256"].append(
                active_state_sha256(simulation)
            )
            for output_key, invariant_key in (
                (
                    "checkpoint_max_relative_linear_drift_ordinary",
                    "linear_ordinary",
                ),
                (
                    "checkpoint_max_relative_linear_drift_compensated",
                    "linear_compensated",
                ),
                (
                    "checkpoint_max_relative_origin_angular_drift_ordinary",
                    "angular_origin_ordinary",
                ),
                (
                    "checkpoint_max_relative_origin_angular_drift_compensated",
                    "angular_origin_compensated",
                ),
                (
                    "checkpoint_max_relative_com_angular_drift_compensated",
                    "angular_com_compensated",
                ),
            ):
                arm_row[output_key] = max(
                    arm_row[output_key],
                    relative_vector_drift(current[invariant_key], initial[invariant_key]),
                )
            verified_checkpoints += 1
        by_base.setdefault(base_key, []).append(arm_row)

    expected_total_checkpoints = len(audit_specs) * expected_checkpoint_count
    if (
        verified_checkpoints != expected_total_checkpoints
        or len(seen_checkpoint_paths) != expected_total_checkpoints
    ):
        raise ValueError("verified checkpoint set is incomplete or non-unique")

    configurations = []
    for base_key in contract["timestep_audit"]["base_run_keys_without_block"]:
        rows = sorted(by_base[base_key], key=lambda item: item["block"])
        if len(rows) != 2:
            raise ValueError(f"missing duplicate tracer-block audit for {base_key}")
        checkpointed_active_states_exact = (
            rows[0]["checkpointed_active_state_sha256"]
            == rows[1]["checkpointed_active_state_sha256"]
        )
        configurations.append({
            "base_configuration": base_key,
            "tracer_block_checkpointed_active_states_exact": (
                checkpointed_active_states_exact
            ),
            "stored_failed_checks": sorted(
                set(rows[0]["stored_failed_checks"] + rows[1]["stored_failed_checks"])
            ),
            "stored_10yr_sampled_max_relative_energy_drift": max(
                row["stored_10yr_sampled_max_relative_energy_drift"] for row in rows
            ),
            "stored_10yr_sampled_max_relative_origin_angular_drift": max(
                row["stored_10yr_sampled_max_relative_origin_angular_drift"] for row in rows
            ),
            "stored_10yr_sampled_max_relative_linear_drift": max(
                row["stored_10yr_sampled_max_relative_linear_drift"] for row in rows
            ),
            "checkpoint_max_relative_linear_drift_ordinary": max(
                row["checkpoint_max_relative_linear_drift_ordinary"] for row in rows
            ),
            "checkpoint_max_relative_linear_drift_compensated": max(
                row["checkpoint_max_relative_linear_drift_compensated"] for row in rows
            ),
            "checkpoint_max_relative_origin_angular_drift_ordinary": max(
                row["checkpoint_max_relative_origin_angular_drift_ordinary"] for row in rows
            ),
            "checkpoint_max_relative_origin_angular_drift_compensated": max(
                row["checkpoint_max_relative_origin_angular_drift_compensated"] for row in rows
            ),
            "checkpoint_max_relative_com_angular_drift_compensated": max(
                row["checkpoint_max_relative_com_angular_drift_compensated"] for row in rows
            ),
        })
    receipt = {
        "schema": "jx-e1-long-post-failure-numerical-audit/v1",
        "experiment_id": contract["experiment_id"],
        "artifact_class": (
            "POST_OUTCOME_DIAGNOSTIC_USING_STORED_CHECKPOINTS_AND_LOCKED_INITIAL_RECONSTRUCTION"
        ),
        "audit_state": "INVALID_RESULT_PRESERVED_AND_CHARACTERIZED",
        "claim_ceiling": "POST_FAILURE_NUMERICAL_CHARACTERIZATION_ONLY",
        "new_dynamics_executed": False,
        "initial_reference": "DETERMINISTIC_LOCKED_T0_RECONSTRUCTION_NO_INTEGRATION",
        "execution_b_authorized_or_started": False,
        "thresholds_changed": False,
        "source_artifacts": {
            **actual_bindings,
            "result_semantic_sha256": result["semantic_sha256"],
            "audit_script_sha256": long.sha256_file(script_path),
        },
        "integrity": {
            "audit_arm_records_verified": len(audit_specs),
            "stored_checkpoints_verified": verified_checkpoints,
            "all_duplicate_tracer_blocks_have_identical_checkpointed_active_states": all(
                item["tracer_block_checkpointed_active_states_exact"]
                for item in configurations
            ),
        },
        "gates_preserved": contract["gates"],
        "configurations": configurations,
        "interpretation": {
            "locked_E1_verdict": "ENGINEERING_LONG_INVALID",
            "failed_distinct_active_configurations": ["CI07-B", "CI09-A", "CI09-D"],
            "diagnostic_hypothesis": (
                "The origin-based angular-momentum overshoot is largely coupled to "
                "center-of-mass motion, while a small linear-momentum drift remains in "
                "the stored double-precision states. This post-outcome diagnosis cannot "
                "replace the locked gate or rehabilitate E1."
            ),
            "next_action": (
                "STOP_E1_NO_B. Any new numerical-method study requires a separately "
                "locked experiment and cannot change this result."
            ),
        },
        "mandatory_nonclaim": contract["mandatory_nonclaim"],
    }
    long.atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = audit(
        arguments.contract.resolve(),
        arguments.output_dir.resolve(),
        arguments.receipt.resolve(),
    )
    print({
        "audit_state": receipt["audit_state"],
        "audit_arm_records_verified": receipt["integrity"]["audit_arm_records_verified"],
        "stored_checkpoints_verified": receipt["integrity"]["stored_checkpoints_verified"],
        "receipt": str(arguments.receipt.resolve()),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
