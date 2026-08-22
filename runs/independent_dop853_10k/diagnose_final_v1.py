#!/usr/bin/env python3
"""Diagnose the locked v1 endpoint-consistency failure without changing it."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from jxplanetx.independent_dop853 import (  # noqa: E402
    _active_only_audit,
    _atomic_json,
    _load_state,
    _sha256_file,
)


def _resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base.parent / candidate).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _endpoint(record: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(
        [
            [*row["position_AU"], *row["velocity_AU_per_year"]]
            for row in record[key]
        ],
        dtype=np.float64,
    )


def _distance(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape:
        raise ValueError("active endpoint shapes differ")
    return {
        "maximum_position_AU": float(np.linalg.norm(left[:, :3] - right[:, :3], axis=1).max()),
        "maximum_velocity_AU_per_year": float(
            np.linalg.norm(left[:, 3:] - right[:, 3:], axis=1).max()
        ),
    }


def _run_refined(task: tuple[str, str, float, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    arm, state_path, duration, solver = task
    return arm, _active_only_audit(_load_state(Path(state_path)), duration, solver)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    output = arguments.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic result: {output}")
    contract = _read_json(contract_path)
    if contract.get("schema") != "jx-independent-dop853-v1-diagnostic-contract/v1":
        raise ValueError("unexpected diagnostic contract schema")
    verified_files: dict[str, dict[str, str]] = {}
    for label, specification in contract["locked_files"].items():
        path = _resolve(contract_path, specification["path"])
        observed = _sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(f"locked hash mismatch for {label}: {observed}")
        verified_files[label] = {"path": str(path), "sha256": observed}

    result_path = _resolve(contract_path, contract["failed_result"]["path"])
    result = _read_json(result_path)
    if _sha256_file(result_path) != contract["failed_result"]["sha256"]:
        raise ValueError("failed result hash mismatch")
    if result["verdict"] != "INVALID":
        raise ValueError("diagnostic input is not the locked INVALID result")
    failed_checks = sorted(key for key, value in result["numerical_checks"].items() if not value)
    if failed_checks != ["active_endpoint_position_consistency"]:
        raise ValueError(f"unexpected numerical failure set: {failed_checks}")

    summaries: dict[str, list[tuple[int, np.ndarray]]] = {"control": [], "source": []}
    verified_summaries = []
    for record in result["block_records"]:
        summary_path = Path(record["summary_json"]).resolve()
        observed = _sha256_file(summary_path)
        if observed != record["summary_json_sha256"]:
            raise ValueError(f"summary hash mismatch: {summary_path}")
        summary = _read_json(summary_path)
        arm = str(record["arm"])
        block = int(record["block_index"])
        summaries[arm].append((block, _endpoint(summary, "active_endpoint_state")))
        verified_summaries.append(
            {"arm": arm, "block_index": block, "path": str(summary_path), "sha256": observed}
        )
    if any(len(records) != 10 for records in summaries.values()):
        raise ValueError("expected ten block summaries in each arm")

    pairwise: dict[str, dict[str, Any]] = {}
    for arm, records in summaries.items():
        maximum_position = (-1.0, None)
        maximum_velocity = (-1.0, None)
        for left_index in range(len(records)):
            for right_index in range(left_index + 1, len(records)):
                values = _distance(records[left_index][1], records[right_index][1])
                pair = [records[left_index][0], records[right_index][0]]
                if values["maximum_position_AU"] > maximum_position[0]:
                    maximum_position = (values["maximum_position_AU"], pair)
                if values["maximum_velocity_AU_per_year"] > maximum_velocity[0]:
                    maximum_velocity = (values["maximum_velocity_AU_per_year"], pair)
        pairwise[arm] = {
            "maximum_position_AU": maximum_position[0],
            "maximum_position_block_pair": maximum_position[1],
            "maximum_velocity_AU_per_year": maximum_velocity[0],
            "maximum_velocity_block_pair": maximum_velocity[1],
        }

    duration = float(contract["refinement"]["duration_years"])
    solver = contract["refinement"]["solver"]
    state_tasks = []
    for arm in ("control", "source"):
        state_path = _resolve(contract_path, contract["refinement"]["states"][arm])
        state_tasks.append((arm, str(state_path), duration, solver))
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=2) as executor:
        refined = dict(executor.map(_run_refined, state_tasks))

    comparisons: dict[str, Any] = {}
    for arm in ("control", "source"):
        baseline = _endpoint(result["active_only_audits"][arm], "endpoint_state")
        refined_endpoint = _endpoint(refined[arm], "endpoint_state")
        block_distances = [
            {
                "block_index": block,
                **_distance(endpoint, refined_endpoint),
            }
            for block, endpoint in summaries[arm]
        ]
        comparisons[arm] = {
            "baseline_active_only_to_refined": _distance(baseline, refined_endpoint),
            "maximum_block_to_refined_position_AU": max(
                row["maximum_position_AU"] for row in block_distances
            ),
            "maximum_block_to_refined_velocity_AU_per_year": max(
                row["maximum_velocity_AU_per_year"] for row in block_distances
            ),
            "block_to_refined": block_distances,
        }

    refined_energy = max(
        float(record["maximum_relative_energy_drift"]) for record in refined.values()
    )
    refined_angular = max(
        float(record["maximum_relative_angular_momentum_vector_drift"])
        for record in refined.values()
    )
    diagnostic_checks = {
        "only_v1_failure_is_endpoint_position": True,
        "all_v1_cross_software_checks_pass": all(result["cross_software_checks"].values()),
        "tracer_loaded_active_endpoints_are_internally_consistent": max(
            record["maximum_position_AU"] for record in pairwise.values()
        )
        <= float(contract["diagnostic_gates"]["max_pairwise_block_position_AU"]),
        "refined_active_energy_conservation": refined_energy
        <= float(contract["diagnostic_gates"]["max_refined_relative_energy_drift"]),
        "refined_active_angular_conservation": refined_angular
        <= float(contract["diagnostic_gates"]["max_refined_relative_angular_drift"]),
    }
    diagnosis = (
        "ADAPTIVE_RESOLUTION_FAILURE_CONFIRMED"
        if all(diagnostic_checks.values())
        else "DIAGNOSIS_INCONCLUSIVE"
    )
    payload = {
        "schema": "jx-independent-dop853-v1-diagnostic-result/v1",
        "diagnosis": diagnosis,
        "contract_path": str(contract_path),
        "contract_sha256": _sha256_file(contract_path),
        "failed_result_path": str(result_path),
        "failed_result_sha256": _sha256_file(result_path),
        "v1_verdict_remains": "INVALID",
        "failed_numerical_checks": failed_checks,
        "v1_cross_software_checks": result["cross_software_checks"],
        "pairwise_tracer_loaded_endpoint_dispersion": pairwise,
        "refined_active_only_audits": refined,
        "comparisons_to_refined_active_only": comparisons,
        "maximum_refined_relative_energy_drift": refined_energy,
        "maximum_refined_relative_angular_momentum_vector_drift": refined_angular,
        "diagnostic_checks": diagnostic_checks,
        "verified_locked_files": verified_files,
        "verified_block_summaries": verified_summaries,
        "elapsed_seconds": time.perf_counter() - started,
        "recommended_corrective_action": contract["recommended_corrective_action"],
        "interpretation": "This post-failure diagnosis may motivate a new prelocked rerun, but it cannot change the immutable v1 INVALID verdict or retroactively relax its gate."
    }
    _atomic_json(output, payload)
    print(json.dumps({"diagnosis": diagnosis, "output": str(output)}), flush=True)
    return 0 if diagnosis == "ADAPTIVE_RESOLUTION_FAILURE_CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
