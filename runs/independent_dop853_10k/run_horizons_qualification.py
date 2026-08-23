#!/usr/bin/env python3
"""Qualify the independent DOP853 force path against JPL Horizons/DE441."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from jxplanetx.independent_dop853 import (  # noqa: E402
    DOP853,
    _active_invariants,
    _atomic_csv,
    _atomic_json,
    _read_json,
    _rhs_factory,
    _runtime_manifest,
    _sha256_file,
)


SCHEMA = "jx-independent-dop853-horizons-contract/v1"
RESULT_SCHEMA = "jx-independent-dop853-horizons-result/v1"
AU_KM = 149597870.700
SECONDS_PER_DAY = 86400.0


def resolve(contract_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (contract_path.parent / path).resolve()


def load_initial(path: Path) -> tuple[list[int], list[str], np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    first_epoch = min(float(row["jd_tdb"]) for row in rows)
    selected = [row for row in rows if float(row["jd_tdb"]) == first_epoch and row["run"] == "coarse"]
    selected.sort(key=lambda row: int(row["body_id"]))
    if len(selected) != 10:
        raise ValueError("initial JX state must contain ten bodies")
    body_ids = [int(row["body_id"]) for row in selected]
    names = [row["body_name"] for row in selected]
    positions = np.array([[float(row[field]) for field in ("x", "y", "z")] for row in selected])
    velocities = np.array([[float(row[field]) for field in ("vx", "vy", "vz")] for row in selected])
    return body_ids, names, positions, velocities


def load_masses(path: Path, body_ids: list[int]) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        table = {int(row["body_id"]): float(row["gm_km3_s2"]) for row in csv.DictReader(stream)}
    return np.array(
        [table[body_id] * SECONDS_PER_DAY**2 / AU_KM**3 for body_id in body_ids],
        dtype=np.float64,
    )


def load_reference(path: Path) -> dict[tuple[int, float], tuple[np.ndarray, np.ndarray]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            result[(int(row["body_id"]), float(row["jd_tdb"]))] = (
                np.array([float(row[field]) for field in ("x_au", "y_au", "z_au")]),
                np.array(
                    [
                        float(row[field])
                        for field in ("vx_au_per_day", "vy_au_per_day", "vz_au_per_day")
                    ]
                ),
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--residuals", type=Path, required=True)
    arguments = parser.parse_args()
    contract_path = arguments.contract.resolve()
    output = arguments.output.resolve()
    residual_path = arguments.residuals.resolve()
    if output.exists() or residual_path.exists():
        raise FileExistsError("refusing to overwrite locked qualification output")
    contract = _read_json(contract_path)
    if contract.get("schema") != SCHEMA or contract.get("registration_status") != "PRELOCKED":
        raise ValueError("invalid independent Horizons qualification contract")
    verified = {}
    for label, specification in contract["locked_files"].items():
        path = resolve(contract_path, specification["path"])
        observed = _sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(f"locked input mismatch for {label}")
        verified[label] = {"path": str(path), "sha256": observed}
    runtime = _runtime_manifest(contract_path, contract)
    initial_path = resolve(contract_path, contract["inputs"]["initial_state"])
    mass_path = resolve(contract_path, contract["inputs"]["gm_table"])
    reference_path = resolve(contract_path, contract["inputs"]["reference_vectors"])
    body_ids, names, positions, velocities = load_initial(initial_path)
    masses = load_masses(mass_path, body_ids)
    reference = load_reference(reference_path)
    first_jd = min(epoch for _body, epoch in reference)
    epochs = sorted({epoch for _body, epoch in reference})
    if len(epochs) != 11:
        raise ValueError("reference epoch grid is incomplete")
    state = np.concatenate((positions.ravel(), velocities.ravel()))
    rhs = _rhs_factory(masses, 0)
    initial_energy, initial_angular = _active_invariants(state, masses)
    angular_norm = float(np.linalg.norm(initial_angular))
    maximum_energy = maximum_angular = 0.0
    accepted_steps = rhs_evaluations = 0
    states = {first_jd: np.array(state, copy=True)}
    current_time = 0.0
    started = time.perf_counter()
    solver_settings = contract["solver"]
    for epoch in epochs[1:]:
        target_time = epoch - first_jd
        solver = DOP853(
            rhs,
            current_time,
            state,
            target_time,
            rtol=float(solver_settings["rtol"]),
            atol=float(solver_settings["atol"]),
            max_step=float(solver_settings["max_step_days"]),
        )
        while solver.status == "running":
            message = solver.step()
            if solver.status == "failed":
                raise RuntimeError(f"DOP853 Horizons qualification failed: {message}")
            accepted_steps += 1
            energy, angular = _active_invariants(solver.y, masses)
            maximum_energy = max(maximum_energy, abs((energy - initial_energy) / initial_energy))
            maximum_angular = max(
                maximum_angular,
                float(np.linalg.norm(angular - initial_angular)) / angular_norm,
            )
        rhs_evaluations += int(solver.nfev)
        state = np.asarray(solver.y, dtype=np.float64)
        current_time = target_time
        states[epoch] = np.array(state, copy=True)
    residual_rows = []
    maximum_outer_position = maximum_outer_velocity = 0.0
    sun_index = body_ids.index(10)
    for epoch in epochs:
        epoch_state = states[epoch]
        model_position = epoch_state[:30].reshape(10, 3)
        model_velocity = epoch_state[30:].reshape(10, 3)
        reference_sun_position, reference_sun_velocity = reference[(10, epoch)]
        for index, body_id in enumerate(body_ids):
            reference_position, reference_velocity = reference[(body_id, epoch)]
            position_residual = float(
                np.linalg.norm(
                    (model_position[index] - model_position[sun_index])
                    - (reference_position - reference_sun_position)
                )
            )
            velocity_residual = float(
                np.linalg.norm(
                    (model_velocity[index] - model_velocity[sun_index])
                    - (reference_velocity - reference_sun_velocity)
                )
            )
            residual_rows.append(
                {
                    "jd_tdb": epoch,
                    "body_id": body_id,
                    "body_name": names[index],
                    "heliocentric_position_residual_AU": position_residual,
                    "heliocentric_velocity_residual_AU_per_day": velocity_residual,
                }
            )
            if body_id in (5, 6, 7, 8):
                maximum_outer_position = max(maximum_outer_position, position_residual)
                maximum_outer_velocity = max(maximum_outer_velocity, velocity_residual)
    _atomic_csv(
        residual_path,
        residual_rows,
        (
            "jd_tdb",
            "body_id",
            "body_name",
            "heliocentric_position_residual_AU",
            "heliocentric_velocity_residual_AU_per_day",
        ),
    )
    gates = contract["gates"]
    checks = {
        "complete_epoch_grid": len(states) == len(epochs),
        "outer_position_residual": maximum_outer_position
        <= float(gates["max_outer_position_residual_AU"]),
        "outer_velocity_residual": maximum_outer_velocity
        <= float(gates["max_outer_velocity_residual_AU_per_day"]),
        "energy_drift": maximum_energy <= float(gates["max_relative_energy_drift"]),
        "angular_momentum_drift": maximum_angular
        <= float(gates["max_relative_angular_momentum_vector_drift"]),
        "finite_endpoint": bool(np.all(np.isfinite(state))),
    }
    verdict = "PASSED" if all(checks.values()) else "INVALID"
    result = {
        "schema": RESULT_SCHEMA,
        "verdict": verdict,
        "contract_path": str(contract_path),
        "contract_sha256": _sha256_file(contract_path),
        "classification": "EXTERNAL_DE441_COMPATIBILITY_INDEPENDENT_OF_REBOUND",
        "runtime": runtime,
        "verified_locked_files": verified,
        "accepted_steps": accepted_steps,
        "rhs_evaluations": rhs_evaluations,
        "maximum_outer_position_residual_AU": maximum_outer_position,
        "maximum_outer_velocity_residual_AU_per_day": maximum_outer_velocity,
        "maximum_outer_position_residual_km": maximum_outer_position * AU_KM,
        "maximum_outer_velocity_residual_m_per_s": maximum_outer_velocity
        * AU_KM
        * 1000.0
        / SECONDS_PER_DAY,
        "maximum_relative_energy_drift": maximum_energy,
        "maximum_relative_angular_momentum_vector_drift": maximum_angular,
        "checks": checks,
        "all_gates_passed": verdict == "PASSED",
        "residual_csv": {"path": str(residual_path), "sha256": _sha256_file(residual_path)},
        "elapsed_seconds": time.perf_counter() - started,
        "science_status": "SCREENING_ONLY",
        "nonclaim": contract["nonclaim"],
    }
    _atomic_json(output, result)
    print(json.dumps({"verdict": verdict, "result": str(output)}))
    return 0 if verdict == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
