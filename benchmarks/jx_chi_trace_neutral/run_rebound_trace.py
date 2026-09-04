#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rebound

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTRACT_PATH = HERE / "contract.json"
OUTPUT = ROOT / "runs" / "jx_chi_trace_neutral"
RAW = OUTPUT / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_simulation() -> rebound.Simulation:
    star_m = 1.0
    planet_m = 0.01 / (star_m - 0.01)
    planet_a = 5.2
    star_x = -(planet_m / (star_m + planet_m)) * planet_a
    circular_speed = math.sqrt((star_m + planet_m) / planet_a)
    star_vy = -(planet_m / (star_m + planet_m)) * circular_speed
    planet_x = (star_m / (star_m + planet_m)) * planet_a
    planet_vy = (star_m / (star_m + planet_m)) * circular_speed
    tracer_x_offset = 4.42
    tracer_vy = 0.0072 * 365.25 / (2.0 * math.pi)

    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.add(m=star_m, x=star_x, y=0.0, z=0.0, vx=0.0, vy=star_vy, vz=0.0)
    simulation.add(m=planet_m, x=planet_x, y=0.0, z=0.0, vx=0.0, vy=planet_vy, vz=0.0)
    simulation.add(m=0.0, x=tracer_x_offset + star_x, y=0.0, z=0.0, vx=0.0, vy=tracer_vy + star_vy, vz=0.0)
    return simulation


def state_array(simulation: rebound.Simulation) -> np.ndarray:
    rows: list[float] = []
    for particle in simulation.particles:
        rows.extend((particle.x, particle.y, particle.z, particle.vx, particle.vy, particle.vz))
    return np.asarray(rows, dtype=np.float64)


def jacobi_constant(simulation: rebound.Simulation) -> float:
    star, planet, tracer = simulation.particles[0], simulation.particles[1], simulation.particles[2]
    r_star = np.array((star.x, star.y, star.z), dtype=float)
    r_planet = np.array((planet.x, planet.y, planet.z), dtype=float)
    r = np.array((tracer.x, tracer.y, tracer.z), dtype=float)
    v = np.array((tracer.vx, tracer.vy, tracer.vz), dtype=float)
    kinetic = 0.5 * float(v @ v)
    r1 = r - r_star
    r2 = r - r_planet
    potential = -simulation.G * star.m / float(np.linalg.norm(r1)) - simulation.G * planet.m / float(np.linalg.norm(r2))
    mean_motion = math.sqrt(simulation.G * (star.m + planet.m) / (5.2**3))
    angular_momentum_z = float(np.cross(r, v)[2])
    return 2.0 * mean_motion * angular_momentum_z - 2.0 * (kinetic + potential)


def snapshot(simulation: rebound.Simulation, index: int) -> dict[str, float | int]:
    planet, tracer = simulation.particles[1], simulation.particles[2]
    separation = math.sqrt((planet.x-tracer.x)**2 + (planet.y-tracer.y)**2 + (planet.z-tracer.z)**2)
    row: dict[str, float | int] = {
        "sample_index": index,
        "time": float(simulation.t),
        "system_energy": float(simulation.energy()),
        "jacobi": jacobi_constant(simulation),
        "planet_tracer_separation": separation,
    }
    for body_index, particle in enumerate(simulation.particles):
        prefix = f"b{body_index}_"
        row[prefix + "x"] = float(particle.x)
        row[prefix + "y"] = float(particle.y)
        row[prefix + "z"] = float(particle.z)
        row[prefix + "vx"] = float(particle.vx)
        row[prefix + "vy"] = float(particle.vy)
        row[prefix + "vz"] = float(particle.vz)
    return row


def write_trajectory(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def configure_trace(simulation: rebound.Simulation, dt: float) -> None:
    simulation.integrator = "trace"
    simulation.integrator.peri_mode = 1
    simulation.integrator.r_crit_hill = 3.63
    simulation.integrator.peri_crit_eta = 1.0
    simulation.dt = dt


def run_trace(steps: int, label: str) -> dict[str, Any]:
    simulation = build_simulation()
    dt = 100.0 / steps
    configure_trace(simulation, dt)
    rows = [snapshot(simulation, 0)]
    start = time.perf_counter()
    for index in range(1, steps + 1):
        simulation.steps(1)
        rows.append(snapshot(simulation, index))
    wall = time.perf_counter() - start
    path = RAW / f"trace_{label}.csv"
    write_trajectory(path, rows)
    state_keys = [key for key in rows[0] if key.startswith("b")]
    array = np.asarray([[row[key] for key in state_keys] for row in rows], dtype=np.float64)
    separations = np.asarray([row["planet_tracer_separation"] for row in rows], dtype=float)
    minimum_index = int(np.argmin(separations))
    initial_jacobi = float(rows[0]["jacobi"])
    jacobi_errors = np.abs(np.asarray([row["jacobi"] for row in rows], dtype=float) / initial_jacobi - 1.0)
    initial_energy = float(rows[0]["system_energy"])
    energy_errors = np.abs(np.asarray([row["system_energy"] for row in rows], dtype=float) / initial_energy - 1.0)
    return {
        "label": label,
        "steps": steps,
        "dt": dt,
        "actual_final_time": float(simulation.t),
        "wall_seconds": wall,
        "trajectory_path": str(path.relative_to(OUTPUT)),
        "trajectory_sha256": sha256(path),
        "state_array_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "final_state": state_array(simulation).tolist(),
        "sampled_minimum_separation": float(separations[minimum_index]),
        "sampled_minimum_time": float(rows[minimum_index]["time"]),
        "maximum_relative_jacobi_error": float(np.max(jacobi_errors)),
        "final_relative_jacobi_error": float(jacobi_errors[-1]),
        "maximum_relative_system_energy_error": float(np.max(energy_errors)),
    }


def run_ias15(name: str, epsilon: float, output_steps: int) -> dict[str, Any]:
    simulation = build_simulation()
    simulation.integrator = "ias15"
    simulation.integrator.epsilon = epsilon
    rows = [snapshot(simulation, 0)]
    targets = np.linspace(0.0, 100.0, output_steps + 1)
    start = time.perf_counter()
    for index, target in enumerate(targets[1:], 1):
        simulation.integrate(float(target), exact_finish_time=1)
        rows.append(snapshot(simulation, index))
    wall = time.perf_counter() - start
    path = RAW / f"{name}.csv"
    write_trajectory(path, rows)
    separations = np.asarray([row["planet_tracer_separation"] for row in rows], dtype=float)
    minimum_index = int(np.argmin(separations))
    initial_jacobi = float(rows[0]["jacobi"])
    jacobi_errors = np.abs(np.asarray([row["jacobi"] for row in rows], dtype=float) / initial_jacobi - 1.0)
    return {
        "name": name,
        "epsilon": epsilon,
        "output_steps": output_steps,
        "wall_seconds": wall,
        "trajectory_path": str(path.relative_to(OUTPUT)),
        "trajectory_sha256": sha256(path),
        "final_state": state_array(simulation).tolist(),
        "sampled_minimum_separation": float(separations[minimum_index]),
        "sampled_minimum_time": float(rows[minimum_index]["time"]),
        "maximum_relative_jacobi_error": float(np.max(jacobi_errors)),
        "final_relative_jacobi_error": float(jacobi_errors[-1]),
    }


def load_state_columns(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = [key for key in rows[0] if key.startswith("b")]
    return np.asarray([[float(row[key]) for key in keys] for row in rows], dtype=np.float64)


def reversal_trace(steps: int) -> dict[str, Any]:
    simulation = build_simulation()
    initial = state_array(simulation)
    dt = 100.0 / steps
    configure_trace(simulation, dt)
    start = time.perf_counter()
    for _ in range(steps):
        simulation.steps(1)
    for particle in simulation.particles:
        particle.vx = -particle.vx
        particle.vy = -particle.vy
        particle.vz = -particle.vz
    for _ in range(steps):
        simulation.steps(1)
    for particle in simulation.particles:
        particle.vx = -particle.vx
        particle.vy = -particle.vy
        particle.vz = -particle.vz
    wall = time.perf_counter() - start
    returned = state_array(simulation)
    difference = np.abs(returned - initial)
    position_indices = [0,1,2,6,7,8,12,13,14]
    velocity_indices = [3,4,5,9,10,11,15,16,17]
    return {
        "steps_each_direction": steps,
        "dt": dt,
        "wall_seconds": wall,
        "maximum_position_return_error": float(np.max(difference[position_indices])),
        "maximum_velocity_return_error": float(np.max(difference[velocity_indices])),
        "returned_state": returned.tolist(),
    }


def main() -> int:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if rebound.__version__ != "5.1.1":
        raise RuntimeError(f"expected rebound 5.1.1, observed {rebound.__version__}")
    if OUTPUT.exists():
        import shutil
        shutil.rmtree(OUTPUT)
    RAW.mkdir(parents=True)

    trace_runs: list[dict[str, Any]] = []
    deterministic: list[dict[str, Any]] = []
    for steps in contract["schedule"]["global_step_counts"]:
        first = run_trace(int(steps), f"n{steps}_replay1")
        second = run_trace(int(steps), f"n{steps}_replay2")
        trace_runs.append(first)
        deterministic.append({
            "steps": int(steps),
            "trajectory_bytes_identical": first["trajectory_sha256"] == second["trajectory_sha256"],
            "state_arrays_identical": first["state_array_sha256"] == second["state_array_sha256"],
            "first_trajectory_sha256": first["trajectory_sha256"],
            "second_trajectory_sha256": second["trajectory_sha256"],
        })

    references = [run_ias15(item["name"], float(item["epsilon"]), 1600) for item in contract["ias15_references"]]
    loose = load_state_columns(OUTPUT / references[0]["trajectory_path"])
    tight = load_state_columns(OUTPUT / references[1]["trajectory_path"])
    pair_difference = np.abs(loose - tight)
    position_columns = [i for i in range(pair_difference.shape[1]) if i % 6 < 3]
    velocity_columns = [i for i in range(pair_difference.shape[1]) if i % 6 >= 3]
    reference_position_difference = float(np.max(pair_difference[:, position_columns]))
    reference_velocity_difference = float(np.max(pair_difference[:, velocity_columns]))
    reference_valid = (
        reference_position_difference <= contract["reference_gate"]["maximum_ias15_pair_position_difference"]
        and reference_velocity_difference <= contract["reference_gate"]["maximum_ias15_pair_velocity_difference"]
    )
    replay_valid = all(item["trajectory_bytes_identical"] and item["state_arrays_identical"] for item in deterministic)

    if not reference_valid:
        verdict = "INVALID_REFERENCE"
    elif not replay_valid:
        verdict = "NONDETERMINISTIC_TRACE_REPLAY"
    else:
        verdict = "REBOUND_TRACE_NEUTRAL_ARTIFACT_COMPLETE"

    result = {
        "schema": "jx-chi-vs-trace-neutral-rebound-result/v1",
        "classification": contract["classification"],
        "verdict": verdict,
        "contract_sha256": sha256(CONTRACT_PATH),
        "rebound_version": rebound.__version__,
        "trace_runs": trace_runs,
        "deterministic_replays": deterministic,
        "ias15_references": references,
        "ias15_pair_maximum_position_difference": reference_position_difference,
        "ias15_pair_maximum_velocity_difference": reference_velocity_difference,
        "ias15_reference_valid": reference_valid,
        "trace_replay_valid": replay_valid,
        "velocity_flip_reversal": reversal_trace(int(contract["schedule"]["reversal_step_count"])),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "rebound": rebound.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "claim_ceiling": contract["claim_ceiling"],
    }
    result_path = OUTPUT / "REBOUND_TRACE_RESULT.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    (OUTPUT / "environment.json").write_text(json.dumps(result["environment"], indent=2, sort_keys=True) + "\n", encoding="utf-8")

    checksum_lines = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(OUTPUT)}")
    (OUTPUT / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "verdict": verdict,
        "reference_valid": reference_valid,
        "trace_replay_valid": replay_valid,
        "result_sha256": sha256(result_path),
    }, indent=2, sort_keys=True))
    return 0 if verdict == "REBOUND_TRACE_NEUTRAL_ARTIFACT_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
