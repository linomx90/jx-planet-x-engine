"""Support functions for the native C++ JX Benchmark A replay."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_REBOUND = "5.1.1"
EXPECTED_BODY_IDS = tuple(range(1, 11))
BM6_STEPS = 2_940
BM6_OUTPUT_EVERY = 294
BM6_DT_DAYS = 365.25 / BM6_OUTPUT_EVERY
REBOUND_STEPS = 29_400
REBOUND_OUTPUT_EVERY = 2_940
REBOUND_DT_DAYS = 365.25 / REBOUND_OUTPUT_EVERY
TIMING_REPEATS = 31
STATE_REPLAY_TOLERANCE = 2.0e-15
ENERGY_REPLAY_TOLERANCE = 5.0e-16
ANGULAR_REPLAY_TOLERANCE = 5.0e-16


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_module(root: Path):
    path = root / "benchmarks/jx_benchmark_a.py"
    specification = importlib.util.spec_from_file_location("jx_benchmark_a", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: "
            f"{' '.join(command)}; see {log_path}"
        )


def invoke_native(
    executable: Path,
    root: Path,
    state_path: Path,
    gm_path: Path,
    trajectory_path: Path,
    result_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        str(executable),
        "--state", str(state_path),
        "--gm", str(gm_path),
        "--trajectory", str(trajectory_path),
        "--result", str(result_path),
        "--contest", "equal_force_budget",
        "--dt-days", format(BM6_DT_DAYS, ".17g"),
        "--steps", str(BM6_STEPS),
        "--output-every-steps", str(BM6_OUTPUT_EVERY),
        "--timing-repeats", str(TIMING_REPEATS),
    ]
    run_command(command, root, log_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["force_evaluations"] != 10 * BM6_STEPS:
        raise RuntimeError("native force-evaluation accounting mismatch")
    if result["steps"] != BM6_STEPS:
        raise RuntimeError("native step count mismatch")
    if not math.isclose(
        result["dt_days"], BM6_DT_DAYS, rel_tol=0.0, abs_tol=0.0
    ):
        raise RuntimeError("native timestep mismatch")
    return result


def parse_native_trajectory(path: Path, state: dict[str, Any]) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected_snapshots = BM6_STEPS // BM6_OUTPUT_EVERY + 1
    expected_rows = expected_snapshots * len(EXPECTED_BODY_IDS)
    if len(rows) != expected_rows:
        raise RuntimeError(f"native trajectory row count {len(rows)} != {expected_rows}")

    grouped: dict[int, dict[int, dict[str, str]]] = {}
    for row in rows:
        step = int(row["step"])
        body_id = int(row["body_id"])
        if body_id in grouped.setdefault(step, {}):
            raise RuntimeError(f"duplicate body {body_id} at step {step}")
        grouped[step][body_id] = row

    expected_steps = tuple(range(0, BM6_STEPS + 1, BM6_OUTPUT_EVERY))
    if tuple(sorted(grouped)) != expected_steps:
        raise RuntimeError("native output-step grid mismatch")

    states: list[np.ndarray] = []
    energies: list[float] = []
    angular: list[float] = []
    for output_index, step in enumerate(expected_steps):
        body_rows = grouped[step]
        if tuple(sorted(body_rows)) != EXPECTED_BODY_IDS:
            raise RuntimeError(f"native body roster mismatch at step {step}")
        expected_time = output_index * 365.25
        snapshot = np.empty((len(EXPECTED_BODY_IDS), 6), dtype=float)
        energy_values: list[float] = []
        angular_values: list[float] = []
        for index, body_id in enumerate(EXPECTED_BODY_IDS):
            row = body_rows[body_id]
            if row["body_name"] != state["names"][index]:
                raise RuntimeError(f"body-name mismatch at step {step}")
            if not math.isclose(
                float(row["time_days"]), expected_time, rel_tol=0.0, abs_tol=5e-12
            ):
                raise RuntimeError(f"time-grid mismatch at step {step}")
            if not math.isclose(
                float(row["jd_tdb"]),
                state["epoch"] + expected_time,
                rel_tol=0.0,
                abs_tol=5e-9,
            ):
                raise RuntimeError(f"JD-grid mismatch at step {step}")
            snapshot[index] = [
                float(row[key]) for key in ("x", "y", "z", "vx", "vy", "vz")
            ]
            energy_values.append(float(row["signed_relative_energy_error"]))
            angular_values.append(
                float(row["relative_angular_momentum_vector_error"])
            )
        if max(energy_values) != min(energy_values):
            raise RuntimeError(f"energy diagnostic differs by body at step {step}")
        if max(angular_values) != min(angular_values):
            raise RuntimeError(f"angular diagnostic differs by body at step {step}")
        states.append(snapshot)
        energies.append(energy_values[0])
        angular.append(angular_values[0])

    return {
        "lane": "bm6_native_cpp",
        "contest": "equal_force_budget",
        "dt": BM6_DT_DAYS,
        "steps": BM6_STEPS,
        "calls": 10 * BM6_STEPS,
        "cost_semantics": "measured: ten native force solves per BM6 macro-step",
        "wall": 0.0,
        "t": np.arange(expected_snapshots, dtype=float) * 365.25,
        "states": np.asarray(states),
        "energy": np.asarray(energies),
        "ang": np.asarray(angular),
        "extra": {},
    }


def replay_comparison(
    native: dict[str, Any], python_bm6: dict[str, Any]
) -> dict[str, Any]:
    if native["states"].shape != python_bm6["states"].shape:
        raise RuntimeError("native/Python state-shape mismatch")
    state_delta = native["states"] - python_bm6["states"]
    energy_delta = native["energy"] - python_bm6["energy"]
    angular_delta = native["ang"] - python_bm6["ang"]
    state_max = float(np.max(np.abs(state_delta)))
    state_rms = float(np.sqrt(np.mean(state_delta * state_delta)))
    energy_max = float(np.max(np.abs(energy_delta)))
    angular_max = float(np.max(np.abs(angular_delta)))
    return {
        "state_max_abs_component_delta": state_max,
        "state_rms_component_delta": state_rms,
        "energy_series_max_abs_delta": energy_max,
        "angular_series_max_abs_delta": angular_max,
        "bit_identical_parsed_state": state_max == 0.0,
        "passed": (
            state_max <= STATE_REPLAY_TOLERANCE
            and energy_max <= ENERGY_REPLAY_TOLERANCE
            and angular_max <= ANGULAR_REPLAY_TOLERANCE
        ),
        "limits": {
            "state_max_abs_component_delta": STATE_REPLAY_TOLERANCE,
            "energy_series_max_abs_delta": ENERGY_REPLAY_TOLERANCE,
            "angular_series_max_abs_delta": ANGULAR_REPLAY_TOLERANCE,
        },
    }


def setup_rebound(state: dict[str, Any], dt: float):
    import rebound

    if rebound.__version__ != REQUIRED_REBOUND:
        raise RuntimeError(
            f"BLOCKED: expected REBOUND {REQUIRED_REBOUND}, "
            f"observed {rebound.__version__}"
        )
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.integrator = "leapfrog"
    simulation.dt = dt
    for index in range(len(state["mu"])):
        simulation.add(
            m=state["mu"][index],
            x=state["q"][index, 0], y=state["q"][index, 1],
            z=state["q"][index, 2], vx=state["v"][index, 0],
            vy=state["v"][index, 1], vz=state["v"][index, 2],
        )
    return simulation


def rebound_native_timing(state: dict[str, Any], repeats: int) -> dict[str, Any]:
    # One force evaluation per REBOUND Leapfrog step. Ten times more steps than
    # BM6 are required to match BM6's ten force solves per macro-step.
    warmup = setup_rebound(state, REBOUND_DT_DAYS)
    warmup.steps(REBOUND_STEPS)

    samples: list[float] = []
    checksums: list[float] = []
    for _ in range(repeats):
        simulation = setup_rebound(state, REBOUND_DT_DAYS)
        start = time.perf_counter()
        simulation.steps(REBOUND_STEPS)
        samples.append(time.perf_counter() - start)
        checksums.append(
            math.fsum(
                (index + 1)
                * (particle.x + particle.y + particle.z +
                   particle.vx + particle.vy + particle.vz)
                for index, particle in enumerate(simulation.particles)
            )
        )
    return {
        "repeats": repeats,
        "steps": REBOUND_STEPS,
        "dt_days": REBOUND_DT_DAYS,
        "modeled_force_evaluations": REBOUND_STEPS,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "terminal_checksum_range": [min(checksums), max(checksums)],
        "timing_semantics": (
            "one Python-to-C Simulation.steps call per repeat; setup excluded"
        ),
    }


def write_summary_csv(
    path: Path, native: dict[str, Any], rebound: dict[str, Any]
) -> None:
    rows = []
    for record in (native, rebound):
        metric = record["trajectory_error_vs_tight_dop853"]["all"]
        rows.append({
            "lane": record["lane"],
            "dt_days": record["dt_days"],
            "steps": record["steps"],
            "force_evaluations": record["force_evaluations"],
            "trajectory_run_wall_seconds": record["wall_seconds"],
            "integration_only_median_seconds": record.get(
                "integration_only_median_seconds", record["wall_seconds"]
            ),
            "max_abs_relative_energy_error": record[
                "max_abs_relative_energy_error"
            ],
            "max_relative_angular_momentum_vector_error": record[
                "max_relative_angular_momentum_vector_error"
            ],
            "all_max_position_error_au": metric["max_pos"],
            "all_rms_position_error_au": metric["rms_pos"],
            "all_max_velocity_error_au_per_day": metric["max_vel"],
            "all_rms_velocity_error_au_per_day": metric["rms_vel"],
        })
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
