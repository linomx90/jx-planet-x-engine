#!/usr/bin/env python3
"""Prospective accuracy-gated public JX GR15 versus REBOUND IAS15 screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Callable

import numpy as np

import jxplanetx
from jxplanetx.gr15 import (
    GR15Spec,
    gr15_runtime_identity,
    integrate_gr15,
    prepare_gr15_workspace,
)


SCHEMA = "jxplanetx.gr15-ias15.screening-report.v1"
PROTOCOL_SCHEMA = "jxplanetx.gr15-ias15.protocol.v1"
BENCHMARK_ID = "jx.public-gr15-vs-rebound-ias15.accuracy-gated.v1"
LANES = ("jx_gr15", "rebound_ias15")
TAU = 2.0 * math.pi


class ComparisonError(RuntimeError):
    """The locked GR15/IAS15 comparison could not be completed exactly."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise ComparisonError(f"required source is not regular: {path}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ComparisonError(f"required source escapes repository: {path}") from exc
    raw = resolved.read_bytes()
    return {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}


def _read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ComparisonError("protocol is not valid UTF-8 JSON") from exc
    if (
        type(protocol) is not dict
        or protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("benchmark_id") != BENCHMARK_ID
        or protocol.get("scientific_claim_state") != "SCREENING_ONLY"
        or protocol.get("package_version") != "0.6.0rc8"
        or protocol.get("rebound_version") != "5.1.1"
        or protocol.get("lane_names") != list(LANES)
        or protocol.get("timed_repetitions") != 16
    ):
        raise ComparisonError("fixed protocol identity changed")
    epsilon_grid = protocol.get("epsilon_grid")
    if epsilon_grid != [1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-10, 1.0e-11, 1.0e-12]:
        raise ComparisonError("fixed tolerance-selection grid changed")
    orders = protocol.get("execution_order")
    if (
        type(orders) is not list
        or len(orders) != 16
        or any(type(order) is not list or set(order) != set(LANES) for order in orders)
        or sum(order[0] == LANES[0] for order in orders) != 8
    ):
        raise ComparisonError("balanced execution order changed")
    roster = protocol.get("source_roster")
    if type(roster) is not list or not roster:
        raise ComparisonError("source roster is missing")
    for expected in roster:
        if _file_identity(root / expected["path"], root) != expected:
            raise ComparisonError(f"source binding changed: {expected.get('path')}")
    return protocol, {
        "path": path.resolve(strict=True).as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    turns = math.floor(mean_anomaly / TAU)
    reduced = mean_anomaly - turns * TAU
    eccentric_anomaly = reduced if eccentricity < 0.8 else math.pi
    for _ in range(32):
        residual = (
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - reduced
        )
        correction = residual / (1.0 - eccentricity * math.cos(eccentric_anomaly))
        eccentric_anomaly -= correction
        if abs(correction) <= 2.0e-16 * max(1.0, abs(eccentric_anomaly)):
            break
    else:
        raise ComparisonError("analytic Kepler solve did not converge")
    return eccentric_anomaly + turns * TAU


def _analytic_binary(
    epochs: tuple[float, ...], gm: np.ndarray, eccentricity: float
) -> tuple[np.ndarray, np.ndarray]:
    total_gm = float(np.sum(gm))
    mean_motion = math.sqrt(total_gm)
    root = math.sqrt(1.0 - eccentricity * eccentricity)
    positions = np.empty((len(epochs), 2, 3), dtype=np.float64)
    velocities = np.empty_like(positions)
    for checkpoint, epoch in enumerate(epochs):
        anomaly = _solve_kepler(mean_motion * epoch, eccentricity)
        cosine = math.cos(anomaly)
        sine = math.sin(anomaly)
        relative_position = np.array(
            (cosine - eccentricity, root * sine, 0.0), dtype=np.float64
        )
        anomaly_rate = mean_motion / (1.0 - eccentricity * cosine)
        relative_velocity = np.array(
            (-sine * anomaly_rate, root * cosine * anomaly_rate, 0.0),
            dtype=np.float64,
        )
        positions[checkpoint, 0] = -gm[1] / total_gm * relative_position
        positions[checkpoint, 1] = gm[0] / total_gm * relative_position
        velocities[checkpoint, 0] = -gm[1] / total_gm * relative_velocity
        velocities[checkpoint, 1] = gm[0] / total_gm * relative_velocity
    return positions, velocities


def _fixture(workload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...]]:
    eccentricity = float(workload["eccentricity"])
    gm = np.array(workload["gravitational_parameters"], dtype=np.float64)
    period = TAU / math.sqrt(float(np.sum(gm)))
    periods = int(workload["periods"])
    checkpoint_count = int(workload["checkpoint_count"])
    final_epoch = periods * period
    epochs = tuple(
        final_epoch * index / (checkpoint_count - 1)
        for index in range(checkpoint_count)
    )
    analytic_positions, analytic_velocities = _analytic_binary(
        epochs[:1], gm, eccentricity
    )
    return analytic_positions[0].copy(), analytic_velocities[0].copy(), gm, epochs


def _invariants(
    positions: np.ndarray, velocities: np.ndarray, gm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    energy = 0.5 * np.sum(gm[None, :, None] * velocities * velocities, axis=(1, 2))
    separation = np.linalg.norm(positions[:, 1] - positions[:, 0], axis=1)
    energy -= gm[0] * gm[1] / separation
    angular = np.sum(gm[None, :, None] * np.cross(positions, velocities), axis=1)
    return energy, angular


def _metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray,
    gm: np.ndarray,
) -> dict[str, float]:
    energy, angular = _invariants(positions, velocities, gm)
    initial_energy = energy[0]
    initial_angular_norm = float(np.linalg.norm(angular[0]))
    return {
        "position_vector_l2_max": float(
            np.max(np.linalg.norm(positions - reference_positions, axis=2))
        ),
        "velocity_vector_l2_max": float(
            np.max(np.linalg.norm(velocities - reference_velocities, axis=2))
        ),
        "relative_total_energy_max": float(
            np.max(np.abs((energy - initial_energy) / initial_energy))
        ),
        "relative_angular_momentum_max": float(
            np.max(np.linalg.norm(angular - angular[0], axis=1))
            / initial_angular_norm
        ),
    }


def _passes(metrics: dict[str, float], gate: dict[str, float]) -> bool:
    return all(metrics[key] <= limit for key, limit in gate.items())


def _digest(positions: np.ndarray, velocities: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.public-gr15-vs-ias15.lane.v1\0")
    digest.update(positions.tobytes(order="C"))
    digest.update(velocities.tobytes(order="C"))
    return digest.hexdigest()


def _gr15_runner(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    epochs: tuple[float, ...],
    epsilon: float,
    workload: dict[str, Any],
) -> Callable[[], dict[str, Any]]:
    spec = GR15Spec(
        epochs[0],
        epochs[-1],
        intermediate_epochs=epochs[1:-1],
        initial_step=float(workload["initial_step"]),
        minimum_step=1.0e-15,
        maximum_step=float(workload["maximum_step"]),
        epsilon=epsilon,
    )
    workspace = prepare_gr15_workspace(2, spec)

    def run() -> dict[str, Any]:
        started = time.perf_counter_ns()
        result = integrate_gr15(positions, velocities, gm, spec, workspace=workspace)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        output_positions = result.checkpoint_positions.copy()
        output_velocities = result.checkpoint_velocities.copy()
        return {
            "elapsed_seconds": elapsed,
            "positions": output_positions,
            "velocities": output_velocities,
            "digest": _digest(output_positions, output_velocities),
            "accounting": {
                "attempted_steps": result.attempted_steps,
                "accepted_steps": result.accepted_steps,
                "rejected_steps": result.rejected_steps,
                "force_evaluations": result.force_evaluations,
                "corrector_iterations": result.predictor_corrector_iterations,
            },
        }

    return run


def _ias15_runner(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    epochs: tuple[float, ...],
    epsilon: float,
    workload: dict[str, Any],
    rebound: Any,
) -> Callable[[], dict[str, Any]]:
    def run() -> dict[str, Any]:
        simulation = rebound.Simulation()
        simulation.G = 1.0
        simulation.gravity = "basic"
        simulation.collision = "none"
        simulation.softening = 0.0
        simulation.integrator = "ias15"
        simulation.integrator.epsilon = epsilon
        simulation.integrator.min_dt = 0.0
        simulation.integrator.adaptive_mode = "PRS23"
        simulation.dt = float(workload["initial_step"])
        for index in range(2):
            simulation.add(
                m=float(gm[index]),
                x=float(positions[index, 0]),
                y=float(positions[index, 1]),
                z=float(positions[index, 2]),
                vx=float(velocities[index, 0]),
                vy=float(velocities[index, 1]),
                vz=float(velocities[index, 2]),
            )
        output_positions = np.empty((len(epochs), 2, 3), dtype=np.float64)
        output_velocities = np.empty_like(output_positions)

        def capture(checkpoint: int) -> None:
            for body, particle in enumerate(simulation.particles[:2]):
                output_positions[checkpoint, body] = (particle.x, particle.y, particle.z)
                output_velocities[checkpoint, body] = (
                    particle.vx,
                    particle.vy,
                    particle.vz,
                )

        capture(0)
        started = time.perf_counter_ns()
        for checkpoint, epoch in enumerate(epochs[1:], 1):
            simulation.integrate(epoch, exact_finish_time=1)
            if float(simulation.t) != epoch:
                raise ComparisonError("IAS15 missed an exact checkpoint")
            capture(checkpoint)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        return {
            "elapsed_seconds": elapsed,
            "positions": output_positions,
            "velocities": output_velocities,
            "digest": _digest(output_positions, output_velocities),
            "accounting": {"steps_done": int(simulation.steps_done)},
        }

    return run


def _timing_summary(values: list[float]) -> dict[str, Any]:
    return {
        "samples_seconds": values,
        "minimum_seconds": min(values),
        "median_seconds": statistics.median(values),
        "maximum_seconds": max(values),
        "mean_seconds": statistics.fmean(values),
    }


def _run_workload(
    workload_name: str,
    workload: dict[str, Any],
    protocol: dict[str, Any],
    rebound: Any,
) -> dict[str, Any]:
    positions, velocities, gm, epochs = _fixture(workload)
    reference_positions, reference_velocities = _analytic_binary(
        epochs, gm, float(workload["eccentricity"])
    )
    gate = workload["accuracy_gate"]
    factories = {
        "jx_gr15": lambda epsilon: _gr15_runner(
            positions, velocities, gm, epochs, epsilon, workload
        ),
        "rebound_ias15": lambda epsilon: _ias15_runner(
            positions, velocities, gm, epochs, epsilon, workload, rebound
        ),
    }
    calibration: dict[str, Any] = {}
    selected: dict[str, tuple[float, Callable[[], dict[str, Any]], dict[str, Any]]] = {}
    for lane in LANES:
        trials = []
        for epsilon in protocol["epsilon_grid"]:
            runner = factories[lane](epsilon)
            result = runner()
            metrics = _metrics(
                result["positions"],
                result["velocities"],
                reference_positions,
                reference_velocities,
                gm,
            )
            passed = _passes(metrics, gate)
            trials.append({"epsilon": epsilon, "metrics": metrics, "passed": passed})
            if passed:
                selected[lane] = (epsilon, runner, result)
                break
        calibration[lane] = trials
        if lane not in selected:
            raise ComparisonError(
                f"{workload_name} {lane} found no passing epsilon in the locked grid"
            )

    for _, runner, _ in selected.values():
        runner()
    samples: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    for order in protocol["execution_order"]:
        for lane in order:
            samples[lane].append(selected[lane][1]())
    for lane in LANES:
        if len({sample["digest"] for sample in samples[lane]}) != 1:
            raise ComparisonError(f"{workload_name} {lane} was not deterministic")
    representative = {lane: samples[lane][0] for lane in LANES}
    final_accuracy = {
        lane: _metrics(
            representative[lane]["positions"],
            representative[lane]["velocities"],
            reference_positions,
            reference_velocities,
            gm,
        )
        for lane in LANES
    }
    timing = {
        lane: _timing_summary(
            [sample["elapsed_seconds"] for sample in samples[lane]]
        )
        for lane in LANES
    }
    jx_median = timing["jx_gr15"]["median_seconds"]
    ias15_median = timing["rebound_ias15"]["median_seconds"]
    passed = all(_passes(final_accuracy[lane], gate) for lane in LANES)
    return {
        "status": "PASS_SHARED_ACCURACY_GATE" if passed else "FAIL_SHARED_ACCURACY_GATE",
        "reference_class": "CLOSED_FORM_TWO_BODY_KEPLER_SOLUTION",
        "checkpoint_count": len(epochs),
        "accuracy_gate": gate,
        "calibration": calibration,
        "selected_epsilon": {lane: selected[lane][0] for lane in LANES},
        "accuracy": final_accuracy,
        "determinism": {lane: representative[lane]["digest"] for lane in LANES},
        "accounting": {lane: representative[lane]["accounting"] for lane in LANES},
        "timing": timing,
        "ratios": {
            "jx_over_ias15_median_time": jx_median / ias15_median,
            "jx_over_ias15_throughput": ias15_median / jx_median,
        },
    }


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import rebound
    except ModuleNotFoundError as exc:
        raise ComparisonError("exact REBOUND 5.1.1 is required") from exc
    if rebound.__version__ != protocol["rebound_version"]:
        raise ComparisonError(
            f"exact REBOUND {protocol['rebound_version']} required; found {rebound.__version__}"
        )
    identity = gr15_runtime_identity()
    if (
        jxplanetx.__version__ != protocol["package_version"]
        or identity["source_sha256"] != protocol["gr15_core_source_sha256"]
        or identity["scientific_claim_state"] != "SCREENING_ONLY"
    ):
        raise ComparisonError("public GR15 runtime identity differs from the protocol")
    workloads = {
        name: _run_workload(name, workload, protocol, rebound)
        for name, workload in protocol["workloads"].items()
    }
    passed = all(
        workload["status"] == "PASS_SHARED_ACCURACY_GATE"
        for workload in workloads.values()
    )
    return {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": "PASS_SCREENING_ONLY" if passed else "FAIL_SHARED_GATE",
        "scientific_claim_state": "SCREENING_ONLY",
        "claim_controls": {
            "general_superiority_claimed": False,
            "production_qualification_claimed": False,
            "release_promotion_authorized": False,
            "single_host_observation": True,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jxplanetx": jxplanetx.__version__,
            "rebound": rebound.__version__,
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "gr15": identity,
        },
        "workloads": workloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, protocol_identity = _read_protocol(args.protocol, root)
    report = run(protocol)
    report["protocol"] = protocol_identity
    report["semantic_content_sha256"] = _sha256(
        b"jx.public-gr15-vs-rebound-ias15.report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"] == "PASS_SCREENING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
