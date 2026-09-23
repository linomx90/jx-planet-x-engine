#!/usr/bin/env python3
"""Source-bound holdout qualification and timing for packaged GR15 V3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import numpy as np

import jxplanetx
from benchmarks import jx_gr15_ias15 as comparison
from jxplanetx import _gr15_v3_candidate as v3
from jxplanetx.gr15 import GR15Spec, integrate_gr15, prepare_gr15_workspace


SCHEMA = "jxplanetx.gr15-v3.qualification-report.v1"
PROTOCOL_SCHEMA = "jxplanetx.gr15-v3.qualification-protocol.v1"
BENCHMARK_ID = "jx.public-gr15.corrector-controller-v3.holdout.v1"
LANES = ("gr15_v2", "gr15_v3")


class QualificationError(RuntimeError):
    """The locked GR15 V3 qualification could not be completed exactly."""


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
        raise QualificationError(f"required source is not regular: {path}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise QualificationError(f"required source escapes repository: {path}") from exc
    raw = resolved.read_bytes()
    return {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}


def _read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualificationError("protocol is not valid UTF-8 JSON") from exc
    if (
        type(protocol) is not dict
        or protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("benchmark_id") != BENCHMARK_ID
        or protocol.get("scientific_claim_state") != "SCREENING_ONLY"
        or protocol.get("package_version") != "0.6.0rc8"
        or protocol.get("lane_names") != list(LANES)
        or protocol.get("timed_repetitions") != 16
        or protocol.get("v2_convergence_factor") != 16.0
        or protocol.get("v3_convergence_factor") != 256.0
    ):
        raise QualificationError("fixed qualification protocol changed")
    orders = protocol.get("execution_order")
    if (
        type(orders) is not list
        or len(orders) != 16
        or any(type(order) is not list or set(order) != set(LANES) for order in orders)
        or sum(order[0] == LANES[0] for order in orders) != 8
    ):
        raise QualificationError("balanced timing order changed")
    for expected in protocol.get("source_roster", ()):
        if _file_identity(root / expected["path"], root) != expected:
            raise QualificationError(f"source binding changed: {expected.get('path')}")
    return protocol, {
        "path": path.resolve(strict=True).as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _spec(
    epochs: tuple[float, ...], workload: dict[str, Any], convergence_factor: float
) -> GR15Spec:
    return GR15Spec(
        epochs[0],
        epochs[-1],
        intermediate_epochs=epochs[1:-1],
        initial_step=float(workload["initial_step"]),
        minimum_step=1.0e-15,
        maximum_step=float(workload["maximum_step"]),
        epsilon=float(workload["epsilon"]),
        convergence_factor=convergence_factor,
    )


def _digest(positions: np.ndarray, velocities: np.ndarray, accounting: dict[str, int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"jx.gr15-v3.qualification-lane.v1\0")
    digest.update(positions.tobytes(order="C"))
    digest.update(velocities.tobytes(order="C"))
    digest.update(_canonical(accounting))
    return digest.hexdigest()


def _v2_runner(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    spec: GR15Spec,
) -> Callable[[], dict[str, Any]]:
    workspace = prepare_gr15_workspace(len(gm), spec)

    def run() -> dict[str, Any]:
        started = time.perf_counter_ns()
        result = integrate_gr15(positions, velocities, gm, spec, workspace=workspace)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        output_positions = result.checkpoint_positions.copy()
        output_velocities = result.checkpoint_velocities.copy()
        accounting = {
            "attempted_steps": result.attempted_steps,
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "force_evaluations": result.force_evaluations,
            "corrector_iterations": result.predictor_corrector_iterations,
            "nonconverged_retries": result.nonconverged_retries,
        }
        return {
            "elapsed_seconds": elapsed,
            "positions": output_positions,
            "velocities": output_velocities,
            "accounting": accounting,
            "digest": _digest(output_positions, output_velocities, accounting),
        }

    return run


def _v3_runner(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    spec: GR15Spec,
) -> Callable[[], dict[str, Any]]:
    workspace = v3.prepare_workspace(len(gm), spec)

    def run() -> dict[str, Any]:
        started = time.perf_counter_ns()
        result = v3.integrate(positions, velocities, gm, spec, workspace=workspace)
        elapsed = (time.perf_counter_ns() - started) * 1.0e-9
        output_positions = result.checkpoint_positions.copy()
        output_velocities = result.checkpoint_velocities.copy()
        accounting = {
            "attempted_steps": result.attempted_steps,
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "force_evaluations": result.force_evaluations,
            "corrector_iterations": result.predictor_corrector_iterations,
            "nonconverged_retries": result.nonconverged_retries,
            "terminal_force_reuses": result.terminal_force_reuses,
            "terminal_force_sweeps": result.terminal_force_sweeps,
        }
        return {
            "elapsed_seconds": elapsed,
            "positions": output_positions,
            "velocities": output_velocities,
            "accounting": accounting,
            "maximum_corrector_residual": result.maximum_corrector_residual,
            "corrector_threshold": result.corrector_threshold,
            "digest": _digest(output_positions, output_velocities, accounting),
        }

    return run


def _run_workload(
    name: str, workload: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    positions, velocities, gm, epochs = comparison._fixture(workload)
    reference_positions, reference_velocities = comparison._analytic_binary(
        epochs, gm, float(workload["eccentricity"])
    )
    v2_spec = _spec(epochs, workload, protocol["v2_convergence_factor"])
    v3_spec = _spec(epochs, workload, protocol["v3_convergence_factor"])
    runners = {
        "gr15_v2": _v2_runner(positions, velocities, gm, v2_spec),
        "gr15_v3": _v3_runner(positions, velocities, gm, v3_spec),
    }
    for runner in runners.values():
        runner()
    samples: dict[str, list[dict[str, Any]]] = {lane: [] for lane in LANES}
    for order in protocol["execution_order"]:
        for lane in order:
            samples[lane].append(runners[lane]())
    for lane in LANES:
        if len({sample["digest"] for sample in samples[lane]}) != 1:
            raise QualificationError(f"{name} {lane} changed output during timing")
    representative = {lane: samples[lane][0] for lane in LANES}
    accuracy = {
        lane: comparison._metrics(
            representative[lane]["positions"],
            representative[lane]["velocities"],
            reference_positions,
            reference_velocities,
            gm,
        )
        for lane in LANES
    }
    timing = {
        lane: comparison._timing_summary(
            [sample["elapsed_seconds"] for sample in samples[lane]]
        )
        for lane in LANES
    }
    backward_epochs = tuple(reversed(epochs))
    backward = v3.integrate(
        np.ascontiguousarray(representative["gr15_v3"]["positions"][-1]),
        np.ascontiguousarray(representative["gr15_v3"]["velocities"][-1]),
        gm,
        _spec(backward_epochs, workload, protocol["v3_convergence_factor"]),
    )
    reverse_time = {
        "position_max": float(np.max(np.abs(backward.positions - positions))),
        "velocity_max": float(np.max(np.abs(backward.velocities - velocities))),
    }
    v2_accounting = representative["gr15_v2"]["accounting"]
    v3_accounting = representative["gr15_v3"]["accounting"]
    speedup = (
        timing["gr15_v2"]["median_seconds"]
        / timing["gr15_v3"]["median_seconds"]
    )
    gates = {
        "v2_accuracy": comparison._passes(accuracy["gr15_v2"], workload["accuracy_gate"]),
        "v3_accuracy": comparison._passes(accuracy["gr15_v3"], workload["accuracy_gate"]),
        "v3_force_reduction": v3_accounting["force_evaluations"]
        <= protocol["maximum_v3_force_fraction"]
        * v2_accounting["force_evaluations"],
        "v3_rejections_not_increased": v3_accounting["rejected_steps"]
        <= v2_accounting["rejected_steps"],
        "v3_reverse_position": reverse_time["position_max"]
        <= protocol["reverse_time_position_max"],
        "v3_reverse_velocity": reverse_time["velocity_max"]
        <= protocol["reverse_time_velocity_max"],
        "v3_speedup": speedup >= protocol["minimum_v3_speedup"],
        "v3_residual": representative["gr15_v3"]["maximum_corrector_residual"]
        <= representative["gr15_v3"]["corrector_threshold"],
    }
    return {
        "status": "PASS_FIXED_GATES" if all(gates.values()) else "FAIL_FIXED_GATE",
        "checkpoint_count": len(epochs),
        "accuracy_gate": workload["accuracy_gate"],
        "accuracy": accuracy,
        "reverse_time": reverse_time,
        "accounting": {lane: representative[lane]["accounting"] for lane in LANES},
        "determinism": {lane: representative[lane]["digest"] for lane in LANES},
        "corrector": {
            "maximum_residual": representative["gr15_v3"][
                "maximum_corrector_residual"
            ],
            "threshold": representative["gr15_v3"]["corrector_threshold"],
        },
        "timing": timing,
        "v3_speedup_over_v2": speedup,
        "force_evaluation_reduction_fraction": 1.0
        - v3_accounting["force_evaluations"] / v2_accounting["force_evaluations"],
        "gates": gates,
    }


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    v3_identity = v3.runtime_identity()
    if (
        jxplanetx.__version__ != protocol["package_version"]
        or v3_identity["source_sha256"] != protocol["v3_core_source_sha256"]
        or v3_identity["scientific_claim_state"] != "SCREENING_ONLY"
    ):
        raise QualificationError("GR15 V3 runtime identity differs from the protocol")
    workloads = {
        name: _run_workload(name, workload, protocol)
        for name, workload in protocol["workloads"].items()
    }
    passed = all(row["status"] == "PASS_FIXED_GATES" for row in workloads.values())
    return {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": "PASS_SCREENING_ONLY" if passed else "FAIL_FIXED_GATE",
        "scientific_claim_state": "SCREENING_ONLY",
        "claim_controls": {
            "public_default_promotion_authorized": passed,
            "release_version_change_authorized": False,
            "general_superiority_claimed": False,
            "production_qualification_claimed": False,
            "single_host_observation": True,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jxplanetx": jxplanetx.__version__,
            "gr15_v3": v3_identity,
        },
        "workloads": workloads,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, identity = _read_protocol(args.protocol, root)
    report = run(protocol)
    report["protocol"] = identity
    report["semantic_content_sha256"] = _sha256(
        b"jx.gr15-v3.qualification-report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"] == "PASS_SCREENING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
