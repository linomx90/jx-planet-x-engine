#!/usr/bin/env python3
"""Non-timed convergence calibration for the preregistered GR15 V3 work."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import jx_gr15_ias15 as comparison
from jxplanetx.gr15 import GR15Spec, integrate_gr15


SCHEMA = "jxplanetx.gr15-v3.calibration-report.v1"
PREREGISTRATION_SHA256 = (
    "64c5e9acd9b0a8cfe643c9be0dc2209d3503302dccbb4600973f4c3513d04026"
)
FACTORS = (16.0, 256.0, 4096.0, 65536.0, 1_000_000.0)
EPSILON = 1.0e-6


class CalibrationError(RuntimeError):
    """The locked GR15 V3 calibration could not be completed."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_preregistration(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "benchmarks" / "jx_gr15_v3_preregistration.json"
    raw = path.read_bytes()
    if _sha256(raw) != PREREGISTRATION_SHA256:
        raise CalibrationError("GR15 V3 preregistration identity changed")
    preregistration = json.loads(raw)
    if (
        preregistration.get("schema")
        != "jxplanetx.gr15-v3.preregistration.v1"
        or preregistration["corrector_candidate"][
            "calibration_convergence_factor_grid"
        ]
        != list(FACTORS)
        or preregistration["baseline"]["selected_epsilon"] != EPSILON
    ):
        raise CalibrationError("GR15 V3 preregistration contract changed")
    return preregistration, {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _workloads() -> dict[str, dict[str, Any]]:
    eccentric_gm = [1.0, 1.0e-6]
    eccentric_period = 2.0 * math.pi / math.sqrt(sum(eccentric_gm))
    return {
        "circular_binary_10_periods_11_outputs": {
            "eccentricity": 0.0,
            "gravitational_parameters": [0.5, 0.5],
            "periods": 10,
            "checkpoint_count": 11,
            "initial_step": 0.05,
            "maximum_step": 0.5,
            "accuracy_gate": {
                "position_vector_l2_max": 2.0e-10,
                "velocity_vector_l2_max": 2.0e-10,
                "relative_total_energy_max": 2.0e-12,
                "relative_angular_momentum_max": 2.0e-12,
            },
        },
        "eccentric_binary_e_0_7_one_period_9_outputs": {
            "eccentricity": 0.7,
            "gravitational_parameters": eccentric_gm,
            "periods": 1,
            "checkpoint_count": 9,
            "initial_step": eccentric_period / 1000.0,
            "maximum_step": eccentric_period / 4.0,
            "accuracy_gate": {
                "position_vector_l2_max": 2.0e-10,
                "velocity_vector_l2_max": 2.0e-10,
                "relative_total_energy_max": 2.0e-12,
                "relative_angular_momentum_max": 2.0e-12,
            },
        },
    }


def _spec(
    epochs: tuple[float, ...], workload: dict[str, Any], factor: float
) -> GR15Spec:
    return GR15Spec(
        epochs[0],
        epochs[-1],
        intermediate_epochs=epochs[1:-1],
        initial_step=float(workload["initial_step"]),
        minimum_step=1.0e-15,
        maximum_step=float(workload["maximum_step"]),
        epsilon=EPSILON,
        convergence_factor=factor,
    )


def _accounting(result: Any) -> dict[str, int]:
    return {
        "attempted_steps": result.attempted_steps,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "force_evaluations": result.force_evaluations,
        "corrector_iterations": result.predictor_corrector_iterations,
        "nonconverged_retries": result.nonconverged_retries,
        "polynomial_predictor_trials": result.polynomial_predictor_trials,
        "constant_predictor_trials": result.constant_predictor_trials,
    }


def _run_trial(workload: dict[str, Any], factor: float) -> dict[str, Any]:
    positions, velocities, gm, epochs = comparison._fixture(workload)
    reference_positions, reference_velocities = comparison._analytic_binary(
        epochs, gm, float(workload["eccentricity"])
    )
    spec = _spec(epochs, workload, factor)
    first = integrate_gr15(positions, velocities, gm, spec)
    second = integrate_gr15(positions, velocities, gm, spec)
    deterministic = (
        first.replay_digest == second.replay_digest
        and np.array_equal(first.checkpoint_positions, second.checkpoint_positions)
        and np.array_equal(first.checkpoint_velocities, second.checkpoint_velocities)
    )
    metrics = comparison._metrics(
        first.checkpoint_positions,
        first.checkpoint_velocities,
        reference_positions,
        reference_velocities,
        gm,
    )
    backward_epochs = tuple(reversed(epochs))
    backward = integrate_gr15(
        np.ascontiguousarray(first.positions),
        np.ascontiguousarray(first.velocities),
        gm,
        _spec(backward_epochs, workload, factor),
    )
    reverse_position = float(np.max(np.abs(backward.positions - positions)))
    reverse_velocity = float(np.max(np.abs(backward.velocities - velocities)))
    passed = (
        deterministic
        and comparison._passes(metrics, workload["accuracy_gate"])
        and reverse_position <= 2.0e-9
        and reverse_velocity <= 2.0e-9
    )
    return {
        "passed_without_internal_residual_gate": passed,
        "internal_fixed_point_residual_gate": "PENDING_V3_INSTRUMENTATION",
        "deterministic": deterministic,
        "accuracy_gate": workload["accuracy_gate"],
        "accuracy": metrics,
        "reverse_time": {
            "position_max": reverse_position,
            "velocity_max": reverse_velocity,
        },
        "accounting": _accounting(first),
        "replay_digest": first.replay_digest,
    }


def run(root: Path) -> dict[str, Any]:
    preregistration, identity = _load_preregistration(root)
    workloads = _workloads()
    trials: dict[str, Any] = {}
    provisional_factor: float | None = None
    for factor in FACTORS:
        factor_trials = {
            name: _run_trial(workload, factor)
            for name, workload in workloads.items()
        }
        trials[f"{factor:.17g}"] = factor_trials
        if all(
            trial["passed_without_internal_residual_gate"]
            for trial in factor_trials.values()
        ):
            provisional_factor = factor
    if provisional_factor is None:
        raise CalibrationError("no convergence factor passed the external gates")
    return {
        "schema": SCHEMA,
        "status": "PASS_EXTERNAL_GATES_INTERNAL_RESIDUAL_PENDING",
        "scientific_claim_state": "SCREENING_ONLY",
        "preregistration": identity,
        "epsilon": EPSILON,
        "trials": trials,
        "provisional_largest_passing_factor": provisional_factor,
        "promotion_authorized": False,
        "notes": [
            "No timing was performed or used for selection.",
            "The factor remains provisional until V3 exposes and passes its internal fixed-point residual gate.",
            "The supported V2 core and default configuration were not modified.",
        ],
        "baseline_core_source_sha256": preregistration["baseline"][
            "core_source_sha256"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(root)
    report["semantic_content_sha256"] = _sha256(
        b"jx.gr15-v3.calibration-report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
