#!/usr/bin/env python3
"""Prospective long/adversarial qualification for packaged public GR15 V3.

This runner is correctness-only.  It compares the public JX result with two
independently configured REBOUND 5.1.1 IAS15 integrations at identical retained
epochs, checks analytic truth where it exists, and exercises reverse-time,
determinism, conservation, accounting, and failure-domain gates.  It does not
measure or rank performance and cannot authorize production ephemerides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

import jxplanetx
from jxplanetx.gr15 import (
    GR15ContractError,
    GR15IntegrationError,
    GR15Spec,
    MAXIMUM_BODY_COUNT,
    STATUS_FORCE_SINGULARITY,
    STATUS_MINIMUM_STEP,
    gr15_runtime_identity,
    integrate_gr15,
)


SCHEMA = "jxplanetx.gr15-v3.adversarial-report.v1"
PROTOCOL_SCHEMA = "jxplanetx.gr15-v3.adversarial-protocol.v1"
BENCHMARK_ID = "jx.public-gr15-v3.long-adversarial-qualification.v1"
REQUIRED_CASES = (
    "high_eccentricity_binary_10_periods",
    "extreme_mass_ratio_binary_25_periods",
    "close_scatter_three_body",
    "planetary_hierarchy_11_body_20_periods",
    "planetary_boundary_32_body_2_periods",
)
TAU = 2.0 * math.pi


class AdversarialQualificationError(RuntimeError):
    """The sealed adversarial qualification could not be completed exactly."""


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
        raise AdversarialQualificationError(f"required source is not regular: {path}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise AdversarialQualificationError(
            f"required source escapes repository: {path}"
        ) from exc
    raw = resolved.read_bytes()
    return {"path": relative, "size_bytes": len(raw), "sha256": _sha256(raw)}


def _read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    try:
        protocol = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdversarialQualificationError("protocol is not valid UTF-8 JSON") from exc
    if (
        type(protocol) is not dict
        or protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("benchmark_id") != BENCHMARK_ID
        or protocol.get("status") != "LOCKED_BEFORE_FIRST_ADVERSARIAL_EXECUTION"
        or protocol.get("scientific_claim_state") != "SCREENING_ONLY"
        or protocol.get("package_version") != "0.6.0rc8"
        or protocol.get("rebound_version") != "5.1.1"
        or protocol.get("rebound_githash")
        != "33549d1d50d616a95a6d6a79e5e2c9c3b3730b1f"
        or protocol.get("gr15_core_source_sha256")
        != "036d1dc7e4e2e26c908c6dde9ff27bdf6a5804b88aa3f3e374f1f512642c1e41"
        or protocol.get("convergence_factor") != 256.0
        or tuple(protocol.get("case_order", ())) != REQUIRED_CASES
        or tuple(protocol.get("cases", {}).keys()) != REQUIRED_CASES
    ):
        raise AdversarialQualificationError("fixed adversarial protocol changed")
    if protocol.get("claim_controls") != {
        "general_superiority_claim_authorized": False,
        "performance_claim_authorized": False,
        "production_ephemeris_claim_authorized": False,
        "release_promotion_authorized_by_this_report_alone": False,
        "screening_only": True,
    }:
        raise AdversarialQualificationError("claim controls changed")
    roster = protocol.get("source_roster")
    if type(roster) is not list or not roster:
        raise AdversarialQualificationError("source roster is missing")
    for expected in roster:
        if _file_identity(root / expected["path"], root) != expected:
            raise AdversarialQualificationError(
                f"source binding changed: {expected.get('path')}"
            )
    return protocol, {
        "path": resolved.as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    turns = math.floor(mean_anomaly / TAU)
    reduced = mean_anomaly - turns * TAU
    anomaly = reduced if eccentricity < 0.8 else math.pi
    for _ in range(64):
        residual = anomaly - eccentricity * math.sin(anomaly) - reduced
        correction = residual / (1.0 - eccentricity * math.cos(anomaly))
        anomaly -= correction
        if abs(correction) <= 2.0e-16 * max(1.0, abs(anomaly)):
            return anomaly + turns * TAU
    raise AdversarialQualificationError("analytic Kepler solve did not converge")


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


def _uniform_epochs(final_epoch: float, checkpoint_count: int) -> tuple[float, ...]:
    return tuple(
        final_epoch * index / (checkpoint_count - 1)
        for index in range(checkpoint_count)
    )


def _planetary_fixture(body_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.zeros((body_count, 3), dtype=np.float64)
    velocities = np.zeros_like(positions)
    gm = np.empty(body_count, dtype=np.float64)
    gm[0] = 1.0
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(1, body_count):
        radius = 1.0 + 0.1 * (index - 1)
        angle = golden_angle * index
        gm[index] = 1.0e-6 * (1.0 + 0.01 * index)
        positions[index] = (
            radius * math.cos(angle),
            radius * math.sin(angle),
            0.0,
        )
        speed = math.sqrt(gm[0] / radius)
        velocities[index] = (
            -speed * math.sin(angle),
            speed * math.cos(angle),
            0.0,
        )
    total_gm = float(np.sum(gm))
    positions -= np.sum(positions * gm[:, None], axis=0) / total_gm
    velocities -= np.sum(velocities * gm[:, None], axis=0) / total_gm
    return positions, velocities, gm


def _fixture(
    case: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, ...], float | None]:
    kind = case["fixture"]
    if kind == "analytic_binary":
        gm = np.asarray(case["gravitational_parameters"], dtype=np.float64)
        eccentricity = float(case["eccentricity"])
        period = TAU / math.sqrt(float(np.sum(gm)))
        epochs = _uniform_epochs(
            int(case["periods"]) * period, int(case["checkpoint_count"])
        )
        positions, velocities = _analytic_binary(epochs[:1], gm, eccentricity)
        return positions[0].copy(), velocities[0].copy(), gm, epochs, eccentricity
    if kind == "close_scatter_three_body":
        rows = np.asarray(
            [[float.fromhex(value) for value in row] for row in case["initial_rows_hex"]],
            dtype=np.float64,
        )
        gm = rows[:, 0].copy()
        positions = rows[:, 2:5].copy()
        velocities = rows[:, 5:8].copy()
        epochs = tuple(
            float.fromhex(case["base_step_hex"]) * int(index)
            for index in case["checkpoint_indices"]
        )
        return positions, velocities, gm, epochs, None
    if kind == "planetary_hierarchy":
        body_count = int(case["body_count"])
        positions, velocities, gm = _planetary_fixture(body_count)
        epochs = _uniform_epochs(
            float(case["inner_periods"]) * TAU,
            int(case["checkpoint_count"]),
        )
        return positions, velocities, gm, epochs, None
    raise AdversarialQualificationError(f"unknown fixture kind: {kind!r}")


def _digest(positions: np.ndarray, velocities: np.ndarray) -> str:
    digest = hashlib.sha256(b"jx.gr15-v3.adversarial-state.v1\0")
    for array in (positions, velocities):
        checked = np.ascontiguousarray(array, dtype="<f8")
        digest.update(len(checked.shape).to_bytes(1, "big"))
        for dimension in checked.shape:
            digest.update(int(dimension).to_bytes(8, "big"))
        digest.update(checked.tobytes(order="C"))
    return digest.hexdigest()


def _state_error(
    positions: np.ndarray,
    velocities: np.ndarray,
    reference_positions: np.ndarray,
    reference_velocities: np.ndarray,
) -> dict[str, float]:
    position_error = float(
        np.max(np.linalg.norm(positions - reference_positions, axis=2))
    )
    velocity_error = float(
        np.max(np.linalg.norm(velocities - reference_velocities, axis=2))
    )
    position_scale = max(
        1.0, float(np.max(np.linalg.norm(reference_positions, axis=2)))
    )
    velocity_scale = max(
        1.0, float(np.max(np.linalg.norm(reference_velocities, axis=2)))
    )
    return {
        "position_vector_l2_max": position_error,
        "velocity_vector_l2_max": velocity_error,
        "position_scaled_max": position_error / position_scale,
        "velocity_scaled_max": velocity_error / velocity_scale,
    }


def _trajectory_metrics(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    epochs: tuple[float, ...],
) -> dict[str, float]:
    count = len(epochs)
    energy = np.empty(count, dtype=np.float64)
    angular = np.empty((count, 3), dtype=np.float64)
    momentum = np.empty((count, 3), dtype=np.float64)
    center = np.empty((count, 3), dtype=np.float64)
    minimum_separation = math.inf
    total_gm = float(np.sum(gm))
    for checkpoint in range(count):
        position = positions[checkpoint]
        velocity = velocities[checkpoint]
        energy[checkpoint] = 0.5 * float(
            np.sum(gm[:, None] * velocity * velocity)
        )
        for left in range(len(gm)):
            for right in range(left + 1, len(gm)):
                separation = float(np.linalg.norm(position[right] - position[left]))
                minimum_separation = min(minimum_separation, separation)
                energy[checkpoint] -= gm[left] * gm[right] / separation
        angular[checkpoint] = np.sum(
            gm[:, None] * np.cross(position, velocity), axis=0
        )
        momentum[checkpoint] = np.sum(gm[:, None] * velocity, axis=0)
        center[checkpoint] = np.sum(gm[:, None] * position, axis=0) / total_gm
    position_scale = max(1.0, float(np.max(np.linalg.norm(positions, axis=2))))
    velocity_scale = max(1.0, float(np.max(np.linalg.norm(velocities, axis=2))))
    expected_center = center[0] + (
        np.asarray(epochs, dtype=np.float64) - epochs[0]
    )[:, None] * momentum[0][None, :] / total_gm
    momentum_scale = max(total_gm * velocity_scale, np.finfo(np.float64).tiny)
    angular_scale = max(
        float(np.linalg.norm(angular[0])),
        total_gm * position_scale * velocity_scale,
        np.finfo(np.float64).tiny,
    )
    return {
        "relative_total_energy_max": float(
            np.max(np.abs(energy - energy[0]))
            / max(abs(float(energy[0])), np.finfo(np.float64).tiny)
        ),
        "scaled_angular_momentum_max": float(
            np.max(np.linalg.norm(angular - angular[0], axis=1)) / angular_scale
        ),
        "scaled_linear_momentum_max": float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / momentum_scale
        ),
        "scaled_center_of_mass_linearity_max": float(
            np.max(np.linalg.norm(center - expected_center, axis=1)) / position_scale
        ),
        "minimum_retained_pair_separation": minimum_separation,
    }


def _jx_run(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    epochs: tuple[float, ...],
    case: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    spec = GR15Spec(
        epochs[0],
        epochs[-1],
        intermediate_epochs=epochs[1:-1],
        initial_step=float(case["initial_step"]),
        minimum_step=float(case["minimum_step"]),
        maximum_step=float(case["maximum_step"]),
        epsilon=float(case["jx_epsilon"]),
        convergence_factor=float(protocol["convergence_factor"]),
        maximum_steps=int(case["maximum_steps"]),
        maximum_rejections=int(case["maximum_rejections"]),
    )
    result = integrate_gr15(positions, velocities, gm, spec)
    output_positions = result.checkpoint_positions.copy()
    output_velocities = result.checkpoint_velocities.copy()
    return {
        "positions": output_positions,
        "velocities": output_velocities,
        "digest": _digest(output_positions, output_velocities),
        "accounting": {
            "attempted_steps": result.attempted_steps,
            "accepted_steps": result.accepted_steps,
            "rejected_steps": result.rejected_steps,
            "force_evaluations": result.force_evaluations,
            "predictor_corrector_iterations": result.predictor_corrector_iterations,
            "nonconverged_retries": result.nonconverged_retries,
            "terminal_force_reuses": result.terminal_force_reuses,
            "terminal_force_sweeps": result.terminal_force_sweeps,
        },
        "maximum_corrector_residual": result.maximum_corrector_residual,
        "corrector_threshold": result.corrector_threshold,
    }


def _ias15_run(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    epochs: tuple[float, ...],
    epsilon: float,
    initial_step: float,
    rebound: Any,
) -> dict[str, Any]:
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "ias15"
    simulation.integrator.epsilon = epsilon
    simulation.integrator.min_dt = 0.0
    simulation.integrator.adaptive_mode = "PRS23"
    simulation.dt = initial_step
    for body in range(len(gm)):
        simulation.add(
            m=float(gm[body]),
            x=float(positions[body, 0]),
            y=float(positions[body, 1]),
            z=float(positions[body, 2]),
            vx=float(velocities[body, 0]),
            vy=float(velocities[body, 1]),
            vz=float(velocities[body, 2]),
        )
    output_positions = np.empty((len(epochs), len(gm), 3), dtype=np.float64)
    output_velocities = np.empty_like(output_positions)

    def capture(checkpoint: int) -> None:
        for body, particle in enumerate(simulation.particles[: len(gm)]):
            output_positions[checkpoint, body] = (particle.x, particle.y, particle.z)
            output_velocities[checkpoint, body] = (
                particle.vx,
                particle.vy,
                particle.vz,
            )

    capture(0)
    for checkpoint, epoch in enumerate(epochs[1:], 1):
        simulation.integrate(epoch, exact_finish_time=1)
        if float(simulation.t) != epoch:
            raise AdversarialQualificationError("IAS15 missed an exact checkpoint")
        capture(checkpoint)
    if not (
        np.all(np.isfinite(output_positions))
        and np.all(np.isfinite(output_velocities))
    ):
        raise AdversarialQualificationError("IAS15 returned a nonfinite trajectory")
    return {
        "positions": output_positions,
        "velocities": output_velocities,
        "digest": _digest(output_positions, output_velocities),
        "steps_done": int(simulation.steps_done),
    }


def _passes_state(metrics: dict[str, float], limit: float) -> bool:
    return (
        metrics["position_scaled_max"] <= limit
        and metrics["velocity_scaled_max"] <= limit
    )


def _run_case(
    name: str, case: dict[str, Any], protocol: dict[str, Any], rebound: Any
) -> dict[str, Any]:
    positions, velocities, gm, epochs, eccentricity = _fixture(case)
    if len(gm) != int(case["expected_body_count"]):
        raise AdversarialQualificationError(f"{name} body count changed")
    first = _jx_run(positions, velocities, gm, epochs, case, protocol)
    repeat = _jx_run(positions, velocities, gm, epochs, case, protocol)
    primary = _ias15_run(
        positions,
        velocities,
        gm,
        epochs,
        float(protocol["ias15_primary_epsilon"]),
        float(case["initial_step"]),
        rebound,
    )
    sensitivity = _ias15_run(
        positions,
        velocities,
        gm,
        epochs,
        float(protocol["ias15_sensitivity_epsilon"]),
        float(case["initial_step"]) / 8.0,
        rebound,
    )
    deterministic = (
        first["digest"] == repeat["digest"]
        and first["accounting"] == repeat["accounting"]
        and first["maximum_corrector_residual"]
        == repeat["maximum_corrector_residual"]
    )
    reference_stability = _state_error(
        primary["positions"],
        primary["velocities"],
        sensitivity["positions"],
        sensitivity["velocities"],
    )
    jx_reference = _state_error(
        first["positions"],
        first["velocities"],
        primary["positions"],
        primary["velocities"],
    )
    jx_metrics = _trajectory_metrics(
        first["positions"], first["velocities"], gm, epochs
    )
    backward_case = dict(case)
    backward = _jx_run(
        np.ascontiguousarray(first["positions"][-1]),
        np.ascontiguousarray(first["velocities"][-1]),
        gm,
        tuple(reversed(epochs)),
        backward_case,
        protocol,
    )
    reverse_error = _state_error(
        backward["positions"][-1:],
        backward["velocities"][-1:],
        positions[None, :, :],
        velocities[None, :, :],
    )
    analytic: dict[str, Any] | None = None
    if eccentricity is not None:
        analytic_positions, analytic_velocities = _analytic_binary(
            epochs, gm, eccentricity
        )
        analytic = {
            "jx": _state_error(
                first["positions"],
                first["velocities"],
                analytic_positions,
                analytic_velocities,
            ),
            "ias15_primary": _state_error(
                primary["positions"],
                primary["velocities"],
                analytic_positions,
                analytic_velocities,
            ),
        }
    gates = case["gates"]
    accounting = first["accounting"]
    gate_results = {
        "deterministic_replay": deterministic,
        "ias15_reference_stability": _passes_state(
            reference_stability, float(gates["reference_scaled_state_max"])
        ),
        "jx_vs_ias15_state": _passes_state(
            jx_reference, float(gates["jx_scaled_state_max"])
        ),
        "relative_energy": jx_metrics["relative_total_energy_max"]
        <= float(gates["relative_total_energy_max"]),
        "angular_momentum": jx_metrics["scaled_angular_momentum_max"]
        <= float(gates["scaled_angular_momentum_max"]),
        "linear_momentum": jx_metrics["scaled_linear_momentum_max"]
        <= float(gates["scaled_linear_momentum_max"]),
        "center_of_mass": jx_metrics["scaled_center_of_mass_linearity_max"]
        <= float(gates["scaled_center_of_mass_linearity_max"]),
        "reverse_time": _passes_state(
            reverse_error, float(gates["reverse_scaled_state_max"])
        ),
        "corrector_residual": first["maximum_corrector_residual"]
        <= first["corrector_threshold"],
        "terminal_force_accounting": accounting["terminal_force_reuses"]
        + accounting["terminal_force_sweeps"]
        == accounting["attempted_steps"] - accounting["nonconverged_retries"],
        "positive_retained_separation": jx_metrics[
            "minimum_retained_pair_separation"
        ]
        >= float(gates["minimum_retained_pair_separation"]),
    }
    if "maximum_retained_pair_separation" in gates:
        gate_results["challenge_geometry_reached"] = jx_metrics[
            "minimum_retained_pair_separation"
        ] <= float(gates["maximum_retained_pair_separation"])
    if analytic is not None:
        gate_results["jx_vs_analytic_state"] = _passes_state(
            analytic["jx"], float(gates["analytic_scaled_state_max"])
        )
        gate_results["ias15_vs_analytic_state"] = _passes_state(
            analytic["ias15_primary"],
            float(gates["analytic_scaled_state_max"]),
        )
    return {
        "status": "PASS_FIXED_GATES" if all(gate_results.values()) else "FAIL_FIXED_GATE",
        "fixture": case["fixture"],
        "body_count": len(gm),
        "checkpoint_count": len(epochs),
        "initial_epoch": epochs[0],
        "final_epoch": epochs[-1],
        "jx_digest": first["digest"],
        "jx_repeat_digest": repeat["digest"],
        "ias15_primary_digest": primary["digest"],
        "ias15_sensitivity_digest": sensitivity["digest"],
        "jx_accounting": accounting,
        "ias15_accounting": {
            "primary_steps_done": primary["steps_done"],
            "sensitivity_steps_done": sensitivity["steps_done"],
        },
        "corrector": {
            "maximum_residual": first["maximum_corrector_residual"],
            "threshold": first["corrector_threshold"],
        },
        "reference_stability": reference_stability,
        "jx_vs_ias15": jx_reference,
        "analytic": analytic,
        "jx_trajectory_metrics": jx_metrics,
        "reverse_time": reverse_error,
        "fixed_gates": gates,
        "gate_results": gate_results,
    }


def _failure_domain() -> dict[str, Any]:
    positions = np.asarray(((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)), dtype=np.float64)
    velocities = np.asarray(((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)), dtype=np.float64)
    gm = np.asarray((0.5, 0.5), dtype=np.float64)
    minimum_step_status: int | None = None
    try:
        integrate_gr15(
            positions,
            velocities,
            gm,
            GR15Spec(
                0.0,
                1.0,
                initial_step=0.5,
                minimum_step=0.5,
                maximum_step=0.5,
                epsilon=1.0e-9,
                maximum_iterations=1,
            ),
        )
    except GR15IntegrationError as exc:
        minimum_step_status = exc.status
    coincident = positions.copy()
    coincident[1] = coincident[0]
    singularity_status: int | None = None
    try:
        integrate_gr15(
            coincident,
            velocities,
            gm,
            GR15Spec(0.0, 1.0, initial_step=0.01, maximum_step=0.1),
        )
    except GR15IntegrationError as exc:
        singularity_status = exc.status
    oversized_rejected = False
    try:
        integrate_gr15(
            np.zeros((MAXIMUM_BODY_COUNT + 1, 3), dtype=np.float64),
            np.zeros((MAXIMUM_BODY_COUNT + 1, 3), dtype=np.float64),
            np.ones(MAXIMUM_BODY_COUNT + 1, dtype=np.float64),
            GR15Spec(0.0, 1.0),
        )
    except GR15ContractError:
        oversized_rejected = True
    gates = {
        "minimum_step_failed_closed": minimum_step_status == STATUS_MINIMUM_STEP,
        "coincident_state_failed_closed": singularity_status
        == STATUS_FORCE_SINGULARITY,
        "body_count_above_32_rejected": oversized_rejected,
    }
    return {
        "status": "PASS_FIXED_GATES" if all(gates.values()) else "FAIL_FIXED_GATE",
        "observed": {
            "minimum_step_status": minimum_step_status,
            "singularity_status": singularity_status,
            "oversized_rejected": oversized_rejected,
        },
        "gates": gates,
    }


def run(protocol: dict[str, Any]) -> dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        import rebound
    except ModuleNotFoundError as exc:
        raise AdversarialQualificationError("exact REBOUND 5.1.1 is required") from exc
    if (
        rebound.__version__ != protocol["rebound_version"]
        or rebound.__githash__ != protocol["rebound_githash"]
    ):
        raise AdversarialQualificationError("REBOUND runtime identity changed")
    identity = gr15_runtime_identity()
    if (
        jxplanetx.__version__ != protocol["package_version"]
        or identity["source_sha256"] != protocol["gr15_core_source_sha256"]
        or identity["method_id"] != "JX_GAUSS_RADAU15_V3"
        or identity["scientific_claim_state"] != "SCREENING_ONLY"
    ):
        raise AdversarialQualificationError("public GR15 runtime identity changed")
    cases = {
        name: _run_case(name, protocol["cases"][name], protocol, rebound)
        for name in protocol["case_order"]
    }
    failure_domain = _failure_domain()
    passed = all(case["status"] == "PASS_FIXED_GATES" for case in cases.values())
    passed = passed and failure_domain["status"] == "PASS_FIXED_GATES"
    return {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": "PASS_SCREENING_ONLY" if passed else "FAIL_FIXED_GATE",
        "scientific_claim_state": "SCREENING_ONLY",
        "claim_controls": protocol["claim_controls"],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "jxplanetx": jxplanetx.__version__,
            "rebound": rebound.__version__,
            "rebound_githash": rebound.__githash__,
            "omp_num_threads": os.environ["OMP_NUM_THREADS"],
            "gr15": identity,
        },
        "cases": cases,
        "failure_domain": failure_domain,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, protocol_identity = _read_protocol(args.protocol, root)
    if args.validate_only:
        print(json.dumps(protocol_identity, sort_keys=True))
        return 0
    if args.output is None:
        parser.error("--output is required unless --validate-only is used")
    report = run(protocol)
    report["protocol"] = protocol_identity
    report["semantic_content_sha256"] = _sha256(
        b"jx.gr15-v3.adversarial-report.v1\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0 if report["status"] == "PASS_SCREENING_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
