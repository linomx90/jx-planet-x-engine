#!/usr/bin/env python3
"""Accuracy-gated native JX CPU versus REBOUND WHFast timing screen.

The locked lane is one weak, hierarchical, mutual-Newtonian three-body
trajectory.  It reports the complete native JX numerical loop both with and
without an independent deterministic replay.  REBOUND is an external
GPL-family comparator.  Results are single-host screening evidence only.
"""

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

import jxplanetx as jx_package
from jxplanetx import fast_wisdom_holman as native_loop
from benchmarks import rebound_whfast_comparison as fixture
from jxplanetx import _wisdom_holman_cpu_v2 as native_g
from jxplanetx import __version__ as JX_VERSION
from jxplanetx.engine import wisdom_holman as public_wh


SCHEMA = "jxplanetx.fast-wh-rc6-rebound-race.report.v3"
PROTOCOL_SCHEMA = "jxplanetx.fast-wh-rc6-rebound-race.protocol.v3"
BENCHMARK_ID = "jx.fast-wh-rc6.rebound-whfast.100-period.v3"
REQUIRED_JX_VERSION = "0.6.0rc6"
REQUIRED_REBOUND_VERSION = "5.1.1"
REQUIRED_REBOUND_GITHASH = "33549d1d50d616a95a6d6a79e5e2c9c3b3730b1f"
THREAD_ENVIRONMENT = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
)
LANE_NAMES = ("jx_rc6_single_pass", "jx_rc6_replay", "rebound_whfast")
MAX_PROTOCOL_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_RELEASE_ARTIFACT_BYTES = 16 * 1024 * 1024
SPEED_RESOLUTION_FRACTION = 0.05


class NativeWisdomHolmanRaceError(RuntimeError):
    """The accuracy-gated timing screen could not be completed exactly."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise NativeWisdomHolmanRaceError(
            "report is not finite canonical JSON"
        ) from exc


def _file_identity(path: Path, root: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise NativeWisdomHolmanRaceError(f"required file is not regular: {path}")
    raw = resolved.read_bytes()
    if not 1 <= len(raw) <= MAX_SOURCE_BYTES:
        raise NativeWisdomHolmanRaceError(
            f"required file violates size bounds: {path}"
        )
    label = resolved.as_posix()
    if root is not None:
        try:
            label = resolved.relative_to(root.resolve(strict=True)).as_posix()
        except ValueError as exc:
            raise NativeWisdomHolmanRaceError(
                f"required file escapes project root: {path}"
            ) from exc
    return {"path": label, "size_bytes": len(raw), "sha256": _sha256(raw)}


def read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify the prospectively locked source-bound protocol."""

    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    if path.is_symlink() or not 1 <= len(raw) <= MAX_PROTOCOL_BYTES:
        raise NativeWisdomHolmanRaceError(
            "protocol must be a bounded regular file"
        )

    def reject(value: str) -> None:
        raise NativeWisdomHolmanRaceError(
            f"protocol contains nonfinite JSON {value}"
        )

    try:
        protocol = json.loads(raw, parse_constant=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeWisdomHolmanRaceError(
            "protocol must be finite UTF-8 JSON"
        ) from exc
    exact = {
        "schema": PROTOCOL_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": "LOCKED_AFTER_DISCLOSED_PILOTS_BEFORE_RECORDED_EXECUTION",
        "periods": 100,
        "step_divisor": 64,
        "samples_per_period": 4,
        "timed_repetitions": 9,
        "required_jx_version": REQUIRED_JX_VERSION,
        "required_rebound_version": REQUIRED_REBOUND_VERSION,
        "required_rebound_githash": REQUIRED_REBOUND_GITHASH,
        "native_replay_count": 1,
        "speed_resolution_fraction": SPEED_RESOLUTION_FRACTION,
    }
    if type(protocol) is not dict or any(
        protocol.get(name) != value for name, value in exact.items()
    ):
        raise NativeWisdomHolmanRaceError(
            "protocol fixed execution contract changed"
        )
    if protocol.get("lane_names") != list(LANE_NAMES):
        raise NativeWisdomHolmanRaceError("protocol lane roster changed")
    release_wheel = protocol.get("required_release_wheel")
    if (
        type(release_wheel) is not dict
        or set(release_wheel) != {"filename", "size_bytes", "sha256"}
        or release_wheel.get("filename")
        != "jxplanetx-0.6.0rc6-cp314-cp314-linux_x86_64.whl"
        or release_wheel.get("size_bytes") != 790897
        or release_wheel.get("sha256")
        != "12355a2b69733ee4c315813c9c1a58213cb489716288c3cf6ae65db63c79955e"
    ):
        raise NativeWisdomHolmanRaceError(
            "protocol release-wheel identity changed"
        )
    if protocol.get("accuracy_and_conservation_gate") != dict(
        fixture.LONG_DESCRIPTIVE_ENVELOPE
    ):
        raise NativeWisdomHolmanRaceError(
            "protocol accuracy or conservation gate changed"
        )
    if protocol.get("solver_prerequisite_gate") != {
        "dense_series_argument_minimum": -0.5,
        "dense_series_argument_maximum": 0.5,
        "dense_series_grid_points": 1001,
        "beta_values": [1.0e-12, 1.0e-6, 1.0, 1.0e6, 1.0e12],
        "require_bitwise_equality_to_exact_v1_bundle": True,
        "require_series_term_reduction": True,
        "require_trigonometric_branch_equality": True,
        "require_invalid_domain_rejection": True,
        "require_100_period_public_jx_bitwise_parity": True,
        "require_deterministic_exact_replay": True,
    }:
        raise NativeWisdomHolmanRaceError(
            "protocol solver prerequisite gate changed"
        )
    orders = protocol.get("execution_order")
    if (
        type(orders) is not list
        or len(orders) != protocol["timed_repetitions"]
        or any(
            type(order) is not list
            or len(order) != len(LANE_NAMES)
            or set(order) != set(LANE_NAMES)
            for order in orders
        )
    ):
        raise NativeWisdomHolmanRaceError(
            "protocol execution order is not balanced permutations"
        )
    roster = protocol.get("source_roster")
    if type(roster) is not list or not roster:
        raise NativeWisdomHolmanRaceError("protocol source roster is missing")
    for expected in roster:
        if type(expected) is not dict or set(expected) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise NativeWisdomHolmanRaceError(
                "protocol source identity schema changed"
            )
        if _file_identity(root / expected["path"], root) != expected:
            raise NativeWisdomHolmanRaceError(
                f"protocol source binding changed: {expected.get('path')}"
            )
    return protocol, {
        "path": resolved.as_posix(),
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }


def verify_release_wheel(
    path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise NativeWisdomHolmanRaceError(
            "release wheel must be one regular file"
        )
    raw = resolved.read_bytes()
    observed = {
        "filename": resolved.name,
        "size_bytes": len(raw),
        "sha256": _sha256(raw),
    }
    if not 1 <= len(raw) <= MAX_RELEASE_ARTIFACT_BYTES or observed != expected:
        raise NativeWisdomHolmanRaceError(
            "release wheel does not match the protocol identity"
        )
    observed["path"] = resolved.as_posix()
    return observed


def _require_runtime_environment() -> tuple[Any, dict[str, Any]]:
    observed = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    if any(value != "1" for value in observed.values()):
        raise NativeWisdomHolmanRaceError(
            "MKL/NUMEXPR/OMP/OPENBLAS thread variables must all equal 1"
        )
    if JX_VERSION != REQUIRED_JX_VERSION:
        raise NativeWisdomHolmanRaceError(
            f"exact JX {REQUIRED_JX_VERSION} is required"
        )
    rebound = fixture._load_rebound()
    runtime = fixture._rebound_runtime_provenance(rebound)
    if (
        runtime.get("rebound_version") != REQUIRED_REBOUND_VERSION
        or runtime.get("rebound_githash") != REQUIRED_REBOUND_GITHASH
    ):
        raise NativeWisdomHolmanRaceError(
            "REBOUND version or source identity changed"
        )
    return rebound, {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "jx": JX_VERSION,
        "thread_environment": observed,
        "rebound": runtime,
        "native": native_loop.fast_wisdom_holman_runtime_identity(),
    }


def _timing_summary(values: list[float]) -> dict[str, Any]:
    if not values or any(
        type(value) is not float or not math.isfinite(value) or value <= 0.0
        for value in values
    ):
        raise NativeWisdomHolmanRaceError("timing samples are invalid")
    return {
        "samples_seconds": values,
        "minimum_seconds": min(values),
        "median_seconds": float(statistics.median(values)),
        "maximum_seconds": max(values),
        "population_coefficient_of_variation": float(
            statistics.pstdev(values) / statistics.fmean(values)
        ),
    }


def _speed_classification(
    jx_seconds: float, rebound_seconds: float
) -> dict[str, Any]:
    relative = rebound_seconds / jx_seconds
    if relative >= 1.0 + SPEED_RESOLUTION_FRACTION:
        classification = "JX_FASTER_ON_RECORDED_HOST_AND_WORKLOAD"
    elif relative <= 1.0 / (1.0 + SPEED_RESOLUTION_FRACTION):
        classification = "REBOUND_FASTER_ON_RECORDED_HOST_AND_WORKLOAD"
    else:
        classification = "NO_RESOLVED_DIFFERENCE_AT_FIVE_PERCENT_RULE"
    return {
        "classification": classification,
        "jx_throughput_relative_to_rebound": relative,
        "rebound_throughput_relative_to_jx": 1.0 / relative,
        "resolution_fraction": SPEED_RESOLUTION_FRACTION,
        "portable_or_general_claim": False,
    }


def _accuracy(candidate: Any, reference: Any) -> dict[str, Any]:
    position = fixture._cartesian_metrics(candidate.positions, reference.positions)
    velocity = fixture._cartesian_metrics(candidate.velocities, reference.velocities)
    phase = fixture._phase_proxy_metrics(candidate.positions, reference.positions)
    invariants = fixture._invariant_metrics(candidate.positions, candidate.velocities)
    accepted = fixture._within_long_envelope(
        position, velocity, phase, invariants
    )
    return {
        "position": position,
        "velocity": velocity,
        "phase_proxy": phase,
        "invariants": invariants,
        "envelope": dict(fixture.LONG_DESCRIPTIVE_ENVELOPE),
        "within_preexisting_descriptive_envelope": accepted,
        "reference": "REBOUND_IAS15_EPSILON_1E_MINUS_13_MODEL_OUTPUT",
        "reference_truth_claimed": False,
    }


def _native_lane(workspace: Any, replay: bool) -> tuple[Any, float]:
    started = time.perf_counter()
    result = native_loop.integrate_fast_wisdom_holman_workspace(workspace, exact_replay=replay)
    return result, float(time.perf_counter() - started)


def _solver_prerequisite_gate() -> dict[str, Any]:
    """Run the source-locked v2 accuracy and failure-domain checks untimed."""

    arguments = np.linspace(-0.5, 0.5, 1001, dtype=np.float64)
    arguments = np.concatenate(
        (
            np.asarray(
                (
                    -0.5,
                    math.nextafter(-0.5, 0.0),
                    -0.0,
                    0.0,
                    math.nextafter(0.5, 0.0),
                    0.5,
                ),
                dtype=np.float64,
            ),
            arguments,
        )
    )
    dense_cases = 0
    adaptive_terms = 0
    exact_terms = 0
    for beta in (1.0e-12, 1.0e-6, 1.0, 1.0e6, 1.0e12):
        root = math.sqrt(beta)
        for argument in arguments:
            anomaly = float(argument) / root
            adaptive = native_g.adaptive_g_bundle(beta, anomaly)
            exact = native_g.exact_g_bundle(beta, anomaly)
            if adaptive[:5] != exact[:5] or exact[5] != 256:
                raise NativeWisdomHolmanRaceError(
                    "v2 universal-G dense-grid bitwise gate failed"
                )
            dense_cases += 1
            adaptive_terms += int(adaptive[5])
            exact_terms += int(exact[5])
    if not adaptive_terms < exact_terms:
        raise NativeWisdomHolmanRaceError(
            "v2 universal-G series term reduction gate failed"
        )

    trigonometric_cases = 0
    for beta in (1.0e-12, 1.0, 1.0e12):
        root = math.sqrt(beta)
        for argument in (-2.0, -0.5001, 0.5001, 2.0):
            anomaly = argument / root
            if native_g.adaptive_g_bundle(
                beta, anomaly
            ) != native_g.exact_g_bundle(beta, anomaly):
                raise NativeWisdomHolmanRaceError(
                    "v2 universal-G trigonometric branch gate failed"
                )
            trigonometric_cases += 1

    invalid_cases = (
        (0.0, 0.1),
        (-1.0, 0.1),
        (1.0, math.inf),
        (math.inf, 0.1),
    )
    rejected = 0
    for beta, anomaly in invalid_cases:
        try:
            native_g.adaptive_g_bundle(beta, anomaly)
        except FloatingPointError:
            rejected += 1
        else:
            raise NativeWisdomHolmanRaceError(
                "v2 universal-G invalid-domain gate failed"
            )
    return {
        "status": "PASS",
        "dense_series_cases": dense_cases,
        "dense_series_bitwise_equal_to_exact_v1_bundle": True,
        "adaptive_series_terms": adaptive_terms,
        "exact_v1_series_terms": exact_terms,
        "series_term_reduction_fraction": 1.0
        - adaptive_terms / exact_terms,
        "trigonometric_cases": trigonometric_cases,
        "trigonometric_branch_equal_to_exact_v1_bundle": True,
        "invalid_domain_cases": len(invalid_cases),
        "invalid_domain_cases_rejected": rejected,
    }


def run(
    *,
    periods: int,
    step_divisor: int,
    repetitions: int,
    orders: tuple[tuple[str, ...], ...],
    require_public_parity: bool,
) -> dict[str, Any]:
    """Execute one source-bound or development accuracy/timing screen."""

    for label, value in (
        ("periods", periods),
        ("step_divisor", step_divisor),
        ("repetitions", repetitions),
    ):
        if type(value) is not int or value <= 0:
            raise NativeWisdomHolmanRaceError(f"{label} must be positive")
    if type(require_public_parity) is not bool:
        raise NativeWisdomHolmanRaceError(
            "require_public_parity must be a built-in bool"
        )
    if len(orders) != repetitions or any(
        len(order) != len(LANE_NAMES) or set(order) != set(LANE_NAMES)
        for order in orders
    ):
        raise NativeWisdomHolmanRaceError("execution order contract changed")

    rebound, runtime = _require_runtime_environment()
    solver_gate = _solver_prerequisite_gate()
    profile = fixture.Profile(1, periods, 4)
    snapshot, plan = fixture._state_and_plan(profile)
    spec = fixture._wh_spec(profile, "long", step_divisor)
    workspace = native_loop.prepare_fast_wisdom_holman_workspace(snapshot, plan, spec)

    parity: dict[str, Any]
    if require_public_parity:
        public = public_wh.integrate_wisdom_holman_trajectory(
            snapshot, plan, spec
        )
        native = native_loop.integrate_fast_wisdom_holman_workspace(workspace, exact_replay=True)
        position_equal = np.array_equal(
            native.positions, np.stack(public.positions)
        )
        velocity_equal = np.array_equal(
            native.velocities, np.stack(public.velocities)
        )
        epoch_equal = np.array_equal(
            native.checkpoint_epochs,
            np.asarray(public.checkpoint_epochs, dtype=np.float64),
        )
        if not (position_equal and velocity_equal and epoch_equal):
            raise NativeWisdomHolmanRaceError(
                "native complete loop differs from the public JX trajectory"
            )
        parity = {
            "executed": True,
            "passed": True,
            "position_bitwise_equal": position_equal,
            "velocity_bitwise_equal": velocity_equal,
            "epoch_bitwise_equal": epoch_equal,
            "native_result_content_sha256": native.result_content_sha256,
            "public_result_content_sha256": public.result_content_sha256,
        }
    else:
        parity = {"executed": False, "passed": None}

    runners: dict[str, Callable[[], tuple[Any, float]]] = {
        "jx_rc6_single_pass": lambda: _native_lane(workspace, False),
        "jx_rc6_replay": lambda: _native_lane(workspace, True),
        "rebound_whfast": lambda: _timed_rebound_lane(
            profile, step_divisor, rebound, runtime["rebound"]
        ),
    }
    results: dict[str, list[Any]] = {name: [] for name in LANE_NAMES}
    outer_timings: dict[str, list[float]] = {name: [] for name in LANE_NAMES}
    numerical_timings: dict[str, list[float]] = {
        name: [] for name in LANE_NAMES
    }
    for order in orders:
        for name in order:
            result, elapsed = runners[name]()
            results[name].append(result)
            outer_timings[name].append(elapsed)
            if name == "jx_rc6_single_pass":
                numerical_timings[name].append(result.primary_seconds)
            elif name == "jx_rc6_replay":
                numerical_timings[name].append(
                    result.primary_seconds + result.replay_seconds
                )
            else:
                numerical_timings[name].append(result.integration_seconds)

    native_digests = {
        result.result_content_sha256
        for name in ("jx_rc6_single_pass", "jx_rc6_replay")
        for result in results[name]
    }
    rebound_digests = {
        fixture._lane_content_sha256(result)
        for result in results["rebound_whfast"]
    }
    if len(native_digests) != 1 or len(rebound_digests) != 1:
        raise NativeWisdomHolmanRaceError(
            "a timed lane changed its retained numerical bytes"
        )

    native_result = results["jx_rc6_single_pass"][0]
    rebound_result = results["rebound_whfast"][0]
    native_reference = fixture._ias15_lane(
        "rebound_ias15_native_epoch_reference",
        "long",
        native_result.checkpoint_epochs,
        rebound,
        runtime["rebound"],
    )
    rebound_reference = fixture._ias15_lane(
        "rebound_ias15_whfast_observed_epoch_reference",
        "long",
        rebound_result.observed_epochs,
        rebound,
        runtime["rebound"],
    )
    native_accuracy = _accuracy(native_result, native_reference)
    rebound_accuracy = _accuracy(rebound_result, rebound_reference)
    accuracy_matched = bool(
        native_accuracy["within_preexisting_descriptive_envelope"]
        and rebound_accuracy["within_preexisting_descriptive_envelope"]
    )

    timing = {
        name: {
            "end_to_end": _timing_summary(outer_timings[name]),
            "numerical_execution": _timing_summary(numerical_timings[name]),
        }
        for name in LANE_NAMES
    }
    rebound_median = timing["rebound_whfast"]["end_to_end"][
        "median_seconds"
    ]
    raw_median = timing["jx_rc6_single_pass"]["end_to_end"]["median_seconds"]
    replay_median = timing["jx_rc6_replay"]["end_to_end"][
        "median_seconds"
    ]
    return {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": (
            "PASS_ACCURACY_AND_DETERMINISM_SCREENING_ONLY"
            if accuracy_matched and (not require_public_parity or parity["passed"])
            else "FAIL_ACCURACY_OR_PARITY_GATE"
        ),
        "scientific_claim_state": "SCREENING_ONLY",
        "workload": {
            "model": "WEAK_HIERARCHY_MUTUAL_NEWTONIAN_THREE_BODY",
            "periods": periods,
            "step_divisor": step_divisor,
            "fixed_steps": spec.completed_steps,
            "retained_checkpoints": len(spec.checkpoint_step_indices),
            "jx_raw_complete_native_map_passes": 1,
            "jx_replay_complete_native_map_passes": 2,
            "rebound_whfast_safe_mode": 1,
            "rebound_whfast_coordinates": "jacobi",
        },
        "runtime": runtime,
        "solver_prerequisite_gate": solver_gate,
        "public_jx_bitwise_parity_gate": parity,
        "determinism": {
            "native_result_content_sha256": next(iter(native_digests)),
            "rebound_lane_content_sha256": next(iter(rebound_digests)),
            "all_timed_repetitions_stable": True,
        },
        "accuracy": {
            "both_within_same_preexisting_envelope": accuracy_matched,
            "jx_rc6_supported_api": native_accuracy,
            "rebound_whfast": rebound_accuracy,
        },
        "timing_scope": {
            "end_to_end_primary_comparison": True,
            "jx_includes_output_allocation_hashing_and_result_construction": True,
            "jx_raw_includes_exact_replay": False,
            "jx_replay_includes_exact_replay": True,
            "rebound_includes_configuration_checkpoint_readback_and_evidence_construction": True,
            "state_fixture_construction_excluded": True,
            "ias15_accuracy_reference_excluded": True,
        },
        "timings": timing,
        "speed_classification": {
            "jx_rc6_single_pass_vs_rebound_whfast": _speed_classification(
                raw_median, rebound_median
            ),
            "jx_rc6_replay_vs_rebound_whfast": _speed_classification(
                replay_median, rebound_median
            ),
        },
        "claim_controls": {
            "general_solver_superiority_claimed": False,
            "portable_performance_claimed": False,
            "scientific_qualification_claimed": False,
            "production_authorized": False,
            "public_engine_backend_changed": False,
            "supported_opt_in_api_added": True,
            "single_host_workload_specific_timing_only": True,
            "rebound_parallel_workers_tested": False,
            "long_solar_system_integration_tested": False,
            "whfast_mercurius_trace_workload_roster_complete": False,
        },
    }


def _timed_rebound_lane(
    profile: Any,
    step_divisor: int,
    rebound: Any,
    runtime: dict[str, Any],
) -> tuple[Any, float]:
    started = time.perf_counter()
    lane = fixture._whfast_lane(
        profile, "long", step_divisor, rebound, runtime
    )
    return lane, float(time.perf_counter() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--release-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol, protocol_identity = read_protocol(args.protocol, root)
    package_path = Path(jx_package.__file__).resolve(strict=True)
    if package_path.is_relative_to(root):
        raise NativeWisdomHolmanRaceError(
            "final rc6 race requires an installed release artifact, not source"
        )
    release_wheel_identity = verify_release_wheel(
        args.release_wheel, protocol["required_release_wheel"]
    )
    report = run(
        periods=protocol["periods"],
        step_divisor=protocol["step_divisor"],
        repetitions=protocol["timed_repetitions"],
        orders=tuple(tuple(order) for order in protocol["execution_order"]),
        require_public_parity=True,
    )
    report["protocol"] = protocol_identity
    report["release_wheel"] = release_wheel_identity
    report["installed_package_path"] = package_path.as_posix()
    report["protocol_prior_evidence_disclosure"] = protocol[
        "prior_evidence_disclosure"
    ]
    report["execution_order"] = protocol["execution_order"]
    report["semantic_content_sha256"] = _sha256(
        b"jx.fast-wh-rc6-rebound-race.report.v3\0" + _canonical(report)
    )
    args.output.write_bytes(_canonical(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
