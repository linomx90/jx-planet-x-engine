#!/usr/bin/env python3
"""Locked rc14 supported-API dispatch crossover acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any

import numpy as np

from benchmarks import jx_gr15_cuda_rebound_ias15 as matrix
from jxplanetx import __version__
from jxplanetx.gr15_eih_1pn_batch import (
    CPU_BACKEND_ID,
    CUDA_BACKEND_ID,
    integrate_gr15_eih_1pn_batch,
)


SCHEMA = "jxplanetx.gr15-dispatch-rc14-acceptance-report.v1"
PROTOCOL_SCHEMA = "jxplanetx.gr15-dispatch-rc14-acceptance-protocol.v1"
BENCHMARK_ID = "jx.gr15.supported-dispatch.rc14.v1"
SYSTEM_COUNTS = (64, 96, 128)
HORIZON_DAYS = 3_652.5
CONTROL = 1.0e-6
STRICT_CONTROL = 1.0e-10
POSITION_GATE_KM = 1.0
VELOCITY_GATE_KM_S = 1.0e-7
REPETITIONS = 3
CPU_WORKERS = 8


class AcceptanceError(RuntimeError):
    """The locked supported dispatch acceptance is invalid."""


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if path.is_symlink() or not resolved.is_file():
        raise AcceptanceError(f"source is not regular: {path}")
    try:
        relative = resolved.relative_to(root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise AcceptanceError(f"source escapes repository: {path}") from exc
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _read_protocol(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve(strict=True)
    raw = resolved.read_bytes()
    protocol = json.loads(raw)
    if (
        type(protocol) is not dict
        or protocol.get("schema") != PROTOCOL_SCHEMA
        or protocol.get("benchmark_id") != BENCHMARK_ID
        or protocol.get("status") != "LOCKED_BEFORE_FIRST_TIMING"
        or protocol.get("package_version") != "0.6.0rc14"
        or protocol.get("system_counts") != list(SYSTEM_COUNTS)
        or protocol.get("horizon_days") != HORIZON_DAYS
        or protocol.get("candidate_control") != CONTROL
        or protocol.get("strict_control") != STRICT_CONTROL
        or protocol.get("position_gate_km") != POSITION_GATE_KM
        or protocol.get("velocity_gate_km_s") != VELOCITY_GATE_KM_S
        or protocol.get("timed_repetitions") != REPETITIONS
        or protocol.get("cpu_workers") != CPU_WORKERS
    ):
        raise AcceptanceError("fixed protocol identity changed")
    for expected in protocol.get("source_roster", ()):
        if _file_identity(root / expected["path"], root) != expected:
            raise AcceptanceError(f"source binding changed: {expected.get('path')}")
    return protocol, {
        "path": resolved.as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _summary(samples: list[float]) -> dict[str, Any]:
    if len(samples) != REPETITIONS or any(
        not math.isfinite(value) or value <= 0.0 for value in samples
    ):
        raise AcceptanceError("timing samples are invalid")
    return {
        "samples_seconds": samples,
        "minimum_seconds": min(samples),
        "median_seconds": statistics.median(samples),
        "maximum_seconds": max(samples),
    }


def _timed(
    positions: np.ndarray,
    velocities: np.ndarray,
    gm: np.ndarray,
    spec: Any,
    parameters: Any,
    backend: str,
) -> tuple[float, Any]:
    started = time.perf_counter_ns()
    result = integrate_gr15_eih_1pn_batch(
        positions,
        velocities,
        gm,
        spec,
        parameters,
        backend=backend,
        cpu_workers=CPU_WORKERS,
    )
    return (time.perf_counter_ns() - started) * 1.0e-9, result


def run(root: Path, protocol_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    protocol, protocol_identity = _read_protocol(protocol_path, root)
    if __version__ != "0.6.0rc14":
        raise AcceptanceError("package version changed")
    try:
        import cupy as cp
    except Exception as exc:
        raise AcceptanceError(f"CuPy runtime unavailable: {exc}") from exc
    base_positions, base_velocities, gm = matrix._load_initial_state(
        (root / matrix.INITIAL_STATE_PATH).resolve(strict=True)
    )
    epochs = matrix._epochs(HORIZON_DAYS)
    spec = matrix._spec(epochs, CONTROL)
    parameters = matrix._parameters()
    reference = matrix._strict_jx_reference(
        base_positions, base_velocities, gm, epochs
    )

    # Compile CUDA before any timing sample.
    matrix._run_jx_cuda(
        cp,
        cp.asarray(base_positions[np.newaxis]),
        cp.asarray(base_velocities[np.newaxis]),
        cp.asarray(gm),
        matrix._epochs(1.0),
        CONTROL,
    )

    rows: list[dict[str, Any]] = []
    for systems in SYSTEM_COUNTS:
        positions, velocities, phases = matrix._ensemble(
            base_positions, base_velocities, systems
        )
        expected_positions = matrix._rotated_reference(reference["positions"], phases)
        expected_velocities = matrix._rotated_reference(reference["velocities"], phases)
        samples = {"cpu": [], "cuda": []}
        last: dict[str, Any] = {}
        orders: list[list[str]] = []
        for repetition in range(REPETITIONS):
            order = ["cpu", "cuda"] if repetition % 2 == 0 else ["cuda", "cpu"]
            orders.append(order)
            for backend in order:
                elapsed, result = _timed(
                    positions, velocities, gm, spec, parameters, backend
                )
                samples[backend].append(elapsed)
                last[backend] = result
        errors = {
            backend: matrix._errors(
                result.checkpoint_positions_km,
                result.checkpoint_velocities_km_s,
                expected_positions,
                expected_velocities,
            )
            for backend, result in last.items()
        }
        workload = {
            "position_gate_km": POSITION_GATE_KM,
            "velocity_gate_km_s": VELOCITY_GATE_KM_S,
        }
        gates = {
            "cpu_accuracy": matrix._passes(errors["cpu"], workload),
            "cuda_accuracy": matrix._passes(errors["cuda"], workload),
            "cpu_backend_identity": last["cpu"].decision.backend_id == CPU_BACKEND_ID,
            "cuda_backend_identity": last["cuda"].decision.backend_id == CUDA_BACKEND_ID,
            "cpu_cuda_common_gate": matrix._passes(
                matrix._errors(
                    last["cpu"].checkpoint_positions_km,
                    last["cpu"].checkpoint_velocities_km_s,
                    last["cuda"].checkpoint_positions_km,
                    last["cuda"].checkpoint_velocities_km_s,
                ),
                workload,
            ),
        }
        if not all(gates.values()):
            raise AcceptanceError(f"candidate gate failed at {systems}: {gates}")
        cpu_timing = _summary(samples["cpu"])
        cuda_timing = _summary(samples["cuda"])

        auto_elapsed, auto_result = _timed(
            positions, velocities, gm, spec, parameters, "auto"
        )
        expected_auto = "cpu" if systems < 128 else "cuda"
        expected_backend_id = (
            CPU_BACKEND_ID if expected_auto == "cpu" else CUDA_BACKEND_ID
        )
        auto_matches = (
            auto_result.decision.backend_id == expected_backend_id
            and np.array_equal(
                auto_result.checkpoint_positions_km,
                last[expected_auto].checkpoint_positions_km,
            )
            and np.array_equal(
                auto_result.checkpoint_velocities_km_s,
                last[expected_auto].checkpoint_velocities_km_s,
            )
        )
        if not auto_matches:
            raise AcceptanceError(f"automatic route mismatch at {systems}")
        cpu_median = float(cpu_timing["median_seconds"])
        cuda_median = float(cuda_timing["median_seconds"])
        rows.append(
            {
                "systems": systems,
                "execution_orders": orders,
                "accuracy": errors,
                "gates": {**gates, "automatic_route_exact_match": auto_matches},
                "timing": {"cpu": cpu_timing, "cuda": cuda_timing},
                "cuda_speedup_over_cpu": cpu_median / cuda_median,
                "automatic": {
                    "elapsed_seconds": auto_elapsed,
                    "backend_id": auto_result.decision.backend_id,
                    "reason": auto_result.decision.reason,
                    "performance_calibrated": auto_result.decision.performance_calibrated,
                    "threshold_extrapolated": auto_result.decision.threshold_extrapolated,
                },
            }
        )

    stable_crossover = next(
        (
            row["systems"]
            for index, row in enumerate(rows)
            if all(item["cuda_speedup_over_cpu"] > 1.0 for item in rows[index:])
        ),
        None,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "status": "PASS",
        "scientific_claim_state": "SCREENING_ONLY",
        "general_superiority_claimed": False,
        "portable_crossover_claimed": False,
        "protocol": protocol_identity,
        "source_roster": protocol["source_roster"],
        "runtime": {
            "jx_version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": matrix._cpu_model(),
            "gr15_cuda": matrix.gr15_eih_1pn_cuda_runtime_identity(),
        },
        "configuration": {
            "system_counts": list(SYSTEM_COUNTS),
            "horizon_days": HORIZON_DAYS,
            "candidate_control": CONTROL,
            "strict_control": STRICT_CONTROL,
            "cpu_workers": CPU_WORKERS,
            "timed_repetitions": REPETITIONS,
        },
        "rows": rows,
        "observed_stable_cuda_crossover": stable_crossover,
    }
    report["semantic_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("benchmarks/jx_gr15_dispatch_rc14_protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.validate_only:
        protocol, identity = _read_protocol(args.protocol, root)
        value = {
            "status": "VALID",
            "protocol": identity,
            "source_count": len(protocol["source_roster"]),
        }
    else:
        value = run(root, args.protocol)
    raw = _canonical(value)
    args.output.write_bytes(raw)
    print(raw.decode("ascii"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
