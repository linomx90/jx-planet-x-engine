#!/usr/bin/env python3
"""Bounded exact-parity and throughput screen for episode route waves."""

from __future__ import annotations

import os


NUMERICAL_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _name, _value in NUMERICAL_THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value

import argparse
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any, Callable

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
)
from jxplanetx.cuda_restricted_wh_episode import (
    EPISODE_CLAIMS,
    EPISODE_METHOD_ID,
    advance_cuda_restricted_wh_episodes,
)
from jxplanetx.cuda_restricted_wh_hybrid import (
    HYBRID_CLAIMS,
    HYBRID_METHOD_ID,
    advance_cuda_restricted_wh_hybrid,
)
from jxplanetx.engine import BackendSpec, Provenance


SCHEMA = "jx.cuda-restricted-wh.episode-qualification.v1"
PROTOCOL_DOMAIN = b"jx.cuda-restricted-wh.episode.protocol.v1\0"
RESULT_DOMAIN = b"jx.cuda-restricted-wh.episode.result.v1\0"
STATE_DOMAIN = b"jx.cuda-restricted-wh.episode.state.v1\0"
ROUTE_DOMAIN = b"jx.cuda-restricted-wh.episode.routes.v1\0"
RUN_ID = "restricted-wh-episode-qualification-v1"
ROUTED_LANES = 48
QUIET_LANES = 16
STEP_COUNT = 8
STEP_YEARS = 0.001
ROUTE_WORKER_COUNT = 6
PHYSICAL_CPU_AFFINITY = (0, 1, 2, 3, 4, 5)
REPEATS = 3
EXPECTED_ROUTE_COUNT = ROUTED_LANES * STEP_COUNT
EXPECTED_EPISODE_COUNT = ROUTED_LANES
MINIMUM_REFERENCE_SIX_WORKER_SPEEDUP = 2.0
MINIMUM_EPISODE_VS_REFERENCE_SPEEDUP = 1.03
MAXIMUM_SAMPLE_CV = 0.10
CLAIMS = {
    "production_qualified": False,
    "science_result": False,
    "physical_encounter_rate_represented": False,
    "long_horizon_accuracy_qualified": False,
    "long_horizon_completion_projected": False,
    "disk_checkpoint_restart_qualified": False,
    "cross_runtime_bitwise_reproducible": False,
}
SCHEDULE_SHA256 = hashlib.sha256(
    b"jx-restricted-wh-episode-qualification-v1"
).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _content_record(record: dict[str, Any], domain: bytes) -> dict[str, Any]:
    payload = dict(record)
    payload.pop("content_sha256", None)
    payload["content_sha256"] = hashlib.sha256(
        domain + _canonical_bytes(payload)
    ).hexdigest()
    return payload


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _provenance(label: str) -> Provenance:
    return Provenance(
        source_id=f"jx-qualification.{label}",
        citation="Synthetic deterministic episode throughput fixture",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _tile() -> FrozenSourcePathTile:
    zeros = np.zeros((STEP_COUNT + 1, 1, 3), dtype=np.float64)
    return FrozenSourcePathTile(
        tile_id="synthetic-route-episode",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=STEP_YEARS,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        source_relative_positions_AU=zeros,
        source_relative_velocities_AU_per_year=zeros.copy(),
        source_gm_AU3_per_year2=np.asarray(
            (4.0 * math.pi * math.pi,), dtype=np.float64
        ),
        source_encounter_radii_AU=np.asarray((0.997,), dtype=np.float64),
        certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
            (100.0,), dtype=np.float64
        ),
        provenance=_provenance("source-tile"),
    )


def _state(cp: Any) -> CUDARestrictedWHState:
    lane_count = ROUTED_LANES + QUIET_LANES
    radii = cp.concatenate(
        (
            cp.ones(ROUTED_LANES, dtype=cp.float64),
            cp.full(QUIET_LANES, 1.5, dtype=cp.float64),
        )
    )
    positions = cp.zeros((lane_count, 3), dtype=cp.float64)
    velocities = cp.zeros_like(positions)
    positions[:, 0] = radii
    velocities[:, 1] = cp.sqrt(cp.float64(4.0 * math.pi * math.pi) / radii)
    return CUDARestrictedWHState(
        snapshot_id="synthetic-route-episode",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=STEP_YEARS,
        time_scale="TDB",
        frame="ICRF",
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id="AU-YEAR",
        source_ids=("Sun",),
        primary_source_id="Sun",
        source_schedule_sha256=SCHEDULE_SHA256,
        tracer_relative_positions_AU=positions,
        tracer_relative_velocities_AU_per_year=velocities,
        tracer_indices=cp.arange(lane_count, dtype=cp.uint64),
        provenance=_provenance("tracer-state"),
    )


def _array_digest(digest: Any, label: str, value: np.ndarray) -> None:
    canonical = np.ascontiguousarray(value)
    encoded_label = label.encode("ascii")
    digest.update(len(encoded_label).to_bytes(4, "little"))
    digest.update(encoded_label)
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(len(canonical.shape).to_bytes(4, "little"))
    for dimension in canonical.shape:
        digest.update(int(dimension).to_bytes(8, "little"))
    raw = canonical.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def _state_digest(cp: Any, result: Any) -> str:
    digest = hashlib.sha256(STATE_DOMAIN)
    digest.update(int(result.state.global_source_step).to_bytes(8, "little"))
    for label, value in (
        (
            "positions",
            cp.asnumpy(result.state.tracer_relative_positions_AU),
        ),
        (
            "velocities",
            cp.asnumpy(result.state.tracer_relative_velocities_AU_per_year),
        ),
        ("indices", cp.asnumpy(result.state.tracer_indices)),
        ("minimum", cp.asnumpy(result.minimum_source_distance_AU)),
    ):
        _array_digest(digest, label, value)
    digest.update(int(result.maximum_kepler_iterations).to_bytes(8, "little"))
    return digest.hexdigest()


def _route_digest(records: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(ROUTE_DOMAIN)
    digest.update(len(records).to_bytes(8, "little"))
    for record in records:
        route = record.route_result
        metadata = {
            "map_step_index": record.map_step_index,
            "lane_index": record.lane_index,
            "tracer_index": record.tracer_index,
            "guard_phase": record.guard_phase,
            "guard_reason": record.guard_reason,
            "guard_source_index": record.guard_source_index,
            "source_tile_sha256": route.source_tile_sha256,
            "source_segment_sha256": route.source_segment_sha256,
            "global_source_step": route.global_source_step,
            "source_steps_per_map_step": route.source_steps_per_map_step,
            "accepted_step_count": route.accepted_step_count,
            "derivative_evaluation_count": route.derivative_evaluation_count,
            "minimum_evaluated_hex": (
                route.minimum_evaluated_stage_distance_AU.hex()
            ),
            "minimum_committed_hex": (
                route.minimum_committed_stage_distance_AU.hex()
            ),
            "maximum_accepted_phase_hex": (
                route.maximum_accepted_primary_dynamical_phase_radians.hex()
            ),
            "maximum_committed_phase_hex": (
                route.maximum_committed_primary_dynamical_phase_radians.hex()
            ),
            "completed_map_fraction_hex": route.completed_map_fraction.hex(),
            "events": len(route.events),
        }
        encoded = _canonical_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        _array_digest(
            digest,
            "endpoint_position",
            route.endpoint_relative_position_AU,
        )
        _array_digest(
            digest,
            "endpoint_velocity",
            route.endpoint_relative_velocity_AU_per_year,
        )
    return digest.hexdigest()


def _run(
    cp: Any,
    advance: Callable[..., Any],
    state: CUDARestrictedWHState,
    tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    workers: int,
) -> tuple[Any, float]:
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    result = advance(state, tile, spec, route_worker_count=workers)
    cp.cuda.get_current_stream().synchronize()
    return result, time.perf_counter() - started


def _cv(samples: list[float]) -> float:
    mean = statistics.mean(samples)
    return statistics.stdev(samples) / mean if len(samples) > 1 else 0.0


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/restricted_wh_episode_qualification_v1"),
    )
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("output root already exists; qualification is create-only")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(PHYSICAL_CPU_AFFINITY))
    if tuple(sorted(os.sched_getaffinity(0))) != PHYSICAL_CPU_AFFINITY:
        parser.error("process affinity is not the six declared physical cores")

    import cupy as cp

    args.output_root.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    source_paths = {
        "benchmark": Path(__file__).resolve(),
        "episode": root / "src/jxplanetx/cuda_restricted_wh_episode.py",
        "hybrid": root / "src/jxplanetx/cuda_restricted_wh_hybrid.py",
        "restricted_wh": root / "src/jxplanetx/cuda_restricted_wh.py",
        "route": root / "src/jxplanetx/restricted_wh_route.py",
        "event_locator": root / "src/jxplanetx/cuda_tracer_events.py",
        "all_conic": root / "src/jxplanetx/_all_conic_kepler.py",
    }
    tile = _tile()
    state = _state(cp)
    spec = FixedStepCUDARestrictedWHSpec(
        STEP_COUNT, 1, BackendSpec("cupy", "cuda:0", 128)
    )
    protocol = _content_record(
        {
            "schema": SCHEMA,
            "run_id": RUN_ID,
            "classification": "ENGINEERING_QUALIFICATION_ONLY",
            "git_head_at_start": _git_head(),
            "fixture": {
                "routed_lanes": ROUTED_LANES,
                "quiet_lanes": QUIET_LANES,
                "step_count": STEP_COUNT,
                "step_years_hex": STEP_YEARS.hex(),
                "expected_route_count": EXPECTED_ROUTE_COUNT,
                "expected_episode_count": EXPECTED_EPISODE_COUNT,
                "source_tile_sha256": tile.content_sha256,
            },
            "execution": {
                "physical_cpu_affinity": list(PHYSICAL_CPU_AFFINITY),
                "route_worker_count": ROUTE_WORKER_COUNT,
                "repeats": REPEATS,
                "numerical_thread_environment": NUMERICAL_THREAD_ENVIRONMENT,
                "modes": [
                    "reference_one_worker",
                    "reference_six_workers",
                    "episode_six_workers",
                ],
            },
            "acceptance": {
                "minimum_reference_six_worker_speedup": (
                    MINIMUM_REFERENCE_SIX_WORKER_SPEEDUP
                ),
                "minimum_episode_vs_reference_speedup": (
                    MINIMUM_EPISODE_VS_REFERENCE_SPEEDUP
                ),
                "maximum_sample_cv": MAXIMUM_SAMPLE_CV,
                "bitwise_state_and_route_equality_required": True,
                "all_claims_false_required": True,
            },
            "source_sha256": {
                label: _source_hash(path) for label, path in source_paths.items()
            },
            "claims": CLAIMS,
        },
        PROTOCOL_DOMAIN,
    )
    _write_json(args.output_root / "PROTOCOL.json", protocol)

    # Untimed compilation/process-start smoke.  Timed rows still include their
    # own fresh worker-pool startup because that is part of the public call.
    warm_state = replace(
        state,
        tracer_relative_positions_AU=cp.ascontiguousarray(
            state.tracer_relative_positions_AU[:2]
        ),
        tracer_relative_velocities_AU_per_year=cp.ascontiguousarray(
            state.tracer_relative_velocities_AU_per_year[:2]
        ),
        tracer_indices=cp.ascontiguousarray(state.tracer_indices[:2]),
    )
    warm_tile = FrozenSourcePathTile(
        **{
            **{
                field: getattr(tile, field)
                for field in tile.__dataclass_fields__
                if field != "content_sha256"
            },
            "tile_id": "synthetic-route-episode-warmup",
            "source_relative_positions_AU": np.ascontiguousarray(
                tile.source_relative_positions_AU[:2]
            ),
            "source_relative_velocities_AU_per_year": np.ascontiguousarray(
                tile.source_relative_velocities_AU_per_year[:2]
            ),
        }
    )
    warm_spec = FixedStepCUDARestrictedWHSpec(
        1, 1, BackendSpec("cupy", "cuda:0", 128)
    )
    _run(cp, advance_cuda_restricted_wh_hybrid, warm_state, warm_tile, warm_spec, 1)
    _run(cp, advance_cuda_restricted_wh_episodes, warm_state, warm_tile, warm_spec, 1)

    modes = {
        "reference_one_worker": (advance_cuda_restricted_wh_hybrid, 1),
        "reference_six_workers": (
            advance_cuda_restricted_wh_hybrid,
            ROUTE_WORKER_COUNT,
        ),
        "episode_six_workers": (
            advance_cuda_restricted_wh_episodes,
            ROUTE_WORKER_COUNT,
        ),
    }
    rows: dict[str, dict[str, Any]] = {
        label: {
            "samples_seconds": [],
            "state_sha256_samples": [],
            "route_ledger_sha256_samples": [],
            "runtime_identity_sha256_samples": [],
        }
        for label in modes
    }
    canonical_state_sha256: str | None = None
    canonical_route_sha256: str | None = None
    for _repeat in range(REPEATS):
        for label, (advance, workers) in modes.items():
            result, elapsed = _run(cp, advance, state, tile, spec, workers)
            state_sha256 = _state_digest(cp, result)
            route_sha256 = _route_digest(result.route_records)
            if canonical_state_sha256 is None:
                canonical_state_sha256 = state_sha256
                canonical_route_sha256 = route_sha256
            if (
                state_sha256 != canonical_state_sha256
                or route_sha256 != canonical_route_sha256
            ):
                raise RuntimeError("episode mode changed the exact reference result")
            if len(result.route_records) != EXPECTED_ROUTE_COUNT:
                raise RuntimeError("route count differs from the frozen fixture")
            if label == "episode_six_workers":
                if (
                    len(result.episodes) != EXPECTED_EPISODE_COUNT
                    or any(episode.route_count != STEP_COUNT for episode in result.episodes)
                ):
                    raise RuntimeError("episode grouping differs from the frozen fixture")
            runtime_identity = _plain(result.runtime_identity)
            runtime_identity_sha256 = hashlib.sha256(
                _canonical_bytes(runtime_identity)
            ).hexdigest()
            rows[label]["samples_seconds"].append(elapsed)
            rows[label]["state_sha256_samples"].append(state_sha256)
            rows[label]["route_ledger_sha256_samples"].append(route_sha256)
            rows[label]["runtime_identity_sha256_samples"].append(
                runtime_identity_sha256
            )
            rows[label]["state_sha256"] = state_sha256
            rows[label]["route_ledger_sha256"] = route_sha256
            rows[label]["route_count"] = len(result.route_records)
            route_runtime_identities = runtime_identity.pop(
                "route_runtime_identities", []
            )
            rows[label]["runtime_identity"] = runtime_identity
            rows[label]["route_runtime_identity_count"] = len(
                route_runtime_identities
            )
            rows[label]["route_runtime_identity_unique_sha256"] = sorted(
                {
                    hashlib.sha256(_canonical_bytes(identity)).hexdigest()
                    for identity in route_runtime_identities
                }
            )
            if label == "episode_six_workers":
                rows[label]["episode_count"] = len(result.episodes)
                rows[label]["longest_episode_route_count"] = max(
                    episode.route_count for episode in result.episodes
                )

    for row in rows.values():
        samples = row["samples_seconds"]
        row["median_seconds"] = statistics.median(samples)
        row["mad_seconds"] = statistics.median(
            [abs(value - row["median_seconds"]) for value in samples]
        )
        row["sample_cv"] = _cv(samples)

    serial = rows["reference_one_worker"]["median_seconds"]
    reference_six = rows["reference_six_workers"]["median_seconds"]
    episode_six = rows["episode_six_workers"]["median_seconds"]
    reference_speedup = serial / reference_six
    episode_vs_reference = reference_six / episode_six
    episode_vs_serial = serial / episode_six
    gates = {
        "all_claims_false": not any(CLAIMS.values()),
        "child_claim_ceilings_false": (
            not any(HYBRID_CLAIMS.values()) and not any(EPISODE_CLAIMS.values())
        ),
        "exact_route_count": all(
            row["route_count"] == EXPECTED_ROUTE_COUNT for row in rows.values()
        ),
        "bitwise_state_equal_all_modes_repeats": len(
            {
                digest
                for row in rows.values()
                for digest in row["state_sha256_samples"]
            }
        )
        == 1,
        "exact_route_ledger_equal_all_modes_repeats": len(
            {
                digest
                for row in rows.values()
                for digest in row["route_ledger_sha256_samples"]
            }
        )
        == 1,
        "runtime_identity_stable_within_each_mode": all(
            len(set(row["runtime_identity_sha256_samples"])) == 1
            for row in rows.values()
        ),
        "episode_grouping_exact": (
            rows["episode_six_workers"]["episode_count"]
            == EXPECTED_EPISODE_COUNT
            and rows["episode_six_workers"]["longest_episode_route_count"]
            == STEP_COUNT
        ),
        "six_physical_cpu_affinity_exact": tuple(
            sorted(os.sched_getaffinity(0))
        )
        == PHYSICAL_CPU_AFFINITY,
        "numerical_thread_environment_exact": all(
            os.environ.get(name) == value
            for name, value in NUMERICAL_THREAD_ENVIRONMENT.items()
        ),
        "reference_six_worker_speedup_passed": (
            reference_speedup >= MINIMUM_REFERENCE_SIX_WORKER_SPEEDUP
        ),
        "episode_speedup_over_reference_passed": (
            episode_vs_reference >= MINIMUM_EPISODE_VS_REFERENCE_SPEEDUP
        ),
        "sample_variation_bounded": all(
            row["sample_cv"] <= MAXIMUM_SAMPLE_CV for row in rows.values()
        ),
    }
    result = _content_record(
        {
            "schema": SCHEMA,
            "run_id": RUN_ID,
            "protocol_sha256": protocol["content_sha256"],
            "status": (
                "PASSED_ENGINEERING_QUALIFICATION_NOT_SCIENCE"
                if all(gates.values())
                else "FAILED_ENGINEERING_QUALIFICATION"
            ),
            "classification": "ENGINEERING_QUALIFICATION_ONLY",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "methods": {
                "reference": HYBRID_METHOD_ID,
                "episode": EPISODE_METHOD_ID,
            },
            "rows": rows,
            "reference_six_worker_speedup": reference_speedup,
            "episode_speedup_over_reference_six_workers": episode_vs_reference,
            "episode_speedup_over_reference_one_worker": episode_vs_serial,
            "gates": gates,
            "claims": CLAIMS,
            "limitations": [
                "synthetic primary-only route-heavy fixture",
                "episode means consecutive guarded macrosteps, not a physical encounter",
                "same-runtime deterministic comparison only",
                "in-memory execution only; no episode checkpoint contract",
                "no long-horizon accuracy or completion inference",
            ],
        },
        RESULT_DOMAIN,
    )
    _write_json(args.output_root / "RESULT.json", result)
    manifest_lines = []
    for path in sorted(args.output_root.iterdir(), key=lambda item: item.name):
        if path.name == "SHA256SUMS" or not path.is_file():
            continue
        manifest_lines.append(f"{_source_hash(path)}  {path.name}")
    (args.output_root / "SHA256SUMS").write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
