#!/usr/bin/env python3
"""Exact-parity and throughput screen for persistent episode ownership."""

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
from jxplanetx.cuda_restricted_wh_episode_owner import (
    OWNED_EPISODE_CLAIMS,
    OWNED_EPISODE_METHOD_ID,
    advance_cuda_restricted_wh_owned_episodes,
)
from jxplanetx.engine import BackendSpec, Provenance


SCHEMA = "jx.cuda-restricted-wh.owned-episode-qualification.v1"
PROTOCOL_DOMAIN = b"jx.cuda-restricted-wh.owned-episode.protocol.v1\0"
RESULT_DOMAIN = b"jx.cuda-restricted-wh.owned-episode.result.v1\0"
STATE_DOMAIN = b"jx.cuda-restricted-wh.owned-episode.state.v1\0"
ROUTE_DOMAIN = b"jx.cuda-restricted-wh.owned-episode.routes.v1\0"
ASSIGNMENT_DOMAIN = b"jx.cuda-restricted-wh.owned-episode.assignments.v1\0"
RUN_ID = "restricted-wh-owned-episode-qualification-v1"
ROUTED_LANES = 6
QUIET_LANES = 6
STEP_COUNT = 196
STEP_YEARS = 0.001
ROUTE_WORKER_COUNT = 6
ROUTE_LOOKAHEAD_STEPS = 64
PHYSICAL_CPU_AFFINITY = (0, 1, 2, 3, 4, 5)
REPEATS = 4
EXPECTED_ROUTE_COUNT = ROUTED_LANES * STEP_COUNT
EXPECTED_EPISODE_COUNT = ROUTED_LANES
EXPECTED_CHAIN_SUBMISSIONS = ROUTED_LANES * 5
EXPECTED_CACHE_HITS = ROUTED_LANES * (STEP_COUNT - 5)
MINIMUM_OWNED_EPISODE_SPEEDUP = 1.02
MAXIMUM_SAMPLE_CV = 0.05
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
    b"jx-restricted-wh-owned-episode-qualification-v1"
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
        citation="Synthetic deterministic persistent-owner fixture",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _tile() -> FrozenSourcePathTile:
    zeros = np.zeros((STEP_COUNT + 1, 1, 3), dtype=np.float64)
    return FrozenSourcePathTile(
        tile_id="synthetic-owned-route-episode",
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
        snapshot_id="synthetic-owned-route-episode",
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
        ("positions", cp.asnumpy(result.state.tracer_relative_positions_AU)),
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
            "runtime_identity": _plain(route.runtime_identity),
        }
        encoded = _canonical_bytes(metadata)
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        _array_digest(digest, "endpoint_position", route.endpoint_relative_position_AU)
        _array_digest(digest, "endpoint_velocity", route.endpoint_relative_velocity_AU_per_year)
    return digest.hexdigest()


def _assignment_digest(result: Any) -> str | None:
    if not hasattr(result, "route_worker_indices"):
        return None
    payload = [
        {
            "map_step_index": record.map_step_index,
            "lane_index": record.lane_index,
            "tracer_index": record.tracer_index,
            "worker_index": worker,
        }
        for record, worker in zip(
            result.route_records, result.route_worker_indices, strict=True
        )
    ]
    return hashlib.sha256(ASSIGNMENT_DOMAIN + _canonical_bytes(payload)).hexdigest()


def _persistent_ownership_exact(result: Any) -> bool | None:
    if not hasattr(result, "route_worker_indices"):
        return None
    worker_by_record = {
        (record.map_step_index, record.lane_index, record.tracer_index): worker
        for record, worker in zip(
            result.route_records, result.route_worker_indices, strict=True
        )
    }
    return all(
        len(
            {
                worker_by_record[
                    (record.map_step_index, record.lane_index, record.tracer_index)
                ]
                for record in episode.route_records
            }
        )
        == 1
        for episode in result.episodes
    )


def _run(
    cp: Any,
    advance: Callable[..., Any],
    state: CUDARestrictedWHState,
    tile: FrozenSourcePathTile,
    spec: FixedStepCUDARestrictedWHSpec,
    *,
    owned: bool,
) -> tuple[Any, float]:
    cp.cuda.get_current_stream().synchronize()
    started = time.perf_counter()
    kwargs = {"route_worker_count": ROUTE_WORKER_COUNT}
    if owned:
        kwargs["route_lookahead_steps"] = ROUTE_LOOKAHEAD_STEPS
    result = advance(state, tile, spec, **kwargs)
    cp.cuda.get_current_stream().synchronize()
    return result, time.perf_counter() - started


def _cv(samples: list[float]) -> float:
    mean = statistics.mean(samples)
    return statistics.stdev(samples) / mean if len(samples) > 1 else 0.0


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/restricted_wh_owned_episode_qualification_v1"),
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
        "owned_episode": root / "src/jxplanetx/cuda_restricted_wh_episode_owner.py",
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
                "expected_chain_submissions": EXPECTED_CHAIN_SUBMISSIONS,
                "expected_cache_hits": EXPECTED_CACHE_HITS,
                "source_tile_sha256": tile.content_sha256,
            },
            "execution": {
                "physical_cpu_affinity": list(PHYSICAL_CPU_AFFINITY),
                "route_worker_count": ROUTE_WORKER_COUNT,
                "route_lookahead_steps": ROUTE_LOOKAHEAD_STEPS,
                "repeats": REPEATS,
                "balanced_mode_order": True,
                "numerical_thread_environment": NUMERICAL_THREAD_ENVIRONMENT,
                "modes": ["episode_six_workers", "owned_episode_six_workers"],
            },
            "acceptance": {
                "minimum_owned_episode_speedup": MINIMUM_OWNED_EPISODE_SPEEDUP,
                "maximum_sample_cv": MAXIMUM_SAMPLE_CV,
                "bitwise_state_and_route_equality_required": True,
                "persistent_worker_ownership_required": True,
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
            "tile_id": "synthetic-owned-route-episode-warmup",
            "source_relative_positions_AU": np.ascontiguousarray(
                tile.source_relative_positions_AU[:3]
            ),
            "source_relative_velocities_AU_per_year": np.ascontiguousarray(
                tile.source_relative_velocities_AU_per_year[:3]
            ),
        }
    )
    warm_spec = FixedStepCUDARestrictedWHSpec(
        2, 1, BackendSpec("cupy", "cuda:0", 128)
    )
    _run(
        cp,
        advance_cuda_restricted_wh_episodes,
        warm_state,
        warm_tile,
        warm_spec,
        owned=False,
    )
    _run(
        cp,
        advance_cuda_restricted_wh_owned_episodes,
        warm_state,
        warm_tile,
        warm_spec,
        owned=True,
    )

    modes = {
        "episode_six_workers": (advance_cuda_restricted_wh_episodes, False),
        "owned_episode_six_workers": (
            advance_cuda_restricted_wh_owned_episodes,
            True,
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
    for repeat in range(REPEATS):
        order = list(modes)
        if repeat % 2:
            order.reverse()
        for label in order:
            advance, owned = modes[label]
            result, elapsed = _run(
                cp, advance, state, tile, spec, owned=owned
            )
            state_sha256 = _state_digest(cp, result)
            route_sha256 = _route_digest(result.route_records)
            if canonical_state_sha256 is None:
                canonical_state_sha256 = state_sha256
                canonical_route_sha256 = route_sha256
            if (
                state_sha256 != canonical_state_sha256
                or route_sha256 != canonical_route_sha256
            ):
                raise RuntimeError(
                    "persistent ownership changed the exact committed result"
                )
            if len(result.route_records) != EXPECTED_ROUTE_COUNT:
                raise RuntimeError("route count differs from the frozen fixture")
            runtime_identity = _plain(result.runtime_identity)
            runtime_identity_sha256 = hashlib.sha256(
                _canonical_bytes(runtime_identity)
            ).hexdigest()
            row = rows[label]
            row["samples_seconds"].append(elapsed)
            row["state_sha256_samples"].append(state_sha256)
            row["route_ledger_sha256_samples"].append(route_sha256)
            row["runtime_identity_sha256_samples"].append(
                runtime_identity_sha256
            )
            row["state_sha256"] = state_sha256
            row["route_ledger_sha256"] = route_sha256
            row["route_count"] = len(result.route_records)
            route_runtime_identities = runtime_identity.pop(
                "route_runtime_identities", []
            )
            row["runtime_identity"] = runtime_identity
            row["route_runtime_identity_count"] = len(route_runtime_identities)
            row["route_runtime_identity_unique_sha256"] = sorted(
                {
                    hashlib.sha256(_canonical_bytes(identity)).hexdigest()
                    for identity in route_runtime_identities
                }
            )
            row["episode_count"] = len(result.episodes)
            row["longest_episode_route_count"] = max(
                episode.route_count for episode in result.episodes
            )
            row["route_worker_assignment_evidence_sha256"] = _assignment_digest(
                result
            )
            row["persistent_owner_per_episode"] = _persistent_ownership_exact(
                result
            )
            row["route_worker_indices_used"] = (
                sorted(set(result.route_worker_indices))
                if hasattr(result, "route_worker_indices")
                else None
            )

    for row in rows.values():
        samples = row["samples_seconds"]
        row["median_seconds"] = statistics.median(samples)
        row["mad_seconds"] = statistics.median(
            [abs(value - row["median_seconds"]) for value in samples]
        )
        row["sample_cv"] = _cv(samples)

    reference_seconds = rows["episode_six_workers"]["median_seconds"]
    owned_seconds = rows["owned_episode_six_workers"]["median_seconds"]
    speedup = reference_seconds / owned_seconds
    owned_runtime = rows["owned_episode_six_workers"]["runtime_identity"]
    gates = {
        "all_claims_false": not any(CLAIMS.values()),
        "child_claim_ceilings_false": (
            not any(EPISODE_CLAIMS.values())
            and not any(OWNED_EPISODE_CLAIMS.values())
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
        "episode_grouping_exact": all(
            row["episode_count"] == EXPECTED_EPISODE_COUNT
            and row["longest_episode_route_count"] == STEP_COUNT
            for row in rows.values()
        ),
        "persistent_worker_ownership_exact": (
            rows["owned_episode_six_workers"][
                "route_worker_assignment_evidence_sha256"
            ]
            is not None
            and rows["owned_episode_six_workers"][
                "persistent_owner_per_episode"
            ]
            is True
            and rows["owned_episode_six_workers"]["route_worker_indices_used"]
            == list(range(ROUTE_WORKER_COUNT))
            and owned_runtime["new_episode_assignment_count"]
            == EXPECTED_EPISODE_COUNT
            and len(owned_runtime["route_worker_assignment_sha256"]) == 64
        ),
        "bounded_lookahead_accounting_exact": (
            owned_runtime["route_chain_submission_count"]
            == EXPECTED_CHAIN_SUBMISSIONS
            and owned_runtime["committed_cached_route_step_count"]
            == EXPECTED_CACHE_HITS
            and owned_runtime["speculative_route_step_count"]
            == EXPECTED_ROUTE_COUNT
            and owned_runtime["discarded_speculative_route_step_count"] == 0
        ),
        "six_physical_cpu_affinity_exact": tuple(
            sorted(os.sched_getaffinity(0))
        )
        == PHYSICAL_CPU_AFFINITY,
        "numerical_thread_environment_exact": all(
            os.environ.get(name) == value
            for name, value in NUMERICAL_THREAD_ENVIRONMENT.items()
        ),
        "owned_episode_speedup_passed": speedup >= MINIMUM_OWNED_EPISODE_SPEEDUP,
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
                "episode": EPISODE_METHOD_ID,
                "owned_episode": OWNED_EPISODE_METHOD_ID,
            },
            "rows": rows,
            "owned_episode_speedup_over_episode_scheduler": speedup,
            "gates": gates,
            "claims": CLAIMS,
            "limitations": [
                "synthetic primary-only route-heavy fixture",
                "episode means consecutive guarded macrosteps, not a physical encounter",
                "prospective routes commit only after the exact CUDA guard and start-state match",
                "same-runtime deterministic comparison only",
                "in-memory execution only; no owned-episode checkpoint contract",
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
