#!/usr/bin/env python3
"""Exact real-input benchmark for owned-episode checkpoint appends.

This is a bounded engineering benchmark, not a science run.  It compares the
standalone full-predecessor-replay append API with the locked append-session
API on the tracked DE441-derived control source state and the complete frozen
16,800-tracer synthetic population.  Both paths must publish byte-identical
checkpoint records and finish at the same replay-verified state.
"""

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
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import cupy as cp
import numpy as np

import jx_cuda_restricted_wh_refinement as frozen_inputs
from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
)
from jxplanetx.cuda_restricted_wh_episode_owner import (
    advance_cuda_restricted_wh_owned_episodes,
)
from jxplanetx.cuda_restricted_wh_hybrid import _slice_tile
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.restricted_wh_episode_checkpoint import (
    CHECKPOINT_CLAIMS,
    dump_owned_episode_delta_checkpoint,
    load_latest_owned_episode_delta_checkpoint,
    open_owned_episode_delta_append_session,
    owned_episode_checkpoint_implementation_identity,
    owned_episode_checkpoint_runtime_identity,
    owned_episode_state_sha256,
)
import jxplanetx.restricted_wh_episode_checkpoint as checkpoint_module
from jxplanetx.restricted_wh_route import (
    RestrictedWHRouteSpec,
    restricted_wh_route_policy_identity,
)
from jxplanetx.restricted_wh_source import (
    WisdomHolmanSourceTileSpec,
    build_wisdom_holman_source_tile,
)


SCHEMA = "jx.restricted-wh.real-episode-checkpoint-benchmark.v2"
PROTOCOL_DOMAIN = b"jx.restricted-wh.real-episode.protocol.v2\0"
RESULT_DOMAIN = b"jx.restricted-wh.real-episode.result.v2\0"
RUN_ID = "restricted-wh-real-episode-checkpoint-benchmark-v2"
PHYSICAL_CPU_AFFINITY = (0, 1, 2, 3, 4, 5)
ROUTE_WORKER_COUNT = 6
ROUTE_LOOKAHEAD_STEPS = 64
SOURCE_STEP_YEARS = 0.0625
SEGMENT_STEP_COUNT = 1_000
SEGMENT_COUNT = 4
TOTAL_STEP_COUNT = SEGMENT_STEP_COUNT * SEGMENT_COUNT
EXPECTED_TRACER_COUNT = 16_800
EXPECTED_ROUTE_DELTAS = (0, 20, 82, 0)
EXPECTED_ROUTE_TOTAL = 102
EXPECTED_EPISODE_TOTAL = 2
EXPECTED_FINAL_STATE_SHA256 = (
    "5e37562402f5053b4303fb809c89714046f544ab773b12d4e332d247ce705469"
)
MINIMUM_FINAL_QUIET_APPEND_SPEEDUP = 5.0
MINIMUM_TOTAL_APPEND_SPEEDUP = 1.20
DEFAULT_OUTPUT_ROOT = Path(
    "runs/restricted_wh_real_episode_checkpoint_benchmark_v2"
)
COLLISION_STATE_PATH = Path(
    "runs/cuda_four_arm_screen_v1/qualification/arms/control/INITIAL_STATE.npz"
)
SOURCE_PATHS = {
    "all_conic": "src/jxplanetx/_all_conic_kepler.py",
    "restricted_wh": "src/jxplanetx/cuda_restricted_wh.py",
    "hybrid": "src/jxplanetx/cuda_restricted_wh_hybrid.py",
    "episode": "src/jxplanetx/cuda_restricted_wh_episode.py",
    "owned_episode": "src/jxplanetx/cuda_restricted_wh_episode_owner.py",
    "route": "src/jxplanetx/restricted_wh_route.py",
    "event_locator": "src/jxplanetx/cuda_tracer_events.py",
    "checkpoint_codec": "src/jxplanetx/restricted_wh_checkpoint.py",
    "delta_checkpoint": (
        "src/jxplanetx/restricted_wh_episode_checkpoint.py"
    ),
    "source_provider": "src/jxplanetx/restricted_wh_source.py",
    "frozen_input_adapter": (
        "benchmarks/jx_cuda_restricted_wh_refinement.py"
    ),
    "benchmark": (
        "benchmarks/jx_restricted_wh_real_episode_checkpoint_benchmark.py"
    ),
}
CLAIMS = {
    "production_qualified": False,
    "science_result": False,
    "physical_encounter_rate_represented": False,
    "long_horizon_accuracy_qualified": False,
    "long_horizon_completion_projected": False,
    "cross_runtime_bitwise_reproducible": False,
}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _content_record(value: Mapping[str, Any], domain: bytes) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        domain + _canonical_bytes(result)
    ).hexdigest()
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        label: _sha256_file(Path(path)) for label, path in SOURCE_PATHS.items()
    }


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _state(
    *,
    tracer_position: np.ndarray,
    tracer_velocity: np.ndarray,
    tracer_indices: np.ndarray,
    metadata: Mapping[str, Any],
    schedule_sha256: str,
    provenance: Provenance,
) -> CUDARestrictedWHState:
    return CUDARestrictedWHState(
        snapshot_id="de441-control-real-episode-checkpoint-v1",
        epoch_origin_years=0.0,
        global_source_step=0,
        source_step_years=SOURCE_STEP_YEARS,
        time_scale=metadata["time_scale"],
        frame=metadata["frame"],
        axes="J2000_ECLIPTIC",
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id=metadata["unit_system_id"],
        source_ids=tuple(metadata["source_ids"]),
        primary_source_id=metadata["primary_source_id"],
        source_schedule_sha256=schedule_sha256,
        tracer_relative_positions_AU=cp.asarray(tracer_position),
        tracer_relative_velocities_AU_per_year=cp.asarray(tracer_velocity),
        tracer_indices=cp.asarray(tracer_indices),
        provenance=provenance,
    )


def _checkpoint_size(root: Path, protocol_sha256: str, index: int) -> int:
    namespace = checkpoint_module._checkpoint_namespace(
        root, RUN_ID, protocol_sha256
    )
    return (namespace / f"checkpoint_{index:020d}.json").stat().st_size


def _run_mode(
    *,
    mode: str,
    output_root: Path,
    protocol_sha256: str,
    genesis_state_sha256: str,
    initial_state: CUDARestrictedWHState,
    source_tile: Any,
    backend: BackendSpec,
    route_spec: RestrictedWHRouteSpec,
) -> dict[str, Any]:
    root = output_root / mode
    root.mkdir(exist_ok=False)
    current = initial_state
    rows: list[dict[str, Any]] = []
    predecessor: str | None = None
    session = None
    if mode == "locked_session":
        session = open_owned_episode_delta_append_session(
            root,
            run_id=RUN_ID,
            protocol_sha256=protocol_sha256,
            genesis_state_sha256=genesis_state_sha256,
            backend=backend,
        )
    try:
        for index in range(SEGMENT_COUNT):
            tile = _slice_tile(
                source_tile, index * SEGMENT_STEP_COUNT, SEGMENT_STEP_COUNT
            )
            spec = FixedStepCUDARestrictedWHSpec(
                SEGMENT_STEP_COUNT, 1, backend
            )
            started = time.perf_counter()
            result = advance_cuda_restricted_wh_owned_episodes(
                current,
                tile,
                spec,
                route_spec=route_spec,
                route_worker_count=ROUTE_WORKER_COUNT,
                route_lookahead_steps=ROUTE_LOOKAHEAD_STEPS,
            )
            advance_seconds = time.perf_counter() - started
            started = time.perf_counter()
            if session is None:
                record = dump_owned_episode_delta_checkpoint(
                    root,
                    run_id=RUN_ID,
                    protocol_sha256=protocol_sha256,
                    genesis_state_sha256=genesis_state_sha256,
                    checkpoint_index=index,
                    input_state=current,
                    result=result,
                    source_tile=tile,
                    spec=spec,
                    route_spec=route_spec,
                    route_worker_count=ROUTE_WORKER_COUNT,
                    route_lookahead_steps=ROUTE_LOOKAHEAD_STEPS,
                    previous_checkpoint_content_sha256=predecessor,
                )
                current = result.state
            else:
                head = session.append(
                    input_state=current,
                    result=result,
                    source_tile=tile,
                    spec=spec,
                    route_spec=route_spec,
                    route_worker_count=ROUTE_WORKER_COUNT,
                    route_lookahead_steps=ROUTE_LOOKAHEAD_STEPS,
                )
                record = head.record
                current = head.state
            append_seconds = time.perf_counter() - started
            predecessor = record["checkpoint_content_sha256"]
            rows.append(
                {
                    "checkpoint_index": index,
                    "advance_seconds": advance_seconds,
                    "append_seconds": append_seconds,
                    "checkpoint_content_sha256": predecessor,
                    "state_sha256": record["state_sha256"],
                    "route_delta": record["route_record_count_delta"],
                    "route_cumulative": record[
                        "route_record_count_cumulative"
                    ],
                    "episode_delta": record[
                        "segment_episode_count_delta"
                    ],
                    "json_size_bytes": _checkpoint_size(
                        root, protocol_sha256, index
                    ),
                }
            )
    finally:
        if session is not None:
            session.close()
    assert predecessor is not None
    started = time.perf_counter()
    loaded = load_latest_owned_episode_delta_checkpoint(
        root,
        run_id=RUN_ID,
        protocol_sha256=protocol_sha256,
        genesis_state_sha256=genesis_state_sha256,
        expected_checkpoint_index=SEGMENT_COUNT - 1,
        expected_checkpoint_content_sha256=predecessor,
        backend=backend,
    )
    load_seconds = time.perf_counter() - started
    return {
        "mode": mode,
        "segments": rows,
        "total_advance_seconds": sum(row["advance_seconds"] for row in rows),
        "total_append_seconds": sum(row["append_seconds"] for row in rows),
        "full_chain_load_seconds": load_seconds,
        "final_checkpoint_content_sha256": predecessor,
        "final_state_sha256": loaded.state_sha256,
        "final_route_chain_sha256": loaded.route_chain_sha256,
        "route_record_count_cumulative": (
            loaded.route_record_count_cumulative
        ),
        "segment_episode_count_cumulative": (
            loaded.segment_episode_count_cumulative
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    arguments = parser.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(PHYSICAL_CPU_AFFINITY))
    output_root = arguments.output_root
    output_root.mkdir(parents=True, exist_ok=False)

    repository_root = Path.cwd()
    start_source_hashes = _source_hashes()
    start_git_head = _git_head()
    backend = BackendSpec("cupy", "cuda:0", 128)
    runtime_identity = _plain(
        owned_episode_checkpoint_runtime_identity(backend)
    )
    implementation_identity = _plain(
        owned_episode_checkpoint_implementation_identity()
    )
    (
        tracer_position,
        tracer_velocity,
        tracer_indices,
        source_position,
        source_velocity,
        source_gm,
        source_radii,
        acceleration_bounds,
        metadata,
        schedule_sha256,
        input_hashes,
    ) = frozen_inputs._load_inputs(repository_root)
    if tracer_indices.size != EXPECTED_TRACER_COUNT:
        raise RuntimeError("tracked tracer population count changed")
    collision_state = repository_root / COLLISION_STATE_PATH
    with np.load(collision_state, allow_pickle=False) as archive:
        terminal_radii = np.ascontiguousarray(
            archive["source_collision_radii"], dtype=np.float64
        )
    provenance = Provenance(
        source_id="jx.real-episode-checkpoint-benchmark",
        citation="Tracked DE441-derived control and frozen synthetic population",
        version="v1",
        sha256=input_hashes["SOURCE_STATE.npz"],
    )
    source = build_wisdom_holman_source_tile(
        tile_id="de441-control-real-episode-checkpoint-v1",
        epoch_origin_years=0.0,
        global_source_step=0,
        time_scale=metadata["time_scale"],
        frame=metadata["frame"],
        axes="J2000_ECLIPTIC",
        source_ids=tuple(metadata["source_ids"]),
        primary_source_id=metadata["primary_source_id"],
        source_schedule_sha256=schedule_sha256,
        initial_source_relative_positions_AU=source_position,
        initial_source_relative_velocities_AU_per_year=source_velocity,
        source_gm_AU3_per_year2=source_gm,
        source_encounter_radii_AU=source_radii,
        certified_relative_acceleration_bounds_AU_per_year2=(
            acceleration_bounds
        ),
        provenance=provenance,
        spec=WisdomHolmanSourceTileSpec(
            SOURCE_STEP_YEARS, TOTAL_STEP_COUNT
        ),
    )
    route_spec = RestrictedWHRouteSpec(
        terminal_source_radii_AU=tuple(float(value) for value in terminal_radii)
    )
    route_policy = _plain(restricted_wh_route_policy_identity(route_spec))
    initial = _state(
        tracer_position=tracer_position,
        tracer_velocity=tracer_velocity,
        tracer_indices=tracer_indices,
        metadata=metadata,
        schedule_sha256=schedule_sha256,
        provenance=provenance,
    )
    genesis = owned_episode_state_sha256(initial, backend)
    protocol = _content_record(
        {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head": start_git_head,
            "source_hashes": start_source_hashes,
            "implementation_identity": implementation_identity,
            "runtime_identity": runtime_identity,
            "numerical_thread_environment": NUMERICAL_THREAD_ENVIRONMENT,
            "physical_cpu_affinity": list(PHYSICAL_CPU_AFFINITY),
            "route_worker_count": ROUTE_WORKER_COUNT,
            "route_lookahead_steps": ROUTE_LOOKAHEAD_STEPS,
            "source_step_years_hex": SOURCE_STEP_YEARS.hex(),
            "segment_step_count": SEGMENT_STEP_COUNT,
            "segment_count": SEGMENT_COUNT,
            "total_step_count": TOTAL_STEP_COUNT,
            "tracer_count": int(tracer_indices.size),
            "source_count": int(source_gm.size),
            "input_hashes": {
                **input_hashes,
                str(COLLISION_STATE_PATH): _sha256_file(collision_state),
            },
            "source_schedule_sha256": schedule_sha256,
            "source_tile_sha256": source.tile.content_sha256,
            "genesis_state_sha256": genesis,
            "route_policy": route_policy,
            "expected_route_deltas": list(EXPECTED_ROUTE_DELTAS),
            "expected_route_total": EXPECTED_ROUTE_TOTAL,
            "expected_episode_total": EXPECTED_EPISODE_TOTAL,
            "expected_final_state_sha256": EXPECTED_FINAL_STATE_SHA256,
            "minimum_final_quiet_append_speedup": (
                MINIMUM_FINAL_QUIET_APPEND_SPEEDUP
            ),
            "minimum_total_append_speedup": MINIMUM_TOTAL_APPEND_SPEEDUP,
            "checkpoint_claims": dict(CHECKPOINT_CLAIMS),
            "claims": CLAIMS,
        },
        PROTOCOL_DOMAIN,
    )
    _write_json(output_root / "PROTOCOL.json", protocol)

    modes = []
    for mode in ("standalone_full_replay", "locked_session"):
        mode_state = _state(
            tracer_position=tracer_position,
            tracer_velocity=tracer_velocity,
            tracer_indices=tracer_indices,
            metadata=metadata,
            schedule_sha256=schedule_sha256,
            provenance=provenance,
        )
        modes.append(
            _run_mode(
                mode=mode,
                output_root=output_root,
                protocol_sha256=protocol["content_sha256"],
                genesis_state_sha256=genesis,
                initial_state=mode_state,
                source_tile=source.tile,
                backend=backend,
                route_spec=route_spec,
            )
        )
    standalone, session = modes
    checkpoint_records_exact = [
        row["checkpoint_content_sha256"]
        for row in standalone["segments"]
    ] == [
        row["checkpoint_content_sha256"] for row in session["segments"]
    ]
    route_deltas = tuple(row["route_delta"] for row in session["segments"])
    final_quiet_speedup = (
        standalone["segments"][-1]["append_seconds"]
        / session["segments"][-1]["append_seconds"]
    )
    total_append_speedup = (
        standalone["total_append_seconds"] / session["total_append_seconds"]
    )
    end_source_hashes = _source_hashes()
    end_git_head = _git_head()
    end_runtime_identity = _plain(
        owned_episode_checkpoint_runtime_identity(backend)
    )
    gates = {
        "source_bytes_unchanged": end_source_hashes == start_source_hashes,
        "git_head_unchanged": end_git_head == start_git_head,
        "runtime_identity_unchanged": (
            end_runtime_identity == runtime_identity
        ),
        "checkpoint_records_exact": checkpoint_records_exact,
        "final_state_exact_between_modes": (
            standalone["final_state_sha256"]
            == session["final_state_sha256"]
        ),
        "expected_final_state_exact": (
            session["final_state_sha256"] == EXPECTED_FINAL_STATE_SHA256
        ),
        "route_chain_exact": (
            standalone["final_route_chain_sha256"]
            == session["final_route_chain_sha256"]
        ),
        "route_profile_exact": route_deltas == EXPECTED_ROUTE_DELTAS,
        "route_total_exact": (
            session["route_record_count_cumulative"] == EXPECTED_ROUTE_TOTAL
        ),
        "episode_total_exact": (
            session["segment_episode_count_cumulative"]
            == EXPECTED_EPISODE_TOTAL
        ),
        "final_quiet_append_speedup_passed": (
            final_quiet_speedup >= MINIMUM_FINAL_QUIET_APPEND_SPEEDUP
        ),
        "total_append_speedup_passed": (
            total_append_speedup >= MINIMUM_TOTAL_APPEND_SPEEDUP
        ),
        "claim_ceiling_all_false": not any(CLAIMS.values()),
    }
    passed = all(gates.values())
    result = _content_record(
        {
            "schema": SCHEMA,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "status": (
                "PASSED_ENGINEERING_BENCHMARK_NOT_SCIENCE"
                if passed
                else "FAILED_ENGINEERING_BENCHMARK"
            ),
            "protocol_sha256": protocol["content_sha256"],
            "modes": modes,
            "final_quiet_append_speedup": final_quiet_speedup,
            "total_append_speedup": total_append_speedup,
            "gates": gates,
            "claims": CLAIMS,
            "notes": [
                "The complete tracer population is a synthetic engineering fixture.",
                "Route episodes are consecutive guarded map steps, not inferred physical encounters.",
                "Timing is same-machine monotonic wall time and is not a completion projection.",
                "Every public final load still replays the complete chain from genesis.",
            ],
        },
        RESULT_DOMAIN,
    )
    _write_json(output_root / "RESULT.json", result)
    artifact_paths = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_text = "".join(
        f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}\n"
        for path in artifact_paths
    )
    checksum_path = output_root / "SHA256SUMS"
    with checksum_path.open("x", encoding="ascii") as stream:
        stream.write(checksum_text)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
