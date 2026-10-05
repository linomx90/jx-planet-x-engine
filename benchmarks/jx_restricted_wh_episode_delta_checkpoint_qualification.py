#!/usr/bin/env python3
"""Retained exact-replay screen for owned-episode delta checkpoints."""

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
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from jxplanetx.cuda_restricted_wh import (
    COORDINATE_SYSTEM,
    CUDARestrictedWHState,
    FixedStepCUDARestrictedWHSpec,
    FrozenSourcePathTile,
)
from jxplanetx.cuda_restricted_wh_episode_owner import (
    OWNED_EPISODE_CLAIMS,
    OWNED_EPISODE_METHOD_ID,
    advance_cuda_restricted_wh_owned_episodes,
)
from jxplanetx.engine import BackendSpec, Provenance
from jxplanetx.restricted_wh_episode_checkpoint import (
    CHECKPOINT_CLAIMS,
    CHECKPOINT_METHOD_ID,
    dump_owned_episode_delta_checkpoint,
    load_latest_owned_episode_delta_checkpoint,
    owned_episode_checkpoint_implementation_identity,
    owned_episode_state_sha256,
)
import jxplanetx.restricted_wh_episode_checkpoint as checkpoint_module


SCHEMA = "jx.restricted-wh.owned-episode-delta-qualification.v1"
PROTOCOL_DOMAIN = b"jx.restricted-wh.owned-episode-delta.protocol.v1\0"
RESULT_DOMAIN = b"jx.restricted-wh.owned-episode-delta.result.v1\0"
RESULT_EVIDENCE_DOMAIN = b"jx.restricted-wh.owned-episode-delta.evidence.v1\0"
RUN_ID = "restricted-wh-owned-episode-delta-qualification-v1"
PHYSICAL_CPU_AFFINITY = (0, 1, 2, 3, 4, 5)
ROUTE_WORKER_COUNT = 6
ROUTE_LOOKAHEAD_STEPS = 16
LANE_COUNT = 6
SEGMENT_STEP_COUNT = 16
SEGMENT_COUNT = 2
STEP_YEARS = 0.001
EXPECTED_ROUTE_DELTA = LANE_COUNT * SEGMENT_STEP_COUNT
EXPECTED_ROUTE_TOTAL = EXPECTED_ROUTE_DELTA * SEGMENT_COUNT
MAXIMUM_JSON_SIZE_RATIO = 1.20
SCHEDULE_SHA256 = hashlib.sha256(
    b"jx-owned-episode-delta-qualification-source-schedule-v1"
).hexdigest()
CLAIMS = {
    "production_qualified": False,
    "science_result": False,
    "physical_encounter_rate_represented": False,
    "long_horizon_accuracy_qualified": False,
    "long_horizon_completion_projected": False,
    "checkpoint_boundary_owner_persistence_qualified": False,
    "cross_runtime_bitwise_reproducible": False,
}
SOURCE_PATHS = {
    "all_conic": "src/jxplanetx/_all_conic_kepler.py",
    "restricted_wh": "src/jxplanetx/cuda_restricted_wh.py",
    "hybrid": "src/jxplanetx/cuda_restricted_wh_hybrid.py",
    "episode": "src/jxplanetx/cuda_restricted_wh_episode.py",
    "owned_episode": "src/jxplanetx/cuda_restricted_wh_episode_owner.py",
    "route": "src/jxplanetx/restricted_wh_route.py",
    "event_locator": "src/jxplanetx/cuda_tracer_events.py",
    "lifecycle": "src/jxplanetx/restricted_wh_lifecycle.py",
    "checkpoint_codec": "src/jxplanetx/restricted_wh_checkpoint.py",
    "delta_checkpoint": "src/jxplanetx/restricted_wh_episode_checkpoint.py",
    "engine_backends": "src/jxplanetx/engine/backends.py",
    "engine_contracts": "src/jxplanetx/engine/contracts.py",
    "wisdom_holman_contracts": (
        "src/jxplanetx/engine/wisdom_holman_contracts.py"
    ),
    "benchmark": (
        "benchmarks/jx_restricted_wh_episode_delta_checkpoint_qualification.py"
    ),
}


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hash(path: str) -> str:
    return _sha256_file(Path(path))


def _current_source_hashes() -> dict[str, str]:
    return {label: _source_hash(path) for label, path in SOURCE_PATHS.items()}


def _in_memory_source_hashes() -> dict[str, str]:
    identity = _plain(owned_episode_checkpoint_implementation_identity())
    codec = identity["checkpoint_codec_dependency_identity"]
    return {
        "all_conic": codec["all_conic_implementation_source_sha256"],
        "restricted_wh": identity[
            "restricted_wh_implementation_source_sha256"
        ],
        "hybrid": codec["hybrid_implementation_source_sha256"],
        "episode": identity["episode_implementation_source_sha256"],
        "owned_episode": identity[
            "owned_episode_implementation_source_sha256"
        ],
        "route": codec["route_implementation_source_sha256"],
        "event_locator": codec["event_locator_source_sha256"],
        "lifecycle": codec["lifecycle_implementation_source_sha256"],
        "checkpoint_codec": identity[
            "checkpoint_codec_implementation_source_sha256"
        ],
        "delta_checkpoint": identity[
            "checkpoint_implementation_source_sha256"
        ],
        "engine_backends": codec["backend_implementation_source_sha256"],
        "engine_contracts": codec["contracts_implementation_source_sha256"],
        "wisdom_holman_contracts": codec[
            "wisdom_holman_contracts_source_sha256"
        ],
    }


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, record: Mapping[str, Any]) -> None:
    payload = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=True) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _provenance(label: str) -> Provenance:
    return Provenance(
        source_id=f"jx-qualification.{label}",
        citation="Synthetic deterministic owned-episode delta fixture",
        version="v1",
        sha256=hashlib.sha256(label.encode("ascii")).hexdigest(),
    )


def _backend() -> BackendSpec:
    return BackendSpec("cupy", "cuda:0", 128)


def _state(cp: Any) -> CUDARestrictedWHState:
    positions = cp.zeros((LANE_COUNT, 3), dtype=cp.float64)
    velocities = cp.zeros_like(positions)
    positions[:, 0] = 1.0
    velocities[:, 1] = cp.sqrt(cp.float64(4.0 * math.pi * math.pi))
    return CUDARestrictedWHState(
        snapshot_id="synthetic-owned-episode-delta",
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
        tracer_indices=cp.arange(LANE_COUNT, dtype=cp.uint64),
        provenance=_provenance("tracer-state"),
    )


def _tile(start: int) -> FrozenSourcePathTile:
    positions = np.zeros((SEGMENT_STEP_COUNT + 1, 1, 3), dtype=np.float64)
    return FrozenSourcePathTile(
        tile_id=f"synthetic-delta-{start}",
        epoch_origin_years=0.0,
        global_source_step=start,
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
        source_relative_positions_AU=positions,
        source_relative_velocities_AU_per_year=positions.copy(),
        source_gm_AU3_per_year2=np.asarray(
            (4.0 * math.pi * math.pi,), dtype=np.float64
        ),
        source_encounter_radii_AU=np.asarray((0.997,), dtype=np.float64),
        certified_relative_acceleration_bounds_AU_per_year2=np.asarray(
            (100.0,), dtype=np.float64
        ),
        provenance=_provenance("source-tile"),
    )


def _spec() -> FixedStepCUDARestrictedWHSpec:
    return FixedStepCUDARestrictedWHSpec(
        map_step_count=SEGMENT_STEP_COUNT,
        source_steps_per_map_step=1,
        backend=_backend(),
    )


def _advance(state, tile, spec):
    return advance_cuda_restricted_wh_owned_episodes(
        state,
        tile,
        spec,
        route_worker_count=ROUTE_WORKER_COUNT,
        route_lookahead_steps=ROUTE_LOOKAHEAD_STEPS,
    )


def _result_evidence_sha256(result: Any, cp: Any) -> str:
    semantics = checkpoint_module._result_semantics(
        result, cp, segment_map_step_count=SEGMENT_STEP_COUNT
    )
    return hashlib.sha256(
        RESULT_EVIDENCE_DOMAIN
        + checkpoint_module._checkpoint_codec._canonical_json(semantics)
    ).hexdigest()


def _artifact_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name in {"RESULT.json", "SHA256SUMS"}:
            continue
        relative = path.relative_to(root).as_posix()
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return records


def _state_arrays_equal(cp: Any, left: CUDARestrictedWHState, right: CUDARestrictedWHState) -> bool:
    return (
        left.global_source_step == right.global_source_step
        and bool(
            cp.array_equal(
                left.tracer_relative_positions_AU,
                right.tracer_relative_positions_AU,
            ).item()
        )
        and bool(
            cp.array_equal(
                left.tracer_relative_velocities_AU_per_year,
                right.tracer_relative_velocities_AU_per_year,
            ).item()
        )
        and bool(cp.array_equal(left.tracer_indices, right.tracer_indices).item())
    )


def _protocol() -> dict[str, Any]:
    return _content_record(
        {
            "schema": SCHEMA,
            "run_id": RUN_ID,
            "classification": "ENGINEERING_QUALIFICATION_ONLY",
            "git_head_at_start": _git_head(),
            "source_sha256": _current_source_hashes(),
            "fixture": {
                "lane_count": LANE_COUNT,
                "segment_count": SEGMENT_COUNT,
                "segment_step_count": SEGMENT_STEP_COUNT,
                "source_step_years_hex": float(STEP_YEARS).hex(),
                "expected_route_delta_per_segment": EXPECTED_ROUTE_DELTA,
                "expected_route_total": EXPECTED_ROUTE_TOTAL,
                "source_schedule_sha256": SCHEDULE_SHA256,
            },
            "execution": {
                "route_worker_count": ROUTE_WORKER_COUNT,
                "route_lookahead_steps": ROUTE_LOOKAHEAD_STEPS,
                "physical_cpu_affinity": list(PHYSICAL_CPU_AFFINITY),
                "numerical_thread_environment": NUMERICAL_THREAD_ENVIRONMENT,
            },
            "required_gates": {
                "exact_split_restart_state": True,
                "exact_segment_result_evidence": True,
                "delta_only_route_storage": True,
                "route_chain_and_counters": True,
                "exact_six_worker_ownership": True,
                "full_chain_replay": True,
                "executed_source_identity_exact_and_stable": True,
                "all_claims_false": True,
            },
            "claims": CLAIMS,
        },
        PROTOCOL_DOMAIN,
    )


def run(output_root: Path) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to reuse evidence root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("qualification requires Linux CPU affinity controls")
    available = set(os.sched_getaffinity(0))
    required = set(PHYSICAL_CPU_AFFINITY)
    if not required.issubset(available):
        raise RuntimeError("required physical CPU affinity is unavailable")
    os.sched_setaffinity(0, required)

    protocol = _protocol()
    _write_json(output_root / "PROTOCOL.json", protocol)
    start_git_head = protocol["git_head_at_start"]
    start_source_hashes = protocol["source_sha256"]
    in_memory_source_hashes = _in_memory_source_hashes()
    if {
        key: start_source_hashes[key] for key in in_memory_source_hashes
    } != in_memory_source_hashes:
        raise RuntimeError(
            "imported numerical/checkpoint modules do not match protocol source bytes"
        )

    backend = _backend()
    resolved = checkpoint_module.resolve_backend(backend)
    cp = resolved.xp
    checkpoint_root = output_root / "checkpoints"
    with resolved.activate():
        initial = _state(cp)
        genesis = owned_episode_state_sha256(initial, backend)
        spec0 = _spec()
        tile0 = _tile(0)
        first = _advance(initial, tile0, spec0)
        row0 = dump_owned_episode_delta_checkpoint(
            checkpoint_root,
            run_id=RUN_ID,
            protocol_sha256=protocol["content_sha256"],
            genesis_state_sha256=genesis,
            checkpoint_index=0,
            input_state=initial,
            result=first,
            source_tile=tile0,
            spec=spec0,
            route_worker_count=ROUTE_WORKER_COUNT,
            route_lookahead_steps=ROUTE_LOOKAHEAD_STEPS,
            previous_checkpoint_content_sha256=None,
        )
        loaded0 = load_latest_owned_episode_delta_checkpoint(
            checkpoint_root,
            run_id=RUN_ID,
            protocol_sha256=protocol["content_sha256"],
            genesis_state_sha256=genesis,
            expected_checkpoint_index=0,
            expected_checkpoint_content_sha256=row0[
                "checkpoint_content_sha256"
            ],
            backend=backend,
        )

        spec1 = _spec()
        tile1 = _tile(SEGMENT_STEP_COUNT)
        direct_second = _advance(first.state, tile1, spec1)
        restart_second = _advance(loaded0.state, tile1, spec1)
        row1 = dump_owned_episode_delta_checkpoint(
            checkpoint_root,
            run_id=RUN_ID,
            protocol_sha256=protocol["content_sha256"],
            genesis_state_sha256=genesis,
            checkpoint_index=1,
            input_state=loaded0.state,
            result=restart_second,
            source_tile=tile1,
            spec=spec1,
            route_worker_count=ROUTE_WORKER_COUNT,
            route_lookahead_steps=ROUTE_LOOKAHEAD_STEPS,
            previous_checkpoint_content_sha256=row0[
                "checkpoint_content_sha256"
            ],
        )
        loaded1 = load_latest_owned_episode_delta_checkpoint(
            checkpoint_root,
            run_id=RUN_ID,
            protocol_sha256=protocol["content_sha256"],
            genesis_state_sha256=genesis,
            expected_checkpoint_index=1,
            expected_checkpoint_content_sha256=row1[
                "checkpoint_content_sha256"
            ],
            backend=backend,
        )
        cp.cuda.get_current_stream().synchronize()

        direct_state_sha = owned_episode_state_sha256(direct_second.state, backend)
        restart_state_sha = owned_episode_state_sha256(
            restart_second.state, backend
        )
        loaded_state_sha = owned_episode_state_sha256(loaded1.state, backend)
        direct_evidence_sha = _result_evidence_sha256(direct_second, cp)
        restart_evidence_sha = _result_evidence_sha256(restart_second, cp)

    json_paths = [
        path
        for path in checkpoint_root.rglob("checkpoint_*.json")
        if path.is_file()
    ]
    json_paths.sort()
    json_sizes = [path.stat().st_size for path in json_paths]
    json_size_ratio = max(json_sizes) / min(json_sizes)
    rows = [_plain(row0), _plain(row1)]
    route_deltas = [row["route_record_count_delta"] for row in rows]
    route_lists = [len(row["result"]["route_records"]) for row in rows]
    worker_sets = [
        sorted(set(row["result"]["route_worker_indices"])) for row in rows
    ]
    owner_per_episode = all(
        len(
            {
                worker
                for route, worker in zip(
                    row["result"]["route_records"],
                    row["result"]["route_worker_indices"],
                    strict=True,
                )
                if route["tracer_index"] == summary["tracer_index"]
                and summary["first_map_step_index"]
                <= route["map_step_index"]
                < summary["stop_map_step_index"]
            }
        )
        == 1
        for row in rows
        for summary in row["result"]["segment_episode_summaries"]
    )
    artifacts = _artifact_records(output_root)
    artifact_paths = [record["path"] for record in artifacts]
    completion_source_hashes = _current_source_hashes()
    completion_git_head = _git_head()
    gates = {
        "all_claims_false": (
            not any(CLAIMS.values())
            and not any(CHECKPOINT_CLAIMS.values())
            and not any(OWNED_EPISODE_CLAIMS.values())
        ),
        "exact_split_restart_state": (
            direct_state_sha == restart_state_sha == loaded_state_sha
            and _state_arrays_equal(cp, direct_second.state, restart_second.state)
            and _state_arrays_equal(cp, restart_second.state, loaded1.state)
        ),
        "exact_segment_result_evidence": (
            direct_evidence_sha == restart_evidence_sha
        ),
        "delta_only_route_storage": (
            route_deltas == [EXPECTED_ROUTE_DELTA, EXPECTED_ROUTE_DELTA]
            and route_lists == [EXPECTED_ROUTE_DELTA, EXPECTED_ROUTE_DELTA]
            and all(
                "prior_route_records" not in row
                and "cumulative_route_records" not in row
                for row in rows
            )
            and json_size_ratio <= MAXIMUM_JSON_SIZE_RATIO
        ),
        "route_chain_and_counters": (
            row1["previous_route_chain_sha256"] == row0["route_chain_sha256"]
            and row1["route_record_count_cumulative"] == EXPECTED_ROUTE_TOTAL
            and loaded1.route_record_count_cumulative == EXPECTED_ROUTE_TOTAL
            and loaded1.route_chain_sha256 == row1["route_chain_sha256"]
        ),
        "exact_six_worker_ownership": (
            worker_sets == [
                list(range(ROUTE_WORKER_COUNT)),
                list(range(ROUTE_WORKER_COUNT)),
            ]
            and owner_per_episode
        ),
        "full_chain_replay": (
            loaded1.checkpoint_index == 1
            and loaded1.checkpoint_content_sha256
            == row1["checkpoint_content_sha256"]
            and loaded1.state_sha256 == loaded_state_sha
        ),
        "source_chain_join_exact": (
            row1["source_initial_node_sha256"]
            == row0["source_final_node_sha256"]
        ),
        "checkpoint_artifact_roster_exact": (
            len(artifact_paths) == 6
            and artifact_paths.count("PROTOCOL.json") == 1
            and sum(path.endswith(".json") for path in artifact_paths) == 3
            and sum(path.endswith(".npz") for path in artifact_paths) == 2
            and sum(path.endswith("/.append.lock") for path in artifact_paths)
            == 1
        ),
        "imported_source_identity_matches_protocol": (
            {
                key: start_source_hashes[key]
                for key in in_memory_source_hashes
            }
            == in_memory_source_hashes
        ),
        "source_identity_unchanged_through_completion": (
            completion_source_hashes == start_source_hashes
        ),
        "git_head_unchanged_through_completion": (
            completion_git_head == start_git_head
        ),
        "physical_cpu_affinity_exact": (
            tuple(sorted(os.sched_getaffinity(0))) == PHYSICAL_CPU_AFFINITY
        ),
        "numerical_thread_environment_exact": all(
            os.environ.get(name) == value
            for name, value in NUMERICAL_THREAD_ENVIRONMENT.items()
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
                "owned_episode": OWNED_EPISODE_METHOD_ID,
                "delta_checkpoint": CHECKPOINT_METHOD_ID,
            },
            "genesis_state_sha256": genesis,
            "direct_state_sha256": direct_state_sha,
            "restart_state_sha256": restart_state_sha,
            "loaded_state_sha256": loaded_state_sha,
            "direct_result_evidence_sha256": direct_evidence_sha,
            "restart_result_evidence_sha256": restart_evidence_sha,
            "checkpoint_heads": [
                {
                    "checkpoint_index": row["checkpoint_index"],
                    "checkpoint_content_sha256": row[
                        "checkpoint_content_sha256"
                    ],
                    "route_delta_sha256": row["route_delta_sha256"],
                    "route_chain_sha256": row["route_chain_sha256"],
                    "route_record_count_delta": row[
                        "route_record_count_delta"
                    ],
                    "route_record_count_cumulative": row[
                        "route_record_count_cumulative"
                    ],
                    "segment_episode_count_delta": row[
                        "segment_episode_count_delta"
                    ],
                }
                for row in rows
            ],
            "checkpoint_json_sizes_bytes": json_sizes,
            "checkpoint_json_maximum_size_ratio": json_size_ratio,
            "checkpoint_artifacts": artifacts,
            "source_sha256_at_completion": completion_source_hashes,
            "git_head_at_completion": completion_git_head,
            "gates": gates,
            "claims": CLAIMS,
            "limitations": [
                "synthetic primary-only route-heavy fixture",
                "two checkpoint segments only",
                "episode means consecutive guarded macrosteps, not a physical encounter",
                "checkpoint boundary intentionally discards speculative cache and process ownership",
                "same-runtime deterministic replay only",
                "no long-horizon accuracy, completion, collision, ejection, or science inference",
            ],
        },
        RESULT_DOMAIN,
    )
    _write_json(output_root / "RESULT.json", result)
    manifest_lines = []
    for path in sorted(output_root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        manifest_lines.append(
            f"{_sha256_file(path)}  {path.relative_to(output_root).as_posix()}"
        )
    with (output_root / "SHA256SUMS").open("x", encoding="ascii") as stream:
        stream.write("\n".join(manifest_lines) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "runs/restricted_wh_episode_delta_checkpoint_qualification_v1"
        ),
    )
    args = parser.parse_args()
    result = run(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
