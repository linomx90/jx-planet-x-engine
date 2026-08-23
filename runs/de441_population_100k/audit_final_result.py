#!/usr/bin/env python3
"""Independent readback audit for the locked DE441 100k population result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import struct
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PROJECT = SCRIPT.parents[2]
VENDOR = PROJECT.parent / ".vendor"
if VENDOR.is_dir():
    sys.path.insert(0, str(VENDOR))


INITIAL_FLOAT_FIELDS = (
    "a0_AU",
    "q0_AU",
    "e0",
    "i0_deg",
    "Omega0_rad",
    "omega0_rad",
    "M0_rad",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def simulation_digest(simulation: Any) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("!dii", float(simulation.t), int(simulation.N), int(simulation.N_active)))
    digest.update(str(simulation.integrator).encode("ascii"))
    digest.update(struct.pack("!d", float(simulation.dt)))
    for particle in simulation.particles:
        digest.update(
            struct.pack(
                "!7d",
                particle.m,
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            )
        )
    return digest.hexdigest()


def active_state_sha256(simulation: Any) -> str:
    rows = []
    for index in range(simulation.N_active):
        particle = simulation.particles[index]
        rows.append(
            {
                "index": str(index),
                **{
                    field: float(getattr(particle, field)).hex()
                    for field in ("m", "x", "y", "z", "vx", "vy", "vz")
                },
            }
        )
    return canonical_sha256(rows)


def bootstrap_ci(values: list[float], seed: str, repetitions: int) -> list[float]:
    estimates = []
    for repetition in range(repetitions):
        draw = []
        for index in range(len(values)):
            message = (
                f"jx-paired-block-bootstrap/v1\x1f{seed}\x1f{repetition}\x1f{index}"
            ).encode()
            selected = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") % len(values)
            draw.append(values[selected])
        estimates.append(statistics.fmean(draw))
    estimates.sort()

    def quantile(probability: float) -> float:
        position = probability * (len(estimates) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return estimates[lower]
        fraction = position - lower
        return estimates[lower] * (1.0 - fraction) + estimates[upper] * fraction

    return [quantile(0.025), quantile(0.975)]


def wasserstein_equal(values_a: list[float], values_b: list[float]) -> float:
    if not values_a or len(values_a) != len(values_b):
        raise ValueError("equal nonempty populations are required for this audit")
    return math.fsum(abs(left - right) for left, right in zip(sorted(values_a), sorted(values_b))) / len(
        values_a
    )


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def stage_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "tracers": len(rows),
        "sampled_injections": sum(int(row["sampled_injection"]) for row in rows),
        "bound_final": sum(int(row["bound_final"]) for row in rows),
        "logical_ids": {row["logical_id"] for row in rows},
    }


def comparison_metrics(
    primary: list[dict[str, str]], audit: list[dict[str, str]]
) -> dict[str, float]:
    first = {row["logical_id"]: row for row in primary}
    second = {row["logical_id"]: row for row in audit}
    if set(first) != set(second):
        raise ValueError("audit identity set differs from the primary subset")
    identities = sorted(first)
    left = [first[key] for key in identities]
    right = [second[key] for key in identities]
    count = len(left)
    left_injections = sum(int(row["sampled_injection"]) for row in left)
    right_injections = sum(int(row["sampled_injection"]) for row in right)
    left_bound = sum(int(row["bound_final"]) for row in left)
    right_bound = sum(int(row["bound_final"]) for row in right)
    return {
        "absolute_injection_fraction_difference": abs(right_injections - left_injections) / count,
        "injection_identity_disagreement_fraction": sum(
            first[key]["sampled_injection"] != second[key]["sampled_injection"] for key in identities
        )
        / count,
        "absolute_survival_fraction_difference": abs(right_bound - left_bound) / count,
        "wasserstein_minimum_sampled_q_AU": wasserstein_equal(
            [float(row["minimum_sampled_q_AU"]) for row in left],
            [float(row["minimum_sampled_q_AU"]) for row in right],
        ),
        "wasserstein_final_bound_q_AU": wasserstein_equal(
            [float(row["final_q_AU"]) for row in left if int(row["bound_final"])],
            [float(row["final_q_AU"]) for row in right if int(row["bound_final"])],
        ),
        "wasserstein_final_bound_i_deg": wasserstein_equal(
            [float(row["final_i_deg"]) for row in left if int(row["bound_final"])],
            [float(row["final_i_deg"]) for row in right if int(row["bound_final"])],
        ),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    contract_path = arguments.contract.resolve()
    result_path = arguments.result.resolve()
    execution = arguments.execution.resolve()
    population_path = arguments.population.resolve()
    contract = read_json(contract_path)
    result = read_json(result_path)
    population_rows = read_csv(population_path)
    population = {
        (int(row["block_index"]), int(row["local_index"])): row for row in population_rows
    }
    if len(population) != 100000:
        raise ValueError("locked population is not exactly 100000 unique rows")

    expected_counts = {
        ("primary", "control"): 100,
        ("primary", "source"): 100,
        ("dt_half", "control"): 10,
        ("dt_half", "source"): 10,
        ("sample_fine", "control"): 10,
        ("sample_fine", "source"): 10,
    }
    records = {
        (record["stage"], record["arm"], int(record["block_index"])): record
        for record in result["block_records"]
    }
    expected_keys = {
        (stage, arm, block)
        for (stage, arm), blocks in expected_counts.items()
        for block in range(blocks)
    }
    if set(records) != expected_keys or len(result["block_records"]) != 240:
        raise ValueError("result block record set is incomplete or duplicated")

    rows_by_stage_arm: dict[tuple[str, str], list[dict[str, str]]] = {
        key: [] for key in expected_counts
    }
    checkpoint_count = 0
    checkpoint_bytes = 0
    summary_hashes = 0
    tracer_hashes = 0
    import rebound

    for key in sorted(records):
        stage, arm, block = key
        record = records[key]
        directory = execution / stage / arm / f"block_{block:03d}"
        summary_path = directory / "summary.json"
        tracer_path = directory / "tracers.csv"
        if Path(record["summary_json"]).resolve() != summary_path:
            raise ValueError(f"summary path mismatch for {key}")
        if Path(record["tracer_csv"]).resolve() != tracer_path:
            raise ValueError(f"tracer path mismatch for {key}")
        if sha256_file(summary_path) != record["summary_json_sha256"]:
            raise ValueError(f"summary hash mismatch for {key}")
        if sha256_file(tracer_path) != record["tracer_csv_sha256"]:
            raise ValueError(f"tracer hash mismatch for {key}")
        summary_hashes += 1
        tracer_hashes += 1
        summary = read_json(summary_path)
        if (
            summary["schema"] != "jx-de441-population-block/v1"
            or summary["stage"] != stage
            or summary["arm"] != arm
            or int(summary["block_index"]) != block
            or not summary["restart_replay_state_hash_exact"]
            or not summary["cartesian_state_finite"]
            or int(summary["tracers"]) != 1000
        ):
            raise ValueError(f"invalid summary invariants for {key}")

        rows = read_csv(tracer_path)
        if len(rows) != 1000 or len({row["logical_id"] for row in rows}) != 1000:
            raise ValueError(f"invalid tracer row count or identities for {key}")
        for local, row in enumerate(rows):
            expected = population[(block, local)]
            if (
                int(row["block_index"]) != block
                or int(row["local_index"]) != local
                or row["logical_id"] != expected["logical_id"]
                or any(float(row[field]) != float(expected[field]) for field in INITIAL_FLOAT_FIELDS)
            ):
                raise ValueError(f"initial tracer metadata mismatch for {key}, local {local}")
            finite_fields = ["minimum_sampled_q_AU"]
            if int(row["bound_final"]):
                finite_fields.extend(("final_q_AU", "final_i_deg"))
            if any(not math.isfinite(float(row[field])) for field in finite_fields):
                raise ValueError(f"non-finite tracer output for {key}, local {local}")
        rows_by_stage_arm[(stage, arm)].extend(rows)

        checkpoint_paths = sorted(directory.glob("checkpoint_*.json"))
        if len(checkpoint_paths) != 5:
            raise ValueError(f"expected five checkpoint records for {key}")
        for checkpoint_path in checkpoint_paths:
            checkpoint = read_json(checkpoint_path)
            index = int(checkpoint["checkpoint_index"])
            binary_path = directory / f"checkpoint_{index:03d}.bin"
            if (
                checkpoint["schema"] != "jx-de441-population-checkpoint/v1"
                or checkpoint["job_sha256"] != summary["job_sha256"]
                or checkpoint_path != directory / f"checkpoint_{index:03d}.json"
                or not binary_path.is_file()
            ):
                raise ValueError(f"invalid checkpoint metadata for {key}, checkpoint {index}")
            if sha256_file(binary_path) != checkpoint["simulation_archive_sha256"]:
                raise ValueError(f"checkpoint archive hash mismatch for {key}, checkpoint {index}")
            simulation = rebound.Simulation(str(binary_path))
            if simulation_digest(simulation) != checkpoint["simulation_state_sha256"]:
                raise ValueError(f"checkpoint state digest mismatch for {key}, checkpoint {index}")
            if index == 4 and active_state_sha256(simulation) != summary["active_endpoint_state_sha256"]:
                raise ValueError(f"final checkpoint active state mismatch for {key}")
            checkpoint_count += 1
            checkpoint_bytes += binary_path.stat().st_size

    paired_initial_match = True
    for stage, blocks in (("primary", 100), ("dt_half", 10), ("sample_fine", 10)):
        control = {row["logical_id"]: row for row in rows_by_stage_arm[(stage, "control")]}
        source = {row["logical_id"]: row for row in rows_by_stage_arm[(stage, "source")]}
        if set(control) != set(source):
            paired_initial_match = False
            break
        for identity in control:
            fields = ("block_index", "local_index", "logical_id", *INITIAL_FLOAT_FIELDS)
            if any(control[identity][field] != source[identity][field] for field in fields):
                paired_initial_match = False
                break

    summaries = {key: stage_summary(rows) for key, rows in rows_by_stage_arm.items()}
    control_primary = rows_by_stage_arm[("primary", "control")]
    source_primary = rows_by_stage_arm[("primary", "source")]
    control_by_block = Counter(
        int(row["block_index"]) for row in control_primary if int(row["sampled_injection"])
    )
    source_by_block = Counter(
        int(row["block_index"]) for row in source_primary if int(row["sampled_injection"])
    )
    block_effects = [(source_by_block[i] - control_by_block[i]) / 1000 for i in range(100)]
    bootstrap = bootstrap_ci(
        block_effects,
        contract["statistics"]["bootstrap_seed"],
        int(contract["statistics"]["bootstrap_repetitions"]),
    )
    primary_effect = (
        summaries[("primary", "source")]["sampled_injections"]
        - summaries[("primary", "control")]["sampled_injections"]
    ) / 100000
    margin = float(contract["statistics"]["equivalence_margin"])
    classification = (
        "RESOLVED_POSITIVE_SOURCE_EFFECT"
        if bootstrap[0] > 0.0
        else "RESOLVED_NEGATIVE_SOURCE_EFFECT"
        if bootstrap[1] < 0.0
        else "EQUIVALENT_WITHIN_LOCKED_MARGIN"
        if bootstrap[0] >= -margin and bootstrap[1] <= margin
        else "NO_RESOLVED_EFFECT"
    )

    convergence: dict[str, Any] = {}
    metric_names = (
        "absolute_injection_fraction_difference",
        "injection_identity_disagreement_fraction",
        "absolute_survival_fraction_difference",
        "wasserstein_minimum_sampled_q_AU",
        "wasserstein_final_bound_q_AU",
        "wasserstein_final_bound_i_deg",
    )
    for stage, result_key in (
        ("dt_half", "dt_half_convergence"),
        ("sample_fine", "sample_cadence_convergence"),
    ):
        convergence[stage] = {}
        for arm in ("control", "source"):
            primary_subset = [
                row for row in rows_by_stage_arm[("primary", arm)] if int(row["block_index"]) < 10
            ]
            metrics = comparison_metrics(primary_subset, rows_by_stage_arm[(stage, arm)])
            recorded = result[result_key]["arms"][arm]["metrics"]
            if any(not close(metrics[name], float(recorded[name])) for name in metric_names):
                raise ValueError(f"recomputed convergence metrics differ for {stage}/{arm}")
            convergence[stage][arm] = metrics

    recorded_effect = result["population_screening"]["source_minus_control"]
    checks = {
        "result_contract_hash_matches": result["contract_sha256"] == sha256_file(contract_path),
        "result_hash_is_stable": sha256_file(result_path)
        == "24b7572cf130c683acd66a4677dac62d0d30b15b24f9bf5997b612a8045d7efd",
        "all_result_gates_passed": bool(result["all_gates_passed"])
        and all(result["checks"].values()),
        "record_set_exact": set(records) == expected_keys,
        "all_summary_hashes_verified": summary_hashes == 240,
        "all_tracer_hashes_verified": tracer_hashes == 240,
        "all_checkpoint_archives_and_states_verified": checkpoint_count == 1200,
        "all_tracer_rows_verified": sum(len(rows) for rows in rows_by_stage_arm.values()) == 240000,
        "paired_initial_metadata_exact": paired_initial_match,
        "primary_count_recomputed": summaries[("primary", "control")]["tracers"]
        == summaries[("primary", "source")]["tracers"]
        == 100000,
        "primary_injection_counts_recomputed": summaries[("primary", "control")][
            "sampled_injections"
        ]
        == 4377
        and summaries[("primary", "source")]["sampled_injections"] == 4374,
        "primary_survival_recomputed": summaries[("primary", "control")]["bound_final"]
        == summaries[("primary", "source")]["bound_final"]
        == 100000,
        "primary_effect_recomputed": close(
            primary_effect, float(recorded_effect["sampled_injection_fraction"]), 1e-15
        ),
        "bootstrap_interval_recomputed": all(
            close(left, float(right), 1e-15)
            for left, right in zip(bootstrap, recorded_effect["paired_block_bootstrap_95_percent_CI"])
        ),
        "effect_classification_recomputed": classification
        == recorded_effect["effect_classification"],
        "dt_half_metrics_recomputed": True,
        "sample_cadence_metrics_recomputed": True,
    }
    verdict = "AUDIT_PASSED" if all(checks.values()) else "AUDIT_FAILED"
    audit = {
        "schema": "jx-de441-population-final-audit/v1",
        "verdict": verdict,
        "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "population": {"path": str(population_path), "sha256": sha256_file(population_path)},
        "audit_script_sha256": sha256_file(SCRIPT),
        "checks": checks,
        "verified": {
            "block_summaries": summary_hashes,
            "tracer_csvs": tracer_hashes,
            "tracer_rows": sum(len(rows) for rows in rows_by_stage_arm.values()),
            "checkpoint_archives": checkpoint_count,
            "checkpoint_archive_bytes": checkpoint_bytes,
            "record_counts": {
                f"{stage}/{arm}": count for (stage, arm), count in sorted(expected_counts.items())
            },
        },
        "independent_primary_recomputation": {
            "control_sampled_injections": summaries[("primary", "control")]["sampled_injections"],
            "source_sampled_injections": summaries[("primary", "source")]["sampled_injections"],
            "control_bound_final": summaries[("primary", "control")]["bound_final"],
            "source_bound_final": summaries[("primary", "source")]["bound_final"],
            "source_minus_control_injection_fraction": primary_effect,
            "paired_block_bootstrap_95_percent_CI": bootstrap,
            "equivalence_margin": margin,
            "effect_classification": classification,
        },
        "independent_convergence_recomputation": convergence,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "This audit verifies stored computation and deterministic statistics; it is not independent dynamical software replication.",
    }
    atomic_json(arguments.output.resolve(), audit)
    print(json.dumps({"verdict": verdict, "output": str(arguments.output.resolve())}))
    return 0 if verdict == "AUDIT_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
