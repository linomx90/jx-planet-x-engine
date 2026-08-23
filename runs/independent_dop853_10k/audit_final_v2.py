#!/usr/bin/env python3
"""Independently audit stored DOP853 v2 outputs and recompute its verdict."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


POPULATION_COLUMNS = (
    "block_index",
    "local_index",
    "logical_id",
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
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
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=no_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def resolve(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base.parent / candidate).resolve()


def approximately_equal(left: Any, right: Any, tolerance: float = 1e-12) -> bool:
    if isinstance(left, bool) or isinstance(right, bool) or left is None or right is None:
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(
            approximately_equal(left[key], right[key], tolerance) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            approximately_equal(a, b, tolerance) for a, b in zip(left, right)
        )
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    return left == right


def population_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    count = len(rows)
    injections = sum(int(row["sampled_injection"]) for row in rows)
    bound = sum(int(row["bound_final"]) for row in rows)
    return {
        "tracers": count,
        "sampled_injections": injections,
        "sampled_injection_fraction": injections / count,
        "bound_final": bound,
        "survival_fraction": bound / count,
    }


def wasserstein_equal_population(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("audit expects nonempty equal-size populations")
    a = np.sort(np.asarray(left, dtype=np.float64))
    b = np.sort(np.asarray(right, dtype=np.float64))
    return float(np.mean(np.abs(a - b)))


def finite_values(rows: list[dict[str, str]], field: str, bound_only: bool = False) -> list[float]:
    values = []
    for row in rows:
        if bound_only and int(row["bound_final"]) != 1:
            continue
        value = float(row[field])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite {field} for {row['logical_id']}")
        values.append(value)
    return values


def compare_populations(reference: list[dict[str, str]], independent: list[dict[str, str]]) -> dict[str, Any]:
    reference_by_id = {row["logical_id"]: row for row in reference}
    independent_by_id = {row["logical_id"]: row for row in independent}
    if len(reference_by_id) != len(reference) or len(independent_by_id) != len(independent):
        raise ValueError("duplicate tracer identity")
    if set(reference_by_id) != set(independent_by_id):
        raise ValueError("reference/independent identity set mismatch")
    for identity, reference_row in reference_by_id.items():
        if any(reference_row[field] != independent_by_id[identity][field] for field in POPULATION_COLUMNS):
            raise ValueError(f"initial metadata mismatch: {identity}")
    reference_summary = population_summary(reference)
    independent_summary = population_summary(independent)
    disagreements = sum(
        reference_by_id[identity]["sampled_injection"]
        != independent_by_id[identity]["sampled_injection"]
        for identity in reference_by_id
    )
    return {
        "reference": reference_summary,
        "independent": independent_summary,
        "metrics": {
            "absolute_injection_fraction_difference": abs(
                independent_summary["sampled_injection_fraction"]
                - reference_summary["sampled_injection_fraction"]
            ),
            "injection_identity_disagreement_fraction": disagreements / len(reference),
            "absolute_survival_fraction_difference": abs(
                independent_summary["survival_fraction"] - reference_summary["survival_fraction"]
            ),
            "wasserstein_minimum_sampled_q_AU": wasserstein_equal_population(
                finite_values(reference, "minimum_sampled_q_AU"),
                finite_values(independent, "minimum_sampled_q_AU"),
            ),
            "wasserstein_final_bound_q_AU": wasserstein_equal_population(
                finite_values(reference, "final_q_AU", True),
                finite_values(independent, "final_q_AU", True),
            ),
            "wasserstein_final_bound_i_deg": wasserstein_equal_population(
                finite_values(reference, "final_i_deg", True),
                finite_values(independent, "final_i_deg", True),
            ),
        },
    }


def bootstrap_ci(values: list[float], seed: str, repetitions: int) -> list[float]:
    estimates = []
    for repetition in range(repetitions):
        draw = []
        for index in range(len(values)):
            message = f"jx-independent-paired-bootstrap/v1\x1f{seed}\x1f{repetition}\x1f{index}".encode()
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


def effect(
    control: list[dict[str, str]],
    source: list[dict[str, str]],
    blocks: list[int],
    seed: str,
    repetitions: int,
    margin: float,
) -> dict[str, Any]:
    if {row["logical_id"] for row in control} != {row["logical_id"] for row in source}:
        raise ValueError("source/control identities differ")
    block_effects = []
    for block in blocks:
        control_block = [row for row in control if int(row["block_index"]) == block]
        source_block = [row for row in source if int(row["block_index"]) == block]
        block_effects.append(
            population_summary(source_block)["sampled_injection_fraction"]
            - population_summary(control_block)["sampled_injection_fraction"]
        )
    interval = bootstrap_ci(block_effects, seed, repetitions)
    if interval[0] > 0.0:
        classification = "RESOLVED_POSITIVE_SOURCE_EFFECT"
    elif interval[1] < 0.0:
        classification = "RESOLVED_NEGATIVE_SOURCE_EFFECT"
    elif interval[0] >= -margin and interval[1] <= margin:
        classification = "EQUIVALENT_WITHIN_LOCKED_MARGIN"
    else:
        classification = "NO_RESOLVED_EFFECT"
    control_summary = population_summary(control)
    source_summary = population_summary(source)
    return {
        "control": control_summary,
        "source": source_summary,
        "source_minus_control_injection_fraction": (
            source_summary["sampled_injection_fraction"]
            - control_summary["sampled_injection_fraction"]
        ),
        "block_effects": block_effects,
        "paired_block_bootstrap_95_percent_CI": interval,
        "equivalence_margin": margin,
        "classification": classification,
    }


def endpoint_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[*row["position_AU"], *row["velocity_AU_per_year"]] for row in records],
        dtype=np.float64,
    )


def endpoint_disagreement(
    summaries: list[dict[str, Any]], active_audit: dict[str, Any]
) -> dict[str, float]:
    reference = endpoint_array(active_audit["endpoint_state"])
    maximum_position = 0.0
    maximum_velocity = 0.0
    for summary in summaries:
        state = endpoint_array(summary["active_endpoint_state"])
        maximum_position = max(
            maximum_position, float(np.linalg.norm(state[:, :3] - reference[:, :3], axis=1).max())
        )
        maximum_velocity = max(
            maximum_velocity,
            float(np.linalg.norm(state[:, 3:] - reference[:, 3:], axis=1).max()),
        )
    return {
        "maximum_active_endpoint_position_disagreement_AU": maximum_position,
        "maximum_active_endpoint_velocity_disagreement_AU_per_year": maximum_velocity,
    }


def orbital_elements(state: np.ndarray, active_count: int, primary_gm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    particle_count = state.size // 6
    positions = state[: 3 * particle_count].reshape(particle_count, 3)
    velocities = state[3 * particle_count :].reshape(particle_count, 3)
    relative_position = positions[active_count:] - positions[0]
    relative_velocity = velocities[active_count:] - velocities[0]
    radius = np.linalg.norm(relative_position, axis=1)
    speed2 = np.sum(relative_velocity * relative_velocity, axis=1)
    energy = 0.5 * speed2 - primary_gm / radius
    semimajor = np.full_like(energy, np.nan)
    negative = energy < 0.0
    semimajor[negative] = -primary_gm / (2.0 * energy[negative])
    angular = np.cross(relative_position, relative_velocity)
    angular_norm = np.linalg.norm(angular, axis=1)
    eccentricity_vector = np.cross(relative_velocity, angular) / primary_gm - relative_position / radius[:, None]
    eccentricity = np.linalg.norm(eccentricity_vector, axis=1)
    bound = negative & (eccentricity < 1.0) & np.isfinite(semimajor) & (angular_norm > 0.0)
    perihelion = np.full_like(semimajor, np.nan)
    perihelion[bound] = semimajor[bound] * (1.0 - eccentricity[bound])
    inclination = np.full_like(semimajor, np.nan)
    cosine = np.clip(angular[:, 2] / angular_norm, -1.0, 1.0)
    inclination[bound] = np.degrees(np.arccos(cosine[bound]))
    return perihelion, inclination, bound


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    script_path = Path(__file__).resolve()
    contract_path = arguments.contract.resolve()
    result_path = arguments.result.resolve()
    execution_root = arguments.execution_root.resolve()
    output_path = arguments.output.resolve()
    contract = read_json(contract_path)
    result = read_json(result_path)
    blocks = [int(value) for value in contract["selection"]["selected_blocks"]]
    expected_records = {(arm, block) for arm in ("control", "source") for block in blocks}
    checks: dict[str, bool] = {}

    checks["result_contract_hash_matches"] = result["contract_sha256"] == sha256_file(contract_path)
    checks["contract_is_corrective_v2"] = (
        contract["registration_phase"] == "POST_V1_NUMERICAL_FAILURE_AND_BEFORE_V2_TRAJECTORIES"
        and contract["registered_after_v1_outcomes"] is True
    )
    verified_locked_files = []
    for label, specification in contract["locked_files"].items():
        path = resolve(contract_path, specification["path"])
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(f"locked file hash mismatch: {label}")
        verified_locked_files.append({"label": label, "path": str(path), "sha256": observed})
    checks["all_contract_locked_files_verified"] = len(verified_locked_files) == len(
        contract["locked_files"]
    )
    for label, record in result["runtime"]["files"].items():
        if sha256_file(Path(record["path"])) != record["sha256"]:
            raise ValueError(f"runtime hash mismatch: {label}")
    checks["runtime_files_verified"] = True

    selection = read_json(resolve(contract_path, contract["selection"]["path"]))
    checks["selection_exact"] = (
        selection["selected_blocks"] == blocks
        and len(set(blocks)) == 10
        and selection["selection_status"] == "OUTCOME_BLIND_HASH_RANKED"
    )
    record_map = {
        (str(record["arm"]), int(record["block_index"])): record
        for record in result["block_records"]
    }
    checks["record_set_exact"] = set(record_map) == expected_records and len(record_map) == 20
    if not checks["record_set_exact"]:
        raise ValueError("block record set is incomplete or duplicated")

    state_path = resolve(contract_path, contract["states"]["control_path"])
    state_rows = read_csv(state_path)
    primary_gm = float(state_rows[0]["mass"])
    independent_rows: dict[str, list[dict[str, str]]] = {"control": [], "source": []}
    summaries_by_arm: dict[str, list[dict[str, Any]]] = {"control": [], "source": []}
    checkpoint_manifest = []
    tracer_manifest = []
    summary_manifest = []
    checkpoint_state_bytes = 0
    final_state_element_checks = []
    q_lower = float(contract["classification"]["q_threshold_AU"]) - float(
        contract["classification"]["q_hysteresis_AU"]
    )

    for arm, block in sorted(expected_records):
        record = record_map[(arm, block)]
        directory = execution_root / arm / f"block_{block:03d}"
        summary_path = directory / "summary.json"
        tracer_path = directory / "tracers.csv"
        if Path(record["summary_json"]).resolve() != summary_path or Path(record["tracer_csv"]).resolve() != tracer_path:
            raise ValueError("result path does not match the contracted execution root")
        summary_hash = sha256_file(summary_path)
        tracer_hash = sha256_file(tracer_path)
        if summary_hash != record["summary_json_sha256"] or tracer_hash != record["tracer_csv_sha256"]:
            raise ValueError("result file hash mismatch")
        summary = read_json(summary_path)
        rows = read_csv(tracer_path)
        if summary["schema"] != "jx-independent-dop853-block/v1" or summary["arm"] != arm:
            raise ValueError("block summary identity mismatch")
        if int(summary["block_index"]) != block or int(summary["tracers"]) != 1000 or len(rows) != 1000:
            raise ValueError("block cardinality mismatch")
        if summary["tracer_csv_sha256"] != tracer_hash or not summary["checkpoint_replay_binary64_exact"]:
            raise ValueError("summary integrity flag mismatch")
        if canonical_sha256(summary["active_endpoint_state"]) != summary["active_endpoint_state_sha256"]:
            raise ValueError("active endpoint hash mismatch")
        if any(int(row["block_index"]) != block for row in rows):
            raise ValueError("tracer row block mismatch")
        if len({row["logical_id"] for row in rows}) != 1000:
            raise ValueError("duplicate block tracer identity")
        for row in rows:
            for field in (*POPULATION_COLUMNS[3:], "minimum_sampled_q_AU"):
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"nonfinite tracer field: {field}")
            if int(row["sampled_injection"]) != int(float(row["minimum_sampled_q_AU"]) < q_lower):
                raise ValueError("stored injection classification mismatch")
        independent_rows[arm].extend(rows)
        summaries_by_arm[arm].append(summary)
        summary_manifest.append({"arm": arm, "block_index": block, "sha256": summary_hash})
        tracer_manifest.append({"arm": arm, "block_index": block, "sha256": tracer_hash, "rows": len(rows)})

        json_files = sorted(directory.glob("checkpoint_*.json"))
        npz_files = sorted(directory.glob("checkpoint_*.npz"))
        if [path.name for path in json_files] != [f"checkpoint_{index:03d}.json" for index in range(5)]:
            raise ValueError("checkpoint JSON set mismatch")
        if [path.name for path in npz_files] != [f"checkpoint_{index:03d}.npz" for index in range(5)]:
            raise ValueError("checkpoint array set mismatch")
        expected_shape = (6036 if arm == "control" else 6042,)
        final_checkpoint = None
        final_state = None
        for index, (json_path, npz_path) in enumerate(zip(json_files, npz_files)):
            checkpoint = read_json(json_path)
            npz_hash = sha256_file(npz_path)
            if checkpoint["schema"] != "jx-independent-dop853-checkpoint/v1":
                raise ValueError("checkpoint schema mismatch")
            if int(checkpoint["checkpoint_index"]) != index or checkpoint["job_sha256"] != summary["job_sha256"]:
                raise ValueError("checkpoint identity mismatch")
            if not math.isclose(float(checkpoint["time_year"]), index * 2500.0, abs_tol=0.0):
                raise ValueError("checkpoint epoch mismatch")
            if tuple(checkpoint["state_shape"]) != expected_shape or checkpoint["state_npz_sha256"] != npz_hash:
                raise ValueError("checkpoint array manifest mismatch")
            with np.load(npz_path, allow_pickle=False) as archive:
                if archive.files != ["state"]:
                    raise ValueError("unexpected checkpoint array names")
                state = np.asarray(archive["state"])
            if state.dtype != np.float64 or state.shape != expected_shape or not np.all(np.isfinite(state)):
                raise ValueError("invalid checkpoint state")
            if array_sha256(state) != checkpoint["state_array_sha256"]:
                raise ValueError("checkpoint state hash mismatch")
            expected_sample_count = int(index * 2500.0 / float(contract["dynamics"]["sample_years"])) + 1
            if int(checkpoint["tracker"]["sample_count"]) != expected_sample_count:
                raise ValueError("checkpoint sample count mismatch")
            if len(checkpoint["completed_segment_records"]) != index:
                raise ValueError("checkpoint segment count mismatch")
            checkpoint_state_bytes += npz_path.stat().st_size
            checkpoint_manifest.append(
                {
                    "arm": arm,
                    "block_index": block,
                    "checkpoint_index": index,
                    "json_sha256": sha256_file(json_path),
                    "npz_sha256": npz_hash,
                    "state_array_sha256": checkpoint["state_array_sha256"],
                }
            )
            final_checkpoint, final_state = checkpoint, state
        assert final_checkpoint is not None and final_state is not None
        if final_checkpoint["completed_segment_records"] != summary["segments"]:
            raise ValueError("final checkpoint segment records differ from summary")
        tracker = final_checkpoint["tracker"]
        if any(len(tracker[key]) != 1000 for key in (
            "minimum_q", "first_low_q_year", "ever_unbound", "first_unbound_year"
        )):
            raise ValueError("final tracker cardinality mismatch")
        active_count = 6 if arm == "control" else 7
        particle_count = final_state.size // 6
        positions = final_state[: 3 * particle_count].reshape(particle_count, 3)
        velocities = final_state[3 * particle_count :].reshape(particle_count, 3)
        active_record = [
            {
                "index": index,
                "position_AU": [float(value) for value in positions[index]],
                "velocity_AU_per_year": [float(value) for value in velocities[index]],
            }
            for index in range(active_count)
        ]
        if active_record != summary["active_endpoint_state"]:
            raise ValueError("final checkpoint active state differs from summary")
        final_q, final_i, final_bound = orbital_elements(final_state, active_count, primary_gm)
        maximum_q_error = 0.0
        maximum_i_error = 0.0
        for local_index, row in enumerate(rows):
            if float(row["minimum_sampled_q_AU"]) != float(tracker["minimum_q"][local_index]):
                raise ValueError("final tracker minimum q differs from tracer CSV")
            if int(row["ever_unbound_at_sample"]) != int(tracker["ever_unbound"][local_index]):
                raise ValueError("final tracker unbound flag differs from tracer CSV")
            stored_first_low = None if row["first_sampled_low_q_year"] == "" else float(row["first_sampled_low_q_year"])
            stored_first_unbound = None if row["first_unbound_year"] == "" else float(row["first_unbound_year"])
            if stored_first_low != tracker["first_low_q_year"][local_index]:
                raise ValueError("first-low-q epoch mismatch")
            if stored_first_unbound != tracker["first_unbound_year"][local_index]:
                raise ValueError("first-unbound epoch mismatch")
            if int(row["bound_final"]) != int(final_bound[local_index]):
                raise ValueError("final bound flag mismatch")
            if final_bound[local_index]:
                maximum_q_error = max(maximum_q_error, abs(float(row["final_q_AU"]) - float(final_q[local_index])))
                maximum_i_error = max(maximum_i_error, abs(float(row["final_i_deg"]) - float(final_i[local_index])))
        if maximum_q_error > 1e-12 or maximum_i_error > 1e-12:
            raise ValueError("final osculating elements do not reproduce")
        final_state_element_checks.append(
            {
                "arm": arm,
                "block_index": block,
                "maximum_final_q_error_AU": maximum_q_error,
                "maximum_final_i_error_deg": maximum_i_error,
            }
        )

    checks["all_summary_hashes_verified"] = len(summary_manifest) == 20
    checks["all_tracer_hashes_and_rows_verified"] = (
        len(tracer_manifest) == 20 and sum(row["rows"] for row in tracer_manifest) == 20000
    )
    checks["all_checkpoint_arrays_and_trackers_verified"] = len(checkpoint_manifest) == 100
    checks["final_elements_recomputed_from_checkpoint_states"] = len(final_state_element_checks) == 20

    reference_rows: dict[str, list[dict[str, str]]] = {"control": [], "source": []}
    reference_manifest = []
    for arm in ("control", "source"):
        records = result["verified_reference_tracer_files"][arm]
        if {int(record["block_index"]) for record in records} != set(blocks):
            raise ValueError("reference block set mismatch")
        for record in records:
            path = Path(record["path"]).resolve()
            observed = sha256_file(path)
            if observed != record["sha256"]:
                raise ValueError("reference tracer hash mismatch")
            rows = read_csv(path)
            if len(rows) != 1000:
                raise ValueError("reference tracer row count mismatch")
            reference_rows[arm].extend(rows)
            reference_manifest.append(
                {"arm": arm, "block_index": int(record["block_index"]), "sha256": observed}
            )
    checks["all_reference_tracers_verified"] = len(reference_manifest) == 20
    for identity, control_row in {row["logical_id"]: row for row in independent_rows["control"]}.items():
        source_row = {row["logical_id"]: row for row in independent_rows["source"]}[identity]
        if any(control_row[field] != source_row[field] for field in POPULATION_COLUMNS):
            raise ValueError("independent source/control initial metadata mismatch")
    checks["paired_initial_metadata_exact"] = True

    comparisons = {
        arm: compare_populations(reference_rows[arm], independent_rows[arm])
        for arm in ("control", "source")
    }
    statistics_contract = contract["statistics"]
    independent_effect = effect(
        independent_rows["control"],
        independent_rows["source"],
        blocks,
        statistics_contract["bootstrap_seed"],
        int(statistics_contract["bootstrap_repetitions"]),
        float(statistics_contract["equivalence_margin"]),
    )
    reference_effect = effect(
        reference_rows["control"],
        reference_rows["source"],
        blocks,
        statistics_contract["bootstrap_seed"],
        int(statistics_contract["bootstrap_repetitions"]),
        float(statistics_contract["equivalence_margin"]),
    )
    effect_difference = abs(
        independent_effect["source_minus_control_injection_fraction"]
        - reference_effect["source_minus_control_injection_fraction"]
    )
    checks["population_comparisons_recomputed"] = approximately_equal(comparisons, result["comparisons"])
    checks["independent_effect_and_bootstrap_recomputed"] = approximately_equal(
        independent_effect, result["independent_effect"]
    )
    checks["reference_effect_and_bootstrap_recomputed"] = approximately_equal(
        reference_effect, result["reference_effect"]
    )
    checks["effect_difference_recomputed"] = math.isclose(
        effect_difference,
        float(result["absolute_source_control_effect_difference"]),
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    diagnostic = read_json(resolve(contract_path, contract["locked_files"]["v1_diagnostic_result"]["path"]))
    active_audits = result["active_only_audits"]
    diagnostic_matches = True
    audit_keys = (
        "accepted_steps",
        "rhs_evaluations",
        "maximum_relative_energy_drift",
        "maximum_relative_angular_momentum_vector_drift",
        "source_bound_at_every_accepted_step",
        "source_minimum_q_AU",
        "source_maximum_fractional_a_excursion",
        "endpoint_state",
        "endpoint_state_sha256",
    )
    for arm in ("control", "source"):
        pre_v2 = diagnostic["refined_active_only_audits"][arm]
        diagnostic_matches &= all(
            approximately_equal(pre_v2[key], active_audits[arm][key]) for key in audit_keys
        )
        diagnostic_matches &= canonical_sha256(active_audits[arm]["endpoint_state"]) == active_audits[arm][
            "endpoint_state_sha256"
        ]
    checks["active_only_audits_match_pre_v2_diagnostic"] = diagnostic_matches

    endpoint_metrics = {
        arm: endpoint_disagreement(summaries_by_arm[arm], active_audits[arm])
        for arm in ("control", "source")
    }
    gates = contract["gates"]
    comparison_checks = {}
    for arm in ("control", "source"):
        metrics = comparisons[arm]["metrics"]
        comparison_checks[arm] = {
            "injection_fraction": metrics["absolute_injection_fraction_difference"]
            <= float(gates["max_injection_fraction_difference"]),
            "injection_identity": metrics["injection_identity_disagreement_fraction"]
            <= float(gates["max_injection_identity_disagreement_fraction"]),
            "survival_fraction": metrics["absolute_survival_fraction_difference"]
            <= float(gates["max_survival_fraction_difference"]),
            "minimum_q_wasserstein": metrics["wasserstein_minimum_sampled_q_AU"]
            <= float(gates["max_wasserstein_minimum_q_AU"]),
            "final_q_wasserstein": metrics["wasserstein_final_bound_q_AU"]
            <= float(gates["max_wasserstein_final_q_AU"]),
            "final_i_wasserstein": metrics["wasserstein_final_bound_i_deg"]
            <= float(gates["max_wasserstein_final_i_deg"]),
        }
    maximum_roundtrip = max(
        float(value)
        for summaries in summaries_by_arm.values()
        for summary in summaries
        for value in summary["initial_element_roundtrip"].values()
    )
    maximum_energy = max(
        [
            float(summary["maximum_relative_active_energy_drift"])
            for summaries in summaries_by_arm.values()
            for summary in summaries
        ]
        + [float(active_audits[arm]["maximum_relative_energy_drift"]) for arm in ("control", "source")]
    )
    maximum_angular = max(
        [
            float(summary["maximum_relative_active_angular_momentum_vector_drift"])
            for summaries in summaries_by_arm.values()
            for summary in summaries
        ]
        + [
            float(active_audits[arm]["maximum_relative_angular_momentum_vector_drift"])
            for arm in ("control", "source")
        ]
    )
    maximum_endpoint_position = max(
        record["maximum_active_endpoint_position_disagreement_AU"]
        for record in endpoint_metrics.values()
    )
    maximum_endpoint_velocity = max(
        record["maximum_active_endpoint_velocity_disagreement_AU_per_year"]
        for record in endpoint_metrics.values()
    )
    source_summaries = summaries_by_arm["source"]
    numerical_checks = {
        "complete_finite_outputs": True,
        "checkpoint_replay_exact": True,
        "initial_element_roundtrip": maximum_roundtrip
        <= float(gates["max_initial_element_roundtrip_error"]),
        "active_energy_drift": maximum_energy <= float(gates["max_relative_active_energy_drift"]),
        "active_angular_momentum_drift": maximum_angular
        <= float(gates["max_relative_active_angular_momentum_vector_drift"]),
        "active_endpoint_position_consistency": maximum_endpoint_position
        <= float(gates["max_active_endpoint_position_disagreement_AU"]),
        "active_endpoint_velocity_consistency": maximum_endpoint_velocity
        <= float(gates["max_active_endpoint_velocity_disagreement_AU_per_year"]),
        "source_bound": all(summary["source_bound_at_every_accepted_step"] for summary in source_summaries)
        and bool(active_audits["source"]["source_bound_at_every_accepted_step"]),
        "source_minimum_q": min(
            [float(summary["source_minimum_q_AU"]) for summary in source_summaries]
            + [float(active_audits["source"]["source_minimum_q_AU"])]
        )
        >= float(gates["source_minimum_q_AU"]),
        "source_semimajor_axis_excursion": max(
            [float(summary["source_maximum_fractional_a_excursion"]) for summary in source_summaries]
            + [float(active_audits["source"]["source_maximum_fractional_a_excursion"])]
        )
        <= float(gates["source_maximum_fractional_a_excursion"]),
    }
    cross_software_checks = {
        "control_population_comparison": all(comparison_checks["control"].values()),
        "source_population_comparison": all(comparison_checks["source"].values()),
        "source_control_effect_difference": effect_difference
        <= float(gates["max_source_control_effect_difference"]),
    }
    recomputed_verdict = (
        "INVALID"
        if not all(numerical_checks.values())
        else "CONFLICT"
        if not all(cross_software_checks.values())
        else "PASSED"
    )
    checks["endpoint_disagreement_recomputed"] = approximately_equal(
        endpoint_metrics, result["active_endpoint_disagreement"], 1e-14
    )
    checks["numerical_extrema_recomputed"] = (
        math.isclose(maximum_roundtrip, float(result["maximum_initial_element_roundtrip_error"]), abs_tol=1e-15)
        and math.isclose(maximum_energy, float(result["maximum_relative_active_energy_drift"]), abs_tol=1e-15)
        and math.isclose(
            maximum_angular,
            float(result["maximum_relative_active_angular_momentum_vector_drift"]),
            abs_tol=1e-15,
        )
    )
    checks["comparison_gate_decisions_recomputed"] = comparison_checks == result["comparison_checks"]
    checks["numerical_gate_decisions_recomputed"] = numerical_checks == result["numerical_checks"]
    checks["cross_software_gate_decisions_recomputed"] = cross_software_checks == result[
        "cross_software_checks"
    ]
    checks["verdict_recomputed"] = (
        recomputed_verdict == result["verdict"] == "PASSED"
        and result["science_status"] == "SCREENING_ONLY"
        and result["claim_decision"] == "SCREENING_ONLY"
        and result["all_gates_passed"] is True
    )
    audit_verdict = "AUDIT_PASSED" if all(checks.values()) else "AUDIT_FAILED"
    audit = {
        "schema": "jx-independent-dop853-final-audit/v1",
        "verdict": audit_verdict,
        "checks": checks,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "result": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "audit_script_sha256": sha256_file(script_path),
        "verified": {
            "locked_files": len(verified_locked_files),
            "block_summaries": len(summary_manifest),
            "tracer_csvs": len(tracer_manifest),
            "tracer_rows": sum(row["rows"] for row in tracer_manifest),
            "reference_tracer_csvs": len(reference_manifest),
            "checkpoint_jsons": len(checkpoint_manifest),
            "checkpoint_npzs": len(checkpoint_manifest),
            "checkpoint_npz_bytes": checkpoint_state_bytes,
        },
        "independent_recomputation": {
            "comparisons": comparisons,
            "independent_effect": independent_effect,
            "reference_effect": reference_effect,
            "absolute_source_control_effect_difference": effect_difference,
            "active_endpoint_disagreement": endpoint_metrics,
            "maximum_initial_element_roundtrip_error": maximum_roundtrip,
            "maximum_relative_active_energy_drift": maximum_energy,
            "maximum_relative_active_angular_momentum_vector_drift": maximum_angular,
            "comparison_checks": comparison_checks,
            "numerical_checks": numerical_checks,
            "cross_software_checks": cross_software_checks,
            "verdict": recomputed_verdict,
        },
        "manifests": {
            "block_summaries": summary_manifest,
            "tracer_csvs": tracer_manifest,
            "reference_tracer_csvs": reference_manifest,
            "checkpoints": checkpoint_manifest,
        },
        "final_state_element_checks": final_state_element_checks,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "This audit independently rehashes stored files, reconstructs final osculating elements from checkpoint states, and recomputes statistics and gates. It does not rerun the 10000-year trajectories and does not change SCREENING_ONLY scope."
    }
    atomic_json(output_path, audit)
    print(json.dumps({"verdict": audit_verdict, "output": str(output_path)}), flush=True)
    return 0 if audit_verdict == "AUDIT_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
