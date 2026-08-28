#!/usr/bin/env python3
"""Verify two clean JX-E1 50 kyr executions without executing another simulation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import run_long_pilot as long


RECEIPT_SCHEMA = "jx-e1-long-engineering-replay-receipt/v1"
REPLAY_VERDICT = "ENGINEERING_LONG_INDEPENDENT_SEMANTIC_REPLAY_EXACT"
EXPECTED_RESULT_CHECK_KEYS = {
    "complete_primary_matrix",
    "complete_timestep_audit_matrix",
    "paired_diagnostic_count_exact",
    "timestep_comparison_count_exact",
    "all_arm_checks_true",
    "all_M1_common_initial_states_match_block_M0",
    "all_timestep_audit_common_initial_states_match_primary",
    "timestep_bound_and_sampled_event_identities_exact",
    "timestep_minimum_paired_bound_count_met",
    "timestep_max_final_q_difference_within_gate",
    "timestep_max_final_i_difference_within_gate",
    "persistent_cumulative_wall_time_within_cap",
    "final_peak_rss_within_cap",
    "final_free_disk_floor_met",
    "output_size_within_cap_before_result",
    "no_failure_receipt_present",
}


def independent_percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def independent_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    q_values = [row["final_q_AU"] for row in rows if row["final_q_AU"] is not None]
    i_values = [row["final_i_deg"] for row in rows if row["final_i_deg"] is not None]
    neptune = [
        row["minimum_neptune_distance_AU_sampled"]
        for row in rows
        if row["minimum_neptune_distance_AU_sampled"] is not None
    ]
    p9 = [
        row["minimum_p9_distance_AU_sampled"]
        for row in rows
        if row["minimum_p9_distance_AU_sampled"] is not None
    ]
    return {
        "tracer_count": count,
        "bound_fraction_final": sum(row["bound_final"] for row in rows) / count,
        "ever_q_lt_30_fraction_sampled": sum(
            row["ever_q_lt_30_sampled"] for row in rows
        ) / count,
        "ever_i_gt_40_fraction_sampled": sum(
            row["ever_i_gt_40_sampled"] for row in rows
        ) / count,
        "ever_i_gt_60_fraction_sampled": sum(
            row["ever_i_gt_60_sampled"] for row in rows
        ) / count,
        "ever_unbound_fraction_sampled": sum(
            row["ever_unbound_sampled"] for row in rows
        ) / count,
        "orbit_conversion_failure_count": sum(
            row["orbit_conversion_failure"] for row in rows
        ),
        "median_final_q_AU": statistics.median(q_values) if q_values else None,
        "median_final_i_deg": statistics.median(i_values) if i_values else None,
        "minimum_neptune_distance_AU_sampled": min(neptune) if neptune else None,
        "minimum_p9_distance_AU_sampled": min(p9) if p9 else None,
    }


def independent_paired(source: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    source_rows = source["semantic"]["particle_diagnostics"]
    control_rows = control["semantic"]["particle_diagnostics"]
    if [row["logical_id"] for row in source_rows] != [
        row["logical_id"] for row in control_rows
    ]:
        raise ValueError("paired particle identities differ")
    q_changes: list[float] = []
    i_changes: list[float] = []
    for source_row, control_row in zip(source_rows, control_rows):
        if source_row["bound_final"] and control_row["bound_final"]:
            q_changes.append(abs(source_row["final_q_AU"] - control_row["final_q_AU"]))
            i_changes.append(abs(source_row["final_i_deg"] - control_row["final_i_deg"]))
    count = len(source_rows)
    result = {
        "source_run_key": source["semantic"]["run_key"],
        "control_run_key": control["semantic"]["run_key"],
        "paired_bound_count": len(q_changes),
        "median_absolute_final_q_change_AU": (
            statistics.median(q_changes) if q_changes else None
        ),
        "p90_absolute_final_q_change_AU": independent_percentile(q_changes, 0.9),
        "median_absolute_final_i_change_deg": (
            statistics.median(i_changes) if i_changes else None
        ),
        "p90_absolute_final_i_change_deg": independent_percentile(i_changes, 0.9),
    }
    for key in (
        "bound_final",
        "ever_q_lt_30_sampled",
        "ever_i_gt_40_sampled",
        "ever_i_gt_60_sampled",
        "ever_unbound_sampled",
    ):
        result[f"source_minus_control_{key}_fraction"] = (
            sum(row[key] for row in source_rows) - sum(row[key] for row in control_rows)
        ) / count
    return result


def independent_timestep(audit: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    audit_rows = audit["semantic"]["particle_diagnostics"]
    primary_rows = primary["semantic"]["particle_diagnostics"]
    if [row["logical_id"] for row in audit_rows] != [
        row["logical_id"] for row in primary_rows
    ]:
        raise ValueError("timestep particle identities differ")
    boolean_keys = (
        "bound_final",
        "ever_q_lt_30_sampled",
        "ever_i_gt_40_sampled",
        "ever_i_gt_60_sampled",
        "ever_unbound_sampled",
        "orbit_conversion_failure",
    )
    identities = all(
        left[key] == right[key]
        for left, right in zip(audit_rows, primary_rows)
        for key in boolean_keys
    )
    q_changes: list[float] = []
    i_changes: list[float] = []
    for left, right in zip(audit_rows, primary_rows):
        if left["bound_final"] and right["bound_final"]:
            q_changes.append(abs(left["final_q_AU"] - right["final_q_AU"]))
            i_changes.append(abs(left["final_i_deg"] - right["final_i_deg"]))
    return {
        "audit_run_key": audit["semantic"]["run_key"],
        "primary_run_key": primary["semantic"]["run_key"],
        "paired_bound_count": len(q_changes),
        "sampled_event_and_bound_identities_exact": identities,
        "maximum_absolute_final_q_difference_AU": max(q_changes) if q_changes else None,
        "maximum_absolute_final_i_difference_deg": max(i_changes) if i_changes else None,
    }


def load_and_verify_output(
    contract: dict[str, Any],
    contract_sha256: str,
    output_dir: Path,
    expected_execution_label: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, Any]]:
    expected_runtime = long.validate_runtime(contract)
    result_path = output_dir / "result_v1.json"
    if (output_dir / "failure_receipt.json").exists():
        raise ValueError(f"failure receipt exists: {output_dir}")
    result = long.strict_json(result_path)
    if set(result) != {
        "schema",
        "experiment_id",
        "verdict",
        "claim_ceiling",
        "semantic",
        "semantic_sha256",
        "provenance",
        "nonclaim",
    }:
        raise ValueError(f"result keys changed: {output_dir}")
    if result["schema"] != long.RESULT_SCHEMA:
        raise ValueError(f"result schema changed: {output_dir}")
    if result["experiment_id"] != contract["experiment_id"]:
        raise ValueError(f"experiment ID changed: {output_dir}")
    if result["verdict"] != "ENGINEERING_LONG_PROVISIONAL_VALID":
        raise ValueError(f"result is not provisionally valid: {output_dir}")
    if result["claim_ceiling"] != contract["claim_ceiling"]:
        raise ValueError(f"claim ceiling changed: {output_dir}")
    if result["nonclaim"] != contract["mandatory_nonclaim"]:
        raise ValueError(f"nonclaim changed: {output_dir}")
    manifest = long.strict_json(output_dir / "run_manifest.json")
    if set(manifest) != {
        "schema",
        "experiment_id",
        "contract_sha256",
        "runner_sha256",
        "execution_label",
        "execution_instance_id",
        "runtime",
    }:
        raise ValueError(f"run-manifest keys changed: {output_dir}")
    if manifest["schema"] != "jx-e1-long-run-manifest/v1":
        raise ValueError(f"run-manifest schema changed: {output_dir}")
    if manifest["experiment_id"] != contract["experiment_id"]:
        raise ValueError(f"run-manifest experiment changed: {output_dir}")
    if manifest["contract_sha256"] != contract_sha256:
        raise ValueError(f"run-manifest contract hash changed: {output_dir}")
    if manifest["runner_sha256"] != contract["runtime"]["runner_sha256"]:
        raise ValueError(f"run-manifest runner hash changed: {output_dir}")
    if manifest["runtime"] != expected_runtime:
        raise ValueError(f"run-manifest runtime changed: {output_dir}")
    if manifest["execution_label"] != expected_execution_label:
        raise ValueError(f"run-manifest execution label changed: {output_dir}")
    instance_id = manifest["execution_instance_id"]
    if (
        not isinstance(instance_id, str)
        or len(instance_id) != 32
        or any(character not in "0123456789abcdef" for character in instance_id)
    ):
        raise ValueError(f"run-manifest instance ID invalid: {output_dir}")
    semantic = result["semantic"]
    if set(semantic) != {
        "schema",
        "experiment_id",
        "claim_ceiling",
        "contract_sha256",
        "runtime",
        "primary_arms",
        "timestep_audit_arms",
        "paired_M1_minus_M0_diagnostics",
        "timestep_comparisons",
        "checks",
        "replay_status",
        "mandatory_nonclaim",
    }:
        raise ValueError(f"result semantic keys changed: {output_dir}")
    if result["semantic_sha256"] != long.sha256_bytes(long.canonical_bytes(semantic)):
        raise ValueError(f"result semantic hash mismatch: {output_dir}")
    if semantic.get("schema") != long.SEMANTIC_SCHEMA:
        raise ValueError(f"semantic schema changed: {output_dir}")
    if semantic.get("contract_sha256") != contract_sha256:
        raise ValueError(f"semantic contract hash mismatch: {output_dir}")
    if semantic.get("runtime") != {
        key: value for key, value in expected_runtime.items() if key != "rebound_binary_path"
    }:
        raise ValueError(f"semantic runtime changed: {output_dir}")
    if semantic.get("claim_ceiling") != contract["claim_ceiling"]:
        raise ValueError(f"semantic claim ceiling changed: {output_dir}")
    if semantic.get("mandatory_nonclaim") != contract["mandatory_nonclaim"]:
        raise ValueError(f"semantic nonclaim changed: {output_dir}")
    if semantic.get("replay_status") != "PENDING_SEPARATE_CLEAN_EXECUTION_AND_LOCKED_VERIFIER":
        raise ValueError(f"unexpected pre-verification replay status: {output_dir}")
    checks = semantic["checks"]
    if set(checks) != EXPECTED_RESULT_CHECK_KEYS:
        raise ValueError(f"result check keys changed: {output_dir}")
    if any(value is not True for value in checks.values()):
        raise ValueError(f"result checks are not all true: {output_dir}")
    if set(result["provenance"]) != {
        "execution_label",
        "execution_instance_id",
        "cumulative_elapsed_seconds",
        "peak_rss_bytes",
        "free_disk_bytes_before_result",
        "output_directory",
        "arm_records",
    }:
        raise ValueError(f"result provenance keys changed: {output_dir}")
    if result["provenance"]["execution_label"] != expected_execution_label:
        raise ValueError(f"result execution label changed: {output_dir}")
    if result["provenance"]["execution_instance_id"] != instance_id:
        raise ValueError(f"result execution instance changed: {output_dir}")
    if Path(result["provenance"]["output_directory"]).resolve() != output_dir.resolve():
        raise ValueError(f"result output directory does not match actual directory: {output_dir}")
    progress = long.strict_json(output_dir / "progress.json")
    if set(progress) != {
        "schema",
        "contract_sha256",
        "cumulative_elapsed_seconds",
        "active_attempt",
    } or progress.get("schema") != long.PROGRESS_SCHEMA:
        raise ValueError(f"progress schema changed: {output_dir}")
    if progress.get("active_attempt") is not None:
        raise ValueError(f"active progress attempt remains: {output_dir}")
    if progress.get("contract_sha256") != contract_sha256:
        raise ValueError(f"progress contract hash mismatch: {output_dir}")
    if type(progress.get("cumulative_elapsed_seconds")) not in (int, float):
        raise ValueError(f"progress elapsed type changed: {output_dir}")
    cumulative_elapsed = float(progress["cumulative_elapsed_seconds"])
    if (
        not math.isfinite(cumulative_elapsed)
        or cumulative_elapsed < 0.0
        or cumulative_elapsed > float(contract["resource_caps"]["max_wall_seconds_total"])
    ):
        raise ValueError(f"progress elapsed violates cap: {output_dir}")
    if result["provenance"].get("cumulative_elapsed_seconds") != progress.get(
        "cumulative_elapsed_seconds"
    ):
        raise ValueError(f"result/progress elapsed mismatch: {output_dir}")
    if long.output_bytes(output_dir) > int(contract["resource_caps"]["max_output_bytes"]):
        raise ValueError(f"output-size cap exceeded: {output_dir}")
    if (
        type(result["provenance"]["peak_rss_bytes"]) is not int
        or result["provenance"]["peak_rss_bytes"] < 0
        or result["provenance"]["peak_rss_bytes"]
        > int(contract["resource_caps"]["max_peak_rss_bytes"])
    ):
        raise ValueError(f"result peak RSS violates cap: {output_dir}")
    if (
        type(result["provenance"]["free_disk_bytes_before_result"]) is not int
        or result["provenance"]["free_disk_bytes_before_result"]
        < int(contract["resource_caps"]["minimum_free_disk_bytes"])
    ):
        raise ValueError(f"result free-disk floor violated: {output_dir}")

    primary_specs, audit_specs = long.build_matrix_specs(contract)
    records: dict[str, dict[str, Any]] = {}
    for spec in primary_specs + audit_specs:
        record = long.load_completed_record(
            long.record_path(output_dir, spec["run_key"]),
            output_dir,
            contract,
            contract_sha256,
            spec["run_key"],
            spec["arm_class"],
            spec["block"],
            spec["model"],
            spec["angle"],
            spec["dt_years"],
            spec["primary_run_key"],
        )
        if record is None:
            raise ValueError(f"missing arm record: {spec['run_key']}")
        records[spec["run_key"]] = record
    expected_primary = [
        records[spec["run_key"]]["semantic"]
        for spec in sorted(primary_specs, key=lambda item: item["run_key"])
    ]
    expected_audits = [
        records[spec["run_key"]]["semantic"]
        for spec in sorted(audit_specs, key=lambda item: item["run_key"])
    ]
    if semantic["primary_arms"] != expected_primary:
        raise ValueError(f"result primary-arm embedding mismatch: {output_dir}")
    if semantic["timestep_audit_arms"] != expected_audits:
        raise ValueError(f"result audit-arm embedding mismatch: {output_dir}")
    gates = contract["gates"]
    for key, record in records.items():
        arm_semantic = record["semantic"]
        rows = arm_semantic["particle_diagnostics"]
        if arm_semantic["summary"] != independent_summary(rows):
            raise ValueError(f"arm summary does not recompute: {key}")
        if any(row["orbit_conversion_failure"] for row in rows):
            raise ValueError(f"arm has orbit-conversion failure: {key}")
        for field, gate in (
            (
                "maximum_relative_active_energy_drift",
                "max_relative_active_energy_drift",
            ),
            (
                "maximum_relative_active_angular_momentum_vector_drift",
                "max_relative_active_angular_momentum_vector_drift",
            ),
            (
                "maximum_relative_active_linear_momentum_vector_drift",
                "max_relative_active_linear_momentum_vector_drift",
            ),
        ):
            value = arm_semantic[field]
            if (
                type(value) not in (int, float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                or float(value) > float(gates[gate])
            ):
                raise ValueError(f"arm drift violates gate for {field}: {key}")

    primary_records = {
        spec["run_key"]: records[spec["run_key"]] for spec in primary_specs
    }
    audit_records = {
        spec["run_key"]: records[spec["run_key"]] for spec in audit_specs
    }
    controls = {
        block: primary_records[f"M0-b{block:02d}"]
        for block in range(int(contract["tracer_design"]["blocks"]))
    }
    recomputed_paired = [
        independent_paired(record, controls[record["semantic"]["block"]])
        for key, record in sorted(primary_records.items())
        if record["semantic"]["model_id"] is not None
    ]
    recomputed_timestep = [
        independent_timestep(
            audit,
            primary_records[audit["semantic"]["primary_run_key"]],
        )
        for key, audit in sorted(audit_records.items())
    ]
    if semantic["paired_M1_minus_M0_diagnostics"] != recomputed_paired:
        raise ValueError(f"paired diagnostics do not recompute: {output_dir}")
    if semantic["timestep_comparisons"] != recomputed_timestep:
        raise ValueError(f"timestep comparisons do not recompute: {output_dir}")
    timestep_q_differences = [
        item["maximum_absolute_final_q_difference_AU"]
        for item in recomputed_timestep
        if item["maximum_absolute_final_q_difference_AU"] is not None
    ]
    timestep_i_differences = [
        item["maximum_absolute_final_i_difference_deg"]
        for item in recomputed_timestep
        if item["maximum_absolute_final_i_difference_deg"] is not None
    ]
    recomputed_checks = {
        "complete_primary_matrix": len(primary_records) == 74,
        "complete_timestep_audit_matrix": len(audit_records) == 16,
        "paired_diagnostic_count_exact": len(recomputed_paired) == 72,
        "timestep_comparison_count_exact": len(recomputed_timestep) == 16,
        "all_arm_checks_true": all(
            set(record["semantic"]["checks"]) == long.ARM_CHECK_KEYS
            and all(value is True for value in record["semantic"]["checks"].values())
            for record in records.values()
        ),
        "all_M1_common_initial_states_match_block_M0": all(
            record["semantic"]["initial_common_state_sha256"]
            == controls[record["semantic"]["block"]]["semantic"][
                "initial_common_state_sha256"
            ]
            for record in primary_records.values()
            if record["semantic"]["model_id"] is not None
        ),
        "all_timestep_audit_common_initial_states_match_primary": all(
            audit["semantic"]["initial_common_state_sha256"]
            == primary_records[audit["semantic"]["primary_run_key"]]["semantic"][
                "initial_common_state_sha256"
            ]
            for audit in audit_records.values()
        ),
        "timestep_bound_and_sampled_event_identities_exact": all(
            item["sampled_event_and_bound_identities_exact"]
            for item in recomputed_timestep
        ),
        "timestep_minimum_paired_bound_count_met": min(
            item["paired_bound_count"] for item in recomputed_timestep
        ) >= int(gates["minimum_dt_half_paired_bound_count"]),
        "timestep_max_final_q_difference_within_gate": bool(timestep_q_differences)
        and max(timestep_q_differences)
        <= float(gates["max_dt_half_final_q_difference_AU"]),
        "timestep_max_final_i_difference_within_gate": bool(timestep_i_differences)
        and max(timestep_i_differences)
        <= float(gates["max_dt_half_final_i_difference_deg"]),
        "persistent_cumulative_wall_time_within_cap": cumulative_elapsed
        <= float(contract["resource_caps"]["max_wall_seconds_total"]),
        "final_peak_rss_within_cap": result["provenance"]["peak_rss_bytes"]
        <= int(contract["resource_caps"]["max_peak_rss_bytes"]),
        "final_free_disk_floor_met": result["provenance"][
            "free_disk_bytes_before_result"
        ] >= int(contract["resource_caps"]["minimum_free_disk_bytes"]),
        "output_size_within_cap_before_result": long.output_bytes(output_dir)
        <= int(contract["resource_caps"]["max_output_bytes"]),
        "no_failure_receipt_present": not (output_dir / "failure_receipt.json").exists(),
    }
    if semantic["checks"] != recomputed_checks:
        raise ValueError(f"result checks do not recompute: {output_dir}")
    expected_arm_provenance = {
        key: record["provenance"] for key, record in sorted(records.items())
    }
    if result["provenance"]["arm_records"] != expected_arm_provenance:
        raise ValueError(f"result arm provenance embedding mismatch: {output_dir}")
    return result, records, manifest


def verify(
    contract_path: Path,
    output_a: Path,
    output_b: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    if output_a.resolve() == output_b.resolve():
        raise ValueError("independent replay requires two different output directories")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite replay receipt: {receipt_path}")
    contract = long.strict_json(contract_path)
    long.validate_contract(contract, contract_path)
    long.validate_runtime(contract)
    if contract["runtime"]["verifier_sha256"] != long.sha256_file(Path(__file__).resolve()):
        raise ValueError("verifier hash mismatch")
    contract_sha256 = long.sha256_file(contract_path)
    labels = contract["replay_policy"]["clean_execution_labels"]
    result_a, records_a, manifest_a = load_and_verify_output(
        contract, contract_sha256, output_a, labels[0]
    )
    result_b, records_b, manifest_b = load_and_verify_output(
        contract, contract_sha256, output_b, labels[1]
    )
    if manifest_a["execution_instance_id"] == manifest_b["execution_instance_id"]:
        raise ValueError("independent executions reused the same instance ID")
    if result_a["semantic"] != result_b["semantic"]:
        raise ValueError("independent result semantics differ")
    if set(records_a) != set(records_b):
        raise ValueError("independent arm-record key sets differ")
    if any(
        records_a[key]["semantic"] != records_b[key]["semantic"]
        for key in records_a
    ):
        raise ValueError("independent arm-record semantics differ")
    checkpoint_count = sum(
        len(records_a[key]["provenance"]["checkpoint_containers"])
        for key in records_a
    )
    raw_equal_count = sum(
        left["container_sha256_provenance_only"]
        == right["container_sha256_provenance_only"]
        for key in records_a
        for left, right in zip(
            records_a[key]["provenance"]["checkpoint_containers"],
            records_b[key]["provenance"]["checkpoint_containers"],
        )
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "verdict": REPLAY_VERDICT,
        "claim_ceiling": contract["claim_ceiling"],
        "contract_sha256": contract_sha256,
        "runner_sha256": contract["runtime"]["runner_sha256"],
        "verifier_sha256": contract["runtime"]["verifier_sha256"],
        "result_a_sha256": long.sha256_file(output_a / "result_v1.json"),
        "result_b_sha256": long.sha256_file(output_b / "result_v1.json"),
        "execution_a_label": labels[0],
        "execution_b_label": labels[1],
        "execution_a_instance_id": manifest_a["execution_instance_id"],
        "execution_b_instance_id": manifest_b["execution_instance_id"],
        "semantic_sha256": result_a["semantic_sha256"],
        "arm_record_count": len(records_a),
        "checkpoint_count_per_execution": checkpoint_count,
        "decoded_semantics_exact": True,
        "raw_checkpoint_container_equal_count_provenance_only": raw_equal_count,
        "raw_checkpoint_container_comparison_affects_verdict": False,
        "nonclaim": contract["mandatory_nonclaim"],
    }
    long.atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    arguments = parser.parse_args()
    receipt = verify(
        arguments.contract.resolve(),
        arguments.output_a.resolve(),
        arguments.output_b.resolve(),
        arguments.receipt.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
