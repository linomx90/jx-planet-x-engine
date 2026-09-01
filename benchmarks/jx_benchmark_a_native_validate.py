#!/usr/bin/env python3
"""Run the locked native C++ BM6 replay for JX Benchmark A."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import jx_benchmark_a_native_support as support

STATE_FIELDS = ("x", "y", "z", "vx", "vy", "vz")


def exact_golden_replay_gate(
    native_trajectory_path: Path, golden_manifest_path: Path
) -> dict[str, object]:
    manifest = json.loads(golden_manifest_path.read_text(encoding="utf-8"))
    expected_snapshots = {
        int(record["output_index"]): record for record in manifest["snapshots"]
    }
    with native_trajectory_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        step = int(row["step"])
        if step % support.BM6_OUTPUT_EVERY != 0:
            raise RuntimeError(f"native row is off the output grid: step {step}")
        output_index = step // support.BM6_OUTPUT_EVERY
        grouped.setdefault(output_index, []).append(row)

    if tuple(sorted(grouped)) != tuple(sorted(expected_snapshots)):
        raise RuntimeError("golden/native output-index roster mismatch")

    overall = hashlib.sha256()
    observed_snapshot_hashes: dict[str, str] = {}
    snapshot_matches: dict[str, bool] = {}
    energy_delta_max = 0.0
    angular_delta_max = 0.0
    for output_index in sorted(grouped):
        snapshot_rows = sorted(grouped[output_index], key=lambda row: int(row["body_id"]))
        if tuple(int(row["body_id"]) for row in snapshot_rows) != support.EXPECTED_BODY_IDS:
            raise RuntimeError(f"golden/native body roster mismatch at output {output_index}")
        snapshot_hash = hashlib.sha256()
        for row in snapshot_rows:
            canonical_line = (
                "|".join(
                    [str(output_index), row["body_id"]]
                    + [row[field] for field in STATE_FIELDS]
                )
                + "\n"
            ).encode("utf-8")
            snapshot_hash.update(canonical_line)
            overall.update(canonical_line)

        observed = snapshot_hash.hexdigest()
        expected = expected_snapshots[output_index]["state_sha256"]
        observed_snapshot_hashes[str(output_index)] = observed
        snapshot_matches[str(output_index)] = observed == expected

        diagnostic_row = snapshot_rows[0]
        energy_delta_max = max(
            energy_delta_max,
            abs(
                float(diagnostic_row["signed_relative_energy_error"])
                - float(expected_snapshots[output_index]["signed_relative_energy_error"])
            ),
        )
        angular_delta_max = max(
            angular_delta_max,
            abs(
                float(diagnostic_row["relative_angular_momentum_vector_error"])
                - float(
                    expected_snapshots[output_index][
                        "relative_angular_momentum_vector_error"
                    ]
                )
            ),
        )

    observed_overall = overall.hexdigest()
    expected_overall = manifest["overall_state_sha256"]
    energy_limit = float(
        manifest["diagnostic_tolerances"]["energy_series_max_abs_delta"]
    )
    angular_limit = float(
        manifest["diagnostic_tolerances"]["angular_series_max_abs_delta"]
    )
    state_passed = all(snapshot_matches.values()) and observed_overall == expected_overall
    passed = (
        state_passed
        and energy_delta_max <= energy_limit
        and angular_delta_max <= angular_limit
    )
    return {
        "passed": passed,
        "exact_state_passed": state_passed,
        "overall_state_sha256_expected": expected_overall,
        "overall_state_sha256_observed": observed_overall,
        "snapshot_hash_matches": snapshot_matches,
        "snapshot_hashes_observed": observed_snapshot_hashes,
        "energy_series_max_abs_delta": energy_delta_max,
        "energy_series_max_abs_delta_limit": energy_limit,
        "angular_series_max_abs_delta": angular_delta_max,
        "angular_series_max_abs_delta_limit": angular_limit,
        "golden_manifest_sha256": support.sha256(golden_manifest_path),
        "oracle_source": manifest["source"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument(
        "--output", default="runs/jx_benchmark_a_native_cpp",
        help="artifact output directory",
    )
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    executable = Path(arguments.executable).resolve()
    output = (root / arguments.output).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    state_path = root / "runs/de441_horizons_10yr/reference/horizons_de441_vectors.csv"
    gm_path = root / "runs/de441_horizons_10yr/gm_de440_major_barycenters.csv"
    source_paths = [
        root / "native/jx_bm6_types.hpp",
        root / "native/jx_bm6_integrator.hpp",
        root / "native/jx_bm6_io.hpp",
        root / "native/jx_bm6_native.cpp",
    ]
    contract_path = root / "benchmarks/jx_benchmark_a_contract.json"
    native_contract_path = root / "benchmarks/jx_benchmark_a_native_cpp_contract.json"
    oracle_contract_path = (
        root / "benchmarks/jx_benchmark_a_native_replay_oracle_v2_contract.json"
    )
    golden_manifest_path = (
        root / "benchmarks/jx_benchmark_a_bm6_golden_state_v1.json"
    )
    required_paths = [
        executable,
        state_path,
        gm_path,
        contract_path,
        native_contract_path,
        oracle_contract_path,
        golden_manifest_path,
        *source_paths,
    ]
    for path in required_paths:
        if not path.is_file():
            raise RuntimeError(f"required file not found: {path}")

    oracle_contract = json.loads(oracle_contract_path.read_text(encoding="utf-8"))
    if support.sha256(golden_manifest_path) != oracle_contract["exact_oracle"]["file_sha256"]:
        raise RuntimeError("golden replay manifest hash mismatch")

    base = support.load_base_module(root)
    state = base.load(state_path, gm_path)

    self_test_log = output / "native_self_test.log"
    support.run_command([str(executable), "--self-test"], root, self_test_log)
    self_test_passed = "SELF_TEST_PASS" in self_test_log.read_text(encoding="utf-8")

    trajectory_a = output / "trajectory_native_bm6_a.csv"
    result_a_path = output / "native_run_a.json"
    result_a = support.invoke_native(
        executable,
        root,
        state_path,
        gm_path,
        trajectory_a,
        result_a_path,
        output / "native_run_a.log",
    )
    trajectory_b = output / "trajectory_native_bm6_b.csv"
    result_b_path = output / "native_run_b.json"
    result_b = support.invoke_native(
        executable,
        root,
        state_path,
        gm_path,
        trajectory_b,
        result_b_path,
        output / "native_run_b.log",
    )
    deterministic_gate = {
        "trajectory_a_sha256": support.sha256(trajectory_a),
        "trajectory_b_sha256": support.sha256(trajectory_b),
        "byte_identical": trajectory_a.read_bytes() == trajectory_b.read_bytes(),
    }
    deterministic_gate["passed"] = deterministic_gate["byte_identical"]
    golden_replay_gate = exact_golden_replay_gate(
        trajectory_a, golden_manifest_path
    )

    native_run = support.parse_native_trajectory(trajectory_a, state)
    native_run["wall"] = float(result_a["timing_median_seconds"])
    native_run["extra"] = {
        "native_result": result_a,
        "repeat_b_timing_median_seconds": result_b["timing_median_seconds"],
    }
    python_bm6 = base.jx(
        "bm6",
        "equal_force_budget",
        state,
        10.0 * 365.25,
        365.25,
        support.BM6_DT_DAYS,
    )
    live_python_replay = support.replay_comparison(native_run, python_bm6)

    # Reuse the v1 reference settings that passed the primary contest by a
    # large margin. The unresolved equal-timestep lane is not reopened here.
    loose = base.dop(
        "dop853_native_loose",
        state,
        10.0 * 365.25,
        365.25,
        2.5e-13,
        2.5e-16,
        0.5,
    )
    tight = base.dop(
        "dop853_native_tight",
        state,
        10.0 * 365.25,
        365.25,
        2.5e-14,
        2.5e-17,
        0.25,
    )
    rebound_run = base.reb(
        "equal_force_budget",
        state,
        10.0 * 365.25,
        365.25,
        support.REBOUND_DT_DAYS,
    )
    native_summary = base.summary(native_run, tight, state)
    rebound_summary = base.summary(rebound_run, tight, state)
    if native_summary["force_evaluations"] != rebound_summary["force_evaluations"]:
        raise RuntimeError("primary force budgets do not match")
    reference_gate = base.ref_gate(loose, tight, [native_summary, rebound_summary], state)
    accuracy_win = bool(base.beats(native_summary, rebound_summary))

    native_max_position_error = native_summary[
        "trajectory_error_vs_tight_dop853"
    ]["all"]["max_pos"]
    live_python_replay["hard_gate"] = False
    live_python_replay["state_max_delta_over_native_max_position_error"] = (
        live_python_replay["state_max_abs_component_delta"]
        / native_max_position_error
    )
    live_python_replay["interpretation"] = (
        "Informational processor/library-dispatch diagnostic; exact hard replay "
        "is adjudicated by the frozen v1 golden SHA-256 oracle."
    )

    rebound_timing = support.rebound_native_timing(state, support.TIMING_REPEATS)
    native_median = float(result_a["timing_median_seconds"])
    rebound_median = float(rebound_timing["median_seconds"])
    speed_winner = (
        "JX_NATIVE_BM6" if native_median < rebound_median else "REBOUND_5.1.1"
    )
    speed = {
        "native_bm6_median_seconds": native_median,
        "rebound_5_1_1_median_seconds": rebound_median,
        "native_over_rebound_time_ratio": native_median / rebound_median,
        "native_speedup_over_rebound": rebound_median / native_median,
        "rebound_speedup_over_native": native_median / rebound_median,
        "winner": speed_winner,
        "native_details": {
            "repeats": result_a["timing_repeats"],
            "steps": support.BM6_STEPS,
            "dt_days": support.BM6_DT_DAYS,
            "measured_force_evaluations": 10 * support.BM6_STEPS,
            "min_seconds": result_a["timing_min_seconds"],
            "max_seconds": result_a["timing_max_seconds"],
            "timing_semantics": (
                "native C++ integration only; setup, diagnostics, and I/O excluded"
            ),
        },
        "rebound_details": rebound_timing,
    }
    native_summary["integration_only_median_seconds"] = native_median
    rebound_summary["integration_only_median_seconds"] = rebound_median

    if not self_test_passed:
        verdict = "INVALID_NATIVE_SELF_TEST"
    elif not deterministic_gate["passed"]:
        verdict = "INVALID_NATIVE_NONDETERMINISTIC"
    elif not golden_replay_gate["passed"]:
        verdict = "INVALID_NATIVE_REPLAY"
    elif not reference_gate["passed"]:
        verdict = "INVALID_REFERENCE"
    elif accuracy_win:
        verdict = "JX_NATIVE_BM6_WIN"
    else:
        verdict = "NO_NATIVE_WIN"

    result = {
        "schema": "jx-benchmark-a-native-cpp-result/v2",
        "classification": "MODEL_OUTPUT_NUMERICAL_ENGINEERING_ONLY",
        "primary_contest": "equal_force_budget",
        "primary_verdict": verdict,
        "scope": "frozen ten-body Newtonian DE441/Horizons workload over ten years",
        "native_implementation": {
            "language": "C++20",
            "arithmetic": "IEEE-754 binary64",
            "source_files": {
                str(path.relative_to(root)): support.sha256(path) for path in source_paths
            },
            "executable_sha256": support.sha256(executable),
            "strict_fp_contract": "-fno-fast-math -ffp-contract=off",
        },
        "inputs": {
            "base_contract_sha256": support.sha256(contract_path),
            "native_contract_sha256": support.sha256(native_contract_path),
            "replay_oracle_amendment_sha256": support.sha256(oracle_contract_path),
            "golden_replay_manifest_sha256": support.sha256(golden_manifest_path),
            "reference_vectors_sha256": support.sha256(state_path),
            "gm_table_sha256": support.sha256(gm_path),
            "normalized_initial_state_sha256": state["state_hash"],
            "initial_epoch_jd_tdb": state["epoch"],
        },
        "gates": {
            "native_self_test": {
                "passed": self_test_passed,
                "log_sha256": support.sha256(self_test_log),
            },
            "deterministic_native_replay": deterministic_gate,
            "exact_preserved_bm6_replay": golden_replay_gate,
            "independent_reference": reference_gate,
            "accuracy_win_rule": {
                "passed": accuracy_win,
                "rule": (
                    "all four all-body trajectory metrics lower than REBOUND "
                    "and energy/angular gates not worse"
                ),
            },
        },
        "diagnostics": {
            "live_same_run_python_bm6_replay": live_python_replay,
            "preserved_failed_strict_run": oracle_contract[
                "preserved_failed_strict_run"
            ],
        },
        "summaries": {
            "native_bm6": native_summary,
            "rebound_leapfrog_5_1_1": rebound_summary,
        },
        "speed_comparison": speed,
        "reference": {
            "loose": {
                "force_evaluations": loose["calls"],
                "wall_seconds": loose["wall"],
                "configuration": loose["extra"],
            },
            "tight": {
                "force_evaluations": tight["calls"],
                "wall_seconds": tight["wall"],
                "configuration": tight["extra"],
            },
            "loose_vs_tight": base.metric(loose, tight, state),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "rebound": __import__("rebound").__version__,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        },
        "artifacts": {
            "native_trajectory_sha256": support.sha256(trajectory_a),
            "native_run_json_sha256": support.sha256(result_a_path),
        },
        "claim_boundary": oracle_contract["claim_ceiling"],
    }

    result_path = output / "result_native_cpp.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    support.write_summary_csv(
        output / "summary_native_cpp.csv", native_summary, rebound_summary
    )
    (output / "README.md").write_text(
        "# JX Benchmark A — native C++ BM6\n\n"
        f"Primary verdict: **{verdict}**\n\n"
        f"Median integration-only speed winner: **{speed_winner}**\n\n"
        "The hard replay gate uses the exact preserved Benchmark A v1 BM6 "
        "snapshot hashes. The same-run NumPy comparison is retained as a "
        "hardware-dispatch diagnostic. See result_native_cpp.json, "
        "summary_native_cpp.csv, logs, trajectories, and SHA256SUMS.txt.\n",
        encoding="utf-8",
    )

    checksums = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{support.sha256(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "primary_verdict": verdict,
                "exact_golden_replay": golden_replay_gate,
                "live_python_replay_diagnostic": live_python_replay,
                "reference_gate_passed": reference_gate["passed"],
                "speed": speed,
                "result_sha256": support.sha256(result_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if verdict == "JX_NATIVE_BM6_WIN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
