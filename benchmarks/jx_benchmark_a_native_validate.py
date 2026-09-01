#!/usr/bin/env python3
"""Run the locked native C++ BM6 replay for JX Benchmark A."""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

import jx_benchmark_a_native_support as support


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
    required_paths = [
        executable, state_path, gm_path, contract_path, native_contract_path,
        *source_paths,
    ]
    for path in required_paths:
        if not path.is_file():
            raise RuntimeError(f"required file not found: {path}")

    base = support.load_base_module(root)
    state = base.load(state_path, gm_path)

    self_test_log = output / "native_self_test.log"
    support.run_command([str(executable), "--self-test"], root, self_test_log)
    self_test_passed = "SELF_TEST_PASS" in self_test_log.read_text(encoding="utf-8")

    trajectory_a = output / "trajectory_native_bm6_a.csv"
    result_a_path = output / "native_run_a.json"
    result_a = support.invoke_native(
        executable, root, state_path, gm_path, trajectory_a, result_a_path,
        output / "native_run_a.log",
    )
    trajectory_b = output / "trajectory_native_bm6_b.csv"
    result_b_path = output / "native_run_b.json"
    result_b = support.invoke_native(
        executable, root, state_path, gm_path, trajectory_b, result_b_path,
        output / "native_run_b.log",
    )
    deterministic_gate = {
        "trajectory_a_sha256": support.sha256(trajectory_a),
        "trajectory_b_sha256": support.sha256(trajectory_b),
        "byte_identical": trajectory_a.read_bytes() == trajectory_b.read_bytes(),
    }
    deterministic_gate["passed"] = deterministic_gate["byte_identical"]

    native_run = support.parse_native_trajectory(trajectory_a, state)
    native_run["wall"] = float(result_a["timing_median_seconds"])
    native_run["extra"] = {
        "native_result": result_a,
        "repeat_b_timing_median_seconds": result_b["timing_median_seconds"],
    }
    python_bm6 = base.jx(
        "bm6", "equal_force_budget", state, 10.0 * 365.25, 365.25,
        support.PRIMARY_DT_DAYS,
    )
    replay_gate = support.replay_comparison(native_run, python_bm6)

    # Reuse the v1 reference settings that passed the primary contest by a
    # large margin. The unresolved equal-timestep lane is not reopened here.
    loose = base.dop(
        "dop853_native_loose", state, 10.0 * 365.25, 365.25,
        2.5e-13, 2.5e-16, 0.5,
    )
    tight = base.dop(
        "dop853_native_tight", state, 10.0 * 365.25, 365.25,
        2.5e-14, 2.5e-17, 0.25,
    )
    rebound_run = base.reb(
        "equal_force_budget", state, 10.0 * 365.25, 365.25,
        support.PRIMARY_DT_DAYS,
    )
    native_summary = base.summary(native_run, tight, state)
    rebound_summary = base.summary(rebound_run, tight, state)
    reference_gate = base.ref_gate(
        loose, tight, [native_summary, rebound_summary], state
    )
    accuracy_win = bool(base.beats(native_summary, rebound_summary))

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
    elif not replay_gate["passed"]:
        verdict = "INVALID_NATIVE_REPLAY"
    elif not reference_gate["passed"]:
        verdict = "INVALID_REFERENCE"
    elif accuracy_win:
        verdict = "JX_NATIVE_BM6_WIN"
    else:
        verdict = "NO_NATIVE_WIN"

    result = {
        "schema": "jx-benchmark-a-native-cpp-result/v1",
        "classification": "MODEL_OUTPUT_NUMERICAL_ENGINEERING_ONLY",
        "primary_contest": "equal_force_budget",
        "primary_verdict": verdict,
        "scope": "frozen ten-body Newtonian DE441/Horizons workload over ten years",
        "native_implementation": {
            "language": "C++20",
            "arithmetic": "IEEE-754 binary64",
            "source_files": {
                str(path.relative_to(root)): support.sha256(path)
                for path in source_paths
            },
            "executable_sha256": support.sha256(executable),
            "strict_fp_contract": "-fno-fast-math -ffp-contract=off",
        },
        "inputs": {
            "base_contract_sha256": support.sha256(contract_path),
            "native_contract_sha256": support.sha256(native_contract_path),
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
            "deterministic_replay": deterministic_gate,
            "native_vs_python_bm6": replay_gate,
            "independent_reference": reference_gate,
            "accuracy_win_rule": {
                "passed": accuracy_win,
                "rule": (
                    "all four all-body trajectory metrics lower than REBOUND "
                    "and energy/angular gates not worse"
                ),
            },
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
        "claim_boundary": (
            "Scoped native BM6 replay and equal-force accuracy/speed measurement "
            "only; not a universal REBOUND, close-encounter, arbitrary-precision, "
            "full-ephemeris, or Planet-X claim."
        ),
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
        "This is a scoped equal-force-budget replay. See result_native_cpp.json, "
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

    print(json.dumps({
        "primary_verdict": verdict,
        "native_vs_python_replay": replay_gate,
        "reference_gate_passed": reference_gate["passed"],
        "speed": speed,
        "result_sha256": support.sha256(result_path),
    }, indent=2, sort_keys=True))
    return 0 if verdict == "JX_NATIVE_BM6_WIN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
