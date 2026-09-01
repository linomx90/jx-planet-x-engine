#!/usr/bin/env python3
"""Reference-resolution v2 for the immutable JX Benchmark A candidates.

The only scientific change from Benchmark A v1 is a tighter independent
DOP853 loose/tight pair. Candidate code, workload, timesteps, cost model,
metrics, and verdict rules are imported unchanged from the preserved v1
benchmark at commit 847433b819836cf71cf55f2b5fac9f7c566a4243.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


IMMUTABLE_CANDIDATE_COMMIT = "847433b819836cf71cf55f2b5fac9f7c566a4243"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_base_module(root: Path):
    path = root / "benchmarks/jx_benchmark_a.py"
    specification = importlib.util.spec_from_file_location("jx_benchmark_a_v1", path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not import immutable Benchmark A implementation")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def assert_immutable(root: Path, relative_path: str) -> dict[str, str]:
    current = (root / relative_path).read_bytes()
    preserved = subprocess.check_output(
        ["git", "show", f"{IMMUTABLE_CANDIDATE_COMMIT}:{relative_path}"], cwd=root
    )
    current_hash = sha256_bytes(current)
    preserved_hash = sha256_bytes(preserved)
    if current_hash != preserved_hash:
        raise RuntimeError(
            f"BLOCKED: immutable Benchmark A file changed: {relative_path}; "
            f"expected {preserved_hash}, observed {current_hash}"
        )
    return {
        "path": relative_path,
        "sha256": current_hash,
        "preserved_commit": IMMUTABLE_CANDIDATE_COMMIT,
    }


def improvement(candidate: dict, rebound: dict) -> dict[str, float]:
    candidate_error = candidate["trajectory_error_vs_tight_dop853"]["all"]
    rebound_error = rebound["trajectory_error_vs_tight_dop853"]["all"]
    return {
        "max_position_error_factor": rebound_error["max_pos"] / candidate_error["max_pos"],
        "rms_position_error_factor": rebound_error["rms_pos"] / candidate_error["rms_pos"],
        "max_velocity_error_factor": rebound_error["max_vel"] / candidate_error["max_vel"],
        "rms_velocity_error_factor": rebound_error["rms_vel"] / candidate_error["rms_vel"],
        "max_energy_error_factor": rebound["max_abs_relative_energy_error"]
        / candidate["max_abs_relative_energy_error"],
        "wall_time_ratio_candidate_over_rebound": candidate["wall_seconds"]
        / rebound["wall_seconds"],
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = root / "runs/jx_benchmark_a_reference_v2"
    output.mkdir(parents=True, exist_ok=True)

    immutable_files = [
        assert_immutable(root, "benchmarks/jx_benchmark_a.py"),
        assert_immutable(root, "benchmarks/jx_benchmark_a_contract.json"),
    ]
    base = load_base_module(root)

    base_contract_path = root / "benchmarks/jx_benchmark_a_contract.json"
    v2_contract_path = root / "benchmarks/jx_benchmark_a_reference_v2_contract.json"
    reference_path = root / "runs/de441_horizons_10yr/reference/horizons_de441_vectors.csv"
    gm_path = root / "runs/de441_horizons_10yr/gm_de440_major_barycenters.csv"
    base_contract = json.loads(base_contract_path.read_text(encoding="utf-8"))
    v2_contract = json.loads(v2_contract_path.read_text(encoding="utf-8"))
    state = base.load(reference_path, gm_path)
    duration = float(base_contract["workload"]["duration_years"]) * 365.25
    output_interval = float(base_contract["workload"]["output_interval_days"])

    loose_spec = v2_contract["reference_pair"]["loose"]
    tight_spec = v2_contract["reference_pair"]["tight"]
    loose = base.dop(
        "dop853_v2_loose",
        state,
        duration,
        output_interval,
        float(loose_spec["rtol"]),
        float(loose_spec["atol"]),
        float(loose_spec["max_step_days"]),
    )
    tight = base.dop(
        "dop853_v2_tight",
        state,
        duration,
        output_interval,
        float(tight_spec["rtol"]),
        float(tight_spec["atol"]),
        float(tight_spec["max_step_days"]),
    )
    base.write_traj(output / "trajectory_dop853_v2_loose.csv", loose, state)
    base.write_traj(output / "trajectory_dop853_v2_tight.csv", tight, state)

    equal_timestep_dt = float(base_contract["contests"]["equal_timestep"]["dt_days"])
    runs = {
        "equal_timestep": [
            base.jx(lane, "equal_timestep", state, duration, output_interval, equal_timestep_dt)
            for lane in ("kdk", "y6_cached", "bm6")
        ]
        + [base.reb("equal_timestep", state, duration, output_interval, equal_timestep_dt)]
    }

    budget = base_contract["contests"]["equal_force_budget"]
    steps_per_year = {
        "kdk": int(budget["kdk_steps_per_year"]),
        "y6_cached": int(budget["y6_steps_per_year"]),
        "bm6": int(budget["bm6_steps_per_year"]),
        "rebound_leapfrog_5.1.1": int(budget["rebound_steps_per_year"]),
    }
    runs["equal_force_budget"] = [
        base.jx(
            lane,
            "equal_force_budget",
            state,
            duration,
            output_interval,
            365.25 / steps_per_year[lane],
        )
        for lane in ("kdk", "y6_cached", "bm6")
    ] + [
        base.reb(
            "equal_force_budget",
            state,
            duration,
            output_interval,
            365.25 / steps_per_year["rebound_leapfrog_5.1.1"],
        )
    ]

    contests = {}
    summary_rows = []
    for contest_name, contest_runs in runs.items():
        summaries = [base.summary(run, tight, state) for run in contest_runs]
        reference_gate = base.ref_gate(loose, tight, summaries, state)
        lane_map = {record["lane"]: record for record in summaries}
        contests[contest_name] = {
            "verdict": base.verdict(summaries, reference_gate),
            "reference_gate": reference_gate,
            "summaries": summaries,
            "bm6_vs_rebound_improvement": improvement(
                lane_map["bm6"], lane_map["rebound_leapfrog_5.1.1"]
            ),
            "y6_vs_rebound_improvement": improvement(
                lane_map["y6_cached"], lane_map["rebound_leapfrog_5.1.1"]
            ),
        }
        for run in contest_runs:
            base.write_traj(
                output / f"trajectory_{contest_name}_{run['lane']}.csv", run, state
            )
        for record in summaries:
            metrics = record["trajectory_error_vs_tight_dop853"]
            summary_rows.append(
                {
                    "contest": contest_name,
                    "lane": record["lane"],
                    "dt_days": record["dt_days"],
                    "steps": record["steps"],
                    "force_evaluations": record["force_evaluations"],
                    "wall_seconds": record["wall_seconds"],
                    "max_abs_relative_energy_error": record[
                        "max_abs_relative_energy_error"
                    ],
                    "max_relative_angular_momentum_vector_error": record[
                        "max_relative_angular_momentum_vector_error"
                    ],
                    **{f"all_{key}": value for key, value in metrics["all"].items()},
                    **{
                        f"outer_{key}": value
                        for key, value in metrics["outer"].items()
                    },
                }
            )

    result = {
        "schema": "jx-general-dynamics-benchmark-a-result/v2",
        "classification": "MODEL_OUTPUT_NUMERICAL_ENGINEERING_ONLY",
        "authorized_change": v2_contract["only_authorized_change"],
        "immutable_candidate_files": immutable_files,
        "base_contract_sha256": sha256_file(base_contract_path),
        "reference_v2_contract_sha256": sha256_file(v2_contract_path),
        "reference_vectors_sha256": sha256_file(reference_path),
        "gm_table_sha256": sha256_file(gm_path),
        "normalized_initial_state_sha256": state["state_hash"],
        "initial_epoch_jd_tdb": state["epoch"],
        "preserved_v1": v2_contract["preserve_v1"],
        "reference": {
            "loose": {
                "wall_seconds": loose["wall"],
                "force_evaluations": loose["calls"],
                "max_abs_relative_energy_error": float(max(abs(loose["energy"]))),
                "configuration": loose["extra"],
            },
            "tight": {
                "wall_seconds": tight["wall"],
                "force_evaluations": tight["calls"],
                "max_abs_relative_energy_error": float(max(abs(tight["energy"]))),
                "configuration": tight["extra"],
            },
            "loose_vs_tight": base.metric(loose, tight, state),
        },
        "contests": contests,
        "primary_contest": "equal_force_budget",
        "primary_verdict": contests["equal_force_budget"]["verdict"],
        "equal_timestep_verdict": contests["equal_timestep"]["verdict"],
        "environment": {
            "python": sys.version,
            "platform": base.platform.platform(),
            "numpy": base.np.__version__,
            "scipy": base.__import__("scipy").__version__
            if hasattr(base, "__import__")
            else __import__("scipy").__version__,
            "rebound_required": base.REQ_REBOUND,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
        },
        "nonclaim": v2_contract["nonclaim"],
    }

    # Replace the defensive import expression with the direct imported package version.
    result["environment"]["scipy"] = __import__("scipy").__version__

    result_path = output / "result_v2.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output / "summary_v2.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (output / "base_contract.lock.json").write_text(
        base_contract_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "reference_v2_contract.lock.json").write_text(
        v2_contract_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# JX Benchmark A — reference-resolution v2\n\n"
        f"Primary equal-force verdict: **{result['primary_verdict']}**\n\n"
        f"Equal-timestep verdict: **{result['equal_timestep_verdict']}**\n\n"
        "Candidate code, workload, costs, metrics, and win rules are immutable from v1. "
        "Only the independent DOP853 reference resolution was increased.\n",
        encoding="utf-8",
    )
    checksums = []
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.name}")
    (output / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "primary_verdict": result["primary_verdict"],
                "equal_timestep_verdict": result["equal_timestep_verdict"],
                "result_sha256": sha256_file(result_path),
            },
            indent=2,
        )
    )
    return 0 if all(
        contest["verdict"] not in {"INVALID_REFERENCE", "BLOCKED"}
        for contest in contests.values()
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
