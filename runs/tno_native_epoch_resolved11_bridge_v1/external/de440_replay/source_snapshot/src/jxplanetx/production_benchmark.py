"""Locked import and reproduction of the 2026-08-20 Yoshida-6 benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

from .decimal_math import D, precision_context

BENCHMARK_ID = "JX_ARB_YOSHIDA6_GATE_2026-08-20"
EXPECTED_ARCHIVE_SHA256 = "50a3f04007c302bc47068c1827a7fb4cead653f442609342a3b037ac9c04f9bb"
EXPECTED_CHECKSUM_MANIFEST_SHA256 = "4517fc0b236eb370ff65fbfbaaac5af6ba3710c9c8c97c4bacc59781e78964eb"

GATES: dict[str, Decimal | int] = {
    "massive_position_AU": D("1e-7"),
    "massive_velocity_AU_per_yr": D("1e-9"),
    "tracer_position_AU": D("1e-4"),
    "tracer_velocity_AU_per_yr": D("1e-6"),
    "tracer_q_AU": D("1e-3"),
    "tracer_i_deg": D("1e-3"),
    "bound_mismatches": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Verify the exact preserved manifest and every payload file."""
    root = Path(bundle_dir).resolve()
    manifest = root / "checksums.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(f"missing locked manifest: {manifest}")
    actual_manifest_hash = _sha256(manifest)
    if actual_manifest_hash != EXPECTED_CHECKSUM_MANIFEST_SHA256:
        raise ValueError("benchmark checksum manifest does not match the locked artifact")

    checked: dict[str, str] = {}
    failures: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.removeprefix("*").removeprefix("./")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"unsafe path in benchmark manifest: {relative}") from exc
        if not candidate.is_file():
            failures.append(f"missing:{relative}")
            continue
        actual = _sha256(candidate)
        checked[relative] = actual
        if actual != expected:
            failures.append(f"hash:{relative}")
    if failures:
        raise ValueError("benchmark integrity failure: " + ", ".join(failures))
    return {
        "benchmark_id": BENCHMARK_ID,
        "manifest_sha256": actual_manifest_hash,
        "files_verified": len(checked),
        "integrity_pass": True,
    }


def _load_trajectory(path: Path) -> dict[tuple[int, int], dict[str, str]]:
    rows: dict[tuple[int, int], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["time"]), int(row["body"]))
            if key in rows:
                raise ValueError(f"duplicate trajectory row {key} in {path}")
            rows[key] = row
    return rows


def serialized_state_audit(reference: str | Path, reproduced: str | Path) -> dict[str, Any]:
    """Require exact dynamics fields; report runtime-sensitive derived fields."""
    left = _load_trajectory(Path(reference))
    right = _load_trajectory(Path(reproduced))
    if set(left) != set(right):
        raise ValueError("serialized trajectory support mismatch")
    exact_fields = ("time", "body", "name", "x", "y", "z", "vx", "vy", "vz", "a", "e", "q", "bound")
    mismatch_counts = {field: 0 for field in exact_fields + ("i_deg",)}
    for key in sorted(left):
        for field in mismatch_counts:
            mismatch_counts[field] += left[key][field] != right[key][field]
    exact_pass = all(mismatch_counts[field] == 0 for field in exact_fields)
    return {
        "exact_fields": list(exact_fields),
        "exact_fields_passed": exact_pass,
        "mismatch_counts": {field: count for field, count in mismatch_counts.items() if count},
        "note": "i_deg uses a transcendental acos evaluation and is judged numerically, not bytewise",
    }


def _norm(values: list[Decimal]) -> Decimal:
    return sum((value * value for value in values), D(0)).sqrt()


def compare_trajectories(path_a: str | Path, path_b: str | Path, label: str) -> dict[str, Any]:
    with localcontext(precision_context(100)):
        a = _load_trajectory(Path(path_a))
        b = _load_trajectory(Path(path_b))
        if set(a) != set(b):
            raise ValueError(f"trajectory support mismatch for {label}")
        maxima: dict[str, tuple[Decimal, tuple[int, int], str] | None] = {
            name: None for name in GATES if name != "bound_mismatches"
        }
        bound_mismatches = 0
        for key in sorted(a):
            left, right = a[key], b[key]
            massive = key[1] < 5
            position = _norm([D(left[k]) - D(right[k]) for k in ("x", "y", "z")])
            velocity = _norm([D(left[k]) - D(right[k]) for k in ("vx", "vy", "vz")])
            position_key = "massive_position_AU" if massive else "tracer_position_AU"
            velocity_key = "massive_velocity_AU_per_yr" if massive else "tracer_velocity_AU_per_yr"
            candidates = [(position_key, position), (velocity_key, velocity)]
            if not massive:
                candidates.extend(
                    [
                        ("tracer_q_AU", abs(D(left["q"]) - D(right["q"]))),
                        ("tracer_i_deg", abs(D(left["i_deg"]) - D(right["i_deg"]))),
                    ]
                )
            for metric, value in candidates:
                prior = maxima[metric]
                if prior is None or value > prior[0]:
                    maxima[metric] = (value, key, left["name"])
            bound_mismatches += int(left["bound"] != right["bound"])

        metrics: dict[str, Any] = {}
        for name, maximum in maxima.items():
            if maximum is None:
                raise ValueError(f"no samples for metric {name}")
            value, key, body = maximum
            threshold = GATES[name]
            assert isinstance(threshold, Decimal)
            metrics[name] = {
                "max": str(value),
                "gate": str(threshold),
                "passed": value <= threshold,
                "worst_time_years": key[0],
                "worst_body": body,
            }
        metrics["bound_mismatches"] = {
            "count": bound_mismatches,
            "gate": 0,
            "passed": bound_mismatches == 0,
        }
        return {
            "label": label,
            "rows": len(a),
            "metrics": metrics,
            "passed": all(metric["passed"] for metric in metrics.values()),
        }


def _summary_gates(summary: dict[str, Any]) -> dict[str, Any]:
    projected = D(str(summary["projected_30000yr_wall_seconds"]))
    ratio = D(summary["oscillator_error_ratio_2h_over_h"])
    energy = D(summary["relative_energy_drift"])
    angular = D(summary["relative_angular_momentum_vector_drift"])
    checks = {
        "projected_30000yr_wall_seconds": {"value": str(projected), "gate": "<= 4500", "passed": projected <= D(4500)},
        "oscillator_order_ratio": {"value": str(ratio), "gate": "60 <= ratio <= 68", "passed": D(60) <= ratio <= D(68)},
        "relative_energy_drift": {"value": str(energy), "gate": "<= 1e-9", "passed": energy <= D("1e-9")},
        "relative_angular_momentum_vector_drift": {"value": str(angular), "gate": "<= 1e-10", "passed": angular <= D("1e-10")},
        "macro_steps": {"value": summary["macro_steps"], "gate": "= 5000", "passed": summary["macro_steps"] == 5000},
        "force_calls": {"value": summary["force_calls"], "gate": "= 40000", "passed": summary["force_calls"] == 40000},
    }
    return {"checks": checks, "passed": all(check["passed"] for check in checks.values())}


def reproduce(bundle_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run exact 160/224-bit members and independently audit every locked gate."""
    integrity = verify_bundle(bundle_dir)
    root = Path(bundle_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    binary = root / "yoshida6_gate"
    if not os.access(binary, os.X_OK):
        binary.chmod(binary.stat().st_mode | 0o100)

    runs: dict[str, Any] = {}
    trajectories: dict[int, Path] = {}
    for bits in (160, 224):
        trajectory = output / f"reproduced_y6_{bits}.csv"
        summary_path = output / f"reproduced_y6_{bits}_summary.json"
        log_path = output / f"reproduced_y6_{bits}.log"
        command = [
            str(binary), str(bits), str(root / "initial_state.csv"), str(trajectory),
            str(summary_path), "100", "50",
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"{bits}-bit benchmark failed; see {log_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_trajectory = root / f"y6_run_{bits}.csv"
        serialization = serialized_state_audit(expected_trajectory, trajectory)
        preserved_comparison = compare_trajectories(
            expected_trajectory, trajectory, f"preserved Y6 {bits} vs reproduced Y6 {bits}"
        )
        runs[str(bits)] = {
            "summary": summary,
            "gates": _summary_gates(summary),
            "trajectory_sha256": _sha256(trajectory),
            "preserved_trajectory_sha256": _sha256(expected_trajectory),
            "serialized_state_audit": serialization,
            "preserved_numeric_comparison": preserved_comparison,
        }
        trajectories[bits] = trajectory

    comparisons = [
        compare_trajectories(trajectories[160], trajectories[224], "reproduced Y6 160 vs reproduced Y6 224"),
        compare_trajectories(trajectories[224], root / "reference_bs/run_224.csv", "reproduced Y6 224 vs preserved BS 224"),
    ]
    passed = (
        integrity["integrity_pass"]
        and all(
            run["gates"]["passed"]
            and run["serialized_state_audit"]["exact_fields_passed"]
            and run["preserved_numeric_comparison"]["passed"]
            for run in runs.values()
        )
        and all(comparison["passed"] for comparison in comparisons)
    )
    return {
        "benchmark": integrity,
        "classification": "MODEL_OUTPUT / numerical validation only",
        "runs": runs,
        "comparisons": comparisons,
        "all_gates_passed": passed,
        "claim_decision": "SCREENING_ONLY" if passed else "INVALID",
        "scientific_scope": "validates the numerical tool; says nothing for or against a Planet X source",
    }
