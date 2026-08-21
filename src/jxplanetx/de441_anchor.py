"""Import a transparent DE441-anchored 20-body benchmark state."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import Any

SOURCE_ARCHIVE_SHA256 = "d20caee282eae6905c1e7f826fe65a5ade6a623e7584bf700b82b00a8af101ee"
SOURCE_STATE_SHA256 = "7fc835c1a4400e6a76d1b92701a40caf18d10c9c3014d49ef34a1de41c399f3e"
NORMALIZED_STATE_SHA256 = "371fa2763ea23b6287215d088c54220426951da91e64066c0feb6d73f59f634e"
YOSHIDA6_BINARY_SHA256 = "323728aae09a47cf88c61cf1d484d76313ef0790ead7aaccf2f9363a46024442"
EPOCH_TDB_JD = "2461200.5"
TRACER_LABELS = tuple(f"t{index:03d}" for index in range(0, 45, 3))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_de441_anchor(source_csv: str | Path, output_csv: str | Path, metadata_json: str | Path) -> dict[str, Any]:
    source = Path(source_csv)
    if _sha256(source) != SOURCE_STATE_SHA256:
        raise ValueError("DE441 anchor source-state hash does not match the preserved artifact")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["family"] == "baseline" and row["phase_index"] == "0"]
    massive_names = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune")
    by_label = {row["label"]: row for row in selected}
    labels = massive_names + TRACER_LABELS
    if any(label not in by_label for label in labels):
        raise ValueError("preserved DE441 baseline does not contain the locked 20-body subset")
    chosen = [by_label[label] for label in labels]
    for row in chosen:
        if row["epoch_TDB_JD"] != EPOCH_TDB_JD or row["axes"] != "J2000 ecliptic":
            raise ValueError("epoch or axes changed inside preserved DE441 anchor")

    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz"))
        for index, row in enumerate(chosen):
            writer.writerow(
                (
                    index, row["label"], row["mass_Msun"], row["x_AU"], row["y_AU"], row["z_AU"],
                    row["vx_AU_per_year"], row["vy_AU_per_year"], row["vz_AU_per_year"],
                )
            )
    metadata = {
        "schema": "jx-de441-anchor/v1",
        "classification": "RECONSTRUCTED massive-body state plus ASSUMPTION synthetic tracer subset",
        "epoch_TDB_JD": EPOCH_TDB_JD,
        "massive_body_source": "JPL DE441 part-2 SPK; transformed ICRF/J2000 barycentric to J2000 ecliptic",
        "frame_origin": "system barycenter after move_to_com in preserved constructor",
        "units": "AU, Julian year=365.25 d, solar mass",
        "massive_bodies": list(massive_names),
        "tracers": list(TRACER_LABELS),
        "tracer_source": "synthetic anomaly-zone grid; every third preserved baseline tracer",
        "source_state_sha256": SOURCE_STATE_SHA256,
        "normalized_state_sha256": _sha256(output),
        "numerical_status": "IMPORTED_NOT_PRODUCTION_QUALIFIED",
        "blocker": "requires 160/224-bit run plus an independent high-precision reference trajectory",
        "claim_decision": "SCREENING_ONLY",
    }
    meta_path = Path(metadata_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def run_de441_precision_pair(binary_path: str | Path, state_csv: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Run the locked Yoshida-6 binary at 160/224 bits on the DE441 anchor."""
    from .production_benchmark import _summary_gates, compare_trajectories

    binary = Path(binary_path).resolve()
    state = Path(state_csv).resolve()
    if _sha256(binary) != YOSHIDA6_BINARY_SHA256:
        raise ValueError("Yoshida-6 binary is not the locked validated executable")
    if _sha256(state) != NORMALIZED_STATE_SHA256:
        raise ValueError("normalized DE441 anchor state hash changed")
    if not os.access(binary, os.X_OK):
        binary.chmod(binary.stat().st_mode | 0o100)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories: dict[int, Path] = {}
    runs: dict[str, Any] = {}
    for bits in (160, 224):
        trajectory = output / f"de441_anchor_y6_{bits}.csv"
        summary_path = output / f"de441_anchor_y6_{bits}_summary.json"
        completed = subprocess.run(
            [str(binary), str(bits), str(state), str(trajectory), str(summary_path), "100", "50"],
            check=False, capture_output=True, text=True,
        )
        (output / f"de441_anchor_y6_{bits}.log").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if completed.returncode != 0:
            raise RuntimeError(f"DE441 anchor {bits}-bit run failed")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        runs[str(bits)] = {"summary": summary, "gates": _summary_gates(summary), "trajectory_sha256": _sha256(trajectory)}
        trajectories[bits] = trajectory
    comparison = compare_trajectories(trajectories[160], trajectories[224], "DE441 anchor Y6 160 vs Y6 224")
    pair_passed = all(run["gates"]["passed"] for run in runs.values()) and comparison["passed"]
    return {
        "classification": "MODEL_OUTPUT numerical precision-pair gate",
        "state_sha256": _sha256(state),
        "runs": runs,
        "cross_precision_comparison": comparison,
        "precision_pair_passed": pair_passed,
        "production_qualified": False,
        "status": "PARTIAL_PASS_INDEPENDENT_REFERENCE_MISSING" if pair_passed else "INVALID",
        "blocker": "no independent high-precision DE441-anchor reference trajectory is preserved",
        "claim_decision": "SCREENING_ONLY",
    }


def audit_de441_independent_reference(
    pair_record_path: str | Path,
    y6_trajectory: str | Path,
    bs_trajectory: str | Path,
    bs_summary_path: str | Path,
) -> dict[str, Any]:
    """Close the numerical DE441 gate without promoting model output to evidence."""
    from .decimal_bs import validate_bs_oscillator
    from .production_benchmark import compare_trajectories

    pair_record = json.loads(Path(pair_record_path).read_text(encoding="utf-8"))
    pair_result = pair_record["payload"]["result"]
    summary = json.loads(Path(bs_summary_path).read_text(encoding="utf-8"))
    y6 = Path(y6_trajectory)
    bs = Path(bs_trajectory)
    expected_y6_hash = pair_result["runs"]["224"]["trajectory_sha256"]
    support_comparison = compare_trajectories(
        y6, bs, "DE441 anchor Y6 224 vs independent Decimal BS 78-digit"
    )
    oscillator = validate_bs_oscillator(int(summary["decimal_digits"]))
    reference_checks = {
        "duration_years": {"value": summary["years"], "gate": "= 100", "passed": summary["years"] == 100},
        "decimal_digits": {"value": summary["decimal_digits"], "gate": ">= 68", "passed": summary["decimal_digits"] >= 68},
        "wall_seconds": {"value": summary["wall_seconds"], "gate": "<= 4500", "passed": summary["wall_seconds"] <= 4500},
        "all_block_invariants": {"value": summary["all_block_invariants_passed"], "gate": "true", "passed": summary["all_block_invariants_passed"] is True},
        "massive_path_spread": {"value": summary["maximum_massive_path_spread_across_blocks"], "gate": "<= 1e-25 AU/component", "passed": Decimal(summary["maximum_massive_path_spread_across_blocks"]) <= Decimal("1e-25")},
        "y6_trajectory_identity": {"value": _sha256(y6), "gate": expected_y6_hash, "passed": _sha256(y6) == expected_y6_hash},
    }
    reference_passed = all(check["passed"] for check in reference_checks.values())
    passed = (
        pair_result["precision_pair_passed"]
        and reference_passed
        and oscillator["passed"]
        and support_comparison["passed"]
    )
    return {
        "classification": "MODEL_OUTPUT / numerical validation only",
        "state_sha256": pair_result["state_sha256"],
        "precision_pair_passed": pair_result["precision_pair_passed"],
        "independent_bs_kernel_validation": oscillator,
        "independent_reference_checks": reference_checks,
        "cross_method_comparison": support_comparison,
        "y6_trajectory_sha256": _sha256(y6),
        "bs_trajectory_sha256": _sha256(bs),
        "all_numerical_gates_passed": passed,
        "numerical_status": "PRODUCTION_NUMERICAL_GATE_PASSED" if passed else "INVALID",
        "production_observation_qualified": False,
        "claim_decision": "SCREENING_ONLY" if passed else "INVALID",
        "scientific_scope": "validates numerical propagation for this DE441 anchor; it is not evidence for a Planet X source",
    }
