"""Import a transparent DE441-anchored 20-body benchmark state."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

SOURCE_ARCHIVE_SHA256 = "d20caee282eae6905c1e7f826fe65a5ade6a623e7584bf700b82b00a8af101ee"
SOURCE_STATE_SHA256 = "7fc835c1a4400e6a76d1b92701a40caf18d10c9c3014d49ef34a1de41c399f3e"
NORMALIZED_STATE_SHA256 = "371fa2763ea23b6287215d088c54220426951da91e64066c0feb6d73f59f634e"
YOSHIDA6_BINARY_SHA256 = "323728aae09a47cf88c61cf1d484d76313ef0790ead7aaccf2f9363a46024442"
EPOCH_TDB_JD = "2461200.5"
TRACER_LABELS = tuple(f"t{index:03d}" for index in range(0, 45, 3))
MASSIVE_NAMES = ("Sun", "Jupiter", "Saturn", "Uranus", "Neptune")
PAIR_RESULT_SCHEMA = "jx-de441-precision-pair/v2"
PAIR_RECORD_SCHEMA = "jx-planet-x-run/v1"
PAIR_COMMAND = "run-de441-anchor-gate"
Y6_TRAJECTORY_FIELDS = (
    "time", "body", "name", "x", "y", "z", "vx", "vy", "vz",
    "a", "e", "q", "i_deg", "bound",
)
Y6_SUMMARY_FIELDS = {
    "duration_years", "force_calls", "macro_steps", "method",
    "oscillator_error_2h", "oscillator_error_h",
    "oscillator_error_ratio_2h_over_h", "precision_bits",
    "projected_30000yr_wall_seconds", "relative_angular_momentum_vector_drift",
    "relative_energy_drift", "step_years", "steps_per_year", "wall_seconds",
}
BS_SUMMARY_FIELDS = {
    "method", "years", "decimal_digits", "workers", "wall_seconds",
    "tracer_blocks", "all_block_invariants_passed",
    "maximum_massive_path_spread_across_blocks", "block_summaries",
}
BS_BLOCK_SUMMARY_FIELDS = {
    "method", "decimal_digits", "rtol", "atol", "duration_years",
    "wall_seconds", "accepted_steps", "rejected_steps", "rhs_calls",
    "maximum_extrapolation_level", "relative_energy_drift",
    "relative_angular_momentum_vector_drift",
}
Y6_RUN_FIELDS = {
    "summary", "gates", "trajectory_filename", "trajectory_sha256",
    "summary_filename", "summary_sha256",
}
PAIR_RESULT_FIELDS = {
    "schema", "classification", "state_filename", "state_sha256", "yoshida6_binary_sha256",
    "runs", "cross_precision_comparison", "precision_pair_passed",
    "production_qualified", "status", "blocker", "claim_decision",
}
STATE_FIELDS = ("x", "y", "z", "vx", "vy", "vz")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{context} must be a JSON object")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{context} schema mismatch; missing={missing}, extra={extra}")
    return value


def _require_string(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{context} must be a nonempty string")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{context} must be a JSON boolean")
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        suffix = "" if minimum is None else f" >= {minimum}"
        raise ValueError(f"{context} must be an integer{suffix}")
    return value


def _require_sha256(value: Any, context: str) -> str:
    text = _require_string(value, context)
    if len(text) != 64 or text != text.lower() or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{context} must be a lowercase SHA-256 digest")
    return text


def _decimal_value(value: Any, context: str, *, nonnegative: bool = False) -> Decimal:
    if type(value) not in {str, int, float}:
        raise ValueError(f"{context} must be a finite decimal value")
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{context} must be a finite decimal value") from exc
    if not result.is_finite() or (nonnegative and result < 0):
        raise ValueError(f"{context} must be a finite{' nonnegative' if nonnegative else ''} decimal value")
    return result


def _read_json_object(path: Path, context: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"{context} is not a regular file: {path}")
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = item
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"non-finite JSON number: {token}")),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{context} is not valid UTF-8 JSON: {path}") from exc
    if type(value) is not dict:
        raise ValueError(f"{context} must contain a JSON object")
    return value


def _canonical_equal(left: Any, right: Any) -> bool:
    from .provenance import canonical_json

    try:
        return canonical_json(left) == canonical_json(right)
    except (TypeError, ValueError):
        return False


def _decimal_close(left: Decimal, right: Decimal, relative_tolerance: Decimal) -> bool:
    return abs(left - right) <= relative_tolerance * max(Decimal(1), abs(left), abs(right))


def _recompute_y6_summary_gates(summary: dict[str, Any], bits: int) -> dict[str, Any]:
    """Validate a raw Yoshida summary and recompute every derivable gate."""
    _require_exact_keys(summary, Y6_SUMMARY_FIELDS, f"Y6 {bits}-bit summary")
    if summary["method"] != "Yoshida6 fixed-step KDK composition":
        raise ValueError(f"Y6 {bits}-bit summary method identity changed")
    if _require_int(summary["precision_bits"], f"Y6 {bits}-bit precision_bits") != bits:
        raise ValueError(f"Y6 {bits}-bit summary precision identity changed")
    if _require_int(summary["duration_years"], f"Y6 {bits}-bit duration_years") != 100:
        raise ValueError(f"Y6 {bits}-bit duration must be 100 years")
    if _require_int(summary["steps_per_year"], f"Y6 {bits}-bit steps_per_year") != 50:
        raise ValueError(f"Y6 {bits}-bit steps_per_year must be 50")
    if _decimal_value(summary["step_years"], f"Y6 {bits}-bit step_years") != Decimal("0.02"):
        raise ValueError(f"Y6 {bits}-bit step must be 0.02 years")

    macro_steps = _require_int(summary["macro_steps"], f"Y6 {bits}-bit macro_steps", minimum=0)
    force_calls = _require_int(summary["force_calls"], f"Y6 {bits}-bit force_calls", minimum=0)
    wall = _decimal_value(summary["wall_seconds"], f"Y6 {bits}-bit wall_seconds", nonnegative=True)
    declared_projection = _decimal_value(
        summary["projected_30000yr_wall_seconds"],
        f"Y6 {bits}-bit projected_30000yr_wall_seconds",
        nonnegative=True,
    )
    error_h = _decimal_value(summary["oscillator_error_h"], f"Y6 {bits}-bit oscillator_error_h", nonnegative=True)
    error_2h = _decimal_value(summary["oscillator_error_2h"], f"Y6 {bits}-bit oscillator_error_2h", nonnegative=True)
    if error_h == 0:
        raise ValueError(f"Y6 {bits}-bit oscillator_error_h must be positive")
    declared_ratio = _decimal_value(
        summary["oscillator_error_ratio_2h_over_h"],
        f"Y6 {bits}-bit oscillator ratio",
        nonnegative=True,
    )
    energy = _decimal_value(summary["relative_energy_drift"], f"Y6 {bits}-bit energy drift", nonnegative=True)
    angular = _decimal_value(
        summary["relative_angular_momentum_vector_drift"],
        f"Y6 {bits}-bit angular-momentum drift",
        nonnegative=True,
    )
    with localcontext() as context:
        context.prec = 160
        recomputed_ratio = error_2h / error_h
        recomputed_projection = wall * Decimal(30000) / Decimal(100)
    ratio_tolerance = Decimal(1).scaleb(-max(24, int(bits * 0.30103) - 4))
    if not _decimal_close(declared_ratio, recomputed_ratio, ratio_tolerance):
        raise ValueError(f"Y6 {bits}-bit oscillator ratio is not derived from the raw errors")
    if not _decimal_close(declared_projection, recomputed_projection, Decimal("1e-12")):
        raise ValueError(f"Y6 {bits}-bit projected runtime is not derived from wall_seconds")

    checks = {
        "projected_30000yr_wall_seconds": {
            "value": str(recomputed_projection), "gate": "<= 4500",
            "passed": recomputed_projection <= Decimal(4500),
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
        },
        "oscillator_order_ratio": {
            "value": str(recomputed_ratio), "gate": "60 <= ratio <= 68",
            "passed": Decimal(60) <= recomputed_ratio <= Decimal(68),
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
        },
        "relative_energy_drift": {
            "value": str(energy), "gate": "<= 1e-9", "passed": energy <= Decimal("1e-9"),
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
        },
        "relative_angular_momentum_vector_drift": {
            "value": str(angular), "gate": "<= 1e-10", "passed": angular <= Decimal("1e-10"),
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
        },
        "macro_steps": {
            "value": macro_steps, "gate": "= 5000", "passed": macro_steps == 5000,
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
        },
        "force_calls": {
            "value": force_calls, "gate": "= 40000", "passed": force_calls == 40000,
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
        },
    }
    return {"checks": checks, "passed": all(check["passed"] for check in checks.values())}


def _validate_trajectory(path: Path, context: str) -> dict[tuple[int, int], dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"{context} is not a regular file: {path}")
    rows: dict[tuple[int, int], dict[str, str]] = {}
    expected_names = MASSIVE_NAMES + TRACER_LABELS
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != Y6_TRAJECTORY_FIELDS:
                raise ValueError(f"{context} trajectory header does not match the locked schema")
            for row in reader:
                if None in row or set(row) != set(Y6_TRAJECTORY_FIELDS) or any(
                    row[field] is None for field in Y6_TRAJECTORY_FIELDS
                ):
                    raise ValueError(f"{context} contains a row outside the locked CSV schema")
                try:
                    time_index = int(row["time"])
                    body_index = int(row["body"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{context} contains a noninteger support coordinate") from exc
                if row["time"] != str(time_index) or row["body"] != str(body_index):
                    raise ValueError(f"{context} contains a noncanonical support coordinate")
                key = (time_index, body_index)
                if key in rows:
                    raise ValueError(f"{context} contains duplicate trajectory row {key}")
                if not (0 <= time_index <= 100 and 0 <= body_index < len(expected_names)):
                    raise ValueError(f"{context} contains out-of-range trajectory row {key}")
                if row["name"] != expected_names[body_index]:
                    raise ValueError(f"{context} body/name identity changed at {key}")
                for field in STATE_FIELDS + ("a", "e", "q", "i_deg"):
                    _decimal_value(row[field], f"{context} {key} {field}")
                if row["bound"] not in {"0", "1"}:
                    raise ValueError(f"{context} bound flag is not binary at {key}")
                rows[key] = row
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"{context} is not a readable trajectory CSV") from exc
    expected_support = {(year, body) for year in range(101) for body in range(len(expected_names))}
    if set(rows) != expected_support:
        raise ValueError(f"{context} trajectory support is not the complete locked 101x20 grid")
    return rows


def _require_initial_state_match(
    left: dict[tuple[int, int], dict[str, str]],
    right: dict[tuple[int, int], dict[str, str]],
    context: str,
) -> None:
    for body in range(len(MASSIVE_NAMES) + len(TRACER_LABELS)):
        for field in STATE_FIELDS:
            if Decimal(left[(0, body)][field]) != Decimal(right[(0, body)][field]):
                raise ValueError(f"{context} initial state differs at body={body}, field={field}")


def _validate_anchor_state(path: Path) -> dict[int, dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"locked DE441 anchor state is not a regular file: {path}")
    if _sha256(path) != NORMALIZED_STATE_SHA256:
        raise ValueError("copied DE441 anchor state hash does not match the locked identity")
    fields = ("index", "name", "mass", "x", "y", "z", "vx", "vy", "vz")
    rows: dict[int, dict[str, str]] = {}
    names = MASSIVE_NAMES + TRACER_LABELS
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError("locked DE441 anchor state header changed")
            for row in reader:
                if None in row or set(row) != set(fields) or any(row[field] is None for field in fields):
                    raise ValueError("locked DE441 anchor state contains a row outside the locked CSV schema")
                try:
                    index = int(row["index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError("locked DE441 anchor state contains a noninteger index") from exc
                if row["index"] != str(index):
                    raise ValueError("locked DE441 anchor state contains a noncanonical index")
                if index in rows or not 0 <= index < len(names):
                    raise ValueError("locked DE441 anchor state index support changed")
                if row["name"] != names[index]:
                    raise ValueError(f"locked DE441 anchor body/name identity changed at index {index}")
                mass = _decimal_value(row["mass"], f"locked DE441 anchor mass {index}", nonnegative=True)
                if (index < len(MASSIVE_NAMES)) != (mass > 0):
                    raise ValueError("locked DE441 anchor massless/massive roster changed")
                for field in STATE_FIELDS:
                    _decimal_value(row[field], f"locked DE441 anchor {index} {field}")
                rows[index] = row
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError("locked DE441 anchor state is not readable CSV") from exc
    if set(rows) != set(range(len(names))):
        raise ValueError("locked DE441 anchor state is not the complete 20-body roster")
    return rows


def _require_trajectory_matches_anchor(
    anchor: dict[int, dict[str, str]],
    trajectory: dict[tuple[int, int], dict[str, str]],
    context: str,
) -> None:
    for body, state_row in anchor.items():
        for field in STATE_FIELDS:
            if Decimal(state_row[field]) != Decimal(trajectory[(0, body)][field]):
                raise ValueError(f"{context} is not initialized from the locked state at body={body}, field={field}")


def _validate_source_manifest(manifest: Any) -> dict[str, Any]:
    from .provenance import sha256_data

    parsed = _require_exact_keys(manifest, {"schema", "scope", "files", "tree_sha256"}, "pair source manifest")
    if parsed["schema"] != "jx-source-manifest/v2":
        raise ValueError("pair source manifest schema is not jx-source-manifest/v2")
    if parsed["scope"] not in {"repository", "installed_package"}:
        raise ValueError("pair source manifest scope is invalid")
    files = parsed["files"]
    if type(files) is not dict or not files:
        raise ValueError("pair source manifest files must be a nonempty object")
    for relative, digest in files.items():
        if type(relative) is not str or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("pair source manifest contains an unsafe file name")
        _require_sha256(digest, f"pair source manifest digest for {relative}")
    tree_sha256 = _require_sha256(parsed["tree_sha256"], "pair source manifest tree_sha256")
    if tree_sha256 != sha256_data(files):
        raise ValueError("pair source manifest tree digest does not match its file map")
    package_dir = Path(__file__).resolve().parent
    critical = {
        "src/jxplanetx/__init__.py": package_dir / "__init__.py",
        "src/jxplanetx/de441_anchor.py": package_dir / "de441_anchor.py",
        "src/jxplanetx/decimal_bs.py": package_dir / "decimal_bs.py",
        "src/jxplanetx/production_benchmark.py": package_dir / "production_benchmark.py",
        "src/jxplanetx/provenance.py": package_dir / "provenance.py",
    }
    for relative, current_path in critical.items():
        if relative not in files:
            raise ValueError(f"pair source manifest omits critical source: {relative}")
        if files[relative] != _sha256(current_path):
            raise ValueError(f"pair source manifest does not identify the executing {relative}")
    return {"scope": parsed["scope"], "tree_sha256": tree_sha256, "critical_files_verified": sorted(critical)}


def _validate_pair_record(record_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from . import __version__
    from .provenance import sha256_data

    record = _read_json_object(record_path, "precision-pair run record")
    _require_exact_keys(record, {"schema", "created_utc", "environment", "payload", "payload_sha256"}, "precision-pair run record")
    if record["schema"] != PAIR_RECORD_SCHEMA:
        raise ValueError(f"precision-pair run record schema must be {PAIR_RECORD_SCHEMA}")
    created = _require_string(record["created_utc"], "precision-pair created_utc")
    try:
        parsed_created = datetime.fromisoformat(created.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("precision-pair created_utc is not an ISO-8601 timestamp") from exc
    if parsed_created.tzinfo is None:
        raise ValueError("precision-pair created_utc must include a timezone")
    environment = _require_exact_keys(
        record["environment"], {"python", "implementation", "platform"}, "precision-pair environment",
    )
    for key, value in environment.items():
        _require_string(value, f"precision-pair environment.{key}")
    payload = _require_exact_keys(
        record["payload"], {"engine_version", "command", "result", "software"}, "precision-pair payload",
    )
    if payload["command"] != PAIR_COMMAND:
        raise ValueError(f"precision-pair command must be {PAIR_COMMAND}")
    engine_version = _require_string(payload["engine_version"], "precision-pair engine_version")
    if engine_version != __version__:
        raise ValueError("precision-pair engine_version does not identify the executing package")
    declared_payload_sha256 = _require_sha256(record["payload_sha256"], "precision-pair payload_sha256")
    recomputed_payload_sha256 = sha256_data(payload)
    if declared_payload_sha256 != recomputed_payload_sha256:
        raise ValueError("precision-pair payload digest mismatch")
    source = _validate_source_manifest(payload["software"])
    provenance = {
        "record_sha256": _sha256(record_path),
        "payload_sha256": recomputed_payload_sha256,
        "source_manifest": source,
    }
    return payload["result"], provenance


def _validate_bs_summary(summary: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    _require_exact_keys(summary, BS_SUMMARY_FIELDS, "Decimal BS block summary")
    expected_method = "independent Decimal Bulirsch-Stoer; exact massless-tracer block decomposition"
    if summary["method"] != expected_method:
        raise ValueError("Decimal BS block summary method identity changed")
    years = _require_int(summary["years"], "Decimal BS years", minimum=0)
    digits = _require_int(summary["decimal_digits"], "Decimal BS decimal_digits", minimum=1)
    workers = _require_int(summary["workers"], "Decimal BS workers", minimum=1)
    tracer_blocks = _require_int(summary["tracer_blocks"], "Decimal BS tracer_blocks", minimum=0)
    wall = _decimal_value(summary["wall_seconds"], "Decimal BS wall_seconds", nonnegative=True)
    spread = _decimal_value(
        summary["maximum_massive_path_spread_across_blocks"],
        "Decimal BS maximum massive-path spread",
        nonnegative=True,
    )
    declared_invariants = _require_bool(
        summary["all_block_invariants_passed"], "Decimal BS all_block_invariants_passed",
    )
    blocks = _require_exact_keys(summary["block_summaries"], set(TRACER_LABELS), "Decimal BS block_summaries")
    block_results: dict[str, dict[str, Any]] = {}
    for tracer in TRACER_LABELS:
        block = _require_exact_keys(blocks[tracer], BS_BLOCK_SUMMARY_FIELDS, f"Decimal BS block {tracer}")
        if block["method"] != "independent Decimal Bulirsch-Stoer modified-midpoint extrapolation":
            raise ValueError(f"Decimal BS block {tracer} method identity changed")
        if _require_int(block["decimal_digits"], f"Decimal BS block {tracer} digits", minimum=1) != digits:
            raise ValueError(f"Decimal BS block {tracer} digit count disagrees with the aggregate")
        if _require_int(block["duration_years"], f"Decimal BS block {tracer} duration", minimum=0) != years:
            raise ValueError(f"Decimal BS block {tracer} duration disagrees with the aggregate")
        if _decimal_value(block["rtol"], f"Decimal BS block {tracer} rtol", nonnegative=True) != Decimal("1e-30"):
            raise ValueError(f"Decimal BS block {tracer} rtol changed")
        if _decimal_value(block["atol"], f"Decimal BS block {tracer} atol", nonnegative=True) != Decimal("1e-33"):
            raise ValueError(f"Decimal BS block {tracer} atol changed")
        _decimal_value(block["wall_seconds"], f"Decimal BS block {tracer} wall_seconds", nonnegative=True)
        _require_int(block["accepted_steps"], f"Decimal BS block {tracer} accepted_steps", minimum=1)
        _require_int(block["rejected_steps"], f"Decimal BS block {tracer} rejected_steps", minimum=0)
        _require_int(block["rhs_calls"], f"Decimal BS block {tracer} rhs_calls", minimum=1)
        level = _require_int(
            block["maximum_extrapolation_level"],
            f"Decimal BS block {tracer} maximum_extrapolation_level",
            minimum=1,
        )
        if level > 6:
            raise ValueError(f"Decimal BS block {tracer} extrapolation level exceeds the locked sequence")
        energy = _decimal_value(
            block["relative_energy_drift"], f"Decimal BS block {tracer} energy drift", nonnegative=True,
        )
        angular = _decimal_value(
            block["relative_angular_momentum_vector_drift"],
            f"Decimal BS block {tracer} angular-momentum drift",
            nonnegative=True,
        )
        block_results[tracer] = {
            "relative_energy_drift": str(energy),
            "energy_gate": "<= 1e-9",
            "energy_passed": energy <= Decimal("1e-9"),
            "relative_angular_momentum_vector_drift": str(angular),
            "angular_momentum_gate": "<= 1e-10",
            "angular_momentum_passed": angular <= Decimal("1e-10"),
        }
    recomputed_invariants = all(
        block["energy_passed"] and block["angular_momentum_passed"]
        for block in block_results.values()
    )
    if declared_invariants != recomputed_invariants:
        raise ValueError("Decimal BS aggregate invariant declaration disagrees with its block summaries")
    if tracer_blocks != len(TRACER_LABELS):
        raise ValueError("Decimal BS tracer block count is not the locked 15-block decomposition")
    reported_spread_within_gate = spread <= Decimal("1e-25")
    checks = {
        "duration_years": {"value": years, "gate": "= 100", "passed": years == 100},
        "decimal_digits": {"value": digits, "gate": ">= 68", "passed": digits >= 68},
        "wall_seconds": {"value": str(wall), "gate": "<= 4500", "passed": wall <= Decimal(4500)},
        "tracer_blocks": {"value": tracer_blocks, "gate": "= 15", "passed": tracer_blocks == 15},
        "block_invariants_recomputed_from_summary_fields": {
            "value": recomputed_invariants,
            "gate": "all 15 energy and angular-momentum drifts within limits",
            "passed": recomputed_invariants,
            "evidence_origin": "REPORT_DERIVED_RETHRESHOLDED",
            "blocks": block_results,
        },
        "massive_path_spread_across_raw_blocks": {
            "reported_value": str(spread),
            "gate": "<= 1e-25 AU/component",
            "passed": None,
            "reported_gate_passed": reported_spread_within_gate,
            "status": "UNVERIFIED_RAW_BLOCK_TRAJECTORIES_NOT_BOUND",
            "evidence_origin": "REPORT_DERIVED_UNVERIFIED",
            "note": (
                "the merged trajectory cannot reproduce the across-block spread; "
                "a reported failure blocks screening, but a reported pass cannot certify this gate"
            ),
        },
    }
    recomputable_passed = all(
        check["passed"] for check in checks.values() if check["passed"] is not None
    ) and reported_spread_within_gate
    return checks, recomputable_passed


def import_de441_anchor(source_csv: str | Path, output_csv: str | Path, metadata_json: str | Path) -> dict[str, Any]:
    source = Path(source_csv)
    if _sha256(source) != SOURCE_STATE_SHA256:
        raise ValueError("DE441 anchor source-state hash does not match the preserved artifact")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    selected = [row for row in rows if row["family"] == "baseline" and row["phase_index"] == "0"]
    massive_names = MASSIVE_NAMES
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
    from .production_benchmark import compare_trajectories

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
    state_snapshot = output / "de441_anchor_initial_state.csv"
    if state != state_snapshot:
        shutil.copyfile(state, state_snapshot)
    if _sha256(state_snapshot) != NORMALIZED_STATE_SHA256:
        raise ValueError("copied normalized DE441 anchor state hash changed")
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
        summary = _read_json_object(summary_path, f"Y6 {bits}-bit summary")
        runs[str(bits)] = {
            "summary": summary,
            "gates": _recompute_y6_summary_gates(summary, bits),
            "trajectory_filename": trajectory.name,
            "trajectory_sha256": _sha256(trajectory),
            "summary_filename": summary_path.name,
            "summary_sha256": _sha256(summary_path),
        }
        trajectories[bits] = trajectory
    anchor_rows = _validate_anchor_state(state_snapshot)
    produced_rows = {
        bits: _validate_trajectory(path, f"produced Y6 {bits}-bit")
        for bits, path in trajectories.items()
    }
    for bits, rows in produced_rows.items():
        _require_trajectory_matches_anchor(anchor_rows, rows, f"produced Y6 {bits}-bit trajectory")
    _require_initial_state_match(produced_rows[160], produced_rows[224], "produced Y6 precision pair")
    comparison = compare_trajectories(trajectories[160], trajectories[224], "DE441 anchor Y6 160 vs Y6 224")
    pair_passed = all(run["gates"]["passed"] for run in runs.values()) and comparison["passed"]
    return {
        "schema": PAIR_RESULT_SCHEMA,
        "classification": "MODEL_OUTPUT numerical precision-pair gate",
        "state_filename": state_snapshot.name,
        "state_sha256": _sha256(state_snapshot),
        "yoshida6_binary_sha256": _sha256(binary),
        "runs": runs,
        "cross_precision_comparison": comparison,
        "precision_pair_passed": pair_passed,
        "production_qualified": False,
        "status": "PARTIAL_PASS_DECIMAL_BS_CROSSCHECK_MISSING" if pair_passed else "INVALID",
        "blocker": "separately implemented Decimal BS cross-check not yet audited",
        "claim_decision": "SCREENING_ONLY" if pair_passed else "INVALID",
    }


def audit_de441_independent_reference(
    pair_record_path: str | Path,
    y6_trajectory: str | Path,
    bs_trajectory: str | Path,
    bs_summary_path: str | Path,
) -> dict[str, Any]:
    """Check supplied DE441/Y6 and Decimal-BS artifacts conservatively.

    The legacy interface has no Decimal-BS producer record binding the supplied
    trajectory and summary to one execution.  The function can therefore check
    artifact consistency, but cannot authenticate a cross-method result.
    """
    from .decimal_bs import validate_bs_oscillator
    from .production_benchmark import compare_trajectories

    pair_path = Path(pair_record_path).resolve()
    y6 = Path(y6_trajectory).resolve()
    bs = Path(bs_trajectory).resolve()
    bs_summary_file = Path(bs_summary_path).resolve()
    pair_result, pair_provenance = _validate_pair_record(pair_path)
    _require_exact_keys(pair_result, PAIR_RESULT_FIELDS, "precision-pair result")
    if pair_result["schema"] != PAIR_RESULT_SCHEMA:
        raise ValueError(f"precision-pair result schema must be {PAIR_RESULT_SCHEMA}; legacy records fail closed")
    if pair_result["classification"] != "MODEL_OUTPUT numerical precision-pair gate":
        raise ValueError("precision-pair classification changed")
    if pair_result["state_sha256"] != NORMALIZED_STATE_SHA256:
        raise ValueError("precision-pair result does not identify the locked DE441 anchor state")
    if pair_result["state_filename"] != "de441_anchor_initial_state.csv":
        raise ValueError("precision-pair state filename changed")
    if pair_result["yoshida6_binary_sha256"] != YOSHIDA6_BINARY_SHA256:
        raise ValueError("precision-pair result does not identify the locked Yoshida-6 executable")
    if _require_bool(pair_result["production_qualified"], "precision-pair production_qualified"):
        raise ValueError("precision-pair result may not claim production qualification")
    if pair_result["blocker"] != "separately implemented Decimal BS cross-check not yet audited":
        raise ValueError("precision-pair blocker identity changed")

    runs = _require_exact_keys(pair_result["runs"], {"160", "224"}, "precision-pair runs")
    run_root = y6.parent
    expected_224_path = run_root / "de441_anchor_y6_224.csv"
    if y6 != expected_224_path:
        raise ValueError("the supplied Y6 trajectory must be the fixed 224-bit run filename")
    anchor_rows = _validate_anchor_state(run_root / pair_result["state_filename"])
    trajectory_rows: dict[int, dict[tuple[int, int], dict[str, str]]] = {}
    recomputed_runs: dict[str, Any] = {}
    for bits in (160, 224):
        key = str(bits)
        run = _require_exact_keys(runs[key], Y6_RUN_FIELDS, f"precision-pair run {bits}")
        expected_trajectory_name = f"de441_anchor_y6_{bits}.csv"
        expected_summary_name = f"de441_anchor_y6_{bits}_summary.json"
        if run["trajectory_filename"] != expected_trajectory_name:
            raise ValueError(f"precision-pair run {bits} trajectory filename changed")
        if run["summary_filename"] != expected_summary_name:
            raise ValueError(f"precision-pair run {bits} summary filename changed")
        trajectory_path = run_root / expected_trajectory_name
        summary_path = run_root / expected_summary_name
        summary = _read_json_object(summary_path, f"Y6 {bits}-bit summary file")
        if not _canonical_equal(run["summary"], summary):
            raise ValueError(f"precision-pair run {bits} inline summary differs from its file")
        actual_summary_hash = _sha256(summary_path)
        if _require_sha256(run["summary_sha256"], f"precision-pair run {bits} summary_sha256") != actual_summary_hash:
            raise ValueError(f"precision-pair run {bits} summary file hash mismatch")
        if not trajectory_path.is_file():
            raise ValueError(f"Y6 {bits}-bit trajectory is not a regular file: {trajectory_path}")
        actual_trajectory_hash = _sha256(trajectory_path)
        if _require_sha256(run["trajectory_sha256"], f"precision-pair run {bits} trajectory_sha256") != actual_trajectory_hash:
            raise ValueError(f"precision-pair run {bits} trajectory file hash mismatch")
        recomputed_gates = _recompute_y6_summary_gates(summary, bits)
        if not _canonical_equal(run["gates"], recomputed_gates):
            raise ValueError(f"precision-pair run {bits} declared gates differ from recomputation")
        trajectory_rows[bits] = _validate_trajectory(trajectory_path, f"Y6 {bits}-bit")
        _require_trajectory_matches_anchor(anchor_rows, trajectory_rows[bits], f"Y6 {bits}-bit trajectory")
        recomputed_runs[key] = {
            "summary_sha256": actual_summary_hash,
            "trajectory_sha256": actual_trajectory_hash,
            "gates": recomputed_gates,
        }
    _require_initial_state_match(trajectory_rows[160], trajectory_rows[224], "Y6 precision pair")
    pair_comparison = compare_trajectories(
        run_root / "de441_anchor_y6_160.csv",
        run_root / "de441_anchor_y6_224.csv",
        "DE441 anchor Y6 160 vs Y6 224",
    )
    if not _canonical_equal(pair_result["cross_precision_comparison"], pair_comparison):
        raise ValueError("declared cross-precision comparison differs from recomputation")
    recomputed_pair_passed = (
        all(run["gates"]["passed"] for run in recomputed_runs.values())
        and pair_comparison["passed"]
    )
    declared_pair_passed = _require_bool(pair_result["precision_pair_passed"], "precision_pair_passed")
    if declared_pair_passed != recomputed_pair_passed:
        raise ValueError("declared precision_pair_passed differs from recomputation")
    expected_pair_status = "PARTIAL_PASS_DECIMAL_BS_CROSSCHECK_MISSING" if recomputed_pair_passed else "INVALID"
    expected_pair_claim = "SCREENING_ONLY" if recomputed_pair_passed else "INVALID"
    if pair_result["status"] != expected_pair_status or pair_result["claim_decision"] != expected_pair_claim:
        raise ValueError("precision-pair status or claim decision disagrees with recomputed gates")

    bs_rows = _validate_trajectory(bs, "Decimal BS merged")
    _require_initial_state_match(trajectory_rows[224], bs_rows, "Y6/Decimal-BS comparison")
    summary = _read_json_object(bs_summary_file, "Decimal BS block summary")
    reference_checks, reference_recomputable_passed = _validate_bs_summary(summary)
    support_comparison = compare_trajectories(
        y6, bs, f"DE441 anchor Y6 224 vs same-package Decimal BS {summary['decimal_digits']}-digit"
    )
    oscillator = validate_bs_oscillator(int(summary["decimal_digits"]))
    oscillator_passed = _require_bool(oscillator.get("passed"), "Decimal BS oscillator validation passed")
    artifact_consistency_gates_passed = (
        recomputed_pair_passed
        and reference_recomputable_passed
        and oscillator_passed
        and support_comparison["passed"]
    )
    # The legacy interface accepts the merged BS trajectory and aggregate
    # summary as unrelated caller-supplied paths.  Without a producer record
    # that binds both bytes to the locked state, solver configuration, raw
    # block trajectories, and execution, their agreement is only an artifact
    # consistency observation.  It must not be promoted to a cross-method
    # validation claim.
    bs_artifact_provenance_verified = False
    unresolved = [
        "bs_producer_record_and_trajectory_summary_binding",
        "massive_path_spread_across_raw_blocks",
    ]
    complete = False
    artifact_status = (
        "AUDIT_INCOMPLETE_UNVERIFIED_BS_PROVENANCE"
        if artifact_consistency_gates_passed
        else "INVALID"
    )
    return {
        "schema": "jx-de441-reference-audit/v2",
        "classification": "MODEL_OUTPUT / numerical validation only",
        "state_sha256": NORMALIZED_STATE_SHA256,
        "precision_pair_passed": recomputed_pair_passed,
        "pair_record_provenance": pair_provenance,
        "recomputed_precision_runs": recomputed_runs,
        "reference_relationship": "UNVERIFIED_CALLER_SUPPLIED_BS_ARTIFACT_PAIR",
        "bs_artifact_provenance_verified": bs_artifact_provenance_verified,
        "yoshida6_binary_identity_status": "DECLARED_LOCKED_IDENTITY_NOT_REHASHED_BY_AUDIT",
        "decimal_bs_kernel_validation": oscillator,
        "decimal_bs_reference_checks": reference_checks,
        "cross_method_comparison": support_comparison,
        "artifact_sha256": {
            "y6_trajectory": _sha256(y6),
            "bs_trajectory": _sha256(bs),
            "bs_summary": _sha256(bs_summary_file),
        },
        "artifact_consistency_gates_passed": artifact_consistency_gates_passed,
        "gate_evidence_classification": {
            "trajectory_support_and_cross_comparisons": "RECOMPUTED_FROM_REHASHED_ARTIFACT_BYTES",
            "decimal_bs_oscillator": "RECOMPUTED_BY_CURRENT_SAME_PACKAGE_IMPLEMENTATION",
            "yoshida6_summary_metrics": "REPORT_DERIVED_RETHRESHOLDED",
            "decimal_bs_summary_metrics": "REPORT_DERIVED_RETHRESHOLDED",
            "decimal_bs_massive_path_spread": "REPORT_DERIVED_UNVERIFIED",
        },
        "all_recomputable_gates_passed": False,
        "unverified_gates": unresolved,
        "all_numerical_gates_passed": complete,
        "numerical_status": artifact_status,
        "external_independence_qualified": False,
        "production_observation_qualified": False,
        "claim_decision": "NO_CROSS_METHOD_CLAIM" if artifact_consistency_gates_passed else "INVALID",
        "scientific_scope": (
            "artifact-consistency check only for this locked DE441 anchor; the caller-supplied "
            "BS trajectory and summary lack a common producer record, the raw block trajectories "
            "are unbound, and the result is neither cross-method validation, external validation, "
            "nor evidence for a Planet X source"
        ),
    }
