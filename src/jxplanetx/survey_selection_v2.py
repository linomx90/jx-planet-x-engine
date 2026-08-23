"""Corrected official OSSOS adapter for the JX-O1 v2 experiment.

Version 1 remains frozen so its pilot and failed official qualification can be
reproduced.  This module changes only the official tracked-file boundary and
the functions that depend on it.  Population generation and statistical
evaluation continue to use the frozen v1 implementation.

The pinned F95 Driver writes tracked rows with 14 whitespace-separated fields::

    a e i node peri M H_int q r delta m_rand H_rand color comment

The v1 qualification incorrectly modeled the 20-field detected-object output.
V2 parses the actual 14-field tracked output and requires the comment to be the
locked JX object identity for the declared model and seed block.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import survey_selection as v1
from .provenance import sha256_data, sha256_file


CONTRACT_SCHEMA = v1.CONTRACT_SCHEMA
POOL_SCHEMA = v1.POOL_SCHEMA
RESULT_SCHEMA = v1.RESULT_SCHEMA
DETECTION_COLUMNS = v1.DETECTION_COLUMNS
MODEL_IDS = v1.MODEL_IDS
NONCLAIMS = v1.NONCLAIMS
SurveySelectionError = v1.SurveySelectionError
SurveySelectionVerdict = v1.SurveySelectionVerdict

validate_survey_contract = v1.validate_survey_contract
load_survey_contract = v1.load_survey_contract
generate_intrinsic_population = v1.generate_intrinsic_population
write_ossos_model_file = v1.write_ossos_model_file
write_detection_csv = v1.write_detection_csv
load_detection_csv = v1.load_detection_csv
verify_external_simulator = v1.verify_external_simulator
write_pool_manifest = v1.write_pool_manifest
empirical_pit = v1.empirical_pit
anderson_darling_uniform = v1.anderson_darling_uniform


def parse_ossos_tracked_file(
    path: str | Path,
    model_id: str,
    seed_block: int,
) -> list[dict[str, Any]]:
    """Normalize the pinned F95 Driver's actual 14-field tracked rows."""

    if model_id not in MODEL_IDS:
        v1._fail("invalid_model_id", f"model_id must be one of {MODEL_IDS}")
    seed_block = v1._nonnegative_int(seed_block, "seed_block")
    prefix = "c" if model_id == "correct" else "w"
    identity = re.compile(rf"{prefix}{seed_block:02d}[0-9]{{8}}\Z")
    source = Path(path)
    rows: list[dict[str, Any]] = []
    try:
        for line_number, raw in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 14:
                v1._fail(
                    "invalid_ossos_tracked_row",
                    f"{source}:{line_number} must have exactly 14 tracked-output fields",
                )
            object_id = fields[13]
            if identity.fullmatch(object_id) is None:
                v1._fail(
                    "invalid_ossos_object_id",
                    f"{source}:{line_number} identity does not match {model_id} block {seed_block}",
                )
            row = {
                "object_id": object_id,
                "model_id": model_id,
                "seed_block": seed_block,
                "a_AU": fields[0],
                "q_AU": fields[7],
                "i_deg": fields[2],
                "H_r": fields[6],
                "r_AU": fields[8],
                # Driver.f95 passes m_rand first in format 9010.  Its header
                # labels these two columns in the opposite order.
                "m_r": fields[10],
            }
            rows.append(v1._validate_detection(row, f"{source}:{line_number}"))
    except OSError as exc:
        v1._fail("ossos_read_error", f"cannot read OSSOS output {source}: {exc}")
    return rows


def register_official_ossos_pool(
    contract: Mapping[str, Any],
    simulator_root: str | Path,
    model_id: str,
    raw_blocks: Sequence[tuple[int, str | Path, int]],
    detections_output: str | Path,
    manifest_output: str | Path,
    *,
    checkpoint_replay_passed: bool,
) -> dict[str, Any]:
    """Register raw v2 tracked output and prove exact adapter identity."""

    locked = validate_survey_contract(contract)
    verification = verify_external_simulator(locked, simulator_root)
    if not verification["passed"]:
        v1._fail("external_simulator_mismatch", "pinned external simulator verification failed")
    manifest_target = Path(manifest_output)
    normalized: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    draws: dict[int, int] = {}
    for block, raw_path, intrinsic_draws in raw_blocks:
        parsed = parse_ossos_tracked_file(raw_path, model_id, block)
        normalized.extend(parsed)
        draws[block] = draws.get(block, 0) + v1._nonnegative_int(
            intrinsic_draws, f"intrinsic draws block {block}"
        )
        raw_records.append(
            {
                "seed_block": block,
                "path": os.path.relpath(
                    Path(raw_path).resolve(), manifest_target.parent.resolve()
                ),
                "sha256": sha256_file(raw_path),
                "tracked_count": len(parsed),
            }
        )
    write_detection_csv(detections_output, normalized)
    return write_pool_manifest(
        manifest_output,
        model_id=model_id,
        backend=locked["execution"]["official_backend_name"],
        simulator_commit=locked["external_simulator"]["commit"],
        detections_path=detections_output,
        intrinsic_draws_by_block=draws,
        raw_tracked_files=raw_records,
        checkpoint_replay_passed=checkpoint_replay_passed,
    )


def _load_pool(
    path: str | Path,
    contract: Mapping[str, Any],
    expected_model: str,
) -> dict[str, Any]:
    source = Path(path)
    manifest = v1._load_json(source, "pool_manifest_read_error")
    v1._check_keys(
        manifest,
        {
            "schema",
            "model_id",
            "backend",
            "simulator_commit",
            "detections_path",
            "detections_sha256",
            "detection_semantic_sha256",
            "detection_count",
            "intrinsic_draws_by_block",
            "raw_tracked_files",
            "checkpoint_replay_passed",
        },
        "pool manifest",
    )
    if manifest["schema"] != POOL_SCHEMA or manifest["model_id"] != expected_model:
        v1._fail(
            "pool_manifest_mismatch",
            f"pool manifest does not describe {expected_model!r}",
        )
    v1._nonempty_string(manifest["backend"], "pool backend")
    if manifest["simulator_commit"] is not None:
        v1._nonempty_string(manifest["simulator_commit"], "pool simulator commit")
    relative = v1._nonempty_string(manifest["detections_path"], "pool detections_path")
    if Path(relative).is_absolute():
        v1._fail("absolute_pool_path", "pool detections_path must be relative")
    detection_path = (source.parent / relative).resolve()
    if sha256_file(detection_path) != v1._sha256(
        manifest["detections_sha256"], "detections_sha256"
    ):
        v1._fail(
            "detection_hash_mismatch", f"detection file hash mismatch for {expected_model}"
        )
    rows = load_detection_csv(detection_path)
    if len(rows) != v1._nonnegative_int(manifest["detection_count"], "detection_count"):
        v1._fail("detection_count_mismatch", f"detection count mismatch for {expected_model}")
    if sha256_data(rows) != v1._sha256(
        manifest["detection_semantic_sha256"], "semantic hash"
    ):
        v1._fail(
            "detection_semantic_hash_mismatch",
            f"semantic hash mismatch for {expected_model}",
        )
    if any(row["model_id"] != expected_model for row in rows):
        v1._fail("pool_model_mismatch", f"row model mismatch in {expected_model} pool")

    draws_raw = v1._mapping(
        manifest["intrinsic_draws_by_block"], "intrinsic_draws_by_block"
    )
    draws: dict[int, int] = {}
    for key, value in draws_raw.items():
        try:
            block = int(key)
        except (TypeError, ValueError):
            v1._fail("invalid_seed_block", f"invalid seed block key: {key!r}")
        draws[block] = v1._nonnegative_int(value, f"intrinsic draws block {block}")
    expected_blocks = set(range(int(contract["population"]["seed_blocks"])))
    if set(draws) != expected_blocks:
        v1._fail(
            "seed_block_set_mismatch", f"pool must contain seed blocks {sorted(expected_blocks)}"
        )
    if any(row["seed_block"] not in expected_blocks for row in rows):
        v1._fail("unplanned_seed_block", "detection contains an unplanned seed block")

    raw_files = manifest["raw_tracked_files"]
    if not isinstance(raw_files, list):
        v1._fail("invalid_raw_file_list", "raw_tracked_files must be a list")
    official_backend = contract["execution"]["official_backend_name"]
    adapter_identity_passed = False
    if manifest["backend"] == official_backend:
        if manifest["simulator_commit"] != contract["external_simulator"]["commit"]:
            v1._fail("simulator_commit_mismatch", "official pool uses the wrong simulator commit")
        parsed: list[dict[str, Any]] = []
        raw_blocks: set[int] = set()
        raw_paths: set[Path] = set()
        for index, item_raw in enumerate(raw_files):
            item = v1._mapping(item_raw, f"raw_tracked_files[{index}]")
            v1._check_keys(
                item,
                {"seed_block", "path", "sha256", "tracked_count"},
                "raw tracked file",
            )
            block = v1._nonnegative_int(item["seed_block"], "raw seed block")
            raw_blocks.add(block)
            raw_relative = v1._nonempty_string(item["path"], "raw tracked path")
            if Path(raw_relative).is_absolute():
                v1._fail("absolute_pool_path", "raw tracked paths must be relative")
            raw_path = (source.parent / raw_relative).resolve()
            if raw_path in raw_paths:
                v1._fail("duplicate_raw_file", f"raw tracked file appears twice: {raw_path}")
            raw_paths.add(raw_path)
            if sha256_file(raw_path) != v1._sha256(item["sha256"], "raw tracked hash"):
                v1._fail(
                    "raw_tracked_hash_mismatch", f"raw tracked hash mismatch for block {block}"
                )
            block_rows = parse_ossos_tracked_file(raw_path, expected_model, block)
            if len(block_rows) != v1._nonnegative_int(
                item["tracked_count"], "raw tracked count"
            ):
                v1._fail(
                    "raw_tracked_count_mismatch",
                    f"raw tracked count mismatch for block {block}",
                )
            parsed.extend(block_rows)
        parsed.sort(key=lambda row: row["object_id"])
        adapter_identity_passed = raw_blocks == expected_blocks and sha256_data(
            parsed
        ) == sha256_data(rows)
        if not adapter_identity_passed:
            v1._fail(
                "adapter_identity_mismatch", "raw OSSOS output and normalized pool differ"
            )
    return {
        "manifest": manifest,
        "manifest_sha256": sha256_file(source),
        "rows": rows,
        "intrinsic_draws_by_block": draws,
        "total_intrinsic_draws": sum(draws.values()),
        "adapter_identity_passed": adapter_identity_passed,
        "checkpoint_replay_passed": manifest["checkpoint_replay_passed"] is True,
    }


def _finalize_impl(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
) -> dict[str, Any]:
    contract = load_survey_contract(contract_path)
    correct = _load_pool(correct_manifest_path, contract, "correct")
    wrong = _load_pool(wrong_manifest_path, contract, "wrong")
    invalid_reasons: list[dict[str, str]] = []
    blocked_reasons: list[dict[str, str]] = []

    full = v1._evaluate_statistics(correct["rows"], wrong["rows"], contract, label="full")
    replay = v1._evaluate_statistics(correct["rows"], wrong["rows"], contract, label="full")
    exact_replay_passed = sha256_data(full) == sha256_data(replay)
    if not full["calibration_passed"]:
        invalid_reasons.append(
            {
                "code": "statistical_calibration_failed",
                "message": "one or more calibration gates failed",
            }
        )
    if not full["power_passed"]:
        blocked_reasons.append(
            {
                "code": "insufficient_wrong_model_power",
                "message": "wrong-model rejection power is below the locked minimum",
            }
        )
    if contract["gates"]["exact_replay_required"] and not exact_replay_passed:
        invalid_reasons.append(
            {
                "code": "exact_replay_failed",
                "message": "deterministic statistical replay changed",
            }
        )

    expected_blocks = list(range(int(contract["population"]["seed_blocks"])))
    stability: list[dict[str, Any]] = []
    full_classification = v1._gate_classification(full)
    for excluded in expected_blocks:
        correct_loo = [row for row in correct["rows"] if row["seed_block"] != excluded]
        wrong_loo = [row for row in wrong["rows"] if row["seed_block"] != excluded]
        loo = v1._evaluate_statistics(
            correct_loo,
            wrong_loo,
            contract,
            label=f"leave-one-block-out:{excluded}",
            mock_catalogs=int(contract["statistics"]["seed_stability_catalogs"]),
            reference_catalogs=int(contract["statistics"]["seed_stability_catalogs"]),
        )
        classification = v1._gate_classification(loo)
        stability.append(
            {
                "excluded_seed_block": excluded,
                "classification": classification,
                "false_rejection_rate": loo["correct_model_false_rejection_rate"],
                "wrong_model_rejection_power": loo["wrong_model_rejection_power"],
                "zeta_mean": loo["zeta_mean"],
                "zeta_sd": loo["zeta_sd"],
                "stable": classification == full_classification,
            }
        )
    seed_stability_passed = all(item["stable"] for item in stability)
    if (
        contract["gates"]["seed_block_verdict_stability_required"]
        and not seed_stability_passed
    ):
        invalid_reasons.append(
            {
                "code": "seed_block_verdict_instability",
                "message": "a leave-one-block-out verdict changed",
            }
        )

    official_backend = contract["execution"]["official_backend_name"]
    for name, pool in (("correct", correct), ("wrong", wrong)):
        manifest = pool["manifest"]
        if pool["total_intrinsic_draws"] < int(
            contract["population"]["minimum_intrinsic_draws_per_model"]
        ):
            blocked_reasons.append(
                {
                    "code": f"{name}_intrinsic_scale_incomplete",
                    "message": "intrinsic draw target is incomplete",
                }
            )
        if len(pool["rows"]) < int(
            contract["population"]["minimum_tracked_detections_per_model"]
        ):
            blocked_reasons.append(
                {
                    "code": f"{name}_tracked_scale_incomplete",
                    "message": "tracked-detection target is incomplete",
                }
            )
        detections_by_block = {
            block: sum(row["seed_block"] == block for row in pool["rows"])
            for block in expected_blocks
        }
        for block in expected_blocks:
            if pool["intrinsic_draws_by_block"][block] < int(
                contract["population"]["minimum_intrinsic_draws_per_seed_block"]
            ):
                blocked_reasons.append(
                    {
                        "code": f"{name}_intrinsic_block_{block}_incomplete",
                        "message": f"seed block {block} has too few intrinsic draws",
                    }
                )
            if detections_by_block[block] < int(
                contract["population"]["minimum_tracked_detections_per_seed_block"]
            ):
                blocked_reasons.append(
                    {
                        "code": f"{name}_tracked_block_{block}_incomplete",
                        "message": f"seed block {block} has too few tracked detections",
                    }
                )
        if manifest["backend"] != official_backend:
            blocked_reasons.append(
                {
                    "code": f"{name}_official_backend_missing",
                    "message": "pool was not produced by the pinned official backend",
                }
            )
        if contract["gates"]["adapter_identity_required"] and not pool[
            "adapter_identity_passed"
        ]:
            blocked_reasons.append(
                {
                    "code": f"{name}_adapter_identity_unproven",
                    "message": "raw-to-normalized adapter identity is not proven",
                }
            )
        if contract["execution"]["checkpoint_restart_required"] and not pool[
            "checkpoint_replay_passed"
        ]:
            blocked_reasons.append(
                {
                    "code": f"{name}_checkpoint_replay_unproven",
                    "message": "checkpoint/restart replay is not proven",
                }
            )

    if invalid_reasons:
        verdict = SurveySelectionVerdict.INVALID.value
    elif blocked_reasons:
        verdict = SurveySelectionVerdict.BLOCKED.value
    else:
        verdict = SurveySelectionVerdict.PASSED.value
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "milestone": contract["milestone"],
        "verdict": verdict,
        "claim_decision": "SCREENING_ONLY",
        "contract_sha256": sha256_file(contract_path),
        "input_manifests": {
            "correct": correct["manifest_sha256"],
            "wrong": wrong["manifest_sha256"],
        },
        "pool_summary": {
            "correct": {
                "backend": correct["manifest"]["backend"],
                "intrinsic_draws": correct["total_intrinsic_draws"],
                "tracked_detections": len(correct["rows"]),
                "adapter_identity_passed": correct["adapter_identity_passed"],
                "checkpoint_replay_passed": correct["checkpoint_replay_passed"],
            },
            "wrong": {
                "backend": wrong["manifest"]["backend"],
                "intrinsic_draws": wrong["total_intrinsic_draws"],
                "tracked_detections": len(wrong["rows"]),
                "adapter_identity_passed": wrong["adapter_identity_passed"],
                "checkpoint_replay_passed": wrong["checkpoint_replay_passed"],
            },
        },
        "statistics": full,
        "exact_replay_passed": exact_replay_passed,
        "seed_block_stability": stability,
        "seed_block_stability_passed": seed_stability_passed,
        "invalid_reasons": invalid_reasons,
        "blocked_reasons": blocked_reasons,
        "nonclaims": list(NONCLAIMS),
    }
    core["replay_sha256"] = sha256_data(core)
    return core


def finalize_survey_selection(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Evaluate JX-O1 v2 and emit exactly PASSED, BLOCKED, or INVALID."""

    target = Path(output)
    if target.exists():
        v1._fail("output_exists", f"refusing to overwrite immutable output: {target}")
    try:
        result = _finalize_impl(contract_path, correct_manifest_path, wrong_manifest_path)
    except SurveySelectionError as exc:
        result = {
            "schema": RESULT_SCHEMA,
            "experiment_id": "UNRESOLVED",
            "milestone": "JX-O1_TELESCOPE_SELECTION_VALIDATION",
            "verdict": SurveySelectionVerdict.INVALID.value,
            "claim_decision": "INVALID",
            "invalid_reasons": [{"code": exc.code, "message": exc.message}],
            "blocked_reasons": [],
            "nonclaims": list(NONCLAIMS),
        }
        result["replay_sha256"] = sha256_data(result)
    v1._atomic_json(target, result)
    return result


def run_analytic_survey_pilot(
    contract_path: str | Path,
    run_dir: str | Path,
    output: str | Path,
    *,
    draws_per_block: int = 10_000,
) -> dict[str, Any]:
    """Run the non-final analytic pilot through the corrected v2 finalizer."""

    contract = load_survey_contract(contract_path)
    draws_per_block = v1._positive_int(draws_per_block, "draws_per_block")
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_paths: dict[str, Path] = {}
    for model_id in MODEL_IDS:
        rows, draws = v1._analytic_pilot_detections(contract, model_id, draws_per_block)
        detections = root / f"{model_id}_detections.csv"
        manifest = root / f"{model_id}_pool.json"
        write_detection_csv(detections, rows)
        write_pool_manifest(
            manifest,
            model_id=model_id,
            backend=contract["execution"]["pilot_backend_name"],
            simulator_commit=None,
            detections_path=detections,
            intrinsic_draws_by_block=draws,
            raw_tracked_files=[],
            checkpoint_replay_passed=False,
        )
        manifest_paths[model_id] = manifest
    return finalize_survey_selection(
        contract_path,
        manifest_paths["correct"],
        manifest_paths["wrong"],
        output,
    )

