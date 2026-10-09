"""Exact-zeta corrective replay for the JX-O1 official v2 pools.

V2 estimated the mean and standard deviation of a sum of independent
log10(PIT) draws with 2,000 Monte Carlo catalogs.  Those two moments are known
exactly for sampling with replacement from a finite empirical pool.  V3 keeps
the v2 telescope outputs, AD bootstrap, thresholds, power test, and all
provenance gates unchanged; only the zeta moment estimator is replaced by its
finite-pool formula.  This is a corrective replay, not independent evidence.
"""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import survey_selection as v1
from . import survey_selection_v2 as v2
from .provenance import sha256_data


CONTRACT_SCHEMA = v2.CONTRACT_SCHEMA
RESULT_SCHEMA = v2.RESULT_SCHEMA
SurveySelectionError = v2.SurveySelectionError
SurveySelectionVerdict = v2.SurveySelectionVerdict

validate_survey_contract = v2.validate_survey_contract
load_survey_contract = v2.load_survey_contract


def exact_zeta_moments(
    correct_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> tuple[float, float]:
    """Return exact catalog mean and SD for finite-pool replacement draws."""

    if not correct_rows:
        v1._fail("empty_statistical_pool", "exact zeta moments require a nonempty pool")
    primary = contract["statistics"]["primary_pit_variable"]
    model = sorted(float(row[primary]) for row in correct_rows)
    log_pits = [
        math.log10(v2.empirical_pit(float(row[primary]), model)) for row in correct_rows
    ]
    catalog_size = int(contract["population"]["catalog_size"])
    mean = catalog_size * statistics.fmean(log_pits)
    variance = statistics.pvariance(log_pits)
    return mean, math.sqrt(catalog_size * variance)


def _evaluate_statistics(
    correct_rows: Sequence[Mapping[str, Any]],
    wrong_rows: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    label: str,
    mock_catalogs: int | None = None,
    reference_catalogs: int | None = None,
) -> dict[str, Any]:
    result = v1._evaluate_statistics(
        correct_rows,
        wrong_rows,
        contract,
        label=label,
        mock_catalogs=mock_catalogs,
        reference_catalogs=reference_catalogs,
    )
    monte_carlo_mean = result["zeta_mean"]
    monte_carlo_sd = result["zeta_sd"]
    exact_mean, exact_sd = exact_zeta_moments(correct_rows, contract)
    stats = contract["statistics"]
    gates = contract["gates"]
    result["zeta_estimator"] = "exact_finite_empirical_pool_with_replacement"
    result["zeta_monte_carlo_diagnostic_mean"] = monte_carlo_mean
    result["zeta_monte_carlo_diagnostic_sd"] = monte_carlo_sd
    result["zeta_mean"] = exact_mean
    result["zeta_sd"] = exact_sd
    result["gate_results"]["zeta_mean"] = abs(
        exact_mean - float(stats["zeta_expected_mean"])
    ) <= float(gates["zeta_mean_absolute_tolerance"])
    result["gate_results"]["zeta_sd"] = abs(
        exact_sd - float(stats["zeta_expected_sd"])
    ) <= float(gates["zeta_sd_absolute_tolerance"])
    result["calibration_passed"] = all(
        result["gate_results"][name]
        for name in (
            "correct_model_false_rejection_rate",
            "zeta_mean",
            "zeta_sd",
        )
    )
    return result


def _finalize_impl(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
) -> dict[str, Any]:
    # Compute the frozen v2 result first so all unchanged scale, adapter,
    # checkpoint, and backend gates retain their tested implementation.
    core = v2._finalize_impl(
        contract_path,
        correct_manifest_path,
        wrong_manifest_path,
    )
    contract = load_survey_contract(contract_path)
    correct = v2._load_pool(correct_manifest_path, contract, "correct")
    wrong = v2._load_pool(wrong_manifest_path, contract, "wrong")

    full = _evaluate_statistics(correct["rows"], wrong["rows"], contract, label="full")
    replay = _evaluate_statistics(correct["rows"], wrong["rows"], contract, label="full")
    exact_replay_passed = sha256_data(full) == sha256_data(replay)
    invalid_reasons: list[dict[str, str]] = []
    if not full["calibration_passed"]:
        invalid_reasons.append(
            {
                "code": "statistical_calibration_failed",
                "message": "one or more calibration gates failed",
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
        loo = _evaluate_statistics(
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
                "zeta_estimator": loo["zeta_estimator"],
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

    blocked_reasons = list(core["blocked_reasons"])
    if full["power_passed"]:
        blocked_reasons = [
            item for item in blocked_reasons if item["code"] != "insufficient_wrong_model_power"
        ]
    elif not any(item["code"] == "insufficient_wrong_model_power" for item in blocked_reasons):
        blocked_reasons.append(
            {
                "code": "insufficient_wrong_model_power",
                "message": "wrong-model rejection power is below the locked minimum",
            }
        )

    if invalid_reasons:
        verdict = SurveySelectionVerdict.INVALID.value
    elif blocked_reasons:
        verdict = SurveySelectionVerdict.BLOCKED.value
    else:
        verdict = SurveySelectionVerdict.PASSED.value
    core.update(
        {
            "verdict": verdict,
            "statistics": full,
            "exact_replay_passed": exact_replay_passed,
            "seed_block_stability": stability,
            "seed_block_stability_passed": seed_stability_passed,
            "invalid_reasons": invalid_reasons,
            "blocked_reasons": blocked_reasons,
            "evidence_relationship": "CORRECTIVE_REPLAY_OF_V2_OFFICIAL_POOLS_NOT_INDEPENDENT_CONFIRMATION",
        }
    )
    core.pop("replay_sha256", None)
    core["replay_sha256"] = sha256_data(core)
    return core


def finalize_survey_selection(
    contract_path: str | Path,
    correct_manifest_path: str | Path,
    wrong_manifest_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Emit an immutable v3 corrective verdict."""

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
            "nonclaims": list(v2.NONCLAIMS),
            "evidence_relationship": "CORRECTIVE_REPLAY_NOT_COMPLETED",
        }
        result["replay_sha256"] = sha256_data(result)
    v1._atomic_json(target, result)
    return result

