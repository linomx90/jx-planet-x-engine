import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jxplanetx.ensemble_validation import (
    EnsembleValidationError,
    _bootstrap_effect,
    _compare_aggregates,
    finalize_ensemble_validation,
    load_ensemble_plan,
    prepare_ensemble_plan,
    quantile,
    register_ensemble_member,
    wasserstein_1d,
)
from jxplanetx.provenance import runtime_source_manifest, sha256_data, sha256_file


def small_contract() -> dict:
    thresholds = {
        "low_q_fraction": "0.01",
        "injection_fraction": "0.01",
        "survival_fraction": "0.01",
        "mean_q_AU": "0.01",
        "inclination_width_deg": "0.01",
        "wasserstein_q_AU": "0.01",
        "wasserstein_i_deg": "0.01",
    }
    return {
        "schema": "jx-ensemble-contract/v1",
        "experiment_id": "unit-test",
        "purpose": "software verification only",
        "registration_status": "EXPLORATORY",
        "registration_reference": "",
        "evidence_class": "MODEL_OUTPUT",
        "dynamics_model_sha256": sha256_data({"fixture": "dynamics"}),
        "initial_state_model_sha256": sha256_data({"fixture": "initial-state"}),
        "source_model_sha256": sha256_data({"fixture": "source-control"}),
        "seed_blocks": ["block-a", "block-b"],
        "replicates_per_block": 2,
        "tracers_per_replicate": 2,
        "epochs_year": [0, 1],
        "duration_years": 1,
        "frame": "test frame",
        "origin": "test origin",
        "units": "AU, yr",
        "q_threshold_AU": "30",
        "factors": [
            {
                "name": "replicate_phase_deg",
                "scope": "replicate",
                "distribution": "phase",
                "origin": "0",
                "period": "360",
            },
            {
                "name": "tracer_phase_deg",
                "scope": "tracer",
                "distribution": "phase",
                "origin": "0",
                "period": "360",
            },
        ],
        "gaussian_blocks": [],
        "methods": [
            {
                "method_id": "method-a",
                "implementation": "test implementation A",
                "version": "1",
                "independence_group": "algorithm-a",
                "settings": {"tolerance": "tight"},
            },
            {
                "method_id": "method-b",
                "implementation": "test implementation B",
                "version": "1",
                "independence_group": "algorithm-b",
                "settings": {"tolerance": "tight"},
            },
        ],
        "gates": {
            "minimum_blocks": 2,
            "minimum_replicates_per_block": 2,
            "minimum_tracers_per_replicate": 2,
            "minimum_methods": 2,
            "minimum_independence_groups": 2,
            "require_within_group_repeat": False,
            "minimum_bound_samples_per_epoch": 1,
            "method_equivalence": thresholds,
            "repeat_equivalence": thresholds,
            "max_primary_effect_method_disagreement": "0.01",
        },
        "inference": {
            "primary_endpoint": "injection_fraction",
            "confidence_level": "0.90",
            "bootstrap_repetitions": 99,
            "null_equivalence_margin": "0.10",
            "minimum_material_effect": "0.50",
        },
        "power_plan": "tiny deterministic software fixture; not a scientific power claim",
    }


class EnsembleFixture:
    def __init__(self, root: Path, contract: dict | None = None):
        self.root = root
        self.contract_path = root / "contract.json"
        self.plan_path = root / "plan.lock.json"
        self.run_root = root / "records"
        self.contract = contract or small_contract()
        self.runner_source_manifest = runtime_source_manifest()
        self.contract_path.write_text(json.dumps(self.contract, indent=2, sort_keys=True) + "\n")
        self.plan = prepare_ensemble_plan(self.contract_path, self.plan_path)

    def trajectory(
        self,
        member_id: str,
        arm: str,
        method_id: str,
        *,
        final_q: float = 31.0,
        bound: bool = True,
        duplicate: bool = False,
        nan_bound: bool = False,
        reverse: bool = False,
    ) -> Path:
        path = self.root / f"{member_id}-{arm}-{method_id}.csv"
        rows = []
        for epoch in (0, 1):
            for tracer_index in range(2):
                q = 31.0 + tracer_index if epoch == 0 else final_q + tracer_index
                rows.append(
                    {
                        "time_year": epoch,
                        "name": f"t{tracer_index:04d}",
                        "q": "nan" if nan_bound else repr(q),
                        "i_deg": repr(5.0 + tracer_index),
                        "bound": int(bound),
                    }
                )
        if duplicate:
            rows.append(dict(rows[-1]))
        if reverse:
            rows.reverse()
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("time_year", "name", "q", "i_deg", "bound"), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        return path

    def validity(self, member_id: str, arm: str, method_id: str, passed: bool = True) -> Path:
        path = self.root / f"{member_id}-{arm}-{method_id}-validity.json"
        member = next(row for row in self.plan["members"] if row["member_id"] == member_id)
        method = next(row for row in self.plan["contract"]["methods"] if row["method_id"] == method_id)
        trajectory = self.root / f"{member_id}-{arm}-{method_id}.csv"
        value = {
            "schema": "jx-integrator-validity/v1",
            "plan_sha256": self.plan["plan_sha256"],
            "member_id": member_id,
            "arm": arm,
            "method_id": method_id,
            "method_spec_sha256": sha256_data(method),
            "initial_draw_sha256": member["initial_draw_sha256"],
            "dynamics_model_sha256": self.plan["contract"]["dynamics_model_sha256"],
            "initial_state_model_sha256": self.plan["contract"]["initial_state_model_sha256"],
            "source_model_sha256": self.plan["contract"]["source_model_sha256"],
            "relative_initial_state_sha256": sha256_data(
                {"member_id": member_id, "state": "paired-relative-state"}
            ),
            "full_initial_state_sha256": sha256_data(
                {"member_id": member_id, "arm": arm, "state": "full-state"}
            ),
            "trajectory_sha256": sha256_file(trajectory),
            "duration_years": self.plan["contract"]["duration_years"],
            "epochs_year": self.plan["contract"]["epochs_year"],
            "frame": self.plan["contract"]["frame"],
            "origin": self.plan["contract"]["origin"],
            "units": self.plan["contract"]["units"],
            "runner_source_manifest": self.runner_source_manifest,
            "passed": passed,
            "checks": {"finite_state": {"passed": passed}},
        }
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        return path

    def register(
        self,
        member_id: str,
        arm: str,
        method_id: str,
        *,
        final_q: float = 31.0,
        validity_passed: bool = True,
        validity_overrides: dict | None = None,
        **trajectory_options,
    ) -> dict:
        trajectory = self.trajectory(
            member_id,
            arm,
            method_id,
            final_q=final_q,
            **trajectory_options,
        )
        validity = self.validity(member_id, arm, method_id, validity_passed)
        if validity_overrides:
            value = json.loads(validity.read_text())
            value.update(validity_overrides)
            validity.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output = self.run_root / member_id / arm / f"{method_id}.json"
        return register_ensemble_member(
            self.plan_path,
            member_id,
            arm,
            method_id,
            trajectory,
            validity,
            output,
        )

    def complete(self, source_q_by_method: dict[str, float] | None = None) -> None:
        source_q_by_method = source_q_by_method or {}
        for member in self.plan["members"]:
            for arm in ("control", "source"):
                for method in self.plan["contract"]["methods"]:
                    method_id = method["method_id"]
                    final_q = source_q_by_method.get(method_id, 31.0) if arm == "source" else 31.0
                    self.register(member["member_id"], arm, method_id, final_q=final_q)


class DistributionTests(unittest.TestCase):
    def test_wasserstein_known_equal_size(self):
        self.assertEqual(wasserstein_1d([0, 2], [1, 5]), 2.0)

    def test_wasserstein_known_unequal_size(self):
        self.assertAlmostEqual(wasserstein_1d([0, 4], [1, 2, 7]), 2.0)

    def test_wasserstein_is_permutation_invariant_not_paired_distance(self):
        self.assertEqual(wasserstein_1d([0, 100], [100, 0]), 0.0)

    def test_wasserstein_weight_validation(self):
        with self.assertRaisesRegex(EnsembleValidationError, "weights must be positive"):
            wasserstein_1d([0, 1], [0, 1], [1, 0], [1, 1])

    def test_quantile_linear_interpolation(self):
        self.assertAlmostEqual(quantile([0, 10], 0.16), 1.6)
        self.assertAlmostEqual(quantile([0, 10], 0.84), 8.4)

    def test_upper_gate_is_inclusive_and_next_float_above_fails(self):
        def aggregate(q_value: float) -> dict:
            return {
                "injection_fraction": 0.0,
                "survival_fraction": 1.0,
                "minimum_q_values_AU": [q_value],
                "epochs": {
                    "0": {
                        "low_q_fraction": 0.0,
                        "mean_q_AU": q_value,
                        "inclination_width_deg": 0.0,
                        "q_values_AU": [q_value],
                        "i_values_deg": [5.0],
                    }
                },
            }

        thresholds = {
            "low_q_fraction": "0",
            "injection_fraction": "0",
            "survival_fraction": "0",
            "mean_q_AU": "1",
            "inclination_width_deg": "0",
            "wasserstein_q_AU": "1",
            "wasserstein_i_deg": "0",
        }
        exact = _compare_aggregates(aggregate(10.0), aggregate(11.0), [0], thresholds)
        above = _compare_aggregates(
            aggregate(10.0),
            aggregate(math.nextafter(11.0, math.inf)),
            [0],
            thresholds,
        )
        self.assertTrue(exact["passed"])
        self.assertEqual(exact["metrics"]["wasserstein_q_AU"]["value"], 1.0)
        self.assertFalse(above["passed"])
        self.assertFalse(above["metrics"]["mean_q_AU"]["passed"])
        self.assertFalse(above["metrics"]["wasserstein_q_AU"]["passed"])


class PlanTests(unittest.TestCase):
    def test_plan_generation_matches_literal_golden_draw_vector(self):
        """Freeze the cross-run draw algorithm, not merely self-consistency."""
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            first = fixture.plan["members"][0]
            self.assertEqual(
                {
                    "contract_sha256": fixture.plan["contract_sha256"],
                    "member_manifest_sha256": fixture.plan["member_manifest_sha256"],
                    "plan_sha256": fixture.plan["plan_sha256"],
                    "member_id": first["member_id"],
                    "replicate_phase_deg": first["draws"]["replicate_phase_deg"],
                    "initial_draw_sha256": first["initial_draw_sha256"],
                    "tracer_draws": [
                        (row["tracer_id"], row["draws"]["tracer_phase_deg"], row["draw_sha256"])
                        for row in first["tracers"]
                    ],
                },
                {
                    "contract_sha256": "1c175f8f7f0b245b6b9bc6c7497dd26970644a52a09ff6744b95f7fd4acef83a",
                    "member_manifest_sha256": "1c6c697047764546ae077e4ff3083bb34efbeec0d1c46ba63923fe982098d223",
                    "plan_sha256": "79f558f34b24d2994c2971b643f5e6ddb75c029c30c1accf44481cc332fe902a",
                    "member_id": "b00-r0000",
                    "replicate_phase_deg": "232.41394707808703",
                    "initial_draw_sha256": "98feedd18a7f47f139536894c81b83dd694a5afa73e95927e674e323eb489187",
                    "tracer_draws": [
                        (
                            "t0000",
                            "323.68700280925088",
                            "941c4fc9bcdcfd71967c09c6ceeda0437a6b81bbeff5e2189dc073f85c472b5e",
                        ),
                        (
                            "t0001",
                            "31.023682583408696",
                            "c613550313ebdc463638bc8817a0d6c66bdc11f22e3a622ba52dcc7d65ea1e0e",
                        ),
                    ],
                },
            )

    def test_plan_generation_is_byte_reproducible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract = root / "contract.json"
            contract.write_text(json.dumps(small_contract(), indent=2, sort_keys=True) + "\n")
            first = root / "first.json"
            second = root / "second.json"
            one = prepare_ensemble_plan(contract, first)
            two = prepare_ensemble_plan(contract, second)
            self.assertEqual(one, two)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_different_seed_changes_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first_contract = small_contract()
            second_contract = small_contract()
            second_contract["seed_blocks"][1] = "different-block"
            (root / "a.json").write_text(json.dumps(first_contract))
            (root / "b.json").write_text(json.dumps(second_contract))
            first = prepare_ensemble_plan(root / "a.json", root / "a-plan.json")
            second = prepare_ensemble_plan(root / "b.json", root / "b-plan.json")
            self.assertNotEqual(first["member_manifest_sha256"], second["member_manifest_sha256"])

    def test_tampered_plan_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            raw = json.loads(fixture.plan_path.read_text())
            raw["members"][0]["tracers"][0]["draws"]["tracer_phase_deg"] = "0"
            fixture.plan_path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(EnsembleValidationError, "deterministic regeneration"):
                load_ensemble_plan(fixture.plan_path)

    def test_non_positive_definite_covariance_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract = small_contract()
            contract["gaussian_blocks"] = [
                {
                    "name": "bad",
                    "scope": "replicate",
                    "variables": ["x", "y"],
                    "mean": ["0", "0"],
                    "covariance": [[1, 2], [2, 1]],
                }
            ]
            (root / "contract.json").write_text(json.dumps(contract))
            with self.assertRaisesRegex(EnsembleValidationError, "not positive definite"):
                prepare_ensemble_plan(root / "contract.json", root / "plan.json")

    def test_identical_method_configuration_cannot_claim_a_second_independence_group(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract = small_contract()
            contract["methods"][1] = {
                **contract["methods"][0],
                "method_id": "method-b",
                "independence_group": "algorithm-b",
            }
            (root / "contract.json").write_text(json.dumps(contract))
            with self.assertRaisesRegex(EnsembleValidationError, "duplicate an identical numerical configuration"):
                prepare_ensemble_plan(root / "contract.json", root / "plan.json")

    def test_material_effect_boundary_is_inclusive_and_next_float_below_is_not_material(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            at_boundary = {
                member["member_id"]: 0.5
                for member in fixture.plan["members"]
            }
            below_boundary = {
                member["member_id"]: math.nextafter(0.5, -math.inf)
                for member in fixture.plan["members"]
            }
            exact = _bootstrap_effect(fixture.plan, "method-a", at_boundary)
            below = _bootstrap_effect(fixture.plan, "method-a", below_boundary)
            self.assertEqual(exact["confidence_interval"], [0.5, 0.5])
            self.assertEqual(exact["classification"], "MATERIAL_POSITIVE")
            self.assertEqual(below["classification"], "INCONCLUSIVE")


class StrictRegistrationTests(unittest.TestCase):
    def test_validity_binds_trajectory_method_and_locked_scope(self):
        mismatches = {
            "trajectory_sha256": "f" * 64,
            "method_spec_sha256": "f" * 64,
            "frame": "wrong frame",
        }
        for field, bad_value in mismatches.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                fixture = EnsembleFixture(Path(folder))
                member = fixture.plan["members"][0]["member_id"]
                with self.assertRaises(EnsembleValidationError) as caught:
                    fixture.register(
                        member,
                        "control",
                        "method-a",
                        validity_overrides={field: bad_value},
                    )
                self.assertEqual(caught.exception.code, "validity_scope_mismatch")

    def test_duplicate_row_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            member = fixture.plan["members"][0]["member_id"]
            with self.assertRaisesRegex(EnsembleValidationError, "duplicate row"):
                fixture.register(member, "control", "method-a", duplicate=True)

    def test_nonfinite_bound_value_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            member = fixture.plan["members"][0]["member_id"]
            with self.assertRaisesRegex(EnsembleValidationError, "must be finite"):
                fixture.register(member, "control", "method-a", nan_bound=True)

    def test_row_order_changes_raw_hash_not_semantic_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            member = fixture.plan["members"][0]["member_id"]
            first = fixture.register(member, "control", "method-a")
            trajectory = fixture.trajectory(member, "source", "method-a", reverse=True)
            validity = fixture.validity(member, "source", "method-a")
            second = register_ensemble_member(
                fixture.plan_path,
                member,
                "source",
                "method-a",
                trajectory,
                validity,
                fixture.run_root / member / "source" / "method-a.json",
            )
            self.assertNotEqual(first["payload"]["trajectory_sha256"], second["payload"]["trajectory_sha256"])
            self.assertEqual(
                first["payload"]["trajectory_semantic_sha256"],
                second["payload"]["trajectory_semantic_sha256"],
            )

    def test_q_equal_threshold_is_not_an_injection(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            member = fixture.plan["members"][0]["member_id"]
            record = fixture.register(member, "control", "method-a", final_q=30.0)
            self.assertEqual(record["payload"]["summary"]["injection_count"], 0)


class FinalizationTests(unittest.TestCase):
    def test_complete_equivalent_ensemble_passes_but_stays_screening_only(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "PASSED")
            self.assertEqual(result["effect_classification"], "PRACTICALLY_EQUIVALENT")
            self.assertEqual(result["claim_decision"], "SCREENING_ONLY")
            self.assertEqual(result["evidence_class"], "MODEL_OUTPUT")

    def test_passed_result_exposes_population_summaries_and_source_control_wasserstein(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete({"method-a": 30.5, "method-b": 30.5})
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "PASSED")
            self.assertEqual(len(result["population_summaries"]), 4)
            control = next(
                row
                for row in result["population_summaries"]
                if row["method_id"] == "method-a" and row["arm"] == "control"
            )
            self.assertEqual(control["tracer_count"], 8)
            self.assertEqual(control["minimum_q_distribution_AU"]["median"], 31.5)
            distribution = next(
                row
                for row in result["source_control_distributions"]
                if row["method_id"] == "method-a"
            )
            self.assertEqual(distribution["wasserstein_minimum_q_AU"], 0.5)
            final = next(row for row in distribution["epochs"] if row["epoch_year"] == 1)
            self.assertEqual(final["source_minus_control_mean_q_AU"], -0.5)
            self.assertEqual(final["wasserstein_q_AU"], 0.5)
            self.assertEqual(final["wasserstein_i_deg"], 0.0)

    def test_cross_method_relative_and_full_state_mismatches_are_invalid(self):
        expected_codes = {
            "relative_initial_state_sha256": "initial_state_pair_mismatch",
            "full_initial_state_sha256": "initial_state_method_mismatch",
        }
        for field, expected_code in expected_codes.items():
            with self.subTest(field=field), tempfile.TemporaryDirectory() as folder:
                fixture = EnsembleFixture(Path(folder))
                target = fixture.plan["members"][0]["member_id"]
                replacement = sha256_data({"field": field, "fixture": "cross-method-mismatch"})
                for member in fixture.plan["members"]:
                    for arm in ("control", "source"):
                        for method in fixture.plan["contract"]["methods"]:
                            method_id = method["method_id"]
                            overrides = None
                            if member["member_id"] == target and arm == "control" and method_id == "method-b":
                                overrides = {field: replacement}
                            fixture.register(
                                member["member_id"],
                                arm,
                                method_id,
                                validity_overrides=overrides,
                            )
                result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
                self.assertEqual(result["verdict"], "INVALID")
                self.assertEqual(result["claim_decision"], "INVALID")
                self.assertIn(expected_code, {row["code"] for row in result["invalid_reasons"]})

    def test_precision_disagreement_is_invalid_but_independent_disagreement_is_conflict(self):
        with self.subTest(comparison="same independence group"), tempfile.TemporaryDirectory() as folder:
            contract = small_contract()
            contract["methods"][1]["independence_group"] = "algorithm-a"
            contract["gates"]["minimum_independence_groups"] = 1
            contract["gates"]["require_within_group_repeat"] = True
            fixture = EnsembleFixture(Path(folder), contract)
            fixture.complete({"method-a": 29.0, "method-b": 31.0})
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["claim_decision"], "INVALID")
            self.assertIn("precision_nonconvergence", {row["code"] for row in result["invalid_reasons"]})
            self.assertIn(
                "WITHIN_GROUP_PRECISION",
                {row["comparison_type"] for row in result["method_comparisons"]},
            )

        with self.subTest(comparison="different independence groups"), tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete({"method-a": 29.0, "method-b": 31.0})
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "BLOCKED")
            self.assertEqual(result["claim_decision"], "CONFLICT")
            self.assertIn("independent_method_disagreement", {row["code"] for row in result["blocked_reasons"]})
            self.assertIn(
                "INDEPENDENT_METHOD",
                {row["comparison_type"] for row in result["method_comparisons"]},
            )

    def test_missing_run_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "BLOCKED")
            self.assertIn("missing_required_runs", {row["code"] for row in result["blocked_reasons"]})

    def test_failed_integrator_validity_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            record_path = next(fixture.run_root.rglob("*.json"))
            raw = json.loads(record_path.read_text())
            raw["payload"]["validity_passed"] = False
            raw["payload_sha256"] = sha256_data(raw["payload"])
            record_path.write_text(json.dumps(raw))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["claim_decision"], "INVALID")

    def test_malformed_member_record_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            record_path = next(fixture.run_root.rglob("*.json"))
            raw = json.loads(record_path.read_text())
            del raw["payload"]["summary"]
            raw["payload_sha256"] = sha256_data(raw["payload"])
            record_path.write_text(json.dumps(raw))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["claim_decision"], "INVALID")
            self.assertIn("missing_field", {row["code"] for row in result["invalid_reasons"]})

    def test_deeply_nested_member_json_returns_invalid_instead_of_recursion_error(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            member = fixture.plan["members"][0]["member_id"]
            fixture.register(member, "control", "method-a")
            record_path = fixture.run_root / member / "control" / "method-a.json"
            record_path.write_text("[" * 10000 + "0" + "]" * 10000)
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertIn("invalid_member_json", {row["code"] for row in result["invalid_reasons"]})

    def test_rehashed_nonfinite_member_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            record_path = next(fixture.run_root.rglob("*.json"))
            raw = json.loads(record_path.read_text())
            # A string survives strict JSON encoding and hashing, but still
            # represents a nonfinite numeric value to a careless float parser.
            raw["payload"]["summary"]["epochs"]["1"]["q_values_AU"][0] = "nan"
            raw["payload"]["summary_sha256"] = sha256_data(raw["payload"]["summary"])
            raw["payload_sha256"] = sha256_data(raw["payload"])
            record_path.write_text(json.dumps(raw))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["claim_decision"], "INVALID")
            self.assertIn("nonfinite_value", {row["code"] for row in result["invalid_reasons"]})

    def test_rehashed_injection_evidence_without_matching_q_history_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            record_path = next(fixture.run_root.glob("*/source/*.json"))
            raw = json.loads(record_path.read_text())
            summary = raw["payload"]["summary"]
            summary["injection_count"] = summary["tracer_count"]
            summary["injection_fraction"] = 1.0
            summary["minimum_q_values_AU"] = [29.0] * summary["tracer_count"]
            for outcome in summary["tracer_outcomes"]:
                outcome["minimum_q_AU"] = 29.0
                outcome["injected"] = True
                outcome["injection_epoch_year"] = 1
                outcome["injection_from_q_AU"] = outcome["initial_q_AU"]
                outcome["injection_to_q_AU"] = 29.0
            raw["payload"]["summary_sha256"] = sha256_data(summary)
            raw["payload_sha256"] = sha256_data(raw["payload"])
            record_path.write_text(json.dumps(raw))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertIn("summary_consistency_error", {row["code"] for row in result["invalid_reasons"]})

    def test_unplanned_extra_member_record_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            record_path = next(fixture.run_root.rglob("*.json"))
            raw = json.loads(record_path.read_text())
            raw["payload"]["member_id"] = "unplanned-member"
            raw["payload_sha256"] = sha256_data(raw["payload"])
            extra_path = fixture.run_root / "unplanned-member" / "control" / "method-a.json"
            extra_path.parent.mkdir(parents=True)
            extra_path.write_text(json.dumps(raw))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["claim_decision"], "INVALID")

    def test_member_draw_support_mismatch_is_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.complete()
            record_path = next(fixture.run_root.rglob("*.json"))
            raw = json.loads(record_path.read_text())
            raw["payload"]["initial_draw_sha256"] = "0" * 64
            raw["payload_sha256"] = sha256_data(raw["payload"])
            record_path.write_text(json.dumps(raw))
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "INVALID")
            self.assertEqual(result["claim_decision"], "INVALID")
            self.assertIn("draw_hash_mismatch", {row["code"] for row in result["invalid_reasons"]})

    def test_opposite_method_effect_classification_is_conflict(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            fixture.contract["gates"]["method_equivalence"] = {
                name: "10" for name in fixture.contract["gates"]["method_equivalence"]
            }
            fixture.contract["gates"]["max_primary_effect_method_disagreement"] = "2"
            # Recreate the lock with the modified preregistration.
            other = Path(folder) / "conflict"
            other.mkdir()
            fixture = EnsembleFixture(other, fixture.contract)
            fixture.complete({"method-a": 29.0, "method-b": 31.0})
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "BLOCKED")
            self.assertEqual(result["claim_decision"], "CONFLICT")
            self.assertIn("effect_classification_conflict", {row["code"] for row in result["blocked_reasons"]})

    def test_all_unbound_conditional_distribution_is_blocked(self):
        with tempfile.TemporaryDirectory() as folder:
            fixture = EnsembleFixture(Path(folder))
            for member in fixture.plan["members"]:
                for arm in ("control", "source"):
                    for method in ("method-a", "method-b"):
                        fixture.register(member["member_id"], arm, method, bound=False)
            result = finalize_ensemble_validation(fixture.plan_path, fixture.run_root)
            self.assertEqual(result["verdict"], "BLOCKED")
            self.assertIn("insufficient_bound_population", {row["code"] for row in result["blocked_reasons"]})


class EnsembleCliTests(unittest.TestCase):
    def test_template_then_prepare_cli_produces_loadable_locked_plan(self):
        repository = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(repository / "src"), environment.get("PYTHONPATH")))
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            contract = root / "contract.json"
            plan_path = root / "plan.lock.json"
            write = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jxplanetx.cli",
                    "write-ensemble-contract",
                    "--output",
                    str(contract),
                ],
                cwd=repository,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(write.returncode, 0, write.stderr)
            prepare = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "jxplanetx.cli",
                    "prepare-ensemble",
                    "--contract",
                    str(contract),
                    "--output",
                    str(plan_path),
                ],
                cwd=repository,
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            self.assertEqual(prepare.returncode, 0, prepare.stderr)
            plan = load_ensemble_plan(plan_path)
            self.assertEqual(len(plan["members"]), 64)
            self.assertEqual(sum(len(member["tracers"]) for member in plan["members"]), 4096)


if __name__ == "__main__":
    unittest.main()
