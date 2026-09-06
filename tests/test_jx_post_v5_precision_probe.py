"""Fail-closed tests for the additive post-V5 precision probe.

The default suite exercises the one-period smoke profile.  The exact external
REBOUND lane remains an explicit opt-in because it requires retained 5.1.1
bytes that are intentionally not vendored by this repository.
"""

from __future__ import annotations

import base64
import copy
import decimal
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "jx_post_v5_precision_probe.py"
SPEC = importlib.util.spec_from_file_location("jx_post_v5_precision_probe", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("post-V5 precision probe could not be loaded")
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)

RUN_REBOUND = os.environ.get("JX_RUN_POST_V5_REBOUND_5_1_1") == "1"

FROZEN_V5_FILES = {
    "benchmarks/jx_challenger_v1.py": (
        98_050,
        "0a9f00ae4dd9ba5279a0289af2a0bf0d0394e2ee8441793a6f07896de2ac43a8",
    ),
    "tests/test_jx_challenger_v1.py": (
        35_253,
        "2851cb240965b17740928858a430d8229e1437c543599b99c682b9f7f79e1b26",
    ),
    "docs/JX_CHALLENGER_V1.md": (
        7_345,
        "48c46f895c33b5aa77ebd16485d9aba04c7871c0a9d9e529320c47e2b74398ba",
    ),
}

_SMOKE_REPORT: dict[str, object] | None = None


def _smoke_report() -> dict[str, object]:
    global _SMOKE_REPORT
    if _SMOKE_REPORT is None:
        _SMOKE_REPORT = probe.run_precision_probe(
            probe.PrecisionProfile("smoke"), "disabled"
        )
    return _SMOKE_REPORT


def _canonical(report: dict[str, object]) -> bytes:
    return json.dumps(
        report,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reseal(report: dict[str, object]) -> dict[str, object]:
    candidate = copy.deepcopy(report)
    candidate.pop("content_integrity", None)
    return probe._attach_integrity(candidate)


def _all_mapping_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if type(value) is dict:
        for key, nested in value.items():
            keys.add(key)
            keys.update(_all_mapping_keys(nested))
    elif type(value) is list:
        for nested in value:
            keys.update(_all_mapping_keys(nested))
    return keys


class SmokeReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.profile = probe.PrecisionProfile("smoke")
        cls.report = _smoke_report()

    def test_exact_three_jx_lanes_replay_and_work_ledgers(self):
        report = self.report
        self.assertEqual(report["profile"], {
            "name": "smoke",
            "periods": 1,
            "samples_per_period": 16,
            "checkpoint_count": 17,
            "step_divisors": [256, 512, 1024],
            "cli_full_profile_cost_acknowledgement_required": True,
        })
        self.assertEqual(
            [lane["lane_id"] for lane in report["lanes"]],
            ["jx_kdk_p256", "jx_kdk_p512", "jx_kdk_p1024"],
        )
        self.assertEqual(report["external_status"], {
            "requested_mode": "disabled",
            "required_version": "5.1.1",
            "external_lanes_executed": False,
            "reason": "DISABLED_BY_CALLER",
            "external_code_vendored": False,
            "authority_authorized": False,
        })
        for divisor, lane in zip(probe.STEP_DIVISORS, report["lanes"], strict=True):
            with self.subTest(divisor=divisor):
                self.assertEqual(lane["family"], "jx_kdk")
                self.assertEqual(lane["role"], "JX_FIXED_GRID")
                self.assertEqual(lane["method_id"], probe.FIXED_STEP_KDK_METHOD_ID)
                self.assertEqual(lane["divisor"], divisor)
                self.assertEqual(lane["fixed_step_hex"], (probe.PERIOD / divisor).hex())
                self.assertEqual(lane["replay"], {
                    "scope": "SAME_RUNTIME_COMPLETE_FRESH_STATE_REEXECUTION",
                    "raw_state_sha256": lane["raw_state_evidence"]["sha256"],
                    "status": "BITWISE_IDENTICAL",
                    "process_independence_claimed": False,
                })
                accounting = lane["accounting"]
                self.assertEqual(accounting["checkpoint_count"], 17)
                self.assertEqual(accounting["completed_steps"], divisor)
                self.assertEqual(accounting["force_evaluations"], divisor + 1)
                self.assertEqual(accounting["engine_internal_trajectory_reexecution_count"], 0)
                self.assertIs(accounting["engine_result_constructor_validation_performed"], True)
                self.assertEqual(
                    accounting["effective_settings"],
                    probe._expected_jx_effective_settings(self.profile, divisor),
                )
                self.assertRegex(accounting["schedule_content_sha256"], r"^[0-9a-f]{64}$")
                self.assertRegex(accounting["result_content_sha256"], r"^[0-9a-f]{64}$")

    def test_both_accuracy_bases_have_complete_monotonic_convergence(self):
        family = self.report["convergence"]["jx_kdk"]
        self.assertEqual(
            family["primary_accuracy_basis"],
            "DECLARED_GRID_FIXED_HORIZON_END_TO_END_ACCURACY",
        )
        self.assertEqual(
            family["secondary_accuracy_basis"],
            "ANALYTIC_STATE_AT_EACH_LANE_OBSERVED_CLOCK_CONDITIONAL_ON_EFFECTIVE_READBACK_TIME",
        )
        expected_metrics = {
            "position_vector_l2_maximum",
            "position_vector_l2_rms",
            "velocity_vector_l2_maximum",
            "velocity_vector_l2_rms",
            "phase_maximum_abs",
            "phase_rms",
        }
        for label, expected_basis in (
            ("observed_clock_analytic", "observed_clock_analytic_accuracy"),
            ("declared_grid_delivery_aware", "declared_grid_analytic_accuracy"),
        ):
            with self.subTest(basis=label):
                convergence = family[label]
                self.assertEqual(convergence["accuracy_basis"], expected_basis)
                self.assertEqual(
                    convergence["status"],
                    "OBSERVED_WITHIN_EXACT_FIXTURE_REFINEMENT_GATES",
                )
                self.assertEqual(
                    convergence["absolute_threshold_scope"],
                    "RECORDED_NOT_ENFORCED_SMOKE",
                )
                self.assertEqual(set(convergence["metric_refinement"]), expected_metrics)
                for metric, record in convergence["metric_refinement"].items():
                    values = record["values_p256_p512_p1024"]
                    self.assertEqual(len(values), 3)
                    self.assertGreater(values[0], values[1])
                    self.assertGreater(values[1], values[2])
                    self.assertIs(record["strictly_monotonic"], True)
                    self.assertEqual(len(record["adjacent_orders"]), 2)
                    self.assertTrue(
                        all(1.8 <= order <= 2.2 for order in record["adjacent_orders"])
                    )
                    self.assertIn("STATE_OR_PHASE", record["interpretation"])
                thresholds = convergence["finest_level_thresholds"]
                self.assertEqual(thresholds["phase_maximum_abs"], 1.0e-2)
                self.assertEqual(thresholds["position_vector_l2_maximum"], 5.0e-3)
                self.assertEqual(thresholds["velocity_vector_l2_maximum"], 5.0e-3)

        invariants = family["sampled_invariants"]
        self.assertEqual(
            invariants["status"],
            "OBSERVED_WITHIN_EXACT_FIXTURE_SAMPLED_INVARIANT_GATES",
        )
        self.assertEqual(
            invariants["sampling_scope"],
            "RETAINED_OUTPUT_CHECKPOINTS_ONLY_NOT_CONTINUOUS_TIME",
        )
        self.assertEqual(
            set(invariants["metric_refinement"]),
            {
                "sampled_relative_energy_maximum_abs",
                "sampled_relative_energy_rms",
            },
        )
        for record in invariants["metric_refinement"].values():
            values = record["values_p256_p512_p1024"]
            self.assertGreater(values[0], values[1])
            self.assertGreater(values[1], values[2])
            self.assertTrue(all(order >= 1.8 for order in record["adjacent_orders"]))
            self.assertIn("SUPERCONVERGENCE", record["interpretation"])
        invariant_thresholds = invariants["finest_level_thresholds"]
        self.assertEqual(
            invariant_thresholds["sampled_relative_energy_maximum_abs"], 1.0e-9
        )
        for field in (
            "relative_angular_momentum_vector_norm_maximum",
            "relative_angular_momentum_z_signed_maximum_abs",
            "center_of_mass_position_norm_maximum",
            "total_momentum_norm_maximum",
        ):
            self.assertEqual(invariant_thresholds[field], 1.0e-12)

        # JX reports exact delivery on the declared lattice, so the two oracle
        # bases must be identical rather than merely numerically close.
        for lane in self.report["lanes"]:
            metrics = lane["metrics"]
            self.assertEqual(
                metrics["observed_clock_analytic_accuracy"],
                metrics["declared_grid_analytic_accuracy"],
            )
            self.assertEqual(metrics["clock_error"]["maximum_abs"], 0.0)

    def test_metric_semantics_include_signed_samples_and_per_orbit_ols(self):
        for lane in self.report["lanes"]:
            energy = lane["metrics"]["invariants"]["relative_total_energy"]
            phase = lane["metrics"]["observed_clock_analytic_accuracy"]["phase"]
            for metric in (energy, phase):
                self.assertEqual(
                    set(metric).issuperset(
                        {
                            "maximum_abs",
                            "rms",
                            "final_signed",
                            "minimum_signed",
                            "maximum_signed",
                            "peak_to_peak",
                            "ols_slope_per_orbit",
                        }
                    ),
                    True,
                )
                for field in (
                    "maximum_abs",
                    "rms",
                    "final_signed",
                    "minimum_signed",
                    "maximum_signed",
                    "peak_to_peak",
                    "ols_slope_per_orbit",
                ):
                    self.assertIs(type(metric[field]), float)
                    self.assertTrue(math.isfinite(metric[field]))
            invariants = lane["metrics"]["invariants"]
            self.assertEqual(invariants["sampling_scope"], "RETAINED_OUTPUT_CHECKPOINTS_ONLY")
            self.assertEqual(invariants["energy_baseline"], probe.ENERGY_0)

    def test_claim_ceiling_is_strict_and_report_contains_no_ranking_key(self):
        self.assertEqual(self.report["claim_controls"], probe.CLAIM_CONTROLS)
        self.assertTrue(self.report["claim_controls"])
        self.assertTrue(all(value is False for value in self.report["claim_controls"].values()))
        self.assertIs(self.report["claim_controls"]["pure_python_runtime_claimed"], False)
        self.assertEqual(
            self.report["serialization"]["raw_state_regression_kat_scope"],
            "EXACT_PROFILE_SOURCE_ENGINE_NUMPY_AND_OPTIONAL_REBOUND_BUILD",
        )
        interpretation = self.report["scientific_interpretation"]
        self.assertIs(
            interpretation["sampled_energy_conservation_implies_phase_or_state_correctness"],
            False,
        )
        self.assertIs(interpretation["energy_slope_is_a_physical_secular_drift"], False)
        self.assertIs(interpretation["dyadic_ladder_rules_out_step_size_resonance"], False)
        self.assertFalse(
            _all_mapping_keys(self.report)
            & {"winner", "speedup", "defeated", "production_ready", "qualified"}
        )

    def test_raw_matrices_round_trip_exactly_and_tampering_is_rejected(self):
        analytic = probe._decode_matrix_evidence(
            self.report["analytic_reference"],
            expected_rows=17,
            expected_columns=probe.ANALYTIC_MATRIX_COLUMNS,
        )
        self.assertEqual(analytic.shape, (17, 13))
        self.assertEqual(analytic.dtype.str, "<f8")
        for lane in self.report["lanes"]:
            evidence = lane["raw_state_evidence"]
            matrix = probe._decode_matrix_evidence(
                evidence,
                expected_rows=17,
                expected_columns=probe.MATRIX_COLUMNS,
            )
            self.assertEqual(matrix.shape, (17, 14))
            np.testing.assert_array_equal(matrix[:, 0], matrix[:, 1])
            self.assertEqual(probe._matrix_evidence(matrix, probe.MATRIX_COLUMNS), evidence)

            tampered = copy.deepcopy(evidence)
            raw = bytearray(base64.b64decode(tampered["data_base64"], validate=True))
            raw[-1] ^= 1
            tampered["data_base64"] = base64.b64encode(raw).decode("ascii")
            with self.assertRaisesRegex(probe.PrecisionProbeError, "bytes changed"):
                probe._decode_matrix_evidence(
                    tampered,
                    expected_rows=17,
                    expected_columns=probe.MATRIX_COLUMNS,
                )

            relabeled = copy.deepcopy(evidence)
            relabeled["columns"][0] = "other_epoch"
            with self.assertRaisesRegex(probe.PrecisionProbeError, "metadata changed"):
                probe._decode_matrix_evidence(
                    relabeled,
                    expected_rows=17,
                    expected_columns=probe.MATRIX_COLUMNS,
                )

    def test_semantic_digest_excludes_only_diagnostics_and_tags_numbers_exactly(self):
        original = self.report["content_integrity"]["semantic_content_sha256"]
        changed_timing = copy.deepcopy(self.report)
        changed_timing["execution_diagnostics"]["lane_timings"][0][
            "primary_wall_seconds"
        ] += 10.0
        self.assertEqual(probe._semantic_sha256(changed_timing), original)
        probe.validate_report(changed_timing)

        changed_metric = copy.deepcopy(self.report)
        metric = changed_metric["lanes"][0]["metrics"]["invariants"][
            "relative_total_energy"
        ]
        metric["final_signed"] = math.nextafter(metric["final_signed"], math.inf)
        self.assertNotEqual(probe._semantic_sha256(changed_metric), original)
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(changed_metric)

        signed_zero = copy.deepcopy(self.report)
        signed_zero["fixture"]["initial_positions"][0][1] = -0.0
        self.assertNotEqual(probe._semantic_sha256(signed_zero), original)
        with self.assertRaisesRegex(
            probe.PrecisionProbeError, "fixture changed|semantic integrity"
        ):
            probe.validate_report(signed_zero)

        self.assertNotEqual(probe._semantic_sha256({"x": 1}), probe._semantic_sha256({"x": 1.0}))
        self.assertNotEqual(probe._semantic_sha256({"x": True}), probe._semantic_sha256({"x": 1}))
        self.assertNotEqual(probe._semantic_sha256({"x": 0.0}), probe._semantic_sha256({"x": -0.0}))

    def test_decimal_oracle_is_isolated_from_the_process_global_context(self):
        self.assertEqual(
            self.report["analytic_oracle"], probe._analytic_oracle_contract()
        )
        expected = probe._decimal_sin_cos(0.7)
        original = decimal.getcontext().copy()
        try:
            decimal.getcontext().prec = 6
            decimal.getcontext().rounding = decimal.ROUND_DOWN
            observed = probe._decimal_sin_cos(0.7)
            self.assertEqual(decimal.getcontext().prec, 6)
            self.assertEqual(decimal.getcontext().rounding, decimal.ROUND_DOWN)
        finally:
            decimal.setcontext(original)
        self.assertEqual(observed, expected)

        relabeled = copy.deepcopy(self.report)
        relabeled["analytic_oracle"]["decimal_rounding"] = "ROUND_DOWN"
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(_reseal(relabeled))

    def test_fully_resealed_signed_zero_raw_and_analytic_substitutions_fail(self):
        analytic_attack = copy.deepcopy(self.report)
        analytic = probe._decode_matrix_evidence(
            analytic_attack["analytic_reference"],
            expected_rows=17,
            expected_columns=probe.ANALYTIC_MATRIX_COLUMNS,
        )
        self.assertEqual(analytic[0, 2], 0.0)
        self.assertFalse(np.signbit(analytic[0, 2]))
        analytic[0, 2] = -0.0
        analytic_attack["analytic_reference"] = probe._matrix_evidence(
            analytic, probe.ANALYTIC_MATRIX_COLUMNS
        )
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(_reseal(analytic_attack))

        lane_attack = copy.deepcopy(self.report)
        lane = lane_attack["lanes"][0]
        matrix = probe._decode_matrix_evidence(
            lane["raw_state_evidence"],
            expected_rows=17,
            expected_columns=probe.MATRIX_COLUMNS,
        )
        self.assertEqual(matrix[1, 4], 0.0)
        self.assertFalse(np.signbit(matrix[1, 4]))
        matrix[1, 4] = -0.0
        lane["raw_state_evidence"] = probe._matrix_evidence(
            matrix, probe.MATRIX_COLUMNS
        )
        lane["metrics"] = probe._lane_metrics(matrix)
        lane["replay"]["raw_state_sha256"] = lane["raw_state_evidence"]["sha256"]
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(_reseal(lane_attack))

    def test_fully_resealed_json_type_substitutions_fail(self):
        fixture_attack = copy.deepcopy(self.report)
        fixture_attack["fixture"]["G"] = 1
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(_reseal(fixture_attack))

        claim_attack = copy.deepcopy(self.report)
        claim_attack["claim_controls"]["superiority_claimed"] = 0
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(_reseal(claim_attack))


class StrictJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _smoke_report()
        cls.canonical = _canonical(cls.report)

    def test_canonical_report_with_optional_single_newline_loads(self):
        self.assertEqual(probe.load_report_bytes(self.canonical), self.report)
        self.assertEqual(probe.load_report_bytes(self.canonical + b"\n"), self.report)

    def test_duplicate_nonfinite_huge_number_depth_and_trailing_are_rejected(self):
        cases = (
            (b'{"a":1,"a":2}', "duplicate"),
            (b'{"x":NaN}', "nonfinite"),
            (b'{"x": ' + (b"9" * 1_250) + b"}", "integer.*exceeds"),
            (
                json.dumps({"x": [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]})
                .encode("ascii"),
                "depth|cap",
            ),
            (self.canonical + b" trailing", "strict ASCII JSON"),
        )
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(
                probe.PrecisionProbeError, message
            ):
                probe.load_report_bytes(data)

    def test_noncanonical_and_over_cap_inputs_are_rejected_before_use(self):
        for data in (b'{"b":1,"a":2}', b'{ "a": 1 }'):
            with self.subTest(data=data), self.assertRaisesRegex(
                probe.PrecisionProbeError, "not canonical"
            ):
                probe.load_report_bytes(data)
        with self.assertRaisesRegex(probe.PrecisionProbeError, "exceed the cap"):
            probe.load_report_bytes(b"x" * (probe.MAX_REPORT_BYTES + 1))
        with self.assertRaisesRegex(probe.PrecisionProbeError, "empty"):
            probe.load_report_bytes(b"")


class ValidatorMutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = _smoke_report()

    def test_resealed_role_method_runtime_settings_accounting_and_source_relabels_fail(self):
        def role(report):
            report["lanes"][0]["role"] = "OPTIONAL_EXTERNAL_COMPARATOR"

        def method(report):
            report["lanes"][0]["method_id"] = "other.method"

        def runtime(report):
            report["lanes"][0]["runtime"]["jxplanetx_version"] = "other"

        def fixed_step_label(report):
            report["lanes"][0]["fixed_step_hex"] = float(probe.PERIOD / 512).hex()

        def effective_settings(report):
            report["lanes"][0]["accounting"]["effective_settings"]["backend"][
                "tile_size"
            ] = 4

        def accounting(report):
            report["lanes"][0]["accounting"]["completed_steps"] += 1

        def source(report):
            report["sources"]["analytic_support_source"]["relative_path"] = "other.py"

        def family(report):
            report["lanes"][0]["family"] = "rebound_leapfrog"

        for label, mutation in (
            ("role", role),
            ("method", method),
            ("runtime", runtime),
            ("fixed-step label", fixed_step_label),
            ("effective settings", effective_settings),
            ("accounting", accounting),
            ("source", source),
            ("family", family),
        ):
            with self.subTest(label=label):
                candidate = copy.deepcopy(self.report)
                mutation(candidate)
                candidate = _reseal(candidate)
                with self.assertRaises(probe.PrecisionProbeError):
                    probe.validate_report(candidate)

    def test_resealed_raw_state_and_convergence_relabels_fail(self):
        candidate = copy.deepcopy(self.report)
        candidate["lanes"][0]["raw_state_evidence"]["sha256"] = "0" * 64
        with self.assertRaises(probe.PrecisionProbeError):
            probe.validate_report(_reseal(candidate))

        candidate = copy.deepcopy(self.report)
        candidate["convergence"]["jx_kdk"]["observed_clock_analytic"]["status"] = "OTHER"
        with self.assertRaisesRegex(probe.PrecisionProbeError, "convergence"):
            probe.validate_report(_reseal(candidate))

    def test_disabled_external_state_cannot_be_resealed_as_executed(self):
        candidate = copy.deepcopy(self.report)
        candidate["external_status"]["external_lanes_executed"] = True
        candidate["external_status"]["reason"] = "EXACT_REBOUND_5_1_1_EXECUTED"
        with self.assertRaisesRegex(probe.PrecisionProbeError, "external execution state"):
            probe.validate_report(_reseal(candidate))

    def test_resealed_native_schedule_and_result_digest_substitutions_fail(self):
        for field, replacement in (
            ("schedule_content_sha256", "0" * 64),
            ("result_content_sha256", "1" * 64),
        ):
            with self.subTest(field=field):
                candidate = copy.deepcopy(self.report)
                candidate["lanes"][0]["accounting"][field] = replacement
                with self.assertRaises(probe.PrecisionProbeError):
                    probe.validate_report(_reseal(candidate))


class CliAndPublicationTests(unittest.TestCase):
    def test_full_profile_requires_explicit_cost_acknowledgement(self):
        with self.assertRaisesRegex(probe.PrecisionProbeError, "acknowledge-full-cost"):
            probe.main(["--profile", "full-100-period"])

        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONPYCACHEPREFIX"] = str(Path(tempfile.gettempdir()) / "jx-post-v5-cli-pyc")
        completed = subprocess.run(
            (sys.executable, "-B", str(SCRIPT), "--profile", "full-100-period"),
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, "")
        self.assertIn("requires --acknowledge-full-cost", completed.stderr)

    def test_full_cost_acknowledgement_reaches_the_full_profile_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "not-really-written.json"
            with (
                mock.patch.object(
                    probe,
                    "run_precision_probe",
                    return_value={"synthetic": "test-only"},
                ) as runner,
                mock.patch.object(probe, "_write_new_file") as publisher,
            ):
                status = probe.main(
                    [
                        "--profile",
                        "full-100-period",
                        "--acknowledge-full-cost",
                        "--rebound-mode",
                        "disabled",
                        "--output",
                        str(target),
                    ]
                )
        self.assertEqual(status, 0)
        runner.assert_called_once_with(
            probe.PrecisionProfile("full-100-period"), "disabled"
        )
        published_path, published_bytes = publisher.call_args.args
        self.assertEqual(published_path, target)
        self.assertEqual(
            published_bytes,
            json.dumps(
                {"synthetic": "test-only"},
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n",
        )

    def test_help_is_side_effect_free_and_documents_the_acknowledgement(self):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT / "src")
        environment["PYTHONPYCACHEPREFIX"] = str(Path(tempfile.gettempdir()) / "jx-post-v5-help-pyc")
        completed = subprocess.run(
            (sys.executable, "-B", str(SCRIPT), "--help"),
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--acknowledge-full-cost", completed.stdout)
        self.assertIn("--rebound-mode", completed.stdout)

    def test_output_is_create_only_and_never_clobbers(self):
        payload = b'{"evidence":true}\n'
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "probe.json"
            probe._write_new_file(target, payload)
            self.assertEqual(target.read_bytes(), payload)
            with self.assertRaisesRegex(probe.PrecisionProbeError, "must not exist"):
                probe._write_new_file(target, b"replacement")
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(
                sorted(path.name for path in Path(directory).iterdir()),
                ["probe.json"],
            )

    def test_publication_failure_leaves_no_target_or_stage_file(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "probe.json"
            with mock.patch.object(probe.os, "link", side_effect=OSError("injected")):
                with self.assertRaisesRegex(probe.PrecisionProbeError, "atomically published"):
                    probe._write_new_file(target, b"payload")
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_post_link_directory_fsync_failure_rolls_back_the_target(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "probe.json"
            with mock.patch.object(
                probe.os,
                "fsync",
                side_effect=(None, OSError("injected directory fsync"), None),
            ):
                with self.assertRaisesRegex(
                    probe.PrecisionProbeError, "atomically published"
                ):
                    probe._write_new_file(target, b"payload")
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])


class FrozenBoundaryTests(unittest.TestCase):
    def test_frozen_v5_and_engine_source_hashes_are_unchanged(self):
        for relative, (size, expected_sha256) in FROZEN_V5_FILES.items():
            with self.subTest(relative=relative):
                data = (ROOT / relative).read_bytes()
                self.assertEqual(len(data), size)
                self.assertEqual(hashlib.sha256(data).hexdigest(), expected_sha256)
        self.assertEqual(
            probe._engine_source_manifest()["tree_sha256"],
            "974e31567db7f0e717f76944aedb0c027632f9ab2f4f9f4cbf9b66866d114269",
        )
        support = (ROOT / probe.SUPPORT_SOURCE_RELATIVE_PATH).read_bytes()
        self.assertEqual(len(support), probe.SUPPORT_SOURCE_SIZE)
        self.assertEqual(hashlib.sha256(support).hexdigest(), probe.SUPPORT_SOURCE_SHA256)

    def test_package_initializer_and_exact_numpy_runtime_are_bound(self):
        report = _smoke_report()
        self.assertEqual(
            report["sources"]["package_initializer"],
            probe._package_init_identity(),
        )
        expected_runtime = probe._jx_runtime_contract()
        for lane in report["lanes"]:
            self.assertEqual(lane["runtime"], expected_runtime)
        numpy_runtime = expected_runtime["numpy"]
        self.assertEqual(numpy_runtime["version"], probe.EXPECTED_NUMPY_VERSION)
        self.assertEqual(
            numpy_runtime["identity_scope"],
            "PACKAGE_MODULE_MULTIARRAY_UMATH_AND_LINALG_NATIVE_BINARIES",
        )
        for field in (
            "package_module",
            "multiarray_umath_native_binary",
            "linalg_native_binary",
        ):
            self.assertRegex(numpy_runtime[field]["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(numpy_runtime[field]["size_bytes"], 0)


class OptionalExactReboundTests(unittest.TestCase):
    @unittest.skipUnless(
        RUN_REBOUND,
        "set JX_RUN_POST_V5_REBOUND_5_1_1=1 under the retained exact runtime",
    )
    def test_exact_rebound_smoke_ladder_replays_and_refines_on_both_bases(self):
        report = probe.run_precision_probe(probe.PrecisionProfile("smoke"), "required")
        replay = probe.run_precision_probe(probe.PrecisionProfile("smoke"), "required")
        probe.validate_report(report)
        probe.validate_report(replay)
        self.assertEqual(
            report["content_integrity"]["semantic_content_sha256"],
            replay["content_integrity"]["semantic_content_sha256"],
        )
        self.assertIs(report["external_status"]["external_lanes_executed"], True)
        self.assertEqual(report["external_status"]["reason"], "EXACT_REBOUND_5_1_1_EXECUTED")
        self.assertEqual(len(report["lanes"]), 6)
        rebound_lanes = report["lanes"][3:]
        self.assertEqual(
            [lane["lane_id"] for lane in rebound_lanes],
            [
                "rebound_leapfrog_p256",
                "rebound_leapfrog_p512",
                "rebound_leapfrog_p1024",
            ],
        )
        for divisor, lane, replay_lane in zip(
            probe.STEP_DIVISORS,
            rebound_lanes,
            replay["lanes"][3:],
            strict=True,
        ):
            self.assertEqual(lane["role"], "OPTIONAL_EXTERNAL_COMPARATOR")
            self.assertEqual(lane["replay"]["status"], "BITWISE_IDENTICAL")
            self.assertEqual(
                lane["raw_state_evidence"]["sha256"],
                replay_lane["raw_state_evidence"]["sha256"],
            )
            self.assertIsNone(lane["accounting"]["force_evaluations"])
            self.assertEqual(
                lane["accounting"]["effective_settings"],
                probe._expected_rebound_effective_settings(divisor),
            )
            one_step = lane["accounting"]["one_step_dkd_evidence"]
            self.assertEqual(
                one_step["maximum_component_ulp_distance"],
                probe.EXPECTED_REBOUND_ONE_STEP_ULP[divisor],
            )
            self.assertEqual(one_step["maximum_allowed_component_ulp_distance"], 4)
            self.assertEqual(one_step["zero_and_signed_zero_policy"], "EXACT_BINARY64_BITS")
            matrix = probe._decode_matrix_evidence(
                lane["raw_state_evidence"],
                expected_rows=17,
                expected_columns=probe.MATRIX_COLUMNS,
            )
            self.assertTrue(np.all(np.diff(matrix[:, 1]) > 0.0))
        convergence = report["convergence"]["rebound_leapfrog"]
        for basis in ("observed_clock_analytic", "declared_grid_delivery_aware"):
            self.assertEqual(
                convergence[basis]["status"],
                "OBSERVED_WITHIN_EXACT_FIXTURE_REFINEMENT_GATES",
            )

        for mutation in ("effective_setting", "initial_readback"):
            with self.subTest(mutation=mutation):
                candidate = copy.deepcopy(report)
                settings = candidate["lanes"][3]["accounting"]["effective_settings"]
                if mutation == "effective_setting":
                    settings["G"] = math.nextafter(settings["G"], math.inf)
                else:
                    settings["initial_particle_effective_readback"][0]["x"] = -0.25
                with self.assertRaises(probe.PrecisionProbeError):
                    probe.validate_report(_reseal(candidate))


if __name__ == "__main__":
    unittest.main()
