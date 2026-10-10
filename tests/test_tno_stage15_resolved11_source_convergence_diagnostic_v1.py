from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import struct
import tempfile
import types
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "benchmarks"
    / "jx_tno_stage15_resolved11_source_convergence_diagnostic_v1.py"
)


def load_source() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    module = types.ModuleType("_jx_stage15_source_convergence_diagnostic_test")
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    code = compile(payload, str(SOURCE), "exec", optimize=__import__("sys").flags.optimize)
    exec(code, module.__dict__)
    return module


M = load_source()


def translated(value: float, years: int = 3) -> np.ndarray:
    result = np.zeros((years, 2, M.MASSIVE_COUNT, 3), dtype=np.float64)
    result[:, 0, :, 0] = value
    return result


def invariant(energy: Fraction = Fraction(1, 10**12), angular: float = 1e-12) -> dict:
    return {
        "sample_count": 2,
        "maximum_relative_energy_drift": M._fraction_observation(energy),
        "maximum_relative_angular_momentum_drift_hex": angular.hex(),
        "maximum_absolute_linear_momentum_change_hex": 0.0.hex(),
        "maximum_barycenter_position_change_AU_hex": 0.0.hex(),
        "maximum_barycenter_velocity_change_AU_per_julian_year_hex": 0.0.hex(),
        "sampling": "EVERY_RETAINED_FIXED_SOURCE_NODE",
    }


class SourceConvergenceDiagnosticTests(unittest.TestCase):
    def test_protocol_is_source_only_and_binds_v8_git_and_failure(self) -> None:
        protocol = M.build_protocol()
        self.assertEqual(protocol["schema"], M.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["content_sha256"], M._domain_sha(M.PROTOCOL_DOMAIN, protocol))
        self.assertEqual(
            protocol["predecessor_v8"]["git_custody"],
            {
                "branch": "research/tno-stage15-resolved11-encounter-v8-preregistered-20261010",
                "preregistration_commit": "0fe340feb3fa8b666264e0128ed084697cfb642f",
                "annotated_tag": "tno-stage15-resolved11-encounter-v8-preregistered-20261010",
                "tag_object": "ba548eda088fbb171a61e7efa2906e2bb36ca81d",
                "tag_peeled_commit": "0fe340feb3fa8b666264e0128ed084697cfb642f",
                "external_git_custody_not_reconstructed_by_this_program": True,
            },
        )
        failed = protocol["predecessor_v8"]["failed_result"]
        self.assertEqual((failed["gate_count"], failed["passed_gate_count"], failed["failed_gate_count"]), (20, 12, 8))
        self.assertFalse(failed["qualification_established"])
        self.assertFalse(protocol["claims"]["public_tracer_engine_called"])
        self.assertFalse(protocol["claims"]["adaptive_rerun_authorized"])
        self.assertFalse(protocol["claims"]["science_result"])

    def test_live_v8_source_test_protocol_and_failed_result_bindings(self) -> None:
        for path, size, digest in (
            (M.V8_SOURCE_PATH, M.V8_SOURCE_BYTES, M.V8_SOURCE_SHA256),
            (M.V8_TEST_PATH, M.V8_TEST_BYTES, M.V8_TEST_SHA256),
            (M.V8_PROTOCOL_PATH, M.V8_PROTOCOL_BYTES, M.V8_PROTOCOL_SHA256),
            (M.V4_SOURCE_PATH, M.V4_SOURCE_BYTES, M.V4_SOURCE_SHA256),
            (M.V4_PROTOCOL_PATH, M.V4_PROTOCOL_BYTES, M.V4_PROTOCOL_SHA256),
            (M.V4_RESULT_PATH, M.V4_RESULT_BYTES, M.V4_RESULT_SHA256),
            (M.BRIDGE_PATH, M.BRIDGE_BYTES, M.BRIDGE_SHA256),
            (
                M.DE441_CUSTODY_PATH, M.DE441_CUSTODY_BYTES,
                M.DE441_CUSTODY_SHA256,
            ),
            (
                M.STAGE15_V1_PROTOCOL_PATH, M.STAGE15_V1_PROTOCOL_BYTES,
                M.STAGE15_V1_PROTOCOL_SHA256,
            ),
        ):
            payload = path.read_bytes()
            self.assertEqual(len(payload), size)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
        failed = M._validate_v8_failure_result()
        self.assertEqual(failed["status"], "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS")
        self.assertFalse(failed["claims"]["qualification_established"])
        v8, validated = M._validate_small_prerequisites()
        _v4, _bridge, _custody, batches, _oracle = v8._load_v4_context()
        locators = M._v8_source_locator_map(validated)
        fd = os.open(M.V8_SOURCE_NODES_PATH, os.O_RDONLY)
        try:
            for batch_order, batch in enumerate(batches):
                expected = M._batch_initial_source_payload(np, batch)
                for steps in M.RETAINED_STEPS:
                    M._require_initial_source_payload(
                        fd, locators[(batch_order, steps)], expected,
                        f"batch {batch_order} h{steps}",
                    )
        finally:
            os.close(fd)

    def test_external_v8_json_newline_policy_is_exact(self) -> None:
        payload = M._canonical_json({"value": 1}) + b"\n"
        self.assertEqual(
            M._load_canonical_json(
                payload, "external", trailing_newline=True,
            ),
            {"value": 1},
        )
        with self.assertRaises(M.SourceConvergenceDiagnosticError):
            M._load_canonical_json(payload, "local")
        with self.assertRaises(M.SourceConvergenceDiagnosticError):
            M._load_canonical_json(payload + b"\n", "external", trailing_newline=True)

    def test_exact_compute_and_storage_projection(self) -> None:
        self.assertEqual(M.TOTAL_BASE_FIXED_STEPS, 9_216_000)
        self.assertEqual(M.TOTAL_H1024_MAXIMUM_STEPS, 12_288_000)
        self.assertEqual(M.TOTAL_MAXIMUM_FIXED_STEPS, 21_504_000)
        self.assertEqual(M.BATCH_COUNT * M.HORIZON_YEARS, 12_000)
        self.assertEqual(M.SOURCE_BASE_EXACT_BYTES, 4_866_060_672)
        self.assertEqual(M.SOURCE_H1024_MAXIMUM_BYTES, 6_488_070_336)
        self.assertEqual(M.SOURCE_MAXIMUM_BYTES, 11_354_131_008)
        self.assertEqual(M.ANNUAL_BASE_EXACT_BYTES, 19_027_008)
        self.assertEqual(M.ANNUAL_H1024_MAXIMUM_BYTES, 6_342_336)
        self.assertEqual(M.ANNUAL_MAXIMUM_BYTES, 25_369_344)

    def test_prepare_and_validate_prepared_exact_one_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "prepared"
            prepared = M.prepare(root)
            self.assertEqual(sorted(item.name for item in root.iterdir()), ["PROTOCOL.json"])
            self.assertEqual(prepared, M.validate_prepared(root))
            with self.assertRaises(M.SourceConvergenceDiagnosticError):
                M.prepare(root)
            (root / "EXTRA").write_bytes(b"x")
            with self.assertRaises(M.SourceConvergenceDiagnosticError):
                M.validate_prepared(root)

    def test_no_public_tracer_call_or_tracer_state_construction(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        calls = []
        attributes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)
                elif isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
            if isinstance(node, ast.Attribute):
                attributes.append(node.attr)
        self.assertNotIn("advance_cuda_massless_tracers", calls)
        self.assertNotIn("CUDAMasslessTracerState", calls)
        self.assertNotIn("tracer_positions", attributes)
        self.assertNotIn("tracer_velocities", attributes)
        self.assertIn("_build_source_path", calls)

    def test_representation_roster_and_exact_metric(self) -> None:
        left = translated(1.0)
        right = translated(0.0)
        rows = M._series_comparison(left, right)
        self.assertEqual(len(rows), 76)
        index = M._comparison_index(rows, "test")
        sun = index[("BARYCENTRIC", "SUN")]
        self.assertEqual(
            M._fraction_from_record(sun["position_squared"]["exact"], "position"),
            Fraction(1, 1),
        )
        self.assertEqual(
            M._fraction_from_record(
                index[("SUN_RELATIVE", "MERCURY")]["position_squared"]["exact"],
                "sun relative",
            ),
            Fraction(0, 1),
        )
        self.assertEqual(
            M._fraction_from_record(
                index[("PAIRWISE", "SUN->MERCURY")]["position_squared"]["exact"],
                "pairwise",
            ),
            Fraction(0, 1),
        )

    def test_second_order_window_exact_lower_and_upper_boundaries(self) -> None:
        fine = M._series_comparison(translated(1.0), translated(0.0))
        lower = M._series_comparison(translated(4.0), translated(1.0))
        upper = M._series_comparison(translated(6.0), translated(1.0))
        self.assertTrue(M._order_window(lower, fine, "lower")["all_passed"])
        self.assertTrue(M._order_window(upper, fine, "upper")["all_passed"])
        below = M._series_comparison(translated(math.nextafter(4.0, -math.inf)), translated(1.0))
        self.assertFalse(M._order_window(below, fine, "below")["all_passed"])
        above = M._series_comparison(translated(math.nextafter(6.0, math.inf)), translated(1.0))
        self.assertFalse(M._order_window(above, fine, "above")["all_passed"])

    def test_pre_adjudication_trigger_is_mechanical(self) -> None:
        levels = {
            64: translated(21.0), 128: translated(5.0),
            256: translated(1.0), 512: translated(0.0),
        }
        reference = translated(0.0)
        pre = M._pre_adjudication_diagnostic(levels, reference, reference)
        self.assertFalse(pre["h1024_triggered"])
        self.assertTrue(pre["reference_qualified"])
        bad_reference = translated(1.0)
        triggered = M._pre_adjudication_diagnostic(levels, bad_reference, reference)
        self.assertTrue(triggered["h1024_triggered"])
        self.assertFalse(triggered["reference_qualified"])

    def test_timestep_limited_and_non_timestep_dispositions(self) -> None:
        levels = {
            64: translated(21.0), 128: translated(5.0),
            256: translated(1.0), 512: translated(0.0),
            1024: translated(-0.25),
        }
        tight = translated(-1.0 / 3.0)
        retained = tight.copy()
        pre = M._pre_adjudication_diagnostic(
            {key: levels[key] for key in (64, 128, 256, 512)}, retained, tight,
        )
        self.assertTrue(pre["h1024_triggered"])
        summaries = {
            64: invariant(Fraction(81, 10**12)),
            128: invariant(Fraction(27, 10**12)),
            256: invariant(Fraction(9, 10**12)),
            512: invariant(Fraction(3, 10**12)),
            1024: invariant(Fraction(1, 10**12)),
        }
        result = M._finalize_epoch_diagnostic(pre, levels, tight, summaries)
        self.assertEqual(result["disposition"], "TIMESTEP_LIMITED")
        self.assertTrue(result["step_estimate"]["authorized"])
        invariant_failed = dict(summaries)
        invariant_failed[1024] = invariant(
            M.ENERGY_BUDGET + Fraction(1, 10**20),
        )
        h1024_inconclusive = M._finalize_epoch_diagnostic(
            pre, levels, tight, invariant_failed,
        )
        self.assertEqual(
            h1024_inconclusive["disposition"],
            "INCONCLUSIVE_AT_MAX_RESOLUTION",
        )
        self.assertFalse(h1024_inconclusive["step_estimate"]["authorized"])
        self.assertIsNone(
            h1024_inconclusive["step_estimate"]["estimated_required_steps_per_year"],
        )
        self.assertTrue(
            h1024_inconclusive["step_estimate"][
                "coordinate_only_estimate_suppressed_due_to_finest_invariant_failure"
            ],
        )
        self.assertEqual(
            h1024_inconclusive["step_estimate"]["suppression_reason"],
            "DECISIVE_FINEST_ABSOLUTE_INVARIANT_GATE_FAILED",
        )
        far_reference = translated(1.0)
        pre_far = M._pre_adjudication_diagnostic(
            {key: levels[key] for key in (64, 128, 256, 512)},
            far_reference, far_reference,
        )
        non_timestep = M._finalize_epoch_diagnostic(
            pre_far, levels, far_reference, summaries,
        )
        self.assertEqual(
            non_timestep["disposition"],
            "NON_TIMESTEP_REFERENCE_DISCREPANCY",
        )
        self.assertFalse(non_timestep["step_estimate"]["authorized"])
        self.assertIsNone(
            non_timestep["step_estimate"]["estimated_required_steps_per_year"],
        )
        self.assertEqual(
            non_timestep["step_estimate"]["suppression_reason"],
            "RELEVANT_FINAL_RICHARDSON_OR_EXTRAPOLATE_GATE_FAILED",
        )

    def test_inconclusive_when_reference_is_not_qualified(self) -> None:
        levels = {
            64: translated(21.0), 128: translated(5.0),
            256: translated(1.0), 512: translated(0.0),
            1024: translated(-0.25),
        }
        tight = translated(-1.0 / 3.0)
        retained = translated(10.0)
        pre = M._pre_adjudication_diagnostic(
            {key: levels[key] for key in (64, 128, 256, 512)}, retained, tight,
        )
        summaries = {
            64: invariant(Fraction(81, 10**12)),
            128: invariant(Fraction(27, 10**12)),
            256: invariant(Fraction(9, 10**12)),
            512: invariant(Fraction(3, 10**12)),
            1024: invariant(Fraction(1, 10**12)),
        }
        final = M._finalize_epoch_diagnostic(pre, levels, tight, summaries)
        self.assertEqual(final["disposition"], "INCONCLUSIVE_AT_MAX_RESOLUTION")
        self.assertFalse(final["step_estimate"]["authorized"])
        self.assertIsNone(final["step_estimate"]["estimated_required_steps_per_year"])
        self.assertEqual(
            final["step_estimate"]["suppression_reason"],
            "IAS15_REFERENCE_NOT_QUALIFIED",
        )
        self.assertFalse(final["stage15_qualification_established"])
        self.assertFalse(final["adaptive_rerun_authorized"])

    def test_step_estimate_requires_two_windows(self) -> None:
        value = M._estimated_power_of_two_steps(
            current_steps=512, error_squared=Fraction(1, 100),
            budget=Fraction(1, 500), observed_order=2.0,
            two_consecutive_windows_passed=False,
        )
        self.assertIsNone(value)
        allowed = M._estimated_power_of_two_steps(
            current_steps=512, error_squared=Fraction(1, 100),
            budget=Fraction(1, 500), observed_order=2.0,
            two_consecutive_windows_passed=True,
        )
        self.assertEqual(allowed, 4096)

    def test_invariant_gates_cover_every_retained_and_new_level(self) -> None:
        summaries = {
            64: invariant(Fraction(81, 10**12)),
            128: invariant(Fraction(27, 10**12)),
            256: invariant(Fraction(9, 10**12)),
            512: invariant(Fraction(3, 10**12)),
            1024: invariant(Fraction(1, 10**12)),
        }
        base = M._gate_invariants(summaries, False)
        self.assertTrue(base["all_passed"])
        self.assertTrue(base["all_absolute_levels_passed"])
        self.assertTrue(base["all_adjacent_energy_contractions_passed"])
        self.assertEqual(base["finest_steps_per_year"], 512)
        self.assertTrue(base["finest_absolute_gates_passed"])
        self.assertEqual(
            {row["gate"] for row in base["gates"]},
            {
                *(f"h{step}_all_node_energy" for step in (64, 128, 256, 512)),
                *(
                    f"h{step}_all_node_angular_momentum"
                    for step in (64, 128, 256, 512)
                ),
                "h64_h128_energy_contraction",
                "h128_h256_energy_contraction",
                "h256_h512_energy_contraction",
            },
        )
        adjudicated = M._gate_invariants(summaries, True)
        self.assertTrue(adjudicated["all_passed"])
        self.assertEqual(adjudicated["finest_steps_per_year"], 1024)
        self.assertTrue(adjudicated["finest_absolute_gates_passed"])
        self.assertIn(
            "h512_h1024_energy_contraction",
            {row["gate"] for row in adjudicated["gates"]},
        )

    def test_pinned_v8_coarse_invariants_are_diagnostic_and_finest_controls_disposition(self) -> None:
        levels = {
            64: translated(21.0), 128: translated(5.0),
            256: translated(1.0), 512: translated(0.0),
        }
        reference = translated(0.0)
        pre = M._pre_adjudication_diagnostic(levels, reference, reference)
        self.assertFalse(pre["h1024_triggered"])

        # Exact batch-0 h64/h128 magnitudes pinned in the failed V8 RESULT.
        h64_energy = Fraction(125_908_110_231, 2_101_499_239_164_196)
        h128_energy = Fraction(129_015_457_439, 6_304_497_717_492_588)
        summaries = {
            64: invariant(h64_energy, float.fromhex("0x1.6662ec0f70aecp-47")),
            128: invariant(h128_energy, float.fromhex("0x1.71e4ecb2d386ap-45")),
            256: invariant(Fraction(3, 10**12)),
            512: invariant(Fraction(1, 10**12)),
        }
        passing = M._finalize_epoch_diagnostic(pre, levels, reference, summaries)
        self.assertEqual(passing["disposition"], "PASS")
        self.assertFalse(passing["invariant_gates"]["all_absolute_levels_passed"])
        self.assertFalse(
            passing["invariant_gates"]["all_adjacent_energy_contractions_passed"],
        )
        self.assertTrue(passing["invariant_gates"]["finest_absolute_gates_passed"])

        finest_failed = dict(summaries)
        finest_failed[512] = invariant(M.ENERGY_BUDGET + Fraction(1, 10**20))
        h512_inconclusive = M._finalize_epoch_diagnostic(
            pre, levels, reference, finest_failed,
        )
        self.assertFalse(
            h512_inconclusive["invariant_gates"]["finest_absolute_energy_passed"],
        )
        self.assertEqual(
            h512_inconclusive["disposition"], "INCONCLUSIVE_AT_MAX_RESOLUTION",
        )
        self.assertFalse(h512_inconclusive["step_estimate"]["authorized"])
        self.assertIsNone(
            h512_inconclusive["step_estimate"]["estimated_required_steps_per_year"],
        )
        self.assertTrue(
            h512_inconclusive["step_estimate"][
                "coordinate_only_estimate_suppressed_due_to_finest_invariant_failure"
            ],
        )
        self.assertEqual(
            h512_inconclusive["step_estimate"]["suppression_reason"],
            "DECISIVE_FINEST_ABSOLUTE_INVARIANT_GATE_FAILED",
        )

    def test_outward_sqrt_handles_underflowed_square(self) -> None:
        value = Fraction(1, 1 << 1200)
        observed = M._outward_sqrt(value)
        self.assertEqual(observed, math.ldexp(1.0, -600))
        self.assertGreaterEqual(Fraction.from_float(observed) ** 2, value)
        self.assertLess(Fraction.from_float(math.nextafter(observed, -math.inf)) ** 2, value)

    def test_source_annual_crossbinding_and_tamper(self) -> None:
        gm = tuple(float(index + 1) for index in range(M.MASSIVE_COUNT))
        raw = np.zeros((3, 2, M.MASSIVE_COUNT, 3), dtype=np.float64)
        for node in range(3):
            for source in range(M.MASSIVE_COUNT):
                raw[node, 0, source] = (source + node, source * 0.1, node * 0.2)
                raw[node, 1, source] = (0.0, source * 0.01, node * 0.001)
        centered = M._center_annual(np, raw, gm)
        source_payload = raw.astype("<f8").tobytes(order="C")
        annual_payload = centered.astype("<f8").tobytes(order="C")
        with tempfile.TemporaryDirectory() as temporary:
            source_path = Path(temporary) / "SOURCE.bin"
            annual_path = Path(temporary) / "ANNUAL.bin"
            source_path.write_bytes(source_payload)
            annual_path.write_bytes(annual_payload)
            source_locator = {"byte_offset": 0, "node_count": 3}
            annual_locator = {"byte_offset": 0, "steps_per_julian_year": 1}
            source_fd = os.open(source_path, os.O_RDONLY)
            annual_fd = os.open(annual_path, os.O_RDONLY)
            old_horizon = M.HORIZON_YEARS
            old_count = M.CHECKPOINT_COUNT
            try:
                M.HORIZON_YEARS = 2
                M.CHECKPOINT_COUNT = 3
                M._crossbind_source_annual(
                    source_fd=source_fd, source_locator=source_locator,
                    annual_fd=annual_fd, annual_locator=annual_locator,
                    gm_values=gm, np=np,
                )
                os.close(annual_fd)
                tampered = bytearray(annual_payload)
                tampered[-1] ^= 1
                annual_path.write_bytes(tampered)
                annual_fd = os.open(annual_path, os.O_RDONLY)
                with self.assertRaises(M.SourceConvergenceDiagnosticError):
                    M._crossbind_source_annual(
                        source_fd=source_fd, source_locator=source_locator,
                        annual_fd=annual_fd, annual_locator=annual_locator,
                        gm_values=gm, np=np,
                    )
            finally:
                M.HORIZON_YEARS = old_horizon
                M.CHECKPOINT_COUNT = old_count
                os.close(source_fd)
                os.close(annual_fd)

    def test_initial_source_payload_crossbinding_and_tamper(self) -> None:
        massive = tuple(
            types.SimpleNamespace(
                position_AU=(float(index), 0.25 * index, -0.5 * index),
                velocity_AU_per_julian_year=(0.01 * index, -0.02 * index, 0.03),
            )
            for index in range(M.MASSIVE_COUNT)
        )
        batch = types.SimpleNamespace(massive=massive)
        expected = M._batch_initial_source_payload(np, batch)
        self.assertEqual(len(expected), M.SOURCE_NODE_RECORD_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.bin"
            path.write_bytes(expected + bytes(M.SOURCE_NODE_RECORD_BYTES))
            fd = os.open(path, os.O_RDONLY)
            try:
                M._require_initial_source_payload(
                    fd, {"byte_offset": 0}, expected, "initial",
                )
            finally:
                os.close(fd)
            tampered = bytearray(path.read_bytes())
            tampered[0] ^= 1
            path.write_bytes(tampered)
            fd = os.open(path, os.O_RDONLY)
            try:
                with self.assertRaises(M.SourceConvergenceDiagnosticError):
                    M._require_initial_source_payload(
                        fd, {"byte_offset": 0}, expected, "initial",
                    )
            finally:
                os.close(fd)

    def test_streamed_invariants_rederive_and_detect_nonfinite(self) -> None:
        nodes = 4
        raw = np.zeros((nodes, 2, M.MASSIVE_COUNT, 3), dtype=np.float64)
        for node in range(nodes):
            for source in range(M.MASSIVE_COUNT):
                angle = 0.1 * source + 1e-5 * node
                radius = 1.0 + source
                raw[node, 0, source] = (radius * math.cos(angle), radius * math.sin(angle), source * 0.01)
                raw[node, 1, source] = (-math.sin(angle), math.cos(angle), 0.001 * source)
        payload = raw.astype("<f8").tobytes(order="C")
        locator = {"byte_offset": 0, "byte_length": len(payload), "node_count": nodes}
        gm = tuple(1e-6 * (index + 1) for index in range(M.MASSIVE_COUNT))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nodes.bin"
            path.write_bytes(payload)
            fd = os.open(path, os.O_RDONLY)
            try:
                summary = M._source_invariant_summary(fd, locator, gm, np)
            finally:
                os.close(fd)
            self.assertEqual(summary["sample_count"], nodes)
            self.assertEqual(summary["sampling"], "EVERY_RETAINED_FIXED_SOURCE_NODE")
            bad = bytearray(payload)
            bad[0:8] = struct.pack("<d", math.nan)
            path.write_bytes(bad)
            fd = os.open(path, os.O_RDONLY)
            try:
                with self.assertRaises(M.SourceConvergenceDiagnosticError):
                    M._source_invariant_summary(fd, locator, gm, np)
            finally:
                os.close(fd)

    def test_stable_fd_rejects_hardlink_added_during_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.bin"
            other = Path(temporary) / "hardlink.bin"
            path.write_bytes(b"evidence")
            digest = hashlib.sha256(b"evidence").hexdigest()
            with self.assertRaises(M.SourceConvergenceDiagnosticError):
                with M._stable_fd(path, "evidence", exact_bytes=8, expected_sha256=digest):
                    os.link(path, other)

    def test_stable_fd_rejects_path_replacement_during_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "evidence.bin"
            moved = Path(temporary) / "original.bin"
            path.write_bytes(b"evidence")
            digest = hashlib.sha256(b"evidence").hexdigest()
            with self.assertRaises(M.SourceConvergenceDiagnosticError):
                with M._stable_fd(
                    path, "evidence", exact_bytes=8,
                    expected_sha256=digest,
                ):
                    path.rename(moved)
                    path.write_bytes(b"evidence")

    def test_source_and_annual_locator_bool_identity_rejected(self) -> None:
        source = {
            "schema": M.SOURCE_SCHEMA, "kind": "JX_SOURCE_ONLY_FIXED_STEP",
            "lane_id": "jx_resolved11_source_h256", "steps_per_julian_year": 256,
            "step_julian_years_hex": (1.0 / 256).hex(), "byte_offset": False,
            "byte_length": M.H256_NODE_COUNT * M.SOURCE_NODE_RECORD_BYTES,
            "record_bytes": M.SOURCE_NODE_RECORD_BYTES, "node_count": M.H256_NODE_COUNT,
            "lane_payload_sha256": "0" * 64,
        }
        with self.assertRaises(M.SourceConvergenceDiagnosticError):
            M._validate_source_locator(source, expected_offset=0, batch_order=0, steps=256)
        annual = {
            "schema": M.ANNUAL_SCHEMA, "kind": "JX_SOURCE_ONLY_FIXED_STEP",
            "lane_id": "jx_resolved11_source_h256", "steps_per_julian_year": 256,
            "byte_offset": False,
            "byte_length": M.CHECKPOINT_COUNT * M.ANNUAL_RECORD_BYTES,
            "record_bytes": M.ANNUAL_RECORD_BYTES, "checkpoint_count": M.CHECKPOINT_COUNT,
            "lane_payload_sha256": "0" * 64,
        }
        with self.assertRaises(M.SourceConvergenceDiagnosticError):
            M._validate_annual_locator(
                annual, expected_offset=0, batch_order=0,
                kind="JX_SOURCE_ONLY_FIXED_STEP", steps=256,
            )
        annual["byte_offset"] = 0
        annual["unexpected_semantic_label"] = "SCATTERING"
        with self.assertRaises(M.SourceConvergenceDiagnosticError):
            M._validate_annual_locator(
                annual, expected_offset=0, batch_order=0,
                kind="JX_SOURCE_ONLY_FIXED_STEP", steps=256,
            )

    def test_ias_receipt_bool_identity_rejected(self) -> None:
        receipt = {
            "epsilon_hex": M.IAS15_TIGHT_EPSILON.hex(),
            "initial_dt_hex": M.IAS15_INITIAL_DT.hex(),
            "min_dt_hex": M.IAS15_MIN_DT.hex(), "adaptive_mode": M.IAS15_ADAPTIVE_MODE,
            "annual_integrate_call_count": M.HORIZON_YEARS,
            "initial_steps_done": False, "final_steps_done": 10,
            "annual_steps_done": list(range(M.CHECKPOINT_COUNT)),
            "annual_steps_done_sha256": "0" * 64,
            "final_time_hex": float(M.HORIZON_YEARS).hex(),
            "gravitational_constant_hex": 1.0.hex(),
            "gravity": "basic", "collision": "none", "boundary": "none",
            "softening_hex": 0.0.hex(), "testparticle_type": 0,
            "active_source_count": M.MASSIVE_COUNT,
            "source_count": M.MASSIVE_COUNT, "tracer_count": False,
        }
        with self.assertRaises(M.SourceConvergenceDiagnosticError):
            M._validate_ias_receipt(receipt)

    def test_result_last_publication_and_missing_result_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            staging = base / "staging"
            root.mkdir()
            staging.mkdir()
            (root / "PROTOCOL.json").write_bytes(b"protocol")
            for name in ("SOURCE_NODES.bin", "ANNUAL_CHECKPOINTS.bin", "RUNTIME.json"):
                (staging / name).write_bytes(name.encode("ascii"))
            with self.assertRaises(M.SourceConvergenceDiagnosticError):
                M._publish_result_last(staging, root)
            self.assertFalse((root / "RESULT.json").exists())
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            staging = base / "staging"
            root.mkdir()
            staging.mkdir()
            (root / "PROTOCOL.json").write_bytes(b"protocol")
            for name in ("SOURCE_NODES.bin", "ANNUAL_CHECKPOINTS.bin", "RUNTIME.json", "RESULT.json"):
                (staging / name).write_bytes(name.encode("ascii"))
            M._publish_result_last(staging, root)
            self.assertEqual(sorted(item.name for item in root.iterdir()), sorted(M.EXACT_ROSTER))
            self.assertEqual((root / "RESULT.json").read_bytes(), b"RESULT.json")


if __name__ == "__main__":
    unittest.main()
