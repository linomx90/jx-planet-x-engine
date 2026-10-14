from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_1kyr_reduced5_h128_source_preflight_v1.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_1kyr_reduced5_h128_source_preflight_v1_test_subject", SOURCE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-preflight module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SourcePreflightContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()
        cls.protocol = cls.m.build_protocol()

    def test_protocol_is_prepare_only_and_every_claim_is_false(self) -> None:
        self.assertEqual(
            self.protocol["status"],
            "PREPARED_SOURCE_ONLY_PREREGISTRATION_NO_EXECUTION",
        )
        authority = self.protocol["authority"]
        execution = self.protocol["execution_specification"]
        self.assertIs(authority["preregistration_required_before_execution"], True)
        self.assertIs(authority["execution_in_this_artifact"], False)
        self.assertEqual(
            authority["explicitly_absent_public_cli"],
            ["execute", "resume", "validate-result"],
        )
        self.assertIs(execution["implemented"], False)
        self.assertIs(execution["authorized"], False)
        claims = self.protocol["claims"]
        self.assertIs(claims["prepare_only_protocol_frozen"], True)
        self.assertTrue(
            all(
                value is False
                for key, value in claims.items()
                if key != "prepare_only_protocol_frozen"
            )
        )

    def test_common_epoch_reduced5_state_is_exact(self) -> None:
        state = self.protocol["initial_state"]
        self.assertEqual(state["epoch_jd_tdb_hex"], (2_461_200.5).hex())
        self.assertEqual(state["epoch_scale"], "TDB")
        self.assertEqual(state["source_ids"], list(self.m.SOURCE_IDS))
        self.assertEqual(state["source_naif_ids"], [10, 5, 6, 7, 8])
        self.assertEqual(
            tuple(
                (
                    state["gm_AU3_per_julian_year2_hex"][index],
                    tuple(state["primary_relative_positions_AU_hex"][index]),
                    tuple(state["primary_relative_velocities_AU_per_julian_year_hex"][index]),
                )
                for index in range(5)
            ),
            self.m.EXPECTED_REDUCED_ROWS,
        )
        collapse = state["collapse"]
        self.assertEqual(collapse["central_members_body_ids"], [10, 1, 2, 3, 4])
        self.assertEqual(collapse["separate_outer_body_ids"], [5, 6, 7, 8])
        self.assertEqual(collapse["omitted_body_ids"], [9])
        self.assertIs(collapse["pluto_folded_into_central_gm"], False)

    def test_reduced_primary_is_exact_positive_zero(self) -> None:
        state = self.protocol["initial_state"]
        zero = "0x0.0p+0"
        self.assertEqual(state["primary_relative_positions_AU_hex"][0], [zero] * 3)
        self.assertEqual(
            state["primary_relative_velocities_AU_per_julian_year_hex"][0],
            [zero] * 3,
        )
        expected_central = math.fsum(
            float.fromhex(value)
            for value in (
                "0x1.3bd0beccac6afp+5",
                "0x1.b7cecb4d5b351p-18",
                "0x1.954f090f44727p-14",
                "0x1.f76df0efd3c95p-14",
                "0x1.ab7a3929b393ap-17",
            )
        )
        self.assertEqual(
            state["gm_AU3_per_julian_year2_hex"][0], expected_central.hex()
        )

    def test_h64_h128_geometry_and_annual_checkpoints_are_frozen(self) -> None:
        lanes = self.protocol["execution_specification"]["jx_lanes"]
        self.assertEqual([lane["steps_per_julian_year"] for lane in lanes], [64, 128])
        self.assertEqual([lane["source_step_count"] for lane in lanes], [64_000, 128_000])
        self.assertEqual([lane["source_node_count"] for lane in lanes], [64_001, 128_001])
        for lane in lanes:
            steps = lane["steps_per_julian_year"]
            schedule = lane["schedule"]
            self.assertEqual(schedule["fixed_step_julian_years_hex"], (1.0 / steps).hex())
            self.assertEqual(
                schedule["checkpoint_source_node_indices"],
                {"first": 0, "last": 1000 * steps, "stride": steps, "count": 1001},
            )
            self.assertIs(schedule["exact_replay_required"], True)

    def test_output_geometry_is_exact_and_bounded(self) -> None:
        outputs = self.protocol["outputs"]
        binary = outputs["binary_files"]
        self.assertEqual(binary["SOURCE_GM.bin"]["bytes"], 40)
        self.assertEqual(binary["JX_H64_SOURCE_NODES.bin"]["bytes"], 15_360_240)
        self.assertEqual(binary["JX_H128_SOURCE_NODES.bin"]["bytes"], 30_720_240)
        self.assertEqual(
            binary["JX_H64_ALL_NODE_INVARIANTS.bin"]["bytes"], 2_048_032
        )
        self.assertEqual(
            binary["JX_H128_ALL_NODE_INVARIANTS.bin"]["bytes"], 4_096_032
        )
        self.assertEqual(binary["IAS15_ANNUAL_SOURCE_STATES.bin"]["bytes"], 240_240)
        self.assertEqual(outputs["fixed_binary_evidence_bytes"], 52_464_824)
        self.assertIs(outputs["create_only_no_overwrite"], True)

    def test_retained_rebound_511_reference_is_independently_pinned(self) -> None:
        reference = self.protocol["execution_specification"]["ias15_reference"]
        self.assertIs(reference["independent_implementation"], True)
        self.assertIs(reference["no_jx_code_in_force_or_step_implementation"], True)
        self.assertEqual(reference["required_version"], "5.1.1")
        self.assertEqual(reference["integrator"], "ias15")
        self.assertEqual(reference["adaptive_mode"], "PRS23")
        self.assertEqual(reference["epsilon_hex"], (1.0e-13).hex())
        self.assertEqual(reference["initial_dt_julian_years_hex"], (1.0 / 32.0).hex())
        self.assertEqual(reference["retained_wheel_sha256"], self.m.REBOUND_WHEEL_SHA256)
        self.assertEqual(reference["wheel_rebound_init_sha256"], self.m.REBOUND_INIT_SHA256)
        self.assertEqual(reference["wheel_librebound_sha256"], self.m.REBOUND_LIBRARY_SHA256)

    def test_full_parent_contract_is_inherited_byte_for_byte_in_memory(self) -> None:
        parent = json.loads(self.m.PARENT_PROTOCOL_PATH.read_text(encoding="utf-8"))
        expected = parent["cpu_gpu_integrator"]["short_horizon_qualification_contract"]
        acceptance = self.protocol["acceptance"]
        self.assertEqual(acceptance["inherited_full_qualification_contract_unmodified"], expected)
        self.assertIs(acceptance["source_only_projection_cannot_satisfy_full_contract"], True)
        self.assertEqual(parent["content_sha256"], self.m.PARENT_PROTOCOL_CONTENT_SHA256)

    def test_source_thresholds_are_exact_fractions_not_relaxed_displays(self) -> None:
        gates = self.protocol["acceptance"]["applicable_source_predicates"]
        expected = {
            "richardson_position_AU_max": (1, 500),
            "richardson_velocity_AU_per_julian_year_max": (1, 50000),
            "h128_vs_ias15_position_AU_max": (1, 500),
            "h128_vs_ias15_velocity_AU_per_julian_year_max": (1, 50000),
            "fine_relative_energy_drift_max": (1, 100_000_000),
            "fine_angular_vector_drift_max": (1, 10_000_000_000),
        }
        for name, (numerator, denominator) in expected.items():
            with self.subTest(name=name):
                self.assertEqual(gates[name]["numerator"], numerator)
                self.assertEqual(gates[name]["denominator"], denominator)
                self.assertEqual(
                    gates[name]["nearest_binary64_hex"],
                    (numerator / denominator).hex(),
                )
        self.assertEqual(gates["richardson_divisor"], 3)
        self.assertEqual(gates["energy_contraction_floor_hex"], "0x1.0000000000000p-46")

    def test_tracer_only_gates_remain_frozen_but_cannot_be_claimed(self) -> None:
        omitted = self.protocol["acceptance"]["tracer_only_predicates_not_evaluated"]
        self.assertEqual(omitted["richardson_normalized_semimajor_axis_max"], "3/8000")
        self.assertEqual(omitted["h128_vs_ias15_normalized_semimajor_axis_max"], "3/4000")
        self.assertEqual(
            omitted["h128_vs_ias15_scattering_diagnostic_disagreement_max"],
            "3/4000",
        )
        self.assertEqual(
            omitted["reason"], "NO_TRACER_STATE_EXISTS_IN_THIS_SOURCE_ONLY_PREFLIGHT"
        )
        self.assertIs(omitted["omission_counts_as_pass"], False)

    def test_all_node_invariant_semantics_are_explicit(self) -> None:
        invariants = self.protocol["acceptance"]["invariants"]
        self.assertEqual(
            invariants["jx_sampling"], "EVERY_FIXED_NODE_INCLUDING_BOTH_ENDPOINTS"
        )
        self.assertEqual(
            invariants["energy_contraction"],
            "FINE_LE_MAX_COARSE_DIV_3_2_POW_MINUS_46",
        )
        self.assertEqual(invariants["fine_energy_max"], "1/100000000")
        self.assertEqual(invariants["fine_angular_max"], "1/10000000000")
        self.assertEqual(invariants["zero_or_nonfinite_denominator"], "FAIL")

    def test_exact_scalar_vector_and_energy_contraction_helpers(self) -> None:
        fine = float.fromhex("0x1.0000000000000p-30")
        self.assertTrue(self.m.scalar_contraction_pass(3.0 * fine, fine, 1.0))
        self.assertFalse(self.m.scalar_contraction_pass(2.0 * fine, fine, 1.0))
        self.assertTrue(
            self.m.scalar_contraction_pass(0.0, 64.0 * math.ulp(1.0), 1.0)
        )
        fine_component = float.fromhex("0x1.0000000000000p-20")
        self.assertTrue(
            self.m.vector_contraction_pass(
                (3.0 * fine_component, 0.0, 0.0),
                (fine_component, 0.0, 0.0),
                (1.0, 2.0, 4.0),
            )
        )
        self.assertFalse(
            self.m.vector_contraction_pass(
                (2.0 * fine_component, 0.0, 0.0),
                (fine_component, 0.0, 0.0),
                (1.0, 2.0, 4.0),
            )
        )
        coarse_boundary = float.fromhex("0x1.ac2a2dd310afbp-19")
        fine_boundary = float.fromhex("0x1.1d7173e20b1fdp-20")
        self.assertGreaterEqual(
            coarse_boundary * coarse_boundary,
            9.0 * (fine_boundary * fine_boundary),
        )
        self.assertFalse(
            self.m.vector_contraction_pass(
                (coarse_boundary, 0.0, 0.0),
                (fine_boundary, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        )
        floor = float.fromhex("0x1.0000000000000p-46")
        self.assertTrue(self.m.energy_contraction_pass(0.0, floor))
        self.assertFalse(self.m.energy_contraction_pass(0.0, 2.0 * floor))
        exact_energy_fine = float.fromhex("0x1.0000000000000p-40")
        self.assertTrue(
            self.m.energy_contraction_pass(
                3.0 * exact_energy_fine, exact_energy_fine
            )
        )

    def test_contraction_helpers_fail_closed_on_bad_types_and_nonfinite_values(self) -> None:
        for bad in (True, -1.0, math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(self.m.SourcePreflightError):
                self.m.scalar_contraction_pass(bad, 0.0, 1.0)
        with self.assertRaises(self.m.SourcePreflightError):
            self.m.vector_contraction_pass(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 2.0),
            )
        with self.assertRaises(self.m.SourcePreflightError):
            self.m.vector_contraction_pass(
                (0.0, math.nan, 0.0),
                (0.0, 0.0, 0.0),
                (1.0, 2.0, 4.0),
            )
        with self.assertRaises(self.m.SourcePreflightError):
            self.m.energy_contraction_pass(0.0, math.nan)

    def test_protocol_roundtrip_and_tamper_rejection(self) -> None:
        payload = self.m._canonical_json(self.protocol)
        decoded = self.m._strict_json(payload)
        self.assertEqual(decoded, self.protocol)
        self.assertEqual(
            self.m.validate_protocol(decoded), self.protocol["content_sha256"]
        )
        tampered = json.loads(payload)
        tampered["execution_specification"]["jx_lanes"][1]["steps_per_julian_year"] = 64
        with self.assertRaises(self.m.SourcePreflightError):
            self.m.validate_protocol(tampered)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        with self.assertRaises(self.m.SourcePreflightError):
            self.m._strict_json(b'{"a":1,"a":2}\n')
        for token in (b"NaN", b"Infinity", b"-Infinity"):
            with self.subTest(token=token), self.assertRaises(self.m.SourcePreflightError):
                self.m._strict_json(b'{"a":' + token + b"}\n")

    def test_prepare_and_validate_create_only_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            value = self.m.prepare(root)
            self.assertEqual([item.name for item in root.iterdir()], ["PROTOCOL.json"])
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(
                stat.S_IMODE((root / "PROTOCOL.json").stat().st_mode), 0o600
            )
            self.assertEqual(self.m.validate_prepared(root), value)
            self.assertEqual(
                (root / "PROTOCOL.json").read_bytes(), self.m._canonical_json(value)
            )
            with self.assertRaises(self.m.SourcePreflightError):
                self.m.prepare(root)
            os.chmod(root, 0o755)
            with self.assertRaises(self.m.SourcePreflightError):
                self.m.validate_prepared(root)

    def test_prepare_cleans_partial_write_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            original_write = os.write
            write_calls = 0

            def fail_after_partial_write(descriptor, payload):
                nonlocal write_calls
                write_calls += 1
                if write_calls == 1:
                    original_write(descriptor, payload[:1])
                    raise OSError("injected write failure")
                return original_write(descriptor, payload)

            with mock.patch.object(
                self.m.os, "write", side_effect=fail_after_partial_write
            ):
                with self.assertRaises(self.m.SourcePreflightError):
                    self.m.prepare(root)
            self.assertFalse(root.exists())
            value = self.m.prepare(root)
            self.assertEqual(self.m.validate_prepared(root), value)

    def test_prepare_publishes_only_after_durable_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"

            def stop_before_publication(directory_fd, source, destination):
                self.assertEqual(source, ".PROTOCOL.json.partial")
                self.assertEqual(destination, "PROTOCOL.json")
                self.assertTrue((root / source).is_file())
                self.assertFalse((root / destination).exists())
                raise self.m.SourcePreflightError("injected publication stop")

            with mock.patch.object(
                self.m, "_rename_noreplace", side_effect=stop_before_publication
            ):
                with self.assertRaises(self.m.SourcePreflightError):
                    self.m.prepare(root)
            self.assertFalse(root.exists())

    def test_prepare_source_imports_no_numpy_rebound_or_jx_runtime(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        self.assertTrue({"numpy", "rebound", "jxplanetx"}.isdisjoint(imported_roots))

    def test_cli_exposes_only_prepare_and_validate_prepared(self) -> None:
        parser = self.m._parser()
        subparser_actions = [
            action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
        ]
        self.assertEqual(len(subparser_actions), 1)
        self.assertEqual(
            set(subparser_actions[0].choices), {"prepare", "validate-prepared"}
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["execute"])

    def test_every_static_input_binding_is_verified(self) -> None:
        observed = {
            item["relative_path"]: (item["byte_length"], item["sha256"])
            for item in self.protocol["input_bindings"]
        }
        self.assertEqual(len(observed), len(self.m.INPUT_BINDINGS))
        for path, length, digest, _role in self.m.INPUT_BINDINGS:
            relative = path.relative_to(ROOT).as_posix()
            self.assertEqual(observed[relative], (length, digest))


if __name__ == "__main__":
    unittest.main()
