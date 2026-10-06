from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from benchmarks import jx_long_horizon_m2_source_preflight as qualification
from jxplanetx.long_horizon.physical_models import (
    KNOWN_SOLAR_SYSTEM_V2,
    KNOWN_SYSTEM_BODY_IDS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _zero_residual() -> dict[str, float]:
    return {
        "maximum_position_AU": 0.0,
        "maximum_velocity_AU_per_year": 0.0,
    }


def _passing_metrics() -> dict[str, object]:
    invariants = {
        name: {
            "maximum_relative_energy_drift": 0.0,
            "maximum_relative_angular_momentum_drift": 0.0,
        }
        for name in ("jx_coarse", "jx_fine", "dop853", "ias15")
    }
    return {
        "initial_reference": _zero_residual(),
        "initial_reference_bitwise_exact": True,
        "jx_coarse_vs_dop853_all": {
            "maximum_position_AU": 4.0e-12,
            "maximum_velocity_AU_per_year": 4.0e-12,
        },
        "jx_fine_vs_dop853_all": {
            "maximum_position_AU": 1.0e-12,
            "maximum_velocity_AU_per_year": 1.0e-12,
        },
        "jx_fine_vs_dop853_outer": _zero_residual(),
        "dop853_vs_ias15_all": _zero_residual(),
        "jx_fine_vs_de441_outer": _zero_residual(),
        "jx_full_vs_partition": _zero_residual(),
        "invariants": invariants,
        "native_source_map_guards": {
            lane: {
                "minimum_pair_path_lower_bound": 1.0,
                "minimum_jacobi_periapse": 1.0,
                "maximum_orbit_step_fraction": 1.0e-3,
                "maximum_periapse_step_fraction": 1.0e-3,
            }
            for lane in ("coarse", "fine")
        },
        "independent_source_guards": {
            lane: {
                "minimum_pair_path_lower_bound_AU": 1.0,
                "minimum_primary_two_body_periapse_AU": 1.0,
                "maximum_primary_phase_radians": 1.0e-3,
                "maximum_primary_periapse_phase_radians": 1.0e-3,
            }
            for lane in ("coarse", "fine")
        },
        "tracer_acceleration_bootstrap_exact": True,
        "jx_exact_replay": {"coarse": True, "fine": True, "partitions": True},
        "runtime_identity_stable": True,
        "code_identity_stable": True,
        "input_identity_stable": True,
        "executed_source_archive_retained": True,
    }


class M2ProtocolTests(unittest.TestCase):
    def test_v2_wire_identity_is_literal_and_v1_is_not_alias_compatible(self) -> None:
        self.assertEqual(
            qualification.PROTOCOL_SCHEMA,
            "jx.long-horizon.m2-known-source-protocol.v2",
        )
        self.assertEqual(
            qualification.RESULT_SCHEMA,
            "jx.long-horizon.m2-known-source-result.v2",
        )
        self.assertEqual(
            qualification.ARTIFACT_SCHEMA,
            "jx.long-horizon.m2-known-source-artifacts.v2",
        )
        self.assertEqual(
            qualification.COMPLETION_SCHEMA,
            "jx.long-horizon.m2-known-source-completion.v2",
        )
        self.assertEqual(
            qualification.RUN_ID,
            "jx-long-horizon-m2-known-source-10y-v2",
        )
        self.assertEqual(
            qualification.PROTOCOL_DOMAIN,
            b"jx.long-horizon.m2-known-source.protocol.v2\0",
        )
        self.assertEqual(
            qualification.RESULT_DOMAIN,
            b"jx.long-horizon.m2-known-source.result.v2\0",
        )
        self.assertEqual(
            qualification.ARTIFACT_DOMAIN,
            b"jx.long-horizon.m2-known-source.artifacts.v2\0",
        )
        self.assertEqual(
            qualification.COMPLETION_DOMAIN,
            b"jx.long-horizon.m2-known-source.completion.v2\0",
        )
        self.assertEqual(
            qualification.STATE_DOMAIN,
            b"jx.long-horizon.m2-known-source.state.v2\0",
        )
        self.assertEqual(qualification.CODE_FILES_DOMAIN, b"jx.m2.code-files.v2\0")
        self.assertEqual(
            qualification.RUNTIME_DISTRIBUTION_DOMAIN,
            b"jx.m2.runtime-distribution.v2\0",
        )
        self.assertEqual(
            qualification.SOURCE_SCHEDULE_DOMAIN,
            b"jx.long-horizon.m2-source-schedule.v2\0",
        )
        self.assertEqual(
            qualification.SOURCE_ARCHIVE_SCHEMA,
            "jx.long-horizon.m2-executed-sources.v2",
        )
        self.assertEqual(qualification.PROVENANCE_VERSION, "v2")
        self.assertEqual(
            qualification.DEFAULT_OUTPUT_ROOT,
            Path("runs/long_horizon_m2_source_preflight_v2"),
        )
        record = {"schema": qualification.PROTOCOL_SCHEMA, "value": 1}
        v2 = qualification._content_record(
            record, qualification.PROTOCOL_DOMAIN
        )["content_sha256"]
        v1 = qualification._content_record(
            record, b"jx.long-horizon.m2-known-source.protocol.v1\0"
        )["content_sha256"]
        self.assertNotEqual(v2, v1)
        self.assertEqual(qualification._provenance().version, "v2")

    def test_cli_default_uses_the_v2_output_root(self) -> None:
        fake_result = {"all_gates_passed": True, "status": "PASSED"}
        with mock.patch.object(
            qualification, "run", return_value=fake_result
        ) as run, mock.patch("builtins.print"):
            self.assertEqual(qualification.main(["--execute"]), 0)
        run.assert_called_once_with(qualification.DEFAULT_OUTPUT_ROOT, PROJECT_ROOT)

    def test_exact_lattices_roster_and_claim_ceiling_are_predeclared(self) -> None:
        self.assertEqual(qualification.COARSE_STEP_YEARS.hex(), "0x1.0000000000000p-12")
        self.assertEqual(qualification.FINE_STEP_YEARS.hex(), "0x1.0000000000000p-13")
        self.assertEqual(qualification.COARSE_STEP_COUNT, 40_960)
        self.assertEqual(qualification.FINE_STEP_COUNT, 81_920)
        self.assertEqual(qualification.ANNUAL_SAMPLE_YEARS, tuple(float(i) for i in range(11)))
        self.assertEqual(qualification.OUTER_GATE_BODY_IDS, (5, 6, 7, 8, 9))
        self.assertEqual(KNOWN_SYSTEM_BODY_IDS, (10, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        self.assertFalse(any(qualification.CLAIMS.values()))

    def test_protocol_binds_inputs_code_runtime_and_only_false_claims(self) -> None:
        fake_runtime = {
            "python_version": "test",
            "numpy_version": "test",
            "scipy_version": "test",
            "rebound_version": qualification.REQUIRED_REBOUND_VERSION,
        }
        with mock.patch.object(qualification, "_runtime_identity", return_value=fake_runtime):
            protocol = qualification.build_protocol(PROJECT_ROOT)
        self.assertEqual(protocol["schema"], qualification.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["runtime_identity"], fake_runtime)
        self.assertEqual(
            protocol["inputs"]["known_system_model_sha256"],
            KNOWN_SOLAR_SYSTEM_V2.content_sha256,
        )
        self.assertRegex(protocol["code_identity"]["files_content_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            protocol["model"]["inner_de441_residuals"], "DESCRIPTIVE_ONLY"
        )
        self.assertTrue(protocol["artifacts"]["create_only"])
        self.assertFalse(any(protocol["claims"].values()))
        unhashed = {key: value for key, value in protocol.items() if key != "content_sha256"}
        expected = qualification._content_record(
            unhashed, qualification.PROTOCOL_DOMAIN
        )["content_sha256"]
        self.assertEqual(protocol["content_sha256"], expected)

    def test_wrong_rebound_version_fails_before_execution(self) -> None:
        fake_scipy = mock.Mock(__version__="test")
        fake_rk = mock.Mock(__file__=__file__)
        fake_rebound = mock.Mock(__version__="4.4.11", __file__=__file__)
        modules = {
            "scipy": fake_scipy,
            "scipy.integrate._ivp.rk": fake_rk,
            "rebound": fake_rebound,
        }
        with mock.patch.dict("sys.modules", modules):
            with self.assertRaisesRegex(
                qualification.M2SourceQualificationError, "REBOUND must be exactly 5.1.1"
            ):
                qualification._runtime_identity()


class M2InputAndNumericsTests(unittest.TestCase):
    def test_retained_reference_has_exact_annual_roster_and_genesis(self) -> None:
        epochs, positions, velocities = qualification._load_de441_reference(PROJECT_ROOT)
        self.assertEqual(epochs.shape, (11,))
        self.assertEqual(positions.shape, (11, 10, 3))
        self.assertEqual(velocities.shape, (11, 10, 3))
        expected_position, expected_velocity, _ = qualification._known_arrays()
        np.testing.assert_array_equal(positions[0], expected_position)
        np.testing.assert_array_equal(velocities[0], expected_velocity)
        np.testing.assert_array_equal(positions[:, 0], np.zeros((11, 3)))
        np.testing.assert_array_equal(velocities[:, 0], np.zeros((11, 3)))

    def test_independent_newtonian_rhs_obeys_two_body_action_reaction(self) -> None:
        gm = np.asarray((2.0, 3.0), dtype=np.float64)
        positions = np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)), dtype=np.float64)
        velocities = np.asarray(((0.0, 1.0, 0.0), (0.0, -1.0, 0.0)), dtype=np.float64)
        state = np.concatenate((positions.reshape(-1), velocities.reshape(-1)))
        derivative = qualification._newtonian_rhs(0.0, state, gm)
        np.testing.assert_array_equal(derivative[:6].reshape(2, 3), velocities)
        acceleration = derivative[6:].reshape(2, 3)
        np.testing.assert_allclose(acceleration[0], (0.75, 0.0, 0.0), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(acceleration[1], (-0.5, 0.0, 0.0), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(
            gm[0] * acceleration[0] + gm[1] * acceleration[1],
            np.zeros(3),
            rtol=0.0,
            atol=0.0,
        )

    def test_residual_summary_separates_selected_outer_roster(self) -> None:
        left = np.zeros((11, 10, 3), dtype=np.float64)
        right = left.copy()
        right[-1, 1, 0] = 100.0  # Mercury: descriptive for the selected call.
        right[-1, 5, 0] = 2.0    # Jupiter: hard-gate roster.
        summary = qualification._residual_summary(
            left,
            left,
            right,
            left,
            qualification.OUTER_GATE_INDICES,
        )
        self.assertEqual(summary["body_ids"], [5, 6, 7, 8, 9])
        self.assertEqual(summary["maximum_position_AU"], 2.0)
        self.assertNotIn("1", summary["by_body"])

    def test_bitwise_genesis_comparison_rejects_signed_zero_alias(self) -> None:
        positive = np.asarray((0.0, 1.0), dtype=np.float64)
        negative = np.asarray((-0.0, 1.0), dtype=np.float64)
        self.assertTrue(np.array_equal(positive, negative))
        self.assertFalse(qualification._array_bitwise_equal(positive, negative))

    def test_conservation_is_zero_for_an_unchanged_state(self) -> None:
        position, velocity, gm = qualification._known_arrays()
        bary_position, bary_velocity = qualification._barycentric(position, velocity, gm)
        result = qualification._invariants(
            np.stack((bary_position, bary_position)),
            np.stack((bary_velocity, bary_velocity)),
            gm,
        )
        self.assertEqual(result["maximum_relative_energy_drift"], 0.0)
        self.assertEqual(result["maximum_relative_angular_momentum_drift"], 0.0)

    def test_all_node_conservation_exposes_an_interior_excursion(self) -> None:
        position, velocity, gm = qualification._known_arrays()
        bary_position, bary_velocity = qualification._barycentric(position, velocity, gm)
        positions = np.stack((bary_position, bary_position, bary_position))
        velocities = np.stack((bary_velocity, bary_velocity, bary_velocity))
        velocities[1, 1, 0] += 1.0e-3
        result = qualification._invariants(positions, velocities, gm)
        self.assertEqual(result["sample_count"], 3)
        self.assertGreater(result["maximum_relative_energy_drift"], 0.0)

    def test_independent_source_guard_accepts_known_state_and_detects_sweep(self) -> None:
        position, velocity, gm = qualification._known_arrays()
        safe = qualification._independent_source_guard_metrics(
            np.stack((position, position + velocity * qualification.COARSE_STEP_YEARS)),
            np.stack((velocity, velocity)),
            gm,
            qualification.COARSE_STEP_YEARS,
        )
        self.assertGreater(
            safe["minimum_pair_path_lower_bound_AU"],
            qualification.INDEPENDENT_PAIR_SEPARATION_FLOOR_AU,
        )

        unsafe_position = np.zeros((2, 2, 3), dtype=np.float64)
        unsafe_velocity = np.zeros_like(unsafe_position)
        unsafe_position[:, 1, 0] = 0.02
        unsafe_velocity[0, 1, 0] = -100.0
        unsafe_velocity[:, 1, 1] = 1.0
        unsafe = qualification._independent_source_guard_metrics(
            unsafe_position,
            unsafe_velocity,
            np.asarray((1.0, 1.0e-6), dtype=np.float64),
            1.0e-4,
        )
        self.assertLessEqual(
            unsafe["minimum_pair_path_lower_bound_AU"],
            qualification.INDEPENDENT_PAIR_SEPARATION_FLOOR_AU,
        )

    def test_independent_guard_rejects_underresolved_periapse_timescale(self) -> None:
        gm = np.asarray((1.0, 1.0e-6), dtype=np.float64)
        position = np.zeros((2, 2, 3), dtype=np.float64)
        velocity = np.zeros_like(position)
        position[:, 1, 0] = 1.0
        mu = float(np.sum(gm))
        target_periapse = 0.0500000000000001
        semimajor_axis = (1.0 + target_periapse) / 2.0
        velocity[:, 1, 1] = np.sqrt(mu * (2.0 - 1.0 / semimajor_axis))
        result = qualification._independent_source_guard_metrics(
            position, velocity, gm, 1.0e-3
        )
        self.assertGreater(result["minimum_primary_two_body_periapse_AU"], 0.05)
        self.assertLess(
            result["maximum_primary_phase_radians"],
            qualification.GATE_THRESHOLDS[
                "independent_maximum_primary_phase_radians"
            ],
        )
        self.assertGreater(
            result["maximum_primary_periapse_phase_radians"],
            qualification.GATE_THRESHOLDS[
                "independent_maximum_primary_periapse_phase_radians"
            ],
        )


class M2GateAndCustodyTests(unittest.TestCase):
    def test_predeclared_gates_pass_and_cannot_hide_a_failure(self) -> None:
        metrics = _passing_metrics()
        gates = qualification._evaluate_gates(metrics)
        self.assertTrue(all(row["passed"] for row in gates.values()))
        metrics["jx_fine_vs_de441_outer"]["maximum_position_AU"] = 1.0  # type: ignore[index]
        gates = qualification._evaluate_gates(metrics)
        self.assertFalse(gates["outer_de441_position"]["passed"])
        self.assertEqual(
            gates["outer_de441_position"]["threshold"],
            qualification.GATE_THRESHOLDS["outer_de441_position_AU"],
        )
        metrics = _passing_metrics()
        metrics["independent_source_guards"]["coarse"][  # type: ignore[index]
            "minimum_pair_path_lower_bound_AU"
        ] = 0.0
        gates = qualification._evaluate_gates(metrics)
        self.assertFalse(gates["coarse_independent_pair_path"]["passed"])

    def test_result_records_can_serialize_frozen_runtime_trees(self) -> None:
        frozen = {"outer": mock.MagicMock()}
        # Exercise the real MappingProxy/tuple shape used by source receipts.
        from types import MappingProxyType

        value = MappingProxyType(
            {"child": MappingProxyType({"values": (1, 2, 3)})}
        )
        plain = qualification._plain(value)
        self.assertEqual(plain, {"child": {"values": [1, 2, 3]}})
        json.dumps(plain, allow_nan=False)
        self.assertIn("outer", frozen)

    def test_create_only_json_refuses_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "PROTOCOL.json"
            qualification._write_json_create_only(path, {"status": "first"})
            with self.assertRaises(FileExistsError):
                qualification._write_json_create_only(path, {"status": "second"})
            self.assertEqual(json.loads(path.read_text(encoding="ascii")), {"status": "first"})

    def test_verified_input_and_artifact_roster_reject_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"evidence")
            link = root / "input"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                qualification.M2SourceQualificationError, "symlink"
            ):
                qualification._verified_file(
                    root,
                    Path("input"),
                    hashlib.sha256(b"evidence").hexdigest(),
                )

            artifact = root / "PROTOCOL.json"
            artifact.write_bytes(b"{}\n")
            unexpected = root / "unexpected"
            unexpected.mkdir()
            with self.assertRaisesRegex(
                qualification.M2SourceQualificationError, "roster changed"
            ):
                qualification._artifact_inventory(root, ("PROTOCOL.json", "target", "input"))

    def test_post_protocol_exception_is_retained_as_failed_execution(self) -> None:
        protocol = qualification._content_record(
            {
                "schema": qualification.PROTOCOL_SCHEMA,
                "run_id": qualification.RUN_ID,
                "status": "PREDECLARED_BEFORE_EXECUTION",
                "claims": dict(qualification.CLAIMS),
            },
            qualification.PROTOCOL_DOMAIN,
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "attempt"
            with (
                mock.patch.object(
                    qualification, "build_protocol", return_value=protocol
                ),
                mock.patch.object(
                    qualification,
                    "_execute_after_protocol",
                    side_effect=RuntimeError("injected reference failure"),
                ),
            ):
                result = qualification.run(output, PROJECT_ROOT)
            self.assertEqual(result["status"], "FAILED_EXECUTION")
            self.assertFalse(result["all_gates_passed"])
            self.assertEqual(
                result["execution_error"]["message"],
                "injected reference failure",
            )
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "ARTIFACTS.json",
                    qualification.COMPLETION_MARKER_NAME,
                    "PROTOCOL.json",
                    "RESULT.json",
                ],
            )
            manifest = json.loads(
                (output / "ARTIFACTS.json").read_text(encoding="ascii")
            )
            self.assertEqual(manifest["file_count"], 2)


class M2ExactRuntimeExecutionTests(unittest.TestCase):
    def test_real_full_run_retains_independent_all_node_evidence(self) -> None:
        import rebound

        self.assertEqual(rebound.__version__, qualification.REQUIRED_REBOUND_VERSION)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "m2"
            result = qualification.run(output, PROJECT_ROOT)
            self.assertEqual(result["status"], "PASSED")
            self.assertTrue(result["all_gates_passed"])
            self.assertTrue(all(row["passed"] for row in result["gates"].values()))
            self.assertEqual(
                result["metrics"]["invariants"]["jx_coarse"]["sample_count"],
                qualification.COARSE_STEP_COUNT + 1,
            )
            self.assertEqual(
                result["metrics"]["invariants"]["jx_fine"]["sample_count"],
                qualification.FINE_STEP_COUNT + 1,
            )
            self.assertEqual(
                result["metrics"]["independent_source_guards"]["coarse"][
                    "node_count"
                ],
                qualification.COARSE_STEP_COUNT + 1,
            )
            with np.load(output / "STATES.npz", allow_pickle=False) as archive:
                gm = np.ascontiguousarray(archive["gm_AU3_per_year2"])
                coarse_position = np.ascontiguousarray(
                    archive["jx_coarse_all_node_position_AU"]
                )
                coarse_velocity = np.ascontiguousarray(
                    archive["jx_coarse_all_node_velocity_AU_per_year"]
                )
                fine_position = np.ascontiguousarray(
                    archive["jx_fine_all_node_position_AU"]
                )
                fine_velocity = np.ascontiguousarray(
                    archive["jx_fine_all_node_velocity_AU_per_year"]
                )
            self.assertEqual(
                qualification._invariants(
                    coarse_position,
                    coarse_velocity,
                    gm,
                    primary_relative=True,
                    sample_scope="ALL_SOURCE_MAP_NODES",
                ),
                result["metrics"]["invariants"]["jx_coarse"],
            )
            self.assertEqual(
                qualification._invariants(
                    fine_position,
                    fine_velocity,
                    gm,
                    primary_relative=True,
                    sample_scope="ALL_SOURCE_MAP_NODES",
                ),
                result["metrics"]["invariants"]["jx_fine"],
            )
            self.assertEqual(
                qualification._independent_source_guard_metrics(
                    coarse_position,
                    coarse_velocity,
                    gm,
                    qualification.COARSE_STEP_YEARS,
                ),
                result["metrics"]["independent_source_guards"]["coarse"],
            )
            self.assertEqual(
                qualification._independent_source_guard_metrics(
                    fine_position,
                    fine_velocity,
                    gm,
                    qualification.FINE_STEP_YEARS,
                ),
                result["metrics"]["independent_source_guards"]["fine"],
            )
            self.assertTrue((output / qualification.SOURCE_ARCHIVE_NAME).is_file())
            self.assertTrue(
                (output / qualification.COMPLETION_MARKER_NAME).is_file()
            )
            manifest = json.loads(
                (output / "ARTIFACTS.json").read_text(encoding="ascii")
            )
            self.assertEqual(manifest["file_count"], 4)
            with zipfile.ZipFile(output / qualification.SOURCE_ARCHIVE_NAME) as source:
                names = set(source.namelist())
                self.assertIn(
                    "benchmarks/jx_long_horizon_m2_source_preflight.py", names
                )
                self.assertIn(
                    "src/jxplanetx/data/long_horizon_known_system_v2/MANIFEST.json",
                    names,
                )
                archived_manifest = json.loads(source.read("MANIFEST.json"))
            self.assertEqual(
                archived_manifest["schema"],
                "jx.long-horizon.m2-executed-sources.v2",
            )
            self.assertEqual(
                archived_manifest["files_content_sha256"],
                result["executed_source_archive"]["files_content_sha256"],
            )
            completion = json.loads(
                (output / qualification.COMPLETION_MARKER_NAME).read_text(
                    encoding="ascii"
                )
            )
            self.assertEqual(completion["result_status"], "PASSED")
            self.assertEqual(
                completion["result_content_sha256"], result["content_sha256"]
            )
            self.assertEqual(
                sorted(path.name for path in output.iterdir()),
                [
                    "ARTIFACTS.json",
                    qualification.COMPLETION_MARKER_NAME,
                    qualification.SOURCE_ARCHIVE_NAME,
                    "PROTOCOL.json",
                    "RESULT.json",
                    "STATES.npz",
                ],
            )


if __name__ == "__main__":
    unittest.main()
