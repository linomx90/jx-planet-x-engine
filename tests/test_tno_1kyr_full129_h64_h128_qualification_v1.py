from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import types
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT
    / "benchmarks/jx_tno_1kyr_full129_h64_h128_qualification_v1.py"
)


def load_module() -> types.ModuleType:
    name = (
        f"_test_tno_1kyr_full129_h64_h128_{os.getpid()}_"
        f"{id(object())}_{sys.flags.optimize}"
    )
    payload = SOURCE.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    module.__spec__ = importlib.util.spec_from_loader(name, loader=None)
    sys.modules[name] = module
    try:
        exec(
            compile(
                payload,
                str(SOURCE),
                "exec",
                dont_inherit=True,
                optimize=sys.flags.optimize,
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


m = load_module()


class NativeEpochH64H128QualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = m.build_protocol()

    def test_protocol_is_exact_canonical_and_parent_tag_bound(self) -> None:
        protocol = self.protocol
        self.assertEqual(m.validate_protocol(protocol), protocol["content_sha256"])
        self.assertEqual(
            protocol["content_sha256"],
            m._domain_hash(m.PROTOCOL_DOMAIN, protocol),
        )
        payload = m._canonical_json(protocol)
        self.assertEqual(m._strict_json(payload), protocol)
        authority = protocol["authority"]
        git = authority["git_authority"]
        self.assertEqual(git["parent_tag"], m.PARENT_TAG)
        self.assertEqual(git["parent_tag_object"], m.PARENT_TAG_OBJECT)
        self.assertEqual(git["parent_commit"], m.PARENT_COMMIT)
        self.assertEqual(git["parent_tree"], m.PARENT_TREE)
        self.assertEqual(
            authority["parent_protocol_content_sha256"],
            m.PARENT_PROTOCOL_CONTENT_SHA256,
        )

    def test_all_twelve_all_129_native_epoch_roster_no_selection(self) -> None:
        scope = self.protocol["scope"]
        self.assertEqual(scope["initial_epoch_policy"], "TWELVE_NATIVE_COVARIANCE_EPOCHS")
        self.assertIs(scope["common_epoch_propagation_performed"], False)
        self.assertIs(scope["common_epoch_gate_satisfied"], False)
        self.assertEqual(scope["batch_count"], 12)
        self.assertEqual(scope["batch_orders"], list(range(12)))
        self.assertEqual(scope["tracer_count_per_batch"], 129)
        self.assertEqual(scope["tracer_orders"], list(range(129)))
        self.assertEqual(scope["total_tracer_states_per_resolution"], 1_548)
        batches = scope["native_epoch_batches"]
        self.assertEqual(len(batches), 12)
        self.assertEqual([row["batch_order"] for row in batches], list(range(12)))
        self.assertEqual(len({row["object_id"] for row in batches}), 12)
        self.assertTrue(all(row["tracer_orders"] == list(range(129)) for row in batches))
        self.assertTrue(
            all(
                type(row["covariance_epoch_jd_tdb_hex"]) is str
                and row["covariance_epoch_jd_tdb_hex"].startswith("0x")
                for row in batches
            )
        )

    def test_exact_h64_h128_geometry_and_no_128_bit_claim(self) -> None:
        resolutions = self.protocol["resolutions"]
        h64 = resolutions["h64_control"]
        h128 = resolutions["h128_candidate"]
        self.assertEqual(h64["source_step_julian_years_binary64_hex"], (1.0 / 64.0).hex())
        self.assertEqual(h128["source_step_julian_years_binary64_hex"], (1.0 / 128.0).hex())
        self.assertEqual(h64["fixed_steps_per_batch"], 64_000)
        self.assertEqual(h128["fixed_steps_per_batch"], 128_000)
        self.assertEqual(h64["trajectory_count"], 1_548)
        self.assertEqual(h128["trajectory_count"], 1_548)
        self.assertEqual(h64["charged_gpu_lane_steps"], 99_072_000)
        self.assertEqual(h128["charged_gpu_lane_steps"], 198_144_000)
        self.assertEqual(
            h64["annual_checkpoint_years_inclusive"], list(range(1_001))
        )
        self.assertEqual(
            h128["annual_checkpoint_years_inclusive"], list(range(1_001))
        )
        self.assertEqual(h64["annual_checkpoint_count_per_trajectory"], 1_001)
        self.assertEqual(h128["annual_checkpoint_count_per_trajectory"], 1_001)
        self.assertEqual(h64["arithmetic_dtype"], "IEEE754_BINARY64")
        self.assertEqual(h128["arithmetic_dtype"], "IEEE754_BINARY64")
        self.assertIs(
            h64["resolution_label_encodes_steps_per_julian_year_not_bit_width"],
            True,
        )
        self.assertIs(
            h128["resolution_label_encodes_steps_per_julian_year_not_bit_width"],
            True,
        )
        arithmetic = self.protocol["exact_arithmetic"]
        self.assertEqual(arithmetic["dual_fixed_steps_per_batch"], 192_000)
        self.assertEqual(arithmetic["dual_charged_gpu_lane_steps"], 297_216_000)
        self.assertEqual(arithmetic["dual_cpu_source_primary_steps"], 2_304_000)
        self.assertEqual(
            arithmetic["dual_cpu_source_primary_plus_replay_steps"],
            4_608_000,
        )
        self.assertEqual(arithmetic["dual_tracer_checkpoint_states"], 3_099_096)

    def test_parent_threshold_seal_is_inherited_exactly(self) -> None:
        parent_payload = (
            PROJECT_ROOT
            / "runs/tno_3gyr_full129_robustness_v1/PROTOCOL.json"
        ).read_bytes()
        parent = m._strict_json(parent_payload)
        expected = parent["cpu_gpu_integrator"][
            "short_horizon_qualification_contract"
        ]
        observed = self.protocol["sealed_parent_short_horizon_contract"]
        self.assertEqual(observed, expected)
        self.assertEqual(observed["richardson_cartesian_position_au_max"], "1/500")
        self.assertEqual(
            observed["richardson_cartesian_velocity_au_per_julian_year_max"],
            "1/50000",
        )
        self.assertEqual(observed["richardson_normalized_semimajor_axis_max"], "3/8000")
        self.assertEqual(observed["h128_vs_ias15_cartesian_position_au_max"], "1/500")
        self.assertEqual(
            observed["h128_vs_ias15_cartesian_velocity_au_per_julian_year_max"],
            "1/50000",
        )
        self.assertEqual(observed["h128_vs_ias15_normalized_semimajor_axis_max"], "3/4000")
        self.assertEqual(
            observed["h128_vs_ias15_scattering_diagnostic_disagreement_max"],
            "3/4000",
        )
        self.assertIs(observed["scope"]["common_epoch_bridge_must_pass_first"], True)
        precursor = self.protocol["native_epoch_precursor_interpretation"]
        self.assertIs(precursor["all_parent_numerical_thresholds_reused_without_change"], True)
        self.assertIn("NATIVE_EPOCH_ENGINEERING_EVIDENCE_ONLY", precursor["promotion_rule"])

    def test_integrator_precision_reference_and_restart_boundaries(self) -> None:
        contract = self.protocol["integrator_contract"]
        source = contract["massive_source"]
        quiet = contract["quiet_tracer"]
        route = contract["guarded_route"]
        reference = contract["reference"]
        self.assertEqual(source["method"], "COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN")
        self.assertIs(source["same_runtime_complete_bitwise_replay_required"], True)
        self.assertIs(source["h64_and_h128_1kyr_fit_one_existing_tile"], True)
        self.assertEqual(quiet["dtype"], "float64")
        self.assertIs(quiet["cpu_fallback_allowed"], False)
        self.assertIs(quiet["deterministic_reductions"], True)
        self.assertIs(quiet["fast_math"], False)
        self.assertEqual(quiet["cuda_compiler_options"], ["--std=c++11", "--fmad=false"])
        self.assertEqual(route["method"], "CONTINUOUS_RESTRICTED_FRAME_DOP853")
        self.assertEqual(route["outer_boundary_AU_hex"], float(4096.0).hex())
        self.assertEqual(route["outer_boundary_label"], "MODEL_DOMAIN_EXIT_NOT_PHYSICAL_EJECTION")
        self.assertIs(route["physical_collision_claim"], False)
        self.assertIs(
            reference[
                "rebound_5_1_1_may_satisfy_independent_requirement_if_exactly_pinned_and_audited"
            ],
            True,
        )
        self.assertIs(reference["implementation_and_runtime_currently_pinned"], False)
        restart = self.protocol["restart_qualification_contract"]
        self.assertEqual(restart["h64_restart_global_step"], 32_000)
        self.assertEqual(restart["h128_restart_global_step"], 64_000)
        self.assertIs(restart["deterministic_guarded_route_fixture_required"], True)
        self.assertIs(restart["deterministic_removed_lane_fixture_required"], True)
        self.assertIs(restart["precommit_interruption_fixture_required"], True)
        self.assertIs(restart["existing_batch0_all_active_h64_restart_is_not_general_authority"], True)

    def test_runtime_is_unpinned_and_unsafe_legacy_reuse_is_forbidden(self) -> None:
        runtime = self.protocol["runtime_contract"]
        self.assertTrue(runtime["numpy_requirement"].endswith("EXECUTION_SUCCESSOR"))
        self.assertTrue(runtime["scipy_requirement"].endswith("EXECUTION_SUCCESSOR"))
        self.assertTrue(runtime["cupy_requirement"].endswith("TO_BE_PINNED"))
        self.assertEqual(
            runtime["ias15_requirement"],
            "ONE_EXACT_REBOUND_5_1_1_RUNTIME_TO_BE_PINNED_AND_AUDITED",
        )
        observed = runtime["observed_setup_diagnostic_only"]
        self.assertEqual(observed["python"], "3.14.4")
        self.assertEqual(observed["numpy"], "2.5.3")
        self.assertEqual(observed["cupy"], "14.2.0")
        self.assertIs(observed["target_stack_qualified_by_this_observation"], False)
        self.assertIs(runtime["visible_cuda_device_required_for_future_execution"], True)
        self.assertIs(runtime["native_extension_binary_is_platform_specific"], True)
        blockers = self.protocol["execution_blockers"]
        self.assertIn(
            "INDEPENDENT_ALL129_CPU_IAS15_IMPLEMENTATION_AND_RUNTIME_NOT_PINNED",
            blockers,
        )
        self.assertIn(
            "EXACT_CUPY_CUDA_DRIVER_DEVICE_AND_NVRTC_RUNTIME_NOT_PINNED",
            blockers,
        )
        exclusions = self.protocol["unsafe_reuse_exclusions"]
        self.assertIn(
            "DO_NOT_TREAT_UNPINNED_OR_REUSED_REBOUND_OUTPUT_AS_THE_TARGET_REFERENCE",
            exclusions,
        )
        self.assertIn(
            "DO_NOT_REUSE_SELECTED9_STAGE1_RESULTS_AS_FULL129_EVIDENCE",
            exclusions,
        )

    def test_claims_are_exact_bools_and_keep_parent_gate_false(self) -> None:
        claims = self.protocol["claims"]
        self.assertEqual(claims, m._claims())
        self.assertTrue(all(type(value) is bool for value in claims.values()))
        self.assertIs(claims["prepare_only_protocol_frozen"], True)
        self.assertTrue(
            all(
                value is False
                for key, value in claims.items()
                if key != "prepare_only_protocol_frozen"
            )
        )
        self.assertIs(claims["common_epoch_full_covariance_bridge_qualified"], False)
        self.assertIs(claims["three_gyr_short_horizon_gate_satisfied"], False)
        self.assertIs(claims["all_12_h64_qualification_passed"], False)
        self.assertIs(claims["all_12_h128_qualification_passed"], False)
        self.assertIs(claims["science_result"], False)

    def test_build_imports_no_optional_dynamics_runtime(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imported_roots: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".")[0])
        forbidden = {"jxplanetx", "cupy", "cupyx", "scipy", "rebound"}
        self.assertTrue(forbidden.isdisjoint(imported_roots))

        before = set(sys.modules)
        m.build_protocol()
        added = set(sys.modules) - before
        forbidden = ("jxplanetx", "cupy", "cupyx", "scipy", "rebound")
        self.assertFalse(
            any(
                name == prefix or name.startswith(prefix + ".")
                for name in added
                for prefix in forbidden
            )
        )

    def test_public_cli_is_prepare_only(self) -> None:
        self.assertEqual(
            self.protocol["public_cli"],
            ["build", "audit-prerequisites", "prepare", "validate-prepared"],
        )
        self.assertEqual(
            self.protocol["explicitly_absent_public_cli"],
            ["execute", "execute-worker", "resume", "validate-result"],
        )
        parser = m._parser()
        for forbidden in self.protocol["explicitly_absent_public_cli"]:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args([forbidden])

    def test_prepare_is_create_only_private_and_exact(self) -> None:
        self.assertEqual(m.validate_prepared(m.DEFAULT_OUTPUT_ROOT), self.protocol)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "prepared"
            prepared = m.prepare(root)
            self.assertEqual(prepared, self.protocol)
            self.assertEqual({path.name for path in root.iterdir()}, {"PROTOCOL.json"})
            self.assertEqual(stat.S_IMODE(root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE((root / "PROTOCOL.json").stat().st_mode), 0o600)
            self.assertEqual(m.validate_prepared(root), self.protocol)
            payload = (root / "PROTOCOL.json").read_bytes()
            self.assertEqual(payload, m._canonical_json(self.protocol))
            self.assertEqual(
                hashlib.sha256(payload).hexdigest(),
                hashlib.sha256(m._canonical_json(self.protocol)).hexdigest(),
            )
            with self.assertRaises(m.TNOH64H128QualificationError):
                m.prepare(root)
            root.chmod(0o755)
            with self.assertRaises(m.TNOH64H128QualificationError):
                m.validate_prepared(root)
    def test_validate_prepared_rejects_tamper_extra_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            base = Path(raw)
            root = base / "prepared"
            m.prepare(root)
            (root / "EXTRA").write_bytes(b"unexpected")
            with self.assertRaises(m.TNOH64H128QualificationError):
                m.validate_prepared(root)

            root2 = base / "prepared2"
            m.prepare(root2)
            protocol_path = root2 / "PROTOCOL.json"
            protocol_path.chmod(0o600)
            payload = bytearray(protocol_path.read_bytes())
            payload[-2] = ord(" ")
            protocol_path.write_bytes(bytes(payload))
            protocol_path.chmod(0o600)
            with self.assertRaises(m.TNOH64H128QualificationError):
                m.validate_prepared(root2)

            real = base / "real"
            real.mkdir()
            link = base / "link"
            link.symlink_to(real, target_is_directory=True)
            with self.assertRaises(m.TNOH64H128QualificationError):
                m.prepare(link / "prepared3")

    def test_strict_json_rejects_duplicate_float_nonfinite_and_bool_alias(self) -> None:
        bad_payloads = (
            b'{"a":1,"a":2}\n',
            b'{"a":1.5}\n',
            b'{"a":NaN}\n',
            b'[]\n',
        )
        for payload in bad_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(m.TNOH64H128QualificationError):
                    m._strict_json(payload)
        forged = dict(self.protocol)
        forged["scope"] = dict(forged["scope"])
        forged["scope"]["batch_count"] = True
        forged["content_sha256"] = m._domain_hash(m.PROTOCOL_DOMAIN, forged)
        with self.assertRaises(m.TNOH64H128QualificationError):
            m.validate_protocol(forged)

    def test_file_pins_cover_every_executable_component_without_legacy_ias15(self) -> None:
        roles = {pin.role for pin in m.FILE_PINS}
        self.assertEqual(len(roles), len(m.FILE_PINS))
        self.assertTrue(
            {
                "cpu_massive_source",
                "cpu_massive_map",
                "cpu_massive_native_extension",
                "gpu_quiet_tracer",
                "gpu_all_conic_kepler",
                "gpu_cpu_hybrid",
                "cpu_dop853_route",
                "cpu_dop853_event_locator",
                "fixed_lane_lifecycle",
                "replay_verified_checkpoint",
                "backend_contract",
                "backend_resolver",
                "dependency_metadata",
            }.issubset(roles)
        )
        self.assertFalse(any("ias15_gate" in pin.relative_path for pin in m.FILE_PINS))
        for pin in m.FILE_PINS:
            self.assertEqual(len(pin.sha256), 64)
            self.assertTrue(all(character in "0123456789abcdef" for character in pin.sha256))


if __name__ == "__main__":
    unittest.main()
