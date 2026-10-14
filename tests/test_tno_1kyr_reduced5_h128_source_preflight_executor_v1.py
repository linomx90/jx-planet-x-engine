from __future__ import annotations

import ast
import base64
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "benchmarks"
    / "jx_tno_1kyr_reduced5_h128_source_preflight_executor_v1.py"
)
PROTOCOL = ROOT / "runs" / "tno_1kyr_reduced5_h128_source_preflight_v1" / "PROTOCOL.json"


def load_module():
    before = frozenset(sys.modules)
    spec = importlib.util.spec_from_file_location(
        "jx_tno_1kyr_reduced5_h128_source_preflight_executor_v1_test_subject",
        SOURCE,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load source-preflight executor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    added = frozenset(sys.modules).difference(before)
    return module, added


MODULE, IMPORT_ADDED = load_module()
import numpy as NP

PROTOCOL_VALUE = json.loads(PROTOCOL.read_text(encoding="utf-8"))


class SourcePreflightExecutorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = MODULE
        cls.np = NP
        cls.protocol = PROTOCOL_VALUE

    def test_import_is_runtime_lazy_and_cpu_only(self) -> None:
        forbidden = ("numpy", "rebound", "jxplanetx", "cupy")
        self.assertFalse(
            any(
                name == prefix or name.startswith(prefix + ".")
                for name in IMPORT_ADDED
                for prefix in forbidden
            )
        )

    def test_exact_preregistration_authority_and_every_input_binding(self) -> None:
        protocol, payload, authority = self.m._load_sealed_authority()
        self.assertEqual(payload, PROTOCOL.read_bytes())
        self.assertEqual(protocol, self.protocol)
        self.assertEqual(
            authority["protocol_file_sha256"],
            "e37730efc8cee79fadfc8324d8e33b1a97b9732df92b0cb1162529c439601141",
        )
        self.assertEqual(
            authority["protocol_content_sha256"],
            "1c275624d21f815c7a2b9c9a8e18f77f63caf9356cf383bd1de3a6ad7228ae66",
        )
        self.assertEqual(
            authority["preregistration_tag"],
            "tno-1kyr-reduced5-h128-source-preflight-v1-preregistered-20261014",
        )
        self.assertEqual(
            authority["preregistration_tag_object"],
            "e6b0d930eb15c9e640476b5592f2594f3f79f0cc",
        )
        self.assertEqual(authority["input_binding_count"], 15)
        self.assertIs(authority["every_input_binding_verified"], True)

    def test_successor_commit_cannot_rewrite_any_sealed_authority_blob(self) -> None:
        successor = "f" * 40
        changed_path = self.m.SEALED_PROTOCOL_RELATIVE_PATH

        def fake_git(*arguments, input_payload=None):
            del input_payload
            self.assertEqual(arguments[0], "show")
            revision, relative = arguments[1].split(":", 1)
            if revision == successor and relative == changed_path:
                return b"changed"
            return b"preregistered"

        with mock.patch.object(self.m, "_git", side_effect=fake_git):
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError,
                "changed sealed authority blob",
            ):
                self.m._verify_successor_preserves_sealed_authority(successor)

    def test_successor_tree_scope_is_exactly_two_additions(self) -> None:
        successor = "f" * 40
        test_payload = b"exact committed test payload\n"
        exact_diff = (
            f"A\t{self.m.SCRIPT_PATH.relative_to(self.m.PROJECT_ROOT).as_posix()}\n"
            f"A\t{self.m.EXECUTOR_TEST_RELATIVE_PATH}\n"
        ).encode("utf-8")

        def exact_git(*arguments, input_payload=None):
            del input_payload
            if arguments[0] == "diff-tree":
                return exact_diff
            if arguments == (
                "show",
                f"{successor}:{self.m.EXECUTOR_TEST_RELATIVE_PATH}",
            ):
                return test_payload
            raise AssertionError(f"unexpected git call: {arguments!r}")

        with mock.patch.object(self.m, "_git", side_effect=exact_git):
            receipt = self.m._verify_successor_add_only_scope(
                successor, test_payload
            )
        self.assertEqual(receipt["test_bytes"], len(test_payload))
        self.assertEqual(receipt["test_sha256"], hashlib.sha256(test_payload).hexdigest())

        changed_diff = exact_diff + b"M\tsrc/jxplanetx/cuda_restricted_wh.py\n"
        with mock.patch.object(self.m, "_git", return_value=changed_diff):
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError,
                "add only the exact executor and test",
            ):
                self.m._verify_successor_add_only_scope(
                    successor, test_payload
                )

    def test_numpy_bundled_shared_libraries_are_record_bound(self) -> None:
        def record_row(relative, payload):
            digest = base64.urlsafe_b64encode(
                hashlib.sha256(payload).digest()
            ).decode("ascii").rstrip("=")
            return f"{relative},sha256={digest},{len(payload)}"

        init_payload = b"numpy initializer"
        openblas_payload = b"pinned openblas bytes"
        record_payload = (
            record_row("numpy/__init__.py", init_payload)
            + "\n"
            + record_row(
                "numpy.libs/libscipy_openblas64_-test.so",
                openblas_payload,
            )
            + "\n"
        ).encode("utf-8")
        manifest = self.m._parse_numpy_record_manifest(record_payload)
        identity = manifest["numpy.libs/libscipy_openblas64_-test.so"]
        libraries = [
            {
                "locator": "LOADED_SHARED_LIBRARY:libscipy_openblas64_-test.so",
                "bytes": identity[0],
                "sha256": identity[1],
            }
        ]
        self.m._validate_numpy_bundled_shared_libraries(libraries, manifest)
        tampered = [dict(libraries[0], sha256="0" * 64)]
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "differs from its pinned RECORD",
        ):
            self.m._validate_numpy_bundled_shared_libraries(tampered, manifest)
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "differs from its pinned RECORD",
        ):
            self.m._validate_numpy_bundled_shared_libraries([], manifest)

    def test_source_map_metric_roster_is_exact_ordered_and_nonnegative(self) -> None:
        metrics = [
            {"name": name, "value_hex": 1.0.hex()}
            for name in self.m.SOURCE_MAP_METRIC_NAMES
        ]
        self.assertEqual(
            self.m._validate_source_map_metrics_hex(metrics),
            (1.0,) * len(self.m.SOURCE_MAP_METRIC_NAMES),
        )
        malformed = [dict(item) for item in metrics]
        malformed[1]["name"] = malformed[0]["name"]
        with self.assertRaises(self.m.SourcePreflightExecutionError):
            self.m._validate_source_map_metrics_hex(malformed)
        malformed = [dict(item) for item in metrics]
        malformed[-1]["value_hex"] = (-1.0).hex()
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "negative",
        ):
            self.m._validate_source_map_metrics_hex(malformed)
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "roster",
        ):
            self.m._validate_source_map_metrics_hex(metrics[:-1])

    def test_retained_nodes_bind_tile_digest_and_coordinate_metrics(self) -> None:
        np = self.np
        gm, raw_positions, raw_velocities = self.m._initial_arrays(
            np, self.protocol
        )
        (
            _,
            _,
            canonical_positions,
            canonical_velocities,
            expected_fields,
        ) = self.m._recompute_coordinate_custody(
            np,
            raw_positions,
            raw_velocities,
            gm,
        )
        self.m._validate_coordinate_custody_fields(
            dict(expected_fields), expected_fields
        )
        falsified = dict(expected_fields)
        falsified["position_changed_component_count"] += 1
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "derived metrics",
        ):
            self.m._validate_coordinate_custody_fields(
                falsified, expected_fields
            )
        positions = np.repeat(canonical_positions[None, :, :], 2, axis=0)
        velocities = np.repeat(canonical_velocities[None, :, :], 2, axis=0)
        with mock.patch.dict(self.m.NODE_COUNTS, {64: 2}):
            retained_digest = self.m._recompute_source_tile_sha256(
                np,
                self.protocol,
                64,
                positions,
                velocities,
                gm,
            )
            altered = positions.copy()
            altered[1, 1, 0] = math.nextafter(
                float(altered[1, 1, 0]), math.inf
            )
            altered_digest = self.m._recompute_source_tile_sha256(
                np,
                self.protocol,
                64,
                altered,
                velocities,
                gm,
            )
        self.assertRegex(retained_digest, r"^[0-9a-f]{64}$")
        self.assertNotEqual(retained_digest, altered_digest)

    def _complete_lane_validation_fixture(self):
        m = self.m
        np = self.np
        protocol = self.protocol
        gm, raw_positions, raw_velocities = m._initial_arrays(np, protocol)
        (
            barycentric_positions,
            barycentric_velocities,
            canonical_positions,
            canonical_velocities,
            coordinate_fields,
        ) = m._recompute_coordinate_custody(
            np, raw_positions, raw_velocities, gm
        )
        raw_sha = m._provider_coordinate_state_sha256(
            np,
            b"jx.restricted-wh.source-provider.input.v1",
            raw_positions,
            raw_velocities,
            gm,
        )
        canonical_sha = m._provider_coordinate_state_sha256(
            np,
            b"jx.restricted-wh.source-provider.input.v1",
            canonical_positions,
            canonical_velocities,
            gm,
        )
        barycentric_sha = m._provider_coordinate_state_sha256(
            np,
            b"jx.restricted-wh.source-provider.canonical-barycentric-input.v1",
            barycentric_positions,
            barycentric_velocities,
            gm,
        )
        metric_records = [
            {"name": name, "value_hex": 1.0.hex()}
            for name in m.SOURCE_MAP_METRIC_NAMES
        ]
        binding_digests = {
            item["relative_path"]: item["sha256"]
            for item in protocol["input_bindings"]
        }
        float_fields = (
            "forward_error_ulp_factor",
            "maximum_allowed_input_relative_error_in_eps",
            "maximum_allowed_input_relative_error",
            "position_maximum_component_ulps",
            "velocity_maximum_component_ulps",
            "position_maximum_forward_error_bound_fraction",
            "velocity_maximum_forward_error_bound_fraction",
            "position_maximum_input_relative_error",
            "velocity_maximum_input_relative_error",
            "position_maximum_absolute_delta_AU",
            "velocity_maximum_absolute_delta_AU_per_year",
        )
        count_fields = (
            "position_changed_component_count",
            "velocity_changed_component_count",
            "fixed_point_iteration_count",
            "maximum_fixed_point_iteration_count",
        )

        def coordinate_receipt(tile_sha):
            receipt = {
                "schema": "jx.restricted-wh.source-coordinate-canonicalization.v2",
                "policy": (
                    "PRIMARY_RELATIVE_TO_GM_BARYCENTRIC_TO_PRIMARY_RELATIVE_V2"
                ),
                "forward_error_bound": (
                    "SUM_OF_FULL_ULPS_OF_INTERMEDIATE_PRIMARY_AND_RESULT"
                ),
                **coordinate_fields,
                "raw_relative_input_sha256": raw_sha,
                "canonical_relative_input_sha256": canonical_sha,
                "canonical_barycentric_input_sha256": barycentric_sha,
                "source_tile_sha256": tile_sha,
                "canonicalization_idempotent": True,
                "tile_node_zero_exact": True,
            }
            canonical_record = {
                "schema": receipt["schema"],
                "policy": receipt["policy"],
                "forward_error_bound": receipt["forward_error_bound"],
                **{f"{name}_hex": receipt[name] for name in float_fields},
                **{name: receipt[name] for name in count_fields},
                "raw_relative_input_sha256": raw_sha,
                "canonical_relative_input_sha256": canonical_sha,
                "canonical_barycentric_input_sha256": barycentric_sha,
                "source_tile_sha256": tile_sha,
                "canonicalization_idempotent": True,
                "tile_node_zero_exact": True,
            }
            receipt["content_sha256"] = hashlib.sha256(
                b"jx.restricted-wh.source-coordinate-canonicalization.v2\0"
                + json.dumps(
                    canonical_record,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                ).encode("ascii")
            ).hexdigest()
            return receipt

        positions = {
            steps: np.repeat(canonical_positions[None, :, :], 2, axis=0)
            for steps in (64, 128)
        }
        velocities = {
            steps: np.repeat(canonical_velocities[None, :, :], 2, axis=0)
            for steps in (64, 128)
        }
        total_gm = math.fsum(float(value) for value in gm)
        bound_hex = math.nextafter(2.0 * total_gm, math.inf).hex()

        def lane_receipt(steps):
            tile_sha = m._recompute_source_tile_sha256(
                np,
                protocol,
                steps,
                positions[steps],
                velocities[steps],
                gm,
            )
            coordinate = coordinate_receipt(tile_sha)
            source_map_runtime = {
                "kernel_id": (
                    "jx.wisdom-holman.native-complete-map."
                    "checkpoint-sync.prototype.v3"
                ),
                "g_kernel_id": (
                    "jx.wisdom-holman.native-g-bundle.early-stop.prototype.v2"
                ),
                "exact_fallback_kernel_id": (
                    "jx.wisdom-holman.native-g-bundle.prototype.v1"
                ),
                "scientific_claim_state": "SCREENING_ONLY",
                "maximum_bodies": 32,
                "metric_count": 17,
                "counter_count": 12,
                "compiler_version": "15.2.0",
                "compiler_flags": (
                    "-O3 -g0 -std=c11 -fno-fast-math -ffp-contract=off -lm"
                ),
                "release_wrapper_source_sha256": binding_digests[
                    "src/jxplanetx/_wisdom_holman_cpu_v3_module.c"
                ],
                "release_loop_source_sha256": binding_digests[
                    "src/jxplanetx/_wisdom_holman_loop_v3.c"
                ],
                "release_base_loop_source_sha256": binding_digests[
                    "src/jxplanetx/_wisdom_holman_loop_v2.c"
                ],
                "extension_binary": {
                    "locator": f"PROJECT_ROOT:{m.NATIVE_EXTENSION_RELATIVE_PATH}",
                    "size_bytes": m.NATIVE_EXTENSION_BYTES,
                    "sha256": m.NATIVE_EXTENSION_SHA256,
                },
            }
            runtime_identity = {
                "provider_id": (
                    "jx.restricted-wh.source-provider.ordered-jacobi.v1"
                ),
                "provider_implementation_source_sha256": binding_digests[
                    "src/jxplanetx/restricted_wh_source.py"
                ],
                "python_version": m.PINNED_PYTHON_VERSION,
                "platform": "Linux-test-fixture",
                "numpy_version": m.PINNED_NUMPY_VERSION,
                "source_map_runtime_identity": source_map_runtime,
                "exact_replay_required": True,
                "coordinate_canonicalization": coordinate,
                "coordinate_canonicalization_receipt_sha256": coordinate[
                    "content_sha256"
                ],
                "source_tile_sha256": tile_sha,
            }
            fresh = {
                "tile_content_sha256": tile_sha,
                "input_content_sha256": raw_sha,
                "source_map_input_content_sha256": "a" * 64,
                "source_map_result_content_sha256": "b" * 64,
                "source_map_verification_content_sha256": "c" * 64,
                "source_map_metrics": [
                    [item["name"], item["value_hex"]]
                    for item in metric_records
                ],
                "coordinate_canonicalization": coordinate,
                "runtime_identity": runtime_identity,
            }
            fresh_sha = hashlib.sha256(
                b"jx.tno.1kyr-reduced5-h128-source-preflight."
                b"fresh-workspace-replay.v1\0"
                + m._canonical_json(fresh)
            ).hexdigest()
            schedule = next(
                item["schedule"]
                for item in protocol["execution_specification"]["jx_lanes"]
                if item["steps_per_julian_year"] == steps
            )
            return {
                "lane_id": f"jx_ordered_jacobi_h{steps}",
                "engine": "JX_COMPILED_ORDERED_JACOBI_WISDOM_HOLMAN",
                "steps_per_julian_year": steps,
                "fixed_step_julian_years_hex": (1.0 / steps).hex(),
                "source_step_count": steps,
                "source_node_count": 2,
                "schedule_content_sha256": schedule["content_sha256"],
                "tile_content_sha256": tile_sha,
                "raw_relative_input_sha256": raw_sha,
                "source_map_input_content_sha256": "a" * 64,
                "source_map_result_content_sha256": "b" * 64,
                "source_map_verification_content_sha256": "c" * 64,
                "source_map_metrics_hex": [dict(item) for item in metric_records],
                "coordinate_canonicalization": coordinate,
                "coordinate_canonicalization_content_sha256": coordinate[
                    "content_sha256"
                ],
                "acceleration_bound_certificate": {
                    "formula": (
                        "NEXTUP_OF_2_TIMES_TOTAL_REDUCED_GM_"
                        "DIVIDED_BY_ONE_AU_SQUARED"
                    ),
                    "value_in_source_order_hex": [bound_hex] * m.SOURCE_COUNT,
                },
                "exact_replay_verified": True,
                "fresh_workspace_replay_verified": True,
                "fresh_workspace_receipt_domain": (
                    "jx.tno.1kyr-reduced5-h128-source-preflight."
                    "fresh-workspace-replay.v1"
                ),
                "fresh_workspace_primary_receipt": fresh,
                "fresh_workspace_replay_receipt": dict(fresh),
                "fresh_workspace_primary_receipt_sha256": fresh_sha,
                "fresh_workspace_replay_receipt_sha256": fresh_sha,
                "public_builder_call_count": 2,
                "internal_primary_execution_count": 2,
                "internal_replay_execution_count": 2,
                "total_native_map_execution_count": 4,
                "retained_stream": (
                    "FIRST_FRESH_WORKSPACE_PRIMARY_AFTER_INTERNAL_"
                    "AND_OUTER_BITWISE_REPLAY"
                ),
            }

        h64_lane = lane_receipt(64)
        h128_lane = lane_receipt(128)
        initial_configuration = {
            "integrator": "ias15",
            "epsilon_hex": (1.0e-13).hex(),
            "minimum_dt_julian_years_hex": "0x0.0p+0",
            "adaptive_mode": "PRS23",
            "initial_dt_julian_years_hex": (1.0 / 32.0).hex(),
            "initial_time_julian_years_hex": "0x0.0p+0",
            "G_hex": "0x1.0000000000000p+0",
            "gravity": "basic",
            "collision": "none",
            "boundary": "none",
            "softening_hex": "0x0.0p+0",
            "additional_forces_c_pointer_is_null": True,
            "force_is_velocity_dependent": 0,
            "exit_min_distance_AU_hex": "0x0.0p+0",
            "exit_max_distance_AU_hex": "0x0.0p+0",
            "N": 5,
            "N_active": 5,
            "testparticle_type": 0,
        }
        final_configuration = {
            key: value
            for key, value in initial_configuration.items()
            if key not in {"initial_dt_julian_years_hex", "initial_time_julian_years_hex"}
        }
        final_configuration.update(
            {
                "final_time_julian_years_hex": 1.0.hex(),
                "steps_done": 1,
            }
        )
        ias_lane = {
            "lane_id": "rebound511_ias15_reduced5",
            "engine": "REBOUND_5.1.1_IAS15",
            "model": "AUTONOMOUS_REDUCED5_NO_TIDE_NO_STARS",
            "initial_effective_configuration": initial_configuration,
            "final_effective_configuration": final_configuration,
            "integration_call_count": 1,
            "annual_checkpoint_count": 2,
            "heartbeat_binding": (
                "PINNED_REBOUND_5_1_1_C_FIELD_WITH_LOCAL_CALLBACK_"
                "REFERENCE_DUE_MISSING_HB_SLOT"
            ),
            "heartbeat_callback_count": 1,
            "finite_roster_observation_count": 2,
            "heartbeat_callback_exception": None,
            "force_and_step_implementation": (
                "INDEPENDENT_REBOUND_5_1_1_NO_JX_FORCE_OR_STEP_CODE"
            ),
        }
        result = {
            "lanes": {
                "jx_h64": h64_lane,
                "jx_h128": h128_lane,
                "rebound511_ias15": ias_lane,
            },
            "initial_state_binding": {
                "preregistered_initial_state_content_sha256": protocol[
                    "initial_state"
                ]["content_sha256"],
                "raw_relative_input_sha256": raw_sha,
                "canonical_relative_input_sha256": canonical_sha,
                "canonical_barycentric_input_sha256": barycentric_sha,
                "h64_coordinate_canonicalization_receipt_sha256": h64_lane[
                    "coordinate_canonicalization_content_sha256"
                ],
                "h128_coordinate_canonicalization_receipt_sha256": h128_lane[
                    "coordinate_canonicalization_content_sha256"
                ],
                "h64_h128_node_zero_bitwise_identical": True,
                "ias15_initialized_from_owned_canonical_node_zero_copies": True,
                "h64_h128_ias15_year_zero_barycentric_bitwise_identical": True,
            },
        }
        return result, gm, canonical_positions, canonical_velocities, positions, velocities

    def test_complete_lane_validator_accepts_bound_fixture_and_rejects_tamper(self):
        with mock.patch.dict(self.m.NODE_COUNTS, {64: 2, 128: 2}):
            with mock.patch.object(self.m, "HORIZON_JULIAN_YEARS", 1):
                with mock.patch.object(self.m, "CHECKPOINT_COUNT", 2):
                    (
                        result,
                        gm,
                        canonical_positions,
                        canonical_velocities,
                        positions,
                        velocities,
                    ) = self._complete_lane_validation_fixture()
                    self.m._validate_lane_receipts(
                        self.np,
                        result,
                        self.protocol,
                        gm,
                        canonical_positions,
                        canonical_velocities,
                        positions[64],
                        velocities[64],
                        positions[128],
                        velocities[128],
                    )
                    altered = positions[128].copy()
                    altered[1, 1, 0] = math.nextafter(
                        float(altered[1, 1, 0]), math.inf
                    )
                    with self.assertRaisesRegex(
                        self.m.SourcePreflightExecutionError,
                        "JX execution receipt changed",
                    ):
                        self.m._validate_lane_receipts(
                            self.np,
                            result,
                            self.protocol,
                            gm,
                            canonical_positions,
                            canonical_velocities,
                            positions[64],
                            velocities[64],
                            altered,
                            velocities[128],
                        )

    def test_result_claim_ceiling_is_source_only(self) -> None:
        self.assertIs(self.m.CLAIMS["dynamics_executed"], True)
        self.assertIs(self.m.CLAIMS["source_preflight_executed"], True)
        self.assertIs(self.m.CLAIMS["source_preflight_passed"], True)
        for name in (
            "common_epoch_full_covariance_bridge_qualified",
            "h128_tracer_route_qualified",
            "all12_full129_qualification_passed",
            "external_forcing_qualified",
            "three_gyr_execution_authorized",
            "three_gyr_execution_completed",
            "classification_generated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            self.assertIs(self.m.CLAIMS[name], False, name)
        self.assertEqual(tuple(self.protocol["nonclaims"]), self.m.NONCLAIMS)

    def test_success_roster_and_binary_geometry_match_the_sealed_protocol(self) -> None:
        self.assertEqual(
            tuple(self.protocol["outputs"]["success_roster"]),
            self.m.SUCCESS_ROSTER,
        )
        observed = {
            name: value["bytes"]
            for name, value in self.protocol["outputs"]["binary_files"].items()
        }
        self.assertEqual(observed, self.m.EXPECTED_BINARY_BYTES)
        self.assertEqual(
            sum(self.m.EXPECTED_BINARY_BYTES.values()),
            self.m.FIXED_BINARY_EVIDENCE_BYTES,
        )

    def test_public_cli_has_execute_and_offline_validation_but_no_resume(self) -> None:
        parser = self.m._parser()
        action = next(
            item for item in parser._actions if item.dest == "command"
        )
        self.assertEqual(set(action.choices), {"execute", "validate-result"})

    def test_ast_has_no_module_level_numerical_import_or_gpu_execution_surface(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        imports = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        self.assertFalse(any(name.startswith("numpy") for name in imports))
        self.assertFalse(any(name.startswith("rebound") for name in imports))
        self.assertFalse(any(name.startswith("jxplanetx") for name in imports))
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("--threshold", text)
        self.assertNotIn("resume_parser", text)

    def test_thread_environment_must_be_fixed_before_runtime_import(self) -> None:
        values = {name: "1" for name in self.m.THREAD_ENVIRONMENT}
        with mock.patch.dict(os.environ, values, clear=True):
            self.assertEqual(self.m._require_single_thread_environment(), values)
        values["OMP_NUM_THREADS"] = "2"
        with mock.patch.dict(os.environ, values, clear=True):
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError, "OMP_NUM_THREADS"
            ):
                self.m._require_single_thread_environment()

    def test_runtime_preload_guard_fails_before_lane_zero(self) -> None:
        with mock.patch.dict(sys.modules, {"cupy": types.ModuleType("cupy")}):
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError,
                "imported before custody verification",
            ):
                self.m._reject_preloaded_runtime()

    def test_coordinate_receipt_digest_and_policy_are_recomputed(self) -> None:
        float_fields = (
            "forward_error_ulp_factor",
            "maximum_allowed_input_relative_error_in_eps",
            "maximum_allowed_input_relative_error",
            "position_maximum_component_ulps",
            "velocity_maximum_component_ulps",
            "position_maximum_forward_error_bound_fraction",
            "velocity_maximum_forward_error_bound_fraction",
            "position_maximum_input_relative_error",
            "velocity_maximum_input_relative_error",
            "position_maximum_absolute_delta_AU",
            "velocity_maximum_absolute_delta_AU_per_year",
        )
        receipt = {
            "schema": "jx.restricted-wh.source-coordinate-canonicalization.v2",
            "policy": "PRIMARY_RELATIVE_TO_GM_BARYCENTRIC_TO_PRIMARY_RELATIVE_V2",
            "forward_error_bound": (
                "SUM_OF_FULL_ULPS_OF_INTERMEDIATE_PRIMARY_AND_RESULT"
            ),
            **{name: 0.0.hex() for name in float_fields},
            "position_changed_component_count": 0,
            "velocity_changed_component_count": 0,
            "fixed_point_iteration_count": 1,
            "maximum_fixed_point_iteration_count": 4,
            "raw_relative_input_sha256": "1" * 64,
            "canonical_relative_input_sha256": "2" * 64,
            "canonical_barycentric_input_sha256": "3" * 64,
            "source_tile_sha256": "4" * 64,
            "canonicalization_idempotent": True,
            "tile_node_zero_exact": True,
        }
        receipt["forward_error_ulp_factor"] = 1.0.hex()
        receipt["maximum_allowed_input_relative_error_in_eps"] = 512.0.hex()
        receipt["maximum_allowed_input_relative_error"] = (
            512.0 * sys.float_info.epsilon
        ).hex()
        canonical = {
            "schema": receipt["schema"],
            "policy": receipt["policy"],
            "forward_error_bound": receipt["forward_error_bound"],
            **{f"{name}_hex": receipt[name] for name in float_fields},
            "position_changed_component_count": 0,
            "velocity_changed_component_count": 0,
            "fixed_point_iteration_count": 1,
            "maximum_fixed_point_iteration_count": 4,
            "raw_relative_input_sha256": "1" * 64,
            "canonical_relative_input_sha256": "2" * 64,
            "canonical_barycentric_input_sha256": "3" * 64,
            "source_tile_sha256": "4" * 64,
            "canonicalization_idempotent": True,
            "tile_node_zero_exact": True,
        }
        payload = json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        receipt["content_sha256"] = hashlib.sha256(
            b"jx.restricted-wh.source-coordinate-canonicalization.v2\0" + payload
        ).hexdigest()
        self.m._validate_coordinate_receipt(receipt)
        tampered = dict(receipt)
        tampered["position_changed_component_count"] = 1
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "content digest",
        ):
            self.m._validate_coordinate_receipt(tampered)

    def test_existing_output_refuses_before_authority_or_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "already-exists"
            target.mkdir()
            with mock.patch.object(self.m, "_load_sealed_authority") as authority:
                with mock.patch.object(self.m, "_runtime_preflight") as runtime:
                    with self.assertRaisesRegex(
                        self.m.SourcePreflightExecutionError,
                        "already exists",
                    ):
                        self.m.execute(target)
            authority.assert_not_called()
            runtime.assert_not_called()

    def test_create_only_member_and_atomic_root_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            member = parent / "evidence.bin"
            self.m._write_create_only(member, b"one")
            self.assertEqual(member.read_bytes(), b"one")
            with self.assertRaises(self.m.SourcePreflightExecutionError):
                self.m._write_create_only(member, b"two")
            source = parent / "source"
            destination = parent / "destination"
            source.mkdir()
            destination.mkdir()
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError,
                "already exists",
            ):
                self.m._rename_directory_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertTrue(destination.is_dir())
            oversized = parent / "oversized.json"
            oversized.write_bytes(b"x" * (self.m.MAXIMUM_JSON_RECEIPT_BYTES + 1))
            oversized.chmod(0o600)
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError,
                "byte length changed",
            ):
                self.m._secure_result_bytes(parent, oversized.name)

    def test_post_rename_durability_failure_rolls_publication_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            stage = parent / ".result.partial"
            root = parent / "result"
            stage.mkdir(mode=0o700)
            (stage / "evidence").write_bytes(b"sealed")
            with mock.patch.object(
                self.m.os,
                "fsync",
                side_effect=OSError("injected directory fsync failure"),
            ):
                with self.assertRaisesRegex(
                    self.m.SourcePreflightExecutionError,
                    "rolled back",
                ):
                    self.m._publish_staged_directory(stage, root)
            self.assertFalse(root.exists())
            self.assertTrue(stage.is_dir())
            self.assertEqual((stage / "evidence").read_bytes(), b"sealed")


class ExactNumericalPredicateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = MODULE
        cls.np = NP

    def test_exact_vector_threshold_uses_squared_rational_arithmetic(self) -> None:
        richardson_boundary = 3.0 / 500.0
        self.assertTrue(
            self.m._exact_vector_threshold(
                (math.nextafter(richardson_boundary, -math.inf), 0.0, 0.0),
                numerator=1,
                denominator=500,
                divisor=3,
            )
        )
        self.assertFalse(
            self.m._exact_vector_threshold(
                (math.nextafter(richardson_boundary, math.inf), 0.0, 0.0),
                numerator=1,
                denominator=500,
                divisor=3,
            )
        )
        direct_boundary = 1.0 / 500.0
        self.assertTrue(
            self.m._exact_vector_threshold(
                (math.nextafter(direct_boundary, -math.inf), 0.0, 0.0),
                numerator=1,
                denominator=500,
            )
        )
        self.assertFalse(
            self.m._exact_vector_threshold(
                (math.nextafter(direct_boundary, math.inf), 0.0, 0.0),
                numerator=1,
                denominator=500,
            )
        )

    def test_vector_decisions_subtract_original_binary64_operands_exactly(self) -> None:
        tiny = float.fromhex("0x1.0000000000000p-54")
        rounded_difference = 1.0 - (-tiny)
        self.assertEqual(rounded_difference, 1.0)
        self.assertTrue(
            self.m._exact_vector_threshold(
                (rounded_difference, 0.0, 0.0), numerator=1, denominator=1
            )
        )
        self.assertFalse(
            self.m._exact_vector_threshold_between(
                (1.0, 0.0, 0.0),
                (-tiny, 0.0, 0.0),
                numerator=1,
                denominator=1,
            )
        )

    def test_exact_vector_contraction_has_64_ulp_floor(self) -> None:
        reference = (1.0, 2.0, 4.0)
        floor = 64.0 * math.ulp(1.0)
        self.assertTrue(
            self.m._exact_vector_contraction(
                (0.0, 0.0, 0.0), (floor, 0.0, 0.0), reference
            )
        )
        fine = float.fromhex("0x1.0000000000000p-30")
        self.assertTrue(
            self.m._exact_vector_contraction(
                (3.0 * fine, 0.0, 0.0), (fine, 0.0, 0.0), reference
            )
        )
        self.assertFalse(
            self.m._exact_vector_contraction(
                (2.0 * fine, 0.0, 0.0), (fine, 0.0, 0.0), reference
            )
        )

    def test_invariant_drift_decisions_use_exact_binary64_ratios(self) -> None:
        threshold = self.m.Fraction(1, 100_000_000)
        below = math.nextafter(1.0 + 1.0e-8, -math.inf)
        above = math.nextafter(1.0 + 1.0e-8, math.inf)
        self.assertLessEqual(
            self.m._exact_energy_relative_drift(below, 1.0), threshold
        )
        self.assertGreater(
            self.m._exact_energy_relative_drift(above, 1.0), threshold
        )

        angular_threshold_squared = self.m.Fraction(1, 10_000_000_000) ** 2
        angular_below = math.nextafter(1.0e-10, -math.inf)
        angular_above = math.nextafter(1.0e-10, math.inf)
        self.assertLessEqual(
            self.m._exact_angular_relative_drift_squared(
                (1.0, angular_below, 0.0), (1.0, 0.0, 0.0)
            ),
            angular_threshold_squared,
        )
        self.assertGreater(
            self.m._exact_angular_relative_drift_squared(
                (1.0, angular_above, 0.0), (1.0, 0.0, 0.0)
            ),
            angular_threshold_squared,
        )

    def _state(self):
        np = self.np
        gm = np.array([1.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [-4.0, 0.0, 0.0],
                [0.0, -5.0, 1.0],
            ],
            dtype=np.float64,
        )
        velocities = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [-0.4, 0.0, 0.0],
                [0.0, -0.3, 0.0],
                [0.2, 0.0, 0.1],
            ],
            dtype=np.float64,
        )
        return gm, positions, velocities

    def test_barycentric_reconstruction_uses_common_gm_center(self) -> None:
        np = self.np
        gm, positions, velocities = self._state()
        bary_position, bary_velocity = self.m._barycentric_node(
            np, positions, velocities, gm
        )
        for component in range(3):
            self.assertAlmostEqual(
                math.fsum(
                    float(gm[index]) * float(bary_position[index, component])
                    for index in range(5)
                ),
                0.0,
                places=15,
            )
            self.assertAlmostEqual(
                math.fsum(
                    float(gm[index]) * float(bary_velocity[index, component])
                    for index in range(5)
                ),
                0.0,
                places=15,
            )

    def test_invariants_are_finite_and_translation_independent(self) -> None:
        np = self.np
        gm, positions, velocities = self._state()
        baseline = self.m._invariant_row(np, positions, velocities, gm)
        translated_positions = positions + np.array([10.0, -20.0, 5.0])
        translated_velocities = velocities + np.array([1.0, -2.0, 0.5])
        translated = self.m._invariant_row(
            np, translated_positions, translated_velocities, gm
        )
        self.assertTrue(all(math.isfinite(value) for value in baseline))
        for left, right in zip(baseline, translated, strict=True):
            self.assertAlmostEqual(left, right, places=12)

    def _constant_streams(self):
        np = self.np
        gm, positions, velocities = self._state()
        h64_position = np.repeat(positions[None, :, :], 3, axis=0)
        h64_velocity = np.repeat(velocities[None, :, :], 3, axis=0)
        h128_position = np.repeat(positions[None, :, :], 5, axis=0)
        h128_velocity = np.repeat(velocities[None, :, :], 5, axis=0)
        ias = np.empty((2, 5, 6), dtype=np.float64)
        ias[:, :, :3] = np.repeat(positions[None, :, :], 2, axis=0)
        ias[:, :, 3:] = np.repeat(velocities[None, :, :], 2, axis=0)
        h64_invariants, h64_summary = self.m._invariant_stream(
            np, h64_position, h64_velocity, gm
        )
        h128_invariants, h128_summary = self.m._invariant_stream(
            np, h128_position, h128_velocity, gm
        )
        return (
            gm,
            h64_position,
            h64_velocity,
            h128_position,
            h128_velocity,
            h64_invariants,
            h128_invariants,
            ias,
            h64_summary,
            h128_summary,
        )

    def test_tiny_identical_fixture_passes_every_numeric_gate(self) -> None:
        values = self._constant_streams()
        gates = self.m._evaluate_numeric_gates(
            self.np,
            gm=values[0],
            h64_positions=values[1],
            h64_velocities=values[2],
            h128_positions=values[3],
            h128_velocities=values[4],
            h64_invariants=values[5],
            h128_invariants=values[6],
            ias15_annual=values[7],
            h64_stride=2,
            h128_stride=4,
        )
        self.assertIs(gates["all_applicable_source_predicates"]["pass"], True)
        self.assertTrue(all(record["pass"] for record in gates.values()))

    def test_tiny_fixture_fails_closed_on_large_ias15_disagreement(self) -> None:
        values = list(self._constant_streams())
        values[7] = values[7].copy()
        values[7][1, 1, 0] += 1.0
        gates = self.m._evaluate_numeric_gates(
            self.np,
            gm=values[0],
            h64_positions=values[1],
            h64_velocities=values[2],
            h128_positions=values[3],
            h128_velocities=values[4],
            h64_invariants=values[5],
            h128_invariants=values[6],
            ias15_annual=values[7],
            h64_stride=2,
            h128_stride=4,
        )
        self.assertIs(gates["h128_vs_ias15_position"]["pass"], False)
        self.assertIs(gates["position_vector_contraction"]["pass"], False)
        self.assertIs(gates["all_applicable_source_predicates"]["pass"], False)

    def test_tiny_fixture_detects_year_zero_coordinate_mismatch(self) -> None:
        values = list(self._constant_streams())
        values[7] = values[7].copy()
        values[7][0, 2, 1] += 1.0e-12
        gates = self.m._evaluate_numeric_gates(
            self.np,
            gm=values[0],
            h64_positions=values[1],
            h64_velocities=values[2],
            h128_positions=values[3],
            h128_velocities=values[4],
            h64_invariants=values[5],
            h128_invariants=values[6],
            ias15_annual=values[7],
            h64_stride=2,
            h128_stride=4,
        )
        self.assertIs(gates["canonical_year_zero_identity"]["pass"], False)
        self.assertIs(gates["all_applicable_source_predicates"]["pass"], False)


class MockedExecutionSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = MODULE
        cls.np = NP
        cls.protocol = PROTOCOL_VALUE

    @dataclass(frozen=True)
    class FakeProvenance:
        source_id: str
        citation: str
        version: str
        sha256: str

    class FakeSpec:
        def __init__(self, **values):
            self.values = values

    class FakeCanonicalization:
        raw_relative_input_sha256 = "1" * 64
        canonical_relative_input_sha256 = "2" * 64
        canonical_barycentric_input_sha256 = "3" * 64
        content_sha256 = "4" * 64
        canonicalization_idempotent = True
        tile_node_zero_exact = True

        def __init__(self, tile_sha):
            self.source_tile_sha256 = tile_sha

        def as_record(self):
            return {
                "raw_relative_input_sha256": self.raw_relative_input_sha256,
                "canonical_relative_input_sha256": self.canonical_relative_input_sha256,
                "canonical_barycentric_input_sha256": self.canonical_barycentric_input_sha256,
                "source_tile_sha256": self.source_tile_sha256,
                "canonicalization_idempotent": True,
                "tile_node_zero_exact": True,
                "content_sha256": self.content_sha256,
                "roundoff_hex": 0.0,
            }

    def _fake_context(self, *, change_second=False):
        np = self.np
        calls = []

        def builder(**arguments):
            calls.append(
                (
                    arguments["initial_source_relative_positions_AU"],
                    arguments["initial_source_relative_velocities_AU_per_year"],
                    arguments["source_gm_AU3_per_year2"],
                    arguments["source_encounter_radii_AU"],
                    arguments[
                        "certified_relative_acceleration_bounds_AU_per_year2"
                    ],
                )
            )
            positions = np.repeat(
                arguments["initial_source_relative_positions_AU"][None, :, :],
                2,
                axis=0,
            ).astype(np.float64, copy=True)
            velocities = np.repeat(
                arguments["initial_source_relative_velocities_AU_per_year"][None, :, :],
                2,
                axis=0,
            ).astype(np.float64, copy=True)
            if change_second and len(calls) == 2:
                positions[-1, 1, 0] += 1.0
            tile_sha = "5" * 64 if not (change_second and len(calls) == 2) else "6" * 64
            tile = types.SimpleNamespace(
                source_relative_positions_AU=positions,
                source_relative_velocities_AU_per_year=velocities,
                source_gm_AU3_per_year2=arguments["source_gm_AU3_per_year2"].copy(),
                content_sha256=tile_sha,
            )
            canonicalization = self.FakeCanonicalization(tile_sha)
            return types.SimpleNamespace(
                tile=tile,
                input_content_sha256="1" * 64,
                source_map_input_content_sha256="7" * 64,
                source_map_result_content_sha256="8" * 64,
                source_map_verification_content_sha256="9" * 64,
                source_map_metrics=tuple(
                    (name, 1.0) for name in self.m.SOURCE_MAP_METRIC_NAMES
                ),
                source_map_primary_seconds=1.0,
                source_map_replay_seconds=1.0,
                coordinate_canonicalization=canonicalization,
                runtime_identity={
                    "exact_replay_required": True,
                    "source_tile_sha256": tile_sha,
                    "coordinate_canonicalization_receipt_sha256": "4" * 64,
                    "source_map_runtime_identity": {
                        "extension_binary": {
                            "path": str(
                                self.m.PROJECT_ROOT
                                / self.m.NATIVE_EXTENSION_RELATIVE_PATH
                            ),
                            "size_bytes": self.m.NATIVE_EXTENSION_BYTES,
                            "sha256": self.m.NATIVE_EXTENSION_SHA256,
                        }
                    },
                },
                __post_init__=lambda: None,
            )

        source_module = types.SimpleNamespace(
            WisdomHolmanSourceTileSpec=self.FakeSpec,
            build_wisdom_holman_source_tile=builder,
        )
        context = self.m.RuntimeContext(
            np=np,
            rebound=None,
            source_module=source_module,
            provenance_type=self.FakeProvenance,
            runtime_record={},
        )
        return context, calls

    def test_jx_lane_requires_two_fresh_public_builds_and_four_total_runs(self) -> None:
        context, calls = self._fake_context()
        with mock.patch.dict(self.m.NODE_COUNTS, {64: 2}):
            build, receipt = self.m._run_jx_lane(context, self.protocol, 64)
        self.assertEqual(len(calls), 2)
        for index in range(5):
            self.assertIsNot(calls[0][index], calls[1][index])
        self.assertEqual(build.tile.source_relative_positions_AU.shape, (2, 5, 3))
        self.assertIs(receipt["fresh_workspace_replay_verified"], True)
        self.assertEqual(receipt["public_builder_call_count"], 2)
        self.assertEqual(receipt["total_native_map_execution_count"], 4)
        extension = receipt["fresh_workspace_primary_receipt"][
            "runtime_identity"
        ]["source_map_runtime_identity"]["extension_binary"]
        self.assertEqual(
            extension["locator"],
            f"PROJECT_ROOT:{self.m.NATIVE_EXTENSION_RELATIVE_PATH}",
        )
        self.assertNotIn(str(self.m.PROJECT_ROOT), json.dumps(receipt))

    def test_jx_lane_fails_if_fresh_workspace_replay_differs(self) -> None:
        context, calls = self._fake_context(change_second=True)
        with mock.patch.dict(self.m.NODE_COUNTS, {64: 2}):
            with self.assertRaisesRegex(
                self.m.SourcePreflightExecutionError,
                "fresh-workspace replay differs bitwise",
            ):
                self.m._run_jx_lane(context, self.protocol, 64)
        self.assertEqual(len(calls), 2)

    class FakeIntegrator:
        def __init__(self):
            self.epsilon = 0.0
            self.min_dt = 0.0
            self.adaptive_mode = ""

        def __str__(self):
            return "ias15"

    class FakeParticle:
        def __init__(self, **values):
            self.x = values["x"]
            self.y = values["y"]
            self.z = values["z"]
            self.vx = values["vx"]
            self.vy = values["vy"]
            self.vz = values["vz"]

    class FakeSimulation:
        def __init__(self):
            self.G = 1.0
            self.gravity = "basic"
            self.collision = "none"
            self.boundary = "none"
            self.softening = 0.0
            self._additional_forces = None
            self.force_is_velocity_dependent = 0
            self.exit_min_distance = 0.0
            self.exit_max_distance = 0.0
            self.dt = 0.0
            self.t = 0.0
            self.steps_done = 0
            self.N_active = 0
            self.testparticle_type = 0
            self.particles = []
            self._integrator = MockedExecutionSurfaceTests.FakeIntegrator()
            self._heartbeat = None

        @property
        def N(self):
            return len(self.particles)

        @property
        def additional_forces(self):
            raise AttributeError(
                "You can only set C function pointers from python."
            )

        @property
        def integrator(self):
            return self._integrator

        @integrator.setter
        def integrator(self, value):
            if value != "ias15":
                raise AssertionError("unexpected integrator")
            self._integrator = MockedExecutionSurfaceTests.FakeIntegrator()

        def add(self, **values):
            self.particles.append(MockedExecutionSurfaceTests.FakeParticle(**values))

        def integrate(self, target, exact_finish_time=0):
            if exact_finish_time != 1:
                raise AssertionError("exact finish time is required")
            self.t = float(target)
            self.steps_done += 1
            self._heartbeat(self)

        def synchronize(self):
            return None

    def test_ias15_surface_uses_canonical_owned_state_and_heartbeat(self) -> None:
        np = self.np
        simulation_holder = []

        def simulation_factory():
            value = self.FakeSimulation()
            simulation_holder.append(value)
            return value

        rebound = types.SimpleNamespace(
            Simulation=simulation_factory,
            simulation=types.SimpleNamespace(AFF=lambda callback: callback),
        )
        context = self.m.RuntimeContext(
            np=np,
            rebound=rebound,
            source_module=None,
            provenance_type=self.FakeProvenance,
            runtime_record={},
        )
        gm, positions, velocities = self.m._initial_arrays(np, self.protocol)
        with mock.patch.object(self.m, "CHECKPOINT_COUNT", 3):
            with mock.patch.object(self.m, "HORIZON_JULIAN_YEARS", 2):
                annual, receipt = self.m._run_ias15(
                    context,
                    positions.copy(),
                    velocities.copy(),
                    gm.copy(),
                )
        self.assertEqual(annual.shape, (3, 5, 6))
        self.assertTrue(
            self.m._array_bitwise_equal(np, annual[0, :, :3], positions)
        )
        self.assertEqual(receipt["integration_call_count"], 2)
        self.assertEqual(receipt["heartbeat_callback_count"], 2)
        self.assertEqual(simulation_holder[0].steps_done, 2)

    def test_result_builder_refuses_any_failed_gate(self) -> None:
        gates = {
            "one": {"pass": False},
            "all_applicable_source_predicates": {"pass": False},
        }
        with self.assertRaisesRegex(
            self.m.SourcePreflightExecutionError,
            "did not all pass",
        ):
            self.m._build_result(
                authority={},
                runtime={
                    "successor_implementation": {},
                    "content_sha256": "0" * 64,
                },
                artifacts={},
                h64_receipt={},
                h128_receipt={},
                ias15_receipt={},
                h64_invariant_summary={},
                h128_invariant_summary={},
                gates=gates,
                initial_state_binding={},
            )


if __name__ == "__main__":
    unittest.main()
