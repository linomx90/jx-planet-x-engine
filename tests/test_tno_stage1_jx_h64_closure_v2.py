from __future__ import annotations

import ast
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_stage1_jx_h64_closure_v2.py"


def load_module() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    name = "_test_jx_tno_stage1_jx_h64_closure_v2"
    module = types.ModuleType(name)
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(payload, str(SOURCE), "exec", dont_inherit=True), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    return module


class H64ClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    def test_geometry_is_exactly_the_bounded_24_lane_successor(self) -> None:
        m = self.module
        self.assertEqual(m.STEPS_PER_YEAR, 64)
        self.assertEqual(m.MIDPOINT_RECORD_COUNT, 24)
        self.assertEqual(m.SOURCE_NODE_COUNT, 64_001)
        self.assertEqual(m.EXPECTED_SOURCE_BYTES, 368_645_800)
        self.assertEqual(m.EXPECTED_ARRAY_BYTES, 16_144_128)
        self.assertEqual(m.EXPECTED_MIDPOINT_BYTES, 10_368)
        self.assertEqual(m.EXPECTED_RESTART_STREAM_BYTES, 865_728)
        self.assertEqual(
            m.EXPECTED_SOURCE_BYTES + m.EXPECTED_ARRAY_BYTES
            + m.EXPECTED_MIDPOINT_BYTES + m.EXPECTED_RESTART_STREAM_BYTES,
            385_666_024,
        )
        self.assertEqual(m._selected_restart_cases(), m.RESTART_CASES)

    def test_protocol_discloses_retrospective_policy_and_claim_ceiling(self) -> None:
        m = self.module
        protocol = m.build_protocol()
        self.assertEqual(
            protocol["content_sha256"],
            m._content_sha256(m.PROTOCOL_DOMAIN, protocol),
        )
        policy = protocol["gate_policy"]
        self.assertFalse(policy["thresholds_changed"])
        self.assertIn("ROSTER_CHANGED", policy["post_outcome_policy_disclosure"])
        self.assertEqual(policy["fine_absolute_energy_ceiling_rational"], "1/100000000")
        self.assertEqual(policy["source_velocity_ceiling_AU_per_julian_year_rational"], "1/50000")
        self.assertEqual(protocol["execution_matrix"]["new_fixed_source_steps"], 1_536_000)
        self.assertEqual(protocol["execution_matrix"]["new_tracer_step_updates"], 13_824_000)
        self.assertFalse(protocol["publication_policy"]["multi_file_atomicity_claimed"])
        self.assertTrue(
            protocol["preregistration"]["execution_allowed_only_after_commit_and_tag_are_pushed"]
        )
        self.assertEqual(protocol["preregistration"]["tag"], m.PREREGISTRATION_TAG)
        self.assertTrue(protocol["claims"]["retrospective_outcome_seen"])
        self.assertFalse(protocol["claims"]["failed_v1_result_published"])
        self.assertTrue(protocol["claims"]["v2_implementation_repair_only"])
        self.assertTrue(protocol["claims"]["threshold_application_roster_changed_after_outcome"])
        for key in (
            "h64_refinement_closure_established", "dynamical_classification_generated",
            "resonance_classification_generated", "population_inference",
            "planet_nine_evidence", "science_result",
        ):
            self.assertIs(protocol["claims"][key], False)

    def test_failed_v1_custody_and_exact_repair_are_disclosed(self) -> None:
        m = self.module
        protocol = m.build_protocol()
        failed = protocol["failed_attempt_v1"]
        repair = protocol["implementation_repair"]
        self.assertEqual(failed["git_commit"], m.FAILED_V1_GIT_COMMIT)
        self.assertEqual(failed["source_sha256"], m.FAILED_V1_SOURCE_SHA256)
        self.assertEqual(failed["test_sha256"], m.FAILED_V1_TEST_SHA256)
        self.assertEqual(failed["retained_root_roster"], ["PROTOCOL.json"])
        self.assertFalse(failed["result_published"])
        self.assertEqual(failed["exception_type"], "NameError")
        self.assertFalse(repair["scientific_contract_changed"])
        self.assertFalse(repair["numerical_contract_changed"])
        self.assertFalse(repair["gate_formulas_changed"])
        self.assertFalse(repair["thresholds_changed"])
        self.assertEqual(
            m._validate_failed_v1_custody()["content_sha256"],
            m.FAILED_V1_PROTOCOL_CONTENT_SHA256,
        )

    def test_restart_receipts_are_explicitly_returned_and_destructured(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        runtime = functions["_runtime_and_lanes"]
        returns = [node for node in ast.walk(runtime) if isinstance(node, ast.Return)]
        returned_names = [
            item.id
            for node in returns
            if isinstance(node.value, ast.Tuple)
            for item in node.value.elts
            if isinstance(item, ast.Name)
        ]
        self.assertEqual(returned_names.count("restart_receipts"), 1)
        self.assertLess(
            returned_names.index("restart_records"),
            returned_names.index("restart_receipts"),
        )
        self.assertLess(
            returned_names.index("restart_receipts"),
            returned_names.index("source_payload"),
        )

        worker = functions["_execute_worker"]
        destructured = []
        for node in ast.walk(worker):
            if (
                isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == "_runtime_and_lanes"
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Tuple)
            ):
                destructured = [
                    item.id for item in node.targets[0].elts
                    if isinstance(item, ast.Name)
                ]
        self.assertEqual(destructured.count("restart_receipts"), 1)
        self.assertLess(
            destructured.index("restart_records"),
            destructured.index("restart_receipts"),
        )
        self.assertLess(
            destructured.index("restart_receipts"),
            destructured.index("source_payload"),
        )

    def test_execution_worker_carries_restart_receipts_into_metadata(self) -> None:
        m = self.module
        receipts = [{"case_order": 0}, {"case_order": 1}]
        captured = {}

        class FakeDelegate:
            @staticmethod
            def _fresh_child_interpreter_receipt():
                return {"fresh": True}

            @staticmethod
            def _reject_preloaded_runtime_modules():
                return {"registry": True}

            @staticmethod
            def _load_json(_path, _label):
                return b"{}", {"probe": True}

            @staticmethod
            def _validate_import_probe_receipt(_receipt, *, expected_path):
                self.assertEqual(expected_path, Path("probe.json"))

            @staticmethod
            def validate_execution_prerequisites(_root):
                return {"custody": True}

            @staticmethod
            def materialize_cartesian_payload(_lanes):
                return b"\0" * m.EXPECTED_ARRAY_BYTES, [{"lane": "compact"}], [{"lane": "gate"}]

        originals = {
            "_delegate": m._delegate,
            "validate_prepared": m.validate_prepared,
            "_validate_v4_artifacts": m._validate_v4_artifacts,
            "_runtime_and_lanes": m._runtime_and_lanes,
            "_combined_gate_result": m._combined_gate_result,
            "_publish_result_artifacts": m._publish_result_artifacts,
        }
        try:
            m._delegate = lambda: FakeDelegate()
            m.validate_prepared = lambda _root: m.build_protocol()
            m._validate_v4_artifacts = lambda **_kwargs: ({}, {})
            m._runtime_and_lanes = lambda *_args: (
                [{"lane": "raw"}], b"midpoint", [{"midpoint_order": 0}],
                b"restart", [{"stream_order": 0}], receipts,
                b"source", [{"source_order": 0}], {"runtime": True},
            )
            m._combined_gate_result = lambda _lanes: {
                "all_predeclared_gates_passed": False,
            }
            m._publish_result_artifacts = lambda _root, artifacts: captured.update(
                artifacts=dict(artifacts)
            )
            result = m._execute_worker(
                Path("unused-root"), Path("unused-de441"),
                Path("unused-rebound"), Path("probe.json"),
            )
        finally:
            for name, value in originals.items():
                setattr(m, name, value)

        self.assertEqual(
            result["jx_midpoint_checkpoints"]["fresh_process_restart_cases"],
            receipts,
        )
        self.assertEqual(
            json.loads(captured["artifacts"]["RESULT.json"])[
                "jx_midpoint_checkpoints"
            ]["fresh_process_restart_cases"],
            receipts,
        )

    def test_protocol_binds_current_source_test_and_exact_v4_artifacts(self) -> None:
        m = self.module
        protocol = m.build_protocol()
        self.assertEqual(
            protocol["implementation"]["sha256"], hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        )
        self.assertEqual(
            protocol["test"]["sha256"], hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        )
        self.assertEqual(len(protocol["predecessor_v4"]["bindings"]), 7)
        self.assertEqual(
            protocol["predecessor_v4"]["result_content_sha256"],
            m.V4_RESULT_CONTENT_SHA256,
        )
        m._validate_v4_artifacts(full_hashes=False)

    def test_delegate_is_restricted_to_h64(self) -> None:
        m = self.module
        d = m._delegate()
        self.assertEqual(d.JX_STEPS_PER_YEAR, (64,))
        self.assertEqual(d.JX_STEP_YEARS, (1.0 / 64.0,))
        self.assertEqual(d.JX_FRESH_PROCESS_RESTART_CASES, m.RESTART_CASES)
        self.assertEqual(tuple(row["lane_id"] for row in d.LANE_SPECS), (m.LANE_ID,))
        self.assertIs(d.build_protocol, m.build_protocol)

    def test_source_ast_has_no_ias_or_rebound_lane_execution(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        self.assertNotIn("_run_ias15_lane", calls)
        self.assertNotIn("Simulation", calls)
        self.assertIn("_run_jx_lane", calls)

    def test_v4_gate_lane_reuse_roster_is_exact(self) -> None:
        lanes, payload = self.module._load_v4_gate_lanes()
        self.assertEqual(len(payload), 71_495_424)
        self.assertEqual(len(lanes), 72)
        self.assertEqual(
            {key[2] for key in lanes},
            {
                "jx_reduced5_h_1_32",
                "rebound511_ias15_reduced5",
                "rebound511_ias15_resolved11",
            },
        )

    def test_gate_mapping_uses_h64_only_for_absolute_energy(self) -> None:
        m = self.module

        class FakeBase:
            def __init__(self) -> None:
                self.combined = None

            @staticmethod
            def _fraction_from_record(value, _label):
                return Fraction(value["numerator"], value["denominator"])

            @staticmethod
            def _exact_fraction_gate(values, limit):
                maximum = max(values)
                return {
                    "passed": maximum <= limit,
                    "maximum": {"numerator": maximum.numerator, "denominator": maximum.denominator},
                    "limit": {"numerator": limit.numerator, "denominator": limit.denominator},
                }

            def evaluate_qualification_gates(self, combined):
                self.combined = combined
                return {
                    "gates": {
                        "jx_reduced_max_relative_energy_drift": {"passed": False},
                        "unchanged_gate": {"passed": True},
                    },
                    "all_predeclared_gates_passed": False,
                }

        fake = FakeBase()
        old = {}
        new = []
        for ephemeris in ("DE440S", "DE441"):
            for batch in range(m.OBJECT_COUNT):
                object_id = m.OBJECT_IDS[batch]
                for lane_id in (
                    "jx_reduced5_h_1_32",
                    "rebound511_ias15_reduced5",
                    "rebound511_ias15_resolved11",
                ):
                    old[(ephemeris, batch, lane_id)] = {
                        "ephemeris": ephemeris,
                        "batch_order": batch,
                        "object_id": object_id,
                        "lane": {
                            "lane_id": lane_id,
                            "summary": {
                                "maximum_relative_massive_energy_drift_exact": {
                                    "numerator": 2, "denominator": 100_000_000,
                                }
                            },
                        },
                    }
                new.append({
                    "ephemeris": ephemeris,
                    "batch_order": batch,
                    "object_id": object_id,
                    "lane": {
                        "lane_id": m.LANE_ID,
                        "summary": {
                            "maximum_relative_massive_energy_drift_exact": {
                                "numerator": 1, "denominator": 200_000_000,
                            }
                        },
                    },
                })
        original_base = m._BASE
        original_loader = m._load_v4_gate_lanes
        try:
            m._BASE = fake
            m._load_v4_gate_lanes = lambda: (old, b"")
            result = m._combined_gate_result(new)
        finally:
            m._BASE = original_base
            m._load_v4_gate_lanes = original_loader
        self.assertTrue(result["gates"]["jx_reduced_max_relative_energy_drift"]["passed"])
        self.assertFalse(result["retained_h32_absolute_energy_diagnostic"]["passed"])
        self.assertFalse(result["retained_h32_absolute_energy_diagnostic"]["used_for_acceptance"])
        self.assertTrue(result["all_predeclared_gates_passed"])
        self.assertEqual(len(fake.combined), 96)
        self.assertEqual(
            [row["lane"]["lane_id"] for row in fake.combined[:4]],
            [
                "jx_reduced5_h_1_16", "jx_reduced5_h_1_32",
                "rebound511_ias15_reduced5", "rebound511_ias15_resolved11",
            ],
        )

    def test_selected_restart_ledger_receives_fresh_registry(self) -> None:
        m = self.module
        selection = m.RESTART_CASES[0]
        batch = types.SimpleNamespace(batch_order=m.OBJECT_IDS.index(selection[2]), object_id=selection[2])
        registry = {"fresh_registry_observed": True, "nonce": "bounded-test"}
        ledger = {"steps_per_year": 64}
        lane = {
            "_jx_midpoint_payload": b"\0" * m.MIDPOINT_RECORD_BYTES,
            "_jx_midpoint_revalidated_before_continuation": True,
            "_jx_selected_suffix_payload": b"suffix",
            "_jx_selected_suffix_ledger": ledger,
            "_jx_source_evidence_payload": b"source",
            "_jx_source_contract": {"test": True},
            "_jx_source_gm_payload": b"\0" * m.SOURCE_GM_BYTES,
            "full_source_tile_sha256": "a" * 64,
            "annual_source_slice_sha256": ["b" * 64],
        }
        selected = {}
        m._extract_jx_lane(
            types.SimpleNamespace(), lane, selection[0], batch, io.BytesIO(), None,
            [], bytearray(), [], selected, registry,
        )
        self.assertEqual(selected[selection][1]["fresh_process_registry"], registry)
        self.assertNotIn("fresh_process_registry", ledger)

    def test_midpoint_metadata_is_exact_and_tamper_evident(self) -> None:
        m = self.module
        payload = b"\0" * m.EXPECTED_MIDPOINT_BYTES
        records = []
        offset = 0
        for order, (ephemeris, batch_order) in enumerate(
            (e, b) for e in ("DE440S", "DE441") for b in range(m.OBJECT_COUNT)
        ):
            record_payload = payload[offset : offset + m.MIDPOINT_RECORD_BYTES]
            record = {
                "midpoint_order": order, "ephemeris": ephemeris,
                "batch_order": batch_order, "object_id": m.OBJECT_IDS[batch_order],
                "lane_id": m.LANE_ID, "steps_per_year": 64,
                "midpoint_year": 500, "global_source_step": 32_000,
                "byte_offset": offset, "byte_length": m.MIDPOINT_RECORD_BYTES,
                "record_sha256": m._sha256(record_payload),
                "revalidated_before_continuation": True,
            }
            record["record_content_sha256"] = m._sha256(
                m.MIDPOINT_DOMAIN + m._canonical_json(record) + record_payload
            )
            records.append(record)
            offset += m.MIDPOINT_RECORD_BYTES
        metadata = {
            "schema": m.MIDPOINT_SCHEMA, "relative_path": "JX_MIDPOINTS.bin",
            "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_PRIMARY_RELATIVE_TRACER_POSITIONS_THEN_VELOCITIES",
            "midpoint_year": 500, "byte_length": len(payload),
            "file_sha256": m._sha256(payload),
            "content_sha256": m._sha256(m.MIDPOINT_DOMAIN + payload),
            "record_count": 24, "record_bytes": m.MIDPOINT_RECORD_BYTES,
            "record_order": "EPHEMERIS_THEN_BATCH_THEN_JX_STEP_64",
            "records": records, "fresh_process_restart_cases": [],
        }
        bridge = {"batches": [{"object_id": value} for value in m.OBJECT_IDS]}
        fake = types.SimpleNamespace(
            _unpack_jx_internal_tracer_state=lambda value: self.assertEqual(
                len(value), m.MIDPOINT_RECORD_BYTES
            )
        )
        m._validate_midpoint_metadata(fake, metadata, payload, bridge)
        changed = json.loads(json.dumps(metadata))
        changed["records"][3]["object_id"] = "relabelled"
        with self.assertRaises(m.ClosureError):
            m._validate_midpoint_metadata(fake, changed, payload, bridge)

    def test_source_order_is_truthful_and_compatibility_projection_is_private(self) -> None:
        m = self.module
        observed = {}

        class Fake:
            def _validate_jx_source_evidence(self, handle, metadata, *args):
                observed["order"] = metadata["order"]
                observed["handle"] = handle

        retained = {
            "order": "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_64_POSITIONS_THEN_VELOCITIES"
        }
        marker = object()
        m._validate_h64_source_evidence(
            Fake(), marker, retained, [], {}, {}, types.ModuleType("numpy")
        )
        self.assertEqual(
            retained["order"],
            "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_64_POSITIONS_THEN_VELOCITIES",
        )
        self.assertEqual(
            observed["order"],
            "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_16_32_POSITIONS_THEN_VELOCITIES",
        )
        self.assertIs(observed["handle"], marker)

    def test_restart_receipt_roster_fails_closed_before_runtime_validation(self) -> None:
        m = self.module
        fake = types.SimpleNamespace(RESTART_REQUEST_KEYS=frozenset())
        with self.assertRaises(m.ClosureError):
            m._validate_restart_receipts(
                fake,
                {"records": [], "fresh_process_restart_cases": []},
                {},
            )

    def test_array_metadata_rejects_relabel(self) -> None:
        m = self.module
        payload = b"\0" * m.EXPECTED_ARRAY_BYTES
        metadata = {
            "schema": m.ARRAY_SCHEMA, "relative_path": "ARRAYS.bin",
            "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
            "byte_length": len(payload), "file_sha256": m._sha256(payload),
            "content_sha256": m._sha256(m.ARRAY_DOMAIN + payload),
            "batch_lane_count": 24,
            "ordering": "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL",
            "all_vectors_origin": "LANE_INSTANTANEOUS_MASSIVE_SYSTEM_COMMON_GM_BARYCENTER",
            "position_unit": "AU", "velocity_unit": "AU_PER_JULIAN_YEAR",
        }
        m._validate_array_metadata(metadata, payload)
        metadata["batch_lane_count"] = 23
        with self.assertRaises(m.ClosureError):
            m._validate_array_metadata(metadata, payload)

    def test_create_only_prepare_and_validation(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            protocol = m.prepare(root)
            self.assertEqual(m.validate_prepared(root), protocol)
            with self.assertRaises(m.ClosureError):
                m.prepare(root)

    def test_create_only_result_publication_uses_exact_roster(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            (root / "PROTOCOL.json").write_bytes(b"protocol")
            artifacts = tuple(
                (name, name.encode("ascii"))
                for name in (
                    "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin",
                    "JX_SOURCE_NODES.bin", "ARRAYS.bin", "RESULT.json",
                )
            )
            m._publish_result_artifacts(root, artifacts)
            self.assertEqual(
                {item.name for item in root.iterdir()},
                {"PROTOCOL.json", *(name for name, _payload in artifacts)},
            )
            with self.assertRaises(m.ClosureError):
                m._publish_result_artifacts(root, artifacts)

    def test_stable_file_rejects_symlink_and_multiple_links(self) -> None:
        m = self.module
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            original.write_bytes(b"sealed")
            self.assertEqual(m._read_small(original, "original", 6), b"sealed")
            link = root / "link"
            link.symlink_to(original)
            with self.assertRaises(m.ClosureError):
                m._read_small(link, "symlink", 6)
            hard = root / "hard"
            os.link(original, hard)
            with self.assertRaises(m.ClosureError):
                m._read_small(original, "multiply linked", 6)

    def test_protocol_is_optimization_invariant(self) -> None:
        m = self.module
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-OO", "-B", str(SOURCE), "protocol"],
            cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, check=False, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        observed = json.loads(completed.stdout.decode("ascii"))
        self.assertEqual(observed["content_sha256"], m.build_protocol()["content_sha256"])


if __name__ == "__main__":
    unittest.main()
