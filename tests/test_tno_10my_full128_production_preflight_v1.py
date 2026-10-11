from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_full128_production_preflight_v1.py"


def load_module():
    name = "jx_tno_10my_full128_production_preflight_v1_tested"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load full128 preflight source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Full128ProductionPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.m = load_module()
        cls.protocol = cls.m.build_protocol()

    def bound_semantic_inputs(self):
        _records, payloads = self.m._bound_inputs()
        return self.m._validate_semantic_inputs(payloads)

    def synthetic_smoke_prefix(self) -> bytes:
        retained = (
            PROJECT_ROOT
            / "runs"
            / "tno_1kyr_reduced5_lifecycle_gate_v3"
            / "PREFIX_STATES.bin"
        ).read_bytes()
        records: list[bytes] = []
        for checkpoint_order in range(len(self.m.SMOKE_CHECKPOINT_YEARS)):
            payload = self.m._v3_checkpoint(retained, 0, checkpoint_order)
            values = struct.unpack("<84d", payload)
            rows = [tuple(values[index * 6:(index + 1) * 6]) for index in range(14)]
            tracers = [rows[5] for _ in range(129)]
            for selected_order, tracer_index in enumerate(self.m.HOLDOUT_TRACER_INDICES):
                tracers[tracer_index] = rows[5 + selected_order]
            records.append(self.m._pack_rows(tuple(rows[:5]) + tuple(tracers)))
        stage0, bridge, custody, prefix = self.bound_semantic_inputs()
        *_summary, checkpoint_payloads = self.m._initial_state_records(
            stage0, bridge, custody, prefix
        )
        records[0] = checkpoint_payloads[0]
        result = b"".join(records)
        self.assertEqual(len(result), self.m.SMOKE_PREFIX_BYTES)
        return result

    def test_geometry_and_storage_arithmetic(self) -> None:
        m = self.m
        self.assertEqual((m.BATCH_COUNT, m.TRACER_COUNT, m.BODY_COUNT), (12, 129, 134))
        self.assertEqual(m.STATE_RECORD_BYTES, 6_432)
        self.assertEqual(m.ELEMENT_ENTITY_COUNT, 130)
        self.assertEqual(m.ELEMENT_RECORD_BYTES, 6_240)
        self.assertEqual(m.B22_NODE_COUNT, 50_001)
        self.assertEqual(m.ML_SHORT_NODE_COUNT, 10_001)
        self.assertEqual(m.ML_LONG_NODE_COUNT, 10_001)
        self.assertEqual(m.UNION_NODE_COUNT, 57_501)
        self.assertEqual(
            m.OUTPUT_SIZES,
            {
                "B22_STATES.f64le": 3_859_277_184,
                "B22_ELEMENTS.f64le": 3_744_074_880,
                "B22_NODE_SHA256.raw": 19_200_384,
                "ML_SHORT_STATES.f64le": 771_917_184,
            },
        )
        self.assertEqual(m.TOTAL_RETAINED_BINARY_BYTES, 8_394_469_632)

    def test_protocol_reconstructs_and_claims_stay_bounded(self) -> None:
        m = self.m
        protocol = self.protocol
        self.assertEqual(protocol["schema"], m.PROTOCOL_SCHEMA)
        self.assertEqual(m.validate_protocol(protocol), protocol["content_sha256"])
        self.assertEqual(protocol["full_clone_initial_states"]["total_tracers"], 1_548)
        self.assertEqual(
            protocol["implementation"]["filesystem_trust_boundary"],
            "NO_CONCURRENT_SAME_UID_FILESYSTEM_MUTATOR",
        )
        self.assertEqual(
            protocol["publication"]["filesystem_trust_boundary"],
            self.m.FILESYSTEM_TRUST_BOUNDARY,
        )
        self.assertEqual(
            protocol["full_clone_initial_states"]["bridge_holdout_rotation_bitwise_crosschecks"],
            108,
        )
        self.assertEqual(
            protocol["full_clone_initial_states"]["v3_year_zero_row_bitwise_crosschecks"],
            168,
        )
        claims = protocol["claims"]
        for key in (
            "bounded_gpu_smoke_executed", "gpu_executed",
            "ten_myr_execution_authorized", "ten_myr_execution_completed",
            "classification_generated", "robust_128_clone_classification_generated",
            "population_inference", "planet_nine_evidence", "science_result",
        ):
            self.assertIs(claims[key], False)
        self.assertIn(
            "CURRENT_GRID_ADAPTER_IS_FIXED_TO_NINE_TRACERS_AND_CANNOT_BE_REUSED_UNCHANGED",
            protocol["blockers"],
        )

    def test_exact_v3_eight_file_result_and_git_lineage_are_bound(self) -> None:
        records = self.protocol["lineage"]["bound_inputs"]
        roles = {row["role"]: row for row in records}
        expected = {
            "v3_protocol", "v3_prefix", "v3_restart", "v3_segments",
            "v3_routes", "v3_preflight", "v3_runtime", "v3_result",
        }
        self.assertEqual(expected, {role for role in roles if role.startswith("v3_") and role not in {"v3_source", "v3_test"}})
        self.assertEqual(self.protocol["lineage"]["v3_result_root_exact_file_count"], 8)
        self.assertEqual(self.protocol["lineage"]["v3_result_archive_commit"], self.m.V3_ARCHIVE_COMMIT)
        self.assertEqual(self.protocol["lineage"]["v3_result_archive_tag_object"], self.m.V3_ARCHIVE_TAG_OBJECT)

    def test_full_clone_order_and_per_batch_payloads(self) -> None:
        initial = self.protocol["full_clone_initial_states"]
        order = initial["sample_order"]
        self.assertEqual(len(order), 129)
        self.assertEqual(order[0], {"sample_kind": "NOMINAL", "clone_index": None})
        self.assertEqual([row["clone_index"] for row in order[1:]], list(range(128)))
        self.assertEqual(len(initial["batches"]), 12)
        for batch_order, row in enumerate(initial["batches"]):
            self.assertEqual(row["batch_order"], batch_order)
            self.assertEqual(row["object_id"], self.m.OBJECT_IDS[batch_order])
            self.assertEqual(row["primary_relative_payload_bytes"], 6_432)
            self.assertEqual(row["barycentric_checkpoint_payload_bytes"], 6_432)
            self.assertEqual(row["bridge_holdout_rotation_crosscheck_count"], 9)
            self.assertEqual(row["v3_year_zero_bitwise_crosscheck_count"], 14)
            self.assertRegex(row["primary_relative_payload_sha256"], r"^[0-9a-f]{64}$")

    def test_rotation_and_v3_crosschecks_fail_on_relabel(self) -> None:
        stage0, bridge, custody, prefix = self.bound_semantic_inputs()
        attacked_stage0 = copy.deepcopy(stage0)
        attacked_stage0["states"][1]["clone_index"] = 1
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "full-clone order"):
            self.m._initial_state_records(attacked_stage0, bridge, custody, prefix)

        attacked_bridge = copy.deepcopy(bridge)
        token = attacked_bridge["batches"][0]["tnos"][0]["heliocentric_j2000_position_AU_hex"][0]
        attacked_bridge["batches"][0]["tnos"][0]["heliocentric_j2000_position_AU_hex"][0] = math.nextafter(float.fromhex(token), math.inf).hex()
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "rotation differs"):
            self.m._initial_state_records(stage0, attacked_bridge, custody, prefix)

        attacked_custody = copy.deepcopy(custody)
        attacked_custody["state_records"][0]["object_id"] = self.m.OBJECT_IDS[1]
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "epoch/body roster"):
            self.m._initial_state_records(stage0, bridge, attacked_custody, prefix)

        attacked_stage0 = copy.deepcopy(stage0)
        clone_one = next(
            row for row in attacked_stage0["states"]
            if row.get("object_id") == self.m.OBJECT_IDS[0] and row.get("clone_index") == 1
        )
        clone_one["clone_index"] = True
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "full-clone order"):
            self.m._initial_state_records(attacked_stage0, bridge, custody, prefix)

        attacked_custody = copy.deepcopy(custody)
        attacked_custody["state_records"][0]["center_naif_id"] = False
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "epoch/body roster"):
            self.m._initial_state_records(stage0, bridge, attacked_custody, prefix)

    def test_bound_byte_hash_attack_fails(self) -> None:
        original = self.m.BOUND_FILES
        row = list(original[0])
        row[3] = "0" * 64
        self.m.BOUND_FILES = (tuple(row), *original[1:])
        try:
            with self.assertRaisesRegex(self.m.ProductionPreflightError, "SHA-256 differs"):
                self.m.build_protocol()
        finally:
            self.m.BOUND_FILES = original

    def test_smoke_projection_receipt_accepts_bound_fixture_without_claiming_pass(self) -> None:
        prefix = self.synthetic_smoke_prefix()
        evidence = self.m.validate_smoke_evidence_v1(
            prefix,
            active_counts=[129] * 21,
            routes_jsonl=b"",
            removals_jsonl=b"",
            elapsed_seconds=1.25,
        )
        self.assertEqual(evidence["schema"], self.m.SMOKE_EVIDENCE_SCHEMA)
        self.assertIs(evidence["v3_holdout_projection_bitwise_equal"], True)
        self.assertIs(evidence["claims"]["projection_only_receipt_validated"], True)
        self.assertIs(evidence["claims"]["bounded_gpu_smoke_passed"], False)
        self.assertIs(evidence["claims"]["gpu_executed"], False)
        self.assertIs(evidence["claims"]["ten_myr_execution_authorized"], False)
        self.assertEqual(evidence["elapsed_time_claim"], "DIAGNOSTIC_ONLY_NO_PERFORMANCE_GATE")
        self.assertEqual(
            evidence["content_sha256"],
            self.m._domain_hash(self.m.SMOKE_EVIDENCE_DOMAIN, evidence),
        )

    def test_smoke_rejects_fabricated_nonholdout_year_zero_cohort(self) -> None:
        prefix = bytearray(self.synthetic_smoke_prefix())
        first = bytes(prefix[:self.m.STATE_RECORD_BYTES])
        rows = self.m._unpack_rows(first)
        fabricated = self.m._pack_rows(
            (*rows[:self.m.MASSIVE_COUNT],)
            + tuple(rows[self.m.MASSIVE_COUNT] for _ in range(self.m.TRACER_COUNT))
        )
        for selected_order, tracer_index in enumerate(self.m.HOLDOUT_TRACER_INDICES):
            target = (self.m.MASSIVE_COUNT + tracer_index) * 6 * 8
            source = (self.m.MASSIVE_COUNT + selected_order) * 6 * 8
            fabricated = (
                fabricated[:target]
                + first[source:source + 6 * 8]
                + fabricated[target + 6 * 8:]
            )
        prefix[:self.m.STATE_RECORD_BYTES] = fabricated
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "full129 year-zero"):
            self.m.validate_smoke_evidence_v1(
                bytes(prefix), active_counts=[129] * 21,
                routes_jsonl=b"", removals_jsonl=b"", elapsed_seconds=1.0,
            )

    def test_smoke_rejects_nonfinite_unselected_clone(self) -> None:
        prefix = bytearray(self.synthetic_smoke_prefix())
        nonholdout_tracer_index = 2
        component_offset = (self.m.MASSIVE_COUNT + nonholdout_tracer_index) * 6 * 8
        struct.pack_into("<d", prefix, component_offset, math.nan)
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "non-finite"):
            self.m.validate_smoke_evidence_v1(
                bytes(prefix), active_counts=[129] * 21,
                routes_jsonl=b"", removals_jsonl=b"", elapsed_seconds=1.0,
            )

    def test_smoke_rejects_holdout_active_route_removal_and_timing_attacks(self) -> None:
        prefix = self.synthetic_smoke_prefix()
        attacked = bytearray(prefix)
        offset = self.m.STATE_RECORD_BYTES + self.m.MASSIVE_COUNT * 6 * 8
        value = struct.unpack_from("<d", attacked, offset)[0]
        struct.pack_into("<d", attacked, offset, math.nextafter(value, math.inf))
        common = dict(active_counts=[129] * 21, routes_jsonl=b"", removals_jsonl=b"", elapsed_seconds=1.0)
        with self.assertRaisesRegex(self.m.ProductionPreflightError, "differs bitwise"):
            self.m.validate_smoke_evidence_v1(bytes(attacked), **common)
        for replacement, message in (
            ({"active_counts": [129] * 20 + [128]}, "active-count"),
            ({"active_counts": [129.0] * 21}, "active-count"),
            ({"routes_jsonl": b"{}\n"}, "route evidence"),
            ({"removals_jsonl": b"{}\n"}, "removal evidence"),
            ({"elapsed_seconds": True}, "elapsed time"),
        ):
            kwargs = dict(common)
            kwargs.update(replacement)
            with self.assertRaisesRegex(self.m.ProductionPreflightError, message):
                self.m.validate_smoke_evidence_v1(prefix, **kwargs)

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with self.assertRaises(self.m.ProductionPreflightError):
            self.m._strict_json(b'{"a":1,"a":2}\n')
        with self.assertRaises(self.m.ProductionPreflightError):
            self.m._strict_json(b'{"a":NaN}\n')

    def test_exact_directory_roster_rejects_extra_name_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "root"
            root.mkdir()
            (root / "A").write_bytes(b"a")
            self.m._require_directory_roster(root, {"A"}, "fixture")
            (root / "B").write_bytes(b"b")
            with self.assertRaisesRegex(self.m.ProductionPreflightError, "roster differs"):
                self.m._require_directory_roster(root, {"A"}, "fixture")
            link = parent / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(self.m.ProductionPreflightError):
                self.m._require_directory_roster(link, {"A", "B"}, "fixture link")

    def test_prepare_validate_create_only_and_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "prepared"
            observed = self.m.prepare(root)
            self.assertEqual(observed, self.protocol)
            self.assertEqual(self.m.validate_prepared(root), self.protocol)
            with self.assertRaisesRegex(self.m.ProductionPreflightError, "already exists"):
                self.m.prepare(root)
            (root / "EXTRA").write_bytes(b"x")
            with self.assertRaisesRegex(self.m.ProductionPreflightError, "roster"):
                self.m.validate_prepared(root)

    def test_prepared_protocol_tamper_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "prepared"
            self.m.prepare(root)
            value = json.loads((root / "PROTOCOL.json").read_bytes())
            value["claims"]["ten_myr_execution_authorized"] = True
            (root / "PROTOCOL.json").write_bytes(self.m._canonical_json(value))
            with self.assertRaises(self.m.ProductionPreflightError):
                self.m.validate_prepared(root)
            target = parent / "target"
            target.mkdir()
            link = parent / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaises(self.m.ProductionPreflightError):
                self.m.validate_prepared(link)

    @unittest.skipIf(sys.flags.optimize >= 2, "normal interpreter owns opposite-mode reconstruction")
    def test_optimized_build_is_byte_identical(self) -> None:
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-OO", str(SOURCE), "build"],
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=180,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))
        self.assertEqual(completed.stdout, self.m._canonical_json(self.protocol))

    def test_default_root_is_not_materialized_by_tests(self) -> None:
        root = self.m.DEFAULT_OUTPUT_ROOT
        if root.exists():
            self.assertEqual(set(path.name for path in root.iterdir()), {"PROTOCOL.json"})
            self.assertEqual(self.m.validate_prepared(root), self.protocol)


if __name__ == "__main__":
    unittest.main()
