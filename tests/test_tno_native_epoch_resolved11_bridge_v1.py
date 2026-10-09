"""Focused tests for the prepare-only native-epoch resolved-11 bridge."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "jx_tno_native_epoch_resolved11_bridge_v1.py"
OUTPUT = ROOT / "runs" / "tno_native_epoch_resolved11_bridge_v1"
SPEC = importlib.util.spec_from_file_location("jx_tno_native_epoch_resolved11_bridge_v1_tested", SOURCE)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("bridge implementation cannot be loaded")
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


EXPECTED_ET = {
    "jpl-20090377": 593_265_600,
    "jpl-20474640": 401_112_000,
    "jpl-54526835": 733_492_800,
    "jpl-20541132": 562_248_000,
    "jpl-50666516": 436_449_600,
    "jpl-20148209": 88_430_400,
    "jpl-20574372": 583_588_800,
    "jpl-20303775": 624_024_000,
    "jpl-20026181": 571_233_600,
    "jpl-20042301": 585_748_800,
    "jpl-20026375": 554_385_600,
    "jpl-20015874": 566_222_400,
}


class ProviderAndTransformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.stage0 = module._validate_stage0()
        cls.protocol = module.build_protocol()

    def test_exact_production_provider_without_test_fixture_dependency(self) -> None:
        provider = module.production_provider()
        self.assertEqual(
            provider.content_sha256,
            "2a560649817b498d98172a3e6da9a63be0363a773ee6dae77f98b3d4d590d09f",
        )
        self.assertEqual(
            tuple(item.artifact_id for item in provider.artifacts),
            (
                "artifact.cspice.de440s",
                "artifact.cspice.license",
                "artifact.cspice.naif-rules-license",
                "artifact.cspice.native",
                "artifact.cspice.numpy-license",
                "artifact.cspice.numpy-wheel",
                "artifact.cspice.spiceypy-wheel",
            ),
        )
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("tests.test_solar_system", source)
        self.assertNotIn("from tests", source)

    def test_obliquity_is_independently_rerounded_and_exact_pxform_kat(self) -> None:
        radians, sine, cosine = module.independently_rounded_obliquity()
        self.assertEqual(radians.hex(), "0x1.a2e9395ca7c40p-2")
        self.assertEqual(sine.hex(), "0x1.9752e50f4b399p-2")
        self.assertEqual(cosine.hex(), "0x1.d5c0357681ef3p-1")
        kat = module._pxform_kat()
        self.assertEqual(kat["loaded_cspice_sha256"], "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84")
        self.assertEqual(kat["matrix_binary64_hex"][1][2], "-0x1.9752e50f4b399p-2")

    def test_rotation_uses_exact_dyadic_dot_then_one_round(self) -> None:
        vector = (float.fromhex("0x1.23456789abcdep+4"), float.fromhex("-0x1.13579bdf2468ap+5"), float.fromhex("0x1.fedcba9876543p+3"))
        observed = module._rotate_ecliptic_to_j2000(vector)
        sine = Fraction.from_float(float.fromhex(module.OBLIQUITY_SIN_HEX))
        cosine = Fraction.from_float(float.fromhex(module.OBLIQUITY_COS_HEX))
        expected = (
            vector[0],
            float(cosine * Fraction.from_float(vector[1]) - sine * Fraction.from_float(vector[2])),
            float(sine * Fraction.from_float(vector[1]) + cosine * Fraction.from_float(vector[2])),
        )
        self.assertEqual(tuple(item.hex() for item in observed), tuple(item.hex() for item in expected))
        recovered = module._inverse_rotate_j2000_to_ecliptic(observed)
        self.assertLessEqual(max(abs(a - b) for a, b in zip(recovered, vector, strict=True)), 8.0e-15)

    def test_stage0_selection_is_twelve_distinct_native_epoch_batches(self) -> None:
        groups = module._selected_stage0_rows(self.stage0)
        self.assertEqual(len(groups), 12)
        self.assertEqual(len({object_id for object_id, _epoch, _rows in groups}), 12)
        self.assertEqual(len({epoch.hex() for _object_id, epoch, _rows in groups}), 12)
        self.assertEqual(sum(len(rows) for _object_id, _epoch, rows in groups), 108)
        for object_id, epoch, rows in groups:
            self.assertEqual(module._et_seconds(epoch.hex()), EXPECTED_ET[object_id])
            self.assertEqual(rows[0]["clone_index"], None)
            self.assertEqual(tuple(row["clone_index"] for row in rows[1:]), module.HOLDOUT_CLONE_INDICES)
            self.assertTrue(all(row["object_id"] == object_id for row in rows))
            self.assertEqual(len({row["covariance_epoch_jd_tdb_hex"] for row in rows}), 1)

    def test_protocol_fails_closed_on_mismatch_and_authority_claims(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["selection"]["selected_tno_state_count"], 108)
        self.assertFalse(protocol["selection"]["mixed_epoch_snapshot_created"])
        self.assertFalse(protocol["selection"]["full_1548_state_dynamics_tile_created"])
        self.assertEqual(
            {row["object_id"]: row["spice_et_seconds"] for row in protocol["selection"]["native_epoch_roster"]},
            EXPECTED_ET,
        )
        self.assertFalse(protocol["model_mismatch_gate"]["de440_de441_equivalence_established"])
        self.assertFalse(protocol["model_mismatch_gate"]["future_gate_executed"])
        self.assertFalse(protocol["model_mismatch_gate"]["coordinate_difference_cutoff_sufficient"])
        self.assertFalse(protocol["claims"]["source_tile_created"])
        self.assertFalse(protocol["claims"]["dynamics_executed"])
        self.assertFalse(protocol["claims"]["one_thousand_year_holdout_executed"])
        self.assertFalse(protocol["claims"]["science_result"])
        self.assertFalse(protocol["execution_authorized"])
        self.assertFalse(protocol["outcomes_generated"])
        tampered = dict(protocol)
        tampered["execution_authorized"] = True
        self.assertNotEqual(module._content_sha256(module.PROTOCOL_DOMAIN, tampered), protocol["content_sha256"])

    def test_external_bundle_and_runtime_source_bindings_are_exact(self) -> None:
        bundle = module._validate_external_bundle(OUTPUT)
        self.assertEqual(bundle["file_count"], 86)
        self.assertEqual(bundle["sha256sums_entry_count"], 85)
        self.assertEqual(bundle["source_tree_sha256"], "84cc729c6ac29fa6f95b8a36762d530962279ce3d3dac97ccff5b2cb4a4df88b")
        self.assertEqual(len(module._source_records()), len(module.RUNTIME_SOURCE_BINDINGS))

    def test_isolated_worker_is_hash_bound_but_not_parent_import_required(self) -> None:
        bound = {row["relative_path"]: row for row in module._source_records()}
        worker = module.ISOLATED_WORKER_SOURCE_RELATIVE_PATH
        self.assertIn(worker, bound)
        self.assertEqual(bound[worker]["bytes"], 40_504)
        self.assertEqual(bound[worker]["file_sha256"], "09824658b6acb407ea683ff6d9c24df07bff3684549f5b478844f24c7d6e0951")
        observed = {row["relative_path"] for row in module._loaded_project_source_observations()}
        required_parent = {
            row[0]
            for row in module.RUNTIME_SOURCE_BINDINGS
            if row[0] != worker
        }
        self.assertTrue(required_parent.issubset(observed))
        self.assertNotIn(worker, observed)

    def test_receipt_serializer_roundtrips_exact_contract_dataclasses(self) -> None:
        provider = module.production_provider()
        frozen = module._freeze(provider)
        restored = module._thaw(frozen)
        self.assertEqual(restored.content_sha256, provider.content_sha256)
        hostile = json.loads(json.dumps(frozen))
        hostile["fields"]["content_sha256"] = "f" * 64
        with self.assertRaises(module.BridgeError):
            module._thaw(hostile)

    def test_prepare_only_surface_refuses_unstaged_or_nonempty_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(module.BridgeError):
                module.materialize(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_strict_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with self.assertRaises(module.BridgeError):
            module._strict_json(b'{"a":1,"a":2}')
        with self.assertRaises(module.BridgeError):
            module._strict_json(b'{"a":NaN}')


class RetainedOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        if not (OUTPUT / "PROTOCOL.json").exists():
            self.skipTest("create-only bridge output has not yet been materialized")

    def test_full_retained_artifact_reconstructs_offline(self) -> None:
        artifact = module.validate_output(OUTPUT)
        self.assertEqual(artifact["batch_count"], 12)
        self.assertEqual(artifact["massive_state_rows_total"], 132)
        self.assertEqual(artifact["tno_state_rows_total"], 108)
        self.assertEqual(len({batch["covariance_epoch_jd_tdb_hex"] for batch in artifact["batches"]}), 12)
        self.assertTrue(all(batch["massive_body_count"] == 11 for batch in artifact["batches"]))
        self.assertTrue(all(batch["tno_state_count"] == 9 for batch in artifact["batches"]))
        self.assertTrue(all(batch["tno_excluded_from_centroid"] for batch in artifact["batches"]))
        self.assertTrue(all(batch["roundtrip"]["all_nine_pass"] for batch in artifact["batches"]))
        self.assertTrue(all(batch["massive_sun_primary_crosscheck"]["all_eleven_pass"] for batch in artifact["batches"]))
        for batch in artifact["batches"]:
            sun = batch["massive_bodies"][0]
            self.assertEqual(sun["sun_primary_position_AU_hex"], ["0x0.0p+0"] * 3)
            self.assertEqual(sun["sun_primary_velocity_AU_per_julian_year_hex"], ["0x0.0p+0"] * 3)
        self.assertFalse(artifact["claims"]["dynamics_executed"])
        self.assertFalse(artifact["claims"]["de440_de441_model_mismatch_resolved"])

    def test_receipt_graphs_are_full_exact_and_massless_tnos_never_enter_centroid(self) -> None:
        protocol = module._load_canonical(OUTPUT / "PROTOCOL.json")
        receipts = module._load_canonical(OUTPUT / "RECEIPTS.json")
        initials = module.validate_receipts(receipts, protocol)
        self.assertEqual(len(initials), 12)
        for initial, entry in zip(initials, receipts["entries"], strict=True):
            self.assertGreater(len(entry["nested_content_receipt_inventory"]), 400)
            self.assertEqual(initial.naif_ids, module.RESOLVED11_NAIF_IDS)
            self.assertTrue(all(initial.massive))
            self.assertNotIn(entry["object_id"], initial.body_ids)
            self.assertEqual(
                initial.preparation_receipt.source_execution.semantic_replay_status,
                "DISTINCT_CALLER_INSTANCE_NONCES_AND_EQUAL_SEMANTIC_CONTENT_OBSERVED",
            )

    def test_swapped_or_relabelled_valid_epoch_graphs_fail_closed(self) -> None:
        protocol = module._load_canonical(OUTPUT / "PROTOCOL.json")
        original = module._load_canonical(OUTPUT / "RECEIPTS.json")
        swapped = json.loads(json.dumps(original))
        swapped["entries"][0]["sealed_initial_state"], swapped["entries"][1]["sealed_initial_state"] = (
            swapped["entries"][1]["sealed_initial_state"],
            swapped["entries"][0]["sealed_initial_state"],
        )
        swapped["content_sha256"] = module._content_sha256(module.RECEIPTS_DOMAIN, swapped)
        with self.assertRaises(module.BridgeError):
            module.validate_receipts(swapped, protocol)

        initial = module._thaw(original["entries"][0]["sealed_initial_state"])
        wrong = protocol["selection"]["native_epoch_roster"][1]
        with self.assertRaises(module.BridgeError):
            module._assert_initial_identity(
                initial,
                wrong["object_id"],
                float.fromhex(wrong["covariance_epoch_jd_tdb_hex"]),
                wrong["spice_et_seconds"],
            )

    def test_optimized_python_reconstructs_identical_hashes(self) -> None:
        code = f"""
import importlib.util,json
from pathlib import Path
p=Path({str(SOURCE)!r})
s=importlib.util.spec_from_file_location('bridge_oo',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
x=m.validate_output(Path({str(OUTPUT)!r}))
print(json.dumps([x['content_sha256'],x['binary_payload_sha256']]))
"""
        completed = subprocess.run(
            [sys.executable, "-OO", "-B", "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
            timeout=1_200,
        )
        observed = json.loads(completed.stdout)
        normal = module.validate_output(OUTPUT)
        self.assertEqual(observed, [normal["content_sha256"], normal["binary_payload_sha256"]])


if __name__ == "__main__":
    unittest.main()
