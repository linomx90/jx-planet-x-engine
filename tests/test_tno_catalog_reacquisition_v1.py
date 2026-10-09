from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "jx_tno_catalog_reacquisition_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_catalog_reacquisition_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load TNO reacquisition module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_module()


def _fixture_spec(payload: bytes = b"fixture-payload\n"):
    return ACQ.ArtifactSpec(
        artifact_id="FIXTURE",
        staging_name="fixture.bin",
        retained_name="fixture.bin",
        url="https://example.invalid/fixture.bin",
        byte_count=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        survey="FIXTURE",
        role="test_only",
        record_count=1,
        license_status="TEST_ONLY",
    )


class TnoCatalogReacquisitionTests(unittest.TestCase):
    def test_protocol_is_exact_catalog_custody_only(self) -> None:
        protocol = ACQ.build_protocol()
        self.assertEqual(protocol["schema"], ACQ.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["classification"], ACQ.CLASSIFICATION)
        self.assertEqual(protocol["claim_ceiling"], "INPUT_CUSTODY_ONLY")
        self.assertIs(protocol["execution_authorized"], False)
        self.assertIs(protocol["outcomes_generated"], False)
        self.assertEqual(len(protocol["artifact_roster"]), 9)
        self.assertEqual(
            ACQ.validate_protocol(protocol), protocol["content_sha256"]
        )
        self.assertTrue(all(value is False for value in protocol["claims"].values()))
        forbidden = protocol["authorization"]["forbidden"]
        self.assertIn("sample_covariances", forbidden)
        self.assertIn("run_gpu_dynamics", forbidden)
        self.assertIn("make_population_or_planet_nine_claims", forbidden)
        exact_roster = {
            row["artifact_id"]: (row["url"], row["byte_count"], row["sha256"])
            for row in protocol["artifact_roster"]
        }
        self.assertEqual(
            exact_roster["OSSOS_CHARACTERIZED_CATALOG"],
            (
                "https://cdsarc.cds.unistra.fr/ftp/J/ApJS/236/18/t3char.dat",
                225_960,
                "35c6af083804788d589e1f7fbcbf3db30fecd6d32d94be3de359815e99da596f",
            ),
        )
        self.assertEqual(
            exact_roster["DES_Y6_ENHANCED_AUTHOR_CATALOG"],
            (
                "https://raw.githubusercontent.com/bernardinelli/des_tno_catalog/8f254ca30fa0e8fc1576a5a9fad7300124d3b3ee/y6_des_tnos_color.fits",
                527_040,
                "4f2153f8726485bb661cbf9027a840c69e11c6d87c925d386055b3fdff4ea6e8",
            ),
        )
        changed = json.loads(json.dumps(protocol))
        changed["claims"]["science_result"] = True
        with self.assertRaisesRegex(ACQ.ReacquisitionError, "differs"):
            ACQ.validate_protocol(changed)

    def test_prepare_is_exact_idempotent_and_refuses_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = ACQ.prepare(output)
            second = ACQ.prepare(output)
            self.assertEqual(first, second)
            raw = (output / "PROTOCOL.json").read_bytes()
            self.assertEqual(raw, ACQ._canonical_json(first))
            (output / "PROTOCOL.json").write_bytes(raw + b" ")
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "different protocol"):
                ACQ.prepare(output)

    def test_g0_inventory_parity_and_root_roster_are_fail_closed(self) -> None:
        changed = replace(ACQ.ARTIFACTS[0], url="https://example.invalid/changed")
        with mock.patch.object(ACQ, "ARTIFACTS", (changed, *ACQ.ARTIFACTS[1:])):
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "G0 inventory"):
                ACQ.build_protocol()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            ACQ.prepare(output)
            (output / "SCIENCE_READY.json").write_text("{}", encoding="ascii")
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "unexpected entries"):
                ACQ.prepare(output)

    def test_generic_staging_requires_exact_roster_bytes_and_regular_files(self) -> None:
        payload = b"fixture-payload\n"
        spec = _fixture_spec(payload)
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir()
            (staging / spec.staging_name).write_bytes(payload)
            records = ACQ.verify_staging(staging, (spec,))
            self.assertEqual(records[0]["verification_status"], "EXACT_BYTE_MATCH")

            (staging / "extra").write_bytes(b"x")
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "roster"):
                ACQ.verify_staging(staging, (spec,))
            (staging / "extra").unlink()

            (staging / spec.staging_name).write_bytes(b"wrong")
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "byte count"):
                ACQ.verify_staging(staging, (spec,))

            (staging / spec.staging_name).unlink()
            (staging / spec.staging_name).symlink_to(Path(directory) / "missing")
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "linked or nonregular"):
                ACQ.verify_staging(staging, (spec,))

    def test_fixture_publish_validate_and_tamper_fail_closed(self) -> None:
        payload = b"fixture-payload\n"
        spec = _fixture_spec(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            output = root / "output"
            staging.mkdir()
            (staging / spec.staging_name).write_bytes(payload)
            with (
                mock.patch.object(ACQ, "ARTIFACTS", (spec,)),
                mock.patch.object(ACQ, "_validate_artifact_inventory_parity"),
            ):
                protocol = ACQ.prepare(output)
                self.assertEqual(len(protocol["artifact_roster"]), 1)
                receipt = ACQ.publish(staging, output)
                self.assertEqual(receipt["artifact_count"], 1)
                self.assertTrue(
                    receipt["readiness"]["catalog_payload_custody_established"]
                )
                self.assertFalse(receipt["readiness"]["science_execution_authorized"])
                self.assertEqual(receipt["claim_ceiling"], "INPUT_CUSTODY_ONLY")
                self.assertIs(receipt["execution_authorized"], False)
                self.assertIs(receipt["outcomes_generated"], False)
                self.assertEqual(
                    ACQ.validate_output(output)["content_sha256"],
                    receipt["content_sha256"],
                )
                with self.assertRaisesRegex(ACQ.ReacquisitionError, "already exists"):
                    ACQ.publish(staging, output)
                (output / "SCIENCE_READY.json").write_text("{}", encoding="ascii")
                with self.assertRaisesRegex(ACQ.ReacquisitionError, "root roster"):
                    ACQ.validate_output(output)
                (output / "SCIENCE_READY.json").unlink()
                manifest_path = output / "CUSTODY.json"
                original_manifest = manifest_path.read_bytes()
                forged = json.loads(original_manifest)
                forged["status"] = "SCIENCE_READY"
                forged["content_sha256"] = ACQ._content_sha256(
                    ACQ.CUSTODY_DOMAIN, forged
                )
                manifest_path.write_bytes(ACQ._canonical_json(forged))
                with self.assertRaisesRegex(ACQ.ReacquisitionError, "reconstruction"):
                    ACQ.validate_output(output)
                manifest_path.write_bytes(original_manifest)
                (output / "raw" / spec.retained_name).write_bytes(b"altered-payload\n")
                with self.assertRaisesRegex(ACQ.ReacquisitionError, "retained bytes"):
                    ACQ.validate_output(output)

    def test_changed_fixture_metadata_or_unexpected_raw_entry_rejects(self) -> None:
        payload = b"fixture-payload\n"
        spec = _fixture_spec(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            output = root / "output"
            staging.mkdir()
            (staging / spec.staging_name).write_bytes(payload)
            bad = replace(spec, sha256="0" * 64)
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "SHA-256"):
                ACQ.verify_staging(staging, (bad,))
            with (
                mock.patch.object(ACQ, "ARTIFACTS", (spec,)),
                mock.patch.object(ACQ, "_validate_artifact_inventory_parity"),
            ):
                ACQ.prepare(output)
                (output / "raw").mkdir()
                (output / "raw" / "unexpected").write_bytes(b"x")
                with self.assertRaisesRegex(ACQ.ReacquisitionError, "unexpected"):
                    ACQ.publish(staging, output)

    def test_staging_root_symlink_rejects(self) -> None:
        payload = b"fixture-payload\n"
        spec = _fixture_spec(payload)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            actual.mkdir()
            (actual / spec.staging_name).write_bytes(payload)
            linked = root / "linked"
            linked.symlink_to(actual, target_is_directory=True)
            with self.assertRaisesRegex(ACQ.ReacquisitionError, "linked"):
                ACQ.verify_staging(linked, (spec,))

    def test_optimized_mode_smoke(self) -> None:
        code = (
            "import importlib.util, pathlib, sys; "
            f"p=pathlib.Path({str(SCRIPT_PATH)!r}); "
            "s=importlib.util.spec_from_file_location('tno_acq_oo',p); "
            "m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; "
            "s.loader.exec_module(m); q=m.build_protocol(); "
            "assert m.validate_protocol(q)==q['content_sha256']; "
            "assert not any(q['claims'].values())"
        )
        completed = subprocess.run(
            [sys.executable, "-OO", "-c", code],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
