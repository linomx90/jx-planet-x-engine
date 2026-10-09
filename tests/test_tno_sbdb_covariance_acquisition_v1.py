from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "jx_tno_sbdb_covariance_acquisition_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_sbdb_covariance_acquisition_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load covariance acquisition module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACQ = _load_module()


def _fixture_target():
    return ACQ.Target("jpl-20090377", 20090377, "90377", "FIXTURE")


def _fixture_document(target=None):
    if target is None:
        target = _fixture_target()
    labels = ("e", "q", "tp", "node", "peri", "i")
    names = ("e", "q", "tp", "om", "w", "i")
    units = (None, "au", "TDB", "deg", "deg", "deg")
    sigmas = ("1", "2", "3", "4", "5", "6")
    return {
        "signature": {
            "version": "1.3",
            "source": "NASA/JPL Small-Body Database (SBDB) API",
        },
        "object": {
            "spkid": str(target.spk_id),
            "des": target.primary_designation,
            "fullname": "90377 Sedna (2003 VB12)",
            "orbit_id": "51",
            "orbit_class": {"code": "TNO", "name": "TransNeptunian Object"},
        },
        "orbit": {
            "orbit_id": "51",
            "soln_date": "2026-06-06 14:12:24",
            "equinox": "J2000",
            "source": "JPL",
            "pe_used": "DE441",
            "epoch": "2461200.5",
            "n_obs_used": 671,
            "data_arc": "12939",
            "covariance": {
                "epoch": "2458411.5",
                "labels": list(labels),
                "elements": [
                    {
                        "label": label,
                        "name": name,
                        "units": unit,
                        "value": str(index + 1),
                        "sigma": sigmas[index],
                    }
                    for index, (label, name, unit) in enumerate(
                        zip(labels, names, units)
                    )
                ],
                "data": [
                    [str((row + 1) ** 2) if row == column else "0" for column in range(6)]
                    for row in range(6)
                ],
            },
        },
    }


def _fixture_payload(target=None):
    return json.dumps(
        _fixture_document(target),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")


class TnoSbdbCovarianceAcquisitionV1Tests(unittest.TestCase):
    def test_protocol_freezes_exact_twelve_object_input_only_scope(self) -> None:
        protocol = ACQ.build_protocol()
        self.assertEqual(protocol["schema"], ACQ.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["classification"], ACQ.CLASSIFICATION)
        self.assertEqual(len(protocol["target_roster"]), 12)
        self.assertEqual(
            [row["spk_id"] for row in protocol["target_roster"]],
            [target.spk_id for target in ACQ.TARGETS],
        )
        self.assertEqual(
            ACQ.validate_protocol(protocol), protocol["content_sha256"]
        )
        self.assertIs(protocol["execution_authorized"], False)
        self.assertIs(protocol["outcomes_generated"], False)
        self.assertIs(protocol["claims"]["covariance_sampling_authorized"], False)
        self.assertIs(protocol["claims"]["science_result"], False)
        self.assertIs(
            protocol["claims"]["sbdb_schema_conformant_raw_covariance_bytes_retained"],
            True,
        )
        self.assertIs(protocol["claims"]["provider_transport_authenticated"], False)
        self.assertEqual(
            protocol["implementation"]["file_sha256"],
            ACQ._sha256_file(SCRIPT_PATH),
        )
        changed = json.loads(json.dumps(protocol))
        changed["claims"]["science_result"] = True
        with self.assertRaisesRegex(ACQ.CovarianceAcquisitionError, "differs"):
            ACQ.validate_protocol(changed)

    def test_valid_response_seals_matrix_mean_and_native_epoch(self) -> None:
        target = _fixture_target()
        record = ACQ.validate_response(_fixture_payload(target), target)
        self.assertEqual(record["spk_id"], target.spk_id)
        self.assertEqual(record["covariance_epoch_jd_tdb"], "2458411.5")
        self.assertEqual(record["standard_epoch_jd_tdb"], "2461200.5")
        self.assertEqual(record["covariance_labels"], list(ACQ.EXPECTED_LABELS))
        self.assertTrue(record["covariance_exactly_symmetric"])
        self.assertTrue(record["covariance_positive_definite"])
        self.assertEqual(len(record["covariance_matrix_sha256"]), 64)
        self.assertEqual(len(record["covariance_mean_sha256"]), 64)

    def test_response_rejects_identity_semantics_symmetry_and_indefiniteness(self) -> None:
        target = _fixture_target()
        wrong = _fixture_document(target)
        wrong["object"]["spkid"] = "1"
        with self.assertRaisesRegex(ACQ.CovarianceAcquisitionError, "SPK ID"):
            ACQ.validate_response(
                json.dumps(wrong, separators=(",", ":")).encode("ascii"), target
            )

        wrong = _fixture_document(target)
        wrong["orbit"]["covariance"]["labels"][0] = "a"
        with self.assertRaisesRegex(ACQ.CovarianceAcquisitionError, "labels"):
            ACQ.validate_response(
                json.dumps(wrong, separators=(",", ":")).encode("ascii"), target
            )

        wrong = _fixture_document(target)
        wrong["orbit"]["covariance"]["data"][0][1] = "1"
        with self.assertRaisesRegex(ACQ.CovarianceAcquisitionError, "symmetric"):
            ACQ.validate_response(
                json.dumps(wrong, separators=(",", ":")).encode("ascii"), target
            )

        wrong = _fixture_document(target)
        wrong["orbit"]["covariance"]["data"][0][0] = "-1"
        with self.assertRaisesRegex(ACQ.CovarianceAcquisitionError, "diagonal"):
            ACQ.validate_response(
                json.dumps(wrong, separators=(",", ":")).encode("ascii"), target
            )

        wrong = _fixture_document(target)
        wrong["orbit"]["covariance"]["elements"][0]["sigma"] = "999999"
        with self.assertRaisesRegex(
            ACQ.CovarianceAcquisitionError, "sigmas disagree"
        ):
            ACQ.validate_response(
                json.dumps(wrong, separators=(",", ":")).encode("ascii"), target
            )

    def test_prepare_is_idempotent_and_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            first = ACQ.prepare(output)
            second = ACQ.prepare(output)
            self.assertEqual(first, second)
            path = output / "PROTOCOL.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaisesRegex(
                ACQ.CovarianceAcquisitionError, "different protocol"
            ):
                ACQ.prepare(output)

    def test_fixture_publish_validate_and_tamper_fail_closed(self) -> None:
        target = _fixture_target()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            output = root / "output"
            staging.mkdir()
            (staging / target.filename).write_bytes(_fixture_payload(target))
            with mock.patch.object(ACQ, "TARGETS", (target,)):
                custody = ACQ.publish(staging, output)
                self.assertEqual(custody["record_count"], 1)
                self.assertIs(custody["execution_authorized"], False)
                self.assertIs(custody["claims"]["science_result"], False)
                self.assertEqual(
                    ACQ.validate_output(output)["content_sha256"],
                    custody["content_sha256"],
                )
                with self.assertRaisesRegex(
                    ACQ.CovarianceAcquisitionError, "already exists"
                ):
                    ACQ.publish(staging, output)

                raw = output / "raw" / target.filename
                original = raw.read_bytes()
                raw.write_bytes(original.replace(b'"2458411.5"', b'"2458412.5"'))
                with self.assertRaises(ACQ.CovarianceAcquisitionError):
                    ACQ.validate_output(output)
                raw.write_bytes(original)

                receipt = output / "CUSTODY.json"
                forged = json.loads(receipt.read_bytes())
                forged["claims"]["science_result"] = True
                forged["content_sha256"] = ACQ._content_sha256(
                    ACQ.CUSTODY_DOMAIN, forged
                )
                receipt.write_bytes(ACQ._canonical_json(forged))
                with self.assertRaisesRegex(
                    ACQ.CovarianceAcquisitionError, "reconstruction"
                ):
                    ACQ.validate_output(output)

    def test_staging_roster_and_symlinks_fail_closed(self) -> None:
        target = _fixture_target()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir()
            (staging / target.filename).write_bytes(_fixture_payload(target))
            (staging / "extra.json").write_bytes(b"{}")
            with mock.patch.object(ACQ, "TARGETS", (target,)):
                with self.assertRaisesRegex(
                    ACQ.CovarianceAcquisitionError, "roster"
                ):
                    ACQ.verify_staging(staging)
                (staging / "extra.json").unlink()
                payload = (staging / target.filename).read_bytes()
                (staging / target.filename).unlink()
                source = root / "source.json"
                source.write_bytes(payload)
                (staging / target.filename).symlink_to(source)
                with self.assertRaisesRegex(
                    ACQ.CovarianceAcquisitionError, "linked"
                ):
                    ACQ.verify_staging(staging)

    def test_cli_prepare_and_optimized_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    "prepare",
                    "--output-root",
                    str(output),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "PROTOCOL.json").is_file())
            optimized = subprocess.run(
                [sys.executable, "-OO", "-c", "import runpy; runpy.run_path(%r)" % str(SCRIPT_PATH)],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(optimized.returncode, 0, optimized.stderr)


if __name__ == "__main__":
    unittest.main()
