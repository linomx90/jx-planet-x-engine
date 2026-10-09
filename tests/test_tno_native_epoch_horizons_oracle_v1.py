from __future__ import annotations

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "jx_tno_native_epoch_horizons_oracle_v1.py"
ORACLE_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_horizons_oracle_v1"


def _load_source_module() -> types.ModuleType:
    name = "jx_tno_native_epoch_horizons_oracle_v1_tested_source"
    payload = SCRIPT_PATH.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SCRIPT_PATH)
    sys.modules[name] = module
    exec(compile(payload, str(SCRIPT_PATH), "exec"), module.__dict__)
    return module


ORACLE = _load_source_module()


class TnoNativeEpochHorizonsOracleV1Tests(unittest.TestCase):
    def test_retained_oracle_reconstructs_exactly_and_keeps_claim_ceiling(self) -> None:
        custody = ORACLE.validate_output(ORACLE_ROOT)
        self.assertEqual(custody["schema"], ORACLE.SCHEMA_CUSTODY)
        self.assertEqual(custody["record_count"], 12)
        self.assertEqual(
            [row["spk_id"] for row in custody["records"]],
            [target.spk_id for target in ORACLE.TARGETS],
        )
        self.assertEqual(
            custody["content_sha256"],
            "45dd99dbce486f173516bc14304e03d6129ebb44e404bfe5c3a3b4519c3af978",
        )
        self.assertIs(custody["claims"]["twelve_nominal_native_epoch_vectors_retained"], True)
        for claim in (
            "provider_transport_authenticated",
            "provider_origin_cryptographically_proven",
            "element_to_cartesian_transform_qualified",
            "clone_cartesian_states_materialized",
            "dynamics_executed",
            "classification_generated",
            "observational_likelihood_evaluated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            self.assertIs(custody["claims"][claim], False)
        self.assertIs(custody["execution_authorized"], False)
        self.assertIs(custody["outcomes_generated"], False)

    def test_all_raw_responses_bind_exact_native_epoch_frame_and_vectors(self) -> None:
        custody = ORACLE.validate_output(ORACLE_ROOT)
        for target, record in zip(ORACLE.TARGETS, custody["records"], strict=True):
            payload = (ORACLE_ROOT / record["retained_relative_path"]).read_bytes()
            rebuilt = ORACLE.validate_response(payload, target)
            self.assertEqual(rebuilt, record)
            self.assertEqual(record["center"], "SUN_NAIF_10")
            self.assertEqual(record["planetary_ephemeris"], "DE441")
            self.assertEqual(record["frame"], "HELIOCENTRIC_IAU76_80_ECLIPTIC_J2000")
            self.assertEqual(record["corrections"], "GEOMETRIC_NONE")
            self.assertEqual(
                record["printed_osculating_element_labels"],
                ["e", "q", "tp", "node", "peri", "i"],
            )
            self.assertEqual(len(record["printed_osculating_element_decimal_tokens"]), 6)
            self.assertEqual(
                [float(token).hex() for token in record["printed_osculating_element_decimal_tokens"]],
                record["printed_osculating_element_float_hex"],
            )
            self.assertEqual(len(record["position_AU_decimal_tokens"]), 3)
            self.assertEqual(len(record["velocity_AU_per_day_decimal_tokens"]), 3)
            self.assertEqual(
                [float(token).hex() for token in record["position_AU_decimal_tokens"]],
                record["position_AU_float_hex"],
            )
            self.assertEqual(
                [float(token).hex() for token in record["velocity_AU_per_day_decimal_tokens"]],
                record["velocity_AU_per_day_float_hex"],
            )

    def test_official_horizons_gm_decimal_conversion_is_exactly_bound(self) -> None:
        custody = ORACLE.validate_output(ORACLE_ROOT)
        source = ORACLE_ROOT / custody["gm_source"]["retained_relative_path"]
        payload = source.read_bytes()
        record = ORACLE._gm_record(payload)
        self.assertEqual(record, custody["gm_source"])
        text = payload.decode("ascii")
        self.assertGreaterEqual(text.count("BODY10_GM"), 2)
        with localcontext() as context:
            context.prec = 100
            daily = (
                Decimal("132712440041.27942")
                * Decimal(86400) ** 2
                / Decimal("149597870.7") ** 3
            )
            yearly = daily * Decimal("365.25") ** 2
        self.assertEqual(float(daily).hex(), "0x1.3649541cf2bdcp-12")
        self.assertEqual(float(yearly).hex(), "0x1.3bd0beccac6afp+5")

    def test_printed_horizons_elements_cross_bind_sbdb_solution(self) -> None:
        sbdb_root = PROJECT_ROOT / "runs" / "tno_sbdb_covariance_acquisition_v1"
        sbdb = json.loads((sbdb_root / "CUSTODY.json").read_bytes())
        by_spk = {row["spk_id"]: row for row in sbdb["records"]}
        horizons = ORACLE.validate_output(ORACLE_ROOT)
        labels = ("e", "q", "tp", "node", "peri", "i")
        maximum_tp_delta_seconds = Decimal(0)
        for record in horizons["records"]:
            source_record = by_spk[record["spk_id"]]
            raw = json.loads(
                (sbdb_root / source_record["retained_relative_path"]).read_bytes()
            )
            source_tokens = {
                row["label"]: row["value"]
                for row in raw["orbit"]["covariance"]["elements"]
            }
            printed = record["printed_osculating_element_decimal_tokens"]
            for index in (0, 1, 3, 4, 5):
                self.assertEqual(Decimal(source_tokens[labels[index]]), Decimal(printed[index]))
            source_tp = float(source_tokens["tp"])
            printed_tp = float(printed[2])
            self.assertLessEqual(abs(source_tp - printed_tp), math.ulp(source_tp))
            maximum_tp_delta_seconds = max(
                maximum_tp_delta_seconds,
                abs(Decimal(source_tokens["tp"]) - Decimal(printed[2])) * Decimal(86400),
            )
        self.assertEqual(maximum_tp_delta_seconds, Decimal("0.000018748800"))

    def test_create_only_acquisition_replays_retained_provider_bytes(self) -> None:
        blobs = {
            target.query_url: (ORACLE_ROOT / "raw" / target.filename).read_bytes()
            for target in ORACLE.TARGETS
        }
        blobs[ORACLE.GM_URL] = (ORACLE_ROOT / "source" / "gm_Horizons.pck").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle"
            with mock.patch.object(ORACLE, "_fetch", side_effect=lambda url: blobs[url]):
                custody = ORACLE.acquire(output)
            self.assertEqual(custody, ORACLE.validate_output(output))
            self.assertEqual(custody["content_sha256"], ORACLE.validate_output(ORACLE_ROOT)["content_sha256"])
            with self.assertRaisesRegex(ORACLE.HorizonsOracleError, "already exists"):
                ORACLE.acquire(output)

    def test_raw_and_receipt_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle"
            import shutil

            shutil.copytree(ORACLE_ROOT, output)
            raw = output / "raw" / ORACLE.TARGETS[0].filename
            payload = raw.read_bytes()
            raw.write_bytes(payload.replace(b"4.584520564730028E+01", b"4.684520564730028E+01"))
            with self.assertRaisesRegex(ORACLE.HorizonsOracleError, "custody differs"):
                ORACLE.validate_output(output)

            raw.write_bytes(payload)
            custody_path = output / "CUSTODY.json"
            forged = json.loads(custody_path.read_bytes())
            forged["claims"]["science_result"] = True
            forged["content_sha256"] = ORACLE._content_sha256(ORACLE.DOMAIN_CUSTODY, forged)
            custody_path.write_bytes(ORACLE._canonical_json(forged))
            with self.assertRaisesRegex(ORACLE.HorizonsOracleError, "custody differs"):
                ORACLE.validate_output(output)

    def test_optimized_validation_uses_retained_bytes_without_network(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-OO",
                str(SCRIPT_PATH),
                "validate",
                "--output-root",
                str(ORACLE_ROOT),
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        parsed = json.loads(completed.stdout)
        self.assertEqual(parsed["content_sha256"], ORACLE.validate_output(ORACLE_ROOT)["content_sha256"])


if __name__ == "__main__":
    unittest.main()
