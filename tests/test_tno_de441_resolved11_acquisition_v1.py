from __future__ import annotations

import json
import os
from decimal import Decimal
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "tools" / "jx_tno_de441_resolved11_acquisition_v1.py"


def _load_source_module() -> types.ModuleType:
    name = "jx_tno_de441_resolved11_acquisition_v1_tested_source"
    payload = SCRIPT_PATH.read_bytes()
    module = types.ModuleType(name)
    module.__file__ = str(SCRIPT_PATH)
    sys.modules[name] = module
    exec(compile(payload, str(SCRIPT_PATH), "exec"), module.__dict__)
    return module


ACQUISITION = _load_source_module()


def _result_text(body: object) -> str:
    rows = []
    provider_epochs = sorted(
        ACQUISITION.EPOCHS,
        key=lambda epoch: Decimal(epoch.jd_tdb),
    )
    for provider_row_index, epoch in enumerate(provider_epochs):
        epoch_order = ACQUISITION.EPOCHS.index(epoch)
        epoch_token = f"{float(epoch.jd_tdb):.9f}"
        state_tokens = [
            f"{body.naif_id + epoch_order / 100 + component / 1000:.15E}"
            for component in range(6)
        ]
        rows.append(
            ", ".join(
                (
                    epoch_token,
                    f"A.D. 2000-Jan-{provider_row_index + 1:02d} 00:00:00.0000",
                    *state_tokens,
                )
            )
            + ","
        )
    return "\n".join(
        (
            "*******************************************************************************",
            f"Target body name: Synthetic {body.semantic_id} ({body.naif_id}) "
            "{source: DE441}",
            "Center body name: Solar System Barycenter (0)     {source: DE441}",
            "Center-site name: BODY CENTER",
            "Start time      : A.D. 2000-Jan-01 00:00:00.0000 TDB",
            "Stop  time      : A.D. 2000-Jan-12 00:00:00.0000 TDB",
            "Step-size       : DISCRETE TIME-LIST",
            "Output units    : AU-D",
            "Output type     : GEOMETRIC cartesian states",
            "Output format   : 2 (position and velocity)",
            "Reference frame : ICRF",
            "            JDTDB,            Calendar Date (TDB),"
            "                      X,                      Y,                      Z,"
            "                     VX,                     VY,                     VZ,",
            "$$SOE",
            *rows,
            "$$EOE",
            'Barycentric Dynamical Time ("TDB" or T_eph) output was requested.',
            "Geometric state vectors have NO corrections or aberrations applied.",
            "",
        )
    )


def _response(body: object, replacement: tuple[str, str] | None = None) -> bytes:
    result = _result_text(body)
    if replacement is not None:
        result = result.replace(replacement[0], replacement[1], 1)
    return (
        json.dumps(
            {"signature": dict(ACQUISITION.EXPECTED_API_SIGNATURE), "result": result},
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


class FakeFetcher:
    def __init__(self, replacement: tuple[str, str] | None = None) -> None:
        self.urls: list[str] = []
        self.replacement = replacement

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        query = parse_qs(urlparse(url).query)
        naif_id = int(query["COMMAND"][0].strip("'"))
        body = next(body for body in ACQUISITION.BODIES if body.naif_id == naif_id)
        replacement = self.replacement if body == ACQUISITION.BODIES[0] else None
        return _response(body, replacement)


class TnoDe441Resolved11AcquisitionV1Tests(unittest.TestCase):
    def test_exact_query_stage0_binding_and_claim_ceiling(self) -> None:
        expected_common = {
            "format": ["json"],
            "OBJ_DATA": ["'NO'"],
            "MAKE_EPHEM": ["'YES'"],
            "EPHEM_TYPE": ["'VECTORS'"],
            "CENTER": ["'@0'"],
            "TLIST": [",".join(f"'{epoch.jd_tdb}'" for epoch in ACQUISITION.EPOCHS)],
            "TLIST_TYPE": ["'JD'"],
            "TIME_TYPE": ["'TDB'"],
            "REF_PLANE": ["'FRAME'"],
            "REF_SYSTEM": ["'ICRF'"],
            "OUT_UNITS": ["'AU-D'"],
            "VEC_TABLE": ["'2'"],
            "VEC_CORR": ["'NONE'"],
            "CSV_FORMAT": ["'YES'"],
            "VEC_LABELS": ["'YES'"],
            "TIME_DIGITS": ["'FRACSEC'"],
        }
        self.assertEqual(
            [body.naif_id for body in ACQUISITION.BODIES],
            [10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9],
        )
        self.assertEqual(len(ACQUISITION.EPOCHS), 12)
        for body in ACQUISITION.BODIES:
            parsed = urlparse(ACQUISITION.build_query_url(body))
            self.assertEqual(
                f"{parsed.scheme}://{parsed.netloc}{parsed.path}",
                ACQUISITION.HORIZONS_ENDPOINT,
            )
            query = parse_qs(parsed.query)
            self.assertEqual(query.pop("COMMAND"), [f"'{body.naif_id}'"])
            self.assertEqual(query, expected_common)
        protocol = ACQUISITION.build_protocol()
        self.assertEqual(protocol["schema"], ACQUISITION.SCHEMA_PROTOCOL)
        self.assertEqual(protocol["claim_ceiling"], ACQUISITION.CLAIM_CEILING)
        self.assertEqual(
            protocol["stage0_input_binding"]["states_file_sha256"],
            ACQUISITION.STAGE0_STATES_FILE_SHA256,
        )
        self.assertEqual(protocol["query_contract"]["expected_state_row_count"], 132)
        self.assertIs(protocol["claims"]["stage0_epoch_roster_content_bound"], True)
        self.assertIs(protocol["claims"]["de441_provider_responses_retained"], False)
        self.assertIs(protocol["execution_authorized"], False)
        self.assertIs(protocol["outcomes_generated"], False)

    def test_mocked_create_only_acquisition_and_exact_132_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "de441"
            fetcher = FakeFetcher()
            custody = ACQUISITION.acquire(output, fetcher=fetcher)
            self.assertEqual(len(fetcher.urls), 11)
            self.assertEqual(custody, ACQUISITION.validate_output(output))
            self.assertEqual(custody["request_count"], 11)
            self.assertEqual(custody["epoch_count"], 12)
            self.assertEqual(custody["state_record_count"], 132)
            self.assertEqual(
                custody["state_record_order"],
                "EPOCH_MAJOR_THEN_RESOLVED11_BODY_ORDER",
            )
            states = custody["state_records"]
            self.assertEqual(
                [row["naif_id"] for row in states[:11]],
                [body.naif_id for body in ACQUISITION.BODIES],
            )
            self.assertEqual(
                [row["object_id"] for row in states[::11]],
                [epoch.object_id for epoch in ACQUISITION.EPOCHS],
            )
            self.assertEqual([row["state_order"] for row in states], list(range(132)))
            provider_epochs = sorted(
                ACQUISITION.EPOCHS,
                key=lambda epoch: Decimal(epoch.jd_tdb),
            )
            for row in states:
                expected_epoch = ACQUISITION.EPOCHS[row["epoch_order"]]
                self.assertEqual(
                    row["raw_response_row_index"],
                    provider_epochs.index(expected_epoch),
                )
                self.assertEqual(len(row["position_AU_decimal_tokens"]), 3)
                self.assertEqual(len(row["velocity_AU_per_day_decimal_tokens"]), 3)
                self.assertEqual(
                    [float(token).hex() for token in row["position_AU_decimal_tokens"]],
                    row["position_AU_binary64_hex"],
                )
                self.assertEqual(
                    [float(token).hex() for token in row["velocity_AU_per_day_decimal_tokens"]],
                    row["velocity_AU_per_day_binary64_hex"],
                )
            self.assertIs(
                custody["claims"]["exact_132_state_rows_retained_and_schema_validated"],
                True,
            )
            for claim in (
                "provider_transport_authenticated",
                "provider_origin_cryptographically_proven",
                "continuous_custody_proven",
                "de440_de441_sensitivity_gate_executed",
                "dynamics_executed",
                "classification_generated",
                "population_inference",
                "planet_nine_evidence",
                "science_result",
            ):
                self.assertIs(custody["claims"][claim], False)
            expected_raw = {body.filename for body in ACQUISITION.BODIES}
            self.assertEqual({path.name for path in (output / "raw").iterdir()}, expected_raw)
            before = len(fetcher.urls)
            with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "already exists"):
                ACQUISITION.acquire(output, fetcher=fetcher)
            self.assertEqual(len(fetcher.urls), before)

    def test_response_contract_fails_closed_on_header_epoch_row_and_number(self) -> None:
        body = ACQUISITION.BODIES[0]
        sorted_epochs = sorted(
            ACQUISITION.EPOCHS,
            key=lambda epoch: Decimal(epoch.jd_tdb),
        )
        first_token = f"{float(sorted_epochs[0].jd_tdb):.9f}"
        second_token = f"{float(sorted_epochs[1].jd_tdb):.9f}"
        first_row = _result_text(body).split("$$SOE\n", 1)[1].split("\n", 1)[0]
        cases = (
            (("{source: DE441}", "{source: DE440}"), "DE441"),
            ((first_token, "2450000.500000000"), "unexpected epoch"),
            ((second_token, first_token), "duplicate epoch"),
            ((first_row + "\n", ""), "incomplete"),
            (("$$EOE", first_row + "\n$$EOE"), "extra"),
            ((f"{body.naif_id:.15E}", "NaN"), "decimal"),
        )
        for replacement, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ACQUISITION.AcquisitionError, message):
                    ACQUISITION.validate_response(_response(body, replacement), body)

    def test_canonical_type_raw_roster_and_link_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base"
            ACQUISITION.acquire(base, fetcher=FakeFetcher())

            noncanonical = Path(directory) / "noncanonical"
            shutil.copytree(base, noncanonical)
            protocol_path = noncanonical / "PROTOCOL.json"
            value = json.loads(protocol_path.read_bytes())
            protocol_path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "not canonical"):
                ACQUISITION.validate_output(noncanonical)

            wrong_type = Path(directory) / "wrong-type"
            shutil.copytree(base, wrong_type)
            custody_path = wrong_type / "CUSTODY.json"
            custody = json.loads(custody_path.read_bytes())
            custody["state_record_count"] = True
            custody["content_sha256"] = ACQUISITION._content_sha256(
                ACQUISITION.DOMAIN_CUSTODY, custody
            )
            custody_path.write_bytes(ACQUISITION._canonical_json(custody))
            with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "custody differs"):
                ACQUISITION.validate_output(wrong_type)

            raw_tamper = Path(directory) / "raw-tamper"
            shutil.copytree(base, raw_tamper)
            raw_path = raw_tamper / "raw" / ACQUISITION.BODIES[0].filename
            raw_path.write_bytes(raw_path.read_bytes().replace(b"DE441", b"DE440", 1))
            with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "DE441"):
                ACQUISITION.validate_output(raw_tamper)

            extra = Path(directory) / "extra"
            shutil.copytree(base, extra)
            (extra / "unexpected").write_bytes(b"x")
            with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "roster"):
                ACQUISITION.validate_output(extra)

            linked = Path(directory) / "linked"
            shutil.copytree(base, linked)
            linked_raw = linked / "raw" / ACQUISITION.BODIES[0].filename
            original = linked / "raw" / "original.json"
            linked_raw.rename(original)
            os.link(original, linked_raw)
            original.unlink()
            # Keep a second link outside the exact output roster.
            outside_link = Path(directory) / "outside-link.json"
            os.link(linked_raw, outside_link)
            with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "linked"):
                ACQUISITION.validate_output(linked)

    def test_stage0_hash_binding_and_source_only_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "STATES.json"
            changed_value = json.loads(ACQUISITION.STAGE0_STATES_PATH.read_bytes())
            changed_value["status"] = "TAMPERED"
            changed.write_bytes(ACQUISITION._canonical_json(changed_value))
            with mock.patch.object(ACQUISITION, "STAGE0_STATES_PATH", changed):
                with self.assertRaisesRegex(ACQUISITION.AcquisitionError, "states file differs"):
                    ACQUISITION.build_protocol()
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import jxplanetx", source)
        self.assertNotIn("from jxplanetx", source)
        self.assertNotIn("tests.", source)
        self.assertNotIn("assert ", source)
        self.assertNotIn("integrate(", source)

    def test_optimized_protocol_reconstruction_matches_normal(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-OO", str(SCRIPT_PATH), "protocol"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        optimized = json.loads(completed.stdout)
        normal = ACQUISITION.build_protocol()
        self.assertEqual(optimized, normal)
        self.assertEqual(optimized["content_sha256"], normal["content_sha256"])


if __name__ == "__main__":
    unittest.main()
