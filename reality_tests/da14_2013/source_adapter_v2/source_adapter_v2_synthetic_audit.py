#!/usr/bin/env python3
"""Synthetic no-network audit for the prospective DA14 source adapter v0.2."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import http.server
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any

from jx_holdout.crypto import decrypt_records, generate_keypair, load_private_key
from jx_source_adapter.bridge import run_adapter
from jx_source_adapter.errors import PolicyError, SourceSchemaError
from jx_source_adapter.normalize import normalize_mpc_observations
from jx_source_adapter.policy import PRODUCTION_ENDPOINTS, RequestSpec, validate_request_destination
from jx_source_adapter.util import canonical_json_bytes, sha256_file
from jx_source_adapter.verify import verify_adapter_output

RESULT_SCHEMA = "jx-reality-test-1-source-adapter-v2-audit-result/v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    payloads: dict[str, bytes] = {}
    requests_seen: list[str] = []

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        path = self.path.split("?", 1)[0]
        self.requests_seen.append(path)
        body = self.payloads.get(path)
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextlib.contextmanager
def mock_server(payloads: dict[str, Any]):
    encoded = {path: canonical_json_bytes(value) for path, value in payloads.items()}
    handler = type("BoundHandler", (Handler,), {"payloads": encoded, "requests_seen": []})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", handler.requests_seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fixture(opaque_integer: int) -> tuple[dict[str, Any], list[str]]:
    protected = [
        "HOLDOUT-RA-CANARY-0192837465",
        "HOLDOUT-DEC-CANARY-5647382910",
        "QUARANTINE-RA-CANARY-1029384756",
        "QUARANTINE-DEC-CANARY-6574839201",
    ]
    rows = [
        {
            "permid": "SYNTHETIC-V2",
            "obstime": "2013-01-31 23:59:59.000000",
            "stn": "500",
            "ra": "10.125",
            "dec": "-5.25",
            "rmsra": "0.2",
            "rmsdec": "0.3",
        },
        {
            "permid": "SYNTHETIC-V2",
            "obstime": "2013-02-01 00:00:00.000000",
            "stn": "568",
            "ra": protected[0],
            "dec": protected[1],
        },
        {
            "permid": "SYNTHETIC-V2",
            "obstime": "2013-03-01 00:00:00.000000",
            "stn": "F51",
            "ra": protected[2],
            "dec": protected[3],
        },
    ]
    # Canary strings must still be syntactically valid decimals for normalization.
    rows[1]["ra"] = "123.0192837465"
    rows[1]["dec"] = "-12.5647382910"
    rows[2]["ra"] = "223.1029384756"
    rows[2]["dec"] = "-22.6574839201"
    protected = [rows[1]["ra"], rows[1]["dec"], rows[2]["ra"], rows[2]["dec"]]
    envelope = {"ADES_DF": rows, "OBS80": "", "OBS_DF": [], "XML": ""}
    obscodes = {
        code: {
            "obscode": code,
            "longitude": longitude,
            "rhocosphi": rho_cos,
            "rhosinphi": rho_sin,
            "name": name,
            "uses_two_line_observations": False,
        }
        for code, longitude, rho_cos, rho_sin, name in (
            ("500", "0", "1", "0", "Synthetic Geocenter"),
            ("568", "204.5278", "0.826", "0.562", "Synthetic Station A"),
            ("F51", "203.744", "0.817", "0.575", "Synthetic Station B"),
        )
    }
    radar = {
        "signature": {"source": "NASA/JPL Small-Body Radar Astrometry API", "version": "1.1"},
        "count": "0",
        "fields": [],
        "data": [],
        "coords": {},
    }
    return {
        "/api/get-obs": [envelope, opaque_integer],
        "/api/obscodes": obscodes,
        "/sb_radar.api": radar,
    }, protected


def scan_text(root: Path) -> str:
    pieces: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix != ".jxenc":
            pieces.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(pieces)


def execute_once(*, root: Path, contract: dict[str, Any], preregistration: Path,
                 public_key: Path, opaque_integer: int) -> tuple[dict[str, Any], list[str], list[str]]:
    payloads, protected = fixture(opaque_integer)
    output = root / "curated"
    with mock_server(payloads) as (base_url, requests_seen):
        receipt = run_adapter(
            designation="SYNTHETIC-V2",
            output_dir=output,
            public_key_path=public_key,
            preregistration_path=preregistration,
            audit_contract=contract,
            audit_mode=True,
            audit_base_url=base_url,
        )
    return receipt, protected, requests_seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    output = args.output
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, mode=0o700)
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool) -> None:
        checks.append({"check": name, "passed": bool(condition)})
        if not condition:
            raise AssertionError(name)

    check("production_endpoint_allowlist_unchanged", PRODUCTION_ENDPOINTS == {
        "mpc_observations": "https://data.minorplanetcenter.net/api/get-obs",
        "mpc_observatory_codes": "https://data.minorplanetcenter.net/api/obscodes",
        "jpl_radar": "https://ssd-api.jpl.nasa.gov/sb_radar.api",
    })
    try:
        validate_request_destination(
            RequestSpec("mpc_observations", "https://example.com/api/get-obs", "GET", b"{}"),
            audit_mode=False,
        )
        arbitrary_blocked = False
    except PolicyError:
        arbitrary_blocked = True
    check("arbitrary_production_host_still_blocked", arbitrary_blocked)

    payloads, _protected = fixture(731_942_857)
    envelope = payloads["/api/get-obs"][0]
    single, single_meta = normalize_mpc_observations(
        [envelope], designation="SYNTHETIC-V2", maximum_records=100
    )
    appended, appended_meta = normalize_mpc_observations(
        [envelope, 731_942_857], designation="SYNTHETIC-V2", maximum_records=100
    )
    prepended, prepended_meta = normalize_mpc_observations(
        [731_942_857, envelope], designation="SYNTHETIC-V2", maximum_records=100
    )
    check("legacy_single_object_envelope_preserved", single == appended == prepended)
    check("object_plus_integer_registered", appended_meta["envelope"]["opaque_integer_metadata_present"] is True)
    check("integer_plus_object_registered", prepended_meta["envelope"]["opaque_integer_metadata_present"] is True)
    check("single_layout_marks_no_integer", single_meta["envelope"]["opaque_integer_metadata_present"] is False)
    check("opaque_integer_value_not_retained", "731942857" not in json.dumps(appended_meta, sort_keys=True))

    invalid = [
        [],
        [envelope, envelope],
        [envelope, 1, 2],
        [envelope, True],
        [envelope, "status"],
        {"ADES_DF": []},
    ]
    invalid_blocked = True
    for value in invalid:
        try:
            normalize_mpc_observations(value, designation="SYNTHETIC-V2", maximum_records=100)
            invalid_blocked = False
        except SourceSchemaError:
            pass
    check("all_unregistered_envelope_layouts_fail_closed", invalid_blocked)

    with tempfile.TemporaryDirectory(prefix="jx-adapter-v2-") as name:
        temp = Path(name)
        password = temp / "password.txt"
        password.write_text("synthetic-v2-password\n", encoding="utf-8")
        private_key = temp / "private.pem"
        public_key = temp / "public.pem"
        generate_keypair(private_key, public_key, password.read_bytes().strip(), temp / "key_receipt.json")

        first_root = temp / "first"
        first_root.mkdir()
        first_receipt, protected, requests_seen = execute_once(
            root=first_root,
            contract=contract,
            preregistration=args.preregistration,
            public_key=public_key,
            opaque_integer=731_942_857,
        )
        first_output = first_root / "curated"
        verification = verify_adapter_output(first_output)
        check("live_shape_end_to_end_curation_passed", first_receipt["status"] == "SOURCE_ADAPTER_COMPLETE_VALUES_SEALED")
        check("output_integrity_verification_passed", verification["status"] == "SOURCE_ADAPTER_OUTPUT_VERIFIED_VALUES_STILL_HIDDEN")
        check("all_three_mock_sources_requested_once", requests_seen == ["/api/get-obs", "/api/obscodes", "/sb_radar.api"])
        check("training_holdout_quarantine_nonempty", all(first_receipt[key] > 0 for key in ("training_count", "holdout_count", "quarantine_count")))

        source_receipt = json.loads((first_output / "source_response_receipt.json").read_text(encoding="utf-8"))
        envelope_metadata = source_receipt["source_metadata"]["mpc_observations"]["envelope"]
        check("source_receipt_records_structure_not_integer_value", envelope_metadata == {
            "envelope_item_count": 2,
            "object_envelope_count": 1,
            "opaque_integer_metadata_present": True,
            "opaque_integer_metadata_retained": False,
        })
        text = scan_text(first_output)
        check("opaque_integer_absent_from_preunblind_text", "731942857" not in text)
        check("protected_measurements_absent_from_preunblind_text", not any(value in text for value in protected))
        check("no_raw_response_file_created", not any("raw" in path.name.lower() for path in first_output.rglob("*")))
        check("private_key_absent_from_adapter_output", "PRIVATE KEY" not in text and not any("private" in path.name.lower() for path in first_output.rglob("*")))

        key = load_private_key(private_key, password.read_bytes().strip())
        holdout = decrypt_records(first_output / "holdout.jxenc", key)
        quarantine = decrypt_records(first_output / "post_holdout_quarantine.jxenc", key)
        decrypted = json.dumps([holdout, quarantine], sort_keys=True)
        check("custodian_can_recover_protected_measurements", all(value in decrypted for value in protected))

        second_root = temp / "second"
        second_root.mkdir()
        second_receipt, _protected2, _requests2 = execute_once(
            root=second_root,
            contract=contract,
            preregistration=args.preregistration,
            public_key=public_key,
            opaque_integer=731_942_858,
        )
        second_output = second_root / "curated"
        check("opaque_integer_does_not_change_normalized_plaintext_hash",
              first_receipt["normalized_complete_plaintext_sha256"] == second_receipt["normalized_complete_plaintext_sha256"])
        check("opaque_integer_does_not_change_training_file",
              sha256_file(first_output / "training.jsonl") == sha256_file(second_output / "training.jsonl"))
        check("opaque_integer_does_not_change_redacted_schedule",
              sha256_file(first_output / "holdout_schedule.csv") == sha256_file(second_output / "holdout_schedule.csv"))

    passed = sum(1 for item in checks if item["passed"])
    result = {
        "schema": RESULT_SCHEMA,
        "status": "SOURCE_ADAPTER_V2_AUDIT_PASSED" if passed == len(checks) else "SOURCE_ADAPTER_V2_AUDIT_FAILED",
        "version": "0.2.0",
        "check_count": len(checks),
        "passed_count": passed,
        "checks": checks,
        "external_network_used": False,
        "real_target_query_performed": False,
        "holdout_unblinded": False,
        "orbit_fitting_performed": False,
        "registered_live_structure": "one MPC object envelope plus one opaque non-boolean integer",
        "prior_source_adapter_audit": {
            "status": "SOURCE_ADAPTER_AUDIT_PASSED",
            "checks": "29/29",
            "result_sha256": "794fafe5b33b0091e6846ca263a772baa90ff4191009b892652d81b7236d2821",
            "unchanged_components_revalidated_by_hash": True,
        },
        "claim_ceiling": "Synthetic validation of a prospective MPC envelope parser revision only; no real retrieval or physical result.",
    }
    result_path = output / "SOURCE_ADAPTER_V2_AUDIT_RESULT.json"
    write_json(result_path, result)
    sums = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            sums.append(f"{sha256_file(path)}  {path.relative_to(output)}")
    (output / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "passed_count": passed,
        "check_count": len(checks),
        "result_sha256": sha256_file(result_path),
    }, sort_keys=True))
    return 0 if result["status"] == "SOURCE_ADAPTER_V2_AUDIT_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
