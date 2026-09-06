"""Focused tests for the opt-in M4C2B isolated CSPICE execution boundary."""

from __future__ import annotations

import base64
import contextlib
import csv
import dataclasses
import hashlib
import importlib.util
import inspect
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
import zipfile

from jxplanetx import solar_system
from jxplanetx.solar_system import _cspice_worker as worker_module
from jxplanetx.solar_system import cspice_execution as module
from jxplanetx.solar_system import cspice_execution_contracts as evidence
from jxplanetx.solar_system.contracts import (
    ArtifactBinding,
    CoverageInterval,
    SolarSystemContractError,
    SolarSystemCoverageError,
    SolarSystemDataError,
    validate_integrity,
)
from jxplanetx.solar_system.cspice_execution import execute_cspice_spkgeo_replay
from jxplanetx.solar_system.cspice_execution_contracts import (
    CspiceSpkgeoExecutionReceipt,
    validate_cspice_spkgeo_execution_receipt,
)
from jxplanetx.solar_system.cspice_time import project_cspice_binary64_query


_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "_m4c2a_fixture_module",
    Path(__file__).with_name("test_solar_system_cspice_execution_contracts.py"),
)
if _FIXTURE_SPEC is None or _FIXTURE_SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("frozen M4C2A fixture module is unavailable")
fixtures = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(fixtures)


INTEGRITY = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
COMPOSITE_ID = "artifact.cspice.license"
RULES_ID = "artifact.cspice.naif-rules-license"
NUMPY_LICENSE_ID = "artifact.cspice.numpy-license"
COMPOSITE_SPDX = "MIT AND LicenseRef-NAIF-SPICE-Rules"
RULES_SPDX = "LicenseRef-NAIF-SPICE-Rules"
NUMPY_SPDX = "LicenseRef-NumPy-2.3.5-Wheel-License-Manifest"
EXPECTED_ET0_STATE_SHA256 = (
    "392d2d7be8e9e9e35e87141df7a7d327fd7e733624d7e8256423e95dc30db274"
)
EXPECTED_ET1_STATE_SHA256 = (
    "126bc0ae26aa4063864c80c7dc2b01941fa2ad5fd51ec7cb6dd8ea2b4bdd5281"
)
EXPECTED_ET0_ET1_STATE_SHA256 = (
    "bb8c254ad25f72e4f9df1a3e5415778ed661fc6bdf1644cb40de38432f5813a1"
)
EXPECTED_ET0_LIGHT_SHA256 = (
    "9afa93bb208751f4b509b7f8e838d2939d6a7eb5e8ec86ad14a92abdef8134d1"
)
EXPECTED_ET1_LIGHT_SHA256 = (
    "0d6ce7306258de45c5d649e439ce809271b2bf120320f957797627a33713a554"
)
EXPECTED_ET0_CHAIN_PREIMAGE_LENGTH = 4_211
EXPECTED_ET0_CHAIN_SHA256 = (
    "4c21e7e807fabd1e5a72b37b8fbebb1e85788c6669604ae0d3ee308696e5fdec"
)
EXPECTED_ET1_CHAIN_PREIMAGE_LENGTH = 4_271
EXPECTED_ET1_CHAIN_SHA256 = (
    "26cff7162bb3998ec3d43e8778e6674b17af07d35b2b8d77f27364acf9cf3d98"
)
EXPECTED_ENDPOINT_STATE_SHA256 = (
    "112fb534fc40ac4f26159708ac571c1006620c96291b5ba504117ee1b1e1063d"
)
ENDPOINT_CASES = (
    (
        "lower-endpoint",
        -4_734_072_000.0,
        "6727d655daf2db6a53d312356109fadfd52305038f21b175d700ac4d8ca7c468",
        "1f9225467dd7c4187358da4be9ab6be2643fa821452852e52ff1d321d9d4028f",
    ),
    (
        "lower-inside",
        math.nextafter(-4_734_072_000.0, math.inf),
        "33f2adabc590650a2393929e4846eab9d16141f8726ffe33ffa315df7bc01078",
        "640a8be940a8554262a0e06d6558c5573efdcd85b47a2765d1b943b375a61d4d",
    ),
    (
        "upper-inside",
        math.nextafter(4_735_368_000.0, -math.inf),
        "6259e5591cc66a7b0b9d55b2b0b6339e41d766f40d062813844111a9cef6c0af",
        "acc8d3b2593cf0c3d6bc893102022aff81421f57f3c245165db8cf9c4a18e445",
    ),
    (
        "upper-endpoint",
        4_735_368_000.0,
        "00437b95aaff141c320b7df9165898937d9a3bdeb9572f66142bc174663c69f6",
        "6713967dd31af5cef641b67e1842decf5afe9aa8600a1321410a4edfdd83e49c",
    ),
)


def _binding(
    artifact_id: str,
    role: str,
    version: str,
    locator: str,
    length: int,
    digest: str,
    media_type: str,
    spdx: str,
    license_id: str | None,
    *,
    coverage: tuple[CoverageInterval, ...] | None = None,
    load_order: int | None = None,
) -> ArtifactBinding:
    return ArtifactBinding(
        artifact_id=artifact_id,
        artifact_role=role,
        provider_id=fixtures.PROVIDER_ID,
        version=version,
        logical_locator=locator,
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=length,
        artifact_sha256=digest,
        media_type=media_type,
        coverage_status=(
            "COARSE_ARTIFACT_TIME_ENVELOPE" if coverage is not None else "NOT_APPLICABLE"
        ),
        coverage=coverage,
        license_evidence_status=(
            "RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING"
            if role == "LICENSE"
            else "RETAINED_HASH_BOUND_LICENSE_ARTIFACT"
        ),
        license_spdx=spdx,
        license_artifact_id=None if role == "LICENSE" else license_id,
        redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
        load_order_status=("ORDERED_LOAD_MEMBER" if load_order is not None else "NOT_LOADABLE"),
        load_order=load_order,
        extrapolation_policy="FORBID",
        content_integrity_class=INTEGRITY,
    )


def production_provider(abi: str):
    profile = evidence._ABI_PROFILES[abi]
    coverage = (
        CoverageInterval(
            fixtures.epoch(-4_734_072_000, 0.0),
            fixtures.epoch(4_735_368_000, 0.0),
            "CLOSED_CLOSED",
        ),
    )
    artifacts = (
        _binding(
            module._SPK_ID,
            "SPK",
            "de440s",
            "kernels/de440s.bsp",
            module._SPK_LENGTH,
            module._SPK_SHA256,
            "application/octet-stream",
            RULES_SPDX,
            RULES_ID,
            coverage=coverage,
            load_order=0,
        ),
        _binding(
            COMPOSITE_ID,
            "LICENSE",
            "SpiceyPy-8.2.0-bundled-N0067-derived-patched-composite-license-provenance.v1",
            "licenses/spiceypy-cspice-composite-license-provenance.v1.json",
            module._COMPOSITE_LENGTH,
            module._COMPOSITE_SHA256,
            "application/json",
            COMPOSITE_SPDX,
            None,
        ),
        _binding(
            RULES_ID,
            "LICENSE",
            "NAIF-SPICE-Rules-ae85f851.v1",
            "licenses/naif-spice-rules.html",
            module._RULES_LENGTH,
            module._RULES_SHA256,
            "text/html",
            RULES_SPDX,
            None,
        ),
        _binding(
            module._NATIVE_ID,
            "NATIVE_LIBRARY",
            fixtures.CSPICE_ARTIFACT_VERSION,
            "native/libcspice.so",
            module._NATIVE_LENGTH,
            module._NATIVE_SHA256,
            "application/x-sharedlib",
            RULES_SPDX,
            RULES_ID,
        ),
        _binding(
            NUMPY_LICENSE_ID,
            "LICENSE",
            "NumPy-2.3.5-wheel-license-manifest-2046a313.v1",
            "licenses/numpy-2.3.5-LICENSE.txt",
            module._NUMPY_LICENSE_LENGTH,
            module._NUMPY_LICENSE_SHA256,
            "text/plain",
            NUMPY_SPDX,
            None,
        ),
        _binding(
            module._NUMPY_ID,
            "SOFTWARE_DISTRIBUTION",
            "2.3.5",
            f"wheels/numpy-2.3.5-{abi.split('-')[0]}-{abi.split('-')[1]}-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
            profile["numpy_wheel_length"],
            profile["numpy_wheel_sha256"],
            "application/zip",
            NUMPY_SPDX,
            NUMPY_LICENSE_ID,
        ),
        _binding(
            module._SPICEYPY_ID,
            "SOFTWARE_DISTRIBUTION",
            "8.2.0",
            f"wheels/spiceypy-8.2.0-{abi.split('-')[0]}-{abi.split('-')[1]}-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl",
            profile["spiceypy_wheel_length"],
            profile["spiceypy_wheel_sha256"],
            "application/zip",
            COMPOSITE_SPDX,
            COMPOSITE_ID,
        ),
    )
    base, _ = fixtures.context(abi)
    production_frame = dataclasses.replace(
        base.native_frame,
        coverage=coverage,
        content_sha256="",
    )
    provider = dataclasses.replace(
        base,
        artifacts=tuple(sorted(artifacts, key=lambda item: item.artifact_id)),
        native_frame=production_frame,
        content_sha256="",
    )
    validate_integrity(provider)
    return provider


def projection_at(
    abi: str,
    *,
    targets: tuple[int, ...] = (10, 3, 399, 301, 5),
    whole: int = 0,
    fraction: float = 0.0,
):
    provider = production_provider(abi)
    _, query = fixtures.context(abi)
    requested = dataclasses.replace(
        query,
        query_id=f"query.cspice.requested.m4c2b.{whole}",
        provider=provider,
        epoch=fixtures.epoch(whole, fraction),
        target_naif_ids=targets,
        frame=provider.native_frame,
        output_unit_system=provider.native_output_unit_system,
        content_sha256="",
    )
    return project_cspice_binary64_query(
        requested,
        effective_query_id=f"query.cspice.effective.m4c2b.{whole}",
    )


def worker_result(
    abi: str,
    *,
    targets: tuple[int, ...] = (10,),
    worker_length: int = module._EXPECTED_WORKER_BYTE_LENGTH,
    worker_sha: str = module._EXPECTED_WORKER_SHA256,
    python_length: int = 7_000_000,
    python_sha: str = "a" * 64,
) -> dict[str, object]:
    profile = evidence._ABI_PROFILES[abi]
    zero = "0000000000000000"
    one = struct.pack(">d", 1.0).hex()
    rows = []
    for target in targets:
        rows.append(
            {
                "legs": [
                    {
                        "begin_daf_address": 100 + target,
                        "center_naif_id": 0,
                        "end_daf_address": 199 + target,
                        "first_et_be_hex": struct.pack(">d", -100.0).hex(),
                        "frame_naif_id": 1,
                        "last_et_be_hex": struct.pack(">d", 100.0).hex(),
                        "segment_identifier": f"SEGMENT-{target}",
                        "segment_type": 2,
                        "selected_body_naif_id": target,
                        "spkpvn_returned_center_naif_id": 0,
                        "spkpvn_returned_frame_naif_id": 1,
                        "spkpvn_state_be_hex": [one, zero, zero, zero, zero, zero],
                    }
                ],
                "spkgeo_light_time_be_hex": zero,
                "spkgeo_state_be_hex": [one, zero, zero, zero, zero, zero],
                "target_naif_id": target,
            }
        )
    return {
        "abi_tag": abi,
        "after_clear_pool_counts": [0, 0],
        "cspice_toolkit_version": "CSPICE_N0067",
        "effective_et_be_hex": zero,
        "fegetround_after": 0,
        "fegetround_before": 0,
        "fresh_pool_counts": [0, 0],
        "host_native_dependencies": [
            ["ld-linux-x86-64.so.2", 1, "b" * 64],
            ["libc.so.6", 1, "c" * 64],
            ["libm.so.6", 1, "d" * 64],
        ],
        "libc_name": "glibc",
        "libc_version": "2.fixture",
        "loaded_distribution_native_files": [
            ["spiceypy/utils/libcspice.so", module._NATIVE_LENGTH, module._NATIVE_SHA256],
            *[[path, length, digest] for path, length, digest in profile["loaded_numpy_natives"]],
        ],
        "loaded_handle": 1,
        "loaded_pool_counts": [1, 1],
        "nonce": "1" * 64,
        "numpy_cpu_feature_payload_byte_length": 1_086,
        "numpy_cpu_feature_payload_sha256": "b050bd98f1d010d3f10c7d685e2eb6875793ddb9909110de8474aba96740fe74",
        "numpy_native_tree": list(profile["numpy_native_tree"]),
        "numpy_record_sha256": profile["numpy_record_sha256"],
        "numpy_tree": list(profile["numpy_tree"]),
        "post_query_pool_counts": [1, 1],
        "python_executable_byte_length": python_length,
        "python_executable_sha256": python_sha,
        "python_implementation": "CPython",
        "python_version": profile["python_version"],
        "request_sha256": "2" * 64,
        "role": "PRIMARY",
        "schema": module._RESULT_SCHEMA,
        "spiceypy_record_sha256": profile["spiceypy_record_sha256"],
        "spiceypy_tree": list(profile["spiceypy_tree"]),
        "spk_byte_length": module._SPK_LENGTH,
        "spk_offset_after": 0,
        "spk_offset_before": 0,
        "spk_sha256": module._SPK_SHA256,
        "targets": rows,
        "worker_source_byte_length": worker_length,
        "worker_source_sha256": worker_sha,
    }


def validate_worker_fixture(result: dict[str, object], abi: str, targets: tuple[int, ...]) -> None:
    module._validate_worker_result(
        result,
        abi=abi,
        role="PRIMARY",
        nonce="1" * 64,
        request_sha256="2" * 64,
        python_executable_byte_length=7_000_000,
        python_executable_sha256="a" * 64,
        worker_source_byte_length=module._EXPECTED_WORKER_BYTE_LENGTH,
        worker_sha256=module._EXPECTED_WORKER_SHA256,
        effective_et_bits="0000000000000000",
        targets=targets,
    )


def framed(value: object) -> bytes:
    encoded = module._canonical_json(value)
    return struct.pack(">I", len(encoded)) + encoded


def state_payload(lane) -> bytes:
    payload = bytearray()
    for index in range(len(lane.state_batch.target_naif_ids)):
        offset = index * 3
        payload.extend(
            struct.pack(
                ">6d",
                *lane.state_batch.positions[offset : offset + 3],
                *lane.state_batch.velocities[offset : offset + 3],
            )
        )
    return bytes(payload)


def light_time_payload(lane) -> bytes:
    return b"".join(
        struct.pack(">d", chain.spkgeo_light_time_seconds)
        for chain in lane.target_chains
    )


def chain_evidence_preimage(lane) -> bytes:
    value = []
    for chain in lane.target_chains:
        value.append(
            {
                "effective_et": chain.effective_et.hex(),
                "legs": [
                    {
                        "begin": leg.begin_daf_address,
                        "body": leg.selected_body_naif_id,
                        "center": leg.center_naif_id,
                        "end": leg.end_daf_address,
                        "first": leg.first_et.hex(),
                        "frame": leg.frame_naif_id,
                        "id": leg.segment_identifier,
                        "kernel": leg.kernel_artifact_sha256,
                        "last": leg.last_et.hex(),
                        "returned_center": leg.spkpvn_returned_center_naif_id,
                        "returned_frame": leg.spkpvn_returned_frame_naif_id,
                        "state": [component.hex() for component in leg.spkpvn_state],
                        "type": leg.segment_type,
                    }
                    for leg in chain.legs
                ],
                "lt": chain.spkgeo_light_time_seconds.hex(),
                "observer": chain.observer_naif_id,
                "state": [component.hex() for component in chain.spkgeo_state],
                "target": chain.target_naif_id,
            }
        )
    return module._canonical_json(value)


class PublicSurfaceAndPreflightTests(unittest.TestCase):
    def test_exact_public_surface_signature_and_root_nonpublication(self) -> None:
        self.assertEqual(module.__all__, ["execute_cspice_spkgeo_replay"])
        self.assertEqual(solar_system.__all__, [])
        signature = inspect.signature(execute_cspice_spkgeo_replay)
        self.assertEqual(
            tuple(signature.parameters),
            (
                "projection",
                "artifact_root_directory_fd",
                "receipt_id",
                "primary_batch_id",
                "replay_batch_id",
                "primary_execution_instance_nonce",
                "replay_execution_instance_nonce",
            ),
        )
        self.assertEqual(signature.parameters["projection"].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        for name in tuple(signature.parameters)[1:]:
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_parent_import_wall_and_private_worker_pin(self) -> None:
        source_root = str(Path(__file__).parents[1] / "src")
        code = f'''
import json,sys
sys.path.insert(0,{source_root!r})
before=set(sys.modules)
from jxplanetx import solar_system
from jxplanetx.solar_system import cspice_execution as m
blocked=('numpy','spiceypy','astropy','erfa','jplephem','spiceypy.cyice')
print(json.dumps({{'exports':m.__all__,'root':solar_system.__all__,'loaded':[x for x in blocked if x in sys.modules and x not in before]}},sort_keys=True))
'''
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-c", code],
            cwd=Path(__file__).parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
        observed = json.loads(completed.stdout)
        self.assertEqual(observed["loaded"], [])
        self.assertEqual(observed["exports"], ["execute_cspice_spkgeo_replay"])
        self.assertEqual(observed["root"], [])
        worker = Path(module.__file__).with_name("_cspice_worker.py").read_bytes()
        self.assertEqual(len(worker), module._EXPECTED_WORKER_BYTE_LENGTH)
        self.assertEqual(hashlib.sha256(worker).hexdigest(), module._EXPECTED_WORKER_SHA256)

    def test_exact_both_abi_provider_profiles_and_projection(self) -> None:
        for abi, expected in module._PROVIDER_SEALS.items():
            with self.subTest(abi=abi):
                provider = production_provider(abi)
                self.assertEqual(provider.content_sha256, expected)
                self.assertEqual(
                    tuple(item.content_sha256 for item in provider.artifacts),
                    tuple(module._ARTIFACT_SEALS[abi][item.artifact_id] for item in provider.artifacts),
                )
                projected = projection_at(abi)
                self.assertEqual(projected.binary64_projection.rounded_value.hex(), "0x0.0p+0")
                self.assertEqual(
                    tuple(
                        (interval.start.whole, interval.end.whole)
                        for interval in provider.native_frame.coverage
                    ),
                    ((-4_734_072_000, 4_735_368_000),),
                )
                spk = next(item for item in provider.artifacts if item.artifact_id == module._SPK_ID)
                self.assertEqual(provider.native_frame.coverage, spk.coverage)
        for outside in (
            math.nextafter(-4_734_072_000.0, -math.inf),
            math.nextafter(4_735_368_000.0, math.inf),
        ):
            whole = math.floor(outside + 0.5)
            with self.assertRaises(SolarSystemCoverageError):
                projection_at(module._current_abi(), whole=whole, fraction=outside - whole)

    def test_literal_real_kernel_kat_roster_is_locked(self) -> None:
        self.assertEqual(EXPECTED_ET0_STATE_SHA256, "392d2d7be8e9e9e35e87141df7a7d327fd7e733624d7e8256423e95dc30db274")
        self.assertEqual(EXPECTED_ET1_STATE_SHA256, "126bc0ae26aa4063864c80c7dc2b01941fa2ad5fd51ec7cb6dd8ea2b4bdd5281")
        self.assertEqual(EXPECTED_ET0_ET1_STATE_SHA256, "bb8c254ad25f72e4f9df1a3e5415778ed661fc6bdf1644cb40de38432f5813a1")
        self.assertEqual(EXPECTED_ENDPOINT_STATE_SHA256, "112fb534fc40ac4f26159708ac571c1006620c96291b5ba504117ee1b1e1063d")
        self.assertEqual(tuple(item[0] for item in ENDPOINT_CASES), ("lower-endpoint", "lower-inside", "upper-inside", "upper-endpoint"))

    def test_scalar_and_roster_failures_precede_os(self) -> None:
        abi = module._current_abi()
        projection = projection_at(abi)
        with mock.patch.object(module.os, "dup", side_effect=AssertionError("OS_REACHED")):
            with self.assertRaises(SolarSystemContractError):
                execute_cspice_spkgeo_replay(
                    projection,
                    artifact_root_directory_fd=0,
                    receipt_id="x" * 257,
                    primary_batch_id="p",
                    replay_batch_id="r",
                    primary_execution_instance_nonce="1" * 64,
                    replay_execution_instance_nonce="2" * 64,
                )
        bad_provider = dataclasses.replace(
            projection.requested_query.provider,
            spec_id="provider-spec.relabelled",
            content_sha256="",
        )
        bad_requested = dataclasses.replace(
            projection.requested_query,
            provider=bad_provider,
            frame=bad_provider.native_frame,
            output_unit_system=bad_provider.native_output_unit_system,
            content_sha256="",
        )
        bad_projection = project_cspice_binary64_query(
            bad_requested,
            effective_query_id="query.bad-profile.effective",
        )
        with mock.patch.object(module.os, "dup", side_effect=AssertionError("OS_REACHED")):
            with self.assertRaises(SolarSystemContractError):
                execute_cspice_spkgeo_replay(
                    bad_projection,
                    artifact_root_directory_fd=0,
                    receipt_id="receipt",
                    primary_batch_id="p",
                    replay_batch_id="r",
                    primary_execution_instance_nonce="1" * 64,
                    replay_execution_instance_nonce="2" * 64,
                )

    def test_nonclaims_are_explicit(self) -> None:
        prose = " ".join((module.__doc__ or "").split())
        for phrase in (
            "neither process identity nor process independence",
            "network sandbox",
            "mapped-page or continuous byte custody",
            "physical-accuracy qualification",
            "legal/redistribution authority",
            "do not bound latency",
            "same UID",
            "standard-library path custody",
        ):
            self.assertIn(phrase, prose)


class ProtocolAndRawEvidenceTests(unittest.TestCase):
    def test_worker_result_fixture_is_exact_and_parent_cross_bound(self) -> None:
        abi = module._current_abi()
        result = worker_result(abi)
        validate_worker_fixture(result, abi, (10,))
        for key, changed in (
            ("python_executable_sha256", "e" * 64),
            ("worker_source_byte_length", module._EXPECTED_WORKER_BYTE_LENGTH + 1),
            ("numpy_cpu_feature_payload_byte_length", True),
            ("spk_offset_before", False),
            ("fegetround_after", False),
        ):
            with self.subTest(key=key):
                mutated = dict(result)
                mutated[key] = changed
                with self.assertRaises(SolarSystemDataError):
                    validate_worker_fixture(mutated, abi, (10,))

    def test_bool_target_frame_type_and_readback_integers_reject(self) -> None:
        abi = module._current_abi()
        for path in ("target", "frame", "type", "returned_frame", "returned_center"):
            with self.subTest(path=path):
                result = worker_result(abi, targets=(1,))
                row = result["targets"][0]
                leg = row["legs"][0]
                if path == "target":
                    row["target_naif_id"] = True
                elif path == "frame":
                    leg["frame_naif_id"] = True
                elif path == "type":
                    leg["segment_type"] = True
                elif path == "returned_frame":
                    leg["spkpvn_returned_frame_naif_id"] = True
                else:
                    leg["spkpvn_returned_center_naif_id"] = False
                with self.assertRaises(SolarSystemDataError):
                    validate_worker_fixture(result, abi, (1,))

    def test_success_transcript_coverage_inconsistency_is_data(self) -> None:
        abi = module._current_abi()
        result = worker_result(abi)
        result["targets"][0]["legs"][0]["first_et_be_hex"] = struct.pack(">d", 1.0).hex()
        with self.assertRaises(SolarSystemDataError) as caught:
            validate_worker_fixture(result, abi, (10,))
        self.assertNotIsInstance(caught.exception, SolarSystemCoverageError)

    def test_global_leg_cap_precedes_nested_record_traversal(self) -> None:
        abi = module._current_abi()
        targets = tuple(range(1, 65))
        result = worker_result(abi, targets=targets)
        bomb = object()
        for row in result["targets"]:
            row["legs"] = [bomb] * 5
        with self.assertRaises(SolarSystemDataError) as caught:
            validate_worker_fixture(result, abi, targets)
        self.assertIn("global cap", str(caught.exception))

    def test_response_error_kind_is_bound_to_exact_exit_status(self) -> None:
        failure = {
            "error_kind": "COVERAGE",
            "error_status": "FAIL_CLOSED",
            "schema": module._RESULT_SCHEMA,
            "status": "ERROR",
        }
        with self.assertRaises(SolarSystemCoverageError):
            module._parse_response(framed(failure), b"", 70)
        for code in (0, 1, 69, 71):
            with self.subTest(code=code), self.assertRaises(SolarSystemDataError):
                module._parse_response(framed(failure), b"", code)

    def test_response_rejects_duplicates_noncanonical_numbers_and_trailing_data(self) -> None:
        duplicates = b'{"status":"OK","status":"OK"}'
        cases = (
            struct.pack(">I", len(duplicates)) + duplicates,
            framed({"result": {}, "status": "OK"}).replace(b"{}", b"{ }", 1),
            framed({"result": {"value": 1.0}, "status": "OK"}),
            framed({"result": {"value": 1 << 80}, "status": "OK"}),
            framed({"result": {}, "status": "OK"}) + b"x",
        )
        for index, value in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(SolarSystemDataError):
                module._parse_response(value, b"", 0)


class StageAndProcessBoundaryTests(unittest.TestCase):
    def test_private_stage_cleanup_is_fd_stable_and_does_not_recreate_parents(self) -> None:
        before = len(os.listdir("/proc/self/fd"))
        names = []
        for _ in range(3):
            with module._Stage() as stage:
                names.append(stage.name)
                fd = stage.create_file("nested/value.bin")
                os.write(fd, b"value")
                os.close(fd)
        self.assertEqual(len(os.listdir("/proc/self/fd")), before)
        for name in names:
            self.assertFalse(os.path.lexists(os.path.join("/tmp", name)))

    def test_secure_open_rejects_symlink_and_closes_on_fstat_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "value").write_bytes(b"x")
            (root / "link").symlink_to("value")
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with self.assertRaises(OSError):
                    module._open_relative(root_fd, "link")
                before = len(os.listdir("/proc/self/fd"))
                with mock.patch.object(module.os, "fstat", side_effect=OSError("injected")):
                    with self.assertRaises(OSError):
                        module._open_relative(root_fd, "value")
                self.assertEqual(len(os.listdir("/proc/self/fd")), before)
            finally:
                os.close(root_fd)

    def test_copy_partial_acquisition_closes_source_fd(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.bin").write_bytes(b"source")
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                with module._Stage() as stage:
                    before = len(os.listdir("/proc/self/fd"))
                    artifact = SimpleNamespace(logical_locator="source.bin", byte_length=6, artifact_sha256=hashlib.sha256(b"source").hexdigest())
                    with mock.patch.object(stage, "create_file", side_effect=OSError("injected")):
                        with self.assertRaises(OSError):
                            module._copy_artifact(root_fd, artifact, stage)
                    self.assertEqual(len(os.listdir("/proc/self/fd")), before)
            finally:
                os.close(root_fd)

    def test_wheel_rejects_symlink_zero_mode_and_cross_catalog_prefix_collision(self) -> None:
        def wheel_bytes(mode: int) -> bytes:
            payload = b"payload"
            digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode("ascii")
            record = f"pkg/value,sha256={digest},{len(payload)}\npkg-1.dist-info/RECORD,,\n".encode()
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                info = zipfile.ZipInfo("pkg/value")
                info.external_attr = mode << 16
                archive.writestr(info, payload)
                record_info = zipfile.ZipInfo("pkg-1.dist-info/RECORD")
                record_info.external_attr = (stat.S_IFREG | 0o644) << 16
                archive.writestr(record_info, record)
            return stream.getvalue()

        for mode in (0, stat.S_IFLNK | 0o777):
            with self.subTest(mode=mode), module._Stage() as stage:
                data = wheel_bytes(mode)
                fd = stage.create_file("bad.whl")
                os.write(fd, data)
                os.close(fd)
                artifact = SimpleNamespace(logical_locator="bad.whl", byte_length=len(data), artifact_sha256=hashlib.sha256(data).hexdigest(), artifact_id="bad")
                with self.assertRaises(SolarSystemDataError):
                    module._wheel_catalog(artifact, stage)
        left = module._WheelCatalog("left", b"", ("a/b",), (), "r", {})
        right = module._WheelCatalog("right", b"", ("a",), (), "r", {})
        with self.assertRaises(SolarSystemDataError):
            module._validate_catalog_union((left, right))

    def test_exchange_setup_failure_terminates_and_reaps_worker(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        with mock.patch.object(module.selectors, "DefaultSelector", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                module._bounded_exchange(process, b"request")
        self.assertIsNotNone(process.poll())

    def test_hash_fd_cap_precedes_reads(self) -> None:
        fake = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_size=module._MAXIMUM_HELD_FILE_BYTES + 1)
        with (
            mock.patch.object(module.os, "fstat", return_value=fake),
            mock.patch.object(module.os, "pread", side_effect=AssertionError("READ_REACHED")),
        ):
            with self.assertRaises(SolarSystemDataError):
                module._hash_fd(7)

    def test_held_spk_mutation_takes_precedence_over_worker_coverage(self) -> None:
        before = (module._SPK_LENGTH, module._SPK_SHA256, (1, 2, 3))
        changed = (module._SPK_LENGTH, "f" * 64, (1, 2, 4))
        with (
            mock.patch.object(module, "_hash_fd", side_effect=(before, changed)),
            mock.patch.object(
                module,
                "_run_worker",
                side_effect=SolarSystemCoverageError("provider coverage failure"),
            ),
            mock.patch.object(module.os, "lseek", return_value=0),
        ):
            with self.assertRaises(SolarSystemDataError) as caught:
                module._run_worker_with_held_spk_observation(
                    abi=module._current_abi(),
                    role="PRIMARY",
                    nonce="1" * 64,
                    runtime_fd=10,
                    spk_fd=11,
                    python_fd=12,
                    worker_fd=13,
                    worker_source_byte_length=module._EXPECTED_WORKER_BYTE_LENGTH,
                    worker_sha256=module._EXPECTED_WORKER_SHA256,
                    effective_et=0.0,
                    targets=(10,),
                )
        self.assertIn("changed", str(caught.exception))
        coverage = worker_module._CoverageError("provider coverage failure")
        with self.assertRaises(worker_module._WorkerError) as worker_caught:
            worker_module._finish_provider_observation(
                coverage,
                cleanup_valid=True,
                fenv_after=0,
                spk_offset_before=0,
                spk_offset_after=0,
                spk_pre_stat=(1, 2, 3),
                spk_post_stat=(1, 2, 4),
                spk_length_before=module._SPK_LENGTH,
                spk_length_after=module._SPK_LENGTH,
                spk_sha_before=module._SPK_SHA256,
                spk_sha_after=module._SPK_SHA256,
            )
        self.assertNotIsInstance(worker_caught.exception, worker_module._CoverageError)

    def test_selected_chain_cycle_and_cap_are_data_not_coverage(self) -> None:
        class FakeSpice:
            def __init__(self, centers):
                self.centers = centers

            def spksfs(self, body, et, idlen):
                return 1, (body,), "SEGMENT"

            def spkuds(self, descriptor):
                body = descriptor[0]
                return body, self.centers[body], 1, 2, -100.0, 100.0, body * 100 + 1, body * 100 + 9

            def spkpvn(self, handle, descriptor, et):
                return 1, (1.0, 0.0, 0.0, 0.0, 0.0, 0.0), self.centers[descriptor[0]]

        cases = (
            FakeSpice({10: 3, 3: 10}),
            FakeSpice({index: index + 1 for index in range(1, 18)}),
        )
        starts = (10, 1)
        for spice, start in zip(cases, starts):
            with self.subTest(start=start), self.assertRaises(worker_module._WorkerError) as caught:
                worker_module._selected_chain(spice, start, 0.0, 1, ())
            self.assertNotIsInstance(caught.exception, worker_module._CoverageError)
            self.assertEqual(worker_module._error_kind(caught.exception), "DATA")


class OptionalRealKernelTests(unittest.TestCase):
    def _root_path(self) -> str:
        root_path = os.environ.get("JXPLANETX_M4C2B_ARTIFACT_ROOT")
        if not root_path:
            self.skipTest("set JXPLANETX_M4C2B_ARTIFACT_ROOT for exact-wheel/de440s execution")
        return root_path

    def _execute(self, root_fd: int, projection, label: str, seed: str):
        receipt = execute_cspice_spkgeo_replay(
            projection,
            artifact_root_directory_fd=root_fd,
            receipt_id=f"receipt.cspice.m4c2b.{label}",
            primary_batch_id=f"batch.cspice.m4c2b.{label}.primary",
            replay_batch_id=f"batch.cspice.m4c2b.{label}.replay",
            primary_execution_instance_nonce=hashlib.sha256((seed + ".primary").encode()).hexdigest(),
            replay_execution_instance_nonce=hashlib.sha256((seed + ".replay").encode()).hexdigest(),
        )
        validate_cspice_spkgeo_execution_receipt(receipt)
        self.assertEqual(
            receipt.primary_lane.semantic_content_sha256,
            receipt.replay_lane.semantic_content_sha256,
        )
        return receipt

    def test_opt_in_exact_et0_et1_and_endpoint_replays(self) -> None:
        root_path = self._root_path()
        abi = module._current_abi()
        root_fd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            et0 = self._execute(root_fd, projection_at(abi), "et0", "et0")
            receipt = et0
            self.assertIs(type(receipt), CspiceSpkgeoExecutionReceipt)
            self.assertEqual(receipt.primary_lane.state_payload_sha256, EXPECTED_ET0_STATE_SHA256)
            self.assertEqual(receipt.replay_lane.state_payload_sha256, EXPECTED_ET0_STATE_SHA256)
            self.assertEqual(hashlib.sha256(state_payload(receipt.primary_lane)).hexdigest(), EXPECTED_ET0_STATE_SHA256)
            self.assertEqual(len(light_time_payload(receipt.primary_lane)), 40)
            self.assertEqual(hashlib.sha256(light_time_payload(receipt.primary_lane)).hexdigest(), EXPECTED_ET0_LIGHT_SHA256)
            chain_preimage = chain_evidence_preimage(receipt.primary_lane)
            self.assertEqual(len(chain_preimage), EXPECTED_ET0_CHAIN_PREIMAGE_LENGTH)
            self.assertEqual(hashlib.sha256(chain_preimage).hexdigest(), EXPECTED_ET0_CHAIN_SHA256)
            self.assertEqual(
                tuple(
                    tuple(
                        (
                            leg.selected_body_naif_id,
                            leg.center_naif_id,
                            leg.frame_naif_id,
                            leg.segment_type,
                            leg.begin_daf_address,
                            leg.end_daf_address,
                            leg.segment_identifier,
                            leg.spkpvn_returned_frame_naif_id,
                            leg.spkpvn_returned_center_naif_id,
                        )
                        for leg in chain.legs
                    )
                    for chain in receipt.primary_lane.target_chains
                ),
                (
                    ((10, 0, 1, 2, 1_604_151, 1_843_904, "DE-0440LE-0440", 1, 0),),
                    ((3, 0, 1, 2, 830_073, 1_110_926, "DE-0440LE-0440", 1, 0),),
                    (
                        (399, 3, 1, 2, 2_967_309, 4_090_712, "DE-0440LE-0440", 1, 3),
                        (3, 0, 1, 2, 830_073, 1_110_926, "DE-0440LE-0440", 1, 0),
                    ),
                    (
                        (301, 3, 1, 2, 1_843_905, 2_967_308, "DE-0440LE-0440", 1, 3),
                        (3, 0, 1, 2, 830_073, 1_110_926, "DE-0440LE-0440", 1, 0),
                    ),
                    ((5, 0, 1, 2, 1_230_806, 1_319_859, "DE-0440LE-0440", 1, 0),),
                ),
            )

            et1_projection = projection_at(
                abi,
                whole=1,
                fraction=float.fromhex("0x1p-53"),
            )
            self.assertEqual(et1_projection.binary64_projection.rounded_value.hex(), "0x1.0000000000000p+0")
            self.assertEqual(
                (et1_projection.exact_error_numerator, et1_projection.exact_error_denominator),
                (-1, 1 << 53),
            )
            et1 = self._execute(root_fd, et1_projection, "et1", "et1")
            self.assertEqual(et1.primary_lane.state_payload_sha256, EXPECTED_ET1_STATE_SHA256)
            self.assertEqual(hashlib.sha256(light_time_payload(et1.primary_lane)).hexdigest(), EXPECTED_ET1_LIGHT_SHA256)
            et1_chain = chain_evidence_preimage(et1.primary_lane)
            self.assertEqual(len(et1_chain), EXPECTED_ET1_CHAIN_PREIMAGE_LENGTH)
            self.assertEqual(hashlib.sha256(et1_chain).hexdigest(), EXPECTED_ET1_CHAIN_SHA256)
            self.assertEqual(
                hashlib.sha256(state_payload(et0.primary_lane) + state_payload(et1.primary_lane)).hexdigest(),
                EXPECTED_ET0_ET1_STATE_SHA256,
            )

            endpoint_payload = bytearray()
            for label, value, state_sha, light_sha in ENDPOINT_CASES:
                whole = math.floor(value + 0.5)
                projected = projection_at(abi, whole=whole, fraction=value - whole)
                self.assertEqual(projected.binary64_projection.rounded_value.hex(), value.hex())
                endpoint = self._execute(root_fd, projected, label, label)
                observed_state = state_payload(endpoint.primary_lane)
                self.assertEqual(len(observed_state), 240)
                self.assertEqual(hashlib.sha256(observed_state).hexdigest(), state_sha)
                self.assertEqual(hashlib.sha256(light_time_payload(endpoint.primary_lane)).hexdigest(), light_sha)
                endpoint_payload.extend(observed_state)
            self.assertEqual(len(endpoint_payload), 960)
            self.assertEqual(hashlib.sha256(endpoint_payload).hexdigest(), EXPECTED_ENDPOINT_STATE_SHA256)

            unavailable = projection_at(abi, targets=(123_456,))
            with self.assertRaises(SolarSystemCoverageError):
                execute_cspice_spkgeo_replay(
                    unavailable,
                    artifact_root_directory_fd=root_fd,
                    receipt_id="receipt.cspice.m4c2b.unavailable",
                    primary_batch_id="batch.cspice.m4c2b.unavailable.primary",
                    replay_batch_id="batch.cspice.m4c2b.unavailable.replay",
                    primary_execution_instance_nonce="3" * 64,
                    replay_execution_instance_nonce="4" * 64,
                )
        finally:
            os.close(root_fd)

    def test_opt_in_private_worker_outside_endpoints_are_zero_payload_coverage(self) -> None:
        root_path = self._root_path()
        abi = module._current_abi()
        provider = production_provider(abi)
        by_id = {item.artifact_id: item for item in provider.artifacts}
        source_root_fd = os.open(root_path, os.O_RDONLY | os.O_DIRECTORY)
        python_fd = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
        worker_fd = os.open(Path(module.__file__).with_name("_cspice_worker.py"), os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            with module._Stage() as stage:
                module._prepare_stage(stage, by_id, source_root_fd, abi)
                runtime_fd = os.open("runtime", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=stage.root_fd)
                spk_fd = stage.open_file(by_id[module._SPK_ID].logical_locator)
                try:
                    for index, effective_et in enumerate(
                        (
                            math.nextafter(-4_734_072_000.0, -math.inf),
                            math.nextafter(4_735_368_000.0, math.inf),
                        )
                    ):
                        captured: list[tuple[bytes, bytes, int]] = []
                        real_parse = module._parse_response

                        def observing_parse(stdout: bytes, stderr: bytes, return_code: int):
                            captured.append((stdout, stderr, return_code))
                            return real_parse(stdout, stderr, return_code)

                        with mock.patch.object(module, "_parse_response", observing_parse):
                            with self.assertRaises(SolarSystemCoverageError):
                                module._run_worker(
                                    abi=abi,
                                    role="PRIMARY",
                                    nonce=hashlib.sha256(f"outside-{index}".encode()).hexdigest(),
                                    runtime_fd=runtime_fd,
                                    spk_fd=spk_fd,
                                    python_fd=python_fd,
                                    worker_fd=worker_fd,
                                    worker_source_byte_length=module._EXPECTED_WORKER_BYTE_LENGTH,
                                    worker_sha256=module._EXPECTED_WORKER_SHA256,
                                    effective_et=effective_et,
                                    targets=(10, 3, 399, 301, 5),
                                )
                        self.assertEqual(len(captured), 1)
                        stdout, stderr, return_code = captured[0]
                        length = struct.unpack(">I", stdout[:4])[0]
                        response = json.loads(stdout[4 : 4 + length].decode("ascii"))
                        self.assertEqual(stderr, b"")
                        self.assertEqual(return_code, 70)
                        self.assertEqual(
                            response,
                            {
                                "error_kind": "COVERAGE",
                                "error_status": "FAIL_CLOSED",
                                "schema": module._RESULT_SCHEMA,
                                "status": "ERROR",
                            },
                        )
                        self.assertNotIn("result", response)
                finally:
                    os.close(spk_fd)
                    os.close(runtime_fd)
        finally:
            os.close(worker_fd)
            os.close(python_fd)
            os.close(source_root_fd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
