#!/usr/bin/env python3
"""Deterministic JX v2 operationalization of the B22 TNO classifier.

This module extends the frozen v1 classifier with one backward-compatible
outer-resonance band, motivated by the published 36:1 label for (474640).
It is deliberately a
pure, CPU-only analysis component: importing it does not import NumPy, CuPy,
REBOUND, or JX dynamics modules, and it has no dynamics execution surface.

The resonance detector is a JX operationalization of underspecified details
in Bernardinelli et al. (2022), not a claim of exact reproduction of the
authors' unpublished implementation.  The retained DES Y6 labels are used
only as source-backed catalog-label custody.  They are not trajectory
fixtures and therefore cannot establish scientific classifier qualification.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib.machinery
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import types
from typing import Any, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_b22_classifier_v2.py"
PREDECESSOR_SOURCE_PATH = (
    PROJECT_ROOT / "benchmarks" / "jx_tno_b22_classifier_v1.py"
)
PREDECESSOR_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_b22_classifier_v1.py"
QUALIFICATION_SOURCE_PATH = (
    PROJECT_ROOT / "benchmarks" / "jx_tno_b22_classifier_qualification_v1.py"
)
QUALIFICATION_TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_tno_b22_classifier_qualification_v1.py"
)
QUALIFICATION_PROTOCOL_PATH = (
    PROJECT_ROOT / "runs" / "tno_b22_classifier_qualification_v1" / "PROTOCOL.json"
)
STAGE1_SOURCE_PATH = (
    PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v4.py"
)
DES_CATALOG_PATH = (
    PROJECT_ROOT / "runs" / "tno_catalog_reacquisition_v1" / "raw" / "des_cds_catalog.dat"
)
DES_README_PATH = (
    PROJECT_ROOT / "runs" / "tno_catalog_reacquisition_v1" / "raw" / "des_cds_ReadMe"
)
PREREG_SOURCE_PATH = (
    PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_classification_pilot_v1.py"
)
PREREG_TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_tno_10my_reduced5_classification_pilot_v1.py"
)
PREREG_PROTOCOL_PATH = (
    PROJECT_ROOT / "runs" / "tno_10my_reduced5_classification_pilot_v1" / "PROTOCOL.json"
)

STAGE1_SOURCE_BYTES = 436_125
STAGE1_SOURCE_SHA256 = (
    "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"
)
DES_CATALOG_BYTES = 498_168
DES_CATALOG_SHA256 = (
    "457554872af8b55b44404f4dfbaeeceafb8120cc43cb09b1350f4e48c9bc1a92"
)
DES_README_BYTES = 11_035
DES_README_SHA256 = (
    "4dff6940798eafd4ad172ab81b8e531a9952839db5d67de9ca8e1f869f2eb5ca"
)
PREREG_SOURCE_BYTES = 39_082
PREREG_SOURCE_SHA256 = (
    "6a5005946c84ce5321c029178e4bb5a855388d6b7870d572c63f505d70e7c1c8"
)
PREREG_TEST_BYTES = 19_244
PREREG_TEST_SHA256 = (
    "4f0ae8485c5b8fed9fcb88a2d0cbd85731033ba3c69c83d9e96b67922d510027"
)
PREREG_PROTOCOL_BYTES = 17_595
PREREG_PROTOCOL_SHA256 = (
    "9a00d2d30cf59267ff9d2474d6e3d42dde8bb4e5713d1a0c3c073d56ff8f9c2c"
)
PREREG_PROTOCOL_CONTENT_SHA256 = (
    "4b6c4d4627ca39ec85b551f9d920d92b16f3da8a47dad3727e3467c3a649c699"
)
PREREG_PROTOCOL_DOMAIN = (
    b"jx.tno.10my-reduced5-classification-pilot.protocol.v1\0"
)
PREREG_GIT_COMMIT = "80975c82226e5cb82e145e6687acc3a8325cfde9"
PREREG_GIT_TAG = (
    "tno-10my-reduced5-classification-pilot-v1-preregistered-20261010"
)
PREREG_GIT_TAG_OBJECT = "b41675f66dabf0115cd2e763b6d1ce7f08dfa0cf"

PREDECESSOR_SOURCE_BYTES = 82_137
PREDECESSOR_SOURCE_SHA256 = (
    "e3855734072b604941c91a0d598388c4665de2a1903677876a0fce84e62de600"
)
PREDECESSOR_TEST_BYTES = 35_099
PREDECESSOR_TEST_SHA256 = (
    "f5457810ad6ca6c3d4b83a369759bf7fd22726cf2edb1adda4dfc1df04e2b084"
)
QUALIFICATION_SOURCE_BYTES = 37_044
QUALIFICATION_SOURCE_SHA256 = (
    "e93688c1d94903ec0f474d4c2aa1d5e23e6480ed7d6b138a9d3bfd80b3e6079b"
)
QUALIFICATION_TEST_BYTES = 16_034
QUALIFICATION_TEST_SHA256 = (
    "1e4c606fc5960d2b0df4ffae1810a7e8a7167436b8f6215c2dde8db097ff1834"
)
QUALIFICATION_PROTOCOL_BYTES = 10_923
QUALIFICATION_PROTOCOL_SHA256 = (
    "eb0da4c8f19a26f99adac1b863b9f65f69454f7814d0d13865e08f705729b5a0"
)
QUALIFICATION_PROTOCOL_CONTENT_SHA256 = (
    "ceb790d65d87f1099390f210c547100c6b284f5b39896c1e23fd3bcd542c9079"
)
QUALIFICATION_PROTOCOL_DOMAIN = b"jx.tno.b22-classifier-qualification.protocol.v1\0"

PYTHON_RESOLVED_EXECUTABLE = Path("/usr/bin/python3.14")
PYTHON_EXECUTABLE_BYTES = 7_477_160
PYTHON_EXECUTABLE_SHA256 = (
    "52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41"
)
PYTHON_VERSION = "3.14.4 (main, Aug 20 2026, 10:41:58) [GCC 15.2.0]"
PYTHON_CACHE_TAG = "cpython-314"
LIBM_PATH = Path("/usr/lib/x86_64-linux-gnu/libm.so.6")
LIBM_BYTES = 1_198_376
LIBM_SHA256 = "1c488db6f4fe8cecb72456bc018de000b7a9b539c2994864b0416b5e8371b3d8"

SCHEMA = "jx.tno.b22-classifier.mechanical-result.v2"
DOMAIN = b"jx.tno.b22-classifier.mechanical-result.v2\0"
ELEMENT_SERIES_DOMAIN = b"jx.tno.b22-classifier.element-series.v2\0"
BATCH_STATE_DOMAIN = b"jx.tno.b22-classifier.batch-state.v2\0"
NODE_STATE_DOMAIN = b"jx.tno.b22-classifier.node-state.v2\0"
CANDIDATE_ROSTER_DOMAIN = b"jx.tno.b22-classifier.candidate-roster.v2\0"
ARGUMENT_ROSTER_DOMAIN = b"jx.tno.b22-classifier.argument-roster.v2\0"
INHERITED_CANDIDATE_ROSTER_DOMAIN = (
    b"jx.tno.b22-classifier.inherited-candidate-roster.v2\0"
)
NEW_BAND_CANDIDATE_ROSTER_DOMAIN = (
    b"jx.tno.b22-classifier.new-band-candidate-roster.v2\0"
)
INHERITED_ARGUMENT_ROSTER_DOMAIN = (
    b"jx.tno.b22-classifier.inherited-argument-roster.v2\0"
)
NEW_BAND_ARGUMENT_ROSTER_DOMAIN = (
    b"jx.tno.b22-classifier.new-band-argument-roster.v2\0"
)
CLASSIFIER_NAME = "JX_B22_OPERATIONALIZATION_V2_36_TO_1_NOT_EXACT_PAPER_REPRODUCTION"

NODE_COUNT = 50_001
RESONANCE_NODE_COUNT = 50_000
CADENCE_JULIAN_YEARS = 200
WINDOW_COUNT = 50
POINTS_PER_WINDOW = 1_000
ANGLE_BIN_COUNT = 18
MINIMUM_EMPTY_BINS = 2
MINIMUM_CONTIGUOUS_WINDOWS = 46
P_Q_MIN = 1
INHERITED_P_Q_MAX = 29
P_MAX = 36
Q_MAX = 29
P_Q_MAX = P_MAX
INHERITED_COEFFICIENT_LIMIT = 29
NEW_BAND_COEFFICIENT_LIMIT = 35
COEFFICIENT_MIN = -NEW_BAND_COEFFICIENT_LIMIT
COEFFICIENT_MAX = NEW_BAND_COEFFICIENT_LIMIT
EXPECTED_PERIOD_PAIR_UNIVERSE_COUNT = 653
EXPECTED_INHERITED_PERIOD_PAIR_UNIVERSE_COUNT = 539
EXPECTED_NEW_BAND_PERIOD_PAIR_UNIVERSE_COUNT = 114
EXPECTED_ARGUMENT_UNIVERSE_COUNT = 1_712_871
EXPECTED_INHERITED_ARGUMENT_UNIVERSE_COUNT = 1_327_797
EXPECTED_NEW_BAND_ARGUMENT_UNIVERSE_COUNT = 385_074
PERIOD_FRACTION = Fraction(3, 20)
SCATTERING_THRESHOLD = Fraction(3, 80)
DETACHED_THRESHOLD = Fraction(6, 25)
OORT_THRESHOLD = Fraction(2000, 1)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
OBJECT_IDS = (
    "jpl-20090377",
    "jpl-20474640",
    "jpl-54526835",
    "jpl-20541132",
    "jpl-50666516",
    "jpl-20148209",
    "jpl-20574372",
    "jpl-20303775",
    "jpl-20026181",
    "jpl-20042301",
    "jpl-20026375",
    "jpl-20015874",
)
SPK_IDS = (
    20090377,
    20474640,
    54526835,
    20541132,
    50666516,
    20148209,
    20574372,
    20303775,
    20026181,
    20042301,
    20026375,
    20015874,
)
NATIVE_EPOCHS_JD_TDB = (
    2458411.5,
    2456187.5,
    2460034.5,
    2458052.5,
    2456596.5,
    2452568.5,
    2458299.5,
    2458767.5,
    2458156.5,
    2458324.5,
    2457961.5,
    2458098.5,
)
REDUCED_BODY_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
REDUCED_NAIF_IDS = (10, 5, 6, 7, 8)
REDUCED_GM_HEX = (
    "0x1.3bd13a81d2261p+5",
    "0x1.34c6609ebf23fp-5",
    "0x1.71d0e2fd3b84dp-7",
    "0x1.c3d8f106aa648p-10",
    "0x1.0a8c8f8c38c6ap-9",
)
RESOLVED11_GM_HEX = (
    "0x1.3bd0beccac6afp+5",
    "0x1.b7cecb4d5b352p-18",
    "0x1.954f090f44726p-14",
    "0x1.f14fffb533693p-14",
    "0x1.877c4ea8180a5p-20",
    "0x1.ab7a3929b393ap-17",
    "0x1.34c6609ebf23fp-5",
    "0x1.71d0e2fd3b84dp-7",
    "0x1.c3d8f106aa648p-10",
    "0x1.0a8c8f8c38c6ap-9",
    "0x1.379282522bdf1p-22",
)
BRIDGE_BATCH_STATE_SHA256_V1 = (
    "d261a41e84fb1ca6ad3068d31e83792c989d15fc0662c0bb94498f3d35020961",
    "f81cd8486f2790c8254a3e27fe697453f89dc00513d11d1e4af6cd8ab3284d12",
    "edc14df81e96d276b3ae81c9311c63a5bda06d1af5290fd480b84787f08114e3",
    "ef6b38e0844cd8c17c5f2ec5d7e2041a5f5edf784dbca70a9c00ee2ea3cb8b47",
    "6e680e4dcf510c8f98d9258b66991ac69b7f77c7989442ceafb3ab456c091827",
    "aa0aca8ff706ca2f57705cb184dd8f936d7ce046204f2db2d0b0357dceb0e44a",
    "c0cc879a9bf542c57047d6c0f2ee6bd10f8242fecc18416c49d24538ab54105c",
    "fa177eec84307ccc223eb8852e1f798b19ade2bfdf287f3470606b7a1526fbcd",
    "44ff96b214e37fd787e2a2e6ea774fda1afb42a90e18f949b4653eb58b5122ab",
    "7ea2579119abd4b77a577edd6e229639735b1ac84c337d7b5f59cc2b3747ecf2",
    "997f7e5a7981da981bd280a858e391843917c9958c41ad7114d97cdd01126e96",
    "c011fe45a166101ddca449de1701162afb48f7f46406964f440164833d7b3b92",
)
BRIDGE_BATCH_STATE_SHA256 = (
    "fe21970957d39e6c17b8f88d81262e0e8fdca6c518459506c010230153880a03",
    "f9d82ed0625c025efc5b51eddf012878d720de887f46eaed911c6bfb1f872221",
    "93a420a8a4d4dd26193ac218f85b292da22e6e8c8e96c937f9fb6b167385c438",
    "7981c62666751f9664594ca3357adcbc18b5854d3ffbc7869cbd22c87941010a",
    "c981c00fc5e0dbab99e8b3a632c9acb2331934d06ead5758e6b224f48880a9af",
    "4fd07995c2f21eb2a520f4dd49755b2e4283d45a35b92801c9d59ab04bae4b26",
    "5788ed16542555dad3915e1c7981d0e41b6e9b5dbb14366edee6c86c94b64332",
    "5c1cbf69c7e042008c6d1bec60a922e51ad49f88f53aed2846b9fd4a73792aad",
    "917254a6d47c43b71062592750f963204155835832a1bfd06df21b183ff0f30e",
    "73a9d65cfd627ffec3f7dbace5eeb0d28ee3a73e4c5d8e343b27f58a066cca1a",
    "12cef0b83d400f31b4bab41b7887839c96be20a1e0e4f6f7f5766269431f686a",
    "1bef75d4097692510987413ec4ed574977946fc10fa4adedc6a74d83339eb6b0",
)
ELEMENT_MU = math.fsum(float.fromhex(value) for value in REDUCED_GM_HEX)
ELEMENT_MU_HEX = "0x1.3c3d3b0c49e7dp+5"
MATH_PI_HEX = "0x1.921fb54442d18p+1"
MATH_TAU_HEX = "0x1.921fb54442d18p+2"
COS_OBLIQUITY = float.fromhex("0x1.d5c0357681ef3p-1")
SIN_OBLIQUITY = float.fromhex("0x1.9752e50f4b399p-2")
ANGLE_BIN_WIDTH = math.tau / float(ANGLE_BIN_COUNT)
ANGLE_BIN_WIDTH_HEX = "0x1.657184ae74487p-2"
EPHEMERIS = "DE441"
MATH_CALLABLE_NAMES = (
    "acos",
    "atan2",
    "cos",
    "fmod",
    "fsum",
    "gcd",
    "hypot",
    "isfinite",
    "ldexp",
    "nextafter",
    "sin",
    "sqrt",
    "ulp",
)
_BOOTSTRAP_MATH_MODULE = math
_BOOTSTRAP_MATH_CALLABLES = tuple(
    getattr(math, name) for name in MATH_CALLABLE_NAMES
)

LABEL_RESONANT = "RESONANT_OR_RESONANT_CANDIDATE"
LABEL_OORT = "OORT"
LABEL_CENTAUR = "CENTAUR"
LABEL_SCATTERING = "SCATTERING"
LABEL_DETACHED = "DETACHED"
LABEL_CLASSICAL = "CLASSICAL"
ALLOWED_LABELS = frozenset(
    {
        LABEL_RESONANT,
        LABEL_OORT,
        LABEL_CENTAUR,
        LABEL_SCATTERING,
        LABEL_DETACHED,
        LABEL_CLASSICAL,
    }
)
ELEMENT_FRAME = "ECLIPJ2000_MODEL_BARYCENTRIC_INERTIAL"
NODE_ORDER = "ASCENDING_RELATIVE_TIME_INDEX_0_THROUGH_50000"


class ClassifierError(ValueError):
    """A classifier input or retained record failed closed."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise ClassifierError("value is not canonical finite JSON") from exc


def _content_sha256(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_sha256", None)
    return _sha256(DOMAIN + _canonical_json(payload))


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ClassifierError(f"{label} cannot be opened as a regular file") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ClassifierError(f"{label} is linked or nonregular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ClassifierError(f"{label} changed while being read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


_BOOTSTRAP_IMPLEMENTATION_PAYLOAD = _read_regular_bytes(
    SCRIPT_PATH, "classifier implementation at bootstrap"
)
_BOOTSTRAP_IMPLEMENTATION_BYTES = len(_BOOTSTRAP_IMPLEMENTATION_PAYLOAD)
_BOOTSTRAP_IMPLEMENTATION_SHA256 = _sha256(_BOOTSTRAP_IMPLEMENTATION_PAYLOAD)
del _BOOTSTRAP_IMPLEMENTATION_PAYLOAD


def _require_file(path: Path, size: int, digest: str, label: str) -> bytes:
    payload = _read_regular_bytes(path, label)
    if len(payload) != size or _sha256(payload) != digest:
        raise ClassifierError(f"{label} differs from its frozen byte identity")
    return payload


def _strict_json_mapping(payload: bytes, label: str) -> dict[str, Any]:
    def strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if type(key) is not str or key in result:
                raise ClassifierError(f"{label} has a duplicate or invalid key")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ClassifierError(f"{label} contains non-finite constant {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=strict_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClassifierError(f"{label} is not strict JSON") from exc
    if type(value) is not dict:
        raise ClassifierError(f"{label} must be a JSON object")
    return value


def validate_preregistration_v2() -> dict[str, object]:
    """Validate the exact prospective classifier protocol and Git identity."""

    source = _require_file(
        PREREG_SOURCE_PATH,
        PREREG_SOURCE_BYTES,
        PREREG_SOURCE_SHA256,
        "10-Myr preregistration implementation",
    )
    test = _require_file(
        PREREG_TEST_PATH,
        PREREG_TEST_BYTES,
        PREREG_TEST_SHA256,
        "10-Myr preregistration test",
    )
    protocol_payload = _require_file(
        PREREG_PROTOCOL_PATH,
        PREREG_PROTOCOL_BYTES,
        PREREG_PROTOCOL_SHA256,
        "10-Myr preregistration protocol",
    )
    del source, test
    protocol = _strict_json_mapping(protocol_payload, "10-Myr preregistration protocol")
    if protocol_payload != _canonical_json(protocol):
        raise ClassifierError("10-Myr preregistration protocol is not canonical")
    content_record = dict(protocol)
    content_record.pop("content_sha256", None)
    observed_content = _sha256(
        PREREG_PROTOCOL_DOMAIN + _canonical_json(content_record)
    )
    if (
        protocol.get("schema")
        != "jx.tno.10my-reduced5-classification-pilot.protocol.v1"
        or protocol.get("content_sha256") != PREREG_PROTOCOL_CONTENT_SHA256
        or observed_content != PREREG_PROTOCOL_CONTENT_SHA256
    ):
        raise ClassifierError("10-Myr preregistration protocol content differs")
    cohort = protocol.get("cohort")
    classifier_input = protocol.get("classifier_input")
    if (
        type(cohort) is not dict
        or cohort.get("object_ids_in_exact_order") != list(OBJECT_IDS)
        or cohort.get("holdout_clone_indices") != list(HOLDOUT_CLONE_INDICES)
        or cohort.get("states_per_object") != 9
        or type(classifier_input) is not dict
        or classifier_input.get("cadence_julian_years") != CADENCE_JULIAN_YEARS
        or classifier_input.get("node_count_including_final_endpoint") != NODE_COUNT
        or classifier_input.get("element_frame") != ELEMENT_FRAME
    ):
        raise ClassifierError("classifier constants differ from preregistration")
    return {
        "implementation": {
            "path": str(PREREG_SOURCE_PATH.relative_to(PROJECT_ROOT)),
            "byte_count": PREREG_SOURCE_BYTES,
            "sha256": PREREG_SOURCE_SHA256,
        },
        "test": {
            "path": str(PREREG_TEST_PATH.relative_to(PROJECT_ROOT)),
            "byte_count": PREREG_TEST_BYTES,
            "sha256": PREREG_TEST_SHA256,
        },
        "protocol": {
            "path": str(PREREG_PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
            "byte_count": PREREG_PROTOCOL_BYTES,
            "sha256": PREREG_PROTOCOL_SHA256,
            "content_sha256": PREREG_PROTOCOL_CONTENT_SHA256,
        },
        "git": {
            "commit": PREREG_GIT_COMMIT,
            "annotated_tag": PREREG_GIT_TAG,
            "tag_object": PREREG_GIT_TAG_OBJECT,
            "identity_source": "EXACT_EXTERNALLY_RETAINED_GIT_CUSTODY",
        },
    }


def validate_successor_basis_v2() -> dict[str, object]:
    """Bind v1 byte custody and the published 36:1 expansion rationale."""

    _require_file(
        PREDECESSOR_SOURCE_PATH,
        PREDECESSOR_SOURCE_BYTES,
        PREDECESSOR_SOURCE_SHA256,
        "B22 classifier v1 implementation",
    )
    _require_file(
        PREDECESSOR_TEST_PATH,
        PREDECESSOR_TEST_BYTES,
        PREDECESSOR_TEST_SHA256,
        "B22 classifier v1 test",
    )
    _require_file(
        QUALIFICATION_SOURCE_PATH,
        QUALIFICATION_SOURCE_BYTES,
        QUALIFICATION_SOURCE_SHA256,
        "B22 qualification implementation",
    )
    _require_file(
        QUALIFICATION_TEST_PATH,
        QUALIFICATION_TEST_BYTES,
        QUALIFICATION_TEST_SHA256,
        "B22 qualification test",
    )
    protocol_payload = _require_file(
        QUALIFICATION_PROTOCOL_PATH,
        QUALIFICATION_PROTOCOL_BYTES,
        QUALIFICATION_PROTOCOL_SHA256,
        "B22 qualification protocol",
    )
    protocol = _strict_json_mapping(
        protocol_payload, "B22 qualification protocol"
    )
    if protocol_payload != _canonical_json(protocol):
        raise ClassifierError("B22 qualification protocol is not canonical")
    content_record = dict(protocol)
    content_record.pop("content_sha256", None)
    observed_content = _sha256(
        QUALIFICATION_PROTOCOL_DOMAIN + _canonical_json(content_record)
    )
    finding = protocol.get("decisive_search_domain_finding")
    if (
        protocol.get("schema")
        != "jx.tno.b22-classifier-qualification.protocol.v1"
        or protocol.get("content_sha256")
        != QUALIFICATION_PROTOCOL_CONTENT_SHA256
        or observed_content != QUALIFICATION_PROTOCOL_CONTENT_SHA256
        or type(finding) is not dict
        or finding.get("object_id") != "jpl-20474640"
        or finding.get("published_numbered_object") != "474640"
        or finding.get("published_catalog_class") != "resonant"
        or finding.get("published_resonance") != {"p": 36, "q": 1}
        or finding.get("within_frozen_jx_v1_search_domain") is not False
    ):
        raise ClassifierError("B22 36:1 successor basis differs")
    return {
        "predecessor": {
            "implementation_path": str(
                PREDECESSOR_SOURCE_PATH.relative_to(PROJECT_ROOT)
            ),
            "implementation_byte_count": PREDECESSOR_SOURCE_BYTES,
            "implementation_sha256": PREDECESSOR_SOURCE_SHA256,
            "test_path": str(PREDECESSOR_TEST_PATH.relative_to(PROJECT_ROOT)),
            "test_byte_count": PREDECESSOR_TEST_BYTES,
            "test_sha256": PREDECESSOR_TEST_SHA256,
        },
        "qualification": {
            "implementation_sha256": QUALIFICATION_SOURCE_SHA256,
            "test_sha256": QUALIFICATION_TEST_SHA256,
            "protocol_path": str(
                QUALIFICATION_PROTOCOL_PATH.relative_to(PROJECT_ROOT)
            ),
            "protocol_byte_count": QUALIFICATION_PROTOCOL_BYTES,
            "protocol_sha256": QUALIFICATION_PROTOCOL_SHA256,
            "protocol_content_sha256": QUALIFICATION_PROTOCOL_CONTENT_SHA256,
        },
        "published_36_to_1": {
            "object_id": "jpl-20474640",
            "numbered_object": "474640",
            "p": 36,
            "q": 1,
        },
    }


def implementation_identity_v2() -> dict[str, object]:
    payload = _read_regular_bytes(SCRIPT_PATH, "classifier implementation")
    if (
        len(payload) != _BOOTSTRAP_IMPLEMENTATION_BYTES
        or _sha256(payload) != _BOOTSTRAP_IMPLEMENTATION_SHA256
    ):
        raise ClassifierError(
            "classifier implementation differs from its bootstrap source identity"
        )
    return {
        "path": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        "byte_count": _BOOTSTRAP_IMPLEMENTATION_BYTES,
        "sha256": _BOOTSTRAP_IMPLEMENTATION_SHA256,
        "identity_scope": "BOOTSTRAP_SOURCE_BYTES_RECHECKED_AT_USE",
    }


_BOUND_STAGE1: types.ModuleType | None = None


def validate_stage1_source_v2() -> dict[str, object]:
    _require_file(
        STAGE1_SOURCE_PATH,
        STAGE1_SOURCE_BYTES,
        STAGE1_SOURCE_SHA256,
        "Stage-1 v4 implementation",
    )
    return {
        "path": str(STAGE1_SOURCE_PATH.relative_to(PROJECT_ROOT)),
        "byte_count": STAGE1_SOURCE_BYTES,
        "sha256": STAGE1_SOURCE_SHA256,
    }


def load_bound_stage1_v2() -> types.ModuleType:
    """Compile/execute the exact Stage-1 source bytes without a pyc path."""

    global _BOUND_STAGE1
    if _BOUND_STAGE1 is not None:
        return _BOUND_STAGE1
    payload = _require_file(
        STAGE1_SOURCE_PATH,
        STAGE1_SOURCE_BYTES,
        STAGE1_SOURCE_SHA256,
        "Stage-1 v4 implementation",
    )
    forbidden_prefixes = ("numpy", "cupy", "cupyx", "rebound", "jxplanetx")
    before = frozenset(sys.modules)
    module_name = "_jx_tno_b22_bound_stage1_v4"
    module = types.ModuleType(module_name)
    module.__file__ = str(STAGE1_SOURCE_PATH)
    module.__package__ = ""
    module.__spec__ = None
    sys.modules[module_name] = module
    try:
        code = compile(payload, str(STAGE1_SOURCE_PATH), "exec", dont_inherit=True, optimize=0)
        exec(code, module.__dict__)
    except Exception as exc:
        raise ClassifierError("exact Stage-1 source-only execution failed") from exc
    finally:
        sys.modules.pop(module_name, None)
    added_forbidden = sorted(
        name
        for name in set(sys.modules).difference(before)
        if name in forbidden_prefixes
        or name.startswith(tuple(item + "." for item in forbidden_prefixes))
    )
    if added_forbidden:
        raise ClassifierError("Stage-1 source-only loader imported a dynamics dependency")
    required = {
        "_angle",
        "_weighted_centroid",
        "_checkpoint",
        "extract_elliptic_orbital_elements",
        "collapse_reduced_model",
        "MassiveState",
        "TracerState",
    }
    if any(not callable(getattr(module, name, None)) for name in required):
        raise ClassifierError("Stage-1 source-only callable roster differs")
    if tuple(module.REDUCED_BODY_IDS) != REDUCED_BODY_IDS:
        raise ClassifierError("Stage-1 reduced-five roster differs")
    _BOUND_STAGE1 = module
    return module


def runtime_identity_v2() -> dict[str, object]:
    """Return and validate the exact interpreter/libm execution identity."""

    resolved = Path(sys.executable).resolve()
    if resolved != PYTHON_RESOLVED_EXECUTABLE:
        raise ClassifierError("classifier interpreter path differs")
    payload = _require_file(
        resolved,
        PYTHON_EXECUTABLE_BYTES,
        PYTHON_EXECUTABLE_SHA256,
        "classifier interpreter",
    )
    del payload
    if sys.version != PYTHON_VERSION or sys.implementation.cache_tag != PYTHON_CACHE_TAG:
        raise ClassifierError("classifier interpreter identity differs")
    if (
        math is not _BOOTSTRAP_MATH_MODULE
        or type(math) is not types.ModuleType
        or sys.modules.get("math") is not math
    ):
        raise ClassifierError("math module registry identity differs")
    math_spec = getattr(math, "__spec__", None)
    if (
        math_spec is None
        or math_spec.origin != "built-in"
        or math_spec.loader is not importlib.machinery.BuiltinImporter
    ):
        raise ClassifierError("math module is not the frozen built-in implementation")
    for name, bootstrap_function in zip(
        MATH_CALLABLE_NAMES,
        _BOOTSTRAP_MATH_CALLABLES,
        strict=True,
    ):
        function = getattr(math, name, None)
        if (
            function is not bootstrap_function
            or type(function) is not types.BuiltinFunctionType
            or function.__module__ != "math"
            or function.__name__ != name
        ):
            raise ClassifierError(f"math.{name} callable identity differs")
    if (
        type(math.pi) is not float
        or math.pi.hex() != MATH_PI_HEX
        or type(math.tau) is not float
        or math.tau.hex() != MATH_TAU_HEX
        or type(math.inf) is not float
        or math.inf != float("inf")
        or ELEMENT_MU.hex() != ELEMENT_MU_HEX
        or ANGLE_BIN_WIDTH.hex() != ANGLE_BIN_WIDTH_HEX
    ):
        raise ClassifierError("math constants or derived classifier constants differ")
    _require_file(LIBM_PATH, LIBM_BYTES, LIBM_SHA256, "classifier libm")
    if os.environ.get("LD_PRELOAD") not in (None, "") or os.environ.get(
        "LD_AUDIT"
    ) not in (None, ""):
        raise ClassifierError("dynamic-loader interposition environment is not empty")
    try:
        maps_lines = Path("/proc/self/maps").read_text(encoding="ascii").splitlines()
        libm_stat = LIBM_PATH.stat()
    except (OSError, UnicodeError) as exc:
        raise ClassifierError("loaded libm mapping cannot be inspected") from exc
    expected_device = f"{os.major(libm_stat.st_dev):02x}:{os.minor(libm_stat.st_dev):02x}"
    mapped = False
    for line in maps_lines:
        parts = line.split(maxsplit=5)
        if len(parts) != 6 or parts[5] != str(LIBM_PATH):
            continue
        if parts[3].lower() == expected_device and parts[4] == str(libm_stat.st_ino):
            mapped = True
            break
    if not mapped:
        raise ClassifierError("frozen libm file is not mapped into this process")
    return {
        "python_resolved_executable": str(resolved),
        "python_executable_bytes": PYTHON_EXECUTABLE_BYTES,
        "python_executable_sha256": PYTHON_EXECUTABLE_SHA256,
        "python_version": PYTHON_VERSION,
        "python_cache_tag": PYTHON_CACHE_TAG,
        "math_module_origin": "built-in",
        "math_module_loader": "BuiltinImporter",
        "math_callable_roster": list(MATH_CALLABLE_NAMES),
        "math_pi_hex": MATH_PI_HEX,
        "math_tau_hex": MATH_TAU_HEX,
        "element_mu_hex": ELEMENT_MU_HEX,
        "angle_bin_width_hex": ANGLE_BIN_WIDTH_HEX,
        "libm": {
            "path": str(LIBM_PATH),
            "byte_count": LIBM_BYTES,
            "sha256": LIBM_SHA256,
            "mapped_device": expected_device,
            "mapped_inode": libm_stat.st_ino,
            "mapped_into_process": True,
        },
        "float_radix": sys.float_info.radix,
        "float_mantissa_bits": sys.float_info.mant_dig,
        "execution_assumption": (
            "FRESH_PROCESS_WITH_NO_HOSTILE_CODE_AND_EMPTY_LD_PRELOAD_LD_AUDIT"
        ),
        "math_libm_custody_scope": (
            "BOOTSTRAP_BUILTIN_OBJECT_IDENTITIES_CONSTANTS_AND_MAPPED_LIBM_FILE;"
            "DYNAMIC_SYMBOL_INTERPOSITION_NOT_FULLY_AUTHENTICATED"
        ),
        "full_interpreter_authentication_claimed": False,
        "classifier_dynamics_executed": False,
    }


def _require_finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ClassifierError(f"{label} must be a finite built-in float")
    return value


def _validate_angle(value: object, label: str, *, inclination: bool = False) -> float:
    number = _require_finite_float(value, label)
    upper = math.pi if inclination else math.tau
    if number < 0.0 or (number > upper if inclination else number >= upper):
        raise ClassifierError(f"{label} is outside its frozen angular domain")
    return number


@dataclass(frozen=True, slots=True)
class ElementSeriesV2:
    body_id: str
    role: str
    sample_kind: str
    clone_index: int | None
    cadence_julian_years: int
    semimajor_axis_AU: tuple[float, ...]
    eccentricity: tuple[float, ...]
    inclination_radians: tuple[float, ...]
    longitude_of_ascending_node_radians: tuple[float, ...]
    argument_of_perihelion_radians: tuple[float, ...]
    mean_anomaly_radians: tuple[float, ...]
    ephemeris: str
    spk_id: int
    upstream_trajectory_sha256: str
    removed: bool = False
    frame: str = ELEMENT_FRAME
    node_order: str = NODE_ORDER
    element_mu: float = ELEMENT_MU
    batch_order: int = 0
    native_epoch_jd_tdb: float = NATIVE_EPOCHS_JD_TDB[0]
    tracer_order: int | None = 0

    def __post_init__(self) -> None:
        if type(self.body_id) is not str or not self.body_id:
            raise ClassifierError("element-series body_id must be a nonempty string")
        if self.role not in ("TNO", "NEPTUNE"):
            raise ClassifierError("element-series role must be TNO or NEPTUNE")
        if self.ephemeris != EPHEMERIS:
            raise ClassifierError("element-series ephemeris differs")
        if type(self.spk_id) is not int:
            raise ClassifierError("element-series SPK identity differs")
        if (
            type(self.upstream_trajectory_sha256) is not str
            or len(self.upstream_trajectory_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.upstream_trajectory_sha256
            )
        ):
            raise ClassifierError("element-series upstream trajectory digest differs")
        if type(self.removed) is not bool or self.removed:
            raise ClassifierError("removed trajectories fail closed")
        if self.frame != ELEMENT_FRAME or self.node_order != NODE_ORDER:
            raise ClassifierError("element-series frame or node order differs")
        if type(self.element_mu) is not float or self.element_mu.hex() != ELEMENT_MU.hex():
            raise ClassifierError("element-series mu differs")
        if (
            type(self.batch_order) is not int
            or not 0 <= self.batch_order < len(OBJECT_IDS)
            or type(self.native_epoch_jd_tdb) is not float
            or self.native_epoch_jd_tdb.hex()
            != NATIVE_EPOCHS_JD_TDB[self.batch_order].hex()
        ):
            raise ClassifierError("element-series batch or native epoch differs")
        if type(self.cadence_julian_years) is not int or self.cadence_julian_years != CADENCE_JULIAN_YEARS:
            raise ClassifierError("element-series cadence differs")
        if self.role == "NEPTUNE":
            if (
                self.body_id != "NEPTUNE_BARYCENTER"
                or self.spk_id != 8
                or self.sample_kind != "MASSIVE_SOURCE"
                or self.clone_index is not None
                or self.tracer_order is not None
            ):
                raise ClassifierError("Neptune element-series identity differs")
        else:
            if self.body_id != OBJECT_IDS[self.batch_order]:
                raise ClassifierError("TNO element-series object/batch identity differs")
            if self.spk_id != SPK_IDS[self.batch_order]:
                raise ClassifierError("TNO element-series SPK/batch identity differs")
            if self.sample_kind == "NOMINAL":
                if (
                    self.clone_index is not None
                    or type(self.tracer_order) is not int
                    or self.tracer_order != 0
                ):
                    raise ClassifierError("nominal trajectory has a clone index")
            elif self.sample_kind == "COVARIANCE_CLONE":
                if (
                    type(self.clone_index) is not int
                    or self.clone_index not in HOLDOUT_CLONE_INDICES
                    or type(self.tracer_order) is not int
                    or not 1 <= self.tracer_order <= len(HOLDOUT_CLONE_INDICES)
                    or HOLDOUT_CLONE_INDICES[self.tracer_order - 1] != self.clone_index
                ):
                    raise ClassifierError("clone trajectory index differs")
            else:
                raise ClassifierError("TNO sample kind differs")
        fields = (
            ("semimajor_axis_AU", self.semimajor_axis_AU),
            ("eccentricity", self.eccentricity),
            ("inclination_radians", self.inclination_radians),
            (
                "longitude_of_ascending_node_radians",
                self.longitude_of_ascending_node_radians,
            ),
            ("argument_of_perihelion_radians", self.argument_of_perihelion_radians),
            ("mean_anomaly_radians", self.mean_anomaly_radians),
        )
        for name, values in fields:
            if type(values) is not tuple or len(values) != NODE_COUNT:
                raise ClassifierError(f"{name} must contain exactly {NODE_COUNT} nodes")
            for index, value in enumerate(values):
                if name == "semimajor_axis_AU":
                    if _require_finite_float(value, f"{name}[{index}]") <= 0.0:
                        raise ClassifierError("semimajor axes must be positive")
                elif name == "eccentricity":
                    number = _require_finite_float(value, f"{name}[{index}]")
                    if not 0.0 <= number < 1.0:
                        raise ClassifierError("only elliptic eccentricities are accepted")
                else:
                    _validate_angle(
                        value,
                        f"{name}[{index}]",
                        inclination=name == "inclination_radians",
                    )


def resonance_band_v2(p: int, q: int) -> str:
    """Return the exact backward-compatible search band for a period pair."""

    if type(p) is not int or type(q) is not int:
        raise ClassifierError("resonance period indices must be exact integers")
    if (
        P_Q_MIN <= p <= INHERITED_P_Q_MAX
        and P_Q_MIN <= q <= INHERITED_P_Q_MAX
        and math.gcd(p, q) == 1
    ):
        return "INHERITED_V1"
    if (
        INHERITED_P_Q_MAX < p <= P_MAX
        and P_Q_MIN <= q <= Q_MAX
        and math.gcd(p, q) == 1
    ):
        return "NEW_P30_TO_P36"
    raise ClassifierError("resonance p:q is outside the frozen v2 union")


def coefficient_limit_v2(p: int, q: int) -> int:
    band = resonance_band_v2(p, q)
    return (
        INHERITED_COEFFICIENT_LIMIT
        if band == "INHERITED_V1"
        else NEW_BAND_COEFFICIENT_LIMIT
    )


def search_domain_summary_v2() -> dict[str, object]:
    pair_counts = {"INHERITED_V1": 0, "NEW_P30_TO_P36": 0}
    argument_counts = {"INHERITED_V1": 0, "NEW_P30_TO_P36": 0}
    for p in range(P_Q_MIN, P_MAX + 1):
        for q in range(P_Q_MIN, Q_MAX + 1):
            if math.gcd(p, q) != 1:
                continue
            band = resonance_band_v2(p, q)
            limit = coefficient_limit_v2(p, q)
            pair_counts[band] += 1
            for m in range(-limit, limit + 1):
                for n in range(-limit, limit + 1):
                    r = q - p - m + n
                    if -limit <= r <= limit:
                        argument_counts[band] += 1
    observed = {
        "inherited_period_pair_count": pair_counts["INHERITED_V1"],
        "new_band_period_pair_count": pair_counts["NEW_P30_TO_P36"],
        "combined_period_pair_count": sum(pair_counts.values()),
        "inherited_argument_count": argument_counts["INHERITED_V1"],
        "new_band_argument_count": argument_counts["NEW_P30_TO_P36"],
        "combined_argument_count": sum(argument_counts.values()),
    }
    expected = {
        "inherited_period_pair_count": EXPECTED_INHERITED_PERIOD_PAIR_UNIVERSE_COUNT,
        "new_band_period_pair_count": EXPECTED_NEW_BAND_PERIOD_PAIR_UNIVERSE_COUNT,
        "combined_period_pair_count": EXPECTED_PERIOD_PAIR_UNIVERSE_COUNT,
        "inherited_argument_count": EXPECTED_INHERITED_ARGUMENT_UNIVERSE_COUNT,
        "new_band_argument_count": EXPECTED_NEW_BAND_ARGUMENT_UNIVERSE_COUNT,
        "combined_argument_count": EXPECTED_ARGUMENT_UNIVERSE_COUNT,
    }
    if observed != expected:
        raise ClassifierError("v2 search-domain universe differs")
    return {
        **observed,
        "inherited_period_pair_domain": {"p": [1, 29], "q": [1, 29]},
        "new_period_pair_domain": {"p": [30, 36], "q": [1, 29]},
        "inherited_coefficient_domain": [-29, 29],
        "new_band_coefficient_domain": [-35, 35],
        "principal_36_to_1_argument": [36, 1, -35, 0, 0, 0],
    }


@dataclass(frozen=True, slots=True)
class PeriodCandidateV2:
    p: int
    q: int
    predicted_period: float
    absolute_mismatch: Fraction

    def __post_init__(self) -> None:
        try:
            resonance_band_v2(self.p, self.q)
        except ClassifierError as exc:
            raise ClassifierError("period candidate p:q is not canonical")
        if _require_finite_float(self.predicted_period, "predicted period") <= 0.0:
            raise ClassifierError("predicted period must be positive")
        if type(self.absolute_mismatch) is not Fraction or self.absolute_mismatch < 0:
            raise ClassifierError("period mismatch must be a nonnegative Fraction")


@dataclass(frozen=True, slots=True)
class ResonanceArgumentV2:
    p: int
    q: int
    m: int
    n: int
    r: int
    s: int
    predicted_period: float
    absolute_period_mismatch: Fraction

    def __post_init__(self) -> None:
        try:
            coefficient_limit = coefficient_limit_v2(self.p, self.q)
        except ClassifierError as exc:
            raise ClassifierError("resonance p:q is not canonical")
        for name, value in (("m", self.m), ("n", self.n), ("r", self.r)):
            if (
                type(value) is not int
                or not -coefficient_limit <= value <= coefficient_limit
            ):
                raise ClassifierError(f"resonance {name} is outside the frozen domain")
        if type(self.s) is not int or self.s != 0:
            raise ClassifierError("resonance s must be exact integer zero")
        if self.p - self.q + self.m - self.n + self.r != 0:
            raise ClassifierError("resonance argument violates the D'Alembert relation")
        if _require_finite_float(self.predicted_period, "predicted period") <= 0.0:
            raise ClassifierError("predicted period must be positive")
        if (
            type(self.absolute_period_mismatch) is not Fraction
            or self.absolute_period_mismatch < 0
        ):
            raise ClassifierError("period mismatch must be a nonnegative Fraction")

    @property
    def coefficient_tuple(self) -> tuple[int, int, int, int, int, int]:
        return (self.p, self.q, self.m, self.n, self.r, self.s)


@dataclass(frozen=True, slots=True)
class ArgumentEvidenceV2:
    argument: ResonanceArgumentV2
    librating_window_bitmap: int
    total_librating_windows: int
    longest_contiguous_librating_windows: int
    windows_evaluated: int

    def __post_init__(self) -> None:
        if type(self.argument) is not ResonanceArgumentV2:
            raise ClassifierError("argument evidence lacks an exact resonance argument")
        if (
            type(self.librating_window_bitmap) is not int
            or not 0 <= self.librating_window_bitmap < (1 << WINDOW_COUNT)
            or type(self.total_librating_windows) is not int
            or type(self.longest_contiguous_librating_windows) is not int
            or type(self.windows_evaluated) is not int
            or not 1 <= self.windows_evaluated <= WINDOW_COUNT
            or not 0 <= self.longest_contiguous_librating_windows <= self.total_librating_windows <= self.windows_evaluated
            or self.librating_window_bitmap >= (1 << self.windows_evaluated)
            or self.librating_window_bitmap.bit_count() != self.total_librating_windows
        ):
            raise ClassifierError("argument evidence counters differ")
        observed_longest = 0
        current = 0
        for index in range(self.windows_evaluated):
            if self.librating_window_bitmap & (1 << index):
                current += 1
                observed_longest = max(observed_longest, current)
            else:
                current = 0
        if observed_longest != self.longest_contiguous_librating_windows:
            raise ClassifierError("argument evidence contiguous run differs")
        if (
            self.longest_contiguous_librating_windows
            >= MINIMUM_CONTIGUOUS_WINDOWS
            and self.windows_evaluated != WINDOW_COUNT
        ):
            raise ClassifierError(
                "resonant evidence must evaluate all windows for the tie-break"
            )

    @property
    def resonant(self) -> bool:
        return self.longest_contiguous_librating_windows >= MINIMUM_CONTIGUOUS_WINDOWS


@dataclass(frozen=True, slots=True)
class TrajectoryClassificationV2:
    object_id: str
    sample_kind: str
    clone_index: int | None
    label: str
    scattering_fraction: Fraction
    initial_semimajor_axis_AU: float
    initial_eccentricity: float
    initial_neptune_semimajor_axis_AU: float
    resonance: ArgumentEvidenceV2 | None

    def __post_init__(self) -> None:
        if type(self.object_id) is not str or not self.object_id:
            raise ClassifierError("classification object_id differs")
        if self.sample_kind == "NOMINAL":
            if self.clone_index is not None:
                raise ClassifierError("nominal classification has a clone index")
        elif self.sample_kind == "COVARIANCE_CLONE":
            if type(self.clone_index) is not int or not 0 <= self.clone_index < 128:
                raise ClassifierError("clone classification index differs")
        else:
            raise ClassifierError("classification sample kind differs")
        if self.label not in ALLOWED_LABELS:
            raise ClassifierError("classification label differs")
        if type(self.scattering_fraction) is not Fraction or self.scattering_fraction < 0:
            raise ClassifierError("scattering fraction differs")
        for label, value in (
            ("initial semimajor axis", self.initial_semimajor_axis_AU),
            ("initial eccentricity", self.initial_eccentricity),
            ("initial Neptune semimajor axis", self.initial_neptune_semimajor_axis_AU),
        ):
            _require_finite_float(value, label)
        if self.initial_semimajor_axis_AU <= 0.0 or self.initial_neptune_semimajor_axis_AU <= 0.0:
            raise ClassifierError("initial semimajor axes must be positive")
        if not 0.0 <= self.initial_eccentricity < 1.0:
            raise ClassifierError("initial eccentricity differs")
        if (self.label == LABEL_RESONANT) != (self.resonance is not None):
            raise ClassifierError("resonance evidence and label differ")
        if self.resonance is not None and not self.resonance.resonant:
            raise ClassifierError("winning resonance evidence is not resonant")


def _rotate_equatorial_to_ecliptic(vector: Sequence[float]) -> tuple[float, float, float]:
    if type(vector) not in (tuple, list) or len(vector) != 3:
        raise ClassifierError("rotation input must be an exact three-vector")
    x, y, z = (_require_finite_float(item, "rotation component") for item in vector)
    return (
        x,
        (COS_OBLIQUITY * y) + (SIN_OBLIQUITY * z),
        ((-SIN_OBLIQUITY) * y) + (COS_OBLIQUITY * z),
    )


def _validate_reduced5_node_identity_v2(
    massive: Sequence[object],
    tracers: Sequence[object],
    batch_order: int,
    native_epoch_jd_tdb: float,
) -> None:
    validate_stage1_source_v2()
    bound = load_bound_stage1_v2()
    if (
        type(batch_order) is not int
        or not 0 <= batch_order < len(OBJECT_IDS)
        or type(native_epoch_jd_tdb) is not float
        or native_epoch_jd_tdb.hex()
        != NATIVE_EPOCHS_JD_TDB[batch_order].hex()
    ):
        raise ClassifierError("element extraction batch or native epoch differs")
    if (
        type(massive) not in (tuple, list)
        or len(massive) != 5
        or any(type(item) is not bound.MassiveState for item in massive)
        or any(
            type(item.naif_id) is not int
            or type(item.body_id) is not str
            or type(item.gm_AU3_per_julian_year2) is not float
            or not math.isfinite(item.gm_AU3_per_julian_year2)
            or item.gm_AU3_per_julian_year2 <= 0.0
            for item in massive
        )
        or tuple(item.naif_id for item in massive) != REDUCED_NAIF_IDS
        or tuple(item.body_id for item in massive) != REDUCED_BODY_IDS
        or tuple(item.gm_AU3_per_julian_year2.hex() for item in massive)
        != REDUCED_GM_HEX
    ):
        raise ClassifierError("reduced-five massive roster identity differs")
    if (
        type(tracers) not in (tuple, list)
        or len(tracers) != 9
        or any(type(item) is not bound.TracerState for item in tracers)
        or any(
            type(item.object_id) is not str
            or type(item.spk_id) is not int
            or type(item.sample_kind) is not str
            or (
                item.clone_index is not None
                and type(item.clone_index) is not int
            )
            for item in tracers
        )
    ):
        raise ClassifierError("classifier tracer roster differs")
    object_id = OBJECT_IDS[batch_order]
    spk_id = SPK_IDS[batch_order]
    if (
        any(item.object_id != object_id or item.spk_id != spk_id for item in tracers)
        or tracers[0].sample_kind != "NOMINAL"
        or tracers[0].clone_index is not None
        or tuple(item.sample_kind for item in tracers[1:])
        != ("COVARIANCE_CLONE",) * len(HOLDOUT_CLONE_INDICES)
        or tuple(item.clone_index for item in tracers[1:])
        != HOLDOUT_CLONE_INDICES
    ):
        raise ClassifierError("classifier tracer identity/order differs")
    for row in (*massive, *tracers):
        for vector, label in (
            (row.position_AU, "position"),
            (row.velocity_AU_per_julian_year, "velocity"),
        ):
            if type(vector) is not tuple or len(vector) != 3:
                raise ClassifierError(f"classifier node {label} shape differs")
            for value in vector:
                _require_finite_float(value, f"classifier node {label}")


def node_state_sha256_v2(
    massive: Sequence[object],
    tracers: Sequence[object],
    batch_order: int,
    native_epoch_jd_tdb: float,
) -> str:
    """Digest one exact reduced-five checkpoint and its full batch identity."""

    _validate_reduced5_node_identity_v2(
        massive,
        tracers,
        batch_order,
        native_epoch_jd_tdb,
    )
    metadata = {
        "ephemeris": EPHEMERIS,
        "batch_order": batch_order,
        "native_epoch_jd_tdb_hex": native_epoch_jd_tdb.hex(),
        "object_id": OBJECT_IDS[batch_order],
        "spk_id": SPK_IDS[batch_order],
        "massive_naif_ids": [item.naif_id for item in massive],
        "massive_body_ids": [item.body_id for item in massive],
        "massive_gm_hex": [
            item.gm_AU3_per_julian_year2.hex() for item in massive
        ],
        "tracer_identity": [
            {
                "object_id": item.object_id,
                "spk_id": item.spk_id,
                "sample_kind": item.sample_kind,
                "clone_index": item.clone_index,
            }
            for item in tracers
        ],
        "binary64_field_order": (
            "MASSIVE_POSITION_VELOCITY_THEN_TRACER_POSITION_VELOCITY"
        ),
    }
    digest = hashlib.sha256()
    digest.update(NODE_STATE_DOMAIN)
    digest.update(_canonical_json(metadata))
    for row in massive:
        for value in (*row.position_AU, *row.velocity_AU_per_julian_year):
            digest.update(struct.pack("<d", value))
    for row in tracers:
        for value in (*row.position_AU, *row.velocity_AU_per_julian_year):
            digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def extract_batch_elements_v2(
    massive: Sequence[object],
    tracers: Sequence[object],
    *,
    batch_order: int,
    native_epoch_jd_tdb: float,
    upstream_node_sha256: str,
) -> dict[str, object]:
    """Extract one reduced-five node through the exact bound Stage-1 code.

    The bound checkpoint receives the metadata sentinel year ``0`` because its
    year field is irrelevant to centering and its original Stage-1 contract is
    limited to 1,000 years.  Physical 10-Myr node times are carried by the
    future element artifact, not passed through that metadata-only field.
    """

    validate_stage1_source_v2()
    bound = load_bound_stage1_v2()
    _validate_reduced5_node_identity_v2(
        massive,
        tracers,
        batch_order,
        native_epoch_jd_tdb,
    )
    object_id = OBJECT_IDS[batch_order]
    spk_id = SPK_IDS[batch_order]
    observed_node_sha256 = node_state_sha256_v2(
        massive,
        tracers,
        batch_order,
        native_epoch_jd_tdb,
    )
    if (
        type(upstream_node_sha256) is not str
        or upstream_node_sha256 != observed_node_sha256
    ):
        raise ClassifierError("retained upstream node digest differs")
    try:
        centered = bound._checkpoint(0, tuple(massive), tuple(tracers))
    except Exception as exc:
        raise ClassifierError("bound Stage-1 checkpoint centering failed") from exc
    positions = centered["massive_positions_AU"]
    velocities = centered["massive_velocities_AU_per_julian_year"]
    tracer_positions = centered["tracer_positions_AU"]
    tracer_velocities = centered["tracer_velocities_AU_per_julian_year"]
    mu = math.fsum(item.gm_AU3_per_julian_year2 for item in massive)
    if mu.hex() != ELEMENT_MU.hex():
        raise ClassifierError("reduced-five element mu differs")

    def extract(position: Sequence[float], velocity: Sequence[float]) -> dict[str, float]:
        rotated_position = _rotate_equatorial_to_ecliptic(position)
        rotated_velocity = _rotate_equatorial_to_ecliptic(velocity)
        try:
            return bound.extract_elliptic_orbital_elements(
                rotated_position, rotated_velocity, mu
            )
        except Exception as exc:
            raise ClassifierError("bound elliptic element extraction failed") from exc

    neptune = extract(positions[4], velocities[4])
    tracer_rows = tuple(
        extract(tracer_positions[index], tracer_velocities[index])
        for index in range(9)
    )
    return {
        "centering_metadata_year_sentinel": 0,
        "physical_time_from_artifact_not_sentinel": True,
        "batch_order": batch_order,
        "native_epoch_jd_tdb_hex": native_epoch_jd_tdb.hex(),
        "object_id": object_id,
        "spk_id": spk_id,
        "ephemeris": EPHEMERIS,
        "upstream_node_sha256": upstream_node_sha256,
        "tracer_order": [
            {
                "tracer_order": index,
                "sample_kind": item.sample_kind,
                "clone_index": item.clone_index,
            }
            for index, item in enumerate(tracers)
        ],
        "frame": ELEMENT_FRAME,
        "element_mu_hex": mu.hex(),
        "neptune": neptune,
        "tracers": tracer_rows,
    }


def _validate_resolved11_batch_identity_v2(batch: object) -> types.ModuleType:
    validate_stage1_source_v2()
    bound = load_bound_stage1_v2()
    if type(batch) is not bound.BatchState:
        raise ClassifierError("initial collapse requires the exact bound BatchState")
    if (
        batch.ephemeris != EPHEMERIS
        or type(batch.batch_order) is not int
        or not 0 <= batch.batch_order < len(OBJECT_IDS)
        or batch.object_id != OBJECT_IDS[batch.batch_order]
        or type(batch.spk_id) is not int
        or batch.spk_id != SPK_IDS[batch.batch_order]
        or type(batch.epoch_jd_tdb) is not float
        or batch.epoch_jd_tdb.hex()
        != NATIVE_EPOCHS_JD_TDB[batch.batch_order].hex()
        or type(batch.massive) is not tuple
        or len(batch.massive) != len(bound.RESOLVED11_NAIF_IDS)
        or any(type(item) is not bound.MassiveState for item in batch.massive)
        or any(
            type(item.naif_id) is not int
            or type(item.body_id) is not str
            or type(item.gm_AU3_per_julian_year2) is not float
            or not math.isfinite(item.gm_AU3_per_julian_year2)
            or item.gm_AU3_per_julian_year2 <= 0.0
            for item in batch.massive
        )
        or tuple(item.naif_id for item in batch.massive)
        != tuple(bound.RESOLVED11_NAIF_IDS)
        or tuple(item.body_id for item in batch.massive)
        != tuple(bound.RESOLVED11_BODY_IDS)
        or tuple(item.gm_AU3_per_julian_year2.hex() for item in batch.massive)
        != RESOLVED11_GM_HEX
        or type(batch.tracers) is not tuple
        or len(batch.tracers) != 9
        or any(type(item) is not bound.TracerState for item in batch.tracers)
        or any(
            type(item.object_id) is not str
            or type(item.spk_id) is not int
            or type(item.sample_kind) is not str
            or (
                item.clone_index is not None
                and type(item.clone_index) is not int
            )
            for item in batch.tracers
        )
        or any(
            item.object_id != batch.object_id or item.spk_id != batch.spk_id
            for item in batch.tracers
        )
        or batch.tracers[0].sample_kind != "NOMINAL"
        or batch.tracers[0].clone_index is not None
        or tuple(item.sample_kind for item in batch.tracers[1:])
        != ("COVARIANCE_CLONE",) * len(HOLDOUT_CLONE_INDICES)
        or tuple(item.clone_index for item in batch.tracers[1:])
        != HOLDOUT_CLONE_INDICES
    ):
        raise ClassifierError("resolved-11 batch identity differs")
    for row in (*batch.massive, *batch.tracers):
        vectors = (
            (row.position_AU, "position"),
            (row.velocity_AU_per_julian_year, "velocity"),
        )
        for vector, label in vectors:
            if type(vector) is not tuple or len(vector) != 3:
                raise ClassifierError(f"resolved-11 batch {label} shape differs")
            for value in vector:
                _require_finite_float(value, f"resolved-11 batch {label}")
    return bound


def batch_state_sha256_v2(batch: object) -> str:
    """Digest a fully validated resolved-11 native-epoch input batch."""

    _validate_resolved11_batch_identity_v2(batch)
    metadata = {
        "ephemeris": batch.ephemeris,
        "batch_order": batch.batch_order,
        "object_id": batch.object_id,
        "spk_id": batch.spk_id,
        "native_epoch_jd_tdb_hex": batch.epoch_jd_tdb.hex(),
        "massive_naif_ids": [item.naif_id for item in batch.massive],
        "massive_body_ids": [item.body_id for item in batch.massive],
        "massive_gm_hex": [
            item.gm_AU3_per_julian_year2.hex() for item in batch.massive
        ],
        "tracer_identity": [
            {
                "object_id": item.object_id,
                "spk_id": item.spk_id,
                "sample_kind": item.sample_kind,
                "clone_index": item.clone_index,
            }
            for item in batch.tracers
        ],
        "binary64_field_order": (
            "MASSIVE_POSITION_VELOCITY_THEN_TRACER_POSITION_VELOCITY"
        ),
    }
    digest = hashlib.sha256()
    digest.update(BATCH_STATE_DOMAIN)
    digest.update(_canonical_json(metadata))
    for row in (*batch.massive, *batch.tracers):
        for value in (*row.position_AU, *row.velocity_AU_per_julian_year):
            digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def validate_frozen_de441_initial_batches_v2() -> dict[str, object]:
    """Rebuild and bind all twelve initial batches from retained custody."""

    bound = load_bound_stage1_v2()
    if not callable(getattr(bound, "_verify_input_documents", None)) or not callable(
        getattr(bound, "_batch_states", None)
    ):
        raise ClassifierError("bound Stage-1 input reconstruction surface differs")
    try:
        bridge, custody, _bindings = bound._verify_input_documents()
        batches = bound._batch_states(bridge, custody, EPHEMERIS)
    except Exception as exc:
        raise ClassifierError("retained DE441 input reconstruction failed") from exc
    observed = tuple(batch_state_sha256_v2(batch) for batch in batches)
    if observed != BRIDGE_BATCH_STATE_SHA256:
        raise ClassifierError("retained DE441 initial batch roster differs")
    return {
        "ephemeris": EPHEMERIS,
        "batch_count": len(observed),
        "object_ids_in_exact_order": list(OBJECT_IDS),
        "spk_ids_in_exact_order": list(SPK_IDS),
        "predecessor_v1_batch_state_sha256_in_exact_order": list(
            BRIDGE_BATCH_STATE_SHA256_V1
        ),
        "batch_state_sha256_in_exact_order": list(observed),
    }


def collapse_initial_resolved11_v2(
    batch: object, *, upstream_batch_sha256: str
) -> object:
    """Collapse a resolved-11 initial batch exactly once through Stage-1.

    Integrated checkpoints are already reduced-five and must go directly to
    :func:`extract_batch_elements_v2`; repeatedly folding them is rejected by
    the bound Stage-1 roster check.
    """

    validate_stage1_source_v2()
    bound = _validate_resolved11_batch_identity_v2(batch)
    observed_batch_sha256 = batch_state_sha256_v2(batch)
    expected_batch_sha256 = BRIDGE_BATCH_STATE_SHA256[batch.batch_order]
    if (
        type(upstream_batch_sha256) is not str
        or upstream_batch_sha256 != expected_batch_sha256
        or observed_batch_sha256 != expected_batch_sha256
    ):
        raise ClassifierError(
            "resolved-11 batch differs from the frozen DE441 bridge input"
        )
    try:
        reduced = bound.collapse_reduced_model(batch)
    except Exception as exc:
        raise ClassifierError("bound initial resolved-11 collapse failed") from exc
    _validate_reduced5_node_identity_v2(
        reduced.massive,
        reduced.tracers,
        reduced.batch_order,
        reduced.epoch_jd_tdb,
    )
    if (
        reduced.ephemeris != EPHEMERIS
        or reduced.batch_order != batch.batch_order
        or reduced.object_id != batch.object_id
        or reduced.spk_id != batch.spk_id
        or reduced.epoch_jd_tdb.hex() != batch.epoch_jd_tdb.hex()
    ):
        raise ClassifierError("bound initial collapse produced a different roster")
    return reduced


def bound_angle_v2(value: float) -> float:
    number = _require_finite_float(value, "angle input")
    validate_stage1_source_v2()
    try:
        result = load_bound_stage1_v2()._angle(number)
    except Exception as exc:
        raise ClassifierError("bound angle reduction failed") from exc
    if type(result) is not float or not math.isfinite(result) or not 0.0 <= result < math.tau:
        raise ClassifierError("bound angle reduction left the half-open domain")
    return result


def orbital_period_v2(semimajor_axis_AU: float, element_mu: float) -> float:
    a = _require_finite_float(semimajor_axis_AU, "semimajor axis")
    mu = _require_finite_float(element_mu, "element mu")
    if a <= 0.0 or mu <= 0.0:
        raise ClassifierError("period inputs must be positive")
    result = (2.0 * math.pi) * math.sqrt(((a * a) * a) / mu)
    if not math.isfinite(result) or result <= 0.0:
        raise ClassifierError("period evaluation failed")
    return result


def candidate_period_pairs_v2(
    tno_period: float, neptune_period: float
) -> tuple[PeriodCandidateV2, ...]:
    tno = _require_finite_float(tno_period, "TNO period")
    neptune = _require_finite_float(neptune_period, "Neptune period")
    if tno <= 0.0 or neptune <= 0.0:
        raise ClassifierError("candidate periods must be positive")
    threshold = PERIOD_FRACTION * Fraction.from_float(neptune)
    rows: list[PeriodCandidateV2] = []
    for p in range(P_Q_MIN, P_MAX + 1):
        for q in range(P_Q_MIN, Q_MAX + 1):
            if math.gcd(p, q) != 1:
                continue
            predicted = (float(p) / float(q)) * neptune
            mismatch = abs(Fraction.from_float(predicted) - Fraction.from_float(tno))
            if mismatch < threshold:
                rows.append(PeriodCandidateV2(p, q, predicted, mismatch))
    rows.sort(key=lambda row: (row.absolute_mismatch, row.p + row.q, row.p, row.q))
    return tuple(rows)


def enumerate_resonance_arguments_v2(
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    element_mu: float,
) -> Iterator[ResonanceArgumentV2]:
    _require_series_pair(tno, neptune)
    mu = _require_finite_float(element_mu, "element mu")
    if mu.hex() != tno.element_mu.hex():
        raise ClassifierError("classification element mu differs from the series")
    tno_period = orbital_period_v2(tno.semimajor_axis_AU[0], mu)
    neptune_period = orbital_period_v2(neptune.semimajor_axis_AU[0], mu)
    for candidate in candidate_period_pairs_v2(tno_period, neptune_period):
        coefficient_limit = coefficient_limit_v2(candidate.p, candidate.q)
        for m in range(-coefficient_limit, coefficient_limit + 1):
            for n in range(-coefficient_limit, coefficient_limit + 1):
                r = candidate.q - candidate.p - m + n
                if -coefficient_limit <= r <= coefficient_limit:
                    yield ResonanceArgumentV2(
                        candidate.p,
                        candidate.q,
                        m,
                        n,
                        r,
                        0,
                        candidate.predicted_period,
                        candidate.absolute_mismatch,
                    )


def _prepared_longitudes(
    series: ElementSeriesV2,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    varpi: list[float] = []
    longitude: list[float] = []
    validate_stage1_source_v2()
    bound = load_bound_stage1_v2()._angle
    for index in range(RESONANCE_NODE_COUNT):
        node = series.longitude_of_ascending_node_radians[index]
        peri = series.argument_of_perihelion_radians[index]
        mean = series.mean_anomaly_radians[index]
        varpi.append(bound(node + peri))
        longitude.append(bound((node + peri) + mean))
    return tuple(longitude), tuple(varpi)


def _require_series_pair(tno: object, neptune: object) -> None:
    if type(tno) is not ElementSeriesV2 or tno.role != "TNO":
        raise ClassifierError("resonance TNO series differs")
    if type(neptune) is not ElementSeriesV2 or neptune.role != "NEPTUNE":
        raise ClassifierError("resonance Neptune series differs")
    if (
        tno.frame != neptune.frame
        or tno.node_order != neptune.node_order
        or tno.cadence_julian_years != neptune.cadence_julian_years
        or tno.element_mu.hex() != neptune.element_mu.hex()
        or tno.batch_order != neptune.batch_order
        or tno.native_epoch_jd_tdb.hex() != neptune.native_epoch_jd_tdb.hex()
        or tno.ephemeris != neptune.ephemeris
        or tno.upstream_trajectory_sha256
        != neptune.upstream_trajectory_sha256
    ):
        raise ClassifierError("TNO and Neptune element-series contracts differ")


def _critical_angle_prepared(
    argument: ResonanceArgumentV2,
    index: int,
    tno_longitude: Sequence[float],
    tno_varpi: Sequence[float],
    tno_node: Sequence[float],
    neptune_longitude: Sequence[float],
    neptune_varpi: Sequence[float],
) -> float:
    value = math.fsum(
        (
            argument.p * tno_longitude[index],
            -argument.q * neptune_longitude[index],
            argument.m * tno_varpi[index],
            -argument.n * neptune_varpi[index],
            argument.r * tno_node[index],
        )
    )
    return load_bound_stage1_v2()._angle(value)


def critical_angle_v2(
    argument: ResonanceArgumentV2,
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    index: int,
) -> float:
    if type(index) is not int or not 0 <= index < RESONANCE_NODE_COUNT:
        raise ClassifierError("critical-angle node index differs")
    _require_series_pair(tno, neptune)
    tno_longitude, tno_varpi = _prepared_longitudes(tno)
    neptune_longitude, neptune_varpi = _prepared_longitudes(neptune)
    return _critical_angle_prepared(
        argument,
        index,
        tno_longitude,
        tno_varpi,
        tno.longitude_of_ascending_node_radians,
        neptune_longitude,
        neptune_varpi,
    )


def angle_bin_index_v2(angle: float) -> int:
    sigma = bound_angle_v2(_require_finite_float(angle, "critical angle"))
    result = min(ANGLE_BIN_COUNT - 1, int(sigma / ANGLE_BIN_WIDTH))
    if not 0 <= result < ANGLE_BIN_COUNT:
        raise ClassifierError("critical-angle bin index differs")
    return result


def window_librates_v2(angles: Sequence[float]) -> bool:
    if type(angles) not in (tuple, list) or len(angles) != POINTS_PER_WINDOW:
        raise ClassifierError("resonance window must contain exactly 1,000 points")
    validated = tuple(_require_finite_float(value, "critical angle") for value in angles)
    occupied = 0
    for value in validated:
        occupied |= 1 << angle_bin_index_v2(value)
        if occupied.bit_count() > ANGLE_BIN_COUNT - MINIMUM_EMPTY_BINS:
            return False
    return True


def _evaluate_argument_prepared(
    argument: ResonanceArgumentV2,
    tno: ElementSeriesV2,
    tno_longitude: Sequence[float],
    tno_varpi: Sequence[float],
    neptune_longitude: Sequence[float],
    neptune_varpi: Sequence[float],
) -> ArgumentEvidenceV2:
    bitmap = 0
    total = 0
    current_run = 0
    longest_run = 0
    for window in range(WINDOW_COUNT):
        occupied = 0
        start = window * POINTS_PER_WINDOW
        stop = start + POINTS_PER_WINDOW
        for index in range(start, stop):
            sigma = _critical_angle_prepared(
                argument,
                index,
                tno_longitude,
                tno_varpi,
                tno.longitude_of_ascending_node_radians,
                neptune_longitude,
                neptune_varpi,
            )
            occupied |= 1 << min(
                ANGLE_BIN_COUNT - 1, int(sigma / ANGLE_BIN_WIDTH)
            )
            if occupied.bit_count() > ANGLE_BIN_COUNT - MINIMUM_EMPTY_BINS:
                break
        librating = occupied.bit_count() <= ANGLE_BIN_COUNT - MINIMUM_EMPTY_BINS
        if librating:
            bitmap |= 1 << window
            total += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
        remaining = WINDOW_COUNT - window - 1
        if max(longest_run, current_run + remaining) < MINIMUM_CONTIGUOUS_WINDOWS:
            break
    return ArgumentEvidenceV2(argument, bitmap, total, longest_run, window + 1)


def evaluate_argument_v2(
    argument: ResonanceArgumentV2,
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
) -> ArgumentEvidenceV2:
    if type(argument) is not ResonanceArgumentV2:
        raise ClassifierError("resonance argument type differs")
    _require_series_pair(tno, neptune)
    tno_longitude, tno_varpi = _prepared_longitudes(tno)
    neptune_longitude, neptune_varpi = _prepared_longitudes(neptune)
    return _evaluate_argument_prepared(
        argument,
        tno,
        tno_longitude,
        tno_varpi,
        neptune_longitude,
        neptune_varpi,
    )


def _winner_key(evidence: ArgumentEvidenceV2) -> tuple[object, ...]:
    argument = evidence.argument
    return (
        -evidence.longest_contiguous_librating_windows,
        -evidence.total_librating_windows,
        argument.absolute_period_mismatch,
        argument.p + argument.q,
        argument.p,
        argument.q,
        abs(argument.m) + abs(argument.n) + abs(argument.r),
        argument.m,
        argument.n,
        argument.r,
    )


def select_winning_evidence_v2(
    evidence_rows: Iterable[ArgumentEvidenceV2],
) -> ArgumentEvidenceV2 | None:
    rows = tuple(evidence_rows)
    if any(type(row) is not ArgumentEvidenceV2 for row in rows):
        raise ClassifierError("winning-evidence roster contains a wrong type")
    resonant = tuple(row for row in rows if row.resonant)
    return min(resonant, key=_winner_key) if resonant else None


def select_resonance_v2(
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    arguments: Iterable[ResonanceArgumentV2],
) -> ArgumentEvidenceV2 | None:
    _require_series_pair(tno, neptune)
    tno_longitude, tno_varpi = _prepared_longitudes(tno)
    neptune_longitude, neptune_varpi = _prepared_longitudes(neptune)
    seen: set[tuple[int, int, int, int, int, int]] = set()
    winner: ArgumentEvidenceV2 | None = None
    for argument in arguments:
        if type(argument) is not ResonanceArgumentV2:
            raise ClassifierError("resonance argument roster contains a wrong type")
        if argument.coefficient_tuple in seen:
            raise ClassifierError("resonance argument roster contains a duplicate")
        seen.add(argument.coefficient_tuple)
        evidence = _evaluate_argument_prepared(
            argument,
            tno,
            tno_longitude,
            tno_varpi,
            neptune_longitude,
            neptune_varpi,
        )
        if evidence.resonant and (winner is None or _winner_key(evidence) < _winner_key(winner)):
            winner = evidence
    return winner


def scattering_fraction_v2(semimajor_axes_AU: Sequence[float]) -> Fraction:
    if type(semimajor_axes_AU) is not tuple or len(semimajor_axes_AU) != NODE_COUNT:
        raise ClassifierError("scattering series must be the exact 50,001-node tuple")
    initial = _require_finite_float(semimajor_axes_AU[0], "initial semimajor axis")
    if initial <= 0.0:
        raise ClassifierError("initial semimajor axis must be positive")
    initial_exact = Fraction.from_float(initial)
    maximum = Fraction(0, 1)
    for index, value in enumerate(semimajor_axes_AU):
        number = _require_finite_float(value, f"semimajor axis[{index}]")
        if number <= 0.0:
            raise ClassifierError("semimajor axes must be positive")
        delta = abs(Fraction.from_float(number) - initial_exact) / initial_exact
        maximum = max(maximum, delta)
    return maximum


def classify_nonresonant_metrics_v2(
    initial_semimajor_axis_AU: float,
    initial_eccentricity: float,
    initial_neptune_semimajor_axis_AU: float,
    scattering_fraction: Fraction,
) -> str:
    a0 = _require_finite_float(initial_semimajor_axis_AU, "initial semimajor axis")
    e0 = _require_finite_float(initial_eccentricity, "initial eccentricity")
    neptune_a0 = _require_finite_float(
        initial_neptune_semimajor_axis_AU, "initial Neptune semimajor axis"
    )
    if a0 <= 0.0 or neptune_a0 <= 0.0 or not 0.0 <= e0 < 1.0:
        raise ClassifierError("nonresonant initial elements differ")
    if type(scattering_fraction) is not Fraction or scattering_fraction < 0:
        raise ClassifierError("scattering fraction differs")
    if Fraction.from_float(a0) > OORT_THRESHOLD:
        return LABEL_OORT
    if Fraction.from_float(a0) < Fraction.from_float(neptune_a0):
        return LABEL_CENTAUR
    if scattering_fraction > SCATTERING_THRESHOLD:
        return LABEL_SCATTERING
    if Fraction.from_float(e0) > DETACHED_THRESHOLD:
        return LABEL_DETACHED
    return LABEL_CLASSICAL


def classify_trajectory_v2(
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    element_mu: float,
) -> TrajectoryClassificationV2:
    _require_series_pair(tno, neptune)
    mu = _require_finite_float(element_mu, "element mu")
    if mu.hex() != tno.element_mu.hex():
        raise ClassifierError("classification element mu differs from the series")
    scattering = scattering_fraction_v2(tno.semimajor_axis_AU)
    winner = select_resonance_v2(
        tno,
        neptune,
        enumerate_resonance_arguments_v2(tno, neptune, mu),
    )
    label = (
        LABEL_RESONANT
        if winner is not None
        else classify_nonresonant_metrics_v2(
            tno.semimajor_axis_AU[0],
            tno.eccentricity[0],
            neptune.semimajor_axis_AU[0],
            scattering,
        )
    )
    return TrajectoryClassificationV2(
        tno.body_id,
        tno.sample_kind,
        tno.clone_index,
        label,
        scattering,
        tno.semimajor_axis_AU[0],
        tno.eccentricity[0],
        neptune.semimajor_axis_AU[0],
        winner,
    )


def _fraction_record(value: Fraction) -> dict[str, int]:
    if type(value) is not Fraction or value < 0:
        raise ClassifierError("fraction record requires a nonnegative Fraction")
    return {"numerator": value.numerator, "denominator": value.denominator}


def element_series_sha256_v2(series: ElementSeriesV2) -> str:
    """Digest exact metadata and every retained binary64 element node."""

    if type(series) is not ElementSeriesV2:
        raise ClassifierError("element-series digest input type differs")
    fields = (
        ("semimajor_axis_AU", series.semimajor_axis_AU),
        ("eccentricity", series.eccentricity),
        ("inclination_radians", series.inclination_radians),
        (
            "longitude_of_ascending_node_radians",
            series.longitude_of_ascending_node_radians,
        ),
        ("argument_of_perihelion_radians", series.argument_of_perihelion_radians),
        ("mean_anomaly_radians", series.mean_anomaly_radians),
    )
    metadata = {
        "body_id": series.body_id,
        "role": series.role,
        "sample_kind": series.sample_kind,
        "clone_index": series.clone_index,
        "ephemeris": series.ephemeris,
        "spk_id": series.spk_id,
        "upstream_trajectory_sha256": series.upstream_trajectory_sha256,
        "batch_order": series.batch_order,
        "native_epoch_jd_tdb_hex": series.native_epoch_jd_tdb.hex(),
        "tracer_order": series.tracer_order,
        "cadence_julian_years": series.cadence_julian_years,
        "node_count": NODE_COUNT,
        "frame": series.frame,
        "node_order": series.node_order,
        "element_mu_hex": series.element_mu.hex(),
        "removed": series.removed,
        "binary64_byte_order": "LITTLE_ENDIAN_IEEE754",
        "binary_field_order": [name for name, _values in fields],
    }
    digest = hashlib.sha256()
    digest.update(ELEMENT_SERIES_DOMAIN)
    digest.update(_canonical_json(metadata))
    for _name, values in fields:
        for value in values:
            digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def _roster_digest_v2(
    domain: bytes, rows: Iterable[Mapping[str, object]]
) -> tuple[int, str]:
    digest = hashlib.sha256()
    digest.update(domain)
    count = 0
    for row in rows:
        digest.update(_canonical_json(dict(row)))
        count += 1
    return count, digest.hexdigest()


def _split_roster_digest_v2(
    combined_domain: bytes,
    inherited_domain: bytes,
    new_band_domain: bytes,
    rows: Iterable[Mapping[str, object]],
) -> dict[str, object]:
    digests = {
        "combined": hashlib.sha256(combined_domain),
        "inherited": hashlib.sha256(inherited_domain),
        "new_band": hashlib.sha256(new_band_domain),
    }
    counts = {"combined": 0, "inherited": 0, "new_band": 0}
    for source in rows:
        row = dict(source)
        band = row.get("band")
        if band == "INHERITED_V1":
            key = "inherited"
        elif band == "NEW_P30_TO_P36":
            key = "new_band"
        else:
            raise ClassifierError("search roster row has an invalid band")
        payload = _canonical_json(row)
        digests["combined"].update(payload)
        digests[key].update(payload)
        counts["combined"] += 1
        counts[key] += 1
    return {
        "combined_count": counts["combined"],
        "combined_sha256": digests["combined"].hexdigest(),
        "inherited_count": counts["inherited"],
        "inherited_sha256": digests["inherited"].hexdigest(),
        "new_band_count": counts["new_band"],
        "new_band_sha256": digests["new_band"].hexdigest(),
    }


def search_roster_identity_v2(
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    element_mu: float,
) -> dict[str, object]:
    """Bind the exact period candidates and complete argument search roster."""

    _require_series_pair(tno, neptune)
    mu = _require_finite_float(element_mu, "element mu")
    if mu.hex() != ELEMENT_MU.hex():
        raise ClassifierError("search-roster element mu differs")
    tno_period = orbital_period_v2(tno.semimajor_axis_AU[0], mu)
    neptune_period = orbital_period_v2(neptune.semimajor_axis_AU[0], mu)
    candidates = candidate_period_pairs_v2(tno_period, neptune_period)
    candidate_identity = _split_roster_digest_v2(
        CANDIDATE_ROSTER_DOMAIN,
        INHERITED_CANDIDATE_ROSTER_DOMAIN,
        NEW_BAND_CANDIDATE_ROSTER_DOMAIN,
        (
            {
                "band": resonance_band_v2(row.p, row.q),
                "p": row.p,
                "q": row.q,
                "predicted_period_hex": row.predicted_period.hex(),
                "absolute_mismatch": _fraction_record(row.absolute_mismatch),
            }
            for row in candidates
        ),
    )
    argument_identity = _split_roster_digest_v2(
        ARGUMENT_ROSTER_DOMAIN,
        INHERITED_ARGUMENT_ROSTER_DOMAIN,
        NEW_BAND_ARGUMENT_ROSTER_DOMAIN,
        (
            {
                "band": resonance_band_v2(row.p, row.q),
                "coefficients": list(row.coefficient_tuple),
                "predicted_period_hex": row.predicted_period.hex(),
                "absolute_period_mismatch": _fraction_record(
                    row.absolute_period_mismatch
                ),
            }
            for row in enumerate_resonance_arguments_v2(tno, neptune, mu)
        ),
    )
    return {
        "element_mu_hex": mu.hex(),
        "initial_tno_period_hex": tno_period.hex(),
        "initial_neptune_period_hex": neptune_period.hex(),
        "search_union": {
            "inherited_period_pair_domain": [1, 29, 1, 29],
            "new_period_pair_domain": [30, 36, 1, 29],
            "inherited_coefficient_limit": INHERITED_COEFFICIENT_LIMIT,
            "new_band_coefficient_limit": NEW_BAND_COEFFICIENT_LIMIT,
        },
        "candidate_count": candidate_identity["combined_count"],
        "candidate_roster_sha256": candidate_identity["combined_sha256"],
        "inherited_candidate_count": candidate_identity["inherited_count"],
        "inherited_candidate_roster_sha256": candidate_identity[
            "inherited_sha256"
        ],
        "new_band_candidate_count": candidate_identity["new_band_count"],
        "new_band_candidate_roster_sha256": candidate_identity[
            "new_band_sha256"
        ],
        "argument_count": argument_identity["combined_count"],
        "argument_roster_sha256": argument_identity["combined_sha256"],
        "inherited_argument_count": argument_identity["inherited_count"],
        "inherited_argument_roster_sha256": argument_identity[
            "inherited_sha256"
        ],
        "new_band_argument_count": argument_identity["new_band_count"],
        "new_band_argument_roster_sha256": argument_identity[
            "new_band_sha256"
        ],
    }


def _argument_record(evidence: ArgumentEvidenceV2 | None) -> dict[str, object] | None:
    if evidence is None:
        return None
    argument = evidence.argument
    return {
        "coefficients": list(argument.coefficient_tuple),
        "predicted_period_hex": argument.predicted_period.hex(),
        "absolute_period_mismatch": _fraction_record(argument.absolute_period_mismatch),
        "librating_window_bitmap_hex": hex(evidence.librating_window_bitmap),
        "total_librating_windows": evidence.total_librating_windows,
        "longest_contiguous_librating_windows": (
            evidence.longest_contiguous_librating_windows
        ),
        "windows_evaluated": evidence.windows_evaluated,
    }


def _classification_record_from_replayed_result_v2(
    result: TrajectoryClassificationV2,
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    element_mu: float,
) -> dict[str, object]:
    if type(result) is not TrajectoryClassificationV2:
        raise ClassifierError("classification record input type differs")
    _require_series_pair(tno, neptune)
    mu = _require_finite_float(element_mu, "classification record element mu")
    if mu.hex() != ELEMENT_MU.hex():
        raise ClassifierError("classification record element mu differs")
    preregistration = validate_preregistration_v2()
    runtime = runtime_identity_v2()
    record: dict[str, object] = {
        "schema": SCHEMA,
        "classifier": CLASSIFIER_NAME,
        "implementation": implementation_identity_v2(),
        "successor_basis": validate_successor_basis_v2(),
        "preregistration": preregistration,
        "runtime": runtime,
        "stage1_source": validate_stage1_source_v2(),
        "input_identity": {
            "batch_order": tno.batch_order,
            "native_epoch_jd_tdb_hex": tno.native_epoch_jd_tdb.hex(),
            "object_id": tno.body_id,
            "spk_id": tno.spk_id,
            "ephemeris": tno.ephemeris,
            "upstream_trajectory_sha256": tno.upstream_trajectory_sha256,
            "sample_kind": tno.sample_kind,
            "clone_index": tno.clone_index,
            "tracer_order": tno.tracer_order,
            "frame": tno.frame,
            "node_order": tno.node_order,
            "cadence_julian_years": tno.cadence_julian_years,
            "node_count": NODE_COUNT,
            "element_mu_hex": mu.hex(),
            "tno_element_series_sha256": element_series_sha256_v2(tno),
            "neptune_element_series_sha256": element_series_sha256_v2(neptune),
        },
        "search_identity": search_roster_identity_v2(tno, neptune, mu),
        "object_id": result.object_id,
        "sample_kind": result.sample_kind,
        "clone_index": result.clone_index,
        "label": result.label,
        "scattering_fraction": _fraction_record(result.scattering_fraction),
        "initial_semimajor_axis_AU_hex": result.initial_semimajor_axis_AU.hex(),
        "initial_eccentricity_hex": result.initial_eccentricity.hex(),
        "initial_neptune_semimajor_axis_AU_hex": (
            result.initial_neptune_semimajor_axis_AU.hex()
        ),
        "resonance": _argument_record(result.resonance),
        "claims": {
            "mechanical_classifier_output": True,
            "search_domain_includes_36_to_1": True,
            "v1_search_band_preserved_exactly": True,
            "upstream_trajectory_digest_retained": True,
            "upstream_trajectory_authenticated_against_external_artifact": False,
            "exact_bernardinelli_2022_reproduction": False,
            "source_backed_labelled_trajectory_fixture_passed": False,
            "independent_classifier_crosscheck_passed": False,
            "classifier_scientifically_qualified": False,
            "object_classification_authorized": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
    }
    record["content_sha256"] = _content_sha256(record)
    return record


def classification_record_v2(
    result: TrajectoryClassificationV2,
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    element_mu: float,
) -> dict[str, object]:
    """Replay the trajectory before serializing its classification."""

    expected = classify_trajectory_v2(tno, neptune, element_mu)
    if result != expected:
        raise ClassifierError("classification result differs from replayed trajectory")
    return _classification_record_from_replayed_result_v2(
        result,
        tno,
        neptune,
        element_mu,
    )


def validate_classifier_result_v2(
    value: Mapping[str, Any],
    tno: ElementSeriesV2,
    neptune: ElementSeriesV2,
    element_mu: float,
) -> str:
    """Replay the full classifier and require byte-exact record equality."""

    if type(value) is not dict:
        raise ClassifierError("classification record must be an exact dictionary")
    content_sha256 = value.get("content_sha256")
    if (
        type(content_sha256) is not str
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
        or content_sha256 != _content_sha256(value)
    ):
        raise ClassifierError("classification record content digest differs")
    expected_result = classify_trajectory_v2(tno, neptune, element_mu)
    expected_record = _classification_record_from_replayed_result_v2(
        expected_result,
        tno,
        neptune,
        element_mu,
    )
    if _canonical_json(value) != _canonical_json(expected_record):
        raise ClassifierError(
            "classification record differs from full trajectory replay"
        )
    return content_sha256


def _outcome_key(result: TrajectoryClassificationV2) -> tuple[object, ...]:
    if result.resonance is None:
        return (result.label,)
    return (result.label, *result.resonance.argument.coefficient_tuple)


def summarize_nine_state_object_v2(
    rows: Sequence[TrajectoryClassificationV2],
) -> dict[str, object]:
    if type(rows) not in (tuple, list) or len(rows) != 9:
        raise ClassifierError("nine-state summary requires exactly nine rows")
    if any(type(row) is not TrajectoryClassificationV2 for row in rows):
        raise ClassifierError("nine-state summary row type differs")
    object_id = rows[0].object_id
    if (
        rows[0].sample_kind != "NOMINAL"
        or rows[0].clone_index is not None
        or tuple(row.clone_index for row in rows[1:]) != HOLDOUT_CLONE_INDICES
        or any(row.sample_kind != "COVARIANCE_CLONE" for row in rows[1:])
        or any(row.object_id != object_id for row in rows)
    ):
        raise ClassifierError("nine-state identity/order differs")
    label_counts = Counter(row.label for row in rows)
    outcome_counts = Counter(_outcome_key(row) for row in rows)
    nominal_key = _outcome_key(rows[0])
    return {
        "object_id": object_id,
        "state_count": 9,
        "labels": {key: label_counts[key] for key in sorted(label_counts)},
        "nominal_exact_outcome_support": outcome_counts[nominal_key],
        "all_nine_exact_outcomes_match": outcome_counts[nominal_key] == 9,
        "object_classification_authorized": False,
        "interpretation": "NINE_STATE_HOLDOUT_AGREEMENT_DIAGNOSTIC_ONLY",
    }


def evaluate_future_128_clone_rule_v2(
    nominal: TrajectoryClassificationV2,
    clones: Sequence[TrajectoryClassificationV2],
) -> dict[str, object]:
    if type(nominal) is not TrajectoryClassificationV2 or nominal.sample_kind != "NOMINAL":
        raise ClassifierError("future rule nominal identity differs")
    if type(clones) not in (tuple, list) or len(clones) != 128:
        raise ClassifierError("future rule requires exactly 128 clones")
    if any(type(row) is not TrajectoryClassificationV2 for row in clones):
        raise ClassifierError("future rule clone type differs")
    if (
        tuple(row.clone_index for row in clones) != tuple(range(128))
        or any(row.sample_kind != "COVARIANCE_CLONE" for row in clones)
        or any(row.object_id != nominal.object_id for row in clones)
    ):
        raise ClassifierError("future rule clone identity/order differs")
    nominal_key = _outcome_key(nominal)
    support = sum(_outcome_key(row) == nominal_key for row in clones)
    return {
        "object_id": nominal.object_id,
        "clone_denominator": 128,
        "minimum_same_exact_outcome": 123,
        "same_exact_outcome_support": support,
        "nominal_matches_supported_outcome": True,
        "jx_robust_rule_passed": support >= 123,
        "used_by_current_108_state_pilot": False,
        "scientific_probability_interpretation": False,
    }


@dataclass(frozen=True, slots=True)
class PublishedCatalogLabelV2:
    designation: str
    label: str
    notes: str
    record_index: int

    def __post_init__(self) -> None:
        if type(self.designation) is not str or not self.designation:
            raise ClassifierError("catalog designation differs")
        if type(self.label) is not str or not self.label:
            raise ClassifierError("catalog label differs")
        if type(self.notes) is not str or type(self.record_index) is not int or self.record_index < 0:
            raise ClassifierError("catalog row identity differs")


DES_CUSTODY_EXPECTED = {
    "2012 WG37": ("Scattering", ""),
    "2013 RP98": ("Classical", ""),
    "2014 RV86": ("Resonant 8:1", ""),
    "2014 RL86": ("Resonant 25:3", ""),
    "2014 SQ403": ("Resonant candidate 10:1", ""),
    "2013 TK227": ("Resonant 1:1", "1,1,-2,0,1,0"),
    "2014 SC374": ("Resonant 1:1", "1,1,-3,0,0,0"),
}


def load_des_y6_label_custody_v2() -> dict[str, PublishedCatalogLabelV2]:
    catalog = _require_file(
        DES_CATALOG_PATH,
        DES_CATALOG_BYTES,
        DES_CATALOG_SHA256,
        "DES Y6 CDS catalog",
    )
    readme = _require_file(
        DES_README_PATH,
        DES_README_BYTES,
        DES_README_SHA256,
        "DES Y6 CDS readme",
    )
    if b"573- 596 A24" not in readme or b"598- 611 A14" not in readme:
        raise ClassifierError("DES fixed-width label schema differs")
    lines = catalog.splitlines()
    if len(lines) != 814 or any(len(line) != 611 for line in lines):
        raise ClassifierError("DES fixed-width row roster differs")
    parsed: dict[str, PublishedCatalogLabelV2] = {}
    for index, line in enumerate(lines):
        try:
            designation = line[0:12].decode("ascii").strip()
            label = line[572:596].decode("ascii").strip()
            delimiter = line[596:597].decode("ascii")
            notes = line[597:611].decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ClassifierError("DES catalog contains non-ASCII text") from exc
        if not designation or not label or delimiter != "|" or designation in parsed:
            raise ClassifierError("DES fixed-width row failed closed")
        parsed[designation] = PublishedCatalogLabelV2(
            designation, label, notes, index
        )
    for designation, expected in DES_CUSTODY_EXPECTED.items():
        observed = parsed.get(designation)
        if observed is None or (observed.label, observed.notes) != expected:
            raise ClassifierError("DES source-backed example label differs")
    return parsed


def binding_report_v2() -> dict[str, object]:
    stage1_source = validate_stage1_source_v2()
    bound = load_bound_stage1_v2()
    labels = load_des_y6_label_custody_v2()
    return {
        "classifier": CLASSIFIER_NAME,
        "implementation": implementation_identity_v2(),
        "successor_basis": validate_successor_basis_v2(),
        "search_domain": search_domain_summary_v2(),
        "preregistration": validate_preregistration_v2(),
        "frozen_de441_initial_batches": validate_frozen_de441_initial_batches_v2(),
        "stage1_source": stage1_source,
        "stage1_source_sha256": STAGE1_SOURCE_SHA256,
        "bound_angle_callable": bound._angle.__name__,
        "bound_element_callable": bound.extract_elliptic_orbital_elements.__name__,
        "runtime": runtime_identity_v2(),
        "des_catalog_sha256": DES_CATALOG_SHA256,
        "des_readme_sha256": DES_README_SHA256,
        "des_record_count": len(labels),
        "source_backed_example_labels": {
            key: {"label": labels[key].label, "notes": labels[key].notes}
            for key in sorted(DES_CUSTODY_EXPECTED)
        },
        "claims": {
            "mechanical_classifier_implemented": True,
            "search_domain_includes_36_to_1": True,
            "v1_search_band_preserved_exactly": True,
            "frozen_de441_initial_batches_reconstructed": True,
            "future_10my_trajectory_artifact_authenticated": False,
            "des_label_parser_and_schema_custody": True,
            "published_labelled_trajectory_fixtures_retained_and_passed": False,
            "independent_resonance_classifier_crosscheck_passed": False,
            "classifier_scientifically_qualified": False,
            "dynamics_executed": False,
            "object_classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "open_blockers": [
            "SOURCE_BACKED_PUBLISHED_LABELLED_TRAJECTORY_FIXTURES_NOT_RETAINED_AND_PASSED",
            "INDEPENDENT_RESONANCE_CLASSIFIER_CROSSCHECK_NOT_RETAINED_AND_PASSED",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("validate-bindings",),
        help="validate exact source/runtime/catalog bindings without dynamics",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-bindings":
        sys.stdout.buffer.write(_canonical_json(binding_report_v2()))
        return 0
    raise ClassifierError("unknown command")


if __name__ == "__main__":
    raise SystemExit(main())
