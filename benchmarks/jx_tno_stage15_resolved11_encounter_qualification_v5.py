#!/usr/bin/env python3
"""Predeclare, execute, and replay the Stage-1.5 encounter qualification.

This is an experiment adapter around the pinned internal source-path builder
and the public CUDA massless-tracer entry point.  It is a 1,000-Julian-year
numerical qualification only.  It never emits a TNO dynamical class, a
resonance result, a population inference, or Planet Nine evidence.

Import and ``prepare`` are source-only.  ``execute`` and public offline
``validate_output`` each enter a fresh project-venv child before importing
NumPy, SciPy, CuPy, CUDA, or JX dynamics modules.

``continuation-smoke`` is deliberately different: it runs exactly one real
DE441 h64 annual segment, validates that endpoint as the next continuation,
and emits only a temporary/standard-output setup receipt.  It is partial
dynamics, never a qualification or scientific result.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
from fractions import Fraction
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import types
from typing import Any, BinaryIO, Callable, Iterable, Iterator, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v5.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage15_resolved11_encounter_qualification_v5.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_stage15_resolved11_encounter_qualification_v5"
V1_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v1.py"
V1_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage15_resolved11_encounter_qualification_v1.py"
V1_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_stage15_resolved11_encounter_qualification_v1" / "PROTOCOL.json"
V2_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v2.py"
V2_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage15_resolved11_encounter_qualification_v2.py"
V2_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_stage15_resolved11_encounter_qualification_v2"
V2_PROTOCOL_PATH = V2_OUTPUT_ROOT / "PROTOCOL.json"
V3_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v3.py"
V3_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage15_resolved11_encounter_qualification_v3.py"
V3_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_stage15_resolved11_encounter_qualification_v3"
V3_PROTOCOL_PATH = V3_OUTPUT_ROOT / "PROTOCOL.json"
STAGE15_V4_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage15_resolved11_encounter_qualification_v4.py"
STAGE15_V4_TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_stage15_resolved11_encounter_qualification_v4.py"
STAGE15_V4_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_stage15_resolved11_encounter_qualification_v4"
STAGE15_V4_PROTOCOL_PATH = STAGE15_V4_OUTPUT_ROOT / "PROTOCOL.json"
V4_SOURCE_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v4.py"
V4_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4" / "PROTOCOL.json"
V4_RESULT_PATH = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4" / "RESULT.json"
V4_ARRAYS_PATH = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v4" / "ARRAYS.bin"
BRIDGE_PATH = PROJECT_ROOT / "runs" / "tno_native_epoch_resolved11_bridge_v1" / "BRIDGE.json"
DE441_CUSTODY_PATH = PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1" / "CUSTODY.json"
ENCOUNTER_SOURCE_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "cuda_massless_tracer.py"
ENCOUNTER_TEST_PATH = PROJECT_ROOT / "tests" / "test_cuda_massless_tracer.py"
EVENT_SOURCE_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "cuda_tracer_events.py"
BACKEND_SOURCE_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "engine" / "backends.py"
CONTRACT_SOURCE_PATH = PROJECT_ROOT / "src" / "jxplanetx" / "engine" / "contracts.py"
PROJECT_VENV_ROOT = PROJECT_ROOT / ".venv"
PROJECT_VENV_PYTHON = PROJECT_VENV_ROOT / "bin" / "python"
PROJECT_SITE_PACKAGES = PROJECT_VENV_ROOT / "lib" / "python3.14" / "site-packages"

PROTOCOL_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.protocol.v5"
RESULT_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.result.v5"
RUNTIME_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.runtime.v5"
ARRAY_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.arrays.v5"
SOURCE_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.source-nodes.v5"
ROUTE_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.routes.v5"
MIDPOINT_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.midpoints.v5"
RESTART_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.restart-streams.v5"
RESTART_RECEIPT_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.restart-receipt.v5"
VALIDATION_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.validation.v5"
RUNTIME_PREFLIGHT_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.runtime-preflight.v5"
CONTINUATION_SMOKE_SCHEMA = "jx.tno.stage15-resolved11-encounter-qualification.continuation-smoke.v5"
PROTOCOL_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.protocol.v5\0"
RESULT_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.result.v5\0"
RUNTIME_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.runtime.v5\0"
ARRAY_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.arrays.v5\0"
SOURCE_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.source-nodes.v5\0"
ROUTE_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.routes.v5\0"
MIDPOINT_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.midpoints.v5\0"
RESTART_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.restart-streams.v5\0"
VALIDATION_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.validation.v5\0"
RUNTIME_PREFLIGHT_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.runtime-preflight.v5\0"
CONTINUATION_SMOKE_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.continuation-smoke.v5\0"
SCHEDULE_DOMAIN = b"jx.tno.stage15-resolved11-encounter-qualification.schedule.v5\0"

V1_SOURCE_SHA256 = "6fb25c05f64b95bce8d2fc005429598ccff6eedeac3c21a65dba47ec464bfdd0"
V1_TEST_SHA256 = "1812796ef0a71a19bf23dc2010ad706a1647e68739768fd3a13bf62fe253bc79"
V1_PROTOCOL_SHA256 = "7cd7d81c9fb094ed5e01f8e0a1c513b309268e90e92ddc782d9b42ca1d293aee"
V1_PROTOCOL_CONTENT_SHA256 = "48abc9f4e58c57e4b1c5e28c00ed829aaae10fafbca48216d90695b7b255918d"
V1_GIT_COMMIT = "4c05c21c572b542a5773393aa972cd5edcd7fdad"
V1_GIT_BRANCH = "research/tno-stage1-h64-closure-v1-preregistered-20261010"
V1_GIT_TAG = "tno-stage15-resolved11-encounter-v1-preregistered-20261010"

V2_SOURCE_SHA256 = "8ac9bbb51ad4fcf8543e83ec1f1beaa053d99d5ae8de7bbe633f68caabb9e6ed"
V2_TEST_SHA256 = "e8d26902a28897ac8764d471b2c7704e956498f882b74c838ea847d348ae0439"
V2_PROTOCOL_SHA256 = "9ce4e861ec2582a6a9e0e30c04f943655cbfc75f60e5b8a84119ce06761020ad"
V2_PROTOCOL_CONTENT_SHA256 = "4a9c85556b76c18bd007dd566d70ec49a39230ea1e48678f04a36554651d1181"
V2_GIT_COMMIT = "b55326decf5166c8f4736f5634633801891615e1"
V2_GIT_BRANCH = "research/tno-stage15-resolved11-encounter-v2-preregistered-20261010"
V2_GIT_TAG = "tno-stage15-resolved11-encounter-v2-preregistered-20261010"
V2_GIT_TAG_OBJECT = "b5f08b960b633616787bafc1d22c6ad817cc0d39"
V2_FAILURE_ID = "NONEXISTENT_CUPY_CUDA_NVRTC_IMPORT_BEFORE_DYNAMICS"
V2_FAILURE_CLASS = "ModuleNotFoundError"
V2_FAILURE_MESSAGE = "No module named 'cupy.cuda.nvrtc'"

V3_SOURCE_SHA256 = "f7787b43b7a00ed8c5af4ba0dfa3f2e69c1c642e8296adc45dbcbee5e4d4f4cf"
V3_TEST_SHA256 = "9b81b2feaa1d79c675a5ae1dab6607db0d7d8193a3f4f74f879d60af0ba255df"
V3_PROTOCOL_SHA256 = "c0cb0aa85ff0c4cf72bc0447acb065717a40dce4a41ef3d80ebe49353938591c"
V3_PROTOCOL_CONTENT_SHA256 = "82cad5510bd988f1b1e20f3964a10eae79a70f5fd1ecb2f1c8383ac1ade91db9"
V3_GIT_COMMIT = "8795059e760be721ada7caf42ab046223b3d8db4"
V3_GIT_BRANCH = "research/tno-stage15-resolved11-encounter-v3-preregistered-20261010"
V3_GIT_TAG = "tno-stage15-resolved11-encounter-v3-preregistered-20261010"
V3_GIT_TAG_OBJECT = "448e60451a70dc7e30ec44ff10f68bc6bfdb6745"
V3_FAILURE_ID = "CUPY_CUDA_RUNTIME_SHIM_SHADOWED_BACKEND_BEFORE_DYNAMICS"
V3_FAILURE_CLASS = "AttributeError"
V3_FAILURE_MESSAGE = "module 'cupy.cuda.runtime' has no attribute '_getLocalRuntimeVersion'"

STAGE15_V4_SOURCE_SHA256 = "e1fd53a74bf0b77db2ab20d23257a4e57d3a0204a8eae834a7f09d1186530065"
STAGE15_V4_TEST_SHA256 = "847bcba99dee11de771c8c1f8e0758f7e7ae4b637fb5c19f0c818ef95300a85b"
STAGE15_V4_PROTOCOL_SHA256 = "cf1dcf922aa1bcfffc542f11f3acc28ca9347a92d172f90e8b214971c7251ffb"
STAGE15_V4_PROTOCOL_CONTENT_SHA256 = "214acc0eb6cce965ff1fd531acd5ca2b46046572a127361aa51cd401b88c3853"
STAGE15_V4_GIT_COMMIT = "ae44acf56cf1831f962d62a4ffa1943af4f9db33"
STAGE15_V4_GIT_BRANCH = "research/tno-stage15-resolved11-encounter-v4-preregistered-20261010"
STAGE15_V4_GIT_TAG = "tno-stage15-resolved11-encounter-v4-preregistered-20261010"
STAGE15_V4_GIT_TAG_OBJECT = "92a2a9f9cfa5a7d628e932d0e86ea249d0b1627c"
STAGE15_V4_FAILURE_ID = "ACTIVE_ENDPOINT_MINIMUM_CROSS_IMPLEMENTATION_ROUNDING_AT_ANNUAL_CONTINUATION"
STAGE15_V4_FAILURE_CLASS = "CUDAMasslessTracerContractError"
STAGE15_V4_FAILURE_MESSAGE = "cumulative minimum excludes an active tracer endpoint"

V4_SOURCE_SHA256 = "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"
V4_PROTOCOL_SHA256 = "5aaa8a2db5808c1b343fb25473a50aa2bcbc28059e2d724170cd94063f72c05d"
V4_RESULT_SHA256 = "bebd9186da0ef8943e05a8c4afc057073d48a8a47e9a4d006dae6e59921d5c42"
V4_ARRAYS_SHA256 = "e7ce12bed175e51831b4d55e558b3e813fbd555fd18c5833f91203feb2921346"
V4_ARRAYS_BYTES = 71_495_424
ENCOUNTER_SOURCE_SHA256 = "344aa7749e7888b5d63bd6348311efcc0ba149607eb0e4f32423cf0e1202fc5d"
ENCOUNTER_SOURCE_BYTES = 223_940
ENCOUNTER_TEST_SHA256 = "d555e122cf0c8bdf619b0ee180a9607bfdfe223f6958b4845201cf97d8e29fb7"
ENCOUNTER_TEST_BYTES = 55_996
EVENT_SOURCE_SHA256 = "a3c5f1f9de29f504420aa892bd50f8a947b24d6df21cadf91dc243c7976dad89"
BACKEND_SOURCE_SHA256 = "c039ae04aab43c0809b200304a47e9d6e63bb7cbea76e0665724c75ab513deba"
CONTRACT_SOURCE_SHA256 = "11c834620deedacd219a35bda46772f9be9f0d6243c644dee4fb02eb6c0c144d"
BRIDGE_SHA256 = "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f"
DE441_CUSTODY_SHA256 = "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3"

OBJECT_IDS = (
    "jpl-20090377", "jpl-20474640", "jpl-54526835", "jpl-20541132",
    "jpl-50666516", "jpl-20148209", "jpl-20574372", "jpl-20303775",
    "jpl-20026181", "jpl-20042301", "jpl-20026375", "jpl-20015874",
)
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
TRACER_INDICES = tuple(range(9))
MASSIVE_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
MASSIVE_IDS = (
    "SUN", "MERCURY", "VENUS", "EARTH", "MOON", "MARS_BARYCENTER",
    "JUPITER_BARYCENTER", "SATURN_BARYCENTER", "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER", "PLUTO_BARYCENTER",
)
LANES = (
    ("jx_resolved11_encounter_h64", 64),
    ("jx_resolved11_encounter_h128", 128),
)
LANE_ENGINE = "JX_CUDA_MASSLESS_TRACER_RESTRICTED_KDK_V2"
LANE_MODEL = "RESOLVED_11_BODY"
ACTIVE_ENDPOINT_MINIMUM_POLICY = (
    "ACTIVE_RETURNED_ENDPOINT_HOST_BINARY64_V1:FOR_EACH_SOURCE_IN_SOURCE_ORDER;"
    "DX_DY_DZ_ARE_PYTHON_BINARY64_SUBTRACTIONS;"
    "SQUARED_NORM_IS_LEFT_ASSOCIATED_(DX*DX+DY*DY)+DZ*DZ;"
    "DISTANCE_IS_MATH_SQRT;CUMULATIVE_VALUE_IS_EXACT_OPERAND_MINIMUM;"
    "REMOVED_TRACERS_REMAIN_BOUND_TO_THEIR_FROZEN_EVENT_MINIMUM"
)
SMOKE_NATIVE_EPOCH_JD_TDB_HEX = "0x1.2c195c0000000p+21"
HORIZON_YEARS = 1_000
CHECKPOINT_COUNT = 1_001
MASSIVE_COUNT = 11
TRACER_COUNT = 9
ARRAY_RECORD_DOUBLES = 6 * (MASSIVE_COUNT + TRACER_COUNT)
ARRAY_RECORD_BYTES = ARRAY_RECORD_DOUBLES * 8
ARRAY_LANE_BYTES = CHECKPOINT_COUNT * ARRAY_RECORD_BYTES
ARRAYS_EXACT_BYTES = 12 * 2 * ARRAY_LANE_BYTES
SOURCE_NODE_RECORD_BYTES = MASSIVE_COUNT * 6 * 8
SOURCE_H64_NODES = 64_001
SOURCE_H128_NODES = 128_001
SOURCE_H64_LANE_BYTES = SOURCE_H64_NODES * SOURCE_NODE_RECORD_BYTES
SOURCE_H128_LANE_BYTES = SOURCE_H128_NODES * SOURCE_NODE_RECORD_BYTES
SOURCE_NODES_EXACT_BYTES = 12 * (SOURCE_H64_LANE_BYTES + SOURCE_H128_LANE_BYTES)
MIDPOINT_RECORD_BYTES = TRACER_COUNT * 6 * 8
MIDPOINTS_EXACT_BYTES = 24 * MIDPOINT_RECORD_BYTES
RESTART_YEAR_COUNT = 501
RESTART_STREAM_BYTES = RESTART_YEAR_COUNT * MIDPOINT_RECORD_BYTES
RESTART_CASES = (
    ("jx_resolved11_encounter_h64", OBJECT_IDS[0]),
    ("jx_resolved11_encounter_h64", OBJECT_IDS[-1]),
    ("jx_resolved11_encounter_h128", OBJECT_IDS[0]),
    ("jx_resolved11_encounter_h128", OBJECT_IDS[-1]),
)
RESTART_STREAMS_EXACT_BYTES = len(RESTART_CASES) * 2 * RESTART_STREAM_BYTES
ROUTE_MAX_RECORDS = 100_108
ROUTE_MAX_LINE_BYTES = 65_536
ROUTE_MAX_BYTES = ROUTE_MAX_RECORDS * ROUTE_MAX_LINE_BYTES
RUNTIME_MAX_BYTES = 100_000_000
RESULT_MAX_BYTES = 100_000_000
PROTOCOL_MAX_BYTES = 2_000_000
FIXED_ARTIFACTS = {
    "ARRAYS.bin": ARRAYS_EXACT_BYTES,
    "SOURCE_NODES.bin": SOURCE_NODES_EXACT_BYTES,
    "MIDPOINTS.bin": MIDPOINTS_EXACT_BYTES,
    "RESTART_STREAMS.bin": RESTART_STREAMS_EXACT_BYTES,
}
ARTIFACT_ROSTER = (
    "PROTOCOL.json", "ARRAYS.bin", "SOURCE_NODES.bin", "ROUTES.jsonl",
    "MIDPOINTS.bin", "RESTART_STREAMS.bin", "RUNTIME.json", "RESULT.json",
)
RESULT_LAST_PREFIX = ARTIFACT_ROSTER[:-1]

SCATTERING_THRESHOLD = Fraction(3, 80)
RICHARDSON_A = Fraction(3, 8_000)
FINE_IAS_A = Fraction(3, 4_000)
POSITION_BUDGET = Fraction(1, 500)
VELOCITY_BUDGET = Fraction(1, 50_000)
ENERGY_BUDGET = Fraction(1, 100_000_000)
ANGULAR_BUDGET = Fraction(1, 10_000_000_000)
ENERGY_FLOOR = Fraction(1, 2**46)

RUNTIME_PREFIXES = ("jxplanetx", "numpy", "scipy", "cupy", "cupyx", "cupy_backends")
NVRTC_MODULE_NAME = "cupy_backends.cuda.libs.nvrtc"
NVRTC_MODULE_RELATIVE_PATH = (
    "cupy_backends/cuda/libs/nvrtc.cpython-314-x86_64-linux-gnu.so"
)
NVRTC_MODULE_BYTES = 1_217_120
NVRTC_MODULE_SHA256 = "e38ebdee2e7f2cb1bdaf97d19a5ca43b1baa485ee462f5340bd385b848830c0f"
NVRTC_RECORD_DIGEST = "sha256=44697i5_LLG9r5fRmlykOxuqSF7kYvU0C9OFuEiDDA8"
NVRTC_REQUIRED_API = (
    "getVersion", "createProgram", "compileProgram", "getPTX",
    "getProgramLog", "destroyProgram",
)
CUDA_RUNTIME_MODULE_NAME = "cupy_backends.cuda.api.runtime"
CUDA_RUNTIME_MODULE_RELATIVE_PATH = (
    "cupy_backends/cuda/api/runtime.cpython-314-x86_64-linux-gnu.so"
)
CUDA_RUNTIME_MODULE_BYTES = 4_049_104
CUDA_RUNTIME_MODULE_SHA256 = "bedb6d887526a7726ee3b07d5f8d85b8385a66203261c30511851882bdc1c1a9"
CUDA_RUNTIME_RECORD_DIGEST = "sha256=vtttiHUmp3Ju47B9X42FuDhaZiAyYcMFEYUYgr3Bwak"
CUDA_RUNTIME_REQUIRED_API = (
    "_getLocalRuntimeVersion", "runtimeGetVersion", "driverGetVersion",
    "getDeviceCount",
)
SCIPY_UARRAY_REGISTRY_NAME = "scipy._lib._uarray._uarray"
SCIPY_UARRAY_DECLARED_NAME = "uarray._uarray"
SCIPY_UARRAY_RELATIVE_PATH = (
    "scipy/_lib/_uarray/_uarray.cpython-314-x86_64-linux-gnu.so"
)
SCIPY_UARRAY_BYTES = 115_080
SCIPY_UARRAY_SHA256 = "2747bb027b53fbd354792d8166b9126279c323d023f40ffa072f01d3070181c1"
SCIPY_UARRAY_RECORD_DIGEST = "sha256=J0e7AntT-9NUeS2BZrkSYnnDI9Aj9A_6By8B0wcBgcE"
SCIPY_HIGHS_CORE_RELATIVE_PATH = (
    "scipy/optimize/_highspy/_core.cpython-314-x86_64-linux-gnu.so"
)
SCIPY_HIGHS_CORE_BYTES = 6_348_360
SCIPY_HIGHS_CORE_SHA256 = "3517b3272da1969311a3dd6bdd8ebd3abb0ea65e356130cbfef7c6852e86799f"
SCIPY_HIGHS_CORE_RECORD_DIGEST = "sha256=NRezJy2hlpMRo91r3Y69OrsOpl41YTDL_vfGhS6GeZ8"
MODULE_DECLARED_NAME_ALIASES = {
    SCIPY_UARRAY_REGISTRY_NAME: SCIPY_UARRAY_DECLARED_NAME,
}
SPECLESS_FILE_BACKED_MODULES = {
    "scipy.optimize._highspy._core.simplex_constants": {
        "relative_path": SCIPY_HIGHS_CORE_RELATIVE_PATH,
        "bytes": SCIPY_HIGHS_CORE_BYTES,
        "sha256": SCIPY_HIGHS_CORE_SHA256,
    },
    "scipy.optimize._highspy._core.cb": {
        "relative_path": SCIPY_HIGHS_CORE_RELATIVE_PATH,
        "bytes": SCIPY_HIGHS_CORE_BYTES,
        "sha256": SCIPY_HIGHS_CORE_SHA256,
    },
}
RUNTIME_PREFIX_NONMODULE_EXCEPTIONS = {
    "cupy.fft.config": {
        "kind": "EXACT_OBJECT_ALIAS",
        "type_module": "cupy.fft._config",
        "type_qualname": "_FFTConfig",
        "owner_module": "cupy.fft._config",
        "owner_attribute": "config",
    },
    "cupy.testing": {
        "kind": "EXACT_LAZY_MODULE",
        "type_module": "importlib.util",
        "type_qualname": "_LazyModule",
        "relative_path": "cupy/testing/__init__.py",
        "bytes": 3_051,
        "sha256": "2802c93a36f3eacde4df8297e44a9a54bcf19962fc5066fc8b145a15b19ffcc7",
    },
    "cupyx": {
        "kind": "EXACT_LAZY_MODULE",
        "type_module": "importlib.util",
        "type_qualname": "_LazyModule",
        "relative_path": "cupyx/__init__.py",
        "bytes": 1_405,
        "sha256": "66fbcd6be3df575e3f54b6611e8bd68c342618ab903bbbd73e7dd0d443a8a777",
    },
}
IMPORTLIB_LAZY_MODULE_CLASS = importlib.util._LazyModule  # type: ignore[attr-defined]
IMPORTLIB_SOURCE_FILE_LOADER_CLASS = importlib.machinery.SourceFileLoader
_RUNTIME_PREFIX_OWNER_CLASS_ANCHORS: dict[str, type] = {}
RUNTIME_IMPORT_MODULE_NAMES = (
    "numpy", "numpy._core._multiarray_umath", "scipy",
    "scipy.integrate._ivp.rk", "scipy.integrate._ivp.dop853_coefficients",
    "scipy.optimize", "scipy.optimize._minimize", "scipy.optimize._zeros_py",
    "cupy", "cupy._core.core", "cupy.fft._config",
    CUDA_RUNTIME_MODULE_NAME, NVRTC_MODULE_NAME,
    "jxplanetx.cuda_massless_tracer", "jxplanetx.cuda_tracer_events",
    "jxplanetx.engine.backends", "jxplanetx.engine.contracts",
)
RUNTIME_VERSIONS = {
    "python": "3.14.4", "numpy": "2.5.3", "scipy": "1.17.0",
    "cupy_distribution": "cupy-cuda13x", "cupy": "14.2.0",
}
RUNTIME_MANIFESTS = {
    "numpy": {
        "version": "2.5.3",
        "record": ("numpy-2.5.3.dist-info/RECORD", 111_517, "9d23d1a798b32c85a52dd64d7f484ef38cc29b84692b05c64c23a3a1aada2ca9"),
        "metadata": ("numpy-2.5.3.dist-info/METADATA", 6_635, "451a9b8028000588e66b0b415587b6aef0bbc51a96d8e8a0cba0dc23acf64f99"),
    },
    "scipy": {
        "version": "1.17.0",
        "record": ("scipy-1.17.0.dist-info/RECORD", 198_248, "c81bfef387fa5527abab3a68da6a5344cff20d5fe851a951616682c48c97d5d0"),
        "metadata": ("scipy-1.17.0.dist-info/METADATA", 62_142, "6a5fbbbd6ccb4bedd34782730dd78cb840429a82e2f69e55390af1b51bf7999b"),
    },
    "cupy-cuda13x": {
        "version": "14.2.0",
        "record": ("cupy_cuda13x-14.2.0.dist-info/RECORD", 317_866, "b8fa3fc62720ee9b5090dd3dd1eb5721b30aa849c4f25c93e1507f639f17399b"),
        "metadata": ("cupy_cuda13x-14.2.0.dist-info/METADATA", 2_783, "ede6e52dda954865511bb782b62d6e8d1448b797e081711c05501c81f94ce047"),
    },
}

CLAIMS = {
    "numerical_qualification_only": True,
    "experiment_adapter_not_public_engine_claim": True,
    "dynamical_classification_generated": False,
    "resonance_classification_generated": False,
    "classical_population_inferred": False,
    "scattering_population_inferred": False,
    "detached_population_inferred": False,
    "planet_nine_evidence": False,
    "population_inference": False,
    "science_result": False,
}

RESULT_KEYS = frozenset({
    "schema", "status", "protocol_content_sha256", "artifacts", "lanes",
    "restart_evidence", "route_evidence", "qualification_gates",
    "runtime_content_sha256", "runtime_projection_sha256", "claims",
    "outcomes_generated", "classification_generated", "science_result",
    "content_sha256",
})
LANE_KEYS = frozenset({
    "lane_order", "batch_order", "object_id", "lane_id", "engine", "model",
    "steps_per_julian_year", "step_julian_years_hex", "final_global_step",
    "source_nodes", "cartesian_array", "routes", "midpoint", "annual_summary",
    "all_node_summary", "final_state_sha256", "final_route_ledger_sha256",
    "final_removal_ledger_sha256", "runtime_projection_sha256",
    "no_removal_no_drop",
})
SOURCE_LOCATOR_KEYS = frozenset({
    "schema", "byte_offset", "byte_length", "node_count", "record_bytes",
    "lane_payload_sha256", "annual_segment_receipts",
})
ARRAY_LOCATOR_KEYS = frozenset({
    "schema", "byte_offset", "byte_length", "checkpoint_count", "record_bytes",
    "lane_payload_sha256",
})
ROUTE_LOCATOR_KEYS = frozenset({
    "schema", "byte_offset", "byte_length", "record_count",
    "lane_payload_sha256", "route_count", "removal_count",
})
MIDPOINT_LOCATOR_KEYS = frozenset({
    "midpoint_order", "byte_offset", "byte_length", "record_sha256",
    "batch_order", "object_id", "lane_id", "steps_per_julian_year",
    "midpoint_year", "global_step", "continuation",
})

_INTEGER_KEYS = frozenset({
    "bytes", "native_epoch_batch_count", "tracers_per_batch", "total_tracer_count",
    "massive_body_count", "horizon_julian_years", "annual_checkpoint_count_per_lane",
    "lane_count", "lane_order", "batch_order", "steps_per_julian_year",
    "annual_segment_count", "annual_segment_step_count", "total_fixed_steps",
    "maximum_routes_per_tracer", "maximum_total_routes", "maximum_total_removals",
    "engine_maximum_cumulative_routes_per_lane", "engine_maximum_removals_per_lane",
    "experiment_global_route_and_removal_record_cap",
    "maximum_records", "maximum_line_bytes", "maximum_file_bytes", "midpoint_year",
    "retained_midpoint_count", "midpoint_order", "restart_order", "start_year", "end_year",
    "stream_record_bytes", "stream_bytes", "checkpoint_count", "record_bytes",
    "byte_offset", "byte_length", "node_count", "year_start", "year_end",
    "starting_global_step", "ending_global_step", "private_path_node_count",
    "retained_unique_node_count", "global_step", "record_order", "segment_year",
    "route_count", "removal_count", "case_count", "gate_count", "record_count",
    "suffix_checkpoint_count", "suffix_byte_length", "final_global_step",
    "source_index", "tracer_index", "cpu_accepted_steps", "cpu_rejected_steps",
    "device_index", "cuda_runtime_version", "cuda_local_runtime_version",
    "cuda_driver_version", "device_total_global_memory_bytes",
    "device_multiprocessor_count", "first_line", "resolved_bytes",
    "pyvenv_cfg_bytes", "prior_encounter_route_count", "prior_removal_event_count",
    "source_step_count", "tracer_count",
    "observation_count", "failure_count", "sample_count", "massive_invariant_sample_count",
    "observed_exact_numerator", "observed_exact_denominator",
    "observed_squared_exact_numerator", "observed_squared_exact_denominator",
    "attempt_count", "completed_lane_count", "result_artifact_count",
    "predecessor_source_line", "smoke_segment_count",
    "source_count", "active_tracer_count", "source_node_count",
    "source_payload_bytes", "initial_global_step",
})
_INTEGER_LIST_KEYS = frozenset({
    "holdout_clone_indices", "tracer_indices_required_exactly",
    "massive_naif_ids_in_order", "annual_checkpoint_indices_inclusive",
    "years_inclusive", "encounter_route_counts", "tracer_lifecycle",
})


def _guard_json_integer_identities(value: Any, key: str | None = None) -> None:
    if key in _INTEGER_KEYS and type(value) is not int:
        raise Stage15QualificationError(f"JSON integer identity {key} is not exact int")
    if key in _INTEGER_LIST_KEYS:
        if type(value) is not list or any(type(item) is not int for item in value):
            raise Stage15QualificationError(f"JSON integer-list identity {key} differs")
    if type(value) is dict:
        for child_key, child in value.items():
            _guard_json_integer_identities(child, child_key)
    elif type(value) is list:
        for child in value:
            _guard_json_integer_identities(child, None)


class Stage15QualificationError(RuntimeError):
    """The frozen Stage-1.5 contract or evidence failed closed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _domain_sha(domain: bytes, value: Mapping[str, Any]) -> str:
    copy = dict(value)
    copy.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(copy))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise Stage15QualificationError("JSON has a duplicate or invalid key")
        result[key] = value
    return result


def _strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise Stage15QualificationError(f"{label} contains {value}")
    try:
        value = json.loads(
            payload.decode("utf-8"), object_pairs_hook=_strict_pairs,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage15QualificationError(f"{label} is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise Stage15QualificationError(f"{label} root must be an object")
    return value


def _load_canonical_json(payload: bytes, label: str) -> dict[str, Any]:
    value = _strict_json_bytes(payload, label)
    if _canonical_json(value) != payload:
        raise Stage15QualificationError(f"{label} is not canonical JSON")
    _guard_json_integer_identities(value)
    return value


def _exact_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return set(left) == set(right) and all(
            _exact_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _exact_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if not _exact_equal(actual, expected):
        raise Stage15QualificationError(f"{label} differs from the frozen contract")


def _require_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise Stage15QualificationError(f"{label} must be an exact integer")
    return value


def _require_sha(value: Any, label: str) -> str:
    if (
        type(value) is not str or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise Stage15QualificationError(f"{label} is not a SHA-256 digest")
    return value


def _require_exact_mapping_keys(
    value: Any, expected: Iterable[str], label: str,
) -> Mapping[str, Any]:
    expected_keys = set(expected)
    if type(value) is not dict or set(value) != expected_keys:
        raise Stage15QualificationError(f"{label} key set differs")
    return value


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_ancestors(path: Path, *, allow_leaf_symlink: bool = False) -> None:
    absolute = _lexical_absolute(path)
    parts = absolute.parts
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        if allow_leaf_symlink and index == len(parts) - 1:
            break
        try:
            info = current.lstat()
        except OSError as exc:
            raise Stage15QualificationError(f"path component is unavailable: {current}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise Stage15QualificationError(f"symlinked path component is forbidden: {current}")


@contextmanager
def _stable_fd(
    path: Path, label: str, *, exact_bytes: int | None = None,
    maximum_bytes: int | None = None,
) -> Iterator[tuple[int, os.stat_result]]:
    lexical = _lexical_absolute(path)
    _reject_symlink_ancestors(lexical)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lexical, flags)
    except OSError as exc:
        raise Stage15QualificationError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise Stage15QualificationError(f"{label} is linked or nonregular")
        if exact_bytes is not None and before.st_size != exact_bytes:
            raise Stage15QualificationError(f"{label} byte length differs")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise Stage15QualificationError(f"{label} exceeds its frozen cap")
        yield descriptor, before
        after = os.fstat(descriptor)
        path_after = lexical.lstat()
        identity = lambda item: (
            item.st_dev, item.st_ino, item.st_mode, item.st_nlink,
            item.st_size, item.st_mtime_ns, item.st_ctime_ns,
        )
        if identity(before) != identity(after) or identity(before) != identity(path_after):
            raise Stage15QualificationError(f"{label} changed during held-fd validation")
    finally:
        os.close(descriptor)


def _pread_all(fd: int, size: int, offset: int = 0) -> bytes:
    _require_int(size, "pread size", minimum=0)
    _require_int(offset, "pread offset", minimum=0)
    output = bytearray()
    while len(output) < size:
        block = os.pread(fd, min(1 << 20, size - len(output)), offset + len(output))
        if not block:
            raise Stage15QualificationError("held-fd read ended early")
        output.extend(block)
    return bytes(output)


def _fd_sha256(fd: int, size: int, offset: int = 0) -> str:
    digest = hashlib.sha256()
    consumed = 0
    while consumed < size:
        block = os.pread(fd, min(1 << 20, size - consumed), offset + consumed)
        if not block:
            raise Stage15QualificationError("held-fd digest read ended early")
        digest.update(block)
        consumed += len(block)
    return digest.hexdigest()


def _regular_bytes(
    path: Path, label: str, *, exact_bytes: int | None = None,
    maximum_bytes: int | None = None,
) -> bytes:
    with _stable_fd(path, label, exact_bytes=exact_bytes, maximum_bytes=maximum_bytes) as (fd, info):
        return _pread_all(fd, info.st_size)


def _binding(path: Path, label: str, expected_sha: str, expected_bytes: int | None = None) -> dict[str, Any]:
    payload = _regular_bytes(path, label, exact_bytes=expected_bytes)
    if _sha256(payload) != expected_sha:
        raise Stage15QualificationError(f"{label} digest differs")
    return {
        "relative_path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": len(payload), "file_sha256": expected_sha,
    }


def _load_source_module(path: Path, expected_sha: str, name: str) -> types.ModuleType:
    payload = _regular_bytes(path, f"bound source {path.name}")
    if _sha256(payload) != expected_sha:
        raise Stage15QualificationError(f"bound source changed: {path.name}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(
            compile(payload, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize),
            module.__dict__,
        )
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_v1_predecessor() -> dict[str, Any]:
    source = _binding(V1_SOURCE_PATH, "v1 source", V1_SOURCE_SHA256, 53_428)
    test = _binding(V1_TEST_PATH, "v1 test", V1_TEST_SHA256, 11_208)
    protocol_payload = _regular_bytes(V1_PROTOCOL_PATH, "v1 protocol", exact_bytes=31_382)
    if _sha256(protocol_payload) != V1_PROTOCOL_SHA256:
        raise Stage15QualificationError("v1 retained protocol digest differs")
    protocol = _load_canonical_json(protocol_payload, "v1 protocol")
    if (
        protocol.get("schema") != "jx.tno.stage15-resolved11-encounter-qualification.protocol.v1"
        or protocol.get("content_sha256") != V1_PROTOCOL_CONTENT_SHA256
        or protocol.get("execution_authorized") is not False
        or protocol.get("outcomes_generated") is not False
        or protocol.get("preparation_only") is not True
    ):
        raise Stage15QualificationError("v1 protocol semantic identity differs")
    return {
        "source": source, "test": test,
        "protocol": {
            "relative_path": str(V1_PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
            "bytes": len(protocol_payload), "file_sha256": V1_PROTOCOL_SHA256,
            "content_sha256": V1_PROTOCOL_CONTENT_SHA256,
        },
        "git_custody": {
            "commit": V1_GIT_COMMIT, "branch": V1_GIT_BRANCH, "tag": V1_GIT_TAG,
            "external_git_custody_not_reconstructed_by_this_program": True,
        },
        "blocked_adapter_superseded_without_mutating_v1": True,
    }


def _validate_v2_predecessor() -> dict[str, Any]:
    source = _binding(V2_SOURCE_PATH, "v2 source", V2_SOURCE_SHA256, 180_350)
    test = _binding(V2_TEST_PATH, "v2 test", V2_TEST_SHA256, 35_609)
    if (
        not V2_OUTPUT_ROOT.is_dir()
        or V2_OUTPUT_ROOT.is_symlink()
        or sorted(path.name for path in V2_OUTPUT_ROOT.iterdir()) != ["PROTOCOL.json"]
    ):
        raise Stage15QualificationError("v2 failed root is not the exact protocol-only roster")
    protocol_payload = _regular_bytes(V2_PROTOCOL_PATH, "v2 protocol", exact_bytes=12_920)
    if _sha256(protocol_payload) != V2_PROTOCOL_SHA256:
        raise Stage15QualificationError("v2 retained protocol digest differs")
    protocol = _load_canonical_json(protocol_payload, "v2 protocol")
    claims = protocol.get("claims")
    implementation = protocol.get("implementation")
    if (
        protocol.get("schema")
        != "jx.tno.stage15-resolved11-encounter-qualification.protocol.v2"
        or protocol.get("content_sha256") != V2_PROTOCOL_CONTENT_SHA256
        or protocol.get("outcomes_generated") is not False
        or type(claims) is not dict
        or claims.get("dynamics_executed") is not False
        or claims.get("result_artifacts_published") is not False
        or claims.get("qualification_established") is not False
        or type(implementation) is not dict
        or implementation.get("file_sha256") != V2_SOURCE_SHA256
        or implementation.get("test_file_sha256") != V2_TEST_SHA256
    ):
        raise Stage15QualificationError("v2 protocol semantic identity differs")
    return {
        "source": source,
        "test": test,
        "protocol": {
            "relative_path": str(V2_PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
            "bytes": len(protocol_payload),
            "file_sha256": V2_PROTOCOL_SHA256,
            "content_sha256": V2_PROTOCOL_CONTENT_SHA256,
        },
        "git_custody": {
            "commit": V2_GIT_COMMIT,
            "branch": V2_GIT_BRANCH,
            "annotated_tag": V2_GIT_TAG,
            "tag_object": V2_GIT_TAG_OBJECT,
            "tag_peeled_commit": V2_GIT_COMMIT,
            "external_git_custody_not_reconstructed_by_this_program": True,
        },
        "failed_root": {
            "relative_path": str(V2_OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
            "exact_roster": ["PROTOCOL.json"],
            "attempt_count": 2,
            "completed_lane_count": 0,
            "result_artifact_count": 0,
            "result_artifacts_published": False,
            "dynamics_outcome_generated": False,
        },
        "setup_failure": {
            "failure_id": V2_FAILURE_ID,
            "failure_class": V2_FAILURE_CLASS,
            "failure_message": V2_FAILURE_MESSAGE,
            "failed_callsite": "_fresh_runtime_imports",
            "failed_phase": "FRESH_PROJECT_VENV_RUNTIME_IMPORT_BEFORE_BACKEND_RESOLUTION_OR_LANE_DYNAMICS",
            "invalid_module_name": "cupy.cuda.nvrtc",
            "supported_pinned_module_name": NVRTC_MODULE_NAME,
            "lane_dynamics_started": False,
            "completed_lane_count": 0,
            "result_artifacts_published": False,
            "dynamics_outcome_generated": False,
            "qualification_established": False,
            "scientific_disproof": False,
            "disposition": "SUPERSEDED_BY_V3_WITHOUT_MUTATING_V2",
        },
    }


def _validate_v3_predecessor() -> dict[str, Any]:
    source = _binding(V3_SOURCE_PATH, "v3 source", V3_SOURCE_SHA256, 187_692)
    test = _binding(V3_TEST_PATH, "v3 test", V3_TEST_SHA256, 41_357)
    if (
        not V3_OUTPUT_ROOT.is_dir()
        or V3_OUTPUT_ROOT.is_symlink()
        or sorted(path.name for path in V3_OUTPUT_ROOT.iterdir()) != ["PROTOCOL.json"]
    ):
        raise Stage15QualificationError("v3 failed root is not the exact protocol-only roster")
    protocol_payload = _regular_bytes(V3_PROTOCOL_PATH, "v3 protocol", exact_bytes=15_975)
    if _sha256(protocol_payload) != V3_PROTOCOL_SHA256:
        raise Stage15QualificationError("v3 retained protocol digest differs")
    protocol = _load_canonical_json(protocol_payload, "v3 protocol")
    claims = protocol.get("claims")
    implementation = protocol.get("implementation")
    if (
        protocol.get("schema")
        != "jx.tno.stage15-resolved11-encounter-qualification.protocol.v3"
        or protocol.get("content_sha256") != V3_PROTOCOL_CONTENT_SHA256
        or protocol.get("outcomes_generated") is not False
        or type(claims) is not dict
        or claims.get("dynamics_executed") is not False
        or claims.get("result_artifacts_published") is not False
        or claims.get("qualification_established") is not False
        or type(implementation) is not dict
        or implementation.get("file_sha256") != V3_SOURCE_SHA256
        or implementation.get("test_file_sha256") != V3_TEST_SHA256
    ):
        raise Stage15QualificationError("v3 protocol semantic identity differs")
    return {
        "source": source,
        "test": test,
        "protocol": {
            "relative_path": str(V3_PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
            "bytes": len(protocol_payload),
            "file_sha256": V3_PROTOCOL_SHA256,
            "content_sha256": V3_PROTOCOL_CONTENT_SHA256,
        },
        "git_custody": {
            "commit": V3_GIT_COMMIT,
            "branch": V3_GIT_BRANCH,
            "annotated_tag": V3_GIT_TAG,
            "tag_object": V3_GIT_TAG_OBJECT,
            "tag_peeled_commit": V3_GIT_COMMIT,
            "external_git_custody_not_reconstructed_by_this_program": True,
        },
        "failed_root": {
            "relative_path": str(V3_OUTPUT_ROOT.relative_to(PROJECT_ROOT)),
            "exact_roster": ["PROTOCOL.json"],
            "attempt_count": 1,
            "completed_lane_count": 0,
            "result_artifact_count": 0,
            "result_artifacts_published": False,
            "dynamics_executed": False,
            "dynamics_outcome_generated": False,
        },
        "setup_failure": {
            "failure_id": V3_FAILURE_ID,
            "failure_class": V3_FAILURE_CLASS,
            "failure_message": V3_FAILURE_MESSAGE,
            "failed_callsite": "_runtime_receipt",
            "failed_phase": "FRESH_PROJECT_VENV_RUNTIME_RECEIPT_BEFORE_ANY_LANE_DYNAMICS",
            "mechanism": (
                "EXPLICIT_IMPORT_CUPY_CUDA_RUNTIME_REBOUND_CP_CUDA_RUNTIME_FROM_"
                "CUPY_BACKENDS_CUDA_API_RUNTIME_TO_THE_PYTHON_SHIM"
            ),
            "shadowing_import": "cupy.cuda.runtime",
            "required_backend_module": CUDA_RUNTIME_MODULE_NAME,
            "missing_backend_attribute": "_getLocalRuntimeVersion",
            "lane_dynamics_started": False,
            "completed_lane_count": 0,
            "result_artifacts_published": False,
            "dynamics_executed": False,
            "dynamics_outcome_generated": False,
            "qualification_established": False,
            "scientific_disproof": False,
            "disposition": "SUPERSEDED_BY_V4_WITHOUT_MUTATING_V3",
        },
    }


def _validate_stage15_v4_predecessor() -> dict[str, Any]:
    source = _binding(
        STAGE15_V4_SOURCE_PATH,
        "Stage-1.5 v4 source",
        STAGE15_V4_SOURCE_SHA256,
        211_580,
    )
    test = _binding(
        STAGE15_V4_TEST_PATH,
        "Stage-1.5 v4 test",
        STAGE15_V4_TEST_SHA256,
        54_463,
    )
    if (
        not STAGE15_V4_OUTPUT_ROOT.is_dir()
        or STAGE15_V4_OUTPUT_ROOT.is_symlink()
        or sorted(path.name for path in STAGE15_V4_OUTPUT_ROOT.iterdir())
        != ["PROTOCOL.json"]
    ):
        raise Stage15QualificationError(
            "Stage-1.5 v4 failed root is not the exact protocol-only roster"
        )
    protocol_payload = _regular_bytes(
        STAGE15_V4_PROTOCOL_PATH,
        "Stage-1.5 v4 protocol",
        exact_bytes=20_506,
    )
    if _sha256(protocol_payload) != STAGE15_V4_PROTOCOL_SHA256:
        raise Stage15QualificationError("Stage-1.5 v4 protocol digest differs")
    protocol = _load_canonical_json(protocol_payload, "Stage-1.5 v4 protocol")
    claims = protocol.get("claims")
    implementation = protocol.get("implementation")
    if (
        protocol.get("schema")
        != "jx.tno.stage15-resolved11-encounter-qualification.protocol.v4"
        or protocol.get("content_sha256")
        != STAGE15_V4_PROTOCOL_CONTENT_SHA256
        or protocol.get("outcomes_generated") is not False
        or type(claims) is not dict
        or claims.get("dynamics_executed") is not False
        or claims.get("result_artifacts_published") is not False
        or claims.get("qualification_established") is not False
        or type(implementation) is not dict
        or implementation.get("file_sha256") != STAGE15_V4_SOURCE_SHA256
        or implementation.get("test_file_sha256") != STAGE15_V4_TEST_SHA256
    ):
        raise Stage15QualificationError("Stage-1.5 v4 protocol identity differs")
    return {
        "source": source,
        "test": test,
        "protocol": {
            "relative_path": str(
                STAGE15_V4_PROTOCOL_PATH.relative_to(PROJECT_ROOT)
            ),
            "bytes": len(protocol_payload),
            "file_sha256": STAGE15_V4_PROTOCOL_SHA256,
            "content_sha256": STAGE15_V4_PROTOCOL_CONTENT_SHA256,
        },
        "git_custody": {
            "commit": STAGE15_V4_GIT_COMMIT,
            "branch": STAGE15_V4_GIT_BRANCH,
            "annotated_tag": STAGE15_V4_GIT_TAG,
            "tag_object": STAGE15_V4_GIT_TAG_OBJECT,
            "tag_peeled_commit": STAGE15_V4_GIT_COMMIT,
            "external_git_custody_not_reconstructed_by_this_program": True,
        },
        "failed_root": {
            "relative_path": str(
                STAGE15_V4_OUTPUT_ROOT.relative_to(PROJECT_ROOT)
            ),
            "exact_roster": ["PROTOCOL.json"],
            "attempt_count": 1,
            "completed_lane_count": 0,
            "result_artifact_count": 0,
            "result_artifacts_published": False,
            "partial_dynamics_executed": True,
            "completed_segment_count_exactly_known": False,
            "completed_segment_count": None,
            "dynamics_outcome_generated": False,
        },
        "execution_failure": {
            "failure_id": STAGE15_V4_FAILURE_ID,
            "failure_class": STAGE15_V4_FAILURE_CLASS,
            "failure_message": STAGE15_V4_FAILURE_MESSAGE,
            "failed_callsite": "_validate_continuation",
            "predecessor_source_line": 2546,
            "failed_phase": (
                "FIRST_LANE_ANNUAL_CONTINUATION_AFTER_AT_LEAST_ONE_"
                "SEGMENT_COMPLETED_BEFORE_ANY_COMPLETE_LANE"
            ),
            "mechanism": (
                "CUDA_SQRT_CUMULATIVE_MINIMUM_WAS_STRICTLY_GREATER_THAN_"
                "THE_RECOMPUTED_NUMPY_ENDPOINT_NORM_UNDER_CROSS_"
                "IMPLEMENTATION_BINARY64_ROUNDING"
            ),
            "repair_policy": ACTIVE_ENDPOINT_MINIMUM_POLICY,
            "lane_dynamics_started": True,
            "partial_dynamics_executed": True,
            "completed_segment_count_exactly_known": False,
            "completed_lane_count": 0,
            "result_artifacts_published": False,
            "dynamics_outcome_generated": False,
            "qualification_established": False,
            "scientific_disproof": False,
            "disposition": "SUPERSEDED_BY_V5_WITHOUT_MUTATING_V4",
        },
    }


def _runtime_manifest_bindings() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for distribution, record in RUNTIME_MANIFESTS.items():
        for kind in ("record", "metadata"):
            relative, size, digest = record[kind]
            path = PROJECT_SITE_PACKAGES / relative
            payload = _regular_bytes(path, f"{distribution} {kind}", exact_bytes=size)
            if _sha256(payload) != digest:
                raise Stage15QualificationError(f"{distribution} {kind} digest differs")
            if distribution == "cupy-cuda13x" and kind == "record":
                rows = payload.decode("utf-8").splitlines()
                expected_rows = (
                    (
                        f"{NVRTC_MODULE_RELATIVE_PATH},{NVRTC_RECORD_DIGEST},"
                        f"{NVRTC_MODULE_BYTES}"
                    ),
                    (
                        f"{CUDA_RUNTIME_MODULE_RELATIVE_PATH},"
                        f"{CUDA_RUNTIME_RECORD_DIGEST},{CUDA_RUNTIME_MODULE_BYTES}"
                    ),
                )
                if any(rows.count(expected_row) != 1 for expected_row in expected_rows):
                    raise Stage15QualificationError("CuPy RECORD runtime member differs")
            if distribution == "scipy" and kind == "record":
                rows = payload.decode("utf-8").splitlines()
                expected_rows = (
                    (
                        f"{SCIPY_UARRAY_RELATIVE_PATH},"
                        f"{SCIPY_UARRAY_RECORD_DIGEST},{SCIPY_UARRAY_BYTES}"
                    ),
                    (
                        f"{SCIPY_HIGHS_CORE_RELATIVE_PATH},"
                        f"{SCIPY_HIGHS_CORE_RECORD_DIGEST},{SCIPY_HIGHS_CORE_BYTES}"
                    ),
                )
                if any(rows.count(expected_row) != 1 for expected_row in expected_rows):
                    raise Stage15QualificationError("SciPy RECORD runtime member differs")
            result.append({
                "distribution": distribution, "kind": kind,
                "relative_to_project_site_packages": relative,
                "bytes": size, "file_sha256": digest,
            })
    return result


def _input_bindings(*, full_arrays_hash: bool) -> dict[str, Any]:
    records = [
        _binding(V4_SOURCE_PATH, "Stage-1 v4 source", V4_SOURCE_SHA256, 436_125),
        _binding(V4_PROTOCOL_PATH, "Stage-1 v4 protocol", V4_PROTOCOL_SHA256, 74_186),
        _binding(V4_RESULT_PATH, "Stage-1 v4 result", V4_RESULT_SHA256, 10_602_296),
        _binding(BRIDGE_PATH, "resolved-11 bridge", BRIDGE_SHA256, 168_684),
        _binding(DE441_CUSTODY_PATH, "DE441 custody", DE441_CUSTODY_SHA256, 141_501),
        _binding(
            ENCOUNTER_SOURCE_PATH, "encounter engine",
            ENCOUNTER_SOURCE_SHA256, ENCOUNTER_SOURCE_BYTES,
        ),
        _binding(
            ENCOUNTER_TEST_PATH, "encounter engine test",
            ENCOUNTER_TEST_SHA256, ENCOUNTER_TEST_BYTES,
        ),
        _binding(EVENT_SOURCE_PATH, "event locator", EVENT_SOURCE_SHA256, 67_504),
        _binding(BACKEND_SOURCE_PATH, "backend source", BACKEND_SOURCE_SHA256, 7_367),
        _binding(CONTRACT_SOURCE_PATH, "contract source", CONTRACT_SOURCE_SHA256, 16_390),
    ]
    with _stable_fd(V4_ARRAYS_PATH, "Stage-1 v4 Cartesian oracle", exact_bytes=V4_ARRAYS_BYTES) as (fd, info):
        if full_arrays_hash and _fd_sha256(fd, info.st_size) != V4_ARRAYS_SHA256:
            raise Stage15QualificationError("Stage-1 v4 Cartesian oracle digest differs")
    records.append({
        "relative_path": str(V4_ARRAYS_PATH.relative_to(PROJECT_ROOT)),
        "bytes": V4_ARRAYS_BYTES, "file_sha256": V4_ARRAYS_SHA256,
        "full_digest_recomputed_before_execution": True,
    })
    return {
        "predecessor_v1": _validate_v1_predecessor(),
        "predecessor_v2_setup_failure": _validate_v2_predecessor(),
        "predecessor_v3_setup_failure": _validate_v3_predecessor(),
        "predecessor_v4_partial_dynamics_failure": _validate_stage15_v4_predecessor(),
        "files": records,
        "runtime_manifests": _runtime_manifest_bindings(),
    }


def _protocol_lanes() -> list[dict[str, Any]]:
    return [
        {
            "lane_order": order, "lane_id": lane_id,
            "steps_per_julian_year": steps,
            "step_julian_years_hex": (1.0 / steps).hex(),
            "annual_segment_count": HORIZON_YEARS,
            "annual_segment_step_count": steps,
            "total_fixed_steps": HORIZON_YEARS * steps,
        }
        for order, (lane_id, steps) in enumerate(LANES)
    ]


def build_protocol() -> dict[str, Any]:
    inputs = _input_bindings(full_arrays_hash=False)
    source_payload = _regular_bytes(SCRIPT_PATH, "Stage-1.5 v5 source")
    test_payload = _regular_bytes(TEST_PATH, "Stage-1.5 v5 test")
    record: dict[str, Any] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": "TNO_STAGE15_DE441_RESOLVED11_ENCOUNTER_NUMERICAL_QUALIFICATION_ONLY",
        "status": "PREDECLARED_V5_READY_FOR_EXPLICIT_QUALIFICATION_EXECUTION",
        "claim_ceiling": "1000_YEAR_NUMERICAL_QUALIFICATION_ONLY_NO_TNO_CLASS_OR_SCIENCE",
        "scope": {
            "ephemeris": "DE441", "native_epoch_batch_count": 12,
            "object_ids_in_order": list(OBJECT_IDS),
            "tracer_order": "NOMINAL_THEN_FIXED_EIGHT_COVARIANCE_HOLDOUTS",
            "holdout_clone_indices": list(HOLDOUT_CLONE_INDICES),
            "tracer_indices_required_exactly": list(TRACER_INDICES),
            "tracers_per_batch": TRACER_COUNT, "total_tracer_count": 108,
            "massive_body_count": MASSIVE_COUNT,
            "massive_naif_ids_in_order": list(MASSIVE_NAIF_IDS),
            "horizon_julian_years": HORIZON_YEARS,
            "annual_checkpoint_indices_inclusive": [0, HORIZON_YEARS],
            "annual_checkpoint_count_per_lane": CHECKPOINT_COUNT,
            "lane_count": 24, "lanes": _protocol_lanes(),
        },
        "inputs": inputs,
        "physics": {
            "force_law": "NEWTONIAN_POINT_MASS", "model": LANE_MODEL,
            "massive_ids_in_order": list(MASSIVE_IDS),
            "gm_policy": "EXACT_COMMON_DE440_BRIDGE_BINARY64_GMS_USED_BY_STAGE1_V4",
            "tno_massless": True, "tno_backreaction": False,
            "tracer_tracer_interactions": False,
            "frame": "ICRF_J2000_INERTIAL",
            "origin_for_retained_arrays": "INSTANTANEOUS_RESOLVED11_COMMON_GM_BARYCENTER",
            "length_unit": "AU", "time_unit": "JULIAN_YEAR_EXACTLY_365_25_TDB_DAYS",
            "time_scale_token": "TDB",
            "epoch_origin": "NATIVE_COVARIANCE_EPOCH_JD_TDB_MINUS_2451545_DIVIDED_BY_365_25",
            "source_encounter_radius_AU_hex": "0x1.0000000000000p+0",
            "source_collision_radius_AU_hex": "0x0.0p+0",
            "primary_outer_boundary_AU_hex": "0x1.0000000000000p+12",
            "removal_or_drop_policy": "ANY_REMOVAL_DROP_OR_NONFINITE_STATE_FAILS_QUALIFICATION",
        },
        "adapter": {
            "status": "IMPLEMENTED_EXPERIMENT_ADAPTER_NOT_A_WIDENED_PUBLIC_ENGINE_CLAIM",
            "lane_engine": LANE_ENGINE,
            "lane_model": LANE_MODEL,
            "pinned_module": "jxplanetx.cuda_massless_tracer",
            "pinned_private_callables": [
                "_path_kernels", "_build_source_path",
                "_source_segment_sha256", "_validate_route_witness",
                "_validate_continuation",
                "_host_binary64_active_endpoint_minimum",
                "_canonicalize_active_endpoint_minimum_host",
            ],
            "public_entrypoint": "advance_cuda_massless_tracers",
            "annual_rule": "FOR_EACH_OF_1000_SEGMENTS_BUILD_AN_INDEPENDENT_PINNED_PRIVATE_SOURCE_PATH_FROM_THE_EXACT_STARTING_STATE_THEN_CALL_PUBLIC_ADVANCE",
            "source_evidence_rule": "WRITE_ALL_64001_OR_128001_UNIQUE_FIXED_NODES_INCLUDING_ONLY_ONE_COPY_OF_SHARED_YEAR_BOUNDARIES",
            "endpoint_rule": "PRECOMPUTED_PATH_ENDPOINT_AND_PUBLIC_RETURNED_SOURCE_ENDPOINT_MUST_BE_BITWISE_EQUAL_EACH_YEAR",
            "active_endpoint_minimum_policy": ACTIVE_ENDPOINT_MINIMUM_POLICY,
            "active_endpoint_minimum_rule": (
                "AFTER_EACH_SUCCESSFUL_PUBLIC_SEGMENT_COPY_THE_EXACT_RETURNED_"
                "SOURCE_TRACER_AND_LIFECYCLE_ENDPOINTS_TO_HOST;_RECOMPUTE_EVERY_"
                "ACTIVE_TRACER_SOURCE_DISTANCE_UNDER_THE_FROZEN_BINARY64_RULE;_"
                "TAKE_THE_EXACT_OPERAND_MINIMUM_WITH_THE_CUMULATIVE_DIAGNOSTIC;_"
                "NO_TOLERANCE_OR_ULP_SLACK"
            ),
            "removed_tracer_minimum_rule": (
                "REMOVED_TRACERS_KEEP_THEIR_FROZEN_TERMINAL_EVENT_MINIMUM_AND_"
                "ARE_NOT_RECANONICALIZED_AS_ACTIVE_ENDPOINTS"
            ),
            "engine_source_sha256": ENCOUNTER_SOURCE_SHA256,
            "engine_contract_test_sha256": ENCOUNTER_TEST_SHA256,
            "event_source_sha256": EVENT_SOURCE_SHA256,
            "source_path_steps_per_public_call": {"h64": 64, "h128": 128},
        },
        "routes": {
            "method": "SCIPY_DOP853_CONTINUOUS_EVENT_FULL_STEP_REDO_V2",
            "rtol_hex": (1e-12).hex(), "atol_hex": (1e-14).hex(),
            "maximum_substep_fraction_hex": (0.125).hex(),
            "maximum_primary_dynamical_phase_hex": (0.05).hex(),
            "event_time_tolerance_fraction_hex": (1e-12).hex(),
            "maximum_routes_per_tracer": 16_384,
            "engine_maximum_cumulative_routes_per_lane": 100_000,
            "engine_maximum_removals_per_lane": TRACER_COUNT,
            "experiment_global_route_and_removal_record_cap": ROUTE_MAX_RECORDS,
            "maximum_records": ROUTE_MAX_RECORDS,
            "maximum_line_bytes": ROUTE_MAX_LINE_BYTES,
            "maximum_file_bytes": ROUTE_MAX_BYTES,
            "cap_checked_before_every_write": True,
            "all_route_and_removal_records_retained": True,
            "offline_replay": "EVERY_ROUTE_SOURCE_SEGMENT_IS_EXACTLY_ANCHORED_TO_ADJACENT_RETAINED_SOURCE_NODES_AND_THE_PINNED_GUARD_AND_DOP853_LOCATOR_REPLAY_IS_REQUIRED",
        },
        "restart": {
            "midpoint_year": 500, "retained_midpoint_count": 24,
            "selected_fresh_process_cases": [
                {"restart_order": order, "lane_id": lane, "object_id": object_id}
                for order, (lane, object_id) in enumerate(RESTART_CASES)
            ],
            "streams_per_case": ["MONOLITHIC", "FRESH_RESUMED"],
            "years_inclusive": [500, 1_000],
            "stream_record_bytes": MIDPOINT_RECORD_BYTES,
            "stream_bytes": RESTART_STREAM_BYTES,
            "acceptance": "BITWISE_EQUAL_TRACER_SUFFIX_AND_EQUAL_SOURCE_SEGMENT_ENDPOINT_ROUTE_REMOVAL_RUNTIME_AND_FINAL_GLOBAL_STEP_LEDGERS",
        },
        "evidence": {
            "exact_roster": list(ARTIFACT_ROSTER),
            "result_created_last": True, "create_only": True,
            "multi_file_atomicity_claimed": False,
            "partial_destination_on_failure": "INVALID_AND_REQUIRES_MANUAL_CLEANUP;_PUBLISHED_NAMES_ARE_NEVER_UNLINKED_BY_ROLLBACK",
            "post_result_creation_fsync_failure": "COMPLETE_ROSTER_MAY_REMAIN_BUT_EXECUTION_RAISES_AND_REQUIRES_EXPLICIT_OFFLINE_VALIDATION",
            "fixed_exact_bytes": dict(FIXED_ARTIFACTS),
            "routes_maximum_bytes": ROUTE_MAX_BYTES,
            "runtime_maximum_bytes": RUNTIME_MAX_BYTES,
            "result_maximum_bytes": RESULT_MAX_BYTES,
            "large_artifact_validation": "ONE_O_NOFOLLOW_FD_HELD_FOR_GLOBAL_HASH_SEMANTIC_PREADS_AND_FINAL_FSTAT",
            "arrays_order": "BATCH_THEN_H64_H128_THEN_YEAR_THEN_11_POS_11_VEL_9_POS_9_VEL",
            "source_order": "BATCH_THEN_H64_H128_THEN_UNIQUE_GLOBAL_NODE_THEN_11_POS_THEN_11_VEL",
            "restart_stream_binary_order": "CASE_THEN_MONOLITHIC_FRESH_RESUMED_THEN_YEAR_500_1000_THEN_9_POS_THEN_9_VEL",
        },
        "runtime": {
            "fresh_project_venv_child_for_execute": True,
            "fresh_project_venv_child_for_offline_validation": True,
            "reject_preloaded_prefixes": list(RUNTIME_PREFIXES),
            "explicit_import_modules_in_order": list(RUNTIME_IMPORT_MODULE_NAMES),
            "nvrtc_module": NVRTC_MODULE_NAME,
            "nvrtc_module_is_manifest_bound_cupy_cuda13x_extension": True,
            "nvrtc_module_relative_to_project_site_packages": NVRTC_MODULE_RELATIVE_PATH,
            "nvrtc_module_bytes": NVRTC_MODULE_BYTES,
            "nvrtc_module_sha256": NVRTC_MODULE_SHA256,
            "nvrtc_record_digest": NVRTC_RECORD_DIGEST,
            "nvrtc_required_api": list(NVRTC_REQUIRED_API),
            "cuda_runtime_module": CUDA_RUNTIME_MODULE_NAME,
            "cuda_runtime_shim_forbidden": "cupy.cuda.runtime",
            "cuda_runtime_module_relative_to_project_site_packages": CUDA_RUNTIME_MODULE_RELATIVE_PATH,
            "cuda_runtime_module_bytes": CUDA_RUNTIME_MODULE_BYTES,
            "cuda_runtime_module_sha256": CUDA_RUNTIME_MODULE_SHA256,
            "cuda_runtime_record_digest": CUDA_RUNTIME_RECORD_DIGEST,
            "cuda_runtime_required_api": list(CUDA_RUNTIME_REQUIRED_API),
            "exact_declared_module_name_aliases": dict(MODULE_DECLARED_NAME_ALIASES),
            "scipy_uarray_alias_binding": {
                "registry_name": SCIPY_UARRAY_REGISTRY_NAME,
                "declared_module_name": SCIPY_UARRAY_DECLARED_NAME,
                "spec_name": SCIPY_UARRAY_REGISTRY_NAME,
                "relative_to_project_site_packages": SCIPY_UARRAY_RELATIVE_PATH,
                "bytes": SCIPY_UARRAY_BYTES,
                "sha256": SCIPY_UARRAY_SHA256,
                "record_digest": SCIPY_UARRAY_RECORD_DIGEST,
                "loader_type": "ExtensionFileLoader",
            },
            "exact_specless_file_backed_modules": {
                name: dict(binding)
                for name, binding in SPECLESS_FILE_BACKED_MODULES.items()
            },
            "scipy_highs_core_record_digest": SCIPY_HIGHS_CORE_RECORD_DIGEST,
            "runtime_prefix_scan_declared_name_alias_count": 1,
            "arbitrary_module_aliases_or_specless_modules_accepted": False,
            "post_import_alias_invariant": (
                "CP_CUDA_RUNTIME_IS_EXACT_CUPY_BACKENDS_CUDA_API_RUNTIME_AND_"
                "CUPY_CUDA_RUNTIME_SHIM_IS_ABSENT_FROM_SYS_MODULES"
            ),
            "import_only_preflight_performs_no_backend_resolution_or_device_query": True,
            "full_runtime_receipt_preflight": {
                "command": (
                    ".venv/bin/python -B benchmarks/"
                    "jx_tno_stage15_resolved11_encounter_qualification_v5.py "
                    "runtime-preflight"
                ),
                "fresh_exact_venv_worker": True,
                "backend_resolution": True,
                "engine_runtime_identity": True,
                "source_step_count": 0,
                "tracer_count": 0,
                "dynamics_executed": False,
                "scientific_output": False,
                "retained_scientific_output": False,
            },
            "one_segment_continuation_smoke": {
                "command": (
                    ".venv/bin/python -B benchmarks/"
                    "jx_tno_stage15_resolved11_encounter_qualification_v5.py "
                    "continuation-smoke"
                ),
                "fresh_exact_venv_worker": True,
                "real_retained_input": "DE441_BATCH_0_NOMINAL_PLUS_EIGHT_HOLDOUTS",
                "lane_id": "jx_resolved11_encounter_h64",
                "steps_per_julian_year": 64,
                "smoke_segment_count": 1,
                "next_continuation_validation": True,
                "second_segment_executed": False,
                "partial_dynamics_executed": True,
                "setup_numerical_outcome_generated": True,
                "setup_numerical_outcome_disclosed": True,
                "qualification_established": False,
                "qualification_outcome_generated": False,
                "retained_scientific_output": False,
                "default_output_root_mutated": False,
            },
            "versions": dict(RUNTIME_VERSIONS),
            "bind_runtime_before_and_after_execution_and_engine_identity_on_every_lane": True,
            "bind_each_restart_worker_interpreter_registry_guard_and_runtime_projection": True,
            "module_registry_exact_moduletype_only": True,
            "runtime_prefix_nonmodule_exceptions": {
                name: dict(binding)
                for name, binding in RUNTIME_PREFIX_NONMODULE_EXCEPTIONS.items()
            },
            "runtime_prefix_capture_policy": (
                "EVERY_PREFIX_KEY_MUST_BE_AN_EXACT_FILE_BACKED_MODULE_OR_"
                "ONE_EXPLICIT_PINNED_EXCEPTION;_UNKNOWN_NONMODULE_AND_"
                "FILELESS_MODULE_VALUES_FAIL_CLOSED"
            ),
            "post_execution_module_registry_transition": (
                "PRE_EXECUTION_REGISTRY_IS_AN_EXACT_UNCHANGED_SUBSET_OF_"
                "POST_EXECUTION_REGISTRY;_EVERY_ADDITION_IS_AN_EXACT_"
                "LIVE_FILE_BACKED_RUNTIME_PREFIX_MODULE;_REMOVAL_OR_"
                "MUTATION_FAILS_CLOSED"
            ),
            "pycache_and_external_cupy_kernel_cache_isolated": True,
            "custody_ceiling": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATIONS_WITH_NO_CONCURRENT_MUTATION_NOT_AUTHENTICATION",
        },
        "gates": {
            "inherited_stage1_formulas": True,
            "fine_h128_vs_v4_de441_resolved11_ias15": {
                "semimajor_axis": "3/4000", "position_AU": "1/500",
                "velocity_AU_per_julian_year": "1/50000",
                "scattering_diagnostic_disagreement": "3/4000",
            },
            "h64_h128_richardson_divisor": 3,
            "richardson_semimajor_axis_tail": "3/8000",
            "source_and_tracer_cartesian_richardson_budgets": ["1/500", "1/50000"],
            "contraction": "EXACT_FINE_LE_EXACT_64_ULP_FLOOR_OR_COARSE_GE_3_FINE;_VECTORS_USE_EXACT_SQUARES_AND_FACTOR_9",
            "all_node_fine_energy": "1/100000000",
            "all_node_fine_angular_momentum": "1/10000000000",
            "energy_contraction_floor": "1/70368744177664",
            "preliminary_scattering_threshold": "3/80",
            "preliminary_scattering_is_not_acceptance_or_classification": True,
            "all_individual_predicates_must_pass": True,
        },
        "execution": {
            "mode_present": True,
            "authorized_only_after_exact_retained_protocol_and_all_pinned_prerequisites_validate": True,
            "fresh_worker_required": True,
            "pre_result_creation_failures_publish_no_result": True,
            "result_creation_is_irreversible_publication_commit": True,
            "post_commit_failure_requires_explicit_offline_validation": True,
        },
        "implementation": {
            "relative_path": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
            "bytes": len(source_payload), "file_sha256": _sha256(source_payload),
            "test_relative_path": str(TEST_PATH.relative_to(PROJECT_ROOT)),
            "test_bytes": len(test_payload), "test_file_sha256": _sha256(test_payload),
            "prepare_imports_dynamics_runtimes": False,
        },
        "claims": {
            **CLAIMS, "protocol_predeclared_before_execution": True,
            "externally_preregistered": False, "dynamics_executed": False,
            "result_artifacts_published": False,
            "qualification_established": False,
        },
        "outcomes_generated": False,
        "nonclaim": "NOT_A_DYNAMICAL_CLASS_RESONANCE_POPULATION_OR_PLANET_NINE_RESULT",
    }
    record["content_sha256"] = _domain_sha(PROTOCOL_DOMAIN, record)
    return record


def validate_protocol(value: Mapping[str, Any]) -> str:
    expected = build_protocol()
    if type(value) is not dict or not _exact_equal(value, expected):
        raise Stage15QualificationError("protocol differs from exact reconstruction")
    if value.get("content_sha256") != _domain_sha(PROTOCOL_DOMAIN, value):
        raise Stage15QualificationError("protocol content digest differs")
    return str(value["content_sha256"])


def _write_create_only(path: Path, payload: bytes, *, maximum: int) -> None:
    if len(payload) > maximum:
        raise Stage15QualificationError(f"{path.name} exceeds its frozen cap before write")
    _reject_symlink_ancestors(path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise Stage15QualificationError(f"refusing to replace {path}") from exc
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(fd, view[written:])
            if count <= 0:
                raise Stage15QualificationError(f"short write for {path.name}")
            written += count
        os.fsync(fd)
    finally:
        os.close(fd)


class _CappedBinaryWriter:
    def __init__(self, path: Path, maximum: int) -> None:
        self.path = path
        self.maximum = _require_int(maximum, "writer cap", minimum=0)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        _reject_symlink_ancestors(path.parent)
        try:
            self.fd = os.open(path, flags, 0o600)
        except OSError as exc:
            raise Stage15QualificationError(f"cannot create {path.name}") from exc
        self.count = 0
        self.record_count = 0
        self.digest = hashlib.sha256()

    def write(self, payload: bytes) -> None:
        if type(payload) is not bytes:
            raise Stage15QualificationError("binary evidence append must be bytes")
        if self.count + len(payload) > self.maximum:
            raise Stage15QualificationError("evidence cap would be exceeded before write")
        view = memoryview(payload)
        done = 0
        while done < len(view):
            count = os.write(self.fd, view[done:])
            if count <= 0:
                raise Stage15QualificationError("evidence write made no progress")
            done += count
        self.count += len(payload)
        self.digest.update(payload)

    def close(self, *, exact: int | None = None) -> dict[str, Any]:
        if self.fd < 0:
            raise Stage15QualificationError("writer already closed")
        if exact is not None and self.count != exact:
            raise Stage15QualificationError("evidence writer exact length differs")
        os.fsync(self.fd)
        os.close(self.fd)
        self.fd = -1
        return {"bytes": self.count, "file_sha256": self.digest.hexdigest()}

    def abort(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _lexical_absolute(output_root)
    if root.exists() or root.is_symlink():
        raise Stage15QualificationError("prepare output root must not exist")
    _reject_symlink_ancestors(root.parent)
    root.mkdir(mode=0o700)
    try:
        protocol = build_protocol()
        _write_create_only(root / "PROTOCOL.json", _canonical_json(protocol), maximum=PROTOCOL_MAX_BYTES)
        _fsync_directory(root)
        validate_prepared(root)
        return protocol
    except BaseException:
        # Only remove the directory if it is still the exact empty/own protocol
        # directory.  Never path-unlink a possibly replaced destination.
        try:
            entries = list(root.iterdir())
            if entries == []:
                root.rmdir()
        except OSError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _lexical_absolute(output_root)
    if not root.is_dir() or root.is_symlink() or sorted(p.name for p in root.iterdir()) != ["PROTOCOL.json"]:
        raise Stage15QualificationError("prepared root must contain only PROTOCOL.json")
    payload = _regular_bytes(root / "PROTOCOL.json", "retained v5 protocol", maximum_bytes=PROTOCOL_MAX_BYTES)
    value = _load_canonical_json(payload, "retained v5 protocol")
    validate_protocol(value)
    return value


def _runtime_prefix_keys() -> list[str]:
    keys: list[str] = []
    for name in sys.modules:
        if type(name) is not str:
            raise Stage15QualificationError("sys.modules contains a non-string key")
        if any(name == prefix or name.startswith(prefix + ".") for prefix in RUNTIME_PREFIXES):
            keys.append(name)
    return sorted(keys)


def _reject_preloaded_runtime_modules() -> dict[str, Any]:
    found = _runtime_prefix_keys()
    if found:
        raise Stage15QualificationError(f"fresh runtime registry is contaminated: {found[:8]}")
    return {
        "checked_prefixes": list(RUNTIME_PREFIXES),
        "preloaded_module_names": [],
        "fresh_registry_observed": True,
        "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
    }


def _project_venv_launcher_receipt() -> dict[str, Any]:
    launcher = _lexical_absolute(PROJECT_VENV_PYTHON)
    _reject_symlink_ancestors(launcher, allow_leaf_symlink=True)
    if not launcher.is_symlink():
        raise Stage15QualificationError("project venv Python launcher must be the pinned symlink")
    first = os.readlink(launcher)
    first_path = _lexical_absolute(launcher.parent / first)
    _reject_symlink_ancestors(first_path, allow_leaf_symlink=True)
    if not first_path.is_symlink():
        raise Stage15QualificationError("project venv python3 link changed")
    second = os.readlink(first_path)
    second_path = _lexical_absolute(first_path.parent / second)
    _reject_symlink_ancestors(second_path, allow_leaf_symlink=True)
    third = os.readlink(second_path) if second_path.is_symlink() else None
    resolved = launcher.resolve(strict=True)
    payload = _regular_bytes(resolved, "resolved Python executable", exact_bytes=7_477_160)
    if _sha256(payload) != "52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41":
        raise Stage15QualificationError("resolved Python executable changed")
    cfg = _regular_bytes(PROJECT_VENV_ROOT / "pyvenv.cfg", "pyvenv.cfg", exact_bytes=250)
    if _sha256(cfg) != "5c81a7f0372c2a7687272a8e0068fe9f036e6422e2c6f5b11588cbf568dfbae3":
        raise Stage15QualificationError("project pyvenv.cfg changed")
    return {
        "launcher_path": str(launcher), "launcher_target": first,
        "python3_path": str(first_path), "python3_target": second,
        "system_link_path": str(second_path), "system_link_target": third,
        "resolved_path": str(resolved), "resolved_bytes": len(payload),
        "resolved_sha256": _sha256(payload),
        "pyvenv_cfg_path": str(PROJECT_VENV_ROOT / "pyvenv.cfg"),
        "pyvenv_cfg_bytes": len(cfg), "pyvenv_cfg_sha256": _sha256(cfg),
    }


def _fresh_child_interpreter_receipt() -> dict[str, Any]:
    launcher = _project_venv_launcher_receipt()
    if (
        _lexical_absolute(Path(sys.executable)) != _lexical_absolute(PROJECT_VENV_PYTHON)
        or _lexical_absolute(Path(sys.prefix)) != _lexical_absolute(PROJECT_VENV_ROOT)
        or platform.python_version() != RUNTIME_VERSIONS["python"]
    ):
        raise Stage15QualificationError("worker did not use the exact project venv Python")
    return {
        "schema": "jx.tno.stage15.fresh-child-interpreter.v5",
        "sys_executable": str(_lexical_absolute(Path(sys.executable))),
        "sys_prefix": str(_lexical_absolute(Path(sys.prefix))),
        "sys_base_prefix": str(_lexical_absolute(Path(sys.base_prefix))),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "launcher": launcher,
    }


def _runtime_child_environment(cache_root: Path) -> dict[str, str]:
    root = _lexical_absolute(cache_root)
    if root.exists() or root.is_symlink():
        raise Stage15QualificationError("runtime cache root must start absent")
    pycache = root / "pycache-must-remain-absent"
    cupy_cache = root / "cupy-kernel-cache"
    cupy_cache.mkdir(parents=True, mode=0o700)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)))
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPYCACHEPREFIX"] = str(pycache)
    env["CUPY_CACHE_DIR"] = str(cupy_cache)
    env["CUPY_CACHE_IN_MEMORY"] = "1"
    env["CUDA_CACHE_DISABLE"] = "1"
    env["PYTHONHASHSEED"] = "0"
    return env


def _file_receipt(path: Path, label: str, *, allow_empty: bool = False) -> dict[str, Any]:
    payload = _regular_bytes(path, label)
    if not payload and not allow_empty:
        raise Stage15QualificationError(f"{label} is unexpectedly empty")
    return {"path": str(_lexical_absolute(path)), "bytes": len(payload), "sha256": _sha256(payload)}


def _module_receipt(module: types.ModuleType, expected_name: str) -> dict[str, Any]:
    if type(expected_name) is not str or not expected_name:
        raise Stage15QualificationError("runtime module registry name differs")
    expected_declared_name = MODULE_DECLARED_NAME_ALIASES.get(
        expected_name, expected_name,
    )
    if (
        type(module) is not types.ModuleType
        or type(module.__name__) is not str
        or module.__name__ != expected_declared_name
    ):
        raise Stage15QualificationError(f"{expected_name} is not an exact ModuleType")
    spec = module.__spec__
    path_value = getattr(module, "__file__", None)
    if type(path_value) is not str:
        raise Stage15QualificationError(f"{expected_name} module spec is incomplete")
    path = _lexical_absolute(Path(path_value))
    receipt = _file_receipt(path, f"module {expected_name}", allow_empty=True)
    specless_binding = SPECLESS_FILE_BACKED_MODULES.get(expected_name)
    if specless_binding is not None:
        expected_path = _lexical_absolute(
            PROJECT_SITE_PACKAGES / str(specless_binding["relative_path"])
        )
        if (
            spec is not None
            or path != expected_path
            or receipt.get("bytes") != specless_binding["bytes"]
            or receipt.get("sha256") != specless_binding["sha256"]
        ):
            raise Stage15QualificationError(
                f"{expected_name} exact spec-less module binding differs"
            )
        return {
            "module": expected_name,
            "declared_module_name": module.__name__,
            "spec_name": None,
            "spec_origin": None,
            "loader_type": None,
            **receipt,
        }
    if (
        spec is None
        or spec.name != expected_name
        or type(spec.origin) is not str
        or _lexical_absolute(Path(spec.origin)) != path
        or spec.loader is None
    ):
        raise Stage15QualificationError(f"{expected_name} module spec differs")
    if expected_name == SCIPY_UARRAY_REGISTRY_NAME:
        expected_path = _lexical_absolute(
            PROJECT_SITE_PACKAGES / SCIPY_UARRAY_RELATIVE_PATH
        )
        if (
            path != expected_path
            or receipt.get("bytes") != SCIPY_UARRAY_BYTES
            or receipt.get("sha256") != SCIPY_UARRAY_SHA256
            or type(spec.loader).__name__ != "ExtensionFileLoader"
        ):
            raise Stage15QualificationError("pinned SciPy uarray alias differs")
    return {
        "module": expected_name,
        "declared_module_name": module.__name__,
        "spec_name": spec.name,
        "spec_origin": str(path),
        "loader_type": type(spec.loader).__name__,
        **receipt,
    }


def _nvrtc_module_receipt(module: types.ModuleType) -> dict[str, Any]:
    receipt = _module_receipt(module, NVRTC_MODULE_NAME)
    expected_path = _lexical_absolute(PROJECT_SITE_PACKAGES / NVRTC_MODULE_RELATIVE_PATH)
    if (
        receipt.get("path") != str(expected_path)
        or receipt.get("bytes") != NVRTC_MODULE_BYTES
        or receipt.get("sha256") != NVRTC_MODULE_SHA256
        or any(not callable(getattr(module, name, None)) for name in NVRTC_REQUIRED_API)
    ):
        raise Stage15QualificationError("pinned CuPy NVRTC module or API differs")
    return receipt


def _cuda_runtime_module_receipt(module: types.ModuleType) -> dict[str, Any]:
    receipt = _module_receipt(module, CUDA_RUNTIME_MODULE_NAME)
    expected_path = _lexical_absolute(
        PROJECT_SITE_PACKAGES / CUDA_RUNTIME_MODULE_RELATIVE_PATH
    )
    if (
        receipt.get("path") != str(expected_path)
        or receipt.get("bytes") != CUDA_RUNTIME_MODULE_BYTES
        or receipt.get("sha256") != CUDA_RUNTIME_MODULE_SHA256
        or any(
            not callable(getattr(module, name, None))
            for name in CUDA_RUNTIME_REQUIRED_API
        )
    ):
        raise Stage15QualificationError("pinned CuPy CUDA runtime module or API differs")
    return receipt


def _require_cuda_runtime_alias_invariant(
    cupy_module: types.ModuleType,
    modules: Mapping[str, Any],
) -> types.ModuleType:
    backend_runtime = modules.get(CUDA_RUNTIME_MODULE_NAME)
    cuda_namespace = getattr(cupy_module, "cuda", None)
    if (
        type(backend_runtime) is not types.ModuleType
        or getattr(cuda_namespace, "runtime", None) is not backend_runtime
        or backend_runtime.__name__ != CUDA_RUNTIME_MODULE_NAME
        or "cupy.cuda.runtime" in sys.modules
        or any(
            not callable(getattr(backend_runtime, name, None))
            for name in CUDA_RUNTIME_REQUIRED_API
        )
    ):
        raise Stage15QualificationError(
            "CuPy CUDA runtime backend was shadowed or its API differs"
        )
    return backend_runtime


def _unique_distribution(name: str, version: str) -> importlib.metadata.Distribution:
    normalized = name.lower().replace("_", "-")
    matches = [
        distribution
        for distribution in importlib.metadata.distributions(path=[str(PROJECT_SITE_PACKAGES)])
        if str(distribution.metadata.get("Name", "")).lower().replace("_", "-") == normalized
    ]
    if len(matches) != 1 or matches[0].version != version:
        raise Stage15QualificationError(f"distribution identity differs for {name}")
    return matches[0]


def _distribution_receipt(name: str) -> dict[str, Any]:
    expected = RUNTIME_MANIFESTS[name]
    distribution = _unique_distribution(name, str(expected["version"]))
    rows: dict[str, Any] = {}
    for kind in ("record", "metadata"):
        relative, size, digest = expected[kind]
        path = PROJECT_SITE_PACKAGES / relative
        receipt = _file_receipt(path, f"{name} {kind}", allow_empty=False)
        if receipt["bytes"] != size or receipt["sha256"] != digest:
            raise Stage15QualificationError(f"{name} {kind} no longer matches protocol")
        rows[kind] = receipt
    dist_path = _lexical_absolute(Path(distribution._path))  # type: ignore[attr-defined]
    expected_dist = _lexical_absolute((PROJECT_SITE_PACKAGES / expected["metadata"][0]).parent)
    if dist_path != expected_dist:
        raise Stage15QualificationError(f"{name} distribution path differs")
    return {
        "distribution_name": name, "distribution_version": distribution.version,
        "distribution_path": str(dist_path), **rows,
    }


def _runtime_prefix_nonmodule_exception_receipt(
    name: str, value: Any,
) -> dict[str, Any]:
    binding = RUNTIME_PREFIX_NONMODULE_EXCEPTIONS.get(name)
    if type(binding) is not dict:
        raise Stage15QualificationError(
            f"runtime prefix entry is not an exact ModuleType: {name}"
        )
    kind = binding.get("kind")
    if kind == "EXACT_OBJECT_ALIAS":
        owner = sys.modules.get(binding.get("owner_module"))
        owner_class = (
            getattr(owner, "_FFTConfig", None)
            if type(owner) is types.ModuleType else None
        )
        anchored_class = _RUNTIME_PREFIX_OWNER_CLASS_ANCHORS.get(name)
        if anchored_class is None and type(owner_class) is type:
            _RUNTIME_PREFIX_OWNER_CLASS_ANCHORS[name] = owner_class
            anchored_class = owner_class
        if (
            type(owner) is not types.ModuleType
            or type(owner_class) is not type
            or owner_class is not anchored_class
            or type(value) is not owner_class
            or getattr(owner, binding.get("owner_attribute", ""), None) is not value
        ):
            raise Stage15QualificationError(
                f"pinned runtime prefix object alias differs: {name}"
            )
        return {
            "registry_name": name,
            "kind": kind,
            "type_module": owner_class.__module__,
            "type_qualname": owner_class.__qualname__,
            "owner_module": binding["owner_module"],
            "owner_attribute": binding["owner_attribute"],
            "owner_class_attribute": "_FFTConfig",
            "class_object_identity_checked": True,
            "owner_object_identity_checked": True,
        }
    if kind != "EXACT_LAZY_MODULE":
        raise Stage15QualificationError(
            f"pinned runtime prefix exception kind differs: {name}"
        )
    if type(value) is not IMPORTLIB_LAZY_MODULE_CLASS:
        raise Stage15QualificationError(
            f"pinned lazy runtime prefix class differs: {name}"
        )
    namespace = object.__getattribute__(value, "__dict__")
    if type(namespace) is not dict or set(namespace) != {
        "__name__", "__doc__", "__package__", "__loader__", "__spec__",
        "__path__", "__file__", "__cached__",
    }:
        raise Stage15QualificationError(
            f"pinned lazy runtime prefix namespace differs: {name}"
        )
    path_value = namespace.get("__file__")
    cached_value = namespace.get("__cached__")
    spec = namespace.get("__spec__")
    loader = namespace.get("__loader__")
    expected_path = _lexical_absolute(
        PROJECT_SITE_PACKAGES / str(binding.get("relative_path"))
    )
    expected_package_path = expected_path.parent
    expected_cached = importlib.util.cache_from_source(str(expected_path))
    pycache_prefix = sys.pycache_prefix
    if (
        namespace.get("__name__") != name
        or namespace.get("__package__") != name
        or type(path_value) is not str
        or _lexical_absolute(Path(path_value)) != expected_path
        or type(cached_value) is not str
        or _lexical_absolute(Path(cached_value))
        != _lexical_absolute(Path(expected_cached))
        or type(namespace.get("__path__")) is not list
        or namespace.get("__path__") != [str(expected_package_path)]
        or spec is None
        or getattr(spec, "name", None) != name
        or type(getattr(spec, "origin", None)) is not str
        or _lexical_absolute(Path(spec.origin)) != expected_path
        or getattr(spec, "parent", None) != name
        or getattr(spec, "has_location", None) is not True
        or getattr(spec, "cached", None) != cached_value
        or list(getattr(spec, "submodule_search_locations", ()))
        != [str(expected_package_path)]
        or loader is not getattr(spec, "loader", None)
        or type(loader) is not IMPORTLIB_SOURCE_FILE_LOADER_CLASS
        or getattr(loader, "name", None) != name
        or type(getattr(loader, "path", None)) is not str
        or _lexical_absolute(Path(loader.path)) != expected_path
        or type(pycache_prefix) is not str
        or not Path(cached_value).is_relative_to(
            _lexical_absolute(Path(pycache_prefix))
        )
    ):
        raise Stage15QualificationError(
            f"pinned lazy runtime prefix identity differs: {name}"
        )
    receipt = _file_receipt(expected_path, f"lazy runtime prefix {name}")
    if (
        receipt.get("bytes") != binding.get("bytes")
        or receipt.get("sha256") != binding.get("sha256")
    ):
        raise Stage15QualificationError(
            f"pinned lazy runtime prefix bytes differ: {name}"
        )
    cached_suffix = str(
        _lexical_absolute(Path(cached_value)).relative_to(
            _lexical_absolute(Path(pycache_prefix))
        )
    )
    return {
        "registry_name": name,
        "kind": kind,
        "type_module": IMPORTLIB_LAZY_MODULE_CLASS.__module__,
        "type_qualname": IMPORTLIB_LAZY_MODULE_CLASS.__qualname__,
        "declared_name": namespace["__name__"],
        "package": namespace["__package__"],
        "path": str(expected_path),
        "bytes": receipt["bytes"],
        "sha256": receipt["sha256"],
        "cached_relative_to_pycache_prefix": cached_suffix,
        "package_search_locations": [str(expected_package_path)],
        "spec_name": spec.name,
        "spec_origin": str(expected_path),
        "spec_parent": spec.parent,
        "spec_has_location": spec.has_location,
        "loader_type_module": IMPORTLIB_SOURCE_FILE_LOADER_CLASS.__module__,
        "loader_type_qualname": IMPORTLIB_SOURCE_FILE_LOADER_CLASS.__qualname__,
        "loader_name": loader.name,
        "loader_path": str(expected_path),
        "class_object_identity_checked": True,
        "loader_class_object_identity_checked": True,
        "loader_object_crossbound_to_spec": True,
    }


def _runtime_prefix_capture() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    exceptions: list[dict[str, Any]] = []
    for name in _runtime_prefix_keys():
        value = sys.modules.get(name)
        if type(value) is not types.ModuleType:
            exceptions.append(
                _runtime_prefix_nonmodule_exception_receipt(name, value)
            )
            continue
        if name in RUNTIME_PREFIX_NONMODULE_EXCEPTIONS:
            raise Stage15QualificationError(
                f"pinned runtime prefix exception was replaced: {name}"
            )
        if type(getattr(value, "__file__", None)) is not str:
            raise Stage15QualificationError(
                f"runtime prefix module is not exact file-backed: {name}"
            )
        records.append(_module_receipt(value, name))
    expected_exceptions = sorted(RUNTIME_PREFIX_NONMODULE_EXCEPTIONS)
    if [row["registry_name"] for row in exceptions] != expected_exceptions:
        raise Stage15QualificationError(
            "runtime prefix exception presence roster differs"
        )
    return records, exceptions


def _runtime_module_registry() -> list[dict[str, Any]]:
    records, _exceptions = _runtime_prefix_capture()
    return records


def _validate_runtime_prefix_exception_records(value: Any) -> None:
    if type(value) is not list or len(value) != len(
        RUNTIME_PREFIX_NONMODULE_EXCEPTIONS
    ):
        raise Stage15QualificationError(
            "retained runtime prefix exception roster differs"
        )
    if [row.get("registry_name") for row in value if type(row) is dict] != sorted(
        RUNTIME_PREFIX_NONMODULE_EXCEPTIONS
    ):
        raise Stage15QualificationError(
            "retained runtime prefix exception order differs"
        )
    for row in value:
        if type(row) is not dict:
            raise Stage15QualificationError(
                "retained runtime prefix exception row differs"
            )
        name = row.get("registry_name")
        binding = RUNTIME_PREFIX_NONMODULE_EXCEPTIONS.get(name)
        if type(binding) is not dict:
            raise Stage15QualificationError(
                "retained runtime prefix exception name differs"
            )
        if binding["kind"] == "EXACT_OBJECT_ALIAS":
            expected = {
                "registry_name": name,
                "kind": "EXACT_OBJECT_ALIAS",
                "type_module": binding["type_module"],
                "type_qualname": binding["type_qualname"],
                "owner_module": binding["owner_module"],
                "owner_attribute": binding["owner_attribute"],
                "owner_class_attribute": "_FFTConfig",
                "class_object_identity_checked": True,
                "owner_object_identity_checked": True,
            }
            if not _exact_equal(row, expected):
                raise Stage15QualificationError(
                    "retained runtime prefix object alias receipt differs"
                )
            continue
        expected_path = _lexical_absolute(
            PROJECT_SITE_PACKAGES / str(binding["relative_path"])
        )
        relative_source = Path(*expected_path.parts[1:])
        cached_suffix = str(
            relative_source.parent
            / f"{relative_source.stem}.{sys.implementation.cache_tag}.pyc"
        )
        expected = {
            "registry_name": name,
            "kind": "EXACT_LAZY_MODULE",
            "type_module": IMPORTLIB_LAZY_MODULE_CLASS.__module__,
            "type_qualname": IMPORTLIB_LAZY_MODULE_CLASS.__qualname__,
            "declared_name": name,
            "package": name,
            "path": str(expected_path),
            "bytes": binding["bytes"],
            "sha256": binding["sha256"],
            "cached_relative_to_pycache_prefix": cached_suffix,
            "package_search_locations": [str(expected_path.parent)],
            "spec_name": name,
            "spec_origin": str(expected_path),
            "spec_parent": name,
            "spec_has_location": True,
            "loader_type_module": IMPORTLIB_SOURCE_FILE_LOADER_CLASS.__module__,
            "loader_type_qualname": IMPORTLIB_SOURCE_FILE_LOADER_CLASS.__qualname__,
            "loader_name": name,
            "loader_path": str(expected_path),
            "class_object_identity_checked": True,
            "loader_class_object_identity_checked": True,
            "loader_object_crossbound_to_spec": True,
        }
        if not _exact_equal(row, expected):
            raise Stage15QualificationError(
                "retained lazy runtime prefix exception receipt differs"
            )
        _validate_retained_file_receipt(
            {key: row[key] for key in ("path", "bytes", "sha256")},
            f"retained lazy runtime prefix {name}",
        )


def _callable_receipt(module: types.ModuleType, name: str) -> dict[str, Any]:
    value = getattr(module, name, None)
    if not callable(value) or getattr(value, "__module__", None) != module.__name__:
        raise Stage15QualificationError(f"pinned callable changed: {name}")
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError) as exc:
        raise Stage15QualificationError(f"cannot bind callable source: {name}") from exc
    code = getattr(value, "__code__", None)
    if code is None:
        raise Stage15QualificationError(f"pinned callable has no Python code: {name}")
    return {
        "module": module.__name__, "name": name,
        "source_sha256": _sha256(source),
        "bytecode_sha256": _sha256(code.co_code),
        "first_line": code.co_firstlineno,
    }


def _runtime_guard_receipt(cache_root: Path) -> dict[str, Any]:
    root = _lexical_absolute(cache_root)
    pycache = root / "pycache-must-remain-absent"
    cupy_cache = root / "cupy-kernel-cache"
    return {
        "dont_write_bytecode": sys.dont_write_bytecode,
        "pycache_prefix": str(pycache),
        "pycache_absent": not pycache.exists() and not pycache.is_symlink(),
        "cupy_cache_directory": str(cupy_cache),
        "cupy_cache_entries": sorted(p.name for p in cupy_cache.iterdir()),
        "cupy_cache_in_memory": os.environ.get("CUPY_CACHE_IN_MEMORY"),
        "cuda_cache_disable": os.environ.get("CUDA_CACHE_DISABLE"),
        "claim": "POINT_OBSERVATION_IN_TRUSTED_PROCESS_NOT_CONCURRENT_MUTATION_PROOF",
    }


def _runtime_receipt(
    *, cache_root: Path, fresh_registry: Mapping[str, Any],
    numpy_module: types.ModuleType, scipy_module: types.ModuleType,
    cupy_module: types.ModuleType, engine_module: types.ModuleType,
    backend_spec: Any,
) -> dict[str, Any]:
    if (
        getattr(numpy_module, "__version__", None) != RUNTIME_VERSIONS["numpy"]
        or getattr(scipy_module, "__version__", None) != RUNTIME_VERSIONS["scipy"]
        or getattr(cupy_module, "__version__", None) != RUNTIME_VERSIONS["cupy"]
    ):
        raise Stage15QualificationError("runtime module version differs")
    _require_cuda_runtime_alias_invariant(cupy_module, sys.modules)
    critical = []
    for name in RUNTIME_IMPORT_MODULE_NAMES:
        value = sys.modules.get(name)
        if type(value) is not types.ModuleType:
            raise Stage15QualificationError(f"critical runtime module missing: {name}")
        critical.append(
            _nvrtc_module_receipt(value)
            if name == NVRTC_MODULE_NAME
            else (
                _cuda_runtime_module_receipt(value)
                if name == CUDA_RUNTIME_MODULE_NAME
                else _module_receipt(value, name)
            )
        )
    engine_runtime = dict(engine_module.cuda_massless_tracer_runtime_identity(backend_spec))
    _require_cuda_runtime_alias_invariant(cupy_module, sys.modules)
    module_registry, runtime_prefix_exceptions = _runtime_prefix_capture()
    record: dict[str, Any] = {
        "schema": RUNTIME_SCHEMA,
        "fresh_child_interpreter": _fresh_child_interpreter_receipt(),
        "fresh_process_registry": dict(fresh_registry),
        "guard": _runtime_guard_receipt(cache_root),
        "distributions": [
            _distribution_receipt("numpy"), _distribution_receipt("scipy"),
            _distribution_receipt("cupy-cuda13x"),
        ],
        "critical_modules": critical,
        "module_registry": module_registry,
        "runtime_prefix_exceptions": runtime_prefix_exceptions,
        "adapter_callables": [
            _callable_receipt(engine_module, "_path_kernels"),
            _callable_receipt(engine_module, "_build_source_path"),
            _callable_receipt(engine_module, "advance_cuda_massless_tracers"),
            _callable_receipt(engine_module, "_source_segment_sha256"),
            _callable_receipt(engine_module, "_validate_route_witness"),
            _callable_receipt(engine_module, "_validate_continuation"),
            _callable_receipt(
                engine_module, "_host_binary64_active_endpoint_minimum"
            ),
            _callable_receipt(
                engine_module, "_canonicalize_active_endpoint_minimum_host"
            ),
        ],
        "engine_runtime_identity": engine_runtime,
        "implementation": _file_receipt(SCRIPT_PATH, "Stage-1.5 v5 implementation"),
        "claim": "EXACT_FRESH_PROCESS_RUNTIME_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION",
    }
    record["content_sha256"] = _domain_sha(RUNTIME_DOMAIN, record)
    return record


def _runtime_process_independent_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    if type(value) is not dict or value.get("schema") != RUNTIME_SCHEMA:
        raise Stage15QualificationError("runtime receipt schema differs")
    return {
        "distributions": value.get("distributions"),
        "critical_modules": value.get("critical_modules"),
        "module_registry": value.get("module_registry"),
        "runtime_prefix_exceptions": value.get("runtime_prefix_exceptions"),
        "adapter_callables": value.get("adapter_callables"),
        "engine_runtime_identity": value.get("engine_runtime_identity"),
        "implementation": value.get("implementation"),
    }


def _runtime_projection_without_module_registry(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    projection = _runtime_process_independent_projection(value)
    projection.pop("module_registry")
    return projection


def _validate_post_execution_module_registry_transition(
    before: Any, after: Any,
) -> dict[str, Any]:
    """Validate the only permitted before/after runtime projection change.

    CuPy may import additional file-backed implementation modules while it
    compiles or launches the first real kernel.  Such imports are evidence,
    not runtime drift: every pre-execution row must remain byte-for-byte
    present, and every newly observed row is retained and reread from one of
    the already pinned runtime roots.  Removal, mutation, aliases, spec-less
    additions, or modules outside the declared runtime prefixes fail closed.
    """

    if type(before) is not list or type(after) is not list:
        raise Stage15QualificationError("runtime module registry is absent")

    def indexed(value: list[Any], label: str) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        observed_order: list[str] = []
        for row in value:
            if type(row) is not dict or set(row) != {
                "module", "declared_module_name", "spec_name", "spec_origin",
                "loader_type", "path", "bytes", "sha256",
            }:
                raise Stage15QualificationError(
                    f"{label} runtime module registry row differs"
                )
            name = row.get("module")
            if (
                type(name) is not str
                or not any(
                    name == prefix or name.startswith(prefix + ".")
                    for prefix in RUNTIME_PREFIXES
                )
                or name in rows
            ):
                raise Stage15QualificationError(
                    f"{label} runtime module registry identity differs"
                )
            rows[name] = row
            observed_order.append(name)
        if observed_order != sorted(observed_order):
            raise Stage15QualificationError(
                f"{label} runtime module registry order differs"
            )
        return rows

    before_rows = indexed(before, "pre-execution")
    after_rows = indexed(after, "post-execution")
    removed = sorted(set(before_rows) - set(after_rows))
    changed = sorted(
        name for name in set(before_rows) & set(after_rows)
        if not _exact_equal(before_rows[name], after_rows[name])
    )
    if removed or changed:
        raise Stage15QualificationError(
            "post-execution runtime module registry removed or mutated a module"
        )
    added = sorted(set(after_rows) - set(before_rows))
    for name in added:
        row = after_rows[name]
        if (
            row.get("declared_module_name") != name
            or row.get("spec_name") != name
            or row.get("spec_origin") != row.get("path")
            or type(row.get("loader_type")) is not str
            or name in MODULE_DECLARED_NAME_ALIASES
            or name in SPECLESS_FILE_BACKED_MODULES
        ):
            raise Stage15QualificationError(
                "post-execution runtime module registry addition is not an exact module"
            )
        _validate_retained_file_receipt(
            {key: row.get(key) for key in ("path", "bytes", "sha256")},
            f"post-execution runtime module {name}",
        )
    return {
        "policy": (
            "PRE_EXECUTION_EXACT_UNCHANGED_SUBSET_WITH_EXACT_"
            "LIVE_FILE_BACKED_ADDITIONS"
        ),
        "before_count": len(before_rows),
        "after_count": len(after_rows),
        "added_module_names": added,
        "added_module_receipts_sha256": _sha256(
            _canonical_json([after_rows[name] for name in added])
        ),
    }


def _normalized_runtime_guard(value: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "dont_write_bytecode", "pycache_prefix", "pycache_absent",
        "cupy_cache_directory", "cupy_cache_entries", "cupy_cache_in_memory",
        "cuda_cache_disable", "claim",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise Stage15QualificationError("runtime guard receipt shape differs")
    pycache_value = value.get("pycache_prefix")
    cupy_value = value.get("cupy_cache_directory")
    if type(pycache_value) is not str or type(cupy_value) is not str:
        raise Stage15QualificationError("runtime guard path differs")
    if not Path(pycache_value).is_absolute() or not Path(cupy_value).is_absolute():
        raise Stage15QualificationError("runtime guard path must be absolute")
    pycache = _lexical_absolute(Path(pycache_value))
    cupy_cache = _lexical_absolute(Path(cupy_value))
    if (
        pycache.name != "pycache-must-remain-absent"
        or cupy_cache.name != "cupy-kernel-cache"
        or pycache.parent != cupy_cache.parent
        or value.get("dont_write_bytecode") is not True
        or value.get("pycache_absent") is not True
        or value.get("cupy_cache_entries") != []
        or value.get("cupy_cache_in_memory") != "1"
        or value.get("cuda_cache_disable") != "1"
        or value.get("claim")
        != "POINT_OBSERVATION_IN_TRUSTED_PROCESS_NOT_CONCURRENT_MUTATION_PROOF"
    ):
        raise Stage15QualificationError("runtime guard security values differ")
    return {
        "dont_write_bytecode": True,
        "pycache_leaf": pycache.name,
        "pycache_absent": True,
        "cupy_cache_leaf": cupy_cache.name,
        "cupy_cache_entries": [],
        "cupy_cache_in_memory": "1",
        "cuda_cache_disable": "1",
        "claim": value["claim"],
    }


def _import_runtime_modules_only(
    cache_root: Path,
) -> tuple[dict[str, Any], dict[str, types.ModuleType]]:
    fresh = _reject_preloaded_runtime_modules()
    expected_pycache = _lexical_absolute(cache_root) / "pycache-must-remain-absent"
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(expected_pycache)
    modules: dict[str, types.ModuleType] = {}
    try:
        for name in RUNTIME_IMPORT_MODULE_NAMES:
            module = importlib.import_module(name)
            if type(module) is not types.ModuleType or module.__name__ != name:
                raise Stage15QualificationError(
                    f"fresh runtime import is not the exact module: {name}"
                )
            modules[name] = module
    except Exception as exc:
        raise Stage15QualificationError("fresh dynamics runtime import failed") from exc
    if tuple(modules) != RUNTIME_IMPORT_MODULE_NAMES:
        raise Stage15QualificationError("fresh runtime import roster differs")
    _require_cuda_runtime_alias_invariant(modules["cupy"], modules)
    return fresh, modules


def _fresh_runtime_imports(cache_root: Path) -> tuple[Any, ...]:
    fresh, modules = _import_runtime_modules_only(cache_root)
    np = modules["numpy"]
    scipy = modules["scipy"]
    cp = modules["cupy"]
    engine = modules["jxplanetx.cuda_massless_tracer"]
    backend_module = modules["jxplanetx.engine.backends"]
    contract_module = modules["jxplanetx.engine.contracts"]
    resolve_backend = getattr(backend_module, "resolve_backend", None)
    BackendSpec = getattr(contract_module, "BackendSpec", None)
    Provenance = getattr(contract_module, "Provenance", None)
    if not callable(resolve_backend) or not isinstance(BackendSpec, type) or not isinstance(Provenance, type):
        raise Stage15QualificationError("fresh runtime API surface differs")
    backend_spec = BackendSpec("cupy", "cuda:0", 128)
    backend = resolve_backend(backend_spec)
    runtime = _runtime_receipt(
        cache_root=cache_root, fresh_registry=fresh,
        numpy_module=np, scipy_module=scipy, cupy_module=cp,
        engine_module=engine, backend_spec=backend_spec,
    )
    return np, scipy, cp, engine, backend_spec, backend, Provenance, runtime


def _require_bitwise_endpoint(
    cp: types.ModuleType, path_positions: Any, path_velocities: Any,
    returned_state: Any,
) -> tuple[str, str]:
    if not bool(cp.asnumpy(cp.array_equal(path_positions[-1], returned_state.source_positions))):
        raise Stage15QualificationError("public source-position endpoint differs from precomputed path")
    if not bool(cp.asnumpy(cp.array_equal(path_velocities[-1], returned_state.source_velocities))):
        raise Stage15QualificationError("public source-velocity endpoint differs from precomputed path")
    position = cp.asnumpy(path_positions[-1]).astype("<f8", copy=False).tobytes(order="C")
    velocity = cp.asnumpy(path_velocities[-1]).astype("<f8", copy=False).tobytes(order="C")
    return _sha256(position), _sha256(velocity)


def _segment_source_payload(np: types.ModuleType, positions: Any, velocities: Any, *, include_first: bool) -> bytes:
    p = np.ascontiguousarray(positions, dtype="<f8")
    v = np.ascontiguousarray(velocities, dtype="<f8")
    if p.shape != v.shape or p.ndim != 3 or p.shape[1:] != (MASSIVE_COUNT, 3):
        raise Stage15QualificationError("private source path shape differs")
    if not include_first:
        p, v = p[1:], v[1:]
    interleaved = np.empty((p.shape[0], 2, MASSIVE_COUNT, 3), dtype="<f8")
    interleaved[:, 0] = p
    interleaved[:, 1] = v
    return interleaved.tobytes(order="C")


def _schedule_sha(batch: Any, steps: int) -> str:
    value = {
        "ephemeris": "DE441", "batch_order": batch.batch_order,
        "object_id": batch.object_id, "native_epoch_jd_tdb_hex": batch.epoch_jd_tdb.hex(),
        "steps_per_julian_year": steps,
        "massive": [
            {
                "naif_id": item.naif_id, "gm_hex": item.gm_AU3_per_julian_year2.hex(),
                "position_hex": [x.hex() for x in item.position_AU],
                "velocity_hex": [x.hex() for x in item.velocity_AU_per_julian_year],
            }
            for item in batch.massive
        ],
    }
    return _sha256(SCHEDULE_DOMAIN + _canonical_json(value))


def _make_engine_state(
    *, np: types.ModuleType, cp: types.ModuleType, engine: types.ModuleType,
    Provenance: type, batch: Any, steps: int,
) -> Any:
    schedule = _schedule_sha(batch, steps)
    gm = np.asarray([item.gm_AU3_per_julian_year2 for item in batch.massive], dtype=np.float64)
    bound = math.nextafter(2.0 * math.fsum(float(x) for x in gm), math.inf)
    provenance = Provenance(
        source_id=f"jx-tno-stage15-v5-de441-batch-{batch.batch_order}-h{steps}",
        citation="retained Stage-0/DE441/Stage-1-v4 lineage",
        version="stage15-v5", sha256=schedule,
    )
    return engine.CUDAMasslessTracerState(
        snapshot_id=f"stage15-v5-{batch.object_id}-h{steps}-year-0",
        epoch=(batch.epoch_jd_tdb - 2_451_545.0) / 365.25,
        global_step=0, time_scale="TDB", frame="ICRF", origin="RESOLVED11_COMMON_GM_BARYCENTER",
        axes="J2000_EQUATORIAL", length_unit="AU", time_unit="year", unit_system_id="AU-YEAR",
        source_ids=tuple(item.body_id for item in batch.massive), primary_source_id=batch.massive[0].body_id,
        source_positions=cp.asarray(np.asarray([item.position_AU for item in batch.massive], dtype=np.float64)),
        source_velocities=cp.asarray(np.asarray([item.velocity_AU_per_julian_year for item in batch.massive], dtype=np.float64)),
        source_gm=cp.asarray(gm), source_encounter_radii=cp.ones(MASSIVE_COUNT, dtype=cp.float64),
        source_collision_radii=cp.zeros(MASSIVE_COUNT, dtype=cp.float64),
        relative_acceleration_bounds=cp.full(MASSIVE_COUNT, bound, dtype=cp.float64),
        tracer_positions=cp.asarray(np.asarray([item.position_AU for item in batch.tracers], dtype=np.float64)),
        tracer_velocities=cp.asarray(np.asarray([item.velocity_AU_per_julian_year for item in batch.tracers], dtype=np.float64)),
        tracer_indices=cp.arange(TRACER_COUNT, dtype=cp.uint64), provenance=provenance,
        force_schedule_sha256=schedule, matched_schedule_set_sha256=schedule,
        primary_outer_boundary_AU=4096.0,
    )


def _annual_spec(engine: types.ModuleType, backend_spec: Any, steps: int) -> Any:
    return engine.FixedStepCUDAMasslessTracerSpec(
        step_years=1.0 / steps, step_count=steps, backend=backend_spec,
        maximum_encounter_routes_per_tracer=16_384,
        maximum_total_encounter_routes=100_000,
        encounter_route_rtol=1e-12, encounter_route_atol=1e-14,
        encounter_route_maximum_substep_fraction=0.125,
        maximum_primary_dynamical_phase=0.05,
        event_time_tolerance_fraction=1e-12,
        source_path_steps_per_chunk=steps, threads_per_block=128,
    )


def _advance_one_annual_segment(
    *, np: types.ModuleType, cp: types.ModuleType, engine: types.ModuleType,
    backend: Any, state: Any, spec: Any, year: int,
) -> tuple[Any, bytes, dict[str, Any]]:
    _require_int(year, "annual segment year", minimum=0)
    with backend.activate():
        validated = engine._validate_state(backend, state)
        source_kernel, _tracer_kernel = engine._path_kernels(backend)
        private_path = engine._build_source_path(
            backend, source_kernel, validated.source_positions,
            validated.source_velocities, validated.source_gm,
            step_count=spec.step_count, step=spec.step_years,
        )
        private_positions = cp.asnumpy(private_path.positions)
        private_velocities = cp.asnumpy(private_path.velocities)
        result = engine.advance_cuda_massless_tracers(state, spec)
        endpoint_position_sha, endpoint_velocity_sha = _require_bitwise_endpoint(
            cp, private_path.positions, private_path.velocities, result.state,
        )
    payload = _segment_source_payload(
        np, private_positions, private_velocities, include_first=(year == 0),
    )
    receipt = {
        "year_start": year, "year_end": year + 1,
        "starting_global_step": year * spec.step_count,
        "ending_global_step": (year + 1) * spec.step_count,
        "private_path_node_count": spec.step_count + 1,
        "retained_unique_node_count": spec.step_count + (1 if year == 0 else 0),
        "private_path_payload_sha256": _sha256(_segment_source_payload(np, private_positions, private_velocities, include_first=True)),
        "retained_payload_sha256": _sha256(payload),
        "precomputed_endpoint_position_sha256": endpoint_position_sha,
        "public_endpoint_position_sha256": endpoint_position_sha,
        "precomputed_endpoint_velocity_sha256": endpoint_velocity_sha,
        "public_endpoint_velocity_sha256": endpoint_velocity_sha,
        "public_input_state_sha256": result.input_state_sha256,
        "bitwise_source_endpoint_match": True,
    }
    return result, payload, receipt


def _load_v4_context() -> tuple[types.ModuleType, dict[str, Any], dict[str, Any], tuple[Any, ...], list[dict[str, Any]]]:
    v4 = _load_source_module(V4_SOURCE_PATH, V4_SOURCE_SHA256, "_jx_stage15_v4_bound_stage1_v4")
    bridge = _load_canonical_json(
        _regular_bytes(BRIDGE_PATH, "resolved-11 bridge", exact_bytes=168_684),
        "resolved-11 bridge",
    )
    custody = _load_canonical_json(
        _regular_bytes(DE441_CUSTODY_PATH, "DE441 custody", exact_bytes=141_501),
        "DE441 custody",
    )
    batches = v4._batch_states(bridge, custody, "DE441")
    if (
        type(batches) is not tuple or len(batches) != 12
        or tuple(item.object_id for item in batches) != OBJECT_IDS
        or any(len(item.massive) != MASSIVE_COUNT or len(item.tracers) != TRACER_COUNT for item in batches)
    ):
        raise Stage15QualificationError("Stage-1 v4 DE441 batch reconstruction changed")
    v1 = _load_canonical_json(
        _regular_bytes(V1_PROTOCOL_PATH, "v1 protocol", exact_bytes=31_382),
        "v1 protocol",
    )
    locators = v1.get("inputs", {}).get("stage1_v4_de441_resolved11_ias15_oracle_lanes")
    if type(locators) is not list or len(locators) != 12:
        raise Stage15QualificationError("v1 oracle locator roster changed")
    return v4, bridge, custody, batches, locators


def _checkpoint_and_payload(
    v4: types.ModuleType, np: types.ModuleType, cp: types.ModuleType,
    batch: Any, state: Any, year: int,
) -> tuple[dict[str, Any], bytes]:
    source_position = cp.asnumpy(state.source_positions)
    source_velocity = cp.asnumpy(state.source_velocities)
    tracer_position = cp.asnumpy(state.tracer_positions)
    tracer_velocity = cp.asnumpy(state.tracer_velocities)
    massive = tuple(
        v4.MassiveState(
            item.naif_id, item.body_id, item.gm_AU3_per_julian_year2,
            tuple(float(x) for x in source_position[index]),
            tuple(float(x) for x in source_velocity[index]),
        )
        for index, item in enumerate(batch.massive)
    )
    tracers = tuple(
        v4.TracerState(
            item.object_id, item.spk_id, item.sample_kind, item.clone_index,
            tuple(float(x) for x in tracer_position[index]),
            tuple(float(x) for x in tracer_velocity[index]),
        )
        for index, item in enumerate(batch.tracers)
    )
    checkpoint = v4._checkpoint(year, massive, tracers)
    values: list[float] = []
    for key in (
        "massive_positions_AU", "massive_velocities_AU_per_julian_year",
        "tracer_positions_AU", "tracer_velocities_AU_per_julian_year",
    ):
        values.extend(float(component) for row in checkpoint[key] for component in row)
    if len(values) != ARRAY_RECORD_DOUBLES:
        raise Stage15QualificationError("annual Cartesian record width changed")
    return checkpoint, struct.pack(f"<{ARRAY_RECORD_DOUBLES}d", *values)


def _raw_tracer_payload(cp: types.ModuleType, state: Any) -> bytes:
    positions = cp.asnumpy(state.tracer_positions).astype("<f8", copy=False)
    velocities = cp.asnumpy(state.tracer_velocities).astype("<f8", copy=False)
    if positions.shape != (TRACER_COUNT, 3) or velocities.shape != positions.shape:
        raise Stage15QualificationError("raw tracer state shape changed")
    return positions.tobytes(order="C") + velocities.tobytes(order="C")


def _state_sha256(cp: types.ModuleType, state: Any) -> str:
    digest = hashlib.sha256(b"jx.tno.stage15.state.v5\0")
    for value in (
        state.source_positions, state.source_velocities,
        state.tracer_positions, state.tracer_velocities,
        state.minimum_source_distance_AU, state.ever_unbound,
        state.encounter_route_counts, state.tracer_lifecycle,
    ):
        if value is None:
            digest.update(b"NONE\0")
        else:
            array = cp.asnumpy(value)
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(str(array.shape).encode("ascii") + b"\0")
            digest.update(array.tobytes(order="C"))
    digest.update(struct.pack("<qd", state.global_step, state.epoch))
    return digest.hexdigest()


def _route_line(
    *, lane_order: int, batch_order: int, lane_id: str, object_id: str,
    segment_year: int, kind: str, record_order: int, payload: Mapping[str, Any],
) -> bytes:
    if kind not in ("ROUTE", "REMOVAL"):
        raise Stage15QualificationError("route evidence kind changed")
    row: dict[str, Any] = {
        "schema": ROUTE_SCHEMA, "record_order": record_order,
        "lane_order": lane_order, "batch_order": batch_order,
        "lane_id": lane_id, "object_id": object_id,
        "segment_year": segment_year, "kind": kind,
        "record": dict(payload),
    }
    row["content_sha256"] = _domain_sha(ROUTE_DOMAIN, row)
    line = _canonical_json(row)
    if len(line) > ROUTE_MAX_LINE_BYTES:
        raise Stage15QualificationError("route evidence line exceeds cap")
    return line


def _append_new_route_records(
    writer: _CappedBinaryWriter, *, lane_order: int, batch_order: int,
    lane_id: str, object_id: str, year: int, result: Any,
    prior_route_count: int, prior_removal_count: int, global_record_order: int,
) -> tuple[int, int, int, bytes]:
    routes = result.encounter_routes
    removals = result.removal_events
    if type(routes) is not tuple or type(removals) is not tuple:
        raise Stage15QualificationError("engine route/removal ledger type changed")
    if len(routes) < prior_route_count or len(removals) < prior_removal_count:
        raise Stage15QualificationError("cumulative route/removal ledger shrank")
    lane_lines = bytearray()
    for kind, rows, start in (
        ("ROUTE", routes, prior_route_count),
        ("REMOVAL", removals, prior_removal_count),
    ):
        for row in rows[start:]:
            if writer.record_count + 1 > ROUTE_MAX_RECORDS:
                raise Stage15QualificationError(
                    "experiment-global route/removal record cap would be exceeded before write"
                )
            line = _route_line(
                lane_order=lane_order, batch_order=batch_order,
                lane_id=lane_id, object_id=object_id, segment_year=year,
                kind=kind, record_order=global_record_order,
                payload=asdict(row),
            )
            writer.write(line)
            writer.record_count += 1
            lane_lines.extend(line)
            global_record_order += 1
    return len(routes), len(removals), global_record_order, bytes(lane_lines)


def _midpoint_continuation(cp: types.ModuleType, state: Any) -> dict[str, Any]:
    def vector_hex(value: Any, dtype: Any) -> list[str]:
        array = cp.asnumpy(value).astype(dtype, copy=False)
        return [float(x).hex() for x in array]
    route_rows = [asdict(row) for row in state.prior_encounter_routes]
    removal_rows = [asdict(row) for row in state.prior_removal_events]
    return {
        "global_step": state.global_step, "epoch_hex": float(state.epoch).hex(),
        "minimum_source_distance_AU_hex": vector_hex(state.minimum_source_distance_AU, "<f8"),
        "ever_unbound": [bool(x) for x in cp.asnumpy(state.ever_unbound)],
        "encounter_route_counts": [int(x) for x in cp.asnumpy(state.encounter_route_counts)],
        "tracer_lifecycle": [int(x) for x in cp.asnumpy(state.tracer_lifecycle)],
        "prior_encounter_route_count": len(route_rows),
        "prior_removal_event_count": len(removal_rows),
        "route_prefix_sha256": _sha256(_canonical_json(route_rows)),
        "removal_prefix_sha256": _sha256(_canonical_json(removal_rows)),
    }


def _source_summary_from_fd(
    *, fd: int, offset: int, byte_length: int, node_count: int,
    np: types.ModuleType, v4: types.ModuleType, gm_values: Sequence[float],
) -> dict[str, Any]:
    payload = _pread_all(fd, byte_length, offset)
    values = np.frombuffer(payload, dtype="<f8")
    expected = node_count * 2 * MASSIVE_COUNT * 3
    if values.size != expected:
        raise Stage15QualificationError("source-node payload width differs")
    rows = values.reshape(node_count, 2, MASSIVE_COUNT, 3)
    return v4._source_node_invariant_summary(
        rows[:, 0], rows[:, 1], np.asarray(gm_values, dtype=np.float64),
        numpy_module=np,
    )


def _run_lane(
    *, lane_order: int, batch: Any, lane_id: str, steps: int,
    np: types.ModuleType, cp: types.ModuleType, engine: types.ModuleType,
    backend_spec: Any, backend: Any, Provenance: type,
    v4: types.ModuleType, arrays_writer: _CappedBinaryWriter,
    source_writer: _CappedBinaryWriter, route_writer: _CappedBinaryWriter,
    midpoint_writer: _CappedBinaryWriter, runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes | None]:
    state = _make_engine_state(
        np=np, cp=cp, engine=engine, Provenance=Provenance,
        batch=batch, steps=steps,
    )
    spec = _annual_spec(engine, backend_spec, steps)
    source_offset = source_writer.count
    arrays_offset = arrays_writer.count
    routes_offset = route_writer.count
    midpoint_offset = midpoint_writer.count
    segment_receipts: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    initial_checkpoint, initial_payload = _checkpoint_and_payload(v4, np, cp, batch, state, 0)
    checkpoints.append(initial_checkpoint)
    arrays_writer.write(initial_payload)
    route_count = 0
    removal_count = 0
    global_route_order = 0
    lane_route_digest = hashlib.sha256()
    monolithic_suffix = bytearray()
    midpoint: dict[str, Any] | None = None
    selected_restart = (lane_id, batch.object_id) in RESTART_CASES
    for year in range(HORIZON_YEARS):
        result, source_payload, segment = _advance_one_annual_segment(
            np=np, cp=cp, engine=engine, backend=backend,
            state=state, spec=spec, year=year,
        )
        if dict(result.runtime_identity) != runtime["engine_runtime_identity"]:
            raise Stage15QualificationError("lane runtime identity differs from retained runtime")
        source_writer.write(source_payload)
        route_count, removal_count, global_route_order, new_lines = _append_new_route_records(
            route_writer, lane_order=lane_order, batch_order=batch.batch_order,
            lane_id=lane_id, object_id=batch.object_id, year=year,
            result=result, prior_route_count=route_count,
            prior_removal_count=removal_count,
            global_record_order=global_route_order,
        )
        lane_route_digest.update(new_lines)
        state = result.state
        if state.global_step != (year + 1) * steps:
            raise Stage15QualificationError("public result global step differs")
        checkpoint, checkpoint_payload = _checkpoint_and_payload(
            v4, np, cp, batch, state, year + 1,
        )
        checkpoints.append(checkpoint)
        arrays_writer.write(checkpoint_payload)
        raw_tracer = _raw_tracer_payload(cp, state)
        if year + 1 == 500:
            midpoint_writer.write(raw_tracer)
            midpoint = {
                "midpoint_order": lane_order,
                "byte_offset": midpoint_offset,
                "byte_length": MIDPOINT_RECORD_BYTES,
                "record_sha256": _sha256(raw_tracer),
                "batch_order": batch.batch_order, "object_id": batch.object_id,
                "lane_id": lane_id, "steps_per_julian_year": steps,
                "midpoint_year": 500, "global_step": 500 * steps,
                "continuation": _midpoint_continuation(cp, state),
            }
        if selected_restart and year + 1 >= 500:
            monolithic_suffix.extend(raw_tracer)
        segment_receipts.append(segment)
        if removal_count:
            raise Stage15QualificationError("a tracer removal fails Stage-1.5 qualification")
        lifecycle = cp.asnumpy(state.tracer_lifecycle)
        if lifecycle.shape != (TRACER_COUNT,) or not bool(np.all(lifecycle == 0)):
            raise Stage15QualificationError("tracer lifecycle is not fully active")
    if midpoint is None:
        raise Stage15QualificationError("lane midpoint was not captured")
    expected_source_bytes = (
        SOURCE_H64_LANE_BYTES if steps == 64 else SOURCE_H128_LANE_BYTES
    )
    if source_writer.count - source_offset != expected_source_bytes:
        raise Stage15QualificationError("lane source evidence length differs")
    if arrays_writer.count - arrays_offset != ARRAY_LANE_BYTES:
        raise Stage15QualificationError("lane Cartesian evidence length differs")
    if selected_restart and len(monolithic_suffix) != RESTART_STREAM_BYTES:
        raise Stage15QualificationError("monolithic restart suffix length differs")
    lane = {
        "lane_order": lane_order, "batch_order": batch.batch_order,
        "object_id": batch.object_id, "lane_id": lane_id,
        "engine": LANE_ENGINE,
        "model": LANE_MODEL, "steps_per_julian_year": steps,
        "step_julian_years_hex": (1.0 / steps).hex(),
        "final_global_step": HORIZON_YEARS * steps,
        "source_nodes": {
            "schema": SOURCE_SCHEMA, "byte_offset": source_offset,
            "byte_length": expected_source_bytes,
            "node_count": HORIZON_YEARS * steps + 1,
            "record_bytes": SOURCE_NODE_RECORD_BYTES,
            "lane_payload_sha256": "PENDING_HELD_FD_FINALIZATION",
            "annual_segment_receipts": segment_receipts,
        },
        "cartesian_array": {
            "schema": ARRAY_SCHEMA, "byte_offset": arrays_offset,
            "byte_length": ARRAY_LANE_BYTES,
            "checkpoint_count": CHECKPOINT_COUNT,
            "record_bytes": ARRAY_RECORD_BYTES,
            "lane_payload_sha256": "PENDING_HELD_FD_FINALIZATION",
        },
        "routes": {
            "schema": ROUTE_SCHEMA, "byte_offset": routes_offset,
            "byte_length": route_writer.count - routes_offset,
            "record_count": global_route_order,
            "lane_payload_sha256": lane_route_digest.hexdigest(),
            "route_count": route_count, "removal_count": removal_count,
        },
        "midpoint": midpoint,
        "annual_summary": v4._lane_summary(checkpoints),
        "all_node_summary": "PENDING_HELD_FD_FINALIZATION",
        "final_state_sha256": _state_sha256(cp, state),
        "final_route_ledger_sha256": _sha256(_canonical_json([asdict(row) for row in state.prior_encounter_routes])),
        "final_removal_ledger_sha256": _sha256(_canonical_json([asdict(row) for row in state.prior_removal_events])),
        "runtime_projection_sha256": _sha256(_canonical_json(_runtime_process_independent_projection(runtime))),
        "no_removal_no_drop": removal_count == 0,
    }
    return lane, checkpoints, (bytes(monolithic_suffix) if selected_restart else None)


def _decode_checkpoint_payload(
    payload: bytes, *, year: int, batch: Any, v4: types.ModuleType,
) -> dict[str, Any]:
    if len(payload) != ARRAY_RECORD_BYTES:
        raise Stage15QualificationError("Cartesian record byte length differs")
    values = struct.unpack(f"<{ARRAY_RECORD_DOUBLES}d", payload)
    cursor = 0

    def rows(count: int) -> tuple[tuple[float, float, float], ...]:
        nonlocal cursor
        output = tuple(
            (values[cursor + index * 3], values[cursor + index * 3 + 1], values[cursor + index * 3 + 2])
            for index in range(count)
        )
        cursor += count * 3
        return output

    massive_position = rows(MASSIVE_COUNT)
    massive_velocity = rows(MASSIVE_COUNT)
    tracer_position = rows(TRACER_COUNT)
    tracer_velocity = rows(TRACER_COUNT)
    if cursor != ARRAY_RECORD_DOUBLES:
        raise Stage15QualificationError("Cartesian record has trailing values")
    massive = tuple(
        v4.MassiveState(
            item.naif_id, item.body_id, item.gm_AU3_per_julian_year2,
            massive_position[index], massive_velocity[index],
        )
        for index, item in enumerate(batch.massive)
    )
    tracers = tuple(
        v4.TracerState(
            item.object_id, item.spk_id, item.sample_kind, item.clone_index,
            tracer_position[index], tracer_velocity[index],
        )
        for index, item in enumerate(batch.tracers)
    )
    return v4._checkpoint_from_centered(year, massive, tracers)


def _decode_lane_from_fd(
    fd: int, locator: Mapping[str, Any], *, batch: Any,
    v4: types.ModuleType,
) -> dict[str, Any]:
    if type(locator) is not dict:
        raise Stage15QualificationError("Cartesian locator is absent")
    offset = _require_int(locator.get("byte_offset"), "Cartesian byte offset", minimum=0)
    length = _require_int(locator.get("byte_length"), "Cartesian byte length", minimum=1)
    if length != ARRAY_LANE_BYTES:
        raise Stage15QualificationError("Cartesian lane byte length differs")
    payload = _pread_all(fd, length, offset)
    if _sha256(payload) != locator.get("lane_payload_sha256"):
        raise Stage15QualificationError("Cartesian lane digest differs")
    checkpoints = [
        _decode_checkpoint_payload(
            payload[year * ARRAY_RECORD_BYTES : (year + 1) * ARRAY_RECORD_BYTES],
            year=year, batch=batch, v4=v4,
        )
        for year in range(CHECKPOINT_COUNT)
    ]
    return {"model": LANE_MODEL, "checkpoints": checkpoints}


def _decode_v4_oracle_lane(
    fd: int, locator: Mapping[str, Any], *, batch: Any,
    v4: types.ModuleType,
) -> dict[str, Any]:
    if type(locator) is not dict:
        raise Stage15QualificationError("v4 oracle locator is absent")
    offset = _require_int(locator.get("byte_offset"), "v4 oracle offset", minimum=0)
    length = _require_int(locator.get("byte_length"), "v4 oracle length", minimum=1)
    if (
        length != ARRAY_LANE_BYTES
        or locator.get("checkpoint_count") != CHECKPOINT_COUNT
        or locator.get("record_bytes") != ARRAY_RECORD_BYTES
        or locator.get("massive_count") != MASSIVE_COUNT
        or locator.get("tracer_count") != TRACER_COUNT
    ):
        raise Stage15QualificationError("v4 oracle locator shape differs")
    payload = _pread_all(fd, length, offset)
    if _sha256(payload) != locator.get("lane_payload_sha256"):
        raise Stage15QualificationError("v4 oracle lane digest differs")
    checkpoints = [
        _decode_checkpoint_payload(
            payload[year * ARRAY_RECORD_BYTES : (year + 1) * ARRAY_RECORD_BYTES],
            year=year, batch=batch, v4=v4,
        )
        for year in range(CHECKPOINT_COUNT)
    ]
    return {"model": LANE_MODEL, "checkpoints": checkpoints}


def _massive_series(lane: Mapping[str, Any], source_order: int) -> tuple[Any, Any]:
    _require_int(source_order, "source order", minimum=0)
    if source_order >= MASSIVE_COUNT:
        raise Stage15QualificationError("source order is outside resolved-11 roster")
    checkpoints = lane.get("checkpoints")
    if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT:
        raise Stage15QualificationError("massive comparison checkpoint roster differs")
    positions = []
    velocities = []
    for year, checkpoint in enumerate(checkpoints):
        if type(checkpoint) is not dict or checkpoint.get("year_index") != year:
            raise Stage15QualificationError("massive comparison year differs")
        positions.append(tuple(checkpoint["massive_positions_AU"][source_order]))
        velocities.append(tuple(checkpoint["massive_velocities_AU_per_julian_year"][source_order]))
    return tuple(positions), tuple(velocities)


def _massive_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], source_order: int,
    v4: types.ModuleType,
) -> dict[str, Fraction]:
    left_r, left_v = _massive_series(left, source_order)
    right_r, right_v = _massive_series(right, source_order)
    return {
        "position": max(v4._exact_squared_vector_difference(a, b) for a, b in zip(left_r, right_r, strict=True)),
        "velocity": max(v4._exact_squared_vector_difference(a, b) for a, b in zip(left_v, right_v, strict=True)),
    }


def _massive_floors(reference: Mapping[str, Any], source_order: int, v4: types.ModuleType) -> dict[str, Fraction]:
    positions, velocities = _massive_series(reference, source_order)
    return {
        "position": max(v4._exact_squared_reference_ulp_floor(value) for value in positions),
        "velocity": max(v4._exact_squared_reference_ulp_floor(value) for value in velocities),
    }


def _evaluate_gates(
    generated: Mapping[tuple[int, int], Mapping[str, Any]],
    oracle: Mapping[int, Mapping[str, Any]],
    lane_records: Sequence[Mapping[str, Any]], v4: types.ModuleType,
) -> dict[str, Any]:
    if set(generated) != {(batch, steps) for batch in range(12) for steps in (64, 128)}:
        raise Stage15QualificationError("generated gate lane roster differs")
    if set(oracle) != set(range(12)):
        raise Stage15QualificationError("oracle gate lane roster differs")
    fraction_obs: dict[str, list[Fraction]] = {
        "richardson_a": [], "fine_ias_a": [], "fine_ias_scattering": [],
        "fine_energy": [],
    }
    squared_obs: dict[str, list[Fraction]] = {
        "richardson_r": [], "richardson_v": [],
        "fine_ias_r": [], "fine_ias_v": [],
        "source_richardson_r": [], "source_richardson_v": [],
        "source_fine_ias_r": [], "source_fine_ias_v": [],
    }
    contractions: dict[str, list[bool]] = {
        "tracer_a": [], "tracer_r": [], "tracer_v": [],
        "source_r": [], "source_v": [], "energy": [],
    }
    angular_values: list[float] = []
    lane_by_key = {
        (record["batch_order"], record["steps_per_julian_year"]): record
        for record in lane_records
    }
    if len(lane_by_key) != 24:
        raise Stage15QualificationError("retained lane record roster differs")
    for batch in range(12):
        coarse = generated[(batch, 64)]
        fine = generated[(batch, 128)]
        reference = oracle[batch]
        if coarse["checkpoints"][0] != reference["checkpoints"][0] or fine["checkpoints"][0] != reference["checkpoints"][0]:
            raise Stage15QualificationError("year-zero state differs from exact v4 IAS15 oracle")
        for tracer in range(TRACER_COUNT):
            rich = v4._comparison_metrics(coarse, fine, tracer)
            fine_ref = v4._comparison_metrics(fine, reference, tracer)
            coarse_ref = v4._comparison_metrics(coarse, reference, tracer)
            floors = v4._reference_ulp_floors(fine, reference, tracer)
            fraction_obs["richardson_a"].append(rich["semimajor_axis"] / 3)
            squared_obs["richardson_r"].append(rich["position_AU_squared"] / 9)
            squared_obs["richardson_v"].append(rich["velocity_AU_per_julian_year_squared"] / 9)
            fraction_obs["fine_ias_a"].append(fine_ref["semimajor_axis"])
            squared_obs["fine_ias_r"].append(fine_ref["position_AU_squared"])
            squared_obs["fine_ias_v"].append(fine_ref["velocity_AU_per_julian_year_squared"])
            fraction_obs["fine_ias_scattering"].append(abs(
                v4._scattering_value(fine, tracer) - v4._scattering_value(reference, tracer)
            ))
            contractions["tracer_a"].append(v4._exact_scalar_contraction_passes(
                fine_ref["semimajor_axis"], coarse_ref["semimajor_axis"], floors["semimajor_axis"]
            ))
            contractions["tracer_r"].append(v4._exact_squared_contraction_passes(
                fine_ref["position_AU_squared"], coarse_ref["position_AU_squared"], floors["position_AU_squared"]
            ))
            contractions["tracer_v"].append(v4._exact_squared_contraction_passes(
                fine_ref["velocity_AU_per_julian_year_squared"], coarse_ref["velocity_AU_per_julian_year_squared"], floors["velocity_AU_per_julian_year_squared"]
            ))
        for source in range(MASSIVE_COUNT):
            rich = _massive_metrics(coarse, fine, source, v4)
            fine_ref = _massive_metrics(fine, reference, source, v4)
            coarse_ref = _massive_metrics(coarse, reference, source, v4)
            floors = _massive_floors(reference, source, v4)
            squared_obs["source_richardson_r"].append(rich["position"] / 9)
            squared_obs["source_richardson_v"].append(rich["velocity"] / 9)
            squared_obs["source_fine_ias_r"].append(fine_ref["position"])
            squared_obs["source_fine_ias_v"].append(fine_ref["velocity"])
            contractions["source_r"].append(v4._exact_squared_contraction_passes(
                fine_ref["position"], coarse_ref["position"], floors["position"]
            ))
            contractions["source_v"].append(v4._exact_squared_contraction_passes(
                fine_ref["velocity"], coarse_ref["velocity"], floors["velocity"]
            ))
        coarse_summary = lane_by_key[(batch, 64)]["all_node_summary"]
        fine_summary = lane_by_key[(batch, 128)]["all_node_summary"]
        coarse_energy = v4._fraction_from_record(
            coarse_summary["maximum_relative_massive_energy_drift_exact"], "coarse energy"
        )
        fine_energy = v4._fraction_from_record(
            fine_summary["maximum_relative_massive_energy_drift_exact"], "fine energy"
        )
        fraction_obs["fine_energy"].append(fine_energy)
        contractions["energy"].append(fine_energy <= max(coarse_energy / 3, ENERGY_FLOOR))
        angular_values.append(float.fromhex(fine_summary["maximum_relative_massive_angular_momentum_drift_hex"]))
    gates = {
        "richardson_semimajor_axis": v4._exact_fraction_gate(fraction_obs["richardson_a"], RICHARDSON_A),
        "richardson_position": v4._exact_squared_gate(squared_obs["richardson_r"], POSITION_BUDGET),
        "richardson_velocity": v4._exact_squared_gate(squared_obs["richardson_v"], VELOCITY_BUDGET),
        "fine_vs_ias_semimajor_axis": v4._exact_fraction_gate(fraction_obs["fine_ias_a"], FINE_IAS_A),
        "fine_vs_ias_position": v4._exact_squared_gate(squared_obs["fine_ias_r"], POSITION_BUDGET),
        "fine_vs_ias_velocity": v4._exact_squared_gate(squared_obs["fine_ias_v"], VELOCITY_BUDGET),
        "fine_vs_ias_scattering_diagnostic": v4._exact_fraction_gate(fraction_obs["fine_ias_scattering"], FINE_IAS_A),
        "source_richardson_position": v4._exact_squared_gate(squared_obs["source_richardson_r"], POSITION_BUDGET),
        "source_richardson_velocity": v4._exact_squared_gate(squared_obs["source_richardson_v"], VELOCITY_BUDGET),
        "source_fine_vs_ias_position": v4._exact_squared_gate(squared_obs["source_fine_ias_r"], POSITION_BUDGET),
        "source_fine_vs_ias_velocity": v4._exact_squared_gate(squared_obs["source_fine_ias_v"], VELOCITY_BUDGET),
        "tracer_semimajor_axis_contraction": v4._boolean_gate(contractions["tracer_a"], rule="EXACT_FINE_LE_64_ULP_FLOOR_OR_COARSE_GE_3_FINE"),
        "tracer_position_contraction": v4._boolean_gate(contractions["tracer_r"], rule="EXACT_FINE_SQUARED_LE_64_ULP_FLOOR_SQUARED_OR_COARSE_SQUARED_GE_9_FINE_SQUARED"),
        "tracer_velocity_contraction": v4._boolean_gate(contractions["tracer_v"], rule="EXACT_FINE_SQUARED_LE_64_ULP_FLOOR_SQUARED_OR_COARSE_SQUARED_GE_9_FINE_SQUARED"),
        "source_position_contraction": v4._boolean_gate(contractions["source_r"], rule="EXACT_FINE_SQUARED_LE_64_ULP_FLOOR_SQUARED_OR_COARSE_SQUARED_GE_9_FINE_SQUARED"),
        "source_velocity_contraction": v4._boolean_gate(contractions["source_v"], rule="EXACT_FINE_SQUARED_LE_64_ULP_FLOOR_SQUARED_OR_COARSE_SQUARED_GE_9_FINE_SQUARED"),
        "fine_all_node_energy": v4._exact_fraction_gate(fraction_obs["fine_energy"], ENERGY_BUDGET),
        "fine_all_node_angular_momentum": v4._threshold_gate(angular_values, ANGULAR_BUDGET),
        "energy_contraction": v4._boolean_gate(contractions["energy"], rule="EXACT_FINE_LE_MAX_COARSE_OVER_3,2_POW_MINUS_46"),
        "no_removal_no_drop": v4._boolean_gate(
            [record.get("no_removal_no_drop") is True for record in lane_records],
            rule="ALL_24_LANES_RETAIN_EXACT_NINE_ACTIVE_TRACERS_WITH_NO_REMOVAL",
        ),
    }
    return {
        "schema": "jx.tno.stage15-resolved11-encounter-qualification.gates.v5",
        "gate_count": len(gates), "gates": gates,
        "all_predeclared_gates_passed": all(item["passed"] for item in gates.values()),
        "preliminary_scattering_used_as_classification": False,
        "classification_generated": False,
    }


def _parse_route_lane_payload(
    payload: bytes, *, lane: Mapping[str, Any], engine: types.ModuleType,
    through_year_exclusive: int | None = None,
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    routes: list[Any] = []
    removals: list[Any] = []
    if payload:
        lines = payload.splitlines(keepends=True)
    else:
        lines = []
    expected_order = 0
    expected_row_keys = {
        "schema", "record_order", "lane_order", "batch_order", "lane_id",
        "object_id", "segment_year", "kind", "record", "content_sha256",
    }
    for line in lines:
        if not line.endswith(b"\n") or len(line) > ROUTE_MAX_LINE_BYTES:
            raise Stage15QualificationError("route JSONL record framing differs")
        row = _load_canonical_json(line, "route JSONL record")
        if (
            set(row) != expected_row_keys
            or row.get("schema") != ROUTE_SCHEMA
            or row.get("content_sha256") != _domain_sha(ROUTE_DOMAIN, row)
            or _require_int(row.get("record_order"), "route record order", minimum=0) != expected_order
            or row.get("lane_order") != lane.get("lane_order")
            or row.get("batch_order") != lane.get("batch_order")
            or row.get("lane_id") != lane.get("lane_id")
            or row.get("object_id") != lane.get("object_id")
        ):
            raise Stage15QualificationError("route JSONL identity differs")
        expected_order += 1
        segment_year = _require_int(row.get("segment_year"), "route segment year", minimum=0)
        if segment_year >= HORIZON_YEARS:
            raise Stage15QualificationError("route segment year is outside horizon")
        if through_year_exclusive is not None and segment_year >= through_year_exclusive:
            continue
        record = row.get("record")
        if type(record) is not dict:
            raise Stage15QualificationError("route JSONL payload is not an object")
        try:
            if row.get("kind") == "ROUTE":
                routes.append(engine.TracerEncounterRouteRecord(**record))
            elif row.get("kind") == "REMOVAL":
                removals.append(engine.TracerRemovalEventRecord(**record))
            else:
                raise Stage15QualificationError("route JSONL kind differs")
        except Exception as exc:
            raise Stage15QualificationError("route/removal dataclass reconstruction failed") from exc
    return tuple(routes), tuple(removals)


def _repeated_year_epoch(batch: Any, year: int) -> float:
    _require_int(year, "epoch year", minimum=0)
    epoch = (float(batch.epoch_jd_tdb) - 2_451_545.0) / 365.25
    for _ in range(year):
        epoch += 1.0
    return epoch


def _source_node_tuple(payload: bytes) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    if len(payload) != SOURCE_NODE_RECORD_BYTES:
        raise Stage15QualificationError("route source-node record length differs")
    values = struct.unpack(f"<{MASSIVE_COUNT * 6}d", payload)
    if not all(math.isfinite(value) for value in values):
        raise Stage15QualificationError("route source-node record is nonfinite")
    positions = tuple(
        tuple(values[index * 3 : index * 3 + 3])
        for index in range(MASSIVE_COUNT)
    )
    start = MASSIVE_COUNT * 3
    velocities = tuple(
        tuple(values[start + index * 3 : start + index * 3 + 3])
        for index in range(MASSIVE_COUNT)
    )
    return positions, velocities


def _validate_one_route_source_witness(
    *, record: Any, kind: str, segment_year: int, steps: int, batch: Any,
    source_fd: int, source_offset: int, np: types.ModuleType,
    engine: types.ModuleType, endpoint_position: Any, endpoint_velocity: Any,
) -> None:
    global_step = _require_int(record.global_step, "route global step", minimum=0)
    tracer_index = _require_int(record.tracer_index, "route tracer index", minimum=0)
    if (
        kind not in {"ROUTE", "REMOVAL"}
        or tracer_index >= TRACER_COUNT
        or global_step >= HORIZON_YEARS * steps
        or segment_year != global_step // steps
    ):
        raise Stage15QualificationError("route lattice identity differs")
    p0, v0 = _source_node_tuple(
        _pread_all(
            source_fd, SOURCE_NODE_RECORD_BYTES,
            source_offset + global_step * SOURCE_NODE_RECORD_BYTES,
        )
    )
    p1, v1 = _source_node_tuple(
        _pread_all(
            source_fd, SOURCE_NODE_RECORD_BYTES,
            source_offset + (global_step + 1) * SOURCE_NODE_RECORD_BYTES,
        )
    )
    gm = tuple(float(item.gm_AU3_per_julian_year2) for item in batch.massive)
    if (
        record.source_segment_position_0_AU != p0
        or record.source_segment_velocity_0_AU_per_year != v0
        or record.source_segment_position_1_AU != p1
        or record.source_segment_velocity_1_AU_per_year != v1
        or record.source_segment_gm_AU3_per_year2 != gm
        or record.source_segment_step_years != 1.0 / steps
    ):
        raise Stage15QualificationError("route source segment differs from retained nodes")
    p0_array = np.asarray(p0, dtype=np.float64)
    v0_array = np.asarray(v0, dtype=np.float64)
    p1_array = np.asarray(p1, dtype=np.float64)
    v1_array = np.asarray(v1, dtype=np.float64)
    gm_array = np.asarray(gm, dtype=np.float64)
    if (
        engine._source_segment_sha256(
            p0_array, v0_array, p1_array, v1_array, gm_array, 1.0 / steps,
        )
        != record.source_segment_sha256
    ):
        raise Stage15QualificationError("route source-segment digest differs")
    source_index = _require_int(record.source_index, "route source index")
    if kind == "ROUTE":
        if (
            not 0 <= source_index < MASSIVE_COUNT
            or record.source_id != batch.massive[source_index].body_id
            or record.source_segment_epoch_0
            != _repeated_year_epoch(batch, segment_year)
            + (global_step - segment_year * steps) * (1.0 / steps)
            or record.route_relative_tolerance != 1.0e-12
            or record.route_absolute_tolerance != 1.0e-14
            or record.route_maximum_substep_fraction != 0.125
            or record.route_event_fraction_tolerance != 1.0e-12
            or record.route_maximum_primary_dynamical_phase != 0.05
        ):
            raise Stage15QualificationError("route identity or frozen configuration differs")
        acceleration_bound = math.nextafter(2.0 * math.fsum(gm), math.inf)
        state = types.SimpleNamespace(
            epoch=_repeated_year_epoch(batch, HORIZON_YEARS),
            global_step=HORIZON_YEARS * steps,
            primary_outer_boundary_AU=4096.0,
        )
        validated = types.SimpleNamespace(source_count=MASSIVE_COUNT)
        try:
            engine._validate_route_witness(
                state, validated, record,
                source_positions_host=endpoint_position,
                source_velocities_host=endpoint_velocity,
                source_gm_host=gm_array,
                encounter_radii_host=np.ones(MASSIVE_COUNT, dtype=np.float64),
                collision_radii_host=np.zeros(MASSIVE_COUNT, dtype=np.float64),
                acceleration_bounds_host=np.full(
                    MASSIVE_COUNT, acceleration_bound, dtype=np.float64,
                ),
            )
        except Exception as exc:
            raise Stage15QualificationError(
                "route deterministic guard/locator replay failed"
            ) from exc
    elif record.event_type == "SOURCE_RADIUS":
        if (
            not 0 <= source_index < MASSIVE_COUNT
            or record.source_id != batch.massive[source_index].body_id
        ):
            raise Stage15QualificationError("source-radius removal identity differs")
    elif source_index != -1 or record.source_id != "NONE":
        raise Stage15QualificationError("outer-boundary removal identity differs")


def _validate_route_payload_source_witnesses(
    payload: bytes, *, lane: Mapping[str, Any], batch: Any, source_fd: int,
    source_offset: int, np: types.ModuleType, engine: types.ModuleType,
) -> None:
    steps = _require_int(lane.get("steps_per_julian_year"), "route lane steps", minimum=1)
    endpoint_position, endpoint_velocity = _source_node_tuple(
        _pread_all(
            source_fd, SOURCE_NODE_RECORD_BYTES,
            source_offset + HORIZON_YEARS * steps * SOURCE_NODE_RECORD_BYTES,
        )
    )
    routes: list[Any] = []
    removals: list[Any] = []
    route_pairs: set[tuple[int, int]] = set()
    route_counts = [0] * TRACER_COUNT
    last_kind_by_year: dict[int, str] = {}
    for line in payload.splitlines(keepends=True):
        row = _load_canonical_json(line, "route source-witness record")
        kind = row.get("kind")
        segment_year = _require_int(
            row.get("segment_year"), "route source-witness year", minimum=0,
        )
        record_payload = row.get("record")
        if type(record_payload) is not dict:
            raise Stage15QualificationError("route source-witness payload differs")
        try:
            if kind == "ROUTE":
                if last_kind_by_year.get(segment_year) == "REMOVAL":
                    raise Stage15QualificationError("route/removal line order differs")
                record = engine.TracerEncounterRouteRecord(**record_payload)
                routes.append(record)
                pair = (record.global_step, record.tracer_index)
                if pair in route_pairs:
                    raise Stage15QualificationError("duplicate tracer route step")
                route_pairs.add(pair)
                if not 0 <= record.tracer_index < TRACER_COUNT:
                    raise Stage15QualificationError("route tracer index is outside roster")
                route_counts[record.tracer_index] += 1
                if route_counts[record.tracer_index] > 16_384:
                    raise Stage15QualificationError("per-tracer route ceiling exceeded")
                last_kind_by_year[segment_year] = "ROUTE"
            elif kind == "REMOVAL":
                record = engine.TracerRemovalEventRecord(**record_payload)
                removals.append(record)
                last_kind_by_year[segment_year] = "REMOVAL"
            else:
                raise Stage15QualificationError("route source-witness kind differs")
        except Stage15QualificationError:
            raise
        except Exception as exc:
            raise Stage15QualificationError("route source-witness reconstruction failed") from exc
        _validate_one_route_source_witness(
            record=record, kind=kind, segment_year=segment_year, steps=steps,
            batch=batch, source_fd=source_fd, source_offset=source_offset,
            np=np, engine=engine,
            endpoint_position=np.asarray(endpoint_position, dtype=np.float64),
            endpoint_velocity=np.asarray(endpoint_velocity, dtype=np.float64),
        )
    if (
        len(routes) > 100_000
        or len(removals) > TRACER_COUNT
        or routes != sorted(routes, key=lambda row: (row.global_step, row.tracer_index))
        or removals != sorted(
            removals,
            key=lambda row: (
                row.global_step, row.step_fraction, row.tracer_index,
                row.event_type, row.source_index,
            ),
        )
    ):
        raise Stage15QualificationError("route/removal canonical ledger order differs")


def _midpoint_state(
    *, request: Mapping[str, Any], lane: Mapping[str, Any], batch: Any,
    np: types.ModuleType, cp: types.ModuleType, engine: types.ModuleType,
    Provenance: type, source_fd: int, midpoint_fd: int, routes_fd: int,
) -> Any:
    steps = _require_int(lane.get("steps_per_julian_year"), "restart steps", minimum=1)
    source_locator = lane.get("source_nodes")
    midpoint = lane.get("midpoint")
    route_locator = lane.get("routes")
    if not all(type(item) is dict for item in (source_locator, midpoint, route_locator)):
        raise Stage15QualificationError("restart lane locators are incomplete")
    node_offset = (
        _require_int(source_locator["byte_offset"], "source offset", minimum=0)
        + 500 * steps * SOURCE_NODE_RECORD_BYTES
    )
    source_raw = _pread_all(source_fd, SOURCE_NODE_RECORD_BYTES, node_offset)
    source_values = np.frombuffer(source_raw, dtype="<f8").reshape(2, MASSIVE_COUNT, 3)
    midpoint_raw = _pread_all(
        midpoint_fd,
        _require_int(midpoint["byte_length"], "midpoint length", minimum=1),
        _require_int(midpoint["byte_offset"], "midpoint offset", minimum=0),
    )
    if len(midpoint_raw) != MIDPOINT_RECORD_BYTES or _sha256(midpoint_raw) != midpoint.get("record_sha256"):
        raise Stage15QualificationError("restart midpoint record differs")
    tracer_values = np.frombuffer(midpoint_raw, dtype="<f8").reshape(2, TRACER_COUNT, 3)
    route_payload = _pread_all(
        routes_fd,
        _require_int(route_locator["byte_length"], "route lane length", minimum=0),
        _require_int(route_locator["byte_offset"], "route lane offset", minimum=0),
    )
    if _sha256(route_payload) != route_locator.get("lane_payload_sha256"):
        raise Stage15QualificationError("restart route lane digest differs")
    routes, removals = _parse_route_lane_payload(
        route_payload, lane=lane, engine=engine, through_year_exclusive=500,
    )
    continuation = midpoint.get("continuation")
    if type(continuation) is not dict:
        raise Stage15QualificationError("restart continuation receipt is absent")
    if (
        len(routes) != continuation.get("prior_encounter_route_count")
        or len(removals) != continuation.get("prior_removal_event_count")
        or _sha256(_canonical_json([asdict(row) for row in routes])) != continuation.get("route_prefix_sha256")
        or _sha256(_canonical_json([asdict(row) for row in removals])) != continuation.get("removal_prefix_sha256")
    ):
        raise Stage15QualificationError("restart route prefix differs")
    base = _make_engine_state(
        np=np, cp=cp, engine=engine, Provenance=Provenance,
        batch=batch, steps=steps,
    )
    minimum = continuation.get("minimum_source_distance_AU_hex")
    ever_unbound = continuation.get("ever_unbound")
    counts = continuation.get("encounter_route_counts")
    lifecycle = continuation.get("tracer_lifecycle")
    expected_epoch = _repeated_year_epoch(batch, 500)
    if (
        type(minimum) is not list or len(minimum) != TRACER_COUNT
        or type(ever_unbound) is not list or len(ever_unbound) != TRACER_COUNT
        or any(type(x) is not bool for x in ever_unbound)
        or type(counts) is not list or len(counts) != TRACER_COUNT
        or any(type(x) is not int or x < 0 for x in counts)
        or type(lifecycle) is not list or len(lifecycle) != TRACER_COUNT
        or any(type(x) is not int or x != 0 for x in lifecycle)
        or continuation.get("epoch_hex") != expected_epoch.hex()
    ):
        raise Stage15QualificationError("restart continuation arrays differ")
    state = engine.CUDAMasslessTracerState(
        # The engine input-state digest includes snapshot_id.  A resumed state
        # therefore keeps the monolithic lane's immutable identity rather than
        # inventing a restart-only label.
        snapshot_id=base.snapshot_id,
        epoch=float.fromhex(continuation["epoch_hex"]),
        global_step=_require_int(continuation["global_step"], "midpoint global step", minimum=0),
        time_scale=base.time_scale, frame=base.frame, origin=base.origin,
        axes=base.axes, length_unit=base.length_unit, time_unit=base.time_unit,
        unit_system_id=base.unit_system_id, source_ids=base.source_ids,
        primary_source_id=base.primary_source_id,
        source_positions=cp.asarray(source_values[0]), source_velocities=cp.asarray(source_values[1]),
        source_gm=base.source_gm, source_encounter_radii=base.source_encounter_radii,
        source_collision_radii=base.source_collision_radii,
        relative_acceleration_bounds=base.relative_acceleration_bounds,
        tracer_positions=cp.asarray(tracer_values[0]), tracer_velocities=cp.asarray(tracer_values[1]),
        tracer_indices=base.tracer_indices, provenance=base.provenance,
        force_schedule_sha256=base.force_schedule_sha256,
        matched_schedule_set_sha256=base.matched_schedule_set_sha256,
        primary_outer_boundary_AU=base.primary_outer_boundary_AU,
        lattice_step_years=1.0 / steps,
        minimum_source_distance_AU=cp.asarray([float.fromhex(x) for x in minimum]),
        ever_unbound=cp.asarray(ever_unbound),
        encounter_route_counts=cp.asarray(counts, dtype=cp.int64),
        tracer_lifecycle=cp.asarray(lifecycle, dtype=cp.int32),
        prior_encounter_routes=routes, prior_removal_events=removals,
    )
    if state.global_step != 500 * steps:
        raise Stage15QualificationError("restart midpoint global step differs")
    return state


def _restart_request(
    *, lane: Mapping[str, Any], restart_order: int,
    staging_root: Path, main_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": "jx.tno.stage15-resolved11-encounter-qualification.restart-request.v5",
        "restart_order": restart_order,
        "batch_order": lane["batch_order"], "object_id": lane["object_id"],
        "lane_id": lane["lane_id"], "steps_per_julian_year": lane["steps_per_julian_year"],
        "start_year": 500, "end_year": 1_000,
        "staging_root": str(_lexical_absolute(staging_root)),
        "lane": dict(lane),
        "expected_final_state_sha256": lane["final_state_sha256"],
        "expected_final_route_ledger_sha256": lane["final_route_ledger_sha256"],
        "expected_final_removal_ledger_sha256": lane["final_removal_ledger_sha256"],
        "expected_runtime_projection": _runtime_process_independent_projection(main_runtime),
    }
    value["content_sha256"] = _domain_sha(RESTART_DOMAIN, value)
    return value


def _restart_worker(request_path: Path, output_root: Path, cache_root: Path) -> dict[str, Any]:
    request_payload = _regular_bytes(request_path, "restart request", maximum_bytes=20_000_000)
    request = _load_canonical_json(request_payload, "restart request")
    if (
        request.get("schema") != "jx.tno.stage15-resolved11-encounter-qualification.restart-request.v5"
        or request.get("content_sha256") != _domain_sha(RESTART_DOMAIN, request)
        or request.get("start_year") != 500 or request.get("end_year") != 1_000
    ):
        raise Stage15QualificationError("restart request differs")
    np, scipy, cp, engine, backend_spec, backend, Provenance, runtime = _fresh_runtime_imports(cache_root)
    if _canonical_json(_runtime_process_independent_projection(runtime)) != _canonical_json(request.get("expected_runtime_projection")):
        raise Stage15QualificationError("restart runtime differs from monolithic runtime")
    v4, _bridge, _custody, batches, _locators = _load_v4_context()
    batch_order = _require_int(request.get("batch_order"), "restart batch order", minimum=0)
    if batch_order >= len(batches):
        raise Stage15QualificationError("restart batch order is outside roster")
    batch = batches[batch_order]
    lane = request.get("lane")
    if type(lane) is not dict or lane.get("object_id") != batch.object_id:
        raise Stage15QualificationError("restart lane identity differs")
    staging = _lexical_absolute(Path(request["staging_root"]))
    with _stable_fd(staging / "SOURCE_NODES.bin", "restart source evidence", exact_bytes=SOURCE_NODES_EXACT_BYTES) as (source_fd, _), _stable_fd(
        staging / "MIDPOINTS.bin", "restart midpoint evidence", exact_bytes=MIDPOINTS_EXACT_BYTES
    ) as (midpoint_fd, _), _stable_fd(
        staging / "ROUTES.jsonl", "restart route evidence", maximum_bytes=ROUTE_MAX_BYTES
    ) as (routes_fd, _):
        state = _midpoint_state(
            request=request, lane=lane, batch=batch, np=np, cp=cp,
            engine=engine, Provenance=Provenance, source_fd=source_fd,
            midpoint_fd=midpoint_fd, routes_fd=routes_fd,
        )
    spec = _annual_spec(engine, backend_spec, lane["steps_per_julian_year"])
    suffix = bytearray(_raw_tracer_payload(cp, state))
    segment_receipts: list[dict[str, Any]] = []
    for year in range(500, 1_000):
        result, _source_payload, receipt = _advance_one_annual_segment(
            np=np, cp=cp, engine=engine, backend=backend,
            state=state, spec=spec, year=year,
        )
        if dict(result.runtime_identity) != runtime["engine_runtime_identity"]:
            raise Stage15QualificationError("restart lane runtime identity drifted")
        state = result.state
        suffix.extend(_raw_tracer_payload(cp, state))
        segment_receipts.append(receipt)
    suffix_payload = bytes(suffix)
    if len(suffix_payload) != RESTART_STREAM_BYTES:
        raise Stage15QualificationError("restart suffix exact length differs")
    if _state_sha256(cp, state) != request.get("expected_final_state_sha256"):
        raise Stage15QualificationError("restart final state differs from monolithic lane")
    route_sha = _sha256(_canonical_json([asdict(row) for row in state.prior_encounter_routes]))
    removal_sha = _sha256(_canonical_json([asdict(row) for row in state.prior_removal_events]))
    if (
        route_sha != request.get("expected_final_route_ledger_sha256")
        or removal_sha != request.get("expected_final_removal_ledger_sha256")
        or state.global_step != HORIZON_YEARS * spec.step_count
    ):
        raise Stage15QualificationError("restart ledger or final global step differs")
    runtime_after = _runtime_receipt(
        cache_root=cache_root,
        fresh_registry=runtime["fresh_process_registry"],
        numpy_module=np, scipy_module=scipy, cupy_module=cp,
        engine_module=engine, backend_spec=backend_spec,
    )
    _validate_runtime_document(
        runtime, runtime_after,
        allow_post_execution_module_expansion=True,
    )
    output = _lexical_absolute(output_root)
    if output.exists() or output.is_symlink():
        raise Stage15QualificationError("restart output root must start absent")
    output.mkdir(mode=0o700)
    _write_create_only(output / "SUFFIX.bin", suffix_payload, maximum=RESTART_STREAM_BYTES)
    receipt: dict[str, Any] = {
        "schema": RESTART_RECEIPT_SCHEMA,
        "request_file_sha256": _sha256(request_payload),
        "request_content_sha256": request["content_sha256"],
        "restart_order": request["restart_order"],
        "batch_order": batch_order, "object_id": batch.object_id,
        "lane_id": lane["lane_id"], "steps_per_julian_year": spec.step_count,
        "start_year": 500, "end_year": 1_000,
        "suffix_checkpoint_count": RESTART_YEAR_COUNT,
        "suffix_byte_length": len(suffix_payload),
        "suffix_file_sha256": _sha256(suffix_payload),
        "source_segment_receipts_500_through_999": segment_receipts,
        "source_segment_receipts_sha256": _sha256(_canonical_json(segment_receipts)),
        "final_global_step": state.global_step,
        "final_state_sha256": request["expected_final_state_sha256"],
        "final_route_ledger_sha256": route_sha,
        "final_removal_ledger_sha256": removal_sha,
        "runtime_projection": _runtime_process_independent_projection(runtime),
        "runtime_fresh_child_interpreter": runtime["fresh_child_interpreter"],
        "runtime_fresh_process_registry": runtime["fresh_process_registry"],
        "runtime_guard": runtime["guard"],
        "fresh_process": True, "midpoint_consumed": True,
    }
    receipt["content_sha256"] = _domain_sha(RESTART_DOMAIN, receipt)
    _write_create_only(output / "RECEIPT.json", _canonical_json(receipt), maximum=20_000_000)
    _fsync_directory(output)
    return receipt


def _require_restart_pair(
    monolithic: bytes, resumed: bytes, receipt: Mapping[str, Any],
    lane: Mapping[str, Any], request: Mapping[str, Any],
) -> None:
    if len(monolithic) != RESTART_STREAM_BYTES or len(resumed) != RESTART_STREAM_BYTES:
        raise Stage15QualificationError("restart stream length differs")
    if monolithic != resumed:
        raise Stage15QualificationError("fresh restart suffix is not bitwise identical")
    if (
        type(receipt) is not dict or receipt.get("schema") != RESTART_RECEIPT_SCHEMA
        or receipt.get("content_sha256") != _domain_sha(RESTART_DOMAIN, receipt)
        or receipt.get("request_content_sha256") != request.get("content_sha256")
        or receipt.get("suffix_file_sha256") != _sha256(resumed)
        or receipt.get("final_global_step") != HORIZON_YEARS * lane["steps_per_julian_year"]
        or receipt.get("final_state_sha256") != lane.get("final_state_sha256")
        or receipt.get("final_route_ledger_sha256") != lane.get("final_route_ledger_sha256")
        or receipt.get("final_removal_ledger_sha256") != lane.get("final_removal_ledger_sha256")
        or receipt.get("suffix_checkpoint_count") != RESTART_YEAR_COUNT
        or receipt.get("fresh_process") is not True
        or receipt.get("midpoint_consumed") is not True
    ):
        raise Stage15QualificationError("restart receipt differs from monolithic lane")
    main_segments = lane["source_nodes"]["annual_segment_receipts"][500:]
    if _canonical_json(receipt.get("source_segment_receipts_500_through_999")) != _canonical_json(main_segments):
        raise Stage15QualificationError("restart source-path ledger differs from monolithic lane")


def _run_restart_checks(
    *, staging_root: Path, lanes: Sequence[Mapping[str, Any]],
    monolithic_suffixes: Mapping[tuple[str, str], bytes],
    runtime: Mapping[str, Any], restart_writer: _CappedBinaryWriter,
) -> list[dict[str, Any]]:
    index = {(row["lane_id"], row["object_id"]): row for row in lanes}
    receipts: list[dict[str, Any]] = []
    work = staging_root / ".restart-work"
    if work.exists() or work.is_symlink():
        raise Stage15QualificationError("restart work directory must start absent")
    work.mkdir(mode=0o700)
    try:
        for restart_order, key in enumerate(RESTART_CASES):
            lane = index.get(key)
            monolithic = monolithic_suffixes.get(key)
            if type(lane) is not dict or type(monolithic) is not bytes:
                raise Stage15QualificationError("selected restart case is absent")
            case_root = work / f"case-{restart_order}"
            case_root.mkdir(mode=0o700)
            request = _restart_request(
                lane=lane, restart_order=restart_order,
                staging_root=staging_root, main_runtime=runtime,
            )
            request_path = case_root / "REQUEST.json"
            _write_create_only(request_path, _canonical_json(request), maximum=20_000_000)
            output_root = case_root / "output"
            cache_root = case_root / "runtime-cache"
            completed = subprocess.run(
                [
                    str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                    "restart-worker", "--restart-request", str(request_path),
                    "--restart-output-root", str(output_root),
                    "--cache-root", str(cache_root),
                ],
                cwd=PROJECT_ROOT, env=_runtime_child_environment(cache_root),
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, check=False, timeout=86_400,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[-8_000:]
                raise Stage15QualificationError(f"fresh restart worker failed: {detail}")
            if sorted(p.name for p in output_root.iterdir()) != ["RECEIPT.json", "SUFFIX.bin"]:
                raise Stage15QualificationError("restart worker artifact roster differs")
            resumed = _regular_bytes(
                output_root / "SUFFIX.bin", "fresh restart suffix",
                exact_bytes=RESTART_STREAM_BYTES,
            )
            receipt_payload = _regular_bytes(
                output_root / "RECEIPT.json", "fresh restart receipt",
                maximum_bytes=20_000_000,
            )
            receipt = _load_canonical_json(receipt_payload, "fresh restart receipt")
            _require_restart_pair(monolithic, resumed, receipt, lane, request)
            restart_offset = restart_writer.count
            restart_writer.write(monolithic)
            restart_writer.write(resumed)
            receipts.append({
                "restart_order": restart_order, "lane_id": key[0], "object_id": key[1],
                "batch_order": lane["batch_order"],
                "byte_offset": restart_offset,
                "stream_byte_length": RESTART_STREAM_BYTES,
                "monolithic_sha256": _sha256(monolithic),
                "fresh_resumed_sha256": _sha256(resumed),
                "bitwise_suffix_match": True,
                "request": request, "worker_receipt": receipt,
                "worker_receipt_file_sha256": _sha256(receipt_payload),
            })
        return receipts
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _finalize_lane_evidence(
    *, staging_root: Path, lanes: Sequence[dict[str, Any]],
    np: types.ModuleType, v4: types.ModuleType, batches: Sequence[Any],
) -> None:
    with _stable_fd(
        staging_root / "ARRAYS.bin", "Stage-1.5 Cartesian arrays",
        exact_bytes=ARRAYS_EXACT_BYTES,
    ) as (arrays_fd, _), _stable_fd(
        staging_root / "SOURCE_NODES.bin", "Stage-1.5 source nodes",
        exact_bytes=SOURCE_NODES_EXACT_BYTES,
    ) as (source_fd, _):
        for lane in lanes:
            batch_order = _require_int(lane.get("batch_order"), "lane batch order", minimum=0)
            batch = batches[batch_order]
            array = lane["cartesian_array"]
            source = lane["source_nodes"]
            array_payload = _pread_all(arrays_fd, array["byte_length"], array["byte_offset"])
            array["lane_payload_sha256"] = _sha256(array_payload)
            source["lane_payload_sha256"] = _fd_sha256(
                source_fd, source["byte_length"], source["byte_offset"]
            )
            lane["all_node_summary"] = _source_summary_from_fd(
                fd=source_fd, offset=source["byte_offset"],
                byte_length=source["byte_length"], node_count=source["node_count"],
                np=np, v4=v4,
                gm_values=[item.gm_AU3_per_julian_year2 for item in batch.massive],
            )


def _artifact_receipt(path: Path, *, maximum: int, exact: int | None = None) -> dict[str, Any]:
    with _stable_fd(path, path.name, exact_bytes=exact, maximum_bytes=maximum) as (fd, info):
        digest = _fd_sha256(fd, info.st_size)
    return {
        "relative_path": path.name, "bytes": info.st_size,
        "file_sha256": digest,
    }


def _execute_worker(staging_root: Path, cache_root: Path) -> dict[str, Any]:
    staging = _lexical_absolute(staging_root)
    if not staging.is_dir() or staging.is_symlink() or sorted(p.name for p in staging.iterdir()) != ["PROTOCOL.json"]:
        raise Stage15QualificationError("execution staging root must contain only protocol")
    protocol_payload = _regular_bytes(staging / "PROTOCOL.json", "staged protocol", maximum_bytes=PROTOCOL_MAX_BYTES)
    protocol = _load_canonical_json(protocol_payload, "staged protocol")
    validate_protocol(protocol)
    _input_bindings(full_arrays_hash=True)
    np, scipy, cp, engine, backend_spec, backend, Provenance, runtime = _fresh_runtime_imports(cache_root)
    v4, _bridge, _custody, batches, oracle_locators = _load_v4_context()
    del oracle_locators
    arrays_writer = _CappedBinaryWriter(staging / "ARRAYS.bin", ARRAYS_EXACT_BYTES)
    source_writer = _CappedBinaryWriter(staging / "SOURCE_NODES.bin", SOURCE_NODES_EXACT_BYTES)
    route_writer = _CappedBinaryWriter(staging / "ROUTES.jsonl", ROUTE_MAX_BYTES)
    midpoint_writer = _CappedBinaryWriter(staging / "MIDPOINTS.bin", MIDPOINTS_EXACT_BYTES)
    lanes: list[dict[str, Any]] = []
    gate_lanes: dict[tuple[int, int], dict[str, Any]] = {}
    monolithic: dict[tuple[str, str], bytes] = {}
    try:
        lane_order = 0
        for batch in batches:
            for lane_id, steps in LANES:
                lane, checkpoints, suffix = _run_lane(
                    lane_order=lane_order, batch=batch, lane_id=lane_id, steps=steps,
                    np=np, cp=cp, engine=engine, backend_spec=backend_spec,
                    backend=backend, Provenance=Provenance, v4=v4,
                    arrays_writer=arrays_writer, source_writer=source_writer,
                    route_writer=route_writer, midpoint_writer=midpoint_writer,
                    runtime=runtime,
                )
                lanes.append(lane)
                gate_lanes[(batch.batch_order, steps)] = {
                    "model": LANE_MODEL, "checkpoints": checkpoints,
                }
                if suffix is not None:
                    monolithic[(lane_id, batch.object_id)] = suffix
                lane_order += 1
        if (
            route_writer.record_count
            != sum(row["routes"]["record_count"] for row in lanes)
            or route_writer.record_count > ROUTE_MAX_RECORDS
        ):
            raise Stage15QualificationError("experiment-global route record count differs")
        arrays_receipt = arrays_writer.close(exact=ARRAYS_EXACT_BYTES)
        source_receipt = source_writer.close(exact=SOURCE_NODES_EXACT_BYTES)
        routes_receipt = route_writer.close()
        midpoint_receipt = midpoint_writer.close(exact=MIDPOINTS_EXACT_BYTES)
    except BaseException:
        arrays_writer.abort(); source_writer.abort(); route_writer.abort(); midpoint_writer.abort()
        raise
    _finalize_lane_evidence(
        staging_root=staging, lanes=lanes, np=np, v4=v4, batches=batches,
    )
    runtime_payload = _canonical_json(runtime)
    _write_create_only(staging / "RUNTIME.json", runtime_payload, maximum=RUNTIME_MAX_BYTES)
    restart_writer = _CappedBinaryWriter(
        staging / "RESTART_STREAMS.bin", RESTART_STREAMS_EXACT_BYTES
    )
    try:
        restart_receipts = _run_restart_checks(
            staging_root=staging, lanes=lanes,
            monolithic_suffixes=monolithic, runtime=runtime,
            restart_writer=restart_writer,
        )
        restart_file = restart_writer.close(exact=RESTART_STREAMS_EXACT_BYTES)
    except BaseException:
        restart_writer.abort()
        raise
    runtime_after = _runtime_receipt(
        cache_root=cache_root,
        fresh_registry=runtime["fresh_process_registry"],
        numpy_module=np, scipy_module=scipy, cupy_module=cp,
        engine_module=engine, backend_spec=backend_spec,
    )
    _validate_runtime_document(
        runtime, runtime_after,
        allow_post_execution_module_expansion=True,
    )
    shutil.rmtree(_lexical_absolute(cache_root))
    if _lexical_absolute(cache_root).exists() or _lexical_absolute(cache_root).is_symlink():
        raise Stage15QualificationError("execution runtime cache was not removed before publication")
    with _stable_fd(
        V4_ARRAYS_PATH, "Stage-1 v4 oracle arrays", exact_bytes=V4_ARRAYS_BYTES
    ) as (oracle_fd, _):
        if _fd_sha256(oracle_fd, V4_ARRAYS_BYTES) != V4_ARRAYS_SHA256:
            raise Stage15QualificationError("v4 oracle arrays changed before gate evaluation")
        v1 = _load_canonical_json(
            _regular_bytes(V1_PROTOCOL_PATH, "v1 protocol", exact_bytes=31_382), "v1 protocol"
        )
        oracle_locators = v1["inputs"]["stage1_v4_de441_resolved11_ias15_oracle_lanes"]
        oracle = {
            batch_order: _decode_v4_oracle_lane(
                oracle_fd, oracle_locators[batch_order],
                batch=batches[batch_order], v4=v4,
            )
            for batch_order in range(12)
        }
    gates = _evaluate_gates(gate_lanes, oracle, lanes, v4)
    artifacts = {
        "PROTOCOL.json": {
            "relative_path": "PROTOCOL.json", "bytes": len(protocol_payload),
            "file_sha256": _sha256(protocol_payload),
        },
        "ARRAYS.bin": {"relative_path": "ARRAYS.bin", **arrays_receipt},
        "SOURCE_NODES.bin": {"relative_path": "SOURCE_NODES.bin", **source_receipt},
        "ROUTES.jsonl": {"relative_path": "ROUTES.jsonl", **routes_receipt},
        "MIDPOINTS.bin": {"relative_path": "MIDPOINTS.bin", **midpoint_receipt},
        "RESTART_STREAMS.bin": {"relative_path": "RESTART_STREAMS.bin", **restart_file},
        "RUNTIME.json": {
            "relative_path": "RUNTIME.json", "bytes": len(runtime_payload),
            "file_sha256": _sha256(runtime_payload),
        },
    }
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "status": (
            "QUALIFICATION_EXECUTED_PASS"
            if gates["all_predeclared_gates_passed"]
            else "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
        ),
        "protocol_content_sha256": protocol["content_sha256"],
        "artifacts": artifacts, "lanes": lanes,
        "restart_evidence": {
            "schema": RESTART_SCHEMA, "case_count": len(restart_receipts),
            "cases": restart_receipts,
            "file_sha256": restart_file["file_sha256"],
            "byte_length": restart_file["bytes"],
        },
        "route_evidence": {
            "schema": ROUTE_SCHEMA,
            "record_count": sum(row["routes"]["record_count"] for row in lanes),
            "file_sha256": routes_receipt["file_sha256"],
            "byte_length": routes_receipt["bytes"],
        },
        "qualification_gates": gates,
        "runtime_content_sha256": runtime["content_sha256"],
        "runtime_projection_sha256": _sha256(_canonical_json(_runtime_process_independent_projection(runtime))),
        "claims": {
            **CLAIMS, "dynamics_executed": True,
            "result_serialized_before_publication": True,
            "publication_state_not_claimed_inside_immutable_result": True,
            "qualification_established": gates["all_predeclared_gates_passed"],
        },
        "outcomes_generated": True,
        "classification_generated": False,
        "science_result": False,
    }
    result["content_sha256"] = _domain_sha(RESULT_DOMAIN, result)
    _write_create_only(staging / "RESULT.json", _canonical_json(result), maximum=RESULT_MAX_BYTES)
    _fsync_directory(staging)
    return result


def _link_create_only(source: Path, destination: Path) -> None:
    source_info = source.lstat()
    if not stat.S_ISREG(source_info.st_mode) or source_info.st_nlink != 1:
        raise Stage15QualificationError("staged publication source is linked or nonregular")
    try:
        os.link(source, destination, follow_symlinks=False)
    except OSError as exc:
        raise Stage15QualificationError(f"create-only publication failed for {destination.name}") from exc
    destination_info = destination.lstat()
    if (destination_info.st_dev, destination_info.st_ino) != (source_info.st_dev, source_info.st_ino):
        raise Stage15QualificationError("published inode differs from staged inode")


def _publish_result_last(staging_root: Path, output_root: Path) -> None:
    staging = _lexical_absolute(staging_root)
    output = _lexical_absolute(output_root)
    if sorted(p.name for p in staging.iterdir()) != sorted(ARTIFACT_ROSTER):
        raise Stage15QualificationError("staging artifact roster differs")
    if not output.is_dir() or output.is_symlink() or sorted(p.name for p in output.iterdir()) != ["PROTOCOL.json"]:
        raise Stage15QualificationError("publication destination is not the prepared protocol-only root")
    if _regular_bytes(staging / "PROTOCOL.json", "staged protocol") != _regular_bytes(output / "PROTOCOL.json", "retained protocol"):
        raise Stage15QualificationError("staged and retained protocols differ")
    # Deliberately never unlink destination names after a failure: POSIX lacks
    # a compare-inode-and-unlink primitive.  A partial roster is fail-closed.
    for name in RESULT_LAST_PREFIX:
        if name == "PROTOCOL.json":
            continue
        _link_create_only(staging / name, output / name)
        # Retire only the private staged name.  The published destination is
        # never removed on failure; after this unlink its link count is one and
        # held-fd offline validation can apply the single-link custody rule.
        (staging / name).unlink()
    _fsync_directory(output)
    result_payload = _regular_bytes(
        staging / "RESULT.json", "staged result", maximum_bytes=RESULT_MAX_BYTES,
    )
    _write_create_only(
        output / "RESULT.json", result_payload, maximum=RESULT_MAX_BYTES,
    )
    _fsync_directory(output)


def execute(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = _lexical_absolute(output_root)
    protocol = validate_prepared(root)
    _input_bindings(full_arrays_hash=True)
    parent = root.parent
    staging = Path(tempfile.mkdtemp(prefix=".stage15-v5-staging-", dir=parent))
    runtime_work = Path(tempfile.mkdtemp(prefix=".stage15-v5-runtime-", dir=parent))
    cache_root = runtime_work / "runtime-cache"
    try:
        _write_create_only(
            staging / "PROTOCOL.json", _canonical_json(protocol),
            maximum=PROTOCOL_MAX_BYTES,
        )
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                "execute-worker", "--output-root", str(staging),
                "--cache-root", str(cache_root),
            ],
            cwd=PROJECT_ROOT, env=_runtime_child_environment(cache_root),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, check=False, timeout=604_800,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
            raise Stage15QualificationError(f"fresh execution worker failed: {detail}")
        # Fresh validator child must accept the complete staging root before
        # any artifact name is linked into the prepared destination.
        _validate_output_in_fresh_child(staging)
        _publish_result_last(staging, root)
        result = validate_output(root)
        return result
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(runtime_work, ignore_errors=True)


def _validate_retained_file_receipt(value: Mapping[str, Any], label: str) -> None:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise Stage15QualificationError(f"{label} file receipt shape differs")
    path_value = value.get("path")
    if type(path_value) is not str or not Path(path_value).is_absolute():
        raise Stage15QualificationError(f"{label} file path differs")
    path = _lexical_absolute(Path(path_value))
    permitted = (
        path.is_relative_to(_lexical_absolute(PROJECT_ROOT))
        or path.is_relative_to(_lexical_absolute(PROJECT_SITE_PACKAGES))
    )
    if not permitted:
        raise Stage15QualificationError(f"{label} file path is outside frozen roots")
    size = _require_int(value.get("bytes"), f"{label} bytes", minimum=0)
    digest = _require_sha(value.get("sha256"), f"{label} SHA-256")
    with _stable_fd(path, label, exact_bytes=size) as (fd, _):
        if _fd_sha256(fd, size) != digest:
            raise Stage15QualificationError(f"{label} live bytes differ")


def _validate_runtime_document(
    retained: Mapping[str, Any], current: Mapping[str, Any], *,
    allow_post_execution_module_expansion: bool = False,
) -> dict[str, Any]:
    runtime_keys = {
        "schema", "fresh_child_interpreter", "fresh_process_registry",
        "guard", "distributions", "critical_modules", "module_registry",
        "runtime_prefix_exceptions",
        "adapter_callables", "engine_runtime_identity", "implementation",
        "claim", "content_sha256",
    }
    _require_exact_mapping_keys(retained, runtime_keys, "retained runtime receipt")
    _require_exact_mapping_keys(current, runtime_keys, "current runtime receipt")
    if (
        retained.get("schema") != RUNTIME_SCHEMA
        or retained.get("content_sha256") != _domain_sha(RUNTIME_DOMAIN, retained)
        or retained.get("claim")
        != "EXACT_FRESH_PROCESS_RUNTIME_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION"
        or current.get("schema") != RUNTIME_SCHEMA
        or current.get("content_sha256") != _domain_sha(RUNTIME_DOMAIN, current)
        or current.get("claim") != retained.get("claim")
    ):
        raise Stage15QualificationError("retained runtime receipt differs")
    _validate_runtime_prefix_exception_records(
        retained.get("runtime_prefix_exceptions")
    )
    _validate_runtime_prefix_exception_records(
        current.get("runtime_prefix_exceptions")
    )
    if not _exact_equal(
        _runtime_projection_without_module_registry(retained),
        _runtime_projection_without_module_registry(current),
    ):
        raise Stage15QualificationError(
            "live validation non-registry runtime projection differs from execution runtime"
        )
    if allow_post_execution_module_expansion:
        registry_transition = _validate_post_execution_module_registry_transition(
            retained.get("module_registry"), current.get("module_registry"),
        )
    else:
        if not _exact_equal(
            retained.get("module_registry"), current.get("module_registry"),
        ):
            raise Stage15QualificationError(
                "live validation runtime module registry differs from execution runtime"
            )
        registry = retained.get("module_registry")
        registry_transition = {
            "policy": "EXACT_REGISTRY_EQUALITY",
            "before_count": len(registry) if type(registry) is list else -1,
            "after_count": len(registry) if type(registry) is list else -1,
            "added_module_names": [],
            "added_module_receipts_sha256": _sha256(_canonical_json([])),
        }
    if not _exact_equal(
        retained.get("fresh_child_interpreter"),
        current.get("fresh_child_interpreter"),
    ):
        raise Stage15QualificationError("retained fresh-child interpreter differs")
    registry = retained.get("fresh_process_registry")
    if not _exact_equal(registry, current.get("fresh_process_registry")) or (
        type(registry) is not dict
        or set(registry)
        != {"checked_prefixes", "preloaded_module_names", "fresh_registry_observed", "claim"}
        or registry.get("fresh_registry_observed") is not True
        or registry.get("preloaded_module_names") != []
        or registry.get("checked_prefixes") != list(RUNTIME_PREFIXES)
        or registry.get("claim")
        != "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION"
    ):
        raise Stage15QualificationError("retained fresh-process registry differs")
    if not _exact_equal(
        _normalized_runtime_guard(retained.get("guard")),
        _normalized_runtime_guard(current.get("guard")),
    ):
        raise Stage15QualificationError("retained runtime guard differs")
    distributions = retained.get("distributions")
    if type(distributions) is not list or len(distributions) != 3:
        raise Stage15QualificationError("retained distribution roster differs")
    for distribution in distributions:
        if type(distribution) is not dict:
            raise Stage15QualificationError("retained distribution receipt is absent")
        name = distribution.get("distribution_name")
        if name not in RUNTIME_MANIFESTS:
            raise Stage15QualificationError("retained distribution name differs")
        expected = RUNTIME_MANIFESTS[str(name)]
        if distribution.get("distribution_version") != expected["version"]:
            raise Stage15QualificationError("retained distribution version differs")
        for kind in ("record", "metadata"):
            _validate_retained_file_receipt(distribution.get(kind), f"{name} {kind}")
    for roster_name in ("critical_modules", "module_registry"):
        roster = retained.get(roster_name)
        if type(roster) is not list or not roster:
            raise Stage15QualificationError(f"retained {roster_name} roster differs")
        names: list[str] = []
        for row in roster:
            if type(row) is not dict or set(row) != {
                "module", "declared_module_name", "spec_name", "spec_origin",
                "loader_type", "path", "bytes", "sha256",
            }:
                raise Stage15QualificationError(f"retained {roster_name} row differs")
            module_name = row.get("module")
            if type(module_name) is not str:
                raise Stage15QualificationError("retained module identity differs")
            expected_declared_name = MODULE_DECLARED_NAME_ALIASES.get(
                module_name, module_name,
            )
            if row.get("declared_module_name") != expected_declared_name:
                raise Stage15QualificationError("retained declared module name differs")
            specless_binding = SPECLESS_FILE_BACKED_MODULES.get(module_name)
            if specless_binding is not None:
                if any(
                    row.get(key) is not None
                    for key in ("spec_name", "spec_origin", "loader_type")
                ):
                    raise Stage15QualificationError(
                        "retained exact spec-less module receipt differs"
                    )
                expected_path = _lexical_absolute(
                    PROJECT_SITE_PACKAGES / str(specless_binding["relative_path"])
                )
                if (
                    row.get("path") != str(expected_path)
                    or row.get("bytes") != specless_binding["bytes"]
                    or row.get("sha256") != specless_binding["sha256"]
                ):
                    raise Stage15QualificationError(
                        "retained exact spec-less module binding differs"
                    )
            else:
                if (
                    row.get("spec_name") != module_name
                    or row.get("spec_origin") != row.get("path")
                    or type(row.get("loader_type")) is not str
                ):
                    raise Stage15QualificationError(
                        "retained module loader receipt differs"
                    )
                if module_name == SCIPY_UARRAY_REGISTRY_NAME and (
                    row.get("path") != str(_lexical_absolute(
                        PROJECT_SITE_PACKAGES / SCIPY_UARRAY_RELATIVE_PATH
                    ))
                    or row.get("bytes") != SCIPY_UARRAY_BYTES
                    or row.get("sha256") != SCIPY_UARRAY_SHA256
                    or row.get("loader_type") != "ExtensionFileLoader"
                ):
                    raise Stage15QualificationError(
                        "retained SciPy uarray alias binding differs"
                    )
            _validate_retained_file_receipt(
                {key: row.get(key) for key in ("path", "bytes", "sha256")},
                f"runtime module {module_name}",
            )
            names.append(module_name)
        if len(names) != len(set(names)):
            raise Stage15QualificationError(f"retained {roster_name} has duplicate modules")
        if roster_name == "module_registry":
            declared_name_aliases = [
                str(row["module"])
                for row in roster
                if row["module"] != row["declared_module_name"]
            ]
            specless_names = [
                str(row["module"])
                for row in roster
                if row["spec_name"] is None
            ]
            if declared_name_aliases != [SCIPY_UARRAY_REGISTRY_NAME]:
                raise Stage15QualificationError(
                    "runtime declared-name alias roster differs"
                )
            if specless_names != sorted(SPECLESS_FILE_BACKED_MODULES):
                raise Stage15QualificationError(
                    "runtime exact spec-less module roster differs"
                )
    implementation = retained.get("implementation")
    _validate_retained_file_receipt(implementation, "Stage-1.5 implementation")
    return registry_transition


def _build_runtime_preflight_document(
    runtime_before: Mapping[str, Any], runtime_after: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_runtime_document(runtime_before, runtime_after)
    before_projection = _runtime_process_independent_projection(runtime_before)
    after_projection = _runtime_process_independent_projection(runtime_after)
    record: dict[str, Any] = {
        "schema": RUNTIME_PREFLIGHT_SCHEMA,
        "status": "SETUP_ONLY_FULL_RUNTIME_RECEIPT_PREFLIGHT_PASS",
        "runtime_before": dict(runtime_before),
        "runtime_after": dict(runtime_after),
        "runtime_before_projection_sha256": _sha256(_canonical_json(before_projection)),
        "runtime_after_projection_sha256": _sha256(_canonical_json(after_projection)),
        "backend_spec": {
            "backend": "cupy", "device": "cuda:0", "block_size": 128,
        },
        "backend_resolution_completed": True,
        "engine_runtime_identity_observed": True,
        "cuda_runtime_alias_invariant_checked_before_and_after": True,
        "source_step_count": 0,
        "tracer_count": 0,
        "dynamics_executed": False,
        "dynamics_outcome_generated": False,
        "retained_scientific_output_generated": False,
        "classification_generated": False,
        "science_result": False,
        "claim": (
            "SETUP_ONLY_FRESH_PROCESS_RUNTIME_POINT_OBSERVATION_"
            "NOT_QUALIFICATION_NOT_SCIENTIFIC_OUTPUT"
        ),
    }
    record["content_sha256"] = _domain_sha(RUNTIME_PREFLIGHT_DOMAIN, record)
    return record


def _validate_runtime_preflight_document(value: Mapping[str, Any]) -> None:
    keys = {
        "schema", "status", "runtime_before", "runtime_after",
        "runtime_before_projection_sha256", "runtime_after_projection_sha256",
        "backend_spec", "backend_resolution_completed",
        "engine_runtime_identity_observed",
        "cuda_runtime_alias_invariant_checked_before_and_after",
        "source_step_count", "tracer_count", "dynamics_executed",
        "dynamics_outcome_generated", "retained_scientific_output_generated",
        "classification_generated", "science_result", "claim", "content_sha256",
    }
    _require_exact_mapping_keys(value, keys, "runtime preflight")
    before = value.get("runtime_before")
    after = value.get("runtime_after")
    if type(before) is not dict or type(after) is not dict:
        raise Stage15QualificationError("runtime preflight receipts are absent")
    _validate_runtime_document(before, after)
    expected_backend = {"backend": "cupy", "device": "cuda:0", "block_size": 128}
    if (
        value.get("schema") != RUNTIME_PREFLIGHT_SCHEMA
        or value.get("status") != "SETUP_ONLY_FULL_RUNTIME_RECEIPT_PREFLIGHT_PASS"
        or value.get("content_sha256") != _domain_sha(RUNTIME_PREFLIGHT_DOMAIN, value)
        or value.get("runtime_before_projection_sha256")
        != _sha256(_canonical_json(_runtime_process_independent_projection(before)))
        or value.get("runtime_after_projection_sha256")
        != _sha256(_canonical_json(_runtime_process_independent_projection(after)))
        or not _exact_equal(value.get("backend_spec"), expected_backend)
        or value.get("backend_resolution_completed") is not True
        or value.get("engine_runtime_identity_observed") is not True
        or value.get("cuda_runtime_alias_invariant_checked_before_and_after") is not True
        or value.get("source_step_count") != 0
        or type(value.get("source_step_count")) is not int
        or value.get("tracer_count") != 0
        or type(value.get("tracer_count")) is not int
        or value.get("dynamics_executed") is not False
        or value.get("dynamics_outcome_generated") is not False
        or value.get("retained_scientific_output_generated") is not False
        or value.get("classification_generated") is not False
        or value.get("science_result") is not False
        or value.get("claim")
        != "SETUP_ONLY_FRESH_PROCESS_RUNTIME_POINT_OBSERVATION_NOT_QUALIFICATION_NOT_SCIENTIFIC_OUTPUT"
    ):
        raise Stage15QualificationError("runtime preflight claim or binding differs")


def _runtime_preflight_worker(cache_root: Path, receipt_path: Path) -> dict[str, Any]:
    (
        np, scipy, cp, engine, backend_spec, backend, _Provenance, runtime_before,
    ) = _fresh_runtime_imports(cache_root)
    if backend is None:
        raise Stage15QualificationError("runtime preflight backend resolution failed")
    runtime_after = _runtime_receipt(
        cache_root=cache_root,
        fresh_registry=runtime_before["fresh_process_registry"],
        numpy_module=np, scipy_module=scipy, cupy_module=cp,
        engine_module=engine, backend_spec=backend_spec,
    )
    document = _build_runtime_preflight_document(runtime_before, runtime_after)
    _write_create_only(
        _lexical_absolute(receipt_path), _canonical_json(document),
        maximum=RUNTIME_MAX_BYTES,
    )
    return document


def runtime_preflight() -> dict[str, Any]:
    """Run setup-only backend/runtime custody in a fresh exact-venv child."""

    with tempfile.TemporaryDirectory(prefix="jx-stage15-v5-runtime-preflight-") as raw:
        work = Path(raw)
        cache_root = work / "runtime-cache"
        receipt_path = work / "RUNTIME_PREFLIGHT.json"
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                "runtime-preflight-worker", "--cache-root", str(cache_root),
                "--runtime-preflight-receipt", str(receipt_path),
            ],
            cwd=PROJECT_ROOT, env=_runtime_child_environment(cache_root),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, check=False, timeout=300,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
            raise Stage15QualificationError(
                f"fresh runtime preflight worker failed: {detail}"
            )
        payload = _regular_bytes(
            receipt_path, "runtime preflight receipt", maximum_bytes=RUNTIME_MAX_BYTES,
        )
        document = _load_canonical_json(payload, "runtime preflight receipt")
        _validate_runtime_preflight_document(document)
        return document


def _build_continuation_smoke_document(
    *, runtime_before: Mapping[str, Any], runtime_after: Mapping[str, Any],
    batch: Any, result: Any, source_payload: bytes,
    segment_receipt: Mapping[str, Any], stored_minimum: Sequence[float],
    canonical_endpoint_minimum: Sequence[float], lifecycle: Sequence[int],
    final_state_sha256: str,
) -> dict[str, Any]:
    """Build the explicitly nonqualifying one-segment continuation receipt."""

    _validate_runtime_document(
        runtime_before, runtime_after,
        allow_post_execution_module_expansion=True,
    )
    if (
        batch.batch_order != 0
        or batch.object_id != OBJECT_IDS[0]
        or batch.epoch_jd_tdb.hex() != SMOKE_NATIVE_EPOCH_JD_TDB_HEX
        or result.state.global_step != 64
        or len(source_payload) != 65 * SOURCE_NODE_RECORD_BYTES
        or type(segment_receipt) is not dict
        or segment_receipt.get("year_start") != 0
        or segment_receipt.get("year_end") != 1
        or segment_receipt.get("ending_global_step") != 64
        or segment_receipt.get("bitwise_source_endpoint_match") is not True
        or len(stored_minimum) != TRACER_COUNT
        or len(canonical_endpoint_minimum) != TRACER_COUNT
        or len(lifecycle) != TRACER_COUNT
    ):
        raise Stage15QualificationError("continuation smoke inputs differ")
    stored_hex = [float(value).hex() for value in stored_minimum]
    endpoint_hex = [float(value).hex() for value in canonical_endpoint_minimum]
    lifecycle_values = [
        _require_int(value, "continuation smoke lifecycle", minimum=0)
        for value in lifecycle
    ]
    active_slots = [
        index for index, value in enumerate(lifecycle_values) if value == 0
    ]
    if not active_slots or any(
        not math.isfinite(float(stored_minimum[index]))
        or not math.isfinite(float(canonical_endpoint_minimum[index]))
        or float(stored_minimum[index])
        > float(canonical_endpoint_minimum[index])
        for index in active_slots
    ):
        raise Stage15QualificationError(
            "continuation smoke active endpoint minimum is not canonical"
        )
    route_rows = [asdict(row) for row in result.encounter_routes]
    removal_rows = [asdict(row) for row in result.removal_events]
    record: dict[str, Any] = {
        "schema": CONTINUATION_SMOKE_SCHEMA,
        "status": "SETUP_SMOKE_ONE_REAL_SEGMENT_AND_NEXT_CONTINUATION_PASS",
        "input": {
            "ephemeris": "DE441", "batch_order": 0,
            "object_id": OBJECT_IDS[0],
            "native_epoch_jd_tdb_hex": SMOKE_NATIVE_EPOCH_JD_TDB_HEX,
            "lane_id": "jx_resolved11_encounter_h64",
            "steps_per_julian_year": 64,
            "initial_global_step": 0, "final_global_step": 64,
            "source_count": MASSIVE_COUNT, "tracer_count": TRACER_COUNT,
        },
        "adapter": {
            "engine_source_sha256": ENCOUNTER_SOURCE_SHA256,
            "engine_contract_test_sha256": ENCOUNTER_TEST_SHA256,
            "active_endpoint_minimum_policy": ACTIVE_ENDPOINT_MINIMUM_POLICY,
            "public_segment_entrypoint": "advance_cuda_massless_tracers",
            "next_continuation_validator": "_validate_continuation",
        },
        "segment": {
            "smoke_segment_count": 1,
            "source_node_count": 65,
            "source_payload_bytes": len(source_payload),
            "source_payload_sha256": _sha256(source_payload),
            "segment_receipt": dict(segment_receipt),
            "final_state_sha256": _require_sha(
                final_state_sha256, "continuation smoke final state"
            ),
            "route_count": len(route_rows),
            "removal_count": len(removal_rows),
            "routes": route_rows,
            "removals": removal_rows,
            "route_ledger_sha256": _sha256(_canonical_json(route_rows)),
            "removal_ledger_sha256": _sha256(_canonical_json(removal_rows)),
            "result_runtime_identity_sha256": _sha256(
                _canonical_json(dict(result.runtime_identity))
            ),
        },
        "endpoint_diagnostics": {
            "stored_minimum_source_distance_AU_hex": stored_hex,
            "canonical_active_endpoint_minimum_AU_hex": endpoint_hex,
            "tracer_lifecycle": lifecycle_values,
            "active_tracer_count": len(active_slots),
            "stored_minimum_includes_every_active_endpoint": True,
            "comparison_tolerance_used": False,
        },
        "runtime_before": dict(runtime_before),
        "runtime_after": dict(runtime_after),
        "claims": {
            "fresh_exact_venv_worker": True,
            "partial_dynamics_executed": True,
            "smoke_segment_executed": True,
            "next_continuation_validation_completed": True,
            "second_segment_executed": False,
            "setup_numerical_outcome_generated": True,
            "setup_numerical_outcome_disclosed": True,
            "qualification_established": False,
            "qualification_outcome_generated": False,
            "retained_scientific_output_generated": False,
            "default_output_root_mutated": False,
            "dynamical_classification_generated": False,
            "resonance_classification_generated": False,
            "planet_nine_evidence": False,
            "science_result": False,
        },
        "claim": (
            "BOUNDED_REAL_INPUT_SETUP_SMOKE_NOT_QUALIFICATION_NOT_A_"
            "CLASSIFICATION_OR_SCIENTIFIC_RESULT"
        ),
    }
    record["content_sha256"] = _domain_sha(CONTINUATION_SMOKE_DOMAIN, record)
    return record


def _validate_continuation_smoke_document(value: Mapping[str, Any]) -> None:
    keys = {
        "schema", "status", "input", "adapter", "segment",
        "endpoint_diagnostics", "runtime_before", "runtime_after", "claims",
        "claim", "content_sha256",
    }
    _require_exact_mapping_keys(value, keys, "continuation smoke")
    runtime_before = value.get("runtime_before")
    runtime_after = value.get("runtime_after")
    if type(runtime_before) is not dict or type(runtime_after) is not dict:
        raise Stage15QualificationError("continuation smoke runtime is absent")
    _validate_runtime_document(
        runtime_before, runtime_after,
        allow_post_execution_module_expansion=True,
    )
    expected_input = {
        "ephemeris": "DE441", "batch_order": 0,
        "object_id": OBJECT_IDS[0],
        "native_epoch_jd_tdb_hex": SMOKE_NATIVE_EPOCH_JD_TDB_HEX,
        "lane_id": "jx_resolved11_encounter_h64",
        "steps_per_julian_year": 64,
        "initial_global_step": 0, "final_global_step": 64,
        "source_count": MASSIVE_COUNT, "tracer_count": TRACER_COUNT,
    }
    input_value = value.get("input")
    if (
        type(input_value) is not dict
        or set(input_value) != set(expected_input)
        or not _exact_equal(input_value, expected_input)
    ):
        raise Stage15QualificationError("continuation smoke input differs")
    try:
        native_epoch = float.fromhex(input_value["native_epoch_jd_tdb_hex"])
    except (TypeError, ValueError) as exc:
        raise Stage15QualificationError("continuation smoke epoch differs") from exc
    if not math.isfinite(native_epoch):
        raise Stage15QualificationError("continuation smoke epoch is nonfinite")
    adapter = value.get("adapter")
    expected_adapter = {
        "engine_source_sha256": ENCOUNTER_SOURCE_SHA256,
        "engine_contract_test_sha256": ENCOUNTER_TEST_SHA256,
        "active_endpoint_minimum_policy": ACTIVE_ENDPOINT_MINIMUM_POLICY,
        "public_segment_entrypoint": "advance_cuda_massless_tracers",
        "next_continuation_validator": "_validate_continuation",
    }
    if not _exact_equal(adapter, expected_adapter):
        raise Stage15QualificationError("continuation smoke adapter differs")
    segment = value.get("segment")
    if type(segment) is not dict or set(segment) != {
        "smoke_segment_count", "source_node_count", "source_payload_bytes",
        "source_payload_sha256", "segment_receipt", "final_state_sha256",
        "route_count", "removal_count", "routes", "removals",
        "route_ledger_sha256", "removal_ledger_sha256",
        "result_runtime_identity_sha256",
    }:
        raise Stage15QualificationError("continuation smoke segment differs")
    receipt = segment.get("segment_receipt")
    receipt_keys = {
        "year_start", "year_end", "starting_global_step",
        "ending_global_step", "private_path_node_count",
        "retained_unique_node_count", "private_path_payload_sha256",
        "retained_payload_sha256", "precomputed_endpoint_position_sha256",
        "public_endpoint_position_sha256",
        "precomputed_endpoint_velocity_sha256",
        "public_endpoint_velocity_sha256", "public_input_state_sha256",
        "bitwise_source_endpoint_match",
    }
    if (
        segment.get("smoke_segment_count") != 1
        or type(segment.get("smoke_segment_count")) is not int
        or segment.get("source_node_count") != 65
        or type(segment.get("source_node_count")) is not int
        or segment.get("source_payload_bytes") != 65 * SOURCE_NODE_RECORD_BYTES
        or type(segment.get("source_payload_bytes")) is not int
        or type(receipt) is not dict
        or set(receipt) != receipt_keys
        or receipt.get("year_start") != 0
        or type(receipt.get("year_start")) is not int
        or receipt.get("year_end") != 1
        or type(receipt.get("year_end")) is not int
        or receipt.get("ending_global_step") != 64
        or type(receipt.get("ending_global_step")) is not int
        or receipt.get("starting_global_step") != 0
        or type(receipt.get("starting_global_step")) is not int
        or receipt.get("private_path_node_count") != 65
        or type(receipt.get("private_path_node_count")) is not int
        or receipt.get("retained_unique_node_count") != 65
        or type(receipt.get("retained_unique_node_count")) is not int
        or receipt.get("bitwise_source_endpoint_match") is not True
    ):
        raise Stage15QualificationError("continuation smoke segment binding differs")
    for key in receipt_keys - {
        "year_start", "year_end", "starting_global_step", "ending_global_step",
        "private_path_node_count", "retained_unique_node_count",
        "bitwise_source_endpoint_match",
    }:
        _require_sha(receipt.get(key), f"continuation smoke receipt {key}")
    if (
        receipt["precomputed_endpoint_position_sha256"]
        != receipt["public_endpoint_position_sha256"]
        or receipt["precomputed_endpoint_velocity_sha256"]
        != receipt["public_endpoint_velocity_sha256"]
        or receipt["retained_payload_sha256"]
        != segment.get("source_payload_sha256")
    ):
        raise Stage15QualificationError(
            "continuation smoke endpoint or source-payload hash differs"
        )
    for key in (
        "source_payload_sha256", "final_state_sha256", "route_ledger_sha256",
        "removal_ledger_sha256", "result_runtime_identity_sha256",
    ):
        _require_sha(segment.get(key), f"continuation smoke {key}")
    routes = segment.get("routes")
    removals = segment.get("removals")
    runtime_identity = runtime_before.get("engine_runtime_identity")
    if (
        type(routes) is not list
        or type(removals) is not list
        or segment.get("route_count") != len(routes)
        or type(segment.get("route_count")) is not int
        or segment.get("removal_count") != len(removals)
        or type(segment.get("removal_count")) is not int
        or segment.get("route_ledger_sha256")
        != _sha256(_canonical_json(routes))
        or segment.get("removal_ledger_sha256")
        != _sha256(_canonical_json(removals))
        or type(runtime_identity) is not dict
        or segment.get("result_runtime_identity_sha256")
        != _sha256(_canonical_json(runtime_identity))
    ):
        raise Stage15QualificationError(
            "continuation smoke route, removal, or runtime evidence differs"
        )
    diagnostics = value.get("endpoint_diagnostics")
    if type(diagnostics) is not dict or set(diagnostics) != {
        "stored_minimum_source_distance_AU_hex",
        "canonical_active_endpoint_minimum_AU_hex", "tracer_lifecycle",
        "active_tracer_count", "stored_minimum_includes_every_active_endpoint",
        "comparison_tolerance_used",
    }:
        raise Stage15QualificationError("continuation smoke diagnostics differ")
    stored_hex = diagnostics.get("stored_minimum_source_distance_AU_hex")
    endpoint_hex = diagnostics.get("canonical_active_endpoint_minimum_AU_hex")
    lifecycle = diagnostics.get("tracer_lifecycle")
    if (
        type(stored_hex) is not list or len(stored_hex) != TRACER_COUNT
        or type(endpoint_hex) is not list or len(endpoint_hex) != TRACER_COUNT
        or type(lifecycle) is not list or len(lifecycle) != TRACER_COUNT
        or any(type(item) is not int for item in lifecycle)
        or any(item not in (0, 1, 2) for item in lifecycle)
    ):
        raise Stage15QualificationError("continuation smoke diagnostic roster differs")
    try:
        stored = [float.fromhex(item) for item in stored_hex]
        endpoint = [float.fromhex(item) for item in endpoint_hex]
    except (TypeError, ValueError) as exc:
        raise Stage15QualificationError("continuation smoke diagnostic hex differs") from exc
    active = [index for index, item in enumerate(lifecycle) if item == 0]
    if (
        diagnostics.get("active_tracer_count") != len(active)
        or type(diagnostics.get("active_tracer_count")) is not int
        or not active
        or diagnostics.get("stored_minimum_includes_every_active_endpoint") is not True
        or diagnostics.get("comparison_tolerance_used") is not False
        or any(
            not math.isfinite(stored[index])
            or not math.isfinite(endpoint[index])
            or stored[index] > endpoint[index]
            for index in active
        )
    ):
        raise Stage15QualificationError("continuation smoke endpoint binding differs")
    claims = value.get("claims")
    expected_claims = {
        "fresh_exact_venv_worker": True,
        "partial_dynamics_executed": True,
        "smoke_segment_executed": True,
        "next_continuation_validation_completed": True,
        "second_segment_executed": False,
        "setup_numerical_outcome_generated": True,
        "setup_numerical_outcome_disclosed": True,
        "qualification_established": False,
        "qualification_outcome_generated": False,
        "retained_scientific_output_generated": False,
        "default_output_root_mutated": False,
        "dynamical_classification_generated": False,
        "resonance_classification_generated": False,
        "planet_nine_evidence": False,
        "science_result": False,
    }
    if (
        value.get("schema") != CONTINUATION_SMOKE_SCHEMA
        or value.get("status")
        != "SETUP_SMOKE_ONE_REAL_SEGMENT_AND_NEXT_CONTINUATION_PASS"
        or value.get("content_sha256")
        != _domain_sha(CONTINUATION_SMOKE_DOMAIN, value)
        or not _exact_equal(claims, expected_claims)
        or value.get("claim")
        != "BOUNDED_REAL_INPUT_SETUP_SMOKE_NOT_QUALIFICATION_NOT_A_CLASSIFICATION_OR_SCIENTIFIC_RESULT"
    ):
        raise Stage15QualificationError("continuation smoke claim differs")


def _continuation_smoke_worker(
    cache_root: Path, receipt_path: Path,
) -> dict[str, Any]:
    _input_bindings(full_arrays_hash=False)
    (
        np, scipy, cp, engine, backend_spec, backend, Provenance, runtime_before,
    ) = _fresh_runtime_imports(cache_root)
    engine_rows = [
        row for row in runtime_before.get("critical_modules", [])
        if type(row) is dict and row.get("module")
        == "jxplanetx.cuda_massless_tracer"
    ]
    if (
        getattr(engine, "ACTIVE_ENDPOINT_MINIMUM_POLICY", None)
        != ACTIVE_ENDPOINT_MINIMUM_POLICY
        or len(engine_rows) != 1
        or engine_rows[0].get("path")
        != str(_lexical_absolute(ENCOUNTER_SOURCE_PATH))
        or engine_rows[0].get("bytes") != ENCOUNTER_SOURCE_BYTES
        or engine_rows[0].get("sha256") != ENCOUNTER_SOURCE_SHA256
        or runtime_before.get("engine_runtime_identity", {}).get(
            "implementation_source_sha256"
        ) != ENCOUNTER_SOURCE_SHA256
        or runtime_before.get("engine_runtime_identity", {}).get(
            "active_endpoint_minimum_policy"
        ) != ACTIVE_ENDPOINT_MINIMUM_POLICY
    ):
        raise Stage15QualificationError(
            "continuation smoke endpoint-minimum engine policy differs"
        )
    _v4, _bridge, _custody, batches, _locators = _load_v4_context()
    batch = batches[0]
    state = _make_engine_state(
        np=np, cp=cp, engine=engine, Provenance=Provenance,
        batch=batch, steps=64,
    )
    spec = _annual_spec(engine, backend_spec, 64)
    result, source_payload, segment_receipt = _advance_one_annual_segment(
        np=np, cp=cp, engine=engine, backend=backend,
        state=state, spec=spec, year=0,
    )
    if not _exact_equal(
        dict(result.runtime_identity),
        runtime_before.get("engine_runtime_identity"),
    ):
        raise Stage15QualificationError(
            "continuation smoke result runtime identity differs"
        )
    with backend.activate():
        validated = engine._validate_state(backend, result.state)
        continuation = engine._validate_continuation(
            backend, result.state, validated
        )
    if continuation is None:
        raise Stage15QualificationError(
            "continuation smoke next-continuation validation returned no value"
        )
    source_host = np.ascontiguousarray(
        cp.asnumpy(result.state.source_positions), dtype=np.float64
    )
    tracer_host = np.ascontiguousarray(
        cp.asnumpy(result.state.tracer_positions), dtype=np.float64
    )
    lifecycle_host = np.ascontiguousarray(
        cp.asnumpy(result.state.tracer_lifecycle), dtype=np.int32
    )
    stored_host = np.ascontiguousarray(
        cp.asnumpy(result.minimum_source_distance_AU), dtype=np.float64
    )
    canonical_endpoint = engine._host_binary64_active_endpoint_minimum(
        source_host, tracer_host, lifecycle_host
    )
    runtime_after = _runtime_receipt(
        cache_root=cache_root,
        fresh_registry=runtime_before["fresh_process_registry"],
        numpy_module=np, scipy_module=scipy, cupy_module=cp,
        engine_module=engine, backend_spec=backend_spec,
    )
    document = _build_continuation_smoke_document(
        runtime_before=runtime_before, runtime_after=runtime_after,
        batch=batch, result=result, source_payload=source_payload,
        segment_receipt=segment_receipt,
        stored_minimum=[float(value) for value in stored_host],
        canonical_endpoint_minimum=[float(value) for value in canonical_endpoint],
        lifecycle=[int(value) for value in lifecycle_host],
        final_state_sha256=_state_sha256(cp, result.state),
    )
    _validate_continuation_smoke_document(document)
    _write_create_only(
        _lexical_absolute(receipt_path), _canonical_json(document),
        maximum=RUNTIME_MAX_BYTES,
    )
    return document


def continuation_smoke() -> dict[str, Any]:
    """Run one real DE441 annual segment and validate its next continuation."""

    with tempfile.TemporaryDirectory(
        prefix="jx-stage15-v5-continuation-smoke-"
    ) as raw:
        work = Path(raw)
        cache_root = work / "runtime-cache"
        receipt_path = work / "CONTINUATION_SMOKE.json"
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                "continuation-smoke-worker", "--cache-root", str(cache_root),
                "--continuation-smoke-receipt", str(receipt_path),
            ],
            cwd=PROJECT_ROOT, env=_runtime_child_environment(cache_root),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, check=False, timeout=3_600,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
            raise Stage15QualificationError(
                f"fresh continuation smoke worker failed: {detail}"
            )
        payload = _regular_bytes(
            receipt_path, "continuation smoke receipt",
            maximum_bytes=RUNTIME_MAX_BYTES,
        )
        document = _load_canonical_json(payload, "continuation smoke receipt")
        _validate_continuation_smoke_document(document)
        return document


def _pack_checkpoint_document(checkpoint: Mapping[str, Any]) -> bytes:
    values: list[float] = []
    for key in (
        "massive_positions_AU", "massive_velocities_AU_per_julian_year",
        "tracer_positions_AU", "tracer_velocities_AU_per_julian_year",
    ):
        rows = checkpoint.get(key)
        if type(rows) is not list:
            raise Stage15QualificationError("checkpoint Cartesian rows are absent")
        values.extend(float(component) for row in rows for component in row)
    if len(values) != ARRAY_RECORD_DOUBLES or not all(math.isfinite(item) for item in values):
        raise Stage15QualificationError("checkpoint Cartesian payload differs")
    return struct.pack(f"<{ARRAY_RECORD_DOUBLES}d", *values)


def _source_states_from_record(
    payload: bytes, *, batch: Any, v4: types.ModuleType,
) -> tuple[Any, ...]:
    if len(payload) != SOURCE_NODE_RECORD_BYTES:
        raise Stage15QualificationError("source-node record length differs")
    values = struct.unpack(f"<{MASSIVE_COUNT * 6}d", payload)
    if not all(math.isfinite(item) for item in values):
        raise Stage15QualificationError("source-node record is nonfinite")
    positions = values[: MASSIVE_COUNT * 3]
    velocities = values[MASSIVE_COUNT * 3 :]
    return tuple(
        v4.MassiveState(
            item.naif_id, item.body_id, item.gm_AU3_per_julian_year2,
            tuple(positions[index * 3 : index * 3 + 3]),
            tuple(velocities[index * 3 : index * 3 + 3]),
        )
        for index, item in enumerate(batch.massive)
    )


def _checkpoint_payload_from_raw(
    *, source_payload: bytes, tracer_payload: bytes, year: int,
    batch: Any, v4: types.ModuleType,
) -> bytes:
    if len(tracer_payload) != MIDPOINT_RECORD_BYTES:
        raise Stage15QualificationError("raw tracer record length differs")
    values = struct.unpack(f"<{TRACER_COUNT * 6}d", tracer_payload)
    if not all(math.isfinite(item) for item in values):
        raise Stage15QualificationError("raw tracer record is nonfinite")
    positions = values[: TRACER_COUNT * 3]
    velocities = values[TRACER_COUNT * 3 :]
    tracers = tuple(
        v4.TracerState(
            item.object_id, item.spk_id, item.sample_kind, item.clone_index,
            tuple(positions[index * 3 : index * 3 + 3]),
            tuple(velocities[index * 3 : index * 3 + 3]),
        )
        for index, item in enumerate(batch.tracers)
    )
    checkpoint = v4._checkpoint(
        year, _source_states_from_record(source_payload, batch=batch, v4=v4), tracers,
    )
    return _pack_checkpoint_document(checkpoint)


def _centered_source_payload(
    source_payload: bytes, *, batch: Any, v4: types.ModuleType,
) -> bytes:
    massive = _source_states_from_record(source_payload, batch=batch, v4=v4)
    center_position, center_velocity = v4._weighted_centroid(massive)
    values: list[float] = []
    values.extend(
        component
        for item in massive
        for component in v4._sub(item.position_AU, center_position)
    )
    values.extend(
        component
        for item in massive
        for component in v4._sub(item.velocity_AU_per_julian_year, center_velocity)
    )
    return struct.pack(f"<{MASSIVE_COUNT * 6}d", *values)


def _validate_segment_receipts(
    *, source_fd: int, lane: Mapping[str, Any], source: Mapping[str, Any],
) -> None:
    steps = _require_int(lane.get("steps_per_julian_year"), "lane steps", minimum=1)
    receipts = source.get("annual_segment_receipts")
    if type(receipts) is not list or len(receipts) != HORIZON_YEARS:
        raise Stage15QualificationError("annual source-segment receipt roster differs")
    base = _require_int(source.get("byte_offset"), "source lane offset", minimum=0)
    expected_keys = {
        "year_start", "year_end", "starting_global_step", "ending_global_step",
        "private_path_node_count", "retained_unique_node_count",
        "private_path_payload_sha256", "retained_payload_sha256",
        "precomputed_endpoint_position_sha256", "public_endpoint_position_sha256",
        "precomputed_endpoint_velocity_sha256", "public_endpoint_velocity_sha256",
        "public_input_state_sha256", "bitwise_source_endpoint_match",
    }
    position_bytes = MASSIVE_COUNT * 3 * 8
    for year, row in enumerate(receipts):
        if type(row) is not dict or set(row) != expected_keys:
            raise Stage15QualificationError("annual source-segment receipt shape differs")
        expected_scalars = {
            "year_start": year, "year_end": year + 1,
            "starting_global_step": year * steps,
            "ending_global_step": (year + 1) * steps,
            "private_path_node_count": steps + 1,
            "retained_unique_node_count": steps + (1 if year == 0 else 0),
        }
        for key, expected in expected_scalars.items():
            if _require_int(row.get(key), key, minimum=0) != expected:
                raise Stage15QualificationError("annual source-segment arithmetic differs")
        full_offset = base + year * steps * SOURCE_NODE_RECORD_BYTES
        full_length = (steps + 1) * SOURCE_NODE_RECORD_BYTES
        retained_offset = full_offset + (0 if year == 0 else SOURCE_NODE_RECORD_BYTES)
        retained_length = (steps + (1 if year == 0 else 0)) * SOURCE_NODE_RECORD_BYTES
        if _fd_sha256(source_fd, full_length, full_offset) != row.get("private_path_payload_sha256"):
            raise Stage15QualificationError("private annual source path digest differs")
        if _fd_sha256(source_fd, retained_length, retained_offset) != row.get("retained_payload_sha256"):
            raise Stage15QualificationError("retained annual source path digest differs")
        endpoint = _pread_all(
            source_fd, SOURCE_NODE_RECORD_BYTES,
            base + (year + 1) * steps * SOURCE_NODE_RECORD_BYTES,
        )
        position_sha = _sha256(endpoint[:position_bytes])
        velocity_sha = _sha256(endpoint[position_bytes:])
        if (
            row.get("precomputed_endpoint_position_sha256") != position_sha
            or row.get("public_endpoint_position_sha256") != position_sha
            or row.get("precomputed_endpoint_velocity_sha256") != velocity_sha
            or row.get("public_endpoint_velocity_sha256") != velocity_sha
            or row.get("bitwise_source_endpoint_match") is not True
        ):
            raise Stage15QualificationError("private/public source endpoint binding differs")
        _require_sha(row.get("public_input_state_sha256"), "public input-state SHA-256")


def _validate_continuation(
    continuation: Mapping[str, Any], *, steps: int,
    prefix_routes: Sequence[Any], prefix_removals: Sequence[Any], batch: Any,
) -> None:
    expected_keys = {
        "global_step", "epoch_hex", "minimum_source_distance_AU_hex",
        "ever_unbound", "encounter_route_counts", "tracer_lifecycle",
        "prior_encounter_route_count", "prior_removal_event_count",
        "route_prefix_sha256", "removal_prefix_sha256",
    }
    if type(continuation) is not dict or set(continuation) != expected_keys:
        raise Stage15QualificationError("midpoint continuation is absent")
    if (
        _require_int(continuation.get("global_step"), "midpoint global step", minimum=0) != 500 * steps
        or continuation.get("prior_encounter_route_count") != len(prefix_routes)
        or continuation.get("prior_removal_event_count") != len(prefix_removals)
        or continuation.get("route_prefix_sha256")
        != _sha256(_canonical_json([asdict(row) for row in prefix_routes]))
        or continuation.get("removal_prefix_sha256")
        != _sha256(_canonical_json([asdict(row) for row in prefix_removals]))
    ):
        raise Stage15QualificationError("midpoint continuation ledger differs")
    try:
        epoch = float.fromhex(continuation["epoch_hex"])
        minimum = [float.fromhex(value) for value in continuation["minimum_source_distance_AU_hex"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise Stage15QualificationError("midpoint continuation float encoding differs") from exc
    ever = continuation.get("ever_unbound")
    counts = continuation.get("encounter_route_counts")
    lifecycle = continuation.get("tracer_lifecycle")
    if (
        not math.isfinite(epoch)
        or continuation.get("epoch_hex") != _repeated_year_epoch(batch, 500).hex()
        or type(minimum) is not list or len(minimum) != TRACER_COUNT
        or not all(math.isfinite(value) and value >= 0.0 for value in minimum)
        or type(ever) is not list or len(ever) != TRACER_COUNT
        or any(type(value) is not bool for value in ever)
        or type(counts) is not list or len(counts) != TRACER_COUNT
        or any(type(value) is not int or value < 0 for value in counts)
        or type(lifecycle) is not list or lifecycle != [0] * TRACER_COUNT
    ):
        raise Stage15QualificationError("midpoint continuation vector differs")
    expected_counts = [0] * TRACER_COUNT
    for row in prefix_routes:
        index = _require_int(row.tracer_index, "route tracer index", minimum=0)
        if index >= TRACER_COUNT:
            raise Stage15QualificationError("route tracer index is outside roster")
        expected_counts[index] += 1
    if counts != expected_counts:
        raise Stage15QualificationError("midpoint encounter-route counts differ")


def _validate_lane_identity(
    lane: Mapping[str, Any], *, expected_order: int, batch: Any,
    runtime_projection_sha256: str,
) -> tuple[str, int, int]:
    _require_exact_mapping_keys(lane, LANE_KEYS, "lane result")
    lane_id, steps = LANES[expected_order % 2]
    expected_batch = expected_order // 2
    if (
        _require_int(lane.get("lane_order"), "lane order", minimum=0)
        != expected_order
        or lane.get("batch_order") != expected_batch
        or lane.get("object_id") != batch.object_id
        or lane.get("lane_id") != lane_id
        or lane.get("engine") != LANE_ENGINE
        or lane.get("model") != LANE_MODEL
        or lane.get("steps_per_julian_year") != steps
        or lane.get("step_julian_years_hex") != (1.0 / steps).hex()
        or lane.get("final_global_step") != HORIZON_YEARS * steps
        or lane.get("runtime_projection_sha256") != runtime_projection_sha256
        or lane.get("no_removal_no_drop") is not True
    ):
        raise Stage15QualificationError("lane identity differs")
    return lane_id, steps, expected_batch


def _validate_lane_evidence(
    *, lane: Mapping[str, Any], expected_order: int, batch: Any,
    arrays_fd: int, source_fd: int, routes_fd: int, midpoint_fd: int,
    np: types.ModuleType, engine: types.ModuleType, v4: types.ModuleType,
    runtime_projection_sha256: str, expected_source_offset: int,
    expected_route_offset: int,
) -> tuple[dict[str, Any], int, int]:
    lane_id, steps, expected_batch = _validate_lane_identity(
        lane, expected_order=expected_order, batch=batch,
        runtime_projection_sha256=runtime_projection_sha256,
    )
    source = lane.get("source_nodes")
    array = lane.get("cartesian_array")
    route = lane.get("routes")
    midpoint = lane.get("midpoint")
    if not all(type(item) is dict for item in (source, array, route, midpoint)):
        raise Stage15QualificationError("lane evidence locator is absent")
    if (
        set(source) != SOURCE_LOCATOR_KEYS
        or set(array) != ARRAY_LOCATOR_KEYS
        or set(route) != ROUTE_LOCATOR_KEYS
        or set(midpoint) != MIDPOINT_LOCATOR_KEYS
    ):
        raise Stage15QualificationError("lane evidence locator shape differs")
    source_length = SOURCE_H64_LANE_BYTES if steps == 64 else SOURCE_H128_LANE_BYTES
    expected_source = {
        "schema": SOURCE_SCHEMA, "byte_offset": expected_source_offset,
        "byte_length": source_length, "node_count": HORIZON_YEARS * steps + 1,
        "record_bytes": SOURCE_NODE_RECORD_BYTES,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise Stage15QualificationError("source-node locator differs")
    if _fd_sha256(source_fd, source_length, expected_source_offset) != source.get("lane_payload_sha256"):
        raise Stage15QualificationError("source-node lane digest differs")
    _validate_segment_receipts(source_fd=source_fd, lane=lane, source=source)
    if (
        array.get("schema") != ARRAY_SCHEMA
        or array.get("byte_offset") != expected_order * ARRAY_LANE_BYTES
        or array.get("byte_length") != ARRAY_LANE_BYTES
        or array.get("checkpoint_count") != CHECKPOINT_COUNT
        or array.get("record_bytes") != ARRAY_RECORD_BYTES
        or _fd_sha256(arrays_fd, ARRAY_LANE_BYTES, expected_order * ARRAY_LANE_BYTES)
        != array.get("lane_payload_sha256")
    ):
        raise Stage15QualificationError("Cartesian lane locator differs")
    generated = _decode_lane_from_fd(arrays_fd, array, batch=batch, v4=v4)
    if not _exact_equal(lane.get("annual_summary"), v4._lane_summary(generated["checkpoints"])):
        raise Stage15QualificationError("annual lane summary differs from Cartesian arrays")
    summary = _source_summary_from_fd(
        fd=source_fd, offset=expected_source_offset,
        byte_length=source_length, node_count=HORIZON_YEARS * steps + 1,
        np=np, v4=v4,
        gm_values=[item.gm_AU3_per_julian_year2 for item in batch.massive],
    )
    if not _exact_equal(lane.get("all_node_summary"), summary):
        raise Stage15QualificationError("all-node invariant summary differs")
    for year in range(CHECKPOINT_COUNT):
        raw_source = _pread_all(
            source_fd, SOURCE_NODE_RECORD_BYTES,
            expected_source_offset + year * steps * SOURCE_NODE_RECORD_BYTES,
        )
        array_record = _pread_all(
            arrays_fd, ARRAY_RECORD_BYTES,
            expected_order * ARRAY_LANE_BYTES + year * ARRAY_RECORD_BYTES,
        )
        if _centered_source_payload(raw_source, batch=batch, v4=v4) != array_record[:SOURCE_NODE_RECORD_BYTES]:
            raise Stage15QualificationError("annual source node differs from Cartesian arrays")
    route_offset = _require_int(route.get("byte_offset"), "route lane offset", minimum=0)
    route_length = _require_int(route.get("byte_length"), "route lane length", minimum=0)
    if route.get("schema") != ROUTE_SCHEMA or route_offset != expected_route_offset:
        raise Stage15QualificationError("route locator differs")
    route_payload = _pread_all(routes_fd, route_length, route_offset)
    if _sha256(route_payload) != route.get("lane_payload_sha256"):
        raise Stage15QualificationError("route lane digest differs")
    routes, removals = _parse_route_lane_payload(route_payload, lane=lane, engine=engine)
    _validate_route_payload_source_witnesses(
        route_payload, lane=lane, batch=batch, source_fd=source_fd,
        source_offset=expected_source_offset, np=np, engine=engine,
    )
    prefix_routes, prefix_removals = _parse_route_lane_payload(
        route_payload, lane=lane, engine=engine, through_year_exclusive=500,
    )
    if (
        route.get("record_count") != len(routes) + len(removals)
        or route.get("route_count") != len(routes)
        or route.get("removal_count") != len(removals)
        or removals
        or lane.get("final_route_ledger_sha256")
        != _sha256(_canonical_json([asdict(row) for row in routes]))
        or lane.get("final_removal_ledger_sha256")
        != _sha256(_canonical_json([asdict(row) for row in removals]))
    ):
        raise Stage15QualificationError("route/removal ledger differs")
    midpoint_offset = expected_order * MIDPOINT_RECORD_BYTES
    midpoint_payload = _pread_all(midpoint_fd, MIDPOINT_RECORD_BYTES, midpoint_offset)
    if (
        midpoint.get("midpoint_order") != expected_order
        or midpoint.get("byte_offset") != midpoint_offset
        or midpoint.get("byte_length") != MIDPOINT_RECORD_BYTES
        or midpoint.get("record_sha256") != _sha256(midpoint_payload)
        or midpoint.get("batch_order") != expected_batch
        or midpoint.get("object_id") != batch.object_id
        or midpoint.get("lane_id") != lane_id
        or midpoint.get("steps_per_julian_year") != steps
        or midpoint.get("midpoint_year") != 500
        or midpoint.get("global_step") != 500 * steps
    ):
        raise Stage15QualificationError("midpoint locator differs")
    raw_source = _pread_all(
        source_fd, SOURCE_NODE_RECORD_BYTES,
        expected_source_offset + 500 * steps * SOURCE_NODE_RECORD_BYTES,
    )
    expected_checkpoint = _checkpoint_payload_from_raw(
        source_payload=raw_source, tracer_payload=midpoint_payload,
        year=500, batch=batch, v4=v4,
    )
    actual_checkpoint = _pread_all(
        arrays_fd, ARRAY_RECORD_BYTES,
        expected_order * ARRAY_LANE_BYTES + 500 * ARRAY_RECORD_BYTES,
    )
    if expected_checkpoint != actual_checkpoint:
        raise Stage15QualificationError("midpoint does not forward-map to Cartesian arrays")
    _validate_continuation(
        midpoint.get("continuation"), steps=steps,
        prefix_routes=prefix_routes, prefix_removals=prefix_removals,
        batch=batch,
    )
    _require_sha(lane.get("final_state_sha256"), "lane final-state SHA-256")
    return generated, expected_source_offset + source_length, route_offset + route_length


def _validate_restart_evidence(
    *, result: Mapping[str, Any], lanes: Sequence[Mapping[str, Any]],
    restart_fd: int, midpoint_fd: int, source_fd: int, arrays_fd: int,
    batches: Sequence[Any], v4: types.ModuleType,
    runtime_projection: Mapping[str, Any], runtime_document: Mapping[str, Any],
) -> None:
    evidence = result.get("restart_evidence")
    if (
        type(evidence) is not dict
        or set(evidence) != {"schema", "case_count", "cases", "file_sha256", "byte_length"}
        or evidence.get("schema") != RESTART_SCHEMA
        or evidence.get("case_count") != len(RESTART_CASES)
        or evidence.get("byte_length") != RESTART_STREAMS_EXACT_BYTES
        or evidence.get("file_sha256") != _fd_sha256(restart_fd, RESTART_STREAMS_EXACT_BYTES)
    ):
        raise Stage15QualificationError("restart evidence header differs")
    cases = evidence.get("cases")
    if type(cases) is not list or len(cases) != len(RESTART_CASES):
        raise Stage15QualificationError("restart case roster differs")
    lane_index = {(row["lane_id"], row["object_id"]): row for row in lanes}
    for order, key in enumerate(RESTART_CASES):
        row = cases[order]
        lane = lane_index.get(key)
        if (
            type(row) is not dict
            or set(row) != {
                "restart_order", "lane_id", "object_id", "batch_order",
                "byte_offset", "stream_byte_length", "monolithic_sha256",
                "fresh_resumed_sha256", "bitwise_suffix_match", "request",
                "worker_receipt", "worker_receipt_file_sha256",
            }
            or type(lane) is not dict
        ):
            raise Stage15QualificationError("restart case is absent")
        offset = order * 2 * RESTART_STREAM_BYTES
        monolithic = _pread_all(restart_fd, RESTART_STREAM_BYTES, offset)
        resumed = _pread_all(restart_fd, RESTART_STREAM_BYTES, offset + RESTART_STREAM_BYTES)
        request = row.get("request")
        receipt = row.get("worker_receipt")
        request_keys = {
            "schema", "restart_order", "batch_order", "object_id", "lane_id",
            "steps_per_julian_year", "start_year", "end_year", "staging_root",
            "lane", "expected_final_state_sha256",
            "expected_final_route_ledger_sha256",
            "expected_final_removal_ledger_sha256", "expected_runtime_projection",
            "content_sha256",
        }
        receipt_keys = {
            "schema", "request_file_sha256", "request_content_sha256",
            "restart_order", "batch_order", "object_id", "lane_id",
            "steps_per_julian_year", "start_year", "end_year",
            "suffix_checkpoint_count", "suffix_byte_length", "suffix_file_sha256",
            "source_segment_receipts_500_through_999",
            "source_segment_receipts_sha256", "final_global_step",
            "final_state_sha256", "final_route_ledger_sha256",
            "final_removal_ledger_sha256", "runtime_projection",
            "runtime_fresh_child_interpreter", "runtime_fresh_process_registry",
            "runtime_guard", "fresh_process", "midpoint_consumed",
            "content_sha256",
        }
        if (
            type(request) is not dict or set(request) != request_keys
            or type(receipt) is not dict or set(receipt) != receipt_keys
        ):
            raise Stage15QualificationError("restart request/receipt is absent")
        if (
            row.get("restart_order") != order or row.get("lane_id") != key[0]
            or row.get("object_id") != key[1]
            or row.get("batch_order") != lane.get("batch_order")
            or row.get("byte_offset") != offset
            or row.get("stream_byte_length") != RESTART_STREAM_BYTES
            or row.get("monolithic_sha256") != _sha256(monolithic)
            or row.get("fresh_resumed_sha256") != _sha256(resumed)
            or row.get("bitwise_suffix_match") is not True
            or row.get("worker_receipt_file_sha256") != _sha256(_canonical_json(receipt))
        ):
            raise Stage15QualificationError("restart case locator differs")
        if (
            request.get("schema")
            != "jx.tno.stage15-resolved11-encounter-qualification.restart-request.v5"
            or request.get("content_sha256") != _domain_sha(RESTART_DOMAIN, request)
            or request.get("restart_order") != order
            or request.get("batch_order") != lane.get("batch_order")
            or request.get("object_id") != key[1]
            or request.get("lane_id") != key[0]
            or request.get("steps_per_julian_year") != lane.get("steps_per_julian_year")
            or request.get("start_year") != 500 or request.get("end_year") != 1_000
            or not _exact_equal(request.get("lane"), lane)
            or not _exact_equal(request.get("expected_runtime_projection"), runtime_projection)
            or request.get("expected_final_state_sha256") != lane.get("final_state_sha256")
            or request.get("expected_final_route_ledger_sha256") != lane.get("final_route_ledger_sha256")
            or request.get("expected_final_removal_ledger_sha256") != lane.get("final_removal_ledger_sha256")
            or type(request.get("staging_root")) is not str
            or not Path(request["staging_root"]).is_absolute()
        ):
            raise Stage15QualificationError("restart request binding differs")
        if (
            receipt.get("request_file_sha256") != _sha256(_canonical_json(request))
            or receipt.get("restart_order") != order
            or receipt.get("batch_order") != lane.get("batch_order")
            or receipt.get("object_id") != key[1]
            or receipt.get("lane_id") != key[0]
            or receipt.get("steps_per_julian_year") != lane.get("steps_per_julian_year")
            or receipt.get("start_year") != 500 or receipt.get("end_year") != 1_000
            or receipt.get("suffix_byte_length") != RESTART_STREAM_BYTES
            or receipt.get("source_segment_receipts_sha256")
            != _sha256(_canonical_json(receipt.get("source_segment_receipts_500_through_999")))
            or not _exact_equal(receipt.get("runtime_projection"), runtime_projection)
            or not _exact_equal(
                receipt.get("runtime_fresh_child_interpreter"),
                runtime_document.get("fresh_child_interpreter"),
            )
            or not _exact_equal(
                receipt.get("runtime_fresh_process_registry"),
                runtime_document.get("fresh_process_registry"),
            )
            or not _exact_equal(
                _normalized_runtime_guard(receipt.get("runtime_guard")),
                _normalized_runtime_guard(runtime_document.get("guard")),
            )
        ):
            raise Stage15QualificationError("restart worker receipt binding differs")
        _require_restart_pair(monolithic, resumed, receipt, lane, request)
        midpoint = _pread_all(
            midpoint_fd, MIDPOINT_RECORD_BYTES,
            _require_int(lane["midpoint"]["byte_offset"], "restart midpoint offset", minimum=0),
        )
        if monolithic[:MIDPOINT_RECORD_BYTES] != midpoint:
            raise Stage15QualificationError("restart stream does not begin at retained midpoint")
        batch = batches[_require_int(lane.get("batch_order"), "restart batch", minimum=0)]
        steps = _require_int(lane.get("steps_per_julian_year"), "restart steps", minimum=1)
        source_base = _require_int(lane["source_nodes"]["byte_offset"], "restart source offset", minimum=0)
        arrays_base = _require_int(lane["cartesian_array"]["byte_offset"], "restart array offset", minimum=0)
        for suffix_order, year in enumerate(range(500, 1_001)):
            tracer_payload = monolithic[
                suffix_order * MIDPOINT_RECORD_BYTES : (suffix_order + 1) * MIDPOINT_RECORD_BYTES
            ]
            source_payload = _pread_all(
                source_fd, SOURCE_NODE_RECORD_BYTES,
                source_base + year * steps * SOURCE_NODE_RECORD_BYTES,
            )
            expected = _checkpoint_payload_from_raw(
                source_payload=source_payload, tracer_payload=tracer_payload,
                year=year, batch=batch, v4=v4,
            )
            actual = _pread_all(
                arrays_fd, ARRAY_RECORD_BYTES,
                arrays_base + year * ARRAY_RECORD_BYTES,
            )
            if expected != actual:
                raise Stage15QualificationError("restart stream does not forward-map to Cartesian arrays")


def _validate_artifact_receipt(
    value: Mapping[str, Any], *, name: str, fd: int, size: int,
) -> None:
    if (
        type(value) is not dict or set(value) != {"relative_path", "bytes", "file_sha256"}
        or value.get("relative_path") != name
        or value.get("bytes") != size
        or value.get("file_sha256") != _fd_sha256(fd, size)
    ):
        raise Stage15QualificationError(f"{name} artifact receipt differs")


def _validate_output_worker(output_root: Path, cache_root: Path) -> dict[str, Any]:
    root = _lexical_absolute(output_root)
    if (
        not root.is_dir() or root.is_symlink()
        or sorted(path.name for path in root.iterdir()) != sorted(ARTIFACT_ROSTER)
    ):
        raise Stage15QualificationError("output root artifact roster differs")
    protocol_payload = _regular_bytes(
        root / "PROTOCOL.json", "retained v5 protocol", maximum_bytes=PROTOCOL_MAX_BYTES,
    )
    protocol = _load_canonical_json(protocol_payload, "retained v5 protocol")
    validate_protocol(protocol)
    _input_bindings(full_arrays_hash=True)
    result_payload = _regular_bytes(
        root / "RESULT.json", "retained v5 result", maximum_bytes=RESULT_MAX_BYTES,
    )
    result = _load_canonical_json(result_payload, "retained v5 result")
    _require_exact_mapping_keys(result, RESULT_KEYS, "result root")
    runtime_payload = _regular_bytes(
        root / "RUNTIME.json", "retained v5 runtime", maximum_bytes=RUNTIME_MAX_BYTES,
    )
    retained_runtime = _load_canonical_json(runtime_payload, "retained v5 runtime")
    np, _scipy, _cp, engine, _backend_spec, _backend, _Provenance, current_runtime = _fresh_runtime_imports(cache_root)
    _validate_runtime_document(retained_runtime, current_runtime)
    runtime_projection = _runtime_process_independent_projection(retained_runtime)
    runtime_projection_sha = _sha256(_canonical_json(runtime_projection))
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("content_sha256") != _domain_sha(RESULT_DOMAIN, result)
        or result.get("protocol_content_sha256") != protocol.get("content_sha256")
        or result.get("runtime_content_sha256") != retained_runtime.get("content_sha256")
        or result.get("runtime_projection_sha256") != runtime_projection_sha
    ):
        raise Stage15QualificationError("result root binding differs")
    v4, _bridge, _custody, batches, oracle_locators = _load_v4_context()
    artifacts = result.get("artifacts")
    lanes = result.get("lanes")
    if type(artifacts) is not dict or set(artifacts) != set(RESULT_LAST_PREFIX):
        raise Stage15QualificationError("result artifact receipt roster differs")
    if type(lanes) is not list or len(lanes) != 24:
        raise Stage15QualificationError("result lane roster differs")
    with _stable_fd(root / "ARRAYS.bin", "Stage-1.5 arrays", exact_bytes=ARRAYS_EXACT_BYTES) as (arrays_fd, arrays_info), _stable_fd(
        root / "SOURCE_NODES.bin", "Stage-1.5 source nodes", exact_bytes=SOURCE_NODES_EXACT_BYTES
    ) as (source_fd, source_info), _stable_fd(
        root / "ROUTES.jsonl", "Stage-1.5 routes", maximum_bytes=ROUTE_MAX_BYTES
    ) as (routes_fd, routes_info), _stable_fd(
        root / "MIDPOINTS.bin", "Stage-1.5 midpoints", exact_bytes=MIDPOINTS_EXACT_BYTES
    ) as (midpoint_fd, midpoint_info), _stable_fd(
        root / "RESTART_STREAMS.bin", "Stage-1.5 restart streams", exact_bytes=RESTART_STREAMS_EXACT_BYTES
    ) as (restart_fd, restart_info), _stable_fd(
        V4_ARRAYS_PATH, "Stage-1 v4 oracle arrays", exact_bytes=V4_ARRAYS_BYTES
    ) as (oracle_fd, oracle_info):
        fd_rows = {
            "ARRAYS.bin": (arrays_fd, arrays_info.st_size),
            "SOURCE_NODES.bin": (source_fd, source_info.st_size),
            "ROUTES.jsonl": (routes_fd, routes_info.st_size),
            "MIDPOINTS.bin": (midpoint_fd, midpoint_info.st_size),
            "RESTART_STREAMS.bin": (restart_fd, restart_info.st_size),
        }
        for name, (fd, size) in fd_rows.items():
            _validate_artifact_receipt(artifacts.get(name), name=name, fd=fd, size=size)
        if _fd_sha256(oracle_fd, oracle_info.st_size) != V4_ARRAYS_SHA256:
            raise Stage15QualificationError("Stage-1 v4 oracle array digest differs")
        generated: dict[tuple[int, int], Mapping[str, Any]] = {}
        source_offset = 0
        route_offset = 0
        for order, lane in enumerate(lanes):
            batch = batches[order // 2]
            decoded, source_offset, route_offset = _validate_lane_evidence(
                lane=lane, expected_order=order, batch=batch,
                arrays_fd=arrays_fd, source_fd=source_fd, routes_fd=routes_fd,
                midpoint_fd=midpoint_fd, np=np, engine=engine, v4=v4,
                runtime_projection_sha256=runtime_projection_sha,
                expected_source_offset=source_offset, expected_route_offset=route_offset,
            )
            generated[(order // 2, LANES[order % 2][1])] = decoded
        if source_offset != SOURCE_NODES_EXACT_BYTES or route_offset != routes_info.st_size:
            raise Stage15QualificationError("lane evidence does not cover retained files exactly")
        route_record_total = sum(row["routes"]["record_count"] for row in lanes)
        if route_record_total > ROUTE_MAX_RECORDS:
            raise Stage15QualificationError("experiment-global route record cap was exceeded")
        oracle = {
            batch_order: _decode_v4_oracle_lane(
                oracle_fd, oracle_locators[batch_order],
                batch=batches[batch_order], v4=v4,
            )
            for batch_order in range(12)
        }
        gates = _evaluate_gates(generated, oracle, lanes, v4)
        if not _exact_equal(result.get("qualification_gates"), gates):
            raise Stage15QualificationError("qualification gates differ from offline replay")
        _validate_restart_evidence(
            result=result, lanes=lanes, restart_fd=restart_fd,
            midpoint_fd=midpoint_fd, source_fd=source_fd, arrays_fd=arrays_fd,
            batches=batches, v4=v4, runtime_projection=runtime_projection,
            runtime_document=retained_runtime,
        )
        route_evidence = result.get("route_evidence")
        if (
            type(route_evidence) is not dict
            or set(route_evidence)
            != {"schema", "record_count", "file_sha256", "byte_length"}
            or route_evidence.get("schema") != ROUTE_SCHEMA
            or route_evidence.get("record_count")
            != route_record_total
            or route_evidence.get("file_sha256") != _fd_sha256(routes_fd, routes_info.st_size)
            or route_evidence.get("byte_length") != routes_info.st_size
        ):
            raise Stage15QualificationError("global route evidence differs")
    with _stable_fd(root / "PROTOCOL.json", "protocol receipt", exact_bytes=len(protocol_payload)) as (fd, info):
        _validate_artifact_receipt(artifacts.get("PROTOCOL.json"), name="PROTOCOL.json", fd=fd, size=info.st_size)
    with _stable_fd(root / "RUNTIME.json", "runtime receipt", exact_bytes=len(runtime_payload)) as (fd, info):
        _validate_artifact_receipt(artifacts.get("RUNTIME.json"), name="RUNTIME.json", fd=fd, size=info.st_size)
    expected_status = (
        "QUALIFICATION_EXECUTED_PASS"
        if result["qualification_gates"]["all_predeclared_gates_passed"]
        else "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
    )
    expected_claims = {
        **CLAIMS, "dynamics_executed": True,
        "result_serialized_before_publication": True,
        "publication_state_not_claimed_inside_immutable_result": True,
        "qualification_established": result["qualification_gates"]["all_predeclared_gates_passed"],
    }
    if (
        result.get("status") != expected_status
        or not _exact_equal(result.get("claims"), expected_claims)
        or result.get("outcomes_generated") is not True
        or result.get("classification_generated") is not False
        or result.get("science_result") is not False
    ):
        raise Stage15QualificationError("result status or claim ceiling differs")
    return result


def _validation_receipt(result: Mapping[str, Any], root: Path) -> dict[str, Any]:
    # Hash the exact canonical document already replayed by the worker.  Do
    # not reopen RESULT here: a replacement between replay and receipt must
    # make the parent's later path hash differ from this validated byte image.
    result_payload = _canonical_json(result)
    value: dict[str, Any] = {
        "schema": VALIDATION_SCHEMA,
        "output_root": str(_lexical_absolute(root)),
        "result_file_sha256": _sha256(result_payload),
        "result_content_sha256": result.get("content_sha256"),
        "protocol_content_sha256": result.get("protocol_content_sha256"),
        "artifact_roster": list(ARTIFACT_ROSTER),
        "held_fd_semantic_replay_complete": True,
        "fresh_runtime_crossbinding_complete": True,
        "classification_generated": False,
        "science_result": False,
    }
    value["content_sha256"] = _domain_sha(VALIDATION_DOMAIN, value)
    return value


def _validate_output_in_fresh_child(output_root: Path) -> dict[str, Any]:
    root = _lexical_absolute(output_root)
    work = Path(tempfile.mkdtemp(prefix=".stage15-v5-validation-", dir=root.parent))
    cache_root = work / "runtime-cache"
    receipt_path = work / "VALIDATION.json"
    try:
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH), "validate-worker",
                "--output-root", str(root), "--cache-root", str(cache_root),
                "--validation-receipt", str(receipt_path),
            ],
            cwd=PROJECT_ROOT, env=_runtime_child_environment(cache_root),
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, check=False, timeout=604_800,
        )
        if completed.returncode != 0:
            detail = completed.stderr.decode("utf-8", errors="replace")[-12_000:]
            raise Stage15QualificationError(f"fresh offline validator failed: {detail}")
        payload = _regular_bytes(
            receipt_path, "fresh validation receipt", maximum_bytes=2_000_000,
        )
        receipt = _load_canonical_json(payload, "fresh validation receipt")
        result_payload = _regular_bytes(
            root / "RESULT.json", "post-validation result", maximum_bytes=RESULT_MAX_BYTES,
        )
        result = _load_canonical_json(result_payload, "post-validation result")
        if (
            receipt.get("schema") != VALIDATION_SCHEMA
            or receipt.get("content_sha256") != _domain_sha(VALIDATION_DOMAIN, receipt)
            or receipt.get("output_root") != str(root)
            or receipt.get("result_file_sha256") != _sha256(result_payload)
            or receipt.get("result_content_sha256") != result.get("content_sha256")
            or receipt.get("protocol_content_sha256") != result.get("protocol_content_sha256")
            or receipt.get("artifact_roster") != list(ARTIFACT_ROSTER)
            or receipt.get("held_fd_semantic_replay_complete") is not True
            or receipt.get("fresh_runtime_crossbinding_complete") is not True
            or receipt.get("classification_generated") is not False
            or receipt.get("science_result") is not False
        ):
            raise Stage15QualificationError("fresh validation receipt differs")
        return result
    finally:
        shutil.rmtree(work, ignore_errors=True)


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    """Replay all retained qualification evidence in a fresh exact-venv child."""

    return _validate_output_in_fresh_child(_lexical_absolute(output_root))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "protocol", "prepare", "validate-prepared", "execute",
            "execute-worker", "restart-worker", "validate", "validate-worker",
            "runtime-preflight", "runtime-preflight-worker",
            "continuation-smoke", "continuation-smoke-worker",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--restart-request", type=Path)
    parser.add_argument("--restart-output-root", type=Path)
    parser.add_argument("--validation-receipt", type=Path)
    parser.add_argument("--runtime-preflight-receipt", type=Path)
    parser.add_argument("--continuation-smoke-receipt", type=Path)
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "protocol":
        sys.stdout.buffer.write(_canonical_json(build_protocol()))
        return 0
    if args.command == "prepare":
        prepare(args.output_root)
        return 0
    if args.command == "validate-prepared":
        validate_prepared(args.output_root)
        return 0
    if args.command == "execute":
        execute(args.output_root)
        return 0
    if args.command == "validate":
        validate_output(args.output_root)
        return 0
    if args.command == "runtime-preflight":
        sys.stdout.buffer.write(_canonical_json(runtime_preflight()))
        return 0
    if args.command == "continuation-smoke":
        sys.stdout.buffer.write(_canonical_json(continuation_smoke()))
        return 0
    if args.cache_root is None:
        raise Stage15QualificationError("worker command requires --cache-root")
    if args.command == "execute-worker":
        _execute_worker(args.output_root, args.cache_root)
        return 0
    if args.command == "restart-worker":
        if args.restart_request is None or args.restart_output_root is None:
            raise Stage15QualificationError("restart worker arguments are incomplete")
        _restart_worker(
            args.restart_request, args.restart_output_root, args.cache_root,
        )
        return 0
    if args.command == "validate-worker":
        if args.validation_receipt is None:
            raise Stage15QualificationError("validate worker requires a receipt path")
        result = _validate_output_worker(args.output_root, args.cache_root)
        receipt = _validation_receipt(result, _lexical_absolute(args.output_root))
        _write_create_only(
            args.validation_receipt, _canonical_json(receipt), maximum=2_000_000,
        )
        return 0
    if args.command == "runtime-preflight-worker":
        if args.runtime_preflight_receipt is None:
            raise Stage15QualificationError(
                "runtime preflight worker requires a receipt path"
            )
        _runtime_preflight_worker(args.cache_root, args.runtime_preflight_receipt)
        return 0
    if args.command == "continuation-smoke-worker":
        if args.continuation_smoke_receipt is None:
            raise Stage15QualificationError(
                "continuation smoke worker requires a receipt path"
            )
        _continuation_smoke_worker(
            args.cache_root, args.continuation_smoke_receipt
        )
        return 0
    raise Stage15QualificationError("unknown command")


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except Stage15QualificationError as exc:
        print(f"Stage-1.5 v5 qualification error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
