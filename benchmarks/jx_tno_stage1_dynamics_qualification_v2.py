#!/usr/bin/env python3
"""Predeclare and execute the bounded Stage-1 v2 TNO dynamics qualification.

This is a numerical qualification, not a dynamical classifier.  It consumes
the sealed 12-object / 108-tracer native-epoch inputs, compares DE440s and
DE441 initial massive-body states, and runs two JX reduced-model refinements
plus independent REBOUND 5.1.1 IAS15 reduced- and resolved-model controls.

Importing this module and the ``prepare`` command do not import CuPy, JX, or
REBOUND and do not run dynamics.  ``execute`` is explicit, fail-closed, and
requires the complete retained DE441 acquisition before loading a dynamics
runtime.  No result produced here is a resonant, classical, scattering, or
detached classification and no population or Planet Nine inference is made.

V2 preserves the protocol-only v1 setup failure and uses JX's exact ``TDB``
time-scale token.  Julian-year units and the J2000-relative TDB epoch origin
remain explicit, separate coordinate metadata; the physics was not relabeled.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import stat
import struct
import subprocess
import sys
import tempfile
import time
import types
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v2.py"
PREDECESSOR_V1_SOURCE_PATH = (
    PROJECT_ROOT / "benchmarks" / "jx_tno_stage1_dynamics_qualification_v1.py"
)
PREDECESSOR_V1_TEST_PATH = (
    PROJECT_ROOT / "tests" / "test_tno_stage1_dynamics_qualification_v1.py"
)
PREDECESSOR_V1_ROOT = (
    PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v1"
)
PREDECESSOR_V1_PROTOCOL_PATH = PREDECESSOR_V1_ROOT / "PROTOCOL.json"
PILOT_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_covariance_classification_pilot_v1" / "PROTOCOL.json"
STAGE0_PROTOCOL_PATH = PROJECT_ROOT / "runs" / "tno_native_epoch_cartesian_qualification_v1" / "PROTOCOL.json"
STAGE0_STATES_PATH = PROJECT_ROOT / "runs" / "tno_native_epoch_cartesian_qualification_v1" / "STATES.json"
BRIDGE_ROOT = PROJECT_ROOT / "runs" / "tno_native_epoch_resolved11_bridge_v1"
BRIDGE_PROTOCOL_PATH = BRIDGE_ROOT / "PROTOCOL.json"
BRIDGE_RECEIPTS_PATH = BRIDGE_ROOT / "RECEIPTS.json"
BRIDGE_STATES_PATH = BRIDGE_ROOT / "BRIDGE.json"
DE441_TOOL_PATH = PROJECT_ROOT / "tools" / "jx_tno_de441_resolved11_acquisition_v1.py"
DE441_ROOT = PROJECT_ROOT / "runs" / "tno_de441_resolved11_native_epochs_v1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_stage1_dynamics_qualification_v2"
REBOUND_RUNTIME_ROOT = PROJECT_ROOT / "runs" / "tno_stage1_rebound511_runtime_v1"
REBOUND_WHEEL_PATH = REBOUND_RUNTIME_ROOT / "rebound-5.1.1-cp314-manylinux_x86_64.whl"
DEFAULT_REBOUND_TARGET = Path("/tmp/jx-m4-rebound511")

PROTOCOL_SCHEMA = "jx.tno.stage1-dynamics-qualification.protocol.v2"
RESULT_SCHEMA = "jx.tno.stage1-dynamics-qualification.result.v2"
IMPORT_PROBE_SCHEMA = "jx.tno.stage1-dynamics-qualification.import-probe.v2"
ARRAY_SCHEMA = "jx.tno.stage1-dynamics-qualification.cartesian-arrays.v2"
MIDPOINT_SCHEMA = "jx.tno.stage1-dynamics-qualification.jx-midpoints.v2"
RESTART_STREAM_SCHEMA = "jx.tno.stage1-dynamics-qualification.jx-restart-streams.v2"
JX_SOURCE_SCHEMA = "jx.tno.stage1-dynamics-qualification.jx-source-nodes.v2"
IAS_TRACE_SCHEMA = "jx.tno.stage1-dynamics-qualification.ias15-accepted-step-trace.v2"
RESTART_RECEIPT_SCHEMA = "jx.tno.stage1-dynamics-qualification.restart-receipt.v2"
SOURCE_BUILDER_PREFLIGHT_SCHEMA = (
    "jx.tno.stage1-dynamics-qualification.source-builder-semantic-preflight.v2"
)
PROTOCOL_DOMAIN = b"jx.tno.stage1-dynamics-qualification.protocol.v2\0"
RESULT_DOMAIN = b"jx.tno.stage1-dynamics-qualification.result.v2\0"
ARRAY_DOMAIN = b"jx.tno.stage1-dynamics-qualification.cartesian-arrays.v2\0"
MIDPOINT_DOMAIN = b"jx.tno.stage1-dynamics-qualification.jx-midpoints.v2\0"
RESTART_STREAM_DOMAIN = b"jx.tno.stage1-dynamics-qualification.jx-restart-streams.v2\0"
JX_SOURCE_DOMAIN = b"jx.tno.stage1-dynamics-qualification.jx-source-nodes.v2\0"
IAS_TRACE_DOMAIN = b"jx.tno.stage1-dynamics-qualification.ias15-accepted-step-trace.v2\0"
RESTART_DOMAIN = b"jx.tno.stage1-dynamics-qualification.restart-receipt.v2\0"
IMPORT_PROBE_DOMAIN = b"jx.tno.stage1-dynamics-qualification.import-probe.v2\0"
SCHEDULE_DOMAIN = b"jx.tno.stage1-dynamics-qualification.schedule.v2\0"
SOURCE_BUILDER_PREFLIGHT_DOMAIN = (
    b"jx.tno.stage1-dynamics-qualification.source-builder-semantic-preflight.v2\0"
)
JX_TILE_DOMAIN = b"jxplanetx.cuda-restricted-wh.source-tile.v1\0"
CLASSIFICATION = "TNO_STAGE1_V2_1000_JULIAN_YEAR_NUMERICAL_QUALIFICATION_ONLY"
CLAIM_CEILING = "NUMERICAL_AND_MODEL_SENSITIVITY_QUALIFICATION_NOT_DYNAMICAL_CLASSIFICATION"
JX_TIME_SCALE = "TDB"
JX_REJECTED_VERBOSE_TIME_SCALE = "TDB_JULIAN_YEARS_SINCE_J2000"
JX_TIME_COORDINATE_UNIT = "JULIAN_YEAR_EXACTLY_365_25_TDB_DAYS"
JX_EPOCH_ORIGIN_SEMANTICS = (
    "NATIVE_COVARIANCE_EPOCH_JD_TDB_MINUS_J2000_JD_2451545_0_DIVIDED_BY_365_25"
)
JX_PROVENANCE_VERSION = "v2"
JX_TILE_ID_PREFIX = "stage1-v2"
JX_PROVENANCE_ID_PREFIX = "jx-tno-stage1-v2"
PREDECESSOR_V1_SOURCE_SHA256 = (
    "ce1603d3e70f0fb7c9d139d91031e5afdd19d0af6bbab01852b15c7d6a0dda28"
)
PREDECESSOR_V1_TEST_SHA256 = (
    "024990f37b78d5f64a1825d73bf8bf9a7696523418f76ca92416624d5dcfeaed"
)
PREDECESSOR_V1_PROTOCOL_SHA256 = (
    "af5f57854317d047b989d9a97f6d13b1c776e23a07e900c674126cba153cd326"
)
PREDECESSOR_V1_PROTOCOL_CONTENT_SHA256 = (
    "b6ea80a50de6fb6c598f5d04df88f21fb021425a530b82ebd6984b4ede428066"
)
PREDECESSOR_V1_SETUP_FAILURE_CLASS = (
    "jxplanetx.engine.trajectory.TrajectoryContractError"
)
PREDECESSOR_V1_SETUP_FAILURE_MESSAGE = (
    "Wisdom--Holman time_scale must be one of ('TDB', 'SYNTHETIC')"
)
RESTART_WORKER_RECEIPT_KEYS = frozenset(
    {
        "schema", "request_content_sha256", "request_file_sha256",
        "implementation_file_sha256", "midpoint_record_sha256",
        "midpoint_consumed", "fresh_process_resume_start_year",
        "suffix_checkpoint_count", "suffix_byte_length", "suffix_file_sha256",
        "suffix_content_sha256", "final_state_sha256", "ledger",
        "ledger_content_sha256", "runtime_identity",
        "bitwise_match_checked_by_parent",
        "content_sha256",
    }
)
RESTART_REQUEST_KEYS = frozenset(
    {
        "schema", "ephemeris", "batch_order", "object_id", "steps_per_year",
        "midpoint_year", "midpoint_relative_path", "midpoint_byte_offset",
        "midpoint_byte_length", "midpoint_record_sha256", "content_sha256",
    }
)

HORIZON_JULIAN_YEARS = 1_000
CHECKPOINT_CADENCE_JULIAN_YEARS = 1
CHECKPOINT_COUNT = HORIZON_JULIAN_YEARS + 1
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
TNO_STATES_PER_BATCH = 9
OBJECT_COUNT = 12
TNO_STATE_COUNT = OBJECT_COUNT * TNO_STATES_PER_BATCH
EXPECTED_CARTESIAN_ARRAY_BYTES = 71_495_424
JX_MIDPOINT_YEAR = 500
JX_MIDPOINT_RECORD_DOUBLES = TNO_STATES_PER_BATCH * 6
JX_MIDPOINT_RECORD_BYTES = JX_MIDPOINT_RECORD_DOUBLES * 8
JX_MIDPOINT_RECORD_COUNT = OBJECT_COUNT * 2 * 2
EXPECTED_JX_MIDPOINT_BYTES = 20_736
REQUIRED_REBOUND_VERSION = "5.1.1"
REBOUND_WHEEL_SHA256 = "3dbef0d7cad0d70606c6c7abe87a059c154e55ee854bc3fa212b01a244dabd19"
REBOUND_INIT_SHA256 = "924e19bfddb22bca9cbff0c641eaf3b2f791287f33df2d8d852f80aac78f8cdd"
REBOUND_LIBRARY_SHA256 = "52df4fe1050b98d6b0db11f31c18ec14ed00697a987e904bd7587cce4a0ccdef"
PYTHON_EXECUTABLE_PATH = Path("/usr/bin/python3.14")
PYTHON_EXECUTABLE_BYTES = 7_477_160
PYTHON_EXECUTABLE_SHA256 = "52e0a13e60a981d8c4b6478be2ba5176f69da07948a056bf49cf6f077e30cb41"
PROJECT_VENV_ROOT = PROJECT_ROOT / ".venv"
PROJECT_VENV_PYTHON = PROJECT_VENV_ROOT / "bin" / "python"
PROJECT_VENV_PYVENV_CFG = PROJECT_VENV_ROOT / "pyvenv.cfg"
PROJECT_VENV_PYVENV_CFG_BYTES = 250
PROJECT_VENV_PYVENV_CFG_SHA256 = "5c81a7f0372c2a7687272a8e0068fe9f036e6422e2c6f5b11588cbf568dfbae3"
CUPY_DISTRIBUTION_VERSION = "14.2.0"
CUPY_SITE_PACKAGES = PROJECT_ROOT / ".venv" / "lib" / "python3.14" / "site-packages"
CUPY_RECORD_PATH = (
    CUPY_SITE_PACKAGES / "cupy_cuda13x-14.2.0.dist-info" / "RECORD"
)
CUPY_RECORD_BYTES = 317_866
CUPY_RECORD_SHA256 = "b8fa3fc62720ee9b5090dd3dd1eb5721b30aa849c4f25c93e1507f639f17399b"
CUPY_INIT_RELATIVE_PATH = "cupy/__init__.py"
CUPY_INIT_BYTES = 41_705
CUPY_INIT_SHA256 = "8c4724758587dea5f1c1d7c217c74a9fa0e4ed7f9d76a2b86fa001117cf3c718"
CUPY_CORE_RELATIVE_PATH = "cupy/_core/core.cpython-314-x86_64-linux-gnu.so"
CUPY_CORE_BYTES = 6_062_088
CUPY_CORE_SHA256 = "2c69571caa7814cb2efa8398b7c917310852203faaffc4425dda864a84d02be7"
CUPY_FFT_CONFIG_RELATIVE_PATH = "cupy/fft/_config.py"
CUPY_FFT_CONFIG_BYTES = 3_638
CUPY_FFT_CONFIG_SHA256 = "f988591ebac66e67cbe232ad5001940eb93e093d814717d97a3da1dc50a01e17"
NUMPY_DISTRIBUTION_VERSION = "2.5.3"
NUMPY_RECORD_PATH = CUPY_SITE_PACKAGES / "numpy-2.5.3.dist-info" / "RECORD"
NUMPY_RECORD_BYTES = 111_517
NUMPY_RECORD_SHA256 = "9d23d1a798b32c85a52dd64d7f484ef38cc29b84692b05c64c23a3a1aada2ca9"
NUMPY_METADATA_RELATIVE_PATH = "numpy-2.5.3.dist-info/METADATA"
NUMPY_METADATA_PATH = CUPY_SITE_PACKAGES / NUMPY_METADATA_RELATIVE_PATH
NUMPY_METADATA_BYTES = 6_635
NUMPY_METADATA_SHA256 = "451a9b8028000588e66b0b415587b6aef0bbc51a96d8e8a0cba0dc23acf64f99"
NUMPY_INIT_RELATIVE_PATH = "numpy/__init__.py"
NUMPY_INIT_BYTES = 25_220
NUMPY_INIT_SHA256 = "09295a80660f17925ae23765ce8cbd7ff7ceae968d5f2f89349f1cb74c0b9e11"
NUMPY_CORE_RELATIVE_PATH = "numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so"
NUMPY_CORE_BYTES = 10_629_105
NUMPY_CORE_SHA256 = "6fa6c8396aa1569ca1dbcfa44463b57f90e2dff42043f3ff22aa9e0165b839a3"
JX_NATIVE_RELATIVE_PATH = "src/jxplanetx/_wisdom_holman_cpu_v3.cpython-314-x86_64-linux-gnu.so"
JX_NATIVE_BYTES = 42_520
JX_NATIVE_SHA256 = "cb81c2f5c43aae638654413e1112a5d5f2b51b40f43f30d0e9b56074eca10ab3"
RUNTIME_MODULE_PREFIXES = (
    "jxplanetx", "numpy", "cupy", "cupyx", "cupy_backends", "rebound"
)
REQUIRED_RUNTIME_MODULE_NAMES = (
    "jxplanetx._wisdom_holman_cpu_v3",
    "numpy",
    "numpy._core._multiarray_umath",
    "cupy",
    "cupy._core.core",
    "cupy.fft._config",
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
IAS15_EPSILON = 1.0e-13
IAS15_INITIAL_DT_JULIAN_YEARS = 1.0 / 32.0
IAS15_MIN_DT_JULIAN_YEARS = 0.0
IAS15_ADAPTIVE_MODE = "PRS23"
JX_STEP_YEARS = (1.0 / 16.0, 1.0 / 32.0)
JX_STEPS_PER_YEAR = (16, 32)
JX_FRESH_PROCESS_RESTART_CASES = (
    ("DE440S", 16, "jpl-20026181"),
    ("DE440S", 32, "jpl-20303775"),
    ("DE441", 16, "jpl-20148209"),
    ("DE441", 32, "jpl-50666516"),
)
SCATTERING_DIAGNOSTIC_THRESHOLD = 0.0375
AU_METRES_EXACT = 149_597_870_700
SECONDS_PER_JULIAN_YEAR_EXACT = 31_557_600
SECONDS_PER_DAY_EXACT = 86_400

RESOLVED11_NAIF_IDS = (10, 199, 299, 399, 301, 4, 5, 6, 7, 8, 9)
RESOLVED11_BODY_IDS = (
    "SUN",
    "MERCURY",
    "VENUS",
    "EARTH",
    "MOON",
    "MARS_BARYCENTER",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
    "PLUTO_BARYCENTER",
)
INNER_COLLAPSE_NAIF_IDS = (10, 199, 299, 399, 301, 4)
REDUCED_OUTER_NAIF_IDS = (5, 6, 7, 8)
REDUCED_BODY_IDS = (
    "SUN_PLUS_MERCURY_VENUS_EARTH_MOON_MARS",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)
JX_RESTART_STREAM_RECORD_COUNT = len(JX_FRESH_PROCESS_RESTART_CASES) * 2
JX_RESTART_SUFFIX_CHECKPOINT_COUNT = HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1
JX_RESTART_SUFFIX_BYTES = JX_RESTART_SUFFIX_CHECKPOINT_COUNT * JX_MIDPOINT_RECORD_BYTES
EXPECTED_JX_RESTART_STREAM_BYTES = (
    JX_RESTART_STREAM_RECORD_COUNT * JX_RESTART_SUFFIX_BYTES
)
JX_SOURCE_RECORD_COUNT = OBJECT_COUNT * 2 * len(JX_STEPS_PER_YEAR)
JX_SOURCE_COUNT = len(REDUCED_BODY_IDS)
JX_SOURCE_GM_BYTES = JX_SOURCE_COUNT * 8
EXPECTED_JX_SOURCE_BYTES = JX_SOURCE_GM_BYTES + OBJECT_COUNT * 2 * sum(
    (HORIZON_JULIAN_YEARS * steps + 1) * JX_SOURCE_COUNT * 3 * 2 * 8
    for steps in JX_STEPS_PER_YEAR
)
EXPECTED_FIXED_BINARY_EVIDENCE_BYTES = (
    EXPECTED_CARTESIAN_ARRAY_BYTES
    + EXPECTED_JX_MIDPOINT_BYTES
    + EXPECTED_JX_RESTART_STREAM_BYTES
    + EXPECTED_JX_SOURCE_BYTES
)
DEFAULT_STAGED_FILE_MAXIMUM_BYTES = 1_000_000_000
IAS_TRACE_MAXIMUM_BYTES = 100_000_000_000
IAS_TRACE_HEADER = struct.Struct("<QIIIId")
IAS_TRACE_EVENT_NONFINITE = 1 << 0
IAS_TRACE_EVENT_ROSTER = 1 << 1
IAS_TRACE_EVENT_ENDPOINT = 1 << 2
IAS_TRACE_EVENT_CHORD = 1 << 3
IAS_TRACE_EVENT_OUTER = 1 << 4
IAS_TRACE_EVENT_CALLBACK = 1 << 5
IAS_TRACE_EVENT_MASK = (
    IAS_TRACE_EVENT_NONFINITE
    | IAS_TRACE_EVENT_ROSTER
    | IAS_TRACE_EVENT_ENDPOINT
    | IAS_TRACE_EVENT_CHORD
    | IAS_TRACE_EVENT_OUTER
    | IAS_TRACE_EVENT_CALLBACK
)

# These outcome-blind budgets are exact rational fractions of the future
# 0.0375 scattering diagnostic.  The binary64 values are serialization aids;
# acceptance comparisons below use the exact Fraction policies.
SCATTERING_THRESHOLD_RATIONAL = Fraction(3, 80)
NUMERICAL_TAIL_BUDGET_RATIONAL = Fraction(3, 8_000)
INTEGRATOR_BUDGET_RATIONAL = Fraction(3, 4_000)
MODEL_REDUCTION_BUDGET_RATIONAL = Fraction(3, 2_000)
EPHEMERIS_BUDGET_RATIONAL = Fraction(3, 2_000)
TOTAL_CHAIN_BUDGET_RATIONAL = Fraction(3, 800)
CARTESIAN_POSITION_BUDGET_AU_RATIONAL = Fraction(1, 500)
CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL = Fraction(1, 50_000)

# These thresholds qualify this exact experiment only; they are not
# population or classification thresholds.
GATE_THRESHOLDS = {
    "jx_richardson_semimajor_axis_tail": float(NUMERICAL_TAIL_BUDGET_RATIONAL),
    "jx_fine_vs_ias15_reduced_semimajor_axis": float(INTEGRATOR_BUDGET_RATIONAL),
    "ias15_reduced_vs_resolved11_semimajor_axis": float(MODEL_REDUCTION_BUDGET_RATIONAL),
    "de440s_vs_de441_ias15_semimajor_axis": float(EPHEMERIS_BUDGET_RATIONAL),
    "scattering_disagreement_chain": float(TOTAL_CHAIN_BUDGET_RATIONAL),
    "cartesian_position_AU": float(CARTESIAN_POSITION_BUDGET_AU_RATIONAL),
    "cartesian_velocity_AU_per_julian_year": float(CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL),
    "jx_reduced_max_relative_energy_drift": 1.0e-8,
    "jx_reduced_max_relative_angular_momentum_drift": 1.0e-10,
    "ias15_max_relative_energy_drift": 1.0e-10,
    "ias15_max_relative_angular_momentum_drift": 1.0e-11,
}

LANE_SPECS = (
    {
        "lane_id": "jx_reduced5_h_1_16",
        "engine": "JX_RESTRICTED_WH",
        "model": "REDUCED_5_BODY",
        "integration_step_julian_years_hex": (1.0 / 16.0).hex(),
        "steps_per_julian_year": 16,
        "total_fixed_steps": 16_000,
    },
    {
        "lane_id": "jx_reduced5_h_1_32",
        "engine": "JX_RESTRICTED_WH",
        "model": "REDUCED_5_BODY",
        "integration_step_julian_years_hex": (1.0 / 32.0).hex(),
        "steps_per_julian_year": 32,
        "total_fixed_steps": 32_000,
    },
    {
        "lane_id": "rebound511_ias15_reduced5",
        "engine": "REBOUND_5.1.1_IAS15",
        "model": "REDUCED_5_BODY",
        "adaptive": True,
        "epsilon_hex": IAS15_EPSILON.hex(),
        "initial_dt_julian_years_hex": IAS15_INITIAL_DT_JULIAN_YEARS.hex(),
        "min_dt_julian_years_hex": IAS15_MIN_DT_JULIAN_YEARS.hex(),
        "adaptive_mode": IAS15_ADAPTIVE_MODE,
    },
    {
        "lane_id": "rebound511_ias15_resolved11",
        "engine": "REBOUND_5.1.1_IAS15",
        "model": "RESOLVED_11_BODY",
        "adaptive": True,
        "epsilon_hex": IAS15_EPSILON.hex(),
        "initial_dt_julian_years_hex": IAS15_INITIAL_DT_JULIAN_YEARS.hex(),
        "min_dt_julian_years_hex": IAS15_MIN_DT_JULIAN_YEARS.hex(),
        "adaptive_mode": IAS15_ADAPTIVE_MODE,
    },
)

STATIC_BINDINGS = (
    ("benchmarks/jx_tno_covariance_classification_pilot_v1.py", "7d90ff40e71400071eb7d0ecf315113342e7c76cf8ae91935310f8895fcde2e1", "pilot_implementation"),
    ("runs/tno_covariance_classification_pilot_v1/PROTOCOL.json", "583260b2d9886f0745f1cffd37bd800d18238dfb791d976d5a539a72d5d87f4f", "pilot_protocol"),
    ("benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py", "8cf9ae282e46ddf740c98f3b340d00501079c65c3f53559272a2e9ae4d58e367", "stage0_implementation"),
    ("runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184", "stage0_protocol"),
    ("runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3", "stage0_states"),
    ("benchmarks/jx_tno_native_epoch_resolved11_bridge_v1.py", "5cb0b5380e7eac3e1c3c5932715dcbe43372b03581e65ef981ba491e78087e7e", "de440s_bridge_implementation"),
    ("runs/tno_native_epoch_resolved11_bridge_v1/PROTOCOL.json", "3c6c008698528581463e7c55f381e6a9c77443f80c330a60d0897058b1c48459", "de440s_bridge_protocol"),
    ("runs/tno_native_epoch_resolved11_bridge_v1/RECEIPTS.json", "3d7969ec02ffee40384029e9fb952082b5f7f112c832ee09230ba7118b310bd5", "de440s_bridge_receipts"),
    ("runs/tno_native_epoch_resolved11_bridge_v1/BRIDGE.json", "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f", "de440s_bridge_states"),
    ("tools/jx_tno_de441_resolved11_acquisition_v1.py", "84df09da90c16d49cc245d72512290962639390ab9e0966a5cedd07f91b68e00", "de441_acquisition_implementation"),
    ("tests/test_tno_de441_resolved11_acquisition_v1.py", "220215bae324d8119d03417f44997920b7a4304fb59eae19a375d94a8555d8b3", "de441_acquisition_test"),
    ("runs/tno_de441_resolved11_native_epochs_v1/PROTOCOL.json", "502b066e7edec7630d0b7d0f873d608a34bcf231ace5206f16c8a7392e24e62b", "de441_protocol"),
    ("runs/tno_de441_resolved11_native_epochs_v1/CUSTODY.json", "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3", "de441_custody"),
    ("src/jxplanetx/restricted_wh_source.py", "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52", "jx_massive_source"),
    ("src/jxplanetx/cuda_restricted_wh.py", "3c22c3d717fe5b07eb4ed929e5d32be591838fca8c1adbf02dec1263d3150927", "jx_tracer_map"),
    ("src/jxplanetx/_all_conic_kepler.py", "3d38c569764957bc0c781a4c2a7c767ccf2b630a38b0c88688e9716ac2b79691", "jx_all_conic_kepler"),
    ("src/jxplanetx/fast_wisdom_holman.py", "99da08305f5c0a191dd53e64d02a9a396b471aaf9f7b3a062545b1de220f7a72", "jx_massive_map"),
    ("src/jxplanetx/engine/contracts.py", "11c834620deedacd219a35bda46772f9be9f0d6243c644dee4fb02eb6c0c144d", "jx_contracts"),
    ("src/jxplanetx/engine/backends.py", "c039ae04aab43c0809b200304a47e9d6e63bb7cbea76e0665724c75ab513deba", "jx_backend"),
    ("runs/tno_stage1_rebound511_runtime_v1/rebound-5.1.1-cp314-manylinux_x86_64.whl", REBOUND_WHEEL_SHA256, "rebound511_official_wheel"),
    ("runs/tno_stage1_rebound511_runtime_v1/PYPI.json", "35bb559a4a6b0237e9de4941b0348b220f87d9280cbbe9e47a92c240ae572698", "rebound511_pypi_metadata"),
    ("benchmarks/jx_tno_stage1_dynamics_qualification_v1.py", PREDECESSOR_V1_SOURCE_SHA256, "predecessor_v1_implementation"),
    ("tests/test_tno_stage1_dynamics_qualification_v1.py", PREDECESSOR_V1_TEST_SHA256, "predecessor_v1_test"),
    ("runs/tno_stage1_dynamics_qualification_v1/PROTOCOL.json", PREDECESSOR_V1_PROTOCOL_SHA256, "predecessor_v1_protocol"),
    ("src/jxplanetx/engine/wisdom_holman.py", "2c3ed08e04d6a622af63f6be82afa1a222582e772a891179cdc01aa0f2a7383f", "jx_wisdom_holman_semantic_boundary"),
    ("src/jxplanetx/engine/wisdom_holman_contracts.py", "cc9d08abcaf166804d1876d9a8c1c0d55b1b5a7df161f245f1842f3b1b7beef7", "jx_wisdom_holman_semantic_constants"),
    ("src/jxplanetx/engine/trajectory.py", "2a34e30bb248e0dfb60b7bd559bc1a540ebb5133419b49c80ac973480971a1e0", "jx_trajectory_contract_error"),
)

EXPECTED_CONTENT_HASHES = {
    "pilot_protocol": "af27a4c5014b83d6b7c0241edba807e10ae635aaf641e6c5ad1eaf1fc1176db0",
    "stage0_protocol": "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e",
    "stage0_states": "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b",
    "bridge_protocol": "a7dbc361ffa4ccaf138f3c7afce269cad8bbb505b77930bcea9d04f5167b3c15",
    "bridge_receipts": "7e5f3ef0a731647a9adaebedae65d6f703e44b8a923eb624d6dc872d935b8058",
    "bridge_states": "5be6b00855d996c2717a4745631e32fe89f99d61615080bdf22fc6d44ea1da13",
    "de441_protocol": "f6fc67209bf69ba04f750903a9d368c95b2ef5d3ecdb9a789a651a3866698e8e",
    "de441_custody": "1d3e0fc4e6cda5e5c764b574027292b807477d893b9363a0ec33eee9af06b4b3",
}

PROTOCOL_CLAIMS = {
    "input_custody_and_rosters_prechecked": True,
    "qualification_design_predeclared_before_execution": True,
    "externally_preregistered": False,
    "dynamics_executed": False,
    "one_thousand_year_qualification_executed": False,
    "de440s_de441_sensitivity_executed": False,
    "dynamical_classification_generated": False,
    "resonance_classification_generated": False,
    "observational_selection_modeled": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class Stage1QualificationError(ValueError):
    """The Stage-1 contract, custody, runtime, or execution failed closed."""


@dataclass(frozen=True, slots=True)
class _StableReadHandle:
    descriptor: int
    path: Path
    label: str
    device: int
    inode: int
    size: int
    link_count: int
    modified_ns: int
    changed_ns: int


def _require_no_symlink_ancestors(path: Path, label: str) -> Path:
    """Return an absolute lexical path whose existing parents are real dirs."""
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:-1]:
            current /= part
            metadata = current.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise Stage1QualificationError(
                    f"{label} has a symlinked or non-directory ancestor: {current}"
                )
    except OSError as exc:
        raise Stage1QualificationError(
            f"{label} ancestor chain is unavailable: {current}"
        ) from exc
    return absolute


@contextmanager
def _stable_read_handle(
    path: Path, label: str, *, maximum: int
) -> Iterable[_StableReadHandle]:
    """Hold one O_NOFOLLOW descriptor across hashing and semantic replay."""

    if type(maximum) is not int or maximum <= 0:
        raise Stage1QualificationError(f"{label} maximum is invalid")
    absolute = _require_no_symlink_ancestors(path, label)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            absolute, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise Stage1QualificationError(
                f"{label} has an invalid bounded file identity"
            )
        handle = _StableReadHandle(
            descriptor=descriptor,
            path=absolute,
            label=label,
            device=before.st_dev,
            inode=before.st_ino,
            size=before.st_size,
            link_count=before.st_nlink,
            modified_ns=before.st_mtime_ns,
            changed_ns=before.st_ctime_ns,
        )
        yield handle
        after = os.fstat(descriptor)
        if (
            after.st_dev != handle.device
            or after.st_ino != handle.inode
            or after.st_size != handle.size
            or after.st_nlink != handle.link_count
            or after.st_mtime_ns != handle.modified_ns
            or after.st_ctime_ns != handle.changed_ns
        ):
            raise Stage1QualificationError(
                f"{label} changed during retained-byte validation"
            )
    except OSError as exc:
        raise Stage1QualificationError(
            f"cannot hold stable descriptor for {label}: {absolute}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


@contextmanager
def _source_only_runtime_import_guard(cache_root: Path) -> Iterable[dict[str, object]]:
    cache_root = Path(os.path.abspath(os.fspath(cache_root)))
    if cache_root.exists() or cache_root.is_symlink():
        raise Stage1QualificationError("source-only runtime cache path must start absent")
    previous_prefix = sys.pycache_prefix
    previous_dont_write = sys.dont_write_bytecode
    previous_environment = {
        key: os.environ.get(key)
        for key in ("CUPY_CACHE_DIR", "CUPY_CACHE_IN_MEMORY", "CUDA_CACHE_DISABLE")
    }
    kernel_cache = tempfile.TemporaryDirectory(prefix="jx-tno-stage1-v2-cupy-cache-")
    kernel_cache_root = Path(kernel_cache.name)
    if any(kernel_cache_root.iterdir()):
        kernel_cache.cleanup()
        raise Stage1QualificationError("isolated CuPy cache did not start empty")
    receipt = {
        "pycache_prefix": str(cache_root),
        "verified_absent_before": True,
        "dont_write_bytecode": True,
        "cupy_external_kernel_cache": {
            "directory": str(kernel_cache_root),
            "verified_empty_before": True,
            "cupy_cache_in_memory": "1",
            "cuda_cache_disable": "1",
        },
        "claim": "POINT_OBSERVATION_IN_TRUSTED_PROCESS_NOT_CONCURRENT_MUTATION_PROOF",
    }
    try:
        sys.pycache_prefix = str(cache_root)
        sys.dont_write_bytecode = True
        os.environ["CUPY_CACHE_DIR"] = str(kernel_cache_root)
        os.environ["CUPY_CACHE_IN_MEMORY"] = "1"
        os.environ["CUDA_CACHE_DISABLE"] = "1"
        yield receipt
        if cache_root.exists() or cache_root.is_symlink():
            raise Stage1QualificationError("source-only runtime cache path was created")
        if any(kernel_cache_root.iterdir()):
            raise Stage1QualificationError("disabled CuPy external kernel cache was written")
        receipt["verified_absent_after"] = True
        receipt["cupy_external_kernel_cache"]["verified_empty_after"] = True
    finally:
        sys.pycache_prefix = previous_prefix
        sys.dont_write_bytecode = previous_dont_write
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        kernel_cache.cleanup()
        receipt["cupy_external_kernel_cache"]["directory_removed_after"] = (
            not kernel_cache_root.exists()
        )


def _validate_runtime_import_guard(
    value: object,
    label: str,
    *,
    expected_pycache_prefix: Path | None = None,
) -> None:
    kernel_cache = (
        value.get("cupy_external_kernel_cache")
        if type(value) is dict else None
    )
    if (
        type(value) is not dict
        or set(value) != {
            "pycache_prefix", "verified_absent_before", "dont_write_bytecode",
            "cupy_external_kernel_cache", "claim", "verified_absent_after",
        }
        or type(value.get("pycache_prefix")) is not str
        or not Path(value["pycache_prefix"]).is_absolute()
        or (
            expected_pycache_prefix is not None
            and value.get("pycache_prefix")
            != str(Path(expected_pycache_prefix).resolve())
        )
        or value.get("verified_absent_before") is not True
        or value.get("verified_absent_after") is not True
        or value.get("dont_write_bytecode") is not True
        or value.get("claim")
        != "POINT_OBSERVATION_IN_TRUSTED_PROCESS_NOT_CONCURRENT_MUTATION_PROOF"
        or type(kernel_cache) is not dict
        or set(kernel_cache) != {
            "directory", "verified_empty_before", "cupy_cache_in_memory",
            "cuda_cache_disable", "verified_empty_after",
            "directory_removed_after",
        }
        or type(kernel_cache.get("directory")) is not str
        or not Path(kernel_cache["directory"]).is_absolute()
        or kernel_cache.get("verified_empty_before") is not True
        or kernel_cache.get("verified_empty_after") is not True
        or kernel_cache.get("directory_removed_after") is not True
        or kernel_cache.get("cupy_cache_in_memory") != "1"
        or kernel_cache.get("cuda_cache_disable") != "1"
    ):
        raise Stage1QualificationError(f"{label} runtime import guard changed")


@dataclass(frozen=True, slots=True)
class MassiveState:
    naif_id: int
    body_id: str
    gm_AU3_per_julian_year2: float
    position_AU: tuple[float, float, float]
    velocity_AU_per_julian_year: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TracerState:
    object_id: str
    spk_id: int
    sample_kind: str
    clone_index: int | None
    position_AU: tuple[float, float, float]
    velocity_AU_per_julian_year: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BatchState:
    ephemeris: str
    batch_order: int
    object_id: str
    spk_id: int
    epoch_jd_tdb: float
    massive: tuple[MassiveState, ...]
    tracers: tuple[TracerState, ...]


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
        raise Stage1QualificationError("value is not canonical-JSON encodable") from exc


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise Stage1QualificationError("SHA-256 payload must be bytes")
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(domain: bytes, value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise Stage1QualificationError("content-hash value must be an exact dictionary")
    copy = dict(value)
    copy.pop("content_sha256", None)
    return _sha256(domain + _canonical_json(copy))


def _regular_bytes(
    path: Path,
    label: str,
    maximum: int = 200_000_000,
    *,
    allow_empty: bool = False,
) -> bytes:
    path = _require_no_symlink_ancestors(path, label)
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise Stage1QualificationError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise Stage1QualificationError(f"{label} must be a single-link regular file")
    if (
        type(allow_empty) is not bool
        or metadata.st_size < 0
        or (metadata.st_size == 0 and not allow_empty)
        or metadata.st_size > maximum
    ):
        raise Stage1QualificationError(f"{label} has an invalid bounded byte size")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Stage1QualificationError(f"cannot read {label}") from exc
    if len(payload) != metadata.st_size:
        raise Stage1QualificationError(f"{label} changed while read")
    return payload


def _stream_file_digests(
    path: Path,
    label: str,
    *,
    maximum: int,
    content_domain: bytes | None = None,
) -> dict[str, object]:
    """Hash a bounded regular file incrementally without materializing it."""

    with _stable_read_handle(path, label, maximum=maximum) as handle:
        return _stream_handle_digests(handle, content_domain=content_domain)


def _stream_handle_digests(
    handle: _StableReadHandle,
    *,
    content_domain: bytes | None = None,
) -> dict[str, object]:
    if type(handle) is not _StableReadHandle:
        raise Stage1QualificationError("stable digest handle changed")
    file_digest = hashlib.sha256()
    content_digest = hashlib.sha256()
    if content_domain is not None:
        if type(content_domain) is not bytes:
            raise Stage1QualificationError("content digest domain must be bytes")
        content_digest.update(content_domain)
    cursor = 0
    while cursor < handle.size:
        length = min(8 * 1024 * 1024, handle.size - cursor)
        try:
            block = os.pread(handle.descriptor, length, cursor)
        except OSError as exc:
            raise Stage1QualificationError(
                f"cannot hash retained {handle.label}"
            ) from exc
        if len(block) != length:
            raise Stage1QualificationError(
                f"{handle.label} ended before its declared size"
            )
        cursor += length
        file_digest.update(block)
        if content_domain is not None:
            content_digest.update(block)
    result: dict[str, object] = {
        "path": str(handle.path),
        "bytes": handle.size,
        "sha256": file_digest.hexdigest(),
    }
    if content_domain is not None:
        result["content_sha256"] = content_digest.hexdigest()
    return result


def _read_exact_handle_region(
    handle: _StableReadHandle, offset: int, length: int, label: str
) -> bytes:
    if (
        type(handle) is not _StableReadHandle
        or type(offset) is not int
        or offset < 0
        or type(length) is not int
        or length <= 0
        or length > 20_000_000
        or offset + length > handle.size
    ):
        raise Stage1QualificationError(f"{label} region is invalid")
    try:
        payload = os.pread(handle.descriptor, length, offset)
    except OSError as exc:
        raise Stage1QualificationError(f"cannot read {label} region") from exc
    if len(payload) != length:
        raise Stage1QualificationError(f"{label} region is truncated")
    return payload


def _iter_exact_handle_records(
    handle: _StableReadHandle,
    offset: int,
    record_count: int,
    record_bytes: int,
    label: str,
) -> Iterable[bytes]:
    if (
        type(handle) is not _StableReadHandle
        or type(offset) is not int
        or offset < 0
        or type(record_count) is not int
        or record_count <= 0
        or type(record_bytes) is not int
        or record_bytes <= 0
        or record_bytes > 1_000_000
        or offset + record_count * record_bytes > handle.size
    ):
        raise Stage1QualificationError(f"{label} record layout is invalid")
    cursor = offset
    for _record_order in range(record_count):
        try:
            payload = os.pread(handle.descriptor, record_bytes, cursor)
        except OSError as exc:
            raise Stage1QualificationError(f"cannot stream {label}") from exc
        if len(payload) != record_bytes:
            raise Stage1QualificationError(f"{label} record is truncated")
        yield payload
        cursor += record_bytes


def _read_exact_region(path: Path, offset: int, length: int, label: str) -> bytes:
    if (
        type(offset) is not int
        or offset < 0
        or type(length) is not int
        or length <= 0
        or length > 20_000_000
    ):
        raise Stage1QualificationError(f"{label} region is invalid")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            Path(os.path.abspath(os.fspath(path))),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise Stage1QualificationError(f"{label} must be a single-link regular file")
        if offset + length > metadata.st_size:
            raise Stage1QualificationError(f"{label} region exceeds the file")
        payload = os.pread(descriptor, length, offset)
        if len(payload) != length:
            raise Stage1QualificationError(f"{label} region is truncated")
        after = os.fstat(descriptor)
        if (
            after.st_dev != metadata.st_dev
            or after.st_ino != metadata.st_ino
            or after.st_size != metadata.st_size
            or after.st_mtime_ns != metadata.st_mtime_ns
        ):
            raise Stage1QualificationError(f"{label} changed while read")
        return payload
    except OSError as exc:
        raise Stage1QualificationError(f"cannot read {label} region") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _iter_exact_fixed_records(
    path: Path,
    offset: int,
    record_count: int,
    record_bytes: int,
    label: str,
) -> Iterable[bytes]:
    """Yield fixed records from one stable descriptor with bounded memory."""

    if (
        type(offset) is not int or offset < 0
        or type(record_count) is not int or record_count <= 0
        or type(record_bytes) is not int or record_bytes <= 0
        or record_bytes > 1_000_000
    ):
        raise Stage1QualificationError(f"{label} record layout is invalid")
    path = Path(os.path.abspath(os.fspath(path)))
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or offset + record_count * record_bytes > before.st_size
        ):
            raise Stage1QualificationError(f"{label} file identity changed")
        cursor = offset
        for _record_order in range(record_count):
            payload = os.pread(descriptor, record_bytes, cursor)
            if len(payload) != record_bytes:
                raise Stage1QualificationError(f"{label} record is truncated")
            yield payload
            cursor += record_bytes
        after = os.fstat(descriptor)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise Stage1QualificationError(f"{label} changed while streamed")
    except OSError as exc:
        raise Stage1QualificationError(f"cannot stream {label}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _exact_live_file_receipt(
    path: Path,
    label: str,
    *,
    expected_path: Path | None = None,
    expected_bytes: int | None = None,
    expected_sha256: str | None = None,
    maximum: int = 200_000_000,
    allow_empty: bool = False,
) -> dict[str, object]:
    resolved = Path(os.path.abspath(os.fspath(path)))
    payload = _regular_bytes(
        resolved, label, maximum=maximum, allow_empty=allow_empty
    )
    receipt = {
        "path": str(resolved),
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }
    if (
        (
            expected_path is not None
            and resolved
            != Path(os.path.abspath(os.fspath(expected_path)))
        )
        or (expected_bytes is not None and len(payload) != expected_bytes)
        or (expected_sha256 is not None and receipt["sha256"] != expected_sha256)
    ):
        raise Stage1QualificationError(f"{label} differs from its predeclared receipt")
    return receipt


def _cupy_record_manifest() -> dict[str, tuple[int, str]]:
    payload = _regular_bytes(
        CUPY_RECORD_PATH, "predeclared CuPy RECORD", maximum=1_000_000
    )
    if len(payload) != CUPY_RECORD_BYTES or _sha256(payload) != CUPY_RECORD_SHA256:
        raise Stage1QualificationError("predeclared CuPy RECORD changed")
    try:
        rows = csv.reader(payload.decode("utf-8", errors="strict").splitlines())
        result: dict[str, tuple[int, str]] = {}
        for row in rows:
            if len(row) != 3:
                raise Stage1QualificationError("CuPy RECORD row shape changed")
            relative, digest_token, size_token = row
            relative_path = Path(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or "\\" in relative
            ):
                raise Stage1QualificationError("CuPy RECORD path is unsafe")
            if not digest_token and not size_token:
                continue
            if not digest_token.startswith("sha256=") or not size_token.isdigit():
                raise Stage1QualificationError("CuPy RECORD digest contract changed")
            encoded = digest_token.removeprefix("sha256=")
            decoded = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
            if len(decoded) != 32 or relative in result:
                raise Stage1QualificationError("CuPy RECORD entry is invalid or duplicate")
            result[relative] = (int(size_token), decoded.hex())
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise Stage1QualificationError("CuPy RECORD cannot be parsed exactly") from exc
    for relative, expected_bytes, expected_sha in (
        (CUPY_INIT_RELATIVE_PATH, CUPY_INIT_BYTES, CUPY_INIT_SHA256),
        (CUPY_CORE_RELATIVE_PATH, CUPY_CORE_BYTES, CUPY_CORE_SHA256),
        (
            CUPY_FFT_CONFIG_RELATIVE_PATH,
            CUPY_FFT_CONFIG_BYTES,
            CUPY_FFT_CONFIG_SHA256,
        ),
    ):
        if result.get(relative) != (expected_bytes, expected_sha):
            raise Stage1QualificationError("CuPy core receipt changed in RECORD")
    return result


def _numpy_record_manifest() -> dict[str, tuple[int, str]]:
    payload = _regular_bytes(
        NUMPY_RECORD_PATH, "NumPy RECORD", maximum=1_000_000
    )
    if len(payload) != NUMPY_RECORD_BYTES or _sha256(payload) != NUMPY_RECORD_SHA256:
        raise Stage1QualificationError("NumPy RECORD changed")
    result: dict[str, tuple[int, str]] = {}
    try:
        rows = csv.reader(payload.decode("utf-8", errors="strict").splitlines())
        for row in rows:
            if len(row) != 3:
                raise Stage1QualificationError("NumPy RECORD row shape changed")
            relative, digest_token, size_token = row
            relative_path = Path(relative)
            if (
                not relative
                or relative_path.is_absolute()
                or "\\" in relative
            ):
                raise Stage1QualificationError("NumPy RECORD path is unsafe")
            # The pinned wheel legitimately records its console script outside
            # site-packages.  It cannot own an imported NumPy module and is
            # therefore deliberately excluded from the runtime module map.
            if ".." in relative_path.parts:
                continue
            if not digest_token and not size_token:
                continue
            if not digest_token.startswith("sha256=") or not size_token.isdigit():
                raise Stage1QualificationError("NumPy RECORD digest contract changed")
            encoded = digest_token.removeprefix("sha256=")
            decoded = base64.urlsafe_b64decode(
                encoded + "=" * (-len(encoded) % 4)
            )
            if len(decoded) != 32 or relative in result:
                raise Stage1QualificationError(
                    "NumPy RECORD entry is invalid or duplicate"
                )
            result[relative] = (int(size_token), decoded.hex())
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise Stage1QualificationError("NumPy RECORD cannot be parsed exactly") from exc
    for relative, expected_bytes, expected_sha in (
        (
            NUMPY_METADATA_RELATIVE_PATH,
            NUMPY_METADATA_BYTES,
            NUMPY_METADATA_SHA256,
        ),
        (NUMPY_INIT_RELATIVE_PATH, NUMPY_INIT_BYTES, NUMPY_INIT_SHA256),
        (NUMPY_CORE_RELATIVE_PATH, NUMPY_CORE_BYTES, NUMPY_CORE_SHA256),
    ):
        if result.get(relative) != (expected_bytes, expected_sha):
            raise Stage1QualificationError("NumPy core receipt changed in RECORD")
    return result


def _predeclared_jx_file_manifest() -> list[dict[str, object]]:
    source_root = PROJECT_ROOT / "src" / "jxplanetx"
    paths = sorted(source_root.rglob("*.py"))
    native_path = PROJECT_ROOT / JX_NATIVE_RELATIVE_PATH
    if native_path not in paths:
        paths.append(native_path)
    records: list[dict[str, object]] = []
    for path in sorted(paths):
        payload = _regular_bytes(path, f"predeclared JX runtime file {path}")
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        records.append(
            {
                "relative_path": relative,
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )
    native = next(
        (item for item in records if item["relative_path"] == JX_NATIVE_RELATIVE_PATH),
        None,
    )
    if native != {
        "relative_path": JX_NATIVE_RELATIVE_PATH,
        "bytes": JX_NATIVE_BYTES,
        "sha256": JX_NATIVE_SHA256,
    }:
        raise Stage1QualificationError("predeclared native Wisdom-Holman binary changed")
    return records


def _project_venv_launcher_receipt() -> dict[str, object]:
    expected_links = (
        (PROJECT_VENV_PYTHON, "python3"),
        (PROJECT_VENV_ROOT / "bin" / "python3", "/usr/bin/python3"),
        (Path("/usr/bin/python3"), "python3.14"),
    )
    links: list[dict[str, str]] = []
    for path, expected_target in expected_links:
        _require_no_symlink_ancestors(
            path, "project venv launcher link"
        )
        try:
            metadata = path.lstat()
            observed_target = os.readlink(path)
        except OSError as exc:
            raise Stage1QualificationError(
                f"project venv launcher link is unavailable: {path}"
            ) from exc
        if not stat.S_ISLNK(metadata.st_mode) or observed_target != expected_target:
            raise Stage1QualificationError(
                f"project venv launcher link changed: {path}"
            )
        links.append({"path": str(path), "target": observed_target})
    pyvenv = _exact_live_file_receipt(
        PROJECT_VENV_PYVENV_CFG,
        "project venv pyvenv.cfg",
        expected_path=PROJECT_VENV_PYVENV_CFG,
        expected_bytes=PROJECT_VENV_PYVENV_CFG_BYTES,
        expected_sha256=PROJECT_VENV_PYVENV_CFG_SHA256,
        maximum=10_000,
    )
    resolved = _exact_live_file_receipt(
        PYTHON_EXECUTABLE_PATH,
        "project venv resolved Python",
        expected_path=PYTHON_EXECUTABLE_PATH,
        expected_bytes=PYTHON_EXECUTABLE_BYTES,
        expected_sha256=PYTHON_EXECUTABLE_SHA256,
        maximum=20_000_000,
    )
    return {
        "launcher_path": str(PROJECT_VENV_PYTHON),
        "symlink_chain": links,
        "pyvenv_cfg": pyvenv,
        "resolved_executable": resolved,
        "expected_sys_prefix": str(PROJECT_VENV_ROOT),
        "expected_sys_base_prefix": "/usr",
    }


def _fresh_child_interpreter_receipt() -> dict[str, object]:
    static = _project_venv_launcher_receipt()
    if (
        sys.executable != str(PROJECT_VENV_PYTHON)
        or sys.prefix != str(PROJECT_VENV_ROOT)
        or sys.base_prefix != "/usr"
    ):
        raise Stage1QualificationError(
            "fresh child was not launched by the exact project venv interpreter"
        )
    return {
        **static,
        "observed_sys_executable": sys.executable,
        "observed_sys_prefix": sys.prefix,
        "observed_sys_base_prefix": sys.base_prefix,
        "claim": "LEXICAL_PROJECT_VENV_LAUNCH_AND_EXACT_RESOLVED_INTERPRETER_POINT_OBSERVATION",
    }


def _predeclared_runtime_contract() -> dict[str, object]:
    python = _exact_live_file_receipt(
        Path(sys.executable).resolve(strict=True),
        "predeclared Python executable",
        expected_path=PYTHON_EXECUTABLE_PATH,
        expected_bytes=PYTHON_EXECUTABLE_BYTES,
        expected_sha256=PYTHON_EXECUTABLE_SHA256,
        maximum=20_000_000,
    )
    record = _exact_live_file_receipt(
        CUPY_RECORD_PATH,
        "predeclared CuPy RECORD",
        expected_path=CUPY_RECORD_PATH,
        expected_bytes=CUPY_RECORD_BYTES,
        expected_sha256=CUPY_RECORD_SHA256,
        maximum=1_000_000,
    )
    manifest = _cupy_record_manifest()
    numpy_manifest = _numpy_record_manifest()
    return {
        "claim": "PREDECLARED_FILE_BYTES_AND_TRUSTED_FRESH_PROCESS_OBSERVATIONS_NOT_SOFTWARE_AUTHENTICATION",
        "python_executable": python,
        "project_venv_launcher": _project_venv_launcher_receipt(),
        "cupy_distribution": {
            "distribution": "cupy-cuda13x",
            "version": CUPY_DISTRIBUTION_VERSION,
            "site_packages": str(CUPY_SITE_PACKAGES.resolve(strict=True)),
            "record": record,
            "hashed_member_count": len(manifest),
            "required_init": {
                "relative_path": CUPY_INIT_RELATIVE_PATH,
                "bytes": CUPY_INIT_BYTES,
                "sha256": CUPY_INIT_SHA256,
            },
            "required_core_extension": {
                "relative_path": CUPY_CORE_RELATIVE_PATH,
                "bytes": CUPY_CORE_BYTES,
                "sha256": CUPY_CORE_SHA256,
            },
            "required_fft_config": {
                "relative_path": CUPY_FFT_CONFIG_RELATIVE_PATH,
                "bytes": CUPY_FFT_CONFIG_BYTES,
                "sha256": CUPY_FFT_CONFIG_SHA256,
            },
        },
        "numpy_distribution": {
            "distribution": "numpy",
            "version": NUMPY_DISTRIBUTION_VERSION,
            "site_packages": str(CUPY_SITE_PACKAGES.resolve(strict=True)),
            "record": _exact_live_file_receipt(
                NUMPY_RECORD_PATH,
                "predeclared NumPy RECORD",
                expected_path=NUMPY_RECORD_PATH,
                expected_bytes=NUMPY_RECORD_BYTES,
                expected_sha256=NUMPY_RECORD_SHA256,
                maximum=1_000_000,
            ),
            "metadata": _exact_live_file_receipt(
                NUMPY_METADATA_PATH,
                "predeclared NumPy METADATA",
                expected_path=NUMPY_METADATA_PATH,
                expected_bytes=NUMPY_METADATA_BYTES,
                expected_sha256=NUMPY_METADATA_SHA256,
                maximum=100_000,
            ),
            "hashed_member_count": len(numpy_manifest),
            "required_init": {
                "relative_path": NUMPY_INIT_RELATIVE_PATH,
                "bytes": NUMPY_INIT_BYTES,
                "sha256": NUMPY_INIT_SHA256,
            },
            "required_core_extension": {
                "relative_path": NUMPY_CORE_RELATIVE_PATH,
                "bytes": NUMPY_CORE_BYTES,
                "sha256": NUMPY_CORE_SHA256,
            },
        },
        "jx_runtime_files": _predeclared_jx_file_manifest(),
        "rebound_wheel_sha256": REBOUND_WHEEL_SHA256,
    }


_EXACT_JSON_INT_KEYS = frozenset(
    {
        "bytes", "year", "steps_per_year", "steps_done",
        "global_source_step", "source_step_count", "exact_finish_time",
        "optimization_level", "device_index", "cuda_device_id",
        "cuda_runtime_version", "cuda_local_runtime_version",
        "cuda_driver_version", "maximum_bodies", "batch_order",
        "source_order", "midpoint_order", "restart_order",
        "epoch_order", "body_order", "tracer_order", "midpoint_year",
        "fresh_process_resume_start_year", "initial_observation_count",
        "heartbeat_callback_count", "integrate_call_count",
        "duplicate_entry_record_count", "total_observation_count",
        "source_tracer_pair_observation_count",
        "encounter_or_guard_event_count", "numerator", "denominator",
        "components", "record_doubles", "observed_N", "observed_N_active",
        "observed_pair_count", "event_flags", "metric_count",
        "counter_count", "tile_size", "source_steps_per_map_step",
        "steps_per_julian_year", "total_fixed_steps", "stream_order",
        "start_year", "end_year", "next_duplicate_entry_year",
        "previous_trace_step", "exact_numerator", "exact_denominator",
        "observed_exact_numerator", "observed_exact_denominator",
        "observed_squared_exact_numerator",
        "observed_squared_exact_denominator",
        "tracers_per_batch", "exact_tracer_count_per_ephemeris",
        "horizon_julian_years", "julian_year_seconds_exact",
        "checkpoint_cadence_julian_years", "checkpoint_count_per_lane",
        "midpoint_checkpoint_year", "records_per_stream",
        "fixed_binary_bytes_excluding_ias15_trace",
        "maximum_total_binary_bytes_including_ias15_trace",
        "final_global_source_step", "naif_id", "spk_id",
        "spice_et_seconds", "massive_state_rows_total",
        "tno_state_rows_total",
        "center_naif_id",
        "AU_metres_exact", "additional_horizon_after_stage_2_julian_years",
        "article", "begin_daf_address", "clone_denominator",
        "clones_per_object", "end_daf_address", "epochs_per_request",
        "exact_error_denominator", "exact_error_numerator",
        "fegetround_after", "fegetround_before", "frame_naif_id",
        "input_denominator", "input_numerator", "issue", "load_order",
        "maximum_data_read_calls", "maximum_tp_binary64_ulp_distance",
        "minimum_agreeing_clones_for_robust_class", "observer_naif_id",
        "origin_naif_id", "pointer_bits", "read_chunk_byte_limit",
        "seconds_per_julian_year_exact", "segment_type",
        "selected_body_naif_id", "spkpvn_returned_center_naif_id",
        "spkpvn_returned_frame_naif_id", "stage",
        "states_per_triggered_object", "target_naif_id",
        "target_total_horizon_julian_years", "terminal_center_naif_id",
        "tp_binary64_ulp_distance", "vector_table", "volume", "whole",
    }
)
_EXACT_JSON_INT_SUFFIXES = (
    "_offset", "_length", "_count", "_bytes", "_index", "_iterations"
)
_EXACT_JSON_INT_LIST_KEYS = frozenset(
    {
        "tracer_indices", "nvrtc_version", "naif_ids", "naif_ids_in_order",
        "holdout_clone_indices", "checkpoint_indices_inclusive",
        "central_members_naif_ids", "separate_outer_naif_ids",
        "tracer_indices_required_exactly",
        "provider_row_epoch_orders", "state_order_indexes",
        "centroid_members", "command_target_naif_ids_in_order",
        "holdout_clone_indices_per_object", "one_ulp_tp_spk_ids",
        "resolved11_naif_ids", "selected_clone_indices",
    }
)
_EXACT_JSON_INT_OR_STRING_KEYS = frozenset(
    {"seconds_per_julian_year_exact", "state_order"}
)
_EXACT_JSON_BOOLEAN_TUPLE_KEYS = frozenset({"massive"})


def _guard_exact_json_integer_identities(
    value: object, label: str, path: str = "$"
) -> None:
    if type(value) is dict:
        if value.get("__kind__") == "tuple" and type(value.get("items")) is list:
            tuple_items = value["items"]
            tuple_key = path.rsplit(".", 1)[-1]
            if any(type(entry) is bool for entry in tuple_items) and not (
                tuple_key in _EXACT_JSON_BOOLEAN_TUPLE_KEYS
                and all(type(entry) is bool for entry in tuple_items)
            ):
                raise Stage1QualificationError(
                    f"{label} typed non-boolean tuple contains a boolean at {path}.items"
                )
        for key, item in value.items():
            if type(key) is not str:
                raise Stage1QualificationError(
                    f"{label} contains a non-string JSON key at {path}"
                )
            child_path = f"{path}.{key}"
            integer_field = (
                key in _EXACT_JSON_INT_KEYS
                or key.endswith(_EXACT_JSON_INT_SUFFIXES)
            )
            if (
                integer_field
                and key not in _EXACT_JSON_INT_OR_STRING_KEYS
                and item is not None
                and type(item) is not int
            ):
                raise Stage1QualificationError(
                    f"{label} integer identity is not an exact int at {child_path}"
                )
            if (
                key in _EXACT_JSON_INT_OR_STRING_KEYS
                and type(item) not in (int, str)
            ):
                raise Stage1QualificationError(
                    f"{label} mixed integer identity is invalid at {child_path}"
                )
            if key in _EXACT_JSON_INT_LIST_KEYS:
                roster = item
                if (
                    type(roster) is dict
                    and set(roster) == {"__kind__", "items"}
                    and roster.get("__kind__") == "tuple"
                ):
                    roster = roster.get("items")
                if type(roster) is not list or any(type(entry) is not int for entry in roster):
                    raise Stage1QualificationError(
                        f"{label} integer roster is not exact at {child_path}"
                    )
            _guard_exact_json_integer_identities(item, label, child_path)
    elif type(value) is list:
        for index, item in enumerate(value):
            _guard_exact_json_integer_identities(
                item, label, f"{path}[{index}]"
            )


def _strict_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage1QualificationError(f"{label} is not strict JSON") from exc
    if type(value) is not dict or _canonical_json(value) != payload:
        raise Stage1QualificationError(f"{label} is not canonical JSON")
    _guard_exact_json_integer_identities(value, label)
    return value


def _load_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _regular_bytes(path, label)
    return payload, _strict_json(payload, label)


def _exact_float_hex(value: object, label: str) -> float:
    if type(value) is not str:
        raise Stage1QualificationError(f"{label} must be a binary64 hex string")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise Stage1QualificationError(f"{label} is not binary64 hex") from exc
    if not math.isfinite(result) or result.hex() != value:
        raise Stage1QualificationError(f"{label} is not canonical finite binary64 hex")
    return result


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _vec3(value: object, label: str) -> tuple[float, float, float]:
    if type(value) not in (list, tuple) or len(value) != 3:
        raise Stage1QualificationError(f"{label} must be an exact three-vector")
    result: list[float] = []
    for index, item in enumerate(value):
        if type(item) not in (int, float) or type(item) is bool:
            raise Stage1QualificationError(f"{label}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number):
            raise Stage1QualificationError(f"{label}[{index}] must be finite")
        result.append(number)
    return (result[0], result[1], result[2])


def _add(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[i]) + float(right[i]) for i in range(3))  # type: ignore[return-value]


def _sub(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return tuple(float(left[i]) - float(right[i]) for i in range(3))  # type: ignore[return-value]


def _scale(value: Sequence[float], factor: float) -> tuple[float, float, float]:
    return tuple(float(value[i]) * factor for i in range(3))  # type: ignore[return-value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(float(left[i]) * float(right[i]) for i in range(3))


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    )


def _norm(value: Sequence[float]) -> float:
    return math.sqrt(_dot(value, value))


def _fraction_to_outward_float(value: Fraction) -> float:
    """Return the least binary64 not smaller than a nonnegative fraction."""

    if type(value) is not Fraction or value < 0:
        raise Stage1QualificationError("outward conversion requires a nonnegative Fraction")
    result = float(value)
    if not math.isfinite(result):
        raise Stage1QualificationError("outward conversion overflowed")
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _fraction_record(value: Fraction) -> dict[str, object]:
    if type(value) is not Fraction or value < 0:
        raise Stage1QualificationError("fraction record requires a nonnegative Fraction")
    return {
        "exact_numerator": value.numerator,
        "exact_denominator": value.denominator,
        "outward_binary64_hex": _fraction_to_outward_float(value).hex(),
    }


def _fraction_from_record(value: object, label: str) -> Fraction:
    if type(value) is not dict or set(value) != {
        "exact_numerator",
        "exact_denominator",
        "outward_binary64_hex",
    }:
        raise Stage1QualificationError(f"{label} exact-fraction record shape changed")
    numerator = value.get("exact_numerator")
    denominator = value.get("exact_denominator")
    if (
        type(numerator) is not int
        or numerator < 0
        or type(denominator) is not int
        or denominator <= 0
    ):
        raise Stage1QualificationError(f"{label} exact-fraction record is invalid")
    result = Fraction(numerator, denominator)
    if (
        numerator != result.numerator
        or denominator != result.denominator
        or value.get("outward_binary64_hex")
        != _fraction_to_outward_float(result).hex()
    ):
        raise Stage1QualificationError(f"{label} exact-fraction record is noncanonical")
    return result


def _exact_absolute_difference(left: float, right: float) -> Fraction:
    if type(left) is not float or type(right) is not float:
        raise Stage1QualificationError("exact difference operands must be built-in floats")
    if not (math.isfinite(left) and math.isfinite(right)):
        raise Stage1QualificationError("exact difference operands must be finite")
    return abs(Fraction.from_float(left) - Fraction.from_float(right))


def _exact_squared_vector_difference(
    left: Sequence[float], right: Sequence[float]
) -> Fraction:
    left_vector = _vec3(left, "left exact comparison vector")
    right_vector = _vec3(right, "right exact comparison vector")
    return sum(
        (
            _exact_absolute_difference(left_vector[index], right_vector[index]) ** 2
            for index in range(3)
        ),
        Fraction(0, 1),
    )


def _exact_squared_reference_ulp_floor(reference: Sequence[float]) -> Fraction:
    vector = _vec3(reference, "exact ULP reference vector")
    return sum(
        (
            Fraction.from_float(64.0 * math.ulp(component)) ** 2
            for component in vector
        ),
        Fraction(0, 1),
    )


def _sqrt_fraction_to_outward_float(value: Fraction) -> float:
    """Return a binary64 whose square is not smaller than ``value``."""

    if type(value) is not Fraction or value < 0:
        raise Stage1QualificationError("outward square-root requires a nonnegative Fraction")
    if value == 0:
        return 0.0
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    even_exponent = exponent if exponent % 2 == 0 else exponent - 1
    if even_exponent >= 0:
        scaled = value / Fraction(1 << even_exponent, 1)
    else:
        scaled = value * Fraction(1 << (-even_exponent), 1)
    result = math.ldexp(math.sqrt(float(scaled)), even_exponent // 2)
    if not math.isfinite(result):
        raise Stage1QualificationError("outward square-root overflowed")
    while Fraction.from_float(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    while result > 0.0:
        predecessor = math.nextafter(result, -math.inf)
        if Fraction.from_float(predecessor) ** 2 < value:
            break
        result = predecessor
    return result


def _outward_absolute_difference(left: float, right: float) -> float:
    if not (math.isfinite(left) and math.isfinite(right)):
        raise Stage1QualificationError("difference operands must be finite")
    return _fraction_to_outward_float(
        abs(Fraction.from_float(left) - Fraction.from_float(right))
    )


def _outward_divide(numerator: float, denominator: float) -> float:
    if not (math.isfinite(numerator) and numerator >= 0.0):
        raise Stage1QualificationError("ratio numerator must be finite and nonnegative")
    if not (math.isfinite(denominator) and denominator > 0.0):
        raise Stage1QualificationError("ratio denominator must be finite and positive")
    return _fraction_to_outward_float(
        Fraction.from_float(numerator) / Fraction.from_float(denominator)
    )


def _outward_vector_difference(
    left: Sequence[float], right: Sequence[float]
) -> float:
    left_vector = _vec3(left, "left comparison vector")
    right_vector = _vec3(right, "right comparison vector")
    component_bounds = tuple(
        _outward_absolute_difference(left_vector[index], right_vector[index])
        for index in range(3)
    )
    value = math.hypot(*component_bounds)
    if not math.isfinite(value):
        raise Stage1QualificationError("vector comparison norm overflowed")
    return math.nextafter(value, math.inf)


def _outward_reference_ulp_norm(reference: Sequence[float]) -> float:
    vector = _vec3(reference, "ULP reference vector")
    floors = tuple(64.0 * math.ulp(component) for component in vector)
    value = math.hypot(*floors)
    if not math.isfinite(value):
        raise Stage1QualificationError("ULP reference norm overflowed")
    return math.nextafter(value, math.inf)


def _passes_rational(observed: float, threshold: Fraction) -> bool:
    if not math.isfinite(observed) or observed < 0.0 or type(threshold) is not Fraction:
        return False
    return Fraction.from_float(observed) <= threshold


def _angle(value: float) -> float:
    result = math.fmod(value, 2.0 * math.pi)
    return result + 2.0 * math.pi if result < 0.0 else result


def extract_elliptic_orbital_elements(
    position: Sequence[float], velocity: Sequence[float], mu: float
) -> dict[str, float]:
    """Return pure barycentric elliptic elements in AU, Julian years, radians.

    Circular convention: ``peri=0`` and mean anomaly is argument of latitude.
    Equatorial convention: ``node=0`` and peri is longitude of periapsis.
    Circular-equatorial convention: node=peri=0 and M is true longitude.
    """

    r = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    if type(mu) is not float or not math.isfinite(mu) or not mu > 0.0:
        raise Stage1QualificationError("mu must be a positive finite built-in float")
    radius = _norm(r)
    h = _cross(r, v)
    hnorm = _norm(h)
    if radius == 0.0 or hnorm == 0.0:
        raise Stage1QualificationError("orbital state is singular")
    speed2 = _dot(v, v)
    rv = _dot(r, v)
    eccentricity_vector = _scale(
        _sub(_scale(r, speed2 - mu / radius), _scale(v, rv)), 1.0 / mu
    )
    eccentricity = _norm(eccentricity_vector)
    specific_energy = 0.5 * speed2 - mu / radius
    if not specific_energy < 0.0 or not 0.0 <= eccentricity < 1.0:
        raise Stage1QualificationError("only finite bound elliptic states are accepted")
    semimajor_axis = -mu / (2.0 * specific_energy)
    perihelion = semimajor_axis * (1.0 - eccentricity)
    inclination = math.acos(max(-1.0, min(1.0, h[2] / hnorm)))
    node_vector = (-h[1], h[0], 0.0)
    node_norm = _norm(node_vector)
    tolerance = 64.0 * sys.float_info.epsilon
    node = _angle(math.atan2(node_vector[1], node_vector[0])) if node_norm > tolerance * hnorm else 0.0
    if eccentricity > tolerance:
        if node_norm > tolerance * hnorm:
            sin_peri = _dot(_cross(node_vector, eccentricity_vector), h) / (node_norm * eccentricity * hnorm)
            cos_peri = _dot(node_vector, eccentricity_vector) / (node_norm * eccentricity)
            peri = _angle(math.atan2(sin_peri, cos_peri))
        else:
            peri = _angle(math.atan2(eccentricity_vector[1], eccentricity_vector[0]))
        sin_true = _dot(_cross(eccentricity_vector, r), h) / (eccentricity * radius * hnorm)
        cos_true = _dot(eccentricity_vector, r) / (eccentricity * radius)
        true_anomaly = math.atan2(sin_true, cos_true)
    else:
        peri = 0.0
        if node_norm > tolerance * hnorm:
            sin_true = _dot(_cross(node_vector, r), h) / (node_norm * radius * hnorm)
            cos_true = _dot(node_vector, r) / (node_norm * radius)
            true_anomaly = math.atan2(sin_true, cos_true)
        else:
            true_anomaly = math.atan2(r[1], r[0])
    eccentric_anomaly = math.atan2(
        math.sqrt(max(0.0, 1.0 - eccentricity * eccentricity)) * math.sin(true_anomaly),
        eccentricity + math.cos(true_anomaly),
    )
    mean_anomaly = _angle(eccentric_anomaly - eccentricity * math.sin(eccentric_anomaly))
    values = {
        "semimajor_axis_AU": semimajor_axis,
        "eccentricity": eccentricity,
        "perihelion_AU": perihelion,
        "inclination_radians": inclination,
        "longitude_of_ascending_node_radians": node,
        "argument_of_perihelion_radians": peri,
        "mean_anomaly_radians": mean_anomaly,
    }
    if not all(math.isfinite(item) for item in values.values()):
        raise Stage1QualificationError("orbital element extraction produced nonfinite output")
    return values


def preliminary_scattering_metric(
    semimajor_axis_samples_AU: Sequence[float], *, expected_count: int | None = None
) -> dict[str, object]:
    """Compute the preliminary Gladman-style delta-a/a diagnostic only.

    The threshold bit is a qualification observable and MUST NOT be presented
    as a 1,000-year dynamical classification.
    """

    if type(semimajor_axis_samples_AU) not in (list, tuple) or not semimajor_axis_samples_AU:
        raise Stage1QualificationError("semimajor-axis samples must be a nonempty sequence")
    if expected_count is not None and (
        type(expected_count) is not int
        or expected_count <= 0
        or len(semimajor_axis_samples_AU) != expected_count
    ):
        raise Stage1QualificationError("semimajor-axis sample count differs")
    values: list[float] = []
    for index, item in enumerate(semimajor_axis_samples_AU):
        if type(item) is not float or not math.isfinite(item) or not item > 0.0:
            raise Stage1QualificationError(f"semimajor-axis sample {index} is invalid")
        values.append(item)
    initial = values[0]
    maximum_delta_fraction = max(
        abs(Fraction.from_float(item) - Fraction.from_float(initial))
        for item in values
    )
    ratio_fraction = maximum_delta_fraction / Fraction.from_float(initial)
    maximum_delta = _fraction_to_outward_float(maximum_delta_fraction)
    ratio = _fraction_to_outward_float(ratio_fraction)
    return {
        "sample_count": len(values),
        "initial_semimajor_axis_AU": initial,
        "minimum_semimajor_axis_AU": min(values),
        "maximum_semimajor_axis_AU": max(values),
        "maximum_absolute_delta_AU": maximum_delta,
        "maximum_absolute_delta_exact": _fraction_record(maximum_delta_fraction),
        "maximum_absolute_delta_over_initial": ratio,
        "maximum_absolute_delta_over_initial_exact": _fraction_record(ratio_fraction),
        "threshold": SCATTERING_DIAGNOSTIC_THRESHOLD,
        "comparison": "STRICTLY_GREATER_THAN",
        "threshold_exceeded_diagnostic_only": ratio_fraction > SCATTERING_THRESHOLD_RATIONAL,
        "dynamical_classification_generated": False,
    }


def _verify_static_bindings() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for relative, expected, role in STATIC_BINDINGS:
        payload = _regular_bytes(PROJECT_ROOT / relative, role)
        actual = _sha256(payload)
        if actual != expected:
            raise Stage1QualificationError(f"bound input changed: {relative}")
        records.append(
            {"relative_path": relative, "role": role, "bytes": len(payload), "file_sha256": actual}
        )
    return records


def _predecessor_v1_failure_record() -> dict[str, object]:
    if (
        not PREDECESSOR_V1_ROOT.is_dir()
        or PREDECESSOR_V1_ROOT.is_symlink()
        or sorted(entry.name for entry in PREDECESSOR_V1_ROOT.iterdir())
        != ["PROTOCOL.json"]
    ):
        raise Stage1QualificationError(
            "predecessor v1 output root is not the sealed protocol-only roster"
        )
    protocol_payload, protocol = _load_json(
        PREDECESSOR_V1_PROTOCOL_PATH, "predecessor v1 protocol"
    )
    if (
        _sha256(protocol_payload) != PREDECESSOR_V1_PROTOCOL_SHA256
        or protocol.get("schema")
        != "jx.tno.stage1-dynamics-qualification.protocol.v1"
        or protocol.get("content_sha256")
        != PREDECESSOR_V1_PROTOCOL_CONTENT_SHA256
    ):
        raise Stage1QualificationError("predecessor v1 protocol identity changed")
    return {
        "status": "SETUP_FAILED_BEFORE_FIRST_COMPLETED_LANE",
        "failure_id": "V1_UNSUPPORTED_JX_TIME_SCALE_TOKEN",
        "predecessor_source_relative_path": (
            "benchmarks/jx_tno_stage1_dynamics_qualification_v1.py"
        ),
        "predecessor_source_file_sha256": PREDECESSOR_V1_SOURCE_SHA256,
        "predecessor_test_relative_path": (
            "tests/test_tno_stage1_dynamics_qualification_v1.py"
        ),
        "predecessor_test_file_sha256": PREDECESSOR_V1_TEST_SHA256,
        "predecessor_protocol_relative_path": (
            "runs/tno_stage1_dynamics_qualification_v1/PROTOCOL.json"
        ),
        "predecessor_protocol_file_sha256": PREDECESSOR_V1_PROTOCOL_SHA256,
        "predecessor_protocol_content_sha256": (
            PREDECESSOR_V1_PROTOCOL_CONTENT_SHA256
        ),
        "observed_output_root_roster": ["PROTOCOL.json"],
        "failure_stage": "PRIMARY_JX_SOURCE_TILE_BUILD_SETUP",
        "failure_field": "JX_SOURCE_TILE.time_scale",
        "failure_class": PREDECESSOR_V1_SETUP_FAILURE_CLASS,
        "failure_message": PREDECESSOR_V1_SETUP_FAILURE_MESSAGE,
        "rejected_time_scale_token": JX_REJECTED_VERBOSE_TIME_SCALE,
        "required_time_scale_token": JX_TIME_SCALE,
        "affected_construction_paths": [
            "PRIMARY_EXECUTION", "RESTART_WORKER"
        ],
        "detection_phase": "EXPLICIT_EXECUTION_SETUP_BEFORE_FIRST_LANE",
        "reported_elapsed_seconds_approximate": 3.4,
        "completed_lane_count": 0,
        "dynamics_executed": False,
        "result_artifacts_published": False,
        "dynamics_outcome_generated": False,
        "qualification_established": False,
        "scientific_disproof": False,
        "disposition": "SUPERSEDED_BY_V2_WITH_V1_BYTES_PRESERVED",
        "predecessor_bytes_mutated": False,
    }


def _require_content(value: Mapping[str, Any], expected: str, label: str) -> None:
    if value.get("content_sha256") != expected:
        raise Stage1QualificationError(f"{label} content digest changed")


def _verify_input_documents() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, object]]]:
    bindings = _verify_static_bindings()
    _, pilot = _load_json(PILOT_PROTOCOL_PATH, "pilot protocol")
    _, stage0_protocol = _load_json(STAGE0_PROTOCOL_PATH, "Stage-0 protocol")
    _, stage0_states = _load_json(STAGE0_STATES_PATH, "Stage-0 states")
    _, bridge_protocol = _load_json(BRIDGE_PROTOCOL_PATH, "bridge protocol")
    _, bridge_receipts = _load_json(BRIDGE_RECEIPTS_PATH, "bridge receipts")
    _, bridge = _load_json(BRIDGE_STATES_PATH, "bridge states")
    _, de441_protocol = _load_json(DE441_ROOT / "PROTOCOL.json", "DE441 protocol")
    _, custody = _load_json(DE441_ROOT / "CUSTODY.json", "DE441 custody")
    for value, expected, label in (
        (pilot, EXPECTED_CONTENT_HASHES["pilot_protocol"], "pilot protocol"),
        (stage0_protocol, EXPECTED_CONTENT_HASHES["stage0_protocol"], "Stage-0 protocol"),
        (stage0_states, EXPECTED_CONTENT_HASHES["stage0_states"], "Stage-0 states"),
        (bridge_protocol, EXPECTED_CONTENT_HASHES["bridge_protocol"], "bridge protocol"),
        (bridge_receipts, EXPECTED_CONTENT_HASHES["bridge_receipts"], "bridge receipts"),
        (bridge, EXPECTED_CONTENT_HASHES["bridge_states"], "bridge states"),
        (de441_protocol, EXPECTED_CONTENT_HASHES["de441_protocol"], "DE441 protocol"),
        (custody, EXPECTED_CONTENT_HASHES["de441_custody"], "DE441 custody"),
    ):
        _require_content(value, expected, label)
    if (
        pilot.get("execution_authorized") is not False
        or stage0_protocol.get("execution_authorized") is not False
        or stage0_states.get("execution_authorized") is not False
        or bridge.get("schema") != "jx.tno.native-epoch-resolved11-bridge.states.v1"
        or bridge.get("batch_count") != OBJECT_COUNT
        or bridge.get("tno_state_rows_total") != TNO_STATE_COUNT
        or bridge.get("massive_state_rows_total") != 132
        or bridge.get("execution_authorized") is not False
        or custody.get("schema") != "jx.tno.de441-resolved11-native-epochs.custody.v1"
        or custody.get("state_record_count") != 132
        or custody.get("state_roster_sha256") != "f57f04d68909312d5fe53918eb8c7527310282034d678690b2f99619cafa3cd2"
        or custody.get("execution_authorized") is not False
    ):
        raise Stage1QualificationError("upstream input claim, schema, or count changed")
    requests = custody.get("requests")
    if type(requests) is not list or len(requests) != len(RESOLVED11_NAIF_IDS):
        raise Stage1QualificationError("DE441 raw request roster changed")
    for request, naif_id in zip(requests, RESOLVED11_NAIF_IDS, strict=True):
        if type(request) is not dict or request.get("naif_id") != naif_id:
            raise Stage1QualificationError("DE441 raw request order changed")
        relative = request.get("retained_relative_path")
        expected = request.get("raw_sha256")
        if type(relative) is not str or type(expected) is not str:
            raise Stage1QualificationError("DE441 raw request binding is invalid")
        payload = _regular_bytes(DE441_ROOT / relative, f"DE441 {relative}")
        if _sha256(payload) != expected:
            raise Stage1QualificationError(f"DE441 raw response changed: {relative}")
        bindings.append(
            {
                "relative_path": f"runs/tno_de441_resolved11_native_epochs_v1/{relative}",
                "role": "de441_raw_response",
                "bytes": len(payload),
                "file_sha256": expected,
            }
        )
    _validate_bridge_roster(bridge, custody)
    return bridge, custody, bindings


def _validate_bridge_roster(bridge: Mapping[str, Any], custody: Mapping[str, Any]) -> None:
    batches = bridge.get("batches")
    records = custody.get("state_records")
    if type(batches) is not list or len(batches) != OBJECT_COUNT or type(records) is not list or len(records) != 132:
        raise Stage1QualificationError("native batch roster changed")
    object_ids: list[str] = []
    for batch_order, batch in enumerate(batches):
        if type(batch) is not dict:
            raise Stage1QualificationError("bridge batch type changed")
        object_id = batch.get("object_id")
        massive = batch.get("massive_bodies")
        tnos = batch.get("tnos")
        if (
            type(object_id) is not str
            or type(massive) is not list
            or [row.get("naif_id") for row in massive if type(row) is dict] != list(RESOLVED11_NAIF_IDS)
            or type(tnos) is not list
            or len(tnos) != TNO_STATES_PER_BATCH
            or [row.get("clone_index") for row in tnos if type(row) is dict] != [None, *HOLDOUT_CLONE_INDICES]
        ):
            raise Stage1QualificationError("bridge batch selection or order changed")
        epoch_rows = records[batch_order * 11 : (batch_order + 1) * 11]
        if [row.get("naif_id") for row in epoch_rows if type(row) is dict] != list(RESOLVED11_NAIF_IDS):
            raise Stage1QualificationError("DE441 body order changed")
        if any(row.get("object_id") != object_id or row.get("epoch_order") != batch_order for row in epoch_rows):
            raise Stage1QualificationError("DE441 epoch-to-object binding changed")
        object_ids.append(object_id)
    if len(set(object_ids)) != OBJECT_COUNT:
        raise Stage1QualificationError("object roster contains duplicates")


def _batch_roster(bridge: Mapping[str, Any]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, batch in enumerate(bridge["batches"]):
        tnos = batch["tnos"]
        result.append(
            {
                "batch_order": index,
                "object_id": batch["object_id"],
                "epoch_jd_tdb_hex": batch["covariance_epoch_jd_tdb_hex"],
                "tracers": [
                    {
                        "tracer_order": order,
                        "sample_kind": row["sample_kind"],
                        "clone_index": row["clone_index"],
                        "native_vector_sha256": row["native_vector_sha256"],
                    }
                    for order, row in enumerate(tnos)
                ],
            }
        )
    return result


def build_protocol() -> dict[str, object]:
    bridge, _custody, bindings = _verify_input_documents()
    implementation = _regular_bytes(SCRIPT_PATH, "Stage-1 implementation")
    predecessor = _predecessor_v1_failure_record()
    protocol: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "claim_ceiling": CLAIM_CEILING,
        "status": "PREDECLARED_V2_READY_FOR_EXPLICIT_QUALIFICATION_EXECUTION",
        "predecessor_v1_setup_failure": predecessor,
        "nonclaim": (
            "A successful 1000-Julian-year execution qualifies only the stated numerical and "
            "model-sensitivity gates. It is not a final TNO dynamical or resonance classification, "
            "survey-bias analysis, population inference, or Planet Nine result."
        ),
        "selection": {
            "native_epoch_batch_count": OBJECT_COUNT,
            "tracers_per_batch": TNO_STATES_PER_BATCH,
            "exact_tracer_count_per_ephemeris": TNO_STATE_COUNT,
            "holdout_clone_indices": list(HOLDOUT_CLONE_INDICES),
            "batches": _batch_roster(bridge),
        },
        "time_contract": {
            "horizon_julian_years": HORIZON_JULIAN_YEARS,
            "julian_year_seconds_exact": SECONDS_PER_JULIAN_YEAR_EXACT,
            "jx_engine_time_scale_token": JX_TIME_SCALE,
            "jx_time_coordinate_unit": JX_TIME_COORDINATE_UNIT,
            "jx_epoch_origin_semantics": JX_EPOCH_ORIGIN_SEMANTICS,
            "jx_time_scale_token_does_not_encode_epoch_origin_or_unit": True,
            "checkpoint_cadence_julian_years": CHECKPOINT_CADENCE_JULIAN_YEARS,
            "checkpoint_indices_inclusive": [0, HORIZON_JULIAN_YEARS],
            "checkpoint_count_per_lane": CHECKPOINT_COUNT,
            "integration_step_is_not_checkpoint_cadence": True,
        },
        "ephemerides": ["DE440S", "DE441"],
        "lanes_per_ephemeris": [dict(value) for value in LANE_SPECS],
        "expected_batch_lane_count": OBJECT_COUNT * 2 * len(LANE_SPECS),
        "physics": {
            "force_law": "NEWTONIAN_POINT_MASS",
            "frame": "ICRF_J2000_INERTIAL",
            "length_unit": "AU",
            "time_unit": "JULIAN_YEAR_365.25_DAYS",
            "common_gm_policy": "EXACT_DE440_BRIDGE_GM_BINARY64_VALUES_USED_FOR_BOTH_EPHEMERIDES",
            "tno_mass": "CANONICAL_POSITIVE_ZERO",
            "tno_backreaction": False,
            "tracer_tracer_interactions": False,
            "collisions": "NONE",
            "removal_or_ejection": "FORBIDDEN_ANY_GUARD_OR_ROSTER_CHANGE_FAILS_WHOLE_QUALIFICATION",
            "reduced_model": {
                "central_members_naif_ids": list(INNER_COLLAPSE_NAIF_IDS),
                "central_state_rule": "RETAIN_SUN_POSITION_AND_VELOCITY",
                "central_gm_rule": "SUM_COMMON_GMS_OF_INNER_MEMBERS",
                "separate_outer_naif_ids": list(REDUCED_OUTER_NAIF_IDS),
                "pluto_policy": "OMITTED_FROM_REDUCED_MODEL_NOT_FOLDED_INTO_CENTRAL_GM",
                "final_origin": "REDUCED_5_BODY_COMMON_GM_BARYCENTER",
            },
            "resolved_model": {"naif_ids_in_order": list(RESOLVED11_NAIF_IDS)},
        },
        "jx_adapter": {
            "massive_provider": "build_wisdom_holman_source_tile",
            "tracer_integrator": "advance_cuda_restricted_wh_fused",
            "time_scale_token": JX_TIME_SCALE,
            "time_coordinate_unit": JX_TIME_COORDINATE_UNIT,
            "epoch_origin_semantics": JX_EPOCH_ORIGIN_SEMANTICS,
            "legacy_verbose_time_scale_token_forbidden": (
                JX_REJECTED_VERBOSE_TIME_SCALE
            ),
            "cpu_source_builder_semantic_preflight": {
                "schema": SOURCE_BUILDER_PREFLIGHT_SCHEMA,
                "public_builder": (
                    "jxplanetx.restricted_wh_source."
                    "build_wisdom_holman_source_tile"
                ),
                "construction_paths": [
                    "PRIMARY_EXECUTION", "RESTART_WORKER"
                ],
                "accepted_token": JX_TIME_SCALE,
                "rejected_token": JX_REJECTED_VERBOSE_TIME_SCALE,
                "expected_rejection_class": (
                    PREDECESSOR_V1_SETUP_FAILURE_CLASS
                ),
                "expected_rejection_message": (
                    PREDECESSOR_V1_SETUP_FAILURE_MESSAGE
                ),
                "trajectory": "SYNTHETIC_TWO_BODY_ONE_FIXED_CPU_SOURCE_STEP",
                "gpu_import_or_query": False,
                "dynamics_qualification_generated": False,
            },
            "full_source_tile_per_batch_lane": True,
            "annual_slice_rule": "FULL_TILE_NODES[YEAR*N:(YEAR+1)*N+1]",
            "slice_global_source_step_rule": "YEAR_TIMES_STEPS_PER_YEAR",
            "source_steps_per_map_step": 1,
            "source_encounter_radius_AU_hex": float(1.0).hex(),
            "minimum_massive_pair_separation_AU_hex": float(1.0).hex(),
            "primary_outer_boundary_AU_hex": float(4096.0).hex(),
            "acceleration_certificate": "NEXTUP_OF_2_TIMES_TOTAL_REDUCED_GM_DIVIDED_BY_MIN_RADIUS_SQUARED",
            "guard_policy": "TRANSACTIONAL_FAIL_NO_LIFECYCLE_REMOVAL",
            "tracer_indices_required_exactly": list(range(TNO_STATES_PER_BATCH)),
            "python_bytecode_policy": "FRESH_VERIFIED_ABSENT_PYCACHE_PREFIX_AND_DONT_WRITE_BYTECODE_THROUGH_RUNTIME",
            "cupy_kernel_cache_policy": "ISOLATED_EMPTY_DIRECTORY_WITH_CUPY_IN_MEMORY_CACHE_AND_CUDA_DISK_CACHE_DISABLED",
            "midpoint_checkpoint_year": JX_MIDPOINT_YEAR,
            "all_48_jx_midpoints_retained_and_revalidated_before_continuation": True,
            "fresh_process_restart_cases": [
                {"ephemeris": ephemeris, "steps_per_year": steps, "object_id": object_id}
                for ephemeris, steps, object_id in JX_FRESH_PROCESS_RESTART_CASES
            ],
            "fresh_process_restart_acceptance": "BITWISE_IDENTICAL_INTERNAL_TRACER_STATES_YEARS_500_THROUGH_1000_FINAL_STATE_AND_SOURCE_SLICE_LEDGER",
            "retained_restart_stream": {
                "schema": RESTART_STREAM_SCHEMA,
                "relative_path": "JX_RESTART_STREAMS.bin",
                "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64",
                "order": "CASE_THEN_MONOLITHIC_RESUMED_THEN_YEAR_500_THROUGH_1000_THEN_TRACER_POSITIONS_VELOCITIES",
                "record_bytes": JX_MIDPOINT_RECORD_BYTES,
                "records_per_stream": JX_RESTART_SUFFIX_CHECKPOINT_COUNT,
                "stream_bytes": JX_RESTART_SUFFIX_BYTES,
                "stream_count": JX_RESTART_STREAM_RECORD_COUNT,
                "exact_file_bytes": EXPECTED_JX_RESTART_STREAM_BYTES,
                "offline_acceptance": "PAIRED_STREAM_BYTES_EQUAL_AND_FORWARD_CENTERED_AGAINST_ARRAYS_WITH_SOURCE_NODES",
            },
            "retained_full_source_nodes": {
                "schema": JX_SOURCE_SCHEMA,
                "relative_path": "JX_SOURCE_NODES.bin",
                "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64",
                "order": "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_16_32_POSITIONS_THEN_VELOCITIES",
                "common_gm_bytes": JX_SOURCE_GM_BYTES,
                "lane_count": JX_SOURCE_RECORD_COUNT,
                "exact_file_bytes": EXPECTED_JX_SOURCE_BYTES,
                "offline_acceptance": "REBUILD_FULL_AND_ANNUAL_TILE_DIGESTS_ALL_NODE_INVARIANTS_AND_FORWARD_CENTERED_ANNUAL_ARRAY_ROWS",
            },
            "cuda_backend": {"backend_id": "cupy", "device": "cuda:0", "tile_size": 128, "float64": True, "fallback": False, "fast_math": False},
        },
        "rebound_control": {
            "required_version": REQUIRED_REBOUND_VERSION,
            "retained_wheel_relative_path": "runs/tno_stage1_rebound511_runtime_v1/rebound-5.1.1-cp314-manylinux_x86_64.whl",
            "retained_wheel_sha256": REBOUND_WHEEL_SHA256,
            "wheel_rebound_init_sha256": REBOUND_INIT_SHA256,
            "wheel_librebound_sha256": REBOUND_LIBRARY_SHA256,
            "runtime_target_is_external_and_must_exactly_match_wheel_members": True,
            "runtime_target_default": str(DEFAULT_REBOUND_TARGET),
            "integrator": "IAS15",
            "epsilon_hex": IAS15_EPSILON.hex(),
            "initial_dt_julian_years_hex": IAS15_INITIAL_DT_JULIAN_YEARS.hex(),
            "min_dt_julian_years_hex": IAS15_MIN_DT_JULIAN_YEARS.hex(),
            "adaptive_mode": IAS15_ADAPTIVE_MODE,
            "exact_finish_time": 1,
            "same_reduced_model_control_separates_integrator_error": True,
            "resolved11_control_separates_model_reduction_error": True,
            "guard_sampling": "INITIAL_PLUS_EVERY_REBOUND_HEARTBEAT_ACCEPTED_STEP_AND_STRAIGHT_CHORD_BETWEEN_SAMPLES",
            "guard_count_rule": "INITIAL_OBSERVATION_COUNT_EQUALS_1; HEARTBEAT_CALLBACK_COUNT_EQUALS_STEPS_DONE_PLUS_1000_INTEGRATE_ENTRY_DUPLICATES; EXACTLY_ONE_BITWISE_DUPLICATE_AT_EACH_INTEGER_YEAR_0_THROUGH_999; EVERY_NONDUPLICATE_RECORD_INCREMENTS_STEP_BY_ONE_AND_STRICTLY_ADVANCES_TIME; TOTAL_EQUALS_INITIAL_PLUS_HEARTBEAT; PAIR_COUNT_EQUALS_TOTAL_TIMES_TRACERS_TIMES_SOURCES",
            "guard_is_not_continuous_certified_geometry": True,
            "retained_accepted_step_trace": {
                "schema": IAS_TRACE_SCHEMA,
                "relative_path": "IAS15_ACCEPTED_STEPS.bin",
                "encoding": "FIXED_PER_LANE_RAW_LITTLE_ENDIAN_RECORDS",
                "lane_order": "EPHEMERIS_BATCH_THEN_REDUCED5_RESOLVED11_CALLBACK_ORDER",
                "header_struct": "<QIIIId",
                "header_fields": ["steps_done", "observed_N", "observed_N_active", "observed_pair_count", "event_flags", "time_julian_years"],
                "state_order": "MASSIVE_POSITIONS_MASSIVE_VELOCITIES_TRACER_POSITIONS_TRACER_VELOCITIES",
                "reduced5_record_bytes": 704,
                "resolved11_record_bytes": 992,
                "record_count_runtime_variable": True,
                "offline_acceptance": "RECOMPUTE_ROSTER_FINITE_ENDPOINT_CHORD_OUTER_COUNTS_TIMES_FLAGS_MINIMA_AND_ANNUAL_ARRAY_BINDINGS",
            },
            "source_tracer_encounter_radius_AU_hex": float(1.0).hex(),
            "massive_massive_encounters_are_not_tested_against_one_AU": True,
        },
        "observables": {
            "retained_annual_cartesian_states": "MASSIVE_AND_ALL_NINE_TRACERS_POSITION_AU_AND_VELOCITY_AU_PER_JULIAN_YEAR",
            "cartesian_payload_encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
            "cartesian_payload_order": "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL",
            "derived_barycentric_osculating_elliptic_elements": ["a", "e", "q", "i", "node", "peri", "M"],
            "element_units": {"a": "AU", "q": "AU", "angles": "radian"},
            "element_origin": "INSTANTANEOUS_COMMON_GM_WEIGHTED_ACTIVE_MODEL_BARYCENTER",
            "retained_cartesian_origin": "INSTANTANEOUS_COMMON_GM_WEIGHTED_ACTIVE_MODEL_BARYCENTER",
            "maximum_allowed_retained_barycenter_residual_position_AU_hex": float(1.0e-12).hex(),
            "maximum_allowed_retained_barycenter_residual_velocity_AU_per_julian_year_hex": float(1.0e-12).hex(),
            "element_mu": "SUM_COMMON_GMS_OF_ALL_ACTIVE_MASSIVE_BODIES_IN_LANE",
            "reduced_mu_note": "EXCLUDES_OMITTED_PLUTO_GM",
            "resolved11_mu_note": "INCLUDES_ALL_RESOLVED11_GMS",
            "preliminary_scattering_metric": "max_t(abs(a(t)-a(0)))/a(0)",
            "preliminary_scattering_threshold_hex": SCATTERING_DIAGNOSTIC_THRESHOLD.hex(),
            "threshold_comparison": "STRICTLY_GREATER_THAN",
            "classification_generated": False,
            "resonance_analysis_present": False,
            "materialized_element_series": "SEMIMAJOR_AXIS_ONLY_PLUS_PRELIMINARY_SCATTERING_SUMMARY",
            "all_elements_rederivable_from_retained_cartesian_payload": True,
            "massive_invariant_sampling": {
                "JX": "EVERY_FIXED_SOURCE_NODE_INCLUDING_ENDPOINTS",
                "IAS15": "ANNUAL_CHECKPOINTS_ONLY",
            },
        },
        "predeclared_gates": {key: value.hex() for key, value in GATE_THRESHOLDS.items()},
        "exact_gate_budgets": {
            "scattering_diagnostic": {"numerator": 3, "denominator": 80, "binary64_hex": float(SCATTERING_THRESHOLD_RATIONAL).hex()},
            "numerical_richardson_tail": {"numerator": 3, "denominator": 8_000, "binary64_hex": float(NUMERICAL_TAIL_BUDGET_RATIONAL).hex()},
            "integrator_comparison": {"numerator": 3, "denominator": 4_000, "binary64_hex": float(INTEGRATOR_BUDGET_RATIONAL).hex()},
            "model_reduction": {"numerator": 3, "denominator": 2_000, "binary64_hex": float(MODEL_REDUCTION_BUDGET_RATIONAL).hex()},
            "ephemeris_sensitivity": {"numerator": 3, "denominator": 2_000, "binary64_hex": float(EPHEMERIS_BUDGET_RATIONAL).hex()},
            "total_scattering_disagreement_chain": {"numerator": 3, "denominator": 800, "nearest_binary64_hex": float(TOTAL_CHAIN_BUDGET_RATIONAL).hex(), "outward_ceiling_binary64_hex": "0x1.eb851eb851eb9p-9"},
            "cartesian_position_AU": {"numerator": 1, "denominator": 500, "binary64_hex": float(CARTESIAN_POSITION_BUDGET_AU_RATIONAL).hex()},
            "cartesian_velocity_AU_per_julian_year": {"numerator": 1, "denominator": 50_000, "binary64_hex": float(CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL).hex()},
        },
        "gate_semantics": {
            "all_96_batch_lanes_complete": True,
            "all_1001_checkpoint_rosters_exact": True,
            "no_nonfinite_or_nonelliptic_tracer_observation": True,
            "no_tracer_drop_removal_or_relabel": True,
            "semimajor_axis_difference": "MAX_OVER_1001_CHECKPOINTS_ABS_A_LEFT_MINUS_A_RIGHT_DIVIDED_BY_MAX_OF_TWO_INITIAL_A_VALUES",
            "cartesian_difference": "MAX_OVER_1001_CHECKPOINTS_EUCLIDEAN_NORM_AFTER_INSTANTANEOUS_MASSIVE_GM_BARYCENTER_CANONICALIZATION",
            "richardson_tail": "EXACT_SEMIMAJOR_DIFFERENCE_DIVIDED_BY_THREE; EXACT_CARTESIAN_SQUARED_DIFFERENCE_DIVIDED_BY_NINE",
            "contraction": "EXACT_SEMIMAJOR_FINE_LE_EXACT_64_ULP_FLOOR_OR_COARSE_GE_3_FINE; EXACT_SQUARED_CARTESIAN_FINE_LE_EXACT_SQUARED_64_ULP_FLOOR_OR_COARSE_SQUARED_GE_9_FINE_SQUARED",
            "reduced_massive_source_convergence": "EACH_OF_FIVE_SOURCES_AT_ALL_1001_ANNUAL_EPOCHS_WITH_THE_SAME_POSITION_VELOCITY_RICHARDSON_AND_CONTRACTION_RULES",
            "fine_jx_massive_energy_contraction": "PER_BATCH_AND_EPHEMERIS_FINE_ALL_NODE_DRIFT_LE_MAX_COARSE_ALL_NODE_DRIFT_DIVIDED_BY_THREE,2_POW_MINUS_46",
            "ulp_floor": "PER_EPOCH_EXACT_64_TIMES_REFERENCE_SCALAR_ULP; VECTOR_FLOORS_ARE_EXACT_SUM_OF_THREE_SQUARED_COMPONENT_FLOORS; MAXIMUM_OVER_THE_SAME_1001_CHECKPOINTS",
            "semimajor_scattering_and_energy_acceptance_uses_exact_binary64_fractions": True,
            "cartesian_acceptance_uses_exact_squared_fraction_without_sqrt": True,
            "semimajor_scattering_energy_and_cartesian_display_values_only_are_outward_rounded_after_decision": True,
            "angular_momentum_acceptance_uses_conservative_outward_binary64_observations": True,
            "scattering_disagreement_chains": "BOTH_FINE_TO_REDUCED_TO_RESOLVED_TO_OTHER_RESOLVED_AND_FINE_TO_REDUCED_TO_OTHER_REDUCED_TO_OTHER_RESOLVED_IN_BOTH_EPHEMERIS_DIRECTIONS",
            "preliminary_scattering_bit_is_not_an_acceptance_gate": True,
            "thresholds_are_data_independent_and_frozen_here": True,
            "failure_means_qualification_not_established_not_scientific_disproof": True,
        },
        "input_bindings": bindings,
        "upstream_content_sha256": dict(EXPECTED_CONTENT_HASHES),
        "implementation": {
            "relative_path": "benchmarks/jx_tno_stage1_dynamics_qualification_v2.py",
            "file_sha256": _sha256(implementation),
            "prepare_imports_dynamics_runtimes": False,
        },
        "de441_validation_interface": {
            "source_relative_path": "tools/jx_tno_de441_resolved11_acquisition_v1.py",
            "source_file_sha256": STATIC_BINDINGS[9][1],
            "artifact_root_relative_path": "runs/tno_de441_resolved11_native_epochs_v1",
            "protocol_schema": "jx.tno.de441-resolved11-native-epochs.protocol.v1",
            "custody_schema": "jx.tno.de441-resolved11-native-epochs.custody.v1",
            "public_offline_validator": "validate_output",
        },
        "runtime_identity_required_in_result": [
            "lexical_project_venv_launcher_pyvenv_cfg_and_resolved_python_hash",
            "fresh_child_import_only_preflight_without_cuda_device_query_or_dynamics",
            "platform",
            "NumPy_distribution_RECORD_module_and_core_extension",
            "JX_source_hashes",
            "JX_native_wisdom_holman_extension_and_runtime_identity",
            "JX_fused_tracer_result_runtime_identity",
            "fresh_restart_worker_exact_module_registry_and_cuda_identity",
            "CuPy_and_CUDA_device_runtime",
            "CuPy_FFT_configuration_defining_module",
            "isolated_disabled_external_kernel_cache_receipt",
            "REBOUND_5.1.1_module_distribution_and_shared_library_hashes",
            "all_exact_input_file_hashes",
        ],
        "retained_evidence_size_projection": {
            "fixed_binary_bytes_excluding_ias15_trace": EXPECTED_FIXED_BINARY_EVIDENCE_BYTES,
            "jx_source_nodes_bytes": EXPECTED_JX_SOURCE_BYTES,
            "jx_restart_streams_bytes": EXPECTED_JX_RESTART_STREAM_BYTES,
            "jx_midpoints_bytes": EXPECTED_JX_MIDPOINT_BYTES,
            "cartesian_arrays_bytes": EXPECTED_CARTESIAN_ARRAY_BYTES,
            "ias15_trace_exact_formula": "SUM_OVER_24_REDUCED_LANES(704*RECORD_COUNT)+SUM_OVER_24_RESOLVED11_LANES(992*RECORD_COUNT)",
            "ias15_trace_validation_hard_maximum_bytes": IAS_TRACE_MAXIMUM_BYTES,
            "maximum_total_binary_bytes_including_ias15_trace": EXPECTED_FIXED_BINARY_EVIDENCE_BYTES + IAS_TRACE_MAXIMUM_BYTES,
            "validation_memory_policy": "MATERIALIZE_ARRAYS_EXACTLY_71495424_BYTES_PLUS_JX_MIDPOINTS_JX_RESTART_STREAMS_AND_JSON; STREAM_JX_SOURCE_NODES_AND_IAS15_TRACE_FROM_ONE_STABLE_FD_EACH; MATERIALIZE_AT_MOST_ONE_JX_LANE_OR_ONE_IAS_RECORD",
        },
        "predeclared_runtime_contract": _predeclared_runtime_contract(),
        "result_artifact_roster": [
            "PROTOCOL.json", "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin",
            "JX_SOURCE_NODES.bin", "IAS15_ACCEPTED_STEPS.bin", "ARRAYS.bin",
            "RESULT.json",
        ],
        "publication": {
            "create_only": True,
            "preparation_failure_never_unlinks_created_names": True,
            "payloads_staged_and_file_fsynced_before_publication": True,
            "publication_uses_no_replace_hard_links": True,
            "result_json_committed_last": True,
            "output_directory_fsynced_after_commit": True,
            "publication_failure_never_unlinks_published_destination_names": True,
            "partial_or_concurrently_replaced_roster_is_fail_closed_and_requires_manual_cleanup": True,
            "crash_recovery_is_manual_cleanup_of_an_invalid_incomplete_roster": True,
            "multi_file_filesystem_atomicity_claimed": False,
            "validity_requires_exact_complete_artifact_roster": True,
        },
        "claims": dict(PROTOCOL_CLAIMS),
        "execution_is_explicit_only": True,
        "outcomes_generated": False,
    }
    protocol["content_sha256"] = _content_sha256(PROTOCOL_DOMAIN, protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    expected = build_protocol()
    if type(value) is not dict or _canonical_json(value) != _canonical_json(expected):
        raise Stage1QualificationError("Stage-1 protocol differs from exact reconstruction")
    return str(expected["content_sha256"])


def _write_create_only(path: Path, payload: bytes) -> None:
    path = Path(os.path.abspath(os.fspath(path)))
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise Stage1QualificationError(f"create-only publication failed: {path}") from exc


def _fsync_directory(path: Path) -> None:
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise Stage1QualificationError(f"directory fsync failed: {path}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _copy_create_only(
    source: Path,
    destination: Path,
    label: str,
    *,
    maximum: int = DEFAULT_STAGED_FILE_MAXIMUM_BYTES,
) -> None:
    """Stream a bounded regular file into a create-only, fsynced destination."""

    source = Path(os.path.abspath(os.fspath(source)))
    destination = Path(os.path.abspath(os.fspath(destination)))
    source_fd: int | None = None
    destination_fd: int | None = None
    if type(maximum) is not int or maximum <= 0:
        raise Stage1QualificationError(f"{label} staging maximum changed")
    try:
        source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > maximum
        ):
            raise Stage1QualificationError(f"{label} source identity changed")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        remaining = before.st_size
        while remaining:
            block = os.read(source_fd, min(8 * 1024 * 1024, remaining))
            if not block:
                raise Stage1QualificationError(f"{label} source was truncated")
            view = memoryview(block)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise Stage1QualificationError(f"{label} staging write failed")
                view = view[written:]
            remaining -= len(block)
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if (
            after.st_dev != before.st_dev
            or after.st_ino != before.st_ino
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise Stage1QualificationError(f"{label} source changed while staged")
    except OSError as exc:
        raise Stage1QualificationError(f"cannot stage {label}") from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)


def _publish_result_artifacts(
    root: Path, artifacts: Sequence[tuple[str, bytes | Path]]
) -> None:
    """Stage and durably publish without replacement, committing RESULT last.

    This is durable create-only publication, not an indivisible multi-file
    transaction. Validation requires the complete exact roster including
    RESULT.json; failures never unlink destination names and can therefore
    leave a fail-closed partial roster requiring manual cleanup.
    """

    root = Path(os.path.abspath(os.fspath(root)))
    expected_names = (
        "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin", "JX_SOURCE_NODES.bin",
        "IAS15_ACCEPTED_STEPS.bin", "ARRAYS.bin", "RESULT.json",
    )
    if (
        not root.is_dir()
        or root.is_symlink()
        or {entry.name for entry in root.iterdir()} != {"PROTOCOL.json"}
        or type(artifacts) not in (list, tuple)
        or tuple(name for name, _payload in artifacts) != expected_names
        or any(
            (type(payload) is bytes and not payload)
            or (type(payload) is not bytes and not isinstance(payload, Path))
            for _name, payload in artifacts
        )
    ):
        raise Stage1QualificationError("staged publication precondition changed")
    stage = Path(tempfile.mkdtemp(prefix=".stage1-v2-publish-", dir=root))
    staged: list[Path] = []
    try:
        for name, payload in artifacts:
            path = stage / name
            if type(payload) is bytes:
                _write_create_only(path, payload)
            else:
                _copy_create_only(
                    payload,
                    path,
                    f"staged {name}",
                    maximum=(
                        IAS_TRACE_MAXIMUM_BYTES
                        if name == "IAS15_ACCEPTED_STEPS.bin"
                        else DEFAULT_STAGED_FILE_MAXIMUM_BYTES
                    ),
                )
            staged.append(path)
        _fsync_directory(stage)
        for path in staged:
            destination = root / path.name
            try:
                os.link(path, destination, follow_symlinks=False)
            except OSError as exc:
                raise Stage1QualificationError(
                    f"no-replace publication link failed: {destination}"
                ) from exc
            identity = destination.lstat()
            staged_identity = path.lstat()
            if (
                not stat.S_ISREG(identity.st_mode)
                or identity.st_dev != staged_identity.st_dev
                or identity.st_ino != staged_identity.st_ino
            ):
                raise Stage1QualificationError(
                    f"published hard-link identity changed: {destination}"
                )
        _fsync_directory(root)
    except Exception:
        # POSIX has no compare-inode-and-unlink primitive.  Never unlink a
        # destination name on rollback: a concurrent owner could replace it
        # between observation and unlink.  Any partial roster is invalid and
        # deliberately requires manual cleanup before a new create-only run.
        try:
            _fsync_directory(root)
        except Stage1QualificationError:
            pass
        raise
    finally:
        for path in staged:
            try:
                path.unlink()
            except OSError:
                pass
        try:
            stage.rmdir()
        except OSError:
            pass
    if {entry.name for entry in root.iterdir()} != {
        "PROTOCOL.json", *expected_names
    }:
        raise Stage1QualificationError("staged publication final roster changed")


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if root.exists() or root.is_symlink():
        raise Stage1QualificationError("output root must be absent for create-only preparation")
    protocol = build_protocol()
    try:
        root.mkdir(mode=0o700)
        _write_create_only(root / "PROTOCOL.json", _canonical_json(protocol))
        return protocol
    except Exception:
        # Never unlink by path during rollback: another actor could replace
        # the just-created name between any identity check and unlink.  A
        # failed preparation therefore leaves a fail-closed partial root for
        # explicit/manual cleanup.
        raise


def validate_prepared(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.is_dir() or root.is_symlink() or {entry.name for entry in root.iterdir()} != {"PROTOCOL.json"}:
        raise Stage1QualificationError("prepared output root roster differs")
    payload, protocol = _load_json(root / "PROTOCOL.json", "Stage-1 protocol")
    expected = build_protocol()
    if payload != _canonical_json(expected):
        raise Stage1QualificationError("retained Stage-1 protocol differs")
    return protocol


def _source_only_de441_validator() -> Callable[[Path], dict[str, Any]]:
    payload = _regular_bytes(DE441_TOOL_PATH, "DE441 acquisition source")
    if _sha256(payload) != "84df09da90c16d49cc245d72512290962639390ab9e0966a5cedd07f91b68e00":
        raise Stage1QualificationError("DE441 acquisition source changed")
    name = "_jx_tno_de441_acquisition_stage1_v2_bound"
    module = types.ModuleType(name)
    module.__file__ = str(DE441_TOOL_PATH)
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(payload, str(DE441_TOOL_PATH), "exec", dont_inherit=True, optimize=0), module.__dict__)
    finally:
        sys.modules.pop(name, None)
    validator = getattr(module, "validate_output", None)
    if not callable(validator):
        raise Stage1QualificationError("DE441 acquisition validator interface changed")
    return validator


def validate_execution_prerequisites(de441_root: Path = DE441_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(de441_root)))
    if root != Path(os.path.abspath(os.fspath(DE441_ROOT))):
        raise Stage1QualificationError("DE441 artifact root must be the pinned project path")
    if not root.is_dir() or root.is_symlink():
        raise Stage1QualificationError("complete retained DE441 artifact is required before runtime import")
    validator = _source_only_de441_validator()
    try:
        custody = validator(root)
    except Exception as exc:
        raise Stage1QualificationError("full DE441 acquisition reconstruction failed") from exc
    if (
        type(custody) is not dict
        or custody.get("content_sha256") != EXPECTED_CONTENT_HASHES["de441_custody"]
        or custody.get("state_roster_sha256") != "f57f04d68909312d5fe53918eb8c7527310282034d678690b2f99619cafa3cd2"
    ):
        raise Stage1QualificationError("validated DE441 custody identity changed")
    if _sha256(_regular_bytes(DE441_TOOL_PATH, "DE441 acquisition source post-validation")) != STATIC_BINDINGS[9][1]:
        raise Stage1QualificationError("DE441 acquisition source changed during validation")
    return custody


def _gm_si_to_au_year(value: float) -> float:
    return value * (SECONDS_PER_JULIAN_YEAR_EXACT**2) / (AU_METRES_EXACT**3)


def _weighted_centroid(states: Sequence[MassiveState]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if not states:
        raise Stage1QualificationError("centroid requires massive states")
    total = math.fsum(item.gm_AU3_per_julian_year2 for item in states)
    if not total > 0.0:
        raise Stage1QualificationError("centroid total GM must be positive")
    position = tuple(math.fsum(item.gm_AU3_per_julian_year2 * item.position_AU[k] for item in states) / total for k in range(3))
    velocity = tuple(math.fsum(item.gm_AU3_per_julian_year2 * item.velocity_AU_per_julian_year[k] for item in states) / total for k in range(3))
    return position, velocity  # type: ignore[return-value]


def _common_gms(bridge: Mapping[str, Any]) -> tuple[float, ...]:
    rows = bridge["batches"][0]["massive_bodies"]
    values = tuple(_gm_si_to_au_year(_exact_float_hex(row["gravitational_parameter_m3_per_s2_hex"], "GM")) for row in rows)
    for batch in bridge["batches"][1:]:
        observed = tuple(_gm_si_to_au_year(_exact_float_hex(row["gravitational_parameter_m3_per_s2_hex"], "GM")) for row in batch["massive_bodies"])
        if tuple(item.hex() for item in observed) != tuple(item.hex() for item in values):
            raise Stage1QualificationError("common GM roster differs between batches")
    return values


def _batch_states(bridge: Mapping[str, Any], custody: Mapping[str, Any], ephemeris: str) -> tuple[BatchState, ...]:
    if ephemeris not in ("DE440S", "DE441"):
        raise Stage1QualificationError("unknown ephemeris")
    gms = _common_gms(bridge)
    custody_rows = custody["state_records"]
    result: list[BatchState] = []
    for batch_order, batch in enumerate(bridge["batches"]):
        object_id = batch["object_id"]
        epoch = _exact_float_hex(batch["covariance_epoch_jd_tdb_hex"], "native epoch")
        if ephemeris == "DE440S":
            massive = tuple(
                MassiveState(
                    row["naif_id"],
                    row["body_id"],
                    gms[index],
                    tuple(_exact_float_hex(item, "DE440 position") / AU_METRES_EXACT for item in row["position_m_hex"]),
                    tuple(_exact_float_hex(item, "DE440 velocity") * SECONDS_PER_JULIAN_YEAR_EXACT / AU_METRES_EXACT for item in row["velocity_m_per_s_hex"]),
                )
                for index, row in enumerate(batch["massive_bodies"])
            )
            tracers = tuple(
                TracerState(
                    row["object_id"], row["spk_id"], row["sample_kind"], row["clone_index"],
                    tuple(_exact_float_hex(item, "DE440 TNO position") / AU_METRES_EXACT for item in row["resolved11_barycenter_position_m_hex"]),
                    tuple(_exact_float_hex(item, "DE440 TNO velocity") * SECONDS_PER_JULIAN_YEAR_EXACT / AU_METRES_EXACT for item in row["resolved11_barycenter_velocity_m_per_s_hex"]),
                )
                for row in batch["tnos"]
            )
        else:
            rows = custody_rows[batch_order * 11 : (batch_order + 1) * 11]
            ssb_massive = tuple(
                MassiveState(
                    row["naif_id"], row["semantic_id"], gms[index],
                    tuple(_exact_float_hex(item, "DE441 position") for item in row["position_AU_binary64_hex"]),
                    tuple(_exact_float_hex(item, "DE441 velocity") * 365.25 for item in row["velocity_AU_per_day_binary64_hex"]),
                )
                for index, row in enumerate(rows)
            )
            center_position, center_velocity = _weighted_centroid(ssb_massive)
            massive = tuple(
                MassiveState(item.naif_id, item.body_id, item.gm_AU3_per_julian_year2, _sub(item.position_AU, center_position), _sub(item.velocity_AU_per_julian_year, center_velocity))
                for item in ssb_massive
            )
            sun_ssb = ssb_massive[0]
            tracers = tuple(
                TracerState(
                    row["object_id"], row["spk_id"], row["sample_kind"], row["clone_index"],
                    _sub(_add(tuple(_exact_float_hex(item, "heliocentric J2000 position") for item in row["heliocentric_j2000_position_AU_hex"]), sun_ssb.position_AU), center_position),
                    _sub(_add(tuple(_exact_float_hex(item, "heliocentric J2000 velocity") for item in row["heliocentric_j2000_velocity_AU_per_julian_year_hex"]), sun_ssb.velocity_AU_per_julian_year), center_velocity),
                )
                for row in batch["tnos"]
            )
        result.append(BatchState(ephemeris, batch_order, object_id, batch["tnos"][0]["spk_id"], epoch, massive, tracers))
    return tuple(result)


def collapse_reduced_model(batch: BatchState) -> BatchState:
    if tuple(item.naif_id for item in batch.massive) != RESOLVED11_NAIF_IDS:
        raise Stage1QualificationError("resolved-11 massive roster changed")
    inner = batch.massive[:6]
    outer = batch.massive[6:10]
    sun = inner[0]
    central = MassiveState(
        10,
        REDUCED_BODY_IDS[0],
        math.fsum(item.gm_AU3_per_julian_year2 for item in inner),
        sun.position_AU,
        sun.velocity_AU_per_julian_year,
    )
    active = (central, *outer)
    center_position, center_velocity = _weighted_centroid(active)
    recentered = tuple(
        MassiveState(item.naif_id, item.body_id, item.gm_AU3_per_julian_year2, _sub(item.position_AU, center_position), _sub(item.velocity_AU_per_julian_year, center_velocity))
        for item in active
    )
    tracers = tuple(
        TracerState(item.object_id, item.spk_id, item.sample_kind, item.clone_index, _sub(item.position_AU, center_position), _sub(item.velocity_AU_per_julian_year, center_velocity))
        for item in batch.tracers
    )
    return BatchState(batch.ephemeris, batch.batch_order, batch.object_id, batch.spk_id, batch.epoch_jd_tdb, recentered, tracers)


def _elements_record(position: Sequence[float], velocity: Sequence[float], mu: float) -> dict[str, str]:
    elements = extract_elliptic_orbital_elements(position, velocity, mu)
    return {name: value.hex() for name, value in elements.items()}


def _massive_invariants_centered(
    massive: Sequence[MassiveState],
) -> tuple[float, tuple[float, float, float]]:
    kinetic = math.fsum(0.5 * item.gm_AU3_per_julian_year2 * _dot(item.velocity_AU_per_julian_year, item.velocity_AU_per_julian_year) for item in massive)
    potential_terms: list[float] = []
    for i, left in enumerate(massive):
        for right in massive[i + 1 :]:
            distance = _norm(_sub(left.position_AU, right.position_AU))
            if distance == 0.0:
                raise Stage1QualificationError("massive bodies became coincident")
            potential_terms.append(-left.gm_AU3_per_julian_year2 * right.gm_AU3_per_julian_year2 / distance)
    angular = [0.0, 0.0, 0.0]
    for item in massive:
        cross = _cross(item.position_AU, item.velocity_AU_per_julian_year)
        for component in range(3):
            angular[component] += item.gm_AU3_per_julian_year2 * cross[component]
    return kinetic + math.fsum(potential_terms), (angular[0], angular[1], angular[2])


def _maximum_exact_relative_energy_drift(values: Sequence[float]) -> Fraction:
    if type(values) not in (list, tuple) or not values:
        raise Stage1QualificationError("energy series must be a nonempty sequence")
    if any(type(item) is not float or not math.isfinite(item) for item in values):
        raise Stage1QualificationError("energy series must contain finite built-in floats")
    initial = Fraction.from_float(values[0])
    denominator = abs(initial)
    if denominator == 0:
        raise Stage1QualificationError("massive energy denominator is zero")
    return max(abs(Fraction.from_float(item) - initial) / denominator for item in values)


def _jx_source_contract(full: Any) -> dict[str, object]:
    radii = tuple(float(item) for item in full.source_encounter_radii_AU)
    bounds = tuple(
        float(item) for item in full.certified_relative_acceleration_bounds_AU_per_year2
    )
    if (
        tuple(str(item) for item in full.source_ids) != REDUCED_BODY_IDS
        or str(full.primary_source_id) != REDUCED_BODY_IDS[0]
        or any(item != 1.0 for item in radii)
        or len(set(bounds)) != 1
    ):
        raise Stage1QualificationError("JX source tile roster/guard arrays changed")
    provenance = full.provenance
    return {
        "tile_id": str(full.tile_id),
        "epoch_origin_years_hex": float(full.epoch_origin_years).hex(),
        "global_source_step": int(full.global_source_step),
        "source_step_years_hex": float(full.source_step_years).hex(),
        "time_scale": str(full.time_scale),
        "frame": str(full.frame),
        "axes": str(full.axes),
        "coordinate_system": str(full.coordinate_system),
        "length_unit": str(full.length_unit),
        "time_unit": str(full.time_unit),
        "unit_system_id": str(full.unit_system_id),
        "source_ids": list(full.source_ids),
        "primary_source_id": str(full.primary_source_id),
        "source_schedule_sha256": str(full.source_schedule_sha256),
        "provenance": {
            "citation": str(provenance.citation),
            "sha256": str(provenance.sha256),
            "source_id": str(provenance.source_id),
            "version": str(provenance.version),
        },
        "node_count": int(full.node_count),
        "source_count": len(full.source_ids),
        "encounter_radii_AU_hex": [item.hex() for item in radii],
        "certified_relative_acceleration_bounds_AU_per_year2_hex": [
            item.hex() for item in bounds
        ],
    }


def _pack_jx_source_node_blocks(full: Any) -> bytes:
    import numpy as np

    position = np.ascontiguousarray(
        full.source_relative_positions_AU, dtype=np.dtype("<f8")
    )
    velocity = np.ascontiguousarray(
        full.source_relative_velocities_AU_per_year, dtype=np.dtype("<f8")
    )
    if position.shape != velocity.shape or position.shape[1:] != (JX_SOURCE_COUNT, 3):
        raise Stage1QualificationError("JX source-node evidence shape changed")
    payload = position.tobytes(order="C") + velocity.tobytes(order="C")
    expected = position.shape[0] * JX_SOURCE_COUNT * 3 * 2 * 8
    if len(payload) != expected:
        raise Stage1QualificationError("JX source-node evidence length changed")
    return payload


def _jx_tile_metadata_bytes(
    contract: Mapping[str, Any], *, tile_id: str, global_source_step: int
) -> bytes:
    metadata = {
        "axes": contract["axes"],
        "coordinate_system": contract["coordinate_system"],
        "epoch_origin_years_hex": contract["epoch_origin_years_hex"],
        "frame": contract["frame"],
        "global_source_step": global_source_step,
        "length_unit": contract["length_unit"],
        "primary_source_id": contract["primary_source_id"],
        "provenance": contract["provenance"],
        "source_ids": contract["source_ids"],
        "source_schedule_sha256": contract["source_schedule_sha256"],
        "source_step_years_hex": contract["source_step_years_hex"],
        "tile_id": tile_id,
        "time_scale": contract["time_scale"],
        "time_unit": contract["time_unit"],
        "unit_system_id": contract["unit_system_id"],
    }
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _jx_tile_sha256_from_raw(
    contract: Mapping[str, Any],
    positions: bytes,
    velocities: bytes,
    gm_payload: bytes,
    *,
    node_count: int,
    tile_id: str,
    global_source_step: int,
) -> str:
    if (
        type(positions) is not bytes
        or type(velocities) is not bytes
        or type(gm_payload) is not bytes
        or type(node_count) is not int
        or node_count < 2
        or len(positions) != node_count * JX_SOURCE_COUNT * 3 * 8
        or len(velocities) != len(positions)
        or len(gm_payload) != JX_SOURCE_GM_BYTES
    ):
        raise Stage1QualificationError("JX source tile raw evidence is invalid")
    radii = tuple(
        _exact_float_hex(item, "JX encounter radius")
        for item in contract.get("encounter_radii_AU_hex", [])
    )
    bounds = tuple(
        _exact_float_hex(item, "JX acceleration bound")
        for item in contract.get(
            "certified_relative_acceleration_bounds_AU_per_year2_hex", []
        )
    )
    if len(radii) != JX_SOURCE_COUNT or len(bounds) != JX_SOURCE_COUNT:
        raise Stage1QualificationError("JX source guard roster changed")
    radii_payload = struct.pack(f"<{JX_SOURCE_COUNT}d", *radii)
    bounds_payload = struct.pack(f"<{JX_SOURCE_COUNT}d", *bounds)
    digest = hashlib.sha256()
    digest.update(JX_TILE_DOMAIN)
    metadata = _jx_tile_metadata_bytes(
        contract, tile_id=tile_id, global_source_step=global_source_step
    )
    digest.update(len(metadata).to_bytes(8, "little"))
    digest.update(metadata)
    arrays = (
        ("source_relative_positions_AU", 3, (node_count, JX_SOURCE_COUNT, 3), positions),
        ("source_relative_velocities_AU_per_year", 3, (node_count, JX_SOURCE_COUNT, 3), velocities),
        ("source_gm_AU3_per_year2", 1, (JX_SOURCE_COUNT,), gm_payload),
        ("source_encounter_radii_AU", 1, (JX_SOURCE_COUNT,), radii_payload),
        (
            "certified_relative_acceleration_bounds_AU_per_year2",
            1,
            (JX_SOURCE_COUNT,),
            bounds_payload,
        ),
    )
    for label, ndim, shape, raw in arrays:
        encoded = label.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        digest.update(ndim.to_bytes(4, "little"))
        for dimension in shape:
            digest.update(int(dimension).to_bytes(8, "little"))
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_jx_source_contract(value: object, steps_per_year: int) -> None:
    expected_keys = {
        "tile_id", "epoch_origin_years_hex", "global_source_step",
        "source_step_years_hex", "time_scale", "frame", "axes",
        "coordinate_system", "length_unit", "time_unit", "unit_system_id",
        "source_ids", "primary_source_id", "source_schedule_sha256",
        "provenance", "node_count", "source_count", "encounter_radii_AU_hex",
        "certified_relative_acceleration_bounds_AU_per_year2_hex",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("global_source_step") != 0
        or value.get("source_step_years_hex") != (1.0 / steps_per_year).hex()
        or value.get("node_count") != HORIZON_JULIAN_YEARS * steps_per_year + 1
        or value.get("source_count") != JX_SOURCE_COUNT
        or value.get("source_ids") != list(REDUCED_BODY_IDS)
        or value.get("primary_source_id") != REDUCED_BODY_IDS[0]
        or value.get("time_scale") != JX_TIME_SCALE
        or value.get("frame") != "ICRF_J2000"
        or value.get("axes") != "J2000_EQUATORIAL"
        or value.get("coordinate_system") != "PRIMARY_RELATIVE_NONROTATING"
        or value.get("length_unit") != "AU"
        or value.get("time_unit") != "year"
        or value.get("unit_system_id") != "AU-YEAR"
        or not _is_sha256(value.get("source_schedule_sha256"))
        or type(value.get("tile_id")) is not str
        or type(value.get("provenance")) is not dict
    ):
        raise Stage1QualificationError("JX source tile contract changed")
    radii = [
        _exact_float_hex(item, "JX source encounter radius")
        for item in value["encounter_radii_AU_hex"]
    ]
    bounds = [
        _exact_float_hex(item, "JX source acceleration bound")
        for item in value["certified_relative_acceleration_bounds_AU_per_year2_hex"]
    ]
    if len(radii) != JX_SOURCE_COUNT or any(item != 1.0 for item in radii):
        raise Stage1QualificationError("JX source encounter radii changed")
    if len(bounds) != JX_SOURCE_COUNT or len(set(bounds)) != 1 or not bounds[0] > 0.0:
        raise Stage1QualificationError("JX source acceleration bounds changed")


def _validate_jx_source_evidence(
    handle: _StableReadHandle,
    metadata: object,
    reconstructed_lanes: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
    numpy_module: types.ModuleType,
) -> None:
    _guard_exact_json_integer_identities(metadata, "JX source evidence metadata")
    """Stream-validate every retained JX source node and derived gate input."""

    if (
        type(numpy_module) is not types.ModuleType
        or numpy_module.__name__ != "numpy"
        or sys.modules.get("numpy") is not numpy_module
    ):
        raise Stage1QualificationError(
            "JX source replay did not receive the bound offline NumPy module"
        )
    if type(metadata) is not dict or set(metadata) != {
        "schema", "relative_path", "encoding", "byte_length", "file_sha256",
        "content_sha256", "common_gm_byte_offset", "common_gm_byte_length",
        "common_gm_sha256", "lane_count", "order", "records",
    }:
        raise Stage1QualificationError("JX source evidence metadata shape changed")
    digest = _stream_handle_digests(
        handle, content_domain=JX_SOURCE_DOMAIN,
    )
    if (
        metadata.get("schema") != JX_SOURCE_SCHEMA
        or metadata.get("relative_path") != "JX_SOURCE_NODES.bin"
        or metadata.get("encoding") != "RAW_LITTLE_ENDIAN_IEEE754_BINARY64"
        or metadata.get("byte_length") != EXPECTED_JX_SOURCE_BYTES
        or digest["bytes"] != EXPECTED_JX_SOURCE_BYTES
        or metadata.get("file_sha256") != digest["sha256"]
        or metadata.get("content_sha256") != digest["content_sha256"]
        or metadata.get("common_gm_byte_offset") != 0
        or metadata.get("common_gm_byte_length") != JX_SOURCE_GM_BYTES
        or metadata.get("lane_count") != JX_SOURCE_RECORD_COUNT
        or metadata.get("order")
        != "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_16_32_POSITIONS_THEN_VELOCITIES"
    ):
        raise Stage1QualificationError("JX source evidence identity changed")
    gm_payload = _read_exact_handle_region(
        handle, 0, JX_SOURCE_GM_BYTES, "JX common source GM"
    )
    if metadata.get("common_gm_sha256") != _sha256(gm_payload):
        raise Stage1QualificationError("JX common source GM digest changed")
    gms = struct.unpack(f"<{JX_SOURCE_COUNT}d", gm_payload)
    batches = {
        ephemeris: _batch_states(bridge, custody, ephemeris)
        for ephemeris in ("DE440S", "DE441")
    }
    expected_gms = tuple(
        item.gm_AU3_per_julian_year2
        for item in collapse_reduced_model(batches["DE440S"][0]).massive
    )
    if struct.pack(f"<{JX_SOURCE_COUNT}d", *expected_gms) != gm_payload:
        raise Stage1QualificationError("JX source GM bytes differ from sealed inputs")
    lane_index = {
        (
            record.get("ephemeris"), record.get("batch_order"),
            record.get("lane", {}).get("lane_id")
            if type(record.get("lane")) is dict else None,
        ): record
        for record in reconstructed_lanes
        if type(record) is dict
        and type(record.get("lane")) is dict
        and record["lane"].get("engine") == "JX_RESTRICTED_WH"
    }
    records = metadata.get("records")
    expected_order = [
        (ephemeris, batch_order, steps)
        for ephemeris in ("DE440S", "DE441")
        for batch_order in range(OBJECT_COUNT)
        for steps in JX_STEPS_PER_YEAR
    ]
    if type(records) is not list or len(records) != len(expected_order):
        raise Stage1QualificationError("JX source record roster changed")
    expected_offset = JX_SOURCE_GM_BYTES
    for source_order, (record, expected_identity) in enumerate(
        zip(records, expected_order, strict=True)
    ):
        ephemeris, batch_order, steps = expected_identity
        lane_id = f"jx_reduced5_h_1_{steps}"
        lane_wrapper = lane_index.get((ephemeris, batch_order, lane_id))
        if type(record) is not dict or type(lane_wrapper) is not dict:
            raise Stage1QualificationError("JX source lane identity is missing")
        lane = lane_wrapper["lane"]
        contract = record.get("contract")
        node_count = HORIZON_JULIAN_YEARS * steps + 1
        block_bytes = node_count * JX_SOURCE_COUNT * 3 * 8
        expected_keys = {
            "source_order", "ephemeris", "batch_order", "object_id",
            "lane_id", "steps_per_year", "node_count", "source_count",
            "byte_offset", "byte_length", "positions_block_bytes",
            "velocities_block_bytes", "lane_payload_sha256",
            "positions_sha256", "velocities_sha256", "contract",
            "full_source_tile_sha256", "annual_source_slice_sha256",
            "record_content_sha256",
        }
        batch = batches[ephemeris][batch_order]
        if (
            set(record) != expected_keys
            or record.get("source_order") != source_order
            or record.get("ephemeris") != ephemeris
            or record.get("batch_order") != batch_order
            or record.get("object_id") != batch.object_id
            or record.get("lane_id") != lane_id
            or record.get("steps_per_year") != steps
            or record.get("node_count") != node_count
            or record.get("source_count") != JX_SOURCE_COUNT
            or record.get("byte_offset") != expected_offset
            or record.get("byte_length") != 2 * block_bytes
            or record.get("positions_block_bytes") != block_bytes
            or record.get("velocities_block_bytes") != block_bytes
        ):
            raise Stage1QualificationError("JX source record layout changed")
        _validate_jx_source_contract(contract, steps)
        reduced = collapse_reduced_model(batch)
        expected_tile_id = (
            f"{JX_TILE_ID_PREFIX}-{ephemeris}-{batch.object_id}-h1_{steps}"
        )
        expected_schedule = _source_schedule_sha256(reduced, 1.0 / steps)
        expected_epoch = ((batch.epoch_jd_tdb - 2_451_545.0) / 365.25).hex()
        expected_bound = math.nextafter(2.0 * math.fsum(gms), math.inf)
        if (
            contract.get("tile_id") != expected_tile_id
            or contract.get("epoch_origin_years_hex") != expected_epoch
            or contract.get("source_schedule_sha256") != expected_schedule
            or contract.get("provenance") != {
                "citation": "Retained Stage-1 v2 TNO native-epoch bridge and common GM contract",
                "sha256": expected_schedule,
                "source_id": (
                    f"{JX_PROVENANCE_ID_PREFIX}-{ephemeris}-"
                    f"{batch.object_id}-{steps}"
                ),
                "version": JX_PROVENANCE_VERSION,
            }
            or contract.get("certified_relative_acceleration_bounds_AU_per_year2_hex")
            != [expected_bound.hex()] * JX_SOURCE_COUNT
        ):
            raise Stage1QualificationError("JX source contract is not bound to its batch")
        payload = _read_exact_handle_region(
            handle, expected_offset, 2 * block_bytes, "JX source lane"
        )
        positions = payload[:block_bytes]
        velocities = payload[block_bytes:]
        record_without_content = dict(record)
        content = record_without_content.pop("record_content_sha256", None)
        if (
            record.get("lane_payload_sha256") != _sha256(payload)
            or record.get("positions_sha256") != _sha256(positions)
            or record.get("velocities_sha256") != _sha256(velocities)
            or content
            != _sha256(
                JX_SOURCE_DOMAIN
                + _canonical_json(record_without_content)
                + payload
            )
            or record.get("full_source_tile_sha256")
            != _jx_tile_sha256_from_raw(
                contract, positions, velocities, gm_payload,
                node_count=node_count, tile_id=expected_tile_id,
                global_source_step=0,
            )
            or record.get("full_source_tile_sha256")
            != lane.get("full_source_tile_sha256")
        ):
            raise Stage1QualificationError("JX full source tile did not reconstruct")
        lane_ref = lane.get("source_node_evidence")
        if lane_ref != {
            "source_order": source_order,
            "byte_offset": expected_offset,
            "byte_length": 2 * block_bytes,
            "lane_payload_sha256": record["lane_payload_sha256"],
            "record_content_sha256": record["record_content_sha256"],
        }:
            raise Stage1QualificationError("JX lane/source evidence cross-binding changed")
        annual_digests: list[str] = []
        source_stride = JX_SOURCE_COUNT * 3 * 8
        for year in range(HORIZON_JULIAN_YEARS):
            start, stop = annual_source_node_bounds(year, steps)
            position_slice = positions[start * source_stride : stop * source_stride]
            velocity_slice = velocities[start * source_stride : stop * source_stride]
            annual_digests.append(
                _jx_tile_sha256_from_raw(
                    contract, position_slice, velocity_slice, gm_payload,
                    node_count=steps + 1,
                    tile_id=f"{expected_tile_id}.annual.{year:04d}",
                    global_source_step=start,
                )
            )
        if (
            record.get("annual_source_slice_sha256") != annual_digests
            or lane.get("annual_source_slice_sha256") != annual_digests
        ):
            raise Stage1QualificationError("JX annual source tile digest changed")
        # NumPy is an exact-bound offline decoder only; CuPy/JX/REBOUND are
        # not imported by validation.
        position_array = numpy_module.frombuffer(
            positions, dtype="<f8"
        ).reshape(
            node_count, JX_SOURCE_COUNT, 3
        )
        velocity_array = numpy_module.frombuffer(
            velocities, dtype="<f8"
        ).reshape(
            node_count, JX_SOURCE_COUNT, 3
        )
        invariant = _source_node_invariant_summary(
            position_array, velocity_array, gms,
            numpy_module=numpy_module,
        )
        summary = lane.get("summary")
        _require_source_invariant_summary(summary, invariant)
        checkpoints = lane.get("checkpoints")
        if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT:
            raise Stage1QualificationError("JX annual checkpoint roster changed")
        for year, checkpoint in enumerate(checkpoints):
            node = year * steps
            raw_massive = tuple(
                MassiveState(
                    reduced.massive[index].naif_id,
                    reduced.massive[index].body_id,
                    gms[index],
                    tuple(float(value) for value in position_array[node, index]),
                    tuple(float(value) for value in velocity_array[node, index]),
                )
                for index in range(JX_SOURCE_COUNT)
            )
            center_position, center_velocity = _weighted_centroid(raw_massive)
            centered_position = [
                list(_sub(item.position_AU, center_position)) for item in raw_massive
            ]
            centered_velocity = [
                list(_sub(item.velocity_AU_per_julian_year, center_velocity))
                for item in raw_massive
            ]
            if (
                checkpoint.get("massive_positions_AU") != centered_position
                or checkpoint.get("massive_velocities_AU_per_julian_year")
                != centered_velocity
            ):
                raise Stage1QualificationError(
                    "JX annual source node differs from Cartesian arrays"
                )
        expected_offset += 2 * block_bytes
    if expected_offset != EXPECTED_JX_SOURCE_BYTES:
        raise Stage1QualificationError("JX source evidence has trailing bytes")


def _source_node_invariant_summary(
    positions: Any,
    velocities: Any,
    gm_values: Any,
    *,
    numpy_module: types.ModuleType | None = None,
) -> dict[str, object]:
    """Conservatively summarize E/L over every retained JX source node."""

    if numpy_module is None:
        import numpy as np
    else:
        if (
            type(numpy_module) is not types.ModuleType
            or numpy_module.__name__ != "numpy"
            or sys.modules.get("numpy") is not numpy_module
        ):
            raise Stage1QualificationError(
                "source invariant replay NumPy binding changed"
            )
        np = numpy_module

    position = np.asarray(positions, dtype=np.float64)
    velocity = np.asarray(velocities, dtype=np.float64)
    gm = np.asarray(gm_values, dtype=np.float64)
    if (
        position.ndim != 3
        or velocity.shape != position.shape
        or position.shape[2] != 3
        or gm.shape != (position.shape[1],)
        or position.shape[0] < 2
        or not np.isfinite(position).all()
        or not np.isfinite(velocity).all()
        or not np.isfinite(gm).all()
        or not np.all(gm > 0.0)
    ):
        raise Stage1QualificationError("JX source-node invariant array shape changed")
    total_gm = float(np.sum(gm, dtype=np.float64))
    center_position = np.sum(position * gm[None, :, None], axis=1) / total_gm
    center_velocity = np.sum(velocity * gm[None, :, None], axis=1) / total_gm
    centered_position = position - center_position[:, None, :]
    centered_velocity = velocity - center_velocity[:, None, :]
    kinetic = np.sum(
        0.5 * gm[None, :] * np.sum(centered_velocity * centered_velocity, axis=2),
        axis=1,
    )
    potential = np.zeros(position.shape[0], dtype=np.float64)
    for left in range(position.shape[1]):
        for right in range(left + 1, position.shape[1]):
            distance = np.linalg.norm(
                centered_position[:, left, :] - centered_position[:, right, :], axis=1
            )
            if not np.isfinite(distance).all() or np.any(distance <= 0.0):
                raise Stage1QualificationError("JX source nodes contain coincident bodies")
            potential -= gm[left] * gm[right] / distance
    energy = kinetic + potential
    angular = np.sum(
        gm[None, :, None]
        * np.cross(centered_position, centered_velocity, axis=2),
        axis=1,
    )
    if not np.isfinite(energy).all() or not np.isfinite(angular).all():
        raise Stage1QualificationError("JX source-node invariant is nonfinite")
    initial_energy = float(energy[0])
    initial_angular = tuple(float(item) for item in angular[0])
    angular_norm = math.hypot(*initial_angular)
    angular_denominator = math.nextafter(angular_norm, -math.inf)
    if initial_energy == 0.0 or not angular_denominator > 0.0:
        raise Stage1QualificationError("JX source-node invariant denominator is zero")
    maximum_energy_fraction = _maximum_exact_relative_energy_drift(
        tuple(float(item) for item in energy)
    )
    maximum_energy = _fraction_to_outward_float(maximum_energy_fraction)
    maximum_angular = max(
        _outward_divide(
            _outward_vector_difference(
                tuple(float(component) for component in item), initial_angular
            ),
            angular_denominator,
        )
        for item in angular
    )
    return {
        "massive_invariant_sampling": "ALL_JX_FIXED_SOURCE_NODES",
        "massive_invariant_sample_count": int(position.shape[0]),
        "maximum_relative_massive_energy_drift_hex": maximum_energy.hex(),
        "maximum_relative_massive_energy_drift_exact": _fraction_record(
            maximum_energy_fraction
        ),
        "maximum_relative_massive_angular_momentum_drift_hex": maximum_angular.hex(),
    }


def _require_source_invariant_summary(
    retained_summary: object, recomputed: Mapping[str, Any]
) -> None:
    if (
        type(retained_summary) is not dict
        or set(recomputed) != {
            "massive_invariant_sampling", "massive_invariant_sample_count",
            "maximum_relative_massive_energy_drift_hex",
            "maximum_relative_massive_energy_drift_exact",
            "maximum_relative_massive_angular_momentum_drift_hex",
        }
        or any(retained_summary.get(key) != value for key, value in recomputed.items())
    ):
        raise Stage1QualificationError(
            "JX all-node invariant summary differs from retained source nodes"
        )


def _checkpoint_from_centered(
    year: int,
    massive: Sequence[MassiveState],
    tracers: Sequence[TracerState],
) -> dict[str, object]:
    if type(year) is not int or not 0 <= year <= HORIZON_JULIAN_YEARS:
        raise Stage1QualificationError("checkpoint year is outside the frozen horizon")
    if len(tracers) != TNO_STATES_PER_BATCH or not massive:
        raise Stage1QualificationError("checkpoint particle roster changed")
    residual_position, residual_velocity = _weighted_centroid(massive)
    if (
        max(abs(item) for item in residual_position) > 1.0e-12
        or max(abs(item) for item in residual_velocity) > 1.0e-12
    ):
        raise Stage1QualificationError(
            "checkpoint is not in the declared instantaneous massive GM barycenter"
        )
    mu = math.fsum(item.gm_AU3_per_julian_year2 for item in massive)
    energy, angular = _massive_invariants_centered(massive)
    tracer_rows: list[dict[str, object]] = []
    for order, tracer in enumerate(tracers):
        elements = extract_elliptic_orbital_elements(
            tracer.position_AU,
            tracer.velocity_AU_per_julian_year,
            mu,
        )
        tracer_rows.append(
            {
                "tracer_order": order,
                "sample_kind": tracer.sample_kind,
                "clone_index": tracer.clone_index,
                "elements": {
                    "semimajor_axis_AU": elements["semimajor_axis_AU"].hex()
                },
            }
        )
    return {
        "year_index": year,
        "time_julian_years_hex": float(year).hex(),
        "massive_energy_hex": energy.hex(),
        "massive_angular_momentum_hex": [item.hex() for item in angular],
        "tracers": tracer_rows,
        "massive_positions_AU": [
            list(item.position_AU) for item in massive
        ],
        "massive_velocities_AU_per_julian_year": [
            list(item.velocity_AU_per_julian_year) for item in massive
        ],
        "tracer_positions_AU": [
            list(item.position_AU) for item in tracers
        ],
        "tracer_velocities_AU_per_julian_year": [
            list(item.velocity_AU_per_julian_year) for item in tracers
        ],
    }


def _checkpoint(year: int, massive: Sequence[MassiveState], tracers: Sequence[TracerState]) -> dict[str, object]:
    """Canonicalize one observation to the instantaneous massive GM barycenter."""

    center_position, center_velocity = _weighted_centroid(massive)
    centered_massive = tuple(
        MassiveState(
            item.naif_id,
            item.body_id,
            item.gm_AU3_per_julian_year2,
            _sub(item.position_AU, center_position),
            _sub(item.velocity_AU_per_julian_year, center_velocity),
        )
        for item in massive
    )
    centered_tracers = tuple(
        TracerState(
            item.object_id,
            item.spk_id,
            item.sample_kind,
            item.clone_index,
            _sub(item.position_AU, center_position),
            _sub(item.velocity_AU_per_julian_year, center_velocity),
        )
        for item in tracers
    )
    return _checkpoint_from_centered(year, centered_massive, centered_tracers)


def _lane_summary(checkpoints: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    if type(checkpoints) not in (list, tuple) or len(checkpoints) != CHECKPOINT_COUNT:
        raise Stage1QualificationError("lane summary checkpoint count changed")
    summaries: list[dict[str, object]] = []
    for tracer_order in range(TNO_STATES_PER_BATCH):
        values = [
            _exact_float_hex(checkpoint["tracers"][tracer_order]["elements"]["semimajor_axis_AU"], "retained semimajor axis")
            for checkpoint in checkpoints
        ]
        metric = preliminary_scattering_metric(values, expected_count=CHECKPOINT_COUNT)
        summaries.append(
            {
                "tracer_order": tracer_order,
                **{key: (value.hex() if type(value) is float else value) for key, value in metric.items()},
            }
        )
    initial_energy = _exact_float_hex(checkpoints[0]["massive_energy_hex"], "initial energy")
    initial_angular = tuple(_exact_float_hex(item, "initial angular momentum") for item in checkpoints[0]["massive_angular_momentum_hex"])
    angular_norm = math.hypot(*initial_angular)
    if initial_energy == 0.0 or not angular_norm > 0.0:
        raise Stage1QualificationError("massive invariant denominator is zero")
    angular_denominator = math.nextafter(angular_norm, -math.inf)
    if not angular_denominator > 0.0:
        raise Stage1QualificationError("massive angular-momentum denominator underflowed")
    max_energy_fraction = _maximum_exact_relative_energy_drift(
        tuple(
            _exact_float_hex(item["massive_energy_hex"], "energy")
            for item in checkpoints
        )
    )
    max_energy = _fraction_to_outward_float(max_energy_fraction)
    max_angular = max(
        _outward_divide(
            _outward_vector_difference(
                tuple(
                    _exact_float_hex(value, "angular momentum")
                    for value in item["massive_angular_momentum_hex"]
                ),
                initial_angular,
            ),
            angular_denominator,
        )
        for item in checkpoints
    )
    return {
        "tracer_diagnostics": summaries,
        "maximum_relative_massive_energy_drift_hex": max_energy.hex(),
        "maximum_relative_massive_energy_drift_exact": _fraction_record(
            max_energy_fraction
        ),
        "maximum_relative_massive_angular_momentum_drift_hex": max_angular.hex(),
    }


def _validate_retained_lane_summary(
    lane: Mapping[str, Any], checkpoints: Sequence[Mapping[str, Any]]
) -> None:
    annual = _lane_summary(checkpoints)
    summary = lane.get("summary")
    if type(summary) is not dict:
        raise Stage1QualificationError("lane summary type changed")
    if lane.get("engine") == "JX_RESTRICTED_WH":
        expected_keys = {
            "tracer_diagnostics",
            "maximum_relative_massive_energy_drift_hex",
            "maximum_relative_massive_energy_drift_exact",
            "maximum_relative_massive_angular_momentum_drift_hex",
            "annual_maximum_relative_massive_energy_drift_hex",
            "annual_maximum_relative_massive_energy_drift_exact",
            "annual_maximum_relative_massive_angular_momentum_drift_hex",
            "massive_invariant_sampling",
            "massive_invariant_sample_count",
        }
        steps_per_year = lane.get("steps_per_year")
        if (
            set(summary) != expected_keys
            or summary.get("massive_invariant_sampling")
            != "ALL_JX_FIXED_SOURCE_NODES"
            or type(steps_per_year) is not int
            or steps_per_year not in JX_STEPS_PER_YEAR
            or summary.get("massive_invariant_sample_count")
            != HORIZON_JULIAN_YEARS * steps_per_year + 1
            or _canonical_json(summary.get("tracer_diagnostics"))
            != _canonical_json(annual["tracer_diagnostics"])
            or summary.get("annual_maximum_relative_massive_energy_drift_hex")
            != annual["maximum_relative_massive_energy_drift_hex"]
            or summary.get("annual_maximum_relative_massive_energy_drift_exact")
            != annual["maximum_relative_massive_energy_drift_exact"]
            or summary.get("annual_maximum_relative_massive_angular_momentum_drift_hex")
            != annual["maximum_relative_massive_angular_momentum_drift_hex"]
        ):
            raise Stage1QualificationError("JX lane summary differs from annual states")
        _exact_float_hex(
            summary.get("maximum_relative_massive_energy_drift_hex"),
            "JX all-source-node energy drift",
        )
        energy_fraction = _fraction_from_record(
            summary.get("maximum_relative_massive_energy_drift_exact"),
            "JX all-source-node energy drift",
        )
        if (
            summary.get("maximum_relative_massive_energy_drift_hex")
            != _fraction_to_outward_float(energy_fraction).hex()
        ):
            raise Stage1QualificationError("JX energy drift display is not outward exact")
        _exact_float_hex(
            summary.get("maximum_relative_massive_angular_momentum_drift_hex"),
            "JX all-source-node angular drift",
        )
    elif lane.get("engine") == "REBOUND_5.1.1_IAS15":
        expected = {
            **annual,
            "massive_invariant_sampling": "ANNUAL_CHECKPOINTS_ONLY",
            "massive_invariant_sample_count": CHECKPOINT_COUNT,
        }
        if _canonical_json(summary) != _canonical_json(expected):
            raise Stage1QualificationError("IAS15 lane summary differs from annual states")
    else:
        raise Stage1QualificationError("lane engine changed")


_CARTESIAN_KEYS = (
    "massive_positions_AU",
    "massive_velocities_AU_per_julian_year",
    "tracer_positions_AU",
    "tracer_velocities_AU_per_julian_year",
)


def _packed_lane_arrays(lane: Mapping[str, Any]) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    if type(lane) is not dict:
        raise Stage1QualificationError("lane must be an exact dictionary")
    if any(type(key) is not str or key.startswith("_") for key in lane):
        raise Stage1QualificationError("private or non-string lane key reached publication")
    checkpoints = lane.get("checkpoints")
    model = lane.get("model")
    massive_count = len(RESOLVED11_NAIF_IDS) if model == "RESOLVED_11_BODY" else len(REDUCED_BODY_IDS) if model == "REDUCED_5_BODY" else 0
    if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT or massive_count == 0:
        raise Stage1QualificationError("lane checkpoint or massive-model roster changed")
    record_doubles = 6 * (massive_count + TNO_STATES_PER_BATCH)
    payload = bytearray()
    gate_checkpoints: list[dict[str, Any]] = []
    for year, checkpoint in enumerate(checkpoints):
        if type(checkpoint) is not dict or checkpoint.get("year_index") != year:
            raise Stage1QualificationError("lane checkpoint schedule changed")
        shapes = (
            ("massive_positions_AU", massive_count),
            ("massive_velocities_AU_per_julian_year", massive_count),
            ("tracer_positions_AU", TNO_STATES_PER_BATCH),
            ("tracer_velocities_AU_per_julian_year", TNO_STATES_PER_BATCH),
        )
        values: list[float] = []
        for key, row_count in shapes:
            rows = checkpoint.get(key)
            if type(rows) is not list or len(rows) != row_count:
                raise Stage1QualificationError(f"{key} shape changed")
            for row_index, row in enumerate(rows):
                vector = _vec3(row, f"{key}[{row_index}]")
                values.extend(vector)
        if len(values) != record_doubles:
            raise Stage1QualificationError("Cartesian checkpoint width changed")
        payload.extend(struct.pack(f"<{record_doubles}d", *values))
        gate_checkpoints.append(dict(checkpoint))
    lane_payload = bytes(payload)
    compact_lane = {key: value for key, value in lane.items() if key != "checkpoints"}
    compact_lane["cartesian_array"] = {
        "schema": ARRAY_SCHEMA,
        "checkpoint_count": CHECKPOINT_COUNT,
        "massive_count": massive_count,
        "tracer_count": TNO_STATES_PER_BATCH,
        "components": 3,
        "record_doubles": record_doubles,
        "record_bytes": record_doubles * 8,
        "byte_length": len(lane_payload),
        "lane_payload_sha256": _sha256(lane_payload),
        "field_order": list(_CARTESIAN_KEYS),
    }
    gate_lane = dict(compact_lane)
    gate_lane.pop("cartesian_array")
    gate_lane["checkpoints"] = gate_checkpoints
    return lane_payload, compact_lane, gate_lane


def materialize_cartesian_payload(
    batch_lanes: Sequence[Mapping[str, Any]],
) -> tuple[bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    if type(batch_lanes) not in (list, tuple):
        raise Stage1QualificationError("batch lanes must be a list or tuple")
    payload = bytearray()
    compact_records: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    for record in batch_lanes:
        if type(record) is not dict or type(record.get("lane")) is not dict:
            raise Stage1QualificationError("batch-lane record type changed")
        lane_payload, compact_lane, gate_lane = _packed_lane_arrays(record["lane"])
        offset = len(payload)
        compact_lane["cartesian_array"]["byte_offset"] = offset
        payload.extend(lane_payload)
        wrapper = {key: value for key, value in record.items() if key != "lane"}
        compact_records.append({**wrapper, "lane": compact_lane})
        gate_records.append({**wrapper, "lane": gate_lane})
    return bytes(payload), compact_records, gate_records


def _unpack_lane_arrays(
    payload: bytes,
    record: Mapping[str, Any],
    template: BatchState,
) -> dict[str, Any]:
    if type(record) is not dict or type(record.get("lane")) is not dict:
        raise Stage1QualificationError("compact batch-lane record type changed")
    lane = record["lane"]
    array = lane.get("cartesian_array")
    if type(array) is not dict or array.get("schema") != ARRAY_SCHEMA:
        raise Stage1QualificationError("Cartesian lane metadata changed")
    model = lane.get("model")
    expected_template = template if model == "RESOLVED_11_BODY" else collapse_reduced_model(template) if model == "REDUCED_5_BODY" else None
    if expected_template is None:
        raise Stage1QualificationError("unknown retained lane model")
    massive_count = len(expected_template.massive)
    record_doubles = 6 * (massive_count + TNO_STATES_PER_BATCH)
    offset = array.get("byte_offset")
    byte_length = array.get("byte_length")
    expected_length = CHECKPOINT_COUNT * record_doubles * 8
    if (
        set(array) != {
            "schema", "checkpoint_count", "massive_count", "tracer_count",
            "components", "record_doubles", "record_bytes", "byte_length",
            "lane_payload_sha256", "field_order", "byte_offset",
        }
        or array.get("schema") != ARRAY_SCHEMA
        or type(offset) is not int
        or offset < 0
        or type(byte_length) is not int
        or byte_length != expected_length
        or array.get("checkpoint_count") != CHECKPOINT_COUNT
        or array.get("massive_count") != massive_count
        or array.get("tracer_count") != TNO_STATES_PER_BATCH
        or array.get("components") != 3
        or array.get("record_doubles") != record_doubles
        or array.get("record_bytes") != record_doubles * 8
        or array.get("field_order") != list(_CARTESIAN_KEYS)
        or offset + byte_length > len(payload)
    ):
        raise Stage1QualificationError("Cartesian lane layout changed")
    lane_payload = payload[offset : offset + byte_length]
    if array.get("lane_payload_sha256") != _sha256(lane_payload):
        raise Stage1QualificationError("Cartesian lane payload digest changed")
    if lane.get("engine") == "JX_RESTRICTED_WH":
        primary = expected_template.massive[0]
        initial_massive = tuple(
            MassiveState(
                item.naif_id,
                item.body_id,
                item.gm_AU3_per_julian_year2,
                _sub(item.position_AU, primary.position_AU),
                _sub(item.velocity_AU_per_julian_year, primary.velocity_AU_per_julian_year),
            )
            for item in expected_template.massive
        )
        initial_tracers = tuple(
            TracerState(
                item.object_id,
                item.spk_id,
                item.sample_kind,
                item.clone_index,
                _sub(item.position_AU, primary.position_AU),
                _sub(item.velocity_AU_per_julian_year, primary.velocity_AU_per_julian_year),
            )
            for item in expected_template.tracers
        )
        expected_initial = _checkpoint(0, initial_massive, initial_tracers)
    else:
        expected_initial = _checkpoint(
            0, expected_template.massive, expected_template.tracers
        )
    expected_initial_values: list[float] = []
    for key in _CARTESIAN_KEYS:
        for row_index, row in enumerate(expected_initial[key]):
            expected_initial_values.extend(_vec3(row, f"expected initial {key}[{row_index}]"))
    expected_initial_payload = struct.pack(
        f"<{record_doubles}d", *expected_initial_values
    )
    if lane_payload[: record_doubles * 8] != expected_initial_payload:
        raise Stage1QualificationError(
            "Cartesian year-zero state is not the sealed canonical initial state"
        )
    checkpoints: list[dict[str, Any]] = []
    cursor = 0
    for year in range(CHECKPOINT_COUNT):
        values = struct.unpack_from(f"<{record_doubles}d", lane_payload, cursor)
        cursor += record_doubles * 8
        if not all(math.isfinite(value) for value in values):
            raise Stage1QualificationError("Cartesian payload contains nonfinite data")
        index = 0
        def vectors(count: int) -> tuple[tuple[float, float, float], ...]:
            nonlocal index
            result = tuple(
                (values[index + 3 * row], values[index + 3 * row + 1], values[index + 3 * row + 2])
                for row in range(count)
            )
            index += count * 3
            return result
        massive_positions = vectors(massive_count)
        massive_velocities = vectors(massive_count)
        tracer_positions = vectors(TNO_STATES_PER_BATCH)
        tracer_velocities = vectors(TNO_STATES_PER_BATCH)
        if index != record_doubles:
            raise Stage1QualificationError("Cartesian decoder width changed")
        massive = tuple(
            MassiveState(
                item.naif_id,
                item.body_id,
                item.gm_AU3_per_julian_year2,
                massive_positions[row],
                massive_velocities[row],
            )
            for row, item in enumerate(expected_template.massive)
        )
        tracers = tuple(
            TracerState(
                item.object_id,
                item.spk_id,
                item.sample_kind,
                item.clone_index,
                tracer_positions[row],
                tracer_velocities[row],
            )
            for row, item in enumerate(expected_template.tracers)
        )
        checkpoint = _checkpoint_from_centered(year, massive, tracers)
        checkpoints.append(checkpoint)
    if cursor != len(lane_payload):
        raise Stage1QualificationError("Cartesian lane payload has trailing bytes")
    _validate_retained_lane_summary(lane, checkpoints)
    gate_lane = {key: value for key, value in lane.items() if key != "cartesian_array"}
    gate_lane["checkpoints"] = checkpoints
    return {**{key: value for key, value in record.items() if key != "lane"}, "lane": gate_lane}


def _validate_fused_tracer_runtime_identity(value: object) -> None:
    expected_keys = {
        "method_id", "backend", "device", "device_index", "device_name",
        "device_uuid", "device_compute_capability",
        "device_pci_domain_bus_device", "cupy_version",
        "cuda_runtime_version", "cuda_local_runtime_version",
        "cuda_driver_version", "nvrtc_version", "kernel_id",
        "execution_kernel_source_sha256", "implementation_source_sha256",
        "all_conic_implementation_source_sha256",
        "all_conic_device_source_sha256", "kick_cuda_source_sha256",
        "fused_cuda_source_sha256", "cuda_options", "determinism_scope",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise Stage1QualificationError("fused tracer runtime identity shape changed")
    if (
        value.get("backend") != "cupy"
        or value.get("device") != "cuda:0"
        or value.get("device_index") != 0
        or value.get("determinism_scope") != "SAME_CUPY_CUDA_RUNTIME_AND_DEVICE"
        or value.get("implementation_source_sha256") != STATIC_BINDINGS[14][1]
        or value.get("all_conic_implementation_source_sha256")
        != STATIC_BINDINGS[15][1]
        or any(
            not _is_sha256(value.get(key))
            for key in (
                "execution_kernel_source_sha256",
                "implementation_source_sha256",
                "all_conic_implementation_source_sha256",
                "all_conic_device_source_sha256",
                "kick_cuda_source_sha256",
                "fused_cuda_source_sha256",
            )
        )
        or type(value.get("device_name")) is not str
        or not value.get("device_name")
        or type(value.get("device_uuid")) is not str
        or not value.get("device_uuid")
        or type(value.get("cupy_version")) is not str
        or not value.get("cupy_version")
        or any(
            type(value.get(key)) is not int or value.get(key) <= 0
            for key in (
                "cuda_runtime_version",
                "cuda_local_runtime_version",
                "cuda_driver_version",
            )
        )
        or type(value.get("nvrtc_version")) is not list
        or len(value.get("nvrtc_version")) != 2
        or any(type(item) is not int or item < 0 for item in value["nvrtc_version"])
        or type(value.get("cuda_options")) is not list
        or any(type(item) is not str for item in value["cuda_options"])
    ):
        raise Stage1QualificationError("fused tracer runtime identity changed")


def _validate_source_build_runtime_identity(
    value: object, full_source_tile_sha256: object
) -> None:
    _guard_exact_json_integer_identities(value, "source-build runtime identity")
    if type(value) is not dict:
        raise Stage1QualificationError("source-build runtime identity type changed")
    expected_keys = {
        "provider_id", "provider_implementation_source_sha256",
        "python_version", "platform", "numpy_version",
        "numpy_runtime_identity",
        "source_map_runtime_identity", "exact_replay_required",
        "coordinate_canonicalization",
        "coordinate_canonicalization_receipt_sha256", "source_tile_sha256",
    }
    source_map = value.get("source_map_runtime_identity")
    extension = source_map.get("extension_binary") if type(source_map) is dict else None
    if (
        set(value) != expected_keys
        or value.get("provider_implementation_source_sha256")
        != STATIC_BINDINGS[13][1]
        or value.get("exact_replay_required") is not True
        or value.get("source_tile_sha256") != full_source_tile_sha256
        or not _is_sha256(value.get("coordinate_canonicalization_receipt_sha256"))
        or type(source_map) is not dict
        or type(extension) is not dict
        or set(extension) != {"path", "size_bytes", "sha256"}
        or type(extension.get("path")) is not str
        or type(extension.get("size_bytes")) is not int
        or extension.get("size_bytes") <= 0
        or not _is_sha256(extension.get("sha256"))
    ):
        raise Stage1QualificationError("source-build runtime identity changed")
    if value.get("numpy_version") != NUMPY_DISTRIBUTION_VERSION:
        raise Stage1QualificationError("source-build NumPy version changed")
    _validate_numpy_runtime_identity(value.get("numpy_runtime_identity"))


def _validate_compact_lane_shape(
    lane: Mapping[str, Any], lane_spec: Mapping[str, Any]
) -> None:
    _guard_exact_json_integer_identities(lane, "compact lane")
    if lane.get("engine") == "JX_RESTRICTED_WH":
        expected_keys = {
            "lane_id", "engine", "model", "step_julian_years_hex",
            "steps_per_year", "full_source_tile_sha256",
            "annual_source_slice_sha256", "source_input_content_sha256",
            "source_map_result_content_sha256",
            "source_map_verification_content_sha256",
            "source_build_runtime_identity", "fused_tracer_runtime_identity",
            "guard_contract", "maximum_kepler_iterations",
            "elapsed_seconds_hex", "summary", "source_node_evidence",
            "cartesian_array",
        }
        digests = lane.get("annual_source_slice_sha256")
        runtime_identity = lane.get("source_build_runtime_identity")
        fused_runtime_identity = lane.get("fused_tracer_runtime_identity")
        guard_contract = lane.get("guard_contract")
        if (
            set(lane) != expected_keys
            or lane.get("steps_per_year") != lane_spec.get("steps_per_julian_year")
            or lane.get("step_julian_years_hex")
            != lane_spec.get("integration_step_julian_years_hex")
            or not _is_sha256(lane.get("full_source_tile_sha256"))
            or type(digests) is not list
            or len(digests) != HORIZON_JULIAN_YEARS
            or any(not _is_sha256(item) for item in digests)
            or any(
                not _is_sha256(lane.get(key))
                for key in (
                    "source_input_content_sha256",
                    "source_map_result_content_sha256",
                    "source_map_verification_content_sha256",
                )
            )
            or type(lane.get("maximum_kepler_iterations")) is not int
            or lane.get("maximum_kepler_iterations") < 0
            or type(lane.get("source_node_evidence")) is not dict
            or set(lane["source_node_evidence"]) != {
                "source_order", "byte_offset", "byte_length",
                "lane_payload_sha256", "record_content_sha256",
            }
            or type(lane["source_node_evidence"].get("source_order")) is not int
            or type(lane["source_node_evidence"].get("byte_offset")) is not int
            or type(lane["source_node_evidence"].get("byte_length")) is not int
            or any(
                not _is_sha256(lane["source_node_evidence"].get(key))
                for key in ("lane_payload_sha256", "record_content_sha256")
            )
        ):
            raise Stage1QualificationError("compact JX lane shape changed")
        _validate_source_build_runtime_identity(
            runtime_identity, lane.get("full_source_tile_sha256")
        )
        _validate_fused_tracer_runtime_identity(fused_runtime_identity)
        if (
            type(guard_contract) is not dict
            or set(guard_contract) != {
                "source_encounter_radius_AU_hex",
                "minimum_massive_pair_separation_AU_hex",
                "minimum_jacobi_periapse_AU_hex",
                "primary_outer_boundary_AU_hex",
                "acceleration_bound_rule",
                "tracer_indices",
            }
            or guard_contract.get("source_encounter_radius_AU_hex")
            != float(1.0).hex()
            or guard_contract.get("minimum_massive_pair_separation_AU_hex")
            != float(1.0).hex()
            or guard_contract.get("minimum_jacobi_periapse_AU_hex")
            != float(0.1).hex()
            or guard_contract.get("primary_outer_boundary_AU_hex")
            != float(4096.0).hex()
            or guard_contract.get("acceleration_bound_rule")
            != "NEXTUP_OF_2_TIMES_TOTAL_REDUCED_GM_DIVIDED_BY_ONE_AU_SQUARED"
            or guard_contract.get("tracer_indices")
            != list(range(TNO_STATES_PER_BATCH))
        ):
            raise Stage1QualificationError("compact JX guard contract changed")
        elapsed = _exact_float_hex(lane.get("elapsed_seconds_hex"), "JX elapsed time")
        if elapsed < 0.0:
            raise Stage1QualificationError("JX elapsed time is negative")
    elif lane.get("engine") == "REBOUND_5.1.1_IAS15":
        expected_keys = {
            "lane_id", "engine", "model", "epsilon_hex",
            "initial_dt_julian_years_hex", "min_dt_julian_years_hex",
            "adaptive_mode",
            "final_time_julian_years_hex", "steps_done", "elapsed_seconds_hex",
            "accepted_step_guard", "summary", "cartesian_array",
        }
        if (
            set(lane) != expected_keys
            or lane.get("epsilon_hex") != IAS15_EPSILON.hex()
            or lane.get("initial_dt_julian_years_hex")
            != IAS15_INITIAL_DT_JULIAN_YEARS.hex()
            or lane.get("min_dt_julian_years_hex")
            != IAS15_MIN_DT_JULIAN_YEARS.hex()
            or lane.get("adaptive_mode") != IAS15_ADAPTIVE_MODE
            or lane.get("final_time_julian_years_hex")
            != float(HORIZON_JULIAN_YEARS).hex()
            or type(lane.get("steps_done")) is not int
            or lane.get("steps_done") <= 0
            or type(lane.get("accepted_step_guard")) is not dict
        ):
            raise Stage1QualificationError("compact IAS15 lane shape changed")
        elapsed = _exact_float_hex(lane.get("elapsed_seconds_hex"), "IAS15 elapsed time")
        if elapsed < 0.0:
            raise Stage1QualificationError("IAS15 elapsed time is negative")
    else:
        raise Stage1QualificationError("compact lane engine changed")


def reconstruct_gate_lanes(
    payload: bytes,
    compact_records: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if type(payload) is not bytes:
        raise Stage1QualificationError("Cartesian payload must be bytes")
    if len(payload) != EXPECTED_CARTESIAN_ARRAY_BYTES:
        raise Stage1QualificationError("Cartesian payload has the wrong exact byte length")
    if type(compact_records) not in (list, tuple) or len(compact_records) != OBJECT_COUNT * 2 * len(LANE_SPECS):
        raise Stage1QualificationError("compact batch-lane roster changed")
    templates = {
        ephemeris: _batch_states(bridge, custody, ephemeris)
        for ephemeris in ("DE440S", "DE441")
    }
    result: list[dict[str, Any]] = []
    expected_offset = 0
    expected_order = [
        (ephemeris, batch_order, spec)
        for ephemeris in ("DE440S", "DE441")
        for batch_order in range(OBJECT_COUNT)
        for spec in LANE_SPECS
    ]
    for record, (expected_ephemeris, expected_batch_order, lane_spec) in zip(
        compact_records, expected_order, strict=True
    ):
        if type(record) is not dict or set(record) != {
            "ephemeris", "batch_order", "object_id", "native_epoch_jd_tdb_hex", "lane"
        }:
            raise Stage1QualificationError("compact batch-lane wrapper shape changed")
        ephemeris = record.get("ephemeris")
        batch_order = record.get("batch_order")
        if ephemeris != expected_ephemeris or batch_order != expected_batch_order:
            raise Stage1QualificationError("compact batch identity changed")
        template = templates[ephemeris][batch_order]
        lane = record.get("lane")
        if (
            record.get("object_id") != template.object_id
            or record.get("native_epoch_jd_tdb_hex") != template.epoch_jd_tdb.hex()
            or type(lane) is not dict
            or lane.get("lane_id") != lane_spec["lane_id"]
            or lane.get("engine") != lane_spec["engine"]
            or lane.get("model") != lane_spec["model"]
            or "checkpoints" in lane
        ):
            raise Stage1QualificationError("compact lane identity or sealed epoch changed")
        _validate_compact_lane_shape(lane, lane_spec)
        array = lane.get("cartesian_array")
        if type(array) is not dict or array.get("byte_offset") != expected_offset:
            raise Stage1QualificationError("Cartesian lane offsets are not canonical and contiguous")
        decoded = _unpack_lane_arrays(payload, record, template)
        expected_offset += array["byte_length"]
        result.append(decoded)
    if expected_offset != len(payload):
        raise Stage1QualificationError("Cartesian payload has trailing or unbound bytes")
    return result


def _runtime_prefix_keys() -> list[str]:
    if any(type(name) is not str for name in sys.modules):
        raise Stage1QualificationError(
            "runtime registry contains a non-string module key"
        )
    return sorted(
        name
        for name in sys.modules
        if any(
            name == prefix or name.startswith(prefix + ".")
            for prefix in RUNTIME_MODULE_PREFIXES
        )
    )


def _runtime_module_names() -> list[str]:
    return [
        name
        for name in _runtime_prefix_keys()
        if type(sys.modules.get(name)) is types.ModuleType
    ]


def _reject_preloaded_runtime_modules() -> dict[str, object]:
    observed = _runtime_prefix_keys()
    if observed:
        raise Stage1QualificationError(
            "fresh dynamics worker registry is contaminated: " + ",".join(observed[:8])
        )
    return {
        "checked_prefixes": list(RUNTIME_MODULE_PREFIXES),
        "preloaded_module_names": [],
        "fresh_registry_observed": True,
        "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
    }


def _runtime_file(
    path: Path, label: str, *, allow_empty: bool = False
) -> dict[str, object]:
    return _exact_live_file_receipt(path, label, allow_empty=allow_empty)


def _module_runtime_receipt(
    module_name: str,
    expected_files: Mapping[str, tuple[int, str, str]] | None = None,
) -> dict[str, object]:
    module = sys.modules.get(module_name)
    if type(module) is not types.ModuleType:
        raise Stage1QualificationError(f"required runtime module was not imported: {module_name}")
    spec = getattr(module, "__spec__", None)
    loader = getattr(spec, "loader", None)
    path_value = getattr(module, "__file__", None)
    origin = getattr(spec, "origin", None)
    if (
        type(path_value) is not str
        or type(origin) is not str
        or loader is None
        or getattr(module, "__loader__", None) is not loader
    ):
        raise Stage1QualificationError(f"runtime module spec is incomplete: {module_name}")
    path = Path(os.path.abspath(path_value))
    origin_path = Path(os.path.abspath(origin))
    if path != origin_path:
        raise Stage1QualificationError(f"runtime module file/spec origin differ: {module_name}")
    suffix = path.suffix
    if suffix == ".py":
        kind = "PYTHON_SOURCE"
    elif suffix in (".so", ".pyd"):
        kind = "EXTENSION_BINARY"
    else:
        raise Stage1QualificationError(f"runtime module file kind is not source/binary: {module_name}")
    cached = getattr(module, "__cached__", None)
    if cached is not None and type(cached) is not str:
        raise Stage1QualificationError(f"runtime module cache locator changed: {module_name}")
    expected_file = (
        expected_files.get(str(path)) if expected_files is not None else None
    )
    if expected_files is not None and expected_file is None:
        raise Stage1QualificationError(
            f"runtime module is outside predeclared manifests: {module_name}"
        )
    allow_empty = False
    if expected_file is not None:
        expected_size, expected_digest, _authority = expected_file
        allow_empty = expected_size == 0 and expected_digest == EMPTY_SHA256
        if expected_size == 0 and not allow_empty:
            raise Stage1QualificationError(
                f"zero-byte runtime authority is invalid: {module_name}"
            )
    file_receipt = _runtime_file(
        path, f"runtime module {module_name}", allow_empty=allow_empty
    )
    if expected_file is not None and (
        file_receipt["bytes"] != expected_file[0]
        or file_receipt["sha256"] != expected_file[1]
    ):
        raise Stage1QualificationError(
            f"runtime module differs from predeclared bytes: {module_name}"
        )
    return {
        "module": module_name,
        "spec_name": getattr(spec, "name", None),
        "spec_origin": str(origin_path),
        "loader_type": f"{type(loader).__module__}.{type(loader).__qualname__}",
        "file_kind": kind,
        "cached_path": cached,
        **file_receipt,
    }


def _runtime_expected_file_map(
    rebound_target_receipt: Mapping[str, Any] | None,
) -> tuple[dict[str, tuple[int, str, str]], dict[str, tuple[int, str, str]]]:
    jx: dict[str, tuple[int, str, str]] = {}
    for item in _predeclared_jx_file_manifest():
        absolute = str(
            Path(os.path.abspath(
                os.fspath(PROJECT_ROOT / str(item["relative_path"]))
            ))
        )
        jx[absolute] = (int(item["bytes"]), str(item["sha256"]), "JX_PREDECLARED_MANIFEST")
    cupy: dict[str, tuple[int, str, str]] = {}
    for relative, (size, digest) in _cupy_record_manifest().items():
        absolute = str(Path(os.path.abspath(os.fspath(CUPY_SITE_PACKAGES / relative))))
        cupy[absolute] = (size, digest, "PINNED_CUPY_RECORD")
    numpy: dict[str, tuple[int, str, str]] = {}
    for relative, (size, digest) in _numpy_record_manifest().items():
        absolute = str(Path(os.path.abspath(os.fspath(CUPY_SITE_PACKAGES / relative))))
        numpy[absolute] = (size, digest, "PINNED_NUMPY_RECORD")
    rebound: dict[str, tuple[int, str, str]] = {}
    if rebound_target_receipt is not None:
        members = rebound_target_receipt.get("members")
        if type(members) is not list:
            raise Stage1QualificationError("REBOUND target receipt member roster changed")
        for item in members:
            if type(item) is not dict:
                raise Stage1QualificationError("REBOUND target receipt member changed")
            rebound[str(Path(os.path.abspath(str(item["installed_path"]))))] = (
                int(item["bytes"]),
                str(item["sha256"]),
                "PINNED_REBOUND_WHEEL",
            )
    return {**jx, **cupy, **numpy}, rebound


def _runtime_module_registry_receipts(
    rebound_target_receipt: Mapping[str, Any] | None,
    *,
    require_rebound: bool,
) -> list[dict[str, object]]:
    expected, rebound_expected = _runtime_expected_file_map(rebound_target_receipt)
    names = _runtime_module_names()
    required_prefixes = ("jxplanetx", "numpy", "cupy", "cupy_backends")
    if require_rebound:
        required_prefixes += ("rebound",)
    for prefix in required_prefixes:
        if not any(name == prefix or name.startswith(prefix + ".") for name in names):
            raise Stage1QualificationError(f"required runtime module prefix is absent: {prefix}")
    combined_expected = {**expected, **rebound_expected}
    receipts: list[dict[str, object]] = []
    for name in names:
        receipt = _module_runtime_receipt(name, combined_expected)
        path = str(receipt["path"])
        expected_item = rebound_expected.get(path) if name == "rebound" or name.startswith("rebound.") else expected.get(path)
        if expected_item is None:
            raise Stage1QualificationError(f"runtime module is outside predeclared manifests: {name}")
        size, digest, authority = expected_item
        if receipt["bytes"] != size or receipt["sha256"] != digest:
            raise Stage1QualificationError(f"runtime module differs from predeclared bytes: {name}")
        receipt["predeclared_authority"] = authority
        receipts.append(receipt)
    for required_name in REQUIRED_RUNTIME_MODULE_NAMES:
        if required_name not in names:
            raise Stage1QualificationError(
                f"required exact runtime module is absent: {required_name}"
            )
    native_path = str((PROJECT_ROOT / JX_NATIVE_RELATIVE_PATH).resolve(strict=True))
    core_path = str((CUPY_SITE_PACKAGES / CUPY_CORE_RELATIVE_PATH).resolve(strict=True))
    observed_paths = {str(item["path"]) for item in receipts}
    if native_path not in observed_paths or core_path not in observed_paths:
        raise Stage1QualificationError("native Wisdom-Holman or CuPy core was not actually imported")
    if not require_rebound and any(name == "rebound" or name.startswith("rebound.") for name in names):
        raise Stage1QualificationError("restart worker unexpectedly imported REBOUND")
    return receipts


def _imported_module_runtime_file(module_name: str) -> dict[str, object]:
    return _module_runtime_receipt(module_name)


def _numpy_runtime_identity() -> dict[str, object]:
    numpy_module = sys.modules.get("numpy")
    if (
        type(numpy_module) is not types.ModuleType
        or getattr(numpy_module, "__version__", None) != NUMPY_DISTRIBUTION_VERSION
    ):
        raise Stage1QualificationError("exact NumPy runtime was not imported")
    if type(sys.modules.get("numpy._core._multiarray_umath")) is not types.ModuleType:
        raise Stage1QualificationError("NumPy core extension was not imported")
    distributions = [
        item
        for item in importlib.metadata.distributions(path=[str(CUPY_SITE_PACKAGES)])
        if (item.metadata.get("Name") or "").lower().replace("_", "-")
        == "numpy"
    ]
    if len(distributions) != 1:
        raise Stage1QualificationError(
            "exactly one NumPy distribution must own the runtime"
        )
    distribution = distributions[0]
    distribution_path = getattr(distribution, "_path", None)
    if (
        distribution.version != NUMPY_DISTRIBUTION_VERSION
        or not isinstance(distribution_path, Path)
        or (distribution_path / "RECORD").resolve(strict=True)
        != NUMPY_RECORD_PATH.resolve(strict=True)
    ):
        raise Stage1QualificationError("NumPy distribution metadata changed")
    return {
        "distribution_name": str(distribution.metadata["Name"]),
        "distribution_version": str(distribution.version),
        "module_version": str(numpy_module.__version__),
        "module": _imported_module_runtime_file("numpy"),
        "core_module": _imported_module_runtime_file(
            "numpy._core._multiarray_umath"
        ),
        "distribution_record": _exact_live_file_receipt(
            NUMPY_RECORD_PATH,
            "NumPy distribution RECORD",
            expected_path=NUMPY_RECORD_PATH,
            expected_bytes=NUMPY_RECORD_BYTES,
            expected_sha256=NUMPY_RECORD_SHA256,
            maximum=1_000_000,
        ),
        "distribution_metadata": _exact_live_file_receipt(
            NUMPY_METADATA_PATH,
            "NumPy distribution METADATA",
            expected_path=NUMPY_METADATA_PATH,
            expected_bytes=NUMPY_METADATA_BYTES,
            expected_sha256=NUMPY_METADATA_SHA256,
            maximum=100_000,
        ),
        "claim": "EXACT_LIVE_FILE_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION",
    }


def _validate_numpy_runtime_identity(
    value: object,
    registry_by_name: Mapping[str, Any] | None = None,
) -> None:
    if type(value) is not dict or set(value) != {
        "distribution_name", "distribution_version", "module_version",
        "module", "core_module", "distribution_record",
        "distribution_metadata", "claim"
    }:
        raise Stage1QualificationError("NumPy runtime identity shape changed")
    if (
        value.get("distribution_name") != "numpy"
        or value.get("distribution_version") != NUMPY_DISTRIBUTION_VERSION
        or value.get("module_version") != NUMPY_DISTRIBUTION_VERSION
        or value.get("claim")
        != "EXACT_LIVE_FILE_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION"
    ):
        raise Stage1QualificationError("NumPy runtime identity changed")
    module_keys = {
        "module", "spec_name", "spec_origin", "loader_type", "file_kind",
        "cached_path", "path", "bytes", "sha256",
    }
    expected = (
        (
            "module", "numpy", NUMPY_INIT_RELATIVE_PATH,
            NUMPY_INIT_BYTES, NUMPY_INIT_SHA256,
        ),
        (
            "core_module", "numpy._core._multiarray_umath",
            NUMPY_CORE_RELATIVE_PATH, NUMPY_CORE_BYTES, NUMPY_CORE_SHA256,
        ),
    )
    for key, module_name, relative, size, digest in expected:
        receipt = value.get(key)
        if (
            type(receipt) is not dict
            or set(receipt) != module_keys
            or receipt.get("module") != module_name
            or receipt.get("spec_name") != module_name
            or receipt.get("path")
            != str((CUPY_SITE_PACKAGES / relative).resolve(strict=True))
            or receipt.get("bytes") != size
            or receipt.get("sha256") != digest
        ):
            raise Stage1QualificationError("NumPy module receipt changed")
        _validate_runtime_file_receipt(
            {field: receipt[field] for field in ("path", "bytes", "sha256")},
            f"NumPy runtime module {module_name}",
        )
        if registry_by_name is not None and registry_by_name.get(module_name) != {
            **receipt,
            "predeclared_authority": "PINNED_NUMPY_RECORD",
        }:
            raise Stage1QualificationError(
                "NumPy runtime identity does not match the module registry"
            )
    record = value.get("distribution_record")
    _validate_runtime_file_receipt(record, "NumPy distribution RECORD")
    if record != {
        "path": str(NUMPY_RECORD_PATH.resolve(strict=True)),
        "bytes": NUMPY_RECORD_BYTES,
        "sha256": NUMPY_RECORD_SHA256,
    }:
        raise Stage1QualificationError("NumPy distribution RECORD changed")
    metadata = value.get("distribution_metadata")
    _validate_runtime_file_receipt(metadata, "NumPy distribution METADATA")
    if metadata != {
        "path": str(NUMPY_METADATA_PATH.resolve(strict=True)),
        "bytes": NUMPY_METADATA_BYTES,
        "sha256": NUMPY_METADATA_SHA256,
    }:
        raise Stage1QualificationError("NumPy distribution METADATA changed")
    distributions = [
        item
        for item in importlib.metadata.distributions(path=[str(CUPY_SITE_PACKAGES)])
        if (item.metadata.get("Name") or "").lower().replace("_", "-")
        == "numpy"
    ]
    if len(distributions) != 1:
        raise Stage1QualificationError("live NumPy distribution roster changed")
    distribution_path = getattr(distributions[0], "_path", None)
    if (
        distributions[0].version != NUMPY_DISTRIBUTION_VERSION
        or not isinstance(distribution_path, Path)
        or (distribution_path / "RECORD").resolve(strict=True)
        != NUMPY_RECORD_PATH.resolve(strict=True)
    ):
        raise Stage1QualificationError("live NumPy distribution changed")


def _module_runtime_cross_binding(value: object) -> dict[str, object]:
    """Return the process-independent part of an exact module receipt."""
    expected_keys = {
        "module", "spec_name", "spec_origin", "loader_type", "file_kind",
        "cached_path", "path", "bytes", "sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise Stage1QualificationError(
            "module runtime cross-binding receipt shape changed"
        )
    return {
        key: value[key]
        for key in expected_keys
        if key != "cached_path"
    }


def _numpy_runtime_cross_binding(value: object) -> dict[str, object]:
    """Project a validated NumPy receipt across fresh pycache namespaces."""
    if type(value) is not dict or set(value) != {
        "distribution_name", "distribution_version", "module_version",
        "module", "core_module", "distribution_record",
        "distribution_metadata", "claim",
    }:
        raise Stage1QualificationError(
            "NumPy runtime cross-binding receipt shape changed"
        )
    return {
        "distribution_name": value["distribution_name"],
        "distribution_version": value["distribution_version"],
        "module_version": value["module_version"],
        "module": _module_runtime_cross_binding(value["module"]),
        "core_module": _module_runtime_cross_binding(value["core_module"]),
        "distribution_record": value["distribution_record"],
        "distribution_metadata": value["distribution_metadata"],
        "claim": value["claim"],
    }


def _require_numpy_runtime_cross_binding(
    observed: object, expected: object, label: str
) -> None:
    if _canonical_json(
        _numpy_runtime_cross_binding(observed)
    ) != _canonical_json(_numpy_runtime_cross_binding(expected)):
        raise Stage1QualificationError(label)


def _numpy_prefix_module_keys() -> list[str]:
    if any(type(name) is not str for name in sys.modules):
        raise Stage1QualificationError(
            "offline validation module registry contains a non-string key"
        )
    return sorted(
        name
        for name in sys.modules
        if name == "numpy" or name.startswith("numpy.")
    )


@contextmanager
def _bound_offline_numpy_runtime(
    retained_identity: object,
) -> Iterable[types.ModuleType]:
    """Import and bind exact NumPy before offline numerical replay."""

    preloaded = _numpy_prefix_module_keys()
    if preloaded:
        raise Stage1QualificationError(
            "offline numerical validation requires a fresh NumPy registry: "
            + ",".join(preloaded[:8])
        )
    with tempfile.TemporaryDirectory(
        prefix="jx-tno-stage1-v2-offline-numpy-"
    ) as directory:
        cache_root = Path(directory) / "pycache-must-remain-absent"
        try:
            with _source_only_runtime_import_guard(cache_root):
                numpy_module = importlib.import_module("numpy")
                importlib.import_module("numpy._core._multiarray_umath")
                if type(numpy_module) is not types.ModuleType:
                    raise Stage1QualificationError(
                        "offline NumPy import is not an exact module"
                    )
                expected_files = {
                    str(Path(os.path.abspath(
                        os.fspath(CUPY_SITE_PACKAGES / relative)
                    ))): (size, digest, "PINNED_NUMPY_RECORD")
                    for relative, (size, digest) in _numpy_record_manifest().items()
                }
                registry: dict[str, dict[str, object]] = {}
                for module_name in (
                    "numpy", "numpy._core._multiarray_umath"
                ):
                    receipt = _module_runtime_receipt(
                        module_name, expected_files
                    )
                    expected = expected_files.get(str(receipt["path"]))
                    if expected is None or expected[2] != "PINNED_NUMPY_RECORD":
                        raise Stage1QualificationError(
                            "offline NumPy module lacks exact manifest authority"
                        )
                    registry[module_name] = {
                        **receipt,
                        "predeclared_authority": expected[2],
                    }
                live_identity = _numpy_runtime_identity()
                _validate_numpy_runtime_identity(live_identity, registry)
                _require_numpy_runtime_cross_binding(
                    live_identity,
                    retained_identity,
                    "offline NumPy runtime differs from retained execution identity",
                )
                yield numpy_module
        finally:
            for name in reversed(_numpy_prefix_module_keys()):
                sys.modules.pop(name, None)


def _validate_fresh_child_interpreter_receipt(value: object) -> None:
    expected = _project_venv_launcher_receipt()
    if type(value) is not dict or set(value) != {
        *expected.keys(),
        "observed_sys_executable", "observed_sys_prefix",
        "observed_sys_base_prefix", "claim",
    }:
        raise Stage1QualificationError("fresh-child interpreter receipt shape changed")
    if (
        {key: value.get(key) for key in expected} != expected
        or value.get("observed_sys_executable") != str(PROJECT_VENV_PYTHON)
        or value.get("observed_sys_prefix") != str(PROJECT_VENV_ROOT)
        or value.get("observed_sys_base_prefix") != "/usr"
        or value.get("claim")
        != "LEXICAL_PROJECT_VENV_LAUNCH_AND_EXACT_RESOLVED_INTERPRETER_POINT_OBSERVATION"
    ):
        raise Stage1QualificationError("fresh-child interpreter receipt changed")


def _restart_runtime_identity(
    cp: Any,
    fresh_registry_receipt: Mapping[str, Any],
    fresh_child_interpreter: Mapping[str, Any],
) -> dict[str, object]:
    executable = _exact_live_file_receipt(
        Path(sys.executable).resolve(strict=True), "restart Python executable",
        expected_path=PYTHON_EXECUTABLE_PATH,
        expected_bytes=PYTHON_EXECUTABLE_BYTES,
        expected_sha256=PYTHON_EXECUTABLE_SHA256,
        maximum=20_000_000,
    )
    registry = _runtime_module_registry_receipts(None, require_rebound=False)
    fast_module = sys.modules.get("jxplanetx.fast_wisdom_holman")
    identity_function = getattr(
        fast_module, "fast_wisdom_holman_runtime_identity", None
    )
    if not callable(identity_function):
        raise Stage1QualificationError(
            "restart native Wisdom-Holman identity API is missing"
        )
    native = json.loads(
        _canonical_json(identity_function()).decode("ascii")
    )
    properties = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))
    raw_uuid = properties.get("uuid")
    if raw_uuid is None:
        raise Stage1QualificationError("restart CUDA device UUID is unavailable")
    return {
        "schema": "jx.tno.stage1-dynamics-qualification.restart-runtime-identity.v2",
        "python_executable": executable,
        "fresh_child_interpreter": json.loads(
            _canonical_json(dict(fresh_child_interpreter)).decode("ascii")
        ),
        "stage1_implementation": _runtime_file(
            SCRIPT_PATH, "restart Stage-1 implementation"
        ),
        "runtime_module_registry": registry,
        "numpy_runtime_identity": _numpy_runtime_identity(),
        "native_wisdom_holman_runtime_identity": native,
        "cupy_cuda": {
            "cupy_version": str(cp.__version__),
            "cupy_module": _imported_module_runtime_file("cupy"),
            "cupy_core_module": _imported_module_runtime_file("cupy._core.core"),
            "cupy_fft_config_module": _imported_module_runtime_file(
                "cupy.fft._config"
            ),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(cp.cuda.get_local_runtime_version()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_device_id": int(cp.cuda.Device().id),
            "cuda_device_name": _cuda_device_name(cp),
            "cuda_device_uuid": bytes(raw_uuid).hex(),
            "cuda_compute_capability": (
                f"{int(properties['major'])}.{int(properties['minor'])}"
            ),
            "cuda_pci_domain_bus_device": (
                f"{int(properties['pciDomainID']):04x}:"
                f"{int(properties['pciBusID']):02x}:"
                f"{int(properties['pciDeviceID']):02x}"
            ),
        },
        "fresh_process_registry": json.loads(
            _canonical_json(dict(fresh_registry_receipt)).decode("ascii")
        ),
        "claim": "TRUSTED_FRESH_PROCESS_AND_EXACT_LIVE_FILE_REREAD_OBSERVATIONS_NOT_RUNTIME_AUTHENTICATION",
    }


def _validate_restart_runtime_identity(
    value: object,
    fused_tracer_runtime_identity: object,
    expected_main_runtime_identity: object | None = None,
) -> None:
    _guard_exact_json_integer_identities(value, "restart runtime identity")
    if type(value) is not dict or set(value) != {
        "schema", "python_executable", "fresh_child_interpreter",
        "stage1_implementation", "numpy_runtime_identity",
        "runtime_module_registry", "native_wisdom_holman_runtime_identity",
        "cupy_cuda", "fresh_process_registry", "claim",
    }:
        raise Stage1QualificationError("restart runtime identity shape changed")
    if (
        value.get("schema")
        != "jx.tno.stage1-dynamics-qualification.restart-runtime-identity.v2"
        or value.get("claim")
        != "TRUSTED_FRESH_PROCESS_AND_EXACT_LIVE_FILE_REREAD_OBSERVATIONS_NOT_RUNTIME_AUTHENTICATION"
        or value.get("fresh_process_registry") != {
            "checked_prefixes": list(RUNTIME_MODULE_PREFIXES),
            "preloaded_module_names": [],
            "fresh_registry_observed": True,
            "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
        }
    ):
        raise Stage1QualificationError("restart runtime identity claim changed")
    _validate_runtime_file_receipt(
        value.get("python_executable"), "restart Python executable"
    )
    _validate_fresh_child_interpreter_receipt(
        value.get("fresh_child_interpreter")
    )
    if value["python_executable"] != {
        "path": str(PYTHON_EXECUTABLE_PATH.resolve(strict=True)),
        "bytes": PYTHON_EXECUTABLE_BYTES,
        "sha256": PYTHON_EXECUTABLE_SHA256,
    }:
        raise Stage1QualificationError("restart Python executable changed")
    _validate_runtime_file_receipt(
        value.get("stage1_implementation"), "restart Stage-1 implementation"
    )
    implementation = _regular_bytes(SCRIPT_PATH, "Stage-1 implementation")
    if value["stage1_implementation"] != {
        "path": str(SCRIPT_PATH.resolve(strict=True)),
        "bytes": len(implementation),
        "sha256": _sha256(implementation),
    }:
        raise Stage1QualificationError("restart implementation receipt changed")
    _validate_runtime_module_registry(
        value.get("runtime_module_registry"), None, require_rebound=False
    )
    registry = {
        row["module"]: row
        for row in value["runtime_module_registry"]
    }
    _validate_numpy_runtime_identity(
        value.get("numpy_runtime_identity"), registry
    )
    native = value.get("native_wisdom_holman_runtime_identity")
    native_receipt = registry.get("jxplanetx._wisdom_holman_cpu_v3")
    extension = native.get("extension_binary") if type(native) is dict else None
    expected_native_keys = {
        "kernel_id", "g_kernel_id", "exact_fallback_kernel_id",
        "scientific_claim_state", "maximum_bodies", "metric_count",
        "counter_count", "compiler_version", "compiler_flags",
        "release_wrapper_source_sha256", "release_loop_source_sha256",
        "release_base_loop_source_sha256", "extension_binary",
    }
    if (
        type(native) is not dict
        or set(native) != expected_native_keys
        or type(native_receipt) is not dict
        or type(extension) is not dict
        or set(extension) != {"path", "size_bytes", "sha256"}
        or extension.get("path") != native_receipt.get("path")
        or extension.get("size_bytes") != native_receipt.get("bytes")
        or extension.get("sha256") != native_receipt.get("sha256")
        or any(
            not _is_sha256(native.get(key))
            for key in (
                "release_wrapper_source_sha256",
                "release_loop_source_sha256",
                "release_base_loop_source_sha256",
            )
        )
    ):
        raise Stage1QualificationError("restart native runtime cross-binding changed")
    if expected_main_runtime_identity is not None:
        expected_native = (
            expected_main_runtime_identity.get(
                "native_wisdom_holman_runtime_identity"
            )
            if type(expected_main_runtime_identity) is dict
            else None
        )
        if _canonical_json(native) != _canonical_json(expected_native):
            raise Stage1QualificationError(
                "restart native runtime differs from the main worker runtime"
            )
        _require_numpy_runtime_cross_binding(
            value.get("numpy_runtime_identity"),
            expected_main_runtime_identity.get("numpy_runtime_identity"),
            "restart NumPy runtime differs from the main worker runtime",
        )
    cupy = value.get("cupy_cuda")
    fused = fused_tracer_runtime_identity
    if (
        type(cupy) is not dict
        or set(cupy) != {
            "cupy_version", "cupy_module", "cupy_core_module",
            "cupy_fft_config_module", "cuda_runtime_version", "cuda_local_runtime_version",
            "cuda_driver_version", "cuda_device_id", "cuda_device_name",
            "cuda_device_uuid", "cuda_compute_capability",
            "cuda_pci_domain_bus_device",
        }
        or cupy.get("cupy_version") != CUPY_DISTRIBUTION_VERSION
        or cupy.get("cuda_device_id") != 0
        or any(
            type(cupy.get(key)) is not int or cupy.get(key) <= 0
            for key in (
                "cuda_runtime_version", "cuda_local_runtime_version",
                "cuda_driver_version",
            )
        )
        or any(
            type(cupy.get(key)) is not str or not cupy.get(key)
            for key in (
                "cuda_device_name", "cuda_device_uuid",
                "cuda_compute_capability", "cuda_pci_domain_bus_device",
            )
        )
        or type(fused) is not dict
        or cupy.get("cupy_version") != fused.get("cupy_version")
        or cupy.get("cuda_runtime_version")
        != fused.get("cuda_runtime_version")
        or cupy.get("cuda_local_runtime_version")
        != fused.get("cuda_local_runtime_version")
        or cupy.get("cuda_driver_version")
        != fused.get("cuda_driver_version")
        or cupy.get("cuda_device_id") != fused.get("device_index")
        or cupy.get("cuda_device_name") != fused.get("device_name")
        or cupy.get("cuda_device_uuid") != fused.get("device_uuid")
        or cupy.get("cuda_compute_capability")
        != fused.get("device_compute_capability")
        or cupy.get("cuda_pci_domain_bus_device")
        != fused.get("device_pci_domain_bus_device")
    ):
        raise Stage1QualificationError("restart CuPy/CUDA identity changed")
    module_keys = {
        "module", "spec_name", "spec_origin", "loader_type", "file_kind",
        "cached_path", "path", "bytes", "sha256",
    }
    for key, module_name, relative, size, digest in (
        (
            "cupy_module", "cupy", CUPY_INIT_RELATIVE_PATH,
            CUPY_INIT_BYTES, CUPY_INIT_SHA256,
        ),
        (
            "cupy_core_module", "cupy._core.core", CUPY_CORE_RELATIVE_PATH,
            CUPY_CORE_BYTES, CUPY_CORE_SHA256,
        ),
        (
            "cupy_fft_config_module", "cupy.fft._config",
            CUPY_FFT_CONFIG_RELATIVE_PATH, CUPY_FFT_CONFIG_BYTES,
            CUPY_FFT_CONFIG_SHA256,
        ),
    ):
        receipt = cupy.get(key)
        if (
            type(receipt) is not dict
            or set(receipt) != module_keys
            or receipt.get("module") != module_name
            or receipt.get("path")
            != str((CUPY_SITE_PACKAGES / relative).resolve(strict=True))
            or receipt.get("bytes") != size
            or receipt.get("sha256") != digest
            or registry.get(module_name) != {
                **receipt,
                "predeclared_authority": "PINNED_CUPY_RECORD",
            }
        ):
            raise Stage1QualificationError(
                "restart CuPy module receipt changed"
            )
        _validate_runtime_file_receipt(
            {field: receipt[field] for field in ("path", "bytes", "sha256")},
            f"restart runtime module {module_name}",
        )


def _runtime_identity(
    rebound: Any,
    cp: Any,
    rebound_target_receipt: Mapping[str, Any],
    import_guard: Mapping[str, Any],
    fresh_registry_receipt: Mapping[str, Any],
    fresh_child_interpreter: Mapping[str, Any],
    import_probe_receipt: Mapping[str, Any],
) -> dict[str, object]:
    if getattr(rebound, "__version__", None) != REQUIRED_REBOUND_VERSION:
        raise Stage1QualificationError("REBOUND must be exactly 5.1.1")
    module_path = Path(str(getattr(rebound, "__file__", ""))).resolve(strict=True)
    library_paths: set[Path] = set()
    declared = getattr(rebound, "__libpath__", None)
    if type(declared) is str:
        library_paths.add(Path(declared).resolve(strict=True))
    library_paths.update(path.resolve() for path in module_path.parent.glob("librebound*") if path.is_file())
    target = module_path.parent.parent
    distributions = [
        item
        for item in importlib.metadata.distributions(path=[str(target)])
        if (item.metadata.get("Name") or "").lower() == "rebound"
    ]
    if len(distributions) != 1:
        raise Stage1QualificationError(
            "REBOUND target must contain exactly one distribution"
        )
    try:
        distribution_version = distributions[0].version
    except (KeyError, TypeError) as exc:
        raise Stage1QualificationError("REBOUND distribution metadata is missing") from exc
    if distribution_version != REQUIRED_REBOUND_VERSION:
        raise Stage1QualificationError("REBOUND distribution version changed")
    executable = Path(sys.executable).resolve(strict=True)
    python_receipt = _exact_live_file_receipt(
        executable,
        "runtime Python executable",
        expected_path=PYTHON_EXECUTABLE_PATH,
        expected_bytes=PYTHON_EXECUTABLE_BYTES,
        expected_sha256=PYTHON_EXECUTABLE_SHA256,
        maximum=20_000_000,
    )
    jx_modules = (
        ("jxplanetx.restricted_wh_source", STATIC_BINDINGS[13][0], STATIC_BINDINGS[13][1]),
        ("jxplanetx.cuda_restricted_wh", STATIC_BINDINGS[14][0], STATIC_BINDINGS[14][1]),
        ("jxplanetx._all_conic_kepler", STATIC_BINDINGS[15][0], STATIC_BINDINGS[15][1]),
        ("jxplanetx.fast_wisdom_holman", STATIC_BINDINGS[16][0], STATIC_BINDINGS[16][1]),
        ("jxplanetx.engine.contracts", STATIC_BINDINGS[17][0], STATIC_BINDINGS[17][1]),
        ("jxplanetx.engine.backends", STATIC_BINDINGS[18][0], STATIC_BINDINGS[18][1]),
    )
    jx_receipts: list[dict[str, object]] = []
    for module_name, relative_path, expected_sha256 in jx_modules:
        if module_name not in sys.modules:
            raise Stage1QualificationError(
                f"required JX runtime module was not actually used: {module_name}"
            )
        receipt = _imported_module_runtime_file(module_name)
        if (
            Path(str(receipt["path"])) != (PROJECT_ROOT / relative_path).resolve()
            or receipt["sha256"] != expected_sha256
        ):
            raise Stage1QualificationError(f"imported JX module changed: {module_name}")
        jx_receipts.append(receipt)
    native_receipt = _imported_module_runtime_file(
        "jxplanetx._wisdom_holman_cpu_v3"
    )
    fast_module = sys.modules.get("jxplanetx.fast_wisdom_holman")
    runtime_function = getattr(fast_module, "fast_wisdom_holman_runtime_identity", None)
    if not callable(runtime_function):
        raise Stage1QualificationError("native Wisdom-Holman identity API is missing")
    native_identity = json.loads(
        _canonical_json(runtime_function()).decode("ascii")
    )
    if (
        type(native_identity.get("extension_binary")) is not dict
        or native_identity["extension_binary"].get("sha256")
        != native_receipt["sha256"]
        or Path(str(native_identity["extension_binary"].get("path"))).resolve()
        != Path(str(native_receipt["path"]))
    ):
        raise Stage1QualificationError("native Wisdom-Holman receipt did not cross-bind")
    cupy_module = _imported_module_runtime_file("cupy")
    if "cupy._core.core" not in sys.modules:
        raise Stage1QualificationError("CuPy core extension was not actually used")
    cupy_core_receipt = _imported_module_runtime_file("cupy._core.core")
    cupy_fft_config_receipt = _imported_module_runtime_file(
        "cupy.fft._config"
    )
    numpy_identity = _numpy_runtime_identity()
    runtime_modules = _runtime_module_registry_receipts(
        rebound_target_receipt, require_rebound=True
    )
    properties = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))
    raw_uuid = properties.get("uuid")
    if raw_uuid is None:
        raise Stage1QualificationError("CUDA device UUID is unavailable")
    return {
        "schema": "jx.tno.stage1-dynamics-qualification.runtime-identity.v2",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "optimization_level": sys.flags.optimize,
            "executable": python_receipt,
        },
        "fresh_child_interpreter": json.loads(
            _canonical_json(dict(fresh_child_interpreter)).decode("ascii")
        ),
        "import_only_preflight": json.loads(
            _canonical_json(dict(import_probe_receipt)).decode("ascii")
        ),
        "host": {"platform": platform.platform(), "machine": platform.machine()},
        "stage1_implementation": _runtime_file(SCRIPT_PATH, "Stage-1 implementation"),
        "bound_sources": _verify_static_bindings(),
        "rebound": {
            "version": rebound.__version__,
            "distribution_version": distribution_version,
            "module": _runtime_file(module_path, "REBOUND module"),
            "shared_libraries": [
                _runtime_file(path, "REBOUND shared library")
                for path in sorted(library_paths)
            ],
            "target_wheel_verification": json.loads(
                _canonical_json(dict(rebound_target_receipt)).decode("ascii")
            ),
        },
        "jx_imported_modules": jx_receipts,
        "native_wisdom_holman_module": native_receipt,
        "native_wisdom_holman_runtime_identity": native_identity,
        "runtime_module_registry": runtime_modules,
        "numpy_runtime_identity": numpy_identity,
        "cupy_cuda": {
            "cupy_version": str(cp.__version__),
            "cupy_module": cupy_module,
            "cupy_core_module": cupy_core_receipt,
            "cupy_fft_config_module": cupy_fft_config_receipt,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_local_runtime_version": int(cp.cuda.get_local_runtime_version()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_device_id": int(cp.cuda.Device().id),
            "cuda_device_name": _cuda_device_name(cp),
            "cuda_device_uuid": bytes(raw_uuid).hex(),
            "cuda_compute_capability": f"{int(properties['major'])}.{int(properties['minor'])}",
            "cuda_pci_domain_bus_device": (
                f"{int(properties['pciDomainID']):04x}:"
                f"{int(properties['pciBusID']):02x}:"
                f"{int(properties['pciDeviceID']):02x}"
            ),
        },
        "fresh_process_registry": json.loads(
            _canonical_json(dict(fresh_registry_receipt)).decode("ascii")
        ),
        "source_only_runtime_import_guard": import_guard,
        "claim": "TRUSTED_FRESH_PROCESS_AND_EXACT_LIVE_FILE_REREAD_OBSERVATIONS_NOT_RUNTIME_AUTHENTICATION",
    }


def _cuda_device_name(cp: Any) -> str:
    value = cp.cuda.runtime.getDeviceProperties(int(cp.cuda.Device().id))["name"]
    if type(value) is bytes:
        return value.decode("utf-8", errors="strict")
    if type(value) is str and value:
        return value
    raise Stage1QualificationError("CUDA device name has an unexpected type")


def _validate_runtime_file_receipt(
    value: object, label: str, *, allow_empty: bool = False
) -> None:
    if (
        type(value) is not dict
        or set(value) != {"path", "bytes", "sha256"}
        or type(value.get("path")) is not str
        or not Path(value["path"]).is_absolute()
        or type(value.get("bytes")) is not int
        or value.get("bytes") < 0
        or (value.get("bytes") == 0 and not allow_empty)
        or (
            value.get("bytes") == 0
            and value.get("sha256") != EMPTY_SHA256
        )
        or not _is_sha256(value.get("sha256"))
    ):
        raise Stage1QualificationError(f"{label} runtime file receipt changed")
    live = _exact_live_file_receipt(
        Path(value["path"]), label, maximum=200_000_000,
        allow_empty=allow_empty,
    )
    if live != value:
        raise Stage1QualificationError(f"{label} no longer matches live bytes")


def _validate_runtime_module_registry(
    value: object,
    rebound_target_receipt: Mapping[str, Any] | None,
    *,
    require_rebound: bool,
) -> None:
    if type(value) is not list or not value:
        raise Stage1QualificationError("runtime module registry is empty")
    expected, rebound_expected = _runtime_expected_file_map(rebound_target_receipt)
    observed_names: list[str] = []
    observed_paths: set[str] = set()
    keys = {
        "module", "spec_name", "spec_origin", "loader_type", "file_kind",
        "cached_path", "path", "bytes", "sha256", "predeclared_authority",
    }
    for receipt in value:
        if type(receipt) is not dict or set(receipt) != keys:
            raise Stage1QualificationError("runtime module receipt schema changed")
        name = receipt.get("module")
        path = receipt.get("path")
        if (
            type(name) is not str
            or not any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in RUNTIME_MODULE_PREFIXES
            )
            or receipt.get("spec_name") != name
            or type(path) is not str
            or receipt.get("spec_origin") != path
            or type(receipt.get("loader_type")) is not str
            or not receipt.get("loader_type")
            or receipt.get("file_kind") not in ("PYTHON_SOURCE", "EXTENSION_BINARY")
            or (
                receipt.get("cached_path") is not None
                and type(receipt.get("cached_path")) is not str
            )
        ):
            raise Stage1QualificationError("runtime module spec/origin receipt changed")
        expected_item = (
            rebound_expected.get(path)
            if name == "rebound" or name.startswith("rebound.")
            else expected.get(path)
        )
        if expected_item is None:
            raise Stage1QualificationError("runtime module path is not predeclared")
        size, digest, authority = expected_item
        if (
            receipt.get("bytes") != size
            or receipt.get("sha256") != digest
            or receipt.get("predeclared_authority") != authority
        ):
            raise Stage1QualificationError("runtime module bytes differ from authority")
        basic = {key: receipt[key] for key in ("path", "bytes", "sha256")}
        allow_empty = size == 0 and digest == EMPTY_SHA256
        if size == 0 and not allow_empty:
            raise Stage1QualificationError(
                "runtime module has an invalid zero-byte authority"
            )
        _validate_runtime_file_receipt(
            basic, f"runtime module {name}", allow_empty=allow_empty
        )
        observed_names.append(name)
        observed_paths.add(path)
    if observed_names != sorted(observed_names) or len(observed_names) != len(set(observed_names)):
        raise Stage1QualificationError("runtime module registry order/uniqueness changed")
    for prefix in ("jxplanetx", "numpy", "cupy", "cupy_backends"):
        if not any(name == prefix or name.startswith(prefix + ".") for name in observed_names):
            raise Stage1QualificationError(f"runtime registry lacks prefix: {prefix}")
    rebound_names = [
        name for name in observed_names
        if name == "rebound" or name.startswith("rebound.")
    ]
    if require_rebound != bool(rebound_names):
        raise Stage1QualificationError("runtime REBOUND registry presence changed")
    required_paths = {
        str((PROJECT_ROOT / JX_NATIVE_RELATIVE_PATH).resolve(strict=True)),
        str((CUPY_SITE_PACKAGES / CUPY_CORE_RELATIVE_PATH).resolve(strict=True)),
        str((CUPY_SITE_PACKAGES / CUPY_FFT_CONFIG_RELATIVE_PATH).resolve(strict=True)),
        str((CUPY_SITE_PACKAGES / NUMPY_CORE_RELATIVE_PATH).resolve(strict=True)),
    }
    if not required_paths <= observed_paths:
        raise Stage1QualificationError("required native runtime modules are absent")
    for required_name in REQUIRED_RUNTIME_MODULE_NAMES:
        if required_name not in observed_names:
            raise Stage1QualificationError(
                f"required exact runtime module is absent: {required_name}"
            )


def _validate_runtime_identity(value: object) -> None:
    _guard_exact_json_integer_identities(value, "runtime identity")
    expected_keys = {
        "schema", "python", "fresh_child_interpreter",
        "import_only_preflight", "host", "stage1_implementation",
        "bound_sources", "rebound", "jx_imported_modules",
        "native_wisdom_holman_module",
        "native_wisdom_holman_runtime_identity", "runtime_module_registry",
        "numpy_runtime_identity", "cupy_cuda", "fresh_process_registry",
        "source_only_runtime_import_guard", "claim",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema")
        != "jx.tno.stage1-dynamics-qualification.runtime-identity.v2"
        or value.get("claim")
        != "TRUSTED_FRESH_PROCESS_AND_EXACT_LIVE_FILE_REREAD_OBSERVATIONS_NOT_RUNTIME_AUTHENTICATION"
    ):
        raise Stage1QualificationError("runtime identity schema changed")
    python = value.get("python")
    host = value.get("host")
    if (
        type(python) is not dict
        or set(python) != {
            "version", "implementation", "optimization_level", "executable"
        }
        or type(python.get("version")) is not str
        or not python.get("version")
        or type(python.get("implementation")) is not str
        or not python.get("implementation")
        or type(python.get("optimization_level")) is not int
        or python.get("optimization_level") not in (0, 1, 2)
        or type(host) is not dict
        or set(host) != {"platform", "machine"}
        or any(type(host.get(key)) is not str or not host.get(key) for key in host)
    ):
        raise Stage1QualificationError("Python or host runtime identity changed")
    _validate_runtime_file_receipt(python["executable"], "Python executable")
    _validate_fresh_child_interpreter_receipt(
        value.get("fresh_child_interpreter")
    )
    _validate_import_probe_receipt(value.get("import_only_preflight"))
    if python["executable"] != {
        "path": str(PYTHON_EXECUTABLE_PATH.resolve(strict=True)),
        "bytes": PYTHON_EXECUTABLE_BYTES,
        "sha256": PYTHON_EXECUTABLE_SHA256,
    }:
        raise Stage1QualificationError("Python executable is not the predeclared binary")
    implementation = value.get("stage1_implementation")
    _validate_runtime_file_receipt(implementation, "Stage-1 implementation")
    implementation_payload = _regular_bytes(
        SCRIPT_PATH, "Stage-1 implementation"
    )
    if (
        implementation.get("path") != str(SCRIPT_PATH.resolve())
        or implementation.get("bytes") != len(implementation_payload)
        or implementation.get("sha256") != _sha256(implementation_payload)
    ):
        raise Stage1QualificationError("runtime Stage-1 implementation changed")
    if _canonical_json(value.get("bound_sources")) != _canonical_json(
        _verify_static_bindings()
    ):
        raise Stage1QualificationError("runtime bound-source roster changed")

    rebound = value.get("rebound")
    if (
        type(rebound) is not dict
        or set(rebound) != {
            "version", "distribution_version", "module", "shared_libraries",
            "target_wheel_verification",
        }
        or rebound.get("version") != REQUIRED_REBOUND_VERSION
        or rebound.get("distribution_version") != REQUIRED_REBOUND_VERSION
        or type(rebound.get("shared_libraries")) is not list
        or len(rebound["shared_libraries"]) != 1
    ):
        raise Stage1QualificationError("REBOUND runtime identity changed")
    _validate_runtime_file_receipt(rebound["module"], "REBOUND module")
    _validate_runtime_file_receipt(
        rebound["shared_libraries"][0], "REBOUND shared library"
    )
    if (
        rebound["module"].get("sha256") != REBOUND_INIT_SHA256
        or rebound["shared_libraries"][0].get("sha256")
        != REBOUND_LIBRARY_SHA256
    ):
        raise Stage1QualificationError("REBOUND runtime hashes changed")
    target_receipt = rebound.get("target_wheel_verification")
    if (
        type(target_receipt) is not dict
        or set(target_receipt) != {
            "target", "wheel_sha256", "verified_executable_member_count",
            "members",
        }
        or type(target_receipt.get("target")) is not str
        or not Path(target_receipt["target"]).is_absolute()
        or target_receipt.get("wheel_sha256") != REBOUND_WHEEL_SHA256
        or type(target_receipt.get("verified_executable_member_count")) is not int
        or type(target_receipt.get("members")) is not list
        or target_receipt.get("verified_executable_member_count")
        != len(target_receipt["members"])
    ):
        raise Stage1QualificationError("REBOUND wheel runtime receipt changed")
    for member in target_receipt["members"]:
        if (
            type(member) is not dict
            or set(member) != {"wheel_member", "installed_path", "bytes", "sha256"}
            or type(member.get("wheel_member")) is not str
            or type(member.get("installed_path")) is not str
            or not Path(member["installed_path"]).is_absolute()
            or type(member.get("bytes")) is not int
            or member.get("bytes") <= 0
            or not _is_sha256(member.get("sha256"))
        ):
            raise Stage1QualificationError("REBOUND wheel member receipt changed")
    try:
        with zipfile.ZipFile(REBOUND_WHEEL_PATH, "r") as archive:
            expected_members = []
            for name in archive.namelist():
                if name.endswith("/") or not (
                    (name.startswith("rebound/") and name.endswith(".py"))
                    or (name.startswith("librebound") and name.endswith(".so"))
                ):
                    continue
                payload = archive.read(name)
                expected_members.append(
                    {
                        "wheel_member": name,
                        "installed_path": str(
                            Path(target_receipt["target"]).joinpath(*Path(name).parts)
                        ),
                        "bytes": len(payload),
                        "sha256": _sha256(payload),
                    }
                )
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise Stage1QualificationError(
            "retained REBOUND wheel could not be reconstructed offline"
        ) from exc
    if _canonical_json(target_receipt["members"]) != _canonical_json(
        expected_members
    ):
        raise Stage1QualificationError(
            "REBOUND runtime member receipts differ from the retained wheel"
        )
    live_target = _verify_rebound_target(Path(target_receipt["target"]))
    if _canonical_json(live_target) != _canonical_json(target_receipt):
        raise Stage1QualificationError("REBOUND target no longer matches the wheel")
    member_by_path = {
        item["installed_path"]: item for item in target_receipt["members"]
    }
    for file_receipt in (rebound["module"], *rebound["shared_libraries"]):
        member = member_by_path.get(file_receipt["path"])
        if (
            type(member) is not dict
            or file_receipt["bytes"] != member["bytes"]
            or file_receipt["sha256"] != member["sha256"]
        ):
            raise Stage1QualificationError("REBOUND live file is not a wheel member")

    _validate_runtime_module_registry(
        value.get("runtime_module_registry"), target_receipt, require_rebound=True
    )
    registry_by_name = {
        item["module"]: item for item in value["runtime_module_registry"]
    }
    _validate_numpy_runtime_identity(
        value.get("numpy_runtime_identity"), registry_by_name
    )

    expected_jx = (
        ("jxplanetx.restricted_wh_source", STATIC_BINDINGS[13][0], STATIC_BINDINGS[13][1]),
        ("jxplanetx.cuda_restricted_wh", STATIC_BINDINGS[14][0], STATIC_BINDINGS[14][1]),
        ("jxplanetx._all_conic_kepler", STATIC_BINDINGS[15][0], STATIC_BINDINGS[15][1]),
        ("jxplanetx.fast_wisdom_holman", STATIC_BINDINGS[16][0], STATIC_BINDINGS[16][1]),
        ("jxplanetx.engine.contracts", STATIC_BINDINGS[17][0], STATIC_BINDINGS[17][1]),
        ("jxplanetx.engine.backends", STATIC_BINDINGS[18][0], STATIC_BINDINGS[18][1]),
    )
    jx_receipts = value.get("jx_imported_modules")
    if type(jx_receipts) is not list or len(jx_receipts) != len(expected_jx):
        raise Stage1QualificationError("imported JX runtime roster changed")
    for receipt, (module_name, relative_path, expected_sha256) in zip(
        jx_receipts, expected_jx, strict=True
    ):
        module_keys = {
            "module", "spec_name", "spec_origin", "loader_type", "file_kind",
            "cached_path", "path", "bytes", "sha256",
        }
        if type(receipt) is not dict or set(receipt) != module_keys:
            raise Stage1QualificationError("imported JX module receipt shape changed")
        _validate_runtime_file_receipt(
            {key: receipt[key] for key in ("path", "bytes", "sha256")},
            module_name,
        )
        if (
            receipt.get("module") != module_name
            or receipt.get("path") != str((PROJECT_ROOT / relative_path).resolve())
            or receipt.get("sha256") != expected_sha256
            or registry_by_name.get(module_name)
            != {**receipt, "predeclared_authority": "JX_PREDECLARED_MANIFEST"}
        ):
            raise Stage1QualificationError("imported JX module receipt changed")
    native_module = value.get("native_wisdom_holman_module")
    if type(native_module) is not dict or set(native_module) != module_keys:
        raise Stage1QualificationError("native Wisdom-Holman module receipt changed")
    _validate_runtime_file_receipt(
        {key: native_module[key] for key in ("path", "bytes", "sha256")},
        "native Wisdom-Holman module",
    )
    if (
        native_module.get("module") != "jxplanetx._wisdom_holman_cpu_v3"
        or native_module.get("path")
        != str((PROJECT_ROOT / JX_NATIVE_RELATIVE_PATH).resolve(strict=True))
        or native_module.get("bytes") != JX_NATIVE_BYTES
        or native_module.get("sha256") != JX_NATIVE_SHA256
        or registry_by_name.get("jxplanetx._wisdom_holman_cpu_v3")
        != {**native_module, "predeclared_authority": "JX_PREDECLARED_MANIFEST"}
    ):
        raise Stage1QualificationError("native Wisdom-Holman module name changed")
    native_identity = value.get("native_wisdom_holman_runtime_identity")
    expected_native_keys = {
        "kernel_id", "g_kernel_id", "exact_fallback_kernel_id",
        "scientific_claim_state", "maximum_bodies", "metric_count",
        "counter_count", "compiler_version", "compiler_flags",
        "release_wrapper_source_sha256", "release_loop_source_sha256",
        "release_base_loop_source_sha256", "extension_binary",
    }
    extension = native_identity.get("extension_binary") if type(native_identity) is dict else None
    if (
        type(native_identity) is not dict
        or set(native_identity) != expected_native_keys
        or type(extension) is not dict
        or set(extension) != {"path", "size_bytes", "sha256"}
        or extension.get("path") != native_module.get("path")
        or extension.get("size_bytes") != native_module.get("bytes")
        or extension.get("sha256") != native_module.get("sha256")
        or any(
            not _is_sha256(native_identity.get(key))
            for key in (
                "release_wrapper_source_sha256",
                "release_loop_source_sha256",
                "release_base_loop_source_sha256",
            )
        )
    ):
        raise Stage1QualificationError("native Wisdom-Holman identity changed")

    cupy = value.get("cupy_cuda")
    if (
        type(cupy) is not dict
        or set(cupy) != {
            "cupy_version", "cupy_module", "cupy_core_module",
            "cupy_fft_config_module",
            "cuda_runtime_version", "cuda_local_runtime_version",
            "cuda_driver_version", "cuda_device_id", "cuda_device_name",
            "cuda_device_uuid", "cuda_compute_capability",
            "cuda_pci_domain_bus_device",
        }
        or cupy.get("cupy_version") != CUPY_DISTRIBUTION_VERSION
        or cupy.get("cuda_device_id") != 0
        or any(
            type(cupy.get(key)) is not int or cupy.get(key) <= 0
            for key in (
                "cuda_runtime_version", "cuda_local_runtime_version",
                "cuda_driver_version",
            )
        )
        or any(
            type(cupy.get(key)) is not str or not cupy.get(key)
            for key in (
                "cuda_device_name", "cuda_device_uuid",
                "cuda_compute_capability", "cuda_pci_domain_bus_device",
            )
        )
    ):
        raise Stage1QualificationError("CuPy/CUDA runtime identity changed")
    for key, module_name, label in (
        ("cupy_module", "cupy", "CuPy module"),
        ("cupy_core_module", "cupy._core.core", "CuPy core module"),
        (
            "cupy_fft_config_module", "cupy.fft._config",
            "CuPy FFT configuration module",
        ),
    ):
        receipt = cupy[key]
        if type(receipt) is not dict or set(receipt) != module_keys or receipt.get("module") != module_name:
            raise Stage1QualificationError(f"{label} receipt shape changed")
        _validate_runtime_file_receipt(
            {item: receipt[item] for item in ("path", "bytes", "sha256")},
            label,
        )
        if registry_by_name.get(module_name) != {
            **receipt,
            "predeclared_authority": "PINNED_CUPY_RECORD",
        }:
            raise Stage1QualificationError(f"{label} does not match the full registry")
    if (
        cupy["cupy_module"].get("path")
        != str((CUPY_SITE_PACKAGES / CUPY_INIT_RELATIVE_PATH).resolve(strict=True))
        or cupy["cupy_module"].get("bytes") != CUPY_INIT_BYTES
        or cupy["cupy_module"].get("sha256") != CUPY_INIT_SHA256
        or cupy["cupy_core_module"].get("path")
        != str((CUPY_SITE_PACKAGES / CUPY_CORE_RELATIVE_PATH).resolve(strict=True))
        or cupy["cupy_core_module"].get("bytes") != CUPY_CORE_BYTES
        or cupy["cupy_core_module"].get("sha256") != CUPY_CORE_SHA256
        or cupy["cupy_fft_config_module"].get("path")
        != str(
            (CUPY_SITE_PACKAGES / CUPY_FFT_CONFIG_RELATIVE_PATH).resolve(
                strict=True
            )
        )
        or cupy["cupy_fft_config_module"].get("bytes")
        != CUPY_FFT_CONFIG_BYTES
        or cupy["cupy_fft_config_module"].get("sha256")
        != CUPY_FFT_CONFIG_SHA256
    ):
        raise Stage1QualificationError("CuPy init/core files differ from predeclared receipts")
    preflight = value.get("import_only_preflight")
    _require_numpy_runtime_cross_binding(
        preflight.get("numpy_runtime_identity"),
        value.get("numpy_runtime_identity"),
        "import-only preflight differs from the dynamics-worker NumPy runtime",
    )
    if (
        _canonical_json(
            _module_runtime_cross_binding(
                preflight.get("cupy_import_identity", {}).get("module")
            )
        )
        != _canonical_json(
            _module_runtime_cross_binding(cupy.get("cupy_module"))
        )
        or _canonical_json(
            _module_runtime_cross_binding(
                preflight.get("cupy_import_identity", {}).get("core_module")
            )
        )
        != _canonical_json(
            _module_runtime_cross_binding(cupy.get("cupy_core_module"))
        )
        or _canonical_json(
            _module_runtime_cross_binding(
                preflight.get("cupy_import_identity", {}).get(
                    "fft_config_module"
                )
            )
        )
        != _canonical_json(
            _module_runtime_cross_binding(cupy.get("cupy_fft_config_module"))
        )
    ):
        raise Stage1QualificationError(
            "import-only preflight differs from the dynamics-worker runtime"
        )
    fresh = value.get("fresh_process_registry")
    if fresh != {
        "checked_prefixes": list(RUNTIME_MODULE_PREFIXES),
        "preloaded_module_names": [],
        "fresh_registry_observed": True,
        "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
    }:
        raise Stage1QualificationError("fresh-process registry receipt changed")
    guard = value.get("source_only_runtime_import_guard")
    _validate_runtime_import_guard(guard, "main dynamics worker")


def _verify_rebound_target(target: Path) -> dict[str, object]:
    target = Path(os.path.abspath(os.fspath(target)))
    if not target.is_dir() or target.is_symlink():
        raise Stage1QualificationError("REBOUND target must be a real directory")
    wheel = _regular_bytes(REBOUND_WHEEL_PATH, "retained REBOUND wheel", maximum=10_000_000)
    if _sha256(wheel) != REBOUND_WHEEL_SHA256:
        raise Stage1QualificationError("retained REBOUND wheel changed")
    try:
        with zipfile.ZipFile(REBOUND_WHEEL_PATH, "r") as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise Stage1QualificationError("REBOUND wheel contains duplicate members")
            checked: list[dict[str, object]] = []
            for name in names:
                if name.endswith("/") or not (
                    (name.startswith("rebound/") and name.endswith(".py"))
                    or (name.startswith("librebound") and name.endswith(".so"))
                ):
                    continue
                member = Path(name)
                if member.is_absolute() or ".." in member.parts or "\\" in name:
                    raise Stage1QualificationError("REBOUND wheel member path is unsafe")
                expected_payload = archive.read(name)
                installed_path = target.joinpath(*member.parts)
                actual_payload = _regular_bytes(installed_path, f"REBOUND runtime member {name}", maximum=20_000_000)
                if actual_payload != expected_payload:
                    raise Stage1QualificationError(f"REBOUND runtime member differs from wheel: {name}")
                checked.append(
                    {
                        "wheel_member": name,
                        "installed_path": str(installed_path),
                        "bytes": len(actual_payload),
                        "sha256": _sha256(actual_payload),
                    }
                )
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise Stage1QualificationError("cannot verify retained REBOUND wheel") from exc
    observed = {item["wheel_member"]: item["sha256"] for item in checked}
    library_names = [name for name in observed if str(name).startswith("librebound")]
    if (
        observed.get("rebound/__init__.py") != REBOUND_INIT_SHA256
        or len(library_names) != 1
        or observed[library_names[0]] != REBOUND_LIBRARY_SHA256
    ):
        raise Stage1QualificationError("REBOUND wheel member identities changed")
    return {
        "target": str(target),
        "wheel_sha256": REBOUND_WHEEL_SHA256,
        "verified_executable_member_count": len(checked),
        "members": checked,
    }


def _load_dynamics_runtime(
    rebound_target: Path = DEFAULT_REBOUND_TARGET,
    *,
    require_cuda_device: bool = True,
) -> tuple[Any, Any, dict[str, object]]:
    target_receipt = _verify_rebound_target(rebound_target)
    try:
        import numpy as np
        import cupy as cp
        importlib.import_module("numpy._core._multiarray_umath")
        importlib.import_module("cupy._core.core")
        importlib.import_module("cupy.fft._config")
        for module_name in (
            "jxplanetx.restricted_wh_source",
            "jxplanetx.cuda_restricted_wh",
            "jxplanetx._all_conic_kepler",
            "jxplanetx.fast_wisdom_holman",
            "jxplanetx.engine.contracts",
            "jxplanetx.engine.backends",
            "jxplanetx._wisdom_holman_cpu_v3",
        ):
            importlib.import_module(module_name)
    except Exception as exc:
        raise Stage1QualificationError(
            "required exact NumPy/CuPy/JX runtime is unavailable"
        ) from exc
    if getattr(np, "__version__", None) != NUMPY_DISTRIBUTION_VERSION:
        raise Stage1QualificationError("NumPy must be exactly 2.5.3")
    if require_cuda_device and int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise Stage1QualificationError("a CUDA device is required for the JX lanes")
    if "rebound" in sys.modules:
        raise Stage1QualificationError("REBOUND was imported before the isolated runtime boundary")
    target = Path(str(target_receipt["target"]))
    previous_prefix = sys.pycache_prefix
    previous_dont_write = sys.dont_write_bytecode
    previous_path = list(sys.path)
    try:
        # An empty alternate cache prevents a timestamp-valid target __pycache__
        # from replacing the exact wheel-matched sources checked above.
        cache_root = target / ".stage1-v2-source-only-pycache-must-be-absent"
        if cache_root.exists() or cache_root.is_symlink():
            raise Stage1QualificationError("isolated REBOUND bytecode cache path must be absent")
        sys.pycache_prefix = str(cache_root)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(target))
        rebound = importlib.import_module("rebound")
    except Stage1QualificationError:
        raise
    except Exception as exc:
        raise Stage1QualificationError("isolated REBOUND import failed") from exc
    finally:
        sys.pycache_prefix = previous_prefix
        sys.dont_write_bytecode = previous_dont_write
        sys.path[:] = previous_path
    if getattr(rebound, "__version__", None) != REQUIRED_REBOUND_VERSION:
        raise Stage1QualificationError("REBOUND must be exactly 5.1.1")
    module_path = Path(str(getattr(rebound, "__file__", ""))).resolve(strict=True)
    if target not in module_path.parents or _sha256(_regular_bytes(module_path, "imported REBOUND module")) != REBOUND_INIT_SHA256:
        raise Stage1QualificationError("imported REBOUND module is not the verified target")
    return cp, rebound, target_receipt


def _production_child_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _import_probe_worker(
    rebound_target: Path,
    output_path: Path,
) -> dict[str, object]:
    fresh_child_interpreter = _fresh_child_interpreter_receipt()
    fresh_registry_receipt = _reject_preloaded_runtime_modules()
    output = Path(os.path.abspath(os.fspath(output_path)))
    if not output.parent.is_dir() or output.exists() or output.is_symlink():
        raise Stage1QualificationError(
            "import-only probe output must be an absent file in an existing directory"
        )
    with _source_only_runtime_import_guard(
        output.parent / ".import-probe-pycache-must-remain-absent"
    ) as import_guard:
        cp, _rebound, target_receipt = _load_dynamics_runtime(
            rebound_target, require_cuda_device=False
        )
        registry = _runtime_module_registry_receipts(
            target_receipt, require_rebound=True
        )
        receipt: dict[str, object] = {
            "schema": IMPORT_PROBE_SCHEMA,
            "probe_output_path": str(output),
            "fresh_child_interpreter": fresh_child_interpreter,
            "fresh_process_registry": fresh_registry_receipt,
            "stage1_implementation": _runtime_file(
                SCRIPT_PATH, "import-probe Stage-1 implementation"
            ),
            "rebound_target_verification": target_receipt,
            "runtime_module_registry": registry,
            "numpy_runtime_identity": _numpy_runtime_identity(),
            "cupy_import_identity": {
                "version": str(cp.__version__),
                "module": _imported_module_runtime_file("cupy"),
                "core_module": _imported_module_runtime_file("cupy._core.core"),
                "fft_config_module": _imported_module_runtime_file(
                    "cupy.fft._config"
                ),
            },
            "source_only_runtime_import_guard": import_guard,
            "cuda_device_queried": False,
            "dynamics_executed": False,
            "claim": "EXACT_IMPORT_ONLY_FRESH_CHILD_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION",
            "content_sha256": "",
        }
    receipt["content_sha256"] = _content_sha256(
        IMPORT_PROBE_DOMAIN, receipt
    )
    _validate_import_probe_receipt(receipt, expected_path=output)
    _write_create_only(output, _canonical_json(receipt))
    return receipt


def _validate_import_probe_receipt(
    value: object, *, expected_path: Path | None = None
) -> None:
    _guard_exact_json_integer_identities(value, "import-only preflight")
    expected_keys = {
        "schema", "probe_output_path", "fresh_child_interpreter", "fresh_process_registry",
        "stage1_implementation", "rebound_target_verification",
        "runtime_module_registry", "numpy_runtime_identity",
        "cupy_import_identity", "source_only_runtime_import_guard",
        "cuda_device_queried", "dynamics_executed", "claim",
        "content_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema") != IMPORT_PROBE_SCHEMA
        or type(value.get("probe_output_path")) is not str
        or not Path(value["probe_output_path"]).is_absolute()
        or Path(value["probe_output_path"]).name != "IMPORT_PROBE.json"
        or (
            expected_path is not None
            and value.get("probe_output_path")
            != str(Path(os.path.abspath(os.fspath(expected_path))))
        )
        or value.get("cuda_device_queried") is not False
        or value.get("dynamics_executed") is not False
        or value.get("claim")
        != "EXACT_IMPORT_ONLY_FRESH_CHILD_POINT_OBSERVATION_NOT_SOFTWARE_AUTHENTICATION"
        or value.get("content_sha256")
        != _content_sha256(IMPORT_PROBE_DOMAIN, value)
    ):
        raise Stage1QualificationError("import-only preflight receipt changed")
    _validate_fresh_child_interpreter_receipt(
        value.get("fresh_child_interpreter")
    )
    if value.get("fresh_process_registry") != {
        "checked_prefixes": list(RUNTIME_MODULE_PREFIXES),
        "preloaded_module_names": [],
        "fresh_registry_observed": True,
        "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
    }:
        raise Stage1QualificationError("import-only preflight was not fresh")
    implementation = value.get("stage1_implementation")
    _validate_runtime_file_receipt(
        implementation, "import-probe Stage-1 implementation"
    )
    implementation_payload = _regular_bytes(
        SCRIPT_PATH, "import-probe Stage-1 implementation"
    )
    if implementation != {
        "path": str(SCRIPT_PATH.resolve(strict=True)),
        "bytes": len(implementation_payload),
        "sha256": _sha256(implementation_payload),
    }:
        raise Stage1QualificationError("import-probe implementation changed")
    target_receipt = value.get("rebound_target_verification")
    if type(target_receipt) is not dict or type(target_receipt.get("target")) is not str:
        raise Stage1QualificationError("import-probe REBOUND target receipt changed")
    if _canonical_json(_verify_rebound_target(Path(target_receipt["target"]))) != _canonical_json(target_receipt):
        raise Stage1QualificationError("import-probe REBOUND target no longer matches")
    _validate_runtime_module_registry(
        value.get("runtime_module_registry"), target_receipt,
        require_rebound=True,
    )
    registry = {
        item["module"]: item
        for item in value["runtime_module_registry"]
    }
    _validate_numpy_runtime_identity(
        value.get("numpy_runtime_identity"), registry
    )
    cupy = value.get("cupy_import_identity")
    module_keys = {
        "module", "spec_name", "spec_origin", "loader_type", "file_kind",
        "cached_path", "path", "bytes", "sha256",
    }
    if type(cupy) is not dict or set(cupy) != {
        "version", "module", "core_module", "fft_config_module"
    } or cupy.get("version") != CUPY_DISTRIBUTION_VERSION:
        raise Stage1QualificationError("import-probe CuPy identity changed")
    for key, module_name in (
        ("module", "cupy"),
        ("core_module", "cupy._core.core"),
        ("fft_config_module", "cupy.fft._config"),
    ):
        receipt = cupy.get(key)
        if (
            type(receipt) is not dict
            or set(receipt) != module_keys
            or receipt.get("module") != module_name
            or registry.get(module_name) != {
                **receipt,
                "predeclared_authority": "PINNED_CUPY_RECORD",
            }
        ):
            raise Stage1QualificationError(
                "import-probe CuPy module receipt changed"
            )
    _validate_runtime_import_guard(
        value.get("source_only_runtime_import_guard"),
        "import-probe",
        expected_pycache_prefix=(
            Path(value["probe_output_path"]).parent
            / ".import-probe-pycache-must-remain-absent"
        ),
    )


def _run_import_only_preflight(
    rebound_target: Path,
    temporary_root: Path,
) -> tuple[Path, dict[str, object]]:
    output_path = Path(os.path.abspath(
        os.fspath(temporary_root / "IMPORT_PROBE.json")
    ))
    completed = subprocess.run(
        [
            str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
            "import-probe-worker", "--rebound-target",
            str(Path(rebound_target).resolve()), "--probe-output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        env=_production_child_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=False,
        timeout=300,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-8_000:]
        raise Stage1QualificationError(
            f"fresh import-only runtime preflight failed: {detail}"
        )
    _payload, receipt = _load_json(
        output_path, "fresh import-only runtime preflight"
    )
    _validate_import_probe_receipt(receipt, expected_path=output_path)
    return output_path, receipt


def _source_schedule_sha256(batch: BatchState, step: float) -> str:
    payload = {
        "ephemeris": batch.ephemeris,
        "object_id": batch.object_id,
        "epoch_jd_tdb_hex": batch.epoch_jd_tdb.hex(),
        "step_hex": step.hex(),
        "massive": [
            {
                "naif_id": item.naif_id,
                "gm": item.gm_AU3_per_julian_year2.hex(),
                "position": [value.hex() for value in item.position_AU],
                "velocity": [value.hex() for value in item.velocity_AU_per_julian_year],
            }
            for item in batch.massive
        ],
    }
    return _sha256(SCHEDULE_DOMAIN + _canonical_json(payload))


def cpu_source_builder_time_scale_semantic_preflight() -> dict[str, object]:
    """Exercise the public CPU source builder at the corrected time boundary."""

    gpu_prefixes = ("cupy", "cupyx", "cupy_backends")
    preloaded_gpu_modules = sorted(
        name
        for name in sys.modules
        if type(name) is str
        and any(name == prefix or name.startswith(prefix + ".") for prefix in gpu_prefixes)
    )
    if preloaded_gpu_modules:
        raise Stage1QualificationError(
            "CPU source-builder preflight requires a fresh GPU-module registry"
        )
    try:
        import numpy as np
        from jxplanetx.engine import Provenance
        from jxplanetx.restricted_wh_source import (
            WisdomHolmanSourceTileSpec,
            build_wisdom_holman_source_tile,
        )
    except Exception as exc:
        raise Stage1QualificationError(
            "CPU source-builder semantic preflight import failed"
        ) from exc

    mu = 4.0 * math.pi * math.pi
    secondary_mu = 1.0e-3 * mu
    radius = 5.2
    records: list[dict[str, object]] = []
    for construction_path in ("PRIMARY_EXECUTION", "RESTART_WORKER"):
        schedule = _sha256(
            SOURCE_BUILDER_PREFLIGHT_DOMAIN
            + construction_path.encode("ascii")
        )
        provenance = Provenance(
            source_id=(
                "jx.tno.stage1-dynamics-qualification."
                f"source-builder-semantic-preflight.v2.{construction_path.lower()}"
            ),
            citation=(
                "Stage-1 v2 CPU-only public source-builder time-scale "
                "semantic preflight"
            ),
            version=JX_PROVENANCE_VERSION,
            sha256=schedule,
        )
        arguments = {
            "tile_id": (
                "tno-stage1-v2-time-scale-preflight-"
                + construction_path.lower().replace("_", "-")
            ),
            "epoch_origin_years": 0.0,
            "global_source_step": 0,
            "frame": "ICRF_J2000",
            "axes": "CARTESIAN_RIGHT_HANDED",
            "source_ids": ("primary", "secondary"),
            "primary_source_id": "primary",
            "source_schedule_sha256": schedule,
            "initial_source_relative_positions_AU": np.asarray(
                ((0.0, 0.0, 0.0), (radius, 0.0, 0.0)), dtype=np.float64
            ),
            "initial_source_relative_velocities_AU_per_year": np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (0.0, math.sqrt((mu + secondary_mu) / radius), 0.0),
                ),
                dtype=np.float64,
            ),
            "source_gm_AU3_per_year2": np.asarray(
                (mu, secondary_mu), dtype=np.float64
            ),
            "source_encounter_radii_AU": np.asarray(
                (0.01, 0.01), dtype=np.float64
            ),
            "certified_relative_acceleration_bounds_AU_per_year2": np.asarray(
                (100.0, 100.0), dtype=np.float64
            ),
            "provenance": provenance,
            "spec": WisdomHolmanSourceTileSpec(
                source_step_years=1.0 / 64.0,
                source_step_count=1,
            ),
        }
        accepted = build_wisdom_holman_source_tile(
            time_scale=JX_TIME_SCALE, **arguments
        )
        if (
            accepted.tile.time_scale != JX_TIME_SCALE
            or accepted.tile.node_count != 2
            or accepted.tile.global_source_step != 0
            or not _is_sha256(accepted.tile.content_sha256)
        ):
            raise Stage1QualificationError(
                "public CPU source builder did not preserve the TDB contract"
            )
        try:
            build_wisdom_holman_source_tile(
                time_scale=JX_REJECTED_VERBOSE_TIME_SCALE, **arguments
            )
        except Exception as exc:
            rejection_class = f"{type(exc).__module__}.{type(exc).__name__}"
            rejection_message = str(exc)
        else:
            raise Stage1QualificationError(
                "legacy verbose JX time-scale token was unexpectedly accepted"
            )
        if (
            rejection_class != PREDECESSOR_V1_SETUP_FAILURE_CLASS
            or rejection_message != PREDECESSOR_V1_SETUP_FAILURE_MESSAGE
        ):
            raise Stage1QualificationError(
                "legacy verbose JX time-scale rejection semantics changed"
            )
        records.append(
            {
                "construction_path": construction_path,
                "accepted_time_scale": accepted.tile.time_scale,
                "accepted_node_count": accepted.tile.node_count,
                "accepted_completed_source_steps": 1,
                "accepted_tile_content_sha256": accepted.tile.content_sha256,
                "rejected_time_scale": JX_REJECTED_VERBOSE_TIME_SCALE,
                "rejection_class": rejection_class,
                "rejection_message": rejection_message,
            }
        )

    loaded_gpu_modules = sorted(
        name
        for name in sys.modules
        if type(name) is str
        and any(name == prefix or name.startswith(prefix + ".") for prefix in gpu_prefixes)
    )
    if loaded_gpu_modules:
        raise Stage1QualificationError(
            "CPU source-builder preflight imported a GPU runtime"
        )
    receipt: dict[str, object] = {
        "schema": SOURCE_BUILDER_PREFLIGHT_SCHEMA,
        "public_builder": (
            "jxplanetx.restricted_wh_source.build_wisdom_holman_source_tile"
        ),
        "time_scale": JX_TIME_SCALE,
        "time_coordinate_unit": JX_TIME_COORDINATE_UNIT,
        "epoch_origin_semantics": JX_EPOCH_ORIGIN_SEMANTICS,
        "construction_path_count": 2,
        "records": records,
        "trajectory_scope": "SYNTHETIC_TWO_BODY_ONE_FIXED_CPU_SOURCE_STEP_PER_PATH",
        "gpu_modules_loaded": False,
        "gpu_device_queried": False,
        "qualification_or_science_outcome_generated": False,
    }
    receipt["content_sha256"] = _content_sha256(
        SOURCE_BUILDER_PREFLIGHT_DOMAIN, receipt
    )
    validate_cpu_source_builder_time_scale_semantic_preflight(receipt)
    return receipt


def validate_cpu_source_builder_time_scale_semantic_preflight(
    value: object,
) -> str:
    expected_keys = {
        "schema", "public_builder", "time_scale", "time_coordinate_unit",
        "epoch_origin_semantics", "construction_path_count", "records",
        "trajectory_scope", "gpu_modules_loaded", "gpu_device_queried",
        "qualification_or_science_outcome_generated", "content_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_keys
        or value.get("schema") != SOURCE_BUILDER_PREFLIGHT_SCHEMA
        or value.get("public_builder")
        != "jxplanetx.restricted_wh_source.build_wisdom_holman_source_tile"
        or value.get("time_scale") != JX_TIME_SCALE
        or value.get("time_coordinate_unit") != JX_TIME_COORDINATE_UNIT
        or value.get("epoch_origin_semantics") != JX_EPOCH_ORIGIN_SEMANTICS
        or type(value.get("construction_path_count")) is not int
        or value.get("construction_path_count") != 2
        or value.get("trajectory_scope")
        != "SYNTHETIC_TWO_BODY_ONE_FIXED_CPU_SOURCE_STEP_PER_PATH"
        or value.get("gpu_modules_loaded") is not False
        or value.get("gpu_device_queried") is not False
        or value.get("qualification_or_science_outcome_generated") is not False
    ):
        raise Stage1QualificationError(
            "CPU source-builder semantic preflight receipt changed"
        )
    records = value.get("records")
    if type(records) is not list or len(records) != 2:
        raise Stage1QualificationError(
            "CPU source-builder semantic preflight path roster changed"
        )
    for record, construction_path in zip(
        records, ("PRIMARY_EXECUTION", "RESTART_WORKER"), strict=True
    ):
        if (
            type(record) is not dict
            or set(record) != {
                "construction_path", "accepted_time_scale",
                "accepted_node_count", "accepted_completed_source_steps",
                "accepted_tile_content_sha256", "rejected_time_scale",
                "rejection_class", "rejection_message",
            }
            or record.get("construction_path") != construction_path
            or record.get("accepted_time_scale") != JX_TIME_SCALE
            or type(record.get("accepted_node_count")) is not int
            or record.get("accepted_node_count") != 2
            or type(record.get("accepted_completed_source_steps")) is not int
            or record.get("accepted_completed_source_steps") != 1
            or not _is_sha256(record.get("accepted_tile_content_sha256"))
            or record.get("rejected_time_scale")
            != JX_REJECTED_VERBOSE_TIME_SCALE
            or record.get("rejection_class")
            != PREDECESSOR_V1_SETUP_FAILURE_CLASS
            or record.get("rejection_message")
            != PREDECESSOR_V1_SETUP_FAILURE_MESSAGE
        ):
            raise Stage1QualificationError(
                "CPU source-builder semantic preflight observation changed"
            )
    if value.get("content_sha256") != _content_sha256(
        SOURCE_BUILDER_PREFLIGHT_DOMAIN, value
    ):
        raise Stage1QualificationError(
            "CPU source-builder semantic preflight digest changed"
        )
    return str(value["content_sha256"])


def annual_source_node_bounds(year_index: int, steps_per_year: int) -> tuple[int, int]:
    """Return the half-open full-tile node slice for one annual segment."""

    if type(year_index) is not int or not 0 <= year_index < HORIZON_JULIAN_YEARS:
        raise Stage1QualificationError("annual source slice year is outside the horizon")
    if type(steps_per_year) is not int or steps_per_year not in JX_STEPS_PER_YEAR:
        raise Stage1QualificationError("annual source slice step count is not predeclared")
    start = year_index * steps_per_year
    return start, start + steps_per_year + 1


def _pack_jx_internal_tracer_state(
    positions: Any, velocities: Any
) -> bytes:
    position_rows = tuple(
        _vec3(tuple(float(value) for value in row), f"JX internal position[{index}]")
        for index, row in enumerate(positions)
    )
    velocity_rows = tuple(
        _vec3(tuple(float(value) for value in row), f"JX internal velocity[{index}]")
        for index, row in enumerate(velocities)
    )
    if len(position_rows) != TNO_STATES_PER_BATCH or len(velocity_rows) != TNO_STATES_PER_BATCH:
        raise Stage1QualificationError("JX internal midpoint tracer roster changed")
    values = tuple(value for row in position_rows for value in row) + tuple(
        value for row in velocity_rows for value in row
    )
    payload = struct.pack(f"<{JX_MIDPOINT_RECORD_DOUBLES}d", *values)
    if len(payload) != JX_MIDPOINT_RECORD_BYTES:
        raise Stage1QualificationError("JX internal midpoint width changed")
    return payload


def _unpack_jx_internal_tracer_state(
    payload: bytes,
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    if type(payload) is not bytes or len(payload) != JX_MIDPOINT_RECORD_BYTES:
        raise Stage1QualificationError("JX midpoint record byte length changed")
    values = struct.unpack(f"<{JX_MIDPOINT_RECORD_DOUBLES}d", payload)
    if not all(math.isfinite(item) for item in values):
        raise Stage1QualificationError("JX midpoint record contains nonfinite values")
    positions = tuple(
        (values[3 * index], values[3 * index + 1], values[3 * index + 2])
        for index in range(TNO_STATES_PER_BATCH)
    )
    start = TNO_STATES_PER_BATCH * 3
    velocities = tuple(
        (
            values[start + 3 * index],
            values[start + 3 * index + 1],
            values[start + 3 * index + 2],
        )
        for index in range(TNO_STATES_PER_BATCH)
    )
    if _pack_jx_internal_tracer_state(positions, velocities) != payload:
        raise Stage1QualificationError("JX midpoint record does not round-trip bitwise")
    return positions, velocities


def _require_exact_jx_tracer_indices(cp: Any, state: Any) -> None:
    try:
        values = cp.asnumpy(state.tracer_indices)
        observed = tuple(int(item) for item in values.tolist())
    except Exception as exc:
        raise Stage1QualificationError("JX tracer indices could not be inspected") from exc
    if observed != tuple(range(TNO_STATES_PER_BATCH)):
        raise Stage1QualificationError("JX tracer indices are not exactly 0 through 8")


def _jx_checkpoint_from_arrays(year: int, batch: BatchState, source_positions: Any, source_velocities: Any, tracer_positions: Any, tracer_velocities: Any) -> dict[str, object]:
    massive = tuple(
        MassiveState(item.naif_id, item.body_id, item.gm_AU3_per_julian_year2, tuple(float(value) for value in source_positions[index]), tuple(float(value) for value in source_velocities[index]))
        for index, item in enumerate(batch.massive)
    )
    tracers = tuple(
        TracerState(item.object_id, item.spk_id, item.sample_kind, item.clone_index, tuple(float(value) for value in tracer_positions[index]), tuple(float(value) for value in tracer_velocities[index]))
        for index, item in enumerate(batch.tracers)
    )
    return _checkpoint(year, massive, tracers)


def _run_jx_lane(batch: BatchState, steps_per_year: int, cp: Any) -> dict[str, object]:
    import numpy as np
    from jxplanetx.cuda_restricted_wh import COORDINATE_SYSTEM, CUDARestrictedWHState, FixedStepCUDARestrictedWHSpec, FrozenSourcePathTile, advance_cuda_restricted_wh_fused
    from jxplanetx.engine import BackendSpec, Provenance
    from jxplanetx.restricted_wh_source import WisdomHolmanSourceTileSpec, build_wisdom_holman_source_tile

    reduced = collapse_reduced_model(batch)
    step = 1.0 / steps_per_year
    central = reduced.massive[0]
    source_positions = np.asarray([_sub(item.position_AU, central.position_AU) for item in reduced.massive], dtype=np.float64)
    source_velocities = np.asarray([_sub(item.velocity_AU_per_julian_year, central.velocity_AU_per_julian_year) for item in reduced.massive], dtype=np.float64)
    source_positions[0] = 0.0
    source_velocities[0] = 0.0
    source_gm = np.asarray([item.gm_AU3_per_julian_year2 for item in reduced.massive], dtype=np.float64)
    radii = np.ones(len(reduced.massive), dtype=np.float64)
    required_bound = math.nextafter(2.0 * math.fsum(float(value) for value in source_gm), math.inf)
    bounds = np.full(len(reduced.massive), required_bound, dtype=np.float64)
    schedule = _source_schedule_sha256(reduced, step)
    provenance = Provenance(
        source_id=(
            f"{JX_PROVENANCE_ID_PREFIX}-{batch.ephemeris}-"
            f"{batch.object_id}-{steps_per_year}"
        ),
        citation="Retained Stage-1 v2 TNO native-epoch bridge and common GM contract",
        version=JX_PROVENANCE_VERSION,
        sha256=schedule,
    )
    source_spec = WisdomHolmanSourceTileSpec(
        source_step_years=step,
        source_step_count=HORIZON_JULIAN_YEARS * steps_per_year,
        minimum_encounter_pair_separation_AU=1.0,
        minimum_jacobi_periapse_AU=0.1,
    )
    build = build_wisdom_holman_source_tile(
        tile_id=(
            f"{JX_TILE_ID_PREFIX}-{batch.ephemeris}-"
            f"{batch.object_id}-h1_{steps_per_year}"
        ),
        epoch_origin_years=(batch.epoch_jd_tdb - 2_451_545.0) / 365.25,
        global_source_step=0,
        time_scale=JX_TIME_SCALE,
        frame="ICRF_J2000",
        axes="J2000_EQUATORIAL",
        source_ids=REDUCED_BODY_IDS,
        primary_source_id=REDUCED_BODY_IDS[0],
        source_schedule_sha256=schedule,
        initial_source_relative_positions_AU=source_positions,
        initial_source_relative_velocities_AU_per_year=source_velocities,
        source_gm_AU3_per_year2=source_gm,
        source_encounter_radii_AU=radii,
        certified_relative_acceleration_bounds_AU_per_year2=bounds,
        provenance=provenance,
        spec=source_spec,
    )
    full = build.tile
    source_contract = _jx_source_contract(full)
    source_evidence_payload = _pack_jx_source_node_blocks(full)
    source_block_bytes = full.node_count * JX_SOURCE_COUNT * 3 * 8
    source_gm_payload = np.ascontiguousarray(
        full.source_gm_AU3_per_year2, dtype=np.dtype("<f8")
    ).tobytes(order="C")
    if (
        len(source_evidence_payload) != 2 * source_block_bytes
        or _jx_tile_sha256_from_raw(
            source_contract,
            source_evidence_payload[:source_block_bytes],
            source_evidence_payload[source_block_bytes:],
            source_gm_payload,
            node_count=full.node_count,
            tile_id=full.tile_id,
            global_source_step=0,
        )
        != full.content_sha256
    ):
        raise Stage1QualificationError(
            "retained JX source-node evidence does not reconstruct the full tile"
        )
    tracer_position = np.asarray([_sub(item.position_AU, central.position_AU) for item in reduced.tracers], dtype=np.float64)
    tracer_velocity = np.asarray([_sub(item.velocity_AU_per_julian_year, central.velocity_AU_per_julian_year) for item in reduced.tracers], dtype=np.float64)
    backend = BackendSpec("cupy", "cuda:0", 128)
    state = CUDARestrictedWHState(
        snapshot_id=f"stage1-v2-{batch.ephemeris}-{batch.object_id}-tracers",
        epoch_origin_years=full.epoch_origin_years,
        global_source_step=0,
        source_step_years=step,
        time_scale=full.time_scale,
        frame=full.frame,
        axes=full.axes,
        coordinate_system=COORDINATE_SYSTEM,
        length_unit="AU",
        time_unit="year",
        unit_system_id=full.unit_system_id,
        source_ids=REDUCED_BODY_IDS,
        primary_source_id=REDUCED_BODY_IDS[0],
        source_schedule_sha256=schedule,
        tracer_relative_positions_AU=cp.asarray(tracer_position),
        tracer_relative_velocities_AU_per_year=cp.asarray(tracer_velocity),
        tracer_indices=cp.arange(TNO_STATES_PER_BATCH, dtype=cp.uint64),
        provenance=provenance,
    )
    _require_exact_jx_tracer_indices(cp, state)
    checkpoints = [_jx_checkpoint_from_arrays(0, reduced, full.source_relative_positions_AU[0], full.source_relative_velocities_AU_per_year[0], tracer_position, tracer_velocity)]
    slice_digests: list[str] = []
    maximum_kepler_iterations = 0
    selected_suffix_maximum_kepler_iterations = 0
    midpoint_payload: bytes | None = None
    midpoint_revalidated_before_continuation = False
    fused_tracer_runtime_identity: dict[str, object] | None = None
    selected_restart_case = (
        batch.ephemeris,
        steps_per_year,
        batch.object_id,
    ) in JX_FRESH_PROCESS_RESTART_CASES
    selected_suffix = bytearray()
    started = time.perf_counter()
    for year in range(HORIZON_JULIAN_YEARS):
        start, node_stop = annual_source_node_bounds(year, steps_per_year)
        stop = node_stop - 1
        tile = FrozenSourcePathTile(
            tile_id=f"{full.tile_id}.annual.{year:04d}",
            epoch_origin_years=full.epoch_origin_years,
            global_source_step=start,
            source_step_years=step,
            time_scale=full.time_scale,
            frame=full.frame,
            axes=full.axes,
            coordinate_system=full.coordinate_system,
            length_unit=full.length_unit,
            time_unit=full.time_unit,
            unit_system_id=full.unit_system_id,
            source_ids=full.source_ids,
            primary_source_id=full.primary_source_id,
            source_schedule_sha256=full.source_schedule_sha256,
            source_relative_positions_AU=np.ascontiguousarray(full.source_relative_positions_AU[start:node_stop]),
            source_relative_velocities_AU_per_year=np.ascontiguousarray(full.source_relative_velocities_AU_per_year[start:node_stop]),
            source_gm_AU3_per_year2=np.ascontiguousarray(full.source_gm_AU3_per_year2),
            source_encounter_radii_AU=np.ascontiguousarray(full.source_encounter_radii_AU),
            certified_relative_acceleration_bounds_AU_per_year2=np.ascontiguousarray(full.certified_relative_acceleration_bounds_AU_per_year2),
            provenance=full.provenance,
        )
        slice_digests.append(str(tile.content_sha256))
        result = advance_cuda_restricted_wh_fused(
            state,
            tile,
            FixedStepCUDARestrictedWHSpec(
                steps_per_year,
                1,
                backend,
                threads_per_block=128,
                primary_outer_boundary_AU=4096.0,
            ),
        )
        segment_runtime_identity = json.loads(
            _canonical_json(dict(result.runtime_identity)).decode("ascii")
        )
        if fused_tracer_runtime_identity is None:
            fused_tracer_runtime_identity = segment_runtime_identity
        elif _canonical_json(segment_runtime_identity) != _canonical_json(
            fused_tracer_runtime_identity
        ):
            raise Stage1QualificationError("JX fused tracer runtime identity changed")
        state = result.state
        maximum_kepler_iterations = max(maximum_kepler_iterations, result.maximum_kepler_iterations)
        if selected_restart_case and year >= JX_MIDPOINT_YEAR:
            selected_suffix_maximum_kepler_iterations = max(
                selected_suffix_maximum_kepler_iterations,
                result.maximum_kepler_iterations,
            )
        if state.global_source_step != stop:
            raise Stage1QualificationError("JX returned an incomplete or relabeled tracer roster")
        _require_exact_jx_tracer_indices(cp, state)
        current_position = cp.asnumpy(state.tracer_relative_positions_AU)
        current_velocity = cp.asnumpy(state.tracer_relative_velocities_AU_per_year)
        internal_payload = _pack_jx_internal_tracer_state(
            current_position, current_velocity
        )
        if year + 1 == JX_MIDPOINT_YEAR:
            midpoint_payload = internal_payload
            unpacked_positions, unpacked_velocities = (
                _unpack_jx_internal_tracer_state(midpoint_payload)
            )
            if (
                _pack_jx_internal_tracer_state(
                    unpacked_positions, unpacked_velocities
                )
                != midpoint_payload
            ):
                raise Stage1QualificationError(
                    "JX midpoint failed immediate unpack/repack validation"
                )
            midpoint_revalidated_before_continuation = True
        if selected_restart_case and year + 1 >= JX_MIDPOINT_YEAR:
            selected_suffix.extend(internal_payload)
        checkpoints.append(_jx_checkpoint_from_arrays(year + 1, reduced, full.source_relative_positions_AU[stop], full.source_relative_velocities_AU_per_year[stop], current_position, current_velocity))
    elapsed = time.perf_counter() - started
    lane_id = f"jx_reduced5_h_1_{steps_per_year}"
    annual_summary = _lane_summary(checkpoints)
    all_node_invariants = _source_node_invariant_summary(
        full.source_relative_positions_AU,
        full.source_relative_velocities_AU_per_year,
        full.source_gm_AU3_per_year2,
    )
    summary = {
        **annual_summary,
        "annual_maximum_relative_massive_energy_drift_hex": annual_summary[
            "maximum_relative_massive_energy_drift_hex"
        ],
        "annual_maximum_relative_massive_energy_drift_exact": annual_summary[
            "maximum_relative_massive_energy_drift_exact"
        ],
        "annual_maximum_relative_massive_angular_momentum_drift_hex": annual_summary[
            "maximum_relative_massive_angular_momentum_drift_hex"
        ],
        **all_node_invariants,
    }
    if midpoint_payload is None or not midpoint_revalidated_before_continuation:
        raise Stage1QualificationError("JX midpoint checkpoint was not captured")
    if fused_tracer_runtime_identity is None:
        raise Stage1QualificationError("JX fused tracer runtime identity is missing")
    if selected_restart_case and len(selected_suffix) != (HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1) * JX_MIDPOINT_RECORD_BYTES:
        raise Stage1QualificationError("selected JX restart suffix length changed")
    source_build_runtime_identity = json.loads(
        _canonical_json(dict(build.runtime_identity)).decode("ascii")
    )
    source_build_runtime_identity["numpy_runtime_identity"] = (
        _numpy_runtime_identity()
    )
    return {
        "lane_id": lane_id,
        "engine": "JX_RESTRICTED_WH",
        "model": "REDUCED_5_BODY",
        "step_julian_years_hex": step.hex(),
        "steps_per_year": steps_per_year,
        "full_source_tile_sha256": full.content_sha256,
        "annual_source_slice_sha256": slice_digests,
        "source_input_content_sha256": build.input_content_sha256,
        "source_map_result_content_sha256": build.source_map_result_content_sha256,
        "source_map_verification_content_sha256": build.source_map_verification_content_sha256,
        "source_build_runtime_identity": source_build_runtime_identity,
        "fused_tracer_runtime_identity": fused_tracer_runtime_identity,
        "guard_contract": {
            "source_encounter_radius_AU_hex": float(1.0).hex(),
            "minimum_massive_pair_separation_AU_hex": float(1.0).hex(),
            "minimum_jacobi_periapse_AU_hex": float(0.1).hex(),
            "primary_outer_boundary_AU_hex": float(4096.0).hex(),
            "acceleration_bound_rule": "NEXTUP_OF_2_TIMES_TOTAL_REDUCED_GM_DIVIDED_BY_ONE_AU_SQUARED",
            "tracer_indices": list(range(TNO_STATES_PER_BATCH)),
        },
        "maximum_kepler_iterations": maximum_kepler_iterations,
        "elapsed_seconds_hex": elapsed.hex(),
        "checkpoints": checkpoints,
        "summary": summary,
        "_jx_midpoint_payload": midpoint_payload,
        "_jx_midpoint_revalidated_before_continuation": (
            midpoint_revalidated_before_continuation
        ),
        "_jx_selected_suffix_payload": bytes(selected_suffix) if selected_restart_case else None,
        "_jx_selected_suffix_ledger": {
            "full_source_tile_sha256": full.content_sha256,
            "annual_source_slice_sha256_years_500_through_999": slice_digests[JX_MIDPOINT_YEAR:],
            "steps_per_year": steps_per_year,
            "tracer_indices": list(range(TNO_STATES_PER_BATCH)),
            "maximum_kepler_iterations": selected_suffix_maximum_kepler_iterations,
            "final_global_source_step": state.global_source_step,
            "fused_tracer_runtime_identity": fused_tracer_runtime_identity,
        } if selected_restart_case else None,
        "_jx_source_evidence_payload": source_evidence_payload,
        "_jx_source_contract": source_contract,
        "_jx_source_gm_payload": source_gm_payload,
    }


def _run_jx_restart_worker(request_path: Path, output_root: Path) -> dict[str, object]:
    """Fresh-process midpoint resume used only by the four frozen restart cases."""

    fresh_child_interpreter = _fresh_child_interpreter_receipt()
    fresh_registry_receipt = _reject_preloaded_runtime_modules()
    request_payload, request = _load_json(request_path, "JX restart request")
    if (
        set(request) != RESTART_REQUEST_KEYS
        or request.get("schema") != "jx.tno.stage1-dynamics-qualification.restart-request.v2"
        or request.get("midpoint_year") != JX_MIDPOINT_YEAR
        or request.get("midpoint_byte_length") != JX_MIDPOINT_RECORD_BYTES
        or request.get("content_sha256")
        != _content_sha256(RESTART_DOMAIN, request)
    ):
        raise Stage1QualificationError("JX restart request changed")
    ephemeris = request.get("ephemeris")
    batch_order = request.get("batch_order")
    steps_per_year = request.get("steps_per_year")
    object_id = request.get("object_id")
    if (
        (ephemeris, steps_per_year, object_id) not in JX_FRESH_PROCESS_RESTART_CASES
        or type(batch_order) is not int
        or not 0 <= batch_order < OBJECT_COUNT
    ):
        raise Stage1QualificationError("JX restart selection is not predeclared")
    midpoint_path_value = request.get("midpoint_relative_path")
    offset = request.get("midpoint_byte_offset")
    if (
        midpoint_path_value != "JX_MIDPOINTS.bin"
        or type(offset) is not int
        or offset < 0
    ):
        raise Stage1QualificationError("JX restart midpoint locator changed")
    midpoint_path = request_path.resolve(strict=True).parent / midpoint_path_value
    midpoint_file = _regular_bytes(
        midpoint_path, "JX midpoint file", maximum=EXPECTED_JX_MIDPOINT_BYTES
    )
    if len(midpoint_file) != EXPECTED_JX_MIDPOINT_BYTES:
        raise Stage1QualificationError("JX midpoint file length changed")
    midpoint = midpoint_file[offset : offset + JX_MIDPOINT_RECORD_BYTES]
    if (
        len(midpoint) != JX_MIDPOINT_RECORD_BYTES
        or _sha256(midpoint) != request.get("midpoint_record_sha256")
    ):
        raise Stage1QualificationError("JX restart midpoint record changed")
    midpoint_positions, midpoint_velocities = _unpack_jx_internal_tracer_state(midpoint)
    bridge, custody, _bindings = _verify_input_documents()
    batch = _batch_states(bridge, custody, str(ephemeris))[batch_order]
    if batch.object_id != object_id:
        raise Stage1QualificationError("JX restart object binding changed")
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.is_dir() or root.is_symlink() or any(root.iterdir()):
        raise Stage1QualificationError("JX restart worker output root must be empty")
    with _source_only_runtime_import_guard(root / ".source-only-pycache-must-remain-absent"):
        try:
            import cupy as cp
            import numpy as np
            from jxplanetx.cuda_restricted_wh import COORDINATE_SYSTEM, CUDARestrictedWHState, FixedStepCUDARestrictedWHSpec, FrozenSourcePathTile, advance_cuda_restricted_wh_fused
            from jxplanetx.engine import BackendSpec, Provenance
            from jxplanetx.restricted_wh_source import WisdomHolmanSourceTileSpec, build_wisdom_holman_source_tile
            importlib.import_module("numpy._core._multiarray_umath")
            importlib.import_module("cupy._core.core")
            importlib.import_module("cupy.fft._config")
            importlib.import_module("jxplanetx._wisdom_holman_cpu_v3")
        except Exception as exc:
            raise Stage1QualificationError("JX restart runtime import failed") from exc
        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            raise Stage1QualificationError("JX restart requires a CUDA device")
        reduced = collapse_reduced_model(batch)
        step = 1.0 / steps_per_year
        central = reduced.massive[0]
        source_positions = np.asarray([_sub(item.position_AU, central.position_AU) for item in reduced.massive], dtype=np.float64)
        source_velocities = np.asarray([_sub(item.velocity_AU_per_julian_year, central.velocity_AU_per_julian_year) for item in reduced.massive], dtype=np.float64)
        source_positions[0] = 0.0
        source_velocities[0] = 0.0
        source_gm = np.asarray([item.gm_AU3_per_julian_year2 for item in reduced.massive], dtype=np.float64)
        schedule = _source_schedule_sha256(reduced, step)
        provenance = Provenance(
            source_id=(
                f"{JX_PROVENANCE_ID_PREFIX}-{batch.ephemeris}-"
                f"{batch.object_id}-{steps_per_year}"
            ),
            citation="Retained Stage-1 v2 TNO native-epoch bridge and common GM contract",
            version=JX_PROVENANCE_VERSION,
            sha256=schedule,
        )
        build = build_wisdom_holman_source_tile(
            tile_id=(
                f"{JX_TILE_ID_PREFIX}-{batch.ephemeris}-"
                f"{batch.object_id}-h1_{steps_per_year}"
            ),
            epoch_origin_years=(batch.epoch_jd_tdb - 2_451_545.0) / 365.25,
            global_source_step=0,
            time_scale=JX_TIME_SCALE,
            frame="ICRF_J2000",
            axes="J2000_EQUATORIAL",
            source_ids=REDUCED_BODY_IDS,
            primary_source_id=REDUCED_BODY_IDS[0],
            source_schedule_sha256=schedule,
            initial_source_relative_positions_AU=source_positions,
            initial_source_relative_velocities_AU_per_year=source_velocities,
            source_gm_AU3_per_year2=source_gm,
            source_encounter_radii_AU=np.ones(len(reduced.massive), dtype=np.float64),
            certified_relative_acceleration_bounds_AU_per_year2=np.full(
                len(reduced.massive),
                math.nextafter(2.0 * math.fsum(float(value) for value in source_gm), math.inf),
                dtype=np.float64,
            ),
            provenance=provenance,
            spec=WisdomHolmanSourceTileSpec(
                source_step_years=step,
                source_step_count=HORIZON_JULIAN_YEARS * steps_per_year,
                minimum_encounter_pair_separation_AU=1.0,
                minimum_jacobi_periapse_AU=0.1,
            ),
        )
        full = build.tile
        global_step = JX_MIDPOINT_YEAR * steps_per_year
        state = CUDARestrictedWHState(
            snapshot_id=f"stage1-v2-restart-{batch.ephemeris}-{batch.object_id}-{steps_per_year}",
            epoch_origin_years=full.epoch_origin_years,
            global_source_step=global_step,
            source_step_years=step,
            time_scale=full.time_scale,
            frame=full.frame,
            axes=full.axes,
            coordinate_system=COORDINATE_SYSTEM,
            length_unit="AU",
            time_unit="year",
            unit_system_id=full.unit_system_id,
            source_ids=REDUCED_BODY_IDS,
            primary_source_id=REDUCED_BODY_IDS[0],
            source_schedule_sha256=schedule,
            tracer_relative_positions_AU=cp.asarray(midpoint_positions, dtype=cp.float64),
            tracer_relative_velocities_AU_per_year=cp.asarray(midpoint_velocities, dtype=cp.float64),
            tracer_indices=cp.arange(TNO_STATES_PER_BATCH, dtype=cp.uint64),
            provenance=provenance,
        )
        _require_exact_jx_tracer_indices(cp, state)
        backend = BackendSpec("cupy", "cuda:0", 128)
        suffix = bytearray(midpoint)
        slice_digests: list[str] = []
        maximum_kepler_iterations = 0
        fused_tracer_runtime_identity: dict[str, object] | None = None
        for year in range(JX_MIDPOINT_YEAR, HORIZON_JULIAN_YEARS):
            start, node_stop = annual_source_node_bounds(year, steps_per_year)
            stop = node_stop - 1
            tile = FrozenSourcePathTile(
                tile_id=f"{full.tile_id}.annual.{year:04d}",
                epoch_origin_years=full.epoch_origin_years,
                global_source_step=start,
                source_step_years=step,
                time_scale=full.time_scale,
                frame=full.frame,
                axes=full.axes,
                coordinate_system=full.coordinate_system,
                length_unit=full.length_unit,
                time_unit=full.time_unit,
                unit_system_id=full.unit_system_id,
                source_ids=full.source_ids,
                primary_source_id=full.primary_source_id,
                source_schedule_sha256=full.source_schedule_sha256,
                source_relative_positions_AU=np.ascontiguousarray(full.source_relative_positions_AU[start:node_stop]),
                source_relative_velocities_AU_per_year=np.ascontiguousarray(full.source_relative_velocities_AU_per_year[start:node_stop]),
                source_gm_AU3_per_year2=np.ascontiguousarray(full.source_gm_AU3_per_year2),
                source_encounter_radii_AU=np.ascontiguousarray(full.source_encounter_radii_AU),
                certified_relative_acceleration_bounds_AU_per_year2=np.ascontiguousarray(full.certified_relative_acceleration_bounds_AU_per_year2),
                provenance=full.provenance,
            )
            slice_digests.append(str(tile.content_sha256))
            result = advance_cuda_restricted_wh_fused(
                state,
                tile,
                FixedStepCUDARestrictedWHSpec(
                    steps_per_year,
                    1,
                    backend,
                    threads_per_block=128,
                    primary_outer_boundary_AU=4096.0,
                ),
            )
            segment_runtime_identity = json.loads(
                _canonical_json(dict(result.runtime_identity)).decode("ascii")
            )
            if fused_tracer_runtime_identity is None:
                fused_tracer_runtime_identity = segment_runtime_identity
            elif _canonical_json(segment_runtime_identity) != _canonical_json(
                fused_tracer_runtime_identity
            ):
                raise Stage1QualificationError(
                    "JX restart fused tracer runtime identity changed"
                )
            state = result.state
            maximum_kepler_iterations = max(
                maximum_kepler_iterations, result.maximum_kepler_iterations
            )
            if state.global_source_step != stop:
                raise Stage1QualificationError("JX restart global source step changed")
            _require_exact_jx_tracer_indices(cp, state)
            suffix.extend(
                _pack_jx_internal_tracer_state(
                    cp.asnumpy(state.tracer_relative_positions_AU),
                    cp.asnumpy(state.tracer_relative_velocities_AU_per_year),
                )
            )
        suffix_payload = bytes(suffix)
        expected_suffix_bytes = (
            HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1
        ) * JX_MIDPOINT_RECORD_BYTES
        if len(suffix_payload) != expected_suffix_bytes:
            raise Stage1QualificationError("JX restart suffix length changed")
        if fused_tracer_runtime_identity is None:
            raise Stage1QualificationError(
                "JX restart fused tracer runtime identity is missing"
            )
        ledger = {
            "full_source_tile_sha256": full.content_sha256,
            "annual_source_slice_sha256_years_500_through_999": slice_digests,
            "steps_per_year": steps_per_year,
            "tracer_indices": list(range(TNO_STATES_PER_BATCH)),
            "maximum_kepler_iterations": maximum_kepler_iterations,
            "final_global_source_step": state.global_source_step,
            "fused_tracer_runtime_identity": fused_tracer_runtime_identity,
            "fresh_process_registry": fresh_registry_receipt,
        }
        restart_runtime = _restart_runtime_identity(
            cp, fresh_registry_receipt, fresh_child_interpreter
        )
        receipt: dict[str, object] = {
            "schema": RESTART_RECEIPT_SCHEMA,
            "request_content_sha256": request["content_sha256"],
            "request_file_sha256": _sha256(request_payload),
            "implementation_file_sha256": _sha256(_regular_bytes(SCRIPT_PATH, "Stage-1 implementation")),
            "midpoint_record_sha256": _sha256(midpoint),
            "midpoint_consumed": True,
            "fresh_process_resume_start_year": JX_MIDPOINT_YEAR,
            "suffix_checkpoint_count": HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1,
            "suffix_byte_length": len(suffix_payload),
            "suffix_file_sha256": _sha256(suffix_payload),
            "suffix_content_sha256": _sha256(RESTART_DOMAIN + suffix_payload),
            "final_state_sha256": _sha256(suffix_payload[-JX_MIDPOINT_RECORD_BYTES:]),
            "ledger": ledger,
            "ledger_content_sha256": _sha256(RESTART_DOMAIN + _canonical_json(ledger)),
            "runtime_identity": restart_runtime,
            "bitwise_match_checked_by_parent": False,
            "content_sha256": "",
        }
        receipt["content_sha256"] = _content_sha256(RESTART_DOMAIN, receipt)
        _validate_restart_worker_receipt(receipt)
        _write_create_only(root / "SUFFIX.bin", suffix_payload)
        _write_create_only(root / "RECEIPT.json", _canonical_json(receipt))
    return receipt


def _validate_restart_worker_receipt(
    value: object, expected_main_runtime_identity: object | None = None
) -> None:
    _guard_exact_json_integer_identities(value, "restart worker receipt")
    if type(value) is not dict or set(value) != RESTART_WORKER_RECEIPT_KEYS:
        raise Stage1QualificationError("restart worker receipt key set changed")
    ledger = value.get("ledger")
    if (
        value.get("schema") != RESTART_RECEIPT_SCHEMA
        or not _is_sha256(value.get("request_content_sha256"))
        or not _is_sha256(value.get("request_file_sha256"))
        or value.get("implementation_file_sha256")
        != _sha256(_regular_bytes(SCRIPT_PATH, "Stage-1 implementation"))
        or not _is_sha256(value.get("midpoint_record_sha256"))
        or value.get("midpoint_consumed") is not True
        or value.get("fresh_process_resume_start_year") != JX_MIDPOINT_YEAR
        or value.get("suffix_checkpoint_count")
        != HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1
        or value.get("suffix_byte_length")
        != (HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1)
        * JX_MIDPOINT_RECORD_BYTES
        or any(
            not _is_sha256(value.get(key))
            for key in (
                "suffix_file_sha256", "suffix_content_sha256",
                "final_state_sha256", "ledger_content_sha256",
                "content_sha256",
            )
        )
        or value.get("bitwise_match_checked_by_parent") is not False
        or type(ledger) is not dict
        or set(ledger) != {
            "full_source_tile_sha256",
            "annual_source_slice_sha256_years_500_through_999",
            "steps_per_year", "tracer_indices", "maximum_kepler_iterations",
            "final_global_source_step",
            "fused_tracer_runtime_identity", "fresh_process_registry",
        }
        or not _is_sha256(ledger.get("full_source_tile_sha256"))
        or type(ledger.get("annual_source_slice_sha256_years_500_through_999"))
        is not list
        or len(ledger["annual_source_slice_sha256_years_500_through_999"])
        != HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR
        or any(
            not _is_sha256(item)
            for item in ledger["annual_source_slice_sha256_years_500_through_999"]
        )
        or type(ledger.get("maximum_kepler_iterations")) is not int
        or ledger.get("maximum_kepler_iterations") < 0
        or ledger.get("steps_per_year") not in JX_STEPS_PER_YEAR
        or ledger.get("tracer_indices") != list(range(TNO_STATES_PER_BATCH))
        or type(ledger.get("final_global_source_step")) is not int
        or ledger.get("final_global_source_step")
        != HORIZON_JULIAN_YEARS * ledger.get("steps_per_year")
        or value.get("ledger_content_sha256")
        != _sha256(RESTART_DOMAIN + _canonical_json(ledger))
        or value.get("content_sha256") != _content_sha256(RESTART_DOMAIN, value)
    ):
        raise Stage1QualificationError("restart worker receipt changed")
    _validate_fused_tracer_runtime_identity(
        ledger.get("fused_tracer_runtime_identity")
    )
    _validate_restart_runtime_identity(
        value.get("runtime_identity"),
        ledger.get("fused_tracer_runtime_identity"),
        expected_main_runtime_identity,
    )
    if ledger.get("fresh_process_registry") != {
        "checked_prefixes": list(RUNTIME_MODULE_PREFIXES),
        "preloaded_module_names": [],
        "fresh_registry_observed": True,
        "claim": "TRUSTED_FRESH_PROCESS_POINT_OBSERVATION_NOT_PROCESS_AUTHENTICATION",
    }:
        raise Stage1QualificationError("restart worker did not prove a fresh registry")


def _run_fresh_process_restart_checks(
    midpoint_payload: bytes,
    midpoint_records: Sequence[Mapping[str, Any]],
    selected_suffixes: Mapping[tuple[str, int, str], tuple[bytes, Mapping[str, Any]]],
) -> tuple[list[dict[str, object]], bytes, list[dict[str, object]]]:
    if (
        type(midpoint_payload) is not bytes
        or len(midpoint_payload) != EXPECTED_JX_MIDPOINT_BYTES
        or type(midpoint_records) not in (list, tuple)
        or len(midpoint_records) != JX_MIDPOINT_RECORD_COUNT
        or type(selected_suffixes) is not dict
        or set(selected_suffixes) != set(JX_FRESH_PROCESS_RESTART_CASES)
    ):
        raise Stage1QualificationError("JX restart evidence roster changed")
    record_index = {
        (record.get("ephemeris"), record.get("steps_per_year"), record.get("object_id")): record
        for record in midpoint_records
        if type(record) is dict
    }
    if len(record_index) != JX_MIDPOINT_RECORD_COUNT:
        raise Stage1QualificationError("JX midpoint record identities are not unique")
    receipts: list[dict[str, object]] = []
    retained_stream = bytearray()
    retained_records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="jx-tno-stage1-v2-restart-") as directory:
        temporary = Path(directory)
        midpoint_path = temporary / "JX_MIDPOINTS.bin"
        _write_create_only(midpoint_path, midpoint_payload)
        for restart_order, selection in enumerate(JX_FRESH_PROCESS_RESTART_CASES):
            record = record_index.get(selection)
            if type(record) is not dict:
                raise Stage1QualificationError("selected JX midpoint record is missing")
            request: dict[str, object] = {
                "schema": "jx.tno.stage1-dynamics-qualification.restart-request.v2",
                "ephemeris": selection[0],
                "batch_order": record["batch_order"],
                "object_id": selection[2],
                "steps_per_year": selection[1],
                "midpoint_year": JX_MIDPOINT_YEAR,
                "midpoint_relative_path": "JX_MIDPOINTS.bin",
                "midpoint_byte_offset": record["byte_offset"],
                "midpoint_byte_length": JX_MIDPOINT_RECORD_BYTES,
                "midpoint_record_sha256": record["record_sha256"],
                "content_sha256": "",
            }
            request["content_sha256"] = _content_sha256(RESTART_DOMAIN, request)
            case_root = temporary / f"case-{restart_order}"
            case_root.mkdir(mode=0o700)
            request_path = temporary / f"REQUEST-{restart_order}.json"
            _write_create_only(request_path, _canonical_json(request))
            completed = subprocess.run(
                [
                    str(PROJECT_VENV_PYTHON), "-B",
                    str(SCRIPT_PATH),
                    "resume-worker",
                    "--resume-request",
                    str(request_path),
                    "--resume-output-root",
                    str(case_root),
                ],
                cwd=PROJECT_ROOT,
                env=_production_child_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=86_400,
            )
            if completed.returncode != 0:
                detail = completed.stderr.decode("utf-8", errors="replace")[-4_000:]
                raise Stage1QualificationError(
                    f"fresh-process JX restart failed: {detail}"
                )
            if {entry.name for entry in case_root.iterdir()} != {"SUFFIX.bin", "RECEIPT.json"}:
                raise Stage1QualificationError("fresh-process JX restart output roster changed")
            suffix = _regular_bytes(
                case_root / "SUFFIX.bin", "fresh-process JX suffix", maximum=1_000_000
            )
            receipt_payload, receipt = _load_json(
                case_root / "RECEIPT.json", "fresh-process JX receipt"
            )
            expected_suffix, expected_ledger = selected_suffixes[selection]
            _validate_restart_worker_receipt(receipt)
            if (
                receipt.get("request_content_sha256") != request["content_sha256"]
                or receipt.get("request_file_sha256") != _sha256(_canonical_json(request))
                or receipt.get("implementation_file_sha256")
                != _sha256(_regular_bytes(SCRIPT_PATH, "Stage-1 implementation"))
                or receipt.get("midpoint_record_sha256") != record["record_sha256"]
                or receipt.get("midpoint_consumed") is not True
                or receipt.get("fresh_process_resume_start_year")
                != JX_MIDPOINT_YEAR
                or receipt.get("suffix_checkpoint_count")
                != HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1
                or receipt.get("suffix_byte_length") != len(suffix)
                or receipt.get("suffix_file_sha256") != _sha256(suffix)
                or receipt.get("suffix_content_sha256") != _sha256(RESTART_DOMAIN + suffix)
                or receipt.get("final_state_sha256") != _sha256(suffix[-JX_MIDPOINT_RECORD_BYTES:])
                or receipt.get("ledger") != expected_ledger
                or receipt.get("ledger_content_sha256") != _sha256(RESTART_DOMAIN + _canonical_json(expected_ledger))
                or receipt.get("content_sha256") != _content_sha256(RESTART_DOMAIN, receipt)
                or receipt.get("bitwise_match_checked_by_parent") is not False
                or suffix != expected_suffix
            ):
                raise Stage1QualificationError("fresh-process JX restart replay differs bitwise")
            for stream_kind, stream_payload, stream_ledger in (
                ("MONOLITHIC", expected_suffix, expected_ledger),
                ("RESUMED", suffix, receipt["ledger"]),
            ):
                stream_offset = len(retained_stream)
                retained_stream.extend(stream_payload)
                stream_record: dict[str, object] = {
                    "stream_order": len(retained_records),
                    "restart_order": restart_order,
                    "stream_kind": stream_kind,
                    "ephemeris": selection[0],
                    "batch_order": record["batch_order"],
                    "object_id": selection[2],
                    "steps_per_year": selection[1],
                    "start_year": JX_MIDPOINT_YEAR,
                    "end_year": HORIZON_JULIAN_YEARS,
                    "checkpoint_count": JX_RESTART_SUFFIX_CHECKPOINT_COUNT,
                    "record_bytes": JX_MIDPOINT_RECORD_BYTES,
                    "byte_offset": stream_offset,
                    "byte_length": len(stream_payload),
                    "stream_sha256": _sha256(stream_payload),
                    "stream_content_sha256": _sha256(
                        RESTART_STREAM_DOMAIN + stream_payload
                    ),
                    "first_record_sha256": _sha256(
                        stream_payload[:JX_MIDPOINT_RECORD_BYTES]
                    ),
                    "final_record_sha256": _sha256(
                        stream_payload[-JX_MIDPOINT_RECORD_BYTES:]
                    ),
                    "ledger": json.loads(
                        _canonical_json(dict(stream_ledger)).decode("ascii")
                    ),
                }
                stream_record["record_content_sha256"] = _sha256(
                    RESTART_STREAM_DOMAIN
                    + _canonical_json(stream_record)
                    + stream_payload
                )
                retained_records.append(stream_record)
            receipts.append(
                {
                    "restart_order": restart_order,
                    "ephemeris": selection[0],
                    "steps_per_year": selection[1],
                    "object_id": selection[2],
                    "batch_order": record["batch_order"],
                    "midpoint_record_sha256": record["record_sha256"],
                    "worker_receipt_file_sha256": _sha256(receipt_payload),
                    "worker_receipt_content_sha256": receipt["content_sha256"],
                    "suffix_file_sha256": _sha256(suffix),
                    "final_state_sha256": _sha256(suffix[-JX_MIDPOINT_RECORD_BYTES:]),
                    "suffix_checkpoint_count": HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1,
                    "worker_request": request,
                    "worker_receipt": receipt,
                    "bitwise_midpoint_through_year_1000_match": True,
                    "bitwise_final_state_match": True,
                    "ledger_match": True,
                    "fresh_process": True,
                }
            )
    payload = bytes(retained_stream)
    if (
        len(retained_records) != JX_RESTART_STREAM_RECORD_COUNT
        or len(payload) != EXPECTED_JX_RESTART_STREAM_BYTES
    ):
        raise Stage1QualificationError("retained JX restart stream layout changed")
    return receipts, payload, retained_records


def _require_restart_stream_pair(
    monolithic: bytes, resumed: bytes, midpoint: bytes
) -> None:
    if (
        type(monolithic) is not bytes
        or type(resumed) is not bytes
        or type(midpoint) is not bytes
        or len(monolithic) != JX_RESTART_SUFFIX_BYTES
        or len(resumed) != JX_RESTART_SUFFIX_BYTES
        or len(midpoint) != JX_MIDPOINT_RECORD_BYTES
        or monolithic[:JX_MIDPOINT_RECORD_BYTES] != midpoint
        or resumed[:JX_MIDPOINT_RECORD_BYTES] != midpoint
        or monolithic != resumed
    ):
        raise Stage1QualificationError(
            "JX restart streams are not a midpoint-bound bitwise pair"
        )


def _require_restart_worker_suffix_matches(
    resumed_stream: bytes, worker_receipt: Mapping[str, Any]
) -> None:
    if (
        type(resumed_stream) is not bytes
        or type(worker_receipt) is not dict
        or worker_receipt.get("suffix_checkpoint_count")
        != JX_RESTART_SUFFIX_CHECKPOINT_COUNT
        or worker_receipt.get("suffix_byte_length") != len(resumed_stream)
        or worker_receipt.get("suffix_file_sha256")
        != _sha256(resumed_stream)
        or worker_receipt.get("suffix_content_sha256")
        != _sha256(RESTART_DOMAIN + resumed_stream)
        or worker_receipt.get("final_state_sha256")
        != _sha256(resumed_stream[-JX_MIDPOINT_RECORD_BYTES:])
    ):
        raise Stage1QualificationError(
            "JX resumed stream differs from the retained worker suffix receipt"
        )


def _forward_center_primary_relative_tracers(
    source_positions: Sequence[Sequence[float]],
    source_velocities: Sequence[Sequence[float]],
    gms: Sequence[float],
    tracer_positions: Sequence[Sequence[float]],
    tracer_velocities: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    if (
        len(source_positions) != JX_SOURCE_COUNT
        or len(source_velocities) != JX_SOURCE_COUNT
        or len(gms) != JX_SOURCE_COUNT
        or len(tracer_positions) != TNO_STATES_PER_BATCH
        or len(tracer_velocities) != TNO_STATES_PER_BATCH
    ):
        raise Stage1QualificationError("JX forward-centering roster changed")
    total = math.fsum(gms)
    if not math.isfinite(total) or not total > 0.0:
        raise Stage1QualificationError("JX forward-centering GM sum is invalid")
    source_position = tuple(
        _vec3(row, f"JX source position[{index}]")
        for index, row in enumerate(source_positions)
    )
    source_velocity = tuple(
        _vec3(row, f"JX source velocity[{index}]")
        for index, row in enumerate(source_velocities)
    )
    center_position = tuple(
        math.fsum(gms[index] * source_position[index][component]
                  for index in range(JX_SOURCE_COUNT)) / total
        for component in range(3)
    )
    center_velocity = tuple(
        math.fsum(gms[index] * source_velocity[index][component]
                  for index in range(JX_SOURCE_COUNT)) / total
        for component in range(3)
    )
    return (
        [list(_sub(_vec3(row, "JX raw tracer position"), center_position))
         for row in tracer_positions],
        [list(_sub(_vec3(row, "JX raw tracer velocity"), center_velocity))
         for row in tracer_velocities],
    )


def _require_centered_restart_checkpoint_matches(
    source_positions: Sequence[Sequence[float]],
    source_velocities: Sequence[Sequence[float]],
    gms: Sequence[float],
    tracer_positions: Sequence[Sequence[float]],
    tracer_velocities: Sequence[Sequence[float]],
    checkpoint: Mapping[str, Any],
) -> None:
    centered_positions, centered_velocities = (
        _forward_center_primary_relative_tracers(
            source_positions,
            source_velocities,
            gms,
            tracer_positions,
            tracer_velocities,
        )
    )
    if (
        type(checkpoint) is not dict
        or checkpoint.get("tracer_positions_AU") != centered_positions
        or checkpoint.get("tracer_velocities_AU_per_julian_year")
        != centered_velocities
    ):
        raise Stage1QualificationError(
            "JX restart stream differs from retained Cartesian arrays"
        )


def _validate_restart_stream_evidence(
    payload: bytes,
    metadata: object,
    midpoint_payload: bytes,
    midpoint_metadata: Mapping[str, Any],
    jx_source_handle: _StableReadHandle,
    jx_source_metadata: Mapping[str, Any],
    reconstructed_lanes: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
) -> None:
    _guard_exact_json_integer_identities(
        metadata, "JX restart-stream evidence metadata"
    )
    if type(metadata) is not dict or set(metadata) != {
        "schema", "relative_path", "encoding", "byte_length", "file_sha256",
        "content_sha256", "stream_count", "record_bytes",
        "records_per_stream", "stream_bytes", "order", "records",
    }:
        raise Stage1QualificationError("JX restart stream metadata shape changed")
    if (
        type(payload) is not bytes
        or len(payload) != EXPECTED_JX_RESTART_STREAM_BYTES
        or metadata.get("schema") != RESTART_STREAM_SCHEMA
        or metadata.get("relative_path") != "JX_RESTART_STREAMS.bin"
        or metadata.get("encoding") != "RAW_LITTLE_ENDIAN_IEEE754_BINARY64"
        or metadata.get("byte_length") != len(payload)
        or metadata.get("file_sha256") != _sha256(payload)
        or metadata.get("content_sha256")
        != _sha256(RESTART_STREAM_DOMAIN + payload)
        or metadata.get("stream_count") != JX_RESTART_STREAM_RECORD_COUNT
        or metadata.get("record_bytes") != JX_MIDPOINT_RECORD_BYTES
        or metadata.get("records_per_stream")
        != JX_RESTART_SUFFIX_CHECKPOINT_COUNT
        or metadata.get("stream_bytes") != JX_RESTART_SUFFIX_BYTES
        or metadata.get("order")
        != "CASE_THEN_MONOLITHIC_RESUMED_THEN_YEAR_500_THROUGH_1000_THEN_TRACER_POSITIONS_VELOCITIES"
    ):
        raise Stage1QualificationError("JX restart stream identity changed")
    records = metadata.get("records")
    midpoint_records = midpoint_metadata.get("records")
    restart_receipts = midpoint_metadata.get("fresh_process_restart_cases")
    source_records = jx_source_metadata.get("records")
    if (
        type(records) is not list
        or len(records) != JX_RESTART_STREAM_RECORD_COUNT
        or type(midpoint_records) is not list
        or type(source_records) is not list
        or type(restart_receipts) is not list
        or len(restart_receipts) != len(JX_FRESH_PROCESS_RESTART_CASES)
    ):
        raise Stage1QualificationError("JX restart cross-binding roster changed")
    midpoint_index = {
        (row.get("ephemeris"), row.get("steps_per_year"), row.get("object_id")): row
        for row in midpoint_records if type(row) is dict
    }
    source_index = {
        (row.get("ephemeris"), row.get("steps_per_year"), row.get("object_id")): row
        for row in source_records if type(row) is dict
    }
    lane_index = {
        (
            row.get("ephemeris"),
            row.get("lane", {}).get("steps_per_year")
            if type(row.get("lane")) is dict else None,
            row.get("object_id"),
        ): row
        for row in reconstructed_lanes
        if type(row) is dict
        and type(row.get("lane")) is dict
        and row["lane"].get("engine") == "JX_RESTRICTED_WH"
    }
    batches = {
        ephemeris: _batch_states(bridge, custody, ephemeris)
        for ephemeris in ("DE440S", "DE441")
    }
    gm_payload = _read_exact_handle_region(
        jx_source_handle, 0, JX_SOURCE_GM_BYTES, "JX restart GM binding"
    )
    gms = struct.unpack(f"<{JX_SOURCE_COUNT}d", gm_payload)
    expected_stream_offset = 0
    for restart_order, selection in enumerate(JX_FRESH_PROCESS_RESTART_CASES):
        midpoint = midpoint_index.get(selection)
        source = source_index.get(selection)
        lane_wrapper = lane_index.get(selection)
        retained_restart = restart_receipts[restart_order]
        retained_worker = (
            retained_restart.get("worker_receipt")
            if type(retained_restart) is dict else None
        )
        if not all(
            type(item) is dict
            for item in (midpoint, source, lane_wrapper, retained_restart, retained_worker)
        ):
            raise Stage1QualificationError("JX restart selected cross-binding is absent")
        lane = lane_wrapper["lane"]
        pair_payloads: list[bytes] = []
        pair_ledgers: list[Mapping[str, Any]] = []
        for stream_kind in ("MONOLITHIC", "RESUMED"):
            stream_order = restart_order * 2 + len(pair_payloads)
            record = records[stream_order]
            expected_keys = {
                "stream_order", "restart_order", "stream_kind", "ephemeris",
                "batch_order", "object_id", "steps_per_year", "start_year",
                "end_year", "checkpoint_count", "record_bytes", "byte_offset",
                "byte_length", "stream_sha256", "stream_content_sha256",
                "first_record_sha256", "final_record_sha256", "ledger",
                "record_content_sha256",
            }
            if (
                type(record) is not dict
                or set(record) != expected_keys
                or record.get("stream_order") != stream_order
                or record.get("restart_order") != restart_order
                or record.get("stream_kind") != stream_kind
                or (record.get("ephemeris"), record.get("steps_per_year"), record.get("object_id"))
                != selection
                or record.get("batch_order") != midpoint["batch_order"]
                or record.get("start_year") != JX_MIDPOINT_YEAR
                or record.get("end_year") != HORIZON_JULIAN_YEARS
                or record.get("checkpoint_count")
                != JX_RESTART_SUFFIX_CHECKPOINT_COUNT
                or record.get("record_bytes") != JX_MIDPOINT_RECORD_BYTES
                or record.get("byte_offset") != expected_stream_offset
                or record.get("byte_length") != JX_RESTART_SUFFIX_BYTES
            ):
                raise Stage1QualificationError("JX restart stream record changed")
            stream = payload[
                expected_stream_offset : expected_stream_offset + JX_RESTART_SUFFIX_BYTES
            ]
            record_without_content = dict(record)
            content = record_without_content.pop("record_content_sha256", None)
            if (
                record.get("stream_sha256") != _sha256(stream)
                or record.get("stream_content_sha256")
                != _sha256(RESTART_STREAM_DOMAIN + stream)
                or record.get("first_record_sha256")
                != _sha256(stream[:JX_MIDPOINT_RECORD_BYTES])
                or record.get("final_record_sha256")
                != _sha256(stream[-JX_MIDPOINT_RECORD_BYTES:])
                or content
                != _sha256(
                    RESTART_STREAM_DOMAIN
                    + _canonical_json(record_without_content)
                    + stream
                )
            ):
                raise Stage1QualificationError("JX restart stream digest changed")
            ledger = record.get("ledger")
            if (
                type(ledger) is not dict
                or set(ledger) != {
                    "full_source_tile_sha256",
                    "annual_source_slice_sha256_years_500_through_999",
                    "steps_per_year", "tracer_indices",
                    "maximum_kepler_iterations", "final_global_source_step",
                    "fused_tracer_runtime_identity", "fresh_process_registry",
                }
                or ledger.get("steps_per_year") != selection[1]
                or ledger.get("final_global_source_step")
                != HORIZON_JULIAN_YEARS * selection[1]
                or ledger.get("full_source_tile_sha256")
                != source["full_source_tile_sha256"]
                or ledger.get("annual_source_slice_sha256_years_500_through_999")
                != source["annual_source_slice_sha256"][JX_MIDPOINT_YEAR:]
                or _canonical_json(ledger.get("fused_tracer_runtime_identity"))
                != _canonical_json(lane.get("fused_tracer_runtime_identity"))
                or type(ledger.get("maximum_kepler_iterations")) is not int
                or ledger.get("maximum_kepler_iterations") < 0
                or ledger.get("maximum_kepler_iterations")
                > lane.get("maximum_kepler_iterations")
            ):
                raise Stage1QualificationError("JX restart ledger binding changed")
            midpoint_record = midpoint_payload[
                midpoint["byte_offset"] : midpoint["byte_offset"]
                + JX_MIDPOINT_RECORD_BYTES
            ]
            if stream[:JX_MIDPOINT_RECORD_BYTES] != midpoint_record:
                raise Stage1QualificationError("JX restart did not consume its midpoint")
            pair_payloads.append(stream)
            pair_ledgers.append(ledger)
            expected_stream_offset += JX_RESTART_SUFFIX_BYTES
        selected_midpoint = midpoint_payload[
            midpoint["byte_offset"] : midpoint["byte_offset"]
            + JX_MIDPOINT_RECORD_BYTES
        ]
        _require_restart_stream_pair(
            pair_payloads[0], pair_payloads[1], selected_midpoint
        )
        # All replay-diagnostic ledger fields, including maximum Kepler
        # iterations and fused-runtime identity, must agree exactly.
        if _canonical_json(pair_ledgers[0]) != _canonical_json(pair_ledgers[1]):
            raise Stage1QualificationError("monolithic and resumed JX ledgers differ")
        if _canonical_json(pair_ledgers[1]) != _canonical_json(
            retained_worker.get("ledger")
        ):
            raise Stage1QualificationError(
                "JX restart stream ledger differs from retained worker receipt"
            )
        _require_restart_worker_suffix_matches(
            pair_payloads[1], retained_worker
        )
        batch = batches[selection[0]][midpoint["batch_order"]]
        reduced = collapse_reduced_model(batch)
        steps = selection[1]
        node_count = HORIZON_JULIAN_YEARS * steps + 1
        source_block_bytes = node_count * JX_SOURCE_COUNT * 3 * 8
        positions = _read_exact_handle_region(
            jx_source_handle, source["byte_offset"], source_block_bytes,
            "JX restart source positions",
        )
        checkpoints = lane.get("checkpoints")
        if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT:
            raise Stage1QualificationError("JX restart ARRAY lane is incomplete")
        source_stride = JX_SOURCE_COUNT * 3 * 8
        for relative_year in range(JX_RESTART_SUFFIX_CHECKPOINT_COUNT):
            year = JX_MIDPOINT_YEAR + relative_year
            raw = pair_payloads[0][
                relative_year * JX_MIDPOINT_RECORD_BYTES :
                (relative_year + 1) * JX_MIDPOINT_RECORD_BYTES
            ]
            tracer_positions, tracer_velocities = _unpack_jx_internal_tracer_state(raw)
            source_values = struct.unpack_from(
                f"<{JX_SOURCE_COUNT * 3}d", positions,
                year * steps * source_stride,
            )
            # Velocity centering uses the retained source velocity node.
            velocity_offset = source["byte_offset"] + source_block_bytes
            velocity_node = _read_exact_handle_region(
                jx_source_handle,
                velocity_offset + year * steps * source_stride,
                source_stride,
                "JX restart source velocity node",
            )
            velocity_values = struct.unpack(
                f"<{JX_SOURCE_COUNT * 3}d", velocity_node
            )
            _require_centered_restart_checkpoint_matches(
                tuple(
                    tuple(source_values[index * 3 + component]
                          for component in range(3))
                    for index in range(JX_SOURCE_COUNT)
                ),
                tuple(
                    tuple(velocity_values[index * 3 + component]
                          for component in range(3))
                    for index in range(JX_SOURCE_COUNT)
                ),
                gms,
                tracer_positions,
                tracer_velocities,
                checkpoints[year],
            )
    if expected_stream_offset != len(payload):
        raise Stage1QualificationError("JX restart stream has trailing bytes")


def _rebound_particle_states(simulation: Any, template: BatchState) -> tuple[tuple[MassiveState, ...], tuple[TracerState, ...]]:
    if int(simulation.N) != len(template.massive) + TNO_STATES_PER_BATCH:
        raise Stage1QualificationError("REBOUND particle roster changed")
    particles = simulation.particles
    massive = tuple(
        MassiveState(item.naif_id, item.body_id, item.gm_AU3_per_julian_year2, (float(particles[i].x), float(particles[i].y), float(particles[i].z)), (float(particles[i].vx), float(particles[i].vy), float(particles[i].vz)))
        for i, item in enumerate(template.massive)
    )
    offset = len(template.massive)
    tracers = tuple(
        TracerState(item.object_id, item.spk_id, item.sample_kind, item.clone_index, (float(particles[offset + i].x), float(particles[offset + i].y), float(particles[offset + i].z)), (float(particles[offset + i].vx), float(particles[offset + i].vy), float(particles[offset + i].vz)))
        for i, item in enumerate(template.tracers)
    )
    return massive, tracers


def _ias_trace_record_size(massive_count: int) -> int:
    if massive_count not in (len(REDUCED_BODY_IDS), len(RESOLVED11_BODY_IDS)):
        raise Stage1QualificationError("IAS trace massive count changed")
    return IAS_TRACE_HEADER.size + 6 * (massive_count + TNO_STATES_PER_BATCH) * 8


def _pack_ias_trace_record(
    *,
    steps_done: int,
    observed_n: int,
    observed_n_active: int,
    observed_pair_count: int,
    event_flags: int,
    time_julian_years: float,
    massive_positions: Sequence[Sequence[float]],
    massive_velocities: Sequence[Sequence[float]],
    tracer_positions: Sequence[Sequence[float]],
    tracer_velocities: Sequence[Sequence[float]],
) -> bytes:
    massive_count = len(massive_positions)
    if (
        type(steps_done) is not int
        or steps_done < 0
        or type(observed_n) is not int
        or observed_n < 0
        or type(observed_n_active) is not int
        or observed_n_active < 0
        or type(observed_pair_count) is not int
        or observed_pair_count < 0
        or type(event_flags) is not int
        or event_flags < 0
        or event_flags & ~IAS_TRACE_EVENT_MASK
        or not math.isfinite(time_julian_years)
        or len(massive_velocities) != massive_count
        or len(tracer_positions) != TNO_STATES_PER_BATCH
        or len(tracer_velocities) != TNO_STATES_PER_BATCH
    ):
        raise Stage1QualificationError("IAS trace record header or shape changed")
    values: list[float] = []
    for label, rows in (
        ("IAS massive position", massive_positions),
        ("IAS massive velocity", massive_velocities),
        ("IAS tracer position", tracer_positions),
        ("IAS tracer velocity", tracer_velocities),
    ):
        for index, row in enumerate(rows):
            # Nonfinite values are deliberately retainable evidence; they set
            # an event flag and make the eventual qualification fail closed.
            if type(row) not in (list, tuple) or len(row) != 3 or any(
                type(value) is not float for value in row
            ):
                raise Stage1QualificationError(f"{label}[{index}] shape changed")
            values.extend(row)
    payload = IAS_TRACE_HEADER.pack(
        steps_done, observed_n, observed_n_active, observed_pair_count,
        event_flags, time_julian_years,
    ) + struct.pack(f"<{len(values)}d", *values)
    if len(payload) != _ias_trace_record_size(massive_count):
        raise Stage1QualificationError("IAS trace record byte width changed")
    return payload


def _run_rebound_lane(
    batch: BatchState,
    *,
    resolved11: bool,
    rebound: Any,
    trace_handle: Any,
) -> dict[str, object]:
    model = batch if resolved11 else collapse_reduced_model(batch)
    simulation = rebound.Simulation()
    simulation.G = 1.0
    simulation.gravity = "basic"
    simulation.collision = "none"
    simulation.boundary = "none"
    simulation.softening = 0.0
    simulation.integrator = "ias15"
    simulation.dt = IAS15_INITIAL_DT_JULIAN_YEARS
    simulation.integrator.epsilon = IAS15_EPSILON
    simulation.integrator.min_dt = IAS15_MIN_DT_JULIAN_YEARS
    simulation.integrator.adaptive_mode = IAS15_ADAPTIVE_MODE
    for item in model.massive:
        simulation.add(m=item.gm_AU3_per_julian_year2, x=item.position_AU[0], y=item.position_AU[1], z=item.position_AU[2], vx=item.velocity_AU_per_julian_year[0], vy=item.velocity_AU_per_julian_year[1], vz=item.velocity_AU_per_julian_year[2])
    for item in model.tracers:
        simulation.add(m=0.0, x=item.position_AU[0], y=item.position_AU[1], z=item.position_AU[2], vx=item.velocity_AU_per_julian_year[0], vy=item.velocity_AU_per_julian_year[1], vz=item.velocity_AU_per_julian_year[2])
    simulation.N_active = len(model.massive)
    simulation.testparticle_type = 0
    monitor: dict[str, Any] = {
        "initial_observation_count": 0,
        "heartbeat_callback_count": 0,
        "total_observation_count": 0,
        "source_tracer_pair_observation_count": 0,
        "minimum_source_tracer_endpoint_distance_AU": math.inf,
        "minimum_source_tracer_chord_distance_AU": math.inf,
        "encounter_or_guard_event_count": 0,
        "previous_relative": None,
        "callback_exception": None,
        "trace_byte_offset": int(trace_handle.tell()),
        "trace_record_count": 0,
        "trace_sha256": hashlib.sha256(),
        "initial_record_sha256": None,
        "final_record_sha256": None,
        "previous_trace_record": None,
        "previous_trace_step": -1,
        "previous_trace_time": -math.inf,
        "next_duplicate_entry_year": 0,
        "duplicate_entry_record_count": 0,
    }

    def monitor_state(current: Any, *, initial: bool) -> None:
        particles = current.particles
        observed_n = int(current.N)
        observed_n_active = int(current.N_active)
        expected_n = len(model.massive) + TNO_STATES_PER_BATCH
        flags = 0
        event_count = 0
        if observed_n != expected_n or observed_n_active != len(model.massive):
            flags |= IAS_TRACE_EVENT_ROSTER
            event_count += 1
            monitor["encounter_or_guard_event_count"] += event_count
            raise Stage1QualificationError("IAS15 particle roster changed in callback")
        current_relative: list[tuple[float, float, float]] = []
        massive_positions = [
            (float(particles[index].x), float(particles[index].y), float(particles[index].z))
            for index in range(len(model.massive))
        ]
        massive_velocities = [
            (float(particles[index].vx), float(particles[index].vy), float(particles[index].vz))
            for index in range(len(model.massive))
        ]
        finite_massive = all(
            math.isfinite(value)
            for vector in (*massive_positions, *massive_velocities)
            for value in vector
        )
        if not finite_massive:
            flags |= IAS_TRACE_EVENT_NONFINITE
            event_count += 1
        tracer_positions: list[tuple[float, float, float]] = []
        tracer_velocities: list[tuple[float, float, float]] = []
        for tracer_order in range(TNO_STATES_PER_BATCH):
            particle = particles[len(model.massive) + tracer_order]
            tracer_position = (float(particle.x), float(particle.y), float(particle.z))
            tracer_velocity = (float(particle.vx), float(particle.vy), float(particle.vz))
            tracer_positions.append(tracer_position)
            tracer_velocities.append(tracer_velocity)
            if not all(math.isfinite(value) for value in (*tracer_position, *tracer_velocity)):
                flags |= IAS_TRACE_EVENT_NONFINITE
                event_count += 1
                continue
            outer_distance = math.nextafter(
                math.hypot(*_sub(tracer_position, massive_positions[0])), math.inf
            )
            if outer_distance >= 4096.0:
                flags |= IAS_TRACE_EVENT_OUTER
                event_count += 1
            for source_position in massive_positions:
                relative = _sub(tracer_position, source_position)
                current_relative.append(relative)
                endpoint = math.nextafter(math.hypot(*relative), -math.inf)
                monitor["minimum_source_tracer_endpoint_distance_AU"] = min(
                    monitor["minimum_source_tracer_endpoint_distance_AU"], endpoint
                )
                if endpoint <= 1.0:
                    flags |= IAS_TRACE_EVENT_ENDPOINT
                    event_count += 1
        previous = monitor["previous_relative"]
        if previous is not None:
            if len(previous) != len(current_relative):
                flags |= IAS_TRACE_EVENT_ROSTER
                event_count += 1
            else:
                for start_relative, stop_relative in zip(
                    previous, current_relative, strict=True
                ):
                    delta = _sub(stop_relative, start_relative)
                    denominator = _dot(delta, delta)
                    fraction = 0.0 if denominator == 0.0 else max(
                        0.0, min(1.0, -_dot(start_relative, delta) / denominator)
                    )
                    chord = _add(start_relative, _scale(delta, fraction))
                    chord_distance = math.nextafter(math.hypot(*chord), -math.inf)
                    monitor["minimum_source_tracer_chord_distance_AU"] = min(
                        monitor["minimum_source_tracer_chord_distance_AU"],
                        chord_distance,
                    )
                    if chord_distance <= 1.0:
                        flags |= IAS_TRACE_EVENT_CHORD
                        event_count += 1
        observed_pair_count = len(current_relative)
        record = _pack_ias_trace_record(
            steps_done=int(current.steps_done),
            observed_n=observed_n,
            observed_n_active=observed_n_active,
            observed_pair_count=observed_pair_count,
            event_flags=flags,
            time_julian_years=float(current.t),
            massive_positions=massive_positions,
            massive_velocities=massive_velocities,
            tracer_positions=tracer_positions,
            tracer_velocities=tracer_velocities,
        )
        duplicate_entry, next_duplicate_year = _validate_ias_trace_transition(
            monitor["trace_record_count"],
            record,
            int(current.steps_done),
            float(current.t),
            monitor["previous_trace_record"],
            monitor["previous_trace_step"],
            monitor["previous_trace_time"],
            monitor["next_duplicate_entry_year"],
        )
        _write_ias_trace_record_bounded(trace_handle, record)
        monitor["trace_sha256"].update(record)
        record_sha = _sha256(record)
        if monitor["initial_record_sha256"] is None:
            monitor["initial_record_sha256"] = record_sha
        monitor["final_record_sha256"] = record_sha
        monitor["trace_record_count"] += 1
        monitor["previous_trace_record"] = record
        monitor["previous_trace_step"] = int(current.steps_done)
        monitor["previous_trace_time"] = float(current.t)
        monitor["next_duplicate_entry_year"] = next_duplicate_year
        if duplicate_entry:
            monitor["duplicate_entry_record_count"] += 1
        monitor["previous_relative"] = tuple(current_relative)
        if initial:
            monitor["initial_observation_count"] += 1
        else:
            monitor["heartbeat_callback_count"] += 1
        monitor["total_observation_count"] += 1
        monitor["source_tracer_pair_observation_count"] += len(current_relative)
        monitor["encounter_or_guard_event_count"] += event_count

    monitor_state(simulation, initial=True)

    def heartbeat(simulation_pointer: Any) -> None:
        current = (
            simulation_pointer.contents
            if hasattr(simulation_pointer, "contents")
            else simulation_pointer
        )
        try:
            monitor_state(current, initial=False)
        except BaseException as exc:  # ctypes callbacks may otherwise swallow it.
            monitor["callback_exception"] = (
                f"{type(exc).__module__}.{type(exc).__qualname__}: {exc}"
            )

    simulation.heartbeat = heartbeat
    initial_massive, initial_tracers = _rebound_particle_states(simulation, model)
    checkpoints = [_checkpoint(0, initial_massive, initial_tracers)]
    started = time.perf_counter()
    for year in range(1, CHECKPOINT_COUNT):
        simulation.integrate(float(year), exact_finish_time=1)
        if (
            monitor["callback_exception"] is not None
            or monitor["encounter_or_guard_event_count"] != 0
        ):
            raise Stage1QualificationError(
                "IAS15 accepted-step/chord source-tracer guard failed"
            )
        if float(simulation.t).hex() != float(year).hex():
            raise Stage1QualificationError("IAS15 missed an exact annual checkpoint")
        massive, tracers = _rebound_particle_states(simulation, model)
        checkpoints.append(_checkpoint(year, massive, tracers))
    elapsed = time.perf_counter() - started
    lane_id = "rebound511_ias15_resolved11" if resolved11 else "rebound511_ias15_reduced5"
    summary = _lane_summary(checkpoints)
    summary.update(
        {
            "massive_invariant_sampling": "ANNUAL_CHECKPOINTS_ONLY",
            "massive_invariant_sample_count": CHECKPOINT_COUNT,
        }
    )
    minimum_chord = monitor["minimum_source_tracer_chord_distance_AU"]
    if minimum_chord == math.inf:
        minimum_chord = monitor["minimum_source_tracer_endpoint_distance_AU"]
    if (
        monitor["initial_observation_count"] != 1
        or monitor["heartbeat_callback_count"]
        != int(simulation.steps_done) + HORIZON_JULIAN_YEARS
        or monitor["duplicate_entry_record_count"] != HORIZON_JULIAN_YEARS
        or monitor["next_duplicate_entry_year"] != HORIZON_JULIAN_YEARS
        or monitor["total_observation_count"]
        != monitor["initial_observation_count"]
        + monitor["heartbeat_callback_count"]
        or not math.isfinite(minimum_chord)
        or minimum_chord <= 1.0
        or monitor["callback_exception"] is not None
        or monitor["trace_record_count"] != monitor["total_observation_count"]
    ):
        raise Stage1QualificationError("IAS15 accepted-step monitor is incomplete")
    return {
        "lane_id": lane_id,
        "engine": "REBOUND_5.1.1_IAS15",
        "model": "RESOLVED_11_BODY" if resolved11 else "REDUCED_5_BODY",
        "epsilon_hex": float(simulation.integrator.epsilon).hex(),
        "initial_dt_julian_years_hex": IAS15_INITIAL_DT_JULIAN_YEARS.hex(),
        "min_dt_julian_years_hex": float(simulation.integrator.min_dt).hex(),
        "adaptive_mode": str(simulation.integrator.adaptive_mode),
        "final_time_julian_years_hex": float(simulation.t).hex(),
        "steps_done": int(simulation.steps_done),
        "elapsed_seconds_hex": elapsed.hex(),
        "accepted_step_guard": {
            "sampling": "INITIAL_PLUS_REBOUND_HEARTBEAT_ACCEPTED_STEPS_WITH_STRAIGHT_CHORD_BETWEEN_SAMPLES_NOT_CONTINUOUS_CERTIFIED_GEOMETRY",
            "initial_observation_count": monitor["initial_observation_count"],
            "heartbeat_callback_count": monitor["heartbeat_callback_count"],
            "integrate_call_count": HORIZON_JULIAN_YEARS,
            "duplicate_entry_record_count": monitor[
                "duplicate_entry_record_count"
            ],
            "total_observation_count": monitor["total_observation_count"],
            "source_tracer_pair_observation_count": monitor["source_tracer_pair_observation_count"],
            "minimum_source_tracer_endpoint_distance_AU_hex": monitor["minimum_source_tracer_endpoint_distance_AU"].hex(),
            "minimum_source_tracer_chord_distance_AU_hex": minimum_chord.hex(),
            "encounter_radius_AU_hex": float(1.0).hex(),
            "outer_boundary_AU_hex": float(4096.0).hex(),
            "encounter_or_guard_event_count": monitor["encounter_or_guard_event_count"],
            "callback_exception": monitor["callback_exception"],
            "trace": {
                "schema": IAS_TRACE_SCHEMA,
                "byte_offset": monitor["trace_byte_offset"],
                "byte_length": monitor["trace_record_count"]
                * _ias_trace_record_size(len(model.massive)),
                "record_count": monitor["trace_record_count"],
                "record_bytes": _ias_trace_record_size(len(model.massive)),
                "lane_payload_sha256": monitor["trace_sha256"].hexdigest(),
                "initial_record_sha256": monitor["initial_record_sha256"],
                "final_record_sha256": monitor["final_record_sha256"],
                "header_struct": "<QIIIId",
            },
        },
        "checkpoints": checkpoints,
        "summary": summary,
    }


def _decode_ias_trace_record(
    payload: bytes, massive_count: int
) -> dict[str, object]:
    expected = _ias_trace_record_size(massive_count)
    if type(payload) is not bytes or len(payload) != expected:
        raise Stage1QualificationError("IAS trace record length changed")
    steps, observed_n, active, pair_count, flags, time_value = (
        IAS_TRACE_HEADER.unpack_from(payload, 0)
    )
    if flags & ~IAS_TRACE_EVENT_MASK or not math.isfinite(time_value):
        raise Stage1QualificationError("IAS trace header changed")
    values = struct.unpack_from(
        f"<{6 * (massive_count + TNO_STATES_PER_BATCH)}d",
        payload,
        IAS_TRACE_HEADER.size,
    )
    cursor = 0
    def vectors(count: int) -> tuple[tuple[float, float, float], ...]:
        nonlocal cursor
        result = tuple(
            tuple(values[cursor + row * 3 + component] for component in range(3))
            for row in range(count)
        )
        cursor += count * 3
        return result
    result = {
        "steps_done": int(steps),
        "observed_N": int(observed_n),
        "observed_N_active": int(active),
        "observed_pair_count": int(pair_count),
        "event_flags": int(flags),
        "time_julian_years": float(time_value),
        "massive_positions_AU": vectors(massive_count),
        "massive_velocities_AU_per_julian_year": vectors(massive_count),
        "tracer_positions_AU": vectors(TNO_STATES_PER_BATCH),
        "tracer_velocities_AU_per_julian_year": vectors(TNO_STATES_PER_BATCH),
    }
    if cursor != len(values):
        raise Stage1QualificationError("IAS trace decoder width changed")
    return result


def _ias_trace_observation_metrics(
    decoded: Mapping[str, Any],
    previous_relative: Sequence[Sequence[float]] | None,
    massive_count: int,
) -> dict[str, object]:
    expected_n = massive_count + TNO_STATES_PER_BATCH
    flags = 0
    events = 0
    if (
        decoded.get("observed_N") != expected_n
        or decoded.get("observed_N_active") != massive_count
    ):
        flags |= IAS_TRACE_EVENT_ROSTER
        events += 1
    massive_position = decoded.get("massive_positions_AU")
    massive_velocity = decoded.get("massive_velocities_AU_per_julian_year")
    tracer_position = decoded.get("tracer_positions_AU")
    tracer_velocity = decoded.get("tracer_velocities_AU_per_julian_year")
    if not all(type(rows) is tuple for rows in (
        massive_position, massive_velocity, tracer_position, tracer_velocity
    )):
        raise Stage1QualificationError("IAS trace decoded state shape changed")
    finite_massive = all(
        math.isfinite(value)
        for vector in (*massive_position, *massive_velocity)
        for value in vector
    )
    if not finite_massive:
        flags |= IAS_TRACE_EVENT_NONFINITE
        events += 1
    relative_rows: list[tuple[float, float, float]] = []
    endpoint_minimum = math.inf
    chord_minimum = math.inf
    for tracer_order in range(TNO_STATES_PER_BATCH):
        position = tracer_position[tracer_order]
        velocity = tracer_velocity[tracer_order]
        if not all(math.isfinite(value) for value in (*position, *velocity)):
            flags |= IAS_TRACE_EVENT_NONFINITE
            events += 1
            continue
        outer = math.nextafter(
            math.hypot(*_sub(position, massive_position[0])), math.inf
        )
        if outer >= 4096.0:
            flags |= IAS_TRACE_EVENT_OUTER
            events += 1
        for source_position in massive_position:
            relative = _sub(position, source_position)
            relative_rows.append(relative)
            endpoint = math.nextafter(math.hypot(*relative), -math.inf)
            endpoint_minimum = min(endpoint_minimum, endpoint)
            if endpoint <= 1.0:
                flags |= IAS_TRACE_EVENT_ENDPOINT
                events += 1
    if previous_relative is not None:
        if len(previous_relative) != len(relative_rows):
            flags |= IAS_TRACE_EVENT_ROSTER
            events += 1
        else:
            for start, stop in zip(previous_relative, relative_rows, strict=True):
                delta = _sub(stop, start)
                denominator = _dot(delta, delta)
                fraction = 0.0 if denominator == 0.0 else max(
                    0.0, min(1.0, -_dot(start, delta) / denominator)
                )
                chord = _add(start, _scale(delta, fraction))
                distance = math.nextafter(math.hypot(*chord), -math.inf)
                chord_minimum = min(chord_minimum, distance)
                if distance <= 1.0:
                    flags |= IAS_TRACE_EVENT_CHORD
                    events += 1
    return {
        "event_flags": flags,
        "event_count": events,
        "relative_rows": tuple(relative_rows),
        "endpoint_minimum": endpoint_minimum,
        "chord_minimum": chord_minimum,
    }


def _validate_ias_trace_transition(
    record_order: int,
    raw_record: bytes,
    step: int,
    time_value: float,
    previous_raw_record: bytes | None,
    previous_step: int,
    previous_time: float,
    next_duplicate_entry_year: int,
) -> tuple[bool, int]:
    if (
        type(record_order) is not int or record_order < 0
        or type(raw_record) is not bytes or not raw_record
        or type(step) is not int or step < 0
        or type(time_value) is not float or not math.isfinite(time_value)
        or (previous_raw_record is not None and type(previous_raw_record) is not bytes)
        or type(previous_step) is not int
        or type(previous_time) is not float
        or type(next_duplicate_entry_year) is not int
        or not 0 <= next_duplicate_entry_year <= HORIZON_JULIAN_YEARS
    ):
        raise Stage1QualificationError("IAS callback transition type changed")
    if record_order == 0:
        if (
            previous_raw_record is not None
            or previous_step != -1
            or previous_time != -math.inf
            or step != 0
            or time_value.hex() != 0.0.hex()
            or next_duplicate_entry_year != 0
        ):
            raise Stage1QualificationError("IAS initial callback record changed")
        return False, 0
    if previous_raw_record is None:
        raise Stage1QualificationError("IAS callback lacks its previous record")
    if raw_record == previous_raw_record:
        if (
            next_duplicate_entry_year >= HORIZON_JULIAN_YEARS
            or step != previous_step
            or time_value.hex()
            != float(next_duplicate_entry_year).hex()
        ):
            raise Stage1QualificationError(
                "IAS duplicate integrate-entry callback changed"
            )
        return True, next_duplicate_entry_year + 1
    if step != previous_step + 1 or not time_value > previous_time:
        raise Stage1QualificationError(
            "IAS nonduplicate callback is not one strictly advancing step"
        )
    return False, next_duplicate_entry_year


def _write_ias_trace_record_bounded(trace_handle: Any, record: bytes) -> None:
    if type(record) is not bytes or not record:
        raise Stage1QualificationError("IAS trace write record changed")
    try:
        offset = trace_handle.tell()
    except Exception as exc:
        raise Stage1QualificationError("IAS trace write offset is unavailable") from exc
    if (
        type(offset) is not int
        or offset < 0
        or offset > IAS_TRACE_MAXIMUM_BYTES - len(record)
    ):
        raise Stage1QualificationError(
            "IAS trace would exceed its predeclared hard byte cap"
        )
    try:
        written = trace_handle.write(record)
    except Exception as exc:
        raise Stage1QualificationError("IAS trace record write failed") from exc
    if written is not None and (type(written) is not int or written != len(record)):
        raise Stage1QualificationError("IAS trace record write was incomplete")


def _require_ias_integer_checkpoint_matches(
    centered: Mapping[str, Any], checkpoint: Mapping[str, Any]
) -> None:
    if (
        type(centered) is not dict
        or type(checkpoint) is not dict
        or not all(centered.get(key) == checkpoint.get(key) for key in _CARTESIAN_KEYS)
    ):
        raise Stage1QualificationError(
            "IAS integer-year trace record differs from Cartesian arrays"
        )


def _validate_ias_trace_evidence(
    handle: _StableReadHandle,
    metadata: object,
    reconstructed_lanes: Sequence[Mapping[str, Any]],
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
) -> None:
    _guard_exact_json_integer_identities(
        metadata, "IAS accepted-step trace metadata"
    )
    if type(metadata) is not dict or set(metadata) != {
        "schema", "relative_path", "encoding", "byte_length", "file_sha256",
        "content_sha256", "lane_count", "order",
    }:
        raise Stage1QualificationError("IAS trace metadata shape changed")
    digest = _stream_handle_digests(
        handle, content_domain=IAS_TRACE_DOMAIN,
    )
    if (
        metadata.get("schema") != IAS_TRACE_SCHEMA
        or metadata.get("relative_path") != "IAS15_ACCEPTED_STEPS.bin"
        or metadata.get("encoding")
        != "FIXED_PER_LANE_RAW_LITTLE_ENDIAN_RECORDS"
        or metadata.get("byte_length") != digest["bytes"]
        or metadata.get("file_sha256") != digest["sha256"]
        or metadata.get("content_sha256") != digest["content_sha256"]
        or metadata.get("lane_count") != OBJECT_COUNT * 2 * 2
        or metadata.get("order")
        != "EPHEMERIS_BATCH_THEN_REDUCED5_RESOLVED11_CALLBACK_ORDER"
    ):
        raise Stage1QualificationError("IAS trace identity changed")
    templates = {
        ephemeris: _batch_states(bridge, custody, ephemeris)
        for ephemeris in ("DE440S", "DE441")
    }
    expected_offset = 0
    ias_lanes = [
        row for row in reconstructed_lanes
        if type(row) is dict and type(row.get("lane")) is dict
        and row["lane"].get("engine") == "REBOUND_5.1.1_IAS15"
    ]
    expected_order = [
        (ephemeris, batch_order, resolved)
        for ephemeris in ("DE440S", "DE441")
        for batch_order in range(OBJECT_COUNT)
        for resolved in (False, True)
    ]
    if len(ias_lanes) != len(expected_order):
        raise Stage1QualificationError("IAS trace lane roster changed")
    for wrapper, (ephemeris, batch_order, resolved) in zip(
        ias_lanes, expected_order, strict=True
    ):
        lane = wrapper["lane"]
        expected_lane = (
            "rebound511_ias15_resolved11"
            if resolved else "rebound511_ias15_reduced5"
        )
        if (
            wrapper.get("ephemeris") != ephemeris
            or wrapper.get("batch_order") != batch_order
            or lane.get("lane_id") != expected_lane
        ):
            raise Stage1QualificationError("IAS trace lane order changed")
        model = (
            templates[ephemeris][batch_order]
            if resolved else collapse_reduced_model(templates[ephemeris][batch_order])
        )
        massive_count = len(model.massive)
        record_bytes = _ias_trace_record_size(massive_count)
        guard = lane.get("accepted_step_guard")
        trace = guard.get("trace") if type(guard) is dict else None
        if (
            type(trace) is not dict
            or set(trace) != {
                "schema", "byte_offset", "byte_length", "record_count",
                "record_bytes", "lane_payload_sha256", "initial_record_sha256",
                "final_record_sha256", "header_struct",
            }
            or trace.get("schema") != IAS_TRACE_SCHEMA
            or trace.get("byte_offset") != expected_offset
            or type(trace.get("record_count")) is not int
            or trace.get("record_count") < 2
            or trace.get("record_bytes") != record_bytes
            or trace.get("byte_length") != trace["record_count"] * record_bytes
            or trace.get("header_struct") != "<QIIIId"
        ):
            raise Stage1QualificationError("IAS trace lane locator changed")
        lane_digest = hashlib.sha256()
        initial_record_sha256: str | None = None
        final_record_sha256: str | None = None
        previous_relative: Sequence[Sequence[float]] | None = None
        previous_raw_record: bytes | None = None
        previous_step = -1
        previous_time = -math.inf
        next_duplicate_entry_year = 0
        duplicate_entry_record_count = 0
        endpoint_minimum = math.inf
        chord_minimum = math.inf
        event_count = 0
        pair_count = 0
        annual_matches = [False] * CHECKPOINT_COUNT
        checkpoints = lane.get("checkpoints")
        if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT:
            raise Stage1QualificationError("IAS ARRAY checkpoints are missing")
        for record_order, raw in enumerate(
            _iter_exact_handle_records(
                handle, expected_offset, trace["record_count"], record_bytes,
                "IAS trace lane",
            )
        ):
            lane_digest.update(raw)
            raw_digest = _sha256(raw)
            if initial_record_sha256 is None:
                initial_record_sha256 = raw_digest
            final_record_sha256 = raw_digest
            decoded = _decode_ias_trace_record(raw, massive_count)
            step = decoded["steps_done"]
            time_value = decoded["time_julian_years"]
            duplicate_entry, next_duplicate_entry_year = (
                _validate_ias_trace_transition(
                    record_order,
                    raw,
                    step,
                    time_value,
                    previous_raw_record,
                    previous_step,
                    previous_time,
                    next_duplicate_entry_year,
                )
            )
            if duplicate_entry:
                duplicate_entry_record_count += 1
            metrics = _ias_trace_observation_metrics(
                decoded, previous_relative, massive_count
            )
            if (
                decoded["observed_pair_count"] != TNO_STATES_PER_BATCH * massive_count
                or decoded["event_flags"] != metrics["event_flags"]
            ):
                raise Stage1QualificationError("IAS callback predicate evidence changed")
            previous_relative = metrics["relative_rows"]
            endpoint_minimum = min(endpoint_minimum, metrics["endpoint_minimum"])
            chord_minimum = min(chord_minimum, metrics["chord_minimum"])
            event_count += metrics["event_count"]
            pair_count += decoded["observed_pair_count"]
            if time_value.is_integer() and 0 <= int(time_value) <= HORIZON_JULIAN_YEARS:
                year = int(time_value)
                massive = tuple(
                    MassiveState(
                        item.naif_id, item.body_id,
                        item.gm_AU3_per_julian_year2,
                        decoded["massive_positions_AU"][index],
                        decoded["massive_velocities_AU_per_julian_year"][index],
                    )
                    for index, item in enumerate(model.massive)
                )
                tracers = tuple(
                    TracerState(
                        item.object_id, item.spk_id, item.sample_kind,
                        item.clone_index,
                        decoded["tracer_positions_AU"][index],
                        decoded["tracer_velocities_AU_per_julian_year"][index],
                    )
                    for index, item in enumerate(model.tracers)
                )
                centered = _checkpoint(year, massive, tracers)
                _require_ias_integer_checkpoint_matches(
                    centered, checkpoints[year]
                )
                annual_matches[year] = True
            previous_step = step
            previous_time = time_value
            previous_raw_record = raw
        if (
            trace.get("lane_payload_sha256") != lane_digest.hexdigest()
            or trace.get("initial_record_sha256") != initial_record_sha256
            or trace.get("final_record_sha256") != final_record_sha256
        ):
            raise Stage1QualificationError("IAS trace lane digest changed")
        if chord_minimum == math.inf:
            chord_minimum = endpoint_minimum
        heartbeat_count = trace["record_count"] - 1
        if (
            previous_step != lane.get("steps_done")
            or previous_time.hex() != float(HORIZON_JULIAN_YEARS).hex()
            or heartbeat_count
            != lane.get("steps_done") + HORIZON_JULIAN_YEARS
            or duplicate_entry_record_count != HORIZON_JULIAN_YEARS
            or next_duplicate_entry_year != HORIZON_JULIAN_YEARS
            or not all(annual_matches)
            or guard.get("initial_observation_count") != 1
            or guard.get("heartbeat_callback_count") != heartbeat_count
            or guard.get("integrate_call_count") != HORIZON_JULIAN_YEARS
            or guard.get("duplicate_entry_record_count")
            != duplicate_entry_record_count
            or guard.get("total_observation_count") != trace["record_count"]
            or guard.get("source_tracer_pair_observation_count") != pair_count
            or guard.get("encounter_or_guard_event_count") != event_count
            or guard.get("callback_exception") is not None
            or guard.get("minimum_source_tracer_endpoint_distance_AU_hex")
            != endpoint_minimum.hex()
            or guard.get("minimum_source_tracer_chord_distance_AU_hex")
            != chord_minimum.hex()
            or event_count != 0
            or endpoint_minimum <= 1.0
            or chord_minimum <= 1.0
        ):
            raise Stage1QualificationError("IAS callback trace did not rederive its guard")
        expected_offset += trace["byte_length"]
    if expected_offset != metadata["byte_length"]:
        raise Stage1QualificationError("IAS trace has trailing or unbound bytes")


def _checkpoint_tracer(
    checkpoint: Mapping[str, Any], year: int, tracer_order: int
) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    if (
        type(checkpoint) is not dict
        or checkpoint.get("year_index") != year
        or checkpoint.get("time_julian_years_hex") != float(year).hex()
    ):
        raise Stage1QualificationError("annual checkpoint schedule changed")
    tracers = checkpoint.get("tracers")
    positions = checkpoint.get("tracer_positions_AU")
    velocities = checkpoint.get("tracer_velocities_AU_per_julian_year")
    if (
        type(tracers) is not list
        or len(tracers) != TNO_STATES_PER_BATCH
        or type(positions) is not list
        or len(positions) != TNO_STATES_PER_BATCH
        or type(velocities) is not list
        or len(velocities) != TNO_STATES_PER_BATCH
    ):
        raise Stage1QualificationError("annual tracer observable roster changed")
    tracer = tracers[tracer_order]
    if type(tracer) is not dict or tracer.get("tracer_order") != tracer_order:
        raise Stage1QualificationError("annual tracer label changed")
    elements = tracer.get("elements")
    if type(elements) is not dict or set(elements) != {"semimajor_axis_AU"}:
        raise Stage1QualificationError("annual semimajor-axis observable changed")
    semimajor_axis = _exact_float_hex(
        elements["semimajor_axis_AU"], "annual semimajor axis"
    )
    if not semimajor_axis > 0.0:
        raise Stage1QualificationError("annual semimajor axis is not positive")
    return (
        semimajor_axis,
        _vec3(positions[tracer_order], "annual tracer position"),
        _vec3(velocities[tracer_order], "annual tracer velocity"),
    )


def _lane_tracer_series(
    lane: Mapping[str, Any], tracer_order: int
) -> tuple[
    tuple[float, ...],
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    if type(lane) is not dict or type(tracer_order) is not int or not 0 <= tracer_order < TNO_STATES_PER_BATCH:
        raise Stage1QualificationError("lane/tracer selection changed")
    checkpoints = lane.get("checkpoints")
    if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT:
        raise Stage1QualificationError("lane annual checkpoint count changed")
    rows = tuple(
        _checkpoint_tracer(checkpoint, year, tracer_order)
        for year, checkpoint in enumerate(checkpoints)
    )
    return (
        tuple(row[0] for row in rows),
        tuple(row[1] for row in rows),
        tuple(row[2] for row in rows),
    )


def _lane_massive_series(
    lane: Mapping[str, Any], source_order: int
) -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[tuple[float, float, float], ...],
]:
    if (
        type(lane) is not dict
        or lane.get("model") != "REDUCED_5_BODY"
        or type(source_order) is not int
        or not 0 <= source_order < len(REDUCED_BODY_IDS)
    ):
        raise Stage1QualificationError("reduced massive-source selection changed")
    checkpoints = lane.get("checkpoints")
    if type(checkpoints) is not list or len(checkpoints) != CHECKPOINT_COUNT:
        raise Stage1QualificationError("massive-source checkpoint count changed")
    positions: list[tuple[float, float, float]] = []
    velocities: list[tuple[float, float, float]] = []
    for year, checkpoint in enumerate(checkpoints):
        if (
            type(checkpoint) is not dict
            or checkpoint.get("year_index") != year
            or checkpoint.get("time_julian_years_hex") != float(year).hex()
        ):
            raise Stage1QualificationError("massive-source annual schedule changed")
        position_rows = checkpoint.get("massive_positions_AU")
        velocity_rows = checkpoint.get("massive_velocities_AU_per_julian_year")
        if (
            type(position_rows) is not list
            or len(position_rows) != len(REDUCED_BODY_IDS)
            or type(velocity_rows) is not list
            or len(velocity_rows) != len(REDUCED_BODY_IDS)
        ):
            raise Stage1QualificationError("reduced massive-source roster changed")
        positions.append(_vec3(position_rows[source_order], "massive-source position"))
        velocities.append(_vec3(velocity_rows[source_order], "massive-source velocity"))
    return tuple(positions), tuple(velocities)


def _massive_comparison_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], source_order: int
) -> dict[str, Fraction]:
    left_r, left_v = _lane_massive_series(left, source_order)
    right_r, right_v = _lane_massive_series(right, source_order)
    return {
        "position_AU_squared": max(
            _exact_squared_vector_difference(left_r[year], right_r[year])
            for year in range(CHECKPOINT_COUNT)
        ),
        "velocity_AU_per_julian_year_squared": max(
            _exact_squared_vector_difference(left_v[year], right_v[year])
            for year in range(CHECKPOINT_COUNT)
        ),
    }


def _massive_reference_ulp_floors(
    reference: Mapping[str, Any], source_order: int
) -> dict[str, Fraction]:
    reference_r, reference_v = _lane_massive_series(reference, source_order)
    return {
        "position_AU_squared": max(
            _exact_squared_reference_ulp_floor(item) for item in reference_r
        ),
        "velocity_AU_per_julian_year_squared": max(
            _exact_squared_reference_ulp_floor(item) for item in reference_v
        ),
    }


def _comparison_metrics(
    left: Mapping[str, Any], right: Mapping[str, Any], tracer_order: int
) -> dict[str, Fraction]:
    left_a, left_r, left_v = _lane_tracer_series(left, tracer_order)
    right_a, right_r, right_v = _lane_tracer_series(right, tracer_order)
    denominator = max(left_a[0], right_a[0])
    if not denominator > 0.0:
        raise Stage1QualificationError("fixed initial semimajor-axis denominator is invalid")
    denominator_fraction = Fraction.from_float(denominator)
    maximum_a = Fraction(0, 1)
    maximum_r_squared = Fraction(0, 1)
    maximum_v_squared = Fraction(0, 1)
    for year in range(CHECKPOINT_COUNT):
        a_difference = abs(
            Fraction.from_float(left_a[year]) - Fraction.from_float(right_a[year])
        ) / denominator_fraction
        if a_difference > maximum_a:
            maximum_a = a_difference
        maximum_r_squared = max(
            maximum_r_squared,
            _exact_squared_vector_difference(left_r[year], right_r[year]),
        )
        maximum_v_squared = max(
            maximum_v_squared,
            _exact_squared_vector_difference(left_v[year], right_v[year]),
        )
    return {
        "semimajor_axis": maximum_a,
        "position_AU_squared": maximum_r_squared,
        "velocity_AU_per_julian_year_squared": maximum_v_squared,
    }


def _reference_ulp_floors(
    compared: Mapping[str, Any], reference: Mapping[str, Any], tracer_order: int
) -> dict[str, Fraction]:
    compared_a, _compared_r, _compared_v = _lane_tracer_series(compared, tracer_order)
    reference_a, reference_r, reference_v = _lane_tracer_series(reference, tracer_order)
    denominator = max(compared_a[0], reference_a[0])
    if not denominator > 0.0:
        raise Stage1QualificationError("ULP-floor initial denominator is invalid")
    a_floor = Fraction(0, 1)
    r_floor_squared = Fraction(0, 1)
    v_floor_squared = Fraction(0, 1)
    for year in range(CHECKPOINT_COUNT):
        a_floor = max(
            a_floor,
            Fraction.from_float(64.0 * math.ulp(reference_a[year]))
            / Fraction.from_float(denominator),
        )
        r_floor_squared = max(
            r_floor_squared, _exact_squared_reference_ulp_floor(reference_r[year])
        )
        v_floor_squared = max(
            v_floor_squared, _exact_squared_reference_ulp_floor(reference_v[year])
        )
    return {
        "semimajor_axis": a_floor,
        "position_AU_squared": r_floor_squared,
        "velocity_AU_per_julian_year_squared": v_floor_squared,
    }


def _scattering_value(lane: Mapping[str, Any], tracer_order: int) -> Fraction:
    semimajor_axis, _positions, _velocities = _lane_tracer_series(
        lane, tracer_order
    )
    metric = preliminary_scattering_metric(
        list(semimajor_axis), expected_count=CHECKPOINT_COUNT
    )
    return _fraction_from_record(
        metric["maximum_absolute_delta_over_initial_exact"],
        "scattering diagnostic",
    )


def _threshold_gate(observations: Sequence[float], threshold: Fraction) -> dict[str, object]:
    if type(observations) not in (list, tuple) or not observations:
        raise Stage1QualificationError("gate observation roster is empty")
    observed = max(observations)
    if not math.isfinite(observed) or observed < 0.0:
        raise Stage1QualificationError("gate observation is invalid")
    return {
        "observation_count": len(observations),
        "observed_max_hex": observed.hex(),
        "threshold_rational": f"{threshold.numerator}/{threshold.denominator}",
        "threshold_nearest_binary64_hex": float(threshold).hex(),
        "comparison": "EXACT_FRACTION_FROM_OBSERVED_BINARY64_LESS_THAN_OR_EQUAL_TO_RATIONAL_POLICY",
        "passed": _passes_rational(observed, threshold),
    }


def _boolean_gate(values: Sequence[bool], *, rule: str) -> dict[str, object]:
    if type(values) not in (list, tuple) or not values or any(type(item) is not bool for item in values):
        raise Stage1QualificationError("boolean gate observation roster is invalid")
    failures = sum(not item for item in values)
    return {
        "observation_count": len(values),
        "failure_count": failures,
        "comparison": rule,
        "passed": failures == 0,
    }


def _exact_fraction_gate(
    observations: Sequence[Fraction], threshold: Fraction
) -> dict[str, object]:
    if (
        type(observations) not in (list, tuple)
        or not observations
        or any(type(item) is not Fraction or item < 0 for item in observations)
        or type(threshold) is not Fraction
        or threshold < 0
    ):
        raise Stage1QualificationError("exact-fraction gate roster is invalid")
    observed = max(observations)
    return {
        "observation_count": len(observations),
        "observed_exact_numerator": observed.numerator,
        "observed_exact_denominator": observed.denominator,
        "observed_outward_binary64_hex": _fraction_to_outward_float(observed).hex(),
        "threshold_rational": f"{threshold.numerator}/{threshold.denominator}",
        "threshold_nearest_binary64_hex": float(threshold).hex(),
        "comparison": "EXACT_RATIONAL_OBSERVATION_LESS_THAN_OR_EQUAL_TO_EXACT_RATIONAL_POLICY",
        "passed": observed <= threshold,
    }


def _exact_squared_gate(
    observations: Sequence[Fraction], threshold: Fraction
) -> dict[str, object]:
    if (
        type(observations) not in (list, tuple)
        or not observations
        or any(type(item) is not Fraction or item < 0 for item in observations)
        or type(threshold) is not Fraction
        or threshold < 0
    ):
        raise Stage1QualificationError("exact-squared gate roster is invalid")
    observed_squared = max(observations)
    threshold_squared = threshold * threshold
    return {
        "observation_count": len(observations),
        "observed_squared_exact_numerator": observed_squared.numerator,
        "observed_squared_exact_denominator": observed_squared.denominator,
        "observed_norm_outward_binary64_hex": _sqrt_fraction_to_outward_float(
            observed_squared
        ).hex(),
        "threshold_rational": f"{threshold.numerator}/{threshold.denominator}",
        "threshold_nearest_binary64_hex": float(threshold).hex(),
        "comparison": "EXACT_SQUARED_RATIONAL_OBSERVATION_LESS_THAN_OR_EQUAL_TO_EXACT_SQUARED_RATIONAL_POLICY",
        "passed": observed_squared <= threshold_squared,
    }


def _exact_scalar_contraction_passes(
    fine: Fraction, coarse: Fraction, saturation_floor: Fraction
) -> bool:
    if any(
        type(value) is not Fraction or value < 0
        for value in (fine, coarse, saturation_floor)
    ):
        raise Stage1QualificationError("exact scalar contraction input is invalid")
    return fine <= saturation_floor or coarse >= 3 * fine


def _exact_squared_contraction_passes(
    fine_squared: Fraction,
    coarse_squared: Fraction,
    saturation_floor_squared: Fraction,
) -> bool:
    if any(
        type(value) is not Fraction or value < 0
        for value in (fine_squared, coarse_squared, saturation_floor_squared)
    ):
        raise Stage1QualificationError("exact squared contraction input is invalid")
    return (
        fine_squared <= saturation_floor_squared
        or coarse_squared >= 9 * fine_squared
    )


def _exact_energy_contraction_passes(
    fine: Fraction, coarse: Fraction
) -> bool:
    if any(type(value) is not Fraction or value < 0 for value in (fine, coarse)):
        raise Stage1QualificationError("exact energy contraction input is invalid")
    return fine <= max(coarse / 3, Fraction(1, 2**46))


def evaluate_qualification_gates(batch_lanes: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    expected_count = OBJECT_COUNT * 2 * len(LANE_SPECS)
    if type(batch_lanes) not in (list, tuple) or len(batch_lanes) != expected_count:
        raise Stage1QualificationError("result batch-lane count differs")
    index: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    expected_order = [
        (ephemeris, batch_order, spec["lane_id"])
        for ephemeris in ("DE440S", "DE441")
        for batch_order in range(OBJECT_COUNT)
        for spec in LANE_SPECS
    ]
    observed_order: list[tuple[str, int, str]] = []
    for record in batch_lanes:
        if type(record) is not dict or type(record.get("lane")) is not dict:
            raise Stage1QualificationError("gate batch-lane record type changed")
        lane = record["lane"]
        key = (record.get("ephemeris"), record.get("batch_order"), lane.get("lane_id"))
        if key in index:
            raise Stage1QualificationError("duplicate result batch lane")
        index[key] = lane
        observed_order.append(key)
    if observed_order != expected_order:
        raise Stage1QualificationError("result batch-lane order changed")

    exact_observations: dict[str, list[Fraction]] = {
        "richardson_a": [], "fine_ias_a": [], "fine_ias_s": [],
        "model_a": [], "model_s": [], "ephemeris_a": [], "ephemeris_s": [],
        "jx_energy": [], "ias_energy": [],
    }
    squared_observations: dict[str, list[Fraction]] = {
        "richardson_r": [], "richardson_v": [],
        "fine_ias_r": [], "fine_ias_v": [],
        "source_richardson_r": [], "source_richardson_v": [],
        "source_fine_ias_r": [], "source_fine_ias_v": [],
    }
    float_observations: dict[str, list[float]] = {
        "jx_angular": [], "ias_angular": [],
    }
    contraction: dict[str, list[bool]] = {
        "semimajor_axis": [],
        "position_AU": [],
        "velocity_AU_per_julian_year": [],
        "source_position_AU": [],
        "source_velocity_AU_per_julian_year": [],
        "massive_energy": [],
        "ias15_accepted_step_guard": [],
    }
    scattering: dict[tuple[str, int, str, int], Fraction] = {}
    scattering_chain_fractions: list[Fraction] = []
    for ephemeris in ("DE440S", "DE441"):
        for batch_order in range(OBJECT_COUNT):
            coarse = index[(ephemeris, batch_order, "jx_reduced5_h_1_16")]
            fine = index[(ephemeris, batch_order, "jx_reduced5_h_1_32")]
            reduced = index[(ephemeris, batch_order, "rebound511_ias15_reduced5")]
            resolved = index[(ephemeris, batch_order, "rebound511_ias15_resolved11")]
            for tracer_order in range(TNO_STATES_PER_BATCH):
                coarse_fine = _comparison_metrics(coarse, fine, tracer_order)
                fine_reference = _comparison_metrics(fine, reduced, tracer_order)
                coarse_reference = _comparison_metrics(coarse, reduced, tracer_order)
                floors = _reference_ulp_floors(fine, reduced, tracer_order)
                model = _comparison_metrics(reduced, resolved, tracer_order)
                exact_observations["richardson_a"].append(
                    coarse_fine["semimajor_axis"] / 3
                )
                squared_observations["richardson_r"].append(
                    coarse_fine["position_AU_squared"] / 9
                )
                squared_observations["richardson_v"].append(
                    coarse_fine["velocity_AU_per_julian_year_squared"] / 9
                )
                exact_observations["fine_ias_a"].append(
                    fine_reference["semimajor_axis"]
                )
                squared_observations["fine_ias_r"].append(
                    fine_reference["position_AU_squared"]
                )
                squared_observations["fine_ias_v"].append(
                    fine_reference["velocity_AU_per_julian_year_squared"]
                )
                exact_observations["model_a"].append(model["semimajor_axis"])
                contraction["semimajor_axis"].append(
                    _exact_scalar_contraction_passes(
                        fine_reference["semimajor_axis"],
                        coarse_reference["semimajor_axis"],
                        floors["semimajor_axis"],
                    )
                )
                for component in (
                    "position_AU",
                    "velocity_AU_per_julian_year",
                ):
                    squared_key = f"{component}_squared"
                    contraction[component].append(
                        _exact_squared_contraction_passes(
                            fine_reference[squared_key],
                            coarse_reference[squared_key],
                            floors[squared_key],
                        )
                    )
                for lane in (coarse, fine, reduced, resolved):
                    lane_id = str(lane["lane_id"])
                    scattering[(ephemeris, batch_order, lane_id, tracer_order)] = _scattering_value(lane, tracer_order)
                fine_s = scattering[(ephemeris, batch_order, "jx_reduced5_h_1_32", tracer_order)]
                reduced_s = scattering[(ephemeris, batch_order, "rebound511_ias15_reduced5", tracer_order)]
                resolved_s = scattering[(ephemeris, batch_order, "rebound511_ias15_resolved11", tracer_order)]
                exact_observations["fine_ias_s"].append(abs(fine_s - reduced_s))
                exact_observations["model_s"].append(abs(reduced_s - resolved_s))
            for source_order in range(len(REDUCED_BODY_IDS)):
                source_coarse_fine = _massive_comparison_metrics(
                    coarse, fine, source_order
                )
                source_fine_reference = _massive_comparison_metrics(
                    fine, reduced, source_order
                )
                source_coarse_reference = _massive_comparison_metrics(
                    coarse, reduced, source_order
                )
                source_floors = _massive_reference_ulp_floors(
                    reduced, source_order
                )
                squared_observations["source_richardson_r"].append(
                    source_coarse_fine["position_AU_squared"] / 9
                )
                squared_observations["source_richardson_v"].append(
                    source_coarse_fine["velocity_AU_per_julian_year_squared"] / 9
                )
                squared_observations["source_fine_ias_r"].append(
                    source_fine_reference["position_AU_squared"]
                )
                squared_observations["source_fine_ias_v"].append(
                    source_fine_reference["velocity_AU_per_julian_year_squared"]
                )
                contraction["source_position_AU"].append(
                    _exact_squared_contraction_passes(
                        source_fine_reference["position_AU_squared"],
                        source_coarse_reference["position_AU_squared"],
                        source_floors["position_AU_squared"],
                    )
                )
                contraction["source_velocity_AU_per_julian_year"].append(
                    _exact_squared_contraction_passes(
                        source_fine_reference[
                            "velocity_AU_per_julian_year_squared"
                        ],
                        source_coarse_reference[
                            "velocity_AU_per_julian_year_squared"
                        ],
                        source_floors[
                            "velocity_AU_per_julian_year_squared"
                        ],
                    )
                )
            coarse_energy = _fraction_from_record(
                coarse["summary"]["maximum_relative_massive_energy_drift_exact"],
                "coarse JX all-node energy drift",
            )
            fine_energy = _fraction_from_record(
                fine["summary"]["maximum_relative_massive_energy_drift_exact"],
                "fine JX all-node energy drift",
            )
            contraction["massive_energy"].append(
                _exact_energy_contraction_passes(fine_energy, coarse_energy)
            )
            for lane in (coarse, fine):
                exact_observations["jx_energy"].append(
                    _fraction_from_record(
                        lane["summary"]["maximum_relative_massive_energy_drift_exact"],
                        "JX energy drift",
                    )
                )
                float_observations["jx_angular"].append(_exact_float_hex(lane["summary"]["maximum_relative_massive_angular_momentum_drift_hex"], "JX angular drift"))
            for lane in (reduced, resolved):
                exact_observations["ias_energy"].append(
                    _fraction_from_record(
                        lane["summary"]["maximum_relative_massive_energy_drift_exact"],
                        "IAS15 energy drift",
                    )
                )
                float_observations["ias_angular"].append(_exact_float_hex(lane["summary"]["maximum_relative_massive_angular_momentum_drift_hex"], "IAS15 angular drift"))
                guard = lane.get("accepted_step_guard")
                if type(guard) is not dict:
                    raise Stage1QualificationError("IAS15 accepted-step guard is missing")
                endpoint = _exact_float_hex(
                    guard.get("minimum_source_tracer_endpoint_distance_AU_hex"),
                    "IAS15 minimum endpoint distance",
                )
                chord = _exact_float_hex(
                    guard.get("minimum_source_tracer_chord_distance_AU_hex"),
                    "IAS15 minimum chord distance",
                )
                initial_count = guard.get("initial_observation_count")
                heartbeat_count = guard.get("heartbeat_callback_count")
                total_count = guard.get("total_observation_count")
                pair_count = guard.get("source_tracer_pair_observation_count")
                steps_done = lane.get("steps_done")
                contraction["ias15_accepted_step_guard"].append(
                    set(guard)
                    == {
                        "sampling", "initial_observation_count",
                        "heartbeat_callback_count", "integrate_call_count",
                        "duplicate_entry_record_count", "total_observation_count",
                        "source_tracer_pair_observation_count",
                        "minimum_source_tracer_endpoint_distance_AU_hex",
                        "minimum_source_tracer_chord_distance_AU_hex",
                        "encounter_radius_AU_hex", "outer_boundary_AU_hex",
                        "encounter_or_guard_event_count", "callback_exception",
                        "trace",
                    }
                    and guard.get("sampling")
                    == "INITIAL_PLUS_REBOUND_HEARTBEAT_ACCEPTED_STEPS_WITH_STRAIGHT_CHORD_BETWEEN_SAMPLES_NOT_CONTINUOUS_CERTIFIED_GEOMETRY"
                    and guard.get("encounter_radius_AU_hex") == float(1.0).hex()
                    and guard.get("outer_boundary_AU_hex") == float(4096.0).hex()
                    and guard.get("encounter_or_guard_event_count") == 0
                    and guard.get("callback_exception") is None
                    and type(guard.get("trace")) is dict
                    and guard["trace"].get("schema") == IAS_TRACE_SCHEMA
                    and guard["trace"].get("record_count") == total_count
                    and type(initial_count) is int
                    and initial_count == 1
                    and type(heartbeat_count) is int
                    and type(steps_done) is int
                    and heartbeat_count
                    == steps_done + HORIZON_JULIAN_YEARS
                    and type(guard.get("integrate_call_count")) is int
                    and guard.get("integrate_call_count")
                    == HORIZON_JULIAN_YEARS
                    and type(guard.get("duplicate_entry_record_count")) is int
                    and guard.get("duplicate_entry_record_count")
                    == HORIZON_JULIAN_YEARS
                    and type(total_count) is int
                    and total_count == initial_count + heartbeat_count
                    and type(pair_count) is int
                    and pair_count
                    == total_count
                    * TNO_STATES_PER_BATCH
                    * len(lane["checkpoints"][0]["massive_positions_AU"])
                    and endpoint > 1.0
                    and chord > 1.0
                )

    for batch_order in range(OBJECT_COUNT):
        for lane_id in (
            "rebound511_ias15_reduced5",
            "rebound511_ias15_resolved11",
        ):
            left = index[("DE440S", batch_order, lane_id)]
            right = index[("DE441", batch_order, lane_id)]
            for tracer_order in range(TNO_STATES_PER_BATCH):
                exact_observations["ephemeris_a"].append(
                    _comparison_metrics(left, right, tracer_order)["semimajor_axis"]
                )
                exact_observations["ephemeris_s"].append(
                    abs(
                        scattering[("DE440S", batch_order, lane_id, tracer_order)]
                        - scattering[("DE441", batch_order, lane_id, tracer_order)]
                    )
                )
        for ephemeris in ("DE440S", "DE441"):
            other_ephemeris = "DE441" if ephemeris == "DE440S" else "DE440S"
            for tracer_order in range(TNO_STATES_PER_BATCH):
                fine_here = scattering[(ephemeris, batch_order, "jx_reduced5_h_1_32", tracer_order)]
                reduced_here = scattering[(ephemeris, batch_order, "rebound511_ias15_reduced5", tracer_order)]
                resolved_here = scattering[(ephemeris, batch_order, "rebound511_ias15_resolved11", tracer_order)]
                reduced_other = scattering[(other_ephemeris, batch_order, "rebound511_ias15_reduced5", tracer_order)]
                resolved_other = scattering[(other_ephemeris, batch_order, "rebound511_ias15_resolved11", tracer_order)]
                chain_paths = (
                    (fine_here, reduced_here, resolved_here, resolved_other),
                    (fine_here, reduced_here, reduced_other, resolved_other),
                )
                for path in chain_paths:
                    chain_fraction = sum(
                        (
                            abs(left - right)
                            for left, right in zip(
                                path[:-1], path[1:], strict=True
                            )
                        ),
                        Fraction(0, 1),
                    )
                    scattering_chain_fractions.append(chain_fraction)

    gates: dict[str, object] = {
        "jx_richardson_semimajor_axis_tail": _exact_fraction_gate(exact_observations["richardson_a"], NUMERICAL_TAIL_BUDGET_RATIONAL),
        "jx_richardson_position_tail_AU": _exact_squared_gate(squared_observations["richardson_r"], CARTESIAN_POSITION_BUDGET_AU_RATIONAL),
        "jx_richardson_velocity_tail_AU_per_julian_year": _exact_squared_gate(squared_observations["richardson_v"], CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL),
        "jx_fine_vs_ias15_reduced_semimajor_axis": _exact_fraction_gate(exact_observations["fine_ias_a"], INTEGRATOR_BUDGET_RATIONAL),
        "jx_fine_vs_ias15_reduced_position_AU": _exact_squared_gate(squared_observations["fine_ias_r"], CARTESIAN_POSITION_BUDGET_AU_RATIONAL),
        "jx_fine_vs_ias15_reduced_velocity_AU_per_julian_year": _exact_squared_gate(squared_observations["fine_ias_v"], CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL),
        "jx_fine_vs_ias15_reduced_scattering_diagnostic": _exact_fraction_gate(exact_observations["fine_ias_s"], INTEGRATOR_BUDGET_RATIONAL),
        "jx_contraction_semimajor_axis": _boolean_gate(contraction["semimajor_axis"], rule="FINE_ERROR_LE_64_ULP_FLOOR_OR_COARSE_ERROR_GE_THREE_TIMES_FINE_ERROR"),
        "jx_contraction_position_AU": _boolean_gate(contraction["position_AU"], rule="EXACT_FINE_SQUARED_LE_EXACT_64_ULP_FLOOR_SQUARED_OR_EXACT_COARSE_SQUARED_GE_NINE_TIMES_FINE_SQUARED"),
        "jx_contraction_velocity_AU_per_julian_year": _boolean_gate(contraction["velocity_AU_per_julian_year"], rule="EXACT_FINE_SQUARED_LE_EXACT_64_ULP_FLOOR_SQUARED_OR_EXACT_COARSE_SQUARED_GE_NINE_TIMES_FINE_SQUARED"),
        "jx_reduced_source_richardson_position_tail_AU": _exact_squared_gate(squared_observations["source_richardson_r"], CARTESIAN_POSITION_BUDGET_AU_RATIONAL),
        "jx_reduced_source_richardson_velocity_tail_AU_per_julian_year": _exact_squared_gate(squared_observations["source_richardson_v"], CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL),
        "jx_reduced_source_fine_vs_ias15_position_AU": _exact_squared_gate(squared_observations["source_fine_ias_r"], CARTESIAN_POSITION_BUDGET_AU_RATIONAL),
        "jx_reduced_source_fine_vs_ias15_velocity_AU_per_julian_year": _exact_squared_gate(squared_observations["source_fine_ias_v"], CARTESIAN_VELOCITY_BUDGET_AU_PER_YEAR_RATIONAL),
        "jx_reduced_source_contraction_position_AU": _boolean_gate(contraction["source_position_AU"], rule="EXACT_FINE_SQUARED_LE_EXACT_64_ULP_FLOOR_SQUARED_OR_EXACT_COARSE_SQUARED_GE_NINE_TIMES_FINE_SQUARED"),
        "jx_reduced_source_contraction_velocity_AU_per_julian_year": _boolean_gate(contraction["source_velocity_AU_per_julian_year"], rule="EXACT_FINE_SQUARED_LE_EXACT_64_ULP_FLOOR_SQUARED_OR_EXACT_COARSE_SQUARED_GE_NINE_TIMES_FINE_SQUARED"),
        "jx_fine_massive_energy_contraction": _boolean_gate(contraction["massive_energy"], rule="FINE_ALL_NODE_DRIFT_LE_MAX_COARSE_ALL_NODE_DRIFT_DIVIDED_BY_THREE,2_POW_MINUS_46"),
        "ias15_accepted_step_source_tracer_guard": _boolean_gate(contraction["ias15_accepted_step_guard"], rule="ZERO_EVENTS_AND_MINIMUM_ENDPOINT_AND_CHORD_DISTANCE_STRICTLY_GREATER_THAN_ONE_AU_WITH_COMPLETE_CALLBACK_ROSTER"),
        "ias15_reduced_vs_resolved11_semimajor_axis": _exact_fraction_gate(exact_observations["model_a"], MODEL_REDUCTION_BUDGET_RATIONAL),
        "ias15_reduced_vs_resolved11_scattering_diagnostic": _exact_fraction_gate(exact_observations["model_s"], MODEL_REDUCTION_BUDGET_RATIONAL),
        "de440s_vs_de441_ias15_semimajor_axis": _exact_fraction_gate(exact_observations["ephemeris_a"], EPHEMERIS_BUDGET_RATIONAL),
        "de440s_vs_de441_ias15_scattering_diagnostic": _exact_fraction_gate(exact_observations["ephemeris_s"], EPHEMERIS_BUDGET_RATIONAL),
        "scattering_disagreement_chain": _exact_fraction_gate(scattering_chain_fractions, TOTAL_CHAIN_BUDGET_RATIONAL),
        "jx_reduced_max_relative_energy_drift": _exact_fraction_gate(exact_observations["jx_energy"], Fraction(1, 100_000_000)),
        "jx_reduced_max_relative_angular_momentum_drift": _threshold_gate(float_observations["jx_angular"], Fraction(1, 10_000_000_000)),
        "ias15_max_relative_energy_drift": _exact_fraction_gate(exact_observations["ias_energy"], Fraction(1, 10_000_000_000)),
        "ias15_max_relative_angular_momentum_drift": _threshold_gate(float_observations["ias_angular"], Fraction(1, 100_000_000_000)),
    }
    return {
        "gates": gates,
        "all_predeclared_gates_passed": all(value["passed"] for value in gates.values()),
        "qualification_only": True,
        "preliminary_scattering_bit_used_for_acceptance": False,
        "classification_generated": False,
    }


def _execute_worker(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    de441_root: Path = DE441_ROOT,
    rebound_target: Path = DEFAULT_REBOUND_TARGET,
    import_probe_receipt_path: Path,
) -> dict[str, object]:
    fresh_child_interpreter = _fresh_child_interpreter_receipt()
    fresh_registry_receipt = _reject_preloaded_runtime_modules()
    import_probe_path = Path(os.path.abspath(
        os.fspath(import_probe_receipt_path)
    ))
    _probe_payload, import_probe_receipt = _load_json(
        import_probe_path, "fresh import-only runtime preflight"
    )
    _validate_import_probe_receipt(
        import_probe_receipt, expected_path=import_probe_path
    )
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.is_dir() or root.is_symlink() or {entry.name for entry in root.iterdir()} != {"PROTOCOL.json"}:
        raise Stage1QualificationError("execute requires an exact prepared root and no prior result")
    protocol = validate_prepared(root)
    custody = validate_execution_prerequisites(de441_root)
    bridge = _load_json(BRIDGE_STATES_PATH, "bridge states")[1]
    batch_lanes: list[dict[str, object]] = []
    midpoint_payload_builder = bytearray()
    midpoint_records: list[dict[str, object]] = []
    selected_suffixes: dict[
        tuple[str, int, str], tuple[bytes, Mapping[str, Any]]
    ] = {}
    evidence_temporary = tempfile.TemporaryDirectory(
        prefix=".stage1-v2-evidence-", dir=root.parent
    )
    evidence_root = Path(evidence_temporary.name)
    jx_source_path = evidence_root / "JX_SOURCE_NODES.bin"
    ias_trace_path = evidence_root / "IAS15_ACCEPTED_STEPS.bin"
    jx_source_handle = jx_source_path.open("xb")
    ias_trace_handle = ias_trace_path.open("xb")
    jx_source_records: list[dict[str, object]] = []
    common_source_gm_payload: bytes | None = None
    # Runtime imports happen only after every retained input has passed, and
    # the source-only cache guard remains active through every JX import/run.
    with _source_only_runtime_import_guard(
        root / ".stage1-v2-runtime-pycache-must-remain-absent"
    ) as import_guard:
        cp, rebound, rebound_target_receipt = _load_dynamics_runtime(rebound_target)
        for ephemeris in ("DE440S", "DE441"):
            for batch in _batch_states(bridge, custody, ephemeris):
                lanes = [
                    _run_jx_lane(batch, 16, cp),
                    _run_jx_lane(batch, 32, cp),
                    _run_rebound_lane(
                        batch, resolved11=False, rebound=rebound,
                        trace_handle=ias_trace_handle,
                    ),
                    _run_rebound_lane(
                        batch, resolved11=True, rebound=rebound,
                        trace_handle=ias_trace_handle,
                    ),
                ]
                for lane in lanes:
                    if lane.get("engine") == "JX_RESTRICTED_WH":
                        midpoint_payload = lane.pop("_jx_midpoint_payload", None)
                        midpoint_revalidated = lane.pop(
                            "_jx_midpoint_revalidated_before_continuation", None
                        )
                        selected_suffix = lane.pop("_jx_selected_suffix_payload", None)
                        selected_ledger = lane.pop("_jx_selected_suffix_ledger", None)
                        source_payload = lane.pop(
                            "_jx_source_evidence_payload", None
                        )
                        source_contract = lane.pop("_jx_source_contract", None)
                        source_gm_payload = lane.pop(
                            "_jx_source_gm_payload", None
                        )
                        if (
                            type(midpoint_payload) is not bytes
                            or midpoint_revalidated is not True
                            or type(source_payload) is not bytes
                            or type(source_contract) is not dict
                            or type(source_gm_payload) is not bytes
                        ):
                            raise Stage1QualificationError(
                                "JX midpoint or source evidence is missing"
                            )
                        if common_source_gm_payload is None:
                            common_source_gm_payload = source_gm_payload
                            if len(source_gm_payload) != JX_SOURCE_GM_BYTES:
                                raise Stage1QualificationError(
                                    "JX source GM payload width changed"
                                )
                            jx_source_handle.write(source_gm_payload)
                        elif source_gm_payload != common_source_gm_payload:
                            raise Stage1QualificationError(
                                "JX lanes used different common GM bytes"
                            )
                        source_offset = int(jx_source_handle.tell())
                        jx_source_handle.write(source_payload)
                        node_count = (
                            HORIZON_JULIAN_YEARS * lane["steps_per_year"] + 1
                        )
                        source_block_bytes = node_count * JX_SOURCE_COUNT * 3 * 8
                        source_record: dict[str, object] = {
                            "source_order": len(jx_source_records),
                            "ephemeris": ephemeris,
                            "batch_order": batch.batch_order,
                            "object_id": batch.object_id,
                            "lane_id": lane["lane_id"],
                            "steps_per_year": lane["steps_per_year"],
                            "node_count": node_count,
                            "source_count": JX_SOURCE_COUNT,
                            "byte_offset": source_offset,
                            "byte_length": len(source_payload),
                            "positions_block_bytes": source_block_bytes,
                            "velocities_block_bytes": source_block_bytes,
                            "lane_payload_sha256": _sha256(source_payload),
                            "positions_sha256": _sha256(
                                source_payload[:source_block_bytes]
                            ),
                            "velocities_sha256": _sha256(
                                source_payload[source_block_bytes:]
                            ),
                            "contract": source_contract,
                            "full_source_tile_sha256": lane[
                                "full_source_tile_sha256"
                            ],
                            "annual_source_slice_sha256": lane[
                                "annual_source_slice_sha256"
                            ],
                        }
                        source_record["record_content_sha256"] = _sha256(
                            JX_SOURCE_DOMAIN
                            + _canonical_json(source_record)
                            + source_payload
                        )
                        jx_source_records.append(source_record)
                        lane["source_node_evidence"] = {
                            "source_order": source_record["source_order"],
                            "byte_offset": source_offset,
                            "byte_length": len(source_payload),
                            "lane_payload_sha256": source_record[
                                "lane_payload_sha256"
                            ],
                            "record_content_sha256": source_record[
                                "record_content_sha256"
                            ],
                        }
                        offset = len(midpoint_payload_builder)
                        midpoint_payload_builder.extend(midpoint_payload)
                        identity = {
                            "midpoint_order": len(midpoint_records),
                            "ephemeris": ephemeris,
                            "batch_order": batch.batch_order,
                            "object_id": batch.object_id,
                            "lane_id": lane["lane_id"],
                            "steps_per_year": lane["steps_per_year"],
                            "midpoint_year": JX_MIDPOINT_YEAR,
                            "global_source_step": JX_MIDPOINT_YEAR * lane["steps_per_year"],
                            "byte_offset": offset,
                            "byte_length": JX_MIDPOINT_RECORD_BYTES,
                            "record_sha256": _sha256(midpoint_payload),
                            "revalidated_before_continuation": midpoint_revalidated,
                        }
                        identity["record_content_sha256"] = _sha256(
                            MIDPOINT_DOMAIN + _canonical_json(identity) + midpoint_payload
                        )
                        midpoint_records.append(identity)
                        selection = (ephemeris, lane["steps_per_year"], batch.object_id)
                        if selection in JX_FRESH_PROCESS_RESTART_CASES:
                            if type(selected_suffix) is not bytes or type(selected_ledger) is not dict:
                                raise Stage1QualificationError("selected JX restart evidence is missing")
                            selected_ledger = dict(selected_ledger)
                            selected_ledger["fresh_process_registry"] = (
                                fresh_registry_receipt
                            )
                            selected_suffixes[selection] = (
                                selected_suffix,
                                selected_ledger,
                            )
                        elif selected_suffix is not None or selected_ledger is not None:
                            raise Stage1QualificationError("unselected JX lane emitted restart evidence")
                    elif any(type(key) is str and key.startswith("_") for key in lane):
                        raise Stage1QualificationError("private runtime field leaked from IAS15 lane")
                    batch_lanes.append(
                        {
                            "ephemeris": ephemeris,
                            "batch_order": batch.batch_order,
                            "object_id": batch.object_id,
                            "native_epoch_jd_tdb_hex": batch.epoch_jd_tdb.hex(),
                            "lane": lane,
                        }
                    )
        midpoint_payload = bytes(midpoint_payload_builder)
        if (
            len(midpoint_records) != JX_MIDPOINT_RECORD_COUNT
            or len(midpoint_payload) != EXPECTED_JX_MIDPOINT_BYTES
        ):
            raise Stage1QualificationError("JX midpoint artifact roster or length changed")
        restart_receipts, restart_stream_payload, restart_stream_records = (
            _run_fresh_process_restart_checks(
            midpoint_payload, midpoint_records, selected_suffixes
            )
        )
        runtime = _runtime_identity(
            rebound, cp, rebound_target_receipt, import_guard,
            fresh_registry_receipt, fresh_child_interpreter,
            import_probe_receipt,
        )
    jx_source_handle.flush()
    os.fsync(jx_source_handle.fileno())
    jx_source_handle.close()
    ias_trace_handle.flush()
    os.fsync(ias_trace_handle.fileno())
    ias_trace_handle.close()
    if import_guard.get("verified_absent_after") is not True:
        raise Stage1QualificationError("source-only runtime import guard did not close")
    jx_source_digest = _stream_file_digests(
        jx_source_path,
        "Stage-1 JX source nodes",
        maximum=EXPECTED_JX_SOURCE_BYTES,
        content_domain=JX_SOURCE_DOMAIN,
    )
    if (
        jx_source_digest["bytes"] != EXPECTED_JX_SOURCE_BYTES
        or len(jx_source_records) != JX_SOURCE_RECORD_COUNT
        or common_source_gm_payload is None
    ):
        raise Stage1QualificationError("JX source evidence file layout changed")
    ias_trace_digest = _stream_file_digests(
        ias_trace_path,
        "Stage-1 IAS15 accepted-step trace",
        maximum=IAS_TRACE_MAXIMUM_BYTES,
        content_domain=IAS_TRACE_DOMAIN,
    )
    arrays_payload, compact_batch_lanes, gate_batch_lanes = materialize_cartesian_payload(
        batch_lanes
    )
    if len(arrays_payload) != EXPECTED_CARTESIAN_ARRAY_BYTES:
        raise Stage1QualificationError("Cartesian payload byte length changed")
    gate_result = evaluate_qualification_gates(gate_batch_lanes)
    midpoint_metadata = {
        "schema": MIDPOINT_SCHEMA,
        "relative_path": "JX_MIDPOINTS.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_PRIMARY_RELATIVE_TRACER_POSITIONS_THEN_VELOCITIES",
        "midpoint_year": JX_MIDPOINT_YEAR,
        "byte_length": len(midpoint_payload),
        "file_sha256": _sha256(midpoint_payload),
        "content_sha256": _sha256(MIDPOINT_DOMAIN + midpoint_payload),
        "record_count": len(midpoint_records),
        "record_bytes": JX_MIDPOINT_RECORD_BYTES,
        "record_order": "EPHEMERIS_THEN_BATCH_THEN_JX_STEP_16_32",
        "records": midpoint_records,
        "fresh_process_restart_cases": restart_receipts,
    }
    restart_stream_metadata = {
        "schema": RESTART_STREAM_SCHEMA,
        "relative_path": "JX_RESTART_STREAMS.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64",
        "byte_length": len(restart_stream_payload),
        "file_sha256": _sha256(restart_stream_payload),
        "content_sha256": _sha256(
            RESTART_STREAM_DOMAIN + restart_stream_payload
        ),
        "stream_count": len(restart_stream_records),
        "record_bytes": JX_MIDPOINT_RECORD_BYTES,
        "records_per_stream": JX_RESTART_SUFFIX_CHECKPOINT_COUNT,
        "stream_bytes": JX_RESTART_SUFFIX_BYTES,
        "order": "CASE_THEN_MONOLITHIC_RESUMED_THEN_YEAR_500_THROUGH_1000_THEN_TRACER_POSITIONS_VELOCITIES",
        "records": restart_stream_records,
    }
    jx_source_metadata = {
        "schema": JX_SOURCE_SCHEMA,
        "relative_path": "JX_SOURCE_NODES.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64",
        "byte_length": jx_source_digest["bytes"],
        "file_sha256": jx_source_digest["sha256"],
        "content_sha256": jx_source_digest["content_sha256"],
        "common_gm_byte_offset": 0,
        "common_gm_byte_length": JX_SOURCE_GM_BYTES,
        "common_gm_sha256": _sha256(common_source_gm_payload),
        "lane_count": len(jx_source_records),
        "order": "COMMON_REDUCED_GMS_THEN_EPHEMERIS_BATCH_STEP_16_32_POSITIONS_THEN_VELOCITIES",
        "records": jx_source_records,
    }
    ias_trace_metadata = {
        "schema": IAS_TRACE_SCHEMA,
        "relative_path": "IAS15_ACCEPTED_STEPS.bin",
        "encoding": "FIXED_PER_LANE_RAW_LITTLE_ENDIAN_RECORDS",
        "byte_length": ias_trace_digest["bytes"],
        "file_sha256": ias_trace_digest["sha256"],
        "content_sha256": ias_trace_digest["content_sha256"],
        "lane_count": OBJECT_COUNT * 2 * 2,
        "order": "EPHEMERIS_BATCH_THEN_REDUCED5_RESOLVED11_CALLBACK_ORDER",
    }
    arrays_metadata = {
        "schema": ARRAY_SCHEMA,
        "relative_path": "ARRAYS.bin",
        "encoding": "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER",
        "byte_length": len(arrays_payload),
        "file_sha256": _sha256(arrays_payload),
        "content_sha256": _sha256(ARRAY_DOMAIN + arrays_payload),
        "batch_lane_count": len(compact_batch_lanes),
        "ordering": "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL",
        "all_vectors_origin": "LANE_INSTANTANEOUS_MASSIVE_SYSTEM_COMMON_GM_BARYCENTER",
        "position_unit": "AU",
        "velocity_unit": "AU_PER_JULIAN_YEAR",
    }
    result: dict[str, object] = {
        "schema": RESULT_SCHEMA,
        "classification": CLASSIFICATION,
        "claim_ceiling": CLAIM_CEILING,
        "status": "QUALIFICATION_EXECUTED_PASS" if gate_result["all_predeclared_gates_passed"] else "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS",
        "protocol_content_sha256": protocol["content_sha256"],
        "protocol_file_sha256": _sha256(_regular_bytes(root / "PROTOCOL.json", "Stage-1 protocol")),
        "runtime_identity": runtime,
        "de441_custody_content_sha256": custody["content_sha256"],
        "batch_lane_count": len(compact_batch_lanes),
        "cartesian_arrays": arrays_metadata,
        "jx_midpoint_checkpoints": midpoint_metadata,
        "jx_restart_streams": restart_stream_metadata,
        "jx_source_nodes": jx_source_metadata,
        "ias15_accepted_step_trace": ias_trace_metadata,
        "batch_lanes": compact_batch_lanes,
        "qualification_gates": gate_result,
        "claims": {
            **PROTOCOL_CLAIMS,
            "dynamics_executed": True,
            "one_thousand_year_qualification_executed": True,
            "de440s_de441_sensitivity_executed": True,
        },
        "outcomes_generated": True,
        "dynamical_classification_generated": False,
        "resonance_classification_generated": False,
        "science_result": False,
    }
    result["content_sha256"] = _content_sha256(RESULT_DOMAIN, result)
    _publish_result_artifacts(
        root,
        (
            ("JX_MIDPOINTS.bin", midpoint_payload),
            ("JX_RESTART_STREAMS.bin", restart_stream_payload),
            ("JX_SOURCE_NODES.bin", jx_source_path),
            ("IAS15_ACCEPTED_STEPS.bin", ias_trace_path),
            ("ARRAYS.bin", arrays_payload),
            ("RESULT.json", _canonical_json(result)),
        ),
    )
    evidence_temporary.cleanup()
    return result


def execute(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    de441_root: Path = DE441_ROOT,
    rebound_target: Path = DEFAULT_REBOUND_TARGET,
) -> dict[str, object]:
    """Execute only in a newly spawned interpreter with an empty runtime registry."""

    _reject_preloaded_runtime_modules()
    root = Path(os.path.abspath(os.fspath(output_root)))
    validate_prepared(root)
    # Deliberately validate every external input before a child can import a
    # dynamics runtime.  The worker repeats these checks independently.
    validate_execution_prerequisites(de441_root)
    with tempfile.TemporaryDirectory(
        prefix="jx-tno-stage1-v2-import-probe-"
    ) as directory:
        probe_path, _probe_receipt = _run_import_only_preflight(
            rebound_target, Path(directory)
        )
        completed = subprocess.run(
            [
                str(PROJECT_VENV_PYTHON), "-B", str(SCRIPT_PATH),
                "execute-worker", "--output-root", str(root),
                "--de441-root", str(Path(de441_root).resolve()),
                "--rebound-target", str(Path(rebound_target).resolve()),
                "--import-probe-receipt", str(probe_path),
            ],
            cwd=PROJECT_ROOT,
            env=_production_child_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=False,
            timeout=604_800,
        )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-8_000:]
        raise Stage1QualificationError(f"fresh Stage-1 execution worker failed: {detail}")
    return validate_output(root)


def validate_result_document(
    value: Mapping[str, Any],
    protocol: Mapping[str, Any],
    arrays_payload: bytes,
    midpoint_payload: bytes,
    restart_stream_payload: bytes,
    jx_source_path: Path,
    ias_trace_path: Path,
    bridge: Mapping[str, Any],
    custody: Mapping[str, Any],
    offline_numpy_module: types.ModuleType,
) -> str:
    if (
        type(offline_numpy_module) is not types.ModuleType
        or offline_numpy_module.__name__ != "numpy"
        or sys.modules.get("numpy") is not offline_numpy_module
    ):
        raise Stage1QualificationError(
            "result validation did not receive the bound offline NumPy module"
        )
    _guard_exact_json_integer_identities(value, "Stage-1 result")
    if type(value) is not dict or value.get("schema") != RESULT_SCHEMA:
        raise Stage1QualificationError("result schema changed")
    expected_keys = {
        "schema", "classification", "claim_ceiling", "status", "protocol_content_sha256",
        "protocol_file_sha256", "runtime_identity", "de441_custody_content_sha256",
        "batch_lane_count", "cartesian_arrays", "jx_midpoint_checkpoints",
        "jx_restart_streams", "jx_source_nodes",
        "ias15_accepted_step_trace", "batch_lanes", "qualification_gates", "claims",
        "outcomes_generated", "dynamical_classification_generated",
        "resonance_classification_generated", "science_result", "content_sha256",
    }
    if set(value) != expected_keys:
        raise Stage1QualificationError("result top-level shape changed")
    if (
        value.get("classification") != CLASSIFICATION
        or value.get("claim_ceiling") != CLAIM_CEILING
        or value.get("protocol_content_sha256") != protocol.get("content_sha256")
        or value.get("protocol_file_sha256") != _sha256(_canonical_json(protocol))
        or value.get("de441_custody_content_sha256") != EXPECTED_CONTENT_HASHES["de441_custody"]
        or value.get("batch_lane_count") != OBJECT_COUNT * 2 * len(LANE_SPECS)
        or value.get("dynamical_classification_generated") is not False
        or value.get("resonance_classification_generated") is not False
        or value.get("science_result") is not False
    ):
        raise Stage1QualificationError("result identity, count, or claim ceiling changed")
    _validate_runtime_identity(value.get("runtime_identity"))
    arrays = value.get("cartesian_arrays")
    if (
        type(arrays_payload) is not bytes
        or type(arrays) is not dict
        or set(arrays) != {
            "schema", "relative_path", "encoding", "byte_length", "file_sha256",
            "content_sha256", "batch_lane_count", "ordering", "all_vectors_origin",
            "position_unit", "velocity_unit",
        }
        or arrays.get("schema") != ARRAY_SCHEMA
        or arrays.get("relative_path") != "ARRAYS.bin"
        or arrays.get("encoding") != "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_NO_HEADER"
        or arrays.get("byte_length") != EXPECTED_CARTESIAN_ARRAY_BYTES
        or len(arrays_payload) != EXPECTED_CARTESIAN_ARRAY_BYTES
        or arrays.get("file_sha256") != _sha256(arrays_payload)
        or arrays.get("content_sha256") != _sha256(ARRAY_DOMAIN + arrays_payload)
        or arrays.get("batch_lane_count") != OBJECT_COUNT * 2 * len(LANE_SPECS)
        or arrays.get("ordering") != "EPHEMERIS_BATCH_LANE_YEAR_THEN_MASSIVE_POS_VEL_THEN_TRACER_POS_VEL"
        or arrays.get("all_vectors_origin") != "LANE_INSTANTANEOUS_MASSIVE_SYSTEM_COMMON_GM_BARYCENTER"
        or arrays.get("position_unit") != "AU"
        or arrays.get("velocity_unit") != "AU_PER_JULIAN_YEAR"
    ):
        raise Stage1QualificationError("Cartesian payload metadata or digest changed")
    midpoint = value.get("jx_midpoint_checkpoints")
    if (
        type(midpoint_payload) is not bytes
        or len(midpoint_payload) != EXPECTED_JX_MIDPOINT_BYTES
        or type(midpoint) is not dict
        or set(midpoint) != {
            "schema", "relative_path", "encoding", "midpoint_year", "byte_length",
            "file_sha256", "content_sha256", "record_count", "record_bytes",
            "record_order", "records", "fresh_process_restart_cases",
        }
        or midpoint.get("schema") != MIDPOINT_SCHEMA
        or midpoint.get("relative_path") != "JX_MIDPOINTS.bin"
        or midpoint.get("encoding") != "RAW_LITTLE_ENDIAN_IEEE754_BINARY64_PRIMARY_RELATIVE_TRACER_POSITIONS_THEN_VELOCITIES"
        or midpoint.get("midpoint_year") != JX_MIDPOINT_YEAR
        or midpoint.get("byte_length") != EXPECTED_JX_MIDPOINT_BYTES
        or midpoint.get("file_sha256") != _sha256(midpoint_payload)
        or midpoint.get("content_sha256") != _sha256(MIDPOINT_DOMAIN + midpoint_payload)
        or midpoint.get("record_count") != JX_MIDPOINT_RECORD_COUNT
        or midpoint.get("record_bytes") != JX_MIDPOINT_RECORD_BYTES
        or midpoint.get("record_order") != "EPHEMERIS_THEN_BATCH_THEN_JX_STEP_16_32"
    ):
        raise Stage1QualificationError("JX midpoint artifact metadata changed")
    records = midpoint.get("records")
    expected_midpoint_order = [
        (ephemeris, batch_order, steps)
        for ephemeris in ("DE440S", "DE441")
        for batch_order in range(OBJECT_COUNT)
        for steps in JX_STEPS_PER_YEAR
    ]
    if type(records) is not list or len(records) != JX_MIDPOINT_RECORD_COUNT:
        raise Stage1QualificationError("JX midpoint record roster changed")
    expected_offset = 0
    record_selection: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for record, (ephemeris, batch_order, steps) in zip(
        records, expected_midpoint_order, strict=True
    ):
        if type(record) is not dict:
            raise Stage1QualificationError("JX midpoint record type changed")
        batch = bridge["batches"][batch_order]
        expected_record_keys = {
            "midpoint_order", "ephemeris", "batch_order", "object_id", "lane_id",
            "steps_per_year", "midpoint_year", "global_source_step", "byte_offset",
            "byte_length", "record_sha256", "revalidated_before_continuation",
            "record_content_sha256",
        }
        if (
            set(record) != expected_record_keys
            or record.get("midpoint_order") != len(record_selection)
            or record.get("ephemeris") != ephemeris
            or record.get("batch_order") != batch_order
            or record.get("object_id") != batch["object_id"]
            or record.get("lane_id") != f"jx_reduced5_h_1_{steps}"
            or record.get("steps_per_year") != steps
            or record.get("midpoint_year") != JX_MIDPOINT_YEAR
            or record.get("global_source_step") != JX_MIDPOINT_YEAR * steps
            or record.get("byte_offset") != expected_offset
            or record.get("byte_length") != JX_MIDPOINT_RECORD_BYTES
            or record.get("revalidated_before_continuation") is not True
        ):
            raise Stage1QualificationError("JX midpoint record identity changed")
        payload = midpoint_payload[
            expected_offset : expected_offset + JX_MIDPOINT_RECORD_BYTES
        ]
        record_without_content = dict(record)
        record_content = record_without_content.pop("record_content_sha256", None)
        if (
            record.get("record_sha256") != _sha256(payload)
            or record_content
            != _sha256(MIDPOINT_DOMAIN + _canonical_json(record_without_content) + payload)
        ):
            raise Stage1QualificationError("JX midpoint record digest changed")
        _unpack_jx_internal_tracer_state(payload)
        record_selection[(ephemeris, steps, batch["object_id"])] = record
        expected_offset += JX_MIDPOINT_RECORD_BYTES
    if expected_offset != len(midpoint_payload):
        raise Stage1QualificationError("JX midpoint payload has trailing bytes")
    restart_receipts = midpoint.get("fresh_process_restart_cases")
    if type(restart_receipts) is not list or len(restart_receipts) != len(JX_FRESH_PROCESS_RESTART_CASES):
        raise Stage1QualificationError("fresh-process restart receipt roster changed")
    for restart_order, (receipt, selection) in enumerate(
        zip(restart_receipts, JX_FRESH_PROCESS_RESTART_CASES, strict=True)
    ):
        record = record_selection.get(selection)
        expected_restart_keys = {
            "restart_order", "ephemeris", "steps_per_year", "object_id",
            "batch_order", "midpoint_record_sha256",
            "worker_receipt_file_sha256", "worker_receipt_content_sha256",
            "suffix_file_sha256", "final_state_sha256",
            "suffix_checkpoint_count", "worker_request", "worker_receipt",
            "bitwise_midpoint_through_year_1000_match",
            "bitwise_final_state_match", "ledger_match", "fresh_process",
        }
        if (
            type(receipt) is not dict
            or type(record) is not dict
            or set(receipt) != expected_restart_keys
            or receipt.get("restart_order") != restart_order
            or (receipt.get("ephemeris"), receipt.get("steps_per_year"), receipt.get("object_id")) != selection
            or receipt.get("batch_order") != record["batch_order"]
            or receipt.get("midpoint_record_sha256") != record["record_sha256"]
            or receipt.get("suffix_checkpoint_count") != HORIZON_JULIAN_YEARS - JX_MIDPOINT_YEAR + 1
            or receipt.get("bitwise_midpoint_through_year_1000_match") is not True
            or receipt.get("bitwise_final_state_match") is not True
            or receipt.get("ledger_match") is not True
            or receipt.get("fresh_process") is not True
            or any(
                not _is_sha256(receipt.get(key))
                for key in (
                    "midpoint_record_sha256", "worker_receipt_file_sha256",
                    "worker_receipt_content_sha256", "suffix_file_sha256",
                    "final_state_sha256",
                )
            )
        ):
            raise Stage1QualificationError("fresh-process restart receipt changed")
        worker_receipt = receipt.get("worker_receipt")
        worker_request = receipt.get("worker_request")
        if (
            type(worker_request) is not dict
            or set(worker_request) != RESTART_REQUEST_KEYS
            or worker_request.get("schema")
            != "jx.tno.stage1-dynamics-qualification.restart-request.v2"
            or worker_request.get("ephemeris") != selection[0]
            or worker_request.get("batch_order") != record["batch_order"]
            or worker_request.get("object_id") != selection[2]
            or worker_request.get("steps_per_year") != selection[1]
            or worker_request.get("midpoint_year") != JX_MIDPOINT_YEAR
            or worker_request.get("midpoint_relative_path")
            != "JX_MIDPOINTS.bin"
            or worker_request.get("midpoint_byte_offset") != record["byte_offset"]
            or worker_request.get("midpoint_byte_length")
            != JX_MIDPOINT_RECORD_BYTES
            or worker_request.get("midpoint_record_sha256")
            != record["record_sha256"]
            or worker_request.get("content_sha256")
            != _content_sha256(RESTART_DOMAIN, worker_request)
        ):
            raise Stage1QualificationError("retained restart worker request changed")
        _validate_restart_worker_receipt(
            worker_receipt, value.get("runtime_identity")
        )
        if (
            worker_receipt.get("request_content_sha256")
            != worker_request["content_sha256"]
            or worker_receipt.get("request_file_sha256")
            != _sha256(_canonical_json(worker_request))
            or
            worker_receipt.get("midpoint_record_sha256")
            != record["record_sha256"]
            or worker_receipt.get("suffix_file_sha256")
            != receipt.get("suffix_file_sha256")
            or worker_receipt.get("final_state_sha256")
            != receipt.get("final_state_sha256")
            or worker_receipt.get("suffix_checkpoint_count")
            != receipt.get("suffix_checkpoint_count")
            or worker_receipt.get("content_sha256")
            != receipt.get("worker_receipt_content_sha256")
            or _sha256(_canonical_json(worker_receipt))
            != receipt.get("worker_receipt_file_sha256")
        ):
            raise Stage1QualificationError(
                "retained restart worker receipt cross-binding changed"
            )
    lanes = value.get("batch_lanes")
    if type(lanes) is not list:
        raise Stage1QualificationError("result batch lanes must be a list")
    expected_order = [(ephemeris, batch, spec["lane_id"]) for ephemeris in ("DE440S", "DE441") for batch in range(OBJECT_COUNT) for spec in LANE_SPECS]
    observed_order: list[tuple[str, int, str]] = []
    for record in lanes:
        if type(record) is not dict or type(record.get("lane")) is not dict:
            raise Stage1QualificationError("batch-lane record type changed")
        lane = record["lane"]
        if "checkpoints" in lane or type(lane.get("cartesian_array")) is not dict:
            raise Stage1QualificationError("result must bind compact lanes to Cartesian arrays")
        if lane.get("engine") == "JX_RESTRICTED_WH":
            fused = lane.get("fused_tracer_runtime_identity")
            source_build = lane.get("source_build_runtime_identity")
            runtime = value["runtime_identity"]
            cupy = runtime["cupy_cuda"]
            native = runtime["native_wisdom_holman_module"]
            native_identity = runtime[
                "native_wisdom_holman_runtime_identity"
            ]
            source_map = (
                source_build.get("source_map_runtime_identity")
                if type(source_build) is dict
                else None
            )
            extension = (
                source_map.get("extension_binary")
                if type(source_map) is dict
                else None
            )
            if (
                type(fused) is not dict
                or fused.get("cupy_version") != cupy["cupy_version"]
                or fused.get("cuda_runtime_version")
                != cupy["cuda_runtime_version"]
                or fused.get("cuda_local_runtime_version")
                != cupy["cuda_local_runtime_version"]
                or fused.get("cuda_driver_version")
                != cupy["cuda_driver_version"]
                or fused.get("device_index") != cupy["cuda_device_id"]
                or fused.get("device_name") != cupy["cuda_device_name"]
                or fused.get("device_uuid") != cupy["cuda_device_uuid"]
                or fused.get("device_compute_capability")
                != cupy["cuda_compute_capability"]
                or fused.get("device_pci_domain_bus_device")
                != cupy["cuda_pci_domain_bus_device"]
                or _canonical_json(source_map)
                != _canonical_json(native_identity)
                or _canonical_json(
                    _numpy_runtime_cross_binding(
                        source_build.get("numpy_runtime_identity")
                    )
                )
                != _canonical_json(
                    _numpy_runtime_cross_binding(
                        runtime.get("numpy_runtime_identity")
                    )
                )
                or type(extension) is not dict
                or extension.get("path") != native["path"]
                or extension.get("size_bytes") != native["bytes"]
                or extension.get("sha256") != native["sha256"]
            ):
                raise Stage1QualificationError(
                    "JX lane runtime identity does not cross-bind the result runtime"
                )
        observed_order.append((record.get("ephemeris"), record.get("batch_order"), lane.get("lane_id")))
    if observed_order != expected_order:
        raise Stage1QualificationError("result batch-lane order changed")
    reconstructed = reconstruct_gate_lanes(
        arrays_payload, lanes, bridge, custody
    )
    with _stable_read_handle(
        jx_source_path,
        "retained JX source nodes",
        maximum=EXPECTED_JX_SOURCE_BYTES,
    ) as jx_source_handle, _stable_read_handle(
        ias_trace_path,
        "retained IAS15 accepted-step trace",
        maximum=IAS_TRACE_MAXIMUM_BYTES,
    ) as ias_trace_handle:
        _validate_jx_source_evidence(
            jx_source_handle, value.get("jx_source_nodes"), reconstructed,
            bridge, custody, offline_numpy_module,
        )
        _validate_restart_stream_evidence(
            restart_stream_payload,
            value.get("jx_restart_streams"),
            midpoint_payload,
            midpoint,
            jx_source_handle,
            value["jx_source_nodes"],
            reconstructed,
            bridge,
            custody,
        )
        _validate_ias_trace_evidence(
            ias_trace_handle, value.get("ias15_accepted_step_trace"),
            reconstructed, bridge, custody,
        )
    recomputed = evaluate_qualification_gates(reconstructed)
    if _canonical_json(recomputed) != _canonical_json(value.get("qualification_gates")):
        raise Stage1QualificationError("qualification gates differ from retained observables")
    expected_status = (
        "QUALIFICATION_EXECUTED_PASS"
        if recomputed["all_predeclared_gates_passed"]
        else "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS"
    )
    expected_claims = {
        **PROTOCOL_CLAIMS,
        "dynamics_executed": True,
        "one_thousand_year_qualification_executed": True,
        "de440s_de441_sensitivity_executed": True,
    }
    if (
        value.get("status") != expected_status
        or value.get("claims") != expected_claims
        or value.get("outcomes_generated") is not True
    ):
        raise Stage1QualificationError("result status or execution claims differ from gates")
    if value.get("content_sha256") != _content_sha256(RESULT_DOMAIN, value):
        raise Stage1QualificationError("result content digest changed")
    return str(value["content_sha256"])


def validate_output(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.is_dir() or root.is_symlink() or {entry.name for entry in root.iterdir()} != {
        "PROTOCOL.json", "JX_MIDPOINTS.bin", "JX_RESTART_STREAMS.bin",
        "JX_SOURCE_NODES.bin", "IAS15_ACCEPTED_STEPS.bin", "ARRAYS.bin",
        "RESULT.json",
    }:
        raise Stage1QualificationError("executed output root roster differs")
    protocol_payload, protocol = _load_json(root / "PROTOCOL.json", "Stage-1 protocol")
    expected_protocol = build_protocol()
    if protocol_payload != _canonical_json(expected_protocol):
        raise Stage1QualificationError("retained protocol differs from reconstruction")
    arrays_payload = _regular_bytes(
        root / "ARRAYS.bin", "Stage-1 Cartesian arrays", maximum=200_000_000
    )
    midpoint_payload = _regular_bytes(
        root / "JX_MIDPOINTS.bin", "Stage-1 JX midpoints", maximum=100_000
    )
    restart_stream_payload = _regular_bytes(
        root / "JX_RESTART_STREAMS.bin", "Stage-1 JX restart streams",
        maximum=2_000_000,
    )
    _result_payload, result = _load_json(root / "RESULT.json", "Stage-1 result")
    bridge, custody, _bindings = _verify_input_documents()
    runtime_identity = result.get("runtime_identity")
    retained_numpy_identity = (
        runtime_identity.get("numpy_runtime_identity")
        if type(runtime_identity) is dict else None
    )
    with _bound_offline_numpy_runtime(retained_numpy_identity) as numpy_module:
        validate_result_document(
            result, protocol, arrays_payload, midpoint_payload,
            restart_stream_payload, root / "JX_SOURCE_NODES.bin",
            root / "IAS15_ACCEPTED_STEPS.bin", bridge, custody,
            numpy_module,
        )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "protocol", "prepare", "execute", "execute-worker",
            "import-probe-worker", "source-builder-semantic-preflight",
            "validate-prepared", "validate", "resume-worker",
        ),
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--de441-root", type=Path, default=DE441_ROOT)
    parser.add_argument("--rebound-target", type=Path, default=DEFAULT_REBOUND_TARGET)
    parser.add_argument("--resume-request", type=Path)
    parser.add_argument("--resume-output-root", type=Path)
    parser.add_argument("--import-probe-receipt", type=Path)
    parser.add_argument("--probe-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "resume-worker":
            if arguments.resume_request is None or arguments.resume_output_root is None:
                raise Stage1QualificationError("resume-worker requires exact request and output paths")
            value = _run_jx_restart_worker(
                arguments.resume_request, arguments.resume_output_root
            )
        elif arguments.command == "import-probe-worker":
            if arguments.probe_output is None:
                raise Stage1QualificationError(
                    "import-probe-worker requires an output path"
                )
            value = _import_probe_worker(
                arguments.rebound_target, arguments.probe_output
            )
        elif arguments.command == "source-builder-semantic-preflight":
            value = cpu_source_builder_time_scale_semantic_preflight()
        elif arguments.command == "protocol":
            value = build_protocol()
        elif arguments.command == "prepare":
            value = prepare(arguments.output_root)
        elif arguments.command == "execute":
            value = execute(
                arguments.output_root,
                de441_root=arguments.de441_root,
                rebound_target=arguments.rebound_target,
            )
        elif arguments.command == "execute-worker":
            if arguments.import_probe_receipt is None:
                raise Stage1QualificationError(
                    "execute-worker requires the retained import-probe receipt"
                )
            value = _execute_worker(
                arguments.output_root,
                de441_root=arguments.de441_root,
                rebound_target=arguments.rebound_target,
                import_probe_receipt_path=arguments.import_probe_receipt,
            )
        elif arguments.command == "validate-prepared":
            value = validate_prepared(arguments.output_root)
        else:
            value = validate_output(arguments.output_root)
    except Stage1QualificationError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
