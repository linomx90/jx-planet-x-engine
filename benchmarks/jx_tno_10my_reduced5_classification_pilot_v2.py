#!/usr/bin/env python3
"""Freeze the v2 10 Myr TNO pilot without executing it.

This artifact deliberately stops at a prepare-only boundary.  It binds the
retained 1 kyr reduced-five h64 closure, the 108 native-epoch holdout states,
the separately versioned 36:1-capable classifier, and the native short/long
checkpoint grids needed for an independent broad-class crosscheck.  It does
not run dynamics, classify a trajectory, or make a Planet Nine claim.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import types
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_10my_reduced5_classification_pilot_v2.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_10my_reduced5_classification_pilot_v2.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_10my_reduced5_classification_pilot_v2"

SCHEMA = "jx.tno.10my-reduced5-classification-pilot.protocol.v2"
DOMAIN = b"jx.tno.10my-reduced5-classification-pilot.protocol.v2\0"
CLASSIFICATION = "DE441_REDUCED5_H64_10MY_108_STATE_CLASSIFICATION_PILOT_V2_PREPARE_ONLY"

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
HOLDOUT_CLONE_INDICES = (0, 18, 36, 54, 73, 91, 109, 127)
REDUCED_SOURCE_IDS = (
    "SUN_WITH_TERRESTRIAL_MASSES_FOLDED_IN",
    "JUPITER_BARYCENTER",
    "SATURN_BARYCENTER",
    "URANUS_BARYCENTER",
    "NEPTUNE_BARYCENTER",
)

HORIZON_JULIAN_YEARS = 10_000_000
STEP_JULIAN_YEARS = Fraction(1, 64)
CLASSIFIER_CADENCE_JULIAN_YEARS = 200
CLASSIFIER_NODE_COUNT = 50_001
ML_SHORT_HORIZON_JULIAN_YEARS = 500_000
ML_SHORT_CADENCE_JULIAN_YEARS = 50
ML_SHORT_NODE_COUNT = 10_001
ML_LONG_CADENCE_JULIAN_YEARS = 1_000
ML_LONG_NODE_COUNT = 10_001
UNION_NODE_COUNT = 57_501
H64_STEPS_PER_JULIAN_YEAR = 64
B22_ELEMENTS_BYTES = 288_005_760
B22_NODE_SHA256_BYTES = 19_200_384
B22_STATES_BYTES = 403_208_064
ML_SHORT_STATES_BYTES = 80_648_064
FIXED_CLASSIFIER_BINARY_BYTES = 791_062_272
RESONANCE_WINDOW_JULIAN_YEARS = 200_000
RESONANCE_WINDOW_COUNT = 50
POINTS_PER_RESONANCE_WINDOW = 1_000
ANGLE_BIN_COUNT = 18
MINIMUM_EMPTY_ANGLE_BINS = 2
MINIMUM_CONTIGUOUS_LIBRATING_WINDOWS = 46

BOUND_FILES = (
    ("h64_implementation", "benchmarks/jx_tno_stage1_jx_h64_closure_v2.py", 66_627, "04337f7fa63abbcbafd2dcacf6ad26ea5cd73cbccb8562e2a233b0e2c4aaefbd"),
    ("h64_test", "tests/test_tno_stage1_jx_h64_closure_v2.py", 23_647, "8dc67ff8ad87c3a34a333fd3f862852f7cf7dfe6ba20f09601952101c458ca5d"),
    ("h64_protocol", "runs/tno_stage1_jx_h64_closure_v2/PROTOCOL.json", 6_965, "f3ce7587145ddbda17442e4a47a97c4973d3d656aee7337aec49771731eaf1a0"),
    ("h64_result", "runs/tno_stage1_jx_h64_closure_v2/RESULT.json", 5_464_509, "3dd021735f817d4dbfb1a5c92370058347eff3606b47453137b5a3b44d41d54d"),
    ("h64_arrays", "runs/tno_stage1_jx_h64_closure_v2/ARRAYS.bin", 16_144_128, "9da897bb9a386db8429cd607147f241a3a5a1113f716a2f05d2c2d07c876ad41"),
    ("h64_source_nodes", "runs/tno_stage1_jx_h64_closure_v2/JX_SOURCE_NODES.bin", 368_645_800, "fd9b51e18aa6bb5fac095799955ecc6b3d6afeb2253b1000e2def41ccf376dbd"),
    ("h64_restart_streams", "runs/tno_stage1_jx_h64_closure_v2/JX_RESTART_STREAMS.bin", 865_728, "e18d17a8b4b51cf840110bb01730476f0535295acc875a2e994761fccb50c96a"),
    ("h64_midpoints", "runs/tno_stage1_jx_h64_closure_v2/JX_MIDPOINTS.bin", 10_368, "92d80bab7b4c39c70f0eada943ec625c8665dc0e1414f48693a3fdb0587e3f04"),
    ("stage1_v4_implementation", "benchmarks/jx_tno_stage1_dynamics_qualification_v4.py", 436_125, "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"),
    ("stage1_v4_result", "runs/tno_stage1_dynamics_qualification_v4/RESULT.json", 10_602_296, "bebd9186da0ef8943e05a8c4afc057073d48a8a47e9a4d006dae6e59921d5c42"),
    ("de441_acquisition", "tools/jx_tno_de441_resolved11_acquisition_v1.py", 37_145, "84df09da90c16d49cc245d72512290962639390ab9e0966a5cedd07f91b68e00"),
    ("de441_test", "tests/test_tno_de441_resolved11_acquisition_v1.py", 14_556, "220215bae324d8119d03417f44997920b7a4304fb59eae19a375d94a8555d8b3"),
    ("de441_protocol", "runs/tno_de441_resolved11_native_epochs_v1/PROTOCOL.json", 11_986, "502b066e7edec7630d0b7d0f873d608a34bcf231ace5206f16c8a7392e24e62b"),
    ("de441_custody", "runs/tno_de441_resolved11_native_epochs_v1/CUSTODY.json", 141_501, "9bd22da8b51dbc8735f24ebe8e047af4f5e1ec32129271f4ffad7f7fd1fccde3"),
    ("bridge_implementation", "benchmarks/jx_tno_native_epoch_resolved11_bridge_v1.py", 81_161, "5cb0b5380e7eac3e1c3c5932715dcbe43372b03581e65ef981ba491e78087e7e"),
    ("bridge_protocol", "runs/tno_native_epoch_resolved11_bridge_v1/PROTOCOL.json", 11_386, "3c6c008698528581463e7c55f381e6a9c77443f80c330a60d0897058b1c48459"),
    ("bridge_receipts", "runs/tno_native_epoch_resolved11_bridge_v1/RECEIPTS.json", 5_810_062, "3d7969ec02ffee40384029e9fb952082b5f7f112c832ee09230ba7118b310bd5"),
    ("bridge_states", "runs/tno_native_epoch_resolved11_bridge_v1/BRIDGE.json", 168_684, "dc0165aeb0817cfe356771100c5c851c2ca1d8ce3ea18a2495778838650fad6f"),
    ("stage0_implementation", "benchmarks/jx_tno_native_epoch_cartesian_qualification_v1.py", 70_064, "8cf9ae282e46ddf740c98f3b340d00501079c65c3f53559272a2e9ae4d58e367"),
    ("stage0_protocol", "runs/tno_native_epoch_cartesian_qualification_v1/PROTOCOL.json", 16_068, "4c7ec7df664b9dc54afc18f5a9db7d28d9488b28aaf3b238fdb90eb651500184"),
    ("stage0_states", "runs/tno_native_epoch_cartesian_qualification_v1/STATES.json", 1_026_550, "a01d53b2c7ac534e523ba0de436c56b39a8daa95742ec3bd6c7d3d33f8b1ede3"),
    ("pilot_implementation", "benchmarks/jx_tno_covariance_classification_pilot_v1.py", 31_585, "7d90ff40e71400071eb7d0ecf315113342e7c76cf8ae91935310f8895fcde2e1"),
    ("pilot_protocol", "runs/tno_covariance_classification_pilot_v1/PROTOCOL.json", 13_599, "583260b2d9886f0745f1cffd37bd800d18238dfb791d976d5a539a72d5d87f4f"),
    ("qualified_source_engine", "src/jxplanetx/restricted_wh_source.py", 61_803, "69d30e621dfb28c60c5a07d63752d4b92d993ba5d74524d7c3b404cd19fe7b52"),
    ("qualified_tracer_engine", "src/jxplanetx/cuda_restricted_wh.py", 74_758, "3c22c3d717fe5b07eb4ed929e5d32be591838fca8c1adbf02dec1263d3150927"),
    ("encounter_engine_candidate", "src/jxplanetx/cuda_massless_tracer.py", 223_940, "344aa7749e7888b5d63bd6348311efcc0ba149607eb0e4f32423cf0e1202fc5d"),
    ("encounter_engine_test", "tests/test_cuda_massless_tracer.py", 55_996, "d555e122cf0c8bdf619b0ee180a9607bfdfe223f6958b4845201cf97d8e29fb7"),
    ("predecessor_pilot_implementation", "benchmarks/jx_tno_10my_reduced5_classification_pilot_v1.py", 39_082, "6a5005946c84ce5321c029178e4bb5a855388d6b7870d572c63f505d70e7c1c8"),
    ("predecessor_pilot_test", "tests/test_tno_10my_reduced5_classification_pilot_v1.py", 19_244, "4f0ae8485c5b8fed9fcb88a2d0cbd85731033ba3c69c83d9e96b67922d510027"),
    ("predecessor_pilot_protocol", "runs/tno_10my_reduced5_classification_pilot_v1/PROTOCOL.json", 17_595, "9a00d2d30cf59267ff9d2474d6e3d42dde8bb4e5713d1a0c3c073d56ff8f9c2c"),
    ("b22_classifier_v1_implementation", "benchmarks/jx_tno_b22_classifier_v1.py", 82_137, "e3855734072b604941c91a0d598388c4665de2a1903677876a0fce84e62de600"),
    ("b22_classifier_v1_test", "tests/test_tno_b22_classifier_v1.py", 35_099, "f5457810ad6ca6c3d4b83a369759bf7fd22726cf2edb1adda4dfc1df04e2b084"),
    ("b22_qualification_implementation", "benchmarks/jx_tno_b22_classifier_qualification_v1.py", 37_044, "e93688c1d94903ec0f474d4c2aa1d5e23e6480ed7d6b138a9d3bfd80b3e6079b"),
    ("b22_qualification_test", "tests/test_tno_b22_classifier_qualification_v1.py", 16_034, "1e4c606fc5960d2b0df4ffae1810a7e8a7167436b8f6215c2dde8db097ff1834"),
    ("b22_qualification_protocol", "runs/tno_b22_classifier_qualification_v1/PROTOCOL.json", 10_923, "eb0da4c8f19a26f99adac1b863b9f65f69454f7814d0d13865e08f705729b5a0"),
    ("b22_classifier_v2_implementation", "benchmarks/jx_tno_b22_classifier_v2.py", 94_534, "f60333e290c72fd769ac9655b5a6bc154f5464915c0988e1972f01b642f30c98"),
    ("b22_classifier_v2_test", "tests/test_tno_b22_classifier_v2.py", 39_313, "f97f26b30f0d9895fef1a7f3a6094109fc7f56ac798be339df83c01b79cb4c90"),
)

EXPECTED_CONTENT = {
    "h64_protocol": "16712543d49432d110bf96e6c8e5be0436d922fdeaef74068606447c88b1a14e",
    "h64_result": "ea0627dd3c48c444cb4a0d9d5c11dedd41b3c3be886daa06208c9d4930896836",
    "stage1_v4_result": "df4e089aad8d436e99871c943343ea8b631ca714c946edb388387307c75d87d2",
    "de441_protocol": "f6fc67209bf69ba04f750903a9d368c95b2ef5d3ecdb9a789a651a3866698e8e",
    "de441_custody": "1d3e0fc4e6cda5e5c764b574027292b807477d893b9363a0ec33eee9af06b4b3",
    "bridge_protocol": "a7dbc361ffa4ccaf138f3c7afce269cad8bbb505b77930bcea9d04f5167b3c15",
    "bridge_states": "5be6b00855d996c2717a4745631e32fe89f99d61615080bdf22fc6d44ea1da13",
    "stage0_protocol": "2e7d4afd0da108584d8785bbfd1539828bb67862a8c9ec11f35a3bc4c2a53c1e",
    "stage0_states": "f32d722484c68db9289312bf23c8186a25423c0642efa75922b0f8989222081b",
    "pilot_protocol": "af27a4c5014b83d6b7c0241edba807e10ae635aaf641e6c5ad1eaf1fc1176db0",
    "predecessor_pilot_protocol": "4b6c4d4627ca39ec85b551f9d920d92b16f3da8a47dad3727e3467c3a649c699",
    "b22_qualification_protocol": "ceb790d65d87f1099390f210c547100c6b284f5b39896c1e23fd3bcd542c9079",
}

BLOCKERS = (
    "JX_B22_V2_SOURCE_BACKED_NUMERICAL_TRAJECTORY_FIXTURES_NOT_RETAINED_AND_PASSED",
    "JX_B22_V2_RUNTIME_AND_MEMORY_PROFILE_NOT_QUALIFIED",
    "SOURCE_BACKED_PUBLISHED_LABELLED_TRAJECTORY_FIXTURES_NOT_RETAINED_AND_PASSED",
    "INDEPENDENT_RESONANCE_CLASSIFIER_CROSSCHECK_NOT_RETAINED_AND_PASSED",
    "VOLK_MALHOTRA_ML_RUNTIME_MODEL_AND_FEATURE_PORT_NOT_FROZEN_OR_QUALIFIED",
    "NATIVE_50_YEAR_SHORT_AND_FULL_200_YEAR_B22_CARTESIAN_GRID_WRITER_NOT_IMPLEMENTED_OR_QUALIFIED",
    "REDUCED5_H64_TO_ENCOUNTER_CAPABLE_CHUNKED_ADAPTER_NOT_IMPLEMENTED_OR_QUALIFIED",
    "FIRST_1000_YEAR_PREFIX_NOT_BITWISE_OR_NUMERICALLY_CROSS_BOUND_TO_RETAINED_H64_EVIDENCE",
    "TEN_MYR_RESTART_ROUTE_INVARIANT_AND_STORAGE_CONTRACT_NOT_IMPLEMENTED_OR_QUALIFIED",
    "FUTURE_EXECUTION_NOT_EXTERNALLY_PREREGISTERED",
)

CLAIMS = {
    "prepare_only_protocol_frozen": True,
    "retained_h64_numerical_entry_gate_passed": True,
    "resolved11_timestep_refinement_required_before_this_pilot": False,
    "jx_b22_operationalization_parameters_frozen": True,
    "jx_b22_v2_search_domain_includes_36_to_1": True,
    "multi_grid_retention_contract_frozen": True,
    "exact_bernardinelli_2022_classifier_reproduction": False,
    "classifier_implementation_complete": True,
    "classifier_scientifically_qualified": False,
    "classifier_runtime_profile_qualified": False,
    "independent_broad_classifier_crosscheck_passed": False,
    "b22_grid_retained": False,
    "ml_short_grid_retained": False,
    "ml_long_grid_retained": False,
    "all_grid_crossbindings_passed": False,
    "reduced5_encounter_adapter_qualified": False,
    "ten_myr_execution_authorized": False,
    "ten_myr_execution_completed": False,
    "dynamical_classification_generated": False,
    "resonance_classification_generated": False,
    "robust_128_clone_classification_generated": False,
    "population_inference": False,
    "planet_nine_evidence": False,
    "science_result": False,
}


class PilotProtocolError(ValueError):
    """A bound input or prepare-only protocol failed closed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256_bytes(DOMAIN + _canonical_json(record))


def _strict_pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise PilotProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise PilotProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotProtocolError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise PilotProtocolError("JSON root must be an object")
    return value


def _stable_stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_regular_bytes(path: Path) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PilotProtocolError(f"required regular file cannot be opened: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PilotProtocolError(f"file is linked or nonregular: {path}")
        try:
            path_before = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise PilotProtocolError(f"file path cannot be resolved: {path}") from exc
        if (
            not stat.S_ISREG(path_before.st_mode)
            or (before.st_dev, before.st_ino)
            != (path_before.st_dev, path_before.st_ino)
        ):
            raise PilotProtocolError(f"file path differs from opened inode: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as exc:
            raise PilotProtocolError(f"file path changed while being read: {path}") from exc
        if (
            _stable_stat_identity(before) != _stable_stat_identity(after)
            or not stat.S_ISREG(path_after.st_mode)
            or (after.st_dev, after.st_ino)
            != (path_after.st_dev, path_after.st_ino)
            or _stable_stat_identity(path_before)
            != _stable_stat_identity(path_after)
        ):
            raise PilotProtocolError(f"file changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


_BOOTSTRAP_IMPLEMENTATION_PAYLOAD = _read_regular_bytes(SCRIPT_PATH)
_BOOTSTRAP_IMPLEMENTATION_BYTE_LENGTH = len(_BOOTSTRAP_IMPLEMENTATION_PAYLOAD)
_BOOTSTRAP_IMPLEMENTATION_SHA256 = _sha256_bytes(
    _BOOTSTRAP_IMPLEMENTATION_PAYLOAD
)
del _BOOTSTRAP_IMPLEMENTATION_PAYLOAD


def _implementation_identity() -> dict[str, object]:
    payload = _read_regular_bytes(SCRIPT_PATH)
    observed_sha256 = _sha256_bytes(payload)
    if (
        len(payload) != _BOOTSTRAP_IMPLEMENTATION_BYTE_LENGTH
        or observed_sha256 != _BOOTSTRAP_IMPLEMENTATION_SHA256
    ):
        raise PilotProtocolError(
            "implementation source differs from its module-bootstrap observation"
        )
    return {
        "relative_path": str(SCRIPT_PATH.relative_to(PROJECT_ROOT)),
        "byte_length": _BOOTSTRAP_IMPLEMENTATION_BYTE_LENGTH,
        "sha256": _BOOTSTRAP_IMPLEMENTATION_SHA256,
        "custody_scope": (
            "MODULE_BOOTSTRAP_SOURCE_OBSERVATION_WITH_LIVE_REREAD_NOT_EXECUTED_BYTE_PROOF"
        ),
    }


def _directory_open_flags() -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _read_regular_bytes_at(directory_fd: int, name: str) -> bytes:
    if type(name) is not str or not name or Path(name).name != name:
        raise PilotProtocolError("directory-relative file name is invalid")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise PilotProtocolError(f"required prepared file cannot be opened: {name}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PilotProtocolError(f"prepared file is linked or nonregular: {name}")
        try:
            path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise PilotProtocolError(
                f"prepared file path cannot be resolved: {name}"
            ) from exc
        if not stat.S_ISREG(path_before.st_mode) or not _same_node(
            before, path_before
        ):
            raise PilotProtocolError(
                f"prepared file path differs from opened inode: {name}"
            )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
        try:
            path_after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except OSError as exc:
            raise PilotProtocolError(
                f"prepared file path changed while being read: {name}"
            ) from exc
        if (
            _stable_stat_identity(before) != _stable_stat_identity(after)
            or not stat.S_ISREG(path_after.st_mode)
            or not _same_node(after, path_after)
            or _stable_stat_identity(path_before)
            != _stable_stat_identity(path_after)
        ):
            raise PilotProtocolError(f"prepared file changed while being read: {name}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _bound_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for role, relative_path, expected_size, expected_sha in BOUND_FILES:
        payload = _read_regular_bytes(PROJECT_ROOT / relative_path)
        observed_sha = _sha256_bytes(payload)
        if len(payload) != expected_size or observed_sha != expected_sha:
            raise PilotProtocolError(f"bound file differs: {relative_path}")
        records.append(
            {
                "role": role,
                "relative_path": relative_path,
                "byte_length": expected_size,
                "sha256": expected_sha,
            }
        )
    return records


def _load_bound_source_module(role: str, module_name: str) -> types.ModuleType:
    for bound_role, relative_path, expected_size, expected_sha in BOUND_FILES:
        if bound_role != role:
            continue
        path = PROJECT_ROOT / relative_path
        payload = _read_regular_bytes(path)
        if len(payload) != expected_size or _sha256_bytes(payload) != expected_sha:
            raise PilotProtocolError(f"bound source differs: {relative_path}")
        module = types.ModuleType(module_name)
        module.__file__ = str(path)
        module.__package__ = ""
        module.__spec__ = None
        if module_name in sys.modules:
            raise PilotProtocolError("bound source module name is already occupied")
        sys.modules[module_name] = module
        try:
            exec(
                compile(
                    payload,
                    str(path),
                    "exec",
                    dont_inherit=True,
                    optimize=sys.flags.optimize,
                ),
                module.__dict__,
            )
        except Exception as exc:
            raise PilotProtocolError(
                f"bound source could not be loaded: {relative_path}"
            ) from exc
        finally:
            sys.modules.pop(module_name, None)
        return module
    raise PilotProtocolError(f"unknown bound source role: {role}")


def _validate_classifier_successor() -> dict[str, object]:
    classifier = _load_bound_source_module(
        "b22_classifier_v2_implementation",
        f"_jx_tno_bound_b22_classifier_v2_{sys.flags.optimize}",
    )
    try:
        report = classifier.binding_report_v2()
    except Exception as exc:
        raise PilotProtocolError("B22 classifier v2 binding report failed") from exc
    search = report.get("search_domain")
    claims = report.get("claims")
    if (
        report.get("classifier")
        != "JX_B22_OPERATIONALIZATION_V2_36_TO_1_NOT_EXACT_PAPER_REPRODUCTION"
        or type(search) is not dict
        or search.get("combined_period_pair_count") != 653
        or search.get("inherited_period_pair_count") != 539
        or search.get("new_band_period_pair_count") != 114
        or search.get("combined_argument_count") != 1_712_871
        or search.get("principal_36_to_1_argument") != [36, 1, -35, 0, 0, 0]
        or type(claims) is not dict
        or claims.get("search_domain_includes_36_to_1") is not True
        or claims.get("classifier_scientifically_qualified") is not False
        or claims.get("science_result") is not False
    ):
        raise PilotProtocolError("B22 classifier v2 binding report differs")
    qualification = _load_bound_source_module(
        "b22_qualification_implementation",
        f"_jx_tno_bound_b22_qualification_v1_{sys.flags.optimize}",
    )
    try:
        qualified = qualification.validate_prepared(
            PROJECT_ROOT / "runs" / "tno_b22_classifier_qualification_v1"
        )
    except Exception as exc:
        raise PilotProtocolError("B22 qualification custody failed") from exc
    if (
        qualified.get("content_sha256")
        != EXPECTED_CONTENT["b22_qualification_protocol"]
        or qualified.get("claims", {}).get("classifier_scientifically_qualified")
        is not False
    ):
        raise PilotProtocolError("B22 qualification claim ceiling differs")
    return {
        "classifier": report,
        "qualification_protocol_content_sha256": qualified["content_sha256"],
    }


def _load_bound_json(role: str) -> dict[str, Any]:
    for bound_role, relative_path, _size, _sha in BOUND_FILES:
        if bound_role == role:
            payload = _read_regular_bytes(PROJECT_ROOT / relative_path)
            value = _strict_json(payload)
            if payload != _canonical_json(value):
                raise PilotProtocolError(f"bound JSON is not canonical: {relative_path}")
            expected_content = EXPECTED_CONTENT.get(role)
            if expected_content is not None and value.get("content_sha256") != expected_content:
                raise PilotProtocolError(f"bound JSON content identity differs: {relative_path}")
            return value
    raise PilotProtocolError(f"unknown bound JSON role: {role}")


def _validate_scientific_inputs() -> None:
    h64_protocol = _load_bound_json("h64_protocol")
    h64_result = _load_bound_json("h64_result")
    v4_result = _load_bound_json("stage1_v4_result")
    de441_protocol = _load_bound_json("de441_protocol")
    de441_custody = _load_bound_json("de441_custody")
    bridge = _load_bound_json("bridge_states")
    stage0 = _load_bound_json("stage0_states")
    pilot = _load_bound_json("pilot_protocol")
    predecessor = _load_bound_json("predecessor_pilot_protocol")
    qualification = _load_bound_json("b22_qualification_protocol")

    if h64_protocol.get("schema") != "jx.tno.stage1-jx-h64-closure.protocol.v2":
        raise PilotProtocolError("h64 protocol schema differs")
    if h64_result.get("status") != "H64_REFINEMENT_CLOSURE_PASS":
        raise PilotProtocolError("h64 closure did not pass")
    gates = h64_result.get("qualification_gates")
    if type(gates) is not dict or gates.get("all_predeclared_gates_passed") is not True:
        raise PilotProtocolError("h64 closure gates did not all pass")
    h64_claims = h64_result.get("claims")
    if (
        type(h64_claims) is not dict
        or h64_claims.get("h64_refinement_closure_established") is not True
        or h64_claims.get("v4_reclassified_as_pass") is not False
        or h64_claims.get("science_result") is not False
    ):
        raise PilotProtocolError("h64 claim ceiling differs")
    if v4_result.get("status") != "QUALIFICATION_EXECUTED_GATES_NOT_ALL_PASS":
        raise PilotProtocolError("predecessor V4 failure custody differs")
    if de441_protocol.get("schema") != "jx.tno.de441-resolved11-native-epochs.protocol.v1":
        raise PilotProtocolError("DE441 protocol schema differs")
    if de441_custody.get("schema") != "jx.tno.de441-resolved11-native-epochs.custody.v1":
        raise PilotProtocolError("DE441 custody schema differs")
    if bridge.get("status") != "TWELVE_NATIVE_EPOCH_PREPARED_BATCHES_NO_DYNAMICS":
        raise PilotProtocolError("native-epoch bridge status differs")
    if stage0.get("schema") != "jx.tno.native-epoch-cartesian.states.v1":
        raise PilotProtocolError("Stage-0 state schema differs")
    if pilot.get("schema") != "jx.tno.covariance-classification-pilot.protocol.v1":
        raise PilotProtocolError("original pilot schema differs")
    if (
        predecessor.get("schema")
        != "jx.tno.10my-reduced5-classification-pilot.protocol.v1"
        or predecessor.get("outcomes_generated") is not False
        or predecessor.get("execution_authorized") is not False
    ):
        raise PilotProtocolError("predecessor 10-Myr pilot custody differs")
    finding = qualification.get("decisive_search_domain_finding")
    if (
        qualification.get("schema")
        != "jx.tno.b22-classifier-qualification.protocol.v1"
        or type(finding) is not dict
        or finding.get("published_resonance") != {"p": 36, "q": 1}
        or finding.get("within_frozen_jx_v1_search_domain") is not False
        or qualification.get("outcomes_generated") is not False
    ):
        raise PilotProtocolError("B22 qualification custody differs")


def _float_record(value: float, exact_rational: str) -> dict[str, object]:
    if not math.isfinite(value):
        raise PilotProtocolError("non-finite protocol constant")
    return {"exact_rational": exact_rational, "binary64_hex": value.hex()}


def build_protocol() -> dict[str, Any]:
    """Reconstruct the complete prepare-only protocol from pinned inputs."""
    _bound_records()
    _validate_scientific_inputs()
    successor = _validate_classifier_successor()
    implementation_identity = _implementation_identity()
    test_payload = _read_regular_bytes(TEST_PATH)

    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "design_status": "PREPARE_ONLY_EXECUTION_BLOCKED",
        "objective": (
            "Freeze one DE441-only, reduced-five, h64, 10-Myr classification-readiness "
            "pilot over the already-bridged 108 states, with a 36:1-capable B22 search and "
            "native short/long independent-classifier grids; do not execute it until every "
            "blocker closes."
        ),
        "implementation": {
            **implementation_identity,
            "test_relative_path": str(TEST_PATH.relative_to(PROJECT_ROOT)),
            "test_byte_length": len(test_payload),
            "test_sha256": _sha256_bytes(test_payload),
            "cli_surface": "BUILD_PREPARE_VALIDATE_ONLY",
            "dynamics_surface_present": False,
            "classifier_execution_surface_present": False,
        },
        "bound_inputs": {
            "records": _bound_records(),
            "content_identities": dict(EXPECTED_CONTENT),
            "classifier_successor_validation": successor,
            "large_h64_evidence_custody": (
                "LOCAL_CONTENT_ADDRESSED_EXTERNAL_TO_COMPACT_GITHUB_SOURCE_COMMIT"
            ),
        },
        "cohort": {
            "object_ids_in_exact_order": list(OBJECT_IDS),
            "object_count": 12,
            "states_per_object": 9,
            "state_order": "NOMINAL_THEN_FIXED_HOLDOUT_CLONE_ORDER",
            "holdout_clone_indices": list(HOLDOUT_CLONE_INDICES),
            "total_state_count": 108,
            "full_clone_denominator": 128,
            "full_1548_state_execution": False,
            "holdout_result_claim_ceiling": (
                "PER_TRAJECTORY_LABELS_AND_NINE_STATE_AGREEMENT_DIAGNOSTIC_ONLY"
            ),
        },
        "dynamics": {
            "ephemeris": "DE441_NATIVE_EPOCH_STATE_SOURCE",
            "massive_model": "SUN_PLUS_FOUR_GIANT_PLANETS",
            "source_ids_in_exact_order": list(REDUCED_SOURCE_IDS),
            "input_role_separation": {
                "tno_initial_states": "STAGE0_1548_STATE_NATIVE_EPOCH_CARTESIAN_ARTIFACT",
                "massive_source_initial_states": "DE441_RESOLVED11_CUSTODY_COLLAPSED_TO_REDUCED5",
                "de440s_bridge_role": (
                    "ADAPTER_AND_COORDINATE_CONTRACT_CUSTODY_ONLY_NOT_THE_PRODUCTION_MASSIVE_SOURCE"
                ),
                "de440s_de441_equivalence_claimed": False,
            },
            "tno_semantics": "MASSLESS_NONINTERACTING_TRACERS",
            "time_scale": "TDB",
            "time_unit": "JULIAN_YEAR_EXACTLY_31557600_SI_SECONDS",
            "horizon_julian_years": HORIZON_JULIAN_YEARS,
            "step_julian_years": _float_record(float(STEP_JULIAN_YEARS), "1/64"),
            "production_lane_count": 12,
            "refinement_lane_count": 0,
            "another_timestep_ladder_authorized": False,
            "first_1000_years_must_cross_bind_to_retained_h64": True,
            "encounter_route_evidence_required": True,
            "restart_and_sharding_evidence_required": True,
        },
        "classifier_input": {
            "input_frame": "ICRF_J2000_EQUATORIAL_MODEL_BARYCENTRIC_INERTIAL",
            "element_frame": "ECLIPJ2000_MODEL_BARYCENTRIC_INERTIAL",
            "elements": ["a", "e", "i", "Omega", "omega", "M"],
            "angle_unit": "RADIAN_WRAPPED_TO_HALF_OPEN_0_TWO_PI",
            "primary_and_gm": "SAME_FROZEN_REDUCED5_MODEL_USED_BY_DYNAMICS",
            "cadence_julian_years": CLASSIFIER_CADENCE_JULIAN_YEARS,
            "cadence_binary64_hex": float(CLASSIFIER_CADENCE_JULIAN_YEARS).hex(),
            "node_count_including_final_endpoint": CLASSIFIER_NODE_COUNT,
            "resonance_nodes": "INDICES_0_THROUGH_49999_HALF_OPEN",
            "final_endpoint": "INDEX_50000_RETAINED_AND_USED_BY_NONRESONANT_METRICS",
            "missing_nonfinite_nonelliptic_or_removed_trajectory_policy": "FAIL_CLOSED_NO_LABEL",
        },
        "retention_grids": {
            "time_origin": "EACH_BATCH_NATIVE_JD_TDB_EPOCH",
            "sampling_requirement": "EXACT_OUTER_H64_STEP_BOUNDARIES",
            "interpolation_permitted": False,
            "union_grid": {
                "node_count": UNION_NODE_COUNT,
                "piecewise_rule": [
                    {
                        "start_julian_year": 0,
                        "stop_julian_year_inclusive": ML_SHORT_HORIZON_JULIAN_YEARS,
                        "cadence_julian_years": ML_SHORT_CADENCE_JULIAN_YEARS,
                        "node_count": ML_SHORT_NODE_COUNT,
                    },
                    {
                        "start_julian_year_exclusive": ML_SHORT_HORIZON_JULIAN_YEARS,
                        "stop_julian_year_inclusive": HORIZON_JULIAN_YEARS,
                        "cadence_julian_years": CLASSIFIER_CADENCE_JULIAN_YEARS,
                        "additional_node_count": 47_500,
                    },
                ],
            },
            "ml_short": {
                "times": "t_j=50*j_FOR_j_0_THROUGH_10000",
                "node_count": ML_SHORT_NODE_COUNT,
                "h64_step_index": "3200*j",
                "horizon_julian_years": ML_SHORT_HORIZON_JULIAN_YEARS,
            },
            "b22_primary": {
                "times": "t_k=200*k_FOR_k_0_THROUGH_50000",
                "node_count": CLASSIFIER_NODE_COUNT,
                "h64_step_index": "12800*k",
                "horizon_julian_years": HORIZON_JULIAN_YEARS,
            },
            "ml_long": {
                "times": "t_l=1000*l_FOR_l_0_THROUGH_10000",
                "node_count": ML_LONG_NODE_COUNT,
                "h64_step_index": "64000*l",
                "horizon_julian_years": HORIZON_JULIAN_YEARS,
                "cartesian_view": (
                    "EXACT_B22_STATES_STRIDE_5_NO_DUPLICATE_CARTESIAN_FILE"
                ),
                "element_view": "EXACT_B22_ELEMENTS_STRIDE_5_NO_DUPLICATE_ELEMENT_FILE",
            },
            "exact_cross_grid_mappings": [
                "NODE_SHA256(B22_STATES[k])_EQUALS_B22_NODE_SHA256[k]_FOR_EVERY_k_0_THROUGH_50000",
                "EXTRACT_ELEMENTS(B22_STATES[k])_BITWISE_EQUALS_B22_ELEMENTS[k]_FOR_EVERY_k_0_THROUGH_50000",
                "ML_LONG_VIEW[l]_IS_B22_STATES[5*l]_FOR_l_0_THROUGH_10000",
                "ML_LONG_ELEMENT_VIEW[l]_IS_B22_ELEMENTS[5*l]_FOR_l_0_THROUGH_10000",
                "ML_SHORT[4*k]_BITWISE_EQUALS_B22_STATES[k]_FOR_k_0_THROUGH_2500",
                "NODE_SHA256(ML_SHORT[4*k])_EQUALS_B22_NODE_SHA256[k]_FOR_k_0_THROUGH_2500",
                "ML_SHORT[20*l]_BITWISE_EQUALS_B22_STATES[5*l]_FOR_l_0_THROUGH_500",
                "EXTRACT_ELEMENTS(ML_SHORT[4*k])_BITWISE_EQUALS_B22_ELEMENTS[k]_FOR_k_0_THROUGH_2500",
            ],
            "cross_grid_comparison_types": {
                "cartesian_to_cartesian": "RAW_BINARY64_BITWISE_EQUALITY",
                "cartesian_to_node_digest": "RECOMPUTE_EXACT_NODE_SHA256_THEN_DIGEST_EQUALITY",
                "cartesian_to_elements": "RUN_BOUND_V2_ELEMENT_EXTRACTOR_THEN_BINARY64_BITWISE_EQUALITY",
                "direct_cartesian_bytes_equal_orbital_element_bytes": False,
                "offline_replay_coverage": "EVERY_B22_NODE_IN_EVERY_BATCH",
            },
        },
        "prospective_binary_artifacts": {
            "publication_order": [
                "B22_STATES.f64le",
                "B22_ELEMENTS.f64le",
                "B22_NODE_SHA256.raw",
                "ML_SHORT_STATES.f64le",
                "MANIFEST.json",
                "RESULT.json_LAST",
            ],
            "fixed_binary_byte_count": FIXED_CLASSIFIER_BINARY_BYTES,
            "files": [
                {
                    "name": "B22_STATES.f64le",
                    "byte_count": B22_STATES_BYTES,
                    "dtype": "IEEE754_BINARY64_LITTLE_ENDIAN",
                    "shape": [12, CLASSIFIER_NODE_COUNT, 14, 6],
                    "order": "BATCH_NODE_5_MASSIVE_THEN_9_TRACERS_X_Y_Z_VX_VY_VZ",
                    "frame": "ICRF_J2000_EQUATORIAL_OPERATIONAL_REDUCED5_MODEL_BARYCENTER",
                    "units": "AU_AND_AU_PER_EXACT_JULIAN_YEAR",
                    "offline_replay_role": (
                        "COMPLETE_PREIMAGE_FOR_EVERY_B22_NODE_DIGEST_AND_ELEMENT_ROW"
                    ),
                    "padding_or_trailing_bytes": False,
                },
                {
                    "name": "B22_ELEMENTS.f64le",
                    "byte_count": B22_ELEMENTS_BYTES,
                    "dtype": "IEEE754_BINARY64_LITTLE_ENDIAN",
                    "shape": [12, 10, 6, CLASSIFIER_NODE_COUNT],
                    "order": "BATCH_THEN_NEPTUNE_THEN_9_TRACERS_THEN_FIELD_THEN_NODE",
                    "fields": ["a", "e", "i", "Omega", "omega", "M"],
                    "padding_or_trailing_bytes": False,
                },
                {
                    "name": "B22_NODE_SHA256.raw",
                    "byte_count": B22_NODE_SHA256_BYTES,
                    "dtype": "RAW_SHA256_32_BYTE_DIGEST",
                    "shape": [12, CLASSIFIER_NODE_COUNT, 32],
                    "binding": "COMPLETE_5_MASSIVE_PLUS_9_TRACER_REDUCED5_CHECKPOINT",
                    "digest_domain_ascii_without_terminal_nul": (
                        "jx.tno.b22-classifier.node-state.v2"
                    ),
                    "digest_domain_hex_including_terminal_nul": (
                        "6a782e746e6f2e6232322d636c61737369666965722e6e6f64652d73746174652e763200"
                    ),
                    "digest_preimage": [
                        "DOMAIN_BYTES",
                        "CANONICAL_JSON_METADATA_WITH_ONE_TRAILING_LF",
                        "FOR_EACH_OF_5_MASSIVE_THEN_9_TRACERS_POSITION_XYZ_THEN_VELOCITY_XYZ_AS_IEEE754_BINARY64_LITTLE_ENDIAN",
                    ],
                    "metadata_exact_keys": [
                        "batch_order",
                        "binary64_field_order",
                        "ephemeris",
                        "massive_body_ids",
                        "massive_gm_hex",
                        "massive_naif_ids",
                        "native_epoch_jd_tdb_hex",
                        "object_id",
                        "spk_id",
                        "tracer_identity",
                    ],
                    "metadata_binary64_field_order": (
                        "MASSIVE_POSITION_VELOCITY_THEN_TRACER_POSITION_VELOCITY"
                    ),
                    "authoritative_implementation": (
                        "b22_classifier_v2_implementation.node_state_sha256_v2"
                    ),
                    "padding_or_trailing_bytes": False,
                },
                {
                    "name": "ML_SHORT_STATES.f64le",
                    "byte_count": ML_SHORT_STATES_BYTES,
                    "dtype": "IEEE754_BINARY64_LITTLE_ENDIAN",
                    "shape": [12, ML_SHORT_NODE_COUNT, 14, 6],
                    "order": "BATCH_NODE_5_MASSIVE_THEN_9_TRACERS_X_Y_Z_VX_VY_VZ",
                    "frame": "ICRF_J2000_EQUATORIAL_OPERATIONAL_REDUCED5_MODEL_BARYCENTER",
                    "units": "AU_AND_AU_PER_EXACT_JULIAN_YEAR",
                    "padding_or_trailing_bytes": False,
                },
            ],
            "manifest_must_bind": [
                "EXACT_FILE_SIZE_SHA256_AND_DOMAIN_SEPARATED_PAYLOAD_HASH",
                "PER_BATCH_STREAM_ROOT_AND_PER_SERIES_CLASSIFIER_DIGEST",
                "LAYOUTS_GRIDS_OBJECT_NATIVE_EPOCH_SOURCE_GM_AND_TRACER_ROSTERS",
                "RUNTIME_IDENTITY_RESTART_ROUTE_AND_FIRST_1000_YEAR_CROSSBINDINGS",
            ],
        },
        "element_extraction_contract": {
            "bound_source_role": "stage1_v4_implementation",
            "bound_source_sha256": (
                "6ffbd4f6f6025cb413790a8d0addf7a620ce4d3a993e99b01426a4c2749ec7f2"
            ),
            "required_source_only_callables": [
                "collapse_reduced_model",
                "_weighted_centroid",
                "_checkpoint",
                "extract_elliptic_orbital_elements",
                "_angle",
            ],
            "reimplementation_permitted": False,
            "state_origin": (
                "INSTANTANEOUS_GM_WEIGHTED_BARYCENTER_OF_EXACT_REDUCED5_SOURCE_ROSTER"
            ),
            "centering_sequence": (
                "CALL_BOUND_COLLAPSE_REDUCED_MODEL_THEN_BOUND_CHECKPOINT_CENTERING_BEFORE_ROTATION"
            ),
            "input_axis_orientation": "ICRF_J2000_EQUATORIAL_RIGHT_HANDED",
            "element_axis_orientation": "ECLIPJ2000_RIGHT_HANDED",
            "j2000_equatorial_to_eclipj2000_rotation": {
                "cos_obliquity_binary64_hex": "0x1.d5c0357681ef3p-1",
                "sin_obliquity_binary64_hex": "0x1.9752e50f4b399p-2",
                "input": "ALREADY_CENTERED_X_Y_Z",
                "x_out": "x",
                "y_out_binary64_evaluation": "(cos_obliquity*y)+(sin_obliquity*z)",
                "z_out_binary64_evaluation": "((-sin_obliquity)*y)+(cos_obliquity*z)",
                "same_rotation_applied_separately_to_position_and_velocity": True,
                "fused_multiply_add_permitted": False,
            },
            "element_mu": (
                "BOUND_EXTRACTOR_WITH_MATH_FSUM_OF_EXACT_ACTIVE_REDUCED5_GMS_"
                "INCLUDING_FOLDED_CENTRAL_GM_AND_EXCLUDING_PLUTO"
            ),
            "neptune_elements": (
                "BOUND_EXTRACTOR_ON_CENTERED_ROTATED_NEPTUNE_STATE_WITH_THE_SAME_ELEMENT_MU"
            ),
            "period_binary64_evaluation": (
                "(2.0*math.pi)*math.sqrt(((a*a)*a)/element_mu)"
            ),
            "longitude_binary64_evaluation": {
                "varpi": "BOUND__ANGLE(Omega+omega)",
                "lambda": "BOUND__ANGLE((Omega+omega)+M)",
                "critical_angle": (
                    "BOUND__ANGLE(math.fsum((p*lambda,-q*lambda_N,m*varpi,"
                    "-n*varpi_N,r*Omega)))"
                ),
            },
            "runtime_and_libm_identity": "MUST_BE_PINNED_BY_FUTURE_CLASSIFIER_IMPLEMENTATION",
        },
        "classifier_precedence": [
            "RESONANT_OR_RESONANT_CANDIDATE",
            "OORT_IF_A0_STRICTLY_GREATER_THAN_2000_AU",
            "CENTAUR_IF_A0_STRICTLY_LESS_THAN_NEPTUNE_A0",
            "SCATTERING_IF_MAX_ABS_A_MINUS_A0_OVER_A0_STRICTLY_GREATER_THAN_3_OVER_80",
            "DETACHED_IF_E0_STRICTLY_GREATER_THAN_6_OVER_25",
            "CLASSICAL_OTHERWISE",
        ],
        "nonresonant_thresholds": {
            "oort_a0_au": _float_record(2000.0, "2000"),
            "scattering_fraction": _float_record(0.0375, "3/80"),
            "detached_e0": _float_record(0.24, "6/25"),
            "scattering_metric": "MAX_OVER_ALL_50001_NODES_OF_ABS_A_T_MINUS_A0_DIVIDED_BY_A0",
            "all_threshold_comparisons": "STRICT_GREATER_THAN",
        },
        "resonance_operationalization": {
            "name": "JX_B22_OPERATIONALIZATION_V2_36_TO_1_NOT_EXACT_PAPER_REPRODUCTION",
            "critical_angle": "p*lambda-q*lambda_N+m*varpi-n*varpi_N+r*Omega-s*Omega_N",
            "lambda_definition": "Omega+omega+M",
            "varpi_definition": "Omega+omega",
            "s": 0,
            "dalembert_relation": "p-q+m-n+r=0",
            "period_pair_union": [
                {
                    "band": "INHERITED_V1",
                    "p_domain_inclusive": [1, 29],
                    "q_domain_inclusive": [1, 29],
                    "coefficient_domain_inclusive": [-29, 29],
                },
                {
                    "band": "NEW_P30_TO_P36",
                    "p_domain_inclusive": [30, 36],
                    "q_domain_inclusive": [1, 29],
                    "coefficient_domain_inclusive": [-35, 35],
                },
            ],
            "p_q_normalization": "POSITIVE_AND_GCD_ONE",
            "r_rule": "r=q-p-m+n_AND_REQUIRE_WITHIN_THE_SELECTED_BAND_COEFFICIENT_DOMAIN",
            "zero_coefficients_allowed": True,
            "principal_36_to_1_argument": [36, 1, -35, 0, 0, 0],
            "search_universe_counts": {
                "inherited_period_pairs": 539,
                "new_band_period_pairs": 114,
                "combined_period_pairs": 653,
                "inherited_arguments": 1_327_797,
                "new_band_arguments": 385_074,
                "combined_arguments": 1_712_871,
            },
            "compatibility_rule": (
                "V1_RESULTS_MUST_BE_BIT_FOR_BIT_EQUIVALENT_WHEN_NO_NEW_BAND_ARGUMENT_IS_RESONANT"
            ),
            "winner_rule_across_bands": "ONE_UNIFIED_UNCHANGED_TIE_BREAK_WITH_NO_INHERITED_BAND_PREFERENCE",
            "duplicate_rule": "ONE_CANONICAL_ROW_PER_EXACT_NORMALIZED_COEFFICIENT_TUPLE",
            "candidate_period_rule": "ABS((p/q)*P_N0-P_TNO0)_STRICTLY_LESS_THAN_(3/20)*P_N0",
            "candidate_period_fraction": _float_record(0.15, "3/20"),
            "candidate_sort": [
                "ABS_PERIOD_MISMATCH_ASCENDING",
                "p_PLUS_q_ASCENDING",
                "p_ASCENDING",
                "q_ASCENDING",
            ],
            "window_count": RESONANCE_WINDOW_COUNT,
            "window_julian_years": RESONANCE_WINDOW_JULIAN_YEARS,
            "points_per_window": POINTS_PER_RESONANCE_WINDOW,
            "window_layout": "NONOVERLAPPING_HALF_OPEN_STRIDE_200000_JULIAN_YEARS",
            "angle_bin_count": ANGLE_BIN_COUNT,
            "angle_bin_edges_degrees": [20 * value for value in range(ANGLE_BIN_COUNT + 1)],
            "angle_bin_membership": (
                "LEFT_CLOSED_RIGHT_OPEN_INDEX_MIN_17_INT(SIGMA_DIVIDED_BY_"
                "(MATH_TAU_DIVIDED_BY_18))_AFTER_BOUND__ANGLE"
            ),
            "librating_window_rule": "AT_LEAST_TWO_EMPTY_ANGLE_BINS",
            "minimum_empty_angle_bins": MINIMUM_EMPTY_ANGLE_BINS,
            "contiguity_rule": "ADJACENT_WINDOW_INDICES_FOR_THE_SAME_EXACT_ARGUMENT",
            "clone_resonant_rule": "LARGEST_CONTIGUOUS_RUN_STRICTLY_GREATER_THAN_90_PERCENT",
            "minimum_contiguous_windows": MINIMUM_CONTIGUOUS_LIBRATING_WINDOWS,
            "winning_argument_tie_break": [
                "LONGEST_CONTIGUOUS_RUN_DESCENDING",
                "TOTAL_LIBRATING_WINDOWS_DESCENDING",
                "ABS_PERIOD_MISMATCH_ASCENDING",
                "p_PLUS_q_ASCENDING",
                "p_ASCENDING",
                "q_ASCENDING",
                "ABS_m_PLUS_ABS_n_PLUS_ABS_r_ASCENDING",
                "m_ASCENDING",
                "n_ASCENDING",
                "r_ASCENDING",
            ],
            "paper_underspecification_disclosed": True,
            "paper_missing_items_filled_by_jx": [
                "EXACT_CADENCE_AND_ENDPOINT_LAYOUT",
                "COEFFICIENT_ZERO_SIGN_GCD_AND_DUPLICATE_RULES",
                "ANGLE_BIN_COUNT_EDGES_AND_WRAP",
                "MULTIPLE_EMPTY_BINS_DEFINED_AS_AT_LEAST_TWO",
                "WINDOW_STRIDE_AND_CONTIGUITY",
                "TIE_BREAKING_AND_FAIL_CLOSED_INPUT_POLICY",
                "SEPARATELY_VERSIONED_36_TO_1_SEARCH_DOMAIN_EXTENSION",
            ],
        },
        "aggregation": {
            "this_108_state_pilot": {
                "object_classification_authorized": False,
                "report": "NOMINAL_PLUS_EIGHT_FIXED_HOLDOUT_LABEL_AGREEMENT_ONLY",
            },
            "future_full_128_clone_jx_rule": {
                "clone_denominator": 128,
                "minimum_same_class_clones": 123,
                "minimum_same_class_fraction": "123/128",
                "nominal_must_match": True,
                "resonance_consensus_requires_same_exact_argument": True,
                "failure_label": "MIXED_UNDER_ORBIT_COVARIANCE",
                "interpretation": "FORMAL_COVARIANCE_SENSITIVITY_FREQUENCY_NOT_POSTERIOR_PROBABILITY",
            },
            "bernardinelli_2022_literal_30_clone_reference_only": {
                "scattering_at_least": 15,
                "resonant_candidate_strictly_over_half_at_least": 16,
                "secure_same_argument_strictly_over_80_percent_at_least": 25,
                "transferred_to_jx_128_clone_rule": False,
            },
        },
        "required_classifier_fixtures": {
            "source_backed": [
                "KHAIN_S12_GOOD_5_2_TO_7_RESONANT",
                "KHAIN_2012_WG37_SCATTERING",
                "KHAIN_S14_GOOD_4_DETACHED",
                "KHAIN_2013_RP98_CLASSICAL",
                "BERNARDINELLI_Y6_1_TO_1_8_TO_1_25_TO_3_AND_10_TO_1_CANDIDATE_LABELS_AND_ARGUMENTS",
                "KAT_VOLK_474640_PUBLISHED_36_TO_1_LABEL_WITH_NUMERICAL_TRAJECTORY_STILL_REQUIRED",
            ],
            "mechanical_boundaries": [
                "SCATTERING_EXACT_3_OVER_80_NEGATIVE_AND_NEXT_FLOAT_ABOVE_POSITIVE",
                "DETACHED_EXACT_6_OVER_25_NEGATIVE_AND_NEXT_FLOAT_ABOVE_POSITIVE",
                "PERIOD_FILTER_EXACT_3_OVER_20_EXCLUDED",
                "45_OF_50_CONTIGUOUS_NEGATIVE_AND_46_OF_50_POSITIVE",
                "ZERO_DEGREE_WRAP_LIBRATION_POSITIVE_AND_UNIFORM_CIRCULATION_NEGATIVE",
                "SPLIT_RESONANCE_ARGUMENTS_NOT_SECURE",
                "MISSING_NONFINITE_REMOVED_REORDERED_OR_CADENCE_TAMPER_FAILS_CLOSED",
            ],
            "qualification_status": "CATALOG_LABEL_CUSTODY_PASSED_NUMERICAL_TRAJECTORY_FIXTURES_NOT_YET_RETAINED_OR_PASSED",
        },
        "independent_broad_classifier_contract": {
            "implementation_family": "VOLK_MALHOTRA_SOLAR_SYSTEM_ML_CHAPTER_7",
            "source_commit": "5a112dfe933fe0be48a149f39237eab2803b9c58",
            "license": "MIT",
            "retained_source_bundle_validated": True,
            "short_grid": "0_TO_500000_JULIAN_YEARS_INCLUSIVE_AT_50_YEAR_CADENCE",
            "long_grid": "0_TO_10000000_JULIAN_YEARS_INCLUSIVE_AT_1000_YEAR_CADENCE",
            "feature_extraction_requires_native_state_grids": True,
            "post_run_interpolation_or_reconstruction": False,
            "independence_scope": (
                "INDEPENDENT_FEATURE_AND_MODEL_IMPLEMENTATION_ON_SHARED_JX_TRAJECTORIES_"
                "NOT_AN_INDEPENDENT_DYNAMICS_INTEGRATION"
            ),
            "output_scope": "BROAD_CLASS_ONLY_NOT_SPECIFIC_RESONANCE_IDENTITY",
            "crosscheck_executed": False,
            "crosscheck_passed": False,
        },
        "literature": [
            {
                "citation": "Bernardinelli et al. 2022, ApJS 258:41",
                "doi": "10.3847/1538-4365/ac3914",
                "url": "https://arxiv.org/abs/2109.03758",
                "use": "PRIMARY_CLASS_TAXONOMY_AND_OUTER_RESONANCE_CONTRACT",
                "exact_implementation_published": False,
            },
            {
                "citation": "Khain et al. 2020, AJ 159:133",
                "doi": "10.3847/1538-3881/ab7002",
                "url": "https://arxiv.org/abs/2001.06060",
                "use": "CLASSIFICATION_PRECEDENT_AND_SOURCE_BACKED_EXAMPLE_LABELS",
                "exact_implementation_published": False,
            },
            {
                "citation": "Gladman, Marsden, and VanLaerhoven 2008",
                "url": "https://web.gps.caltech.edu/~mbrown/out/kbbook/Chapters/Gladman_Nomenclature.pdf",
                "use": "HISTORICAL_DYNAMICAL_CLASSIFICATION_FRAMEWORK",
                "mechanical_resonance_detector_published": False,
            },
        ],
        "execution_gate": {
            "blockers": list(BLOCKERS),
            "all_blockers_closed": False,
            "execution_authorized": False,
            "no_automatic_run_after_prepare": True,
            "next_decision": (
                "IMPLEMENT_AND_QUALIFY_THE_MULTI_GRID_WRITER_AND_REDUCED5_ENCOUNTER_ADAPTER_"
                "THEN_EXTERNALLY_PREREGISTER_BEFORE_ANY_10MY_DYNAMICS"
            ),
        },
        "claims": dict(CLAIMS),
        "outcomes_generated": False,
        "execution_authorized": False,
    }
    protocol["content_sha256"] = _content_sha256(protocol)
    return protocol


def validate_protocol(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise PilotProtocolError("protocol must be an exact object")
    if value.get("content_sha256") != _content_sha256(value):
        raise PilotProtocolError("protocol content digest differs")
    expected = build_protocol()
    if _canonical_json(value) != _canonical_json(expected):
        raise PilotProtocolError("protocol differs from exact frozen reconstruction")
    return str(value["content_sha256"])


def _same_node(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _require_lexical_root_identity(
    root: Path,
    held_parent_fd: int,
    held_root_identity: os.stat_result,
) -> None:
    try:
        fresh_parent_fd = os.open(root.parent, _directory_open_flags())
    except OSError as exc:
        raise PilotProtocolError("prepared root parent path was replaced") from exc
    try:
        if not _same_node(os.fstat(held_parent_fd), os.fstat(fresh_parent_fd)):
            raise PilotProtocolError("prepared root parent path was replaced")
        try:
            published_identity = os.stat(
                root.name,
                dir_fd=fresh_parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise PilotProtocolError("prepared root path was replaced") from exc
        if (
            not stat.S_ISDIR(published_identity.st_mode)
            or _stable_stat_identity(held_root_identity)
            != _stable_stat_identity(published_identity)
        ):
            raise PilotProtocolError("prepared root path was replaced")
    finally:
        os.close(fresh_parent_fd)


def _validate_prepared_directory_fd(root_fd: int) -> dict[str, Any]:
    before = os.fstat(root_fd)
    if not stat.S_ISDIR(before.st_mode):
        raise PilotProtocolError("prepared root is not a directory")
    if sorted(os.listdir(root_fd)) != ["PROTOCOL.json"]:
        raise PilotProtocolError("prepared root roster differs")
    payload = _read_regular_bytes_at(root_fd, "PROTOCOL.json")
    value = _strict_json(payload)
    if payload != _canonical_json(value):
        raise PilotProtocolError("prepared protocol is not canonical")
    validate_protocol(value)
    if sorted(os.listdir(root_fd)) != ["PROTOCOL.json"]:
        raise PilotProtocolError("prepared root roster changed during validation")
    after = os.fstat(root_fd)
    if _stable_stat_identity(before) != _stable_stat_identity(after):
        raise PilotProtocolError("prepared root changed during validation")
    return value


def validate_prepared(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(root)
    if not root.name or root.name in (".", ".."):
        raise PilotProtocolError("prepared root name is invalid")
    try:
        parent_fd = os.open(root.parent, _directory_open_flags())
    except OSError as exc:
        raise PilotProtocolError("prepared root parent cannot be opened") from exc
    root_fd: int | None = None
    try:
        root_fd = os.open(
            root.name,
            _directory_open_flags(),
            dir_fd=parent_fd,
        )
        opened_identity = os.fstat(root_fd)
        value = _validate_prepared_directory_fd(root_fd)
        _require_lexical_root_identity(root, parent_fd, opened_identity)
        return value
    except OSError as exc:
        raise PilotProtocolError(
            "prepared root is absent, linked, or not a directory"
        ) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def prepare(root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]:
    root = Path(root)
    if not root.name or root.name in (".", ".."):
        raise PilotProtocolError("output root name is invalid")
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    try:
        parent_fd = os.open(root.parent, _directory_open_flags())
    except OSError as exc:
        raise PilotProtocolError("output root parent cannot be opened") from exc
    root_fd: int | None = None
    root_identity: os.stat_result | None = None
    root_created = False
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor: int | None = None
    file_identity: os.stat_result | None = None
    try:
        try:
            os.mkdir(root.name, mode=0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise PilotProtocolError("create-only output root already exists") from exc
        root_created = True
        root_identity = os.stat(
            root.name,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(root_identity.st_mode):
            raise PilotProtocolError("new output root is not a directory")
        root_fd = os.open(root.name, _directory_open_flags(), dir_fd=parent_fd)
        if not _same_node(root_identity, os.fstat(root_fd)):
            raise PilotProtocolError("new output root was replaced before publication")
        descriptor = os.open("PROTOCOL.json", flags, 0o600, dir_fd=root_fd)
        file_identity = os.fstat(descriptor)
        if not stat.S_ISREG(file_identity.st_mode) or file_identity.st_nlink != 1:
            raise PilotProtocolError("new protocol path is linked or nonregular")
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.fsync(root_fd)
        published_root_identity = os.fstat(root_fd)
        value = _validate_prepared_directory_fd(root_fd)
        _require_lexical_root_identity(
            root,
            parent_fd,
            published_root_identity,
        )
        return value
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        if root_fd is not None and file_identity is not None:
            try:
                current = os.stat(
                    "PROTOCOL.json",
                    dir_fd=root_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISREG(current.st_mode) and _same_node(
                    file_identity, current
                ):
                    os.unlink("PROTOCOL.json", dir_fd=root_fd)
            except OSError:
                pass
        if root_created and root_identity is not None:
            try:
                current_root = os.stat(
                    root.name,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if stat.S_ISDIR(current_root.st_mode) and _same_node(
                    root_identity, current_root
                ):
                    os.rmdir(root.name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate_parser = subparsers.add_parser("validate-prepared")
    validate_parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        value = build_protocol()
    elif arguments.command == "prepare":
        value = prepare(arguments.output_root)
    else:
        value = validate_prepared(arguments.output_root)
    sys.stdout.buffer.write(_canonical_json(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
