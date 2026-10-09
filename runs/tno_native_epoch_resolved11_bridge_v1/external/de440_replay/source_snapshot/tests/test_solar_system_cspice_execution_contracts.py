"""Tests for the unpublished M4C2A CSPICE execution evidence contracts."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
from pathlib import Path
import stat
import struct
import subprocess
import sys
import unittest
from unittest import mock

from jxplanetx.solar_system import cspice_execution_contracts as module
from jxplanetx.solar_system.artifacts import (
    MAXIMUM_LOCAL_ARTIFACT_BYTES,
    ArtifactByteMatchReceipt,
    LocalArtifactVerificationReceipt,
)
from jxplanetx.solar_system.contracts import (
    ArtifactBinding,
    CoordinateEpoch,
    CoverageInterval,
    EphemerisProviderSpec,
    EphemerisQuerySpec,
    ExactUnitScale,
    FrameRealization,
    ProviderIdentity,
    SolarSystemContractError,
    UnitSystemDefinition,
)
from jxplanetx.solar_system.cspice_execution_contracts import (
    CspiceRuntimeObservation,
    CspiceSelectedSegmentRecord,
    CspiceSpkgeoExecutionLane,
    CspiceSpkgeoExecutionReceipt,
    CspiceTargetChainEvidence,
    validate_cspice_runtime_observation,
    validate_cspice_selected_segment_record,
    validate_cspice_spkgeo_execution_lane,
    validate_cspice_spkgeo_execution_receipt,
    validate_cspice_target_chain_evidence,
)
from jxplanetx.solar_system.cspice_time import project_cspice_binary64_query
from jxplanetx.solar_system.serialization import canonical_json, domain_sha256
from jxplanetx.solar_system.states import ProviderNativeStateBatch
from jxplanetx.solar_system.units import KILOGRAM, KILOMETRE, SECOND


EXPECTED_EXPORTS = [
    "CspiceRuntimeObservation",
    "CspiceSelectedSegmentRecord",
    "CspiceSpkgeoExecutionLane",
    "CspiceSpkgeoExecutionReceipt",
    "CspiceTargetChainEvidence",
    "validate_cspice_runtime_observation",
    "validate_cspice_selected_segment_record",
    "validate_cspice_spkgeo_execution_lane",
    "validate_cspice_spkgeo_execution_receipt",
    "validate_cspice_target_chain_evidence",
]

RUNTIME_FIELDS = (
    "provider_identity",
    "worker_source_sha256",
    "worker_source_status",
    "python_implementation",
    "python_version",
    "python_wheel_interpreter_abi_tag",
    "python_executable_byte_length",
    "python_executable_sha256",
    "sys_platform",
    "machine",
    "byteorder",
    "pointer_bits",
    "libc_name",
    "libc_version",
    "binary64_format",
    "fegetround_before",
    "fegetround_after",
    "floating_point_environment_status",
    "python_invocation_status",
    "execution_environment_status",
    "numpy_cpu_feature_payload_byte_length",
    "numpy_cpu_feature_payload_sha256",
    "numpy_cpu_feature_payload_status",
    "distribution_tree_manifest_recipe",
    "numpy_native_tree_selection_and_manifest_recipe",
    "spiceypy_distribution_artifact_id",
    "spiceypy_version",
    "spiceypy_module_relative_path",
    "spiceypy_record_sha256",
    "spiceypy_content_tree_file_count",
    "spiceypy_content_tree_total_byte_length",
    "spiceypy_content_tree_manifest_byte_length",
    "spiceypy_content_tree_sha256",
    "spiceypy_record_validation_status",
    "spiceypy_loaded_module_origin_status",
    "numpy_distribution_artifact_id",
    "numpy_version",
    "numpy_module_relative_path",
    "numpy_record_sha256",
    "numpy_content_tree_file_count",
    "numpy_content_tree_total_byte_length",
    "numpy_content_tree_manifest_byte_length",
    "numpy_content_tree_sha256",
    "numpy_native_tree_file_count",
    "numpy_native_tree_total_byte_length",
    "numpy_native_tree_manifest_byte_length",
    "numpy_native_tree_sha256",
    "numpy_record_validation_status",
    "numpy_loaded_module_origin_status",
    "loaded_cspice_artifact_id",
    "loaded_cspice_relative_path",
    "loaded_cspice_byte_length",
    "loaded_cspice_sha256",
    "cspice_toolkit_version",
    "cspice_loader_resolution_status",
    "cyice_execution_status",
    "loaded_distribution_native_files",
    "loaded_native_map_observation_status",
    "host_native_dependencies",
    "transitive_native_dependency_custody_status",
    "runtime_scope",
    "content_integrity_class",
    "content_sha256",
)
SEGMENT_FIELDS = (
    "kernel_artifact_id",
    "kernel_artifact_sha256",
    "kernel_load_index",
    "selection_et",
    "selected_body_naif_id",
    "center_naif_id",
    "frame_naif_id",
    "segment_type",
    "first_et",
    "last_et",
    "begin_daf_address",
    "end_daf_address",
    "segment_identifier",
    "spkpvn_returned_frame_naif_id",
    "spkpvn_returned_center_naif_id",
    "spkpvn_state",
    "selection_status",
    "evidence_scope",
    "content_sha256",
)
CHAIN_FIELDS = (
    "target_naif_id",
    "observer_naif_id",
    "reference_frame_name",
    "effective_et",
    "spkgeo_state",
    "spkgeo_light_time_seconds",
    "legs",
    "terminal_center_naif_id",
    "reconstructed_state",
    "comparison_recipe",
    "comparison_status",
    "evidence_scope",
    "content_sha256",
)
LANE_FIELDS = (
    "lane_role",
    "execution_instance_nonce",
    "execution_instance_evidence_status",
    "provider_profile",
    "projection",
    "runtime",
    "implementation_artifact_verifications",
    "pre_spk_verifications",
    "post_spk_verifications",
    "ordered_loaded_artifact_ids",
    "kernel_pool_count_field_order",
    "fresh_pool_counts",
    "loaded_pool_counts",
    "post_query_pool_counts",
    "after_clear_pool_counts",
    "loaded_kernel_inventory",
    "kernel_pool_status",
    "cspice_error_state_status",
    "provider_call",
    "state_batch",
    "target_chains",
    "state_payload_recipe",
    "state_payload_byte_length",
    "state_payload_sha256",
    "provider_execution_status",
    "nested_batch_evidence_status",
    "atomicity_status",
    "artifact_at_use_custody_status",
    "process_isolation_status",
    "network_status",
    "fallback_status",
    "extrapolation_status",
    "portability_scope",
    "semantic_content_sha256",
    "content_integrity_class",
    "content_sha256",
)
RECEIPT_FIELDS = (
    "receipt_id",
    "primary_lane",
    "replay_lane",
    "semantic_replay_comparison_scope",
    "semantic_replay_status",
    "claim_scope",
    "content_integrity_class",
    "content_sha256",
)

INTEGRITY = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
PROVIDER_ID = "provider.spiceypy-8.2.0.cspice-n0067-derived.v1"
IMPLEMENTATION_ID = "implementation.spiceypy-8.2.0.cspice-n0067-derived.v1"
PROVIDER_SPEC_ID = (
    "provider-spec.spiceypy-8.2.0.cspice-n0067-derived.single-spk-j2000.v1"
)
CSPICE_ARTIFACT_VERSION = "SpiceyPy-8.2.0-bundled-N0067-derived-patched"
LICENSE_ID = "artifact.cspice.license"
SPICEYPY_ID = "artifact.cspice.spiceypy-wheel"
NUMPY_ID = "artifact.cspice.numpy-wheel"
CSPICE_ID = "artifact.cspice.native"
SPK_ID = "artifact.cspice.de440s"
CSPICE_SHA = "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84"
SPK_SHA = "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2"
COMPOSITE_LICENSE = "MIT AND LicenseRef-NAIF-SPICE-Rules"

ABI_FIXTURES = {
    "cp312-cp312": {
        "python_version": "3.12.13",
        "runtime_id": (
            "runtime.cpython-3.12.13.cp312-linux-x86_64.host-local-observed.v1"
        ),
        "spiceypy_wheel": (
            2_430_343,
            "c91643e43d0b1d0de5616ffd9a2a82a7e2ae7ea50c1bf5f740e89ed1e67d4996",
        ),
        "spiceypy_record": (
            "da050a7e1cb4abbea030a804090f40ca7ec3283a8b059478fcd5d4754a515d47"
        ),
        "spiceypy_tree": (
            29,
            7_197_075,
            2_950,
            "1a03ac00e4816b237c93d1d0643494f50f0d04a9df2a253cdf211b519d6bd4ab",
        ),
        "numpy_wheel": (
            16_606_086,
            "0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28",
        ),
        "numpy_record": (
            "ddd8a4670b2a83f7b3ca989ae0256137e5b75dd78b5faede406301db81fe9d79"
        ),
        "numpy_tree": (
            900,
            57_522_497,
            95_533,
            "c45b6e7c5c50537127e5f7cc7f861c6ecc032471a78de972ea1d02ac066dfaf7",
        ),
        "numpy_native_tree": (
            22,
            45_593_806,
            2_816,
            "c6e2662d4564486d186fa8dc066225f67b16e44bb55e8216ce8497a289d45c8d",
        ),
        "loaded_numpy_natives": (
            (
                "numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0",
                2_833_617,
                "c6090048eccc763522c12ef016f81da6b627cb3a044f55cf0479a839c41c0980",
            ),
            (
                "numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0",
                250_985,
                "6ed5137f412781ad7863439fb543613f620b43c32b63292a0029246162f5bbc6",
            ),
            (
                "numpy.libs/libscipy_openblas64_-fdde5778.so",
                25_034_001,
                "ee2039cf13a45e1e89f04d0cea9585657d93ffaa425594e1935161013fdf4e12",
            ),
            (
                "numpy/_core/_multiarray_umath.cpython-312-x86_64-linux-gnu.so",
                10_808_937,
                "ba96467ae494babaf7c3458b09cc03c4200c2e10010860d80e452ffb8b01c8cd",
            ),
            (
                "numpy/linalg/_umath_linalg.cpython-312-x86_64-linux-gnu.so",
                231_833,
                "64bfb8bb8a1a09ca31f7a6e307a87b4c78a75f3489aaf3c61ed8aa4ffada9845",
            ),
        ),
    },
    "cp314-cp314": {
        "python_version": "3.14.4",
        "runtime_id": (
            "runtime.cpython-3.14.4.cp314-linux-x86_64.host-local-observed.v1"
        ),
        "spiceypy_wheel": (
            2_442_346,
            "00c2714d3dd67180794ad0b66735d6cac470e19627d6e99f9628a50b52133d90",
        ),
        "spiceypy_record": (
            "ae6ccb8132cec7e02cf020785df7e6c2b36c344eb215253757f8dcb100ee1002"
        ),
        "spiceypy_tree": (
            29,
            7_209_395,
            2_950,
            "f8f65e07e738b4beab2a277bac53ced6338ce9ed001f9c84930e1b54988700db",
        ),
        "numpy_wheel": (
            16_597_350,
            "fffe29a1ef00883599d1dc2c51aa2e5d80afe49523c261a74933df395c15c520",
        ),
        "numpy_record": (
            "47521114855987c11cea7b07e2414bd40fafac4fe21921c389da6cbee9d82f4b"
        ),
        "numpy_tree": (
            900,
            57_505_893,
            95_533,
            "685ec34fdf369dc90dd4cc8719e36b24279a8adba0f0b118304406e1f04f8ee4",
        ),
        "numpy_native_tree": (
            22,
            45_577_206,
            2_816,
            "2c3a6d532868e754cba081d053fd3e3b18ad5cf4f08b2a692be2734ced194192",
        ),
        "loaded_numpy_natives": (
            (
                "numpy.libs/libgfortran-040039e1-0352e75f.so.5.0.0",
                2_833_617,
                "c6090048eccc763522c12ef016f81da6b627cb3a044f55cf0479a839c41c0980",
            ),
            (
                "numpy.libs/libquadmath-96973f99-934c22de.so.0.0.0",
                250_985,
                "6ed5137f412781ad7863439fb543613f620b43c32b63292a0029246162f5bbc6",
            ),
            (
                "numpy.libs/libscipy_openblas64_-fdde5778.so",
                25_034_001,
                "ee2039cf13a45e1e89f04d0cea9585657d93ffaa425594e1935161013fdf4e12",
            ),
            (
                "numpy/_core/_multiarray_umath.cpython-314-x86_64-linux-gnu.so",
                10_800_697,
                "61052ef771d926665ca135b1840662516f058d5c98843aa429ef2abdd7f46a04",
            ),
            (
                "numpy/linalg/_umath_linalg.cpython-314-x86_64-linux-gnu.so",
                231_833,
                "ce9f59d6d4adf07b7ad262400d87484af53a3c3bafbfcca10c25327cbc75f5e9",
            ),
        ),
    },
}

BYTE_SCOPE = "EXACT_DECLARED_ARTIFACT_BYTES_ONE_READ_STREAM"
BYTE_STATUS = "OBSERVED_LENGTH_AND_SHA256_EQUAL_DECLARATION"
VERIFICATION_STAGE = "LOCAL_BYTE_STREAM_VERIFICATION_NO_PROVIDER_EXECUTION"
PLATFORM_SCOPE = "CAPABILITY_GATED_POSIX_PROFILE_NO_PORTABILITY_CLAIM"
ROOT_STATUS = "CALLER_DIRECTORY_FD_DUPLICATED_ROOT_PATH_ACQUISITION_UNATTESTED"
LOCATOR_POLICY = "SEQUENTIAL_COMPONENTWISE_DIR_FD_O_NOFOLLOW"
FINAL_OPEN_POLICY = "ONE_FINAL_OPEN_DESCRIPTION_READ_ONLY_NONBLOCKING_NOCTTY"
READ_POLICY = "BOUNDED_OS_READ_DECLARED_LENGTH_PLUS_ONE_EOF_PROBE"
METADATA_STATUS = "SELECTED_PREOPEN_PRE_READ_POST_READ_AND_SECOND_STAT_FIELDS_IDENTICAL"
SECOND_STATUS = "SECOND_NAMESPACE_OBSERVATION_MATCHED_OPEN_FILE_IDENTITY_NOT_CONTINUITY_PROOF"
LOCAL_METADATA = "LOCAL_NONPORTABLE_OBSERVATION_NOT_CONTENT_IDENTITY"
CUSTODY_SCOPE = "READ_STREAM_ONLY_NO_FUTURE_PATH_OR_FD_USE_AUTHORITY"
LOCAL_EXECUTION = "LOCAL_VERIFIER_EXECUTION_ONLY_NO_PROVIDER_AT_USE_OR_PROVIDER_EXECUTION"
LOCAL_REPLAY = "REQUIRES_FRESH_VERIFICATION_WITH_NEW_ROOT_CAPABILITY"
ROOT_STAT_FIELDS = ("st_dev", "st_ino", "st_mode")
FILE_STAT_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def epoch(whole: int, fraction: float) -> CoordinateEpoch:
    return CoordinateEpoch(
        time_scale="TDB",
        representation="SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=SECOND.unit_id,
        origin_id="SPICE_J2000_TDB_ORIGIN",
        realization_id="de440s.spice-et.v1",
        artifact_ids=(),
    )


def artifact(
    artifact_id: str,
    role: str,
    version: str,
    length: int,
    digest: str,
    *,
    provider_id: str = PROVIDER_ID,
    coverage: tuple[CoverageInterval, ...] | None = None,
    load_order: int | None = None,
) -> ArtifactBinding:
    return ArtifactBinding(
        artifact_id=artifact_id,
        artifact_role=role,
        provider_id=provider_id,
        version=version,
        logical_locator=f"retained/{artifact_id}.bin",
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=length,
        artifact_sha256=digest,
        media_type="application/octet-stream",
        coverage_status=(
            "COARSE_ARTIFACT_TIME_ENVELOPE" if coverage is not None else "NOT_APPLICABLE"
        ),
        coverage=coverage,
        license_evidence_status="RETAINED_HASH_BOUND_LICENSE_ARTIFACT",
        license_spdx=COMPOSITE_LICENSE,
        license_artifact_id=LICENSE_ID,
        redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
        load_order_status=("ORDERED_LOAD_MEMBER" if load_order is not None else "NOT_LOADABLE"),
        load_order=load_order,
        extrapolation_policy="FORBID",
        content_integrity_class=INTEGRITY,
    )


def license_artifact(*, provider_id: str = PROVIDER_ID) -> ArtifactBinding:
    return ArtifactBinding(
        artifact_id=LICENSE_ID,
        artifact_role="LICENSE",
        provider_id=provider_id,
        version="fixture-composite-v1",
        logical_locator="retained/COMPOSITE-LICENSE.txt",
        locator_kind="LOCAL_REGULAR_FILE",
        byte_length=128,
        artifact_sha256="9" * 64,
        media_type="text/plain",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        license_evidence_status="RETAINED_LICENSE_TEXT_SELF_EVIDENCE_NONAUTHORIZING",
        license_spdx=COMPOSITE_LICENSE,
        license_artifact_id=None,
        redistribution_status="BUNDLED_WITH_RETAINED_LICENSE_EVIDENCE",
        load_order_status="NOT_LOADABLE",
        load_order=None,
        extrapolation_policy="FORBID",
        content_integrity_class=INTEGRITY,
    )


def context(
    abi: str = "cp314-cp314",
    *,
    provider_id: str = PROVIDER_ID,
    implementation_id: str = IMPLEMENTATION_ID,
    runtime_id: str | None = None,
    spec_id: str = PROVIDER_SPEC_ID,
    cspice_artifact_version: str = CSPICE_ARTIFACT_VERSION,
) -> tuple[EphemerisProviderSpec, EphemerisQuerySpec]:
    profile = ABI_FIXTURES[abi]
    if runtime_id is None:
        runtime_id = profile["runtime_id"]
    coverage = CoverageInterval(epoch(-100, 0.0), epoch(100, 0.0), "CLOSED_CLOSED")
    spiceypy = artifact(
        SPICEYPY_ID,
        "SOFTWARE_DISTRIBUTION",
        "8.2.0",
        profile["spiceypy_wheel"][0],
        profile["spiceypy_wheel"][1],
        provider_id=provider_id,
    )
    numpy = artifact(
        NUMPY_ID,
        "SOFTWARE_DISTRIBUTION",
        "2.3.5",
        profile["numpy_wheel"][0],
        profile["numpy_wheel"][1],
        provider_id=provider_id,
    )
    cspice = artifact(
        CSPICE_ID,
        "NATIVE_LIBRARY",
        cspice_artifact_version,
        3_561_056,
        CSPICE_SHA,
        provider_id=provider_id,
    )
    spk = artifact(
        SPK_ID,
        "SPK",
        "de440s",
        32_726_016,
        SPK_SHA,
        provider_id=provider_id,
        coverage=(coverage,),
        load_order=0,
    )
    identity = ProviderIdentity(
        provider_id=provider_id,
        implementation_id=implementation_id,
        provider_kind="LOCAL_OFFLINE_SPK",
        version="8.2.0",
        runtime_id=runtime_id,
        module_name="spiceypy",
        distribution_name="spiceypy",
        distribution_version="8.2.0",
        ordered_implementation_artifact_ids=(SPICEYPY_ID, NUMPY_ID, CSPICE_ID),
    )
    frame = FrameRealization(
        frame_id="frame.naif-j2000.ssb.tdb-compatible.v1",
        frame_kind="TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
        origin_kind="SOLAR_SYSTEM_BARYCENTER",
        origin_naif_id=0,
        origin_realization_id="NAIF_BODY_0_SOLAR_SYSTEM_BARYCENTER",
        axes_realization_id="NAIF_BUILTIN_INERTIAL_FRAME_1_J2000",
        orientation_model_id="NAIF_BUILTIN_J2000_STATIC_ORIENTATION",
        orientation_time_dependence="STATIC",
        coordinate_time_scale="TDB",
        coverage_status="RETAINED",
        coverage=(coverage,),
        artifact_ids=(SPK_ID,),
    )
    gm = ExactUnitScale(
        unit_id="unit.kilometre3-per-second2.defined-exact.v1",
        dimension="GRAVITATIONAL_PARAMETER",
        si_unit_id="si.metre3-per-second2",
        numerator=1_000_000_000,
        denominator=1,
        definition_classification="DEFINED_EXACT",
        artifact_ids=(),
    )
    units = UnitSystemDefinition(
        unit_system_id="units.cspice.kilometre-second-kilogram.v1",
        length=KILOMETRE,
        time=SECOND,
        mass=KILOGRAM,
        gravitational_parameter=gm,
        coordinate_time_scale="TDB",
    )
    artifacts = tuple(
        sorted(
            (spiceypy, numpy, cspice, spk, license_artifact(provider_id=provider_id)),
            key=lambda x: x.artifact_id,
        )
    )
    provider = EphemerisProviderSpec(
        spec_id=spec_id,
        identity=identity,
        artifacts=artifacts,
        ordered_load_artifact_ids=(SPK_ID,),
        capabilities=(
            "COARSE_ARTIFACT_TIME_ENVELOPE",
            "GEOMETRIC_CARTESIAN_STATE",
            "TDB_COORDINATE_EPOCH",
        ),
        native_time_scale="TDB",
        native_frame=frame,
        native_output_unit_system=units,
        network_access=False,
        fallback_allowed=False,
        extrapolation_policy="FORBID",
        evidence_class="MODEL_OUTPUT",
        registry_authorized=False,
        qualification_authorized=False,
    )
    query = EphemerisQuerySpec(
        query_id="query.cspice.requested.fixture",
        provider=provider,
        epoch=epoch(1, float.fromhex("0x1p-53")),
        target_naif_ids=(10, 399),
        observer_naif_id=0,
        frame=frame,
        aberration_correction="NONE",
        output_unit_system=units,
        state_kind="GEOMETRIC",
        target_chain_availability_status=(
            "REQUIRES_RUNTIME_PROVIDER_TARGET_AVAILABILITY_VALIDATION"
        ),
    )
    return provider, query


def local_receipt(bound: ArtifactBinding, seed: int) -> LocalArtifactVerificationReceipt:
    byte_match = ArtifactByteMatchReceipt(
        artifact=bound,
        verification_scope=BYTE_SCOPE,
        hash_algorithm="SHA-256",
        expected_byte_length=bound.byte_length,
        observed_byte_length=bound.byte_length,
        expected_sha256=bound.artifact_sha256,
        observed_sha256=bound.artifact_sha256,
        byte_match_status=BYTE_STATUS,
        content_integrity_class=INTEGRITY,
    )
    root_stat = (2_049, seed, stat.S_IFDIR | 0o700)
    file_stat = (
        2_049,
        seed + 10_000,
        stat.S_IFREG | 0o600,
        1,
        1_000,
        1_000,
        bound.byte_length,
        1_700_000_000_000_000_000 + seed,
        1_700_000_000_100_000_000 + seed,
    )
    return LocalArtifactVerificationReceipt(
        byte_match=byte_match,
        verification_stage=VERIFICATION_STAGE,
        platform_scope=PLATFORM_SCOPE,
        root_capability_status=ROOT_STATUS,
        locator_resolution_policy=LOCATOR_POLICY,
        locator_components=tuple(bound.logical_locator.split("/")),
        final_open_policy=FINAL_OPEN_POLICY,
        read_policy=READ_POLICY,
        maximum_byte_length=MAXIMUM_LOCAL_ARTIFACT_BYTES,
        read_chunk_byte_limit=1 << 20,
        maximum_data_read_calls=8_192,
        data_read_call_count=(bound.byte_length + (1 << 20) - 1) // (1 << 20),
        eof_probe_call_count=1,
        root_stat_field_order=ROOT_STAT_FIELDS,
        pre_root_stat=root_stat,
        post_root_stat=root_stat,
        file_stat_field_order=FILE_STAT_FIELDS,
        pre_open_path_file_stat=file_stat,
        pre_read_file_stat=file_stat,
        post_read_file_stat=file_stat,
        second_resolution_file_stat=file_stat,
        metadata_stability_status=METADATA_STATUS,
        second_resolution_status=SECOND_STATUS,
        local_metadata_classification=LOCAL_METADATA,
        custody_scope=CUSTODY_SCOPE,
        execution_evidence_status=LOCAL_EXECUTION,
        semantic_replay_status=LOCAL_REPLAY,
        content_integrity_class=INTEGRITY,
    )


def runtime(
    provider: EphemerisProviderSpec,
    abi: str = "cp314-cp314",
) -> CspiceRuntimeObservation:
    profile = ABI_FIXTURES[abi]
    native_entries = tuple(
        sorted(
            ( 
                (
                    SPICEYPY_ID,
                    "spiceypy/utils/libcspice.so",
                    3_561_056,
                    CSPICE_SHA,
                ),
            )
            + tuple(
                (NUMPY_ID, path, byte_length, sha256)
                for path, byte_length, sha256 in profile["loaded_numpy_natives"]
            )
        )
    )
    host = (
        (
            "ld-linux-x86-64.so.2",
            254_864,
            "c5e80a563850d6ab5c2f2482e4202d9c1b71fbf44854b8c399e63527202c64e1",
        ),
        (
            "libc.so.6",
            2_186_512,
            "a3947513a02831ec692ebf13053c07614882ab54a2101fb91a1b15724062ed0c",
        ),
        (
            "libm.so.6",
            1_198_376,
            "beea4eeacfcfa2cd96011b959a826c97cf4a774017e214f6a34d7eea3d49cd88",
        ),
    )
    return CspiceRuntimeObservation(
        provider_identity=provider.identity,
        worker_source_sha256="a" * 64,
        worker_source_status=(
            "CALLER_OBSERVED_SOURCE_DIGEST_NOT_QUALIFIED_UNTIL_M4C2B"
        ),
        python_implementation="CPython",
        python_version=profile["python_version"],
        python_wheel_interpreter_abi_tag=abi,
        python_executable_byte_length=7_481_192,
        python_executable_sha256="b8d8288faefdd300201f43fcf00f6f539a27218eeed3a3dff5ab10b9c4c99700",
        sys_platform="linux",
        machine="x86_64",
        byteorder="little",
        pointer_bits=64,
        libc_name="glibc",
        libc_version="2.43",
        binary64_format="IEEE, little-endian",
        fegetround_before=0,
        fegetround_after=0,
        floating_point_environment_status="FE_TONEAREST_BEFORE_AND_AFTER_PROVIDER_CALLS",
        python_invocation_status=(
            "CPYTHON_ISOLATED_MODE_IGNORE_PYTHON_ENVIRONMENT_AND_NO_BYTECODE_WRITES"
        ),
        execution_environment_status=(
            "MINIMAL_C_UTF8_ENVIRONMENT_CSPICE_AND_DYNAMIC_LOADER_OVERRIDES_ABSENT_"
            "OPENBLAS_OMP_MKL_AND_NUMEXPR_THREADS_ONE"
        ),
        numpy_cpu_feature_payload_byte_length=1_086,
        numpy_cpu_feature_payload_sha256="b050bd98f1d010d3f10c7d685e2eb6875793ddb9909110de8474aba96740fe74",
        numpy_cpu_feature_payload_status=(
            "SORTED_COMPACT_UTF8_JSON_OF_NUMPY_CPU_BASELINE_DISPATCH_AND_FEATURES"
        ),
        distribution_tree_manifest_recipe=(
            "RECORD_ROWS_SORTED_BY_UTF8_RELATIVE_PATH_PATH_NUL_DECIMAL_SIZE_NUL_"
            "LOWERCASE_SHA256_LF"
        ),
        numpy_native_tree_selection_and_manifest_recipe=(
            "RECORD_PATH_SUFFIX_SO_DLL_DYLIB_OR_BASENAME_CONTAINS_DOT_SO_DOT_THEN_"
            "DISTRIBUTION_MANIFEST_RECIPE"
        ),
        spiceypy_distribution_artifact_id=SPICEYPY_ID,
        spiceypy_version="8.2.0",
        spiceypy_module_relative_path="spiceypy/__init__.py",
        spiceypy_record_sha256=profile["spiceypy_record"],
        spiceypy_content_tree_file_count=profile["spiceypy_tree"][0],
        spiceypy_content_tree_total_byte_length=profile["spiceypy_tree"][1],
        spiceypy_content_tree_manifest_byte_length=profile["spiceypy_tree"][2],
        spiceypy_content_tree_sha256=profile["spiceypy_tree"][3],
        spiceypy_record_validation_status=(
            "WHEEL_RECORD_VALIDATED_ALL_HASHED_MEMBERS_CLOSED_EXTRACTION_NO_SYMLINKS_NO_PYC"
        ),
        spiceypy_loaded_module_origin_status="MODULE_ORIGIN_WITHIN_EXACT_VALIDATED_EXTRACTION",
        numpy_distribution_artifact_id=NUMPY_ID,
        numpy_version="2.3.5",
        numpy_module_relative_path="numpy/__init__.py",
        numpy_record_sha256=profile["numpy_record"],
        numpy_content_tree_file_count=profile["numpy_tree"][0],
        numpy_content_tree_total_byte_length=profile["numpy_tree"][1],
        numpy_content_tree_manifest_byte_length=profile["numpy_tree"][2],
        numpy_content_tree_sha256=profile["numpy_tree"][3],
        numpy_native_tree_file_count=profile["numpy_native_tree"][0],
        numpy_native_tree_total_byte_length=profile["numpy_native_tree"][1],
        numpy_native_tree_manifest_byte_length=profile["numpy_native_tree"][2],
        numpy_native_tree_sha256=profile["numpy_native_tree"][3],
        numpy_record_validation_status=(
            "WHEEL_RECORD_VALIDATED_ALL_HASHED_MEMBERS_CLOSED_EXTRACTION_NO_SYMLINKS_NO_PYC"
        ),
        numpy_loaded_module_origin_status="MODULE_ORIGIN_WITHIN_EXACT_VALIDATED_EXTRACTION",
        loaded_cspice_artifact_id=CSPICE_ID,
        loaded_cspice_relative_path="spiceypy/utils/libcspice.so",
        loaded_cspice_byte_length=3_561_056,
        loaded_cspice_sha256=CSPICE_SHA,
        cspice_toolkit_version="CSPICE_N0067",
        cspice_loader_resolution_status=(
            "LIBSPICEHELPER_BUNDLED_RELATIVE_PATH_AND_FILE_SHA256_MATCHED_NO_OVERRIDE_"
            "OR_SYSTEM_FALLBACK"
        ),
        cyice_execution_status=(
            "CALLER_ASSERTED_PACKAGED_CYICE_NOT_IMPORTED_OR_MAPPED_NO_RETAINED_"
            "IMPORT_OR_MAP_CORRELATION_RECEIPT"
        ),
        loaded_distribution_native_files=native_entries,
        loaded_native_map_observation_status=(
            "CALLER_ASSERTED_POINT_OBSERVED_FILE_BACKED_MAP_ROSTER_NO_RETAINED_"
            "CORRELATION_RECEIPT_OR_MAPPED_PAGE_CUSTODY"
        ),
        host_native_dependencies=host,
        transitive_native_dependency_custody_status=(
            "CALLER_ASSERTED_DISTRIBUTION_NATIVE_AND_HOST_LIBC_LIBM_LOADER_ROSTERS_"
            "NO_RETAINED_MAP_CORRELATION_OR_COMPLETE_HOST_SBOM"
        ),
        runtime_scope="HOST_LOCAL_EMPIRICAL_LINUX_X86_64_GLIBC_CPU_FEATURE_BOUND_ONLY",
        content_integrity_class=INTEGRITY,
    )


def segment(
    body: int,
    center: int,
    state: tuple[float, float, float, float, float, float],
    selection_et: float,
    begin: int,
) -> CspiceSelectedSegmentRecord:
    return CspiceSelectedSegmentRecord(
        kernel_artifact_id=SPK_ID,
        kernel_artifact_sha256=SPK_SHA,
        kernel_load_index=0,
        selection_et=selection_et,
        selected_body_naif_id=body,
        center_naif_id=center,
        frame_naif_id=1,
        segment_type=2,
        first_et=-100.0,
        last_et=100.0,
        begin_daf_address=begin,
        end_daf_address=begin + 99,
        segment_identifier="DE-0440LE-0440",
        spkpvn_returned_frame_naif_id=1,
        spkpvn_returned_center_naif_id=center,
        spkpvn_state=state,
        selection_status="SPKSFS_IDLEN_41_SELECTED_SPKUDS_DECODED_SPKPVN_EVALUATED",
        evidence_scope=(
            "SELECTED_SEGMENT_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_INTERNAL_TRACE"
        ),
    )


def chain_fixture(
    projection,
) -> tuple[CspiceTargetChainEvidence, CspiceTargetChainEvidence]:
    et = projection.binary64_projection.rounded_value
    first_state = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    first = segment(10, 0, first_state, et, 100)
    chain_one = CspiceTargetChainEvidence(
        target_naif_id=10,
        observer_naif_id=0,
        reference_frame_name="J2000",
        effective_et=et,
        spkgeo_state=first_state,
        spkgeo_light_time_seconds=10.0,
        legs=(first,),
        terminal_center_naif_id=0,
        reconstructed_state=first_state,
        comparison_recipe=(
            "FIRST_LEG_STATE_THEN_COMPONENTWISE_BINARY64_ADDITION_BODY_TO_CENTER_"
            "FE_TONEAREST"
        ),
        comparison_status=(
            "RECONSTRUCTED_AND_SPKGEO_STATES_RAW_BIG_ENDIAN_F8_IDENTICAL"
        ),
        evidence_scope=(
            "SPKSFS_SPKUDS_SPKPVN_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_TRACE"
        ),
    )
    first_leg = segment(399, 3, (10.0, 20.0, 30.0, 40.0, 50.0, 60.0), et, 200)
    second_leg = segment(3, 0, first_state, et, 300)
    result = (11.0, 22.0, 33.0, 44.0, 55.0, 66.0)
    chain_two = CspiceTargetChainEvidence(
        target_naif_id=399,
        observer_naif_id=0,
        reference_frame_name="J2000",
        effective_et=et,
        spkgeo_state=result,
        spkgeo_light_time_seconds=20.0,
        legs=(first_leg, second_leg),
        terminal_center_naif_id=0,
        reconstructed_state=result,
        comparison_recipe=(
            "FIRST_LEG_STATE_THEN_COMPONENTWISE_BINARY64_ADDITION_BODY_TO_CENTER_"
            "FE_TONEAREST"
        ),
        comparison_status=(
            "RECONSTRUCTED_AND_SPKGEO_STATES_RAW_BIG_ENDIAN_F8_IDENTICAL"
        ),
        evidence_scope=(
            "SPKSFS_SPKUDS_SPKPVN_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_TRACE"
        ),
    )
    return chain_one, chain_two


def state_batch(projection) -> ProviderNativeStateBatch:
    query = projection.effective_query
    return ProviderNativeStateBatch(
        batch_id="batch.cspice.executed.value-container.fixture",
        query=query,
        state_stage=(
            "DECLARED_PROVIDER_NATIVE_GEOMETRIC_TARGET_RELATIVE_OBSERVER_STATE_"
            "NO_JX_CONVERSION"
        ),
        execution_evidence_status="CALLER_SUPPLIED_UNAUTHENTICATED_NO_EXECUTION_RECEIPT",
        target_naif_ids=query.target_naif_ids,
        component_order=("X", "Y", "Z"),
        state_shape=(2, 3),
        position_unit_id=KILOMETRE.unit_id,
        velocity_length_unit_id=KILOMETRE.unit_id,
        velocity_time_unit_id=SECOND.unit_id,
        velocity_semantics="COORDINATE_DERIVATIVE_PER_PROVIDER_NATIVE_TIME_UNIT",
        positions=(1.0, 2.0, 3.0, 11.0, 22.0, 33.0),
        velocities=(4.0, 5.0, 6.0, 44.0, 55.0, 66.0),
        target_availability_status=query.target_chain_availability_status,
        artifact_custody_status="NO_ARTIFACT_AT_USE_CUSTODY_EVIDENCE",
        semantic_replay_status="REQUIRES_PROVIDER_SPECIFIC_SEMANTIC_REPLAY",
        content_integrity_class=INTEGRITY,
    )


def state_payload(batch: ProviderNativeStateBatch) -> bytes:
    result = bytearray()
    for index in range(len(batch.target_naif_ids)):
        offset = index * 3
        result.extend(
            struct.pack(
                ">6d",
                *batch.positions[offset : offset + 3],
                *batch.velocities[offset : offset + 3],
            )
        )
    return bytes(result)


def lane(
    role: str,
    nonce: str,
    *,
    local_seed: int = 10,
    abi: str = "cp314-cp314",
    provider_id: str = PROVIDER_ID,
    implementation_id: str = IMPLEMENTATION_ID,
    runtime_id: str | None = None,
    spec_id: str = PROVIDER_SPEC_ID,
    cspice_artifact_version: str = CSPICE_ARTIFACT_VERSION,
) -> CspiceSpkgeoExecutionLane:
    provider, query = context(
        abi,
        provider_id=provider_id,
        implementation_id=implementation_id,
        runtime_id=runtime_id,
        spec_id=spec_id,
        cspice_artifact_version=cspice_artifact_version,
    )
    projection = project_cspice_binary64_query(
        query,
        effective_query_id="query.cspice.effective.fixture",
    )
    by_id = {item.artifact_id: item for item in provider.artifacts}
    implementation = tuple(
        local_receipt(by_id[item], local_seed + index)
        for index, item in enumerate(provider.identity.ordered_implementation_artifact_ids)
    )
    before = local_receipt(by_id[SPK_ID], local_seed + 100)
    after = local_receipt(by_id[SPK_ID], local_seed + 101)
    batch = state_batch(projection)
    chains = chain_fixture(projection)
    payload = state_payload(batch)
    return CspiceSpkgeoExecutionLane(
        lane_role=role,
        execution_instance_nonce=nonce,
        execution_instance_evidence_status=(
            "CALLER_SUPPLIED_NONSECRET_UNAUTHENTICATED_INSTANCE_NONCE_NO_PROCESS_"
            "IDENTITY_PROOF"
        ),
        provider_profile=(
            "SPICEYPY_8_2_0_N0067_DERIVED_CSPICE_SPKGEO_J2000_TYPE2_KM_PER_S_V1"
        ),
        projection=projection,
        runtime=runtime(provider, abi),
        implementation_artifact_verifications=implementation,
        pre_spk_verifications=(before,),
        post_spk_verifications=(after,),
        ordered_loaded_artifact_ids=(SPK_ID,),
        kernel_pool_count_field_order=("ALL", "SPK"),
        fresh_pool_counts=(0, 0),
        loaded_pool_counts=(1, 1),
        post_query_pool_counts=(1, 1),
        after_clear_pool_counts=(0, 0),
        loaded_kernel_inventory=((SPK_ID, "DAF", "SPK", "DIRECT_FURNSH_SOURCE_EMPTY_NO_META_KERNEL"),),
        kernel_pool_status="EMPTY_THEN_ONE_DIRECT_SPK_LOAD_STABLE_THROUGH_QUERY_THEN_CLEARED",
        cspice_error_state_status=(
            "FRESH_ERROR_STATE_CLEAR_AND_NO_FAILED_STATE_AFTER_THIS_LANE_PROVIDER_"
            "AND_SELECTOR_CALLS"
        ),
        provider_call=(
            "CSPICE_SPKGEO_TARGET_EFFECTIVE_BINARY64_ET_J2000_OBSERVER_0_GEOMETRIC_"
            "NO_ABERRATION"
        ),
        state_batch=batch,
        target_chains=chains,
        state_payload_recipe="TARGET_ORDER_X_Y_Z_VX_VY_VZ_BIG_ENDIAN_BINARY64",
        state_payload_byte_length=len(payload),
        state_payload_sha256=hashlib.sha256(payload).hexdigest(),
        provider_execution_status=(
            "CSPICE_SPKGEO_GEOMETRIC_STATE_AT_M4C1_EFFECTIVE_BINARY64_ET_NO_JX_CONVERSION"
        ),
        nested_batch_evidence_status=(
            "NESTED_BATCH_VALUE_CONTAINER_OUTER_RECEIPT_SUPPLIES_DECLARED_EXECUTION_EVIDENCE"
        ),
        atomicity_status="NO_ATOMIC_SNAPSHOT_CONTINUOUS_CUSTODY_OR_MUTATE_RESTORE_PROOF",
        artifact_at_use_custody_status=(
            "PRE_AND_POST_POINT_OBSERVATIONS_ONLY_NO_PROVIDER_MAPPED_PAGE_AT_USE_PROOF"
        ),
        process_isolation_status=(
            "FRESH_PROCESS_PER_LANE_CALLER_ASSERTED_NO_HERMETIC_SANDBOX_OR_"
            "INDEPENDENCE_PROOF"
        ),
        network_status=(
            "CALLER_ASSERTED_NO_NETWORK_AVAILABLE_OR_USED_NO_SYSCALL_OR_SANDBOX_"
            "ATTESTATION"
        ),
        fallback_status=(
            "DECLARED_FIXED_PROFILE_NO_FALLBACK_PATH_USED_NOT_INDEPENDENTLY_ENFORCED"
        ),
        extrapolation_status=(
            "DECLARED_FIXED_PROFILE_NO_EXTRAPOLATION_PATH_USED_NOT_INDEPENDENTLY_"
            "ENFORCED"
        ),
        portability_scope="HOST_LOCAL_EMPIRICAL_PROFILE_ONLY",
        content_integrity_class=INTEGRITY,
    )


def receipt() -> CspiceSpkgeoExecutionReceipt:
    return CspiceSpkgeoExecutionReceipt(
        receipt_id="receipt.cspice.execution.fixture",
        primary_lane=lane("PRIMARY", "a" * 64, local_seed=10),
        replay_lane=lane("REPLAY", "b" * 64, local_seed=1_000),
        semantic_replay_comparison_scope=(
            "DELIBERATE_RESULT_CORE_INCLUDES_EXACT_RUNTIME_EXCLUDES_ROLE_INSTANCE_"
            "LOCAL_FILESYSTEM_METADATA_AND_BATCH_IDENTIFIER"
        ),
        semantic_replay_status=(
            "DISTINCT_CALLER_INSTANCE_NONCES_AND_EQUAL_SEMANTIC_CONTENT_OBSERVED"
        ),
        claim_scope=(
            "UNAUTHENTICATED_EMPIRICAL_REPLAY_NOT_PROCESS_INDEPENDENCE_AUTHENTICITY_"
            "QUALIFICATION_OR_CROSS_PLATFORM_PROOF"
        ),
        content_integrity_class=INTEGRITY,
    )


class SchemaAndConstructionTests(unittest.TestCase):
    def test_exact_exports_fields_and_validators(self) -> None:
        self.assertEqual(module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(module.__all__, sorted(EXPECTED_EXPORTS))
        self.assertEqual(tuple(item.name for item in dataclasses.fields(CspiceRuntimeObservation)), RUNTIME_FIELDS)
        self.assertEqual(tuple(item.name for item in dataclasses.fields(CspiceSelectedSegmentRecord)), SEGMENT_FIELDS)
        self.assertEqual(tuple(item.name for item in dataclasses.fields(CspiceTargetChainEvidence)), CHAIN_FIELDS)
        self.assertEqual(tuple(item.name for item in dataclasses.fields(CspiceSpkgeoExecutionLane)), LANE_FIELDS)
        self.assertEqual(tuple(item.name for item in dataclasses.fields(CspiceSpkgeoExecutionReceipt)), RECEIPT_FIELDS)
        for function in (
            validate_cspice_runtime_observation,
            validate_cspice_selected_segment_record,
            validate_cspice_target_chain_evidence,
            validate_cspice_spkgeo_execution_lane,
            validate_cspice_spkgeo_execution_receipt,
        ):
            self.assertEqual(tuple(inspect.signature(function).parameters), ("value",))
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_complete_fixture_validates_and_is_frozen(self) -> None:
        value = receipt()
        validate_cspice_spkgeo_execution_receipt(value)
        self.assertEqual(value.primary_lane.semantic_content_sha256, value.replay_lane.semantic_content_sha256)
        self.assertNotEqual(value.primary_lane.content_sha256, value.replay_lane.content_sha256)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.receipt_id = "changed"  # type: ignore[misc]

    def test_new_classes_are_not_accepted_directly_by_frozen_serializer(self) -> None:
        value = receipt()
        for item in (
            value.primary_lane.runtime,
            value.primary_lane.target_chains[0].legs[0],
            value.primary_lane.target_chains[0],
            value.primary_lane,
            value,
        ):
            with self.assertRaises(SolarSystemContractError):
                canonical_json(item)

    def test_fixed_provider_profile_and_effective_et_bits(self) -> None:
        value = receipt().primary_lane
        effective = value.projection.binary64_projection.rounded_value
        self.assertEqual(effective.hex(), "0x1.0000000000000p+0")
        for chain in value.target_chains:
            self.assertEqual(chain.effective_et.hex(), effective.hex())
            self.assertTrue(all(leg.frame_naif_id == 1 and leg.segment_type == 2 for leg in chain.legs))
        self.assertEqual(value.state_batch.position_unit_id, KILOMETRE.unit_id)
        self.assertEqual(value.state_batch.velocity_time_unit_id, SECOND.unit_id)


class RuntimeAndSegmentBoundaryTests(unittest.TestCase):
    def test_runtime_abi_version_cpu_recipe_and_native_roster_are_closed(self) -> None:
        value = receipt().primary_lane.runtime
        for changes in (
            {"python_version": "9.9"},
            {"worker_source_status": "QUALIFIED"},
            {"numpy_cpu_feature_payload_sha256": "1" * 64},
            {"numpy_cpu_feature_payload_status": "OPAQUE"},
            {"distribution_tree_manifest_recipe": "OPAQUE"},
            {"spiceypy_module_relative_path": "spiceypy/other.py"},
            {"spiceypy_record_sha256": "1" * 64},
            {"spiceypy_content_tree_sha256": "1" * 64},
            {"numpy_record_sha256": "1" * 64},
            {"numpy_native_tree_sha256": "1" * 64},
            {"loaded_cspice_sha256": "1" * 64},
            {"cyice_execution_status": "PACKAGED_CYICE_NOT_IMPORTED_OR_MAPPED"},
            {"transitive_native_dependency_custody_status": "LOADED_MAPS_RETAINED"},
            {"loaded_distribution_native_files": tuple(item for item in value.loaded_distribution_native_files if item[0] != NUMPY_ID)},
            {"loaded_distribution_native_files": value.loaded_distribution_native_files + ((SPICEYPY_ID, "spiceypy/cyice.so", 1, "1" * 64),)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(value, **changes, content_sha256="")

    def test_runtime_observation_digests_cannot_use_all_zero_placeholders(self) -> None:
        value = receipt().primary_lane.runtime
        zero = "0" * 64
        for changes in (
            {"python_executable_sha256": zero},
            {"numpy_cpu_feature_payload_sha256": zero},
            {"spiceypy_record_sha256": zero},
            {"spiceypy_content_tree_sha256": zero},
            {"numpy_record_sha256": zero},
            {"numpy_content_tree_sha256": zero},
            {"numpy_native_tree_sha256": zero},
            {"loaded_cspice_sha256": zero},
            {
                "host_native_dependencies": tuple(
                    (name, byte_length, zero)
                    for name, byte_length, _ in value.host_native_dependencies
                )
            },
        ):
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(value, **changes, content_sha256="")

    def test_profile_ids_runtime_pair_and_native_provenance_are_closed(self) -> None:
        invalid_lanes = (
            {"provider_id": "provider.relabelled"},
            {"implementation_id": "implementation.relabelled"},
            {"runtime_id": "runtime.cpython.cross-paired"},
            {"spec_id": "provider-spec.relabelled"},
            {"cspice_artifact_version": "OFFICIAL_UNMODIFIED_N0067"},
        )
        for changes in invalid_lanes:
            with self.subTest(changes=changes):
                with self.assertRaises(SolarSystemContractError):
                    lane("PRIMARY", "f" * 64, **changes)

    def test_both_calibrated_abi_profiles_accept_and_cross_pairing_rejects(self) -> None:
        cp312_lane = lane(
            "PRIMARY",
            "c" * 64,
            abi="cp312-cp312",
        )
        validate_cspice_spkgeo_execution_lane(cp312_lane)
        self.assertEqual(
            cp312_lane.runtime.spiceypy_record_sha256,
            ABI_FIXTURES["cp312-cp312"]["spiceypy_record"],
        )
        self.assertTrue(
            any(
                "cpython-312" in entry[1]
                for entry in cp312_lane.runtime.loaded_distribution_native_files
            )
        )

        cp314_lane = lane("PRIMARY", "d" * 64)
        cp314_provider, _ = context("cp314-cp314")
        with self.assertRaises(SolarSystemContractError):
            runtime(cp314_provider, "cp312-cp312")
        validate_cspice_spkgeo_execution_lane(cp314_lane)

    def test_segment_exact_spiceint_profile_and_closed_endpoints(self) -> None:
        base = receipt().primary_lane.target_chains[0].legs[0]
        for changes in (
            {"kernel_load_index": 1},
            {"selected_body_naif_id": base.center_naif_id},
            {"frame_naif_id": 2},
            {"segment_type": 3},
            {"begin_daf_address": 0},
            {"begin_daf_address": base.end_daf_address + 1},
            {"kernel_artifact_sha256": "0" * 64},
            {"segment_identifier": "X" * 41},
            {"selection_et": math.nextafter(base.last_et, math.inf)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(base, **changes, content_sha256="")
        dataclasses.replace(base, selection_et=base.first_et, content_sha256="").validate_integrity()
        dataclasses.replace(base, selection_et=base.last_et, content_sha256="").validate_integrity()

    def test_chain_signed_zero_first_leg_and_addition(self) -> None:
        lane_value = receipt().primary_lane
        et = lane_value.projection.binary64_projection.rounded_value
        negzeros = (-0.0,) * 6
        leg1 = segment(10, 3, negzeros, et, 400)
        leg2 = segment(3, 0, negzeros, et, 500)
        chain = CspiceTargetChainEvidence(
            target_naif_id=10,
            observer_naif_id=0,
            reference_frame_name="J2000",
            effective_et=et,
            spkgeo_state=negzeros,
            spkgeo_light_time_seconds=0.0,
            legs=(leg1, leg2),
            terminal_center_naif_id=0,
            reconstructed_state=negzeros,
            comparison_recipe=(
                "FIRST_LEG_STATE_THEN_COMPONENTWISE_BINARY64_ADDITION_BODY_TO_CENTER_"
                "FE_TONEAREST"
            ),
            comparison_status="RECONSTRUCTED_AND_SPKGEO_STATES_RAW_BIG_ENDIAN_F8_IDENTICAL",
            evidence_scope=(
                "SPKSFS_SPKUDS_SPKPVN_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_TRACE"
            ),
        )
        self.assertTrue(all(math.copysign(1.0, item) < 0 for item in chain.reconstructed_state))
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(chain, spkgeo_light_time_seconds=-0.0, content_sha256="")

    def test_chain_cannot_reach_observer_then_leave_and_return(self) -> None:
        lane_value = receipt().primary_lane
        et = lane_value.projection.binary64_projection.rounded_value
        zeros = (0.0,) * 6
        reaches_observer = segment(10, 0, zeros, et, 600)
        leaves_observer = segment(0, 3, zeros, et, 700)
        returns_to_observer = segment(3, 0, zeros, et, 800)
        with self.assertRaises(SolarSystemContractError):
            CspiceTargetChainEvidence(
                target_naif_id=10,
                observer_naif_id=0,
                reference_frame_name="J2000",
                effective_et=et,
                spkgeo_state=zeros,
                spkgeo_light_time_seconds=0.0,
                legs=(reaches_observer, leaves_observer, returns_to_observer),
                terminal_center_naif_id=0,
                reconstructed_state=zeros,
                comparison_recipe=(
                    "FIRST_LEG_STATE_THEN_COMPONENTWISE_BINARY64_ADDITION_BODY_TO_"
                    "CENTER_FE_TONEAREST"
                ),
                comparison_status=(
                    "RECONSTRUCTED_AND_SPKGEO_STATES_RAW_BIG_ENDIAN_F8_IDENTICAL"
                ),
                evidence_scope=(
                    "SPKSFS_SPKUDS_SPKPVN_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_TRACE"
                ),
            )


class LaneReplayAndIntegrityTests(unittest.TestCase):
    def test_runtime_and_state_changes_are_semantic_but_local_metadata_is_not(self) -> None:
        primary = lane("PRIMARY", "a" * 64, local_seed=10)
        operationally_different = lane("REPLAY", "b" * 64, local_seed=2_000)
        self.assertEqual(primary.semantic_content_sha256, operationally_different.semantic_content_sha256)
        self.assertNotEqual(primary.content_sha256, operationally_different.content_sha256)
        changed_runtime = dataclasses.replace(
            operationally_different.runtime,
            worker_source_sha256="b" * 64,
            content_sha256="",
        )
        changed_lane = dataclasses.replace(
            operationally_different,
            runtime=changed_runtime,
            semantic_content_sha256="",
            content_sha256="",
        )
        self.assertNotEqual(primary.semantic_content_sha256, changed_lane.semantic_content_sha256)

    def test_semantic_core_binds_chain_evidence_but_excludes_batch_identifier(self) -> None:
        base = lane("PRIMARY", "a" * 64)
        renamed_batch = dataclasses.replace(
            base.state_batch,
            batch_id="batch.cspice.renamed.nonsemantic-container",
            content_sha256="",
        )
        renamed_lane = dataclasses.replace(
            base,
            state_batch=renamed_batch,
            semantic_content_sha256="",
            content_sha256="",
        )
        self.assertEqual(base.semantic_content_sha256, renamed_lane.semantic_content_sha256)
        self.assertNotEqual(base.content_sha256, renamed_lane.content_sha256)

        first_chain = base.target_chains[0]
        changed_light_time = dataclasses.replace(
            first_chain,
            spkgeo_light_time_seconds=math.nextafter(
                first_chain.spkgeo_light_time_seconds,
                math.inf,
            ),
            content_sha256="",
        )
        light_time_lane = dataclasses.replace(
            base,
            target_chains=(changed_light_time, base.target_chains[1]),
            semantic_content_sha256="",
            content_sha256="",
        )
        self.assertNotEqual(base.semantic_content_sha256, light_time_lane.semantic_content_sha256)

        changed_leg = dataclasses.replace(
            first_chain.legs[0],
            segment_identifier="DE-0440LE-0440-REPLAY",
            content_sha256="",
        )
        changed_chain = dataclasses.replace(
            first_chain,
            legs=(changed_leg,),
            content_sha256="",
        )
        segment_lane = dataclasses.replace(
            base,
            target_chains=(changed_chain, base.target_chains[1]),
            semantic_content_sha256="",
            content_sha256="",
        )
        self.assertNotEqual(base.semantic_content_sha256, segment_lane.semantic_content_sha256)

    def test_lane_state_signed_zero_bits_are_semantic_and_replay_binding(self) -> None:
        def with_first_x_zero(
            value: CspiceSpkgeoExecutionLane,
            zero: float,
        ) -> CspiceSpkgeoExecutionLane:
            first_chain = value.target_chains[0]
            first_leg = first_chain.legs[0]
            state = (zero,) + first_chain.spkgeo_state[1:]
            changed_leg = dataclasses.replace(
                first_leg,
                spkpvn_state=state,
                content_sha256="",
            )
            changed_chain = dataclasses.replace(
                first_chain,
                spkgeo_state=state,
                legs=(changed_leg,),
                reconstructed_state=state,
                content_sha256="",
            )
            changed_batch = dataclasses.replace(
                value.state_batch,
                positions=(zero,) + value.state_batch.positions[1:],
                content_sha256="",
            )
            payload = state_payload(changed_batch)
            return dataclasses.replace(
                value,
                state_batch=changed_batch,
                target_chains=(changed_chain, value.target_chains[1]),
                state_payload_sha256=hashlib.sha256(payload).hexdigest(),
                semantic_content_sha256="",
                content_sha256="",
            )

        positive = with_first_x_zero(lane("PRIMARY", "a" * 64), 0.0)
        negative = with_first_x_zero(lane("REPLAY", "b" * 64, local_seed=1_000), -0.0)
        self.assertNotEqual(positive.state_payload_sha256, negative.state_payload_sha256)
        self.assertNotEqual(positive.semantic_content_sha256, negative.semantic_content_sha256)
        self.assertGreater(math.copysign(1.0, positive.state_batch.positions[0]), 0.0)
        self.assertLess(math.copysign(1.0, negative.state_batch.positions[0]), 0.0)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt(),
                primary_lane=positive,
                replay_lane=negative,
                content_sha256="",
            )

    def test_lane_global_segment_selection_and_daf_ranges_are_consistent(self) -> None:
        base = lane("PRIMARY", "a" * 64)
        first_chain, second_chain = base.target_chains
        conflicting_leg = dataclasses.replace(
            second_chain.legs[0],
            begin_daf_address=first_chain.legs[0].begin_daf_address,
            end_daf_address=first_chain.legs[0].end_daf_address,
            content_sha256="",
        )
        conflicting_chain = dataclasses.replace(
            second_chain,
            legs=(conflicting_leg, second_chain.legs[1]),
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                base,
                target_chains=(first_chain, conflicting_chain),
                semantic_content_sha256="",
                content_sha256="",
            )

        overlapping_leg = dataclasses.replace(
            second_chain.legs[0],
            begin_daf_address=150,
            end_daf_address=249,
            content_sha256="",
        )
        overlapping_chain = dataclasses.replace(
            second_chain,
            legs=(overlapping_leg, second_chain.legs[1]),
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                base,
                target_chains=(first_chain, overlapping_chain),
                semantic_content_sha256="",
                content_sha256="",
            )

        shared_center_leg = second_chain.legs[1]
        leading_leg = segment(
            10,
            3,
            (0.0,) * 6,
            first_chain.effective_et,
            400,
        )
        shared_chain = dataclasses.replace(
            first_chain,
            legs=(leading_leg, shared_center_leg),
            content_sha256="",
        )
        shared_lane = dataclasses.replace(
            base,
            target_chains=(shared_chain, second_chain),
            semantic_content_sha256="",
            content_sha256="",
        )
        validate_cspice_spkgeo_execution_lane(shared_lane)

        inconsistent_center_leg = dataclasses.replace(
            shared_center_leg,
            begin_daf_address=500,
            end_daf_address=599,
            content_sha256="",
        )
        inconsistent_chain = dataclasses.replace(
            first_chain,
            legs=(leading_leg, inconsistent_center_leg),
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                base,
                target_chains=(inconsistent_chain, second_chain),
                semantic_content_sha256="",
                content_sha256="",
            )

    def test_outer_requires_roles_distinct_nonce_projection_and_semantic_match(self) -> None:
        value = receipt()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                value,
                replay_lane=dataclasses.replace(
                    value.replay_lane,
                    lane_role="PRIMARY",
                    semantic_content_sha256="",
                    content_sha256="",
                ),
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(value, replay_lane=dataclasses.replace(value.replay_lane, execution_instance_nonce=value.primary_lane.execution_instance_nonce, semantic_content_sha256="", content_sha256=""), content_sha256="")
        alternate_projection = project_cspice_binary64_query(
            value.replay_lane.projection.requested_query,
            effective_query_id="query.cspice.effective.alternate-replay",
        )
        alternate_batch = dataclasses.replace(
            value.replay_lane.state_batch,
            query=alternate_projection.effective_query,
            content_sha256="",
        )
        alternate_lane = dataclasses.replace(
            value.replay_lane,
            projection=alternate_projection,
            state_batch=alternate_batch,
            semantic_content_sha256="",
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                value,
                replay_lane=alternate_lane,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(value, replay_lane=dataclasses.replace(value.replay_lane, runtime=dataclasses.replace(value.replay_lane.runtime, worker_source_sha256="b" * 64, content_sha256=""), semantic_content_sha256="", content_sha256=""), content_sha256="")

    def test_state_payload_is_exact_big_endian_and_slice_bound(self) -> None:
        value = receipt().primary_lane
        payload = state_payload(value.state_batch)
        self.assertEqual(len(payload), 96)
        self.assertEqual(hashlib.sha256(payload).hexdigest(), value.state_payload_sha256)
        little = b"".join(
            struct.pack("<6d", *chain.spkgeo_state) for chain in value.target_chains
        )
        self.assertNotEqual(hashlib.sha256(little).hexdigest(), value.state_payload_sha256)
        changed_batch = dataclasses.replace(
            value.state_batch,
            positions=(2.0,) + value.state_batch.positions[1:],
            content_sha256="",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(value, state_batch=changed_batch, state_payload_sha256=hashlib.sha256(state_payload(changed_batch)).hexdigest(), semantic_content_sha256="", content_sha256="")

    def test_stale_children_bool_subtype_and_seals_fail(self) -> None:
        value = receipt()
        stale = dataclasses.replace(value.primary_lane.runtime)
        object.__setattr__(stale, "pointer_bits", 32)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(value.primary_lane, runtime=stale, semantic_content_sha256="", content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(value.primary_lane.runtime, pointer_bits=True, content_sha256="")

        class RuntimeSubtype(CspiceRuntimeObservation):
            pass

        with self.assertRaises((TypeError, SolarSystemContractError)):
            RuntimeSubtype(**{field.name: getattr(value.primary_lane.runtime, field.name) for field in dataclasses.fields(CspiceRuntimeObservation)})

        class EqualityBomb:
            def __eq__(self, other: object) -> bool:
                raise RuntimeError("equality must not be reached")

        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                value.primary_lane,
                kernel_pool_count_field_order=(EqualityBomb(), "SPK"),
                semantic_content_sha256="",
                content_sha256="",
            )

    def test_parent_payloads_bind_child_refs_not_recursive_graphs(self) -> None:
        value = receipt()
        with mock.patch.object(module, "_projection_ref", wraps=module._projection_ref) as projection_ref:
            validate_cspice_spkgeo_execution_receipt(value)
        self.assertGreater(projection_ref.call_count, 0)
        serialized = canonical_json(module._receipt_payload(value))
        self.assertLess(len(serialized), 32_000)

    def test_shallow_and_global_graph_caps_precede_child_validation(self) -> None:
        base = lane("PRIMARY", "a" * 64)
        runtime_value = base.runtime
        oversized_native_roster = runtime_value.loaded_distribution_native_files * 13
        self.assertGreater(len(oversized_native_roster), 64)
        with mock.patch.object(
            module,
            "_provider_identity_ref",
            side_effect=AssertionError("nested provider validation reached"),
        ):
            with self.assertRaises(SolarSystemContractError):
                dataclasses.replace(
                    runtime_value,
                    loaded_distribution_native_files=oversized_native_roster,
                    content_sha256="",
                )

        stale_chain = dataclasses.replace(base.target_chains[0])
        object.__setattr__(stale_chain, "legs", stale_chain.legs * 16)
        oversized_graph = (stale_chain,) * 17
        with mock.patch.object(
            CspiceTargetChainEvidence,
            "validate_integrity",
            side_effect=AssertionError("nested chain validation reached"),
        ):
            with self.assertRaises(SolarSystemContractError):
                dataclasses.replace(
                    base,
                    target_chains=oversized_graph,
                    semantic_content_sha256="",
                    content_sha256="",
                )


class KatAndImportWallTests(unittest.TestCase):
    def test_literal_domains_and_hashes_are_locked(self) -> None:
        value = receipt()
        semantic_payload = module._lane_semantic_payload(value.primary_lane)
        semantic_preimage = (
            module._LANE_SEMANTIC_DOMAIN.encode("ascii")
            + b"\0"
            + module._LANE_SEMANTIC_SCHEMA.encode("ascii")
            + b"\0"
            + canonical_json(semantic_payload)
        )
        self.assertEqual(
            domain_sha256(
                module._LANE_SEMANTIC_DOMAIN,
                module._LANE_SEMANTIC_SCHEMA,
                semantic_payload,
            ),
            value.primary_lane.semantic_content_sha256,
        )
        self.assertEqual(len(semantic_preimage), 4_499)
        self.assertEqual(
            hashlib.sha256(semantic_preimage).hexdigest(),
            "98c039f369cb3c4a8ef89db599adf8079b4b4834a25a30ca4380e5d27d55861d",
        )
        records = (
            (module._RUNTIME_DOMAIN, module._RUNTIME_SCHEMA, module._runtime_payload(value.primary_lane.runtime), value.primary_lane.runtime.content_sha256),
            (module._SEGMENT_DOMAIN, module._SEGMENT_SCHEMA, module._segment_payload(value.primary_lane.target_chains[0].legs[0]), value.primary_lane.target_chains[0].legs[0].content_sha256),
            (module._CHAIN_DOMAIN, module._CHAIN_SCHEMA, module._chain_payload(value.primary_lane.target_chains[0]), value.primary_lane.target_chains[0].content_sha256),
            (module._LANE_DOMAIN, module._LANE_SCHEMA, module._lane_full_payload(value.primary_lane), value.primary_lane.content_sha256),
            (module._RECEIPT_DOMAIN, module._RECEIPT_SCHEMA, module._receipt_payload(value), value.content_sha256),
        )
        observed_kats = []
        for domain, schema, payload, expected in records:
            self.assertEqual(domain_sha256(domain, schema, payload), expected)
            preimage = (
                domain.encode("ascii")
                + b"\0"
                + schema.encode("ascii")
                + b"\0"
                + canonical_json(payload)
            )
            self.assertEqual(hashlib.sha256(preimage).hexdigest(), expected)
            observed_kats.append((len(preimage), expected))
        self.assertEqual(
            tuple(observed_kats),
            (
                (
                    5_844,
                    "0471ffa1959de7a94e37e5df1fa9369ec711294de1c051ceeea116a74690a5ae",
                ),
                (
                    1_193,
                    "f8a84205662e9e7a4c3ebd3d199cf8603d4ab140ab5e4eaa5714621a648df1ac",
                ),
                (
                    1_389,
                    "a9cc13a3446dd48b07670e9d11786f2f317ed7c5e1936839bc7f5234aa67d7a1",
                ),
                (
                    4_359,
                    "a0ac864e72c36f2a7af7d4662928536cbd86819eed020a663b5af2ff175abd7c",
                ),
                (
                    1_099,
                    "69a190bca3ab98064adf715fb4e79737f35d48ad17e8df093ddfda44db94a075",
                ),
            ),
        )

    def test_import_wall_and_nonclaims(self) -> None:
        source_root = str(Path(__file__).resolve().parents[1] / "src")
        code = f"""
import json, sys
sys.path.insert(0, {source_root!r})
before=set(sys.modules)
import jxplanetx.solar_system.cspice_execution_contracts as m
blocked=('numpy','spiceypy','erfa','astropy','jplephem')
print(json.dumps({{'exports':m.__all__,'loaded':[x for x in blocked if x in sys.modules and x not in before]}}))
"""
        completed = subprocess.run(
            [sys.executable, "-I", "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )
        observed = json.loads(completed.stdout)
        self.assertEqual(observed["loaded"], [])
        prose = module.__doc__ or ""
        for phrase in (
            "imports no astronomy provider",
            "do not instrument SPKGEO's internal path",
            "continuous custody",
            "legal",
            "cross-platform",
            "frame transformation",
            "caller-observed worker-source digest",
        ):
            self.assertIn(phrase, prose)


if __name__ == "__main__":
    unittest.main()
