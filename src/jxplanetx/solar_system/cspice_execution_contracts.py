"""Unpublished, nonexecuting contracts for one restricted CSPICE profile.

These records can retain caller-supplied observations from a future worker.
Constructing or validating them imports no astronomy provider, opens no file,
starts no process, and performs no network operation.  They do not prove that
an observation occurred.

The profile is deliberately narrow: SpiceyPy 8.2.0 with its N0067-derived and
patched bundled CSPICE library, one directly furnished SPK, geometric SPKGEO
states at the exact M4C1 effective binary64 TDB ET, observer 0, J2000 frame 1,
type-2 segments, kilometres, and kilometres per defined SI second.  The M1
frame strings acquire meaning only through this closed crosswalk; they are not
an independent ICRF identity or transformation proof.

Segment-chain records are post-query corroboration.  SPKSFS, SPKUDS, and
SPKPVN do not instrument SPKGEO's internal path.  Coarse coverage does not
prove target availability.  Local pre/post observations do not prove an
atomic snapshot, continuous custody, immutable mapped pages, protection from
mutate-and-restore races, or that CSPICE consumed the observed bytes.
The loaded-map roster is a caller assertion without a retained ``dladdr`` or
``/proc`` correlation receipt, and the network/fallback/extrapolation fields
are likewise declared observations rather than syscall or sandbox evidence.

All records are caller-supplied, forgeable, unauthenticated integrity
envelopes.  They provide no authenticity, signature, legal, redistribution,
registry, qualification, reproducible-build, hermetic-process, independent
replay, or cross-platform guarantee.  They do not establish apparent-state or
aberration semantics, a frame transformation, state conversion, physical
accuracy, enforced network denial, or enforced fallback/extrapolation policy.
The fixed wheel profiles constrain only the retained distribution identities;
the caller-observed worker-source digest is recorded but is not a qualified
worker implementation until a later executable-worker milestone.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, fields
from fractions import Fraction
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Callable, ClassVar

from .artifacts import (
    ArtifactByteMatchReceipt,
    LocalArtifactVerificationReceipt,
    validate_local_artifact_verification_receipt,
)
from .contracts import (
    MAXIMUM_ARTIFACT_COUNT,
    MAXIMUM_BODY_COUNT,
    MAXIMUM_IDENTIFIER_CODEPOINTS,
    MAXIMUM_LOGICAL_LOCATOR_CODEPOINTS,
    MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
    ProviderIdentity,
    SolarSystemContractError,
    validate_integrity,
)
from .cspice_time import (
    CspiceBinary64QueryProjectionReceipt,
    validate_cspice_binary64_query_projection_receipt,
)
from .serialization import (
    MAXIMUM_CANONICAL_BYTES,
    MAXIMUM_CANONICAL_EXACT_INTEGER_BITS,
    MAXIMUM_CANONICAL_STRING_CODEPOINTS,
    domain_sha256,
    validate_sha256,
)
from .states import ProviderNativeStateBatch, validate_provider_native_state_batch
from .units import KILOMETRE, SECOND, _round_fraction_to_binary64


_CONTENT_INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
_MAXIMUM_CHAIN_LENGTH = 16
_MAXIMUM_TOTAL_SEGMENT_REFERENCES = 256
_MAXIMUM_LOADED_NATIVE_FILES = 64
_MAXIMUM_HOST_NATIVE_FILES = 16
_MAXIMUM_MANIFEST_FILES = 4_096
_MAXIMUM_NATIVE_TREE_FILES = 256
_MAXIMUM_RUNTIME_FILE_BYTES = 1 << 30
_MAXIMUM_CPU_FEATURE_PAYLOAD_BYTES = 1 << 20
_MAXIMUM_SEGMENT_IDENTIFIER_CODEPOINTS = 40
_MAXIMUM_STATE_PAYLOAD_BYTES = MAXIMUM_BODY_COUNT * 48
_MAXIMUM_SPICE_INT = (1 << 31) - 1

_PROVIDER_PROFILE = (
    "SPICEYPY_8_2_0_N0067_DERIVED_CSPICE_SPKGEO_J2000_TYPE2_KM_PER_S_V1"
)
_PROVIDER_ID = "provider.spiceypy-8.2.0.cspice-n0067-derived.v1"
_IMPLEMENTATION_ID = "implementation.spiceypy-8.2.0.cspice-n0067-derived.v1"
_PROVIDER_SPEC_ID = (
    "provider-spec.spiceypy-8.2.0.cspice-n0067-derived.single-spk-j2000.v1"
)
_FRAME_ID = "frame.naif-j2000.ssb.tdb-compatible.v1"
_AXES_ID = "NAIF_BUILTIN_INERTIAL_FRAME_1_J2000"
_ORIENTATION_ID = "NAIF_BUILTIN_J2000_STATIC_ORIENTATION"
_ORIGIN_REALIZATION_ID = "NAIF_BODY_0_SOLAR_SYSTEM_BARYCENTER"

_RUNTIME_DOMAIN = (
    "jxplanetx.solar-system.cspice-execution-contracts.runtime-observation."
    "content-integrity.v1"
)
_RUNTIME_SCHEMA = "CspiceRuntimeObservation.v1"
_RUNTIME_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_execution_contracts.CspiceRuntimeObservation"
)
_SEGMENT_DOMAIN = (
    "jxplanetx.solar-system.cspice-execution-contracts.selected-segment-record."
    "content-integrity.v1"
)
_SEGMENT_SCHEMA = "CspiceSelectedSegmentRecord.v1"
_SEGMENT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_execution_contracts.CspiceSelectedSegmentRecord"
)
_CHAIN_DOMAIN = (
    "jxplanetx.solar-system.cspice-execution-contracts.target-chain-evidence."
    "content-integrity.v1"
)
_CHAIN_SCHEMA = "CspiceTargetChainEvidence.v1"
_CHAIN_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_execution_contracts.CspiceTargetChainEvidence"
)
_LANE_DOMAIN = (
    "jxplanetx.solar-system.cspice-execution-contracts.spkgeo-execution-lane."
    "content-integrity.v1"
)
_LANE_SCHEMA = "CspiceSpkgeoExecutionLane.v1"
_LANE_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_execution_contracts.CspiceSpkgeoExecutionLane"
)
_LANE_SEMANTIC_DOMAIN = (
    "jxplanetx.solar-system.cspice-execution-contracts.spkgeo-execution-lane."
    "semantic-result.v1"
)
_LANE_SEMANTIC_SCHEMA = "CspiceSpkgeoExecutionLane.semantic-result.v1"
_RECEIPT_DOMAIN = (
    "jxplanetx.solar-system.cspice-execution-contracts.spkgeo-execution-receipt."
    "content-integrity.v1"
)
_RECEIPT_SCHEMA = "CspiceSpkgeoExecutionReceipt.v1"
_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_execution_contracts.CspiceSpkgeoExecutionReceipt"
)

_PROVIDER_IDENTITY_SCHEMA = "jxplanetx.solar-system.provider-identity.payload.v1"
_PROVIDER_IDENTITY_QUALIFIED_NAME = (
    "jxplanetx.solar_system.contracts.ProviderIdentity"
)
_PROJECTION_SCHEMA = "CspiceBinary64QueryProjectionReceipt.v1"
_PROJECTION_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_time.CspiceBinary64QueryProjectionReceipt"
)
_STATE_BATCH_SCHEMA = "ProviderNativeStateBatch.v1"
_STATE_BATCH_QUALIFIED_NAME = (
    "jxplanetx.solar_system.states.ProviderNativeStateBatch"
)
_LOCAL_RECEIPT_SCHEMA = "LocalArtifactVerificationReceipt.v1"
_LOCAL_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.artifacts.LocalArtifactVerificationReceipt"
)
_BYTE_MATCH_SCHEMA = "ArtifactByteMatchReceipt.v1"
_BYTE_MATCH_QUALIFIED_NAME = (
    "jxplanetx.solar_system.artifacts.ArtifactByteMatchReceipt"
)

_RUNTIME_FLOATING_POINT_STATUS = "FE_TONEAREST_BEFORE_AND_AFTER_PROVIDER_CALLS"
_RUNTIME_PYTHON_INVOCATION_STATUS = (
    "CPYTHON_ISOLATED_MODE_IGNORE_PYTHON_ENVIRONMENT_AND_NO_BYTECODE_WRITES"
)
_RUNTIME_ENVIRONMENT_STATUS = (
    "MINIMAL_C_UTF8_ENVIRONMENT_CSPICE_AND_DYNAMIC_LOADER_OVERRIDES_ABSENT_"
    "OPENBLAS_OMP_MKL_AND_NUMEXPR_THREADS_ONE"
)
_RUNTIME_CPU_FEATURE_STATUS = (
    "SORTED_COMPACT_UTF8_JSON_OF_NUMPY_CPU_BASELINE_DISPATCH_AND_FEATURES"
)
_RUNTIME_DISTRIBUTION_MANIFEST_RECIPE = (
    "RECORD_ROWS_SORTED_BY_UTF8_RELATIVE_PATH_PATH_NUL_DECIMAL_SIZE_NUL_"
    "LOWERCASE_SHA256_LF"
)
_RUNTIME_NATIVE_MANIFEST_RECIPE = (
    "RECORD_PATH_SUFFIX_SO_DLL_DYLIB_OR_BASENAME_CONTAINS_DOT_SO_DOT_THEN_"
    "DISTRIBUTION_MANIFEST_RECIPE"
)
_RUNTIME_RECORD_STATUS = (
    "WHEEL_RECORD_VALIDATED_ALL_HASHED_MEMBERS_CLOSED_EXTRACTION_NO_SYMLINKS_NO_PYC"
)
_RUNTIME_MODULE_ORIGIN_STATUS = (
    "MODULE_ORIGIN_WITHIN_EXACT_VALIDATED_EXTRACTION"
)
_RUNTIME_LOADER_STATUS = (
    "LIBSPICEHELPER_BUNDLED_RELATIVE_PATH_AND_FILE_SHA256_MATCHED_NO_OVERRIDE_"
    "OR_SYSTEM_FALLBACK"
)
_RUNTIME_CYICE_STATUS = (
    "CALLER_ASSERTED_PACKAGED_CYICE_NOT_IMPORTED_OR_MAPPED_NO_RETAINED_"
    "IMPORT_OR_MAP_CORRELATION_RECEIPT"
)
_RUNTIME_LOADED_MAP_STATUS = (
    "CALLER_ASSERTED_POINT_OBSERVED_FILE_BACKED_MAP_ROSTER_NO_RETAINED_"
    "CORRELATION_RECEIPT_OR_MAPPED_PAGE_CUSTODY"
)
_RUNTIME_DEPENDENCY_CUSTODY_STATUS = (
    "CALLER_ASSERTED_DISTRIBUTION_NATIVE_AND_HOST_LIBC_LIBM_LOADER_ROSTERS_"
    "NO_RETAINED_MAP_CORRELATION_OR_COMPLETE_HOST_SBOM"
)
_RUNTIME_SCOPE = (
    "HOST_LOCAL_EMPIRICAL_LINUX_X86_64_GLIBC_CPU_FEATURE_BOUND_ONLY"
)
_RUNTIME_WORKER_SOURCE_STATUS = (
    "CALLER_OBSERVED_SOURCE_DIGEST_NOT_QUALIFIED_UNTIL_M4C2B"
)

_COMMON_CSPICE_LENGTH = 3_561_056
_COMMON_CSPICE_ARTIFACT_VERSION = (
    "SpiceyPy-8.2.0-bundled-N0067-derived-patched"
)
_COMMON_CSPICE_SHA256 = (
    "1d9273fc9afce5201e904569a9439a0b010d2461dfc88c0de549d2a1b5ecdf84"
)
_COMMON_CPU_FEATURE_PAYLOAD_LENGTH = 1_086
_COMMON_CPU_FEATURE_PAYLOAD_SHA256 = (
    "b050bd98f1d010d3f10c7d685e2eb6875793ddb9909110de8474aba96740fe74"
)

_ABI_PROFILES = MappingProxyType(
    {
        "cp312-cp312": MappingProxyType(
            {
                "python_version": "3.12.13",
                "runtime_id": (
                    "runtime.cpython-3.12.13.cp312-linux-x86_64."
                    "host-local-observed.v1"
                ),
                "spiceypy_wheel_length": 2_430_343,
                "spiceypy_wheel_sha256": (
                    "c91643e43d0b1d0de5616ffd9a2a82a7e2ae7ea50c1bf5f740e89ed1e67d4996"
                ),
                "spiceypy_record_sha256": (
                    "da050a7e1cb4abbea030a804090f40ca7ec3283a8b059478fcd5d4754a515d47"
                ),
                "spiceypy_tree": (
                    29,
                    7_197_075,
                    2_950,
                    "1a03ac00e4816b237c93d1d0643494f50f0d04a9df2a253cdf211b519d6bd4ab",
                ),
                "numpy_wheel_length": 16_606_086,
                "numpy_wheel_sha256": (
                    "0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28"
                ),
                "numpy_record_sha256": (
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
            }
        ),
        "cp314-cp314": MappingProxyType(
            {
                "python_version": "3.14.4",
                "runtime_id": (
                    "runtime.cpython-3.14.4.cp314-linux-x86_64."
                    "host-local-observed.v1"
                ),
                "spiceypy_wheel_length": 2_442_346,
                "spiceypy_wheel_sha256": (
                    "00c2714d3dd67180794ad0b66735d6cac470e19627d6e99f9628a50b52133d90"
                ),
                "spiceypy_record_sha256": (
                    "ae6ccb8132cec7e02cf020785df7e6c2b36c344eb215253757f8dcb100ee1002"
                ),
                "spiceypy_tree": (
                    29,
                    7_209_395,
                    2_950,
                    "f8f65e07e738b4beab2a277bac53ced6338ce9ed001f9c84930e1b54988700db",
                ),
                "numpy_wheel_length": 16_597_350,
                "numpy_wheel_sha256": (
                    "fffe29a1ef00883599d1dc2c51aa2e5d80afe49523c261a74933df395c15c520"
                ),
                "numpy_record_sha256": (
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
            }
        ),
    }
)

_SEGMENT_SELECTION_STATUS = (
    "SPKSFS_IDLEN_41_SELECTED_SPKUDS_DECODED_SPKPVN_EVALUATED"
)
_SEGMENT_EVIDENCE_SCOPE = (
    "SELECTED_SEGMENT_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_INTERNAL_TRACE"
)
_CHAIN_COMPARISON_RECIPE = (
    "FIRST_LEG_STATE_THEN_COMPONENTWISE_BINARY64_ADDITION_BODY_TO_CENTER_"
    "FE_TONEAREST"
)
_CHAIN_COMPARISON_STATUS = (
    "RECONSTRUCTED_AND_SPKGEO_STATES_RAW_BIG_ENDIAN_F8_IDENTICAL"
)
_CHAIN_EVIDENCE_SCOPE = (
    "SPKSFS_SPKUDS_SPKPVN_CORROBORATION_NOT_INSTRUMENTED_SPKGEO_TRACE"
)

_LANE_ROLES = frozenset(("PRIMARY", "REPLAY"))
_LANE_INSTANCE_STATUS = (
    "CALLER_SUPPLIED_NONSECRET_UNAUTHENTICATED_INSTANCE_NONCE_NO_PROCESS_"
    "IDENTITY_PROOF"
)
_LANE_POOL_STATUS = (
    "EMPTY_THEN_ONE_DIRECT_SPK_LOAD_STABLE_THROUGH_QUERY_THEN_CLEARED"
)
_LANE_ERROR_STATUS = (
    "FRESH_ERROR_STATE_CLEAR_AND_NO_FAILED_STATE_AFTER_THIS_LANE_PROVIDER_"
    "AND_SELECTOR_CALLS"
)
_LANE_PROVIDER_CALL = (
    "CSPICE_SPKGEO_TARGET_EFFECTIVE_BINARY64_ET_J2000_OBSERVER_0_GEOMETRIC_"
    "NO_ABERRATION"
)
_LANE_STATE_PAYLOAD_RECIPE = (
    "TARGET_ORDER_X_Y_Z_VX_VY_VZ_BIG_ENDIAN_BINARY64"
)
_LANE_EXECUTION_STATUS = (
    "CSPICE_SPKGEO_GEOMETRIC_STATE_AT_M4C1_EFFECTIVE_BINARY64_ET_NO_JX_CONVERSION"
)
_LANE_NESTED_BATCH_STATUS = (
    "NESTED_BATCH_VALUE_CONTAINER_OUTER_RECEIPT_SUPPLIES_DECLARED_EXECUTION_EVIDENCE"
)
_LANE_ATOMICITY_STATUS = (
    "NO_ATOMIC_SNAPSHOT_CONTINUOUS_CUSTODY_OR_MUTATE_RESTORE_PROOF"
)
_LANE_ARTIFACT_CUSTODY_STATUS = (
    "PRE_AND_POST_POINT_OBSERVATIONS_ONLY_NO_PROVIDER_MAPPED_PAGE_AT_USE_PROOF"
)
_LANE_PROCESS_STATUS = (
    "FRESH_PROCESS_PER_LANE_CALLER_ASSERTED_NO_HERMETIC_SANDBOX_OR_"
    "INDEPENDENCE_PROOF"
)
_LANE_NETWORK_STATUS = (
    "CALLER_ASSERTED_NO_NETWORK_AVAILABLE_OR_USED_NO_SYSCALL_OR_SANDBOX_"
    "ATTESTATION"
)
_LANE_FALLBACK_STATUS = (
    "DECLARED_FIXED_PROFILE_NO_FALLBACK_PATH_USED_NOT_INDEPENDENTLY_ENFORCED"
)
_LANE_EXTRAPOLATION_STATUS = (
    "DECLARED_FIXED_PROFILE_NO_EXTRAPOLATION_PATH_USED_NOT_INDEPENDENTLY_ENFORCED"
)
_LANE_PORTABILITY_SCOPE = "HOST_LOCAL_EMPIRICAL_PROFILE_ONLY"

_REPLAY_COMPARISON_SCOPE = (
    "DELIBERATE_RESULT_CORE_INCLUDES_EXACT_RUNTIME_EXCLUDES_ROLE_INSTANCE_"
    "LOCAL_FILESYSTEM_METADATA_AND_BATCH_IDENTIFIER"
)
_REPLAY_STATUS = (
    "DISTINCT_CALLER_INSTANCE_NONCES_AND_EQUAL_SEMANTIC_CONTENT_OBSERVED"
)
_REPLAY_CLAIM_SCOPE = (
    "UNAUTHENTICATED_EMPIRICAL_REPLAY_NOT_PROCESS_INDEPENDENCE_AUTHENTICITY_"
    "QUALIFICATION_OR_CROSS_PLATFORM_PROOF"
)


def _preflight_optional_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if value != "":
        validate_sha256(value, label)
    return value


def _nonzero_sha256(value: object, label: str) -> str:
    digest = validate_sha256(value, label)
    if digest == "0" * 64:
        raise SolarSystemContractError(f"{label} must be a nonzero digest")
    return digest


def _text(
    value: object,
    label: str,
    *,
    maximum_codepoints: int = MAXIMUM_IDENTIFIER_CODEPOINTS,
) -> str:
    if type(value) is not str:
        raise SolarSystemContractError(f"{label} must be an exact string")
    if len(value) > maximum_codepoints:
        raise SolarSystemContractError(f"{label} exceeds its code-point cap")
    if (
        not value
        or value.strip() != value
        or any(ord(character) < 0x20 or ord(character) > 0x7E for character in value)
    ):
        raise SolarSystemContractError(
            f"{label} must be nonempty trimmed printable ASCII text"
        )
    return value


def _token(value: object, expected: str, label: str) -> str:
    result = _text(
        value,
        label,
        maximum_codepoints=MAXIMUM_CANONICAL_STRING_CODEPOINTS,
    )
    if result != expected:
        raise SolarSystemContractError(f"{label} is not the exact required token")
    return result


def _integer(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        raise SolarSystemContractError(f"{label} must be an exact integer")
    if value.bit_length() > MAXIMUM_CANONICAL_EXACT_INTEGER_BITS:
        raise SolarSystemContractError(f"{label} exceeds the exact integer bit cap")
    if minimum is not None and value < minimum:
        raise SolarSystemContractError(f"{label} is below its minimum")
    if maximum is not None and value > maximum:
        raise SolarSystemContractError(f"{label} exceeds its maximum")
    return value


def _finite_float(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise SolarSystemContractError(f"{label} must be an exact finite binary64 value")
    return value


def _float_bits_equal(left: float, right: float) -> bool:
    return left.hex() == right.hex()


def _exact_float_tuple(value: object, length: int, label: str) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise SolarSystemContractError(f"{label} must be an exact tuple")
    if len(value) != length:
        raise SolarSystemContractError(f"{label} must contain exactly {length} items")
    return tuple(
        _finite_float(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )


def _normalized_relative_path(value: object, label: str) -> str:
    path_text = _text(
        value,
        label,
        maximum_codepoints=MAXIMUM_LOGICAL_LOCATOR_CODEPOINTS,
    )
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or str(path) != path_text
        or len(path.parts) > 64
        or any(
            part in ("", ".", "..") or len(part) > MAXIMUM_IDENTIFIER_CODEPOINTS
            for part in path.parts
        )
    ):
        raise SolarSystemContractError(f"{label} is not a normalized relative path")
    return path_text


def _child_ref(
    value: object,
    expected_type: type[object],
    validator: Callable[[object], None],
    qualified_name: str,
    schema: str,
    label: str,
) -> tuple[str, str, str]:
    if type(value) is not expected_type:
        raise SolarSystemContractError(
            f"{label} must have exact type {qualified_name}"
        )
    validator(value)
    digest = validate_sha256(getattr(value, "content_sha256"), f"{label}.content_sha256")
    return (qualified_name, schema, digest)


def _provider_identity_ref(value: ProviderIdentity) -> tuple[str, str, str]:
    return _child_ref(
        value,
        ProviderIdentity,
        validate_integrity,
        _PROVIDER_IDENTITY_QUALIFIED_NAME,
        _PROVIDER_IDENTITY_SCHEMA,
        "provider_identity",
    )


def _projection_ref(
    value: CspiceBinary64QueryProjectionReceipt,
) -> tuple[str, str, str]:
    return _child_ref(
        value,
        CspiceBinary64QueryProjectionReceipt,
        validate_cspice_binary64_query_projection_receipt,
        _PROJECTION_QUALIFIED_NAME,
        _PROJECTION_SCHEMA,
        "projection",
    )


def _state_batch_ref(value: ProviderNativeStateBatch) -> tuple[str, str, str]:
    return _child_ref(
        value,
        ProviderNativeStateBatch,
        validate_provider_native_state_batch,
        _STATE_BATCH_QUALIFIED_NAME,
        _STATE_BATCH_SCHEMA,
        "state_batch",
    )


def _local_receipt_ref(
    value: LocalArtifactVerificationReceipt,
    label: str,
) -> tuple[str, str, str]:
    return _child_ref(
        value,
        LocalArtifactVerificationReceipt,
        validate_local_artifact_verification_receipt,
        _LOCAL_RECEIPT_QUALIFIED_NAME,
        _LOCAL_RECEIPT_SCHEMA,
        label,
    )


def _byte_match_ref(
    value: ArtifactByteMatchReceipt,
    label: str,
) -> tuple[str, str, str]:
    if type(value) is not ArtifactByteMatchReceipt:
        raise SolarSystemContractError(
            f"{label} must be an exact ArtifactByteMatchReceipt"
        )
    value.validate_integrity()
    return (
        _BYTE_MATCH_QUALIFIED_NAME,
        _BYTE_MATCH_SCHEMA,
        validate_sha256(value.content_sha256, f"{label}.content_sha256"),
    )


def _record_payload(
    value: object,
    qualified_name: str,
    replacements: dict[str, object] | None = None,
) -> tuple[object, ...]:
    selected = replacements or {}
    return (
        qualified_name,
        tuple(
            (
                descriptor.name,
                selected.get(descriptor.name, getattr(value, descriptor.name)),
            )
            for descriptor in fields(value)
            if descriptor.name != "content_sha256"
        ),
    )


def _digest(domain: str, schema: str, payload: object) -> str:
    return domain_sha256(domain, schema, payload)


def _runtime_payload(value: CspiceRuntimeObservation) -> tuple[object, ...]:
    return _record_payload(
        value,
        _RUNTIME_QUALIFIED_NAME,
        {"provider_identity": _provider_identity_ref(value.provider_identity)},
    )


def _runtime_digest(value: CspiceRuntimeObservation) -> str:
    return _digest(_RUNTIME_DOMAIN, _RUNTIME_SCHEMA, _runtime_payload(value))


@dataclass(frozen=True, slots=True, eq=False)
class CspiceRuntimeObservation:
    """Caller-supplied identity and numerical-runtime observation."""

    provider_identity: ProviderIdentity
    worker_source_sha256: str
    worker_source_status: str
    python_implementation: str
    python_version: str
    python_wheel_interpreter_abi_tag: str
    python_executable_byte_length: int
    python_executable_sha256: str
    sys_platform: str
    machine: str
    byteorder: str
    pointer_bits: int
    libc_name: str
    libc_version: str
    binary64_format: str
    fegetround_before: int
    fegetround_after: int
    floating_point_environment_status: str
    python_invocation_status: str
    execution_environment_status: str
    numpy_cpu_feature_payload_byte_length: int
    numpy_cpu_feature_payload_sha256: str
    numpy_cpu_feature_payload_status: str
    distribution_tree_manifest_recipe: str
    numpy_native_tree_selection_and_manifest_recipe: str
    spiceypy_distribution_artifact_id: str
    spiceypy_version: str
    spiceypy_module_relative_path: str
    spiceypy_record_sha256: str
    spiceypy_content_tree_file_count: int
    spiceypy_content_tree_total_byte_length: int
    spiceypy_content_tree_manifest_byte_length: int
    spiceypy_content_tree_sha256: str
    spiceypy_record_validation_status: str
    spiceypy_loaded_module_origin_status: str
    numpy_distribution_artifact_id: str
    numpy_version: str
    numpy_module_relative_path: str
    numpy_record_sha256: str
    numpy_content_tree_file_count: int
    numpy_content_tree_total_byte_length: int
    numpy_content_tree_manifest_byte_length: int
    numpy_content_tree_sha256: str
    numpy_native_tree_file_count: int
    numpy_native_tree_total_byte_length: int
    numpy_native_tree_manifest_byte_length: int
    numpy_native_tree_sha256: str
    numpy_record_validation_status: str
    numpy_loaded_module_origin_status: str
    loaded_cspice_artifact_id: str
    loaded_cspice_relative_path: str
    loaded_cspice_byte_length: int
    loaded_cspice_sha256: str
    cspice_toolkit_version: str
    cspice_loader_resolution_status: str
    cyice_execution_status: str
    loaded_distribution_native_files: tuple[tuple[str, str, int, str], ...]
    loaded_native_map_observation_status: str
    host_native_dependencies: tuple[tuple[str, int, str], ...]
    transitive_native_dependency_custody_status: str
    runtime_scope: str
    content_integrity_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _RUNTIME_DOMAIN
    _SCHEMA: ClassVar[str] = _RUNTIME_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not CspiceRuntimeObservation:
            raise SolarSystemContractError(
                "runtime observation must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _runtime_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact runtime observation"
            )

    def _validate(self) -> None:
        if type(self) is not CspiceRuntimeObservation:
            raise SolarSystemContractError(
                "runtime observation must have its exact concrete type"
            )
        if (
            type(self.loaded_distribution_native_files) is not tuple
            or not self.loaded_distribution_native_files
            or len(self.loaded_distribution_native_files) > _MAXIMUM_LOADED_NATIVE_FILES
        ):
            raise SolarSystemContractError(
                "loaded_distribution_native_files is outside its shallow count cap"
            )
        if (
            type(self.host_native_dependencies) is not tuple
            or not self.host_native_dependencies
            or len(self.host_native_dependencies) > _MAXIMUM_HOST_NATIVE_FILES
        ):
            raise SolarSystemContractError(
                "host_native_dependencies is outside its shallow count cap"
            )
        identity = self.provider_identity
        _provider_identity_ref(identity)
        if (
            identity.provider_id != _PROVIDER_ID
            or identity.implementation_id != _IMPLEMENTATION_ID
            or identity.provider_kind != "LOCAL_OFFLINE_SPK"
            or identity.version != "8.2.0"
            or identity.module_name != "spiceypy"
            or identity.distribution_name != "spiceypy"
            or identity.distribution_version != "8.2.0"
        ):
            raise SolarSystemContractError(
                "provider_identity is outside the exact SpiceyPy 8.2.0 profile"
            )
        _preflight_optional_sha256(self.worker_source_sha256, "worker_source_sha256")
        if self.worker_source_sha256 == "" or self.worker_source_sha256 == "0" * 64:
            raise SolarSystemContractError("worker_source_sha256 must be a nonzero digest")
        _token(
            self.worker_source_status,
            _RUNTIME_WORKER_SOURCE_STATUS,
            "worker_source_status",
        )
        _token(self.python_implementation, "CPython", "python_implementation")
        python_version = _text(
            self.python_version,
            "python_version",
            maximum_codepoints=64,
        )
        abi = _text(
            self.python_wheel_interpreter_abi_tag,
            "python_wheel_interpreter_abi_tag",
            maximum_codepoints=32,
        )
        profile = _ABI_PROFILES.get(abi)
        if profile is None:
            raise SolarSystemContractError(
                "python_wheel_interpreter_abi_tag is outside the exact wheel-tag roster"
            )
        expected_version = profile["python_version"]
        if python_version != expected_version:
            raise SolarSystemContractError(
                "python_version does not match the exact supported ABI runtime"
            )
        if identity.runtime_id != profile["runtime_id"]:
            raise SolarSystemContractError(
                "provider runtime_id does not match the exact supported ABI runtime"
            )
        _integer(
            self.python_executable_byte_length,
            "python_executable_byte_length",
            minimum=1,
            maximum=_MAXIMUM_RUNTIME_FILE_BYTES,
        )
        _nonzero_sha256(self.python_executable_sha256, "python_executable_sha256")
        _token(self.sys_platform, "linux", "sys_platform")
        _token(self.machine, "x86_64", "machine")
        _token(self.byteorder, "little", "byteorder")
        _integer(self.pointer_bits, "pointer_bits", minimum=64, maximum=64)
        _token(self.libc_name, "glibc", "libc_name")
        _text(self.libc_version, "libc_version", maximum_codepoints=64)
        _token(self.binary64_format, "IEEE, little-endian", "binary64_format")
        _integer(self.fegetround_before, "fegetround_before", minimum=0, maximum=0)
        _integer(self.fegetround_after, "fegetround_after", minimum=0, maximum=0)
        _token(
            self.floating_point_environment_status,
            _RUNTIME_FLOATING_POINT_STATUS,
            "floating_point_environment_status",
        )
        _token(
            self.python_invocation_status,
            _RUNTIME_PYTHON_INVOCATION_STATUS,
            "python_invocation_status",
        )
        _token(
            self.execution_environment_status,
            _RUNTIME_ENVIRONMENT_STATUS,
            "execution_environment_status",
        )
        cpu_payload_length = _integer(
            self.numpy_cpu_feature_payload_byte_length,
            "numpy_cpu_feature_payload_byte_length",
            minimum=1,
            maximum=_MAXIMUM_CPU_FEATURE_PAYLOAD_BYTES,
        )
        cpu_payload_sha256 = _nonzero_sha256(
            self.numpy_cpu_feature_payload_sha256,
            "numpy_cpu_feature_payload_sha256",
        )
        if (
            cpu_payload_length != _COMMON_CPU_FEATURE_PAYLOAD_LENGTH
            or cpu_payload_sha256 != _COMMON_CPU_FEATURE_PAYLOAD_SHA256
        ):
            raise SolarSystemContractError(
                "NumPy CPU feature payload is outside the calibrated exact profile"
            )
        _token(
            self.numpy_cpu_feature_payload_status,
            _RUNTIME_CPU_FEATURE_STATUS,
            "numpy_cpu_feature_payload_status",
        )
        _token(
            self.distribution_tree_manifest_recipe,
            _RUNTIME_DISTRIBUTION_MANIFEST_RECIPE,
            "distribution_tree_manifest_recipe",
        )
        _token(
            self.numpy_native_tree_selection_and_manifest_recipe,
            _RUNTIME_NATIVE_MANIFEST_RECIPE,
            "numpy_native_tree_selection_and_manifest_recipe",
        )
        implementation_ids = identity.ordered_implementation_artifact_ids
        spiceypy_id = _text(
            self.spiceypy_distribution_artifact_id,
            "spiceypy_distribution_artifact_id",
        )
        numpy_id = _text(
            self.numpy_distribution_artifact_id,
            "numpy_distribution_artifact_id",
        )
        cspice_id = _text(
            self.loaded_cspice_artifact_id,
            "loaded_cspice_artifact_id",
        )
        if len({spiceypy_id, numpy_id, cspice_id}) != 3:
            raise SolarSystemContractError(
                "runtime distribution/native artifact IDs must be distinct"
            )
        if any(item not in implementation_ids for item in (spiceypy_id, numpy_id, cspice_id)):
            raise SolarSystemContractError(
                "runtime distribution/native artifacts are absent from provider identity"
            )
        if implementation_ids != (spiceypy_id, numpy_id, cspice_id):
            raise SolarSystemContractError(
                "provider implementation roster must be exactly SpiceyPy, NumPy, CSPICE"
            )
        _token(self.spiceypy_version, "8.2.0", "spiceypy_version")
        spiceypy_module_path = _normalized_relative_path(
            self.spiceypy_module_relative_path,
            "spiceypy_module_relative_path",
        )
        if spiceypy_module_path != "spiceypy/__init__.py":
            raise SolarSystemContractError(
                "SpiceyPy module path is outside the calibrated exact profile"
            )
        spiceypy_record_sha256 = _nonzero_sha256(
            self.spiceypy_record_sha256,
            "spiceypy_record_sha256",
        )
        self._validate_manifest(
            "spiceypy_content_tree",
            self.spiceypy_content_tree_file_count,
            self.spiceypy_content_tree_total_byte_length,
            self.spiceypy_content_tree_manifest_byte_length,
            self.spiceypy_content_tree_sha256,
            _MAXIMUM_MANIFEST_FILES,
        )
        if (
            spiceypy_record_sha256 != profile["spiceypy_record_sha256"]
            or (
                self.spiceypy_content_tree_file_count,
                self.spiceypy_content_tree_total_byte_length,
                self.spiceypy_content_tree_manifest_byte_length,
                self.spiceypy_content_tree_sha256,
            )
            != profile["spiceypy_tree"]
        ):
            raise SolarSystemContractError(
                "SpiceyPy RECORD/tree identity is outside the calibrated ABI profile"
            )
        _token(
            self.spiceypy_record_validation_status,
            _RUNTIME_RECORD_STATUS,
            "spiceypy_record_validation_status",
        )
        _token(
            self.spiceypy_loaded_module_origin_status,
            _RUNTIME_MODULE_ORIGIN_STATUS,
            "spiceypy_loaded_module_origin_status",
        )
        _token(self.numpy_version, "2.3.5", "numpy_version")
        numpy_module_path = _normalized_relative_path(
            self.numpy_module_relative_path,
            "numpy_module_relative_path",
        )
        if numpy_module_path != "numpy/__init__.py":
            raise SolarSystemContractError(
                "NumPy module path is outside the calibrated exact profile"
            )
        numpy_record_sha256 = _nonzero_sha256(
            self.numpy_record_sha256,
            "numpy_record_sha256",
        )
        self._validate_manifest(
            "numpy_content_tree",
            self.numpy_content_tree_file_count,
            self.numpy_content_tree_total_byte_length,
            self.numpy_content_tree_manifest_byte_length,
            self.numpy_content_tree_sha256,
            _MAXIMUM_MANIFEST_FILES,
        )
        self._validate_manifest(
            "numpy_native_tree",
            self.numpy_native_tree_file_count,
            self.numpy_native_tree_total_byte_length,
            self.numpy_native_tree_manifest_byte_length,
            self.numpy_native_tree_sha256,
            _MAXIMUM_NATIVE_TREE_FILES,
        )
        if (
            numpy_record_sha256 != profile["numpy_record_sha256"]
            or (
                self.numpy_content_tree_file_count,
                self.numpy_content_tree_total_byte_length,
                self.numpy_content_tree_manifest_byte_length,
                self.numpy_content_tree_sha256,
            )
            != profile["numpy_tree"]
            or (
                self.numpy_native_tree_file_count,
                self.numpy_native_tree_total_byte_length,
                self.numpy_native_tree_manifest_byte_length,
                self.numpy_native_tree_sha256,
            )
            != profile["numpy_native_tree"]
        ):
            raise SolarSystemContractError(
                "NumPy RECORD/tree identity is outside the calibrated ABI profile"
            )
        _token(
            self.numpy_record_validation_status,
            _RUNTIME_RECORD_STATUS,
            "numpy_record_validation_status",
        )
        _token(
            self.numpy_loaded_module_origin_status,
            _RUNTIME_MODULE_ORIGIN_STATUS,
            "numpy_loaded_module_origin_status",
        )
        cspice_path = _normalized_relative_path(
            self.loaded_cspice_relative_path,
            "loaded_cspice_relative_path",
        )
        cspice_length = _integer(
            self.loaded_cspice_byte_length,
            "loaded_cspice_byte_length",
            minimum=1,
            maximum=_MAXIMUM_RUNTIME_FILE_BYTES,
        )
        cspice_sha = _nonzero_sha256(
            self.loaded_cspice_sha256,
            "loaded_cspice_sha256",
        )
        if (
            cspice_path != "spiceypy/utils/libcspice.so"
            or cspice_length != _COMMON_CSPICE_LENGTH
            or cspice_sha != _COMMON_CSPICE_SHA256
        ):
            raise SolarSystemContractError(
                "loaded CSPICE bytes are outside the calibrated exact profile"
            )
        _token(self.cspice_toolkit_version, "CSPICE_N0067", "cspice_toolkit_version")
        _token(
            self.cspice_loader_resolution_status,
            _RUNTIME_LOADER_STATUS,
            "cspice_loader_resolution_status",
        )
        _token(
            self.cyice_execution_status,
            _RUNTIME_CYICE_STATUS,
            "cyice_execution_status",
        )
        loaded = self._validate_loaded_native_files()
        expected_cspice_entry = (
            spiceypy_id,
            cspice_path,
            cspice_length,
            cspice_sha,
        )
        expected_loaded = tuple(
            sorted(
                (expected_cspice_entry,)
                + tuple(
                    (numpy_id, path, byte_length, sha256)
                    for path, byte_length, sha256 in profile["loaded_numpy_natives"]
                )
            )
        )
        if loaded != expected_loaded:
            raise SolarSystemContractError(
                "loaded native roster differs from the calibrated exact ABI roster"
            )
        _token(
            self.loaded_native_map_observation_status,
            _RUNTIME_LOADED_MAP_STATUS,
            "loaded_native_map_observation_status",
        )
        self._validate_host_native_dependencies()
        _token(
            self.transitive_native_dependency_custody_status,
            _RUNTIME_DEPENDENCY_CUSTODY_STATUS,
            "transitive_native_dependency_custody_status",
        )
        _token(self.runtime_scope, _RUNTIME_SCOPE, "runtime_scope")
        _token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )

    @staticmethod
    def _validate_manifest(
        label: str,
        file_count: object,
        total_length: object,
        manifest_length: object,
        manifest_sha256: object,
        maximum_files: int,
    ) -> None:
        _integer(file_count, f"{label}_file_count", minimum=1, maximum=maximum_files)
        _integer(
            total_length,
            f"{label}_total_byte_length",
            minimum=1,
            maximum=_MAXIMUM_RUNTIME_FILE_BYTES,
        )
        _integer(
            manifest_length,
            f"{label}_manifest_byte_length",
            minimum=1,
            maximum=MAXIMUM_CANONICAL_BYTES,
        )
        _nonzero_sha256(manifest_sha256, f"{label}_sha256")

    def _validate_loaded_native_files(self) -> tuple[tuple[str, str, int, str], ...]:
        value = self.loaded_distribution_native_files
        if type(value) is not tuple:
            raise SolarSystemContractError(
                "loaded_distribution_native_files must be an exact tuple"
            )
        if not value or len(value) > _MAXIMUM_LOADED_NATIVE_FILES:
            raise SolarSystemContractError(
                "loaded_distribution_native_files is outside its count cap"
            )
        checked: list[tuple[str, str, int, str]] = []
        allowed_ids = {
            self.spiceypy_distribution_artifact_id,
            self.numpy_distribution_artifact_id,
        }
        for index, entry in enumerate(value):
            if type(entry) is not tuple or len(entry) != 4:
                raise SolarSystemContractError(
                    f"loaded_distribution_native_files[{index}] must be an exact four-item tuple"
                )
            artifact_id = _text(entry[0], f"loaded native[{index}].artifact_id")
            path = _normalized_relative_path(entry[1], f"loaded native[{index}].path")
            length = _integer(
                entry[2],
                f"loaded native[{index}].byte_length",
                minimum=1,
                maximum=_MAXIMUM_RUNTIME_FILE_BYTES,
            )
            digest = _nonzero_sha256(entry[3], f"loaded native[{index}].sha256")
            if artifact_id not in allowed_ids:
                raise SolarSystemContractError(
                    "loaded native entry is not owned by an exact observed distribution"
                )
            checked.append((artifact_id, path, length, digest))
        result = tuple(checked)
        if result != tuple(sorted(set(result))):
            raise SolarSystemContractError(
                "loaded native entries must be sorted and unique"
            )
        paths = tuple(item[1] for item in result)
        if len(set(paths)) != len(paths):
            raise SolarSystemContractError(
                "loaded native paths must be globally unique across distributions"
            )
        for artifact_id, path, _, _ in result:
            if artifact_id == self.spiceypy_distribution_artifact_id:
                if not path.startswith("spiceypy/"):
                    raise SolarSystemContractError(
                        "SpiceyPy-owned loaded native paths require the spiceypy prefix"
                    )
            elif not (path.startswith("numpy/") or path.startswith("numpy.libs/")):
                raise SolarSystemContractError(
                    "NumPy-owned loaded native paths require numpy or numpy.libs prefix"
                )
        return result

    def _validate_host_native_dependencies(self) -> None:
        value = self.host_native_dependencies
        if type(value) is not tuple:
            raise SolarSystemContractError(
                "host_native_dependencies must be an exact tuple"
            )
        if not value or len(value) > _MAXIMUM_HOST_NATIVE_FILES:
            raise SolarSystemContractError(
                "host_native_dependencies is outside its count cap"
            )
        checked: list[tuple[str, int, str]] = []
        for index, entry in enumerate(value):
            if type(entry) is not tuple or len(entry) != 3:
                raise SolarSystemContractError(
                    f"host_native_dependencies[{index}] must be an exact three-item tuple"
                )
            name = _text(entry[0], f"host native[{index}].name")
            length = _integer(
                entry[1],
                f"host native[{index}].byte_length",
                minimum=1,
                maximum=_MAXIMUM_RUNTIME_FILE_BYTES,
            )
            digest = _nonzero_sha256(entry[2], f"host native[{index}].sha256")
            checked.append((name, length, digest))
        result = tuple(checked)
        if result != tuple(sorted(set(result))):
            raise SolarSystemContractError(
                "host native entries must be sorted and unique"
            )
        if tuple(item[0] for item in result) != (
            "ld-linux-x86-64.so.2",
            "libc.so.6",
            "libm.so.6",
        ):
            raise SolarSystemContractError(
                "host native roster must be exactly loader, libc, and libm"
            )

    def validate_integrity(self) -> None:
        if type(self) is not CspiceRuntimeObservation:
            raise SolarSystemContractError(
                "runtime observation must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _runtime_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact runtime observation"
            )


def validate_cspice_runtime_observation(value: object) -> None:
    if type(value) is not CspiceRuntimeObservation:
        raise SolarSystemContractError(
            "value must be an exact CspiceRuntimeObservation"
        )
    value.validate_integrity()


def _segment_payload(value: CspiceSelectedSegmentRecord) -> tuple[object, ...]:
    return _record_payload(value, _SEGMENT_QUALIFIED_NAME)


def _segment_digest(value: CspiceSelectedSegmentRecord) -> str:
    return _digest(_SEGMENT_DOMAIN, _SEGMENT_SCHEMA, _segment_payload(value))


@dataclass(frozen=True, slots=True, eq=False)
class CspiceSelectedSegmentRecord:
    """One caller-supplied selected-segment corroboration record."""

    kernel_artifact_id: str
    kernel_artifact_sha256: str
    kernel_load_index: int
    selection_et: float
    selected_body_naif_id: int
    center_naif_id: int
    frame_naif_id: int
    segment_type: int
    first_et: float
    last_et: float
    begin_daf_address: int
    end_daf_address: int
    segment_identifier: str
    spkpvn_returned_frame_naif_id: int
    spkpvn_returned_center_naif_id: int
    spkpvn_state: tuple[float, float, float, float, float, float]
    selection_status: str
    evidence_scope: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _SEGMENT_DOMAIN
    _SCHEMA: ClassVar[str] = _SEGMENT_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not CspiceSelectedSegmentRecord:
            raise SolarSystemContractError(
                "selected segment record must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _segment_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact selected segment record"
            )

    def _validate(self) -> None:
        if type(self) is not CspiceSelectedSegmentRecord:
            raise SolarSystemContractError(
                "selected segment record must have its exact concrete type"
            )
        _text(self.kernel_artifact_id, "kernel_artifact_id")
        _nonzero_sha256(self.kernel_artifact_sha256, "kernel_artifact_sha256")
        _integer(self.kernel_load_index, "kernel_load_index", minimum=0, maximum=0)
        selection = _finite_float(self.selection_et, "selection_et")
        _integer(
            self.selected_body_naif_id,
            "selected_body_naif_id",
            minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
        )
        _integer(
            self.center_naif_id,
            "center_naif_id",
            minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
        )
        if self.selected_body_naif_id == self.center_naif_id:
            raise SolarSystemContractError(
                "selected segment body and center must be distinct"
            )
        _integer(self.frame_naif_id, "frame_naif_id", minimum=1, maximum=1)
        _integer(self.segment_type, "segment_type", minimum=2, maximum=2)
        first = _finite_float(self.first_et, "first_et")
        last = _finite_float(self.last_et, "last_et")
        if first > last or selection < first or selection > last:
            raise SolarSystemContractError(
                "selected segment must contain selection_et under CLOSED_CLOSED bounds"
            )
        begin = _integer(
            self.begin_daf_address,
            "begin_daf_address",
            minimum=1,
            maximum=_MAXIMUM_SPICE_INT,
        )
        end = _integer(
            self.end_daf_address,
            "end_daf_address",
            minimum=1,
            maximum=_MAXIMUM_SPICE_INT,
        )
        if begin > end:
            raise SolarSystemContractError(
                "begin_daf_address cannot exceed end_daf_address"
            )
        _text(
            self.segment_identifier,
            "segment_identifier",
            maximum_codepoints=_MAXIMUM_SEGMENT_IDENTIFIER_CODEPOINTS,
        )
        _integer(
            self.spkpvn_returned_frame_naif_id,
            "spkpvn_returned_frame_naif_id",
            minimum=1,
            maximum=1,
        )
        returned_center = _integer(
            self.spkpvn_returned_center_naif_id,
            "spkpvn_returned_center_naif_id",
            minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
        )
        if returned_center != self.center_naif_id:
            raise SolarSystemContractError(
                "SPKPVN returned center must equal the decoded segment center"
            )
        _exact_float_tuple(self.spkpvn_state, 6, "spkpvn_state")
        _token(
            self.selection_status,
            _SEGMENT_SELECTION_STATUS,
            "selection_status",
        )
        _token(self.evidence_scope, _SEGMENT_EVIDENCE_SCOPE, "evidence_scope")

    def validate_integrity(self) -> None:
        if type(self) is not CspiceSelectedSegmentRecord:
            raise SolarSystemContractError(
                "selected segment record must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _segment_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact selected segment record"
            )


def validate_cspice_selected_segment_record(value: object) -> None:
    if type(value) is not CspiceSelectedSegmentRecord:
        raise SolarSystemContractError(
            "value must be an exact CspiceSelectedSegmentRecord"
        )
    value.validate_integrity()


def _segment_ref(value: CspiceSelectedSegmentRecord) -> tuple[str, str, str]:
    return _child_ref(
        value,
        CspiceSelectedSegmentRecord,
        validate_cspice_selected_segment_record,
        _SEGMENT_QUALIFIED_NAME,
        _SEGMENT_SCHEMA,
        "selected_segment",
    )


def _chain_payload(value: CspiceTargetChainEvidence) -> tuple[object, ...]:
    return _record_payload(
        value,
        _CHAIN_QUALIFIED_NAME,
        {"legs": tuple(_segment_ref(leg) for leg in value.legs)},
    )


def _chain_digest(value: CspiceTargetChainEvidence) -> str:
    return _digest(_CHAIN_DOMAIN, _CHAIN_SCHEMA, _chain_payload(value))


def _binary64_sum(left: float, right: float) -> float:
    exact = Fraction.from_float(left) + Fraction.from_float(right)
    if exact == 0:
        if (
            left == 0.0
            and right == 0.0
            and math.copysign(1.0, left) < 0.0
            and math.copysign(1.0, right) < 0.0
        ):
            return -0.0
        return 0.0
    return _round_fraction_to_binary64(exact).rounded_value


def _reconstruct_chain_state(
    legs: tuple[CspiceSelectedSegmentRecord, ...],
) -> tuple[float, float, float, float, float, float]:
    first = legs[0].spkpvn_state
    accumulated = [first[index] for index in range(6)]
    for leg in legs[1:]:
        for index in range(6):
            accumulated[index] = _binary64_sum(
                accumulated[index],
                leg.spkpvn_state[index],
            )
    return (
        accumulated[0],
        accumulated[1],
        accumulated[2],
        accumulated[3],
        accumulated[4],
        accumulated[5],
    )


def _packed_state(value: tuple[float, ...]) -> bytes:
    return struct.pack(">6d", *value)


@dataclass(frozen=True, slots=True, eq=False)
class CspiceTargetChainEvidence:
    """SPKGEO result plus nonauthorizing selected-chain corroboration."""

    target_naif_id: int
    observer_naif_id: int
    reference_frame_name: str
    effective_et: float
    spkgeo_state: tuple[float, float, float, float, float, float]
    spkgeo_light_time_seconds: float
    legs: tuple[CspiceSelectedSegmentRecord, ...]
    terminal_center_naif_id: int
    reconstructed_state: tuple[float, float, float, float, float, float]
    comparison_recipe: str
    comparison_status: str
    evidence_scope: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _CHAIN_DOMAIN
    _SCHEMA: ClassVar[str] = _CHAIN_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not CspiceTargetChainEvidence:
            raise SolarSystemContractError(
                "target chain evidence must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _chain_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact target chain evidence"
            )

    def _validate(self) -> None:
        if type(self) is not CspiceTargetChainEvidence:
            raise SolarSystemContractError(
                "target chain evidence must have its exact concrete type"
            )
        target = _integer(
            self.target_naif_id,
            "target_naif_id",
            minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
        )
        observer = _integer(
            self.observer_naif_id,
            "observer_naif_id",
            minimum=0,
            maximum=0,
        )
        if target == observer:
            raise SolarSystemContractError("target cannot equal observer")
        _token(self.reference_frame_name, "J2000", "reference_frame_name")
        effective = _finite_float(self.effective_et, "effective_et")
        spkgeo = _exact_float_tuple(self.spkgeo_state, 6, "spkgeo_state")
        light_time = _finite_float(
            self.spkgeo_light_time_seconds,
            "spkgeo_light_time_seconds",
        )
        if light_time < 0.0 or (
            light_time == 0.0 and math.copysign(1.0, light_time) < 0.0
        ):
            raise SolarSystemContractError(
                "SPKGEO light time must be nonnegative with canonical positive zero"
            )
        legs = self.legs
        if type(legs) is not tuple:
            raise SolarSystemContractError("legs must be an exact tuple")
        if not legs or len(legs) > _MAXIMUM_CHAIN_LENGTH:
            raise SolarSystemContractError("legs is outside the exact chain-length cap")
        for index, leg in enumerate(legs):
            if type(leg) is not CspiceSelectedSegmentRecord:
                raise SolarSystemContractError(
                    f"legs[{index}] must be an exact CspiceSelectedSegmentRecord"
                )
        checked = tuple(
            (leg.validate_integrity(), leg)[1]
            for leg in legs
        )
        if checked[0].selected_body_naif_id != target:
            raise SolarSystemContractError("first selected segment must start at target")
        selected_bodies: list[int] = []
        expected_body = target
        for index, leg in enumerate(checked):
            if leg.selected_body_naif_id != expected_body:
                raise SolarSystemContractError(
                    f"legs[{index}] does not continue the body-to-center chain"
                )
            if leg.selected_body_naif_id == observer:
                raise SolarSystemContractError(
                    "selected segment traversal cannot leave the observer"
                )
            is_terminal_leg = index == len(checked) - 1
            if (leg.center_naif_id == observer) is not is_terminal_leg:
                raise SolarSystemContractError(
                    "the selected chain must first reach observer on its final leg"
                )
            if not _float_bits_equal(leg.selection_et, effective):
                raise SolarSystemContractError(
                    f"legs[{index}] selection ET differs from chain effective ET"
                )
            selected_bodies.append(leg.selected_body_naif_id)
            expected_body = leg.center_naif_id
        if len(set(selected_bodies)) != len(selected_bodies):
            raise SolarSystemContractError("selected segment chain contains a cycle")
        terminal = _integer(
            self.terminal_center_naif_id,
            "terminal_center_naif_id",
            minimum=0,
            maximum=0,
        )
        if expected_body != terminal or terminal != observer:
            raise SolarSystemContractError(
                "selected segment chain must terminate at observer 0"
            )
        reconstructed = _exact_float_tuple(
            self.reconstructed_state,
            6,
            "reconstructed_state",
        )
        expected_reconstructed = _reconstruct_chain_state(checked)
        if any(
            not _float_bits_equal(actual, expected)
            for actual, expected in zip(reconstructed, expected_reconstructed)
        ):
            raise SolarSystemContractError(
                "reconstructed_state does not match exact sequential RN-even addition"
            )
        _token(
            self.comparison_recipe,
            _CHAIN_COMPARISON_RECIPE,
            "comparison_recipe",
        )
        _token(
            self.comparison_status,
            _CHAIN_COMPARISON_STATUS,
            "comparison_status",
        )
        if _packed_state(reconstructed) != _packed_state(spkgeo):
            raise SolarSystemContractError(
                "reconstructed and SPKGEO states are not raw binary64 identical"
            )
        _token(self.evidence_scope, _CHAIN_EVIDENCE_SCOPE, "evidence_scope")

    def validate_integrity(self) -> None:
        if type(self) is not CspiceTargetChainEvidence:
            raise SolarSystemContractError(
                "target chain evidence must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _chain_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact target chain evidence"
            )


def validate_cspice_target_chain_evidence(value: object) -> None:
    if type(value) is not CspiceTargetChainEvidence:
        raise SolarSystemContractError(
            "value must be an exact CspiceTargetChainEvidence"
        )
    value.validate_integrity()


def _chain_ref(value: CspiceTargetChainEvidence) -> tuple[str, str, str]:
    return _child_ref(
        value,
        CspiceTargetChainEvidence,
        validate_cspice_target_chain_evidence,
        _CHAIN_QUALIFIED_NAME,
        _CHAIN_SCHEMA,
        "target_chain",
    )


def _runtime_ref(value: CspiceRuntimeObservation) -> tuple[str, str, str]:
    return _child_ref(
        value,
        CspiceRuntimeObservation,
        validate_cspice_runtime_observation,
        _RUNTIME_QUALIFIED_NAME,
        _RUNTIME_SCHEMA,
        "runtime",
    )


def _lane_ref(value: CspiceSpkgeoExecutionLane) -> tuple[str, str, str]:
    return _child_ref(
        value,
        CspiceSpkgeoExecutionLane,
        validate_cspice_spkgeo_execution_lane,
        _LANE_QUALIFIED_NAME,
        _LANE_SCHEMA,
        "execution_lane",
    )


def _local_receipt_refs(
    values: tuple[LocalArtifactVerificationReceipt, ...],
    label: str,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        _local_receipt_ref(value, f"{label}[{index}]")
        for index, value in enumerate(values)
    )


def _byte_match_refs(
    values: tuple[LocalArtifactVerificationReceipt, ...],
    label: str,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        _byte_match_ref(value.byte_match, f"{label}[{index}].byte_match")
        for index, value in enumerate(values)
    )


def _state_batch_semantic_subset(value: ProviderNativeStateBatch) -> tuple[object, ...]:
    return (
        ("effective_query_content_sha256", value.query.content_sha256),
        ("state_stage", value.state_stage),
        ("target_naif_ids", value.target_naif_ids),
        ("component_order", value.component_order),
        ("state_shape", value.state_shape),
        ("position_unit_id", value.position_unit_id),
        ("velocity_length_unit_id", value.velocity_length_unit_id),
        ("velocity_time_unit_id", value.velocity_time_unit_id),
        ("velocity_semantics", value.velocity_semantics),
        ("positions", value.positions),
        ("velocities", value.velocities),
        ("target_availability_status", value.target_availability_status),
    )


def _lane_full_payload(value: CspiceSpkgeoExecutionLane) -> tuple[object, ...]:
    replacements = {
        "projection": _projection_ref(value.projection),
        "runtime": _runtime_ref(value.runtime),
        "implementation_artifact_verifications": _local_receipt_refs(
            value.implementation_artifact_verifications,
            "implementation_artifact_verifications",
        ),
        "pre_spk_verifications": _local_receipt_refs(
            value.pre_spk_verifications,
            "pre_spk_verifications",
        ),
        "post_spk_verifications": _local_receipt_refs(
            value.post_spk_verifications,
            "post_spk_verifications",
        ),
        "state_batch": _state_batch_ref(value.state_batch),
        "target_chains": tuple(_chain_ref(chain) for chain in value.target_chains),
    }
    return _record_payload(value, _LANE_QUALIFIED_NAME, replacements)


def _lane_content_digest(value: CspiceSpkgeoExecutionLane) -> str:
    return _digest(_LANE_DOMAIN, _LANE_SCHEMA, _lane_full_payload(value))


def _lane_semantic_payload(value: CspiceSpkgeoExecutionLane) -> tuple[object, ...]:
    return (
        _LANE_QUALIFIED_NAME,
        (
            ("projection", _projection_ref(value.projection)),
            ("provider_profile", value.provider_profile),
            ("runtime", _runtime_ref(value.runtime)),
            (
                "implementation_artifact_byte_matches",
                _byte_match_refs(
                    value.implementation_artifact_verifications,
                    "implementation_artifact_verifications",
                ),
            ),
            (
                "pre_spk_byte_matches",
                _byte_match_refs(value.pre_spk_verifications, "pre_spk_verifications"),
            ),
            (
                "post_spk_byte_matches",
                _byte_match_refs(value.post_spk_verifications, "post_spk_verifications"),
            ),
            ("ordered_loaded_artifact_ids", value.ordered_loaded_artifact_ids),
            ("kernel_pool_count_field_order", value.kernel_pool_count_field_order),
            ("fresh_pool_counts", value.fresh_pool_counts),
            ("loaded_pool_counts", value.loaded_pool_counts),
            ("post_query_pool_counts", value.post_query_pool_counts),
            ("after_clear_pool_counts", value.after_clear_pool_counts),
            ("loaded_kernel_inventory", value.loaded_kernel_inventory),
            ("kernel_pool_status", value.kernel_pool_status),
            ("cspice_error_state_status", value.cspice_error_state_status),
            ("provider_call", value.provider_call),
            ("state_result", _state_batch_semantic_subset(value.state_batch)),
            ("target_chains", tuple(_chain_ref(chain) for chain in value.target_chains)),
            ("state_payload_recipe", value.state_payload_recipe),
            ("state_payload_byte_length", value.state_payload_byte_length),
            ("state_payload_sha256", value.state_payload_sha256),
            ("provider_execution_status", value.provider_execution_status),
            ("nested_batch_evidence_status", value.nested_batch_evidence_status),
            ("network_status", value.network_status),
            ("fallback_status", value.fallback_status),
            ("extrapolation_status", value.extrapolation_status),
        ),
    )


def _lane_semantic_digest(value: CspiceSpkgeoExecutionLane) -> str:
    return _digest(
        _LANE_SEMANTIC_DOMAIN,
        _LANE_SEMANTIC_SCHEMA,
        _lane_semantic_payload(value),
    )


def _require_exact_local_receipt_tuple(
    value: object,
    label: str,
    *,
    minimum_count: int,
    maximum_count: int,
) -> tuple[LocalArtifactVerificationReceipt, ...]:
    if type(value) is not tuple:
        raise SolarSystemContractError(f"{label} must be an exact tuple")
    if len(value) < minimum_count or len(value) > maximum_count:
        raise SolarSystemContractError(f"{label} is outside its exact count cap")
    for index, item in enumerate(value):
        if type(item) is not LocalArtifactVerificationReceipt:
            raise SolarSystemContractError(
                f"{label}[{index}] must be an exact LocalArtifactVerificationReceipt"
            )
    for item in value:
        validate_local_artifact_verification_receipt(item)
    return value


def _require_exact_string_tuple(
    value: object,
    label: str,
    *,
    minimum_count: int,
    maximum_count: int,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise SolarSystemContractError(f"{label} must be an exact tuple")
    if len(value) < minimum_count or len(value) > maximum_count:
        raise SolarSystemContractError(f"{label} is outside its exact count cap")
    checked = tuple(_text(item, f"{label}[{index}]") for index, item in enumerate(value))
    if len(set(checked)) != len(checked):
        raise SolarSystemContractError(f"{label} cannot contain duplicates")
    return checked


def _require_pool_counts(value: object, label: str) -> tuple[int, int]:
    if type(value) is not tuple or len(value) != 2:
        raise SolarSystemContractError(f"{label} must be an exact two-item tuple")
    return (
        _integer(value[0], f"{label}[0]", minimum=0, maximum=MAXIMUM_ARTIFACT_COUNT),
        _integer(value[1], f"{label}[1]", minimum=0, maximum=MAXIMUM_ARTIFACT_COUNT),
    )


def _require_cspice_provider_profile(
    projection: CspiceBinary64QueryProjectionReceipt,
) -> None:
    query = projection.effective_query
    provider = query.provider
    identity = provider.identity
    frame = query.frame
    if (
        provider.spec_id != _PROVIDER_SPEC_ID
        or identity.provider_id != _PROVIDER_ID
        or identity.implementation_id != _IMPLEMENTATION_ID
        or identity.provider_kind != "LOCAL_OFFLINE_SPK"
        or identity.module_name != "spiceypy"
        or identity.distribution_name != "spiceypy"
        or identity.version != "8.2.0"
        or identity.distribution_version != "8.2.0"
    ):
        raise SolarSystemContractError(
            "effective query provider is outside the exact SpiceyPy profile"
        )
    if (
        query.observer_naif_id != 0
        or query.aberration_correction != "NONE"
        or query.state_kind != "GEOMETRIC"
        or provider.network_access is not False
        or provider.fallback_allowed is not False
        or provider.extrapolation_policy != "FORBID"
    ):
        raise SolarSystemContractError(
            "effective query operation is outside the exact geometric offline profile"
        )
    epoch = query.epoch
    if (
        epoch.time_scale != "TDB"
        or epoch.representation != "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART"
        or epoch.coordinate_unit_id != SECOND.unit_id
        or epoch.origin_id != "SPICE_J2000_TDB_ORIGIN"
    ):
        raise SolarSystemContractError(
            "effective query epoch is outside the exact SPICE TDB ET profile"
        )
    if (
        frame.frame_id != _FRAME_ID
        or frame.frame_kind != "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL"
        or frame.origin_kind != "SOLAR_SYSTEM_BARYCENTER"
        or frame.origin_naif_id != 0
        or frame.origin_realization_id != _ORIGIN_REALIZATION_ID
        or frame.axes_realization_id != _AXES_ID
        or frame.orientation_model_id != _ORIENTATION_ID
        or frame.orientation_time_dependence != "STATIC"
        or frame.coordinate_time_scale != "TDB"
    ):
        raise SolarSystemContractError(
            "effective query frame does not match the exact NAIF J2000 frame-1 crosswalk"
        )
    validate_integrity(KILOMETRE)
    validate_integrity(SECOND)
    units = query.output_unit_system
    if (
        units.length.content_sha256 != KILOMETRE.content_sha256
        or units.time.content_sha256 != SECOND.content_sha256
    ):
        raise SolarSystemContractError(
            "effective query native units must be exact sealed kilometre and second"
        )
    if len(provider.ordered_load_artifact_ids) != 1:
        raise SolarSystemContractError(
            "the exact v1 CSPICE profile requires one directly furnished SPK"
        )
    by_id = {artifact.artifact_id: artifact for artifact in provider.artifacts}
    load_id = provider.ordered_load_artifact_ids[0]
    if by_id[load_id].artifact_role != "SPK":
        raise SolarSystemContractError("the direct load artifact must be an SPK")


def _preflight_lane_graph(value: CspiceSpkgeoExecutionLane) -> None:
    for label, sequence, minimum, maximum in (
        (
            "implementation_artifact_verifications",
            value.implementation_artifact_verifications,
            1,
            MAXIMUM_ARTIFACT_COUNT,
        ),
        ("pre_spk_verifications", value.pre_spk_verifications, 1, 1),
        ("post_spk_verifications", value.post_spk_verifications, 1, 1),
        ("ordered_loaded_artifact_ids", value.ordered_loaded_artifact_ids, 1, 1),
        ("loaded_kernel_inventory", value.loaded_kernel_inventory, 1, 1),
        ("target_chains", value.target_chains, 1, MAXIMUM_BODY_COUNT),
    ):
        if type(sequence) is not tuple:
            raise SolarSystemContractError(f"{label} must be an exact tuple")
        if len(sequence) < minimum or len(sequence) > maximum:
            raise SolarSystemContractError(f"{label} is outside its shallow count cap")
    total = 0
    shallow_legs: list[CspiceSelectedSegmentRecord] = []
    for index, chain in enumerate(value.target_chains):
        if type(chain) is not CspiceTargetChainEvidence:
            raise SolarSystemContractError(
                f"target_chains[{index}] must be an exact CspiceTargetChainEvidence"
            )
        if type(chain.legs) is not tuple:
            raise SolarSystemContractError(
                f"target_chains[{index}].legs must be an exact tuple"
            )
        count = len(chain.legs)
        if count < 1 or count > _MAXIMUM_CHAIN_LENGTH:
            raise SolarSystemContractError(
                f"target_chains[{index}].legs is outside its shallow count cap"
            )
        total += count
        if total > _MAXIMUM_TOTAL_SEGMENT_REFERENCES:
            raise SolarSystemContractError(
                "total selected segment references exceed the global graph cap"
            )
        for leg_index, leg in enumerate(chain.legs):
            if type(leg) is not CspiceSelectedSegmentRecord:
                raise SolarSystemContractError(
                    f"target_chains[{index}].legs[{leg_index}] must be an exact "
                    "CspiceSelectedSegmentRecord"
                )
            shallow_legs.append(leg)

    selected_body_records: dict[tuple[str, int], str] = {}
    observed_ranges: list[tuple[str, int, int, str]] = []
    for index, leg in enumerate(shallow_legs):
        kernel_sha256 = _nonzero_sha256(
            leg.kernel_artifact_sha256,
            f"shallow_legs[{index}].kernel_artifact_sha256",
        )
        selected_body = _integer(
            leg.selected_body_naif_id,
            f"shallow_legs[{index}].selected_body_naif_id",
            minimum=-MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
            maximum=MAXIMUM_NAIF_IDENTIFIER_MAGNITUDE,
        )
        begin = _integer(
            leg.begin_daf_address,
            f"shallow_legs[{index}].begin_daf_address",
            minimum=1,
            maximum=_MAXIMUM_SPICE_INT,
        )
        end = _integer(
            leg.end_daf_address,
            f"shallow_legs[{index}].end_daf_address",
            minimum=1,
            maximum=_MAXIMUM_SPICE_INT,
        )
        if begin > end:
            raise SolarSystemContractError(
                f"shallow_legs[{index}] has reversed DAF addresses"
            )
        record_sha256 = validate_sha256(
            leg.content_sha256,
            f"shallow_legs[{index}].content_sha256",
        )
        body_key = (kernel_sha256, selected_body)
        prior_body_record = selected_body_records.get(body_key)
        if prior_body_record is not None and prior_body_record != record_sha256:
            raise SolarSystemContractError(
                "one loaded SPK and ET cannot select different records for one body"
            )
        selected_body_records[body_key] = record_sha256
        for prior_kernel, prior_begin, prior_end, prior_record in observed_ranges:
            if kernel_sha256 != prior_kernel:
                continue
            overlaps = not (end < prior_begin or begin > prior_end)
            identical_shared_record = (
                begin == prior_begin
                and end == prior_end
                and record_sha256 == prior_record
            )
            if overlaps and not identical_shared_record:
                raise SolarSystemContractError(
                    "one SPK cannot assign overlapping DAF ranges to distinct records"
                )
        observed_ranges.append((kernel_sha256, begin, end, record_sha256))


def _state_payload_bytes(batch: ProviderNativeStateBatch) -> bytes:
    payload = bytearray()
    for target_index in range(len(batch.target_naif_ids)):
        offset = target_index * 3
        payload.extend(
            struct.pack(
                ">6d",
                batch.positions[offset],
                batch.positions[offset + 1],
                batch.positions[offset + 2],
                batch.velocities[offset],
                batch.velocities[offset + 1],
                batch.velocities[offset + 2],
            )
        )
    return bytes(payload)


@dataclass(frozen=True, slots=True, eq=False)
class CspiceSpkgeoExecutionLane:
    """One declared SPKGEO lane with operational and semantic-result seals."""

    lane_role: str
    execution_instance_nonce: str
    execution_instance_evidence_status: str
    provider_profile: str
    projection: CspiceBinary64QueryProjectionReceipt
    runtime: CspiceRuntimeObservation
    implementation_artifact_verifications: tuple[LocalArtifactVerificationReceipt, ...]
    pre_spk_verifications: tuple[LocalArtifactVerificationReceipt, ...]
    post_spk_verifications: tuple[LocalArtifactVerificationReceipt, ...]
    ordered_loaded_artifact_ids: tuple[str, ...]
    kernel_pool_count_field_order: tuple[str, str]
    fresh_pool_counts: tuple[int, int]
    loaded_pool_counts: tuple[int, int]
    post_query_pool_counts: tuple[int, int]
    after_clear_pool_counts: tuple[int, int]
    loaded_kernel_inventory: tuple[tuple[str, str, str, str], ...]
    kernel_pool_status: str
    cspice_error_state_status: str
    provider_call: str
    state_batch: ProviderNativeStateBatch
    target_chains: tuple[CspiceTargetChainEvidence, ...]
    state_payload_recipe: str
    state_payload_byte_length: int
    state_payload_sha256: str
    provider_execution_status: str
    nested_batch_evidence_status: str
    atomicity_status: str
    artifact_at_use_custody_status: str
    process_isolation_status: str
    network_status: str
    fallback_status: str
    extrapolation_status: str
    portability_scope: str
    semantic_content_sha256: str = ""
    content_integrity_class: str = _CONTENT_INTEGRITY_CLASS
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _LANE_DOMAIN
    _SCHEMA: ClassVar[str] = _LANE_SCHEMA
    _SEMANTIC_DOMAIN: ClassVar[str] = _LANE_SEMANTIC_DOMAIN
    _SEMANTIC_SCHEMA: ClassVar[str] = _LANE_SEMANTIC_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not CspiceSpkgeoExecutionLane:
            raise SolarSystemContractError(
                "SPKGEO execution lane must have its exact concrete type"
            )
        _preflight_optional_sha256(
            self.semantic_content_sha256,
            "semantic_content_sha256",
        )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected_semantic = _lane_semantic_digest(self)
        if self.semantic_content_sha256 == "":
            object.__setattr__(self, "semantic_content_sha256", expected_semantic)
        elif self.semantic_content_sha256 != expected_semantic:
            raise SolarSystemContractError(
                "semantic_content_sha256 does not bind the exact result core"
            )
        expected_content = _lane_content_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected_content)
        elif self.content_sha256 != expected_content:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact execution lane"
            )

    def _validate(self) -> None:
        if type(self) is not CspiceSpkgeoExecutionLane:
            raise SolarSystemContractError(
                "SPKGEO execution lane must have its exact concrete type"
            )
        _preflight_lane_graph(self)
        role = _text(self.lane_role, "lane_role")
        if role not in _LANE_ROLES:
            raise SolarSystemContractError("lane_role is outside its closed roster")
        nonce = validate_sha256(self.execution_instance_nonce, "execution_instance_nonce")
        if nonce == "0" * 64:
            raise SolarSystemContractError("execution_instance_nonce cannot be all zero")
        _token(
            self.execution_instance_evidence_status,
            _LANE_INSTANCE_STATUS,
            "execution_instance_evidence_status",
        )
        _token(self.provider_profile, _PROVIDER_PROFILE, "provider_profile")
        _projection_ref(self.projection)
        _require_cspice_provider_profile(self.projection)
        _runtime_ref(self.runtime)
        provider = self.projection.effective_query.provider
        identity = provider.identity
        if self.runtime.provider_identity.content_sha256 != identity.content_sha256:
            raise SolarSystemContractError(
                "runtime provider identity must equal the effective query provider"
            )
        profile = _ABI_PROFILES[self.runtime.python_wheel_interpreter_abi_tag]
        by_id = {artifact.artifact_id: artifact for artifact in provider.artifacts}
        for artifact_id, role_name, version, byte_length, sha256 in (
            (
                self.runtime.spiceypy_distribution_artifact_id,
                "SOFTWARE_DISTRIBUTION",
                "8.2.0",
                profile["spiceypy_wheel_length"],
                profile["spiceypy_wheel_sha256"],
            ),
            (
                self.runtime.numpy_distribution_artifact_id,
                "SOFTWARE_DISTRIBUTION",
                "2.3.5",
                profile["numpy_wheel_length"],
                profile["numpy_wheel_sha256"],
            ),
        ):
            artifact = by_id.get(artifact_id)
            if (
                artifact is None
                or artifact.artifact_role != role_name
                or artifact.version != version
                or artifact.byte_length != byte_length
                or artifact.artifact_sha256 != sha256
            ):
                raise SolarSystemContractError(
                    "runtime distribution artifact differs from its calibrated wheel binding"
                )
        native = by_id.get(self.runtime.loaded_cspice_artifact_id)
        if (
            native is None
            or native.artifact_role != "NATIVE_LIBRARY"
            or native.version != _COMMON_CSPICE_ARTIFACT_VERSION
            or native.byte_length != _COMMON_CSPICE_LENGTH
            or native.artifact_sha256 != _COMMON_CSPICE_SHA256
            or native.byte_length != self.runtime.loaded_cspice_byte_length
            or native.artifact_sha256 != self.runtime.loaded_cspice_sha256
        ):
            raise SolarSystemContractError(
                "loaded CSPICE identity does not match its exact provider artifact"
            )
        implementation = _require_exact_local_receipt_tuple(
            self.implementation_artifact_verifications,
            "implementation_artifact_verifications",
            minimum_count=1,
            maximum_count=MAXIMUM_ARTIFACT_COUNT,
        )
        implementation_ids = tuple(
            receipt.byte_match.artifact.artifact_id for receipt in implementation
        )
        if implementation_ids != identity.ordered_implementation_artifact_ids:
            raise SolarSystemContractError(
                "implementation verification order must equal provider identity order"
            )
        for receipt in implementation:
            artifact = by_id.get(receipt.byte_match.artifact.artifact_id)
            if artifact is None or artifact.content_sha256 != receipt.byte_match.artifact.content_sha256:
                raise SolarSystemContractError(
                    "implementation verification must bind an exact provider artifact"
                )
        before = _require_exact_local_receipt_tuple(
            self.pre_spk_verifications,
            "pre_spk_verifications",
            minimum_count=1,
            maximum_count=1,
        )
        after = _require_exact_local_receipt_tuple(
            self.post_spk_verifications,
            "post_spk_verifications",
            minimum_count=1,
            maximum_count=1,
        )
        load_ids = _require_exact_string_tuple(
            self.ordered_loaded_artifact_ids,
            "ordered_loaded_artifact_ids",
            minimum_count=1,
            maximum_count=1,
        )
        if load_ids != provider.ordered_load_artifact_ids:
            raise SolarSystemContractError(
                "ordered load IDs must equal the exact provider load roster"
            )
        load_id = load_ids[0]
        load_artifact = by_id[load_id]
        for label, receipt in (("pre", before[0]), ("post", after[0])):
            if receipt.byte_match.artifact.content_sha256 != load_artifact.content_sha256:
                raise SolarSystemContractError(
                    f"{label}-SPK verification does not bind the exact load artifact"
                )
        if before[0].byte_match.content_sha256 != after[0].byte_match.content_sha256:
            raise SolarSystemContractError(
                "pre/post SPK declared-byte matches must be identical"
            )
        if type(self.kernel_pool_count_field_order) is not tuple:
            raise SolarSystemContractError(
                "kernel_pool_count_field_order must be an exact tuple"
            )
        if len(self.kernel_pool_count_field_order) != 2:
            raise SolarSystemContractError(
                "kernel_pool_count_field_order must contain exactly two items"
            )
        kernel_pool_field_order = tuple(
            _text(item, f"kernel_pool_count_field_order[{index}]")
            for index, item in enumerate(self.kernel_pool_count_field_order)
        )
        if kernel_pool_field_order != ("ALL", "SPK"):
            raise SolarSystemContractError(
                "kernel pool count field order must be exactly ALL, SPK"
            )
        if _require_pool_counts(self.fresh_pool_counts, "fresh_pool_counts") != (0, 0):
            raise SolarSystemContractError("fresh kernel pool must be empty")
        if _require_pool_counts(self.loaded_pool_counts, "loaded_pool_counts") != (1, 1):
            raise SolarSystemContractError("loaded kernel pool must contain one SPK")
        if _require_pool_counts(
            self.post_query_pool_counts,
            "post_query_pool_counts",
        ) != (1, 1):
            raise SolarSystemContractError("post-query kernel pool must remain one SPK")
        if _require_pool_counts(
            self.after_clear_pool_counts,
            "after_clear_pool_counts",
        ) != (0, 0):
            raise SolarSystemContractError("after-clear kernel pool must be empty")
        inventory = self.loaded_kernel_inventory
        if type(inventory) is not tuple or len(inventory) != 1:
            raise SolarSystemContractError(
                "loaded_kernel_inventory must contain exactly one entry"
            )
        entry = inventory[0]
        if type(entry) is not tuple or len(entry) != 4:
            raise SolarSystemContractError(
                "loaded kernel inventory entry must be an exact four-item tuple"
            )
        for index, item in enumerate(entry):
            _text(item, f"loaded_kernel_inventory[0][{index}]")
        if entry != (load_id, "DAF", "SPK", "DIRECT_FURNSH_SOURCE_EMPTY_NO_META_KERNEL"):
            raise SolarSystemContractError(
                "loaded kernel inventory is outside the one-direct-SPK profile"
            )
        _token(self.kernel_pool_status, _LANE_POOL_STATUS, "kernel_pool_status")
        _token(
            self.cspice_error_state_status,
            _LANE_ERROR_STATUS,
            "cspice_error_state_status",
        )
        _token(self.provider_call, _LANE_PROVIDER_CALL, "provider_call")
        _state_batch_ref(self.state_batch)
        effective_query = self.projection.effective_query
        if self.state_batch.query.content_sha256 != effective_query.content_sha256:
            raise SolarSystemContractError(
                "nested state batch must bind the exact M4C1 effective query"
            )
        if (
            self.state_batch.position_unit_id != KILOMETRE.unit_id
            or self.state_batch.velocity_length_unit_id != KILOMETRE.unit_id
            or self.state_batch.velocity_time_unit_id != SECOND.unit_id
        ):
            raise SolarSystemContractError(
                "nested state batch units must be exact kilometre and kilometre per second"
            )
        chains = self.target_chains
        if type(chains) is not tuple:
            raise SolarSystemContractError("target_chains must be an exact tuple")
        target_count = len(effective_query.target_naif_ids)
        if len(chains) != target_count or target_count < 1 or target_count > MAXIMUM_BODY_COUNT:
            raise SolarSystemContractError(
                "target_chains must have the exact capped query target count"
            )
        total_segment_references = 0
        for index, chain in enumerate(chains):
            if type(chain) is not CspiceTargetChainEvidence:
                raise SolarSystemContractError(
                    f"target_chains[{index}] must be an exact CspiceTargetChainEvidence"
                )
            if type(chain.legs) is not tuple:
                raise SolarSystemContractError(
                    f"target_chains[{index}].legs must be an exact tuple"
                )
            if len(chain.legs) < 1 or len(chain.legs) > _MAXIMUM_CHAIN_LENGTH:
                raise SolarSystemContractError(
                    f"target_chains[{index}].legs is outside the shallow cap"
                )
            total_segment_references += len(chain.legs)
            if total_segment_references > _MAXIMUM_TOTAL_SEGMENT_REFERENCES:
                raise SolarSystemContractError(
                    "total selected segment references exceed the global graph cap"
                )
        effective_et = self.projection.binary64_projection.rounded_value
        for index, chain in enumerate(chains):
            chain.validate_integrity()
            target = effective_query.target_naif_ids[index]
            if chain.target_naif_id != target or chain.observer_naif_id != 0:
                raise SolarSystemContractError(
                    f"target_chains[{index}] does not preserve query target/observer order"
                )
            if not _float_bits_equal(chain.effective_et, effective_et):
                raise SolarSystemContractError(
                    f"target_chains[{index}] ET is not the exact projection binary64 value"
                )
            for leg in chain.legs:
                if (
                    leg.kernel_artifact_id != load_id
                    or leg.kernel_artifact_sha256 != load_artifact.artifact_sha256
                    or leg.kernel_load_index != 0
                ):
                    raise SolarSystemContractError(
                        "selected segment does not bind the exact single SPK artifact"
                    )
            offset = index * 3
            expected_state = (
                self.state_batch.positions[offset],
                self.state_batch.positions[offset + 1],
                self.state_batch.positions[offset + 2],
                self.state_batch.velocities[offset],
                self.state_batch.velocities[offset + 1],
                self.state_batch.velocities[offset + 2],
            )
            if any(
                not _float_bits_equal(actual, expected)
                for actual, expected in zip(chain.spkgeo_state, expected_state)
            ):
                raise SolarSystemContractError(
                    "SPKGEO chain state does not bit-bind the nested M4A value slice"
                )
        _token(
            self.state_payload_recipe,
            _LANE_STATE_PAYLOAD_RECIPE,
            "state_payload_recipe",
        )
        payload = _state_payload_bytes(self.state_batch)
        expected_length = target_count * 48
        if expected_length > _MAXIMUM_STATE_PAYLOAD_BYTES:
            raise SolarSystemContractError("state payload exceeds the hard byte cap")
        if _integer(
            self.state_payload_byte_length,
            "state_payload_byte_length",
            minimum=48,
            maximum=_MAXIMUM_STATE_PAYLOAD_BYTES,
        ) != expected_length:
            raise SolarSystemContractError(
                "state_payload_byte_length does not match the target roster"
            )
        expected_payload_sha = hashlib.sha256(payload).hexdigest()
        if _nonzero_sha256(self.state_payload_sha256, "state_payload_sha256") != expected_payload_sha:
            raise SolarSystemContractError(
                "state_payload_sha256 does not bind the exact big-endian state bytes"
            )
        _token(
            self.provider_execution_status,
            _LANE_EXECUTION_STATUS,
            "provider_execution_status",
        )
        _token(
            self.nested_batch_evidence_status,
            _LANE_NESTED_BATCH_STATUS,
            "nested_batch_evidence_status",
        )
        _token(self.atomicity_status, _LANE_ATOMICITY_STATUS, "atomicity_status")
        _token(
            self.artifact_at_use_custody_status,
            _LANE_ARTIFACT_CUSTODY_STATUS,
            "artifact_at_use_custody_status",
        )
        _token(
            self.process_isolation_status,
            _LANE_PROCESS_STATUS,
            "process_isolation_status",
        )
        _token(self.network_status, _LANE_NETWORK_STATUS, "network_status")
        _token(self.fallback_status, _LANE_FALLBACK_STATUS, "fallback_status")
        _token(
            self.extrapolation_status,
            _LANE_EXTRAPOLATION_STATUS,
            "extrapolation_status",
        )
        _token(self.portability_scope, _LANE_PORTABILITY_SCOPE, "portability_scope")
        _token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )

    def validate_integrity(self) -> None:
        if type(self) is not CspiceSpkgeoExecutionLane:
            raise SolarSystemContractError(
                "SPKGEO execution lane must have its exact concrete type"
            )
        validate_sha256(self.semantic_content_sha256, "semantic_content_sha256")
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.semantic_content_sha256 != _lane_semantic_digest(self):
            raise SolarSystemContractError(
                "semantic_content_sha256 does not bind the exact result core"
            )
        if self.content_sha256 != _lane_content_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact execution lane"
            )


def validate_cspice_spkgeo_execution_lane(value: object) -> None:
    if type(value) is not CspiceSpkgeoExecutionLane:
        raise SolarSystemContractError(
            "value must be an exact CspiceSpkgeoExecutionLane"
        )
    value.validate_integrity()


def _receipt_payload(value: CspiceSpkgeoExecutionReceipt) -> tuple[object, ...]:
    return _record_payload(
        value,
        _RECEIPT_QUALIFIED_NAME,
        {
            "primary_lane": _lane_ref(value.primary_lane),
            "replay_lane": _lane_ref(value.replay_lane),
        },
    )


def _receipt_digest(value: CspiceSpkgeoExecutionReceipt) -> str:
    return _digest(_RECEIPT_DOMAIN, _RECEIPT_SCHEMA, _receipt_payload(value))


@dataclass(frozen=True, slots=True, eq=False)
class CspiceSpkgeoExecutionReceipt:
    """Two declared lanes with equal deliberate semantic-result cores."""

    receipt_id: str
    primary_lane: CspiceSpkgeoExecutionLane
    replay_lane: CspiceSpkgeoExecutionLane
    semantic_replay_comparison_scope: str
    semantic_replay_status: str
    claim_scope: str
    content_integrity_class: str
    content_sha256: str = ""

    _DOMAIN: ClassVar[str] = _RECEIPT_DOMAIN
    _SCHEMA: ClassVar[str] = _RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not CspiceSpkgeoExecutionReceipt:
            raise SolarSystemContractError(
                "SPKGEO execution receipt must have its exact concrete type"
            )
        _preflight_optional_sha256(self.content_sha256, "content_sha256")
        self._validate()
        expected = _receipt_digest(self)
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
        elif self.content_sha256 != expected:
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact SPKGEO execution receipt"
            )

    def _validate(self) -> None:
        if type(self) is not CspiceSpkgeoExecutionReceipt:
            raise SolarSystemContractError(
                "SPKGEO execution receipt must have its exact concrete type"
            )
        _text(self.receipt_id, "receipt_id")
        if type(self.primary_lane) is not CspiceSpkgeoExecutionLane:
            raise SolarSystemContractError(
                "primary_lane must be an exact CspiceSpkgeoExecutionLane"
            )
        if type(self.replay_lane) is not CspiceSpkgeoExecutionLane:
            raise SolarSystemContractError(
                "replay_lane must be an exact CspiceSpkgeoExecutionLane"
            )
        self.primary_lane.validate_integrity()
        self.replay_lane.validate_integrity()
        if self.primary_lane.lane_role != "PRIMARY":
            raise SolarSystemContractError("primary_lane must have PRIMARY role")
        if self.replay_lane.lane_role != "REPLAY":
            raise SolarSystemContractError("replay_lane must have REPLAY role")
        if (
            self.primary_lane.execution_instance_nonce
            == self.replay_lane.execution_instance_nonce
        ):
            raise SolarSystemContractError(
                "primary and replay lanes require distinct caller instance nonces"
            )
        if (
            self.primary_lane.projection.content_sha256
            != self.replay_lane.projection.content_sha256
        ):
            raise SolarSystemContractError(
                "primary and replay lanes must bind the exact same projection"
            )
        if (
            self.primary_lane.semantic_content_sha256
            != self.replay_lane.semantic_content_sha256
        ):
            raise SolarSystemContractError(
                "primary and replay lane semantic result cores differ"
            )
        if self.primary_lane.content_sha256 == self.replay_lane.content_sha256:
            raise SolarSystemContractError(
                "operational lane seals must differ through role and instance nonce"
            )
        _token(
            self.semantic_replay_comparison_scope,
            _REPLAY_COMPARISON_SCOPE,
            "semantic_replay_comparison_scope",
        )
        _token(
            self.semantic_replay_status,
            _REPLAY_STATUS,
            "semantic_replay_status",
        )
        _token(self.claim_scope, _REPLAY_CLAIM_SCOPE, "claim_scope")
        _token(
            self.content_integrity_class,
            _CONTENT_INTEGRITY_CLASS,
            "content_integrity_class",
        )

    def validate_integrity(self) -> None:
        if type(self) is not CspiceSpkgeoExecutionReceipt:
            raise SolarSystemContractError(
                "SPKGEO execution receipt must have its exact concrete type"
            )
        validate_sha256(self.content_sha256, "content_sha256")
        self._validate()
        if self.content_sha256 != _receipt_digest(self):
            raise SolarSystemContractError(
                "content_sha256 does not bind the exact SPKGEO execution receipt"
            )


def validate_cspice_spkgeo_execution_receipt(value: object) -> None:
    if type(value) is not CspiceSpkgeoExecutionReceipt:
        raise SolarSystemContractError(
            "value must be an exact CspiceSpkgeoExecutionReceipt"
        )
    value.validate_integrity()


__all__ = [
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
