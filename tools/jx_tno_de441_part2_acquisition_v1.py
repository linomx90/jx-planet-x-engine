#!/usr/bin/env python3
"""Preregister exact local custody of the official DE441 part-2 SPK.

This module is deliberately prepare-only.  It freezes the official source
identity, the portable external-asset layout, and the future custody and
offline-validation contracts needed by the common-epoch TNO bridge.  It has
no network, download, publish, execute, or resume surface and retains no
kernel bytes.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = PROJECT_ROOT / "tools" / "jx_tno_de441_part2_acquisition_v1.py"
TEST_PATH = PROJECT_ROOT / "tests" / "test_tno_de441_part2_acquisition_v1.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "runs" / "tno_de441_part2_acquisition_v1"
PREPARED_ROSTER = frozenset({"PROTOCOL.json"})

PROTOCOL_SCHEMA = "jx.tno.de441-part2-acquisition.protocol.v1"
PROTOCOL_DOMAIN = b"jx.tno.de441-part2-acquisition.protocol.v1\0"
FUTURE_CUSTODY_SCHEMA = "jx.tno.de441-part2-acquisition.custody.v1"
FUTURE_CUSTODY_DOMAIN_HEX = (
    b"jx.tno.de441-part2-acquisition.custody.v1\0".hex()
)
CLASSIFICATION = "DE441_PART2_OFFICIAL_KERNEL_LOCAL_BYTE_CUSTODY_PREREGISTRATION_ONLY"
CLAIM_CEILING = "LOCAL_EXACT_KERNEL_BYTE_CUSTODY_AND_OFFICIAL_METADATA_LINEAGE_ONLY"
EXPECTED_BRANCH = "research/tno-de441-part2-acquisition-v1-20261014"
EXPECTED_PREREGISTRATION_TAG = (
    "tno-de441-part2-acquisition-v1-preregistered-20261014"
)

PARENT_TAG = "tno-common-epoch-full129-bridge-v1-preregistered-20261014"
PARENT_TAG_OBJECT = "4620bf745540da9569d9f94bc3677ab176bca8ef"
PARENT_TAG_PAYLOAD_BYTES = 327
PARENT_TAG_PAYLOAD_SHA256 = (
    "671d24cf40b4cc29ef7e9c5126880b3c8d0f645e2246c697e2cec3b9ab6ab863"
)
PARENT_COMMIT = "fca1d7b9ea3a97a8127665dcd0b87c600971a803"
PARENT_TREE = "300507c69d9c66996da51bb01ce4b7b7bd909979"
PARENT_PROTOCOL_CONTENT_SHA256 = (
    "e3c6fa8f9cb9701f9faf667068a87202204815e1bac942da46fc934dc6e66bc8"
)

KERNEL_FILENAME = "de441_part-2.bsp"
KERNEL_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
    "de441_part-2.bsp"
)
KERNEL_BYTE_LENGTH = 1_656_830_976
KERNEL_OFFICIAL_MD5 = "ad8dfa4e505ef0e3a5d587a5b4705632"
KERNEL_COVERAGE_START_JD_TDB = "2440400.50"
KERNEL_COVERAGE_STOP_JD_TDB = "8000016.50"
REQUIRED_INTERVAL_START_JD_TDB = "2452568.5"
REQUIRED_INTERVAL_STOP_JD_TDB = "2461200.5"
LOGICAL_LOCATOR_TEMPLATE = (
    "objects/sha256/<computed_lowercase_sha256>/de441_part-2.bsp"
)

CHECKSUMS_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
    "aa_checksums.txt"
)
SUMMARIES_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
    "aa_summaries.txt"
)
TECHNICAL_COMMENTS_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/"
    "de441_tech-comments.txt"
)


@dataclass(frozen=True, slots=True)
class FileBinding:
    role: str
    relative_path: str
    byte_length: int
    sha256: str


PARENT_BINDINGS = (
    FileBinding(
        "parent_source",
        "benchmarks/jx_tno_common_epoch_full129_bridge_v1.py",
        54_643,
        "b13cd45d35bc56a8a6a1864d8f8719a3edc540715d7acac999a7dea4c6c8a5be",
    ),
    FileBinding(
        "parent_test",
        "tests/test_tno_common_epoch_full129_bridge_v1.py",
        15_059,
        "38119e03c970af9b6f7a9486e3440a152849fc3641c881db9d7572691eb770c0",
    ),
    FileBinding(
        "parent_protocol",
        "runs/tno_common_epoch_full129_bridge_v1/PROTOCOL.json",
        29_996,
        "895025b3c1c37e27147e8555a74763e10d43d29396182a91decd02476694521d",
    ),
)

OFFICIAL_METADATA = (
    {
        "role": "official_md5_roster",
        "url": CHECKSUMS_URL,
        "byte_length": 559,
        "sha256": "3c618bed7d51a48ee059269097eeeab1f69f3b2641ec2aeb96132757418c845b",
        "required_extraction": (
            "EXACTLY_ONE_ASCII_LINE_WITH_MD5_"
            "AD8DFA4E505EF0E3A5D587A5B4705632_AND_FILENAME_DE441_PART-2.BSP"
        ),
    },
    {
        "role": "official_spk_summary_roster",
        "url": SUMMARIES_URL,
        "byte_length": 12_911,
        "sha256": "2b34f045b63731a6a315a83b1e2f240dd208a4b5a19ee3419be0573485c15f87",
        "required_extraction": (
            "EXACT_DE441_PART2_TARGET_CENTER_ROSTER_AND_1969_JUN_28_TO_"
            "17191_MAR_15_INTERVAL_STANZA"
        ),
    },
    {
        "role": "official_de441_technical_comments",
        "url": TECHNICAL_COMMENTS_URL,
        "byte_length": 59_453,
        "sha256": "eda144896f7a09fd810c6261036d05d767ce8c77cb2b3795bc824fb623881007",
        "required_extraction": (
            "EXACT_PART2_JD_2440400.50_TO_8000016.50_AND_INCLUDED_BODY_ROSTER"
        ),
    },
)

SPK_TARGET_CENTER_ROSTER = (
    (1, 0, "MERCURY_BARYCENTER"),
    (2, 0, "VENUS_BARYCENTER"),
    (3, 0, "EARTH_BARYCENTER"),
    (4, 0, "MARS_BARYCENTER"),
    (5, 0, "JUPITER_BARYCENTER"),
    (6, 0, "SATURN_BARYCENTER"),
    (7, 0, "URANUS_BARYCENTER"),
    (8, 0, "NEPTUNE_BARYCENTER"),
    (9, 0, "PLUTO_BARYCENTER"),
    (10, 0, "SUN"),
    (199, 1, "MERCURY"),
    (299, 2, "VENUS"),
    (301, 3, "MOON"),
    (399, 3, "EARTH"),
)

RESOLVED11_CHAINS = (
    (10, (10, 0), "SUN"),
    (199, (199, 1, 0), "MERCURY"),
    (299, (299, 2, 0), "VENUS"),
    (399, (399, 3, 0), "EARTH"),
    (301, (301, 3, 0), "MOON"),
    (4, (4, 0), "MARS_BARYCENTER"),
    (5, (5, 0), "JUPITER_BARYCENTER"),
    (6, (6, 0), "SATURN_BARYCENTER"),
    (7, (7, 0), "URANUS_BARYCENTER"),
    (8, (8, 0), "NEPTUNE_BARYCENTER"),
    (9, (9, 0), "PLUTO_BARYCENTER"),
)


class KernelProtocolError(RuntimeError):
    """A frozen source, ancestry, schema, or publication invariant failed."""


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
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise KernelProtocolError("record is not strict canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    if type(payload) is not bytes:
        raise KernelProtocolError("SHA-256 payload must be exact bytes")
    return hashlib.sha256(payload).hexdigest()


def _content_sha256(value: Mapping[str, Any]) -> str:
    if type(value) is not dict:
        raise KernelProtocolError("content record must be an exact object")
    record = dict(value)
    record.pop("content_sha256", None)
    return _sha256(PROTOCOL_DOMAIN + _canonical_json(record))


def _pairs(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise KernelProtocolError("JSON contains a duplicate or invalid key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise KernelProtocolError(f"JSON contains non-finite constant {value}")


def _strict_json(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes or not payload:
        raise KernelProtocolError("JSON payload must be nonempty exact bytes")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_constant,
        )
    except KernelProtocolError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
        raise KernelProtocolError("input is not strict UTF-8 JSON") from exc
    if type(value) is not dict:
        raise KernelProtocolError("JSON root must be an exact object")
    return value


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if type(relative_path) is not str:
        raise KernelProtocolError("binding path must be text")
    pure = PurePosixPath(relative_path)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in ("", ".", "..") for part in pure.parts)
    ):
        raise KernelProtocolError("binding path is not a contained POSIX path")
    return pure.parts


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_relative(root_fd: int, binding: FileBinding) -> bytes:
    parts = _relative_parts(binding.relative_path)
    directory_fd = os.dup(root_fd)
    try:
        for part in parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = next_fd
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        try:
            before = os.fstat(descriptor)
            path_before = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(before.st_mode)
                or before.st_nlink != 1
                or before.st_size != binding.byte_length
                or _stat_identity(before) != _stat_identity(path_before)
            ):
                raise KernelProtocolError(
                    f"{binding.role} type, link, identity, or byte length differs"
                )
            payload = bytearray()
            while len(payload) < before.st_size:
                block = os.pread(
                    descriptor,
                    min(1 << 20, before.st_size - len(payload)),
                    len(payload),
                )
                if not block:
                    raise KernelProtocolError(f"{binding.role} ended early")
                payload.extend(block)
            if os.pread(descriptor, 1, before.st_size):
                raise KernelProtocolError(f"{binding.role} has trailing bytes")
            after = os.fstat(descriptor)
            path_after = os.stat(parts[-1], dir_fd=directory_fd, follow_symlinks=False)
            if (
                _stat_identity(before) != _stat_identity(after)
                or _stat_identity(before) != _stat_identity(path_after)
            ):
                raise KernelProtocolError(f"{binding.role} changed while held")
        finally:
            os.close(descriptor)
    finally:
        os.close(directory_fd)
    exact = bytes(payload)
    if _sha256(exact) != binding.sha256:
        raise KernelProtocolError(f"{binding.role} SHA-256 differs")
    return exact


def _read_small_file(path: Path, label: str, maximum: int = 4_000_000) -> bytes:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        before = os.fstat(descriptor)
        path_before = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
            or _stat_identity(before) != _stat_identity(path_before)
        ):
            raise KernelProtocolError(f"{label} is linked, nonregular, or oversized")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(payload)),
                len(payload),
            )
            if not block:
                raise KernelProtocolError(f"{label} ended early")
            payload.extend(block)
        after = os.fstat(descriptor)
        path_after = path.stat(follow_symlinks=False)
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
        ):
            raise KernelProtocolError(f"{label} changed while held")
    finally:
        os.close(descriptor)
    return bytes(payload)


def _git(*arguments: str) -> bytes:
    process = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise KernelProtocolError("Git ancestry check failed")
    return process.stdout


def _verify_parent_ancestry() -> dict[str, object]:
    if _git("rev-parse", f"{PARENT_TAG}^{{tag}}").decode("ascii").strip() != PARENT_TAG_OBJECT:
        raise KernelProtocolError("parent tag object differs")
    if _git("rev-parse", f"{PARENT_TAG}^{{commit}}").decode("ascii").strip() != PARENT_COMMIT:
        raise KernelProtocolError("parent tag commit differs")
    if _git("rev-parse", f"{PARENT_TAG}^{{tree}}").decode("ascii").strip() != PARENT_TREE:
        raise KernelProtocolError("parent tag tree differs")
    tag_payload = _git("cat-file", "tag", PARENT_TAG)
    if (
        len(tag_payload) != PARENT_TAG_PAYLOAD_BYTES
        or _sha256(tag_payload) != PARENT_TAG_PAYLOAD_SHA256
    ):
        raise KernelProtocolError("parent tag payload differs")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", PARENT_COMMIT, "HEAD"],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if ancestry.returncode != 0:
        raise KernelProtocolError("HEAD does not descend from the parent preregistration")
    return {
        "tag": PARENT_TAG,
        "tag_object": PARENT_TAG_OBJECT,
        "tag_payload_bytes": PARENT_TAG_PAYLOAD_BYTES,
        "tag_payload_sha256": PARENT_TAG_PAYLOAD_SHA256,
        "commit": PARENT_COMMIT,
        "tree": PARENT_TREE,
    }


def _decimal(token: str, label: str) -> Decimal:
    if type(token) is not str:
        raise KernelProtocolError(f"{label} must be decimal text")
    try:
        value = Decimal(token)
    except InvalidOperation as exc:
        raise KernelProtocolError(f"{label} is not exact decimal text") from exc
    if not value.is_finite():
        raise KernelProtocolError(f"{label} is non-finite")
    return value


def audit_prerequisites() -> dict[str, object]:
    root_fd = os.open(
        PROJECT_ROOT,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        payloads = {
            binding.role: _read_relative(root_fd, binding)
            for binding in PARENT_BINDINGS
        }
    finally:
        os.close(root_fd)
    parent_protocol = _strict_json(payloads["parent_protocol"])
    if payloads["parent_protocol"] != _canonical_json(parent_protocol):
        raise KernelProtocolError("parent protocol is not canonical JSON")
    if (
        parent_protocol.get("schema")
        != "jx.tno.common-epoch-full129-bridge.protocol.v1"
        or parent_protocol.get("content_sha256")
        != PARENT_PROTOCOL_CONTENT_SHA256
        or parent_protocol.get("claims", {}).get("de441_kernel_bound") is not False
        or parent_protocol.get("claims", {}).get("execution_authorized") is not False
        or "EXACT_RETAINED_HASHED_DE441_KERNEL_WITH_FULL_INTERVAL_COVERAGE_NOT_BOUND"
        not in parent_protocol.get("execution_blockers", [])
    ):
        raise KernelProtocolError("parent bridge protocol semantics differ")
    requirement = (
        parent_protocol.get("physical_model", {})
        .get("de441_kernel_acquisition_requirement", {})
    )
    if (
        requirement.get("official_url") != KERNEL_URL
        or requirement.get("byte_length") != KERNEL_BYTE_LENGTH
        or requirement.get("official_md5") != KERNEL_OFFICIAL_MD5
        or requirement.get("coverage_jd_tdb")
        != [KERNEL_COVERAGE_START_JD_TDB, KERNEL_COVERAGE_STOP_JD_TDB]
    ):
        raise KernelProtocolError("parent DE441 part-2 requirement differs")
    parent_sources = parent_protocol.get("physical_model", {}).get("source_naif_ids")
    if parent_sources != [row[0] for row in RESOLVED11_CHAINS]:
        raise KernelProtocolError("parent resolved-11 source order differs")
    coverage_start = _decimal(KERNEL_COVERAGE_START_JD_TDB, "coverage start")
    coverage_stop = _decimal(KERNEL_COVERAGE_STOP_JD_TDB, "coverage stop")
    required_start = _decimal(REQUIRED_INTERVAL_START_JD_TDB, "required start")
    required_stop = _decimal(REQUIRED_INTERVAL_STOP_JD_TDB, "required stop")
    if not coverage_start <= required_start <= required_stop <= coverage_stop:
        raise KernelProtocolError("official kernel coverage does not contain the bridge")
    return {
        "binding_count": len(PARENT_BINDINGS),
        "parent_ancestry": _verify_parent_ancestry(),
        "parent_protocol_content_sha256": PARENT_PROTOCOL_CONTENT_SHA256,
        "coverage_contains_required_interval": True,
        "resolved11_source_order": list(parent_sources),
    }


def _binding_record(binding: FileBinding) -> dict[str, object]:
    return {
        "role": binding.role,
        "relative_path": binding.relative_path,
        "bytes": binding.byte_length,
        "sha256": binding.sha256,
    }


def _implementation_record(path: Path, relative_path: str) -> dict[str, object]:
    payload = _read_small_file(path, relative_path)
    return {
        "relative_path": relative_path,
        "bytes": len(payload),
        "sha256": _sha256(payload),
    }


def build_protocol() -> dict[str, object]:
    audit = audit_prerequisites()
    protocol: dict[str, object] = {
        "schema": PROTOCOL_SCHEMA,
        "classification": CLASSIFICATION,
        "claim_ceiling": CLAIM_CEILING,
        "status": "PREPARED_DE441_PART2_SOURCE_ACQUISITION_PREREGISTRATION_NO_BYTES",
        "objective": (
            "Freeze exact official-source identity and portable local custody for the "
            "DE441 part-2 SPK required by the common-epoch full129 bridge."
        ),
        "authority": {
            "expected_branch_for_preparation": EXPECTED_BRANCH,
            "public_cli": [
                "build",
                "audit-prerequisites",
                "prepare",
                "validate-prepared",
            ],
            "explicitly_absent_public_cli": [
                "download",
                "acquire",
                "publish",
                "execute",
                "execute-worker",
                "resume",
                "validate-custody",
            ],
            "network_access": False,
            "kernel_bytes_read_or_written": False,
            "scientific_execution_authorized": False,
            "future_preregistration_tag_name": EXPECTED_PREREGISTRATION_TAG,
            "future_successor_rule": (
                "A separately reviewed successor must bind the annotated preregistration "
                "tag object, peeled commit, tree, and exact source/test/protocol bytes "
                "before it may inspect staged kernel bytes."
            ),
            "git_custody_sequence": [
                "COMMIT_SOURCE_TEST_PROTOCOL_AND_SORTED_MATRIX_ENTRY_ONLY",
                "CREATE_ANNOTATED_PREREGISTRATION_TAG_ON_THE_EXACT_PREREGISTRATION_COMMIT",
                "SUCCESSOR_BINDS_TAG_OBJECT_PEELED_COMMIT_TREE_AND_EXACT_BYTES_BEFORE_"
                "STAGED_KERNEL_READ",
                "AFTER_OFFLINE_VALIDATION_COMMIT_ONLY_PORTABLE_CUSTODY_JSON_NEVER_THE_"
                "EXTERNAL_BSP",
                "CREATE_ANNOTATED_CUSTODY_TAG_ONLY_AFTER_THE_CUSTODY_COMMIT",
            ],
        },
        "parent_bridge": {
            "ancestry": audit["parent_ancestry"],
            "bindings": [_binding_record(binding) for binding in PARENT_BINDINGS],
            "protocol_content_sha256": PARENT_PROTOCOL_CONTENT_SHA256,
            "kernel_blocker_closed_by_this_protocol": False,
        },
        "official_source": {
            "artifact_id": "artifact.naif.de441-part2.spk.v1",
            "canonical_filename": KERNEL_FILENAME,
            "official_url": KERNEL_URL,
            "expected_byte_length": KERNEL_BYTE_LENGTH,
            "official_md5": KERNEL_OFFICIAL_MD5,
            "local_sha256": None,
            "local_sha256_policy": (
                "COMPUTE_ONLY_FROM_THE_COMPLETE_HELD_STAGED_FILE_AND_RECOMPUTE_FROM_"
                "THE_COMPLETE_HELD_PUBLISHED_FILE"
            ),
            "size_evidence": "OFFICIAL_HTTPS_HEAD_CONTENT_LENGTH_OBSERVED_2026-10-14",
            "metadata_evidence": [dict(record) for record in OFFICIAL_METADATA],
            "provider_evidence_limit": (
                "The frozen official URLs, metadata hashes, and MD5 establish source "
                "lineage and content checks, but a later local receipt does not prove "
                "continuous custody or cryptographically attest the provider."
            ),
        },
        "kernel_semantics": {
            "format": "NAIF_DAF_SPK",
            "reference_frame": "J2000_ICRF_COMPATIBLE",
            "coverage_jd_tdb": [
                KERNEL_COVERAGE_START_JD_TDB,
                KERNEL_COVERAGE_STOP_JD_TDB,
            ],
            "official_calendar_coverage": [
                "1969 JUN 28 00:00:00.000",
                "17191 MAR 15 00:00:00.000",
            ],
            "required_bridge_interval_jd_tdb": [
                REQUIRED_INTERVAL_START_JD_TDB,
                REQUIRED_INTERVAL_STOP_JD_TDB,
            ],
            "coverage_contains_required_bridge_interval": True,
            "stored_target_center_roster": [
                {
                    "target_naif_id": target,
                    "center_naif_id": center,
                    "semantic_id": semantic_id,
                }
                for target, center, semantic_id in SPK_TARGET_CENTER_ROSTER
            ],
            "resolved11_ssb_evaluation_order": [
                {
                    "source_order": order,
                    "target_naif_id": target,
                    "target_to_ssb_chain": list(chain),
                    "semantic_id": semantic_id,
                }
                for order, (target, chain, semantic_id) in enumerate(
                    RESOLVED11_CHAINS
                )
            ],
            "spk_structure_validation_deferred": (
                "THE_FUTURE_CUSTODY_SUCCESSOR_MUST_USE_A_SEPARATELY_PINNED_CSPICE_"
                "RUNTIME_TO_CONFIRM_TARGET_CENTER_FRAME_SEGMENT_AND_COVERAGE_"
                "SEMANTICS;THIS_PREPARE_ONLY_SOURCE_IMPORTS_NO_SPICE_RUNTIME"
            ),
            "generic_jpl_ephemeris_role": (
                "THE_EXISTING_JPL_EPHEMERIS_API_CAN_ADMIT_AND_HASH_A_KERNEL_OF_THIS_"
                "SIZE_FOR_ISOLATED_POINT_QUERIES_BUT_REHASHES_AND_SPAWNS_AN_ISOLATED_"
                "WORKER_PER_STATE_REQUEST;IT_IS_NOT_A_PERSISTENT_FORCE_CALLBACK_OR_"
                "ACCEPTABLE_BRIDGE_SOURCE_EVALUATOR"
            ),
            "persistent_force_callback_evaluator_qualified": False,
        },
        "transport_contract": {
            "transport_is_external_to_this_verifier": True,
            "required_http_behavior": (
                "HTTPS_GET_FOLLOW_REDIRECTS_FAIL_ON_HTTP_ERROR_NO_CONTENT_"
                "REENCODING_EXACT_RESPONSE_BYTES"
            ),
            "exact_staging_filename": KERNEL_FILENAME,
            "resumable_transfer_permitted": False,
            "staging_must_be_one_complete_response_body": True,
            "http_headers_are_not_scientific_authority": True,
        },
        "storage_contract": {
            "asset_root_argument_required_by_future_successor": True,
            "asset_root_input_form": "CALLER_SUPPLIED_ABSOLUTE_REAL_DIRECTORY",
            "asset_root_relationship": (
                "REAL_DIRECTORY_SIBLING_OF_THE_REAL_GIT_WORKTREE_ROOT_AND_NEVER_"
                "INSIDE_THE_WORKTREE"
            ),
            "reject_asset_root_inside_git_worktree": True,
            "reject_symlink_root_or_ancestors": True,
            "absolute_asset_root_recorded_in_protocol_or_custody": False,
            "portable_logical_locator_template": LOGICAL_LOCATOR_TEMPLATE,
            "locator_digest_substitution": (
                "REPLACE_COMPUTED_LOWERCASE_SHA256_PLACEHOLDER_WITH_THE_OBSERVED_"
                "64_HEX_SHA256_ONLY_AFTER_ALL_STAGING_GATES_PASS"
            ),
            "directory_mode_octal": "0700",
            "kernel_mode_octal": "0600",
            "kernel_link_count": 1,
            "minimum_two_copy_free_bytes_before_publication": 3_313_661_952,
            "space_basis": (
                "EXACTLY_TWO_TIMES_1656830976_FOR_ONE_COMPLETE_STAGED_COPY_AND_ONE_"
                "COMPLETE_PUBLISHED_COPY;FILESYSTEM_METADATA_MARGIN_IS_ADDITIONAL"
            ),
            "create_only_no_overwrite": True,
            "kernel_never_stored_in_git_worktree": True,
            "kernel_never_committed_to_git": True,
            "multi_file_atomicity_claimed": False,
        },
        "future_successor_gates": {
            "before_any_staged_byte_read": [
                "BIND_EXACT_ANNOTATED_PREREGISTRATION_TAG_OBJECT_COMMIT_TREE_AND_BYTES",
                "VALIDATE_EXACT_REBUILT_PROTOCOL_AND_PARENT_ANCESTRY",
                "VALIDATE_EXTERNAL_SIBLING_ASSET_ROOT_AND_PRIVATE_DIRECTORY_CHAIN",
            ],
            "staging": [
                "ONE_O_NOFOLLOW_O_CLOEXEC_HELD_FD_FOR_COMPLETE_VALIDATION",
                "REGULAR_FILE_LINK_COUNT_ONE_AND_EXACT_1656830976_BYTES",
                "IDENTITY_STABLE_BEFORE_DURING_AFTER_HASH_AND_PATH_STILL_NAMES_HELD_INODE",
                "EXACT_EOF_AFTER_DECLARED_BYTE_LENGTH",
                "COMPLETE_STREAMING_MD5_EQUALS_OFFICIAL_MD5",
                "COMPLETE_STREAMING_SHA256_COMPUTED_AS_AUTHORITATIVE_LOCAL_IDENTITY",
            ],
            "publication": [
                "CREATE_PRIVATE_CONTENT_ADDRESSED_PARENT_DIRECTORIES_ONLY",
                "STREAM_COPY_WITHOUT_MATERIALIZING_KERNEL_IN_MEMORY",
                "FSYNC_KERNEL_THEN_ATOMIC_NOREPLACE_PUBLICATION_THEN_FSYNC_DIRECTORIES",
                "ONE_HELD_PUBLISHED_FD_FOR_INDEPENDENT_COMPLETE_SIZE_MD5_AND_SHA256_"
                "REHASH",
                "CUSTODY_JSON_CANONICAL_CREATE_ONLY_AND_PUBLISHED_LAST",
                "FAILURE_NEVER_COUNTS_AS_CUSTODY_AND_PARTIAL_ROSTERS_FAIL_CLOSED",
                "AMBIGUOUS_RENAME_OR_INTERRUPT_REQUIRES_HELD_DIRECTORY_FD_INSPECTION_"
                "OF_BOTH_NAMES_AND_FULL_DESTINATION_VALIDATION_NEVER_BLIND_RETRY",
            ],
            "spk_semantics": [
                "PIN_EXACT_SPICEYPY_DISTRIBUTION_AND_LOADED_CSPICE_LIBRARY",
                "CONFIRM_DAF_SPK_STRUCTURE_AND_SEGMENT_DESCRIPTOR_INTEGRITY",
                "CONFIRM_EXACT_14_TARGET_CENTER_ROSTER_AND_J2000_FRAME",
                "CONFIRM_EVERY_REQUIRED_RESOLVED11_CHAIN_AT_BOTH_INTERVAL_ENDPOINTS",
                "CONFIRM_COVERAGE_CONTAINS_EVERY_NATIVE_EPOCH_TARGET_AND_ROUNDTRIP_TIME",
            ],
            "all_gates_must_pass": True,
            "failure_semantics": (
                "NO_CUSTODY_JSON_MEANS_NO_CUSTODY;A_FINAL_KERNEL_NAME_MAY_SURVIVE_AN_"
                "AMBIGUOUS_OR_POST_RENAME_FAILURE_BUT MUST_NEVER_BE_UNLINKED_BY_AN_"
                "UNVERIFIED_PATH_OR_COUNT_AS_VALID_UNTIL_THE_OFFLINE_VALIDATOR_PASSES"
            ),
        },
        "future_custody_artifact": {
            "schema": FUTURE_CUSTODY_SCHEMA,
            "content_hash_domain_hex": FUTURE_CUSTODY_DOMAIN_HEX,
            "tracked_relative_path": (
                "runs/tno_de441_part2_acquisition_v1/CUSTODY.json"
            ),
            "tracked_post_acquisition_roster": ["CUSTODY.json"],
            "external_kernel_only": True,
            "official_metadata_source_bytes_in_tracked_roster": False,
            "required_top_level_keys": [
                "schema",
                "classification",
                "claim_ceiling",
                "status",
                "protocol",
                "preregistration_git_custody",
                "source",
                "artifact",
                "spk_semantics",
                "validation",
                "claims",
                "published_at_utc",
                "content_sha256",
            ],
            "artifact_required_keys": [
                "artifact_id",
                "logical_locator",
                "filename",
                "byte_length",
                "official_md5",
                "sha256",
            ],
            "forbidden_values": [
                "ABSOLUTE_LOCAL_PATH",
                "DEVICE_ID",
                "INODE_NUMBER",
                "HOSTNAME",
                "USERNAME",
            ],
            "content_address_rule": (
                "ARTIFACT_SHA256_MUST_EQUAL_THE_64_HEX_LOCATOR_COMPONENT_AND_THE_"
                "INDEPENDENT_POST_PUBLICATION_REHASH"
            ),
            "receipt_created_last": True,
        },
        "offline_validator_contract": {
            "network_access": False,
            "asset_root_required": True,
            "rebuild_protocol_exactly": True,
            "reconstruct_custody_except_frozen_utc_value": True,
            "strict_duplicate_key_rejection": True,
            "canonical_json_required": True,
            "full_kernel_rehash_required_on_every_validation": True,
            "hashes": ["MD5_SOURCE_LINEAGE_ONLY", "SHA256_LOCAL_CONTENT_IDENTITY"],
            "held_fd_and_stable_path_identity_required": True,
            "exact_size_and_eof_required": True,
            "official_metadata_offline_validation": (
                "COMPARE_CUSTODY_RECEIPT_FIELDS_TO_EXACT_FROZEN_PREREGISTRATION_"
                "VALUES_ONLY_NO_SOURCE_BYTE_REEXTRACTION"
            ),
            "official_metadata_source_bytes_retained_or_revalidated_offline": False,
            "spk_semantics_recomputed_from_external_held_kernel": True,
            "read_only": True,
        },
        "implementation": {
            "source": _implementation_record(
                SOURCE_PATH, "tools/jx_tno_de441_part2_acquisition_v1.py"
            ),
            "test": _implementation_record(
                TEST_PATH, "tests/test_tno_de441_part2_acquisition_v1.py"
            ),
            "prepare_imports_network_or_spice_runtime": False,
        },
        "execution_blockers_after_this_protocol": [
            "ANNOTATED_ACQUISITION_PREREGISTRATION_TAG_NOT_YET_BOUND_BY_A_SUCCESSOR",
            "OFFICIAL_DE441_PART2_KERNEL_BYTES_NOT_RETAINED_OR_HASHED_LOCALLY",
            "EXACT_SPICEYPY_CSPICE_RUNTIME_AND_SPK_SEMANTIC_VALIDATOR_NOT_BOUND",
            "PORTABLE_POST_DOWNLOAD_CUSTODY_RECEIPT_NOT_PUBLISHED",
            "DE441_PERSISTENT_SOURCE_EVALUATOR_AND_SOURCE_NODE_GATES_NOT_BOUND",
        ],
        "claims": {
            "prepare_only_protocol_frozen": True,
            "official_source_identity_frozen": True,
            "official_metadata_hashes_frozen": True,
            "external_portable_storage_policy_frozen": True,
            "future_custody_schema_frozen": True,
            "kernel_downloaded": False,
            "kernel_bytes_read": False,
            "kernel_locally_sha256_bound": False,
            "kernel_custody_established": False,
            "spk_semantics_validated": False,
            "source_evaluator_qualified": False,
            "source_node_gate_passed": False,
            "common_epoch_bridge_executed": False,
            "dynamics_executed": False,
            "gpu_executed": False,
            "classification_generated": False,
            "population_inference": False,
            "planet_nine_evidence": False,
            "science_result": False,
            "scientific_execution_authorized": False,
        },
        "nonclaim": (
            "This prepare-only record freezes a future source acquisition. It retains "
            "no DE441 kernel bytes, establishes no local kernel custody, validates no "
            "SPK segment through CSPICE, authorizes no dynamics, and is not evidence "
            "for or against any TNO classification or Planet Nine hypothesis."
        ),
        "publication": {
            "prepared_roster": ["PROTOCOL.json"],
            "root_must_not_exist": True,
            "create_only_no_overwrite": True,
            "directory_mode_octal": "0700",
            "file_mode_octal": "0600",
            "file_and_directory_fsync_required": True,
            "validation_is_read_only": True,
        },
    }
    protocol["content_sha256"] = _content_sha256(protocol)
    return protocol


def _validate_protocol(value: Mapping[str, Any]) -> None:
    if type(value) is not dict or value.get("schema") != PROTOCOL_SCHEMA:
        raise KernelProtocolError("protocol schema differs")
    if value.get("content_sha256") != _content_sha256(value):
        raise KernelProtocolError("protocol content SHA-256 differs")
    claims = value.get("claims")
    authority = value.get("authority")
    storage = value.get("storage_contract")
    if (
        type(claims) is not dict
        or claims.get("kernel_downloaded") is not False
        or claims.get("kernel_custody_established") is not False
        or claims.get("scientific_execution_authorized") is not False
        or type(authority) is not dict
        or authority.get("network_access") is not False
        or authority.get("kernel_bytes_read_or_written") is not False
        or type(storage) is not dict
        or storage.get("absolute_asset_root_recorded_in_protocol_or_custody") is not False
        or storage.get("kernel_never_stored_in_git_worktree") is not True
    ):
        raise KernelProtocolError("prepare-only authority or storage boundary differs")
    if value != build_protocol():
        raise KernelProtocolError("protocol differs from the exact frozen construction")


def _current_branch() -> str:
    return _git("branch", "--show-current").decode("utf-8").strip()


def _require_no_symlink_ancestors(path: Path, label: str) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    for component in (absolute, *absolute.parents):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise KernelProtocolError(f"cannot inspect {label} ancestor") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise KernelProtocolError(f"{label} has a symlink ancestor")


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise KernelProtocolError("protocol write made no progress")
        view = view[written:]


def _rename_noreplace(directory_fd: int, source: str, destination: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise KernelProtocolError("renameat2 is unavailable")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(
        directory_fd,
        source.encode("ascii"),
        directory_fd,
        destination.encode("ascii"),
        1,
    ) != 0:
        observed_errno = ctypes.get_errno()
        if observed_errno == errno.EEXIST:
            raise KernelProtocolError("protocol destination already exists")
        raise KernelProtocolError(
            f"renameat2 failed: {os.strerror(observed_errno)}"
        )


def _validate_prepared_fd(
    root_fd: int,
    expected_payload: bytes,
) -> dict[str, object]:
    root_before = os.fstat(root_fd)
    if (
        not stat.S_ISDIR(root_before.st_mode)
        or stat.S_IMODE(root_before.st_mode) != 0o700
        or root_before.st_uid != os.geteuid()
    ):
        raise KernelProtocolError("prepared root is not an owner-private directory")
    if frozenset(os.listdir(root_fd)) != PREPARED_ROSTER:
        raise KernelProtocolError("prepared root roster differs")
    metadata = os.stat("PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
        or metadata.st_size != len(expected_payload)
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_uid != os.geteuid()
    ):
        raise KernelProtocolError("prepared protocol identity differs")
    descriptor = os.open(
        "PROTOCOL.json",
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        dir_fd=root_fd,
    )
    try:
        before = os.fstat(descriptor)
        if _stat_identity(before) != _stat_identity(metadata):
            raise KernelProtocolError("prepared protocol path identity differs")
        payload = bytearray()
        while len(payload) < before.st_size:
            block = os.pread(
                descriptor,
                min(1 << 20, before.st_size - len(payload)),
                len(payload),
            )
            if not block:
                raise KernelProtocolError("prepared protocol ended early")
            payload.extend(block)
        if os.pread(descriptor, 1, before.st_size):
            raise KernelProtocolError("prepared protocol has trailing bytes")
        after = os.fstat(descriptor)
        path_after = os.stat(
            "PROTOCOL.json", dir_fd=root_fd, follow_symlinks=False
        )
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(before) != _stat_identity(path_after)
        ):
            raise KernelProtocolError("prepared protocol changed while held")
    finally:
        os.close(descriptor)
    if (
        _stat_identity(root_before) != _stat_identity(os.fstat(root_fd))
        or frozenset(os.listdir(root_fd)) != PREPARED_ROSTER
    ):
        raise KernelProtocolError("prepared root changed while held")
    exact_payload = bytes(payload)
    if exact_payload != expected_payload:
        raise KernelProtocolError("prepared protocol bytes differ")
    value = _strict_json(exact_payload)
    if exact_payload != _canonical_json(value):
        raise KernelProtocolError("prepared protocol is not canonical JSON")
    _validate_protocol(value)
    return value


def _cleanup_private_root(root_fd: int) -> None:
    for name in tuple(os.listdir(root_fd)):
        metadata = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
        if not stat.S_ISREG(metadata.st_mode):
            raise KernelProtocolError("failed preparation left a nonregular entry")
        os.unlink(name, dir_fd=root_fd)
    os.fsync(root_fd)


def prepare(output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict[str, object]:
    if _current_branch() != EXPECTED_BRANCH:
        raise KernelProtocolError("preparation branch differs")
    protocol = build_protocol()
    payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    if not root.name or root.name in (".", ".."):
        raise KernelProtocolError("prepared root name differs")
    _require_no_symlink_ancestors(root, "prepared root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        parent_fd = os.open(root.parent, flags)
    except OSError as exc:
        raise KernelProtocolError("cannot open prepared parent") from exc
    created = False
    prepared: dict[str, object] | None = None
    try:
        try:
            os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            created = True
        except FileExistsError as exc:
            raise KernelProtocolError(
                "prepared root already exists; create-only publication refused"
            ) from exc
        root_fd = os.open(root.name, flags, dir_fd=parent_fd)
        try:
            os.fchmod(root_fd, 0o700)
            if os.listdir(root_fd):
                raise KernelProtocolError("new prepared root is not empty")
            partial = ".PROTOCOL.json.partial"
            file_fd = os.open(
                partial,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=root_fd,
            )
            try:
                os.fchmod(file_fd, 0o600)
                _write_all(file_fd, payload)
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            _rename_noreplace(root_fd, partial, "PROTOCOL.json")
            os.fsync(root_fd)
            prepared = _validate_prepared_fd(root_fd, payload)
            os.fsync(parent_fd)
        except BaseException:
            _cleanup_private_root(root_fd)
            raise
        finally:
            os.close(root_fd)
    except BaseException as exc:
        if created:
            try:
                os.rmdir(root.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
            except OSError:
                pass
        if isinstance(exc, OSError) and not isinstance(exc, KernelProtocolError):
            raise KernelProtocolError("cannot create prepared protocol") from exc
        raise
    finally:
        os.close(parent_fd)
    if prepared is None:
        raise KernelProtocolError("prepared protocol validation did not complete")
    return prepared


def validate_prepared(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    protocol = build_protocol()
    expected_payload = _canonical_json(protocol)
    root = Path(os.path.abspath(os.fspath(output_root)))
    _require_no_symlink_ancestors(root, "prepared root")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise KernelProtocolError("cannot open prepared root") from exc
    try:
        value = _validate_prepared_fd(root_fd, expected_payload)
    finally:
        os.close(root_fd)
    if value != protocol:
        raise KernelProtocolError("prepared protocol differs from rebuilt protocol")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build", help="build and print the protocol without writing")
    subparsers.add_parser(
        "audit-prerequisites", help="verify frozen parent custody without writing"
    )
    prepare_parser = subparsers.add_parser(
        "prepare", help="create only a private PROTOCOL.json"
    )
    prepare_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    validate_parser = subparsers.add_parser(
        "validate-prepared", help="read-only validation of a prepared root"
    )
    validate_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        value = build_protocol()
        print(_canonical_json(value).decode("ascii"), end="")
    elif arguments.command == "audit-prerequisites":
        value = audit_prerequisites()
        print(
            json.dumps(
                {
                    "binding_count": value["binding_count"],
                    "parent_commit": value["parent_ancestry"]["commit"],
                    "status": "PASSED",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "prepare":
        value = prepare(arguments.output_root)
        print(
            json.dumps(
                {
                    "content_sha256": value["content_sha256"],
                    "status": value["status"],
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "validate-prepared":
        value = validate_prepared(arguments.output_root)
        print(
            json.dumps(
                {"content_sha256": value["content_sha256"], "status": "VALID"},
                sort_keys=True,
            )
        )
    else:  # pragma: no cover - argparse enforces the exact command set.
        raise KernelProtocolError("unknown command")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
