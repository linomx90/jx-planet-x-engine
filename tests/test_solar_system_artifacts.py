"""Local artifact-byte verification tests for Solar-System Milestone 4B."""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from jxplanetx.solar_system import artifacts as artifacts_module
from jxplanetx.solar_system.artifacts import (
    ArtifactByteMatchReceipt,
    LocalArtifactVerificationReceipt,
    MAXIMUM_LOCAL_ARTIFACT_BYTES,
    validate_local_artifact_verification_receipt,
    verify_local_artifact_bytes,
)
from jxplanetx.solar_system.contracts import (
    ArtifactBinding,
    SolarSystemContractError,
    SolarSystemDataError,
    SolarSystemDependencyUnavailableError,
)
from jxplanetx.solar_system.serialization import canonical_json
from tests.test_solar_system_contracts import SOURCE_ID, external_artifact


EXPECTED_EXPORTS = [
    "ArtifactByteMatchReceipt",
    "LocalArtifactVerificationReceipt",
    "MAXIMUM_LOCAL_ARTIFACT_BYTES",
    "validate_local_artifact_verification_receipt",
    "verify_local_artifact_bytes",
]
BYTE_MATCH_FIELDS = (
    "artifact",
    "verification_scope",
    "hash_algorithm",
    "expected_byte_length",
    "observed_byte_length",
    "expected_sha256",
    "observed_sha256",
    "byte_match_status",
    "content_integrity_class",
    "content_sha256",
)
LOCAL_RECEIPT_FIELDS = (
    "byte_match",
    "verification_stage",
    "platform_scope",
    "root_capability_status",
    "locator_resolution_policy",
    "locator_components",
    "final_open_policy",
    "read_policy",
    "maximum_byte_length",
    "read_chunk_byte_limit",
    "maximum_data_read_calls",
    "data_read_call_count",
    "eof_probe_call_count",
    "root_stat_field_order",
    "pre_root_stat",
    "post_root_stat",
    "file_stat_field_order",
    "pre_open_path_file_stat",
    "pre_read_file_stat",
    "post_read_file_stat",
    "second_resolution_file_stat",
    "metadata_stability_status",
    "second_resolution_status",
    "local_metadata_classification",
    "custody_scope",
    "execution_evidence_status",
    "semantic_replay_status",
    "content_integrity_class",
    "content_sha256",
)

BYTE_SCOPE = "EXACT_DECLARED_ARTIFACT_BYTES_ONE_READ_STREAM"
HASH_ALGORITHM = "SHA-256"
BYTE_STATUS = "OBSERVED_LENGTH_AND_SHA256_EQUAL_DECLARATION"
INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"
VERIFICATION_STAGE = "LOCAL_BYTE_STREAM_VERIFICATION_NO_PROVIDER_EXECUTION"
PLATFORM_SCOPE = "CAPABILITY_GATED_POSIX_PROFILE_NO_PORTABILITY_CLAIM"
ROOT_STATUS = "CALLER_DIRECTORY_FD_DUPLICATED_ROOT_PATH_ACQUISITION_UNATTESTED"
LOCATOR_POLICY = "SEQUENTIAL_COMPONENTWISE_DIR_FD_O_NOFOLLOW"
FINAL_OPEN_POLICY = "ONE_FINAL_OPEN_DESCRIPTION_READ_ONLY_NONBLOCKING_NOCTTY"
READ_POLICY = "BOUNDED_OS_READ_DECLARED_LENGTH_PLUS_ONE_EOF_PROBE"
METADATA_STATUS = (
    "SELECTED_PREOPEN_PRE_READ_POST_READ_AND_SECOND_STAT_FIELDS_IDENTICAL"
)
SECOND_STATUS = (
    "SECOND_NAMESPACE_OBSERVATION_MATCHED_OPEN_FILE_IDENTITY_NOT_CONTINUITY_PROOF"
)
LOCAL_METADATA = "LOCAL_NONPORTABLE_OBSERVATION_NOT_CONTENT_IDENTITY"
CUSTODY_SCOPE = "READ_STREAM_ONLY_NO_FUTURE_PATH_OR_FD_USE_AUTHORITY"
EXECUTION_STATUS = (
    "LOCAL_VERIFIER_EXECUTION_ONLY_NO_PROVIDER_AT_USE_OR_PROVIDER_EXECUTION"
)
REPLAY_STATUS = "REQUIRES_FRESH_VERIFICATION_WITH_NEW_ROOT_CAPABILITY"
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

BYTE_DOMAIN = (
    "jxplanetx.solar-system.artifacts.artifact-byte-match-receipt."
    "content-integrity.v1"
)
BYTE_SCHEMA = "ArtifactByteMatchReceipt.v1"
LOCAL_DOMAIN = (
    "jxplanetx.solar-system.artifacts.local-artifact-verification-receipt."
    "content-integrity.v1"
)
LOCAL_SCHEMA = "LocalArtifactVerificationReceipt.v1"

LITERAL_BYTES = b"jx-m4b-literal\n"


def make_artifact(
    data: bytes = LITERAL_BYTES,
    *,
    locator: str = "nested/payload.bin",
    digest: str | None = None,
    byte_length: int | None = None,
) -> ArtifactBinding:
    base = external_artifact(artifact_id=SOURCE_ID, role="SOFTWARE_SOURCE")
    return dataclasses.replace(
        base,
        logical_locator=locator,
        byte_length=len(data) if byte_length is None else byte_length,
        artifact_sha256=hashlib.sha256(data).hexdigest() if digest is None else digest,
        content_sha256="",
    )


def make_external_reference_artifact() -> ArtifactBinding:
    base = external_artifact(
        artifact_id="artifact.fixture.external",
        role="CONSTANTS",
    )
    return dataclasses.replace(
        base,
        logical_locator="https://example.invalid/constants",
        locator_kind="EXTERNAL_REFERENCE_ONLY",
        content_sha256="",
    )


def write_artifact(root: Path, artifact: ArtifactBinding, data: bytes) -> Path:
    path = root.joinpath(*artifact.logical_locator.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def open_root(root: Path) -> int:
    return os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)


def literal_receipts() -> tuple[
    ArtifactByteMatchReceipt,
    LocalArtifactVerificationReceipt,
]:
    artifact = make_artifact()
    byte_match = ArtifactByteMatchReceipt(
        artifact=artifact,
        verification_scope=BYTE_SCOPE,
        hash_algorithm=HASH_ALGORITHM,
        expected_byte_length=len(LITERAL_BYTES),
        observed_byte_length=len(LITERAL_BYTES),
        expected_sha256=artifact.artifact_sha256,
        observed_sha256=artifact.artifact_sha256,
        byte_match_status=BYTE_STATUS,
        content_integrity_class=INTEGRITY_CLASS,
    )
    root_stat = (2_049, 77, stat.S_IFDIR | 0o755)
    file_stat = (
        2_049,
        88,
        stat.S_IFREG | 0o644,
        1,
        1_000,
        1_000,
        len(LITERAL_BYTES),
        1_700_000_000_000_000_000,
        1_700_000_000_100_000_000,
    )
    local = LocalArtifactVerificationReceipt(
        byte_match=byte_match,
        verification_stage=VERIFICATION_STAGE,
        platform_scope=PLATFORM_SCOPE,
        root_capability_status=ROOT_STATUS,
        locator_resolution_policy=LOCATOR_POLICY,
        locator_components=("nested", "payload.bin"),
        final_open_policy=FINAL_OPEN_POLICY,
        read_policy=READ_POLICY,
        maximum_byte_length=MAXIMUM_LOCAL_ARTIFACT_BYTES,
        read_chunk_byte_limit=1 << 20,
        maximum_data_read_calls=8_192,
        data_read_call_count=1,
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
        execution_evidence_status=EXECUTION_STATUS,
        semantic_replay_status=REPLAY_STATUS,
        content_integrity_class=INTEGRITY_CLASS,
    )
    return byte_match, local


def exact_posix_profile_available() -> bool:
    required = ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK", "O_NOCTTY")
    return (
        os.name == "posix"
        and all(type(getattr(os, name, None)) is int and getattr(os, name) for name in required)
        and os.open in getattr(os, "supports_dir_fd", ())
        and os.stat in getattr(os, "supports_dir_fd", ())
        and os.stat in getattr(os, "supports_follow_symlinks", ())
    )


POSIX_PROFILE = exact_posix_profile_available()


class ArtifactReceiptSchemaTests(unittest.TestCase):
    def test_exact_public_roster_fields_and_signatures(self) -> None:
        self.assertEqual(artifacts_module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(artifacts_module.__all__, sorted(artifacts_module.__all__))
        self.assertEqual(MAXIMUM_LOCAL_ARTIFACT_BYTES, 1 << 30)
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(ArtifactByteMatchReceipt)),
            BYTE_MATCH_FIELDS,
        )
        self.assertEqual(
            tuple(
                field.name
                for field in dataclasses.fields(LocalArtifactVerificationReceipt)
            ),
            LOCAL_RECEIPT_FIELDS,
        )
        self.assertEqual(
            tuple(inspect.signature(verify_local_artifact_bytes).parameters),
            ("artifact", "root_directory_fd"),
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    validate_local_artifact_verification_receipt
                ).parameters
            ),
            ("value",),
        )
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_exact_status_roster_and_stat_orders(self) -> None:
        byte_match, local = literal_receipts()
        self.assertEqual(byte_match.verification_scope, BYTE_SCOPE)
        self.assertEqual(byte_match.hash_algorithm, HASH_ALGORITHM)
        self.assertEqual(byte_match.byte_match_status, BYTE_STATUS)
        self.assertEqual(local.verification_stage, VERIFICATION_STAGE)
        self.assertEqual(local.platform_scope, PLATFORM_SCOPE)
        self.assertEqual(local.root_capability_status, ROOT_STATUS)
        self.assertEqual(local.locator_resolution_policy, LOCATOR_POLICY)
        self.assertEqual(local.final_open_policy, FINAL_OPEN_POLICY)
        self.assertEqual(local.read_policy, READ_POLICY)
        self.assertEqual(local.root_stat_field_order, ROOT_STAT_FIELDS)
        self.assertEqual(local.file_stat_field_order, FILE_STAT_FIELDS)
        self.assertEqual(local.metadata_stability_status, METADATA_STATUS)
        self.assertEqual(local.second_resolution_status, SECOND_STATUS)
        self.assertEqual(local.local_metadata_classification, LOCAL_METADATA)
        self.assertEqual(local.custody_scope, CUSTODY_SCOPE)
        self.assertEqual(local.execution_evidence_status, EXECUTION_STATUS)
        self.assertEqual(local.semantic_replay_status, REPLAY_STATUS)
        self.assertEqual(local.content_integrity_class, INTEGRITY_CLASS)


class ArtifactReceiptIntegrityTests(unittest.TestCase):
    def test_literal_production_preimages_and_hashes(self) -> None:
        byte_match, local = literal_receipts()
        byte_preimage = (
            BYTE_DOMAIN.encode("ascii")
            + b"\x00"
            + BYTE_SCHEMA.encode("ascii")
            + b"\x00"
            + canonical_json(artifacts_module._byte_match_payload(byte_match))
        )
        local_preimage = (
            LOCAL_DOMAIN.encode("ascii")
            + b"\x00"
            + LOCAL_SCHEMA.encode("ascii")
            + b"\x00"
            + canonical_json(artifacts_module._local_receipt_payload(local))
        )
        self.assertEqual(byte_preimage, BYTE_LITERAL_PREIMAGE)
        self.assertEqual(len(byte_preimage), BYTE_LITERAL_LENGTH)
        self.assertEqual(hashlib.sha256(byte_preimage).hexdigest(), BYTE_LITERAL_SHA256)
        self.assertEqual(byte_match.content_sha256, BYTE_LITERAL_SHA256)
        self.assertEqual(local_preimage, LOCAL_LITERAL_PREIMAGE)
        self.assertEqual(len(local_preimage), LOCAL_LITERAL_LENGTH)
        self.assertEqual(hashlib.sha256(local_preimage).hexdigest(), LOCAL_LITERAL_SHA256)
        self.assertEqual(local.content_sha256, LOCAL_LITERAL_SHA256)

    def test_outer_payload_expands_and_binds_the_sealed_child(self) -> None:
        byte_match, local = literal_receipts()
        payload = canonical_json(artifacts_module._local_receipt_payload(local))
        self.assertIn(byte_match.content_sha256.encode("ascii"), payload)
        self.assertNotIn(local.content_sha256.encode("ascii"), payload)
        self.assertNotIn(
            byte_match.content_sha256.encode("ascii"),
            canonical_json(artifacts_module._byte_match_payload(byte_match)),
        )

    def test_frozen_m1_serializer_rejects_both_new_receipts_directly(self) -> None:
        byte_match, local = literal_receipts()
        with self.assertRaises(SolarSystemContractError):
            canonical_json(byte_match)
        with self.assertRaises(SolarSystemContractError):
            canonical_json(local)

    def test_domain_schema_field_order_and_metadata_mutations_change_hash(self) -> None:
        mutations = (
            BYTE_LITERAL_PREIMAGE.replace(BYTE_DOMAIN.encode(), b"mutated.byte.domain", 1),
            BYTE_LITERAL_PREIMAGE.replace(BYTE_SCHEMA.encode(), b"MutatedByte.v1", 1),
            BYTE_LITERAL_PREIMAGE.replace(b'[["artifact",', b'[["hash_algorithm",', 1),
            LOCAL_LITERAL_PREIMAGE.replace(LOCAL_DOMAIN.encode(), b"mutated.local.domain", 1),
            LOCAL_LITERAL_PREIMAGE.replace(LOCAL_SCHEMA.encode(), b"MutatedLocal.v1", 1),
            LOCAL_LITERAL_PREIMAGE.replace(b'[["byte_match",', b'[["platform_scope",', 1),
            LOCAL_LITERAL_PREIMAGE.replace(b"1700000000000000000", b"1700000000000000001", 1),
        )
        for mutation in mutations:
            with self.subTest(length=len(mutation)):
                self.assertNotEqual(
                    hashlib.sha256(mutation).hexdigest(),
                    BYTE_LITERAL_SHA256,
                )
                self.assertNotEqual(
                    hashlib.sha256(mutation).hexdigest(),
                    LOCAL_LITERAL_SHA256,
                )

    def test_child_and_outer_seals_are_preflighted_before_nested_traversal(self) -> None:
        byte_match, local = literal_receipts()
        byte_values = {
            field.name: getattr(byte_match, field.name)
            for field in dataclasses.fields(ArtifactByteMatchReceipt)
            if field.name != "content_sha256"
        }
        with mock.patch.object(
            artifacts_module,
            "_byte_match_digest",
            side_effect=AssertionError("byte digest traversal reached"),
        ) as byte_digest:
            with self.assertRaises(SolarSystemContractError):
                ArtifactByteMatchReceipt(**byte_values, content_sha256=False)  # type: ignore[arg-type]
            byte_digest.assert_not_called()

        local_values = {
            field.name: getattr(local, field.name)
            for field in dataclasses.fields(LocalArtifactVerificationReceipt)
            if field.name != "content_sha256"
        }
        with mock.patch.object(
            artifacts_module,
            "_local_receipt_digest",
            side_effect=AssertionError("outer digest traversal reached"),
        ) as local_digest:
            with self.assertRaises(SolarSystemContractError):
                LocalArtifactVerificationReceipt(
                    **local_values,
                    content_sha256="not-a-sha",
                )
            local_digest.assert_not_called()

    def test_every_byte_match_field_and_nested_artifact_seal_are_binding(self) -> None:
        alternatives: dict[str, object] = {
            "artifact": make_artifact(b"other bytes\n"),
            "verification_scope": "OTHER_SCOPE",
            "hash_algorithm": "SHA-512",
            "expected_byte_length": len(LITERAL_BYTES) + 1,
            "observed_byte_length": len(LITERAL_BYTES) + 1,
            "expected_sha256": "1" * 64,
            "observed_sha256": "1" * 64,
            "byte_match_status": "OTHER_STATUS",
            "content_integrity_class": "OTHER_INTEGRITY",
        }
        for field_name, replacement in alternatives.items():
            byte_match, _ = literal_receipts()
            object.__setattr__(byte_match, field_name, replacement)
            with self.subTest(field_name=field_name):
                with self.assertRaises(SolarSystemContractError):
                    byte_match.validate_integrity()

        byte_match, _ = literal_receipts()
        object.__setattr__(byte_match.artifact, "artifact_sha256", "1" * 64)
        with self.assertRaises(SolarSystemContractError):
            byte_match.validate_integrity()

    def test_every_outer_field_and_stat_observation_is_binding(self) -> None:
        _, reference = literal_receipts()
        stale_child = literal_receipts()[0]
        object.__setattr__(stale_child, "content_sha256", "0" * 64)
        alternatives: dict[str, object] = {
            "byte_match": stale_child,
            "verification_stage": "OTHER_STAGE",
            "platform_scope": "OTHER_PLATFORM",
            "root_capability_status": "OTHER_ROOT",
            "locator_resolution_policy": "OTHER_LOCATOR",
            "locator_components": ("other", "payload.bin"),
            "final_open_policy": "OTHER_OPEN",
            "read_policy": "OTHER_READ",
            "maximum_byte_length": MAXIMUM_LOCAL_ARTIFACT_BYTES - 1,
            "read_chunk_byte_limit": (1 << 20) - 1,
            "maximum_data_read_calls": 8_191,
            "data_read_call_count": 0,
            "eof_probe_call_count": 0,
            "root_stat_field_order": tuple(reversed(ROOT_STAT_FIELDS)),
            "pre_root_stat": (2_049, 78, stat.S_IFDIR | 0o755),
            "post_root_stat": (2_049, 78, stat.S_IFDIR | 0o755),
            "file_stat_field_order": tuple(reversed(FILE_STAT_FIELDS)),
            "pre_open_path_file_stat": tuple(
                value + 1 if index == 1 else value
                for index, value in enumerate(reference.pre_open_path_file_stat)
            ),
            "pre_read_file_stat": tuple(
                value + 1 if index == 2 else value
                for index, value in enumerate(reference.pre_read_file_stat)
            ),
            "post_read_file_stat": tuple(
                value + 1 if index == 3 else value
                for index, value in enumerate(reference.post_read_file_stat)
            ),
            "second_resolution_file_stat": tuple(
                value + 1 if index == 4 else value
                for index, value in enumerate(reference.second_resolution_file_stat)
            ),
            "metadata_stability_status": "OTHER_METADATA",
            "second_resolution_status": "OTHER_SECOND",
            "local_metadata_classification": "OTHER_LOCAL_METADATA",
            "custody_scope": "OTHER_CUSTODY",
            "execution_evidence_status": "OTHER_EXECUTION",
            "semantic_replay_status": "OTHER_REPLAY",
            "content_integrity_class": "OTHER_INTEGRITY",
        }
        for field_name, replacement in alternatives.items():
            _, local = literal_receipts()
            object.__setattr__(local, field_name, replacement)
            with self.subTest(field_name=field_name):
                with self.assertRaises(SolarSystemContractError):
                    validate_local_artifact_verification_receipt(local)

    def test_exact_types_subclasses_bool_and_stat_caps_reject(self) -> None:
        byte_match, local = literal_receipts()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(byte_match, expected_byte_length=True, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(local, data_read_call_count=True, content_sha256="")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                local,
                data_read_call_count=local.byte_match.observed_byte_length + 1,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(local, pre_root_stat=list(local.pre_root_stat), content_sha256="")
        oversized_stat = (1 << 128,) + local.pre_root_stat[1:]
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(local, pre_root_stat=oversized_stat, content_sha256="")
        oversized_negative_stat = (-(1 << 128),) + local.pre_root_stat[1:]
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                local,
                pre_root_stat=oversized_negative_stat,
                content_sha256="",
            )

        class ByteSubclass(ArtifactByteMatchReceipt):
            pass

        class LocalSubclass(LocalArtifactVerificationReceipt):
            pass

        byte_values = {
            field.name: getattr(byte_match, field.name)
            for field in dataclasses.fields(ArtifactByteMatchReceipt)
            if field.name != "content_sha256"
        }
        local_values = {
            field.name: getattr(local, field.name)
            for field in dataclasses.fields(LocalArtifactVerificationReceipt)
            if field.name != "content_sha256"
        }
        with self.assertRaises(SolarSystemContractError):
            ByteSubclass(**byte_values)
        with self.assertRaises(SolarSystemContractError):
            LocalSubclass(**local_values)
        with self.assertRaises(SolarSystemContractError):
            validate_local_artifact_verification_receipt(object())  # type: ignore[arg-type]

    def test_root_and_file_cannot_share_one_live_inode_identity(self) -> None:
        _, local = literal_receipts()
        file_stat = local.pre_read_file_stat
        impossible_root = (file_stat[0], file_stat[1], stat.S_IFDIR | 0o755)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                local,
                pre_root_stat=impossible_root,
                post_root_stat=impossible_root,
                content_sha256="",
            )

    def test_hostile_equality_objects_reject_before_any_comparison(self) -> None:
        class EqualityBomb:
            def __eq__(self, other: object) -> bool:
                raise RuntimeError("hostile equality reached")

        _, local = literal_receipts()
        for replacement in (
            {"locator_components": (EqualityBomb(), "payload.bin")},
            {"root_stat_field_order": (EqualityBomb(), "st_ino", "st_mode")},
            {
                "file_stat_field_order": (
                    EqualityBomb(),
                    *FILE_STAT_FIELDS[1:],
                )
            },
        ):
            with self.subTest(field=next(iter(replacement))):
                with self.assertRaises(SolarSystemContractError):
                    dataclasses.replace(local, **replacement, content_sha256="")

    def test_runtime_stat_values_are_not_coerced(self) -> None:
        class Convertible:
            def __int__(self) -> int:
                raise RuntimeError("hostile integer conversion reached")

        class RuntimeStat:
            st_dev = Convertible()
            st_ino = 2
            st_mode = stat.S_IFDIR | 0o755

        with self.assertRaises(SolarSystemDataError):
            artifacts_module._selected_stat(RuntimeStat(), ROOT_STAT_FIELDS)

        RuntimeStat.st_dev = 1.0
        with self.assertRaises(SolarSystemDataError):
            artifacts_module._selected_stat(RuntimeStat(), ROOT_STAT_FIELDS)

        RuntimeStat.st_dev = True
        with self.assertRaises(SolarSystemDataError):
            artifacts_module._selected_stat(RuntimeStat(), ROOT_STAT_FIELDS)


@unittest.skipUnless(POSIX_PROFILE, "exact POSIX capability profile unavailable")
class LiveArtifactVerificationTests(unittest.TestCase):
    def test_success_binds_one_stream_four_stats_and_preserves_caller_fd(self) -> None:
        data = b"bounded local artifact bytes\n"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, artifact, data)
            root_fd = open_root(root)
            try:
                receipt = verify_local_artifact_bytes(artifact, root_fd)
                os.fstat(root_fd)
            finally:
                os.close(root_fd)
        validate_local_artifact_verification_receipt(receipt)
        self.assertEqual(receipt.byte_match.artifact.content_sha256, artifact.content_sha256)
        self.assertEqual(receipt.byte_match.observed_byte_length, len(data))
        self.assertEqual(receipt.byte_match.observed_sha256, hashlib.sha256(data).hexdigest())
        self.assertEqual(receipt.locator_components, ("nested", "payload.bin"))
        self.assertEqual(receipt.data_read_call_count, 1)
        self.assertEqual(receipt.eof_probe_call_count, 1)
        self.assertEqual(receipt.pre_root_stat, receipt.post_root_stat)
        self.assertEqual(
            receipt.pre_open_path_file_stat,
            receipt.pre_read_file_stat,
        )
        self.assertEqual(receipt.pre_read_file_stat, receipt.post_read_file_stat)
        self.assertEqual(
            receipt.post_read_file_stat,
            receipt.second_resolution_file_stat,
        )

    def test_same_bytes_different_roots_share_byte_match_not_local_metadata(self) -> None:
        data = b"same declared bytes\n"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            roots = (base / "one", base / "two")
            receipts: list[LocalArtifactVerificationReceipt] = []
            for root in roots:
                root.mkdir()
                write_artifact(root, artifact, data)
                root_fd = open_root(root)
                try:
                    receipts.append(verify_local_artifact_bytes(artifact, root_fd))
                finally:
                    os.close(root_fd)
        self.assertEqual(
            receipts[0].byte_match.content_sha256,
            receipts[1].byte_match.content_sha256,
        )
        self.assertNotEqual(receipts[0].pre_root_stat, receipts[1].pre_root_stat)
        self.assertNotEqual(receipts[0].content_sha256, receipts[1].content_sha256)

    def test_schema_validation_never_touches_the_filesystem(self) -> None:
        _, receipt = literal_receipts()
        with (
            mock.patch.object(artifacts_module.os, "open", side_effect=AssertionError),
            mock.patch.object(artifacts_module.os, "stat", side_effect=AssertionError),
            mock.patch.object(artifacts_module.os, "fstat", side_effect=AssertionError),
            mock.patch.object(artifacts_module.os, "read", side_effect=AssertionError),
            mock.patch.object(artifacts_module.os, "dup", side_effect=AssertionError),
        ):
            validate_local_artifact_verification_receipt(receipt)

    def test_partial_reads_are_bounded_counted_and_exact(self) -> None:
        data = bytes(range(251)) * 5
        artifact = make_artifact(data)
        real_read = os.read

        def partial_read(file_fd: int, requested: int) -> bytes:
            return real_read(file_fd, min(requested, 7))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, artifact, data)
            root_fd = open_root(root)
            try:
                with mock.patch.object(artifacts_module.os, "read", partial_read):
                    receipt = verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)
        self.assertEqual(receipt.data_read_call_count, (len(data) + 6) // 7)
        self.assertEqual(receipt.byte_match.observed_sha256, hashlib.sha256(data).hexdigest())

    def test_early_eof_extra_eof_byte_and_read_call_exhaustion_reject(self) -> None:
        data = b"12345678"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, artifact, data)
            root_fd = open_root(root)
            try:
                with mock.patch.object(artifacts_module.os, "read", return_value=b""):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)

                real_read = os.read
                calls = 0

                def extra_at_eof(file_fd: int, requested: int) -> bytes:
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        return b"x"
                    return real_read(file_fd, requested)

                with mock.patch.object(artifacts_module.os, "read", extra_at_eof):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

        cap_data = b"x" * 8_193
        cap_artifact = make_artifact(cap_data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, cap_artifact, cap_data)
            root_fd = open_root(root)
            real_read = os.read

            def one_byte(file_fd: int, requested: int) -> bytes:
                return real_read(file_fd, min(1, requested))

            try:
                with mock.patch.object(artifacts_module.os, "read", one_byte):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(cap_artifact, root_fd)
            finally:
                os.close(root_fd)

    def test_size_and_hash_mismatches_reject(self) -> None:
        data = b"declared bytes"
        wrong_size = make_artifact(data, byte_length=len(data) + 1)
        wrong_hash = make_artifact(data, digest="1" * 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root_fd = open_root(root)
            try:
                write_artifact(root, wrong_size, data)
                with self.assertRaises(SolarSystemDataError):
                    verify_local_artifact_bytes(wrong_size, root_fd)
                write_artifact(root, wrong_hash, data)
                with self.assertRaises(SolarSystemDataError):
                    verify_local_artifact_bytes(wrong_hash, root_fd)
            finally:
                os.close(root_fd)

    def test_runtime_zero_link_count_is_a_data_error_before_receipt_construction(self) -> None:
        data = b"runtime stat profile"
        artifact = make_artifact(data)
        original_selected = artifacts_module._selected_stat

        def zero_link_count(
            value: os.stat_result,
            order: tuple[str, ...],
        ) -> tuple[int, ...]:
            selected = original_selected(value, order)
            if order == FILE_STAT_FIELDS:
                return selected[:3] + (0,) + selected[4:]
            return selected

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, artifact, data)
            root_fd = open_root(root)
            try:
                with mock.patch.object(
                    artifacts_module,
                    "_selected_stat",
                    zero_link_count,
                ):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

    def test_runtime_root_file_inode_collision_is_a_data_error(self) -> None:
        data = b"runtime inode collision"
        artifact = make_artifact(data)
        original_selected = artifacts_module._selected_stat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_artifact(root, artifact, data)
            file_observation = path.stat()
            root_fd = open_root(root)

            def collide_root_with_file(
                value: os.stat_result,
                order: tuple[str, ...],
            ) -> tuple[int, ...]:
                selected = original_selected(value, order)
                if order == ROOT_STAT_FIELDS:
                    return (
                        file_observation.st_dev,
                        file_observation.st_ino,
                        selected[2],
                    )
                return selected

            try:
                with mock.patch.object(
                    artifacts_module,
                    "_selected_stat",
                    collide_root_with_file,
                ):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

    def test_intermediate_and_final_symlinks_reject(self) -> None:
        data = b"symlink target"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "real").mkdir()
            (root / "real" / "payload.bin").write_bytes(data)
            (root / "linked").symlink_to("real", target_is_directory=True)
            intermediate = make_artifact(data, locator="linked/payload.bin")
            (root / "leaf-link.bin").symlink_to("real/payload.bin")
            final = make_artifact(data, locator="leaf-link.bin")
            root_fd = open_root(root)
            try:
                with self.assertRaises(SolarSystemDataError):
                    verify_local_artifact_bytes(intermediate, root_fd)
                with self.assertRaises(SolarSystemDataError):
                    verify_local_artifact_bytes(final, root_fd)
            finally:
                os.close(root_fd)

    def test_directory_fifo_and_socket_are_rejected_without_blocking(self) -> None:
        data = b"nonregular declaration"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "as-directory").mkdir()
            os.mkfifo(root / "as-fifo")
            unix_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            locators = ["as-directory", "as-fifo"]
            try:
                unix_socket.bind(str(root / "as-socket"))
            except PermissionError:
                unix_socket.close()
                unix_socket = None
            else:
                locators.append("as-socket")
            root_fd = open_root(root)
            platform_flags = artifacts_module._require_platform_capabilities()
            try:
                with (
                    mock.patch.object(
                        artifacts_module,
                        "_require_platform_capabilities",
                        return_value=platform_flags,
                    ),
                    mock.patch.object(
                        artifacts_module.os,
                        "open",
                        side_effect=AssertionError("nonregular leaf was opened"),
                    ) as leaf_open,
                ):
                    for locator in locators:
                        with self.subTest(locator=locator):
                            with self.assertRaises(SolarSystemDataError):
                                verify_local_artifact_bytes(
                                    make_artifact(data, locator=locator),
                                    root_fd,
                                )
                    leaf_open.assert_not_called()
            finally:
                os.close(root_fd)
                if unix_socket is not None:
                    unix_socket.close()

    def test_hardlinks_are_allowed_but_link_count_is_retained(self) -> None:
        data = b"hardlinked declared bytes"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_artifact(root, artifact, data)
            os.link(path, root / "second-link.bin")
            root_fd = open_root(root)
            try:
                receipt = verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)
        self.assertGreaterEqual(receipt.pre_read_file_stat[3], 2)

    def test_classification_open_race_in_place_mutation_and_replacement_reject(self) -> None:
        data = b"race-observed-bytes"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_artifact(root, artifact, data)
            replacement = root / "replacement.bin"
            replacement.write_bytes(b"R" * len(data))
            self.assertEqual(path.stat().st_size, replacement.stat().st_size)
            root_fd = open_root(root)
            original_selected = artifacts_module._selected_stat
            selected_calls = 0

            def swap_after_preopen(value: os.stat_result, order: tuple[str, ...]) -> tuple[int, ...]:
                nonlocal selected_calls
                result = original_selected(value, order)
                selected_calls += 1
                if selected_calls == 2:
                    os.replace(replacement, path)
                return result

            try:
                with mock.patch.object(
                    artifacts_module,
                    "_selected_stat",
                    swap_after_preopen,
                ):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_artifact(root, artifact, data)
            root_fd = open_root(root)
            original_read = artifacts_module._read_declared_bytes

            def mutate_before_read(file_fd: int, declared: int) -> tuple[int, str]:
                path.write_bytes(b"M" * len(data))
                return original_read(file_fd, declared)

            try:
                with mock.patch.object(
                    artifacts_module,
                    "_read_declared_bytes",
                    mutate_before_read,
                ):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_artifact(root, artifact, data)
            root_fd = open_root(root)
            original_walk = artifacts_module._walk_parent_directory
            walk_calls = 0

            def replace_before_second(
                root_fd_value: int,
                components: tuple[str, ...],
                flags: int,
                stack: object,
            ) -> int:
                nonlocal walk_calls
                walk_calls += 1
                if walk_calls == 2:
                    old = root / "old-payload.bin"
                    os.replace(path, old)
                    path.write_bytes(data)
                return original_walk(root_fd_value, components, flags, stack)  # type: ignore[arg-type]

            try:
                with mock.patch.object(
                    artifacts_module,
                    "_walk_parent_directory",
                    replace_before_second,
                ):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

    def test_final_file_description_remains_open_during_second_resolution(self) -> None:
        data = b"open-description-lifetime"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, artifact, data)
            root_fd = open_root(root)
            original_read = artifacts_module._read_declared_bytes
            original_walk = artifacts_module._walk_parent_directory
            captured_file_fd = -1
            walk_calls = 0
            observed_open_during_second_walk = False

            def capture_file_fd(file_fd: int, declared: int) -> tuple[int, str]:
                nonlocal captured_file_fd
                captured_file_fd = file_fd
                return original_read(file_fd, declared)

            def require_live_fd_during_second_walk(
                root_fd_value: int,
                components: tuple[str, ...],
                flags: int,
                stack: object,
            ) -> int:
                nonlocal walk_calls, observed_open_during_second_walk
                walk_calls += 1
                if walk_calls == 2:
                    os.fstat(captured_file_fd)
                    observed_open_during_second_walk = True
                return original_walk(root_fd_value, components, flags, stack)  # type: ignore[arg-type]

            try:
                with (
                    mock.patch.object(
                        artifacts_module,
                        "_read_declared_bytes",
                        capture_file_fd,
                    ),
                    mock.patch.object(
                        artifacts_module,
                        "_walk_parent_directory",
                        require_live_fd_during_second_walk,
                    ),
                ):
                    receipt = verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

        self.assertEqual(receipt.byte_match.observed_sha256, artifact.artifact_sha256)
        self.assertTrue(observed_open_during_second_walk)

    def test_root_and_owned_fd_failures_are_normalized_without_leaks(self) -> None:
        data = b"fd lifecycle"
        artifact = make_artifact(data)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_artifact(root, artifact, data)
            closed_fd = open_root(root)
            os.close(closed_fd)
            with self.assertRaises(SolarSystemDataError):
                verify_local_artifact_bytes(artifact, closed_fd)

            file_fd = os.open(root / "nested" / "payload.bin", os.O_RDONLY)
            try:
                with self.assertRaises(SolarSystemDataError):
                    verify_local_artifact_bytes(artifact, file_fd)
            finally:
                os.close(file_fd)

            root_fd = open_root(root)
            try:
                with mock.patch.object(artifacts_module.os, "dup", side_effect=OverflowError):
                    with self.assertRaises(SolarSystemDataError):
                        verify_local_artifact_bytes(artifact, root_fd)
            finally:
                os.close(root_fd)

        if Path("/proc/self/fd").is_dir():
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_artifact(root, artifact, data)
                root_fd = open_root(root)
                before = len(os.listdir("/proc/self/fd"))
                try:
                    verify_local_artifact_bytes(artifact, root_fd)
                    self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                    with mock.patch.object(
                        artifacts_module.os,
                        "read",
                        side_effect=OSError("injected"),
                    ):
                        with self.assertRaises(SolarSystemDataError):
                            verify_local_artifact_bytes(artifact, root_fd)
                    self.assertEqual(len(os.listdir("/proc/self/fd")), before)
                finally:
                    os.close(root_fd)


class ArtifactVerificationPreflightAndBoundaryTests(unittest.TestCase):
    def test_contract_profile_caps_reject_before_dup_or_open(self) -> None:
        oversized = make_artifact(
            b"x",
            byte_length=MAXIMUM_LOCAL_ARTIFACT_BYTES + 1,
        )
        too_deep = make_artifact(
            b"x",
            locator="/".join(("a",) * 64 + ("payload.bin",)),
        )
        external = make_external_reference_artifact()
        oversized_component = make_artifact(
            b"x",
            locator=f"{'a' * 257}/payload.bin",
        )
        cases = (
            (oversized, 0),
            (too_deep, 0),
            (oversized_component, 0),
            (external, 0),
            (make_artifact(), False),
            (make_artifact(), -1),
            (make_artifact(), 1 << 31),
        )
        for artifact, root_fd in cases:
            with self.subTest(locator=artifact.logical_locator, root_fd=root_fd):
                with mock.patch.object(
                    artifacts_module.os,
                    "dup",
                    side_effect=AssertionError("OS access reached"),
                ) as duplicate:
                    with self.assertRaises(SolarSystemContractError):
                        verify_local_artifact_bytes(artifact, root_fd)  # type: ignore[arg-type]
                    duplicate.assert_not_called()

    def test_missing_posix_profile_rejects_before_dup(self) -> None:
        artifact = make_artifact()
        with (
            mock.patch.object(artifacts_module.os, "name", "nt"),
            mock.patch.object(
                artifacts_module.os,
                "dup",
                side_effect=AssertionError("dup reached"),
            ) as duplicate,
        ):
            with self.assertRaises(SolarSystemDependencyUnavailableError):
                verify_local_artifact_bytes(artifact, 0)
            duplicate.assert_not_called()

    def test_clean_import_is_optional_dependency_and_network_client_free(self) -> None:
        code = """
import json, sys
import jxplanetx.solar_system.artifacts as module
from jxplanetx import solar_system
forbidden = [
    name for name in (
        'numpy', 'astropy', 'erfa', 'jplephem', 'spiceypy', 'skyfield',
        'rebound', 'socket', 'urllib.request', 'http.client', 'requests'
    ) if name in sys.modules
]
print(json.dumps({
    'exports': module.__all__,
    'root_exports': solar_system.__all__,
    'forbidden': forbidden,
}))
"""
        environment = dict(os.environ)
        source = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        environment["PYTHONPATH"] = source
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["exports"], EXPECTED_EXPORTS)
        self.assertEqual(payload["root_exports"], [])
        self.assertEqual(payload["forbidden"], [])

    def test_module_nonclaims_and_scope_are_explicit(self) -> None:
        text = " ".join(artifacts_module.__doc__.lower().split())
        for phrase in (
            "not an atomic filesystem snapshot",
            "continuous pathname or byte immutability",
            "adversarial-toctou",
            "hard-link/mount/bind provenance",
            "hostile device-node namespaces",
            "may update access metadata",
            "atime is therefore deliberately absent",
            "do not bound latency or guarantee termination",
            "may still block",
            "only a later namespace observation",
            "future path or descriptor use",
            "provider consumption or execution",
            "artifact-at-use custody",
            "authenticity or signature",
            "license or redistribution authority",
            "spk validity or coverage",
            "interpolation accuracy",
            "network/fallback/extrapolation enforcement",
            "forgeable unauthenticated content integrity only",
        ):
            self.assertIn(phrase, text)


# Locked after both production schemas and the fixed literal fixture compile.
BYTE_LITERAL_PREIMAGE = base64.b64decode(
    b"anhwbGFuZXR4LnNvbGFyLXN5c3RlbS5hcnRpZmFjdHMuYXJ0aWZhY3QtYnl0ZS1tYXRjaC1yZWNlaXB0LmNvbnRlbnQtaW50"
    b"ZWdyaXR5LnYxAEFydGlmYWN0Qnl0ZU1hdGNoUmVjZWlwdC52MQBbImp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uYXJ0aWZhY3Rz"
    b"LkFydGlmYWN0Qnl0ZU1hdGNoUmVjZWlwdCIsW1siYXJ0aWZhY3QiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5"
    b"c3RlbS5jb250cmFjdHMuQXJ0aWZhY3RCaW5kaW5nIiwiZmllbGRzIjpbWyJhcnRpZmFjdF9pZCIsImFydGlmYWN0LnByb3Zp"
    b"ZGVyLnNvdXJjZSJdLFsiYXJ0aWZhY3Rfcm9sZSIsIlNPRlRXQVJFX1NPVVJDRSJdLFsicHJvdmlkZXJfaWQiLCJwcm92aWRl"
    b"ci5maXh0dXJlIl0sWyJ2ZXJzaW9uIiwiMSJdLFsibG9naWNhbF9sb2NhdG9yIiwibmVzdGVkL3BheWxvYWQuYmluIl0sWyJs"
    b"b2NhdG9yX2tpbmQiLCJMT0NBTF9SRUdVTEFSX0ZJTEUiXSxbImJ5dGVfbGVuZ3RoIiwxNV0sWyJhcnRpZmFjdF9zaGEyNTYi"
    b"LCJlODM2YWI1N2Q3YTA0OTk0ZDljOTc5YjU1NWEyYTFmMGE5MDg1OWNkYzkwNGM1YjA0OTk1Njk4M2MwYzk4ODkwIl0sWyJt"
    b"ZWRpYV90eXBlIiwiYXBwbGljYXRpb24vb2N0ZXQtc3RyZWFtIl0sWyJjb3ZlcmFnZV9zdGF0dXMiLCJOT1RfQVBQTElDQUJM"
    b"RSJdLFsiY292ZXJhZ2UiLG51bGxdLFsibGljZW5zZV9ldmlkZW5jZV9zdGF0dXMiLCJSRVRBSU5FRF9IQVNIX0JPVU5EX0xJ"
    b"Q0VOU0VfQVJUSUZBQ1QiXSxbImxpY2Vuc2Vfc3BkeCIsIkJTRC0zLUNsYXVzZSJdLFsibGljZW5zZV9hcnRpZmFjdF9pZCIs"
    b"ImFydGlmYWN0LnByb3ZpZGVyLmxpY2Vuc2UiXSxbInJlZGlzdHJpYnV0aW9uX3N0YXR1cyIsIkJVTkRMRURfV0lUSF9SRVRB"
    b"SU5FRF9MSUNFTlNFX0VWSURFTkNFIl0sWyJsb2FkX29yZGVyX3N0YXR1cyIsIk5PVF9MT0FEQUJMRSJdLFsibG9hZF9vcmRl"
    b"ciIsbnVsbF0sWyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJRCJdLFsiY29udGVudF9pbnRlZ3JpdHlfY2xhc3MiLCJV"
    b"TkFVVEhFTlRJQ0FURURfQ09OVEVOVF9JTlRFR1JJVFlfT05MWSJdLFsiY29udGVudF9zaGEyNTYiLCI1N2Q4NmJiZjc1YWEz"
    b"MzVjYWVjZmYwM2U4YTE4OGJkY2NiYzE3NmRmYTRjYjE1OGZiZGU4NTlhYTU0YmQ1ZjQ4Il1dfV0sWyJ2ZXJpZmljYXRpb25f"
    b"c2NvcGUiLCJFWEFDVF9ERUNMQVJFRF9BUlRJRkFDVF9CWVRFU19PTkVfUkVBRF9TVFJFQU0iXSxbImhhc2hfYWxnb3JpdGht"
    b"IiwiU0hBLTI1NiJdLFsiZXhwZWN0ZWRfYnl0ZV9sZW5ndGgiLDE1XSxbIm9ic2VydmVkX2J5dGVfbGVuZ3RoIiwxNV0sWyJl"
    b"eHBlY3RlZF9zaGEyNTYiLCJlODM2YWI1N2Q3YTA0OTk0ZDljOTc5YjU1NWEyYTFmMGE5MDg1OWNkYzkwNGM1YjA0OTk1Njk4"
    b"M2MwYzk4ODkwIl0sWyJvYnNlcnZlZF9zaGEyNTYiLCJlODM2YWI1N2Q3YTA0OTk0ZDljOTc5YjU1NWEyYTFmMGE5MDg1OWNk"
    b"YzkwNGM1YjA0OTk1Njk4M2MwYzk4ODkwIl0sWyJieXRlX21hdGNoX3N0YXR1cyIsIk9CU0VSVkVEX0xFTkdUSF9BTkRfU0hB"
    b"MjU2X0VRVUFMX0RFQ0xBUkFUSU9OIl0sWyJjb250ZW50X2ludGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05U"
    b"RU5UX0lOVEVHUklUWV9PTkxZIl1dXQ==",
    validate=True,
)
BYTE_LITERAL_LENGTH = 1606
BYTE_LITERAL_SHA256 = "4ced959a1371bc22f3c1f2f26740cc0dcaa2abf4213679ea1959f90f72c94ccd"
LOCAL_LITERAL_PREIMAGE = base64.b64decode(
    b"anhwbGFuZXR4LnNvbGFyLXN5c3RlbS5hcnRpZmFjdHMubG9jYWwtYXJ0aWZhY3QtdmVyaWZpY2F0aW9uLXJlY2VpcHQuY29u"
    b"dGVudC1pbnRlZ3JpdHkudjEATG9jYWxBcnRpZmFjdFZlcmlmaWNhdGlvblJlY2VpcHQudjEAWyJqeHBsYW5ldHguc29sYXJf"
    b"c3lzdGVtLmFydGlmYWN0cy5Mb2NhbEFydGlmYWN0VmVyaWZpY2F0aW9uUmVjZWlwdCIsW1siYnl0ZV9tYXRjaCIsW1sianhw"
    b"bGFuZXR4LnNvbGFyX3N5c3RlbS5hcnRpZmFjdHMuQXJ0aWZhY3RCeXRlTWF0Y2hSZWNlaXB0IixbWyJhcnRpZmFjdCIseyJk"
    b"YXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5BcnRpZmFjdEJpbmRpbmciLCJmaWVsZHMiOltb"
    b"ImFydGlmYWN0X2lkIiwiYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl0sWyJhcnRpZmFjdF9yb2xlIiwiU09GVFdBUkVfU09V"
    b"UkNFIl0sWyJwcm92aWRlcl9pZCIsInByb3ZpZGVyLmZpeHR1cmUiXSxbInZlcnNpb24iLCIxIl0sWyJsb2dpY2FsX2xvY2F0"
    b"b3IiLCJuZXN0ZWQvcGF5bG9hZC5iaW4iXSxbImxvY2F0b3Jfa2luZCIsIkxPQ0FMX1JFR1VMQVJfRklMRSJdLFsiYnl0ZV9s"
    b"ZW5ndGgiLDE1XSxbImFydGlmYWN0X3NoYTI1NiIsImU4MzZhYjU3ZDdhMDQ5OTRkOWM5NzliNTU1YTJhMWYwYTkwODU5Y2Rj"
    b"OTA0YzViMDQ5OTU2OTgzYzBjOTg4OTAiXSxbIm1lZGlhX3R5cGUiLCJhcHBsaWNhdGlvbi9vY3RldC1zdHJlYW0iXSxbImNv"
    b"dmVyYWdlX3N0YXR1cyIsIk5PVF9BUFBMSUNBQkxFIl0sWyJjb3ZlcmFnZSIsbnVsbF0sWyJsaWNlbnNlX2V2aWRlbmNlX3N0"
    b"YXR1cyIsIlJFVEFJTkVEX0hBU0hfQk9VTkRfTElDRU5TRV9BUlRJRkFDVCJdLFsibGljZW5zZV9zcGR4IiwiQlNELTMtQ2xh"
    b"dXNlIl0sWyJsaWNlbnNlX2FydGlmYWN0X2lkIiwiYXJ0aWZhY3QucHJvdmlkZXIubGljZW5zZSJdLFsicmVkaXN0cmlidXRp"
    b"b25fc3RhdHVzIiwiQlVORExFRF9XSVRIX1JFVEFJTkVEX0xJQ0VOU0VfRVZJREVOQ0UiXSxbImxvYWRfb3JkZXJfc3RhdHVz"
    b"IiwiTk9UX0xPQURBQkxFIl0sWyJsb2FkX29yZGVyIixudWxsXSxbImV4dHJhcG9sYXRpb25fcG9saWN5IiwiRk9SQklEIl0s"
    b"WyJjb250ZW50X2ludGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05URU5UX0lOVEVHUklUWV9PTkxZIl0sWyJj"
    b"b250ZW50X3NoYTI1NiIsIjU3ZDg2YmJmNzVhYTMzNWNhZWNmZjAzZThhMTg4YmRjY2JjMTc2ZGZhNGNiMTU4ZmJkZTg1OWFh"
    b"NTRiZDVmNDgiXV19XSxbInZlcmlmaWNhdGlvbl9zY29wZSIsIkVYQUNUX0RFQ0xBUkVEX0FSVElGQUNUX0JZVEVTX09ORV9S"
    b"RUFEX1NUUkVBTSJdLFsiaGFzaF9hbGdvcml0aG0iLCJTSEEtMjU2Il0sWyJleHBlY3RlZF9ieXRlX2xlbmd0aCIsMTVdLFsi"
    b"b2JzZXJ2ZWRfYnl0ZV9sZW5ndGgiLDE1XSxbImV4cGVjdGVkX3NoYTI1NiIsImU4MzZhYjU3ZDdhMDQ5OTRkOWM5NzliNTU1"
    b"YTJhMWYwYTkwODU5Y2RjOTA0YzViMDQ5OTU2OTgzYzBjOTg4OTAiXSxbIm9ic2VydmVkX3NoYTI1NiIsImU4MzZhYjU3ZDdh"
    b"MDQ5OTRkOWM5NzliNTU1YTJhMWYwYTkwODU5Y2RjOTA0YzViMDQ5OTU2OTgzYzBjOTg4OTAiXSxbImJ5dGVfbWF0Y2hfc3Rh"
    b"dHVzIiwiT0JTRVJWRURfTEVOR1RIX0FORF9TSEEyNTZfRVFVQUxfREVDTEFSQVRJT04iXSxbImNvbnRlbnRfaW50ZWdyaXR5"
    b"X2NsYXNzIiwiVU5BVVRIRU5USUNBVEVEX0NPTlRFTlRfSU5URUdSSVRZX09OTFkiXV1dLFsiY29udGVudF9zaGEyNTYiLCI0"
    b"Y2VkOTU5YTEzNzFiYzIyZjNjMWYyZjI2NzQwY2MwZGNhYTJhYmY0MjEzNjc5ZWExOTU5ZjkwZjcyYzk0Y2NkIl1dXSxbInZl"
    b"cmlmaWNhdGlvbl9zdGFnZSIsIkxPQ0FMX0JZVEVfU1RSRUFNX1ZFUklGSUNBVElPTl9OT19QUk9WSURFUl9FWEVDVVRJT04i"
    b"XSxbInBsYXRmb3JtX3Njb3BlIiwiQ0FQQUJJTElUWV9HQVRFRF9QT1NJWF9QUk9GSUxFX05PX1BPUlRBQklMSVRZX0NMQUlN"
    b"Il0sWyJyb290X2NhcGFiaWxpdHlfc3RhdHVzIiwiQ0FMTEVSX0RJUkVDVE9SWV9GRF9EVVBMSUNBVEVEX1JPT1RfUEFUSF9B"
    b"Q1FVSVNJVElPTl9VTkFUVEVTVEVEIl0sWyJsb2NhdG9yX3Jlc29sdXRpb25fcG9saWN5IiwiU0VRVUVOVElBTF9DT01QT05F"
    b"TlRXSVNFX0RJUl9GRF9PX05PRk9MTE9XIl0sWyJsb2NhdG9yX2NvbXBvbmVudHMiLFsibmVzdGVkIiwicGF5bG9hZC5iaW4i"
    b"XV0sWyJmaW5hbF9vcGVuX3BvbGljeSIsIk9ORV9GSU5BTF9PUEVOX0RFU0NSSVBUSU9OX1JFQURfT05MWV9OT05CTE9DS0lO"
    b"R19OT0NUVFkiXSxbInJlYWRfcG9saWN5IiwiQk9VTkRFRF9PU19SRUFEX0RFQ0xBUkVEX0xFTkdUSF9QTFVTX09ORV9FT0Zf"
    b"UFJPQkUiXSxbIm1heGltdW1fYnl0ZV9sZW5ndGgiLDEwNzM3NDE4MjRdLFsicmVhZF9jaHVua19ieXRlX2xpbWl0IiwxMDQ4"
    b"NTc2XSxbIm1heGltdW1fZGF0YV9yZWFkX2NhbGxzIiw4MTkyXSxbImRhdGFfcmVhZF9jYWxsX2NvdW50IiwxXSxbImVvZl9w"
    b"cm9iZV9jYWxsX2NvdW50IiwxXSxbInJvb3Rfc3RhdF9maWVsZF9vcmRlciIsWyJzdF9kZXYiLCJzdF9pbm8iLCJzdF9tb2Rl"
    b"Il1dLFsicHJlX3Jvb3Rfc3RhdCIsWzIwNDksNzcsMTY4NzddXSxbInBvc3Rfcm9vdF9zdGF0IixbMjA0OSw3NywxNjg3N11d"
    b"LFsiZmlsZV9zdGF0X2ZpZWxkX29yZGVyIixbInN0X2RldiIsInN0X2lubyIsInN0X21vZGUiLCJzdF9ubGluayIsInN0X3Vp"
    b"ZCIsInN0X2dpZCIsInN0X3NpemUiLCJzdF9tdGltZV9ucyIsInN0X2N0aW1lX25zIl1dLFsicHJlX29wZW5fcGF0aF9maWxl"
    b"X3N0YXQiLFsyMDQ5LDg4LDMzMTg4LDEsMTAwMCwxMDAwLDE1LDE3MDAwMDAwMDAwMDAwMDAwMDAsMTcwMDAwMDAwMDEwMDAw"
    b"MDAwMF1dLFsicHJlX3JlYWRfZmlsZV9zdGF0IixbMjA0OSw4OCwzMzE4OCwxLDEwMDAsMTAwMCwxNSwxNzAwMDAwMDAwMDAw"
    b"MDAwMDAwLDE3MDAwMDAwMDAxMDAwMDAwMDBdXSxbInBvc3RfcmVhZF9maWxlX3N0YXQiLFsyMDQ5LDg4LDMzMTg4LDEsMTAw"
    b"MCwxMDAwLDE1LDE3MDAwMDAwMDAwMDAwMDAwMDAsMTcwMDAwMDAwMDEwMDAwMDAwMF1dLFsic2Vjb25kX3Jlc29sdXRpb25f"
    b"ZmlsZV9zdGF0IixbMjA0OSw4OCwzMzE4OCwxLDEwMDAsMTAwMCwxNSwxNzAwMDAwMDAwMDAwMDAwMDAwLDE3MDAwMDAwMDAx"
    b"MDAwMDAwMDBdXSxbIm1ldGFkYXRhX3N0YWJpbGl0eV9zdGF0dXMiLCJTRUxFQ1RFRF9QUkVPUEVOX1BSRV9SRUFEX1BPU1Rf"
    b"UkVBRF9BTkRfU0VDT05EX1NUQVRfRklFTERTX0lERU5USUNBTCJdLFsic2Vjb25kX3Jlc29sdXRpb25fc3RhdHVzIiwiU0VD"
    b"T05EX05BTUVTUEFDRV9PQlNFUlZBVElPTl9NQVRDSEVEX09QRU5fRklMRV9JREVOVElUWV9OT1RfQ09OVElOVUlUWV9QUk9P"
    b"RiJdLFsibG9jYWxfbWV0YWRhdGFfY2xhc3NpZmljYXRpb24iLCJMT0NBTF9OT05QT1JUQUJMRV9PQlNFUlZBVElPTl9OT1Rf"
    b"Q09OVEVOVF9JREVOVElUWSJdLFsiY3VzdG9keV9zY29wZSIsIlJFQURfU1RSRUFNX09OTFlfTk9fRlVUVVJFX1BBVEhfT1Jf"
    b"RkRfVVNFX0FVVEhPUklUWSJdLFsiZXhlY3V0aW9uX2V2aWRlbmNlX3N0YXR1cyIsIkxPQ0FMX1ZFUklGSUVSX0VYRUNVVElP"
    b"Tl9PTkxZX05PX1BST1ZJREVSX0FUX1VTRV9PUl9QUk9WSURFUl9FWEVDVVRJT04iXSxbInNlbWFudGljX3JlcGxheV9zdGF0"
    b"dXMiLCJSRVFVSVJFU19GUkVTSF9WRVJJRklDQVRJT05fV0lUSF9ORVdfUk9PVF9DQVBBQklMSVRZIl0sWyJjb250ZW50X2lu"
    b"dGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05URU5UX0lOVEVHUklUWV9PTkxZIl1dXQ==",
    validate=True,
)
LOCAL_LITERAL_LENGTH = 3730
LOCAL_LITERAL_SHA256 = "1a2ca3e29e34fcc4a86dee20da8109dfe27532c536ffd2cd7566951c83fdcb0b"


if __name__ == "__main__":
    unittest.main()
