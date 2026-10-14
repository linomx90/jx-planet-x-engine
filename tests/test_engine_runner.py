import dataclasses
import hashlib
import io
import json
import struct
import tokenize
import unittest
from unittest import mock
import zipfile

import numpy as np

import jxplanetx
import jxplanetx.engine as engine_package
import jxplanetx.engine.api as engine_api
import jxplanetx.engine.runner as runner_module
import jxplanetx.engine.simulation as simulation_module
import jxplanetx.engine.simulation_archive as archive_module
from jxplanetx.engine.runner import (
    ENGINE_JOB_BUNDLE_SCHEMA,
    ENGINE_JOB_V1_EXECUTION_POLICY,
    ENGINE_RUN_RECEIPT_SCHEMA,
    EngineRunnerError,
    engine_job_sha256,
    run_engine_job_bytes,
    run_scenario_manifest,
    verify_engine_job_bytes,
)
from jxplanetx.engine.scenario_manifest import (
    dump_dynamics_scenario_manifest,
    dynamics_scenario_manifest_sha256,
)
from tests.test_engine_scenario import scenario_pair


def fixture_manifest() -> tuple[bytes, str]:
    raw = dump_dynamics_scenario_manifest(scenario_pair()[0])
    return raw, dynamics_scenario_manifest_sha256(raw)


class EngineJobRunnerTests(unittest.TestCase):
    def test_complete_job_is_deterministic_and_verifies_without_rerun(self):
        manifest, manifest_sha256 = fixture_manifest()
        first = run_scenario_manifest(manifest, manifest_sha256)
        second = run_scenario_manifest(manifest, manifest_sha256)

        self.assertEqual(first.job_bytes, second.job_bytes)
        self.assertEqual(first.job_sha256, second.job_sha256)
        self.assertEqual(
            run_engine_job_bytes(
                manifest,
                expected_manifest_sha256=manifest_sha256,
            ),
            first.job_bytes,
        )
        verified = verify_engine_job_bytes(
            manifest,
            expected_manifest_sha256=manifest_sha256,
            job_bytes=first.job_bytes,
            expected_job_sha256=first.job_sha256,
        )
        self.assertEqual(verified.run_id, first.run_id)
        self.assertEqual(verified.job_sha256, first.job_sha256)
        self.assertEqual(verified.artifact.manifest_sha256, manifest_sha256)
        self.assertEqual(
            verified.archive.continuation.parent_native_result_digest,
            verified.artifact.native_result_digest,
        )
        self.assertFalse(
            verified.archive.continuation.snapshot.positions.flags.writeable
        )
        self.assertFalse(verified.production_authorized)
        self.assertFalse(verified.qualification_authorized)
        self.assertFalse(verified.registry_authorized)
        self.assertFalse(verified.execution_authenticated)

        with mock.patch(
            "jxplanetx.engine.simulation.JXSimulation.integrate",
            side_effect=AssertionError("verification must not execute dynamics"),
        ):
            second_verification = verify_engine_job_bytes(
                manifest,
                expected_manifest_sha256=manifest_sha256,
                job_bytes=first.job_bytes,
                expected_job_sha256=first.job_sha256,
            )
        self.assertEqual(second_verification.job_sha256, first.job_sha256)

    def test_outer_bundle_and_receipt_have_exact_nonauthorizing_contract(self):
        manifest, manifest_sha256 = fixture_manifest()
        bundle = run_scenario_manifest(manifest, manifest_sha256)
        with zipfile.ZipFile(io.BytesIO(bundle.job_bytes), "r") as archive:
            self.assertEqual(
                tuple(item.filename for item in archive.infolist()),
                ("trajectory.zip", "restart.zip", "receipt.json"),
            )
            receipt = json.loads(archive.read("receipt.json"))

        self.assertEqual(receipt["schema"], ENGINE_RUN_RECEIPT_SCHEMA)
        self.assertEqual(receipt["bundle"]["schema"], ENGINE_JOB_BUNDLE_SCHEMA)
        self.assertEqual(
            receipt["execution_policy"]["id"],
            ENGINE_JOB_V1_EXECUTION_POLICY,
        )
        self.assertEqual(receipt["manifest"]["sha256"], manifest_sha256)
        self.assertEqual(receipt["run_identity"]["sha256"], bundle.run_id)
        self.assertEqual(
            receipt["restart_archive"]["parent_native_result_digest"],
            receipt["run"]["native_result_digest"],
        )
        expected_false = (
            "accuracy_claimed",
            "cross_platform_replay_claimed",
            "execution_authenticated",
            "general_superiority_claimed",
            "improvement_claimed",
            "native_result_digest_recomputed",
            "numerical_execution_replayed",
            "physics_claimed",
            "production_authorized",
            "qualification_authorized",
            "registry_authorized",
            "source_runtime_reproduction_claimed",
        )
        self.assertTrue(all(receipt["claim"][key] is False for key in expected_false))
        self.assertIs(receipt["claim"]["content_integrity_only"], True)
        self.assertIs(receipt["claim"]["deterministic_encoding_only"], True)
        self.assertEqual(
            receipt["run"]["native_result_digest_verification"],
            "DECLARED_CROSS_BOUND_NOT_RECOMPUTED",
        )

    def test_external_hash_manifest_identity_and_exact_types_fail_closed(self):
        manifest, manifest_sha256 = fixture_manifest()
        bundle = run_scenario_manifest(manifest, manifest_sha256)
        tampered = bundle.job_bytes[:-1] + bytes((bundle.job_bytes[-1] ^ 1,))

        with self.assertRaisesRegex(EngineRunnerError, "bundle SHA-256"):
            verify_engine_job_bytes(
                manifest,
                expected_manifest_sha256=manifest_sha256,
                job_bytes=tampered,
                expected_job_sha256=bundle.job_sha256,
            )
        with self.assertRaisesRegex(EngineRunnerError, "expected_job_sha256"):
            verify_engine_job_bytes(
                manifest,
                expected_manifest_sha256=manifest_sha256,
                job_bytes=bundle.job_bytes,
                expected_job_sha256=False,  # type: ignore[arg-type]
            )
        changed_document = json.loads(manifest)
        changed_document["scenario"]["description"] += " changed"
        changed_manifest = (
            json.dumps(
                changed_document,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        changed_sha = hashlib.sha256(changed_manifest).hexdigest()
        with self.assertRaises(EngineRunnerError):
            verify_engine_job_bytes(
                changed_manifest,
                expected_manifest_sha256=changed_sha,
                job_bytes=bundle.job_bytes,
                expected_job_sha256=bundle.job_sha256,
            )

    def test_execution_policy_is_shared_by_runner_and_verifier(self):
        scenario = scenario_pair()[0]
        over_limit = dataclasses.replace(
            scenario,
            integration_spec=dataclasses.replace(
                scenario.integration_spec,
                maximum_steps=100_001,
            ),
        )
        over_manifest = dump_dynamics_scenario_manifest(over_limit)
        over_sha256 = dynamics_scenario_manifest_sha256(over_manifest)
        valid_manifest, valid_sha256 = fixture_manifest()
        valid_bundle = run_scenario_manifest(valid_manifest, valid_sha256)

        with self.assertRaisesRegex(EngineRunnerError, "execution policy"):
            run_scenario_manifest(over_manifest, over_sha256)
        with self.assertRaisesRegex(EngineRunnerError, "execution policy"):
            verify_engine_job_bytes(
                over_manifest,
                expected_manifest_sha256=over_sha256,
                job_bytes=valid_bundle.job_bytes,
                expected_job_sha256=valid_bundle.job_sha256,
            )

        oversized_identity = dataclasses.replace(
            scenario,
            scenario_id="x" * 5000,
        )
        oversized_manifest = dump_dynamics_scenario_manifest(oversized_identity)
        oversized_sha256 = dynamics_scenario_manifest_sha256(oversized_manifest)
        with mock.patch(
            "jxplanetx.engine.simulation.JXSimulation.integrate",
            side_effect=AssertionError("identity preflight must precede dynamics"),
        ):
            with self.assertRaisesRegex(EngineRunnerError, "encoding policy"):
                run_scenario_manifest(oversized_manifest, oversized_sha256)

    def test_exported_envelopes_reject_identity_relabeling(self):
        manifest, manifest_sha256 = fixture_manifest()
        bundle = run_scenario_manifest(manifest, manifest_sha256)
        verified = verify_engine_job_bytes(
            manifest,
            expected_manifest_sha256=manifest_sha256,
            job_bytes=bundle.job_bytes,
            expected_job_sha256=bundle.job_sha256,
        )
        for field in ("run_id", "manifest_sha256"):
            with self.subTest(bundle_field=field):
                with self.assertRaises(EngineRunnerError):
                    dataclasses.replace(bundle, **{field: "f" * 64})
        for field in (
            "run_id",
            "manifest_sha256",
            "artifact_sha256",
            "restart_archive_sha256",
            "receipt_sha256",
            "job_sha256",
        ):
            with self.subTest(verification_field=field):
                with self.assertRaises(EngineRunnerError):
                    dataclasses.replace(verified, **{field: "f" * 64})
        with self.assertRaises(runner_module.EngineRunArtifactError):
            dataclasses.replace(
                verified.artifact,
                artifact_sha256="f" * 64,
            )
        with self.assertRaises(runner_module.SimulationArchiveError):
            dataclasses.replace(
                verified.archive,
                archive_sha256="f" * 64,
            )
        forged_run = dataclasses.replace(bundle.run, native_result=object())
        with self.assertRaises(EngineRunnerError):
            dataclasses.replace(bundle, run=forged_run)

        with zipfile.ZipFile(io.BytesIO(bundle.job_bytes), "r") as archive:
            members = {
                info.filename: archive.read(info.filename)
                for info in archive.infolist()
            }
        receipt = json.loads(members["receipt.json"])
        receipt["claim"]["production_authorized"] = True
        members["receipt.json"] = runner_module._canonical(receipt)
        escalated_job = runner_module._dump_job_bundle(
            members["trajectory.zip"],
            members["restart.zip"],
            members["receipt.json"],
        )
        with self.assertRaises(EngineRunnerError):
            dataclasses.replace(
                verified,
                receipt_sha256=runner_module.engine_run_receipt_sha256(
                    members["receipt.json"]
                ),
                job_bytes=escalated_job,
                job_sha256=runner_module.engine_job_sha256(escalated_job),
            )

        type_confused_receipt = json.loads(bundle.receipt_bytes)
        type_confused_receipt["claim"]["production_authorized"] = 0
        type_confused_receipt["run"]["run_index"] = False
        type_confused_receipt["restart_archive"]["parent_run_index"] = False
        type_confused_receipt_bytes = runner_module._canonical(
            type_confused_receipt
        )
        type_confused_job = runner_module._dump_job_bundle(
            bundle.artifact_bytes,
            bundle.restart_archive_bytes,
            type_confused_receipt_bytes,
        )
        type_confused_receipt_sha256 = (
            runner_module.engine_run_receipt_sha256(
                type_confused_receipt_bytes
            )
        )
        type_confused_job_sha256 = runner_module.engine_job_sha256(
            type_confused_job
        )
        with self.assertRaisesRegex(EngineRunnerError, "exact verified bundle"):
            verify_engine_job_bytes(
                manifest,
                expected_manifest_sha256=manifest_sha256,
                job_bytes=type_confused_job,
                expected_job_sha256=type_confused_job_sha256,
            )
        with self.assertRaisesRegex(EngineRunnerError, "exact retained-member"):
            dataclasses.replace(
                bundle,
                receipt_bytes=type_confused_receipt_bytes,
                receipt_sha256=type_confused_receipt_sha256,
                job_bytes=type_confused_job,
                job_sha256=type_confused_job_sha256,
            )
        with self.assertRaisesRegex(EngineRunnerError, "identities"):
            dataclasses.replace(
                verified,
                receipt_sha256=type_confused_receipt_sha256,
                job_bytes=type_confused_job,
                job_sha256=type_confused_job_sha256,
            )

        damaged_archive = (
            bundle.restart_archive_bytes[:-1]
            + bytes((bundle.restart_archive_bytes[-1] ^ 1,))
        )
        receipt = json.loads(bundle.receipt_bytes)
        receipt["restart_archive"]["sha256"] = (
            runner_module.simulation_archive_sha256(damaged_archive)
        )
        receipt["restart_archive"]["size_bytes"] = len(damaged_archive)
        damaged_receipt = runner_module._canonical(receipt)
        damaged_job = runner_module._dump_job_bundle(
            bundle.artifact_bytes,
            damaged_archive,
            damaged_receipt,
        )
        with self.assertRaises(EngineRunnerError):
            dataclasses.replace(
                bundle,
                restart_archive_bytes=damaged_archive,
                restart_archive_sha256=(
                    runner_module.simulation_archive_sha256(damaged_archive)
                ),
                receipt_bytes=damaged_receipt,
                receipt_sha256=runner_module.engine_run_receipt_sha256(
                    damaged_receipt
                ),
                job_bytes=damaged_job,
                job_sha256=runner_module.engine_job_sha256(damaged_job),
            )

        # A coherently rehashed restart cannot relabel manifest-owned static
        # state.  This exercises the public constructors, not only the factory.
        with zipfile.ZipFile(io.BytesIO(bundle.restart_archive_bytes), "r") as archive:
            archive_members = {
                name: archive.read(name) for name in archive_module._MEMBER_NAMES
            }
        archive_manifest = json.loads(
            archive_members["manifest.json"].decode("ascii")
        )
        gravitational_parameters = np.load(
            io.BytesIO(archive_members["arrays/gravitational_parameters.npy"]),
            allow_pickle=False,
        ).copy()
        gravitational_parameters[0] *= 1.0001
        gravitational_parameters.setflags(write=False)
        gm_raw = archive_module._array_bytes(
            gravitational_parameters,
            "gravitational_parameters",
        )
        archive_members["arrays/gravitational_parameters.npy"] = gm_raw
        gm_record = archive_manifest["arrays"]["gravitational_parameters"]
        gm_record["sha256"] = hashlib.sha256(gm_raw).hexdigest()
        gm_record["size_bytes"] = len(gm_raw)
        continuation = verified.archive.continuation
        altered_snapshot = dataclasses.replace(
            continuation.snapshot,
            gravitational_parameters=gravitational_parameters,
        )
        altered_state_sha256 = (
            simulation_module._continuation_state_identity_sha256(
                simulation_id=continuation.simulation_id,
                parent_run_index=continuation.parent_run_index,
                parent_integrator_id=continuation.parent_integrator_id,
                parent_native_result_digest=(
                    continuation.parent_native_result_digest
                ),
                force_model_ids=continuation.force_model_ids,
                device="cpu",
                snapshot=altered_snapshot,
            )
        )
        archive_manifest["continuation"]["state_sha256"] = altered_state_sha256
        archive_manifest["snapshot"]["provenance"][
            "sha256"
        ] = altered_state_sha256
        archive_members["manifest.json"] = archive_module._canonical(
            archive_manifest
        )
        archive_stream = io.BytesIO()
        with zipfile.ZipFile(
            archive_stream,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for name in archive_module._MEMBER_NAMES:
                archive.writestr(
                    archive_module._zip_info(name),
                    archive_members[name],
                )
        altered_archive = archive_stream.getvalue()
        altered_archive_sha256 = runner_module.simulation_archive_sha256(
            altered_archive
        )
        loaded_altered_archive = runner_module.load_simulation_archive(
            altered_archive,
            expected_sha256=altered_archive_sha256,
            force_plan=scenario_pair()[0].force_plan,
        )
        altered_receipt = json.loads(bundle.receipt_bytes)
        altered_receipt["restart_archive"].update(
            {
                "continuation_state_sha256": altered_state_sha256,
                "sha256": altered_archive_sha256,
                "size_bytes": len(altered_archive),
            }
        )
        altered_receipt_bytes = runner_module._canonical(altered_receipt)
        altered_job = runner_module._dump_job_bundle(
            bundle.artifact_bytes,
            altered_archive,
            altered_receipt_bytes,
        )
        altered_receipt_sha256 = runner_module.engine_run_receipt_sha256(
            altered_receipt_bytes
        )
        altered_job_sha256 = runner_module.engine_job_sha256(altered_job)
        with self.assertRaisesRegex(EngineRunnerError, "restart archive"):
            dataclasses.replace(
                bundle,
                restart_archive_bytes=altered_archive,
                restart_archive_sha256=altered_archive_sha256,
                receipt_bytes=altered_receipt_bytes,
                receipt_sha256=altered_receipt_sha256,
                job_bytes=altered_job,
                job_sha256=altered_job_sha256,
            )
        with self.assertRaisesRegex(EngineRunnerError, "restart archive"):
            dataclasses.replace(
                verified,
                restart_archive_sha256=altered_archive_sha256,
                receipt_sha256=altered_receipt_sha256,
                job_bytes=altered_job,
                job_sha256=altered_job_sha256,
                archive=loaded_altered_archive,
            )

    def test_fixed_roster_preflight_precedes_zipfile_allocation(self):
        manifest, manifest_sha256 = fixture_manifest()
        scenario = scenario_pair()[0]
        huge_roster_eocd = struct.pack(
            "<4s4H2IH",
            b"PK\x05\x06",
            0,
            0,
            0xFFFF,
            0xFFFF,
            0,
            0,
            0,
        )
        digest = hashlib.sha256(huge_roster_eocd).hexdigest()
        with mock.patch.object(
            runner_module.zipfile,
            "ZipFile",
            side_effect=AssertionError("ZipFile must not parse a hostile roster"),
        ):
            with self.assertRaisesRegex(EngineRunnerError, "member roster"):
                runner_module._load_job_bundle(huge_roster_eocd, digest)
            with self.assertRaisesRegex(
                runner_module.EngineRunArtifactError,
                "member roster",
            ):
                runner_module.load_engine_run_artifact(
                    huge_roster_eocd,
                    expected_sha256=digest,
                    expected_manifest_sha256=manifest_sha256,
                    expected_run_id="jx-engine-job-v1:hostile-roster",
                )
            with self.assertRaisesRegex(
                runner_module.SimulationArchiveError,
                "member roster",
            ):
                runner_module.load_simulation_archive(
                    huge_roster_eocd,
                    expected_sha256=digest,
                    force_plan=scenario.force_plan,
                )

        # The manifest is intentionally retained here to establish this test's
        # trusted scenario fixture without invoking numerical execution.
        self.assertEqual(
            dynamics_scenario_manifest_sha256(manifest),
            manifest_sha256,
        )

    def test_hostile_restart_npy_header_stays_inside_public_error_envelope(self):
        manifest, manifest_sha256 = fixture_manifest()
        bundle = run_scenario_manifest(manifest, manifest_sha256)
        with zipfile.ZipFile(io.BytesIO(bundle.restart_archive_bytes), "r") as archive:
            members = {
                name: archive.read(name) for name in archive_module._MEMBER_NAMES
            }
        archive_manifest = json.loads(members["manifest.json"].decode("ascii"))
        nested_shape = "(" * 300 + "(2, 3)" + ")" * 300
        hostile_header = (
            "{'descr': '<f8', 'fortran_order': False, 'shape': "
            + nested_shape
            + ", }\n"
        ).encode("ascii")
        hostile_positions = (
            b"\x93NUMPY\x01\x00"
            + len(hostile_header).to_bytes(2, "little")
            + hostile_header
        )
        members["arrays/positions.npy"] = hostile_positions
        positions_record = archive_manifest["arrays"]["positions"]
        positions_record["sha256"] = hashlib.sha256(hostile_positions).hexdigest()
        positions_record["size_bytes"] = len(hostile_positions)
        members["manifest.json"] = archive_module._canonical(archive_manifest)
        archive_stream = io.BytesIO()
        with zipfile.ZipFile(
            archive_stream,
            "w",
            compression=zipfile.ZIP_STORED,
        ) as archive:
            for name in archive_module._MEMBER_NAMES:
                archive.writestr(archive_module._zip_info(name), members[name])
        hostile_archive = archive_stream.getvalue()
        hostile_archive_sha256 = runner_module.simulation_archive_sha256(
            hostile_archive
        )
        receipt = json.loads(bundle.receipt_bytes)
        receipt["restart_archive"]["sha256"] = hostile_archive_sha256
        receipt["restart_archive"]["size_bytes"] = len(hostile_archive)
        receipt_bytes = runner_module._canonical(receipt)
        hostile_job = runner_module._dump_job_bundle(
            bundle.artifact_bytes,
            hostile_archive,
            receipt_bytes,
        )
        with self.assertRaises(EngineRunnerError) as caught:
            verify_engine_job_bytes(
                manifest,
                expected_manifest_sha256=manifest_sha256,
                job_bytes=hostile_job,
                expected_job_sha256=runner_module.engine_job_sha256(hostile_job),
            )
        self.assertNotIsInstance(caught.exception.__cause__, tokenize.TokenError)

    def test_public_export_identity_and_deep_json_fail_closed(self):
        for name in (
            "ENGINE_JOB_BUNDLE_SCHEMA",
            "ENGINE_RUN_ARTIFACT_SCHEMA",
            "EngineRunnerError",
            "engine_job_sha256",
            "run_engine_job_bytes",
            "verify_engine_job_bytes",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(jxplanetx, name), getattr(engine_package, name))
                self.assertIs(getattr(engine_package, name), getattr(engine_api, name))

        nested = ("[" * 20_000 + "0" + "]" * 20_000 + "\n").encode("ascii")
        with self.assertRaises(EngineRunnerError):
            run_scenario_manifest(nested, hashlib.sha256(nested).hexdigest())

    def test_engine_job_hash_accepts_only_exact_bytes(self):
        self.assertEqual(
            engine_job_sha256(b"fixture"),
            hashlib.sha256(b"fixture").hexdigest(),
        )
        with self.assertRaisesRegex(EngineRunnerError, "exact bytes"):
            engine_job_sha256(bytearray(b"fixture"))  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
