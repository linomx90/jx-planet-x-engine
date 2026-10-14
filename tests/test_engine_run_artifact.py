from __future__ import annotations

import copy
import io
import json
import unittest
from unittest import mock
import zipfile

import numpy as np

from jxplanetx.engine import (
    ADAPTIVE_RKF78_METHOD_ID,
    AdaptiveRKF78Spec,
    BackendSpec,
    ForcePlan,
    JXSimulation,
    NewtonianPointMass,
    ParameterMetadata,
    Provenance,
    StateSnapshot,
)
import jxplanetx.engine.run_artifact as run_artifact
from jxplanetx.engine.run_artifact import (
    ENGINE_RUN_ARTIFACT_SCHEMA,
    ENGINE_RUN_ARTIFACT_SCOPE,
    EngineRunArtifactError,
    JXEngineRunArtifact,
    dump_engine_run_artifact,
    engine_run_artifact_sha256,
    load_engine_run_artifact,
)


MANIFEST_SHA256 = "c" * 64
RUN_ID = "job.fixture.run-0000"
BODY_IDS = ("PRIMARY", "SECONDARY")
UNIT_SYSTEM_ID = "fixture.engine-run-artifact.canonical"


def provenance() -> Provenance:
    return Provenance(
        source_id="fixture.engine-run-artifact",
        citation="Synthetic deterministic run-artifact fixture",
        version="1",
        sha256="a" * 64,
    )


def source_run():
    state = StateSnapshot(
        snapshot_id="fixture.engine-run-artifact.initial",
        epoch=0.0,
        time_scale="TDB",
        frame="BARYCENTRIC_INERTIAL",
        origin="SYSTEM_BARYCENTER",
        axes="ICRS_ALIGNED",
        length_unit="CANONICAL_LENGTH",
        time_unit="CANONICAL_TIME",
        mass_unit="CANONICAL_MASS",
        unit_system_id=UNIT_SYSTEM_ID,
        body_ids=BODY_IDS,
        positions=np.array(
            ((-0.5, 0.0, 0.0), (0.5, 0.0, 0.0)),
            dtype=np.float64,
        ),
        velocities=np.array(
            ((0.0, -0.5, 0.0), (0.0, 0.5, 0.0)),
            dtype=np.float64,
        ),
        gravitational_parameters=np.array((0.5, 0.5), dtype=np.float64),
        masses=np.array((0.5, 0.5), dtype=np.float64),
        radii=np.zeros(2, dtype=np.float64),
        massive=np.ones(2, dtype=np.bool_),
        provenance=provenance(),
    )
    plan = ForcePlan(
        plan_id="fixture.engine-run-artifact.plan",
        backend=BackendSpec("numpy", "cpu", 2),
        models=(
            NewtonianPointMass(
                source_ids=BODY_IDS,
                target_ids=BODY_IDS,
                unit_system_id=UNIT_SYSTEM_ID,
                parameter_metadata=(
                    ParameterMetadata(
                        parameter_id="state.gravitational_parameters",
                        units="CANONICAL_LENGTH^3/CANONICAL_TIME^2",
                        provenance=provenance(),
                        uncertainty=None,
                        covariance_group=None,
                        validity_start=-1.0,
                        validity_end=1.0,
                    ),
                ),
            ),
        ),
    )
    spec = AdaptiveRKF78Spec(
        checkpoint_epochs=(0.0, 0.01, 0.02),
        initial_step=0.005,
        minimum_step=1.0e-12,
        maximum_step=0.01,
        position_atol=np.full((2, 3), 1.0e-13, dtype=np.float64),
        position_rtol=1.0e-12,
        velocity_atol=np.full((2, 3), 1.0e-13, dtype=np.float64),
        velocity_rtol=1.0e-12,
        maximum_steps=1_000,
        maximum_rejections=100,
        safety_factor=0.9,
        minimum_scale_factor=0.2,
        maximum_scale_factor=4.0,
    )
    simulation = JXSimulation(
        "fixture.engine-run-artifact.simulation",
        state,
        plan,
    )
    return simulation.integrate(ADAPTIVE_RKF78_METHOD_ID, spec)


def artifact_members(artifact: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(artifact), "r") as archive:
        return {
            info.filename: archive.read(info.filename)
            for info in archive.infolist()
        }


def rewrite_manifest(artifact: bytes, mutation) -> bytes:
    members = artifact_members(artifact)
    manifest = json.loads(members["manifest.json"].decode("ascii"))
    mutation(manifest)
    members["manifest.json"] = run_artifact._canonical(manifest)
    return run_artifact._zip_bytes(members)


class EngineRunArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_result = source_run()
        cls.artifact = dump_engine_run_artifact(
            cls.source_result,
            manifest_sha256=MANIFEST_SHA256,
            run_id=RUN_ID,
        )
        cls.artifact_sha256 = engine_run_artifact_sha256(cls.artifact)

    def load(self, artifact: bytes | None = None) -> JXEngineRunArtifact:
        selected = self.artifact if artifact is None else artifact
        return load_engine_run_artifact(
            selected,
            expected_sha256=engine_run_artifact_sha256(selected),
            expected_manifest_sha256=MANIFEST_SHA256,
            expected_run_id=RUN_ID,
        )

    def test_deterministic_round_trip_and_crossbindings(self) -> None:
        second = dump_engine_run_artifact(
            self.source_result,
            manifest_sha256=MANIFEST_SHA256,
            run_id=RUN_ID,
        )
        self.assertEqual(second, self.artifact)
        self.assertEqual(
            engine_run_artifact_sha256(second),
            self.artifact_sha256,
        )
        with zipfile.ZipFile(io.BytesIO(second), "r") as archive:
            self.assertEqual(
                tuple(info.filename for info in archive.infolist()),
                (
                    "arrays/checkpoint_epochs.npy",
                    "arrays/checkpoint_positions.npy",
                    "arrays/checkpoint_velocities.npy",
                    "manifest.json",
                ),
            )
            self.assertTrue(
                all(
                    info.compress_type == zipfile.ZIP_STORED
                    for info in archive.infolist()
                )
            )

        loaded = self.load(second)
        self.assertIs(type(loaded), JXEngineRunArtifact)
        self.assertEqual(loaded.artifact_sha256, self.artifact_sha256)
        self.assertEqual(loaded.manifest_sha256, MANIFEST_SHA256)
        self.assertEqual(loaded.run_id, RUN_ID)
        self.assertEqual(loaded.simulation_id, self.source_result.simulation_id)
        self.assertEqual(loaded.run_index, self.source_result.run_index)
        self.assertEqual(loaded.integrator_id, ADAPTIVE_RKF78_METHOD_ID)
        self.assertEqual(
            loaded.native_result_digest,
            self.source_result.native_result_digest,
        )
        self.assertEqual(loaded.scientific_claim_state, "SCREENING_ONLY")
        self.assertEqual(loaded.evidence_class, "MODEL_OUTPUT")
        self.assertFalse(loaded.accuracy_claimed)
        self.assertFalse(loaded.improvement_claimed)
        self.assertFalse(loaded.physics_claimed)
        self.assertFalse(loaded.execution_authenticated)
        self.assertFalse(loaded.production_authorized)
        self.assertFalse(loaded.registry_authorized)
        self.assertFalse(loaded.qualification_authorized)
        self.assertFalse(loaded.general_superiority_claimed)
        self.assertEqual(loaded.schema, ENGINE_RUN_ARTIFACT_SCHEMA)
        self.assertEqual(loaded.scope, ENGINE_RUN_ARTIFACT_SCOPE)
        self.assertEqual(loaded.body_ids, BODY_IDS)
        self.assertEqual(
            loaded.checkpoint_epochs,
            self.source_result.checkpoint_epochs,
        )
        np.testing.assert_array_equal(
            loaded.checkpoint_positions,
            np.stack(self.source_result.checkpoint_positions),
        )
        np.testing.assert_array_equal(
            loaded.checkpoint_velocities,
            np.stack(self.source_result.checkpoint_velocities),
        )
        self.assertTrue(loaded.checkpoint_positions.flags.owndata)
        self.assertFalse(loaded.checkpoint_positions.flags.writeable)
        self.assertTrue(loaded.checkpoint_velocities.flags.owndata)
        self.assertFalse(loaded.checkpoint_velocities.flags.writeable)

    def test_source_type_route_identity_and_claim_gates_fail_closed(self) -> None:
        with self.assertRaisesRegex(EngineRunArtifactError, "exact JXSimulationRun"):
            dump_engine_run_artifact(
                object(),  # type: ignore[arg-type]
                manifest_sha256=MANIFEST_SHA256,
                run_id=RUN_ID,
            )
        for field, value, message in (
            ("backend_id", "cupy", "NumPy/CPU"),
            ("integrator_id", "forged.integrator", "NumPy/CPU"),
            ("production_authorized", True, "claim ceiling"),
            ("native_result_digest", "b" * 64, "identity is inconsistent"),
        ):
            with self.subTest(field=field):
                forged = copy.copy(self.source_result)
                object.__setattr__(forged, field, value)
                with self.assertRaisesRegex(EngineRunArtifactError, message):
                    dump_engine_run_artifact(
                        forged,
                        manifest_sha256=MANIFEST_SHA256,
                        run_id=RUN_ID,
                    )
        for manifest_sha256, run_id, message in (
            ("C" * 64, RUN_ID, "lowercase"),
            (MANIFEST_SHA256, " bad ", "trimmed"),
            (MANIFEST_SHA256, True, "trimmed"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(EngineRunArtifactError, message):
                    dump_engine_run_artifact(
                        self.source_result,
                        manifest_sha256=manifest_sha256,
                        run_id=run_id,  # type: ignore[arg-type]
                    )

    def test_expected_artifact_manifest_and_run_bindings_are_mandatory(self) -> None:
        with self.assertRaisesRegex(EngineRunArtifactError, "expected_sha256"):
            load_engine_run_artifact(
                self.artifact,
                expected_sha256="0" * 64,
                expected_manifest_sha256=MANIFEST_SHA256,
                expected_run_id=RUN_ID,
            )
        with self.assertRaisesRegex(EngineRunArtifactError, "manifest binding"):
            load_engine_run_artifact(
                self.artifact,
                expected_sha256=self.artifact_sha256,
                expected_manifest_sha256="d" * 64,
                expected_run_id=RUN_ID,
            )
        with self.assertRaisesRegex(EngineRunArtifactError, "run binding"):
            load_engine_run_artifact(
                self.artifact,
                expected_sha256=self.artifact_sha256,
                expected_manifest_sha256=MANIFEST_SHA256,
                expected_run_id="job.fixture.run-other",
            )
        with self.assertRaisesRegex(EngineRunArtifactError, "exact bytes"):
            engine_run_artifact_sha256(bytearray(self.artifact))  # type: ignore[arg-type]

    def test_member_roster_claim_and_canonical_json_mutations_are_rejected(self) -> None:
        members = artifact_members(self.artifact)
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in run_artifact._MEMBER_NAMES:
                archive.writestr(run_artifact._zip_info(name), members[name])
            archive.writestr(run_artifact._zip_info("extra.txt"), b"hostile")
        with self.assertRaisesRegex(EngineRunArtifactError, "member roster"):
            self.load(stream.getvalue())

        claim_changed = rewrite_manifest(
            self.artifact,
            lambda document: document["claim"].update(
                {"production_authorized": True}
            ),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "claim ceiling"):
            self.load(claim_changed)

        type_confused_claim = rewrite_manifest(
            self.artifact,
            lambda document: document["claim"].update(
                {"production_authorized": 0}
            ),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "claim ceiling"):
            self.load(type_confused_claim)

        noncanonical_members = artifact_members(self.artifact)
        manifest = json.loads(noncanonical_members["manifest.json"].decode("ascii"))
        noncanonical_members["manifest.json"] = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ).encode("ascii")
        noncanonical = run_artifact._zip_bytes(noncanonical_members)
        with self.assertRaisesRegex(EngineRunArtifactError, "not canonical"):
            self.load(noncanonical)

        numeric_run_id = rewrite_manifest(
            self.artifact,
            lambda document: document["binding"].update({"run_id": 7}),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "trimmed text"):
            self.load(numeric_run_id)

    def test_array_hash_type_shape_and_npy_encoding_mutations_are_rejected(self) -> None:
        raw = run_artifact._array_bytes(
            np.zeros((1,), dtype=np.float64),
            "fixture",
        )
        aliased_shape_record = {
            "dtype": run_artifact.ENGINE_RUN_ARTIFACT_ARRAY_DTYPE,
            "member": "arrays/fixture.npy",
            "sha256": run_artifact._sha256(raw),
            "shape": [True],
            "size_bytes": len(raw),
        }
        with self.assertRaisesRegex(EngineRunArtifactError, "metadata"):
            run_artifact._read_array(
                raw,
                aliased_shape_record,
                "fixture",
                (1,),
            )

        members = artifact_members(self.artifact)
        changed = bytearray(members["arrays/checkpoint_positions.npy"])
        changed[-1] ^= 1
        members["arrays/checkpoint_positions.npy"] = bytes(changed)
        stale_hash = run_artifact._zip_bytes(members)
        with self.assertRaisesRegex(EngineRunArtifactError, "byte identity mismatch"):
            self.load(stale_hash)

        members = artifact_members(self.artifact)
        manifest = json.loads(members["manifest.json"].decode("ascii"))
        array_stream = io.BytesIO()
        np.lib.format.write_array(
            array_stream,
            np.zeros((3, 2, 3), dtype=np.int64),
            version=(2, 0),
            allow_pickle=False,
        )
        wrong_type = array_stream.getvalue()
        members["arrays/checkpoint_positions.npy"] = wrong_type
        record = manifest["arrays"]["checkpoint_positions"]
        record["sha256"] = run_artifact._sha256(wrong_type)
        record["size_bytes"] = len(wrong_type)
        members["manifest.json"] = run_artifact._canonical(manifest)
        typed = run_artifact._zip_bytes(members)
        with self.assertRaisesRegex(EngineRunArtifactError, "header dtype"):
            self.load(typed)

        members = artifact_members(self.artifact)
        raw = members["arrays/checkpoint_epochs.npy"] + b"trailing"
        manifest = json.loads(members["manifest.json"].decode("ascii"))
        members["arrays/checkpoint_epochs.npy"] = raw
        record = manifest["arrays"]["checkpoint_epochs"]
        record["sha256"] = run_artifact._sha256(raw)
        record["size_bytes"] = len(raw)
        members["manifest.json"] = run_artifact._canonical(manifest)
        trailing = run_artifact._zip_bytes(members)
        with self.assertRaisesRegex(EngineRunArtifactError, "payload length"):
            self.load(trailing)

        members = artifact_members(self.artifact)
        manifest = json.loads(members["manifest.json"].decode("ascii"))
        huge_count = 1_000_000_000
        array_stream = io.BytesIO()
        np.lib.format.write_array_header_2_0(
            array_stream,
            {
                "descr": np.dtype("<f8").str,
                "fortran_order": False,
                "shape": (huge_count,),
            },
        )
        array_stream.write(b"\x00" * 8)
        huge_shape = array_stream.getvalue()
        members["arrays/checkpoint_epochs.npy"] = huge_shape
        record = manifest["arrays"]["checkpoint_epochs"]
        record["sha256"] = run_artifact._sha256(huge_shape)
        record["shape"] = [huge_count]
        record["size_bytes"] = len(huge_shape)
        manifest["run"].update(
            {
                "accepted_steps": huge_count,
                "attempted_steps": huge_count,
                "checkpoint_count": huge_count,
                "force_evaluations": huge_count * 13,
                "rejected_steps": 0,
            }
        )
        members["manifest.json"] = run_artifact._canonical(manifest)
        allocation_probe = run_artifact._zip_bytes(members)
        with mock.patch.object(
            np,
            "load",
            side_effect=AssertionError("np.load must not run before NPY preflight"),
        ):
            with self.assertRaisesRegex(EngineRunArtifactError, "payload length"):
                self.load(allocation_probe)

        members = artifact_members(self.artifact)
        manifest = json.loads(members["manifest.json"].decode("ascii"))
        nested_shape = "(" * 300 + "(3,)" + ")" * 300
        hostile_header = (
            "{'descr': '<f8', 'fortran_order': False, 'shape': "
            + nested_shape
            + ", }\n"
        ).encode("ascii")
        hostile_array = (
            b"\x93NUMPY\x02\x00"
            + len(hostile_header).to_bytes(4, "little")
            + hostile_header
        )
        members["arrays/checkpoint_epochs.npy"] = hostile_array
        record = manifest["arrays"]["checkpoint_epochs"]
        record["sha256"] = run_artifact._sha256(hostile_array)
        record["size_bytes"] = len(hostile_array)
        members["manifest.json"] = run_artifact._canonical(manifest)
        hostile_header_artifact = run_artifact._zip_bytes(members)
        with self.assertRaisesRegex(EngineRunArtifactError, "invalid.*NPY header"):
            self.load(hostile_header_artifact)

    def test_lineage_accounting_and_size_mutations_are_rejected_under_oo(self) -> None:
        bad_accounting = rewrite_manifest(
            self.artifact,
            lambda document: document["run"].update({"attempted_steps": 999}),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "accounting"):
            self.load(bad_accounting)

        bad_force_evaluations = rewrite_manifest(
            self.artifact,
            lambda document: document["run"].update(
                {"force_evaluations": document["run"]["force_evaluations"] + 1}
            ),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "force-evaluation"):
            self.load(bad_force_evaluations)

        too_few_accepted = rewrite_manifest(
            self.artifact,
            lambda document: document["run"].update(
                {
                    "accepted_steps": 1,
                    "attempted_steps": 1,
                    "force_evaluations": 13,
                    "rejected_steps": 0,
                }
            ),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "checkpoint interval"):
            self.load(too_few_accepted)

        bad_lineage = rewrite_manifest(
            self.artifact,
            lambda document: document["run"].update(
                {"parent_run_index": 0}
            ),
        )
        with self.assertRaisesRegex(EngineRunArtifactError, "parent_run_index"):
            self.load(bad_lineage)

        with mock.patch.object(
            run_artifact,
            "ENGINE_RUN_ARTIFACT_MAX_BYTES",
            len(self.artifact) - 1,
        ):
            with self.assertRaisesRegex(EngineRunArtifactError, "size ceiling"):
                self.load()

        manifest_size = len(artifact_members(self.artifact)["manifest.json"])
        with mock.patch.object(
            run_artifact,
            "ENGINE_RUN_ARTIFACT_MAX_MANIFEST_BYTES",
            manifest_size - 1,
        ):
            with self.assertRaisesRegex(EngineRunArtifactError, "manifest exceeds"):
                self.load()

    def test_deep_json_is_wrapped_by_the_public_loader(self) -> None:
        members = artifact_members(self.artifact)
        members["manifest.json"] = (
            "[" * 20_000 + "0" + "]" * 20_000 + "\n"
        ).encode("ascii")
        hostile = run_artifact._zip_bytes(members)
        with self.assertRaises(EngineRunArtifactError):
            load_engine_run_artifact(
                hostile,
                expected_sha256=engine_run_artifact_sha256(hostile),
                expected_manifest_sha256=MANIFEST_SHA256,
                expected_run_id=RUN_ID,
            )


if __name__ == "__main__":
    unittest.main()
