"""Classification-only frame relationship tests for Solar-System Milestone 3."""

from __future__ import annotations

import dataclasses
import inspect
import json
import os
import subprocess
import sys
import unittest

from jxplanetx.solar_system import frames as frames_module
from jxplanetx.solar_system.contracts import (
    CoordinateEpoch,
    CoverageInterval,
    FrameRealization,
    SolarSystemContractError,
    validate_integrity,
)
from jxplanetx.solar_system.frames import (
    FRAME_RELATION_TOKENS,
    classify_frame_relation,
    require_exact_frame_contract_identity,
)


EXPECTED_EXPORTS = [
    "FRAME_RELATION_TOKENS",
    "classify_frame_relation",
    "require_exact_frame_contract_identity",
]

EXACT_IDENTITY = "EXACT_FRAME_CONTRACT_IDENTITY"
SAME_MAP_DIFFERENT_CUSTODY = (
    "SAME_DECLARED_COORDINATE_MAP_DIFFERENT_CONTRACT_CUSTODY_UNSUPPORTED"
)
UNRESOLVED_ALIAS = "UNRESOLVED_FRAME_ALIAS"
DIFFERENT_ORIGIN = (
    "SAME_DECLARED_AXES_AND_COORDINATE_CONTEXT_DIFFERENT_ORIGIN_"
    "TRANSLATION_REQUIRED_UNSUPPORTED"
)
AXES_OR_CONTEXT = "AXES_OR_COORDINATE_CONTEXT_TRANSFORMATION_REQUIRED_UNSUPPORTED"

SECOND_UNIT = "unit.second.defined-exact.v1"
COMMON_AXES = "axes.spice-j2000-icrf-aligned.fixture"
COMMON_ORIENTATION = "orientation.spice-j2000-icrf-aligned-inertial.v1"


def inertial_origin_frame(
    frame_id: str = "frame.body399.tdb.fixture",
    *,
    origin_kind: str = "BODY_CENTER",
    origin_naif_id: int | None = 399,
    origin_realization_id: str = "origin.body399.fixture",
    axes_realization_id: str = COMMON_AXES,
    orientation_model_id: str = COMMON_ORIENTATION,
    coordinate_time_scale: str = "TDB",
    artifact_ids: tuple[str, ...] = (),
) -> FrameRealization:
    return FrameRealization(
        frame_id=frame_id,
        frame_kind="INERTIAL_ORIGIN_CENTERED",
        origin_kind=origin_kind,
        origin_naif_id=origin_naif_id,
        origin_realization_id=origin_realization_id,
        axes_realization_id=axes_realization_id,
        orientation_model_id=orientation_model_id,
        orientation_time_dependence="STATIC",
        coordinate_time_scale=coordinate_time_scale,
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=artifact_ids,
    )


def et_epoch(whole: int) -> CoordinateEpoch:
    return CoordinateEpoch(
        time_scale="TDB",
        representation="SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        whole=whole,
        fraction=0.0,
        coordinate_unit_id=SECOND_UNIT,
        origin_id="SPICE_J2000_TDB_ORIGIN",
        realization_id="time.spice-j2000-tdb.fixture",
        artifact_ids=(),
    )


def barycentric_tdb_frame(
    frame_id: str = "frame.ssb.tdb-compatible.fixture",
    *,
    origin_realization_id: str = "origin.ssb.de-release.fixture",
    axes_realization_id: str = COMMON_AXES,
    orientation_model_id: str = COMMON_ORIENTATION,
    artifact_ids: tuple[str, ...] = ("artifact.spk.fixture",),
) -> FrameRealization:
    coverage = CoverageInterval(
        start=et_epoch(-10_000),
        end=et_epoch(10_000),
        endpoint_policy="CLOSED_CLOSED",
    )
    return FrameRealization(
        frame_id=frame_id,
        frame_kind="TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
        origin_kind="SOLAR_SYSTEM_BARYCENTER",
        origin_naif_id=0,
        origin_realization_id=origin_realization_id,
        axes_realization_id=axes_realization_id,
        orientation_model_id=orientation_model_id,
        orientation_time_dependence="STATIC",
        coordinate_time_scale="TDB",
        coverage_status="RETAINED",
        coverage=(coverage,),
        artifact_ids=artifact_ids,
    )


def model_barycenter_frame(
    frame_id: str = "frame.model-barycenter.tdb.fixture",
) -> FrameRealization:
    return inertial_origin_frame(
        frame_id,
        origin_kind="NEWTONIAN_MODEL_BARYCENTER",
        origin_naif_id=None,
        origin_realization_id="origin.newtonian-model-barycenter.fixture",
    )


def synthetic_frame(
    frame_id: str = "frame.synthetic.fixture",
    *,
    origin_realization_id: str = "origin.synthetic.fixture",
) -> FrameRealization:
    return FrameRealization(
        frame_id=frame_id,
        frame_kind="SYNTHETIC",
        origin_kind="SYNTHETIC",
        origin_naif_id=None,
        origin_realization_id=origin_realization_id,
        axes_realization_id="axes.synthetic.fixture",
        orientation_model_id="orientation.synthetic.static.fixture",
        orientation_time_dependence="STATIC",
        coordinate_time_scale="SYNTHETIC",
        coverage_status="NOT_APPLICABLE",
        coverage=None,
        artifact_ids=(),
    )


class FrameRelationPrecedenceTests(unittest.TestCase):
    def test_exact_exports_tokens_and_signatures(self) -> None:
        self.assertEqual(frames_module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(frames_module.__all__, sorted(frames_module.__all__))
        self.assertEqual(
            FRAME_RELATION_TOKENS,
            (
                EXACT_IDENTITY,
                SAME_MAP_DIFFERENT_CUSTODY,
                UNRESOLVED_ALIAS,
                DIFFERENT_ORIGIN,
                AXES_OR_CONTEXT,
            ),
        )
        self.assertEqual(
            tuple(inspect.signature(classify_frame_relation).parameters),
            ("source_frame", "target_frame"),
        )
        self.assertEqual(
            tuple(inspect.signature(require_exact_frame_contract_identity).parameters),
            ("source_frame", "target_frame"),
        )
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_separately_constructed_equal_content_is_exact_identity(self) -> None:
        source = inertial_origin_frame()
        target = inertial_origin_frame()
        self.assertIsNot(source, target)
        self.assertEqual(source.content_sha256, target.content_sha256)
        self.assertEqual(classify_frame_relation(source, target), EXACT_IDENTITY)
        self.assertIsNone(require_exact_frame_contract_identity(source, target))

    def test_same_id_and_map_but_different_custody_precedes_alias(self) -> None:
        source = inertial_origin_frame(artifact_ids=("artifact.frame.a",))
        target = inertial_origin_frame(artifact_ids=("artifact.frame.b",))
        self.assertNotEqual(source.content_sha256, target.content_sha256)
        self.assertEqual(
            classify_frame_relation(source, target), SAME_MAP_DIFFERENT_CUSTODY
        )
        with self.assertRaises(SolarSystemContractError):
            require_exact_frame_contract_identity(source, target)

    def test_same_id_semantic_collision_raises_before_diagnostic(self) -> None:
        source = inertial_origin_frame()
        collisions = (
            inertial_origin_frame(
                origin_naif_id=10,
                origin_realization_id="origin.body10.fixture",
            ),
            inertial_origin_frame(axes_realization_id="axes.other.fixture"),
            inertial_origin_frame(coordinate_time_scale="TCB"),
        )
        for collision in collisions:
            with self.subTest(collision=collision.content_sha256):
                with self.assertRaisesRegex(
                    SolarSystemContractError,
                    "one frame_id cannot name different declared coordinate-map semantics",
                ):
                    classify_frame_relation(source, collision)

    def test_different_id_same_map_is_alias_even_with_different_custody(self) -> None:
        source = inertial_origin_frame(
            "frame.alias.a", artifact_ids=("artifact.frame.a",)
        )
        target = inertial_origin_frame(
            "frame.alias.b", artifact_ids=("artifact.frame.b",)
        )
        self.assertEqual(classify_frame_relation(source, target), UNRESOLVED_ALIAS)
        self.assertEqual(classify_frame_relation(target, source), UNRESOLVED_ALIAS)
        with self.assertRaises(SolarSystemContractError):
            require_exact_frame_contract_identity(source, target)

    def test_same_kind_origin_only_diagnostic_is_symmetric(self) -> None:
        earth = inertial_origin_frame("frame.earth", origin_naif_id=399)
        sun = inertial_origin_frame(
            "frame.sun",
            origin_naif_id=10,
            origin_realization_id="origin.sun.fixture",
        )
        earth_other_realization = inertial_origin_frame(
            "frame.earth.other-realization",
            origin_realization_id="origin.earth.other-realization",
        )
        emb = inertial_origin_frame(
            "frame.emb",
            origin_kind="PLANETARY_SYSTEM_BARYCENTER",
            origin_naif_id=3,
            origin_realization_id="origin.emb.fixture",
        )
        for source, target in (
            (earth, sun),
            (sun, earth),
            (earth, earth_other_realization),
            (earth_other_realization, earth),
            (earth, emb),
            (emb, earth),
        ):
            with self.subTest(source=source.frame_id, target=target.frame_id):
                self.assertEqual(classify_frame_relation(source, target), DIFFERENT_ORIGIN)

    def test_ssb_to_body_or_planetary_barycenter_is_narrow_origin_diagnostic(self) -> None:
        ssb = barycentric_tdb_frame()
        earth = inertial_origin_frame("frame.earth", origin_naif_id=399)
        emb = inertial_origin_frame(
            "frame.emb",
            origin_kind="PLANETARY_SYSTEM_BARYCENTER",
            origin_naif_id=3,
            origin_realization_id="origin.emb.fixture",
        )
        for local in (earth, emb):
            for source, target in ((ssb, local), (local, ssb)):
                with self.subTest(source=source.frame_id, target=target.frame_id):
                    self.assertEqual(
                        classify_frame_relation(source, target), DIFFERENT_ORIGIN
                    )

    def test_narrow_cross_kind_rule_excludes_tcb_model_and_axes_changes(self) -> None:
        ssb = barycentric_tdb_frame()
        earth_tcb = inertial_origin_frame(
            "frame.earth.tcb", coordinate_time_scale="TCB"
        )
        model = model_barycenter_frame()
        other_axes = inertial_origin_frame(
            "frame.earth.other-axes", axes_realization_id="axes.other.fixture"
        )
        for candidate in (earth_tcb, model, other_axes):
            for source, target in ((ssb, candidate), (candidate, ssb)):
                with self.subTest(source=source.frame_id, target=target.frame_id):
                    self.assertEqual(classify_frame_relation(source, target), AXES_OR_CONTEXT)

    def test_origin_diagnostic_excludes_synthetic_model_tcb_and_ssb_realizations(self) -> None:
        synthetic_a = synthetic_frame("frame.synthetic.a")
        synthetic_b = synthetic_frame(
            "frame.synthetic.b",
            origin_realization_id="origin.synthetic.other",
        )
        model_a = model_barycenter_frame("frame.model.a")
        model_b = inertial_origin_frame(
            "frame.model.b",
            origin_kind="NEWTONIAN_MODEL_BARYCENTER",
            origin_naif_id=None,
            origin_realization_id="origin.newtonian-model-barycenter.other",
        )
        body = inertial_origin_frame("frame.body")
        tcb_earth = inertial_origin_frame(
            "frame.tcb.earth", coordinate_time_scale="TCB"
        )
        tcb_sun = inertial_origin_frame(
            "frame.tcb.sun",
            origin_naif_id=10,
            origin_realization_id="origin.sun.fixture",
            coordinate_time_scale="TCB",
        )
        ssb_a = barycentric_tdb_frame("frame.ssb.a")
        ssb_b = barycentric_tdb_frame(
            "frame.ssb.b",
            origin_realization_id="origin.ssb.other-realization",
        )
        pairs = (
            (synthetic_a, synthetic_b),
            (model_a, body),
            (model_a, model_b),
            (tcb_earth, tcb_sun),
            (ssb_a, ssb_b),
        )
        for left, right in pairs:
            for source, target in ((left, right), (right, left)):
                with self.subTest(source=source.frame_id, target=target.frame_id):
                    self.assertEqual(classify_frame_relation(source, target), AXES_OR_CONTEXT)

    def test_axes_and_coordinate_context_changes_are_generic_unsupported(self) -> None:
        source = inertial_origin_frame("frame.source")
        candidates = (
            inertial_origin_frame(
                "frame.other-axes", axes_realization_id="axes.other.fixture"
            ),
            inertial_origin_frame(
                "frame.other-orientation",
                orientation_model_id="orientation.other.fixture",
            ),
            inertial_origin_frame("frame.tcb", coordinate_time_scale="TCB"),
        )
        for target in candidates:
            with self.subTest(target=target.frame_id):
                self.assertEqual(classify_frame_relation(source, target), AXES_OR_CONTEXT)
                with self.assertRaises(SolarSystemContractError):
                    require_exact_frame_contract_identity(source, target)


class FrameContractBoundaryTests(unittest.TestCase):
    def test_explicit_tdb_barycentric_fixture_is_only_a_declared_context(self) -> None:
        frame = barycentric_tdb_frame()
        validate_integrity(frame)
        self.assertEqual(frame.frame_kind, "TDB_COMPATIBLE_BARYCENTRIC_INERTIAL")
        self.assertEqual((frame.origin_kind, frame.origin_naif_id), ("SOLAR_SYSTEM_BARYCENTER", 0))
        self.assertEqual(frame.orientation_time_dependence, "STATIC")
        self.assertEqual(frame.coordinate_time_scale, "TDB")
        self.assertEqual(
            frame.content_sha256,
            "1b294dd8bbf99fe412ae71a9a3a09374719e4355e799eac7a5a69a8ed5fd6358",
        )

    def test_earth_origin_is_not_relabelled_as_gcrs_or_body_fixed(self) -> None:
        earth = inertial_origin_frame()
        self.assertEqual(earth.frame_kind, "INERTIAL_ORIGIN_CENTERED")
        self.assertEqual(earth.origin_kind, "BODY_CENTER")
        self.assertEqual(earth.origin_naif_id, 399)
        for forbidden_label in ("GCRS", "ITRS", "BODY_FIXED", "ECLIPTIC"):
            self.assertNotIn(forbidden_label, earth.frame_id)
            self.assertNotIn(forbidden_label, classify_frame_relation(earth, earth))

    def test_diagnostic_token_is_not_an_operation_plan_or_sufficiency_claim(self) -> None:
        token = classify_frame_relation(
            barycentric_tdb_frame(), inertial_origin_frame("frame.earth")
        )
        self.assertEqual(token, DIFFERENT_ORIGIN)
        self.assertIs(type(token), str)
        self.assertNotIn("PROVIDER_AVAILABLE", token)
        self.assertNotIn("TRANSFORM_EXECUTABLE", token)
        self.assertNotIn("SUFFICIENT", token)
        self.assertIn("diagnostic", frames_module.__doc__.lower())
        self.assertIn("not thereby gcrs", frames_module.__doc__.lower())

    def test_wrong_types_and_subclasses_fail_closed(self) -> None:
        frame = inertial_origin_frame()
        for invalid in (None, False, 0, 0.0, "frame", object()):
            with self.subTest(invalid=type(invalid).__name__):
                with self.assertRaises(SolarSystemContractError):
                    classify_frame_relation(frame, invalid)  # type: ignore[arg-type]
                with self.assertRaises(SolarSystemContractError):
                    classify_frame_relation(invalid, frame)  # type: ignore[arg-type]

        class FrameSubclass(FrameRealization):
            pass

        with self.assertRaises(SolarSystemContractError):
            FrameSubclass(
                frame_id="frame.subclass",
                frame_kind="INERTIAL_ORIGIN_CENTERED",
                origin_kind="BODY_CENTER",
                origin_naif_id=399,
                origin_realization_id="origin.body399.fixture",
                axes_realization_id=COMMON_AXES,
                orientation_model_id=COMMON_ORIENTATION,
                orientation_time_dependence="STATIC",
                coordinate_time_scale="TDB",
                coverage_status="NOT_APPLICABLE",
                coverage=None,
                artifact_ids=(),
            )

    def test_stale_outer_and_nested_seals_reject_before_classification(self) -> None:
        good = barycentric_tdb_frame("frame.good")

        stale_outer = barycentric_tdb_frame("frame.stale-outer")
        object.__setattr__(stale_outer, "artifact_ids", ("artifact.changed",))

        stale_interval = barycentric_tdb_frame("frame.stale-interval")
        assert stale_interval.coverage is not None
        object.__setattr__(stale_interval.coverage[0], "endpoint_policy", "OPEN_OPEN")

        stale_epoch = barycentric_tdb_frame("frame.stale-epoch")
        assert stale_epoch.coverage is not None
        object.__setattr__(stale_epoch.coverage[0].start, "whole", -9_999)

        for stale in (stale_outer, stale_interval, stale_epoch):
            with self.subTest(frame_id=stale.frame_id):
                with self.assertRaises(SolarSystemContractError):
                    classify_frame_relation(good, stale)
                with self.assertRaises(SolarSystemContractError):
                    classify_frame_relation(stale, good)

    def test_m1_coverage_count_cap_precedes_nested_traversal(self) -> None:
        class TraversalBomb:
            def __getattribute__(self, name: str) -> object:
                raise AssertionError("coverage item traversed before count rejection")

        oversized = tuple(TraversalBomb() for _ in range(17))
        with self.assertRaises(SolarSystemContractError):
            FrameRealization(
                frame_id="frame.oversized",
                frame_kind="TDB_COMPATIBLE_BARYCENTRIC_INERTIAL",
                origin_kind="SOLAR_SYSTEM_BARYCENTER",
                origin_naif_id=0,
                origin_realization_id="origin.ssb.fixture",
                axes_realization_id=COMMON_AXES,
                orientation_model_id=COMMON_ORIENTATION,
                orientation_time_dependence="STATIC",
                coordinate_time_scale="TDB",
                coverage_status="RETAINED",
                coverage=oversized,  # type: ignore[arg-type]
                artifact_ids=("artifact.spk.fixture",),
            )

    def test_relation_does_not_merge_or_authenticate_evidence(self) -> None:
        source = inertial_origin_frame(
            "frame.evidence.a", artifact_ids=("artifact.frame.a",)
        )
        target = inertial_origin_frame(
            "frame.evidence.b", artifact_ids=("artifact.frame.b",)
        )
        before = (source.artifact_ids, target.artifact_ids)
        self.assertEqual(classify_frame_relation(source, target), UNRESOLVED_ALIAS)
        self.assertEqual((source.artifact_ids, target.artifact_ids), before)
        self.assertNotEqual(source.artifact_ids, target.artifact_ids)


class FrameImportBoundaryTests(unittest.TestCase):
    def test_clean_import_is_dependency_free_and_root_remains_unpublished(self) -> None:
        code = """
import json, sys
import jxplanetx.solar_system.frames as frames_module
from jxplanetx import solar_system
forbidden = [
    name for name in (
        'numpy', 'astropy', 'erfa', 'spiceypy', 'jplephem', 'skyfield', 'rebound'
    ) if name in sys.modules
]
print(json.dumps({
    'exports': frames_module.__all__,
    'tokens': frames_module.FRAME_RELATION_TOKENS,
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
        self.assertEqual(tuple(payload["tokens"]), FRAME_RELATION_TOKENS)
        self.assertEqual(payload["root_exports"], [])
        self.assertEqual(payload["forbidden"], [])


if __name__ == "__main__":
    unittest.main()
