"""CSPICE binary64 query-projection tests for Solar-System Milestone 4C1."""

from __future__ import annotations

import base64
import dataclasses
from fractions import Fraction
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import unittest
from unittest import mock

from jxplanetx.solar_system import cspice_time as cspice_time_module
from jxplanetx.solar_system.contracts import (
    CoordinateEpoch,
    CoverageInterval,
    EphemerisQuerySpec,
    SolarSystemContractError,
    SolarSystemCoverageError,
    validate_integrity,
)
from jxplanetx.solar_system.cspice_time import (
    CspiceBinary64QueryProjectionReceipt,
    project_cspice_binary64_query,
    validate_cspice_binary64_query_projection_receipt,
)
from jxplanetx.solar_system.serialization import canonical_json
from jxplanetx.solar_system.units import (
    KILOMETRE,
    SECOND,
    convert_fraction_to_binary64,
)
from tests.test_solar_system_contracts import (
    SOURCE_ID,
    spk_context,
)


EXPECTED_EXPORTS = [
    "CspiceBinary64QueryProjectionReceipt",
    "project_cspice_binary64_query",
    "validate_cspice_binary64_query_projection_receipt",
]
EXPECTED_FIELDS = (
    "requested_query",
    "effective_query",
    "binary64_projection",
    "projection_stage",
    "exact_error_numerator",
    "exact_error_denominator",
    "exact_error_unit_id",
    "time_scale_operation_status",
    "provider_execution_status",
    "evidence_preservation_status",
    "artifact_custody_status",
    "content_integrity_class",
    "content_sha256",
)

PROJECTION_STAGE = (
    "REQUESTED_SPICE_TDB_ET_SECONDS_TO_EFFECTIVE_CSPICE_BINARY64_ET"
)
TIME_STATUS = "NO_TIME_SCALE_OR_ORIGIN_CONVERSION_SAME_TDB_ET_COORDINATE"
PROVIDER_STATUS = "NOT_EXECUTED_ARITHMETIC_INTERFACE_PROJECTION_ONLY"
EVIDENCE_STATUS = (
    "REQUESTED_EPOCH_ARTIFACT_IDS_PRESERVED_UNCHANGED_NO_NEW_EVIDENCE"
)
ARTIFACT_STATUS = "NO_ARTIFACT_AT_USE_CUSTODY_EVIDENCE"
INTEGRITY_CLASS = "UNAUTHENTICATED_CONTENT_INTEGRITY_ONLY"

RECEIPT_DOMAIN = (
    "jxplanetx.solar-system.cspice-time.binary64-query-projection-receipt."
    "content-integrity.v1"
)
RECEIPT_SCHEMA = "CspiceBinary64QueryProjectionReceipt.v1"
RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.cspice_time.CspiceBinary64QueryProjectionReceipt"
)
BINARY64_RECEIPT_QUALIFIED_NAME = (
    "jxplanetx.solar_system.units.Binary64UnitConversionReceipt"
)


def query_at(
    whole: int,
    fraction: float,
    *,
    query_id: str = "query.fixture.cspice-projection.requested",
    artifact_ids: tuple[str, ...] = (),
) -> EphemerisQuerySpec:
    query = spk_context()[1]
    epoch = CoordinateEpoch(
        time_scale=query.epoch.time_scale,
        representation=query.epoch.representation,
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=query.epoch.coordinate_unit_id,
        origin_id=query.epoch.origin_id,
        realization_id=query.epoch.realization_id,
        artifact_ids=artifact_ids,
    )
    return dataclasses.replace(
        query,
        query_id=query_id,
        epoch=epoch,
        content_sha256="",
    )


def query_with_exact_coverage(
    start: CoordinateEpoch,
    end: CoordinateEpoch,
    epoch: CoordinateEpoch,
    *,
    query_id: str = "query.fixture.cspice-projection.coverage",
) -> EphemerisQuerySpec:
    provider, query, _, _ = spk_context()
    coverage = CoverageInterval(start, end, "CLOSED_CLOSED")
    artifacts = tuple(
        dataclasses.replace(artifact, coverage=(coverage,), content_sha256="")
        if artifact.artifact_role == "SPK"
        else artifact
        for artifact in provider.artifacts
    )
    frame = dataclasses.replace(
        provider.native_frame,
        coverage=(coverage,),
        content_sha256="",
    )
    bounded_provider = dataclasses.replace(
        provider,
        artifacts=artifacts,
        native_frame=frame,
        content_sha256="",
    )
    return dataclasses.replace(
        query,
        query_id=query_id,
        provider=bounded_provider,
        epoch=epoch,
        frame=frame,
        content_sha256="",
    )


def make_receipt() -> CspiceBinary64QueryProjectionReceipt:
    return project_cspice_binary64_query(
        query_at(1, float.fromhex("0x1p-53")),
        effective_query_id="query.fixture.cspice-projection.effective",
    )


def independent_expanded_binary64_payload(
    receipt: CspiceBinary64QueryProjectionReceipt,
) -> tuple[object, ...]:
    child = receipt.binary64_projection
    return (
        BINARY64_RECEIPT_QUALIFIED_NAME,
        tuple(
            (field.name, getattr(child, field.name))
            for field in dataclasses.fields(child)
            if field.name != "content_sha256"
        ),
        ("content_sha256", child.content_sha256),
    )


def independent_projection_payload(
    receipt: CspiceBinary64QueryProjectionReceipt,
) -> tuple[object, ...]:
    return (
        RECEIPT_QUALIFIED_NAME,
        tuple(
            (
                field.name,
                independent_expanded_binary64_payload(receipt)
                if field.name == "binary64_projection"
                else getattr(receipt, field.name),
            )
            for field in dataclasses.fields(receipt)
            if field.name != "content_sha256"
        ),
    )


def independent_preimage(receipt: CspiceBinary64QueryProjectionReceipt) -> bytes:
    return (
        RECEIPT_DOMAIN.encode("ascii")
        + b"\x00"
        + RECEIPT_SCHEMA.encode("ascii")
        + b"\x00"
        + canonical_json(independent_projection_payload(receipt))
    )


class CspiceProjectionSchemaTests(unittest.TestCase):
    def test_exact_exports_fields_signatures_and_root_boundary(self) -> None:
        self.assertEqual(cspice_time_module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(cspice_time_module.__all__, sorted(EXPECTED_EXPORTS))
        self.assertEqual(
            tuple(
                field.name
                for field in dataclasses.fields(CspiceBinary64QueryProjectionReceipt)
            ),
            EXPECTED_FIELDS,
        )
        signature = inspect.signature(project_cspice_binary64_query)
        self.assertEqual(tuple(signature.parameters), ("requested_query", "effective_query_id"))
        self.assertEqual(
            signature.parameters["effective_query_id"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            tuple(
                inspect.signature(
                    validate_cspice_binary64_query_projection_receipt
                ).parameters
            ),
            ("value",),
        )
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_exact_fixed_stage_and_nonclaim_tokens(self) -> None:
        receipt = make_receipt()
        self.assertEqual(receipt.projection_stage, PROJECTION_STAGE)
        self.assertEqual(receipt.exact_error_unit_id, SECOND.unit_id)
        self.assertEqual(receipt.time_scale_operation_status, TIME_STATUS)
        self.assertEqual(receipt.provider_execution_status, PROVIDER_STATUS)
        self.assertEqual(receipt.evidence_preservation_status, EVIDENCE_STATUS)
        self.assertEqual(receipt.artifact_custody_status, ARTIFACT_STATUS)
        self.assertEqual(receipt.content_integrity_class, INTEGRITY_CLASS)

    def test_receipt_and_children_are_frozen_and_new_type_is_not_in_m1_serializer(self) -> None:
        receipt = make_receipt()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            receipt.projection_stage = "changed"  # type: ignore[misc]
        with self.assertRaises(SolarSystemContractError):
            canonical_json(receipt)
        self.assertTrue(canonical_json(independent_projection_payload(receipt)))


class CspiceProjectionArithmeticTests(unittest.TestCase):
    def test_zero_and_negative_half_have_canonical_effective_splits(self) -> None:
        zero = project_cspice_binary64_query(
            query_at(0, 0.0, query_id="query.zero"),
            effective_query_id="query.zero.effective",
        )
        self.assertEqual(zero.binary64_projection.rounded_value.hex(), "0x0.0p+0")
        self.assertEqual(
            (zero.exact_error_numerator, zero.exact_error_denominator),
            (0, 1),
        )
        self.assertEqual(zero.effective_query.epoch.whole, 0)
        self.assertEqual(zero.effective_query.epoch.fraction.hex(), "0x0.0p+0")

        negative_half = project_cspice_binary64_query(
            query_at(0, -0.5, query_id="query.negative-half"),
            effective_query_id="query.negative-half.effective",
        )
        self.assertEqual(
            negative_half.binary64_projection.rounded_value.hex(),
            "-0x1.0000000000000p-1",
        )
        self.assertEqual(negative_half.effective_query.epoch.whole, 0)
        self.assertEqual(
            negative_half.effective_query.epoch.fraction.hex(),
            "-0x1.0000000000000p-1",
        )

    def test_exact_projection_preserves_value_with_a_distinct_query(self) -> None:
        requested = query_at(0, 0.25, artifact_ids=(SOURCE_ID,))
        receipt = project_cspice_binary64_query(
            requested,
            effective_query_id="query.fixture.cspice-projection.exact-effective",
        )
        validate_cspice_binary64_query_projection_receipt(receipt)
        self.assertEqual(receipt.binary64_projection.rounded_value.hex(), "0x1.0000000000000p-2")
        self.assertEqual(receipt.binary64_projection.rounding_status, "EXACT")
        self.assertEqual(receipt.binary64_projection.rounding_direction, "EXACT")
        self.assertEqual((receipt.exact_error_numerator, receipt.exact_error_denominator), (0, 1))
        self.assertIs(receipt.requested_query, requested)
        self.assertNotEqual(receipt.effective_query.query_id, requested.query_id)
        self.assertNotEqual(receipt.effective_query.content_sha256, requested.content_sha256)
        self.assertEqual(receipt.effective_query.epoch.whole, 0)
        self.assertEqual(receipt.effective_query.epoch.fraction.hex(), "0x1.0000000000000p-2")
        self.assertEqual(receipt.effective_query.epoch.artifact_ids, (SOURCE_ID,))

    def test_positive_ties_round_both_directions_with_exact_signed_error(self) -> None:
        cases = (
            (
                float.fromhex("0x1p-53"),
                "0x1.0000000000000p+0",
                "BELOW_EXACT",
                (-1, 1 << 53),
            ),
            (
                float.fromhex("0x1.8p-52"),
                "0x1.0000000000002p+0",
                "ABOVE_EXACT",
                (1, 1 << 53),
            ),
        )
        for index, (fraction, result_hex, direction, error) in enumerate(cases):
            with self.subTest(direction=direction):
                receipt = project_cspice_binary64_query(
                    query_at(1, fraction, query_id=f"query.tie.{index}"),
                    effective_query_id=f"query.tie.{index}.effective",
                )
                self.assertEqual(receipt.binary64_projection.rounded_value.hex(), result_hex)
                self.assertEqual(receipt.binary64_projection.rounding_status, "ROUNDED")
                self.assertEqual(receipt.binary64_projection.rounding_direction, direction)
                self.assertEqual(
                    (receipt.exact_error_numerator, receipt.exact_error_denominator),
                    error,
                )

    def test_negative_tie_mirrors_direction_and_error(self) -> None:
        receipt = project_cspice_binary64_query(
            query_at(-1, -float.fromhex("0x1p-53"), query_id="query.negative.tie"),
            effective_query_id="query.negative.tie.effective",
        )
        self.assertEqual(receipt.binary64_projection.rounded_value.hex(), "-0x1.0000000000000p+0")
        self.assertEqual(receipt.binary64_projection.rounding_direction, "ABOVE_EXACT")
        self.assertEqual(
            (receipt.exact_error_numerator, receipt.exact_error_denominator),
            (1, 1 << 53),
        )

    def test_rounding_to_half_boundary_uses_canonical_next_whole_split(self) -> None:
        receipt = project_cspice_binary64_query(
            query_at(
                1,
                0.5 - float.fromhex("0x1p-53"),
                query_id="query.canonical-half",
            ),
            effective_query_id="query.canonical-half.effective",
        )
        self.assertEqual(receipt.binary64_projection.rounded_value.hex(), "0x1.8000000000000p+0")
        self.assertEqual(receipt.effective_query.epoch.whole, 2)
        self.assertEqual(receipt.effective_query.epoch.fraction.hex(), "-0x1.0000000000000p-1")
        self.assertEqual(
            (receipt.exact_error_numerator, receipt.exact_error_denominator),
            (1, 1 << 53),
        )

    def test_large_whole_rounding_error_is_retained_exactly(self) -> None:
        huge = 1 << 53
        template = spk_context()[1].epoch
        start = dataclasses.replace(
            template,
            whole=huge - 2,
            fraction=0.0,
            content_sha256="",
        )
        end = dataclasses.replace(
            template,
            whole=huge + 2,
            fraction=0.0,
            content_sha256="",
        )
        epoch = dataclasses.replace(
            template,
            whole=huge,
            fraction=0.25,
            content_sha256="",
        )
        query = query_with_exact_coverage(start, end, epoch, query_id="query.large-whole")
        receipt = project_cspice_binary64_query(
            query,
            effective_query_id="query.large-whole.effective",
        )
        self.assertEqual(receipt.binary64_projection.rounded_value.hex(), "0x1.0000000000000p+53")
        self.assertEqual(receipt.binary64_projection.rounding_direction, "BELOW_EXACT")
        self.assertEqual(
            (receipt.exact_error_numerator, receipt.exact_error_denominator),
            (-1, 4),
        )

    def test_effective_query_preserves_every_non_id_non_epoch_field_and_epoch_context(self) -> None:
        requested = query_at(1, float.fromhex("0x1p-53"), artifact_ids=(SOURCE_ID,))
        receipt = project_cspice_binary64_query(
            requested,
            effective_query_id="query.preservation.effective",
        )
        effective = receipt.effective_query
        for field in dataclasses.fields(EphemerisQuerySpec):
            if field.name in ("query_id", "epoch", "content_sha256"):
                continue
            self.assertIs(getattr(effective, field.name), getattr(requested, field.name))
        for field in (
            "time_scale",
            "representation",
            "coordinate_unit_id",
            "origin_id",
            "realization_id",
            "artifact_ids",
        ):
            self.assertEqual(getattr(effective.epoch, field), getattr(requested.epoch, field))
        self.assertEqual(effective.epoch.artifact_ids, (SOURCE_ID,))

    def test_outward_rounding_past_retained_coarse_boundary_rejects(self) -> None:
        end = query_at(1, float.fromhex("0x1.8p-52")).epoch
        start = dataclasses.replace(end, whole=0, fraction=0.0, content_sha256="")
        query = query_with_exact_coverage(start, end, end)
        with self.assertRaises(SolarSystemCoverageError):
            project_cspice_binary64_query(
                query,
                effective_query_id="query.coverage.rounded-out",
            )

    def test_overflow_threshold_rejects_without_receipt(self) -> None:
        threshold = (1 << 1024) - (1 << 970)
        start = CoordinateEpoch(
            "TDB",
            "SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
            threshold - 1,
            0.0,
            SECOND.unit_id,
            "SPICE_J2000_TDB_ORIGIN",
            "fixture.spk-et-realization",
            (),
        )
        end = dataclasses.replace(start, whole=threshold, content_sha256="")
        query = query_with_exact_coverage(start, end, end, query_id="query.overflow")
        with self.assertRaises(SolarSystemContractError):
            project_cspice_binary64_query(
                query,
                effective_query_id="query.overflow.effective",
            )

    def test_requested_et_retained_rational_cap_rejects(self) -> None:
        huge = 1 << 4_095
        template = spk_context()[1].epoch
        start = dataclasses.replace(
            template,
            whole=huge,
            fraction=0.0,
            content_sha256="",
        )
        end = dataclasses.replace(
            template,
            whole=huge + 1,
            fraction=0.0,
            content_sha256="",
        )
        epoch = dataclasses.replace(
            template,
            whole=huge,
            fraction=float.fromhex("0x0.0000000000001p-1022"),
            content_sha256="",
        )
        query = query_with_exact_coverage(
            start,
            end,
            epoch,
            query_id="query.retained-cap",
        )
        with self.assertRaises(SolarSystemContractError):
            project_cspice_binary64_query(
                query,
                effective_query_id="query.retained-cap.effective",
            )


class CspiceProjectionIntegrityTests(unittest.TestCase):
    def test_fixed_tokens_error_and_own_seal_mutations_reject(self) -> None:
        receipt = make_receipt()
        mutations = {
            "projection_stage": "OTHER_STAGE",
            "exact_error_numerator": 1,
            "exact_error_denominator": 3,
            "exact_error_unit_id": "unit.day.86400-si-seconds.defined-exact.v1",
            "time_scale_operation_status": "CONVERTED",
            "provider_execution_status": "EXECUTED",
            "evidence_preservation_status": "NEW_EVIDENCE",
            "artifact_custody_status": "VERIFIED",
            "content_integrity_class": "AUTHENTICATED",
            "content_sha256": "0" * 64,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field), self.assertRaises(SolarSystemContractError):
                dataclasses.replace(
                    receipt,
                    **{field: replacement},
                    **({} if field == "content_sha256" else {"content_sha256": ""}),
                )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                exact_error_numerator=False,
                content_sha256="",
            )

    def test_coherent_id_rewrites_reseal_but_semantic_and_child_rewrites_reject(self) -> None:
        receipt = make_receipt()
        changed_requested = dataclasses.replace(
            receipt.requested_query,
            query_id="query.changed.requested",
            content_sha256="",
        )
        reidentified_requested = dataclasses.replace(
            receipt,
            requested_query=changed_requested,
            content_sha256="",
        )
        validate_cspice_binary64_query_projection_receipt(reidentified_requested)
        self.assertNotEqual(reidentified_requested.content_sha256, receipt.content_sha256)

        changed_requested_epoch = query_at(
            2,
            0.0,
            query_id="query.changed.requested.epoch",
        )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                requested_query=changed_requested_epoch,
                content_sha256="",
            )

        changed_effective = dataclasses.replace(
            receipt.effective_query,
            query_id="query.changed.effective",
            content_sha256="",
        )
        reidentified = dataclasses.replace(
            receipt,
            effective_query=changed_effective,
            content_sha256="",
        )
        validate_cspice_binary64_query_projection_receipt(reidentified)
        self.assertNotEqual(reidentified.content_sha256, receipt.content_sha256)

        changed_child = convert_fraction_to_binary64(Fraction(1), SECOND, SECOND)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                binary64_projection=changed_child,
                content_sha256="",
            )

    def test_stale_nested_query_epoch_provider_and_child_seals_reject(self) -> None:
        receipt = make_receipt()
        stale_requested = dataclasses.replace(receipt.requested_query)
        object.__setattr__(stale_requested.epoch, "whole", 2)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                requested_query=stale_requested,
                content_sha256="",
            )

        fresh = make_receipt()
        stale_effective = dataclasses.replace(fresh.effective_query)
        object.__setattr__(stale_effective.provider.identity, "version", "changed")
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                fresh,
                effective_query=stale_effective,
                content_sha256="",
            )

        fresh = make_receipt()
        detached_source_unit = dataclasses.replace(SECOND)
        stale_child = convert_fraction_to_binary64(
            Fraction(1) + Fraction(1, 1 << 53),
            detached_source_unit,
            SECOND,
        )
        object.__setattr__(detached_source_unit, "numerator", 2)
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                fresh,
                binary64_projection=stale_child,
                content_sha256="",
            )
        validate_integrity(SECOND)

    def test_postconstruction_outer_mutation_rejects(self) -> None:
        receipt = make_receipt()
        object.__setattr__(receipt, "projection_stage", "OTHER")
        with self.assertRaises(SolarSystemContractError):
            receipt.validate_integrity()

    def test_exact_types_subclass_bool_and_same_query_id_reject(self) -> None:
        requested = query_at(0, 0.0)
        with self.assertRaises(SolarSystemContractError):
            project_cspice_binary64_query(
                requested,
                effective_query_id=requested.query_id,
            )
        for invalid_id in (False, "", " x", "x\n", "x" * 257):
            with self.subTest(invalid_id=invalid_id), self.assertRaises(SolarSystemContractError):
                project_cspice_binary64_query(
                    requested,
                    effective_query_id=invalid_id,  # type: ignore[arg-type]
                )
        with self.assertRaises(SolarSystemContractError):
            validate_cspice_binary64_query_projection_receipt(False)  # type: ignore[arg-type]

        class ReceiptSubclass(CspiceBinary64QueryProjectionReceipt):
            pass

        receipt = make_receipt()
        values = {
            field.name: getattr(receipt, field.name)
            for field in dataclasses.fields(receipt)
        }
        values["content_sha256"] = ""
        with self.assertRaises(SolarSystemContractError):
            ReceiptSubclass(**values)

    def test_effective_id_cap_precedes_requested_query_traversal(self) -> None:
        requested = query_at(0, 0.0)
        with mock.patch.object(
            cspice_time_module,
            "validate_integrity",
            side_effect=AssertionError("REQUESTED_QUERY_WAS_TRAVERSED"),
        ):
            with self.assertRaises(SolarSystemContractError) as caught:
                project_cspice_binary64_query(
                    requested,
                    effective_query_id="x" * 257,
                )
        self.assertNotIn("REQUESTED_QUERY_WAS_TRAVERSED", str(caught.exception))

    def test_integer_caps_and_reduced_error_pair_reject_before_gcd_work(self) -> None:
        receipt = make_receipt()
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                exact_error_numerator=1 << 4_096,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                exact_error_numerator=-2,
                exact_error_denominator=2,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                exact_error_numerator=0,
                exact_error_denominator=2,
                content_sha256="",
            )

    def test_literal_production_domain_schema_field_order_and_child_expansion(self) -> None:
        receipt = make_receipt()
        preimage = independent_preimage(receipt)
        self.assertEqual(len(preimage), PROJECTION_LITERAL_PREIMAGE_LENGTH)
        self.assertEqual(hashlib.sha256(preimage).hexdigest(), PROJECTION_LITERAL_SHA256)
        self.assertEqual(preimage, PROJECTION_LITERAL_PREIMAGE)
        self.assertEqual(receipt.content_sha256, PROJECTION_LITERAL_SHA256)
        self.assertIn(receipt.binary64_projection.content_sha256.encode("ascii"), preimage)

        variants = (
            preimage.replace(RECEIPT_DOMAIN.encode("ascii"), (RECEIPT_DOMAIN + ".changed").encode("ascii"), 1),
            preimage.replace(RECEIPT_SCHEMA.encode("ascii"), (RECEIPT_SCHEMA + ".changed").encode("ascii"), 1),
            preimage.replace(b'[["requested_query",', b'[["effective_query",', 1),
            preimage.replace(
                receipt.binary64_projection.content_sha256.encode("ascii"),
                b"0" * 64,
                1,
            ),
        )
        for variant in variants:
            self.assertNotEqual(hashlib.sha256(variant).hexdigest(), receipt.content_sha256)

    def test_sorted_json_roundtrip_of_explicit_payload_preserves_content(self) -> None:
        receipt = make_receipt()
        encoded = canonical_json(independent_projection_payload(receipt))
        loaded = json.loads(encoded)
        pretty = json.dumps(loaded, sort_keys=True, indent=2)
        self.assertEqual(json.loads(pretty), loaded)
        self.assertEqual(hashlib.sha256(independent_preimage(receipt)).hexdigest(), receipt.content_sha256)


class CspiceProjectionBoundaryTests(unittest.TestCase):
    def test_clean_import_has_no_provider_numeric_or_network_dependencies(self) -> None:
        code = """
import json, sys
import jxplanetx.solar_system.cspice_time as module
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

    def test_module_nonclaims_and_effective_query_rule_are_explicit(self) -> None:
        text = " ".join(cspice_time_module.__doc__.lower().split())
        for phrase in (
            "effective et - requested et",
            "not a time-scale or origin conversion",
            "an ephemeris interpolation or state-error bound",
            "does not execute",
            "no astronomy provider",
            "target-specific coverage",
            "artifact-at-use custody",
            "future provider-native state and execution receipts must bind ``effective_query``",
            "never relabel a rounded result",
            "unauthenticated content integrity only",
        ):
            self.assertIn(phrase, text)


# Locked after the complete production-domain fixture compiles coherently.
PROJECTION_LITERAL_PREIMAGE_LENGTH = 28_672
PROJECTION_LITERAL_SHA256 = "331e8638ffdb3802ad1d18b0d2d64d46b68c9a41b5d5cc1e10a2d581a1980772"
PROJECTION_LITERAL_PREIMAGE = base64.b64decode(
    "anhwbGFuZXR4LnNvbGFyLXN5c3RlbS5jc3BpY2UtdGltZS5iaW5hcnk2NC1xdWVyeS1wcm9qZWN0aW9uLXJlY2VpcHQuY29udGVu"
    "dC1pbnRlZ3JpdHkudjEAQ3NwaWNlQmluYXJ5NjRRdWVyeVByb2plY3Rpb25SZWNlaXB0LnYxAFsianhwbGFuZXR4LnNvbGFyX3N5"
    "c3RlbS5jc3BpY2VfdGltZS5Dc3BpY2VCaW5hcnk2NFF1ZXJ5UHJvamVjdGlvblJlY2VpcHQiLFtbInJlcXVlc3RlZF9xdWVyeSIs"
    "eyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FcGhlbWVyaXNRdWVyeVNwZWMiLCJmaWVsZHMi"
    "OltbInF1ZXJ5X2lkIiwicXVlcnkuZml4dHVyZS5jc3BpY2UtcHJvamVjdGlvbi5yZXF1ZXN0ZWQiXSxbInByb3ZpZGVyIix7ImRh"
    "dGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkVwaGVtZXJpc1Byb3ZpZGVyU3BlYyIsImZpZWxkcyI6"
    "W1sic3BlY19pZCIsInByb3ZpZGVyLmZpeHR1cmUuc3BlYy52MSJdLFsiaWRlbnRpdHkiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4"
    "LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuUHJvdmlkZXJJZGVudGl0eSIsImZpZWxkcyI6W1sicHJvdmlkZXJfaWQiLCJwcm92aWRl"
    "ci5maXh0dXJlIl0sWyJpbXBsZW1lbnRhdGlvbl9pZCIsInByb3ZpZGVyLmZpeHR1cmUuaW1wbGVtZW50YXRpb24udjEiXSxbInBy"
    "b3ZpZGVyX2tpbmQiLCJMT0NBTF9PRkZMSU5FX1NQSyJdLFsidmVyc2lvbiIsIjEiXSxbInJ1bnRpbWVfaWQiLCJjcHl0aG9uLnRl"
    "c3QiXSxbIm1vZHVsZV9uYW1lIiwiZml4dHVyZV9wcm92aWRlciJdLFsiZGlzdHJpYnV0aW9uX25hbWUiLCJmaXh0dXJlLXByb3Zp"
    "ZGVyIl0sWyJkaXN0cmlidXRpb25fdmVyc2lvbiIsIjEiXSxbIm9yZGVyZWRfaW1wbGVtZW50YXRpb25fYXJ0aWZhY3RfaWRzIixb"
    "ImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiMTRkMzFjNzg1ZjdkOGU4MWI4OTI1ZjExMGE5"
    "NzUxZmUyYjNjOThjMmE1Mzk0NzMwYjk5MzRkNTNkY2Q0Zjk2ZiJdXX1dLFsiYXJ0aWZhY3RzIixbeyJkYXRhY2xhc3MiOiJqeHBs"
    "YW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5BcnRpZmFjdEJpbmRpbmciLCJmaWVsZHMiOltbImFydGlmYWN0X2lkIiwiYXJ0"
    "aWZhY3QuZml4dHVyZS5zcGsiXSxbImFydGlmYWN0X3JvbGUiLCJTUEsiXSxbInByb3ZpZGVyX2lkIiwicHJvdmlkZXIuZml4dHVy"
    "ZSJdLFsidmVyc2lvbiIsIjEiXSxbImxvZ2ljYWxfbG9jYXRvciIsInJldGFpbmVkL2FydGlmYWN0LmZpeHR1cmUuc3BrLmJpbiJd"
    "LFsibG9jYXRvcl9raW5kIiwiTE9DQUxfUkVHVUxBUl9GSUxFIl0sWyJieXRlX2xlbmd0aCIsMTI4XSxbImFydGlmYWN0X3NoYTI1"
    "NiIsIjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIiXSxbIm1l"
    "ZGlhX3R5cGUiLCJhcHBsaWNhdGlvbi9vY3RldC1zdHJlYW0iXSxbImNvdmVyYWdlX3N0YXR1cyIsIkNPQVJTRV9BUlRJRkFDVF9U"
    "SU1FX0VOVkVMT1BFIl0sWyJjb3ZlcmFnZSIsW3siZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMu"
    "Q292ZXJhZ2VJbnRlcnZhbCIsImZpZWxkcyI6W1sic3RhcnQiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5j"
    "b250cmFjdHMuQ29vcmRpbmF0ZUVwb2NoIiwiZmllbGRzIjpbWyJ0aW1lX3NjYWxlIiwiVERCIl0sWyJyZXByZXNlbnRhdGlvbiIs"
    "IlNQSUNFX1REQl9KMjAwMF9PRkZTRVRfU0VDT05EU19UV09fUEFSVCJdLFsid2hvbGUiLC04NjQwMF0sWyJmcmFjdGlvbiIseyJm"
    "bG9hdF9oZXgiOiIweDAuMHArMCJ9XSxbImNvb3JkaW5hdGVfdW5pdF9pZCIsInVuaXQuc2Vjb25kLmRlZmluZWQtZXhhY3QudjEi"
    "XSxbIm9yaWdpbl9pZCIsIlNQSUNFX0oyMDAwX1REQl9PUklHSU4iXSxbInJlYWxpemF0aW9uX2lkIiwiZml4dHVyZS5zcGstZXQt"
    "cmVhbGl6YXRpb24iXSxbImFydGlmYWN0X2lkcyIsW11dLFsiY29udGVudF9zaGEyNTYiLCI5YTVkYjU0MDQ4MzVlNWFmNDgwNGQ4"
    "MTQ3ZjE0NmRjNTVjYTAyNmNjZTQxMTlmYzA2ODhjNzhiYTJjM2I3YzNkIl1dfV0sWyJlbmQiLHsiZGF0YWNsYXNzIjoianhwbGFu"
    "ZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuQ29vcmRpbmF0ZUVwb2NoIiwiZmllbGRzIjpbWyJ0aW1lX3NjYWxlIiwiVERCIl0s"
    "WyJyZXByZXNlbnRhdGlvbiIsIlNQSUNFX1REQl9KMjAwMF9PRkZTRVRfU0VDT05EU19UV09fUEFSVCJdLFsid2hvbGUiLDg2NDAw"
    "XSxbImZyYWN0aW9uIix7ImZsb2F0X2hleCI6IjB4MC4wcCswIn1dLFsiY29vcmRpbmF0ZV91bml0X2lkIiwidW5pdC5zZWNvbmQu"
    "ZGVmaW5lZC1leGFjdC52MSJdLFsib3JpZ2luX2lkIiwiU1BJQ0VfSjIwMDBfVERCX09SSUdJTiJdLFsicmVhbGl6YXRpb25faWQi"
    "LCJmaXh0dXJlLnNway1ldC1yZWFsaXphdGlvbiJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsImU4ZDA3"
    "NGQ3MzcwZTRiYjk4OWM0MDc2MDg0MDViNjQzNTI0YWNjNDI0ZWFhYjNkOWUwMmRmNmRmMmE5NWEwZjYiXV19XSxbImVuZHBvaW50"
    "X3BvbGljeSIsIkNMT1NFRF9DTE9TRUQiXSxbImNvbnRlbnRfc2hhMjU2IiwiZTc5YTE3NTUwYjBkZTBhZGQyNTNiYWVkZTBlMmQy"
    "MTMwYmQ4MmEzZTA5NTk0YTkyODExYmFiNDM1MTQ4Nzc4ZSJdXX1dXSxbImxpY2Vuc2VfZXZpZGVuY2Vfc3RhdHVzIiwiRVhURVJO"
    "QUxfTk9UX1JFRElTVFJJQlVURURfTElDRU5TRV9OT1RfUkVUQUlORURfTk9OQVVUSE9SSVpJTkciXSxbImxpY2Vuc2Vfc3BkeCIs"
    "IkJTRC0zLUNsYXVzZSJdLFsibGljZW5zZV9hcnRpZmFjdF9pZCIsbnVsbF0sWyJyZWRpc3RyaWJ1dGlvbl9zdGF0dXMiLCJFWFRF"
    "Uk5BTF9SRUZFUkVOQ0VfTk9UX1JFRElTVFJJQlVURUQiXSxbImxvYWRfb3JkZXJfc3RhdHVzIiwiT1JERVJFRF9MT0FEX01FTUJF"
    "UiJdLFsibG9hZF9vcmRlciIsMF0sWyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJRCJdLFsiY29udGVudF9pbnRlZ3JpdHlf"
    "Y2xhc3MiLCJVTkFVVEhFTlRJQ0FURURfQ09OVEVOVF9JTlRFR1JJVFlfT05MWSJdLFsiY29udGVudF9zaGEyNTYiLCIyZDgwNzU3"
    "MGQzOTJiODhmNGQ1Y2Q3NWRjZjgwMDNlZmIwN2JlMjdlOTUyZDFlNThiMzZiY2FkZDU0OWVjMDljIl1dfSx7ImRhdGFjbGFzcyI6"
    "Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkFydGlmYWN0QmluZGluZyIsImZpZWxkcyI6W1siYXJ0aWZhY3RfaWQi"
    "LCJhcnRpZmFjdC5wcm92aWRlci5saWNlbnNlIl0sWyJhcnRpZmFjdF9yb2xlIiwiTElDRU5TRSJdLFsicHJvdmlkZXJfaWQiLCJw"
    "cm92aWRlci5maXh0dXJlIl0sWyJ2ZXJzaW9uIiwiMSJdLFsibG9naWNhbF9sb2NhdG9yIiwicmV0YWluZWQvTElDRU5TRS50eHQi"
    "XSxbImxvY2F0b3Jfa2luZCIsIkxPQ0FMX1JFR1VMQVJfRklMRSJdLFsiYnl0ZV9sZW5ndGgiLDY0XSxbImFydGlmYWN0X3NoYTI1"
    "NiIsIjMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMiXSxbIm1l"
    "ZGlhX3R5cGUiLCJ0ZXh0L3BsYWluIl0sWyJjb3ZlcmFnZV9zdGF0dXMiLCJOT1RfQVBQTElDQUJMRSJdLFsiY292ZXJhZ2UiLG51"
    "bGxdLFsibGljZW5zZV9ldmlkZW5jZV9zdGF0dXMiLCJSRVRBSU5FRF9MSUNFTlNFX1RFWFRfU0VMRl9FVklERU5DRV9OT05BVVRI"
    "T1JJWklORyJdLFsibGljZW5zZV9zcGR4IiwiQlNELTMtQ2xhdXNlIl0sWyJsaWNlbnNlX2FydGlmYWN0X2lkIixudWxsXSxbInJl"
    "ZGlzdHJpYnV0aW9uX3N0YXR1cyIsIkJVTkRMRURfV0lUSF9SRVRBSU5FRF9MSUNFTlNFX0VWSURFTkNFIl0sWyJsb2FkX29yZGVy"
    "X3N0YXR1cyIsIk5PVF9MT0FEQUJMRSJdLFsibG9hZF9vcmRlciIsbnVsbF0sWyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJ"
    "RCJdLFsiY29udGVudF9pbnRlZ3JpdHlfY2xhc3MiLCJVTkFVVEhFTlRJQ0FURURfQ09OVEVOVF9JTlRFR1JJVFlfT05MWSJdLFsi"
    "Y29udGVudF9zaGEyNTYiLCI0OWM0NWM5NWZlYWYyODI4YjMxNGQ2NDU3OGFmYmNhZjNmYzQ4NzZlZWZjZWMyMDcxNTkzMzI4NTZh"
    "NDhlZmEwIl1dfSx7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkFydGlmYWN0QmluZGluZyIs"
    "ImZpZWxkcyI6W1siYXJ0aWZhY3RfaWQiLCJhcnRpZmFjdC5wcm92aWRlci5zb3VyY2UiXSxbImFydGlmYWN0X3JvbGUiLCJTT0ZU"
    "V0FSRV9TT1VSQ0UiXSxbInByb3ZpZGVyX2lkIiwicHJvdmlkZXIuZml4dHVyZSJdLFsidmVyc2lvbiIsIjEiXSxbImxvZ2ljYWxf"
    "bG9jYXRvciIsInJldGFpbmVkL2FydGlmYWN0LnByb3ZpZGVyLnNvdXJjZS5iaW4iXSxbImxvY2F0b3Jfa2luZCIsIkxPQ0FMX1JF"
    "R1VMQVJfRklMRSJdLFsiYnl0ZV9sZW5ndGgiLDEyOF0sWyJhcnRpZmFjdF9zaGEyNTYiLCIxMTExMTExMTExMTExMTExMTExMTEx"
    "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExIl0sWyJtZWRpYV90eXBlIiwiYXBwbGljYXRpb24vb2N0"
    "ZXQtc3RyZWFtIl0sWyJjb3ZlcmFnZV9zdGF0dXMiLCJOT1RfQVBQTElDQUJMRSJdLFsiY292ZXJhZ2UiLG51bGxdLFsibGljZW5z"
    "ZV9ldmlkZW5jZV9zdGF0dXMiLCJSRVRBSU5FRF9IQVNIX0JPVU5EX0xJQ0VOU0VfQVJUSUZBQ1QiXSxbImxpY2Vuc2Vfc3BkeCIs"
    "IkJTRC0zLUNsYXVzZSJdLFsibGljZW5zZV9hcnRpZmFjdF9pZCIsImFydGlmYWN0LnByb3ZpZGVyLmxpY2Vuc2UiXSxbInJlZGlz"
    "dHJpYnV0aW9uX3N0YXR1cyIsIkJVTkRMRURfV0lUSF9SRVRBSU5FRF9MSUNFTlNFX0VWSURFTkNFIl0sWyJsb2FkX29yZGVyX3N0"
    "YXR1cyIsIk5PVF9MT0FEQUJMRSJdLFsibG9hZF9vcmRlciIsbnVsbF0sWyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJRCJd"
    "LFsiY29udGVudF9pbnRlZ3JpdHlfY2xhc3MiLCJVTkFVVEhFTlRJQ0FURURfQ09OVEVOVF9JTlRFR1JJVFlfT05MWSJdLFsiY29u"
    "dGVudF9zaGEyNTYiLCIzOWE5OTUxZWFmZjFhZDBjODk3OWJjNGMwMzc0MWI2MDdjZDQyNTkzY2FjYTk0OWVmM2FjNzVlNDI4NDFl"
    "MzM3Il1dfV1dLFsib3JkZXJlZF9sb2FkX2FydGlmYWN0X2lkcyIsWyJhcnRpZmFjdC5maXh0dXJlLnNwayJdXSxbImNhcGFiaWxp"
    "dGllcyIsWyJDT0FSU0VfQVJUSUZBQ1RfVElNRV9FTlZFTE9QRSIsIkdFT01FVFJJQ19DQVJURVNJQU5fU1RBVEUiLCJUREJfQ09P"
    "UkRJTkFURV9FUE9DSCJdXSxbIm5hdGl2ZV90aW1lX3NjYWxlIiwiVERCIl0sWyJuYXRpdmVfZnJhbWUiLHsiZGF0YWNsYXNzIjoi"
    "anhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRnJhbWVSZWFsaXphdGlvbiIsImZpZWxkcyI6W1siZnJhbWVfaWQiLCJm"
    "cmFtZS5maXh0dXJlLmRlLWljcmYiXSxbImZyYW1lX2tpbmQiLCJUREJfQ09NUEFUSUJMRV9CQVJZQ0VOVFJJQ19JTkVSVElBTCJd"
    "LFsib3JpZ2luX2tpbmQiLCJTT0xBUl9TWVNURU1fQkFSWUNFTlRFUiJdLFsib3JpZ2luX25haWZfaWQiLDBdLFsib3JpZ2luX3Jl"
    "YWxpemF0aW9uX2lkIiwiZml4dHVyZS5kZS1yZWxlYXNlLnNzYiJdLFsiYXhlc19yZWFsaXphdGlvbl9pZCIsImZpeHR1cmUuZGUt"
    "cmVsZWFzZS5pY3JmIl0sWyJvcmllbnRhdGlvbl9tb2RlbF9pZCIsImZpeHR1cmUuZGUtcmVsZWFzZS5vcmllbnRhdGlvbiJdLFsi"
    "b3JpZW50YXRpb25fdGltZV9kZXBlbmRlbmNlIiwiU1RBVElDIl0sWyJjb29yZGluYXRlX3RpbWVfc2NhbGUiLCJUREIiXSxbImNv"
    "dmVyYWdlX3N0YXR1cyIsIlJFVEFJTkVEIl0sWyJjb3ZlcmFnZSIsW3siZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3Rl"
    "bS5jb250cmFjdHMuQ292ZXJhZ2VJbnRlcnZhbCIsImZpZWxkcyI6W1sic3RhcnQiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNv"
    "bGFyX3N5c3RlbS5jb250cmFjdHMuQ29vcmRpbmF0ZUVwb2NoIiwiZmllbGRzIjpbWyJ0aW1lX3NjYWxlIiwiVERCIl0sWyJyZXBy"
    "ZXNlbnRhdGlvbiIsIlNQSUNFX1REQl9KMjAwMF9PRkZTRVRfU0VDT05EU19UV09fUEFSVCJdLFsid2hvbGUiLC04NjQwMF0sWyJm"
    "cmFjdGlvbiIseyJmbG9hdF9oZXgiOiIweDAuMHArMCJ9XSxbImNvb3JkaW5hdGVfdW5pdF9pZCIsInVuaXQuc2Vjb25kLmRlZmlu"
    "ZWQtZXhhY3QudjEiXSxbIm9yaWdpbl9pZCIsIlNQSUNFX0oyMDAwX1REQl9PUklHSU4iXSxbInJlYWxpemF0aW9uX2lkIiwiZml4"
    "dHVyZS5zcGstZXQtcmVhbGl6YXRpb24iXSxbImFydGlmYWN0X2lkcyIsW11dLFsiY29udGVudF9zaGEyNTYiLCI5YTVkYjU0MDQ4"
    "MzVlNWFmNDgwNGQ4MTQ3ZjE0NmRjNTVjYTAyNmNjZTQxMTlmYzA2ODhjNzhiYTJjM2I3YzNkIl1dfV0sWyJlbmQiLHsiZGF0YWNs"
    "YXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuQ29vcmRpbmF0ZUVwb2NoIiwiZmllbGRzIjpbWyJ0aW1lX3Nj"
    "YWxlIiwiVERCIl0sWyJyZXByZXNlbnRhdGlvbiIsIlNQSUNFX1REQl9KMjAwMF9PRkZTRVRfU0VDT05EU19UV09fUEFSVCJdLFsi"
    "d2hvbGUiLDg2NDAwXSxbImZyYWN0aW9uIix7ImZsb2F0X2hleCI6IjB4MC4wcCswIn1dLFsiY29vcmRpbmF0ZV91bml0X2lkIiwi"
    "dW5pdC5zZWNvbmQuZGVmaW5lZC1leGFjdC52MSJdLFsib3JpZ2luX2lkIiwiU1BJQ0VfSjIwMDBfVERCX09SSUdJTiJdLFsicmVh"
    "bGl6YXRpb25faWQiLCJmaXh0dXJlLnNway1ldC1yZWFsaXphdGlvbiJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3No"
    "YTI1NiIsImU4ZDA3NGQ3MzcwZTRiYjk4OWM0MDc2MDg0MDViNjQzNTI0YWNjNDI0ZWFhYjNkOWUwMmRmNmRmMmE5NWEwZjYiXV19"
    "XSxbImVuZHBvaW50X3BvbGljeSIsIkNMT1NFRF9DTE9TRUQiXSxbImNvbnRlbnRfc2hhMjU2IiwiZTc5YTE3NTUwYjBkZTBhZGQy"
    "NTNiYWVkZTBlMmQyMTMwYmQ4MmEzZTA5NTk0YTkyODExYmFiNDM1MTQ4Nzc4ZSJdXX1dXSxbImFydGlmYWN0X2lkcyIsWyJhcnRp"
    "ZmFjdC5maXh0dXJlLnNwayJdXSxbImNvbnRlbnRfc2hhMjU2IiwiNmM3NDIzZDQ2YmM2MWJiOWFjOGFmNjk2NDc4ZjI3YWZlZTAy"
    "NTg3NTI2MjIzYjExZTI5MjgyZTc5NDE3MzY0MiJdXX1dLFsibmF0aXZlX291dHB1dF91bml0X3N5c3RlbSIseyJkYXRhY2xhc3Mi"
    "OiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Vbml0U3lzdGVtRGVmaW5pdGlvbiIsImZpZWxkcyI6W1sidW5pdF9z"
    "eXN0ZW1faWQiLCJ1bml0cy5zaS10ZGIiXSxbImxlbmd0aCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNv"
    "bnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9pZCIsInVuaXQubWV0cmUiXSxbImRpbWVuc2lvbiIsIkxF"
    "TkdUSCJdLFsic2lfdW5pdF9pZCIsInNpLm1ldHJlIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3IiLDFdLFsiZGVmaW5p"
    "dGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRpZmFjdC5wcm92aWRlci5z"
    "b3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsImQ1Y2YzNGI2Nzk5NjdiOTYyMWY5NjY2NWJiZGVjNDlkMmI5ODViNWIwMmUwOTli"
    "MzIxZjUwNmE4NjZkNmU2ZmMiXV19XSxbInRpbWUiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFj"
    "dHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0LnNlY29uZC5kZWZpbmVkLWV4YWN0LnYxIl0sWyJk"
    "aW1lbnNpb24iLCJUSU1FIl0sWyJzaV91bml0X2lkIiwic2kuc2Vjb25kIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3Ii"
    "LDFdLFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRpZmFj"
    "dC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsIjU4M2E1YjA4YTU4MGVjNDJiNTY0NDIwZmE0YWZmYzNlZDQz"
    "YzFkZjljNDA0NzFiZDU3OWViODRkMDA3YTM5NzQiXV19XSxbIm1hc3MiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5"
    "c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0LmtpbG9ncmFtIl0sWyJkaW1l"
    "bnNpb24iLCJNQVNTIl0sWyJzaV91bml0X2lkIiwic2kua2lsb2dyYW0iXSxbIm51bWVyYXRvciIsMV0sWyJkZW5vbWluYXRvciIs"
    "MV0sWyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0"
    "LnByb3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiNDk2NWRiYjI4N2ZhNWY3MWJjZWUwODg5MWU0MzcwZTJmM2M1"
    "OGNlMzBiYmVhNWI0MmZkYjYxNTQ5MmU0YWIwNyJdXX1dLFsiZ3Jhdml0YXRpb25hbF9wYXJhbWV0ZXIiLHsiZGF0YWNsYXNzIjoi"
    "anhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0"
    "Lm1ldHJlMy1wZXItc2Vjb25kMiJdLFsiZGltZW5zaW9uIiwiR1JBVklUQVRJT05BTF9QQVJBTUVURVIiXSxbInNpX3VuaXRfaWQi"
    "LCJzaS5tZXRyZTMtcGVyLXNlY29uZDIiXSxbIm51bWVyYXRvciIsMV0sWyJkZW5vbWluYXRvciIsMV0sWyJkZWZpbml0aW9uX2Ns"
    "YXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJd"
    "XSxbImNvbnRlbnRfc2hhMjU2IiwiMDgxZWZmYzMxZWU1NTNjMTJjNTZiODVjNmY3OTJkZDgwMTQ0NTliMGU5MThmNTE4MjZhYjM4"
    "MTUwYWExY2ZhNyJdXX1dLFsiY29vcmRpbmF0ZV90aW1lX3NjYWxlIiwiVERCIl0sWyJjb250ZW50X3NoYTI1NiIsIjIzMjg1NDY5"
    "ZGU4MTliYjIwYThhNGU4NTIxMjJhOWE5ZDNmM2JhMGFhM2I2YjI0Zjc4ZGZiMTc5NDJhMzZmNDAiXV19XSxbIm5ldHdvcmtfYWNj"
    "ZXNzIixmYWxzZV0sWyJmYWxsYmFja19hbGxvd2VkIixmYWxzZV0sWyJleHRyYXBvbGF0aW9uX3BvbGljeSIsIkZPUkJJRCJdLFsi"
    "ZXZpZGVuY2VfY2xhc3MiLCJNT0RFTF9PVVRQVVQiXSxbInJlZ2lzdHJ5X2F1dGhvcml6ZWQiLGZhbHNlXSxbInF1YWxpZmljYXRp"
    "b25fYXV0aG9yaXplZCIsZmFsc2VdLFsiY29udGVudF9zaGEyNTYiLCIzZjc1NGI3ZmM5NzUzNWMyYjE1YzkxYjc0OWY3YWM3MDY1"
    "NTEyNDkxMTJiOTg1Nzg4Y2IwMDJhZDAxN2ZkOTRiIl1dfV0sWyJlcG9jaCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJf"
    "c3lzdGVtLmNvbnRyYWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltbInRpbWVfc2NhbGUiLCJUREIiXSxbInJlcHJlc2Vu"
    "dGF0aW9uIiwiU1BJQ0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19QQVJUIl0sWyJ3aG9sZSIsMV0sWyJmcmFjdGlvbiIs"
    "eyJmbG9hdF9oZXgiOiIweDEuMDAwMDAwMDAwMDAwMHAtNTMifV0sWyJjb29yZGluYXRlX3VuaXRfaWQiLCJ1bml0LnNlY29uZC5k"
    "ZWZpbmVkLWV4YWN0LnYxIl0sWyJvcmlnaW5faWQiLCJTUElDRV9KMjAwMF9UREJfT1JJR0lOIl0sWyJyZWFsaXphdGlvbl9pZCIs"
    "ImZpeHR1cmUuc3BrLWV0LXJlYWxpemF0aW9uIl0sWyJhcnRpZmFjdF9pZHMiLFtdXSxbImNvbnRlbnRfc2hhMjU2IiwiMzQwNmE2"
    "NTE3NjU3ZDI0MWNmMDE2Y2Q5N2Y5OTBlNjkyODE0Zjk3ZDNhYjU0Njk5NDhhNTNmMzI1ODkyMDA3MyJdXX1dLFsidGFyZ2V0X25h"
    "aWZfaWRzIixbMTAsMzk5XV0sWyJvYnNlcnZlcl9uYWlmX2lkIiwwXSxbImZyYW1lIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5z"
    "b2xhcl9zeXN0ZW0uY29udHJhY3RzLkZyYW1lUmVhbGl6YXRpb24iLCJmaWVsZHMiOltbImZyYW1lX2lkIiwiZnJhbWUuZml4dHVy"
    "ZS5kZS1pY3JmIl0sWyJmcmFtZV9raW5kIiwiVERCX0NPTVBBVElCTEVfQkFSWUNFTlRSSUNfSU5FUlRJQUwiXSxbIm9yaWdpbl9r"
    "aW5kIiwiU09MQVJfU1lTVEVNX0JBUllDRU5URVIiXSxbIm9yaWdpbl9uYWlmX2lkIiwwXSxbIm9yaWdpbl9yZWFsaXphdGlvbl9p"
    "ZCIsImZpeHR1cmUuZGUtcmVsZWFzZS5zc2IiXSxbImF4ZXNfcmVhbGl6YXRpb25faWQiLCJmaXh0dXJlLmRlLXJlbGVhc2UuaWNy"
    "ZiJdLFsib3JpZW50YXRpb25fbW9kZWxfaWQiLCJmaXh0dXJlLmRlLXJlbGVhc2Uub3JpZW50YXRpb24iXSxbIm9yaWVudGF0aW9u"
    "X3RpbWVfZGVwZW5kZW5jZSIsIlNUQVRJQyJdLFsiY29vcmRpbmF0ZV90aW1lX3NjYWxlIiwiVERCIl0sWyJjb3ZlcmFnZV9zdGF0"
    "dXMiLCJSRVRBSU5FRCJdLFsiY292ZXJhZ2UiLFt7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3Rz"
    "LkNvdmVyYWdlSW50ZXJ2YWwiLCJmaWVsZHMiOltbInN0YXJ0Iix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0u"
    "Y29udHJhY3RzLkNvb3JkaW5hdGVFcG9jaCIsImZpZWxkcyI6W1sidGltZV9zY2FsZSIsIlREQiJdLFsicmVwcmVzZW50YXRpb24i"
    "LCJTUElDRV9UREJfSjIwMDBfT0ZGU0VUX1NFQ09ORFNfVFdPX1BBUlQiXSxbIndob2xlIiwtODY0MDBdLFsiZnJhY3Rpb24iLHsi"
    "ZmxvYXRfaGV4IjoiMHgwLjBwKzAifV0sWyJjb29yZGluYXRlX3VuaXRfaWQiLCJ1bml0LnNlY29uZC5kZWZpbmVkLWV4YWN0LnYx"
    "Il0sWyJvcmlnaW5faWQiLCJTUElDRV9KMjAwMF9UREJfT1JJR0lOIl0sWyJyZWFsaXphdGlvbl9pZCIsImZpeHR1cmUuc3BrLWV0"
    "LXJlYWxpemF0aW9uIl0sWyJhcnRpZmFjdF9pZHMiLFtdXSxbImNvbnRlbnRfc2hhMjU2IiwiOWE1ZGI1NDA0ODM1ZTVhZjQ4MDRk"
    "ODE0N2YxNDZkYzU1Y2EwMjZjY2U0MTE5ZmMwNjg4Yzc4YmEyYzNiN2MzZCJdXX1dLFsiZW5kIix7ImRhdGFjbGFzcyI6Imp4cGxh"
    "bmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkNvb3JkaW5hdGVFcG9jaCIsImZpZWxkcyI6W1sidGltZV9zY2FsZSIsIlREQiJd"
    "LFsicmVwcmVzZW50YXRpb24iLCJTUElDRV9UREJfSjIwMDBfT0ZGU0VUX1NFQ09ORFNfVFdPX1BBUlQiXSxbIndob2xlIiw4NjQw"
    "MF0sWyJmcmFjdGlvbiIseyJmbG9hdF9oZXgiOiIweDAuMHArMCJ9XSxbImNvb3JkaW5hdGVfdW5pdF9pZCIsInVuaXQuc2Vjb25k"
    "LmRlZmluZWQtZXhhY3QudjEiXSxbIm9yaWdpbl9pZCIsIlNQSUNFX0oyMDAwX1REQl9PUklHSU4iXSxbInJlYWxpemF0aW9uX2lk"
    "IiwiZml4dHVyZS5zcGstZXQtcmVhbGl6YXRpb24iXSxbImFydGlmYWN0X2lkcyIsW11dLFsiY29udGVudF9zaGEyNTYiLCJlOGQw"
    "NzRkNzM3MGU0YmI5ODljNDA3NjA4NDA1YjY0MzUyNGFjYzQyNGVhYWIzZDllMDJkZjZkZjJhOTVhMGY2Il1dfV0sWyJlbmRwb2lu"
    "dF9wb2xpY3kiLCJDTE9TRURfQ0xPU0VEIl0sWyJjb250ZW50X3NoYTI1NiIsImU3OWExNzU1MGIwZGUwYWRkMjUzYmFlZGUwZTJk"
    "MjEzMGJkODJhM2UwOTU5NGE5MjgxMWJhYjQzNTE0ODc3OGUiXV19XV0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QuZml4dHVy"
    "ZS5zcGsiXV0sWyJjb250ZW50X3NoYTI1NiIsIjZjNzQyM2Q0NmJjNjFiYjlhYzhhZjY5NjQ3OGYyN2FmZWUwMjU4NzUyNjIyM2Ix"
    "MWUyOTI4MmU3OTQxNzM2NDIiXV19XSxbImFiZXJyYXRpb25fY29ycmVjdGlvbiIsIk5PTkUiXSxbIm91dHB1dF91bml0X3N5c3Rl"
    "bSIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Vbml0U3lzdGVtRGVmaW5pdGlvbiIsImZp"
    "ZWxkcyI6W1sidW5pdF9zeXN0ZW1faWQiLCJ1bml0cy5zaS10ZGIiXSxbImxlbmd0aCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHgu"
    "c29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9pZCIsInVuaXQubWV0cmUiXSxb"
    "ImRpbWVuc2lvbiIsIkxFTkdUSCJdLFsic2lfdW5pdF9pZCIsInNpLm1ldHJlIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0"
    "b3IiLDFdLFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRp"
    "ZmFjdC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsImQ1Y2YzNGI2Nzk5NjdiOTYyMWY5NjY2NWJiZGVjNDlk"
    "MmI5ODViNWIwMmUwOTliMzIxZjUwNmE4NjZkNmU2ZmMiXV19XSxbInRpbWUiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFy"
    "X3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0LnNlY29uZC5kZWZpbmVk"
    "LWV4YWN0LnYxIl0sWyJkaW1lbnNpb24iLCJUSU1FIl0sWyJzaV91bml0X2lkIiwic2kuc2Vjb25kIl0sWyJudW1lcmF0b3IiLDFd"
    "LFsiZGVub21pbmF0b3IiLDFdLFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0"
    "X2lkcyIsWyJhcnRpZmFjdC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsIjU4M2E1YjA4YTU4MGVjNDJiNTY0"
    "NDIwZmE0YWZmYzNlZDQzYzFkZjljNDA0NzFiZDU3OWViODRkMDA3YTM5NzQiXV19XSxbIm1hc3MiLHsiZGF0YWNsYXNzIjoianhw"
    "bGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0Lmtp"
    "bG9ncmFtIl0sWyJkaW1lbnNpb24iLCJNQVNTIl0sWyJzaV91bml0X2lkIiwic2kua2lsb2dyYW0iXSxbIm51bWVyYXRvciIsMV0s"
    "WyJkZW5vbWluYXRvciIsMV0sWyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3Rf"
    "aWRzIixbImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiNDk2NWRiYjI4N2ZhNWY3MWJjZWUw"
    "ODg5MWU0MzcwZTJmM2M1OGNlMzBiYmVhNWI0MmZkYjYxNTQ5MmU0YWIwNyJdXX1dLFsiZ3Jhdml0YXRpb25hbF9wYXJhbWV0ZXIi"
    "LHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltb"
    "InVuaXRfaWQiLCJ1bml0Lm1ldHJlMy1wZXItc2Vjb25kMiJdLFsiZGltZW5zaW9uIiwiR1JBVklUQVRJT05BTF9QQVJBTUVURVIi"
    "XSxbInNpX3VuaXRfaWQiLCJzaS5tZXRyZTMtcGVyLXNlY29uZDIiXSxbIm51bWVyYXRvciIsMV0sWyJkZW5vbWluYXRvciIsMV0s"
    "WyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LnBy"
    "b3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiMDgxZWZmYzMxZWU1NTNjMTJjNTZiODVjNmY3OTJkZDgwMTQ0NTli"
    "MGU5MThmNTE4MjZhYjM4MTUwYWExY2ZhNyJdXX1dLFsiY29vcmRpbmF0ZV90aW1lX3NjYWxlIiwiVERCIl0sWyJjb250ZW50X3No"
    "YTI1NiIsIjIzMjg1NDY5ZGU4MTliYjIwYThhNGU4NTIxMjJhOWE5ZDNmM2JhMGFhM2I2YjI0Zjc4ZGZiMTc5NDJhMzZmNDAiXV19"
    "XSxbInN0YXRlX2tpbmQiLCJHRU9NRVRSSUMiXSxbInRhcmdldF9jaGFpbl9hdmFpbGFiaWxpdHlfc3RhdHVzIiwiUkVRVUlSRVNf"
    "UlVOVElNRV9QUk9WSURFUl9UQVJHRVRfQVZBSUxBQklMSVRZX1ZBTElEQVRJT04iXSxbImNvbnRlbnRfc2hhMjU2IiwiZTA0Nzc3"
    "NjViZTJkY2I3NmY4MjBiODk0NjQ4M2I0MDgxZjZmNTc1YWRmZWMzN2NmMDhlNjYxODVhYzA1NjlmNiJdXX1dLFsiZWZmZWN0aXZl"
    "X3F1ZXJ5Iix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkVwaGVtZXJpc1F1ZXJ5U3BlYyIs"
    "ImZpZWxkcyI6W1sicXVlcnlfaWQiLCJxdWVyeS5maXh0dXJlLmNzcGljZS1wcm9qZWN0aW9uLmVmZmVjdGl2ZSJdLFsicHJvdmlk"
    "ZXIiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRXBoZW1lcmlzUHJvdmlkZXJTcGVjIiwi"
    "ZmllbGRzIjpbWyJzcGVjX2lkIiwicHJvdmlkZXIuZml4dHVyZS5zcGVjLnYxIl0sWyJpZGVudGl0eSIseyJkYXRhY2xhc3MiOiJq"
    "eHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Qcm92aWRlcklkZW50aXR5IiwiZmllbGRzIjpbWyJwcm92aWRlcl9pZCIs"
    "InByb3ZpZGVyLmZpeHR1cmUiXSxbImltcGxlbWVudGF0aW9uX2lkIiwicHJvdmlkZXIuZml4dHVyZS5pbXBsZW1lbnRhdGlvbi52"
    "MSJdLFsicHJvdmlkZXJfa2luZCIsIkxPQ0FMX09GRkxJTkVfU1BLIl0sWyJ2ZXJzaW9uIiwiMSJdLFsicnVudGltZV9pZCIsImNw"
    "eXRob24udGVzdCJdLFsibW9kdWxlX25hbWUiLCJmaXh0dXJlX3Byb3ZpZGVyIl0sWyJkaXN0cmlidXRpb25fbmFtZSIsImZpeHR1"
    "cmUtcHJvdmlkZXIiXSxbImRpc3RyaWJ1dGlvbl92ZXJzaW9uIiwiMSJdLFsib3JkZXJlZF9pbXBsZW1lbnRhdGlvbl9hcnRpZmFj"
    "dF9pZHMiLFsiYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCIxNGQzMWM3ODVmN2Q4ZTgxYjg5"
    "MjVmMTEwYTk3NTFmZTJiM2M5OGMyYTUzOTQ3MzBiOTkzNGQ1M2RjZDRmOTZmIl1dfV0sWyJhcnRpZmFjdHMiLFt7ImRhdGFjbGFz"
    "cyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkFydGlmYWN0QmluZGluZyIsImZpZWxkcyI6W1siYXJ0aWZhY3Rf"
    "aWQiLCJhcnRpZmFjdC5maXh0dXJlLnNwayJdLFsiYXJ0aWZhY3Rfcm9sZSIsIlNQSyJdLFsicHJvdmlkZXJfaWQiLCJwcm92aWRl"
    "ci5maXh0dXJlIl0sWyJ2ZXJzaW9uIiwiMSJdLFsibG9naWNhbF9sb2NhdG9yIiwicmV0YWluZWQvYXJ0aWZhY3QuZml4dHVyZS5z"
    "cGsuYmluIl0sWyJsb2NhdG9yX2tpbmQiLCJMT0NBTF9SRUdVTEFSX0ZJTEUiXSxbImJ5dGVfbGVuZ3RoIiwxMjhdLFsiYXJ0aWZh"
    "Y3Rfc2hhMjU2IiwiMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MiJdLFsibWVkaWFfdHlwZSIsImFwcGxpY2F0aW9uL29jdGV0LXN0cmVhbSJdLFsiY292ZXJhZ2Vfc3RhdHVzIiwiQ09BUlNFX0FS"
    "VElGQUNUX1RJTUVfRU5WRUxPUEUiXSxbImNvdmVyYWdlIixbeyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNv"
    "bnRyYWN0cy5Db3ZlcmFnZUludGVydmFsIiwiZmllbGRzIjpbWyJzdGFydCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJf"
    "c3lzdGVtLmNvbnRyYWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltbInRpbWVfc2NhbGUiLCJUREIiXSxbInJlcHJlc2Vu"
    "dGF0aW9uIiwiU1BJQ0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19QQVJUIl0sWyJ3aG9sZSIsLTg2NDAwXSxbImZyYWN0"
    "aW9uIix7ImZsb2F0X2hleCI6IjB4MC4wcCswIn1dLFsiY29vcmRpbmF0ZV91bml0X2lkIiwidW5pdC5zZWNvbmQuZGVmaW5lZC1l"
    "eGFjdC52MSJdLFsib3JpZ2luX2lkIiwiU1BJQ0VfSjIwMDBfVERCX09SSUdJTiJdLFsicmVhbGl6YXRpb25faWQiLCJmaXh0dXJl"
    "LnNway1ldC1yZWFsaXphdGlvbiJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsIjlhNWRiNTQwNDgzNWU1"
    "YWY0ODA0ZDgxNDdmMTQ2ZGM1NWNhMDI2Y2NlNDExOWZjMDY4OGM3OGJhMmMzYjdjM2QiXV19XSxbImVuZCIseyJkYXRhY2xhc3Mi"
    "OiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltbInRpbWVfc2NhbGUi"
    "LCJUREIiXSxbInJlcHJlc2VudGF0aW9uIiwiU1BJQ0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19QQVJUIl0sWyJ3aG9s"
    "ZSIsODY0MDBdLFsiZnJhY3Rpb24iLHsiZmxvYXRfaGV4IjoiMHgwLjBwKzAifV0sWyJjb29yZGluYXRlX3VuaXRfaWQiLCJ1bml0"
    "LnNlY29uZC5kZWZpbmVkLWV4YWN0LnYxIl0sWyJvcmlnaW5faWQiLCJTUElDRV9KMjAwMF9UREJfT1JJR0lOIl0sWyJyZWFsaXph"
    "dGlvbl9pZCIsImZpeHR1cmUuc3BrLWV0LXJlYWxpemF0aW9uIl0sWyJhcnRpZmFjdF9pZHMiLFtdXSxbImNvbnRlbnRfc2hhMjU2"
    "IiwiZThkMDc0ZDczNzBlNGJiOTg5YzQwNzYwODQwNWI2NDM1MjRhY2M0MjRlYWFiM2Q5ZTAyZGY2ZGYyYTk1YTBmNiJdXX1dLFsi"
    "ZW5kcG9pbnRfcG9saWN5IiwiQ0xPU0VEX0NMT1NFRCJdLFsiY29udGVudF9zaGEyNTYiLCJlNzlhMTc1NTBiMGRlMGFkZDI1M2Jh"
    "ZWRlMGUyZDIxMzBiZDgyYTNlMDk1OTRhOTI4MTFiYWI0MzUxNDg3NzhlIl1dfV1dLFsibGljZW5zZV9ldmlkZW5jZV9zdGF0dXMi"
    "LCJFWFRFUk5BTF9OT1RfUkVESVNUUklCVVRFRF9MSUNFTlNFX05PVF9SRVRBSU5FRF9OT05BVVRIT1JJWklORyJdLFsibGljZW5z"
    "ZV9zcGR4IiwiQlNELTMtQ2xhdXNlIl0sWyJsaWNlbnNlX2FydGlmYWN0X2lkIixudWxsXSxbInJlZGlzdHJpYnV0aW9uX3N0YXR1"
    "cyIsIkVYVEVSTkFMX1JFRkVSRU5DRV9OT1RfUkVESVNUUklCVVRFRCJdLFsibG9hZF9vcmRlcl9zdGF0dXMiLCJPUkRFUkVEX0xP"
    "QURfTUVNQkVSIl0sWyJsb2FkX29yZGVyIiwwXSxbImV4dHJhcG9sYXRpb25fcG9saWN5IiwiRk9SQklEIl0sWyJjb250ZW50X2lu"
    "dGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05URU5UX0lOVEVHUklUWV9PTkxZIl0sWyJjb250ZW50X3NoYTI1NiIs"
    "IjJkODA3NTcwZDM5MmI4OGY0ZDVjZDc1ZGNmODAwM2VmYjA3YmUyN2U5NTJkMWU1OGIzNmJjYWRkNTQ5ZWMwOWMiXV19LHsiZGF0"
    "YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuQXJ0aWZhY3RCaW5kaW5nIiwiZmllbGRzIjpbWyJhcnRp"
    "ZmFjdF9pZCIsImFydGlmYWN0LnByb3ZpZGVyLmxpY2Vuc2UiXSxbImFydGlmYWN0X3JvbGUiLCJMSUNFTlNFIl0sWyJwcm92aWRl"
    "cl9pZCIsInByb3ZpZGVyLmZpeHR1cmUiXSxbInZlcnNpb24iLCIxIl0sWyJsb2dpY2FsX2xvY2F0b3IiLCJyZXRhaW5lZC9MSUNF"
    "TlNFLnR4dCJdLFsibG9jYXRvcl9raW5kIiwiTE9DQUxfUkVHVUxBUl9GSUxFIl0sWyJieXRlX2xlbmd0aCIsNjRdLFsiYXJ0aWZh"
    "Y3Rfc2hhMjU2IiwiMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMz"
    "MyJdLFsibWVkaWFfdHlwZSIsInRleHQvcGxhaW4iXSxbImNvdmVyYWdlX3N0YXR1cyIsIk5PVF9BUFBMSUNBQkxFIl0sWyJjb3Zl"
    "cmFnZSIsbnVsbF0sWyJsaWNlbnNlX2V2aWRlbmNlX3N0YXR1cyIsIlJFVEFJTkVEX0xJQ0VOU0VfVEVYVF9TRUxGX0VWSURFTkNF"
    "X05PTkFVVEhPUklaSU5HIl0sWyJsaWNlbnNlX3NwZHgiLCJCU0QtMy1DbGF1c2UiXSxbImxpY2Vuc2VfYXJ0aWZhY3RfaWQiLG51"
    "bGxdLFsicmVkaXN0cmlidXRpb25fc3RhdHVzIiwiQlVORExFRF9XSVRIX1JFVEFJTkVEX0xJQ0VOU0VfRVZJREVOQ0UiXSxbImxv"
    "YWRfb3JkZXJfc3RhdHVzIiwiTk9UX0xPQURBQkxFIl0sWyJsb2FkX29yZGVyIixudWxsXSxbImV4dHJhcG9sYXRpb25fcG9saWN5"
    "IiwiRk9SQklEIl0sWyJjb250ZW50X2ludGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05URU5UX0lOVEVHUklUWV9P"
    "TkxZIl0sWyJjb250ZW50X3NoYTI1NiIsIjQ5YzQ1Yzk1ZmVhZjI4MjhiMzE0ZDY0NTc4YWZiY2FmM2ZjNDg3NmVlZmNlYzIwNzE1"
    "OTMzMjg1NmE0OGVmYTAiXV19LHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuQXJ0aWZhY3RC"
    "aW5kaW5nIiwiZmllbGRzIjpbWyJhcnRpZmFjdF9pZCIsImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJdLFsiYXJ0aWZhY3Rfcm9s"
    "ZSIsIlNPRlRXQVJFX1NPVVJDRSJdLFsicHJvdmlkZXJfaWQiLCJwcm92aWRlci5maXh0dXJlIl0sWyJ2ZXJzaW9uIiwiMSJdLFsi"
    "bG9naWNhbF9sb2NhdG9yIiwicmV0YWluZWQvYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlLmJpbiJdLFsibG9jYXRvcl9raW5kIiwi"
    "TE9DQUxfUkVHVUxBUl9GSUxFIl0sWyJieXRlX2xlbmd0aCIsMTI4XSxbImFydGlmYWN0X3NoYTI1NiIsIjExMTExMTExMTExMTEx"
    "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTEiXSxbIm1lZGlhX3R5cGUiLCJhcHBsaWNh"
    "dGlvbi9vY3RldC1zdHJlYW0iXSxbImNvdmVyYWdlX3N0YXR1cyIsIk5PVF9BUFBMSUNBQkxFIl0sWyJjb3ZlcmFnZSIsbnVsbF0s"
    "WyJsaWNlbnNlX2V2aWRlbmNlX3N0YXR1cyIsIlJFVEFJTkVEX0hBU0hfQk9VTkRfTElDRU5TRV9BUlRJRkFDVCJdLFsibGljZW5z"
    "ZV9zcGR4IiwiQlNELTMtQ2xhdXNlIl0sWyJsaWNlbnNlX2FydGlmYWN0X2lkIiwiYXJ0aWZhY3QucHJvdmlkZXIubGljZW5zZSJd"
    "LFsicmVkaXN0cmlidXRpb25fc3RhdHVzIiwiQlVORExFRF9XSVRIX1JFVEFJTkVEX0xJQ0VOU0VfRVZJREVOQ0UiXSxbImxvYWRf"
    "b3JkZXJfc3RhdHVzIiwiTk9UX0xPQURBQkxFIl0sWyJsb2FkX29yZGVyIixudWxsXSxbImV4dHJhcG9sYXRpb25fcG9saWN5Iiwi"
    "Rk9SQklEIl0sWyJjb250ZW50X2ludGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05URU5UX0lOVEVHUklUWV9PTkxZ"
    "Il0sWyJjb250ZW50X3NoYTI1NiIsIjM5YTk5NTFlYWZmMWFkMGM4OTc5YmM0YzAzNzQxYjYwN2NkNDI1OTNjYWNhOTQ5ZWYzYWM3"
    "NWU0Mjg0MWUzMzciXV19XV0sWyJvcmRlcmVkX2xvYWRfYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LmZpeHR1cmUuc3BrIl1dLFsi"
    "Y2FwYWJpbGl0aWVzIixbIkNPQVJTRV9BUlRJRkFDVF9USU1FX0VOVkVMT1BFIiwiR0VPTUVUUklDX0NBUlRFU0lBTl9TVEFURSIs"
    "IlREQl9DT09SRElOQVRFX0VQT0NIIl1dLFsibmF0aXZlX3RpbWVfc2NhbGUiLCJUREIiXSxbIm5hdGl2ZV9mcmFtZSIseyJkYXRh"
    "Y2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5GcmFtZVJlYWxpemF0aW9uIiwiZmllbGRzIjpbWyJmcmFt"
    "ZV9pZCIsImZyYW1lLmZpeHR1cmUuZGUtaWNyZiJdLFsiZnJhbWVfa2luZCIsIlREQl9DT01QQVRJQkxFX0JBUllDRU5UUklDX0lO"
    "RVJUSUFMIl0sWyJvcmlnaW5fa2luZCIsIlNPTEFSX1NZU1RFTV9CQVJZQ0VOVEVSIl0sWyJvcmlnaW5fbmFpZl9pZCIsMF0sWyJv"
    "cmlnaW5fcmVhbGl6YXRpb25faWQiLCJmaXh0dXJlLmRlLXJlbGVhc2Uuc3NiIl0sWyJheGVzX3JlYWxpemF0aW9uX2lkIiwiZml4"
    "dHVyZS5kZS1yZWxlYXNlLmljcmYiXSxbIm9yaWVudGF0aW9uX21vZGVsX2lkIiwiZml4dHVyZS5kZS1yZWxlYXNlLm9yaWVudGF0"
    "aW9uIl0sWyJvcmllbnRhdGlvbl90aW1lX2RlcGVuZGVuY2UiLCJTVEFUSUMiXSxbImNvb3JkaW5hdGVfdGltZV9zY2FsZSIsIlRE"
    "QiJdLFsiY292ZXJhZ2Vfc3RhdHVzIiwiUkVUQUlORUQiXSxbImNvdmVyYWdlIixbeyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29s"
    "YXJfc3lzdGVtLmNvbnRyYWN0cy5Db3ZlcmFnZUludGVydmFsIiwiZmllbGRzIjpbWyJzdGFydCIseyJkYXRhY2xhc3MiOiJqeHBs"
    "YW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltbInRpbWVfc2NhbGUiLCJUREIi"
    "XSxbInJlcHJlc2VudGF0aW9uIiwiU1BJQ0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19QQVJUIl0sWyJ3aG9sZSIsLTg2"
    "NDAwXSxbImZyYWN0aW9uIix7ImZsb2F0X2hleCI6IjB4MC4wcCswIn1dLFsiY29vcmRpbmF0ZV91bml0X2lkIiwidW5pdC5zZWNv"
    "bmQuZGVmaW5lZC1leGFjdC52MSJdLFsib3JpZ2luX2lkIiwiU1BJQ0VfSjIwMDBfVERCX09SSUdJTiJdLFsicmVhbGl6YXRpb25f"
    "aWQiLCJmaXh0dXJlLnNway1ldC1yZWFsaXphdGlvbiJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsIjlh"
    "NWRiNTQwNDgzNWU1YWY0ODA0ZDgxNDdmMTQ2ZGM1NWNhMDI2Y2NlNDExOWZjMDY4OGM3OGJhMmMzYjdjM2QiXV19XSxbImVuZCIs"
    "eyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltb"
    "InRpbWVfc2NhbGUiLCJUREIiXSxbInJlcHJlc2VudGF0aW9uIiwiU1BJQ0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19Q"
    "QVJUIl0sWyJ3aG9sZSIsODY0MDBdLFsiZnJhY3Rpb24iLHsiZmxvYXRfaGV4IjoiMHgwLjBwKzAifV0sWyJjb29yZGluYXRlX3Vu"
    "aXRfaWQiLCJ1bml0LnNlY29uZC5kZWZpbmVkLWV4YWN0LnYxIl0sWyJvcmlnaW5faWQiLCJTUElDRV9KMjAwMF9UREJfT1JJR0lO"
    "Il0sWyJyZWFsaXphdGlvbl9pZCIsImZpeHR1cmUuc3BrLWV0LXJlYWxpemF0aW9uIl0sWyJhcnRpZmFjdF9pZHMiLFtdXSxbImNv"
    "bnRlbnRfc2hhMjU2IiwiZThkMDc0ZDczNzBlNGJiOTg5YzQwNzYwODQwNWI2NDM1MjRhY2M0MjRlYWFiM2Q5ZTAyZGY2ZGYyYTk1"
    "YTBmNiJdXX1dLFsiZW5kcG9pbnRfcG9saWN5IiwiQ0xPU0VEX0NMT1NFRCJdLFsiY29udGVudF9zaGEyNTYiLCJlNzlhMTc1NTBi"
    "MGRlMGFkZDI1M2JhZWRlMGUyZDIxMzBiZDgyYTNlMDk1OTRhOTI4MTFiYWI0MzUxNDg3NzhlIl1dfV1dLFsiYXJ0aWZhY3RfaWRz"
    "IixbImFydGlmYWN0LmZpeHR1cmUuc3BrIl1dLFsiY29udGVudF9zaGEyNTYiLCI2Yzc0MjNkNDZiYzYxYmI5YWM4YWY2OTY0Nzhm"
    "MjdhZmVlMDI1ODc1MjYyMjNiMTFlMjkyODJlNzk0MTczNjQyIl1dfV0sWyJuYXRpdmVfb3V0cHV0X3VuaXRfc3lzdGVtIix7ImRh"
    "dGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLlVuaXRTeXN0ZW1EZWZpbml0aW9uIiwiZmllbGRzIjpb"
    "WyJ1bml0X3N5c3RlbV9pZCIsInVuaXRzLnNpLXRkYiJdLFsibGVuZ3RoIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9z"
    "eXN0ZW0uY29udHJhY3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1bml0X2lkIiwidW5pdC5tZXRyZSJdLFsiZGltZW5z"
    "aW9uIiwiTEVOR1RIIl0sWyJzaV91bml0X2lkIiwic2kubWV0cmUiXSxbIm51bWVyYXRvciIsMV0sWyJkZW5vbWluYXRvciIsMV0s"
    "WyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LnBy"
    "b3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiZDVjZjM0YjY3OTk2N2I5NjIxZjk2NjY1YmJkZWM0OWQyYjk4NWI1"
    "YjAyZTA5OWIzMjFmNTA2YTg2NmQ2ZTZmYyJdXX1dLFsidGltZSIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVt"
    "LmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9pZCIsInVuaXQuc2Vjb25kLmRlZmluZWQtZXhhY3Qu"
    "djEiXSxbImRpbWVuc2lvbiIsIlRJTUUiXSxbInNpX3VuaXRfaWQiLCJzaS5zZWNvbmQiXSxbIm51bWVyYXRvciIsMV0sWyJkZW5v"
    "bWluYXRvciIsMV0sWyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3RfaWRzIixb"
    "ImFydGlmYWN0LnByb3ZpZGVyLnNvdXJjZSJdXSxbImNvbnRlbnRfc2hhMjU2IiwiNTgzYTViMDhhNTgwZWM0MmI1NjQ0MjBmYTRh"
    "ZmZjM2VkNDNjMWRmOWM0MDQ3MWJkNTc5ZWI4NGQwMDdhMzk3NCJdXX1dLFsibWFzcyIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHgu"
    "c29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9pZCIsInVuaXQua2lsb2dyYW0i"
    "XSxbImRpbWVuc2lvbiIsIk1BU1MiXSxbInNpX3VuaXRfaWQiLCJzaS5raWxvZ3JhbSJdLFsibnVtZXJhdG9yIiwxXSxbImRlbm9t"
    "aW5hdG9yIiwxXSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJERUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMiLFsi"
    "YXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCI0OTY1ZGJiMjg3ZmE1ZjcxYmNlZTA4ODkxZTQz"
    "NzBlMmYzYzU4Y2UzMGJiZWE1YjQyZmRiNjE1NDkyZTRhYjA3Il1dfV0sWyJncmF2aXRhdGlvbmFsX3BhcmFtZXRlciIseyJkYXRh"
    "Y2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6W1sidW5pdF9p"
    "ZCIsInVuaXQubWV0cmUzLXBlci1zZWNvbmQyIl0sWyJkaW1lbnNpb24iLCJHUkFWSVRBVElPTkFMX1BBUkFNRVRFUiJdLFsic2lf"
    "dW5pdF9pZCIsInNpLm1ldHJlMy1wZXItc2Vjb25kMiJdLFsibnVtZXJhdG9yIiwxXSxbImRlbm9taW5hdG9yIiwxXSxbImRlZmlu"
    "aXRpb25fY2xhc3NpZmljYXRpb24iLCJERUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3QucHJvdmlkZXIu"
    "c291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCIwODFlZmZjMzFlZTU1M2MxMmM1NmI4NWM2Zjc5MmRkODAxNDQ1OWIwZTkxOGY1"
    "MTgyNmFiMzgxNTBhYTFjZmE3Il1dfV0sWyJjb29yZGluYXRlX3RpbWVfc2NhbGUiLCJUREIiXSxbImNvbnRlbnRfc2hhMjU2Iiwi"
    "MjMyODU0NjlkZTgxOWJiMjBhOGE0ZTg1MjEyMmE5YTlkM2YzYmEwYWEzYjZiMjRmNzhkZmIxNzk0MmEzNmY0MCJdXX1dLFsibmV0"
    "d29ya19hY2Nlc3MiLGZhbHNlXSxbImZhbGxiYWNrX2FsbG93ZWQiLGZhbHNlXSxbImV4dHJhcG9sYXRpb25fcG9saWN5IiwiRk9S"
    "QklEIl0sWyJldmlkZW5jZV9jbGFzcyIsIk1PREVMX09VVFBVVCJdLFsicmVnaXN0cnlfYXV0aG9yaXplZCIsZmFsc2VdLFsicXVh"
    "bGlmaWNhdGlvbl9hdXRob3JpemVkIixmYWxzZV0sWyJjb250ZW50X3NoYTI1NiIsIjNmNzU0YjdmYzk3NTM1YzJiMTVjOTFiNzQ5"
    "ZjdhYzcwNjU1MTI0OTExMmI5ODU3ODhjYjAwMmFkMDE3ZmQ5NGIiXV19XSxbImVwb2NoIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0"
    "eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkNvb3JkaW5hdGVFcG9jaCIsImZpZWxkcyI6W1sidGltZV9zY2FsZSIsIlREQiJdLFsi"
    "cmVwcmVzZW50YXRpb24iLCJTUElDRV9UREJfSjIwMDBfT0ZGU0VUX1NFQ09ORFNfVFdPX1BBUlQiXSxbIndob2xlIiwxXSxbImZy"
    "YWN0aW9uIix7ImZsb2F0X2hleCI6IjB4MC4wcCswIn1dLFsiY29vcmRpbmF0ZV91bml0X2lkIiwidW5pdC5zZWNvbmQuZGVmaW5l"
    "ZC1leGFjdC52MSJdLFsib3JpZ2luX2lkIiwiU1BJQ0VfSjIwMDBfVERCX09SSUdJTiJdLFsicmVhbGl6YXRpb25faWQiLCJmaXh0"
    "dXJlLnNway1ldC1yZWFsaXphdGlvbiJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsImYzZGRiZTY1ZWIw"
    "YzM3YzIwMDUxNjYxMGU4MTlkZTUxMWYyZjYxZmMyYjE3MjIwOGU3OWJhMGVjN2RiMmVlMWEiXV19XSxbInRhcmdldF9uYWlmX2lk"
    "cyIsWzEwLDM5OV1dLFsib2JzZXJ2ZXJfbmFpZl9pZCIsMF0sWyJmcmFtZSIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJf"
    "c3lzdGVtLmNvbnRyYWN0cy5GcmFtZVJlYWxpemF0aW9uIiwiZmllbGRzIjpbWyJmcmFtZV9pZCIsImZyYW1lLmZpeHR1cmUuZGUt"
    "aWNyZiJdLFsiZnJhbWVfa2luZCIsIlREQl9DT01QQVRJQkxFX0JBUllDRU5UUklDX0lORVJUSUFMIl0sWyJvcmlnaW5fa2luZCIs"
    "IlNPTEFSX1NZU1RFTV9CQVJZQ0VOVEVSIl0sWyJvcmlnaW5fbmFpZl9pZCIsMF0sWyJvcmlnaW5fcmVhbGl6YXRpb25faWQiLCJm"
    "aXh0dXJlLmRlLXJlbGVhc2Uuc3NiIl0sWyJheGVzX3JlYWxpemF0aW9uX2lkIiwiZml4dHVyZS5kZS1yZWxlYXNlLmljcmYiXSxb"
    "Im9yaWVudGF0aW9uX21vZGVsX2lkIiwiZml4dHVyZS5kZS1yZWxlYXNlLm9yaWVudGF0aW9uIl0sWyJvcmllbnRhdGlvbl90aW1l"
    "X2RlcGVuZGVuY2UiLCJTVEFUSUMiXSxbImNvb3JkaW5hdGVfdGltZV9zY2FsZSIsIlREQiJdLFsiY292ZXJhZ2Vfc3RhdHVzIiwi"
    "UkVUQUlORUQiXSxbImNvdmVyYWdlIixbeyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Db3Zl"
    "cmFnZUludGVydmFsIiwiZmllbGRzIjpbWyJzdGFydCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRy"
    "YWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltbInRpbWVfc2NhbGUiLCJUREIiXSxbInJlcHJlc2VudGF0aW9uIiwiU1BJ"
    "Q0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19QQVJUIl0sWyJ3aG9sZSIsLTg2NDAwXSxbImZyYWN0aW9uIix7ImZsb2F0"
    "X2hleCI6IjB4MC4wcCswIn1dLFsiY29vcmRpbmF0ZV91bml0X2lkIiwidW5pdC5zZWNvbmQuZGVmaW5lZC1leGFjdC52MSJdLFsi"
    "b3JpZ2luX2lkIiwiU1BJQ0VfSjIwMDBfVERCX09SSUdJTiJdLFsicmVhbGl6YXRpb25faWQiLCJmaXh0dXJlLnNway1ldC1yZWFs"
    "aXphdGlvbiJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsIjlhNWRiNTQwNDgzNWU1YWY0ODA0ZDgxNDdm"
    "MTQ2ZGM1NWNhMDI2Y2NlNDExOWZjMDY4OGM3OGJhMmMzYjdjM2QiXV19XSxbImVuZCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHgu"
    "c29sYXJfc3lzdGVtLmNvbnRyYWN0cy5Db29yZGluYXRlRXBvY2giLCJmaWVsZHMiOltbInRpbWVfc2NhbGUiLCJUREIiXSxbInJl"
    "cHJlc2VudGF0aW9uIiwiU1BJQ0VfVERCX0oyMDAwX09GRlNFVF9TRUNPTkRTX1RXT19QQVJUIl0sWyJ3aG9sZSIsODY0MDBdLFsi"
    "ZnJhY3Rpb24iLHsiZmxvYXRfaGV4IjoiMHgwLjBwKzAifV0sWyJjb29yZGluYXRlX3VuaXRfaWQiLCJ1bml0LnNlY29uZC5kZWZp"
    "bmVkLWV4YWN0LnYxIl0sWyJvcmlnaW5faWQiLCJTUElDRV9KMjAwMF9UREJfT1JJR0lOIl0sWyJyZWFsaXphdGlvbl9pZCIsImZp"
    "eHR1cmUuc3BrLWV0LXJlYWxpemF0aW9uIl0sWyJhcnRpZmFjdF9pZHMiLFtdXSxbImNvbnRlbnRfc2hhMjU2IiwiZThkMDc0ZDcz"
    "NzBlNGJiOTg5YzQwNzYwODQwNWI2NDM1MjRhY2M0MjRlYWFiM2Q5ZTAyZGY2ZGYyYTk1YTBmNiJdXX1dLFsiZW5kcG9pbnRfcG9s"
    "aWN5IiwiQ0xPU0VEX0NMT1NFRCJdLFsiY29udGVudF9zaGEyNTYiLCJlNzlhMTc1NTBiMGRlMGFkZDI1M2JhZWRlMGUyZDIxMzBi"
    "ZDgyYTNlMDk1OTRhOTI4MTFiYWI0MzUxNDg3NzhlIl1dfV1dLFsiYXJ0aWZhY3RfaWRzIixbImFydGlmYWN0LmZpeHR1cmUuc3Br"
    "Il1dLFsiY29udGVudF9zaGEyNTYiLCI2Yzc0MjNkNDZiYzYxYmI5YWM4YWY2OTY0NzhmMjdhZmVlMDI1ODc1MjYyMjNiMTFlMjky"
    "ODJlNzk0MTczNjQyIl1dfV0sWyJhYmVycmF0aW9uX2NvcnJlY3Rpb24iLCJOT05FIl0sWyJvdXRwdXRfdW5pdF9zeXN0ZW0iLHsi"
    "ZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuVW5pdFN5c3RlbURlZmluaXRpb24iLCJmaWVsZHMi"
    "OltbInVuaXRfc3lzdGVtX2lkIiwidW5pdHMuc2ktdGRiIl0sWyJsZW5ndGgiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFy"
    "X3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJmaWVsZHMiOltbInVuaXRfaWQiLCJ1bml0Lm1ldHJlIl0sWyJkaW1l"
    "bnNpb24iLCJMRU5HVEgiXSxbInNpX3VuaXRfaWQiLCJzaS5tZXRyZSJdLFsibnVtZXJhdG9yIiwxXSxbImRlbm9taW5hdG9yIiwx"
    "XSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJERUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMiLFsiYXJ0aWZhY3Qu"
    "cHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCJkNWNmMzRiNjc5OTY3Yjk2MjFmOTY2NjViYmRlYzQ5ZDJiOTg1"
    "YjViMDJlMDk5YjMyMWY1MDZhODY2ZDZlNmZjIl1dfV0sWyJ0aW1lIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0"
    "ZW0uY29udHJhY3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1bml0X2lkIiwidW5pdC5zZWNvbmQuZGVmaW5lZC1leGFj"
    "dC52MSJdLFsiZGltZW5zaW9uIiwiVElNRSJdLFsic2lfdW5pdF9pZCIsInNpLnNlY29uZCJdLFsibnVtZXJhdG9yIiwxXSxbImRl"
    "bm9taW5hdG9yIiwxXSxbImRlZmluaXRpb25fY2xhc3NpZmljYXRpb24iLCJERUZJTkVEX0VYQUNUIl0sWyJhcnRpZmFjdF9pZHMi"
    "LFsiYXJ0aWZhY3QucHJvdmlkZXIuc291cmNlIl1dLFsiY29udGVudF9zaGEyNTYiLCI1ODNhNWIwOGE1ODBlYzQyYjU2NDQyMGZh"
    "NGFmZmMzZWQ0M2MxZGY5YzQwNDcxYmQ1NzllYjg0ZDAwN2EzOTc0Il1dfV0sWyJtYXNzIix7ImRhdGFjbGFzcyI6Imp4cGxhbmV0"
    "eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1bml0X2lkIiwidW5pdC5raWxvZ3Jh"
    "bSJdLFsiZGltZW5zaW9uIiwiTUFTUyJdLFsic2lfdW5pdF9pZCIsInNpLmtpbG9ncmFtIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVu"
    "b21pbmF0b3IiLDFdLFsiZGVmaW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIs"
    "WyJhcnRpZmFjdC5wcm92aWRlci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsIjQ5NjVkYmIyODdmYTVmNzFiY2VlMDg4OTFl"
    "NDM3MGUyZjNjNThjZTMwYmJlYTViNDJmZGI2MTU0OTJlNGFiMDciXV19XSxbImdyYXZpdGF0aW9uYWxfcGFyYW1ldGVyIix7ImRh"
    "dGFjbGFzcyI6Imp4cGxhbmV0eC5zb2xhcl9zeXN0ZW0uY29udHJhY3RzLkV4YWN0VW5pdFNjYWxlIiwiZmllbGRzIjpbWyJ1bml0"
    "X2lkIiwidW5pdC5tZXRyZTMtcGVyLXNlY29uZDIiXSxbImRpbWVuc2lvbiIsIkdSQVZJVEFUSU9OQUxfUEFSQU1FVEVSIl0sWyJz"
    "aV91bml0X2lkIiwic2kubWV0cmUzLXBlci1zZWNvbmQyIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3IiLDFdLFsiZGVm"
    "aW5pdGlvbl9jbGFzc2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsWyJhcnRpZmFjdC5wcm92aWRl"
    "ci5zb3VyY2UiXV0sWyJjb250ZW50X3NoYTI1NiIsIjA4MWVmZmMzMWVlNTUzYzEyYzU2Yjg1YzZmNzkyZGQ4MDE0NDU5YjBlOTE4"
    "ZjUxODI2YWIzODE1MGFhMWNmYTciXV19XSxbImNvb3JkaW5hdGVfdGltZV9zY2FsZSIsIlREQiJdLFsiY29udGVudF9zaGEyNTYi"
    "LCIyMzI4NTQ2OWRlODE5YmIyMGE4YTRlODUyMTIyYTlhOWQzZjNiYTBhYTNiNmIyNGY3OGRmYjE3OTQyYTM2ZjQwIl1dfV0sWyJz"
    "dGF0ZV9raW5kIiwiR0VPTUVUUklDIl0sWyJ0YXJnZXRfY2hhaW5fYXZhaWxhYmlsaXR5X3N0YXR1cyIsIlJFUVVJUkVTX1JVTlRJ"
    "TUVfUFJPVklERVJfVEFSR0VUX0FWQUlMQUJJTElUWV9WQUxJREFUSU9OIl0sWyJjb250ZW50X3NoYTI1NiIsIjZmOTU2ZmY4ZGEy"
    "OTNhYzlmNjQ2NjhmYjcwNTYyNzc0MDU5ZWRjMGFhYTRlMzc3YjRjYzBjMDQ1M2RhOGY5ZjciXV19XSxbImJpbmFyeTY0X3Byb2pl"
    "Y3Rpb24iLFsianhwbGFuZXR4LnNvbGFyX3N5c3RlbS51bml0cy5CaW5hcnk2NFVuaXRDb252ZXJzaW9uUmVjZWlwdCIsW1sic291"
    "cmNlX3VuaXQiLHsiZGF0YWNsYXNzIjoianhwbGFuZXR4LnNvbGFyX3N5c3RlbS5jb250cmFjdHMuRXhhY3RVbml0U2NhbGUiLCJm"
    "aWVsZHMiOltbInVuaXRfaWQiLCJ1bml0LnNlY29uZC5kZWZpbmVkLWV4YWN0LnYxIl0sWyJkaW1lbnNpb24iLCJUSU1FIl0sWyJz"
    "aV91bml0X2lkIiwic2kuc2Vjb25kIl0sWyJudW1lcmF0b3IiLDFdLFsiZGVub21pbmF0b3IiLDFdLFsiZGVmaW5pdGlvbl9jbGFz"
    "c2lmaWNhdGlvbiIsIkRFRklORURfRVhBQ1QiXSxbImFydGlmYWN0X2lkcyIsW11dLFsiY29udGVudF9zaGEyNTYiLCJkNGUyM2Uz"
    "ODMwNWIyYjljMjgzNDg2YTMzZWVmMzlmYzM1Y2RlYmVhODcwMGU4Y2IzNThhNDIzNTRhYjUwNzVmIl1dfV0sWyJ0YXJnZXRfdW5p"
    "dCIseyJkYXRhY2xhc3MiOiJqeHBsYW5ldHguc29sYXJfc3lzdGVtLmNvbnRyYWN0cy5FeGFjdFVuaXRTY2FsZSIsImZpZWxkcyI6"
    "W1sidW5pdF9pZCIsInVuaXQuc2Vjb25kLmRlZmluZWQtZXhhY3QudjEiXSxbImRpbWVuc2lvbiIsIlRJTUUiXSxbInNpX3VuaXRf"
    "aWQiLCJzaS5zZWNvbmQiXSxbIm51bWVyYXRvciIsMV0sWyJkZW5vbWluYXRvciIsMV0sWyJkZWZpbml0aW9uX2NsYXNzaWZpY2F0"
    "aW9uIiwiREVGSU5FRF9FWEFDVCJdLFsiYXJ0aWZhY3RfaWRzIixbXV0sWyJjb250ZW50X3NoYTI1NiIsImQ0ZTIzZTM4MzA1YjJi"
    "OWMyODM0ODZhMzNlZWYzOWZjMzVjZGViZWE4NzAwZThjYjM1OGE0MjM1NGFiNTA3NWYiXV19XSxbImlucHV0X251bWVyYXRvciIs"
    "OTAwNzE5OTI1NDc0MDk5M10sWyJpbnB1dF9kZW5vbWluYXRvciIsOTAwNzE5OTI1NDc0MDk5Ml0sWyJleGFjdF9udW1lcmF0b3Ii"
    "LDkwMDcxOTkyNTQ3NDA5OTNdLFsiZXhhY3RfZGVub21pbmF0b3IiLDkwMDcxOTkyNTQ3NDA5OTJdLFsicm91bmRlZF92YWx1ZSIs"
    "eyJmbG9hdF9oZXgiOiIweDEuMDAwMDAwMDAwMDAwMHArMCJ9XSxbImRlZmluaXRpb25fc2NvcGUiLCJERUZJTkVEX09OTFkiXSxb"
    "InJvdW5kaW5nX21vZGUiLCJORUFSRVNUX1RJRVNfVE9fRVZFTiJdLFsicm91bmRpbmdfc3RhdHVzIiwiUk9VTkRFRCJdLFsicm91"
    "bmRpbmdfZGlyZWN0aW9uIiwiQkVMT1dfRVhBQ1QiXSxbInJlc3VsdF9jbGFzcyIsIk5PUk1BTCJdXSxbImNvbnRlbnRfc2hhMjU2"
    "IiwiYjQ4NGYxMjJlMWI2Y2FkNzZjNzJkNzY3MmVjODVlNTdmMGY5MWI3YWZhMjVmODQ1ZmE4MmQ1Nzc1MDEzNDU4MSJdXV0sWyJw"
    "cm9qZWN0aW9uX3N0YWdlIiwiUkVRVUVTVEVEX1NQSUNFX1REQl9FVF9TRUNPTkRTX1RPX0VGRkVDVElWRV9DU1BJQ0VfQklOQVJZ"
    "NjRfRVQiXSxbImV4YWN0X2Vycm9yX251bWVyYXRvciIsLTFdLFsiZXhhY3RfZXJyb3JfZGVub21pbmF0b3IiLDkwMDcxOTkyNTQ3"
    "NDA5OTJdLFsiZXhhY3RfZXJyb3JfdW5pdF9pZCIsInVuaXQuc2Vjb25kLmRlZmluZWQtZXhhY3QudjEiXSxbInRpbWVfc2NhbGVf"
    "b3BlcmF0aW9uX3N0YXR1cyIsIk5PX1RJTUVfU0NBTEVfT1JfT1JJR0lOX0NPTlZFUlNJT05fU0FNRV9UREJfRVRfQ09PUkRJTkFU"
    "RSJdLFsicHJvdmlkZXJfZXhlY3V0aW9uX3N0YXR1cyIsIk5PVF9FWEVDVVRFRF9BUklUSE1FVElDX0lOVEVSRkFDRV9QUk9KRUNU"
    "SU9OX09OTFkiXSxbImV2aWRlbmNlX3ByZXNlcnZhdGlvbl9zdGF0dXMiLCJSRVFVRVNURURfRVBPQ0hfQVJUSUZBQ1RfSURTX1BS"
    "RVNFUlZFRF9VTkNIQU5HRURfTk9fTkVXX0VWSURFTkNFIl0sWyJhcnRpZmFjdF9jdXN0b2R5X3N0YXR1cyIsIk5PX0FSVElGQUNU"
    "X0FUX1VTRV9DVVNUT0RZX0VWSURFTkNFIl0sWyJjb250ZW50X2ludGVncml0eV9jbGFzcyIsIlVOQVVUSEVOVElDQVRFRF9DT05U"
    "RU5UX0lOVEVHUklUWV9PTkxZIl1dXQ=="
)


if __name__ == "__main__":
    unittest.main()
