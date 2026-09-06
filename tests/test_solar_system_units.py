"""Defined-unit and binary64 boundary tests for Solar-System Milestone 2."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import hashlib
import json
import math
import os
import subprocess
import sys
import unittest

from jxplanetx.solar_system import units as units_module
from jxplanetx.solar_system.contracts import (
    ExactUnitScale,
    SolarSystemContractError,
    validate_integrity,
)
from jxplanetx.solar_system.serialization import canonical_json
from jxplanetx.solar_system.units import (
    ASTRONOMICAL_UNIT,
    Binary64UnitConversionReceipt,
    DAY,
    KILOGRAM,
    KILOMETRE,
    METRE,
    SECOND,
    convert_fraction,
    convert_fraction_to_binary64,
    unit_ratio,
)


EXPECTED_EXPORTS = [
    "ASTRONOMICAL_UNIT",
    "Binary64UnitConversionReceipt",
    "DAY",
    "KILOGRAM",
    "KILOMETRE",
    "METRE",
    "SECOND",
    "convert_fraction",
    "convert_fraction_to_binary64",
    "unit_ratio",
]

EXPECTED_UNIT_HASHES = {
    "ASTRONOMICAL_UNIT": "0f90471c4b19957b37ea6cd745f088c9be01347b9000a47610f94d4ef5d50b41",
    "DAY": "6da5bbdb86e73d6de3499c0a5233d2e70a0b9d73d126951af8e62950f0cd20d9",
    "KILOGRAM": "1fd421de1d700c192bd822aee89333b4afb706dd1194d941a8c806b650235df5",
    "KILOMETRE": "d32770a3727eb693503804e02dde3979bc0e3f1438cf83951e899c1470387b8c",
    "METRE": "ce00c46dade9467c44a96521b87e8650f132e45a81129f17955bbd732c57f649",
    "SECOND": "d4e23e38305b2b9c283486a33eef39fc35cdebea8700e8cb358a42354ab5075f",
}

RECEIPT_FIELDS = (
    "source_unit",
    "target_unit",
    "input_numerator",
    "input_denominator",
    "exact_numerator",
    "exact_denominator",
    "rounded_value",
    "definition_scope",
    "rounding_mode",
    "rounding_status",
    "rounding_direction",
    "result_class",
    "content_sha256",
)

# Independent literal COMPLETE preimage for Fraction(1, 10), metre -> metre.
# It does not call the production receipt-payload or digest builder.
RECEIPT_LITERAL_PREIMAGE = (
    b"jxplanetx.solar-system.units.binary64-unit-conversion-receipt.v1\x00"
    b"Binary64UnitConversionReceipt.v1\x00"
    b'["jxplanetx.solar_system.units.Binary64UnitConversionReceipt",'
    b'[["source_unit",{"dataclass":"jxplanetx.solar_system.contracts.ExactUnitScale",'
    b'"fields":[["unit_id","unit.metre.defined-exact.v1"],["dimension","LENGTH"],'
    b'['
    b'"si_unit_id","si.metre"],["numerator",1],["denominator",1],'
    b'["definition_classification","DEFINED_EXACT"],["artifact_ids",[]],'
    b'["content_sha256","ce00c46dade9467c44a96521b87e8650f132e45a81129f17955bbd732c57f649"]]}],'
    b'["target_unit",{"dataclass":"jxplanetx.solar_system.contracts.ExactUnitScale",'
    b'"fields":[["unit_id","unit.metre.defined-exact.v1"],["dimension","LENGTH"],'
    b'['
    b'"si_unit_id","si.metre"],["numerator",1],["denominator",1],'
    b'["definition_classification","DEFINED_EXACT"],["artifact_ids",[]],'
    b'["content_sha256","ce00c46dade9467c44a96521b87e8650f132e45a81129f17955bbd732c57f649"]]}],'
    b'["input_numerator",1],["input_denominator",10],["exact_numerator",1],'
    b'["exact_denominator",10],["rounded_value",{"float_hex":"0x1.999999999999ap-4"}],'
    b'["definition_scope","DEFINED_ONLY"],["rounding_mode","NEAREST_TIES_TO_EVEN"],'
    b'["rounding_status","ROUNDED"],["rounding_direction","ABOVE_EXACT"],'
    b'["result_class","NORMAL"]]]'
)
RECEIPT_LITERAL_LENGTH = 1_217
RECEIPT_LITERAL_SHA256 = (
    "e5ea1ada6fd1de324b4266cbb432fa7d9b3079dfa2bc20c8e5994fbeeb50dd75"
)


def nominal_metre() -> ExactUnitScale:
    return ExactUnitScale(
        unit_id="unit.nominal-metre.fixture",
        dimension="LENGTH",
        si_unit_id="si.metre",
        numerator=1,
        denominator=1,
        definition_classification="NOMINAL_EXACT",
        artifact_ids=("artifact.nominal.fixture",),
    )


class UnitConstantTests(unittest.TestCase):
    def test_exact_exports_and_root_nonpublication(self) -> None:
        self.assertEqual(units_module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(units_module.__all__, sorted(units_module.__all__))
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_exact_defined_unit_roster_and_hashes(self) -> None:
        expected = {
            "METRE": (
                "unit.metre.defined-exact.v1",
                "LENGTH",
                "si.metre",
                1,
            ),
            "SECOND": (
                "unit.second.defined-exact.v1",
                "TIME",
                "si.second",
                1,
            ),
            "KILOGRAM": (
                "unit.kilogram.defined-exact.v1",
                "MASS",
                "si.kilogram",
                1,
            ),
            "KILOMETRE": (
                "unit.kilometre.1000-si-metres.defined-exact.v1",
                "LENGTH",
                "si.metre",
                1_000,
            ),
            "DAY": (
                "unit.day.86400-si-seconds.defined-exact.v1",
                "TIME",
                "si.second",
                86_400,
            ),
            "ASTRONOMICAL_UNIT": (
                "unit.astronomical-unit.149597870700-si-metres.defined-exact.v1",
                "LENGTH",
                "si.metre",
                149_597_870_700,
            ),
        }
        for name, (unit_id, dimension, si_unit, numerator) in expected.items():
            value = getattr(units_module, name)
            self.assertIs(type(value), ExactUnitScale)
            validate_integrity(value)
            self.assertEqual(
                (
                    value.unit_id,
                    value.dimension,
                    value.si_unit_id,
                    value.numerator,
                    value.denominator,
                    value.definition_classification,
                    value.artifact_ids,
                ),
                (unit_id, dimension, si_unit, numerator, 1, "DEFINED_EXACT", ()),
            )
            self.assertEqual(value.content_sha256, EXPECTED_UNIT_HASHES[name])

    def test_exact_ratios_and_conversions(self) -> None:
        self.assertEqual(unit_ratio(KILOMETRE, METRE), Fraction(1_000))
        self.assertEqual(unit_ratio(DAY, SECOND), Fraction(86_400))
        self.assertEqual(
            unit_ratio(ASTRONOMICAL_UNIT, METRE),
            Fraction(149_597_870_700),
        )
        self.assertEqual(
            unit_ratio(ASTRONOMICAL_UNIT, KILOMETRE),
            Fraction(1_495_978_707, 10),
        )
        self.assertEqual(
            convert_fraction(Fraction(3, 2), KILOMETRE, METRE),
            Fraction(1_500),
        )

    def test_ratio_group_properties_on_fixed_grid(self) -> None:
        length_units = (METRE, KILOMETRE, ASTRONOMICAL_UNIT)
        values = (Fraction(-7, 3), Fraction(0), Fraction(11, 5))
        for source in length_units:
            self.assertEqual(unit_ratio(source, source), Fraction(1))
            for target in length_units:
                self.assertEqual(
                    unit_ratio(source, target) * unit_ratio(target, source),
                    Fraction(1),
                )
                for third in length_units:
                    self.assertEqual(
                        unit_ratio(source, target) * unit_ratio(target, third),
                        unit_ratio(source, third),
                    )
                for value in values:
                    converted = convert_fraction(value, source, target)
                    self.assertEqual(
                        convert_fraction(converted, target, source),
                        value,
                    )

    def test_dimension_nominal_type_and_stale_seal_rejections(self) -> None:
        with self.assertRaises(SolarSystemContractError):
            unit_ratio(METRE, SECOND)
        with self.assertRaises(SolarSystemContractError):
            unit_ratio(nominal_metre(), METRE)
        with self.assertRaises(SolarSystemContractError):
            convert_fraction(1, METRE, METRE)  # type: ignore[arg-type]
        with self.assertRaises(SolarSystemContractError):
            convert_fraction(1.0, METRE, METRE)  # type: ignore[arg-type]

        class FractionSubclass(Fraction):
            pass

        with self.assertRaises(SolarSystemContractError):
            convert_fraction(FractionSubclass(1, 2), METRE, METRE)

        stale = dataclasses.replace(METRE)
        object.__setattr__(stale, "numerator", 2)
        with self.assertRaises(SolarSystemContractError):
            unit_ratio(stale, METRE)

    def test_same_identifier_cannot_name_different_seals(self) -> None:
        conflicting = ExactUnitScale(
            unit_id=METRE.unit_id,
            dimension="LENGTH",
            si_unit_id="si.metre",
            numerator=2,
            denominator=1,
            definition_classification="DEFINED_EXACT",
            artifact_ids=(),
        )
        with self.assertRaises(SolarSystemContractError):
            unit_ratio(METRE, conflicting)

    def test_input_and_reduced_output_caps(self) -> None:
        with self.assertRaises(SolarSystemContractError):
            convert_fraction(Fraction(1 << 4096, 1), METRE, METRE)
        with self.assertRaises(SolarSystemContractError):
            convert_fraction(Fraction(1, 1 << 4096), METRE, METRE)

        huge_source = ExactUnitScale(
            unit_id="unit.huge-source.fixture",
            dimension="LENGTH",
            si_unit_id="si.metre",
            numerator=1 << 4095,
            denominator=1,
            definition_classification="DEFINED_EXACT",
            artifact_ids=(),
        )
        huge_target = ExactUnitScale(
            unit_id="unit.huge-target.fixture",
            dimension="LENGTH",
            si_unit_id="si.metre",
            numerator=1,
            denominator=(1 << 4095) - 1,
            definition_classification="DEFINED_EXACT",
            artifact_ids=(),
        )
        with self.assertRaises(SolarSystemContractError):
            unit_ratio(huge_source, huge_target)


class Binary64ConversionTests(unittest.TestCase):
    def test_receipt_field_roster_and_literal_production_kat(self) -> None:
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(Binary64UnitConversionReceipt)),
            RECEIPT_FIELDS,
        )
        receipt = convert_fraction_to_binary64(Fraction(1, 10), METRE, METRE)
        self.assertEqual(len(RECEIPT_LITERAL_PREIMAGE), RECEIPT_LITERAL_LENGTH)
        self.assertEqual(
            hashlib.sha256(RECEIPT_LITERAL_PREIMAGE).hexdigest(),
            RECEIPT_LITERAL_SHA256,
        )
        self.assertEqual(receipt.content_sha256, RECEIPT_LITERAL_SHA256)
        production_preimage = (
            units_module._RECEIPT_DOMAIN.encode("utf-8")
            + b"\x00"
            + units_module._RECEIPT_SCHEMA.encode("utf-8")
            + b"\x00"
            + canonical_json(units_module._receipt_payload(receipt))
        )
        self.assertEqual(production_preimage, RECEIPT_LITERAL_PREIMAGE)
        decoded = json.loads(
            RECEIPT_LITERAL_PREIMAGE.split(b"\x00", 2)[2].decode("utf-8")
        )
        self.assertEqual(decoded[0], units_module._RECEIPT_QUALIFIED_NAME)
        self.assertEqual(tuple(item[0] for item in decoded[1]), RECEIPT_FIELDS[:-1])

    def test_one_tenth_receipt_is_fully_classified(self) -> None:
        receipt = convert_fraction_to_binary64(Fraction(1, 10), METRE, METRE)
        receipt.validate_integrity()
        self.assertEqual(receipt.rounded_value.hex(), "0x1.999999999999ap-4")
        self.assertEqual(
            (
                receipt.definition_scope,
                receipt.rounding_mode,
                receipt.rounding_status,
                receipt.rounding_direction,
                receipt.result_class,
            ),
            (
                "DEFINED_ONLY",
                "NEAREST_TIES_TO_EVEN",
                "ROUNDED",
                "ABOVE_EXACT",
                "NORMAL",
            ),
        )

    def test_zero_is_canonical_positive_and_float_inputs_are_out_of_domain(self) -> None:
        receipt = convert_fraction_to_binary64(Fraction(0), METRE, METRE)
        self.assertEqual(receipt.rounded_value.hex(), "0x0.0p+0")
        self.assertEqual(receipt.result_class, "ZERO")
        self.assertEqual(receipt.rounding_status, "EXACT")
        with self.assertRaises(SolarSystemContractError):
            convert_fraction_to_binary64(-0.0, METRE, METRE)  # type: ignore[arg-type]
        with self.assertRaises(SolarSystemContractError):
            convert_fraction_to_binary64(0.0, METRE, METRE)  # type: ignore[arg-type]
        with self.assertRaises(SolarSystemContractError):
            convert_fraction_to_binary64(False, METRE, METRE)  # type: ignore[arg-type]

        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(receipt, rounded_value=-0.0, content_sha256="")

    def test_ties_to_even_both_parities_and_signs(self) -> None:
        cases = (
            (Fraction((1 << 53) + 1, 1 << 53), "0x1.0000000000000p+0", "BELOW_EXACT"),
            (Fraction((1 << 53) + 3, 1 << 53), "0x1.0000000000002p+0", "ABOVE_EXACT"),
            (Fraction(-((1 << 53) + 1), 1 << 53), "-0x1.0000000000000p+0", "ABOVE_EXACT"),
            (Fraction(-((1 << 53) + 3), 1 << 53), "-0x1.0000000000002p+0", "BELOW_EXACT"),
        )
        for value, expected_hex, expected_direction in cases:
            receipt = convert_fraction_to_binary64(value, METRE, METRE)
            self.assertEqual(receipt.rounded_value.hex(), expected_hex)
            self.assertEqual(receipt.rounding_direction, expected_direction)
            self.assertEqual(receipt.rounding_status, "ROUNDED")

    def test_significand_carry_to_next_exponent(self) -> None:
        value = Fraction((1 << 54) - 1, 1 << 53)
        receipt = convert_fraction_to_binary64(value, METRE, METRE)
        self.assertEqual(receipt.rounded_value.hex(), "0x1.0000000000000p+1")
        self.assertEqual(receipt.rounding_direction, "ABOVE_EXACT")
        negative = convert_fraction_to_binary64(-value, METRE, METRE)
        self.assertEqual(negative.rounded_value.hex(), "-0x1.0000000000000p+1")
        self.assertEqual(negative.rounding_direction, "BELOW_EXACT")

    def test_normal_subnormal_boundary(self) -> None:
        midpoint = Fraction((1 << 53) - 1, 1 << 1075)
        at_midpoint = convert_fraction_to_binary64(midpoint, METRE, METRE)
        self.assertEqual(at_midpoint.rounded_value.hex(), "0x1.0000000000000p-1022")
        self.assertEqual(at_midpoint.result_class, "NORMAL")
        self.assertEqual(at_midpoint.rounding_direction, "ABOVE_EXACT")

        below_midpoint = Fraction((1 << 54) - 3, 1 << 1076)
        below = convert_fraction_to_binary64(below_midpoint, METRE, METRE)
        self.assertEqual(below.rounded_value.hex(), "0x0.fffffffffffffp-1022")
        self.assertEqual(below.result_class, "SUBNORMAL")
        self.assertEqual(below.rounding_direction, "BELOW_EXACT")

        negative_midpoint = convert_fraction_to_binary64(-midpoint, METRE, METRE)
        self.assertEqual(
            negative_midpoint.rounded_value.hex(),
            "-0x1.0000000000000p-1022",
        )
        self.assertEqual(negative_midpoint.result_class, "NORMAL")
        self.assertEqual(negative_midpoint.rounding_direction, "BELOW_EXACT")

    def test_subnormal_and_nonzero_to_zero_boundaries(self) -> None:
        exact_min = convert_fraction_to_binary64(Fraction(1, 1 << 1074), METRE, METRE)
        self.assertEqual(exact_min.rounded_value.hex(), "0x0.0000000000001p-1022")
        self.assertEqual(exact_min.rounding_status, "EXACT")
        self.assertEqual(exact_min.result_class, "SUBNORMAL")

        just_above_half = convert_fraction_to_binary64(
            Fraction(3, 1 << 1076),
            METRE,
            METRE,
        )
        self.assertEqual(
            just_above_half.rounded_value.hex(),
            "0x0.0000000000001p-1022",
        )
        negative_exact_min = convert_fraction_to_binary64(
            Fraction(-1, 1 << 1074),
            METRE,
            METRE,
        )
        self.assertEqual(
            negative_exact_min.rounded_value.hex(),
            "-0x0.0000000000001p-1022",
        )
        self.assertEqual(negative_exact_min.rounding_direction, "EXACT")
        negative_above_half = convert_fraction_to_binary64(
            Fraction(-3, 1 << 1076),
            METRE,
            METRE,
        )
        self.assertEqual(
            negative_above_half.rounded_value.hex(),
            "-0x0.0000000000001p-1022",
        )
        self.assertEqual(negative_above_half.rounding_direction, "BELOW_EXACT")
        for sign in (1, -1):
            with self.assertRaises(SolarSystemContractError):
                convert_fraction_to_binary64(
                    Fraction(sign, 1 << 1075),
                    METRE,
                    METRE,
                )

    def test_max_finite_and_overflow_threshold_h(self) -> None:
        maximum = Fraction(((1 << 53) - 1) << 971, 1)
        exact_maximum = convert_fraction_to_binary64(maximum, METRE, METRE)
        self.assertEqual(exact_maximum.rounded_value.hex(), "0x1.fffffffffffffp+1023")
        self.assertEqual(exact_maximum.rounding_status, "EXACT")

        threshold = Fraction((1 << 1024) - (1 << 970), 1)
        below = convert_fraction_to_binary64(threshold - 1, METRE, METRE)
        self.assertEqual(below.rounded_value.hex(), "0x1.fffffffffffffp+1023")
        self.assertEqual(below.rounding_direction, "BELOW_EXACT")
        negative_below = convert_fraction_to_binary64(-(threshold - 1), METRE, METRE)
        self.assertEqual(negative_below.rounded_value.hex(), "-0x1.fffffffffffffp+1023")
        self.assertEqual(negative_below.rounding_direction, "ABOVE_EXACT")
        for value in (threshold, -threshold, Fraction(1 << 1024), Fraction(-(1 << 1024))):
            with self.assertRaises(SolarSystemContractError):
                convert_fraction_to_binary64(value, METRE, METRE)

    def test_receipt_exact_schema_and_coherent_mutations_reject(self) -> None:
        receipt = convert_fraction_to_binary64(Fraction(1, 10), METRE, METRE)
        mutations = {
            "source_unit": KILOMETRE,
            "target_unit": KILOMETRE,
            "input_numerator": 2,
            "input_denominator": 11,
            "exact_numerator": 2,
            "exact_denominator": 11,
            "rounded_value": math.nextafter(receipt.rounded_value, math.inf),
            "definition_scope": "INCLUDES_NOMINAL",
            "rounding_mode": "TOWARD_ZERO",
            "rounding_status": "EXACT",
            "rounding_direction": "BELOW_EXACT",
            "result_class": "SUBNORMAL",
            "content_sha256": "0" * 64,
        }
        for field, replacement in mutations.items():
            kwargs = {field: replacement}
            if field != "content_sha256":
                kwargs["content_sha256"] = receipt.content_sha256
            with self.subTest(field=field), self.assertRaises(SolarSystemContractError):
                dataclasses.replace(receipt, **kwargs)

        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(
                receipt,
                input_denominator=20,
                exact_denominator=20,
                content_sha256="",
            )
        with self.assertRaises(SolarSystemContractError):
            dataclasses.replace(receipt, input_numerator=True, content_sha256="")

    def test_token_and_remainder_caps_precede_membership_or_shift_work(self) -> None:
        class MembershipBomb:
            def __contains__(self, value: object) -> bool:
                raise RuntimeError("MEMBERSHIP_WAS_REACHED")

        with self.assertRaises(SolarSystemContractError) as caught:
            units_module._require_token(
                "x" * 4_097,
                MembershipBomb(),  # type: ignore[arg-type]
                "fixture_token",
            )
        self.assertNotIn("MEMBERSHIP_WAS_REACHED", str(caught.exception))

        with self.assertRaises(SolarSystemContractError):
            units_module._round_quotient_ties_to_even(
                0,
                1 << 16_383,
                (1 << 16_384) + 1,
            )

    def test_receipt_detects_postconstruction_nested_and_own_mutation(self) -> None:
        receipt = convert_fraction_to_binary64(Fraction(1, 10), METRE, METRE)
        object.__setattr__(receipt, "rounding_status", "EXACT")
        with self.assertRaises(SolarSystemContractError):
            receipt.validate_integrity()

        source = dataclasses.replace(METRE)
        nested = convert_fraction_to_binary64(Fraction(1), source, METRE)
        object.__setattr__(source, "numerator", 2)
        with self.assertRaises(SolarSystemContractError):
            nested.validate_integrity()

    def test_receipt_subclass_rejects(self) -> None:
        class ReceiptSubclass(Binary64UnitConversionReceipt):
            pass

        with self.assertRaises(SolarSystemContractError):
            ReceiptSubclass(
                source_unit=METRE,
                target_unit=METRE,
                input_numerator=1,
                input_denominator=1,
                exact_numerator=1,
                exact_denominator=1,
                rounded_value=1.0,
                definition_scope="DEFINED_ONLY",
                rounding_mode="NEAREST_TIES_TO_EVEN",
                rounding_status="EXACT",
                rounding_direction="EXACT",
                result_class="NORMAL",
            )

    def test_literal_domain_schema_order_and_signed_zero_mutations_change_sha(self) -> None:
        variants = (
            RECEIPT_LITERAL_PREIMAGE.replace(
                b"jxplanetx.solar-system.units.binary64-unit-conversion-receipt.v1",
                b"jxplanetx.solar-system.units.binary64-unit-conversion-receipt.v2",
                1,
            ),
            RECEIPT_LITERAL_PREIMAGE.replace(
                b"Binary64UnitConversionReceipt.v1",
                b"Binary64UnitConversionReceipt.v2",
                1,
            ),
            RECEIPT_LITERAL_PREIMAGE.replace(
                b'[["source_unit",',
                b'[["target_unit",',
                1,
            ),
            RECEIPT_LITERAL_PREIMAGE.replace(
                b'"0x1.999999999999ap-4"',
                b'"-0x0.0p+0"',
                1,
            ),
        )
        for variant in variants:
            self.assertNotEqual(hashlib.sha256(variant).hexdigest(), RECEIPT_LITERAL_SHA256)


class UnitImportBoundaryTests(unittest.TestCase):
    def test_clean_import_is_dependency_free(self) -> None:
        code = """
import json, sys
import jxplanetx.solar_system.units as units
forbidden = [
    name for name in (
        'numpy', 'astropy', 'erfa', 'spiceypy', 'jplephem', 'skyfield', 'rebound'
    ) if name in sys.modules
]
print(json.dumps({'exports': units.__all__, 'forbidden': forbidden}))
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
        self.assertEqual(payload["forbidden"], [])


if __name__ == "__main__":
    unittest.main()
