"""Same-context two-part epoch tests for Solar-System Milestone 2."""

from __future__ import annotations

import dataclasses
from fractions import Fraction
import inspect
import json
import math
import os
import subprocess
import sys
import unittest

from jxplanetx.solar_system import time as time_module
from jxplanetx.solar_system.contracts import (
    CoordinateEpoch,
    SolarSystemContractError,
    validate_integrity,
)
from jxplanetx.solar_system.time import (
    compare_coordinate_epochs,
    coordinate_epoch_difference,
    shift_coordinate_epoch,
)


DAY_UNIT = "unit.day.86400-si-seconds.defined-exact.v1"
SECOND_UNIT = "unit.second.defined-exact.v1"
EXPECTED_EXPORTS = [
    "compare_coordinate_epochs",
    "coordinate_epoch_difference",
    "shift_coordinate_epoch",
]


def jd_epoch(
    whole: int,
    fraction: float,
    *,
    time_scale: str = "TDB",
    realization_id: str = "realization.fixture.jd",
    artifact_ids: tuple[str, ...] = (),
) -> CoordinateEpoch:
    return CoordinateEpoch(
        time_scale=time_scale,
        representation="JD_TWO_PART",
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=DAY_UNIT,
        origin_id="JULIAN_DATE",
        realization_id=realization_id,
        artifact_ids=artifact_ids,
    )


def et_epoch(
    whole: int,
    fraction: float,
    *,
    realization_id: str = "realization.fixture.spice-et",
    artifact_ids: tuple[str, ...] = (),
) -> CoordinateEpoch:
    return CoordinateEpoch(
        time_scale="TDB",
        representation="SPICE_TDB_J2000_OFFSET_SECONDS_TWO_PART",
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=SECOND_UNIT,
        origin_id="SPICE_J2000_TDB_ORIGIN",
        realization_id=realization_id,
        artifact_ids=artifact_ids,
    )


def synthetic_epoch(
    whole: int,
    fraction: float,
    *,
    unit_id: str = "unit.synthetic.tick.fixture",
    origin_id: str = "origin.synthetic.fixture",
    realization_id: str = "realization.synthetic.fixture",
    artifact_ids: tuple[str, ...] = (),
) -> CoordinateEpoch:
    return CoordinateEpoch(
        time_scale="SYNTHETIC",
        representation="SYNTHETIC_OFFSET",
        whole=whole,
        fraction=fraction,
        coordinate_unit_id=unit_id,
        origin_id=origin_id,
        realization_id=realization_id,
        artifact_ids=artifact_ids,
    )


def jd_from_fraction(value: Fraction, **kwargs: object) -> CoordinateEpoch:
    whole = ((value.numerator << 1) + value.denominator) // (
        value.denominator << 1
    )
    remainder = value - whole
    as_float = float(remainder)
    if Fraction.from_float(as_float) != remainder:
        raise AssertionError("test grid requires an exactly representable remainder")
    return jd_epoch(whole, as_float, **kwargs)  # type: ignore[arg-type]


class CoordinateEpochArithmeticTests(unittest.TestCase):
    def test_exact_exports_and_shift_parameter_name(self) -> None:
        self.assertEqual(time_module.__all__, EXPECTED_EXPORTS)
        self.assertEqual(time_module.__all__, sorted(time_module.__all__))
        self.assertEqual(
            tuple(inspect.signature(shift_coordinate_epoch).parameters),
            ("epoch", "offset_in_coordinate_units"),
        )
        from jxplanetx import solar_system

        self.assertEqual(solar_system.__all__, [])

    def test_compare_and_difference_convention(self) -> None:
        left = jd_epoch(2_451_545, 0.25)
        right = jd_epoch(2_451_544, -0.25)
        self.assertEqual(compare_coordinate_epochs(left, right), 1)
        self.assertEqual(compare_coordinate_epochs(right, left), -1)
        self.assertEqual(compare_coordinate_epochs(left, left), 0)
        self.assertEqual(coordinate_epoch_difference(left, right), Fraction(3, 2))
        self.assertEqual(coordinate_epoch_difference(right, left), Fraction(-3, 2))
        self.assertEqual(coordinate_epoch_difference(left, left), Fraction(0, 1))

    def test_fraction_bits_are_not_decimal_reinterpreted(self) -> None:
        left = jd_epoch(0, 0.1)
        right = jd_epoch(0, 0.0)
        self.assertEqual(
            coordinate_epoch_difference(left, right),
            Fraction.from_float(0.1),
        )
        self.assertNotEqual(coordinate_epoch_difference(left, right), Fraction(1, 10))

    def test_order_properties_on_exact_dyadic_grid(self) -> None:
        coordinates = (
            Fraction(-9, 4),
            Fraction(-1, 2),
            Fraction(0),
            Fraction(7, 8),
            Fraction(13, 4),
        )
        epochs = tuple(jd_from_fraction(value) for value in coordinates)
        for left_index, left in enumerate(epochs):
            for right_index, right in enumerate(epochs):
                expected = -1 if left_index < right_index else 1 if left_index > right_index else 0
                self.assertEqual(compare_coordinate_epochs(left, right), expected)
                self.assertEqual(
                    coordinate_epoch_difference(left, right),
                    coordinates[left_index] - coordinates[right_index],
                )
                self.assertEqual(
                    coordinate_epoch_difference(left, right),
                    -coordinate_epoch_difference(right, left),
                )
        for first in epochs:
            for second in epochs:
                for third in epochs:
                    if (
                        compare_coordinate_epochs(first, second) <= 0
                        and compare_coordinate_epochs(second, third) <= 0
                    ):
                        self.assertLessEqual(compare_coordinate_epochs(first, third), 0)

    def test_evidence_is_not_coordinate_identity_or_merged(self) -> None:
        left = jd_epoch(10, 0.25, artifact_ids=("artifact.time.a",))
        right = jd_epoch(10, 0.25, artifact_ids=("artifact.time.b",))
        self.assertEqual(compare_coordinate_epochs(left, right), 0)
        self.assertEqual(coordinate_epoch_difference(left, right), Fraction(0))
        shifted = shift_coordinate_epoch(left, Fraction(1, 4))
        self.assertEqual(shifted.artifact_ids, left.artifact_ids)
        self.assertNotEqual(shifted.artifact_ids, right.artifact_ids)

    def test_every_coordinate_context_component_is_binding(self) -> None:
        base = jd_epoch(10, 0.0)
        mismatches = (
            jd_epoch(10, 0.0, time_scale="TCB"),
            jd_epoch(10, 0.0, realization_id="realization.fixture.other"),
            et_epoch(10, 0.0),
            synthetic_epoch(10, 0.0),
            synthetic_epoch(10, 0.0, unit_id="unit.synthetic.other"),
            synthetic_epoch(10, 0.0, origin_id="origin.synthetic.other"),
            synthetic_epoch(10, 0.0, realization_id="realization.synthetic.other"),
        )
        for mismatch in mismatches:
            with self.subTest(context=mismatch.content_sha256):
                with self.assertRaises(SolarSystemContractError):
                    compare_coordinate_epochs(base, mismatch)
                with self.assertRaises(SolarSystemContractError):
                    coordinate_epoch_difference(base, mismatch)

        synthetic_base = synthetic_epoch(10, 0.0)
        synthetic_mismatches = (
            synthetic_epoch(10, 0.0, unit_id="unit.synthetic.other"),
            synthetic_epoch(10, 0.0, origin_id="origin.synthetic.other"),
            synthetic_epoch(10, 0.0, realization_id="realization.synthetic.other"),
        )
        for mismatch in synthetic_mismatches:
            with self.subTest(synthetic_context=mismatch.content_sha256):
                with self.assertRaises(SolarSystemContractError):
                    compare_coordinate_epochs(synthetic_base, mismatch)
                with self.assertRaises(SolarSystemContractError):
                    coordinate_epoch_difference(synthetic_base, mismatch)

    def test_tdb_et_jd_and_synthetic_walls_are_not_bridges(self) -> None:
        jd = jd_epoch(2_451_545, 0.0)
        et = et_epoch(0, 0.0)
        synthetic = synthetic_epoch(0, 0.0)
        for left, right in ((jd, et), (et, synthetic), (jd, synthetic)):
            with self.assertRaises(SolarSystemContractError):
                coordinate_epoch_difference(left, right)
            with self.assertRaises(SolarSystemContractError):
                compare_coordinate_epochs(left, right)

    def test_large_whole_difference_is_exact_within_cap(self) -> None:
        left = jd_epoch(1 << 4094, 0.25)
        right = jd_epoch(1 << 4094, -0.25)
        self.assertEqual(coordinate_epoch_difference(left, right), Fraction(1, 2))


class CoordinateEpochShiftTests(unittest.TestCase):
    def test_half_open_canonical_carry_kat(self) -> None:
        source = jd_epoch(10, 0.25, artifact_ids=("artifact.time.fixture",))
        shifted = shift_coordinate_epoch(source, Fraction(1, 4))
        validate_integrity(shifted)
        self.assertEqual((shifted.whole, shifted.fraction), (11, -0.5))
        self.assertEqual(shifted.artifact_ids, source.artifact_ids)
        self.assertEqual(
            shifted.content_sha256,
            "2d10a1480bbc5941a88235dbb6be1d838d5e76e8bd0431e1de978f6878a3d057",
        )

    def test_borrow_and_exact_zero_canonicalization(self) -> None:
        borrowed = shift_coordinate_epoch(jd_epoch(0, -0.25), Fraction(-1, 2))
        self.assertEqual((borrowed.whole, borrowed.fraction), (-1, 0.25))
        zero = shift_coordinate_epoch(jd_epoch(0, -0.25), Fraction(1, 4))
        self.assertEqual((zero.whole, zero.fraction.hex()), (0, "0x0.0p+0"))

    def test_zero_shift_reconstructs_identical_sealed_content(self) -> None:
        source = et_epoch(
            123,
            -0.125,
            artifact_ids=("artifact.time.a", "artifact.time.b"),
        )
        shifted = shift_coordinate_epoch(source, Fraction(0))
        self.assertIsNot(shifted, source)
        self.assertEqual(shifted.content_sha256, source.content_sha256)
        self.assertEqual(shifted.artifact_ids, source.artifact_ids)

    def test_shift_difference_inverse_on_fixed_dyadic_grid(self) -> None:
        sources = (
            jd_epoch(-5, -0.5),
            jd_epoch(0, 0.0),
            jd_epoch(7, 0.375),
            et_epoch(100, -0.25),
            synthetic_epoch(-3, 0.125),
        )
        offsets = (
            Fraction(-9, 8),
            Fraction(-1, 4),
            Fraction(0),
            Fraction(3, 8),
            Fraction(2),
        )
        for source in sources:
            for offset in offsets:
                shifted = shift_coordinate_epoch(source, offset)
                self.assertEqual(
                    coordinate_epoch_difference(shifted, source),
                    offset,
                )
                restored = shift_coordinate_epoch(shifted, -offset)
                self.assertEqual(restored.content_sha256, source.content_sha256)

    def test_nonrepresentable_remainder_fails_without_rounding(self) -> None:
        for source in (jd_epoch(0, 0.0), et_epoch(0, 0.0), synthetic_epoch(0, 0.0)):
            with self.assertRaises(SolarSystemContractError):
                shift_coordinate_epoch(source, Fraction(1, 3))

    def test_subnormal_exact_remainder_and_underflow_wall(self) -> None:
        shifted = shift_coordinate_epoch(jd_epoch(0, 0.0), Fraction(1, 1 << 1074))
        self.assertEqual(shifted.fraction.hex(), "0x0.0000000000001p-1022")
        with self.assertRaises(SolarSystemContractError):
            shift_coordinate_epoch(jd_epoch(0, 0.0), Fraction(1, 1 << 1075))

    def test_offset_and_output_caps_fail_closed(self) -> None:
        with self.assertRaises(SolarSystemContractError):
            shift_coordinate_epoch(jd_epoch(0, 0.0), Fraction(1 << 4096, 1))
        with self.assertRaises(SolarSystemContractError):
            shift_coordinate_epoch(jd_epoch(0, 0.0), Fraction(1, 1 << 4096))

        maximum = (1 << 4096) - 1
        with self.assertRaises(SolarSystemContractError):
            shift_coordinate_epoch(jd_epoch(maximum, 0.0), Fraction(1))

        left = jd_epoch(maximum, 0.0)
        right = jd_epoch(-maximum, 0.0)
        with self.assertRaises(SolarSystemContractError):
            coordinate_epoch_difference(left, right)

    def test_wrong_types_subclasses_and_stale_seals_reject(self) -> None:
        source = jd_epoch(0, 0.0)
        for invalid in (0, 0.0, False, None):
            with self.subTest(invalid=invalid), self.assertRaises(SolarSystemContractError):
                shift_coordinate_epoch(source, invalid)  # type: ignore[arg-type]

        class FractionSubclass(Fraction):
            pass

        with self.assertRaises(SolarSystemContractError):
            shift_coordinate_epoch(source, FractionSubclass(1, 2))

        stale = dataclasses.replace(source)
        object.__setattr__(stale, "whole", 1)
        with self.assertRaises(SolarSystemContractError):
            shift_coordinate_epoch(stale, Fraction(0))
        with self.assertRaises(SolarSystemContractError):
            compare_coordinate_epochs(stale, source)
        with self.assertRaises(SolarSystemContractError):
            compare_coordinate_epochs(source, object())  # type: ignore[arg-type]

    def test_epoch_subclass_is_rejected(self) -> None:
        class EpochSubclass(CoordinateEpoch):
            pass

        with self.assertRaises(SolarSystemContractError):
            EpochSubclass(
                time_scale="TDB",
                representation="JD_TWO_PART",
                whole=0,
                fraction=0.0,
                coordinate_unit_id=DAY_UNIT,
                origin_id="JULIAN_DATE",
                realization_id="realization.fixture.jd",
                artifact_ids=(),
            )


class TimeImportBoundaryTests(unittest.TestCase):
    def test_clean_import_is_dependency_free(self) -> None:
        code = """
import json, sys
import jxplanetx.solar_system.time as time_module
forbidden = [
    name for name in (
        'numpy', 'astropy', 'erfa', 'spiceypy', 'jplephem', 'skyfield', 'rebound'
    ) if name in sys.modules
]
print(json.dumps({'exports': time_module.__all__, 'forbidden': forbidden}))
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
