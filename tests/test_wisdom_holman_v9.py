from __future__ import annotations

import concurrent.futures
import ctypes
import hashlib
import itertools
import math
from pathlib import Path
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

from benchmarks import rebound_whfast_comparison as fixture
from jxplanetx import _wisdom_holman_cpu_v3 as native_v3
from jxplanetx import _wisdom_holman_cpu_v8 as native_v8
from jxplanetx import _wisdom_holman_cpu_v9 as native_v9
from jxplanetx import fast_wisdom_holman as fast_wh


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "src/jxplanetx/_wisdom_holman_loop.c": (
        "2d66d776bd493a5c3ef0bed2deb5d78bb27f397a9e302daa186290d52588315c"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v2.c": (
        "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v3.c": (
        "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v4.c": (
        "10981b416e4bb90a89babfc05cdd9c97bf1e5b49a0956490300bc5cbc591a16c"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v6.c": (
        "c1da285ec602185f093c7f702097650f009827285332e2aeb1c5f06aa848590d"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v7.c": (
        "337c92c1ea4aba54f64e8e336889a02dccb23d2bd6c275dedfac39e73fd9c0bb"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v8.c": (
        "6fe8b4fcda8947bc21aacce56d9e3383a1d5d579176c742a994628967927b004"
    ),
    "src/jxplanetx/_wisdom_holman_loop_v9.c": (
        "680b4f887fa82561bb503a7982367f68c9c6648cd67ed8b926f1ae87e0d41bd2"
    ),
    "src/jxplanetx/_wisdom_holman_cpu_v7_module.c": (
        "dba08b8c899f4039e5e24fbb897f23c57c99da00deb9e084960fb910cd96b94c"
    ),
    "src/jxplanetx/_wisdom_holman_cpu_v8_module.c": (
        "fcf77481d241902f2c05eabaaac9bfa89f208e645d9ae760baf67dcd03e9d288"
    ),
    "src/jxplanetx/_wisdom_holman_cpu_v9_module.c": (
        "90468c9f527eefb6347b31e8e371310d1e054e2703bb6043baaae71d4078d654"
    ),
}
PERIOD = float.fromhex("0x1.91ec4658ac4a3p+2")
INITIAL_CARTESIAN_HEX = (
    (
        "0x1.56f40f71a182dp-9",
        "-0x1.3a5c50c2d6a40p-11",
        "0x0p+0",
        "0x1.3d399c60c981cp-11",
        "0x1.2b6efeed14806p-11",
        "0x0p+0",
    ),
    (
        "0x1.ad1b090b71629p-1",
        "0x1.d2c3e5372a878p-2",
        "0x0p+0",
        "-0x1.f55cc646a3929p-2",
        "0x1.db8934f99b592p-1",
        "0x0p+0",
    ),
    (
        "-0x1.ba311957d4125p+0",
        "0x1.2670d10ac1618p-4",
        "0x0p+0",
        "-0x1.d8defc9997fb1p-5",
        "-0x1.7ff9ccf690af4p-1",
        "0x0p+0",
    ),
)


class WisdomHolmanV9PromotionTests(unittest.TestCase):
    @staticmethod
    def _run(
        native: object,
        positions: np.ndarray,
        velocities: np.ndarray,
        gm: np.ndarray,
        fixed_step: float,
        checkpoints: tuple[int, ...] = (0, 1),
        radii: np.ndarray | None = None,
        minimum_encounter: float = 1.0e-8,
        minimum_periapse: float = 1.0e-8,
        maximum_barycenter_position: float = 1.0e-8,
        maximum_barycenter_velocity: float = 1.0e-8,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        body_count = int(gm.size)
        if radii is None:
            radii = np.zeros(body_count, dtype=np.float64)
        output_positions = np.full(
            (len(checkpoints), body_count, 3),
            math.pi,
            dtype=np.float64,
        )
        output_velocities = np.full_like(output_positions, -math.e)
        metrics = np.full(
            int(native.MAP_METRIC_COUNT),
            -12345.25,
            dtype=np.float64,
        )
        counters = np.full(
            int(native.MAP_COUNTER_COUNT),
            -123456789,
            dtype=np.int64,
        )
        status = int(
            native.integrate_map(
                body_count,
                len(checkpoints),
                np.ascontiguousarray(positions, dtype=np.float64),
                np.ascontiguousarray(velocities, dtype=np.float64),
                np.ascontiguousarray(gm, dtype=np.float64),
                np.ascontiguousarray(radii, dtype=np.float64),
                np.asarray(checkpoints, dtype=np.int64),
                float(fixed_step),
                float(minimum_encounter),
                float(minimum_periapse),
                float(maximum_barycenter_position),
                float(maximum_barycenter_velocity),
                output_positions,
                output_velocities,
                metrics,
                counters,
            )
        )
        return status, output_positions, output_velocities, metrics, counters

    @staticmethod
    def _two_body_state(
        semimajor_axis: float,
        eccentricity: float,
        true_anomaly: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        gm = np.asarray((1.0, 1.0e-3), dtype=np.float64)
        total = float(np.sum(gm))
        parameter = semimajor_axis * (1.0 - eccentricity * eccentricity)
        radius = parameter / (
            1.0 + eccentricity * math.cos(true_anomaly)
        )
        relative_position = np.asarray(
            (
                radius * math.cos(true_anomaly),
                radius * math.sin(true_anomaly),
                0.0,
            ),
            dtype=np.float64,
        )
        velocity_scale = math.sqrt(total / parameter)
        relative_velocity = np.asarray(
            (
                -velocity_scale * math.sin(true_anomaly),
                velocity_scale * (eccentricity + math.cos(true_anomaly)),
                0.0,
            ),
            dtype=np.float64,
        )
        positions = np.stack(
            (
                -(gm[1] / total) * relative_position,
                (gm[0] / total) * relative_position,
            )
        )
        velocities = np.stack(
            (
                -(gm[1] / total) * relative_velocity,
                (gm[0] / total) * relative_velocity,
            )
        )
        period = 2.0 * math.pi * math.sqrt(semimajor_axis**3 / total)
        periapse_timescale = period * math.sqrt(
            (1.0 - eccentricity) ** 3 / (1.0 + eccentricity)
        )
        step = min(period / 64.0, periapse_timescale / 32.0)
        return positions, velocities, gm, step, velocity_scale

    @staticmethod
    def _official_arguments(
        checkpoints: tuple[int, ...],
    ) -> dict[str, object]:
        values = np.asarray(
            [[float.fromhex(value) for value in row]
             for row in INITIAL_CARTESIAN_HEX],
            dtype=np.float64,
        )
        return {
            "positions": np.ascontiguousarray(values[:, :3]),
            "velocities": np.ascontiguousarray(values[:, 3:]),
            "gm": np.asarray((1.0, 0.001, 0.002), dtype=np.float64),
            "fixed_step": PERIOD / 64.0,
            "checkpoints": checkpoints,
            "minimum_encounter": 0.01,
            "minimum_periapse": 0.05,
            "maximum_barycenter_position": 1.0e-12,
            "maximum_barycenter_velocity": 1.0e-12,
        }

    @staticmethod
    def _double_from_bits(bits: int) -> float:
        return struct.unpack("=d", struct.pack("=Q", bits))[0]

    @staticmethod
    def _double_bits(value: float) -> int:
        return struct.unpack("=Q", struct.pack("=d", value))[0]

    def assertArrayBitsEqual(
        self,
        left: np.ndarray,
        right: np.ndarray,
    ) -> None:
        self.assertIs(type(left), np.ndarray)
        self.assertIs(type(right), np.ndarray)
        self.assertEqual(left.dtype.str, right.dtype.str)
        self.assertEqual(left.shape, right.shape)
        self.assertEqual(left.tobytes(order="C"), right.tobytes(order="C"))

    def assertNativeResultsBitsEqual(
        self,
        left: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        right: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        self.assertEqual(left[0], right[0])
        for left_array, right_array in zip(left[1:], right[1:], strict=True):
            self.assertArrayBitsEqual(left_array, right_array)

    def test_frozen_sources_and_native_identity_are_exact(self) -> None:
        for relative, expected in SOURCE_SHA256.items():
            with self.subTest(path=relative):
                path = ROOT / relative
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    expected,
                )
        self.assertEqual(
            native_v9.KERNEL_ID,
            (
                "jx.wisdom-holman.native-g-bundle."
                "bundled-series-bit-next.prototype.v9"
            ),
        )
        self.assertEqual(
            native_v9.MAP_KERNEL_ID,
            (
                "jx.wisdom-holman.native-complete-map."
                "bundled-series-bit-next.prototype.v9"
            ),
        )
        self.assertEqual(native_v9.SCIENTIFIC_CLAIM_STATE, "SCREENING_ONLY")
        self.assertEqual(
            native_v9.EXACT_FALLBACK_KERNEL_ID,
            native_v8.EXACT_FALLBACK_KERNEL_ID,
        )
        self.assertEqual(native_v9.MAP_MAX_BODIES, native_v8.MAP_MAX_BODIES)
        self.assertEqual(
            native_v9.MAP_METRIC_COUNT,
            native_v8.MAP_METRIC_COUNT,
        )
        self.assertEqual(
            native_v9.MAP_COUNTER_COUNT,
            native_v8.MAP_COUNTER_COUNT,
        )
        self.assertEqual(native_v9.COMPILER_FLAGS, native_v8.COMPILER_FLAGS)

    def test_runtime_fp_environment_is_exact_and_supported_by_default(
        self,
    ) -> None:
        environment = native_v9._runtime_fp_environment()
        self.assertIs(type(environment), dict)
        self.assertEqual(
            set(environment),
            {
                "compiler_family",
                "compiled_linux",
                "compiled_x86_64",
                "compiled_little_endian",
                "runtime_linux",
                "runtime_x86_64",
                "runtime_little_endian",
                "ieee_binary64_layout",
                "rounding_mode",
                "fe_tonearest",
                "round_to_nearest",
                "mxcsr_available",
                "mxcsr",
                "mxcsr_all_exceptions_masked",
                "mxcsr_rounding_control",
                "mxcsr_round_to_nearest",
                "mxcsr_ftz",
                "mxcsr_daz",
                "supported",
            },
        )
        self.assertIs(type(environment["compiler_family"]), str)
        self.assertIn(
            environment["compiler_family"],
            {"GCC", "CLANG", "MSVC", "OTHER"},
        )
        bool_fields = {
            "compiled_linux",
            "compiled_x86_64",
            "compiled_little_endian",
            "runtime_linux",
            "runtime_x86_64",
            "runtime_little_endian",
            "ieee_binary64_layout",
            "round_to_nearest",
            "mxcsr_available",
            "mxcsr_all_exceptions_masked",
            "mxcsr_round_to_nearest",
            "mxcsr_ftz",
            "mxcsr_daz",
            "supported",
        }
        int_fields = {
            "rounding_mode",
            "fe_tonearest",
            "mxcsr",
            "mxcsr_rounding_control",
        }
        for field in bool_fields:
            with self.subTest(field=field):
                self.assertIs(type(environment[field]), bool)
        for field in int_fields:
            with self.subTest(field=field):
                self.assertIs(type(environment[field]), int)

        runtime_is_linux = sys.platform.startswith("linux")
        runtime_is_x86_64 = platform.machine().lower() in {
            "x86_64",
            "amd64",
        }
        self.assertEqual(environment["runtime_linux"], runtime_is_linux)
        self.assertEqual(environment["runtime_x86_64"], runtime_is_x86_64)
        self.assertEqual(
            environment["runtime_little_endian"],
            sys.byteorder == "little",
        )
        self.assertTrue(environment["ieee_binary64_layout"])
        self.assertEqual(
            environment["rounding_mode"],
            environment["fe_tonearest"],
        )
        self.assertTrue(environment["round_to_nearest"])

        if environment["mxcsr_available"]:
            self.assertEqual(
                environment["mxcsr_all_exceptions_masked"],
                environment["mxcsr"] & 0x1F80 == 0x1F80,
            )
            self.assertEqual(
                environment["mxcsr_rounding_control"],
                (environment["mxcsr"] >> 13) & 3,
            )
            self.assertEqual(
                environment["mxcsr_round_to_nearest"],
                environment["mxcsr_rounding_control"] == 0,
            )
            self.assertEqual(
                environment["mxcsr_ftz"],
                bool(environment["mxcsr"] & (1 << 15)),
            )
            self.assertEqual(
                environment["mxcsr_daz"],
                bool(environment["mxcsr"] & (1 << 6)),
            )
        else:
            self.assertEqual(environment["mxcsr"], 0)
            self.assertFalse(environment["mxcsr_all_exceptions_masked"])
            self.assertEqual(environment["mxcsr_rounding_control"], -1)
            self.assertFalse(environment["mxcsr_round_to_nearest"])
            self.assertFalse(environment["mxcsr_ftz"])
            self.assertFalse(environment["mxcsr_daz"])

        expected_support = (
            environment["compiler_family"] == "GCC"
            and environment["compiled_linux"]
            and environment["compiled_x86_64"]
            and environment["compiled_little_endian"]
            and environment["runtime_linux"]
            and environment["runtime_x86_64"]
            and environment["runtime_little_endian"]
            and environment["ieee_binary64_layout"]
            and environment["round_to_nearest"]
            and environment["mxcsr_available"]
            and environment["mxcsr_all_exceptions_masked"]
            and environment["mxcsr_round_to_nearest"]
            and not environment["mxcsr_ftz"]
            and not environment["mxcsr_daz"]
        )
        self.assertEqual(environment["supported"], expected_support)
        if runtime_is_linux and runtime_is_x86_64:
            self.assertTrue(environment["compiled_linux"])
            self.assertTrue(environment["compiled_x86_64"])
            self.assertEqual(
                environment["compiled_little_endian"],
                sys.byteorder == "little",
            )
        if (
            environment["compiler_family"] == "GCC"
            and runtime_is_linux
            and runtime_is_x86_64
            and sys.byteorder == "little"
        ):
            self.assertTrue(environment["supported"])

    def test_runtime_fp_environment_rejects_non_nearest_rounding(
        self,
    ) -> None:
        default_environment = native_v9._runtime_fp_environment()
        if not (
            default_environment["runtime_linux"]
            and default_environment["runtime_x86_64"]
        ):
            self.skipTest("fenv mutation is limited to the Linux x86_64 gate")

        libm = ctypes.CDLL(None)
        libm.fegetround.argtypes = []
        libm.fegetround.restype = ctypes.c_int
        libm.fesetround.argtypes = [ctypes.c_int]
        libm.fesetround.restype = ctypes.c_int
        original_rounding = int(libm.fegetround())
        self.assertEqual(
            original_rounding,
            default_environment["rounding_mode"],
        )

        # FE_DOWNWARD is 0x400 in the guarded Linux x86_64 ABI.
        non_nearest_rounding = 0x400
        changed_environment: dict[str, object] | None = None
        try:
            self.assertEqual(libm.fesetround(non_nearest_rounding), 0)
            changed_environment = native_v9._runtime_fp_environment()
        finally:
            if libm.fesetround(original_rounding) != 0:
                raise RuntimeError("failed to restore the floating-point mode")

        self.assertIsNotNone(changed_environment)
        assert changed_environment is not None
        self.assertEqual(
            changed_environment["rounding_mode"],
            non_nearest_rounding,
        )
        self.assertFalse(changed_environment["round_to_nearest"])
        self.assertFalse(changed_environment["mxcsr_round_to_nearest"])
        self.assertFalse(changed_environment["supported"])
        restored_environment = native_v9._runtime_fp_environment()
        self.assertEqual(
            restored_environment["rounding_mode"],
            original_rounding,
        )
        self.assertEqual(
            restored_environment["mxcsr_rounding_control"],
            default_environment["mxcsr_rounding_control"],
        )
        self.assertEqual(
            restored_environment["mxcsr_all_exceptions_masked"],
            default_environment["mxcsr_all_exceptions_masked"],
        )
        self.assertEqual(
            restored_environment["mxcsr_ftz"],
            default_environment["mxcsr_ftz"],
        )
        self.assertEqual(
            restored_environment["mxcsr_daz"],
            default_environment["mxcsr_daz"],
        )
        self.assertEqual(
            restored_environment["supported"],
            default_environment["supported"],
        )

    def test_runtime_fp_environment_rejects_unmasked_mxcsr_exception(
        self,
    ) -> None:
        default_environment = native_v9._runtime_fp_environment()
        if not (
            default_environment["runtime_linux"]
            and default_environment["runtime_x86_64"]
            and default_environment["mxcsr_available"]
        ):
            self.skipTest("MXCSR mutation is limited to the Linux x86_64 gate")
        gcc = shutil.which("gcc")
        self.assertIsNotNone(gcc, "supported GCC gate requires gcc")
        assert gcc is not None

        helper_source = """
#include <xmmintrin.h>

unsigned int jx_test_get_mxcsr(void)
{
    return _mm_getcsr();
}

void jx_test_set_mxcsr(unsigned int value)
{
    _mm_setcsr(value);
}
"""
        child_source = """
import ctypes
import sys

from jxplanetx import _wisdom_holman_cpu_v9 as native_v9

helper = ctypes.CDLL(sys.argv[1])
helper.jx_test_get_mxcsr.argtypes = []
helper.jx_test_get_mxcsr.restype = ctypes.c_uint
helper.jx_test_set_mxcsr.argtypes = [ctypes.c_uint]
helper.jx_test_set_mxcsr.restype = None

original = int(helper.jx_test_get_mxcsr())
if original & 0x1F80 != 0x1F80:
    raise RuntimeError("child did not start with all MXCSR exceptions masked")
changed = original & ~(1 << 7)
try:
    helper.jx_test_set_mxcsr(changed)
    report = native_v9._runtime_fp_environment()
    if report["mxcsr_all_exceptions_masked"] is not False:
        raise RuntimeError("unmasked MXCSR exception was not reported")
    if report["supported"] is not False:
        raise RuntimeError("unmasked MXCSR exception was accepted")
finally:
    helper.jx_test_set_mxcsr(original)

restored = native_v9._runtime_fp_environment()
if restored["mxcsr_all_exceptions_masked"] is not True:
    raise RuntimeError("MXCSR exception masks were not restored")
if restored["supported"] is not True:
    raise RuntimeError("restored MXCSR environment was not supported")
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mxcsr_helper.c"
            library = root / "mxcsr_helper.so"
            source.write_text(helper_source, encoding="utf-8")
            compiled = subprocess.run(
                (
                    gcc,
                    "-shared",
                    "-fPIC",
                    "-O2",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    str(source),
                    "-o",
                    str(library),
                ),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stdout)
            child = subprocess.run(
                (sys.executable, "-c", child_source, str(library)),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=60.0,
            )
            self.assertEqual(child.returncode, 0, child.stdout)

    def test_supported_fast_route_remains_on_v3(self) -> None:
        self.assertIs(fast_wh._native, native_v3)
        self.assertEqual(
            fast_wh.KERNEL_ID,
            (
                "jx.wisdom-holman.native-complete-map."
                "checkpoint-sync.prototype.v3"
            ),
        )
        self.assertNotEqual(fast_wh.KERNEL_ID, native_v9.MAP_KERNEL_ID)
        identity = fast_wh.fast_wisdom_holman_runtime_identity()
        self.assertEqual(identity["kernel_id"], native_v3.MAP_KERNEL_ID)
        self.assertEqual(identity["g_kernel_id"], native_v3.KERNEL_ID)
        self.assertEqual(
            Path(identity["extension_binary"]["path"]).resolve(),
            Path(native_v3.__file__).resolve(),
        )

        profile = fixture.Profile(1, 1, 4)
        snapshot, plan = fixture._state_and_plan(profile)
        spec = fixture._wh_spec(profile, "long", 64)
        result = fast_wh.integrate_fast_wisdom_holman_trajectory(
            snapshot,
            plan,
            spec,
        )
        self.assertEqual(result.kernel_id, native_v3.MAP_KERNEL_ID)
        self.assertEqual(result.certified_steps, 64)
        self.assertEqual(result.certified_kepler_solves, 128)

    def test_v9_is_not_promoted_over_v3_divergence_cases(self) -> None:
        forward = self._official_arguments((0, 1, 2, 7, 19, 31, 64))
        backward = self._official_arguments((0, 3, 11, 37, 64))
        backward["fixed_step"] = -float(backward["fixed_step"])

        positions, velocities, gm, step, speed_scale = self._two_body_state(
            1.0,
            0.1,
            0.0,
        )
        two_body = {
            "positions": positions,
            "velocities": velocities,
            "gm": gm,
            "fixed_step": step,
            "checkpoints": (0, 1, 7, 64),
            "minimum_encounter": 1.0e-8,
            "minimum_periapse": 1.0e-8,
            "maximum_barycenter_position": 1.0e-8,
            "maximum_barycenter_velocity": speed_scale * 1.0e-8,
        }

        for name, arguments in (
            ("irregular-forward", forward),
            ("irregular-backward", backward),
            ("two-body", two_body),
        ):
            with self.subTest(case=name):
                supported = self._run(native_v3, **arguments)
                screening = self._run(native_v9, **arguments)
                self.assertEqual(supported[0], 0)
                self.assertEqual(screening[0], 0)
                self.assertNotEqual(
                    supported[1].tobytes(order="C"),
                    screening[1].tobytes(order="C"),
                )
                self.assertNotEqual(
                    supported[2].tobytes(order="C"),
                    screening[2].tobytes(order="C"),
                )

    def test_bit_next_matches_libm_edges_binades_and_random_bits(self) -> None:
        sign_mask = 1 << 63
        exponent_mask = 0x7FF << 52
        fraction_mask = (1 << 52) - 1
        positive_infinity = exponent_mask
        negative_infinity = sign_mask | exponent_mask

        def check(from_bits: int, toward_bits: int) -> None:
            from_value = self._double_from_bits(from_bits)
            toward_value = self._double_from_bits(toward_bits)
            expected = math.nextafter(from_value, toward_value)
            observed = native_v9._nextafter_bits(from_value, toward_value)
            self.assertEqual(
                self._double_bits(observed),
                self._double_bits(expected),
                (
                    f"from=0x{from_bits:016x}, "
                    f"toward=0x{toward_bits:016x}"
                ),
            )

        edge_bits = (
            0x0000000000000000,
            0x8000000000000000,
            0x0000000000000001,
            0x8000000000000001,
            0x000FFFFFFFFFFFFF,
            0x800FFFFFFFFFFFFF,
            0x0010000000000000,
            0x8010000000000000,
            0x3FEFFFFFFFFFFFFF,
            0xBFEFFFFFFFFFFFFF,
            0x3FF0000000000000,
            0xBFF0000000000000,
            0x7FEFFFFFFFFFFFFF,
            0xFFEFFFFFFFFFFFFF,
            positive_infinity,
            negative_infinity,
            0x7FF8000000000001,
            0xFFF8000000000001,
        )
        checked = 0
        for from_bits, toward_bits in itertools.product(edge_bits, repeat=2):
            check(from_bits, toward_bits)
            checked += 1
        for exponent in range(0x7FF):
            for sign in (0, sign_mask):
                for fraction in (0, 1, fraction_mask):
                    from_bits = sign | (exponent << 52) | fraction
                    for toward_bits in (negative_infinity, positive_infinity):
                        check(from_bits, toward_bits)
                        checked += 1
        generator = np.random.Generator(np.random.PCG64(2026101601))
        raw_pairs = generator.bit_generator.random_raw((100_000, 2))
        for raw_from, raw_toward in raw_pairs:
            from_bits = int(raw_from)
            toward_bits = int(raw_toward)
            if from_bits & exponent_mask == exponent_mask:
                from_bits ^= 1 << 52
            if toward_bits & exponent_mask == exponent_mask:
                toward_bits ^= 1 << 52
            check(from_bits, toward_bits)
            checked += 1
        self.assertEqual(checked, len(edge_bits) ** 2 + 24_564 + 100_000)

    def test_g_bundle_is_bitwise_v8_at_series_boundaries(self) -> None:
        beta_values = (
            math.ldexp(1.0, -600),
            math.ldexp(1.0, -400),
            1.0e-12,
            1.0e-6,
            1.0,
            1.0e6,
            1.0e12,
            math.ldexp(1.0, 400),
            math.ldexp(1.0, 600),
        )
        arguments = (
            -math.nextafter(0.5, math.inf),
            -0.5,
            math.nextafter(-0.5, 0.0),
            -0.0,
            0.0,
            math.nextafter(0.5, 0.0),
            0.5,
            math.nextafter(0.5, math.inf),
        )
        checked = 0
        for beta in beta_values:
            root_beta = math.sqrt(beta)
            for argument in arguments:
                anomaly = argument / root_beta
                reference = native_v8.adaptive_g_bundle(beta, anomaly)
                candidate = native_v9.adaptive_g_bundle(beta, anomaly)
                for index in range(4):
                    self.assertEqual(
                        self._double_bits(candidate[index]),
                        self._double_bits(reference[index]),
                    )
                self.assertEqual(candidate[4:], reference[4:])
                checked += 1
        self.assertEqual(checked, 72)

    def test_elliptic_grid_is_bitwise_v8_for_all_480_cases(self) -> None:
        cases = 0
        successes = 0
        for semimajor_axis in (1.0e-6, 1.0, 1.0e6):
            for eccentricity in (0.0, 0.1, 0.5, 0.85, 0.9):
                for anomaly_index in range(16):
                    anomaly = 2.0 * math.pi * anomaly_index / 16.0
                    positions, velocities, gm, step, speed_scale = (
                        self._two_body_state(
                            semimajor_axis,
                            eccentricity,
                            anomaly,
                        )
                    )
                    for direction in (-1.0, 1.0):
                        guard_length = semimajor_axis * 1.0e-8
                        arguments = {
                            "positions": positions,
                            "velocities": velocities,
                            "gm": gm,
                            "fixed_step": direction * step,
                            "minimum_encounter": guard_length,
                            "minimum_periapse": guard_length,
                            "maximum_barycenter_position": guard_length,
                            "maximum_barycenter_velocity": speed_scale * 1.0e-8,
                        }
                        reference = self._run(native_v8, **arguments)
                        reference_fallback = native_v8.last_fallback_count()
                        reference_cache = native_v8.last_cache_stats()
                        candidate = self._run(native_v9, **arguments)
                        self.assertNativeResultsBitsEqual(candidate, reference)
                        self.assertEqual(
                            native_v9.last_fallback_count(),
                            reference_fallback,
                        )
                        self.assertEqual(
                            native_v9.last_cache_stats(),
                            reference_cache,
                        )
                        successes += int(candidate[0] == 0)
                        cases += 1
        self.assertEqual(cases, 480)
        self.assertGreater(successes, 0)

    def test_long_full_endpoint_and_replay_are_bitwise_v8(self) -> None:
        full_checkpoints = tuple(range(0, 6401, 16))
        full_arguments = self._official_arguments(full_checkpoints)
        endpoint_arguments = self._official_arguments((0, 6400))
        reference_full = self._run(native_v8, **full_arguments)
        candidate_full = self._run(native_v9, **full_arguments)
        self.assertNativeResultsBitsEqual(candidate_full, reference_full)
        self.assertEqual(candidate_full[0], 0)
        self.assertEqual(native_v9.last_fallback_count(), 0)
        self.assertEqual(native_v9.last_cache_stats(), (1, 19200, 1))

        reference_endpoint = self._run(native_v8, **endpoint_arguments)
        candidate_endpoint = self._run(native_v9, **endpoint_arguments)
        self.assertNativeResultsBitsEqual(
            candidate_endpoint,
            reference_endpoint,
        )
        self.assertEqual(candidate_endpoint[0], 0)
        self.assertArrayBitsEqual(
            candidate_endpoint[1],
            np.ascontiguousarray(candidate_full[1][[0, -1]]),
        )
        self.assertArrayBitsEqual(
            candidate_endpoint[2],
            np.ascontiguousarray(candidate_full[2][[0, -1]]),
        )
        self.assertArrayBitsEqual(candidate_endpoint[3], candidate_full[3])

        replay_first = self._run(native_v9, **full_arguments)
        replay_second = self._run(native_v9, **full_arguments)
        self.assertNativeResultsBitsEqual(replay_first, replay_second)
        self.assertNativeResultsBitsEqual(replay_first, reference_full)

    def test_concurrent_long_calls_are_bitwise_and_tls_diagnostics_frozen(
        self,
    ) -> None:
        full_arguments = self._official_arguments(tuple(range(0, 6401, 16)))
        expected = self._run(native_v9, **full_arguments)
        self.assertEqual(expected[0], 0)
        self.assertEqual(native_v9.last_fallback_count(), 0)
        self.assertEqual(native_v9.last_cache_stats(), (1, 19200, 1))

        workers = 4
        calls_per_worker = 3
        start = threading.Barrier(workers)

        def run_worker(
            _: int,
        ) -> tuple[
            int,
            list[
                tuple[
                    tuple[
                        int,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                        np.ndarray,
                    ],
                    int,
                    tuple[int, int, int],
                ]
            ],
        ]:
            start.wait(timeout=60.0)
            records = []
            for _ in range(calls_per_worker):
                result = self._run(native_v9, **full_arguments)
                records.append(
                    (
                        result,
                        native_v9.last_fallback_count(),
                        native_v9.last_cache_stats(),
                    )
                )
            return threading.get_ident(), records

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=workers,
        ) as executor:
            concurrent_results = tuple(executor.map(run_worker, range(workers)))

        self.assertEqual(
            len({thread_id for thread_id, _ in concurrent_results}),
            workers,
        )
        self.assertEqual(
            sum(len(records) for _, records in concurrent_results),
            workers * calls_per_worker,
        )
        for _, records in concurrent_results:
            for observed, fallback_count, cache_stats in records:
                self.assertNativeResultsBitsEqual(observed, expected)
                self.assertEqual(fallback_count, 0)
                self.assertEqual(cache_stats, (1, 19200, 1))

        # Worker calls must not replace the calling thread's frozen TLS state.
        self.assertEqual(native_v9.last_fallback_count(), 0)
        self.assertEqual(native_v9.last_cache_stats(), (1, 19200, 1))

    def test_failures_buffers_diagnostics_and_boundary_match_v8(self) -> None:
        positions, velocities, gm, step, speed_scale = self._two_body_state(
            1.0,
            0.1,
            0.0,
        )
        base = {
            "positions": positions,
            "velocities": velocities,
            "gm": gm,
            "fixed_step": step,
            "minimum_encounter": 1.0e-8,
            "minimum_periapse": 1.0e-8,
            "maximum_barycenter_position": 1.0e-8,
            "maximum_barycenter_velocity": speed_scale * 1.0e-8,
        }
        changes = (
            {"gm": np.asarray((1.0, -1.0e-3), dtype=np.float64)},
            {"positions": np.full((2, 3), np.nan, dtype=np.float64)},
            {"positions": np.zeros((2, 3), dtype=np.float64)},
            {"gm": np.asarray((1.0, 0.02), dtype=np.float64)},
            {"fixed_step": 10.0 * step},
            {"fixed_step": math.nan},
            {"minimum_encounter": 2.0},
            {"radii": np.asarray((0.6, 0.6), dtype=np.float64)},
        )
        for index, change in enumerate(changes):
            with self.subTest(case=index):
                reference = self._run(native_v8, **(base | change))
                reference_fallback = native_v8.last_fallback_count()
                reference_cache = native_v8.last_cache_stats()
                candidate = self._run(native_v9, **(base | change))
                self.assertNotEqual(candidate[0], 0)
                self.assertNativeResultsBitsEqual(candidate, reference)
                self.assertEqual(
                    native_v9.last_fallback_count(),
                    reference_fallback,
                )
                self.assertEqual(native_v9.last_cache_stats(), reference_cache)

        values = [
            2,
            2,
            positions.copy(),
            velocities.copy(),
            gm.copy(),
            np.zeros(2, dtype=np.float64),
            np.asarray((0, 1), dtype=np.int64),
            step,
            1.0e-8,
            1.0e-8,
            1.0e-8,
            speed_scale * 1.0e-8,
            np.empty((2, 2, 3), dtype=np.float64),
            np.empty((2, 2, 3), dtype=np.float64),
            np.empty(17, dtype=np.float64),
            np.empty(12, dtype=np.int64),
        ]
        invalid = list(values)
        invalid[2] = np.asarray(positions, dtype=np.float32)

        def failure(native: object) -> tuple[type[BaseException], str]:
            try:
                native.integrate_map(*invalid)
            except Exception as error:
                return type(error), str(error)
            self.fail("invalid native schema was accepted")

        self.assertEqual(failure(native_v9), failure(native_v8))
        self.assertIs(type(native_v9.last_fallback_count()), int)
        self.assertTrue(
            all(type(value) is int for value in native_v9.last_cache_stats())
        )


if __name__ == "__main__":
    unittest.main()
