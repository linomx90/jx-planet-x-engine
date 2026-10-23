from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np

from benchmarks import rebound_whfast_comparison as fixture
from jxplanetx import _wisdom_holman_cpu_v3 as native_v3
from jxplanetx import _wisdom_holman_cpu_v9 as native_v9
from jxplanetx import fast_wisdom_holman as fast_v3
from jxplanetx import fast_wisdom_holman_v9 as fast_v9
from jxplanetx.engine import wisdom_holman as public_wh


ROOT = Path(__file__).resolve().parents[1]
V9_SOURCE_SHA256 = {
    "_wisdom_holman_cpu_v9_module.c": (
        "90468c9f527eefb6347b31e8e371310d1e054e2703bb6043baaae71d4078d654"
    ),
    "_wisdom_holman_cpu_v8_module.c": (
        "fcf77481d241902f2c05eabaaac9bfa89f208e645d9ae760baf67dcd03e9d288"
    ),
    "_wisdom_holman_loop_v9.c": (
        "680b4f887fa82561bb503a7982367f68c9c6648cd67ed8b926f1ae87e0d41bd2"
    ),
    "_wisdom_holman_loop_v8.c": (
        "6fe8b4fcda8947bc21aacce56d9e3383a1d5d579176c742a994628967927b004"
    ),
    "_wisdom_holman_loop_v7.c": (
        "337c92c1ea4aba54f64e8e336889a02dccb23d2bd6c275dedfac39e73fd9c0bb"
    ),
    "_wisdom_holman_loop_v6.c": (
        "c1da285ec602185f093c7f702097650f009827285332e2aeb1c5f06aa848590d"
    ),
    "_wisdom_holman_loop_v4.c": (
        "10981b416e4bb90a89babfc05cdd9c97bf1e5b49a0956490300bc5cbc591a16c"
    ),
    "_wisdom_holman_loop_v3.c": (
        "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629"
    ),
    "_wisdom_holman_loop_v2.c": (
        "38b8898bc6c1358522a0f653fc55849719e955889f556fd9f5850ad199346750"
    ),
}


class FastWisdomHolmanV9Tests(unittest.TestCase):
    @staticmethod
    def _case(periods: int = 1) -> tuple[object, object, object]:
        profile = fixture.Profile(1, periods, 4)
        snapshot, plan = fixture._state_and_plan(profile)
        spec = fixture._wh_spec(profile, "long", 64)
        return snapshot, plan, spec

    @staticmethod
    def _assert_array_bits_equal(left: np.ndarray, right: np.ndarray) -> None:
        if (
            type(left) is not np.ndarray
            or type(right) is not np.ndarray
            or left.dtype != right.dtype
            or left.shape != right.shape
            or left.tobytes(order="C") != right.tobytes(order="C")
        ):
            raise AssertionError("array bytes differ")

    def test_v3_public_default_and_serialized_identity_remain_exact(self) -> None:
        self.assertIs(fast_v3._native, native_v3)
        self.assertEqual(
            fast_v3.SCHEMA, "jxplanetx.fast-wisdom-holman.screening-api.v2"
        )
        self.assertEqual(
            fast_v3.API_ID, "jx.fast-wisdom-holman.screening-api.v2"
        )
        self.assertEqual(
            fast_v3.KERNEL_ID,
            "jx.wisdom-holman.native-complete-map.checkpoint-sync.prototype.v3",
        )
        self.assertEqual(
            fast_v3.NATIVE_WRAPPER_SOURCE_SHA256,
            "ee0e40a075b481516d8d30f8824b1fe31c9051c8c0de1dc8b321155882c411b3",
        )
        self.assertEqual(
            fast_v3.NATIVE_LOOP_SOURCE_SHA256,
            "7143fe32304b04901d107b48bf8fe5ae9043118b81c25c635e5d4b412c93b629",
        )
        identity = fast_v3.fast_wisdom_holman_runtime_identity()
        self.assertEqual(identity["kernel_id"], native_v3.MAP_KERNEL_ID)
        self.assertEqual(identity["g_kernel_id"], native_v3.KERNEL_ID)

        snapshot, plan, spec = self._case()
        expected = public_wh.integrate_wisdom_holman_trajectory(
            snapshot, plan, spec
        )
        observed = fast_v3.integrate_fast_wisdom_holman_trajectory(
            snapshot, plan, spec
        )
        self._assert_array_bits_equal(
            observed.positions, np.stack(expected.positions)
        )
        self._assert_array_bits_equal(
            observed.velocities, np.stack(expected.velocities)
        )
        self.assertEqual(observed.api_id, fast_v3.API_ID)
        self.assertEqual(observed.kernel_id, native_v3.MAP_KERNEL_ID)

    def test_v9_identity_is_opt_in_path_free_and_binds_full_closure(self) -> None:
        identity = fast_v9.fast_wisdom_holman_v9_runtime_identity()
        self.assertEqual(identity["schema"], fast_v9.SCHEMA)
        self.assertEqual(identity["api_id"], fast_v9.API_ID)
        self.assertEqual(identity["kernel_id"], native_v9.MAP_KERNEL_ID)
        self.assertEqual(identity["g_kernel_id"], native_v9.KERNEL_ID)
        self.assertEqual(identity["scientific_claim_state"], "SCREENING_ONLY")
        self.assertEqual(
            identity["support_status"],
            "SUPPORTED_OPT_IN_LINUX_X86_64_GCC",
        )
        self.assertIs(identity["public_default"], False)
        for name in (
            "registry_authorized",
            "qualification_authorized",
            "production_authorized",
            "superiority_claimed",
        ):
            self.assertIs(identity[name], False)
        self.assertEqual(identity["compiler_family"], "GCC")
        self.assertEqual(identity["compiler_version"], native_v9.COMPILER_VERSION)
        self.assertEqual(identity["compiler_flags"], native_v9.COMPILER_FLAGS)
        self.assertRegex(identity["content_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(identity["source_closure_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("path", identity["extension_binary"])
        self.assertEqual(
            tuple(record["name"] for record in identity["compiled_sources"]),
            tuple(V9_SOURCE_SHA256),
        )
        for record in identity["compiled_sources"]:
            self.assertNotIn("path", record)
            self.assertEqual(record["sha256"], V9_SOURCE_SHA256[record["name"]])
            source = ROOT / "src" / "jxplanetx" / record["name"]
            self.assertTrue(source.is_file())
            self.assertFalse(source.is_symlink())
            self.assertEqual(
                hashlib.sha256(source.read_bytes()).hexdigest(),
                record["sha256"],
            )
        environment = dict(identity["runtime_fp_environment"])
        self.assertEqual(environment["compiler_family"], "GCC")
        self.assertTrue(environment["supported"])
        self.assertEqual(environment["rounding_mode"], environment["fe_tonearest"])
        self.assertEqual(environment["mxcsr_rounding_control"], 0)
        self.assertTrue(environment["mxcsr_all_exceptions_masked"])
        self.assertEqual(environment["mxcsr"] & 0x1F80, 0x1F80)
        self.assertFalse(environment["mxcsr_ftz"])
        self.assertFalse(environment["mxcsr_daz"])
        self.assertEqual(environment["mxcsr"] & 0x3F, 0)
        self.assertEqual(
            identity["content_sha256"],
            fast_v9.fast_wisdom_holman_v9_runtime_identity()["content_sha256"],
        )

    def test_official_full_endpoint_and_exact_replay(self) -> None:
        snapshot, plan, full_spec = self._case(periods=100)
        full = fast_v9.integrate_fast_wisdom_holman_v9_trajectory(
            snapshot, plan, full_spec
        )
        endpoint_spec = replace(
            full_spec, checkpoint_step_indices=(0, 6400)
        )
        endpoint = fast_v9.integrate_fast_wisdom_holman_v9_trajectory(
            snapshot, plan, endpoint_spec, exact_replay=True
        )

        self.assertEqual(full.certified_steps, 6400)
        self.assertEqual(full.certified_kepler_solves, 12800)
        self.assertEqual(full.replay_count, 0)
        self.assertEqual(endpoint.replay_count, 1)
        self.assertTrue(endpoint.exact_replay_verified)
        self.assertEqual(endpoint.primary_counters, endpoint.replay_counters)
        self.assertEqual(
            endpoint.primary_fallback_count, endpoint.replay_fallback_count
        )
        self.assertEqual(endpoint.primary_cache_stats, endpoint.replay_cache_stats)
        self.assertEqual(full.primary_fallback_count, 0)
        self.assertEqual(
            full.primary_cache_stats,
            (("entries", 1), ("hits", 19200), ("misses", 1)),
        )
        self.assertEqual(endpoint.primary_cache_stats, full.primary_cache_stats)
        self._assert_array_bits_equal(
            endpoint.positions,
            np.ascontiguousarray(full.positions[[0, -1]]),
        )
        self._assert_array_bits_equal(
            endpoint.velocities,
            np.ascontiguousarray(full.velocities[[0, -1]]),
        )
        self.assertEqual(endpoint.metrics, full.metrics)
        self.assertEqual(endpoint.schema, fast_v9.SCHEMA)
        self.assertEqual(endpoint.api_id, fast_v9.API_ID)
        self.assertEqual(endpoint.result_type_id, fast_v9.RESULT_TYPE_ID)
        self.assertEqual(endpoint.kernel_id, native_v9.MAP_KERNEL_ID)
        self.assertIs(endpoint.public_default, False)
        self.assertFalse(endpoint.production_authorized)
        for array in (
            endpoint.positions,
            endpoint.velocities,
            endpoint.checkpoint_epochs,
        ):
            self.assertIs(type(array), np.ndarray)
            self.assertTrue(array.flags.owndata)
            self.assertTrue(array.flags.aligned)
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)

    def test_per_call_diagnostics_and_fp_hooks_bracket_success_and_failure(self) -> None:
        snapshot, plan, spec = self._case()
        workspace = fast_v9.prepare_fast_wisdom_holman_v9_workspace(
            snapshot, plan, spec
        )
        events: list[str] = []
        original_fp = native_v9._runtime_fp_environment
        original_integrate = native_v9.integrate_map
        original_fallback = native_v9.last_fallback_count
        original_cache = native_v9.last_cache_stats

        def fp() -> object:
            events.append("fp")
            return original_fp()

        def integrate(*args: object) -> object:
            events.append("integrate")
            return original_integrate(*args)

        def fallback() -> object:
            events.append("fallback")
            return original_fallback()

        def cache() -> object:
            events.append("cache")
            return original_cache()

        with (
            mock.patch.object(native_v9, "_runtime_fp_environment", side_effect=fp),
            mock.patch.object(native_v9, "integrate_map", side_effect=integrate),
            mock.patch.object(native_v9, "last_fallback_count", side_effect=fallback),
            mock.patch.object(native_v9, "last_cache_stats", side_effect=cache),
        ):
            fast_v9.integrate_fast_wisdom_holman_v9_workspace(workspace)
        self.assertEqual(
            events,
            ["fp", "fp", "fp", "integrate", "fp", "fallback", "cache"],
        )

        failing_spec = replace(spec, minimum_encounter_pair_separation=3.0)
        failing = fast_v9.prepare_fast_wisdom_holman_v9_workspace(
            snapshot, plan, failing_spec
        )
        events.clear()
        with (
            mock.patch.object(native_v9, "_runtime_fp_environment", side_effect=fp),
            mock.patch.object(native_v9, "integrate_map", side_effect=integrate),
            mock.patch.object(native_v9, "last_fallback_count", side_effect=fallback),
            mock.patch.object(native_v9, "last_cache_stats", side_effect=cache),
            self.assertRaisesRegex(fast_v9.FastWisdomHolmanV9Error, "status"),
        ):
            fast_v9.integrate_fast_wisdom_holman_v9_workspace(failing)
        self.assertEqual(
            events,
            ["fp", "fp", "fp", "integrate", "fp", "fallback", "cache"],
        )

    def test_unmasked_mxcsr_exception_fails_before_workspace_math(self) -> None:
        source = r'''
import ctypes
import mmap
import resource

from benchmarks import rebound_whfast_comparison as fixture
from jxplanetx import fast_wisdom_holman_v9 as v9

resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
profile = fixture.Profile(1, 1, 4)
snapshot, plan = fixture._state_and_plan(profile)
spec = fixture._wh_spec(profile, "long", 64)
workspace = v9.prepare_fast_wisdom_holman_v9_workspace(snapshot, plan, spec)

page = mmap.mmap(
    -1,
    mmap.PAGESIZE,
    prot=mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC,
)
page.write(bytes.fromhex("0fae1fc30fae17c3"))
address = ctypes.addressof(ctypes.c_char.from_buffer(page))
operation = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint32))
get_mxcsr = operation(address)
set_mxcsr = operation(address + 4)
original = ctypes.c_uint32()
get_mxcsr(ctypes.byref(original))
altered = ctypes.c_uint32((original.value & ~0x3F) & ~(1 << 12))
rejected = False
try:
    set_mxcsr(ctypes.byref(altered))
    try:
        v9.integrate_fast_wisdom_holman_v9_workspace(workspace)
    except v9.FastWisdomHolmanV9Error:
        rejected = True
finally:
    set_mxcsr(ctypes.byref(original))
if not rejected:
    raise RuntimeError("public V9 route accepted an unmasked MXCSR exception")

original_integrate = v9._native.integrate_map
post_rejected = False
def hostile_integrate(*args):
    status = original_integrate(*args)
    set_mxcsr(ctypes.byref(altered))
    return status
v9._native.integrate_map = hostile_integrate
try:
    try:
        v9.integrate_fast_wisdom_holman_v9_workspace(workspace)
    except v9.FastWisdomHolmanV9Error:
        post_rejected = True
finally:
    set_mxcsr(ctypes.byref(original))
    v9._native.integrate_map = original_integrate
if not post_rejected:
    raise RuntimeError("public V9 route accepted hostile post-run MXCSR controls")
print("REJECTED_BOTH", flush=True)
'''
        environment = dict(os.environ)
        python_path = (str(ROOT / "src"), str(ROOT))
        if environment.get("PYTHONPATH"):
            python_path += (environment["PYTHONPATH"],)
        environment["PYTHONPATH"] = os.pathsep.join(python_path)
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)
        self.assertEqual(completed.stdout.strip(), "REJECTED_BOTH")

    def test_workspace_platform_and_result_custody_fail_closed(self) -> None:
        snapshot, plan, spec = self._case()
        workspace = fast_v9.prepare_fast_wisdom_holman_v9_workspace(
            snapshot, plan, spec
        )
        writable = workspace.positions.copy()
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error, "aligned owned"
        ):
            fast_v9.integrate_fast_wisdom_holman_v9_workspace(
                replace(workspace, positions=writable)
            )
        owned = workspace.positions.copy()
        owned.setflags(write=False)
        nonowned = owned.view()
        self.assertFalse(nonowned.flags.owndata)
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error, "aligned owned"
        ):
            fast_v9.integrate_fast_wisdom_holman_v9_workspace(
                replace(workspace, positions=nonowned)
            )
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error, "built-in bool"
        ):
            fast_v9.integrate_fast_wisdom_holman_v9_workspace(
                workspace, exact_replay=1
            )

        unsupported = dict(native_v9._runtime_fp_environment())
        unsupported["mxcsr"] &= ~(1 << 12)
        unsupported["mxcsr_all_exceptions_masked"] = False
        unsupported["supported"] = False
        integrate = native_v9.integrate_map
        custody = fast_v9._validate_workspace_custody
        with (
            mock.patch.object(
                native_v9,
                "_runtime_fp_environment",
                return_value=unsupported,
            ),
            mock.patch.object(
                fast_v9,
                "_validate_workspace_custody",
                wraps=custody,
            ) as custody_call,
            mock.patch.object(native_v9, "integrate_map", wraps=integrate) as call,
            self.assertRaisesRegex(
                fast_v9.FastWisdomHolmanV9Error,
                "floating-point environment",
            ),
        ):
            fast_v9.integrate_fast_wisdom_holman_v9_workspace(workspace)
        self.assertEqual(custody_call.call_count, 0)
        self.assertEqual(call.call_count, 0)

        result = fast_v9.integrate_fast_wisdom_holman_v9_workspace(workspace)
        changed_positions = result.positions.copy()
        changed_positions[0, 0, 0] = np.nextafter(
            changed_positions[0, 0, 0], np.inf
        )
        changed_positions.setflags(write=False)
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error, "result content digest"
        ):
            replace(result, positions=changed_positions)
        changed_epochs = result.checkpoint_epochs.copy()
        changed_epochs[-1] = np.nextafter(changed_epochs[-1], np.inf)
        changed_epochs.setflags(write=False)
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error, "result content digest"
        ):
            replace(result, checkpoint_epochs=changed_epochs)
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error, "claim ceiling"
        ):
            replace(result, production_authorized=0)
        forged_environment = dict(result.primary_fp_environment_before)
        forged_environment["mxcsr"] |= 1
        forged_environment_tuple = tuple(
            (name, forged_environment[name])
            for name in fast_v9.FP_ENVIRONMENT_KEYS
        )
        with self.assertRaisesRegex(
            fast_v9.FastWisdomHolmanV9Error,
            "primary pre-run environment is unsupported",
        ):
            replace(
                result,
                primary_fp_environment_before=forged_environment_tuple,
                primary_fp_environment_after=forged_environment_tuple,
            )
        with self.assertRaises(ValueError):
            result.positions[0, 0, 0] = 0.0

    def test_known_v3_divergence_has_distinct_public_identities(self) -> None:
        snapshot, plan, spec = self._case()
        irregular = replace(
            spec, checkpoint_step_indices=(0, 1, 2, 7, 19, 31, 64)
        )
        v3_result = fast_v3.integrate_fast_wisdom_holman_trajectory(
            snapshot, plan, irregular
        )
        v9_result = fast_v9.integrate_fast_wisdom_holman_v9_trajectory(
            snapshot, plan, irregular
        )
        self.assertNotEqual(v3_result.api_id, v9_result.api_id)
        self.assertNotEqual(v3_result.kernel_id, v9_result.kernel_id)
        self.assertNotEqual(
            v3_result.positions.tobytes(order="C"),
            v9_result.positions.tobytes(order="C"),
        )
        self.assertNotEqual(
            v3_result.velocities.tobytes(order="C"),
            v9_result.velocities.tobytes(order="C"),
        )


if __name__ == "__main__":
    unittest.main()
