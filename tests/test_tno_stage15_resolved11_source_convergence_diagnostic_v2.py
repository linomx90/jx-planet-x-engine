from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import re
import struct
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "benchmarks"
    / "jx_tno_stage15_resolved11_source_convergence_diagnostic_v2.py"
)
V1_TEST = (
    PROJECT_ROOT / "tests"
    / "test_tno_stage15_resolved11_source_convergence_diagnostic_v1.py"
)


def load_source() -> types.ModuleType:
    payload = SOURCE.read_bytes()
    module = types.ModuleType("_jx_stage15_source_convergence_diagnostic_v2_test")
    module.__file__ = str(SOURCE)
    module.__package__ = ""
    code = compile(
        payload, str(SOURCE), "exec", optimize=__import__("sys").flags.optimize,
    )
    exec(code, module.__dict__)
    return module


M = load_source()


def load_v1_tests() -> types.ModuleType:
    module = types.ModuleType("_jx_stage15_source_convergence_v1_tests_bound_to_v2")
    module.__file__ = str(V1_TEST)
    module.__package__ = "tests"
    code = compile(
        V1_TEST.read_bytes(), str(V1_TEST), "exec",
        optimize=__import__("sys").flags.optimize,
    )
    exec(code, module.__dict__)
    module.M = M
    module.SOURCE = SOURCE
    return module


V1_TESTS = load_v1_tests()


def ias_receipt(annual: list[int]) -> dict:
    return {
        "epsilon_hex": M.IAS15_TIGHT_EPSILON.hex(),
        "initial_dt_hex": M.IAS15_INITIAL_DT.hex(),
        "min_dt_hex": M.IAS15_MIN_DT.hex(),
        "adaptive_mode": M.IAS15_ADAPTIVE_MODE,
        "annual_integrate_call_count": M.HORIZON_YEARS,
        "initial_steps_done": annual[0],
        "final_steps_done": annual[-1],
        "annual_steps_done": annual,
        "annual_steps_done_sha256": hashlib.sha256(
            struct.pack(f"<{len(annual)}Q", *annual)
        ).hexdigest(),
        "final_time_hex": float(M.HORIZON_YEARS).hex(),
        "gravitational_constant_hex": 1.0.hex(),
        "gravity": "basic",
        "collision": "none",
        "boundary": "none",
        "softening_hex": 0.0.hex(),
        "testparticle_type": 0,
        "active_source_count": M.MASSIVE_COUNT,
        "source_count": M.MASSIVE_COUNT,
        "tracer_count": 0,
    }


def run_stubbed_replay(receipt: dict) -> tuple[list[str], list[dict]]:
    order: list[str] = []
    real_validator = M._validate_ias_receipt

    def validator(value: dict) -> None:
        real_validator(value)
        order.append("ias_receipt_validated")

    def finalizer(_pre: dict, _levels: dict, _tight: object, _invariants: dict) -> dict:
        if order != ["ias_receipt_validated"]:
            raise AssertionError("finalizer ran before annual receipt validation")
        order.append("diagnostic_finalized")
        return {
            "disposition": "PASS",
            "h1024_triggered": False,
            "reference_qualified": True,
        }

    massive = tuple(
        types.SimpleNamespace(gm_AU3_per_julian_year2=float(index + 1))
        for index in range(M.MASSIVE_COUNT)
    )
    batch = types.SimpleNamespace(massive=massive, epoch_jd_tdb=2_460_000.5)
    v8 = types.SimpleNamespace(
        _load_v4_context=lambda: (
            types.SimpleNamespace(), None, None, [batch], [{}],
        ),
    )
    source_lanes = [
        {
            "steps_per_julian_year": 256, "byte_offset": 0,
            "byte_length": 1, "lane_payload_sha256": "a" * 64,
        },
        {
            "steps_per_julian_year": 512, "byte_offset": 1,
            "byte_length": 1, "lane_payload_sha256": "b" * 64,
        },
    ]
    annual_lanes = [
        {"lane_id": "jx_resolved11_source_h256", "byte_offset": 0},
        {"lane_id": "jx_resolved11_source_h512", "byte_offset": 1},
        {
            "lane_id": "rebound511_ias15_tight_resolved11_source_only",
            "byte_offset": 2,
        },
    ]
    skeleton = {
        "batch_order": 0,
        "object_id": M.OBJECT_IDS[0],
        "source_lanes": source_lanes,
        "annual_lanes": annual_lanes,
        "tight_ias15_execution_receipt": receipt,
    }

    def source_locator(
        _locator: dict, *, expected_offset: int, batch_order: int, steps: int,
    ) -> int:
        del batch_order, steps
        return expected_offset + 1

    def annual_locator(
        _locator: dict, *, expected_offset: int, batch_order: int,
        kind: str, steps: int,
    ) -> int:
        del batch_order, kind, steps
        return expected_offset + 1

    replacements = {
        "BATCH_COUNT": 1,
        "_v8_source_locator_map": lambda _result: {(0, 64): {}, (0, 128): {}},
        "_batch_initial_source_payload": lambda _np, _batch: b"initial",
        "_center_single_node": lambda _np, _payload, _gm: b"centered",
        "_require_initial_source_payload": lambda *_args, **_kwargs: None,
        "_validate_source_locator": source_locator,
        "_validate_annual_locator": annual_locator,
        "_extract_v8_annual": lambda *_args, **_kwargs: object(),
        "_decode_annual_locator": lambda *_args, **_kwargs: object(),
        "_extract_v4_oracle_annual": lambda *_args, **_kwargs: object(),
        "_pread_all": lambda *_args, **_kwargs: b"centered",
        "_pre_adjudication_diagnostic": lambda *_args, **_kwargs: {
            "h1024_triggered": False,
        },
        "_source_invariant_summary": lambda *_args, **_kwargs: {"stub": True},
        "_hash_fd_range": lambda _fd, _offset, _length: (
            source_lanes[_offset]["lane_payload_sha256"]
        ),
        "_crossbind_source_annual": lambda **_kwargs: None,
        "_validate_ias_receipt": validator,
        "_finalize_epoch_diagnostic": finalizer,
    }
    with mock.patch.dict(M._BASE_NAMESPACE, replacements):
        rows = M._replay_evidence(
            v8=v8, v8_result={}, v8_source_fd=-1, v4_arrays_fd=-1,
            source_fd=-1, source_bytes=2, annual_fd=-1, annual_bytes=3,
            skeleton_batches=[skeleton], np=np,
        )
    return order, rows


class InheritedV1ScientificContractTests(V1_TESTS.SourceConvergenceDiagnosticTests):
    def test_no_public_tracer_call_or_tracer_state_construction(self) -> None:
        self.assertNotIn("advance_cuda_massless_tracers", M._replay_evidence.__code__.co_names)
        self.assertNotIn("CUDAMasslessTracerState", M._execute_worker.__code__.co_names)
        self.assertTrue(callable(M._build_source_lane))

    def test_source_annual_crossbinding_and_tamper(self) -> None:
        old_horizon = M._BASE_NAMESPACE["HORIZON_YEARS"]
        old_count = M._BASE_NAMESPACE["CHECKPOINT_COUNT"]
        M._BASE_NAMESPACE["HORIZON_YEARS"] = 2
        M._BASE_NAMESPACE["CHECKPOINT_COUNT"] = 3
        try:
            super().test_source_annual_crossbinding_and_tamper()
        finally:
            M._BASE_NAMESPACE["HORIZON_YEARS"] = old_horizon
            M._BASE_NAMESPACE["CHECKPOINT_COUNT"] = old_count


class SourceConvergenceDiagnosticV2SuccessorTests(unittest.TestCase):
    def test_public_no_arg_entrypoints_and_cli_dispatch_target_v2_root(self) -> None:
        functions = {
            "prepare": M.prepare,
            "validate_prepared": M.validate_prepared,
            "validate_output": M.validate_output,
            "execute": M.execute,
        }
        for name, function in functions.items():
            self.assertEqual(function.__defaults__, (M.DEFAULT_OUTPUT_ROOT,), name)
            self.assertIs(M.main.__globals__[name], function)
        self.assertEqual(
            M._parser().parse_args(["prepare"]).output_root,
            M.DEFAULT_OUTPUT_ROOT,
        )
        self.assertNotEqual(M.DEFAULT_OUTPUT_ROOT, M.V1_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            observed: list[tuple[str, Path]] = []

            def recorder(name: str):
                def call(root: Path) -> dict:
                    observed.append((name, root))
                    return {"temporary_context": temporary}
                return call

            with (
                mock.patch.object(M, "_V1_PREPARE", recorder("prepare")),
                mock.patch.object(
                    M, "_V1_VALIDATE_PREPARED", recorder("validate_prepared"),
                ),
                mock.patch.object(
                    M, "_V1_VALIDATE_OUTPUT", recorder("validate_output"),
                ),
                mock.patch.object(M, "_V1_EXECUTE", recorder("execute")),
            ):
                M.prepare()
                M.validate_prepared()
                M.validate_output()
                M.execute()
            self.assertEqual(
                observed,
                [(name, M.DEFAULT_OUTPUT_ROOT) for name in functions],
            )
            self.assertTrue(all(root != M.V1_ROOT for _name, root in observed))

    def test_v1_bytes_git_and_protocol_only_failure_are_bound(self) -> None:
        self.assertEqual(
            hashlib.sha256(M.V1_SOURCE_PATH.read_bytes()).hexdigest(),
            M.V1_SOURCE_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(M.V1_TEST_PATH.read_bytes()).hexdigest(),
            M.V1_TEST_SHA256,
        )
        self.assertEqual(
            hashlib.sha256(M.V1_PROTOCOL_PATH.read_bytes()).hexdigest(),
            M.V1_PROTOCOL_SHA256,
        )
        self.assertEqual(
            sorted(item.name for item in M.V1_ROOT.iterdir()), ["PROTOCOL.json"],
        )
        protocol = M.build_protocol()
        predecessor = protocol["predecessor_v1"]
        self.assertEqual(
            predecessor["git_custody"],
            {
                "branch": "research/tno-stage15-resolved11-source-convergence-v1-preregistered-20261010",
                "preregistration_commit": "9da26171a080062bda5cdd064d723ef4df20ba9d",
                "annotated_tag": "tno-stage15-resolved11-source-convergence-v1-preregistered-20261010",
                "tag_object": "9982af3245ba94de22ad1e161f5f3003e1345c72",
                "tag_peeled_commit": "9da26171a080062bda5cdd064d723ef4df20ba9d",
                "external_git_custody_not_reconstructed_by_this_program": True,
            },
        )
        failure = predecessor["failed_attempt"]
        self.assertEqual(
            (
                failure["mandatory_jx_integrations_completed"],
                failure["conditional_h1024_integrations_completed"],
                failure["tight_ias15_replicates_completed"],
            ),
            (24, 11, 12),
        )
        self.assertEqual(
            failure["failure_message"],
            "zip() argument 2 is shorter than argument 1",
        )
        self.assertTrue(failure["all_integrations_completed"])
        self.assertEqual(failure["published_result_artifact_count"], 0)
        self.assertFalse(failure["result_artifacts_published"])
        self.assertFalse(failure["science_result"])

    def test_all_local_addresses_are_v2_and_v1_input_is_explicit(self) -> None:
        for value in (
            M.PROTOCOL_SCHEMA, M.RESULT_SCHEMA, M.RUNTIME_SCHEMA,
            M.SOURCE_SCHEMA, M.ANNUAL_SCHEMA, M.VALIDATION_SCHEMA,
        ):
            self.assertTrue(value.endswith(".v2"), value)
        for value in (
            M.PROTOCOL_DOMAIN, M.RESULT_DOMAIN,
            M.RUNTIME_DOMAIN, M.VALIDATION_DOMAIN,
        ):
            self.assertTrue(value.endswith(b".v2\0"), value)
        self.assertTrue(M.SCRIPT_PATH.name.endswith("_v2.py"))
        self.assertTrue(M.TEST_PATH.name.endswith("_v2.py"))
        self.assertTrue(M.DEFAULT_OUTPUT_ROOT.name.endswith("_v2"))
        paths = {row["relative_path"] for row in M.build_protocol()["inputs"]["files"]}
        self.assertIn(M.V1_SOURCE_PATH.relative_to(M.PROJECT_ROOT).as_posix(), paths)
        self.assertIn(M.V1_TEST_PATH.relative_to(M.PROJECT_ROOT).as_posix(), paths)
        self.assertIn(M.V1_PROTOCOL_PATH.relative_to(M.PROJECT_ROOT).as_posix(), paths)

    def test_production_shaped_1001_epoch_receipt_reaches_downstream_replay_logic(self) -> None:
        annual = list(range(M.CHECKPOINT_COUNT))
        receipt = ias_receipt(annual)
        order, rows = run_stubbed_replay(receipt)
        self.assertEqual(
            order, ["ias_receipt_validated", "diagnostic_finalized"],
        )
        self.assertEqual(rows[0]["diagnostic"]["disposition"], "PASS")

    def test_duplicate_and_nonmonotonic_annual_counts_fail_closed(self) -> None:
        cases: dict[str, list[int]] = {}
        duplicate = list(range(M.CHECKPOINT_COUNT))
        duplicate[500] = duplicate[499]
        cases["duplicate"] = duplicate
        descending = list(range(M.CHECKPOINT_COUNT))
        descending[500] = descending[499] - 1
        cases["nonmonotonic"] = descending
        for label, annual in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(M.SourceConvergenceDiagnosticError):
                    run_stubbed_replay(ias_receipt(annual))

    def test_v2_has_no_unequal_adjacent_strict_zip(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        self.assertIsNone(
            re.search(
                r"zip\s*\(\s*annual\s*,\s*annual\s*\[\s*1\s*:\s*\]\s*,\s*strict\s*=\s*True",
                text,
            ),
        )
        self.assertNotIn("zip", M._validate_ias_receipt.__code__.co_names)
        self.assertIn("range", M._validate_ias_receipt.__code__.co_names)


if __name__ == "__main__":
    unittest.main()
