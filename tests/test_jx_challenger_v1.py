from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "jx_challenger_v1.py"
SPEC = importlib.util.spec_from_file_location("jx_challenger_v1_under_test", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load Challenger V1")
CHALLENGER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = CHALLENGER
SPEC.loader.exec_module(CHALLENGER)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _all_keys(value: object) -> set[str]:
    keys: set[str] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            keys.update(item)
            stack.extend(item.values())
        elif type(item) is list:
            stack.extend(item)
    return keys


class ChallengerV1StaticTests(unittest.TestCase):
    def test_exact_smoke_study_and_command_roster(self) -> None:
        self.assertEqual(
            tuple(study.study_id for study in CHALLENGER._STUDIES),
            (
                "leapfrog_binary",
                "whfast_weak_three_body",
                "hybrid_close_scatter",
            ),
        )
        self.assertEqual(
            tuple(study.child_arguments for study in CHALLENGER._STUDIES),
            (
                ("--periods", "1", "--samples-per-period", "16"),
                ("--profile", "smoke"),
                ("--profile", "smoke"),
            ),
        )
        for study in CHALLENGER._STUDIES:
            command = CHALLENGER._child_command(ROOT, study, "disabled")
            self.assertEqual(command[:4], (sys.executable, "-I", "-B", "-c"))
            self.assertEqual(command[4], CHALLENGER._BOOTSTRAP)
            self.assertEqual(Path(command[5]), (ROOT / "src").resolve())
            self.assertEqual(Path(command[6]), (ROOT / study.script_relative_path).resolve())
            self.assertEqual(command[-2:], ("--rebound-mode", "disabled"))
            self.assertNotIn("--output", command)

    def test_parser_requires_output_and_defaults_to_disabled(self) -> None:
        with self.assertRaises(SystemExit):
            CHALLENGER._parser().parse_args([])
        parsed = CHALLENGER._parser().parse_args(["--output-dir", "/tmp/example"])
        self.assertEqual(parsed.rebound_mode, "disabled")
        self.assertEqual(parsed.output_dir, Path("/tmp/example"))

    def test_execution_failure_is_not_relabelled_as_cli_usage(self) -> None:
        stderr = io.StringIO()
        with (
            mock.patch.object(
                CHALLENGER,
                "run_challenger",
                side_effect=CHALLENGER.ChallengerError("dependency unavailable"),
            ),
            mock.patch("sys.stderr", stderr),
        ):
            status = CHALLENGER.main(["--output-dir", "/tmp/not-created"])
        self.assertEqual(status, 1)
        self.assertIn("execution failed: dependency unavailable", stderr.getvalue())

    def test_parent_import_is_standard_library_only(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import numpy", source)
        self.assertNotIn("import rebound", source)
        self.assertNotIn("from jxplanetx", source)
        self.assertNotIn("integrate_", source)

    def test_strict_json_parser_rejects_ambiguous_or_unbounded_inputs(self) -> None:
        bad = (
            b'{"x":1,"x":2}',
            b'{"x":NaN}',
            b'{"x":Infinity}',
            b'{"x":1e999}',
            b'{"x":1} trailing',
            b'\xff',
            b'[]',
        )
        for payload in bad:
            with self.subTest(payload=payload[:30]):
                with self.assertRaises(CHALLENGER.ChallengerError):
                    CHALLENGER._parse_json_document(payload)
        with mock.patch.object(CHALLENGER, "MAX_JSON_DEPTH", 2):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER._parse_json_document(b'{"a":{"b":{"c":1}}}')
        with mock.patch.object(CHALLENGER, "MAX_JSON_NODES", 3):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER._parse_json_document(b'{"a":[1,2,3]}')
        with mock.patch.object(CHALLENGER, "MAX_JSON_STRING_BYTES", 3):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER._parse_json_document(b'{"abcd":1}')
        huge_integer = ("1" + "0" * 1234).encode("ascii")
        with self.assertRaises(CHALLENGER.ChallengerError):
            CHALLENGER._parse_json_document(b'{"x":' + huge_integer + b"}")

    def test_canonical_digest_preserves_signed_zero_and_ignores_mapping_order(self) -> None:
        positive = {"a": 0.0, "b": 1}
        negative = {"a": -0.0, "b": 1}
        reordered = {"b": 1, "a": 0.0}
        self.assertNotEqual(
            CHALLENGER._domain_sha256("test", positive),
            CHALLENGER._domain_sha256("test", negative),
        )
        self.assertEqual(
            CHALLENGER._domain_sha256("test", positive),
            CHALLENGER._domain_sha256("test", reordered),
        )
        self.assertNotEqual(
            CHALLENGER._domain_sha256("test", 1.0),
            CHALLENGER._domain_sha256(
                "test", {"binary64_hex": "0x1.0000000000000p+0"}
            ),
        )

    def test_child_process_error_timeout_and_output_caps_fail_closed(self) -> None:
        with self.assertRaises(CHALLENGER.ChallengerError):
            CHALLENGER._run_bounded_child(
                (sys.executable, "-I", "-B", "-c", "raise SystemExit(7)"),
                repository_root=ROOT,
                timeout_seconds=5.0,
            )
        with self.assertRaises(CHALLENGER.ChallengerError):
            CHALLENGER._run_bounded_child(
                (sys.executable, "-I", "-B", "-c", "import time;time.sleep(5)"),
                repository_root=ROOT,
                timeout_seconds=0.05,
            )
        with mock.patch.object(CHALLENGER, "MAX_CHILD_STDOUT_BYTES", 3):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER._run_bounded_child(
                    (sys.executable, "-I", "-B", "-c", "print('abcd')"),
                    repository_root=ROOT,
                    timeout_seconds=5.0,
                )
        with mock.patch.object(CHALLENGER, "MAX_CHILD_STDERR_BYTES", 3):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER._run_bounded_child(
                    (
                        sys.executable,
                        "-I",
                        "-B",
                        "-c",
                        "import sys;sys.stderr.write('abcd')",
                    ),
                    repository_root=ROOT,
                    timeout_seconds=5.0,
                )

    def test_preexisting_output_types_are_rejected_without_change(self) -> None:
        with tempfile.TemporaryDirectory(prefix="jx-challenger-targets-") as temporary:
            parent = Path(temporary)
            file_target = parent / "file"
            file_target.write_bytes(b"keep")
            directory_target = parent / "directory"
            directory_target.mkdir()
            link_target = parent / "dangling"
            link_target.symlink_to(parent / "missing")
            for target in (file_target, directory_target, link_target):
                with self.subTest(target=target.name):
                    with self.assertRaises(CHALLENGER.ChallengerError):
                        CHALLENGER._resolve_new_output_directory(target)
            self.assertEqual(file_target.read_bytes(), b"keep")
            self.assertTrue(directory_target.is_dir())
            self.assertTrue(os.path.lexists(link_target))


class ChallengerV1EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="jx-challenger-v1-")
        cls.parent = Path(cls.temporary.name)
        cls.first = cls.parent / "first"
        cls.second = cls.parent / "second"
        cls.first_report = CHALLENGER.run_challenger(cls.first)
        cls.second_report = CHALLENGER.run_challenger(cls.second)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_bundle_roster_checksums_and_round_trip(self) -> None:
        expected = {
            "SHA256SUMS",
            "challenger_v1.json",
            "hybrid_smoke.json",
            "leapfrog_smoke.json",
            "whfast_smoke.json",
        }
        self.assertEqual({path.name for path in self.first.iterdir()}, expected)
        self.assertEqual(CHALLENGER.verify_bundle_directory(self.first), self.first_report)
        lines = (self.first / "SHA256SUMS").read_text(encoding="ascii").splitlines()
        self.assertEqual(len(lines), 4)
        for line in lines:
            digest, filename = line.split("  ")
            self.assertEqual(digest, _sha((self.first / filename).read_bytes()))

    def test_scientific_identity_and_exact_disabled_rosters(self) -> None:
        report = self.first_report
        self.assertEqual(report["schema"], "jx.challenger.smoke-bundle.v1")
        self.assertEqual(
            report["benchmark_id"], "jx.challenger.solver-portfolio.smoke.v1"
        )
        self.assertEqual(report["profile"]["rebound_mode"], "disabled")
        self.assertEqual(
            report["jx_engine_source_tree_sha256"],
            "974e31567db7f0e717f76944aedb0c027632f9ab2f4f9f4cbf9b66866d114269",
        )
        self.assertEqual(
            [study["study_id"] for study in report["studies"]],
            [study.study_id for study in CHALLENGER._STUDIES],
        )
        expected_lanes = (
            {
                "jx_kdk",
                "jx_rkf78_adaptive_reference",
                "rebound_leapfrog",
            },
            {"jx_wh_long_p64", "jx_wh_short_p128", "jx_wh_short_p64"},
            {"jx_hybrid_p256"},
        )
        for study, lanes in zip(report["studies"], expected_lanes):
            self.assertEqual(set(study["lanes"]), lanes)
            self.assertFalse(study["external_status"]["external_lanes_executed"])
            self.assertEqual(study["external_lane_ids"], [])

    def test_path_independent_scientific_digests_are_locked(self) -> None:
        studies = {study["study_id"]: study for study in self.first_report["studies"]}
        leapfrog = studies["leapfrog_binary"]["lanes"]
        self.assertEqual(
            leapfrog["jx_kdk"]["native_digests"][0]["sha256"],
            "b5d2bdd408a420fffb80c1d52160605857e9d3e38f93beae18be40f78f99d382",
        )
        self.assertEqual(
            leapfrog["jx_rkf78_adaptive_reference"]["native_digests"][0]["sha256"],
            "21d5a289c6a3cd4a3e2925f3b0bd78486dbe209bef6da6ee153663deb4b50de0",
        )
        wh_spec = CHALLENGER._STUDIES[1]
        wh_raw = CHALLENGER._parse_json_document(
            (self.first / wh_spec.report_filename).read_bytes()
        )
        self.assertEqual(
            {
                lane_id: (
                    lane["accounting"]["result_content_sha256"],
                    lane["accounting"]["schedule_content_sha256"],
                )
                for lane_id, lane in wh_raw["lanes"].items()
            },
            {
                "jx_wh_long_p64": (
                    "5067e7803b1840b168ea353c5482f3de3c86d53167601b4f7e5aa37e8187bf8a",
                    "78d5ef6669849f673abdd6031112181f54f2a95fae1bc0e1939ea3ef869d9e08",
                ),
                "jx_wh_short_p128": (
                    "6d61e7fcb24badc56dc0652ba8063360997e3a1566c65a579ad8462d169f3189",
                    "23b64746a7c82c554fa4ae2e040efd8f348cd652eaf2676309a6137eb89ef61d",
                ),
                "jx_wh_short_p64": (
                    "102652f0d2151b1434c63c3b05b529e0aecc39ec11b2a22289c4e1041baacc72",
                    "8dd2dbaa0031e9623ab859945e674f19799bad3f9adc17bac5340f60fe9277c9",
                ),
            },
        )
        hybrid_spec = CHALLENGER._STUDIES[2]
        hybrid_raw = CHALLENGER._parse_json_document(
            (self.first / hybrid_spec.report_filename).read_bytes()
        )["headline_lanes"]["jx_hybrid_p256"]
        self.assertEqual(
            {
                key: hybrid_raw[key]
                for key in (
                    "headline_final_state_raw_sha256",
                    "headline_trajectory_raw_sha256",
                    "declared_epochs_content_sha256",
                    "observed_epochs_content_sha256",
                    "positions_content_sha256",
                    "velocities_content_sha256",
                )
            },
            {
                "headline_final_state_raw_sha256": "b1689a0dc1f0b4e128d00a3c10457976aabdc6c562b07620be2847fd72a06a20",
                "headline_trajectory_raw_sha256": "96457e0fccdfcc0f636d08a3d4c1a7484fd83cd75bee4c96b3b9c4876b06c798",
                "declared_epochs_content_sha256": "706cddb8be27397f5b0ccf42eb34220a97779994a824c666ea5261573d250685",
                "observed_epochs_content_sha256": "086b395f8a5dd0e99f354bc99c89d88878b66de484395e843b60ddbebe0ca22c",
                "positions_content_sha256": "3947147542b24bcaee1ab71d5346e41c325a2dd1e01e0b2adbba40ccefb007a9",
                "velocities_content_sha256": "859a9c5470d55f327a6359054ee63e6e4f60facde103adcb7f959233a0f59ed6",
            },
        )
        second_studies = {
            study["study_id"]: study for study in self.second_report["studies"]
        }
        self.assertEqual(
            studies["whfast_weak_three_body"]["lanes"],
            second_studies["whfast_weak_three_body"]["lanes"],
        )
        self.assertEqual(
            studies["hybrid_close_scatter"],
            second_studies["hybrid_close_scatter"],
        )

    def test_phase_observables_are_not_homogenized_or_invented(self) -> None:
        studies = {study["study_id"]: study for study in self.first_report["studies"]}
        leapfrog_phase = studies["leapfrog_binary"]["lanes"]["jx_kdk"]["phase_metric"]
        self.assertEqual(
            leapfrog_phase["observable"],
            "RELATIVE_BINARY_ORBIT_ANGLE_IN_XY_PLANE",
        )
        self.assertEqual(
            leapfrog_phase["sign_convention"],
            "CANDIDATE_MINUS_ANALYTIC_REFERENCE_UNWRAPPED",
        )
        for lane in studies["whfast_weak_three_body"]["lanes"].values():
            self.assertIsNone(lane["phase_metric"])
        self.assertIsNone(
            studies["hybrid_close_scatter"]["lanes"]["jx_hybrid_p256"]["phase_metric"]
        )

    def test_exact_external_lane_shapes_are_supported_without_work_relabeling(self) -> None:
        leap_spec, wh_spec, hybrid_spec = CHALLENGER._STUDIES
        leap_report = CHALLENGER._parse_json_document(
            (self.first / leap_spec.report_filename).read_bytes()
        )
        leap_external = copy.deepcopy(leap_report["lanes"]["jx_kdk"])
        leap_external["engine_id"] = "rebound_leapfrog"
        leap_external["accounting"] = {
            "completed_integer_steps": 256,
            "force_evaluations_exposed": False,
        }
        leap_summary = CHALLENGER._leapfrog_lane(
            "rebound_leapfrog", leap_external, executed=True
        )
        self.assertEqual(
            leap_summary["method_id"], "rebound.integrator.leapfrog.5.1.1"
        )
        self.assertIsNone(leap_summary["work_accounting"]["force_evaluations"])

        wh_report = CHALLENGER._parse_json_document(
            (self.first / wh_spec.report_filename).read_bytes()
        )
        wh_external = copy.deepcopy(wh_report["lanes"]["jx_wh_short_p64"])
        wh_external["engine_id"] = "rebound_whfast_short_p64"
        wh_external["accounting"] = {"completed_integer_steps": 64}
        wh_summary = CHALLENGER._whfast_lane(
            "rebound_whfast_short_p64", wh_external
        )
        self.assertEqual(
            wh_summary["method_id"], "rebound.integrator.whfast.5.1.1"
        )
        self.assertIsNone(
            wh_summary["work_accounting"]["primary_force_evaluations"]
        )

        hybrid_report = CHALLENGER._parse_json_document(
            (self.first / hybrid_spec.report_filename).read_bytes()
        )
        hybrid_external = copy.deepcopy(
            hybrid_report["headline_lanes"]["jx_hybrid_p256"]
        )
        hybrid_external["engine_id"] = "rebound_mercurius_p256"
        hybrid_external["method_id"] = "rebound.integrator.mercurius.5.1.1"
        hybrid_external["accounting"] = {
            "steps_call_count": 128,
            "steps_done_delta": 128,
        }
        hybrid_summary = CHALLENGER._hybrid_lane(
            "rebound_mercurius_p256", hybrid_external
        )
        self.assertIsNone(hybrid_summary["work_accounting"]["mode_counts"])
        self.assertFalse(
            hybrid_summary["work_accounting"]["internal_force_evaluations_exposed"]
        )

        for spec, raw_report in (
            (wh_spec, wh_report),
            (hybrid_spec, hybrid_report),
        ):
            loaded = copy.deepcopy(raw_report)
            loaded["runtime_provenance"]["external_status"] = {
                "requested_mode": "required",
                "external_lanes_executed": True,
                "reason": "EXACT_REBOUND_5_1_1_LOADED",
                "required_version": "5.1.1",
                "authority_authorized": False,
            }
            status = CHALLENGER._external_status(loaded, spec, "required")
            self.assertTrue(status["external_lanes_executed"])
            self.assertEqual(status["reason"], "EXECUTED_EXACT_REBOUND_5_1_1")

    def test_synthetic_external_phase_and_reference_crosslinks_are_exact(self) -> None:
        summaries = {
            item["study_id"]: item for item in self.first_report["studies"]
        }
        support = summaries["leapfrog_binary"]["child_source"]

        wh_spec = CHALLENGER._STUDIES[1]
        wh_report = CHALLENGER._parse_json_document(
            (self.first / wh_spec.report_filename).read_bytes()
        )
        wh_report["runtime_provenance"]["external_status"] = {
            "requested_mode": "required",
            "external_lanes_executed": True,
            "reason": "EXACT_REBOUND_5_1_1_LOADED",
            "required_version": "5.1.1",
            "authority_authorized": False,
        }
        wh_reference_for = dict(CHALLENGER._WHFAST_PHASE_REFERENCES)
        for lane_id, reference_id in wh_reference_for.items():
            if lane_id not in wh_report["lanes"]:
                source_id = lane_id.replace("rebound_whfast_", "jx_wh_")
                wh_report["lanes"][lane_id] = copy.deepcopy(
                    wh_report["lanes"][source_id]
                )
                wh_report["lanes"][lane_id]["engine_id"] = lane_id
                wh_report["lanes"][lane_id]["accounting"] = {
                    "completed_integer_steps": (
                        128 if lane_id.endswith("p128") else 64
                    )
                }
            lane = wh_report["lanes"][lane_id]
            lane["observed_epoch_phase_proxy"] = {
                "definition": (
                    "WRAPPED_PLANAR_POLAR_ANGLE_OF_BODY_MINUS_PRIMARY_"
                    "CANDIDATE_MINUS_IAS15_REFERENCE"
                ),
                "bodies": {
                    body: {
                        "max_abs_radians": 0.0,
                        "rms_radians": 0.0,
                        "final_wrapped_radians": 0.0,
                    }
                    for body in ("INNER", "OUTER")
                },
                "max_abs_radians": 0.0,
                "rms_radians": 0.0,
            }
            lane["observed_epoch_state_error"] = {
                "position": {"vector_l2_max": 0.0},
                "velocity": {"vector_l2_max": 0.0},
                "reference_engine_id": reference_id,
            }
        wh_reference_ids = {
            "rebound_ias15_short_declared_reference",
            "rebound_ias15_long_declared_reference",
            "rebound_ias15_short_whfast_p64_observed_reference",
            "rebound_ias15_short_whfast_p128_observed_reference",
            "rebound_ias15_long_whfast_p64_observed_reference",
        }
        wh_report["ias15_numerical_references"] = {
            reference_id: {"lane_content_integrity": {"sha256": "a" * 64}}
            for reference_id in wh_reference_ids
        }
        wh_summary = CHALLENGER._build_study_summary(
            wh_report,
            wh_spec,
            "required",
            summaries[wh_spec.study_id]["child_source"],
            support,
        )
        self.assertEqual(
            set(wh_summary["numerical_reference_content_sha256"]),
            wh_reference_ids,
        )
        self.assertEqual(
            {
                lane_id: lane["phase_metric"]["reference_id"]
                for lane_id, lane in wh_summary["lanes"].items()
            },
            wh_reference_for,
        )

        hybrid_spec = CHALLENGER._STUDIES[2]
        hybrid_report = CHALLENGER._parse_json_document(
            (self.first / hybrid_spec.report_filename).read_bytes()
        )
        hybrid_report["runtime_provenance"]["external_status"] = {
            "requested_mode": "required",
            "external_lanes_executed": True,
            "reason": "EXACT_REBOUND_5_1_1_LOADED",
            "required_version": "5.1.1",
            "authority_authorized": False,
        }
        hybrid_accuracy = {
            "inner_outer_relative_vector_phase_proxy": {
                "proxy": (
                    "INNER_TO_OUTER_RELATIVE_POSITION_VECTOR_ANGLE_IN_XY_PLANE"
                ),
                "sign_convention": (
                    "ATAN2_CROSS_CANDIDATE_REFERENCE_REFERENCE_MINUS_CANDIDATE"
                ),
                "maximum_abs_radians": 0.0,
                "rms_radians": 0.0,
                "final_signed_radians": 0.0,
            },
            "position": {"vector_l2_max": 0.0},
            "velocity": {"vector_l2_max": 0.0},
        }
        hybrid_jx = hybrid_report["headline_lanes"]["jx_hybrid_p256"]
        hybrid_jx["headline_accuracy_against_numerical_reference"] = copy.deepcopy(
            hybrid_accuracy
        )
        for lane_id, method_id in (
            ("rebound_mercurius_p256", "rebound.integrator.mercurius.5.1.1"),
            ("rebound_trace_p256", "rebound.integrator.trace.5.1.1"),
        ):
            lane = copy.deepcopy(hybrid_jx)
            lane["engine_id"] = lane_id
            lane["method_id"] = method_id
            lane["accounting"] = {
                "steps_call_count": 128,
                "steps_done_delta": 128,
            }
            hybrid_report["headline_lanes"][lane_id] = lane
        hybrid_reference_ids = set(CHALLENGER._HYBRID_PHASE_REFERENCES.values()) | {
            "rebound_ias15_jx_exact_labels_dt_p1024_sensitivity"
        }
        hybrid_report["numerical_references"] = {
            reference_id: {"lane_content_sha256": "b" * 64}
            for reference_id in hybrid_reference_ids
        }
        hybrid_summary = CHALLENGER._build_study_summary(
            hybrid_report,
            hybrid_spec,
            "required",
            summaries[hybrid_spec.study_id]["child_source"],
            support,
        )
        self.assertEqual(
            set(hybrid_summary["numerical_reference_content_sha256"]),
            hybrid_reference_ids,
        )
        self.assertEqual(
            {
                lane_id: lane["phase_metric"]["reference_id"]
                for lane_id, lane in hybrid_summary["lanes"].items()
            },
            CHALLENGER._HYBRID_PHASE_REFERENCES,
        )

    def test_semantic_digest_replays_while_raw_timing_is_diagnostic(self) -> None:
        first = self.first_report
        second = self.second_report
        self.assertEqual(
            first["content_integrity"]["semantic_content_sha256"],
            second["content_integrity"]["semantic_content_sha256"],
        )
        changed = copy.deepcopy(first)
        changed["execution_diagnostics"]["children"][0]["wall_seconds"] += 1.0
        self.assertEqual(
            CHALLENGER._domain_sha256(CHALLENGER._SEMANTIC_DOMAIN, CHALLENGER._semantic_core(first)),
            CHALLENGER._domain_sha256(CHALLENGER._SEMANTIC_DOMAIN, CHALLENGER._semantic_core(changed)),
        )
        changed_science = copy.deepcopy(first)
        changed_science["studies"][0]["lanes"]["jx_kdk"]["relative_total_energy_max"] += 1e-12
        self.assertNotEqual(
            CHALLENGER._domain_sha256(CHALLENGER._SEMANTIC_DOMAIN, CHALLENGER._semantic_core(first)),
            CHALLENGER._domain_sha256(
                CHALLENGER._SEMANTIC_DOMAIN,
                CHALLENGER._semantic_core(changed_science),
            ),
        )

    def test_claims_remain_false_and_forbidden_ranking_keys_are_absent(self) -> None:
        self.assertEqual(self.first_report["claim_controls"], CHALLENGER._AGGREGATE_CLAIM_CONTROLS)
        self.assertTrue(all(value is False for value in self.first_report["claim_controls"].values()))
        lowered = {key.lower() for key in _all_keys(self.first_report)}
        for fragment in ("winner", "defeated", "speedup"):
            self.assertFalse(any(fragment in key for key in lowered))

    def test_mutated_child_or_aggregate_seal_is_rejected(self) -> None:
        copied = self.parent / "mutated"
        shutil_copy = __import__("shutil").copytree
        shutil_copy(self.first, copied)
        child_path = copied / "leapfrog_smoke.json"
        child = json.loads(child_path.read_text(encoding="utf-8"))
        child["schema"] = "forged"
        child_path.write_text(json.dumps(child, sort_keys=True) + "\n", encoding="utf-8")
        with self.assertRaises(CHALLENGER.ChallengerError):
            CHALLENGER.verify_bundle_directory(copied)
        stale = copy.deepcopy(self.first_report)
        stale["studies"][0]["lanes"]["jx_kdk"]["relative_total_energy_max"] += 1e-12
        with self.assertRaises(CHALLENGER.ChallengerError):
            CHALLENGER.validate_challenger_report(stale)

    def test_self_consistent_frozen_source_relabeling_is_rejected(self) -> None:
        changed_engine = copy.deepcopy(self.first_report)
        changed_engine["jx_engine_source_tree_sha256"] = "f" * 64
        del changed_engine["content_integrity"]
        CHALLENGER._attach_integrity(changed_engine)
        with self.assertRaises(CHALLENGER.ChallengerError):
            CHALLENGER.validate_challenger_report(changed_engine)

        for field in ("child_source", "transitive_source_identities"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.first_report)
                study = changed["studies"][1]
                if field == "child_source":
                    study[field]["sha256"] = "e" * 64
                else:
                    study[field][0]["sha256"] = "e" * 64
                del study["study_summary_semantic_content_sha256"]
                study["study_summary_semantic_content_sha256"] = (
                    CHALLENGER._domain_sha256(CHALLENGER._STUDY_DOMAIN, study)
                )
                del changed["content_integrity"]
                CHALLENGER._attach_integrity(changed)
                with self.assertRaises(CHALLENGER.ChallengerError):
                    CHALLENGER.validate_challenger_report(changed)

    def test_self_consistent_lane_meaning_relabeling_is_rejected(self) -> None:
        def reseal(report: dict[str, object], lane: dict[str, object]) -> None:
            study = report["studies"][0]
            del lane["summary_sha256"]
            lane["summary_sha256"] = CHALLENGER._domain_sha256(
                CHALLENGER._LANE_DOMAIN, lane
            )
            del study["study_summary_semantic_content_sha256"]
            study["study_summary_semantic_content_sha256"] = (
                CHALLENGER._domain_sha256(CHALLENGER._STUDY_DOMAIN, study)
            )
            del report["content_integrity"]
            CHALLENGER._attach_integrity(report)

        mutations = (
            ("role", "OPTIONAL_EXTERNAL_COMPARATOR"),
            ("method_id", "integrator.forged.v1"),
            ("native_scope", "forged_scope"),
            ("force_evaluations", 1),
            (
                "phase_sign",
                "REFERENCE_MINUS_CANDIDATE_WRONGLY_RELABELLED",
            ),
        )
        for field, replacement in mutations:
            with self.subTest(field=field):
                changed = copy.deepcopy(self.first_report)
                lane = changed["studies"][0]["lanes"]["jx_kdk"]
                if field == "phase_sign":
                    lane["phase_metric"]["sign_convention"] = replacement
                elif field == "native_scope":
                    lane["native_digests"][0]["scope"] = replacement
                elif field == "force_evaluations":
                    lane["work_accounting"]["force_evaluations"] = replacement
                else:
                    lane[field] = replacement
                if field == "role":
                    changed["studies"][0]["jx_lane_ids"].remove("jx_kdk")
                    changed["studies"][0]["external_lane_ids"].append("jx_kdk")
                    changed["studies"][0]["external_lane_ids"].sort()
                reseal(changed, lane)
                with self.assertRaises(CHALLENGER.ChallengerError):
                    CHALLENGER.validate_challenger_report(changed)

    def test_source_change_or_child_failure_never_publishes(self) -> None:
        sources = CHALLENGER._source_roster(ROOT)
        changed = list(copy.deepcopy(sources))
        changed[0]["sha256"] = "f" * 64
        raw_reports = [
            (self.first / study.report_filename).read_bytes() for study in CHALLENGER._STUDIES
        ]
        returns = [(payload, b"", 0, 0.01) for payload in raw_reports]
        output = self.parent / "source-change"
        with (
            mock.patch.object(CHALLENGER, "_source_roster", side_effect=[sources, tuple(changed)]),
            mock.patch.object(CHALLENGER, "_run_bounded_child", side_effect=returns),
        ):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER.run_challenger(output)
        self.assertFalse(os.path.lexists(output))
        failed = self.parent / "child-failure"
        with mock.patch.object(
            CHALLENGER,
            "_run_bounded_child",
            side_effect=CHALLENGER.ChallengerError("injected child failure"),
        ):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER.run_challenger(failed)
        self.assertFalse(os.path.lexists(failed))

    def test_stage_write_failure_is_cleaned_without_partial_publication(self) -> None:
        report = self.first_report
        payloads = {
            study.report_filename: (self.first / study.report_filename).read_bytes()
            for study in CHALLENGER._STUDIES
        }
        output = self.parent / "write-failure"
        original = CHALLENGER._write_file
        calls = 0

        def fail_second(path: Path, payload: bytes) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise CHALLENGER.ChallengerError("injected write failure")
            original(path, payload)

        with mock.patch.object(CHALLENGER, "_write_file", side_effect=fail_second):
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER._publish_bundle(output, payloads, report)
        self.assertFalse(os.path.lexists(output))
        self.assertEqual(
            [path for path in self.parent.iterdir() if path.name.startswith(".write-failure.stage-")],
            [],
        )

    def test_existing_output_is_rejected_before_any_child_runs(self) -> None:
        with mock.patch.object(CHALLENGER, "_run_bounded_child") as child:
            with self.assertRaises(CHALLENGER.ChallengerError):
                CHALLENGER.run_challenger(self.first)
        child.assert_not_called()


@unittest.skipUnless(
    os.environ.get("JX_RUN_CHALLENGER_REBOUND_5_1_1") == "1",
    "set JX_RUN_CHALLENGER_REBOUND_5_1_1=1 in an exact REBOUND 5.1.1 environment",
)
class ChallengerV1ExactReboundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(
            prefix="jx-challenger-rebound-v1-"
        )
        parent = Path(cls.temporary.name)
        cls.first_path = parent / "first"
        cls.second_path = parent / "second"
        cls.first = CHALLENGER.run_challenger(
            cls.first_path, rebound_mode="required"
        )
        cls.second = CHALLENGER.run_challenger(
            cls.second_path, rebound_mode="required"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_exact_rebound_lane_reference_and_replay_contract(self) -> None:
        self.assertEqual(
            CHALLENGER.verify_bundle_directory(self.first_path), self.first
        )
        self.assertEqual(
            CHALLENGER.verify_bundle_directory(self.second_path), self.second
        )
        self.assertEqual(
            self.first["content_integrity"]["semantic_content_sha256"],
            self.second["content_integrity"]["semantic_content_sha256"],
        )
        studies = {item["study_id"]: item for item in self.first["studies"]}
        self.assertTrue(
            all(
                item["external_status"]["external_lanes_executed"]
                for item in studies.values()
            )
        )
        self.assertEqual(
            set(studies["whfast_weak_three_body"]["lanes"]),
            set(CHALLENGER._LANE_CONTRACTS["whfast_weak_three_body"]),
        )
        self.assertEqual(
            set(studies["hybrid_close_scatter"]["lanes"]),
            set(CHALLENGER._LANE_CONTRACTS["hybrid_close_scatter"]),
        )
        self.assertEqual(
            {
                lane_id: lane["phase_metric"]["reference_id"]
                for lane_id, lane in studies["whfast_weak_three_body"][
                    "lanes"
                ].items()
            },
            CHALLENGER._WHFAST_PHASE_REFERENCES,
        )
        self.assertEqual(
            {
                lane_id: lane["phase_metric"]["reference_id"]
                for lane_id, lane in studies["hybrid_close_scatter"][
                    "lanes"
                ].items()
            },
            CHALLENGER._HYBRID_PHASE_REFERENCES,
        )
        self.assertEqual(
            set(
                studies["hybrid_close_scatter"][
                    "numerical_reference_content_sha256"
                ]
            ),
            set(CHALLENGER._HYBRID_PHASE_REFERENCES.values())
            | {"rebound_ias15_jx_exact_labels_dt_p1024_sensitivity"},
        )


if __name__ == "__main__":
    unittest.main()
