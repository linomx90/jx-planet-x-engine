#!/usr/bin/env python3
"""No-dynamics tests for the frozen JX-XP1 implementation package."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
import types
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
PYTHON = pathlib.Path(sys.executable)


def load(name: str, path: pathlib.Path):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    source = path.read_bytes()
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


runner = load("xp1_runner_under_test", ROOT / "run_exploratory.py")
verifier = load("xp1_verifier_under_test", ROOT / "verify_replay.py")


def registration_payload(locked_files: dict[str, str], nonclaim: str) -> dict:
    return {
        "schema": "jx-xp1-local-registration/v1",
        "experiment_id": "jx-xp1-public-synthetic-response-v1",
        "artifact_class": "LOCAL_CONTENT_HASH_REGISTRATION_ONLY",
        "registration_state": "LOCAL_CONTENT_HASH_LOCK_COMPLETE_BEFORE_ANY_XP1_NUMERICAL_OUTPUT",
        "recorded_at_utc": "2026-08-23T00:00:00Z",
        "timestamp_authority": "LOCAL_CONTENT_HASH_ONLY_NO_EXTERNAL_TIMESTAMP",
        "externally_timestamped": False,
        "scientific_evidence_artifact": False,
        "outcomes_generated": False,
        "execution_permissions": {
            "execution_a_authorized": True,
            "execution_b_authorized_only_after_verified_a": True,
            "local_cpu_only": True,
            "network_access_authorized": False,
            "gpu_execution_authorized": False,
            "observed_data_access_authorized": False,
            "survey_adapter_execution_authorized": False,
            "jx_o2_execution_or_g0_evidence_authorized": False,
            "planet_x_claim_authorized": False,
        },
        "locked_files": locked_files,
        "mandatory_nonclaim": nonclaim,
    }


def particle_row(logical_id: str, block: int, minimum_q: float, bound: bool = True) -> dict:
    return {
        "logical_id": logical_id,
        "block_index": block,
        "index_within_block": int(logical_id[-2:]),
        "minimum_sampled_q_AU": minimum_q,
        "first_sampled_q_below_35_time_year": 0.0 if minimum_q < 35.0 else None,
        "first_sampled_q_below_30_time_year": 0.0 if minimum_q < 30.0 else None,
        "final_a_AU": 200.0 if bound else -200.0,
        "final_e": 0.8 if bound else 1.2,
        "final_i_deg": 10.0,
        "final_q_AU": 40.0,
        "final_distance_AU": 200.0,
        "final_finite_and_bound": bound,
        "all_samples_finite_osculating_orbit": True,
    }


def fake_arm(arm_id: str, hit_count: int) -> dict:
    rows = []
    for block in range(4):
        for index in range(16):
            logical_id = f"XP1-B{block:02d}-T{index:02d}"
            rows.append(particle_row(logical_id, block, 29.0 if len(rows) < hit_count else 40.0))
    return {
        "arm_id": arm_id,
        "particle_metrics": rows,
        "summary": runner.summarize_particles(rows),
    }


class TestFrozenInputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = runner.strict_json(ROOT / "contract_v1.json")
        cls.manifest, cls.seeds = runner.validate_seed_manifest(
            cls.contract, ROOT / "seed_manifest_v1.json"
        )

    def test_contract_and_design_hashes(self) -> None:
        runner.validate_contract(self.contract, ROOT / "contract_v1.json")
        self.assertEqual(
            runner.sha256_bytes(runner.canonical_bytes(self.contract["design_core"])),
            "0865266fa46b3cdf080d783f366f4988a76fb1667bf334bd79b005e9ad68380c",
        )
        self.assertEqual(len(runner.arm_specifications(self.contract)), 14)
        self.assertEqual(
            [row["arm_id"] for row in runner.arm_specifications(self.contract)],
            list(runner.PRIMARY_ARM_IDS + runner.AUDIT_ARM_IDS),
        )

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            duplicate = pathlib.Path(directory) / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.strict_json(duplicate)
            nonfinite = pathlib.Path(directory) / "nonfinite.json"
            nonfinite.write_text('{"a":NaN}', encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.strict_json(nonfinite)
            underflow = pathlib.Path(directory) / "underflow.json"
            underflow.write_text('{"a":1e-9999}', encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.strict_json(underflow)
            with self.assertRaises(ValueError):
                verifier.read_json(underflow)

    def test_all_seed_bytes_recompute(self) -> None:
        self.assertEqual(len(self.manifest["streams"]), 24)
        for row in self.manifest["streams"]:
            expected = runner.derive_seed(
                self.manifest["domain_ascii"], self.manifest["design_core_sha256"],
                row["stream_label"], row["counter"],
            )[:16].hex()
            self.assertEqual(row["seed_hex_128"], expected)

    def test_portable_lhs_and_tracer_digest(self) -> None:
        tracers, canonical_rows, digest = runner.make_tracers(self.contract, self.seeds)
        self.assertEqual(len(tracers), 64)
        self.assertEqual(len(canonical_rows), 64)
        self.assertEqual(digest, "b98c8c27f3301f54afff72a0b71847e1508d6ed51dc2ce566c4ca9daec7133ab")
        permutation, fractions = runner.lhs_values(self.seeds["LHS_BLOCK_0_LOG_A"])
        self.assertEqual(sorted(permutation), list(range(16)))
        self.assertEqual(
            [value.hex() for value in fractions[:4]],
            ["0x1.0cf6f52b2afc6p-1", "0x1.a7245b502b2fep-4", "0x1.9c3eba048b96cp-1", "0x1.04ce7c89d6844p-3"],
        )
        for block in range(4):
            block_rows = tracers[block * 16:(block + 1) * 16]
            self.assertEqual([row["block_index"] for row in block_rows], [block] * 16)
            self.assertTrue(all(150.0 <= row["a_AU"] <= 800.0 for row in block_rows))
            self.assertTrue(all(35.0 <= row["q_AU"] <= 80.0 for row in block_rows))

    def test_verifier_independently_recomputes_tracers(self) -> None:
        rows, digest = verifier.independently_realize_tracers(self.contract, self.manifest)
        self.assertEqual(len(rows), 64)
        self.assertEqual(digest, "b98c8c27f3301f54afff72a0b71847e1508d6ed51dc2ce566c4ca9daec7133ab")


class TestRegistrationAndRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = runner.strict_json(ROOT / "contract_v1.json")

    def test_registration_locks_every_implementation_file(self) -> None:
        expected_inventory = runner.LOCKED_FILES | {"registration_v1.json"}
        self.assertEqual({path.name for path in ROOT.iterdir()}, expected_inventory)
        self.assertTrue(all(
            path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            for path in ROOT.iterdir()
        ))
        registration, anchor = runner.validate_registration(
            ROOT / "registration_v1.json", ROOT / "contract_v1.json",
            ROOT / "run_exploratory.py",
        )
        self.assertEqual(set(registration["locked_files"]), runner.LOCKED_FILES)
        self.assertEqual(
            registration["locked_files"],
            {
                relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
                for relative in sorted(runner.LOCKED_FILES)
            },
        )
        self.assertEqual(anchor, hashlib.sha256(
            (ROOT / "registration_v1.json").read_bytes()
        ).hexdigest())

    def test_registration_validator_accepts_exact_temp_anchor_and_rejects_mutant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for relative in runner.LOCKED_FILES:
                shutil.copy2(ROOT / relative, root / relative)
            locked = {
                relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
                for relative in sorted(runner.LOCKED_FILES)
            }
            registration = registration_payload(locked, self.contract["mandatory_nonclaim"])
            path = root / "registration_v1.json"
            path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            parsed, anchor = runner.validate_registration(
                path, root / "contract_v1.json", root / "run_exploratory.py"
            )
            self.assertEqual(parsed["locked_files"], locked)
            self.assertEqual(anchor, hashlib.sha256(path.read_bytes()).hexdigest())
            registration["locked_files"]["README.md"] = "0" * 64
            path.write_text(json.dumps(registration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            with self.assertRaises(runner.IntegrityError):
                runner.validate_registration(path, root / "contract_v1.json", root / "run_exploratory.py")

    def test_dangling_output_symlink_is_rejected_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            link = root / "output"
            link.symlink_to(root / "absent-target", target_is_directory=True)
            with self.assertRaises(ValueError):
                runner.validate_clean_output_directory(link, self.contract, ROOT)

    def test_source_only_runtime_in_fresh_process(self) -> None:
        code = (
            "import pathlib,types;"
            f"p=pathlib.Path({str(ROOT / 'run_exploratory.py')!r});"
            "m=types.ModuleType('r');m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);"
            f"c=m.strict_json(pathlib.Path({str(ROOT / 'contract_v1.json')!r}));"
            "x=m.validate_runtime(c);assert x['rebound_python_source_file_count']==29"
        )
        completed = subprocess.run(
            [str(PYTHON), "-B", "-c", code], check=False, capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_validate_only_cannot_bypass_missing_registration_or_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            marker = pathlib.Path(directory) / "must_not_exist"
            missing_registration = pathlib.Path(directory) / "registration_v1.json"
            completed = subprocess.run([
                str(PYTHON), "-B", str(ROOT / "run_exploratory.py"),
                "--contract", str(ROOT / "contract_v1.json"),
                "--seed-manifest", str(ROOT / "seed_manifest_v1.json"),
                "--registration", str(missing_registration),
                "--validate-only",
            ], check=False, capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(marker.exists())

    def test_registered_validate_only_builds_all_t0_states_without_dynamics(self) -> None:
        completed = subprocess.run([
            str(PYTHON), "-B", str(ROOT / "run_exploratory.py"),
            "--contract", str(ROOT / "contract_v1.json"),
            "--seed-manifest", str(ROOT / "seed_manifest_v1.json"),
            "--registration", str(ROOT / "registration_v1.json"),
            "--validate-only",
        ], check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual(receipt["preflight_arm_count"], 14)
        self.assertFalse(receipt["dynamics_executed"])


class TestNoDynamicsInitialConstruction(unittest.TestCase):
    def test_all_fourteen_t0_states_are_paired_without_integrating(self) -> None:
        contract = runner.strict_json(ROOT / "contract_v1.json")
        _, seeds = runner.validate_seed_manifest(contract, ROOT / "seed_manifest_v1.json")
        tracers, _, _ = runner.make_tracers(contract, seeds)
        bindings = runner.preflight_initial_states(contract, tracers)
        self.assertEqual(len(bindings), 14)
        self.assertEqual(
            len({row["pre_translation_common_relative_state_sha256"] for row in bindings.values()}),
            1,
        )
        self.assertLessEqual(
            max(row["post_com_max_binary64_epsilon_units"] for row in bindings.values()), 64.0
        )
        self.assertEqual(bindings["M0"]["particle_count"], 69)
        self.assertEqual(bindings["CI01-A"]["particle_count"], 70)

    def test_hyperbolic_finite_orbit_is_an_outcome_not_conversion_failure(self) -> None:
        class Particle:
            x = 1.0; y = 0.0; z = 0.0; vx = 0.0; vy = 1.0; vz = 0.0
            def orbit(self, primary=None):
                return types.SimpleNamespace(a=-100.0, e=1.5, inc=0.2)
        class Sun:
            x = y = z = vx = vy = vz = 0.0
        class Particles:
            def __init__(self): self.rows = [Sun(), Particle()]
            def __iter__(self): return iter(self.rows)
            def __getitem__(self, key): return self.rows[0] if key == "Sun" else self.rows[key]
        simulation = types.SimpleNamespace(particles=Particles(), N_active=1)
        tracker = [runner.blank_tracker([{"logical_id":"X","block_index":0,"index_within_block":0}])[0]]
        runner.sample_tracers(simulation, tracker, 250000.0)
        self.assertTrue(tracker[0]["all_samples_finite_osculating_orbit"])
        self.assertFalse(tracker[0]["final_finite_and_bound"])
        self.assertEqual(tracker[0]["final_q_AU"], 50.0)


class TestIndependentAnalysis(unittest.TestCase):
    def test_wasserstein_known_values(self) -> None:
        self.assertEqual(runner.empirical_w1([0.0, 2.0], [1.0, 3.0]), 1.0)
        self.assertEqual(verifier.w1([0.0, 2.0], [1.0, 3.0]), 1.0)

    def test_integer_effects_and_independent_recomputation(self) -> None:
        arms = {"M0": fake_arm("M0", 0)}
        for index, arm_id in enumerate(runner.PRIMARY_ARM_IDS[1:], start=1):
            arms[arm_id] = fake_arm(arm_id, index)
        first = runner.mixture_analysis(arms)
        second = verifier.recompute_analysis(arms)
        self.assertEqual(first, second)
        self.assertEqual(first["mixture_effects"]["q_below_30_effect_numerator"], 21)
        self.assertEqual(first["mixture_effects"]["q_below_30_effect_denominator"], 384)
        self.assertEqual([row["q_below_30_effect_denominator"] for row in first["block_effects"]], [96] * 4)

    def test_exact_integer_classification_boundaries(self) -> None:
        def analysis(numerator, blocks):
            return {
                "mixture_effects": {"q_below_30_effect_numerator": numerator},
                "block_effects": [
                    {"q_below_30_effect_numerator": value} for value in blocks
                ],
            }
        self.assertEqual(runner.raw_classification(analysis(20, [1,1,1,1])), "DIRECTIONALLY_STABLE_INCREASE")
        self.assertEqual(runner.raw_classification(analysis(-20, [-1,-1,-1,-1])), "DIRECTIONALLY_STABLE_DECREASE")
        self.assertEqual(runner.raw_classification(analysis(7, [4,-4,0,1])), "PRACTICALLY_SMALL")
        self.assertEqual(runner.raw_classification(analysis(8, [0,0,0,0])), "INCONCLUSIVE")
        for numerator, blocks in ((20,[1,1,1,1]),(-20,[-1,-1,-1,-1]),(7,[4,-4,0,1]),(8,[0,0,0,0])):
            self.assertEqual(runner.raw_classification(analysis(numerator, blocks)), verifier.expected_raw_classification(analysis(numerator, blocks)))

    def test_incomplete_orbit_metrics_take_locked_unresolved_path(self) -> None:
        contract = runner.strict_json(ROOT / "contract_v1.json")
        arms = {
            arm_id: fake_arm(arm_id, 1)
            for arm_id in runner.PRIMARY_ARM_IDS + runner.AUDIT_ARM_IDS
        }
        incomplete = arms["AUDIT-CI09-B"]["particle_metrics"][0]
        incomplete["all_samples_finite_osculating_orbit"] = False
        incomplete["minimum_sampled_q_AU"] = None
        incomplete["first_sampled_q_below_35_time_year"] = None
        incomplete["first_sampled_q_below_30_time_year"] = None
        arms["AUDIT-CI09-B"]["summary"] = runner.summarize_particles(
            arms["AUDIT-CI09-B"]["particle_metrics"]
        )
        analysis, timestep, timestep_pass, class_agreement = runner.two_resolution_analysis(
            arms, contract["numerical_gates"]
        )
        self.assertEqual(analysis, {
            "status": runner.ANALYSIS_SUPPRESSED_STATUS,
            "primary_dt": None,
            "audit_dt_half": None,
            "primary_raw_classification": None,
            "audit_raw_classification": None,
        })
        self.assertEqual(timestep, [])
        self.assertFalse(timestep_pass)
        self.assertFalse(class_agreement)

    def test_supervisor_success_and_timeout_paths_without_dynamics(self) -> None:
        contract = runner.strict_json(ROOT / "contract_v1.json")
        caps = dict(contract["resource_caps_per_execution"])
        caps["minimum_free_disk_bytes"] = 0
        caps["max_wall_seconds_total"] = 2.0
        caps["max_wall_seconds_per_arm"] = 1.0
        caps["watchdog_poll_seconds"] = 0.01
        local_contract = {**contract, "resource_caps_per_execution": caps}
        original = runner._arm_worker
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory)
            try:
                def success(write_file_descriptor, *unused):
                    runner.write_framed_worker_response(write_file_descriptor, {
                        "ok": True,
                        "semantic": {"arm_id": "TEST"},
                        "provenance": {
                            "arm_id": "TEST", "elapsed_seconds": 0.01,
                            "peak_rss_bytes": 1,
                        },
                    })
                    os.close(write_file_descriptor)
                runner._arm_worker = success
                semantic, provenance = runner.run_arm_supervised(
                    local_contract, [], {"arm_id": "TEST"}, {}, output,
                    time.monotonic_ns(),
                )
                self.assertEqual(semantic, {"arm_id": "TEST"})
                self.assertEqual(provenance["arm_id"], "TEST")

                memory_cap = runner.peak_rss_bytes() + 16 * 1024 * 1024
                def late_memory(write_file_descriptor, *unused):
                    runner.write_framed_worker_response(write_file_descriptor, {
                        "ok": True,
                        "semantic": {"arm_id": "TEST"},
                        "provenance": {
                            "arm_id": "TEST", "elapsed_seconds": 0.01,
                            "peak_rss_bytes": 1,
                        },
                    })
                    allocation = bytearray(memory_cap)
                    for offset in range(0, len(allocation), 4096):
                        allocation[offset] = 1
                    os.close(write_file_descriptor)
                runner._arm_worker = late_memory
                memory_contract = {
                    **local_contract,
                    "resource_caps_per_execution": {
                        **caps, "max_peak_rss_bytes": memory_cap,
                    },
                }
                with self.assertRaises(runner.ResourceLimitError):
                    runner.run_arm_supervised(
                        memory_contract, [], {"arm_id": "TEST"}, {}, output,
                        time.monotonic_ns(),
                    )

                def sleeper(write_file_descriptor, *unused):
                    os.write(write_file_descriptor, (100).to_bytes(8, "big")[:4])
                    time.sleep(1.0)
                    os.close(write_file_descriptor)
                runner._arm_worker = sleeper
                timeout_contract = {
                    **local_contract,
                    "resource_caps_per_execution": {
                        **caps,
                        "max_wall_seconds_total": 0.1,
                        "max_wall_seconds_per_arm": 0.05,
                    },
                }
                timeout_started = time.monotonic()
                with self.assertRaises(runner.ResourceLimitError):
                    runner.run_arm_supervised(
                        timeout_contract, [], {"arm_id": "TEST"}, {}, output,
                        time.monotonic_ns(),
                    )
                self.assertLess(time.monotonic() - timeout_started, 0.5)
            finally:
                runner._arm_worker = original

    def test_watchdog_deadline_and_proc_peak_parsing(self) -> None:
        start = 9_000_000_000
        deadline = runner.deadline_ns(start, 0.25)
        self.assertEqual(deadline, 9_250_000_000)
        self.assertFalse(runner.deadline_expired(deadline - 1, deadline))
        self.assertTrue(runner.deadline_expired(deadline, deadline))
        status = "Name:\tpython\nVmHWM:\t12345 kB\nThreads:\t1\n"
        self.assertEqual(
            runner.parse_proc_status_integer(status, "VmHWM", "kB"), 12345
        )
        self.assertEqual(
            runner.parse_proc_status_integer(status, "Threads", None), 1
        )
        self.assertEqual(
            runner.peak_rss_from_proc_status("State:\tZ (zombie)\nThreads:\t1\n"), 0
        )

    def test_runner_and_verifier_do_not_reuse_e1_or_persist_trajectories(self) -> None:
        runner_source = (ROOT / "run_exploratory.py").read_text(encoding="utf-8")
        verifier_source = (ROOT / "verify_replay.py").read_text(encoding="utf-8")
        self.assertNotIn("import run_long_pilot", runner_source)
        self.assertNotIn("import run_pilot", runner_source)
        self.assertNotIn("save_to_file", runner_source)
        self.assertEqual(runner_source.count("run_arm_supervised("), 2)
        self.assertEqual(runner_source.count("run_arm("), 2)
        self.assertNotIn("import run_exploratory", verifier_source)
        self.assertNotIn(".integrate(", verifier_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
