from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_covariance_classification_pilot_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_covariance_classification_pilot_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load TNO covariance pilot protocol")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PILOT = _load_module()


class TnoCovarianceClassificationPilotV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = PILOT.build_protocol()

    def test_scope_inputs_and_unauthenticated_custody_are_exact(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], PILOT.SCHEMA)
        self.assertEqual(protocol["classification"], PILOT.CLASSIFICATION)
        self.assertEqual(protocol["scope"]["object_count"], 12)
        self.assertIs(protocol["scope"]["final_36_object_science_sample"], False)
        self.assertIs(protocol["scope"]["population_study"], False)
        self.assertIs(protocol["scope"]["planet_nine_test"], False)
        acquisition = protocol["inputs"]["acquisition"]
        clones = protocol["inputs"]["clones"]
        self.assertEqual(
            acquisition["custody_content_sha256"],
            PILOT.ACQUISITION_CUSTODY_CONTENT_SHA256,
        )
        self.assertEqual(
            acquisition["authentication_status"],
            "SCHEMA_CONFORMANT_UNAUTHENTICATED_RETAINED_BYTES",
        )
        self.assertEqual(
            clones["artifact_content_sha256"],
            PILOT.CLONE_ARTIFACT_CONTENT_SHA256,
        )
        self.assertEqual(
            clones["binary_payload_sha256"], PILOT.CLONE_BINARY_PAYLOAD_SHA256
        )
        self.assertEqual(clones["total_clone_count"], 1536)
        self.assertIs(
            protocol["claims"]["retained_covariance_bytes_transport_authenticated"],
            False,
        )
        self.assertIn(
            "NOT_POSTERIOR_SAMPLES_OR_LIKELIHOOD_DRAWS",
            protocol["custody_interpretation"]["clone_interpretation"],
        )
        self.assertIn(
            "DOES_NOT_ESTABLISH_TAIL_PROBABILITIES_OR_POPULATION_PRECISION",
            protocol["custody_interpretation"]["clone_count_limitation"],
        )

    def test_staged_design_counts_steps_and_triggers_are_frozen(self) -> None:
        stages = {row["stage"]: row for row in self.protocol["stages"]}
        self.assertEqual(set(stages), {0, 1, 2, 3})
        self.assertEqual(stages[0]["total_tracer_state_count"], 1548)
        self.assertEqual(
            stages[0]["input_semantics_to_bind"],
            {
                "covariance_epoch": "JULIAN_DATE_TDB",
                "tp_coordinate": "ABSOLUTE_JULIAN_DATE_TDB_WITH_COVARIANCE_DIFFERENTIALS_IN_DAYS",
                "element_frame": "HELIOCENTRIC_J2000_ECLIPTIC",
                "binding_status": "REQUIRED_NOT_YET_IMPLEMENTED_OR_QUALIFIED",
            },
        )
        self.assertEqual(
            stages[1]["holdout_clone_indices_per_object"],
            [0, 18, 36, 54, 73, 91, 109, 127],
        )
        self.assertEqual(stages[1]["horizon_julian_years"], 1000)
        self.assertEqual(stages[1]["total_tracer_state_count"], 108)
        self.assertEqual(stages[1]["primary_step_julian_years"], 0.0625)
        self.assertEqual(stages[1]["refinement_step_julian_years"], 0.03125)
        self.assertEqual(stages[2]["horizon_julian_years"], 10_000_000)
        self.assertEqual(stages[2]["total_tracer_state_count"], 1548)
        self.assertEqual(stages[2]["primary_step_julian_years"], 0.5)
        self.assertEqual(stages[2]["refinement_step_julian_years"], 0.25)
        self.assertEqual(stages[3]["target_total_horizon_julian_years"], 100_000_000)
        self.assertEqual(stages[3]["additional_horizon_after_stage_2_julian_years"], 90_000_000)
        self.assertEqual(
            stages[3]["trigger_labels"],
            [
                "MIXED_UNDER_ORBIT_COVARIANCE",
                "BORDERLINE_NUMERICAL_CLASSIFICATION",
            ],
        )
        self.assertIs(stages[3]["stage_2_label_must_remain_reported"], True)
        self.assertTrue(all(row["execution_authorized"] is False for row in stages.values()))

    def test_resolved_sources_consensus_rule_and_blockers_are_exact(self) -> None:
        model = self.protocol["known_planets_model"]
        self.assertEqual(
            model["required_default_source_ids_in_exact_order"],
            list(PILOT.RESOLVED_SOURCE_IDS),
        )
        self.assertEqual(model["required_default_massive_source_count"], 11)
        self.assertIs(model["unqualified_five_body_or_inner_monopole_collapse_permitted"], False)
        self.assertIs(model["planet_nine_included"], False)
        decision = self.protocol["classification_decision_rule"]
        self.assertEqual(decision["clone_denominator"], 128)
        self.assertEqual(decision["minimum_agreeing_clones_for_robust_class"], 123)
        self.assertEqual(decision["minimum_fraction_exact"], "123/128")
        self.assertIs(decision["nominal_must_equal_clone_consensus_class"], True)
        self.assertEqual(
            decision["covariance_disagreement_label"],
            "MIXED_UNDER_ORBIT_COVARIANCE",
        )
        gate = self.protocol["execution_gate"]
        self.assertEqual(gate["blockers"], list(PILOT.BLOCKERS))
        self.assertIs(gate["execution_authorized"], False)
        self.assertIs(gate["external_immutable_commit_present"], False)
        self.assertIs(gate["trusted_preexecution_timestamp_present"], False)
        self.assertIn("DE440_OR_DE440S", gate["ephemeris_mismatch_caution"])

    def test_claim_ceiling_references_and_no_execution_surface(self) -> None:
        claims = self.protocol["claims"]
        for key in (
            "future_execution_externally_preregistered",
            "pilot_execution_authorized",
            "pilot_executed",
            "dynamical_classifications_generated",
            "observational_likelihood_evaluated",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            self.assertIs(claims[key], False, key)
        self.assertIs(self.protocol["execution_authorized"], False)
        self.assertIs(self.protocol["outcomes_generated"], False)
        self.assertEqual(
            {row.get("doi") for row in self.protocol["references"] if "doi" in row},
            {"10.3847/1538-4365/ac3914"},
        )
        self.assertEqual(
            self.protocol["references"][0]["url"],
            "https://ssd-api.jpl.nasa.gov/doc/sbdb.html",
        )
        self.assertEqual(self.protocol["implementation"]["cli_surface"], "PREPARE_ONLY")
        self.assertFalse(hasattr(PILOT, "execute"))
        self.assertFalse(hasattr(PILOT, "run"))

    def test_retained_protocol_is_canonical_and_exactly_reconstructed(self) -> None:
        retained = PILOT.validate_file(PILOT.DEFAULT_OUTPUT)
        self.assertEqual(retained, self.protocol)
        self.assertEqual(
            PILOT.DEFAULT_OUTPUT.read_bytes(), PILOT._canonical_json(retained)
        )

    def test_protocol_validation_is_type_strict_and_rejects_rehashed_tamper(self) -> None:
        self.assertEqual(
            PILOT.validate_protocol(self.protocol), self.protocol["content_sha256"]
        )
        for path, replacement in (
            (("claims", "science_result"), 0),
            (("stages", 2, "total_tracer_state_count"), 1548.0),
            (("classification_decision_rule", "minimum_agreeing_clones_for_robust_class"), 122),
        ):
            forged = copy.deepcopy(self.protocol)
            target = forged
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = replacement
            forged["content_sha256"] = PILOT._content_sha256(forged)
            with self.assertRaisesRegex(
                PILOT.PilotProtocolError, "exact frozen reconstruction"
            ):
                PILOT.validate_protocol(forged)

    def test_bound_file_drift_and_rederived_clone_drift_fail_closed(self) -> None:
        changed_binding = list(PILOT.CLONE_BINDINGS)
        role, path, size, _digest = changed_binding[-1]
        changed_binding[-1] = (role, path, size, "0" * 64)
        with mock.patch.object(PILOT, "CLONE_BINDINGS", tuple(changed_binding)):
            with self.assertRaisesRegex(PILOT.PilotProtocolError, "digest differs"):
                PILOT.build_protocol()

        clone_module = PILOT._load_clone_module()
        forged = PILOT._load_canonical(PILOT.CLONE_ROOT / "CLONES.json")
        forged["claims"]["science_result"] = 0
        with mock.patch.object(PILOT, "_load_clone_module", return_value=clone_module):
            with mock.patch.object(clone_module, "build_clones", return_value=forged):
                with self.assertRaisesRegex(
                    PILOT.PilotProtocolError, "clone artifact reconstruction"
                ):
                    PILOT._validate_and_rederive_inputs()

    def test_clone_loader_executes_pinned_source_bytes_not_pyc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fake_clones.py"
            safe = b"SENTINEL = 'PINNED_SOURCE'".ljust(96, b" ") + b"\n"
            malicious = b"SENTINEL = 'UNPINNED_PYC'".ljust(96, b" ") + b"\n"
            self.assertEqual(len(safe), len(malicious))
            source.write_bytes(malicious)
            timestamp_ns = 1_700_000_000_000_000_000
            os.utime(source, ns=(timestamp_ns, timestamp_ns))
            optimization = "" if sys.flags.optimize == 0 else str(sys.flags.optimize)
            cache = Path(importlib.util.cache_from_source(str(source), optimization=optimization))
            cache.parent.mkdir(parents=True, exist_ok=True)
            py_compile.compile(
                str(source),
                cfile=str(cache),
                doraise=True,
                optimize=sys.flags.optimize,
            )
            source.write_bytes(safe)
            os.utime(source, ns=(timestamp_ns, timestamp_ns))
            binding = (
                "implementation",
                str(source),
                len(safe),
                hashlib.sha256(safe).hexdigest(),
            )
            with mock.patch.object(PILOT, "CLONE_TOOL", source):
                with mock.patch.object(
                    PILOT,
                    "CLONE_BINDINGS",
                    (binding,) + PILOT.CLONE_BINDINGS[1:],
                ):
                    loaded = PILOT._load_clone_module()
            self.assertEqual(loaded.SENTINEL, "PINNED_SOURCE")

    def test_create_only_canonical_output_and_symlink_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot" / "PROTOCOL.json"
            prepared = PILOT.prepare(output)
            self.assertEqual(PILOT.validate_file(output), prepared)
            self.assertEqual(output.read_bytes(), PILOT._canonical_json(prepared))
            with self.assertRaisesRegex(PILOT.PilotProtocolError, "not empty"):
                PILOT.prepare(output)

        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real.json"
            real.write_text("{}\n", encoding="ascii")
            linked = Path(directory) / "linked.json"
            linked.symlink_to(real)
            with self.assertRaises(PILOT.PilotProtocolError):
                PILOT._read_regular_bytes(linked)

    def test_optimized_build_matches_and_cli_requires_prepare_only(self) -> None:
        script = (
            "import importlib.util,sys;"
            f"p={str(SCRIPT_PATH)!r};"
            "s=importlib.util.spec_from_file_location('jx_tno_pilot_oo',p);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "q=m.build_protocol();m.validate_protocol(q);print(q['content_sha256'])"
        )
        completed = subprocess.run(
            [sys.executable, "-OO", "-c", script],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), self.protocol["content_sha256"])

        rejected = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("--prepare-only is required", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
