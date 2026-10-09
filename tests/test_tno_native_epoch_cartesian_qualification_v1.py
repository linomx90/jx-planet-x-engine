from __future__ import annotations

import copy
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "benchmarks" / "jx_tno_native_epoch_cartesian_qualification_v1.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "jx_tno_native_epoch_cartesian_qualification_v1", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load native-epoch Cartesian qualifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


STAGE0 = _load_module()


def _fixture_gm():
    path = (
        PROJECT_ROOT
        / "runs/tno_native_epoch_horizons_oracle_v1/source/gm_Horizons.pck"
    )
    return STAGE0._derive_official_gm(
        path, "169cfed3b0927e73929d0a1b5c931f9afb5167a83b921064127ffc54a673df0c"
    )


_DECIMAL_PI = Decimal(
    "3.141592653589793238462643383279502884197169399375105820974944592307816406286"
)


def _decimal_sine(value: Decimal) -> Decimal:
    term = value
    total = value
    index = 1
    while True:
        term *= -(value * value) / Decimal((2 * index) * (2 * index + 1))
        updated = total + term
        if updated == total:
            return updated
        total = updated
        index += 1
        if index > 300:
            raise RuntimeError("decimal sine series did not converge")


def _decimal_kepler(mean: str, eccentricity: str) -> Decimal:
    with localcontext() as context:
        context.prec = 90
        target = Decimal(mean)
        e = Decimal(eccentricity)
        lower = -_DECIMAL_PI
        upper = _DECIMAL_PI
        for _ in range(340):
            middle = (lower + upper) / 2
            residual = middle - e * _decimal_sine(middle) - target
            if residual > 0:
                upper = middle
            elif residual < 0:
                lower = middle
            else:
                return +middle
        return +((lower + upper) / 2)


class TnoNativeEpochCartesianQualificationV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gm = _fixture_gm()
        _bindings, cls.clone_artifact, cls.native_records = STAGE0._validate_upstream()
        cls.protocol = STAGE0.build_protocol()
        cls.artifact = STAGE0.build_states(cls.protocol)

    def test_official_gm_exact_decimal_units_and_identity(self) -> None:
        gm = self.gm
        self.assertEqual(
            gm.gm_AU3_per_julian_year2.hex(),
            STAGE0.HORIZONS_BODY10_GM_AU3_PER_JULIAN_YEAR2_HEX,
        )
        self.assertEqual(
            STAGE0.HORIZONS_BODY10_GM_AU3_PER_DAY2_HEX,
            "0x1.3649541cf2bdcp-12",
        )
        self.assertEqual(gm.primary_id, "NAIF_10_SUN")
        self.assertEqual(gm.source_token_km3_per_s2, "1.3271244004127942E+11")
        self.assertNotIn("MONOPOLE", gm.semantic_id)

    def test_sealed_clone_artifact_yields_exact_1548_native_records(self) -> None:
        records = self.native_records
        self.assertEqual(len(records), 1548)
        self.assertEqual(sum(row.clone_index is None for row in records), 12)
        self.assertEqual(sum(row.clone_index is not None for row in records), 1536)
        for offset in range(0, len(records), 129):
            block = records[offset : offset + 129]
            self.assertIsNone(block[0].clone_index)
            self.assertEqual([row.clone_index for row in block[1:]], list(range(128)))
            self.assertTrue(all(row.covariance_epoch_jd_tdb == block[0].covariance_epoch_jd_tdb for row in block))

    def test_kepler_solver_matches_high_precision_reference_at_boundaries(self) -> None:
        cases = (
            ("0", "0"),
            ("1e-15", "0.5"),
            ("0.001", "0.9"),
            ("1e-8", "0.999999"),
            ("3.141592653589793", "0.84"),
            ("-3.141592653589793", "0.84"),
        )
        for mean_text, eccentricity_text in cases:
            expected = float(_decimal_kepler(mean_text, eccentricity_text))
            actual = STAGE0.solve_elliptic_kepler(
                float(mean_text), float(eccentricity_text)
            )
            self.assertLessEqual(abs(actual - expected), 3.0e-14, (mean_text, eccentricity_text))
            residual = actual - float(eccentricity_text) * math.sin(actual) - float(mean_text)
            self.assertLessEqual(abs(residual), STAGE0.KEPLER_RESIDUAL_TOLERANCE)

        near_tau = math.nextafter(math.tau, 0.0)
        root = STAGE0.solve_elliptic_kepler(near_tau, 0.7)
        self.assertLess(root, 0.0)
        self.assertLess(abs(root - 0.7 * math.sin(root) - math.remainder(near_tau, math.tau)), 2.0e-14)

    def test_epoch_tp_phase_and_angle_wrap_edges_roundtrip(self) -> None:
        at_perihelion = STAGE0.NativeElementRecord(
            "fixture-perihelion",
            1,
            "TEST_ONLY",
            2_451_545.0,
            None,
            (0.42, 35.0, 2_451_545.0, 359.999999999, -0.000000001, 17.0),
        )
        state = STAGE0.native_elements_to_cartesian(at_perihelion, self.gm)
        self.assertEqual(state.mean_anomaly_rad, 0.0)
        errors = STAGE0.roundtrip_errors(at_perihelion, state, self.gm)
        self.assertLess(errors["ascending_node_rad"], 1.0e-12)
        self.assertLess(errors["argument_of_perihelion_rad"], 1.0e-12)

        a = 50.0
        e = 0.2
        q = a * (1.0 - e)
        mean_motion = math.sqrt(self.gm.gm_AU3_per_julian_year2 / a**3)
        target_mean = math.nextafter(math.tau, 0.0)
        epoch = 2_451_545.0
        tp = epoch - (target_mean / mean_motion) * STAGE0.JULIAN_YEAR_DAYS
        near_wrap = STAGE0.NativeElementRecord(
            "fixture-wrap", 2, "TEST_ONLY", epoch, None, (e, q, tp, 0.0, 360.0, 0.001)
        )
        wrapped = STAGE0.native_elements_to_cartesian(near_wrap, self.gm)
        self.assertLess(abs(wrapped.mean_anomaly_rad), 1.0e-10)

    def test_all_retained_states_pass_inverse_roundtrip(self) -> None:
        maxima = {key: 0.0 for key in (
            "semimajor_axis_relative",
            "perihelion_relative",
            "eccentricity_absolute",
            "inclination_rad",
            "ascending_node_rad",
            "argument_of_perihelion_rad",
            "mean_anomaly_rad",
        )}
        for native in self.native_records:
            state = STAGE0.native_elements_to_cartesian(native, self.gm)
            errors = STAGE0.roundtrip_errors(native, state, self.gm)
            for key, value in errors.items():
                maxima[key] = max(maxima[key], value)
        self.assertLessEqual(maxima["semimajor_axis_relative"], STAGE0.ROUNDTRIP_A_RELATIVE_TOLERANCE)
        self.assertLessEqual(maxima["perihelion_relative"], STAGE0.ROUNDTRIP_Q_RELATIVE_TOLERANCE)
        self.assertLessEqual(maxima["eccentricity_absolute"], STAGE0.ROUNDTRIP_E_ABSOLUTE_TOLERANCE)
        self.assertTrue(all(maxima[key] <= STAGE0.ROUNDTRIP_ANGLE_TOLERANCE_RAD for key in maxima if key.endswith("_rad")))

    def test_rebound_independent_library_conversion_matches(self) -> None:
        try:
            import rebound
        except ImportError:
            self.skipTest("optional REBOUND independent comparator is unavailable")
        selected = (
            self.native_records[0],
            self.native_records[1],
            self.native_records[129],
            self.native_records[-1],
        )
        for native in selected:
            state = STAGE0.native_elements_to_cartesian(native, self.gm)
            eccentricity, perihelion, _tp, node, argument, inclination = native.values
            simulation = rebound.Simulation()
            simulation.G = self.gm.gm_AU3_per_julian_year2
            simulation.add(m=1.0)
            simulation.add(
                m=0.0,
                a=perihelion / (1.0 - eccentricity),
                e=eccentricity,
                inc=math.radians(inclination),
                Omega=math.radians(node),
                omega=math.radians(argument),
                M=state.mean_anomaly_rad,
            )
            particle = simulation.particles[1]
            reference = (
                particle.x,
                particle.y,
                particle.z,
                particle.vx,
                particle.vy,
                particle.vz,
            )
            actual = state.position_AU + state.velocity_AU_per_julian_year
            self.assertLessEqual(max(abs(a - b) for a, b in zip(actual[:3], reference[:3], strict=True)), 2.0e-11)
            self.assertLessEqual(max(abs(a - b) for a, b in zip(actual[3:], reference[3:], strict=True)), 2.0e-13)

    def test_protocol_and_artifact_are_input_only_and_oracle_qualified(self) -> None:
        protocol = self.protocol
        self.assertEqual(protocol["schema"], STAGE0.PROTOCOL_SCHEMA)
        self.assertEqual(protocol["input_contract"]["total_state_count"], 1548)
        self.assertEqual(
            protocol["input_contract"]["tp_semantics"],
            "ABSOLUTE_JD_TDB_WITH_COVARIANCE_DIFFERENTIALS_IN_DAYS",
        )
        self.assertEqual(
            protocol["central_gm"]["gm_AU3_per_day2_binary64_hex"],
            "0x1.3649541cf2bdcp-12",
        )
        self.assertIs(
            protocol["qualification"]["independent_horizons_nominal_oracles"]["all_nominal_oracles_pass"],
            True,
        )
        element_crosscheck = protocol["qualification"][
            "sbdb_to_horizons_printed_element_crosscheck"
        ]
        self.assertIs(
            element_crosscheck["all_e_q_node_peri_i_decimal_tokens_exactly_equal"],
            True,
        )
        self.assertEqual(
            element_crosscheck["maximum_tp_printing_difference_seconds_decimal"],
            "0.0000187488",
        )
        self.assertEqual(element_crosscheck["one_ulp_tp_spk_ids"], [20574372])
        self.assertEqual(self.artifact["total_state_count"], 1548)
        self.assertEqual(len(self.artifact["states"]), 1548)
        for key in (
            "clone_dynamics_executed",
            "one_thousand_year_holdout_executed",
            "gpu_executed",
            "dynamical_classifications_generated",
            "long_horizon_execution_authorized",
            "population_inference",
            "planet_nine_evidence",
            "science_result",
        ):
            self.assertIs(protocol["claims"][key], False, key)
        self.assertIs(protocol["execution_authorized"], False)
        self.assertIs(protocol["outcomes_generated"], False)

    def test_oracle_schema_and_printed_element_tampering_fail_closed(self) -> None:
        _bindings, custody, _gm = STAGE0._validate_oracle()
        forged_vector = copy.deepcopy(custody)
        forged_vector["records"][0]["spk_id"] = True
        with self.assertRaisesRegex(STAGE0.CartesianQualificationError, "frozen contract"):
            STAGE0._oracle_vectors(forged_vector)

        forged_elements = copy.deepcopy(custody)
        forged_elements["records"][0]["printed_osculating_element_decimal_tokens"][0] = "0.5"
        with self.assertRaises(STAGE0.CartesianQualificationError):
            STAGE0._oracle_element_crosscheck(
                self.native_records, self.clone_artifact, forged_elements
            )

    def test_bound_verifier_loader_ignores_timestamp_valid_unbound_pyc(self) -> None:
        malicious = b"VALUE = 'MALICE'\n"
        trusted = b"VALUE = 'SOURCE'\n"
        self.assertEqual(len(malicious), len(trusted))
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT, prefix=".stage0-loader-") as directory:
            source = Path(directory) / "fixture.py"
            source.write_bytes(malicious)
            source_stat = source.stat()
            py_compile.compile(str(source), doraise=True)
            source.write_bytes(trusted)
            os.utime(source, ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns))
            relative = source.relative_to(PROJECT_ROOT).as_posix()
            binding = (
                "hostile_pyc_fixture",
                relative,
                len(trusted),
                hashlib.sha256(trusted).hexdigest(),
            )
            module_name = "jx_tno_stage0_hostile_pyc_fixture"
            try:
                loaded = STAGE0._load_bound_module(module_name, binding)
                self.assertEqual(loaded.VALUE, "SOURCE")
            finally:
                sys.modules.pop(module_name, None)
            bad_binding = (binding[0], binding[1], binding[2], "0" * 64)
            with self.assertRaisesRegex(STAGE0.CartesianQualificationError, "digest differs"):
                STAGE0._load_bound_module(module_name, bad_binding)

    def test_type_tamper_binding_drift_and_create_only_fail_closed(self) -> None:
        forged = copy.deepcopy(self.protocol)
        forged["claims"]["science_result"] = 0
        forged["content_sha256"] = STAGE0._content_sha256(STAGE0.PROTOCOL_DOMAIN, forged)
        with self.assertRaisesRegex(STAGE0.CartesianQualificationError, "exact reconstruction"):
            STAGE0.validate_protocol(forged)

        changed = list(STAGE0.UPSTREAM_BINDINGS)
        role, path, size, _digest = changed[0]
        changed[0] = (role, path, size, "0" * 64)
        with mock.patch.object(STAGE0, "UPSTREAM_BINDINGS", tuple(changed)):
            with self.assertRaisesRegex(STAGE0.CartesianQualificationError, "digest differs"):
                STAGE0.build_protocol()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stage0"
            artifact = STAGE0.materialize(output)
            self.assertEqual(STAGE0.validate_output(output), artifact)
            with self.assertRaisesRegex(STAGE0.CartesianQualificationError, "empty directory"):
                STAGE0.materialize(output)

    def test_retained_output_is_canonical_exact_and_optimized_stable(self) -> None:
        retained = STAGE0.validate_output(STAGE0.DEFAULT_OUTPUT_ROOT)
        self.assertEqual(retained, self.artifact)
        self.assertEqual(
            (STAGE0.DEFAULT_OUTPUT_ROOT / "PROTOCOL.json").read_bytes(),
            STAGE0._canonical_json(self.protocol),
        )
        self.assertEqual(
            (STAGE0.DEFAULT_OUTPUT_ROOT / "STATES.json").read_bytes(),
            STAGE0._canonical_json(self.artifact),
        )
        script = (
            "import importlib.util,sys;"
            f"p={str(SCRIPT_PATH)!r};"
            "s=importlib.util.spec_from_file_location('jx_tno_stage0_oo',p);"
            "m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);"
            "q=m.build_protocol();a=m.build_states(q);"
            "print(q['content_sha256']);print(a['content_sha256'])"
        )
        completed = subprocess.run(
            [sys.executable, "-OO", "-c", script],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.strip().splitlines(),
            [self.protocol["content_sha256"], self.artifact["content_sha256"]],
        )

    def test_cli_has_no_dynamics_mode(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--qualify-only is required", completed.stderr)
        self.assertFalse(hasattr(STAGE0, "integrate"))
        self.assertFalse(hasattr(STAGE0, "run_dynamics"))


if __name__ == "__main__":
    unittest.main()
