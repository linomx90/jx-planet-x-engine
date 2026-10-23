from __future__ import annotations

from decimal import Decimal, getcontext
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/jx_wh_v9_hierarchical_holdout.py"
SPEC = importlib.util.spec_from_file_location("jx_wh_v9_holdout_constructor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class _DecimalMP:
    @staticmethod
    def mpf(value: object) -> Decimal:
        if type(value) is float:
            return Decimal.from_float(value)
        return Decimal(value)  # type: ignore[arg-type]

    @staticmethod
    def power(base: object, exponent: int) -> Decimal:
        return Decimal(base) ** exponent


class HoldoutConstructorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        getcontext().prec = 100
        cls.protocol, cls.canonical_protocol = RUNNER._protocol()

    def test_revision_five_protocol_identity_and_import_boundary(self) -> None:
        self.assertEqual(self.protocol["protocol_revision"], 5)
        self.assertEqual(len(self.canonical_protocol), RUNNER.PROTOCOL_CANONICAL_SIZE)
        self.assertEqual(
            RUNNER._sha256(self.canonical_protocol), RUNNER.PROTOCOL_CANONICAL_SHA256
        )
        probe = (
            "import importlib.util,pathlib,sys;"
            f"p=pathlib.Path({str(SCRIPT)!r});"
            "s=importlib.util.spec_from_file_location('holdout_probe',p);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
            "m._protocol();"
            "assert not any(n == r or n.startswith(r+'.') "
            "for n in sys.modules for r in ('jxplanetx','scipy','rebound'))"
        )
        completed = subprocess.run(
            (sys.executable, "-S", "-c", probe),
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_draw_known_answers_and_literal_tokens(self) -> None:
        seed = "00" * 32
        self.assertEqual(
            RUNNER._draw_digest(
                self.protocol,
                seed,
                "JXHHO-I-N03-00",
                "-",
                "interior_cell_rank",
            ).hex(),
            "e62c4b897de9d912e576b4e0c71ced31fa0b0439a9aa2c35a9691463eb48ba78",
        )
        self.assertEqual(
            RUNNER._draw_digest(
                self.protocol,
                seed,
                "JXHHO-I-N03-00",
                "1",
                "eccentricity_u",
            ).hex(),
            "6ac5f525e7ac83fac76cfe7e28c28a68d6ed83cec93fe13021a4534ff4b35050",
        )
        with self.assertRaises(RUNNER.HoldoutError):
            RUNNER._draw_digest(
                self.protocol, seed, "JXHHO-I-N03-00", "01", "eccentricity_u"
            )
        with self.assertRaises(RUNNER.HoldoutError):
            RUNNER._draw_digest(
                self.protocol, seed, "JXHHO-I-N03-00", "1", "unknown"
            )

    def test_mass_vectors_close_the_frozen_binary64_recurrence(self) -> None:
        mp = _DecimalMP()
        ordinary = {"body_count": 32, "stress": "MASS_RATIO_0_99"}
        values = RUNNER._case_mass_vector(mp, ordinary, 0, self.protocol)
        total = 0.0
        for value in values[1:]:
            total = float(total + value)
        self.assertEqual(total.hex(), float(Decimal("0.0099")).hex())
        self.assertTrue(all(value > 0.0 for value in values))

        interaction = {
            "body_count": 8,
            "stress": "INTERACTION_FORCE_DYNAMIC_STRESS_0_15_UTILIZATION",
        }
        values = RUNNER._case_mass_vector(mp, interaction, 0, self.protocol)
        self.assertEqual(values[2].hex(), "0x1.26e978d4fdf3bp-7")
        total = 0.0
        for value in values[1:]:
            total = float(total + value)
        self.assertEqual(total.hex(), "0x1.4467381d7dbf5p-7")

        control = {"body_count": 33, "control_id": "JXHHO-C-N33-OUT"}
        values = RUNNER._control_mass_vector(mp, control)
        total = 0.0
        for value in values[1:]:
            total = float(total + value)
        self.assertEqual(total.hex(), float(Decimal(2) ** -10).hex())

    def test_cumulative_binding_is_ordered_and_domain_separated(self) -> None:
        gm = [1.0, float.fromhex("0x1.0p-10"), float.fromhex("0x1.0p-11")]
        cumulative = RUNNER._cumulative_gm(gm)
        hex_values = [value.hex() for value in cumulative]
        digest = RUNNER._cumulative_binding("case-a", hex_values)
        self.assertNotEqual(digest, RUNNER._cumulative_binding("case-b", hex_values))
        self.assertNotEqual(
            digest, RUNNER._cumulative_binding("case-a", list(reversed(hex_values)))
        )
        with self.assertRaises(RUNNER.HoldoutError):
            RUNNER._cumulative_gm([1.0, 0.0])

    def test_guard_replica_retains_native_momentum_round_trip(self) -> None:
        gm = [1.0, 0.0099]
        raw_relative = float.fromhex("0x1.ec2a47e4de4aep-1")
        guard = RUNNER._standalone_initial_guard(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, raw_relative, 0.0]],
            gm,
            [0.0, 0.0],
            1.0e-6,
            float.fromhex("0x1.0p-40"),
        )
        inertia = gm[1] * gm[0] / (gm[0] + gm[1])
        expected = (inertia * raw_relative) / inertia
        self.assertNotEqual(expected, raw_relative)
        self.assertEqual(guard["relative_velocities"][1][1], expected)
        self.assertEqual(guard["minimum_mutual_hill_separation"], 0.0)
        self.assertTrue(guard["translation_force_residual"] >= 0.0)

    def test_failed_calibration_preserves_all_branch_decisions(self) -> None:
        mp = _DecimalMP()

        def evaluator(value: Decimal) -> tuple[float, None]:
            return float(value), None

        with self.assertRaises(RUNNER.CalibrationUnavailable) as caught:
            RUNNER._calibrate_parameter(
                mp,
                "0",
                "1",
                0.5,
                (0.6, 0.7),
                evaluator,
                decreasing=False,
                label="synthetic",
            )
        self.assertEqual(len(caught.exception.branches), 80)

    def test_shadowed_mpmath_is_rejected_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "mpmath"
            metadata = root / "mpmath-1.3.0.dist-info"
            package.mkdir()
            metadata.mkdir()
            (package / "__init__.py").write_text("__version__='1.3.0'\n", "ascii")
            (metadata / "METADATA").write_text(
                "Metadata-Version: 2.1\nName: mpmath\nVersion: 1.3.0\n", "ascii"
            )
            (metadata / "RECORD").write_text(
                "mpmath/__init__.py,,\nmpmath-1.3.0.dist-info/METADATA,,\n", "ascii"
            )
            probe = (
                "import importlib.util,pathlib,sys;"
                f"sys.path.insert(0,{str(root)!r});"
                f"p=pathlib.Path({str(SCRIPT)!r});"
                "s=importlib.util.spec_from_file_location('holdout_shadow_probe',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "\ntry:m._installed_mpmath_record()\n"
                "except m.HoldoutError:raise SystemExit(0)\n"
                "raise SystemExit(1)"
            )
            completed = subprocess.run(
                (sys.executable, "-S", "-c", probe),
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_mpmath_importing_forbidden_module_fails_closed(self) -> None:
        expected = {
            "name": "mpmath",
            "version": "1.3.0",
            "package_root": "mpmath",
            "file_count": 1,
            "total_size_bytes": 1,
            "tree_domain": RUNNER.MPMATH_TREE_DOMAIN,
            "tree_sha256": "1" * 64,
        }
        origin = Path("/nonexistent/site-packages/mpmath/__init__.py")
        fake = types.SimpleNamespace(__version__="1.3.0", __file__=str(origin))

        def poisoned_import(name: str) -> object:
            self.assertEqual(name, "mpmath")
            sys.modules["scipy"] = types.ModuleType("scipy")
            return fake

        try:
            with mock.patch.object(
                RUNNER, "_installed_mpmath_record", return_value=(expected, origin)
            ), mock.patch.object(
                RUNNER.importlib, "import_module", side_effect=poisoned_import
            ):
                with self.assertRaises(RUNNER.HoldoutError):
                    RUNNER._load_mpmath(expected)
        finally:
            sys.modules.pop("scipy", None)

    def test_zero_preserving_hill_case_is_dual_precision_stable(self) -> None:
        if importlib.util.find_spec("mpmath") is None:
            self.skipTest("mpmath is not installed in this test interpreter")
        installed, _ = RUNNER._installed_mpmath_record()
        mpmath = RUNNER._load_mpmath(installed)
        seed = "00" * 32
        interiors = RUNNER._interior_cells(self.protocol, seed)
        roster = next(
            item
            for item in self.protocol["case_roster"]
            if item["case_id"] == "JXHHO-S-N04-HILL"
        )
        cell = RUNNER._base_cell(self.protocol, seed, roster, interiors)
        low = RUNNER._construct_case_at_precision(
            mpmath, self.protocol, seed, roster, cell, 160
        )
        high = RUNNER._construct_case_at_precision(
            mpmath, self.protocol, seed, roster, cell, 220
        )
        RUNNER._compare_precision_results(low, high, roster["case_id"])

    def test_create_only_io_accepts_posix_paths_and_rejects_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "artifact.json"
            RUNNER._write_new_file(output, b"{}\n")
            self.assertEqual(RUNNER._read_regular(output, 16, "artifact"), b"{}\n")
            with self.assertRaises(RUNNER.HoldoutError):
                RUNNER._write_new_file(output, b"{}\n")
            target = root / "target"
            target.write_bytes(b"x")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaises(RUNNER.HoldoutError):
                RUNNER._read_regular(link, 16, "symlink")

    def test_arbitrary_self_hashed_source_registration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source_registration_v1.json"
            core = {
                "schema": RUNNER.SOURCE_REGISTRATION_SCHEMA,
                "holdout_id": "JX_WH_V9_HIERARCHICAL_HOLDOUT_V1",
                "attacker_selected": True,
            }
            value = core | {
                "content_integrity": {
                    "algorithm": "SHA-256",
                    "canonicalization": "SORTED_KEYS_COMPACT_ASCII_EXCLUDING_CONTENT_INTEGRITY",
                    "domain": RUNNER.SOURCE_REGISTRATION_DOMAIN,
                    "content_sha256": RUNNER._domain_hash(
                        RUNNER.SOURCE_REGISTRATION_DOMAIN, core
                    ),
                }
            }
            path.write_bytes(RUNNER._canonical_bytes(value, newline=True))
            with mock.patch.object(RUNNER, "ROOT", Path(directory)), mock.patch.object(
                RUNNER, "SOURCE_REGISTRATION_PATH", path
            ), mock.patch.object(
                RUNNER, "BEACON_RAW_PATH", Path(directory) / "beacon_raw_v1.json"
            ), mock.patch.object(
                RUNNER, "BEACON_BINDING_PATH", Path(directory) / "beacon_binding_v1.json"
            ), mock.patch.object(
                RUNNER, "INPUT_MANIFEST_PATH", Path(directory) / "input_manifest_v1.json"
            ):
                with self.assertRaises(RUNNER.HoldoutError):
                    RUNNER._source_registration(path)

    def test_beacon_binding_requires_success_status_and_retained_bytes(self) -> None:
        registered = self.protocol["future_public_randomness"]
        source_record = {
            "path": "synthetic/source.json",
            "publication_commit": "1" * 40,
            "source_freeze_commit": "2" * 40,
            "size_bytes": 1,
            "sha256": "3" * 64,
            "content_sha256": "4" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "beacon_raw_v1.json"
            source_path = Path(directory) / "source.json"

            def response(status: int, output: str = "ab" * 64) -> bytes:
                return RUNNER._canonical_bytes(
                    {
                        "pulse": {
                            "chainIndex": registered["chain_index"],
                            "timeStamp": registered["pulse_timestamp_utc"],
                            "statusCode": status,
                            "outputValue": output,
                        }
                    },
                    newline=True,
                )

            raw_path.write_bytes(response(1))
            with mock.patch.object(RUNNER, "ROOT", Path(directory)), mock.patch.object(
                RUNNER, "BEACON_RAW_PATH", raw_path
            ), mock.patch.object(
                RUNNER, "_source_registration", return_value=({}, source_record)
            ):
                with self.assertRaises(RUNNER.HoldoutError):
                    RUNNER.create_beacon_binding(raw_path, source_path)
                raw_path.write_bytes(response(0))
                binding = RUNNER.create_beacon_binding(raw_path, source_path)
                RUNNER.validate_beacon_binding(binding, source_path)
                raw_path.write_bytes(response(0, "cd" * 64))
                with self.assertRaises(RUNNER.HoldoutError):
                    RUNNER.validate_beacon_binding(binding, source_path)

    def test_canonical_json_rejects_duplicate_keys_and_nonfinite_values(self) -> None:
        with self.assertRaises(RUNNER.HoldoutError):
            RUNNER._parse_json(b'{"a":1,"a":2}', "duplicate")
        with self.assertRaises(RUNNER.HoldoutError):
            RUNNER._canonical_bytes({"x": float("nan")})


if __name__ == "__main__":
    unittest.main()
