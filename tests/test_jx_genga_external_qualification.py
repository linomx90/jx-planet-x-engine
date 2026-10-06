from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from benchmarks.jx_genga_external_qualification import (
    Body,
    GENGA_BINARY_DTYPE,
    GENGA_DAY_UNIT,
    GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN,
    JX_DAYS_PER_YEAR,
    JX_SOURCE_BODY_IDS,
    JX_TRACER_ID_OFFSET,
    TOY_SOURCE_IDS,
    TOY_PARTICLE_MINIMUM_MASS,
    _toy_sources,
    _genga_command,
    _artifact_inventory,
    _runtime_identity,
    _verify_artifact_inventory,
    adapt_jx_bodies,
    gm_au3_per_year2_to_msun,
    pairwise_source_phase_space,
    read_binary_output,
    render_initial_conditions,
    render_param,
    terminal_success,
    velocity_au_per_year_to_genga,
)


class GengaUnitConversionTests(unittest.TestCase):
    def test_mass_conversion_round_trips_def_kg(self) -> None:
        gm = GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN * JX_DAYS_PER_YEAR**2
        self.assertEqual(float(gm_au3_per_year2_to_msun(gm)), 1.0)

    def test_velocity_conversion_round_trips_day_unit(self) -> None:
        physical = 2.75
        stored = float(velocity_au_per_year_to_genga(physical))
        self.assertAlmostEqual(
            stored * GENGA_DAY_UNIT * JX_DAYS_PER_YEAR, physical, places=15
        )


class GengaRenderingTests(unittest.TestCase):
    def test_runtime_identity_detects_engine_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            engine = Path(temporary) / "source" / "genga"
            engine.parent.mkdir()
            engine.write_bytes(b"engine-v1")
            before = _runtime_identity(engine)
            engine.write_bytes(b"engine-v2")
            after = _runtime_identity(engine)
        self.assertNotEqual(before["engine"], after["engine"])
        self.assertEqual(before["runner"], after["runner"])

    def test_device_argument_is_last_due_to_genga_argv_parser(self) -> None:
        self.assertEqual(
            _genga_command(Path("/engine"), 0, 64, 32),
            ["/engine", "-I", "64", "-R", "32", "-dev", "0"],
        )

    def test_every_toy_source_is_above_finite_tp_threshold(self) -> None:
        self.assertTrue(
            all(
                body.mass_msun > TOY_PARTICLE_MINIMUM_MASS
                for body in _toy_sources()
            )
        )

    def test_initial_conditions_have_explicit_ids_and_preserve_zero_mass(self) -> None:
        text = render_initial_conditions(
            (
                Body(7, 1.0e-3, 1.0e-12, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
                Body(9, 0.0, 1.0e-12, (7.0, 8.0, 9.0), (0.1, 0.2, 0.4)),
            )
        )
        rows = text.splitlines()
        self.assertTrue(rows[0].startswith("7 "))
        self.assertTrue(rows[1].startswith("9 0 "))
        self.assertEqual(float(rows[1].split()[-1]), 0.4)

    def test_param_freezes_binary_serial_double_precision_configuration(self) -> None:
        text = render_param(
            output_name="unit",
            input_name="initial.dat",
            central_mass_msun=1.0,
            use_test_particles=1,
            particle_minimum_mass=0.0,
            integration_steps=64,
            timestep_days=6.0,
            coordinate_interval=32,
            energy_interval=32,
            minimum_massive_bodies=2,
            outer_radius_au=10_000.0,
        )
        self.assertIn("Use output binary files = 1", text)
        self.assertIn("Serial Grouping = 1", text)
        self.assertIn("Do kernel tuning = 0", text)
        self.assertIn("Do Kick in single precision = 0", text)
        self.assertIn("Symplectic recursion levels = 1", text)
        self.assertIn("Outer truncation radius = 10000", text)
        self.assertNotIn("Outer truncation radius = 50\n", text)

    def test_terminal_success_requires_both_exit_zero_and_message(self) -> None:
        self.assertTrue(terminal_success(0, "GENGA terminated successfully", ""))
        self.assertFalse(terminal_success(0, "", ""))
        self.assertFalse(terminal_success(1, "GENGA terminated successfully", ""))


class GengaBinaryOutputTests(unittest.TestCase):
    def test_dtype_is_unaligned_140_byte_wire_record(self) -> None:
        self.assertEqual(GENGA_BINARY_DTYPE.itemsize, 140)

    def test_binary_reader_and_pairwise_state_are_translation_invariant(self) -> None:
        rows = np.zeros(2, dtype=GENGA_BINARY_DTYPE)
        rows["i"] = TOY_SOURCE_IDS
        rows["x"] = (3.0, -2.0)
        rows["y"] = (4.0, 1.0)
        rows["z"] = (5.0, 8.0)
        rows["vx"] = (0.1, 0.2)
        rows["vy"] = (0.3, 0.5)
        rows["vz"] = (0.7, 0.9)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Out.bin"
            rows.tofile(path)
            decoded = read_binary_output(path)
        reference = pairwise_source_phase_space(decoded, TOY_SOURCE_IDS)
        translated = decoded.copy()
        for field, offset in zip(("x", "y", "z", "vx", "vy", "vz"), (11, 12, 13, 14, 15, 16)):
            translated[field] += offset
        np.testing.assert_allclose(
            reference,
            pairwise_source_phase_space(translated, TOY_SOURCE_IDS),
            rtol=0.0,
            atol=1.0e-14,
        )

    def test_binary_reader_rejects_partial_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.bin"
            path.write_bytes(b"not-a-record")
            with self.assertRaisesRegex(ValueError, "record size"):
                read_binary_output(path)

    def test_prepared_artifact_manifest_rejects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = root / "input.dat"
            payload.write_text("frozen\n", encoding="utf-8")
            manifest = root / "ARTIFACTS.json"
            import json

            manifest.write_text(
                json.dumps(_artifact_inventory(root), sort_keys=True),
                encoding="utf-8",
            )
            _verify_artifact_inventory(root, manifest)
            payload.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact .* changed"):
                _verify_artifact_inventory(root, manifest)


class JxAdapterTests(unittest.TestCase):
    def test_adapter_omits_primary_and_assigns_massless_explicit_ids(self) -> None:
        one_msun_gm = GENGA_DEF_KG_AU3_PER_DAY2_PER_MSUN * JX_DAYS_PER_YEAR**2
        arrays = {
            "source_positions": np.asarray(
                [[1.0, 2.0, 3.0]] + [[float(i), 0.0, 0.0] for i in range(2, 7)]
            ),
            "source_velocities": np.asarray(
                [[0.5, 1.0, 1.5]] + [[0.5, float(i), 1.5] for i in range(2, 7)]
            ),
            "source_gm": np.asarray([one_msun_gm] + [one_msun_gm * 1.0e-3] * 5),
            "tracer_positions": np.asarray([[11.0, 12.0, 13.0], [21.0, 22.0, 23.0]]),
            "tracer_velocities": np.asarray([[1.5, 2.0, 2.5], [2.5, 3.0, 3.5]]),
            "tracer_indices": np.asarray([4, 8], dtype=np.uint64),
        }
        central, sources, tracers = adapt_jx_bodies(arrays, 2)
        self.assertEqual(central, 1.0)
        self.assertEqual(tuple(body.body_id for body in sources), JX_SOURCE_BODY_IDS)
        self.assertEqual(tuple(body.body_id for body in tracers), (JX_TRACER_ID_OFFSET + 4, JX_TRACER_ID_OFFSET + 8))
        self.assertTrue(all(body.mass_msun == 0.0 for body in tracers))
        self.assertEqual(sources[0].position_au, (1.0, -2.0, -3.0))
        self.assertEqual(tracers[0].position_au, (10.0, 10.0, 10.0))

    def test_adapter_rejects_non_prefix_count(self) -> None:
        arrays = {
            "source_positions": np.zeros((6, 3)),
            "source_velocities": np.zeros((6, 3)),
            "source_gm": np.ones(6),
            "tracer_positions": np.zeros((1, 3)),
            "tracer_velocities": np.zeros((1, 3)),
            "tracer_indices": np.zeros(1, dtype=np.uint64),
        }
        with self.assertRaisesRegex(ValueError, "prefix_count"):
            adapt_jx_bodies(arrays, 0)


if __name__ == "__main__":
    unittest.main()
