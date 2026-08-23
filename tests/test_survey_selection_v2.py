from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from jxplanetx.survey_selection import SurveySelectionError
from jxplanetx.survey_selection_v2 import parse_ossos_tracked_file
from runs.survey_selection_o1.run_experiment_v2 import (
    _driver_totals,
    _validate_batch_size,
)


TRACKED_ROW = (
    "151.5858 0.8710 10.4206 44.9099 176.7582 358.1715 11.14 "
    "19.5587 19.5737 18.6284 23.80 10.78 0.00 c0000016714\n"
)


class ActualTrackedAdapterTests(unittest.TestCase):
    def test_actual_fourteen_field_row_maps_documented_driver_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SimulTrack.dat"
            path.write_text("# tracked header\n" + TRACKED_ROW, encoding="utf-8")
            rows = parse_ossos_tracked_file(path, "correct", 0)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["object_id"], "c0000016714")
            self.assertEqual(rows[0]["a_AU"], 151.5858)
            self.assertEqual(rows[0]["q_AU"], 19.5587)
            self.assertEqual(rows[0]["i_deg"], 10.4206)
            self.assertEqual(rows[0]["H_r"], 11.14)
            self.assertEqual(rows[0]["r_AU"], 19.5737)
            self.assertEqual(rows[0]["m_r"], 23.8)

    def test_detected_twenty_field_row_is_rejected_at_tracked_boundary(self) -> None:
        detected = (
            "120 0.8 15 24 25 2 3 4 23.1 7.2 0 4 24 23 7.1 0.8 10 -5 "
            "2013AE c0000000001\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SimulTrack.dat"
            path.write_text(detected, encoding="utf-8")
            with self.assertRaisesRegex(SurveySelectionError, "exactly 14"):
                parse_ossos_tracked_file(path, "correct", 0)

    def test_identity_must_match_declared_model_and_seed_block(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SimulTrack.dat"
            path.write_text(TRACKED_ROW.replace("c0000016714", "w0000016714"), encoding="utf-8")
            with self.assertRaisesRegex(SurveySelectionError, "identity does not match"):
                parse_ossos_tracked_file(path, "correct", 0)

    def test_extra_comment_tokens_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SimulTrack.dat"
            path.write_text(TRACKED_ROW.rstrip() + " extra\n", encoding="utf-8")
            with self.assertRaisesRegex(SurveySelectionError, "exactly 14"):
                parse_ossos_tracked_file(path, "correct", 0)


class LegacyDriverBoundaryTests(unittest.TestCase):
    def test_batch_size_avoids_exact_read_model_chunk(self) -> None:
        self.assertEqual(_validate_batch_size(100_001), 100_001)
        with self.assertRaisesRegex(ValueError, "must not be divisible by 100"):
            _validate_batch_size(100_000)

    def test_detected_footer_totals_are_strictly_parsed(self) -> None:
        text = (
            "# header\n"
            "# Total number of objects:       100001.\n"
            "# Number of detections:            4\n"
            "# Number of tracked objects:       4\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SimulDetect.dat"
            path.write_text(text, encoding="utf-8")
            self.assertEqual(
                _driver_totals(path),
                {"objects": 100001, "detections": 4, "tracked": 4},
            )

    def test_inconsistent_detected_footer_is_rejected(self) -> None:
        text = (
            "# Total number of objects:       100001.\n"
            "# Number of detections:            3\n"
            "# Number of tracked objects:       4\n"
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "SimulDetect.dat"
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "inconsistent footer totals"):
                _driver_totals(path)


if __name__ == "__main__":
    unittest.main()

