from __future__ import annotations

import unittest

from jxplanetx.cli import _survey_selection_finalizer
from jxplanetx.survey_selection_v2 import finalize_survey_selection as finalize_v2
from jxplanetx.survey_selection_v3 import finalize_survey_selection as finalize_v3
from jxplanetx.survey_selection_v4 import (
    EXPERIMENT_ID,
    finalize_survey_selection as finalize_v4,
)


class SurveySelectionCliDispatchTests(unittest.TestCase):
    def test_v4_contract_uses_independent_finalizer(self) -> None:
        self.assertIs(_survey_selection_finalizer({"experiment_id": EXPERIMENT_ID}), finalize_v4)

    def test_v3_contract_uses_corrective_finalizer(self) -> None:
        contract = {"experiment_id": "jx-o1-ossos-b-telescope-selection-v3-exact-zeta-corrective-replay"}
        self.assertIs(_survey_selection_finalizer(contract), finalize_v3)

    def test_earlier_contract_uses_v2_compatible_finalizer(self) -> None:
        self.assertIs(_survey_selection_finalizer({"experiment_id": "jx-o1-v2"}), finalize_v2)


if __name__ == "__main__":
    unittest.main()
