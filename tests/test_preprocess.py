"""Tests for section labels recovered from irregular source record boundaries."""

from __future__ import annotations

import unittest

from src.ingestion.preprocess import normalize, section_identifier


class PreprocessSectionIdentifierTests(unittest.TestCase):
    def test_exception_continuation_uses_prior_explicit_section_label(self) -> None:
        section = {"section_content": "Exception 1.-Culpable homicide is not murder."}
        self.assertEqual(section_identifier(section, 342, parent_section="300"), "300 (Exception 1)")

    def test_leading_dot_before_section_number_is_accepted(self) -> None:
        self.assertEqual(section_identifier({"section_content": ".301. If a person..."}, 343), "301")

    def test_normalize_keeps_source_positions_and_parent_exception_label(self) -> None:
        acts = [{
            "source_file": "act-print-11.json",
            "act_title": "Penal Code",
            "sections": [
                {"section_content": "300. Culpable homicide is murder."},
                {"section_content": "Exception 1.-Culpable homicide is not murder."},
                {"section_content": ".301. A person causes death."},
            ],
        }]
        records = normalize(acts, "test.json")
        self.assertEqual([record["section_id"] for record in records], ["300", "300 (Exception 1)", "301"])
        self.assertEqual([record["section_position"] for record in records], [1, 2, 3])
        self.assertEqual(records[1]["chunk_id"], "act_act-print-11_section_0002")


if __name__ == "__main__":
    unittest.main()
