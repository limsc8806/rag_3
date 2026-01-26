from __future__ import annotations

import pathlib
import unittest

import yaml

from dram_rag.agent.graph import AdaptiveRAGAgent
from dram_rag.config import load_settings
from dram_rag.index.stores import IndexBundle


class E2EAnswerQualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = load_settings("config/settings.yaml")
        settings.llm.provider = "none"
        settings.generation.grade_mode = "heuristic"
        cls.settings = settings
        cls.bundle = IndexBundle.load(settings.paths.index_dir)
        cls.agent = AdaptiveRAGAgent(settings, cls.bundle)

        q_path = pathlib.Path(__file__).with_name("e2e_questions.yaml")
        data = yaml.safe_load(q_path.read_text(encoding="utf-8")) or {}
        cls.questions = data.get("questions", [])

    def test_e2e_answer_contains_expected_tokens(self) -> None:
        self.assertTrue(self.questions, "No E2E questions loaded.")

        failures = []
        for q in self.questions:
            query = q["query"]
            expected = [t.casefold() for t in q.get("expect_answer_contains", [])]

            state = self.agent.run(query)
            answer = (state.generation or "").casefold()
            if not all(t in answer for t in expected):
                failures.append(
                    f"Query='{query}' missing={ [t for t in expected if t not in answer] }"
                )

        if failures:
            msg = "E2E answer mismatches:\n- " + "\n- ".join(failures)
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
