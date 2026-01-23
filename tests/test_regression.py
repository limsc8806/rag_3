from __future__ import annotations

import pathlib
import unittest
import warnings

import yaml

from dram_rag.config import load_settings
from dram_rag.index.stores import IndexBundle
from dram_rag.retrieval.retrievers import retrieve


class RetrievalRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings(
            "ignore",
            message="Trying to unpickle estimator",
            category=UserWarning,
        )
        settings = load_settings("config/settings.yaml")
        settings.llm.provider = "none"
        cls.settings = settings
        cls.bundle = IndexBundle.load(settings.paths.index_dir)

        q_path = pathlib.Path(__file__).with_name("regression_questions.yaml")
        data = yaml.safe_load(q_path.read_text(encoding="utf-8")) or {}
        cls.questions = data.get("questions", [])

    def test_retrieval_expectations(self) -> None:
        self.assertTrue(self.questions, "No regression questions loaded.")

        failures = []
        for q in self.questions:
            query = q["query"]
            expected_headings = [h.casefold() for h in q.get("expect_heading_contains", [])]
            expected_doc_types = [t.casefold() for t in q.get("expect_doc_types", [])]
            expected_text = [t.casefold() for t in q.get("expect_text_contains", [])]

            results = retrieve(
                self.bundle,
                query,
                top_k_text=self.settings.retrieval.top_k_text,
                top_k_img=self.settings.retrieval.top_k_img,
                merge_top_k=self.settings.retrieval.merge_top_k,
            )

            matched = False
            for r in results:
                heading = (r.document.metadata or {}).get("heading", "")
                heading_cf = heading.casefold()
                doc_type = r.document.doc_type.casefold()
                text_cf = (r.document.text or "").casefold()

                if expected_doc_types and doc_type not in expected_doc_types:
                    continue
                if expected_headings and not any(h in heading_cf for h in expected_headings):
                    continue
                if expected_text and not all(t in text_cf for t in expected_text):
                    continue
                matched = True
                break

            if not matched:
                failures.append(
                    f"Query='{query}' expected headings={expected_headings} "
                    f"doc_types={expected_doc_types} text_contains={expected_text}"
                )

        if failures:
            msg = "Regression retrieval mismatches:\n- " + "\n- ".join(failures)
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
