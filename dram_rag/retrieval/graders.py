from __future__ import annotations

from typing import List

from ..types import RetrievalResult


def filter_relevant(results: List[RetrievalResult], min_relevance_score: float = 0.12) -> List[RetrievalResult]:
    """Heuristic relevance grader.

    In the original Adaptive RAG notebook, this is an LLM-based grader.
    Here we use TF-IDF cosine similarity threshold as a deterministic stand-in.
    """
    return [r for r in results if r.score >= min_relevance_score]
