from __future__ import annotations

import re
from typing import List, Optional

from ..generation.llm_clients import LLMClient, NoopLLMClient


_LLM_REWRITE_PROMPT = """You are a query rewriting assistant for DRAM specifications.
Rewrite the user's question to improve retrieval recall and precision.

Rules:
- Keep key technical terms (e.g., tRCD, RTT_NOM, MRxx, CA[13:0]) unchanged.
- Add 2-5 concise keywords that likely appear in the spec.
- Do NOT answer the question.
- Output only the rewritten query, no extra text.
"""


def rewrite_query(
    original_question: str,
    prev_query: str | None = None,
    *,
    mode: str = "heuristic",
    llm: Optional[LLMClient] = None,
) -> str:
    """Query rewriter: heuristic or LLM-based.

    The Adaptive RAG notebook uses an LLM to rewrite the query when retrieval fails.
    Here we do deterministic rewriting that tends to help technical specs.
    """

    if mode.lower() == "llm" and llm is not None and not isinstance(llm, NoopLLMClient):
        base = (prev_query or original_question).strip()
        prompt = f"Question: {original_question}\nCurrent query: {base}"
        rewritten = llm.generate(system=_LLM_REWRITE_PROMPT, user=prompt).strip()
        if rewritten:
            return rewritten

    q = (prev_query or original_question).strip()

    # If it already contains strong spec keywords, just append mild expansion
    expansions: List[str] = []

    # DRAM timing params (tRCD, tRP, tRAS...) tend to be case-sensitive tokens
    if re.search(r"\bt[A-Za-z0-9]+\b", q):
        expansions.extend(["definition", "constraint", "timing parameter"])

    # If user asks 'what is', add spec context
    if any(k in q.lower() for k in ["what is", "meaning", "정의", "뭐야", "무엇"]):
        expansions.extend(["specification", "definition"])

    # If the query is very short, broaden with DRAM/DDR keywords
    if len(q.split()) <= 3:
        expansions.extend(["DRAM", "DDR", "LPDDR", "JEDEC", "spec"])

    # Remove duplicates while keeping order
    seen = set()
    unique = []
    for e in expansions:
        if e.lower() not in seen:
            unique.append(e)
            seen.add(e.lower())

    if unique:
        return q + " " + " ".join(unique)
    return q
