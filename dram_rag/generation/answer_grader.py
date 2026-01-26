from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .llm_clients import LLMClient, NoopLLMClient


@dataclass
class GenerationGrade:
    supported: bool
    useful: bool
    reason: str


_LLM_GRADE_PROMPT = """You are a strict grader for DRAM specification QA.
Determine if the answer is supported by the provided content and is useful.

Respond in the following format:
supported: yes|no
useful: yes|no
reason: <short reason>
"""


def grade_generation(
    question: str,
    answer: str,
    *,
    mode: str = "heuristic",
    llm: Optional[LLMClient] = None,
) -> GenerationGrade:
    """Heuristic or LLM-based grader.

    The original Adaptive RAG notebook uses LLM-based grading.
    Here we do deterministic checks:
      - supported: has an Evidence section or citations-style markers
      - useful: overlaps with at least one content token from the question
    """

    if mode.lower() == "llm" and llm is not None and not isinstance(llm, NoopLLMClient):
        prompt = f"Question:\n{question}\n\nAnswer:\n{answer}"
        resp = llm.generate(system=_LLM_GRADE_PROMPT, user=prompt).strip()
        supported = bool(re.search(r"supported:\s*yes", resp, flags=re.IGNORECASE))
        useful = bool(re.search(r"useful:\s*yes", resp, flags=re.IGNORECASE))
        reason_match = re.search(r"reason:\s*(.*)", resp, flags=re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else "LLM"
        if "supported:" in resp.lower() and "useful:" in resp.lower():
            return GenerationGrade(supported=supported, useful=useful, reason=reason)

    a = answer or ""
    supported = bool(re.search(r"\bEvidence\b", a, flags=re.IGNORECASE)) or "[#" in a

    # Token overlap
    q_tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9_]+", question) if len(t) >= 3]
    a_low = a.lower()
    useful = any(t in a_low for t in q_tokens[:12]) if q_tokens else True

    if not supported:
        reason = "Answer does not appear to cite evidence." 
    elif not useful:
        reason = "Answer does not appear to address the question terms." 
    else:
        reason = "OK"

    return GenerationGrade(supported=supported, useful=useful, reason=reason)
