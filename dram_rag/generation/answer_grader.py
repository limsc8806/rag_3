from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class GenerationGrade:
    supported: bool
    useful: bool
    reason: str


def grade_generation(question: str, answer: str) -> GenerationGrade:
    """Heuristic grader.

    The original Adaptive RAG notebook uses LLM-based grading.
    Here we do deterministic checks:
      - supported: has an Evidence section or citations-style markers
      - useful: overlaps with at least one content token from the question
    """

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
