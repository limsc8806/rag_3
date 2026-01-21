from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """A minimal document type (LangChain Document-like)."""

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: str = "text"  # "text" or "image"


@dataclass
class RetrievalResult:
    document: Document
    score: float


@dataclass
class AgentState:
    question: str
    query: str
    retrieved: List[RetrievalResult] = field(default_factory=list)
    generation: str = ""
    loop_count: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)


def safe_short(text: str, n: int = 240) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"
