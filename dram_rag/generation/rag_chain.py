from __future__ import annotations

from typing import List

from ..types import RetrievalResult, safe_short
from .llm_clients import LLMClient, NoopLLMClient
from .prompts import RAG_SYSTEM_PROMPT, format_context


def _extractive_answer(question: str, retrieved: List[RetrievalResult]) -> str:
    if not retrieved:
        return "스펙 근거를 찾지 못했습니다. 질문을 더 구체화하거나 다른 키워드로 재시도해야 합니다."

    # Summarize by presenting the most relevant excerpt(s) directly.
    top_text = [r for r in retrieved if r.document.doc_type in ("text", "table")]
    top_img = [r for r in retrieved if r.document.doc_type == "image"]

    lines = []
    lines.append("[Extractive RAG Answer]")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Answer (draft):")
    if top_text:
        lines.append("- 아래 스펙 발췌를 근거로 답변을 구성할 수 있습니다. 현재는 생성 LLM이 비활성화되어 발췌 중심으로 제공합니다.")
    else:
        lines.append("- 텍스트 근거가 부족합니다. 관련 그림/표 문서가 검색되었습니다.")

    lines.append("")
    lines.append("Evidence:")
    for i, r in enumerate(retrieved, start=1):
        d = r.document
        md = d.metadata or {}
        if d.doc_type == "text":
            tag = f"[#%d] TEXT score=%.3f heading=%s chunk_id=%s" % (
                i,
                r.score,
                md.get("heading", ""),
                md.get("chunk_id", ""),
            )
        elif d.doc_type == "table":
            tag = f"[#%d] TABLE score=%.3f heading=%s lines=%s-%s" % (
                i,
                r.score,
                md.get("heading", ""),
                md.get("start_line", ""),
                md.get("end_line", ""),
            )
        else:
            tag = f"[#%d] IMAGE score=%.3f file=%s heading=%s" % (
                i,
                r.score,
                md.get("image_filename", ""),
                md.get("heading", ""),
            )
        lines.append(tag)
        lines.append(safe_short(d.text, 520))
        lines.append("")

    if top_img:
        lines.append("Related Figures:")
        for r in top_img[:4]:
            md = r.document.metadata or {}
            lines.append(f"- {md.get('image_filename','')} (score={r.score:.3f}): {safe_short(md.get('context','') or r.document.text, 160)}")

    return "\n".join(lines).strip()


def generate_answer(question: str, retrieved: List[RetrievalResult], llm: LLMClient) -> str:
    # If LLM is disabled, fall back
    if isinstance(llm, NoopLLMClient):
        return _extractive_answer(question, retrieved)

    context = format_context(retrieved)
    user_prompt = f"""Question:\n{question}\n\nContext Excerpts:\n{context}\n\nWrite the answer following the Output requirements."""
    return llm.generate(system=RAG_SYSTEM_PROMPT, user=user_prompt)
