from __future__ import annotations

RAG_SYSTEM_PROMPT = """You are a DRAM specification assistant.
Answer strictly using the provided context excerpts.
If the context does not contain enough information, say you cannot verify from the spec.

Output requirements:
1) Provide a concise answer.
2) Provide 'Evidence' section listing the excerpts used, with their metadata (heading, chunk_id, source_md).
3) If any image documents are relevant, provide 'Related Figures' section listing image_filename and why it is relevant.
"""


def format_context(retrieved):
    """Format retrieval results into a context string."""
    parts = []
    for i, r in enumerate(retrieved, start=1):
        d = r.document
        md = d.metadata or {}
        header = f"[#{i}][{d.doc_type.upper()}] score={r.score:.3f} heading={md.get('heading','')!s}"
        parts.append(header + "\n" + d.text)
    return "\n\n".join(parts)
