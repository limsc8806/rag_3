from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import re
import html

from ..types import Document
from .md_loader import load_markdown_lines, split_into_sections
from .image_extractor import extract_image_refs
from .image_captioner import CaptionCache


def _heading_str(heading_path: List[str]) -> str:
    return " > ".join(heading_path) if heading_path else "(root)"


_TABLE_START_RE = re.compile(r"<table\b", re.IGNORECASE)
_TABLE_END_RE = re.compile(r"</table>", re.IGNORECASE)
_PIPE_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")


def _table_html_to_text(table_html: str) -> str:
    """Best-effort HTML table to plain text."""
    s = html.unescape(table_html)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", s, flags=re.IGNORECASE | re.DOTALL)
    if rows:
        lines: List[str] = []
        for row in rows:
            cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, flags=re.IGNORECASE | re.DOTALL)
            clean_cells: List[str] = []
            for cell in cells:
                cell = re.sub(r"<br\s*/?>", "\n", cell, flags=re.IGNORECASE)
                cell = re.sub(r"<[^>]+>", "", cell)
                cell = re.sub(r"\s+", " ", cell).strip()
                clean_cells.append(cell)
            while clean_cells and clean_cells[-1] == "":
                clean_cells.pop()
            if clean_cells:
                lines.append(" | ".join(clean_cells))
        return "\n".join(lines).strip()

    # Fallback for malformed HTML tables
    s = re.sub(r"</tr\s*>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<tr[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</t[dh]\s*>", " | ", s, flags=re.IGNORECASE)
    s = re.sub(r"<t[dh][^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[ \t]*\|[ \t]*", " | ", s)
    return s.strip()


def _extract_table_blocks(md_lines: List[str]) -> Tuple[List[dict], set[int]]:
    """Extract HTML table blocks and return (tables, table_line_ids)."""
    sections = split_into_sections(md_lines)
    line_heading: List[List[str]] = [[] for _ in range(len(md_lines))]
    for s in sections:
        for j in range(s.start_line, min(len(md_lines), s.start_line + len(s.lines))):
            line_heading[j] = s.heading_path

    tables: List[dict] = []
    table_lines: set[int] = set()
    i = 0
    while i < len(md_lines):
        line = md_lines[i]
        if _TABLE_START_RE.search(line):
            start = i
            end = i
            if not _TABLE_END_RE.search(line):
                j = i + 1
                while j < len(md_lines):
                    if _TABLE_END_RE.search(md_lines[j]):
                        end = j
                        break
                    j += 1
                else:
                    end = len(md_lines) - 1
            else:
                end = i

            table_lines.update(range(start, end + 1))
            table_html = "\n".join(md_lines[start : end + 1]).strip()
            tables.append(
                {
                    "start_line": start,
                    "end_line": end,
                    "heading_path": line_heading[start] or [],
                    "html": table_html,
                    "format": "html",
                }
            )
            i = end + 1
            continue
        i += 1

    return tables, table_lines


def _extract_pipe_tables(md_lines: List[str]) -> Tuple[List[dict], set[int]]:
    """Extract markdown pipe tables and return (tables, table_line_ids)."""
    sections = split_into_sections(md_lines)
    line_heading: List[List[str]] = [[] for _ in range(len(md_lines))]
    for s in sections:
        for j in range(s.start_line, min(len(md_lines), s.start_line + len(s.lines))):
            line_heading[j] = s.heading_path

    tables: List[dict] = []
    table_lines: set[int] = set()
    i = 0
    while i + 1 < len(md_lines):
        line = md_lines[i]
        sep = md_lines[i + 1]
        if "|" in line and _PIPE_TABLE_SEP_RE.match(sep):
            start = i
            end = i + 1
            j = i + 2
            while j < len(md_lines):
                if "|" not in md_lines[j] or not md_lines[j].strip():
                    break
                end = j
                j += 1
            table_lines.update(range(start, end + 1))
            tables.append(
                {
                    "start_line": start,
                    "end_line": end,
                    "heading_path": line_heading[start] or [],
                    "md_lines": md_lines[start : end + 1],
                    "format": "pipe",
                }
            )
            i = end + 1
            continue
        i += 1
    return tables, table_lines


def _pipe_table_to_text(lines: List[str]) -> str:
    rows: List[str] = []
    if len(lines) < 2:
        return ""
    for idx, line in enumerate(lines):
        if idx == 1 and _PIPE_TABLE_SEP_RE.match(line):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        while parts and parts[-1] == "":
            parts.pop()
        if parts:
            rows.append(" | ".join(parts))
    return "\n".join(rows).strip()


def chunk_markdown(
    md_path: str | Path,
    images_dir: str | Path | None = None,
    caption_cache_path: str | Path | None = None,
    chunk_size_chars: int = 1400,
    chunk_overlap_chars: int = 250,
) -> Tuple[List[Document], List[Document], List[Document]]:
    """Return (text_documents, image_documents, table_documents).

    - caption_cache_path: optional jsonl cache created by an external vision captioning step
      (see ingest/image_captioner.py for expected format).
    """

    md_path = Path(md_path)
    md_lines = load_markdown_lines(md_path)
    sections = split_into_sections(md_lines)

    caption_cache = None
    if caption_cache_path:
        caption_cache = CaptionCache.load(caption_cache_path)

    # Pre-extract image refs to attach metadata
    img_refs = extract_image_refs(md_lines, md_path=md_path, images_dir=images_dir)

    # Index image refs by line number for fast association
    refs_by_line = {}
    for r in img_refs:
        refs_by_line.setdefault(r.line_no, []).append(r)

    tables_html, html_table_lines = _extract_table_blocks(md_lines)
    tables_pipe, pipe_table_lines = _extract_pipe_tables(md_lines)
    tables = tables_html + tables_pipe
    table_lines = html_table_lines | pipe_table_lines

    text_docs: List[Document] = []
    chunk_id = 0

    for sec in sections:
        sec_text = "\n".join(
            line for idx, line in enumerate(sec.lines, start=sec.start_line) if idx not in table_lines
        ).strip()
        if not sec_text:
            continue

        heading = _heading_str(sec.heading_path)
        # Character-based chunking with overlap
        start = 0
        while start < len(sec_text):
            end = min(len(sec_text), start + chunk_size_chars)
            chunk = sec_text[start:end]

            # Best-effort: attach all image refs within the section
            sec_line_range = range(sec.start_line, sec.start_line + len(sec.lines))
            associated = []
            for ln in sec_line_range:
                for r in refs_by_line.get(ln, []):
                    associated.append(
                        {
                            "filename": Path(r.resolved_path).name,
                            "resolved_path": r.resolved_path,
                            "alt_text": r.alt_text,
                            "line_no": r.line_no,
                        }
                    )

            text_docs.append(
                Document(
                    text=chunk,
                    doc_type="text",
                    metadata={
                        "source_md": str(md_path),
                        "heading": heading,
                        "heading_path": sec.heading_path,
                        "section_start_line": sec.start_line,
                        "chunk_id": chunk_id,
                        "char_start": start,
                        "char_end": end,
                        "image_refs": associated,
                    },
                )
            )
            chunk_id += 1

            if end == len(sec_text):
                break
            start = max(0, end - chunk_overlap_chars)

    # Build image docs (caption-lite) using: caption_cache -> alt text -> filename
    image_docs: List[Document] = []
    for r in img_refs:
        filename = Path(r.resolved_path).name

        caption = None
        if caption_cache:
            caption = caption_cache.get(r.resolved_path)
        if not caption:
            caption = (r.alt_text or "").strip()
        if not caption:
            caption = f"Image {filename}"

        img_text = "\n".join(
            [
                f"[IMAGE] {filename}",
                f"CAPTION: {caption}",
                f"HEADING: {_heading_str(r.heading_path)}",
                "CONTEXT:\n" + (r.context or ""),
            ]
        )
        image_docs.append(
            Document(
                text=img_text,
                doc_type="image",
                metadata={
                    "source_md": str(md_path),
                    "heading_path": r.heading_path,
                    "heading": _heading_str(r.heading_path),
                    "line_no": r.line_no,
                    "image_ref_path": r.ref_path,
                    "image_resolved_path": r.resolved_path,
                    "image_filename": filename,
                    "alt_text": r.alt_text,
                    "caption": caption,
                    "context": r.context,
                },
            )
        )

    table_docs: List[Document] = []
    for t in tables:
        if t.get("format") == "pipe":
            table_text = _pipe_table_to_text(t["md_lines"])
        else:
            table_text = _table_html_to_text(t["html"])
        if not table_text:
            continue
        table_docs.append(
            Document(
                text="\n".join(
                    [
                        "[TABLE]",
                        f"HEADING: {_heading_str(t['heading_path'])}",
                        table_text,
                    ]
                ),
                doc_type="table",
                metadata={
                    "source_md": str(md_path),
                    "heading_path": t["heading_path"],
                    "heading": _heading_str(t["heading_path"]),
                    "start_line": t["start_line"],
                    "end_line": t["end_line"],
                    "raw_html": t.get("html", ""),
                    "raw_md": "\n".join(t.get("md_lines", [])) if t.get("format") == "pipe" else "",
                },
            )
        )

    return text_docs, image_docs, table_docs
