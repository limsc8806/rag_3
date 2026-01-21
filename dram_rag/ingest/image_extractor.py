from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .md_loader import split_into_sections


_IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")


@dataclass
class ImageRef:
    """Reference to an image in markdown."""

    alt_text: str
    ref_path: str
    resolved_path: str
    line_no: int
    heading_path: List[str]
    context: str


def _normalize_ref_path(raw: str) -> str:
    # Drop fragment or query in markdown links, if any
    raw = raw.split("#", 1)[0]
    raw = raw.split("?", 1)[0]
    return raw.strip().strip('"').strip("'")


def extract_image_refs(
    md_lines: List[str],
    md_path: str | Path,
    images_dir: Optional[str | Path] = None,
    context_window: int = 2,
) -> List[ImageRef]:
    """Extract image references from markdown.

    - md_path: used to resolve relative image links
    - images_dir: if provided, prefer resolving under this directory
    """

    md_path = Path(md_path)
    base_dir = md_path.parent
    images_dir_path = Path(images_dir) if images_dir else None

    # Build heading path per line by reusing the section splitter
    sections = split_into_sections(md_lines)
    line_heading: List[List[str]] = [[] for _ in range(len(md_lines))]
    for s in sections:
        for j in range(s.start_line, min(len(md_lines), s.start_line + len(s.lines))):
            line_heading[j] = s.heading_path

    refs: List[ImageRef] = []
    for i, line in enumerate(md_lines):
        for m in _IMAGE_RE.finditer(line):
            alt = (m.group(1) or "").strip()
            raw_path = _normalize_ref_path(m.group(2) or "")
            if not raw_path:
                continue

            # Resolve
            candidate_paths: List[Path] = []
            rel = Path(raw_path)
            if rel.is_absolute():
                candidate_paths.append(rel)
            else:
                candidate_paths.append((base_dir / rel).resolve())
                if images_dir_path:
                    candidate_paths.append((images_dir_path / rel.name).resolve())
                    candidate_paths.append((images_dir_path / rel).resolve())

            resolved = None
            for p in candidate_paths:
                if p.exists():
                    resolved = str(p)
                    break
            if resolved is None:
                # Fall back to best-effort resolution
                resolved = str(candidate_paths[0])

            # Context around the markdown line
            start = max(0, i - context_window)
            end = min(len(md_lines), i + context_window + 1)
            ctx_lines = [l.strip() for l in md_lines[start:end] if l.strip()]
            context = "\n".join(ctx_lines)

            refs.append(
                ImageRef(
                    alt_text=alt,
                    ref_path=raw_path,
                    resolved_path=resolved,
                    line_no=i,
                    heading_path=line_heading[i] or [],
                    context=context,
                )
            )

    return refs
