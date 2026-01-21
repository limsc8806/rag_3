from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)\s*$")


@dataclass
class Section:
    """A markdown section grouped by heading."""

    heading_path: List[str]
    level: int
    start_line: int
    lines: List[str]


def load_markdown_lines(md_path: str | Path) -> List[str]:
    md_path = Path(md_path)
    return md_path.read_text(encoding="utf-8", errors="ignore").splitlines()


def split_into_sections(lines: List[str]) -> List[Section]:
    """Split markdown into sections by headings while tracking heading path."""

    sections: List[Section] = []
    heading_stack: List[str] = []
    level_stack: List[int] = []

    current_lines: List[str] = []
    current_start = 0
    current_level = 0

    def flush():
        nonlocal current_lines, current_start, current_level
        if current_lines:
            sections.append(
                Section(
                    heading_path=heading_stack.copy(),
                    level=current_level,
                    start_line=current_start,
                    lines=current_lines,
                )
            )
            current_lines = []

    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            # Start new section
            flush()

            hashes, title = m.group(1), m.group(2).strip()
            level = len(hashes)

            # Maintain heading stack
            while level_stack and level_stack[-1] >= level:
                level_stack.pop()
                heading_stack.pop()
            level_stack.append(level)
            heading_stack.append(title)

            current_start = i
            current_level = level
            # Keep heading line inside the section content for provenance
            current_lines.append(line)
        else:
            current_lines.append(line)

    flush()

    # If no headings exist, treat whole doc as a single section
    if not sections and lines:
        sections = [Section(heading_path=[], level=0, start_line=0, lines=lines)]

    return sections
