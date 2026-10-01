"""Turn data/resume.md into retrievable chunks.

Chunk boundaries follow the resume's own structure rather than a fixed token window:
one chunk per project, per role, per degree, and one per flat section. A resume is
already authored as semantically complete units, so splitting on `###`/`##` keeps
each project's bullets with its technology list -- which is exactly what a caller
asks about together ("what did he use in that project?").

Oversized entries are split on bullet boundaries only, never mid-sentence.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_CHARS = 1400  # ~350 tokens: comfortably inside one embedding call, still one topic
RETRIEVABLE_FALSE = re.compile(r"retrievable:\s*false", re.I)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section: str          # "Projects", "Experience", ...
    entry: str | None     # "AgentIQ — Multi-Step Agentic Research Assistant", or None
    text: str             # what gets embedded and shown as evidence
    part: int = 1         # >1 only when an oversized entry had to be split
    parts: int = 1

    @property
    def citation(self) -> str:
        """Human-readable provenance, shown in the debug panel -- never spoken."""
        base = f"{self.section} › {self.entry}" if self.entry else self.section
        return f"{base} (part {self.part}/{self.parts})" if self.parts > 1 else base


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def _entry_slug(heading: str) -> str:
    """Short, stable id from an entry heading.

    Headings carry employer and dates ("Machine Learning Intern - Data Engineering,
    Codecasa, India (Mar 2023 - Aug 2023)"); keep only the leading title so chunk ids
    stay readable in eval labels and debug output.
    """
    lead = re.split(r"\s[\u2014\u2013-]\s|,|\(", heading, maxsplit=1)[0]
    return _slug(lead)


def _split_oversized(body: str) -> list[str]:
    """Split on bullet/line boundaries, packing lines until MAX_CHARS."""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    parts: list[str] = []
    current: list[str] = []
    for line in lines:
        candidate = "\n".join(current + [line])
        if current and len(candidate) > MAX_CHARS:
            parts.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        parts.append("\n".join(current))
    return parts or [body]


def parse_resume(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")

    # Track which sections an HTML comment marked as non-retrievable, then drop comments.
    blocked: set[str] = set()
    for comment, following in re.findall(r"(<!--.*?-->)\s*##\s+([^\n]+)", raw, re.S):
        if RETRIEVABLE_FALSE.search(comment):
            blocked.add(following.strip())
    text = re.sub(r"<!--.*?-->", "", raw, flags=re.S)

    chunks: list[Chunk] = []
    # Sections are `## X` up to the next `## ` or EOF.
    for sec_match in re.finditer(r"^##\s+(.+?)\n(.*?)(?=^##\s|\Z)", text, re.S | re.M):
        section = sec_match.group(1).strip()
        body = sec_match.group(2).strip()
        if section in blocked or not body:
            continue

        entries = list(re.finditer(r"^###\s+(.+?)\n(.*?)(?=^###\s|\Z)", body, re.S | re.M))
        if entries:
            for entry_match in entries:
                entry = entry_match.group(1).strip()
                entry_body = entry_match.group(2).strip()
                # The heading carries the dates and employer -- keep it in the embedded
                # text so "when did he work at Codecasa?" can match.
                full = f"{entry}\n{entry_body}"
                pieces = _split_oversized(full) if len(full) > MAX_CHARS else [full]
                for i, piece in enumerate(pieces, start=1):
                    chunks.append(
                        Chunk(
                            chunk_id=f"{_slug(section)}--{_entry_slug(entry)}--{i}",
                            section=section,
                            entry=entry,
                            text=piece if i == 1 else f"{entry}\n{piece}",
                            part=i,
                            parts=len(pieces),
                        )
                    )
        else:
            pieces = _split_oversized(body) if len(body) > MAX_CHARS else [body]
            for i, piece in enumerate(pieces, start=1):
                chunks.append(
                    Chunk(
                        chunk_id=f"{_slug(section)}--{i}",
                        section=section,
                        entry=None,
                        text=f"{section}\n{piece}",
                        part=i,
                        parts=len(pieces),
                    )
                )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Dump chunks to JSON so the knowledge source is inspectable without running the app."""
    path.write_text(
        json.dumps([asdict(c) | {"citation": c.citation} for c in chunks], indent=2),
        encoding="utf-8",
    )
