"""Core data structures.

Everything that flows through the pipeline is a plain dataclass so it is easy to
print, diff in tests, and serialise to JSON. There is no ORM and no framework
object graph -- a chunk is a chunk.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Page:
    """One page of a source document. `number` is 1-indexed, or None for
    formats that have no pagination (markdown, plain text)."""

    number: int | None
    text: str


@dataclass(slots=True)
class Document:
    """A source file as it was found on disk."""

    doc_id: str
    path: str
    title: str
    n_pages: int
    content_hash: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    """A retrievable span of a document.

    `page_start`/`page_end` are what make a citation trustworthy: they are
    derived from the real page boundaries of the source, not guessed after the
    fact. For unpaginated formats they are None and `line_start`/`line_end`
    carry the location instead.
    """

    chunk_id: str
    doc_id: str
    ordinal: int
    text: str
    char_start: int
    char_end: int
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    # Denormalised for display; filled in when chunks are read back out.
    doc_path: str = ""
    doc_title: str = ""

    @property
    def locator(self) -> str:
        """Human-readable position inside the source, e.g. 'p.3' or 'L12-40'."""
        if self.page_start is not None:
            if self.page_end and self.page_end != self.page_start:
                return f"pp.{self.page_start}-{self.page_end}"
            return f"p.{self.page_start}"
        if self.line_start is not None:
            if self.line_end and self.line_end != self.line_start:
                return f"L{self.line_start}-{self.line_end}"
            return f"L{self.line_start}"
        return f"#{self.ordinal}"


@dataclass(slots=True)
class Hit:
    """A chunk the retriever thinks is relevant, plus the evidence for why."""

    chunk: Chunk
    score: float
    dense_score: float | None = None
    lexical_score: float | None = None
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None

    def why(self) -> str:
        bits = []
        if self.dense_rank is not None:
            bits.append(f"dense#{self.dense_rank}")
        if self.lexical_rank is not None:
            bits.append(f"bm25#{self.lexical_rank}")
        if self.rerank_score is not None:
            bits.append(f"rerank={self.rerank_score:.2f}")
        return "+".join(bits) or "n/a"


@dataclass(slots=True)
class Citation:
    """A resolved `[n]` marker from the model's answer."""

    marker: int
    chunk_id: str
    path: str
    title: str
    locator: str

    def render(self) -> str:
        return f"[{self.marker}] {self.title} {self.locator}  ({self.path})"


@dataclass(slots=True)
class Answer:
    """The end of the pipeline."""

    question: str
    text: str
    citations: list[Citation]
    hits: list[Hit]
    abstained: bool = False
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    # Markers the model used that did not map to a retrieved chunk. Non-empty
    # means the answer cited something that was never in its context -- the
    # clearest signal of a hallucinated citation.
    dangling_markers: list[int] = field(default_factory=list)

    def render(self) -> str:
        out = [self.text.strip()]
        if self.citations:
            out.append("")
            out.append("Sources:")
            out.extend("  " + c.render() for c in self.citations)
        if self.dangling_markers:
            out.append("")
            out.append(
                "  !! answer cited markers with no matching source: "
                + ", ".join(f"[{m}]" for m in self.dangling_markers)
            )
        return "\n".join(out)

    def to_json(self, indent: int | None = 2) -> str:
        payload = {
            "question": self.question,
            "answer": self.text,
            "abstained": self.abstained,
            "model": self.model,
            "usage": self.usage,
            "citations": [asdict(c) for c in self.citations],
            "dangling_markers": self.dangling_markers,
            "retrieved": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "path": h.chunk.doc_path,
                    "locator": h.chunk.locator,
                    "score": round(h.score, 6),
                    "why": h.why(),
                }
                for h in self.hits
            ],
        }
        return json.dumps(payload, indent=indent)
