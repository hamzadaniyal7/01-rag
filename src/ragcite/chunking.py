"""Recursive, boundary-aware chunking that keeps track of where text came from.

Why not "split every 512 tokens"? Because a chunk that starts mid-sentence
retrieves badly and reads worse once it lands in the prompt. The splitter here
walks a list of separators from coarse to fine (blank lines -> newlines ->
sentences -> words) and only falls back to a hard cut when a single span has no
breakpoint at all.

Two invariants the rest of the pipeline relies on:

  * every chunk carries `char_start`/`char_end` into the concatenated document
    text, so its page (PDF) or line range (text) can always be recovered;
  * chunks are emitted in reading order with a stable `ordinal`, which is what
    makes neighbour expansion in retrieval.py possible.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from ragcite.types import Chunk, Document, Page

# ~4 characters per token is a good enough rule of thumb for English prose.
# It is only used to turn a token budget into a character budget at split time;
# the *real* token count of the final prompt is measured with the Anthropic
# token-counting endpoint (see ragcite.tokens).
CHARS_PER_TOKEN = 4

SEPARATORS: tuple[str, ...] = ("\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")

PAGE_JOINER = "\n\n"


@dataclass(slots=True)
class FlatText:
    """Document text concatenated into one string, with a map back to pages."""

    text: str
    # Sorted start offsets of each page within `text`, parallel to page_numbers.
    page_starts: list[int]
    page_numbers: list[int | None]
    # Sorted start offsets of each line, for unpaginated sources.
    line_starts: list[int]

    def page_at(self, offset: int) -> int | None:
        if not self.page_starts:
            return None
        idx = bisect.bisect_right(self.page_starts, offset) - 1
        idx = max(0, min(idx, len(self.page_numbers) - 1))
        return self.page_numbers[idx]

    def line_at(self, offset: int) -> int:
        return bisect.bisect_right(self.line_starts, offset)


def flatten(pages: list[Page]) -> FlatText:
    parts: list[str] = []
    page_starts: list[int] = []
    page_numbers: list[int | None] = []
    cursor = 0
    for page in pages:
        page_starts.append(cursor)
        page_numbers.append(page.number)
        parts.append(page.text)
        cursor += len(page.text) + len(PAGE_JOINER)
    text = PAGE_JOINER.join(parts)
    line_starts = [0] + [i + 1 for i, ch in enumerate(text) if ch == "\n"]
    return FlatText(
        text=text, page_starts=page_starts, page_numbers=page_numbers, line_starts=line_starts
    )


def chunk_document(
    doc: Document,
    pages: list[Page],
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_size: int = 48,
) -> list[Chunk]:
    """Split one document into overlapping, location-aware chunks."""
    flat = flatten(pages)
    target = chunk_size * CHARS_PER_TOKEN
    overlap = chunk_overlap * CHARS_PER_TOKEN
    minimum = min_chunk_size * CHARS_PER_TOKEN

    spans = _split_span(flat.text, 0, len(flat.text), target)
    spans = _merge_small(spans, flat.text, minimum, target)
    spans = _add_overlap(spans, flat.text, len(flat.text), overlap)

    chunks: list[Chunk] = []
    for ordinal, (start, end) in enumerate(spans):
        body = flat.text[start:end].strip()
        if not body:
            continue
        # Re-anchor offsets onto the stripped body so locators stay honest.
        lead = len(flat.text[start:end]) - len(flat.text[start:end].lstrip())
        real_start = start + lead
        real_end = real_start + len(body)
        page_start = flat.page_at(real_start)
        page_end = flat.page_at(max(real_start, real_end - 1))
        paginated = page_start is not None
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{ordinal:05d}",
                doc_id=doc.doc_id,
                ordinal=ordinal,
                text=body,
                char_start=real_start,
                char_end=real_end,
                page_start=page_start,
                page_end=page_end,
                line_start=None if paginated else flat.line_at(real_start),
                line_end=None if paginated else flat.line_at(max(real_start, real_end - 1)),
                doc_path=doc.path,
                doc_title=doc.title,
            )
        )
    return chunks


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _split_span(text: str, start: int, end: int, target: int) -> list[tuple[int, int]]:
    """Recursively break [start, end) into spans of at most `target` chars."""
    if end - start <= target:
        return [(start, end)]

    for sep in SEPARATORS:
        cut = _best_cut(text, start, end, target, sep)
        if cut is not None:
            left = (
                [(start, cut)] if cut - start <= target else _split_span(text, start, cut, target)
            )
            return left + _split_span(text, cut, end, target)

    # No separator anywhere (e.g. a base64 blob): hard-cut on the budget.
    return [
        (start, start + target),
        *_split_span(text, start + target, end, target),
    ]


def _best_cut(text: str, start: int, end: int, target: int, sep: str) -> int | None:
    """Last occurrence of `sep` inside the budget, returned as an absolute offset."""
    window_end = min(end, start + target)
    idx = text.rfind(sep, start, window_end)
    if idx <= start:
        return None
    cut = idx + len(sep)
    if cut >= end or cut - start < len(sep) + 1:
        return None
    return cut


def _merge_small(
    spans: list[tuple[int, int]], text: str, minimum: int, target: int
) -> list[tuple[int, int]]:
    """Fold runt chunks (headings, stray lines) into their neighbour."""
    out: list[tuple[int, int]] = []
    for span in spans:
        if out:
            prev_start, prev_end = out[-1]
            cur_len = len(text[span[0] : span[1]].strip())
            prev_len = prev_end - prev_start
            if cur_len < minimum and prev_len + (span[1] - span[0]) <= target * 1.3:
                out[-1] = (prev_start, span[1])
                continue
        out.append(span)
    return out


def _add_overlap(
    spans: list[tuple[int, int]], text: str, total: int, overlap: int
) -> list[tuple[int, int]]:
    """Extend each chunk backwards so adjacent chunks share context.

    Overlap exists so a fact that straddles a boundary is retrievable from at
    least one chunk in full. Only the start moves -- extending both ends would
    double the duplicated text for no extra recall.
    """
    if overlap <= 0:
        return spans
    out: list[tuple[int, int]] = []
    for i, (start, end) in enumerate(spans):
        new_start = start
        if i > 0:
            new_start = _snap_forward(text, max(0, start - overlap), start)
        out.append((new_start, min(end, total)))
    return out


def _snap_forward(text: str, start: int, limit: int) -> int:
    """Move `start` forward to the cleanest boundary before `limit`.

    A chunk that opens mid-word ("ms marked final sale") retrieves badly and
    looks broken in a citation, and one that opens mid-sentence reads badly in
    the prompt. Prefer a sentence boundary inside the overlap window, fall back
    to a word boundary, and only keep the raw offset if the window has neither.
    """
    for terminator in (". ", "? ", "! ", "\n"):
        idx = text.find(terminator, start, limit)
        if idx != -1:
            return idx + len(terminator)
    space = text.find(" ", start, limit)
    return space + 1 if space != -1 else start
