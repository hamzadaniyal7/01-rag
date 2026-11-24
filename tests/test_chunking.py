from __future__ import annotations

from ragcite.chunking import CHARS_PER_TOKEN, chunk_document
from ragcite.types import Document, Page

DOC = Document(doc_id="d1", path="x.pdf", title="X", n_pages=3, content_hash="h")


def _pages(*texts: str) -> list[Page]:
    return [Page(number=i, text=t) for i, t in enumerate(texts, 1)]


def test_chunks_cover_every_page_and_record_it():
    pages = _pages(
        "Page one talks about refunds. " * 20,
        "Page two talks about shipping. " * 20,
        "Page three talks about security. " * 20,
    )
    chunks = chunk_document(DOC, pages, chunk_size=64, chunk_overlap=0)

    assert len(chunks) > 3
    assert all(c.page_start is not None for c in chunks)
    # every source page is represented by at least one chunk
    assert {1, 2, 3} <= {c.page_start for c in chunks}
    # ordinals are contiguous and in reading order
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_page_attribution_is_correct_not_guessed():
    pages = _pages("alpha " * 200, "bravo " * 200, "charlie " * 200)
    chunks = chunk_document(DOC, pages, chunk_size=48, chunk_overlap=0)
    for chunk in chunks:
        words = {w for w in chunk.text.split() if w}
        # a chunk that only contains 'charlie' must be attributed to page 3
        if words == {"charlie"}:
            assert chunk.page_start == 3
        if words == {"alpha"}:
            assert chunk.page_start == 1


def test_respects_target_size():
    pages = _pages("word " * 4000)
    size = 128
    chunks = chunk_document(DOC, pages, chunk_size=size, chunk_overlap=0)
    budget = size * CHARS_PER_TOKEN
    # merging runts can push a chunk slightly over; the 1.3x cap is the contract
    assert all(len(c.text) <= budget * 1.35 for c in chunks)


def test_overlap_shares_text_between_neighbours():
    pages = _pages(". ".join(f"sentence number {i}" for i in range(200)))
    chunks = chunk_document(DOC, pages, chunk_size=64, chunk_overlap=16)
    assert len(chunks) > 2
    overlaps = 0
    for a, b in zip(chunks, chunks[1:], strict=False):
        tail = set(a.text.split()[-8:])
        if tail & set(b.text.split()[:16]):
            overlaps += 1
    assert overlaps >= len(chunks) // 2


def test_unpaginated_sources_get_line_numbers():
    text = "\n".join(f"line {i} about refunds and policies" for i in range(100))
    chunks = chunk_document(DOC, [Page(number=None, text=text)], chunk_size=32, chunk_overlap=0)
    assert all(c.page_start is None for c in chunks)
    assert all(c.line_start is not None for c in chunks)
    assert chunks[0].line_start == 1
    assert chunks[-1].line_start > 1
    assert chunks[0].locator.startswith("L")


def test_text_without_separators_is_still_split():
    pages = _pages("x" * 5000)
    chunks = chunk_document(DOC, pages, chunk_size=64, chunk_overlap=0)
    assert len(chunks) > 1
