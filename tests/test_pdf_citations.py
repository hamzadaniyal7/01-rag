"""The headline claim, tested end to end on a real paginated PDF.

"[1] policy.pdf p.3" is only worth something if page 3 is genuinely where the
sentence came from. These tests ingest a generated three-page PDF and assert
that the citation on the answer points at the page holding the fact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ragcite.config import Config
from ragcite.loaders import UnsupportedFile, load_file
from ragcite.pipeline import Rag

from ._minipdf import write_pdf

PAGES = [
    [
        "Refund Policy",
        "Eligibility: a customer may request a refund within 30 days of delivery.",
        "Items must be unused and in their original packaging.",
    ],
    [
        "Processing time",
        "Refunds are issued to the original payment method within 5 business days.",
        "Bank processing adds a further 3 to 10 days depending on the card issuer.",
    ],
    [
        "Escalation",
        "If a refund has not appeared after 15 business days, contact support with the RMA number.",
        "Finance re-issues the payment within 2 business days.",
    ],
]


@pytest.fixture
def pdf_rag(tmp_path: Path) -> Rag:
    write_pdf(tmp_path / "policy.pdf", PAGES)
    cfg = Config(
        db_path=str(tmp_path / "index.db"),
        embedder="hash:512",
        llm="fake",
        chunk_size=32,
        chunk_overlap=4,
        min_chunk_size=8,
        top_k=3,
        candidate_k=10,
    )
    with Rag(cfg) as rag:
        rag.ingest([str(tmp_path / "policy.pdf")])
        yield rag


def test_loader_preserves_pages(tmp_path: Path):
    path = write_pdf(tmp_path / "p.pdf", PAGES)
    doc, pages = load_file(path)
    assert doc.n_pages == 3
    assert [p.number for p in pages] == [1, 2, 3]
    assert "5 business days" in pages[1].text


def test_chunks_know_which_page_they_came_from(pdf_rag: Rag):
    chunks = pdf_rag.store.all_chunks()
    assert chunks
    assert all(c.page_start is not None for c in chunks)
    by_page: dict[int, str] = {}
    for chunk in chunks:
        by_page.setdefault(chunk.page_start, "")
        by_page[chunk.page_start] += " " + chunk.text
    assert "5 business days" in by_page[2]
    assert "RMA" in by_page[3]


def test_citation_points_at_the_page_holding_the_fact(pdf_rag: Rag):
    answer = pdf_rag.ask("How long until a refund reaches the original payment method?")
    assert answer.citations
    assert answer.citations[0].path.endswith("policy.pdf")
    assert answer.citations[0].locator in {"p.2", "pp.1-2", "pp.2-3"}


def test_escalation_question_cites_the_escalation_page(pdf_rag: Rag):
    answer = pdf_rag.ask("What should I do if the refund has not appeared and I have an RMA?")
    assert answer.citations
    pages = {c.locator for c in answer.citations}
    assert any("3" in p for p in pages), pages


def test_a_pdf_with_no_extractable_text_fails_loudly(tmp_path: Path):
    write_pdf(tmp_path / "blank.pdf", [[], []])
    with pytest.raises(UnsupportedFile, match="no extractable text"):
        load_file(tmp_path / "blank.pdf")
