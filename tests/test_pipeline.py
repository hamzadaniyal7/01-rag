"""End-to-end: ingest the sample corpus, ask a question, check the citation.

Runs with the hashing embedder and the fake LLM, so it needs no API key -- the
point is to prove the *plumbing* (chunk -> index -> retrieve -> prompt -> cite)
is intact, not to measure answer quality. That is what `ragcite eval` is for.
"""

from __future__ import annotations

import re

from ragcite.evaluate import EvalCase, evaluate
from ragcite.pipeline import Rag
from ragcite.prompts import ABSTAIN_TEXT


def test_ingest_indexed_the_corpus(rag: Rag):
    stats = rag.stats()
    assert stats["documents"] == 3
    assert stats["chunks"] > 3
    assert stats["embedder"] == "hash:512"


def test_answer_cites_the_source_it_came_from(rag: Rag):
    answer = rag.ask("How long do refunds take to reach my card?")
    assert answer.citations, "expected at least one citation"
    assert any("refund-policy.md" in c.path for c in answer.citations)
    assert not answer.dangling_markers
    assert "days" in answer.text.lower()
    # The citation must carry a real location, not just a filename.
    assert answer.citations[0].locator


def test_every_citation_points_at_a_chunk_that_was_in_context(rag: Rag):
    answer = rag.ask("How often are encryption keys rotated?")
    context_ids = {h.chunk.chunk_id for h in answer.hits}
    assert {c.chunk_id for c in answer.citations} <= context_ids


def test_abstains_when_nothing_clears_the_score_floor(rag: Rag):
    """The pipeline abstains before spending a token when retrieval is empty.

    Whether the *model* abstains on an unanswerable question is a different
    property, graded by `ragcite eval` against a real model -- a lexical stand-in
    cannot tell "no refund policy exists" from "a refund policy exists but says
    nothing about this". See the expect_abstain cases in eval/questions.jsonl.
    """
    rag.cfg.min_score = 10.0  # nothing can clear this
    answer = rag.ask("How long do refunds take?")
    assert answer.abstained
    assert not answer.citations
    assert answer.text == ABSTAIN_TEXT


def test_render_includes_file_and_location(rag: Rag):
    """Every rendered citation carries a file AND a position inside it.

    Which document wins is a retrieval-quality question (graded by `ragcite
    eval`, and dependent on the embedder); what must always hold is that the
    rendered source is checkable -- a path plus a locator.
    """
    answer = rag.ask("What is the deadline for requesting a refund?")
    rendered = answer.render()
    assert "Sources:" in rendered
    assert answer.citations
    for citation in answer.citations:
        assert citation.path.endswith(".md")
        assert re.match(r"^(p\.\d|pp\.\d|L\d|#\d)", citation.locator), citation.locator
        assert citation.path in rendered
        assert citation.locator in rendered


def test_eval_harness_scores_retrieval_without_calling_a_model(rag: Rag):
    cases = [
        EvalCase(question="How long do refunds take?", expect_sources=["refund-policy.md"]),
        EvalCase(question="How much is express delivery?", expect_sources=["shipping.md"]),
    ]
    report = evaluate(rag, cases, answer=False)
    assert report["metrics"]["recall_at_k"] == 1.0
    assert report["metrics"]["mrr"] > 0
