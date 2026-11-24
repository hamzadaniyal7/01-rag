from __future__ import annotations

from ragcite.generate import parse_citations, uncited_sentences
from ragcite.prompts import ABSTAIN_TEXT, render_context
from ragcite.types import Chunk, Hit


def _hit(n: int, page: int) -> Hit:
    chunk = Chunk(
        chunk_id=f"doc:{n}",
        doc_id="doc",
        ordinal=n,
        text=f"body {n}" * 10,
        char_start=0,
        char_end=10,
        page_start=page,
        page_end=page,
        doc_path=f"corpus/policy{n}.pdf",
        doc_title=f"Policy {n}",
    )
    return Hit(chunk=chunk, score=1.0 - n / 10)


def test_markers_resolve_to_the_prompt_slot_they_were_shown_in():
    used = [_hit(1, 3), _hit(2, 7)]
    citations, dangling = parse_citations("Refunds take 5 days [1] and [2] agrees.", used)
    assert [c.marker for c in citations] == [1, 2]
    assert citations[0].locator == "p.3"
    assert citations[1].locator == "p.7"
    assert dangling == []


def test_a_marker_the_model_was_never_shown_is_flagged_not_dropped():
    citations, dangling = parse_citations("Clearly true [9].", [_hit(1, 1)])
    assert citations == []
    assert dangling == [9]


def test_repeated_markers_are_reported_once():
    citations, _ = parse_citations("[1] and again [1] and [1].", [_hit(1, 1)])
    assert len(citations) == 1


def test_uncited_sentences_are_detected():
    text = "Refunds take five business days [1]. The warehouse also recycles all packaging."
    assert uncited_sentences(text) == ["The warehouse also recycles all packaging."]


def test_abstention_is_not_treated_as_uncited():
    assert uncited_sentences(ABSTAIN_TEXT) == []


def test_context_respects_the_char_budget_and_reports_what_fit():
    hits = [_hit(i, i) for i in range(1, 6)]
    context, used = render_context(hits, char_budget=200)
    assert len(used) < len(hits)
    assert len(context) <= 400  # one block may exceed the budget; the rest are dropped
    # markers are renumbered densely from 1 so they always match `used`
    assert context.startswith("[1] source: Policy 1 | location: p.1")
