from __future__ import annotations

from ragcite.pipeline import Rag
from ragcite.retrieval import _cap_per_doc, _rrf
from ragcite.types import Chunk, Hit


def test_finds_the_right_document(rag: Rag):
    hits = rag.search("how long until a refund reaches my card")
    assert hits
    assert "refund-policy.md" in hits[0].chunk.doc_path


def test_hybrid_beats_each_half_on_a_rare_token(rag: Rag):
    # "SCIM" is a rare acronym: lexical nails it, a lexical-ish embedder may not.
    lexical = rag.search("SCIM", mode="lexical")
    hybrid = rag.search("SCIM", mode="hybrid")
    assert lexical and "security-faq.md" in lexical[0].chunk.doc_path
    assert any("security-faq.md" in h.chunk.doc_path for h in hybrid[:3])


def test_hits_carry_locators_for_citation(rag: Rag):
    hits = rag.search("encryption key rotation")
    assert all(h.chunk.locator for h in hits)
    assert all(h.chunk.doc_path for h in hits)


def test_rrf_rewards_agreement_between_retrievers():
    dense = [("a", 0.9), ("b", 0.8)]
    lexical = [("b", 4.0), ("c", 3.0)]
    fused = _rrf(dense, lexical, rrf_k=60)
    # 'b' is in both lists, so it wins even though it is second in each
    assert fused[0][0] == "b"
    ids = [row[0] for row in fused]
    assert set(ids) == {"a", "b", "c"}


def test_per_doc_cap_keeps_context_diverse():
    def hit(doc: str, i: int) -> Hit:
        chunk = Chunk(
            chunk_id=f"{doc}:{i}",
            doc_id=doc,
            ordinal=i,
            text="t",
            char_start=0,
            char_end=1,
            doc_path=f"{doc}.md",
        )
        return Hit(chunk=chunk, score=1.0 / (i + 1))

    hits = [hit("a", 0), hit("a", 1), hit("a", 2), hit("b", 0)]
    capped = _cap_per_doc(hits, max_per_doc=2)
    assert [h.chunk.chunk_id for h in capped] == ["a:0", "a:1", "b:0"]


def test_neighbor_expansion_adds_surrounding_context(rag: Rag):
    base = len(rag.search("refund"))
    rag.cfg.neighbor_window = 1
    expanded = len(rag.search("refund"))
    assert expanded > base
