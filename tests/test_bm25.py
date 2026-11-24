from __future__ import annotations

from ragcite.bm25 import BM25, tokenize


def test_tokenize_drops_stopwords_and_punctuation():
    assert tokenize("The refund, of course, takes 5 days!") == ["refund", "course", "takes", "days"]


def test_ranks_the_document_containing_the_term():
    bm25 = BM25(
        ["a", "b", "c"],
        [
            "Refunds are issued to the original payment method within five business days.",
            "Express delivery costs twelve dollars.",
            "Encryption keys rotate every ninety days.",
        ],
    )
    ranked = bm25.search("refund payment method", k=3)
    assert ranked[0][0] == "a"


def test_rare_terms_outrank_common_ones():
    bm25 = BM25(
        ["common1", "common2", "rare"],
        ["days days days", "days days", "SCIM provisioning is enterprise only"],
    )
    ranked = bm25.search("SCIM", k=3)
    assert ranked and ranked[0][0] == "rare"


def test_unknown_terms_return_nothing():
    bm25 = BM25(["a"], ["refunds and shipping"])
    assert bm25.search("quantum chromodynamics", k=5) == []
