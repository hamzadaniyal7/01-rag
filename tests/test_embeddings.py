from __future__ import annotations

import numpy as np

from ragcite.embeddings import get_embedder
from ragcite.embeddings.hashing import HashingEmbedder


def test_deterministic_across_instances():
    a = HashingEmbedder(dim=256).embed_documents(["refunds take five business days"])
    b = HashingEmbedder(dim=256).embed_documents(["refunds take five business days"])
    assert np.allclose(a, b)


def test_vectors_are_unit_length():
    vecs = HashingEmbedder(dim=128).embed_documents(["one", "two words here", ""])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms[:2], 1.0, atol=1e-5)


def test_lexical_overlap_beats_unrelated_text():
    emb = HashingEmbedder(dim=512)
    docs = emb.embed_documents(
        [
            "Refunds are issued within five business days of receipt.",
            "Encryption keys are rotated every ninety days.",
        ]
    )
    query = emb.embed_query("how many business days for a refund")
    scores = docs @ query
    assert scores[0] > scores[1]


def test_registry_parses_specs():
    emb = get_embedder("hash:64")
    assert emb.dim == 64
    assert emb.name == "hash:64"
    assert emb.is_semantic is False
