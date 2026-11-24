"""Embedder registry.

An embedder is named by a `kind:argument` spec so it can live in a config file
and be recorded inside the index:

    hash:512                 deterministic, offline, lexical (default)
    st:all-MiniLM-L6-v2      local sentence-transformers model
    voyage:voyage-3          hosted Voyage AI embeddings
"""

from __future__ import annotations

from ragcite.embeddings.base import Embedder, l2_normalize
from ragcite.embeddings.hashing import HashingEmbedder

__all__ = ["Embedder", "HashingEmbedder", "get_embedder", "l2_normalize"]


def get_embedder(spec: str) -> Embedder:
    kind, _, arg = spec.partition(":")
    kind = kind.strip().lower()
    arg = arg.strip()

    if kind == "hash":
        return HashingEmbedder(dim=int(arg or 512))

    if kind in {"st", "sentence-transformers"}:
        from ragcite.embeddings.sentence_transformers import (
            DEFAULT_MODEL,
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(model=arg or DEFAULT_MODEL)

    if kind == "voyage":
        from ragcite.embeddings.voyage import DEFAULT_MODEL, VoyageEmbedder

        return VoyageEmbedder(model=arg or DEFAULT_MODEL)

    raise ValueError(f"unknown embedder {spec!r}; expected hash:/st:/voyage:")
