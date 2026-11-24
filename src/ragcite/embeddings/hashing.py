"""A dependency-free embedder: hashed word + character n-grams.

This is *not* a semantic model -- it cannot tell you that "reimbursement" and
"refund" are related. What it is: deterministic, instant, offline, and good
enough at lexical overlap that the entire pipeline (and its test suite) runs
with no API key and no model download. It is the default so that `ragcite
ingest && ragcite search` works the minute you clone the repo.

Swap to `st:...` or `voyage:...` in ragcite.toml for real semantics; the
index records which embedder built it and refuses to mix the two.
"""

from __future__ import annotations

import hashlib
import math
import re

import numpy as np

from ragcite.embeddings.base import Embedder, l2_normalize

TOKEN_RE = re.compile(r"[a-z0-9]+")


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512, char_ngram: int = 4) -> None:
        self.dim = dim
        self.char_ngram = char_ngram
        self.name = f"hash:{dim}"

    @property
    def is_semantic(self) -> bool:
        return False

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            self._accumulate(text, out[row])
        return l2_normalize(out)

    def _accumulate(self, text: str, row: np.ndarray) -> None:
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for a, b in zip(tokens, tokens[1:], strict=False):
            bigram = f"{a}_{b}"
            counts[bigram] = counts.get(bigram, 0) + 1
        # Character n-grams give a little robustness to plurals and typos.
        for tok in set(tokens):
            if len(tok) > self.char_ngram:
                for i in range(len(tok) - self.char_ngram + 1):
                    gram = f"#{tok[i : i + self.char_ngram]}"
                    counts[gram] = counts.get(gram, 0) + 1
        for feature, count in counts.items():
            idx, sign = _bucket(feature, self.dim)
            # Sublinear term frequency: the 10th "refund" says little the 1st didn't.
            row[idx] += sign * (1.0 + math.log(count))


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    # The sign bit halves collision bias (the "hashing trick" / signed hashing).
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0
