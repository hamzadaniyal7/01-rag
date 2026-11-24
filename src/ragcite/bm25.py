"""BM25-Okapi, implemented directly in pure Python.

Dense retrieval is weak at exactly the queries users actually type: product
codes, error strings, proper nouns, acronyms. BM25 nails those and costs
nothing, which is why hybrid search beats either half on its own.

The index is built in memory from the chunk table on first use. For this
corpus size that takes milliseconds; persisting it (SQLite FTS5) is on the
roadmap, not a correctness issue.
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short: aggressive stopword lists hurt phrase queries.
STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "will",
        "with",
    )
)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25:
    def __init__(self, doc_ids: list[str], texts: list[str]) -> None:
        self.doc_ids = doc_ids
        self.tokens = [tokenize(t) for t in texts]
        self.lengths = [len(t) for t in self.tokens]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        self.freqs = [Counter(t) for t in self.tokens]

        df: Counter[str] = Counter()
        for counter in self.freqs:
            df.update(counter.keys())
        n = len(self.tokens)
        # Robertson/Sparck-Jones IDF with the +0.5 smoothing that keeps it positive.
        self.idf = {
            term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()
        }

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        terms = tokenize(query)
        if not terms or not self.doc_ids:
            return []
        scores = [0.0] * len(self.doc_ids)
        for term in terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                tf = freq.get(term)
                if not tf:
                    continue
                norm = 1 - B + B * (self.lengths[i] / self.avg_len if self.avg_len else 1.0)
                scores[i] += idf * (tf * (K1 + 1)) / (tf + K1 * norm)
        ranked = sorted(
            ((self.doc_ids[i], s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: (-x[1], x[0]),
        )
        return ranked[:k]
