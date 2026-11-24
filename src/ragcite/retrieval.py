"""Retrieval: dense, lexical, and the hybrid fusion of the two.

The ranking pipeline, in order:

    query -> dense top-C  ─┐
                           ├─ reciprocal rank fusion -> per-doc cap -> top-K
    query -> BM25  top-C  ─┘                                   -> (optional rerank)
                                                               -> (optional neighbours)

Reciprocal Rank Fusion is used instead of blending the raw scores because
cosine similarity and BM25 live on incomparable scales -- any weighted sum of
the two needs retuning every time the corpus or the embedder changes, while
RRF only looks at ranks and needs no tuning at all.

"""

from __future__ import annotations

import numpy as np

from ragcite.bm25 import BM25
from ragcite.config import Config
from ragcite.embeddings import Embedder
from ragcite.store import Store
from ragcite.types import Chunk, Hit


class Retriever:
    def __init__(self, store: Store, embedder: Embedder, cfg: Config) -> None:
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self._bm25: BM25 | None = None

    # ------------------------------------------------------------------ api
    def retrieve(
        self, question: str, *, k: int | None = None, mode: str | None = None
    ) -> list[Hit]:
        k = k or self.cfg.top_k
        mode = mode or self.cfg.retrieval_mode
        pool = max(self.cfg.candidate_k, k)

        dense = self._dense(question, pool) if mode in {"dense", "hybrid"} else []
        lexical = self._lexical(question, pool) if mode in {"lexical", "hybrid"} else []

        if mode == "dense":
            fused = [
                (cid, score, score, None, rank, None) for rank, (cid, score) in enumerate(dense, 1)
            ]
        elif mode == "lexical":
            fused = [
                (cid, score, None, score, None, rank)
                for rank, (cid, score) in enumerate(lexical, 1)
            ]
        else:
            fused = _rrf(dense, lexical, self.cfg.rrf_k)

        chunks = self.store.chunks_by_id([row[0] for row in fused])
        hits: list[Hit] = []
        for cid, score, dscore, lscore, drank, lrank in fused:
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            hits.append(
                Hit(
                    chunk=chunk,
                    score=score,
                    dense_score=dscore,
                    lexical_score=lscore,
                    dense_rank=drank,
                    lexical_rank=lrank,
                )
            )

        hits = _cap_per_doc(hits, self.cfg.max_per_doc)
        hits = hits[:k]

        if self.cfg.neighbor_window:
            hits = self._expand(hits)
        return hits

    # -------------------------------------------------------------- halves
    def _dense(self, question: str, k: int) -> list[tuple[str, float]]:
        ids, mat = self.store.matrix()
        if not ids:
            return []
        self.store.require_embedder(self.embedder.name)
        q = self.embedder.embed_query(question)
        if q.shape[0] != mat.shape[1]:
            raise ValueError(f"query vector has dim {q.shape[0]} but index has dim {mat.shape[1]}")
        # Both sides are L2-normalised, so a dot product *is* cosine similarity.
        scores = mat @ q.astype(np.float32)
        top = np.argsort(-scores)[:k]
        return [(ids[i], float(scores[i])) for i in top]

    def _lexical(self, question: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            chunks = self.store.all_chunks()
            self._bm25 = BM25([c.chunk_id for c in chunks], [c.text for c in chunks])
        return self._bm25.search(question, k)

    # ------------------------------------------------------------ expansion
    def _expand(self, hits: list[Hit]) -> list[Hit]:
        """Append the chunks on either side of each hit, for continuity.

        Scored 0 and marked as context-only: they are there so a sentence cut
        by a chunk boundary still reads correctly, not because they matched.
        """
        seen = {h.chunk.chunk_id for h in hits}
        extra: list[Hit] = []
        for hit in hits:
            for nb in self.store.neighbors(hit.chunk, self.cfg.neighbor_window):
                if nb.chunk_id not in seen:
                    seen.add(nb.chunk_id)
                    extra.append(Hit(chunk=nb, score=0.0))
        merged = hits + extra
        merged.sort(key=lambda h: (h.chunk.doc_path, h.chunk.ordinal))
        return merged

    def invalidate(self) -> None:
        self._bm25 = None


# --------------------------------------------------------------------------- #
# fusion helpers
# --------------------------------------------------------------------------- #
def _rrf(
    dense: list[tuple[str, float]],
    lexical: list[tuple[str, float]],
    rrf_k: int,
) -> list[tuple[str, float, float | None, float | None, int | None, int | None]]:
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (k + rank(d))."""
    dense_rank = {cid: i for i, (cid, _) in enumerate(dense, 1)}
    lex_rank = {cid: i for i, (cid, _) in enumerate(lexical, 1)}
    dense_score = dict(dense)
    lex_score = dict(lexical)

    out = []
    for cid in set(dense_rank) | set(lex_rank):
        score = 0.0
        if cid in dense_rank:
            score += 1.0 / (rrf_k + dense_rank[cid])
        if cid in lex_rank:
            score += 1.0 / (rrf_k + lex_rank[cid])
        out.append(
            (
                cid,
                score,
                dense_score.get(cid),
                lex_score.get(cid),
                dense_rank.get(cid),
                lex_rank.get(cid),
            )
        )
    out.sort(key=lambda row: (-row[1], row[0]))
    return out


def _cap_per_doc(hits: list[Hit], max_per_doc: int) -> list[Hit]:
    """Keep the context diverse: one verbose document shouldn't own the prompt."""
    if max_per_doc <= 0:
        return hits
    counts: dict[str, int] = {}
    out: list[Hit] = []
    for hit in hits:
        doc = hit.chunk.doc_id
        if counts.get(doc, 0) >= max_per_doc:
            continue
        counts[doc] = counts.get(doc, 0) + 1
        out.append(hit)
    return out


def format_hits(hits: list[Chunk] | list[Hit], width: int = 100) -> str:
    """Pretty-print a ranking for `ragcite search`."""
    lines = []
    for i, hit in enumerate(hits, 1):
        chunk = hit.chunk if isinstance(hit, Hit) else hit
        score = f"{hit.score:.4f}" if isinstance(hit, Hit) else "-"
        snippet = " ".join(chunk.text.split())[:width]
        lines.append(
            f"[{i}] {score}  {chunk.doc_title} {chunk.locator}  "
            f"({chunk.chunk_id}, {hit.why() if isinstance(hit, Hit) else ''})"
        )
        lines.append(f"    {snippet}...")
    return "\n".join(lines) if lines else "(no hits)"
