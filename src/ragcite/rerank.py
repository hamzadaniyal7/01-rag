"""Optional LLM reranking (cross-encoder style, without a cross-encoder).

Retrieval is fast and approximate; reranking is slow and precise. Running a
cheap model over ~40 candidates and keeping the top 8 reliably lifts answer
quality on corpora where the right chunk is *in* the pool but not at the top.

It is off by default because it adds a second API call to every query. Turn it
on, then prove it helped with `ragcite eval` -- that is the entire reason the
eval harness exists.
"""

from __future__ import annotations

import json

from ragcite.config import Config
from ragcite.llm import get_llm
from ragcite.llm.base import LLM
from ragcite.prompts import RERANK_SCHEMA, RERANK_SYSTEM
from ragcite.types import Hit

SNIPPET_CHARS = 700


def rerank(question: str, hits: list[Hit], cfg: Config, llm: LLM | None = None) -> list[Hit]:
    if not hits:
        return hits
    llm = llm or _default_llm(cfg)

    excerpts = "\n\n".join(
        f"[{i}] {' '.join(h.chunk.text.split())[:SNIPPET_CHARS]}" for i, h in enumerate(hits, 1)
    )
    user = f"Question: {question}\n\nExcerpts:\n{excerpts}"

    result = llm.complete(RERANK_SYSTEM, user, max_tokens=2048, json_schema=RERANK_SCHEMA)
    scores = _parse_scores(result.text, len(hits))
    if not scores:
        return hits  # reranker failed -- keep the retriever's order, don't drop results

    for i, hit in enumerate(hits, 1):
        hit.rerank_score = scores.get(i)
    # Unscored candidates sink below scored ones rather than disappearing.
    return sorted(
        hits, key=lambda h: (-(h.rerank_score if h.rerank_score is not None else -1.0), -h.score)
    )


def _parse_scores(text: str, n: int) -> dict[int, float]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    out: dict[int, float] = {}
    for entry in data.get("scores", []):
        try:
            idx = int(entry["id"])
            if 1 <= idx <= n:
                out[idx] = float(entry["score"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _default_llm(cfg: Config) -> LLM:
    if cfg.llm == "fake":
        return get_llm("fake")
    return get_llm(
        "anthropic",
        model=cfg.rerank_model,
        effort=cfg.rerank_effort,
        thinking=False,
        refusal_fallbacks=False,
    )
