"""Generation + citation resolution.

The generation step is small. The part that matters is what happens *after*:
every `[n]` the model wrote is mapped back to the chunk that was actually in
slot n of the prompt. Markers with no matching chunk are collected separately
as `dangling_markers` -- a non-empty list means the model cited something it
was never shown, which is the single clearest hallucination signal a RAG
system can produce. It is surfaced, never swallowed.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragcite.config import Config
from ragcite.llm.base import LLM
from ragcite.prompts import (
    ABSTAIN_TEXT,
    ANSWER_SCHEMA,
    SYSTEM,
    build_user_message,
    render_context,
)
from ragcite.types import Answer, Citation, Hit

MARKER_RE = re.compile(r"\[(\d{1,3})\]")


def generate(
    question: str,
    hits: list[Hit],
    llm: LLM,
    cfg: Config,
    *,
    structured: bool = False,
) -> Answer:
    """Answer `question` from `hits`, or abstain if there is nothing to go on."""
    usable = [h for h in hits if h.score >= cfg.min_score or h.score == 0.0]
    if not usable:
        return Answer(
            question=question,
            text=ABSTAIN_TEXT,
            citations=[],
            hits=hits,
            abstained=True,
            model=getattr(llm, "name", ""),
        )

    context, used = render_context(usable, cfg.context_char_budget)
    user = build_user_message(question, context)

    result = llm.complete(
        SYSTEM,
        user,
        max_tokens=cfg.max_tokens,
        json_schema=ANSWER_SCHEMA if structured else None,
    )

    if result.refused:
        return Answer(
            question=question,
            text="The model declined to answer this request.",
            citations=[],
            hits=used,
            abstained=True,
            model=result.model,
            usage=result.usage,
        )

    text = result.text.strip()
    sufficient = True
    if structured:
        text, sufficient = _parse_structured(text)

    citations, dangling = parse_citations(text, used)
    abstained = not sufficient or _is_abstention(text)

    return Answer(
        question=question,
        text=text,
        citations=citations,
        hits=used,
        abstained=abstained,
        model=result.model,
        usage=result.usage,
        dangling_markers=dangling,
    )


def parse_citations(text: str, used: list[Hit]) -> tuple[list[Citation], list[int]]:
    """Resolve `[n]` markers to the chunks that occupied slot n in the prompt."""
    citations: list[Citation] = []
    dangling: list[int] = []
    seen: set[int] = set()
    for match in MARKER_RE.finditer(text):
        marker = int(match.group(1))
        if marker in seen:
            continue
        seen.add(marker)
        if 1 <= marker <= len(used):
            chunk = used[marker - 1].chunk
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=chunk.chunk_id,
                    path=chunk.doc_path,
                    title=chunk.doc_title,
                    locator=chunk.locator,
                )
            )
        else:
            dangling.append(marker)
    citations.sort(key=lambda c: c.marker)
    return citations, sorted(dangling)


def uncited_sentences(text: str) -> list[str]:
    """Factual-looking sentences with no marker -- a groundedness smell test.

    Used by `ragcite eval` as a cheap, model-free proxy for "did it cite
    everything it claimed". Not a proof of anything; a signal.
    """
    if _is_abstention(text):
        return []
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        s = sentence.strip()
        if len(s) < 25:
            continue
        if not MARKER_RE.search(s):
            out.append(s)
    return out


def _parse_structured(text: str) -> tuple[str, bool]:
    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return text, True
    return str(data.get("answer", text)).strip(), bool(data.get("sufficient", True))


def _is_abstention(text: str) -> bool:
    normalized = text.strip().rstrip(".").lower()
    return normalized == ABSTAIN_TEXT.rstrip(".").lower()
