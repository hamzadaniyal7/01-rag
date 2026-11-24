"""A deterministic stand-in for the real model.

It does not reason -- it extracts. Given the numbered excerpts the pipeline
builds, it ranks every sentence with the same BM25 implementation the retriever
uses and returns the best one, cited to the excerpt it came from.

That is enough to exercise every other moving part -- prompt assembly, marker
parsing, citation resolution, dangling-marker detection, the abstention path --
with no API key and no network, which is why the entire test suite and
`--llm fake` run offline.

What it deliberately cannot do is judge whether an excerpt *actually answers*
the question. A lexical matcher will happily answer "what is the CEO's home
address?" from a paragraph about changing a delivery address. Knowing when to
say "I don't know" is exactly the part that needs a real model, which is why
`ragcite eval` scores abstention against `claude-opus-5` rather than here.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ragcite.bm25 import BM25, tokenize
from ragcite.llm.base import LLM, LLMResult

BLOCK_RE = re.compile(r"^\[(\d+)\]\s*source:.*$", re.MULTILINE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ABSTAIN = "I don't know based on the provided documents."
MIN_SENTENCE_CHARS = 25


class FakeLLM(LLM):
    name = "fake"

    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        question = _last_question(user)
        sentences = _sentences(user)
        best = _best_sentence(question, sentences)
        usage = {"input_tokens": len(user) // 4}

        if best is None:
            text = (
                json.dumps({"answer": ABSTAIN, "sufficient": False, "citations": []})
                if json_schema is not None
                else ABSTAIN
            )
            return LLMResult(text=text, model="fake", usage=usage)

        marker, sentence = best
        answer = f"{sentence} [{marker}]"
        if json_schema is not None:
            answer = json.dumps(
                {
                    "answer": answer,
                    "sufficient": True,
                    "citations": [{"marker": marker, "quote": sentence[:160]}],
                }
            )
        return LLMResult(text=answer, model="fake", usage=usage)


def _last_question(user: str) -> str:
    match = re.search(r"Question:\s*(.+?)\s*$", user, re.DOTALL)
    return match.group(1).strip() if match else user[-200:]


def _sentences(user: str) -> list[tuple[int, str]]:
    """Flatten the context section into [(marker, sentence), ...]."""
    out: list[tuple[int, str]] = []
    matches = list(BLOCK_RE.finditer(user))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user)
        body = user[start:end].split("\nQuestion:")[0]
        body = body.replace("-" * 10, " ")
        marker = int(m.group(1))
        for raw in SENTENCE_RE.split(body):
            sentence = " ".join(raw.split()).strip(" -*#|")
            if len(sentence) >= MIN_SENTENCE_CHARS:
                out.append((marker, sentence))
    return out


def _best_sentence(
    question: str, sentences: list[tuple[int, str]], top_excerpts: int = 1
) -> tuple[int, str] | None:
    """Best sentence from the highest-ranked excerpt(s).

    Excerpts arrive in retrieval order, so restricting to the first few means
    the fake inherits the retriever's judgement instead of second-guessing it
    with a bag of words -- which is both more faithful to what a real model
    does with the context and far more stable across chunking changes.
    """
    if not sentences or not tokenize(question):
        return None
    allowed = sorted({marker for marker, _ in sentences})[:top_excerpts]
    pool = [(m, s) for m, s in sentences if m in allowed]
    if not pool:
        return None
    index = BM25([str(i) for i in range(len(pool))], [s for _, s in pool])
    ranked = index.search(question, k=1)
    if not ranked:
        # No lexical overlap at all with the best excerpt: fall back to its
        # opening sentence rather than inventing an abstention the retriever
        # does not support.
        return pool[0]
    return pool[int(ranked[0][0])]
