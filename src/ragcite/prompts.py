"""Prompts, kept in one file so they can be diffed and A/B-tested.

The grounding rules are load-bearing. Each line exists because of a specific
failure mode:

  "only the context"          -> stops the model answering from pretraining
  "cite every factual claim"  -> makes an unsupported sentence visibly uncited
  "exact abstain string"      -> makes abstention machine-detectable, so
                                 `ragcite eval` can score "knows what it
                                 doesn't know" as a first-class metric
  "do not invent markers"     -> dangling markers are the tell-tale of a
                                 fabricated citation; parse_citations() checks
"""

from __future__ import annotations

from ragcite.types import Hit

ABSTAIN_TEXT = "I don't know based on the provided documents."

SYSTEM = f"""You answer questions strictly from a set of numbered source excerpts.

Rules:
1. Use ONLY the information in the provided excerpts. Do not use outside knowledge, and do not
   guess or extrapolate beyond what the text states.
2. Cite your sources inline with bracketed markers that match the excerpt numbers, e.g. [1] or
   [2][5]. Every sentence that states a fact must carry at least one marker.
3. Never invent a marker. Only use numbers that appear in the excerpts given to you.
4. If the excerpts do not contain enough information to answer, reply with exactly this sentence
   and nothing else: {ABSTAIN_TEXT}
5. If the excerpts contradict each other, say so and cite both sides.
6. Be direct. Lead with the answer, then the supporting detail. No preamble, no restating the
   question, no closing summary.
"""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "description": "The answer, with inline [n] citation markers.",
        },
        "sufficient": {
            "type": "boolean",
            "description": "False if the excerpts did not contain enough information.",
        },
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "marker": {"type": "integer"},
                    "quote": {
                        "type": "string",
                        "description": "Verbatim sentence from that excerpt supporting the claim.",
                    },
                },
                "required": ["marker", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "sufficient", "citations"],
    "additionalProperties": False,
}


def render_context(hits: list[Hit], char_budget: int) -> tuple[str, list[Hit]]:
    """Format hits as numbered excerpts, dropping the tail if over budget.

    Returns the rendered context and the hits that actually made it in -- the
    caller needs that list to resolve `[n]` markers back to real sources.
    """
    parts: list[str] = []
    used: list[Hit] = []
    total = 0
    for hit in hits:
        chunk = hit.chunk
        header = (
            f"[{len(used) + 1}] source: {chunk.doc_title} | location: {chunk.locator} "
            f"| file: {chunk.doc_path}"
        )
        block = f"{header}\n{chunk.text}\n"
        if used and total + len(block) > char_budget:
            break
        parts.append(block)
        used.append(hit)
        total += len(block)
    return "\n".join(parts), used


def build_user_message(question: str, context: str) -> str:
    return (
        "Source excerpts:\n"
        "-----------------------------------------\n"
        f"{context}"
        "-----------------------------------------\n\n"
        f"Question: {question}"
    )


RERANK_SYSTEM = """You score how well each excerpt answers a question.

For each excerpt return a relevance score from 0 to 10:
  10  directly and completely answers the question
   7  contains part of the answer
   4  same topic, does not answer the question
   0  irrelevant

Judge only whether the text answers THIS question. Ignore writing quality and length.
Return a score for every excerpt id you were given, and no others."""

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"},
                },
                "required": ["id", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["scores"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You grade a retrieval-augmented answer against the excerpts it was given.

faithfulness: is every claim in the answer supported by the excerpts?
  (1 = fabricated, 5 = fully grounded)
completeness: does the answer cover what the question asked?
  (1 = misses the point, 5 = complete)
citation_quality: do the markers point at the excerpts that actually support each claim?
  (1 = wrong or missing, 5 = precise)

Be strict. An answer that is correct but cites the wrong excerpt scores low on citation_quality."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
        "citation_quality": {"type": "integer", "minimum": 1, "maximum": 5},
        "reason": {"type": "string"},
    },
    "required": ["faithfulness", "completeness", "citation_quality", "reason"],
    "additionalProperties": False,
}
