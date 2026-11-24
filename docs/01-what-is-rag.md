# 1. What RAG is, and what it isn't

## The problem

A language model knows what was in its training data. It does not know your company's refund
policy, your runbooks, last quarter's board deck, or the ticket a customer filed yesterday. Ask it
anyway and you get one of three bad outcomes:

1. **A refusal** — "I don't have access to that information."
2. **A confident fabrication** — a refund window that sounds plausible and is wrong.
3. **A stale answer** — the policy as it was two years ago.

The third is the dangerous one, because it looks exactly like a correct answer.

## The fix

**Retrieval-Augmented Generation** inserts a lookup step before the model answers:

```
question ──▶ find the passages in YOUR documents most likely to contain the answer
         ──▶ put those passages in the prompt
         ──▶ ask the model to answer using only those passages
         ──▶ return the answer WITH pointers back to the passages
```

That's it. No training, no fine-tuning, no model weights touched. The model stops being a
knowledge store and becomes a reading-comprehension engine over text you control.

Three properties follow directly, and they're the reason RAG is the default architecture for
"chat with my documents":

| Property | Why it follows |
|---|---|
| **Fresh** | Re-index a changed document and the next answer reflects it. No retraining. |
| **Attributable** | You know which passages went into the prompt, so you can show them. |
| **Bounded** | The model is instructed to answer from the context or abstain — a wrong answer becomes a *visible* retrieval failure rather than an invisible hallucination. |

## Why citations are the whole point

An unsourced answer from a RAG system is strictly worse than a search result. A search result at
least tells you where it came from; an unsourced paragraph asks you to trust a system you have no
way to audit.

With a citation, verification is a ten-second job: open the file, go to the page, read the
sentence. That's the difference between a demo that impresses people in a meeting and a tool
someone will actually use to answer a customer.

So this project treats the citation as the product, not a decoration:

* page boundaries are preserved from the PDF loader all the way to the stored chunk, so `p.3`
  means page 3 and not "roughly a third of the way in" ([`chunking.py`](../src/ragcite/chunking.py));
* every `[n]` the model writes is resolved back to the chunk that occupied slot *n* of the prompt;
* a marker that doesn't resolve is reported as a **dangling citation** rather than silently
  dropped — that's the system telling you it cited something it was never shown.

## RAG vs. fine-tuning vs. long context

These get conflated constantly. They solve different problems:

| | What it changes | Good for | Bad for |
|---|---|---|---|
| **RAG** | What's in the prompt | Facts that change; large or private corpora; anything needing attribution | Teaching a *style* or a new output format |
| **Fine-tuning** | The weights | Tone, format, structured-output reliability, domain jargon | Facts — they go stale and can't be cited |
| **Long context** | Nothing; you paste everything in | Corpora that genuinely fit, one-off analysis of a single large document | Recurring queries over a corpus bigger than the window, or where you pay for the whole corpus on every question |

The honest version: **long context is a real competitor to RAG for small corpora.** With a 1M-token
window, a few hundred pages fit. If your corpus fits and you don't mind paying for all of it on
every question, paste it in — you'll get excellent answers.

RAG wins when the corpus doesn't fit, when per-query cost matters, when you need to know *which*
document produced the answer, or when documents change often enough that re-pasting is a pipeline
of its own. In practice the two combine: retrieve generously, then let a long-context model read
what you retrieved.

## When NOT to use RAG

* **The answer isn't in any document.** Retrieval can't invent a policy that was never written down.
* **The question is a computation**, not a lookup ("what was total revenue by region last quarter?").
  That's a database query. RAG over a spreadsheet dump is a worse SQL.
* **Your corpus is three pages.** Paste them into the prompt and move on.
* **You need an exhaustive answer** ("list every clause mentioning liability"). Top-k retrieval is
  built to find the *best* passages, not all of them. That's a different query shape.

## The parts nobody tells you about

Most RAG tutorials stop at "embed, store, retrieve, generate" and produce something that works on
the demo question and falls over on the second one. The failures are predictable:

| Failure | Cause | What this repo does |
|---|---|---|
| Right document, wrong passage | Chunks too big; the answer is buried next to three irrelevant paragraphs | Chunk-size and overlap are tuned knobs with an eval to measure them |
| Misses exact terms (SKUs, error codes, acronyms) | Pure vector search is bad at rare literal tokens | Hybrid search: BM25 alongside dense ([`bm25.py`](../src/ragcite/bm25.py)) |
| One chatty document floods the prompt | Top-k is blind to diversity | Per-document cap on hits |
| Model answers from pretraining anyway | Prompt didn't forbid it | Explicit grounding rules plus a mandatory abstention string |
| Fabricated citations | Nobody checked the markers | Markers resolved and dangling ones reported |
| "It got worse after I changed the chunk size" | No measurement | [`ragcite eval`](04-evaluation.md) |

Each of those is a stage in [the pipeline](03-pipeline.md).

---

Next: [2. Architecture](02-architecture.md) — how the code is laid out.
