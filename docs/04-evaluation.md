# 4. Evaluation

> "I changed the chunk size and it feels better" is not engineering.

Every knob in this pipeline trades something off. Bigger chunks mean more context and less
precision. Reranking costs a second API call. A higher `top_k` costs tokens and dilutes the prompt.
There is no default that is right for your corpus, which means **the eval harness isn't optional
tooling — it's the only thing that turns tuning into work with a result.**

## Running it

```bash
ragcite eval --retrieval-only    # free, deterministic, no API calls -- the fast loop
ragcite eval                     # + answer quality and citation correctness
ragcite eval --judge             # + an LLM grader for faithfulness
ragcite eval --verbose           # print every case, not just the aggregates
```

Reports are written to `eval/reports/<timestamp>.json` with the full config embedded, so a run is
reproducible and two runs are diffable. `--no-save` skips that.

## Two layers

### Layer 1 — retrieval (free, deterministic, instant)

| Metric | What it measures | What a low value means |
|---|---|---|
| `recall_at_k` | Did *any* chunk from an expected source make the top-k? | The answer never reached the prompt. Nothing downstream can fix this. |
| `mrr` | Mean reciprocal rank of the first correct source | It's in there but ranked low — a reranking or fusion problem, not a chunking one. |

This layer needs no model and no key, so run it on every change. If recall is broken, stop: no
amount of prompt engineering recovers a passage that was never retrieved.

### Layer 2 — answers (needs the model)

| Metric | What it measures | What it catches |
|---|---|---|
| `must_contain` | Fraction of required facts stated in the answer | Retrieved the right passage, failed to extract the fact |
| `citation_ok` | Every marker resolves **and** an expected source is cited | Right answer, wrong attribution — the failure that destroys trust |
| `abstention_correct` | Did it abstain exactly when it should have? | Both over-answering and over-refusing |
| `uncited_sentences_total` | Factual sentences with no marker | Groundedness drift (a smell test, not a proof) |
| `dangling_citations_total` | Markers pointing at nothing | **Fabricated citations. Should always be zero.** |
| `judge_*` | LLM grades for faithfulness / completeness / citation quality | Nuance the string checks miss |

## The eval set

[`eval/questions.jsonl`](../eval/questions.jsonl) — one JSON object per line:

```json
{"question": "How long does a refund take to reach my card?",
 "expect_sources": ["refund-policy.md"],
 "must_contain": ["5 business days"]}

{"question": "What is the CEO's home address?",
 "expect_abstain": true,
 "note": "unanswerable -- must abstain, not guess"}
```

| Field | Meaning |
|---|---|
| `question` | What gets asked |
| `expect_sources` | Path substrings that should be retrieved *and* cited |
| `must_contain` | Strings the answer must state (case-insensitive) |
| `expect_abstain` | `true` when the corpus cannot answer — "I don't know" is the correct response |
| `note` | Why this case is in the set |

### Why a third of the cases are unanswerable

**A RAG system that answers everything is worse than useless.** Retrieval always returns *something* —
top-k is a ranking, not a relevance threshold — so for a question your corpus can't answer, the
model is handed the three least-irrelevant passages and invited to be helpful. Without an explicit
instruction to abstain and a metric that scores it, you ship a system that confidently invents a
refund policy for a product line you don't sell.

The hardest cases are the near-misses, and they're in the set deliberately:

* *"What is the refund policy for enterprise SaaS subscriptions billed annually?"* — the corpus has
  a refund policy, and it has nothing to do with SaaS subscriptions. Retrieval will return the
  refund policy with high confidence. The right answer is still "I don't know".

The abstention metric is the one that makes abstaining measurable, which is why the system prompt
mandates an **exact** sentence: `I don't know based on the provided documents.` A paraphrase would
be unparseable, and an unparseable behaviour is an unmeasurable one.

## Reading the shipped numbers honestly

The three sample documents produce **seven chunks**. With `top_k = 8`, every chunk is retrieved for
every question, so `recall_at_k` is trivially `1.0` and tells you nothing:

```
$ ragcite eval --retrieval-only
cases: 13

  recall_at_k     1.0      <- meaningless at this corpus size
  mrr             0.9      <- this one still discriminates
  latency_s_mean  0.0002
```

That is the demo corpus doing its job (proving the harness runs) and not doing another (measuring
retrieval). **Point it at your own corpus with a few hundred chunks and the numbers start to mean
something.** `mrr` is informative even here, because ranking still varies.

Running the full eval against the offline stand-in model shows the same thing from the other side:

```
$ ragcite eval --llm fake
  must_contain              0.2
  citation_ok               0.8
  abstention_correct        0.7692
  dangling_citations_total  0
```

The fake model is a lexical extractor — it grabs the best-matching sentence and cites it correctly
(`citation_ok 0.8`, zero dangling markers: the plumbing works) but it cannot state facts in its own
words (`must_contain 0.2`) and it cannot tell "no answer exists" from "an adjacent answer exists"
(`abstention_correct 0.77`). Those two numbers are exactly the gap a real model closes, and seeing
the harness *detect* the gap is the point.

## The tuning loop

```bash
# 0. Baseline
ragcite eval --retrieval-only --no-save | tee /tmp/before.txt

# 1. One change at a time
$EDITOR ragcite.toml         # chunk_size 512 -> 256
ragcite ingest corpus/ --reset

# 2. Measure
ragcite eval --retrieval-only --no-save | tee /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt
```

Worth trying, in roughly this order of expected payoff:

1. **A real embedder.** `hash:512` → `st:all-MiniLM-L6-v2` is usually the biggest single jump,
   because the default isn't semantic at all.
2. **Chunk size.** 512 → 256 for reference material, → 1024 for narrative text.
3. **`retrieval_mode`.** Compare `dense`, `lexical`, `hybrid` on your real queries.
4. **Reranking.** `rerank = true`. Costs a call per query — make it earn that.
5. **`top_k`.** More context isn't free: it dilutes attention and costs tokens.
6. **`effort`.** `medium` → `high` for multi-hop questions.

## Adding your own cases

Pull them from **real questions people actually asked** — support tickets, Slack, the search log.
Synthetic questions written while looking at the documents are the trap: they use the document's
own vocabulary, so retrieval looks great and real users (who use their own words) do not get the
same experience.

Rules of thumb for a set that's worth running:

* **20+ cases minimum.** Below that, one case flipping moves the metric by 5 points.
* **Keep the hard ones.** A set everything passes has stopped measuring anything.
* **Include multi-hop cases** that need two documents — that's where naive top-k quietly fails.
* **Include near-miss unanswerables.** They're the ones that catch over-confidence.
* **Split train/test** once you're tuning seriously, or you'll overfit the knobs to the set.

## What this harness does not do

* **No statistical significance.** Thirteen cases, one run. Treat small deltas as noise.
* **No cost tracking.** `usage` is on every `Answer`; aggregating it is on the [roadmap](06-roadmap.md).
* **LLM-as-judge is itself approximate.** It's a signal, not ground truth. Spot-check it against
  your own reading before you trust a judge score more than your eyes.

---

Next: [5. Design decisions](05-decisions.md).
