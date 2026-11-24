# 5. Design decisions (ADRs)

Short records of the choices that shaped this codebase, including the alternatives that were
rejected and the conditions under which each decision should be revisited.

---

## ADR-001 — No RAG framework

**Decision.** Build the pipeline from standard library + numpy + pypdf + the Anthropic SDK. No
LangChain, no LlamaIndex, no Haystack.

**Why.** The entire pipeline is ~2,400 lines of code across 24 modules. A framework would replace
maybe a quarter of that with configuration, and in exchange every stage becomes a method call into
code you don't read. The
specific costs that mattered here:

* **Debuggability.** When retrieval returns the wrong passage you need to see ranks and scores from
  both retrievers. That's `ragcite search --mode dense`, three lines of output, no stack to trace.
* **The interesting parts are the ones frameworks hide.** Page-accurate citations, dangling-marker
  detection, RRF fusion, per-doc capping — these are the project. Delegating them delegates the point.
* **Dependency weight.** The core install is four packages.

**Costs accepted.** No ecosystem of connectors (Notion, Confluence, S3). No off-the-shelf parent-
document or multi-query retrievers. Re-implementing BM25 and RRF, which are ~70 and ~20 lines.

**Revisit if** the corpus needs a dozen exotic source connectors — that's precisely what frameworks
are good at. Wrap this pipeline's ingest, don't replace the retrieval.

---

## ADR-002 — Providers behind narrow ABCs

**Decision.** `Embedder` has two methods. `LLM` has one. Both are abstract, both have a
zero-dependency implementation used by the tests.

**Why.** The whole test suite runs offline in under a second, and it asserts *behaviour* — "this
answer cites refund-policy.md", "this invented marker is flagged", "page 2 of this PDF holds the
fact" — not mocks. A suite that needs an API key doesn't run in CI, and a RAG project whose
citation logic isn't tested has no business claiming citations.

The secondary benefit is swap cost: changing the embedding model is one config string, and the
index refuses to serve queries from a model it wasn't built with.

**Costs accepted.** The narrow ABC can't express provider-specific features (Matryoshka dimensions,
multi-vector retrieval) without widening. The fake model is a lexical extractor and cannot judge
relevance — so model-level abstention is graded by `ragcite eval` against a real model, not by a
unit test.

---

## ADR-003 — SQLite + a flat numpy matrix, not a vector database

**Decision.** Text and metadata in SQLite, vectors as float32 `BLOB`s, stacked into one `(n, dim)`
matrix in memory for search. No pgvector, no FAISS, no Chroma, no hosted index.

**Why.** Brute force is genuinely fast at this scale. 100k chunks × 384 dims is ~150MB and a single
`mat @ q` scores all of them in milliseconds — faster than most ANN indexes *build*. In exchange:
one file to back up, no service to run, no index to rebuild, exact search (recall 1.0 by
construction, no approximation to tune), and `cp index.db` is a complete snapshot.

**Costs accepted.** Memory is linear in corpus size and the matrix is loaded whole. Beyond ~1M
chunks (~1.5GB) this stops being reasonable. No distributed search, no metadata pre-filtering in
the index.

**Revisit at ~500k chunks or when filtered search is needed.** The migration is contained: the
retriever only calls `matrix()`, `chunks_by_id()`, and `neighbors()`. Implement those three against
pgvector and nothing else changes.

---

## ADR-004 — Hybrid retrieval with RRF, not a weighted score blend

**Decision.** Run dense and BM25 independently, fuse by **Reciprocal Rank Fusion**
(`Σ 1/(60 + rank)`).

**Why.** Dense retrieval is bad at the queries users actually type — `SCIM`, `ERR_CONN_RESET`,
`SKU-44812`, a surname. Rare literal tokens carry little semantic signal. BM25 is excellent at them
and costs nothing, so the two are complementary rather than redundant.

The fusion method matters as much as the hybrid: cosine lives in [-1, 1], BM25 is unbounded and
corpus-dependent, so any `α·dense + β·bm25` needs retuning whenever the corpus or the embedder
changes — and it will be retuned by whoever has the least context. RRF reads only ranks: no
normalisation, no constants to fit, and a chunk both retrievers like beats one that only one loves.

**Costs accepted.** RRF discards score *magnitude*, so a runaway-best match ranks the same as a
merely-good one. `rrf_k = 60` is the published default, not a tuned value.

**Rejected alternative:** normalising both score distributions per query and blending. More knobs,
more fragility, no measured gain on this corpus.

---

## ADR-005 — A hashing embedder as the default

**Decision.** Ship `hash:512` — feature-hashed word and character n-grams — as the out-of-box
embedder, with sentence-transformers and Voyage AI as opt-in extras.

**Why.** `git clone && ragcite ingest corpus/ && ragcite search "refunds"` works in under a minute
with no API key, no account, no 500MB of torch, and no model download. Tests are deterministic
everywhere, and CI needs no secrets.

**Costs accepted, and stated loudly.** It is **not semantic**. It cannot relate "reimbursement" to
"refund". A user who never reads the docs gets mediocre retrieval and may blame RAG rather than the
embedder. Mitigations: the README, `ragcite stats`, and the docs all say so, and `Embedder.is_semantic`
exposes it programmatically.

**Rejected alternative:** defaulting to `st:all-MiniLM-L6-v2`. Better retrieval, but the first
command a new reader runs then downloads PyTorch, and on some platforms (see the troubleshooting
note in [README](../README.md)) that stack doesn't resolve cleanly at all. A default that fails on
clone is worse than a default that is honest about being a baseline.

---

## ADR-006 — Page provenance carried through chunking, not reconstructed

**Decision.** Loaders keep per-page text; the chunker concatenates while recording page start
offsets, and every chunk stores `char_start`/`char_end`. Page or line range is derived from those.

**Why.** It is the difference between a citation and a decoration. Reconstructing a page number
afterwards — by counting characters or scaling a page-count ratio — drifts the moment extraction
yields uneven page lengths, which it always does. An approximately-right page number is worse than
none: it looks authoritative and sends the reader to the wrong page.

[`tests/test_pdf_citations.py`](../tests/test_pdf_citations.py) asserts this end to end against a
generated three-page PDF.

**Costs accepted.** Loaders must preserve structure they'd otherwise discard; four extra columns on
`chunks`; a hand-rolled minimal PDF writer in the test suite (rather than a reportlab dependency
for one fixture).

---

## ADR-007 — Dangling citations are reported, never swallowed

**Decision.** A `[n]` marker that doesn't resolve to a retrieved chunk is collected in
`Answer.dangling_markers`, printed by the CLI, and counted by the eval harness.

**Why.** It is the clearest hallucination signal a RAG system can produce: the model cited
something it was never shown. The tempting implementations — drop the marker, or renumber around
it — both make a fabricated citation *look like a clean answer*. That would be the worst bug in
this codebase, because it would hide exactly the failure the project exists to surface.

**Costs accepted.** Output can look noisy on a bad answer. That is the intended trade.

---

## ADR-008 — Grounding enforced by prompt, with an exact abstention string

**Decision.** The system prompt forbids outside knowledge, mandates a marker on every factual
sentence, and requires a verbatim sentence when the context is insufficient:
`I don't know based on the provided documents.`

**Why.** An exact string makes abstention **machine-detectable**, so the eval harness can score
"knows what it doesn't know" as a first-class metric rather than something you eyeball. A
paraphrase ("I'm not sure the documents cover that") is unparseable, and an unparseable behaviour
is an unmeasurable one.

**Costs accepted.** Brittle to prompt edits — change the string in `prompts.py` and the abstention
metric silently goes to zero. Both live in the same module to make that obvious. Prompt-level
grounding is also not a guarantee; it is strong pressure plus the marker check as verification.

**Rejected alternative:** a second LLM call to verify groundedness on every answer. Doubles cost
and latency on every query to catch something the marker check catches for free. Available on
demand instead, via `ragcite eval --judge`.

---

## ADR-009 — `claude-opus-5`, adaptive thinking, `effort = medium`

**Decision.** Default to the strongest current model with adaptive thinking on, streaming on,
server-side refusal fallbacks on, and `effort` at `medium` rather than the API default of `high`.

**Why.**

* **Model.** Grounded answering is token-cheap and instruction-heavy. The failure mode of a weaker
  model here is subtle — it answers correctly but attributes to the wrong excerpt, which is worse
  than being wrong, because it looks checked.
* **Adaptive thinking.** The model decides when reasoning is warranted. Fixed `budget_tokens` is
  gone on this generation and returns a 400.
* **`effort = medium`.** Single-hop extraction over 8 excerpts doesn't need `high`. This is a
  *configured, measurable* choice, not a hidden downgrade: raise it in `ragcite.toml` and let
  `ragcite eval` tell you whether it bought anything.
* **Streaming.** Avoids HTTP timeouts on long answers and lets the CLI print tokens as they arrive.
* **Refusal fallbacks.** On a policy decline the API retries on a fallback model inside the same
  call instead of returning nothing. A 400 from a proxy that rejects the parameter falls back to a
  plain call rather than failing the query.

**Costs accepted.** Opus is the most expensive option; `--model` and `ragcite.toml` exist for
anyone who disagrees. The reranker deliberately uses a cheaper model (`claude-sonnet-5`) because
scoring 40 snippets is a different, easier job.

---

## ADR-010 — Prompt caching deliberately unused

**Decision.** No `cache_control` breakpoints on the query path.

**Why.** Caching is a prefix match, and the prefix here changes on every query — different
excerpts, different question. A breakpoint would miss every time. The system prompt alone (~300
tokens) is below the minimum cacheable prefix on every current model.

**Revisit if** you pin a large stable corpus into the system prompt, or add a long fixed set of
few-shot examples. Then the ordering rule applies: stable content first, volatile content after
the last breakpoint.

---

Next: [6. Roadmap and known limits](06-roadmap.md).
