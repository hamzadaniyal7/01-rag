# 3. The pipeline, stage by stage

Every stage below lists what it does, the knobs it exposes, and the specific failure it exists to
prevent. If you only read one section, read [Chunking](#2-chunking) — it is where most RAG quality
is won or lost.

---

## 1. Loading

**`loaders.py`** — files in, `(Document, [Page])` out.

* **PDF** via pypdf, one `Page` per physical page, numbered from 1.
* **Markdown / text / rst** as a single unpaginated page; the chunker records line numbers instead.
* Title comes from a leading `# H1` if present, otherwise the filename.
* A content hash is computed so re-ingesting an unchanged file is a no-op.

**Why pagination is preserved here and not reconstructed later:** a citation that says `p.3` is
only trustworthy if page 3 is genuinely where the text was. Recovering that after concatenation —
by counting characters, or guessing from a page-count ratio — drifts the moment extraction
produces uneven page lengths, which it always does.

**Failure handled:** scanned PDFs. A PDF with no extractable text raises rather than indexing
empty chunks that quietly poison retrieval. OCR it first (`ocrmypdf in.pdf out.pdf`).

---

## 2. Chunking

**`chunking.py`** — the highest-leverage stage.

### Why not just split every N characters?

A chunk that begins mid-sentence retrieves badly (its embedding is a blend of two unrelated
thoughts) and reads worse once it's in the prompt. So the splitter walks separators from coarse to
fine and only hard-cuts when a span has no breakpoint at all:

```
"\n\n\n" → "\n\n" → "\n" → ". " → "? " → "! " → "; " → ", " → " " → hard cut
```

### Knobs

| Knob | Default | Effect |
|---|---|---|
| `chunk_size` | 512 tokens | Bigger: more context per hit, less precision, fewer hits fit the budget. Smaller: sharper retrieval, more risk of splitting a fact in half. |
| `chunk_overlap` | 64 tokens | Each chunk extends backwards into its predecessor, so a fact straddling a boundary is retrievable in full from at least one chunk. |
| `min_chunk_size` | 48 tokens | Runts (a bare heading, a stray line) are folded into the previous chunk instead of becoming their own useless hit. |

**Picking a size.** 256–512 tokens suits prose policies and documentation. Go smaller (128–256)
for dense reference material like FAQs or API docs where each entry is self-contained. Go larger
(512–1024) for narrative text where an answer needs surrounding context to make sense. Then stop
guessing and run [`ragcite eval`](04-evaluation.md) — recall@k and MRR move visibly with chunk size.

**Overlap only extends backwards.** Extending both ends would double the duplicated text for no
extra recall. The start is also snapped forward to the next word boundary, so a chunk never opens
mid-word (`"ms marked final sale"` instead of `"items marked final sale"` costs you the retrieval
of "items" and looks broken in a citation).

### Token approximation

Chunk sizes are specified in tokens and applied as `tokens × 4` characters. That ratio is a rule
of thumb for English prose, and it's fine here because chunk size is a *tuning knob*, not a
contract — being 10% off changes nothing you weren't already going to tune.

Where exactness matters — the actual prompt you're about to pay for — the real number comes from
the API, never a local tokenizer:

```bash
ragcite ask "..." --dry-run      # retrieves, then reports exact prompt tokens
```

---

## 3. Embedding

**`embeddings/`** — text to vectors. Three implementations behind one 2-method ABC.

| Spec | What it is | Use when |
|---|---|---|
| `hash:512` *(default)* | Feature-hashed word + char n-grams. Deterministic, instant, offline, **not semantic**. | You want the repo to work on clone, and CI to run with no key. |
| `st:all-MiniLM-L6-v2` | Local sentence-transformers. 384-dim, ~90MB, CPU-fine. | Real semantic retrieval, nothing leaves the machine. |
| `voyage:voyage-3` | Hosted Voyage AI embeddings. | Best quality; Anthropic's recommended embedding partner. |

All implementations return **L2-normalised float32**, so cosine similarity is a plain dot product
and the store can hold one contiguous matrix.

**On the hashing default:** it is honest about what it is. It cannot tell you "reimbursement" is
related to "refund" — it matches surface forms. It exists so `git clone && ragcite search` works
in 30 seconds with no key, no download, and no account, and so the test suite is deterministic.
Switch to `st:` or `voyage:` for anything real; the quality difference on paraphrased questions is
large.

**Asymmetric models.** `embed_query` is separate from `embed_documents` because some models
(Voyage among them) encode queries and passages differently. The ABC exposes both; symmetric
models just alias one to the other.

---

## 4. Indexing

**`store.py`** — SQLite plus a numpy matrix.

Text and metadata go in SQLite; vectors go in a `BLOB` column as raw float32 and are stacked into
one `(n, dim)` matrix on first search, cached until the next write. For corpora up to roughly
100k chunks this brute-forces faster than an ANN index builds, and it removes an entire category
of operational problem. See [ADR-003](05-decisions.md) for the graduation path.

**The one guardrail that matters:** the index records which embedder built it. Searching with a
different one raises `EmbedderMismatch` instead of returning plausible noise.

```
error: index at .ragcite/index.db was built with embedder 'hash:512' but the current config
says 'st:all-MiniLM-L6-v2'. Vectors from two models are not comparable.
Either switch back, or rebuild:  ragcite ingest <paths> --reset
```

**Re-ingest semantics:** unchanged files (same content hash) are skipped; changed files are
deleted and re-indexed wholesale. Partial updates would leave orphan chunks pointing at text that
no longer exists — which is a citation pointing at a line that isn't there any more.

---

## 5. Retrieval

**`retrieval.py` + `bm25.py`** — two retrievers, fused.

### Why hybrid

Dense retrieval understands meaning and is weak at exactly the queries users type: `SCIM`,
`ERR_CONN_RESET`, `SKU-44812`, a surname. A rare literal token carries little semantic signal and
gets smeared across the embedding. BM25 nails those and costs nothing.

```
"How long before money is back?"  → dense wins (paraphrase, no shared words)
"SCIM provisioning"               → BM25 wins (rare exact token)
"refund processing time"          → both agree, and agreement is signal
```

### Why RRF and not a weighted sum

Cosine similarity lives in [-1, 1]; BM25 is unbounded and corpus-dependent. Any `α·dense + β·bm25`
needs retuning every time the corpus or the embedder changes. **Reciprocal Rank Fusion** only
looks at ranks:

```
score(d) = Σ  1 / (rrf_k + rank_in_that_list(d))        rrf_k = 60
```

No tuning, no normalisation, and a document both retrievers like beats one that only one loves —
which is the behaviour you want.

### Knobs

| Knob | Default | Effect |
|---|---|---|
| `retrieval_mode` | `hybrid` | `dense` / `lexical` / `hybrid`. Compare them with `ragcite search --mode`. |
| `candidate_k` | 40 | Pool from each retriever before fusion. Bigger pool helps the reranker, costs nothing without one. |
| `top_k` | 8 | How many chunks reach the prompt. |
| `max_per_doc` | 3 | Hits from one document. Stops a verbose file owning the whole context. |
| `neighbor_window` | 0 | Also pull N chunks either side of each hit, for continuity across boundaries. |
| `rrf_k` | 60 | Fusion smoothing. Lower = top ranks dominate more. Rarely worth touching. |

**`max_per_doc` earns its place.** Without it, one 40-page handbook that mentions your query term
on every page takes all 8 slots and the three other documents that actually answer the question
never make the prompt.

---

## 6. Reranking (optional)

**`rerank.py`** — retrieval is fast and approximate; reranking is slow and precise.

A cheaper model (`claude-sonnet-5` by default) scores each of the ~40 candidates 0–10 against the
question, using structured output so the scores parse deterministically. Top-k survive.

```bash
ragcite ask "..." --rerank
```

Off by default: it adds a second API call to every query. Turn it on, then **prove it helped** with
`ragcite eval`. If the right chunk was already at rank 1, you paid for nothing.

A reranker failure (unparseable response) keeps the retriever's order rather than dropping results —
degrade, don't disappear.

---

## 7. Generation

**`prompts.py` + `generate.py` + `llm/anthropic_llm.py`**

Context is rendered as numbered excerpts, each with its source, location, and file:

```
[1] source: Refund Policy | location: p.2 | file: corpus/refund-policy.pdf
Once a return is received at our warehouse, refunds are issued to the original payment method
within 5 business days...
```

Markers are renumbered densely from 1 over the hits that *actually fit* the character budget, so
`[n]` always lines up with the list used to resolve citations. Excerpts past the budget are dropped
from the bottom of the ranking.

### Model settings and why

| Setting | Value | Reasoning |
|---|---|---|
| `model` | `claude-opus-5` | Grounded answering is cheap in tokens and unforgiving about instruction-following. The value of the project is an answer you can trust. |
| `thinking` | `{"type": "adaptive"}` | The model decides when reasoning is warranted. Fixed `budget_tokens` is gone on this generation and returns a 400. |
| `effort` | `medium` | Single-hop extraction over 8 excerpts doesn't need `high`. Raise it for multi-hop corpora — and measure, don't assume. |
| streaming | on | Avoids HTTP timeouts on long answers; the CLI prints tokens as they arrive. |
| `fallbacks` | `"default"` | If a policy classifier declines, the API retries on a fallback model inside the same call instead of returning nothing. Disable with `refusal_fallbacks = false`. |

### The grounding rules

Each line of the system prompt exists because of a specific failure:

| Rule | Prevents |
|---|---|
| Use only the excerpts | Answering from pretraining — the stale-answer failure |
| Cite every factual sentence | Unsupported claims becoming invisible |
| Never invent a marker | Fabricated citations (and `parse_citations` checks) |
| Abstain with an exact sentence | Unmeasurable abstention — the exact string is what makes it machine-detectable |
| Flag contradictions | Silently picking one side of a conflict between documents |

### Prompt caching: deliberately not used

The prompt prefix changes on every query (different excerpts), so a cache breakpoint would miss
every time and the ~300-token system prompt is below the minimum cacheable prefix anyway. If you
pin a large stable corpus into the system prompt, revisit this.

---

## 8. Citation resolution

**`generate.py`** — the stage that makes the rest worth having.

Every `[n]` is mapped back to the chunk that occupied slot *n* in the prompt, producing a
`Citation` with the file path and a real locator (`p.3`, `pp.2-3`, `L41-91`).

Markers that don't resolve go to `dangling_markers` and are **printed**:

```
  !! answer cited markers with no matching source: [9]
```

A dangling marker means the model cited something it was never shown. That is the clearest
hallucination signal a RAG system can emit, and swallowing it would be the single worst bug this
code could have.

`uncited_sentences()` is the companion check: factual-looking sentences carrying no marker. It's a
smell test, not a proof — and it's what the eval harness scores as `uncited_sentences_total`.

---

## Debugging a bad answer

In order, because the first two find most problems and cost nothing:

```bash
# 1. Did retrieval find it at all? (no model, no cost)
ragcite search "the question" -k 10

# 2. Which half is failing?
ragcite search "the question" --mode dense
ragcite search "the question" --mode lexical

# 3. What did the model actually see?
ragcite ask "the question" --show-context

# 4. How big is the prompt really?
ragcite ask "the question" --dry-run

# 5. Did the change help, or just feel better?
ragcite eval --retrieval-only
```

If `search` can't find it, no prompt engineering will save you. Fix chunking or the embedder.

---

Next: [4. Evaluation](04-evaluation.md).
