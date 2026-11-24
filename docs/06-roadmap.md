# 6. Known limits and roadmap

Written plainly, because a project that pretends it has no limits is telling you its author didn't
look for them.

## Known limits

### Scale
* **~100k chunks** is the comfortable ceiling. Vectors load into one in-memory matrix
  (`n × dim × 4` bytes); past ~1M chunks that stops being reasonable. See
  [ADR-003](05-decisions.md) for the migration path.
* **BM25 is rebuilt per process.** Milliseconds for thousands of chunks, seconds for hundreds of
  thousands. A long-running server pays it once; the CLI pays it per invocation.
* **Single-process, no concurrency control.** Two `ingest` runs against one index will interleave
  badly. SQLite's WAL protects the file, not your intent.

### Retrieval
* **The default embedder is not semantic.** `hash:512` matches surface forms. Paraphrased
  questions ("when does the money come back?" vs. "refund processing time") retrieve poorly until
  you switch to `st:` or `voyage:` ([ADR-005](05-decisions.md)).
* **No metadata filtering.** You cannot scope a query to "only documents from 2026" or "only the
  security folder". Everything is searched.
* **No query rewriting.** A vague or multi-part question is embedded as-is. Multi-query expansion
  and HyDE both measurably help and neither is implemented.
* **Top-k is a ranking, not a threshold.** Something is always returned, even for a question your
  corpus cannot answer. Abstention is enforced at the prompt, not at retrieval.

### Documents
* **Scanned PDFs fail loudly, by design.** No OCR. Run `ocrmypdf in.pdf out.pdf` first.
* **Tables are flattened** to whatever pypdf extracts. A multi-column pricing table often becomes
  unreadable text, and no amount of retrieval quality fixes that.
* **No HTML, DOCX, CSV, or EPUB loader.** Markdown, text, rst, PDF only.
* **No chunk-level dedup.** The same policy in two files is indexed twice and can occupy two slots
  in one prompt.

### Answers
* **Grounding is prompt-enforced, not proven.** The marker check catches citations that point at
  nothing; it cannot catch a claim that cites a real excerpt which doesn't actually support it.
  `--judge` approximates that; it is a signal, not a proof.
* **Single-turn only.** No conversation history, so no follow-up questions and no pronoun
  resolution ("what about express?").
* **No cost accounting.** `usage` is attached to every `Answer` but nothing aggregates it.

### Operations
* **The HTTP API is a demo.** No auth, no rate limiting, no streaming. Don't expose it.
* **`[local]` extra doesn't resolve on every platform.** On x86-64 macOS, PyTorch is capped at
  2.2.2, which needs `numpy<2`, while current `transformers` needs `numpy>=2`. Pinning
  `transformers<4.40 numpy<2 sentence-transformers<3.1` works; the alternative is `voyage:` or the
  hashing default. Documented rather than pinned, because pinning would hobble every other platform.

## Roadmap

### Next — the highest value per line of code

1. **Query rewriting.** Expand the question into 2–3 paraphrases, retrieve for each, fuse with the
   same RRF that already exists. Usually the single biggest recall win after a real embedder.
2. **Persisted lexical index.** SQLite FTS5 instead of in-memory BM25, removing the per-process
   rebuild.
3. **Metadata filters.** `--filter "path:security/*"`, `--filter "after:2026-01-01"`, pushed into
   the SQL rather than applied after scoring.
4. **Cost and latency in the eval report.** Aggregate `usage` so a tuning decision can be read as
   "+4 points of MRR for +60% tokens" instead of just "+4 points".

### Then

5. **Conversation memory.** Rewrite follow-ups into standalone queries before retrieving.
6. **More loaders.** HTML (readability extraction), DOCX, CSV-as-rows.
7. **Table-aware chunking.** Keep a markdown table intact rather than splitting it mid-row.
8. **Contextual chunk headers.** Prepend the document title and section heading to each chunk
   before embedding — cheap, and known to lift retrieval meaningfully.
9. **Incremental re-embedding.** Only re-embed chunks whose text actually changed, not every chunk
   in a touched document.

### Maybe

10. **pgvector backend** for the >500k-chunk case ([ADR-003](05-decisions.md)).
11. **Streaming HTTP API** (SSE) plus a small web UI that shows the cited passage inline.
12. **Cross-encoder reranking** with a local model, for when per-query API calls are unacceptable.
13. **Answer caching** keyed on `(question, retrieved chunk ids, config hash)`.

### Explicitly not planned

* **A framework adapter.** [ADR-001](05-decisions.md) — the readable pipeline is the deliverable.
* **Fine-tuning anything.** Different tool, different problem — see
  [docs/01](01-what-is-rag.md#rag-vs-fine-tuning-vs-long-context).
* **An agent loop.** Retrieval-as-a-tool with planning is a genuinely different project. (It's the
  second project in this series.)

## Contributing

The bar for a retrieval or generation change is a measurement:

```bash
ragcite eval --retrieval-only --no-save   # before
# ...change one thing...
ragcite ingest corpus/ --reset
ragcite eval --retrieval-only --no-save   # after
```

Include both numbers in the PR. `pytest` and `ruff check .` must pass; neither needs an API key.
