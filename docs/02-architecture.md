# 2. Architecture

## Shape of the thing

One Python package, no services, no framework. Two flows through the same components.

```
INGEST
  corpus/*.{pdf,md,txt}
        │
        ▼
   loaders.py ──── Document + [Page]        page boundaries preserved
        │
        ▼
   chunking.py ─── [Chunk]                  each carries char offsets -> page/line
        │
        ▼
  embeddings/ ──── float32 unit vectors     hash | sentence-transformers | voyage
        │
        ▼
    store.py ───── SQLite (text + metadata) + BLOB vectors


ASK
   question
        │
        ├──▶ embeddings/ ──▶ store.matrix() ──▶ cosine top-C ─┐
        │                                                      ├─ retrieval.py: RRF
        └──▶ bm25.py ─────────────────────── BM25 top-C ──────┘        │
                                                                        ▼
                                                            per-document cap
                                                                        │
                                                        (optional) rerank.py
                                                                        │
                                                                        ▼
                                                   prompts.py: numbered excerpts
                                                                        │
                                                                        ▼
                                                 llm/anthropic_llm.py: claude-opus-5
                                                                        │
                                                                        ▼
                                             generate.py: resolve [n] ──▶ Answer
```

`pipeline.py` is the only module that knows this order. Everything else is a component with a
narrow contract, which is what makes the swaps in [ADR-002](05-decisions.md) cheap.

## Module map

| File | Lines | Responsibility |
|---|---|---|
| `types.py` | 161 | `Document`, `Page`, `Chunk`, `Hit`, `Citation`, `Answer`. Plain dataclasses. |
| `config.py` | 139 | One dataclass of knobs; TOML → env → kwargs precedence; validation. |
| `loaders.py` | 111 | Files → `(Document, [Page])`. PDF via pypdf, markdown/text direct. |
| `chunking.py` | 208 | Recursive boundary-aware splitting; offset → page/line mapping. |
| `embeddings/` | 240 | `Embedder` ABC + hashing, sentence-transformers, Voyage implementations. |
| `store.py` | 280 | SQLite schema, vector BLOBs, the flat matrix, embedder binding. |
| `bm25.py` | 99 | BM25-Okapi over chunk text. |
| `retrieval.py` | 186 | Dense + lexical + RRF fusion, per-doc cap, neighbour expansion. |
| `rerank.py` | 73 | Optional LLM reranking of the candidate pool. |
| `prompts.py` | 154 | System prompt, excerpt formatting, JSON schemas. |
| `generate.py` | 147 | Model call orchestration, marker parsing, dangling-marker detection. |
| `llm/` | 331 | `LLM` ABC + Anthropic provider + deterministic fake. |
| `pipeline.py` | 202 | The wiring: `Rag.ingest()` and `Rag.ask()`. |
| `evaluate.py` | 248 | Eval harness and metrics. |
| `cli.py` | 315 | argparse front-end. |
| `api.py` | 68 | Optional FastAPI layer. |

## Data model

```
documents            chunks                       embeddings
─────────            ──────                       ──────────
doc_id      ◀────────  doc_id                     chunk_id  ◀── chunk_id
path (uniq)            chunk_id (pk)              vec BLOB  (float32, dim from meta)
title                  ordinal       ← reading order, enables neighbour expansion
n_pages                text
content_hash  ← skip unchanged files on re-ingest
ingested_at            char_start / char_end      ← provenance back into the source
                       page_start / page_end      ← PDF citations
                       line_start / line_end      ← markdown/text citations
```

Plus a `meta` key-value table holding `embedder` and `dim`. That one row prevents the worst silent
failure in RAG: querying an index built with model A using vectors from model B. The dot products
are still *numbers*, the search still returns *results*, and the results are noise. `store.py`
makes it an exception instead.

## Why the provenance columns matter

`char_start`/`char_end` are the whole trick. The chunker concatenates a document's pages into one
string and records where each page began; a chunk's offsets are then enough to recover its page
range by binary search, and its line range for unpaginated formats. Nothing downstream has to
guess, and `Chunk.locator` renders `p.3`, `pp.2-3`, or `L41-91` from stored facts.

## Extension points

| You want to | Change | Nothing else moves because |
|---|---|---|
| Use a different embedding model | `embedder = "..."` in config | `Embedder` ABC: 2 methods, normalised output |
| Use pgvector / FAISS | Implement `Store`'s read methods | `Retriever` only calls `matrix()`, `chunks_by_id()`, `neighbors()` |
| Use a different model provider | Implement `LLM.complete()` | Prompting and citation parsing live outside the provider |
| Add a file format | Add a branch in `loaders.py` returning `[Page]` | Chunking is format-agnostic |
| Change the grounding rules | Edit `prompts.py` | It's one string in one file, so it can be A/B'd with `eval` |

## Testability

The hashing embedder and the fake model are not toys — they're what makes the test suite
meaningful. 47 tests run offline in under a second and assert real behaviour: *this answer cites
refund-policy.md*, *this invented marker is flagged*, *page 2 of this PDF is where the fact lives*.
A test suite that needs an API key is a test suite that doesn't run in CI, and a RAG project whose
citation logic isn't tested has no business claiming citations.

---

Next: [3. The pipeline, stage by stage](03-pipeline.md).
