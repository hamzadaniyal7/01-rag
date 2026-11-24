# 01-rag — RAG with citations you can actually check

Ask a question in plain English, get an answer built **only** from your own documents, with a
marker on every claim that points at the exact file and page it came from.

```
$ ragcite ask "How long does a refund take to reach my card?"

Refunds are issued to the original payment method within 5 business days of the return arriving
at the warehouse [1]. Bank processing can add a further 3 to 10 days depending on the card issuer [1].

Sources:
  [1] Refund Policy p.2  (corpus/refund-policy.pdf)
```

*(Answer text varies with your corpus and model; the `Sources` block is produced by the pipeline.
Page locators like `p.2` come from PDFs — the samples in this repo are markdown, so they cite line
ranges. [`tests/test_pdf_citations.py`](tests/test_pdf_citations.py) asserts page-accurate
citations end to end against a real three-page PDF.)*

The whole point is that last block. An LLM that answers from your documents is easy; an LLM whose
answers you can **verify in ten seconds** is the thing people actually ship. Everything in this
repo exists to make that citation trustworthy — page-aware chunking, hybrid retrieval, a prompt
that forces abstention, and a check that flags any marker the model invented.

Built from scratch — no LangChain, no LlamaIndex, no vector-database service. About 3,000 lines of
Python across 24 small modules, roughly one line in seven a comment or docstring explaining *why*,
so every stage of the pipeline is something you can open, read, and change.

---

## Quickstart

```bash
git clone https://github.com/hamzadaniyal7/01-rag && cd 01-rag
python -m venv .venv && source .venv/bin/activate     # Python 3.11+
pip install -e '.[dev]'

ragcite ingest corpus/                      # 3 sample documents ship with the repo
ragcite search "refund processing time"     # retrieval only — no model, no cost, no API key
```

`search` works offline with zero configuration. To generate answers, add a key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...         # or run `ant auth login`
ragcite ask "How long do refunds take?"
```

No key handy? The whole pipeline runs against a deterministic stand-in model:

```bash
ragcite ask "How long do refunds take?" --llm fake
```

---

## What it does, stage by stage

```
  ingest:  load ──▶ chunk ──▶ embed ──▶ index
                 (page-aware)        (sqlite + numpy)

     ask:  question ─┬─▶ dense search  ─┐
                     │                  ├─▶ RRF fusion ─▶ per-doc cap ─▶ [rerank] ─▶ prompt
                     └─▶ BM25 search   ─┘                                              │
                                                                                       ▼
                        answer + resolved citations  ◀── parse [n] markers ◀── claude-opus-5
```

| Stage | What happens | Where |
|---|---|---|
| **Load** | PDF (per page), Markdown, text. Page boundaries are preserved, not reconstructed. | `loaders.py` |
| **Chunk** | Recursive split on paragraph → line → sentence → word, with overlap. Every chunk keeps its char offsets, so its page or line range is always recoverable. | `chunking.py` |
| **Embed** | Pluggable: offline hashing (default), local sentence-transformers, or hosted Voyage AI. | `embeddings/` |
| **Index** | SQLite for text and metadata, a flat float32 matrix for vectors. The embedder name is stored and verified — mixing two embedding spaces is a hard error, not a silent quality drop. | `store.py` |
| **Retrieve** | Dense cosine + BM25, fused with Reciprocal Rank Fusion, then capped per document so one verbose file can't own the prompt. | `retrieval.py`, `bm25.py` |
| **Rerank** | Optional: a cheaper model scores the candidate pool 0–10 and the top-k survive. Off by default — prove it helps with `eval` first. | `rerank.py` |
| **Generate** | `claude-opus-5` with adaptive thinking, streamed. The prompt forbids outside knowledge and mandates a marker on every factual sentence. | `generate.py`, `prompts.py` |
| **Cite** | Each `[n]` is mapped back to the chunk that sat in slot *n* of the prompt. Markers with no match are reported as `dangling_markers`. | `generate.py` |
| **Evaluate** | recall@k, MRR, citation correctness, abstention accuracy, optional LLM judge. | `evaluate.py` |

Full write-ups live in [`docs/`](docs/) — start with
[what RAG is and why each stage exists](docs/01-what-is-rag.md), then
[the architecture](docs/02-architecture.md).

---

## Commands

```bash
ragcite init                              # write a starter ragcite.toml
ragcite ingest corpus/ [--force] [--reset]
ragcite search "query" [-k 8] [--mode dense|lexical|hybrid] [--json]
ragcite ask "question" [--rerank] [--show-context] [--dry-run] [--json] [--structured]
ragcite stats [--documents]
ragcite eval [--retrieval-only] [--judge] [--verbose]
ragcite serve                             # HTTP API — pip install -e '.[api]'
```

Two flags worth knowing:

* `--show-context` prints the retrieved chunks to stderr, so you can see *what the model saw*.
  Most "the RAG is wrong" bugs are retrieval bugs, and this is how you find them.
* `--dry-run` retrieves and reports the exact prompt token count (from the Anthropic token-counting
  endpoint, not a guess) without generating anything.

As a library:

```python
from ragcite import Rag, Config

with Rag(Config.load()) as rag:
    rag.ingest(["corpus/"])
    answer = rag.ask("How often are encryption keys rotated?")
    print(answer.text)
    for citation in answer.citations:
        print(citation.render())
```

---

## Is it any good? Measure it

Tuning a RAG system by eye is how you end up with a pipeline nobody trusts. The repo ships an eval
set ([`eval/questions.jsonl`](eval/questions.jsonl)) with three kinds of case: single-hop lookups,
multi-hop questions that span two documents, and **unanswerable** questions where the only correct
response is "I don't know".

```bash
ragcite eval --retrieval-only    # free, deterministic, no API calls — the fast loop
ragcite eval                     # adds answer quality + citation correctness
ragcite eval --judge             # adds an LLM grader for faithfulness
```

Run against the three sample documents with the offline stand-in model:

```
$ ragcite eval --llm fake
cases: 13

  recall_at_k               1.0        <- meaningless here: 7 chunks, top_k 8
  mrr                       0.9
  must_contain              0.2        <- a lexical extractor can't state facts in its own words
  citation_ok               0.8
  abstention_correct        0.7692     <- it can't tell "absent" from "adjacent"
  uncited_sentences_total   7
  dangling_citations_total  0          <- no fabricated citations: the plumbing is sound
```

Those middle numbers are the gap a real model closes, and the harness **detecting** the gap is the
point. `recall_at_k` is honestly useless on a seven-chunk corpus — point it at a few hundred chunks
of your own and the numbers start to mean something.

Change `chunk_size`, flip on `rerank`, swap the embedder — then run it again and see whether the
number moved. That loop is the difference between a demo and an engineered system.

---

## Configuration

Every knob lives in `ragcite.toml` (create one with `ragcite init`), and any of them can be
overridden with a `RAGCITE_*` environment variable or a CLI flag.

```toml
[ragcite]
chunk_size = 512          # target tokens per chunk
chunk_overlap = 64
embedder = "hash:512"     # or "st:all-MiniLM-L6-v2" / "voyage:voyage-3"
retrieval_mode = "hybrid" # dense | lexical | hybrid
top_k = 8
candidate_k = 40          # pool size before fusion and reranking
max_per_doc = 3
model = "claude-opus-5"
effort = "medium"         # low | medium | high | xhigh | max
rerank = false
```

**On embedders:** the default `hash:512` is a hashing embedder — deterministic, instant, offline,
and *not semantic*. It makes the repo work the second you clone it and it makes the test suite run
anywhere, but it cannot tell you that "reimbursement" relates to "refund". For real semantic
retrieval install `pip install -e '.[local]'` and set `embedder = "st:all-MiniLM-L6-v2"`. The
index records which embedder built it and refuses to serve queries from a different one:

```
error: index was built with 'hash:512', current config uses 'st:all-MiniLM-L6-v2'.
Rebuild with `ragcite ingest <paths> --reset` or switch the config back.
```

The difference is visible immediately on a paraphrased question — "how long before the money is
back in my account" puts the refund policy first with `st:`, and the security FAQ first with
`hash:` (the words "account" and "back" dominate a lexical match).

<details>
<summary><strong>Troubleshooting the <code>[local]</code> extra on x86-64 macOS</strong></summary>

PyTorch is capped at 2.2.2 on Intel Macs, which requires `numpy<2`, while current `transformers`
requires `numpy>=2`. The combination that resolves:

```bash
pip install -e '.[local]' 'transformers<4.40' 'numpy<2' 'sentence-transformers<3.1'
```

These aren't pinned in `pyproject.toml` because doing so would hold back every other platform. On
Linux and Apple Silicon the plain extra works as-is.
</details>

---

## Development

```bash
pytest                 # 47 tests, no API key required, <1s
ruff check . && ruff format --check .
```

The test suite runs entirely offline: the hashing embedder plus a deterministic fake model let it
assert real behaviour — "this answer cites refund-policy.md", "this invented marker is flagged",
"this unanswerable question gets an abstention" — without a single network call. CI runs the same
thing on every push.

---

## Reading order

| Doc | What it covers |
|---|---|
| [docs/01-what-is-rag.md](docs/01-what-is-rag.md) | What RAG is, when to use it, why fine-tuning isn't the same tool |
| [docs/02-architecture.md](docs/02-architecture.md) | Module map, data model, how the pieces fit |
| [docs/03-pipeline.md](docs/03-pipeline.md) | Every stage, every knob, and the failure mode it addresses |
| [docs/04-evaluation.md](docs/04-evaluation.md) | What's measured, what each metric catches, how to tune with it |
| [docs/05-decisions.md](docs/05-decisions.md) | ADRs — the trade-offs, including the ones rejected |
| [docs/06-roadmap.md](docs/06-roadmap.md) | Known limits and what comes next |

## Where this sits

Project 1 of 3 in a build-it-yourself AI-engineering series:

| | Project | What it teaches |
|---|---|---|
| **01** | **rag** *(this repo)* | Retrieval, grounding, and citations you can verify |
| 02 | [agents](https://github.com/hamzadaniyal7/02-agents) | Loop engineering — plan → act → check over many steps |
| 03 | [deploy](https://github.com/hamzadaniyal7/03-deploy) | Serving an open model yourself with vLLM or SGLang |

## License

MIT — see [LICENSE](LICENSE).
