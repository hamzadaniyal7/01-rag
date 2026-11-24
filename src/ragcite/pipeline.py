"""The pipeline, assembled.

    load -> chunk -> embed -> index          (ingest)
    embed query -> search -> fuse -> rerank -> prompt -> cite   (ask)

Everything above is a component with a narrow contract; this module is the
only place that knows the order. Keeping the wiring in one small file is what
makes it possible to swap the embedder, the store, or the model without
touching the rest.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ragcite.chunking import chunk_document
from ragcite.config import Config
from ragcite.embeddings import Embedder, get_embedder
from ragcite.generate import generate
from ragcite.llm import get_llm
from ragcite.llm.base import LLM
from ragcite.loaders import UnsupportedFile, iter_files, load_file
from ragcite.retrieval import Retriever
from ragcite.store import Store
from ragcite.types import Answer, Hit


@dataclass(slots=True)
class IngestReport:
    files_seen: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_written: int = 0
    failures: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.failures is None:
            self.failures = []

    def summary(self) -> str:
        lines = [
            f"{self.documents_indexed} document(s) indexed, "
            f"{self.chunks_written} chunk(s) written, "
            f"{self.documents_skipped} unchanged.",
        ]
        for path, err in self.failures:
            lines.append(f"  ! {path}: {err}")
        return "\n".join(lines)


class Rag:
    def __init__(
        self,
        cfg: Config | None = None,
        *,
        store: Store | None = None,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        stream_to: object | None = None,
    ) -> None:
        self.cfg = cfg or Config.load()
        self.store = store or Store(self.cfg.db_path)
        self.embedder = embedder or get_embedder(self.cfg.embedder)
        self._llm = llm
        self._stream_to = stream_to
        self.retriever = Retriever(self.store, self.embedder, self.cfg)

    # ------------------------------------------------------------- ingest
    def ingest(
        self,
        paths: list[str],
        *,
        force: bool = False,
        reset: bool = False,
        on_file: Callable[[str, int], None] | None = None,
    ) -> IngestReport:
        if reset:
            self.store.reset()
        self.store.bind_embedder(self.embedder.name, self.embedder.dim)

        report = IngestReport()
        for path in iter_files(paths, self.cfg.include_globs):
            report.files_seen += 1
            try:
                doc, pages = load_file(path)
            except (UnsupportedFile, OSError, ImportError) as exc:
                report.failures.append((str(path), str(exc)))
                continue

            existing = self.store.document_by_path(doc.path)
            if existing and existing["content_hash"] == doc.content_hash and not force:
                report.documents_skipped += 1
                continue

            chunks = chunk_document(
                doc,
                pages,
                chunk_size=self.cfg.chunk_size,
                chunk_overlap=self.cfg.chunk_overlap,
                min_chunk_size=self.cfg.min_chunk_size,
            )
            if not chunks:
                report.failures.append((str(path), "no text after chunking"))
                continue

            # Re-ingest replaces the document wholesale: partial updates would
            # leave orphan chunks pointing at text that no longer exists.
            if existing:
                self.store.delete_document(existing["doc_id"])

            vectors = self.embedder.embed_documents([c.text for c in chunks])
            self.store.upsert_document(doc)
            self.store.add_chunks(chunks, vectors)
            self.store.commit()

            report.documents_indexed += 1
            report.chunks_written += len(chunks)
            if on_file:
                on_file(doc.path, len(chunks))

        self.retriever.invalidate()
        return report

    # ---------------------------------------------------------------- ask
    def search(self, question: str, *, k: int | None = None, mode: str | None = None) -> list[Hit]:
        return self.retriever.retrieve(question, k=k, mode=mode)

    def ask(
        self,
        question: str,
        *,
        k: int | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
        structured: bool = False,
    ) -> Answer:
        use_rerank = self.cfg.rerank if rerank is None else rerank
        pool_k = max(self.cfg.candidate_k, k or self.cfg.top_k) if use_rerank else k
        hits = self.search(question, k=pool_k, mode=mode)

        if use_rerank and hits:
            from ragcite.rerank import rerank as rerank_hits

            hits = rerank_hits(question, hits, self.cfg, self.llm_for_rerank())
            hits = hits[: (k or self.cfg.top_k)]

        return generate(question, hits, self.llm, self.cfg, structured=structured)

    # ------------------------------------------------------------- lazily
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = self._build_llm()
        return self._llm

    def _build_llm(self) -> LLM:
        if self.cfg.llm == "fake":
            return get_llm("fake")
        return get_llm(
            "anthropic",
            model=self.cfg.model,
            effort=self.cfg.effort,
            thinking=self.cfg.thinking,
            refusal_fallbacks=self.cfg.refusal_fallbacks,
            stream_to=self._stream_to,
        )

    def llm_for_rerank(self) -> LLM | None:
        return get_llm("fake") if self.cfg.llm == "fake" else None

    def count_prompt_tokens(self, question: str, hits: list[Hit]) -> int | None:
        """Exact prompt size via the Anthropic token-counting endpoint."""
        from ragcite.prompts import SYSTEM, build_user_message, render_context

        context, _ = render_context(hits, self.cfg.context_char_budget)
        user = build_user_message(question, context)
        llm = self.llm
        counter = getattr(llm, "count_tokens", None)
        return counter(SYSTEM, user) if counter else None

    def stats(self) -> dict[str, object]:
        return self.store.stats()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Rag:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def default_corpus(cfg: Config) -> list[str]:
    path = Path(cfg.corpus_dir)
    if not path.exists():
        print(f"corpus directory {path} does not exist", file=sys.stderr)
        return []
    return [str(path)]
