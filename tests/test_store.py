from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ragcite.config import Config
from ragcite.pipeline import Rag
from ragcite.store import EmbedderMismatch, Store
from ragcite.types import Chunk, Document


def _doc(path: str = "a.md") -> Document:
    return Document(doc_id="d1", path=path, title="A", n_pages=0, content_hash="h1")


def _chunk(i: int) -> Chunk:
    return Chunk(
        chunk_id=f"d1:{i:05d}", doc_id="d1", ordinal=i, text=f"chunk {i}", char_start=0, char_end=7
    )


def test_roundtrip(tmp_path: Path):
    with Store(tmp_path / "i.db") as store:
        store.bind_embedder("hash:4", 4)
        store.upsert_document(_doc())
        chunks = [_chunk(i) for i in range(3)]
        store.add_chunks(chunks, np.eye(3, 4, dtype=np.float32))
        store.commit()

        ids, mat = store.matrix()
        assert len(ids) == 3
        assert mat.shape == (3, 4)
        assert store.stats()["chunks"] == 3
        assert store.chunks_by_id([chunks[1].chunk_id])[chunks[1].chunk_id].text == "chunk 1"


def test_mixing_embedders_is_a_hard_error(tmp_path: Path):
    with Store(tmp_path / "i.db") as store:
        store.bind_embedder("hash:512", 512)
        with pytest.raises(EmbedderMismatch):
            store.bind_embedder("st:all-MiniLM-L6-v2", 384)


def test_neighbors_are_adjacent_only(tmp_path: Path):
    with Store(tmp_path / "i.db") as store:
        store.bind_embedder("hash:4", 4)
        store.upsert_document(_doc())
        chunks = [_chunk(i) for i in range(5)]
        store.add_chunks(chunks, np.eye(5, 4, dtype=np.float32))
        store.commit()
        got = store.neighbors(chunks[2], window=1)
        assert {c.ordinal for c in got} == {1, 3}


def test_unchanged_files_are_skipped_but_edits_are_reindexed(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    doc = corpus / "policy.md"
    doc.write_text("# Policy\n\nRefunds take five business days.\n")

    cfg = Config(db_path=str(tmp_path / "i.db"), embedder="hash:128", llm="fake")
    with Rag(cfg) as rag:
        first = rag.ingest([str(corpus)])
        assert first.documents_indexed == 1

        second = rag.ingest([str(corpus)])
        assert second.documents_indexed == 0
        assert second.documents_skipped == 1

        doc.write_text("# Policy\n\nRefunds take ten business days.\n")
        third = rag.ingest([str(corpus)])
        assert third.documents_indexed == 1
        assert rag.stats()["documents"] == 1  # replaced, not duplicated

        hits = rag.search("how many business days for a refund")
        assert "ten business days" in hits[0].chunk.text
