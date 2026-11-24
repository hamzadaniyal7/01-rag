"""SQLite + a flat numpy matrix. That is the whole vector database.

For corpora up to roughly 100k chunks a brute-force dot product over a
contiguous float32 matrix is both simpler and faster than anything with an
index build step -- 100k x 384 floats is ~150MB and scores in milliseconds.
The point of keeping it this small is that nothing about retrieval is hidden
behind a service; see docs/05-decisions.md (ADR-003) for when to graduate to
pgvector or FAISS.

The index records which embedder built it. Searching with a different one is a
hard error, not a silent quality regression.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from ragcite.types import Chunk, Document

SCHEMA_VERSION = "1"

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    title        TEXT NOT NULL,
    n_pages      INTEGER NOT NULL DEFAULT 0,
    content_hash TEXT NOT NULL,
    ingested_at  TEXT NOT NULL DEFAULT (datetime('now')),
    meta         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   TEXT PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    ordinal    INTEGER NOT NULL,
    text       TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end   INTEGER NOT NULL,
    page_start INTEGER,
    page_end   INTEGER,
    line_start INTEGER,
    line_end   INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc_ordinal ON chunks(doc_id, ordinal);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    vec      BLOB NOT NULL
);
"""


class EmbedderMismatch(RuntimeError):
    """The index was built with a different embedding model."""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript(SCHEMA)
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.db.commit()
        self._matrix: tuple[list[str], np.ndarray] | None = None

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def get_meta(self, key: str) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def bind_embedder(self, name: str, dim: int) -> None:
        """Record the embedder on first write; verify it on every later one."""
        existing = self.get_meta("embedder")
        if existing is None:
            self.set_meta("embedder", name)
            self.set_meta("dim", str(dim))
            self.db.commit()
            return
        if existing != name:
            raise EmbedderMismatch(
                f"index at {self.path} was built with embedder {existing!r} but the current "
                f"config says {name!r}. Vectors from two models are not comparable.\n"
                f"Either switch back, or rebuild:  ragcite ingest <paths> --reset"
            )

    def require_embedder(self, name: str) -> None:
        existing = self.get_meta("embedder")
        if existing is None:
            raise EmbedderMismatch(f"index at {self.path} is empty -- run `ragcite ingest` first")
        if existing != name:
            raise EmbedderMismatch(
                f"index was built with {existing!r}, current config uses {name!r}. "
                f"Rebuild with `ragcite ingest <paths> --reset` or switch the config back."
            )

    # ------------------------------------------------------------- documents
    def document_by_path(self, path: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()

    def upsert_document(self, doc: Document) -> None:
        self.db.execute(
            """INSERT INTO documents(doc_id, path, title, n_pages, content_hash, meta)
               VALUES(?, ?, ?, ?, ?, ?)
               ON CONFLICT(doc_id) DO UPDATE SET
                 path=excluded.path, title=excluded.title, n_pages=excluded.n_pages,
                 content_hash=excluded.content_hash, meta=excluded.meta,
                 ingested_at=datetime('now')""",
            (doc.doc_id, doc.path, doc.title, doc.n_pages, doc.content_hash, json.dumps(doc.meta)),
        )

    def delete_document(self, doc_id: str) -> None:
        self.db.execute(
            "DELETE FROM embeddings WHERE chunk_id IN "
            "(SELECT chunk_id FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._matrix = None

    def reset(self) -> None:
        self.db.executescript(
            "DELETE FROM embeddings; DELETE FROM chunks; DELETE FROM documents; DELETE FROM meta;"
        )
        self.set_meta("schema_version", SCHEMA_VERSION)
        self.db.commit()
        self._matrix = None

    # ---------------------------------------------------------------- chunks
    def add_chunks(self, chunks: Iterable[Chunk], vectors: np.ndarray) -> None:
        chunks = list(chunks)
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.db.executemany(
            """INSERT OR REPLACE INTO chunks
               (chunk_id, doc_id, ordinal, text, char_start, char_end,
                page_start, page_end, line_start, line_end)
               VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    c.chunk_id,
                    c.doc_id,
                    c.ordinal,
                    c.text,
                    c.char_start,
                    c.char_end,
                    c.page_start,
                    c.page_end,
                    c.line_start,
                    c.line_end,
                )
                for c in chunks
            ],
        )
        self.db.executemany(
            "INSERT OR REPLACE INTO embeddings(chunk_id, vec) VALUES(?, ?)",
            [
                (c.chunk_id, np.asarray(v, dtype=np.float32).tobytes())
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
        self._matrix = None

    def commit(self) -> None:
        self.db.commit()

    def all_chunks(self) -> list[Chunk]:
        rows = self.db.execute(
            """SELECT c.*, d.path AS doc_path, d.title AS doc_title
               FROM chunks c JOIN documents d USING(doc_id)
               ORDER BY d.path, c.ordinal"""
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def chunks_by_id(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" * len(chunk_ids))
        rows = self.db.execute(
            f"""SELECT c.*, d.path AS doc_path, d.title AS doc_title
                FROM chunks c JOIN documents d USING(doc_id)
                WHERE c.chunk_id IN ({marks})""",
            chunk_ids,
        ).fetchall()
        return {r["chunk_id"]: _row_to_chunk(r) for r in rows}

    def neighbors(self, chunk: Chunk, window: int) -> list[Chunk]:
        if window <= 0:
            return []
        rows = self.db.execute(
            """SELECT c.*, d.path AS doc_path, d.title AS doc_title
               FROM chunks c JOIN documents d USING(doc_id)
               WHERE c.doc_id = ? AND c.ordinal BETWEEN ? AND ? AND c.chunk_id != ?
               ORDER BY c.ordinal""",
            (chunk.doc_id, chunk.ordinal - window, chunk.ordinal + window, chunk.chunk_id),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    # --------------------------------------------------------------- vectors
    def matrix(self) -> tuple[list[str], np.ndarray]:
        """(chunk_ids, (n, dim) float32 matrix). Cached until the next write."""
        if self._matrix is not None:
            return self._matrix
        rows = self.db.execute("SELECT chunk_id, vec FROM embeddings ORDER BY chunk_id").fetchall()
        if not rows:
            self._matrix = ([], np.zeros((0, 0), dtype=np.float32))
            return self._matrix
        ids = [r["chunk_id"] for r in rows]
        mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        self._matrix = (ids, mat)
        return self._matrix

    # ----------------------------------------------------------------- stats
    def stats(self) -> dict[str, object]:
        docs = self.db.execute("SELECT COUNT(*) n FROM documents").fetchone()["n"]
        chunks = self.db.execute("SELECT COUNT(*) n FROM chunks").fetchone()["n"]
        chars = self.db.execute("SELECT COALESCE(SUM(LENGTH(text)), 0) n FROM chunks").fetchone()[
            "n"
        ]
        return {
            "path": str(self.path),
            "embedder": self.get_meta("embedder"),
            "dim": self.get_meta("dim"),
            "documents": docs,
            "chunks": chunks,
            "chars": chars,
            "avg_chunk_chars": round(chars / chunks) if chunks else 0,
        }

    def documents(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM documents ORDER BY path").fetchall()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _row_to_chunk(r: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=r["chunk_id"],
        doc_id=r["doc_id"],
        ordinal=r["ordinal"],
        text=r["text"],
        char_start=r["char_start"],
        char_end=r["char_end"],
        page_start=r["page_start"],
        page_end=r["page_end"],
        line_start=r["line_start"],
        line_end=r["line_end"],
        doc_path=r["doc_path"],
        doc_title=r["doc_title"],
    )
