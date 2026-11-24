from __future__ import annotations

from pathlib import Path

import pytest

from ragcite.config import Config
from ragcite.pipeline import Rag

CORPUS = Path(__file__).resolve().parents[1] / "corpus"


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """A config that runs entirely offline: hashing embedder + fake LLM.

    `chunk_size` is smaller than the shipped default so the three short sample
    documents split into several chunks each -- otherwise retrieval is a
    document-level lookup and the tests never exercise the interesting parts
    (per-doc caps, neighbour expansion, page/line attribution).
    """
    return Config(
        db_path=str(tmp_path / "index.db"),
        embedder="hash:512",
        llm="fake",
        chunk_size=96,
        chunk_overlap=16,
        top_k=5,
        candidate_k=20,
    )


@pytest.fixture
def rag(cfg: Config) -> Rag:
    with Rag(cfg) as r:
        r.ingest([str(CORPUS)])
        yield r
