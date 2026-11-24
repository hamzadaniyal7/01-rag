"""Local semantic embeddings via sentence-transformers (`pip install .[local]`).

Runs on CPU, no API key, no data leaves the machine. all-MiniLM-L6-v2 is the
default because it is 90MB and fast; bge-small-en-v1.5 scores better on
retrieval benchmarks if you can spare the download.
"""

from __future__ import annotations

import numpy as np

from ragcite.embeddings.base import Embedder, l2_normalize

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "sentence-transformers is not installed. Install the extra:\n"
                "    pip install -e '.[local]'"
            ) from exc
        self._model = SentenceTransformer(model)
        self.name = f"st:{model}"
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return l2_normalize(np.asarray(vecs, dtype=np.float32))
