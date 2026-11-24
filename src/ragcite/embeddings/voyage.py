"""Hosted embeddings from Voyage AI (`pip install .[voyage]`).

Anthropic does not serve an embedding endpoint -- Voyage is the partner they
point at, and its models are asymmetric, so queries and documents are encoded
with different `input_type` hints. Requires VOYAGE_API_KEY.
"""

from __future__ import annotations

import os

import numpy as np

from ragcite.embeddings.base import Embedder, l2_normalize

DEFAULT_MODEL = "voyage-3"
_DIMS = {"voyage-3": 1024, "voyage-3-lite": 512, "voyage-3-large": 1024, "voyage-code-3": 1024}


class VoyageEmbedder(Embedder):
    def __init__(self, model: str = DEFAULT_MODEL, batch_size: int = 64) -> None:
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "voyageai is not installed. Install the extra:\n    pip install -e '.[voyage]'"
            ) from exc
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is not set (see .env.example)")
        self._client = voyageai.Client()
        self._model = model
        self.name = f"voyage:{model}"
        self.dim = _DIMS.get(model, 1024)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, "document")

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], "query")[0]

    def _encode(self, texts: list[str], input_type: str) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self._client.embed(batch, model=self._model, input_type=input_type)
            out.extend(resp.embeddings)
        arr = np.asarray(out, dtype=np.float32)
        self.dim = arr.shape[1]
        return l2_normalize(arr)
