"""The embedder contract.

Three rules the whole index depends on:
  1. vectors are float32 and L2-normalised, so cosine similarity is a dot product;
  2. `name` uniquely identifies the model -- it is written into the index and
     checked on every search, because mixing two embedding spaces silently
     returns garbage instead of raising;
  3. queries and documents may be embedded differently (asymmetric models care),
     hence two methods rather than one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    name: str
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 array of unit vectors."""

    def embed_query(self, text: str) -> np.ndarray:
        """Return a (dim,) float32 unit vector. Symmetric by default."""
        return self.embed_documents([text])[0]

    @property
    def is_semantic(self) -> bool:
        """False for lexical stand-ins like the hashing embedder."""
        return True


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)
