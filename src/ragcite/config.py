"""Configuration: one dataclass, three sources, documented precedence.

Precedence (last wins):
  1. the defaults below
  2. ``ragcite.toml`` in the current directory (or the path given to ``load``)
  3. ``RAGCITE_*`` environment variables

Every knob that changes retrieval or generation quality lives here, so an
experiment is a config diff rather than a code diff.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "ragcite.toml"


@dataclass(slots=True)
class Config:
    # --- storage -----------------------------------------------------------
    db_path: str = ".ragcite/index.db"

    # --- ingestion ---------------------------------------------------------
    # Target chunk size in *tokens* (approximated as 4 chars/token at chunk
    # time; see docs/03-pipeline.md for why that approximation is fine here).
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 48
    include_globs: tuple[str, ...] = ("*.md", "*.txt", "*.rst", "*.pdf")

    # --- embeddings --------------------------------------------------------
    # "hash:<dim>"  deterministic, offline, zero dependencies (default: runs anywhere)
    # "st:<model>"  sentence-transformers, local, semantic  (pip install .[local])
    # "voyage:<model>"  hosted embeddings from Voyage AI    (pip install .[voyage])
    embedder: str = "hash:512"

    # --- retrieval ---------------------------------------------------------
    retrieval_mode: str = "hybrid"  # dense | lexical | hybrid
    top_k: int = 8
    candidate_k: int = 40  # per-retriever pool before fusion/reranking
    rrf_k: int = 60  # reciprocal-rank-fusion smoothing constant
    max_per_doc: int = 3  # stop one document from flooding the context
    neighbor_window: int = 0  # also pull N chunks either side of each hit

    # --- reranking ---------------------------------------------------------
    rerank: bool = False
    rerank_model: str = "claude-sonnet-5"
    rerank_effort: str = "low"

    # --- generation --------------------------------------------------------
    llm: str = "anthropic"  # anthropic | fake
    model: str = "claude-opus-5"
    max_tokens: int = 4096
    effort: str = "medium"  # low | medium | high | xhigh | max
    thinking: bool = True
    # Server-side refusal fallbacks: on a policy decline the API re-runs the
    # request on a fallback model inside the same call instead of giving up.
    refusal_fallbacks: bool = True
    # Hard ceiling on retrieved context, in characters. Chunks are dropped from
    # the bottom of the ranking until the context fits.
    context_char_budget: int = 24000
    # Refuse to answer unless at least one retrieved chunk clears this score.
    min_score: float = 0.0

    # --- misc ---------------------------------------------------------------
    corpus_dir: str = "corpus"

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path | None = None, **overrides: Any) -> Config:
        cfg = cls()
        toml_path = Path(path) if path else Path(CONFIG_FILENAME)
        if toml_path.is_file():
            with toml_path.open("rb") as fh:
                data = tomllib.load(fh)
            cfg = cfg.merged(data.get("ragcite", data))
        cfg = cfg.merged(cls._from_env())
        if overrides:
            cfg = cfg.merged({k: v for k, v in overrides.items() if v is not None})
        cfg.validate()
        return cfg

    @staticmethod
    def _from_env() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for f in fields(Config):
            raw = os.environ.get(f"RAGCITE_{f.name.upper()}")
            if raw is not None:
                out[f.name] = raw
        return out

    def merged(self, data: dict[str, Any]) -> Config:
        """Return a copy with `data` applied, coercing strings to field types."""
        types = {f.name: f.type for f in fields(Config)}
        current = {f.name: getattr(self, f.name) for f in fields(Config)}
        for key, value in data.items():
            if key not in types:
                raise ValueError(f"unknown config key: {key!r}")
            current[key] = _coerce(value, current[key])
        return Config(**current)

    def validate(self) -> None:
        if self.retrieval_mode not in {"dense", "lexical", "hybrid"}:
            raise ValueError(
                f"retrieval_mode must be dense|lexical|hybrid, got {self.retrieval_mode!r}"
            )
        if self.llm not in {"anthropic", "fake"}:
            raise ValueError(f"llm must be anthropic|fake, got {self.llm!r}")
        if self.effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"effort must be low|medium|high|xhigh|max, got {self.effort!r}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.top_k > self.candidate_k:
            raise ValueError("candidate_k must be >= top_k")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(Config)}


def _coerce(value: Any, like: Any) -> Any:
    """Coerce a TOML/env value to the type of the existing default."""
    if isinstance(like, bool):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(like, int) and not isinstance(like, bool):
        return int(value)
    if isinstance(like, float):
        return float(value)
    if isinstance(like, tuple):
        if isinstance(value, str):
            return tuple(p.strip() for p in value.split(",") if p.strip())
        return tuple(value)
    return value
