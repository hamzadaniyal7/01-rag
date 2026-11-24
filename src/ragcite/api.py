"""A thin HTTP layer over the same pipeline (`pip install -e '.[api]'`).

Deliberately thin: no auth, no rate limiting, no streaming SSE. It exists to
show the pipeline is a library first and a CLI second, and to make the project
easy to point a front-end at. Harden before exposing it.
"""

from __future__ import annotations

from typing import Any

from ragcite.config import Config
from ragcite.pipeline import Rag


def create_app(cfg: Config | None = None) -> Any:
    try:
        from fastapi import FastAPI, HTTPException
        from pydantic import BaseModel, Field
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError("pip install -e '.[api]'") from exc

    cfg = cfg or Config.load()
    app = FastAPI(
        title="ragcite", version="0.1.0", description="RAG over your own documents, with citations."
    )

    class AskRequest(BaseModel):
        question: str = Field(min_length=1)
        k: int | None = None
        mode: str | None = None
        rerank: bool | None = None

    # One Rag per process: the store is a SQLite handle and the embedder may
    # hold a loaded model, so neither should be rebuilt per request.
    rag = Rag(cfg)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **rag.stats()}

    @app.post("/search")
    def search(req: AskRequest) -> dict[str, Any]:
        hits = rag.search(req.question, k=req.k, mode=req.mode)
        return {
            "hits": [
                {
                    "chunk_id": h.chunk.chunk_id,
                    "path": h.chunk.doc_path,
                    "locator": h.chunk.locator,
                    "score": h.score,
                    "text": h.chunk.text,
                }
                for h in hits
            ]
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict[str, Any]:
        try:
            answer = rag.ask(req.question, k=req.k, mode=req.mode, rerank=req.rerank)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        import json as _json

        return _json.loads(answer.to_json())

    return app
