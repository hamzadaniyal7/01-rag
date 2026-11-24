"""ragcite -- a small, readable RAG pipeline with real citations.

The public surface is deliberately tiny:

    from ragcite import Rag, Config

    rag = Rag(Config.load())
    rag.ingest(["corpus/"])
    answer = rag.ask("How long do refunds take?")
    print(answer.render())
"""

from ragcite.config import Config
from ragcite.pipeline import Rag
from ragcite.types import Answer, Chunk, Citation, Document, Hit

__all__ = ["Rag", "Config", "Answer", "Chunk", "Citation", "Document", "Hit"]
__version__ = "0.1.0"
