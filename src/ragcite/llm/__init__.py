"""LLM provider registry."""

from __future__ import annotations

from typing import Any

from ragcite.llm.base import LLM, LLMResult
from ragcite.llm.fake import FakeLLM

__all__ = ["LLM", "LLMResult", "FakeLLM", "get_llm"]


def get_llm(kind: str, **kwargs: Any) -> LLM:
    if kind == "fake":
        return FakeLLM()
    if kind == "anthropic":
        from ragcite.llm.anthropic_llm import AnthropicLLM

        return AnthropicLLM(**kwargs)
    raise ValueError(f"unknown llm provider {kind!r}; expected anthropic|fake")
