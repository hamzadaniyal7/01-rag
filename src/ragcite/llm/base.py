"""The LLM contract, kept narrow on purpose.

One method, one return type. Everything model-specific (thinking, effort,
refusal fallbacks, streaming) lives inside the Anthropic provider; everything
pipeline-specific (prompt construction, citation parsing) lives outside it.
That split is what lets the whole test suite run against `FakeLLM` with no
network and no API key.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class LLMResult:
    text: str
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    # True when the model declined on policy grounds (stop_reason == "refusal")
    # and no fallback model was able to answer either.
    refused: bool = False
    stop_reason: str | None = None


class LLM(ABC):
    name: str

    @abstractmethod
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        """Single-turn completion. `json_schema` constrains the output format."""
