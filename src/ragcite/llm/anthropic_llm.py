"""The Anthropic provider.

Defaults chosen here, and why:

* ``claude-opus-5`` -- the strongest current model; grounded answering is
  cheap in tokens but unforgiving about instruction-following, and the whole
  value of this project is an answer you can trust.
* **adaptive thinking** (``thinking={"type": "adaptive"}``) -- the model decides
  when reasoning is warranted. ``budget_tokens`` is gone on this generation and
  returns a 400.
* **effort = medium** -- single-hop extraction over 8 chunks does not need
  ``high``; raise it in ``ragcite.toml`` for multi-hop corpora and measure the
  difference with ``ragcite eval`` rather than guessing.
* **streaming** -- avoids HTTP timeouts on long answers and lets the CLI print
  tokens as they arrive. ``get_final_message()`` still gives the whole message.
* **server-side refusal fallbacks** -- if a policy classifier declines the
  request, the API re-runs it on a fallback model inside the same call instead
  of handing back nothing. Turn off with ``refusal_fallbacks = false``.
"""

from __future__ import annotations

import os
from typing import Any

from ragcite.llm.base import LLM, LLMResult

FALLBACK_BETA = "server-side-fallback-2026-07-01"

# `output_config.effort` is not accepted by every model (Haiku 4.5 and the 4.5
# generation reject it). Sending it anyway is a 400, so it is filtered here
# rather than leaving a trap for anyone who swaps the model in ragcite.toml.
NO_EFFORT_MODELS = ("haiku", "-4-5")


class AnthropicLLM(LLM):
    name = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5",
        *,
        effort: str = "medium",
        thinking: bool = True,
        refusal_fallbacks: bool = True,
        stream_to: Any = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError("pip install anthropic") from exc

        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            # Not fatal: the SDK also resolves an `ant auth login` profile from
            # ~/.config/anthropic. Only the constructor below can tell for sure.
            pass

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.refusal_fallbacks = refusal_fallbacks
        self.stream_to = stream_to

    # ------------------------------------------------------------------ api
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 4096,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResult:
        output_config: dict[str, Any] = {}
        if self.effort and self._supports_effort():
            output_config["effort"] = self.effort
        if json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": json_schema}

        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if output_config:
            params["output_config"] = output_config
        if self.thinking:
            params["thinking"] = {"type": "adaptive"}

        try:
            if self.refusal_fallbacks:
                message = self._stream(
                    beta=True, betas=[FALLBACK_BETA], fallbacks="default", **params
                )
            else:
                message = self._stream(beta=False, **params)
        except self._anthropic.APIStatusError as exc:
            # A proxy or an older deployment may reject the fallback parameter;
            # retry once without it rather than failing the whole query.
            if self.refusal_fallbacks and exc.status_code == 400:
                message = self._stream(beta=False, **params)
            else:
                raise

        return _to_result(message)

    def _supports_effort(self) -> bool:
        return not any(marker in self.model for marker in NO_EFFORT_MODELS)

    def count_tokens(self, system: str, user: str) -> int:
        """Exact prompt size, straight from the API (never a local tokenizer)."""
        resp = self.client.messages.count_tokens(
            model=self.model,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return int(resp.input_tokens)

    # ------------------------------------------------------------ internals
    def _stream(self, *, beta: bool, **params: Any) -> Any:
        endpoint = self.client.beta.messages if beta else self.client.messages
        with endpoint.stream(**params) as stream:
            if self.stream_to is not None:
                for text in stream.text_stream:
                    self.stream_to.write(text)
                    self.stream_to.flush()
            return stream.get_final_message()


def _to_result(message: Any) -> LLMResult:
    text = "".join(block.text for block in message.content if block.type == "text")
    refused = message.stop_reason == "refusal"
    usage: dict[str, Any] = {}
    if getattr(message, "usage", None) is not None:
        usage = {
            "input_tokens": getattr(message.usage, "input_tokens", None),
            "output_tokens": getattr(message.usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(message.usage, "cache_read_input_tokens", None),
        }
    # One `fallback` block per model that declined this turn.
    switches = [f"{b.from_.model} -> {b.to.model}" for b in message.content if b.type == "fallback"]
    if switches:
        usage["fallbacks"] = switches
    if refused:
        details = getattr(message, "stop_details", None)
        usage["refusal_category"] = getattr(details, "category", None) if details else None
    return LLMResult(
        text=text,
        model=getattr(message, "model", ""),
        usage={k: v for k, v in usage.items() if v is not None},
        refused=refused,
        stop_reason=message.stop_reason,
    )
