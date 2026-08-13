"""Backend that talks to one local vLLM / SGLang OpenAI-compatible server."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from openai import APIConnectionError, APIStatusError, OpenAI

from ..config import Config, ModelSpec
from .base import Backend, BackendError, Completion, ToolCall, Usage

log = logging.getLogger(__name__)


class OpenAICompatBackend(Backend):
    """Wraps the OpenAI SDK against one served Qwen model.

    Handles the two Qwen-specific wrinkles: the ``enable_thinking`` switch
    (passed through ``chat_template_kwargs``) and the ``reasoning_content``
    field that carries the thinking trace separately from the answer.

    Server lifecycle is the pool's job, not this class's -- several backends
    may share one process, and none of them should be able to kill it.
    """

    def __init__(self, config: Config, key: str, spec: ModelSpec) -> None:
        self.config = config
        self.key = key
        self.spec = spec

        self._client = OpenAI(
            base_url=spec.base_url,
            api_key=spec.api_key or "EMPTY",
            timeout=float(config.runtime.request_timeout_s),
            max_retries=2,
        )
        self._model_id = self._resolve_served_model_id(spec.id)

    def _resolve_served_model_id(self, configured: str) -> str:
        """Ask the server what name it serves the model under.

        vLLM defaults to the ``--model`` value, but a manually started server
        may use a different ``--served-model-name``. Asking avoids a confusing 404.
        """
        try:
            models = self._client.models.list()
        except (APIConnectionError, APIStatusError) as exc:
            raise BackendError(
                f"cannot reach server for {self.key!r} at {self.spec.base_url}: {exc}"
            ) from exc

        served = [m.id for m in models.data]
        if configured in served:
            return configured
        if len(served) == 1:
            log.info("server serves %r; using that instead of %r", served[0], configured)
            return served[0]
        raise BackendError(
            f"model {configured!r} is not served at {self.spec.base_url}. "
            f"Available: {served or '(none)'}"
        )

    # ---- request construction -------------------------------------------------

    def _base_kwargs(
        self,
        enable_thinking: bool | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        generation = self.config.generation
        return {
            "model": self._model_id,
            "temperature": generation.temperature if temperature is None else temperature,
            "top_p": generation.top_p,
            "max_tokens": max_tokens or generation.max_output_tokens,
            # vLLM and SGLang both forward this into the Jinja chat template.
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": True if enable_thinking is None else enable_thinking
                }
            },
        }

    # ---- Backend protocol -----------------------------------------------------

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        kwargs = self._base_kwargs(enable_thinking, temperature, max_tokens)
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = self._client.chat.completions.create(messages=messages, **kwargs)
        except (APIConnectionError, APIStatusError) as exc:
            raise BackendError(f"completion request to {self.key!r} failed: {exc}") from exc

        if not response.choices:
            raise BackendError(f"server for {self.key!r} returned no choices")

        choice = response.choices[0]
        message = choice.message

        usage = Usage()
        if response.usage:
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )

        return Completion(
            text=message.content or "",
            # Present when the server was started with --reasoning-parser qwen3.
            reasoning=getattr(message, "reasoning_content", None) or "",
            tool_calls=_parse_tool_calls(message),
            finish_reason=choice.finish_reason or "stop",
            usage=usage,
            model_key=self.key,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        kwargs = self._base_kwargs(enable_thinking, temperature, max_tokens)
        try:
            stream = self._client.chat.completions.create(
                messages=messages, stream=True, **kwargs
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if piece:
                    yield piece
        except (APIConnectionError, APIStatusError) as exc:
            raise BackendError(f"streaming request to {self.key!r} failed: {exc}") from exc

    def close(self) -> None:
        # The pool owns the server process; nothing to release here.
        return None


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    """Normalise the SDK's tool-call payload, tolerating malformed JSON args.

    Local models occasionally emit not-quite-JSON arguments. Surfacing that as
    a tool error the model can see and retry beats crashing the session.
    """
    raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        raw_args = raw.function.arguments or "{}"
        try:
            parsed = json.loads(raw_args)
        except json.JSONDecodeError:
            log.warning("tool %s returned unparseable arguments: %r", raw.function.name, raw_args)
            parsed = {"__parse_error__": raw_args}
        if not isinstance(parsed, dict):
            parsed = {"__parse_error__": raw_args}
        calls.append(
            ToolCall(
                id=raw.id or f"call_{index}",
                name=raw.function.name,
                arguments=parsed,
            )
        )
    return calls
