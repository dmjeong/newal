"""Backend that talks to a local vLLM / SGLang OpenAI-compatible server."""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from openai import OpenAI
from openai import APIConnectionError, APIStatusError

from ..config import Config
from .base import Backend, BackendError, Completion, ToolCall, Usage
from .launcher import ServerProcess, ensure_server

log = logging.getLogger(__name__)


class OpenAICompatBackend(Backend):
    """Wraps the OpenAI SDK against a locally served Qwen model.

    Handles the two Qwen-specific wrinkles: the ``enable_thinking`` switch
    (passed through ``chat_template_kwargs``) and the ``reasoning_content``
    field that carries the thinking trace separately from the answer.
    """

    def __init__(self, config: Config, *, autostart: bool = True) -> None:
        self.config = config
        self._server: ServerProcess | None = None

        if autostart:
            self._server = ensure_server(
                config.model,
                config.backend,
                log_path=".newal/server.log",
            )

        self._client = OpenAI(
            base_url=config.backend.base_url,
            api_key=config.backend.api_key or "EMPTY",
            timeout=float(config.backend.request_timeout_s),
            max_retries=2,
        )
        self._model_id = self._resolve_served_model_id(config.model.id)

    def _resolve_served_model_id(self, configured: str) -> str:
        """Ask the server what name it serves the model under.

        vLLM defaults to the ``--model`` value, but a manually started server
        may use ``--served-model-name``. Asking avoids a confusing 404.
        """
        try:
            models = self._client.models.list()
        except (APIConnectionError, APIStatusError) as exc:
            raise BackendError(
                f"cannot reach inference server at {self.config.backend.base_url}: {exc}"
            ) from exc

        served = [m.id for m in models.data]
        if configured in served:
            return configured
        if len(served) == 1:
            log.info("server serves %r; using that instead of %r", served[0], configured)
            return served[0]
        raise BackendError(
            f"model {configured!r} is not served here. Available: {served or '(none)'}"
        )

    # ---- request construction -------------------------------------------------

    def _base_kwargs(
        self,
        enable_thinking: bool | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        model_cfg = self.config.model
        thinking = model_cfg.enable_thinking if enable_thinking is None else enable_thinking
        return {
            "model": self._model_id,
            "temperature": model_cfg.temperature if temperature is None else temperature,
            "top_p": model_cfg.top_p,
            "max_tokens": max_tokens or model_cfg.max_output_tokens,
            # vLLM and SGLang both forward this into the Jinja chat template.
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
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
            raise BackendError(f"completion request failed: {exc}") from exc

        if not response.choices:
            raise BackendError("server returned no choices")

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
            raise BackendError(f"streaming request failed: {exc}") from exc

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
            self._server = None


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
