"""In-process backend using transformers directly.

Much slower than vLLM/SGLang and without paged attention, but it needs no
server and is handy on machines where the serving stack will not install.
Qwen3.5+ exposes multimodal weights through ``AutoModelForMultimodalLM``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from ..config import Config
from .base import Backend, BackendError, Completion, ToolCall, Usage

log = logging.getLogger(__name__)

# Qwen emits tool calls as <tool_call>{"name": ..., "arguments": {...}}</tool_call>
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


class TransformersBackend(Backend):
    def __init__(self, config: Config) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise BackendError(
                "the transformers backend needs extra packages: "
                "pip install 'newal[transformers]'"
            ) from exc

        self.config = config
        self._torch = torch

        log.info("loading %s in-process (this can take several minutes)", config.model.id)
        self._processor = AutoProcessor.from_pretrained(config.model.id)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            config.model.id,
            device_map="auto",
            dtype="auto",
        )
        self._model.eval()

    def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        enable_thinking: bool | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, Usage]:
        model_cfg = self.config.model
        thinking = model_cfg.enable_thinking if enable_thinking is None else enable_thinking

        template_kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": thinking,
        }
        if tools:
            template_kwargs["tools"] = tools

        inputs = self._processor.apply_chat_template(messages, **template_kwargs)
        inputs = inputs.to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        temp = model_cfg.temperature if temperature is None else temperature
        with self._torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens or model_cfg.max_output_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=model_cfg.top_p,
            )

        # generate() returns prompt + completion; keep only what was added.
        new_tokens = generated[0][prompt_tokens:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)
        usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=int(new_tokens.shape[-1]))
        return text, usage

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        raw, usage = self._generate(messages, tools, enable_thinking, temperature, max_tokens)

        reasoning = ""
        think_match = THINK_RE.search(raw)
        if think_match:
            reasoning = think_match.group(1).strip()
            raw = THINK_RE.sub("", raw, count=1)

        tool_calls: list[ToolCall] = []
        for index, match in enumerate(TOOL_CALL_RE.finditer(raw)):
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                log.warning("could not parse tool call: %r", match.group(1))
                continue
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {"__parse_error__": str(arguments)}
            tool_calls.append(
                ToolCall(id=f"call_{index}", name=payload.get("name", ""), arguments=arguments)
            )

        return Completion(
            text=TOOL_CALL_RE.sub("", raw).strip(),
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "stop",
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
        # No incremental decoding here; emit the finished text in one piece so
        # callers can treat both backends uniformly.
        text, _ = self._generate(messages, None, enable_thinking, temperature, max_tokens)
        yield THINK_RE.sub("", text).strip()

    def close(self) -> None:
        self._model = None
        self._processor = None
        if hasattr(self._torch, "cuda") and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
