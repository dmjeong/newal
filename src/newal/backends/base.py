"""Backend protocol shared by the server-backed and in-process implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


@dataclass
class Completion:
    """One assistant turn: prose, optional reasoning trace, optional tool calls."""

    text: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class BackendError(RuntimeError):
    """Raised when the inference backend is unreachable or returns garbage."""


@runtime_checkable
class Backend(Protocol):
    """Minimal surface the agent loop needs from an inference engine."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Run one non-streaming completion."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        enable_thinking: bool | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Yield response text incrementally. Tool calls are not streamed."""

    def close(self) -> None:
        """Release any resources (server processes, model weights)."""
