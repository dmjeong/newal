"""Inference backends."""

from __future__ import annotations

from ..config import Config
from .base import Backend, BackendError, Completion, ToolCall, Usage


def build_backend(config: Config) -> Backend:
    """Instantiate the backend named by ``config.backend.kind``."""
    if config.backend.kind == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(config, autostart=config.backend.autostart)

    if config.backend.kind == "transformers":
        from .transformers_local import TransformersBackend

        return TransformersBackend(config)

    raise BackendError(f"unknown backend kind: {config.backend.kind}")


__all__ = [
    "Backend",
    "BackendError",
    "Completion",
    "ToolCall",
    "Usage",
    "build_backend",
]
