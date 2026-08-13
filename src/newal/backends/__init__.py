"""Inference backends.

Backends serve a single model. Assembling several of them into a working pool
is :mod:`newal.models`.
"""

from __future__ import annotations

from ..config import Config, ModelSpec
from .base import Backend, BackendError, Completion, ToolCall, Usage


def build_backend(config: Config, key: str, spec: ModelSpec) -> Backend:
    """Instantiate a backend for one pool member."""
    if config.runtime.kind == "openai_compat":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(config, key, spec)

    if config.runtime.kind == "transformers":
        from .transformers_local import TransformersBackend

        return TransformersBackend(config, key, spec)

    raise BackendError(f"unknown backend kind: {config.runtime.kind}")


__all__ = [
    "Backend",
    "BackendError",
    "Completion",
    "ToolCall",
    "Usage",
    "build_backend",
]
