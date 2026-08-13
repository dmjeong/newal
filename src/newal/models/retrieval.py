"""Embedding and reranking clients.

These are the pool members that most clearly earn their VRAM. The co-failure
analysis (arXiv:2606.27288) is blunt about same-family ensembles: gains come
from members failing on *different* inputs, and a 4B Qwen and a 35B Qwen asked
the same coding question fail together far more often than their sizes suggest.
A retrieval model is not doing the coder's job at all, so its errors are
structurally independent -- which is why hybrid retrieval plus reranking beats
adding a third generator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx
from openai import APIConnectionError, APIStatusError, OpenAI

from ..config import Config, ModelSpec

log = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    pass


@dataclass
class Scored:
    index: int
    score: float


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn texts into vectors.

    Declared as a protocol so the indexer and the routing classifier can be
    typed against the capability rather than against ``object``, and so tests
    can substitute a deterministic stand-in.
    """

    def embed(self, texts: list[str], batch_size: int = ...) -> list[list[float]]:
        ...


@runtime_checkable
class Reranker(Protocol):
    """Anything that can order documents by relevance to a query."""

    def rerank(self, query: str, documents: list[str], *, top_k: int) -> list[Scored]:
        ...


class EmbeddingClient:
    """Dense embeddings over the OpenAI-compatible ``/embeddings`` endpoint."""

    def __init__(self, config: Config, key: str, spec: ModelSpec) -> None:
        self.key = key
        self.spec = spec
        self._client = OpenAI(
            base_url=spec.base_url,
            api_key=spec.api_key or "EMPTY",
            timeout=float(config.runtime.request_timeout_s),
            max_retries=2,
        )

    def embed(self, texts: list[str], *, batch_size: int = 32) -> list[list[float]]:
        """Embed texts, batching so a large index pass does not blow the request size."""
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                response = self._client.embeddings.create(model=self.spec.id, input=batch)
            except (APIConnectionError, APIStatusError) as exc:
                raise RetrievalError(f"embedding request to {self.key!r} failed: {exc}") from exc
            # The API does not guarantee ordering; sort by the returned index.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)

        if len(vectors) != len(texts):
            raise RetrievalError(
                f"embedding server returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors


class RerankClient:
    """Cross-encoder reranking.

    vLLM exposes this as ``/rerank`` (Jina-compatible) on newer builds and as
    ``/score`` on older ones. We try one and fall back to the other, since which
    is available depends on the user's installed version rather than anything
    we control.
    """

    def __init__(self, config: Config, key: str, spec: ModelSpec) -> None:
        self.key = key
        self.spec = spec
        self._root = spec.base_url.rstrip("/").removesuffix("/v1")
        self._timeout = float(config.runtime.request_timeout_s)
        self._endpoint: str | None = None

    def _post(self, path: str, payload: dict) -> httpx.Response:
        return httpx.post(f"{self._root}{path}", json=payload, timeout=self._timeout)

    def rerank(self, query: str, documents: list[str], *, top_k: int) -> list[Scored]:
        """Return the best ``top_k`` documents as (original index, score)."""
        if not documents:
            return []

        attempts = [self._endpoint] if self._endpoint else ["/rerank", "/score"]
        last_error: Exception | None = None

        for path in attempts:
            payload = (
                {"model": self.spec.id, "query": query, "documents": documents, "top_n": top_k}
                if path == "/rerank"
                else {"model": self.spec.id, "text_1": query, "text_2": documents}
            )
            try:
                response = self._post(path, payload)
                if response.status_code == 404:
                    last_error = RetrievalError(f"{path} not available")
                    continue
                response.raise_for_status()
                scored = _parse_rerank(response.json())
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                last_error = exc
                continue

            self._endpoint = path  # remember what worked
            scored.sort(key=lambda item: -item.score)
            return scored[:top_k]

        raise RetrievalError(f"rerank request to {self.key!r} failed: {last_error}")


def _parse_rerank(payload: dict) -> list[Scored]:
    """Normalise the two response shapes into ``Scored`` entries."""
    rows = payload.get("results") or payload.get("data")
    if not isinstance(rows, list):
        raise ValueError(f"unexpected rerank response: {list(payload)[:5]}")

    scored: list[Scored] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        score = row.get("relevance_score", row.get("score"))
        if score is None:
            continue
        scored.append(Scored(index=int(row.get("index", position)), score=float(score)))

    if not scored:
        raise ValueError("rerank response contained no scores")
    return scored


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity without a numpy round trip for a single pair."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
