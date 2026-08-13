"""The model pool: several specialists, started and shut down together."""

from __future__ import annotations

import logging
from typing import Callable

from ..backends import Backend, BackendError, build_backend
from ..backends.launcher import ServerProcess, ensure_server, is_server_up
from ..config import Config, ModelSpec
from .retrieval import EmbeddingClient, RerankClient
from .roles import Role
from .router import Route, Router

log = logging.getLogger(__name__)

ProgressCallback = Callable[[str], None]


class ModelPool:
    """Owns every server process and backend for a session.

    Members are started eagerly (so a VRAM problem surfaces at startup rather
    than mid-task) but backends are built lazily per member.
    """

    def __init__(
        self,
        config: Config,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.config = config
        self._notify = on_progress or (lambda _message: None)
        self._servers: list[ServerProcess] = []
        self._backends: dict[str, Backend] = {}
        self._embedder: EmbeddingClient | None = None
        self._reranker: RerankClient | None = None

        self._start_servers()
        self.router = self._build_router()

    # ---- startup --------------------------------------------------------------

    def _start_servers(self) -> None:
        for key, spec in self.config.enabled_models().items():
            if is_server_up(spec.base_url):
                self._notify(f"{key}: reusing server at {spec.base_url}")
                continue
            self._notify(f"{key}: starting {spec.id}")
            server = ensure_server(self.config, key, spec)
            if server is not None:
                self._servers.append(server)

    def _build_router(self) -> Router:
        generators = self.config.enabled_models(task="generate")
        return Router(
            tiers=[(key, spec.tier) for key, spec in generators.items()],
            strategy=self.config.router.strategy,
            escalate_threshold=self.config.router.escalate_threshold,
            thinking_mode=self.config.router.thinking.mode,
            thinking_threshold=self.config.router.thinking.threshold,
        )

    # ---- access ---------------------------------------------------------------

    def backend(self, key: str) -> Backend:
        """Get (building on first use) the backend for a pool member."""
        if key not in self._backends:
            spec = self.config.models.get(key)
            if spec is None or not spec.enabled:
                raise BackendError(f"model {key!r} is not an enabled pool member")
            self._backends[key] = build_backend(self.config, key, spec)
        return self._backends[key]

    def for_route(self, route: Route) -> Backend:
        """Resolve a routing decision to a backend, degrading if it is unavailable.

        A pool member can fail at request time even though its server started.
        Falling back to another tier keeps the session alive; failing the whole
        turn because the cheap model died would be worse than answering slowly.
        """
        try:
            return self.backend(route.model_key)
        except BackendError as exc:
            fallback = self.router.strongest
            if route.model_key == fallback:
                raise
            log.warning("model %r unavailable (%s); falling back to %r",
                        route.model_key, exc, fallback)
            self._notify(f"{route.model_key} unavailable, using {fallback}")
            return self.backend(fallback)

    @property
    def embedder(self) -> EmbeddingClient | None:
        if self._embedder is None:
            found = self.config.first_model("embed")
            if found is None:
                return None
            key, spec = found
            self._embedder = EmbeddingClient(self.config, key, spec)
        return self._embedder

    @property
    def reranker(self) -> RerankClient | None:
        if self._reranker is None:
            found = self.config.first_model("rerank")
            if found is None:
                return None
            key, spec = found
            self._reranker = RerankClient(self.config, key, spec)
        return self._reranker

    def describe(self) -> list[str]:
        """One line per enabled member, for the startup banner."""
        lines: list[str] = []
        for key, spec in sorted(
            self.config.enabled_models().items(), key=lambda i: (i[1].task, i[1].tier)
        ):
            extras = []
            if spec.speculative_draft:
                extras.append(f"draft={spec.speculative_draft.split('/')[-1]}")
            if spec.task == "generate":
                extras.append(f"tier {spec.tier}")
            suffix = f" [{', '.join(extras)}]" if extras else ""
            lines.append(f"{key}: {spec.id} ({spec.task}){suffix}")
        return lines

    def close(self) -> None:
        for backend in self._backends.values():
            backend.close()
        self._backends.clear()
        for server in self._servers:
            server.stop()
        self._servers.clear()


__all__ = ["ModelPool", "Role", "Route", "Router", "ModelSpec"]
