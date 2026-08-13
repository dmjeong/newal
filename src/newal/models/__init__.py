"""Multi-model orchestration: a pool of specialists and a router over them.

The design follows two findings from recent work rather than the intuition
that "more models is better":

* **Ensembling correlated models barely helps.** Across 67 frontier models,
  combining via routing, voting, or mixture-of-agents cannot beat ``1 - beta``,
  where beta is the rate at which every member is wrong on the same query, and
  measured co-failure runs ~2.5x worse than a Gaussian-copula estimate predicts
  (arXiv:2606.27288). Stacking Qwen at three sizes is close to Self-MoA: same
  pretraining data, same blind spots. So this pool holds *specialists doing
  different jobs*, not the same job at different sizes.

* **Cascades pay for themselves on cost, not accuracy** (arXiv:2606.27457,
  arXiv:2605.18796). Serving cheaply and escalating on evidence saves latency
  and VRAM; the accuracy comes from escalating on *execution results*, which
  are the only signal here uncorrelated with model error.
"""

from __future__ import annotations

from .pool import ModelPool
from .retrieval import EmbeddingClient, RerankClient, RetrievalError, cosine_similarity
from .roles import HEAVY_ONLY_ROLES, ROLE_DIFFICULTY, Role
from .router import Route, Router, RouteSignals, difficulty_score

__all__ = [
    "HEAVY_ONLY_ROLES",
    "ROLE_DIFFICULTY",
    "EmbeddingClient",
    "ModelPool",
    "RerankClient",
    "RetrievalError",
    "Role",
    "Route",
    "RouteSignals",
    "Router",
    "cosine_similarity",
    "difficulty_score",
]
