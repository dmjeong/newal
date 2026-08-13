"""Rank fusion for hybrid retrieval.

BM25 scores and cosine similarities live on incomparable scales -- BM25 is
unbounded and corpus-dependent, cosine is in [-1, 1] -- so blending the raw
numbers makes the weight meaningless. Reciprocal Rank Fusion combines the
*rankings* instead, which is scale-free and needs no per-corpus calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Standard RRF damping. Large enough that the top few ranks do not dominate
#: outright, small enough that deep results still contribute very little.
RRF_K = 60


@dataclass
class FusedHit:
    doc_id: str
    score: float
    #: Rank in each contributing list, for debugging why something surfaced.
    ranks: dict[str, int]


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    *,
    weights: dict[str, float] | None = None,
    k: int = RRF_K,
) -> list[FusedHit]:
    """Fuse several ranked ID lists into one.

    ``ranked_lists`` maps a source name to its ranking, best first. ``weights``
    scales each source's contribution; missing entries default to 1.0. A
    document absent from a list simply contributes nothing for that source,
    which is what makes this robust to lists of different lengths.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source, ordering in ranked_lists.items():
        weight = weights.get(source, 1.0)
        if weight == 0.0:
            continue
        for position, doc_id in enumerate(ordering):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + position + 1)
            ranks.setdefault(doc_id, {})[source] = position + 1

    fused = [
        FusedHit(doc_id=doc_id, score=score, ranks=ranks.get(doc_id, {}))
        for doc_id, score in scores.items()
    ]
    # Tie-break on id so the ordering is deterministic across runs.
    fused.sort(key=lambda hit: (-hit.score, hit.doc_id))
    return fused
