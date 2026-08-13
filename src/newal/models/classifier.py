"""Learned query classification for routing.

The v1 router scored queries with hand-written patterns. That is free and
deterministic, but it only knows the phrasings someone thought to write down,
and maintaining a regex list is not a strategy. This module adds the two
classifiers a hosted assistant would use:

* :class:`SemanticClassifier` -- embed the query, compare against labelled
  exemplars, vote among the nearest. Roughly one embedding call (~20ms) and it
  generalises to phrasings nobody enumerated.
* :class:`LLMClassifier` -- ask the cheapest model to judge. Slower (one short
  completion) but the most general fallback.

They are layered rather than stacked: the heuristic decides on its own unless
its score lands in an uncertainty band around the threshold, and only then does
the *routing decision itself* escalate. Most queries are obviously easy or
obviously hard, so the expensive classifier runs on the few that are neither --
the same cascade logic the router applies to the work, applied to the choice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .retrieval import Embedder

log = logging.getLogger(__name__)

#: Labels a classifier can return. "strong" means route to the top tier.
STRONG = "strong"
CHEAP = "cheap"

#: Seed exemplars so semantic routing works before the user has any history.
#: Both languages, because the assistant answers in the user's language and a
#: Korean-only user would otherwise start with an empty index.
SEED_EXEMPLARS: list[tuple[str, str]] = [
    # -- cheap: mechanical, local, single-step -------------------------------
    ("rename this variable to something clearer", CHEAP),
    ("fix the typo in this comment", CHEAP),
    ("add a docstring to this function", CHEAP),
    ("format this file", CHEAP),
    ("add a log line here", CHEAP),
    ("sort these imports", CHEAP),
    ("change this string literal", CHEAP),
    ("what does this file contain?", CHEAP),
    ("변수 이름 바꿔줘", CHEAP),
    ("오타 고쳐줘", CHEAP),
    ("이 함수에 주석 달아줘", CHEAP),
    ("import 정리해줘", CHEAP),
    ("로그 한 줄 추가해줘", CHEAP),
    ("이 파일 뭐 하는 거야?", CHEAP),
    # -- strong: investigative, cross-file, design ---------------------------
    ("why does this test fail intermittently?", STRONG),
    ("find the root cause of this deadlock", STRONG),
    ("there is a race condition somewhere, track it down", STRONG),
    ("refactor this module to remove the circular dependency", STRONG),
    ("this leaks memory under load, find out where", STRONG),
    ("redesign the caching layer for concurrent access", STRONG),
    ("migrate this code to the new API across the project", STRONG),
    ("the output is wrong for edge cases, figure out why", STRONG),
    ("review this for security problems", STRONG),
    ("이 버그 원인이 뭐야?", STRONG),
    ("가끔 실패하는 테스트 원인 찾아줘", STRONG),
    ("순환 참조 없애도록 리팩터링해줘", STRONG),
    ("메모리 누수 어디서 나는지 찾아줘", STRONG),
    ("동시성 문제 있는지 검토해줘", STRONG),
    ("아키텍처 다시 설계하자", STRONG),
    ("전체 프로젝트에서 이 API 마이그레이션해줘", STRONG),
]


@dataclass
class Verdict:
    """A classifier's opinion about one query."""

    label: str
    confidence: float
    source: str

    @property
    def prefer_strong(self) -> bool:
        return self.label == STRONG


@runtime_checkable
class Classifier(Protocol):
    def classify(self, prompt: str) -> Verdict | None:
        """Return a verdict, or ``None`` when this classifier cannot decide."""


class SemanticClassifier:
    """k-nearest-neighbour vote over embedded exemplars.

    Exemplars come from two places: a seed list, and outcomes observed on this
    user's own repository. The second source is what a hosted router cannot
    have -- it learns which requests *in this codebase* turned out to need the
    strong model.
    """

    def __init__(
        self,
        embedder: Embedder,
        *,
        exemplars: list[tuple[str, str]] | None = None,
        neighbours: int = 5,
    ) -> None:
        self._embedder = embedder
        self._neighbours = max(1, neighbours)
        self._texts: list[str] = []
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None
        self.fit(exemplars if exemplars is not None else SEED_EXEMPLARS)

    def fit(self, exemplars: list[tuple[str, str]]) -> None:
        """Embed and cache the exemplar set. Safe to call again to relearn."""
        self._texts = [text for text, _ in exemplars]
        self._labels = [label for _, label in exemplars]
        self._matrix = None
        if not self._texts:
            return

        try:
            vectors = self._embedder.embed(self._texts)
        except Exception as exc:  # noqa: BLE001 - routing must never hard-fail
            log.warning("could not embed routing exemplars: %s", exc)
            self._texts, self._labels = [], []
            return

        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-9
        self._matrix = matrix / norms

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else len(self._labels)

    def classify(self, prompt: str) -> Verdict | None:
        if self._matrix is None or not prompt.strip():
            return None

        try:
            raw = self._embedder.embed([prompt])[0]
            query = np.asarray(raw, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001 - degrade to the heuristic
            log.warning("semantic routing unavailable: %s", exc)
            return None

        norm = float(np.linalg.norm(query)) or 1e-9
        sims = self._matrix @ (query / norm)

        k = min(self._neighbours, len(sims))
        top = np.argsort(-sims)[:k]

        # Similarity-weighted vote. Clamping at zero keeps an opposing exemplar
        # from voting negatively just because it points the other way.
        votes: dict[str, float] = {}
        for index in top:
            weight = max(0.0, float(sims[index]))
            votes[self._labels[index]] = votes.get(self._labels[index], 0.0) + weight

        if not votes:
            return None

        ordered = sorted(votes.items(), key=lambda item: -item[1])
        label, best = ordered[0]

        # Nothing in the exemplar set resembles this query. Abstain rather than
        # return whichever label happened to sort first at zero weight -- a
        # confident-looking label with no evidence behind it is worse than none.
        if best <= 1e-6:
            return None

        runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
        total = best + runner_up
        # Confidence is the margin between the top two labels, so a 5-0 vote
        # reads as certain and a 3-2 vote reads as a coin flip.
        confidence = (best - runner_up) / total if total > 0 else 0.0
        return Verdict(label=label, confidence=float(confidence), source="semantic")


CLASSIFY_PROMPT = """\
Classify the following coding request by how much reasoning it needs.

Answer with exactly one word:
SIMPLE  - a mechanical, local change or a direct lookup (rename, typo, \
formatting, reading one file)
COMPLEX - needs investigation, spans several files, or involves debugging, \
design, concurrency, or security

Request:
{prompt}

Answer:"""

_ANSWER_RE = re.compile(r"\b(SIMPLE|COMPLEX)\b", re.IGNORECASE)


class LLMClassifier:
    """Ask the cheapest model to judge. The most general fallback, and the slowest."""

    def __init__(self, backend: object, *, max_prompt_chars: int = 2000) -> None:
        self._backend = backend
        self._max_prompt_chars = max_prompt_chars

    def classify(self, prompt: str) -> Verdict | None:
        if not prompt.strip():
            return None

        message = CLASSIFY_PROMPT.format(prompt=prompt[: self._max_prompt_chars])
        try:
            completion = self._backend.complete(  # type: ignore[attr-defined]
                [{"role": "user", "content": message}],
                enable_thinking=False,   # a one-word answer never needs a think block
                temperature=0.0,
                max_tokens=8,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to the heuristic
            log.warning("llm routing unavailable: %s", exc)
            return None

        match = _ANSWER_RE.search(completion.text or "")
        if match is None:
            log.debug("llm classifier gave no usable answer: %r", completion.text)
            return None

        label = STRONG if match.group(1).upper() == "COMPLEX" else CHEAP
        # A parsed answer is taken at face value; the model was asked for a
        # binary judgement and gave one.
        return Verdict(label=label, confidence=1.0, source="llm")


class LayeredClassifier:
    """Try classifiers in order, stopping at the first confident verdict."""

    def __init__(
        self, classifiers: list[Classifier], *, min_confidence: float = 0.3
    ) -> None:
        self._classifiers = classifiers
        self._min_confidence = min_confidence

    def classify(self, prompt: str) -> Verdict | None:
        for classifier in self._classifiers:
            verdict = classifier.classify(prompt)
            if verdict is not None and verdict.confidence >= self._min_confidence:
                return verdict
        return None


def label_from_outcome(
    *, routed_to_strong: bool, escalated: bool, verification_failed: bool
) -> str | None:
    """Derive a training label from what actually happened, or ``None``.

    Only outcomes that carry real information are labelled:

    * A cheap route that escalated or failed verification is direct evidence it
      should have been strong.
    * A cheap route that finished clean confirms cheap was enough.
    * A strong route that finished clean says nothing -- we never find out
      whether the cheap model would also have managed, so recording it would
      just bias the exemplar set toward "strong" over time.
    """
    if routed_to_strong:
        return STRONG if (escalated or verification_failed) else None
    if escalated or verification_failed:
        return STRONG
    return CHEAP
