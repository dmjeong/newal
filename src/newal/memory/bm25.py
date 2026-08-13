"""A small BM25 index.

Implemented here rather than pulled in as a dependency: it is ~60 lines, it
keeps the base install free of native wheels, and it is the retrieval floor the
optional dense reranker builds on top of.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

K1 = 1.5
B = 0.75

# Split on non-alphanumerics but keep intra-identifier structure, so
# "prepare_video" also matches a query for "video".
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, with snake_case and camelCase split into parts."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        tokens.append(lowered)
        pieces = [p for chunk in raw.split("_") for p in _CAMEL_RE.split(chunk) if p]
        if len(pieces) > 1:
            tokens.extend(p.lower() for p in pieces)
    return tokens


@dataclass
class BM25Index:
    """In-memory BM25 over a fixed document set."""

    doc_ids: list[str] = field(default_factory=list)
    _term_freqs: list[Counter] = field(default_factory=list)
    _doc_lengths: list[int] = field(default_factory=list)
    _doc_freq: Counter = field(default_factory=Counter)
    _avg_length: float = 0.0

    @classmethod
    def build(cls, documents: dict[str, str]) -> "BM25Index":
        index = cls()
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            counts = Counter(tokens)
            index.doc_ids.append(doc_id)
            index._term_freqs.append(counts)
            index._doc_lengths.append(len(tokens))
            for term in counts:
                index._doc_freq[term] += 1
        total = sum(index._doc_lengths)
        index._avg_length = total / len(index.doc_ids) if index.doc_ids else 0.0
        return index

    def __len__(self) -> int:
        return len(self.doc_ids)

    def _idf(self, term: str) -> float:
        n_docs = len(self.doc_ids)
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # Robertson/Sparck Jones idf with +1 to keep it non-negative for terms
        # that appear in more than half the corpus.
        return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 8) -> list[tuple[str, float]]:
        """Return ``(doc_id, score)`` for the best matches, highest first."""
        if not self.doc_ids or top_k <= 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[str, float]] = []
        for position, doc_id in enumerate(self.doc_ids):
            counts = self._term_freqs[position]
            length = self._doc_lengths[position]
            norm = K1 * (1 - B + B * (length / self._avg_length if self._avg_length else 1.0))

            score = 0.0
            for term in query_terms:
                freq = counts.get(term, 0)
                if not freq:
                    continue
                score += self._idf(term) * (freq * (K1 + 1)) / (freq + norm)
            if score > 0:
                scores.append((doc_id, score))

        scores.sort(key=lambda item: (-item[1], item[0]))
        return scores[:top_k]
