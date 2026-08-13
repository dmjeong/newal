"""Repository indexing and hybrid retrieval.

Retrieval runs in up to three stages, each optional and each degrading
gracefully when its model is not enabled:

1. **BM25** over chunk text -- always available, no model needed.
2. **Dense** cosine similarity against embeddings from the ``embed`` pool
   member, fused with BM25 by reciprocal rank.
3. **Cross-encoder rerank** of the fused candidates by the ``rerank`` member.

Indexing is incremental: files whose mtime and size are unchanged are skipped,
so re-indexing a large repo after one edit costs milliseconds.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import MemoryConfig
from .bm25 import BM25Index
from .fusion import reciprocal_rank_fusion
from .store import Chunk, MemoryStore

log = logging.getLogger(__name__)

# Skip anything that is probably not source we can read as text.
MAX_INDEXABLE_BYTES = 2_000_000


@dataclass
class IndexStats:
    files_indexed: int = 0
    files_skipped: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    chunks_embedded: int = 0


@dataclass
class Retrieved:
    chunk: Chunk
    score: float
    #: Which stages contributed: "bm25", "dense", "rerank".
    sources: tuple[str, ...] = ()


def chunk_text(
    path: str, text: str, *, chunk_lines: int, overlap_lines: int
) -> list[Chunk]:
    """Split a file into overlapping line ranges.

    Overlap keeps a function that straddles a boundary retrievable from either
    side, which matters because most useful chunks are function-sized.
    """
    if chunk_lines <= 0:
        raise ValueError("chunk_lines must be positive")
    if overlap_lines >= chunk_lines:
        raise ValueError("overlap_lines must be smaller than chunk_lines")

    lines = text.splitlines()
    if not lines:
        return []

    stride = chunk_lines - overlap_lines
    chunks: list[Chunk] = []
    start = 0
    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        body = "\n".join(lines[start:end])
        if body.strip():
            digest = hashlib.sha1(f"{path}:{start}:{body}".encode()).hexdigest()[:16]
            chunks.append(
                Chunk(
                    id=digest,
                    path=path,
                    start_line=start + 1,
                    end_line=end,
                    content=body,
                )
            )
        if end >= len(lines):
            break
        start += stride
    return chunks


def _is_excluded(rel_path: str, patterns: list[str]) -> bool:
    posix = rel_path.replace("\\", "/")
    for pattern in patterns:
        if fnmatch.fnmatch(posix, pattern):
            return True
        # Let "node_modules/**" also match a bare "node_modules" prefix.
        prefix = pattern.split("*", 1)[0].rstrip("/")
        if prefix and (posix == prefix or posix.startswith(prefix + "/")):
            return True
    return False


def discover_files(root: Path, config: MemoryConfig) -> list[Path]:
    """Collect files matching ``index_globs`` and not matching ``index_exclude``."""
    seen: set[Path] = set()
    for pattern in config.index_globs:
        for candidate in root.glob(pattern):
            if not candidate.is_file():
                continue
            rel = candidate.relative_to(root).as_posix()
            if _is_excluded(rel, config.index_exclude):
                continue
            seen.add(candidate)
    return sorted(seen)


class RepoIndex:
    """Owns the store, a lazily rebuilt BM25 index, and the retrieval models."""

    def __init__(
        self,
        root: Path,
        config: MemoryConfig,
        store: MemoryStore,
        *,
        embedder: object | None = None,
        reranker: object | None = None,
    ) -> None:
        self.root = root
        self.config = config
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self._bm25: BM25Index | None = None

    def attach_models(self, *, embedder: object | None, reranker: object | None) -> None:
        """Wire in retrieval models after the pool has started."""
        self.embedder = embedder
        self.reranker = reranker

    # ---- indexing -------------------------------------------------------------

    def refresh(self) -> IndexStats:
        """Bring the index in line with what is on disk."""
        stats = IndexStats()
        files = discover_files(self.root, self.config)
        present: set[str] = set()

        for path in files:
            rel = path.relative_to(self.root).as_posix()
            present.add(rel)
            try:
                info = path.stat()
            except OSError:
                continue

            if info.st_size > MAX_INDEXABLE_BYTES:
                stats.files_skipped += 1
                continue
            if self.store.file_is_current(rel, info.st_mtime, info.st_size):
                stats.files_skipped += 1
                continue

            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                stats.files_skipped += 1  # binary or unreadable
                continue

            chunks = chunk_text(
                rel,
                text,
                chunk_lines=self.config.chunk_lines,
                overlap_lines=self.config.chunk_overlap_lines,
            )
            self.store.replace_file(rel, info.st_mtime, info.st_size, chunks)
            stats.files_indexed += 1
            stats.chunks_written += len(chunks)

        for stale in self.store.indexed_paths() - present:
            self.store.forget_file(stale)
            stats.files_removed += 1

        if stats.files_indexed or stats.files_removed:
            self._bm25 = None  # invalidate; rebuilt on next search

        stats.chunks_embedded = self.embed_pending()
        return stats

    def embed_pending(self) -> int:
        """Embed any chunks that do not have a vector yet.

        Separate from ``refresh`` so indexing still works when the embedding
        server is down -- retrieval simply falls back to BM25 alone.
        """
        if self.embedder is None or not self.config.hybrid_retrieval:
            return 0

        pending = self.store.chunks_missing_embeddings()
        if not pending:
            return 0

        try:
            vectors = self.embedder.embed([c.content for c in pending])  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - retrieval must degrade, not crash
            log.warning("embedding pass failed, continuing with BM25 only: %s", exc)
            return 0

        self.store.set_embeddings(dict(zip((c.id for c in pending), vectors)))
        return len(vectors)

    # ---- retrieval ------------------------------------------------------------

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is None:
            chunks = self.store.all_chunks()
            # Prepend the path so a query naming a file ranks its chunks up.
            self._bm25 = BM25Index.build(
                {c.id: f"{c.path}\n{c.content}" for c in chunks}
            )
        return self._bm25

    def _dense_search(self, query: str, limit: int) -> list[str]:
        """Rank chunk ids by cosine similarity to the query embedding."""
        if self.embedder is None or not self.config.hybrid_retrieval:
            return []

        chunks = self.store.embedded_chunks()
        if not chunks:
            return []

        try:
            query_vector = self.embedder.embed([query])[0]  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - degrade to BM25
            log.warning("dense search unavailable: %s", exc)
            return []

        matrix = np.asarray([c.embedding for c in chunks], dtype=np.float32)
        vector = np.asarray(query_vector, dtype=np.float32)

        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
        # Guard against zero-norm rows rather than emitting nan into the ranking.
        norms[norms == 0.0] = 1e-9
        sims = (matrix @ vector) / norms

        order = np.argsort(-sims)[:limit]
        return [chunks[i].id for i in order]

    def _rerank(self, query: str, candidates: list[Chunk], top_k: int) -> list[Retrieved]:
        try:
            scored = self.reranker.rerank(  # type: ignore[attr-defined]
                query, [c.content for c in candidates], top_k=top_k
            )
        except Exception as exc:  # noqa: BLE001 - degrade to fused order
            log.warning("rerank unavailable, keeping fused order: %s", exc)
            return []

        results: list[Retrieved] = []
        for item in scored:
            if 0 <= item.index < len(candidates):
                results.append(
                    Retrieved(chunk=candidates[item.index], score=item.score,
                              sources=("rerank",))
                )
        return results

    def search(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        """Retrieve the most relevant chunks, using whichever stages are available."""
        k = top_k or self.config.retrieve_top_k
        if k <= 0:
            return []

        use_rerank = self.reranker is not None and self.config.use_reranker
        candidate_k = max(k, self.config.rerank_candidates) if use_rerank else k * 2

        bm25_ranking = [doc_id for doc_id, _ in self._ensure_bm25().search(query, candidate_k)]
        dense_ranking = self._dense_search(query, candidate_k)

        if not bm25_ranking and not dense_ranking:
            return []

        if dense_ranking:
            weight = self.config.dense_weight
            fused = reciprocal_rank_fusion(
                {"bm25": bm25_ranking, "dense": dense_ranking},
                weights={"bm25": 1.0 - weight, "dense": weight},
            )
        else:
            fused = reciprocal_rank_fusion({"bm25": bm25_ranking})

        ordered_ids = [hit.doc_id for hit in fused[:candidate_k]]
        by_id = {c.id: c for c in self.store.get_chunks(ordered_ids)}
        candidates = [by_id[doc_id] for doc_id in ordered_ids if doc_id in by_id]
        if not candidates:
            return []

        if use_rerank:
            reranked = self._rerank(query, candidates, k)
            if reranked:
                return reranked

        sources = ("bm25", "dense") if dense_ranking else ("bm25",)
        score_by_id = {hit.doc_id: hit.score for hit in fused}
        return [
            Retrieved(chunk=chunk, score=score_by_id.get(chunk.id, 0.0), sources=sources)
            for chunk in candidates[:k]
        ]


def format_context(results: list[Retrieved], *, max_chars: int = 12000) -> str:
    """Render retrieved chunks as a citable context block for the prompt."""
    if not results:
        return ""
    blocks: list[str] = []
    used = 0
    for item in results:
        block = f"--- {item.chunk.cite()} ---\n{item.chunk.content}"
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n".join(blocks)
