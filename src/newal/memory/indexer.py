"""Repository indexing and retrieval.

Incremental: files whose mtime and size are unchanged are skipped, so a
re-index of a large repo after one edit costs milliseconds.
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

from ..config import MemoryConfig
from .bm25 import BM25Index
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


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


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
    """Owns the store plus a lazily rebuilt BM25 index over its chunks."""

    def __init__(self, root: Path, config: MemoryConfig, store: MemoryStore) -> None:
        self.root = root
        self.config = config
        self.store = store
        self._bm25: BM25Index | None = None

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
        return stats

    def _ensure_bm25(self) -> BM25Index:
        if self._bm25 is None:
            chunks = self.store.all_chunks()
            # Prepend the path so a query naming a file ranks its chunks up.
            self._bm25 = BM25Index.build(
                {c.id: f"{c.path}\n{c.content}" for c in chunks}
            )
        return self._bm25

    def search(self, query: str, top_k: int | None = None) -> list[Retrieved]:
        k = top_k or self.config.retrieve_top_k
        hits = self._ensure_bm25().search(query, top_k=k * 3 if self._dense() else k)
        if not hits:
            return []

        chunks = self.store.get_chunks([doc_id for doc_id, _ in hits])
        scores = dict(hits)
        results = [Retrieved(chunk=c, score=scores.get(c.id, 0.0)) for c in chunks]

        if self._dense():
            results = self._rerank(query, results)
        return results[:k]

    def _dense(self) -> bool:
        return self.config.dense_rerank

    def _rerank(self, query: str, candidates: list[Retrieved]) -> list[Retrieved]:
        """Re-order BM25 candidates by embedding similarity, if available."""
        try:
            from sentence_transformers import SentenceTransformer, util
        except ImportError:
            log.warning("dense_rerank is on but sentence-transformers is missing; "
                        "install with: pip install 'newal[dense]'")
            return candidates

        model = _load_encoder(self.config.dense_model, SentenceTransformer)
        query_vec = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        doc_vecs = model.encode(
            [c.chunk.content for c in candidates],
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        sims = util.cos_sim(query_vec, doc_vecs)[0]
        for candidate, sim in zip(candidates, sims):
            candidate.score = float(sim)
        candidates.sort(key=lambda r: -r.score)
        return candidates


_ENCODER_CACHE: dict[str, object] = {}


def _load_encoder(name: str, factory: type) -> object:
    """Cache the encoder; loading it per query dominates retrieval cost."""
    if name not in _ENCODER_CACHE:
        _ENCODER_CACHE[name] = factory(name)
    return _ENCODER_CACHE[name]


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
