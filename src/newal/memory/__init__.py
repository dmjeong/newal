"""Persistent project memory: repo retrieval plus cross-session notes."""

from __future__ import annotations

from pathlib import Path

from ..config import MemoryConfig
from .bm25 import BM25Index, tokenize
from .indexer import IndexStats, RepoIndex, Retrieved, chunk_text, format_context
from .store import Chunk, MemoryStore, Note


def build_index(root: Path, config: MemoryConfig) -> RepoIndex:
    """Open (creating if needed) the memory DB for ``root``."""
    db_path = Path(config.db_path)
    if not db_path.is_absolute():
        db_path = root / db_path
    return RepoIndex(root, config, MemoryStore(db_path))


__all__ = [
    "BM25Index",
    "Chunk",
    "IndexStats",
    "MemoryStore",
    "Note",
    "RepoIndex",
    "Retrieved",
    "build_index",
    "chunk_text",
    "format_context",
    "tokenize",
]
