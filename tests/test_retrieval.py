"""Hybrid retrieval: BM25, dense fusion, reranking, and their fallbacks.

Retrieval must degrade rather than fail: an embedding or rerank server that is
down should cost result quality, never the session.
"""

from __future__ import annotations

import pytest

from newal.config import MemoryConfig
from newal.memory import build_index
from newal.models.retrieval import Scored, cosine_similarity


class FakeEmbedder:
    """Deterministic bag-of-characters embedding, good enough to rank by overlap."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def embed(self, texts, batch_size: int = 32):
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding server is down")
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([float(lowered.count(chr(c))) for c in range(97, 123)])
        return vectors


class FakeReranker:
    """Ranks by substring overlap; optionally fails to exercise the fallback."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    def rerank(self, query, documents, *, top_k):
        self.calls += 1
        if self.fail:
            raise RuntimeError("rerank server is down")
        terms = set(query.lower().split())
        scored = [
            Scored(index=i, score=float(sum(t in doc.lower() for t in terms)))
            for i, doc in enumerate(documents)
        ]
        scored.sort(key=lambda s: -s.score)
        return scored[:top_k]


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "sampler.py").write_text(
        "def select_keyframes(distances):\n"
        "    'pick video frames at scene changes'\n"
        "    return distances\n",
        encoding="utf-8",
    )
    (tmp_path / "config.py").write_text(
        "def load_config(path):\n    'merge yaml configuration layers'\n    return {}\n",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text("# coffee\nunrelated prose about beans\n", encoding="utf-8")
    return tmp_path


def _index(repo, **overrides):
    config = MemoryConfig(
        db_path=str(repo / ".newal" / "m.db"),
        index_globs=["**/*.py", "**/*.md"],
        **overrides,
    )
    return build_index(repo, config)


def test_bm25_only_retrieval_works_without_any_model(repo):
    index = _index(repo, hybrid_retrieval=False, use_reranker=False)
    index.refresh()

    results = index.search("keyframes scene changes", top_k=1)
    assert results
    assert results[0].chunk.path == "sampler.py"


def test_embeddings_are_stored_and_reused(repo):
    embedder = FakeEmbedder()
    index = _index(repo, use_reranker=False)
    index.attach_models(embedder=embedder, reranker=None)

    stats = index.refresh()
    assert stats.chunks_embedded == index.store.chunk_count()
    assert index.store.chunks_without_embeddings_count() == 0

    # A second refresh must not re-embed unchanged files.
    before = embedder.calls
    index.refresh()
    assert embedder.calls == before


def test_hybrid_retrieval_reports_both_sources(repo):
    index = _index(repo, use_reranker=False)
    index.attach_models(embedder=FakeEmbedder(), reranker=None)
    index.refresh()

    results = index.search("configuration layers", top_k=2)
    assert results
    assert "dense" in results[0].sources


def test_reranker_reorders_and_is_marked_as_the_source(repo):
    reranker = FakeReranker()
    index = _index(repo)
    index.attach_models(embedder=FakeEmbedder(), reranker=reranker)
    index.refresh()

    results = index.search("scene changes video frames", top_k=2)
    assert reranker.calls == 1
    assert results[0].sources == ("rerank",)
    assert results[0].chunk.path == "sampler.py"


def test_failed_embedding_degrades_to_bm25(repo):
    index = _index(repo, use_reranker=False)
    index.attach_models(embedder=FakeEmbedder(fail=True), reranker=None)

    stats = index.refresh()
    assert stats.chunks_embedded == 0

    results = index.search("keyframes", top_k=1)
    assert results  # still answers
    assert results[0].sources == ("bm25",)


def test_failed_rerank_falls_back_to_the_fused_order(repo):
    index = _index(repo)
    index.attach_models(embedder=FakeEmbedder(), reranker=FakeReranker(fail=True))
    index.refresh()

    results = index.search("configuration layers", top_k=2)
    assert results
    assert "rerank" not in results[0].sources


def test_retrieval_respects_top_k(repo):
    index = _index(repo, use_reranker=False)
    index.attach_models(embedder=FakeEmbedder(), reranker=None)
    index.refresh()
    assert len(index.search("def", top_k=2)) <= 2


def test_zero_top_k_returns_nothing(repo):
    index = _index(repo)
    index.refresh()
    assert index.search("anything", top_k=0) == []


def test_deleted_files_leave_the_index(repo):
    index = _index(repo, use_reranker=False)
    index.refresh()
    assert index.search("coffee beans", top_k=1)

    (repo / "notes.md").unlink()
    stats = index.refresh()
    assert stats.files_removed == 1
    assert not index.search("coffee beans", top_k=1)


def test_schema_bump_rebuilds_the_index_but_keeps_notes(repo):
    index = _index(repo)
    index.refresh()
    index.store.add_note("build", "run pytest -q")
    db_path = index.store.db_path
    index.store.close()

    import sqlite3

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA user_version = 1")  # pretend an older schema wrote this
    conn.commit()
    conn.close()

    reopened = _index(repo)
    assert reopened.store.chunk_count() == 0          # derived data dropped
    assert [n.content for n in reopened.store.recent_notes()] == ["run pytest -q"]


# ---- cosine helper -----------------------------------------------------------


def test_cosine_similarity_basics():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_handles_degenerate_input():
    assert cosine_similarity([], [1.0]) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0
