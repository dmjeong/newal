"""BM25 retrieval and chunking."""

from __future__ import annotations

from newal.memory.bm25 import BM25Index, tokenize
from newal.memory.indexer import chunk_text


def test_tokenize_splits_snake_case():
    tokens = tokenize("prepare_video")
    assert "prepare_video" in tokens
    assert "prepare" in tokens
    assert "video" in tokens


def test_tokenize_splits_camel_case():
    tokens = tokenize("prepareVideoFrames")
    assert "prepare" in tokens
    assert "video" in tokens
    assert "frames" in tokens


def test_search_ranks_the_relevant_document_first():
    index = BM25Index.build(
        {
            "a": "the video sampler picks keyframes from a recording",
            "b": "the config loader merges yaml files",
            "c": "unrelated notes about coffee",
        }
    )
    results = index.search("keyframe video sampler")
    assert results
    assert results[0][0] == "a"


def test_search_returns_nothing_for_unknown_terms():
    index = BM25Index.build({"a": "alpha beta"})
    assert index.search("zzzz nonexistent") == []


def test_search_respects_top_k():
    index = BM25Index.build({str(i): "shared term here" for i in range(10)})
    assert len(index.search("shared", top_k=3)) == 3


def test_empty_index_is_safe_to_query():
    assert BM25Index.build({}).search("anything") == []


def test_chunking_overlaps_and_covers_the_file():
    text = "\n".join(f"line {i}" for i in range(200))
    chunks = chunk_text("f.py", text, chunk_lines=80, overlap_lines=15)

    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 200
    # Consecutive chunks must overlap, so a symbol on a boundary stays findable.
    assert chunks[1].start_line <= chunks[0].end_line


def test_chunking_skips_whitespace_only_content():
    assert chunk_text("f.py", "\n\n\n   \n", chunk_lines=10, overlap_lines=2) == []


def test_chunk_ids_are_stable_across_runs():
    text = "\n".join(f"line {i}" for i in range(50))
    first = chunk_text("f.py", text, chunk_lines=20, overlap_lines=5)
    second = chunk_text("f.py", text, chunk_lines=20, overlap_lines=5)
    assert [c.id for c in first] == [c.id for c in second]
