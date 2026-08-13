"""Rank fusion for hybrid retrieval."""

from __future__ import annotations

from newal.memory.fusion import RRF_K, reciprocal_rank_fusion


def test_document_ranked_well_by_both_sources_wins():
    fused = reciprocal_rank_fusion(
        {"bm25": ["a", "b", "c"], "dense": ["a", "c", "b"]}
    )
    assert fused[0].doc_id == "a"


def test_fusion_is_scale_free():
    """Only ranks matter, so incomparable score scales cannot skew the result."""
    fused = reciprocal_rank_fusion({"bm25": ["x", "y"], "dense": ["y", "x"]})
    # Symmetric input -> equal scores, deterministic tie-break by id.
    assert {hit.doc_id for hit in fused} == {"x", "y"}
    assert fused[0].score == fused[1].score
    assert fused[0].doc_id == "x"


def test_weights_shift_the_ordering():
    lists = {"bm25": ["a", "b"], "dense": ["b", "a"]}

    bm25_heavy = reciprocal_rank_fusion(lists, weights={"bm25": 0.9, "dense": 0.1})
    dense_heavy = reciprocal_rank_fusion(lists, weights={"bm25": 0.1, "dense": 0.9})

    assert bm25_heavy[0].doc_id == "a"
    assert dense_heavy[0].doc_id == "b"


def test_zero_weight_removes_a_source():
    fused = reciprocal_rank_fusion(
        {"bm25": ["a"], "dense": ["z"]}, weights={"dense": 0.0}
    )
    assert [hit.doc_id for hit in fused] == ["a"]


def test_documents_missing_from_one_list_still_rank():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["c"]})
    assert {hit.doc_id for hit in fused} == {"a", "b", "c"}


def test_lists_of_different_lengths_are_handled():
    fused = reciprocal_rank_fusion(
        {"bm25": [f"d{i}" for i in range(50)], "dense": ["d49"]}
    )
    # d49 is last in BM25 but first in dense, so fusion should lift it.
    ranked = [hit.doc_id for hit in fused]
    assert ranked.index("d49") < 49


def test_ranks_are_reported_for_debugging():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b"], "dense": ["b", "a"]})
    by_id = {hit.doc_id: hit for hit in fused}
    assert by_id["a"].ranks == {"bm25": 1, "dense": 2}


def test_empty_input_is_safe():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"bm25": []}) == []


def test_single_source_preserves_its_order():
    fused = reciprocal_rank_fusion({"bm25": ["a", "b", "c"]})
    assert [hit.doc_id for hit in fused] == ["a", "b", "c"]


def test_k_damps_the_top_rank_advantage():
    tight = reciprocal_rank_fusion({"s": ["a", "b"]}, k=1000)
    loose = reciprocal_rank_fusion({"s": ["a", "b"]}, k=1)
    tight_gap = tight[0].score - tight[1].score
    loose_gap = loose[0].score - loose[1].score
    assert tight_gap < loose_gap
    assert RRF_K == 60
