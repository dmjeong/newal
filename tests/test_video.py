"""Keyframe selection policy and frame scoring.

The selection policy is a pure function over change scores, so it is tested
without decoding an actual video file.
"""

from __future__ import annotations

import numpy as np
import pytest

from newal.media.video import (
    HIST_BINS,
    frame_signature,
    select_keyframes,
    signature_distance,
)


def test_all_frames_kept_when_under_the_limit():
    assert select_keyframes([0.0, 0.1, 0.2], max_frames=10, threshold=0.3) == [0, 1, 2]


def test_first_frame_is_always_selected():
    distances = [0.0] + [0.01] * 50
    assert select_keyframes(distances, max_frames=5, threshold=0.3)[0] == 0


def test_scene_cuts_are_preferred_over_static_frames():
    # Two sharp changes at 10 and 30; everything else is static.
    distances = [0.0] * 40
    distances[10] = 0.9
    distances[30] = 0.8

    selected = select_keyframes(distances, max_frames=4, threshold=0.3)
    assert 10 in selected
    assert 30 in selected


def test_selection_respects_the_frame_cap():
    distances = [0.9] * 100
    assert len(select_keyframes(distances, max_frames=8, threshold=0.3)) == 8


def test_static_video_falls_back_to_uniform_spread():
    """With no cuts, frames should be spread out rather than bunched at the start."""
    distances = [0.0] * 100
    selected = select_keyframes(distances, max_frames=5, threshold=0.3)

    assert len(selected) == 5
    assert selected == sorted(selected)
    assert max(selected) > 50  # reaches the back half of the clip


def test_selection_returns_unique_sorted_indices():
    distances = [0.5 if i % 3 == 0 else 0.0 for i in range(60)]
    selected = select_keyframes(distances, max_frames=12, threshold=0.3)
    assert selected == sorted(set(selected))


def test_empty_and_zero_budget_cases():
    assert select_keyframes([], max_frames=5, threshold=0.3) == []
    assert select_keyframes([0.0, 0.5], max_frames=0, threshold=0.3) == []


def test_identical_frames_have_zero_distance():
    frame = np.full((32, 32, 3), 128, dtype=np.uint8)
    assert signature_distance(frame_signature(frame), frame_signature(frame)) == pytest.approx(
        0.0, abs=1e-9
    )


def test_opposite_frames_have_large_distance():
    black = np.zeros((32, 32, 3), dtype=np.uint8)
    white = np.full((32, 32, 3), 255, dtype=np.uint8)
    assert signature_distance(frame_signature(black), frame_signature(white)) > 0.9


def test_distance_is_bounded_and_symmetric():
    rng = np.random.default_rng(0)
    a = frame_signature(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    b = frame_signature(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))

    forward = signature_distance(a, b)
    assert 0.0 <= forward <= 1.0
    assert forward == pytest.approx(signature_distance(b, a))


def test_signature_is_normalised():
    rng = np.random.default_rng(1)
    signature = frame_signature(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
    assert signature.sum() == pytest.approx(1.0)
    assert np.all(signature >= 0)


def test_signature_smoothing_keeps_channels_separate():
    """Red-only content must not leak histogram mass into the green channel."""
    red = np.zeros((16, 16, 3), dtype=np.uint8)
    red[:, :, 0] = 255

    signature = frame_signature(red)
    green_band = signature[HIST_BINS : 2 * HIST_BINS]
    # Green is uniformly zero, so all its mass sits in the lowest bins.
    assert green_band[-1] == pytest.approx(0.0, abs=1e-12)


def test_small_lighting_change_scores_below_a_cut():
    base = np.full((32, 32, 3), 100, dtype=np.uint8)
    brighter = np.full((32, 32, 3), 115, dtype=np.uint8)
    cut = np.zeros((32, 32, 3), dtype=np.uint8)
    cut[:, :, 0] = 255

    lighting = signature_distance(frame_signature(base), frame_signature(brighter))
    scene_cut = signature_distance(frame_signature(base), frame_signature(cut))
    assert lighting < scene_cut
