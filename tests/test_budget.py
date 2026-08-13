"""Visual token accounting."""

from __future__ import annotations

import pytest

from newal.media.budget import (
    PATCH_SIZE,
    estimate_visual_tokens,
    fit_to_pixel_budget,
    plan_video_frames,
)


def test_token_estimate_scales_with_area():
    small = estimate_visual_tokens(280, 280)
    large = estimate_visual_tokens(560, 560)
    assert large == pytest.approx(small * 4, rel=0.1)


def test_token_estimate_rejects_degenerate_sizes():
    assert estimate_visual_tokens(0, 100) == 0
    assert estimate_visual_tokens(-5, 100) == 0


def test_fit_preserves_aspect_ratio_within_rounding():
    width, height = fit_to_pixel_budget(1920, 1080, 409_600)
    assert width * height <= 409_600
    assert width / height == pytest.approx(1920 / 1080, rel=0.05)


def test_fit_snaps_to_patch_multiples():
    width, height = fit_to_pixel_budget(1000, 700, 200_000)
    assert width % PATCH_SIZE == 0
    assert height % PATCH_SIZE == 0


def test_fit_never_upscales():
    assert fit_to_pixel_budget(100, 100, 10_000_000) == (
        100 // PATCH_SIZE * PATCH_SIZE,
        100 // PATCH_SIZE * PATCH_SIZE,
    )


def test_fit_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        fit_to_pixel_budget(0, 100, 1000)


def test_plan_drops_frames_to_fit_budget():
    plan = plan_video_frames(
        1920, 1080, requested_frames=32, frame_max_pixels=409_600, token_budget=2000
    )
    assert plan.frame_count < 32
    assert plan.total_tokens <= 2000


def test_plan_keeps_all_frames_when_budget_is_ample():
    plan = plan_video_frames(
        640, 360, requested_frames=8, frame_max_pixels=409_600, token_budget=100_000
    )
    assert plan.frame_count == 8


def test_plan_shrinks_a_single_oversized_frame():
    """A tiny budget must still yield one usable frame rather than zero."""
    plan = plan_video_frames(
        3840, 2160, requested_frames=1, frame_max_pixels=3840 * 2160, token_budget=200
    )
    assert plan.frame_count == 1
    assert plan.tokens_per_frame <= 200


def test_plan_handles_zero_budget():
    plan = plan_video_frames(
        640, 360, requested_frames=8, frame_max_pixels=409_600, token_budget=0
    )
    assert plan.frame_count == 0
    assert plan.total_tokens == 0
