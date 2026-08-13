"""Visual token accounting.

Qwen's vision encoder turns an image into patches of 28x28 pixels, then merges
each 2x2 block of patches into one token. So a WxH image costs roughly
``ceil(W/28) * ceil(H/28) / 4`` tokens. A 1280x720 frame is ~300 tokens, and a
32-frame video clip is ~10k tokens -- enough to blow a context window if left
unchecked, which is why every attachment is priced before it is sent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

PATCH_SIZE = 28
MERGE_FACTOR = 4  # 2x2 spatial merge in the projector


def estimate_visual_tokens(width: int, height: int) -> int:
    """Estimate the token cost of a single image of the given pixel size."""
    if width <= 0 or height <= 0:
        return 0
    patches_w = math.ceil(width / PATCH_SIZE)
    patches_h = math.ceil(height / PATCH_SIZE)
    return max(1, (patches_w * patches_h) // MERGE_FACTOR)


def fit_to_pixel_budget(
    width: int, height: int, max_pixels: int
) -> tuple[int, int]:
    """Scale (width, height) down to at most ``max_pixels``, keeping aspect.

    Dimensions are rounded to a multiple of ``PATCH_SIZE`` so the encoder does
    not have to pad, and never fall below one full patch.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid dimensions: {width}x{height}")

    scale = min(1.0, math.sqrt(max_pixels / float(width * height)))
    new_w = max(PATCH_SIZE, int(width * scale) // PATCH_SIZE * PATCH_SIZE)
    new_h = max(PATCH_SIZE, int(height * scale) // PATCH_SIZE * PATCH_SIZE)
    return new_w, new_h


@dataclass
class BudgetPlan:
    """How many frames fit in the budget, and at what resolution."""

    frame_count: int
    frame_width: int
    frame_height: int
    tokens_per_frame: int

    @property
    def total_tokens(self) -> int:
        return self.frame_count * self.tokens_per_frame


def plan_video_frames(
    width: int,
    height: int,
    requested_frames: int,
    *,
    frame_max_pixels: int,
    token_budget: int,
) -> BudgetPlan:
    """Decide frame count and resolution that fit inside ``token_budget``.

    Frames are dropped before resolution is reduced: for code review, UI bugs,
    and OCR-ish tasks a legible frame is worth more than a smooth sequence.
    Only when a single frame already exceeds the budget do we shrink it.
    """
    if requested_frames <= 0 or token_budget <= 0:
        return BudgetPlan(0, 0, 0, 0)

    fw, fh = fit_to_pixel_budget(width, height, frame_max_pixels)
    per_frame = estimate_visual_tokens(fw, fh)

    # A single frame that cannot fit: shrink until it does.
    while per_frame > token_budget and (fw > PATCH_SIZE or fh > PATCH_SIZE):
        fw, fh = fit_to_pixel_budget(fw, fh, int(fw * fh * 0.5) or 1)
        per_frame = estimate_visual_tokens(fw, fh)

    affordable = token_budget // per_frame if per_frame else 0
    return BudgetPlan(
        frame_count=max(0, min(requested_frames, affordable)),
        frame_width=fw,
        frame_height=fh,
        tokens_per_frame=per_frame,
    )
