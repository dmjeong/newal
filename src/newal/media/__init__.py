"""Multimodal attachment handling: images and video into chat content parts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import MediaConfig
from .budget import estimate_visual_tokens
from .images import PreparedImage, is_image_path, prepare_image
from .video import PreparedVideo, VideoUnavailable, is_video_path, prepare_video

log = logging.getLogger(__name__)


@dataclass
class Attachment:
    """A user-supplied file resolved into model-ready content parts."""

    source: str
    kind: str  # "image" | "video"
    parts: list[dict[str, Any]]
    tokens: int
    summary: str


class AttachmentError(ValueError):
    pass


def prepare_attachments(
    paths: list[str | Path], config: MediaConfig
) -> list[Attachment]:
    """Prepare every path, sharing one visual token budget across all of them.

    The budget is split evenly up front rather than first-come-first-served, so
    attaching a video alongside an image does not starve the image.
    """
    if not paths:
        return []

    per_item_budget = max(1, config.visual_token_budget // len(paths))
    attachments: list[Attachment] = []
    spent = 0

    for path in paths:
        # Roll any budget the previous items did not use into this one.
        remaining_items = len(paths) - len(attachments)
        budget = max(
            per_item_budget,
            (config.visual_token_budget - spent) // max(1, remaining_items),
        )
        attachment = _prepare_one(path, config, budget)
        spent += attachment.tokens
        attachments.append(attachment)

    total = sum(a.tokens for a in attachments)
    if total > config.visual_token_budget:
        log.warning(
            "attachments cost ~%s visual tokens, over the %s budget",
            total,
            config.visual_token_budget,
        )
    return attachments


def _prepare_one(
    path: str | Path, config: MediaConfig, budget: int
) -> Attachment:
    resolved = Path(path).expanduser()

    if is_image_path(resolved):
        image: PreparedImage = prepare_image(
            resolved, max_pixels=config.image_max_pixels, token_budget=budget
        )
        return Attachment(
            source=str(resolved),
            kind="image",
            parts=[image.as_content_part()],
            tokens=image.tokens,
            summary=f"{resolved.name} ({image.width}x{image.height}, ~{image.tokens} tok)",
        )

    if is_video_path(resolved):
        video: PreparedVideo = prepare_video(resolved, config.video, token_budget=budget)
        if not video.frames:
            raise AttachmentError(f"no frames could be sampled from {resolved}")
        parts = video.as_content_parts(include_timestamps=config.video.include_timestamps)
        return Attachment(
            source=str(resolved),
            kind="video",
            parts=parts,
            tokens=video.tokens,
            summary=(
                f"{resolved.name} ({video.info.duration_s:.1f}s, "
                f"{len(video.frames)} frames via {video.strategy}, ~{video.tokens} tok)"
            ),
        )

    raise AttachmentError(
        f"unsupported attachment type: {resolved.name}. "
        "Supported: images (png/jpg/webp/...) and video (mp4/mov/mkv/...)"
    )


__all__ = [
    "Attachment",
    "AttachmentError",
    "PreparedImage",
    "PreparedVideo",
    "VideoUnavailable",
    "estimate_visual_tokens",
    "is_image_path",
    "is_video_path",
    "prepare_attachments",
]
