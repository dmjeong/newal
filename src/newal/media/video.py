"""Video ingestion: sample the frames that carry information, drop the rest.

A 60-second screen recording at 30fps is 1800 frames. The model can afford
maybe 32. Uniform sampling wastes most of them on a static screen; this module
scores candidate frames by visual change and keeps the ones where something
actually happened, which is what makes video useful for bug reports and UI
walkthroughs rather than just a novelty input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from ..config import VideoConfig
from .budget import estimate_visual_tokens, plan_video_frames
from .images import encode_pil_image

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}

# Upper bound on frames we decode for scoring. Beyond this the scan cost stops
# paying for itself -- scene cuts are already well localised.
MAX_CANDIDATES = 240
HIST_BINS = 32
# Histogram bins are hard edges: a uniform brightness shift of a few levels can
# move every pixel across a boundary and score as a full scene cut. Blurring
# each channel's histogram makes the score degrade smoothly with the size of
# the change, so gradual lighting drift ranks well below a real cut.
HIST_SMOOTHING_PASSES = 2
_SMOOTHING_KERNEL = np.array([0.25, 0.5, 0.25])


class VideoUnavailable(RuntimeError):
    """Raised when video support is not installed or the file cannot be read."""


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def _load_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:
        raise VideoUnavailable(
            "video support needs OpenCV: pip install opencv-python-headless"
        ) from exc
    return cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass
class PreparedFrame:
    data_uri: str
    timestamp_s: float
    frame_index: int
    tokens: int


@dataclass
class PreparedVideo:
    info: VideoInfo
    frames: list[PreparedFrame] = field(default_factory=list)
    strategy: str = "uniform"

    @property
    def tokens(self) -> int:
        return sum(f.tokens for f in self.frames)

    def as_content_parts(self, *, include_timestamps: bool = True) -> list[dict[str, Any]]:
        """Render as chat content parts.

        Frames go in as individual images with a timestamp label rather than as
        a single ``video`` part: it keeps the payload backend-agnostic and lets
        the model refer to a moment by time when it answers.
        """
        parts: list[dict[str, Any]] = []
        for frame in self.frames:
            if include_timestamps:
                parts.append({"type": "text", "text": f"[t={frame.timestamp_s:.2f}s]"})
            parts.append({"type": "image_url", "image_url": {"url": frame.data_uri}})
        return parts


def probe_video(path: str | Path) -> VideoInfo:
    cv2 = _load_cv2()
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"video not found: {source}")

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoUnavailable(f"could not open video: {source}")
    try:
        return VideoInfo(
            path=str(source),
            fps=float(capture.get(cv2.CAP_PROP_FPS)) or 0.0,
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def _smooth(histogram: np.ndarray) -> np.ndarray:
    """Blur a histogram along its bins, keeping the edges from losing mass."""
    smoothed = histogram
    for _ in range(HIST_SMOOTHING_PASSES):
        padded = np.pad(smoothed, 1, mode="edge")
        smoothed = np.convolve(padded, _SMOOTHING_KERNEL, mode="valid")
    return smoothed


def frame_signature(frame_rgb: np.ndarray) -> np.ndarray:
    """Reduce a frame to a normalised, smoothed per-channel histogram.

    Histograms ignore camera shake and compression noise while still reacting
    to cuts, scrolls, and dialogs opening -- the transitions worth a frame.
    Channels are smoothed independently so mass never bleeds from the top bin
    of one channel into the bottom bin of the next.

    The returned vector sums to 1 across all channels combined.
    """
    channels = []
    for channel in range(frame_rgb.shape[2]):
        hist, _ = np.histogram(
            frame_rgb[:, :, channel], bins=HIST_BINS, range=(0, 256)
        )
        channels.append(_smooth(hist.astype(np.float64)))
    signature = np.concatenate(channels)
    total = signature.sum()
    return signature / total if total > 0 else signature


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Bhattacharyya-style distance in [0, 1]; 0 means identical."""
    overlap = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
    return float(np.clip(1.0 - overlap, 0.0, 1.0))


def select_keyframes(
    distances: Sequence[float],
    *,
    max_frames: int,
    threshold: float,
) -> list[int]:
    """Pick candidate indices to keep, given per-candidate change scores.

    ``distances[i]`` is how much candidate ``i`` differs from candidate
    ``i - 1`` (``distances[0]`` is 0 by construction). Pure function over
    plain numbers so the selection policy can be tested without a video file.

    Returns sorted candidate indices, always including the first frame.
    """
    count = len(distances)
    if count == 0 or max_frames <= 0:
        return []
    if count <= max_frames:
        return list(range(count))

    selected = {0}
    # Scene cuts first, strongest change wins ties for the last slots.
    cuts = [i for i, d in enumerate(distances) if d >= threshold]
    cuts.sort(key=lambda i: distances[i], reverse=True)
    for index in cuts:
        if len(selected) >= max_frames:
            break
        selected.add(index)

    # Static video, or too few cuts: spread the remaining slots evenly so long
    # uneventful stretches are still represented.
    if len(selected) < max_frames:
        remaining = max_frames - len(selected)
        step = count / float(remaining + 1)
        for slot in range(1, remaining + 1):
            candidate = min(count - 1, int(round(slot * step)))
            # Walk forward to the nearest unused index.
            while candidate in selected and candidate < count - 1:
                candidate += 1
            if candidate not in selected:
                selected.add(candidate)
            if len(selected) >= max_frames:
                break

    return sorted(selected)


def _candidate_indices(frame_count: int) -> list[int]:
    if frame_count <= MAX_CANDIDATES:
        return list(range(max(frame_count, 0)))
    step = frame_count / float(MAX_CANDIDATES)
    return [min(frame_count - 1, int(i * step)) for i in range(MAX_CANDIDATES)]


def prepare_video(
    path: str | Path,
    config: VideoConfig,
    *,
    token_budget: int,
) -> PreparedVideo:
    """Decode, score, select, resize, and encode frames from a video file."""
    cv2 = _load_cv2()
    info = probe_video(path)

    if info.frame_count <= 0 or info.width <= 0 or info.height <= 0:
        raise VideoUnavailable(f"video reports no readable frames: {info.path}")

    plan = plan_video_frames(
        info.width,
        info.height,
        config.max_frames,
        frame_max_pixels=config.frame_max_pixels,
        token_budget=token_budget,
    )
    if plan.frame_count == 0:
        log.warning("visual token budget (%s) cannot fit a single frame", token_budget)
        return PreparedVideo(info=info, frames=[], strategy=config.strategy)

    capture = cv2.VideoCapture(info.path)
    if not capture.isOpened():
        raise VideoUnavailable(f"could not open video: {info.path}")

    try:
        candidates = _candidate_indices(info.frame_count)
        decoded: list[tuple[int, np.ndarray]] = []
        for index in candidates:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame_bgr = capture.read()
            if not ok or frame_bgr is None:
                continue  # seek past a damaged frame rather than aborting
            decoded.append((index, cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))

        if not decoded:
            raise VideoUnavailable(f"could not decode any frame from {info.path}")

        if config.strategy == "scene":
            signatures = [frame_signature(frame) for _, frame in decoded]
            distances = [0.0] + [
                signature_distance(signatures[i - 1], signatures[i])
                for i in range(1, len(signatures))
            ]
            keep = select_keyframes(
                distances,
                max_frames=plan.frame_count,
                threshold=config.scene_threshold,
            )
        else:
            keep = select_keyframes(
                [0.0] * len(decoded),
                max_frames=plan.frame_count,
                threshold=1.1,  # unreachable -> pure uniform spread
            )

        frames: list[PreparedFrame] = []
        for position in keep:
            frame_index, frame_rgb = decoded[position]
            image = Image.fromarray(frame_rgb)
            if (image.width, image.height) != (plan.frame_width, plan.frame_height):
                image = image.resize((plan.frame_width, plan.frame_height), Image.LANCZOS)
            frames.append(
                PreparedFrame(
                    data_uri=encode_pil_image(image),
                    timestamp_s=frame_index / info.fps if info.fps > 0 else 0.0,
                    frame_index=frame_index,
                    tokens=estimate_visual_tokens(image.width, image.height),
                )
            )

        return PreparedVideo(info=info, frames=frames, strategy=config.strategy)
    finally:
        capture.release()
