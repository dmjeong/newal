"""Image loading and encoding into chat-message content parts."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .budget import estimate_visual_tokens, fit_to_pixel_budget

SUPPORTED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class PreparedImage:
    """An image resized and encoded, with its measured token cost."""

    data_uri: str
    width: int
    height: int
    tokens: int
    source: str

    def as_content_part(self) -> dict[str, Any]:
        return {"type": "image_url", "image_url": {"url": self.data_uri}}


def is_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_SUFFIXES


def encode_pil_image(image: Image.Image, *, quality: int = 88) -> str:
    """Encode a PIL image as a base64 JPEG data URI.

    JPEG rather than PNG: at these resolutions the payload is several times
    smaller, and the vision encoder is not sensitive to the difference.
    """
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def prepare_image(
    path: str | Path,
    *,
    max_pixels: int,
    token_budget: int | None = None,
) -> PreparedImage:
    """Load, orient, downscale, and encode an image for the model."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"image not found: {source}")

    with Image.open(source) as raw:
        # Phone screenshots and camera photos carry EXIF rotation; without this
        # the model reads them sideways.
        image = ImageOps.exif_transpose(raw)
        image = image.convert("RGB")

        effective_max = max_pixels
        if token_budget is not None:
            width, height = fit_to_pixel_budget(image.width, image.height, max_pixels)
            # Shrink further if the budget cannot cover the image at max_pixels.
            while estimate_visual_tokens(width, height) > token_budget and width > 28:
                effective_max = int(width * height * 0.5) or 1
                width, height = fit_to_pixel_budget(width, height, effective_max)
            effective_max = width * height

        new_size = fit_to_pixel_budget(image.width, image.height, effective_max)
        if new_size != (image.width, image.height):
            image = image.resize(new_size, Image.LANCZOS)

        return PreparedImage(
            data_uri=encode_pil_image(image),
            width=image.width,
            height=image.height,
            tokens=estimate_visual_tokens(image.width, image.height),
            source=str(source),
        )
