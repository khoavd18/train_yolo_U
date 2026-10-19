"""Shared crop geometry and Windows-safe deterministic naming."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Any


WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}


def sanitize_class_name(name: str, fallback: str = "unnamed") -> str:
    """Produce a Windows-safe directory name; provenance retains the original."""

    value = unicodedata.normalize("NFKC", str(name)).strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" .")
    if not value:
        value = fallback
    if value.casefold() in WINDOWS_RESERVED_NAMES:
        value = f"category_{value}"
    return value[:150]


def stable_identity_hash(*parts: Any, length: int = 10) -> str:
    """Return a short stable hash while keeping full identities in provenance."""

    if length < 8 or length > 12:
        raise ValueError("Stable identity hashes must contain 8-12 characters")
    normalized = "\x1f".join(
        unicodedata.normalize("NFC", str(part)) for part in parts
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:length]


def make_crop_filename(
    source_file_name: str,
    image_id: int,
    annotation_id: int,
    class_index: int = 0,
    *,
    source_zip: str = "",
    original_split: str = "",
    original_image_id: Any | None = None,
    compact: bool = False,
) -> str:
    """Generate a short deterministic name suitable for Windows paths."""

    class_index = int(class_index)
    image_id = int(image_id)
    annotation_id = int(annotation_id)
    source_hash = stable_identity_hash(
        source_zip,
        original_split,
        source_file_name,
        image_id if original_image_id is None else original_image_id,
        length=8 if compact else 10,
    )
    if compact:
        return f"c{class_index}_i{image_id}_a{annotation_id}_{source_hash}.jpg"
    return (
        f"c{class_index:03d}_i{image_id:06d}_a{annotation_id:06d}_"
        f"{source_hash}.jpg"
    )


def bbox_to_crop_box(
    bbox: Any, image_width: int, image_height: int, padding: float = 0.05
) -> tuple[int, int, int, int] | None:
    """Convert explicit COCO ``[x, y, width, height]`` to a clipped Pillow box."""

    if image_width <= 0 or image_height <= 0 or padding < 0:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    horizontal_padding = width * padding
    vertical_padding = height * padding
    left = max(0, math.floor(x - horizontal_padding))
    top = max(0, math.floor(y - vertical_padding))
    right = min(image_width, math.ceil(x + width + horizontal_padding))
    bottom = min(image_height, math.ceil(y + height + vertical_padding))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
