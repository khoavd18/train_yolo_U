"""Strict parsers for the annotation formats supported by this repository.

Geometry is selected by an explicit :class:`AnnotationFormat`.  In particular,
the parser never guesses whether eight coordinates describe an OBB or a
segmentation polygon.  Dataset/source configuration owns that decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


class AnnotationFormat(str, Enum):
    """Annotation encodings intentionally supported by the data workflows."""

    COCO_BBOX = "coco_bbox"
    YOLO_BBOX = "yolo_bbox"
    YOLO_OBB = "yolo_obb"
    YOLO_SEGMENTATION = "yolo_segmentation"

    @classmethod
    def parse(cls, value: str | "AnnotationFormat") -> "AnnotationFormat":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().casefold().replace("-", "_")
        aliases = {
            "bbox": cls.YOLO_BBOX,
            "detect": cls.YOLO_BBOX,
            "obb": cls.YOLO_OBB,
            "segment": cls.YOLO_SEGMENTATION,
            "segmentation": cls.YOLO_SEGMENTATION,
            "polygon": cls.YOLO_SEGMENTATION,
        }
        try:
            return aliases.get(normalized, cls(normalized))
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported annotation format {value!r}; expected one of: {supported}"
            ) from exc


@dataclass(frozen=True)
class NormalizedBox:
    """A clipped axis-aligned box in normalized image coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    was_clipped: bool = False

    @property
    def x_center(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def y_center(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min


@dataclass(frozen=True)
class ParsedYoloAnnotation:
    """One validated YOLO row with its declared source geometry retained."""

    class_id: int
    class_name: str
    annotation_format: AnnotationFormat
    box: NormalizedBox
    source_values: tuple[float, ...]


def normalize_yolo_names(value: Any) -> tuple[str, ...]:
    """Normalize YAML ``names`` while rejecting sparse or duplicate class IDs."""

    if isinstance(value, list):
        names = tuple(str(item) for item in value)
    elif isinstance(value, Mapping):
        try:
            by_id = {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("YOLO names keys must be integer-like") from exc
        if sorted(by_id) != list(range(len(by_id))):
            raise ValueError("YOLO names IDs must be contiguous from zero")
        names = tuple(by_id[index] for index in range(len(by_id)))
    else:
        raise ValueError("YOLO data.yaml 'names' must be a list or ID mapping")
    if not names or any(not name.strip() for name in names):
        raise ValueError("YOLO class names must be non-empty")
    return names


def load_yolo_class_names(data_yaml: str | Path) -> tuple[str, ...]:
    """Read the authoritative class order from an export's ``data.yaml``."""

    path = Path(data_yaml)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = normalize_yolo_names(payload.get("names"))
    declared_count = payload.get("nc")
    if declared_count is not None and int(declared_count) != len(names):
        raise ValueError(
            f"{path} declares nc={declared_count}, but names contains {len(names)} classes"
        )
    return names


def _finite(values: Sequence[float]) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("annotation contains a non-finite coordinate")
    return converted


def _clip_edges(
    x_min: float, y_min: float, x_max: float, y_max: float
) -> NormalizedBox:
    raw = (x_min, y_min, x_max, y_max)
    clipped = (
        max(0.0, x_min),
        max(0.0, y_min),
        min(1.0, x_max),
        min(1.0, y_max),
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("annotation has no non-degenerate intersection with the image")
    return NormalizedBox(
        *clipped,
        was_clipped=any(
            not math.isclose(before, after, abs_tol=1e-12)
            for before, after in zip(raw, clipped)
        ),
    )


def normalized_box_from_values(
    values: Sequence[float], annotation_format: str | AnnotationFormat
) -> NormalizedBox:
    """Convert explicitly declared YOLO geometry to a clipped AABB."""

    fmt = AnnotationFormat.parse(annotation_format)
    coordinates = _finite(values)
    if fmt is AnnotationFormat.YOLO_BBOX:
        if len(coordinates) != 4:
            raise ValueError("YOLO bbox requires exactly x_center y_center width height")
        x_center, y_center, width, height = coordinates
        if width <= 0.0 or height <= 0.0:
            raise ValueError("YOLO bbox width and height must be positive")
        return _clip_edges(
            x_center - width / 2.0,
            y_center - height / 2.0,
            x_center + width / 2.0,
            y_center + height / 2.0,
        )
    if fmt is AnnotationFormat.YOLO_OBB:
        if len(coordinates) != 8:
            raise ValueError("YOLO OBB requires exactly four normalized x/y points")
    elif fmt is AnnotationFormat.YOLO_SEGMENTATION:
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise ValueError(
                "YOLO segmentation requires at least three normalized x/y points"
            )
    else:
        raise ValueError(f"{fmt.value} is not a YOLO row format")
    xs = coordinates[0::2]
    ys = coordinates[1::2]
    return _clip_edges(min(xs), min(ys), max(xs), max(ys))


def parse_yolo_annotation(
    raw_line: str,
    class_names: Sequence[str],
    annotation_format: str | AnnotationFormat,
) -> ParsedYoloAnnotation:
    """Parse one non-empty YOLO row using a source-declared format."""

    fields = raw_line.split()
    if not fields:
        raise ValueError("annotation row is empty")
    try:
        class_value = float(fields[0])
        class_id = int(class_value)
    except ValueError as exc:
        raise ValueError("annotation class ID is not numeric") from exc
    if class_value != class_id:
        raise ValueError("annotation class ID must be an integer")
    if not 0 <= class_id < len(class_names):
        raise ValueError(
            f"annotation class ID {class_id} is outside data.yaml names (0-{len(class_names)-1})"
        )
    try:
        values = tuple(float(value) for value in fields[1:])
    except ValueError as exc:
        raise ValueError("annotation coordinates must be numeric") from exc
    fmt = AnnotationFormat.parse(annotation_format)
    return ParsedYoloAnnotation(
        class_id=class_id,
        class_name=str(class_names[class_id]),
        annotation_format=fmt,
        box=normalized_box_from_values(values, fmt),
        source_values=values,
    )


def validate_coco_document(payload: Any, source: str | Path = "<memory>") -> dict[str, Any]:
    """Validate the minimum COCO structure without inferring annotation geometry."""

    if not isinstance(payload, dict):
        raise ValueError(f"{source} is not a COCO object")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"{source} COCO field {key!r} must be a list")
    category_ids: set[int] = set()
    for category in payload["categories"]:
        try:
            category_id = int(category["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source} contains a category without an integer ID") from exc
        if category_id in category_ids:
            raise ValueError(f"{source} contains duplicate COCO category ID {category_id}")
        category_ids.add(category_id)
    return payload
