"""Audited preparation of extracted YOLO detection datasets.

The workflow in this module is deliberately source-format explicit and has no
training or inference side effects.  It reads one declared extracted-directory
source, reconciles exact-content duplicates, keeps normalized source families
inside one split, quarantines unsafe family components, and writes an immutable
derived YOLO detection dataset through an atomic staging directory.
"""

from __future__ import annotations

import colorsys
import csv
import hashlib
import io
import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps

from scripts.build_clean_axe_dataset import normalize_source_family
from src.config.registry import BrandRegistry, load_brand_registry
from src.data.annotations import (
    AnnotationFormat,
    NormalizedBox,
    normalize_yolo_names,
    parse_yolo_annotation,
)
from src.data.sources import DatasetSource, load_dataset_sources
from src.data.yolo_archive import sha256_bytes, sha256_file, write_csv, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_LAYOUT_CANDIDATES = {
    "train": ("train",),
    "val": ("valid", "val"),
    "test": ("test",),
}
FINGERPRINT_SCHEMA_VERSION = 1
PREPARATION_SCHEMA_VERSION = 1
BOUNDARY_CLIP_TOLERANCE = 1e-5
NEAR_DUPLICATE_IOU_THRESHOLD = 0.90
CLASS_ORDER_CHOICES = ("registry", "source_data_yaml")
_ROBOFLOW_HASH_RE = re.compile(r"\.rf\.[^.]+$", flags=re.IGNORECASE)
_JPG_MARKER_RE = re.compile(r"_jpg$", flags=re.IGNORECASE)


@dataclass(frozen=True)
class DetectionBox:
    """One target-class axis-aligned normalized box."""

    class_id: int
    class_name: str
    source_class_id: int
    source_class_name: str
    source_line: int
    x_min: float
    y_min: float
    x_max: float
    y_max: float

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

    def signature(self, digits: int = 12) -> tuple[Any, ...]:
        return (
            self.class_id,
            round(self.x_min, digits),
            round(self.y_min, digits),
            round(self.x_max, digits),
            round(self.y_max, digits),
        )

    def yolo_row(self) -> str:
        # Twelve decimal places keep a clipped edge within the parser's 1e-12
        # round-trip tolerance. Eight places can move x/y + width/height / 2
        # outside [0, 1] even when the in-memory edges are exactly clipped.
        return (
            f"{self.class_id} {self.x_center:.12f} {self.y_center:.12f} "
            f"{self.width:.12f} {self.height:.12f}"
        )


@dataclass
class SourceOccurrence:
    """One image/label occurrence in the extracted source tree."""

    source_split: str
    image_path: Path
    label_path: Path
    source_image_relative: str
    source_label_relative: str
    source_filename: str
    family_id: str
    variant_id: str
    image_sha256: str
    label_sha256: str | None
    image_size_bytes: int
    image_mtime_ns: int
    label_size_bytes: int | None
    label_mtime_ns: int | None
    width: int
    height: int
    boxes: list[DetectionBox]
    label_present: bool
    explicit_background: bool
    invalid_rows: list[dict[str, Any]]
    duplicate_rows: list[dict[str, Any]]
    near_duplicate_rows: list[dict[str, Any]]
    boundary_corrections: list[dict[str, Any]]

    def annotation_signature(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(sorted(box.signature() for box in self.boxes))


@dataclass
class HashAudit:
    image_sha256: str
    sources: list[SourceOccurrence]
    family_ids: set[str]
    candidate_sets: list[tuple[DetectionBox, ...]]
    status: str
    reason: str
    selected_candidate_id: str | None = None


@dataclass
class ImageUnit:
    image_sha256: str
    canonical: SourceOccurrence
    sources: list[SourceOccurrence]
    family_ids: set[str]
    boxes: list[DetectionBox]
    resolution: str
    explicit_background: bool
    component_id: str = ""
    output_split: str = ""
    output_name: str = ""
    forced_split: str | None = None
    force_reasons: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class SourceMapping:
    source_id: int
    source_name: str
    target_id: int
    target_name: str
    registry_id: int


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def normalize_source_variant(filename_or_path: str | Path) -> str:
    """Normalize one Roboflow variant while deliberately retaining ``_augN``."""

    normalized = str(filename_or_path).replace("\\", "/")
    stem = PurePosixPath(normalized).stem
    stem = _ROBOFLOW_HASH_RE.sub("", stem)
    stem = _JPG_MARKER_RE.sub("", stem)
    return stem.casefold()


def _class_multiplicity_signature(
    boxes: Sequence[DetectionBox],
) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(Counter(box.class_id for box in boxes).items()))


def _same_variant_analysis(
    occurrences: Sequence[SourceOccurrence],
) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    by_variant: dict[str, list[SourceOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_variant[occurrence.variant_id].append(occurrence)
    conflict_rows: list[dict[str, Any]] = []
    suspicious: set[str] = set()
    cross_split = 0
    differing_multiplicity = 0
    differing_class_sets = 0
    for variant_id, sources in sorted(by_variant.items()):
        splits = {source.source_split for source in sources}
        if len(splits) <= 1:
            continue
        cross_split += 1
        signatures = {
            _class_multiplicity_signature(source.boxes) for source in sources
        }
        # The set comparison intentionally ignores multiplicity.
        distinct_class_sets = {
            tuple(sorted({box.class_id for box in source.boxes}))
            for source in sources
        }
        multiplicity_differs = len(signatures) > 1
        class_set_differs = len(distinct_class_sets) > 1
        if not multiplicity_differs:
            continue
        suspicious.add(variant_id)
        differing_multiplicity += 1
        differing_class_sets += int(class_set_differs)
        conflict_rows.append(
            {
                "variant_id": variant_id,
                "family_ids": json.dumps(
                    sorted({source.family_id for source in sources}),
                    ensure_ascii=False,
                ),
                "source_splits": json.dumps(sorted(splits)),
                "source_occurrence_count": len(sources),
                "class_multiplicity_signatures": json.dumps(
                    [
                        {
                            "source_split": source.source_split,
                            "source_image": source.source_image_relative,
                            "signature": [list(item) for item in _class_multiplicity_signature(source.boxes)],
                        }
                        for source in sorted(
                            sources,
                            key=lambda item: (
                                item.source_split,
                                item.source_image_relative.casefold(),
                            ),
                        )
                    ],
                    ensure_ascii=False,
                ),
                "class_set_differs": class_set_differs,
                "reason": (
                    "cross-split occurrences of the same normalized augmentation variant "
                    "have different target-class multiplicities"
                ),
            }
        )
    return (
        {
            "normalization": (
                "case-fold filename stem; remove terminal .rf hash and _jpg; retain _augN"
            ),
            "cross_split_same_variant_groups": cross_split,
            "differing_class_multiplicity_groups": differing_multiplicity,
            "differing_class_set_groups": differing_class_sets,
            "policy": (
                "quarantine the complete broader family component only when a cross-split "
                "same-variant key disagrees in target-class multiplicity; differences between "
                "distinct _augN variants are not conflicts"
            ),
            "examples": conflict_rows[:100],
        },
        conflict_rows,
        suspicious,
    )


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalized_split(value: str) -> str:
    lowered = str(value).strip().casefold()
    if lowered == "valid":
        return "val"
    if lowered not in OUTPUT_SPLITS:
        raise ValueError(f"Unsupported detector split {value!r}")
    return lowered


def _resolve_yaml_split_path(
    value: Any,
    *,
    yaml_path: Path,
    yaml_root: Path,
) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = Path(value.strip())
    if raw.is_absolute():
        return raw.resolve()
    return (yaml_root / raw).resolve()


def _resolve_source_layout(
    source_root: Path,
    yaml_path: Path,
    payload: Mapping[str, Any],
) -> tuple[dict[str, tuple[Path, Path]], dict[str, Any]]:
    configured_root = payload.get("path")
    if configured_root:
        configured = Path(str(configured_root))
        yaml_root = (
            configured.resolve()
            if configured.is_absolute()
            else (yaml_path.parent / configured).resolve()
        )
    else:
        yaml_root = yaml_path.parent.resolve()

    layouts: dict[str, tuple[Path, Path]] = {}
    rows: list[dict[str, Any]] = []
    for output_split in OUTPUT_SPLITS:
        yaml_key = "val" if output_split == "val" else output_split
        declared = _resolve_yaml_split_path(
            payload.get(yaml_key), yaml_path=yaml_path, yaml_root=yaml_root
        )
        declared_exists = bool(declared and declared.is_dir())
        selected: Path | None = None
        status = ""
        if declared_exists and declared is not None:
            if not _path_within(declared, source_root):
                raise ValueError(
                    f"Source data.yaml split {yaml_key!r} escapes the declared source root: "
                    f"{declared}"
                )
            selected = declared
            status = "declared_path_exists"
        else:
            candidates = [
                (source_root / directory / "images").resolve()
                for directory in SOURCE_LAYOUT_CANDIDATES[output_split]
            ]
            existing = [candidate for candidate in candidates if candidate.is_dir()]
            if len(existing) != 1:
                raise FileNotFoundError(
                    f"Could not resolve exactly one image directory for {output_split}: "
                    f"declared={declared}, observed_candidates={existing}"
                )
            selected = existing[0]
            status = "declared_path_missing_used_observed_standard_layout"

        if selected.name.casefold() != "images":
            raise ValueError(
                f"Resolved source split is not an images directory: {selected}"
            )
        label_dir = selected.parent / "labels"
        if not label_dir.is_dir():
            raise FileNotFoundError(
                f"Resolved labels directory does not exist for {output_split}: {label_dir}"
            )
        if not _path_within(label_dir, source_root):
            raise ValueError(f"Labels directory escapes source root: {label_dir}")
        layouts[output_split] = (selected, label_dir.resolve())
        rows.append(
            {
                "split": output_split,
                "yaml_key": yaml_key,
                "declared_value": payload.get(yaml_key),
                "declared_resolved": str(declared) if declared else None,
                "declared_exists": declared_exists,
                "selected_images": str(selected),
                "selected_labels": str(label_dir.resolve()),
                "status": status,
            }
        )
    return layouts, {
        "yaml_path_root": str(yaml_root),
        "split_paths": rows,
        "all_declared_paths_exist": all(row["declared_exists"] for row in rows),
        "used_observed_layout_fallback": any(
            row["status"] != "declared_path_exists" for row in rows
        ),
    }


def _source_mapping(
    source_names: Sequence[str],
    registry: BrandRegistry,
    profile_name: str,
    class_order: str,
) -> tuple[tuple[str, ...], list[SourceMapping]]:
    if class_order not in CLASS_ORDER_CHOICES:
        raise ValueError(
            f"Unsupported class order {class_order!r}; choose from {CLASS_ORDER_CHOICES}"
        )
    profile_brands = registry.profile(profile_name)
    registry_names = tuple(brand.name for brand in profile_brands)
    registry_ids = {brand.name: brand.id for brand in profile_brands}
    inferred_names: list[str] = []
    for source_id, source_name in enumerate(source_names):
        target_name = registry.infer_brand(source_name, profile_name)
        if target_name is None:
            raise ValueError(
                f"Source class {source_id} {source_name!r} is not mapped by profile "
                f"{profile_name!r}"
            )
        if target_name in inferred_names:
            raise ValueError(
                f"Multiple source classes map to target class {target_name!r}"
            )
        inferred_names.append(target_name)
    if set(inferred_names) != set(registry_names):
        missing = sorted(set(registry_names) - set(inferred_names))
        raise ValueError(
            f"Source data.yaml does not cover every target profile class: {missing}"
        )
    target_names = (
        registry_names
        if class_order == "registry"
        else tuple(inferred_names)
    )
    target_ids = {name: index for index, name in enumerate(target_names)}
    rows = [
        SourceMapping(
            source_id=source_id,
            source_name=source_name,
            target_id=target_ids[target_name],
            target_name=target_name,
            registry_id=registry_ids[target_name],
        )
        for source_id, (source_name, target_name) in enumerate(
            zip(source_names, inferred_names)
        )
    ]
    return target_names, rows


def _box_from_parsed(
    parsed_box: NormalizedBox,
    *,
    source_class_id: int,
    source_class_name: str,
    source_line: int,
    mapping_by_source: Mapping[int, SourceMapping],
) -> DetectionBox:
    mapping = mapping_by_source[source_class_id]
    return DetectionBox(
        class_id=mapping.target_id,
        class_name=mapping.target_name,
        source_class_id=source_class_id,
        source_class_name=source_class_name,
        source_line=source_line,
        x_min=parsed_box.x_min,
        y_min=parsed_box.y_min,
        x_max=parsed_box.x_max,
        y_max=parsed_box.y_max,
    )


def _box_iou(first: DetectionBox, second: DetectionBox) -> float:
    intersection_width = max(
        0.0, min(first.x_max, second.x_max) - max(first.x_min, second.x_min)
    )
    intersection_height = max(
        0.0, min(first.y_max, second.y_max) - max(first.y_min, second.y_min)
    )
    intersection = intersection_width * intersection_height
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union > 0.0 else 0.0


def _parse_label_bytes(
    payload: bytes,
    *,
    source_names: Sequence[str],
    annotation_format: AnnotationFormat,
    mapping_by_source: Mapping[int, SourceMapping],
    source_label: str,
) -> tuple[
    list[DetectionBox],
    bool,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    invalid: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    corrections: list[dict[str, Any]] = []
    near_duplicates: list[dict[str, Any]] = []
    boxes: list[DetectionBox] = []
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return [], False, [{"line_number": "", "reason": f"invalid UTF-8: {exc}", "raw_line": ""}], [], [], []

    normalized_rows: dict[str, int] = {}
    nonempty_rows = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue
        nonempty_rows += 1
        normalized = " ".join(raw_line.split())
        if normalized in normalized_rows:
            duplicate_rows.append(
                {
                    "line_number": line_number,
                    "first_line_number": normalized_rows[normalized],
                    "reason": "exact duplicate annotation row",
                    "raw_line": raw_line,
                }
            )
        else:
            normalized_rows[normalized] = line_number
        try:
            parsed = parse_yolo_annotation(
                raw_line, source_names, annotation_format
            )
            if parsed.box.was_clipped:
                x_center, y_center, width, height = parsed.source_values
                raw_edges = {
                    "x_min": x_center - width / 2.0,
                    "y_min": y_center - height / 2.0,
                    "x_max": x_center + width / 2.0,
                    "y_max": y_center + height / 2.0,
                }
                overrun = max(
                    0.0,
                    -raw_edges["x_min"],
                    -raw_edges["y_min"],
                    raw_edges["x_max"] - 1.0,
                    raw_edges["y_max"] - 1.0,
                )
                if overrun > BOUNDARY_CLIP_TOLERANCE:
                    raise ValueError(
                        "annotation box exceeds normalized image boundaries by "
                        f"{overrun:.12g}, above tolerance {BOUNDARY_CLIP_TOLERANCE:.12g}"
                    )
                corrections.append(
                    {
                        "line_number": line_number,
                        "source_class_id": parsed.class_id,
                        "source_class_name": parsed.class_name,
                        "maximum_overrun": overrun,
                        **{f"raw_{key}": value for key, value in raw_edges.items()},
                        "clipped_x_min": parsed.box.x_min,
                        "clipped_y_min": parsed.box.y_min,
                        "clipped_x_max": parsed.box.x_max,
                        "clipped_y_max": parsed.box.y_max,
                        "raw_line": raw_line,
                    }
                )
            boxes.append(
                _box_from_parsed(
                    parsed.box,
                    source_class_id=parsed.class_id,
                    source_class_name=parsed.class_name,
                    source_line=line_number,
                    mapping_by_source=mapping_by_source,
                )
            )
        except ValueError as exc:
            invalid.append(
                {
                    "line_number": line_number,
                    "reason": str(exc),
                    "raw_line": raw_line,
                    "source_label": source_label,
                }
            )
    if duplicate_rows:
        invalid.extend(duplicate_rows)
    for left_index, first in enumerate(boxes):
        for second in boxes[left_index + 1 :]:
            if first.class_id != second.class_id:
                continue
            iou = _box_iou(first, second)
            if iou + 1e-12 < NEAR_DUPLICATE_IOU_THRESHOLD:
                continue
            row = {
                "line_number": second.source_line,
                "first_line_number": first.source_line,
                "second_line_number": second.source_line,
                "class_id": first.class_id,
                "class_name": first.class_name,
                "iou": iou,
                "threshold": NEAR_DUPLICATE_IOU_THRESHOLD,
                "reason": (
                    "same-class annotation boxes overlap at or above duplicate-review "
                    f"threshold IoU {NEAR_DUPLICATE_IOU_THRESHOLD:.2f}"
                ),
                "raw_line": "",
            }
            near_duplicates.append(row)
    if near_duplicates:
        invalid.extend(near_duplicates)
    return (
        boxes,
        nonempty_rows == 0,
        invalid,
        duplicate_rows,
        near_duplicates,
        corrections,
    )


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def _label_files(directory: Path) -> list[Path]:
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.casefold() == ".txt"),
        key=lambda path: path.name.casefold(),
    )


def _unique_stem_map(paths: Sequence[Path], description: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in paths:
        key = path.stem.casefold()
        if key in result:
            raise ValueError(
                f"Duplicate case-insensitive {description} stem {path.stem!r}: "
                f"{result[key]} and {path}"
            )
        result[key] = path
    return result


def _read_image_metadata(payload: bytes, source: Path) -> tuple[int, int, str | None]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image.verify()
        if width <= 0 or height <= 0:
            raise ValueError(f"non-positive image dimensions {width}x{height}")
        return width, height, None
    except Exception as exc:  # Pillow exposes several format-specific exceptions.
        return 0, 0, f"unreadable image: {source}: {exc}"


def _metadata_files(source_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in ("README.dataset.txt", "README.roboflow.txt"):
        path = source_root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        statements = [
            line.strip()
            for line in text.splitlines()
            if "augmentation" in line.casefold() or "pre-processing" in line.casefold()
        ]
        rows.append(
            {
                "path": name,
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "augmentation_statements": statements,
            }
        )
    return rows


def audit_extracted_source(
    source_dir: str | Path,
    *,
    brand_profile: str,
    class_order: str = "registry",
    brand_registry_path: str | Path | None = None,
    source_registry_path: str | Path | None = None,
) -> tuple[
    dict[str, Any],
    list[SourceOccurrence],
    tuple[str, ...],
    list[SourceMapping],
    DatasetSource,
]:
    """Read and validate one declared extracted-directory YOLO source."""

    source_root = Path(source_dir).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Extracted detector source does not exist: {source_root}")
    source_registry = (
        load_dataset_sources(source_registry_path)
        if source_registry_path is not None
        else load_dataset_sources()
    )
    configured_source = source_registry.match(source_root)
    if configured_source.annotation_format is not AnnotationFormat.YOLO_BBOX:
        raise ValueError(
            f"Extracted detector source {configured_source.name!r} must explicitly declare "
            f"yolo_bbox, found {configured_source.annotation_format.value}"
        )
    if configured_source.class_mapping_source != "data_yaml":
        raise ValueError(
            f"Extracted detector source {configured_source.name!r} must use "
            f"class_mapping_source=data_yaml"
        )

    yaml_path = source_root / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Extracted detector source has no data.yaml: {yaml_path}")
    yaml_bytes = yaml_path.read_bytes()
    payload = yaml.safe_load(yaml_bytes.decode("utf-8-sig")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Source data.yaml must contain a mapping: {yaml_path}")
    source_names = normalize_yolo_names(payload.get("names"))
    if "nc" not in payload:
        raise ValueError("Source data.yaml must explicitly declare nc")
    try:
        declared_nc = int(payload["nc"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Source data.yaml nc is not an integer: {payload.get('nc')!r}") from exc
    if declared_nc != len(source_names):
        raise ValueError(
            f"Source data.yaml declares nc={declared_nc}, but names contains {len(source_names)} classes"
        )

    registry = (
        load_brand_registry(brand_registry_path)
        if brand_registry_path is not None
        else load_brand_registry()
    )
    target_names, mapping_rows = _source_mapping(
        source_names, registry, brand_profile, class_order
    )
    mapping_by_source = {row.source_id: row for row in mapping_rows}
    layouts, path_report = _resolve_source_layout(source_root, yaml_path, payload)

    occurrences: list[SourceOccurrence] = []
    pairing_rows: list[dict[str, Any]] = []
    orphan_labels: list[str] = []
    source_snapshot: dict[str, tuple[int, int]] = {}
    raw_counts: dict[str, Any] = {}
    dimension_counter: Counter[tuple[int, int]] = Counter()
    family_splits: dict[str, set[str]] = defaultdict(set)
    names_with_augmentation_marker = 0
    source_content_hashes: dict[str, str] = {}

    for split in OUTPUT_SPLITS:
        image_dir, label_dir = layouts[split]
        images = _image_files(image_dir)
        labels = _label_files(label_dir)
        image_by_stem = _unique_stem_map(images, "image")
        label_by_stem = _unique_stem_map(labels, "label")
        split_orphans = sorted(set(label_by_stem) - set(image_by_stem))
        orphan_labels.extend(
            _relative(label_by_stem[stem], source_root) for stem in split_orphans
        )
        pairing_rows.append(
            {
                "split": split,
                "images": len(images),
                "labels": len(labels),
                "missing_labels": len(set(image_by_stem) - set(label_by_stem)),
                "orphan_labels": len(split_orphans),
            }
        )
        class_instances: Counter[str] = Counter()
        class_images: Counter[str] = Counter()
        empty_labels = 0
        invalid_images = 0
        invalid_rows = 0
        duplicate_rows = 0
        for image_path in images:
            image_relative = _relative(image_path, source_root)
            image_stat = image_path.stat()
            source_snapshot[image_relative] = (image_stat.st_size, image_stat.st_mtime_ns)
            image_payload = image_path.read_bytes()
            image_hash = sha256_bytes(image_payload)
            source_content_hashes[image_relative] = image_hash
            width, height, image_error = _read_image_metadata(image_payload, image_path)
            if image_error:
                invalid_images += 1
            else:
                dimension_counter[(width, height)] += 1

            label_path = label_by_stem.get(image_path.stem.casefold())
            label_present = label_path is not None
            label_relative = (
                _relative(label_path, source_root)
                if label_path is not None
                else _relative(label_dir / f"{image_path.stem}.txt", source_root)
            )
            label_hash: str | None = None
            label_stat_size: int | None = None
            label_stat_mtime: int | None = None
            boxes: list[DetectionBox] = []
            row_errors: list[dict[str, Any]] = []
            duplicated: list[dict[str, Any]] = []
            near_duplicated: list[dict[str, Any]] = []
            corrections: list[dict[str, Any]] = []
            explicit_background = False
            if label_path is None:
                row_errors.append(
                    {
                        "line_number": "",
                        "reason": "missing label for image",
                        "raw_line": "",
                        "source_label": label_relative,
                    }
                )
            else:
                label_stat = label_path.stat()
                label_stat_size = label_stat.st_size
                label_stat_mtime = label_stat.st_mtime_ns
                source_snapshot[label_relative] = (
                    label_stat.st_size,
                    label_stat.st_mtime_ns,
                )
                label_payload = label_path.read_bytes()
                label_hash = sha256_bytes(label_payload)
                source_content_hashes[label_relative] = label_hash
                (
                    boxes,
                    explicit_background,
                    row_errors,
                    duplicated,
                    near_duplicated,
                    corrections,
                ) = _parse_label_bytes(
                    label_payload,
                    source_names=source_names,
                    annotation_format=configured_source.annotation_format,
                    mapping_by_source=mapping_by_source,
                    source_label=label_relative,
                )
            if image_error:
                row_errors.append(
                    {
                        "line_number": "",
                        "reason": image_error,
                        "raw_line": "",
                        "source_label": label_relative,
                    }
                )
            invalid_rows += len(row_errors)
            duplicate_rows += len(duplicated)
            empty_labels += int(explicit_background)
            present_names = {box.class_name for box in boxes}
            class_instances.update(box.class_name for box in boxes)
            class_images.update(present_names)
            family_id = normalize_source_family(image_path.name)
            variant_id = normalize_source_variant(image_path.name)
            family_splits[family_id].add(split)
            names_with_augmentation_marker += int(
                bool(re.search(r"(?:[_-](?:aug|augmentation)[_-]?\d+)(?:_jpg)?\.rf\.", image_path.stem, re.IGNORECASE))
            )
            occurrences.append(
                SourceOccurrence(
                    source_split=split,
                    image_path=image_path,
                    label_path=label_path or (label_dir / f"{image_path.stem}.txt"),
                    source_image_relative=image_relative,
                    source_label_relative=label_relative,
                    source_filename=image_path.name,
                    family_id=family_id,
                    variant_id=variant_id,
                    image_sha256=image_hash,
                    label_sha256=label_hash,
                    image_size_bytes=image_stat.st_size,
                    image_mtime_ns=image_stat.st_mtime_ns,
                    label_size_bytes=label_stat_size,
                    label_mtime_ns=label_stat_mtime,
                    width=width,
                    height=height,
                    boxes=boxes,
                    label_present=label_present,
                    explicit_background=explicit_background,
                    invalid_rows=row_errors,
                    duplicate_rows=duplicated,
                    near_duplicate_rows=near_duplicated,
                    boundary_corrections=corrections,
                )
            )
        raw_counts[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty_labels,
            "invalid_images": invalid_images,
            "invalid_rows": invalid_rows,
            "duplicate_annotation_rows": duplicate_rows,
            "near_duplicate_annotation_pairs": sum(
                len(item.near_duplicate_rows)
                for item in occurrences
                if item.source_split == split
            ),
            "boundary_corrections": sum(
                len(item.boundary_corrections)
                for item in occurrences
                if item.source_split == split
            ),
            "per_class": {
                name: {
                    "images": class_images[name],
                    "instances": class_instances[name],
                }
                for name in target_names
            },
        }
        for label_path in labels:
            relative = _relative(label_path, source_root)
            if relative not in source_snapshot:
                stat = label_path.stat()
                source_snapshot[relative] = (stat.st_size, stat.st_mtime_ns)
                source_content_hashes[relative] = sha256_file(label_path)

    yaml_stat = yaml_path.stat()
    source_snapshot["data.yaml"] = (yaml_stat.st_size, yaml_stat.st_mtime_ns)
    source_content_hashes["data.yaml"] = sha256_bytes(yaml_bytes)
    metadata = _metadata_files(source_root)
    for row in metadata:
        path = source_root / row["path"]
        stat = path.stat()
        source_snapshot[row["path"]] = (stat.st_size, stat.st_mtime_ns)
        source_content_hashes[row["path"]] = row["sha256"]

    cross_split_families = {
        family: sorted(splits)
        for family, splits in sorted(family_splits.items())
        if len(splits) > 1
    }
    same_variant_report, _same_variant_rows, _suspicious_variants = (
        _same_variant_analysis(occurrences)
    )
    audit = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "source_name": configured_source.name,
        "source_root": str(source_root),
        "data_yaml": str(yaml_path),
        "data_yaml_sha256": sha256_bytes(yaml_bytes),
        "declared_nc": declared_nc,
        "source_names": list(source_names),
        "target_profile": brand_profile,
        "class_order": class_order,
        "target_names": list(target_names),
        "annotation_format": configured_source.annotation_format.value,
        "class_mapping_source": configured_source.class_mapping_source,
        "path_resolution": path_report,
        "pairing": pairing_rows,
        "orphan_labels": orphan_labels,
        "counts": raw_counts,
        "image_dimensions": {
            "distinct": len(dimension_counter),
            "min_width": min((size[0] for size in dimension_counter), default=0),
            "max_width": max((size[0] for size in dimension_counter), default=0),
            "min_height": min((size[1] for size in dimension_counter), default=0),
            "max_height": max((size[1] for size in dimension_counter), default=0),
            "most_common": [
                {"width": size[0], "height": size[1], "images": count}
                for size, count in dimension_counter.most_common(20)
            ],
        },
        "family_analysis": {
            "normalization": (
                "case-fold filename stem; remove terminal .rf hash, _jpg and _augN; "
                "exact duplicate hashes connect all observed family IDs"
            ),
            "heuristic": True,
            "families": len(family_splits),
            "cross_split_families": len(cross_split_families),
            "train_val_families": sum(
                {"train", "val"} <= set(splits)
                for splits in cross_split_families.values()
            ),
            "train_test_families": sum(
                {"train", "test"} <= set(splits)
                for splits in cross_split_families.values()
            ),
            "val_test_conflicts": sum(
                {"val", "test"} <= set(splits)
                for splits in cross_split_families.values()
            ),
            "cross_split_examples": [
                {"family_id": family, "splits": splits}
                for family, splits in list(cross_split_families.items())[:100]
            ],
        },
        "same_variant_analysis": same_variant_report,
        "roboflow_metadata": metadata,
        "original_zip_yaml_comparison": {
            "status": "unverified_source_zip_not_provided",
            "zipinfo_entries_compared": 0,
            "duplicate_yaml_entries_verified": False,
            "limitation": (
                "An extracted directory cannot preserve or prove duplicate ZipInfo member "
                "semantics; provide and audit the original ZIP separately if it becomes available."
            ),
        },
        "offline_augmentation_evidence": {
            "filenames_with_explicit_aug_marker": names_with_augmentation_marker,
            "interpretation": (
                "Roboflow metadata describes export-time preprocessing/augmentation only; "
                "filename family markers are upstream evidence but do not identify a transform "
                "such as mosaic."
            ),
        },
        "boundary_policy": {
            "clip_and_log_maximum_overrun": BOUNDARY_CLIP_TOLERANCE,
            "larger_overrun_action": "invalid row and whole connected-family quarantine",
            "corrections": sum(len(item.boundary_corrections) for item in occurrences),
            "maximum_observed_correction": max(
                (
                    float(row["maximum_overrun"])
                    for item in occurrences
                    for row in item.boundary_corrections
                ),
                default=0.0,
            ),
        },
        "near_duplicate_annotation_policy": {
            "same_target_class_only": True,
            "iou_threshold": NEAR_DUPLICATE_IOU_THRESHOLD,
            "action": "invalid annotation set and whole connected-family quarantine",
            "pairs": sum(len(item.near_duplicate_rows) for item in occurrences),
        },
        "source_snapshot": {
            "file_count": len(source_snapshot),
            "fingerprint": "sha256:"
            + _sha256_json(
                [
                    {"path": path, "sha256": digest}
                    for path, digest in sorted(source_content_hashes.items())
                ]
            ),
            "fingerprint_basis": "sorted source-relative paths and exact file-content SHA-256 values",
        },
        "_source_snapshot_private": source_snapshot,
    }
    return audit, occurrences, target_names, mapping_rows, configured_source


def load_review_decisions(path: str | Path | None) -> dict[str, str]:
    """Load explicit exact-image candidate choices from a prior review."""

    if path is None:
        return {}
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Review decisions file does not exist: {resolved}")
    decisions: dict[str, str] = {}
    for row in _read_csv(resolved):
        image_hash = row.get("image_sha256", "").strip().casefold()
        candidate_id = row.get("selected_candidate_id", "").strip()
        if not image_hash or not candidate_id:
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", image_hash):
            raise ValueError(f"Invalid image SHA-256 in review decisions: {image_hash!r}")
        previous = decisions.setdefault(image_hash, candidate_id)
        if previous != candidate_id:
            raise ValueError(f"Conflicting review decisions for {image_hash}")
    return decisions


def _source_reference(source: SourceOccurrence) -> dict[str, Any]:
    return {
        "source_split": source.source_split,
        "image": source.source_image_relative,
        "label": source.source_label_relative,
        "family_id": source.family_id,
        "variant_id": source.variant_id,
        "label_present": source.label_present,
        "explicit_background": source.explicit_background,
        "invalid_rows": len(source.invalid_rows),
        "box_count": len(source.boxes),
    }


def reconcile_exact_duplicates(
    occurrences: Sequence[SourceOccurrence],
    decisions: Mapping[str, str] | None = None,
) -> tuple[
    list[ImageUnit],
    list[HashAudit],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Deduplicate equal image bytes without guessing between conflicting labels."""

    decisions = dict(decisions or {})
    by_hash: dict[str, list[SourceOccurrence]] = defaultdict(list)
    for occurrence in occurrences:
        by_hash[occurrence.image_sha256].append(occurrence)

    units: list[ImageUnit] = []
    audits: list[HashAudit] = []
    conflict_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    used_decisions: set[str] = set()

    for image_hash, hash_sources in sorted(by_hash.items()):
        sources = sorted(
            hash_sources,
            key=lambda source: (
                source.source_split,
                source.source_image_relative.casefold(),
            ),
        )
        candidate_by_signature: dict[
            tuple[tuple[Any, ...], ...], tuple[DetectionBox, ...]
        ] = {}
        for source in sources:
            signature = source.annotation_signature()
            candidate_by_signature.setdefault(
                signature, tuple(sorted(source.boxes, key=lambda box: box.signature()))
            )
        candidate_signatures = sorted(candidate_by_signature)
        candidate_sets = [candidate_by_signature[item] for item in candidate_signatures]
        candidate_ids = {
            signature: f"candidate_{index:02d}"
            for index, signature in enumerate(candidate_signatures)
        }
        invalid = [row for source in sources for row in source.invalid_rows]
        family_ids = {source.family_id for source in sources}
        status = ""
        reason = ""
        selected: tuple[DetectionBox, ...] | None = None
        selected_candidate_id: str | None = None
        resolution = ""

        if invalid:
            status = "unresolved_invalid_source"
            reason = "one or more exact-image occurrences contain invalid image/label data"
        elif len(candidate_sets) == 1:
            status = "accepted_equivalent"
            selected = candidate_sets[0]
            selected_candidate_id = candidate_ids[candidate_signatures[0]]
            resolution = (
                "all exact-content occurrences have the same target-class annotation set"
            )
        else:
            status = "unresolved_annotation_conflict"
            reason = (
                "exact image bytes have conflicting target-class annotation sets; "
                "no source-split precedence or box union was applied"
            )

        requested = decisions.get(image_hash)
        if requested and status == "unresolved_annotation_conflict":
            matching = [
                index
                for index, signature in enumerate(candidate_signatures)
                if candidate_ids[signature] == requested
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"Review decision {requested!r} is unavailable for {image_hash}"
                )
            selected = candidate_sets[matching[0]]
            selected_candidate_id = requested
            status = "accepted_manual"
            reason = ""
            resolution = f"manual review selected {requested}"
            used_decisions.add(image_hash)
        elif requested and status != "unresolved_annotation_conflict":
            raise ValueError(
                f"Review decision for {image_hash} is not applicable to status {status}"
            )

        for index, candidate in enumerate(candidate_sets):
            signature = candidate_signatures[index]
            candidate_id = candidate_ids[signature]
            matching_sources = [
                _source_reference(source)
                for source in sources
                if source.annotation_signature() == signature
            ]
            candidate_rows.append(
                {
                    "image_sha256": image_hash,
                    "candidate_id": candidate_id,
                    "box_count": len(candidate),
                    "class_counts": json.dumps(
                        dict(sorted(Counter(box.class_name for box in candidate).items())),
                        ensure_ascii=False,
                    ),
                    "yolo_rows": json.dumps(
                        [box.yolo_row() for box in candidate], ensure_ascii=False
                    ),
                    "source_occurrences": json.dumps(
                        matching_sources, ensure_ascii=False
                    ),
                }
            )

        audit = HashAudit(
            image_sha256=image_hash,
            sources=sources,
            family_ids=family_ids,
            candidate_sets=candidate_sets,
            status=status,
            reason=reason,
            selected_candidate_id=selected_candidate_id,
        )
        audits.append(audit)
        status_counts[status] += 1

        if status.startswith("unresolved"):
            conflict_rows.append(
                {
                    "image_sha256": image_hash,
                    "status": status,
                    "reason": reason,
                    "source_occurrence_count": len(sources),
                    "candidate_set_count": len(candidate_sets),
                    "candidate_box_counts": json.dumps(
                        [len(candidate) for candidate in candidate_sets]
                    ),
                    "candidate_class_counts": json.dumps(
                        [
                            dict(
                                sorted(
                                    Counter(
                                        box.class_name for box in candidate
                                    ).items()
                                )
                            )
                            for candidate in candidate_sets
                        ],
                        ensure_ascii=False,
                    ),
                    "family_ids": json.dumps(sorted(family_ids), ensure_ascii=False),
                    "sources": json.dumps(
                        [_source_reference(source) for source in sources],
                        ensure_ascii=False,
                    ),
                }
            )

        if selected is not None:
            canonical = min(
                sources,
                key=lambda source: (
                    source.source_image_relative.casefold(),
                    source.source_split,
                ),
            )
            units.append(
                ImageUnit(
                    image_sha256=image_hash,
                    canonical=canonical,
                    sources=sources,
                    family_ids=family_ids,
                    boxes=list(selected),
                    resolution=resolution,
                    explicit_background=len(selected) == 0,
                )
            )

    unused = sorted(set(decisions) - used_decisions)
    if unused:
        raise ValueError(
            f"Review decisions were not used by an unresolved annotation conflict: {unused[:10]}"
        )
    return units, audits, conflict_rows, candidate_rows, {
        "raw_image_occurrences": len(occurrences),
        "unique_image_hashes": len(by_hash),
        "duplicate_occurrences_removed": len(occurrences) - len(by_hash),
        "exact_duplicate_hash_groups": sum(
            len(items) > 1 for items in by_hash.values()
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "manual_decisions_loaded": len(decisions),
        "manual_decisions_used": len(used_decisions),
        "conflict_policy": (
            "quarantine the complete connected filename-family component for invalid or "
            "conflicting exact-image annotations unless an explicit candidate review decision "
            "resolves an annotation-only conflict"
        ),
    }


def load_historical_holdout_locks(
    dataset_dirs: Sequence[str | Path],
    *,
    target_names: Sequence[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, Any]]:
    """Hash-verify historical provenance and import only validation/test locks."""

    hash_locks: dict[str, set[str]] = defaultdict(set)
    family_locks: dict[str, set[str]] = defaultdict(set)
    reports: list[dict[str, Any]] = []
    for raw_directory in dataset_dirs:
        dataset_dir = Path(raw_directory).resolve()
        if not dataset_dir.exists():
            reports.append(
                {
                    "path": str(dataset_dir),
                    "status": "not_available",
                    "locked_rows": 0,
                }
            )
            continue
        required = (
            dataset_dir / "mapping.json",
            dataset_dir / "dataset_checks.json",
            dataset_dir / "provenance.csv",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Historical detector provenance is incomplete: {missing}"
            )
        mapping = json.loads(required[0].read_text(encoding="utf-8"))
        historical_names_value = mapping.get("output_names")
        if not isinstance(historical_names_value, Mapping):
            raise ValueError(
                f"Historical mapping has no output_names object: {required[0]}"
            )
        historical_names = [
            str(historical_names_value[str(index)])
            for index in range(len(historical_names_value))
        ]
        if len(historical_names) != len(set(historical_names)):
            raise ValueError(
                f"Historical class mapping contains duplicate names: {dataset_dir}"
            )
        missing_members = sorted(set(historical_names) - set(target_names))
        if missing_members:
            raise ValueError(
                "Historical class membership is incompatible with current target profile: "
                f"{dataset_dir}; missing={missing_members}"
            )
        checks = json.loads(required[1].read_text(encoding="utf-8"))
        exact_overlap = checks.get("exact_image_cross_split_overlap", {})
        if (
            int(checks.get("cross_split_family_overlap", -1)) != 0
            or not isinstance(exact_overlap, Mapping)
            or any(int(value) != 0 for value in exact_overlap.values())
        ):
            raise ValueError(
                f"Historical detector split checks are not trustworthy: {dataset_dir}"
            )
        verified_rows = 0
        locked_rows: Counter[str] = Counter()
        for row in _read_csv(required[2]):
            image_path = dataset_dir / PurePosixPath(row["output_image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Historical provenance image is missing: {image_path}"
                )
            actual_hash = sha256_file(image_path)
            if actual_hash.casefold() != row["image_sha256"].casefold():
                raise ValueError(
                    f"Historical provenance image hash differs: {image_path}"
                )
            verified_rows += 1
            split = _normalized_split(row["output_split"])
            if split not in {"val", "test"}:
                continue
            digest = row["image_sha256"].casefold()
            hash_locks[digest].add(split)
            locked_rows[split] += 1
            family_values = json.loads(row.get("family_ids", "[]"))
            for family in family_values:
                family_locks[str(family).casefold()].add(split)
        reports.append(
            {
                "path": str(dataset_dir),
                "status": "verified",
                "historical_class_count": len(historical_names),
                "historical_class_membership_verified": True,
                "historical_class_order_required": False,
                "provenance_rows_hash_verified": verified_rows,
                "locked_rows": dict(sorted(locked_rows.items())),
            }
        )
    return hash_locks, family_locks, {
        "datasets": reports,
        "locked_hashes": len(hash_locks),
        "locked_families": len(family_locks),
    }


def build_family_components(
    units: Sequence[ImageUnit],
    audits: Sequence[HashAudit],
    historical_hash_locks: Mapping[str, set[str]],
    historical_family_locks: Mapping[str, set[str]],
    *,
    target_names: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Connect filename families through exact duplicates and preserve holdouts."""

    parent: dict[str, str] = {}
    all_occurrences = [source for audit in audits for source in audit.sources]
    same_variant_report, _same_variant_rows, suspicious_variants = (
        _same_variant_analysis(all_occurrences)
    )

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left != right:
            low, high = sorted((left, right))
            parent[high] = low

    for audit in audits:
        families = sorted(audit.family_ids)
        for family in families:
            find(family)
        for family in families[1:]:
            union(families[0], family)

    audits_by_component: dict[str, list[HashAudit]] = defaultdict(list)
    for audit in audits:
        roots = [find(family) for family in audit.family_ids]
        component = min(roots)
        audits_by_component[component].append(audit)
    unit_by_hash = {unit.image_sha256: unit for unit in units}
    clean: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()

    for component, component_audits in sorted(audits_by_component.items()):
        families = sorted(
            {
                family
                for audit in component_audits
                for family in audit.family_ids
            }
        )
        variants = sorted(
            {
                source.variant_id
                for audit in component_audits
                for source in audit.sources
            }
        )
        component_variant_conflicts = sorted(
            set(variants) & suspicious_variants
        )
        component_units = sorted(
            (
                unit_by_hash[audit.image_sha256]
                for audit in component_audits
                if audit.image_sha256 in unit_by_hash
            ),
            key=lambda unit: unit.image_sha256,
        )
        unresolved = sorted(
            audit.image_sha256
            for audit in component_audits
            if audit.status.startswith("unresolved")
        )
        source_splits = {
            source.source_split
            for audit in component_audits
            for source in audit.sources
        }
        holdout_sources: dict[str, set[str]] = defaultdict(set)
        if "val" in source_splits:
            holdout_sources["val"].add("source validation assignment")
        if "test" in source_splits:
            holdout_sources["test"].add("source test assignment")
        for audit in component_audits:
            for split in historical_hash_locks.get(audit.image_sha256, set()):
                holdout_sources[split].add(
                    f"historical exact-image lock:{audit.image_sha256}"
                )
        for family in families:
            for split in historical_family_locks.get(family.casefold(), set()):
                holdout_sources[split].add(f"historical family lock:{family}")
        holdout_splits = set(holdout_sources)
        reasons: list[str] = []
        if unresolved:
            reasons.append("connected family contains invalid or unresolved annotations")
            reason_counts["annotation_or_source_conflict_family"] += 1
        if component_variant_conflicts:
            reasons.append(
                "connected family contains a cross-split same-variant class-multiplicity conflict"
            )
            reason_counts["same_variant_annotation_conflict_family"] += 1
        if len(holdout_splits) > 1:
            reasons.append("family component links validation and test holdout assignments")
            reason_counts["validation_test_holdout_conflict_family"] += 1
        output_split = (
            next(iter(holdout_splits))
            if len(holdout_splits) == 1
            else ("train" if not holdout_splits else None)
        )
        brand_instances = Counter(
            box.class_name for unit in component_units for box in unit.boxes
        )
        brand_images = {
            name: sum(any(box.class_name == name for box in unit.boxes) for unit in component_units)
            for name in target_names
        }
        group = {
            "group_id": component,
            "family_ids": families,
            "variant_ids": variants,
            "same_variant_conflicts": component_variant_conflicts,
            "audits": sorted(
                component_audits, key=lambda audit: audit.image_sha256
            ),
            "units": component_units,
            "source_splits": sorted(source_splits),
            "output_split": output_split,
            "forced_split": output_split,
            "force_reasons": sorted(
                reason for values in holdout_sources.values() for reason in values
            ),
            "image_count": len(component_units),
            "audit_image_count": len(component_audits),
            "class_instance_counts": dict(sorted(brand_instances.items())),
            "class_image_counts": brand_images,
            "unresolved_hashes": unresolved,
            "quarantine_reasons": reasons,
        }
        if reasons:
            quarantined.append(group)
            continue
        if not component_units:
            continue
        assert output_split in OUTPUT_SPLITS
        for unit in component_units:
            unit.component_id = component
            unit.output_split = output_split
            unit.forced_split = output_split
            unit.force_reasons.update(group["force_reasons"])
        clean.append(group)

    return clean, quarantined, {
        "family_components": len(audits_by_component),
        "clean_family_components": len(clean),
        "quarantined_family_components": len(quarantined),
        "quarantine_reason_counts": dict(sorted(reason_counts.items())),
        "same_variant_analysis": same_variant_report,
        "split_policy": {
            "train_only": "train",
            "train_plus_validation": "validation",
            "train_plus_test": "test",
            "validation_plus_test": "quarantine complete connected component",
            "historical_locks": "validation/test locks participate in the same rule",
            "random_split_used": False,
        },
    }


def _read_verified_source_image(occurrence: SourceOccurrence) -> bytes:
    stat = occurrence.image_path.stat()
    if (
        stat.st_size != occurrence.image_size_bytes
        or stat.st_mtime_ns != occurrence.image_mtime_ns
    ):
        raise RuntimeError(
            f"Source image changed during preparation: {occurrence.image_path}"
        )
    payload = occurrence.image_path.read_bytes()
    if sha256_bytes(payload) != occurrence.image_sha256:
        raise RuntimeError(
            f"Source image content changed during preparation: {occurrence.image_path}"
        )
    return payload


def _verify_source_snapshot(
    source_root: Path, snapshot: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    for relative, expected in snapshot.items():
        path = source_root / PurePosixPath(relative)
        if not path.is_file():
            raise RuntimeError(f"Source file disappeared during preparation: {path}")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (int(expected[0]), int(expected[1])):
            raise RuntimeError(f"Source file changed during preparation: {path}")
    return {
        "source_unchanged_verified": True,
        "verified_files": len(snapshot),
        "verification": "size and nanosecond modification time rechecked after all derived writes",
    }


def assign_output_names(units: Sequence[ImageUnit]) -> None:
    """Assign short names that are stable for the same exact image bytes."""

    used: dict[str, str] = {}
    for unit in sorted(units, key=lambda item: item.image_sha256):
        suffix = unit.canonical.image_path.suffix.casefold()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".jpg"
        name = f"i_{unit.image_sha256[:24]}{suffix}"
        previous = used.setdefault(name.casefold(), unit.image_sha256)
        if previous != unit.image_sha256:
            raise RuntimeError(
                f"Short deterministic image-name collision: {previous} and {unit.image_sha256}"
            )
        unit.output_name = name


def _preview_color(class_id: int, class_count: int) -> tuple[int, int, int]:
    hue = (class_id / max(class_count, 1) + 0.08) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return int(red * 255), int(green * 255), int(blue * 255)


def draw_boxes(
    image: Image.Image,
    boxes: Sequence[DetectionBox],
    title: str,
    *,
    class_count: int,
) -> Image.Image:
    canvas = ImageOps.exif_transpose(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, min(canvas.size) // 300)
    for box in boxes:
        color = _preview_color(box.class_id, class_count)
        coords = (
            int(round(box.x_min * canvas.width)),
            int(round(box.y_min * canvas.height)),
            int(round(box.x_max * canvas.width)),
            int(round(box.y_max * canvas.height)),
        )
        draw.rectangle(coords, outline=color, width=line_width)
        label = f"{box.class_id} {box.class_name}"
        text_box = draw.textbbox((coords[0], coords[1]), label, font=font)
        draw.rectangle(text_box, fill=color)
        draw.text((coords[0], coords[1]), label, fill="black", font=font)
    banner_height = 26
    output = Image.new("RGB", (canvas.width, canvas.height + banner_height), "white")
    output.paste(canvas, (0, banner_height))
    ImageDraw.Draw(output).text((5, 6), title, fill="black", font=font)
    return output


def _make_contact_sheet(paths: Sequence[Path], output_path: Path) -> None:
    if not paths:
        return
    thumbs: list[Image.Image] = []
    for path in paths:
        with Image.open(path) as image:
            thumbs.append(
                ImageOps.contain(
                    image.convert("RGB"), (320, 240), Image.Resampling.LANCZOS
                )
            )
    columns = 4
    cell_width, cell_height = 330, 250
    rows = math.ceil(len(thumbs) / columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    for index, thumb in enumerate(thumbs):
        sheet.paste(
            thumb,
            ((index % columns) * cell_width + 5, (index // columns) * cell_height + 5),
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)


def _make_review_sheet(
    image_payload: bytes,
    audit: HashAudit,
    output_path: Path,
    *,
    class_count: int,
) -> bool:
    try:
        with Image.open(io.BytesIO(image_payload)) as image:
            original = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        return False
    panels = [
        draw_boxes(original.copy(), (), "original", class_count=class_count)
    ]
    for index, candidate in enumerate(audit.candidate_sets):
        counts = dict(sorted(Counter(box.class_name for box in candidate).items()))
        panels.append(
            draw_boxes(
                original.copy(),
                candidate,
                f"candidate_{index:02d}: {json.dumps(counts, ensure_ascii=False)}",
                class_count=class_count,
            )
        )
    fitted = [
        ImageOps.contain(panel, (480, 420), Image.Resampling.LANCZOS)
        for panel in panels
    ]
    sheet = Image.new(
        "RGB", (sum(panel.width for panel in fitted), max(panel.height for panel in fitted)), "white"
    )
    offset = 0
    for panel in fitted:
        sheet.paste(panel, (offset, 0))
        offset += panel.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=88, optimize=True)
    return True


def select_preview_units(
    units: Sequence[ImageUnit],
    target_names: Sequence[str],
    limit: int,
) -> list[ImageUnit]:
    remaining = sorted(units, key=lambda unit: unit.image_sha256)
    selected: list[ImageUnit] = []
    uncovered = {
        (split, name) for split in OUTPUT_SPLITS for name in target_names
    }
    while remaining and len(selected) < limit and uncovered:
        best = max(
            remaining,
            key=lambda unit: (
                sum(
                    (unit.output_split, name) in uncovered
                    for name in {box.class_name for box in unit.boxes}
                ),
                len({box.class_name for box in unit.boxes}),
                -len(unit.boxes),
                unit.image_sha256,
            ),
        )
        selected.append(best)
        remaining.remove(best)
        for name in {box.class_name for box in best.boxes}:
            uncovered.discard((best.output_split, name))
    for unit in remaining:
        if len(selected) >= limit:
            break
        selected.append(unit)
    return selected


def _source_references(sources: Sequence[SourceOccurrence]) -> str:
    return json.dumps(
        [_source_reference(source) for source in sources], ensure_ascii=False
    )


def _fingerprint_payload(
    mapping_document: Mapping[str, Any],
    provenance_rows: Sequence[Mapping[str, Any]],
    quarantine_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    source_mapping = [
        {
            "source_id": int(row["source_id"]),
            "source_name": str(row["source_name"]),
            "target_id": int(row["target_id"]),
            "target_name": str(row["target_name"]),
            "registry_id": int(row["registry_id"]),
        }
        for row in mapping_document["source_to_target"]
    ]
    clean = sorted(
        (
            {
                "split": str(row["output_split"]),
                "image": str(row["output_image_path"]),
                "label": str(row["output_label_path"]),
                "image_sha256": str(row["image_sha256"]),
                "label_sha256": str(row["label_sha256"]),
                "family_component_id": str(row["family_component_id"]),
            }
            for row in provenance_rows
        ),
        key=lambda row: (row["split"], row["image"]),
    )
    excluded = sorted(
        (
            {
                "image_sha256": str(row["image_sha256"]),
                "family_component_id": str(row["family_component_id"]),
                "audit_status": str(row["audit_status"]),
                "quarantine_reasons": str(row["quarantine_reasons"]),
            }
            for row in quarantine_rows
        ),
        key=lambda row: (row["family_component_id"], row["image_sha256"]),
    )
    return {
        "schema_version": FINGERPRINT_SCHEMA_VERSION,
        "source_name": mapping_document["source_name"],
        "source_data_yaml_sha256": mapping_document["source_data_yaml_sha256"],
        "annotation_format": mapping_document["annotation_format"],
        "target_profile": mapping_document["target_profile"],
        "class_order": mapping_document["class_order"],
        "target_names": mapping_document["target_names"],
        "source_to_target": source_mapping,
        "clean_records": clean,
        "quarantined_records": excluded,
    }


def _write_source_audit_markdown(path: Path, audit: Mapping[str, Any]) -> None:
    path_report = audit["path_resolution"]
    family = audit["family_analysis"]
    variants = audit["same_variant_analysis"]
    lines = [
        "# Extracted detector source audit",
        "",
        f"- Source: `{audit['source_root']}`",
        f"- Declared source: `{audit['source_name']}`",
        f"- Annotation format: `{audit['annotation_format']}`",
        f"- Classes: {audit['declared_nc']}",
        f"- Source data.yaml SHA-256: `{audit['data_yaml_sha256']}`",
        f"- All declared split paths existed: {str(path_report['all_declared_paths_exist']).lower()}",
        f"- Observed-layout fallback used: {str(path_report['used_observed_layout_fallback']).lower()}",
        f"- Orphan labels: {len(audit['orphan_labels'])}",
        f"- Normalized families: {family['families']}",
        f"- Cross-split families: {family['cross_split_families']}",
        f"- Validation/test family conflicts: {family['val_test_conflicts']}",
        f"- Cross-split same-variant groups: {variants['cross_split_same_variant_groups']}",
        f"- Same-variant class-multiplicity conflicts: {variants['differing_class_multiplicity_groups']}",
        f"- Same-variant class-set conflicts: {variants['differing_class_set_groups']}",
        "",
        "## Split paths",
        "",
        "| Split | Declared | Exists | Selected | Status |",
        "|---|---|---:|---|---|",
    ]
    for row in path_report["split_paths"]:
        lines.append(
            f"| {row['split']} | `{row['declared_value']}` | "
            f"{str(row['declared_exists']).lower()} | `{row['selected_images']}` | "
            f"{row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Raw counts",
            "",
        "| Split | Images | Labels | Empty | Invalid images | Invalid rows | Exact duplicate rows | Near-duplicate pairs | Boundary clips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    pairing = {row["split"]: row for row in audit["pairing"]}
    for split in OUTPUT_SPLITS:
        values = audit["counts"][split]
        lines.append(
            f"| {split} | {values['images']} | {pairing[split]['labels']} | "
            f"{values['empty_labels']} | {values['invalid_images']} | "
            f"{values['invalid_rows']} | {values['duplicate_annotation_rows']} | "
            f"{values['near_duplicate_annotation_pairs']} | "
            f"{values['boundary_corrections']} |"
        )
    lines.extend(
        [
            "",
            "Roboflow metadata statements refer to export-time processing. Filename family "
            "markers are reported separately and are not proof of a particular transform such as mosaic.",
            "",
            "The original ZIP was not inspected by this directory workflow; duplicate ZipInfo data.yaml "
            "entries remain unverified unless a source ZIP is audited separately.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _counts_rows(
    source_audit: Mapping[str, Any],
    clean_units: Sequence[ImageUnit],
    target_names: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for phase in ("source", "prepared"):
        for split in OUTPUT_SPLITS:
            if phase == "source":
                per_class = source_audit["counts"][split]["per_class"]
                family_by_class: dict[str, set[str]] = defaultdict(set)
                families = {name: 0 for name in target_names}
            else:
                split_units = [unit for unit in clean_units if unit.output_split == split]
                instance_counts = Counter(
                    box.class_name for unit in split_units for box in unit.boxes
                )
                image_counts = Counter()
                family_by_class = defaultdict(set)
                for unit in split_units:
                    names = {box.class_name for box in unit.boxes}
                    image_counts.update(names)
                    for name in names:
                        family_by_class[name].add(unit.component_id)
                per_class = {
                    name: {
                        "images": image_counts[name],
                        "instances": instance_counts[name],
                    }
                    for name in target_names
                }
                families = {
                    name: len(family_by_class[name]) for name in target_names
                }
            for class_id, name in enumerate(target_names):
                rows.append(
                    {
                        "phase": phase,
                        "split": split,
                        "class_id": class_id,
                        "class_name": name,
                        "images": per_class[name]["images"],
                        "instances": per_class[name]["instances"],
                        "families": families[name],
                    }
                )
    return rows


def write_prepared_dataset(
    staging: Path,
    final_output_dir: Path,
    *,
    source_audit: dict[str, Any],
    target_names: Sequence[str],
    mapping_rows: Sequence[SourceMapping],
    configured_source: DatasetSource,
    clean_units: Sequence[ImageUnit],
    clean_groups: Sequence[Mapping[str, Any]],
    quarantined_groups: Sequence[Mapping[str, Any]],
    conflict_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    preview_count: int,
) -> dict[str, Any]:
    """Write every derived artifact below an isolated staging directory."""

    assign_output_names(clean_units)
    for split in OUTPUT_SPLITS:
        (staging / "images" / split).mkdir(parents=True, exist_ok=True)
        (staging / "labels" / split).mkdir(parents=True, exist_ok=True)
    for directory in (
        staging / "quarantine" / "images",
        staging / "quarantine" / "candidate_labels",
        staging / "review_sheets",
        staging / "previews",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    provenance_rows: list[dict[str, Any]] = []
    for unit in sorted(clean_units, key=lambda item: item.image_sha256):
        payload = _read_verified_source_image(unit.canonical)
        image_path = staging / "images" / unit.output_split / unit.output_name
        label_path = staging / "labels" / unit.output_split / Path(
            unit.output_name
        ).with_suffix(".txt")
        image_path.write_bytes(payload)
        label_text = "\n".join(
            box.yolo_row() for box in sorted(unit.boxes, key=lambda box: box.signature())
        )
        label_path.write_text(label_text + ("\n" if label_text else ""), encoding="utf-8")
        label_hash = sha256_file(label_path)
        counts = Counter(box.class_name for box in unit.boxes)
        provenance_rows.append(
            {
                "output_split": unit.output_split,
                "output_image_path": image_path.relative_to(staging).as_posix(),
                "output_label_path": label_path.relative_to(staging).as_posix(),
                "image_sha256": unit.image_sha256,
                "label_sha256": label_hash,
                "family_component_id": unit.component_id,
                "family_ids": json.dumps(sorted(unit.family_ids), ensure_ascii=False),
                "forced_holdout": unit.forced_split or "",
                "force_reasons": json.dumps(sorted(unit.force_reasons), ensure_ascii=False),
                "resolution": unit.resolution,
                "explicit_background": unit.explicit_background,
                "selected_source_split": unit.canonical.source_split,
                "selected_source_image": unit.canonical.source_image_relative,
                "selected_source_label": unit.canonical.source_label_relative,
                "source_occurrence_count": len(unit.sources),
                "source_occurrences": _source_references(unit.sources),
                "width": unit.canonical.width,
                "height": unit.canonical.height,
                "box_count": len(unit.boxes),
                "class_box_counts": json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
            }
        )

    quarantine_rows: list[dict[str, Any]] = []
    review_paths: list[Path] = []
    copied_hashes: set[str] = set()
    for group in quarantined_groups:
        for audit in group["audits"]:
            canonical = audit.sources[0]
            suffix = canonical.image_path.suffix.casefold()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".bin"
            quarantine_path = (
                staging
                / "quarantine"
                / "images"
                / f"q_{audit.image_sha256[:24]}{suffix}"
            )
            if audit.image_sha256 not in copied_hashes:
                quarantine_path.write_bytes(_read_verified_source_image(canonical))
                copied_hashes.add(audit.image_sha256)
            candidate_ids: list[str] = []
            for index, candidate in enumerate(audit.candidate_sets):
                candidate_id = f"candidate_{index:02d}"
                candidate_ids.append(candidate_id)
                candidate_path = (
                    staging
                    / "quarantine"
                    / "candidate_labels"
                    / f"{audit.image_sha256}__{candidate_id}.txt"
                )
                text = "\n".join(box.yolo_row() for box in candidate)
                candidate_path.write_text(text + ("\n" if text else ""), encoding="utf-8")
            review_relative = ""
            if audit.status.startswith("unresolved"):
                review_path = staging / "review_sheets" / f"{audit.image_sha256}.jpg"
                if _make_review_sheet(
                    _read_verified_source_image(canonical),
                    audit,
                    review_path,
                    class_count=len(target_names),
                ):
                    review_paths.append(review_path)
                    review_relative = review_path.relative_to(staging).as_posix()
            quarantine_rows.append(
                {
                    "family_component_id": group["group_id"],
                    "image_sha256": audit.image_sha256,
                    "audit_status": audit.status,
                    "audit_reason": audit.reason,
                    "quarantine_reasons": json.dumps(
                        group["quarantine_reasons"], ensure_ascii=False
                    ),
                    "source_splits": json.dumps(group["source_splits"]),
                    "family_ids": json.dumps(sorted(audit.family_ids), ensure_ascii=False),
                    "holdout_force_reasons": json.dumps(
                        group["force_reasons"], ensure_ascii=False
                    ),
                    "candidate_ids": ";".join(candidate_ids),
                    "candidate_box_counts": json.dumps(
                        [len(candidate) for candidate in audit.candidate_sets]
                    ),
                    "quarantine_image": quarantine_path.relative_to(staging).as_posix(),
                    "review_sheet": review_relative,
                    "source_occurrences": _source_references(audit.sources),
                }
            )
    for batch_start in range(0, len(review_paths), 20):
        _make_contact_sheet(
            review_paths[batch_start : batch_start + 20],
            staging
            / "review_sheets"
            / f"contact_sheet_{batch_start // 20 + 1:03d}.jpg",
        )

    preview_rows: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    for index, unit in enumerate(
        select_preview_units(clean_units, target_names, preview_count), start=1
    ):
        try:
            with Image.open(io.BytesIO(_read_verified_source_image(unit.canonical))) as image:
                rendered = draw_boxes(
                    image,
                    unit.boxes,
                    f"{unit.output_split} {unit.image_sha256[:12]}",
                    class_count=len(target_names),
                )
            rendered = ImageOps.contain(
                rendered, (960, 720), Image.Resampling.LANCZOS
            )
        except Exception:
            continue
        preview_path = staging / "previews" / f"bbox_{index:03d}.jpg"
        rendered.save(preview_path, format="JPEG", quality=90, optimize=True)
        preview_paths.append(preview_path)
        preview_rows.append(
            {
                "preview": preview_path.relative_to(staging).as_posix(),
                "image_sha256": unit.image_sha256,
                "split": unit.output_split,
                "classes": ";".join(
                    sorted({box.class_name for box in unit.boxes})
                ),
                "instances": len(unit.boxes),
            }
        )
    _make_contact_sheet(preview_paths, staging / "previews" / "contact_sheet.jpg")

    yaml_document = {
        "path": final_output_dir.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(target_names),
        "names": {index: name for index, name in enumerate(target_names)},
    }
    (staging / "data.yaml").write_text(
        yaml.safe_dump(yaml_document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    mapping_document = {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "source_name": configured_source.name,
        "source_data_yaml_sha256": source_audit["data_yaml_sha256"],
        "annotation_format": configured_source.annotation_format.value,
        "class_mapping_source": configured_source.class_mapping_source,
        "target_profile": source_audit["target_profile"],
        "class_order": source_audit["class_order"],
        "target_names": {
            str(index): name for index, name in enumerate(target_names)
        },
        "output_names": {
            str(index): name for index, name in enumerate(target_names)
        },
        "source_to_target": [
            {
                "source_id": row.source_id,
                "source_name": row.source_name,
                "target_id": row.target_id,
                "target_name": row.target_name,
                "registry_id": row.registry_id,
            }
            for row in mapping_rows
        ],
        "mapping_rule": (
            "source class IDs come only from source data.yaml; normalized canonical names and "
            "explicit aliases come only from the selected brand registry profile; target_id is "
            "the training-dataset class ID while registry_id remains the stable global identity"
        ),
    }
    write_json(staging / "mapping.json", mapping_document)

    write_csv(
        staging / "provenance.csv",
        (
            "output_split",
            "output_image_path",
            "output_label_path",
            "image_sha256",
            "label_sha256",
            "family_component_id",
            "family_ids",
            "forced_holdout",
            "force_reasons",
            "resolution",
            "explicit_background",
            "selected_source_split",
            "selected_source_image",
            "selected_source_label",
            "source_occurrence_count",
            "source_occurrences",
            "width",
            "height",
            "box_count",
            "class_box_counts",
        ),
        provenance_rows,
    )
    write_csv(
        staging / "quarantine_manifest.csv",
        (
            "family_component_id",
            "image_sha256",
            "audit_status",
            "audit_reason",
            "quarantine_reasons",
            "source_splits",
            "family_ids",
            "holdout_force_reasons",
            "candidate_ids",
            "candidate_box_counts",
            "quarantine_image",
            "review_sheet",
            "source_occurrences",
        ),
        quarantine_rows,
    )
    write_csv(
        staging / "annotation_conflicts.csv",
        (
            "image_sha256",
            "status",
            "reason",
            "source_occurrence_count",
            "candidate_set_count",
            "candidate_box_counts",
            "candidate_class_counts",
            "family_ids",
            "sources",
        ),
        conflict_rows,
    )
    write_csv(
        staging / "annotation_candidates.csv",
        (
            "image_sha256",
            "candidate_id",
            "box_count",
            "class_counts",
            "yolo_rows",
            "source_occurrences",
        ),
        candidate_rows,
    )
    write_csv(
        staging / "review_decisions.csv",
        (
            "image_sha256",
            "candidate_options",
            "selected_candidate_id",
            "reviewer",
            "notes",
        ),
        (
            {
                "image_sha256": row["image_sha256"],
                "candidate_options": ";".join(
                    f"candidate_{index:02d}"
                    for index in range(int(row["candidate_set_count"]))
                ),
                "selected_candidate_id": "",
                "reviewer": "",
                "notes": "",
            }
            for row in conflict_rows
            if row["status"] == "unresolved_annotation_conflict"
        ),
    )
    invalid_rows = []
    correction_rows = []
    near_duplicate_rows = []
    all_occurrences_for_reports: list[SourceOccurrence] = []
    for group in list(clean_groups) + list(quarantined_groups):
        for audit in group["audits"]:
            for source in audit.sources:
                all_occurrences_for_reports.append(source)
                for row in source.invalid_rows:
                    invalid_rows.append(
                        {
                            "source_split": source.source_split,
                            "source_image": source.source_image_relative,
                            "source_label": source.source_label_relative,
                            "image_sha256": source.image_sha256,
                            "family_id": source.family_id,
                            "line_number": row.get("line_number", ""),
                            "reason": row.get("reason", ""),
                            "raw_line": row.get("raw_line", ""),
                        }
                    )
                for row in source.boundary_corrections:
                    correction_rows.append(
                        {
                            "source_split": source.source_split,
                            "source_image": source.source_image_relative,
                            "source_label": source.source_label_relative,
                            "image_sha256": source.image_sha256,
                            "family_id": source.family_id,
                            **row,
                        }
                    )
                for row in source.near_duplicate_rows:
                    near_duplicate_rows.append(
                        {
                            "source_split": source.source_split,
                            "source_image": source.source_image_relative,
                            "source_label": source.source_label_relative,
                            "image_sha256": source.image_sha256,
                            "family_id": source.family_id,
                            **row,
                        }
                    )
    write_csv(
        staging / "invalid_annotations.csv",
        (
            "source_split",
            "source_image",
            "source_label",
            "image_sha256",
            "family_id",
            "line_number",
            "reason",
            "raw_line",
        ),
        invalid_rows,
    )
    write_csv(
        staging / "boundary_corrections.csv",
        (
            "source_split",
            "source_image",
            "source_label",
            "image_sha256",
            "family_id",
            "line_number",
            "source_class_id",
            "source_class_name",
            "maximum_overrun",
            "raw_x_min",
            "raw_y_min",
            "raw_x_max",
            "raw_y_max",
            "clipped_x_min",
            "clipped_y_min",
            "clipped_x_max",
            "clipped_y_max",
            "raw_line",
        ),
        correction_rows,
    )
    write_csv(
        staging / "near_duplicate_annotations.csv",
        (
            "source_split",
            "source_image",
            "source_label",
            "image_sha256",
            "family_id",
            "first_line_number",
            "second_line_number",
            "class_id",
            "class_name",
            "iou",
            "threshold",
            "reason",
        ),
        near_duplicate_rows,
    )
    _same_variant_report, same_variant_rows, _suspicious_variants = (
        _same_variant_analysis(all_occurrences_for_reports)
    )
    write_csv(
        staging / "same_variant_conflicts.csv",
        (
            "variant_id",
            "family_ids",
            "source_splits",
            "source_occurrence_count",
            "class_multiplicity_signatures",
            "class_set_differs",
            "reason",
        ),
        same_variant_rows,
    )
    write_csv(
        staging / "counts_by_split_class.csv",
        ("phase", "split", "class_id", "class_name", "images", "instances", "families"),
        _counts_rows(source_audit, clean_units, target_names),
    )
    write_csv(
        staging / "previews" / "preview_manifest.csv",
        ("preview", "image_sha256", "split", "classes", "instances"),
        preview_rows,
    )
    write_json(
        staging / "source_family_manifest.json",
        {
            "normalization": source_audit["family_analysis"]["normalization"],
            "heuristic": True,
            "split_policy": {
                "train_only": "train",
                "train_plus_validation": "val",
                "train_plus_test": "test",
                "validation_plus_test": "quarantine",
                "random_split_used": False,
            },
            "same_variant_policy": _same_variant_report,
            "clean_components": [
                {
                    "group_id": group["group_id"],
                    "family_ids": group["family_ids"],
                    "variant_ids": group["variant_ids"],
                    "source_splits": group["source_splits"],
                    "output_split": group["output_split"],
                    "force_reasons": group["force_reasons"],
                    "image_count": group["image_count"],
                }
                for group in clean_groups
            ],
            "quarantined_components": [
                {
                    "group_id": group["group_id"],
                    "family_ids": group["family_ids"],
                    "variant_ids": group["variant_ids"],
                    "same_variant_conflicts": group["same_variant_conflicts"],
                    "source_splits": group["source_splits"],
                    "quarantine_reasons": group["quarantine_reasons"],
                    "unresolved_hashes": group["unresolved_hashes"],
                    "audit_image_count": group["audit_image_count"],
                }
                for group in quarantined_groups
            ],
        },
    )

    public_audit = dict(source_audit)
    public_audit.pop("_source_snapshot_private", None)
    write_json(staging / "source_audit.json", public_audit)
    _write_source_audit_markdown(staging / "source_audit.md", public_audit)

    fingerprint_payload = _fingerprint_payload(
        mapping_document, provenance_rows, quarantine_rows
    )
    fingerprint = "sha256:" + _sha256_json(fingerprint_payload)
    write_json(
        staging / "dataset_fingerprint.json",
        {
            "schema_version": FINGERPRINT_SCHEMA_VERSION,
            "algorithm": "sha256",
            "fingerprint": fingerprint,
            "canonical_payload": fingerprint_payload,
        },
    )
    return {
        "provenance_rows": len(provenance_rows),
        "quarantine_rows": len(quarantine_rows),
        "annotation_conflicts": len(conflict_rows),
        "invalid_annotation_rows": len(invalid_rows),
        "boundary_corrections": len(correction_rows),
        "near_duplicate_annotation_pairs": len(near_duplicate_rows),
        "same_variant_conflicts": len(same_variant_rows),
        "bbox_previews": len(preview_rows),
        "review_sheets": len(review_paths),
        "dataset_fingerprint": fingerprint,
    }


def _mapping_names(mapping_document: Mapping[str, Any]) -> tuple[str, ...]:
    raw = mapping_document.get("output_names")
    if not isinstance(raw, Mapping):
        raise ValueError("Prepared mapping.json has no output_names object")
    try:
        keyed = {int(key): str(value) for key, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("Prepared output_names keys must be integer-like") from exc
    if sorted(keyed) != list(range(len(keyed))):
        raise ValueError("Prepared output_names IDs must be contiguous from zero")
    return tuple(keyed[index] for index in range(len(keyed)))


def validate_prepared_dataset(
    dataset_dir: str | Path,
    *,
    expected_final_root: str | Path | None = None,
) -> dict[str, Any]:
    """Re-read a prepared dataset and prove its mapping/split/hash invariants."""

    root = Path(dataset_dir).resolve()
    final_root = (
        Path(expected_final_root).resolve()
        if expected_final_root is not None
        else root
    )
    required = (
        root / "data.yaml",
        root / "mapping.json",
        root / "provenance.csv",
        root / "quarantine_manifest.csv",
        root / "annotation_conflicts.csv",
        root / "source_family_manifest.json",
        root / "source_audit.json",
        root / "dataset_fingerprint.json",
        root / "counts_by_split_class.csv",
        root / "invalid_annotations.csv",
        root / "boundary_corrections.csv",
        root / "near_duplicate_annotations.csv",
        root / "same_variant_conflicts.csv",
        root / "source_audit.md",
        root / "previews" / "preview_manifest.csv",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Prepared detector metadata is incomplete: {missing}")

    data_document = yaml.safe_load(required[0].read_text(encoding="utf-8")) or {}
    if not isinstance(data_document, Mapping):
        raise ValueError("Prepared data.yaml must contain a mapping")
    data_path = Path(str(data_document.get("path", ""))).resolve()
    if data_path != final_root:
        raise ValueError(
            f"Prepared data.yaml path differs from final output root: {data_path} != {final_root}"
        )
    if not data_path.is_absolute():
        raise ValueError("Prepared data.yaml path must be absolute")
    names = normalize_yolo_names(data_document.get("names"))
    if int(data_document.get("nc", -1)) != len(names):
        raise ValueError("Prepared data.yaml nc differs from names")
    mapping_document = json.loads(required[1].read_text(encoding="utf-8"))
    if _mapping_names(mapping_document) != names:
        raise ValueError("Prepared mapping.json class order differs from data.yaml")
    if mapping_document.get("class_order") not in CLASS_ORDER_CHOICES:
        raise ValueError("Prepared mapping.json has an unsupported class_order")
    source_mapping_rows = mapping_document.get("source_to_target")
    if not isinstance(source_mapping_rows, list) or len(source_mapping_rows) != len(names):
        raise ValueError("Prepared mapping.json source_to_target is incomplete")
    stable_registry_ids: dict[str, int] = {}
    for item in source_mapping_rows:
        target_id = int(item["target_id"])
        target_name = str(item["target_name"])
        registry_id = int(item["registry_id"])
        if not 0 <= target_id < len(names) or names[target_id] != target_name:
            raise ValueError("Prepared source_to_target target ID/name is inconsistent")
        key = str(target_id)
        if key in stable_registry_ids:
            raise ValueError("Prepared source_to_target duplicates a target ID")
        stable_registry_ids[key] = registry_id
    if len(set(stable_registry_ids.values())) != len(names):
        raise ValueError("Prepared source_to_target duplicates a stable registry ID")

    provenance = _read_csv(required[2])
    provenance_by_image = {
        row["output_image_path"]: row for row in provenance
    }
    if len(provenance_by_image) != len(provenance):
        raise ValueError("Prepared provenance contains duplicate output image paths")
    quarantine = _read_csv(required[3])
    conflicts = _read_csv(required[4])
    quarantined_hashes = {row["image_sha256"] for row in quarantine}
    output_hashes: set[str] = set()
    hash_splits: dict[str, set[str]] = defaultdict(set)
    family_splits: dict[str, set[str]] = defaultdict(set)
    holdout_violations: list[dict[str, str]] = []
    split_stats: dict[str, Any] = {}

    for split in OUTPUT_SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Prepared split directories are missing: {split}")
        images = _image_files(image_dir)
        labels = _label_files(label_dir)
        image_stems = _unique_stem_map(images, "prepared image")
        label_stems = _unique_stem_map(labels, "prepared label")
        if set(image_stems) != set(label_stems):
            raise ValueError(f"Prepared image/label stems differ in split {split}")
        class_instances: Counter[str] = Counter()
        class_images: Counter[str] = Counter()
        class_families: dict[str, set[str]] = defaultdict(set)
        explicit_backgrounds = 0
        split_hashes: set[str] = set()
        for image_path in images:
            if len(image_path.name) > 64:
                raise ValueError(f"Prepared filename is not short: {image_path.name}")
            relative_image = image_path.relative_to(root).as_posix()
            row = provenance_by_image.get(relative_image)
            if row is None:
                raise ValueError(f"Prepared image is absent from provenance: {relative_image}")
            digest = sha256_file(image_path)
            if digest != row["image_sha256"]:
                raise ValueError(f"Prepared image hash differs from provenance: {image_path}")
            if digest in split_hashes:
                raise ValueError(f"Duplicate exact image bytes within split {split}: {digest}")
            split_hashes.add(digest)
            output_hashes.add(digest)
            hash_splits[digest].add(split)
            family_ids = [str(value).casefold() for value in json.loads(row["family_ids"])]
            for family in family_ids:
                family_splits[family].add(split)
            forced = row.get("forced_holdout", "")
            if forced and forced != split:
                holdout_violations.append(row)

            label_path = label_dir / f"{image_path.stem}.txt"
            label_hash = sha256_file(label_path)
            if label_hash != row["label_sha256"]:
                raise ValueError(f"Prepared label hash differs from provenance: {label_path}")
            label_names: set[str] = set()
            rows = [
                line for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
            ]
            for line_number, line in enumerate(rows, start=1):
                try:
                    parsed = parse_yolo_annotation(line, names, AnnotationFormat.YOLO_BBOX)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid prepared detection row {label_path}:{line_number}: {exc}"
                    ) from exc
                if parsed.box.was_clipped:
                    raise ValueError(
                        f"Prepared box exceeds normalized bounds: {label_path}:{line_number}"
                    )
                class_name = names[parsed.class_id]
                class_instances[class_name] += 1
                label_names.add(class_name)
            if not rows:
                explicit_backgrounds += 1
            component = row["family_component_id"]
            for class_name in label_names:
                class_images[class_name] += 1
                class_families[class_name].add(component)
        split_stats[split] = {
            "images": len(images),
            "labels": len(labels),
            "instances": sum(class_instances.values()),
            "explicit_background_images": explicit_backgrounds,
            "per_class": {
                name: {
                    "images": class_images[name],
                    "instances": class_instances[name],
                    "families": len(class_families[name]),
                }
                for name in names
            },
        }

    if len(provenance) != sum(values["images"] for values in split_stats.values()):
        raise ValueError("Prepared provenance row count differs from image count")
    hash_overlap = {
        f"{left}-{right}": len(
            {
                digest
                for digest, splits in hash_splits.items()
                if {left, right} <= splits
            }
        )
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    family_overlap = {
        family: sorted(splits)
        for family, splits in family_splits.items()
        if len(splits) > 1
    }
    if any(hash_overlap.values()):
        raise ValueError(f"Exact image hashes cross prepared splits: {hash_overlap}")
    if family_overlap:
        raise ValueError(
            f"Normalized source families cross prepared splits: {dict(list(family_overlap.items())[:10])}"
        )
    if holdout_violations:
        raise ValueError(f"Prepared holdout locks were violated: {holdout_violations[:5]}")
    overlap_with_quarantine = output_hashes & quarantined_hashes
    if overlap_with_quarantine:
        raise ValueError(
            f"Quarantined images leaked into prepared splits: {sorted(overlap_with_quarantine)[:10]}"
        )

    source_audit = json.loads(required[6].read_text(encoding="utf-8"))
    all_classes = all(
        split_stats[split]["per_class"][name]["images"] > 0
        for split in OUTPUT_SPLITS
        for name in names
    )
    fingerprint_document = json.loads(required[7].read_text(encoding="utf-8"))
    expected_payload = _fingerprint_payload(mapping_document, provenance, quarantine)
    expected_fingerprint = "sha256:" + _sha256_json(expected_payload)
    if fingerprint_document.get("canonical_payload") != expected_payload:
        raise ValueError("Prepared dataset fingerprint payload differs from current manifests")
    if fingerprint_document.get("fingerprint") != expected_fingerprint:
        raise ValueError("Prepared dataset fingerprint differs from current manifests")

    source_ready = (
        not conflicts
        and not quarantine
        and not source_audit.get("orphan_labels")
    )
    clean_subset_ready = all_classes and not family_overlap and not any(hash_overlap.values())
    return {
        "schema_version": PREPARATION_SCHEMA_VERSION,
        "dataset_version": final_root.name,
        "status": "structural_checks_passed",
        "training_ready": source_ready and clean_subset_ready,
        "source_training_ready": source_ready,
        "clean_subset_ready": clean_subset_ready,
        "mapping_verified": {str(index): name for index, name in enumerate(names)},
        "class_order": mapping_document["class_order"],
        "stable_registry_ids": stable_registry_ids,
        "target_class_count": len(names),
        "ordinary_yolo_detection_rows_only": True,
        "explicit_background_labels_supported": True,
        "splits": split_stats,
        "exact_image_cross_split_overlap": hash_overlap,
        "cross_split_family_overlap": 0,
        "known_holdout_violations": 0,
        "quarantined_hashes_in_splits": 0,
        "unresolved_annotation_hashes": len(
            {row["image_sha256"] for row in conflicts}
        ),
        "quarantined_image_hashes": len(quarantined_hashes),
        "orphan_source_labels": len(source_audit.get("orphan_labels", [])),
        "all_target_classes_present_in_every_split": all_classes,
        # Backward-compatibility for the pre-existing generic preflight policy.
        # The value refers to all selected profile classes, not literally nine.
        "all_nine_brands_present_in_every_split": all_classes,
        "dataset_fingerprint": expected_fingerprint,
        "augmentation_audit": {
            "export_metadata_statements": [
                statement
                for item in source_audit.get("roboflow_metadata", [])
                for statement in item.get("augmentation_statements", [])
            ],
            "filenames_with_explicit_aug_marker": int(
                source_audit.get("offline_augmentation_evidence", {}).get(
                    "filenames_with_explicit_aug_marker", 0
                )
            ),
            "upstream_family_variants_evidenced": bool(
                source_audit.get("offline_augmentation_evidence", {}).get(
                    "filenames_with_explicit_aug_marker", 0
                )
            ),
            "offline_mosaic_status": "not_confirmed",
            "interpretation": source_audit.get(
                "offline_augmentation_evidence", {}
            ).get("interpretation"),
        },
        "source_unchanged_verified": bool(
            source_audit.get("source_integrity", {}).get("source_unchanged_verified")
        ),
        "annotation_completeness_visually_proven": False,
        "test_split_evaluated": False,
        "training_started": False,
    }


def _write_dataset_readme(
    path: Path,
    checks: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    mapping = checks["mapping_verified"]
    lines = [
        f"# Prepared {len(mapping)}-class YOLO detection dataset",
        "",
        "Class order: "
        + ", ".join(f"{index} {mapping[str(index)]}" for index in range(len(mapping)))
        + ".",
        "",
        f"Dataset fingerprint: `{checks['dataset_fingerprint']}`.",
        f"Source fully ready: {str(checks['source_training_ready']).lower()}.",
        f"Clean subset structurally ready: {str(checks['clean_subset_ready']).lower()}.",
        f"Quarantined image hashes: {checks['quarantined_image_hashes']}.",
        f"Unresolved annotation hashes: {checks['unresolved_annotation_hashes']}.",
        "",
        "The source export is unchanged. Exact-content annotation conflicts, invalid rows, "
        "and validation/test family conflicts are excluded as complete family components and "
        "remain auditable under `quarantine/` and the review manifests.",
        "",
        "## Prepared counts",
        "",
        "| Split | Images | Instances | Backgrounds |",
        "|---|---:|---:|---:|",
    ]
    for split in OUTPUT_SPLITS:
        values = checks["splits"][split]
        lines.append(
            f"| {split} | {values['images']} | {values['instances']} | "
            f"{values['explicit_background_images']} |"
        )
    lines.extend(
        [
            "",
            "## Manual review",
            "",
            "1. Inspect `review_sheets/`, `annotation_conflicts.csv`, and `annotation_candidates.csv`.",
            "2. Fill `selected_candidate_id` in a copy of `review_decisions.csv`.",
            "3. Re-run preparation into a new output directory with `--decisions`.",
            "",
            "`previews/` contains deterministic source-bbox previews only. Online training-transform "
            "previews are a separate training-config artifact.",
            "",
            f"Historical holdout sources checked: {len(report['historical_holdouts']['datasets'])}.",
            "No training, model inference, tuning, or test-split evaluation was performed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_extracted_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    brand_profile: str,
    class_order: str = "registry",
    brand_registry_path: str | Path | None = None,
    source_registry_path: str | Path | None = None,
    historical_datasets: Sequence[str | Path] | None = None,
    decisions_path: str | Path | None = None,
    preview_count: int = 45,
) -> dict[str, Any]:
    """Prepare a new derived dataset without mutating or overwriting source/artifacts."""

    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output already exists; refusing to overwrite: {output_root}"
        )
    if source_root == output_root or _path_within(output_root, source_root):
        raise ValueError("Derived output must be separate from and outside the source tree")
    if preview_count < 0:
        raise ValueError("preview_count must be non-negative")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    source_audit, occurrences, target_names, mapping_rows, configured_source = (
        audit_extracted_source(
            source_root,
            brand_profile=brand_profile,
            class_order=class_order,
            brand_registry_path=brand_registry_path,
            source_registry_path=source_registry_path,
        )
    )
    source_snapshot = dict(source_audit["_source_snapshot_private"])
    decisions = load_review_decisions(decisions_path)
    units, audits, conflicts, candidates, duplicate_report = (
        reconcile_exact_duplicates(occurrences, decisions)
    )

    if historical_datasets is None:
        historical_datasets = (
            PROJECT_ROOT / "data" / "brand_9brands_yolo_reviewed",
        )
    hash_locks, family_locks, historical_report = load_historical_holdout_locks(
        historical_datasets,
        target_names=target_names,
    )
    clean_groups, quarantined_groups, family_report = build_family_components(
        units,
        audits,
        hash_locks,
        family_locks,
        target_names=target_names,
    )
    clean_hashes = {
        unit.image_sha256 for group in clean_groups for unit in group["units"]
    }
    clean_units = sorted(
        (unit for unit in units if unit.image_sha256 in clean_hashes),
        key=lambda unit: unit.image_sha256,
    )

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.staging-", dir=output_root.parent
        )
    ).resolve()
    try:
        artifact_report = write_prepared_dataset(
            staging,
            output_root,
            source_audit=source_audit,
            target_names=target_names,
            mapping_rows=mapping_rows,
            configured_source=configured_source,
            clean_units=clean_units,
            clean_groups=clean_groups,
            quarantined_groups=quarantined_groups,
            conflict_rows=conflicts,
            candidate_rows=candidates,
            preview_count=preview_count,
        )
        integrity = _verify_source_snapshot(source_root, source_snapshot)
        source_audit["source_integrity"] = integrity
        public_audit = dict(source_audit)
        public_audit.pop("_source_snapshot_private", None)
        write_json(staging / "source_audit.json", public_audit)
        _write_source_audit_markdown(staging / "source_audit.md", public_audit)

        checks = validate_prepared_dataset(
            staging, expected_final_root=output_root
        )
        write_json(staging / "dataset_checks.json", checks)
        report = {
            "schema_version": PREPARATION_SCHEMA_VERSION,
            "status": (
                "prepared_with_quarantine"
                if quarantined_groups
                else "prepared"
            ),
            "source": {
                "name": configured_source.name,
                "path": str(source_root),
                "annotation_format": configured_source.annotation_format.value,
                "data_yaml_sha256": source_audit["data_yaml_sha256"],
                "source_unchanged_verified": True,
            },
            "target_profile": brand_profile,
            "class_order": class_order,
            "target_names": list(target_names),
            "source_to_target_mapping": [
                {
                    "source_id": row.source_id,
                    "source_name": row.source_name,
                    "target_id": row.target_id,
                    "target_name": row.target_name,
                    "registry_id": row.registry_id,
                }
                for row in mapping_rows
            ],
            "exact_duplicate_reconciliation": duplicate_report,
            "historical_holdouts": historical_report,
            "family_components": family_report,
            "artifacts": artifact_report,
            "checks": checks,
            "training_ready": checks["training_ready"],
            "clean_subset_ready": checks["clean_subset_ready"],
            "test_evaluation_performed": False,
            "training_started": False,
            "model_inference_performed": False,
        }
        write_json(staging / "dataset_report.json", report)
        _write_dataset_readme(staging / "README.md", checks, report)
        staging.rename(output_root)
        return report
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
