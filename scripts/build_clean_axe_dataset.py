"""Build a leakage-free five-SKU AXE COCO dataset directly from ZIP exports.

The source archives are opened read-only with :mod:`zipfile`. Roboflow's original
train/valid/test assignments are treated as metadata only; normalized source
families are grouped and assigned to one new split so augmented variants cannot
cross split boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catalog.profiles import load_product_profile


SPLITS = ("train", "valid", "test")
ANNOTATION_NAME = "_annotations.coco.json"
EXPECTED_SKUS = load_product_profile("axe5").legacy_class_names

_ROBOFLOW_HASH_RE = re.compile(r"\.rf\.[^.]+$", flags=re.IGNORECASE)
_JPG_MARKER_RE = re.compile(r"_jpg$", flags=re.IGNORECASE)
_AUGMENTATION_RE = re.compile(r"_aug\d+$", flags=re.IGNORECASE)
_UNSAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def normalize_source_family(filename_or_path: str | Path) -> str:
    """Return a conservative, case-insensitive Roboflow source-family ID.

    Only a final ``.rf.<hash>``, ``_jpg``, and ``_augN`` marker are removed, in
    that order. Similar natural names such as ``product_august`` are untouched.
    """

    normalized = str(filename_or_path).replace("\\", "/")
    stem = PurePosixPath(normalized).stem
    stem = _ROBOFLOW_HASH_RE.sub("", stem)
    stem = _JPG_MARKER_RE.sub("", stem)
    stem = _AUGMENTATION_RE.sub("", stem)
    return stem.casefold()


def sku_slug(name: str) -> str:
    """Create a stable filename-safe SKU slug without losing readable casing."""

    value = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
    if not value:
        raise ValueError(f"Cannot derive a filename slug from SKU name {name!r}")
    return value


def _safe_source_basename(file_name: str) -> str:
    name = PurePosixPath(str(file_name).replace("\\", "/")).name
    name = _UNSAFE_FILENAME_RE.sub("_", name).strip(" .")
    if not name:
        name = "image"
    # Leave ample room for the SKU/ZIP/ID prefix on Windows.
    suffix = Path(name).suffix[:16]
    stem = Path(name).stem[:96]
    return f"{stem}{suffix}"


def _is_excluded_category(category: dict[str, Any]) -> bool:
    # Roboflow's real SKU categories legitimately have supercategory="Stock".
    # The excluded supercategory sentinel is the category named "Stock" itself.
    return str(category.get("name", "")).strip().casefold() in {"stock", "0"}


def _zip_fingerprint(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _resolve_image_member(
    members: set[str], split: str, file_name: str, archive_name: str
) -> str:
    normalized = str(file_name).replace("\\", "/").lstrip("/")
    parts = PurePosixPath(normalized).parts
    if not normalized or ".." in parts:
        raise ValueError(f"Unsafe image path {file_name!r} in {archive_name}")
    candidates = [normalized]
    if not normalized.startswith(f"{split}/"):
        candidates.insert(0, f"{split}/{normalized}")
    for candidate in candidates:
        if candidate in members:
            return candidate
    raise FileNotFoundError(
        f"Image {file_name!r} referenced by {split}/{ANNOTATION_NAME} "
        f"is missing from {archive_name}"
    )


def _read_coco_member(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    try:
        payload = json.loads(archive.read(member).decode("utf-8-sig"))
    except KeyError as error:
        raise FileNotFoundError(f"Missing required ZIP member: {member}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid COCO JSON in {archive.filename}: {member}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"COCO root must be an object in {archive.filename}: {member}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"COCO field {key!r} must be a list in {archive.filename}: {member}")
    return payload


def read_source_archives(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read and validate all five source archives without extracting them."""

    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"AXE ZIP directory does not exist: {input_dir}")
    zip_paths = sorted(input_dir.glob("*.zip"), key=lambda path: path.name.casefold())
    if len(zip_paths) != len(EXPECTED_SKUS):
        raise ValueError(
            f"Expected exactly {len(EXPECTED_SKUS)} ZIP files in {input_dir}, found {len(zip_paths)}"
        )

    records: list[dict[str, Any]] = []
    archives: list[dict[str, Any]] = []
    detected_skus: list[str] = []

    for zip_index, zip_path in enumerate(zip_paths, start=1):
        fingerprint = _zip_fingerprint(zip_path)
        archive_skus: set[str] = set()
        archive_record_start = len(records)
        with zipfile.ZipFile(zip_path, mode="r") as archive:
            members = set(archive.namelist())
            for source_split in SPLITS:
                annotation_member = f"{source_split}/{ANNOTATION_NAME}"
                coco = _read_coco_member(archive, annotation_member)
                valid_categories = [
                    category for category in coco["categories"] if not _is_excluded_category(category)
                ]
                if len(valid_categories) != 1:
                    names = [str(category.get("name", "")) for category in valid_categories]
                    raise ValueError(
                        f"Expected one valid SKU category in {zip_path.name}:{annotation_member}, "
                        f"found {len(valid_categories)}: {names}"
                    )
                valid_category = valid_categories[0]
                sku_name = str(valid_category.get("name", "")).strip()
                if not sku_name:
                    raise ValueError(f"Empty SKU category name in {zip_path.name}:{annotation_member}")
                archive_skus.add(sku_name)

                category_ids = {category.get("id") for category in coco["categories"]}
                valid_category_id = valid_category.get("id")
                image_by_id: dict[Any, dict[str, Any]] = {}
                for image in coco["images"]:
                    if not isinstance(image, dict) or "id" not in image or "file_name" not in image:
                        raise ValueError(
                            f"Each image needs id and file_name in {zip_path.name}:{annotation_member}"
                        )
                    source_image_id = image["id"]
                    if source_image_id in image_by_id:
                        raise ValueError(
                            f"Duplicate image ID {source_image_id!r} in {zip_path.name}:{annotation_member}"
                        )
                    image_by_id[source_image_id] = image

                valid_annotations: dict[Any, list[dict[str, Any]]] = defaultdict(list)
                source_annotation_ids: set[Any] = set()
                for annotation in coco["annotations"]:
                    if not isinstance(annotation, dict) or "id" not in annotation:
                        raise ValueError(
                            f"Each annotation needs an ID in {zip_path.name}:{annotation_member}"
                        )
                    source_annotation_id = annotation["id"]
                    if source_annotation_id in source_annotation_ids:
                        raise ValueError(
                            f"Duplicate annotation ID {source_annotation_id!r} in "
                            f"{zip_path.name}:{annotation_member}"
                        )
                    source_annotation_ids.add(source_annotation_id)
                    source_image_id = annotation.get("image_id")
                    if source_image_id not in image_by_id:
                        raise ValueError(
                            f"Annotation {source_annotation_id!r} references missing image "
                            f"{source_image_id!r} in {zip_path.name}:{annotation_member}"
                        )
                    source_category_id = annotation.get("category_id")
                    if source_category_id not in category_ids:
                        raise ValueError(
                            f"Annotation {source_annotation_id!r} references unknown category "
                            f"{source_category_id!r} in {zip_path.name}:{annotation_member}"
                        )
                    if source_category_id == valid_category_id:
                        valid_annotations[source_image_id].append(annotation)

                for image in coco["images"]:
                    source_member = _resolve_image_member(
                        members, source_split, str(image["file_name"]), zip_path.name
                    )
                    records.append(
                        {
                            "zip_index": zip_index,
                            "zip_path": zip_path,
                            "source_split": source_split,
                            "source_member": source_member,
                            "source_file_name": str(image["file_name"]),
                            "source_image_id": image["id"],
                            "source_image": image,
                            "source_annotations": valid_annotations.get(image["id"], []),
                            "sku": sku_name,
                            "family_id": normalize_source_family(str(image["file_name"])),
                        }
                    )

        if len(archive_skus) != 1:
            raise ValueError(
                f"All splits in {zip_path.name} must describe one SKU; found {sorted(archive_skus)}"
            )
        sku_name = next(iter(archive_skus))
        detected_skus.append(sku_name)
        archives.append(
            {
                "zip_index": zip_index,
                "file_name": zip_path.name,
                "path": str(zip_path),
                "sku": sku_name,
                "images": len(records) - archive_record_start,
                "fingerprint": fingerprint,
            }
        )

    if len(set(detected_skus)) != len(EXPECTED_SKUS) or set(detected_skus) != set(EXPECTED_SKUS):
        missing = sorted(set(EXPECTED_SKUS) - set(detected_skus))
        unexpected = sorted(set(detected_skus) - set(EXPECTED_SKUS))
        raise ValueError(
            "Detected SKU set does not match the required five classes. "
            f"Missing={missing}; unexpected={unexpected}; detected={detected_skus}"
        )
    return records, archives


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> dict[str, float]:
    ratios = {"train": float(train_ratio), "valid": float(val_ratio), "test": float(test_ratio)}
    if any(not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError(f"All split ratios must be finite and greater than zero: {ratios}")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(ratios.values()):.12g}")
    return ratios


def _split_objective(
    counts: dict[str, Counter[str]],
    targets: dict[str, dict[str, float]],
    sku_totals: Counter[str],
) -> float:
    return sum(
        ((counts[split][sku] - targets[split][sku]) / sku_totals[sku]) ** 2
        for split in SPLITS
        for sku in EXPECTED_SKUS
    )


def assign_grouped_stratified_splits(
    records: list[dict[str, Any]], ratios: dict[str, float], seed: int
) -> tuple[dict[str, str], dict[str, Counter[str]]]:
    """Assign whole families while minimizing per-SKU deviation from target ratios."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["family_id"]].append(record)
    sku_totals = Counter(record["sku"] for record in records)
    for sku in EXPECTED_SKUS:
        available_groups = sum(any(item["sku"] == sku for item in group) for group in groups.values())
        if available_groups < len(SPLITS):
            raise ValueError(
                f"SKU {sku!r} has only {available_groups} independent families; "
                f"at least {len(SPLITS)} are required for class coverage"
            )

    group_counts = {
        family_id: Counter(record["sku"] for record in group)
        for family_id, group in groups.items()
    }
    targets = {
        split: {sku: sku_totals[sku] * ratios[split] for sku in EXPECTED_SKUS}
        for split in SPLITS
    }
    counts = {split: Counter() for split in SPLITS}
    assignments: dict[str, str] = {}
    rng = random.Random(int(seed))
    family_ties = {family_id: rng.random() for family_id in groups}
    split_ties = {
        (family_id, split): rng.random() for family_id in groups for split in SPLITS
    }

    ordered_families = sorted(
        groups,
        key=lambda family_id: (
            -sum(group_counts[family_id].values()),
            -len(group_counts[family_id]),
            family_ties[family_id],
            family_id,
        ),
    )
    for family_id in ordered_families:
        sku_counts = group_counts[family_id]
        best: tuple[float, float, str] | None = None
        for split in SPLITS:
            counts[split].update(sku_counts)
            candidate = (
                _split_objective(counts, targets, sku_totals),
                split_ties[(family_id, split)],
                split,
            )
            counts[split].subtract(sku_counts)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        selected_split = best[2]
        assignments[family_id] = selected_split
        counts[selected_split].update(sku_counts)

    # A repair pass makes class coverage an explicit invariant, including for
    # unusual custom ratios where the greedy objective could leave a tiny split empty.
    while True:
        missing = [
            (split, sku)
            for split in SPLITS
            for sku in EXPECTED_SKUS
            if counts[split][sku] == 0
        ]
        if not missing:
            break
        target_split, missing_sku = missing[0]
        candidates: list[tuple[float, float, str, str]] = []
        for family_id, old_split in assignments.items():
            sku_counts = group_counts[family_id]
            if old_split == target_split or sku_counts[missing_sku] == 0:
                continue
            if any(counts[old_split][sku] - amount <= 0 for sku, amount in sku_counts.items()):
                continue
            counts[old_split].subtract(sku_counts)
            counts[target_split].update(sku_counts)
            cost = _split_objective(counts, targets, sku_totals)
            counts[target_split].subtract(sku_counts)
            counts[old_split].update(sku_counts)
            candidates.append((cost, split_ties[(family_id, target_split)], family_id, old_split))
        if not candidates:
            raise ValueError(f"Cannot place SKU {missing_sku!r} in split {target_split!r}")
        _, _, family_id, old_split = min(candidates)
        sku_counts = group_counts[family_id]
        counts[old_split].subtract(sku_counts)
        counts[target_split].update(sku_counts)
        assignments[family_id] = target_split

    # Deterministic local improvement tightens stratification without breaking coverage.
    for _ in range(len(groups)):
        improved = False
        for family_id in sorted(groups, key=lambda value: (family_ties[value], value)):
            old_split = assignments[family_id]
            sku_counts = group_counts[family_id]
            current_cost = _split_objective(counts, targets, sku_totals)
            best_cost, best_split = current_cost, old_split
            for new_split in SPLITS:
                if new_split == old_split:
                    continue
                if any(counts[old_split][sku] - amount <= 0 for sku, amount in sku_counts.items()):
                    continue
                counts[old_split].subtract(sku_counts)
                counts[new_split].update(sku_counts)
                candidate_cost = _split_objective(counts, targets, sku_totals)
                counts[new_split].subtract(sku_counts)
                counts[old_split].update(sku_counts)
                if candidate_cost < best_cost - 1e-15:
                    best_cost, best_split = candidate_cost, new_split
            if best_split != old_split:
                counts[old_split].subtract(sku_counts)
                counts[best_split].update(sku_counts)
                assignments[family_id] = best_split
                improved = True
        if not improved:
            break

    for record in records:
        record["output_split"] = assignments[record["family_id"]]
    return assignments, counts


def _prepare_global_ids(records: list[dict[str, Any]]) -> dict[str, int]:
    category_ids = {sku: index for index, sku in enumerate(EXPECTED_SKUS, start=1)}
    destination_names: set[str] = set()
    image_ids: set[int] = set()
    annotation_ids: set[int] = set()
    next_annotation_id = 1

    for output_image_id, record in enumerate(records, start=1):
        prefix = f"{sku_slug(record['sku'])}__zip{record['zip_index']:02d}"
        destination_name = (
            f"{prefix}__{record['source_split']}__img{output_image_id:06d}__"
            f"{_safe_source_basename(record['source_file_name'])}"
        )
        if destination_name in destination_names:
            raise AssertionError(f"Duplicate destination filename: {destination_name}")
        if output_image_id in image_ids:
            raise AssertionError(f"Duplicate global image ID: {output_image_id}")
        destination_names.add(destination_name)
        image_ids.add(output_image_id)
        record["destination_file_name"] = destination_name
        record["output_image_id"] = output_image_id

        output_annotations: list[dict[str, Any]] = []
        for source_annotation in record["source_annotations"]:
            output_annotation = dict(source_annotation)
            output_annotation["id"] = next_annotation_id
            output_annotation["image_id"] = output_image_id
            output_annotation["category_id"] = category_ids[record["sku"]]
            if next_annotation_id in annotation_ids:
                raise AssertionError(f"Duplicate global annotation ID: {next_annotation_id}")
            annotation_ids.add(next_annotation_id)
            output_annotations.append(output_annotation)
            next_annotation_id += 1
        record["output_annotations"] = output_annotations

    assert len(destination_names) == len(records), "Destination filenames are not globally unique"
    assert len(image_ids) == len(records), "Global image IDs are not unique"
    assert len(annotation_ids) == next_annotation_id - 1, "Global annotation IDs are not unique"
    return category_ids


def _categories(category_ids: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"id": category_ids[sku], "name": sku, "supercategory": "Stock"}
        for sku in EXPECTED_SKUS
    ]


def _build_coco_documents(
    records: list[dict[str, Any]], category_ids: dict[str, int]
) -> dict[str, dict[str, Any]]:
    documents = {
        split: {
            "info": {
                "description": "Leakage-free grouped AXE five-SKU COCO dataset",
                "split": split,
            },
            "images": [],
            "annotations": [],
            "categories": _categories(category_ids),
        }
        for split in SPLITS
    }
    for record in records:
        split = record["output_split"]
        output_image = dict(record["source_image"])
        output_image["id"] = record["output_image_id"]
        output_image["file_name"] = record["destination_file_name"]
        documents[split]["images"].append(output_image)
        documents[split]["annotations"].extend(record["output_annotations"])
    return documents


def _source_summary(records: list[dict[str, Any]], archives: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["family_id"]].append(record)
    per_sku: dict[str, dict[str, int]] = {}
    for sku in EXPECTED_SKUS:
        sku_records = [record for record in records if record["sku"] == sku]
        sku_family_source_splits: dict[str, set[str]] = defaultdict(set)
        for record in sku_records:
            sku_family_source_splits[record["family_id"]].add(record["source_split"])
        per_sku[sku] = {
            "images": len(sku_records),
            "annotations": sum(len(record["source_annotations"]) for record in sku_records),
            "families": len({record["family_id"] for record in sku_records}),
            "original_cross_split_families": sum(
                len(source_splits) > 1 for source_splits in sku_family_source_splits.values()
            ),
        }
    original_split_families: dict[str, set[str]] = defaultdict(set)
    for record in records:
        original_split_families[record["family_id"]].add(record["source_split"])
    return {
        "zip_files": len(archives),
        "images": len(records),
        "annotations": sum(len(record["source_annotations"]) for record in records),
        "families": len(groups),
        "multi_occurrence_families": sum(len(group) > 1 for group in groups.values()),
        "cross_zip_families": sum(
            len({record["zip_index"] for record in group}) > 1 for group in groups.values()
        ),
        "original_cross_split_families": sum(
            len(source_splits) > 1 for source_splits in original_split_families.values()
        ),
        "per_sku": per_sku,
    }


def _split_summary(records: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    total_images = len(records)
    for split in SPLITS:
        split_records = [record for record in records if record["output_split"] == split]
        per_sku: dict[str, dict[str, int]] = {}
        for sku in EXPECTED_SKUS:
            sku_records = [record for record in split_records if record["sku"] == sku]
            per_sku[sku] = {
                "images": len(sku_records),
                "annotations": sum(len(record["source_annotations"]) for record in sku_records),
                "families": len({record["family_id"] for record in sku_records}),
            }
        summary[split] = {
            "target_ratio": ratios[split],
            "actual_ratio": len(split_records) / total_images if total_images else 0.0,
            "images": len(split_records),
            "annotations": sum(len(record["source_annotations"]) for record in split_records),
            "families": len({record["family_id"] for record in split_records}),
            "per_sku": per_sku,
        }
    return summary


def _assert_output_invariants(
    records: list[dict[str, Any]], category_ids: dict[str, int]
) -> dict[str, Any]:
    family_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        family_splits[record["family_id"]].add(record["output_split"])
    leaking = sorted(family_id for family_id, splits in family_splits.items() if len(splits) > 1)
    assert not leaking, f"Cross-split source-family leakage detected: {leaking[:10]}"

    destination_names = [record["destination_file_name"] for record in records]
    image_ids = [record["output_image_id"] for record in records]
    annotation_ids = [
        annotation["id"] for record in records for annotation in record["output_annotations"]
    ]
    assert len(destination_names) == len(set(destination_names)), "Destination filenames are not unique"
    assert len(image_ids) == len(set(image_ids)), "Image IDs are not unique"
    assert len(annotation_ids) == len(set(annotation_ids)), "Annotation IDs are not unique"
    assert len(category_ids.values()) == len(set(category_ids.values())), "Category IDs are not unique"

    missing_classes = [
        (split, sku)
        for split in SPLITS
        for sku in EXPECTED_SKUS
        if not any(record["output_split"] == split and record["sku"] == sku for record in records)
    ]
    assert not missing_classes, f"Missing SKU coverage in output splits: {missing_classes}"
    return {
        "cross_split_family_leakage": len(leaking),
        "destination_filenames_unique": True,
        "image_ids_globally_unique": True,
        "annotation_ids_globally_unique": True,
        "category_ids_globally_unique": True,
        "all_five_skus_in_every_split": True,
    }


def _family_manifest(
    records: list[dict[str, Any]], assignments: dict[str, str], seed: int, ratios: dict[str, float]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["family_id"]].append(record)
    families = []
    for family_id in sorted(groups):
        group = groups[family_id]
        families.append(
            {
                "family_id": family_id,
                "output_split": assignments[family_id],
                "occurrence_count": len(group),
                "skus": sorted({record["sku"] for record in group}),
                "zip_indices": sorted({record["zip_index"] for record in group}),
                "source_splits": sorted({record["source_split"] for record in group}),
                "images": [
                    {
                        "zip_index": record["zip_index"],
                        "source_archive": record["zip_path"].name,
                        "source_split": record["source_split"],
                        "source_file_name": record["source_file_name"],
                        "source_image_id": record["source_image_id"],
                        "sku": record["sku"],
                        "output_image_id": record["output_image_id"],
                        "destination_file_name": record["destination_file_name"],
                    }
                    for record in group
                ],
            }
        )
    return {"seed": seed, "ratios": ratios, "family_count": len(families), "families": families}


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _archives_unchanged(archives: Iterable[dict[str, Any]]) -> bool:
    return all(
        _zip_fingerprint(Path(archive["path"])) == tuple(archive["fingerprint"])
        for archive in archives
    )


def _materialize_dataset(
    output_dir: Path,
    records: list[dict[str, Any]],
    archives: list[dict[str, Any]],
    documents: dict[str, dict[str, Any]],
    class_mapping: dict[str, Any],
    family_manifest: dict[str, Any],
    report: dict[str, Any],
    overwrite: bool,
) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Pass --overwrite to replace it."
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=str(output_dir.parent))
    )
    try:
        for split in SPLITS:
            (staging_dir / split).mkdir(parents=True, exist_ok=False)

        records_by_archive: dict[Path, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            records_by_archive[record["zip_path"]].append(record)
        for archive_path, archive_records in records_by_archive.items():
            with zipfile.ZipFile(archive_path, mode="r") as archive:
                for record in archive_records:
                    destination = (
                        staging_dir / record["output_split"] / record["destination_file_name"]
                    )
                    with archive.open(record["source_member"], mode="r") as source, destination.open(
                        "xb"
                    ) as target:
                        shutil.copyfileobj(source, target)

        for split in SPLITS:
            _write_json(staging_dir / split / ANNOTATION_NAME, documents[split])
        _write_json(staging_dir / "class_mapping.json", class_mapping)
        _write_json(staging_dir / "source_family_manifest.json", family_manifest)

        if not _archives_unchanged(archives):
            raise AssertionError("One or more source ZIP files changed while building the dataset")
        report["assertions"]["source_archives_unchanged"] = True
        _write_json(staging_dir / "build_report.json", report)

        if output_dir.exists():
            if output_dir.is_dir():
                shutil.rmtree(output_dir)
            else:
                output_dir.unlink()
        staging_dir.replace(output_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def build_clean_axe_dataset(
    input_dir: Path = ROOT / "data" / "axe_zips",
    output_dir: Path = ROOT / "data" / "axe_dataset_clean",
    *,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build (or only plan) the cleaned AXE dataset and return its full report."""

    ratios = validate_ratios(train_ratio, val_ratio, test_ratio)
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    records, archives = read_source_archives(input_dir)
    assignments, _ = assign_grouped_stratified_splits(records, ratios, seed)
    category_ids = _prepare_global_ids(records)
    documents = _build_coco_documents(records, category_ids)
    assertions = _assert_output_invariants(records, category_ids)
    if not _archives_unchanged(archives):
        raise AssertionError("One or more source ZIP files changed while planning the dataset")
    assertions["source_archives_unchanged"] = True

    class_mapping = {
        "categories": [
            {"id": category_ids[sku], "name": sku, "slug": sku_slug(sku)}
            for sku in EXPECTED_SKUS
        ],
        "name_to_id": category_ids,
        "source_archives": [
            {
                key: archive[key]
                for key in ("zip_index", "file_name", "sku", "images")
            }
            for archive in archives
        ],
    }
    family_manifest = _family_manifest(records, assignments, seed, ratios)
    report = {
        "dry_run": bool(dry_run),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": int(seed),
        "ratios": ratios,
        "source": _source_summary(records, archives),
        "splits": _split_summary(records, ratios),
        "assertions": assertions,
        "files_written": 0 if dry_run else len(records) + len(SPLITS) + 3,
    }
    if not dry_run:
        _materialize_dataset(
            output_dir,
            records,
            archives,
            documents,
            class_mapping,
            family_manifest,
            report,
            overwrite,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data" / "axe_zips")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "axe_dataset_clean")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_clean_axe_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
