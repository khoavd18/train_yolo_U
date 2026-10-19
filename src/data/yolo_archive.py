"""Shared, side-effect-free primitives for audited YOLO ZIP exports."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile, ZipInfo


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_info(archive: ZipFile, info: ZipInfo) -> bytes:
    """Address a ZIP member by ZipInfo, preserving duplicate-name semantics."""

    with archive.open(info, "r") as handle:
        return handle.read()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(
    path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_yaml_names(value: Any) -> list[str]:
    """Read a YOLO class order from either list or contiguous integer mapping."""

    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        try:
            keyed = {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("YAML names mapping must have integer-like keys") from exc
        if sorted(keyed) != list(range(len(keyed))):
            raise ValueError(
                f"YAML names keys must be contiguous from zero: {sorted(keyed)}"
            )
        return [keyed[index] for index in range(len(keyed))]
    raise ValueError("YAML names must be a list or integer-keyed mapping")


def label_member_for(image_member: str) -> str:
    """Map an images member to its labels member without using filenames for identity."""

    path = PurePosixPath(image_member)
    parts = list(path.parts)
    lowered = [part.casefold() for part in parts]
    if "images" not in lowered:
        raise ValueError(f"Image is not under an images directory: {image_member}")
    parts[lowered.index("images")] = "labels"
    return str(PurePosixPath(*parts).with_suffix(".txt"))
