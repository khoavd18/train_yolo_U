"""Validated dataset-source registry with explicit annotation encodings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config.paths import PROJECT_ROOT, resolve_config_path, resolve_repo_path
from src.data.annotations import AnnotationFormat


DEFAULT_SOURCE_REGISTRY_PATH = PROJECT_ROOT / "configs" / "datasets" / "sources.yaml"
CLASS_MAPPING_SOURCES = {"archive_data_yaml", "data_yaml", "coco_categories"}


@dataclass(frozen=True)
class DatasetSource:
    name: str
    path: Path
    annotation_format: AnnotationFormat
    class_mapping_source: str
    sha256: str | None = None


class DatasetSourceRegistry:
    def __init__(self, sources: list[DatasetSource], *, source: Path) -> None:
        if not sources:
            raise ValueError("Dataset source registry must not be empty")
        by_name: dict[str, DatasetSource] = {}
        by_path: dict[str, DatasetSource] = {}
        by_hash: dict[str, DatasetSource] = {}
        for item in sources:
            if not item.name or item.name in by_name:
                raise ValueError(f"Duplicate or empty dataset source name: {item.name!r}")
            path_key = str(item.path.resolve()).casefold()
            if path_key in by_path:
                raise ValueError(f"Duplicate dataset source path: {item.path}")
            if item.class_mapping_source not in CLASS_MAPPING_SOURCES:
                raise ValueError(
                    f"Unsupported class_mapping_source {item.class_mapping_source!r}"
                )
            if item.sha256 is not None:
                digest = item.sha256.casefold()
                if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                    raise ValueError(f"Invalid SHA-256 for dataset source {item.name!r}")
                if digest in by_hash:
                    raise ValueError(f"Duplicate dataset source SHA-256: {digest}")
                by_hash[digest] = item
            by_name[item.name] = item
            by_path[path_key] = item
        self.sources = tuple(sources)
        self.source = source
        self._by_name = by_name
        self._by_path = by_path
        self._by_hash = by_hash

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, source: str | Path = "<memory>"
    ) -> "DatasetSourceRegistry":
        if payload.get("schema_version") != 1:
            raise ValueError("Dataset source registry must use schema_version 1")
        rows = payload.get("sources")
        if not isinstance(rows, list):
            raise ValueError("Dataset source registry 'sources' must be a list")
        parsed: list[DatasetSource] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"Dataset source row {index} must be a mapping")
            required = {"name", "path", "annotation_format", "class_mapping_source"}
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"Dataset source row {index} is missing fields: {sorted(missing)}"
                )
            if not all(isinstance(row[key], str) for key in required):
                raise ValueError(f"Dataset source row {index} fields must be strings")
            parsed.append(
                DatasetSource(
                    name=row["name"].strip(),
                    path=resolve_repo_path(row["path"]),
                    annotation_format=AnnotationFormat.parse(row["annotation_format"]),
                    class_mapping_source=row["class_mapping_source"].strip(),
                    sha256=(str(row["sha256"]).strip() if row.get("sha256") else None),
                )
            )
        source_path = (
            resolve_config_path(source)
            if str(source) != "<memory>"
            else Path("<memory>")
        )
        return cls(parsed, source=source_path)

    @classmethod
    def from_file(
        cls, path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH
    ) -> "DatasetSourceRegistry":
        resolved = resolve_config_path(path)
        payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Dataset source registry document must be a mapping")
        return cls.from_mapping(payload, source=resolved)

    def get(self, name: str) -> DatasetSource:
        try:
            return self._by_name[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset source {name!r}") from exc

    def match(
        self, path: str | Path, *, sha256: str | None = None
    ) -> DatasetSource:
        resolved = resolve_repo_path(path)
        item = self._by_path.get(str(resolved).casefold())
        if item is None and sha256:
            item = self._by_hash.get(sha256.casefold())
        if item is None:
            raise ValueError(
                f"Dataset source is not declared in {self.source}: {resolved}. "
                "Add an explicit annotation_format and class_mapping_source first."
            )
        if sha256 and item.sha256 and item.sha256.casefold() != sha256.casefold():
            raise ValueError(
                f"Dataset source checksum differs from configuration for {item.name}: "
                f"expected {item.sha256}, found {sha256}"
            )
        return item


def load_dataset_sources(
    path: str | Path = DEFAULT_SOURCE_REGISTRY_PATH,
) -> DatasetSourceRegistry:
    return DatasetSourceRegistry.from_file(path)
