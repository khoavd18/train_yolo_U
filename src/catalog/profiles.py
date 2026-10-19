"""Validated product identity profiles used by checkpoints and catalog mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.config.paths import PROJECT_ROOT, resolve_config_path


PRODUCT_PROFILE_DIR = PROJECT_ROOT / "configs" / "products"


@dataclass(frozen=True)
class ProductClassIdentity:
    class_index: int
    legacy_class_name: str
    item_code: str


@dataclass(frozen=True)
class ProductIdentityProfile:
    name: str
    brand: str
    classes: tuple[ProductClassIdentity, ...]
    identity_guards: Mapping[str, Mapping[str, str]]
    source: Path

    @property
    def legacy_class_names(self) -> tuple[str, ...]:
        return tuple(row.legacy_class_name for row in self.classes)

    @property
    def item_codes(self) -> tuple[str, ...]:
        return tuple(row.item_code for row in self.classes)

    @property
    def expected_mapping(self) -> dict[str, str]:
        return {row.legacy_class_name: row.item_code for row in self.classes}

    def identity_guard(self, name: str) -> dict[str, str]:
        try:
            return dict(self.identity_guards[name])
        except KeyError as exc:
            raise KeyError(f"Unknown identity guard {name!r} in product profile {self.name!r}") from exc


def _profile_path(name: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Product profile name must be a non-empty string")
    normalized = name.strip()
    if Path(normalized).name != normalized or normalized in {".", ".."}:
        raise ValueError("Product profile name must not contain path separators")
    return PRODUCT_PROFILE_DIR / f"{normalized}.yaml"


def _from_mapping(value: Mapping[str, Any], source: Path) -> ProductIdentityProfile:
    if value.get("schema_version") != 1:
        raise ValueError("Product identity profile must use schema_version 1")
    name = value.get("name")
    brand = value.get("brand")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Product identity profile name must be a non-empty string")
    if not isinstance(brand, str) or not brand.strip():
        raise ValueError("Product identity profile brand must be a non-empty string")

    raw_classes = value.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise ValueError("Product identity profile classes must be a non-empty list")
    classes: list[ProductClassIdentity] = []
    for position, row in enumerate(raw_classes):
        if not isinstance(row, Mapping):
            raise ValueError(f"Product class entry {position} must be a mapping")
        missing = {"class_index", "legacy_class_name", "item_code"} - set(row)
        if missing:
            raise ValueError(f"Product class entry {position} is missing fields: {sorted(missing)}")
        class_index = row["class_index"]
        legacy_name = row["legacy_class_name"]
        item_code = row["item_code"]
        if isinstance(class_index, bool) or not isinstance(class_index, int):
            raise ValueError(f"Product class index must be an integer: {class_index!r}")
        if not isinstance(legacy_name, str) or not legacy_name.strip():
            raise ValueError("Product legacy class names must be non-empty strings")
        if not isinstance(item_code, str) or not item_code.strip():
            raise ValueError("Product item codes must be non-empty strings")
        classes.append(ProductClassIdentity(class_index, legacy_name, item_code))

    indexes = [row.class_index for row in classes]
    if indexes != list(range(len(classes))):
        raise ValueError("Product class indexes must be unique, ordered, and contiguous from zero")
    legacy_names = [row.legacy_class_name for row in classes]
    if len(legacy_names) != len(set(legacy_names)):
        raise ValueError("Product profile contains duplicate legacy class names")
    item_codes = [row.item_code for row in classes]
    if len(item_codes) != len(set(item_codes)):
        raise ValueError("Product profile contains duplicate item codes")

    raw_guards = value.get("identity_guards", {})
    if not isinstance(raw_guards, Mapping):
        raise ValueError("Product identity_guards must be a mapping")
    guards: dict[str, dict[str, str]] = {}
    for raw_guard_name, raw_identities in raw_guards.items():
        guard_name = str(raw_guard_name).strip()
        if not guard_name or not isinstance(raw_identities, Mapping) or len(raw_identities) < 2:
            raise ValueError(f"Identity guard {raw_guard_name!r} must define at least two identities")
        identities: dict[str, str] = {}
        for raw_identity, item_code in raw_identities.items():
            identity = str(raw_identity).strip()
            if not identity or not isinstance(item_code, str) or not item_code.strip():
                raise ValueError(f"Identity guard {guard_name!r} must use string names and item codes")
            identities[identity] = item_code
        if len(set(identities.values())) != len(identities):
            raise ValueError(f"Identity guard {guard_name!r} merges distinct identities")
        guards[guard_name] = identities

    return ProductIdentityProfile(
        name=name,
        brand=brand,
        classes=tuple(classes),
        identity_guards=guards,
        source=source,
    )


def load_product_profile(
    name: str = "axe5", *, path: str | Path | None = None
) -> ProductIdentityProfile:
    """Load a named profile, or an explicit path for fixtures/custom deployments."""

    resolved = resolve_config_path(path) if path is not None else _profile_path(name).resolve()
    value = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(value, Mapping):
        raise ValueError("Product identity profile document must be a mapping")
    profile = _from_mapping(value, resolved)
    if path is None and profile.name != name:
        raise ValueError(
            f"Product profile filename/name mismatch: requested {name!r}, document has {profile.name!r}"
        )
    return profile
