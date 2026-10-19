"""Validated, configuration-driven brand identities and detector profiles."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .paths import PROJECT_ROOT, resolve_config_path


DEFAULT_BRAND_REGISTRY_PATH = PROJECT_ROOT / "configs" / "brands.yaml"
_TOKEN_SEPARATOR_RE = re.compile(r"[^A-Z0-9]+")


def normalize_brand_token(value: str) -> str:
    """Normalize a canonical name, alias, or source-class prefix."""

    token = _TOKEN_SEPARATOR_RE.sub("_", str(value).strip().upper()).strip("_")
    if not token:
        raise ValueError("Brand names and aliases must contain letters or digits")
    return token


@dataclass(frozen=True)
class Brand:
    """One stable detector brand identity."""

    id: int
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def tokens(self) -> tuple[str, ...]:
        return (normalize_brand_token(self.name), *(normalize_brand_token(x) for x in self.aliases))


class BrandRegistry:
    """Validated brands plus explicit detector class-order profiles."""

    def __init__(
        self,
        brands: Sequence[Brand],
        profiles: Mapping[str, Sequence[int]],
        *,
        source: str | Path | None = None,
    ) -> None:
        if not brands:
            raise ValueError("Brand registry must define at least one brand")

        by_id: dict[int, Brand] = {}
        token_owners: dict[str, Brand] = {}
        for brand in brands:
            if isinstance(brand.id, bool) or not isinstance(brand.id, int) or brand.id < 0:
                raise ValueError(f"Brand id must be a non-negative integer: {brand.id!r}")
            if brand.id in by_id:
                raise ValueError(f"Duplicate brand id: {brand.id}")
            canonical = normalize_brand_token(brand.name)
            if brand.name != canonical:
                raise ValueError(
                    f"Canonical brand name must already be normalized as {canonical!r}: {brand.name!r}"
                )
            by_id[brand.id] = brand
            for token in brand.tokens:
                previous = token_owners.get(token)
                if previous is not None:
                    raise ValueError(
                        f"Duplicate or ambiguous brand alias {token!r}: "
                        f"{previous.name!r} and {brand.name!r}"
                    )
                token_owners[token] = brand

        token_rows = list(token_owners.items())
        for index, (left_token, left_owner) in enumerate(token_rows):
            for right_token, right_owner in token_rows[index + 1 :]:
                if left_owner.id == right_owner.id:
                    continue
                if left_token.startswith(right_token + "_") or right_token.startswith(left_token + "_"):
                    raise ValueError(
                        "Ambiguous brand alias prefixes across different brands: "
                        f"{left_token!r} ({left_owner.name}) and {right_token!r} ({right_owner.name})"
                    )

        validated_profiles: dict[str, tuple[int, ...]] = {}
        if not profiles:
            raise ValueError("Brand registry must define at least one profile")
        for raw_name, raw_ids in profiles.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Brand profile names must be non-empty")
            ids = tuple(raw_ids)
            if not ids:
                raise ValueError(f"Brand profile {name!r} must not be empty")
            if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
                raise ValueError(f"Brand profile {name!r} ids must be integers")
            if len(ids) != len(set(ids)):
                raise ValueError(f"Brand profile {name!r} contains duplicate ids")
            missing = sorted(set(ids) - set(by_id))
            if missing:
                raise ValueError(f"Brand profile {name!r} references unknown ids: {missing}")
            if ids != tuple(range(len(ids))):
                raise ValueError(
                    f"Brand profile {name!r} must preserve contiguous detector ids from zero: {ids}"
                )
            validated_profiles[name] = ids

        self._brands = tuple(sorted(by_id.values(), key=lambda item: item.id))
        self._by_id = by_id
        self._token_owners = token_owners
        self._profiles = validated_profiles
        self.source = Path(source).resolve() if source is not None else None

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, source: str | Path | None = None
    ) -> "BrandRegistry":
        if not isinstance(value, Mapping):
            raise ValueError("Brand registry document must be a mapping")
        if value.get("schema_version") != 1:
            raise ValueError("Brand registry must use schema_version 1")
        raw_brands = value.get("brands")
        if not isinstance(raw_brands, list):
            raise ValueError("Brand registry 'brands' must be a list")
        brands: list[Brand] = []
        for index, row in enumerate(raw_brands):
            if not isinstance(row, Mapping):
                raise ValueError(f"Brand entry {index} must be a mapping")
            missing = {"id", "name", "aliases"} - set(row)
            if missing:
                raise ValueError(f"Brand entry {index} is missing fields: {sorted(missing)}")
            aliases = row["aliases"]
            if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
                raise ValueError(f"Brand entry {index} aliases must be a list of strings")
            if not isinstance(row["name"], str):
                raise ValueError(f"Brand entry {index} name must be a string")
            brands.append(Brand(id=row["id"], name=row["name"], aliases=tuple(aliases)))

        raw_profiles = value.get("profiles")
        if not isinstance(raw_profiles, Mapping):
            raise ValueError("Brand registry 'profiles' must be a mapping")
        profiles: dict[str, Sequence[int]] = {}
        for name, row in raw_profiles.items():
            if not isinstance(row, Mapping) or "brand_ids" not in row:
                raise ValueError(f"Brand profile {name!r} must define brand_ids")
            ids = row["brand_ids"]
            if not isinstance(ids, list):
                raise ValueError(f"Brand profile {name!r} brand_ids must be a list")
            profiles[str(name)] = ids
        return cls(brands, profiles, source=source)

    @classmethod
    def from_file(cls, path: str | Path = DEFAULT_BRAND_REGISTRY_PATH) -> "BrandRegistry":
        resolved = resolve_config_path(path)
        value = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        return cls.from_mapping(value, source=resolved)

    @property
    def brands(self) -> tuple[Brand, ...]:
        return self._brands

    @property
    def profile_names(self) -> tuple[str, ...]:
        return tuple(self._profiles)

    def profile(self, name: str) -> tuple[Brand, ...]:
        try:
            ids = self._profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown brand profile {name!r}; choose from {sorted(self._profiles)}") from exc
        return tuple(self._by_id[item] for item in ids)

    def names(self, profile: str) -> tuple[str, ...]:
        return tuple(brand.name for brand in self.profile(profile))

    def class_names(self, profile: str) -> dict[int, str]:
        return {brand.id: brand.name for brand in self.profile(profile)}

    def infer_brand(self, class_name: str, profile: str) -> str | None:
        """Map only a canonical/alias class prefix from the selected profile."""

        try:
            normalized = normalize_brand_token(class_name)
        except ValueError:
            return None
        allowed_ids = {brand.id for brand in self.profile(profile)}
        matches = {
            owner.name
            for token, owner in self._token_owners.items()
            if owner.id in allowed_ids
            and (normalized == token or normalized.startswith(token + "_"))
        }
        if len(matches) > 1:
            raise ValueError(f"Ambiguous brand aliases for source class {class_name!r}: {sorted(matches)}")
        return next(iter(matches)) if matches else None


def load_brand_registry(path: str | Path = DEFAULT_BRAND_REGISTRY_PATH) -> BrandRegistry:
    """Load and validate a brand registry without global mutable caching."""

    return BrandRegistry.from_file(path)
