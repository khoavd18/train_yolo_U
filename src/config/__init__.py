"""Validated project configuration and repository-relative path helpers."""

from .paths import (
    PROJECT_ROOT,
    resolve_config_path,
    resolve_config_relative_path,
    resolve_repo_path,
)
from .registry import (
    Brand,
    BrandRegistry,
    load_brand_registry,
    normalize_brand_token,
)

__all__ = [
    "Brand",
    "BrandRegistry",
    "PROJECT_ROOT",
    "load_brand_registry",
    "normalize_brand_token",
    "resolve_config_path",
    "resolve_config_relative_path",
    "resolve_repo_path",
]
