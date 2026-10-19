"""Path resolution anchored to explicit repository or configuration roots."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve ``value`` independently of the process working directory."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(root).expanduser().resolve() / path).resolve()


def resolve_config_path(value: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve a config filename against the documented repository root."""

    return resolve_repo_path(value, root)


def resolve_config_relative_path(value: str | Path, config_path: str | Path) -> Path:
    """Resolve a value explicitly relative to the directory containing a config."""

    config = Path(config_path).expanduser()
    if not config.is_absolute():
        config = resolve_config_path(config)
    return resolve_repo_path(value, config.resolve().parent)
