"""Shared, import-safe workflow for ordinary Ultralytics YOLO training.

This module prepares and validates a training launch, but importing it never
loads Ultralytics or starts training.  Dataset-specific wrappers may retain
stricter review policies while sharing path handling, config contracts, run
planning, dependency checks, and the final ``YOLO.train`` invocation.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import yaml

from src.config.registry import load_brand_registry
from src.training.yolo_dataset_policy import assess_nine_brand_training_policy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VersionMode = Literal["exact", "prefix"]
BackendLoader = Callable[[], tuple[Callable[[str], Any], str]]
PayloadFactory = Callable[[str], Mapping[str, Any]]
VersionErrorFactory = Callable[[str, str], str]
TrainingCapture = Any

OPTIMIZER_SETTING_KEYS = (
    "optimizer",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
)
AUGMENTATION_SETTING_KEYS = (
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "degrees",
    "shear",
    "perspective",
    "fliplr",
    "flipud",
    "mosaic",
    "close_mosaic",
    "mixup",
    "copy_paste",
)
CLASS_ORDER_CHOICES = ("registry", "source_data_yaml")


@dataclass(frozen=True)
class YoloRunPlan:
    """Resolved paths and arguments for one ordinary detector training run."""

    config_path: Path
    config: dict[str, Any]
    data_yaml: Path
    project_dir: Path
    run_dir: Path
    model_argument: str
    train_args: dict[str, Any]


def resolve_path(value: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    """Resolve repository-relative paths independently of the current directory."""

    path = Path(value)
    base = Path(root)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def read_yolo_config(path: str | Path) -> dict[str, Any]:
    """Read one YOLO YAML config without importing the training backend."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config was not found: {config_path}")
    value = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Training config must contain a YAML mapping: {config_path}")
    if "train" in value and not isinstance(value["train"], dict):
        raise ValueError("Training config field 'train' must be a mapping")
    return value


def load_reviewed_yolo_config(
    path: str | Path,
    *,
    required_settings: Mapping[str, Any],
    mismatch_message: str,
    heldout_message: str,
) -> dict[str, Any]:
    """Load a legacy reviewed config and preserve its exact fixed baseline."""

    config = read_yolo_config(path)
    train = config.get("train", {})
    actual = {
        "model": config.get("model"),
        **{
            key: train.get(key)
            for key in required_settings
            if key != "model"
        },
    }
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in required_settings.items()
        if actual.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"{mismatch_message}: {mismatches}")
    if config.get("artifacts", {}).get("evaluate_test_split") is not False:
        raise ValueError(heldout_message)
    return config


def load_generic_yolo_config(path: str | Path) -> dict[str, Any]:
    """Validate safety invariants while leaving ordinary hyperparameters configurable."""

    config = read_yolo_config(path)
    missing = [
        key
        for key in ("model", "data", "project", "name", "train")
        if key not in config or config[key] in (None, "")
    ]
    if missing:
        raise ValueError(f"Detector training config is missing fields: {missing}")
    train = config["train"]
    missing_train = [
        key
        for key in ("epochs", "imgsz", "batch", "device", "seed")
        if key not in train
    ]
    if missing_train:
        raise ValueError(f"Detector train settings are missing fields: {missing_train}")
    for key in ("epochs", "imgsz", "batch"):
        if int(train[key]) <= 0:
            raise ValueError(f"Detector train setting {key!r} must be positive")
    if str(train.get("task", "detect")) != "detect":
        raise ValueError("Generic detector training requires task: detect")
    class_order = str(config.get("class_order", "registry"))
    if class_order not in CLASS_ORDER_CHOICES:
        raise ValueError(
            f"Unsupported detector class_order {class_order!r}; "
            f"choose from {CLASS_ORDER_CHOICES}"
        )
    if class_order == "source_data_yaml" and not config.get("brand_profile"):
        raise ValueError("class_order: source_data_yaml requires an explicit brand_profile")
    if train.get("resume") is True:
        raise ValueError("Generic detector training requires a new run; resume must be false")
    if train.get("exist_ok") is True:
        raise ValueError("Generic detector training refuses overwrite; exist_ok must be false")
    if train.get("val") is False:
        raise ValueError("Detector training must keep validation enabled")
    if config.get("artifacts", {}).get("evaluate_test_split") is not False:
        raise ValueError("Config must explicitly keep held-out test evaluation disabled")
    compatibility = config.get("compatibility", {})
    if not compatibility.get("verified_ultralytics"):
        raise ValueError("Config must declare compatibility.verified_ultralytics")
    if compatibility.get("install_or_upgrade_automatically") is True:
        raise ValueError("Detector training must not install or upgrade dependencies automatically")
    name = str(config["name"])
    if not name or Path(name).name != name:
        raise ValueError("Detector run name must be one path-safe component")
    return config


def _normalize_class_mapping(names: Any) -> dict[str, str]:
    if isinstance(names, (list, tuple)):
        indexed = {index: value for index, value in enumerate(names)}
    elif isinstance(names, dict):
        indexed: dict[int, Any] = {}
        for raw_index, value in names.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError("Dataset class-map keys must be integer-like") from exc
            if index in indexed:
                raise ValueError(f"Dataset class-map index is duplicated: {index}")
            indexed[index] = value
    else:
        raise ValueError("Dataset data.yaml field 'names' must be a mapping or sequence")
    if not indexed or sorted(indexed) != list(range(len(indexed))):
        raise ValueError("Dataset class indexes must be contiguous from zero")
    normalized = {str(index): str(indexed[index]).strip() for index in sorted(indexed)}
    if any(not name for name in normalized.values()):
        raise ValueError("Dataset class names must be non-empty")
    folded = [name.casefold() for name in normalized.values()]
    if len(folded) != len(set(folded)):
        raise ValueError("Dataset class names must be unique (case-insensitive)")
    return normalized


def load_dataset_class_mapping(data_yaml: str | Path) -> dict[str, str]:
    """Read and validate the authoritative detector class order from data.yaml."""

    path = Path(data_yaml)
    if not path.is_file():
        raise FileNotFoundError(f"Prepared detector data.yaml was not found: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Detector data.yaml must contain a mapping: {path}")
    for split in ("train", "val"):
        if not value.get(split):
            raise ValueError(f"Detector data.yaml must define the {split!r} split")
    mapping = _normalize_class_mapping(value.get("names"))
    if "nc" in value and int(value["nc"]) != len(mapping):
        raise ValueError("Detector data.yaml nc differs from its names mapping")
    return mapping


def resolve_dataset_split_paths(data_yaml: str | Path) -> dict[str, dict[str, Any]]:
    """Resolve configured dataset splits without asking Ultralytics to load data."""

    path = Path(data_yaml).resolve()
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Detector data.yaml must contain a mapping: {path}")
    configured_root = value.get("path")
    if configured_root:
        root_path = Path(str(configured_root))
        dataset_root = (
            root_path.resolve()
            if root_path.is_absolute()
            else (path.parent / root_path).resolve()
        )
    else:
        dataset_root = path.parent

    resolved: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        raw = value.get(split)
        if raw is None:
            resolved[split] = {"configured": None, "resolved": [], "all_exist": False}
            continue
        raw_items = list(raw) if isinstance(raw, (list, tuple)) else [raw]
        paths = []
        for item in raw_items:
            item_path = Path(str(item))
            paths.append(
                item_path.resolve()
                if item_path.is_absolute()
                else (dataset_root / item_path).resolve()
            )
        resolved[split] = {
            "configured": raw,
            "resolved": [str(item) for item in paths],
            "all_exist": bool(paths) and all(item.exists() for item in paths),
        }
    return resolved


def load_dataset_identity(
    data_yaml: str | Path,
    checks: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read prepared dataset version/fingerprint metadata when it is available."""

    dataset_root = Path(data_yaml).resolve().parent
    report_path = dataset_root / "dataset_report.json"
    fingerprint_path = dataset_root / "dataset_fingerprint.json"
    report: dict[str, Any] = {}
    fingerprint_document: dict[str, Any] = {}
    if report_path.is_file():
        loaded = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            report = loaded
    if fingerprint_path.is_file():
        loaded = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            fingerprint_document = loaded

    check_values = checks or {}
    version = next(
        (
            value
            for value in (
                check_values.get("dataset_version"),
                report.get("dataset_version"),
                report.get("version"),
                report.get("source_version"),
            )
            if value not in (None, "")
        ),
        dataset_root.name,
    )
    fingerprint: Any = None
    for source in (check_values, report, fingerprint_document):
        for key in (
            "dataset_fingerprint",
            "dataset_fingerprint_sha256",
            "fingerprint",
            "fingerprint_sha256",
            "sha256",
        ):
            if source.get(key) not in (None, ""):
                fingerprint = source[key]
                break
        if fingerprint is not None:
            break
    return {
        "version": version,
        "fingerprint": fingerprint,
        "fingerprint_path": str(fingerprint_path) if fingerprint_path.is_file() else None,
        "report_path": str(report_path) if report_path.is_file() else None,
    }


def validate_brand_profile_class_mapping(
    data_yaml: str | Path,
    config: Mapping[str, Any],
    mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Validate training IDs separately from stable registry brand IDs.

    Registry order remains the default contract.  ``source_data_yaml`` is an
    explicit opt-in for a prepared dataset whose training/checkpoint IDs follow
    its source YAML; in that mode ``mapping.json`` must prove every source ID,
    training ID, canonical name, and stable registry identity.
    """

    data_path = Path(data_yaml).resolve()
    training_mapping = dict(mapping or load_dataset_class_mapping(data_path))
    class_order = str(config.get("class_order", "registry"))
    if class_order not in CLASS_ORDER_CHOICES:
        raise ValueError(
            f"Unsupported detector class_order {class_order!r}; "
            f"choose from {CLASS_ORDER_CHOICES}"
        )
    profile_name = config.get("brand_profile")
    if not profile_name:
        if class_order != "registry":
            raise ValueError("class_order: source_data_yaml requires an explicit brand_profile")
        return {
            "class_order": class_order,
            "brand_profile": None,
            "training_to_registry": [],
            "source_to_training_verified": None,
            "mapping_path": None,
        }

    registry = load_brand_registry()
    profile = registry.profile(str(profile_name))
    registry_names = tuple(brand.name for brand in profile)
    by_name = {brand.name: brand for brand in profile}
    training_names = tuple(
        training_mapping[str(index)] for index in range(len(training_mapping))
    )
    if len(training_names) != len(registry_names) or set(training_names) != set(
        registry_names
    ):
        raise ValueError(
            f"Prepared dataset classes must be a one-to-one permutation of brand profile "
            f"{profile_name!r}: registry={registry_names}, training={training_names}"
        )

    if class_order == "registry":
        expected = {str(index): name for index, name in enumerate(registry_names)}
        if training_mapping != expected:
            raise ValueError(
                f"Prepared dataset class order does not match brand profile {profile_name!r}: "
                f"expected={expected}, actual={training_mapping}"
            )
        return {
            "class_order": class_order,
            "brand_profile": str(profile_name),
            "training_to_registry": [
                {
                    "training_id": index,
                    "training_name": name,
                    "registry_id": by_name[name].id,
                    "registry_name": name,
                    "source_id": None,
                    "source_name": None,
                }
                for index, name in enumerate(training_names)
            ],
            "source_to_training_verified": None,
            "mapping_path": None,
        }

    mapping_path = data_path.parent / "mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(
            "class_order: source_data_yaml requires prepared mapping metadata: "
            f"{mapping_path}"
        )
    document = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Prepared mapping must contain an object: {mapping_path}")
    if document.get("target_profile") != str(profile_name):
        raise ValueError(
            "Prepared mapping target_profile differs from training config: "
            f"expected={profile_name!r}, actual={document.get('target_profile')!r}"
        )
    if document.get("class_order") != class_order:
        raise ValueError(
            "Prepared mapping class_order differs from training config: "
            f"expected={class_order!r}, actual={document.get('class_order')!r}"
        )
    documented_names = document.get("target_names", document.get("output_names"))
    if documented_names is None or _normalize_class_mapping(documented_names) != training_mapping:
        raise ValueError("Prepared mapping training IDs/names differ from data.yaml")
    if "output_names" in document and (
        _normalize_class_mapping(document["output_names"]) != training_mapping
    ):
        raise ValueError("Prepared mapping output_names differ from data.yaml")

    source_rows = document.get("source_to_target")
    if not isinstance(source_rows, list) or len(source_rows) != len(training_mapping):
        raise ValueError(
            "Prepared mapping source_to_target must contain exactly one row per training class"
        )
    parsed_rows: dict[int, dict[str, Any]] = {}
    source_ids: set[int] = set()
    for index, row in enumerate(source_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Prepared mapping source_to_target row {index} is not an object")
        if any(
            isinstance(row.get(key), bool) or not isinstance(row.get(key), int)
            for key in ("source_id", "target_id")
        ):
            raise ValueError("Prepared mapping class IDs must be integers, not booleans")
        try:
            source_id = row["source_id"]
            training_id = row["target_id"]
            source_name = str(row["source_name"])
            training_name = str(row["target_name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Prepared mapping source_to_target row {index} is incomplete"
            ) from exc
        if source_id in source_ids or training_id in parsed_rows:
            raise ValueError("Prepared mapping source/training class IDs must be unique")
        source_ids.add(source_id)
        if source_id != training_id:
            raise ValueError(
                "source_data_yaml order requires each source_id to equal its training target_id"
            )
        if training_mapping.get(str(training_id)) != training_name:
            raise ValueError(
                f"Prepared mapping target ID/name differs from data.yaml at ID {training_id}"
            )
        inferred = registry.infer_brand(source_name, str(profile_name))
        if inferred != training_name:
            raise ValueError(
                f"Prepared mapping source class {source_id} {source_name!r} does not map "
                f"to canonical target {training_name!r}"
            )
        registry_id = by_name[training_name].id
        if "registry_id" in row:
            documented_registry_id = row["registry_id"]
            if (
                isinstance(documented_registry_id, bool)
                or not isinstance(documented_registry_id, int)
                or documented_registry_id != registry_id
            ):
                raise ValueError(
                    f"Prepared mapping registry_id differs for {training_name!r}: "
                    f"expected={registry_id}, actual={documented_registry_id}"
                )
        parsed_rows[training_id] = {
            "training_id": training_id,
            "training_name": training_name,
            "registry_id": registry_id,
            "registry_name": training_name,
            "source_id": source_id,
            "source_name": source_name,
        }
    expected_ids = set(range(len(training_mapping)))
    if source_ids != expected_ids or set(parsed_rows) != expected_ids:
        raise ValueError("Prepared mapping source/training IDs must be contiguous from zero")
    return {
        "class_order": class_order,
        "brand_profile": str(profile_name),
        "training_to_registry": [parsed_rows[index] for index in sorted(parsed_rows)],
        "source_to_training_verified": True,
        "mapping_path": str(mapping_path),
    }


def validate_generic_dataset_metadata(
    data_yaml: str | Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any], dict[str, Any] | None]:
    """Validate common split/mapping guarantees for the generic detector CLI.

    The nine-brand clean-subset exception remains guarded by its existing
    policy.  Dataset-specific legacy wrappers continue to run their stricter
    validators before reaching the shared launch function.
    """

    path = Path(data_yaml)
    mapping = load_dataset_class_mapping(path)
    checks_path = path.parent / "dataset_checks.json"
    if not checks_path.is_file():
        raise FileNotFoundError(f"Prepared dataset metadata is missing: {checks_path}")
    checks = json.loads(checks_path.read_text(encoding="utf-8"))
    if not isinstance(checks, dict):
        raise ValueError(f"Prepared dataset checks must contain an object: {checks_path}")
    if checks.get("test_split_evaluated") is not False:
        raise ValueError("Prepared dataset unexpectedly records test evaluation")
    if int(checks.get("cross_split_family_overlap", -1)) != 0:
        raise ValueError("Prepared dataset has cross-split family overlap")
    exact_overlap = checks.get("exact_image_cross_split_overlap")
    if exact_overlap is not None and (
        not isinstance(exact_overlap, dict)
        or any(int(value) != 0 for value in exact_overlap.values())
    ):
        raise ValueError("Prepared dataset has exact-image cross-split overlap")
    if "ordinary_yolo_detection_rows_only" in checks and (
        checks.get("ordinary_yolo_detection_rows_only") is not True
    ):
        raise ValueError("Prepared dataset contains unsupported non-detection labels")

    verified = checks.get("mapping_verified")
    if verified is not None:
        if _normalize_class_mapping(verified) != mapping:
            raise ValueError("Prepared dataset class order is incompatible with data.yaml")
    elif not (
        checks.get("mapping_is_single_class_axe") is True
        and mapping == {"0": "AXE"}
    ):
        raise ValueError("Prepared dataset metadata does not verify its class mapping")

    class_mapping_contract = validate_brand_profile_class_mapping(path, config, mapping)
    if class_mapping_contract["class_order"] == "source_data_yaml":
        if checks.get("class_order") != "source_data_yaml":
            raise ValueError(
                "Prepared dataset checks do not verify class_order: source_data_yaml"
            )
        documented_stable_ids = checks.get("stable_registry_ids")
        expected_stable_ids = {
            str(row["training_id"]): row["registry_id"]
            for row in class_mapping_contract["training_to_registry"]
        }
        if not isinstance(documented_stable_ids, Mapping) or {
            str(key): value for key, value in documented_stable_ids.items()
        } != expected_stable_ids:
            raise ValueError(
                "Prepared dataset stable_registry_ids differ from mapping.json/brand registry"
            )

    policy: dict[str, Any] | None = None
    status = checks.get("status")
    if status == "structural_checks_passed":
        policy = assess_nine_brand_training_policy(
            checks,
            allow_quarantined_exclusions=bool(
                config.get("dataset_policy", {}).get("allow_quarantined_exclusions", False)
            ),
        )
        if not policy["effective_training_ready"]:
            raise RuntimeError(
                "Prepared dataset is not approved by its clean-subset policy: "
                f"{policy['failed_safety_requirements']}"
            )
    elif status != "passed":
        raise ValueError(f"Prepared dataset checks did not pass: status={status!r}")
    return mapping, checks, policy


def build_yolo_run_plan(
    config: dict[str, Any],
    config_path: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
    resolve_model_argument: bool = True,
) -> YoloRunPlan:
    """Resolve one config into the exact arguments passed to Ultralytics."""

    base = Path(root)
    data_yaml = resolve_path(config["data"], base)
    project_dir = resolve_path(config["project"], base)
    run_dir = project_dir / str(config["name"])
    model_argument = (
        str(resolve_path(config["model"], base))
        if resolve_model_argument
        else str(config["model"])
    )
    train_args = dict(config["train"])
    train_args.update(
        {
            "data": str(data_yaml),
            "project": str(project_dir),
            "name": str(config["name"]),
        }
    )
    return YoloRunPlan(
        config_path=Path(config_path).resolve(),
        config=config,
        data_yaml=data_yaml,
        project_dir=project_dir,
        run_dir=run_dir,
        model_argument=model_argument,
        train_args=train_args,
    )


def ensure_new_run(run_dir: str | Path) -> None:
    """Refuse every ordinary training launch that could overwrite a run."""

    path = Path(run_dir)
    if path.exists():
        raise FileExistsError(f"Training run already exists; refusing to overwrite: {path}")


def build_generic_preflight(
    config_path: str | Path,
    *,
    root: str | Path = PROJECT_ROOT,
) -> tuple[YoloRunPlan, dict[str, Any]]:
    """Build a configuration-driven, read-only ordinary detector preflight."""

    resolved_config = resolve_path(config_path, root)
    config = load_generic_yolo_config(resolved_config)
    plan = build_yolo_run_plan(config, resolved_config, root=root)
    mapping, checks, policy = validate_generic_dataset_metadata(plan.data_yaml, config)
    class_mapping_contract = validate_brand_profile_class_mapping(
        plan.data_yaml, config, mapping
    )
    ensure_new_run(plan.run_dir)
    dataset_identity = load_dataset_identity(plan.data_yaml, checks)
    summary: dict[str, Any] = {
        "python": sys.executable,
        "config": str(plan.config_path),
        "model": plan.model_argument,
        "data": str(plan.data_yaml),
        "mapping": mapping,
        "brand_profile": class_mapping_contract["brand_profile"],
        "class_order": class_mapping_contract["class_order"],
        "registry_stable_mapping": class_mapping_contract["training_to_registry"],
        "source_to_training_mapping_verified": class_mapping_contract[
            "source_to_training_verified"
        ],
        "run_dir": str(plan.run_dir),
        "epochs": config["train"]["epochs"],
        "dataset_check_status": checks.get("status"),
        "dataset_ready": True,
        "dataset_identity": dataset_identity,
        "dataset_version": dataset_identity["version"],
        "dataset_fingerprint": dataset_identity["fingerprint"],
        "augmentation_audit": checks.get("augmentation_audit"),
        "resolved_splits": resolve_dataset_split_paths(plan.data_yaml),
        "model_path_exists": Path(plan.model_argument).is_file(),
        "test_evaluation": False,
        "model_inference": False,
        "training_started": False,
    }
    if policy is not None:
        summary["dataset_policy"] = policy
    return plan, summary


def inspect_yolo_runtime(plan: YoloRunPlan) -> dict[str, Any]:
    """Validate the installed training API and selected device without loading a model."""

    if importlib.util.find_spec("ultralytics") is None:
        raise ModuleNotFoundError("Ultralytics is not installed in this interpreter")

    import torch
    from ultralytics import __version__ as ultralytics_version
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
    from ultralytics.utils.torch_utils import select_device

    expected_version = str(plan.config["compatibility"]["verified_ultralytics"])
    actual_version = str(ultralytics_version)
    if actual_version != expected_version:
        raise RuntimeError(
            f"Expected reviewed Ultralytics {expected_version}, found {actual_version}"
        )

    # get_cfg performs Ultralytics' own key, type, and value checks. It does not
    # construct a YOLO model, read a checkpoint, or build a dataloader.
    effective = get_cfg(DEFAULT_CFG_DICT, plan.train_args)
    selected = select_device(
        getattr(effective, "device"),
        batch=int(getattr(effective, "batch")),
        verbose=False,
    )
    cuda_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            cuda_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )

    return {
        "ultralytics": {
            "expected": expected_version,
            "installed": actual_version,
            "exact_match": True,
        },
        "training_arguments_supported": True,
        "supported_training_arguments": sorted(plan.train_args),
        "task": str(getattr(effective, "task")),
        "device": {
            "configured": getattr(effective, "device"),
            "selected": str(selected),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": cuda_devices,
        },
        "effective_optimizer": {
            key: getattr(effective, key) for key in OPTIMIZER_SETTING_KEYS
        },
        "effective_augmentation": {
            key: getattr(effective, key) for key in AUGMENTATION_SETTING_KEYS
        },
        "validation_enabled": bool(getattr(effective, "val")),
        "test_evaluation": False,
        "model_inference": False,
        "model_constructed": False,
        "training_started": False,
    }


def _load_ultralytics_backend(
    missing_dependency_message: str,
) -> tuple[Callable[[str], Any], str]:
    if importlib.util.find_spec("ultralytics") is None:
        raise ModuleNotFoundError(missing_dependency_message)
    from ultralytics import YOLO, __version__ as ultralytics_version

    return YOLO, str(ultralytics_version)


def execute_yolo_training(
    plan: YoloRunPlan,
    *,
    expected_version: str,
    version_mode: VersionMode = "exact",
    missing_dependency_message: str = "Ultralytics is not installed in this interpreter",
    version_error_factory: VersionErrorFactory | None = None,
    payload_factory: PayloadFactory | None = None,
    ensure_ascii: bool = True,
    backend_loader: BackendLoader | None = None,
    training_capture: TrainingCapture | None = None,
) -> Any:
    """Load Ultralytics lazily and execute exactly one explicit training call.

    ``backend_loader`` exists for isolated contract tests; production callers
    omit it and receive the installed Ultralytics ``YOLO`` class and version.
    """

    if version_mode not in ("exact", "prefix"):
        raise ValueError(f"Unsupported Ultralytics version mode: {version_mode}")
    loader = backend_loader or (
        lambda: _load_ultralytics_backend(missing_dependency_message)
    )
    model_factory, actual_version = loader()
    actual_version = str(actual_version)
    compatible = (
        actual_version == str(expected_version)
        if version_mode == "exact"
        else actual_version.startswith(str(expected_version))
    )
    if not compatible:
        message = (
            version_error_factory(str(expected_version), actual_version)
            if version_error_factory is not None
            else (
                f"Expected Ultralytics {expected_version!r} ({version_mode}), "
                f"found {actual_version}"
            )
        )
        raise RuntimeError(message)
    payload = (
        dict(payload_factory(actual_version))
        if payload_factory is not None
        else {"ultralytics": actual_version, "run_dir": str(plan.run_dir)}
    )
    print(json.dumps(payload, ensure_ascii=ensure_ascii, indent=2), flush=True)
    # Recheck immediately before model construction. Ultralytics itself may
    # increment a colliding name when exist_ok=false; this guard preserves the
    # exact configured output path instead.
    ensure_new_run(plan.run_dir)
    model = model_factory(plan.model_argument)
    if training_capture is not None:
        attach = getattr(training_capture, "attach", None)
        if not callable(attach):
            raise TypeError("training_capture must provide attach(model)")
        attach(model)
    return model.train(**plan.train_args)
