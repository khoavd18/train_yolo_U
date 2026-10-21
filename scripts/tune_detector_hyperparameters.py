"""Plan or run a small validation-only detector hyperparameter pilot search."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.yolo import (  # noqa: E402
    build_yolo_run_plan,
    execute_yolo_training,
    inspect_yolo_runtime,
    load_generic_yolo_config,
    resolve_path,
    validate_brand_profile_class_mapping,
    validate_generic_dataset_metadata,
)
from src.training.yolo_reporting import (  # noqa: E402
    YoloTrainingCapture,
    write_yolo_training_report,
)

LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
ALLOWED_TRAIN_OVERRIDES = {
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "cos_lr",
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
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/brand_21brands_yolo.yaml")
    parser.add_argument("--search", default="configs/brand_21brands_yolo_search.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--summarize", action="store_true")
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def load_search(path: str | Path, *, root: str | Path = ROOT) -> dict[str, Any]:
    search_path = resolve_path(path, root)
    if not search_path.is_file():
        raise FileNotFoundError(f"Search config was not found: {search_path}")
    search = _load_yaml(search_path)
    if search.get("test_split_used") is not False:
        raise ValueError("Hyperparameter selection must explicitly keep test_split_used: false")
    if search.get("selection_mode") != "max":
        raise ValueError("This detector pilot search requires selection_mode: max")
    if str(search.get("selection_metric")) not in {
        "metrics/mAP50-95(B)",
        "metrics/mAP50(B)",
    }:
        raise ValueError("Unsupported validation selection metric")
    if int(search.get("pilot_epochs", 0)) <= 0:
        raise ValueError("pilot_epochs must be positive")
    trials = search.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("Search config must define at least one trial")
    labels: set[str] = set()
    for index, trial in enumerate(trials):
        if not isinstance(trial, dict):
            raise ValueError(f"Trial {index} must be a mapping")
        label = str(trial.get("label", ""))
        if not LABEL_RE.fullmatch(label) or label in labels:
            raise ValueError(f"Trial labels must be unique path-safe values: {label!r}")
        labels.add(label)
        overrides = trial.get("train", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"Trial {label!r} train overrides must be a mapping")
        unsupported = sorted(set(overrides) - ALLOWED_TRAIN_OVERRIDES)
        if unsupported:
            raise ValueError(f"Trial {label!r} has unsafe overrides: {unsupported}")
    search["_path"] = search_path
    return search


def build_trial_configs(
    base: Mapping[str, Any],
    search: Mapping[str, Any],
    *,
    root: str | Path = ROOT,
) -> list[dict[str, Any]]:
    project = str(search.get("project", "runs/model1_brand21_tuning"))
    prefix = str(search.get("name_prefix", "pilot"))
    if not LABEL_RE.fullmatch(prefix):
        raise ValueError("name_prefix must be one path-safe lowercase value")
    trials: list[dict[str, Any]] = []
    for raw in search["trials"]:
        label = str(raw["label"])
        config = deepcopy(dict(base))
        config["project"] = project
        config["name"] = f"{prefix}_{label}"
        config["train"] = deepcopy(dict(base["train"]))
        config["train"].update(dict(raw.get("train", {})))
        config["train"].update(
            {
                "epochs": int(search["pilot_epochs"]),
                "patience": 0,
                "resume": False,
                "exist_ok": False,
            }
        )
        plan = build_yolo_run_plan(
            config,
            Path(root) / "generated" / f"{config['name']}.yaml",
            root=root,
        )
        trials.append(
            {
                "label": label,
                "overrides": dict(raw.get("train", {})),
                "base_config": deepcopy(dict(base)),
                "config": config,
                "plan": plan,
            }
        )
    return trials


def build_preflight(
    config_path: str | Path,
    search_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    resolved_config = resolve_path(config_path, root)
    base = load_generic_yolo_config(resolved_config)
    base_plan = build_yolo_run_plan(base, resolved_config, root=root)
    mapping, checks, _policy = validate_generic_dataset_metadata(base_plan.data_yaml, base)
    contract = validate_brand_profile_class_mapping(base_plan.data_yaml, base, mapping)
    if contract.get("class_order") != "source_data_yaml":
        raise ValueError("The 21-brand pilot search requires source_data_yaml order")
    search = load_search(search_path, root=root)
    trials = build_trial_configs(base, search, root=root)
    collisions = [str(item["plan"].run_dir) for item in trials if item["plan"].run_dir.exists()]
    runtime = inspect_yolo_runtime(base_plan)
    summary = {
        "base_config": str(resolved_config),
        "search_config": str(search["_path"]),
        "data": str(base_plan.data_yaml),
        "dataset_fingerprint": checks.get("dataset_fingerprint"),
        "class_order": contract["class_order"],
        "selection_split": "val",
        "selection_metric": search["selection_metric"],
        "test_split_used": False,
        "pilot_epochs": int(search["pilot_epochs"]),
        "run_collisions": collisions,
        "trials": [
            {
                "label": item["label"],
                "run_dir": str(item["plan"].run_dir),
                "overrides": item["overrides"],
            }
            for item in trials
        ],
        "model_constructed": False,
        "training_started": False,
        **runtime,
    }
    return base, search, trials, summary


def _write_trial_config(item: Mapping[str, Any], project: Path) -> Path:
    config_dir = project / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{item['plan'].run_dir.name}.yaml"
    if path.exists():
        raise FileExistsError(f"Generated trial config already exists: {path}")
    path.write_text(
        yaml.safe_dump(item["config"], sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def run_trials(trials: Sequence[Mapping[str, Any]], search: Mapping[str, Any]) -> None:
    collisions = [str(item["plan"].run_dir) for item in trials if item["plan"].run_dir.exists()]
    if collisions:
        raise FileExistsError(f"Pilot run directories already exist: {collisions}")
    project = trials[0]["plan"].project_dir
    for item in trials:
        config_path = _write_trial_config(item, project)
        plan = build_yolo_run_plan(
            dict(item["config"]), config_path, root=ROOT
        )
        capture = YoloTrainingCapture()
        metrics = execute_yolo_training(
            plan,
            expected_version=str(item["config"]["compatibility"]["verified_ultralytics"]),
            missing_dependency_message="Ultralytics is not installed",
            payload_factory=lambda version, label=item["label"]: {
                "trial": label,
                "ultralytics": version,
                "selection_split": "val",
                "test_split_used": False,
            },
            ensure_ascii=False,
            training_capture=capture,
        )
        if item["config"].get("artifacts", {}).get("write_training_summary") is True:
            write_yolo_training_report(plan, metrics, capture)
    summarize_trials(trials, search)


def _best_history_value(path: Path, metric: str) -> tuple[float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Pilot history was not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    candidates: list[tuple[float, int]] = []
    for row_number, row in enumerate(rows, start=1):
        raw = row.get(metric)
        if raw not in (None, ""):
            candidates.append((float(raw), int(float(row.get("epoch", row_number - 1))) + 1))
    if not candidates:
        raise ValueError(f"Metric {metric!r} is unavailable in {path}")
    return max(candidates, key=lambda value: value[0])


def summarize_trials(
    trials: Sequence[Mapping[str, Any]], search: Mapping[str, Any]
) -> dict[str, Any]:
    metric = str(search["selection_metric"])
    rows = []
    for item in trials:
        value, epoch = _best_history_value(item["plan"].run_dir / "results.csv", metric)
        rows.append(
            {
                "label": item["label"],
                "run_dir": str(item["plan"].run_dir),
                "overrides": item["overrides"],
                "best_validation_metric": value,
                "best_epoch": epoch,
            }
        )
    winner = max(rows, key=lambda row: row["best_validation_metric"])
    recommended = deepcopy(dict(trials[0]["base_config"]))
    recommended["train"] = deepcopy(dict(trials[0]["base_config"]["train"]))
    recommended["train"].update(winner["overrides"])
    recommended["project"] = "runs/model1_brand21_tuned"
    recommended["name"] = "yolov8n_100ep_s42_tuned"
    project = trials[0]["plan"].project_dir
    output = project / "tuning_summary.json"
    recommended_path = project / "recommended_full_config.yaml"
    if output.exists() or recommended_path.exists():
        raise FileExistsError("Tuning summary/recommended config already exists")
    summary = {
        "status": "pilot_selection_complete",
        "selection_split": "val",
        "test_split_used": False,
        "metric": metric,
        "winner": winner,
        "trials": rows,
        "recommended_full_config": str(recommended_path),
        "limitation": (
            "This is the best candidate in the declared finite pilot search, "
            "not a claim of a global optimum. Confirm with one full validation run."
        ),
    }
    project.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    recommended_path.write_text(
        yaml.safe_dump(recommended, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _base, search, trials, summary = build_preflight(args.config, args.search)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.run:
        run_trials(trials, search)
    elif args.summarize:
        summarize_trials(trials, search)


if __name__ == "__main__":
    main()
