"""Calibrate detector inference settings on validation without reading test."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.yolo import (  # noqa: E402
    build_yolo_run_plan,
    inspect_yolo_runtime,
    load_generic_yolo_config,
    resolve_path,
    validate_brand_profile_class_mapping,
    validate_generic_dataset_metadata,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/brand_21brands_yolo.yaml")
    parser.add_argument(
        "--search", default="configs/brand_21brands_inference_search.yaml"
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument(
        "--output-dir", default="runs/model1_brand21_inference_calibration"
    )
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def load_search(path: str | Path, *, root: str | Path = ROOT) -> dict[str, Any]:
    search_path = resolve_path(path, root)
    if not search_path.is_file():
        raise FileNotFoundError(f"Inference search config was not found: {search_path}")
    search = _load_yaml(search_path)
    if search.get("test_split_used") is not False:
        raise ValueError("Inference calibration must explicitly keep test_split_used: false")
    if search.get("objective") not in {"f1", "map50_95", "recall"}:
        raise ValueError("objective must be one of: f1, map50_95, recall")
    grid = search.get("grid")
    if not isinstance(grid, dict):
        raise ValueError("Inference search must define a grid mapping")
    for key in ("imgsz", "conf", "iou"):
        values = grid.get(key)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Inference grid {key!r} must be a non-empty list")
    if any(int(value) <= 0 for value in grid["imgsz"]):
        raise ValueError("Every imgsz must be positive")
    if any(not 0.0 < float(value) < 1.0 for value in grid["conf"]):
        raise ValueError("Every conf must be between zero and one")
    if any(not 0.0 < float(value) < 1.0 for value in grid["iou"]):
        raise ValueError("Every iou must be between zero and one")
    search["_path"] = search_path
    return search


def _ordered_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(name) for key, name in value.items()}
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    raise ValueError("Checkpoint has no usable class mapping")


def _checkpoint_metadata(path: Path) -> tuple[str | None, dict[int, str]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise ValueError("Checkpoint has no model/EMA payload")
    task = getattr(model, "task", None)
    names = _ordered_names(getattr(model, "names", None))
    return str(task) if task is not None else None, names


def build_preflight(
    config_path: str | Path,
    search_path: str | Path,
    weights: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved_config = resolve_path(config_path, root)
    config = load_generic_yolo_config(resolved_config)
    plan = build_yolo_run_plan(config, resolved_config, root=root)
    mapping, checks, _policy = validate_generic_dataset_metadata(plan.data_yaml, config)
    contract = validate_brand_profile_class_mapping(plan.data_yaml, config, mapping)
    search = load_search(search_path, root=root)
    checkpoint = resolve_path(weights, root)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Detector checkpoint was not found: {checkpoint}")
    task, checkpoint_names = _checkpoint_metadata(checkpoint)
    expected_names = {int(index): name for index, name in mapping.items()}
    if task not in {None, "detect"}:
        raise ValueError(f"Checkpoint task must be detect, found {task!r}")
    if checkpoint_names != expected_names:
        raise ValueError(
            f"Checkpoint class order differs: expected={expected_names}, "
            f"actual={checkpoint_names}"
        )
    output = resolve_path(output_dir, root)
    grid = search["grid"]
    trials = [
        {"imgsz": int(imgsz), "conf": float(conf), "iou": float(iou)}
        for imgsz, conf, iou in product(grid["imgsz"], grid["conf"], grid["iou"])
    ]
    runtime = inspect_yolo_runtime(plan)
    summary = {
        "config": str(resolved_config),
        "search_config": str(search["_path"]),
        "weights": str(checkpoint),
        "data": str(plan.data_yaml),
        "dataset_fingerprint": checks.get("dataset_fingerprint"),
        "class_order": contract["class_order"],
        "selection_split": "val",
        "objective": search["objective"],
        "test_split_used": False,
        "output_dir": str(output),
        "output_exists": output.exists(),
        "trial_count": len(trials),
        "trials": trials,
        "model_constructed": False,
        "inference_started": False,
        **runtime,
    }
    context = {
        "config": config,
        "plan": plan,
        "mapping": mapping,
        "search": search,
        "checkpoint": checkpoint,
        "output": output,
        "trials": trials,
    }
    return context, search, summary


def _objective_value(row: Mapping[str, Any], objective: str) -> float:
    return float(row[objective])


def metrics_at_confidence(metric: Any, confidence: float) -> tuple[float, float, float]:
    """Return macro precision/recall/F1 at the exact deployment threshold."""

    axis = np.asarray(metric.px, dtype=float)
    precision_curve = np.asarray(metric.p_curve, dtype=float)
    recall_curve = np.asarray(metric.r_curve, dtype=float)
    if axis.ndim != 1 or not len(axis):
        raise ValueError("Validation result has no confidence axis")
    if precision_curve.ndim != 2 or recall_curve.shape != precision_curve.shape:
        raise ValueError("Validation result has invalid precision/recall curves")
    index = int(np.abs(axis - float(confidence)).argmin())
    precision_per_class = precision_curve[:, index]
    recall_per_class = recall_curve[:, index]
    f1_per_class = (
        2.0
        * precision_per_class
        * recall_per_class
        / np.maximum(precision_per_class + recall_per_class, 1e-12)
    )
    return (
        float(precision_per_class.mean()),
        float(recall_per_class.mean()),
        float(f1_per_class.mean()),
    )


def run_calibration(context: Mapping[str, Any]) -> dict[str, Any]:
    from ultralytics import YOLO

    output = Path(context["output"])
    if output.exists():
        raise FileExistsError(f"Calibration output already exists: {output}")
    output.mkdir(parents=True)
    model = YOLO(str(context["checkpoint"]))
    rows = []
    for index, trial in enumerate(context["trials"], start=1):
        name = (
            f"trial_{index:02d}_img{trial['imgsz']}_"
            f"conf{trial['conf']:.2f}_iou{trial['iou']:.2f}"
        ).replace(".", "p")
        result = model.val(
            data=str(context["plan"].data_yaml),
            split="val",
            imgsz=trial["imgsz"],
            conf=trial["conf"],
            iou=trial["iou"],
            batch=int(context["config"]["train"]["batch"]),
            device=context["config"]["train"]["device"],
            project=str(output / "trials"),
            name=name,
            exist_ok=False,
            plots=False,
            save_json=False,
            verbose=False,
        )
        precision, recall, f1 = metrics_at_confidence(
            result.box, trial["conf"]
        )
        speed = getattr(result, "speed", {}) or {}
        rows.append(
            {
                **trial,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": float(result.box.map50),
                "map50_95": float(result.box.map),
                "inference_ms_per_image": float(speed.get("inference", math.nan)),
                "output": str(output / "trials" / name),
            }
        )
    objective = str(context["search"]["objective"])
    winner = max(
        rows,
        key=lambda row: (
            _objective_value(row, objective),
            row["map50_95"],
            -row["inference_ms_per_image"]
            if math.isfinite(row["inference_ms_per_image"])
            else float("-inf"),
        ),
    )
    report = {
        "status": "validation_calibration_complete",
        "weights": str(context["checkpoint"]),
        "selection_split": "val",
        "test_split_used": False,
        "objective": objective,
        "winner": winner,
        "trials": rows,
        "limitation": (
            "Winner is optimal only within this declared finite validation grid. "
            "Run the held-out test once after all model and threshold decisions are frozen."
        ),
    }
    (output / "inference_calibration.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context, _search, summary = build_preflight(
        args.config, args.search, args.weights, args.output_dir
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.run:
        run_calibration(context)


if __name__ == "__main__":
    main()
