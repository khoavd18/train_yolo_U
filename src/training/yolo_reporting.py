"""Reports captured from the normal Ultralytics detector-training workflow.

This module never calls ``model.val``.  Per-class metrics come from the
``DetMetrics`` object returned by ``YOLO.train`` after Ultralytics performs its
normal final validation of ``best.pt``; epoch history comes from the native
``results.csv`` written during training.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.yolo import (
    AUGMENTATION_SETTING_KEYS,
    OPTIMIZER_SETTING_KEYS,
    YoloRunPlan,
    load_dataset_class_mapping,
    load_dataset_identity,
    validate_brand_profile_class_mapping,
)


CHECKPOINT_SELECTION_CRITERION = (
    "0.1 * validation_mAP50 + 0.9 * validation_mAP50_95"
)
CHECKPOINT_SELECTION_WEIGHTS = {
    "validation_precision": 0.0,
    "validation_recall": 0.0,
    "validation_mAP50": 0.1,
    "validation_mAP50_95": 0.9,
}


def _number(value: Any) -> float | None:
    """Convert scalar-like values to finite JSON numbers."""

    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    scalar = _number(value)
    return scalar if scalar is not None else str(value)


@dataclass
class YoloTrainingCapture:
    """Capture exact checkpoint choice and runtime data through normal callbacks."""

    best_epoch: int | None = None
    best_fitness: float | None = None
    epoch_runtime: list[dict[str, Any]] = field(default_factory=list)
    _epoch_started: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _recorded_epochs: set[int] = field(default_factory=set, init=False, repr=False)

    def attach(self, model: Any) -> None:
        """Attach callbacks without constructing another model or evaluator."""

        add_callback = getattr(model, "add_callback", None)
        if not callable(add_callback):
            raise TypeError("Ultralytics model does not expose add_callback")
        add_callback("on_train_epoch_start", self.on_train_epoch_start)
        add_callback("on_model_save", self.on_model_save)
        add_callback("on_fit_epoch_end", self.on_fit_epoch_end)

    def on_train_epoch_start(self, trainer: Any) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        if epoch > 0:
            self._epoch_started[epoch] = time.perf_counter()
        try:
            import torch

            device = getattr(trainer, "device", None)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)
        except (RuntimeError, TypeError, ValueError):
            # Runtime metrics are best-effort and must never alter training.
            pass

    def on_model_save(self, trainer: Any) -> None:
        """Mirror Ultralytics' exact ``best_fitness == fitness`` save condition."""

        best = getattr(trainer, "best_fitness", None)
        current = getattr(trainer, "fitness", None)
        if best is None or current is None:
            return
        try:
            is_best = bool(best == current)
        except (TypeError, ValueError):
            is_best = False
        if is_best:
            self.best_epoch = int(getattr(trainer, "epoch", -1)) + 1
            self.best_fitness = _number(current)

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Record one row per training epoch, ignoring final-eval callback reuse."""

        epoch = int(getattr(trainer, "epoch", -1)) + 1
        if epoch <= 0 or epoch in self._recorded_epochs:
            return
        self._recorded_epochs.add(epoch)

        epoch_time = _number(getattr(trainer, "epoch_time", None))
        if epoch_time is None and epoch in self._epoch_started:
            epoch_time = time.perf_counter() - self._epoch_started[epoch]
        record: dict[str, Any] = {
            "epoch": epoch,
            "epoch_time_seconds": epoch_time,
            "selection_fitness": _number(getattr(trainer, "fitness", None)),
            "gpu_peak_allocated_bytes": None,
            "gpu_peak_reserved_bytes": None,
        }
        try:
            import torch

            device = getattr(trainer, "device", None)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                record["gpu_peak_allocated_bytes"] = int(
                    torch.cuda.max_memory_allocated(device)
                )
                record["gpu_peak_reserved_bytes"] = int(
                    torch.cuda.max_memory_reserved(device)
                )
        except (RuntimeError, TypeError, ValueError):
            pass
        self.epoch_runtime.append(record)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_epoch": self.best_epoch,
            "best_fitness": self.best_fitness,
            "best_epoch_reliably_identified": self.best_epoch is not None,
            "epoch_runtime": list(self.epoch_runtime),
        }


def read_results_csv(path: str | Path) -> list[dict[str, str]]:
    """Read native Ultralytics history while normalizing padded headers."""

    results_path = Path(path)
    if not results_path.is_file():
        raise FileNotFoundError(f"Ultralytics results.csv was not found: {results_path}")
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None
            }
            for row in csv.DictReader(handle)
        ]
    if not rows:
        raise ValueError(f"Ultralytics results.csv has no completed epochs: {results_path}")
    return rows


def _row_number(row: Mapping[str, str], key: str) -> float | None:
    return _number(row.get(key))


def _row_epoch(row: Mapping[str, str]) -> int:
    value = _row_number(row, "epoch")
    if value is None or value < 1 or not float(value).is_integer():
        raise ValueError(f"Ultralytics results.csv has an invalid 1-based epoch: {row.get('epoch')!r}")
    return int(value)


def _history_record(
    row: Mapping[str, str], runtime: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    epoch = _row_epoch(row)
    runtime_row = runtime.get(epoch, {})
    return {
        "epoch": epoch,
        "elapsed_seconds": _row_number(row, "time"),
        "epoch_time_seconds": _number(runtime_row.get("epoch_time_seconds")),
        "gpu_peak_allocated_bytes": runtime_row.get("gpu_peak_allocated_bytes"),
        "gpu_peak_reserved_bytes": runtime_row.get("gpu_peak_reserved_bytes"),
        "train_box_loss": _row_number(row, "train/box_loss"),
        "train_classification_loss": _row_number(row, "train/cls_loss"),
        "train_dfl_loss": _row_number(row, "train/dfl_loss"),
        "validation_box_loss": _row_number(row, "val/box_loss"),
        "validation_classification_loss": _row_number(row, "val/cls_loss"),
        "validation_dfl_loss": _row_number(row, "val/dfl_loss"),
        "validation_precision": _row_number(row, "metrics/precision(B)"),
        "validation_recall": _row_number(row, "metrics/recall(B)"),
        "validation_mAP50": _row_number(row, "metrics/mAP50(B)"),
        "validation_mAP50_95": _row_number(row, "metrics/mAP50-95(B)"),
        "learning_rates": {
            key: _row_number(row, key)
            for key in sorted(row)
            if key.startswith("lr/")
        },
    }


def _overall_metrics(metrics: Any) -> dict[str, float | None]:
    box = getattr(metrics, "box", None)
    if box is not None:
        precision = _number(getattr(box, "mp", None))
        recall = _number(getattr(box, "mr", None))
        map50 = _number(getattr(box, "map50", None))
        map50_95 = _number(getattr(box, "map", None))
    else:
        results = getattr(metrics, "results_dict", {}) or {}
        precision = _number(results.get("metrics/precision(B)"))
        recall = _number(results.get("metrics/recall(B)"))
        map50 = _number(results.get("metrics/mAP50(B)"))
        map50_95 = _number(results.get("metrics/mAP50-95(B)"))
    fitness = (
        0.1 * map50 + 0.9 * map50_95
        if map50 is not None and map50_95 is not None
        else None
    )
    return {
        "precision": precision,
        "recall": recall,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "selection_fitness": fitness,
    }


def _normalize_metric_names(names: Any) -> tuple[str, ...] | None:
    if isinstance(names, Mapping):
        try:
            indexed = {int(key): str(value) for key, value in names.items()}
        except (TypeError, ValueError):
            return None
        if sorted(indexed) != list(range(len(indexed))):
            return None
        return tuple(indexed[index] for index in range(len(indexed)))
    if isinstance(names, (list, tuple)):
        return tuple(str(value) for value in names)
    return None


def per_brand_metrics(
    metrics: Any,
    mapping: Mapping[str, str],
    checks: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Extract class metrics using ``ap_class_index`` rather than class ID as row index."""

    expected_names = tuple(mapping[str(index)] for index in range(len(mapping)))
    metric_names = _normalize_metric_names(getattr(metrics, "names", None))
    if metric_names is not None and metric_names != expected_names:
        raise ValueError(
            "Final validation metric class order differs from the prepared dataset: "
            f"expected={expected_names}, actual={metric_names}"
        )
    box = getattr(metrics, "box", None)
    ap_classes = getattr(box, "ap_class_index", ()) if box is not None else ()
    position_by_class = {int(class_id): index for index, class_id in enumerate(ap_classes)}
    validation_split = checks.get("splits", {}).get("val", {})
    validation_counts = validation_split.get("per_brand") or validation_split.get(
        "per_class", {}
    )

    rows: list[dict[str, Any]] = []
    for class_id, brand in enumerate(expected_names):
        position = position_by_class.get(class_id)
        result: Sequence[Any] | None = None
        if position is not None and box is not None:
            result = box.class_result(position)
        counts = validation_counts.get(brand, {})
        rows.append(
            {
                "class_id": class_id,
                "brand": brand,
                "validation_images": int(counts.get("images", 0)),
                "validation_instances": int(counts.get("instances", 0)),
                "precision": _number(result[0]) if result is not None else None,
                "recall": _number(result[1]) if result is not None else None,
                "mAP50": _number(result[2]) if result is not None else None,
                "mAP50_95": _number(result[3]) if result is not None else None,
            }
        )
    return rows


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _artifact_inventory(run_dir: Path, report_paths: Mapping[str, Path]) -> dict[str, Any]:
    paths = {
        "history_csv": run_dir / "results.csv",
        "curves": run_dir / "results.png",
        "best_checkpoint": run_dir / "weights" / "best.pt",
        "last_checkpoint": run_dir / "weights" / "last.pt",
        "confusion_matrix": run_dir / "confusion_matrix.png",
        "confusion_matrix_normalized": run_dir / "confusion_matrix_normalized.png",
        "precision_recall_curve": run_dir / "PR_curve.png",
        **report_paths,
    }
    return {
        name: {"path": str(path.resolve()), "exists": path.is_file()}
        for name, path in paths.items()
    }


def _run_artifact_path(run_dir: Path, value: Any, field: str) -> Path:
    component = Path(str(value))
    if component.is_absolute() or component.name != str(value):
        raise ValueError(f"Training artifact {field!r} must be one path-safe filename")
    return run_dir / component


def _write_per_brand_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "class_id",
        "brand",
        "validation_images",
        "validation_instances",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any) -> str:
    number = _number(value)
    return "unavailable" if number is None else f"{number:.6f}"


def _markdown_summary(summary: Mapping[str, Any]) -> str:
    selection = summary["checkpoint_selection"]
    best = summary["best_checkpoint_validation"]
    final = summary["final_epoch_validation"]
    final_training = summary["final_epoch_training"]
    final_runtime = summary["history"][-1]
    lines = [
        "# Detector training summary",
        "",
        f"- Dataset version: `{summary['dataset']['version']}`",
        f"- Dataset fingerprint: `{summary['dataset']['fingerprint']}`",
        f"- Brand profile: `{summary['brand_profile']}`",
        f"- Training/checkpoint class order: `{summary['class_order']}`",
        f"- Epochs completed: {summary['epochs_completed']}",
        f"- Best-checkpoint criterion: `{selection['criterion']}` (validation only)",
        f"- Best epoch: {selection['best_epoch'] if selection['best_epoch'] is not None else 'unavailable'}",
        f"- Best epoch reliably identified: {selection['best_epoch_reliably_identified']}",
        "- Test metrics used for checkpoint selection: False",
        "",
        "## Training IDs and stable registry IDs",
        "",
        "| Training ID | Training name | Registry ID | Registry name | Source ID | Source name |",
        "|---:|---|---:|---|---:|---|",
    ]
    for row in summary["registry_stable_mapping"]:
        lines.append(
            f"| {row['training_id']} | {row['training_name']} | {row['registry_id']} | "
            f"{row['registry_name']} | "
            f"{row['source_id'] if row['source_id'] is not None else ''} | "
            f"{row['source_name'] if row['source_name'] is not None else ''} |"
        )
    lines.extend(
        [
        "",
        "## Best checkpoint versus final epoch",
        "",
        "| Metric | Best checkpoint | Final epoch |",
        "|---|---:|---:|",
        f"| Precision | {_format_metric(best.get('precision'))} | {_format_metric(final.get('precision'))} |",
        f"| Recall | {_format_metric(best.get('recall'))} | {_format_metric(final.get('recall'))} |",
        f"| mAP50 | {_format_metric(best.get('mAP50'))} | {_format_metric(final.get('mAP50'))} |",
        f"| mAP50-95 | {_format_metric(best.get('mAP50_95'))} | {_format_metric(final.get('mAP50_95'))} |",
        "",
        "## Final epoch training/runtime",
        "",
        f"- Train box/classification/DFL loss: {_format_metric(final_training.get('box_loss'))} / "
        f"{_format_metric(final_training.get('classification_loss'))} / "
        f"{_format_metric(final_training.get('dfl_loss'))}",
        f"- Validation box/classification/DFL loss: {_format_metric(final.get('validation_box_loss'))} / "
        f"{_format_metric(final.get('validation_classification_loss'))} / "
        f"{_format_metric(final.get('validation_dfl_loss'))}",
        f"- Learning rates: `{json.dumps(final_training.get('learning_rates', {}), sort_keys=True)}`",
        f"- Epoch time (seconds): {_format_metric(final_runtime.get('epoch_time_seconds'))}",
        f"- GPU peak reserved bytes: {final_runtime.get('gpu_peak_reserved_bytes')}",
        "",
        "## Strongest validation brands",
        "",
        "| Brand | Validation images | Validation instances | mAP50-95 |",
        "|---|---:|---:|---:|",
        ]
    )
    for row in summary["strongest_brands"]:
        lines.append(
            f"| {row['brand']} | {row['validation_images']} | {row['validation_instances']} | "
            f"{_format_metric(row['mAP50_95'])} |"
        )
    lines.extend(
        [
            "",
            "## Weakest validation brands",
            "",
            "| Brand | Validation images | Validation instances | mAP50-95 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary["weakest_brands"]:
        lines.append(
            f"| {row['brand']} | {row['validation_images']} | {row['validation_instances']} | "
            f"{_format_metric(row['mAP50_95'])} |"
        )
    lines.extend(["", "## Artifacts", ""])
    for name, artifact in summary["artifacts"].items():
        lines.append(f"- {name}: `{artifact['path']}` (exists={artifact['exists']})")
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in summary["known_limitations"])
    return "\n".join(lines) + "\n"


def write_yolo_training_report(
    plan: YoloRunPlan,
    metrics: Any,
    capture: YoloTrainingCapture,
) -> dict[str, Any]:
    """Write future-run summaries without performing another validation pass."""

    if metrics is None or getattr(metrics, "box", None) is None:
        raise ValueError(
            "YOLO.train did not return detector validation metrics; refusing to invent a report"
        )
    run_dir = plan.run_dir.resolve()
    artifacts_config = plan.config.get("artifacts", {})
    results_path = _run_artifact_path(
        run_dir, artifacts_config.get("history_csv", "results.csv"), "history_csv"
    )
    per_brand_csv = _run_artifact_path(
        run_dir,
        artifacts_config.get("per_brand_validation_csv", "per_brand_validation.csv"),
        "per_brand_validation_csv",
    )
    per_brand_json = _run_artifact_path(
        run_dir,
        artifacts_config.get("per_brand_validation_json", "per_brand_validation.json"),
        "per_brand_validation_json",
    )
    summary_markdown = _run_artifact_path(
        run_dir,
        artifacts_config.get("final_summary_markdown", "training_summary.md"),
        "final_summary_markdown",
    )
    summary_json = _run_artifact_path(
        run_dir,
        artifacts_config.get("final_summary_json", "training_summary.json"),
        "final_summary_json",
    )
    generated_paths = (per_brand_csv, per_brand_json, summary_markdown, summary_json)
    existing = [str(path) for path in generated_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Training report output already exists; refusing overwrite: {existing}")

    rows = read_results_csv(results_path)
    capture_payload = capture.as_dict()
    runtime_by_epoch = {
        int(row["epoch"]): row for row in capture_payload["epoch_runtime"]
    }
    history = [_history_record(row, runtime_by_epoch) for row in rows]
    final_history = history[-1]
    completed_epoch = final_history["epoch"]
    mapping = load_dataset_class_mapping(plan.data_yaml)
    class_mapping_contract = validate_brand_profile_class_mapping(
        plan.data_yaml, plan.config, mapping
    )
    checks = _load_json_object(plan.data_yaml.parent / "dataset_checks.json")
    if not checks:
        raise FileNotFoundError(
            f"Prepared dataset metadata is missing: {plan.data_yaml.parent / 'dataset_checks.json'}"
        )
    brand_rows = per_brand_metrics(metrics, mapping, checks)
    best_overall = _overall_metrics(metrics)
    final_overall = {
        "epoch": completed_epoch,
        "precision": final_history["validation_precision"],
        "recall": final_history["validation_recall"],
        "mAP50": final_history["validation_mAP50"],
        "mAP50_95": final_history["validation_mAP50_95"],
        "validation_box_loss": final_history["validation_box_loss"],
        "validation_classification_loss": final_history[
            "validation_classification_loss"
        ],
        "validation_dfl_loss": final_history["validation_dfl_loss"],
    }
    final_overall["selection_fitness"] = (
        0.1 * final_overall["mAP50"] + 0.9 * final_overall["mAP50_95"]
        if final_overall["mAP50"] is not None
        and final_overall["mAP50_95"] is not None
        else None
    )

    ranked = sorted(
        (row for row in brand_rows if row["mAP50_95"] is not None),
        key=lambda row: (float(row["mAP50_95"]), row["brand"]),
        reverse=True,
    )
    strongest = ranked[: min(3, len(ranked))]
    weakest = sorted(ranked[-min(3, len(ranked)) :], key=lambda row: row["mAP50_95"])

    identity = load_dataset_identity(plan.data_yaml, checks)
    mapping_document = _load_json_object(plan.data_yaml.parent / "mapping.json")
    report_paths = {
        "per_brand_validation_csv": per_brand_csv,
        "per_brand_validation_json": per_brand_json,
        "summary_markdown": summary_markdown,
        "summary_json": summary_json,
    }
    limitations = [
        "Per-brand metrics describe the normal final validation of best.pt; no extra validation run was added.",
        "Per-brand metrics for the final epoch are unavailable without another evaluation and are not invented.",
        "The held-out test split was not evaluated and was not used for checkpoint selection or tuning.",
        "Validation metrics do not establish annotation completeness or deployment quality.",
    ]
    if capture.best_epoch is None:
        limitations.append(
            "The exact best epoch was not captured; stripped checkpoint metadata and rounded results.csv were not used to guess it."
        )
    if not any(row.get("gpu_peak_reserved_bytes") is not None for row in capture.epoch_runtime):
        limitations.append("GPU peak memory was unavailable from the training callbacks.")
    policy_limitation = (
        checks.get("dataset_policy_limitation")
        or checks.get("known_limitation")
        or checks.get("augmentation_metadata_limitation")
    )
    if policy_limitation:
        limitations.append(str(policy_limitation))
    if class_mapping_contract["class_order"] == "source_data_yaml":
        limitations.append(
            "Training/checkpoint class IDs follow source data.yaml order; stable registry IDs are separate and must be translated with registry_stable_mapping."
        )
    augmentation_audit = checks.get("augmentation_audit")
    if isinstance(augmentation_audit, Mapping):
        offline_mosaic = augmentation_audit.get("offline_mosaic_status")
        if offline_mosaic not in (None, "confirmed_absent"):
            limitations.append(
                "Offline augmentation audit: "
                + json.dumps(_json_value(augmentation_audit), ensure_ascii=False, sort_keys=True)
            )

    summary: dict[str, Any] = {
        "status": "completed",
        "run_dir": str(run_dir),
        "dataset": {
            **identity,
            "data_yaml": str(plan.data_yaml),
            "check_status": checks.get("status"),
            "training_ready": checks.get("training_ready"),
            "augmentation_audit": _json_value(checks.get("augmentation_audit")),
        },
        "brand_profile": class_mapping_contract["brand_profile"],
        "class_order": class_mapping_contract["class_order"],
        "class_mapping": mapping,
        "registry_stable_mapping": class_mapping_contract["training_to_registry"],
        "source_to_training_mapping_verified": class_mapping_contract[
            "source_to_training_verified"
        ],
        "source_to_training_mapping": mapping_document,
        "training_config_path": str(plan.config_path),
        "training_config": _json_value(plan.config),
        "ultralytics_version": str(
            plan.config.get("compatibility", {}).get("verified_ultralytics")
        ),
        "effective_optimizer": {
            key: plan.train_args.get(key) for key in OPTIMIZER_SETTING_KEYS
        },
        "effective_augmentation": {
            key: plan.train_args.get(key) for key in AUGMENTATION_SETTING_KEYS
        },
        "epochs_completed": completed_epoch,
        "checkpoint_selection": {
            "criterion": CHECKPOINT_SELECTION_CRITERION,
            "weights": CHECKPOINT_SELECTION_WEIGHTS,
            "validation_only": True,
            "test_metrics_used": False,
            "tie_behavior": "later equal-fitness epoch overwrites best.pt",
            "best_epoch": capture.best_epoch,
            "best_fitness": capture.best_fitness,
            "best_epoch_reliably_identified": capture.best_epoch is not None,
            "checkpoint_epoch_metadata_used": False,
        },
        "best_checkpoint_validation": best_overall,
        "final_epoch_validation": final_overall,
        "final_epoch_training": {
            "box_loss": final_history["train_box_loss"],
            "classification_loss": final_history["train_classification_loss"],
            "dfl_loss": final_history["train_dfl_loss"],
            "learning_rates": final_history["learning_rates"],
        },
        "history": history,
        "per_brand_validation": brand_rows,
        "strongest_brands": strongest,
        "weakest_brands": weakest,
        "normal_training_validation_reused": True,
        "extra_evaluation_runs": 0,
        "test_evaluated": False,
        "model_inference_started_by_reporter": False,
        "known_limitations": limitations,
    }
    summary["artifacts"] = _artifact_inventory(run_dir, report_paths)

    per_brand_payload = {
        "weights": str(run_dir / "weights" / "best.pt"),
        "data": str(plan.data_yaml),
        "split": "val",
        "class_order": class_mapping_contract["class_order"],
        "registry_stable_mapping": class_mapping_contract["training_to_registry"],
        "normal_training_validation_reused": True,
        "extra_evaluation_runs": 0,
        "test_evaluated": False,
        "per_brand": brand_rows,
    }
    _write_per_brand_csv(per_brand_csv, brand_rows)
    per_brand_json.write_text(
        json.dumps(per_brand_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Paths are expected outputs at inventory time; mark them present in the
    # serialized summary that is about to be written.
    for name in (
        "per_brand_validation_csv",
        "per_brand_validation_json",
        "summary_markdown",
        "summary_json",
    ):
        summary["artifacts"][name]["exists"] = True
    summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_markdown.write_text(_markdown_summary(summary), encoding="utf-8")
    return summary
