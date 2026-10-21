"""Preflight and manually run a config-driven YOLO detection KD experiment.

The teacher, ordinary student baseline, and distilled student must share the
prepared dataset's exact training/checkpoint class order. Stable application
registry IDs remain metadata and are never substituted for training IDs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from src.training.yolo_detection_kd import (  # noqa: E402
    DetectionKDConfig,
    KDDetectionTrainer,
    make_portable_detection_checkpoint,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "command",
        choices=(
            "preflight",
            "teacher",
            "resume-teacher",
            "baseline",
            "resume-baseline",
            "student-kd",
            "resume-kd",
            "compare",
            "all",
        ),
    )
    parser.add_argument("--config", default="configs/brand_21brands_yolo_kd.yaml")
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional tuned detector config replacing base_config from the KD YAML.",
    )
    parser.add_argument("--teacher", type=Path, default=None)
    parser.add_argument("--baseline", type=Path, default=None)
    parser.add_argument("--student", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Original KD run weights/last.pt checkpoint, including optimizer state.",
    )
    return parser.parse_args(argv)


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


def kd_config_from_dict(value: Mapping[str, Any]) -> DetectionKDConfig:
    config = DetectionKDConfig(
        class_weight=float(value.get("class_weight", 1.0)),
        dfl_weight=float(value.get("dfl_weight", 1.0)),
        temperature=float(value.get("temperature", 2.0)),
        topk=int(value.get("topk", 64)),
        min_confidence=float(value.get("min_confidence", 0.05)),
    )
    config.validate()
    return config


def load_experiment_config(
    path: str | Path,
    *,
    root: str | Path = ROOT,
    base_config_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load KD/base configs and validate dataset identity without model loading."""

    config_path = resolve_path(path, root)
    if not config_path.is_file():
        raise FileNotFoundError(f"KD experiment config was not found: {config_path}")
    experiment = _read_mapping(config_path)
    required = ("teacher", "student", "kd", "evaluation", "compatibility")
    missing = [key for key in required if not isinstance(experiment.get(key), dict)]
    if missing:
        raise ValueError(f"KD experiment config is missing sections: {missing}")
    base_path = resolve_path(
        base_config_override
        if base_config_override is not None
        else experiment.get("base_config", "configs/brand_21brands_yolo.yaml"),
        root,
    )
    base = load_generic_yolo_config(base_path)
    plan = build_yolo_run_plan(base, base_path, root=root)
    mapping, checks, policy = validate_generic_dataset_metadata(plan.data_yaml, base)
    contract = validate_brand_profile_class_mapping(plan.data_yaml, base, mapping)

    expected_version = str(experiment["compatibility"].get("verified_ultralytics"))
    base_version = str(base.get("compatibility", {}).get("verified_ultralytics"))
    if expected_version != base_version:
        raise ValueError(
            f"KD config expects Ultralytics {expected_version}, "
            f"base config expects {base_version}"
        )
    if contract.get("class_order") != "source_data_yaml":
        raise ValueError("The 21-brand KD workflow requires class_order: source_data_yaml")
    if contract.get("source_to_training_verified") is not True:
        raise ValueError("Source-to-training class mapping was not verified")
    if str(experiment["evaluation"].get("split", "val")) != "val":
        raise ValueError("KD selection and comparison must use only split: val")
    if policy is not None and not policy.get("effective_training_ready", False):
        raise RuntimeError(
            "Prepared dataset is blocked by its configured clean-subset policy: "
            f"{policy.get('failed_safety_requirements', [])}"
        )
    project = resolve_path(experiment.get("project", "runs/detector_distillation"), root)
    context = dict(experiment)
    context.update(
        {
            "_config_path": config_path,
            "_base_path": base_path,
            "_base": base,
            "_plan": plan,
            "_mapping": mapping,
            "_checks": checks,
            "_policy": policy,
            "_contract": contract,
            "_project": project,
        }
    )
    kd_config_from_dict(context["kd"])
    return context


def _ordered_names(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(index): str(name) for index, name in value.items()}
    if isinstance(value, (list, tuple)):
        return {index: str(name) for index, name in enumerate(value)}
    raise ValueError("Checkpoint has no usable class mapping")


def expected_names(context: Mapping[str, Any]) -> dict[int, str]:
    mapping = context["_mapping"]
    return {index: str(mapping[str(index)]) for index in range(len(mapping))}


def validate_checkpoint(
    path: str | Path,
    context: Mapping[str, Any],
    *,
    role: str,
) -> Path:
    """Require an ordinary detect checkpoint with the exact dataset class order."""

    from ultralytics import YOLO

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{role} checkpoint was not found: {checkpoint}")
    model = YOLO(str(checkpoint))
    if model.task != "detect":
        raise ValueError(f"{role} task must be 'detect', found {model.task!r}")
    actual = _ordered_names(model.names)
    expected = expected_names(context)
    if actual != expected:
        raise ValueError(
            f"{role} class order is incompatible: expected={expected}, actual={actual}"
        )
    return checkpoint


def build_train_args(
    context: Mapping[str, Any],
    role: str,
    name: str,
) -> dict[str, Any]:
    if role not in {"teacher", "student"}:
        raise ValueError(f"Unknown training role: {role}")
    role_config = context[role]
    args = dict(context["_base"]["train"])
    args.update(
        {
            "data": str(context["_plan"].data_yaml),
            "project": str(context["_project"]),
            "name": name,
            "epochs": int(role_config["epochs"]),
            "batch": int(role_config["batch"]),
            "patience": int(role_config["patience"]),
            "exist_ok": False,
            "resume": False,
        }
    )
    return args


def _run_dir(context: Mapping[str, Any], name: str) -> Path:
    return Path(context["_project"]) / name


def _assert_new_run(context: Mapping[str, Any], name: str) -> Path:
    run_dir = _run_dir(context, name)
    if run_dir.exists():
        raise FileExistsError(f"Training run already exists; refusing to overwrite: {run_dir}")
    return run_dir


def _checkpoint_or_default(
    value: Path | None,
    context: Mapping[str, Any],
    section: str,
    name_key: str,
    filename: str = "best.pt",
) -> Path:
    if value is not None:
        return value.expanduser().resolve()
    return _run_dir(context, str(context[section][name_key])) / "weights" / filename


def preflight_summary(context: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect runtime and proposed runs without constructing a YOLO model."""

    runtime = inspect_yolo_runtime(context["_plan"])
    teacher_name = str(context["teacher"]["name"])
    baseline_name = str(context["student"]["baseline_name"])
    kd_name = str(context["student"]["kd_name"])
    contract = context["_contract"]
    checks = context["_checks"]
    return {
        "config": str(context["_config_path"]),
        "base_config": str(context["_base_path"]),
        "data": str(context["_plan"].data_yaml),
        "dataset_version": checks.get("dataset_version"),
        "dataset_fingerprint": checks.get("dataset_fingerprint"),
        "class_order": contract["class_order"],
        "source_to_training_mapping_verified": contract[
            "source_to_training_verified"
        ],
        "training_to_registry": contract["training_to_registry"],
        "teacher": {
            "initial_weights": str(context["teacher"]["model"]),
            "run_dir": str(_run_dir(context, teacher_name)),
            "exists": _run_dir(context, teacher_name).exists(),
            "batch": int(context["teacher"]["batch"]),
        },
        "student_baseline": {
            "initial_weights": str(context["student"]["model"]),
            "run_dir": str(_run_dir(context, baseline_name)),
            "exists": _run_dir(context, baseline_name).exists(),
            "batch": int(context["student"]["batch"]),
        },
        "student_kd": {
            "initial_weights": str(context["student"]["model"]),
            "run_dir": str(_run_dir(context, kd_name)),
            "exists": _run_dir(context, kd_name).exists(),
            "batch": int(context["student"]["batch"]),
            "kd": vars(kd_config_from_dict(context["kd"])),
        },
        "evaluation_split": "val",
        "test_evaluation": False,
        "model_constructed": False,
        "training_started": False,
        **runtime,
    }


def train_teacher(context: Mapping[str, Any]) -> Path:
    from ultralytics import YOLO

    name = str(context["teacher"]["name"])
    _assert_new_run(context, name)
    model = YOLO(str(context["teacher"]["model"]))
    model.train(**build_train_args(context, "teacher", name))
    return validate_checkpoint(model.trainer.best, context, role="teacher")


def train_baseline(context: Mapping[str, Any]) -> Path:
    from ultralytics import YOLO

    name = str(context["student"]["baseline_name"])
    _assert_new_run(context, name)
    model = YOLO(str(context["student"]["model"]))
    model.train(**build_train_args(context, "student", name))
    return validate_checkpoint(model.trainer.best, context, role="baseline student")


def validate_kd_resume_checkpoint(
    path: str | Path,
    context: Mapping[str, Any],
) -> tuple[Path, int]:
    """Validate a trusted original KD last.pt before resuming optimizer state."""

    checkpoint_path = Path(path).expanduser().resolve()
    kd_run = _run_dir(context, str(context["student"]["kd_name"])).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"KD resume checkpoint was not found: {checkpoint_path}")
    if checkpoint_path.parent.parent != kd_run:
        raise ValueError(
            f"Resume checkpoint belongs to {checkpoint_path.parent.parent}, expected {kd_run}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed_epoch = int(checkpoint.get("epoch", -1)) + 1
    if completed_epoch <= 0:
        raise ValueError("KD checkpoint has no completed epoch to resume")
    if checkpoint.get("optimizer") is None:
        raise ValueError("KD checkpoint optimizer was stripped; use the original last.pt")
    train_args = checkpoint.get("train_args", {})
    checkpoint_data = Path(str(train_args.get("data", ""))).resolve()
    if checkpoint_data != context["_plan"].data_yaml.resolve():
        raise ValueError(
            f"Resume checkpoint dataset differs: {checkpoint_data} != "
            f"{context['_plan'].data_yaml.resolve()}"
        )
    model = checkpoint.get("ema") or checkpoint.get("model")
    actual = _ordered_names(getattr(model, "names", None))
    if actual != expected_names(context):
        raise ValueError("KD resume checkpoint class order is incompatible")
    return checkpoint_path, completed_epoch


def validate_ordinary_resume_checkpoint(
    path: str | Path,
    context: Mapping[str, Any],
    *,
    role: str,
) -> tuple[Path, int]:
    """Validate teacher/baseline last.pt before an Ultralytics resume."""

    if role == "teacher":
        name = str(context["teacher"]["name"])
    elif role == "baseline":
        name = str(context["student"]["baseline_name"])
    else:
        raise ValueError(f"Unsupported ordinary resume role: {role}")
    checkpoint_path = Path(path).expanduser().resolve()
    expected_run = _run_dir(context, name).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"{role} resume checkpoint was not found: {checkpoint_path}")
    if checkpoint_path.parent.parent != expected_run:
        raise ValueError(
            f"Resume checkpoint belongs to {checkpoint_path.parent.parent}, "
            f"expected {expected_run}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    completed_epoch = int(checkpoint.get("epoch", -1)) + 1
    if completed_epoch <= 0:
        raise ValueError(f"{role} checkpoint has no completed epoch to resume")
    if checkpoint.get("optimizer") is None:
        raise ValueError(
            f"{role} checkpoint optimizer was stripped; use the original last.pt"
        )
    train_args = checkpoint.get("train_args", {})
    checkpoint_data = Path(str(train_args.get("data", ""))).resolve()
    if checkpoint_data != context["_plan"].data_yaml.resolve():
        raise ValueError(
            f"{role} checkpoint dataset differs: {checkpoint_data} != "
            f"{context['_plan'].data_yaml.resolve()}"
        )
    model = checkpoint.get("ema") or checkpoint.get("model")
    if _ordered_names(getattr(model, "names", None)) != expected_names(context):
        raise ValueError(f"{role} resume checkpoint class order is incompatible")
    return checkpoint_path, completed_epoch


def resume_ordinary_training(
    context: Mapping[str, Any],
    path: str | Path,
    *,
    role: str,
) -> Path:
    from ultralytics import YOLO

    checkpoint, completed = validate_ordinary_resume_checkpoint(
        path, context, role=role
    )
    print(f"Resuming {role} after {completed} completed epochs.", flush=True)
    model = YOLO(str(checkpoint))
    model.train(resume=True)
    return validate_checkpoint(model.trainer.best, context, role=role)


def train_student_kd(
    context: Mapping[str, Any],
    teacher_path: str | Path,
    *,
    resume_path: str | Path | None = None,
) -> Path:
    teacher = validate_checkpoint(teacher_path, context, role="teacher")
    name = str(context["student"]["kd_name"])
    run_dir = _run_dir(context, name)
    overrides = build_train_args(context, "student", name)
    if resume_path is None:
        _assert_new_run(context, name)
    else:
        resume, completed = validate_kd_resume_checkpoint(resume_path, context)
        print(f"Resuming KD after {completed} completed epochs.", flush=True)
        overrides["resume"] = str(resume)
    overrides["model"] = str(context["student"]["model"])
    trainer = KDDetectionTrainer(
        overrides=overrides,
        teacher_path=teacher,
        kd_config=kd_config_from_dict(context["kd"]),
    )
    trainer.train()
    best = Path(trainer.best).resolve()
    last = Path(trainer.last).resolve()
    portable_best = make_portable_detection_checkpoint(
        best, run_dir / "weights" / "best_portable.pt"
    )
    if last.is_file():
        make_portable_detection_checkpoint(
            last, run_dir / "weights" / "last_portable.pt"
        )
    return validate_checkpoint(portable_best, context, role="KD student")


def evaluate_checkpoint(
    checkpoint: str | Path,
    context: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    from ultralytics import YOLO

    path = validate_checkpoint(checkpoint, context, role=label)
    output_root = Path(context["_project"]) / "validation_comparison"
    output_dir = output_root / label
    if output_dir.exists():
        raise FileExistsError(f"Validation output already exists: {output_dir}")
    result = YOLO(str(path)).val(
        data=str(context["_plan"].data_yaml),
        split="val",
        imgsz=int(context["_base"]["train"]["imgsz"]),
        batch=int(context["evaluation"]["batch"]),
        device=context["_base"]["train"]["device"],
        project=str(output_root),
        name=label,
        exist_ok=False,
        plots=True,
        verbose=False,
    )
    return {
        "checkpoint": str(path),
        "size_mb": path.stat().st_size / 1_000_000,
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map50": float(result.box.map50),
        "map50_95": float(result.box.map),
        "per_class_map50_95": {
            expected_names(context)[index]: float(value)
            for index, value in enumerate(result.box.maps.tolist())
        },
    }


def write_comparison_report(
    context: Mapping[str, Any],
    teacher: str | Path,
    baseline: str | Path,
    student: str | Path,
) -> Path:
    output = Path(context["_project"]) / "comparison_report.json"
    if output.exists():
        raise FileExistsError(f"Comparison report already exists: {output}")
    report = {
        "protocol": {
            "selection_split": "val",
            "test_used": False,
            "class_order": context["_contract"]["class_order"],
            "kd": vars(kd_config_from_dict(context["kd"])),
        },
        "teacher": evaluate_checkpoint(teacher, context, label="teacher"),
        "student_baseline": evaluate_checkpoint(
            baseline, context, label="student_baseline"
        ),
        "student_kd": evaluate_checkpoint(student, context, label="student_kd"),
    }
    report["kd_minus_baseline"] = {
        metric: report["student_kd"][metric] - report["student_baseline"][metric]
        for metric in ("precision", "recall", "map50", "map50_95")
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    context = load_experiment_config(
        args.config, base_config_override=args.base_config
    )
    summary = preflight_summary(context)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if args.command == "preflight":
        return
    if args.command == "teacher":
        print(f"Teacher checkpoint: {train_teacher(context)}", flush=True)
        return
    if args.command == "resume-teacher":
        if args.resume is None:
            raise SystemExit("resume-teacher requires --resume path/to/weights/last.pt")
        print(
            f"Teacher checkpoint: {resume_ordinary_training(context, args.resume, role='teacher')}",
            flush=True,
        )
        return
    if args.command == "baseline":
        print(f"Baseline checkpoint: {train_baseline(context)}", flush=True)
        return
    if args.command == "resume-baseline":
        if args.resume is None:
            raise SystemExit("resume-baseline requires --resume path/to/weights/last.pt")
        print(
            "Baseline checkpoint: "
            f"{resume_ordinary_training(context, args.resume, role='baseline')}",
            flush=True,
        )
        return
    teacher = _checkpoint_or_default(
        args.teacher, context, "teacher", "name"
    )
    if args.command == "student-kd":
        print(
            f"KD student checkpoint: {train_student_kd(context, teacher)}", flush=True
        )
        return
    if args.command == "resume-kd":
        if args.resume is None:
            raise SystemExit("resume-kd requires --resume path/to/weights/last.pt")
        print(
            "KD student checkpoint: "
            f"{train_student_kd(context, teacher, resume_path=args.resume)}",
            flush=True,
        )
        return
    baseline = _checkpoint_or_default(
        args.baseline, context, "student", "baseline_name"
    )
    student = _checkpoint_or_default(
        args.student,
        context,
        "student",
        "kd_name",
        filename="best_portable.pt",
    )
    if args.command == "compare":
        print(
            f"Comparison report: {write_comparison_report(context, teacher, baseline, student)}",
            flush=True,
        )
        return

    for name in (
        context["teacher"]["name"],
        context["student"]["baseline_name"],
        context["student"]["kd_name"],
    ):
        _assert_new_run(context, str(name))
    teacher = train_teacher(context)
    baseline = train_baseline(context)
    student = train_student_kd(context, teacher)
    print(
        f"Comparison report: {write_comparison_report(context, teacher, baseline, student)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
