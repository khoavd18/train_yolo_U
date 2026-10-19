"""Render deterministic samples from Ultralytics' actual training transforms.

This utility constructs only the training dataset/transform pipeline.  It does
not construct a model, run inference, validate a checkpoint, or start training.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.yolo import build_generic_preflight, resolve_path  # noqa: E402


TRAIN_AUGMENTATION_KEYS = (
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", default="configs/brand_21brands_yolo.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def effective_augmentation(train: Mapping[str, Any]) -> dict[str, Any]:
    """Return the augmentation values that will be passed to model.train()."""

    return {key: train[key] for key in TRAIN_AUGMENTATION_KEYS if key in train}


def _class_name(names: Mapping[int, str] | Sequence[str], class_id: int) -> str:
    if isinstance(names, Mapping):
        return str(names.get(class_id, names.get(str(class_id), f"class_{class_id}")))
    return str(names[class_id]) if 0 <= class_id < len(names) else f"class_{class_id}"


def render_transformed_sample(
    sample: Mapping[str, Any],
    names: Mapping[int, str] | Sequence[str],
) -> tuple[Image.Image, list[dict[str, Any]]]:
    """Draw normalized transformed boxes on one Ultralytics dataset sample."""

    image_tensor = sample["img"]
    array = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if array.dtype != np.uint8:
        maximum = float(array.max()) if array.size else 0.0
        if maximum <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    boxes = sample["bboxes"].detach().cpu().numpy()
    classes = sample["cls"].detach().cpu().numpy().reshape(-1)
    if len(boxes) != len(classes):
        raise ValueError("Ultralytics transformed boxes/classes have different lengths")

    rows: list[dict[str, Any]] = []
    for index, (raw_box, raw_class) in enumerate(zip(boxes, classes)):
        class_id = int(raw_class)
        x_center, y_center, box_width, box_height = (float(value) for value in raw_box)
        values = (x_center, y_center, box_width, box_height)
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"Transformed bbox {index} has a non-finite coordinate")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"Transformed bbox {index} is non-positive")
        x_min = x_center - box_width / 2.0
        y_min = y_center - box_height / 2.0
        x_max = x_center + box_width / 2.0
        y_max = y_center + box_height / 2.0
        tolerance = 1e-5
        if (
            x_min < -tolerance
            or y_min < -tolerance
            or x_max > 1.0 + tolerance
            or y_max > 1.0 + tolerance
        ):
            raise ValueError(f"Transformed bbox {index} lies outside the transformed image")
        pixel_box = (
            max(0, round(x_min * width)),
            max(0, round(y_min * height)),
            min(width - 1, round(x_max * width)),
            min(height - 1, round(y_max * height)),
        )
        color = (
            40 + (class_id * 67) % 190,
            40 + (class_id * 101) % 190,
            40 + (class_id * 149) % 190,
        )
        draw.rectangle(pixel_box, outline=color, width=3)
        label = f"{class_id} {_class_name(names, class_id)}"
        label_box = draw.textbbox((pixel_box[0], pixel_box[1]), label, font=font)
        draw.rectangle(label_box, fill=color)
        draw.text((pixel_box[0], pixel_box[1]), label, fill="black", font=font)
        rows.append(
            {
                "class_id": class_id,
                "class_name": _class_name(names, class_id),
                "xywh_normalized": [x_center, y_center, box_width, box_height],
            }
        )
    return image, rows


def make_contact_sheet(paths: Sequence[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if not images:
        raise ValueError("Cannot make an empty augmentation contact sheet")
    thumb_width, thumb_height = 480, 480
    columns = min(4, len(images))
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    for index, image in enumerate(images):
        image.thumbnail((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_width + (thumb_width - image.width) // 2
        y = (index // columns) * thumb_height + (thumb_height - image.height) // 2
        sheet.paste(image, (x, y))
    sheet.save(output_path, format="JPEG", quality=90, optimize=True)


def generate_preview(
    config_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    count: int = 8,
    seed: int = 42,
    root: str | Path = ROOT,
) -> dict[str, Any]:
    """Generate bbox-aware training-transform previews without loading a model."""

    if count <= 0:
        raise ValueError("Preview count must be positive")
    plan, preflight = build_generic_preflight(config_path, root=root)
    expected_version = str(plan.config["compatibility"]["verified_ultralytics"])

    from ultralytics import __version__ as ultralytics_version
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    if str(ultralytics_version) != expected_version:
        raise RuntimeError(
            f"Expected reviewed Ultralytics {expected_version}, found {ultralytics_version}"
        )
    destination = (
        resolve_path(output_dir, root)
        if output_dir is not None
        else plan.data_yaml.parent / "augmentation_previews"
    )
    if destination.exists():
        raise FileExistsError(f"Augmentation preview output already exists: {destination}")
    destination.mkdir(parents=True)

    data = check_det_dataset(str(plan.data_yaml), autodownload=False)
    overrides = dict(plan.config["train"])
    overrides.update({"task": "detect", "mode": "train", "data": str(plan.data_yaml)})
    cfg = get_cfg(overrides=overrides)
    dataset = build_yolo_dataset(
        cfg,
        data["train"],
        int(cfg.batch),
        data,
        mode="train",
        rect=False,
        stride=32,
    )
    if not len(dataset):
        raise ValueError("Prepared training dataset is empty")

    selection_rng = random.Random(seed)
    indices = selection_rng.sample(range(len(dataset)), min(count, len(dataset)))
    preview_paths: list[Path] = []
    preview_rows: list[dict[str, Any]] = []
    for preview_index, dataset_index in enumerate(indices, start=1):
        sample_seed = seed + preview_index - 1
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        try:
            import torch

            torch.manual_seed(sample_seed)
        except ImportError:
            pass
        sample = dataset[dataset_index]
        rendered, boxes = render_transformed_sample(sample, data["names"])
        preview_path = destination / f"transform_{preview_index:02d}.jpg"
        rendered.save(preview_path, format="JPEG", quality=92, optimize=True)
        preview_paths.append(preview_path)
        preview_rows.append(
            {
                "preview": preview_path.name,
                "dataset_index": dataset_index,
                "sample_seed": sample_seed,
                "primary_source": str(sample.get("im_file", "")),
                "transformed_shape": list(rendered.size),
                "box_count": len(boxes),
                "boxes": boxes,
            }
        )

    make_contact_sheet(preview_paths, destination / "contact_sheet.jpg")
    manifest = {
        "config": str(plan.config_path),
        "data": str(plan.data_yaml),
        "ultralytics": str(ultralytics_version),
        "pipeline": "ultralytics.build_yolo_dataset(mode='train')",
        "model_constructed": False,
        "training_started": False,
        "inference_started": False,
        "test_split_read": False,
        "seed": seed,
        "effective_training_augmentation": effective_augmentation(plan.config["train"]),
        "validation_augmentation": False,
        "preflight_dataset_fingerprint": preflight.get("dataset_fingerprint"),
        "previews": preview_rows,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = generate_preview(
        args.config,
        output_dir=args.output_dir,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
