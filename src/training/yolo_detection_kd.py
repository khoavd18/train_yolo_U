"""Knowledge distillation for Ultralytics YOLO detection models.

The project runtime consumes ordinary ``detect`` checkpoints, so this module
distils class logits and DFL box distributions without changing the deployed
task to segmentation.  It targets the repository-pinned Ultralytics 8.3.155
API and deliberately keeps the teacher outside the student ``state_dict``.
"""

from __future__ import annotations

import csv
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK
from ultralytics.utils.torch_utils import de_parallel


@dataclass(frozen=True)
class DetectionKDConfig:
    """Weights and filtering used by detection knowledge distillation."""

    class_weight: float = 1.0
    dfl_weight: float = 1.0
    temperature: float = 2.0
    topk: int = 64
    min_confidence: float = 0.05

    def validate(self) -> None:
        if self.class_weight < 0 or self.dfl_weight < 0:
            raise ValueError("KD loss weights must be non-negative")
        if self.class_weight == 0 and self.dfl_weight == 0:
            raise ValueError("At least one KD loss weight must be positive")
        if self.temperature <= 0:
            raise ValueError("KD temperature must be positive")
        if self.topk <= 0:
            raise ValueError("KD topk must be positive")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("KD min_confidence must be in [0, 1]")


@dataclass
class _KDState:
    teacher_path: Path
    config: DetectionKDConfig
    teacher: DetectionModel | None = None


# Weak keys avoid serializing the large teacher as part of student/EMA
# checkpoints.  Unlike a single module-level teacher, this supports multiple
# trainer instances in one process without sharing incompatible state.
_KD_STATES: weakref.WeakKeyDictionary[DetectionModel, _KDState] = weakref.WeakKeyDictionary()


def parse_detection_predictions(
    predictions: Any,
    *,
    num_classes: int,
    reg_max: int,
) -> tuple[Tensor, Tensor]:
    """Return raw DFL box and class logits in ``[B, C, anchors]`` form."""

    features = predictions[1] if isinstance(predictions, tuple) else predictions
    if isinstance(features, dict):
        # Newer Ultralytics detection heads can expose one-to-many outputs.
        features = features.get("one2many", features.get("one2one"))
    if not isinstance(features, (list, tuple)) or not features:
        raise TypeError("Expected raw YOLO detection feature maps")
    if not all(isinstance(feature, Tensor) and feature.ndim == 4 for feature in features):
        raise ValueError("Every YOLO detection feature must have shape [B, C, H, W]")
    batch_size = features[0].shape[0]
    if any(feature.shape[0] != batch_size for feature in features):
        raise ValueError("YOLO detection features use inconsistent batch sizes")
    concatenated = torch.cat(
        [feature.reshape(batch_size, feature.shape[1], -1) for feature in features],
        dim=2,
    )
    expected_channels = reg_max * 4 + num_classes
    if concatenated.shape[1] != expected_channels:
        raise ValueError(
            "Unexpected detection-head channels: "
            f"expected {expected_channels}, found {concatenated.shape[1]}"
        )
    box_logits, class_logits = concatenated.split((reg_max * 4, num_classes), dim=1)
    return box_logits, class_logits


def detection_kd_loss(
    student_predictions: Any,
    teacher_predictions: Any,
    *,
    num_classes: int,
    reg_max: int,
    config: DetectionKDConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute confidence-weighted class and DFL distillation losses.

    Teacher and student must use the same detection grid and ``reg_max``.  This
    is true for the YOLOv8n/m models used by the guarded experiment, while the
    explicit checks make an incompatible architecture fail early.
    """

    config.validate()
    student_box, student_class = parse_detection_predictions(
        student_predictions, num_classes=num_classes, reg_max=reg_max
    )
    teacher_box, teacher_class = parse_detection_predictions(
        teacher_predictions, num_classes=num_classes, reg_max=reg_max
    )
    if student_box.shape != teacher_box.shape or student_class.shape != teacher_class.shape:
        raise ValueError(
            "Teacher and student raw prediction shapes differ: "
            f"student box/class={tuple(student_box.shape)}/{tuple(student_class.shape)}, "
            f"teacher box/class={tuple(teacher_box.shape)}/{tuple(teacher_class.shape)}"
        )

    student_box = student_box.float()
    student_class = student_class.float()
    teacher_box = teacher_box.detach().float()
    teacher_class = teacher_class.detach().float()
    batch_size, _, anchor_count = student_box.shape

    confidence = teacher_class.sigmoid().amax(dim=1)
    selected_count = min(config.topk, anchor_count)
    selected_indices = confidence.topk(selected_count, dim=1).indices
    selected = torch.zeros_like(confidence, dtype=torch.bool)
    selected.scatter_(1, selected_indices, True)
    selected &= confidence >= config.min_confidence
    weights = confidence * selected
    denominator = weights.sum().clamp_min(1e-6)

    temperature = config.temperature
    teacher_probability = torch.sigmoid(teacher_class / temperature)
    class_per_anchor = F.binary_cross_entropy_with_logits(
        student_class / temperature,
        teacher_probability,
        reduction="none",
    ).mean(dim=1)
    class_loss = (class_per_anchor * weights).sum() / denominator * temperature**2

    student_log_probability = F.log_softmax(
        student_box.reshape(batch_size, 4, reg_max, anchor_count) / temperature,
        dim=2,
    )
    teacher_probability_dfl = F.softmax(
        teacher_box.reshape(batch_size, 4, reg_max, anchor_count) / temperature,
        dim=2,
    )
    dfl_per_anchor = F.kl_div(
        student_log_probability,
        teacher_probability_dfl,
        reduction="none",
    ).sum(dim=2).mean(dim=1)
    dfl_loss = (dfl_per_anchor * weights).sum() / denominator * temperature**2
    total = config.class_weight * class_loss + config.dfl_weight * dfl_loss
    return class_loss, dfl_loss, total


def combine_detection_and_kd_loss(base_loss: Tensor, kd_loss: Tensor, batch_size: int) -> Tensor:
    """Add KD once to Ultralytics' vector-valued detection loss.

    Ultralytics returns ``[box, cls, dfl] * batch_size`` and the trainer sums
    that vector.  Summing first prevents scalar KD from broadcasting into all
    three components, which would accidentally triple its contribution.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return base_loss.sum() + kd_loss * batch_size


def _load_teacher(state: _KDState, student: DetectionModel, device: torch.device) -> DetectionModel:
    teacher_wrapper = YOLO(str(state.teacher_path))
    if teacher_wrapper.task != "detect":
        raise ValueError(f"KD teacher must be a detect checkpoint, found {teacher_wrapper.task!r}")
    teacher = teacher_wrapper.model.to(device).float().eval()
    if int(teacher.nc) != int(student.nc):
        raise ValueError(f"KD teacher has {teacher.nc} classes but student has {student.nc}")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    state.teacher = teacher
    return teacher


class KDDetectionModel(DetectionModel):
    """DetectionModel that adds teacher supervision only while training."""

    def loss(self, batch: dict[str, Tensor], preds: Any = None):
        if preds is None:
            preds = self.predict(batch["img"])
        base_loss, base_items = super().loss(batch, preds)
        state = _KD_STATES.get(self)
        if not self.training or state is None:
            return base_loss, base_items

        images = batch["img"]
        teacher = state.teacher or _load_teacher(state, self, images.device)
        teacher_head = teacher.model[-1]
        previous_head_training = teacher_head.training
        try:
            teacher_head.training = True  # request raw feature maps; backbone BN remains in eval mode
            with torch.no_grad():
                teacher_predictions = teacher(images)
        finally:
            teacher_head.training = previous_head_training

        reg_max = int(self.model[-1].reg_max)
        class_loss, dfl_loss, kd_total = detection_kd_loss(
            preds,
            teacher_predictions,
            num_classes=int(self.nc),
            reg_max=reg_max,
            config=state.config,
        )
        self._last_kd_items = torch.stack((class_loss, dfl_loss, kd_total)).detach()
        total = combine_detection_and_kd_loss(base_loss, kd_total, int(images.shape[0]))
        return total, base_items


def configure_detection_kd(
    model: KDDetectionModel,
    teacher_path: str | Path,
    config: DetectionKDConfig,
) -> None:
    """Attach KD configuration without registering or serializing the teacher."""

    teacher_path = Path(teacher_path).resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"KD teacher checkpoint was not found: {teacher_path}")
    config.validate()
    _KD_STATES[model] = _KDState(teacher_path=teacher_path, config=config)


class KDDetectionTrainer(DetectionTrainer):
    """Ultralytics trainer with import-safe KD model creation and KD metrics."""

    def __init__(
        self,
        *args: Any,
        teacher_path: str | Path,
        kd_config: DetectionKDConfig,
        **kwargs: Any,
    ) -> None:
        self.teacher_path = Path(teacher_path).resolve()
        self.kd_config = kd_config
        self._kd_epoch_sums = [0.0, 0.0, 0.0]
        self._kd_epoch_batches = 0
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._reset_kd_metrics)
        self.add_callback("on_train_batch_end", self._collect_kd_metrics)
        self.add_callback("on_train_epoch_end", self._write_kd_metrics)

    def get_model(self, cfg: str | None = None, weights: Any = None, verbose: bool = True):
        model = KDDetectionModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data.get("channels", 3),
            verbose=verbose,
        )
        if weights:
            model.load(weights)
        configure_detection_kd(model, self.teacher_path, self.kd_config)
        return model

    def _reset_kd_metrics(self, _trainer: Any) -> None:
        self._kd_epoch_sums = [0.0, 0.0, 0.0]
        self._kd_epoch_batches = 0

    def _collect_kd_metrics(self, _trainer: Any) -> None:
        model = de_parallel(self.model)
        values = getattr(model, "_last_kd_items", None)
        if values is None:
            return
        for index, value in enumerate(values.detach().float().cpu().tolist()):
            self._kd_epoch_sums[index] += float(value)
        self._kd_epoch_batches += 1

    def _write_kd_metrics(self, _trainer: Any) -> None:
        if RANK not in {-1, 0} or not self._kd_epoch_batches:
            return
        path = Path(self.save_dir) / "kd_metrics.csv"
        exists = path.exists()
        means = [value / self._kd_epoch_batches for value in self._kd_epoch_sums]
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if not exists:
                writer.writerow(("epoch", "kd_cls_loss", "kd_dfl_loss", "kd_total_loss", "batches"))
            writer.writerow((self.epoch + 1, *means, self._kd_epoch_batches))


def make_portable_detection_checkpoint(
    source: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a trusted KD checkpoint using the standard DetectionModel class."""

    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint was not found: {source}")
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Portable checkpoint already exists: {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        model = checkpoint.get(key)
        if isinstance(model, KDDetectionModel):
            model.__class__ = DetectionModel
        if model is not None and hasattr(model, "_last_kd_items"):
            delattr(model, "_last_kd_items")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)
    return destination
