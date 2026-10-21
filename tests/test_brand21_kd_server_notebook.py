from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from scripts.train_detector_kd import (
    build_train_args,
    expected_names,
    kd_config_from_dict,
    parse_args,
)
from src.training.yolo_detection_kd import (
    DetectionKDConfig,
    combine_detection_and_kd_loss,
    detection_kd_loss,
)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "05_brand21_knowledge_distillation_server.ipynb"
CONFIG = ROOT / "configs" / "brand_21brands_yolo_kd.yaml"


class Brand21KDServerNotebookTests(unittest.TestCase):
    def test_notebook_is_clean_and_every_execution_flag_defaults_false(self) -> None:
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        flags = (
            "RUN_INSTALL_DEPS",
            "RUN_PREFLIGHT",
            "RUN_TEACHER",
            "RUN_RESUME_TEACHER",
            "RUN_BASELINE",
            "RUN_RESUME_BASELINE",
            "RUN_STUDENT_KD",
            "RUN_RESUME_KD",
            "RUN_COMPARE",
        )
        for flag in flags:
            self.assertIn(f"{flag} = False", source)
        self.assertIn("scripts/train_detector_kd.py", source)
        self.assertIn("best_portable.pt", source)
        self.assertIn("source_data_yaml", source)
        self.assertNotIn("split='test'", source)
        self.assertTrue(all(not cell.get("outputs") for cell in notebook["cells"]))
        self.assertTrue(
            all(cell.get("execution_count") is None for cell in notebook["cells"])
        )

    def test_kd_config_uses_source_order_base_and_validation_only(self) -> None:
        value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(value["base_config"], "configs/brand_21brands_yolo.yaml")
        self.assertEqual(value["evaluation"]["split"], "val")
        self.assertEqual(value["teacher"]["model"], "yolov8s.pt")
        self.assertEqual(value["student"]["model"], "yolov8n.pt")
        self.assertLessEqual(value["student"]["batch"], value["teacher"]["batch"])
        config = kd_config_from_dict(value["kd"])
        self.assertEqual(config, DetectionKDConfig(0.25, 0.25, 2.0, 64, 0.05))

    def test_cli_exposes_separate_start_and_resume_commands(self) -> None:
        self.assertEqual(parse_args(["preflight"]).command, "preflight")
        self.assertEqual(parse_args(["teacher"]).command, "teacher")
        self.assertEqual(parse_args(["resume-teacher"]).command, "resume-teacher")
        self.assertEqual(parse_args(["baseline"]).command, "baseline")
        self.assertEqual(parse_args(["resume-baseline"]).command, "resume-baseline")
        self.assertEqual(parse_args(["student-kd"]).command, "student-kd")
        self.assertEqual(parse_args(["resume-kd"]).command, "resume-kd")

    def test_training_args_keep_one_dataset_and_disable_overwrite(self) -> None:
        context = {
            "teacher": {"epochs": 10, "batch": 4, "patience": 2},
            "student": {"epochs": 12, "batch": 2, "patience": 0},
            "_base": {
                "train": {
                    "device": 0,
                    "imgsz": 640,
                    "mosaic": 0.3,
                    "exist_ok": False,
                    "resume": False,
                }
            },
            "_plan": SimpleNamespace(data_yaml=Path("dataset/data.yaml")),
            "_project": Path("runs/kd"),
        }
        teacher = build_train_args(context, "teacher", "teacher_run")
        student = build_train_args(context, "student", "student_run")
        self.assertEqual(teacher["data"], student["data"])
        self.assertEqual(teacher["batch"], 4)
        self.assertEqual(student["batch"], 2)
        self.assertEqual(teacher["mosaic"], 0.3)
        self.assertFalse(teacher["exist_ok"])
        self.assertFalse(student["resume"])

    def test_expected_names_preserve_training_order(self) -> None:
        context = {"_mapping": {"0": "AXE", "1": "BEST_FOODS", "2": "CLEAR"}}
        self.assertEqual(
            expected_names(context), {0: "AXE", 1: "BEST_FOODS", 2: "CLEAR"}
        )

    def test_kd_loss_is_finite_and_added_once(self) -> None:
        # reg_max=2, nc=3 -> 11 channels. Teacher/student share one feature grid.
        student = [torch.randn(2, 11, 2, 2, requires_grad=True)]
        teacher = [torch.randn(2, 11, 2, 2)]
        class_loss, dfl_loss, total = detection_kd_loss(
            student,
            teacher,
            num_classes=3,
            reg_max=2,
            config=DetectionKDConfig(
                class_weight=0.25,
                dfl_weight=0.25,
                temperature=2.0,
                topk=4,
                min_confidence=0.0,
            ),
        )
        self.assertTrue(torch.isfinite(class_loss))
        self.assertTrue(torch.isfinite(dfl_loss))
        self.assertTrue(torch.isfinite(total))
        combined = combine_detection_and_kd_loss(
            torch.tensor([1.0, 2.0, 3.0]), total, batch_size=2
        )
        self.assertTrue(torch.allclose(combined, torch.tensor(6.0) + total * 2))


if __name__ == "__main__":
    unittest.main()
