from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from src.training.yolo import YoloRunPlan
from src.training.yolo_reporting import (
    CHECKPOINT_SELECTION_CRITERION,
    YoloTrainingCapture,
    write_yolo_training_report,
)


class FakeBoxMetrics:
    mp = 0.71
    mr = 0.62
    map50 = 0.81
    map = 0.57
    ap_class_index = [0, 2]

    def class_result(self, position: int) -> tuple[float, float, float, float]:
        return (
            (0.8, 0.7, 0.9, 0.65),
            (0.5, 0.4, 0.6, 0.35),
        )[position]


class FakeDetMetrics:
    names = {0: "ALPHA", 1: "BETA", 2: "GAMMA"}
    box = FakeBoxMetrics()


def _report_fixture(root: Path) -> tuple[YoloRunPlan, Path]:
    dataset = root / "data" / "prepared_v1"
    dataset.mkdir(parents=True)
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "ALPHA", 1: "BETA", 2: "GAMMA"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (dataset / "dataset_checks.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "dataset_version": "prepared_v1",
                "dataset_fingerprint": "sha256:" + "a" * 64,
                "splits": {
                    "val": {
                        "per_class": {
                            "ALPHA": {"images": 10, "instances": 12},
                            "BETA": {"images": 4, "instances": 4},
                            "GAMMA": {"images": 8, "instances": 9},
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (dataset / "mapping.json").write_text(
        json.dumps(
            {
                "output_names": {"0": "ALPHA", "1": "BETA", "2": "GAMMA"},
                "source_to_training": {"0": 2, "1": 0, "2": 1},
            }
        ),
        encoding="utf-8",
    )
    run_dir = root / "runs" / "detector" / "baseline"
    (run_dir / "weights").mkdir(parents=True)
    for path in (
        run_dir / "results.png",
        run_dir / "weights" / "best.pt",
        run_dir / "weights" / "last.pt",
        run_dir / "confusion_matrix.png",
        run_dir / "confusion_matrix_normalized.png",
        run_dir / "PR_curve.png",
    ):
        path.write_bytes(b"fixture")
    fields = [
        "epoch",
        "time",
        "train/box_loss",
        "train/cls_loss",
        "train/dfl_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "val/box_loss",
        "val/cls_loss",
        "val/dfl_loss",
        "lr/pg0",
    ]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "epoch": 1,
                    "time": 10.0,
                    "train/box_loss": 1.0,
                    "train/cls_loss": 2.0,
                    "train/dfl_loss": 3.0,
                    "metrics/precision(B)": 0.6,
                    "metrics/recall(B)": 0.5,
                    "metrics/mAP50(B)": 0.8,
                    "metrics/mAP50-95(B)": 0.5,
                    "val/box_loss": 1.1,
                    "val/cls_loss": 2.1,
                    "val/dfl_loss": 3.1,
                    "lr/pg0": 0.001,
                },
                {
                    "epoch": 2,
                    "time": 22.0,
                    "train/box_loss": 0.9,
                    "train/cls_loss": 1.9,
                    "train/dfl_loss": 2.9,
                    "metrics/precision(B)": 0.61,
                    "metrics/recall(B)": 0.51,
                    "metrics/mAP50(B)": 0.79,
                    "metrics/mAP50-95(B)": 0.49,
                    "val/box_loss": 1.0,
                    "val/cls_loss": 2.0,
                    "val/dfl_loss": 3.0,
                    "lr/pg0": 0.0005,
                },
            ]
        )
    config = {
        "model": "yolov8n.pt",
        "data": str(data_yaml),
        "project": str(run_dir.parent),
        "name": run_dir.name,
        "train": {
            "epochs": 2,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "mosaic": 0.3,
        },
        "artifacts": {
            "history_csv": "results.csv",
            "per_brand_validation_csv": "per_brand_validation.csv",
            "per_brand_validation_json": "per_brand_validation.json",
            "final_summary_markdown": "training_summary.md",
            "final_summary_json": "training_summary.json",
        },
    }
    config_path = root / "configs" / "detector.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return (
        YoloRunPlan(
            config_path=config_path,
            config=config,
            data_yaml=data_yaml,
            project_dir=run_dir.parent,
            run_dir=run_dir,
            model_argument=str(root / "yolov8n.pt"),
            train_args=dict(config["train"]),
        ),
        run_dir,
    )


class YoloReportingTests(unittest.TestCase):
    def test_callbacks_capture_latest_equal_best_and_one_runtime_row(self) -> None:
        capture = YoloTrainingCapture()
        registered: dict[str, object] = {}

        class FakeModel:
            def add_callback(self, event: str, callback: object) -> None:
                registered[event] = callback

        capture.attach(FakeModel())
        self.assertEqual(
            set(registered),
            {"on_train_epoch_start", "on_model_save", "on_fit_epoch_end"},
        )

        trainer = SimpleNamespace(
            epoch=0,
            best_fitness=0.53,
            fitness=0.53,
            epoch_time=3.2,
            device="cpu",
        )
        capture.on_train_epoch_start(trainer)
        capture.on_model_save(trainer)
        capture.on_fit_epoch_end(trainer)
        trainer.epoch = 1
        trainer.best_fitness = trainer.fitness = 0.53
        trainer.epoch_time = 3.0
        capture.on_model_save(trainer)
        capture.on_fit_epoch_end(trainer)
        capture.on_fit_epoch_end(trainer)  # final_eval reuses the same event/epoch

        self.assertEqual(capture.best_epoch, 2)
        self.assertEqual(capture.best_fitness, 0.53)
        self.assertEqual([row["epoch"] for row in capture.epoch_runtime], [1, 2])

    def test_report_reuses_returned_metrics_and_native_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan, run_dir = _report_fixture(Path(temporary_directory))
            capture = YoloTrainingCapture(
                best_epoch=1,
                best_fitness=0.53,
                epoch_runtime=[
                    {
                        "epoch": 1,
                        "epoch_time_seconds": 9.5,
                        "gpu_peak_allocated_bytes": 100,
                        "gpu_peak_reserved_bytes": 200,
                    },
                    {
                        "epoch": 2,
                        "epoch_time_seconds": 11.5,
                        "gpu_peak_allocated_bytes": 110,
                        "gpu_peak_reserved_bytes": 210,
                    },
                ],
            )
            summary = write_yolo_training_report(plan, FakeDetMetrics(), capture)
            saved = json.loads(
                (run_dir / "training_summary.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["epochs_completed"], 2)
            self.assertEqual(summary["history"][0]["epoch"], 1)
            self.assertEqual(summary["history"][1]["epoch_time_seconds"], 11.5)
            self.assertEqual(summary["checkpoint_selection"]["best_epoch"], 1)
            self.assertEqual(summary["class_order"], "registry")
            self.assertEqual(summary["registry_stable_mapping"], [])
            self.assertEqual(
                summary["checkpoint_selection"]["criterion"],
                CHECKPOINT_SELECTION_CRITERION,
            )
            self.assertEqual(summary["extra_evaluation_runs"], 0)
            self.assertFalse(summary["test_evaluated"])
            self.assertFalse(summary["model_inference_started_by_reporter"])
            self.assertAlmostEqual(
                summary["final_epoch_validation"]["selection_fitness"],
                0.1 * 0.79 + 0.9 * 0.49,
            )
            per_brand = summary["per_brand_validation"]
            self.assertEqual(per_brand[0]["mAP50_95"], 0.65)
            self.assertIsNone(per_brand[1]["mAP50_95"])
            self.assertEqual(per_brand[2]["mAP50_95"], 0.35)
            self.assertEqual(per_brand[2]["validation_instances"], 9)
            self.assertEqual(saved["dataset"]["fingerprint"], "sha256:" + "a" * 64)
            self.assertTrue((run_dir / "per_brand_validation.csv").is_file())
            self.assertTrue((run_dir / "per_brand_validation.json").is_file())
            self.assertTrue((run_dir / "training_summary.md").is_file())

    def test_report_does_not_guess_best_epoch_without_callback_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan, _run_dir = _report_fixture(Path(temporary_directory))
            summary = write_yolo_training_report(
                plan, FakeDetMetrics(), YoloTrainingCapture()
            )

        selection = summary["checkpoint_selection"]
        self.assertIsNone(selection["best_epoch"])
        self.assertFalse(selection["best_epoch_reliably_identified"])
        self.assertTrue(
            any("not captured" in item for item in summary["known_limitations"])
        )

    def test_report_serializes_source_order_and_stable_registry_mapping(self) -> None:
        contract = {
            "class_order": "source_data_yaml",
            "brand_profile": "fixture",
            "source_to_training_verified": True,
            "mapping_path": "mapping.json",
            "training_to_registry": [
                {
                    "training_id": index,
                    "training_name": name,
                    "registry_id": registry_id,
                    "registry_name": name,
                    "source_id": index,
                    "source_name": name,
                }
                for index, (name, registry_id) in enumerate(
                    (("ALPHA", 2), ("BETA", 0), ("GAMMA", 1))
                )
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            plan, run_dir = _report_fixture(Path(temporary_directory))
            with patch(
                "src.training.yolo_reporting.validate_brand_profile_class_mapping",
                return_value=contract,
            ):
                summary = write_yolo_training_report(
                    plan, FakeDetMetrics(), YoloTrainingCapture(best_epoch=1)
                )
            markdown = (run_dir / "training_summary.md").read_text(encoding="utf-8")

        self.assertEqual(summary["class_order"], "source_data_yaml")
        self.assertTrue(summary["source_to_training_mapping_verified"])
        self.assertEqual(summary["registry_stable_mapping"][0]["registry_id"], 2)
        self.assertIn("source_data_yaml", markdown)
        self.assertIn("| 0 | ALPHA | 2 | ALPHA | 0 | ALPHA |", markdown)


if __name__ == "__main__":
    unittest.main()
