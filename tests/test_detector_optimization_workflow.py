from __future__ import annotations

import contextlib
import csv
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from scripts.calibrate_detector_inference import (
    load_search as load_inference_search,
)
from scripts.calibrate_detector_inference import metrics_at_confidence
from scripts.tune_detector_hyperparameters import (
    build_trial_configs,
    load_search as load_hyperparameter_search,
    summarize_trials,
)
from src.training.yolo import load_generic_yolo_config


ROOT = Path(__file__).resolve().parents[1]


class DetectorOptimizationWorkflowTests(unittest.TestCase):
    def test_declared_searches_are_validation_only_and_finite(self) -> None:
        hyper = load_hyperparameter_search(
            ROOT / "configs" / "brand_21brands_yolo_search.yaml"
        )
        inference = load_inference_search(
            ROOT / "configs" / "brand_21brands_inference_search.yaml"
        )
        self.assertFalse(hyper["test_split_used"])
        self.assertEqual(hyper["selection_metric"], "metrics/mAP50-95(B)")
        self.assertEqual(len(hyper["trials"]), 5)
        self.assertFalse(inference["test_split_used"])
        self.assertEqual(inference["objective"], "f1")
        trial_count = 1
        for key in ("imgsz", "conf", "iou"):
            trial_count *= len(inference["grid"][key])
        self.assertEqual(trial_count, 12)

    def test_trial_configs_change_only_declared_train_settings(self) -> None:
        base = load_generic_yolo_config(ROOT / "configs" / "brand_21brands_yolo.yaml")
        search = load_hyperparameter_search(
            ROOT / "configs" / "brand_21brands_yolo_search.yaml"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            trials = build_trial_configs(base, search, root=temporary_directory)
        self.assertEqual(len(trials), 5)
        for trial in trials:
            self.assertEqual(trial["config"]["data"], base["data"])
            self.assertEqual(trial["config"]["brand_profile"], base["brand_profile"])
            self.assertEqual(trial["config"]["class_order"], "source_data_yaml")
            self.assertEqual(trial["config"]["train"]["epochs"], 25)
            self.assertFalse(trial["config"]["train"]["resume"])
            self.assertFalse(trial["config"]["train"]["exist_ok"])

    def test_unsafe_hyperparameter_override_is_rejected(self) -> None:
        value = {
            "project": "runs/tuning",
            "name_prefix": "pilot",
            "pilot_epochs": 2,
            "selection_metric": "metrics/mAP50-95(B)",
            "selection_mode": "max",
            "test_split_used": False,
            "trials": [{"label": "unsafe", "train": {"device": "cpu"}}],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "search.yaml"
            path.write_text(yaml.safe_dump(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsafe overrides"):
                load_hyperparameter_search(path)

    def test_inference_metrics_use_exact_declared_confidence(self) -> None:
        metric = SimpleNamespace(
            px=np.array([0.0, 0.5, 1.0]),
            p_curve=np.array([[0.2, 0.8, 1.0], [0.4, 0.6, 1.0]]),
            r_curve=np.array([[1.0, 0.5, 0.0], [1.0, 0.4, 0.0]]),
        )
        precision, recall, f1 = metrics_at_confidence(metric, 0.5)
        self.assertAlmostEqual(precision, 0.7)
        self.assertAlmostEqual(recall, 0.45)
        expected_f1 = ((2 * 0.8 * 0.5 / 1.3) + (2 * 0.6 * 0.4 / 1.0)) / 2
        self.assertAlmostEqual(f1, expected_f1)

    def test_summary_selects_validation_winner_and_restores_full_epochs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            project = root / "runs" / "tuning"
            base = load_generic_yolo_config(
                ROOT / "configs" / "brand_21brands_yolo.yaml"
            )
            trials = []
            for label, score, overrides in (
                ("a", 0.31, {}),
                ("b", 0.42, {"lr0": 0.0005}),
            ):
                run_dir = project / f"pilot_{label}"
                run_dir.mkdir(parents=True)
                with (run_dir / "results.csv").open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(
                        handle,
                        fieldnames=("epoch", "metrics/mAP50-95(B)"),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"epoch": 0, "metrics/mAP50-95(B)": score}
                    )
                pilot = deepcopy(base)
                pilot["train"]["epochs"] = 25
                trials.append(
                    {
                        "label": label,
                        "overrides": overrides,
                        "base_config": deepcopy(base),
                        "config": pilot,
                        "plan": SimpleNamespace(
                            run_dir=run_dir, project_dir=project
                        ),
                    }
                )
            with contextlib.redirect_stdout(io.StringIO()):
                summary = summarize_trials(
                    trials,
                    {
                        "selection_metric": "metrics/mAP50-95(B)",
                        "selection_mode": "max",
                    },
                )
            recommended = yaml.safe_load(
                (project / "recommended_full_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(summary["winner"]["label"], "b")
        self.assertEqual(recommended["train"]["epochs"], base["train"]["epochs"])
        self.assertEqual(recommended["train"]["lr0"], 0.0005)
        self.assertEqual(recommended["class_order"], "source_data_yaml")


if __name__ == "__main__":
    unittest.main()
