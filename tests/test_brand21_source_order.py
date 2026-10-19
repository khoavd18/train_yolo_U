from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src.training.yolo import build_generic_preflight


SOURCE_NAMES = (
    "AXE",
    "BEST FOODS",
    "CLEAR",
    "CLOSE_UP",
    "COMFORT",
    "DOVE",
    "HAZELINE",
    "KNORR",
    "LIFEBUOY",
    "LUX",
    "OMO",
    "PONDS",
    "PS",
    "REXONA",
    "SIMPLE",
    "SUNLIGHT",
    "SUNSILK",
    "SURF",
    "TRESEMME",
    "VASELINE",
    "VIM",
)
TRAINING_NAMES = tuple(name.replace(" ", "_") for name in SOURCE_NAMES)
REGISTRY_IDS = {
    "AXE": 0,
    "VIM": 1,
    "LIFEBUOY": 2,
    "HAZELINE": 3,
    "REXONA": 4,
    "CLEAR": 5,
    "DOVE": 6,
    "SUNSILK": 7,
    "TRESEMME": 8,
    "BEST_FOODS": 9,
    "CLOSE_UP": 10,
    "COMFORT": 11,
    "KNORR": 12,
    "LUX": 13,
    "OMO": 14,
    "PONDS": 15,
    "PS": 16,
    "SIMPLE": 17,
    "SUNLIGHT": 18,
    "SURF": 19,
    "VASELINE": 20,
}


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "data" / "brand_21brands_yolo_v2"
    dataset.mkdir(parents=True)
    mapping = {str(index): name for index, name in enumerate(TRAINING_NAMES)}
    (dataset / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(dataset),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(mapping),
                "names": mapping,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    checks_path = dataset / "dataset_checks.json"
    checks_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "ordinary_yolo_detection_rows_only": True,
                "mapping_verified": mapping,
                "class_order": "source_data_yaml",
                "stable_registry_ids": {
                    str(index): REGISTRY_IDS[name]
                    for index, name in enumerate(TRAINING_NAMES)
                },
                "exact_image_cross_split_overlap": {
                    "train-val": 0,
                    "train-test": 0,
                    "val-test": 0,
                },
                "cross_split_family_overlap": 0,
                "test_split_evaluated": False,
            }
        ),
        encoding="utf-8",
    )
    mapping_path = dataset / "mapping.json"
    mapping_path.write_text(
        json.dumps(
            {
                "target_profile": "brand_21brands",
                "class_order": "source_data_yaml",
                "target_names": mapping,
                "output_names": mapping,
                "source_to_target": [
                    {
                        "source_id": index,
                        "source_name": source_name,
                        "target_id": index,
                        "target_name": training_name,
                        "registry_id": REGISTRY_IDS[training_name],
                    }
                    for index, (source_name, training_name) in enumerate(
                        zip(SOURCE_NAMES, TRAINING_NAMES, strict=True)
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    config_path = root / "configs" / "brand_21brands_yolo.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": "yolov8n.pt",
                "data": "data/brand_21brands_yolo_v2/data.yaml",
                "project": "runs/model1_brand21",
                "name": "sourceorder_v2",
                "brand_profile": "brand_21brands",
                "class_order": "source_data_yaml",
                "train": {
                    "task": "detect",
                    "epochs": 1,
                    "imgsz": 640,
                    "batch": 1,
                    "device": "cpu",
                    "seed": 42,
                    "resume": False,
                    "exist_ok": False,
                    "val": True,
                },
                "artifacts": {"evaluate_test_split": False},
                "compatibility": {
                    "verified_ultralytics": "8.3.155",
                    "install_or_upgrade_automatically": False,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path, mapping_path, checks_path


class Brand21SourceOrderTests(unittest.TestCase):
    def test_source_order_and_stable_registry_ids_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, _mapping_path, _checks_path = _fixture(root)
            _plan, summary = build_generic_preflight(config_path, root=root)

        self.assertEqual(tuple(summary["mapping"].values()), TRAINING_NAMES)
        self.assertEqual(summary["class_order"], "source_data_yaml")
        self.assertTrue(summary["source_to_training_mapping_verified"])
        rows = summary["registry_stable_mapping"]
        self.assertEqual(
            (rows[1]["training_name"], rows[1]["registry_id"]),
            ("BEST_FOODS", 9),
        )
        self.assertEqual(
            (rows[20]["training_name"], rows[20]["registry_id"]),
            ("VIM", 1),
        )

    def test_mapping_or_registry_id_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path, mapping_path, checks_path = _fixture(root)
            document = json.loads(mapping_path.read_text(encoding="utf-8"))
            document["source_to_target"][1]["target_id"] = 2
            mapping_path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source_id to equal"):
                build_generic_preflight(config_path, root=root)

            document["source_to_target"][1]["target_id"] = 1
            mapping_path.write_text(json.dumps(document), encoding="utf-8")
            checks = json.loads(checks_path.read_text(encoding="utf-8"))
            checks["stable_registry_ids"]["20"] = 999
            checks_path.write_text(json.dumps(checks), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "stable_registry_ids differ"):
                build_generic_preflight(config_path, root=root)


if __name__ == "__main__":
    unittest.main()
