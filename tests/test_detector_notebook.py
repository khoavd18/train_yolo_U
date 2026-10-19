from __future__ import annotations

import json
import unittest
from pathlib import Path


class DetectorNotebookContractTests(unittest.TestCase):
    def test_detector_notebook_uses_selected_config_and_keeps_execution_off(self) -> None:
        path = Path("notebooks/01_detector_training.ipynb")
        notebook = json.loads(path.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        )
        self.assertIn('DETECTOR_CONFIG_PATH = "configs/brand_21brands_yolo.yaml"', source)
        self.assertIn('brand_profile = str(config.get("brand_profile", "brand_9brands"))', source)
        self.assertIn('class_order = str(config.get("class_order", "registry"))', source)
        self.assertIn("profile_brands = registry.profile(brand_profile)", source)
        self.assertIn("dataset_mapping[str(class_id)]", source)
        self.assertIn('if class_order == "source_data_yaml":', source)
        self.assertIn("assert set(training_names) == set(registry_names)", source)
        self.assertIn('elif class_order == "registry":', source)
        self.assertIn("assert training_names == registry_names", source)
        self.assertIn('"stable_registry_id": registry_ids_by_name[training_name]', source)
        self.assertIn("checkpoint_names != training_names", source)
        self.assertNotIn("registry_names[class_id]", source)
        for flag in (
            "RUN_PREFLIGHT",
            "RUN_TRAINING",
            "RUN_VALIDATION_EVALUATION",
            "RUN_TEST_EVALUATION",
        ):
            self.assertIn(f"{flag} = False", source)
            self.assertNotIn(f"{flag} = True", source)
        self.assertNotIn("model.train(", source)
        self.assertIn("scripts/train_detector.py", source)


if __name__ == "__main__":
    unittest.main()
