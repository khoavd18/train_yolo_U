from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.preview_yolo_augmentations import (
    effective_augmentation,
    make_contact_sheet,
    render_transformed_sample,
)


class YoloAugmentationPreviewTests(unittest.TestCase):
    def test_render_uses_transformed_boxes_and_rejects_misalignment(self) -> None:
        sample = {
            "img": torch.zeros((3, 64, 64), dtype=torch.uint8),
            "cls": torch.tensor([[1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.25]]),
        }
        image, rows = render_transformed_sample(sample, {0: "AXE", 1: "VIM"})
        self.assertEqual(image.size, (64, 64))
        self.assertEqual(rows[0]["class_name"], "VIM")
        self.assertEqual(rows[0]["xywh_normalized"], [0.5, 0.5, 0.5, 0.25])

        invalid = {**sample, "bboxes": torch.tensor([[0.0, 0.5, 0.5, 0.25]])}
        with self.assertRaisesRegex(ValueError, "outside"):
            render_transformed_sample(invalid, {1: "VIM"})

    def test_effective_settings_are_training_keys_and_contact_sheet_is_written(self) -> None:
        train = {"mosaic": 0.3, "fliplr": 0.0, "augment": True, "epochs": 100}
        self.assertEqual(effective_augmentation(train), {"fliplr": 0.0, "mosaic": 0.3})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = {
                "img": torch.zeros((3, 32, 32), dtype=torch.uint8),
                "cls": torch.empty((0, 1)),
                "bboxes": torch.empty((0, 4)),
            }
            image, _rows = render_transformed_sample(sample, ["AXE"])
            first = root / "first.jpg"
            image.save(first)
            output = root / "sheet.jpg"
            make_contact_sheet([first], output)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
