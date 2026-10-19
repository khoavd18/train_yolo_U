from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from src.data.detector_preparation import (
    DetectionBox,
    audit_extracted_source,
    prepare_extracted_dataset,
    validate_prepared_dataset,
)
from src.data.annotations import AnnotationFormat, parse_yolo_annotation


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), color).save(path, quality=95)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class ExtractedDetectorFixture:
    def __init__(self, root: Path, *, bad_declared_paths: bool = True) -> None:
        self.root = root
        self.source = root / "source"
        for source_split in ("train", "valid", "test"):
            (self.source / source_split / "images").mkdir(parents=True)
            (self.source / source_split / "labels").mkdir(parents=True)

        split_values = (
            {
                "train": "../train/images",
                "val": "../valid/images",
                "test": "../test/images",
            }
            if bad_declared_paths
            else {
                "train": "train/images",
                "val": "valid/images",
                "test": "test/images",
            }
        )
        (self.source / "data.yaml").write_text(
            yaml.safe_dump(
                {
                    **split_values,
                    "nc": 2,
                    # Deliberately opposite the registry order.
                    "names": ["BETA", "ALPHA"],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        (self.source / "README.roboflow.txt").write_text(
            "No pre-processing or augmentation was applied.\n", encoding="utf-8"
        )
        self.brand_registry = root / "brands.yaml"
        self.brand_registry.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "brands": [
                        {"id": 0, "name": "ALPHA", "aliases": []},
                        {"id": 1, "name": "BETA", "aliases": []},
                    ],
                    "profiles": {"fixture_2": {"brand_ids": [0, 1]}},
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        self.source_registry = root / "sources.yaml"
        self.source_registry.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "sources": [
                        {
                            "name": "fixture",
                            "path": self.source.as_posix(),
                            "annotation_format": "yolo_bbox",
                            "class_mapping_source": "data_yaml",
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def add(
        self,
        split: str,
        name: str,
        color: str,
        label: str | None,
    ) -> Path:
        image = self.source / split / "images" / name
        _write_image(image, color)
        if label is not None:
            (self.source / split / "labels" / f"{image.stem}.txt").write_text(
                label, encoding="utf-8"
            )
        return image

    def add_clean_baseline(self) -> dict[str, Path]:
        values = {
            "move_train": self.add(
                "train",
                "move_aug1_jpg.rf.train.jpg",
                "red",
                "0 0.50 0.50 0.30 0.40\n",
            ),
            "train_mix": self.add(
                "train",
                "train_mix_jpg.rf.train.jpg",
                "green",
                "0 0.30 0.50 0.20 0.30\n1 0.70 0.50 0.20 0.30\n",
            ),
            "move_val": self.add(
                "valid",
                "move_aug2_jpg.rf.valid.jpg",
                "blue",
                "1 0.50 0.50 0.25 0.35\n",
            ),
            "val_mix": self.add(
                "valid",
                "val_mix_jpg.rf.valid.jpg",
                "yellow",
                "0 0.30 0.50 0.20 0.30\n1 0.70 0.50 0.20 0.30\n",
            ),
            "test_mix": self.add(
                "test",
                "test_mix_jpg.rf.test.jpg",
                "purple",
                "0 0.30 0.50 0.20 0.30\n1 0.70 0.50 0.20 0.30\n",
            ),
        }
        return values

    def prepare(self, output: Path, **kwargs: object) -> dict[str, object]:
        return prepare_extracted_dataset(
            self.source,
            output,
            brand_profile="fixture_2",
            brand_registry_path=self.brand_registry,
            source_registry_path=self.source_registry,
            historical_datasets=kwargs.pop("historical_datasets", []),
            preview_count=int(kwargs.pop("preview_count", 4)),
            **kwargs,
        )


class ExtractedDetectorPreparationTests(unittest.TestCase):
    def test_source_yaml_class_order_keeps_stable_registry_ids_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExtractedDetectorFixture(root)
            fixture.add_clean_baseline()
            output = root / "derived_source_order"
            fixture.prepare(output, class_order="source_data_yaml")

            data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["names"], {0: "BETA", 1: "ALPHA"})
            mapping = json.loads((output / "mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping["class_order"], "source_data_yaml")
            self.assertEqual(
                mapping["source_to_target"],
                [
                    {
                        "source_id": 0,
                        "source_name": "BETA",
                        "target_id": 0,
                        "target_name": "BETA",
                        "registry_id": 1,
                    },
                    {
                        "source_id": 1,
                        "source_name": "ALPHA",
                        "target_id": 1,
                        "target_name": "ALPHA",
                        "registry_id": 0,
                    },
                ],
            )
            checks = json.loads((output / "dataset_checks.json").read_text(encoding="utf-8"))
            self.assertEqual(checks["class_order"], "source_data_yaml")
            self.assertEqual(checks["stable_registry_ids"], {"0": 1, "1": 0})

    def test_serialized_clipped_box_round_trips_inside_normalized_bounds(self) -> None:
        box = DetectionBox(
            class_id=0,
            class_name="FIRST",
            source_class_id=1,
            source_class_name="FIRST",
            source_line=1,
            x_min=0.0057,
            y_min=0.0,
            x_max=1.0,
            y_max=1.0,
        )
        parsed = parse_yolo_annotation(
            box.yolo_row(), ["FIRST"], AnnotationFormat.YOLO_BBOX
        )
        self.assertFalse(parsed.box.was_clipped)

    def test_clean_directory_remaps_classes_preserves_families_and_is_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExtractedDetectorFixture(root)
            fixture.add_clean_baseline()
            source_state = {
                path.relative_to(fixture.source).as_posix(): (
                    _sha256(path),
                    path.stat().st_mtime_ns,
                )
                for path in fixture.source.rglob("*")
                if path.is_file()
            }

            first = root / "derived_v1"
            report = fixture.prepare(first)
            checks = validate_prepared_dataset(first)
            self.assertEqual(report["status"], "prepared")
            self.assertTrue(checks["training_ready"])
            self.assertTrue(checks["all_target_classes_present_in_every_split"])
            self.assertTrue(checks["source_unchanged_verified"])

            data = yaml.safe_load((first / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["nc"], 2)
            self.assertEqual(Path(data["path"]), first.resolve())
            self.assertEqual(data["names"], {0: "ALPHA", 1: "BETA"})
            mapping = json.loads((first / "mapping.json").read_text(encoding="utf-8"))
            self.assertEqual(
                mapping["source_to_target"],
                [
                    {
                        "source_id": 0,
                        "source_name": "BETA",
                        "target_id": 1,
                        "target_name": "BETA",
                        "registry_id": 1,
                    },
                    {
                        "source_id": 1,
                        "source_name": "ALPHA",
                        "target_id": 0,
                        "target_name": "ALPHA",
                        "registry_id": 0,
                    },
                ],
            )

            provenance = _rows(first / "provenance.csv")
            move_rows = [
                row
                for row in provenance
                if "move_" in row["source_occurrences"]
            ]
            self.assertEqual(len(move_rows), 2)
            self.assertEqual({row["output_split"] for row in move_rows}, {"val"})
            self.assertTrue(
                all(Path(row["output_image_path"]).name.startswith("i_") for row in provenance)
            )
            self.assertTrue(
                all(len(Path(row["output_image_path"]).name) <= 64 for row in provenance)
            )
            train_mix = next(
                row for row in provenance if "train_mix" in row["source_occurrences"]
            )
            label_lines = (
                first / PurePath(train_mix["output_label_path"])
            ).read_text(encoding="utf-8").splitlines()
            self.assertEqual([line.split()[0] for line in label_lines], ["0", "1"])

            audit = json.loads((first / "source_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(audit["path_resolution"]["used_observed_layout_fallback"])
            self.assertEqual(audit["offline_augmentation_evidence"]["filenames_with_explicit_aug_marker"], 2)
            self.assertTrue((first / "previews" / "contact_sheet.jpg").is_file())

            current_source_state = {
                path.relative_to(fixture.source).as_posix(): (
                    _sha256(path),
                    path.stat().st_mtime_ns,
                )
                for path in fixture.source.rglob("*")
                if path.is_file()
            }
            self.assertEqual(source_state, current_source_state)

            second = root / "derived_v2"
            fixture.prepare(second)
            first_fingerprint = json.loads(
                (first / "dataset_fingerprint.json").read_text(encoding="utf-8")
            )["fingerprint"]
            second_fingerprint = json.loads(
                (second / "dataset_fingerprint.json").read_text(encoding="utf-8")
            )["fingerprint"]
            self.assertEqual(first_fingerprint, second_fingerprint)
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                fixture.prepare(first)

    def test_conflicts_invalid_rows_and_val_test_families_are_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExtractedDetectorFixture(root)
            fixture.add_clean_baseline()

            conflict_train = fixture.add(
                "train",
                "conflict_aug1_jpg.rf.a.jpg",
                "orange",
                "0 0.50 0.50 0.20 0.20\n",
            )
            conflict_val = fixture.source / "valid" / "images" / "conflict_aug2_jpg.rf.b.jpg"
            conflict_val.write_bytes(conflict_train.read_bytes())
            (fixture.source / "valid" / "labels" / f"{conflict_val.stem}.txt").write_text(
                "1 0.50 0.50 0.20 0.20\n", encoding="utf-8"
            )
            fixture.add(
                "train",
                "conflict_aug3_jpg.rf.c.jpg",
                "pink",
                "0 0.50 0.50 0.25 0.25\n",
            )
            fixture.add(
                "valid",
                "hold_aug1_jpg.rf.d.jpg",
                "black",
                "0 0.50 0.50 0.20 0.20\n",
            )
            fixture.add(
                "test",
                "hold_aug2_jpg.rf.e.jpg",
                "white",
                "0 0.50 0.50 0.20 0.20\n",
            )
            fixture.add(
                "train",
                "missing_jpg.rf.f.jpg",
                "gray",
                None,
            )
            fixture.add(
                "train",
                "duplicate_jpg.rf.g.jpg",
                "cyan",
                "0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n",
            )
            fixture.add(
                "train",
                "near_duplicate_jpg.rf.h.jpg",
                "brown",
                "0 0.50 0.50 0.30 0.30\n0 0.51 0.50 0.30 0.30\n",
            )
            fixture.add(
                "train",
                "samevariant_aug1_jpg.rf.i.jpg",
                "lime",
                "0 0.30 0.50 0.20 0.20\n",
            )
            fixture.add(
                "valid",
                "samevariant_aug1_jpg.rf.j.jpg",
                "maroon",
                "0 0.30 0.50 0.20 0.20\n0 0.70 0.50 0.20 0.20\n",
            )
            (fixture.source / "train" / "labels" / "orphan.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

            output = root / "derived"
            fixture.prepare(output)
            checks = json.loads((output / "dataset_checks.json").read_text(encoding="utf-8"))
            self.assertFalse(checks["training_ready"])
            self.assertTrue(checks["clean_subset_ready"])
            # The two conflicting files have identical bytes and therefore one
            # quarantined image hash; their family sibling is also excluded.
            self.assertGreaterEqual(checks["quarantined_image_hashes"], 6)
            self.assertGreater(checks["unresolved_annotation_hashes"], 0)
            quarantine = _rows(output / "quarantine_manifest.csv")
            conflict_family = [
                row for row in quarantine if "conflict" in row["source_occurrences"]
            ]
            # Two byte-identical conflicting occurrences share one audit hash;
            # the distinct family sibling is the second quarantined hash.
            self.assertEqual(len(conflict_family), 2)
            self.assertTrue(
                any(
                    "validation and test" in row["quarantine_reasons"]
                    for row in quarantine
                    if "hold_" in row["source_occurrences"]
                )
            )
            self.assertTrue(_rows(output / "review_decisions.csv"))
            invalid = _rows(output / "invalid_annotations.csv")
            self.assertTrue(any(row["reason"] == "missing label for image" for row in invalid))
            self.assertTrue(any("duplicate annotation row" in row["reason"] for row in invalid))
            near_duplicates = _rows(output / "near_duplicate_annotations.csv")
            self.assertTrue(
                any(
                    "near_duplicate" in row["source_image"]
                    and float(row["iou"]) >= 0.90
                    for row in near_duplicates
                )
            )
            same_variant = _rows(output / "same_variant_conflicts.csv")
            self.assertEqual(len(same_variant), 1)
            self.assertEqual(same_variant[0]["variant_id"], "samevariant_aug1")
            self.assertEqual(same_variant[0]["class_set_differs"], "False")
            audit = json.loads((output / "source_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["orphan_labels"], ["train/labels/orphan.txt"])

    def test_boundary_rounding_is_logged_but_larger_overrun_is_quarantined(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExtractedDetectorFixture(root)
            fixture.add_clean_baseline()
            fixture.add(
                "train",
                "tiny_rounding_jpg.rf.a.jpg",
                "navy",
                "0 0.5 0.5 1.00001 0.4\n",
            )
            fixture.add(
                "train",
                "large_overrun_jpg.rf.b.jpg",
                "teal",
                "0 0.5 0.5 1.00004 0.4\n",
            )
            output = root / "derived"
            fixture.prepare(output)
            corrections = _rows(output / "boundary_corrections.csv")
            self.assertEqual(len(corrections), 1)
            self.assertIn("tiny_rounding", corrections[0]["source_image"])
            invalid = _rows(output / "invalid_annotations.csv")
            self.assertTrue(
                any(
                    "above tolerance" in row["reason"]
                    and "large_overrun" in row["source_image"]
                    for row in invalid
                )
            )

    def test_historical_exact_hash_lock_moves_train_family_to_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExtractedDetectorFixture(root)
            values = fixture.add_clean_baseline()
            historical = root / "historical"
            historical_image = historical / "images" / "test" / "h.jpg"
            historical_image.parent.mkdir(parents=True)
            historical_image.write_bytes(values["train_mix"].read_bytes())
            (historical / "mapping.json").write_text(
                json.dumps({"output_names": {"0": "ALPHA", "1": "BETA"}}),
                encoding="utf-8",
            )
            (historical / "dataset_checks.json").write_text(
                json.dumps(
                    {
                        "cross_split_family_overlap": 0,
                        "exact_image_cross_split_overlap": {
                            "train-val": 0,
                            "train-test": 0,
                            "val-test": 0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with (historical / "provenance.csv").open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=(
                        "output_split",
                        "output_image_path",
                        "image_sha256",
                        "family_ids",
                    ),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "output_split": "test",
                        "output_image_path": "images/test/h.jpg",
                        "image_sha256": _sha256(historical_image),
                        "family_ids": "[]",
                    }
                )

            output = root / "derived"
            report = fixture.prepare(
                output,
                historical_datasets=[historical],
                class_order="source_data_yaml",
            )
            train_mix = next(
                row
                for row in _rows(output / "provenance.csv")
                if "train_mix" in row["source_occurrences"]
            )
            self.assertEqual(train_mix["output_split"], "test")
            self.assertEqual(report["historical_holdouts"]["locked_hashes"], 1)
            self.assertTrue(
                report["historical_holdouts"]["datasets"][0][
                    "historical_class_membership_verified"
                ]
            )

    def test_source_registry_must_explicitly_declare_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = ExtractedDetectorFixture(root)
            fixture.add_clean_baseline()
            registry = yaml.safe_load(
                fixture.source_registry.read_text(encoding="utf-8")
            )
            registry["sources"][0]["annotation_format"] = "yolo_obb"
            fixture.source_registry.write_text(
                yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "explicitly declare yolo_bbox"):
                audit_extracted_source(
                    fixture.source,
                    brand_profile="fixture_2",
                    brand_registry_path=fixture.brand_registry,
                    source_registry_path=fixture.source_registry,
                )


# Path deliberately handles POSIX paths stored in CSV on Windows and Linux.
def PurePath(value: str) -> Path:
    return Path(*value.replace("\\", "/").split("/"))


if __name__ == "__main__":
    unittest.main()
