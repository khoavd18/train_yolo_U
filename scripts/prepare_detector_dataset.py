"""Prepare or validate an audited detector dataset.

Extracted-directory sources use the generic workflow in
``src.data.detector_preparation``. Calls that do not select an extracted source
continue to forward to the historical ZIP workflow for compatibility. This
command never trains a model or runs model inference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.detector_preparation import (  # noqa: E402
    prepare_extracted_dataset,
    validate_prepared_dataset,
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parse_extracted_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--source-dir",
        help="Extracted YOLO source directory declared in the dataset-source registry.",
    )
    parser.add_argument("--brand-profile", default="brand_21brands")
    parser.add_argument(
        "--class-order",
        choices=("registry", "source_data_yaml"),
        default="registry",
        help=(
            "Training class order: stable registry profile order, or canonical registry names "
            "in the exact source data.yaml sequence."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--brand-registry", default=None)
    parser.add_argument("--source-registry", default=None)
    parser.add_argument(
        "--historical-dataset",
        action="append",
        default=None,
        help="Prepared dataset whose verified val/test provenance supplies locks; repeatable.",
    )
    parser.add_argument(
        "--no-historical-locks",
        action="store_true",
        help="Disable the default reviewed nine-brand provenance lock source.",
    )
    parser.add_argument("--decisions", default=None)
    parser.add_argument("--preview-count", type=int, default=45)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Revalidate an existing derived dataset without reading source or writing files.",
    )
    return parser.parse_args(argv)


def _option_value(argv: Sequence[str], name: str) -> str | None:
    for index, value in enumerate(argv):
        if value == name and index + 1 < len(argv):
            return argv[index + 1]
        prefix = name + "="
        if value.startswith(prefix):
            return value[len(prefix) :]
    return None


def _uses_extracted_workflow(argv: Sequence[str]) -> bool:
    if "--source-dir" in argv or any(
        value.startswith("--source-dir=") for value in argv
    ):
        return True
    if "--class-order" in argv or any(
        value.startswith("--class-order=") for value in argv
    ):
        return True
    return _option_value(argv, "--brand-profile") == "brand_21brands"


def main(argv: Sequence[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    if not _uses_extracted_workflow(values):
        # Import lazily so legacy ZIP globals never affect directory ingestion.
        from scripts.prepare_brand_9brands_yolo import main as legacy_main

        legacy_main(values)
        return

    args = parse_extracted_args(values)
    output_dir = resolve_path(args.output_dir)
    if args.check_only:
        print(
            json.dumps(
                validate_prepared_dataset(output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.source_dir:
        raise ValueError("--source-dir is required unless --check-only is used")
    if args.no_historical_locks and args.historical_dataset:
        raise ValueError(
            "--no-historical-locks cannot be combined with --historical-dataset"
        )
    historical = (
        []
        if args.no_historical_locks
        else (
            [resolve_path(value) for value in args.historical_dataset]
            if args.historical_dataset
            else None
        )
    )
    report = prepare_extracted_dataset(
        resolve_path(args.source_dir),
        output_dir,
        brand_profile=args.brand_profile,
        class_order=args.class_order,
        brand_registry_path=(
            resolve_path(args.brand_registry) if args.brand_registry else None
        ),
        source_registry_path=(
            resolve_path(args.source_registry) if args.source_registry else None
        ),
        historical_datasets=historical,
        decisions_path=resolve_path(args.decisions) if args.decisions else None,
        preview_count=args.preview_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
