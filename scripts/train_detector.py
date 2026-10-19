"""Preflight or manually train an ordinary YOLO detector from one config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.yolo import (  # noqa: E402
    YoloRunPlan,
    build_generic_preflight,
    execute_yolo_training,
    inspect_yolo_runtime,
)
from src.training.yolo_reporting import (  # noqa: E402
    YoloTrainingCapture,
    write_yolo_training_report,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config", required=True, help="Detector training YAML config.")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate config, dataset metadata, class order, and output path without training.",
    )
    return parser.parse_args(argv)


def preflight(
    config_path: str | Path,
    *,
    root: str | Path = ROOT,
) -> tuple[YoloRunPlan, dict[str, object]]:
    plan, summary = build_generic_preflight(config_path, root=root)
    summary.update(inspect_yolo_runtime(plan))
    return plan, summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    plan, summary = preflight(args.config)
    if args.preflight:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    expected_version = str(plan.config["compatibility"]["verified_ultralytics"])
    capture = YoloTrainingCapture()
    metrics = execute_yolo_training(
        plan,
        expected_version=expected_version,
        missing_dependency_message="Ultralytics is not installed in this interpreter",
        version_error_factory=lambda expected, actual: (
            f"Expected reviewed Ultralytics {expected}, found {actual}"
        ),
        payload_factory=lambda version: {**summary, "ultralytics": version},
        ensure_ascii=False,
        training_capture=capture,
    )
    if plan.config.get("artifacts", {}).get("write_training_summary") is True:
        report = write_yolo_training_report(plan, metrics, capture)
        print(
            json.dumps(
                {
                    "training_summary": report["artifacts"]["summary_json"]["path"],
                    "per_brand_validation": report["artifacts"][
                        "per_brand_validation_json"
                    ]["path"],
                    "extra_evaluation_runs": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
