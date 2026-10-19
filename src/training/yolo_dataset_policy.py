"""Dataset-readiness policies shared by reviewed YOLO training/export."""

from __future__ import annotations

from typing import Any


def assess_nine_brand_training_policy(
    checks: dict[str, Any],
    *,
    allow_quarantined_exclusions: bool,
) -> dict[str, Any]:
    """Assess whether the reviewed dataset or its clean subset may be trained.

    Unresolved/quarantined images may be excluded only when the generated
    train/validation/test splits are structurally clean and contain none of
    those hashes.  This does not relabel or silently approve quarantined data.
    """

    exact_overlap = checks.get("exact_image_cross_split_overlap", {})
    all_targets_present = checks.get("all_target_classes_present_in_every_split")
    if all_targets_present is None:
        all_targets_present = checks.get("all_nine_brands_present_in_every_split")
    safety_requirements = {
        "status_structurally_valid": checks.get("status") == "structural_checks_passed",
        "ordinary_detection_labels": checks.get("ordinary_yolo_detection_rows_only") is True,
        "all_target_classes_present": all_targets_present is True,
        "exact_cross_split_overlap_zero": bool(exact_overlap)
        and all(int(value) == 0 for value in exact_overlap.values()),
        "family_cross_split_overlap_zero": int(checks.get("cross_split_family_overlap", -1)) == 0,
        "known_holdout_violations_zero": int(checks.get("known_holdout_violations", -1)) == 0,
        "quarantined_hashes_in_splits_zero": int(checks.get("quarantined_hashes_in_splits", -1)) == 0,
        "test_split_not_evaluated": checks.get("test_split_evaluated") is False,
    }
    failed = sorted(name for name, passed in safety_requirements.items() if not passed)
    source_ready = checks.get("training_ready") is True
    unresolved = int(checks.get("unresolved_annotation_hashes", 0))
    quarantined = int(checks.get("quarantined_image_hashes", 0))

    if source_ready and not failed:
        effective_ready = True
        mode = "fully_reviewed"
    elif allow_quarantined_exclusions and not failed and unresolved > 0 and quarantined > 0:
        effective_ready = True
        mode = "clean_subset_excluding_quarantine"
    else:
        effective_ready = False
        mode = "blocked"

    return {
        "source_training_ready": source_ready,
        "effective_training_ready": effective_ready,
        "mode": mode,
        "allow_quarantined_exclusions": bool(allow_quarantined_exclusions),
        "unresolved_annotation_hashes_excluded": unresolved,
        "quarantined_image_hashes_excluded": quarantined,
        "safety_requirements": safety_requirements,
        "failed_safety_requirements": failed,
        "limitation": (
            "Quarantined images remain unresolved and are excluded from train/validation/test; "
            "metrics describe the structurally clean subset, not the quarantined population."
            if mode == "clean_subset_excluding_quarantine"
            else None
        ),
    }
