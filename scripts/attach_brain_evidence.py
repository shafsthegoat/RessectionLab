#!/usr/bin/env python3
"""Save unreviewed extraction envelopes separately from working anatomy.

This imports display evidence only. It cannot accept review or enable access.
"""
import argparse
import json
from pathlib import Path

from resectionlab.imaging import (
    import_brain_extraction_evidence, load_case, read_case_artifacts, save_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=["main", "nocsf"], action="append")
    args = parser.parse_args()
    forbidden = {args.case.resolve(), args.source_image.resolve(), args.report.resolve()}
    forbidden.update(path.resolve() for path in args.report.parent.iterdir() if path.is_file())
    if args.output.resolve() in forbidden:
        parser.error("Save to a new case path, preserving the input case, source image and extraction artifacts")
    try:
        original = load_case(args.case)
        updated = original
        for variant in dict.fromkeys(args.variant or ["main", "nocsf"]):
            updated = import_brain_extraction_evidence(
                updated, source_image_path=args.source_image,
                mask_path=args.report.parent / f"{variant}_mask.nii.gz",
                report_path=args.report, variant=variant,
            )
        save_case(updated, args.output, artifacts=read_case_artifacts(args.case))
        reopened = load_case(args.output)
        if reopened.semantic_hash != updated.semantic_hash or reopened.planning_hash != original.planning_hash:
            raise ValueError("Structural proposal import changed planning inputs or failed roundtrip identity")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Structural evidence import failed: {exc}\n")
    print(json.dumps({"bundle": str(args.output), "reopened_identical": True,
                      "working_anatomy_unchanged": True, "cortical_access_permitted": False,
                      "evidence": {key: item.review_status for key, item in reopened.structural_evidence.items()}}))


if __name__ == "__main__":
    main()
