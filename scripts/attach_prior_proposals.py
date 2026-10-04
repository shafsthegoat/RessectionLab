#!/usr/bin/env python3
"""Save verified registered atlas previews as unreviewed, view-only proposals."""
import argparse
import json
from pathlib import Path

from resectionlab.imaging import load_case, read_case_artifacts, save_case
from resectionlab.prior_proposals import import_registered_prior_proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("case", "registration-directory", "source-cache-directory", "prior-manifest",
                   "source-image", "registration-image", "lesion", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = {path.resolve() for path in (args.case, args.prior_manifest, args.source_image,
                                         args.registration_image, args.lesion)}
    if args.output.exists() or args.output.is_symlink():
        parser.error("Output already exists; choose a new case path to preserve existing artifacts")
    if (output in inputs or output.is_relative_to(args.registration_directory.resolve())
            or output.is_relative_to(args.source_cache_directory.resolve())):
        parser.error("Save to a new case path outside source, cache and registration artifacts")
    try:
        original = load_case(args.case)
        updated = import_registered_prior_proposals(
            original, registration_directory=args.registration_directory,
            source_cache_directory=args.source_cache_directory, prior_manifest_path=args.prior_manifest,
            source_image_path=args.source_image, registration_image_path=args.registration_image,
            lesion_path=args.lesion)
        save_case(updated, output, artifacts=read_case_artifacts(args.case))
        reopened = load_case(output)
        if reopened.semantic_hash != updated.semantic_hash or reopened.planning_hash != original.planning_hash:
            raise ValueError("Proposal roundtrip failed or changed planning inputs")
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Registered prior proposal import failed: {exc}\n")
    print(json.dumps({"bundle": str(output), "reopened_identical": True,
                      "planning_inputs_unchanged": True, "template_arrays_embedded": False,
                      "proposals": {key: value.review_status for key, value in reopened.prior_proposals.items()}}))


if __name__ == "__main__":
    main()
