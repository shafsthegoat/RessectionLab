#!/usr/bin/env python3
"""Explicitly attach existing registered priors as unreviewed research evidence.

No registration, source acquisition, anatomy approval or policy update occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.ndimage import binary_fill_holes

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from resectionlab.functional_evidence import population_prior_sensitivity
from resectionlab.imaging import load_case, save_case
from resectionlab.worlds import WorldGeneratorConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--translation-mm", type=float, required=True,
                        help="Explicit uncalibrated sensitivity scale, not a measured registration error")
    parser.add_argument("--rotation-deg", type=float, default=0.)
    parser.add_argument("--family", choices=("rigid_uniform", "rigid_gaussian"), default="rigid_uniform")
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        parser.error("Output and receipt must be new files; existing evidence is never replaced")
    if not np.isfinite([args.translation_mm, args.rotation_deg]).all() or min(args.translation_mm, args.rotation_deg) < 0:
        parser.error("Sensitivity scales must be finite and nonnegative")
    started = time.perf_counter()
    source_sha = hashlib.sha256(args.case.read_bytes()).hexdigest()
    case = load_case(args.case)
    center = case.voxel_to_world((np.asarray(case.mri.shape) - 1.) / 2.)
    if case.frame == "LPS+":
        center = np.diag([-1., -1., 1.]) @ center
    generator = WorldGeneratorConfig(family=args.family,
        translation_scale_mm=(args.translation_mm,) * 3, rotation_scale_deg=(args.rotation_deg,) * 3,
        rotation_center_mm=tuple(center), parameter_basis="declared_uncalibrated_population_prior_registration_sensitivity")
    evidence = population_prior_sensitivity(case, uncertainty=generator)
    enriched = case.revised(functional_evidence=evidence)
    save_case(enriched, args.output)
    restored = load_case(args.output)
    if restored.semantic_hash != enriched.semantic_hash or restored.functional_evidence.fingerprint != evidence.fingerprint:
        raise RuntimeError("Functional evidence failed save/reopen identity")
    if source_sha != hashlib.sha256(args.case.read_bytes()).hexdigest():
        raise RuntimeError("Original case bytes changed")
    if case.brain_mask is not None:
        tissue = np.asarray(case.brain_mask)
        support = "supplied_case_brain_mask_review_status_unchanged"
    else:
        from resectionlab.structural_evidence import declared_mri_support_allowed
        if not declared_mri_support_allowed(case):
            raise ValueError("No eligible tissue support for source-grid coverage reporting")
        tissue = binary_fill_holes(case.mri != 0)
        for mask in case.compartments.values():
            tissue |= mask
        support = "estimated_skull_stripped_nonzero_MRI_envelope_unreviewed"
    coverage = {name: None if getattr(evidence, name) is None else {
        "tissue_cells": int(tissue.sum()), "covered_tissue_cells": int((getattr(evidence, name + "_coverage") & tissue).sum()),
        "covered_tissue_fraction": float(getattr(evidence, name + "_coverage")[tissue].mean())}
        for name in ("motor", "language")}
    report = {"schema_version": 1, "operation": "explicit_unreviewed_population_prior_attachment_and_roundtrip",
        "source_bundle": str(args.case), "source_bundle_sha256": source_sha,
        "source_case_hash": case.semantic_hash, "case_hash": restored.semantic_hash,
        "source_planning_hash": case.planning_hash, "planning_hash": restored.planning_hash,
        "output_bundle": str(args.output), "output_bundle_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "evidence": evidence.to_manifest(), "atlas_sampling_coverage_within_declared_tissue": coverage,
        "tissue_support": support, "original_proposals_unchanged_view_only": True,
        "expert_approval": False, "patient_specific_function": False, "clinical_deficit_probability": None,
        "policy_updates": 0, "source_registration_refitted": False, "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["population_priors_uncalibrated", "alignment_review_required", "patient_tracts_unavailable",
                        "vascular_anatomy_unassessed", "official_UCSF_source_equivalence_not_established_by_this_operation"],
        "source_sha256": {str(path.relative_to(Path(__file__).resolve().parents[1])): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__).resolve(), *[Path(__file__).resolve().parents[1] / "src/resectionlab" / name
                for name in ("functional_evidence.py", "core.py", "imaging.py", "prior_proposals.py", "worlds.py")]]}}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"case_hash": restored.semantic_hash, "evidence_hash": evidence.fingerprint,
                      "coverage": coverage, "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
