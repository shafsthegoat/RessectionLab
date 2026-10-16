"""One opt-in generated FK geometry pilot on the frozen IXI265 host.

The acquired T1 and all original/display sources stay unchanged. This pilot
explicitly uses a homogeneous diffusion domain because the bounded SynthSeg
trials produced no accepted tissue maps. It creates an overlay, not a synthetic
MR appearance, tumor forecast, edema label, or anatomical validation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt

from resectionlab.core import SourceRef, array_digest, thaw_json
from resectionlab.virtual_glioma import (
    FKParameters, UNIFORM_DIFFUSION_DOMAIN, qualify_host, simulate_virtual_glioma,
)
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state

ROOT = Path(__file__).resolve().parents[1]
HOST_SHA = "c735daaf6d60f5f0d1567760a2accc18d4bd4838a096d8e0b5f490ee88abaff1"
HEALTH_PAGE_SHA = "f80f30853eeeeec6caf90ae4410c712fe447161808c238dea8111396020efe8c"
TARGET = "virtual_glioma_threshold_0p1"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def physical_grid(brain, affine, spacing_mm=3.):
    """Axis-aligned grid covering the source brain bounding cells plus one cell.

    Uses all eight transformed bounds, including reflected/oblique source grids.
    Bounding-box coverage does not certify the estimated brain's completeness.
    """
    brain = np.asarray(brain)
    affine = np.asarray(affine, dtype=np.float64)
    if (brain.ndim != 3 or brain.dtype != np.bool_ or not brain.any()
            or affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.isfinite(spacing_mm) or not .5 <= spacing_mm <= 10.):
        raise ValueError("Finite physical source and nonempty binary brain required")
    limits = []
    for axis in range(3):
        occupied = np.flatnonzero(brain.any(axis=tuple(a for a in range(3) if a != axis)))
        limits.append((float(occupied[0]) - .5, float(occupied[-1]) + .5))
    corners = np.array(list(itertools.product(*limits)))
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    low = np.floor(world.min(axis=0)/spacing_mm)*spacing_mm - spacing_mm
    high = np.ceil(world.max(axis=0)/spacing_mm)*spacing_mm + spacing_mm
    shape = tuple(int(x) for x in np.ceil((high-low)/spacing_mm).astype(int) + 1)
    if any(x > 96 or x < 2 for x in shape):
        raise ValueError("Source brain exceeds the declared bounded pilot grid")
    grid = np.diag([spacing_mm, spacing_mm, spacing_mm, 1.])
    grid[:3, 3] = low
    return shape, grid


def mapped_domain(brain, source_affine, grid_shape, grid_affine):
    mapping = np.linalg.inv(source_affine) @ grid_affine
    return affine_transform(np.asarray(brain, np.uint8), mapping[:3, :3], mapping[:3, 3],
        output_shape=grid_shape, order=0, mode="constant", cval=0, prefilter=False) > 0


def choose_seed(domain, grid_affine):
    """A deterministic unreviewed RAS placement, with no inferred lobe label."""
    spacing = np.linalg.norm(grid_affine[:3, :3], axis=0)
    interior = domain & (distance_transform_edt(domain, sampling=spacing) >= 9.)
    indices = np.argwhere(interior)
    if not len(indices):
        raise ValueError("No source-envelope interior for the declared seed support")
    world = indices @ grid_affine[:3, :3].T + grid_affine[:3, 3]
    # A repeatable normalized location is a parameter choice, not atlas anatomy.
    desired = world.min(axis=0) + np.array([.32, .68, .65]) * np.ptp(world, axis=0)
    index = int(np.argmin(np.sum((world-desired)**2, axis=1)))
    return tuple(float(x) for x in world[index]), {
        "method": "nearest_admitted_9mm_interior_cell_to_normalized_RAS_bounds_0.32_0.68_0.65",
        "requested_ras_mm": desired.tolist(), "actual_ras_mm": world[index].tolist(),
        "hemisphere": "UNKNOWN_UNREVIEWED", "lobe": "UNKNOWN_UNREVIEWED",
        "clinical_placement_review": False,
    }


def native_target(coarse_mask, coarse_affine, brain, source_affine):
    """Declared render transfer; separate from the unchanged PDE result."""
    mapping = np.linalg.inv(coarse_affine) @ source_affine
    transferred = affine_transform(np.asarray(coarse_mask, np.uint8), mapping[:3, :3], mapping[:3, 3],
        output_shape=brain.shape, order=0, mode="constant", cval=0, prefilter=False) > 0
    outside = int(np.count_nonzero(transferred & ~brain))
    target = transferred & brain
    if not target.any():
        raise ValueError("Declared threshold has no native display support")
    return target, {
        "method": "nearest_neighbor_voxel_centers_then_explicit_intersection_with_source_brain_estimate",
        "native_to_coarse_voxel": mapping.tolist(),
        "transferred_cells_before_intersection": int(transferred.sum()),
        "cells_outside_source_brain_excluded_by_display_definition": outside,
        "native_cells": int(target.sum()),
        "pde_arrays_modified": False,
        "qualification": "Display overlay construction on acquired T1; not an MR signal simulation or equivalent coarse-grid volume",
    }


def run(host_path, output):
    # Exact frozen admission precedes decoding. No protected payload is searched.
    if sha(host_path) != HOST_SHA:
        raise ValueError("Only the frozen source-qualified IXI265 host is admitted")
    if output.exists():
        raise ValueError("Use a new output directory; existing pilot evidence is immutable")
    host, artifacts, saved = load_workspace(host_path)
    if (host.metadata.get("patient_group") != "IXI:265"
            or host.metadata.get("split", {}).get("role") != "TRAIN"
            or host.metadata.get("training_admitted") is not False
            or host.compartments or saved is None):
        raise ValueError("Preserve exact healthy-cohort source identity and frozen TRAIN/not-training-admitted role")
    pages = [item for item in saved["originals"] if item.sha256 == HEALTH_PAGE_SHA]
    if len(pages) != 1 or b"normal, healthy subjects" not in pages[0].data:
        raise ValueError("Exact archived publisher health evidence is required")
    before = {"case_hash": host.semantic_hash, "image_hash": array_digest(host.mri),
              "brain_hash": array_digest(host.brain_mask), "affine_hash": array_digest(host.affine)}
    shape, affine = physical_grid(host.brain_mask, host.affine)
    qualification = qualify_host(host, patient_group="IXI:265", role="TRAIN",
        healthy_source_evidence={"status": "source_reported_healthy", "patient_group": "IXI:265",
            "source_url": "https://brain-development.org/ixi-dataset/", "source_sha256": HEALTH_PAGE_SHA,
            "statement": "Publisher-described normal healthy cohort; no individual pathology review or tumor-absence certificate"},
        qualified_by="automated_source_and_geometry_QC_no_anatomical_expert_acceptance",
        grid_shape=list(shape), grid_affine=affine, tissue_mode=UNIFORM_DIFFUSION_DOMAIN)
    domain = mapped_domain(host.brain_mask, host.affine, shape, affine)
    seed, placement = choose_seed(domain, affine)
    params = FKParameters(seed_center_ras_mm=seed)
    result = simulate_virtual_glioma(host, qualification, params, domain_mask=domain)
    result.save(output)
    mask, transfer = native_target(result.threshold_masks["virtual_concentration_ge_0p1"],
        affine, host.brain_mask, host.affine)
    transfer["native_volume_mm3"] = int(mask.sum()) * abs(float(np.linalg.det(host.affine[:3, :3])))
    provenance = {"kind": "tumor", "label": "Virtual FK concentration ≥ 0.1 (generated)",
        "provenance": "simulated", "source_role": "simulator_generated_geometry_only",
        "source_sha256": sha(output/"virtual-glioma.npz"), "color": "#f59e42",
        "simulation_id": result.manifest["simulationId"], "threshold": .1,
        "observed_patient_tumor": False, "clinical_validation": False, "display_transfer": transfer}
    metadata = {**thaw_json(host.metadata), "virtual_geometry": {
        "host_case_id": host.case_id, "host_case_hash": host.semantic_hash,
        "host_patient_group": "IXI:265", "inherited_host_role": "TRAIN",
        "virtual_lesion_count": 1, "new_patient_count": 0, "clinical_validation": False,
        "simulation_id": result.manifest["simulationId"], "placement": placement,
        "tissue_mode": UNIFORM_DIFFUSION_DOMAIN,
        "limits": ["No GM/WM classification or CSF barriers", "Uncalibrated growth parameters",
                   "No mass effect, vessel response, functional risk or synthetic MRI appearance"]},
        "anatomy_layers": {**thaw_json(host.metadata.get("anatomy_layers", {})), TARGET: provenance}}
    descendant = replace(host, case_id=host.case_id+"-virtual-FK-pilot-001", compartments={TARGET: mask},
        source_compartments=host.source_compartments, metadata=metadata,
        source_refs=(*host.source_refs, SourceRef("virtual_glioma_fk", "generated://"+result.manifest["simulationId"],
            sha256=provenance["source_sha256"], license="CC-BY-SA-3.0 (host); MIT (solver source)",
            native_frame="RAS+", provenance="simulated")),
        unknowns=(*host.unknowns, "virtual lesion is generated geometry; acquired MRI signal remains healthy-host source"))
    # Display sources are still same-person, unreviewed and excluded from policy.
    state = view_state(descendant, saved["sources"])
    state["states"]["primary"]["viewer"] = {"layerOpacity": {"brain": .13, TARGET: .9},
        "layerColor": {"brain": "#65bfff", TARGET: "#f59e42"}}
    destination = output/(descendant.case_id+".ressectionlab")
    identity = save_workspace(descendant, destination, sources=saved["sources"], imaging=state,
        episode=None, selection=None, artifacts=artifacts, originals=saved["originals"])
    reopened, _, restored = load_workspace(destination)
    if (reopened.semantic_hash != descendant.semantic_hash or restored["sessionHash"] != identity
            or array_digest(reopened.mri) != before["image_hash"]
            or array_digest(reopened.brain_mask) != before["brain_hash"]
            or array_digest(reopened.affine) != before["affine_hash"]
            or [x.sha256 for x in saved["originals"]] != [x.sha256 for x in restored["originals"]]
            or not np.array_equal(reopened.compartments[TARGET], mask)
            or host.semantic_hash != before["case_hash"]):
        raise ValueError("Generated bundle failed source preservation or exact save/reopen")
    receipt = {"schema": "ixi265-virtual-glioma-pilot-v1", "status": "generated_geometry_saved_not_anatomically_accepted",
        "host_bundle_sha256": HOST_SHA, "host": before, "patient_group": "IXI:265", "role": "TRAIN",
        "training_admitted": False, "new_patient_count": 0, "virtual_lesion_count": 1,
        "source_original_files": len(restored["originals"]), "source_arrays_and_originals_unchanged": True,
        "runner_sha256": sha(Path(__file__)), "simulation_id": result.manifest["simulationId"],
        "placement": placement, "display_transfer": transfer, "bundle": str(destination),
        "bundle_sha256": sha(destination), "case_hash": descendant.semantic_hash, "session_hash": identity,
        "reopened_identical": True, "tissue_mode": UNIFORM_DIFFUSION_DOMAIN,
        "anatomical_acceptance": False, "clinical_validation": False,
        "counts_scope": "One generated lesion on one already named host; not added to frozen nine-person catalog"}
    write_json(output/"bundle-receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", type=Path, default=ROOT/"outputs/poc-ixi265/IXI265-Guys-0845.ressectionlab")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/poc-ixi265/virtual-FK-pilot-001")
    args = parser.parse_args()
    print(json.dumps(run(args.host.resolve(), args.output.resolve()), indent=2))
