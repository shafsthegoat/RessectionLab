#!/usr/bin/env python3
"""Stage a reversible whole-component display split beside an exact TOF review."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

from build_eden2020_vascular_companion import CANDIDATE, HDD_ROOT, HDD_UUID
from resectionlab.anatomy_bridge import open_anatomy_manifest
from resectionlab.anatomy_geometry import vascular_cleanup_source
from resectionlab.data_root import atomic_json, checked_root, safe_path, sha256_file
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.vascular_cleanup import separate_reviewed_components

MAIN = "vesselfm-review-intracranial-priority"
FIELD_PRIORITY = "vesselfm-review-field-priority"
COMPARTMENTS = {
    "posterior": {
        "name": "posterior compartment review",
        "main_label": "Main TOF network · posterior estimate separated",
        "id": "vesselfm-review-posterior-superficial",
        "label": "Posterior superficial estimate · structure uncertain",
        "receipt": "posterior_compartment_review",
        "screen": "structural_screen",
        "scope": "Read-only research display using structural alignment estimates",
    },
    "field-edge": {
        "name": "field-edge component review",
        "main_label": "Main TOF network · field-edge signal separated",
        "id": "vesselfm-review-field-edge",
        "label": "Field-edge estimate · structure uncertain",
        "receipt": "field_edge_compartment_review",
        "screen": "native_screen",
        "scope": "Read-only research display using acquired TOF field-edge evidence",
        "screen_version": "native-tof-field-edge-v1",
    },
    "skull-exterior": {
        "name": "skull-exterior component review",
        "main_id": "vesselfm-review-skull-priority",
        "main_label": "Main TOF network · skull priority",
        "id": "vesselfm-review-skull-exterior",
        "label": "Outside-skull estimate · structure uncertain",
        "receipt": "skull_exterior_compartment_review",
        "screen": "structural_screen",
        "scope": "Read-only research display using acquired TOF and estimated skull boundaries",
        "screen_version": "native-tof-mri-skull-exterior-v1",
    },
    "source-disagreement": {
        "name": "source disagreement review",
        "main_id": "vesselfm-review-source-priority",
        "main_label": "Main TOF network · source priority",
        "id": "vesselfm-review-source-disagreement",
        "label": "Source-disputed estimate · structure uncertain",
        "receipt": "source_disagreement_compartment_review",
        "screen": "structural_screen",
        "scope": "Read-only research display using acquired TOF and structural source review",
        "screen_version": "native-tof-mri-source-disagreement-v1",
    },
}


def split_layers(document, compartment):
    review = document["vascular_review"]
    initial = document["display"]["initial_layers"]
    group = COMPARTMENTS[compartment]
    skull = review.get("skull_exterior_compartment_review", {}).get("disjoint_union_matches_selected") is True
    if (group["receipt"] in review or "source_disagreement_compartment_review" in review
            or ("skull_exterior_compartment_review" in review and (not skull or compartment != "source-disagreement"))):
        raise ValueError("The selected review already contains this or a later split")
    posterior = review.get("posterior_compartment_review", {}).get("disjoint_union_matches_selected") is True
    field = review.get("field_edge_compartment_review", {}).get("disjoint_union_matches_selected") is True
    if compartment == "posterior" and field:
        raise ValueError("Stage a posterior split before field-edge review")
    source_id = (FIELD_PRIORITY if posterior and field else MAIN) if posterior or field else CANDIDATE
    if skull:
        source_id = COMPARTMENTS["skull-exterior"]["main_id"]
    if initial[0] != source_id:
        raise ValueError("The selected review does not start with its bound priority layer")
    if compartment in {"skull-exterior", "source-disagreement"}:
        return source_id, group["main_id"], [group["main_id"]]
    main_id = FIELD_PRIORITY if posterior else MAIN
    return source_id, main_id, [main_id, group["id"], *initial[1:]]


def stage(root, selected_path, screen_path, anchors, *, compartment="posterior"):
    group = COMPARTMENTS[compartment]
    screen_sha = sha256_file(screen_path)
    evidence = json.loads(screen_path.read_text())
    if group.get("screen_version") and evidence.get("version") != group["screen_version"]:
        raise ValueError(f"The {compartment} group requires its source-reviewed screen")
    selected_sha = sha256_file(selected_path)
    if evidence["study_sha256"] != selected_sha:
        raise ValueError("The structural screen belongs to another TOF review")
    canonical_path = selected_path.with_name("patient-anatomy.json")
    canonical_sha = sha256_file(canonical_path)
    if evidence["source_index_sha256"] != canonical_sha:
        raise ValueError("The structural screen belongs to another canonical anatomy snapshot")
    structural = evidence.get("structural_source_manifest")
    if compartment == "source-disagreement":
        if not structural or not all(evidence.get(key) is True for key in (
            "zero_frozen_path_overlap", "zero_frozen_junction_overlap",
        )):
            raise ValueError("Source disagreement requires structural review and protected native path checks")
        for key in ("source_evidence", "exact_geometry_audit"):
            binding = evidence.get(key)
            if not binding or sha256_file(safe_path(root, binding["hdd_relative_path"])) != binding["sha256"]:
                raise ValueError(f"The bound {key.replace('_', ' ')} is missing or has changed")
    if structural:
        structural_path = safe_path(root, structural["hdd_relative_path"])
        if sha256_file(structural_path) != structural["sha256"]:
            raise ValueError("The screened structural source manifest has changed")
    study = open_anatomy_manifest(selected_path)
    canonical = PatientImagingCollection(canonical_path.parent)
    source_id, main_id, initial = split_layers(study.document, compartment)
    layer = study.layer(source_id)
    scan = study.series(layer["source_series"])
    baseline = vascular_cleanup_source(study, layer)
    if (study.document["exposure"] != "private" or scan["modality"] != "TOF"
            or study.document["patient_id"] != f'EDEN2020:{evidence["subject"]}'
            or scan["original"]["sha256"] != evidence["scan_sha256"]
            or layer["data"] != evidence["selected_native_grid"] or baseline is None):
        raise ValueError("Review geometry or patient identity differs from structural evidence")
    candidates = {row["source_anchor_cell"]: row for row in evidence["major_components"]}
    if missing := set(anchors) - candidates.keys():
        raise ValueError(f"Reviewed anchors were not measured on the structural screen: {sorted(missing)}")
    choices = {anchor: candidates[anchor]["cells"] for anchor in anchors}
    main_mask, compartment_mask = separate_reviewed_components(study.mask(layer), choices)

    document = deepcopy(study.document)
    document.pop("snapshot_id", None)
    document["label"] += f' · {group["name"]}'
    document.update(clinical_validation=False, training_admitted=False, simulation_admission=False)
    trial = PatientImagingCollection(selected_path.parent, document)
    for name, title, color, mask in (
        (main_id, group["main_label"], layer["color"], main_mask),
        (group["id"], group["label"], "#8d877a", compartment_mask),
    ):
        result = deepcopy(layer)
        result.update(id=name, label=title, color=color, revision=0, data=trial.put_array(mask.astype("uint8")))
        result["provenance"].update(
            method="Review display group of complete native TOF components",
            evidence_use="context_only",
            clinical_validation=False,
            simulation_admission=False,
        )
        result["provenance"]["vascular_cleanup"].update(
            result_mask_sha256=result["data"]["sha256"],
        )
        trial.document["layers"].append(result)
    review = trial.document["vascular_review"]
    review.update(linked_source=True, source_manifest={
        "hdd_relative_path": str(canonical_path.relative_to(root)),
        "sha256": canonical_sha,
        "snapshot_id": canonical.identity(),
    })
    review[group["receipt"]] = {
        "selected_manifest_sha256": selected_sha,
        group["screen"]: {"hdd_relative_path": str(screen_path.relative_to(root)), "sha256": screen_sha},
        "components": [{"source_anchor_cell": anchor, "voxels": size} for anchor, size in sorted(choices.items())],
        "view_scope": group["scope"],
        "disjoint_union_matches_selected": True,
        "simulation_admission": False,
    }
    trial.document["display"]["initial_layers"] = initial
    trial.document["snapshot_id"] = trial.identity()
    path = selected_path.with_name(f"vascular-review.{trial.identity()[:12]}.anatomy.json")
    if (sha256_file(selected_path) != selected_sha or sha256_file(canonical_path) != canonical_sha
            or sha256_file(screen_path) != screen_sha):
        raise ValueError("A bound review source changed during staging")
    if path.exists():
        if json.loads(path.read_text()) != trial.document:
            raise ValueError("The destination revision belongs to different contents")
    else:
        atomic_json(path, trial.document)
    if open_anatomy_manifest(path).identity() != trial.identity():
        raise ValueError("The linked review did not reopen at its staged snapshot")
    return path, trial.identity()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path, required=True, help="Selected review path relative to the verified HDD")
    parser.add_argument("--screen", type=Path, required=True, help="Reviewed screen JSON relative to the verified HDD")
    parser.add_argument("--anchor", type=int, action="append", required=True, help="Reviewed flat native component anchor")
    parser.add_argument("--compartment", choices=COMPARTMENTS, default="posterior")
    args = parser.parse_args()
    root = checked_root(HDD_ROOT, expected_uuid=HDD_UUID)
    manifest, snapshot = stage(root, safe_path(root, args.selected), safe_path(root, args.screen), args.anchor,
                               compartment=args.compartment)
    print(json.dumps({"path": str(manifest.relative_to(root)), "sha256": sha256_file(manifest),
                      "snapshot_id": snapshot}, indent=2))


if __name__ == "__main__":
    main()
