"""Unrun Case2 phase adapter; no patient file IO, launcher, or authorization.

An existing supervised owner must authenticate the exact source bytes, protocol,
frame report and separate phase release before supplying bytes to these calls.
Reuse the canonical parser and static/rigid/IDW mathematics without modification.
This measures conditional sparse observation prediction, never tool response.
"""
from __future__ import annotations

import numpy as np

from resectionlab import mechanics_landmarks as lm
from scripts import mechanics_patient_comparison as comparison

PATIENT = "RESECT:Case2"
BEFORE_SHA = "38fe045a082ab49f008e0b847895d50b109828727a722d7d366aff470bfd6307"
DURING_SHA = "8b5560b8b777316d1ba0cb4b8b48ebb577818b0e300e149d51e0274f376d7251"
TAG_SHA = "a93b42da9a88eec1736fc287501d80b9aba89330ccaf970f6b0398c60b5a4a22"
TAG_BYTES = 1097
CONVENTION = "Creator paired-world-mm convention: first triplet before-US, second during-US; each original NIfTI sform maps voxel centres to RAS millimetres. Identity conversions express that convention, not anatomical registration accuracy."


def native_coverage(points, row_ids, header):
    """Closed native voxel-cell box, not world AABB, anatomy or US signal support."""
    points = np.asarray(points, dtype=float)
    affine = np.asarray(header["selected_affine"], dtype=float)
    shape = np.asarray(header["shape"], dtype=int)
    ids = list(row_ids)
    if (points.shape != (len(ids), 3) or affine.shape != (4, 4)
            or shape.shape != (3,) or np.any(shape < 1)
            or not np.isfinite(points).all() or not np.isfinite(affine).all()):
        raise ValueError("INVALID_COVERAGE_INPUT")
    inverse = np.linalg.inv(affine)
    native = points @ inverse[:3, :3].T + inverse[:3, 3]
    inside = np.all((native >= -.5) & (native <= shape - .5), axis=1)
    return {"meaning": "closed_native_voxel_cell_box_not_signal_or_anatomical_coverage",
            "boundary_tolerance_mm": 0., "excluded_rows": [],
            "rows": [{"row_id": row_id, "native_voxel": xyz.tolist(),
                      "inside_image_cell_box": bool(ok)}
                     for row_id, xyz, ok in zip(ids, native, inside, strict=True)]}


def primary_rms_difference(report):
    rigid = report["methods"]["proper_rigid"]["all_supported"]
    static = report["methods"]["no_shift"]["all_supported"]
    count = report["validation_landmarks"]
    value = (rigid["rms_mm"] - static["rms_mm"]
             if rigid["count"] == static["count"] == count else None)
    return {"definition": "RMS(proper_rigid)-RMS(no_shift), negative favors rigid",
            "rms_difference_mm": value, "required_all_V_count": count,
            "status": "available" if value is not None else "unavailable_incomplete_method_support"}


def source_partition(payload: bytes):
    """Trusted splitter only: parse source coordinates, zero destinations.

    Require >=12 unique rows and the unchanged six-source rank3/FPS rule.
    A failure is terminal for this fixed patient; do not substitute a case.
    The caller persists the returned metadata before requesting B release.
    """
    if len(payload) != TAG_BYTES:
        raise ValueError("EXACT_CASE2_TAG_SIZE_REQUIRED")
    binding = lm.LandmarkPairBinding(PATIENT, lm.DISPLACEMENT_ROLE,
                                    TAG_SHA, BEFORE_SHA, DURING_SHA)
    sources = lm.parse_tag_sources(payload, binding)
    partition = lm.partition_displacement_sources(sources)
    return {
        "schema": "resect-case2-source-partition-v1", "patient_group": PATIENT,
        "role": "TRAIN", "source_hash": partition.source_hash,
        "partition_hash": partition.partition_hash,
        "B_ids": list(partition.boundary_ids), "V_ids": list(partition.validation_ids),
        "source_row_count": len(sources.row_ids), "rank_ratio": partition.rank_ratio,
        "rule": partition.rule, "destination_coordinates_parsed": 0,
        "V_is_fresh_patient_level_evaluation": False,
    }


def _prepared(payload, partition_record, frame, frame_sha256):
    if source_partition(payload) != partition_record:
        raise ValueError("PERSISTED_SOURCE_PARTITION_CHANGED")
    if (frame.get("status") != "coordinate_convention_verified"
            or frame.get("convention") != CONVENTION
            or frame.get("source_image_sha256") != BEFORE_SHA
            or frame.get("destination_image_sha256") != DURING_SHA
            or frame.get("source_world_to_ras_mm") != np.eye(4).tolist()
            or frame.get("destination_world_to_ras_mm") != np.eye(4).tolist()
            or frame.get("anatomical_alignment_accepted") is not False
            or any(frame.get(key) is not None for key in (
                "physical_clearance_mm", "cavity_support", "total_registration_uncertainty_mm"))):
        raise ValueError("EXACT_QUALIFIED_FRAME_REQUIRED")
    qc = lm.LandmarkFrameQC(BEFORE_SHA, DURING_SHA, np.eye(4), np.eye(4),
                           frame_sha256, frame["convention"])
    binding = lm.LandmarkPairBinding(PATIENT, lm.DISPLACEMENT_ROLE,
                                    TAG_SHA, BEFORE_SHA, DURING_SHA, qc)
    partition = lm.partition_displacement_sources(lm.parse_tag_sources(payload, binding))
    return binding, partition


def fit_and_freeze(root, *, payload, partition_record, frame, frame_sha256,
                   protocol_binding, output_directory):
    """Six B destinations only; complete fields saved before evaluator V reveal."""
    binding, partition = _prepared(payload, partition_record, frame, frame_sha256)
    forward = lm.build_forward_landmarks(payload, binding, partition)
    field = comparison.fit_baselines(forward, protocol_sha256=protocol_binding["sha256"])
    frozen = comparison.freeze_comparison(root, forward=forward, source_binding=binding,
        protocol_binding=protocol_binding, field=field,
        output_directory=output_directory, external_field=None)
    return {"freeze": frozen, "B_ids": list(partition.boundary_ids),
            "B_destination_coverage": native_coverage(forward.observed_ras_mm,
                partition.boundary_ids, frame["headers"]["during"]),
            "V_destinations_accessed": [], "methods": list(comparison.METHODS)}


def evaluate_once(root, *, payload, partition_record, frame, frame_sha256,
                  freeze_binding):
    """All disjoint V rows; canonical helper persists a one-shot reveal marker."""
    binding, partition = _prepared(payload, partition_record, frame, frame_sha256)
    # Capture the typed validation already revealed by the canonical evaluator;
    # do not invoke the reveal function again or independently parse destinations.
    original_reveal = lm.reveal_validation_landmarks
    captured = []
    def capture(*args, **kwargs):
        value = original_reveal(*args, **kwargs)
        captured.append(value)
        return value
    try:
        lm.reveal_validation_landmarks = capture
        report_binding, report = comparison.evaluate_frozen_comparison(root,
            freeze_binding=freeze_binding, payload=payload, source_binding=binding,
            partition=partition, external_sampler=None)
    finally:
        lm.reveal_validation_landmarks = original_reveal
    if len(captured) != 1:
        raise ValueError("EXACTLY_ONE_VALIDATION_REVEAL_REQUIRED")
    validation = captured[0]
    if report["validation_landmarks"] != len(partition.validation_ids):
        raise ValueError("ALL_VALIDATION_ROWS_REQUIRED")
    return {"evaluation": report_binding, "V_ids": list(partition.validation_ids),
            "V_destination_coverage": native_coverage(validation.observed_ras_mm,
                partition.validation_ids, frame["headers"]["during"]),
            "primary_comparison": primary_rms_difference(report),
            "patient_group": PATIENT, "role": "TRAIN",
            "fit_rows": len(partition.boundary_ids),
            "validation_rows": len(partition.validation_ids),
            "independent_patients": 1, "calibrated_uncertainty_mm": None,
            "claim": "conditional_sparse_registration_observation_agreement",
            "causal_tool_response_validated": False,
            "future_untouched_validation_for_adapted_methods": False}
