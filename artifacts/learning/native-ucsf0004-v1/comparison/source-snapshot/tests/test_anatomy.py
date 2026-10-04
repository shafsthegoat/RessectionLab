from dataclasses import replace
from datetime import datetime, timezone
import json

import nibabel as nib
import numpy as np
import pytest

from resectionlab.anatomy import (FunctionalPrior, TEMPLATE_FRAME, load_functional_prior,
                                 load_prior_collection, prior_proximity_halo,
                                 register_prior, review_alignment)
from resectionlab.core import SourceRef
from resectionlab.imaging import file_sha256


def fixture_prior(kind="structural_mask"):
    data = np.zeros((9, 7, 7), dtype=np.float32)
    data[2, 2, 2] = 1
    return FunctionalPrior("test_motor", "motor", kind, data, np.diag([2., 3., 4., 1.]),
                           SourceRef("test", "synthetic:test", provenance="prior"), {})


def registered(prior=None, translation=6.):
    prior = prior or fixture_prior()
    transform = np.eye(4)
    transform[0, 3] = translation
    return register_prior(prior, target_shape=prior.data.shape, target_affine=prior.affine_ras_mm,
                          case_hash="case-v1", mni_ras_to_patient_ras_mm=transform,
                          method="synthetic_known_transform", version="1")


def accepted(evidence):
    return review_alignment(evidence, case_hash="case-v1", accepted=True, reviewer="fixture reviewer",
                            notes="Analytic landmark correspondence checked",
                            reviewed_at=datetime(2026, 10, 4, tzinfo=timezone.utc))


def test_translation_uses_physical_units_and_correct_transform_direction():
    evidence = registered()
    assert evidence.data[5, 2, 2] == 1  # source x=4 mm -> patient x=10 mm -> target i=5
    assert evidence.data[2, 2, 2] == 0
    assert not evidence.sampling_coverage[:3].any()
    assert evidence.sampling_coverage[3:].all()
    assert set(np.unique(evidence.data)) == {0, 1}


def test_lps_grid_has_same_physical_result():
    prior = fixture_prior()
    ras = registered(prior, translation=0)
    lps = register_prior(prior, target_shape=prior.data.shape,
                         target_affine=np.diag([-1., -1., 1., 1.]) @ prior.affine_ras_mm,
                         target_frame="LPS+", case_hash="case-v1", mni_ras_to_patient_ras_mm=np.eye(4),
                         method="synthetic_known_transform", version="1")
    np.testing.assert_equal(lps.data, ras.data)
    np.testing.assert_equal(lps.affine_ras_mm, ras.affine_ras_mm)


def test_functional_maps_interpolate_while_structural_masks_stay_categorical():
    prior = fixture_prior("functional_concordance")
    evidence = registered(prior, translation=1.)
    assert evidence.data[2, 2, 2] == pytest.approx(.5)
    assert evidence.data[3, 2, 2] == pytest.approx(.5)
    assert set(np.unique(registered(translation=1.).data)) <= {0, 1}


def test_no_overlay_or_halo_without_review_and_no_patient_specific_claim():
    evidence = registered()
    assert evidence.inventory_item()["overlay_enabled"] is False
    with pytest.raises(ValueError, match="UNREVIEWED"):
        prior_proximity_halo(evidence, case_hash="case-v1", scale_mm=3, threshold=.5)
    evidence = accepted(evidence)
    inventory = evidence.inventory_item()
    assert inventory["overlay_enabled"] is True
    assert inventory["patient_specific"] is False
    assert inventory["clinical_deficit_probability"] is None
    halo = prior_proximity_halo(evidence, case_hash="case-v1", scale_mm=3, threshold=.5)
    assert halo.distance_mm[5, 2, 2] == 0
    assert halo.distance_mm[6, 2, 2] == 2
    assert np.isnan(halo.values[0]).all()


def test_review_is_invalidated_by_case_source_transform_or_map_edit():
    evidence = accepted(registered())
    with pytest.raises(ValueError, match="STALE"):
        evidence.require_current("case-v2")
    for change in ({"case_hash": "case-v2"}, {"data": np.zeros(evidence.data.shape)},
                   {"registration_version": "2"}, {"mni_ras_to_patient_ras_mm": np.eye(4)}):
        with pytest.raises(ValueError, match="stale"):
            replace(evidence, **change)
    with pytest.raises(ValueError):
        evidence.data.setflags(write=True)


def test_rejected_no_overlap_and_reflection_remain_unusable():
    no_overlap = registered(translation=1000.)
    assert not no_overlap.sampling_coverage.any()
    with pytest.raises(ValueError, match="NO_ATLAS_OVERLAP"):
        accepted(no_overlap)
    with pytest.raises(ValueError, match="REFLECTION"):
        replace(no_overlap, mni_ras_to_patient_ras_mm=np.diag([-1., 1., 1., 1.]))
    rejection = review_alignment(registered(), case_hash="case-v1", accepted=False,
                                 reviewer="reviewer", notes="Midline mismatch",
                                 reviewed_at=datetime.now(timezone.utc))
    assert not rejection.inventory_item()["overlay_enabled"]
    with pytest.raises(ValueError, match="UNREVIEWED"):
        rejection.require_current("case-v1")


def test_realistic_qform_only_prior_is_loaded_and_hash_pinned(tmp_path):
    affine = np.array([[-2., 0, 0, 90], [0, 2, 0, -126], [0, 0, 2, -72], [0, 0, 0, 1]])
    image = nib.Nifti1Image(np.ones((3, 4, 5), np.float32), affine)
    image.header.set_xyzt_units("mm")
    image.set_qform(affine, 1)
    image.set_sform(None, 0)
    path = tmp_path / "motor.nii.gz"
    nib.save(image, path)
    spec = {"map_id": "motor", "component": "motor", "map_kind": "functional_concordance",
            "sha256": file_sha256(path), "record_id": 10439149, "archive_member": path.name,
            "license": "CC-BY-4.0", "template_frame": TEMPLATE_FRAME}
    prior = load_functional_prior(path, spec)
    np.testing.assert_equal(prior.affine_ras_mm, affine)
    assert prior.header_qc["sform_code"] == 0
    assert prior.inventory_item()["qc_state"] == "unregistered_prior"
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        load_functional_prior(path, {**spec, "sha256": "0" * 64})


def test_nonfinite_or_nonbinary_maps_and_unsafe_cache_paths_fail(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        replace(fixture_prior(), data=np.full((3, 3, 3), np.nan))
    with pytest.raises(ValueError, match="binary"):
        replace(fixture_prior(), data=np.full((3, 3, 3), .5))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "evidence_type": "prior",
                                    "maps": [{"cache_file": "../outside.nii.gz"}]}))
    with pytest.raises(ValueError, match="escapes"):
        load_prior_collection(tmp_path, manifest)


def test_review_requires_explicit_timezone_and_notes():
    with pytest.raises(ValueError, match="timezone"):
        review_alignment(registered(), case_hash="case-v1", accepted=True, reviewer="reviewer",
                         notes="Inspected", reviewed_at=datetime(2026, 10, 4))
    with pytest.raises(ValueError, match="notes"):
        review_alignment(registered(), case_hash="case-v1", accepted=True, reviewer="reviewer",
                         notes="", reviewed_at=datetime.now(timezone.utc))
