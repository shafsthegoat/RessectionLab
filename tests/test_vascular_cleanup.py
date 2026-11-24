import json
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.anatomy_bridge import VASCULAR_REVIEW_MANIFEST, dispatch, open_anatomy_manifest
from resectionlab.patient_anatomy import PatientImagingCollection, digest, file_hash
from resectionlab.vascular_cleanup import (
    HIDDEN, RETAINED, UNCERTAIN, classify_components, review_component_visibility,
    separate_reviewed_components,
)


def test_native_review_preserves_pial_and_bright_extracranial_candidates():
    shape = (64, 64, 48)
    axes = np.ogrid[:shape[0], :shape[1], :shape[2]]
    inside = sum(((axis - center) / radius) ** 2 for axis, center, radius in zip(
        axes, (32, 32, 24), (16, 17, 15)
    )) < 1
    signal = np.where(inside, 1000, 5).astype(np.float32)
    signal[32, 14:50, 24] = np.linspace(1500, 3000, 36)
    signal[12:14, 14:17, 24] = 1450
    mask = np.zeros(shape, dtype=bool)
    mask[29:36, 14:50, 24] = True
    mask[12:14, 14:17, 24] = True
    mask[48:54, 5:9, 24] = True
    scores = np.where(mask, 0.94, 0.1)
    frame = np.diag([0.5, 0.5, 0.65, 1.0])

    partition, report = classify_components(mask, signal, scores, frame)

    np.testing.assert_array_equal(partition > 0, mask)
    assert np.all(partition[29:36, 14:50, 24] == RETAINED)
    assert np.all(partition[12:14, 14:17, 24] == UNCERTAIN)
    assert np.all(partition[48:54, 5:9, 24] == HIDDEN)
    component = next(row for row in report["components"] if row["state"] == HIDDEN)
    restored, revised = review_component_visibility(partition, report, restore=[component["id"]])
    np.testing.assert_array_equal(restored > 0, mask)
    assert not np.any(restored == HIDDEN)
    assert report["hidden_voxels"] == component["voxels"]
    assert revised["uncertain_voxels"] == report["uncertain_voxels"] + component["voxels"]
    excluded, rereviewed = review_component_visibility(restored, revised, exclude=[component["id"]])
    np.testing.assert_array_equal(excluded, partition)
    assert rereviewed["hidden_voxels"] == report["hidden_voxels"]
    with pytest.raises(ValueError, match="Unknown source component"):
        review_component_visibility(partition, report, restore=["component-0"])


def test_no_source_support_abstains_and_preserves_all_components():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:12, 5, 5] = True
    mask[12, 12, 12] = True
    partition, report = classify_components(mask, np.ones(mask.shape), None, np.eye(4))
    np.testing.assert_array_equal(partition > 0, mask)
    assert report["hidden_components"] == 0
    assert report["reference_signal_available"] is False


def test_compartment_separation_preserves_whole_native_connected_components():
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1:4, 2, 2] = True
    mask[4, 3, 3] = True
    mask[5:7, 6, 1] = True
    anchor = int(np.ravel_multi_index((1, 2, 2), mask.shape))
    main, compartment = separate_reviewed_components(mask, {anchor: 4})

    np.testing.assert_array_equal(main | compartment, mask)
    assert not np.any(main & compartment)
    assert compartment[4, 3, 3] and main[5, 6, 1]
    with pytest.raises(ValueError, match="differs"):
        separate_reviewed_components(mask, {anchor: 3})
    inside = int(np.ravel_multi_index((2, 2, 2), mask.shape))
    with pytest.raises(ValueError, match="differs"):
        separate_reviewed_components(mask, {inside: 4})


def test_alternate_review_manifest_binds_and_preserves_canonical_anatomy(tmp_path):
    base = PatientImagingCollection.create(tmp_path, "generated")
    base.save()
    manifest = tmp_path / "patient-anatomy.json"
    checksum = file_hash(manifest)
    review = {
        **base.document,
        "vascular_review": {
            "schema": "eden2020-restricted-vascular-companion-v1",
            "linked_source": True,
            "source_manifest": {"sha256": checksum, "snapshot_id": base.identity()},
        },
    }
    review.pop("snapshot_id")
    review["snapshot_id"] = digest(review)
    path = tmp_path / VASCULAR_REVIEW_MANIFEST
    path.write_text(json.dumps(review))
    assert open_anatomy_manifest(path).identity() == review["snapshot_id"]
    assert open_anatomy_manifest(manifest).identity() == base.identity()
    session = SimpleNamespace()
    opened = dispatch(session, "anatomyOpen", {"path": str(path)}, None, None)
    assert opened["readOnlyStudy"] and opened["document"] == review
    with pytest.raises(ValueError, match="read-only"):
        dispatch(session, "anatomySaveView", {"collectionId": opened["collectionId"]}, None, None)
    assert file_hash(manifest) == checksum
    base.save("unrelated native revision")
    with pytest.raises(ValueError, match="pinned source manifest has changed"):
        dispatch(session, "anatomyOpen", {"path": str(path)}, None, None)
