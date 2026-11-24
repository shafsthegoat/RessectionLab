import json
from types import SimpleNamespace

import numpy as np
import pytest

from resectionlab.anatomy_bridge import VASCULAR_REVIEW_MANIFEST, dispatch
from resectionlab.patient_anatomy import PatientImagingCollection, digest, file_hash
from resectionlab.vascular_cleanup import HIDDEN, RETAINED, UNCERTAIN, classify_components, restore_hidden_components


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
    restored, revised = restore_hidden_components(partition, report, [component["id"]])
    np.testing.assert_array_equal(restored > 0, mask)
    assert not np.any(restored == HIDDEN)
    assert report["hidden_voxels"] == component["voxels"]
    assert revised["uncertain_voxels"] == report["uncertain_voxels"] + component["voxels"]
    with pytest.raises(ValueError, match="Unknown or visible source component"):
        restore_hidden_components(partition, report, ["component-0"])


def test_no_source_support_abstains_and_preserves_all_components():
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[5:12, 5, 5] = True
    mask[12, 12, 12] = True
    partition, report = classify_components(mask, np.ones(mask.shape), None, np.eye(4))
    np.testing.assert_array_equal(partition > 0, mask)
    assert report["hidden_components"] == 0
    assert report["reference_signal_available"] is False


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
    session = SimpleNamespace()
    opened = dispatch(session, "anatomyOpen", {"path": str(path)}, None, None)
    assert opened["readOnlyStudy"] and opened["document"] == review
    with pytest.raises(ValueError, match="read-only"):
        dispatch(session, "anatomySaveView", {"collectionId": opened["collectionId"]}, None, None)
    assert file_hash(manifest) == checksum
    base.save("unrelated native revision")
    with pytest.raises(ValueError, match="pinned source manifest has changed"):
        dispatch(session, "anatomyOpen", {"path": str(path)}, None, None)
