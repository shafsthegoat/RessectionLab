import numpy as np
import pytest

from resectionlab.vessel_fine_growth import FineGrowth, grow_fine_branches, intracranial_envelope, tube_response

SPACING = (0.5, 0.5, 0.5)
SHAPE = (72, 72, 72)


def head():
    """A soft-tissue ball inside a bone shell in air, with a thick trunk along x through the centre."""
    grid = np.indices(SHAPE).astype(np.float32) - 35.5
    radius = np.sqrt((grid ** 2).sum(axis=0))
    image = np.full(SHAPE, -1000., np.float32)
    image[radius < 32] = 1200.
    image[radius < 28] = 40.
    tree = np.zeros(SHAPE, bool)
    tree[20:40, 34:38, 34:38] = True
    image[tree] = 320.
    return image, tree, radius < 28


def test_a_thin_branch_leaving_the_tree_is_followed_and_scored():
    image, tree, envelope = head()
    image[40:54, 35:37, 35:37] = 180.  # A 1 mm branch continuing the trunk.
    score, grown, receipt = grow_fine_branches(image, SPACING, tree, 'CTA', envelope=envelope)
    assert grown[41:52, 35:37, 35:37].all()
    assert not (grown & tree).any()
    assert np.array_equal(score > 0, grown) and score[grown].min() >= 0.5 and score.max() <= 1
    assert receipt['counts']['piecesKept'] == 1 and receipt['counts']['grownVoxels'] == int(grown.sum())
    assert receipt['envelope']['source'] == 'supplied'
    assert 0 < receipt['evidence']['contrast'] < 1  # Dimmer than the trunk, brighter than the tissue around it.


def test_tubes_that_do_not_touch_the_tree_blobs_and_bone_edges_are_left_out():
    image, tree, envelope = head()
    image[40:54, 35:37, 35:37] = 180.            # Kept: continues the trunk.
    image[24:40, 50:52, 20:22] = 180.            # A tube of the same size that touches nothing.
    image[22:32, 38:48, 30:42] = 200.            # A blob resting on the trunk.
    image[30:44, 36:38, 62:64] = 180.            # A tube lying against the bone.
    _, grown, receipt = grow_fine_branches(image, SPACING, tree, 'CTA', envelope=envelope)
    assert grown[42:52, 35:37, 35:37].all()
    assert not grown[24:40, 50:52, 20:22].any()
    assert not grown[24:30, 41:46, 33:40].any()
    assert not grown[30:44, 36:38, 62:64].any()
    assert receipt['counts']['thickVoxelsExcluded'] > 0


def test_mucosa_between_air_cells_is_not_a_vessel():
    image, tree, envelope = head()
    image[40:54, 35:37, 35:37] = 180.
    image[40:54, 30:35, 30:42] = -1000.  # An air cell beside the branch.
    _, grown, _ = grow_fine_branches(image, SPACING, tree, 'CTA', envelope=envelope)
    assert not grown[42:52, 35:37, 35:37].any()


def test_the_rule_is_refused_for_anything_but_ct_and_for_mismatched_grids():
    image, tree, envelope = head()
    for modality in ('TOF', 'MRA', 'T1'):
        with pytest.raises(ValueError, match='Hounsfield'):
            grow_fine_branches(image, SPACING, tree, modality, envelope=envelope)
    with pytest.raises(ValueError, match='grid'):
        grow_fine_branches(image, SPACING, tree[:-1], 'CT', envelope=envelope)
    with pytest.raises(ValueError, match='non-empty'):
        grow_fine_branches(image, SPACING, np.zeros(SHAPE, bool), 'CT', envelope=envelope)
    with pytest.raises(ValueError):
        FineGrowth(low=10, high=5)


def test_nothing_is_grown_without_an_envelope_or_a_branch():
    image, tree, envelope = head()
    score, grown, receipt = grow_fine_branches(image, SPACING, tree, 'CT', envelope=np.zeros(SHAPE, bool))
    assert not grown.any() and not score.any() and 'reason' in receipt
    _, grown, receipt = grow_fine_branches(image, SPACING, tree, 'CT', envelope=envelope)
    assert not grown.any() and receipt['counts']['piecesKept'] == 0


def test_the_estimated_envelope_is_the_tissue_inside_the_bone_and_supports_the_growth():
    image, tree, inside = head()
    image[40:54, 35:37, 35:37] = 180.
    # Soft tissue outside the skull must not become intracranial.
    scalp = np.zeros(SHAPE, bool)
    scalp[2:6, 20:52, 20:52] = True
    image[scalp] = 40.
    envelope = intracranial_envelope(image, SPACING)
    assert envelope[inside].mean() > 0.95 and not envelope[scalp].any() and not envelope[image >= 700].any()
    _, grown, receipt = grow_fine_branches(image, SPACING, tree, 'CT')
    assert grown[42:52, 35:37, 35:37].all() and receipt['envelope']['source'].startswith('estimated')


def test_the_tube_response_prefers_tubes_to_sheets_and_blobs():
    image = np.full((40, 40, 40), 40., np.float32)
    image[5:35, 9:11, 9:11] = 200.     # Tube.
    image[5:35, 5:35, 29:31] = 200.    # Sheet.
    image[16:24, 24:32, 12:20] = 200.  # Blob.
    response = tube_response(image, SPACING, np.ones(image.shape, bool))
    tube, sheet, blob = response[20, 9:11, 9:11].max(), response[20, 20, 29:31].max(), response[20, 28, 16]
    assert tube > 4 * max(sheet, blob, 1e-3)
