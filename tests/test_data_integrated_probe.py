"""Independent task-FOV restriction controls; no patient qualification."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('integrated_probe', Path(__file__).parents[1] / 'scripts/data_integrated_probe.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def test_full_shaft_outside_grid_cannot_be_admitted_by_tip_only():
    segments = [(np.array([10., 10., -5.]), np.array([10., 10., 20.]), .5, 'shaft')]
    result = probe.envelope_admission(segments, np.eye(4), np.array([64, 64, 64]))
    assert not result['admitted'] and result['reasons'] == ['shaft:outside-source-grid']


def test_review_mask_hole_is_unknown_even_with_in_grid_endpoints():
    observed = np.ones((32, 32, 32), bool); observed[16, 10, 10] = False
    segments = [(np.array([8., 10., 10.]), np.array([24., 10., 10.]), .5, 'shaft')]
    assert probe.envelope_admission(segments, np.eye(4), np.array(observed.shape))['admitted']
    result = probe.envelope_admission(segments, np.eye(4), np.array(observed.shape), observed)
    assert result['reasons'] == ['shaft:outside-reviewed-observation']


def test_oblique_physical_affine_changes_no_admission_by_array_resize():
    affine = np.array([[0., -2., 0., 60.], [1., 0., 0., 0.], [0., 0., 3., -12.], [0., 0., 0., 1.]])
    voxel = np.array([[10., 10., 10.], [20., 10., 10.]])
    world = voxel @ affine[:3, :3].T + affine[:3, 3]
    assert probe.envelope_admission([(world[0], world[1], .5, 'shaft')], affine, np.array([32, 32, 32]))['admitted']


def test_rotation_sweep_requires_observed_whole_instrument():
    tool = probe.TOOLS['cusa']; parameters = {'radiusMm': 1.5, 'depthMm': 2., 'contactGeometryRevision': 2}
    start = {'position': [45., 45., 45.], 'quaternion': [0., 0., 0., 1.]}
    end = {'position': [45., 45., 45.], 'quaternion': [1., 0., 0., 0.]}
    static = probe.envelopes(tool, parameters, start['position'], start['quaternion'])
    assert probe.envelope_admission(static, np.eye(4), np.array([96, 96, 96]))['admitted']
    swept = probe.swept_envelopes(tool, parameters, start, end)
    assert not probe.envelope_admission(swept, np.eye(4), np.array([96, 96, 96]))['admitted']


def test_finite_sweep_partition_preserves_coverage_without_long_stroke_inflation():
    tool = probe.TOOLS['cusa']; params = {'radiusMm': 1.5, 'depthMm': 2., 'contactGeometryRevision': 2}
    start = {'position': [20., 20., 60.], 'quaternion': [0., 0., 0., 1.]}
    end = {'position': [20., 20., 40.], 'quaternion': [0., 0., 0., 1.]}
    cover = probe.bounded_motion_envelopes(tool, params, start, end)
    # The end shaft starts at z=0 and has physical radius .5: this route remains
    # restricted. Partitioning cannot silently crop that unsupported geometry.
    assert len(cover) == 160
    assert not probe.envelope_admission(cover, np.eye(4), np.array([96, 96, 96]))['admitted']


@pytest.mark.parametrize('change', ['physical-stop', 'non-bone-removal', 'target-removal', 'vessel-contact', 'no-progress'])
def test_public_bone_preview_gate_rejects_scope_expansion(change):
    before = {'drillRemovedVolumeMm3': 145., 'nonTargetRemovedMm3': 145., 'targetRemovedMm3': 0., 'vesselEncounters': 0}
    predicted = {**before, 'drillRemovedVolumeMm3': 146., 'nonTargetRemovedMm3': 146.}
    outcome = {'status': 'partial', 'reason': 'BONE_CUTTING_IN_PROGRESS'}
    assert probe.admissible_bone_preview(before, outcome, predicted)
    if change == 'physical-stop': outcome['reason'] = 'UNSUPPORTED_MATERIAL'
    elif change == 'non-bone-removal': predicted['nonTargetRemovedMm3'] += .1
    elif change == 'target-removal': predicted['targetRemovedMm3'] = .1
    elif change == 'vessel-contact': predicted['vesselEncounters'] = 1
    else: predicted = before.copy()
    assert not probe.admissible_bone_preview(before, outcome, predicted)
