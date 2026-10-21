"""Physical ledgers and reward-farming regressions (engineering fixtures)."""
from dataclasses import replace
import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_rewards import RewardConfig, RewardTracker, compute_reward


def engine(affine=None):
    labels = np.ones((3, 3, 3), np.uint16)
    labels[1, 1, 1] = 2
    labels[2, :, :] = 3
    regions = [dict(id=1, name='arbitrary healthy', material='soft_tissue', target=False),
               dict(id=2, name='not a tumor name', material='tumor', target=True),
               dict(id=3, name='tumor', material='vessel', target=True)]
    return SurgicalEngine(SurgicalScene(labels, np.eye(4) if affine is None else affine, regions, {}, {}))


def test_unique_source_removal_and_critical_target_exclusion():
    e = engine(); tracker = RewardTracker(e, protected_source_ids=[13])
    before = tracker.measure(e)
    e.state['removed'] = [13, 13, 18]
    after = tracker.measure(e)
    assert after['tumor_removed_mm3'] == 0
    assert after['healthy_removed_mm3'] == 2
    reward, terms = compute_reward(before, after, {'status': 'accepted'})
    assert reward < 0 and terms['tumor_removal'] == 0
    _, terms = compute_reward(after, tracker.measure(e), {'status': 'accepted'})
    assert terms['healthy_removal'] == 0


def test_target_credit_independent_of_display_label():
    e = engine(); tracker = RewardTracker(e)
    before = tracker.measure(e)
    e.state['removed'] = [13]
    after = tracker.measure(e)
    assert after['tumor_removed_mm3'] == 1
    assert compute_reward(before, after, {'status': 'accepted'})[1]['tumor_removal'] > 0
    e.by_label[2]['name'] = 'critical sounding name'
    assert tracker.measure(e) == after


def test_physical_face_area_union_and_severity_not_clot():
    affine = np.eye(4); affine[:3, :3] = [[2, 1, 0], [0, 3, 0], [0, 0, 4]]
    e = engine(affine); tracker = RewardTracker(e)
    before = tracker.measure(e)
    e.state['cutEdges'] = [[0, 9], [9, 0]]
    e.state['transectedEdges'] = [[0, 9]]
    e.state['vesselsV2'] = {'faces': {'18:21': {'cells': [18, 21], 'damage': .5, 'clot': 0}}}
    after = tracker.measure(e)
    assert after['healthy_cut_mm2'] == pytest.approx(np.sqrt(160))
    assert after['vessel_damage_mm2'] == 4
    e.state['vesselsV2']['faces']['18:21']['clot'] = 1
    assert tracker.measure(e) == after
    e.state['vesselsV2']['faces']['18:21']['damage'] = 1
    _, terms = compute_reward(after, tracker.measure(e), {'status': 'accepted'})
    assert terms['vessel_damage'] < 0


def test_bleeding_suction_irrigation_and_replay():
    e = engine(); tracker = RewardTracker(e); before = tracker.measure(e)
    e.state['fluidAddedMm3'] = 12
    e.state['fluidRemovedMm3'] = 12
    e.history.append({'events': [{'kind': 'irrigation', 'volumeMm3': 5}]})
    after = tracker.measure(e)
    assert after['bleeding_generated_mm3'] == 7
    assert compute_reward(before, after, {'status': 'accepted'})[1]['bleeding'] < 0
    assert RewardTracker(e).measure(e) == after
    assert tracker.measure(e) == after
    e.state['fluidRemovedMm3'] = 100
    assert tracker.measure(e) == after


def test_strain_recovery_does_not_pay_and_repeated_strain_does_not_refund():
    e = engine(); tracker = RewardTracker(e); before = tracker.measure(e)
    e.state['deformation'] = {'0': [1., 0., 0.]}
    injured = tracker.measure(e)
    assert injured['deformation_injury_mm3'] > 0
    assert compute_reward(before, injured, {'status': 'accepted'})[1]['deformation'] < 0
    e.state['deformation'] = {}
    recovered = tracker.measure(e)
    assert recovered == injured
    e.state['deformation'] = {'0': [1., 0., 0.]}
    assert tracker.measure(e) == injured


def test_noop_rejection_and_protection_are_never_positive():
    e = engine(); tracker = RewardTracker(e); before = tracker.measure(e)
    reward, _ = compute_reward(before, before, {'status': 'accepted'})
    assert reward < 0
    rejected, _ = compute_reward(before, before, {'status': 'rejected'})
    assert rejected < reward
    after = tracker.measure(e, {'protected_damage_mm3': 2, 'near_contact_exposure_mm3_seconds': 3})
    _, terms = compute_reward(before, after, {'status': 'accepted'})
    assert terms['protected_damage'] < 0 and terms['near_contact'] < 0
    assert tracker.measure(e) == after


@pytest.mark.parametrize('field,value', [('action_cost', float('nan')), ('tumor_removal_weight', float('inf')),
    ('area_scale_mm2', 0), ('healthy_removal_weight', -1), ('time_scale_seconds', True)])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        replace(RewardConfig(), **{field: value})


def test_blocked_protection_penalty_and_engine_replay():
    e = engine(); tracker = RewardTracker(e); before = tracker.measure(e)
    _, terms = compute_reward(before, before, {'status': 'rejected', 'reason': 'PROTECTED_STRUCTURE: margin'})
    assert terms['protected_blocked'] == -RewardConfig().protected_blocked_weight
    outcome = e.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    after = tracker.measure(e)
    replay = e.replay(e.actions)
    assert RewardTracker(replay).measure(replay) == after
    assert compute_reward(before, after, outcome)[0] <= 0


def test_affine_volume_normalization():
    affine = np.diag([2., 3., 4., 1.]); e = engine(affine)
    tracker = RewardTracker(e); before = tracker.measure(e)
    e.state['removed'] = [13]
    after = tracker.measure(e)
    assert after['tumor_removed_mm3'] == pytest.approx(24.)
    _, terms = compute_reward(before, after, {'status': 'accepted'}, RewardConfig(volume_scale_mm3=24.))
    assert terms['tumor_removal'] == pytest.approx(1.)


def test_microstep_out_and_back_peak_and_snapshot_reconstruction():
    from resectionlab.surgical_rl_rewards import deformation_scores
    import json
    e = engine(); tracker = RewardTracker(e); before = tracker.measure(e)
    e.state['deformation'] = {'0': [1., 0., 0.]}
    peaks = deformation_scores(e)
    assert peaks[0] > 0
    e.state['deformation'] = {}
    assert deformation_scores(e) == {}
    protection = {'deformation_cell_peaks': peaks}
    after = tracker.measure(e, protection)
    assert after['deformation_injury_mm3'] > 0
    assert compute_reward(before, after, {'status': 'accepted'})[1]['deformation'] < 0
    assert tracker.measure(e) == after
    saved = json.loads(json.dumps(protection))
    assert RewardTracker(e).measure(e, saved) == after
    assert tracker.measure(e, saved) == after


@pytest.mark.parametrize('peaks', [{0: float('nan')}, {-1: 1}, {0: -1}])
def test_invalid_microstep_peak(peaks):
    e = engine()
    with pytest.raises(ValueError):
        RewardTracker(e).measure(e, {'deformation_cell_peaks': peaks})


def test_bone_access_cost_separate_from_healthy_damage_and_no_refunds():
    e = engine()
    # Explicit mechanical material, independent of label text and target flag.
    e.by_label[1]['material'] = 'bone'
    e.by_label[1]['target'] = True
    tracker = RewardTracker(e); before = tracker.measure(e)
    e.state['removed'] = [0, 1, 13, 18]
    after = tracker.measure(e)
    assert after['bone_removed_mm3'] == 2
    assert after['healthy_removed_mm3'] == 1
    assert after['tumor_removed_mm3'] == 1
    _, terms = compute_reward(before, after, {'status': 'accepted'})
    assert terms['bone_removal'] < 0
    assert abs(terms['bone_removal']) < abs(terms['healthy_removal'])
    assert compute_reward(after, tracker.measure(e), {'status': 'accepted'})[1]['bone_removal'] == 0
    # Monotone episode accounting does not refund restored or reset geometry.
    e.state['removed'] = []
    assert tracker.measure(e) == after
    assert compute_reward(after, tracker.measure(e), {'status': 'accepted'})[1]['bone_removal'] == 0


def test_healthy_thermal_exposure_cost_is_monotone_and_volume_weighted():
    e = engine(np.diag([2., 1., 1., 1.])); tracker = RewardTracker(e)
    before = tracker.measure(e)
    e.state['thermal'] = {'0': 3., '13': 10.}
    after = tracker.measure(e)
    assert after['healthy_thermal_exposure'] == 6.  # Tumor thermal not rewarded or charged.
    _, terms = compute_reward(before, after, {'status': 'accepted'})
    assert terms['healthy_thermal'] == pytest.approx(-.006)
    e.state['thermal'] = {}
    assert tracker.measure(e) == after
    e.state['thermal'] = {'0': 3.}
    assert tracker.measure(e) == after
    e.state['thermal']['1'] = 2.
    assert tracker.measure(e)['healthy_thermal_exposure'] == 10.


def test_thermal_protected_target_overlap_charged_bone_excluded():
    e = engine(); e.by_label[1]['material'] = 'bone'
    tracker = RewardTracker(e, protected_source_ids=[13]); before = tracker.measure(e)
    e.state['thermal'] = {'0': 100., '13': 2.}
    after = tracker.measure(e)
    assert after['healthy_thermal_exposure'] == 2.
    assert compute_reward(before, after, {'status': 'accepted'})[1]['healthy_thermal'] < 0


def test_lattice_hold_physical_strain_and_release_history():
    from resectionlab.surgical_rl_rewards import deformation_scores
    e = engine(np.diag([2., 1., 1., 1.])); tracker = RewardTracker(e)
    before = tracker.measure(e)
    e.state['holds'] = {'left': {'cells': [0], 'offset': [0, 0, 0]}}
    assert deformation_scores(e) == {}
    assert tracker.measure(e) == before
    e.state['holds']['left']['offset'] = [1, 0, 0]
    # Physical displacement 2 mm, nearest y/z neighbor 1 mm: squared strain 4 * volume 2.
    assert deformation_scores(e)[0] == pytest.approx(8.)
    after = tracker.measure(e)
    assert compute_reward(before, after, {'status': 'accepted'})[1]['deformation'] < 0
    e.state['holds'] = {}
    assert tracker.measure(e) == after


def test_v2_displacements_override_lattice_holds_and_removed_cells_ignored():
    from resectionlab.surgical_rl_rewards import deformation_scores
    e = engine()
    e.state['holds'] = {'left': {'cells': [0], 'offset': [1, 0, 0]}}
    e.state['deformation'] = {'0': [0., 0., 0.]}
    assert deformation_scores(e) == {}
    e.state['deformation'] = {}
    e.state['removed'] = [0]
    assert deformation_scores(e) == {}
