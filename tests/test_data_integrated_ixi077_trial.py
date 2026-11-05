"""The exact fixed-trial gate checks attained source work and rejects stale repeats."""
import copy
from pathlib import Path

import pytest


@pytest.fixture
def receipt(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    from data_integrated_ixi077_trial import verify_preview
    route = {'A_ras_mm': [0., 0., 0.], 'T_ras_mm': [0., 0., 10.], 'orientation_xyzw': [0., 0., 0., 1.]}
    intervals = [(0., .5, {10}, set()), (.5, 1., {11}, {20})]
    slot = {'position': [0., 0., 2.], 'quaternion': route['orientation_xyzw'], 'active': False,
            'inserted': False, 'grip': [], 'parameters': {}}
    fresh = {'left': {**slot, 'instrument': 'cranial_drill'}, 'right': {**slot, 'instrument': 'cusa'}}
    preview_slots = copy.deepcopy(fresh)
    preview_slots['left']['contactGeometry'] = [
        {'a': [0., 0., -40.], 'b': [0., 0., -2.], 'radiusMm': .5, 'part': 'shaft'},
        {'a': [0., 0., -2.], 'b': [0., 0., 0.], 'radiusMm': 2., 'part': 'working'},
    ]
    preview_slots['right']['contactGeometry'] = [
        {'a': [0., 0., -40.], 'b': [0., 0., 0.], 'radiusMm': .5, 'part': 'shaft'},
        {'a': [0., 0., 0.], 'b': [0., 0., 0.], 'radiusMm': 1.5, 'part': 'working'},
    ]
    preview_slots['right']['inserted'] = True
    metrics = {name: 0. for name in ('vesselEncounters', 'activeVesselWounds', 'severedVesselConnections',
                  'modeledVesselFluidGeneratedMm3', 'brainMaxDisplacementMm', 'drillInputEnergyJ', 'collisions')}
    before = {'metrics': {**metrics, 'removedByLabelMm3': {}, 'cusaInputCapacityMm3': 0.},
              'slots': fresh, 'removed': []}
    preview = {'status': 'partial', 'reason': 'TISSUE_ASPIRATION_IN_PROGRESS', 'stateId': 'new',
               'deltas': {'removedCells': [], 'geometryRevision': 'new'}, 'events': [],
               'view': {'stateId': 'new', 'toolMechanicsRevision': 2, 'physicalClock': {'revision': 1},
                        'physics': {'cusaV3': {'profile': {'capacityMm3PerSecond': 2.}}},
                        'metrics': {**metrics, 'removedByLabelMm3': {}, 'cusaInputCapacityMm3': 60.},
                        'slots': preview_slots, 'holds': {}, 'moved': [], 'deformation': {}, 'removed': [],
                        'cusaV3': {'progressMm3': {'10': .3}, 'effectiveActivationSeconds': 30.}}}
    def check(candidate):
        return verify_preview(before, candidate, {'operation': 'apply', 'duration': 30.}, route, intervals, fresh, 1.)
    return check, preview


def test_pending_work_on_attained_segment_is_progress(receipt):
    check, preview = receipt
    result = check(preview)
    assert result['attained_fraction'] == [.2, .2]
    assert result['new_pending_source_ids'] == [10]
    assert result['pending_gain_mm3'] == .3


@pytest.mark.parametrize(('field', 'match'), [
    ('off_line', 'left the fixed line'),
    ('off_segment_work', 'attained segment authority'),
    ('stale', 'WITHOUT_ACHIEVED_TRAVEL_OR_WORK_PROGRESS'),
    ('latch', 'latched'),
])
def test_off_line_stale_or_unauthorized_physical_previews_fail_closed(receipt, field, match):
    check, preview = receipt
    if field == 'off_line':
        preview['view']['slots']['right']['position'] = [1., 0., 2.]
    elif field == 'off_segment_work':
        preview['view']['cusaV3']['progressMm3'] = {'20': .3}
    elif field == 'stale':
        preview['view']['cusaV3']['progressMm3'] = {}
    elif field == 'latch':
        preview['view']['slots']['right']['active'] = True
    with pytest.raises(ValueError, match=match):
        check(preview)
