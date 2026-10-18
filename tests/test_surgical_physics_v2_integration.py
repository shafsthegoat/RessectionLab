"""Public-action integration checks for explicitly opted-in V2 research scenes."""
import copy
import numpy as np
import pytest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_physics_v2 import default_profile


def scene(material='vessel', block=False):
    labels = np.zeros((24, 24, 20), np.uint16)
    if block: labels[6:11, 6:11, 8] = 1
    else: labels[8, 7:10, 8] = 1
    return SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'Synthetic tissue', 'material': material}], {},
                         {'tissuePhysicsV2': default_profile()})


def action(operation, position=None, **extra):
    return {'slot': 'left', 'operation': operation, 'duration': .2,
            **({'pose': {'position': position, 'quaternion': [0., 0., 0., 1.]}} if position else {}), **extra}


def accepted(e, a):
    r = e.step(a)
    assert r['status'] == 'accepted', r
    return r


def prepare_tool(e, tool):
    accepted(e, action('exchange', instrument=tool))
    accepted(e, action('place', [8., 8., 4.], parameters={'radiusMm': 1.5, 'jawOpeningMm': 0.}))


def test_public_brain_hold_couples_neighbors_and_release_recovers_without_source_edits():
    source = scene('soft_tissue', True); baseline = source.labels.copy(); props = copy.deepcopy(source.properties)
    e = SurgicalEngine(source); prepare_tool(e, 'brain_retractor')
    accepted(e, action('hold', [8., 8., 7.]))
    accepted(e, action('move', [8.4, 8., 7.], duration=.4))
    held = set(e.state['holds']['left']['cells']); field = e.state['deformation']
    assert any(int(c) not in held and np.linalg.norm(v) > 0 for c, v in field.items())
    before = max(np.linalg.norm(v) for v in field.values())
    accepted(e, action('release', duration=.01))
    after = max(np.linalg.norm(v) for v in e.state['deformation'].values())
    assert 0 < after < before
    accepted(e, action('inspect', duration=1.))
    assert max(np.linalg.norm(v) for v in e.state['deformation'].values()) < after
    assert np.array_equal(source.labels, baseline) and source.properties == props
    assert SurgicalEngine.restore(source, e.snapshot()).state_id == e.state_id


def test_drill_pending_work_blocks_advance_and_preview_replays_energy():
    e = SurgicalEngine(scene('bone')); prepare_tool(e, 'cranial_drill')
    a = action('apply', [8., 8., 10.], parameters={'rpm': 1000., 'forceN': .1, 'duty': 1.}, duration=.3)
    before = e.state_id; preview = e.preview(a)
    assert e.state_id == before
    result = e.step(a)
    assert result['status'] == 'partial' and result['reason'] == 'BONE_CUTTING_IN_PROGRESS'
    assert preview['stateId'] == result['stateId']
    assert e.state['slots']['left']['position'][2] < 8 and not e.state['removed']
    d = e.state['drillingV2']
    assert sum(d['workJ'].values()) > 0
    assert d['inputEnergyJ'] == pytest.approx(d['cuttingWorkJ'] + d['unusedEnergyJ'])
    assert d['heatGeneratedJ'] == pytest.approx(d['cuttingWorkJ'] * e.scene.properties['tissuePhysicsV2']['bone']['heatFraction'])
    assert SurgicalEngine.restore(e.scene, e.snapshot()).state_id == e.state_id


def test_public_vessel_transection_coagulation_clock_and_conservation():
    e = SurgicalEngine(scene()); prepare_tool(e, 'micro_scissors')
    cut = action('apply', [8., 8., 7.], duration=1.)
    before = e.state_id; preview = e.preview(cut); accepted(e, cut)
    assert e.state_id == preview['stateId'] and before != e.state_id
    assert e.state['transectedEdges'] and e.state['vesselsV2']['wounds'] and not e.state['removed']
    v = e.state['vesselsV2']; assert v['generatedMm3'] > 0
    assert v['initialMm3'] + v['refilledMm3'] == pytest.approx(v['remainingMm3'] + v['generatedMm3'])
    generated = v['generatedMm3']; elapsed = v['elapsedSeconds']
    accepted(e, action('inspect', duration=.4))
    assert e.state['vesselsV2']['generatedMm3'] > generated
    assert e.state['vesselsV2']['elapsedSeconds'] == pytest.approx(elapsed + .4)
    accepted(e, action('withdraw'))
    prepare_tool(e, 'bipolar_forceps'); accepted(e, action('apply', [8., 8., 7.], parameters={'power': 10.}, duration=1.))
    assert e.state['transectedEdges']
    assert any(not s['bleedingActive'] for s in e.state['sources'])
    assert SurgicalEngine.restore(e.scene, e.snapshot()).state_id == e.state_id


@pytest.mark.parametrize('mutate', [lambda p:p.pop('brain'), lambda p:p['brain'].pop('dragNsPerMm'),
    lambda p:p['brain'].update(extra=1), lambda p:p['bone'].update(maxRpm=float('nan')),
    lambda p:p['vessel'].update(reservoirMm3=0), lambda p:p.update(schema='unknown')])
def test_malformed_explicit_profiles_rejected(mutate):
    s = scene(); mutate(s.properties['tissuePhysicsV2'])
    with pytest.raises(ValueError): SurgicalEngine(s)


def test_common_raster_retains_duplicate_soft_source_identities():
    e = SurgicalEngine(scene('soft_tissue'))
    a = int(np.ravel_multi_index((8, 7, 8), e.scene.shape))
    b = int(np.ravel_multi_index((8, 8, 8), e.scene.shape))
    # Deliberately compressed fixture: two source identities share one displayed
    # grid cell. Both must remain removable/contactable, with no volume invented.
    e.state['deformation'] = {str(a): [0., .6, 0.], str(b): [0., 0., 0.]}
    pairs = dict(e.view()['moved'])
    assert pairs[a] == pairs[b] == b
    touched = e.touched([8., 8., 8.], [8., 8., 8.], .1)
    assert touched == [a, b]
    assert int(e.occupancy()[b]) - 1 in touched
    assert not e.state['removed']


def test_public_cut_and_removal_survive_relaxation_and_replay():
    e = SurgicalEngine(scene('soft_tissue', True)); prepare_tool(e, 'scalpel')
    accepted(e, action('apply', [8., 8., 7.]))
    cut = copy.deepcopy(e.state['cutEdges']); assert cut
    accepted(e, action('withdraw')); prepare_tool(e, 'brain_retractor')
    accepted(e, action('hold', [8., 8., 7.]))
    accepted(e, action('move', [8.2, 8., 7.]))
    accepted(e, action('release')); accepted(e, action('withdraw'))
    prepare_tool(e, 'cusa'); accepted(e, action('apply', [8., 8., 7.]))
    removed = e.state['removed'][:]; assert removed
    accepted(e, action('inspect', duration=1.))
    assert e.state['removed'] == removed and e.state['cutEdges'] == cut
    assert not set(map(str, removed)) & set(e.state['deformation'])
    assert not set(removed) & set(e.touched([8., 8., 7.], [8., 8., 7.], 1.5))
    assert SurgicalEngine.restore(e.scene, e.snapshot()).state_id == e.state_id

def test_work_limited_dwell_is_not_collision_and_branch_has_independent_work_ledger():
    e = SurgicalEngine(scene('bone')); prepare_tool(e, 'cranial_drill')
    start = len(e.actions)
    command = action('apply', [8., 8., 10.], parameters={'rpm':1000., 'forceN':.1}, duration=.3)
    assert e.step(command)['reason'] == 'BONE_CUTTING_IN_PROGRESS'
    assert e.metrics()['collisions'] == 0
    original = e.state_id
    branch = e.branch(start)
    assert not branch.state['drillingV2']['workJ']
    assert branch.step({**command, 'parameters':{'rpm':2000., 'forceN':.2}})['status'] == 'partial'
    assert branch.state['drillingV2']['cuttingWorkJ'] > e.state['drillingV2']['cuttingWorkJ']
    assert e.state_id == original
    assert SurgicalEngine.restore(branch.scene, branch.snapshot()).state_id == branch.state_id

def test_drill_cannot_skip_stalled_waypoint_to_later_free_space_leg():
    e = SurgicalEngine(scene('bone')); prepare_tool(e, 'cranial_drill')
    result = e.step({'slot':'left', 'operation':'apply', 'duration':.5,
        'parameters':{'rpm':100., 'forceN':.01},
        'path':[{'position':p, 'quaternion':[0.,0.,0.,1.]} for p in ([8.,8.,10.],[14.,8.,4.])]})
    assert result['status'] == 'partial' and result['reason'] == 'BONE_CUTTING_IN_PROGRESS'
    assert e.state['slots']['left']['position'][0] == 8.
    assert result['elapsedSeconds'] < .5
    assert 0 < result['completedFraction'] < .5
    assert e.metrics()['collisions'] == 0

def test_drill_geometry_cannot_exceed_declared_cutting_radius():
    e = SurgicalEngine(scene('bone')); before = e.state_id
    r = e.step(action('apply', parameters={'radiusMm':12.}))
    assert r['status'] == 'rejected' and r['reasonCode'] == 'DRILL_RADIUS_EXCEEDS_PROFILE'
    assert e.state_id == before
