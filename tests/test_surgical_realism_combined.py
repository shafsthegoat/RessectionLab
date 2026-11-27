"""Injury, bleeding, stimulation mapping, navigation and brain shift together on the generated case.

The generated layered convexity case is engineering geometry, not patient data.
The opening is taken as already reached (scalp and dura reflected, bone flap
parked) by every engine built here, so replay and restore start from it too.
"""
import json
import math

import numpy as np
import pytest
from scipy.spatial import cKDTree

from scripts.generated_glioma_walkthrough_case import build_glioma_case
from resectionlab import surgical_brain_shift as shift
from resectionlab import surgical_functional_mapping as fm
from resectionlab import surgical_navigation as navigation
from resectionlab import surgical_operative_exposure as exposure
from resectionlab import surgical_vessel_bleeding as vb
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_injury import PROPERTY as INJURY, default_profile
from resectionlab.surgical_scene import SurgicalScene, propagate_original_target_memberships

Q = [1., 0., 0., 0.]   # instrument axis pointing down into the opening
SPLIT = 20


def pose(p):
    return {'position': [float(v) for v in p], 'quaternion': Q}


def move(slot, p, duration=1.):
    return {'slot': slot, 'operation': 'move', 'duration': duration, 'path': [pose(p)]}


def tool(slot, name, at):
    return [{'slot': slot, 'operation': 'exchange', 'instrument': name, 'duration': 1.},
            {'slot': slot, 'operation': 'place', 'duration': 1., 'path': [pose(at)]}]


def switch(slot, operation):
    return {'slot': slot, 'operation': operation, 'duration': .1}


def withdraw(slot):
    return {'slot': slot, 'operation': 'withdraw', 'duration': .5}


def inspect(duration=.5, slot='left'):
    return {'slot': slot, 'operation': 'inspect', 'duration': duration}


ACTIONS = [
    inspect(1.), inspect(1.5),
    # Navigation pointer onto the sagged cortex.
    *tool('left', 'navigation_pointer', (27, 33, 82)), move('left', (27, 33, 60)), inspect(.3), move('left', (27, 33, 82)),
    withdraw('left'),
    # A passive dissector forced through the lesion: laceration, not treatment.
    *tool('left', 'dissector', (28, 32, 82)), move('left', (28, 32, 46), 2.), move('left', (28, 32, 82)), withdraw('left'),
    # An active CUSA down beside the cortical artery.
    *tool('right', 'cusa', (38, 28, 82)), switch('right', 'activate'), move('right', (38, 28, 50), 2.),
    switch('right', 'deactivate'), move('right', (38, 28, 82)), withdraw('right'),
    # Subcortical stimulation mapping.
    *tool('left', 'subcortical_probe', (26, 35, 82)), move('left', (26, 35, 52.6)),
    {'slot': 'left', 'operation': 'apply', 'duration': .5, 'parameters': {'amplitudeMa': 3.}},
    {'slot': 'left', 'operation': 'apply', 'duration': .5, 'parameters': {'amplitudeMa': 10.}},
    move('left', (26, 35, 82)), withdraw('left'),
    # Bipolar at the artery.
    *tool('right', 'bipolar_forceps', (38, 28, 82)), move('right', (38, 28, 52)),
    {'slot': 'right', 'operation': 'apply', 'duration': 1., 'parameters': {'power': 5.}},
    move('right', (38, 28, 82)), withdraw('right'),
    # A second CUSA pass and waiting while the field bleeds.
    *tool('right', 'cusa', (36, 32, 82)), switch('right', 'activate'), move('right', (36, 32, 49), 2.),
    switch('right', 'deactivate'), move('right', (36, 32, 82)), inspect(1.),
]


@pytest.fixture(scope='module')
def realism():
    _, base, _ = build_glioma_case()
    properties = {**base.properties,
                  'brainShiftV1': shift.default_property(base, settlingSeconds=2.0, settlingSteps=2),
                  'navigationV1': navigation.default_property(seed=11), INJURY: default_profile(),
                  vb.PROPERTY: vb.declare(base.labels, base.regions, base.affine),
                  fm.PROPERTY: fm.declaration([{'region': 'Subcortical white matter', 'family': 'motor', 'id': 'wm'}])}
    return propagate_original_target_memberships(base, SurgicalScene(
        base.labels, base.affine, base.regions, base.source, properties))


@pytest.fixture
def opened(monkeypatch):
    original = exposure.initialize

    def initialize(engine):
        original(engine)
        bodies = engine.state['operativeExposureV1']['bodies']
        bone = engine.scene.properties['operativeExposureV1']['bodies']['bone']
        bodies['scalp'].update(angleRad=math.pi / 2, status='supported')
        bodies['bone'].update(status='parked', translationMm=np.subtract(bone['parkCenterMm'], bone['restCenterMm']).tolist())
        bodies['dura'].update(angleRad=math.pi / 2, status='supported')
    monkeypatch.setattr(exposure, 'initialize', initialize)


def run(scene, actions):
    engine = SurgicalEngine(scene)
    return engine, [engine.step(action) for action in actions]


def test_forty_mixed_actions_replay_restore_and_respect_injury_and_shift(realism, opened):
    assert len(ACTIONS) == 40
    first, results = run(realism, ACTIONS)
    second, again = run(realism, ACTIONS)
    ids = [r['stateId'] for r in results]
    assert ids == [r['stateId'] for r in again] and first.state_id == second.state_id
    statuses = {r['status'] for r in results}
    assert 'accepted' in statuses and statuses - {'accepted'}
    # Snapshot mid-way, restore by verified replay, continue: the same states.
    middle, _ = run(realism, ACTIONS[:SPLIT])
    assert middle.state_id == ids[SPLIT - 1]
    restored = SurgicalEngine.restore(realism, json.loads(json.dumps(middle.snapshot())), use_cache=False,
                                      force_verify=True)
    assert restored.state_id == middle.state_id
    assert [restored.step(action)['stateId'] for action in ACTIONS[SPLIT:]] == ids[SPLIT:]
    assert first.replay(first.actions).state_id == first.state_id

    engine = first
    scene = engine.scene
    labels = scene.labels.ravel()
    assert engine.state['brainShiftV1']['level'] == 2 and engine.state['navigationV1']['readings']

    # Defect 1: lacerated lesion is gone from the cavity but earns no treatment credit.
    metrics = engine.metrics()
    targets = {r['id'] for r in scene.regions if r.get('target')}
    lacerated = set(engine.state['injuryV1']['lacerated'])
    removed = set(engine.state['removed'])
    torn = sum(int(labels[c]) in targets for c in lacerated)
    treated = sum(int(labels[c]) in targets for c in removed - lacerated)
    assert torn > 0 and lacerated <= removed
    assert metrics['targetRemovedMm3'] == treated * scene.voxel_mm3
    assert metrics['injuryLaceratedTargetMm3'] == torn * scene.voxel_mm3
    assert metrics['targetRemovedMm3'] + metrics['injuryLaceratedTargetMm3'] + metrics['nonTargetRemovedMm3'] == \
        pytest.approx(len(removed) * scene.voxel_mm3)

    # Defect 2: every open wound bleeds from where the vessel lies now.
    current = shift.active_map(engine)
    bleeding = engine.state[vb.PROPERTY]
    assert bleeding['wounds'] and bleeding['bloodLossMm3'] > 0
    occupancy = engine.occupancy()
    reach = int(scene.properties[vb.PROPERTY]['parameters']['outletSearchVoxels']) + 1
    for wound in bleeding['wounds'].values():
        outlet = vb._outlet(engine, occupancy, wound['cells'])
        if outlet is None:
            continue
        now = np.column_stack(np.unravel_index(current.forward(wound['cells']), scene.shape))
        here = np.asarray(np.unravel_index(outlet, scene.shape))
        assert np.abs(now - here).max(axis=1).min() <= reach

    # Defect 3: mapping rows measured the tract where it lies now.
    wm = np.flatnonzero(labels == 8)
    retained = wm[~np.isin(wm, list(removed))]
    rows = [o for o in engine.state['observations'] if o.get('kind') == 'model-estimate']
    assert rows and any(row['result'] == 'model_estimate_response' for row in rows)
    shifted_tree = cKDTree(scene.points(current.forward(retained)))
    unshifted_tree = cKDTree(scene.points(retained))
    sites = {tuple(row['position']) for row in rows}
    differs = False
    for site in sorted(sites):
        estimate = fm.stimulation_observation(engine, 'subcortical', np.asarray(site), [], {'amplitudeMa': 3.})
        expected = shifted_tree.query(site)[0]
        reported = estimate['tracts'][0]['estimatedDistanceMm']
        assert reported == pytest.approx(expected) if expected <= 20. else reported is None
        differs |= abs(expected - unshifted_tree.query(site)[0]) > 1.
    assert differs
    harm = fm.structure_harm(engine)['tracts']['wm']
    cavity = np.asarray([c for c in sorted(removed) if engine.by_label[int(labels[c])]['material'] not in
                         ('bone', 'membrane')], dtype=np.int64)
    distances = cKDTree(scene.points(current.forward(cavity))).query(scene.points(current.forward(retained)))[0]
    assert harm['band0To2Mm']['cells'] == int(np.count_nonzero(distances <= 2. + 1e-9))
    assert harm['minDistanceMm'] == (0. if harm['removedMm3'] else pytest.approx(distances.min()))


def test_patient_position_is_part_of_the_signed_implementation(monkeypatch):
    from resectionlab import surgical_position
    from resectionlab.surgical_verified_cache import _MODULES, implementation_fingerprint
    assert 'surgical_position' in _MODULES
    assert implementation_fingerprint() is not None
    monkeypatch.setattr(surgical_position, 'describe_position', lambda *args, **kwargs: None)
    assert implementation_fingerprint() is None
