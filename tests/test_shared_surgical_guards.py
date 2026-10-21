"""The desktop engine and native archive own the same source-bound exclusions."""
import copy
import json
import os
import subprocess
import sys
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from resectionlab.core import CaseData, SourceRef
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionSet, ProtectionAnnotation


def fixture(displacement=0.):
    mask = np.zeros((50, 50, 50), bool); mask[10:30, 20, 20] = True
    case = CaseData(case_id='shared-guard-fixture', mri=np.zeros(mask.shape, np.float32),
        compartments={'target': mask}, affine=np.eye(4),
        source_refs=(SourceRef('guard-fixture', 'synthetic://guard-fixture', native_frame='RAS+', provenance='simulated'),),
        metadata={'is_synthetic': True, 'surgical_properties': {'attachments': []},
            'anatomy_layers': {'target': {'kind': 'tumor', 'provenance': 'engineering fixture'}}})
    scene = from_case(case)
    protected = np.zeros(scene.shape, bool); protected[29, 20, 20] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', protected,
        provenance='Explicit synthetic test membership; no patient claim')], clearance_mm=0.)
    return case, scene, protection, GuardConfig(critical_max_displacement_mm=displacement)


def pose(position, quaternion=None):
    return {'position': position, 'quaternion': quaternion or [0., 0., 0., 1.]}


def sequence():
    return [dict(slot='left', operation='exchange', instrument='tumor_forceps', duration=1.),
        dict(slot='left', operation='place', pose=pose([10., 20., -5.]), duration=1.,
            parameters=dict(radiusMm=.5, jawOpeningMm=0.)),
        dict(slot='left', operation='grasp', pose=pose([10., 20., 20.]), duration=1.),
        dict(slot='left', operation='move', pose=pose([10., 22., 20.]), duration=1.)]


def test_bare_engine_rl_preview_partial_and_replay_are_identical():
    pytest.importorskip('gymnasium')
    from resectionlab.surgical_rl_env import SurgicalRLEnv
    _, scene, protection, config = fixture()
    engine = SurgicalEngine(scene, protection=protection, guard_config=config)
    env = SurgicalRLEnv(scene, protection=protection); env.reset(seed=17)
    for action in sequence():
        prior = engine.state_id
        preview = engine.preview(action)
        assert engine.state_id == prior
        actual = engine.step(action); rl = env.step_action(action)[4]['outcome']
        assert actual['stateId'] == preview['stateId'] == rl['stateId']
        assert actual['status'] == preview['status'] == rl['status']
    assert actual['status'] == 'partial' and actual['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.state['holds']['left']['offset'] == [0, 0, 0]
    reopened = SurgicalEngine.restore(scene, json.loads(json.dumps(engine.snapshot(), allow_nan=False)))
    assert reopened.state == engine.state
    assert reopened.guard_report() == engine.guard_report()
    next_action = sequence()[-1]
    assert reopened.step(next_action) == engine.step(next_action)
    assert engine.branch(2).guard_report() == engine.guard_report()
    assert engine.replay(sequence()).state_id == env.engine.state_id
    engine.reset()
    assert engine.guard_report()['mode'] == 'strict-research' and not engine.actions


def test_missing_evidence_and_relaxed_research_are_explicit_and_bound():
    _, scene, _, _ = fixture()
    with pytest.raises(ValueError, match='MISSING_CRITICAL_ANNOTATIONS'):
        SurgicalEngine(scene, guard_config=GuardConfig())
    legacy = SurgicalEngine(scene)
    assert legacy.guard_report()['mode'] == 'unconfigured-research'
    relaxed = SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False))
    assert relaxed.guard_report()['mode'] == 'relaxed-research'
    assert relaxed.guard_report()['missingRequired'] == ['brainstem']
    assert relaxed.state_id != legacy.state_id
    record = relaxed.snapshot()
    modified = copy.deepcopy(record); modified['protectionContract']['clearanceMm'] = 0.
    with pytest.raises(ValueError, match='integrity'): SurgicalEngine.restore(scene, modified)
    modified = copy.deepcopy(record); modified.pop('protectionContract')
    with pytest.raises(ValueError, match='identity'): SurgicalEngine.restore(scene, modified)


def test_remote_fragment_extraction_is_rejected_without_removing_membership():
    _, scene, protection, _ = fixture()
    engine = SurgicalEngine(scene, protection=protection)
    for action in sequence()[:3]: assert engine.step(action)['status'] == 'accepted'
    before = engine.snapshot()
    result = engine.step(dict(slot='left', operation='extract', duration=.1))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.snapshot() == before


def test_swept_rotation_and_inactive_other_slot_are_guarded():
    _, scene, _, _ = fixture()
    mask = np.zeros(scene.shape, bool); mask[5, 5, 10] = True
    protection = ProtectionSet(scene, [ProtectionAnnotation.from_mask(scene, 'brainstem', mask,
        provenance='Synthetic empty-material exclusion cell')], clearance_mm=0.)
    engine = SurgicalEngine(scene, protection=protection)
    tilted = Rotation.from_euler('x', 90, degrees=True).as_quat().tolist()
    start = dict(slot='left', operation='place', pose=pose([5., 5., 20.], tilted), duration=.1)
    assert engine.step(start)['status'] == 'accepted'
    action = dict(slot='left', operation='move', pose=pose([5., 5., 20.]), duration=.1)
    result = engine.step(action)
    assert result['reasonCode'] == 'PROTECTED_STRUCTURE'
    # A stationary other slot is also checked during shared simulated time.
    engine.state['slots']['right'].update(position=[5., 5., 20.], quaternion=[0., 0., 0., 1.])
    before = copy.deepcopy(engine.state)
    result = engine.step(dict(slot='left', operation='inspect', duration=.1))
    assert result['status'] == 'rejected' and result['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.state == before


@pytest.mark.parametrize('injury', ['thermal', 'boneTemperature', 'boneDose', 'deformation', 'cut', 'removal'])
def test_indirect_effects_roll_back_atomically(monkeypatch, injury):
    _, scene, protection, _ = fixture()
    cell = int(protection.cells[0]); original = SurgicalEngine._physical_substep
    def inject(engine, *args):
        events = original(engine, *args)
        if injury == 'thermal': engine.state['thermal'][str(cell)] = 1.
        elif injury == 'boneTemperature': engine.state.setdefault('drillingV2', {})['temperatureC'] = {str(cell): 50.}
        elif injury == 'boneDose': engine.state.setdefault('drillingV2', {})['thermalDoseCSeconds'] = {str(cell): 1.}
        elif injury == 'deformation': engine.state.setdefault('deformation', {})[str(cell)] = [0., 1., 0.]
        elif injury == 'cut': engine.state.setdefault('cutEdges', []).append([cell - 1, cell])
        else: engine.state['removed'].append(cell)
        return events
    monkeypatch.setattr(SurgicalEngine, '_physical_substep', inject)
    engine = SurgicalEngine(scene, protection=protection); before = engine.snapshot()
    outcome = engine.step(dict(slot='left', operation='inspect', duration=.1))
    assert outcome['status'] == 'rejected' and outcome['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert engine.snapshot() == before


def test_native_save_open_fresh_process_continue_and_reset(tmp_path):
    from resectionlab.imaging import save_case
    case, scene, protection, config = fixture()
    engine = SurgicalEngine(scene, protection=protection, guard_config=config)
    for action in sequence(): engine.step(action)
    destination = tmp_path / 'guarded.ressectionlab'
    save_case(case, destination, artifacts={'surgical_toolbox': {'mapping': scene.regions,
        'branches': {'main': engine.snapshot()}, 'branchId': 'main', 'position': len(engine.actions), 'parents': {'main': None}}})
    script = '''import json, sys
from resectionlab.desktop_bridge import BridgeSession, _Request
s = BridgeSession(sys.argv[2])
def call(op, args): return s.execute(op, args, _Request(op), lambda *_: None)
try:
 c = call('loadCase', {'path': sys.argv[1]})
 def command(name, previous=None, **extra):
  args = dict(caseHash=c['caseHash'], command=name, **extra)
  if previous: args['expectedStateId'] = previous['view']['stateId']
  return call('surgicalRehearsal', args)
 loaded = command('inspect')
 continued = command('step', loaded, action=json.loads(sys.argv[3]))
 reset = command('reset', continued)
 call('saveCase', dict(caseHash=c['caseHash'], path=sys.argv[4], workspace={}, overwrite=False))
 print(json.dumps(dict(loaded=loaded['view']['stateId'], continued=continued['outcome'],
  loadedGuard=loaded['view']['protection'], resetGuard=reset['view']['protection'], resetRemoved=reset['view']['removed'])))
finally: s.close()
'''
    saved_reset = tmp_path / 'reset.ressectionlab'
    result = subprocess.run([sys.executable, '-c', script, str(destination), str(tmp_path/'fresh'),
        json.dumps(sequence()[-1]), str(saved_reset)], env=dict(os.environ, PYTHONPATH=os.path.abspath('src')),
        capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data['loaded'] == engine.state_id
    assert data['continued']['reasonCode'] == 'PROTECTED_STRUCTURE'
    assert data['loadedGuard'] == data['resetGuard'] == engine.guard_report()
    assert data['resetRemoved'] == []
    pytest.importorskip('gymnasium')
    from resectionlab.surgical_rl_env import SurgicalRLEnv
    env = SurgicalRLEnv.from_patient(saved_reset); env.reset(seed=17)
    assert env.engine.guard_report() == engine.guard_report()
