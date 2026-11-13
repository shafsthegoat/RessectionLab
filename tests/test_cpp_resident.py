"""Differential complete actions through native owned state, without helpers."""
import copy
import json

import numpy as np
import pytest

from resectionlab import native_backend as backend
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_native_engine import create
from resectionlab.surgical_scene import SurgicalScene


@pytest.fixture
def compiled():
    if backend._module is None:
        pytest.skip('resident C++ simulator required')
    return backend._module


def empty():
    return SurgicalEngine(SurgicalScene(np.zeros((8,8,8),np.uint16),np.eye(4),[],{},{}))


def compare_actions(compiled, reference, actions, monkeypatch):
    native, reason = create(reference)
    assert reason is None
    token = compiled.SimulationStopToken()
    assert compiled.simulation_identity(native) == reference.state_id
    for index, action in enumerate(actions):
        with monkeypatch.context() as patch:
            patch.setattr(backend, '_module', None)
            expected = reference.step(copy.deepcopy(action))
        actual = compiled.simulation_step(native, action, token)
        assert actual == expected, (index, action, actual, expected)
        assert compiled.simulation_data(native, 'state') == reference.state, (index, action)
        assert compiled.simulation_data(native, 'history') == reference.history, (index, action)
        assert compiled.simulation_data(native, 'snapshot') == reference.snapshot(), (index, action)
    return native


def test_complete_empty_actions_exact(compiled,monkeypatch):
    from resectionlab.generated_cpp_fixture import request, pose
    compare_actions(compiled,empty(),[
        request('inspect',duration=.3),request('configure',mechanicsRevision=2,clockRevision=1),
        request('exchange',instrument='cusa'),request('place',pose=pose(2.,2.,-5.)),
        request('apply',pose=pose(2.,2.,3.),duration=.3),request('deactivate'),
        request('withdraw'),request('exchange',instrument='dissector'),
        request('move',pose={'position':[0.,0.,0.],'quaternion':[0.,0.,0.,0.]}),
        {'slot':'bad','operation':'inspect'},request('inspect',parameters={'power':True}),
    ],monkeypatch)


def test_complete_cusa_actions_exact(compiled,monkeypatch):
    from resectionlab.generated_cpp_supplement import cusa_fixture
    engine, actions=cusa_fixture()
    compare_actions(compiled,engine,actions,monkeypatch)


def test_complete_native_cranial_opening_and_preparation(compiled,monkeypatch):
    from resectionlab.generated_cpp_fixture import (joint_source,bone_actions,
        membrane_cut_actions,membrane_grasp_actions,request,pose)
    engine,contract,contour,_,_=joint_source()
    actions=bone_actions(contour)+membrane_cut_actions(engine,contract)+membrane_grasp_actions()+[
        request('move',pose=pose(z=54.5)),request('extract'),request('withdraw',pose=pose(z=54.5))]
    compare_actions(compiled,engine,actions,monkeypatch)


def test_live_replay_retains_native_cranial_prefix(compiled):
    from resectionlab.generated_cpp_fixture import (joint_source, bone_actions,
        membrane_cut_actions, membrane_grasp_actions, request, pose)
    from resectionlab.surgical_replay_cache import SurgicalReplayCache
    engine, contract, contour, _, _ = joint_source()
    actions = bone_actions(contour) + membrane_cut_actions(engine, contract) + membrane_grasp_actions() + [
        request('move', pose=pose(z=54.5)), request('extract'), request('withdraw', pose=pose(z=54.5))]
    for action in actions:
        assert engine.step(action)['status'] != 'rejected'
    assert engine.__dict__.get('_resident') is not None
    assert compiled.simulation_query(engine._resident, 'bonePrefix') is not None
    snapshot = engine.snapshot()
    replay = SurgicalReplayCache().seek(engine, snapshot, len(engine.actions))
    assert replay._cranial_stage_bone_prefix is not None
    assert replay.observe()['cranialStages']['bonePrefixVerified']
    assert replay.snapshot() == snapshot


def test_native_clone_preview_and_cancel_are_atomic(compiled):
    native,reason=create(empty())
    assert reason is None
    before=compiled.simulation_data(native,'snapshot')
    action={'slot':'left','operation':'inspect','duration':.2}
    candidate,result=compiled.simulation_prepare(native,action,compiled.SimulationStopToken())
    assert result['status']=='accepted'
    assert compiled.simulation_data(native,'snapshot')==before
    assert compiled.simulation_data(candidate,'snapshot')!=before
    clone=compiled.simulation_clone(candidate)
    token=compiled.SimulationStopToken();token.cancel()
    with pytest.raises(RuntimeError,match='SIMULATION_CANCELLED'):
        compiled.simulation_step(clone,action,token)
    assert compiled.simulation_data(clone,'snapshot')==compiled.simulation_data(candidate,'snapshot')


def test_native_json_preserves_bigint_and_signed_zero(compiled):
    value={'fraction':[2**104,2**1074],'z':-0.,'unicode':'\ud800\U0001f989'}
    assert compiled.simulation_json(value)==json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def test_public_engine_owns_native_state_and_clones(compiled,monkeypatch):
    action={'slot':'left','operation':'inspect','duration':.2}
    expected=empty()
    with monkeypatch.context() as patch:
        patch.setattr(backend,'_module',None)
        receipt=expected.step(action)
    actual=empty()
    assert actual.step(action)==receipt
    assert actual.__dict__.get('_resident') is not None
    assert actual.snapshot()==expected.snapshot()
    cloned=actual.clone()
    assert cloned.__dict__['_resident'] is not actual.__dict__['_resident']
    cloned.state['travelMm']=-0.
    assert cloned.state_id!=actual.state_id
    assert actual.state['travelMm']==0.
    assert compiled.simulation_stats()['physicalSubsteps']>0


def test_initial_method_replacement_declines_resident_execution(compiled, monkeypatch):
    from resectionlab import surgical_native_engine as adapter
    monkeypatch.setattr(adapter, '_stock_capture', None)
    original = SurgicalEngine._physical_substep_reference

    def changed(self, *args):
        return original(self, *args)

    monkeypatch.setattr(SurgicalEngine, '_physical_substep_reference', changed)
    engine = empty()
    assert not adapter._stock(engine)
    engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    assert engine.__dict__.get('_resident') is None


@pytest.mark.parametrize('action', [
    {'slot':'left','operation':'inspect','duration':'0.2'},
    {'slot':'left','operation':'inspect','parameters':{'power':np.float64(1.)}},
    {'slot':'left','operation':'place','pose':{'position':np.array([2.,2.,-5.]),'quaternion':[0.,0.,0.,1.]}},
])
def test_public_custom_inputs_use_reference_before_mutation(compiled, monkeypatch, action):
    expected=empty()
    actual=empty()
    warmup={'slot':'left','operation':'inspect','duration':.1}
    actual.step(warmup)
    with monkeypatch.context() as patch:
        patch.setattr(backend,'_module',None)
        expected.step(warmup)
        receipt=expected.step(copy.deepcopy(action))
    assert actual.step(copy.deepcopy(action))==receipt
    assert actual.state==expected.state
    assert actual.__dict__.get('_resident') is None


def test_resident_reads_invalidate_changed_source_and_metrics(compiled, monkeypatch):
    from resectionlab import surgical_physics
    engine=empty()
    engine.step({'slot':'left','operation':'inspect','duration':.1})
    original=surgical_physics.physics_metrics
    monkeypatch.setattr(surgical_physics,'physics_metrics',lambda owner:{**original(owner),'custom':17})
    assert engine.metrics()['custom']==17
    assert engine.__dict__.get('_resident') is None
    monkeypatch.setattr(surgical_physics,'physics_metrics',original)
    engine.step({'slot':'left','operation':'inspect','duration':.1})
    assert engine.__dict__.get('_resident') is not None
    engine.scene=SurgicalScene(np.zeros((8,8,8),np.uint16),np.eye(4),[],{'version':2},{})
    snapshot=engine.snapshot()
    assert snapshot['sceneId']==engine.scene.identity
    assert engine.__dict__.get('_resident') is None


def test_resident_import_custom_leaf_falls_back(compiled, monkeypatch):
    engine=empty()
    engine.step({'slot':'left','operation':'inspect','duration':.1})
    engine.state['thermal']['0']=np.float64(.25)
    expected=copy.deepcopy(engine.__dict__['state'])
    engine.metrics()
    assert engine.__dict__.get('_resident') is None
    assert engine.state==expected


@pytest.mark.parametrize('alias', [False, True])
def test_external_state_container_ownership_uses_reference(compiled, alias):
    engine=empty()
    engine.step({'slot':'left','operation':'inspect','duration':.1})
    if alias:
        engine.state['contact']=engine.state['thermal']
    else:
        engine.state['removed']=()
    engine.step({'slot':'left','operation':'inspect','duration':.1})
    assert engine.__dict__.get('_resident') is None
    if alias:
        assert engine.state['contact'] is engine.state['thermal']
    else:
        assert type(engine.state['removed']) is tuple


def _ordering_fixture(ids, instrument='dissector'):
    labels = np.zeros((1, 1, 128), np.uint16)
    labels.ravel()[ids] = 1
    regions = [dict(id=1, name='Generated tissue', material='soft_tissue')]
    if instrument == 'cranial_drill':
        regions[0]['material'] = 'bone'
    else:
        labels.ravel()[[i for i in ids if i % 2 == 0]] = 2
        regions.append(dict(id=2, name='Generated target', material='tumor', target=True))
    engine = SurgicalEngine(SurgicalScene(labels, np.eye(4), regions, {}, {'attachments': []}))
    slot = engine.state['slots']['left']
    slot.update(instrument=instrument, position=[0., 0., 7.5],
                quaternion=[0., 2.**-.5, 0., 2.**-.5])
    engine.state['slots']['right']['position'] = [300., 0., -5.]
    return engine


def _ordered_step(compiled, native, reference, action, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(backend, '_module', None)
        expected = reference.step(copy.deepcopy(action))
    actual = compiled.simulation_step(native, action, compiled.SimulationStopToken())
    assert actual == expected
    state = compiled.simulation_data(native, 'state')
    assert state == reference.state
    for field in ('contact', 'debris', 'materialExposure'):
        assert list(state[field]) == list(reference.state[field]), field
    return actual, state


def test_native_two_cell_visible_order(compiled, monkeypatch):
    for instrument in ('dissector', 'cranial_drill'):
        reference = _ordering_fixture([7, 8], instrument)
        native, reason = create(reference)
        assert reason is None
        operation = 'inspect' if instrument == 'dissector' else 'apply'
        actual, state = _ordered_step(compiled, native, reference,
            {'slot': 'left', 'operation': operation, 'duration': .1}, monkeypatch)
        assert actual['status'] == 'accepted'
        assert list(state['contact' if instrument == 'dissector' else 'debris']) == ['8', '7']
        if instrument == 'dissector':
            assert list(state['materialExposure']) == ['tumor', 'soft_tissue']


def test_native_visible_order_after_delete_and_reinsert(compiled, monkeypatch):
    reference = _ordering_fixture([7, 8])
    native, reason = create(reference)
    assert reason is None
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .1}
    _ordered_step(compiled, native, reference, action, monkeypatch)
    reference.state['contact'].pop('8')
    reference.state['removed'] = [8]
    assert compiled.simulation_import(native, {'state': reference.state}) is None
    _, state = _ordered_step(compiled, native, reference, action, monkeypatch)
    assert list(state['contact']) == ['7']
    reference.state['removed'] = []
    assert compiled.simulation_import(native, {'state': reference.state}) is None
    _, state = _ordered_step(compiled, native, reference, action, monkeypatch)
    assert list(state['contact']) == ['7', '8']


def test_native_many_cell_source_order_trace(compiled, monkeypatch):
    reference = _ordering_fixture(list(range(7, 72)))
    native, reason = create(reference)
    assert reason is None
    q = [0., 2.**-.5, 0., 2.**-.5]
    actions = [
        {'slot': 'left', 'operation': 'apply', 'duration': .3,
         'pose': {'position': [0., 0., 24.5], 'quaternion': q}},
        {'slot': 'left', 'operation': 'apply', 'duration': .2,
         'pose': {'position': [0., 0., 56.5], 'quaternion': q}},
        {'slot': 'left', 'operation': 'inspect', 'duration': .2},
    ]
    for action in actions:
        actual, state = _ordered_step(compiled, native, reference, action, monkeypatch)
        assert actual['status'] == 'accepted'
    assert len(state['contact']) >= 40
