"""Clones retain detached public state and exact JSON graph identity."""
import copy
import struct

import numpy as np

from resectionlab.surgical_engine import SurgicalEngine, _copy_engine_data, _owned_digest, _microstep_checkpoint
from resectionlab.core import semantic_digest
import pytest
from resectionlab.surgical_scene import SurgicalScene


def test_copy_preserves_order_float_bits_aliases_cycles_and_unusual_values():
    shared = [1, -0., float('nan'), float('inf'), True, None, 'cell']
    mapping = {'negative': -0., 'value': 1.}
    graph = {'list': shared, 'aliases': [shared, mapping], 'mapping': mapping,
             'tuple': (shared,), 'array': np.array([1., 2.])}
    graph['cycle'] = graph
    actual = _copy_engine_data(graph)
    assert list(actual) == list(graph)
    assert actual['list'] is actual['aliases'][0] is actual['tuple'][0]
    assert actual['mapping'] is actual['aliases'][1]
    assert actual['cycle'] is actual and actual['list'] is not shared
    assert struct.pack('d', actual['list'][1]) == struct.pack('d', shared[1])
    assert np.isnan(actual['list'][2]) and np.isinf(actual['list'][3])
    assert not np.shares_memory(actual['array'], graph['array'])


def test_mechanical_checkpoint_preserves_cross_field_graph_and_subclasses():
    class PublicList(list):
        pass

    vector = [1., -0., 3.]
    corners = {'9': [vector] * 8}
    components = [{'cells': [9]}]
    state = {'removed': [2], 'flapComponents': components,
             'brainContinuumV1': {'cornerDisplacementMm': corners,
                 'memoryDisplacementMm': [corners], 'unusual': PublicList([vector]),
                 'array': np.array([2., 3.])},
             'deformation': {'9': vector}, 'deformationModel': {'publicAlias': corners},
             'externalAlias': vector}
    state['brainContinuumV1']['cycle'] = state
    actual = _microstep_checkpoint(state)
    assert list(actual) == list(state)
    assert actual['brainContinuumV1']['cycle'] is actual
    field = actual['brainContinuumV1']['cornerDisplacementMm']
    assert field is actual['brainContinuumV1']['memoryDisplacementMm'][0]
    assert field is actual['deformationModel']['publicAlias']
    assert all(v is actual['deformation']['9'] is actual['externalAlias'] for v in field['9'])
    assert type(actual['brainContinuumV1']['unusual']) is PublicList
    assert actual['brainContinuumV1']['unusual'][0] is actual['externalAlias']
    assert actual['flapComponents'] is components  # Existing read-only exemption.
    assert struct.pack('d', actual['externalAlias'][1]) == struct.pack('d', -0.)
    actual['externalAlias'][0] = 7.
    actual['removed'].append(9)
    actual['brainContinuumV1']['array'][0] = 6.
    assert vector == [1., -0., 3.] and state['removed'] == [2]
    np.testing.assert_array_equal(state['brainContinuumV1']['array'], [2., 3.])


def test_mechanical_checkpoint_does_not_call_subclass_metaclass_equality():
    class Meta(type):
        def __eq__(self, other):
            raise AssertionError('Scalar classification must use exact type identity')
        __hash__ = type.__hash__

    class PublicInt(int, metaclass=Meta):
        pass

    value = PublicInt(2); value.data = [3.]
    state = {'brainContinuumV1': {'nested': [value], 'mapping': {value: value}}}
    actual = _microstep_checkpoint(state)
    copied = actual['brainContinuumV1']['nested'][0]
    assert type(copied) is PublicInt and copied is not value
    assert actual['brainContinuumV1']['mapping'][copied] is copied
    copied.data.append(4.)
    assert value.data == [3.]


def test_owned_digest_matches_canonical_validation_and_detects_public_edits():
    values = {'state': {'removed': list(range(100)), 'thermal': {'1': -0.},
                       'slots': [{'position': [1., 2., 3.]}]}}
    assert _owned_digest(values) == semantic_digest(values)
    before = _owned_digest(values)
    values['state']['removed'].append(101)
    assert _owned_digest(values) == semantic_digest(values) != before
    assert _owned_digest({'numpy': np.int64(2)}) == semantic_digest({'numpy': np.int64(2)})
    for invalid in [{'key': float('nan')}, {2: 3}]:
        with pytest.raises((ValueError, TypeError)):
            _owned_digest(invalid)


def test_clone_and_fresh_transitions_equal_legacy_deepcopy_and_detach_history():
    labels = np.zeros((16, 16, 16), np.uint16)
    labels[6:9, 6:9, 6:9] = 1
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'target', 'material': 'tumor', 'target': True}], {}, {})
    actual = SurgicalEngine(scene)
    for action in [
        {'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
        {'slot': 'left', 'operation': 'place', 'duration': .1,
         'pose': {'position': [6., 6., 2.], 'quaternion': [0., 0., 0., 1.]}},
        {'slot': 'left', 'operation': 'apply', 'duration': .2,
         'pose': {'position': [6., 6., 8.], 'quaternion': [0., 0., 0., 1.]}},
    ]:
        assert actual.step(action)['status'] == 'accepted'
    reference = copy.copy(actual)
    reference.state = copy.deepcopy(actual.state)
    reference.actions = copy.deepcopy(actual.actions)
    reference.history = copy.deepcopy(actual.history)
    cloned = actual.clone()
    assert cloned.snapshot() == reference.snapshot() and cloned.history == reference.history
    cloned.history[0]['events'].append({'kind': 'caller-owned'})
    cloned.actions[0]['instrument'] = 'suction'
    cloned.state['removed'].clear()
    assert actual.snapshot() == reference.snapshot() and actual.history == reference.history
    for action in [{'slot': 'left', 'operation': 'inspect', 'duration': .2},
                   {'slot': 'left', 'operation': 'withdraw', 'duration': .3}]:
        assert actual.step(action) == reference.step(action)
        assert actual.snapshot() == reference.snapshot() and actual.history == reference.history


def test_public_receipts_cannot_edit_append_only_committed_ancestry():
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    receipt = engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    expected = copy.deepcopy(engine.history[0])
    engine.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    receipt['events'].append({'kind': 'caller annotation'})
    receipt['deltas']['removedCells'].append(1)
    assert engine.history[0] == expected


def test_view_effect_fields_stay_exact_and_detached():
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    # The large flap-component field is display data and remains a detached
    # public value even when its engine-owned JSON copy uses the faster path.
    engine.state['flapComponents'] = [{'id': 'component', 'cells': [0, 1, 2],
                                       'offset': [0., -0., 0.]}]
    expected = copy.deepcopy(engine.state['flapComponents'])
    actual = engine.view()
    assert actual['flapComponents'] == expected
    actual['flapComponents'][0]['cells'].append(7)
    actual['flapComponents'][0]['offset'][0] = 2.
    assert engine.state['flapComponents'] == expected
    complete = engine.view()
    compact = engine.view(_omit_fields=['flapComponents'])
    assert compact == {name: value for name, value in complete.items() if name != 'flapComponents'}
    assert engine.state['flapComponents'] == expected
    with pytest.raises(ValueError, match='Unsupported display field omission'):
        engine.view(_omit_fields=['removed'])


@pytest.mark.parametrize('raise_error', [False, True])
def test_rejected_preview_discards_candidate_mutations_before_rendering(monkeypatch, raise_error):
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .1}
    engine.step(action)
    before = engine.snapshot()
    expected_view = engine.view()
    expected_history = copy.deepcopy(engine.history)

    def mutate_then_reject(candidate, *args, **kwargs):
        candidate.state['elapsedSeconds'] += 7.
        candidate.state['slots']['left']['parameters']['power'] = 4.
        candidate.actions.append({'candidate': 'discard'})
        candidate.history.append({'candidate': 'discard'})
        if raise_error:
            raise ValueError('TEST_CANDIDATE_REJECTION')
        return {'status': 'rejected', 'reason': 'TEST_CANDIDATE_REJECTION',
                'completedFraction': 0., 'events': []}

    monkeypatch.setattr(SurgicalEngine, '_execute', mutate_then_reject)
    result = engine.preview(action)
    assert result['status'] == 'rejected'
    assert result['reasonCode'] == 'TEST_CANDIDATE_REJECTION'
    assert result['stateId'] == before['stateId']
    assert result['view'] == expected_view
    assert result['metrics'] == expected_view['metrics']
    assert engine.snapshot() == before and engine.history == expected_history
    result['view']['history'][0]['events'].append({'caller': 'annotation'})
    assert engine.history == expected_history


def test_preview_preserves_subclass_clone_and_step_hooks():
    class ObservedEngine(SurgicalEngine):
        def clone(self):
            candidate = super().clone()
            candidate.ledger = copy.deepcopy(self.ledger)
            return candidate

        def step(self, action, *, check=None):
            self.ledger.append({'kind': 'override-step'})
            receipt = super().step(action, check=check)
            receipt['subclassLedger'] = copy.deepcopy(self.ledger)
            return receipt

        def _execute(self, *args, **kwargs):
            self.ledger.append({'kind': 'candidate-execution'})
            return super()._execute(*args, **kwargs)

    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = ObservedEngine(scene)
    engine.ledger = [{'kind': 'original'}]
    before = engine.snapshot()
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .1}
    preview = engine.preview(action)
    assert preview['subclassLedger'] == [
        {'kind': 'original'}, {'kind': 'override-step'}, {'kind': 'candidate-execution'}]
    assert engine.ledger == [{'kind': 'original'}] and engine.snapshot() == before
    committed = engine.step(action)
    assert committed == {key: value for key, value in preview.items() if key not in ('preview', 'view')}
    assert engine.ledger == preview['subclassLedger']


def test_preview_receipt_and_view_detach_committed_ancestry_and_action():
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .1,
              'parameters': {'power': 2.}}
    engine.step(action)
    before = engine.snapshot()
    history = copy.deepcopy(engine.history)
    preview = engine.preview(action)
    expected = copy.deepcopy(preview)
    action['parameters']['power'] = 3.
    preview['events'].append({'caller': 'annotation'})
    preview['deltas']['removedCells'].append(0)
    preview['view']['slots']['left']['parameters']['power'] = 9.
    preview['view']['history'][0]['action']['parameters']['power'] = 9.
    preview['view']['history'][-1]['events'].append({'caller': 'annotation'})
    assert engine.snapshot() == before and engine.history == history
    action['parameters']['power'] = 2.
    committed = engine.step(action)
    assert committed == {key: value for key, value in expected.items() if key not in ('preview', 'view')}


@pytest.mark.parametrize('patch_target', ['class', 'instance'])
def test_preview_dispatches_patched_public_step(monkeypatch, patch_target):
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .1}
    calls = []
    original = SurgicalEngine.step
    if patch_target == 'class':
        def step(candidate, supplied_action, *, check=None):
            calls.append((candidate, supplied_action))
            return {**original(candidate, supplied_action, check=check), 'publicStepHook': True}
        monkeypatch.setattr(SurgicalEngine, 'step', step)
    else:
        def step(supplied_action, *, check=None):
            calls.append((None, supplied_action))
            return {'status': 'rejected', 'stateId': before['stateId'], 'publicStepHook': True}
        monkeypatch.setattr(engine, 'step', step)
    result = engine.preview(action, include_view=False)
    assert result['publicStepHook'] is True and result['preview'] is True
    assert len(calls) == 1 and calls[0][1] is action
    if patch_target == 'class':
        assert calls[0][0] is not engine
        assert result['status'] == 'accepted' and result['stateId'] != before['stateId']
    assert engine.snapshot() == before and engine.history == []


@pytest.mark.parametrize('method', ['step', 'preview'])
def test_metrics_failure_preserves_step_publication_and_preview_isolation(monkeypatch, method):
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    action = {'slot': 'left', 'operation': 'inspect', 'duration': .1}
    original = SurgicalEngine.metrics
    failure = ValueError('TEST_POST_COMMIT_METRICS')

    def metrics(candidate):
        if candidate.actions:
            raise failure
        return original(candidate)

    monkeypatch.setattr(SurgicalEngine, 'metrics', metrics)
    with pytest.raises(ValueError) as exc:
        getattr(engine, method)(action)
    assert exc.value is failure
    if method == 'step':
        assert engine.actions == [action] and len(engine.history) == 1
        assert engine.history[0]['stateBefore'] == before['stateId']
        assert engine.history[0]['stateAfter'] == engine.state_id != before['stateId']
        assert engine.state['elapsedSeconds'] == pytest.approx(.1)
    else:
        assert engine.snapshot() == before and engine.history == []


@pytest.mark.parametrize('method', ['step', 'preview'])
def test_finalization_domain_errors_propagate_without_committing(monkeypatch, method):
    import resectionlab.surgical_effects as effects
    scene = SurgicalScene(np.zeros((2, 2, 2), np.uint16), np.eye(4), [], {}, {})
    engine = SurgicalEngine(scene)
    before = engine.snapshot()
    failure = ValueError('TEST_FINALIZATION_FAILURE')
    original = effects.finalize_action

    def finalize(candidate):
        original(candidate)
        raise failure

    monkeypatch.setattr(effects, 'finalize_action', finalize)
    with pytest.raises(ValueError) as exc:
        getattr(engine, method)({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    assert exc.value is failure
    assert engine.snapshot() == before and engine.history == []
