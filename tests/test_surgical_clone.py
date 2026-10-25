"""Clones retain detached public state and exact JSON graph identity."""
import copy
import struct

import numpy as np

from resectionlab.surgical_engine import SurgicalEngine, _copy_engine_data, _owned_digest
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
