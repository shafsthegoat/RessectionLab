"""Optional frame hooks observe actual committed replay, never failed steps."""
import copy
import numpy as np
import pytest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_telemetry import capture_episode


def fixture():
    labels = np.zeros((24, 24, 24), np.uint16)
    labels[8, 8, 8] = 1
    source = SurgicalScene(labels, np.eye(4), [{'id': 1, 'name': 'bone', 'material': 'bone'}],
                           {'kind': 'explicit-telemetry-hook-engineering-fixture'}, {})
    pose = lambda z: {'position': [8., 8., z], 'quaternion': [0., 0., 0., 1.]}
    actions = [
        {'operation': 'place', 'pose': pose(4.), 'parameters': {'radiusMm': .6}},
        {'operation': 'move', 'pose': pose(9.), 'duration': 1.},
        {'operation': 'move', 'pose': pose(9.), 'duration': .01},
        {'operation': 'inspect', 'duration': 0.},
        {'operation': 'apply', 'pose': pose(9.), 'duration': .4},
    ]
    actions = [{'slot': 'left', 'duration': .1, **row} for row in actions]
    engine = SurgicalEngine(source)
    receipts, endpoints = [], []
    for action in actions:
        receipts.append(engine.step(action))
        endpoints.append(engine.snapshot())
    assert [row['status'] for row in receipts] == ['accepted', 'partial', 'rejected', 'rejected', 'accepted']
    return source, actions, engine, receipts, endpoints


class Collector:
    """Stub verifies real public receipts before publishing outside identity."""
    def __init__(self, receipts, endpoints):
        self.receipts, self.endpoints = receipts, endpoints
        self.index = 0
        self.staging = None
        self.published = []
        self.rejected = []
        self.aborted = False
        self.begin_ids, self.observation_ids = set(), set()

    def begin(self, engine, action):
        assert self.staging is None
        self.begin_ids.add(id(engine))
        self.staging = {'action': copy.deepcopy(action), 'before': engine.state_id, 'observations': []}

    def observe(self, engine, slot, effective_active):
        assert self.staging is not None and type(effective_active) is bool
        self.observation_ids.add(id(engine))
        self.staging['observations'].append({'time': engine.state['elapsedSeconds'],
            'stateId': engine.state_id, 'slot': slot, 'effective': effective_active,
            'position': engine.state['slots'][slot]['position'][:]})

    def end(self, engine, action, outcome):
        assert self.staging['action'] == action
        assert outcome == self.receipts[self.index]
        assert engine.snapshot() == self.endpoints[self.index]
        assert engine.state_id == outcome['stateId']
        if outcome['status'] == 'rejected':
            assert not self.staging['observations']
            assert self.staging['before'] == engine.state_id
            self.rejected.append(self.staging)
        else:
            assert self.staging['observations']
            assert engine.history[-1]['stateAfter'] == engine.state_id
            self.published.append(self.staging)
        self.index += 1
        self.staging = None

    def abort(self):
        self.aborted = True
        self.staging = None


def test_collector_observes_successful_steps_verifies_final_endpoints_and_discards_rejections():
    source, actions, base, receipts, endpoints = fixture()
    original = base.snapshot()
    default = capture_episode(base, actions)
    collector = Collector(receipts, endpoints)
    collected = capture_episode(base, actions, frame_collector=collector)
    assert default == collected
    assert collected['finalStateId'] == base.state_id and base.snapshot() == original
    assert collector.index == len(actions)
    assert len(collector.published) == 3 and len(collector.rejected) == 2
    assert not collector.aborted and collector.staging is None
    assert collector.observation_ids - collector.begin_ids, 'Candidate clones must share the same outside collector'
    observed = [row for staged in collector.published for row in staged['observations']]
    microsteps = [row for row in collected['observations'] if row['sampleKind'] == 'microstep']
    assert len(observed) == len(microsteps)
    assert [row['time'] for row in observed] == [row['timeSeconds'] for row in microsteps]
    assert [row['position'] for row in observed] == [row['slots']['left']['positionMm'] for row in microsteps]
    assert all(row['effective'] for row in collector.published[-1]['observations'])


def test_collector_exception_aborts_staging_and_preserves_supplied_engine():
    _, actions, base, receipts, endpoints = fixture()
    original = base.snapshot()
    class InterruptedCollector(Collector):
        def observe(self, engine, slot, effective_active):
            super().observe(engine, slot, effective_active)
            raise InterruptedError('Explicit engineering observer cancellation')
    collector = InterruptedCollector(receipts, endpoints)
    with pytest.raises(InterruptedError, match='observer cancellation'):
        capture_episode(base, actions, frame_collector=collector)
    assert collector.aborted and collector.staging is None
    assert not collector.published and base.snapshot() == original


def test_failed_receipt_verification_aborts_instead_of_publishing_frames():
    _, actions, base, receipts, endpoints = fixture()
    class RefusingCollector(Collector):
        def end(self, engine, action, outcome):
            raise ValueError('Canonical action receipt differs')
    collector = RefusingCollector(receipts, endpoints)
    with pytest.raises(ValueError, match='receipt differs'):
        capture_episode(base, actions, frame_collector=collector)
    assert collector.aborted and collector.staging is None and not collector.published
