"""Narrow bookkeeping checks; these are not patient/generalization experiments."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location('real_learning_runner', ROOT / 'scripts/run_real_patient_learning.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0., 1.]))
    def forward(self, observation):
        return self.weight, self.weight[0]


class Task:
    """Two-action finite control double, without anatomy or geometric claims."""
    def __init__(self, invalid=False):
        self.terminated = False
        self.steps = 0
        self.invalid = invalid
    def clone(self):
        return Task(self.invalid)
    def metrics(self):
        return {'steps': self.steps, 'target_removed_mm3': 1., 'normal_removed_mm3': 0.}
    def observation(self):
        return SimpleNamespace(action_ids=('STOP', 'ACTION'), action_mask=np.array([True, True]), fingerprint='observation')
    def step(self, action):
        if self.invalid:
            raise ValueError('invalid control')
        self.steps += 1
        self.terminated = True
        return SimpleNamespace(reward=1. if action == 'ACTION' else 0., terminated=True, info={'action': action})


def run(tmp_path, **kwargs):
    return runner.run_episode(kwargs.pop('task', Task()), Policy(), torch.Generator().manual_seed(11),
        mode=kwargs.pop('mode', 'argmax'), name='episode', output=tmp_path, guard=lambda:None,
        audit=kwargs.pop('audit', lambda task, metrics: {'accepted':True,'target_access_success':True}), **kwargs)


def test_exact_task_metadata_and_roles_checked_before_case_load(monkeypatch):
    record = runner.declaration()
    assert runner.validate_declaration(record)
    altered = deepcopy(record)
    altered['member']['subject'] = 'sub-PAT26'
    with pytest.raises(ValueError, match='roles'):
        runner.validate_declaration(altered)
    altered = deepcopy(record)
    altered['settings']['optimizer_updates'] = 3
    with pytest.raises(ValueError, match='configuration'):
        runner.validate_declaration(altered)


def test_actual_forward_logs_current_ids_and_complete_transition(tmp_path):
    transitions, receipt = run(tmp_path)
    assert transitions[-1].terminated and transitions[0].action_id == 'ACTION'
    assert receipt['status'] == 'complete'
    assert receipt['committed_transitions'] == receipt['attempted_actions'] == 1
    decision = receipt['decisions'][0]
    assert decision['logits'] == [0.,1.] and decision['selected_index'] == 1
    assert decision['action_ids'] == ['STOP','ACTION']
    assert decision['status'] == 'returned'


def test_geometry_rejection_preserves_history_and_returns_no_batch(tmp_path):
    with pytest.raises(RuntimeError, match='evaluator rejected'):
        run(tmp_path, audit=lambda task, metrics: {'accepted':False,'target_access_success':None})
    receipt=json.loads((tmp_path/'episode.json').read_text())
    assert receipt['status']=='failed' and receipt['committed_transitions']==1
    assert receipt['independent_evaluation']['accepted'] is False


def test_invalid_action_attempt_is_not_a_committed_transition(tmp_path):
    with pytest.raises(ValueError, match='invalid control'):
        run(tmp_path, task=Task(invalid=True))
    receipt=json.loads((tmp_path/'episode.json').read_text())
    assert receipt['attempted_actions']==1 and receipt['committed_transitions']==0
    assert receipt['invalid_actions']==1 and receipt['status']=='failed'


def test_stale_or_partial_episode_never_enters_update():
    from resectionlab.real_patient_learning import PatientEpisode
    from resectionlab.spatial_policy import parameter_hash
    policy=Policy()
    partial=PatientEpisode('BTC:sub-PAT05',parameter_hash(policy),(SimpleNamespace(reward=1.,terminated=False),))
    with pytest.raises(ValueError, match='complete'):
        runner.checked_batch([partial],policy)
    selected=PatientEpisode('BTC:sub-PAT26',parameter_hash(policy),(SimpleNamespace(reward=1.,terminated=True),))
    with pytest.raises(ValueError, match='TRAIN'):
        runner.checked_batch([selected],policy)


def test_search_sequence_cannot_hide_early_termination(tmp_path):
    with pytest.raises(ValueError,match='after its terminal'):
        run(tmp_path,mode='sequence',sequence=('ACTION','STOP'))
    receipt=json.loads((tmp_path/'episode.json').read_text())
    assert receipt['status']=='failed' and receipt['committed_transitions']==1
