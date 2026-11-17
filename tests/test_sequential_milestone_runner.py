"""Frozen protocol and failure-retention controls, without fitting a model."""
import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def runner():
    path=Path(__file__).resolve().parents[1]/'scripts/run_sequential_milestone.py'
    spec=importlib.util.spec_from_file_location('sequential_runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def record(runner,tmp_path):
    return runner.protocol(tmp_path,dict(sourceHashes=runner.source_hashes(),sourceChangedDuringPilot=False,
        tiers=[dict(name=name,max_physical_calls=72,max_wall_seconds=30.,horizon=2,beam_width=2,max_expansions=12)
            for name in ('low','high')]))


def test_protocol_refuses_stale_sources_changed_config_and_cross_split_parent(runner,tmp_path):
    protocol=record(runner,tmp_path);runner.validate_protocol(protocol)
    corrupt=json.loads(json.dumps(protocol));corrupt['sourceHashes']['scripts/run_sequential_milestone.py']='0'*64
    with pytest.raises(ValueError,match='CURRENT_SOURCE'):runner.validate_protocol(corrupt)
    corrupt=json.loads(json.dumps(protocol));corrupt['config']['max_attempts']+=1
    with pytest.raises(ValueError,match='CURRENT_SOURCE'):runner.validate_protocol(corrupt)
    corrupt=json.loads(json.dumps(protocol));corrupt['cohort'][-1]['parent']=0
    with pytest.raises(ValueError,match='GROUPED_COHORT'):runner.validate_protocol(corrupt)


def test_failed_replay_is_saved_and_all16_layouts_and_methods_are_retained(runner,tmp_path,monkeypatch):
    protocol=record(runner,tmp_path)
    def episode(task,method,tier,**kwargs):
        return dict(parent=task.audit['parent'],method=method,seed=0,tier=tier.name,
            skillHistory=[],endpoint=dict(classification='valid-stop-with-residual'),attempts=1,rejections=0)
    def replay(task,row,**kwargs):
        if row['parent']==200 and row['method']=='greedy':raise ValueError('injected-replay-disagreement')
        return dict(verified=True,physicalCalls=0,wallSeconds=0.)
    monkeypatch.setattr(runner,'run_episode',episode);monkeypatch.setattr(runner,'verify_replay',replay)
    with pytest.raises(ValueError,match='retained failures'):
        runner.evaluate(tmp_path,protocol,controls_only=True)
    evaluation=runner.read(tmp_path/'evaluation.json')
    assert len(evaluation['rows'])==64
    assert {r['parent'] for r in evaluation['rows']}==set(range(200,216))
    assert evaluation['failures'] and evaluation['rows'][0]['replay']['verified'] is False
    assert (tmp_path/'results/low-200-greedy-0.json').is_file()
