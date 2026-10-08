"""Independent metadata/control tests; no native tasks, rollouts or gradients."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_native_opening_bc_capacity as runner

FROZEN = {
    'scripts/run_native_opening_bc_capacity.py': '635d6267f7e28583617ab0c5a04a6ce49cbc8ea4290d82f993ac9fc3e8590156',
    'tests/test_native_opening_bc_capacity.py': 'bb683cb6a300ef7872ccc059e7c0be32047729967e845b56fdc0c4b14d2f9230',
    'manifests/experiments/native-opening-bc-capacity-v1.json': '3812bb430b0537c83604c4e18d9236d51a7dc203cafc753e0dc6f52d688b9c84',
    'docs/native-opening-bc-capacity.md': '7ede45ba61e65453887607017fadbd574d6f0d3e11fbb2074207fd62c4efe136',
}


def frozen_record():
    return json.loads((ROOT / 'manifests/experiments/native-opening-bc-capacity-v1.json').read_text())


def test_frozen_sources_inputs_and_teacher_are_exact():
    for name, expected in FROZEN.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    record = frozen_record()
    runner.validate(record)
    assert len(record['source_sha256']) == 32
    assert len(record['input_sha256']) == 15
    bundle = runner.load_frozen_teacher(record)
    proofs = bundle['teacher-demonstration-checks']
    assert len(proofs) == 5 and len(bundle['rows']) == 5
    assert len({proof['observation_hash'] for proof in proofs}) == 5
    assert sum(len(p['prefix']) for p in proofs) == 4
    assert sum(p['selected_action_id'] == 'STOP' for p in proofs) == 2
    assert record['lineage']['real_patient_count'] == 0
    assert record['lineage']['human_roles_opened'] == []


@pytest.mark.parametrize('change', ['source_digest', 'input_digest', 'missing_teacher', 'changed_ancestry'])
def test_bound_declaration_mutations_fail_before_numerical_work(monkeypatch, change):
    import resectionlab.native_spatial_task as tasks
    monkeypatch.setattr(tasks, 'make_native_opening_task', lambda **kw: pytest.fail('Native task forbidden'))
    record = frozen_record()
    if change == 'source_digest':
        record['source_sha256']['src/resectionlab/data_policy.py'] = '0' * 64
    elif change == 'input_digest':
        record['input_sha256']['teacher.json'] = '0' * 64
    elif change == 'missing_teacher':
        del record['input_sha256']['teacher-state-04.json']
    else:
        record['old_declaration_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        runner.load_frozen_teacher(record)


def test_cli_validation_does_not_enter_worker(monkeypatch):
    monkeypatch.setattr(runner, 'worker', lambda *a, **kw: pytest.fail('Worker forbidden'))
    monkeypatch.setattr(runner.io, 'supervise_worker', lambda *a, **kw: pytest.fail('Supervisor forbidden'))
    monkeypatch.setattr(sys, 'argv', [runner.SCRIPT, '--declaration', str(ROOT / 'manifests/experiments/native-opening-bc-capacity-v1.json')])
    runner.main()


def test_existing_attempt_cannot_be_overwritten(monkeypatch, tmp_path):
    marker = tmp_path / 'preserved.json'
    marker.write_text('{"preserved": true}\n')
    before = marker.read_bytes()
    monkeypatch.setattr(runner.io, 'supervise_worker', lambda *a, **kw: pytest.fail('Supervisor forbidden'))
    monkeypatch.setattr(sys, 'argv', [runner.SCRIPT, '--declaration', str(ROOT / 'manifests/experiments/native-opening-bc-capacity-v1.json'), '--execute', '--output', str(tmp_path)])
    with pytest.raises(ValueError, match='new output directory'):
        runner.main()
    assert marker.read_bytes() == before


def test_update16_gate_requires_both_exact_weights_and_update_count(monkeypatch):
    import resectionlab.spatial_policy as policy
    monkeypatch.setattr(policy, 'parameter_hash', lambda model: runner.INITIAL_HASH)
    with pytest.raises(ValueError, match='Update16'):
        runner.verify_update16(object(), [{}] * 16)
    monkeypatch.setattr(policy, 'parameter_hash', lambda model: runner.BC16_HASH)
    with pytest.raises(ValueError, match='Update16'):
        runner.verify_update16(object(), [{}] * 15)
    runner.verify_update16(object(), [{}] * 16)


def stub_worker_dependencies(monkeypatch, *, accepted):
    """Exercise worker orchestration with no native/model/optimizer operations."""
    import torch
    import resectionlab.native_spatial_task as tasks
    import resectionlab.spatial_policy as policy
    import resectionlab.spatial_policy_diagnostics as diagnostics
    bundle = runner.load_frozen_teacher(frozen_record())
    counts = {'loss': 0, 'gradient': 0, 'optimizer': 0, 'evaluation': 0}
    base = SimpleNamespace(case=SimpleNamespace(source_hash='sha256:' + '1' * 64), decision_model_hash='sha256:' + '2' * 64)
    class Profiler:
        def __init__(self, *a): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def snapshot(self): return {}
    class Model:
        architecture_hash = 'fake-for-orchestration-only'
        def parameters(self): return ()
        def state_dict(self): return {}
        def architecture_record(self): return {}
    class Adam:
        def __init__(self, *a, **kw): counts['optimizer'] += 1
        def state_dict(self): return {}
    def loss(*a, **kw):
        counts['loss'] += 1
        return None, {'loss_forward_calls': 5}
    def gradient(*a, **kw):
        counts['gradient'] += 1
        return {'orchestration_stub': True}
    def evaluate(*a, **kw):
        counts['evaluation'] += 1
        return None, {'independent_evaluation': {'accepted': accepted, 'outcomes': {}},
            'decisions': [], 'guarded_online_seconds': 0, 'online_seconds': 0,
            'independent_evaluation_seconds': 0}, None
    monkeypatch.setattr(runner, 'validate', lambda _: None)
    monkeypatch.setattr(runner, 'load_frozen_teacher', lambda _: bundle)
    monkeypatch.setattr(tasks, 'make_native_opening_task', lambda **kw: base)
    monkeypatch.setattr(diagnostics, 'NativePreviewProfiler', Profiler)
    monkeypatch.setattr(runner, 'reconstruct_teacher', lambda *a: ((), {'orchestration_stub': True}))
    monkeypatch.setattr(policy, 'SpatialPolicy', lambda *a: Model())
    monkeypatch.setattr(policy, 'parameter_hash', lambda _: runner.INITIAL_HASH)
    monkeypatch.setattr(policy, 'imitation_loss', loss)
    monkeypatch.setattr(policy, 'gradient_step', gradient)
    monkeypatch.setattr(torch.optim, 'Adam', Adam)
    monkeypatch.setattr(torch, 'save', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'readout', lambda *a: {'forward_calls': 5, 'orchestration_stub': True})
    monkeypatch.setattr(runner.prior, 'guarded_episode', evaluate)
    from resectionlab.data_policy import GeneratedDevelopmentContext
    monkeypatch.setattr(GeneratedDevelopmentContext, 'require_task', lambda *a: None)
    return counts


def test_rejected_initial_audit_prevents_any_optimizer(monkeypatch, tmp_path):
    counts = stub_worker_dependencies(monkeypatch, accepted=False)
    with pytest.raises(ValueError, match='Independent execution rejected'):
        runner.worker(frozen_record(), tmp_path, declaration_sha256='3' * 64)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert counts == {'loss': 0, 'gradient': 0, 'optimizer': 0, 'evaluation': 1}
    assert result['status'] == 'failed' and result['updates'] == 0
    assert result['methods']['initial']['status'] == 'failed'
    assert result['methods']['initial']['outcomes'] is None
    assert result['methods']['BC256'] == {'status': 'not_started', 'outcomes': None}


def test_update16_mismatch_stops_orchestration_at16(monkeypatch, tmp_path):
    counts = stub_worker_dependencies(monkeypatch, accepted=True)
    with pytest.raises(ValueError, match='Update16 did not reproduce'):
        runner.worker(frozen_record(), tmp_path, declaration_sha256='3' * 64)
    result = json.loads((tmp_path / 'result.json').read_text())
    assert counts == {'loss': 16, 'gradient': 16, 'optimizer': 1, 'evaluation': 1}
    assert result['status'] == 'failed' and result['updates'] == 16
    assert result['loss_forward_calls'] == 80
    assert 'update16_exact_reproduction' not in result
    assert result['methods']['BC256'] == {'status': 'not_started', 'outcomes': None}
    assert len(json.loads((tmp_path / 'updates.json').read_text())) == 16
