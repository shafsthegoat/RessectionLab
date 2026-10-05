"""Independent continuation boundaries; mocked processes and analytical JSON only."""
import gzip
import hashlib
import json

import pytest

from scripts import mechanics_hbe_experiment as x


def test_continuation_parent_gets_debited_cap_and_its_own_single_attempt(tmp_path, monkeypatch):
    old_marker = tmp_path/'.specimen-experiment-started.json'
    old_marker.write_bytes(b'original attempt remains immutable\n')
    old = old_marker.read_bytes()
    directory = tmp_path/'continuation-header-v1'
    debit = 4.2655537920072675
    plan = {'raw_root': str(tmp_path), 'experiment': str(directory), 'inputs': {},
            'continuation': {'prior_solver_invocations': 1, 'prior_parent_seconds_debit': debit},
            'protocol': {'budgets': {'aggregate_specimen_seconds': 900,
                                     'sampled_process_family_rss_bytes': 3*1024**3}}}
    calls = []
    monkeypatch.setattr(x, 'preflight', lambda *args: plan)
    monkeypatch.setattr(x.runtime, 'declaration', lambda: {})
    monkeypatch.setattr(x.runtime, 'private_environment', lambda spec: {})

    def supervisor(command, output, **kwargs):
        calls.append(kwargs)
        assert kwargs['seconds'] == 900-debit
        assert kwargs['rss_bytes'] == 3*1024**3
        assert kwargs['cwd'] == tmp_path
        assert output == directory/'supervision'
        x.runtime.write_json(directory/'state.json', {'status': 'failed_or_incomplete'})
        return {'status': 'failed_or_incomplete'}

    monkeypatch.setattr(x.runtime, 'supervise', supervisor)
    result = x.launch(tmp_path, {'test': 'protocol'}, {'test': 'release'}, reuse_verified_coarse=True)
    assert result['status'] == 'failed_or_incomplete'
    assert old_marker.read_bytes() == old
    marker = tmp_path/'.specimen-header-continuation-v1-started.json'
    assert json.loads(marker.read_text())['no_retry'] is True
    with pytest.raises(FileExistsError):
        x.launch(tmp_path, {'test': 'protocol'}, {'test': 'release'}, reuse_verified_coarse=True)
    assert len(calls) == 1
    assert old_marker.read_bytes() == old


def test_successful_coarse_reuse_preserves_execution_identity_and_call_count(tmp_path, monkeypatch):
    accepted = {'passed': True, 'analytical_control_only': True,
                'values': {'float': .125, 'integer': 61}}
    raw = x.access.canonical_json(accepted)
    compressed = tmp_path/'accepted.json.gz'
    compressed.write_bytes(gzip.compress(raw, mtime=0))
    primitives = {'synthetic': 'metadata only; no source or measured data'}
    execution = tmp_path/'original-execution.json'
    execution.write_bytes(x.access.canonical_json({'primitive_bindings': primitives}))
    directory = tmp_path/'continuation-header-v1';directory.mkdir()
    execution_binding = x.binding(tmp_path, execution)
    plan = {'experiment': str(directory), 'protocol_binding': {'sha256': 'a'*64},
            'continuation': {'execution': execution_binding,
                             'accepted_readout': {'gzip': x.binding(tmp_path, compressed),
                                                  'uncompressed_sha256': hashlib.sha256(raw).hexdigest()}}}
    state = {'solver_invocations': 1, 'runs': {x.REUSED_RUN: {'status': 'not_executed'}}}
    calls = []

    def analytical_readout(root, received, **kwargs):
        calls.append(kwargs)
        assert received == primitives
        assert kwargs == {'protocol_sha256': 'a'*64, 'expected_branch': 'compression',
                          'expected_mesh_N': 4, 'expected_steps': 60, 'expected_mu_Pa': 1000.}
        return dict(accepted), None

    monkeypatch.setattr(x.readout, 'read_run', analytical_readout)
    monkeypatch.setattr(x, 'solve', lambda *args: pytest.fail('Reused evidence cannot launch a solver'))
    receipt = x.reuse_coarse(tmp_path, plan, state)
    assert state['solver_invocations'] == 1 and len(calls) == 1
    assert receipt['execution_binding'] == execution_binding
    saved = x.access.verify_binding(tmp_path, state['runs'][x.REUSED_RUN]['readout'], read_json=True)
    assert saved == receipt
    assert json.loads((directory/'state.json').read_text()) == state
    assert execution.read_bytes() == x.access.canonical_json({'primitive_bindings': primitives})
