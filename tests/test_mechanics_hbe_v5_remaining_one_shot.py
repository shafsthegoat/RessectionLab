"""Source-only guards for the eleven-row HBE v5 supervisor; no FEBio call."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import mechanics_hbe_v5_remaining_one_shot as runner
from scripts import mechanics_hbe_v5_n8_one_shot as n8
from scripts import mechanics_hbe_v5_stream as stream
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_source_bindings as sources


def test_preparation_caps_order_and_closed_release():
    prep = json.loads((runner.ROOT / runner.PREPARATION).read_text())
    runner.validate_preparation(prep)
    assert prep['release'] is None
    assert len(prep['per_row']) == 11
    assert sum(runner.NATIVE_WALL[1:]) == 9510
    assert sum(runner.OUTPUT_MIB[1:]) == 5952
    assert (sum(runner.OUTPUT_MIB[1:]) + 64) * 1024**2 < 6 * 1024**3
    assert 9600 + 6600 + 1800 == runner.AGGREGATE['known_stage_total_wall_seconds']
    assert prep['fixed_n8_predecessor']['readout_elapsed_seconds'] is None
    assert prep['fixed_n8_predecessor']['prep_elapsed_seconds'] is None
    for change in ({'release': {}}, {'per_row': prep['per_row'][1:]},
                   {'aggregate': {**runner.AGGREGATE, 'known_stage_total_wall_seconds': 19000}},
                   {'phase_gates': {**prep['phase_gates'], 'fit': True}}):
        with pytest.raises(ValueError):
            runner.validate_preparation({**prep, **change})


def test_real_n8_receipt_is_first_predecessor_with_closed_actual_bytes():
    prior = runner.validate_prior_chain([{
        'run_id': runner.ORDER[0], 'path': runner.N8_PATH,
        'sha256': runner.N8_SHA}], 1)
    assert prior['sha256'] == [runner.N8_SHA]
    assert prior['native_calls'] == 1
    assert prior['native_seconds'] == runner.N8_NATIVE_SECONDS
    assert prior['output_bytes'] == 16075117
    assert runner.N8_RECORDED_OUTPUT_BYTES == 15488457
    assert prior['readout_seconds'] == prior['prep_seconds'] == 0.


def test_predecessor_order_and_hash_mutations_rejected():
    good = [{'run_id': runner.ORDER[0], 'path': runner.N8_PATH,
             'sha256': runner.N8_SHA}]
    for wrong in ([], [{'run_id': runner.ORDER[1], **{k:v for k,v in good[0].items() if k != 'run_id'}}],
                  [{**good[0], 'sha256': '0'*64}],
                  [{**good[0], 'path': runner.output_directory(1) + '/receipt.json'}]):
        with pytest.raises(ValueError):
            runner.validate_prior_chain(wrong, 1)


def test_second_remaining_row_rebinds_prior_release_and_assumptions(tmp_path):
    old_directory = runner.ROOT / Path(runner.N8_PATH).parent
    first_directory = tmp_path / Path(runner.N8_PATH).parent
    first_directory.mkdir(parents=True)
    for path in old_directory.iterdir():
        shutil.copyfile(path, first_directory / path.name)
    n8_item = {'run_id': runner.ORDER[0], 'path': runner.N8_PATH,
               'sha256': runner.N8_SHA}
    directory = tmp_path / runner.output_directory(1)
    directory.mkdir(parents=True)
    for name in runner.FINAL_FILES - {'receipt.json', 'readout-work-order.json'}:
        (directory / name).write_bytes(name.encode())
    token_hash = 'f'*64
    n8.durable_json(directory / 'readout-work-order.json',
                    {'readout_token_sha256': token_hash})
    records = {name: runner._regular_output_binding(directory / name,
               root=tmp_path, limit=64*1024**2)
               for name in runner.FINAL_FILES - {'receipt.json'}}
    profile = {'path': 'frozen-profile', 'sha256': 'a'*64}
    runtime = {'path': 'frozen-runtime', 'sha256': 'b'*64}
    source_bindings = {name: {'path': name, 'sha256': 'c'*64}
                       for name in runner.SOURCE_PATHS}
    release = {'schema': 'hbe-v5-remaining-one-call-release-v1',
               'status': 'root_released_one_native_call', 'ordinal': 1,
               'run_id': runner.ORDER[1], 'output_directory': runner.output_directory(1),
               'caps': runner.caps(1), 'aggregate_caps': runner.AGGREGATE,
               'prior_receipts': [n8_item], 'source_commit': 'd'*40,
               'source_bindings': source_bindings,
               'adapted_deck_sha256': records['specimen.feb']['sha256'],
               'runtime_identity': runtime, 'backend_profile': profile,
               'preparation': {'sha256': '1'*64},
               'v5_declaration': {'sha256': '2'*64},
               'source_map': {'sha256': '3'*64},
               'old_source_deck': {'sha256': '4'*64},
               'native_mesh': {'sha256': '5'*64}}
    release_path = tmp_path / 'release.json'
    n8.durable_json(release_path, release)
    release_sha = n8.file_hash(release_path)
    document = {'schema': 'hbe-v5-remaining-one-call-receipt-v1',
                'status': 'passed_numerical_software_only', 'ordinal': 1,
                'run_id': runner.ORDER[1], 'native_calls_attempted': 1,
                'readout_calls_attempted': 1, 'no_retry': True,
                'caps': runner.caps(1), 'source_commit': 'd'*40,
                'release_path': str(release_path), 'release_sha256': release_sha,
                'adapted_deck_sha256': records['specimen.feb']['sha256'],
                'runtime_identity_sha256': runtime['sha256'],
                'backend_profile_sha256': profile['sha256'],
                'preparation_sha256': '1'*64, 'v5_declaration_sha256': '2'*64,
                'source_map_sha256': '3'*64, 'old_source_deck_sha256': '4'*64,
                'native_mesh_sha256': '5'*64,
                'source_hashes_before': {name: 'c'*64 for name in runner.SOURCE_PATHS},
                'source_hashes_after': {name: 'c'*64 for name in runner.SOURCE_PATHS},
                'observed_head_preflight': 'd'*40,
                'observed_head_after': 'e'*40,
                'runtime_profile_verified_after': True,
                'readout_token_sha256': token_hash,
                'prior_receipt_sha256': [runner.N8_SHA],
                'prior_native_wall_seconds': runner.N8_NATIVE_SECONDS,
                'prior_readout_wall_seconds': 0., 'prior_prep_wall_seconds': 0.,
                'prior_active_output_bytes': runner.N8_OUTPUT_BYTES,
                'native_stage': {'status': 'completed_within_caps', 'exit_code': 0,
                                 'kill_reason': None, 'elapsed_seconds': 1.},
                'readout_stage': {'status': 'completed_within_caps', 'exit_code': 0,
                                  'kill_reason': None, 'elapsed_seconds': 2.},
                'prep_elapsed_seconds': 3.,
                'aggregate_native_wall_seconds': runner.N8_NATIVE_SECONDS + 1.,
                'aggregate_readout_wall_seconds': 2.,
                'aggregate_prep_wall_seconds': 3.,
                'aggregate_native_calls': 2,
                'output_bindings': records,
                'native_output_bindings': {name: records[name]
                                           for name in runner.NATIVE_FILES},
                'saved_numerical_readout': {'run_id': runner.ORDER[1],
                    'frame_count': 61, 'numerical_passed': True,
                    'readout_sha256': records['readout.json']['sha256'],
                    'native_output_observed': True,
                    'physical_validation_pass': None}}
    receipt_path = directory / 'receipt.json'
    n8.durable_json(receipt_path, document)
    chain = [n8_item, {'run_id': runner.ORDER[1],
                       'path': runner.receipt_path(1),
                       'sha256': n8.file_hash(receipt_path)}]
    prior = runner.validate_prior_chain(chain, 2, root=tmp_path,
        expected_profile=profile, expected_runtime=runtime)
    assert prior['native_calls'] == 2
    assert prior['output_bytes'] == runner.N8_OUTPUT_BYTES + runner.active_bytes(
        directory, runner.caps(1)['active_output_bytes'])
    with pytest.raises(ValueError, match='runtime ancestry'):
        runner.validate_prior_chain(chain, 2, root=tmp_path)
    with pytest.raises(ValueError, match='release/source/runtime/deck'):
        runner.validate_prior_chain(chain, 2, root=tmp_path,
            expected_profile=profile, expected_runtime={'sha256': '0'*64})
    release_path.write_text(release_path.read_text() + ' ')
    with pytest.raises(ValueError, match='release bytes changed'):
        runner.validate_prior_chain(chain, 2, root=tmp_path,
            expected_profile=profile, expected_runtime=runtime)


def test_missing_or_malformed_release_does_not_reserve_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda *_a, **_k: pytest.fail('Native or worker launch forbidden'))
    output = tmp_path / runner.output_directory(1)
    with pytest.raises(FileNotFoundError):
        runner.execute(tmp_path / 'absent.json', root=tmp_path)
    release = tmp_path / 'bad.json'
    release.write_text('{"status":"not-released"}')
    with pytest.raises(ValueError, match='one-row'):
        runner.execute(release, root=tmp_path)
    assert not output.exists()


def test_module_cli_missing_release_fails_before_output(tmp_path):
    output = runner.ROOT / runner.output_directory(1)
    existed = output.exists()
    result = subprocess.run([sys.executable, '-B', '-m',
        'scripts.mechanics_hbe_v5_remaining_one_shot', '--execute',
        '--release', str(tmp_path / 'missing.json')], cwd=runner.ROOT,
        capture_output=True, text=True, timeout=15)
    assert result.returncode != 0
    assert output.exists() is existed


def test_source_closure_and_bound_path_guard(tmp_path):
    assert 'scripts/mechanics_hbe_v5_n8_one_shot.py' in runner.SOURCE_PATHS
    assert 'scripts/mechanics_hbe_v5_remaining_one_shot.py' in runner.SOURCE_PATHS
    assert 'scripts/mechanics_nonpatient_sparse_continuation_supervisor.py' not in runner.SOURCE_PATHS
    with pytest.raises(ValueError, match='closure'):
        runner.source_hashes({'source_commit': 'a'*40, 'source_bindings': {}})
    with pytest.raises(ValueError):
        n8.local('../escape', root=tmp_path)


def test_launch_head_is_exact_but_unrelated_commit_after_launch_is_allowed(tmp_path):
    subprocess.run(['/usr/bin/git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'config', 'user.name', 'Fixture'],
                   cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'config', 'user.email', 'fixture@example.invalid'],
                   cwd=tmp_path, check=True)
    bindings = {}
    for relative in runner.SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = relative.encode()
        path.write_bytes(data)
        bindings[relative] = {'path': relative, 'sha256': n8.sha(data)}
    subprocess.run(['/usr/bin/git', 'add', 'scripts'], cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'commit', '-qm', 'fixed source fixture'],
                   cwd=tmp_path, check=True)
    commit = subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=tmp_path,
                            capture_output=True, text=True, check=True).stdout.strip()
    release = {'source_commit': commit, 'source_bindings': bindings}
    assert runner.source_hashes(release, root=tmp_path) == (
        {key: value['sha256'] for key, value in bindings.items()}, commit)
    (tmp_path / 'UNRELATED.md').write_text('documentation only\n')
    subprocess.run(['/usr/bin/git', 'add', 'UNRELATED.md'], cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'commit', '-qm', 'unrelated documentation'],
                   cwd=tmp_path, check=True)
    _, later_head = runner.source_hashes(release, root=tmp_path, require_head=False)
    assert later_head != commit
    with pytest.raises(ValueError, match='Checkout changed'):
        runner.source_hashes(release, root=tmp_path)
    changed = tmp_path / runner.SOURCE_PATHS[0]
    changed.write_bytes(b'changed executed source')
    with pytest.raises(ValueError, match='Bound file hash differs'):
        runner.source_hashes(release, root=tmp_path, require_head=False)


def test_all_eleven_exact_source_mesh_adapters_without_native_or_response(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen',
                        lambda *_a, **_k: pytest.fail('FEBio launch forbidden'))
    root = runner.ROOT
    study, prior = v5.validate_preparation(root)
    sources.validate_binding_manifest(root, inspect_sources=True)
    bound = json.loads((root / sources.BINDING_PATH).read_text())
    assert tuple(row['run_id'] for row in study['ordered_reference_runs']) == runner.ORDER
    for index, run_id in enumerate(runner.ORDER[1:], 1):
        key = bound['run_source_keys'][run_id]
        source = bound['source_decks'][key]
        mesh = bound['meshes'][source['mesh']]
        old = sources._read_bound(root, {k: source[k] for k in ('path', 'sha256')},
                                  maximum=sources.MAX_SOURCE_BYTES)
        native = json.loads(sources._read_bound(root, mesh,
                                                maximum=sources.MAX_MESH_BYTES))
        sources.validate_topology_bc(native, old,
            native_domain=v5.run_spec(study, prior, run_id)['native_domain'])
        _, adapted, receipt = v5.adapt_deck(study, prior, run_id, old)
        assert n8.sha(adapted.encode()) == receipt['adapted_deck_sha256']
        assert len(receipt['times']) == (121 if ':S120:' in run_id else 61)
        assert runner.caps(index)['active_output_bytes'] == runner.OUTPUT_MIB[index]*1024**2


def test_actual_n8_saved_stream_replays_but_remains_unverified_without_receipt():
    manifest = json.loads((runner.ROOT / n8.CANARY).read_text())
    receipt = json.loads((runner.ROOT / n8.OUTPUT / 'receipt.json').read_text())
    native = receipt['native_output_bindings']
    bindings = {'source_deck': manifest['old_source_deck'],
                'mesh': manifest['full_native_mesh'],
                'adapted_deck': {k: native['specimen.feb'][k]
                                 for k in ('path', 'sha256')}}
    for name, key in [('nodes.log','nodes'),('elements.log','elements'),('solver.log','solver')]:
        bindings[key] = {field: native[name][field] for field in ('path', 'sha256')}
    result = stream.read_bound_run(runner.ROOT, runner.ORDER[0], bindings,
                                   manifest['deck_adapter_receipt'])
    assert result['numerical_passed'] is True and result['frame_count'] == 61
    assert result['provenance']['output_origin'] == 'unverified_saved_stream'


class FakeProcess:
    pid = 434343

    def __init__(self, code=None):
        self.code = code
        self.waited = False

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.waited = True
        return self.code if self.code is not None else -9


def _stage(tmp_path, process, observer):
    return runner.supervise_stage('native', ['/private/febio4'], tmp_path, {},
        cwd=tmp_path, environment={}, wall_cap=90, rss_cap=100,
        output_cap=10000, popen=lambda *_a, **_k: process,
        rss_observer=observer, sleep=lambda _x: None)


def test_rss_violation_kills_single_process_family(tmp_path, monkeypatch):
    proc = FakeProcess()
    killed = []
    monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    result = _stage(tmp_path, proc, lambda *_a, **_k: (101, [{'pid': proc.pid}]))
    assert result['kill_reason'] == 'process_group_rss_cap'
    assert result['status'] == 'failed_or_incomplete' and proc.waited and killed


def test_active_output_violation_kills_single_process_family(tmp_path, monkeypatch):
    proc = FakeProcess()
    monkeypatch.setattr(runner.os, 'killpg', lambda *_a: None)
    monkeypatch.setattr(runner, 'active_bytes', lambda *_a: 10001)
    result = _stage(tmp_path, proc, lambda *_a, **_k: (0, [{'pid': proc.pid}]))
    assert result['kill_reason'] == 'active_output_cap'
    assert result['status'] == 'failed_or_incomplete' and proc.waited


def test_wall_violation_kills_single_process_family(tmp_path, monkeypatch):
    proc = FakeProcess()
    killed = []
    clock = iter((0., 2., 3.))
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(clock))
    monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    result = runner.supervise_stage('readout', ['/private/python'], tmp_path, {},
        cwd=tmp_path, environment={}, wall_cap=1, rss_cap=100,
        output_cap=10000, popen=lambda *_a, **_k: proc,
        rss_observer=lambda *_a, **_k: pytest.fail('Should kill before RSS sampling'))
    assert result['kill_reason'] == 'wall_cap'
    assert result['status'] == 'failed_or_incomplete' and proc.waited and killed


def test_clean_exit_is_only_stage_completion(tmp_path):
    proc = FakeProcess(0)
    result = _stage(tmp_path, proc, lambda *_a, **_k: (0, []))
    assert result['status'] == 'completed_within_caps'
    assert result['exit_code'] == 0
    with pytest.raises(ValueError, match='inventory'):
        runner.inspect_native(tmp_path, {'index': 1, 'deck': b''}, root=tmp_path)


def test_backend_must_be_accelerate_in_console_and_solver(tmp_path):
    valid = runner.SELECTION + '\n'
    for name in runner.NATIVE_FILES | {'receipt.json'}:
        (tmp_path / name).write_text('x')
    (tmp_path / 'specimen.feb').write_bytes(b'deck')
    context = {'index': 1, 'deck': b'deck'}
    (tmp_path / 'console.txt').write_text(valid)
    (tmp_path / 'solver.log').write_text(valid)
    assert runner.inspect_native(tmp_path, context, root=tmp_path)['specimen.feb']['sha256'] == n8.sha(b'deck')
    for bad in ('* Selecting linear solver skyline *\n', valid*2,
                valid + 'fallback skyline\n', 'no selection\n'):
        (tmp_path / 'solver.log').write_text(bad)
        with pytest.raises(ValueError, match='Accelerate'):
            runner.inspect_native(tmp_path, context, root=tmp_path)
    (tmp_path / 'solver.log').write_text(valid)
    (tmp_path / 'console.txt').write_text(valid + 'switching solver to skyline\n')
    with pytest.raises(ValueError, match='Accelerate'):
        runner.inspect_native(tmp_path, context, root=tmp_path)


@pytest.mark.parametrize('index,domain,frames', [
    (1, 'full_native', 61), (6, 'lower_half_reconstructed', 121)])
def test_readout_binds_exact_work_order_frames_and_solver(tmp_path, index, domain, frames):
    directory = tmp_path / runner.output_directory(index)
    directory.mkdir(parents=True)
    context = {'index': index, 'run_id': runner.ORDER[index], 'deck': b'adapted',
               'native_domain': domain, 'v5_declaration_sha256': 'e'*64,
               'old_source': {'path': 'old', 'sha256': 'a'*64},
               'mesh': {'path': 'mesh', 'sha256': 'b'*64},
               'adapter_receipt': {'adapted_deck_sha256': n8.sha(b'adapted')}}
    receipt = {'release_sha256': 'c'*64, 'source_commit': 'd'*40}
    for name in runner.NATIVE_FILES | {'receipt.json', 'readout-console.txt'}:
        (directory / name).write_bytes(b'one')
    (directory / 'specimen.feb').write_bytes(b'adapted')
    native = {name: runner._regular_output_binding(directory / name,
              root=tmp_path, limit=64*1024**2) for name in runner.NATIVE_FILES}
    order = runner._work_order(context, receipt, native, 'token')
    receipt['readout_token_sha256'] = order['readout_token_sha256']
    n8.durable_json(directory / 'readout-work-order.json', order)
    provenance = {'output_origin': 'unverified_saved_stream',
                  'source_binding_checked': True,
                  'native_output_observed': None,
                  'generated_fixture_only': None,
                  'physical_validation_pass': None,
                  'measured_response_accessed': False,
                  'patient_data_accessed': False,
                  'v5_declaration_sha256': 'e'*64,
                  'adapted_deck_sha256': n8.sha(b'adapted'),
                  'source_deck_sha256': 'a'*64,
                  'primitive_bindings': order['bindings'],
                  'reconstruction_provenance': ('reflected_native_half_not_native_full'
                                                if domain == 'lower_half_reconstructed'
                                                else 'full_native')}
    readout = {'schema': 'hbe-v5-complete-stream-v1', 'run_id': runner.ORDER[index],
               'frame_count': frames, 'steps': frames-1,
               'representation': ('reconstructed_full'
                                  if domain == 'lower_half_reconstructed'
                                  else 'full_native_fixture'),
               'numerical_passed': True,
               'solver': {'passed': True}, 'full_energy_work': {'passed': True},
               'native_energy_work': ({'passed': True}
                                      if domain == 'lower_half_reconstructed' else None),
               'provenance': provenance}
    n8.durable_json(directory / 'readout.json', readout)
    records, summary = runner.inspect_readout(directory, context, native, receipt,
                                               root=tmp_path)
    assert summary['native_output_observed'] is True
    assert records['readout.json']['sha256'] == n8.file_hash(directory / 'readout.json')
    for field, invalid in [('frame_count', frames-1), ('steps', frames-2),
                           ('representation', 'wrong'), ('numerical_passed', False),
                           ('solver', {'passed': False}),
                           ('native_energy_work', {'passed': False})
                           if domain == 'lower_half_reconstructed' else
                           ('native_energy_work', {'passed': True})]:
        bad = deepcopy(readout)
        bad[field] = invalid
        n8.durable_json(directory / 'readout.json', bad)
        with pytest.raises(ValueError, match='Saved readout'):
            runner.inspect_readout(directory, context, native, receipt, root=tmp_path)
    n8.durable_json(directory / 'readout.json', readout)
    bad = deepcopy(readout)
    bad['provenance']['v5_declaration_sha256'] = '0'*64
    n8.durable_json(directory / 'readout.json', bad)
    with pytest.raises(ValueError, match='Saved readout'):
        runner.inspect_readout(directory, context, native, receipt, root=tmp_path)
    n8.durable_json(directory / 'readout.json', readout)
    order['readout_token_sha256'] = '0'*64
    n8.durable_json(directory / 'readout-work-order.json', order)
    with pytest.raises(ValueError, match='work order'):
        runner.inspect_readout(directory, context, native, receipt, root=tmp_path)
    order['readout_token_sha256'] = receipt['readout_token_sha256']
    order['bindings']['mesh'] = {'path': 'wrong', 'sha256': '0'*64}
    n8.durable_json(directory / 'readout-work-order.json', order)
    with pytest.raises(ValueError, match='work order'):
        runner.inspect_readout(directory, context, native, receipt, root=tmp_path)


def test_worker_requires_unshared_parent_token_before_readout(tmp_path, monkeypatch):
    monkeypatch.delenv('HBE_V5_READOUT_TOKEN', raising=False)
    directory = tmp_path / runner.output_directory(1)
    directory.mkdir(parents=True)
    order = {'schema': 'hbe-v5-supervised-readout-work-order-v1',
             'ordinal': 1, 'run_id': runner.ORDER[1],
             'release_sha256': 'a'*64, 'source_commit': 'a'*40,
             'readout_token_sha256': 'b'*64, 'bindings': {},
             'adapter_receipt': {}}
    n8.durable_json(directory / 'readout-work-order.json', order)
    with pytest.raises(ValueError, match='supervised'):
        runner.readout_worker(runner.output_directory(1) + '/readout-work-order.json',
                              root=tmp_path)
    assert not (directory / 'readout.json').exists()


def test_aggregate_stage_caps_and_closed_output_accounting():
    prior = {'native_seconds': runner.N8_NATIVE_SECONDS, 'readout_seconds': 0.,
             'prep_seconds': 0., 'output_bytes': runner.N8_OUTPUT_BYTES,
             'native_calls': 1}
    values = runner._check_aggregate(prior, 1., 2., 3., 4, 1)
    assert values['aggregate_native_calls'] == 2
    assert values['aggregate_output_bytes'] == runner.N8_OUTPUT_BYTES + 4
    with pytest.raises(ValueError, match='Aggregate'):
        runner._check_aggregate(prior, 9600., 0., 0., 0, 1)
    with pytest.raises(ValueError, match='Aggregate'):
        runner._check_aggregate(prior, 0., 6601., 0., 0, 1)
    with pytest.raises(ValueError, match='Aggregate'):
        runner._check_aggregate(prior, 0., 0., 1801., 0, 1)
