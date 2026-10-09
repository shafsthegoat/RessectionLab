"""Generated metadata and tiny owned processes; never run the 256-grid worker."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import os
import sys
import time

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
loader = importlib.util.spec_from_file_location('streaming_profile_launcher_controls', HERE/'launch_profile.py')
launch = importlib.util.module_from_spec(loader); loader.loader.exec_module(launch)
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_v5_remaining_one_shot as supervisor
from scripts import febio_runtime

FORBIDDEN = []
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower()
        if name.endswith(('.nii', '.nii.gz', '.mat', '.tar', '.pt', '.ckpt', '.npz', '.npy', '.dcm')):
            FORBIDDEN.append(name); raise AssertionError('No patient/model payloads in launcher tests')
sys.addaudithook(guard)


def release_record(released=False):
    return {'schema': 'generated-streaming-profile-supervised-release-v1', 'execution_released': released,
            'launcher_sha256': launch.digest(launch.read_regular(HERE/'launch_profile.py')),
            'source_bindings': launch.SOURCES, 'caps': launch.CAPS, 'output_directory': launch.OUTPUT,
            'child_release_sha256': launch.digest(launch.child_release()), 'patient_or_model_access_permitted': False}


def generated_profile():
    record = {'touched_reference_cells': 1, 'positive_reference_cells': 0, 'unknown_reference_cells': 1,
              'biological_vessel_free': None, 'clinical_injury_probability': None,
              'outside_reference_fov': True, 'annotated_positive_encounter': None,
              'annotation_coverage_complete_for_sweep': False,
              'positive_cell_volume_upper_bound_mm3': 0., 'unknown_in_grid_cell_volume_mm3': .7**3}
    work = {'tiles_scanned': 3072, 'tile_capsule_pairs': 18432, 'tiles_pruned': 3071, 'tiles_evaluated': 1,
            'cell_capsule_pairs': 4096, 'reference_sampled_cells': 1, 'maximum_tile_cells': 4096,
            'maximum_geometry_batch': 256, 'maximum_coarse_geometry_batch': 256,
            'maximum_cell_geometry_batch': 256, 'maximum_action_masks': 3, 'reference_sample_calls': 1}
    budget = {'tile_edge': 16, 'coarse_batch': 256, 'max_tile_capsule_pairs': 2000000,
              'max_cell_capsule_pairs': 4000000, 'max_sampled_cells': 2000000, 'wall_seconds': 20.}
    origins = {'resectionlab.independent_geometry_batch': 'src/resectionlab/independent_geometry_batch.py'}
    return {'scope': 'one_generated_256x256x192_streaming_contact_profile',
            'source_hashes': json.loads(launch.child_release())['source_hashes'],
            'source_bytes_unchanged': True, 'prohibited_actions': [],
            'repository_sources_verified': len(json.loads((HERE/'repository-source-pins.json').read_bytes())),
            'loaded_repository_origins_before': origins, 'loaded_repository_origins_after': origins,
            'profile_seconds': .01, 'tracemalloc_current_bytes': 512, 'tracemalloc_peak_bytes': 1024, 'process_peak_rss_bytes': 16384,
            'result': {'status': 'complete_generated_contact_profile', 'patient_admission': False,
                       'strategy_replay_or_admission_performed': False, 'grid_shape': [256,256,192],
                       'grid_voxels': 12582912, 'capsule_count': 6, 'action_count': 3,
                       'contact_tolerance_squared_mm2': 1e-10, 'work': work, 'budget': budget, 'elapsed_seconds': .01,
                       'shaft': record, 'tip': record, 'whole_tool': record,
                       'per_action': {name: record for name in ('first','repeated','crossing')},
                       'removed_overlap': {'status': 'not_evaluated_by_contact_kernel', 'outcomes': None,
                                           'existing_congruence_requirement_unchanged': True}}}


def test_false_release_and_check_only(tmp_path, monkeypatch, capsys):
    path = tmp_path/'disabled.json'; path.write_text(json.dumps(release_record()))
    args = ['launch', '--release', str(path), '--release-sha256', launch.digest(path.read_bytes())]
    monkeypatch.setattr(sys, 'argv', args+['--check-only'])
    assert launch.main() == 0
    assert json.loads(capsys.readouterr().out)['execution_released'] is False
    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(sys, 'pycache_prefix', str(ROOT/launch.CACHE))
    with pytest.raises(RuntimeError, match='root_release_required'): launch.main()
    assert not (ROOT/launch.OUTPUT).exists()


def test_parent_cache_guard_precedes_dynamic_source_import(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['launch', '--release', 'unused', '--release-sha256', '0'*64])
    monkeypatch.setattr(sys, 'pycache_prefix', None)
    monkeypatch.setattr(launch, 'validate_release', lambda *a: pytest.fail('dynamic source validation preceded cache guard'))
    with pytest.raises(RuntimeError, match='fresh_parent_source_cache_required'):
        launch.main()


@pytest.mark.parametrize('bad', ['refusal', 'origin', 'claim', 'budget', 'duplicate', 'negative_time', 'nan_time',
                               'huge_count', 'encounter', 'volume', 'sample_count', 'origin_path'])
def test_exit_zero_is_insufficient_for_profile_semantics(bad):
    record = generated_profile()
    if bad == 'refusal': record['result']['status'] = 'budget_refused'
    elif bad == 'origin': record['source_bytes_unchanged'] = False
    elif bad == 'claim': record['result']['patient_admission'] = True
    elif bad == 'budget': record['result']['work']['cell_capsule_pairs'] = 4000001
    elif bad == 'duplicate': record['result']['per_action']['repeated'] = {}
    elif bad == 'negative_time': record['profile_seconds'] = -1.
    elif bad == 'nan_time': record['profile_seconds'] = float('nan')
    elif bad == 'huge_count': record['result']['shaft']['touched_reference_cells'] = 12582913
    elif bad == 'encounter': record['result']['shaft']['annotated_positive_encounter'] = False
    elif bad == 'volume': record['result']['shaft']['unknown_in_grid_cell_volume_mm3'] = 7.
    elif bad == 'sample_count': record['result']['work']['reference_sampled_cells'] = 2
    else: record['loaded_repository_origins_after']['resectionlab.independent_geometry_batch'] = 'somewhere/shadow.py'
    with pytest.raises((RuntimeError, KeyError)):
        launch.semantic_profile(json.dumps(record).encode())


@pytest.mark.parametrize('mode', ['success', 'semantic_refusal', 'parent_exception', 'cleanup_exception', 'extra_output',
                                'late_profile', 'late_receipt', 'late_file'])
def test_parent_receipt_and_failure_paths_use_generated_stage_only(mode, tmp_path, monkeypatch, capsys):
    directory = tmp_path/'attempt'
    monkeypatch.setattr(launch, 'OUTPUT', str(directory))
    release = tmp_path/'generated-only-release.json'; release.write_text(json.dumps(release_record(True)))
    release_sha = launch.digest(release.read_bytes())
    profile = SimpleNamespace(source_pins=lambda: None)
    monkeypatch.setattr(launch, 'validate_release', lambda *args: (release_record(True), profile))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(sys, 'pycache_prefix', str(ROOT/launch.CACHE))
    if mode == 'cleanup_exception':
        actual_load = launch.load_module
        def load(name, path):
            module = actual_load(name, path)
            if path == launch.OWNED:
                module.OwnedWorker.cleanup = lambda *a: (_ for _ in ()).throw(RuntimeError('generated cleanup failure'))
            return module
        monkeypatch.setattr(launch, 'load_module', load)
    def fake_stage(stage, command, folder, receipt, **kwargs):
        assert kwargs['wall_cap'] == 35 and kwargs['rss_cap'] == 536870912 and kwargs['output_cap'] == 4194304
        assert command[2:4] == ['-X', 'pycache_prefix='+str(directory/'worker/unused-pycache')]
        assert (folder/'receipt.json').exists()
        if mode == 'parent_exception': raise PermissionError('Generated exception without any child')
        (folder/'readout-console.txt').write_text('generated fake stage only\n')
        worker = folder/'worker'; worker.mkdir()
        (worker/'attempt.json').write_text('{}\n')
        value = generated_profile()
        if mode == 'semantic_refusal': value['result']['status'] = 'budget_refused'
        (worker/'profile.json').write_text(json.dumps(value))
        if mode == 'extra_output': (worker/'unexpected.txt').write_text('not allowed')
        result = {'status': 'completed_within_caps', 'exit_code': 0, 'generated_stage_only': True}
        receipt['readout_stage'] = result
        return result
    monkeypatch.setattr(supervisor, 'supervise_stage', fake_stage)
    actual_durable = supervisor.io.durable_json
    mutated = []
    def final_mutation(path, value):
        actual_durable(path, value)
        if mode.startswith('late_') and value.get('status') == 'generated_streaming_profile_complete' and not mutated:
            mutated.append(True)
            if mode == 'late_profile':
                with (directory/'worker/profile.json').open('ab') as stream: stream.write(b' ')
            elif mode == 'late_receipt':
                with path.open('ab') as stream: stream.write(b' ')
            else: (directory/'late.txt').write_text('late added regular file')
    monkeypatch.setattr(supervisor.io, 'durable_json', final_mutation)
    monkeypatch.setattr(sys, 'argv', ['launch', '--release', str(release), '--release-sha256', release_sha])
    assert launch.main() == (0 if mode == 'success' else 1)
    capsys.readouterr()
    receipt = json.loads((directory/'receipt.json').read_bytes())
    assert receipt['status'] == ('generated_streaming_profile_complete' if mode == 'success' else
                                  'failed_final_evidence_validation' if mode.startswith('late_') else
                                  'failed_cleanup_containment' if mode == 'cleanup_exception' else
                                  'failed_or_incomplete' if mode == 'parent_exception' else 'failed_profile_or_source_validation')


@pytest.mark.parametrize('mode', ['success', 'wall', 'rss', 'output', 'eperm'])
def test_tiny_owned_process_caps_and_reaping(mode, tmp_path, monkeypatch):
    owned = launch.load_module('tiny_owned_worker_'+mode, launch.OWNED)
    owner = owned.OwnedWorker()
    directory = tmp_path/mode; directory.mkdir()
    code = 'import time; print("generated-only",flush=True); time.sleep(.05)'
    wall, rss, output = 2, 128*1024**2, 1024**2
    if mode in ('wall','rss','eperm'): code = 'import time; time.sleep(2)'
    if mode == 'wall': wall = .25
    if mode in ('rss','eperm'): rss = 1
    if mode == 'output': code = 'import time; print("x"*16384,flush=True); time.sleep(2)'; output = 8192
    if mode == 'eperm':
        monkeypatch.setattr(supervisor.os, 'killpg', lambda *a: (_ for _ in ()).throw(PermissionError('generated EPERM')))
    receipt = {'scope': 'tiny_generated_process_only'}; started = time.monotonic()
    stage = supervisor.supervise_stage('readout', [sys.executable, '-B', '-c', code], directory, receipt,
        cwd=directory, environment=dict(os.environ), wall_cap=wall, rss_cap=rss, output_cap=output,
        rss_observer=febio_runtime.process_group_rss, popen=owner)
    cleanup = owner.cleanup(febio_runtime.process_group_rss, started+4)
    assert cleanup['contained'] and cleanup['direct_child_reaped'] and cleanup['remaining_members'] == []
    assert owner.process.poll() is not None
    assert stage['status'] == ('completed_within_caps' if mode == 'success' else 'failed_or_incomplete')
    if mode == 'eperm': assert cleanup['fallback_used'] is True
    receipt['cleanup'] = cleanup
    supervisor.io.durable_json(directory/'receipt.json', receipt)


def test_no_actual_payloads_or_profile_directory():
    assert FORBIDDEN == []
    assert not (ROOT/launch.OUTPUT).exists()
