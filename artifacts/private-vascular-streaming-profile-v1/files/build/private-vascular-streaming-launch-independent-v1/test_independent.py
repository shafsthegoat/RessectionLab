"""Independent generated summaries and tiny child cleanup; no profile evaluation."""
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT/'build/private-vascular-streaming-preparation-v1'
spec = importlib.util.spec_from_file_location('independent_author_fixture', PREP/'test_launch_profile.py')
author = importlib.util.module_from_spec(spec); spec.loader.exec_module(author)
launch, supervisor = author.launch, author.supervisor
ACTUAL_OUTPUT = ROOT/launch.OUTPUT
PROCESS_CALLS = []


def guard(event, args):
    if event == 'subprocess.Popen':
        command = args[1]
        valid = (command[0] in ('/usr/bin/git', '/bin/ps')
                 or (command[0] == sys.executable and command[1:3] == ['-B', '-c']))
        if not valid:
            raise AssertionError('Only historical source checks, numeric ps and tiny -c children allowed')
        PROCESS_CALLS.append(command)
    if event in ('socket.connect', 'os.system'):
        raise AssertionError('No network or shell operation in independent controls')
sys.addaudithook(guard)


@pytest.mark.parametrize('bad', ['partition', 'coverage', 'nonfinite_volume', 'batch', 'action_masks',
                                'trace_partition', 'rss', 'union', 'origin_namespace', 'missing_work'])
def test_additional_semantic_contradictions_refused(bad):
    value = copy.deepcopy(author.generated_profile())
    assert launch.semantic_profile(json.dumps(value).encode())['status'] == 'generated_streaming_profile_complete'
    result = value['result']
    if bad == 'partition': result['shaft']['positive_reference_cells'] = 1
    elif bad == 'coverage': result['shaft']['annotation_coverage_complete_for_sweep'] = True
    elif bad == 'nonfinite_volume': result['shaft']['unknown_in_grid_cell_volume_mm3'] = float('nan')
    elif bad == 'batch': result['work']['maximum_geometry_batch'] = 255
    elif bad == 'action_masks': result['work']['maximum_action_masks'] = 4
    elif bad == 'trace_partition': value['tracemalloc_current_bytes'] = 1025
    elif bad == 'rss': value['process_peak_rss_bytes'] = 536870913
    elif bad == 'union':
        result['shaft'] = copy.deepcopy(result['shaft'])
        result['shaft']['touched_reference_cells'] = 2
    elif bad == 'origin_namespace':
        value['loaded_repository_origins_after']['elsewhere'] = 'src/resectionlab/evaluation.py'
    else: del result['work']['maximum_action_masks']
    with pytest.raises((RuntimeError, KeyError)):
        launch.semantic_profile(json.dumps(value).encode())


def fake_main_setup(tmp_path, monkeypatch):
    directory = tmp_path/'attempt'
    monkeypatch.setattr(launch, 'OUTPUT', str(directory))
    record = author.release_record(True)
    release = tmp_path/'release.json'; release.write_text(json.dumps(record))
    monkeypatch.setattr(launch, 'validate_release', lambda *args: (record, SimpleNamespace(source_pins=lambda: None)))
    monkeypatch.setattr(sys, 'dont_write_bytecode', True)
    monkeypatch.setattr(sys, 'pycache_prefix', str(ROOT/launch.CACHE))
    monkeypatch.setattr(sys, 'argv', ['launch', '--release', str(release), '--release-sha256', launch.digest(release.read_bytes())])
    def fake_stage(stage, command, folder, receipt, **kwargs):
        (folder/'readout-console.txt').write_text('independent generated stage only\n')
        worker = folder/'worker'; worker.mkdir()
        (worker/'attempt.json').write_text('{}\n')
        (worker/'profile.json').write_text(json.dumps(author.generated_profile()))
        result = {'status': 'completed_within_caps', 'exit_code': 0}
        receipt['readout_stage'] = result
        return result
    monkeypatch.setattr(supervisor, 'supervise_stage', fake_stage)
    return directory, release


@pytest.mark.parametrize('target', ['console', 'attempt', 'release', 'launcher'])
def test_additional_final_mutations_cannot_pass(target, tmp_path, monkeypatch, capsys):
    directory, release = fake_main_setup(tmp_path, monkeypatch)
    original_durable, original_read = supervisor.io.durable_json, launch.read_regular
    mutated = []
    def durable(path, value):
        original_durable(path, value)
        if value.get('status') == 'generated_streaming_profile_complete' and not mutated:
            mutated.append(True)
            if target != 'launcher':
                path = {'console': directory/'readout-console.txt', 'attempt': directory/'worker/attempt.json',
                        'release': release}[target]
                with path.open('ab') as stream: stream.write(b' ')
    def read(path, *args, **kwargs):
        if target == 'launcher' and mutated and Path(path) == Path(launch.__file__):
            return b'independent synthetic changed launcher bytes'
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(supervisor.io, 'durable_json', durable)
    monkeypatch.setattr(launch, 'read_regular', read)
    assert launch.main() == 1
    capsys.readouterr()
    assert mutated
    receipt = json.loads((directory/'receipt.json').read_bytes())
    assert receipt['status'] == 'failed_final_evidence_validation'
    assert receipt['output_bindings']


def test_parent_exception_after_spawn_still_kills_and_reaps(tmp_path, monkeypatch, capsys):
    directory, release = fake_main_setup(tmp_path, monkeypatch)
    children = []
    def spawn_then_raise(stage, command, folder, receipt, **kwargs):
        assert (folder/'receipt.json').exists()
        process = kwargs['popen']([sys.executable, '-B', '-c', 'import time; time.sleep(2)'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        children.append(process)
        raise PermissionError('Independent generated parent exception after child spawn')
    monkeypatch.setattr(supervisor, 'supervise_stage', spawn_then_raise)
    try:
        assert launch.main() == 1
        capsys.readouterr()
        receipt = json.loads((directory/'receipt.json').read_bytes())
        assert receipt['status'] == 'failed_or_incomplete'
        assert receipt['parent_error_type'] == 'PermissionError'
        assert receipt['cleanup']['contained'] and receipt['cleanup']['direct_child_reaped']
        assert receipt['cleanup']['fallback_used'] and receipt['cleanup']['remaining_members'] == []
        assert children[0].poll() is not None
    finally:
        for process in children:
            if process.poll() is None: process.kill()
            process.wait(timeout=2)


def test_existing_attempt_refused_before_worker_import(tmp_path, monkeypatch):
    directory, release = fake_main_setup(tmp_path, monkeypatch)
    directory.mkdir()
    monkeypatch.setattr(launch, 'load_module', lambda *args: pytest.fail('Second attempt reached worker import'))
    with pytest.raises(RuntimeError, match='one_attempt_output_exists'):
        launch.main()


def test_source_rebind_exact_and_no_realistic_execution():
    old = json.loads((PREP/'prior-repository-source-pins-v1.json').read_bytes())
    new = json.loads((PREP/'repository-source-pins.json').read_bytes())
    changes = {name: {'before': old.get(name), 'after': new.get(name)}
               for name in sorted(set(old)|set(new)) if old.get(name) != new.get(name)}
    declaration = json.loads((PREP/'source-rebind-v2.json').read_bytes())
    assert changes == declaration['exact_changes']
    assert set(changes) == {'src/resectionlab/matched_private_vascular.py', 'src/resectionlab/private_vascular_evaluation.py'}
    assert len(old) == 76 and len(new) == 77
    assert all(launch.digest((ROOT/name).read_bytes()) == digest for name, digest in new.items())
    assert not ACTUAL_OUTPUT.exists()
    assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)
    assert not author.FORBIDDEN
