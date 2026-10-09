"""No-release N8 supervisor controls; no FEBio or measured response access."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import mechanics_hbe_v5_n8_one_shot as runner
from scripts import mechanics_hbe_v5_frame as frame
from scripts import mechanics_hbe_v5_stream as reader
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts.mechanics_hbe_outputs import _iter_data_records, check_solver_records


def test_preparation_keeps_release_and_comparator_gates_closed():
    prep = json.loads((runner.ROOT / runner.PREPARATION).read_text())
    runner.validate_preparation(prep)
    assert prep['release'] is None and prep['caps'] == runner.CAPS
    assert prep['phase_gates'] == dict.fromkeys(prep['phase_gates'], False)
    assert runner.verify_old_control(prep)['elapsed_seconds'] == 4.249919041059911
    for mutation in ({'release': {}}, {'run_id': 'tension:N8:S60:reference'},
                     {'caps': {**runner.CAPS, 'wall_seconds': 900}},
                     {'phase_gates': {**prep['phase_gates'], 'fit': True}}):
        with pytest.raises(ValueError):
            runner.validate_preparation({**prep, **mutation})


def test_missing_release_fails_before_output_or_native(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.subprocess, 'Popen',
                        lambda *_a, **_k: pytest.fail('Native call forbidden'))
    with pytest.raises(FileNotFoundError):
        runner.execute(tmp_path / 'missing.json', root=tmp_path)
    assert not (tmp_path / runner.OUTPUT).exists()


def test_module_cli_missing_release_fails_before_output(tmp_path):
    target = runner.ROOT / runner.OUTPUT
    existed = target.exists()
    call = subprocess.run([sys.executable, '-B', '-m',
                           'scripts.mechanics_hbe_v5_n8_one_shot', '--execute',
                           '--release', str(tmp_path / 'missing.json')],
                          cwd=runner.ROOT, capture_output=True, text=True, timeout=15)
    assert call.returncode != 0
    assert target.exists() is existed


def test_bad_release_schema_cannot_reserve_attempt(tmp_path):
    release = tmp_path / 'release.json'
    release.write_text('{"status":"not-released"}')
    with pytest.raises(ValueError, match='one-call release'):
        runner.execute(release, root=tmp_path)
    assert not (tmp_path / runner.OUTPUT).exists()


def test_path_and_hash_bound_reject_escape_and_tamper(tmp_path):
    with pytest.raises(ValueError):
        runner.local('../escape', root=tmp_path)
    with pytest.raises(ValueError):
        runner.local('/escape', root=tmp_path)
    path = tmp_path / 'sample.txt'
    path.write_bytes(b'old')
    binding = {'path': 'sample.txt', 'sha256': runner.sha(b'old')}
    assert runner.bound(binding, 'sample.txt', root=tmp_path) == b'old'
    path.write_bytes(b'new')
    with pytest.raises(ValueError, match='hash'):
        runner.bound(binding, 'sample.txt', root=tmp_path)
    path.unlink()
    path.symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(ValueError, match='Symlink'):
        runner.bound(binding, 'sample.txt', root=tmp_path)


def test_complete_source_closure_required_before_git_access():
    with pytest.raises(ValueError, match='closure'):
        runner.source_hashes({'source_commit': 'a'*40, 'source_bindings': {}})
    with pytest.raises(ValueError, match='commit'):
        runner.source_hashes({'source_commit': 'bad',
                              'source_bindings': dict.fromkeys(runner.SOURCE_PATHS, {})})


def test_active_output_rejects_link_and_counts_bytes(tmp_path):
    (tmp_path / 'a').write_bytes(b'abc')
    (tmp_path / 'b').write_bytes(b'd')
    assert runner.active_bytes(tmp_path) == 4
    (tmp_path / 'link').symlink_to(tmp_path / 'a')
    with pytest.raises(ValueError, match='link'):
        runner.active_bytes(tmp_path)


class FakeProcess:
    pid = 424242

    def __init__(self, code=None):
        self.code = code
        self.waited = False

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.waited = True
        return self.code if self.code is not None else -9


def test_sampled_rss_cap_kills_one_process_family(tmp_path, monkeypatch):
    process = FakeProcess()
    killed = []
    monkeypatch.setattr(runner.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    receipt = runner.supervise('/private/febio4', tmp_path, {},
        popen=lambda *_a, **_k: process,
        rss_observer=lambda *_a, **_k: (runner.CAPS['sampled_process_group_rss_bytes']+1,
                                       [{'pid': process.pid}]),
        sleep=lambda _x: None)
    assert receipt['native_calls_attempted'] == 1
    assert receipt['kill_reason'] == 'process_group_rss_cap'
    assert receipt['status'] == 'failed_or_incomplete' and process.waited and killed


def test_active_output_cap_kills_one_process_family(tmp_path, monkeypatch):
    process = FakeProcess()
    monkeypatch.setattr(runner.os, 'killpg', lambda *_a: None)
    monkeypatch.setattr(runner, 'active_bytes', lambda _path: runner.CAPS['active_output_bytes']+1)
    receipt = runner.supervise('/private/febio4', tmp_path, {},
        popen=lambda *_a, **_k: process,
        rss_observer=lambda *_a, **_k: (0, [{'pid': process.pid}]),
        sleep=lambda _x: None)
    assert receipt['kill_reason'] == 'active_output_cap'
    assert receipt['status'] == 'failed_or_incomplete' and process.waited


def test_wall_cap_kills_one_process_family(tmp_path, monkeypatch):
    process = FakeProcess()
    ticks = iter((0., 90.1, 90.2))
    monkeypatch.setattr(runner.time, 'monotonic', lambda: next(ticks))
    monkeypatch.setattr(runner.os, 'killpg', lambda *_a: None)
    receipt = runner.supervise('/private/febio4', tmp_path, {},
        popen=lambda *_a, **_k: process,
        rss_observer=lambda *_a, **_k: pytest.fail('wall cap should fire first'),
        sleep=lambda _x: None)
    assert receipt['kill_reason'] == 'wall_cap'
    assert receipt['status'] == 'failed_or_incomplete' and process.waited


def test_clean_native_exit_is_not_numerical_pass(tmp_path):
    process = FakeProcess(0)
    receipt = runner.supervise('/private/febio4', tmp_path, {},
        popen=lambda *_a, **_k: process,
        rss_observer=lambda *_a, **_k: (0, []), sleep=lambda _x: None)
    assert receipt['status'] == 'native_exit_zero'
    with pytest.raises(ValueError, match='missing native output'):
        runner.inspect_outputs(tmp_path, {'deck': b''}, receipt)


def test_backend_selection_rejects_wrong_or_multiple_lines(tmp_path):
    for name in runner.EXPECTED_NATIVE:
        (tmp_path / name).write_text('placeholder\n')
    (tmp_path / 'specimen.feb').write_text('deck')
    context = {'deck': b'deck'}
    receipt = {'status': 'native_exit_zero'}
    for bad in ('* Selecting linear solver skyline *\n',
                '* Selecting linear solver accelerate *\n'*2,
                '* Selecting linear solver accelerate *\nfallback skyline\n',
                'no backend selection\n'):
        (tmp_path / 'console.txt').write_text(bad)
        with pytest.raises(ValueError, match='Accelerate'):
            runner.inspect_outputs(tmp_path, context, receipt, root=tmp_path)


def test_old_native_n8_saved_61_frame_grammar_and_adversarial_mutations():
    """Historical real FEBio bytes establish grammar only, not new-endpoint fit."""
    base = runner.ROOT / runner.OLD_RUN
    times = tuple(i/60 for i in range(61))
    nodes = (base / 'nodes.log').read_text().splitlines(keepends=True)
    elements = (base / 'elements.log').read_text().splitlines(keepends=True)
    solver = (base / 'solver.log').read_text().splitlines(keepends=True)
    def parse(data, count, fields, name):
        return list(_iter_data_records(data, expected_times=times,
            item_count=count, field_count=fields, record_name=name,
            maximum_bytes=32*1024**2, maximum_items=20000))
    assert len(parse(nodes, 1045, 9, 'mechanics_nodes_si')) == 61
    assert len(parse(elements, 768, 8, 'mechanics_elements_si')) == 61
    assert check_solver_records(solver, expected_times=times,
        residual_floor_N2=1e-30, maximum_bytes=32*1024**2)['passed']
    final = max(i for i, line in enumerate(nodes) if line.startswith('*Step'))
    with pytest.raises(ValueError, match='Missing final'):
        parse(nodes[:final], 1045, 9, 'mechanics_nodes_si')
    malformed = deepcopy(nodes)
    malformed[0] = '*Step = bad\n'
    with pytest.raises(ValueError, match='Malformed step'):
        parse(malformed, 1045, 9, 'mechanics_nodes_si')
    no_normal = [line for line in solver if 'T E R M I N A T I O N' not in line]
    with pytest.raises(ValueError, match='Incomplete converged'):
        check_solver_records(no_normal, expected_times=times,
                             residual_floor_N2=1e-30, maximum_bytes=32*1024**2)
    no_last_residual = deepcopy(solver)
    last = max(i for i, line in enumerate(no_last_residual)
               if line.split()[:1] == ['residual'])
    del no_last_residual[last]
    with pytest.raises(ValueError, match='residual evidence'):
        check_solver_records(no_last_residual, expected_times=times,
                             residual_floor_N2=1e-30, maximum_bytes=32*1024**2)
    excess = deepcopy(solver)
    fields = excess[last].split()
    fields[2] = '1e99'
    excess[last] = ' ' + ' '.join(fields) + '\n'
    assert check_solver_records(excess, expected_times=times,
        residual_floor_N2=1e-30, maximum_bytes=32*1024**2)['passed'] is False


def test_historical_old_endpoint_is_rejected_by_new_endpoint_readout():
    canary = json.loads((runner.ROOT / runner.CANARY).read_text())
    study = (runner.ROOT / v5.DECLARATION_PATH).read_bytes()
    prior = (runner.ROOT / json.loads(study)['previous_v4']['path']).read_bytes()
    old = (runner.ROOT / canary['old_source_deck']['path']).read_bytes()
    adapted = (runner.ROOT / canary['new_endpoint_decks']['new_endpoint_accelerate']['path']).read_bytes()
    mesh = json.loads((runner.ROOT / canary['full_native_mesh']['path']).read_bytes())
    contract = frame.verified_schedule(study, prior, runner.RUN_ID, old, adapted,
                                       canary['deck_adapter_receipt'])
    base = runner.ROOT / runner.OLD_RUN
    with (base / 'nodes.log').open() as nodes, (base / 'elements.log').open() as elements, \
         (base / 'solver.log').open() as solver:
        result = reader.evaluate_stream(contract, mesh, nodes, elements, solver)
    assert result['frame_count'] == 61
    assert result['solver']['passed'] is True
    assert result['numerical_passed'] is False
    assert result['criteria_max_ratio']['native_prescribed_motion'] > 1
