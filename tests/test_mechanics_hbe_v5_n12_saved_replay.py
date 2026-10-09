"""Replay-only N12 packaging recovery; no FEBio launch or admission."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from scripts import mechanics_hbe_v5_n12_saved_replay as replay
from scripts import mechanics_hbe_v5_remaining_one_shot as prior


def preparation():
    return json.loads((replay.ROOT / replay.PREPARATION).read_text())


def test_exact_one_attempt_whitelist_and_closed_phase_gates():
    prep = preparation()
    replay.validate_preparation(prep)
    assert prep['release'] is None
    assert prep['original_receipt_sha256'] == replay.ORIGINAL_RECEIPT_SHA
    assert prep['closed_attempt_files']['readout-console.txt']['bytes'] == 0
    assert prep['closed_attempt_bytes'] == sum(
        record['bytes'] for record in prep['closed_attempt_files'].values())
    assert prep['caps']['native_calls'] == 0
    assert not any(prep['phase_gates'].values())
    for key, bad in [('release', {}), ('original_receipt_sha256', '0'*64),
                     ('closed_attempt_bytes', prep['closed_attempt_bytes']-1)]:
        with pytest.raises(ValueError):
            replay.validate_preparation({**prep, key: bad})


def test_original_failed_receipt_release_source_runtime_and_saved_stream():
    # Use a fresh interpreter: a combined pytest process imports unrelated
    # scripts, whereas the one-shot CLI must audit only its executing closure.
    code = '''
import json
from pathlib import Path
from scripts import mechanics_hbe_v5_n12_saved_replay as replay
from scripts import mechanics_hbe_v5_stream as stream
prep = json.loads((replay.ROOT / replay.PREPARATION).read_text())
original = replay.validate_original(prep)
work = original['work_order']
independent = stream.read_bound_run(replay.ROOT, replay.prior.ORDER[1],
    work['bindings'], work['adapter_receipt'])
print(json.dumps({'original_status': original['receipt']['status'],
    'after_source_recorded': 'source_hashes_after' in original['receipt'],
    'after_runtime_recorded': 'runtime_profile_verified_after' in original['receipt'],
    'source_commit': original['release']['source_commit'],
    'files': len(original['inventory']),
    'empty_console': original['inventory']['readout-console.txt']['bytes'],
    'exact_replay': independent == original['saved_json'],
    'numerical_passed': independent['numerical_passed'],
    'frames': independent['frame_count'], 'steps': independent['steps'],
    'maximum_ratio': max(independent['criteria_max_ratio'].values())}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', code], cwd=replay.ROOT,
                            capture_output=True, text=True, check=True, timeout=30)
    observed = json.loads(result.stdout)
    ratio = observed.pop('maximum_ratio')
    assert observed == {
        'original_status': 'failed_or_incomplete',
        'after_source_recorded': False, 'after_runtime_recorded': False,
        'source_commit': replay.ORIGINAL_SOURCE_COMMIT, 'files': 9,
        'empty_console': 0, 'exact_replay': True, 'numerical_passed': True,
        'frames': 61, 'steps': 60}
    assert ratio < 1


def test_exact_closed_attempt_file_tamper_and_symlink_rejected(tmp_path):
    prep = preparation()
    destination = tmp_path / replay.ORIGINAL
    destination.mkdir(parents=True)
    source = replay.ROOT / replay.ORIGINAL
    for path in source.iterdir():
        shutil.copyfile(path, destination / path.name)
    assert replay.original_inventory(prep, root=tmp_path)['receipt.json']['sha256'] == \
        replay.ORIGINAL_RECEIPT_SHA
    console = destination / 'readout-console.txt'
    console.write_text('unexpected diagnostic\n')
    with pytest.raises(ValueError, match='size changed'):
        replay.original_inventory(prep, root=tmp_path)
    console.write_bytes(b'')
    deck = destination / 'specimen.feb'
    deck.write_bytes(deck.read_bytes() + b' ')
    with pytest.raises(ValueError, match='size changed'):
        replay.original_inventory(prep, root=tmp_path)
    deck.write_bytes((source / 'specimen.feb').read_bytes())
    same_size = bytearray(deck.read_bytes())
    same_size[0] ^= 1
    deck.write_bytes(same_size)
    with pytest.raises(ValueError, match='Exact original file changed'):
        replay.original_inventory(prep, root=tmp_path)
    deck.write_bytes((source / 'specimen.feb').read_bytes())
    console.unlink()
    console.symlink_to(destination / 'console.txt')
    with pytest.raises(ValueError):
        replay.original_inventory(prep, root=tmp_path)


def test_missing_release_and_worker_token_fail_before_replay_output(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.subprocess, 'Popen',
                        lambda *_a, **_k: pytest.fail('No process may launch'))
    with pytest.raises(FileNotFoundError):
        replay.execute(tmp_path / 'absent-release.json', root=tmp_path)
    bad_release = tmp_path / 'bad-release.json'
    bad_release.write_text('{"status":"not-released"}')
    with pytest.raises(ValueError, match='replay-only release'):
        replay.execute(bad_release, root=tmp_path)
    assert not (tmp_path / replay.OUTPUT).exists()
    output = tmp_path / replay.OUTPUT
    output.mkdir(parents=True)
    order = {'schema': 'hbe-v5-n12-saved-replay-work-order-v1',
             'run_id': prior.ORDER[1],
             'source_commit': 'a'*40,
             'original_receipt_sha256': replay.ORIGINAL_RECEIPT_SHA,
             'token_sha256': 'b'*64,
             'bindings': {}, 'adapter_receipt': {}}
    (output / 'replay-work-order.json').write_text(json.dumps(order))
    monkeypatch.delenv('HBE_V5_N12_REPLAY_TOKEN', raising=False)
    with pytest.raises(ValueError, match='token'):
        replay.replay_worker(replay.OUTPUT+'/replay-work-order.json', root=tmp_path)
    assert not (output / 'replay.json').exists()


def test_module_cli_missing_release_fails_before_reservation(tmp_path):
    output = replay.ROOT / replay.OUTPUT
    existed = output.exists()
    result = subprocess.run([sys.executable, '-B', '-m',
        'scripts.mechanics_hbe_v5_n12_saved_replay', '--execute', '--release',
        str(tmp_path / 'missing.json')], cwd=replay.ROOT, capture_output=True,
        text=True, timeout=15)
    assert result.returncode != 0
    assert output.exists() is existed


def test_original_release_git_blob_and_whitelist_reject_forgery():
    prep = preparation()
    release = json.loads((replay.ROOT / replay.ORIGINAL_RELEASE).read_text())
    head = replay.committed_source(replay.ORIGINAL_SOURCE_COMMIT,
        release['source_bindings'], paths=prior.SOURCE_PATHS)
    assert len(head) == 40
    forged = deepcopy(release['source_bindings'])
    name = prior.SOURCE_PATHS[0]
    forged[name]['sha256'] = '0'*64
    with pytest.raises(ValueError, match='immutable Git blob'):
        replay.committed_source(replay.ORIGINAL_SOURCE_COMMIT, forged,
            paths=prior.SOURCE_PATHS)
    wrong = deepcopy(prep)
    wrong['closed_attempt_files']['solver.log']['bytes'] = 0
    with pytest.raises(ValueError):
        replay.validate_preparation(wrong)


def test_replay_source_commit_allows_unrelated_postlaunch_head_only(tmp_path):
    subprocess.run(['/usr/bin/git', 'init', '-q'], cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'config', 'user.name', 'Fixture'],
                   cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'config', 'user.email', 'fixture@example.invalid'],
                   cwd=tmp_path, check=True)
    bindings = {}
    for relative in replay.SOURCE_PATHS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = relative.encode()
        path.write_bytes(data)
        bindings[relative] = {'path': relative, 'sha256': replay.io.sha(data)}
    subprocess.run(['/usr/bin/git', 'add', 'scripts'], cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'commit', '-qm', 'fixed replay source'],
                   cwd=tmp_path, check=True)
    commit = subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=tmp_path,
                            capture_output=True, text=True, check=True).stdout.strip()
    assert replay.committed_source(commit, bindings, paths=replay.SOURCE_PATHS,
        root=tmp_path, working=True, require_head=True) == commit
    (tmp_path / 'UNRELATED.md').write_text('unrelated\n')
    subprocess.run(['/usr/bin/git', 'add', 'UNRELATED.md'], cwd=tmp_path, check=True)
    subprocess.run(['/usr/bin/git', 'commit', '-qm', 'unrelated result note'],
                   cwd=tmp_path, check=True)
    assert replay.committed_source(commit, bindings, paths=replay.SOURCE_PATHS,
        root=tmp_path, working=True, require_head=False) != commit
    with pytest.raises(ValueError, match='launch HEAD'):
        replay.committed_source(commit, bindings, paths=replay.SOURCE_PATHS,
            root=tmp_path, working=True, require_head=True)
    (tmp_path / replay.SOURCE_PATHS[0]).write_text('changed executing source\n')
    with pytest.raises(ValueError, match='Bound file hash differs'):
        replay.committed_source(commit, bindings, paths=replay.SOURCE_PATHS,
            root=tmp_path, working=True, require_head=False)
