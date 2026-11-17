"""Portable path bindings and explicit historical/current integrity scopes."""
import importlib.util
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / 'scripts/rehearsal_support'


def load_verifier():
    spec = importlib.util.spec_from_file_location('maintained_rehearsal_verifier', ROOT / 'scripts/verify_shared_rehearsal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_current_data_scope_keeps_data_hashes_strict_and_historical_source_scope_explicit(tmp_path, monkeypatch):
    verifier = load_verifier()
    original = b'original historical data'
    data = tmp_path / 'testcases/outputs/full-case-rehearsal-20261020/sample.json'
    data.parent.mkdir(parents=True)
    data.write_bytes(original)
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': [
        {'path': str(data.relative_to(tmp_path)), 'bytes': len(original), 'sha256': hashlib.sha256(original).hexdigest()},
        {'path': 'README.md', 'bytes': 0, 'sha256': hashlib.sha256(b'').hexdigest()},
    ]}))
    monkeypatch.setattr(verifier, 'MANIFEST', manifest)
    assert verifier.check_files(tmp_path, data_only=True)[1] == len(original)
    with pytest.raises(ValueError, match='File differs: README.md'):
        verifier.check_files(tmp_path)
    data.write_bytes(b'changed data')
    with pytest.raises(ValueError, match='File differs: testcases/'):
        verifier.check_files(tmp_path, data_only=True)


def test_supported_imports_and_clis_do_not_need_historical_review(tmp_path):
    environment = __import__('os').environ | {'RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT': str(tmp_path)}
    code = f"""
import sys
sys.path[:0] = [{str(SUPPORT)!r}, {str(ROOT / 'desktop/electron')!r}, {str(ROOT / 'src')!r}]
from readonly_step_monitor import ReadOnlyStepMonitor
import accept_native, export_observed_telemetry, verify_every_cached_frame_monitoring
import prefix_playback_monitor, vascular_playback_monitor
from bundle_paths import data_root
assert accept_native.SOURCE == data_root() / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
assert accept_native.PROTECTION == data_root() / 'testcases/environments/meningioma-sub-06/rl-protection.json'
assert vascular_playback_monitor.NATIVE.is_relative_to(data_root())
assert not any('full-case-rehearsal-20261020/review' in p for p in sys.path)
"""
    subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=environment, check=True, capture_output=True)
    for name in ('accept_native.py', 'export_observed_telemetry.py', 'verify_every_cached_frame_monitoring.py'):
        result = subprocess.run([sys.executable, str(SUPPORT / name), '--help'], cwd=ROOT, env=environment,
                                check=True, capture_output=True, text=True)
        assert '--bundle-root' in result.stdout


def test_acceptance_api_resolves_default_source_from_selected_root_without_global_mutation(tmp_path, monkeypatch):
    sys.path.insert(0, str(SUPPORT))
    import accept_native
    original_protection = accept_native.PROTECTION
    seen = []

    def stop_before_loading(path):
        seen.append(path)
        raise RuntimeError('path binding observed before any reader or physics')

    monkeypatch.setattr(accept_native, 'file_hash', stop_before_loading)
    with pytest.raises(RuntimeError, match='path binding observed'):
        accept_native.verify(tmp_path / 'episode.ressectionlab', 'full-primary', bundle_root=tmp_path)
    assert seen == [tmp_path / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab']
    assert accept_native.PROTECTION == original_protection


def test_missing_optional_bundle_has_restoration_guidance(tmp_path):
    verifier = load_verifier()
    with pytest.raises(FileNotFoundError, match='Restore the rehearsal archive.*RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT'):
        verifier.check_files(tmp_path, data_only=True)


def test_launcher_selects_external_root_and_preserves_secondary_display_default(tmp_path):
    import os
    import shutil
    checkout = tmp_path / 'source checkout'
    scripts = checkout / 'scripts'
    scripts.mkdir(parents=True)
    shutil.copy2(ROOT / 'scripts/open_shared_rehearsal.sh', scripts)
    launch = scripts / 'launch_poc.sh'
    launch.write_text('#!/bin/sh\nprintf "%s\\n" "$RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT" "$@"\n')
    launch.chmod(0o755)
    bundle = tmp_path / 'restored bundle'
    native = bundle / 'testcases/outputs/full-case-rehearsal-20261020/final/rehearsal.ressectionlab'
    native.parent.mkdir(parents=True)
    native.write_bytes(b'launcher fixture; no application is launched')
    environment = os.environ | {'RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT': str(tmp_path / 'missing root')}
    result = subprocess.run([str(scripts / 'open_shared_rehearsal.sh'), '--bundle-root', str(bundle)],
                            env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(bundle), '--secondary-display', '--case', str(native)]
    environment['RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT'] = str(bundle)
    result = subprocess.run([str(scripts / 'open_shared_rehearsal.sh'), '--allow-primary-display'],
                            env=environment, capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == [str(bundle), '--case', str(native)]
    result = subprocess.run([str(scripts / 'open_shared_rehearsal.sh'), '--bundle-root', str(tmp_path)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 1
    assert 'Restore the archive' in result.stderr and 'Clone the codex' not in result.stderr


def test_profiler_input_reader_uses_bound_paths_and_rejects_changed_native(tmp_path, monkeypatch):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import profile_surgical_replay_prefix as profiler
    import resectionlab.workspace_bundle as workspace
    native = tmp_path / 'native.ressectionlab'
    native.write_bytes(b'original fixture')
    trace = tmp_path / 'actions.jsonl'
    actions = [{'operation': 'fixture', 'ordinal': i} for i in range(3)]
    trace.write_text(''.join(json.dumps({'action': action}) + '\n' for action in actions))
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps({'playbackIpc': [{'frameIndex': i, 'stateId': str(i)} for i in range(12)]}))
    monkeypatch.setattr(profiler, 'NATIVE', native)
    monkeypatch.setattr(profiler, 'TRACE', trace)
    monkeypatch.setattr(profiler, 'NATIVE_SHA', hashlib.sha256(native.read_bytes()).hexdigest())
    seen = []
    def load(path):
        seen.append(path)
        return None, {'surgical_toolbox': {'branches': {'full-primary': {'actions': actions}}}}, None
    monkeypatch.setattr(workspace, 'load_workspace', load)
    expected, original = profiler.load_inputs(baseline)
    assert len(expected) == 12 and len(original) == 3 and seen == [native]
    native.write_bytes(b'changed fixture')
    with pytest.raises(AssertionError, match='Preserved episode bytes differ'):
        profiler.load_inputs(baseline)
    assert seen == [native]


def test_explicit_cli_root_overrides_unavailable_environment_default(tmp_path):
    import os
    environment = os.environ | {'RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT': str(tmp_path / 'unavailable')}
    for script, extra in [(SUPPORT / 'accept_native.py', []),
                          (ROOT / 'desktop/electron/vascular_playback_monitor.py', ['check-inputs'])]:
        result = subprocess.run([sys.executable, str(script), *extra, '--bundle-root', str(tmp_path), '--help'],
                                cwd=ROOT, env=environment, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert '--bundle-root' in result.stdout
    # Imports used by generated workflows must not require optional input data.
    result = subprocess.run([sys.executable, '-c',
                            f"import sys; sys.path[:0] = [{str(SUPPORT)!r}, {str(ROOT / 'desktop/electron')!r}]; import accept_native, vascular_playback_monitor"],
                            cwd=ROOT, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
