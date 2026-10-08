"""Cache/transport controls using authenticated source bytes, no patient fixtures."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('vitaldb_cache_preparation', ROOT / 'scripts/prepare_vitaldb_cache.py')
cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache
SPEC.loader.exec_module(cache)


@pytest.fixture
def isolated():
    parent = ROOT / 'build/vitaldb-cache-preparation-tests'
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as directory:
        root = Path(directory)
        references = list(cache.REFERENCES)
        component = json.loads((ROOT / 'manifests/vitaldb-native-component-v1.json').read_text())
        references.append(component['native_header_evidence']['path'])
        for relative in references:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        yield root


def copy_assets(root, assets=(cache.PARSER, cache.UTILS, cache.RECORDING)):
    for asset in assets:
        destination = root / asset.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / asset.relative_path, destination)


def no_network(*args, **kwargs):
    raise AssertionError('Network must not be called')


def test_actual_current_cache_is_ready_offline_without_transport(monkeypatch):
    monkeypatch.setattr(cache.subprocess, 'run', no_network)
    result = cache.prepare(ROOT, offline=True, transport=no_network)
    assert result['status'] == 'ready'
    assert result['network_attempts'] == result['downloaded_bytes'] == 0
    assert result['assets'] == {'parser': 'verified_cached', 'utils': 'verified_cached', 'recording': 'verified_cached'}
    assert result['new_people'] == result['training_examples'] == result['RL_transitions'] == 0


def test_fresh_local_cache_prepares_expected_paths_with_real_bytes(isolated):
    calls = []
    def transport(asset, partial, headers):
        calls.append(asset)
        shutil.copyfile(ROOT / asset.relative_path, partial)
        headers.write_text('HTTP/1.1 200 OK\r\n')
        return {'exit_code': 0, 'http_status': '200', 'stderr': ''}
    result = cache.prepare(isolated, transport=transport)
    assert result['status'] == 'ready'
    assert calls == [cache.PARSER, cache.RECORDING]
    assert result['downloaded_bytes'] == 6605435
    assert result['assets'] == {'parser': 'downloaded_verified', 'utils': 'extracted_verified', 'recording': 'downloaded_verified'}
    for asset in (cache.PARSER, cache.UTILS, cache.RECORDING):
        assert hashlib.sha256((isolated / asset.relative_path).read_bytes()).hexdigest() == asset.sha256
    assert not (isolated / 'build/vitaldb-source-v1/upstream-LICENSE.txt').exists()
    assert not (isolated / 'build/vitaldb-source-v1/vitaldb').exists()
    again = cache.prepare(isolated, offline=True, transport=no_network)
    assert again['status'] == 'ready' and again['network_attempts'] == 0


def test_offline_can_extract_missing_utils_only_from_authentic_local_archive(isolated):
    copy_assets(isolated, (cache.PARSER, cache.RECORDING))
    result = cache.prepare(isolated, offline=True, transport=no_network)
    assert result['status'] == 'ready'
    assert result['assets']['utils'] == 'extracted_verified'
    assert result['network_attempts'] == result['downloaded_bytes'] == 0


def test_offline_missing_payload_refuses_without_reserving_attempt(isolated):
    result = cache.prepare(isolated, offline=True, transport=no_network)
    assert result['status'] == 'refused'
    assert result['network_attempts'] == 0
    assert not list((isolated / cache.STATE).glob('*-attempt.json'))


@pytest.mark.parametrize('asset', [cache.PARSER, cache.UTILS, cache.RECORDING])
def test_existing_corruption_refuses_before_any_download_and_never_repairs(isolated, asset):
    copy_assets(isolated, (asset,))
    destination = isolated / asset.relative_path
    original = bytearray(destination.read_bytes())
    original[-1] ^= 1  # File-integrity control; no altered scientific data is admitted.
    destination.write_bytes(original)
    before = destination.read_bytes()
    result = cache.prepare(isolated, transport=no_network)
    assert result['status'] == 'refused' and 'mismatch' in result['error']
    assert result['network_attempts'] == 0
    assert destination.read_bytes() == before


def test_committed_rights_mismatch_blocks_payload_access(isolated):
    rights = isolated / 'artifacts/vitaldb-recorded-component-v1/PhysioNet-CC-BY-4.0.txt'
    rights.write_bytes(rights.read_bytes() + b'\n')
    result = cache.prepare(isolated, transport=no_network)
    assert result['status'] == 'refused' and result['network_attempts'] == 0


@pytest.mark.parametrize('failure', ['timeout', 'redirect', 'wrong_hash'])
def test_failed_transfer_retains_marker_partial_receipt_and_cannot_retry(isolated, failure):
    calls = []
    def transport(asset, partial, headers):
        calls.append(asset.name)
        if failure == 'timeout':
            partial.write_bytes((ROOT / asset.relative_path).read_bytes()[:10])
            raise subprocess.TimeoutExpired('curl', asset.seconds + 5)
        if failure == 'redirect':
            return {'exit_code': 0, 'http_status': '302', 'stderr': ''}
        data = bytearray((ROOT / asset.relative_path).read_bytes())
        data[-1] ^= 1
        partial.write_bytes(data)
        return {'exit_code': 0, 'http_status': '200', 'stderr': ''}
    result = cache.prepare(isolated, transport=transport)
    assert result['status'] == 'refused' and calls == ['parser']
    state = isolated / cache.STATE
    assert (state / 'parser-attempt.json').exists()
    receipt = json.loads((state / 'parser-receipt.json').read_text())
    assert receipt['status'] == 'failed'
    assert (state / 'parser.partial').exists()
    assert not (isolated / cache.PARSER.relative_path).exists()
    again = cache.prepare(isolated, transport=no_network)
    assert again['status'] == 'refused' and again['network_attempts'] == 0
    assert 'Prior local parser attempt' in again['error']


def test_publication_never_overwrites_racing_destination(isolated):
    def transport(asset, partial, headers):
        shutil.copyfile(ROOT / asset.relative_path, partial)
        (isolated / asset.relative_path).write_bytes(b'preserve concurrent file')
        return {'exit_code': 0, 'http_status': '200', 'stderr': ''}
    result = cache.prepare(isolated, transport=transport)
    assert result['status'] == 'refused'
    assert (isolated / cache.PARSER.relative_path).read_bytes() == b'preserve concurrent file'
    assert (isolated / cache.STATE / 'parser.partial').exists()


def test_curl_enforces_config_redirect_retry_size_and_time_limits(monkeypatch, isolated):
    captured = []
    def run(argv, **kwargs):
        captured.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='200', stderr='')
    monkeypatch.setattr(cache.subprocess, 'run', run)
    for asset in (cache.PARSER, cache.RECORDING):
        cache._curl(asset, isolated / 'partial', isolated / 'headers')
    for (argv, kwargs), asset in zip(captured, (cache.PARSER, cache.RECORDING)):
        assert argv[:3] == ['curl', '--disable', '--no-location']
        assert '--location' not in argv
        assert argv[argv.index('--retry') + 1] == '0'
        assert argv[argv.index('--proto') + 1] == '=https'
        assert argv[argv.index('--max-filesize') + 1] == str(asset.size)
        assert argv[argv.index('--max-time') + 1] == str(asset.seconds)
        assert kwargs['timeout'] == asset.seconds + 5
        assert argv[-1] == asset.url


def test_symlink_cache_and_escape_paths_refuse(isolated):
    destination = isolated / cache.PARSER.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(ROOT / cache.PARSER.relative_path)
    result = cache.prepare(isolated, transport=no_network)
    assert result['status'] == 'refused' and result['network_attempts'] == 0
    with pytest.raises(cache.CacheError, match='escapes checkout'):
        cache._contained(isolated, '../outside')


def test_fifo_cache_refuses_without_blocking_or_network(isolated):
    import os
    destination = isolated / cache.PARSER.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(destination)
    result = cache.prepare(isolated, transport=no_network)
    assert result['status'] == 'refused' and 'Nonregular' in result['error']
    assert result['network_attempts'] == 0


def test_orphan_transport_state_refuses_without_counting_network(isolated):
    state = isolated / cache.STATE
    state.mkdir(parents=True)
    (state / 'parser.headers').write_text('interrupted before receipt')
    result = cache.prepare(isolated, transport=no_network)
    assert result['status'] == 'refused' and result['network_attempts'] == 0
