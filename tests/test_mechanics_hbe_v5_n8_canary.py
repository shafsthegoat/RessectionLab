"""Source-only N8 canary controls; no measured curves or native execution."""
from __future__ import annotations

from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess

import pytest

from scripts import mechanics_hbe_v5_n8_canary as canary


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def forbid_native_launch(monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *a, **k: pytest.fail('Native launch forbidden'))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('Subprocess forbidden'))
    original_open = io.open

    def no_measured_or_patient(path, *args, **kwargs):
        mode = args[0] if args else kwargs.get('mode', 'r')
        if any(flag in mode for flag in ('w', 'a', 'x', '+')):
            pytest.fail('Read-only verification wrote a file')
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if (resolved.is_relative_to(ROOT / 'data/mechanics')
                    or resolved.is_relative_to(ROOT / 'data/patients')):
                pytest.fail('Measured or patient data access forbidden')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(io, 'open', no_measured_or_patient)


def test_complete_read_only_source_preparation():
    result = canary.verify_preparation(ROOT)
    assert result == {
        'status': 'source_only_verified_not_execution_ready',
        'run_id': canary.RUN_ID,
        'frames_declared': 61,
        'adapted_deck_sha256': '3cf4156b19191918841498490bc448f88a61b656b566d5d6a3d36db817d0a724',
        'release': None,
    }
    with pytest.raises(ValueError, match='no one-shot supervisor'):
        canary.require_execution_ready()


def test_only_first_canary_source_and_mesh_are_opened(monkeypatch):
    original = canary.sources._read_bound
    opened = []

    def observed(root, binding, **kwargs):
        opened.append(binding['path'])
        return original(root, binding, **kwargs)

    monkeypatch.setattr(canary.sources, '_read_bound', observed)
    canary.expected_preparation(ROOT)
    old_decks = '/mesh-preparation/'
    assert [path for path in opened if old_decks in path and path.endswith('.feb')] == [
        'outputs/mechanics/hbe-01-03-poc-v1/mesh-preparation/N8/generated/'
        'compression-60-reference/specimen.feb']
    assert not any('/tension-' in path or '/N12/' in path for path in opened)


@pytest.mark.parametrize('defect', [
    'endpoint', 'schedule_point', 'release', 'runtime', 'mesh', 'source_hash',
    'accelerate_deck', 'bc_count',
])
def test_manifest_mutations_fail_closed(monkeypatch, defect):
    original = canary.sources._read_bound
    manifest = json.loads((ROOT / canary.MANIFEST_PATH).read_text())
    corrupt = deepcopy(manifest)
    if defect == 'endpoint':
        corrupt['schedule']['endpoint_float64_hex'] = '-0x1.8p-11'
    elif defect == 'schedule_point':
        corrupt['schedule']['full_coordinates_m'][20] *= 0.99
    elif defect == 'release':
        corrupt['release'] = {'approved': True}
    elif defect == 'runtime':
        corrupt['runtime_identity']['sha256'] = '0' * 64
    elif defect == 'mesh':
        corrupt['full_native_mesh']['sha256'] = '0' * 64
    elif defect == 'source_hash':
        corrupt['old_source_deck']['sha256'] = '0' * 64
    elif defect == 'accelerate_deck':
        corrupt['new_endpoint_decks']['new_endpoint_accelerate']['sha256'] = '0' * 64
    else:
        corrupt['topology_and_fixture_bc']['bc_count'] -= 1

    def changed(root, binding, **kwargs):
        if binding['path'] == canary.MANIFEST_PATH:
            return (json.dumps(corrupt) + '\n').encode()
        return original(root, binding, **kwargs)

    monkeypatch.setattr(canary.sources, '_read_bound', changed)
    with pytest.raises(ValueError, match='differs'):
        canary.verify_preparation(ROOT)


def test_pinned_manifest_hash_rejects_changed_bytes():
    with pytest.raises(ValueError, match='SHA256'):
        canary.sources._read_bound(
            ROOT, {'path': canary.MANIFEST_PATH, 'sha256': '0' * 64}, maximum=128 * 1024)


def test_old_source_content_drift_refused(monkeypatch):
    original = canary.sources._read_bound
    old_path = json.loads((ROOT / canary.MANIFEST_PATH).read_text())['old_source_deck']['path']

    def changed(root, binding, **kwargs):
        data = original(root, binding, **kwargs)
        return data + b'\n' if binding['path'] == old_path else data

    monkeypatch.setattr(canary.sources, '_read_bound', changed)
    with pytest.raises(ValueError, match='source'):
        canary.expected_preparation(ROOT)


def test_runtime_profile_drift_refused(monkeypatch):
    original = canary.backend.verify_profile

    def changed(root, binding):
        profile = original(root, binding)
        profile['runtime_identity'] = {'path': 'wrong', 'sha256': '0' * 64}
        return profile

    monkeypatch.setattr(canary.backend, 'verify_profile', changed)
    with pytest.raises(ValueError, match='runtime profile'):
        canary.expected_preparation(ROOT)


def test_materialized_deck_content_drift_refused(monkeypatch):
    original = canary.sources._read_bound
    path = json.loads((ROOT / canary.MANIFEST_PATH).read_text())[
        'new_endpoint_decks']['new_endpoint_accelerate']['path']

    def changed(root, binding, **kwargs):
        data = original(root, binding, **kwargs)
        return data + b'\n' if binding['path'] == path else data

    monkeypatch.setattr(canary.sources, '_read_bound', changed)
    with pytest.raises(ValueError, match='Materialized adapted deck'):
        canary.verify_preparation(ROOT)
