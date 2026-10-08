"""Offline cache/receipt controls using the 24 actually downloaded source bodies."""
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL_STAGE = ROOT / 'build/resect-exact-mirror-acquisition-v1'
ARCHIVE = ROOT / 'artifacts/resect-exact-mirror-acquisition-v1'


@pytest.fixture
def importer(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module = importlib.import_module('resect_exact_mirror_import')
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('network forbidden'))
    return module


@pytest.fixture
def evidence():
    base = ARCHIVE if ARCHIVE.exists() else REAL_STAGE
    if not (base / 'summary.json').exists():
        pytest.skip('frozen actual mirror acquisition evidence unavailable')
    return base


@pytest.fixture
def isolated(importer, evidence, monkeypatch):
    if not (REAL_STAGE / 'objects').exists():
        pytest.skip('actual downloaded mask bodies unavailable')
    with tempfile.TemporaryDirectory(prefix='resect-mirror-import-controls-', dir=ROOT / 'build') as folder:
        folder = Path(folder)
        data, stage = folder / 'canonical', folder / 'stage'
        (data / 'rights').mkdir(parents=True)
        shutil.copyfile(ROOT / 'artifacts/resect-component-admission-v1/rights-review-02/README.txt', data / 'rights/README.txt')
        declaration = json.loads((evidence / 'declaration.json').read_bytes())
        for row in declaration['rows']:
            relative = Path('objects') / row['source_id'] / Path(row['mirror_path']).name
            (stage / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REAL_STAGE / relative, stage / relative)
        monkeypatch.setattr(importer, 'DATA', data)
        yield folder, stage


def prepare(importer, evidence, stage):
    return importer.prepare(evidence, stage, deadline=time.monotonic()+20)


def test_exact_portable_source_evidence_and_no_network(importer, evidence, isolated):
    folder, stage = isolated
    entries, binding = prepare(importer, evidence, stage)
    assert len(entries) == 24 and sum(e['source_authority']['bytes'] for e in entries) == 1561749
    assert binding['authoritative_manifest_sha256'] == importer.authority.MANIFEST_SHA
    assert binding['rights']['license'] == 'CC-BY-NC-SA-4.0'
    assert all(e['mirror_source']['role'] == 'TRAIN' for e in entries)
    assert not any(e['source_id'] in {'Case3-during-mask', 'Case11-during-mask', 'Case15-during-mask'} for e in entries)
    results = importer.preflight(entries, deadline=time.monotonic()+20)
    assert len(results) == 24 and not any(r['existing_target_verified'] for r in results)
    assert not (importer.DATA / 'originals').exists()


@pytest.mark.parametrize('field,value', [('role','SELECT'), ('patient_group','RESECT:Case4'),
    ('mirror_commit','0'*40), ('sha256','0'*64), ('expected_bytes',0),
    ('mirror_url','https://example.com/mask.nii.gz')])
def test_role_source_version_size_and_route_mutation_refusals(importer, evidence, field, value):
    row = json.loads((evidence / 'declaration.json').read_bytes())['rows'][0]
    manifest = importer.authority.require_manifest()
    source = next(s for s in manifest['sources'] if s['id'] == row['source_id'])
    pair = next(p for p in manifest['pairs'] if p['id'] == source['pair_id'])
    row[field] = value
    with pytest.raises(importer.Refusal, match='authority_role_or_version'):
        importer.validate_row(row, source, pair, manifest['release_by_kind']['cavity_annotation'])


def test_authority_payload_relabel_refused(importer, evidence):
    row = json.loads((evidence / 'declaration.json').read_bytes())['rows'][0]
    manifest = importer.authority.require_manifest()
    source = next(s for s in manifest['sources'] if s['id'] == row['source_id'])
    pair = next(p for p in manifest['pairs'] if p['id'] == source['pair_id'])
    row['source_authority']['file_revision'] = 1
    with pytest.raises(importer.Refusal):
        importer.validate_row(row, source, pair, manifest['release_by_kind']['cavity_annotation'])


@pytest.mark.parametrize('member', ['declaration.json', 'summary.json', 'source-snapshot.py',
    'objects/Case2-during-mask/attempt-01/receipt.json',
    'objects/Case2-during-mask/attempt-01/response-02.json'])
def test_frozen_evidence_mutation_refused_before_publication(importer, evidence, isolated, monkeypatch, member):
    folder, stage = isolated
    original = importer.read_small
    changed = evidence / member
    monkeypatch.setattr(importer, 'read_small', lambda p, **k: original(p, **k)+b' ' if p == changed else original(p, **k))
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'failed' and result['verified_files'] == 0
    assert not (importer.DATA / 'originals').exists()
    assert (ROOT / result['attempt'] / 'summary.json').exists()


def test_changed_canonical_rights_refuses_all_publication(importer, evidence, isolated):
    folder, stage = isolated
    path = importer.DATA / 'rights/README.txt'
    path.write_bytes(path.read_bytes()+b' ')
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'failed' and result['verified_files'] == 0
    assert not (importer.DATA / 'originals').exists()


def test_corrupt_last_staged_body_is_detected_before_first_install(importer, evidence, isolated):
    folder, stage = isolated
    entries, _ = prepare(importer, evidence, stage)
    bad = ROOT / entries[-1]['staged_path']
    raw = bad.read_bytes()
    bad.write_bytes(bytes([raw[0]^1])+raw[1:])
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'failed' and result['verified_files'] == 0
    assert len(result['unresolved_source_ids']) == 24
    assert not (importer.DATA / 'originals').exists()


def test_all_actual_bodies_import_and_idempotent_existing_verification(importer, evidence, isolated):
    folder, stage = isolated
    legacy = importer.DATA / 'train-v1/attempts/retained-control.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{"status":"transport_attempts_exhausted"}\n')
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'all_24_mirror_bodies_imported' and result['verified_files'] == 24
    assert len(result['files']) == 24 and all(r['status'] == 'installed_mirror_bytes_verified' for r in result['files'])
    assert result['network_requests'] == 0 and result['header_qc'] == 'not_run' and not result['training_admitted']
    entries, _ = prepare(importer, evidence, stage)
    inodes = {}
    for e in entries:
        staged, target = ROOT/e['staged_path'], ROOT/e['target_path']
        assert staged.read_bytes() == target.read_bytes()
        assert staged.stat().st_ino != target.stat().st_ino
        inodes[e['source_id']] = target.stat().st_ino
    again = importer.import_all(evidence, stage)
    assert again['status'] == 'all_24_mirror_bodies_imported'
    assert all(r['status'] == 'existing_canonical_bytes_verified' and not r['canonical_published'] for r in again['files'])
    assert inodes == {e['source_id']:(ROOT/e['target_path']).stat().st_ino for e in entries}
    assert legacy.read_text() == '{"status":"transport_attempts_exhausted"}\n'


def test_existing_different_target_never_overwritten(importer, evidence, isolated):
    folder, stage = isolated
    entries, _ = prepare(importer, evidence, stage)
    target = ROOT / entries[0]['target_path']
    target.parent.mkdir(parents=True)
    raw = (ROOT/entries[0]['staged_path']).read_bytes()
    changed = bytes([raw[0]^1])+raw[1:]
    target.write_bytes(changed)
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'failed' and result['verified_files'] == 0
    assert target.read_bytes() == changed
    assert len(list((importer.DATA/'originals').iterdir())) == 1


def test_held_cache_lock_blocks_before_attempt_or_target_creation(importer, evidence, isolated):
    folder, stage = isolated
    with (importer.DATA/'pilot.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            importer.import_all(evidence, stage)
    assert not (importer.DATA/'originals').exists()
    assert not (importer.DATA/'mirror-import-v1').exists()


@pytest.mark.parametrize('failure', ['disk', 'interrupt'])
def test_partial_failure_preserves_installed_file_partial_and_receipts(importer, evidence, isolated, monkeypatch, failure):
    folder, stage = isolated
    entries, _ = prepare(importer, evidence, stage)
    original = importer.copy_verified
    second = entries[1]['source_id']
    def interrupted(staged, partial, source, **kwargs):
        if source['id'] != second:
            return original(staged, partial, source, **kwargs)
        with staged.open('rb') as handle:
            partial.write_bytes(handle.read(16))
        if failure == 'interrupt':
            raise KeyboardInterrupt()
        raise OSError('private diagnostic')
    monkeypatch.setattr(importer, 'copy_verified', interrupted)
    result = importer.import_all(evidence, stage)
    assert result['status'] == ('interrupted' if failure == 'interrupt' else 'failed')
    assert result['verified_files'] == 1 and len(result['files']) == 2
    assert (ROOT/entries[0]['target_path']).exists() and not (ROOT/entries[1]['target_path']).exists()
    failed = result['files'][1]
    receipt = json.loads((ROOT/failed['receipt']).read_bytes())
    assert receipt['retained_partial_bytes'] == 16 and not receipt['canonical_published']
    assert importer.digest((ROOT/failed['receipt']).read_bytes()) == failed['receipt_sha256']
    assert 'private diagnostic' not in (ROOT/failed['receipt']).read_text()
    with monkeypatch.context() as context:
        context.setattr(importer, 'copy_verified', original)
        resumed = importer.import_all(evidence, stage)
    assert resumed['status'] == 'all_24_mirror_bodies_imported'


def test_final_bookkeeping_deadline_cannot_report_success(importer, evidence, isolated, monkeypatch):
    folder, stage = isolated
    clock, collect = time.monotonic, importer.collect_receipts
    offset = [0]
    monkeypatch.setattr(importer.time, 'monotonic', lambda: clock()+offset[0])
    def delayed(*args):
        result = collect(*args)
        offset[0] = importer.MAX_SECONDS+1
        return result
    monkeypatch.setattr(importer, 'collect_receipts', delayed)
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'timeout' and result['deadline_exceeded']
    assert result['verified_files'] == 24 and len(result['files']) == 24
    assert (ROOT/result['attempt']/'summary.json').exists()


def test_staged_symlink_refused_without_install(importer, evidence, isolated):
    folder, stage = isolated
    entries, _ = prepare(importer, evidence, stage)
    path = ROOT/entries[0]['staged_path']
    path.unlink()
    original = REAL_STAGE/'objects'/entries[0]['source_id']/Path(entries[0]['source_authority']['path']).name
    path.symlink_to(original)
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'failed' and not (importer.DATA/'originals').exists()


def test_final_receipt_source_mutation_is_not_accepted(importer, evidence, isolated, monkeypatch):
    folder, stage = isolated
    original = importer.collect_receipts
    def changed(trial, entries):
        path = trial/'files'/entries[0]['source_id']/'receipt.json'
        value = json.loads(path.read_bytes())
        value['source_authority']['file_revision'] = 1
        path.write_bytes(importer.encode(value))
        return original(trial, entries)
    monkeypatch.setattr(importer, 'collect_receipts', changed)
    result = importer.import_all(evidence, stage)
    assert result['status'] == 'incomplete' and result['verified_files'] == 23
    assert result['files'][0]['status'] == 'receipt_binding_failed'
    assert len(result['unresolved_source_ids']) == 1


def test_relocated_metadata_archive_needs_no_original_ignored_metadata(importer, evidence, isolated, monkeypatch):
    folder, stage = isolated
    archive = folder/'portable-artifact'
    shutil.copytree(evidence, archive, ignore=shutil.ignore_patterns('*.nii.gz', '__pycache__'))
    read = importer.read_small
    def archived_only(path, **kwargs):
        assert not path.is_relative_to(ROOT/'build/resect-osf-bottleneck-investigation-v1')
        assert not path.is_relative_to(REAL_STAGE)
        return read(path, **kwargs)
    monkeypatch.setattr(importer, 'read_small', archived_only)
    entries, bindings = prepare(importer, archive, stage)
    assert len(importer.preflight(entries, deadline=time.monotonic()+20)) == 24
    assert bindings['evidence_root'] == str(archive.relative_to(ROOT))
