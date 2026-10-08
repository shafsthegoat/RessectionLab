"""Independent metadata/process controls. No scientific fixture or payload is read."""
import base64
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import pickletools
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
intake = importlib.import_module('lausanne_annotation_intake')
io = importlib.import_module('real_intake_io')

@pytest.fixture(autouse=True)
def prohibit_payloads(monkeypatch):
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        if str(path).endswith(('.nii', '.nii.gz', '.nii.gz.partial')):
            pytest.fail('Scientific payload access prohibited')
        return original_open(path, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('Scientific decode or transfer prohibited')
    monkeypatch.setattr(Path, 'open', guarded)
    monkeypatch.setattr(intake, 'open_without_redirect', forbidden)
    monkeypatch.setattr(intake.gzip, 'open', forbidden)


def test_independent_all_source_request_proofs_and_inventory():
    manifest, sessions = intake.preflight()
    qualified = json.loads(intake.source_metadata(manifest['metadata_sources']['qualification']))
    inventory_path = REPO / 'build/lausanne-annotation-expansion-audit-v1/train-candidate-inventory.json'
    payload = inventory_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == qualified['input_sha256']
    inventory = json.loads(payload)
    assert [r['path'] for r in manifest['records']] == [r['path'] for r in inventory['candidates']]
    assert len({r['path'] for r in manifest['records']}) == 148
    prefix = 'https://raw.githubusercontent.com/OpenNeuroDatasets/ds003949/' + manifest['git_commit'] + '/'
    for row, proof, candidate in zip(manifest['records'], qualified['records'], inventory['candidates'], strict=True):
        requests = proof['requests']
        assert len(requests) == 2
        for request, suffix in zip(requests, (row['path'], row['path'].removesuffix('.nii.gz') + '.json'), strict=True):
            assert request['method'] == 'GET'
            assert request['url'] == prefix + suffix
            assert request['scientific_payload_request'] is False and request['automatic_retries'] == 0
            payload = Path(request['body']['path']).read_bytes()
            assert len(payload) == request['body']['bytes'] <= 2048
            assert hashlib.sha256(payload).hexdigest() == request['body']['sha256']
            if request['ok']:
                assert request['status'] == 200 and request['final_url'] == request['url']
        if row['status'] == 'metadata_qualified':
            assert row['pointer_git_blob_sha1'] == candidate['git_pointer_blob_sha1']
            assert row['file']['bytes'] == candidate['s3_listed_bytes']
            assert row['file']['expected_md5'] == candidate['s3_listed_etag']
    assert inventory['statistics']['missing_train_patient_mask_people'] == ['sub-482']


def test_independent_crosswalk_classification_without_pickle_execution():
    manifest, _ = intake.preflight()
    payload = intake.source_metadata(manifest['metadata_sources']['crosswalk'])
    entries = [value for op, value, _ in pickletools.genops(payload)
               if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE')]
    assert len(entries) == 38 and len(set(entries)) == 38
    by_person = {value.split('_')[0]: value for value in entries}
    for row in manifest['records']:
        source = by_person.get(row['subject'])
        expected = ('source_voxelwise_exact_crosswalk' if source == row['subject'] + '_' + row['session'] else
                    'source_voxelwise_person_date_mismatch' if source else 'manual_region_subtype_unresolved')
        assert row['source_crosswalk_entry'] == source and row['annotation_subtype_status'] == expected
        assert row['annotation_available_at'] is None and row['review_available_at'] is None


def test_interrupted_started_worker_is_not_reported_unattempted(tmp_path, monkeypatch):
    manifest, sessions = intake.preflight()
    row = next(r for r in manifest['records'] if r['status'] == 'metadata_qualified')
    entries = (manifest['cohort'], manifest['original_index'], *manifest['metadata_sources'].values())
    source_payloads = {entry['sha256']: intake.source_metadata(entry) for entry in entries}
    source = intake.execution_source()
    monkeypatch.setattr(intake, 'ROOT', tmp_path)
    monkeypatch.setattr(intake, 'DATA', tmp_path / 'data')
    monkeypatch.setattr(intake, 'CACHE', tmp_path / 'data/cache')
    monkeypatch.setattr(intake, 'preflight', lambda: (manifest, sessions))
    monkeypatch.setattr(intake, 'source_metadata', lambda entry: source_payloads[entry['sha256']])
    monkeypatch.setattr(intake, 'execution_source', lambda: source)
    monkeypatch.setattr(intake, 'retain_source', lambda run, source: io.atomic_preserve(run / 'source.json', intake.encoded(source)))
    monkeypatch.setattr(intake, 'validate_execution', lambda source, run: None)
    pids = []
    def interrupted_supervisor(command, log, *, deadline, on_start):
        def started(pid):
            pids.append(pid)
            on_start(pid)
            raise KeyboardInterrupt('independent pure-process interruption after child start')
        return io.supervise([sys.executable, '-B', '-c', 'import time; time.sleep(30)'],
                            log, deadline=time.monotonic() + 3, on_start=started)
    monkeypatch.setattr(intake, 'supervise', interrupted_supervisor)
    report = intake.batch('interruption-control', 10, row['file']['bytes'], only_mask=row['path'], existing_only=True)
    assert report['status'] == 'interrupted_or_failed'
    assert report['attempted_source_bytes'] == row['file']['bytes']
    assert len(pids) == 1
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(pids[0], 0)
    selected = next(r for r in report['outcomes'] if r['path'] == row['path'])
    assert selected['status'] != 'deferred_not_attempted', selected
    assert 'attempt' in selected
