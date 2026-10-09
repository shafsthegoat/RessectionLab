"""Acquire only the declared published TractoInferno TRAIN bytes; never decode them.

The transport remains acquire_public_case.acquire_file. This adapter adds source
version and gradient Git-blob checks, durable attempts, and queue storage budgets.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPSHandler, build_opener
import argparse
import fcntl
import hashlib
import http.client
import json
import os
import re
import shutil
import socket
import ssl
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
BASE_REL = Path('data/acquisition/tractoinferno-train-v1')
FILES = 7622
BYTES = 273788367039
PEOPLE = 198
RESERVE = 100 * 1024**3
OUTPUT_ALLOWANCE = 64 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(1024**2):
            h.update(b)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def load_contract(root, base, declaration, declared_sha):
    assert sha(declaration) == declared_sha
    d = json.loads(declaration.read_text())
    assert sha(Path(__file__)) == d['runner_sha256']
    for name, digest in d['source_files_sha256'].items():
        assert '/' not in name and sha(base / 'source' / name) == digest
    assert sha(root / d['canonical_cohort_path']) == d['canonical_cohort_sha256'] == d['source_files_sha256']['cohort.json']
    assert sha(root / d['scope_path']) == d['scope_sha256'] == d['source_files_sha256']['scope.json']
    cohort = json.loads((base / 'source/cohort.json').read_text())
    scope = json.loads((base / 'source/scope.json').read_text())
    assert scope['cohort_sha256'] == d['canonical_cohort_sha256']
    assert scope['all_payloads_unreviewed'] and not scope['labels_admitted'] and not scope['training_admitted']
    assert d['max_workers'] == 2 and d['max_transport_attempts_per_object_across_restarts'] == 3
    assert d['free_space_reserve_bytes'] == RESERVE and d['active_output_allowance_bytes'] == OUTPUT_ALLOWANCE
    members = cohort['members']
    roles = {m['subject']: m['role'] for m in members}
    assert len(roles) == len(members) == 284
    expected_roles = {'trainset': 'TRAIN', 'validset': 'SELECT', 'testset': 'MEASUREMENT_EVAL'}
    assert all(m['role'] == expected_roles[m['source_partition']] for m in members)
    train = {p for p, role in roles.items() if role == 'TRAIN'}
    assert len(train) == PEOPLE and sum(r == 'SELECT' for r in roles.values()) == 58 and sum(r == 'MEASUREMENT_EVAL' for r in roles.values()) == 28
    raw = json.loads((base / 'source/train-exact-object-manifest.json').read_text())
    assert raw['status'] == 'all_source_objects_bound' and not raw['failures']
    assert raw['source_commit'] == 'bf5b266e9964881b4ba3695a8374c29808f0c4da' and raw['source_tag'] == '1.1.1'
    entries = []
    for row in raw['objects']:
        parts = PurePosixPath(row['path']).parts
        assert len(parts) >= 5 and parts[:2] == ('derivatives', 'trainset') and parts[2] == row['subject'] and row['subject'] in train
        assert '..' not in parts and not row['path'].startswith('/')
        assert row['source_partition'] == 'trainset' and row['proposed_role'] == 'TRAIN'
        assert isinstance(row['bytes'], int) and row['bytes'] > 0 and row['bytes'] == row['provider_size']
        assert re.fullmatch('[0-9a-f]{32}', row['expected_md5']) and row['etag_opaque'] == '"' + row['expected_md5'] + '"'
        key = 'ds003900/' + row['path']
        url = 'https://s3.amazonaws.com/openneuro.org/' + quote(key, safe='/') + '?versionId=' + quote(row['s3_version_id'], safe='')
        assert row['source_url'] == url and row['s3_key'] == key
        vm = row['version_metadata']
        assert vm['Key'] == key and vm['VersionId'] == row['s3_version_id'] and vm['ETag'] == row['etag_opaque'] and int(vm['Size']) == row['bytes']
        if row['expected_git_blob_sha1'] is None:
            assert row['source_md5'] == row['expected_md5'] and row['version_selection'] == 'source_annex_md5_and_size_match'
        else:
            assert row['path'].endswith(('.bval', '.bvec')) and re.fullmatch('[0-9a-f]{40}', row['expected_git_blob_sha1'])
            assert row['expected_git_blob_sha1'] == row['git_blob_sha1']
        entries.append({'path': row['path'], 'subject': row['subject'], 'role': 'TRAIN', 'size_bytes': row['bytes'], 'bytes': row['bytes'], 'sha256': None,
                        'source_url': url, 'expected_md5': row['expected_md5'], 'etag_opaque': row['etag_opaque'], 's3_version_id': row['s3_version_id'], 'expected_git_blob_sha1': row['expected_git_blob_sha1']})
    assert len(entries) == len({e['path'] for e in entries}) == FILES and sum(e['bytes'] for e in entries) == BYTES
    assert {e['subject'] for e in entries} == train and sum(e['expected_git_blob_sha1'] is not None for e in entries) == 396
    assert scope['files'] == FILES and scope['people'] == PEOPLE and scope['exact_total_bytes'] == BYTES
    assert scope['object_contract']['manifest_sha256'] == d['source_files_sha256']['train-exact-object-manifest.json']
    assert d['total_source_files'] == FILES and d['total_source_bytes'] == BYTES
    return d, scope, entries


class RemainingBudget:
    """Conservatively count queued/inflight unwritten bytes, with no double charge
    for already completed entries. Inflight writes stay reserved until terminal.
    """
    def __init__(self, base, out, entries, checked_path, reserve, allowance):
        self.base, self.reserve, self.allowance = base, reserve, allowance
        self.lock = threading.Lock()
        self.reservations = {}
        for e in entries:
            p = checked_path(out, e['path'])
            partial = checked_path(out, e['path'] + '.partial')
            size = partial.stat().st_size if partial.exists() else 0
            if size > e['bytes']:
                raise ValueError('oversized_partial_requires_review:' + e['path'])
            self.reservations[e['path']] = 0 if p.exists() else e['bytes'] - size
        self.remaining = sum(self.reservations.values())

    def check(self):
        with self.lock:
            free = shutil.disk_usage(self.base).free
            return free - self.remaining - self.allowance >= self.reserve

    def terminal(self, entry):
        with self.lock:
            self.remaining -= self.reservations.pop(entry['path'])


def execute(root, base, d, entries, declared_sha):
    """Run the already-validated and released contract; tests use generated entries."""
    out = base / 'verified'
    with ExitStack() as locks:
        # All acquisition queue locks remain held through completion. Model work is
        # separately coordinated by root; this grants no arbitrary process exception.
        for relative in d['predecessor_queue_locks'] + [str(BASE_REL / 'queue.lock')]:
            lock = locks.enter_context((root / relative).open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path[:0] = [str(base / 'source')]
        import acquire_public_case as transport
        import acquire_btc_case as sourceguards
        import real_intake_io as io
        for module in [transport, sourceguards, io]:
            assert Path(module.__file__).resolve().parent == base / 'source'

        def verify(path, entry):
            sourceguards.verify_file(path, entry)
            if entry['expected_git_blob_sha1'] is not None:
                h = hashlib.sha1(('blob ' + str(entry['bytes']) + '\0').encode())
                with path.open('rb') as f:
                    while b := f.read(1024**2):
                        h.update(b)
                if h.hexdigest() != entry['expected_git_blob_sha1']:
                    raise transport.AcquisitionError('pinned_release_git_blob_mismatch')

        # This callback runs on .partial before the unchanged transport publishes it.
        transport.verify_file = verify
        ctx = ssl.create_default_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
        budget = RemainingBudget(base, out, entries, transport.checked_path, d['free_space_reserve_bytes'], d['active_output_allowance_bytes'])
        if not budget.check():
            raise RuntimeError('full_remaining_queue_plus_output_allowance_exceeds_storage_reserve')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]
        run = base / 'runs' / run_id
        run.mkdir(parents=True)
        out.mkdir(exist_ok=True)

        def save(path, value):
            io.atomic_preserve(path, (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode())

        def hashes(path, entry):
            h, m = hashlib.sha256(), hashlib.md5()
            with path.open('rb') as f:
                while b := f.read(1024**2):
                    h.update(b)
                    m.update(b)
            assert m.hexdigest() == entry['expected_md5'] and path.stat().st_size == entry['bytes']
            return {'bytes': entry['bytes'], 'sha256': h.hexdigest(), 'source_md5': m.hexdigest(), 'release_git_blob_sha1': entry['expected_git_blob_sha1']}

        provider_next, provider_lock = {}, threading.Lock()
        cooldowns = base / 'provider-cooldowns'
        cooldowns.mkdir(exist_ok=True)
        for prior in cooldowns.glob('*.json'):
            record = json.loads(prior.read_text())
            provider_next[record['host']] = max(provider_next.get(record['host'], 0), record['next_attempt_unix'])

        def honor_cooldown(host):
            while True:
                with provider_lock:
                    delay = provider_next.get(host, 0) - time.time()
                if delay <= 0:
                    return
                time.sleep(min(delay, 2))

        def cooldown(host, headers):
            value, delay = headers.get('Retry-After', ''), 60
            try:
                delay = max(1, int(value))
            except ValueError:
                try:
                    delay = max(1, parsedate_to_datetime(value).timestamp() - time.time())
                except Exception:
                    pass
            with provider_lock:
                provider_next[host] = max(provider_next.get(host, 0), time.time() + delay)
                next_time = provider_next[host]
            save(cooldowns / (uuid.uuid4().hex + '.json'), {'host': host, 'next_attempt_unix': next_time, 'retry_after': value, 'recorded_utc': utc(), 'run_id': run_id})

        def process(entry):
            try:
                return process_inner(entry)
            finally:
                budget.terminal(entry)

        def process_inner(e):
            keyhash = hashlib.sha256((e['source_url'] + '|' + str(e['bytes']) + '|' + e['expected_md5'] + '|' + str(e['expected_git_blob_sha1'])).encode()).hexdigest()
            journal = base / 'attempts' / keyhash[:2] / keyhash
            journal.mkdir(parents=True, exist_ok=True)
            p, partial = transport.checked_path(out, e['path']), transport.checked_path(out, e['path'] + '.partial')
            host = urlparse(e['source_url']).netloc
            if p.exists():
                try:
                    verify(p, e)
                    return {'path': e['path'], 'subject': e['subject'], 'status': 'existing_byte_verified', **hashes(p, e), 'http_requests': 0, 'finished_utc': utc()}
                except Exception as err:
                    return {'path': e['path'], 'status': 'existing_byte_integrity_failure', 'error_type': type(err).__name__, 'error': str(err)}
            previous = sorted(journal.glob('*-intent.json'))
            if any(not intent.with_name(intent.name.replace('-intent.json', '-result.json')).exists() for intent in previous):
                return {'path': e['path'], 'status': 'orphaned_attempt_requires_reconciliation_no_network', 'attempts_before_run': len(previous)}
            prior_results = [json.loads(p.read_text()) for p in journal.glob('*-result.json')]
            if any(r['status'] == 'integrity_or_scope_refusal' for r in prior_results):
                return {'path': e['path'], 'status': 'prior_integrity_or_scope_refusal_no_network', 'attempts_before_run': len(previous)}
            if len(previous) >= 3:
                return {'path': e['path'], 'status': 'transport_exhausted', 'attempts_before_run': len(previous)}
            for attempt in range(len(previous) + 1, 4):
                honor_cooldown(host)
                if not budget.check():
                    return {'path': e['path'], 'status': 'storage_reserve_refusal', 'http_requests': 0}
                aid = f'{attempt:02d}-{run_id}'
                offset = partial.stat().st_size if partial.exists() else 0
                save(journal / (aid + '-intent.json'), {'run_id': run_id, 'entry': e, 'started_utc': utc(), 'resume_offset': offset, 'declaration_sha256': declared_sha, 'tls_verified': True})
                events = []
                opener = build_opener(sourceguards.RejectRedirects(), HTTPSHandler(context=ctx))

                def observed(request, *, timeout):
                    assert request.full_url == e['source_url']
                    request.add_header('If-Match', e['etag_opaque'])
                    event = {'url': request.full_url, 'method': request.get_method(), 'range': request.get_header('Range'), 'requested_utc': utc(), 'tls_verified': True}
                    try:
                        r = opener.open(request, timeout=timeout)
                    except HTTPError as err:
                        event.update(status=err.code, retry_after=err.headers.get('Retry-After'))
                        events.append(event)
                        save(journal / (aid + '-http.json'), event)
                        if err.code in [429, 503]:
                            cooldown(host, err.headers)
                        raise
                    h = r.headers
                    event.update(status=r.status, effective_url=r.geturl(), headers={k: h.get(k) for k in ['Content-Length', 'Content-Range', 'Content-Encoding', 'ETag', 'x-amz-version-id']})
                    events.append(event)
                    save(journal / (aid + '-http.json'), event)
                    valid = r.geturl() == e['source_url'] and r.status == (206 if offset else 200) and h.get('ETag') == e['etag_opaque'] and h.get('x-amz-version-id') == e['s3_version_id'] and int(h.get('Content-Length', '-1')) == e['bytes'] - offset
                    if not valid:
                        r.close()
                        raise transport.AcquisitionError('source_identity_size_version_or_resume_contract_failed')
                    return r
                try:
                    state = transport.acquire_file(e, out, opener=observed)
                    result = {'path': e['path'], 'subject': e['subject'], 'status': 'byte_verified', 'transport_status': state, **hashes(p, e), 'attempt': attempt, 'resume_offset': offset, 'http_requests': len(events), 'finished_utc': utc()}
                    save(journal / (aid + '-result.json'), result)
                    return result
                except Exception as err:
                    transient = (isinstance(err, HTTPError) and err.code in [429, 500, 502, 503, 504]) or isinstance(err, (TimeoutError, socket.timeout, ConnectionResetError, http.client.IncompleteRead)) or (isinstance(err, URLError) and not isinstance(err.reason, ssl.SSLError))
                    truncated = isinstance(err, transport.AcquisitionError) and str(err).startswith('Pending source size/checksum contract mismatch:') and partial.exists() and partial.stat().st_size < e['bytes'] and bool(events) and events[-1].get('status') in [200, 206]
                    transient = transient or truncated
                    result = {'path': e['path'], 'subject': e['subject'], 'status': 'transport_deferred' if transient and attempt < 3 else 'transport_exhausted' if transient else 'integrity_or_scope_refusal', 'attempt': attempt, 'error_type': type(err).__name__, 'error': str(err)[:300], 'partial_bytes': partial.stat().st_size if partial.exists() else 0, 'http_requests': len(events), 'finished_utc': utc()}
                    save(journal / (aid + '-result.json'), result)
                    if not transient or attempt == 3:
                        return result
                    time.sleep(5 if attempt == 1 else 15)

        started = {'run_id': run_id, 'pid': os.getpid(), 'pgid': os.getpgrp(), 'started_utc': utc(), 'declaration_sha256': declared_sha, 'runner_sha256': sha(Path(__file__)), 'cohort_sha256': d['canonical_cohort_sha256'], 'scope_sha256': d['scope_sha256'], 'scope_commit': d['scope_commit'], 'people': len({e['subject'] for e in entries}), 'files': len(entries), 'bytes': sum(e['bytes'] for e in entries), 'workers': 2, 'remaining_bytes_reserved': budget.remaining, 'active_output_allowance_bytes': d['active_output_allowance_bytes'], 'storage_before': shutil.disk_usage(base)._asdict(), 'decoded_array_bytes': 0, 'training_admitted': False}
        save(run / 'started.json', started)
        print(json.dumps({'event': 'started', **started}), flush=True)
        results, iterator = [], iter(entries)
        with io.termination_cleanup(), ThreadPoolExecutor(max_workers=2) as executor:
            pending = {executor.submit(process, e) for _, e in zip(range(2), iterator)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    if len(results) % 100 == 0 or result['status'] not in ['byte_verified', 'existing_byte_verified']:
                        print(json.dumps({'event': 'progress', 'finished': len(results), 'total': len(entries), 'verified': sum(r['status'] in ['byte_verified', 'existing_byte_verified'] for r in results), 'verified_bytes': sum(r.get('bytes', 0) for r in results), 'latest_status': result['status'], 'remaining_bytes_reserved': budget.remaining}), flush=True)
                    try:
                        entry = next(iterator)
                    except StopIteration:
                        continue
                    pending.add(executor.submit(process, entry))
        good = [r for r in results if r['status'] in ['byte_verified', 'existing_byte_verified']]
        completion = {'status': 'all_bytes_verified' if len(good) == len(entries) else 'completed_with_unresolved_files', 'run_id': run_id, 'finished_utc': utc(), 'declaration_sha256': declared_sha, 'cohort_sha256': d['canonical_cohort_sha256'], 'verified_files': len(good), 'verified_bytes': sum(r['bytes'] for r in good), 'unresolved_files': len(entries) - len(good), 'planned_people': len({e['subject'] for e in entries}), 'role': 'TRAIN', 'source_partition': 'trainset', 'all_payloads_unreviewed': True, 'header_qc': 'not_run', 'anatomy_qc': 'not_run', 'labels_admitted': False, 'training_admitted': False, 'independent_evaluation_claim': False, 'decoded_array_bytes': 0, 'optimizer_updates': 0, 'storage_after': shutil.disk_usage(base)._asdict(), 'files': sorted(results, key=lambda r: r['path'])}
        save(run / 'completion.json', completion)
        if completion['status'] == 'all_bytes_verified' and not (base / 'completion.json').exists():
            save(base / 'completion.json', completion)
        print(json.dumps({'event': 'completed', 'status': completion['status'], 'verified_files': len(good), 'verified_bytes': completion['verified_bytes'], 'receipt': str(run / 'completion.json')}), flush=True)
        return completion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    base = ROOT / BASE_REL
    d, scope, entries = load_contract(ROOT, base, args.declaration, args.declaration_sha256)
    if args.check_only:
        print(json.dumps({'status': 'prepared_contract_valid', 'network_requests': 0, 'payload_access': False, 'files': FILES, 'bytes': BYTES, 'people': PEOPLE, 'execution_released': d['execution_released']}))
        return
    assert d['execution_released'] is True and d['scope_commit'] and scope['declaration_status'] == 'frozen_before_train_payload_access'
    assert scope['intake']['execution_released'] is True
    for prior in d['preceding_terminal_receipts']:
        path = ROOT / prior['path']
        assert sha(path) == prior['sha256'] and json.loads(path.read_text())['status'] == 'all_bytes_verified'
    execute(ROOT, base, d, entries, args.declaration_sha256)


if __name__ == '__main__':
    main()
