"""Stage three declared IXI archives as opaque mixed-role bytes; never decode.

The established acquire_public_case transfer and BTC source-MD5 verifier remain
unchanged. Raw Apache ETags are opaque identity, never interpreted as checksums.
Zenodo supplies the label MD5 but no ETag; size/MD5 still precede atomic publish.
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
BASE_REL = Path('data/acquisition/ixi-t1-mra-vessel-v1')
FILES = 3
BYTES = 17225109877
PEOPLE = 582
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
    for path, digest in d['metadata_files_sha256'].items():
        assert sha(root / path) == digest
    assert sha(root / d['canonical_cohort_path']) == d['canonical_cohort_sha256'] == d['source_files_sha256']['cohort.json']
    assert sha(root / d['scope_path']) == d['scope_sha256'] == d['source_files_sha256']['scope.json']
    cohort = json.loads((base / 'source/cohort.json').read_text())
    scope = json.loads((base / 'source/scope.json').read_text())
    assert scope['cohort_sha256'] == d['canonical_cohort_sha256']
    t = scope['transport']
    assert t['all_payloads_unreviewed'] and not t['labels_admitted'] and not t['training_admitted']
    assert t['opaque_mixed_roles'] and not t['member_extraction_released'] and not t['NIfTI_headers_or_arrays_released']
    assert d['max_workers'] == t['max_workers'] == 1 and d['max_transport_attempts_per_object_across_restarts'] == 3
    assert d['free_space_reserve_bytes'] == RESERVE and d['active_output_allowance_bytes'] == OUTPUT_ALLOWANCE
    assert cohort['people'] == len(cohort['members']) == PEOPLE
    assert len({m['subject'] for m in cohort['members']}) == PEOPLE
    assert cohort['roles_count'] == {'TRAIN': 407, 'SELECT': 88, 'MEASUREMENT_EVAL': 87}
    assert cohort['annotation_roles_count'] == {'TRAIN': 70, 'SELECT': 15, 'MEASUREMENT_EVAL': 15}
    assert not cohort['protected_roles_changed'] and not cohort['image_or_label_member_bodies_accessed_before_freeze']
    assert not cohort['extraction_or_decoding_released'] and not cohort['training_admitted']
    source = json.loads((base / 'source/source-contract.json').read_text())
    selected = [x for x in source['raw_archives'] if x['filename'] in ['IXI-T1.tar', 'IXI-MRA.tar']] + [source['annotation_archive']]
    assert [r['filename'] for r in scope['files']] == [r['filename'] for r in selected]
    entries = []
    for row, original in zip(scope['files'], selected):
        assert all(row[k] == original[k] for k in ['filename', 'url', 'bytes', 'expected_md5'])
        assert row['local_path'] == row['filename'] and '/' not in row['filename'] and '..' not in row['filename']
        assert row.get('etag_opaque') == original.get('etag_opaque')
        assert row['url'].startswith('https://') and re.fullmatch('[0-9a-f]{32}', row['expected_md5'])
        entries.append({'path': row['filename'], 'size_bytes': row['bytes'], 'bytes': row['bytes'], 'sha256': None,
                        'source_url': row['url'], 'expected_md5': row['expected_md5'], 'etag_opaque': row.get('etag_opaque'),
                        'checksum_authority': row['checksum_authority']})
    assert len(entries) == FILES and sum(e['bytes'] for e in entries) == BYTES
    assert scope['source_files'] == d['total_source_files'] == FILES
    assert scope['source_bytes'] == d['total_source_bytes'] == BYTES
    concurrent = d['concurrent_Tracto']
    assert concurrent['base_path'] == 'data/acquisition/tractoinferno-train-v1'
    assert concurrent['manifest_path'] == 'data/acquisition/tractoinferno-train-v1/source/train-exact-object-manifest.json'
    assert concurrent['manifest_sha256'] == '151441d5eb6fe923ad3b1f978edf49b04843956f3209e216a95340466003a846'
    assert sha(root / concurrent['manifest_path']) == concurrent['manifest_sha256']
    raw = json.loads((root / concurrent['manifest_path']).read_text())
    assert len(raw['objects']) == 7622 and sum(r['bytes'] for r in raw['objects']) == 273788367039
    return d, scope, entries


def concurrent_remaining(root, binding):
    """Stat only: reserve every declared Tracto byte not already on disk.

    This is disk accounting, not checksum verification or scientific admission.
    Read immutable metadata once per check; never open downloaded payloads.
    """
    path = root / binding['manifest_path']
    assert sha(path) == binding['manifest_sha256']
    raw = json.loads(path.read_text())
    out = root / binding['base_path'] / 'verified'
    remaining = 0
    for entry in raw['objects']:
        relative = PurePosixPath(entry['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        present = 0
        for suffix in ['', '.partial']:
            p = out / (entry['path'] + suffix)
            assert not p.is_symlink() and p.resolve().is_relative_to(out.resolve())
            try:
                present = max(present, min(p.stat().st_size, entry['bytes']))
            except FileNotFoundError:
                pass
        remaining += entry['bytes'] - present
    return remaining


class RemainingBudget:
    """Conservatively count queued/inflight unwritten bytes, with no double charge
    for already completed entries. Inflight writes stay reserved until terminal.
    """
    def __init__(self, base, out, entries, checked_path, reserve, allowance, concurrent):
        self.base, self.reserve, self.allowance = base, reserve, allowance
        self.concurrent = concurrent
        self.lock = threading.Lock()
        self.reservations = {}
        for e in entries:
            p = checked_path(out, e['path'])
            partial = checked_path(out, e['path'] + '.partial')
            assert not (out / e['path']).is_symlink() and not (out / (e['path'] + '.partial')).is_symlink()
            size = partial.stat().st_size if partial.exists() else 0
            if size > e['bytes']:
                raise ValueError('oversized_partial_requires_review:' + e['path'])
            self.reservations[e['path']] = 0 if p.exists() else e['bytes'] - size
        self.remaining = sum(self.reservations.values())

    def check(self):
        with self.lock:
            free = shutil.disk_usage(self.base).free
            return free - self.remaining - self.concurrent() - self.allowance >= self.reserve

    def terminal(self, entry):
        with self.lock:
            self.remaining -= self.reservations.pop(entry['path'])


def execute(root, base, d, entries, declared_sha):
    """Run the already-validated and released contract; tests use generated entries."""
    out = base / 'verified'
    with ExitStack() as locks:
        # All acquisition queue locks remain held through completion. Model work is
        # separately coordinated by root; this grants no arbitrary process exception.
        for relative in [str(BASE_REL / 'queue.lock')]:
            lock = locks.enter_context((root / relative).open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sys.path[:0] = [str(base / 'source')]
        import acquire_public_case as transport
        import acquire_btc_case as sourceguards
        import real_intake_io as io
        for module in [transport, sourceguards, io]:
            assert Path(module.__file__).resolve().parent == base / 'source'

        verify = sourceguards.verify_file

        # This callback runs on .partial before the unchanged transport publishes it.
        transport.verify_file = verify
        ctx = ssl.create_default_context()
        assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
        budget = RemainingBudget(base, out, entries, transport.checked_path, d['free_space_reserve_bytes'], d['active_output_allowance_bytes'], lambda: concurrent_remaining(root, d['concurrent_Tracto']))
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
            return {'bytes': entry['bytes'], 'sha256': h.hexdigest(), 'source_md5': m.hexdigest(), 'checksum_authority': entry['checksum_authority']}

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
            keyhash = hashlib.sha256((e['source_url'] + '|' + str(e['bytes']) + '|' + e['expected_md5']).encode()).hexdigest()
            journal = base / 'attempts' / keyhash[:2] / keyhash
            journal.mkdir(parents=True, exist_ok=True)
            p, partial = transport.checked_path(out, e['path']), transport.checked_path(out, e['path'] + '.partial')
            host = urlparse(e['source_url']).netloc
            if p.exists():
                try:
                    verify(p, e)
                    return {'path': e['path'], 'status': 'existing_byte_verified', **hashes(p, e), 'http_requests': 0, 'finished_utc': utc()}
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
            opener_calls_this_run = 0
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
                    nonlocal opener_calls_this_run
                    assert request.full_url == e['source_url']
                    if e['etag_opaque'] is not None:
                        request.add_header('If-Match', e['etag_opaque'])
                    event = {'url': request.full_url, 'method': request.get_method(), 'range': request.get_header('Range'), 'requested_utc': utc(), 'tls_verified': True}
                    opener_calls_this_run += 1
                    try:
                        r = opener.open(request, timeout=timeout)
                    except HTTPError as err:
                        event.update(status=err.code, retry_after=err.headers.get('Retry-After'))
                        events.append(event)
                        save(journal / (aid + '-http.json'), event)
                        if err.code in [429, 503]:
                            cooldown(host, err.headers)
                        raise
                    except Exception as err:
                        event.update(error_type=type(err).__name__, error=str(err)[:300])
                        events.append(event)
                        save(journal / (aid + '-http.json'), event)
                        raise
                    h = r.headers
                    event.update(status=r.status, effective_url=r.geturl(), headers={k: h.get(k) for k in ['Content-Length', 'Content-Range', 'Content-Encoding', 'ETag']})
                    events.append(event)
                    save(journal / (aid + '-http.json'), event)
                    valid = r.geturl() == e['source_url'] and r.status == (206 if offset else 200) and (e['etag_opaque'] is None or h.get('ETag') == e['etag_opaque']) and int(h.get('Content-Length', '-1')) == e['bytes'] - offset
                    if not valid:
                        r.close()
                        raise transport.AcquisitionError('source_identity_size_or_resume_contract_failed')
                    return r
                try:
                    state = transport.acquire_file(e, out, opener=observed)
                    result = {'path': e['path'], 'status': 'byte_verified', 'transport_status': state, **hashes(p, e), 'attempt': attempt, 'resume_offset': offset, 'http_requests': len(events), 'opener_calls_this_run': opener_calls_this_run, 'finished_utc': utc()}
                    save(journal / (aid + '-result.json'), result)
                    return result
                except Exception as err:
                    transient = (isinstance(err, HTTPError) and err.code in [429, 500, 502, 503, 504]) or isinstance(err, (TimeoutError, socket.timeout, ConnectionResetError, http.client.IncompleteRead)) or (isinstance(err, URLError) and not isinstance(err.reason, ssl.SSLError))
                    truncated = isinstance(err, transport.AcquisitionError) and str(err).startswith('Pending source size/checksum contract mismatch:') and partial.exists() and partial.stat().st_size < e['bytes'] and bool(events) and events[-1].get('status') in [200, 206]
                    transient = transient or truncated
                    result = {'path': e['path'], 'status': 'transport_deferred' if transient and attempt < 3 else 'transport_exhausted' if transient else 'integrity_or_scope_refusal', 'attempt': attempt, 'error_type': type(err).__name__, 'error': str(err)[:300], 'partial_bytes': partial.stat().st_size if partial.exists() else 0, 'http_requests': len(events), 'opener_calls_this_run': opener_calls_this_run, 'finished_utc': utc()}
                    save(journal / (aid + '-result.json'), result)
                    if not transient or attempt == 3:
                        return result
                    time.sleep(5 if attempt == 1 else 15)

        started = {'run_id': run_id, 'pid': os.getpid(), 'pgid': os.getpgrp(), 'started_utc': utc(), 'declaration_sha256': declared_sha, 'runner_sha256': sha(Path(__file__)), 'cohort_sha256': d['canonical_cohort_sha256'], 'scope_sha256': d['scope_sha256'], 'scope_commit': d['scope_commit'], 'people': PEOPLE, 'files': len(entries), 'bytes': sum(e['bytes'] for e in entries), 'workers': 1, 'remaining_bytes_reserved': budget.remaining, 'active_output_allowance_bytes': d['active_output_allowance_bytes'], 'storage_before': shutil.disk_usage(base)._asdict(), 'decoded_array_bytes': 0, 'training_admitted': False}
        save(run / 'started.json', started)
        print(json.dumps({'event': 'started', **started}), flush=True)
        results, iterator = [], iter(entries)
        with io.termination_cleanup(), ThreadPoolExecutor(max_workers=1) as executor:
            pending = {executor.submit(process, e) for _, e in zip(range(1), iterator)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    result = future.result()
                    results.append(result)
                    if True:
                        print(json.dumps({'event': 'progress', 'finished': len(results), 'total': len(entries), 'verified': sum(r['status'] in ['byte_verified', 'existing_byte_verified'] for r in results), 'verified_bytes': sum(r.get('bytes', 0) for r in results), 'latest_status': result['status'], 'remaining_bytes_reserved': budget.remaining}), flush=True)
                    try:
                        entry = next(iterator)
                    except StopIteration:
                        continue
                    pending.add(executor.submit(process, entry))
        good = [r for r in results if r['status'] in ['byte_verified', 'existing_byte_verified']]
        completion = {'status': 'all_bytes_verified' if len(good) == len(entries) else 'completed_with_unresolved_files', 'run_id': run_id, 'finished_utc': utc(), 'declaration_sha256': declared_sha, 'cohort_sha256': d['canonical_cohort_sha256'], 'verified_files': len(good), 'verified_bytes': sum(r['bytes'] for r in good), 'unresolved_files': len(entries) - len(good), 'planned_people': PEOPLE, 'roles': {'TRAIN': 407, 'SELECT': 88, 'MEASUREMENT_EVAL': 87}, 'transport_role': 'opaque_mixed_role_archives_no_extraction', 'all_payloads_unreviewed': True, 'header_qc': 'not_run', 'anatomy_qc': 'not_run', 'labels_admitted': False, 'training_admitted': False, 'independent_evaluation_claim': False, 'decoded_array_bytes': 0, 'optimizer_updates': 0, 'storage_after': shutil.disk_usage(base)._asdict(), 'files': sorted(results, key=lambda r: r['path'])}
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
    assert d['execution_released'] is True and d['scope_commit'] and scope['declaration_status'] == 'frozen_before_archive_body_intake'
    assert scope['transport']['execution_released'] is True
    assert d['scope_frozen_before_payload_access'] is True
    execute(ROOT, base, d, entries, args.declaration_sha256)


if __name__ == '__main__':
    main()
