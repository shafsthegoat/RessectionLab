#!/usr/bin/env python3
"""Exact-file, noncommercial RESECT TRAIN acquisition; no decoding or fitting.

One declared pair per supervised invocation. Existing pilot code and receipts
remain unchanged. Preflight/status are offline; acquire requires explicit scope.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urljoin, urlsplit
from urllib.request import Request, build_opener

import acquire_resect_cavity as pilot
from real_intake_io import atomic_preserve, check_deadline, supervise, termination_cleanup

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/annotations/resect-seg-v1'
MANIFEST = ROOT / 'manifests/resect-train-cavity-acquisition-v1.json'
MANIFEST_SHA = '09a15ffe52af966c27d53fba4924a7c67fba5590b12e53cb48aa022ef60f04f1'
CODE = ('scripts/resect_train_intake.py', 'scripts/acquire_resect_cavity.py', 'scripts/real_intake_io.py')
HELPER_PINS = {'scripts/acquire_resect_cavity.py': '6ece649ba346c9e7a0e93af462a91a9bf863728bea3fba83f5eccd681e1d0d51',
               'scripts/real_intake_io.py': '496821360b1daee79459952f970d299856baacad75fcc57ee589e55115036fd8'}
META_LIMIT = 2 * 1024**2
GOOD = {'downloaded_verified', 'existing_verified', 'recovered_verified'}
CLAIMS = {'training_admitted': False, 'spatial_planning_admitted': False,
          'optimizer_updates': 0, 'recorded_rl_transitions': 0, 'decoded_array_bytes': 0}


class Refusal(ValueError):
    """Messages are fixed codes, never transport exception text or signed URLs."""


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def safe_path(path: Path) -> Path:
    path = path.absolute()
    try:
        parts = path.relative_to(ROOT.absolute()).parts
    except ValueError:
        raise Refusal('path_outside_checkout') from None
    if '..' in parts or '.' in parts:
        raise Refusal('path_not_canonical')
    current = ROOT.absolute()
    for part in parts:
        current /= part
        if current.is_symlink():
            raise Refusal('symlink_path_refused')
    return path


def stable_stat(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def claims_match(value):
    return all(type(value.get(k)) is type(v) and value[k] == v for k, v in CLAIMS.items())


def read_small(path: Path, *, deadline=None, cap=META_LIMIT) -> bytes:
    path = safe_path(path)
    check_deadline(deadline)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > cap:
            raise Refusal('metadata_size_or_type')
        value = handle.read(cap + 1)
        after = os.fstat(handle.fileno())
    check_deadline(deadline)
    if len(value) != before.st_size or len(value) > cap or stable_stat(before) != stable_stat(after) or stable_stat(safe_path(path).lstat()) != stable_stat(after):
        raise Refusal('metadata_changed')
    return value


def save(path: Path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_preserve(safe_path(path), encode(value))


@dataclass(frozen=True)
class Source:
    id: str
    pair_id: str
    kind: str
    path: str
    source_url: str
    bytes: int
    expected_md5: str
    sha256: str | None
    osf_file_id: str | None = None
    short_alias: str | None = None
    file_revision: int | None = None

    @property
    def binding(self):
        return digest(encode(self.__dict__))


def validate_records(manifest, qualification, cohort):
    roles = {m['patient_group']: m['role'] for m in cohort['members']}
    expected = {(p['patient_group'], p['phase']): p for p in qualification['qualified_pairs']}
    pairs = manifest['pairs']
    entries = {e['id']: e for e in manifest['sources']}
    if (len(pairs) != 25 or len(entries) != 50 or len(manifest['sources']) != 50
            or len({p['id'] for p in pairs}) != 25
            or sum(e['bytes'] for e in entries.values()) != 557790240):
        raise Refusal('source_cardinality_or_bytes')
    seen = set()
    for pair in pairs:
        key = (pair['patient_group'], pair['phase'])
        original = expected.get(key)
        if (original is None or roles.get(key[0]) != 'TRAIN' or pair['role'] != 'TRAIN'
                or pair['id'] != key[0].split(':')[1] + '-' + key[1] or len(pair['source_ids']) != 2):
            raise Refusal('pair_role_or_identity')
        for index, source_id in enumerate(pair['source_ids']):
            entry = entries[source_id]
            ref = original['files'][index]
            expected_kind = 'original_ultrasound' if index == 0 else 'cavity_annotation'
            if (source_id in seen or entry['pair_id'] != pair['id'] or entry['kind'] != expected_kind
                    or entry['path'] != ref['suggested_relative_cache_path']
                    or entry['source_url'] != ref['source_url'] or entry['bytes'] != ref['bytes']
                    or entry['expected_md5'] != ref['published_md5']
                    or entry['sha256'] != ref.get('published_sha256')
                    or (index == 1 and (entry['osf_file_id'] != ref['osf_file_id']
                                        or entry['file_revision'] != 2))):
                raise Refusal('source_metadata_binding')
            if not re.fullmatch(r'originals/Case\d+-US-(during|after)(-resection)?\.nii\.gz', entry['path']):
                raise Refusal('source_cache_path')
            seen.add(source_id)
        if pair['bytes'] != sum(entries[i]['bytes'] for i in pair['source_ids']):
            raise Refusal('pair_byte_bound')
    if (seen != set(entries) or {(p['patient_group'], p['phase']) for p in pairs} != set(expected)
            or {(p['patient_group'], p['phase']) for p in manifest['missing_annotations']}
               != {('RESECT:Case11', 'during'), ('RESECT:Case11', 'after'), ('RESECT:Case15', 'during')}
            or manifest['excluded_members'] != qualification['protected_members_not_descended_into']
            or manifest['training_admitted'] is not False or manifest['optimizer_updates'] != 0
            or manifest['recorded_rl_transitions'] != 0):
        raise Refusal('cohort_or_claim_binding')


def require_manifest(*, deadline=None):
    raw = read_small(MANIFEST, deadline=deadline)
    if digest(raw) != MANIFEST_SHA:
        raise Refusal('manifest_changed')
    manifest = json.loads(raw)
    retained = {}
    for entry in manifest['metadata_bindings']:
        value = read_small(ROOT / entry['path'], deadline=deadline)
        if len(value) != entry['bytes'] or digest(value) != entry['sha256']:
            raise Refusal('retained_metadata_changed')
        retained[entry['path']] = value
    validate_records(manifest, json.loads(retained[manifest['qualification']['path']]),
                     json.loads(retained[manifest['cohort']['path']]))
    return manifest


def resolve_pair(manifest, pair_id):
    for pair in manifest['pairs']:
        if pair['id'] == pair_id:
            return pair
    raise Refusal('pair_not_frozen_train')


def source_by_id(source_id, *, deadline=None):
    manifest = require_manifest(deadline=deadline)
    for entry in manifest['sources']:
        if entry['id'] == source_id:
            return Source(**entry)
    raise Refusal('source_not_frozen_train')


def verify_file(path: Path, entry, *, deadline, receipt_sha=None):
    """One same-handle bounded pass; never unbounded file_digest or repair."""
    size = entry['bytes'] if isinstance(entry, dict) else entry.bytes
    md5_expected = entry['expected_md5'] if isinstance(entry, dict) else entry.expected_md5
    sha_expected = entry.get('sha256') if isinstance(entry, dict) else entry.sha256
    path = safe_path(path)
    check_deadline(deadline)
    sha, md5, count = hashlib.sha256(), hashlib.md5(), 0
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size != size:
            raise Refusal('source_size_or_type')
        while count <= size:
            check_deadline(deadline)
            block = handle.read(min(1024**2, size + 1 - count))
            if not block:
                break
            count += len(block)
            sha.update(block)
            md5.update(block)
        after = os.fstat(handle.fileno())
    check_deadline(deadline)
    if count != size or stable_stat(before) != stable_stat(after) or stable_stat(safe_path(path).lstat()) != stable_stat(after):
        raise Refusal('source_changed_or_size')
    result = sha.hexdigest()
    if (md5.hexdigest() != md5_expected or (sha_expected and result != sha_expected)
            or (receipt_sha and result != receipt_sha)):
        raise Refusal('source_fixity_mismatch')
    return result


def checked_route(source: Source, url: str, *, initial=False, body=False):
    """Exact selected file authority; no hostname-wide scientific downloads."""
    try:
        if not url or len(url) > 16384 or any(not 33 <= ord(c) <= 126 for c in url):
            raise Refusal('route_encoding')
        parsed = urlsplit(url)
        if (parsed.scheme != 'https' or parsed.username or parsed.password or parsed.fragment
                or parsed.port not in (None, 443) or parsed.netloc != parsed.hostname):
            raise Refusal('route_authority')
        if source.kind == 'original_ultrasound':
            if url != source.source_url:
                raise Refusal('original_exact_url_required')
            return
        if initial:
            if url != source.source_url:
                raise Refusal('annotation_initial_revision_url_required')
            return
        osf = {source.source_url, f'https://osf.io/download/{source.osf_file_id}/?revision=2'}
        if url in osf:
            return
        if (parsed.hostname == 'storage.googleapis.com'
                and parsed.path == '/cos-osf-prod-files-de-1/' + source.sha256):
            return
        provider_path = f'/v1/resources/jv8bk/providers/osfstorage/{source.osf_file_id}'
        if parsed.hostname in {'files.osf.io', 'files.de-1.osf.io'} and parsed.path in {provider_path, provider_path + '/'}:
            selectors = [v for k, v in parse_qsl(parsed.query, keep_blank_values=True)
                         if k.lower() in {'version', 'revision'}]
            if len(selectors) > 1 or (selectors and selectors != ['2']) or (body and not selectors):
                raise Refusal('annotation_revision_selector')
            return
        raise Refusal('route_outside_selected_file')
    except (ValueError, TypeError) as error:
        if isinstance(error, Refusal):
            raise
        raise Refusal('malformed_route') from None


def response_contract(response, source, url):
    checked_route(source, url, body=True)
    lengths = response.headers.get_all('Content-Length', [])
    encodings = response.headers.get_all('Content-Encoding', [])
    if (response.status != 200 or response.geturl() != url or lengths != [str(source.bytes)]
            or (encodings and encodings != ['identity'])):
        raise Refusal('response_identity_length_encoding')


def publish(partial, source, deadline, receipt_sha=None):
    sha = verify_file(partial, source, deadline=deadline, receipt_sha=receipt_sha)
    target = safe_path(DATA / source.path)
    target.parent.mkdir(parents=True, exist_ok=True)
    check_deadline(deadline)
    os.link(partial, safe_path(target))
    directory = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    partial.unlink()
    return sha


def transfer_original(source_id, trial, *, deadline, events, receipt_sha=None):
    source = source_by_id(source_id, deadline=deadline)
    if source.kind != 'original_ultrasound':
        raise Refusal('original_kind_required')
    checked_route(source, source.source_url, initial=True)
    runtime = pilot.system_curl_binding(timeout=min(5., max(.001, deadline - time.monotonic())))
    if not runtime['available']:
        raise Refusal('reviewed_native_trust_unavailable')
    event = {'source_id': source.id, **pilot.safe_route(source.source_url), 'transport': runtime}
    events.append(event)
    save(trial / 'request-01.json', event)
    partial, headers_path = safe_path(trial / 'body.partial'), safe_path(trial / 'headers.local')
    try:
        with partial.open('xb') as output, headers_path.open('xb') as headers:
            check_deadline(deadline)
            remaining = deadline - time.monotonic()
            command = ['/usr/bin/curl', '-q', '--no-location', '--max-redirs', '0', '--silent', '--fail',
                       '--proto', '=https', '--tlsv1.2', '--retry', '0', '--connect-timeout', str(min(30., remaining)),
                       '--max-time', str(remaining), '--max-filesize', str(source.bytes), '--header', 'Accept-Encoding: identity',
                       '--dump-header', f'/dev/fd/{headers.fileno()}', '--output', '-', source.source_url]
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.DEVNULL,
                                       pass_fds=(headers.fileno(),), env=pilot.system_curl_environment())
            try:
                event['curl_exit_code'] = process.wait(timeout=max(.001, deadline - time.monotonic()))
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            output.flush()
            os.fsync(output.fileno())
        check_deadline(deadline)
        if event['curl_exit_code']:
            raise Refusal('native_transfer_failed_no_retry')
        status, headers = pilot.nird_response_headers(headers_path)
        event['status'] = status
        if (status != 200 or headers.get_all('Content-Length', []) != [str(source.bytes)]
                or headers.get_all('Content-Encoding', []) not in ([], ['identity'])):
            raise Refusal('native_response_contract')
        return publish(partial, source, deadline, receipt_sha)
    finally:
        save(trial / 'response-01.json', event)


def transfer_annotation(source_id, trial, *, deadline, events, receipt_sha=None):
    source = source_by_id(source_id, deadline=deadline)
    if source.kind != 'cavity_annotation':
        raise Refusal('annotation_kind_required')
    opener = build_opener(pilot.NoRedirect())
    url = source.source_url
    partial = safe_path(trial / 'body.partial')
    for number in range(1, 6):
        check_deadline(deadline)
        checked_route(source, url, initial=number == 1)
        event = {'source_id': source.id, **pilot.safe_route(url)}
        events.append(event)
        save(trial / f'request-{number:02d}.json', event)
        try:
            response = opener.open(Request(url, headers={'Accept-Encoding': 'identity'}),
                                   timeout=min(30., max(.001, deadline - time.monotonic())))
        except HTTPError as error:
            event['status'] = error.code
            retry = error.headers.get('Retry-After', '')
            event['retry_after_seconds'] = int(retry) if re.fullmatch(r'[0-9]{1,8}', retry) else None
            location = error.headers.get('Location')
            error.close()
            save(trial / f'response-{number:02d}.json', event)
            if error.code in (301, 302, 303, 307, 308) and location:
                # The next loop checks the target before any request; never log it.
                url = urljoin(url, location)
                continue
            raise Refusal('annotation_http_failed_no_retry') from None
        with response:
            event['status'] = response.status
            save(trial / f'response-{number:02d}.json', event)
            response_contract(response, source, url)
            count = 0
            with partial.open('xb') as output:
                while count <= source.bytes:
                    check_deadline(deadline)
                    block = response.read(min(65536, source.bytes + 1 - count))
                    if not block:
                        break
                    count += len(block)
                    if count > source.bytes:
                        raise Refusal('annotation_byte_cap')
                    output.write(block)
                output.flush()
                os.fsync(output.fileno())
        check_deadline(deadline)
        return publish(partial, source, deadline, receipt_sha)
    raise Refusal('annotation_request_cap')


def execution_source(manifest, *, deadline):
    if Path(pilot.__file__).resolve() != ROOT / CODE[1]:
        raise Refusal('helper_import_not_checkout')
    import real_intake_io
    if Path(real_intake_io.__file__).resolve() != ROOT / CODE[2]:
        raise Refusal('helper_import_not_checkout')
    files = {name: digest(read_small(ROOT / name, deadline=deadline)) for name in CODE}
    if any(files[name] != expected for name, expected in HELPER_PINS.items()):
        raise Refusal('reviewed_helper_changed')
    return {'files': files, 'python': sys.version, 'manifest_sha256': MANIFEST_SHA,
            'metadata': manifest['metadata_bindings'],
            'native_runtime': pilot.system_curl_binding(timeout=min(5., max(.001, deadline-time.monotonic())))}


def validate_execution(source, attempt, deadline):
    manifest = require_manifest(deadline=deadline)
    if execution_source(manifest, deadline=deadline) != source:
        raise Refusal('execution_binding_changed')
    for name, expected in source['files'].items():
        if digest(read_small(attempt / 'source-snapshot' / name, deadline=deadline)) != expected:
            raise Refusal('execution_snapshot_changed')
    if digest(read_small(attempt / 'manifest.json', deadline=deadline)) != MANIFEST_SHA:
        raise Refusal('declaration_snapshot_changed')
    return manifest


def run_path(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', run_id):
        raise Refusal('run_identity')
    return safe_path(DATA / 'train-v1/attempts' / run_id)


def receipt_record(path, source, run_id):
    raw = read_small(path)
    value = json.loads(raw)
    if (value.get('manifest_sha256') != MANIFEST_SHA or value.get('source_id') != source.id
            or value.get('source_binding') != source.binding or value.get('run_id') != run_id
            or not claims_match(value)):
        raise Refusal('prior_receipt_binding')
    return value, digest(raw)


def prior_state(source):
    directory = safe_path(DATA / 'train-v1/source-state' / source.id)
    if not directory.exists():
        return directory, 1, None
    markers = []
    for path in directory.iterdir():
        if len(markers) >= 100 or not re.fullmatch(r'\d{4}\.json', path.name):
            raise Refusal('source_history_bound_or_format')
        markers.append(path)
    if not markers:
        return directory, 1, None
    markers.sort()
    if [p.name for p in markers] != [f'{i:04d}.json' for i in range(1, len(markers)+1)]:
        raise Refusal('source_history_gap')
    last = json.loads(read_small(markers[-1]))
    if last.get('source_binding') != source.binding or last.get('manifest_sha256') != MANIFEST_SHA:
        raise Refusal('source_history_binding')
    prior = run_path(last['run_id'])
    record = prior / 'files' / source.id / 'receipt.json'
    value, sha = receipt_record(record, source, last['run_id']) if record.exists() else (None, None)
    return directory, len(markers)+1, {'run_id': last['run_id'], 'record': value, 'receipt_sha256': sha,
                                     'receipt_path': str(record.relative_to(ROOT)) if record.exists() else None}


def require_closed_retry(prior, retry_of):
    if retry_of != prior['run_id']:
        raise Refusal('explicit_retry_of_prior_attempt_required')
    closure = json.loads(read_small(run_path(retry_of) / 'supervision.json'))
    if (closure.get('manifest_sha256') != MANIFEST_SHA or closure.get('run_id') != retry_of
            or closure.get('status') == 'completed' or closure.get('status') not in
            {'worker_failed', 'timeout', 'setup_failed', 'worker_start_failed', 'interrupted_after_start'}):
        raise Refusal('prior_attempt_not_closed_failed')
    pid = closure.get('worker_pid')
    if pid is not None:
        for check in (os.kill, os.killpg):
            try:
                check(pid, 0)
            except ProcessLookupError:
                continue
            raise Refusal('prior_worker_or_group_still_exists')


def prepare_sources(manifest, pair, attempt, run_id, retry_of, existing_only, outcomes):
    prepared = []
    pilot_receipt = json.loads(read_small(ROOT / manifest['pilot_acquisition_receipt']['path']))
    pilot_files = {f['path']: f for f in pilot_receipt['files']}
    for source_id in pair['source_ids']:
        source = source_by_id(source_id)
        directory, sequence, prior = prior_state(source)
        target = safe_path(DATA / source.path)
        record = prior['record'] if prior else None
        successful = record and record['status'] in GOOD
        if prior and not (successful and target.exists()):
            require_closed_retry(prior, retry_of)
        if sequence > 100:
            raise Refusal('source_history_bound')
        entry = {'source_id': source.id, 'source_binding': source.binding, 'path': source.path,
                 'sequence': sequence, 'prior': {k: v for k, v in prior.items() if k != 'record'} if prior else None, 'marker_path': str((directory / f'{sequence:04d}.json').relative_to(ROOT)),
                 'receipt_sha': record['sha256'] if successful else None,
                 'mode': 'cache_verify' if target.exists() else 'missing_refuse' if existing_only else 'download'}
        if source.path in pilot_files and entry['receipt_sha'] is None:
            entry['receipt_sha'] = pilot_files[source.path]['sha256']
            entry['pilot_provenance'] = manifest['pilot_acquisition_receipt']
        if record is None and prior and target.exists():
            entry['recovered'] = True
        prepared.append(entry)
    if retry_of is not None and not any(p['prior'] and p['prior']['run_id'] == retry_of for p in prepared):
        raise Refusal('retry_not_for_this_pair')
    # Expose reserved-source links even if a marker write or worker launch fails.
    outcomes.extend(prepared)
    # Both links exist before worker launch, even if it fails before making a request.
    for entry in prepared:
        save(ROOT / entry['marker_path'], {'run_id': run_id, 'pair_id': pair['id'], 'source_id': entry['source_id'],
             'source_binding': entry['source_binding'], 'manifest_sha256': MANIFEST_SHA,
             'attempt': str(attempt.relative_to(ROOT)), 'mode': entry['mode']})
    return prepared


def acquire_source(entry, attempt, run_id, source_sha, deadline, existing_only):
    source = source_by_id(entry['source_id'], deadline=deadline)
    trial = safe_path(attempt / 'files' / source.id)
    trial.mkdir(parents=True, exist_ok=False)
    record = {'source_id': source.id, 'source_binding': source.binding, 'path': source.path,
              'run_id': run_id, 'manifest_sha256': MANIFEST_SHA, 'execution_source_sha256': source_sha,
              'expected_bytes': source.bytes, 'expected_md5': source.expected_md5,
              'published_sha256': source.sha256, 'events': [], 'status': 'failed',
              'prior': entry['prior'], 'pilot_provenance': entry.get('pilot_provenance'), **CLAIMS}
    started = time.monotonic()
    try:
        target = safe_path(DATA / source.path)
        if target.exists():
            record['sha256'] = verify_file(target, source, deadline=deadline, receipt_sha=entry['receipt_sha'])
            record['status'] = 'recovered_verified' if entry.get('recovered') else 'existing_verified'
        else:
            if existing_only or entry['mode'] != 'download':
                raise Refusal('missing_source_or_changed_cache')
            transfer = transfer_original if source.kind == 'original_ultrasound' else transfer_annotation
            record['sha256'] = transfer(source.id, trial, deadline=deadline, events=record['events'], receipt_sha=entry['receipt_sha'])
            record['status'] = 'downloaded_verified'
        record['verified_bytes'] = source.bytes
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__,
                      error_code=str(error) if isinstance(error, Refusal) else 'source_operation_failed')
    finally:
        partial = safe_path(trial / 'body.partial')
        record['retained_partial_bytes'] = partial.stat().st_size if partial.exists() else 0
        record['elapsed_seconds'] = time.monotonic() - started
        save(trial / 'receipt.json', record)
    return record


def worker(run_id, expected_intent_sha):
    attempt = run_path(run_id)
    intent_raw = read_small(attempt / 'intent.json')
    if digest(intent_raw) != expected_intent_sha:
        raise Refusal('worker_intent_binding')
    intent = json.loads(intent_raw)
    deadline = intent['worker_deadline']
    if (not math.isfinite(deadline) or not 0 < deadline-time.monotonic() <= 295
            or intent['supervisor_pid'] != os.getppid() or intent['run_id'] != run_id
            or intent['manifest_sha256'] != MANIFEST_SHA):
        raise Refusal('worker_scope_or_parent')
    with safe_path(attempt / 'worker-claim.json').open('xb') as claim:
        claim.write(encode({'run_id': run_id, 'worker_pid': os.getpid(), 'manifest_sha256': MANIFEST_SHA}))
    source = json.loads(read_small(attempt / 'source.json'))
    record = {'run_id': run_id, 'pair_id': intent['pair_id'], 'manifest_sha256': MANIFEST_SHA,
              'execution_source_sha256': digest(encode(source)), 'intent_sha256': expected_intent_sha,
              'status': 'failed', 'files': [], **CLAIMS}
    started = time.monotonic()
    watchdog = threading.Timer(max(.001, deadline-time.monotonic()), lambda: os._exit(124))
    watchdog.daemon = True
    watchdog.start()
    try:
        manifest = validate_execution(source, attempt, deadline)
        pair = resolve_pair(manifest, intent['pair_id'])
        if [e['source_id'] for e in intent['sources']] != pair['source_ids']:
            raise Refusal('worker_source_scope')
        record.update(patient_group=pair['patient_group'], role='TRAIN', cohort_sha256=manifest['cohort']['sha256'],
                      rights_sha256=manifest['rights_prerequisite']['sha256'])
        verify_file(DATA / manifest['rights_prerequisite']['path'], manifest['rights_prerequisite'], deadline=deadline)
        for entry in intent['sources']:
            check_deadline(deadline)
            current = source_by_id(entry['source_id'], deadline=deadline)
            marker = json.loads(read_small(ROOT / entry['marker_path']))
            if current.binding != entry['source_binding'] or marker['run_id'] != run_id or marker['source_binding'] != current.binding:
                raise Refusal('worker_source_marker')
            item = acquire_source(entry, attempt, run_id, record['execution_source_sha256'], deadline, intent['existing_only'])
            record['files'].append(item)
            if item['status'] not in GOOD:
                raise Refusal('source_failed_pair_stopped')
        validate_execution(source, attempt, deadline)
        record['status'] = 'completed'
    except BaseException as error:
        record.update(status='failed', error_type=type(error).__name__,
                      error_code=str(error) if isinstance(error, Refusal) else 'worker_failed')
    finally:
        watchdog.cancel()
        record['elapsed_seconds'] = time.monotonic() - started
        save(attempt / 'worker-result.json', record)
    return record


def retain_worker(report, attempt):
    path = safe_path(attempt / 'worker-result.json')
    if not path.exists():
        return
    raw = read_small(path)
    report['worker_receipt_sha256'] = digest(raw)
    value = json.loads(raw)
    if (value.get('manifest_sha256') != MANIFEST_SHA or value.get('run_id') != report['run_id']
            or value.get('pair_id') != report['pair_id']
            or value.get('intent_sha256') != report.get('intent_sha256')
            or value.get('execution_source_sha256') != report.get('execution_source_sha256')
            or not claims_match(value)):
        raise Refusal('worker_receipt_binding')
    report['worker_status'] = value['status']
    report['source_results'] = [{k: f.get(k) for k in ('source_id', 'status', 'sha256', 'verified_bytes', 'retained_partial_bytes')}
                                for f in value['files']]


def run_pair(pair_id, run_id, *, seconds=300, retry_of=None, existing_only=False):
    if isinstance(seconds, bool) or not math.isfinite(seconds) or not 0 < seconds <= 300:
        raise Refusal('invocation_time_bound')
    started = time.monotonic()
    deadline = started + seconds
    manifest = require_manifest(deadline=deadline)
    pair = resolve_pair(manifest, pair_id)
    attempt = run_path(run_id)
    if retry_of:
        run_path(retry_of)
    safe_path(DATA).mkdir(parents=True, exist_ok=True)
    with safe_path(DATA / 'pilot.lock').open('a+') as lock, termination_cleanup():
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt.mkdir(parents=True, exist_ok=False)
        report = {'run_id': run_id, 'pair_id': pair_id, 'patient_group': pair['patient_group'], 'role': 'TRAIN',
                  'manifest_sha256': MANIFEST_SHA, 'cohort_sha256': manifest['cohort']['sha256'],
                  'rights_sha256': manifest['rights_prerequisite']['sha256'], 'attempt': str(attempt.relative_to(ROOT)),
                  'max_seconds': seconds, 'max_pair_bytes': pair['bytes'], 'existing_only': existing_only,
                  'automatic_retries': 0, 'retry_of': retry_of, 'status': 'setup_failed', 'sources': [], **CLAIMS}
        try:
            save(attempt / 'declaration.json', report)
            source = execution_source(manifest, deadline=deadline)
            report['execution_source_sha256'] = digest(encode(source))
            for name, expected in source['files'].items():
                content = read_small(ROOT / name, deadline=deadline)
                if digest(content) != expected:
                    raise Refusal('source_changed_during_snapshot')
                atomic_preserve(safe_path(attempt / 'source-snapshot' / name), content)
            save(attempt / 'source.json', source)
            atomic_preserve(safe_path(attempt / 'manifest.json'), read_small(MANIFEST, deadline=deadline))
            prepare_sources(manifest, pair, attempt, run_id, retry_of, existing_only, report['sources'])
            intent = {'run_id': run_id, 'pair_id': pair_id, 'manifest_sha256': MANIFEST_SHA,
                      'supervisor_pid': os.getpid(), 'worker_deadline': min(deadline-.25, time.monotonic()+295),
                      'existing_only': existing_only, 'sources': report['sources']}
            save(attempt / 'intent.json', intent)
            report['intent_sha256'] = digest(encode(intent))
            check_deadline(deadline)
            report['status'] = 'worker_launch_pending'
            command = [sys.executable, str(ROOT / CODE[0]), 'worker', '--run-id', run_id, '--intent-sha', report['intent_sha256']]

            def on_start(pid):
                report.update(worker_pid=pid, status='worker_started')
                save(attempt / 'started.json', {'worker_pid': pid, 'run_id': run_id, 'worker_deadline': intent['worker_deadline']})

            status, code = supervise(command, attempt / 'worker.log', deadline=deadline, on_start=on_start)
            report.update(status=status, exit_code=code)
            retain_worker(report, attempt)
            if report['status'] == 'completed' and report.get('worker_status') != 'completed':
                report['status'] = 'worker_failed'
        except BaseException as error:
            report.update(status='interrupted_after_start' if 'worker_pid' in report else
                          'worker_start_failed' if report['status'] == 'worker_launch_pending' else 'setup_failed',
                          error_type=type(error).__name__, error_code=str(error) if isinstance(error, Refusal) else 'invocation_failed')
        finally:
            # Attempt/PID/source links survive failed launch and interrupted supervision.
            try:
                retain_worker(report, attempt)
            except BaseException as error:
                report.update(status='worker_failed', receipt_error_type=type(error).__name__,
                              receipt_error_code='worker_receipt_not_admitted')
            report['elapsed_seconds'] = time.monotonic() - started
            save(attempt / 'supervision.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'status', 'acquire', 'worker'), nargs='?', default='preflight')
    parser.add_argument('--pair')
    parser.add_argument('--run-id')
    parser.add_argument('--intent-sha')
    parser.add_argument('--max-seconds', type=float, default=300)
    parser.add_argument('--retry-of')
    parser.add_argument('--existing-only', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--manifest-sha')
    args = parser.parse_args(argv)
    try:
        if args.action == 'worker':
            result = worker(args.run_id, args.intent_sha)
        elif args.action in {'preflight', 'status'}:
            manifest = require_manifest()
            pairs = [resolve_pair(manifest, args.pair)] if args.pair else manifest['pairs']
            result = {'status': 'metadata_preflight_passed', 'manifest_sha256': MANIFEST_SHA,
                      'summary': manifest['summary'], **CLAIMS}
            if args.action == 'status':
                result['files'] = [{'source_id': e['id'], 'status': 'present_unverified' if safe_path(DATA/e['path']).exists() else 'absent'}
                                   for e in manifest['sources'] if e['pair_id'] in {p['id'] for p in pairs}]
        else:
            if not args.execute or args.manifest_sha != MANIFEST_SHA or not args.pair or not args.run_id:
                raise Refusal('acquire_requires_exact_manifest_pair_and_new_run_identity')
            result = run_pair(args.pair, args.run_id, seconds=args.max_seconds, retry_of=args.retry_of,
                              existing_only=args.existing_only)
    except BaseException as error:
        # HTTP/URL exceptions can include signed queries: never print traceback/text.
        result = {'status': 'refused', 'error_type': type(error).__name__,
                  'error_code': str(error) if isinstance(error, Refusal) else 'operation_failed', **CLAIMS}
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result['status'] in {'completed', 'metadata_preflight_passed'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
