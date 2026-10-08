#!/usr/bin/env python3
"""Continuous, exact-file TRAIN downloads; byte verification is not image QC.

Run: .venv/bin/python scripts/continuous_source_acquisition.py run
Inspect without network or source-body reads: ... status
Verify existing bytes only, with no network: ... run --offline

One supervised network worker at a time; no batch approval boundaries. Immutable
attempt journals survive restarts. Deferred attempts respect provider cooldowns;
three failed transport attempts per object exhaust automatic retries. Rerunning
does not reset that bound. Integrity/TLS/scope failures never retry automatically.
Lausanne supports validated Range resume; RESECT resumes the queue and reuses
complete files only. Its old partial bodies remain untouched.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import ExitStack
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import ssl
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import build_opener
import uuid

import acquire_btc_case as annex
import lausanne_train_intake as lausanne
import resect_train_intake as resect
from real_intake_io import IntakeDeadline, atomic_preserve, check_deadline, supervise, termination_cleanup

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/acquisition/continuous-source-v1'
INDEX_SHA = '6f1fc7812af0d66550076aa08d37d7f36f08d764fdab701bbbc9ca0608629e66'
CODE = ('scripts/continuous_source_acquisition.py', 'scripts/resect_train_intake.py',
        'scripts/lausanne_train_intake.py', 'scripts/acquire_btc_case.py',
        'scripts/acquire_public_case.py', 'scripts/acquire_lausanne_pilot.py',
        'scripts/acquire_resect_cavity.py', 'scripts/real_intake_io.py')
GOOD = {'byte_verified', 'existing_byte_verified'}
DEFERRED = {'rate_deferred', 'transport_deferred'}
MAX_ATTEMPTS = 3
CLAIMS = {'header_qc': 'not_run', 'geometry_qc': 'not_run', 'anatomy_qc': 'not_run',
          'training_admitted': False, 'spatial_planning_admitted': False,
          'decoded_array_bytes': 0, 'optimizer_updates': 0, 'recorded_rl_transitions': 0}
Refusal = resect.Refusal
encode, digest, read_small, safe_path = resect.encode, resect.digest, resect.read_small, resect.safe_path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def new_id():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8]


def save(path, value):
    path = safe_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_preserve(path, encode(value))


def read_json(path):
    return json.loads(read_small(path))


def source_binding(source):
    return digest(encode(source))


def target_path(source):
    base = lausanne.DATA if source['dataset'] == 'lausanne' else resect.DATA
    return safe_path(base / source['entry']['path'])


def catalog():
    """Resolve only retained frozen metadata; no transport or image inspection."""
    if digest(read_small(lausanne.INDEX)) != INDEX_SHA:
        raise Refusal('lausanne_index_changed')
    index, manifest = lausanne.validated_index(), resect.require_manifest()
    pairs = {p['id']: p for p in manifest['pairs']}
    sources = []
    for row in index['sessions']:
        for entry in row['files']:
            sources.append({'key': 'lausanne-' + digest(entry['path'].encode())[:24],
                'dataset': 'lausanne', 'role': 'TRAIN', 'person': row['subject'],
                'session': row['session'], 'provider': 'embedded' if 'metadata_base64' in entry else 'openneuro_s3',
                'release': {'git_commit': index['git_commit'], 'license': index['license']},
                'manifest_sha256': INDEX_SHA, 'entry': entry})
    for entry in manifest['sources']:
        pair = pairs[entry['pair_id']]
        sources.append({'key': 'resect-' + entry['id'], 'dataset': 'resect',
            'role': pair['role'], 'person': pair['patient_group'], 'pair_id': pair['id'],
            'provider': 'nird' if entry['kind'] == 'original_ultrasound' else 'osf',
            'release': manifest['release_by_kind'][entry['kind']],
            'manifest_sha256': resect.MANIFEST_SHA, 'entry': entry})
    if len(sources) != 890 or len({s['key'] for s in sources}) != 890:
        raise Refusal('catalog_cardinality')
    return sources


def runtime_binding():
    files = {name: digest(read_small(ROOT / name)) for name in CODE}
    if any(files.get(name) != sha for name, sha in resect.HELPER_PINS.items()):
        raise Refusal('reviewed_helper_changed')
    # Imported code must be the retained checkout code, not a shadow module.
    for name in CODE[1:]:
        module = sys.modules[Path(name).stem]
        if Path(module.__file__).resolve() != ROOT / name:
            raise Refusal('import_outside_checkout')
    return {'files': files, 'python': sys.version, 'executable': sys.executable,
            'manifests': {str(lausanne.INDEX.relative_to(ROOT)): INDEX_SHA,
                          str(resect.MANIFEST.relative_to(ROOT)): resect.MANIFEST_SHA}}


def freeze_run(sources, offline):
    run = safe_path(CACHE / 'runs' / new_id())
    run.mkdir(parents=True, exist_ok=False)
    binding = runtime_binding()
    for name, sha in {**binding['files'], **binding['manifests']}.items():
        raw = read_small(ROOT / name)
        if digest(raw) != sha:
            raise Refusal('source_changed_during_snapshot')
        atomic_preserve(safe_path(run / 'snapshot' / name), raw)
    declaration = {'schema': 'continuous-source-acquisition-v1', 'created_utc': utc_now(),
        'execution': binding, 'catalog_sha256': digest(encode(sources)),
        'files': len(sources), 'bytes': sum(s['entry']['bytes'] for s in sources),
        'offline': offline, 'workers': 1, 'automatic_attempt_limit_per_object': MAX_ATTEMPTS,
        'worker_seconds': {'lausanne': 595, 'resect': 295},
        'supervisor_seconds': {'lausanne': 600, 'resect': 300},
        'stream_chunk_bytes': 1048576, 'minimum_free_disk_reserve_bytes': 536870912,
        'rate_limit_minimum_cooldown_seconds': 900,
        'range_resume': {'lausanne': True, 'resect': False},
        'resect_rights': resect.require_manifest()['rights_and_semantics'], **CLAIMS}
    save(run / 'declaration.json', declaration)
    return run, declaration


def retry_after(value, now):
    """Return an absolute UTC epoch, never shorten a server's requested delay."""
    if not isinstance(value, str) or len(value) > 128:
        return None
    value = value.strip()
    if re.fullmatch(r'[0-9]{1,12}', value):
        return now + int(value)
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return max(now, parsed.timestamp())
    except (TypeError, ValueError, OverflowError):
        return None


class ObservedOpener:
    """Observe sanitized status/timing only; underlying transport guards remain."""
    def __init__(self, open_function, trial, deadline):
        self.open_function, self.trial, self.deadline = open_function, trial, deadline
        self.number, self.failure = 0, None

    def __call__(self, request, *, timeout):
        check_deadline(self.deadline)
        self.number += 1
        try:
            response = self.open_function(request, timeout=min(timeout, max(.001, self.deadline-time.monotonic())))
        except HTTPError as error:
            now = time.time()
            event = {'status': error.code, 'observed_utc_epoch': now,
                     'retry_after_utc_epoch': retry_after(error.headers.get('Retry-After'), now)}
            save(self.trial / f'http-{self.number:02d}.json', event)
            if error.code not in (301, 302, 303, 307, 308):
                self.failure = event
            raise
        # Preserve an existing partial if the server refuses Range. The annex
        # helper otherwise restarts by truncating it; this runner never does.
        if request.get_header('Range') and response.status != 206:
            response.close()
            raise Refusal('range_resume_not_honored_partial_preserved')
        return response

    def open(self, request, *, timeout):
        return self(request, timeout=timeout)


def verified_bytes(source, *, deadline):
    entry, target = source['entry'], target_path(source)
    if 'metadata_base64' in entry:
        raw = read_small(target, deadline=deadline, cap=65536)
        if len(raw) != entry['bytes'] or digest(raw) != entry['sha256']:
            raise Refusal('sidecar_fixity_mismatch')
        return digest(raw)
    return resect.verify_file(target, entry, deadline=deadline)


def native_failure(trial):
    path = trial / 'headers.local'
    if not path.exists():
        return None
    try:
        status, headers = resect.pilot.nird_response_headers(path)
        if status >= 400:
            now = time.time()
            return {'status': status, 'observed_utc_epoch': now,
                    'retry_after_utc_epoch': retry_after(headers.get('Retry-After'), now)}
    except Exception:
        pass
    return None


def failure_kind(error, http):
    if http:
        if http['status'] == 429:
            return 'rate_deferred'
        if http['status'] in (408, 425, 500, 502, 503, 504):
            return 'transport_deferred'
        return 'transport_failed'
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, (ssl.SSLError, ssl.CertificateError)):
        return 'tls_failed'
    if isinstance(error, (IntakeDeadline, TimeoutError, socket.timeout)):
        return 'transport_deferred'
    if isinstance(error, URLError) and isinstance(reason, (ConnectionError, socket.gaierror, TimeoutError)):
        return 'transport_deferred'
    return 'integrity_or_transport_failed'


def acquire_one(source, trial, intent, *, deadline):
    """One exact source operation, including existing-file hashing; no retries."""
    entry, target = source['entry'], target_path(source)
    record = {'source_key': source['key'], 'source_binding': source_binding(source),
        'source': source, 'intent_sha256': digest(encode(intent)),
        'declaration_sha256': intent['declaration_sha256'], 'status': 'failed',
        'events': [], 'network_attempted': False, 'started_utc': utc_now(), **CLAIMS}
    opener = None
    try:
        if target.exists():
            record['sha256'] = verified_bytes(source, deadline=deadline)
            record['status'] = 'existing_byte_verified'
        elif 'metadata_base64' in entry:
            raw = base64.b64decode(entry['metadata_base64'], validate=True)
            if len(raw) != entry['bytes'] or digest(raw) != entry['sha256']:
                raise Refusal('embedded_metadata_changed')
            atomic_preserve(target, raw)
            record['sha256'] = verified_bytes(source, deadline=deadline)
            record['status'] = 'byte_verified'
        elif intent['offline']:
            record['status'] = 'missing_offline'
        else:
            if shutil.disk_usage(ROOT).free < entry['bytes'] + 512 * 1024**2:
                raise Refusal('insufficient_disk_reserve')
            record['network_attempted'] = True
            if source['dataset'] == 'lausanne':
                partial = safe_path(target.with_name(target.name + '.partial'))
                if partial.exists() and (not partial.is_file() or partial.stat().st_size > entry['bytes']):
                    raise Refusal('partial_size_or_type')
                opener = ObservedOpener(annex.open_without_redirect, trial, deadline)
                annex.acquire_file(entry, lausanne.DATA, opener=opener)
            elif source['provider'] == 'nird':
                resect.transfer_original(entry['id'], trial, deadline=deadline, events=record['events'])
            else:
                opener = ObservedOpener(build_opener(resect.pilot.NoRedirect()).open, trial, deadline)
                resect.transfer_annotation(entry['id'], trial, deadline=deadline,
                                          events=record['events'], opener=opener)
            record['sha256'] = verified_bytes(source, deadline=deadline)
            record['status'] = 'byte_verified'
        check_deadline(deadline)
        if record['status'] in GOOD:
            record['verified_bytes'] = entry['bytes']
    except Exception as error:
        http = opener.failure if opener else native_failure(trial)
        record.update(status=failure_kind(error, http), http_failure=http,
            error_type=type(error).__name__, error_code=str(error) if isinstance(error, Refusal) else 'source_operation_failed')
        if not http and source['provider'] == 'nird' and record['events']:
            code = record['events'][-1].get('curl_exit_code')
            if code in {5, 6, 7, 18, 28, 52, 55, 56}:
                record['status'] = 'transport_deferred'
            elif code in {35, 51, 58, 59, 60, 64, 66, 77, 80, 82, 83, 90, 91}:
                record['status'] = 'tls_failed'
    finally:
        partial = (target.with_name(target.name + '.partial') if source['dataset'] == 'lausanne'
                   else trial / 'body.partial')
        record['retained_partial_bytes'] = partial.stat().st_size if partial.exists() else 0
        record['finished_utc'] = utc_now()
        save(trial / 'result.json', record)
    return record


def validate_result(source, intent, trial, record, deadline):
    if (record.get('source') != source or record.get('source_key') != source['key']
            or record.get('source_binding') != source_binding(source)
            or record.get('intent_sha256') != digest(encode(intent))
            or record.get('declaration_sha256') != intent['declaration_sha256']
            or any(type(record.get(k)) is not type(v) or record[k] != v for k, v in CLAIMS.items())
            or record.get('status') not in GOOD | DEFERRED | {'failed', 'transport_failed', 'tls_failed',
                'integrity_or_transport_failed', 'missing_offline'}
            or type(record.get('network_attempted')) is not bool):
        raise Refusal('worker_result_binding')
    if record['status'] in GOOD:
        if record.get('verified_bytes') != source['entry']['bytes'] or not re.fullmatch(r'[a-f0-9]{64}', record.get('sha256', '')):
            raise Refusal('worker_fixity_record')
        if verified_bytes(source, deadline=deadline) != record['sha256']:
            raise Refusal('worker_fixity_recheck')
    http = record.get('http_failure')
    if record['status'] == 'rate_deferred' and (not isinstance(http, dict) or http.get('status') != 429):
        raise Refusal('worker_rate_evidence')
    if http:
        if type(http.get('status')) is not int or not 400 <= http['status'] <= 599:
            raise Refusal('worker_http_status')
        for key in ('observed_utc_epoch', 'retry_after_utc_epoch'):
            if http.get(key) is not None and (type(http[key]) not in (int, float) or not math.isfinite(http[key]) or http[key] < 0):
                raise Refusal('worker_http_timing')
    raw = read_small(trial / 'result.json')
    if json.loads(raw) != record:
        raise Refusal('worker_result_changed_during_validation')
    return digest(raw)


def worker(attempt_relative, expected_sha):
    trial = safe_path(ROOT / attempt_relative)
    if not trial.is_relative_to(safe_path(CACHE / 'objects')):
        raise Refusal('worker_attempt_path')
    intent_raw = read_small(trial / 'intent.json')
    if digest(intent_raw) != expected_sha:
        raise Refusal('worker_intent_hash')
    intent = json.loads(intent_raw)
    if intent['supervisor_pid'] != os.getppid() or not 0 < intent['deadline'] - time.monotonic() <= 595:
        raise Refusal('worker_parent_or_deadline')
    run = safe_path(ROOT / intent['run'])
    if not run.is_relative_to(safe_path(CACHE / 'runs')):
        raise Refusal('worker_run_path')
    declaration = read_json(run / 'declaration.json')
    if (digest(encode(declaration)) != intent['declaration_sha256']
            or runtime_binding() != declaration['execution']):
        raise Refusal('worker_execution_binding')
    sources = catalog()
    if digest(encode(sources)) != declaration['catalog_sha256']:
        raise Refusal('worker_catalog_changed')
    source = next((s for s in sources if s['key'] == intent['source_key']), None)
    if source is None or source_binding(source) != intent['source_binding']:
        raise Refusal('worker_exact_source')
    if (intent['offline'] is not declaration['offline']
            or intent['deadline'] - time.monotonic() > (595 if source['dataset'] == 'lausanne' else 295)):
        raise Refusal('worker_declared_resource_scope')
    save(trial / 'worker-claim.json', {'pid': os.getpid(), 'intent_sha256': expected_sha})
    return acquire_one(source, trial, intent, deadline=intent['deadline'])


def object_directory(source):
    return safe_path(CACHE / 'objects' / source['key'])


def history(source):
    directory = object_directory(source) / 'attempts'
    if not directory.exists():
        return []
    records = []
    trials = sorted(directory.iterdir())
    if len(trials) > 10000:
        raise Refusal('history_resource_bound')
    for trial in trials:
        safe_path(trial)
        intent = read_json(trial / 'intent.json')
        if intent.get('source_binding') != source_binding(source) or intent.get('source_key') != source['key']:
            raise Refusal('history_source_binding')
        path = trial / 'outcome.json'
        if not path.exists():
            # An unresolved start/interrupt is kept as a live link. It cannot be
            # bypassed automatically; ordinary signals do write an outcome.
            raise Refusal('unclosed_attempt_preserved')
        value = read_json(path)
        if (value.get('intent_sha256') != digest(encode(intent))
                or value.get('source_binding') != source_binding(source)
                or value.get('source_key') != source['key'] or value.get('provider') != source['provider']
                or value.get('attempt') != str(trial.relative_to(ROOT))
                or any(type(value.get(k)) is not type(v) or value[k] != v for k, v in CLAIMS.items())):
            raise Refusal('history_outcome_binding')
        for key in ('not_before_utc_epoch', 'provider_not_before_utc_epoch'):
            if key in value and (type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] < 0):
                raise Refusal('history_cooldown_invalid')
        if value.get('result_sha256') and digest(read_small(trial / 'result.json')) != value['result_sha256']:
            raise Refusal('history_result_changed')
        if value['status'] in GOOD and not value.get('result_sha256'):
            raise Refusal('history_verified_without_result')
        records.append(value)
    return records


def provider_state(histories):
    providers = {}
    for records in histories.values():
        for record in records:
            if record.get('provider_not_before_utc_epoch') is not None:
                provider = record['provider']
                state = providers.setdefault(provider, {'not_before': 0, 'failures': 0})
                state['not_before'] = max(state['not_before'], record['provider_not_before_utc_epoch'])
                state['failures'] += 1
    return providers


def apply_backoff(outcome, records, providers, now):
    failures = sum(r.get('network_attempted', False) and r['status'] not in GOOD for r in records) + 1
    provider = providers.get(outcome['provider'], {'not_before': 0, 'failures': 0})
    status = outcome['status']
    if status in DEFERRED:
        delay = min(21600, 900 * 2 ** min(provider['failures'], 5))
        requested = (outcome.get('http_failure') or {}).get('retry_after_utc_epoch') or 0
        until = max(now + delay, requested, provider['not_before'])
        outcome['not_before_utc_epoch'] = until
        # All 429 and Retry-After responses cool down their provider. Network
        # failures without provider instructions defer just the affected file.
        if status == 'rate_deferred' or requested:
            outcome['provider_not_before_utc_epoch'] = until
        if failures >= MAX_ATTEMPTS:
            outcome.update(status='transport_attempts_exhausted', deferred_status=status)


def run_one(source, run, declaration, records, providers, *, launcher=supervise):
    trial = object_directory(source) / 'attempts' / new_id()
    trial.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    budget = 600 if source['dataset'] == 'lausanne' else 300
    deadline = started + budget
    intent = {'source_key': source['key'], 'source_binding': source_binding(source),
        'run': str(run.relative_to(ROOT)), 'declaration_sha256': digest(encode(declaration)),
        'supervisor_pid': os.getpid(), 'deadline': deadline - 5, 'offline': declaration['offline'],
        'previous': records[-1]['attempt'] if records else None, 'created_utc': utc_now()}
    # Publish the active link before any worker can start; never remove it on error.
    save(trial / 'intent.json', intent)
    outcome = {'source_key': source['key'], 'source_binding': source_binding(source),
        'provider': source['provider'], 'attempt': str(trial.relative_to(ROOT)),
        'intent_sha256': digest(encode(intent)), 'status': 'setup_failed',
        'network_attempted': False, 'started_utc': utc_now(), **CLAIMS}
    try:
        if target_path(source).exists() or source['provider'] == 'embedded' or declaration['offline']:
            acquire_one(source, trial, intent, deadline=intent['deadline'])
            outcome['supervision'] = 'local_verification'
        else:
            outcome['network_attempted'] = True
            command = [sys.executable, str(ROOT / CODE[0]), '_worker',
                       str(trial.relative_to(ROOT)), digest(encode(intent))]
            def started_worker(pid):
                outcome['worker_pid'] = pid
                save(trial / 'worker-started.json', {'pid': pid, 'intent_sha256': outcome['intent_sha256']})
            state, code = launcher(command, trial / 'worker.log', deadline=deadline, on_start=started_worker)
            outcome.update(supervision=state, returncode=code)
            if state != 'completed':
                outcome['status'] = 'transport_deferred' if state in {'timeout', 'worker_failed'} else 'worker_failed'
        if (trial / 'result.json').exists():
            result = read_json(trial / 'result.json')
            outcome['result_sha256'] = validate_result(source, intent, trial, result, deadline)
            if outcome.get('supervision') in {'completed', 'local_verification'}:
                outcome.update({k: result[k] for k in ('status', 'network_attempted')})
                for key in ('sha256', 'verified_bytes', 'http_failure', 'retained_partial_bytes'):
                    if key in result:
                        outcome[key] = result[key]
        elif outcome.get('supervision') == 'completed':
            raise Refusal('worker_result_missing')
        check_deadline(deadline)
    except KeyboardInterrupt:
        outcome['status'] = 'interrupted'
        raise
    except Exception as error:
        outcome.update(status='transport_deferred' if isinstance(error, IntakeDeadline) else 'control_failed',
            error_type=type(error).__name__, error_code=str(error) if isinstance(error, Refusal) else 'supervision_failed')
    finally:
        elapsed = time.monotonic() - started
        if elapsed >= budget:
            outcome.update(status='transport_deferred', deadline_exceeded=True)
        outcome.update(elapsed_seconds=elapsed, finished_utc=utc_now())
        apply_backoff(outcome, records, providers, time.time())
        save(trial / 'outcome.json', outcome)
    return outcome


def choose_source(sources, histories, providers, now, checked, offline):
    waits = []
    for source in sources:
        key, records = source['key'], histories[source['key']]
        if key in checked:
            continue
        last = records[-1] if records else None
        if last and last['status'] not in GOOD | DEFERRED | {'missing_offline', 'interrupted', 'setup_failed'}:
            continue
        present = target_path(source).exists()
        if offline or present or source['provider'] == 'embedded':
            return source, None
        until = max((last or {}).get('not_before_utc_epoch', 0),
                    providers.get(source['provider'], {}).get('not_before', 0))
        if until <= now:
            return source, None
        waits.append(until)
    return None, min(waits) if waits else None


def summary(sources, histories):
    counts = {}
    for source in sources:
        records = histories[source['key']]
        status = records[-1]['status'] if records else 'pending'
        counts[status] = counts.get(status, 0) + 1
    return {'files': len(sources), 'counts': counts, 'byte_verified_files': sum(counts.get(k, 0) for k in GOOD), **CLAIMS}


def run(*, offline=False):
    sources = catalog()
    safe_path(CACHE).mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, termination_cleanup():
        for path in (CACHE / 'queue.lock', lausanne.CACHE / 'batch.lock', resect.DATA / 'pilot.lock'):
            path = safe_path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context(path.open('a+'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_path, declaration = freeze_run(sources, offline)
        histories = {s['key']: history(s) for s in sources}
        checked, providers = set(), provider_state(histories)
        while True:
            source, wake = choose_source(sources, histories, providers, time.time(), checked, offline)
            if source is None:
                if wake is None:
                    result = summary(sources, histories)
                    result['status'] = ('all_bytes_verified' if result['byte_verified_files'] == len(sources)
                                        else 'completed_with_unresolved_files')
                    result['unresolved_files'] = len(sources) - result['byte_verified_files']
                    save(run_path / 'finished.json', result)
                    print(json.dumps(result), flush=True)
                    return result
                # Durable outcome already records next eligibility. Short sleeps
                # make stop signals responsive without polling either provider.
                time.sleep(min(60., max(.01, wake - time.time())))
                continue
            if runtime_binding() != declaration['execution']:
                raise Refusal('execution_changed_during_run')
            outcome = run_one(source, run_path, declaration, histories[source['key']], providers)
            histories[source['key']].append(outcome)
            providers = provider_state(histories)
            if outcome['status'] not in DEFERRED:
                checked.add(source['key'])
            print(json.dumps({'source_key': source['key'], 'status': outcome['status'],
                              **summary(sources, histories)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status')
    go = sub.add_parser('run')
    go.add_argument('--offline', action='store_true')
    child = sub.add_parser('_worker')
    child.add_argument('attempt')
    child.add_argument('intent_sha256')
    args = parser.parse_args()
    try:
        if args.command == '_worker':
            worker(args.attempt, args.intent_sha256)
        elif args.command == 'run':
            result = run(offline=args.offline)
            return 0 if result['status'] == 'all_bytes_verified' else 2
        else:
            sources = catalog()
            value = summary(sources, {s['key']: history(s) for s in sources})
            value['present_files_unverified_this_command'] = sum(target_path(s).exists() for s in sources)
            print(json.dumps(value, indent=2))
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({'status': 'stopped', 'error_type': type(error).__name__,
                          'error_code': str(error) if isinstance(error, Refusal) else 'operation_stopped'}), file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
