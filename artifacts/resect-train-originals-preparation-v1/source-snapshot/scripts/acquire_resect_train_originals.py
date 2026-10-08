#!/usr/bin/env python3
"""Serial, resumable acquisition of the exact 104 original RESECT TRAIN files.

preflight/status are metadata-only. run acquires bytes; it never decodes images
or landmarks. Old acquisition scopes and helpers remain unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

import acquire_resect_cavity as native
import real_intake_io as io

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifests/resect-train-originals-v1.json'
MANIFEST_SHA = '1764a0ce75b449d542ef51578c54cab6202b66d72c71b811999cb51a55164135'
PREFIX = 'https://s3.nird.sigma2.no/archive-ro/5686d8fa-2003-4837-8e66-8e887fabe21e/'
CODE = ('scripts/acquire_resect_train_originals.py', 'scripts/acquire_resect_cavity.py',
        'scripts/real_intake_io.py')
CHUNK = 1024 * 1024
GOOD = {'byte_verified', 'existing_byte_verified'}
DEFERRED = {'transport_deferred', 'rate_deferred', 'interrupted'}
CLAIMS = {'training_admitted': False, 'spatial_planning_admitted': False,
          'header_qc': 'not_run', 'geometry_qc': 'not_run', 'anatomy_qc': 'not_run',
          'decoded_array_bytes': 0, 'optimizer_updates': 0, 'recorded_rl_transitions': 0}


class Refusal(ValueError):
    """Fixed local reason; never a raw transport exception or response body."""


def encode(value):
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def safe(root, path):
    path = Path(path).absolute()
    try:
        parts = path.relative_to(root.absolute()).parts
    except ValueError:
        raise Refusal('outside_workspace') from None
    if '..' in parts:
        raise Refusal('noncanonical_path')
    current = root
    for part in parts:
        current /= part
        if current.is_symlink():
            raise Refusal('symlink_path')
    return path


def regular(path, limit):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise Refusal('file_type_or_bound')
    return info


def signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def small(root, path, limit=2 * CHUNK):
    path = safe(root, path)
    before = regular(path, limit)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as handle:
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise Refusal('metadata_changed')
        raw = handle.read(limit + 1)
    if len(raw) != before.st_size or signature(path.lstat()) != signature(before):
        raise Refusal('metadata_changed')
    return raw


def read_json(root, path):
    return json.loads(small(root, path))


def file_hash(path, limit, deadline, *, count=None):
    """Bounded byte hashing, including prefixes; no scientific parsing."""
    io.check_deadline(deadline)
    before = regular(path, limit)
    amount = before.st_size if count is None else count
    if not 0 <= amount <= before.st_size:
        raise Refusal('prefix_length')
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as handle:
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise Refusal('body_changed')
        remaining = amount
        while remaining:
            io.check_deadline(deadline)
            block = handle.read(min(CHUNK, remaining))
            if not block:
                raise Refusal('short_local_body')
            sha.update(block)
            md5.update(block)
            remaining -= len(block)
        if signature(os.fstat(handle.fileno())) != signature(before):
            raise Refusal('body_changed')
    if signature(path.lstat()) != signature(before):
        raise Refusal('body_changed')
    io.check_deadline(deadline)
    return {'bytes': amount, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}


def verify(path, source, deadline, expected_sha=None):
    value = file_hash(path, source['bytes'], deadline)
    if (value['bytes'] != source['bytes'] or value['md5'] != source['expected_md5']
            or (expected_sha and value['sha256'] != expected_sha)):
        raise Refusal('source_fixity')
    return value


def validate_manifest(manifest, proofs):
    sources = manifest['sources']
    cohort = json.loads(proofs['manifests/resect-component-cohort-v1.json'])
    roles = {m['patient_group']: m['role'] for m in cohort['members']}
    rows = proofs['data/mechanics/resect-metadata/table-of-contents.csv'].decode().splitlines()
    dataset = json.loads(proofs['data/mechanics/resect-metadata/next-data.json'])['props']['pageProps']['dataset']
    if (len(sources) != 104 or sum(s['bytes'] for s in sources) != 672336857
            or len({s['patient_group'] for s in sources}) != 14
            or dataset['license_id'] != 'CC-BY-4.0'
            or manifest['bounds']['compressed_cache_bytes'] != 1024**3
            or manifest['claims'] != CLAIMS):
        raise Refusal('manifest_scope')
    if any(len({s[key] for s in sources}) != 104 for key in ('id', 'destination', 'source_url')):
        raise Refusal('duplicate_source')
    for source in sources:
        p = PurePosixPath(source['source_path'])
        row = [x.strip() for x in rows[source['inventory_line'] - 1].split('|')]
        if (len(p.parts) != 5 or p.parts[:2] != ('RESECT', 'NIFTI')
                or roles.get(source['patient_group']) != 'TRAIN' or source['role'] != 'TRAIN'
                or source['patient_group'] != 'RESECT:' + p.parts[2]
                or source['id'] != 'resect-original-' + p.name
                or source['destination'] != 'data/anatomy/resect-v1/' + '/'.join(p.parts[2:])
                or source['source_url'] != PREFIX + str(p)
                or len(row) != 7 or row[2] != str(p) or int(row[4]) != source['bytes']
                or row[6] != source['expected_md5'] or row[0] != source['published_http_url']
                or row[1] != source['published_s3_uri']
                or not re.fullmatch('[0-9a-f]{32}', source['expected_md5'])):
            raise Refusal('source_identity_or_role')
        expected = ('source_correspondence_landmarks' if p.parts[3] == 'Landmarks'
                    else 'original_mri' if p.parts[3] == 'MRI' else 'original_ultrasound')
        if (source['kind'] != expected or p.parts[3] not in ('MRI', 'US', 'Landmarks')
                or not p.name.endswith('.tag' if p.parts[3] == 'Landmarks' else '.nii.gz')):
            raise Refusal('source_kind')


def load_manifest(root=ROOT):
    raw = small(root, root / MANIFEST)
    if digest(raw) != MANIFEST_SHA:
        raise Refusal('manifest_changed')
    manifest = json.loads(raw)
    proofs = {}
    for proof in manifest['proofs']:
        value = small(root, root / proof['path'])
        if len(value) != proof['bytes'] or digest(value) != proof['sha256']:
            raise Refusal('portable_proof_changed')
        proofs[proof['original_path']] = value
    role = manifest['role_source']
    if digest(small(root, root / role['path'])) != role['sha256']:
        raise Refusal('active_roles_changed')
    validate_manifest(manifest, proofs)
    return manifest


def runtime_binding(manifest, root=ROOT):
    expected = {p['original_path']: p['sha256'] for p in manifest['proofs']}
    values = {name: digest(small(root, root / name)) for name in CODE}
    for name in CODE[1:]:
        if values[name] != expected[name]:
            raise Refusal('existing_helper_changed')
    if Path(native.__file__).resolve() != root / CODE[1] or Path(io.__file__).resolve() != root / CODE[2]:
        raise Refusal('shadow_helper_import')
    return values


def retry_after(value, now):
    if not isinstance(value, str) or len(value) > 128:
        return None
    try:
        return now + int(value) if re.fullmatch(r'\d{1,12}', value.strip()) else max(
            now, parsedate_to_datetime(value).timestamp())
    except (ValueError, TypeError, OverflowError):
        return None


class Queue:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.bounds = manifest['bounds']
        self.cache = safe(root, root / 'data/acquisition/resect-originals-v1')
        self.data = safe(root, root / 'data/anatomy/resect-v1')
        self.sources = {s['id']: s for s in manifest['sources']}

    def path(self, path):
        return safe(self.root, path)

    def save(self, path, value):
        io.atomic_preserve(self.path(path), encode(value))

    def object_dir(self, source):
        return self.path(self.cache / 'objects' / source['id'])

    def partial(self, source):
        return self.path(self.object_dir(source) / 'body.partial')

    def target(self, source):
        return self.path(self.root / source['destination'])

    def binding(self, source):
        return digest(encode(source))

    def used_bytes(self):
        seen, total = set(), 0
        for base in (self.cache, self.data):
            if not base.exists():
                continue
            for directory, dirs, files in os.walk(base, followlinks=False):
                for name in dirs:
                    self.path(Path(directory) / name)
                for name in files:
                    info = regular(self.path(Path(directory) / name), self.bounds['compressed_cache_bytes'])
                    key = info.st_dev, info.st_ino
                    if key not in seen:
                        total += info.st_size
                        seen.add(key)
        return total

    def reserve(self, additional):
        if self.used_bytes() + additional > self.bounds['compressed_cache_bytes']:
            raise Refusal('compressed_cache_quota')
        if shutil.disk_usage(self.root).free < additional + self.bounds['free_disk_reserve_bytes']:
            raise Refusal('free_disk_reserve')

    def attempts(self, source):
        base = self.object_dir(source) / 'attempts'
        if not base.exists():
            return []
        paths = sorted(base.iterdir())
        if len(paths) > self.bounds['max_attempts_per_object']:
            raise Refusal('attempt_history_bound')
        records = []
        for index, path in enumerate(paths, 1):
            self.path(path)
            if path.name != f'{index:04d}' or not path.is_dir():
                raise Refusal('attempt_history_sequence')
            intent = read_json(self.root, path / 'intent.json')
            if (intent['source_id'] != source['id'] or intent['source_binding'] != self.binding(source)
                    or intent['attempt'] != index or intent['manifest_sha256'] != MANIFEST_SHA):
                raise Refusal('attempt_source_binding')
            result_path = path / 'result.json'
            result = read_json(self.root, result_path) if result_path.exists() else None
            if result and (result.get('intent_sha256') != digest(encode(intent))
                           or any(result.get(k) != v for k, v in CLAIMS.items())):
                raise Refusal('attempt_result_binding')
            records.append((path, intent, result))
        return records

    def completion(self, source):
        path = self.object_dir(source) / 'completion.json'
        if not path.exists():
            return None
        result = read_json(self.root, path)
        if (result.get('source_binding') != self.binding(source) or result.get('bytes') != source['bytes']
                or result.get('md5') != source['expected_md5'] or not re.fullmatch('[0-9a-f]{64}', result.get('sha256', ''))
                or any(result.get(k) != v for k, v in CLAIMS.items())):
            raise Refusal('completion_binding')
        return result


def header_contract(status, headers, source, offset):
    remaining = source['bytes'] - offset
    if (headers.get_all('Content-Length', []) != [str(remaining)]
            or headers.get_all('Content-Encoding', []) not in ([], ['identity'])):
        raise Refusal('response_length_or_encoding')
    if offset:
        if status != 206:
            raise Refusal('range_not_honored')
        expected = f"bytes {offset}-{source['bytes'] - 1}/{source['bytes']}"
        if headers.get_all('Content-Range', []) != [expected]:
            raise Refusal('content_range_mismatch')
    elif status != 200 or headers.get_all('Content-Range', []):
        raise Refusal('initial_response_contract')


def prefix_state(queue, source, deadline):
    latest = {'bytes': 0, 'sha256': digest(b'')}
    for trial, _, _ in queue.attempts(source):
        path = trial / 'merged.json'
        if path.exists():
            merged = read_json(queue.root, path)
            if merged['source_binding'] != queue.binding(source) or merged['before_bytes'] != latest['bytes']:
                raise Refusal('prefix_history')
            latest = {'bytes': merged['after_bytes'], 'sha256': merged['after_sha256']}
    partial = queue.partial(source)
    if partial.exists():
        actual = file_hash(partial, source['bytes'], deadline)
        if actual['bytes'] != latest['bytes'] or actual['sha256'] != latest['sha256']:
            raise Refusal('prefix_changed_or_unclosed_merge')
    elif latest['bytes']:
        raise Refusal('partial_missing')
    return latest


def combined_hash(prefix, offset, suffix, source, deadline):
    sha, md5 = hashlib.sha256(), hashlib.md5()
    for path, size in ((prefix, offset), (suffix, suffix.stat().st_size)):
        if not size:
            continue
        before = regular(path, source['bytes'])
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as handle:
            left = size
            while left:
                io.check_deadline(deadline)
                block = handle.read(min(CHUNK, left))
                if not block:
                    raise Refusal('short_merge_input')
                sha.update(block)
                md5.update(block)
                left -= len(block)
            if signature(os.fstat(handle.fileno())) != signature(before):
                raise Refusal('merge_input_changed')
    return sha.hexdigest(), md5.hexdigest()


def same_tail(partial, suffix, offset, length, deadline):
    with os.fdopen(os.open(partial, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as first, \
            os.fdopen(os.open(suffix, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as second:
        first.seek(offset)
        while length:
            io.check_deadline(deadline)
            size = min(CHUNK, length)
            a, b = first.read(size), second.read(size)
            if len(a) != size or a != b:
                raise Refusal('torn_merge_content')
            length -= size


def merge_suffix(queue, source, trial, intent, deadline):
    """Validate staged response before append; recover an interrupted append."""
    partial, suffix = queue.partial(source), queue.path(trial / 'body.partial')
    offset = intent['prefix']['bytes']
    status, headers = native.nird_response_headers(queue.path(trial / 'headers.local'))
    header_contract(status, headers, source, offset)
    if not suffix.exists():
        raise Refusal('suffix_missing')
    tail = file_hash(suffix, source['bytes'] - offset, deadline)
    if not tail['bytes']:
        return None
    if offset:
        head = file_hash(partial, source['bytes'], deadline, count=offset)
        if head['sha256'] != intent['prefix']['sha256']:
            raise Refusal('prefix_changed')
    after_sha, after_md5 = combined_hash(partial, offset, suffix, source, deadline)
    after_bytes = offset + tail['bytes']
    if after_bytes == source['bytes'] and after_md5 != source['expected_md5']:
        raise Refusal('source_fixity')
    merge = {'source_binding': queue.binding(source), 'intent_sha256': digest(encode(intent)),
             'before_bytes': offset, 'before_sha256': intent['prefix']['sha256'],
             'suffix_bytes': tail['bytes'], 'suffix_sha256': tail['sha256'],
             'after_bytes': after_bytes, 'after_sha256': after_sha, 'after_md5': after_md5}
    queue.save(trial / 'merge.json', merge)
    if not partial.exists():
        if offset:
            raise Refusal('prefix_missing')
        partial.parent.mkdir(parents=True, exist_ok=True)
        with partial.open('xb'):
            pass
    current = regular(partial, after_bytes).st_size
    if not offset <= current <= after_bytes:
        raise Refusal('torn_merge_length')
    same_tail(partial, suffix, offset, current - offset, deadline)
    queue.reserve(after_bytes - current + 4096)
    with os.fdopen(os.open(partial, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW), 'ab') as output, \
            os.fdopen(os.open(suffix, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as incoming:
        if os.fstat(output.fileno()).st_size != current:
            raise Refusal('merge_destination_changed')
        incoming.seek(current - offset)
        left = after_bytes - current
        while left:
            io.check_deadline(deadline)
            block = incoming.read(min(CHUNK, left))
            if not block:
                raise Refusal('short_suffix')
            output.write(block)
            left -= len(block)
        output.flush()
        os.fsync(output.fileno())
    actual = file_hash(partial, source['bytes'], deadline)
    if actual['bytes'] != after_bytes or actual['sha256'] != after_sha:
        raise Refusal('merged_bytes_changed')
    queue.save(trial / 'merged.json', merge)
    # This verified suffix is now retained, byte-for-byte, in the durable prefix.
    # Keep the immutable merge/hash receipt, not a second copy of its payload.
    suffix.unlink()
    return actual


def publish(queue, source, deadline):
    target, partial = queue.target(source), queue.partial(source)
    prior = queue.completion(source)
    if target.exists():
        actual = verify(target, source, deadline, prior['sha256'] if prior else None)
    else:
        actual = verify(partial, source, deadline)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(partial, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    completion = {'source_binding': queue.binding(source), **actual, **CLAIMS}
    queue.save(queue.object_dir(source) / 'completion.json', completion)
    if partial.exists():
        if not os.path.samestat(partial.stat(), target.stat()):
            raise Refusal('publication_inode')
        partial.unlink()
    return completion


def native_fetch(queue, source, trial, intent, deadline):
    client = native.system_curl_binding(timeout=min(5., max(.001, deadline - time.monotonic())))
    if not client['available']:
        raise Refusal('verified_system_trust_unavailable')
    offset = intent['prefix']['bytes']
    remaining = source['bytes'] - offset
    queue.reserve(2 * remaining + 65536)
    queue.save(trial / 'native-request.json', {'client': client, 'source_binding': queue.binding(source),
               'offset': offset, 'length': remaining, 'url_sha256': digest(source['source_url'].encode())})
    body, headers = queue.path(trial / 'body.partial'), queue.path(trial / 'headers.local')
    with body.open('xb') as output, headers.open('xb') as head:
        seconds = max(.001, deadline - time.monotonic())
        command = ['/usr/bin/curl', '-q', '--no-location', '--max-redirs', '0', '--silent', '--fail',
                   '--proto', '=https', '--tlsv1.2', '--retry', '0', '--connect-timeout', str(min(30., seconds)),
                   '--max-time', str(seconds), '--max-filesize', str(remaining), '--header', 'Accept-Encoding: identity',
                   '--dump-header', f'/dev/fd/{head.fileno()}', '--output', '-']
        if offset:
            command += ['--range', f"{offset}-{source['bytes'] - 1}"]
        command += [source['source_url']]
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.DEVNULL,
                                   pass_fds=(head.fileno(),), env=native.system_curl_environment())
        try:
            while process.poll() is None:
                io.check_deadline(deadline)
                if body.stat().st_size > remaining or headers.stat().st_size > 65536:
                    raise Refusal('response_resource_bound')
                time.sleep(.05)
            return process.returncode
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            output.flush()
            os.fsync(output.fileno())
            head.flush()
            os.fsync(head.fileno())


def close_attempt(queue, source, trial, intent, deadline, *, curl_code=None, interrupted=False):
    result = {'intent_sha256': digest(encode(intent)), 'source_binding': queue.binding(source),
              'status': 'transport_deferred', 'curl_exit_code': curl_code,
              'finished_utc': utc(), 'not_before_epoch': time.time() + queue.bounds['minimum_cooldown_seconds'], **CLAIMS}
    try:
        if curl_code in {35, 51, 58, 60, 77, 83, 90, 91}:
            raise Refusal('tls_failure')
        headers_path = trial / 'headers.local'
        if not headers_path.exists() or not headers_path.stat().st_size:
            result['status'] = 'interrupted' if interrupted else 'transport_deferred'
        else:
            status, headers = native.nird_response_headers(headers_path)
            result['http_status'] = status
            advice = retry_after(headers.get('Retry-After'), time.time())
            if status == 429 or advice is not None:
                result['provider_not_before_epoch'] = max(result['not_before_epoch'], advice or 0)
                result['not_before_epoch'] = result['provider_not_before_epoch']
            if status == 429:
                result['status'] = 'rate_deferred'
            elif status in (408, 425, 500, 502, 503, 504):
                result['status'] = 'transport_deferred'
            else:
                merged = trial / 'merged.json'
                if merged.exists():
                    # A crash after durable merge and suffix compaction must
                    # not turn a valid prefix into a missing-suffix failure.
                    saved = read_json(queue.root, merged)
                    if saved['intent_sha256'] != digest(encode(intent)) or saved['source_binding'] != queue.binding(source):
                        raise Refusal('merged_receipt_binding')
                    actual = file_hash(queue.partial(source), source['bytes'], deadline)
                    if actual['bytes'] != saved['after_bytes'] or actual['sha256'] != saved['after_sha256']:
                        raise Refusal('merged_prefix_changed')
                else:
                    actual = merge_suffix(queue, source, trial, intent, deadline)
                if actual and actual['bytes'] == source['bytes']:
                    result.update(status='byte_verified', completed=publish(queue, source, deadline))
                elif actual:
                    result['prefix'] = actual
    except io.IntakeDeadline:
        result['status'] = 'interrupted'
    except (ValueError, OSError) as error:
        result.update(status='terminal_failure', error_code=str(error) if isinstance(error, Refusal) else 'response_or_local_state_invalid')
    if intent['attempt'] >= queue.bounds['max_attempts_per_object'] and result['status'] in DEFERRED:
        result['status'] = 'attempts_exhausted'
    queue.save(trial / 'result.json', result)
    return result


def recover(queue, source, deadline):
    """Recover only locally retained bytes; never starts a request."""
    if queue.target(source).exists():
        return publish(queue, source, deadline)
    for trial, intent, result in queue.attempts(source):
        if result is None:
            started = trial / 'worker-started.json'
            if started.exists():
                pid = read_json(queue.root, started)['pid']
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    raise Refusal('previous_worker_still_live')
            result = close_attempt(queue, source, trial, intent, deadline, interrupted=True)
            if queue.target(source).exists():
                return publish(queue, source, deadline)
        if (trial / 'merge.json').exists() and not (trial / 'merged.json').exists():
            merge_suffix(queue, source, trial, intent, deadline)
        elif (result['status'] in DEFERRED | {'attempts_exhausted'}
              and (trial / 'body.partial').exists() and (trial / 'headers.local').exists()
              and not (trial / 'merged.json').exists()):
            # The deadline may have expired before the merge journal was made.
            # Salvage only a source-bound, correctly framed response. Error or
            # incomplete headers remain retained without changing the prefix.
            try:
                response_status, headers = native.nird_response_headers(trial / 'headers.local')
                header_contract(response_status, headers, source, intent['prefix']['bytes'])
            except (ValueError, OSError):
                pass
            else:
                actual = merge_suffix(queue, source, trial, intent, deadline)
                if actual:
                    queue.save(trial / 'recovery.json', {'source_binding': queue.binding(source),
                               'intent_sha256': digest(encode(intent)), 'recovered_prefix': actual})
    if queue.partial(source).exists() and queue.partial(source).stat().st_size == source['bytes']:
        return publish(queue, source, deadline)
    prefix_state(queue, source, deadline)
    return None


def worker(attempt, intent_sha):
    manifest = load_manifest()
    queue = Queue(ROOT, manifest)
    trial = queue.path(ROOT / attempt)
    intent = read_json(ROOT, trial / 'intent.json')
    source = queue.sources.get(intent['source_id'])
    if (source is None or digest(encode(intent)) != intent_sha or intent['source_binding'] != queue.binding(source)
            or trial != queue.object_dir(source) / 'attempts' / f"{intent['attempt']:04d}"
            or os.getppid() != intent['supervisor_pid'] or runtime_binding(manifest) != intent['runtime']
            or not 0 < intent['deadline'] - time.monotonic() <= manifest['bounds']['worker_seconds']):
        raise Refusal('worker_intent_or_runtime')
    if prefix_state(queue, source, intent['deadline']) != intent['prefix']:
        raise Refusal('worker_prefix_changed')
    curl_code, interrupted = None, False
    try:
        with io.termination_cleanup():
            curl_code = native_fetch(queue, source, trial, intent, intent['deadline'])
    except (KeyboardInterrupt, io.IntakeDeadline):
        interrupted = True
    except Exception as error:
        queue.save(trial / 'transport-error.json', {'type': type(error).__name__,
                   'code': str(error) if isinstance(error, Refusal) else 'native_transport_failed'})
        if isinstance(error, Refusal):
            queue.save(trial / 'result.json', {'intent_sha256': intent_sha, 'source_binding': queue.binding(source),
                       'status': 'terminal_failure', 'error_code': str(error), 'finished_utc': utc(), **CLAIMS})
            return 2
    result = close_attempt(queue, source, trial, intent, intent['deadline'], curl_code=curl_code, interrupted=interrupted)
    return 0 if result['status'] in GOOD else 2


def run_one(queue, source, runtime, *, launcher=io.supervise):
    records = queue.attempts(source)
    if len(records) >= queue.bounds['max_attempts_per_object']:
        raise Refusal('attempt_limit')
    deadline = time.monotonic() + queue.bounds['supervisor_seconds']
    prefix = prefix_state(queue, source, deadline)
    queue.reserve(2 * (source['bytes'] - prefix['bytes']) + 65536)
    number = len(records) + 1
    trial = queue.object_dir(source) / 'attempts' / f'{number:04d}'
    trial.mkdir(parents=True, exist_ok=False)
    intent = {'source_id': source['id'], 'source_binding': queue.binding(source), 'manifest_sha256': MANIFEST_SHA,
              'attempt': number, 'prefix': prefix, 'supervisor_pid': os.getpid(), 'deadline': deadline - 5,
              'runtime': runtime, 'created_utc': utc()}
    queue.save(trial / 'intent.json', intent)
    command = [sys.executable, '-B', str(queue.root / CODE[0]), '_worker',
               str(trial.relative_to(queue.root)), digest(encode(intent))]
    try:
        state, code = launcher(command, trial / 'worker.log', deadline=deadline,
                               on_start=lambda pid: queue.save(trial / 'worker-started.json', {'pid': pid}))
        queue.save(trial / 'supervision.json', {'state': state, 'returncode': code, 'finished_utc': utc()})
    finally:
        if not (trial / 'result.json').exists():
            # The child and descendants have been reaped. Fresh bounded local
            # recovery may preserve a validated suffix, without another GET.
            close_attempt(queue, source, trial, intent, time.monotonic() + 60, interrupted=True)
    return read_json(queue.root, trial / 'result.json')


def states(queue):
    rows, provider_until = [], 0
    for source in queue.sources.values():
        records = queue.attempts(source)
        for _, _, result in records:
            if result:
                value = result.get('provider_not_before_epoch', 0)
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise Refusal('cooldown_invalid')
                provider_until = max(provider_until, value)
        completion = queue.completion(source)
        latest = records[-1][2] if records else None
        rows.append((source, records, completion, latest))
    return rows, provider_until


def status(queue):
    rows, until = states(queue)
    counts = {}
    for _, records, completion, latest in rows:
        label = 'byte_verified' if completion else latest['status'] if latest else 'interrupted' if records else 'pending'
        counts[label] = counts.get(label, 0) + 1
    return {'files': len(rows), 'counts': counts, 'provider_not_before_epoch': until,
            'metadata_only': True, 'all_complete': counts.get('byte_verified', 0) == len(rows), **CLAIMS}


def run(queue):
    queue.cache.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, io.termination_cleanup():
        lock = stack.enter_context(queue.path(queue.cache / 'queue.lock').open('a+'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = runtime_binding(queue.manifest, queue.root)
        declaration = queue.cache / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8])
        queue.save(declaration / 'declaration.json', {'runtime': runtime, 'manifest_sha256': MANIFEST_SHA,
                   'bounds': queue.bounds, 'started_utc': utc(), **CLAIMS})
        for name in (*CODE, MANIFEST):
            io.atomic_preserve(queue.path(declaration / 'source' / name), small(queue.root, queue.root / name))
        checked, failures = set(), []
        while True:
            if runtime_binding(queue.manifest, queue.root) != runtime:
                raise Refusal('runtime_changed_during_queue')
            rows, provider_until = states(queue)
            selected, waits = None, []
            for source, records, complete, latest in rows:
                if source['id'] in checked:
                    continue
                try:
                    recovered = recover(queue, source, time.monotonic() + 295)
                except (ValueError, OSError) as error:
                    failures.append({'source_id': source['id'], 'reason': str(error) if isinstance(error, Refusal) else 'local_recovery_failed'})
                    checked.add(source['id'])
                    continue
                if recovered:
                    checked.add(source['id'])
                    continue
                records = queue.attempts(source)
                latest = records[-1][2] if records else None
                provider_until = max(provider_until, (latest or {}).get('provider_not_before_epoch', 0))
                if (latest and latest['status'] not in DEFERRED) or len(records) >= queue.bounds['max_attempts_per_object']:
                    checked.add(source['id'])
                    continue
                until = max(provider_until, (latest or {}).get('not_before_epoch', 0))
                if until <= time.time():
                    selected = source
                    break
                waits.append(until)
            if selected:
                outcome = run_one(queue, selected, runtime)
                print(json.dumps({'source_id': selected['id'], 'status': outcome['status']}), flush=True)
                continue
            if waits:
                time.sleep(min(30, max(.1, min(waits) - time.time())))
                continue
            report = status(queue)
            receipt_complete = report['all_complete']
            report.update(local_failures=failures, finished_utc=utc(), metadata_only=False,
                          receipt_counts_all_complete=receipt_complete,
                          all_complete=receipt_complete and not failures)
            queue.save(declaration / 'completion.json', report)
            print(json.dumps(report), flush=True)
            return 0 if report['all_complete'] and not failures else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['preflight', 'status', 'run', '_worker'])
    parser.add_argument('worker_arguments', nargs='*')
    args = parser.parse_args()
    if args.command == '_worker':
        if len(args.worker_arguments) != 2:
            raise Refusal('worker_arguments')
        return worker(*args.worker_arguments)
    if args.worker_arguments:
        raise Refusal('unexpected_arguments')
    manifest = load_manifest()
    runtime = runtime_binding(manifest)
    queue = Queue(ROOT, manifest)
    if args.command == 'preflight':
        print(json.dumps({'status': 'metadata_identity_passed', 'files': 104, 'people': 14,
                          'source_bytes': 672336857, 'bounds': queue.bounds, 'runtime': runtime,
                          'network_requests': 0, 'payload_reads': 0, **CLAIMS}))
        return 0
    if args.command == 'status':
        print(json.dumps(status(queue)))
        return 0
    return run(queue)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Refusal, KeyboardInterrupt) as error:
        print(json.dumps({'status': 'stopped', 'reason': str(error)}), file=sys.stderr)
        raise SystemExit(2)
