#!/usr/bin/env python3
"""Acquire one frozen SynthRAD archive as opaque bytes; never extract images.

preflight/status are metadata-only. run is continuous and resumable within the
fixed campaign. The RESECT transfer primitives are reused without modifying
their globals, source declarations, driver, or process supervisor.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time
import uuid

import acquire_resect_train_originals as transfer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = 'manifests/synthrad-task1-original-archive-v1.json'
MANIFEST_SHA = 'f8b854a961ef0c799c96643445501c145630329fe61714e1dc89a263a3544c5b'
FREEZE = 'artifacts/synthrad-acquisition-preparation-v1/root-freeze.json'
FREEZE_SHA = 'd5c9a12c569ab35be44274db14860eb5de0e5b2f9e995cd6cb4204bc6f574a23'
CODE = 'scripts/acquire_synthrad_task1_archive.py'
Refusal = transfer.Refusal
io = transfer.io
CLAIMS = transfer.CLAIMS


def load_manifest(root=ROOT):
    raw = transfer.small(root, root / MANIFEST)
    frozen_raw = transfer.small(root, root / FREEZE)
    if transfer.digest(raw) != MANIFEST_SHA or transfer.digest(frozen_raw) != FREEZE_SHA:
        raise Refusal('manifest_or_root_freeze_changed')
    manifest, frozen = json.loads(raw), json.loads(frozen_raw)
    for path, digest in frozen['files_sha256'].items():
        if transfer.digest(transfer.small(root, root / path)) != digest:
            raise Refusal('frozen_input_changed')
    for proof in manifest['proofs']:
        value = transfer.small(root, root / proof['path'])
        if len(value) != proof['bytes'] or transfer.digest(value) != proof['sha256']:
            raise Refusal('portable_proof_changed')
    role = manifest['role_source']
    if transfer.digest(transfer.small(root, root / role['path'])) != role['sha256']:
        raise Refusal('prospective_roles_changed')
    if manifest['claims'] != CLAIMS or len(manifest['sources']) != 1:
        raise Refusal('archive_scope')
    return manifest


def runtime_binding(manifest, root=ROOT):
    modules = (transfer, transfer.native, io)
    proofs = manifest['implementation']['existing_helpers']
    values = {CODE: transfer.digest(transfer.small(root, root / CODE))}
    for module, proof in zip(modules, proofs, strict=True):
        if Path(module.__file__).resolve() != root / proof['path']:
            raise Refusal('shadow_helper_import')
        raw = transfer.small(root, root / proof['path'])
        if len(raw) != proof['bytes'] or transfer.digest(raw) != proof['sha256']:
            raise Refusal('existing_helper_changed')
        values[proof['path']] = transfer.digest(raw)
    return values


class Queue(transfer.Queue):
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.bounds = manifest['bounds']
        self.cache = transfer.safe(root, root / manifest['cache_root'])
        self.data = self.cache / 'verified'
        self.sources = {s['id']: s for s in manifest['sources']}

    def save(self, path, value):
        raw = transfer.encode(value)
        if len(raw) > self.bounds['source_metadata_file_bytes']:
            raise Refusal('metadata_size_bound')
        self.reserve(2 * len(raw))
        io.atomic_preserve(self.path(path), raw)

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
            intent = transfer.read_json(self.root, path / 'intent.json')
            if (intent['source_id'] != source['id'] or intent['source_binding'] != self.binding(source)
                    or intent['attempt'] != index or intent['manifest_sha256'] != MANIFEST_SHA):
                raise Refusal('attempt_source_binding')
            result_path = path / 'result.json'
            result = transfer.read_json(self.root, result_path) if result_path.exists() else None
            if result and (result.get('intent_sha256') != transfer.digest(transfer.encode(intent))
                           or any(result.get(k) != v for k, v in CLAIMS.items())):
                raise Refusal('attempt_result_binding')
            records.append((path, intent, result))
        return records


def campaign(queue, *, create=False):
    """Immutable wall-clock deadline survives process restarts; never renew it."""
    path = queue.cache / 'campaign.json'
    if not path.exists():
        if not create:
            return None
        now = time.time()
        queue.save(path, {'manifest_sha256': MANIFEST_SHA, 'started_epoch': now,
                         'deadline_epoch': now + queue.bounds['total_campaign_seconds'],
                         'started_utc': transfer.utc(), **CLAIMS})
    value = transfer.read_json(queue.root, path)
    start, end = value['started_epoch'], value['deadline_epoch']
    if (value['manifest_sha256'] != MANIFEST_SHA or type(start) not in (int, float)
            or type(end) not in (int, float) or not math.isfinite(start + end)
            or end - start != queue.bounds['total_campaign_seconds']
            or time.time() < start or any(value.get(k) != v for k, v in CLAIMS.items())):
        raise Refusal('campaign_binding_or_clock')
    return value


def phase_deadline(queue, campaign_deadline):
    return min(campaign_deadline, time.monotonic() + queue.bounds['hash_or_recovery_phase_seconds'])


def supervise(command, log, *, deadline, grace_seconds, log_bytes, on_start):
    """Drain a bounded log and give the dedicated process group SIGTERM grace."""
    io.check_deadline(deadline)
    stored = 0
    state = None
    term_at = None
    started = time.monotonic()
    with log.open('xb') as output, selectors.DefaultSelector() as selector:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   start_new_session=True, bufsize=0)
        os.set_blocking(process.stdout.fileno(), False)
        selector.register(process.stdout, selectors.EVENT_READ)

        def drain():
            nonlocal stored, state
            # Bounded work per iteration also bounds an endlessly noisy child.
            for _ in range(8):
                try:
                    block = os.read(process.stdout.fileno(), 8192)
                except BlockingIOError:
                    break
                if not block:
                    break
                keep = min(len(block), log_bytes - stored)
                output.write(block[:keep])
                stored += keep
                if keep < len(block) and state is None:
                    state = 'log_limit'

        def group_signal(signum):
            try:
                os.killpg(process.pid, signum)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                # Darwin can report EPERM for signal 0 while an owned group
                # is exiting, before waitpid reaps its leader. Keep waiting;
                # this is not evidence that cleanup is complete.
                if signum == 0:
                    return True
                raise

        try:
            on_start(process.pid)
            try:
                while True:
                    drain()
                    code = process.poll()
                    if code is not None:
                        drain()
                        if state is None:
                            state = 'completed' if code == 0 else 'worker_failed'
                        break
                    if state == 'log_limit' or time.monotonic() >= deadline:
                        state = state or 'timeout'
                        break
                    selector.select(min(.1, max(0, deadline - time.monotonic())))
            except KeyboardInterrupt:
                state = 'interrupted'
        finally:
            # A normal worker exit can still leave descendants; cover them too.
            if group_signal(signal.SIGTERM):
                term_at = time.monotonic()
                until = term_at + grace_seconds
                old_int = signal.signal(signal.SIGINT, signal.SIG_IGN)
                old_term = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                try:
                    while time.monotonic() < until:
                        drain()
                        process.poll()
                        if not group_signal(0):
                            break
                        time.sleep(min(.05, max(0, until - time.monotonic())))
                    if group_signal(0):
                        group_signal(signal.SIGKILL)
                finally:
                    signal.signal(signal.SIGINT, old_int)
                    signal.signal(signal.SIGTERM, old_term)
            process.wait()
            drain()
            process.stdout.close()
            output.flush()
            os.fsync(output.fileno())
    return {'state': state, 'returncode': process.returncode, 'log_bytes': stored,
            'sigterm_sent': term_at is not None, 'termination_grace_seconds': grace_seconds,
            'elapsed_seconds': time.monotonic() - started, 'finished_utc': transfer.utc()}


def worker(attempt, intent_sha):
    manifest = load_manifest()
    queue = Queue(ROOT, manifest)
    trial = queue.path(ROOT / attempt)
    intent = transfer.read_json(ROOT, trial / 'intent.json')
    source = queue.sources.get(intent['source_id'])
    now = time.monotonic()
    if (source is None or transfer.digest(transfer.encode(intent)) != intent_sha
            or intent['manifest_sha256'] != MANIFEST_SHA or intent['source_binding'] != queue.binding(source)
            or trial != queue.object_dir(source) / 'attempts' / f"{intent['attempt']:04d}"
            or os.getppid() != intent['supervisor_pid'] or runtime_binding(manifest) != intent['runtime']
            or not 0 < intent['network_deadline'] - now <= queue.bounds['network_attempt_seconds']
            or intent['campaign_deadline'] < intent['network_deadline']
            or intent['client'] != transfer.native.system_curl_binding()):
        raise Refusal('worker_intent_or_runtime')
    active = campaign(queue)
    if active is None or active['deadline_epoch'] != intent['campaign_deadline_epoch']:
        raise Refusal('worker_campaign_binding')
    stop = min(intent['campaign_deadline'], now + max(0, active['deadline_epoch'] - time.time()))
    if transfer.prefix_state(queue, source, phase_deadline(queue, stop)) != intent['prefix']:
        raise Refusal('worker_prefix_changed')
    code, interrupted = None, False
    started = time.monotonic()
    try:
        with io.termination_cleanup():
            code = transfer.native_fetch(queue, source, trial, intent, min(stop, intent['network_deadline']))
    except (KeyboardInterrupt, io.IntakeDeadline):
        interrupted = True
    except Exception as error:
        queue.save(trial / 'transport-error.json', {'type': type(error).__name__,
                   'code': str(error) if isinstance(error, Refusal) else 'native_transport_failed'})
        if isinstance(error, Refusal):
            queue.save(trial / 'result.json', {'intent_sha256': intent_sha,
                       'source_binding': queue.binding(source), 'status': 'terminal_failure',
                       'error_code': str(error), 'finished_utc': transfer.utc(), **CLAIMS})
            return 2
    queue.save(trial / 'transport-finished.json', {'started_utc': intent['created_utc'],
               'finished_utc': transfer.utc(), 'elapsed_seconds': time.monotonic() - started,
               'curl_exit_code': code, 'interrupted': interrupted})
    result = transfer.close_attempt(queue, source, trial, intent, phase_deadline(queue, stop),
                                    curl_code=code, interrupted=interrupted)
    return 0 if result['status'] in transfer.GOOD else 2


def run_one(queue, source, runtime, client, campaign_deadline, active, *, launcher=supervise):
    records = queue.attempts(source)
    if len(records) >= queue.bounds['max_attempts_per_object']:
        raise Refusal('attempt_limit')
    io.check_deadline(campaign_deadline)
    prefix = transfer.prefix_state(queue, source, phase_deadline(queue, campaign_deadline))
    queue.reserve(2 * (source['bytes'] - prefix['bytes']) + 65536)
    grace = queue.bounds['supervisor_termination_grace_seconds']
    worker_limit = campaign_deadline - grace
    io.check_deadline(worker_limit)
    network_deadline = min(worker_limit, time.monotonic() + queue.bounds['network_attempt_seconds'])
    supervisor_deadline = min(worker_limit, network_deadline + queue.bounds['hash_or_recovery_phase_seconds'])
    number = len(records) + 1
    trial = queue.object_dir(source) / 'attempts' / f'{number:04d}'
    trial.mkdir(parents=True, exist_ok=False)
    intent = {'source_id': source['id'], 'source_binding': queue.binding(source),
              'manifest_sha256': MANIFEST_SHA, 'attempt': number, 'prefix': prefix,
              'supervisor_pid': os.getpid(), 'network_deadline': network_deadline,
              'campaign_deadline': campaign_deadline, 'campaign_deadline_epoch': active['deadline_epoch'],
              'runtime': runtime, 'client': client, 'created_utc': transfer.utc()}
    queue.save(trial / 'intent.json', intent)
    command = [sys.executable, '-B', str(queue.root / CODE), '_worker',
               str(trial.relative_to(queue.root)), transfer.digest(transfer.encode(intent))]
    supervision = None
    try:
        supervision = launcher(command, trial / 'worker.log', deadline=supervisor_deadline,
                               grace_seconds=grace, log_bytes=queue.bounds['worker_log_bytes'],
                               on_start=lambda pid: queue.save(trial / 'worker-started.json', {'pid': pid}))
        queue.save(trial / 'supervision.json', supervision)
    finally:
        if not (trial / 'result.json').exists():
            transfer.close_attempt(queue, source, trial, intent, phase_deadline(queue, campaign_deadline),
                                   interrupted=True)
    if supervision and supervision['state'] == 'interrupted':
        raise KeyboardInterrupt()
    return transfer.read_json(queue.root, trial / 'result.json')


def status(queue):
    result = transfer.status(queue)
    result['campaign'] = campaign(queue)
    result['archive_extracted'] = False
    return result


def run(queue):
    queue.cache.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack, io.termination_cleanup():
        lock = stack.enter_context(queue.path(queue.cache / 'queue.lock').open('a+'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime = runtime_binding(queue.manifest, queue.root)
        client = transfer.native.system_curl_binding()
        if not client['available']:
            raise Refusal('verified_system_trust_unavailable')
        active = campaign(queue, create=True)
        deadline = time.monotonic() + max(0, active['deadline_epoch'] - time.time())
        declaration = queue.cache / 'runs' / uuid.uuid4().hex
        queue.save(declaration / 'declaration.json', {'runtime': runtime, 'client': client,
                   'manifest_sha256': MANIFEST_SHA, 'bounds': queue.bounds,
                   'campaign': active, 'started_utc': transfer.utc(), **CLAIMS})
        source = next(iter(queue.sources.values()))
        completed, failure, interrupted = False, None, False
        try:
            while True:
                io.check_deadline(deadline)
                if (runtime_binding(queue.manifest, queue.root) != runtime
                        or transfer.digest(transfer.small(queue.root, queue.root / MANIFEST)) != MANIFEST_SHA):
                    raise Refusal('runtime_changed_during_campaign')
                if transfer.recover(queue, source, phase_deadline(queue, deadline)):
                    completed = True
                    break
                rows, provider_until = transfer.states(queue)
                _, records, _, latest = rows[0]
                if (latest and latest['status'] not in transfer.DEFERRED
                        or len(records) >= queue.bounds['max_attempts_per_object']):
                    break
                until = max(provider_until, (latest or {}).get('not_before_epoch', 0))
                if type(until) not in (int, float) or not math.isfinite(until):
                    raise Refusal('cooldown_invalid')
                if until > time.time():
                    if until >= active['deadline_epoch']:
                        raise Refusal('cooldown_beyond_campaign')
                    # Avoid repeatedly hashing a large preserved prefix during
                    # a provider cooldown; there is still only one writer.
                    while until > time.time():
                        io.check_deadline(deadline)
                        time.sleep(max(0, min(30, until - time.time(), deadline - time.monotonic())))
                outcome = run_one(queue, source, runtime, client, deadline, active)
                print(json.dumps({'source_id': source['id'], 'status': outcome['status']}), flush=True)
        except KeyboardInterrupt:
            interrupted, failure = True, 'interrupted'
        except io.IntakeDeadline:
            failure = 'campaign_or_local_phase_deadline'
        except (ValueError, OSError) as error:
            failure = str(error) if isinstance(error, Refusal) else 'local_state_failure'
        report = {'all_complete': completed, 'failure': failure, 'interrupted': interrupted,
                  'finished_utc': transfer.utc(), 'manifest_sha256': MANIFEST_SHA,
                  'archive_extracted': False, **CLAIMS}
        queue.save(declaration / 'completion.json', report)
        print(json.dumps(report), flush=True)
        return 0 if completed else 130 if interrupted else 2


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
    if args.command == 'status':
        print(json.dumps(status(queue)))
        return 0
    if args.command == 'preflight':
        client = transfer.native.system_curl_binding()
        if not client['available']:
            raise Refusal('verified_system_trust_unavailable')
        source = manifest['sources'][0]
        # Metadata-only reservation reflects retained bytes without hashing or
        # claiming that an existing prefix has passed integrity verification.
        retained = queue.partial(source)
        prefix_bytes = transfer.regular(retained, source['bytes']).st_size if retained.exists() else 0
        additional = 0 if queue.target(source).exists() else 2 * (source['bytes'] - prefix_bytes)
        queue.reserve(additional + 65536)
        print(json.dumps({'status': 'metadata_identity_passed', 'files': 1,
                          'source_bytes': manifest['sources'][0]['bytes'], 'runtime': runtime,
                          'client': client, 'bounds': queue.bounds, 'network_requests': 0,
                          'payload_reads': 0, **CLAIMS}))
        return 0
    return run(queue)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, io.IntakeDeadline) as error:
        print(json.dumps({'status': 'refused', 'reason': str(error) if isinstance(error, Refusal)
                          else type(error).__name__, **CLAIMS}))
        raise SystemExit(2)
