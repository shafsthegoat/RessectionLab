"""One separately authorized recovery attempt per pinned closed Tracto failure.

No payload decoding. Original runner, helpers, journals and roles are immutable.
The original queue lock remains held throughout this single-worker successor.
"""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request, build_opener
import argparse
import fcntl
import hashlib
import http.client
import json
import os
import runpy
import shutil
import socket
import ssl
import stat
import subprocess
import sys
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[3]
PREP = Path('build/tractoinferno-recovery-preparation-v1')
ORIGINAL = Path('data/acquisition/tractoinferno-train-v1')
RECOVERY = Path('data/acquisition/tractoinferno-recovery-v1')
RUNNER = Path('build/tractoinferno-train-preparation-v1/run-intake.py')
PROPOSAL_SHA = 'c5800243aac0bf5afdfe83a1fec4a4534b66bf844a916006e0a1f3ee842ceacd'
RESERVE, ALLOWANCE = 100 * 1024**3, 64 * 1024**3
GOOD = {'byte_verified', 'existing_byte_verified'}


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def key(entry):
    return hashlib.sha256((entry['source_url'] + '|' + str(entry['bytes']) + '|' + entry['expected_md5'] + '|' + str(entry['expected_git_blob_sha1'])).encode()).hexdigest()


def file_stat(path):
    try:
        s = path.lstat()
    except FileNotFoundError:
        return None
    require(stat.S_ISREG(s.st_mode), 'nonregular_or_symlink_file:' + str(path))
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns)


def load_contract(root, path, digest):
    require(sha(path) == digest, 'release_hash_mismatch')
    d = json.loads(path.read_text())
    require(sha(Path(__file__)) == d['runner_sha256'], 'recovery_runner_changed')
    require(d['proposal_sha256'] == PROPOSAL_SHA, 'wrong_proposal')
    ppath = root / PREP / 'proposed-recovery.json'
    require(sha(ppath) == PROPOSAL_SHA, 'proposal_changed')
    p = json.loads(ppath.read_text())
    require(sha(root / RUNNER) == p['original_runner_sha256'], 'original_runner_changed')
    require(sha(root / ORIGINAL / 'declaration.json') == p['original_declaration_sha256'], 'original_declaration_changed')
    require(sha(root / PREP / 'classify.py') == p['classification_source_sha256'], 'classifier_changed')
    original = runpy.run_path(str(root / RUNNER))
    old, _, entries = original['load_contract'](root, root / ORIGINAL, root / ORIGINAL / 'declaration.json', p['original_declaration_sha256'])
    require(len(p['objects']) == 54 and p['selected_files'] == 54, 'proposal_count_changed')
    bypath = {e['path']: e for e in entries}
    require(len({i['entry']['path'] for i in p['objects']}) == 54, 'duplicate_candidate')
    for item in p['objects']:
        require(item['entry'] == bypath.get(item['entry']['path']), 'candidate_source_changed')
        require(item['proposed_additional_payload_attempts'] == 1 and item['proposed_total_attempt_ceiling_including_history'] == item['original_attempts_used'] + 1, 'lifetime_budget_changed')
    require(d['workers'] == 1 and d['additional_payload_attempts_per_object'] == 1, 'recovery_budget_changed')
    require(d['free_space_reserve_bytes'] == RESERVE and d['active_output_allowance_bytes'] == ALLOWANCE, 'storage_bounds_changed')
    require(d['all_payloads_unreviewed'] is True and d['training_admitted'] is False, 'admission_changed')
    return d, p, old, entries


@contextmanager
def original_gate(root, proposal, old):
    # Any live occupant of the original PID is a conservative refusal, including reuse.
    pid = subprocess.run(['ps', '-p', str(proposal['original_pid']), '-o', 'pid='], capture_output=True, text=True, check=False)
    require(pid.returncode in (0, 1), 'process_query_failed')
    require(not pid.stdout.strip(), 'original_pid_still_live_no_recovery')
    with ExitStack() as stack:
        for relative in old['predecessor_queue_locks'] + [str(ORIGINAL / 'queue.lock')]:
            handle = stack.enter_context((root / relative).open('r+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def check_history(root, item):
    e = item['entry']
    journal = root / ORIGINAL / 'attempts' / key(e)[:2] / key(e)
    wanted = {str(root / h['path']): h['sha256'] for h in item['history_records']}
    actual = {str(p): sha(p) for p in journal.glob('*.json')}
    require(actual == wanted, 'original_history_changed:' + e['path'])
    intents = sorted(journal.glob('*-intent.json'))
    require(len(intents) == item['original_attempts_used'], 'original_attempt_count_changed')
    history = []
    for intent in intents:
        result = intent.with_name(intent.name.replace('-intent.json', '-result.json'))
        require(result.exists(), 'original_orphan_intent')
        hpath = intent.with_name(intent.name.replace('-intent.json', '-http.json'))
        history.append(dict(intent=json.loads(intent.read_text()), result=json.loads(result.read_text()), http=json.loads(hpath.read_text()) if hpath.exists() else None))
    classify = runpy.run_path(str(root / PREP / 'classify.py'))['classify']
    old_stat = item['partial_stat']
    label = classify(e, history, [], False, True, old_stat['size'] if old_stat else 0)
    require(label == item['classification'] and label.startswith('proposed_'), 'candidate_no_longer_eligible')


def terminal_selection(root, proposal):
    path = root / ORIGINAL / 'runs' / proposal['original_run_id'] / 'completion.json'
    c = json.loads(path.read_text())
    require(c['run_id'] == proposal['original_run_id'] and c['declaration_sha256'] == proposal['original_declaration_sha256'], 'wrong_original_terminal')
    require(c['status'] in ('all_bytes_verified', 'completed_with_unresolved_files'), 'original_not_terminal')
    rows = {r['path']: r for r in c['files']}
    require(len(rows) == len(c['files']) == 7622, 'incomplete_original_terminal')
    selected, excluded = [], []
    for item in proposal['objects']:
        row = rows[item['entry']['path']]
        if row['status'] in GOOD:
            excluded.append(item['entry']['path'])
            continue
        check_history(root, item)
        require(row['status'] == item['original_terminal_status'] and row.get('error_type') == item['original_error_type'], 'terminal_failure_changed')
        selected.append(item['entry']['path'])
    return {'path': str(path.relative_to(root)), 'sha256': sha(path)}, selected, excluded


def check_partial(root, item):
    target = root / ORIGINAL / 'verified' / item['entry']['path']
    require(not target.exists() and not target.is_symlink(), 'candidate_has_final_requires_reconciliation')
    require(file_stat(root / item['partial_path']) == item['partial_stat'], 'candidate_partial_changed')


def validate_terminal_release(root, d, p):
    require(d['execution_released'] is True and d['scope_frozen'] is True, 'recovery_not_released')
    receipt, selected, excluded = terminal_selection(root, p)
    require(d['original_terminal'] == receipt and d['selected_paths'] == selected and d['excluded_original_successes'] == excluded, 'terminal_scope_changed')
    return [i for i in p['objects'] if i['entry']['path'] in set(selected)]


def transient_error(error, phase):
    chain = [error]
    seen = set()
    while chain:
        current = chain.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            return False
        chain.extend(x for x in (getattr(current, 'reason', None), current.__cause__, current.__context__) if isinstance(x, BaseException))
    if phase not in ('http_open', 'http_read', 'http_close'):
        return False
    if isinstance(error, HTTPError):
        return error.code in (429, 500, 502, 503, 504)
    if isinstance(error, URLError):
        error = error.reason
    return isinstance(error, (TimeoutError, socket.gaierror, ConnectionError, http.client.IncompleteRead))


class Progress:
    def __init__(self, save, folder, output=None):
        self.save, self.folder = save, folder
        self.output = sys.stdout if output is None else output

    def emit(self, value):
        if self.output is None:
            return
        try:
            print(json.dumps(value), file=self.output, flush=True)
        except BrokenPipeError as error:
            record = dict(phase='progress_output', error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), object_attempt_consumed=False, utc=utc())
            self.save(self.folder / 'progress-output-broken-pipe.json', record)
            if self.output is sys.stdout:
                sys.stdout = open(os.devnull, 'w')
            self.output = None


class ObservedResponse:
    def __init__(self, response, phase):
        self.response, self.phase = response, phase
        self.headers, self.status = response.headers, response.status

    def geturl(self):
        return self.response.geturl()

    def __enter__(self):
        self.phase['value'] = 'filesystem_open'
        return self

    def read(self, size):
        self.phase['value'] = 'http_read'
        chunk = self.response.read(size)
        self.phase['value'] = 'filesystem_write' if chunk else 'filesystem_flush'
        return chunk

    def __exit__(self, kind, error, tb):
        # Preserve a read/write failure's phase; only a new close failure changes it.
        try:
            self.response.close()
        except Exception:
            if error is not None:
                error.add_note('Secondary response close failure: ' + traceback.format_exc())
                return False
            self.phase['value'] = 'http_close'
            raise
        return False


def execute(root, d, p, old, all_entries, declared_sha):
    require(d['execution_released'] is True and d['scope_frozen'] is True, 'recovery_not_released')
    with original_gate(root, p, old):
        items = validate_terminal_release(root, d, p)
        base, source, out = root / RECOVERY, root / ORIGINAL / 'source', root / ORIGINAL / 'verified'
        base.mkdir(exist_ok=True)
        with (base / 'queue.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sys.path.insert(0, str(source))
            import acquire_public_case as transport
            import acquire_btc_case as guards
            import real_intake_io as io
            for module in (transport, guards, io):
                require(Path(module.__file__).resolve().parent == source, 'unexpected_helper_import')
            save = lambda path, value: io.atomic_preserve(path, (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode())
            run = base / 'runs' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-' + uuid.uuid4().hex[:8])
            run.mkdir(parents=True)
            progress = Progress(save, run)
            ctx = ssl.create_default_context()
            require(ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname, 'TLS_verification_disabled')
            opener = build_opener(guards.RejectRedirects(), HTTPSHandler(context=ctx))

            def storage_check():
                remaining = 0
                for e in all_entries:
                    final, partial = out / e['path'], out / (e['path'] + '.partial')
                    fs, ps = file_stat(final), file_stat(partial)
                    require(not fs or fs['size'] == e['bytes'], 'existing_final_size_changed')
                    require(not ps or ps['size'] <= e['bytes'], 'oversized_partial')
                    remaining += 0 if fs else e['bytes'] - (ps['size'] if ps else 0)
                require(shutil.disk_usage(base).free - remaining - ALLOWANCE >= RESERVE, 'storage_reserve_refusal')
                return remaining

            def process(item):
                e = item['entry']
                journal = base / 'attempts' / key(e)
                # An existing intent, result, or abandoned journal is never retried.
                if journal.exists():
                    result = journal / 'result.json'
                    if result.exists():
                        previous = json.loads(result.read_text())
                        require(previous['declaration_sha256'] == declared_sha and previous['path'] == e['path'], 'recovery_journal_identity_changed')
                        return dict(path=e['path'], status='prior_recovery_record_no_network', prior_status=previous['status'], original_attempts=item['original_attempts_used'])
                    return dict(path=e['path'], status='orphan_recovery_journal_no_network')
                check_history(root, item)
                check_partial(root, item)
                storage_check()
                journal.mkdir(parents=True)
                phase, events = {'value': 'journal'}, []
                offset = item['partial_stat']['size'] if item['partial_stat'] else 0
                final, partial = out / e['path'], out / (e['path'] + '.partial')
                consumed = False

                def record(path, value):
                    previous = phase['value']
                    phase['value'] = 'journal'
                    save(path, value)
                    phase['value'] = previous

                def verify(path, entry):
                    phase['value'] = 'verify'
                    guards.verify_file(path, entry)
                    if entry['expected_git_blob_sha1'] is not None:
                        h = hashlib.sha1(('blob ' + str(entry['bytes']) + '\0').encode())
                        with path.open('rb') as f:
                            while chunk := f.read(1024**2):
                                h.update(chunk)
                        require(h.hexdigest() == entry['expected_git_blob_sha1'], 'pinned_release_git_blob_mismatch')
                    phase['value'] = 'filesystem_publish'

                transport.verify_file = verify

                def observed(request, *, timeout):
                    require(request.full_url == e['source_url'] and request.get_method() in ('HEAD', 'GET'), 'request_scope_changed')
                    request.add_header('If-Match', e['etag_opaque'])
                    method = request.get_method()
                    event = dict(method=method, url=request.full_url, range=request.get_header('Range'), utc=utc(), tls_verified=True)
                    phase['value'] = 'http_open'
                    try:
                        response = opener.open(request, timeout=timeout)
                    except Exception as error:
                        event.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
                        if isinstance(error, HTTPError):
                            event['status'] = error.code
                        events.append(event)
                        record(journal / (method.lower() + '-http.json'), event)
                        raise
                    h = response.headers
                    event.update(status=response.status, effective_url=response.geturl(), headers={k: h.get(k) for k in ('Content-Length', 'Content-Range', 'Content-Encoding', 'ETag', 'x-amz-version-id')})
                    events.append(event)
                    try:
                        record(journal / (method.lower() + '-http.json'), event)
                        phase['value'] = 'source_identity'
                        expected_offset = offset if method == 'GET' else 0
                        require(response.geturl() == e['source_url'] and response.status == (206 if expected_offset else 200), 'source_url_or_resume_status_mismatch')
                        require(h.get('ETag') == e['etag_opaque'] and h.get('x-amz-version-id') == e['s3_version_id'], 'source_version_or_etag_mismatch')
                        require(h.get('Content-Length') == str(e['bytes'] - expected_offset) and h.get('Content-Encoding', 'identity') == 'identity', 'source_length_or_encoding_mismatch')
                        expected_range = f"bytes {offset}-{e['bytes']-1}/{e['bytes']}" if expected_offset else None
                        require(h.get('Content-Range') == expected_range, 'source_content_range_mismatch')
                    except Exception as primary:
                        try:
                            response.close()
                        except Exception:
                            primary.add_note('Secondary response close failure: ' + traceback.format_exc())
                        raise
                    return ObservedResponse(response, phase)

                try:
                    record(journal / 'scope.json', dict(entry=e, declaration_sha256=declared_sha, original_attempts=item['original_attempts_used'], lifetime_ceiling=item['original_attempts_used'] + 1, partial_stat=item['partial_stat'], started_utc=utc()))
                    # Required metadata refresh happens before reserving the one GET.
                    with observed(Request(e['source_url'], method='HEAD', headers={'Accept-Encoding': 'identity'}), timeout=45):
                        pass
                    record(journal / 'payload-intent.json', dict(entry=e, declaration_sha256=declared_sha, attempt=item['original_attempts_used'] + 1, additional_attempt=1, resume_offset=offset, tls_verified=True, started_utc=utc()))
                    consumed = True
                    phase['value'] = 'filesystem_open'
                    state = transport.acquire_file(e, out, opener=observed)
                    phase['value'] = 'local_hash'
                    h, m = hashlib.sha256(), hashlib.md5()
                    with final.open('rb') as f:
                        while chunk := f.read(1024**2):
                            h.update(chunk)
                            m.update(chunk)
                    require(m.hexdigest() == e['expected_md5'] and final.stat().st_size == e['bytes'], 'local_fixity_failure')
                    result = dict(path=e['path'], status='byte_verified', bytes=e['bytes'], sha256=h.hexdigest(), source_md5=m.hexdigest(), release_git_blob_sha1=e['expected_git_blob_sha1'], transport_status=state, publication_stat=file_stat(final))
                except Exception as error:
                    # No retry loop exists. This label records cause, not future permission.
                    transient = transient_error(error, phase['value'])
                    ps = file_stat(partial)
                    truncated = phase['value'] == 'verify' and isinstance(error, guards.AcquisitionError) and str(error).startswith('Pending source size/checksum contract mismatch:') and ps is not None and ps['size'] < e['bytes'] and any(v.get('method') == 'GET' and v.get('status') in (200, 206) for v in events)
                    result = dict(path=e['path'], status='transport_attempt_failed_budget_closed' if transient or truncated else 'integrity_scope_tls_or_filesystem_refusal', phase=phase['value'], error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), partial_stat=ps)
                result.update(declaration_sha256=declared_sha, original_attempts=item['original_attempts_used'], additional_attempt_reserved=int(consumed), lifetime_attempts_reserved=item['original_attempts_used'] + int(consumed), payload_requests=sum(v['method'] == 'GET' for v in events), metadata_requests=sum(v['method'] == 'HEAD' for v in events), resume_offset=offset, finished_utc=utc())
                record(journal / 'result.json', result)
                return result

            try:
                remaining = storage_check()
                save(run / 'started.json', dict(pid=os.getpid(), pgid=os.getpgrp(), declaration_sha256=declared_sha, files=len(items), workers=1, remaining_queue_bytes=remaining, started_utc=utc()))
                results = []
                with io.termination_cleanup():
                    for item in items:
                        result = process(item)
                        results.append(result)
                        progress.emit(dict(event='progress', finished=len(results), total=len(items), path=result['path'], status=result['status']))
                completion = dict(status='all_selected_bytes_verified' if all(r['status'] == 'byte_verified' for r in results) else 'completed_with_unresolved_files', files=results, declaration_sha256=declared_sha, original_terminal=d['original_terminal'], finished_utc=utc(), all_payloads_unreviewed=True, training_admitted=False, decoded_array_bytes=0)
                save(run / 'completion.json', completion)
                progress.emit(dict(event='completed', status=completion['status'], receipt=str(run / 'completion.json')))
                return completion
            except BaseException as error:
                save(run / 'fatal.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), utc=utc(), status='stopped_no_automatic_retry'))
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--declaration-sha256', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--freeze-release', type=Path)
    args = parser.parse_args()
    d, p, old, entries = load_contract(ROOT, args.declaration, args.declaration_sha256)
    if args.freeze_release:
        require(d['execution_released'] is False and d['scope_frozen'] is False, 'freeze_requires_preserved_false_template')
        require(d['original_terminal'] is None and d['selected_paths'] is None and d['excluded_original_successes'] is None, 'template_not_empty')
        with original_gate(ROOT, p, old):
            receipt, selected, excluded = terminal_selection(ROOT, p)
            for item in p['objects']:
                if item['entry']['path'] in selected:
                    check_partial(ROOT, item)
            released = dict(d, execution_released=True, scope_frozen=True, original_terminal=receipt, selected_paths=selected, excluded_original_successes=excluded)
            # Exclusive creation only; the prepared template is never overwritten.
            with args.freeze_release.open('x') as f:
                json.dump(released, f, indent=2, sort_keys=True)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            print(json.dumps(dict(status='separate_release_frozen_no_network', path=str(args.freeze_release), sha256=sha(args.freeze_release), selected=len(selected), excluded=len(excluded))))
        return
    if args.check_only:
        if d['execution_released']:
            with original_gate(ROOT, p, old):
                validate_terminal_release(ROOT, d, p)
        print(json.dumps(dict(status='contract_valid_no_execution', execution_released=d['execution_released'], original_terminal_bound=d['original_terminal'] is not None, network_requests=0, payload_bytes_read=0)))
        return
    execute(ROOT, d, p, old, entries, args.declaration_sha256)


if __name__ == '__main__':
    main()
