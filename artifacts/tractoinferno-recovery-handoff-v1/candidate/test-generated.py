"""Generated-byte, fake-network tests of the candidate; original data stay unopened."""
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch
import contextlib, fcntl, hashlib, io, json, os, runpy, shutil, socket, ssl, subprocess, sys, tempfile

ROOT = Path.cwd()
PREP = ROOT / 'build/tractoinferno-recovery-preparation-v1/adapter-v1'
RUNNER = PREP / 'run-recovery.py'
OUTPUT = PREP / 'generated-controls'
OUTPUT.mkdir(exist_ok=True)
BODY = b'control!'
RESULTS = []
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


class Response(io.BytesIO):
    def __init__(self, body, status, headers, url, error=None):
        super().__init__(body)
        self.status, self.headers, self.url, self.error = status, headers, url, error
    def geturl(self): return self.url
    def read(self, size):
        if self.error: raise self.error
        return super().read(size)


def fixture(case):
    root = Path(tempfile.mkdtemp(prefix=case + '-', dir=OUTPUT))
    ns = runpy.run_path(str(RUNNER), run_name='generated_only')
    g = ns['execute'].__globals__
    base = root / ns['ORIGINAL']
    source = base / 'source'
    source.mkdir(parents=True)
    for name in ('acquire_public_case.py', 'acquire_btc_case.py', 'real_intake_io.py'):
        shutil.copyfile(ROOT / ns['ORIGINAL'] / 'source' / name, source / name)
    (base / 'queue.lock').touch()
    e = dict(path='derivatives/trainset/sub-test/dwi/generated.bin', subject='sub-test', role='TRAIN', bytes=8, size_bytes=8, sha256=None, source_url='https://example.invalid/generated?versionId=one', expected_md5=hashlib.md5(BODY).hexdigest(), expected_git_blob_sha1=None, etag_opaque='"generated"', s3_version_id='one')
    target = base / 'verified' / e['path']
    target.parent.mkdir(parents=True)
    partial = target.with_name(target.name + '.partial')
    if case in ('resume', 'ignored_range', 'wrong_range', 'read_broken_pipe', 'write_broken_pipe', 'changed_partial'):
        partial.write_bytes(BODY[:3])
    item = dict(entry=e, original_attempts_used=3, partial_path=str(partial.relative_to(root)), partial_stat=ns['file_stat'](partial))
    p = dict(original_pid=2147483647, objects=[item])
    d = dict(execution_released=True, scope_frozen=True, original_terminal={'sha256':'generated'}, workers=1)
    old = dict(predecessor_queue_locks=[])
    g['validate_terminal_release'] = lambda *args: [item]
    g['check_history'] = lambda *args: None
    calls = []
    class Opener:
        def open(self, request, timeout):
            assert request.full_url == e['source_url'] and request.get_header('If-match') == e['etag_opaque']
            method = request.get_method()
            calls.append((method, request.get_header('Range')))
            if case == 'head_timeout': raise TimeoutError('generated HEAD timeout')
            if method == 'GET':
                if case == 'open_broken_pipe': raise BrokenPipeError(32, 'Broken pipe')
                if case == 'open_timeout': raise TimeoutError('generated timeout')
                if case == 'tls': raise ssl.SSLCertVerificationError('generated TLS rejection')
                if case == 'wrapped_tls':
                    from urllib.error import URLError
                    raise URLError(ssl.SSLCertVerificationError('generated TLS rejection'))
                if case == 'open_filesystem': raise PermissionError(13, 'Permission denied')
            offset = int(request.get_header('Range').split('=')[1].split('-')[0]) if request.get_header('Range') else 0
            if method == 'HEAD': offset = 0
            if case == 'ignored_range' and method == 'GET': offset = 0
            headers = {'Content-Length':str(8-offset), 'ETag':e['etag_opaque'], 'x-amz-version-id':'one', 'Content-Encoding':'identity'}
            if offset: headers['Content-Range'] = f'bytes {offset}-7/8'
            if method == 'GET':
                if case == 'wrong_etag': headers['ETag'] = 'bad'
                if case == 'wrong_version': headers['x-amz-version-id'] = 'two'
                if case == 'wrong_range': headers['Content-Range'] = 'bytes 2-7/8'
                if case == 'wrong_length': headers['Content-Length'] = '9'
                if case == 'wrong_encoding': headers['Content-Encoding'] = 'gzip'
            body = b'' if method == 'HEAD' else b'badbytes' if case == 'wrong_md5' else BODY[offset:3] if case == 'truncated' else BODY[offset:]
            error = BrokenPipeError(32, 'Broken pipe') if method == 'GET' and case == 'read_broken_pipe' else None
            return Response(body, 206 if offset else 200, headers, 'https://wrong.invalid/' if method == 'GET' and case == 'wrong_url' else e['source_url'], error)
    g['build_opener'] = lambda *args, **kwargs: Opener()
    return root, ns, g, base, e, target, partial, item, p, d, old, calls


cases = ['fresh', 'resume', 'ignored_range', 'wrong_md5', 'wrong_git', 'wrong_etag', 'wrong_version', 'wrong_range', 'wrong_length', 'wrong_encoding', 'wrong_url', 'open_timeout', 'open_broken_pipe', 'read_broken_pipe', 'tls', 'wrapped_tls', 'open_filesystem', 'write_broken_pipe', 'journal_failure', 'truncated', 'head_timeout', 'orphan', 'held_original_lock', 'live_pid', 'false_release', 'changed_partial', 'storage']
for case in cases:
    root, ns, g, base, e, target, partial, item, p, d, old, calls = fixture(case)
    if case == 'wrong_git': e['expected_git_blob_sha1'] = '0'*40
    if case == 'changed_partial': partial.write_bytes(b'xxxx')
    if case == 'live_pid': p['original_pid'] = os.getpid()
    if case == 'false_release': d['execution_released'] = False
    recovery = root / ns['RECOVERY']
    journal = recovery / 'attempts' / ns['key'](e)
    if case == 'orphan':
        journal.mkdir(parents=True)
        (journal / 'payload-intent.json').write_text('{}\n')
    history_before = {str(x.relative_to(base)):x.read_bytes() for x in base.rglob('*.json')}
    original_open = Path.open
    original_io_atomic = None
    held = (base / 'queue.lock').open('r+') if case == 'held_original_lock' else None
    if held: fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def patched_open(path, *args, **kwargs):
        if case == 'write_broken_pipe' and path == partial and args and args[0] == 'ab':
            raise BrokenPipeError(32, 'generated filesystem pipe')
        return original_open(path, *args, **kwargs)
    def once():
        for name in ('acquire_public_case', 'acquire_btc_case', 'real_intake_io'): sys.modules.pop(name, None)
        if case == 'journal_failure':
            sys.path.insert(0, str(base / 'source'))
            import real_intake_io
            actual = real_intake_io.atomic_preserve
            def fail(path, value):
                if path.name == 'get-http.json': raise OSError(28, 'generated journal disk failure')
                return actual(path, value)
            real_intake_io.atomic_preserve = fail
        with contextlib.redirect_stdout(io.StringIO()), patch.object(Path, 'open', patched_open):
            if case == 'storage':
                Disk = namedtuple('Disk', 'total used free')
                with patch.object(shutil, 'disk_usage', lambda p:Disk(10**12,0,100*1024**3)):
                    return ns['execute'](root, d, p, old, [e], 'generated-declaration')
            return ns['execute'](root, d, p, old, [e], 'generated-declaration')
    try:
        result = once()
        assert case not in ('held_original_lock', 'live_pid', 'false_release', 'changed_partial', 'storage'), case
        row = result['files'][0]
        if case in ('fresh', 'resume'):
            assert row['status'] == 'byte_verified' and target.read_bytes() == BODY and not partial.exists()
            assert row['payload_requests'] == 1 and row['lifetime_attempts_reserved'] == 4
            assert row['sha256'] == hashlib.sha256(BODY).hexdigest()
        elif case == 'orphan': assert row['status'] == 'orphan_recovery_journal_no_network' and not calls
        else:
            expected = 'transport_attempt_failed_budget_closed' if case in ('open_timeout', 'open_broken_pipe', 'read_broken_pipe', 'truncated', 'head_timeout') else 'integrity_scope_tls_or_filesystem_refusal'
            assert row['status'] == expected, (case, row)
            assert 'Traceback (most recent call last)' in row['traceback']
            assert row['payload_requests'] == (0 if case == 'head_timeout' else 1)
            assert row['additional_attempt_reserved'] == (0 if case == 'head_timeout' else 1)
            if case == 'write_broken_pipe': assert row['phase'] == 'filesystem_open'
            if case == 'journal_failure': assert row['phase'] == 'journal'
            if case in ('wrong_md5', 'wrong_git', 'truncated'): assert row['phase'] == 'verify'
            if case in ('ignored_range', 'wrong_range', 'read_broken_pipe', 'write_broken_pipe'): assert partial.read_bytes() == BODY[:3]
            assert not target.exists()
        before = {str(x.relative_to(recovery)):x.read_bytes() for x in (recovery / 'attempts').rglob('*.json')}
        n = len(calls)
        again = once()
        assert len(calls) == n, (case, 'network retried')
        assert before == {str(x.relative_to(recovery)):x.read_bytes() for x in (recovery / 'attempts').rglob('*.json')}
    except (RuntimeError, BlockingIOError) as error:
        assert case in ('held_original_lock', 'live_pid', 'false_release', 'changed_partial', 'storage'), (case, str(error))
        assert not calls
    finally:
        if held: held.close()
    assert history_before == {str(x.relative_to(base)):x.read_bytes() for x in base.rglob('*.json')}
    RESULTS.append(dict(case=case, status='pass', fake_calls=calls, actual_network_requests=0))

# Optional output failure cannot enter the object exception/attempt path.
root = Path(tempfile.mkdtemp(prefix='logger-', dir=OUTPUT))
ns = runpy.run_path(str(RUNNER), run_name='logger_test')
class BrokenOutput:
    def write(self, text): raise BrokenPipeError(32, 'generated logger failure')
    def flush(self): pass
save = lambda path, value: path.write_text(json.dumps(value))
progress = ns['Progress'](save, root, BrokenOutput())
progress.emit({'event':'one'})
progress.emit({'event':'two'})
record = json.loads((root / 'progress-output-broken-pipe.json').read_text())
assert record['phase'] == 'progress_output' and not record['object_attempt_consumed']
RESULTS.append(dict(case='logger_broken_pipe_separate_no_attempt', status='pass', actual_network_requests=0))

# Real terminal selection: all 7,622 metadata results required; successes excluded.
root = Path(tempfile.mkdtemp(prefix='terminal-', dir=OUTPUT))
ns = runpy.run_path(str(RUNNER), run_name='terminal_test')
g = ns['terminal_selection'].__globals__
p = {'original_run_id':'generated','original_declaration_sha256':'sha','objects':[{'entry':{'path':'x0'},'original_terminal_status':'transport_exhausted','original_error_type':'TimeoutError'}]}
path = root / ns['ORIGINAL'] / 'runs/generated/completion.json'
path.parent.mkdir(parents=True)
rows = [{'path':f'x{i}', 'status':'byte_verified'} for i in range(7622)]
c = {'status':'completed_with_unresolved_files','run_id':'generated','declaration_sha256':'sha','files':rows}
path.write_text(json.dumps(c))
g['check_history'] = lambda *args: (_ for _ in ()).throw(RuntimeError('must not include success'))
receipt, selected, excluded = ns['terminal_selection'](root, p)
assert selected == [] and excluded == ['x0']
rows[0] = dict(path='x0',status='integrity_or_scope_refusal',error_type='SSLCertVerificationError')
path.write_text(json.dumps(c));g['check_history'] = lambda *args: None
try: ns['terminal_selection'](root, p); raise AssertionError('new TLS failure accepted')
except RuntimeError as error: assert str(error) == 'terminal_failure_changed'
rows.pop();path.write_text(json.dumps(c))
try: ns['terminal_selection'](root, p); raise AssertionError('incomplete terminal accepted')
except RuntimeError as error: assert str(error) == 'incomplete_original_terminal'
RESULTS.append(dict(case='terminal_success_exclusion_changed_failure_incomplete_refusal', status='pass', actual_network_requests=0))

# Phase classification is conservative even when an exception name is familiar.
ns = runpy.run_path(str(RUNNER), run_name='phase_test')
assert ns['transient_error'](BrokenPipeError(), 'http_read')
assert not ns['transient_error'](BrokenPipeError(), 'filesystem_write')
assert not ns['transient_error'](BrokenPipeError(), 'journal')
assert not ns['transient_error'](ssl.SSLError(), 'http_read')
assert not ns['transient_error'](PermissionError(), 'http_open')
RESULTS.append(dict(case='phase_and_TLS_conservative_classification', status='pass', actual_network_requests=0))

# Original-history membership, hashes, count and known error classification remain binding.
root = Path(tempfile.mkdtemp(prefix='history-', dir=OUTPUT))
ns = runpy.run_path(str(RUNNER), run_name='history_test')
prep = root / ns['PREP']; prep.mkdir(parents=True)
shutil.copyfile(ROOT / ns['PREP'] / 'classify.py', prep / 'classify.py')
e = dict(source_url='https://example.invalid/old', bytes=8, expected_md5='a'*32, expected_git_blob_sha1=None, path='generated')
journal = root / ns['ORIGINAL'] / 'attempts' / ns['key'](e)[:2] / ns['key'](e)
journal.mkdir(parents=True)
intent = journal / '01-old-intent.json'; result = journal / '01-old-result.json'
intent.write_text(json.dumps({'resume_offset':0}))
result.write_text(json.dumps({'status':'transport_exhausted','error_type':'TimeoutError','error':'The read operation timed out'}))
item = dict(entry=e, history_records=[{'path':str(x.relative_to(root)),'sha256':sha(x)} for x in [intent,result]], original_attempts_used=1, partial_stat=None, classification='proposed_exhausted_transport_budget_extension')
ns['check_history'](root,item)
result.write_text(result.read_text()+' ')
try: ns['check_history'](root,item); raise AssertionError('changed original receipt accepted')
except RuntimeError as error: assert str(error).startswith('original_history_changed')
result.write_text(result.read_text()[:-1]); (journal/'02-new-intent.json').write_text('{}')
try: ns['check_history'](root,item); raise AssertionError('new orphan intent accepted')
except RuntimeError as error: assert str(error).startswith('original_history_changed')
RESULTS.append(dict(case='original_history_hash_membership_orphan_immutable',status='pass',actual_network_requests=0))

# A secondary network close error cannot reclassify a primary filesystem error.
ns = runpy.run_path(str(RUNNER),run_name='secondary_close_test')
class CloseFailure:
    headers={}; status=200
    def close(self): raise BrokenPipeError(32,'secondary close')
phase={'value':'filesystem_write'}
wrapped=ns['ObservedResponse'](CloseFailure(),phase)
primary=PermissionError(13,'primary write refusal')
assert wrapped.__exit__(PermissionError,primary,None) is False
assert phase['value']=='filesystem_write' and 'Secondary response close failure' in primary.__notes__[0]
assert not ns['transient_error'](primary,phase['value'])
RESULTS.append(dict(case='secondary_close_preserves_primary_filesystem_failure',status='pass',actual_network_requests=0))

receipt = dict(status='pass', runner_sha256=sha(RUNNER), test_sha256=sha(Path(__file__)), controls=RESULTS, actual_network_requests=0, original_payload_bytes_read=0, generated_only=True)
(PREP / 'generated-controls.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps({'status':'pass','controls':len(RESULTS),'receipt':str(PREP / 'generated-controls.json')}))
