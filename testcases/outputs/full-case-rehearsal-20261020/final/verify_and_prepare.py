"""Run required final checks once, serially; never retry a patient replay."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

if not __debug__:
    raise RuntimeError('Optimized Python disables freeze/output assertions; refusing verification')

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.surgical_verified_cache import implementation_fingerprint

freeze = json.loads((HERE / 'artifact-freeze.json').read_text())
native = HERE / 'rehearsal.ressectionlab'
review = HERE.parent / 'review'
progress = HERE / 'verification-progress.json'
assert not progress.exists(), 'Inspect the existing run; no automatic retry'
for key in ('RESSECTIONLAB_VERIFIED_CACHE_MAX_DISK_BYTES', 'RESSECTIONLAB_OBSERVED_CACHE_MAX_DISK_BYTES'):
    assert os.environ.get(key) == '8589934592'

state = {'status': 'running', 'startedUtc': datetime.now(timezone.utc).isoformat(),
         'native': str(native), 'nativeSha256': freeze['nativeSha256'], 'stages': []}

def check_frozen():
    assert hashlib.sha256(native.read_bytes()).hexdigest() == freeze['nativeSha256']
    assert implementation_fingerprint() == freeze['implementationFingerprint']
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == freeze['commit']
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()

def save():
    temp = progress.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(progress)

def run(label, script, arguments, output, receipt):
    check_frozen()
    assert not output.exists(), 'Preserve existing check output: '+str(output)
    if script == 'accept_native.py':
        assert not output.with_suffix('.actions.jsonl').exists(), 'Preserve existing canonical trace'
    row = {'label': label, 'status': 'running', 'startedUtc': datetime.now(timezone.utc).isoformat()}
    state['stages'].append(row); save(); print(json.dumps(row), flush=True)
    try:
        with (HERE / (label+'.stdout.log')).open('x') as log:
            subprocess.run([sys.executable, str(review / script), *map(str, arguments)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(receipt.read_text())
        assert result.get('status', result.get('auditStatus')) == 'passed'
        assert result['nativeSha256'] == freeze['nativeSha256']
        assert result['implementationFingerprint'] == freeze['implementationFingerprint']
        check_frozen()
    except BaseException as error:
        row.update(status='failed', error=repr(error), finishedUtc=datetime.now(timezone.utc).isoformat())
        state['status'] = 'failed'; save(); print(json.dumps(row), flush=True); raise
    row.update(status='passed', finishedUtc=datetime.now(timezone.utc).isoformat(), receipt=str(receipt))
    save(); print(json.dumps(row), flush=True)

for role in ('primary', 'vascular'):
    branch = 'full-'+role; report = HERE / (role+'-canonical.json')
    run(role+'-canonical', 'accept_native.py', [native, branch, '--report', report], report, report)
for role in ('primary', 'vascular'):
    branch = 'full-'+role; output = HERE / (role+'-telemetry')
    run(role+'-telemetry', 'export_observed_telemetry.py',
        [native, branch, output, '--prepare-frames'], output, output/'export-receipt.json')
for role in ('primary', 'vascular'):
    branch = 'full-'+role; report = HERE / (role+'-every-frame.json')
    run(role+'-every-frame', 'verify_every_cached_frame_monitoring.py',
        [native, branch, HERE/(role+'-canonical.json'), report], report, report)
state.update(status='passed', finishedUtc=datetime.now(timezone.utc).isoformat())
save(); print(json.dumps({'status': 'passed', 'guiVideoAndPersistenceStillRequired': True}), flush=True)
