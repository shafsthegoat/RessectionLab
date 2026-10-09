"""Independent generated-only integration verification; preserve every result."""
from pathlib import Path
import difflib
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CANDIDATE = ROOT/'build/private-vascular-evaluator-preparation-v1'
TESTS = [
    'tests/test_private_vascular_evaluation.py',
    str((HERE/'test_independent.py').relative_to(ROOT)),
    'tests/test_research_estimate_planning.py',
    'tests/test_research_estimate_strategy_record.py',
    'tests/test_native_spatial_evaluation.py',
    'tests/test_functional_events.py',
    'tests/test_functional_events_review.py',
]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def snapshot():
    paths = list((ROOT/'src/resectionlab').rglob('*.py'))
    paths += [ROOT/p for p in TESTS] + [Path(__file__), CANDIDATE/'vascular_evaluator.py', CANDIDATE/'test_generated.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}

if __name__ == '__main__':
    implementation_identical = (CANDIDATE/'vascular_evaluator.py').read_bytes() == (ROOT/'src/resectionlab/private_vascular_evaluation.py').read_bytes()
    assert implementation_identical
    old = (CANDIDATE/'test_generated.py').read_text()
    new = (ROOT/'tests/test_private_vascular_evaluation.py').read_text()
    diff = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='frozen/test_generated.py', tofile='tracked/test_private_vascular_evaluation.py'))
    (HERE/'test-relocation.diff').write_text(diff)
    before = snapshot()
    (HERE/'source-before.json').write_text(json.dumps(before, sort_keys=True, indent=2)+'\n')
    command = [sys.executable, '-B', '-m', 'pytest', '-q', *TESTS, '--basetemp', str(HERE/'generated-tests-v1'), '-p', 'no:cacheprovider']
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    started = time.monotonic()
    with (HERE/'pytest-output.txt').open('xb') as stream:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            returncode, error = result.returncode, None
        except subprocess.TimeoutExpired:
            returncode, error = 124, 'focused_generated_regression_60_second_timeout'
    after = snapshot()
    (HERE/'source-after.json').write_text(json.dumps(after, sort_keys=True, indent=2)+'\n')
    receipt = {'schema':'generated-vascular-integration-independent-review-v1', 'command':command,
               'exit_code':returncode, 'error':error, 'elapsed_seconds':time.monotonic()-started,
               'implementation_byte_identical_to_frozen_candidate':implementation_identical,
               'snapshots_equal':before==after, 'snapshot_file_count':len(before),
               'changed_files':[p for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)],
               'source_before_sha256':sha(HERE/'source-before.json'),
               'source_after_sha256':sha(HERE/'source-after.json'),
               'pytest_output_sha256':sha(HERE/'pytest-output.txt'),
               'test_relocation_diff_sha256':sha(HERE/'test-relocation.diff')}
    (HERE/'run-receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))
    raise SystemExit(returncode if before==after else 2)
