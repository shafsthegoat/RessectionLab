"""Explicit future test command; do not run during an owned scientific slot."""
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
    MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
    PYTHONDONTWRITEBYTECODE='1')
code = '''
import sys
sys.path[:0] = [sys.argv[1]+'/src', sys.argv[1]+'/tests']
import resectionlab
import resectionlab.patient_planning_cohort_spec
import resectionlab.patient_teacher_trace_cache
import resectionlab.patient_planning_cohort_sequential
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider', sys.argv[1]+'/tests/test_patient_teacher_trace_cache.py']))
'''
with tempfile.TemporaryDirectory(prefix='patient-trace-cache-') as directory:
    isolated = Path(directory)
    shutil.copytree(ROOT/'src/resectionlab', isolated/'src/resectionlab',
        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for source in (HERE/'stage/src/resectionlab').glob('*.py'):
        shutil.copy2(source, isolated/'src/resectionlab'/source.name)
    (isolated/'tests').mkdir()
    for name in ('test_patient_cohort_sequential.py', 'test_patient_planning_learning.py',
                 'test_patient_planning_admission.py'):
        shutil.copy2(ROOT/'tests'/name, isolated/'tests'/name)
    shutil.copy2(HERE/'stage/tests/test_patient_teacher_trace_cache.py', isolated/'tests')
    (isolated/'manifests/experiments').mkdir(parents=True)
    shutil.copy2(ROOT/'manifests/experiments/remind-component-cohort-v1.json', isolated/'manifests/experiments')
    started = time.monotonic()
    with (HERE/'generated-controls.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-c', code, str(isolated)],
            cwd=isolated, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            status = child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            status = child.wait(timeout=5)
    elapsed = time.monotonic()-started
(HERE/'generated-controls.json').write_text(json.dumps({
    'status': 'pass' if status == 0 and not timed_out and elapsed < 30 else 'fail',
    'exit_code': status, 'timed_out': timed_out, 'elapsed_seconds': elapsed,
    'one_thread': True, 'generated_only': True, 'patient_reads': 0,
    'scope': 'small generated interface/native/tensor controls; no scientific training'}, indent=2)+'\n')
raise SystemExit(0 if status == 0 and not timed_out and elapsed < 30 else 1)
