"""One generated, single-process, no-model pytest run with a 30-second alarm."""
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
sys.path.insert(0, str(ROOT / 'src'))
sys.dont_write_bytecode = True
paths = sorted((ROOT/'src/resectionlab').glob('*.py')) + [HERE/'test_information_boundary.py', Path(__file__)]
def pins():
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in paths}
before = pins()
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).absolute()
        if any(path.is_relative_to(ROOT/name) for name in ('data', 'outputs', 'sources')):
            raise RuntimeError('Generated audit denied acquired/output read: '+str(path))
        if path.suffix.lower() in ('.pt', '.pth', '.npy', '.npz', '.nii', '.dcm', '.mat'):
            raise RuntimeError('Generated audit denied payload file: '+str(path))
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn'):
        raise RuntimeError('Generated audit denied process launch')
    if event == 'import' and args[0].split('.')[0] in ('torch', 'nibabel', 'pydicom'):
        raise RuntimeError('Generated audit denied model/image import')
sys.addaudithook(guard)
def timeout(*args): raise TimeoutError('30-second generated audit deadline')
signal.signal(signal.SIGALRM, timeout)
signal.alarm(30)
started = time.perf_counter()
status = 1
try:
    import pytest
    status = int(pytest.main(['-q', '-p', 'no:cacheprovider', str(HERE/'test_information_boundary.py'),
                             '--basetemp', str(HERE/'pytest-temp-01')]))
finally:
    signal.alarm(0)
    after = pins()
    record = dict(version='hybrid-information-generated-independent-v1', exit_code=status,
        seconds=time.perf_counter()-started, maxrss_platform_units=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform=sys.platform, source_before=before, source_after=after,
        sources_unchanged=before==after, model_imported='torch' in sys.modules,
        acquired_files_read=False, subprocesses=0,
        scope='tiny generated API controls; no patient or generic hidden-anatomy admission')
    (HERE/'controls-01.json').write_text(json.dumps(record, indent=2, sort_keys=True)+'\n')
    if before != after or 'torch' in sys.modules: status = 2
raise SystemExit(status)
