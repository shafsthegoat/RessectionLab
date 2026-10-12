"""Exercise staged components with generated fixtures; no saved patient/weight reads."""
from pathlib import Path
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
STAGE = HERE.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')] + [
    str(STAGE / folder) for folder in ('model-recovery', 'runner-recovery', 'independent-recovery')]
sys.path.append(str(STAGE / 'model'))
archive = (Path(sys.base_prefix) / 'lib' / ('python%d%d.zip' % sys.version_info[:2])).resolve()
temporary = HERE / 'controls-01-tmp'


def generated_only(event, values):
    if event == 'open' and isinstance(values[0], (str, bytes)):
        path = Path(os.path.abspath(os.fsdecode(values[0])))
        mode, flags = values[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if path == archive and not writing:
            return
        if path.suffix.lower() in ('.npy', '.npz', '.psckpt', '.qckpt', '.mat', '.zip', '.nii', '.gz'):
            if not path.resolve().is_relative_to(temporary.resolve()):
                raise PermissionError('Generated controls cannot access saved patient/model payloads')
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise PermissionError('Generated controls do not require network access')


sys.addaudithook(generated_only)
import pytest

tests = [str(p) for folder in ('model-recovery', 'runner-recovery', 'independent-recovery')
         for p in sorted((STAGE / folder).glob('test_*.py'))]
if not all(any(str(STAGE / folder) in path for path in tests)
           for folder in ('model-recovery', 'runner-recovery', 'independent-recovery')):
    raise ValueError('All three prepared component test sets are required')
result = pytest.main(['-q', '-p', 'no:cacheprovider',
    '--basetemp=' + str(temporary / 'pytest'), *tests])
if result:
    raise SystemExit(result)
# Replace this process so the actual isolated worker imports cannot reuse
# pytest's warmed modules or overlap its tensor memory footprint.
sys.stdout.flush()
sys.stderr.flush()
os.execv(sys.executable, [sys.executable, '-I', '-B', '-X',
    'pycache_prefix=' + str(HERE / 'absent-bootstrap-cache'),
    str(STAGE / 'runner-recovery/bootstrap_smoke.py'), '--output', str(HERE / 'bootstrap')])
