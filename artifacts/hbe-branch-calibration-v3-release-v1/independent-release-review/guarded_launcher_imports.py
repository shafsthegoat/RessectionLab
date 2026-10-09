"""Import-origin-only audit: no study preflight, data access or scientific calls."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

root, frozen = (Path(arg).resolve() for arg in sys.argv[1:3])
expected = json.loads(sys.argv[3])
site_packages = root / '.venv/lib/python3.12/site-packages'
assert Path(sys.prefix).resolve() == root / '.venv'
violations = []
opens = []
source_calls = []

def forbidden(message):
    violations.append(message)
    raise RuntimeError(message)

def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(x in mode for x in 'wax+')) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing:
            forbidden('File write attempted: ' + str(path))
        if path.is_relative_to(root):
            if path.is_relative_to(site_packages):
                return  # Third-party environment used by the exact predecessor invocation.
            if (path.parent == frozen / 'scripts/__pycache__' and path.suffix == '.pyc'
                    and not path.exists()):
                return  # Import machinery probes this even with -B; no cache bytes exist.
            if not path.is_relative_to(frozen) or path not in allowed:
                forbidden('Repository read outside exact frozen source closure: ' + str(path))
            opens.append(str(path.relative_to(root)))
    elif event in {'subprocess.Popen', 'os.system', 'os.exec', 'os.posix_spawn', 'socket.connect'}:
        forbidden('Process or network operation attempted: ' + event)

def profile(frame, event, arg):
    if event != 'call':
        return
    filename = frame.f_code.co_filename
    if filename == __file__:
        return
    if filename.startswith(str(root) + os.sep):
        path = Path(filename).resolve()
        if path.is_relative_to(site_packages):
            return
        if path not in allowed:
            forbidden('Execution outside exact frozen source closure: ' + filename)
        name = frame.f_code.co_name
        if name not in {'<module>', '<listcomp>', '<setcomp>', '<dictcomp>', '<genexpr>',
                        'shape_derivatives', 'finite_array', 'Curve', 'Registry', 'BranchReleasedStudy',
                        'ReleasedStudy', 'HexMesh', 'OutputWatch',
                        'HalfHeightReconstruction', 'ScaleReduction'}:
            forbidden('Scientific or operational source function attempted: ' + filename + ':' + name)
        source_calls.append({'path': str(path.relative_to(root)), 'name': name})

allowed = {frozen / 'scripts' / Path(item['path']).name for item in expected.values()}
sys.path[:] = [str(frozen)] + [] + [p for p in sys.path if p and Path(p).resolve() != Path(__file__).resolve().parent]
assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)
sys.addaudithook(audit)
sys.setprofile(profile)
result = {'schema': 'hbe-v3-frozen-source-import-audit-v1', 'violations': violations}
try:
    importlib.import_module('scripts.mechanics_hbe_branch_calibration_v3_experiment')
    for item in expected.values():
        importlib.import_module('scripts.' + Path(item['path']).stem)
    actual = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith('scripts.'):
            continue
        path = Path(module.__file__).resolve()
        if path not in allowed:
            forbidden('Imported scripts module outside frozen closure: ' + str(path))
        actual[name] = {'path': str(path.relative_to(root)),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    wanted = {'scripts.' + Path(item['path']).stem: item for item in expected.values()}
    assert actual == wanted, (actual, wanted)
    assert list(sys.modules['scripts'].__path__) == [str(frozen / 'scripts')]
    result.update(status='passed', imported_modules=actual,
                  module_count=len(actual), scripts_namespace_paths=list(sys.modules['scripts'].__path__),
                  source_file_opens=sorted(set(opens)), import_source_calls=source_calls,
                  sys_path=sys.path, interpreter=str(Path(sys.executable).resolve()),
                  invocation_path=sys.executable, environment_prefix=sys.prefix,
                  dependency_environment={name: {'version': sys.modules[name].__version__,
                      'path': str(Path(sys.modules[name].__file__).resolve()),
                      'sha256': hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()}
                      for name in ('numpy', 'scipy')},
                  preflight_called=False, measured_archive_opened=False,
                  measured_or_torsion_member_opened=False, fit_called=False, native_called=False)
except BaseException as error:
    result.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
finally:
    sys.setprofile(None)
print(json.dumps(result, indent=2, sort_keys=True))
if result['status'] != 'passed':
    raise SystemExit(1)
