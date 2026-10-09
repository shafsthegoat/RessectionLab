"""One root-released generated profile; never reads patient/model payloads."""
from pathlib import Path
from dataclasses import asdict
import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import stat
import sys
import time
import tracemalloc

PREP = Path(__file__).resolve().parent
ROOT = PREP.parents[1]
SCOPE = 'one_generated_256x256x192_streaming_contact_profile'


def read_regular(path, cap=4*1024**2):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        details = os.fstat(stream.fileno())
        if not stat.S_ISREG(details.st_mode) or details.st_size > cap:
            raise ValueError('Regular bounded source/release file required')
        raw = stream.read(cap+1)
    if len(raw) > cap:
        raise ValueError('Bounded source/release read exceeded')
    return raw


def sha(path):
    return hashlib.sha256(read_regular(path)).hexdigest()


def source_pins():
    pins = {name: sha(PREP/name) for name in ('streaming_contact.py', 'profile_generated.py', 'repository-source-pins.json')}
    inventory = json.loads(read_regular(PREP/'repository-source-pins.json', 65536))
    if not isinstance(inventory, dict) or not 2 <= len(inventory) <= 128:
        raise ValueError('Invalid repository source inventory')
    for name, expected in inventory.items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or not name.startswith('src/resectionlab/') or path.suffix != '.py':
            raise ValueError('Source inventory path refused')
        if sha(ROOT/path) != expected:
            raise ValueError('Repository source pin changed: '+name)
    for required in ('src/resectionlab/independent_geometry_batch.py', 'src/resectionlab/evaluation.py'):
        if required not in inventory:
            raise ValueError('Missing independent geometry source pin')
    return pins, len(inventory)


def verify_loaded_repository_origins():
    """Reject a shadow package while binding every loaded repository module."""
    inventory = json.loads(read_regular(PREP/'repository-source-pins.json', 65536))
    observed = {}
    for name, module in tuple(sys.modules.items()):
        if name != 'resectionlab' and not name.startswith('resectionlab.'):
            continue
        stem = 'src/' + name.replace('.', '/')
        expected = next((path for path in (stem+'.py', stem+'/__init__.py') if path in inventory), None)
        source = getattr(module, '__file__', None)
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        if (expected is None or source is None or origin is None
                or Path(source).resolve() != (ROOT/expected).resolve()
                or Path(origin).resolve() != (ROOT/expected).resolve()
                or sha(Path(source)) != inventory[expected]):
            raise ValueError('Loaded repository origin mismatch: '+name)
        observed[name] = expected
    if 'resectionlab.independent_geometry_batch' not in observed:
        raise ValueError('Independent geometry module was not loaded from pinned repository source')
    return observed


def write_new(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--release', type=Path)
    parser.add_argument('--output-directory', type=Path)
    args = parser.parse_args()
    pins, source_count = source_pins()
    proposal = {'scope': SCOPE, 'source_hashes': pins, 'grid_shape': [256, 256, 192],
                'grid_voxels': 256*256*192, 'spacing_mm': .7, 'capsules': 6,
                'distinct_actions': 3, 'repeated_action': True, 'coverage': 'partial_x_half',
                'cooperative_wall_seconds': 20, 'root_external_supervision_required': True,
                'patient_access': False, 'release': False}
    if args.check_only:
        print(json.dumps(proposal, sort_keys=True)); return
    if args.release is None or args.output_directory is None:
        raise SystemExit('Root release and fresh output directory required')
    release_bytes = read_regular(args.release, 16384)
    release = json.loads(release_bytes)
    if release != {'scope': SCOPE, 'source_hashes': pins, 'root_release': True}:
        raise SystemExit('Exact root release refused before generated evaluation')
    output = args.output_directory.resolve()
    if not output.is_relative_to(ROOT/'build') or output.exists():
        raise SystemExit('Fresh ignored output directory required')
    prefix = output/'unused-pycache'
    if not sys.dont_write_bytecode or sys.pycache_prefix != str(prefix) or prefix.exists() or prefix.is_symlink():
        raise SystemExit('Use -B -X pycache_prefix=<absolute fresh output directory>/unused-pycache')
    output.mkdir(parents=False, exist_ok=False)
    write_new(output/'attempt.json', {**proposal, 'release': True, 'release_sha256': hashlib.sha256(release_bytes).hexdigest(),
              'repository_sources_verified': source_count, 'source_cache_prefix': str(prefix),
              'status': 'reserved_generated_profile'})
    prohibited = []
    def guard(event, arguments):
        if event in ('subprocess.Popen', 'os.system', 'os.exec', 'socket.connect'):
            prohibited.append(event); raise RuntimeError('Prohibited process/network operation')
        if event == 'open' and isinstance(arguments[0], (str, bytes)):
            name = os.fsdecode(arguments[0]).lower()
            if name.endswith(('.nii', '.nii.gz', '.mat', '.tar', '.pt', '.ckpt', '.bin',
                              '.npz', '.npy', '.pkl', '.safetensors', '.dcm', '.h5')):
                prohibited.append(name); raise RuntimeError('Prohibited patient/model payload')
    sys.addaudithook(guard)
    import numpy as np
    spec = importlib.util.spec_from_file_location('profile_streaming_candidate', PREP/'streaming_contact.py')
    s = importlib.util.module_from_spec(spec); sys.modules[spec.name] = s; spec.loader.exec_module(s)
    origins_before = verify_loaded_repository_origins()
    angle = .37
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    affine = np.eye(4); affine[:3, :3] = rotation*.7; affine[:3, 3] = [-60., -90., -67.2]
    reference = s.GeneratedReference(s.Grid((256, 256, 192), affine), 'lattice', 'partial_x_half')
    def cap(action, part, start, end, radius):
        return s.Capsule(action, part, rotation@start+affine[:3, 3], rotation@end+affine[:3, 3], radius)
    caps = []
    for action in ('first', 'repeated'):
        caps.extend((cap(action, 'shaft', np.array([10., 10., -10.]), np.array([140., 140., 103.]), 1.5),
                     cap(action, 'tip', np.array([50., 50., 25.]), np.array([140., 140., 103.]), 2.)))
    caps.extend((cap('crossing', 'shaft', np.array([160., 10., -10.]), np.array([30., 140., 103.]), 1.5),
                 cap('crossing', 'tip', np.array([120., 50., 25.]), np.array([30., 140., 103.]), 2.)))
    capsules = tuple(caps)
    budget = s.Budget(wall_seconds=20.)
    # An arithmetic memory comparison only: never allocate legacy bounding boxes.
    legacy_counts = []
    for c in capsules:
        a, b = [rotation.T@(np.asarray(v)-affine[:3, 3]) for v in (c.start_ras_mm, c.end_ras_mm)]
        lo = np.maximum(np.floor((np.minimum(a, b)-c.radius_mm)/.7-.5), 0)
        hi = np.minimum(np.ceil((np.maximum(a, b)+c.radius_mm)/.7+.5), np.asarray(reference.grid.shape)-1)
        legacy_counts.append(math.prod(int(v) for v in np.maximum(hi-lo+1, 0)))
    started = time.monotonic(); tracemalloc.start()
    try:
        result = s.evaluate_generated_contacts(reference, capsules, budget=budget)
    except s.BudgetExceeded as error:
        result = {'status': 'budget_refused', 'completed': False, 'outcomes': None, 'reason': error.reason,
                  'work': error.work, 'budget': error.budget, 'rejected_charge': error.rejected_charge}
    except Exception as error:
        result = {'status': 'profile_failed', 'completed': False, 'outcomes': None,
                  'error_type': type(error).__name__}
    elapsed = time.monotonic()-started
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    source_unchanged = False
    origins_after = {}
    try:
        origins_after = verify_loaded_repository_origins()
        source_unchanged = source_pins() == (pins, source_count) and not prefix.exists() and not prefix.is_symlink()
    except (OSError, ValueError):
        pass
    if not source_unchanged:
        result = {'status': 'source_or_cache_guard_failed', 'completed': False, 'outcomes': None}
    record = {'scope': SCOPE, 'source_hashes': pins, 'result': result, 'profile_seconds': elapsed,
              'source_bytes_unchanged': source_unchanged, 'repository_sources_verified': source_count,
              'loaded_repository_origins_before': origins_before, 'loaded_repository_origins_after': origins_after,
              'python_version': sys.version, 'numpy_version': np.__version__,
              'source_binding_scope': 'Project Python sources bound before and after; scientific runtime binary closure is not authenticated by this feasibility worker.',
              'tracemalloc_current_bytes': current, 'tracemalloc_peak_bytes': peak,
              'process_peak_rss_bytes': int(rss if sys.platform == 'darwin' else rss*1024),
              'memory_scope': 'Traced allocations cover kernel; process RSS includes interpreter/imports. No hard RSS cap is provided by this worker.',
              'hypothetical_legacy_largest_bounding_box_cells': max(legacy_counts),
              'hypothetical_legacy_four_coordinate_arrays_bytes': max(legacy_counts)*96,
              'legacy_memory_scope': 'Arithmetic estimate for index, centre, lower and upper arrays only; not an executed legacy measurement.',
              'capsule_records': [asdict(c) for c in capsules], 'grid': asdict(reference.grid),
              'prohibited_actions': prohibited, 'root_external_supervision_required': True}
    write_new(output/'profile.json', record)
    print(json.dumps({'status': result['status'], 'profile_seconds': elapsed,
                      'tracemalloc_peak_bytes': peak, 'process_peak_rss_bytes': record['process_peak_rss_bytes']}, sort_keys=True))
    if result['status'] != 'complete_generated_contact_profile' or prohibited:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
