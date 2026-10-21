"""Read-only saved JSON copy kernels, never creates or steps an engine."""
import copy
import hashlib
import json
from pathlib import Path
import statistics
import timeit

from resectionlab.imaging import read_case_artifacts

ROOT = Path(__file__).resolve().parents[4]
PATH = ROOT / 'testcases/outputs/full-case-rehearsal-20261020/primary-v2/checkpoints/primary-over-sixty-percent.ressectionlab'
digest = hashlib.sha256(PATH.read_bytes()).hexdigest()
record = read_case_artifacts(PATH)['surgical_toolbox']
snapshot = record['branches'][record['branchId']]
state = snapshot['checkpoint']['state']
assert hashlib.sha256(PATH.read_bytes()).hexdigest() == digest


def checkpoint(mode='legacy'):
    components = state.get('flapComponents')
    memo = {id(components): components} if components is not None else {}
    names = ('removed', 'debris') if mode == 'minimal' else (
        'removed', 'debris', 'fluid', 'contact', 'thermal', 'materialExposure') if mode == 'all-flat' else ()
    for name in names:
        value = state.get(name)
        if id(value) in memo:
            continue
        if name == 'removed':
            valid = type(value) is list and all(type(item) is int for item in value)
        else:
            valid = type(value) is dict and all(type(k) is str and type(v) in (int, float) for k, v in value.items())
        if valid:
            memo[id(value)] = value.copy()
    return copy.deepcopy(state, memo)


counts = {key: {'type': type(value).__name__, 'length': len(value)}
          for key, value in state.items() if type(value) in (dict, list)}
times = {}
for mode in ('legacy', 'minimal', 'all-flat'):
    clone = checkpoint(mode)
    assert clone == state
    for name in ('removed', 'debris', 'fluid', 'contact', 'thermal', 'materialExposure'):
        assert clone[name] is not state[name]
    elapsed = timeit.repeat(lambda: checkpoint(mode), number=30, repeat=3)
    times[mode] = {'millisecondsPerCopy': statistics.median(elapsed) * 1000 / 30,
                   'samplesSecondsFor30': elapsed}
result = {'scope': 'Saved JSON state kernel only; no live engine, replay or action benchmark.',
          'native': str(PATH), 'nativeSha256': digest, 'actions': len(snapshot['actions']),
          'stateId': snapshot['stateId'], 'counts': counts, 'times': times,
          'minimalSpeedup': times['legacy']['millisecondsPerCopy'] / times['minimal']['millisecondsPerCopy'],
          'allFlatSpeedup': times['legacy']['millisecondsPerCopy'] / times['all-flat']['millisecondsPerCopy'],
          'implementation': 'Exact builtin type guards each call, independent .copy containers seeded into deepcopy memo; existing flapComponents sharing retained.'}
output = Path(__file__).with_name('rollback-copy-kernel-audit.json')
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
