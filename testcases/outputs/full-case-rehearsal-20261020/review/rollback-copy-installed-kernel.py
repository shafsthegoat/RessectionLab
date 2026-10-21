"""Detached saved-state copy parity/timing; never constructs a patient engine."""
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import struct
import timeit

from resectionlab.imaging import read_case_artifacts
from resectionlab.surgical_engine import _microstep_checkpoint

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020'
NATIVE = BASE / 'primary-v2/primary.ressectionlab'
SOURCE = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_sha = sha(SOURCE)
assert source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
for _ in range(3):
    native_sha = sha(NATIVE)
    record = read_case_artifacts(NATIVE)['surgical_toolbox']
    if native_sha == sha(NATIVE): break
else:
    raise ValueError('Saved native changed repeatedly across read')
snapshot = record['branches'][record['branchId']]
state = snapshot['checkpoint']['state']


def frozen_3e0bd07c_checkpoint(state):
    # Literal 3e0bd07c _execute components-memo + deepcopy semantics.
    components = state.get('flapComponents')
    return copy.deepcopy(state, {id(components): components} if components is not None else None)


def float_bits(value):
    digest = hashlib.sha256()
    def visit(value):
        if type(value) is float: digest.update(struct.pack('!d', value))
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
    visit(value)
    return digest.hexdigest()


assert frozen_3e0bd07c_checkpoint is not _microstep_checkpoint
encode = lambda value: json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
initial = encode(state)
legacy, candidate = frozen_3e0bd07c_checkpoint(state), _microstep_checkpoint(state)
assert initial == encode(legacy) == encode(candidate)
assert float_bits(state) == float_bits(legacy) == float_bits(candidate)
for name in ('removed', 'debris'):
    assert legacy[name] is not state[name] and candidate[name] is not state[name]
assert legacy['flapComponents'] is state['flapComponents'] and candidate['flapComponents'] is state['flapComponents']
timings = {}
for name, function in (('frozen3e0Legacy', frozen_3e0bd07c_checkpoint), ('installedCandidate', _microstep_checkpoint)):
    samples = timeit.repeat(lambda: function(state), number=5, repeat=3)
    timings[name] = {'samplesSecondsFor5Copies': samples, 'medianMillisecondsPerCopy': statistics.median(samples) * 1000. / 5.}
assert encode(state) == initial and sha(SOURCE) == source_sha
result = {'scope': 'Uninstrumented copy kernel only on detached JSON state; no patient engine, replay, action/episode benchmark or live-process interaction.',
    'native': str(NATIVE), 'nativeSha256AtStableRead': native_sha, 'branch': record['branchId'],
    'actionsAtRead': len(snapshot['actions']), 'stateIdAtRead': snapshot['stateId'],
    'sourceSha256': source_sha, 'sourceUnchanged': True,
    'counts': {'removed': len(state['removed']), 'debris': len(state['debris']),
        'flapComponents': len(state['flapComponents']),
        'drillTemperature': len(state['drillingV2']['temperatureC']),
        'drillDose': len(state['drillingV2']['thermalDoseCSeconds'])},
    'exactJsonIncludingOrderAndNumericTypes': True, 'exactFloatBits': True,
    'floatBitDigest': float_bits(candidate), 'mutableFieldsIndependentlyOwned': True,
    'existingComponentExemptionPreserved': True,
    'legacyFunctionSourceSha256': hashlib.sha256(inspect.getsource(frozen_3e0bd07c_checkpoint).encode()).hexdigest(),
    'candidateFunctionSourceSha256': hashlib.sha256(inspect.getsource(_microstep_checkpoint).encode()).hexdigest(),
    'timings': timings,
    'speedup': timings['frozen3e0Legacy']['medianMillisecondsPerCopy'] / timings['installedCandidate']['medianMillisecondsPerCopy'],
    'qualification': 'One frozen literal legacy function versus the installed candidate helper. Both receive the same unchanged detached state. Guard scans and copy deallocation are included. Only removed/debris are newly memo-seeded; other fields retain deepcopy and the preexisting flap exemption. Timing is not normal action throughput.'}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = BASE / 'review' / f'rollback-copy-installed-kernel-{stamp}.json'
with output.open('x') as stream: stream.write(json.dumps(result, indent=2) + '\n')
print(json.dumps({'output': str(output), **result}, indent=2))
