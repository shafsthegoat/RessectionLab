"""One detached-state thermal alternative; no surgical engine or physics replay."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import time
from types import SimpleNamespace

from resectionlab import surgical_drilling_v2 as drill
from resectionlab.imaging import load_case, read_case_artifacts

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020'
NATIVE = BASE / 'primary-v2/primary.ressectionlab'
SOURCE = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
source_sha = sha(SOURCE)
assert source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
implementation = Path(drill.__file__); implementation_sha = sha(implementation)
for _ in range(3):
    native_sha = sha(NATIVE)
    record = read_case_artifacts(NATIVE)['surgical_toolbox']
    if native_sha == sha(NATIVE): break
else:
    raise ValueError('Saved primary changed repeatedly while read')
snapshot = record['branches'][record['branchId']]
base_state = snapshot['checkpoint']['state']
bone = record['physicsProfile']['bone']; drill.validate(bone)
voxel = load_case(SOURCE).voxel_volume_mm3


def proxy(state, profile=bone):
    return SimpleNamespace(state=state, scene=SimpleNamespace(voxel_mm3=voxel,
        properties={'tissuePhysicsV2': {'bone': profile}}))


def alternative(engine, dt, powers):
    p = drill.profile(engine); state = engine.state['drillingV2']
    capacity = p['heatCapacityJPerMm3K'] * engine.scene.voxel_mm3
    tau = p['coolingTimeSeconds']; ambient = p['ambientTemperatureC']
    decay = math.exp(-dt / tau)
    one_minus_decay = -math.expm1(-dt / tau)
    threshold = p['doseThresholdC'] - ambient
    can_shortcut = dt >= 0. and tau > 0. and threshold >= 0.
    for key in sorted(set(state['temperatureC']) | set(powers), key=int):
        old = state['temperatureC'].get(key, ambient) - ambient
        power = powers.get(key, 0.)
        equilibrium = power * tau / capacity
        new = old * decay + equilibrium * one_minus_decay
        state['temperatureC'][key] = ambient + new
        dose = 0. if can_shortcut and power == 0. and old < threshold else drill._dose_integral(old, equilibrium, threshold, tau, dt)
        state['thermalDoseCSeconds'][key] = state['thermalDoseCSeconds'].get(key, 0.) + dose
        state['coolingLossJ'] += max(0., power * dt - capacity * (new - old))


def encoded(state):
    return json.dumps(state, separators=(',', ':'), allow_nan=False).encode()


def float_bits(state):
    digest = hashlib.sha256()
    def walk(value):
        if type(value) is float: digest.update(struct.pack('!d', value))
        elif isinstance(value, dict):
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
    walk(state)
    return digest.hexdigest()


def difference(a, b, path='state'):
    if type(a) is not type(b): return {'path': path, 'original': repr(a), 'candidate': repr(b)}
    if isinstance(a, dict):
        if list(a) != list(b): return {'path': path, 'reason': 'dict key order differs'}
        for key in a:
            result = difference(a[key], b[key], path + '/' + str(key))
            if result: return result
    elif isinstance(a, list):
        if len(a) != len(b): return {'path': path, 'reason': 'list length differs'}
        for i, (x, y) in enumerate(zip(a, b)):
            result = difference(x, y, path + '/' + str(i))
            if result: return result
    elif type(a) is float:
        if struct.pack('!d', a) != struct.pack('!d', b):
            return {'path': path, 'original': a.hex(), 'candidate': b.hex()}
    elif a != b: return {'path': path, 'original': repr(a), 'candidate': repr(b)}
    return None


temperatures = base_state['drillingV2']['temperatureC']
threshold = bone['doseThresholdC'] - bone['ambientTemperatureC']
shortcut_count = sum(t - bone['ambientTemperatureC'] < threshold for t in temperatures.values())
report = {'scope': 'One detached saved-state thermal kernel only; no patient engine construction, replay, action timing or production edit.',
    'native': str(NATIVE), 'nativeSha256AtRead': native_sha, 'actions': len(snapshot['actions']),
    'stateId': snapshot['stateId'], 'sourceSha256': source_sha,
    'implementation': str(implementation), 'implementationSha256': implementation_sha,
    'voxelMm3': voxel, 'boneProfile': bone, 'temperatureCells': len(temperatures),
    'doseCells': len(base_state['drillingV2']['thermalDoseCSeconds']),
    'temperatureRangeC': [min(temperatures.values()), max(temperatures.values())],
    'zeroPowerStrictBelowThresholdCells': shortcut_count,
    'candidate': 'Hoisted identical -expm1(-dt/tau); strict old<threshold zero-power integral bypass only; original arithmetic, assignments and key order retained.',
    'checks': [], 'parityPassed': True}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = BASE / 'review' / f'thermal-kernel-audit-{stamp}.json'


def check(name, state, profile, dt, powers):
    a, b = proxy(copy.deepcopy(state), profile), proxy(copy.deepcopy(state), profile)
    drill._thermal(a, dt, powers); alternative(b, dt, powers)
    delta = difference(a.state, b.state)
    same_json = encoded(a.state) == encoded(b.state)
    same_bits = float_bits(a.state) == float_bits(b.state)
    row = {'name': name, 'dt': dt, 'powerCells': len(powers), 'exactJson': same_json, 'exactFloatBits': same_bits}
    report['checks'].append(row)
    if delta or not same_json or not same_bits:
        report.update(parityPassed=False, failure={**row, 'difference': delta}, timing='Not run after parity failure')
        output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'output': str(output), 'failure': report['failure']}, indent=2))
        raise SystemExit(1)


for dt in (0., 5e-324, 1e-20, 1e-16, .001, .1, 30.):
    check('saved zero-power', base_state, bone, dt, {})
key = next(iter(temperatures))
check('saved existing/new nonzero power', base_state, bone, .1, {key: .1, str(max(map(int, temperatures)) + 1): 5.})

profiles = []
for ambient, dose, tau in ((37., 43., 8.), (0., 0., .3), (-10., -4., 10.), (37., 43., 1e300), (-1e308, 1e308, 8.)):
    p = copy.deepcopy(bone); p.update(ambientTemperatureC=ambient, doseThresholdC=dose, coolingTimeSeconds=tau)
    drill.validate(p); profiles.append(p)
for i, p in enumerate(profiles):
    ambient, dose = p['ambientTemperatureC'], p['doseThresholdC']
    temps = [ambient, dose, math.nextafter(dose, -math.inf), math.nextafter(dose, math.inf)]
    if i == 4: temps = [ambient, math.nextafter(ambient, math.inf), 0.]  # keep output temperatures finite
    small = {'preserved': {'nested': [1, -0., 'unchanged']}, 'drillingV2': {
        'temperatureC': {str(j): t for j, t in enumerate(temps)},
        'thermalDoseCSeconds': {str(j): -0. if j % 2 else 0. for j in range(len(temps))},
        'coolingLossJ': 0., 'workJ': {'99': .25}}}
    for dt in (0., 1e-20, 1e-16, .1, 30.):
        check('boundary profile ' + str(i), small, p, dt, {})
    if i < 3:
        check('nonzero/signed-zero profile ' + str(i), small, p, .1, {'0': -0., '1': .1, '9': 5.})

# Timed regions contain only each thermal call; every call receives an equal
# independent start, with all copying outside the timer.
timings = {}
for name, function in (('original', drill._thermal), ('candidate', alternative)):
    samples = []
    for _ in range(5):
        starts = [proxy(copy.deepcopy(base_state)) for _ in range(8)]
        started = time.perf_counter()
        for start in starts: function(start, .1, {})
        samples.append((time.perf_counter() - started) * 1000. / len(starts))
    timings[name] = {'medianMillisecondsPerCall': statistics.median(samples), 'samplesMillisecondsPerCall': samples}
report['timings'] = timings
report['kernelSpeedup'] = timings['original']['medianMillisecondsPerCall'] / timings['candidate']['medianMillisecondsPerCall']
report['medianMillisecondsSavedPerCall'] = timings['original']['medianMillisecondsPerCall'] - timings['candidate']['medianMillisecondsPerCall']
report['guardNotes'] = ['Equality must use legacy integral; cancellation in primitive(dt) can produce a positive floating residual.',
    'Public wrappers validate finite nonnegative dt; validated profiles provide positive tau/capacity factors and doseThreshold>=ambient.',
    'Equilibrium/new/cooling-loss computation still occurs for every cell, preserving any arithmetic failure and every dictionary assignment.',
    'Extreme finite profile differences may overflow threshold to +inf; strict comparison remains valid; NaN comparison falls back.',
    'No whole-cell omission, temperature/dose clipping, changed key order, dictionary pruning or whole-action speed claim.']
assert sha(SOURCE) == source_sha and sha(implementation) == implementation_sha
report['sourceAndImplementationUnchanged'] = True
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'output': str(output), **{key: report[key] for key in ('actions', 'temperatureCells', 'doseCells', 'temperatureRangeC', 'zeroPowerStrictBelowThresholdCells', 'parityPassed', 'timings', 'kernelSpeedup', 'medianMillisecondsSavedPerCall')}, 'checks': len(report['checks'])}, indent=2))
