"""Configuration-only diagnosis against preserved failed execution source.

Intercept the old simulator constructor to capture its actual NativeResectionConfig.
No simulator or geometry engine is constructed; no transition/gradient occurs.
"""
from dataclasses import asdict, fields
from hashlib import sha256
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow
from resectionlab.imaging import load_case
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine, native_config_from_case
import resectionlab.native_simulation as fixed

repo, frozen = map(Path, sys.argv[1:3])
reference = json.loads((frozen / 'manifests/experiments/procedural-native-to-ucsf-v1.json').read_text())
target = reference['target']
bundle = repo / target['bundle_path']
assert sha256(bundle.read_bytes()).hexdigest() == target['bundle_sha256']
case = load_case(bundle)
assert case.semantic_hash == target['semantic_hash'] and case.planning_hash == target['planning_hash']
with patch.object(NativeResectionEngine, '__init__', side_effect=AssertionError('No engine allowed in configuration diagnosis')):
    direct = native_config_from_case(case, access=AccessWindow(**target['access']))
    with patch.object(fixed, 'NativeSequentialSimulator', side_effect=lambda cfg, *a, **k: cfg) as intercepted:
        prototype = fixed.make_native_patient_simulator(case, candidate_count=4, max_steps=3, max_actions=7)
assert intercepted.call_count == 1 and isinstance(prototype, NativeResectionConfig)
assert prototype.fingerprint == target['native_config_hash']
records = {}
for field in fields(NativeResectionConfig):
    name = field.name
    if name.startswith('_'):
        continue
    before, after = getattr(prototype, name), getattr(direct, name)
    if isinstance(before, np.ndarray):
        same = np.array_equal(before, after) and before.dtype == after.dtype and before.shape == after.shape
        describe = lambda a: {'shape': list(a.shape), 'dtype': str(a.dtype), 'array_digest': array_digest(a)}
        row = {'prototype': describe(before), 'direct': describe(after), 'equal': same,
               'different_element_count': int(np.count_nonzero(before != after)) if before.shape == after.shape else None}
    else:
        if name == 'access':
            before, after = asdict(before), asdict(after)
        elif name == 'tools':
            before, after = [asdict(v) for v in before], [asdict(v) for v in after]
        normalized = lambda v: json.loads(json.dumps(v, default=lambda item: np.asarray(item).tolist(), sort_keys=True))
        before, after = normalized(before), normalized(after)
        row = {'prototype': before, 'direct': after, 'equal': before == after}
    records[name] = row
report = {'status':'configuration_comparison_completed','source_root':str(frozen),
    'bundle_sha256':target['bundle_sha256'],'case_hash':case.semantic_hash,'planning_hash':case.planning_hash,
    'expected_prototype_native_config_hash':target['native_config_hash'],
    'observed_prototype_native_config_hash':prototype.fingerprint,
    'observed_direct_native_config_hash':direct.fingerprint,
    'fields':records,'differing_fields':[key for key,row in records.items() if not row['equal']],
    'simulator_constructors_executed':0,'engine_constructors_executed':0,'transitions':0,'gradient_steps':0,
    'method':'Execute both real configuration builders; intercept only the fixed simulator constructor to return its untouched actual config. No fingerprint is replaced.',
    'numerical_source_sha256':{str(path.relative_to(frozen)):sha256(path.read_bytes()).hexdigest() for path in sorted((frozen/'src/resectionlab').glob('*.py'))}}
output=repo/'artifacts/validation/native-axis-config-diagnosis-v1/comparison.json'
output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(json.dumps({key:report[key] for key in ['observed_prototype_native_config_hash','observed_direct_native_config_hash','differing_fields']},indent=2))
print(json.dumps(report['fields']['tissue_support_provenance'],indent=2))
