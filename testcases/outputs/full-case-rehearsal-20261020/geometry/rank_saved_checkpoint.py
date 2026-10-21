"""Read-only ranking proof from source and saved records; no engine/replay."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path.cwd() / 'src'))
from resectionlab.full_case_geometry import rank_remaining_accesses
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace

path = Path('testcases/outputs/full-case-rehearsal-20261020/primary-v2/primary.ressectionlab')
source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
_, artifacts, _ = load_workspace(path)
toolbox = artifacts['surgical_toolbox']
snapshot = toolbox['branches'][toolbox['branchId']]
metadata = toolbox['fullCase'][toolbox['branchId']]
scene = load_environment(path)
labels_hash = hashlib.sha256(scene.labels.tobytes()).hexdigest()
state = copy.deepcopy(snapshot['checkpoint']['state'])
view = SimpleNamespace(scene=scene, state=state, state_id=snapshot['stateId'])
accesses = copy.deepcopy(metadata['accesses'])
original = copy.deepcopy(accesses)
start = metadata['accessIndex'] + 1  # Preserve current access and visited prefix.
ranked = rank_remaining_accesses(view, accesses, start)
assert accesses == original and state == snapshot['checkpoint']['state']
assert ranked[:start] == accesses[:start]
geometry_preserved = all(
    {k: v for k, v in row.items() if k not in ('remainingGeometricCapacityCells', 'capacityStateId')}
    == {k: v for k, v in next(a for a in accesses if a['id'] == row['id']).items()
        if k not in ('remainingGeometricCapacityCells', 'capacityStateId')}
    for row in ranked)
assert geometry_preserved
assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
assert hashlib.sha256(scene.labels.tobytes()).hexdigest() == labels_hash
report = {
    'readOnly': True, 'noEngineConstructionReplayOrPreview': True,
    'savedPath': str(path), 'savedSha256': source_hash,
    'stateId': view.state_id, 'position': len(snapshot['actions']),
    'startIndex': start, 'preservedPrefix': True, 'inputRecordsUnchanged': True,
    'routeGeometryUnchanged': geometry_preserved, 'savedFileUnchanged': True,
    'sourceSceneId': scene.identity,
    'sourceLabelsSha256': labels_hash, 'sourceLabelsUnchanged': True,
    'topRemainingAccesses': [{k: row[k] for k in ('id', 'entryPose', 'drillRadiusMm',
        'remainingGeometricCapacityCells', 'capacityStateId')} for row in ranked[start:start + 8]],
    'originalTailOrder': [row['id'] for row in accesses[start:]],
    'rankedTailOrder': [row['id'] for row in ranked[start:]],
    'limits': 'Optimistic retained original target column capacity only; no claim of collision admission or actual physical removal. Public preview and guards remain authoritative.'}
output = Path('testcases/outputs/full-case-rehearsal-20261020/geometry/rank-saved316-proof.json')
output.write_text(json.dumps(report, indent=2))
print(json.dumps({k: report[k] for k in ('position', 'stateId', 'startIndex', 'topRemainingAccesses')}, indent=2))
