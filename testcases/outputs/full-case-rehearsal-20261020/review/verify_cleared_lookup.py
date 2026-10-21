"""Read-only ownership equivalence; never restores or steps patient physics."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import numpy as np
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_occupancy import MaterialOccupancy
from resectionlab.workspace_bundle import load_workspace

root = Path(__file__).resolve().parent.parent
native = root / 'primary-v2/checkpoints/primary-before-cleared-lookup.ressectionlab'
if not native.exists():
    os.link(root / 'primary-v2/primary.ressectionlab', native)
case, artifacts, _ = load_workspace(native)
scene = load_environment(native)
record = artifacts['surgical_toolbox']
snapshot = record['branches'][record['branchId']]
state = snapshot['checkpoint']['state']
assert not state['holds'] and not state.get('deformation')
source = subprocess.check_output(
    ['git', 'show', 'af54fbff:src/resectionlab/surgical_occupancy.py'], text=True)
scope = {}
exec(compile(source, 'reviewed-previous-occupancy', 'exec'), scope)
old = scope['MaterialOccupancy'](scene.labels, state['removed'], {})
new = MaterialOccupancy(scene.labels, state['removed'], {})
grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
points = [slot['position'] for slot in state['slots'].values()]
for action in snapshot['actions'][-48:]:
    path = action.get('path', [action['pose']] if 'pose' in action else [])
    if path:
        points.extend(path[index]['position'] for index in sorted({0, len(path)//2, len(path)-1}))
queries = []
for index, point in enumerate(points):
    end = points[index-1] if index else point
    for radius in (.5, 1.3, 6.):
        ijk = capsule_voxel_indices(grid, point, end, radius)
        ids = np.ravel_multi_index(ijk.T, scene.shape)
        np.testing.assert_array_equal(new[ids], old[ids])
        assert new.material_ids(ids) == old.material_ids(ids)
        queries.append(ids)
timings = []
for owner in (old, new):
    start = time.perf_counter()
    for _ in range(10):
        for ids in queries:
            owner.material_ids(ids)
    timings.append(time.perf_counter() - start)
original = Path('testcases/cases/meningioma-sub-06-standard-v2.ressectionlab')
original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
assert original_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
report = {
    'scope': 'Read-only source-cell ownership queries anchored at saved tool poses/path points; no physics replay, action-legality or episode-speed claim.',
    'native': str(native), 'nativeSha256': hashlib.sha256(native.read_bytes()).hexdigest(),
    'originalSha256': original_sha, 'sourceSceneId': scene.identity,
    'actions': len(snapshot['actions']), 'stateId': snapshot['stateId'],
    'clearedIds': len(state['removed']), 'queryCount': len(queries),
    'queryCellSizes': [int(ids.size) for ids in queries],
    'ownershipAndMembershipEqual': True, 'repetitions': 10,
    'previousQueryWallSeconds': timings[0], 'newQueryWallSeconds': timings[1],
    'queryOnlySpeedRatio': timings[0] / timings[1],
    'previousCodeCommit': 'af54fbff', 'previousCodeSha256': hashlib.sha256(source.encode()).hexdigest(),
}
(root / 'review/cleared-lookup-source-equivalence.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key != 'queryCellSizes'}), flush=True)
