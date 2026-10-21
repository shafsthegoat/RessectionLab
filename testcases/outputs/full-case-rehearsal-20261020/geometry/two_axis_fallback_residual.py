"""One bounded read-only residual check for prior left/anterior fallback rays."""
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path.cwd() / 'src'))
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace

folder = Path('testcases/outputs/full-case-rehearsal-20261020')
path = folder / 'primary-v2/primary.ressectionlab'
_, artifacts, _ = load_workspace(path)
tb = artifacts['surgical_toolbox']; snapshot = tb['branches'][tb['branchId']]
scene = load_environment(path)
assert scene.identity == snapshot['sceneId']
removed = np.asarray(snapshot['checkpoint']['state']['removed'], dtype=np.int64)
residual = scene.labels.copy(); residual.ravel()[removed] = 0
targets = [r['id'] for r in scene.regions if r.get('target')]
bones = [r['id'] for r in scene.regions if r['material'] == 'bone']

def measure(axis, quaternion):
    view = np.moveaxis(residual, axis, -1)
    transverse = [a for a in range(3) if a != axis]
    k = np.arange(view.shape[-1])
    target = np.isin(view, targets)
    bad = ~((view == 0) | target | np.isin(view, bones))
    bad_top = np.max(np.where(bad, k, -1), axis=2)
    top = np.max(np.where(target, k, -1), axis=2)
    potential = target & (k[None, None, :] > bad_top[:, :, None])
    window = (top >= 0) & (top > bad_top)
    clear = ndi.distance_transform_edt(window, sampling=scene.spacing[transverse])
    at = np.unravel_index(np.argmax(clear), clear.shape)
    index = None; entry = None; outside = None; bone_count = 0
    if window.any():
        index = list(map(int, at)); index.insert(axis, int(top[at]))
        entry = scene.affine[:3, :3] @ index + scene.affine[:3, 3]
        outside_index = list(index); outside_index[axis] = scene.shape[axis] - 1
        outward = scene.affine[:3, axis] / scene.spacing[axis]
        outside = scene.affine[:3, :3] @ outside_index + scene.affine[:3, 3] + outward * 8.
        bone_count = int(np.count_nonzero(np.isin(view[at], bones) & (k > top[at])))
        np.testing.assert_allclose(Rotation.from_quat(quaternion).apply([0, 0, 1]), -outward, atol=1e-12)
    result = {'axis': axis, 'outwardIncreasingIndex': True,
              'remainingPointRayOwnedTargetCells': int(potential.sum()),
              'remainingWindowColumns': int(window.sum()),
              'maxLateralWindowClearanceMm': float(clear.max()),
              'capacityWithWindowInset3Mm': int(potential[window & (clear > 3.)].sum()),
              'bestWindowNativeIndex': index,
              'bestWindowRayRetainedTargetCells': int(potential[at].sum()),
              'remainingOverlyingBoneOnBestRay': bone_count,
              'entryPose': None if entry is None else {'position': entry.tolist(), 'quaternion': quaternion},
              'outsidePose': None if outside is None else {'position': outside.tolist(), 'quaternion': quaternion}}
    return result, np.moveaxis(potential, -1, axis)

sup, sup_mask = measure(2, [1., 0., 0., 0.])
left, left_mask = measure(0, [0., float(np.sqrt(.5)), 0., float(np.sqrt(.5))])
front, front_mask = measure(1, [float(np.sqrt(.5)), 0., 0., float(np.sqrt(.5))])
for row, mask in [(left, left_mask), (front, front_mask)]:
    row['retainedCapacityOutsideCurrentSuperiorPointRays'] = int(np.count_nonzero(mask & ~sup_mask))
report = {'readOnly': True, 'noEnginePhysicsOrSourceMutation': True,
          'stateId': snapshot['stateId'], 'position': len(snapshot['actions']),
          'sourceSceneId': scene.identity,
          'remainingOwnedTargetCells': int(np.isin(residual, targets).sum()),
          'currentDeformationNodeCount': len(snapshot['checkpoint']['state'].get('deformation', {}).get('nodes', {})),
          'currentHolds': bool(snapshot['checkpoint']['state'].get('holds')),
          'superiorReference': sup, 'left': left, 'anterior': front,
          'combinedAlternativeOutsideSuperiorCells': int(np.count_nonzero((left_mask | front_mask) & ~sup_mask)),
          'anteriorAdditionalOutsideSuperiorAndLeftCells': int(np.count_nonzero(front_mask & ~sup_mask & ~left_mask)),
          'limit': 'Point-ray and binary lateral-window upper bounds only. Bone/free may be traversed only after real legal access; normals/vessels block. Every whole-tool movement, working cover and source guard needs public preview. No exact physical admission or successful non-superior execution is claimed.'}
output = folder / f'geometry/two-axis-fallback-residual-{len(snapshot["actions"])}.json'
output.write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2)); print(output)
