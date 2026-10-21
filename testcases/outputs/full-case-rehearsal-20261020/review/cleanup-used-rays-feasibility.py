"""Saved JSON/source geometry only: never construct or step a surgical engine."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from resectionlab.core import array_digest
from resectionlab.geometry import GeometryScene, capsule_voxel_indices
from resectionlab.imaging import read_case_artifacts
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_occupancy import MaterialOccupancy
from resectionlab.surgical_geometry_cache import SurgicalCellCoverCache
from resectionlab.surgical_motion import envelopes, swept_envelopes, pair_conflict
from resectionlab.surgical_tools import TOOLS
from resectionlab.full_case_cleanup import cleanup_stations

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'testcases/outputs/full-case-rehearsal-20261020'
SOURCE = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
NATIVE = BASE / 'primary-v2/primary.ressectionlab'
EXPECTED_SOURCE = 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE) == EXPECTED_SOURCE
for _ in range(3):
    native_hash = sha(NATIVE)
    record = read_case_artifacts(NATIVE)['surgical_toolbox']
    if sha(NATIVE) == native_hash:
        break
else:
    raise ValueError('Saved primary changed during every read')
branch = record['branchId']; snapshot = record['branches'][branch]
state = snapshot['checkpoint']['state']; meta = record['fullCase'][branch]
scene = load_environment(SOURCE)
assert scene.identity == snapshot['sceneId'] == meta['sourceSceneId']
assert not state['holds'] and not state.get('deformation'), 'Sparse unmoved occupancy precondition'
labels_hash, affine_hash = array_digest(scene.labels), array_digest(scene.affine)
owner = MaterialOccupancy(scene.labels, set(state['removed']), {})
grid = GeometryScene(np.zeros(scene.shape, bool), scene.affine)
cover = SurgicalCellCoverCache(grid, scene.identity)
cells = lambda a, b, r: cover.query(grid, a, b, r, kernel=capsule_voxel_indices)
debris = {int(k): float(v) for k, v in state['debris'].items() if v > 0.}
fluid = {int(k): float(v) for k, v in state['fluid'].items() if v > 0.}
mass = lambda ids, field=debris: sum(field.get(int(c), 0.) for c in ids)
pose = lambda p, q: {'position': np.asarray(p).tolist(), 'quaternion': list(q)}
params = {'radiusMm': 3., 'depthMm': 2., 'flowMm3PerSecond': 20., 'forceN': 0.}
frazier = TOOLS['frazier_suction']

# Only actual saved drill actions establish rays. Intended drill endpoints are
# not treated as attained: every candidate independently needs a clear ray now.
tools = {'left': 'cranial_drill', 'right': 'cusa'}
anchors = {}; rays = {}
for index, action in enumerate(snapshot['actions'], 1):
    slot, operation = action['slot'], action['operation']
    if operation == 'exchange':
        tools[slot] = action['instrument']; anchors.pop(slot, None)
    if tools[slot] != 'cranial_drill': continue
    if operation == 'place':
        anchors[slot] = (copy.deepcopy(action['pose']), index)
    if operation != 'apply' or slot not in anchors: continue
    outside, placement = anchors[slot]
    origin = np.asarray(outside['position'], float)
    q = outside['quaternion']; axis = Rotation.from_quat(q).apply([0., 0., 1.])
    for target in ([action['pose']] if 'pose' in action else action.get('path', [])):
        delta = np.asarray(target['position']) - origin
        depth = float(delta @ axis)
        angle = (Rotation.from_quat(target['quaternion']) * Rotation.from_quat(q).inv()).magnitude()
        if depth <= 0 or np.linalg.norm(delta - axis * depth) > 1e-6 or angle > 1e-8: continue
        key = tuple(np.round(origin, 8)) + tuple(np.round(axis, 8))
        ray = rays.setdefault(key, {'outsidePose': outside, 'axis': axis,
            'maximumRecordedRequestedDepthMm': 0., 'drillPlacementActions1Based': [], 'drillApplyActions1Based': []})
        ray['maximumRecordedRequestedDepthMm'] = max(depth, ray['maximumRecordedRequestedDepthMm'])
        if placement not in ray['drillPlacementActions1Based']: ray['drillPlacementActions1Based'].append(placement)
        if index not in ray['drillApplyActions1Based']: ray['drillApplyActions1Based'].append(index)

old = meta['assistantField']; old_origin = np.asarray(old['fieldPose']['position'])
old_axis = -Rotation.from_quat(old['fieldPose']['quaternion']).apply([0., 0., 1.])
proxy = SimpleNamespace(scene=scene, state=state, cells=cells, occupancy=lambda: owner,
                        parameters=lambda slot: {**params, **slot['parameters']})
old_ids = set(); old_stations = cleanup_stations(proxy, old, max_stations=3)
for station in old_stations:
    p = station['pose']['position']
    old_ids.update(map(int, cells(p, p, station['parameters']['radiusMm'])))

left = state['slots']['left']; left_params = {'radiusMm': TOOLS[left['instrument']]['radiusMm'], 'depthMm': 2., 'jawOpeningMm': 2., **left['parameters']}
left_envelopes = envelopes(TOOLS[left['instrument']], left_params, left['position'], left['quaternion'])
start_outside = old['outsidePose']


def exterior_check(outside):
    if (Rotation.from_quat(start_outside['quaternion']) * Rotation.from_quat(outside['quaternion']).inv()).magnitude() > 1e-8:
        return {'clearStaticMaterial': False, 'currentLeftConflict': None, 'reason': 'Different orientation requires a separately checked exterior rotation'}
    start = np.asarray(start_outside['position']); end = np.asarray(outside['position'])
    n = max(1, math.ceil(np.linalg.norm(end - start) / (1. / 3.)))
    blocked = False; conflict = False
    for i in range(n):
        a = pose(start + (end - start) * (i / n), outside['quaternion'])
        b = pose(start + (end - start) * ((i + 1) / n), outside['quaternion'])
        swept = swept_envelopes(frazier, params, a, b)
        if any(owner.material_ids(cells(x, y, r)) for x, y, r, _ in swept): blocked = True
        if pair_conflict(swept, left_envelopes): conflict = True
    return {'clearStaticMaterial': not blocked, 'currentLeftConflict': conflict,
            'fromOutsidePose': start_outside, 'subintervals': n,
            'qualification': 'Exterior translation only, after separately admitted old-ray withdrawal; live guards/other-hand transitions unverified.'}


candidates = []; all_ids = set(); ray_rows = []
for ri, ray in enumerate(rays.values(), 1):
    outside = ray['outsidePose']; origin = np.asarray(outside['position']); axis = ray['axis']
    proximal = origin - 40. * axis
    relocation = exterior_check(outside)
    row = {k: v for k, v in ray.items() if k != 'axis'}
    row.update(id=f'recorded-drill-ray-{ri}', exteriorRelocation=relocation, positiveClearStationCount=0)
    ray_rows.append(row)
    for depth in np.arange(0., ray['maximumRecordedRequestedDepthMm'] + 1e-8, 1.):
        p = origin + axis * depth
        ids = set(map(int, cells(p, p, 3.))); amount = mass(ids)
        if amount <= 1e-9: continue
        if owner.material_ids(cells(p, proximal, .67)): continue
        row['positiveClearStationCount'] += 1; all_ids.update(ids)
        candidates.append({'rayId': row['id'], 'pose': pose(p, outside['quaternion']),
            'outsidePose': outside, 'parameters': params.copy(), 'sourceSceneId': scene.identity,
            'fieldDebrisMm3': amount, 'fieldFluidMm3': mass(ids, fluid),
            'depthFromRecordedOutsideMm': float(depth), 'checkedPhysicalRayRadiusMm': .67,
            'physicalRayBlockedOwnedSourceCells': 0,
            'currentLeftConflictAtStation': pair_conflict(envelopes(frazier, params, p, outside['quaternion']), left_envelopes),
            'exteriorRelocationClearStaticMaterial': relocation['clearStaticMaterial'],
            'exteriorRelocationCurrentLeftConflict': relocation['currentLeftConflict'],
            'drillPlacementActions1Based': ray['drillPlacementActions1Based'],
            'drillApplyActions1Based': ray['drillApplyActions1Based'], '_ids': ids})

# Greedy unique-debris coverage after the old three fields, not a sum of
# overlapping memberships and not a physical aspiration prediction.
covered = set(old_ids); ranked = []
for _ in range(12):
    available = [c for c in candidates if c['exteriorRelocationClearStaticMaterial']]
    if not available: break
    best = max(available, key=lambda c: (mass(c['_ids'] - covered), c['fieldDebrisMm3']))
    gain = mass(best['_ids'] - covered)
    if gain <= 1e-9: break
    candidates.remove(best)
    ids = best.pop('_ids'); covered.update(ids)
    best.update(rank=len(ranked) + 1, newDebrisBeyondOldAndEarlierRankedFieldsMm3=gain,
                maxSingleDwellSeconds=10., maxSingleDwellFlowBudgetMm3=200.,
                requiresPublicPreview=True)
    ranked.append(best)

assert sha(SOURCE) == EXPECTED_SOURCE
assert array_digest(scene.labels) == labels_hash and array_digest(scene.affine) == affine_hash
result = {'schema': 'resectionlab.readonly-used-drill-ray-cleanup-feasibility.v1',
    'createdUtc': datetime.now(timezone.utc).isoformat(), 'native': str(NATIVE), 'nativeSha256AtRead': native_hash,
    'branch': branch, 'actionsAtRead': len(snapshot['actions']), 'stateIdAtRead': snapshot['stateId'],
    'source': str(SOURCE), 'sourceSha256': EXPECTED_SOURCE, 'sceneId': scene.identity,
    'sourceLabelsDigest': labels_hash, 'sourceAffineDigest': affine_hash,
    'noPatientEngineConstructedOrReplayed': True, 'sourceUnchangedAfterRead': True,
    'savedLedger': {'generatedBoneDebrisMm3': state['drillingV2']['removedVolumeMm3'],
        'presentDebrisMm3': state['debrisMm3'], 'aspiratedDebrisMm3': state.get('debrisRemovedMm3', 0.),
        'escapedDebrisMm3': state.get('debrisEscapedMm3', 0.),
        'fluidPresentMm3': sum(fluid.values())},
    'existingThreeOldRayStations': old_stations,
    'oldThreeFieldsUniqueDebrisMm3': mass(old_ids),
    'recordedDrillRayCount': len(rays), 'positiveClearSampledStationCount': sum(r['positiveClearStationCount'] for r in ray_rows),
    'allSampledClearRayFieldsUniqueDebrisMm3': mass(all_ids),
    'allSampledClearRayFieldsAdditionalToOldThreeMm3': mass(all_ids - old_ids),
    'rankedFieldsUniqueDebrisIncludingOldThreeMm3': mass(covered),
    'rankedFieldsAdditionalUniqueDebrisMm3': mass(covered - old_ids),
    'rankedStations': ranked, 'recordedDrillRays': ray_rows,
    'qualifications': ['Read-only saved checkpoint evidence, not independently replayed admission or a future endpoint.',
        '1 mm spaced centers on actual recorded drill command rays only; positive fields independently filtered against current source-minus-removed occupancy.',
        'Requested drill endpoint is not assumed attained; present free 0.67 mm capsule to recorded outside plus proximal 40 mm is the geometry evidence.',
        'Generic 3 mm research suction sphere, 20 mm3/s shared fluid-first flow; at most 200 mm3 flow per 10 s dwell, actual effects may differ.',
        'Field memberships overlap; unique totals use source-grid spatial-cell unions and cannot be added as removal claims.',
        'No protected clearance, withdrawal, exchange, approach, bilateral safety or effect admission; each requires ordinary public preview/step with live guards.',
        'Exterior relocation means sampled conservative swept whole-tool geometry at <=1/3 mm intervals from the old outside anchor; requires prior safe withdrawal and managed left hand.',
        'No new opening, patient source change, clinical force, complete debris clearance or hemostasis claim.']}
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
output = BASE / 'review' / f'cleanup-used-drill-rays-{stamp}.json'
output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'output': str(output), **{k: result[k] for k in ('actionsAtRead', 'stateIdAtRead', 'savedLedger', 'recordedDrillRayCount', 'positiveClearSampledStationCount', 'oldThreeFieldsUniqueDebrisMm3', 'allSampledClearRayFieldsUniqueDebrisMm3', 'allSampledClearRayFieldsAdditionalToOldThreeMm3', 'rankedFieldsAdditionalUniqueDebrisMm3')}, 'rankedStations': [{k: r[k] for k in ('rank', 'rayId', 'pose', 'fieldDebrisMm3', 'newDebrisBeyondOldAndEarlierRankedFieldsMm3', 'currentLeftConflictAtStation', 'exteriorRelocationClearStaticMaterial')} for r in ranked]}, indent=2))
