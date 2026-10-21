"""Read-only saved-path/residual-source decision; no patient engine or physics."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path.cwd() / 'src'))
from resectionlab.full_case_geometry import rank_remaining_accesses, _superior_grid
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace

folder = Path('testcases/outputs/full-case-rehearsal-20261020')
path = folder / 'primary-v2/primary.ressectionlab'
_, artifacts, _ = load_workspace(path)
tb = artifacts['surgical_toolbox']
snapshot = tb['branches'][tb['branchId']]
meta = tb['fullCase'][tb['branchId']]
scene = load_environment(path)
view = SimpleNamespace(scene=scene, state=copy.deepcopy(snapshot['checkpoint']['state']),
                       state_id=snapshot['stateId'])
removed = np.asarray(view.state['removed'], dtype=np.int64)
target, _, bad_top, _, _ = _superior_grid(view)
eligible = target & (np.arange(scene.shape[2])[None, None, :] > bad_top[:, :, None])
eligible.ravel()[removed] = False
retained_target = target.copy(); retained_target.ravel()[removed] = False
column_count = eligible.sum(axis=2)
xx, yy = np.ogrid[:scene.shape[0], :scene.shape[1]]
start = meta['accessIndex'] + 1
ranked = rank_remaining_accesses(view, meta['accesses'], start)
coverage = {}
for name, rows, search in [
    ('allConfiguredCapacityDisks', meta['accesses'], False),
    ('unvisitedCapacityDisks', meta['accesses'][start:], False),
    ('unvisitedHelperSearchDisks', meta['accesses'][start:], True),
]:
    union = np.zeros(scene.shape[:2], dtype=bool)
    for row in rows:
        i, j, _ = row['targetEntryIndex']
        radius = float(row.get('searchRadiusMm', row['drillRadiusMm'] + 2.)) if search else max(1., row['drillRadiusMm'] - 1.)
        union |= ((xx-i)*scene.spacing[0])**2 + ((yy-j)*scene.spacing[1])**2 <= radius**2
    coverage[name] = {'distinctRetainedEligibleTargetCells': int(column_count[union].sum()),
                     'eligibleColumnsCovered': int(np.count_nonzero(union & (column_count > 0)))}
paths = []
outside_z = meta['accesses'][meta['accessIndex']]['outsidePose']['position'][2]
for pos in range(max(1, len(snapshot['actions']) - 19), len(snapshot['actions']) + 1):
    action = snapshot['actions'][pos-1]
    points = [p['position'] for p in action.get('path', [])]
    xyz = np.asarray(points, dtype=float)
    paths.append({'position': pos, 'operation': action['operation'],
                  'instrument': action.get('instrument'), 'workingRadiusMm': action.get('parameters', {}).get('radiusMm'),
                  'pathPoses': len(points),
                  'recordedInteriorTipTravelMm': float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum()) if len(xyz) > 1 else 0.,
                  'visitsOutsidePlane': bool(len(xyz) and np.max(xyz[:, 2]) >= outside_z - 1e-5),
                  'firstRAS': points[0] if points else action.get('pose', {}).get('position'),
                  'lastRAS': points[-1] if points else action.get('pose', {}).get('position')})
logs = []
for line in (folder / 'primary-v2/execution.jsonl').read_text().splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if row.get('executionMode') == 'live-recovery' and row['position'] <= len(snapshot['actions']):
        logs.append(row)
trace = []
for n, row in enumerate(logs):
    previous = logs[n-1] if n else None
    trace.append({'position': row['position'], 'stage': row['stage'], 'phase': row['phase'],
                  'targetRemovedMm3': row['metrics']['targetRemovedMm3'],
                  'boneRemovedMm3': row['metrics']['drillRemovedVolumeMm3'],
                  'targetSincePreviousSlice': row['metrics']['targetRemovedMm3'] - previous['metrics']['targetRemovedMm3'] if previous else None,
                  'secondsSincePreviousSlice': row['wallSeconds'] - previous['wallSeconds'] if previous else None,
                  'cumulativeRecoveryWallSeconds': row['wallSeconds']})
report = {'readOnly': True, 'noPhysicsOrEngineConstruction': True, 'position': len(snapshot['actions']),
          'stateId': snapshot['stateId'], 'sourceSceneId': scene.identity,
          'accessIndex': meta['accessIndex'], 'currentAccessId': meta['accesses'][meta['accessIndex']]['id'],
          'coreAttempts': meta.get('coreAttempts'), 'cycleStartTargetMm3': meta.get('cycleStartTargetMm3'),
          'remainingOwnedTargetCells': int(retained_target.sum()),
          'remainingSuperiorEligibleOwnedTargetCells': int(eligible.sum()),
          'coverage': coverage,
          'remainingAccessCount': len(ranked) - start,
          'remainingPositiveCapacityAccessCount': sum(r['remainingGeometricCapacityCells'] > 0 for r in ranked[start:]),
          'topRemaining': [{k: r[k] for k in ('id', 'entryPose', 'drillRadiusMm', 'remainingGeometricCapacityCells')} for r in ranked[start:start+8]],
          'recentRecordedActions': paths, 'recoveryTrace': trace,
          'methodLimits': ['Retained source-owned identities; current deformation does not reassign source ownership.',
             'Capacity and helper-search disks are geometric upper bounds, not physical admission/removal predictions.',
             'Recorded path travel excludes prior current pose to first explicit waypoint.',
             'Adjacent log differences are wall intervals including proposal/preview/commit/save work; no isolated benchmark.',
             'Source arrays and checkpoint records loaded read-only; live process can advance after this boundary.']}
out = folder / f'geometry/opening-productivity-{len(snapshot["actions"])}.json'
out.write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k not in ('recoveryTrace', 'methodLimits')}, indent=2))
print('saved', out)
