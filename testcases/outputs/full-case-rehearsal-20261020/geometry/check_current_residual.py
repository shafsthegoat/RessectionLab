"""Read-only residual/endpoint inspection; never constructs or replays an engine.

Usage: check_current_residual.py [--native saved.ressectionlab]
                                [--output receipt.json|directory]
Existing receipts are preserved using a numeric suffix on name conflicts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import grid_binding
from resectionlab.workspace_bundle import load_workspace

OUTPUT_FOLDER = Path(__file__).resolve().parent
DEFAULT_NATIVE = OUTPUT_FOLDER.parent / 'primary-v2/primary.ressectionlab'
NAMES = {2: 'superior', 0: 'left', 1: 'anterior'}


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def load_stable_records(path):
    """Read an unchanged atomic native file and verify source/checkpoint hashes.

    Retry at most three times if a concurrent save replaces the path. This
    validates saved content identity, not the execution of historical actions.
    """
    for attempt in range(1, 4):
        before = _signature(path)
        _, artifacts, _ = load_workspace(path)
        scene = load_environment(path)
        native_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if _signature(path) == before:
            break
    else:
        raise ValueError('Native changed during three reads; no receipt written')
    toolbox = artifacts['surgical_toolbox']
    branch = toolbox['branchId']
    snapshot = toolbox['branches'][branch]
    meta = toolbox['fullCase'][branch]
    checkpoint = snapshot['checkpoint']
    if snapshot['sceneId'] != scene.identity or meta['sourceSceneId'] != scene.identity:
        raise ValueError('Snapshot/director/source scene identity differs')
    if checkpoint['position'] != len(snapshot['actions']) or checkpoint['stateId'] != snapshot['stateId']:
        raise ValueError('Checkpoint action boundary or state identity differs')
    contract = snapshot.get('protectionContract')
    if contract is not None:
        body = {key: value for key, value in contract.items() if key != 'contentHash'}
        if (semantic_digest(body) != contract['contentHash'] or contract['sceneId'] != scene.identity
                or contract['sourceBinding'] != grid_binding(scene)):
            raise ValueError('Protection contract identity or source binding differs')
    expected = semantic_digest({'scene': scene.identity, 'state': checkpoint['state'],
                                **({'protectionContract': contract['contentHash']} if contract else {})})
    if expected != snapshot['stateId'] or meta.get('lastStateId', expected) != expected:
        raise ValueError('Checkpoint state content or director boundary differs')
    if not np.allclose(scene.affine[:3, :3], np.diag(np.diag(scene.affine[:3, :3]))):
        raise ValueError('This bounded inspector requires an axis-aligned source grid')
    identity = {'stableAtomicNativeRead': True, 'readAttempts': attempt,
                'nativeSha256': native_hash, 'sourceLabelsDigest': array_digest(scene.labels),
                'checkpointContentIdentityVerified': True,
                'protectionContractId': contract['contentHash'] if contract else None,
                'sourceBinding': grid_binding(scene), 'branchId': branch,
                'actionReplayVerifiedByThisInspector': False}
    return scene, snapshot, meta, identity


def inspect_records(scene, snapshot, meta):
    state = snapshot['checkpoint']['state']
    removed = np.asarray(state['removed'], dtype=np.int64)
    if removed.ndim != 1 or np.any(removed < 0) or np.any(removed >= scene.labels.size):
        raise ValueError('Removed identities must belong to the original grid')
    targets = [r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor']
    bones = [r['id'] for r in scene.regions if r['material'] == 'bone']
    target = np.isin(scene.labels, targets)
    remaining = target.copy(); remaining.ravel()[removed] = False
    components, count = ndi.label(remaining, structure=ndi.generate_binary_structure(3, 1))
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    nonempty = np.flatnonzero(sizes)
    largest = nonempty[np.argsort(-sizes[nonempty], kind='stable')[:10]]
    rows = []; masks = {}
    for axis in (2, 0, 1):
        source_view = np.moveaxis(scene.labels, axis, -1)
        retained = np.moveaxis(remaining, axis, -1)
        target_view = np.moveaxis(target, axis, -1)
        normal = ~((source_view == 0) | target_view | np.isin(source_view, bones))
        k = np.arange(source_view.shape[-1])
        bad_top = np.max(np.where(normal, k, -1), axis=2)
        potential = retained & (k[None, None, :] > bad_top[:, :, None])
        window = potential.any(axis=2)
        transverse = [a for a in range(3) if a != axis]
        clear = ndi.distance_transform_edt(window, sampling=scene.spacing[transverse])
        masks[axis] = np.moveaxis(potential, -1, axis)
        cells = int(potential.sum())
        rows.append({'direction': NAMES[axis], 'nativeAxis': axis, 'outwardIndexSign': 1,
                     'outwardRAS': (scene.affine[:3, axis]/scene.spacing[axis]).tolist(),
                     'retainedPointColumnTargetCells': cells,
                     'retainedPointColumnTargetMm3': cells*scene.voxel_mm3,
                     'retainedWindowColumns': int(window.sum()),
                     'maxLateralWindowClearanceMm': float(clear.max()),
                     'retainedCapacityWith3MmLateralInset': int(potential[clear > 3.].sum())})
    accesses = meta.get('accesses', [])
    access_index = int(meta.get('accessIndex', 0))
    if access_index < 0:
        raise ValueError('Negative configured access cursor')
    configured_axis = meta.get('accessAxis', accesses[0].get('nativeAxis', 2) if accesses else 2)
    if configured_axis not in NAMES:
        raise ValueError('Unsupported configured native axis')
    tail = accesses[min(access_index, len(accesses)):]
    capacity_union = np.zeros(scene.shape, dtype=bool)
    search_union = capacity_union.copy(); access_rows = []
    for access in tail:
        axis = access.get('nativeAxis', configured_axis)
        if axis not in NAMES or access.get('outwardIndexSign', 1) != 1:
            raise ValueError('Only native axes 0/1/2 with outward index +1 are supported')
        if access.get('sourceSceneId', scene.identity) != scene.identity:
            raise ValueError('Configured access source identity differs')
        transverse = [a for a in range(3) if a != axis]
        index = access.get('targetEntryIndex')
        if index is None:
            point = np.asarray(access['entryPose']['position'], dtype=float)
            index = np.rint(scene.inverse[:3, :3] @ point + scene.inverse[:3, 3])
        cells = 0
        if all(np.isfinite(v) and 0 <= v < size for v, size in zip(index, scene.shape)):
            view = np.moveaxis(masks[axis], axis, -1)
            i, j = np.asarray(index)[transverse]
            xx, yy = np.ogrid[:view.shape[0], :view.shape[1]]
            d2 = ((xx-i)*scene.spacing[transverse[0]])**2 + ((yy-j)*scene.spacing[transverse[1]])**2
            radius = float(access['drillRadiusMm'])
            search_radius = float(access.get('searchRadiusMm', radius+2.))
            if not np.isfinite([radius, search_radius]).all() or min(radius, search_radius) <= 0.:
                raise ValueError('Configured access has invalid radius')
            disk = d2 <= max(1., radius-1.)**2
            search = d2 <= search_radius**2
            covered = view & disk[:, :, None]
            capacity_union |= np.moveaxis(covered, -1, axis)
            search_union |= np.moveaxis(view & search[:, :, None], -1, axis)
            cells = int(covered.sum())
        access_rows.append({'id': access['id'], 'nativeAxis': axis,
                            'retainedCapacityDiskCells': cells,
                            'retainedCapacityDiskMm3': cells*scene.voxel_mm3})
    regions = []
    for label in largest:
        member = components == label
        points = np.argwhere(member)
        regions.append({'component': int(label), 'ownedTargetCells': int(sizes[label]),
                        'ownedTargetMm3': int(sizes[label])*scene.voxel_mm3,
                        'nativeBoundsMin': points.min(axis=0).tolist(),
                        'nativeBoundsMax': points.max(axis=0).tolist(),
                        **{NAMES[axis]+'PointColumnCells': int(np.count_nonzero(masks[axis][member]))
                           for axis in (2, 0, 1)}})
    original_cells, remaining_cells = int(target.sum()), int(remaining.sum())
    removed_mm3 = (original_cells-remaining_cells)*scene.voxel_mm3
    pass_start = meta.get('passStartTargetMm3')
    tail_report = {'nativeAxis': configured_axis, 'accesses': len(tail),
                   'positiveCapacityAccesses': sum(r['retainedCapacityDiskCells'] > 0 for r in access_rows),
                   'distinctResidualCapacityDiskCells': int(capacity_union.sum()),
                   'distinctResidualCapacityDiskMm3': int(capacity_union.sum())*scene.voxel_mm3,
                   'distinctResidualHelperSearchDiskCells': int(search_union.sum()),
                   'distinctResidualHelperSearchDiskMm3': int(search_union.sum())*scene.voxel_mm3,
                   'bestRetainedRoutes': sorted(access_rows, key=lambda r: -r['retainedCapacityDiskCells'])[:5]}
    return {'readOnly': True, 'noEngineOrPhysics': True, 'position': len(snapshot['actions']),
            'stateId': snapshot['stateId'], 'sourceSceneId': scene.identity, 'voxelVolumeMm3': scene.voxel_mm3,
            'originalOwnedTargetCells': original_cells, 'originalOwnedTargetMm3': original_cells*scene.voxel_mm3,
            'remainingOwnedTargetCells': remaining_cells, 'remainingOwnedTargetMm3': remaining_cells*scene.voxel_mm3,
            'removedOwnedTargetCells': original_cells-remaining_cells, 'removedOwnedTargetMm3': removed_mm3,
            'connectedRegions6Face': int(count),
            'regionsAtMost8Cells': int(np.count_nonzero((sizes > 0) & (sizes <= 8))),
            'largestRegions': regions, 'axes': rows,
            'alternativeUnionOutsideSuperiorCells': int(np.count_nonzero((masks[0] | masks[1]) & ~masks[2])),
            'anyThreeAxisPointColumnUnionCells': int(np.count_nonzero(masks[2] | masks[0] | masks[1])),
            'currentAccessIndex': access_index, 'configuredAccessCount': len(accesses),
            'currentAccessListExhausted': access_index >= len(accesses),
            'passIndex': meta.get('passIndex', 0), 'passStartTargetMm3': pass_start,
            'targetAddedDuringCurrentPassMm3': (removed_mm3-float(pass_start)) if pass_start is not None else None,
            'currentStage': meta['stage'], 'directorStatus': meta['status'],
            'currentAccessId': tail[0]['id'] if tail else None,
            'configuredTailIncludingCurrent': tail_report,
            **({'configuredSuperiorTailIncludingCurrent': tail_report} if configured_axis == 2 else {}),
            'holds': bool(state.get('holds')),
            'deformationNodes': len(state.get('deformation', {}).get('nodes', {})),
            'limits': 'Six-face grid connectivity and retained original source-identity column/window upper bounds only. No tool/shaft collision, guard, legal drilling, stateful physics, or action replay is verified. Source-native +index rays are called superior/left/anterior for this acquired signed-diagonal frame. Live engine may advance after this saved boundary.'}


def write_new_receipt(report, requested):
    output = Path(requested).expanduser() if requested else OUTPUT_FOLDER
    if output.suffix.lower() != '.json' or output.is_dir():
        output = output / f'current-residual-{report["position"]}-axis{report["configuredTailIncludingCurrent"]["nativeAxis"]}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    base = output; suffix = 1
    while True:
        try:
            with output.open('x') as stream:
                json.dump(report, stream, indent=2)
                stream.write('\n')
            return output
        except FileExistsError:
            suffix += 1
            output = base.with_name(f'{base.stem}-{suffix}{base.suffix}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', type=Path, default=DEFAULT_NATIVE,
                        help='Saved native workspace; read-only, never replayed')
    parser.add_argument('--output', type=Path, help='New receipt JSON path or directory; existing files preserved')
    args = parser.parse_args()
    path = args.native.expanduser().resolve()
    scene, snapshot, meta, identity = load_stable_records(path)
    report = inspect_records(scene, snapshot, meta)
    report.update(nativePath=str(path), identityVerification=identity)
    output = write_new_receipt(report, args.output)
    print(json.dumps(report, indent=2))
    print(output)


if __name__ == '__main__':
    main()
