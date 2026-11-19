"""Compare terminal outcomes of two independently verified, source-bound recordings."""
from __future__ import annotations

import math

import numpy as np

from .patient_anatomy import PatientImagingCollection
from .surgical_comparison_sections import comparison_sections
from .surgical_exact_grid_replay import verify_selected_route
from .surgical_modeled_outcomes import summarize_engine_outcomes
from .surgical_scene import original_target_memberships
from .surgical_study_replay import validate_saved_binding, verify_recording
from .workspace_bundle import load_workspace


def _endpoint(engine):
    outcome = summarize_engine_outcomes(engine)
    sequence = outcome['sequence']
    original = sequence['originalTarget']
    materials = {}
    for row in outcome['nonTargetByLabel']:
        material = row['material']
        materials[material] = materials.get(material, 0.) + row['removedMm3']
    dimensions = sequence['dimensions']
    return {
        'baselineStateId': sequence['baselineStateId'],
        'targetRemovedMm3': dimensions['targetRemovedMm3']['cumulative'],
        'sourceTargetMm3': original.get('sourceVolumeMm3'),
        'residualMm3': original.get('residualMm3'),
        'otherRemovedMm3': dimensions['nonTargetRemovedMm3']['cumulative'],
        'materialRemovalMm3': materials,
        'simulatedSeconds': dimensions['elapsedSeconds']['cumulative'],
        'withdrawn': sequence['withdrawalComplete'],
        'operations': [action['operation'] for action in engine.actions],
    }


def _read_arm(collection, target_id, path, request):
    case, artifacts, _ = load_workspace(path, cancelled=request.cancelled.is_set)
    if 'nativeStudyReplay' in artifacts:
        validate_saved_binding(case, artifacts)
    record = artifacts.get('surgical_toolbox') or {}
    approach = record.get('exactGridReplay', {}).get('recordedApproach') or {}
    if not approach.get('id'):
        raise ValueError('Select a recording that identifies a saved approach on this study')
    endpoint, grid = {}, {}

    def read_endpoint(engine):
        endpoint.update(_endpoint(engine))
        original, = original_target_memberships(engine.scene).values()
        grid.update(shape=engine.scene.shape, affine=engine.scene.affine, original=original,
                    removed=np.asarray(engine.state['removed'], dtype=np.int64))

    receipt, _ = verify_recording(collection, target_id, case, record, approach_id=approach['id'],
                                  check=request.check, on_verified_engine=read_endpoint)
    route = receipt['recordedApproach']
    return {
        'name': route['name'], 'routeId': route['id'], 'radiusMm': route['radiusMm'],
        'lengthMm': math.dist(route['entry'], route['target']), 'entry': route['entry'],
        'target': route['target'], 'generator': receipt.get('actionGenerator'),
        'branchId': receipt['branchId'], 'endpointStateId': receipt['endpointStateId'],
        'actions': receipt['actions'], 'actionsHash': receipt['actionsHash'],
        **endpoint,
    }, receipt, grid


def compare_recordings(collection, target_id, paths, request, progress):
    if len(paths) != 2 or paths[0] == paths[1]:
        raise ValueError('Select two distinct saved engine recordings')
    arms, receipts, grids = [], [], []
    for index, path in enumerate(paths):
        request.check()
        progress(.1 + .45 * index, f'Verifying source images, route and actions for recording {index + 1}')
        arm, receipt, grid = _read_arm(collection, target_id, path, request)
        arms.append(arm)
        receipts.append(receipt)
        grids.append(grid)
    first, second = receipts
    common = ('patientId', 'sourceSnapshotId', 'targetLayerId', 'targetBinding', 'caseHash',
              'sceneId', 'instrumentCatalogHash', 'physicsHash')
    if any(first[k] != second[k] for k in common) or arms[0]['baselineStateId'] != arms[1]['baselineStateId']:
        raise ValueError('The recordings must share the same study, target, engine scene, initial state and instruments')
    if first['actionsHash'] == second['actionsHash']:
        raise ValueError('The selected files contain the same recorded action sequence')
    sections = comparison_sections(collection, arms, grids, request.check)
    # Route edits need not change the acquisition revision while a long replay runs.
    current = PatientImagingCollection(collection.root)
    if current.identity() != first['sourceSnapshotId']:
        raise ValueError('The source study changed during the comparison')
    for receipt in receipts:
        verify_selected_route(current, receipt, receipt['recordedApproach']['id'])
    return {'patientId': first['patientId'], 'sourceSnapshotId': first['sourceSnapshotId'],
            'targetLayerId': target_id, 'arms': arms, 'sections': sections}
