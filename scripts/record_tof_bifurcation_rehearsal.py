#!/usr/bin/env python3
"""Cold-check a native TOF engineering episode and prepare hidden replay optics."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from prepare_tof_bifurcation_rehearsal import BRAIN, CURRENT, HDD, UUID, VESSEL, read, sha256, write
from resectionlab.data_root import checked_root
from resectionlab.imaging import load_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_motion import envelopes
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import save_workspace


def optics(scene, source):
    crop_start = np.asarray(source['nativeCropStartIJK'])
    target = scene.affine[:3, :3] @ (np.asarray(source['nativeBranchJunctionIJK']) - crop_start) + scene.affine[:3, 3]
    outward = scene.affine[:3, 2] / scene.spacing[2]
    # Set anatomy's fixed 34° camera to the microscope's physical focus-plane size.
    angle, azimuth, field = 32, -30, 24
    axis_reference = np.array([0., 1., 0.]) if abs(outward[2]) > .95 else np.array([0., 0., 1.])
    right = np.cross(axis_reference, outward)
    right /= np.linalg.norm(right)
    tilt = Rotation.from_rotvec(np.deg2rad(azimuth) * outward).apply(right)
    direction = Rotation.from_rotvec(np.deg2rad(angle) * tilt).apply(outward)
    up = axis_reference - (axis_reference @ direction) * direction
    up /= np.linalg.norm(up)
    position = target + (field / (2 * math.tan(math.radians(17)))) * direction
    camera = {'position': position.tolist(), 'target': target.tolist(), 'up': up.tolist()}
    microscope = {'target': target.tolist(), 'outward': outward.tolist(), 'angleDeg': angle,
                  'azimuthDeg': azimuth, 'workingDistanceMm': 160, 'fieldMm': field,
                  'focusOffsetMm': 0, 'light': 2.35, 'inspectionTransparency': False,
                  'inspectionCutaway': False, 'tissueStyle': 'operative'}
    return camera, microscope


def pose_contacts(engine, action, pose):
    slot = engine.state['slots'][action['slot']]
    parameters = {**engine.parameters(slot), **action.get('parameters', {})}
    output = {'tip': set(), 'shaft': set()}
    for a, b, r, part in envelopes(engine.tools[slot['instrument']], parameters,
                                    pose['position'], pose['quaternion']):
        output['shaft' if part == 'shaft' else 'tip'].update(engine.touched(a, b, r))
    return {key: sorted(indices) for key, indices in output.items()}


def active_support(mechanics):
    return {(site['sourceCell'], cell) for site in mechanics.get('sites', [])
            for cell in site['activeSupportCells']}


def record(source, output, inputs):
    case = load_case(source)
    scene = from_case(case)
    engine = SurgicalEngine(scene)
    start = engine.state_id
    row_labels = inputs['actions']
    if not row_labels:
        raise ValueError('Supply at least one explicitly recorded action')
    rows = []
    for index, row in enumerate(row_labels):
        action = row['action']
        before = engine.state_id
        before_mechanics = engine.view(include_history=False).get('vesselMechanics', {})
        requested_contact = pose_contacts(engine, action, action['pose']) if 'pose' in action and action['operation'] in ('move', 'apply') else {}
        prior_removed = set(engine.state['removed'])
        result = engine.step(action)
        if result['status'] != row.get('status', 'accepted') or result.get('reason') != row.get('reason'):
            raise ValueError(f"Action {index+1} {row['label']}: {result['status']} {result.get('reason')}")
        if engine.state_id == before:
            raise ValueError(f'Action {index+1} did not advance the accepted state')
        view = engine.view(include_history=False)
        motion = view.get('vesselMechanics', {})
        site = next((s for s in motion.get('sites', []) if s['sourceCell'] == inputs.get('contactCell')), None)
        delta = (np.subtract(site['positionMm'], site['restPositionMm']).tolist() if site else None)
        slot_state = view['slots'][action['slot']]
        actual_motion = {'position': slot_state['position'], 'quaternion': slot_state['quaternion']}
        contact = pose_contacts(engine, action, actual_motion) if slot_state['inserted'] else {}
        removed = sorted(set(engine.state['removed']) - prior_removed)
        if any(engine.by_label[int(scene.labels.flat[cell])]['material'] != 'soft_tissue' for cell in removed):
            raise ValueError('This declared episode may modify only the modeled brain exposure')
        after_support = active_support(motion)
        support_loss = sorted(active_support(before_mechanics) - after_support)
        rows.append({
            **row, 'text': row.get('text', row['label']),
            'index': index + 1, 'stateBefore': before, 'stateAfter': engine.state_id,
            'acceptedPose': actual_motion, 'requestedPoseContactsAtPriorState': requested_contact,
            'acceptedPoseContacts': contact, 'compliantRodContact': motion.get('contactedCells', {}),
            'selectedSiteDisplacementRasMm': delta, 'supportPairsRemoved': support_loss,
            'remainingSupportPairs': len(after_support), 'removedModeledBrainCells': removed,
            'collisionGeometryByLabel': motion.get('geometryModeByLabel', {}),
            'elapsedSeconds': view['metrics']['elapsedSeconds'],
        })
    snapshot = engine.snapshot()
    cold = SurgicalEngine.restore(scene, snapshot, force_verify=True)
    if cold.state_id != engine.state_id or [row['stateAfter'] for row in cold.history] != [row['stateAfter'] for row in rows]:
        raise ValueError('Cold verification changed an accepted state')
    if set(engine.state['removed']) != set(inputs.get('supportRemovedCells', engine.state['removed'])):
        raise ValueError('Removed source occupancy differs from recorded input evidence')
    if inputs.get('supportCell') is not None and not any(row['supportPairsRemoved'] for row in rows):
        raise ValueError('Episode did not detach adjacent support from a mapped source branch')
    toolbox = {'mapping': scene.regions, 'branches': {'main': snapshot}, 'branchId': 'main',
               'position': len(rows), 'parents': {'main': None}, 'timeline': {'main': engine.history}}
    metadata = case.metadata['vascular_motion_source']
    camera, microscope = optics(scene, metadata)
    workspace = {'selectedSeriesId': None, 'states': {'primary': {
        'cursor': microscope['target'], 'visibleLayers': dict.fromkeys(case.compartments, True),
        'viewer': {'camera': camera, 'externalCamera': camera, 'microscope': microscope,
                   'layout': '3d-focus', 'showPlane': False, 'showOrientationGuide': False,
                   'smoothSurfaces': True, 'layerOpacity': {BRAIN: .78, VESSEL: 1.},
                   'layerColor': {VESSEL: '#a64c45'}},
    }}}
    output.mkdir(exist_ok=False)
    path = output / 'tof-bifurcation-rehearsal.ressectionlab'
    save_workspace(case, path, sources=[], imaging=workspace, episode=None, selection=None,
                   artifacts={'surgical_toolbox': toolbox})
    receipt = {
        'schema': 'resectionlab.tof-bifurcation-rehearsal.v1', 'sourceFileSha256': sha256(source),
        'caseHash': case.semantic_hash, 'sceneId': scene.identity, 'initialStateId': start,
        'finalStateId': engine.state_id, 'coldReplayed': True, 'contactBranchId': inputs.get('contactBranchId'),
        'contactCell': inputs.get('contactCell'), 'supportCell': inputs.get('supportCell'),
        'retainedNativeVesselCells': sum(int((scene.labels == row['id']).sum()) for row in scene.regions if row['material'] == 'vessel'),
        'modeledBrainRemovedMm3': len(engine.state['removed']) * scene.voxel_mm3, 'actions': rows,
    }
    write(output / 'engine-evidence.json', receipt)
    groups = inputs.get('chapters', [[0, len(rows)-1, 'Research exposure',
                                     'Acquired TOF vessel estimate with modeled local brain tissue']])
    optical_overrides = inputs.get('chapterShots', {})
    camera_keys = {'focusPoint', 'fieldMm', 'angleDeg', 'azimuthDeg'}
    if (set(optical_overrides) - {str(index) for index in range(1, len(groups))}
            or any(set(shot) - camera_keys for shot in optical_overrides.values())):
        raise ValueError('Detail optics may set only the focus, field, or direction after the matched opening')
    chapters = [
        {'first': first, 'last': last, 'title': name, 'phase': name.split()[0].lower(),
         'subtitle': caption, 'caption': caption, 'focusPoint': microscope['target'],
         'fieldMm': microscope['fieldMm'], 'angleDeg': microscope['angleDeg'],
         'azimuthDeg': microscope['azimuthDeg'], **optical_overrides.get(str(index), {})}
        for index, (first, last, name, caption) in enumerate(groups)
    ]
    plan = {'caseHash': case.semantic_hash, 'branchId': 'main', 'sourceFileSha256': sha256(path),
            'actions': rows, 'chapters': chapters, 'stills': list(range(len(rows))),
            'anatomyLabel': metadata['subject'].replace('_', ' ') + ' · Native TOF and modeled brain exposure',
            'overviewTitle': 'Source-linked bifurcation',
            'overviewLabel': 'Acquired TOF vessel estimate in an explicitly modeled brain exposure',
            'presentationSeconds': inputs.get('presentationSeconds', 30),
            'matchedAnatomy': {'camera': camera, 'focusPlaneFieldMm': microscope['fieldMm']}}
    write(output / 'shot-plan.json', plan)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--actions', type=Path)
    parser.add_argument('--root', type=Path, default=HDD)
    args = parser.parse_args()
    root = checked_root(args.root, expected_uuid=UUID)
    directory = root / CURRENT / 'operative-scene'
    paths = [value.resolve() for value in (args.source, args.output, args.actions) if value is not None]
    if not all(path.is_relative_to(directory) for path in paths):
        parser.error('Use the verified campaign operative-scene folder for restricted inputs and outputs')
    inputs = read(args.actions) if args.actions else {
        'actions': [{'label': 'Initialize replay clock', 'action': {
            'slot': 'left', 'operation': 'configure', 'clockRevision': 1, 'duration': .2}}],
    }
    print(record(args.source, args.output, inputs), flush=True)


if __name__ == '__main__':
    main()
