#!/usr/bin/env python3
"""Audit exported post-exposure routes without importing their planner.

The existing continuous rigid-geometry checker is the second implementation.
This audit covers the declared static checkpoint, not patient physical validity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.geometry import GeometryScene, SphereObstacle, ToolGeometry, ToolPose, check_route
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import standardize
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace


def _pose(value):
    return ToolPose(value['position'], Rotation.from_quat(value['quaternion']).apply([0., 0., 1.]))


def _program_poses(actions):
    return tuple(_pose(action['pose']) for action in actions if action.get('pose') is not None)


def _read(path):
    value = json.loads(Path(path).read_text())
    # Bridge driver receipts may wrap the exported comparison.
    return value.get('interactivePlanning', value)


def _brief(result):
    return {'feasible': result.feasible, 'clearanceMm': result.clearance_mm,
            'failures': [vars(row) for row in result.failures],
            'unknowns': list(result.unknowns), 'sweptCells': len(result.swept_voxel_indices)}


def certify_route(route, *, shaft, full, shaft_scene, working_scene, other_tools=()):
    """Check declared programs using the reusable continuous geometry API.

    The exported program ends inactive before contact: all retained material is
    therefore excluded for the complete working envelope as well as the shaft.
    """
    actions = route['actionProgram']
    withdrawal = route['withdrawalProgram']
    if any(action.get('operation') in ('apply', 'hold', 'grasp', 'cut', 'bladeManeuver') for action in [*actions, *withdrawal]):
        raise ValueError('Inactive approach audit cannot authorize a tissue-effect program')
    poses = _program_poses(actions)
    if len(poses) < 2:
        raise ValueError('A route requires explicit insertion endpoints')
    back = (poses[-1], *_program_poses(withdrawal))
    if len(back) < 2:
        raise ValueError('A route requires an explicit withdrawal endpoint')
    if not np.allclose(back[-1].tip_mm, poses[0].tip_mm, rtol=0, atol=1e-8):
        raise ValueError('Withdrawal does not return to its original outside pose')
    insertion = check_route(shaft, poses, shaft_scene)
    exit_check = check_route(shaft, back, shaft_scene)
    whole = check_route(full, (*poses, *back[1:]), working_scene, other_tools=other_tools)
    return insertion, exit_check, whole


def verify(report, source, *, branch='original-exposure'):
    """Rebuild the exact native prefix and certify exported route programs."""
    source = Path(source)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    case, artifacts, _ = load_workspace(source)
    record = artifacts['surgical_toolbox']
    scene = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
    count = report['prefixActions']
    snapshot = record['branches'][branch]
    protection, guard_config = None, None
    if snapshot.get('protectionContract'):
        from resectionlab.surgical_rl_protection import restore_protection_contract
        protection, guard_config = restore_protection_contract(scene, snapshot['protectionContract'])
    engine = SurgicalEngine(scene, protection=protection, guard_config=guard_config)
    prefix = snapshot['actions'][:count]
    if len(prefix) != count:
        raise ValueError('Source branch is shorter than the declared prefix')
    for action in prefix:
        result = engine.step(action)
        if result['status'] == 'rejected':
            raise ValueError('Source prefix refused during independent replay')
    if scene.identity != report['sourceSceneId'] or engine.state_id != report['startStateId']:
        raise ValueError('Exported comparison is not bound to the fresh source prefix')
    if engine.guard_report()['contractId'] != report.get('guardContractId'):
        raise ValueError('Exported protection differs from the fresh source prefix')
    if engine.state['holds'] or any(np.linalg.norm(v) > 0 for v in engine.state.get('deformation', {}).values()):
        raise ValueError('This independent static audit requires a released, undeformed checkpoint')

    retained = scene.labels != 0
    retained.ravel()[engine.state['removed']] = False
    brain_ids = [region['id'] for region in scene.regions if not region.get('target')]
    target_ids = [region['id'] for region in scene.regions if region.get('target')]
    brain = retained & np.isin(scene.labels, brain_ids)
    target = retained & np.isin(scene.labels, target_ids)
    # The complete native revision-2 shaft is exactly this radius-.5 capsule.
    shaft = ToolGeometry('independent-cusa-shaft', .5, .5, 40., tip_length_mm=2.)
    spheres = ()
    if protection is not None:
        # Native protection excludes circumspheres around entire source cells,
        # plus explicit user clearance. Reconstruct that same conservative set
        # in the other geometry implementation, including removed source cells.
        corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        cell_radius = float(np.linalg.norm(corners @ scene.affine[:3, :3].T, axis=1).max())
        spheres = tuple(SphereObstacle(point, cell_radius+protection.clearance_mm,
                                       'protected-source-'+str(cell))
                        for point, cell in zip(scene.points(protection.cells), protection.cells))
    shaft_scene = GeometryScene(retained, scene.affine, exposure_mask=brain, sphere_obstacles=spheres)
    working_scene = shaft_scene
    radius = float(report['constraints']['tool']['radiusMm'])
    non_target_margin = float(report['constraints'].get('nonTargetClearanceMm', 0.))
    # Preserve the actual source mechanics revision. Original geometry has a
    # 2-mm distal working capsule; revision2 has a spherical tip/full shaft.
    # A 1e-10 mm segment conservatively covers the revision2 sphere.
    working_length = 1e-10 if engine.state.get('toolMechanicsRevision') == 2 else 2.
    full = ToolGeometry('independent-cusa', radius, .5, 40., tip_length_mm=working_length)
    margin_scene = GeometryScene(brain, scene.affine) if non_target_margin else None
    margin_tool = (ToolGeometry('independent-non-target-clearance', radius+non_target_margin,
                    .5+non_target_margin, 40., tip_length_mm=working_length)
                   if non_target_margin else None)
    other_tools = []
    for hand, slot in engine.state['slots'].items():
        if hand == 'left':
            if slot['inserted']:
                raise ValueError('Left tool must be withdrawn at this planning checkpoint')
            continue
        # The native collision model retains physical withdrawn tools too.
        # Include their declared poses rather than treating inserted=false as
        # permission to erase a second instrument from the workspace.
        if slot['instrument'] not in ('cusa', 'frazier_suction'):
            raise ValueError('Independent audit does not certify this secondary tool geometry')
        parameters = engine.parameters(slot)
        tip_length = 1e-10 if slot['instrument'] == 'cusa' and engine.state.get('toolMechanicsRevision') == 2 else min(parameters['depthMm'], 3.)
        other_tools.append((ToolGeometry('independent-'+slot['instrument'], parameters['radiusMm'], .5, 40., tip_length_mm=tip_length), _pose(slot)))

    rows = []
    for route in report['routes']:
        insertion, exit_check, bimanual = certify_route(route, shaft=shaft, full=full,
            shaft_scene=shaft_scene, working_scene=working_scene, other_tools=other_tools)
        # This union is contact evidence only, never a removal measurement.
        cells = bimanual.swept_voxel_indices
        target_contact = int(np.count_nonzero(target[tuple(cells.T)]))*scene.voxel_mm3
        brain_contact = int(np.count_nonzero(brain[tuple(cells.T)]))*scene.voxel_mm3
        passed = insertion.feasible and exit_check.feasible and bimanual.feasible
        margin_check = None
        if margin_scene is not None:
            poses = _program_poses(route['actionProgram'])
            reverse = _program_poses(route['withdrawalProgram'])
            margin_check = check_route(margin_tool, (*poses, *reverse), margin_scene)
            passed = passed and margin_check.feasible
        rows.append({'id': route['id'], 'passed': passed,
                     'insertionShaft': _brief(insertion), 'withdrawalShaft': _brief(exit_check),
                     'fullToolAndSecondary': _brief(bimanual),
                     'wholeEnvelopeTargetContactMm3': target_contact,
                     'wholeEnvelopeBrainContactMm3': brain_contact,
                     'workingContactLimitMm3': 0.,
                     'nonTargetClearanceMm': non_target_margin,
                     'nonTargetClearanceCheck': _brief(margin_check) if margin_check is not None else None,
                     'limitingConflict': 'retained_source_contact' if target_contact or brain_contact else None})
    unchanged = hashlib.sha256(source.read_bytes()).hexdigest() == before
    return {'schema': 'resectionlab.interactive-geometry-independent.v1',
            'sourceArchiveSha256': before, 'sourceSceneId': scene.identity,
            'startStateId': engine.state_id, 'prefixActions': count,
            'sourceArchiveUnchanged': unchanged, 'routes': rows,
            'protectionContractId': engine.guard_report()['contractId'],
            'protectedSourceCells': len(protection.cells) if protection is not None else 0,
            'passed': bool(rows) and all(row['passed'] for row in rows) and unchanged,
            'wholeNonTargetRemovedMm3': engine.metrics()['nonTargetRemovedMm3'],
            'historicalTargetRemovedMm3': engine.metrics()['targetRemovedMm3'],
            'geometryMethod': 'geometry.check_route; conservative continuous midpoint envelopes and exact axial unions',
            'workingTipLengthMm': working_length,
            'workingSphereCoverExtraLengthMm': 1e-10 if working_length == 1e-10 else 0.,
            'scope': 'Independent static source-geometry audit; contact is not removal; missing anatomy and physical validity remain unassessed.'}


def verify_constraint_change_and_refusal(baseline, edited, refusal):
    """Inspect actual exported decisions without calling planner validation."""
    fields = ('sourceSceneId', 'startStateId', 'prefixActions')
    if any(baseline.get(key) != candidate.get(key) for key in fields for candidate in (edited, refusal)):
        raise ValueError('Decision comparisons must share the same exact source checkpoint')
    signature = lambda report: [(route['entryPose']['position'], route['targetPose']['position'])
                                for route in report['routes']]
    changed = (baseline['constraints'] != edited['constraints']
               and signature(baseline) != signature(edited))
    required = set(refusal['constraints']['requiredEvidence'])
    # These anatomy/measured-observation classes are absent from this source:
    # the one-patient scope explicitly supplies only brain and tumor masks.
    missing = required & {'vessels', 'functional', 'shift', 'skull'}
    refused = (bool(missing) and not refusal['routes']
               and refusal['status'] == 'required_evidence_unavailable'
               and missing <= set(refusal.get('missingRequiredEvidence', [])))
    return {'sameSourceCheckpoint': True, 'constraintChangesActualGeometry': changed,
            'requiredUnknownEvidenceRefuses': refused,
            'missingRequiredEvidence': sorted(missing), 'passed': changed and refused}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--branch', default='original-exposure')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(_read(args.report), args.source, branch=args.branch)
    payload = json.dumps(result, indent=2, allow_nan=False)+'\n'
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(ROOT / 'testcases'):
            raise ValueError('Local verification output belongs under testcases/')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
    print(payload)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
