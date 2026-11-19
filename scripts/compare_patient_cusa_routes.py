#!/usr/bin/env python3
"""Compare two prospectively screened CUSA approaches to one unchanged development lesion."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from data_integrated_probe import envelope_admission
from probe_cta_organic_rehearsal import cells, cover, fibonacci_directions
from resectionlab.anatomy_approaches import dispatch_plan
from resectionlab.anatomy_planning import native_target_binding
from resectionlab.anatomy_workspace import save_view
from resectionlab.core import array_digest, semantic_digest
from resectionlab.data_root import sha256_file
from resectionlab.geometry import GeometryScene
from resectionlab.imaging import save_case
from resectionlab.patient_anatomy import PatientImagingCollection
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_patient import _resample
from resectionlab.surgical_scene import original_target_memberships
from resectionlab.surgical_task_admission import verify_source
from resectionlab.workspace_bundle import load_workspace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'testcases/source-preflight-v1/0004-current-derivative-01/patient.ressectionlab'
STUDY = ROOT / 'testcases/environments/native/UCSF-PDGM-0004'
TARGET = 'enhancing_target'
GUARDS = {
    'brain-stem',
    'brainstem',
    'brain stem',
    *(
        side + ' ' + name
        for side in ('left', 'right')
        for name in ('thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus')
    ),
}
STRATEGIES = ['short-access', 'alternate-entry']
LIMITS = dict(
    targetGoalMm3=100.0,
    maxOtherMm3=160.0,
    maxActiveSeconds=150.0,
    actionDurationSeconds=15.0,
    maxApplies=10,
    radiusMm=1.5,
    minimumEntrySeparationMm=5.0,
    maximumStrokeMm=48.0,
    standoffMm=2.0,
    directions=192,
    targetLandmarks=32,
)
SCOPE = (
    'Modeled intracerebral CUSA after assumed scalp, skull and dural exposure. No skull or dura '
    'is mapped or removed. VesselFM and named SynthSeg labels are unreviewed estimates. Functional '
    'outcomes, missed vessels, skull opening burden and clinical feasibility are unknown.'
)


def put(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def inputs():
    case, artifacts, _ = load_workspace(SOURCE)
    scene = load_environment(SOURCE)
    record = artifacts['surgical_toolbox']
    root = record['branches'][record['branchId']]
    if root['actions'] or root['checkpoint']['state']['removed']:
        raise ValueError('Matched treatment requires the intact, zero-action root')
    return case, scene, record, root


def freeze(case, scene, root):
    study = PatientImagingCollection(STUDY)
    verify_source(scene, native_collection=STUDY)
    named = [r['id'] for r in study.document['layers'] if r['label'].lower() in GUARDS]
    if len(named) != 11:
        raise ValueError('Expected eleven explicitly named protection estimates in this revision')
    return dict(
        schema='fixed-lesion-comparison-v1',
        patientId=study.document['patient_id'],
        sourceSnapshotId=study.identity(),
        sourcePackageSha256=sha256_file(SOURCE),
        caseHash=case.semantic_hash,
        sceneId=scene.identity,
        startingStateId=root['stateId'],
        initialProtectionHash=root['protectionContract']['contentHash'],
        physicsHash=semantic_digest(scene.properties),
        originalTargetLayer=TARGET,
        originalTargetBinding=native_target_binding(study, study.layer(TARGET)),
        sourceTargetMaskHash=array_digest(case.compartments['tumor']),
        originalTargetMm3=len(original_target_memberships(scene)['tumor']) * scene.voxel_mm3,
        ownedTargetMm3=float(np.count_nonzero(scene.labels == 2) * scene.voxel_mm3),
        observations=[
            'T1ce',
            'source enhancing tumor',
            'estimated brain',
            'experimental vesselFM',
            'named SynthSeg estimates',
            'accepted public engine metrics and tissue state',
        ],
        extraScreenedExclusions=named,
        limits=LIMITS,
        scope=SCOPE,
        generator='Deterministic geometry search followed by scripted stock CUSA with identical state-feedback stops.',
        selection='Frozen lesion landmarks and spherical directions; rank free complete-tool corridors by estimated working-cell non-target/target, then distance. Distinct entries must be at least 5 mm apart.',
        stopping='Stop on meeting 100 mm³ target goal, budget, refused prospective transition, stalled effect or completed stroke; require recorded safe withdrawal.',
        thermal='Neither arm uses a thermal instrument. Drill temperature and thermal exposure retain their modeled units; no validated patient temperature threshold.',
    )


def exclusions(study, scene, layer_ids):
    rows = [study.layer(key) for key in layer_ids]
    first = rows[0]
    if any(
        r['data'] != first['data']
        or not np.array_equal(study.patient_affine(r), study.patient_affine(first))
        for r in rows
    ):
        raise ValueError('Named estimates no longer share their verified parcellation frame')
    mask = np.isin(study.array(first['data']), [r['label_value'] for r in rows])
    if mask.shape == scene.shape and np.array_equal(study.patient_affine(first), scene.affine):
        return mask
    return _resample(
        mask.astype(np.uint8),
        study.patient_affine(first),
        scene.affine,
        scene.shape,
        label=1,
        conservative=True,
    )


def choose(scene, protected, limits):
    labels = scene.labels
    owned = labels == 2
    forbidden = (labels == 3) | protected
    grid = GeometryScene(forbidden, scene.affine)
    target_cells = np.argwhere(owned)
    points = scene.points(np.ravel_multi_index(target_cells.T, scene.shape))
    # One central seed and deterministic farthest-point landmarks cover the original lesion.
    first = int(np.argmin(np.linalg.norm(points - points.mean(axis=0), axis=1)))
    landmarks = [first]
    nearest = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, limits['targetLandmarks']):
        index = int(nearest.argmax())
        landmarks.append(index)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[index], axis=1))
    directions = fibonacci_directions(limits['directions'])
    ranges = np.arange(0, limits['maximumStrokeMm'] + 0.5, 0.5)
    counts = Counter()
    candidates = []
    for idx in landmarks:
        goal = points[idx]
        for j, out in enumerate(directions):
            xyz = goal + ranges[:, None] * out
            vox = np.rint(xyz @ scene.inverse[:3, :3].T + scene.inverse[:3, 3]).astype(int)
            inside = np.all((vox >= 0) & (vox < scene.shape), axis=1)
            n = int(np.count_nonzero(inside))
            line = labels[tuple(vox[:n].T)]
            # The external end must clear the last tissue crossing; an internal
            # background hole does not become a simulated entry.
            hits = np.flatnonzero(line)
            if not len(hits) or hits[-1] + 4 >= n:
                counts['no_exterior_in_stroke'] += 1
                continue
            length = float(ranges[hits[-1]] + limits['standoffMm'])
            entry = goal + length * out
            if np.any(forbidden[tuple(vox[: hits[-1] + 1].T)]):
                counts['centerline_exclusion'] += 1
                continue
            q = Rotation.align_vectors([-out], [[0.0, 0.0, 1.0]])[0].as_quat().tolist()
            sweep = cover('cusa', entry, goal, q)
            sweep.append((entry + 40 * out, entry, limits['radiusMm'], 'placement'))
            if not envelope_admission(sweep, scene.affine, np.asarray(scene.shape))['admitted']:
                counts['outside_instrument_FOV'] += 1
                continue
            span = cells(grid, sweep)
            if np.any(forbidden.ravel()[span]):
                counts['complete_tool_exclusion'] += 1
                continue
            if np.any(labels.ravel()[cells(grid, [sweep[-1]])]):
                counts['placement_not_free'] += 1
                continue
            active = cells(grid, sweep, 'working')
            targetn = int(np.count_nonzero(owned.ravel()[active]))
            brainn = int(np.count_nonzero(labels.ravel()[active] == 1))
            if not targetn or brainn * scene.voxel_mm3 > limits['maxOtherMm3']:
                counts['no_target_or_over_budget'] += 1
                continue
            candidates.append(
                dict(
                    entry=entry.tolist(),
                    target=goal.tolist(),
                    quaternion=q,
                    radiusMm=limits['radiusMm'],
                    lengthMm=length,
                    workingTargetCells=targetn,
                    workingBrainCells=brainn,
                    directionIndex=j,
                    goalVoxel=target_cells[idx].tolist(),
                    screenedVesselCells=0,
                    screenedNamedProtectionCells=0,
                )
            )
    candidates.sort(
        key=lambda r: (
            r['workingBrainCells'] / r['workingTargetCells'],
            r['lengthMm'],
            r['directionIndex'],
            r['goalVoxel'],
        )
    )
    selected = []
    for row in candidates:
        if (
            selected
            and np.linalg.norm(np.subtract(row['entry'], selected[0]['entry']))
            < limits['minimumEntrySeparationMm']
        ):
            continue
        selected.append(row)
        if len(selected) == 2:
            break
    return dict(candidateCount=len(candidates), refusals=dict(counts), selected=selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['plan', 'execute', 'bind'])
    p.add_argument('output', type=Path)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'testcases/outputs'):
        p.error('Choose an ignored testing output directory')
    case, scene, record, root = inputs()
    if args.stage == 'plan':
        output.mkdir(parents=True, exist_ok=False)
        protocol = freeze(case, scene, root)
        put(output / 'protocol.json', protocol)
        selection = choose(
            scene,
            exclusions(PatientImagingCollection(STUDY), scene, protocol['extraScreenedExclusions']),
            LIMITS,
        )
        put(output / 'routes.json', selection)
        print({k: v for k, v in selection.items() if k != 'selected'}, flush=True)
        if len(selection['selected']) != 2:
            raise ValueError('Fewer than two screened entries; preserve refusal evidence.')
    else:
        run_stage(args.stage, output, case, scene, record, root)


def execute(scene, root, route, guard):
    engine = SurgicalEngine.restore(scene, root, use_cache=False)
    original = np.asarray(original_target_memberships(scene)['tumor'])
    original_mask = np.zeros(scene.shape, dtype=bool)
    original_mask.flat[original] = True
    start = dict(position=route['entry'], quaternion=route['quaternion'])
    goal = dict(position=route['target'], quaternion=route['quaternion'])
    steps = []
    blocked = []
    withdraw_reason = None

    def request(action):
        proposed = engine.preview(action, include_history=False)
        view = proposed['view']
        metrics = view['metrics']
        if proposed['status'] == 'rejected':
            blocked.append(
                dict(
                    operation=action['operation'],
                    reason=proposed.get('reasonCode') or proposed.get('reason'),
                    stateId=engine.state_id,
                )
            )
            return False
        removed = np.asarray(view['removed'], dtype=int)
        checks = {
            'named protection removal': bool(guard.ravel()[removed].any()),
            'vascular removal': bool(np.any(scene.labels.ravel()[removed] == 3)),
            'non-target removal cap': metrics['nonTargetRemovedMm3'] > LIMITS['maxOtherMm3'],
            'collision': bool(metrics['collisions']),
            'mapped vessel encounter': bool(metrics['vesselEncounters']),
        }
        failures = [name for name, tripped in checks.items() if tripped]
        if failures:
            blocked.append(
                dict(operation=action['operation'], reason=', '.join(failures), stateId=engine.state_id)
            )
            return False
        before = engine.state_id
        result = engine.step(action)
        if result['status'] != proposed['status'] or engine.state_id != view['stateId']:
            raise ValueError('Preview differs from committed state')
        counted = int(np.count_nonzero(original_mask.ravel()[removed]))
        if abs(view['originalTargetResidualMm3'] - (len(original) - counted) * scene.voxel_mm3) > 1e-7:
            raise ValueError('Original annotation residual differs from the removed source-cell set')
        steps.append(
            dict(
                operation=action['operation'],
                duration=action['duration'],
                status=result['status'],
                reason=result.get('reasonCode') or result.get('reason'),
                stateBefore=before,
                stateId=engine.state_id,
                targetRemovedMm3=metrics['targetRemovedMm3'],
                nonTargetRemovedMm3=metrics['nonTargetRemovedMm3'],
                originalResidualMm3=view['originalTargetResidualMm3'],
                removedCells=len(removed),
                removalHash=semantic_digest(sorted(map(int, removed))),
                tip=view['slots']['right']['position'],
                elapsedSeconds=metrics['elapsedSeconds'],
            )
        )
        return True

    def action(op, sec, **kw):
        return dict(slot='right', operation=op, duration=sec, **kw)

    configured = request(action('configure', 0.1, mechanicsRevision=2, clockRevision=1))
    placed = configured and request(action('place', 0.1, pose=start))
    applies = 0
    stop = 'placement_refused'
    if placed:
        stop = 'active_time_budget'
        for _ in range(LIMITS['maxApplies']):
            previous = engine.view(include_history=False)
            if not request(action('apply', LIMITS['actionDurationSeconds'], path=[goal])):
                stop = 'prospective_action_refused'
                break
            applies += 1
            if steps[-1]['targetRemovedMm3'] >= LIMITS['targetGoalMm3']:
                stop = 'bounded_target_goal'
                break
            if np.linalg.norm(np.subtract(steps[-1]['tip'], route['target'])) < 1e-6:
                stop = 'completed_straight_stroke'
                break
            current = engine.view(include_history=False)
            if (
                current['slots']['right']['position'] == previous['slots']['right']['position']
                and len(current['removed']) == len(previous['removed'])
                and current['metrics'].get('cusaPendingCapacityMm3', 0)
                == previous['metrics'].get('cusaPendingCapacityMm3', 0)
            ):
                stop = 'no_tip_removal_or_pending_work_progress'
                break
        if not request(action('withdraw', 2.0, path=[start])):
            withdraw_reason = blocked[-1]['reason']
    end = engine.view(include_history=False)
    metrics = end['metrics']
    effect = next((i for i, r in enumerate(steps) if r['targetRemovedMm3'] > 0), None)
    summary = dict(
        route=route,
        initialStateId=root['stateId'],
        finalStateId=engine.state_id,
        stop=stop,
        goalAchieved=bool(
            metrics['targetRemovedMm3'] >= LIMITS['targetGoalMm3'] and placed and not withdraw_reason
        ),
        activeSeconds=applies * LIMITS['actionDurationSeconds'],
        initialOriginalTargetMm3=len(original) * scene.voxel_mm3,
        residualOriginalTargetMm3=end['originalTargetResidualMm3'],
        metrics=metrics,
        skullRemovalMm3=0.0,
        skullOpeningCost='unknown; skull and dura are absent',
        otherAtBoundaryBeforeFirstTargetActionMm3=(
            steps[effect - 1]['nonTargetRemovedMm3'] if effect and effect > 0 else 0.0
        ),
        achievedDepthMm=max(
            (
                float(np.linalg.norm(np.subtract(row['tip'], start['position'])))
                for row in steps
                if row['operation'] == 'apply'
            ),
            default=0.0,
        ),
        namedExclusionCellsRemoved=int(guard.ravel()[np.asarray(end['removed'], dtype=int)].sum()),
        withdrawal=withdraw_reason or ('recorded' if placed else 'never inserted'),
        functionalRisk='unknown',
        blocked=blocked,
        steps=steps,
    )
    return engine, summary


def plan_view(study, selected):
    save_view(
        study,
        dict(
            schema='patient-anatomy-view-v1',
            patientId=study.document['patient_id'],
            snapshotId=study.identity(),
            seriesId=study.document['reference_series'],
            states={},
            selectedPlanningTargetId=TARGET,
            selectedApproachId=selected,
            selectedApproachIds={TARGET: selected},
        ),
    )


def run_stage(stage, output, case, scene, record, root):
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol != freeze(case, scene, root):
        raise ValueError('Frozen protocol, anatomy, target or initial state changed')
    routes = json.loads((output / 'routes.json').read_text())['selected']
    if len(routes) != 2:
        raise ValueError('Two screened entries required')
    selection = json.loads((output / 'routes.json').read_text())
    if stage == 'execute':
        guard = exclusions(PatientImagingCollection(STUDY), scene, protocol['extraScreenedExclusions'])
        if any((output / (name + '-raw.ressectionlab')).exists() for name in STRATEGIES):
            raise ValueError('Preserve the completed execution. Choose a fresh comparison directory.')
        if choose(scene, guard, LIMITS) != selection:
            raise ValueError('Route generation differs from the frozen source and planning evidence')
        if not (output / 'native').exists():
            shutil.copytree(
                STUDY,
                output / 'native',
                ignore=shutil.ignore_patterns('reconstructions', 'meshes', 'revisions', '.write.lock'),
            )
        study = PatientImagingCollection(output / 'native')
        if study.identity() != protocol['sourceSnapshotId']:
            raise ValueError('Study clone differs')
        dispatch_plan(
            study,
            {
                'action': 'selectTarget',
                'targetLayerId': TARGET,
                'candidateBinding': protocol['originalTargetBinding'],
            },
        )
        for name, route in zip(STRATEGIES, routes):
            dispatch_plan(
                study,
                {
                    'action': 'upsert',
                    'approach': dict(
                        id=name,
                        name=name.replace('-', ' ').title() + ' · CUSA',
                        targetLayerId=TARGET,
                        entry=route['entry'],
                        target=route['target'],
                        radiusMm=route['radiusMm'],
                    ),
                },
            )
        for name, route in zip(STRATEGIES, routes):
            engine, summary = execute(scene, root, route, guard)
            updated = copy.deepcopy(record)
            updated['branchId'] = name
            updated['position'] = len(engine.actions)
            updated['branches'][name] = engine.snapshot()
            updated['parents'][name] = {'branchId': record['branchId'], 'position': 0}
            updated['timeline'][name] = engine.history
            summary.update(
                branchId=name,
                protocolHash=semantic_digest(protocol),
                sceneId=scene.identity,
                caseHash=case.semantic_hash,
                selectionHash=semantic_digest(selection),
                generatorSourceSha256=sha256_file(__file__),
            )
            put(output / (name + '-outcome.json'), summary)
            print(
                name,
                summary['stop'],
                summary['metrics']['targetRemovedMm3'],
                summary['metrics']['nonTargetRemovedMm3'],
                summary['blocked'],
                flush=True,
            )
            save_case(
                case, output / (name + '-raw.ressectionlab'), artifacts={'surgical_toolbox': updated}
            )
        plan_view(study, STRATEGIES[0])
    else:
        from resectionlab.surgical_exact_grid_replay import bind_recording
        from resectionlab.surgical_study_replay import verify_recording

        study = PatientImagingCollection(output / 'native')
        for name in STRATEGIES:
            result = json.loads((output / (name + '-outcome.json')).read_text())
            if result['protocolHash'] != semantic_digest(protocol) or result[
                'selectionHash'
            ] != semantic_digest(selection):
                raise ValueError('The saved outcome and frozen comparison inputs differ')
            case, artifacts, _ = load_workspace(output / (name + '-raw.ressectionlab'))
            rec = artifacts['surgical_toolbox']
            plan_view(study, name)
            rec['exactGridReplay'] = bind_recording(
                study,
                case,
                rec,
                approach_id=name,
                target_id=TARGET,
                generator=dict(
                    kind='scripted',
                    name='Budgeted straight CUSA after geometric search',
                    sourceSha256=result['generatorSourceSha256'],
                ),
            )
            receipt, sampling = verify_recording(study, TARGET, case, rec, approach_id=name)
            artifacts['nativeStudyReplay'] = receipt
            save_case(case, output / (name + '.ressectionlab'), artifacts=artifacts)
            put(output / (name + '-binding.json'), dict(receipt=receipt, sampling=sampling))
        plan_view(study, STRATEGIES[0])


if __name__ == '__main__':
    main()
