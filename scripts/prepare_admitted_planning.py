#!/usr/bin/env python3
"""Qualify an existing acquired native case/checkpoint for the common planner.

No patient-specific production conditions, anatomy generation, new exposure,
training or acquisition. Earlier native branches and numeric source data stay
unchanged. Entry defaults derive from completed, replayed CUSA motion/removal;
they are declared proposal bounds, not a measured surgical exposure.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from resectionlab.core import array_digest, semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.surgical_task_admission import source_evidence, task_spec, prepare_admission, bind_admission
from resectionlab.surgical_interactive_planning import interactive_planning_availability, generate_interactive_approaches
from resectionlab.workspace_bundle import load_workspace, save_workspace


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def replay_and_defaults(scene, snapshot):
    options = {}
    if snapshot.get('protectionContract'):
        protection, guard = restore_protection_contract(scene, snapshot['protectionContract'])
        options = {'protection': protection, 'guard_config': guard}
    engine = SurgicalEngine(scene, **options); stroke = None
    for index, action in enumerate(snapshot['actions']):
        slot = engine.state['slots'][action['slot']]
        previous = np.asarray(slot['position'], float)
        instrument = slot['instrument']
        result = engine.step(copy.deepcopy(action))
        if result['status'] == 'rejected': raise ValueError('Source action no longer replays: '+str(result['reason']))
        endpoint = np.asarray(engine.state['slots'][action['slot']]['position'], float)
        delta = endpoint-previous
        if instrument == 'cusa' and action['operation'] == 'apply' and result['status'] == 'accepted' and np.linalg.norm(delta) > 1. and result['deltas']['removedCells']:
            stroke = {'actionIndex': index, 'startMm': previous.tolist(), 'endMm': endpoint.tolist(),
                'radiusMm': engine.parameters(engine.state['slots'][action['slot']])['radiusMm'], 'stateId': engine.state_id}
    if engine.snapshot() != snapshot: raise ValueError('Canonical source snapshot differs')
    if stroke is None: raise ValueError('A completed native CUSA removal stroke is required to propose entry defaults')
    end = np.asarray(stroke['endMm']); axis = end-np.asarray(stroke['startMm']); axis /= np.linalg.norm(axis)
    removed = scene.points(engine.state['removed']); rel = removed-end; depth = rel@axis
    radial = np.linalg.norm(rel-depth[:, None]*axis, axis=1)
    nearby = depth[radial <= stroke['radiusMm']+float(scene.spacing.max())]
    if not len(nearby): raise ValueError('Completed stroke has no locally removed source tissue')
    entry = None; removed_ids = set(engine.state['removed'])
    for distance in np.arange(float(nearby.min())+min(2., stroke['radiusMm']/3.), float(nearby.max())+.01, float(scene.spacing.min())/4.):
        point = end+axis*distance
        if not engine.touched(point, point, .5) and any(cell in removed_ids for cell in engine.cells(point, point, .5)):
            entry = point; break
    if entry is None: raise ValueError('No clear entry center is linked to local completed source removal')
    ids = [r['id'] for r in scene.regions if r.get('target') and r['material'] == 'tumor']
    remaining = np.setdiff1d(np.flatnonzero(np.isin(scene.labels.ravel(), ids)), engine.state['removed'])
    points = scene.points(remaining); forward = (points-end)@axis > 0.
    if np.any(forward): points = points[forward]
    if not len(points): raise ValueError('No retained target remains after the executed exposure')
    target = points[np.argmin(np.linalg.norm(points-end, axis=1))]
    return engine, stroke, {'entry': {'centerMm': entry.tolist(), 'radiusMm': max(.5, min(7., stroke['radiusMm']-2.))},
                            'axis': axis.tolist(), 'target': {'centerMm': target.tolist(), 'radiusMm': 12.}}


def prepare(args):
    output = args.output.resolve(); source = args.source.resolve()
    if not output.is_relative_to(ROOT/'testcases'): raise ValueError('Keep prepared app data under testcases/')
    if output.exists(): raise ValueError('Preserve previous evidence; choose a fresh output directory')
    output.mkdir(parents=True)
    report = {'schema': 'resectionlab.admitted-planning-preparation.v1', 'source': str(source),
        'sourceSha256': sha(source), 'branch': args.branch, 'output': str(output), 'passed': False}
    started = time.perf_counter()
    try:
        case, artifacts, workspace = load_workspace(source)
        scene = load_environment(source); source_evidence(scene)
        record = artifacts['surgical_toolbox']; original_branches = copy.deepcopy(record['branches'])
        engine, stroke, defaults = replay_and_defaults(scene, record['branches'][args.branch])
        if any(slot['active'] or slot['inserted'] or slot['grip'] for slot in engine.state['slots'].values()) or engine.state['holds']:
            raise ValueError('Source checkpoint must end inactive, released and withdrawn')
        for hand, slot in list(engine.state['slots'].items()):
            if slot['instrument'] == 'cusa':
                result = engine.step({'slot': hand, 'operation': 'inspect', 'parameters': {'radiusMm': 1.5, 'power': 1., 'duty': 1.}, 'duration': .01})
                if result['status'] != 'accepted': raise ValueError('Catalog parameter setup refused: '+str(result['reason']))
        result = engine.step({'slot': 'left', 'operation': 'configure', 'clockRevision': 1, 'mechanicsRevision': 2, 'duration': .1})
        if result['status'] != 'accepted': raise ValueError('Model version setup refused: '+str(result['reason']))
        task = task_spec(entry={'centerMm': args.entry or defaults['entry']['centerMm'],
                               'radiusMm': defaults['entry']['radiusMm'] if args.entry_radius is None else args.entry_radius},
            axis=args.axis or defaults['axis'], target={'centerMm': args.target or defaults['target']['centerMm'], 'radiusMm': args.target_radius},
            allow_unreviewed=True)
        contract = prepare_admission(engine, native_collection=args.native_collection, task=task)
        bind_admission(engine, contract)
        branch = 'task-admitted-'+str(len(record['branches']))
        record['branches'][branch] = engine.snapshot(); record.setdefault('parents', {})[branch] = {'branchId': args.branch, 'position': len(original_branches[args.branch]['actions'])}
        record.setdefault('timeline', {})[branch] = copy.deepcopy(engine.history)
        record.update(branchId=branch, position=len(engine.actions), taskAdmission=contract)
        for name, snapshot in original_branches.items():
            if record['branches'][name] != snapshot: raise ValueError('Historical branch changed')
        destination = output/'admitted-planning.ressectionlab'
        if workspace is None: save_case(case, destination, artifacts=artifacts)
        else:
            save_workspace(case, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
                episode=workspace['episode'], selection=workspace['selection'], originals=workspace['originals'], artifacts=artifacts)
        reopened, saved, _ = load_workspace(destination)
        if reopened.semantic_hash != case.semantic_hash or load_environment(destination).identity != scene.identity:
            raise ValueError('Native source identity changed during persistence')
        if saved['surgical_toolbox'] != record: raise ValueError('Native admission/prefix record changed during persistence')
        cold = SurgicalEngine.restore(scene, saved['surgical_toolbox']['branches'][branch], use_cache=False)
        bind_admission(cold, saved['surgical_toolbox']['taskAdmission'])
        available = interactive_planning_availability(cold)
        if not available['available']: raise ValueError(available['reason'])
        comparison = generate_interactive_approaches(cold, available['defaults'])
        (output/'comparison.json').write_text(json.dumps(comparison, indent=2)+'\n')
        missing = copy.deepcopy(available['defaults']); missing['requiredEvidence'].append('functional')
        refusal = generate_interactive_approaches(cold, missing)
        if refusal['routes'] or refusal['status'] != 'required_evidence_unavailable': raise ValueError('Missing functional evidence did not refuse')
        (output/'missing-functional.json').write_text(json.dumps(refusal, indent=2)+'\n')
        if sha(source) != report['sourceSha256']: raise ValueError('Original workspace changed')
        report.update(passed=True, caseHash=case.semantic_hash, sceneId=scene.identity, sourcePatient=contract['sourceEvidence']['patientId'],
            admissionSeal=contract['admissionSeal'], baselineStateId=cold.state_id, preparedCase=str(destination), savedSha256=sha(destination),
            historicalBranchesPreserved=True, sourceIdentityPreserved=True, sourceStroke=stroke,
            declaredTask=task, generation={'status': comparison['status'], 'routes': len(comparison['routes']),
                'rejectionReasons': sorted(set(r['reason'] for r in comparison['rejections']))}, missingEvidenceRefusal=refusal['missingRequiredEvidence'])
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        report['wallSeconds'] = time.perf_counter()-started
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--branch', required=True)
    p.add_argument('--native-collection', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--entry', type=float, nargs=3); p.add_argument('--entry-radius', type=float)
    p.add_argument('--axis', type=float, nargs=3); p.add_argument('--target', type=float, nargs=3)
    p.add_argument('--target-radius', type=float, default=12.)
    prepare(p.parse_args())
