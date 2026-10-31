#!/usr/bin/env python3
"""Execute bounded public preparation, verify natively, then admit its checkpoint.

Existing source arrays/branches are preserved. No source-dependent coordinates,
mask edits, tool shrinking, privileged removal or old task/study replacement.
"""
from __future__ import annotations

import argparse
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'src'))
from resectionlab.core import semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_exposure import DEFAULTS, prepare
from resectionlab.surgical_task_admission import verify_source, task_spec, prepare_admission, bind_admission
from resectionlab.workspace_bundle import load_workspace, save_workspace


def save_native(case, artifacts, workspace, destination):
    if workspace is None: save_case(case, destination, artifacts=artifacts)
    else:
        save_workspace(case, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
            episode=workspace['episode'], selection=workspace['selection'], originals=workspace['originals'], artifacts=artifacts)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--native-collection', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--branch')
    p.add_argument('--source-ray', type=Path, help='JSON surfaceMm/sourceTargetMm at original source-cell centers in RAS mm; no replacement ray')
    p.add_argument('--acknowledge-research', action='store_true', required=True)
    p.add_argument('--entry-radius', type=float, default=.5); p.add_argument('--target-radius', type=float, default=6.)
    p.add_argument('--max-non-target-mm3', type=float, default=DEFAULTS['maxNonTargetMm3'])
    p.add_argument('--max-candidates', type=int, default=DEFAULTS['maxCandidates'])
    p.add_argument('--max-actions', type=int, default=DEFAULTS['maxActions'])
    p.add_argument('--max-seconds', type=float, default=DEFAULTS['maxSeconds'])
    p.add_argument('--lateral-sweep-mm', type=float, default=DEFAULTS['lateralSweepMm'])
    args = p.parse_args(); source, output = args.source.resolve(), args.output.resolve()
    if not source.is_relative_to(ROOT/'testcases') or not output.is_relative_to(ROOT/'testcases') or output.exists():
        p.error('Reuse testcases sources and choose a fresh testcases output directory')
    output.mkdir(parents=True); original = sha256(source.read_bytes()).hexdigest(); started = time.perf_counter()
    report = {'source': str(source), 'sourceSha256': original, 'passed': False, 'researchAcknowledged': True}
    try:
        case, artifacts, workspace = load_workspace(source); scene = load_environment(source)
        record = artifacts['surgical_toolbox']; historical = copy.deepcopy(record['branches'])
        if record.get('taskAdmission') or record.get('strategyComparisons') or record.get('interactiveApproaches'):
            raise ValueError('Preserve existing task/studies; prepare from their original unadmitted source in a separate workspace')
        verify_source(scene, native_collection=args.native_collection)
        branch = args.branch or record['branchId']; engine = SurgicalEngine.restore(scene, record['branches'][branch])
        if any(s['inserted'] or s['active'] or s['grip'] for s in engine.state['slots'].values()) or engine.state['holds']:
            raise ValueError('Preparation starts with both tools inactive, released and withdrawn')
        if engine.state.get('toolMechanicsRevision') != 2 or engine.state.get('physicalClockRevision') != 1:
            result = engine.step({'slot': 'right', 'operation': 'configure', 'mechanicsRevision': 2, 'clockRevision': 1, 'duration': .1})
            if result['status'] != 'accepted': raise ValueError('Model setup refused: '+str(result.get('reason')))
        controls = {**DEFAULTS, 'maxNonTargetMm3': args.max_non_target_mm3, 'maxCandidates': args.max_candidates,
                    'lateralSweepMm': args.lateral_sweep_mm, 'maxActions': args.max_actions, 'maxSeconds': args.max_seconds}
        ray = json.loads(args.source_ray.read_text()) if args.source_ray else None
        selected, proof = prepare(engine, controls, native_collection=args.native_collection, source_ray=ray)
        (output/'preparation.json').write_text(json.dumps(proof, indent=2)+'\n')
        if selected is None:
            prefix = 'preparation-source-'+str(len(record['branches']))
            if prefix in record['branches']: raise ValueError('Preparation source branch name conflict')
            record['branches'][prefix] = engine.snapshot()
            record.setdefault('parents', {})[prefix] = {'branchId':branch,'position':len(historical[branch]['actions'])}
            record.setdefault('timeline', {})[prefix] = copy.deepcopy(engine.history)
            record.update(branchId=prefix,position=len(engine.actions))
            destination=output/'source-checkpoint.ressectionlab';save_native(case,artifacts,workspace,destination)
            cold=SurgicalEngine.restore(load_environment(destination),engine.snapshot(),use_cache=False)
            if cold.snapshot()!=engine.snapshot():raise ValueError('Cold refused-preparation source differs')
            report.update(status='STOP', decision=proof['publicPlan']['decision'], nativeSourceUnchanged=True,
                sourceCheckpoint=str(destination),preparationSeal=proof['preparationSeal'])
            return
        row = proof['publicPlan']['attempts'][proof['publicPlan']['decision']['selectedIndex']]['route']
        task = task_spec(entry={'centerMm': row['surfaceMm'], 'radiusMm': args.entry_radius},
            target={'centerMm': row['sourceTargetMm'], 'radiusMm': args.target_radius}, axis=row['axisRasUnit'], allow_unreviewed=True)
        contract = prepare_admission(selected, native_collection=args.native_collection, task=task); bind_admission(selected, contract)
        prefix = 'preparation-source-'+str(len(record['branches'])); result_branch = 'prepared-task-'+str(len(record['branches']))
        if prefix in record['branches'] or result_branch in record['branches']: raise ValueError('Preparation branch name conflict')
        record['branches'][prefix] = engine.snapshot(); record.setdefault('parents', {})[prefix] = {'branchId': branch, 'position': len(historical[branch]['actions'])}
        record.setdefault('timeline', {})[prefix] = copy.deepcopy(engine.history)
        record['branches'][result_branch] = selected.snapshot(); record['parents'][result_branch] = {'branchId': prefix, 'position': len(engine.actions)}
        record['timeline'][result_branch] = copy.deepcopy(selected.history)
        record.update(branchId=result_branch, position=len(selected.actions), taskAdmission=contract)
        for key, value in historical.items():
            if record['branches'][key] != value: raise ValueError('Historical source branch changed')
        destination = output/'admitted-preparation.ressectionlab'
        save_native(case, artifacts, workspace, destination)
        loaded, saved, _ = load_workspace(destination)
        if loaded.semantic_hash != case.semantic_hash or saved['surgical_toolbox'] != record: raise ValueError('Saved source/record differs')
        cold = SurgicalEngine.restore(load_environment(destination), record['branches'][result_branch], use_cache=False)
        bind_admission(cold, contract)
        from resectionlab.surgical_interactive_planning import interactive_planning_availability
        available = interactive_planning_availability(cold)
        if not available['available']: raise ValueError('Prepared task observation is unavailable: '+str(available['reason']))
        if cold.snapshot() != selected.snapshot(): raise ValueError('Cold prepared native replay differs')
        report.update(passed=True, status='ADMITTED_PREPARATION', case=str(destination), stateId=cold.state_id,
            baselineActions=len(cold.actions), summary=proof['publicPlan']['attempts'][proof['publicPlan']['decision']['selectedIndex']]['summary'],
            task=task, admissionSeal=contract['admissionSeal'], sourceArraysAndHistoricalBranchesUnchanged=True,
            preparationSeal=proof['preparationSeal'])
    except BaseException as error:
        report['error'] = str(error); raise
    finally:
        report.update(elapsedSeconds=time.perf_counter()-started, sourcePreserved=sha256(source.read_bytes()).hexdigest() == original)
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n'); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
