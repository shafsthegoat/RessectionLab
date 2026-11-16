#!/usr/bin/env python3
"""Summarize every frozen case/seed; never choose models or drop failures."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time
from resectionlab.surgical_sequential_task import generated_task

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n')
def write_csv(path,rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def validate_job_grid(raw_rows,protocol,freeze):
    parents=[c['parent'] for c in protocol['cohort'] if c['split']=='heldout']
    jobs=[('greedy',0),('search',0)]+[
        ('learned' if c.get('selectedUpdates',0)>0 and c.get('rewardDrivenUpdates') else 'untrained',c['seed'])
        for c in freeze['checkpoints']]
    expected={(tier['name'],parent,method,seed) for tier in protocol['onlineTiers'] for parent in parents for method,seed in jobs}
    actual=[(r.get('tier'),r.get('parent'),r.get('method'),r.get('seed')) for r in raw_rows]
    typed=all(type(k[1]) is int and type(k[3]) is int and isinstance(k[0],str) and isinstance(k[2],str) for k in actual)
    counts=Counter(actual)
    checkpoint_seeds=[c['seed'] for c in freeze['checkpoints']]
    seed_grid=(len(checkpoint_seeds)==len(set(checkpoint_seeds)) and set(checkpoint_seeds)==set(protocol['seeds']))
    return dict(verified=seed_grid and typed and set(actual)==expected and len(actual)==len(expected) and all(n==1 for n in counts.values()),
        expectedJobs=len(expected),actualRows=len(actual),heldoutParents=parents,
        checkpointSeedsMatchProtocol=seed_grid,requiredSeeds=protocol['seeds'],checkpointSeeds=checkpoint_seeds,
        learnedSeeds=sorted(seed for method,seed in jobs if method=='learned'),
        missing=[list(k) for k in sorted(expected-set(actual))],
        unexpected=[list(k) for k in sorted(set(actual)-expected,key=str)],
        duplicates=[dict(job=list(k),copies=n) for k,n in counts.items() if n>1],typed=typed)


def compact(row,evidence):
    p=row.get('progress',{});e=row.get('endpoint',{});c=row.get('computation',{})
    previews=c.get('previews',[])
    preview_refusals=sum(r['outcome']['status']=='rejected' for s in previews for r in s['receipt']['receipts'])
    interruptions=[s['receipt']['interruption']['reason'] for s in previews if s['receipt'].get('interruption')]
    valid=bool(e.get('terminated') and not e.get('truncated') and e.get('validWithdrawal') and
        row.get('committedHardViolations')==0 and not row.get('failure'))
    fraction=p.get('originalTargetTreatmentFraction')
    target=p.get('originalTargetCells')
    motor=row.get('structureRemovedMm3',{})
    source=generated_task(row['parent']).scene
    if row.get('sourceSceneId') is not None and row['sourceSceneId']!=source.identity:
        raise ValueError('Frozen result source differs from the current protocol generator')
    voxel=source.voxel_mm3
    artifact=evidence/'results'/f"{row['tier']}-{row['parent']}-{row['method']}-{row['seed']}.json"
    if read(artifact)!=row:raise ValueError('Aggregate row differs from its saved case artifact')
    return dict(parent=row['parent'],pattern=row.get('pattern'),method=row['method'],seed=row['seed'],tier=row['tier'],
        original_target_mm3=target*voxel if target is not None else None,
        treated_mm3=p.get('originalTargetTreatmentMm3'),
        residual_mm3=p.get('residualOriginalTargetCells')*voxel if p.get('residualOriginalTargetCells') is not None else None,
        treatment_fraction=fraction,qualified_treatment_fraction=fraction if valid and fraction is not None else 0.,
        declared_task_complete=bool(e.get('completedTask')),full_original_target_removal=bool(e.get('completeOriginalTargetRemoval')),
        valid_stop_withdrawal=valid,withdrawal_complete=e.get('validWithdrawal'),classification=e.get('classification','failed'),
        neural_removed_mm3=row.get('neuralRemovedMm3'),neural_cut_source_mm3=row.get('neuralCutSourceMm3'),
        motor_cortex_removed_mm3=motor.get('motor-cortex'),cst_removed_mm3=motor.get('corticospinal'),
        hard_violations=row.get('committedHardViolations'),bone_mm3=row.get('boneRemovedMm3'),
        debris_mm3=e.get('debrisMm3'),fluid_mm3=e.get('fluidMm3'),
        physical_requests=c.get('physicalCalls'),planning_requests=c.get('planningPhysicalCalls'),
        execution_requests=c.get('executionPhysicalCalls'),canceled_requests=c.get('canceledRequests'),
        actual_attempts=row.get('attempts'),actual_refusals=row.get('rejections'),preview_refusals=preview_refusals,
        planning_interruptions=len(interruptions),planning_stop_reasons=' | '.join(sorted(set(interruptions))),
        online_wall_seconds=c.get('wallSeconds'),candidate_seconds=c.get('candidateSeconds'),
        policy_feature_seconds=c.get('policySeconds'),reset_seconds=row.get('resetSeconds'),
        replay_verified=row.get('replay',{}).get('verified',False),replay_shared=row.get('replay',{}).get('sharedReplay',False),
        replay_requests=row.get('replay',{}).get('physicalCalls'),replay_wall_seconds=row.get('replay',{}).get('wallSeconds'),
        missing_evidence=' | '.join(row.get('missingEvidence',[])),
        failure=row.get('failure',{}).get('error',''),source_scene_id=row.get('sourceSceneId'),
        final_state_id=row.get('finalStateId'),artifact=str(artifact.resolve()),artifact_sha256=sha(artifact))


def aggregate(rows):
    groups=[]
    for tier,method in sorted({(r['tier'],r['method']) for r in rows}):
        group=[r for r in rows if (r['tier'],r['method'])==(tier,method)]
        n=len(group)
        groups.append(dict(tier=tier,method=method,runs=n,layouts=len({r['parent'] for r in group}),
            mean_treatment_fraction_known=mean([r['treatment_fraction'] for r in group]),
            missing_treatment_outcomes=sum(r['treatment_fraction'] is None for r in group),
            mean_qualified_treatment_fraction=mean([r['qualified_treatment_fraction'] for r in group]),
            declared_task_completion_rate=sum(r['declared_task_complete'] for r in group)/n,
            full_original_removal_rate=sum(r['full_original_target_removal'] for r in group)/n,
            valid_stop_withdrawal_rate=sum(r['valid_stop_withdrawal'] for r in group)/n,
            mean_neural_mm3=mean([r['neural_removed_mm3'] for r in group]),
            mean_motor_cortex_mm3=mean([r['motor_cortex_removed_mm3'] for r in group]),
            mean_cst_mm3=mean([r['cst_removed_mm3'] for r in group]),
            committed_hard_violations=sum(r['hard_violations'] or 0 for r in group),
            missing_harm_outcomes=sum(r['hard_violations'] is None or r['neural_removed_mm3'] is None for r in group),
            failures=sum(bool(r['failure']) for r in group),truncations=sum(r['classification']=='truncated' for r in group),
            actual_refusals=sum(r['actual_refusals'] or 0 for r in group),
            actual_attempts=sum(r['actual_attempts'] or 0 for r in group),
            preview_refusals=sum(r['preview_refusals'] for r in group),
            mean_physical_requests=mean([r['physical_requests'] for r in group]),
            mean_online_seconds=mean([r['online_wall_seconds'] for r in group]),
            maximum_debris_mm3=max((r['debris_mm3'] for r in group if r['debris_mm3'] is not None),default=None),
            replay_passes=sum(r['replay_verified'] for r in group)))
    return groups


def paired(rows):
    controls={(r['tier'],r['parent'],r['method']):r for r in rows if r['method'] in ('greedy','search')}
    result=[]
    for r in rows:
        if r['method']!='learned':continue
        for baseline in ('greedy','search'):
            b=controls[r['tier'],r['parent'],baseline]
            difference=lambda key:r[key]-b[key] if r[key] is not None and b[key] is not None else None
            result.append(dict(tier=r['tier'],parent=r['parent'],pattern=r['pattern'],seed=r['seed'],baseline=baseline,
                treatment_difference_mm3=difference('treated_mm3'),treatment_fraction_difference=difference('treatment_fraction'),
                qualified_fraction_difference=difference('qualified_treatment_fraction'),
                valid_stop_difference=int(r['valid_stop_withdrawal'])-int(b['valid_stop_withdrawal']),
                neural_difference_mm3=difference('neural_removed_mm3'),
                online_request_difference=difference('physical_requests'),online_seconds_difference=difference('online_wall_seconds')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--docs',type=Path,default=ROOT/'docs')
    args=parser.parse_args();evidence=args.evidence.resolve();docs=args.docs.resolve();docs.mkdir(exist_ok=True,parents=True)
    started=time.monotonic()
    evaluation=read(evidence/'evaluation.json');protocol=read(evidence/'protocol.json');freeze=read(evidence/'freeze.json')
    if (freeze['protocolSha256']!=sha(evidence/'protocol.json') or freeze['configHash']!=protocol['configHash'] or
            freeze['sourceHashes']!=protocol['sourceHashes']):
        raise ValueError('Frozen checkpoint/protocol bindings differ')
    for checkpoint in freeze['checkpoints']:
        if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Frozen checkpoint bytes changed')
    if any(sha(ROOT/path)!=digest for path,digest in protocol['sourceHashes'].items()):
        raise ValueError('Frozen source hashes differ; use the recorded implementation checkout')
    coverage=validate_job_grid(evaluation['rows'],protocol,freeze)
    save(docs/'sequential-milestone-cohort-check.json',coverage)
    if not coverage['verified']:raise ValueError('Exact unique frozen job grid is incomplete or changed; see cohort-check.json')
    rows=[compact(r,evidence) for r in evaluation['rows']];summary=aggregate(rows);differences=paired(rows)
    csv_path=docs/'sequential-milestone-results.csv';paired_path=docs/'sequential-milestone-paired.csv'
    write_csv(csv_path,rows)
    if differences:write_csv(paired_path,differences)
    save(docs/'sequential-milestone-protocol.json',protocol)
    seed_summaries=[]
    for tier,seed in sorted({(r['tier'],r['seed']) for r in rows if r['method']=='learned'}):
        group=[r for r in rows if r['tier']==tier and r['seed']==seed and r['method']=='learned']
        seed_summaries.append(dict(tier=tier,seed=seed,
            mean_qualified_treatment_fraction=mean([r['qualified_treatment_fraction'] for r in group]),
            mean_treatment_fraction=mean([r['treatment_fraction'] for r in group]),
            valid_stops=sum(r['valid_stop_withdrawal'] for r in group),
            delayed_benefit_mean=mean([r['qualified_treatment_fraction'] for r in group if r['pattern']==0])))
    paired_summary=[]
    for tier,baseline in sorted({(r['tier'],r['baseline']) for r in differences}):
        group=[r for r in differences if r['tier']==tier and r['baseline']==baseline]
        paired_summary.append(dict(tier=tier,baseline=baseline,pairs=len(group),
            mean_qualified_difference=mean([r['qualified_fraction_difference'] for r in group]),
            delayed_benefit_difference=mean([r['qualified_fraction_difference'] for r in group if r['pattern']==0]),
            wins=sum(r['qualified_fraction_difference']>1e-10 for r in group),
            ties=sum(abs(r['qualified_fraction_difference'])<=1e-10 for r in group),
            losses=sum(r['qualified_fraction_difference']< -1e-10 for r in group)))
    unmet=list(freeze.get('unmetCriteria',[]))
    if len(rows)!=160:unmet.append(f'Expected160 formal rows; retained{len(rows)}')
    if not all(r['replay_verified'] for r in rows):unmet.append('Some frozen rows lack verified replay')
    if any(r['hard_violations'] for r in rows):unmet.append('Committed hard violations present')
    if any(r['hard_violations'] is None or r['neural_removed_mm3'] is None for r in rows):
        unmet.append('Required physical harm outcomes unavailable for some rows')
    manifest=dict(schema='resectionlab.sequential-milestone-delivery.v1',scope='generated post-access continuation',
        evidenceRoot=str(evidence),protocolSha256=sha(evidence/'protocol.json'),freezeSha256=sha(evidence/'freeze.json'),
        evaluationSha256=sha(evidence/'evaluation.json'),configHash=protocol['configHash'],sourceHashes=protocol['sourceHashes'],
        protocolSourceCommit=protocol['sourceCommit'],seeds=protocol['seeds'],checkpoints=freeze['checkpoints'],
        trainingPhysicalCallsChargedUpperBound=freeze['trainingPhysicalCalls'],
        currentTrainingPhysicalCalls=freeze.get('currentTrainingPhysicalCalls',freeze['trainingPhysicalCalls']),
        campaignWallSeconds=freeze['trainingWallSeconds'],priorTrainingCosts=freeze.get('priorTrainingCosts'),
        evaluationWallSeconds=evaluation['wallSeconds'],uniqueSuccessfulReplays=evaluation['uniqueReplays'],
        replayPhysicalCalls=evaluation['replayPhysicalCalls'],replayWallSeconds=evaluation['replayWallSeconds'],
        resultRows=dict(path=str(csv_path),sha256=sha(csv_path),count=len(rows)),
        pairedRows=dict(path=str(paired_path),sha256=sha(paired_path) if differences else None,count=len(differences)),
        summary=summary,seedSummaries=seed_summaries,pairedSummary=paired_summary,unmetCriteria=unmet,
        cohortCheck=coverage,
        exclusions=protocol['exclusions'],earlierAccessCosts=None,patientOperationsCompleted=0,
        analysisWallSeconds=time.monotonic()-started,
        prototypeDiagnosis='Preserved v1 interface diagnostic; no held-out evaluation. Actual recovery withdrawal added to all v2 methods before formal fitting.',
        qualifications='All source anatomical constraints/exposures are generated engineering declarations; functional motor/language and unsupported dynamics remain unknown.',
        commands=dict(evaluate=f'PYTHONPATH=src .venv/bin/python scripts/run_sequential_milestone.py --stage evaluate --generated-only --output {evidence}',
            summarize=f'PYTHONPATH=src .venv/bin/python scripts/summarize_sequential_milestone.py --evidence {evidence}'))
    save(docs/'sequential-milestone-manifest.json',manifest)
    lines=['# Generated sequential decision results','',
        f"Retained the exact {coverage['actualRows']}-row job grid across {len(coverage['heldoutParents'])} held-out layouts. Validation-selected learned seeds: {coverage['learnedSeeds']}. Treatment uses the full original target; declared-task completion and 100% removal are separate. Earlier access costs are unknown.",'',
        '| Tier | Method | Target treated | Task complete |100% removed | Valid STOP/exit | Neural mm³ | Calls | Online seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        value=lambda key,scale=1.,digits=2:'unknown' if r[key] is None else f'{r[key]*scale:.{digits}f}'
        lines.append(f"|{r['tier']}|{r['method']}|{value('mean_treatment_fraction_known',100.,1)}%|{100*r['declared_task_completion_rate']:.1f}%|{100*r['full_original_removal_rate']:.1f}%|{100*r['valid_stop_withdrawal_rate']:.1f}%|{value('mean_neural_mm3')}|{value('mean_physical_requests',digits=1)}|{value('mean_online_seconds')}|")
    lines += ['',f"Campaign wall time including the preserved diagnostic and repair interval:{freeze['trainingWallSeconds']/60:.1f}min. Charged training-call upper bound:{freeze['trainingPhysicalCalls']}; current formal requests:{manifest['currentTrainingPhysicalCalls']}. Frozen evaluation:{evaluation['wallSeconds']/60:.1f}min. Replay requests:{evaluation['replayPhysicalCalls']}.",'',
        'The case-level CSV retains actual and preview refusals, interrupted requests, cleanup, direct motor/CST memberships, every endpoint and all artifact hashes. Paired differences use zero qualified treatment for failed/truncated endpoints while retaining their actual physical treatment separately. Seed ranges are descriptive; no clinical or statistical-superiority claim follows from this small generated cohort.','',
        'Paired learned minus baseline (qualified treatment fraction):','']
    for r in paired_summary:
        lines.append(f"- {r['tier']} versus{r['baseline']}: mean{100*r['mean_qualified_difference']:+.1f}points; delayed-benefit{100*r['delayed_benefit_difference']:+.1f}; wins/ties/losses{r['wins']}/{r['ties']}/{r['losses']}.")
    lines += ['',f"Unmet criteria:{'; '.join(unmet) if unmet else 'none in the declared generated experiment'}. Functional consequences, patient execution, earlier access costs and general physics/RL admission remain unavailable.",'']
    (docs/'sequential-milestone-results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=summary,paired=paired_summary,seeds=seed_summaries,unmet=unmet),indent=2))


if __name__=='__main__':main()
