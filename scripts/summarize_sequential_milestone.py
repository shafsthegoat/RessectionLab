#!/usr/bin/env python3
"""Summarize every frozen case/seed; never choose models or drop failures."""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics
import time
from resectionlab.surgical_sequential_task import generated_task
from resectionlab.surgical_procedure_policy import procedure_source_memberships

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
    scope=(set(parents)==set(range(200,216)) and len(parents)==16 and
        protocol['seeds']==[17,43,71] and [t['name'] for t in protocol['onlineTiers']]==['low','high'])
    return dict(verified=scope and seed_grid and typed and set(actual)==expected and len(actual)==len(expected) and all(n==1 for n in counts.values()),
        expectedJobs=len(expected),actualRows=len(actual),heldoutParents=parents,
        declaredStudyScopeMatches=scope,
        checkpointSeedsMatchProtocol=seed_grid,requiredSeeds=protocol['seeds'],checkpointSeeds=checkpoint_seeds,
        learnedSeeds=sorted(seed for method,seed in jobs if method=='learned'),
        missing=[list(k) for k in sorted(expected-set(actual))],
        unexpected=[list(k) for k in sorted(set(actual)-expected,key=str)],
        duplicates=[dict(job=list(k),copies=n) for k,n in counts.items() if n>1],typed=typed)


class NativeAudit:
    """Independent accepted-history replay; audit cost stays separate from online cost."""
    def __init__(self):
        self.cache={};self.requests=0;self.seconds=0.;self.failures=[]

    def restore(self,task,snapshot):
        from resectionlab.surgical_engine import SurgicalEngine
        key=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest()
        if key in self.cache:return self.cache[key]
        owner=self
        class CountedEngine(SurgicalEngine):
            def step(self,action,**kwargs):
                owner.requests+=1
                return super().step(action,**kwargs)
        started=time.monotonic()
        try:
            engine=CountedEngine.restore(task.scene,snapshot,use_cache=False,force_verify=True)
            self.cache[key]=engine
            return engine
        except Exception as error:
            self.failures.append(dict(snapshotHash=key,error=str(error)))
            raise
        finally:self.seconds+=time.monotonic()-started


def compact(row,evidence,audit=None):
    from resectionlab.surgical_sequential_task import endpoint_assessment
    from resectionlab.surgical_procedure_policy import evaluate_procedure_sequence
    from resectionlab.surgical_joint_cranial_search import assess_original_target_progress
    from types import SimpleNamespace
    audit=NativeAudit() if audit is None else audit
    task=generated_task(row['parent']);source=task.scene;voxel=source.voxel_mm3
    pattern=task.audit['pattern'];errors=[]
    if row.get('pattern',pattern)!=pattern:errors.append('pattern differs from frozen source')
    artifact=evidence/'results'/f"{row['tier']}-{row['parent']}-{row['method']}-{row['seed']}.json"
    if read(artifact)!=row:raise ValueError('Aggregate row differs from its saved case artifact')
    c=row.get('computation',{});previews=c.get('previews',[])
    preview_refusals=sum(r['outcome']['status']=='rejected' for s in previews for r in s['receipt']['receipts'])
    interruptions=[s['receipt']['interruption']['reason'] for s in previews if s['receipt'].get('interruption')]
    p={};endpoint={};dims={};engine=None;native_verified=False;measured_hard=None
    initial=assess_original_target_progress(SimpleNamespace(scene=source,state={'removed':[]}))
    snapshot=row.get('snapshot');proof=row.get('replay',{})
    if row.get('sourceSceneId')!=source.identity:errors.append('source scene binding unavailable or wrong')
    if row.get('configHash')!=task.config.fingerprint:errors.append('task config binding unavailable or wrong')
    if snapshot is None:errors.append('native snapshot unavailable')
    else:
        try:
            engine=audit.restore(task,snapshot);native_verified=True
            sequence=evaluate_procedure_sequence(engine);dims=sequence['outcome']['dimensions']
            measured_hard=len(sequence['violations'])
            history=row.get('skillHistory',[])
            attempts=sum(len(s['receipts'])+int(s.get('checkedStopAttempted',False)) for s in history)
            invalid=sum(r['outcome']['status']=='rejected' for s in history for r in s['receipts'])
            invalid+=sum(s.get('checkedStopAttempted',False) and not s['endpoint']['validWithdrawal'] for s in history)
            stopped=bool(history and history[-1]['kind']=='stop' and history[-1].get('checkedStopAttempted') and history[-1]['status']=='accepted')
            endpoint=endpoint_assessment(engine,task.config,attempts=attempts,invalid_actions=invalid,
                stopped=stopped,externally_truncated=not stopped)
            p=endpoint['progress']
            for label,condition in [
                ('final state binding',row.get('finalStateId')==snapshot.get('stateId')==engine.state_id),
                ('proof final state binding',proof.get('finalStateId')==engine.state_id),
                ('proof endpoint classification',proof.get('classification')==endpoint['classification']),
                ('attempt accounting',row.get('attempts')==attempts),
                ('refusal accounting',row.get('rejections')==invalid),
                ('target progress accounting',row.get('progress')==p),
                ('endpoint accounting',row.get('endpoint')==endpoint),
                ('neural accounting',row.get('neuralRemovedMm3')==dims['neuralRemovedMm3']['cumulative']),
                ('neural cut accounting',row.get('neuralCutSourceMm3')==dims['neuralCutSourceMm3']['cumulative']),
                ('hard constraint accounting',row.get('committedHardViolations')==len(sequence['violations'])),
                ('structure accounting',row.get('structureRemovedMm3')=={k:v['cumulative'] for k,v in dims['structureRemovedMm3'].items()})]:
                if not condition:errors.append(label+' mismatch')
        except Exception as error:errors.append('native audit failed: '+str(error))
    valid=bool(native_verified and proof.get('verified') is True and not errors and not row.get('failure') and
        endpoint.get('terminated') and not endpoint.get('truncated') and endpoint.get('validWithdrawal') and
        row.get('committedHardViolations')==0)
    if proof.get('verified') is not True:errors.append('native/adapter replay proof unavailable')
    quantity=lambda name:dims.get(name,{}).get('cumulative')
    struct=lambda name:dims.get('structureRemovedMm3',{}).get(name,{}).get('cumulative')
    cut=lambda name:dims.get('structureCutSourceMm3',{}).get(name,{}).get('cumulative')
    fraction=p.get('originalTargetTreatmentFraction')
    return dict(parent=row['parent'],pattern=pattern,method=row['method'],seed=row['seed'],tier=row['tier'],
        original_target_mm3=initial['originalTargetCells']*voxel,
        treated_mm3=p.get('originalTargetTreatmentMm3'),
        residual_mm3=p.get('residualOriginalTargetCells')*voxel if p.get('residualOriginalTargetCells') is not None else None,
        treatment_fraction=fraction,qualified_treatment_fraction=fraction if valid and fraction is not None else 0.,
        declared_task_complete=bool(valid and endpoint.get('completedTask')),
        raw_declared_task_claim=row.get('endpoint',{}).get('completedTask'),
        full_original_target_removal=p.get('residualOriginalTargetCells')==0 if p else None,
        valid_stop_withdrawal=valid,withdrawal_complete=endpoint.get('validWithdrawal'),
        classification=endpoint.get('classification','failed') if not errors else 'unverified',
        neural_removed_mm3=quantity('neuralRemovedMm3'),neural_cut_source_mm3=quantity('neuralCutSourceMm3'),
        motor_cortex_removed_mm3=struct('motor-cortex'),cst_removed_mm3=struct('corticospinal'),
        motor_cortex_cut_source_mm3=cut('motor-cortex'),cst_cut_source_mm3=cut('corticospinal'),
        brainstem_removed_mm3=struct('brainstem'),cerebellum_removed_mm3=struct('cerebellum'),
        hard_violations=measured_hard,bone_mm3=quantity('boneRemovedMm3'),
        debris_mm3=endpoint.get('debrisMm3'),fluid_mm3=endpoint.get('fluidMm3'),
        physical_requests=c.get('physicalCalls'),planning_requests=c.get('planningPhysicalCalls'),
        execution_requests=c.get('executionPhysicalCalls'),canceled_requests=c.get('canceledRequests'),
        actual_attempts=row.get('attempts'),actual_refusals=row.get('rejections'),preview_refusals=preview_refusals,
        planning_interruptions=len(interruptions),planning_stop_reasons=' | '.join(sorted(set(interruptions))),
        online_wall_seconds=c.get('wallSeconds'),candidate_seconds=c.get('candidateSeconds'),
        policy_feature_seconds=c.get('policySeconds'),reset_seconds=row.get('resetSeconds'),
        replay_verified=proof.get('verified',False),native_audit_verified=native_verified,
        replay_shared=proof.get('sharedReplay',False),replay_requests=proof.get('physicalCalls'),
        replay_wall_seconds=proof.get('wallSeconds'),missing_evidence=' | '.join(row.get('missingEvidence',[])),
        qualification_errors=' | '.join(errors),failure=row.get('failure',{}).get('error',''),
        source_scene_id=row.get('sourceSceneId'),final_state_id=row.get('finalStateId'),
        artifact=str(artifact.resolve()),artifact_sha256=sha(artifact))


def aggregate(rows):
    groups=[]
    for tier,method in sorted({(r['tier'],r['method']) for r in rows}):
        group=[r for r in rows if (r['tier'],r['method'])==(tier,method)]
        n=len(group)
        groups.append(dict(tier=tier,method=method,runs=n,layouts=len({r['parent'] for r in group}),
            mean_treated_mm3_known=mean([r['treated_mm3'] for r in group]),
            mean_residual_mm3_known=mean([r['residual_mm3'] for r in group]),
            mean_treatment_fraction_known=mean([r['treatment_fraction'] for r in group]),
            missing_treatment_outcomes=sum(r['treatment_fraction'] is None for r in group),
            mean_qualified_treatment_fraction=mean([r['qualified_treatment_fraction'] for r in group]),
            declared_task_completion_rate=sum(r['declared_task_complete'] for r in group)/n,
            full_original_removal_rate=sum(bool(r['full_original_target_removal']) for r in group)/n,
            valid_stop_withdrawal_rate=sum(r['valid_stop_withdrawal'] for r in group)/n,
            mean_neural_mm3=mean([r['neural_removed_mm3'] for r in group]),
            mean_motor_cortex_mm3=mean([r['motor_cortex_removed_mm3'] for r in group]),
            mean_cst_mm3=mean([r['cst_removed_mm3'] for r in group]),
            committed_hard_violations=sum(r['hard_violations'] or 0 for r in group),
            missing_harm_outcomes=sum(r['hard_violations'] is None or r['neural_removed_mm3'] is None for r in group),
            failures=sum(bool(r['failure']) for r in group),truncations=sum(r['classification']=='truncated' for r in group),
            actual_refusals=sum(r['actual_refusals'] or 0 for r in group),
            actual_attempts=sum(r['actual_attempts'] or 0 for r in group),
            actual_refusal_rate=(sum(r['actual_refusals'] or 0 for r in group)/sum(r['actual_attempts'] or 0 for r in group)
                if sum(r['actual_attempts'] or 0 for r in group) else None),
            preview_refusals=sum(r['preview_refusals'] for r in group),
            mean_physical_requests=mean([r['physical_requests'] for r in group]),
            mean_planning_requests=mean([r['planning_requests'] for r in group]),
            mean_execution_requests=mean([r['execution_requests'] for r in group]),
            canceled_requests=sum(r['canceled_requests'] or 0 for r in group),
            missing_computation_outcomes=sum(r['physical_requests'] is None or r['online_wall_seconds'] is None for r in group),
            mean_online_seconds=mean([r['online_wall_seconds'] for r in group]),
            maximum_debris_mm3=max((r['debris_mm3'] for r in group if r['debris_mm3'] is not None),default=None),
            maximum_fluid_mm3=max((r['fluid_mm3'] for r in group if r['fluid_mm3'] is not None),default=None),
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


def figure(rows,summary,evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors={'greedy':'#667085','search':'#176a8a','learned':'#8a54a2','untrained':'#b86f24'}
    markers={'low':'o','high':'s'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
    for r in summary:
        if r['mean_physical_requests'] is None:continue
        axes[0].scatter(r['mean_physical_requests'],100*r['mean_qualified_treatment_fraction'],
            color=colors[r['method']],marker=markers[r['tier']],s=85,label=f"{r['method']} · {r['tier']}")
    for tier,seed in sorted({(r['tier'],r['seed']) for r in rows if r['method']=='learned'}):
        group=[r for r in rows if r['tier']==tier and r['seed']==seed and r['method']=='learned']
        axes[0].scatter(mean([r['physical_requests'] for r in group]),100*mean([r['qualified_treatment_fraction'] for r in group]),
            color=colors['learned'],marker=markers[tier],s=22,alpha=.4)
    axes[0].set(xlabel='Online primitive requests per case',ylabel='Qualified original-target treatment (%)',ylim=(-2,102))
    axes[0].grid(alpha=.2);axes[0].legend(fontsize=7,loc='best')
    ordered=sorted(summary,key=lambda r:(['low','high'].index(r['tier']),['greedy','search','learned','untrained'].index(r['method'])))
    x=np.arange(len(ordered));width=.25
    for offset,key,label,color in [(-width,'declared_task_completion_rate','Task goal met','#176a8a'),
            (0.,'full_original_removal_rate','Raw 100% target removed','#8a54a2'),
            (width,'valid_stop_withdrawal_rate','Valid STOP + exit','#adb5bd')]:
        axes[1].bar(x+offset,[100*r[key] for r in ordered],width,label=label,color=color)
    axes[1].set_xticks(x,[r['tier']+'\n'+r['method'] for r in ordered],fontsize=7)
    axes[1].set(ylabel='Runs (%)',ylim=(0,105));axes[1].legend(fontsize=7);axes[1].grid(axis='y',alpha=.2)
    fig.suptitle('Matched generated post-access decisions · 16 held-out layouts',fontsize=12)
    fig.supxlabel('Case means; faint learned points show all frozen seeds. Earlier access costs and functional outcomes remain unknown.',fontsize=8)
    png=evidence/'quality-compute.png';svg=evidence/'quality-compute.svg'
    fig.savefig(png,dpi=240);fig.savefig(svg);plt.close(fig)
    return dict(png=str(png),pngSha256=sha(png),svg=str(svg),svgSha256=sha(svg),
        meaning='Qualified treatment is actual treatment with valid STOP/exterior exit, no truncation and no committed hard violation; raw treatment is retained separately.')


def html_package(manifest,rows,evidence):
    escape=lambda value:html.escape('unknown' if value is None else str(value))
    link=lambda path,label:f'<a href="{html.escape(Path(path).resolve().as_uri())}">{html.escape(label)}</a>'
    desktop=evidence.parent/'desktop';case=desktop/'frozen-cold-v2/desktop-saved.ressectionlab'
    viewer=desktop/'replay-viewer-v2.html';capture=desktop/'frozen-cold-v2/02-checked-alternatives.png'
    command=f'{ROOT}/scripts/launch_poc.sh --case "{case}"'
    summary=[]
    for r in manifest['summary']:
        pct=lambda key:'unknown' if r[key] is None else f'{100*r[key]:.1f}%'
        summary.append('<tr>'+''.join(f'<td>{escape(v)}</td>' for v in [r['tier'],r['method'],r['runs'],
            pct('mean_treatment_fraction_known'),pct('mean_qualified_treatment_fraction'),pct('declared_task_completion_rate'),pct('full_original_removal_rate'),
            pct('valid_stop_withdrawal_rate'),round(r['mean_physical_requests'],1) if r['mean_physical_requests'] is not None else None,
            round(r['mean_online_seconds'],2) if r['mean_online_seconds'] is not None else None])+'</tr>')
    cases=[]
    for r in rows:
        values=[r['parent'],r['method'],r['seed'],r['tier'],r['treated_mm3'],r['residual_mm3'],r['classification'],
            r['neural_removed_mm3'],r['motor_cortex_removed_mm3'],r['cst_removed_mm3'],r['actual_refusals'],
            r['preview_refusals'],r['physical_requests'],round(r['online_wall_seconds'],2) if r['online_wall_seconds'] is not None else None]
        cases.append(f'<tr data-method="{escape(r["method"])}" data-tier="{escape(r["tier"])}">'+
            ''.join(f'<td>{escape(v)}</td>' for v in values)+f'<td>{link(r["artifact"],"full row")}</td></tr>')
    paired_items=''.join(f'<li>{escape(r["tier"])} versus {escape(r["baseline"])}: '
        f'{100*r["mean_qualified_difference"]:+.1f} percentage points; delayed preparation '
        f'{100*r["delayed_benefit_difference"]:+.1f}; wins/ties/losses {r["wins"]}/{r["ties"]}/{r["losses"]}.</li>'
        for r in manifest['pairedSummary'])
    learning_items=''.join(f'<li><b>{escape(r["tier"])}:</b> {escape(r["interpretation"])}</li>' for r in manifest['learningFindings'])
    seed_rows=''.join('<tr>'+''.join(f'<td>{escape(v)}</td>' for v in [r['tier'],r['seed'],
        f"{100*r['mean_treatment_fraction']:.1f}%" if r['mean_treatment_fraction'] is not None else None,
        f"{100*r['mean_qualified_treatment_fraction']:.1f}%",
        f"{100*r['delayed_benefit_mean']:.1f}%",str(r['valid_stops'])+'/16'])+'</tr>'
        for r in manifest['seedSummaries'])
    limitations=''.join(f'<li>{escape(v)}</li>' for v in manifest['exclusions'])
    document=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RessectionLab · Sequential milestone</title><style>
body{{margin:0;background:#0d1513;color:#e4ece8;font:15px/1.5 system-ui,sans-serif}}main{{max-width:1320px;margin:auto;padding:36px 24px}}h1{{font-size:34px;margin:0}}h2{{margin-top:34px}}a{{color:#9de3ce}}code,pre{{background:#17231f;border-radius:8px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px;border-bottom:1px solid #33473f;text-align:left}}th{{background:#17231f;position:sticky;top:0}}.scroll{{overflow:auto;max-height:620px;border:1px solid #33473f;border-radius:10px}}.card{{padding:18px;border:1px solid #33473f;border-radius:12px;background:#111e19;margin:18px 0}}.muted{{color:#a9bbb3}}img{{max-width:100%;height:auto;border-radius:12px}}select{{padding:8px;margin:8px;background:#17231f;color:#e4ece8;border:1px solid #658275;border-radius:6px}}.warning{{color:#ffcf83}}small{{display:block}} </style><main>
<p class="muted">Generated engineering evidence · Patient operations completed: 0</p><h1>Sequential decisions, visible harm and checked stopping</h1>
<p>The saved desktop workspace contains two physically replayed alternatives. The matched post-access study retains every frozen case and seed, including failed or truncated endpoints.</p>
<div class="card"><b>Open the result</b><p>{link(case,'Native desktop workspace')} · {link(viewer,'Six actual replay frames')} · {link(evidence/'protocol.json','Frozen protocol')} · {link(evidence/'freeze.json','Checkpoint freeze')} · {link(manifest['resultRows']['path'],'All case rows')} · {link(manifest['pairedRows']['path'],'Paired differences')}</p><pre>{escape(command)}</pre></div>
<h2>Harm changes the selected physical plan</h2><p>With a zero neural-removal allowance the lateral plan treats 10/27 mm³ and leaves 17 mm³. A 1 mm³ allowance admits the central plan: 18/27 mm³ treated, 9 mm³ residual and 1 mm³ direct neural/motor membership removal. Each sequence retains 285 mm³ bone and 49 mm³ preparation; actual suction collects 236 mm³ debris to zero. Motor function remains unknown.</p>
<a href="{capture.as_uri()}"><img src="{capture.as_uri()}" alt="Frozen desktop comparison with two checked alternatives"></a>
<h2>Frozen comparison</h2><p>Means over {len(manifest['cohortCheck']['heldoutParents'])} held-out layouts; validation-selected learned seeds: {escape(manifest['cohortCheck']['learnedSeeds'])}. Qualified treatment requires a valid STOP/exterior exit, no truncation and no committed hard violation. Raw accepted treatment remains separately inspectable.</p>
<p>Raw target percentages average known native physical outcomes. Qualified target percentages average every run, assigning zero to unverified, failed or truncated endpoints.</p><div class="scroll"><table><thead><tr><th>Tier</th><th>Method</th><th>Runs</th><th>Raw target · known</th><th>Qualified target · all runs</th><th>Verified task goal</th><th>Raw100% removed</th><th>Verified STOP/exit</th><th>Requests</th><th>Online seconds</th></tr></thead><tbody>{''.join(summary)}</tbody></table></div>
<h2>Learning finding</h2><ul>{learning_items}</ul>
<img src="{Path(manifest['figure']['png']).as_uri()}" alt="Qualified treatment versus requested online compute and endpoint rates">
<h2>Paired case-level differences</h2><ul>{paired_items}</ul>
<h2>Variation across frozen seeds</h2><div class="scroll"><table><thead><tr><th>Tier</th><th>Seed</th><th>Raw target · known</th><th>Qualified target</th><th>Qualified delayed preparation</th><th>Verified STOP/exit</th></tr></thead><tbody>{seed_rows}</tbody></table></div>
<p>Training campaign: {manifest['campaignWallSeconds']/60:.1f} continuous minutes including the preserved diagnostic and repair interval. Current formal training requests: {manifest['currentTrainingPhysicalCalls']}; charged campaign upper bound: {manifest['trainingPhysicalCallsChargedUpperBound']}. Frozen evaluation: {manifest['evaluationWallSeconds']/60:.1f} minutes. Every row links to an actual native/adapter replay proof; identical programs share an explicitly identified successful proof.</p>
<h2>Inspect every run</h2><label>Method <select id="method"><option value="">All</option><option>greedy</option><option>search</option><option>learned</option><option>untrained</option></select></label><label>Tier <select id="tier"><option value="">All</option><option>low</option><option>high</option></select></label>
<div class="scroll"><table><thead><tr>{''.join(f'<th>{s}</th>' for s in ['Parent','Method','Seed','Tier','Treated mm³','Residual mm³','Endpoint','Neural mm³','Motor cortex mm³','CST mm³','Actual refusals','Preview refusals','Requests','Seconds','Evidence'])}</tr></thead><tbody id="cases">{''.join(cases)}</tbody></table></div>
<h2>Scope and remaining limitations</h2><p>Earlier access costs are unknown. These are compact generated continuations, separate from the desktop opening demonstration. Direct overlapping motor/CST memberships are anatomical constraints and exposures, not probabilities of neurological loss. The acquired UCSF-PDGM-0004 packet remains {link(evidence.parent/'acquired-ucsf0004-v1/index.html','inspection-only')} with the full source and execution blockers.</p><ul>{limitations}</ul>
<p class="warning">Delivery checks: {escape('; '.join(manifest['unmetCriteria']) or 'all declared evidence checks passed')}. The separate learning finding above reports whether the measured improvement criterion was met.</p>
<details><summary>Exact source, checkpoint and artifact bindings</summary><pre>{escape(json.dumps(manifest,indent=2))}</pre></details>
<script>const controls=[document.querySelector('#method'),document.querySelector('#tier')];function filter(){{for(const row of document.querySelectorAll('#cases tr'))row.hidden=(controls[0].value&&row.dataset.method!==controls[0].value)||(controls[1].value&&row.dataset.tier!==controls[1].value)}}for(const control of controls)control.addEventListener('change',filter);</script></main></html>'''
    path=evidence/'index.html';path.write_text(document)
    return dict(path=str(path),sha256=sha(path))


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
    audit=NativeAudit()
    rows=[compact(r,evidence,audit) for r in evaluation['rows']];summary=aggregate(rows);differences=paired(rows)
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
    learning_findings=[]
    for tier in ('low','high'):
        delayed=[r for r in differences if r['tier']==tier and r['baseline']=='greedy' and r['pattern']==0]
        per_seed=[dict(seed=seed,paired_qualified_difference=mean([r['qualified_fraction_difference'] for r in delayed if r['seed']==seed]))
            for seed in sorted({r['seed'] for r in delayed})]
        delta=mean([r['qualified_fraction_difference'] for r in delayed])
        learning_findings.append(dict(tier=tier,delayedBenefitMeanPairedDifference=delta,
            delayedBenefitPerSeed=per_seed,
            positiveMeanDelayedBenefit=delta is not None and delta>1e-10,
            allSeedsPositiveDelayedBenefit=bool(per_seed) and all(r['paired_qualified_difference']>1e-10 for r in per_seed),
            interpretation='Mean delayed-benefit gain observed; inspect every seed and the quality/compute tradeoff.' if delta is not None and delta>1e-10 else
                'Learning improvement on delayed preparation was not demonstrated; a completed negative experiment remains a valid delivery.'))
    unmet=list(freeze.get('unmetCriteria',[]))
    if len(rows)!=160:unmet.append(f'Expected160 formal rows; retained{len(rows)}')
    if not all(r['replay_verified'] for r in rows):unmet.append('Some frozen rows lack verified replay')
    if any(r['hard_violations'] for r in rows):unmet.append('Committed hard violations present')
    if any(r['hard_violations'] is None or r['neural_removed_mm3'] is None for r in rows):
        unmet.append('Required physical harm outcomes unavailable for some rows')
    if any(r['qualification_errors'] for r in rows):unmet.append('Some result bindings or independently measured outcomes do not agree')
    plot=figure(rows,summary,evidence)
    cost_totals={key:dict(knownTotal=sum(r[key] for r in rows if r[key] is not None),
        missingRows=sum(r[key] is None for r in rows)) for key in
        ('physical_requests','planning_requests','execution_requests','canceled_requests',
         'online_wall_seconds','candidate_seconds','policy_feature_seconds','reset_seconds')}
    manifest=dict(schema='resectionlab.sequential-milestone-delivery.v1',scope='generated post-access continuation',
        evidenceRoot=str(evidence),protocolSha256=sha(evidence/'protocol.json'),freezeSha256=sha(evidence/'freeze.json'),
        evaluationSha256=sha(evidence/'evaluation.json'),configHash=protocol['configHash'],sourceHashes=protocol['sourceHashes'],
        protocolSourceCommit=protocol['sourceCommit'],seeds=protocol['seeds'],checkpoints=freeze['checkpoints'],
        trainingPhysicalCallsChargedUpperBound=freeze['trainingPhysicalCalls'],
        currentTrainingPhysicalCalls=freeze.get('currentTrainingPhysicalCalls',freeze['trainingPhysicalCalls']),
        campaignWallSeconds=freeze['trainingWallSeconds'],priorTrainingCosts=freeze.get('priorTrainingCosts'),
        evaluationWallSeconds=evaluation['wallSeconds'],uniqueSuccessfulReplays=evaluation['uniqueReplays'],
        evaluationCostTotals=cost_totals,
        replayPhysicalCalls=evaluation['replayPhysicalCalls'],replayWallSeconds=evaluation['replayWallSeconds'],
        resultRows=dict(path=str(csv_path),sha256=sha(csv_path),count=len(rows)),
        pairedRows=dict(path=str(paired_path),sha256=sha(paired_path) if differences else None,count=len(differences)),
        summary=summary,seedSummaries=seed_summaries,pairedSummary=paired_summary,unmetCriteria=unmet,
        learningFindings=learning_findings,
        cohortCheck=coverage,
        figure=plot,
        exclusions=protocol['exclusions'],earlierAccessCosts=None,patientOperationsCompleted=0,
        analysisWallSeconds=time.monotonic()-started,
        independentNativeAudit=dict(uniqueSnapshots=len(audit.cache),physicalRequests=audit.requests,wallSeconds=audit.seconds,failures=audit.failures),
        prototypeDiagnosis='Preserved v1 interface diagnostic; no held-out evaluation. Actual recovery withdrawal added to all v2 methods before formal fitting.',
        qualifications='All source anatomical constraints/exposures are generated engineering declarations; functional motor/language and unsupported dynamics remain unknown.',
        commands=dict(evaluate=f'PYTHONPATH=src .venv/bin/python scripts/run_sequential_milestone.py --stage evaluate --generated-only --output {evidence}',
            summarize=f'PYTHONPATH=src .venv/bin/python scripts/summarize_sequential_milestone.py --evidence {evidence}'))
    save(docs/'sequential-milestone-manifest.json',manifest)
    manifest['savedPackage']=html_package(manifest,rows,evidence)
    save(docs/'sequential-milestone-manifest.json',manifest)
    lines=['# Generated sequential decision results','',
        f"Retained the exact {coverage['actualRows']}-row job grid across {len(coverage['heldoutParents'])} held-out layouts. Validation-selected learned seeds: {coverage['learnedSeeds']}. Treatment uses the full original target; declared-task completion and 100% removal are separate. Earlier access costs are unknown.",'',
        '| Tier | Method | Raw target · known | Qualified target · all runs | Verified task complete | Raw100% removed | Verified STOP/exit | Neural mm³ | Calls | Online seconds |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        value=lambda key,scale=1.,digits=2:'unknown' if r[key] is None else f'{r[key]*scale:.{digits}f}'
        lines.append(f"|{r['tier']}|{r['method']}|{value('mean_treatment_fraction_known',100.,1)}%|{value('mean_qualified_treatment_fraction',100.,1)}%|{100*r['declared_task_completion_rate']:.1f}%|{100*r['full_original_removal_rate']:.1f}%|{100*r['valid_stop_withdrawal_rate']:.1f}%|{value('mean_neural_mm3')}|{value('mean_physical_requests',digits=1)}|{value('mean_online_seconds')}|")
    lines += ['',f"Campaign wall time including the preserved diagnostic and repair interval:{freeze['trainingWallSeconds']/60:.1f}min. Charged training-call upper bound:{freeze['trainingPhysicalCalls']}; current formal requests:{manifest['currentTrainingPhysicalCalls']}. Frozen evaluation:{evaluation['wallSeconds']/60:.1f}min. Replay requests:{evaluation['replayPhysicalCalls']}.",'',
        'The case-level CSV retains actual and preview refusals, interrupted requests, cleanup, direct motor/CST memberships, every endpoint and all artifact hashes. Paired differences use zero qualified treatment for failed/truncated endpoints while retaining their actual physical treatment separately. Seed ranges are descriptive; no clinical or statistical-superiority claim follows from this small generated cohort.','',
        'Paired learned minus baseline (qualified treatment fraction):','']
    for r in paired_summary:
        lines.append(f"- {r['tier']} versus {r['baseline']}: mean {100*r['mean_qualified_difference']:+.1f} points; delayed-benefit {100*r['delayed_benefit_difference']:+.1f}; wins/ties/losses {r['wins']}/{r['ties']}/{r['losses']}.")
    for r in learning_findings:lines.extend(['',r['tier']+': '+r['interpretation']])
    lines += ['', '| Tier | Seed | Qualified target | Qualified delayed preparation | Verified STOP/exit |',
        '|---|---:|---:|---:|---:|']
    for r in seed_summaries:
        lines.append(f"|{r['tier']}|{r['seed']}|{100*r['mean_qualified_treatment_fraction']:.1f}%|{100*r['delayed_benefit_mean']:.1f}%|{r['valid_stops']}/16|")
    lines += ['',f"Unmet criteria:{'; '.join(unmet) if unmet else 'none in the declared generated experiment'}. Functional consequences, patient execution, earlier access costs and general physics/RL admission remain unavailable.",'']
    (docs/'sequential-milestone-results.md').write_text('\n'.join(lines))
    print(json.dumps(dict(summary=summary,paired=paired_summary,seeds=seed_summaries,unmet=unmet),indent=2))


if __name__=='__main__':main()
