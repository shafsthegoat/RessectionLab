#!/usr/bin/env python3
"""Independently replay saved assessments and actor inputs; never fit a policy.

Also repairs the first runner's descriptive accounting in a separate report:
RL attempts, public engine calls, and wrapper termination remain distinct.
Original records and failed/partial runs are preserved byte-for-byte.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import numpy as np
from stable_baselines3 import PPO
from patient_rescue_env import from_manifest, OPTIONS, sha
from train_patient_rescue import parameter_hash, export_workspace, write, SCRIPT, RecordingEnv, rollout


def complete_unused_search(env,report,run,out,check):
    """Resume only the unused original search allowance; never update PPO."""
    if report.get('error')!={'type':'TimeoutError','message':'Whole learning comparison wall budget exhausted'}:
        raise ValueError('Recovery requires the original whole-run wall timeout')
    if {u['seed']:u['realOptionDecisions'] for u in report['optimizerUpdates']}!={17:64,29:64}:
        raise ValueError('Both original frozen PPO fits must be complete')
    if any(row['seed']==29 for row in report['search']):raise ValueError('Second search is already complete')
    if report['totalOptionDecisions']!=sum(row['decisions'] for row in report['allEpisodes']):
        raise ValueError('Unrecorded completed decisions cannot be recovered safely')
    old=[row for row in report['allEpisodes'] if row['method']=='search' and row['seed']==29]
    used=sum(row['decisions'] for row in old)
    if not 0<used<64:raise ValueError('Expected an incomplete original seed29 search')
    rng=np.random.default_rng(29);candidates=[]
    for row in old:
        sequence=[str(v) for v in rng.permutation(OPTIONS)]
        payload=json.loads((run/row['path']).read_text())
        expected=[sequence[i] if i<len(sequence) else 'finish_verify' for i in range(row['decisions'])]
        if [d['option'] for d in payload['decisions']]!=expected:raise ValueError('Original search sampler sequence differs')
        candidates.append({**row,'sequence':sequence,'path':str(run/row['path'])})
    # The original traceback records timeout inside the next candidate's first
    # option. It yielded no complete option transition or reward.
    discarded=[str(v) for v in rng.permutation(OPTIONS)]
    recovery=dict(originalCompleteSearchDecisions=used,remainingAllowance=64-used,
        cancelledCandidate=discarded,cancelledCompleteOptionDecisions=0,
        cancelledPartialPublicCalls='unknown in original v1; not treated as zero',
        noCancelledOutcomeUsed=True,optimizerUpdates=0,episodes=[])
    wrapper=RecordingEnv(env,out,check);wrapper.serial=100;wrapper.mode='search';wrapper.seed_label=29
    while used<64:
        sequence=[str(v) for v in rng.permutation(OPTIONS)];limit=env.max_decisions
        env.max_decisions=min(limit,64-used)
        try:row=rollout(wrapper,sequence=sequence,seed=29)
        finally:env.max_decisions=limit
        row['path']=str(out/row['path']);used+=row['decisions']
        candidates.append(dict(sequence=sequence,**row));check()
    best=max(candidates,key=lambda row:row['reward'])
    report['search'].append(dict(seed=29,realOptionDecisions=used,candidates=candidates,selected=best,
        recovery=dict(recovery)))
    wrapper.mode='search-selected';assessment=rollout(wrapper,sequence=best['sequence'],seed=29)
    assessment['path']=str(out/assessment['path']);report['assessments'].append(assessment)
    recovery.update(addedSearchDecisions=64-recovery['originalCompleteSearchDecisions'],
        selectedAssessment=assessment,publicEngineCalls=wrapper.total_engine_calls,
        rlTransitionAttempts=wrapper.total_attempts)
    report['originalRunStatus']=report['status'];report['status']='completed'
    report['limitations'].append(f"Original run hit its 30-minute cap. A separate bounded audit completed only the unused {recovery['remainingAllowance']} search decisions for seed29 with frozen policies. The cancelled candidate yielded no complete option decision; its partial public-call count is unknown.")
    report['searchRecovery']=recovery
    return recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--complete-unused-search',action='store_true',
        help='Finish only unused seed29 search decisions after the original v1 wall timeout; no fitting')
    args=parser.parse_args();out=args.output.resolve();run=args.run.resolve()
    if out.exists() or not out.is_relative_to(ROOT/'testcases'):parser.error('Use a new testcases output directory')
    report_path=run/'result.json'
    report=json.loads(report_path.read_text());out.mkdir(parents=True)
    began=time.perf_counter();torch.set_num_threads(1)
    maximum=900. if args.complete_unused_search else 600.
    audit=dict(schema='resectionlab.patient-rescue-assessment-audit.v1',status='running',
        optimizerUpdates=0,originalReport=str(report_path),originalReportSha256=sha(report_path),
        executedScriptSha256=sha(__file__),assessments=[],maximumWallSeconds=maximum,maximumRssBytes=3*1024**3)
    inputs={report_path:sha(report_path)}
    def check():
        if time.perf_counter()-began>maximum:raise TimeoutError('Assessment audit wall budget exhausted')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)>3*1024**3:
            raise MemoryError('Assessment audit RSS budget exhausted')
    try:
        if sha(report['manifest'])!=report['manifestSha256']:raise ValueError('Original run manifest changed')
        env,manifest=from_manifest(report['manifest'],check=check)
        for path in (Path(report['manifest']),Path(manifest['source']),Path(manifest['taskStartSnapshot'])):
            inputs[path]=sha(path)
        audit['preparation']=env.preparation
        for row in report['assessments']:row['path']=str(run/row['path'])
        if args.complete_unused_search:
            audit['searchRecovery']=complete_unused_search(env,report,run,out,check)
            write(out/'recovered-comparison.json',report)
        models={}
        for update in report['optimizerUpdates']:
            seed=update['seed'];checkpoint=run/update['checkpoint']
            if sha(checkpoint)!=update['checkpointSha256']:raise ValueError('Saved PPO checkpoint changed')
            inputs[checkpoint]=sha(checkpoint)
            initial=PPO('MlpPolicy',env,n_steps=report['settings']['n_steps'],
                batch_size=report['settings']['batch_size'],n_epochs=report['settings']['n_epochs'],
                learning_rate=report['settings']['learning_rate'],gamma=report['settings']['gamma'],
                ent_coef=report['settings']['ent_coef'],policy_kwargs=dict(net_arch=dict(pi=[32],vf=[32])),
                seed=seed,device='cpu',verbose=0)
            if parameter_hash(initial)!=update['initialSha256']:raise ValueError('Untrained initialization differs')
            trained=PPO.load(checkpoint,device='cpu')
            if parameter_hash(trained)!=update['finalSha256']:raise ValueError('Trained policy differs')
            steps=sorted({int(value['step'].item()) for value in trained.policy.optimizer.state.values() if 'step' in value})
            audit.setdefault('checkpointOptimizerStepCounters',[]).append(dict(seed=seed,
                adamParameterStepCounters=steps,provenance='Loaded unchanged saved Adam optimizer state; no fitting in audit'))
            models[(seed,'untrained')]=initial;models[(seed,'trained')]=trained
        for assessment in report['assessments']:
            check();path=run/assessment['path'];inputs[path]=sha(path)
            saved=json.loads(path.read_text());started=time.perf_counter();obs,_=env.reset(seed=saved['seed'])
            calls=env.raw.engine.public_calls;actor=models.get((saved['seed'],saved['method']))
            rows=[];inference=0.
            for index,decision in enumerate(saved['decisions']):
                check();probabilities=None
                if actor is not None:
                    before=time.perf_counter();chosen,_=actor.predict(obs,deterministic=True)
                    tensor,_=actor.policy.obs_to_tensor(obs)
                    with torch.no_grad():probabilities=actor.policy.get_distribution(tensor).distribution.probs.cpu().numpy()[0].tolist()
                    inference+=time.perf_counter()-before
                    if int(chosen)!=OPTIONS.index(decision['option']):raise ValueError('Saved actor decision differs')
                elif saved['method']=='scripted':
                    if decision['option']!=(SCRIPT[index] if index<len(SCRIPT) else 'finish_verify'):
                        raise ValueError('Saved scripted decision differs')
                elif saved['method']=='search-selected':
                    selection=next(row for row in report['search'] if row['seed']==saved['seed'])['selected']['sequence']
                    if decision['option']!=(selection[index] if index<len(selection) else 'finish_verify'):
                        raise ValueError('Saved search-selected decision differs')
                else:raise ValueError('No independent actor/sequence for assessment')
                obs,_,terminated,truncated,info=env.step(OPTIONS.index(decision['option']))
                actual=env.decisions[-1]
                for key in ('option','requests','skills','firstAttempt','attempts','stateId','policyObservationSha256'):
                    if actual[key]!=decision[key]:raise ValueError('Assessment input/request/skill replay differs: '+key)
                rows.append(dict(option=decision['option'],policyObservationSha256=actual['policyObservationSha256'],
                    probabilities=probabilities,completeDecisionSeconds=actual['timing']['complete']))
            if env.native_snapshot()!=saved['native'] or env.raw.attempts!=saved['attempts'] or env.raw.total_reward!=saved['reward']:
                raise ValueError('Assessment physical state or rewards differ')
            if info['termination_reason']!=saved['reason']:raise ValueError('Assessment terminal reason differs')
            audit['assessments'].append(dict(method=saved['method'],seed=saved['seed'],trajectory=str(path),
                exactNativeAndRewardReplay=True,policyInputsRequestsAndSkillsExact=True,terminalReasonExact=True,
                actorPredictionsExact=actor is not None,rlTransitionAttempts=len(env.raw.attempts),
                publicEngineCalls=env.raw.engine.public_calls-calls,decisions=rows,
                inferenceAndProbabilityAuditSeconds=inference,wallSeconds=time.perf_counter()-started,
                nestedOriginalTerminalReason=saved['info']['termination_reason'],actualTerminalReason=info['termination_reason']))
            write(out/'audit-progress.json',audit)
            print(json.dumps(dict(method=saved['method'],seed=saved['seed'],replayExact=True,
                actorPredictionsExact=actor is not None,wallSeconds=time.perf_counter()-started)),flush=True)
        if any(sha(path)!=digest for path,digest in inputs.items()):raise ValueError('Audit modified an input')
        audit.update(status='passed',allInputsPreserved=True)
        derived=copy.deepcopy(report)
        for row in derived['assessments']:row['path']=str(run/row['path'])
        derived.update(originalRun=str(run),independentAudit=str(out/'audit.json'),
            accountingNote='Original v1 publicPrimitive counts are RL transition attempts; exact public engine calls are reported by independent audit.',
            independentAuditedAssessmentPublicEngineCalls=sum(row['publicEngineCalls'] for row in audit['assessments']))
        derived['nativeWorkspace']=export_workspace(manifest,derived,out)
        write(out/'result.json',derived)
    except BaseException as error:
        audit.update(status='failed',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        audit.update(wallSeconds=time.perf_counter()-began,
            peakResidentBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        write(out/'audit.json',audit)


if __name__=='__main__':main()
