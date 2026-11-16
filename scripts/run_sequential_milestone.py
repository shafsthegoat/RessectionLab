#!/usr/bin/env python3
"""Generated-only pilot, bounded RL and frozen comparison. Always writes summary.

No source acquisition, resident procedure policy, dynamic physics, or patient
training. Acquired packet is an optional inspection reference only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from resectionlab.surgical_sequential_task import (COMMON_CONFIG, generated_task,
    execute_skill, observation, training_admission)
from resectionlab.surgical_sequential_benchmark import (ComputeTier, FEATURE_NAMES,
    OnlineLimit, feature_matrix, run_episode, verify_replay)

ROOT=Path(__file__).resolve().parents[1]
TRAIN=list(range(32));VALIDATION=list(range(100,108));HELDOUT=list(range(200,216))
SEEDS=(17,43,71)
REVIEW_FILES=('src/resectionlab/surgical_sequential_task.py',
    'src/resectionlab/surgical_sequential_benchmark.py','scripts/run_sequential_milestone.py',
    'src/resectionlab/surgical_rl_env.py','src/resectionlab/surgical_procedure_policy.py',
    'src/resectionlab/surgical_planning_admission.py','src/resectionlab/surgical_consequences.py',
    'src/resectionlab/surgical_joint_cranial_search.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,sort_keys=True)+'\n');tmp.replace(path)


def read(path):return json.loads(Path(path).read_text())


def source_hashes():return {p:sha(ROOT/p) for p in REVIEW_FILES}


class Selector(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.Tanh(),nn.Linear(32,1))
        self.critic=nn.Sequential(nn.Linear(len(FEATURE_NAMES),32),nn.Tanh(),nn.Linear(32,1))

    def select(self,features,mask,stochastic=False):
        x=torch.as_tensor(features);logits=self.actor(x).squeeze(-1)
        logits=logits.masked_fill(~torch.as_tensor(mask),-1.e9)
        distribution=torch.distributions.Categorical(logits=logits)
        index=distribution.sample() if stochastic else logits.argmax()
        value=self.critic(x.mean(dim=0)).squeeze()
        return int(index),dict(logprob=distribution.log_prob(index),entropy=distribution.entropy(),value=value)


def pilot(out):
    """At most15min, about200 actual representative primitive transitions."""
    started=time.monotonic();pilot_hashes=source_hashes();calls=0;rows=[];resets=0;simulation_seconds=0.
    def charge():
        nonlocal calls
        if calls>=200 or time.monotonic()-started>=900:raise OnlineLimit('pilot-cap')
        calls+=1
    while calls<200 and time.monotonic()-started<900:
        parent=1000+resets%4;task=generated_task(parent);engine=task.initial_engine();resets+=1
        # Alternating preparation creates narrow-tip as well as default-tip
        # physical traces. This is a timing workload, not a demonstration label.
        inventory=[s for s in task.skills if s.kind=='prepare' and s.radius==.4][:resets%2]
        inventory += [s for s in task.skills if s.kind=='work']
        try:
            for skill in inventory:
                before=time.monotonic();receipt=execute_skill(task,engine,skill,charge=charge)
                simulation_seconds+=time.monotonic()-before
                rows.append(dict(parent=parent,kind=skill.kind,status=receipt['status'],
                    primitiveCalls=len(receipt['receipts']),wallSeconds=receipt['wallSeconds'],
                    targetMm3=receipt['targetIncrementMm3']))
                if receipt['status']=='interrupted':break
        except OnlineLimit as error:
            rows.append(dict(parent=parent,kind=skill.kind,status='interrupted',reason=str(error),
                acceptedActions=len(engine.actions),snapshot=engine.snapshot()))
            break
    # Measure feature and policy inference without any fitting.
    model=Selector();task=generated_task(1000);engine=task.initial_engine()
    feature_started=time.monotonic();features=feature_matrix(observation(task,engine),task.skills)
    feature_seconds=time.monotonic()-feature_started
    inference_started=time.monotonic()
    with torch.no_grad():
        for _ in range(200):model.select(features,np.ones(len(task.skills),bool),False)
    inference_seconds=(time.monotonic()-inference_started)/200
    dev=[]
    devtier=ComputeTier('development',160,120.,3,3,24)
    for parent in range(1000,1004):
        for method in ('greedy','search'):
            if time.monotonic()-started>780:break
            task=generated_task(parent);row=run_episode(task,method,devtier,deadline=started+840)
            row['replay']=verify_replay(task,row,deadline=started+900);dev.append(row)
            save(out/f'dev-{parent}-{method}.json',row)
    improvement=[]
    for parent in range(1000,1004):
        pair={r['method']:r for r in dev if r['parent']==parent}
        if len(pair)==2:
            improvement.append(dict(parent=parent,pairedTreatmentDifferenceMm3=
                pair['search']['progress']['originalTargetTreatmentMm3']-pair['greedy']['progress']['originalTargetTreatmentMm3']))
    if not any(r['pairedTreatmentDifferenceMm3']>0 for r in improvement):
        raise ValueError('DEVELOPMENT_LOOKAHEAD_ADVANTAGE_NOT_ESTABLISHED')
    per_call=simulation_seconds/max(1,calls)
    replay_seconds=float(np.mean([r['replay']['wallSeconds'] for r in dev]))
    learned_estimate=per_call*12+feature_seconds*8+inference_seconds*8
    greedy_estimate=float(np.mean([r['computation']['wallSeconds'] for r in dev if r['method']=='greedy']))
    search_estimate=float(np.mean([r['computation']['wallSeconds'] for r in dev if r['method']=='search']))
    # Measured projection includes all16 layouts, both tiers, all3 learned
    # seeds, deterministic controls once, and conservative replay per row.
    evaluation_estimate=2*16*(greedy_estimate+search_estimate+3*learned_estimate+5*replay_seconds)
    tiers=[ComputeTier('low',72,max(20.,per_call*85),2,2,12),
        ComputeTier('high',120,max(35.,per_call*145),3,3,24)]
    if evaluation_estimate>3300:
        tiers=[ComputeTier('low',56,25.,2,2,8),ComputeTier('high',88,40.,3,2,16)]
    report=dict(schema='resectionlab.sequential-timing-pilot.v1',sourceHashes=pilot_hashes,
        sourceChangedDuringPilot=pilot_hashes!=source_hashes(),primitiveTransitions=calls,
        wallSeconds=time.monotonic()-started,simulationSeconds=simulation_seconds,
        secondsPerPrimitive=per_call,featureSeconds=feature_seconds,policySeconds=inference_seconds,
        replayMeanSeconds=replay_seconds,developmentPairedDifferences=improvement,
        frozenEvaluationProjectionSeconds=evaluation_estimate,tiers=[asdict(t) for t in tiers],rows=rows)
    save(out/'pilot.json',report)
    return report


def protocol(out,pilot_report):
    if pilot_report.get('sourceHashes')!=source_hashes() or pilot_report.get('sourceChangedDuringPilot'):
        raise ValueError('CURRENT_FROZEN_SOURCE_TIMING_PILOT_REQUIRED')
    cohort=[]
    for split,parents in [('train',TRAIN),('validation',VALIDATION),('heldout',HELDOUT)]:
        for parent in parents:
            task=generated_task(parent);training_admission(task)
            cohort.append(dict(split=split,**task.audit))
    hashes=[r['geometryHash'] for r in cohort]
    if len(set(hashes))!=len(hashes):raise ValueError('DUPLICATED_PARENT_GEOMETRY')
    result=dict(schema='resectionlab.sequential-benchmark-protocol.v1',
        sourceCommit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sourceHashes=source_hashes(),config=asdict(COMMON_CONFIG),configHash=COMMON_CONFIG.fingerprint,
        cohort=cohort,seeds=list(SEEDS),onlineTiers=pilot_report['tiers'],
        training=dict(algorithm='masked finite-skill episodic actor-critic / REINFORCE with learned baseline',
            model='shared candidate MLP24→32tanh→1; separate mean-feature value MLP24→32tanh→1',
            optimizer='Adam',learningRate=.003,entropyCoefficient=.02,valueCoefficient=.5,
            epochs=8,episodesPerSeed=256,transitionCeilingCampaign=20000,maxWallSecondsCampaign=7200,
            discountPerSecond=COMMON_CONFIG.discount_per_second,warmStart=False,
            checkpointSelection='validation mean treatment fraction with valid withdrawal and no truncation, then fewer refusals; earliest tie'),
        endpoint='explicit checked STOP; zero debris/fluid, whole-tool withdrawal; canonical original-target .5 /3 octants each .05',
        skillRewards='accepted target/27 - accepted neural/27 - .05 bone/27 - .01 modeled seconds - .01 primitive attempts (including STOP)',
        exclusions=['acquired anatomy','graph vascular','thermal','continuum','CUSA v3','resident C++ policy','whole-procedure success','clinical motor/language outcome'],
        information='full supplied source memberships/coverage, removal/tool/current native state, cumulative costs, budgets and accepted history; IDs/split/pattern/hashes and branch outcomes excluded from features',
        compute='all speculative/executed primitive requests charged, canceled atomic requests separately labeled and receive no treatment/attempt reward; inventory/feature/policy/search wall time included; replay/training separate',
        targetScope='supplied post-access continuation, earlier access costs unknown',
        hardware=dict(platform=platform.platform(),python=sys.version,torch=torch.__version__,cpu='Apple M2 Pro',memoryGiB=16,device='cpu',threads=1),
        acceptance=dict(heldoutLayouts=16,committedHardViolations=0,cleanupDebrisMm3=0,
            learningClaim='requires measured paired improvement over greedy and useful online compute tradeoff; negative result retained'))
    save(out/'protocol.json',result);return result


def validate_protocol(record):
    if (record.get('sourceHashes')!=source_hashes() or record.get('config')!=asdict(COMMON_CONFIG) or
            record.get('configHash')!=COMMON_CONFIG.fingerprint or record.get('seeds')!=list(SEEDS)):
        raise ValueError('CURRENT_SOURCE_AND_TASK_PROTOCOL_REQUIRED')
    expected=[dict(split=split,**generated_task(parent).audit) for split,parents in
        [('train',TRAIN),('validation',VALIDATION),('heldout',HELDOUT)] for parent in parents]
    if record.get('cohort')!=expected:raise ValueError('FROZEN_GROUPED_COHORT_CHANGED')


def require_review(path,protocol_record):
    validate_protocol(protocol_record)
    if not path:raise ValueError('INDEPENDENT_GENERATED_TRAINING_REVIEW_REQUIRED')
    review=read(path)
    if (review.get('trainingAdmitted') is not True or review.get('scope')!='resectionlab.generated-sequential-skills.v1' or
            review.get('sourceHashes')!=source_hashes() or review.get('configHash')!=COMMON_CONFIG.fingerprint):
        raise ValueError('INDEPENDENT_TRAINING_REVIEW_NOT_CURRENT_OR_NOT_ADMITTED')
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def train(out,record,review_path):
    reviewed=require_review(review_path,record)
    started=time.monotonic();calls=0;logs=[];checkpoints=[]
    train_tier=ComputeTier('training',64,45.,1,2,1)
    val_tier=ComputeTier(**record['onlineTiers'][-1])
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed)
        model=Selector();optimizer=torch.optim.Adam(model.parameters(),lr=.003)
        best=(-1.,-1.e9);updates=0;last=None
        seed_out=out/'checkpoints'/str(seed);seed_out.mkdir(parents=True,exist_ok=True)
        torch.save(model.state_dict(),seed_out/'initial.pt')
        torch.save(model.state_dict(),seed_out/'selected.pt');selected_epoch=-1;selected_updates=0
        for epoch in range(record['training']['epochs']):
            for parent in rng.permutation(TRAIN):
                if time.monotonic()-started>=7200 or calls>=20000:break
                task=generated_task(int(parent))
                bounded_tier=ComputeTier('training',min(train_tier.max_physical_calls,20000-calls),45.,1,2,1)
                row=run_episode(task,'learned',bounded_tier,selector=model.select,stochastic=True,
                    seed=seed,gradient_records=True,deadline=started+7200)
                gradient=row.pop('_gradient');calls+=row['computation']['physicalCalls']
                returns=[];value=0.
                for receipt in reversed(row['skillHistory']):
                    value=receipt['reward']+receipt['discount']*value;returns.append(value)
                returns=list(reversed(returns))
                terms=[]
                for sample,target in zip(gradient,returns):
                    target=torch.as_tensor(target,dtype=torch.float32)
                    advantage=target-sample['value']
                    terms.append(-sample['logprob']*advantage.detach()+.5*nn.functional.smooth_l1_loss(sample['value'],target)-.02*sample['entropy'])
                if terms:
                    optimizer.zero_grad();loss=torch.stack(terms).mean();loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates+=1
                row.update(epoch=epoch,update=updates,trainingWallSeconds=time.monotonic()-started)
                save(seed_out/f'episode-{epoch:02d}-{int(parent):03d}.json',row)
                logs.append(dict(seed=seed,epoch=epoch,parent=int(parent),update=updates,reward=row['reward'],
                    treatmentFraction=row['progress']['originalTargetTreatmentFraction'],
                    classification=row['endpoint']['classification'],physicalCalls=row['computation']['physicalCalls']))
            torch.save(model.state_dict(),seed_out/f'epoch-{epoch:02d}.pt')
            validation=[]
            with torch.no_grad():
                for parent in VALIDATION:
                    if time.monotonic()-started>=7200 or calls>=20000:break
                    bounded_tier=ComputeTier(val_tier.name,min(val_tier.max_physical_calls,20000-calls),
                        val_tier.max_wall_seconds,val_tier.horizon,val_tier.beam_width,val_tier.max_expansions)
                    row=run_episode(generated_task(parent),'learned',bounded_tier,selector=model.select,seed=seed,
                        deadline=started+7200)
                    validation.append(row);calls+=row['computation']['physicalCalls']
            save(seed_out/f'validation-{epoch:02d}.json',validation)
            if len(validation)!=len(VALIDATION):
                save(seed_out/f'validation-{epoch:02d}-incomplete.json',dict(
                    completed=len(validation),required=len(VALIDATION),eligibleForSelection=False))
                break
            score=(float(np.mean([r['progress']['originalTargetTreatmentFraction'] if
                r['endpoint']['validWithdrawal'] and not r['endpoint']['truncated'] else 0. for r in validation])),
                -float(np.mean([r['rejections'] for r in validation])))
            if score>best:
                best=score;torch.save(model.state_dict(),seed_out/'selected.pt');selected_epoch=epoch;selected_updates=updates
            last=epoch
            save(out/'training-progress.json',dict(logs=logs,physicalCalls=calls,wallSeconds=time.monotonic()-started,review=reviewed))
            if time.monotonic()-started>=7200 or calls>=20000:break
        selected=seed_out/'selected.pt'
        checkpoints.append(dict(seed=seed,path=str(selected.resolve()),sha256=sha(selected),
            updates=updates,selectedUpdates=selected_updates,selectedEpoch=selected_epoch,lastEpoch=last,validationScore=list(best),
            initialSha256=sha(seed_out/'initial.pt'),rewardDrivenUpdates=selected_updates>0,
            label='reward-updated-selector' if selected_updates>0 else 'untrained-initial-fallback'))
        save(out/'training.json',dict(checkpoints=checkpoints,physicalCalls=calls,
            wallSeconds=time.monotonic()-started,review=reviewed,logs=logs))
        if time.monotonic()-started>=7200 or calls>=20000:break
    frozen=dict(schema='resectionlab.frozen-sequential-checkpoints.v1',protocolSha256=sha(out/'protocol.json'),
        sourceHashes=source_hashes(),configHash=COMMON_CONFIG.fingerprint,checkpoints=checkpoints,
        trainingPhysicalCalls=calls,trainingWallSeconds=time.monotonic()-started,review=reviewed,
        unmetCriteria=[f"seed-{c['seed']}-selected-checkpoint-has-no-reward-updates" for c in checkpoints if not c['rewardDrivenUpdates']] +
            ([f'only-{len(checkpoints)}-of-three-seeds-completed'] if len(checkpoints)!=3 else []))
    save(out/'freeze.json',frozen);return frozen


def evaluate(out,record,*,controls_only=False):
    validate_protocol(record)
    frozen=read(out/'freeze.json') if not controls_only else None
    if frozen and (frozen['sourceHashes']!=source_hashes() or frozen['protocolSha256']!=sha(out/'protocol.json')):
        raise ValueError('FROZEN_EVALUATION_SOURCE_OR_PROTOCOL_CHANGED')
    models={};checkpoint_labels={}
    for c in frozen['checkpoints'] if frozen else []:
        if sha(c['path'])!=c['sha256']:raise ValueError('CHECKPOINT_CHANGED_AFTER_FREEZE')
        model=Selector();model.load_state_dict(torch.load(c['path'],map_location='cpu',weights_only=True));model.eval();models[c['seed']]=model
        checkpoint_labels[c['seed']]='learned' if c.get('selectedUpdates',0)>0 and c.get('rewardDrivenUpdates') else 'untrained'
    started=time.monotonic();rows=[];replays={};replay_attempts=[]
    failures=[]
    for tier_spec in record['onlineTiers']:
        tier=ComputeTier(**tier_spec)
        for parent in HELDOUT:
            for method,seed in [('greedy',0),('search',0),*[(checkpoint_labels[s],s) for s in models]]:
                row=dict(parent=parent,method=method,seed=seed,tier=tier.name)
                try:
                    if time.monotonic()-started>=3600:raise TimeoutError('Global frozen evaluation60min cap')
                    task=generated_task(parent)
                    with torch.no_grad():
                        row=run_episode(task,'learned' if method=='untrained' else method,tier,
                            selector=models[seed].select if method in ('learned','untrained') else None,
                            seed=seed,deadline=started+3600)
                        row['method']=method
                    key=hashlib.sha256(json.dumps(dict(scene=task.scene.identity,
                        history=[dict(kind=s['kind'],actions=[r['action'] for r in s['receipts']]) for s in row['skillHistory']],
                        classification=row['endpoint']['classification'],attempts=row['attempts'],invalid=row['rejections']),sort_keys=True).encode()).hexdigest()
                    if key not in replays:
                        try:
                            proof=verify_replay(task,row,deadline=started+3600)
                        except Exception as error:
                            replay_attempts.append(getattr(error,'receipt',dict(verified=False,physicalCalls=0,wallSeconds=0.)))
                            raise
                        replay_attempts.append(proof);replays[key]=proof
                    row['replay']=dict(**replays[key],cacheKey=key,
                        sharedReplay=any(r.get('replay',{}).get('cacheKey')==key for r in rows))
                except Exception as error:
                    failure=dict(errorType=type(error).__name__,error=str(error),traceback=traceback.format_exc())
                    row.update(failure=failure,replay=getattr(error,'receipt',dict(verified=False,physicalCalls=0,wallSeconds=0.)))
                    failures.append(dict(parent=parent,method=method,seed=seed,tier=tier.name,**failure))
                rows.append(row);save(out/'results'/f'{tier.name}-{parent}-{method}-{seed}.json',row)
                save(out/'evaluation-progress.json',dict(completedRows=len(rows),wallSeconds=time.monotonic()-started))
    save(out/'evaluation.json',dict(rows=rows,wallSeconds=time.monotonic()-started,uniqueReplays=len(replays),
        failures=failures,
        replayAttempts=replay_attempts,
        replayPhysicalCalls=sum(r['physicalCalls'] for r in replay_attempts),
        replayWallSeconds=sum(r['wallSeconds'] for r in replay_attempts)))
    if failures:raise ValueError('Frozen evaluation retained failures; inspect evaluation.json before any claims')
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'testcases/overnight-sequential-v1/benchmark')
    parser.add_argument('--stage',choices=('pilot','protocol','train','evaluate','controls','all'),default='pilot')
    parser.add_argument('--training-admission-review',type=Path)
    parser.add_argument('--generated-only',action='store_true',default=True)
    parser.add_argument('--acquired-packet',type=Path,help='optional inspection packet; never an executable task')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    summary=dict(stage=args.stage,startedUtc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running',sourceHashes=source_hashes())
    try:
        if args.stage in ('pilot','all'):pilot(args.output)
        if args.stage in ('protocol','all'):protocol(args.output,read(args.output/'pilot.json'))
        if args.stage in ('train','all'):train(args.output,read(args.output/'protocol.json'),args.training_admission_review)
        if args.stage in ('evaluate','all','controls'):
            rows=evaluate(args.output,read(args.output/'protocol.json'),controls_only=args.stage=='controls')
            summary['rows']=len(rows)
        if args.acquired_packet:summary['acquiredInspectionPacket']=dict(path=str(args.acquired_packet.resolve()),sha256=sha(args.acquired_packet))
        summary['status']='complete'
    except BaseException as error:
        summary.update(status='failed',errorType=type(error).__name__,error=str(error),traceback=traceback.format_exc())
        raise
    finally:
        summary['finishedUtc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());save(args.output/f'summary-{args.stage}.json',summary)


if __name__=='__main__':main()
