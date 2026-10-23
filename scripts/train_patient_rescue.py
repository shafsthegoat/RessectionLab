#!/usr/bin/env python3
"""One bounded PPO/search comparison on a successfully authored rescue task.

No parameter sweep or checkpoint promotion. Every optimization sample is a real
task transition. Historical setup burden, reset costs and primitive counts remain
reported separately. Failed/partial runs retain their actual records.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from stable_baselines3 import PPO
from gymnasium import Wrapper
from patient_rescue_env import from_manifest, OPTIONS, sha
from resectionlab.workspace_bundle import load_workspace, save_workspace

SETTINGS=dict(seeds=[17,29],train_decisions=64,search_decisions=64,n_steps=32,
    batch_size=16,n_epochs=4,learning_rate=.001,gamma=.97,ent_coef=.02,
    max_decisions=12,max_wall_seconds=2400.,max_rss_bytes=3*1024**3)
SCRIPT=('coagulate_center','coagulate_lower','coagulate_forward','aspirate_pool','finish_verify')


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def scalar(value):
    value=float(value)
    return value if math.isfinite(value) else None


def parameter_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,value in sorted(model.policy.state_dict().items()):
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class RecordingEnv(Wrapper):
    def __init__(self,env,output,check):
        super().__init__(env);self.output=output;self.check=check
        self.episodes=[];self.serial=0;self.mode='initial';self.seed_label=0
        self.total_decisions=0;self.total_attempts=0;self.total_engine_calls=0
        self.reset_seconds=0.;self.episode_started=None

    def reset(self,**kwargs):
        if self.episode_started is not None and self.env.decisions:
            self.finish('external_reset')
        before=time.perf_counter();result=self.env.reset(**kwargs)
        self.reset_seconds+=time.perf_counter()-before;self.episode_started=time.perf_counter()
        return result

    def step(self,action):
        self.check();before=len(self.env.raw.attempts);calls=self.env.raw.engine.public_calls
        result=self.env.step(action)
        self.total_decisions+=1;self.total_attempts+=len(self.env.raw.attempts)-before
        self.total_engine_calls+=self.env.raw.engine.public_calls-calls
        if result[2] or result[3]:self.finish(result[4]['termination_reason'],result[4])
        return result

    def finish(self,reason,terminal_info=None):
        if self.episode_started is None:return
        native=self.env.native_snapshot();info=copy.deepcopy(terminal_info or self.env.raw._info())
        name=f'{self.serial:03d}-{self.mode}-seed-{self.seed_label}.json';self.serial+=1
        payload=dict(schema='resectionlab.patient-rescue-trajectory.v1',method=self.mode,seed=self.seed_label,
            environmentId=self.env.raw.environment_id,decisions=copy.deepcopy(self.env.decisions),
            attempts=copy.deepcopy(self.env.raw.attempts),native=native,info=info,
            incompleteOption=copy.deepcopy(self.env.inflightOption),
            reward=self.env.raw.total_reward,reason=reason)
        write(self.output/name,payload)
        row=dict(method=self.mode,seed=self.seed_label,path=name,reward=payload['reward'],
            success=info['success'],reason=reason,stateId=native['stateId'],
            decisions=len(self.env.decisions),rlTransitionAttempts=len(self.env.raw.attempts),
            publicEngineCalls=self.env.raw.engine.public_calls-self.env.raw._prepared.public_calls,
            wallSeconds=time.perf_counter()-self.episode_started,
            modeledField=info['modeledField'],metrics=info['metrics'])
        self.episodes.append(row);self.episode_started=None
        print(json.dumps(dict(episode=name,method=self.mode,success=row['success'],
            decisions=row['decisions'],reward=row['reward'],wallSeconds=row['wallSeconds'])),flush=True)
        return row


class AuditedPPO(PPO):
    def train(self):
        before=parameter_hash(self);started=time.perf_counter()
        original_step=self.policy.optimizer.step;steps=0
        def counted_step(*args,**kwargs):
            nonlocal steps
            steps+=1
            return original_step(*args,**kwargs)
        self.policy.optimizer.step=counted_step
        try:super().train()
        finally:self.policy.optimizer.step=original_step
        self.update_receipts.append(dict(beforeSha256=before,afterSha256=parameter_hash(self),
            optimizerSteps=steps,
            wallSeconds=time.perf_counter()-started,numTimesteps=self.num_timesteps,
            metrics={k:scalar(v) for k,v in self.logger.name_to_value.items()
                     if k.startswith('train/') and np.ndim(v)==0}))


def rollout(env,sequence=None,model=None,seed=0):
    obs,_=env.reset(seed=seed);start=time.perf_counter();inference=0.;outer=[]
    for index in range(env.env.max_decisions):
        env.check();before=time.perf_counter()
        if model is not None:action,_=model.predict(obs,deterministic=True);action=int(action)
        else:action=OPTIONS.index(sequence[index] if index<len(sequence) else 'finish_verify')
        chosen=time.perf_counter();inference+=chosen-before
        obs,_,terminated,truncated,_=env.step(action)
        outer.append(time.perf_counter()-before)
        if terminated or truncated:break
    if env.episode_started is not None:env.finish('rollout_limit')
    return {**env.episodes[-1], 'policySeconds':inference,'completeDecisionSeconds':sum(outer),
            'completeRolloutSeconds':time.perf_counter()-start}


def replay_trajectory(env,path):
    started=time.perf_counter();record=json.loads(Path(path).read_text());env.reset(seed=record['seed'])
    for decision in record['decisions']:
        _,_,terminated,truncated,info=env.step(OPTIONS.index(decision['option']))
        actual=env.decisions[-1]
        for key in ('option','requests','skills','firstAttempt','attempts','stateId','policyObservationSha256'):
            if actual[key]!=decision[key]:raise ValueError('Policy input/request/skill replay differs: '+key)
    if (env.native_snapshot()!=record['native'] or env.raw.attempts!=record['attempts']
            or env.raw.total_reward!=record['reward']):
        raise ValueError('Learned/evaluation trajectory replay differs')
    if record['reason'] not in ('training_budget_end','external_reset','rollout_limit') and info['termination_reason']!=record['reason']:
        raise ValueError('Task terminal reason replay differs')
    return dict(path=str(path),stateId=env.raw.engine.state_id,exactNativeAndRewardReplay=True,
        wallSeconds=time.perf_counter()-started,optionDecisions=len(record['decisions']),
        rlTransitionAttempts=len(env.raw.attempts),
        publicEngineCalls=env.raw.engine.public_calls-env.raw._prepared.public_calls,
        policyInputsAndRequestsExact=True,terminalReasonExact=True)


def export_workspace(manifest,result,out):
    source=Path(manifest['nativeWorkspace'])
    if sha(source)!=manifest['nativeWorkspaceSha256']:raise ValueError('Baseline workspace changed')
    case,artifacts,workspace=load_workspace(source)
    original=copy.deepcopy(artifacts['surgical_toolbox']['branches'])
    record=artifacts['surgical_toolbox'];rows=[];unique=[]
    for assessment in result['assessments']:
        trajectory=json.loads((out/assessment['path']).read_text())
        branch=next((name for name in unique if record['branches'][name]==trajectory['native']),None)
        if branch is None:
            branch=f'rescue-evaluation-{len(unique)+1}'
            if branch in record['branches']:raise ValueError('Learning branch already exists')
            record['branches'][branch]=trajectory['native'];unique.append(branch)
            record.setdefault('parents',{})[branch]=dict(branchId='catalog-rescue-task-start',
                position=len(record['branches']['catalog-rescue-task-start']['actions']))
        rows.append(dict(branchId=branch,**assessment))
    chosen=next((r for r in rows if r['method']=='trained' and r['success']),
        next(r for r in rows if r['method']=='trained'))
    record.update(branchId=chosen['branchId'],position=len(record['branches'][chosen['branchId']]['actions']))
    record.pop('timeline',None)
    artifacts['patient_rescue_learning']=dict(schema=result['schema'],assessments=rows,
        status=result['status'],
        taskManifestSha256=result['manifestSha256'],settings=result['settings'],
        limitations=result['limitations'],report=str(out/'result.json'))
    artifacts['patient_rescue_learning']['sharedTrajectories']=True
    destination=out/'patient-rescue-learning.ressectionlab'
    save_workspace(case,destination,sources=workspace['sources'],imaging=workspace['imagingState'],
        episode=workspace['episode'],selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts)
    reopened,saved,_=load_workspace(destination)
    if reopened.semantic_hash!=case.semantic_hash or saved['surgical_toolbox']['branches']!=record['branches']:
        raise ValueError('Saved learning workspace differs')
    if any(record['branches'][name]!=snapshot for name,snapshot in original.items()):
        raise ValueError('Existing branch changed')
    return dict(path=str(destination),sha256=sha(destination),sourceAndAllBranchesExact=True,
        selectedBranch=chosen['branchId'],uniqueAssessmentTrajectories=len(unique),
        assessmentRows=len(rows),assessments=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'testcases') or out.exists():parser.error('Use a new output directory under repository testcases/')
    out.mkdir(parents=True);began=time.perf_counter();torch.set_num_threads(1)
    result=dict(schema='resectionlab.patient-rescue-learning.v1',status='running',settings=SETTINGS,
        manifest=str(args.manifest.resolve()),manifestSha256=sha(args.manifest),optimizerUpdates=[],
        assessments=[],search=[],replay=[],limitations=[
            'One authored development injury on acquired anatomy; not held-out generalization or clinical validation.',
            'Privileged historical native setup is retained and measured; it is not raw-RL admitted history.',
            'Fixed waypoints/controller are common task inputs. Seeds change policy initialization/search sampling, not this deterministic physical world.',
            'Both PPO and search receive 64 real option transitions per seed; assessments share 12 decision/60 simulated-second/128 primitive limits.',
            'Search samples permutations of all six options without replacement, then verification; this structural prior differs from PPO exploration.',
            'Scripted/untrained baselines have zero optimization experience. Task-authoring diagnostics/cost are separate.'],
        provenance=dict(codeSha256={str(p.relative_to(ROOT)):sha(p) for p in
            (Path(__file__),ROOT/'scripts/patient_rescue_env.py')},
            packages={name:importlib.metadata.version(name) for name in ('stable-baselines3','gymnasium','torch','numpy')}))
    def check():
        if time.perf_counter()-began>SETTINGS['max_wall_seconds']:raise TimeoutError('Whole learning comparison wall budget exhausted')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)>SETTINGS['max_rss_bytes']:
            raise MemoryError('Whole learning comparison memory budget exhausted')
    env=None;model=None
    try:
        task,manifest=from_manifest(args.manifest,check=check)
        result['preparation']=task.preparation
        env=RecordingEnv(task,out,check)
        for seed in SETTINGS['seeds']:
            env.seed_label=seed
            model=AuditedPPO('MlpPolicy',env,n_steps=SETTINGS['n_steps'],batch_size=SETTINGS['batch_size'],
                n_epochs=SETTINGS['n_epochs'],learning_rate=SETTINGS['learning_rate'],gamma=SETTINGS['gamma'],
                ent_coef=SETTINGS['ent_coef'],policy_kwargs=dict(net_arch=dict(pi=[32],vf=[32])),
                seed=seed,device='cpu',verbose=0)
            model.update_receipts=[];initial_hash=parameter_hash(model)
            env.mode='untrained';result['assessments'].append(rollout(env,model=model,seed=seed))
            env.mode='scripted';result['assessments'].append(rollout(env,sequence=SCRIPT,seed=seed))
            if not result['assessments'][-1]['success']:raise ValueError('Successful scripted option baseline required before fitting')
            env.mode='ppo-training';before=env.total_decisions;model.learn(total_timesteps=SETTINGS['train_decisions'])
            if env.episode_started is not None and task.decisions:env.finish('training_budget_end')
            used=env.total_decisions-before
            if used!=SETTINGS['train_decisions']:raise ValueError('PPO interaction budget differs')
            checkpoint=out/f'ppo-seed-{seed}.zip';model.save(checkpoint)
            result['optimizerUpdates'].append(dict(seed=seed,initialSha256=initial_hash,finalSha256=parameter_hash(model),
                realOptionDecisions=used,receipts=model.update_receipts,checkpoint=checkpoint.name,checkpointSha256=sha(checkpoint)))
            env.mode='trained';result['assessments'].append(rollout(env,model=model,seed=seed))
            env.mode='search';rng=np.random.default_rng(seed);best=None;used=0;candidates=[]
            while used<SETTINGS['search_decisions']:
                sequence=[str(v) for v in rng.permutation(OPTIONS)]
                remaining=SETTINGS['search_decisions']-used;old_limit=task.max_decisions
                task.max_decisions=min(old_limit,remaining)
                try:row=rollout(env,sequence=sequence,seed=seed)
                finally:task.max_decisions=old_limit
                used+=row['decisions'];candidates.append(dict(sequence=sequence,**row))
                if best is None or row['reward']>best['reward']:best=dict(sequence=sequence,**row)
                check()
            result['search'].append(dict(seed=seed,realOptionDecisions=used,candidates=candidates,selected=best))
            env.mode='search-selected';result['assessments'].append(rollout(env,sequence=best['sequence'],seed=seed))
            for row in [r for r in result['assessments'] if r['seed']==seed]:
                result['replay'].append(replay_trajectory(task,out/row['path']))
            result['status']='running';write(out/'result.json',result)
        result['status']='completed'
        result['nativeWorkspace']=export_workspace(manifest,result,out)
    except BaseException as error:
        result.update(status='failed',error=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        if env is not None and env.episode_started is not None and task.raw.attempts:
            env.finish('comparison_interrupted')
        if model is not None and model.num_timesteps and not any(
                row['seed']==model.seed for row in result['optimizerUpdates']):
            checkpoint=out/f'ppo-partial-seed-{model.seed}.zip';model.save(checkpoint)
            result['incompleteOptimization']=dict(seed=model.seed,numTimesteps=model.num_timesteps,
                receipts=model.update_receipts,checkpoint=checkpoint.name,checkpointSha256=sha(checkpoint),
                finalSha256=parameter_hash(model),scope='Only completed updates; remaining rollout/fit budget unfinished')
        result['wallSeconds']=time.perf_counter()-began
        result['peakResidentBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        if env is not None:result.update(allEpisodes=env.episodes,totalOptionDecisions=env.total_decisions,
            totalRlTransitionAttempts=sum(row['rlTransitionAttempts'] for row in env.episodes),
            totalPublicEngineCalls=sum(row['publicEngineCalls'] for row in env.episodes),
            totalResetSeconds=env.reset_seconds)
        result['replayOptionDecisions']=sum(r['optionDecisions'] for r in result['replay'])
        result['replayRlTransitionAttempts']=sum(r['rlTransitionAttempts'] for r in result['replay'])
        result['replayPublicEngineCalls']=sum(r['publicEngineCalls'] for r in result['replay'])
        result['replayWallSeconds']=sum(r['wallSeconds'] for r in result['replay'])
        if env is not None:
            result['totalPublicActionsIncludingPreparationAndReplay']=(result['preparation']['publicActions']
                +result['totalPublicEngineCalls']+result['replayPublicEngineCalls'])
            source=Path(manifest['source'])
            result['sourcePreserved']=sha(source)==manifest['sourceSha256']
            if not result['sourcePreserved']:result['status']='failed'
        write(out/'result.json',result)
        print(json.dumps({k:result[k] for k in ('status','wallSeconds','error') if k in result}),flush=True)


if __name__=='__main__':main()
