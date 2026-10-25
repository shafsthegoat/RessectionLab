"""Experimental, explicitly prepared patient rescue task over existing skills.

The historical injury is privileged task setup, not an admitted raw-RL episode.
Its complete actions are publicly replayed once with real exposure ledgers.
Only subsequent catalog-sized skills belong to this distinct task/schema.
Policy functions accept published current observations and fixed task inputs;
snapshots, source maps, face diagnostics and future outcomes stay outside them.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from resectionlab.core import semantic_digest
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_env import (SurgicalRLEnv, RLConfig, _ProtectedEngine,
    INSTRUMENTS, PARAMETERS)
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.surgical_skills import SkillExecutor, SkillTask
from resectionlab.surgical_tools import TOOLS

SCHEMA = 'resectionlab.prepared-patient-rescue.v1'
OPTIONS = ('coagulate_center', 'coagulate_lower', 'coagulate_forward',
           'aspirate_pool', 'wait', 'finish_verify')
_OBS_KEYS = frozenset(('local','slots','tool_parameters','metrics','goal',
    'coverage','local_spacing_mm','cleanup','budgets'))


class TaskProtectedEngine(_ProtectedEngine):
    """Count actual public calls separately from RL/preflight refusals."""
    def __init__(self, *args, **kwargs):
        self.public_calls=0
        super().__init__(*args, **kwargs)

    def step(self, action, *, check=None):
        self.public_calls+=1
        return super().step(action,check=check)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def option_requests(observation, option, waypoints):
    """Nominal hierarchical controls, based only on observed slot controls.

    Every coagulation option withdraws and enters its declared exterior line.
    The controller neither previews contact nor decides which point to treat.
    """
    if set(observation) != _OBS_KEYS or option not in OPTIONS:
        raise ValueError('Published observation and declared option required')
    slots = np.asarray(observation['slots'])
    if slots.shape != (2, 12):
        raise ValueError('Published slot shape differs')
    rows = []
    def add(skill, **values): rows.append(dict(skill=skill, **values))
    def clear(hand):
        state = slots[('left','right').index(hand)]
        if state[10] or state[11]: add('release', slot=hand, duration=.2)
        if state[8] or state[9]: add('withdraw', slot=hand, duration=.2)
    if option in ('wait', 'finish_verify'):
        clear('left'); clear('right')
        add('inspect', duration=5. if option == 'finish_verify' else 2.)
    else:
        clear('left')
        if option.startswith('coagulate_'):
            point = {'coagulate_center':'contact', 'coagulate_lower':'coverage_lower',
                     'coagulate_forward':'coverage_forward'}[option]
            outside = waypoints.get(point + '_outside', waypoints['outside'])
            add('approach', instrument='bipolar_forceps', pose=outside, duration=.2,
                parameters=dict(radiusMm=1., forceN=0., jawOpeningMm=0., power=1.))
            add('coagulate', pose=waypoints[point], duration=1., parameters=dict(power=1.))
            add('coagulate', duration=2., parameters=dict(power=1.))
        else:
            add('approach', instrument='frazier_suction', pose=waypoints['suction_outside'], duration=.2,
                parameters=dict(radiusMm=1.5, forceN=0., depthMm=.1, flowMm3PerSecond=20.))
            add('aspirate', pose=waypoints['suction_pool'], duration=1.)
            add('aspirate', duration=2.)
        # Stock withdrawal travels 50 mm. A second inactive withdrawal reaches
        # the declared exterior before the next option changes entry lines.
        add('withdraw', slot='left', duration=.2)
        add('withdraw', slot='left', duration=.2)
    return copy.deepcopy(rows)


def policy_features(observation, field, counts, verified_seconds, origin):
    """Only current published samples, modeled field quantities and own history.

    Field entries are retained fluid, active outlets, modeled current outlet rate
    and emitted volume since task entry. None is patient-calibrated physiology.
    No face locations, clot doses, route feasibility or candidate outcomes enter.
    """
    if set(observation) != _OBS_KEYS:
        raise ValueError('Unexpected policy input fields')
    slots = np.asarray(observation['slots'], float).copy()
    slots[:, :3] = (slots[:, :3] - np.asarray(origin)) / 100.
    slots[:, 7] /= len(INSTRUMENTS)
    local = np.asarray(observation['local']).mean(axis=(2,3,4)).ravel()
    params = np.asarray(observation['tool_parameters'])[:, [PARAMETERS.index(k) for k in
        ('radiusMm','flowMm3PerSecond','power')]] / np.array([2.,20.,2.])
    result = np.concatenate((slots.ravel(), local, params.ravel(),
        np.asarray(field) / np.array([10.,5.,1.,10.]), observation['budgets'],
        [verified_seconds / 5.], np.asarray(counts) / 12.)).astype(np.float32)
    if result.shape != (60,) or not np.isfinite(result).all():
        raise ValueError('Invalid policy features')
    return result


class PreparedRescueRLEnv(SurgicalRLEnv):
    """Raw suffix transitions with unchanged public geometry/guard/radius checks.

    Configuration bounds remain cumulative, offset explicitly by the measured
    entry burden. Historical near-contact/strain peaks are measured during real
    preparation, rather than invented zero placeholders or imported RL state.
    """
    def __init__(self, prepared, *, seconds=60., max_steps=128):
        if not isinstance(prepared, _ProtectedEngine):
            raise TypeError('Publicly replayed protected task preparation required')
        if prepared.state['holds'] or any(s['active'] or s['inserted'] or s['grip']
                for s in prepared.state['slots'].values()):
            raise ValueError('Task entry requires inactive, withdrawn, released controls')
        if any(prepared.parameters(s)['radiusMm']>TOOLS[s['instrument']]['radiusMm']
                for s in prepared.state['slots'].values()):
            raise ValueError('Task entry radius exceeds the instrument catalog')
        config = RLConfig(max_steps=max_steps, max_seconds=prepared.state['elapsedSeconds']+seconds,
            max_healthy_removed_mm3=1e9, max_vessel_damage_mm2=1e9, max_bleeding_mm3=1e9,
            max_success_fluid_mm3=.1, max_success_debris_mm3=float(prepared.state['debrisMm3']),
            require_critical_coverage=prepared.guard_config.require_critical_coverage,
            critical_max_displacement_mm=prepared.guard_config.critical_max_displacement_mm,
            local_spacing_mm=1.)
        super().__init__(prepared.scene, protection=prepared.protection, config=config)
        self._prepared = prepared.clone()
        self._seconds = float(seconds)
        self._entry_time = float(prepared.state['elapsedSeconds'])
        self.reset(seed=0)
        self.entry_metrics = self.measurements.copy()
        self.config = RLConfig(**{**asdict(config),
            'max_healthy_removed_mm3':self.entry_metrics['healthy_removed_mm3']+.1,
            'max_vessel_damage_mm2':self.entry_metrics['vessel_damage_mm2']+.1,
            'max_bleeding_mm3':self.entry_metrics['bleeding_generated_mm3']+10.})
        self.environment_id = semantic_digest(dict(schema=SCHEMA, scene=self.scene.identity,
            preparedStateId=prepared.state_id, config=asdict(self.config),
            reward='raw engineering injury costs plus outlet/fluid potential and stable-cleanup bonus v1'))

    def reset(self, *, seed=None, options=None):
        if options: raise ValueError('Task reset options are not supported')
        gym.Env.reset(self, seed=seed)
        if seed is not None: self.action_space.seed(seed)
        self.engine = self._prepared.clone()
        from resectionlab.surgical_rl_rewards import RewardTracker
        self.tracker = RewardTracker(self.engine, self.protection.cells)
        self.attempts=[]; self.invalid_actions=0; self.total_reward=0.; self._done=False
        self.terminated=False; self.truncated=False; self.termination_reason=None
        self._verified_seconds=0.; self._verification=None
        self.measurements=self._measure()
        self._entry_emission=self.engine.metrics()['modeledVesselFluidGeneratedMm3']
        return self._observe(), self._info()

    def field(self):
        m=self.engine.metrics()
        rate=sum(float(s.get('rateMm3PerSecond',0.)) for s in self.engine.state['sources']
                 if s.get('modelVersion')==2 and s.get('bleedingActive',False))
        return np.array([m['fluidPresentMm3'],m['activeVesselWounds'],rate,
            m['modeledVesselFluidGeneratedMm3']-self._entry_emission],float)

    def _observe(self):
        observation=super()._observe()
        observation['budgets'][1]=max(0.,1.-(self.engine.state['elapsedSeconds']-self._entry_time)/self._seconds)
        return observation

    def _target_reached(self):
        seconds=self._verified_seconds
        if self._verification is not None and not self._cleanup_state()['pending']:
            start, emission, fluid=self._verification
            m=self.engine.metrics()
            if m['modeledVesselFluidGeneratedMm3']<=emission+1e-9 and m['fluidPresentMm3']<=fluid+1e-9:
                seconds+=self.engine.state['elapsedSeconds']-start
        return seconds>=5.-1e-9

    def _transition(self, action, error=None, *, check=None):
        before=self.field(); clean=not self._cleanup_state()['pending']
        start=float(self.engine.state['elapsedSeconds'])
        m=self.engine.metrics()
        self._verification=(start,m['modeledVesselFluidGeneratedMm3'],m['fluidPresentMm3']) if (
            clean and isinstance(action,dict) and action.get('operation')=='inspect') else None
        observation,reward,terminated,truncated,info=super()._transition(action,error,check=check)
        after=self.field()
        verified=self._target_reached()
        if self._verification is not None and not self._cleanup_state()['pending']:
            if after[3]<=before[3]+1e-9 and after[0]<=before[0]+1e-9:
                self._verified_seconds+=self.engine.state['elapsedSeconds']-start
        elif self._cleanup_state()['pending']:
            self._verified_seconds=0.
        self._verification=None
        # A potential difference cannot reward repeated treatment/aspiration at
        # an unchanged field. Actual newly generated wound volume stays a cost.
        shaping=2.*(before[2]-after[2])+.5*(before[0]-after[0])-.1*max(0.,after[3]-before[3])
        if terminated and verified: shaping+=2.
        self.total_reward+=shaping
        self.attempts[-1]['reward']+=shaping
        self.attempts[-1]['terms']['rescue_field_potential']=float(shaping)
        if self.termination_reason=='target_fraction_reached':
            self.termination_reason='rescue_cleanup_verified'
        return self._observe(),float(reward+shaping),terminated,truncated,self._info(info['outcome'],self.attempts[-1]['terms'])

    def _info(self, outcome=None, terms=None):
        info=super()._info(outcome,terms)
        info.pop('target_fraction_reached',None)
        info.update(schema=SCHEMA, success=self.termination_reason=='rescue_cleanup_verified',
            verifiedCleanSeconds=self._verified_seconds, task='prepared wound rescue',
            historicalPrefixRawRlAdmitted=False, entryMetrics=getattr(self,'entry_metrics',{}).copy(),
            modeledField=self.field().tolist(),
            policy_input_contract='Current observation only; full info, preparation, engine and snapshots are privileged.')
        return info

    def snapshot(self):
        raise ValueError('Prepared rescue uses its task manifest and complete native/attempt trajectory; raw-RL checkpoint export is unsupported')

    def restore(self, record):
        raise ValueError('Prepared rescue requires public task preparation and trajectory replay; raw-RL checkpoint import is unsupported')


class RescueOptionEnv(gym.Env):
    """Six sequential choices over real parameterized multi-primitive skills."""
    def __init__(self, raw, waypoints, *, max_decisions=12, check=lambda:None):
        self.raw=raw; self.waypoints=copy.deepcopy(waypoints); self.max_decisions=max_decisions
        self.check=check; self.action_space=spaces.Discrete(len(OPTIONS))
        self.observation_space=spaces.Box(-1e6,1e6,(60,),np.float32)
        self.task=SkillTask('field_management','preinjured_rescue')
        self.reset(seed=0)

    def _observation(self):
        return policy_features(self.raw._observe(),self.raw.field(),self.counts,
            self.raw._verified_seconds,self.waypoints['contact']['position'])

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self,seed=seed)
        if seed is not None:self.action_space.seed(seed)
        self.raw.reset(seed=seed,options=options);self.counts=np.zeros(len(OPTIONS));self.decisions=[]
        self.inflightOption=None
        self._done=False
        observation=self._observation();self._last_policy_observation=observation.copy()
        return observation,self.raw._info()

    def step(self, action):
        self.check()
        if self._done:raise RuntimeError('Episode finished; reset required')
        if not self.action_space.contains(action):raise ValueError('Unknown task option')
        started=time.perf_counter();before=self.raw.total_reward
        observation=self.raw._observe();observed=time.perf_counter()
        requests=option_requests(observation,OPTIONS[int(action)],self.waypoints);proposed=time.perf_counter()
        rows=[];first=len(self.raw.attempts)
        self.inflightOption=dict(option=OPTIONS[int(action)],requests=requests,skills=rows,
            firstAttempt=first,policyObservationSha256=hashlib.sha256(self._last_policy_observation.tobytes()).hexdigest())
        for request in requests:
            self.check()
            if self.raw._done:break
            self.inflightOption['currentSkill']=request
            try:
                receipt=SkillExecutor(self.raw,self.task).execute(request['skill'],
                    **{k:v for k,v in request.items() if k!='skill'},check=self.check,max_wall_seconds=30.)
            except ValueError as error:
                self.raw._transition(None,'NOMINAL_SKILL_REFUSAL: '+str(error),check=self.check)
                rows.append(dict(status='rejected_nominal_proposal',reason=str(error)));break
            rows.append(receipt)
            if receipt['status'] not in ('completed','episode_finished'):break
        executed=time.perf_counter();self.counts[int(action)]+=1
        self.decisions.append(dict(option=OPTIONS[int(action)],requests=requests,skills=rows,
            firstAttempt=first,attempts=len(self.raw.attempts)-first,stateId=self.raw.engine.state_id,
            policyObservationSha256=hashlib.sha256(self._last_policy_observation.tobytes()).hexdigest()))
        self.inflightOption=None
        terminated=self.raw.terminated
        refused=any(row.get('status') not in ('completed','episode_finished') for row in rows)
        truncated=not terminated and (self.raw.truncated or refused or len(self.decisions)>=self.max_decisions)
        self._done=terminated or truncated
        info=self.raw._info();info['option']=self.decisions[-1]
        if truncated and not self.raw.truncated:
            info['termination_reason']='option_refused' if refused else 'option_decision_limit'
        observation=self._observation();self._last_policy_observation=observation.copy()
        ended=time.perf_counter()
        self.decisions[-1]['timing']=dict(observation=observed-started,proposal=proposed-observed,
            execution=executed-proposed,postObservationAndReceipt=ended-executed,complete=ended-started)
        return observation,self.raw.total_reward-before,terminated,truncated,info

    def native_snapshot(self):return self.raw.engine.snapshot()


def from_manifest(path, *, check=lambda:None):
    """Publicly reconstruct the complete privileged setup, then define the task."""
    began=time.perf_counter();path=Path(path);manifest=json.loads(path.read_text())
    unsigned={k:v for k,v in manifest.items() if k!='contentHash'}
    if manifest.get('contentHash')!=semantic_digest(unsigned):raise ValueError('Task manifest seal differs')
    if manifest.get('schema')!='resectionlab.patient-rescue-task-setup.v1':raise ValueError('Unknown task setup schema')
    source=Path(manifest['source']);snapshot_path=Path(manifest['taskStartSnapshot'])
    if sha(source)!=manifest['sourceSha256'] or sha(snapshot_path)!=manifest['taskStartSnapshotSha256']:
        raise ValueError('Task source/preparation changed')
    baseline=json.loads(Path(manifest['baselineResult']).read_text())
    if not baseline.get('passed') or not baseline.get('exactFullPublicSuffixReplay') or not baseline.get('sourcePreserved'):
        raise ValueError('Successful source-preserving scripted baseline required')
    if (baseline.get('taskManifestContentHash')!=manifest['contentHash']
            or baseline.get('sourceSha256')!=manifest['sourceSha256']
            or baseline.get('taskStartStateId')!=manifest['taskStartStateId']
            or baseline.get('authoredWaypoints')!=manifest['waypoints']
            or baseline.get('authoredRequests')!=manifest['baselineRequests']):
        raise ValueError('Baseline proof differs from the declared task')
    snapshot=json.loads(snapshot_path.read_text());scene=load_environment(source)
    if scene.identity!=manifest['sourceSceneId']:raise ValueError('Task source identity differs')
    protection,guard=restore_protection_contract(scene,snapshot['protectionContract'])
    config=RLConfig(max_seconds=1e6,require_critical_coverage=guard.require_critical_coverage,
        critical_max_displacement_mm=guard.critical_max_displacement_mm,local_spacing_mm=1.)
    engine=TaskProtectedEngine(scene,protection,config)
    for action in snapshot['actions']:
        check();result=engine.step(action,check=check)
        if result['status']=='rejected':raise ValueError('Privileged task preparation no longer replays')
    if (engine.state_id!=snapshot['stateId'] or engine.state_id!=manifest['taskStartStateId']
            or engine.state!=snapshot['checkpoint']['state']
            or engine.guard_report()['contractId']!=manifest['guardContractId']):
        raise ValueError('Privileged preparation physical state differs')
    if sha(source)!=manifest['sourceSha256']:raise ValueError('Task source changed while preparing')
    raw=PreparedRescueRLEnv(engine)
    env=RescueOptionEnv(raw,manifest['waypoints'],check=check)
    env.preparation=dict(seconds=time.perf_counter()-began,publicActions=len(engine.actions),
        sourceSha256=manifest['sourceSha256'],stateId=engine.state_id,
        realHistoricalExposureLedgers=True,rawRlHistoricalAdmission=False)
    return env,manifest
