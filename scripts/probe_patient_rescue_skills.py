#!/usr/bin/env python3
"""Bounded catalog-tool contact/readiness probe on a canonically verified injury.

No fitting, raw-RL archive admission, historical repair or imported checkpoint
promotion. Published observations support nominal proposals only. Native public
suffix primitives pass the unchanged RL radius/duration checks; this is not the
RL environment's reward/termination/snapshot-restore contract.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_planning import _protection_measurements
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.surgical_rl_rewards import RewardTracker
from resectionlab.surgical_skills import SkillExecutor, SkillTask, propose_skill
from resectionlab.surgical_tools import TOOLS
from resectionlab.workspace_bundle import load_workspace, save_workspace


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def caps(actions):
    """Inventory explicit historical radii; not transition feasibility admission."""
    tools={'left':'cusa','right':'cusa'};rows=[]
    for i,action in enumerate(actions):
        slot=action['slot']
        if action['operation']=='exchange':tools[slot]=action['instrument']
        value=action.get('parameters',{}).get('radiusMm')
        if value is not None and value>TOOLS[tools[slot]]['radiusMm']:
            rows.append(dict(index=i,instrument=tools[slot],radiusMm=value,catalogMaximumMm=TOOLS[tools[slot]]['radiusMm']))
    return rows


def observation_adapter(engine):
    """Read-only published sampler; never call raw-RL reset/step/restore here.

    Its two unavailable historical exposure metrics cannot support a learner.
    Proposal functions consume slot/parameter/local-support fields only.
    """
    tracker=RewardTracker(engine,engine._protected_ids)
    measured=tracker.measure(engine,_protection_measurements(engine))
    cfg=RLConfig(max_seconds=engine.state['elapsedSeconds']+60,max_steps=64,
        max_healthy_removed_mm3=measured['healthy_removed_mm3']+5,
        max_vessel_damage_mm2=measured['vessel_damage_mm2']+1,
        max_bleeding_mm3=measured['bleeding_generated_mm3']+20,
        require_critical_coverage=engine.guard_config.require_critical_coverage,
        critical_max_displacement_mm=engine.guard_config.critical_max_displacement_mm,
        local_spacing_mm=1.)
    env=SurgicalRLEnv(engine.scene,protection=engine.protection,config=cfg)
    env.engine=engine;env.measurements=measured;env.attempts=[];env.invalid_actions=0
    return env,tracker


def sample(env,tracker):
    env.measurements=tracker.measure(env.engine,_protection_measurements(env.engine))
    return env._observe()


def summary(engine):
    m=engine.metrics()
    return {key:m[key] for key in ('elapsedSeconds','fluidPresentMm3','modeledVesselFluidGeneratedMm3',
        'modeledVesselReservoirMm3','activeVesselWounds','fluidRemovedMm3','thermalExposure','targetRemovedMm3')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'testcases') or out.exists():p.error('Use a new directory under repository testcases/')
    out.mkdir(parents=True);began=time.perf_counter();suffix_started=None
    report=dict(schema='resectionlab.patient-rescue-skill-readiness.v1',passed=False,
        trainingStarted=False,optimizerUpdates=0,rawRlAdmission=False,source=str(args.case.resolve()),
        sourceSha256=sha(args.case),totalMaximumSeconds=360,suffixMaximumSeconds=180,
        limitations=['Canonical 48-action injury preparation is privileged native setup, not raw-RL admitted actions.',
            'Native suffixes validate catalog radius/duration, but do not implement SurgicalRLEnv reward/termination/restore semantics.',
            'Observation near-contact and historical deformation-peak metrics are unavailable; their sampler slots are placeholders and no learner consumes them.',
            'Fixed source-matched exterior/contact endpoints are declared task setup; no policy inferred a surgical route.',
            'One acquired development anatomy; no clinical accuracy or patient generalization claim.'])
    def check():
        elapsed=time.perf_counter()-began
        if elapsed>360 or suffix_started is not None and time.perf_counter()-suffix_started>180:
            raise TimeoutError('Readiness probe total/suffix budget exhausted')
    try:
        case,artifacts,workspace=load_workspace(args.case);check()
        scene=load_environment(args.case);check()
        record=artifacts['surgical_toolbox'];snapshot=record['branches']['rescue-decision']
        if len(snapshot['actions'])!=48:raise ValueError('Expected explicit retained 48-action injury boundary')
        reference=record['branches']['full-vascular']['actions']
        if snapshot['actions']!=reference[:48]:raise ValueError('Injury prefix differs from retained source actions')
        report['historicalPrefixExplicitRadiusViolations']=caps(snapshot['actions'])
        engine=SurgicalEngine.restore(scene,snapshot,use_cache=True,check=check)
        report.update(canonicalPrefixVerified=True,prefixPreparationSeconds=time.perf_counter()-began,
            prefixStateId=engine.state_id,prefixMetrics=summary(engine),guardContractId=engine.guard_report()['contractId'])
        initial=RewardTracker(engine,engine._protected_ids).measure(engine,_protection_measurements(engine))
        defaults=RLConfig()
        report['defaultRawRlInheritedBudgetFailures']={key:dict(inherited=initial[key],maximum=getattr(defaults,limit))
            for key,limit in [('elapsed_seconds','max_seconds'),('healthy_removed_mm3','max_healthy_removed_mm3'),
                ('vessel_damage_mm2','max_vessel_damage_mm2'),('bleeding_generated_mm3','max_bleeding_mm3')]
            if initial[key]>getattr(defaults,limit)}
        env,tracker=observation_adapter(engine)
        try:
            receipt=SkillExecutor(env,SkillTask('field_management','preinjured_rescue')).execute('withdraw',check=check)
        except ValueError as error:report['existingSkillExecutorEntryBlocker']=str(error)
        else:raise ValueError('Expected inherited oversized scissors to refuse before a public RL transition: '+str(receipt))
        suffix_started=time.perf_counter();setup=[]
        # Explicit native controls preserve the historical geometry. An inserted
        # tool is never silently resized, and no global catalog rule is changed.
        for action in [reference[48],reference[51],{'slot':'left','operation':'exchange','duration':.2,'instrument':'bipolar_forceps'}]:
            check();result=engine.step(action,check=check)
            setup.append(dict(action=copy.deepcopy(action),outcome=result,stateId=engine.state_id))
            if result['status']!='accepted':raise ValueError('Native task setup did not complete')
        report['privilegedNativeTaskSetup']=setup;report['postSetupStateId']=engine.state_id
        exterior=copy.deepcopy(reference[53]['pose']);coagulation=copy.deepcopy(reference[54]['pose']);suction=copy.deepcopy(reference[67]['pose'])
        request_sequence=[('approach',dict(instrument='bipolar_forceps',pose=exterior,parameters={'radiusMm':1.,'forceN':0.,'jawOpeningMm':0.,'power':1.},duration=.2)),
            ('coagulate',dict(pose=coagulation,parameters={'power':1.},duration=1.)),
            ('coagulate',dict(parameters={'power':1.},duration=3.)),('withdraw',dict(duration=.2)),
            ('approach',dict(instrument='frazier_suction',pose=exterior,parameters={'radiusMm':1.5,'forceN':0.,'depthMm':.1,'flowMm3PerSecond':2.},duration=.2)),
            ('aspirate',dict(pose=suction,duration=1.)),('aspirate',dict(duration=2.)),
            ('inspect',dict(duration=5.)),('stop',dict(duration=.2))]
        variants=[]
        for name,wait in [('prompt',False),('delayed',True)]:
            check();candidate=engine.clone();env,tracker=observation_adapter(candidate);rows=[];phase=[];truncated=False
            requests=([('inspect',dict(duration=5.)),('inspect',dict(duration=2.))] if wait else [])+request_sequence
            for skill,request in requests:
                check();start=time.perf_counter();observation=sample(env,tracker);observed=time.perf_counter()
                proposal=propose_skill(observation,skill,**request);proposed=time.perf_counter();transitions=[]
                for action in proposal['actions']:
                    check();env._validate_action_radius(action)
                    if action['duration']>env.config.max_action_seconds:raise ValueError('Skill primitive exceeds unchanged RL duration cap')
                    result=candidate.step(action,check=check)
                    transitions.append(dict(action=copy.deepcopy(action),outcome=result,stateId=candidate.state_id))
                    env.attempts.append(dict(action=action));env.invalid_actions+=int(result['status']=='rejected')
                    if result['status']!='accepted':truncated=True;break
                ended=time.perf_counter();metrics=summary(candidate)
                rows.append(dict(skill=skill,request=request,proposal=proposal,transitions=transitions,
                    observationId=proposal['observationId'],timingSeconds=dict(observation=observed-start,proposal=proposed-observed,
                        publicActions=ended-proposed,complete=ended-start),metrics=metrics))
                phase.append(dict(skill=skill,**metrics));print(json.dumps(dict(variant=name,skill=skill,status='partial' if truncated else 'accepted',metrics=metrics)),flush=True)
                if truncated:break
            replay=engine.clone();replay_start=time.perf_counter()
            for row in rows:
                for transition in row['transitions']:
                    check();actual=replay.step(transition['action'],check=check)
                    if semantic_digest(actual)!=semantic_digest(transition['outcome']) or replay.state_id!=transition['stateId']:
                        raise ValueError('Independent complete public suffix result/state differs')
            if replay.state!=candidate.state:raise ValueError('Independent suffix state differs')
            variants.append(dict(name=name,truncated=truncated,skills=rows,phaseMetrics=phase,endpoint=summary(candidate),
                replaySeconds=time.perf_counter()-replay_start,exactPublicSuffixReplay=True,stateId=candidate.state_id,
                nativeBranchId='skill-readiness-'+name))
            record['branches']['skill-readiness-'+name]=candidate.snapshot()
            record.setdefault('parents',{})['skill-readiness-'+name]={'branchId':'rescue-decision','position':48}
        report['variants']=variants;report['suffixSeconds']=time.perf_counter()-suffix_started
        report['learningReadiness']='blocked: historical raw-RL admission and unknown historical exposure metrics; this probe only establishes native catalog-tool proposal/contact throughput'
        record['branchId']='skill-readiness-prompt';record['position']=len(record['branches'][record['branchId']]['actions']);record.pop('timeline',None)
        destination=out/'catalog-skill-probes.ressectionlab'
        save_workspace(case,destination,sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],
            selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts);check()
        reopened,reopened_artifacts,_=load_workspace(destination);check()
        saved=reopened_artifacts['surgical_toolbox']
        if reopened.semantic_hash!=case.semantic_hash or any(saved['branches'][v['nativeBranchId']]!=record['branches'][v['nativeBranchId']] for v in variants):
            raise ValueError('Saved native branch serialization differs')
        report.update(passed=True,savedNativeWorkspace=str(destination),savedSha256=sha(destination),
            serializedBranchesExact=True,saveReopenMeaning='Byte/source checks and exact native snapshot/action serialization; separate verified suffix replay above, no cold whole-prefix replay.')
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        report['sourcePreserved']=sha(args.case)==report['sourceSha256'];report['wallSeconds']=time.perf_counter()-began
        report['peakResidentBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        write(out/'report.json',report)
        print(json.dumps({k:report[k] for k in ('passed','sourcePreserved','wallSeconds','learningReadiness') if k in report}),flush=True)

if __name__=='__main__':main()
