#!/usr/bin/env python3
"""Author one fixed catalog-tip rescue baseline, without training or new anatomy.

A verified historical native prefix is explicit privileged task preparation;
its actions are never reclassified as raw-RL admitted. All task primitives use
published nominal skill proposals and the unchanged catalog/duration checks.
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
sys.path.insert(0,str(ROOT/'src'))
from probe_patient_rescue_skills import caps, observation_adapter, sample, sha, summary, write
from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_skills import propose_skill
from resectionlab.workspace_bundle import load_workspace, save_workspace


def fixed_waypoints(reference):
    """Declared privileged task-author geometry, fixed before any execution."""
    contact=copy.deepcopy(reference[54]['pose']);outside=copy.deepcopy(reference[53]['pose'])
    def shifted(pose,offset):
        row=copy.deepcopy(pose);row['position']=[x+d for x,d in zip(row['position'],offset)];return row
    return dict(outside=outside,contact=contact,
        # Surface contacts reach the endpoint voxel without burying the tip;
        # exact occupancy queries found clear entry/retreat lines for these.
        coverage_lower=shifted(contact,[0.,-.6,0.]),coverage_lower_outside=shifted(outside,[0.,-.6,0.]),
        coverage_forward=shifted(contact,[-.6,0.,0.]),coverage_forward_outside=copy.deepcopy(outside),
        # Raise the tip .4mm above the known fluid-cell center for floor
        # clearance while the catalog suction footprint still covers the pool.
        suction_outside=shifted(outside,[0.,-2.,-.6]),
        suction_pool=shifted(reference[67]['pose'],[0.,-2.,-.6]))


def coagulation_route(points,name):
    target,outside={'center':('contact','outside'),'lower':('coverage_lower','coverage_lower_outside'),
        'forward':('coverage_forward','coverage_forward_outside')}[name]
    return [dict(id=name+'_approach',skill='approach',instrument='bipolar_forceps',pose=points[outside],
            parameters=dict(radiusMm=1.,forceN=0.,jawOpeningMm=0.,power=1.),duration=.2),
        dict(id=name+'_entry',skill='coagulate',pose=points[target],parameters=dict(power=1.),duration=1.),
        dict(id=name+'_seal',skill='coagulate',parameters=dict(power=1.),duration=2.),
        # Each stock withdrawal retreats50mm. Two reach true source exterior
        # before changing lines, with no energy and unchanged no-new-contact
        # guards. No active coagulation exit is part of the task.
        dict(id=name+'_withdraw',skill='withdraw',duration=.2),
        dict(id=name+'_withdraw_to_exterior',skill='withdraw',duration=.2)]


def fixed_requests(points):
    def request(name,skill,**values):return dict(id=name,skill=skill,**values)
    return [request for name in ('center','lower','forward') for request in coagulation_route(points,name)]+[
        request('approach_suction','approach',instrument='frazier_suction',pose=points['suction_outside'],
            parameters=dict(radiusMm=1.5,forceN=0.,depthMm=.1,flowMm3PerSecond=20.),duration=.2),
        request('enter_pool','aspirate',pose=points['suction_pool'],duration=1.),
        request('clear_pool','aspirate',duration=2.),
        request('clear_controls','stop',duration=.2),
        request('delayed_observation','inspect',duration=5.),
    ]


def cleanup(engine):
    value=summary(engine)
    value['controlsClear']=not engine.state['holds'] and all(not any((s['active'],s['inserted'],s['grip'])) for s in engine.state['slots'].values())
    value['goalReached']=value['activeVesselWounds']==0 and value['fluidPresentMm3']<=.1 and value['controlsClear']
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'testcases') or out.exists():parser.error('Use a new directory under repository testcases/')
    out.mkdir(parents=True);began=time.perf_counter();suffix_started=None
    result=dict(schema='resectionlab.patient-rescue-task-baseline.v1',passed=False,trainingStarted=False,optimizerUpdates=0,
        source=str(args.case.resolve()),sourceSha256=sha(args.case),scriptSha256=sha(__file__),
        totalMaximumSeconds=360,suffixMaximumSeconds=180,maximumRssBytes=3*1024**3,
        rawRlHistoricalAdmission=False,taskSetupPrivilege='Canonically verified historical48 actions, then native deactivate/withdraw/exchange; no historical radius/guard changes.',
        unknowns=['historical near-contact exposure','historical deformation peaks','complete vascular/functional anatomy','clinical injury'],
        scope='Fixed catalog-sized native skill baseline and exact suffix replay; no Gym learner, training, or independent physical validation.')
    def check():
        elapsed=time.perf_counter()-began
        if elapsed>360 or suffix_started is not None and time.perf_counter()-suffix_started>180:raise TimeoutError('Task preparation total/suffix budget exhausted')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)>3*1024**3:raise MemoryError('Task preparation RSS budget exhausted')
    try:
        case,artifacts,workspace=load_workspace(args.case);check();scene=load_environment(args.case);check()
        record=artifacts['surgical_toolbox'];injury=record['branches']['rescue-decision'];reference=record['branches']['full-vascular']['actions']
        if len(injury['actions'])!=48 or injury['actions']!=reference[:48]:raise ValueError('Wrong retained48-action injury prefix')
        points=fixed_waypoints(reference);requests=fixed_requests(points)
        write(out/'authored-waypoints-and-requests.json',dict(waypoints=points,requests=requests,
            provenance='Fixed before execution from retained same-source contact and read-only prior probe face/fluid diagnosis; privileged task authoring only.',
            priorDiagnosis=str(ROOT/'testcases/outputs/product-first-20261023/catalog-skill-readiness-v1/tip-coverage-diagnosis.json')))
        result.update(historicalExplicitRadiusViolations=caps(injury['actions']),authoredWaypoints=points,authoredRequests=requests)
        engine=SurgicalEngine.restore(scene,injury,use_cache=True,check=check)
        result.update(canonicalPrefixVerified=True,prefixPreparationSeconds=time.perf_counter()-began,
            prefixStateId=engine.state_id,prefixMetrics=summary(engine),sourceSceneId=scene.identity,sourceCaseSemanticHash=case.semantic_hash)
        suffix_started=time.perf_counter();injury_engine=engine.clone();setup=[]
        for action in [reference[48],reference[51],dict(slot='left',operation='exchange',duration=.2,instrument='bipolar_forceps')]:
            check();response=engine.step(action,check=check)
            setup.append(dict(action=copy.deepcopy(action),outcome=response,stateId=engine.state_id))
            if response['status']!='accepted':raise ValueError('Privileged native control setup refused')
        start_snapshot=engine.snapshot();start_id=engine.state_id
        write(out/'injury-snapshot.json',injury);write(out/'task-start-snapshot.json',start_snapshot)
        result.update(nativeSetup=setup,taskStartStateId=start_id,taskStartMetrics=summary(engine))
        isolated=[]
        for name in ('center','lower','forward'):
            candidate=engine.clone();probe_env,probe_tracker=observation_adapter(candidate);transitions=[]
            for request in coagulation_route(points,name):
                check();proposal=propose_skill(sample(probe_env,probe_tracker),request['skill'],
                    **{k:v for k,v in request.items() if k not in ('id','skill')})
                for action in proposal['actions']:
                    check();probe_env._validate_action_radius(action)
                    if action['duration']>probe_env.config.max_action_seconds:raise ValueError('Isolated route duration cap differs')
                    response=candidate.step(action,check=check)
                    transitions.append(dict(action=copy.deepcopy(action),outcome=response,stateId=candidate.state_id))
                    probe_env.attempts.append(dict(action=action))
                    if response['status']!='accepted':raise ValueError('Isolated '+name+' route refused: '+str(response.get('reason')))
            route_replay=engine.clone()
            for transition in transitions:
                check();actual=route_replay.step(transition['action'],check=check)
                if semantic_digest(actual)!=semantic_digest(transition['outcome']) or route_replay.state_id!=transition['stateId']:
                    raise ValueError('Isolated route exact replay differs')
            isolated.append(dict(name=name,requests=coagulation_route(points,name),transitions=transitions,
                exactReplay=True,endpoint=cleanup(candidate),privilegedFaceDoses={key:face['coagulationDose']
                    for key,face in candidate.state['vesselsV2']['faces'].items()}))
            write(out/'isolated-routes.json',isolated)
            print(json.dumps(dict(isolatedRoute=name,exactReplay=True,endpoint=cleanup(candidate))),flush=True)
        result['isolatedRoutes']=isolated
        env,tracker=observation_adapter(engine);rows=[];result['skills']=rows
        for request in requests:
            check();start=time.perf_counter();observation=sample(env,tracker);observed=time.perf_counter()
            proposal=propose_skill(observation,request['skill'],**{k:v for k,v in request.items() if k not in ('id','skill')});proposed=time.perf_counter()
            row=dict(request=copy.deepcopy(request),proposal=proposal,observationId=proposal['observationId'],transitions=[]);rows.append(row)
            for action in proposal['actions']:
                check();env._validate_action_radius(action)
                if action['duration']>env.config.max_action_seconds:raise ValueError('Proposed primitive exceeds unchanged RL duration cap')
                response=engine.step(action,check=check)
                row['transitions'].append(dict(action=copy.deepcopy(action),outcome=response,stateId=engine.state_id))
                env.attempts.append(dict(action=action));env.invalid_actions+=int(response['status']=='rejected')
                if response['status']!='accepted':raise ValueError('Fixed guarded skill refused: '+request['id']+': '+str(response.get('reason')))
            ended=time.perf_counter();row.update(metrics=cleanup(engine),timingSeconds=dict(observation=observed-start,proposal=proposed-observed,publicActions=ended-proposed,complete=ended-start))
            print(json.dumps(dict(skill=request['id'],metrics=row['metrics'])),flush=True)
            write(out/'progress.json',dict(skills=rows,wallSeconds=time.perf_counter()-began))
        before_wait=rows[-2]['metrics'];after_wait=rows[-1]['metrics']
        if not before_wait['goalReached'] or not after_wait['goalReached']:
            raise ValueError('Catalog baseline did not achieve zero active wounds, <=0.1mm3 retained fluid and clear controls')
        if after_wait['fluidPresentMm3']>before_wait['fluidPresentMm3']+1e-9 or abs(after_wait['modeledVesselFluidGeneratedMm3']-before_wait['modeledVesselFluidGeneratedMm3'])>1e-9:
            raise ValueError('Catalog baseline cleanup did not remain stable over five seconds')
        # Verify both native preparation controls and the entire proposed suffix
        # from the independently admitted48-action state, not an imported state.
        replay=injury_engine.clone();replay_start=time.perf_counter()
        for transition in setup+[t for row in rows for t in row['transitions']]:
            check();actual=replay.step(transition['action'],check=check)
            if semantic_digest(actual)!=semantic_digest(transition['outcome']) or replay.state_id!=transition['stateId']:
                raise ValueError('Exact complete native setup/skill suffix replay differs')
        if replay.state!=engine.state:raise ValueError('Exact independently replayed endpoint differs')
        result.update(exactFullPublicSuffixReplay=True,replaySeconds=time.perf_counter()-replay_start,
            endpoint=after_wait,beforeDelayedObservation=before_wait,finalStateId=engine.state_id,suffixSeconds=time.perf_counter()-suffix_started)
        record['branches']['catalog-rescue-task-start']=start_snapshot
        record['branches']['catalog-rescue-baseline']=engine.snapshot()
        record.setdefault('parents',{}).update({'catalog-rescue-task-start':dict(branchId='rescue-decision',position=48),
            'catalog-rescue-baseline':dict(branchId='catalog-rescue-task-start',position=len(start_snapshot['actions']))})
        record.update(branchId='catalog-rescue-baseline',position=len(engine.actions));record.pop('timeline',None)
        destination=out/'catalog-rescue-baseline.ressectionlab'
        save_workspace(case,destination,sources=workspace['sources'],imaging=workspace['imagingState'],episode=workspace['episode'],
            selection=workspace['selection'],originals=workspace['originals'],artifacts=artifacts);check()
        reopened,reopened_artifacts,_=load_workspace(destination);check()
        saved=reopened_artifacts['surgical_toolbox']
        if reopened.semantic_hash!=case.semantic_hash or any(saved['branches'][name]!=record['branches'][name] for name in ('catalog-rescue-task-start','catalog-rescue-baseline')):
            raise ValueError('Saved source or task native branches differ')
        manifest=dict(schema='resectionlab.patient-rescue-task-setup.v1',source=result['source'],sourceSha256=result['sourceSha256'],
            sourceSceneId=scene.identity,guardContractId=engine.guard_report()['contractId'],injuryBranch='rescue-decision',prefixActions=48,
            injurySnapshot=str(out/'injury-snapshot.json'),injurySnapshotSha256=sha(out/'injury-snapshot.json'),
            taskStartSnapshot=str(out/'task-start-snapshot.json'),taskStartSnapshotSha256=sha(out/'task-start-snapshot.json'),taskStartStateId=start_id,
            nativeSetup=setup,privilege=result['taskSetupPrivilege'],rawRlHistoricalAdmission=False,
            waypoints=points,baselineRequests=requests,catalogMaximumRadiiMm=dict(bipolar_forceps=1.,frazier_suction=1.5),
            coagulationRoutes={name:coagulation_route(points,name) for name in ('center','lower','forward')},
            isolatedRouteValidation=str(out/'isolated-routes.json'),
            success=dict(activeWounds=0,maximumRetainedFluidIncludingReservoirMm3=.1,delayedObservationSeconds=5.,controlsInactiveWithdrawnReleased=True),
            baselineNativeBranch='catalog-rescue-baseline',nativeWorkspace=str(destination),nativeWorkspaceSha256=sha(destination),
            baselineResult=str(out/'report.json'),unknowns=result['unknowns'],policyContract='Waypoints and option identities are fixed task declarations. Engine snapshots, face-dose diagnosis and full fluid grid are privileged and must not be policy inputs.')
        manifest['contentHash']=semantic_digest(manifest);write(out/'task-manifest.json',manifest)
        result.update(passed=True,savedNativeWorkspace=str(destination),savedSha256=sha(destination),serializedBranchesExact=True,
            taskManifest=str(out/'task-manifest.json'),taskManifestContentHash=manifest['contentHash'])
    except BaseException as error:
        result['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        result['sourcePreserved']=sha(args.case)==result['sourceSha256'];result['wallSeconds']=time.perf_counter()-began
        result['peakResidentBytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024)
        write(out/'report.json',result);print(json.dumps({k:result[k] for k in ('passed','sourcePreserved','wallSeconds','error') if k in result}),flush=True)

if __name__=='__main__':main()
