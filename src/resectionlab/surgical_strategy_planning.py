"""Three bounded, executable rescue continuations from a recorded injury.

This compiler changes intervention ordering and explicit delay in an existing
patient trajectory. It is an open-loop engineering comparison, not a policy,
clinical recommendation or an inference of a new surgical corridor.
"""
from __future__ import annotations
import copy
import time
import numpy as np
from resectionlab.core import array_digest, semantic_digest
from resectionlab.surgical_rl_rewards import RewardTracker
from resectionlab.surgical_planning import _protection_measurements

TASK = 'vascular-rescue'


def rescue_recipes(engine, record):
    """Use only a retained same-source vascular sequence sharing this prefix."""
    n = len(engine.actions)
    if (not engine._protection_contract or not engine.scene.properties.get('tissuePhysicsV2')
            or engine.guard_report().get('mode') != 'strict-research'):
        raise ValueError('A verified V2 environment and source exclusions are required')
    if not any(r.get('target') and r['material']=='tumor' and np.any(engine.scene.labels==r['id']) for r in engine.scene.regions):
        raise ValueError('This comparison requires a nonempty supplied tumor target')
    if n == 0 or engine.state['slots']['left']['instrument'] != 'micro_scissors':
        raise ValueError('Select the recorded boundary immediately after vessel transection')
    if engine.metrics().get('activeVesselWounds', 0) <= 0:
        raise ValueError('The selected state has no active modeled wound')
    reference = None
    for branch_id, snapshot in record.get('branches', {}).items():
        if (record.get('fullCase', {}).get(branch_id, {}).get('scenario') == 'vascular'
                and snapshot.get('sceneId') == engine.scene.identity
                and snapshot['actions'][:n] == engine.actions
                and len(snapshot['actions']) == n + 25):
            reference = copy.deepcopy(snapshot['actions'][n:]); break
    if reference is None:
        raise ValueError('No retained source-matched rescue sequence starts at this boundary')
    expected = ['deactivate','inspect','inspect','withdraw','exchange','place','apply','apply',
                'deactivate','inspect','withdraw','exchange','place','apply','apply','deactivate',
                'withdraw','exchange','place','apply','apply','deactivate','withdraw','inspect','inspect']
    if ([a.get('operation') for a in reference] != expected
            or [reference[i].get('instrument') for i in (4,11,17)] != ['bipolar_forceps','irrigation','frazier_suction']
            or any(a.get('slot') != 'left' for a in reference[:-1])
            or reference[-1].get('slot') != 'right'):
        raise ValueError('The recorded intervention blocks do not match this bounded task')
    return [
        ('prompt-coagulation', 'Prompt coagulation', [reference[0], *reference[3:]]),
        ('suction-first', 'Suction before coagulation', [reference[0],reference[3],*reference[17:23],*reference[4:17],*reference[23:]]),
        ('recorded-delay', 'Recorded delayed response', reference),
    ]


def rescue_availability(engine, record):
    try:
        rescue_recipes(engine, record)
        return {'rescueAvailable': True, 'rescueReason': None}
    except ValueError as error:
        return {'rescueAvailable': False, 'rescueReason': str(error)}


def compare_rescue(engine, record, *, acknowledge_unknowns=False, check=lambda:None, progress=lambda *_:None):
    """Seal real choices and independently replay each continuation atomically.

    The common source state has already passed canonical admission. Each replay
    starts from a detached copy of that admitted state; no imported checkpoint
    grants authority. The full native action log is retained for later reopening.
    """
    if acknowledge_unknowns is not True:
        raise ValueError('Explicitly acknowledge missing anatomy and research assumptions')
    started=time.perf_counter(); outer_check=check
    def check():
        outer_check()
        if time.perf_counter()-started > 240:
            raise TimeoutError('Rescue comparison exceeded its 240-second budget; no plans installed')
    recipes=rescue_recipes(engine,record)
    initial_state=engine.state_id; source_hash=array_digest(engine.scene.labels); initial_metrics=engine.metrics()
    tracker=RewardTracker(engine,engine._protected_ids)
    initial=tracker.measure(engine,_protection_measurements(engine))
    total_target=sum(int(np.count_nonzero(engine.scene.labels==r['id']))*engine.scene.voxel_mm3
                     for r in engine.scene.regions if r.get('target') and r['material']=='tumor')
    unknown=['near_contact_exposure_mm3_seconds','deformation_injury_mm3','clinical_risk','complete_protected_structure_encounters']
    plans=[]
    for number,(method,label,actions) in enumerate(recipes):
        check();progress(.1+number*.28,'Executing and checking '+label)
        candidate=engine.clone(); rows=[]; before=time.perf_counter()
        for index,action in enumerate(actions):
            check()
            observation_hash=semantic_digest(candidate.observe())
            outcome=candidate.step(action,check=check)
            rows.append(dict(index=index,actionId=action.get('instrument',action['operation']),
                action=copy.deepcopy(action),scores=[],observationHash=observation_hash,
                status=outcome['status'],reason=outcome.get('reason'),stateId=candidate.state_id,reward=0.))
            if outcome['status']=='rejected':break
        planning_seconds=time.perf_counter()-before
        endpoint=RewardTracker(candidate,candidate._protected_ids).measure(candidate,_protection_measurements(candidate))
        replay=engine.clone(); before=time.perf_counter()
        for row in rows:
            check();result=replay.step(row['action'],check=check)
            if (result['status']!=row['status'] or result.get('reason')!=row['reason']
                    or replay.state_id!=row['stateId']):
                raise ValueError('Independent continuation replay differs from sealed history')
        measured=RewardTracker(replay,replay._protected_ids).measure(replay,_protection_measurements(replay))
        if replay.state!=candidate.state or measured!=endpoint:
            raise ValueError('Independent continuation endpoint differs')
        complete=len(rows)==len(actions) and rows[-1]['status']!='rejected'
        metrics=candidate.metrics()
        report=dict(method=method,label=label,checkpointSha256=None,startStateId=initial_state,
            finalStateId=candidate.state_id,prefixActions=len(engine.actions),decisions=rows,
            stopReason='sequence_complete' if complete else 'exact_execution_rejected',
            truncated=not complete,earlyStop=False,localReward=0.,
            cumulativeEndpoints={k:None if k in unknown else v for k,v in endpoint.items()},
            incrementalEndpoints={k:None if k in unknown else v-initial[k] for k,v in endpoint.items()},
            unassessedEndpoints=unknown,fullTargetFractionRemoved=endpoint['tumor_removed_mm3']/total_target,
            reachableTargetFractionRemoved=None,neurologicalHarm='unknown; no calibrated outcome model',
            invalidActions=sum(r['status']=='rejected' for r in rows),
            travelMm=candidate.state['travelMm']-engine.state['travelMm'],
            instrumentChanges=sum(r['action']['operation']=='exchange' and r['status']!='rejected' for r in rows),
            planSeal=semantic_digest(dict(sourceSceneId=engine.scene.identity,startStateId=initial_state,decisions=rows)),
            summary={k:metrics.get(k) for k in ('fluidPresentMm3','activeVesselWounds','vesselEncounters','fluidRemovedMm3','fluidEscapedMm3','thermalExposure')},
            cost=dict(planningSeconds=planning_seconds,freshReplaySeconds=time.perf_counter()-before,
                executionCalls=len(rows),freshReplayCalls=len(rows),previewCalls=0,optimizerUpdates=0),
            verification=dict(exactStateMatch=True,endpointVectorMatch=True,
                scope='Separate continuation replay from the same canonically verified common state; full native histories retained. Not independent physical validation.'))
        report['summary']['residualTargetMm3']=total_target-endpoint['tumor_removed_mm3']
        report['summary']['additionalWoundFluidMm3']=metrics['modeledVesselFluidGeneratedMm3']-initial_metrics['modeledVesselFluidGeneratedMm3']
        plans.append((candidate,report))
    if engine.state_id!=initial_state or array_digest(engine.scene.labels)!=source_hash:
        raise ValueError('Planning changed its live source state or anatomy')
    comparison=dict(schema='resectionlab.surgical-planning.v1',task=TASK,sourceSceneId=engine.scene.identity,
        startStateId=initial_state,sourceRole='Source-bound vascular rescue · development simulation',
        sourceLabelsHash=source_hash,guardContractId=engine.guard_report()['contractId'],
        totalWallSeconds=time.perf_counter()-started,optimizerUpdates=0,
        limitations=['Three scripted continuations of the same deliberately recorded vessel injury; not alternative tumor-access plans.',
            'Intervention order and explicit delay differ. Shared access and resection burden are preserved in cumulative endpoints.',
            'No reward model or learning is used; zero reward fields are placeholders for this scripted comparison.',
            'Wound fluid, damage and thermal exposure are uncalibrated model outputs, not patient blood loss or clinical risk.',
            'Missing vascular/functional anatomy and full protected-structure encounters remain unknown.',
            'Separate suffix replays reuse a canonically admitted common prefix; reopening still verifies native action histories.'],
        alternatives=[report for _,report in plans])
    return plans,comparison
