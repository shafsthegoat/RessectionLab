"""Fault-only metadata fixtures; these method labels are not performance evidence."""
import copy
import pytest
from resectionlab.core import semantic_digest
from resectionlab.surgical_recorded_sequential import source_geometry,validate_episode_record,validate_episode_boundary
from resectionlab.surgical_sequential_task import generated_task,endpoint_assessment,COMMON_CONFIG
from resectionlab.surgical_procedure_policy import evaluate_procedure_sequence


def reseal(record):
    p=record['sequentialEpisodes'];p['contentHash']=semantic_digest({k:v for k,v in p.items() if k!='contentHash'})


@pytest.fixture
def packet():
    task=generated_task(200);engine=task.initial_engine();initial=engine.state_id;requests=[];states=[initial]
    for action in [dict(slot='right',operation='exchange',instrument='cusa',parameters={'radiusMm':.4},duration=.1),dict(slot='right',operation='apply',duration=.1)]:
        before=copy.deepcopy(engine.state['slots']['right']);state=engine.state_id;parameters=copy.deepcopy(engine.parameters(before));position=len(engine.actions)
        outcome=engine.step(action);after=engine.state['slots']['right'];accepted=outcome['status'] in ('accepted','partial')
        requests.append(dict(attemptIndex=len(requests),nativePositionBefore=position,nativePositionAfter=len(engine.actions),action=action,
            stateBefore=state,stateAfter=engine.state_id,effectiveParametersBefore=parameters,effectiveParametersAfter=engine.parameters(after),
            instrumentBefore=before['instrument'],instrumentAfter=after['instrument'],toolPositionBefore=before['position'],toolPositionAfter=after['position'],
            status=outcome['status'],reason=outcome.get('reason'),reasonCode=outcome.get('reasonCode'),elapsedSeconds=outcome['elapsedSeconds'],completedFraction=outcome['completedFraction']))
        if accepted:states.append(engine.state_id)
    assert [r['status'] for r in requests]==['accepted','rejected']
    endpoint=endpoint_assessment(engine,COMMON_CONFIG,attempts=3,invalid_actions=1,stopped=True)
    dimensions=evaluate_procedure_sequence(engine)['outcome']['dimensions'];snapshot=engine.snapshot()
    rows=[];record=dict(branches={},branchId='b0',position=1)
    for i,(method,seed) in enumerate([('learned',17),('learned',43),('greedy',0),('search',0),('fixed-narrow-greedy',0)]):
        branch=f'b{i}';record['branches'][branch]=copy.deepcopy(snapshot)
        rows.append(dict(branchId=branch,method=method,seed=seed,snapshotHash=semantic_digest(snapshot),nativeActions=1,acceptedStateIds=states,
            finalStateId=engine.state_id,actionHistoryHash=semantic_digest(engine.history),attempts=copy.deepcopy(requests),checkedStop=True,
            endpoint=copy.deepcopy(endpoint),progress=copy.deepcopy(endpoint['progress']),modeledSeconds=engine.state['elapsedSeconds'],
            neuralRemovedMm3=dimensions['neuralRemovedMm3']['cumulative'],
            structureRemovedMm3={k:v['cumulative'] for k,v in dimensions['structureRemovedMm3'].items()},
            structureCutSourceMm3={k:v['cumulative'] for k,v in dimensions['structureCutSourceMm3'].items()},
            cleanup=dict(debrisMm3=endpoint['debrisMm3'],fluidMm3=endpoint['fluidMm3'],debrisCollectedMm3=engine.metrics()['debrisRemovedMm3'])))
    record['sequentialEpisodes']=dict(schema='resectionlab.recorded-sequential-episodes.v1',caseHash='fault-only-case',sourceSceneId=task.scene.identity,
        sourceStateId=initial,policyHash=snapshot['procedurePolicy']['policyHash'],configHash=COMMON_CONFIG.fingerprint,geometry=source_geometry(task.scene),episodes=rows)
    reseal(record);return task,record,engine


def test_cold_admission_checks_accepted_and_refused_requests(packet):
    task,record,engine=packet
    assert validate_episode_record(task.scene,record,'fault-only-case')
    assert validate_episode_boundary(engine,record)


@pytest.mark.parametrize('fault',['unselected-target','endpoint','cleanup','refused-effect','tool-setting','membership','snapshot','history','case'])
def test_fresh_seal_cannot_authenticate_changed_native_metadata(packet,fault):
    task,record,engine=packet;p=record['sequentialEpisodes'];r=p['episodes'][0]
    if fault=='unselected-target':p['episodes'][-1]['progress']['originalTargetTreatmentMm3']=27.
    elif fault=='endpoint':r['endpoint']['completedTask']=True
    elif fault=='cleanup':r['cleanup']['debrisMm3']=1.
    elif fault=='refused-effect':r['attempts'][1]['status']='accepted'
    elif fault=='tool-setting':r['attempts'][0]['effectiveParametersAfter']['radiusMm']=1.5
    elif fault=='membership':p['geometry']['structures'][0]['sourceCells']=[]
    elif fault=='snapshot':record['branches']['b0']['checkpoint']['state']['removed']=[1];r['snapshotHash']=semantic_digest(record['branches']['b0'])
    elif fault=='history':r['acceptedStateIds'][1]='sha256:'+'0'*64
    elif fault=='case':p['caseHash']='other-case'
    reseal(record)
    with pytest.raises(ValueError,match='RECORDED_SEQUENTIAL'):
        validate_episode_record(task.scene,record,'fault-only-case')


def test_selected_boundary_remeasures_terminal_classification(packet):
    task,record,engine=packet;record['sequentialEpisodes']['episodes'][0]['endpoint']['completedTask']=True
    with pytest.raises(ValueError,match='SELECTED_ENDPOINT_CHANGED'):
        validate_episode_boundary(engine,record)
