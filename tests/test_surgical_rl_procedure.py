"""Evaluation parity and atomic budgets for declared generated anatomy only."""
import copy
import json

import numpy as np
import pytest

pytest.importorskip('gymnasium')

from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig, PROCEDURE_SCHEMA
from resectionlab.surgical_rl_rewards import RewardConfig
from resectionlab.surgical_procedure_policy import (
    PROPERTY, make_procedure_policy, evaluate_procedure_sequence)


def scene(neural_limit=0., *, critical_shortcut=False):
    labels = np.zeros((16, 16, 40), np.uint16)
    labels[8, 8, 20] = 1
    labels[11, 8, 20] = 1
    labels[8, 8, 18] = 2
    labels[2, 2, 20] = 2
    labels[2, 3, 20] = 2
    source = {'kind': 'generated-control',
        'provenance': 'Generated policy/adapter regression geometry; no acquired anatomy'}
    regions = [dict(id=1, name='Declared target', material='tumor', target=True),
        dict(id=2, name='Neural membership owner', material='soft_tissue')]
    affine = np.diag([5., 5., 5., 1.])
    initial = SurgicalScene(labels, affine, regions, source, {})
    flat = lambda ijk: int(np.ravel_multi_index(ijk, labels.shape))
    brainstem = flat((2, 2, 20))
    cerebellum = flat((8, 8, 18)) if critical_shortcut else flat((2, 3, 20))
    cortex = flat((2, 3, 20)) if critical_shortcut else flat((8, 8, 18))
    intracranial = sorted(map(int, np.flatnonzero(labels)))
    permitted = sorted(set(intracranial)-{brainstem, cerebellum})
    evidence = dict(provenance='Independent generated source declaration for engine regression',
        evidenceSha256='a'*64, origin='independent-source-declaration', reviewStatus='generated-engineering')
    structures = {name: dict(evidence, kind='neural', cells=[cell], coverageCells=intracranial)
        for name, cell in [('brainstem', brainstem), ('cerebellum', cerebellum), ('cortex', cortex)]}
    raw = make_procedure_policy(initial, policy_id='generated-adapter-v1', evidence=evidence,
        structures=structures, intracranial_cells=intracranial,
        observed_cells=list(range(labels.size)),
        entry_regions={'declared-entry':dict(evidence, cells=permitted)},
        permitted_compartments={'supratentorial-work':dict(evidence, cells=permitted)},
        work_domain=dict(evidence, minimumVoxelBoundary=[-.5]*3,
            maximumVoxelBoundary=(np.asarray(labels.shape)-.5).tolist()),
        tissue_permissions={'tumor':dict(remove=True, cut=True), 'soft_tissue':dict(remove=True, cut=True)},
        neural_max_mm3=neural_limit,
        bone_max_mm3=0.,preparation_nonbone_max_mm3=0.,modeled_time_max_seconds=240.,
        structure_max_mm3={'brainstem':0., 'cerebellum':0., 'cortex':neural_limit})
    return SurgicalScene(labels, affine, regions, source, {PROPERTY:raw})


def env(source=None, **config):
    return SurgicalRLEnv(source or scene(), config=RLConfig(max_healthy_removed_mm3=1000., **config),
        reward_config=RewardConfig(tumor_removal_weight=1e12))


def command(operation, position=None, **kwargs):
    action = dict(slot='left', operation=operation, duration=.1)
    action.update(kwargs)
    if position is not None:
        action['pose'] = dict(position=position, quaternion=[0., 0., 0., 1.])
    return action


def prepare(e, x=40.):
    e.reset(seed=7)
    for action in [command('exchange', instrument='cusa'), command('place', [x, 40., 65.])]:
        info=e.step_action(action)[4]
        assert info['outcome']['status']=='accepted', info['outcome']


@pytest.mark.parametrize('critical_shortcut', [False, True])
def test_harm_or_cerebellar_shortcut_refused_atomically_despite_tumor_reward(critical_shortcut):
    e=env(scene(critical_shortcut=critical_shortcut));prepare(e)
    before=copy.deepcopy(e.engine.state);before_id=e.engine.state_id
    ledger=e.measurements.copy();reward_before=e.total_reward
    info=e.step_action(command('apply', [40.,40.,100.], duration=1.))[4]
    assert info['outcome']['status']=='rejected',info['outcome']
    assert info['constraint_failure'] and info['constraint_violations']
    assert e.engine.state==before and e.engine.state_id==before_id
    assert e.measurements==ledger and e.total_reward<=reward_before
    assert info['reward_terms']['tumor_removal']==0 and not info['success']
    assert info['computation']['transitionEvaluations']==3
    assert info['computation']['discardedEvaluations']==1
    assert info['procedure']==evaluate_procedure_sequence(e.engine)


def test_useful_resection_then_checked_exit_and_stop_preserves_residual_and_replays():
    e=env();prepare(e,x=55.)
    result=e.step_action(command('apply',[55.,40.,100.],duration=1.))
    assert result[4]['outcome']['status']=='accepted',result[4]['outcome']
    assert 0<e.measurements['tumor_removed_mm3']<e.initial_target_mm3
    rejected=e.stop('Budget-constrained residual')
    assert rejected[4]['outcome']['reasonCode']=='PROCEDURE_STOP_REQUIRES_CHECKED_EXIT'
    assert not rejected[2]
    result=e.step_action(command('withdraw',[55.,40.,65.],duration=1.))
    assert result[4]['outcome']['status']=='accepted',result[4]['outcome']
    result=e.stop('No continuation within declared neural budget')
    assert result[2:4]==(True,False)
    assert result[4]['termination_reason']=='stopped_with_residual' and not result[4]['success']
    outcome=result[4]['procedure']['outcome']
    assert outcome['withdrawalComplete'] and outcome['target']['residualMm3']>0
    assert outcome['dimensions']['neuralRemovedMm3']['cumulative']==0
    snapshot=json.loads(json.dumps(e.snapshot(),allow_nan=False))
    assert snapshot['schema']==PROCEDURE_SCHEMA
    restored=env();restored.restore(snapshot)
    assert restored.engine.state_id==e.engine.state_id
    assert restored.procedure_evaluation==e.procedure_evaluation
    assert restored.computation==e.computation


def test_missing_required_coverage_and_historical_snapshots_refused():
    source=scene();raw=copy.deepcopy(source.properties[PROPERTY]);del raw['structures']['cerebellum']
    from resectionlab.core import semantic_digest
    raw['policyHash']=semantic_digest({k:v for k,v in raw.items() if k!='policyHash'})
    missing=SurgicalScene(source.labels,source.affine,source.regions,source.source,{PROPERTY:raw})
    with pytest.raises(ValueError,match='CEREBELLUM'):
        env(missing)
    e=env();e.reset();record=e.snapshot();record['schema']='resectionlab.surgical-rl.v2'
    with pytest.raises(ValueError,match='historical policies'):
        e.restore(record)
    from resectionlab.surgical_rl_rewards import RewardTracker
    with pytest.raises(ValueError,match='PROCEDURE_POLICY_INCOMPATIBLE'):
        RewardTracker(e.engine)


def test_full_source_observations_expose_same_declared_anatomy_and_cost_limits():
    e=env();observed,info=e.reset(seed=9)
    assert e.observation_space.contains(observed)
    np.testing.assert_array_equal(observed['source_labels'],e.scene.labels)
    np.testing.assert_array_equal(observed['source_affine'],e.scene.affine)
    assert observed['source_memberships']['structure:cerebellum'].sum()==1
    assert observed['source_memberships']['observed'].all()
    assert observed['source_memberships']['original_target'].sum()==2
    neural=info['procedure_cost_names'].index('neuralRemovedMm3')
    assert observed['procedure_costs'][neural]==observed['procedure_limits'][neural]==0
    observed['source_labels'].fill(0)
    observed['source_memberships']['structure:cerebellum'].fill(0)
    assert e._observe()['source_labels'].any()
    assert e._observe()['source_memberships']['structure:cerebellum'].sum()==1


def test_supported_observation_gym_contract():
    from gymnasium.utils.env_checker import check_env
    check_env(env(),skip_render_check=True)


def test_unknown_source_grid_cell_remains_unknown_in_local_observation():
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_procedure_policy import procedure_source_memberships
    original=scene();raw=copy.deepcopy(original.properties[PROPERTY])
    raw['observedCells']=sorted(procedure_source_memberships(original)['observed']-{0})
    raw['policyHash']=semantic_digest({k:v for k,v in raw.items() if k!='policyHash'})
    unknown=SurgicalScene(original.labels,original.affine,original.regions,original.source,{PROPERTY:raw})
    observed,_=env(unknown).reset()
    assert observed['source_memberships']['observed'][0,0,0]==0
    assert observed['local'][0,7,4,4,4]==0


def test_cancelled_preview_charges_computation_without_patient_mutation_and_replays():
    e=env();prepare(e);before=e.engine.state_id;metrics=e.measurements.copy();calls=0
    def cancel():
        nonlocal calls
        calls+=1
        if calls>=6:
            raise InterruptedError('bounded test cancellation')
    with pytest.raises(InterruptedError):
        e.step_action(command('apply',[40.,40.,100.],duration=1.),check=cancel)
    assert e.engine.state_id==before and e.measurements==metrics
    assert e.computation=={'transitionEvaluations':3,'discardedEvaluations':1,'interruptedEvaluations':1}
    restored=env();restored.restore(json.loads(json.dumps(e.snapshot(),allow_nan=False)))
    assert restored.engine.state_id==before and restored.computation==e.computation


def test_shallow_withdraw_refuses_before_any_exit_flag_can_enable_stop():
    e=env();prepare(e,x=55.)
    assert e.step_action(command('apply',[55.,40.,100.],duration=1.))[4]['outcome']['status']=='accepted'
    before=e.engine.snapshot()
    assert e.step_action(command('withdraw',[55.,40.,99.]))[4]['outcome']['reasonCode']=='PROCEDURE_EXTERIOR_WITHDRAWAL_REQUIRED'
    assert e.engine.snapshot()==before
    result=e.stop('The tool has not exited')
    assert result[4]['outcome']['reasonCode']=='PROCEDURE_STOP_REQUIRES_CHECKED_EXIT'
    assert not result[2]


def test_zero_removal_scalpel_cut_is_a_precommit_rl_constraint_failure():
    from test_surgical_procedure_policy import source
    e=env(source(cutting=True));e.reset()
    for action in [command('exchange',instrument='scalpel'),command('place',[40.,42.5,65.])]:
        assert e.step_action(action)[4]['outcome']['status']=='accepted'
    before=e.engine.state_id;measured=e.measurements.copy()
    result=e.step_action(command('apply',[40.,42.5,90.],duration=1.))
    assert result[4]['outcome']['status']=='rejected' and result[4]['constraint_failure']
    assert any(row['metric']=='neuralCutSourceMm3' for row in result[4]['constraint_violations'])
    assert e.engine.state_id==before and e.measurements==measured and result[1]<=0
    cut=result[4]['procedure_cost_names'].index('neuralCutSourceMm3')
    assert result[0]['procedure_costs'][cut]==result[0]['procedure_limits'][cut]==0.


def test_relaxed_shared_limit_changes_feasibility_and_matches_native_sequence_outcome():
    from resectionlab.surgical_engine import SurgicalEngine
    source=scene(neural_limit=125.)
    e=env(source);prepare(e)
    native=SurgicalEngine(source,protection=e.protection,guard_config=e.engine.guard_config)
    for action in e.engine.actions:
        assert native.step(action)['status']=='accepted'
    action=command('apply',[40.,40.,100.],duration=1.)
    result=e.step_action(action)
    direct=native.step(action)
    assert result[4]['outcome']['status']==direct['status']=='accepted'
    assert result[1]>0 and not result[4]['constraint_failure']
    assert result[4]['procedure']==evaluate_procedure_sequence(native)
    assert e.engine.state_id==native.state_id
    assert result[4]['procedure']['outcome']['dimensions']['neuralRemovedMm3']['cumulative']==pytest.approx(125.)
    assert not result[4]['success'] and not result[4]['completion_assessment']['completedTask']


def test_tampered_policy_evaluation_or_computation_snapshot_is_refused_atomically():
    e=env();prepare(e);e.step_action(command('apply',[40.,40.,100.],duration=1.))
    snapshot=e.snapshot();restored=env();restored.reset();before=restored.engine.state_id
    for field in ('computation','procedureEvaluation','procedureProgress'):
        tampered=copy.deepcopy(snapshot);tampered[field]={}
        with pytest.raises(ValueError,match='procedure outcome or computation'):
            restored.restore(tampered)
        assert restored.engine.state_id==before


def test_patient_loader_cannot_reset_existing_treatment_prefix_costs(monkeypatch):
    from resectionlab import imaging,surgical_environment
    original=scene()
    monkeypatch.setattr(surgical_environment,'load_environment',lambda _:original)
    artifacts={'surgical_toolbox':{'branchId':'treated','branches':{
        'treated':{'actions':[command('inspect')]}}}}
    monkeypatch.setattr(imaging,'read_case_artifacts',lambda _:artifacts)
    with pytest.raises(ValueError,match='PROCEDURE_RL_CHECKPOINT_PREFIX_REQUIRED'):
        SurgicalRLEnv.from_patient('not-loaded.ressectionlab')


def test_shared_progress_uses_full_original_target_with_protected_overlap():
    from resectionlab.core import CaseData, SourceRef
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_scene import from_case
    from resectionlab.surgical_joint_cranial_search import assess_original_target_progress
    shape=(32,32,32)
    target=np.zeros(shape,bool);target[12,12,6]=target[12,12,12]=True
    exclusion=np.zeros(shape,bool);exclusion[12,12,12]=True
    case=CaseData(case_id='GENERATED-shared-progress-overlap',mri=np.zeros(shape,np.float32),
        compartments={'target':target,'exclusion':exclusion},affine=np.eye(4),
        source_refs=(SourceRef('generated','synthetic://shared-progress-overlap',provenance='simulated'),),
        metadata={'anatomy_layers':{'target':{'kind':'tumor'},'exclusion':{'kind':'brain','protected':True}},
            'surgical_adapter':{'ownershipPrecedence':['target','exclusion']}})
    native=SurgicalEngine(from_case(case))
    for action in [command('place',[12.,12.,-5.]),command('apply',[12.,12.,6.])]:
        action['slot']='right'
        assert native.step(action)['status']=='accepted'
    progress=assess_original_target_progress(native)
    assert progress['originalTargetCells']==2 and progress['creditedOriginalTargetCells']==1
    assert progress['originalTargetTreatmentFraction']==.5
    assert progress['residualOriginalTargetCells']==1
    assert not progress['completedProgressRequirement']


def test_canonical_stage_source_binds_same_engine_state_and_persists_in_rl():
    from test_structure_aware_cranial_planning import procedure_case
    from resectionlab.surgical_cranial_planning_roots import FIELD,candidate_scene
    from resectionlab.surgical_engine import SurgicalEngine
    _,base,record=procedure_case()
    root=record[FIELD]['roots'][0]
    source=candidate_scene(base,root['openingProperty'],root['openingContract'])
    native=SurgicalEngine.restore(source,root['sourceSnapshot'])
    e=SurgicalRLEnv(source,protection=native.protection,
        config=RLConfig(require_critical_coverage=native.guard_config.require_critical_coverage,
            critical_max_displacement_mm=native.guard_config.critical_max_displacement_mm),
        cranial_stages=native._cranial_stages)
    e.reset(seed=5)
    assert e.engine.state_id==native.state_id
    for action in [command('configure',mechanicsRevision=2,clockRevision=1),command('inspect')]:
        assert e.step_action(action)[4]['outcome']['status']==native.step(action)['status']=='accepted'
    assert e.engine.state_id==native.state_id
    assert e.procedure_evaluation==evaluate_procedure_sequence(native)
    restored=SurgicalRLEnv(source,protection=native.protection,config=e.config,
        cranial_stages=e._cranial_stage_record)
    restored.restore(json.loads(json.dumps(e.snapshot(),allow_nan=False)))
    assert restored.engine.state_id==e.engine.state_id
    assert e.branch(0)._cranial_stage_record==e._cranial_stage_record
