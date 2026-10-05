"""Read existing artifacts and source cells; never instantiate/train a simulator."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'artifacts/learning/procedural-native-to-ucsf-v2'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(RUN/'frozen-source/src'))
from resectionlab.imaging import load_case
from resectionlab.worlds import content_hash
from resectionlab.learning import MaskedPatientPolicy

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-8),(a,b)
def tensor_hash(state,actor=False):
    digest=hashlib.sha256()
    pairs=((k.removeprefix('actor.'),v) for k,v in state.items() if k.startswith('actor.')) if actor else state.items()
    for name,value in sorted(pairs):
        array=value.detach().cpu().contiguous().numpy()
        for part in (name.encode(),str(array.dtype).encode(),str(array.shape).encode(),array.tobytes()):digest.update(part)
    return 'sha256:'+digest.hexdigest()

def load(path):return torch.load(path,map_location='cpu',weights_only=True)

summary=read(RUN/'summary.json'); status=read(RUN/'experiment-status.json')
assert summary['status']==status['status']=='completed'
source=read(RUN/'launch-source.json');worker=read(RUN/'worker-source.json')
assert worker['git_revision'].startswith('68e4fde')
assert source['numerical_runtime_content_hash']==worker['numerical_runtime_content_hash']==summary['source_hash']
for name,digest in source['file_sha256'].items():assert sha(RUN/'frozen-source'/name)==digest,name
for name,digest in source['numerical_runtime_sha256'].items():
    assert sha(RUN/'frozen-source'/name)==digest,name
    committed=subprocess.check_output(['git','show',worker['git_revision']+':'+name],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==digest,name
assert content_hash(source['numerical_runtime_sha256'])==summary['source_hash']
declaration_path='manifests/experiments/procedural-native-to-ucsf-v1.json'
declaration_bytes=(RUN/'frozen-source'/declaration_path).read_bytes()
assert declaration_bytes==subprocess.check_output(['git','show','f26a72c:'+declaration_path],cwd=ROOT)
declaration=json.loads(declaration_bytes)
assert content_hash({k:v for k,v in declaration.items() if k!='declaration_content_hash'})==summary['declaration_hash']==declaration['declaration_content_hash']
assert sha(RUN/'source-case.ressectionlab')==declaration['target']['bundle_sha256']
case=load_case(RUN/'source-case.ressectionlab')
assert case.semantic_hash==declaration['target']['semantic_hash']
assert case.planning_hash==declaration['target']['planning_hash']
labels=np.zeros(case.mri.shape,np.int16)
for label,name in enumerate(sorted(case.compartments),1):
    assert not np.any((labels>0)&case.compartments[name])
    labels[case.compartments[name]]=label
volume=abs(float(np.linalg.det(case.affine[:3,:3])))
comparison=RUN/'comparison'
manifest=read(comparison/'manifest.json');freeze=read(comparison/'candidate-freeze.json')
assert manifest['world_partitions']==declaration['target']['world_partitions']
assert manifest['declaration_hash']==summary['declaration_hash']
assert freeze['optimization_partition_hash']==content_hash(manifest['world_partitions']['optimization'])
assert freeze['selection_partition_hash']==content_hash(manifest['world_partitions']['selection'])
assert set(manifest['world_partitions']['optimization']['seeds']).isdisjoint(manifest['world_partitions']['selection']['seeds'])
cstatus=read(comparison/'status.json')
audit_dir=comparison/cstatus['geometry_validation_run_id']
audits=read(audit_dir/'native-history-audit.json');replays=read(audit_dir/'native-history-replay.json')
candidates={c['plan_id']:c for c in freeze['candidates']}
expected={'STOP','GREEDY','SEARCH','PROCEDURAL_PRETRAINED_FROZEN',*(f'{mode}:{seed}' for mode in ('INITIAL','PATIENT_SCRATCH_RL','PROCEDURAL_PRETRAINED_ADAPTED') for seed in (11,23,47))}
assert set(candidates)==set(audits)==set(replays)==expected
assert len(candidates)==summary['validation']['candidate_count']==13
assert len({a['audit_key'] for a in audits.values()})==summary['validation']['unique_audits']==8
assert all(a['feasible'] and a['complete_tool_checked'] and a['frontier_checked'] and a['unsupported_source_tissue_volume_mm3']==0 for a in audits.values())
weights=declaration['target']['reward']; partial_weight=declaration['target']['partial_contact_weight']
recomputed={}
for name,record in replays.items():
    m=record['metrics']; removed=set(); partial=set(); reward=0.; prior_tool=None
    assert m['seed'] in manifest['world_partitions']['selection']['seeds']
    assert m['decision_model_hash']==declaration['target']['decision_model_hash']
    assert m['native_engine_config_hash']==declaration['target']['native_config_hash']
    assert m['functional_evidence_available']=={'motor':False,'language':False}
    assert m['motor_surrogate'] is m['language_surrogate'] is m['clinical_deficit_probability'] is None
    for h in m['history']:
        cells={tuple(c) for c in h['removed_indices_native']}; touched={tuple(c) for c in h['contact_indices_native']}
        assert len(cells)==len(h['removed_indices_native']) and not cells&removed
        assert cells<=touched
        newpartial=touched-removed-cells-partial
        target=sum(labels[c]>0 for c in cells)*volume
        normal=sum(labels[c]==0 for c in cells)*volume
        partial_normal=sum(labels[c]==0 for c in newpartial)*volume
        close(target,h['target_removed_mm3']);close(normal,h['normal_removed_mm3']);close(partial_normal,h['partial_normal_contact_mm3'])
        assert h['motor_surrogate_delta']==h['language_surrogate_delta']==h['partial_motor_contact_surrogate']==h['partial_language_contact_surrogate']==0
        distance=float(np.linalg.norm(np.array(h['tip_mm'])-h['entry_mm']))
        reward+=weights['target_per_mm3']*target-weights['normal_per_mm3']*normal-partial_weight*weights['normal_per_mm3']*partial_normal-weights['action_cost']-2*weights['motion_per_mm']*distance-weights['tool_change_cost']*(prior_tool is not None and prior_tool!=h['tool_id'])
        removed|=cells;partial|=newpartial;prior_tool=h['tool_id']
    target=sum(labels[c]>0 for c in removed)*volume;normal=sum(labels[c]==0 for c in removed)*volume
    close(reward,m['total_reward']);close(target,m['simulated_removed_target_volume_mm3']);close(normal,m['simulated_removed_normal_volume_mm3'])
    close(sum(labels[c]==0 for c in partial)*volume,m['cumulative_partial_normal_contact_mm3'])
    close(np.count_nonzero(labels)*volume-target,m['modeled_residual_target_volume_mm3'])
    close((len(removed))*volume,audits[name]['claimed_source_tissue_volume_mm3'])
    recomputed[name]={'score':reward,'target_removed_mm3':target,'normal_removed_mm3':normal,'cumulative_partial_normal_contact_mm3':sum(labels[c]==0 for c in partial)*volume,'retained_partial_normal_contact_mm3':sum(labels[c]==0 for c in partial-removed)*volume,'partial_normal_later_removed_mm3':sum(labels[c]==0 for c in partial&removed)*volume,'nonstop_actions':len(m['history'])}
shared=load(RUN/'pretraining/procedural.pt'); shared_hash=tensor_hash(shared['policy'])
assert shared_hash==shared['policy_hash']==summary['shared']['policy_hash']
assert sha(RUN/'pretraining/procedural.pt')==sha(comparison/'procedural-source.pt')==summary['shared']['checkpoint_file_sha256']
offline_result=read(RUN/'pretraining/training/result.json');offline_checkpoint=load(RUN/'pretraining/training/checkpoint.pt')
assert tensor_hash(offline_checkpoint['policy'])==shared_hash==offline_result['latest_checkpoint_hash']
assert offline_result['gradient_steps']==32 and offline_result['optimization_environment_steps']<=256
assert offline_result['actor_parameters_changed'] and offline_result['latest_actor_hash']!=offline_result['initial_actor_hash']
assert tensor_hash(offline_checkpoint['policy'],True)==offline_result['latest_actor_hash']
assert tensor_hash(load(RUN/'pretraining/training/initial.pt')['policy'],True)==offline_result['initial_actor_hash']
assert all(int(state['step'])==32 for state in offline_checkpoint['optimizer']['state'].values())
assert offline_result['elapsed_seconds']<=60
family_sources={r['source_hash'] for r in declaration['procedural_training']['members']}
exposures={}
for update in offline_result['optimization_history']:
    for name in update['episode_source_case_hashes']:exposures[name]=exposures.get(name,0)+1
assert set(exposures)==family_sources and case.semantic_hash not in exposures
assert exposures==shared['procedural_provenance']['gradient_episode_sources']
assert shared['procedural_provenance']['checkpoint_rule']=='fixed_budget_latest'
assert shared['procedural_provenance']['human_patients_in_pretraining']==0
rows=[]
for row in summary['learning']:
    seed=row['seed']; adapted=row['optimizer_mode']=='PROCEDURAL_PRETRAINED_ADAPTED'
    folder=comparison/f'{"adapted" if adapted else "scratch"}-{seed}'
    detail=read(folder/'result.json');contract=read(folder/'contract.json')
    latest=load(folder/'checkpoint.pt');initial=load(folder/'initial.pt')
    expected_config={**declaration['budgets']['online_scratch_and_adapted_each_seed'],'seed':seed}
    assert contract['config']==expected_config
    assert contract['timing_contract']=='optimization_selection_budget_v2_initialization_separate'
    assert contract['decision_model_hash']==declaration['target']['decision_model_hash']
    assert contract['partitions']=={role:{**{k:v for k,v in manifest['world_partitions'][role].items() if k!='interpretation'},'partition_hash':content_hash(manifest['world_partitions'][role])} for role in ('optimization','selection')}
    assert contract['numerical_source_sha256']==shared['procedural_provenance']['source_sha256']
    assert tensor_hash(initial['policy'])==row['initial_checkpoint_hash']==detail['initial_checkpoint_hash']
    assert tensor_hash(latest['policy'])==row['latest_checkpoint_hash']==detail['latest_checkpoint_hash']
    assert tensor_hash(latest['selected_policy'])==row['selected_checkpoint_hash']==detail['selected_checkpoint_hash']
    assert tensor_hash(initial['policy'],True)==row['initial_actor_hash']
    assert tensor_hash(latest['policy'],True)==row['latest_actor_hash']
    assert row['actor_parameters_changed']==(row['initial_actor_hash']!=row['latest_actor_hash'])
    assert row['selected_is_initial']==(row['initial_checkpoint_hash']==row['selected_checkpoint_hash'])
    if adapted:
        assert row['initial_checkpoint_hash']==shared_hash
        assert sha(folder/'procedural-source.pt')==summary['shared']['checkpoint_file_sha256']
    else:
        assert contract['procedural_initialization'] is None and row['shared_checkpoint_hash'] is None
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            seeded=MaskedPatientPolicy(*initial['dimensions'])
        assert tensor_hash(seeded.state_dict())==row['initial_checkpoint_hash']
    history=detail['selection_history']
    assert history and all(x['world_count']==2 for x in history)
    best=max(range(len(history)),key=lambda i:history[i]['mean_return'])
    assert history[best]['checkpoint_hash']==row['selected_checkpoint_hash']
    close(history[best]['mean_return'],row['selected_selection_return'])
    label=f'{row["optimizer_mode"]}:{seed}'
    close(row['selected_selection_return'],recomputed[label]['score'])
    if not adapted:close(row['initial_selection_return'],recomputed[f'INITIAL:{seed}']['score'])
    else:close(row['initial_selection_return'],recomputed['PROCEDURAL_PRETRAINED_FROZEN']['score'])
    assert 0<row['gradient_steps']<=32 and row['optimization_environment_steps']<=256
    assert len(detail['optimization_history'])==row['gradient_steps']
    assert all(int(state['step'])==row['gradient_steps'] for state in latest['optimizer']['state'].values())
    assert row['status']=='wall_time_budget'
    close(row['learner_wall_budget_overshoot_seconds'],max(row['elapsed_seconds']-30,0))
    assert row['trainer_call_seconds']+1e-6>=row['elapsed_seconds']+row['initialization_seconds']
    assert 0<=detail['selection_seconds']<=row['elapsed_seconds']
    rows.append({'arm':row['optimizer_mode'],'seed':seed,'updates':row['gradient_steps'],'optimization_transitions':row['optimization_environment_steps'],'selection_transitions':row['selection_environment_steps'],'score':recomputed[label]['score'],'initial_score':row['initial_selection_return'],'selected_initial':row['selected_is_initial'],'actor_changed':row['actor_parameters_changed'],'cold_setup_seconds':row['preparation_seconds'],'initialization_seconds':row['initialization_seconds'],'algorithm_seconds':row['elapsed_seconds'],'selection_seconds_within_algorithm':detail['selection_seconds'],'trainer_call_seconds':row['trainer_call_seconds'],'extraction_seconds':row['candidate_extraction_seconds'],'atomic_overshoot_seconds':row['learner_wall_budget_overshoot_seconds']})
assert summary['frozen']['gradient_steps']==summary['frozen']['optimization_environment_steps']==0
assert summary['frozen']['shared_checkpoint_hash']==shared_hash
assert summary['frozen']['selection_panel_complete']
close(summary['frozen']['selection_return'],recomputed['PROCEDURAL_PRETRAINED_FROZEN']['score'])
for row in summary['search']:close(row['nominal_score'],recomputed[row['method']]['score'])
assert summary['final_worlds_used'] is status['final_worlds_used'] is False
assert not list(RUN.glob('**/final_evaluation.json')) and not list(RUN.glob('**/stress.json'))
assert not list(RUN.glob('**/ledger.json'))
assert len(cstatus['planned_learning_runs'])==len(cstatus['completed_learning_runs'])==6
assert not summary['validation']['rejected_candidate_ids']
known_phases=sum(summary['preparation'][k] for k in ('target_preparation_seconds','procedural_fixture_construction_seconds','preflight_seconds'))+summary['pretraining_call_seconds']+summary['shared']['initialization_seconds']+sum(r['preparation_seconds']+r['elapsed_seconds'] for r in summary['search'])+summary['frozen']['preparation_seconds']+summary['frozen']['elapsed_seconds']+sum(r['preparation_seconds']+r['trainer_call_seconds']+r['candidate_extraction_seconds'] for r in summary['learning'])+summary['validation']['full_validation_seconds']
assert known_phases<=summary['total_worker_seconds']
means={arm:statistics.mean(r['score'] for r in rows if r['arm']==arm) for arm in ('PATIENT_SCRATCH_RL','PROCEDURAL_PRETRAINED_ADAPTED')}
report={'status':'passed_artifact_consistency_audit','training_or_simulator_execution_performed':False,'source_commit':worker['git_revision'],'launch_git_revision':source['git_revision'],'declaration_unchanged_from_commit':'f26a72c','verified_snapshot_files':len(source['file_sha256']),'source_hash':summary['source_hash'],'verified_candidate_count':13,'unique_independent_native_audits':8,'recomputed_from_native_source_cells':recomputed,'learning_rows':rows,'selected_score_mean_by_arm':means,'adapted_minus_scratch_mean':means['PROCEDURAL_PRETRAINED_ADAPTED']-means['PATIENT_SCRATCH_RL'],'offline':{'updates':offline_result['gradient_steps'],'optimization_transitions':offline_result['optimization_environment_steps'],'selection_transitions':offline_result['selection_environment_steps'],'algorithm_seconds':offline_result['elapsed_seconds'],'initialization_seconds':offline_result['initialization_seconds'],'full_offline_call_seconds':summary['pretraining_call_seconds'],'procedural_family_gradient_episodes':exposures},'timing':{'total_worker_seconds':summary['total_worker_seconds'],'total_including_source_freeze_seconds':status['total_seconds_including_source_freeze'],'sum_explicit_nondoublecounted_phases_seconds':known_phases,'other_worker_orchestration_seconds':summary['total_worker_seconds']-known_phases,'shared_validation_copy_load_seconds':summary['shared']['initialization_seconds'],'independent_native_validation_seconds':summary['validation']['full_validation_seconds'],'max_recorded_process_peak_rss_gib':max(r['execution_context_after']['process_peak_rss_bytes'] for r in summary['learning'])/1024**3},'world_exposure_check':'Saved contracts and replay seeds use optimization/selection only; no final/stress result or ledger, and frozen runner has no final evaluation path','limitations':['Artifact/source consistency audit; did not rerun the geometric collision checker or train models','One previously studied structural development case; zero human patients in pretraining','Three seeds and one offline seed do not establish a population or clinical result','Missing motor/language evidence remains null; no approved anatomical support or clinical safety inference','Sequential arm order and uncontrolled concurrent load limit runtime comparisons','Previous failed v1 remains a separate failed attempt and must not be hidden']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True))
print(json.dumps({k:report[k] for k in ('status','verified_snapshot_files','verified_candidate_count','selected_score_mean_by_arm','adapted_minus_scratch_mean','offline','timing')},indent=2))
