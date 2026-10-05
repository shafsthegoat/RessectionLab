"""Summarize a completed development follow-on without modifying original runs."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
current=root/'comparison'
original=root.parent/'native-ucsf0004-v1/comparison'
def read(path):return json.loads(path.read_text())
status=read(current/'status.json')
assert status['status']=='completed',status
manifest=read(current/'manifest.json')
old_manifest=read(original/'manifest.json')
assert manifest['simulation_hash']==old_manifest['simulation_hash']
assert manifest['training_config']==old_manifest['training_config']
assert status['evaluation_status']=='not_requested_development_selection_only'
source=read(root/'source-file-sha256.json')
assert all(hashlib.sha256((root/'frozen-workspace'/name).read_bytes()).hexdigest()==digest for name,digest in source.items())
training=read(current/'training.json')
old_training=read(original/'training.json')
rows=[]
for seed,run,before in zip((11,23,47),training,old_training):
    result=read(current/f'scratch-{seed}/result.json')
    old=read(original/f'scratch-{seed}/result.json')
    assert result['initial_checkpoint_hash']==old['initial_checkpoint_hash']
    rows.append({'seed':seed,'status':result['status'],'gradient_steps':result['gradient_steps'],
        'optimization_environment_steps':result['optimization_environment_steps'],
        'selection_environment_steps':result['selection_environment_steps'],
        'initial_selection_return':result['initial_selection_return'],
        'selected_selection_return':result['selected_selection_return'],
        'actor_parameters_changed':result['actor_parameters_changed'],
        'selected_initial_checkpoint':result['selected_checkpoint_hash']==result['initial_checkpoint_hash'],
        'preparation_seconds':run['preparation_seconds'],'bounded_training_seconds':result['elapsed_seconds'],
        'selection_seconds_within_training':result['selection_seconds'],
        'training_excluding_selection_seconds':result['elapsed_seconds']-result['selection_seconds'],
        'candidate_extraction_seconds':run['candidate_extraction_seconds'],
        'original_gradient_steps':old['gradient_steps'],
        'original_optimization_environment_steps':old['optimization_environment_steps'],
        'original_selection_environment_steps':old['selection_environment_steps'],
        'original_selected_selection_return':old['selected_selection_return'],
        'selected_score_delta':result['selected_selection_return']-old['selected_selection_return'],
        'initial_checkpoint_hash':result['initial_checkpoint_hash'],
        'selected_checkpoint_hash':result['selected_checkpoint_hash'],'latest_checkpoint_hash':result['latest_checkpoint_hash']})
geometry=current/status['geometry_validation_run_id']
audit=read(geometry/'native-history-audit.json')
summary={'study_role':'development_only_native_patient_cache_follow_on','current_head':read(root/'design.json')['current_head'],
    'source_study':'artifacts/learning/native-ucsf0004-v1','only_runtime_change':'bounded initial-geometry cache in native_simulation.py',
    'source_hash':read(root/'design.json')['source_content_hash'],'simulation_hash':manifest['simulation_hash'],
    'unchanged_simulation_and_training_config':True,'unchanged_initial_policy_checkpoints':True,
    'rows':rows,'search':read(current/'search.json'),'preprocessing_seconds':manifest['preprocessing_seconds'],
    'full_validation_seconds':status['full_validation_seconds'],'native_audit':audit,
    'final_evaluation_worlds_used':False,'clinical_deficit_probability':None,
    'functional_evidence_available':manifest['evidence_available'],
    'interpretation':'Report throughput and selection scores separately. Additional updates under a time budget do not imply a policy or clinical advantage; compare each retained checkpoint with SEARCH.',
    'limitations':['One development patient and three initialization seeds',
        'Historical runtime comparison with uncontrolled concurrent load, not a randomized timing trial',
        'Optimization and selection consume the cooperative 30-second arm budget; setup, extraction and independent geometry validation are additional costs',
        'Uncertainty calibration and postoperative function remain unvalidated; functional evidence is missing']}
(root/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
print(json.dumps({k:summary[k] for k in ('rows','search','preprocessing_seconds','full_validation_seconds','unchanged_simulation_and_training_config','final_evaluation_worlds_used')},indent=2))
