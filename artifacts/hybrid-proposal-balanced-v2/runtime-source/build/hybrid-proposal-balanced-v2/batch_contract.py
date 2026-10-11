"""One fixed TRAIN ridge-proposal pilot. Metadata preflight opens no payloads."""
import hashlib,json,subprocess,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUTPUT=HERE/'attempt-01'
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
ARMS=('old_ridge_top8','balanced_ridge_top8','exhaustive')
STEPS=[1,14,1,13]
CAPS={'worker_seconds':600,'parent_seconds':660,'sampled_rss_bytes':3*1024**3,
      'output_bytes':64*1024**2,'supervision_bytes':8*1024**2,'log_bytes':2*1024**2,
      'native_previews':40000,'native_transitions':634,'source_visits':28,'threads':1,
      'fit_rows':3480,'fits':2,'feature_count':681,'ridge_calls':442}
# 384 worst-case planning/replay ridge scores + 58 teacher-state readouts.
CACHE_LIMITS={'max_payload_bytes':32*1024**2,'max_entries':16384}
OWNER={'path':'build/public-motion-ranking-il64-v1/pilot_contract.py','sha256':'130be4f52979eb40a0919492a806eb4b2a71e6f76b2ba1173065677492a1fa58'}
GATES={'all_four_complete':True,'numeric_tolerance':1e-7,
       'balanced_no_quality_loss_vs_exhaustive':True,'balanced_no_quality_loss_vs_old_ridge':True,
       'balanced_strict_return_improvement_vs_old_ridge':True,
       'online_preview_ratio_max':.5,'source_through_plan_wall_ratio_max':.8,
       'meaning':'prospective loss-contrast gate; not a retroactive pass of v1 or transfer/clinical claim'}
V1_REFS={'attempt-01/result.json': {'path': 'build/hybrid-proposal-pilot-v1/attempt-01/result.json', 'sha256': '77deca8508e6a69d78243870ef1dc362f4403e2e4480197052da10af196bad7c'}, 'attempt-01/training-labels.json': {'path': 'build/hybrid-proposal-pilot-v1/attempt-01/training-labels.json', 'sha256': '437c2620cceeb191ddf38e3cc932c9e6e7f918c8481110e5319d29b963b248de'}, 'attempt-01.supervision/receipt.json': {'path': 'build/hybrid-proposal-pilot-v1/attempt-01.supervision/receipt.json', 'sha256': '51d8877f66cb50086f98141c83e296c965edb20f9256700d714f9364d1463b99'}, 'root-release.json': {'path': 'build/hybrid-proposal-pilot-v1/root-release.json', 'sha256': '5a093996a88d9cddce08431280fb95fda351242d670d571255377eb054a798f3'}}
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(value):return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))
def small(path,digest=None):
    path=ROOT/path;need(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=8*1024**2,'bounded_metadata')
    raw=path.read_bytes()
    if digest is not None:need(hashlib.sha256(raw).hexdigest()==digest,'metadata_pin:'+str(path))
    return json.loads(raw)
def metadata():
    raw=(ROOT/OWNER['path']).read_bytes();need(hashlib.sha256(raw).hexdigest()==OWNER['sha256'],'teacher_owner_source')
    owner=types.ModuleType('fixed_saved_teacher_owner');owner.__file__=str(ROOT/OWNER['path'])
    exec(compile(raw,owner.__file__,'exec'),owner.__dict__)
    prior=owner.inputs()  # Existing exact all-four completed search/teacher checks.
    need(list(owner.TRAIN)==SUBJECTS and list(owner.TEACHER_STEPS.values())==STEPS,'fixed_teacher_denominator')
    baseline={k:owner.small(owner.BASELINE_REFS[k]) for k in ('result','parent','recovery_result','recovery_parent','outcomes')}
    need(baseline['result']['status']=='failed_or_unresolved','preserve_original_IL_failure')
    need(baseline['recovery_result']['independent_accepted'] and baseline['recovery_parent']['status']=='complete','separate_IL045_recovery')
    old_result=small(V1_REFS['attempt-01/result.json']['path'],V1_REFS['attempt-01/result.json']['sha256'])
    old_receipt=small(V1_REFS['attempt-01.supervision/receipt.json']['path'],V1_REFS['attempt-01.supervision/receipt.json']['sha256'])
    old_release=small(V1_REFS['root-release.json']['path'],V1_REFS['root-release.json']['sha256'])
    need(old_result['status']=='complete_fixed_TRAIN_ridge_proposals' and len(old_result['arms'])==12,'v1_complete')
    need(old_receipt['status']=='complete' and old_receipt['exit_code']==0 and not old_receipt['cleanup_errors'] and not old_receipt['remaining_owned_pids'],'v1_clean')
    need(old_receipt['result_sha256']==V1_REFS['attempt-01/result.json']['sha256'] and old_receipt['release_sha256']==V1_REFS['root-release.json']['sha256'],'v1_receipt_join')
    old_plans={}
    for row in old_result['arms']:
        if row['method']=='ridge_top8':
            path='ridge_top8/'+row['subject']+'/plan.json'
            old_plans[row['subject']]=small('build/hybrid-proposal-pilot-v1/attempt-01/'+path,row['files'][path])
    baseline['proposal_v1']={'result':old_result,'release':old_release,'plans':old_plans,
        'labels':small(V1_REFS['attempt-01/training-labels.json']['path'],V1_REFS['attempt-01/training-labels.json']['sha256'])}
    return owner,prior,baseline
def template(head,index):
    owner,prior,baseline=metadata();old=prior['release']
    return {'version':'fixed-four-TRAIN-balanced-ridge-contrast-v2','execution_released':False,
        'expected_head':head,'source_index':index,'caps':CAPS,'subjects':SUBJECTS,'arms':list(ARMS),
        'public_index':old['public_index'],'cohort':old['cohort'],'occupancy_condition':old['occupancy_condition'],
        'post_exposure_condition':old['post_exposure_condition'],'proposal_config':old['proposal_config'],
        'fit':{'states':29,'teacher_steps':STEPS,'ridge_lambda':.001,'state_weights':{'old':'equal_then_uniform_within_state','balanced':'equal_state_then_half_positive_half_zero_when_both_present'},'normalization':'original_equal_state_mean_scale_fixed','solves':2,
               'label':'max(0, nominal immediate native reward) if legal, else zero; not safety'},
        'K':8,'horizon':24,'search_seconds':60,'cache_limits':CACHE_LIMITS,
        'cache_scope':'fresh cache before every planning constructor; restored before authoritative replay',
        'gates':GATES,'historical_v1':{'refs':V1_REFS,'gates':baseline['proposal_v1']['release']['gates'],'comparison':baseline['proposal_v1']['result']['comparison']},'no_optimizer_or_neural_model':True,'SELECT_EVAL_closed':True,
        'source_world_hashes':{row['patient_id']:row['source_hash'] for row in prior['result']['cases']}}
def guard(path,digest,*,execute):
    release=small(path,digest);need(type(release.get('execution_released')) is bool,'typed_release')
    if execute:need(release['execution_released'],'root_release_required')
    need({**release,'execution_released':False}==template(release['expected_head'],release['source_index']),'exact_fixed_pilot_release')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();need(head==release['expected_head'],'HEAD_changed')
    index=small(release['source_index']['path'],release['source_index']['sha256']);need(index['head']==head,'index_HEAD')
    required={str((HERE/n).relative_to(ROOT)) for n in ('batch_contract.py','hybrid_proposal.py','patient_sparse.py','initial_inventory_worker.py','run_owned.py','freeze-runtime.py')}
    required|={'src/resectionlab/patient_planning_admission.py','src/resectionlab/experimental_capsule_cache.py','src/resectionlab/native_spatial_evaluation.py'}
    need(required<=set(index['files']),'full_runtime_closure')
    for p,h in index['files'].items():need(sha(ROOT/p)==h,'source_or_input_changed:'+p)
    return release,index,True
def complete_result(result,release,release_sha):
    need(result['status']=='complete_fixed_TRAIN_balanced_ridge_contrast' and result['release_sha256']==release_sha,'complete_pilot')
    need(result['source_visits']==28 and result['teacher_decisions']==29 and result['fits']==2,'exact_training_and_visits')
    need(result['native_transitions']<=634 and result['native_previews']<=40000 and result['fit_rows']<=3480,'bounded_work')
    need(result['model_calls']==result['optimizer_calls']==result['checkpoint_loads']==result['blocked_external_calls']==0,'zero_unauthorized_work')
    need(result['all_plans_sealed_before_evaluation'] and result['ridge_unchanged'] and result['cache_restored'],'information_and_integrity')
    need(len(result['arms'])==12 and all(row['complete'] and row['independent_accepted'] for row in result['arms']),'all_four_all_methods')
    need([(r['method'],r['subject']) for r in result['arms']]==[(m,s) for m in ARMS for s in SUBJECTS],'fixed_denominator')
    for row in result['arms']:
        for path,h in row['files'].items():need(sha(OUTPUT/path)==h,'arm_output_changed:'+path)
    need(result['old_ridge_parameter_identity'] and result['teacher_labels_identity'] and result['ridge_calls']<=442,'exact_controls')
    for mode,h in result['ridge_file_sha256'].items():need(sha(OUTPUT/(mode+'.json'))==h,'frozen_ridge_file')
