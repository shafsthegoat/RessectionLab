"""One fixed TRAIN ridge-proposal pilot. Metadata preflight opens no payloads."""
import hashlib,json,subprocess,types
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUTPUT=HERE/'attempt-01'
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045']
ARMS=('ridge_top8','sampled_target_top8','exhaustive')
STEPS=[1,14,1,13]
CAPS={'worker_seconds':600,'parent_seconds':660,'sampled_rss_bytes':3*1024**3,
      'output_bytes':64*1024**2,'supervision_bytes':8*1024**2,'log_bytes':2*1024**2,
      'native_previews':40000,'native_transitions':634,'source_visits':28,'threads':1,
      'fit_rows':3480,'fits':1,'feature_count':681,'ridge_calls':192}
CACHE_LIMITS={'max_payload_bytes':32*1024**2,'max_entries':16384}
OWNER={'path':'build/public-motion-ranking-il64-v1/pilot_contract.py','sha256':'130be4f52979eb40a0919492a806eb4b2a71e6f76b2ba1173065677492a1fa58'}
GATES={'all_four_complete':True,'no_quality_loss_vs_exhaustive':True,'numeric_tolerance':1e-7,
       'online_preview_ratio_max':.5,'source_through_plan_wall_ratio_max':.8,
       'heuristic_comparison':'quality improvement, or equal quality and wall ratio <=0.9',
       'meaning':'TRAIN go/no-go only; one ordered measurement, no transfer/clinical/sustained-speed claim'}
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
    return owner,prior,baseline
def template(head,index):
    owner,prior,baseline=metadata();old=prior['release']
    return {'version':'fixed-four-TRAIN-public-ridge-proposal-v1','execution_released':False,
        'expected_head':head,'source_index':index,'caps':CAPS,'subjects':SUBJECTS,'arms':list(ARMS),
        'public_index':old['public_index'],'cohort':old['cohort'],'occupancy_condition':old['occupancy_condition'],
        'post_exposure_condition':old['post_exposure_condition'],'proposal_config':old['proposal_config'],
        'fit':{'states':29,'teacher_steps':STEPS,'ridge_lambda':.001,'state_weights':'equal_then_uniform_within_state',
               'label':'max(0, nominal immediate native reward) if legal, else zero; not safety'},
        'K':8,'horizon':24,'search_seconds':60,'cache_limits':CACHE_LIMITS,
        'cache_scope':'fresh cache before every planning constructor; restored before authoritative replay',
        'gates':GATES,'no_optimizer_or_neural_model':True,'SELECT_EVAL_closed':True,
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
    need(result['status']=='complete_fixed_TRAIN_ridge_proposals' and result['release_sha256']==release_sha,'complete_pilot')
    need(result['source_visits']==28 and result['teacher_decisions']==29 and result['fits']==1,'exact_training_and_visits')
    need(result['native_transitions']<=634 and result['native_previews']<=40000 and result['fit_rows']<=3480,'bounded_work')
    need(result['model_calls']==result['optimizer_calls']==result['checkpoint_loads']==result['blocked_external_calls']==0,'zero_unauthorized_work')
    need(result['all_plans_sealed_before_evaluation'] and result['ridge_unchanged'] and result['cache_restored'],'information_and_integrity')
    need(len(result['arms'])==12 and all(row['complete'] and row['independent_accepted'] for row in result['arms']),'all_four_all_methods')
    need([(r['method'],r['subject']) for r in result['arms']]==[(m,s) for m in ARMS for s in SUBJECTS],'fixed_denominator')
    for row in result['arms']:
        for path,h in row['files'].items():need(sha(OUTPUT/path)==h,'arm_output_changed:'+path)
    need(sha(OUTPUT/'ridge.json')==result['ridge_file_sha256'],'frozen_ridge_file')
